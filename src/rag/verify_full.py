"""
Basira RAG — Unified Verification
=====================================
Verifies the authenticity and completeness of Arabic religious texts,
and optionally evaluates an accompanying explanation — all in a single
GPT-4o-mini call backed by FAISS semantic search.

Replaces: verify.py, detect_truncation.py
"""

from __future__ import annotations

import difflib
import json
import pickle
import re
import sys
from pathlib import Path

import faiss

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search
from src.nlp.preprocessor import preprocess, normalize
from src.rag.pipeline import ground_text

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

# Deterministic truncation-check thresholds
INPUT_MATCH_MIN = 0.8
MATN_COVERAGE_MAX = 0.7

# ---------------------------------------------------------------------------
# Matn cleaning helpers (Task A)
# ---------------------------------------------------------------------------

# Unicode format / bidirectional / quote characters that appear in hadith matn
# but carry no semantic content (U+200E, U+200F, ZWS, ZWNJ, ZWJ, BOM, etc.)
_FORMAT_CHARS_RE = re.compile(
    r'[\u200e\u200f\u200b\u200c\u200d\ufeff'
    r'\u202a-\u202e\u2066-\u2069\u061c'
    r'\u0640'            # Arabic tatweel
    r'"\u0022\'\u0027'
    r'\u00ab\u00bb'      # «»
    r'\u201c\u201d'      # “”
    r'\u2018\u2019'      # ‘’
    r']'
)

# Narration-framing patterns that may appear as a *leading* prefix in matn.
# Each entry is a list of normalised word tokens (longest first so we match
# the most specific pattern before falling back to shorter ones).
_FRAMING_PATTERNS: list[list[str]] = sorted(
    [
        normalize(p).split()
        for p in [
            "قال رسول الله صلى الله عليه وسلم قال",
            "قال رسول الله صلى الله عليه وسلم",
            "سمعت رسول الله صلى الله عليه وسلم يقول",
            "سمعت رسول الله صلى الله عليه وسلم",
            "ان رسول الله صلى الله عليه وسلم قال",
            "ان رسول الله صلى الله عليه وسلم",
            "عن رسول الله صلى الله عليه وسلم",
            "قال النبي صلى الله عليه وسلم",
            "ان النبي صلى الله عليه وسلم",
            "عن النبي صلى الله عليه وسلم",
            "رسول الله صلى الله عليه وسلم",
            "النبي صلى الله عليه وسلم",
            "صلى الله عليه وسلم",
            "يقول",
            "قال",
        ]
    ],
    key=len,
    reverse=True,  # longest first
)


def _strip_format_chars(text: str) -> str:
    """Remove format/bidi/quote characters that add no semantic content."""
    return _FORMAT_CHARS_RE.sub(' ', text)


def _clean_matn_words(matn_raw: str, strip_framing: bool = True) -> list[str]:
    """Return normalised word tokens for *matn_raw*.

    Steps:
    1. Remove U+200F/200E and other format chars.
    2. Normalize (strip tashkeel, unify hamza/alef, etc.).
    3. Optionally strip a *leading* narration-framing prefix (repeated
       until no more patterns match from the front).  Words in the
       *middle* of the matn are never removed.
    """
    cleaned = _strip_format_chars(matn_raw)
    words = normalize(cleaned).split()

    if strip_framing:
        changed = True
        while changed and words:
            changed = False
            for pat in _FRAMING_PATTERNS:
                n = len(pat)
                if n and words[:n] == pat:
                    words = words[n:]
                    changed = True
                    break

    return words

SYSTEM_PROMPT = (
    "أنت عالم إسلامي متخصص في التحقق من النصوص الدينية.\n"
    " ستُعطى نصاً دينياً وربما شرحاً له، بالإضافة إلى مصادر من قاعدة البيانات.\n\n"
    " مهمتك — أجب على كل النقاط التالية:\n"
    " 1. صحة النص: هل هو صحيح أم ضعيف أم موضوع أم غير موجود؟\n"
    " 2. اكتمال النص: هل النص مبتور أو منقوص مقارنةً بالمصدر؟\n"
    " 3. الجزء الناقص: ما هو الجزء الناقص إن وجد؟\n"
    " 4. النص الكامل: ما هو النص الكامل من المصدر؟\n"
    " 5. المصدر: اسم الكتاب ورقم الحديث أو الآية\n"
    " 6. درجة الخطورة: منخفضة أو متوسطة أو عالية (بناءً على مدى تغيير المعنى)\n"
    " إذا أُعطيت شرحاً:\n"
    " 7. تقييم الشرح: موافق أو مخالف أو جزئياً موافق\n"
    " 8. سبب الحكم: لماذا الشرح صحيح أو خاطئ؟\n"
    " 9. الشرح الصحيح: ما هو الشرح الصحيح من المصادر؟\n\n"
    " قواعد صارمة:\n"
    " - لا تجاوب من معرفتك الخاصة أبداً — فقط من المصادر المقدمة\n"
    " - إذا لم تجد النص في المصادر قل: غير موجود\n"
    " - إذا لم يُعطَ شرح، اجعل حقول الشرح فارغة\n\n"
    " أجب بالعربية فقط في شكل JSON بهذه المفاتيح بالضبط:\n"
    " {\n"
    "   'text_status': 'صحيح أو ضعيف أو موضوع أو غير موجود',\n"
    "   'is_truncated': 'نعم أو لا',\n"
    "   'missing_part': 'الجزء الناقص أو فارغ',\n"
    "   'full_text': 'النص الكامل من المصدر',\n"
    "   'source': 'المصدر ورقم الحديث',\n"
    "   'danger_level': 'منخفضة أو متوسطة أو عالية',\n"
    "   'explanation_status': 'موافق أو مخالف أو جزئياً موافق أو لا يوجد',\n"
    "   'explanation_verdict': 'سبب الحكم أو فارغ',\n"
    "   'correct_explanation': 'الشرح الصحيح من المصادر أو فارغ'\n"
    " }"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_key(d: dict, *keys: str, default: str = "") -> str:
    """Return the value of the first key found in *d*, else *default*.

    Useful when a GPT JSON response may use different key names
    (English vs. various Arabic phrasings).
    """
    for key in keys:
        if key in d:
            return d[key]
    return default


def _flatten_source(source_val) -> str:
    """Normalise a source value that may be a dict into a flat string.

    GPT sometimes returns the source as a nested object, e.g.
    ``{"اسم_الكتاب": "…", "رقم_الحديث": "…"}``.
    """
    if isinstance(source_val, dict):
        book = source_val.get("اسم_الكتاب", source_val.get("book", ""))
        number = source_val.get("رقم_الحديث", source_val.get("number", ""))
        return f"{book}:{number}" if number else book
    return str(source_val) if source_val else ""


def _is_yes(val) -> bool:
    """Interpret a GPT yes/no value as a Python bool.

    Handles Arabic (نعم/لا), English (yes/no/true/false), and native bools.
    """
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"نعم", "yes", "true", "1", "صحيح"}


# ---------------------------------------------------------------------------
# RAG-02 post-generation validation helpers
# ---------------------------------------------------------------------------

# Enum taxonomies exactly as requested by SYSTEM_PROMPT above; invalid model
# output falls back to the pre-existing safe defaults.
_TEXT_STATUS_VALUES = {"صحيح", "ضعيف", "موضوع", "غير موجود"}
_DANGER_LEVEL_VALUES = {"منخفضة", "متوسطة", "عالية"}
_EXPLANATION_STATUS_VALUES = {"موافق", "مخالف", "جزئياً موافق", "لا يوجد"}

# Arabic-Indic digits -> ASCII, so ١٢٣ and 123 compare equal.
_ARABIC_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _validated_enum(value, allowed: set[str], default: str) -> str:
    """Return *value* when it is one of the prompt's exact enum values."""
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default


def _normalize_reference(value) -> str:
    """Normalise a citation for comparison against retrieved references.

    Arabic-Indic digits become ASCII, format/bidi characters are stripped
    (via the existing matn cleaner), ':' and '،' become spaces, and all
    whitespace is collapsed.
    """
    s = _flatten_source(value)
    if not s:
        return ""
    s = s.translate(_ARABIC_DIGIT_TABLE)
    s = _strip_format_chars(s)
    s = s.replace(":", " ").replace("،", " ")
    return " ".join(s.split())


def _reference_is_retrieved(candidate: str, results: list[dict]) -> bool:
    """True when *candidate* matches a reference actually retrieved.

    Exact normalised equality or bidirectional containment against ONE
    retrieved reference.  A single generic word with no digit can never
    validate, so a lone book-name fragment cannot carry a fabricated
    hadith number past the check.
    """
    c = _normalize_reference(candidate)
    if not c:
        return False
    if not any(ch.isdigit() for ch in c) and len(c.split()) < 2:
        return False
    for r in results:
        ref = _normalize_reference(r.get("reference", ""))
        if not ref:
            continue
        if c == ref or c in ref or ref in c:
            return True
    return False


def _first_retrieved_reference(results: list[dict]) -> str:
    """Deterministic fallback source: first non-empty retrieved reference."""
    for r in results:
        ref = _flatten_source(r.get("reference", ""))
        if ref.strip():
            return ref.strip()
    return ""


# ---------------------------------------------------------------------------
# FAISS loader
# ---------------------------------------------------------------------------

def load_faiss_index(
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a FAISS index and its associated chunks from disk.

    Args:
        index_path:  Path to the FAISS index file.
        chunks_path: Path to the pickled chunks file.

    Returns:
        A (faiss.IndexFlatIP, chunks) tuple.
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[verify_full] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[verify_full] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


# ---------------------------------------------------------------------------
# Dual-embedding search
# ---------------------------------------------------------------------------

def _dual_search(text: str, client, index, chunks, top_k: int, debug: bool = False) -> list[dict]:
    """Embed *text* as both normalised and raw, return the best FAISS results."""
    processed = preprocess(text)
    normalized_text = processed["normalized"]

    emb_normalized = embed_text(normalized_text, client)
    emb_original = embed_text(text, client)

    results_norm = search(emb_normalized, index, chunks, top_k=top_k)
    results_orig = search(emb_original, index, chunks, top_k=top_k)

    # Pick whichever set has the higher top-1 score
    if results_orig and results_norm:
        results = (
            results_orig
            if results_orig[0]["score"] > results_norm[0]["score"]
            else results_norm
        )
    else:
        results = results_norm or results_orig

    if debug:
        print(f"\n[DEBUG] Top FAISS results for: {text[:60]}…")
        for i, r in enumerate(results[:3]):
            print(
                f"  {i + 1}. score={r['score']:.4f}  "
                f"ref={r.get('reference', 'N/A')}  "
                f"text={r.get('original_text', '')[:80]}…"
            )

    return results


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_full(
    text: str,
    explanation: str | None = None,
    index: faiss.IndexFlatIP | None = None,
    chunks: list[dict] | None = None,
    client=None,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Verify text authenticity + truncation + optional explanation in one call.

    Pipeline:
        1. Dual-embedding search on *text*.
        2. If *explanation* is provided, dual-embedding search on *explanation*
           and merge results (deduplicate by chunk id).
        3. Call GPT-4o-mini with the unified system prompt.
        4. Parse the JSON response with Arabic fallbacks.
        5. Return a dict with all 9 result keys plus input_text and
           input_explanation.

    Args:
        text:        The Arabic text to verify.
        explanation: Optional explanation/interpretation to evaluate.
        index:       A loaded FAISS index (loaded automatically if None).
        chunks:      The chunk dicts aligned with the index.
        client:      An authenticated OpenAI client (created if None).
        top_k:       Number of FAISS results to retrieve per search.
        debug:       If True, print intermediate results.

    Returns:
        Dict with keys: text_status, is_truncated, missing_part, full_text,
        source, danger_level, explanation_status, explanation_verdict,
        correct_explanation, input_text, input_explanation.
    """
    # --- Auto-load resources if not provided ---
    if client is None:
        client = get_client()
    if index is None or chunks is None:
        index, chunks = load_faiss_index()

    # --- Step 1: Wide dual-embedding search on text ---
    wide_k = top_k * 20
    wide_results = _dual_search(text, client, index, chunks, wide_k, debug=debug)

    # Take the top_k overall, then also ensure the top 3 hadith chunks
    # are included (they may be buried deeper in the ranking).
    results = wide_results[:top_k]
    seen_ids = {r.get("id") for r in results}

    hadith_extras = [
        r for r in wide_results
        if r.get("source") == "hadith" and r.get("id") not in seen_ids
    ][:3]
    for r in hadith_extras:
        results.append(r)
        seen_ids.add(r.get("id"))

    # --- Step 2: If explanation provided, search on it too & merge ---
    if explanation:
        expl_results = _dual_search(
            explanation, client, index, chunks, wide_k, debug=debug,
        )

        for r in expl_results:
            if r.get("id") not in seen_ids:
                results.append(r)
                seen_ids.add(r.get("id"))

    # --- Step 3: Build context ---
    sources_text = "\n\n".join(
        f"[مصدر {i + 1}] (تطابق: {r['score']:.2f})\n"
        f"النص: {r['original_text']}\n"
        f"المصدر: {r['source']}\n"
        f"المرجع: {r['reference']}"
        for i, r in enumerate(results)
    )

    # --- Step 3a: Deterministic truncation check (before GPT call) ---
    # Find the highest-scoring hadith chunk that has metadata["matn"]
    computed_is_truncated = None   # None = don't override
    computed_missing_part = None
    computed_full_text = None

    best_matn_chunk = None
    for r in results:
        if r.get("source") != "hadith":
            continue
        chunk_id = r.get("id")
        for c in chunks:
            if c.get("id") == chunk_id:
                meta = c.get("metadata", {})
                if isinstance(meta, dict) and meta.get("matn", "").strip():
                    best_matn_chunk = c
                    break
        if best_matn_chunk is not None:
            break

    if best_matn_chunk is not None:
        matn_raw = best_matn_chunk["metadata"]["matn"]

        # Clean input: strip format chars, then normalize
        input_clean = _strip_format_chars(text)
        input_norm_words = normalize(input_clean).split()

        # Clean matn: strip format chars + leading framing prefix, then normalize
        matn_norm_words = _clean_matn_words(matn_raw, strip_framing=True)

        if input_norm_words and matn_norm_words:
            sm = difflib.SequenceMatcher(
                None, input_norm_words, matn_norm_words,
            )
            matching_blocks = sm.get_matching_blocks()
            matched = sum(block.size for block in matching_blocks)

            input_match = matched / len(input_norm_words)
            matn_coverage = matched / len(matn_norm_words)

            if debug:
                print(
                    f"\n[DEBUG] Truncation check: "
                    f"input_match={input_match:.4f}, "
                    f"matn_coverage={matn_coverage:.4f}"
                )

            if input_match >= INPUT_MATCH_MIN:
                if matn_coverage < MATN_COVERAGE_MAX:
                    matched_matn_indices: set[int] = set()
                    for block in matching_blocks:
                        for offset in range(block.size):
                            matched_matn_indices.add(block.b + offset)
                    missing_words = [
                        w for idx, w in enumerate(matn_norm_words)
                        if idx not in matched_matn_indices
                    ]
                    # Fewer than 2 real words missing → not truncated
                    if len(missing_words) < 2:
                        computed_is_truncated = False
                    else:
                        computed_is_truncated = True
                        computed_missing_part = " ".join(missing_words)
                        computed_full_text = matn_raw
                else:
                    computed_is_truncated = False

    # --- Step 3b: Build user message ---
    user_parts = [f"النص المدخل: {text}"]
    if explanation:
        user_parts.append(f"الشرح المقدم: {explanation}")

    # Inject computed truncation facts so GPT's explanation stays consistent
    if computed_is_truncated is True:
        user_parts.append(
            f"ملاحظة: تحليل آلي يُظهر أن النص مبتور. "
            f"النص الكامل من المصدر: {computed_full_text}\n"
            f"الجزء الناقص: {computed_missing_part}"
        )
    elif computed_is_truncated is False:
        user_parts.append(
            "ملاحظة: تحليل آلي يُظهر أن النص مكتمل وغير مبتور."
        )

    user_parts.append(f"\nالمصادر المسترجعة:\n{sources_text}")
    user_message = "\n\n".join(user_parts)

    if debug:
        print(f"\n[DEBUG] User message length: {len(user_message)} chars")

    # --- Step 3c: Call GPT-4o-mini ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    # --- Step 4: Parse JSON response ---
    raw_answer = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        answer = {}

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    # --- Step 5: Build result dict with Arabic fallbacks ---
    gpt_is_truncated = get_key(
        answer,
        "is_truncated", "هل النص مبتور أو منقوص", "هل النص مبتور", "مبتور",
        "اكتمال النص", "اكتمال_النص",
        default="لا",
    )
    gpt_missing_part = get_key(
        answer,
        "missing_part", "الجزء الناقص", "الجزء_الناقص",
        "الجزء الناقص من النص",
    )
    gpt_full_text = get_key(
        answer,
        "full_text", "النص الكامل", "النص_الكامل",
        "النص الكامل من المصدر",
    )

    # Apply deterministic overrides when the computed check was decisive
    final_is_truncated = (
        computed_is_truncated
        if computed_is_truncated is not None
        else _is_yes(gpt_is_truncated)
    )
    final_missing_part = (
        computed_missing_part
        if computed_missing_part is not None
        else gpt_missing_part
    )
    final_full_text = (
        computed_full_text
        if computed_full_text is not None
        else gpt_full_text
    )

    # --- RAG-02: proportionate post-generation validation ---
    # Enums keep only values the prompt actually defines; a response with no
    # retrieved evidence can never attest authenticity.
    text_status = _validated_enum(
        get_key(
            answer,
            "text_status", "صحة_النص", "صحة النص", "الحالة", "حالة_النص",
            "status", "النتيجة",
            default="غير موجود",
        ),
        _TEXT_STATUS_VALUES,
        "غير موجود",
    )
    if not results:
        text_status = "غير موجود"

    # Citation: a non-empty GPT source is kept only when it matches a
    # reference that was actually retrieved; otherwise it is replaced with
    # the top retrieved reference (deterministic) or emptied.  An empty
    # model source is preserved as the safe empty representation.
    raw_source = _flatten_source(
        get_key(answer, "source", "المصدر", "مصدر", "المصدر_الموثوق")
    )
    if not raw_source.strip():
        source = ""
    elif _reference_is_retrieved(raw_source, results):
        source = raw_source
    else:
        source = _first_retrieved_reference(results)

    # Sourced prose is grounded against the retrieved evidence; rejection
    # empties the field.  explanation_verdict is deliberately left as the
    # model's evaluative reasoning about the provided explanation (prompt
    # taxonomy, not source content), so the citation-oriented matcher is not
    # applied to it.
    raw_correct_explanation = get_key(
        answer, "correct_explanation", "الشرح الصحيح", "الشرح_الصحيح",
        "الشرح الصحيح من المصادر",
    )
    if isinstance(raw_correct_explanation, str) and raw_correct_explanation.strip():
        correct_explanation = ground_text(
            raw_correct_explanation, results, chunks, refusal="",
        )
    else:
        correct_explanation = ""

    return {
        "text_status": text_status,
        "is_truncated": final_is_truncated,
        "missing_part": final_missing_part,
        "full_text": final_full_text,
        "source": source,
        "danger_level": _validated_enum(
            get_key(
                answer,
                "danger_level", "درجة الخطورة", "درجة_الخطورة", "الخطورة",
                default="منخفضة",
            ),
            _DANGER_LEVEL_VALUES,
            "منخفضة",
        ),
        "explanation_status": _validated_enum(
            get_key(
                answer,
                "explanation_status", "تقييم الشرح", "تقييم_الشرح",
                "حكم_الشرح",
                default="لا يوجد",
            ),
            _EXPLANATION_STATUS_VALUES,
            "لا يوجد",
        ),
        "explanation_verdict": get_key(
            answer,
            "explanation_verdict", "سبب الحكم", "سبب_الحكم",
        ),
        "correct_explanation": correct_explanation,
        "input_text": text,
        "input_explanation": explanation or "",
    }


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = get_client()
    index, chunks = load_faiss_index()

    test_cases = [
        # Test 1 — نص بدون شرح
        {
            "label": "Test 1 — نص بدون شرح",
            "text": "الدين النصيحة",
            "explanation": None,
        },
        # Test 2 — نص مع شرح خاطئ
        {
            "label": "Test 2 — نص مع شرح خاطئ",
            "text": "الدين النصيحة",
            "explanation": "يعني أن الإسلام مبني على المصلحة الشخصية فقط",
        },
        # Test 3 — نص مع شرح صحيح
        {
            "label": "Test 3 — نص مع شرح صحيح",
            "text": "إنما الأعمال بالنيات",
            "explanation": "يعني أن ثواب العمل يتوقف على النية",
        },
        # Test 4 — آية مع شرح
        {
            "label": "Test 4 — آية مع شرح",
            "text": "الحمد لله رب العالمين",
            "explanation": "رب العالمين تعني أن الله خالق كل شيء ومالكه ومدبره",
        },
    ]

    for tc in test_cases:
        print(f"\n{'=' * 70}")
        print(f"  {tc['label']}")
        print(f"{'=' * 70}")
        print(f"  النص:   {tc['text']}")
        if tc["explanation"]:
            print(f"  الشرح:  {tc['explanation']}")
        print("-" * 70)

        result = verify_full(
            text=tc["text"],
            explanation=tc["explanation"],
            index=index,
            chunks=chunks,
            client=client,
            debug=True,
        )

        print(f"\n  صحة النص:          {result['text_status']}")
        print(f"  مبتور:             {'نعم' if result['is_truncated'] else 'لا'}")
        print(f"  الجزء الناقص:      {result['missing_part']}")  # full, no truncation
        print(f"  النص الكامل:       {result['full_text']}")
        print(f"  المصدر:            {result['source']}")
        print(f"  درجة الخطورة:      {result['danger_level']}")
        print(f"  تقييم الشرح:       {result['explanation_status']}")
        print(f"  سبب الحكم:         {result['explanation_verdict']}")
        print(f"  الشرح الصحيح:      {result['correct_explanation']}")
        print(f"{'=' * 70}")
