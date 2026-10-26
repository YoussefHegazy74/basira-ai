"""
Basira RAG — Unified Verification
=====================================
Verifies the authenticity and completeness of Arabic religious texts,
and optionally evaluates an accompanying explanation.

The verification service is independent of the chatbot.  Quran quotations
are resolved lexically against the canonical Quran corpus *before* any
semantic retrieval or model call.  Semantic similarity is never treated as
proof that a quotation exists verbatim.  Hadith and other source types keep
the established retrieval + model validation path.

Result schema
-------------
The legacy 11 keys are preserved for existing consumers.  New fields make the
distinction between quotation existence, completeness, interpretation
assessment, evidence sufficiency, and service failure explicit:

    verification_status: verified | partial | contradicted | inconclusive |
                         error
    quotation:           found / match_type / is_complete / canonical_text /
                         reference / source / surah / ayah
    interpretation:      provided / status / label / assessment / correction
    evidence:            source-backed passages used for the result
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
from src.rag.tafsir import match_quran_quotation

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

# Deterministic truncation-check thresholds
INPUT_MATCH_MIN = 0.8
MATN_COVERAGE_MAX = 0.7

NOT_PROVIDED_LABEL = "لم يتم تقديم تفسير لتقييمه"
VERIFICATION_ERROR_LABEL = "تعذّر إكمال التحقق"
INCONCLUSIVE_LABEL = "لا توجد أدلة كافية"

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

INTERPRETATION_SYSTEM_PROMPT = (
    "أنت تقيّم تفسيراً قدمه المستخدم لنص قرآني موثق.\n"
    "لديك النص القرآني الكامل، وتفسير ابن كثير عند توفره في المصادر.\n"
    "قيّم التفسير المقدم فقط، ولا تصدر حكماً على صحة النص القرآني نفسه.\n\n"
    "قواعد صارمة:\n"
    "- لا تعتمد على معرفتك الخاصة، ولا تخترع آية أو حديثاً أو مرجعاً.\n"
    "- إذا كان التفسير يوافق المعنى الذي تدعمه المصادر فقل: موافق.\n"
    "- إذا كان يخالف المعنى أو يحمّله ما لا يحتمله فقل: مخالف، واذكر موضع الخطأ.\n"
    "- إذا كان بعضه صحيحاً وبعضه غير مدعوم فقل: جزئياً موافق.\n"
    "- إذا لم تكف المصادر للحكم فقل: غير مؤكد. لا تجعل عدم كفاية الدليل حكماً بالخطأ.\n"
    "- الشرح الصحيح يجب أن يكون مستنداً إلى النص أو التفسير المقدمين فقط.\n\n"
    "أعد JSON فقط بهذه المفاتيح:\n"
    "{\n"
    "  'explanation_status': 'موافق أو مخالف أو جزئياً موافق أو غير مؤكد',\n"
    "  'explanation_verdict': 'سبب الحكم في جملة أو جملتين',\n"
    "  'correct_explanation': 'تصحيح مستند إلى المصادر أو فارغ'\n"
    "}"
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

# Enum taxonomies.  "غير مؤكد" and "خطأ في التحقق" are deliberately distinct
# from "غير موجود": absence of retrieved evidence is not evidence of absence,
# and a service failure is not a verdict about the text.
_TEXT_STATUS_VALUES = {
    "صحيح", "ضعيف", "موضوع", "غير موجود", "غير مؤكد", "خطأ في التحقق",
}
_DANGER_LEVEL_VALUES = {"منخفضة", "متوسطة", "عالية", "غير محسوبة"}
_EXPLANATION_STATUS_VALUES = {
    "موافق", "مخالف", "جزئياً موافق", "لا يوجد", "غير مؤكد", "تعذّر التقييم",
}

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
# Structured result builders
# ---------------------------------------------------------------------------

_DANGER_MEANING = (
    "درجة الخطورة تقدير لاحتمال أن يؤدي الاقتباس أو تفسيره إلى فهم مضلل، "
    "وليست درجة ثقة في صحة النتيجة."
)


def _empty_interpretation(provided: bool = False) -> dict:
    return {
        "provided": provided,
        "status": "not_provided" if not provided else "inconclusive",
        "label": NOT_PROVIDED_LABEL if not provided else "لم يكتمل تقييم التفسير",
        "assessment": "",
        "correction": "",
        "error": "",
    }


def _base_result(
    text: str,
    explanation: str,
    *,
    verification_status: str,
    status_label: str,
    status_explanation: str,
    text_status: str,
    is_truncated: bool,
    missing_part: str,
    full_text: str,
    source: str,
    danger_level: str,
    explanation_status: str,
    explanation_verdict: str,
    correct_explanation: str,
    quotation: dict,
    interpretation: dict,
    evidence: list[dict],
    evidence_status: str = "",
    evidence_note: str = "",
    retrieval_ok: bool = True,
    verification_error: str = "",
) -> dict:
    danger_applicable = danger_level != "غير محسوبة"
    if not evidence_status:
        evidence_status = "relevant" if evidence else "none"
    return {
        # --- legacy contract (kept for existing consumers) ---
        "text_status": text_status,
        "is_truncated": bool(is_truncated),
        "missing_part": missing_part or "",
        "full_text": full_text or "",
        "source": source or "",
        "danger_level": danger_level,
        "explanation_status": explanation_status,
        "explanation_verdict": explanation_verdict or "",
        "correct_explanation": correct_explanation or "",
        "input_text": text,
        "input_explanation": explanation or "",
        # --- structured result ---
        "verification_status": verification_status,
        "status_label": status_label,
        "status_explanation": status_explanation,
        "quotation": quotation,
        "interpretation": interpretation,
        "evidence": evidence,
        "evidence_status": evidence_status,
        "evidence_note": evidence_note or "",
        "danger_level_meaning": _DANGER_MEANING,
        "danger_level_applicable": danger_applicable,
        "retrieval_ok": retrieval_ok,
        "verification_error": verification_error,
        "overall_status": (
            "error" if verification_status == "error" else verification_status
        ),
        "components": [],
        "component_count": 0,
    }


def _empty_quotation() -> dict:
    return {
        "found": False,
        "ambiguous": False,
        "match_type": "none",
        "is_partial": False,
        "is_complete": False,
        "coverage": 0.0,
        "matched_tokens": 0,
        "query_tokens": 0,
        "canonical_text": "",
        "reference": "",
        "source": "",
        "surah_name": "",
        "surah_number": None,
        "ayah_number": None,
        "candidates": [],
    }


def _inconclusive_result(
    text: str,
    explanation: str,
    *,
    reason: str,
    retrieval_ok: bool = True,
) -> dict:
    interpretation = _empty_interpretation(bool(explanation))
    if explanation:
        interpretation["label"] = "لم يتم تقييم التفسير لعدم كفاية دليل النص"
    return _base_result(
        text,
        explanation,
        verification_status="inconclusive",
        status_label=INCONCLUSIVE_LABEL,
        status_explanation=reason,
        text_status="غير مؤكد",
        is_truncated=False,
        missing_part="",
        full_text="",
        source="",
        danger_level="غير محسوبة",
        explanation_status=(
            "غير مؤكد" if explanation else NOT_PROVIDED_LABEL
        ),
        explanation_verdict="",
        correct_explanation="",
        quotation=_empty_quotation(),
        interpretation=interpretation,
        evidence=[],
        retrieval_ok=retrieval_ok,
    )


def _error_result(
    text: str,
    explanation: str,
    *,
    message: str,
    retrieval_ok: bool = False,
) -> dict:
    interpretation = _empty_interpretation(bool(explanation))
    interpretation.update({
        "status": "error",
        "label": "تعذّر تقييم التفسير",
        "error": message,
    })
    return _base_result(
        text,
        explanation,
        verification_status="error",
        status_label=VERIFICATION_ERROR_LABEL,
        status_explanation=(
            "حدث خطأ أثناء البحث في المصادر أو أثناء تقييم النتيجة. "
            "هذا ليس حكماً بأن النص غير موجود."
        ),
        text_status="خطأ في التحقق",
        is_truncated=False,
        missing_part="",
        full_text="",
        source="",
        danger_level="غير محسوبة",
        explanation_status="تعذّر التقييم",
        explanation_verdict="",
        correct_explanation="",
        quotation=_empty_quotation(),
        interpretation=interpretation,
        evidence=[],
        retrieval_ok=retrieval_ok,
        verification_error=message,
    )


def _find_tafsir_chunk(chunks: list[dict], surah: int | None, ayah: int | None) -> dict | None:
    if surah is None or ayah is None:
        return None
    wanted = f"تفسير ابن كثير {surah}:{ayah}"
    for chunk in chunks:
        if chunk.get("source") != "tafsir":
            continue
        if chunk.get("reference") == wanted:
            return chunk
    for chunk in chunks:
        if chunk.get("source") != "tafsir":
            continue
        metadata = chunk.get("metadata") or {}
        if (
            metadata.get("surah_number") == surah
            and metadata.get("ayah_number") == ayah
        ):
            return chunk
    return None


def _evidence_excerpt(text: str, limit: int = 900) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"


def _assess_interpretation(
    explanation: str,
    quran_chunk: dict,
    tafsir_chunk: dict | None,
    chunks: list[dict],
    client,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate the user's interpretation against the canonical verse/tafsir.

    This function never decides whether the verse exists; that decision was
    already made lexically.  It only returns a supported/contradicted/
    inconclusive assessment for the interpretation.
    """
    if not explanation:
        return _empty_interpretation(False)

    if client is None:
        result = _empty_interpretation(True)
        result.update({
            "status": "error",
            "label": "تعذّر تقييم التفسير",
            "error": "خدمة التقييم غير متاحة.",
        })
        return result

    evidence = [quran_chunk]
    if tafsir_chunk is not None:
        evidence.append(tafsir_chunk)

    context_parts = []
    for index, chunk in enumerate(evidence, 1):
        context_parts.append(
            f"[مصدر {index}] ({chunk.get('source', '')} — "
            f"{chunk.get('reference', '')}):\n"
            f"{chunk.get('original_text', '')}"
        )

    user_message = (
        f"النص القرآني والتفسير المتاح:\n\n"
        + "\n\n".join(context_parts)
        + f"\n\nالتفسير المقدم من المستخدم:\n{explanation}\n\n"
        "قيّم التفسير المقدم فقط."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTERPRETATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if debug:
            print(f"\n[DEBUG] Raw interpretation response:\n{raw}")
        try:
            answer = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            answer = {}
    except Exception as error:  # network/model failure is an error state
        if debug:
            print(f"[DEBUG] Interpretation assessment failed: {error}")
        result = _empty_interpretation(True)
        result.update({
            "status": "error",
            "label": "تعذّر تقييم التفسير",
            "error": "تعذّر الاتصال بخدمة تقييم التفسير.",
        })
        return result

    status = _validated_enum(
        get_key(
            answer,
            "explanation_status", "تقييم الشرح", "تقييم_الشرح",
            "حكم_الشرح", "status",
            default="غير مؤكد",
        ),
        {"موافق", "مخالف", "جزئياً موافق", "غير مؤكد"},
        "غير مؤكد",
    )
    verdict = str(get_key(
        answer,
        "explanation_verdict", "سبب الحكم", "سبب_الحكم", "السبب",
    ) or "").strip()
    correction = str(get_key(
        answer,
        "correct_explanation", "الشرح الصحيح", "الشرح_الصحيح",
        "الشرح الصحيح من المصادر",
    ) or "").strip()

    if correction:
        try:
            correction = ground_text(
                correction, evidence, chunks, refusal="",
            )
        except Exception:
            correction = ""

    status_map = {
        "موافق": ("supported", "التفسير مدعوم بالمصادر"),
        "مخالف": ("contradicted", "التفسير غير مدعوم أو يخالف المصادر"),
        "جزئياً موافق": (
            "partially_supported", "التفسير مدعوم جزئياً",
        ),
        "غير مؤكد": ("inconclusive", "المصادر لا تكفي للحكم على التفسير"),
    }
    key, label = status_map[status]

    return {
        "provided": True,
        "status": key,
        "label": label,
        "assessment": verdict,
        "correction": correction,
        "error": "",
        "source_status": status,
    }


def _interpretation_danger(interpretation: dict, is_partial: bool) -> str:
    if interpretation.get("status") == "contradicted":
        return "عالية"
    if interpretation.get("status") == "partially_supported":
        return "متوسطة"
    if interpretation.get("status") in {"supported", "not_provided"}:
        return "متوسطة" if is_partial else "منخفضة"
    return "غير محسوبة"


def _quran_result(
    text: str,
    explanation: str,
    match: dict,
    chunks: list[dict],
    client,
    *,
    debug: bool = False,
) -> dict:
    chunk = match.get("chunk") or {}
    canonical = match.get("canonical_text", "") or ""
    reference = match.get("reference", "") or ""
    surah = match.get("surah_name", "") or ""
    surah_number = match.get("surah_number")
    ayah_number = match.get("ayah_number")
    complete = bool(match.get("is_complete"))
    match_type = match.get("match_type", "none")
    is_partial = not complete

    tafsir_chunk = _find_tafsir_chunk(chunks, surah_number, ayah_number)
    evidence = [{
        "source": "quran",
        "reference": reference,
        "text": canonical,
        "role": "canonical",
    }]
    if tafsir_chunk is not None:
        evidence.append({
            "source": "tafsir",
            "reference": tafsir_chunk.get("reference", ""),
            "text": _evidence_excerpt(tafsir_chunk.get("original_text", "")),
            "role": "interpretation",
        })

    interpretation = _assess_interpretation(
        explanation, chunk, tafsir_chunk, chunks, client, debug=debug,
    )

    if match_type == "approximate":
        status_label = "تم العثور على اقتباس تقريبي"
        status_explanation = (
            f"النص قريب من {surah} {ayah_number} ({reference}) لكنه ليس "
            "مطابقاً حرفياً للآية كما وردت في المصحف. راجع النص الكامل قبل "
            "الاعتماد على الاقتباس."
        )
        verification_status = "partial"
    elif is_partial:
        status_label = "تم التحقق — اقتباس جزئي"
        status_explanation = (
            f"النص جزء حرفي من الآية {ayah_number} من سورة {surah} "
            f"({reference}). الاقتباس صحيح ولكنه غير مكتمل؛ النص الكامل "
            "معروض أدناه."
        )
        verification_status = "partial"
    else:
        status_label = "تم التحقق — اقتباس مطابق"
        status_explanation = (
            f"النص مطابق للآية {ayah_number} من سورة {surah} ({reference}) "
            "بحسب النص القرآني المحفوظ في قاعدة البيانات."
        )
        verification_status = "verified"

    if interpretation.get("status") == "contradicted":
        status_label += " — التفسير غير مدعوم"
        status_explanation += (
            " التفسير المقدم لا يوافق ما تدعمه المصادر، وقد تم توضيح "
            "موضع المشكلة أدناه."
        )
    elif interpretation.get("status") == "partially_supported":
        status_label += " — التفسير مدعوم جزئياً"
    elif interpretation.get("status") == "error":
        status_label += " — تعذّر تقييم التفسير"

    explanation_status = {
        "supported": "موافق",
        "contradicted": "مخالف",
        "partially_supported": "جزئياً موافق",
        "inconclusive": "غير مؤكد",
        "error": "تعذّر التقييم",
        "not_provided": NOT_PROVIDED_LABEL,
    }.get(interpretation.get("status"), "غير مؤكد")

    danger = _interpretation_danger(interpretation, is_partial)

    quotation = {
        "found": True,
        "ambiguous": False,
        "match_type": match_type,
        "is_partial": is_partial,
        "is_complete": complete,
        "coverage": match.get("coverage", 1.0),
        "matched_tokens": match.get("matched_tokens", 0),
        "query_tokens": match.get("query_tokens", 0),
        "canonical_text": canonical,
        "reference": reference,
        "source": "quran",
        "surah_name": surah,
        "surah_number": surah_number,
        "ayah_number": ayah_number,
        "candidates": [],
    }

    return _base_result(
        text,
        explanation,
        verification_status=verification_status,
        status_label=status_label,
        status_explanation=status_explanation,
        text_status="صحيح",
        is_truncated=is_partial,
        missing_part=canonical if is_partial else "",
        full_text=canonical,
        source=f"القرآن الكريم — {reference}",
        danger_level=danger,
        explanation_status=explanation_status,
        explanation_verdict=interpretation.get("assessment", ""),
        correct_explanation=interpretation.get("correction", ""),
        quotation=quotation,
        interpretation=interpretation,
        evidence=evidence,
        retrieval_ok=True,
    )


def _ambiguous_quran_result(
    text: str,
    explanation: str,
    match: dict,
) -> dict:
    refs = [
        candidate.get("reference", "")
        for candidate in match.get("candidates", [])
        if candidate.get("reference")
    ]
    shown = "، ".join(refs[:5])
    reason = (
        "النص يطابق أكثر من موضع في القرآن"
        + (f" ({shown})" if shown else "")
        + "، ولا يمكن الجزم بمرجع واحد. أضف جزءاً أطول من الآية أو "
        "اذكر اسم السورة ورقم الآية."
    )
    result = _inconclusive_result(text, explanation, reason=reason)
    result["quotation"] = {
        **_empty_quotation(),
        "ambiguous": True,
        "match_type": "ambiguous",
        "query_tokens": match.get("query_tokens", 0),
        "candidates": match.get("candidates", []),
    }
    result["status_label"] = "غير حاسم — تطابق أكثر من موضع"
    result["status_explanation"] = reason
    return result


# ---------------------------------------------------------------------------
# Semantic path for hadith and other source types
# ---------------------------------------------------------------------------

def _semantic_result(
    text: str,
    explanation: str,
    index,
    chunks: list[dict],
    client,
    *,
    top_k: int,
    debug: bool,
) -> dict:
    try:
        wide_results = _dual_search(
            text, client, index, chunks, top_k * 20, debug=debug,
        )
    except Exception as error:
        if debug:
            print(f"[DEBUG] Retrieval failed: {error}")
        return _error_result(
            text,
            explanation,
            message="تعذّر البحث في المصادر. لم يتم إصدار حكم على النص.",
            retrieval_ok=False,
        )

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

    if explanation:
        try:
            expl_results = _dual_search(
                explanation, client, index, chunks, top_k * 20, debug=debug,
            )
            for r in expl_results:
                if r.get("id") not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.get("id"))
        except Exception as error:
            if debug:
                print(f"[DEBUG] Explanation retrieval failed: {error}")

    if not results:
        return _inconclusive_result(
            text,
            explanation,
            reason=(
                "لم نعثر على نص مطابق في المصادر المتاحة. عدم العثور على "
                "دليل لا يعني بالضرورة أن النص غير صحيح؛ راجع المصدر الأصلي "
                "أو أعد المحاولة بمقطع أوضح."
            ),
        )

    # --- Build context ---
    sources_text = "\n\n".join(
        f"[مصدر {i + 1}] (تطابق: {r['score']:.2f})\n"
        f"النص: {r['original_text']}\n"
        f"المصدر: {r['source']}\n"
        f"المرجع: {r['reference']}"
        for i, r in enumerate(results)
    )

    # --- Deterministic truncation check (before GPT call) ---
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

    # --- Build user message ---
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

    # --- Call GPT-4o-mini ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_answer = response.choices[0].message.content
    except Exception as error:
        if debug:
            print(f"[DEBUG] Model verification failed: {error}")
        return _error_result(
            text,
            explanation,
            message=(
                "تعذّر تقييم النص بواسطة خدمة التحقق. لم يتم إصدار حكم "
                "على النص."
            ),
            retrieval_ok=True,
        )

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    try:
        answer = json.loads(raw_answer)
    except (json.JSONDecodeError, TypeError):
        answer = {}

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    # --- Build result dict with Arabic fallbacks ---
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
    text_status = _validated_enum(
        get_key(
            answer,
            "text_status", "صحة_النص", "صحة النص", "الحالة", "حالة_النص",
            "status", "النتيجة",
            default="غير مؤكد",
        ),
        _TEXT_STATUS_VALUES,
        "غير مؤكد",
    )

    # Citation: a non-empty GPT source is kept only when it matches a
    # reference that was actually retrieved; otherwise it is replaced with
    # the top retrieved reference (deterministic) or emptied.
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

    explanation_status = _validated_enum(
        get_key(
            answer,
            "explanation_status", "تقييم الشرح", "تقييم_الشرح",
            "حكم_الشرح",
            default="لا يوجد",
        ),
        _EXPLANATION_STATUS_VALUES,
        "لا يوجد",
    )
    if explanation and explanation_status == "لا يوجد":
        explanation_status = "غير مؤكد"

    if not explanation:
        interpretation = _empty_interpretation(False)
        explanation_status = NOT_PROVIDED_LABEL
    else:
        status_map = {
            "موافق": ("supported", "التفسير مدعوم بالمصادر"),
            "مخالف": ("contradicted", "التفسير غير مدعوم أو يخالف المصادر"),
            "جزئياً موافق": (
                "partially_supported", "التفسير مدعوم جزئياً",
            ),
            "غير مؤكد": ("inconclusive", "المصادر لا تكفي للحكم على التفسير"),
            "لا يوجد": ("inconclusive", "المصادر لا تكفي للحكم على التفسير"),
            "تعذّر التقييم": ("error", "تعذّر تقييم التفسير"),
        }
        key, label = status_map.get(
            explanation_status, ("inconclusive", "المصادر لا تكفي للحكم"),
        )
        interpretation = {
            "provided": True,
            "status": key,
            "label": label,
            "assessment": str(get_key(
                answer, "explanation_verdict", "سبب الحكم", "سبب_الحكم",
            ) or "").strip(),
            "correction": correct_explanation,
            "error": "",
            "source_status": explanation_status,
        }

    is_partial = bool(final_is_truncated)
    found = text_status in {"صحيح", "ضعيف"}
    if text_status == "موضوع":
        verification_status = "contradicted"
        status_label = "المعلومة تخالف الأدلة"
        status_explanation = (
            "المصادر المسترجعة تدعم أن هذا النص موضوع أو منسوب خطأً. "
            "راجع الاقتباس والمصدر."
        )
    elif text_status == "غير موجود":
        # Absence of a lexical match is not proof of fabrication.  Keep the
        # public status honest instead of presenting the model's absence
        # verdict as a confirmed fact.
        text_status = "غير مؤكد"
        verification_status = "inconclusive"
        status_label = "لا توجد أدلة كافية"
        status_explanation = (
            "لم يؤكد التحقق وجود النص في المصادر المسترجعة. "
            "هذا لا يعني بالضرورة أنه غير صحيح."
        )
    elif is_partial:
        verification_status = "partial"
        status_label = "تم التحقق — النص غير مكتمل"
        status_explanation = (
            "تم العثور على النص في المصادر، لكنه مختصر أو ناقص مقارنةً "
            "بالأصل. راجع النص الكامل أدناه."
        )
    else:
        verification_status = "verified"
        status_label = "تم التحقق — النص مدعوم بالمصادر"
        status_explanation = (
            "المصادر المسترجعة تدعم النص. راجع المرجع قبل الاعتماد."
        )

    if explanation and interpretation["status"] == "contradicted":
        status_label += " — التفسير غير مدعوم"
        status_explanation += (
            " التفسير المقدم لا يوافق ما تدعمه المصادر."
        )
    elif explanation and interpretation["status"] == "partially_supported":
        status_label += " — التفسير مدعوم جزئياً"
    elif explanation and interpretation["status"] == "error":
        status_label += " — تعذّر تقييم التفسير"

    if verification_status in {"inconclusive", "error"}:
        danger = "غير محسوبة"
    elif explanation:
        danger = _interpretation_danger(interpretation, is_partial)
    else:
        danger = _interpretation_danger(
            {"status": "not_provided"}, is_partial,
        )

    if not found:
        # An inconclusive or contradicted result must not present a retrieved
        # candidate as the source of the submitted text.
        source = ""

    quotation = {
        "found": found,
        "ambiguous": False,
        "match_type": "semantic" if found else "none",
        "is_partial": is_partial,
        "is_complete": found and not is_partial,
        "coverage": 0.0,
        "matched_tokens": 0,
        "query_tokens": len(normalize(text).split()),
        "canonical_text": (
            final_full_text if isinstance(final_full_text, str) else ""
        ) or (
            results[0].get("original_text", "") if found else ""
        ),
        "reference": (
            source or _first_retrieved_reference(results)
        ) if found else "",
        "source": results[0].get("source", "") if found else "",
        "surah_name": "",
        "surah_number": None,
        "ayah_number": None,
        "candidates": [],
    }

    evidence = [
        {
            "source": r.get("source", ""),
            "reference": r.get("reference", ""),
            "text": _evidence_excerpt(r.get("original_text", "")),
            "role": "retrieved",
        }
        for r in results[:5]
    ]
    if verification_status == "inconclusive" and not found:
        evidence_status = "not_relevant" if evidence else "none"
        evidence_note = (
            "تم البحث في المصادر، لكن لم يتم العثور على نص مرتبط مباشرة "
            "بالادعاء؛ لذلك لا تُعرض نصوص مسترجعة قد توحي بدليل غير موجود."
            if evidence
            else "لم يتم العثور على نصوص مسترجعة مرتبطة بهذا الادعاء."
        )
        evidence = []
    elif evidence:
        evidence_status = "relevant"
        evidence_note = ""
    else:
        evidence_status = "none"
        evidence_note = "لم يتم العثور على نصوص مسترجعة مرتبطة بهذا الادعاء."

    return _base_result(
        text,
        explanation,
        verification_status=verification_status,
        status_label=status_label,
        status_explanation=status_explanation,
        text_status=text_status,
        is_truncated=is_partial,
        missing_part=final_missing_part,
        full_text=final_full_text,
        source=source,
        danger_level=danger,
        explanation_status=explanation_status,
        explanation_verdict=str(get_key(
            answer, "explanation_verdict", "سبب الحكم", "سبب_الحكم",
        ) or "").strip(),
        correct_explanation=correct_explanation,
        quotation=quotation,
        interpretation=interpretation,
        evidence=evidence,
        evidence_status=evidence_status,
        evidence_note=evidence_note,
        retrieval_ok=True,
    )


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
    """Verify text authenticity + completeness + optional explanation.

    Pipeline:
        1. Normalise the text and resolve it lexically against the canonical
           Quran corpus.  A found quotation returns its exact reference and
           full canonical verse without any model call.
        2. If no Quran quotation is found, retrieve hadith/other evidence and
           use the established verification prompt + citation validation.
        3. If an explanation is supplied, assess it independently of whether
           the quotation exists.

    Args:
        text:        The Arabic text to verify.
        explanation: Optional explanation/interpretation to evaluate.
        index:       A loaded FAISS index (loaded automatically if None).
        chunks:      The chunk dicts aligned with the index.
        client:      An authenticated OpenAI client (created if None).
        top_k:       Number of FAISS results to retrieve per search.
        debug:       If True, print intermediate results.

    Returns:
        A dict with the legacy 11 keys plus structured verification fields.
    """
    text = text or ""
    explanation = (explanation or "").strip()

    if not text.strip():
        return _error_result(
            text,
            explanation,
            message="النص مطلوب لإجراء التحقق.",
        )

    # --- Auto-load resources if not provided ---
    if chunks is None or index is None:
        try:
            loaded_index, loaded_chunks = load_faiss_index()
            if index is None:
                index = loaded_index
            if chunks is None:
                chunks = loaded_chunks
        except Exception as error:
            if debug:
                print(f"[DEBUG] Resource loading failed: {error}")
            return _error_result(
                text,
                explanation,
                message=(
                    "تعذّر تحميل قاعدة المصادر. لم يتم إصدار حكم على النص."
                ),
                retrieval_ok=False,
            )

    # --- Step 0: full-post analysis ---
    # A complete post contains several claims/quotes and requires separate
    # source, context, and argument assessments.  Simple single-quotation
    # inputs keep the established two-stage path below.
    try:
        from src.rag.post_analysis import analyze_post, looks_like_post

        if looks_like_post(text):
            if client is None:
                try:
                    client = get_client()
                except Exception:
                    client = None
            post_result = analyze_post(
                text,
                explanation,
                index,
                chunks,
                client,
                top_k=top_k,
                debug=debug,
            )
            if post_result is not None:
                return post_result
    except Exception as error:
        if debug:
            print(f"[DEBUG] Post analysis failed: {error}")
        return _error_result(
            text,
            explanation,
            message=(
                "تعذّر تحليل المنشور أو البحث في المصادر. لم يتم إصدار "
                "حكم على المحتوى."
            ),
            retrieval_ok=False,
        )

    # --- Step 1: deterministic Quran quotation matching ---
    try:
        quotation = match_quran_quotation(text, chunks, debug=debug)
    except Exception as error:
        if debug:
            print(f"[DEBUG] Quran quotation matching failed: {error}")
        quotation = None

    if quotation and quotation.get("ambiguous"):
        # A fragment that matches several verses is genuinely ambiguous.
        # Returning an inconclusive result is safer than letting a semantic
        # search choose one verse or letting the model declare absence.
        return _ambiguous_quran_result(text, explanation, quotation)

    if quotation and quotation.get("found"):
        return _quran_result(
            text,
            explanation,
            quotation,
            chunks,
            client,
            debug=debug,
        )

    # --- Step 2: semantic path for hadith and other sources ---
    if client is None:
        try:
            client = get_client()
        except Exception as error:
            if debug:
                print(f"[DEBUG] Model client initialization failed: {error}")
            return _error_result(
                text,
                explanation,
                message=(
                    "تعذّر تهيئة خدمة التحقق. لم يتم إصدار حكم على النص."
                ),
                retrieval_ok=False,
            )

    return _semantic_result(
        text,
        explanation,
        index,
        chunks,
        client,
        top_k=top_k,
        debug=debug,
    )


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
        # Test 5 — اقتباس جزئي
        {
            "label": "Test 5 — اقتباس جزئي من النساء 43",
            "text": "ولا تقربوا الصلاة",
            "explanation": None,
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

        print(f"\n  حالة التحقق:       {result['verification_status']}")
        print(f"  العنوان:           {result['status_label']}")
        print(f"  صحة النص:          {result['text_status']}")
        print(f"  مبتور:             {'نعم' if result['is_truncated'] else 'لا'}")
        print(f"  المرجع:            {result['quotation']['reference']}")
        print(f"  النص الكامل:       {result['full_text']}")
        print(f"  المصدر:            {result['source']}")
        print(f"  درجة الخطورة:      {result['danger_level']}")
        print(f"  تقييم الشرح:       {result['explanation_status']}")
        print(f"  الشرح الصحيح:      {result['correct_explanation']}")
        print(f"{'=' * 70}")
