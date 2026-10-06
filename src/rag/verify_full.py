"""
Basira RAG — Unified Verification
=====================================
Verifies the authenticity and completeness of Arabic religious texts,
and optionally evaluates an accompanying explanation — all in a single
GPT-4o-mini call backed by FAISS semantic search.

Replaces: verify.py, detect_truncation.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search
from src.nlp.preprocessor import preprocess

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

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

    # --- Step 1: Dual-embedding search on text ---
    results = _dual_search(text, client, index, chunks, top_k, debug=debug)

    # --- Step 2: If explanation provided, search on it too & merge ---
    if explanation:
        expl_results = _dual_search(explanation, client, index, chunks, top_k, debug=debug)

        # Merge, deduplicate by chunk id
        seen_ids = {r.get("id") for r in results}
        for r in expl_results:
            if r.get("id") not in seen_ids:
                results.append(r)
                seen_ids.add(r.get("id"))

    # --- Step 3: Build context and call GPT-4o-mini ---
    sources_text = "\n\n".join(
        f"[مصدر {i + 1}] (تطابق: {r['score']:.2f})\n"
        f"النص: {r['original_text']}\n"
        f"المصدر: {r['source']}\n"
        f"المرجع: {r['reference']}"
        for i, r in enumerate(results)
    )

    user_parts = [f"النص المدخل: {text}"]
    if explanation:
        user_parts.append(f"الشرح المقدم: {explanation}")
    user_parts.append(f"\nالمصادر المسترجعة:\n{sources_text}")
    user_message = "\n\n".join(user_parts)

    if debug:
        print(f"\n[DEBUG] User message length: {len(user_message)} chars")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
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
    is_truncated_val = get_key(
        answer,
        "is_truncated", "هل النص مبتور أو منقوص", "هل النص مبتور", "مبتور",
        "اكتمال النص", "اكتمال_النص",
        default="لا",
    )

    return {
        "text_status": get_key(
            answer,
            "text_status", "صحة_النص", "صحة النص", "الحالة", "حالة_النص",
            "status", "النتيجة",
            default="غير موجود",
        ),
        "is_truncated": _is_yes(is_truncated_val),
        "missing_part": get_key(
            answer,
            "missing_part", "الجزء الناقص", "الجزء_الناقص",
            "الجزء الناقص من النص",
        ),
        "full_text": get_key(
            answer,
            "full_text", "النص الكامل", "النص_الكامل",
            "النص الكامل من المصدر",
        ),
        "source": _flatten_source(
            get_key(answer, "source", "المصدر", "مصدر", "المصدر_الموثوق")
        ),
        "danger_level": get_key(
            answer,
            "danger_level", "درجة الخطورة", "درجة_الخطورة", "الخطورة",
            default="منخفضة",
        ),
        "explanation_status": get_key(
            answer,
            "explanation_status", "تقييم الشرح", "تقييم_الشرح",
            "حكم_الشرح",
            default="لا يوجد",
        ),
        "explanation_verdict": get_key(
            answer,
            "explanation_verdict", "سبب الحكم", "سبب_الحكم",
        ),
        "correct_explanation": get_key(
            answer,
            "correct_explanation", "الشرح الصحيح", "الشرح_الصحيح",
            "الشرح الصحيح من المصادر",
        ),
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
        print(f"  الجزء الناقص:      {result['missing_part']}")
        print(f"  النص الكامل:       {result['full_text']}")
        print(f"  المصدر:            {result['source']}")
        print(f"  درجة الخطورة:      {result['danger_level']}")
        print(f"  تقييم الشرح:       {result['explanation_status']}")
        print(f"  سبب الحكم:         {result['explanation_verdict']}")
        print(f"  الشرح الصحيح:      {result['correct_explanation']}")
        print(f"{'=' * 70}")
