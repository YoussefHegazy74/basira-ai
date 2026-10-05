"""
Basira RAG — Text Verification
================================
Verifies the authenticity of Arabic religious texts by performing
semantic search against a FAISS index of trusted sources and then
classifying the result using GPT-4o-mini.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss

from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search
from src.nlp.preprocessor import preprocess

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

SYSTEM_PROMPT = (
    "أنت متخصص في التحقق من صحة النصوص الإسلامية. بناءً على المصادر المقدمة، حدد:\n"
    "1. هل النص صحيح؟ (صحيح / ضعيف / موضوع / غير موجود)\n"
    "2. درجة التطابق مع المصدر الأصلي (عالية / متوسطة / منخفضة)\n"
    "3. النص الأصلي الكامل من المصدر\n"
    "4. المصدر الموثوق (اسم الكتاب ورقم الحديث أو رقم الآية)\n"
    "أجب بالعربية فقط في شكل JSON."
)


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

    print(f"[verify] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[verify] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


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


def verify_text(
    text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Verify the authenticity of an Arabic religious text.

    Pipeline:
        1. Preprocess the input text (normalize, tokenize, remove stopwords).
        2. Embed the normalized text using OpenAI embeddings.
        3. Search the FAISS index for the top_k most similar chunks.
        4. Send the input + retrieved sources to GPT-4o-mini for classification.
        5. Return a structured verification result.

    Args:
        text:   The Arabic text to verify.
        index:  A loaded FAISS index.
        chunks: The chunk dicts aligned with the index.
        client: An authenticated OpenAI client.
        top_k:  Number of FAISS results to retrieve.

    Returns:
        Dict with keys: status, match_level, original_text, source, input_text.
    """
    # 1. Preprocess
    processed = preprocess(text)
    normalized_text = processed["normalized"]

    # 2. Embed both normalized and original, use the one with higher max score
    emb_normalized = embed_text(normalized_text, client)
    emb_original = embed_text(text, client)

    # 3. Search FAISS with both embeddings
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

    # Debug: show top FAISS hits
    if debug:
        print("\n[DEBUG] Top FAISS results:")
        for i, r in enumerate(results[:3]):
            print(
                f"  {i + 1}. score={r['score']:.4f}  "
                f"ref={r.get('reference', 'N/A')}  "
                f"text={r.get('original_text', '')[:80]}…"
            )

    # 4. Build context from retrieved sources
    sources_text = "\n\n".join(
        f"[مصدر {i + 1}] (تطابق: {r['score']:.2f})\n"
        f"النص: {r['original_text']}\n"
        f"المصدر: {r['source']}\n"
        f"المرجع: {r['reference']}"
        for i, r in enumerate(results)
    )

    user_message = (
        f"النص المدخل: {text}\n\n"
        f"المصادر المسترجعة:\n{sources_text}"
    )

    # 5. Call GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    # 6. Parse the JSON response
    raw_answer = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        answer = {
            "status": "غير موجود",
            "match_level": "منخفضة",
            "original_text": "",
            "source": "",
        }

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    return {
        "status": get_key(
            answer,
            "status", "الحالة", "حالة_النص", "صحة_النص", "النص_صحيح", "النتيجة",
            default="غير موجود",
        ),
        "match_level": get_key(
            answer,
            "match_level", "درجة_التطابق", "مستوى_التطابق", "التطابق",
            default="منخفضة",
        ),
        "original_text": get_key(
            answer,
            "original_text", "النص_الأصلي", "النص الأصلي", "الأصل",
        ),
        "source": _flatten_source(
            get_key(answer, "source", "المصدر", "مصدر", "المصدر_الموثوق")
        ),
        "input_text": text,
    }


if __name__ == "__main__":
    # --- Load resources ---
    client = get_client()
    index, chunks = load_faiss_index()

    # --- Test cases ---
    test_inputs = [
        ("الدين النصيحة", "مبتور"),
        ("إنما الأعمال بالنيات", "صحيح"),
    ]

    for text, expected_label in test_inputs:
        print(f"\n{'=' * 60}")
        print(f"النص المدخل: {text}")
        print(f"التصنيف المتوقع: {expected_label}")
        print("-" * 60)

        result = verify_text(text, index, chunks, client, debug=True)

        print(f"الحالة:        {result['status']}")
        print(f"درجة التطابق:  {result['match_level']}")
        print(f"النص الأصلي:   {result['original_text']}")
        print(f"المصدر:        {result['source']}")
        print(f"النص المدخل:   {result['input_text']}")
        print("=" * 60)
