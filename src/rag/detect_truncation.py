"""
Basira RAG — Truncation Detection
====================================
Detects whether an Arabic religious text has been truncated, taken out
of context, or is missing critical parts that change its meaning.
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
    "أنت متخصص في تحليل النصوص الإسلامية. مهمتك هي:\n"
    "مقارنة النص المدخل من المستخدم بالنصوص الكاملة الموجودة في المصادر.\n\n"
    "قواعد صارمة:\n"
    "1. إذا كان النص المدخل جزءاً من نص أطول موجود في المصادر → is_truncated: نعم\n"
    "2. إذا كان النص المدخل يغير المعنى عند اقتطاعه من سياقه → out_of_context: نعم\n"
    "3. اذكر دائماً الجزء الناقص من النص الكامل في missing_part\n"
    "4. اذكر المصدر الكامل في source (اسم الكتاب ورقم الحديث أو رقم الآية)\n"
    "5. لا تجاوب من معرفتك الخاصة — فقط من المصادر المقدمة\n\n"
    "أجب بالعربية فقط في شكل JSON بهذه المفاتيح بالضبط:\n"
    "{\n"
    "  'is_truncated': 'نعم أو لا',\n"
    "  'out_of_context': 'نعم أو لا',\n"
    "  'missing_part': 'الجزء الناقص',\n"
    "  'full_context': 'النص الكامل والمعنى الصحيح',\n"
    "  'danger_level': 'منخفضة أو متوسطة أو عالية',\n"
    "  'source': 'اسم الكتاب:رقم الحديث'\n"
    "}"
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

    print(f"[detect_truncation] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[detect_truncation] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


def get_key(d: dict, *keys: str, default: str = ""):
    """Return the value of the first key found in *d*, else *default*."""
    for key in keys:
        if key in d:
            return d[key]
    return default


def _is_yes(val) -> bool:
    """Interpret a GPT yes/no value as a Python bool.

    Handles Arabic (نعم/لا), English (yes/no/true/false), and native bools.
    """
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"نعم", "yes", "true", "1", "صحيح"}


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


def detect_truncation(
    text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Detect if an Arabic religious text is truncated or used out of context.

    Pipeline:
        1. Preprocess the input text (normalize, tokenize, remove stopwords).
        2. Embed the normalized text using OpenAI embeddings.
        3. Search the FAISS index for the top_k most similar chunks.
        4. Send the input + retrieved sources to GPT-4o-mini for analysis.
        5. Return a structured truncation-detection result.

    Args:
        text:   The Arabic text to check for truncation.
        index:  A loaded FAISS index.
        chunks: The chunk dicts aligned with the index.
        client: An authenticated OpenAI client.
        top_k:  Number of FAISS results to retrieve.

    Returns:
        Dict with keys: is_truncated, out_of_context, missing_part,
                        full_context, danger_level, source.
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
        answer = {}

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    is_truncated_val = get_key(
        answer,
        "is_truncated", "هل النص مبتور أو منقوص", "هل النص مبتور", "مبتور",
        default="لا",
    )
    out_of_context_val = get_key(
        answer,
        "out_of_context", "هل النص مستخدم خارج سياقه", "هل النص خارج سياقه",
        "خارج_السياق",
        default="لا",
    )

    return {
        "is_truncated": _is_yes(is_truncated_val),
        "out_of_context": _is_yes(out_of_context_val),
        "missing_part": get_key(
            answer,
            "missing_part", "الجزء الناقص من النص", "الجزء الناقص", "الجزء_الناقص",
        ),
        "full_context": get_key(
            answer,
            "full_context", "السياق الكامل والمعنى الصحيح", "السياق الكامل",
            "السياق_الكامل",
        ),
        "danger_level": get_key(
            answer,
            "danger_level", "درجة الخطورة", "درجة_الخطورة", "الخطورة",
            default="منخفضة",
        ),
        "source": _flatten_source(
            get_key(answer, "source", "المصدر", "مصدر", "المصدر_الموثوق", default="")
        ),
    }


if __name__ == "__main__":
    # --- Load resources ---
    client = get_client()
    index, chunks = load_faiss_index()

    # --- Test cases ---
    test_inputs = [
        ("الدين النصيحة", "مبتور — الحديث الكامل أطول"),
        ("لا صلاة لجار المسجد إلا في المسجد", "مبتور ومختلف عليه"),
    ]

    for text, description in test_inputs:
        print(f"\n{'=' * 60}")
        print(f"النص المدخل: {text}")
        print(f"الوصف: {description}")
        print("-" * 60)

        result = detect_truncation(text, index, chunks, client, debug=True)

        print(f"مبتور:         {'نعم' if result['is_truncated'] else 'لا'}")
        print(f"خارج السياق:   {'نعم' if result['out_of_context'] else 'لا'}")
        print(f"الجزء الناقص:  {result['missing_part']}")
        print(f"السياق الكامل: {result['full_context']}")
        print(f"درجة الخطورة:  {result['danger_level']}")
        print(f"المصدر:        {result['source']}")
        print("=" * 60)
