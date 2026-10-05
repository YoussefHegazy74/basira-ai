"""
Basira RAG — Tafsir Retrieval
================================
Retrieves and presents the Tafsir Ibn Kathir for a given Quranic verse
by performing semantic search against a FAISS index of trusted sources
and then summarising the result using GPT-4o-mini.
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
    "أنت متخصص في تفسير القرآن الكريم. بناءً على مقاطع تفسير ابن كثير المقدمة فقط:\n"
    "1. قدّم تفسير الآية المدخلة بأسلوب واضح وموجز\n"
    "2. اذكر السياق والسبب في النزول إن وُجد في المصادر\n"
    "3. اذكر المصدر بدقة (رقم السورة والآية)\n"
    "لا تجاوب من معرفتك الخاصة. إذا لم تجد الآية في المصادر، "
    "قل: 'لم أجد تفسير هذه الآية في قاعدة البيانات'.\n"
    "أجب بالعربية فقط في شكل JSON بهذه المفاتيح:\n"
    "{\n"
    "  'tafsir': 'نص التفسير',\n"
    "  'context': 'السياق وسبب النزول أو فارغ',\n"
    "  'source': 'رقم السورة:رقم الآية'\n"
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

    print(f"[tafsir] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[tafsir] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


def get_key(d: dict, *keys: str, default: str = ""):
    """Return the value of the first key found in *d*, else *default*."""
    for key in keys:
        if key in d:
            return d[key]
    return default


def _flatten_source(source_val) -> str:
    """Normalise a source value that may be a dict into a flat string.

    GPT sometimes returns the source as a nested object, e.g.
    ``{"رقم_السورة": "…", "رقم_الآية": "…"}``.
    """
    if isinstance(source_val, dict):
        surah = source_val.get("رقم_السورة", source_val.get("surah", ""))
        ayah = source_val.get("رقم_الآية", source_val.get("ayah", ""))
        if surah or ayah:
            return f"{surah}:{ayah}" if ayah else str(surah)
        # Fallback: book + number style
        book = source_val.get("اسم_الكتاب", source_val.get("book", ""))
        number = source_val.get("رقم_الحديث", source_val.get("number", ""))
        return f"{book}:{number}" if number else book
    return str(source_val) if source_val else ""


def _filter_tafsir(results: list[dict]) -> list[dict]:
    """Keep only chunks whose source is 'tafsir'."""
    return [r for r in results if r.get("source") == "tafsir"]


def get_tafsir(
    ayah_text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Retrieve Tafsir Ibn Kathir for a Quranic verse.

    Pipeline:
        1. Preprocess the input ayah (normalize, tokenize, remove stopwords).
        2. Embed both normalized and raw text using OpenAI embeddings.
        3. Search FAISS with both embeddings, pick best top-1 score.
        4. Filter results to keep only tafsir chunks.
        5. Send the input + retrieved tafsir to GPT-4o-mini for summarisation.
        6. Return a structured tafsir result.

    Args:
        ayah_text: The Quranic verse text to look up.
        index:     A loaded FAISS index.
        chunks:    The chunk dicts aligned with the index.
        client:    An authenticated OpenAI client.
        top_k:     Number of FAISS results to retrieve.
        debug:     If True, print raw GPT response and top FAISS hits.

    Returns:
        Dict with keys: tafsir, context, source, input_ayah.
    """
    # 1. Preprocess
    processed = preprocess(ayah_text)
    normalized_text = processed["normalized"]

    # 2. Embed both normalized and original, use the one with higher max score
    emb_normalized = embed_text(normalized_text, client)
    emb_original = embed_text(ayah_text, client)

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

    # 4. Filter to tafsir chunks only
    tafsir_results = _filter_tafsir(results)

    # If no tafsir chunks in initial results, progressively widen the search
    for multiplier in (20, 50):
        if tafsir_results:
            break
        wide_k = top_k * multiplier
        if debug:
            print(
                f"[DEBUG] No tafsir chunks found yet, "
                f"widening to top {wide_k}…"
            )
        results_norm_wide = search(emb_normalized, index, chunks, top_k=wide_k)
        results_orig_wide = search(emb_original, index, chunks, top_k=wide_k)

        if results_orig_wide and results_norm_wide:
            results_wide = (
                results_orig_wide
                if results_orig_wide[0]["score"] > results_norm_wide[0]["score"]
                else results_norm_wide
            )
        else:
            results_wide = results_norm_wide or results_orig_wide

        tafsir_results = _filter_tafsir(results_wide)

    # Debug: show top FAISS tafsir hits
    if debug:
        print(f"\n[DEBUG] Top FAISS tafsir results ({len(tafsir_results)} found):")
        for i, r in enumerate(tafsir_results[:3]):
            print(
                f"  {i + 1}. score={r['score']:.4f}  "
                f"ref={r.get('reference', 'N/A')}  "
                f"text={r.get('original_text', '')[:80]}…"
            )

    # 5. Build context from retrieved tafsir sources
    if tafsir_results:
        sources_text = "\n\n".join(
            f"[مصدر {i + 1}] (تطابق: {r['score']:.2f})\n"
            f"النص: {r['original_text']}\n"
            f"المصدر: {r['source']}\n"
            f"المرجع: {r['reference']}"
            for i, r in enumerate(tafsir_results)
        )
    else:
        sources_text = "لا توجد مصادر تفسير متاحة لهذه الآية."

    user_message = (
        f"الآية المدخلة: {ayah_text}\n\n"
        f"مقاطع التفسير المسترجعة:\n{sources_text}"
    )

    # 6. Call GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    # 7. Parse the JSON response
    raw_answer = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        answer = {
            "tafsir": "لم أجد تفسير هذه الآية في قاعدة البيانات",
            "context": "",
            "source": "",
        }

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    return {
        "tafsir": get_key(
            answer,
            "tafsir", "التفسير", "نص_التفسير", "تفسير",
            default="لم أجد تفسير هذه الآية في قاعدة البيانات",
        ),
        "context": get_key(
            answer,
            "context", "السياق", "سبب_النزول", "السياق_وسبب_النزول",
            "السياق وسبب النزول",
        ),
        "source": _flatten_source(
            get_key(answer, "source", "المصدر", "مصدر", "المصدر_الموثوق", default="")
        ),
        "input_ayah": ayah_text,
    }


if __name__ == "__main__":
    # --- Load resources ---
    client = get_client()
    index, chunks = load_faiss_index()

    # --- Diagnostic: check source distribution in chunks ---
    print("=== DIAGNOSTIC ===")
    sources = {}
    for c in chunks:
        s = c.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print("Source distribution:", sources)

    tafsir_chunks = [c for c in chunks if c.get("source") == "tafsir"]
    print(f"Total tafsir chunks: {len(tafsir_chunks)}")
    if tafsir_chunks:
        print("Sample tafsir chunk keys:", list(tafsir_chunks[0].keys()))
        print("Sample tafsir clean_text:", tafsir_chunks[0].get("clean_text", "")[:100])
        print("Sample tafsir reference:", tafsir_chunks[0].get("reference", ""))
    print("==================")

    # --- Test cases ---
    test_inputs = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
    ]

    for ayah in test_inputs:
        print(f"\n{'=' * 60}")
        print(f"الآية المدخلة: {ayah}")
        print("-" * 60)

        result = get_tafsir(ayah, index, chunks, client, debug=True)

        print(f"التفسير:       {result['tafsir']}")
        print(f"السياق:        {result['context']}")
        print(f"المصدر:        {result['source']}")
        print(f"الآية المدخلة: {result['input_ayah']}")
        print("=" * 60)
