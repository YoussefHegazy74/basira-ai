"""
Basira RAG — End-to-End Test
==============================
Loads the saved FAISS index and chunks, runs three Arabic test queries
through the full RAG pipeline, prints answers with sources and timing,
then runs verify_full() tests for text authenticity + explanation evaluation.
"""

from __future__ import annotations

import time

from src.vector_db.faiss_store import load_index, search
from src.embeddings.embedder import get_client, embed_text
from src.rag.pipeline import generate
from src.rag.verify_full import verify_full, load_faiss_index

# Paths to pre-built artefacts
INDEX_PATH = "data/processed/faiss.index"
CHUNKS_PATH = "data/processed/chunks.pkl"

# Test queries
QUERIES = [
    "ما صحة حديث الدين النصيحة؟",
    "ما تفسير بسم الله الرحمن الرحيم؟",
    "ما معنى الصلاة في الإسلام؟",
]

SEPARATOR = "=" * 80


def main() -> None:
    """Run the end-to-end RAG test."""
    # --- Load index and chunks ---
    print(SEPARATOR)
    print("📦  Loading FAISS index and chunks …")
    print(SEPARATOR)
    index, chunks = load_index(INDEX_PATH, CHUNKS_PATH)
    print()

    # --- Initialise OpenAI client ---
    client = get_client()
    print()

    # --- Run each query ---
    total_start = time.time()

    for i, query in enumerate(QUERIES, 1):
        print(SEPARATOR)
        print(f"🔍  Query {i}: {query}")
        print(SEPARATOR)

        query_start = time.time()
        result = generate(query, index, chunks, client)
        query_time = time.time() - query_start

        # Answer
        print(f"\n📝  Answer:\n{result['answer']}\n")

        # Sources
        print("📚  Sources:")
        for j, source in enumerate(result["sources"], 1):
            print(f"   {j}. {source}")

        # Timing
        print(f"\n⏱️  Time: {query_time:.2f}s")
        print()

    total_time = time.time() - total_start
    print(SEPARATOR)
    print(f"✅  All {len(QUERIES)} queries completed in {total_time:.2f}s")
    print(SEPARATOR)

    # -----------------------------------------------------------------
    # Verification tests using verify_full()
    # -----------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("🔎  Test A — verify_full بدون شرح")
    print(SEPARATOR)

    text_a = "الدين النصيحة"
    result_a = verify_full(text_a, index=index, chunks=chunks, client=client)

    print(f"  النص:          {text_a}")
    print(f"  صحة النص:      {result_a['text_status']}")
    print(f"  مبتور:         {'نعم' if result_a['is_truncated'] else 'لا'}")
    print(f"  المصدر:        {result_a['source']}")
    print()

    print(SEPARATOR)
    print("🔎  Test B — verify_full مع شرح خاطئ")
    print(SEPARATOR)

    text_b = "الدين النصيحة"
    explanation_b = "يعني المصلحة الشخصية فقط"
    result_b = verify_full(
        text_b, explanation=explanation_b,
        index=index, chunks=chunks, client=client,
    )

    print(f"  النص:          {text_b}")
    print(f"  الشرح:         {explanation_b}")
    print(f"  صحة النص:      {result_b['text_status']}")
    print(f"  تقييم الشرح:   {result_b['explanation_status']}")
    print(f"  الشرح الصحيح:  {result_b['correct_explanation']}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
