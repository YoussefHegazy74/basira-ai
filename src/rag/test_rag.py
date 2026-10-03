"""
Basira RAG — End-to-End Test
==============================
Loads the saved FAISS index and chunks, runs three Arabic test queries
through the full RAG pipeline, and prints answers with sources and timing.
"""

from __future__ import annotations

import time

from src.vector_db.faiss_store import load_index, search
from src.embeddings.embedder import get_client, embed_text
from src.rag.pipeline import generate

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


if __name__ == "__main__":
    main()
