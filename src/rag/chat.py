"""
Basira RAG — Interactive Chat CLI
====================================
A simple REPL that lets users ask Arabic Islamic questions
and get answers powered by the full RAG pipeline.
"""

from __future__ import annotations

from src.vector_db.faiss_store import load_index
from src.embeddings.embedder import get_client
from src.rag.pipeline import generate

# Paths to pre-built artefacts
INDEX_PATH = "data/processed/faiss.index"
CHUNKS_PATH = "data/processed/chunks.pkl"

SEPARATOR = "=" * 60


def main() -> None:
    """Run the interactive chat loop."""
    # --- Load resources ---
    print(SEPARATOR)
    print("📦  جاري تحميل قاعدة البيانات …")
    print(SEPARATOR)
    index, chunks = load_index(INDEX_PATH, CHUNKS_PATH)
    client = get_client()
    print()

    # --- Welcome ---
    print(SEPARATOR)
    print("مرحباً بك في بصيرة — اسأل أي سؤال ديني")
    print("اكتب «خروج» أو «exit» للخروج")
    print(SEPARATOR)
    print()

    # --- Chat loop ---
    try:
        while True:
            query = input("> ").strip()

            if not query:
                continue

            if query in ("خروج", "exit"):
                print("\nإلى اللقاء! 👋")
                break

            result = generate(query, index, chunks, client)

            # Answer
            print(f"\n{result['answer']}\n")

            # Sources
            print("المصادر:")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source}")

            print()
            print(SEPARATOR)
            print()

    except (KeyboardInterrupt, EOFError):
        print("\n\nإلى اللقاء! 👋")


if __name__ == "__main__":
    main()
