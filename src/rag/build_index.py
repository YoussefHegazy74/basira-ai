"""
Basira RAG — Build Index
==========================
End-to-end script that builds the full Basira knowledge base:
    load raw data → chunk → embed → build FAISS index → save to disk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.loader import load_all
from src.data_pipeline.chunker import chunk_all
from src.embeddings.embedder import get_client, embed_chunks, save_embeddings
from src.vector_db.faiss_store import build_index, save_index

# --- Paths ---
QURAN_PATH = "data/raw/quran/quran_processed.json"
HADITH_PATH = "data/raw/hadith/hadith_processed.json"
TAFSIR_PATH = "data/raw/tafsir/tafsir_processed.json"
FIQH_PATH = "data/raw/fiqh/fiqh_processed.json"
SIRA_PATH = "data/raw/sira/sira_processed.json"

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.pkl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.pkl"


def _fmt_time(seconds: float) -> str:
    """Format elapsed time as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


if __name__ == "__main__":
    overall_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load all data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/4 — Loading raw data")
    print("=" * 60)
    t0 = time.time()

    records = load_all(
        quran_path=QURAN_PATH,
        hadith_path=HADITH_PATH,
        tafsir_path=TAFSIR_PATH,
        fiqh_path=FIQH_PATH,
        sira_path=SIRA_PATH,
    )

    print(f"⏱  Loading took {_fmt_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 2: Chunk all records
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2/4 — Chunking records")
    print("=" * 60)
    t0 = time.time()

    chunks = chunk_all(records)

    print(f"⏱  Chunking took {_fmt_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Confirmation before embedding (costs money)
    # ------------------------------------------------------------------
    total_chunks = len(chunks)
    print(f"\n📊  Total chunks to embed: {total_chunks:,}")
    answer = input(
        f"\nAbout to embed {total_chunks:,} chunks (Quran + Hadith + Tafsir + Fiqh + Sira). "
        f"Continue? (y/n) "
    )
    if answer.strip().lower() != "y":
        print("Aborted by user.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 3: Embed all chunks (with retry on failure)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/4 — Embedding chunks")
    print("=" * 60)
    t0 = time.time()

    client = get_client()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 50
    start_batch = 0
    max_retries = 3

    for attempt in range(max_retries + 1):
        try:
            chunks = embed_chunks(
                chunks, client, batch_size=batch_size, start_batch=start_batch,
            )
            break  # success
        except Exception as e:
            if attempt >= max_retries:
                print(f"\n❌  Embedding failed after {max_retries} retries: {e}")
                sys.exit(1)
            # Figure out which batch we were on by checking which chunks
            # already have embeddings
            embedded_so_far = sum(1 for c in chunks if "embedding" in c)
            start_batch = embedded_so_far // batch_size
            print(f"\n⚠️  Error at batch ~{start_batch}: {e}")
            print(f"    Retrying from batch {start_batch} (attempt {attempt + 2}/{max_retries + 1})...")
            time.sleep(2)

    save_embeddings(chunks, EMBEDDINGS_PATH)

    print(f"⏱  Embedding took {_fmt_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Step 4: Build and save FAISS index
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4/4 — Building FAISS index")
    print("=" * 60)
    t0 = time.time()

    index, chunks = build_index(chunks)
    save_index(index, chunks, FAISS_INDEX_PATH, CHUNKS_PATH)

    print(f"⏱  Indexing took {_fmt_time(time.time() - t0)}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print("✅  Build complete!")
    print("=" * 60)
    print(f"  Total chunks indexed: {total_chunks:,}")
    print(f"  Total time:           {_fmt_time(total_time)}")
    print(f"  Embeddings saved to:  {EMBEDDINGS_PATH}")
    print(f"  FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"  Chunks saved to:      {CHUNKS_PATH}")
