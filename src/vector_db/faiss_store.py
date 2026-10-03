"""
Basira Vector DB — FAISS Store
================================
Builds and queries a FAISS index for semantic search over Arabic text
embeddings using inner-product (cosine) similarity.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Build a FAISS inner-product index from embedded chunks.

    Vectors are L2-normalised so that inner-product equals cosine similarity.

    Args:
        chunks: List of chunk dicts, each containing at minimum:
                'embedding', 'clean_text', 'id', 'source',
                'reference', 'original_text', 'metadata'.

    Returns:
        A (faiss.IndexFlatIP, chunks) tuple.
    """
    embeddings = np.array(
        [chunk["embedding"] for chunk in chunks], dtype=np.float32
    )
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"[faiss_store] Built index: {index.ntotal} vectors, dim={dim}")
    return index, chunks


def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    index_path: str | Path,
    chunks_path: str | Path,
) -> None:
    """Persist a FAISS index and its associated chunks to disk.

    Args:
        index:       The FAISS index to save.
        chunks:      The chunk dicts aligned with the index.
        index_path:  Destination path for the FAISS index file.
        chunks_path: Destination path for the pickled chunks file.
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[faiss_store] Saved index to {index_path}")
    print(f"[faiss_store] Saved {len(chunks):,} chunks to {chunks_path}")


def load_index(
    index_path: str | Path,
    chunks_path: str | Path,
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
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[faiss_store] Loaded index ({index.ntotal} vectors) from {index_path}")
    print(f"[faiss_store] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


def search(
    query_embedding: list[float],
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Search the FAISS index for the most similar chunks.

    The query vector is L2-normalised before search so that scores
    represent cosine similarity.

    Args:
        query_embedding: The query vector as a list of floats.
        index:           The FAISS index to search.
        chunks:          The chunk dicts aligned with the index.
        top_k:           Number of results to return.

    Returns:
        List of dicts with keys: id, source, reference, original_text, score.
    """
    query = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "reference": chunk["reference"],
            "original_text": chunk["original_text"],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    # --- Create 3 sample chunks with random 1536-dim embeddings ---
    np.random.seed(42)
    sample_chunks = []
    for i in range(3):
        vec = np.random.randn(1536).astype(np.float32)
        vec /= np.linalg.norm(vec)  # normalise
        sample_chunks.append({
            "id": f"chunk_{i}",
            "source": f"book_{i}.pdf",
            "reference": f"Page {i + 1}",
            "original_text": f"نص عربي تجريبي رقم {i}",
            "clean_text": f"نص عربي تجريبي رقم {i}",
            "metadata": {"page": i + 1},
            "embedding": vec.tolist(),
        })

    # --- Build index ---
    index, chunks = build_index(sample_chunks)

    # --- Search with a random query ---
    query_vec = np.random.randn(1536).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    results = search(query_vec.tolist(), index, chunks, top_k=2)

    print("\nTop 2 results:")
    for r in results:
        print(f"  id={r['id']}  source={r['source']}  score={r['score']:.4f}")

    print("\nFAISS index working successfully")
