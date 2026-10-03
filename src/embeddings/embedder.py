"""
Basira Embeddings — Embedder
==============================
Generates dense vector embeddings for Arabic text using OpenAI's
text-embedding-3-small model via the OpenAI API.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 8000  # Safe token limit for OpenAI's 8192 cap
DIM = 1536  # Embedding dimension for text-embedding-3-small

# Initialise tiktoken encoding once at module level
_encoding = tiktoken.get_encoding("cl100k_base")


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to fit within a token budget.

    Uses the cl100k_base encoding (same tokeniser family as OpenAI
    embedding models) to count tokens accurately for Arabic text.

    Args:
        text:       The input string.
        max_tokens: Maximum number of tokens to keep.

    Returns:
        The original text if it fits, otherwise a truncated version.
    """
    tokens = _encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return _encoding.decode(tokens)
    return text


def get_client() -> OpenAI:
    """Load environment variables from .env and return an OpenAI client.

    The OPENAI_API_KEY must be set in the project's .env file.

    Returns:
        An authenticated OpenAI client.
    """
    load_dotenv()
    client = OpenAI()
    print(f"[embedder] OpenAI client initialised (model: {EMBEDDING_MODEL})")
    return client


def embed_text(text: str, client: OpenAI) -> list[float]:
    """Embed a single text string and return its vector.

    Args:
        text:   The Arabic text to embed.
        client: An authenticated OpenAI client.

    Returns:
        List of floats representing the embedding vector.
    """
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def embed_chunks(
    chunks: list[dict],
    client: OpenAI,
    batch_size: int = 50,
    start_batch: int = 0,
) -> list[dict]:
    """Add an 'embedding' key to each chunk dict using the clean_text field.

    Calls the OpenAI embeddings API in batches for efficiency and shows
    a tqdm progress bar.  Long texts are truncated via tiktoken to stay
    within the OpenAI 8192-token limit.  If an entire batch fails, every
    chunk in that batch receives a zero-vector embedding and a warning is
    logged so the run can continue.

    Args:
        chunks:      List of chunk dicts, each containing a 'clean_text' key.
        client:      An authenticated OpenAI client.
        batch_size:  Number of texts to encode per API call (default 50).
        start_batch: Batch index to resume from (0 = start from beginning).

    Returns:
        The same list of chunk dicts, each now including an 'embedding' key
        with a list-of-floats value.
    """
    total = len(chunks)
    embedded_count = 0
    skipped_empty = 0
    failed_count = 0
    batch_indices = list(range(0, total, batch_size))
    total_batches = len(batch_indices)

    if start_batch > 0:
        print(f"[embedder] Resuming from batch {start_batch}/{total_batches}")

    for batch_num, i in enumerate(
        tqdm(batch_indices, desc="Embedding chunks", unit="batch",
             initial=start_batch, total=total_batches)
    ):
        # Skip already-completed batches
        if batch_num < start_batch:
            continue

        batch = chunks[i : i + batch_size]

        # Truncate via tiktoken and prepare texts with fallback
        texts = []
        valid_chunks = []
        for chunk in batch:
            text = truncate_to_tokens(chunk.get("clean_text") or "")
            if not text.strip():
                text = truncate_to_tokens(chunk.get("original_text") or "")
            if not text.strip():
                skipped_empty += 1
                chunk["embedding"] = [0.0] * DIM
                continue
            texts.append(text)
            valid_chunks.append(chunk)

        if not texts:
            continue

        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
            )

            for chunk, item in zip(valid_chunks, response.data):
                chunk["embedding"] = item.embedding

            embedded_count += len(valid_chunks)

        except Exception as e:
            # Assign zero vectors so the run can continue
            for chunk in valid_chunks:
                chunk["embedding"] = [0.0] * DIM
                failed_count += 1
                print(
                    f"\n⚠️  Batch {batch_num} failed — chunk '{chunk.get('id', '?')}' "
                    f"got zero vector. Error: {e}"
                )

    dim = len(chunks[0]["embedding"]) if chunks else 0
    print(f"[embedder] Embedded {embedded_count:,} chunks (dim={dim})")
    if skipped_empty:
        print(f"[embedder] Skipped {skipped_empty:,} empty chunks (zero vector)")
    if failed_count:
        print(f"[embedder] Failed {failed_count:,} chunks (zero vector due to API error)")
    return chunks


def save_embeddings(chunks: list[dict], path: str | Path) -> None:
    """Save embedded chunks to a pickle file.

    Args:
        chunks: List of chunk dicts (with 'embedding' key).
        path:   Destination .pkl file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[embedder] Saved {len(chunks):,} embedded chunks to {path}")


def load_embeddings(path: str | Path) -> list[dict]:
    """Load embedded chunks from a pickle file.

    Args:
        path: Path to a .pkl file previously written by save_embeddings.

    Returns:
        List of chunk dicts with embeddings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    print(f"[embedder] Loaded {len(chunks):,} embedded chunks from {path}")
    return chunks


if __name__ == "__main__":
    # Load client
    client = get_client()

    # Embed a single Arabic text
    embedding = embed_text("الدين النصيحة", client)

    # Print results
    print(f"Embedding shape: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print("Embeddings working successfully")
