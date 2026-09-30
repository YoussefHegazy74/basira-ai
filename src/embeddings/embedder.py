"""
Basira Embeddings — Embedder
==============================
Generates dense vector embeddings for Arabic text using OpenAI's
text-embedding-3-small model via the OpenAI API.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

EMBEDDING_MODEL = 'text-embedding-3-small'


def get_client() -> OpenAI:
    """Load environment variables from .env and return an OpenAI client.

    The OPENAI_API_KEY must be set in the project's .env file.

    Returns:
        An authenticated OpenAI client.
    """
    load_dotenv()
    client = OpenAI()
    print(f'[embedder] OpenAI client initialised (model: {EMBEDDING_MODEL})')
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
    batch_size: int = 100,
) -> list[dict]:
    """Add an 'embedding' key to each chunk dict using the clean_text field.

    Calls the OpenAI embeddings API in batches for efficiency and shows
    a tqdm progress bar.

    Args:
        chunks:     List of chunk dicts, each containing a 'clean_text' key.
        client:     An authenticated OpenAI client.
        batch_size: Number of texts to encode per API call (max 2048).

    Returns:
        The same list of chunk dicts, each now including an 'embedding' key
        with a list-of-floats value.
    """
    total = len(chunks)
    embedded_count = 0

    for i in tqdm(range(0, total, batch_size), desc='Embedding chunks', unit='batch'):
        batch = chunks[i : i + batch_size]
        texts = [chunk['clean_text'] for chunk in batch]

        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )

        for chunk, item in zip(batch, response.data):
            chunk['embedding'] = item.embedding

        embedded_count += len(batch)

    dim = len(chunks[0]['embedding']) if chunks else 0
    print(f'[embedder] Embedded {embedded_count:,} chunks (dim={dim})')
    return chunks


def save_embeddings(chunks: list[dict], path: str | Path) -> None:
    """Save embedded chunks to a pickle file.

    Args:
        chunks: List of chunk dicts (with 'embedding' key).
        path:   Destination .pkl file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[embedder] Saved {len(chunks):,} embedded chunks to {path}')


def load_embeddings(path: str | Path) -> list[dict]:
    """Load embedded chunks from a pickle file.

    Args:
        path: Path to a .pkl file previously written by save_embeddings.

    Returns:
        List of chunk dicts with embeddings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Embeddings file not found: {path}')
    with open(path, 'rb') as f:
        chunks = pickle.load(f)
    print(f'[embedder] Loaded {len(chunks):,} embedded chunks from {path}')
    return chunks


if __name__ == '__main__':
    client = get_client()
    print('OpenAI client loaded successfully')
    print('Ready to embed — add your API key to .env to start')
    print(f'Embedding model: {EMBEDDING_MODEL}')
