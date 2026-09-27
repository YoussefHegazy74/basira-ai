"""
Basira Data Pipeline — Chunker
================================
Transforms raw records into normalised chunks ready for embedding.
Each chunk pairs the original Arabic text with its preprocessed (clean) form.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from src.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from src.nlp.preprocessor import preprocess


def chunk_record(record: dict) -> dict:
    """Convert a single raw record into a normalised chunk.

    Args:
        record: Dict with keys id, source, reference, arabic_text, metadata.

    Returns:
        Chunk dict with keys:
            - id:            same as input record
            - source:        same as input record
            - reference:     same as input record
            - original_text: the raw arabic_text
            - clean_text:    preprocessed tokens rejoined as a string
            - metadata:      same as input record
    """
    arabic_text = record.get('arabic_text', '')
    processed = preprocess(arabic_text)
    clean_text = ' '.join(processed['tokens_clean'])

    return {
        'id': record['id'],
        'source': record['source'],
        'reference': record['reference'],
        'original_text': arabic_text,
        'clean_text': clean_text,
        'metadata': record.get('metadata', {}),
    }


def chunk_all(records: list[dict]) -> list[dict]:
    """Apply chunk_record to all records, skipping empty arabic_text entries.

    Args:
        records: List of raw record dicts.

    Returns:
        List of chunk dicts (records with missing/empty arabic_text are skipped).
    """
    chunks = []
    skipped = 0

    for record in tqdm(records, desc='Chunking records', unit='rec', miniters=1000):
        arabic_text = record.get('arabic_text')
        if not arabic_text or not arabic_text.strip():
            skipped += 1
            continue
        chunks.append(chunk_record(record))

    print(f'[chunker] Produced {len(chunks):,} chunks ({skipped:,} skipped)')
    return chunks


if __name__ == '__main__':
    import pprint

    sample = {
        'id': 'quran_1_1',
        'source': 'quran',
        'reference': 'الفاتحة:1',
        'arabic_text': 'بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ',
        'metadata': {},
    }

    chunk = chunk_record(sample)

    print('=== Chunker Test ===')
    pprint.pprint(chunk, width=100)
