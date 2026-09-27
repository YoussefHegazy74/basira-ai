"""
Basira Data Pipeline — Loader
==============================
Loads processed Quran and Hadith JSON datasets for the RAG pipeline.
Each record follows the unified schema: id, source, reference, arabic_text, metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


def _load_json(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSON file and return its contents as a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {path}')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'Expected a JSON array in {path}, got {type(data).__name__}')
    return data


def load_quran(path: str | Path = 'data/raw/quran/quran_processed.json') -> list[dict]:
    """Load the processed Quran dataset.

    Args:
        path: Path to quran_processed.json (default: relative project path).

    Returns:
        List of record dicts with keys: id, source, reference, arabic_text, metadata.
    """
    records = _load_json(path)
    print(f'[loader] Loaded {len(records):,} Quran records from {path}')
    return records


def load_hadith(path: str | Path = 'data/raw/hadith/hadith_processed.json') -> list[dict]:
    """Load the processed Hadith dataset.

    Args:
        path: Path to hadith_processed.json (default: relative project path).

    Returns:
        List of record dicts with keys: id, source, reference, arabic_text, metadata.
    """
    records = _load_json(path)
    print(f'[loader] Loaded {len(records):,} Hadith records from {path}')
    return records


def load_all(
    quran_path: str | Path = 'data/raw/quran/quran_processed.json',
    hadith_path: str | Path = 'data/raw/hadith/hadith_processed.json',
) -> list[dict]:
    """Load both Quran and Hadith datasets into a single combined list.

    Args:
        quran_path:  Path to quran_processed.json.
        hadith_path: Path to hadith_processed.json.

    Returns:
        Combined list of all records (Quran first, then Hadith).
    """
    quran = load_quran(quran_path)
    hadith = load_hadith(hadith_path)
    combined = quran + hadith

    # Print summary breakdown
    source_counts = Counter(r.get('source', 'unknown') for r in combined)
    print('\n=== Loading Summary ===')
    for source, count in sorted(source_counts.items()):
        print(f'  {source}: {count:,} records')
    print(f'  {"—" * 25}')
    print(f'  Grand total: {len(combined):,} records\n')

    return combined


if __name__ == '__main__':
    import pprint

    quran_path = 'data/raw/quran/quran_processed.json'
    hadith_path = 'data/raw/hadith/hadith_processed.json'

    all_records = load_all(quran_path, hadith_path)

    # First Quran record
    quran_first = next(r for r in all_records if r['source'] == 'quran')
    print('--- First Quran Record ---')
    pprint.pprint(quran_first, width=100)

    # First Hadith record
    hadith_first = next(r for r in all_records if r['source'] == 'hadith')
    print('\n--- First Hadith Record ---')
    pprint.pprint(hadith_first, width=100)

    print(f'\nTotal records loaded: {len(all_records):,}')
