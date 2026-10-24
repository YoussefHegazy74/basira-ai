"""Full-corpus Tafsir coverage audit regression test."""
from __future__ import annotations

import json
from pathlib import Path

from src.rag.audit_tafsir_coverage import audit_corpus

ROOT = Path(__file__).resolve().parents[1]
QURAN_PATH = ROOT / "data" / "raw" / "quran" / "quran_processed.json"
TAFSIR_PATH = ROOT / "data" / "raw" / "tafsir" / "tafsir_processed.json"


def test_full_corpus_tafsir_coverage_is_complete():
    quran = json.loads(QURAN_PATH.read_text(encoding="utf-8"))
    tafsir = json.loads(TAFSIR_PATH.read_text(encoding="utf-8"))

    summary, failures, warnings = audit_corpus(quran, tafsir)

    assert summary["quran_records"] == 6236
    assert summary["tafsir_records"] == 6236
    assert summary["missing_tafsir"] == 0
    assert summary["extra_tafsir"] == 0
    assert summary["explicit_reference_resolved"] == 6236
    assert summary["canonical_no_match"] == 0
    assert summary["failures"] == 0
    assert failures == []
    assert summary["canonical_ambiguous"] == 280
    assert summary["duplicate_tafsir_records"] > 0
    assert all(item["severity"] == "warn" for item in warnings)
    assert not any(
        item["reference"] == "8:7" for item in failures
    )
