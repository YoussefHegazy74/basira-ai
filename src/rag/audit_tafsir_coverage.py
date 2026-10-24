"""Deterministic full-corpus audit of Quran -> Tafsir coverage and resolution.

The audit is read-only.  It verifies the canonical corpus, the per-ayah tafsir
lookup, and the lexical resolver without calling the language model, then
writes a compact report plus machine-readable failure/warning lists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp.preprocessor import normalize_verse_text
from src.rag.tafsir import resolve_tafsir_reference


DEFAULT_QURAN = ROOT / "data" / "raw" / "quran" / "quran_processed.json"
DEFAULT_TAFSIR = ROOT / "data" / "raw" / "tafsir" / "tafsir_processed.json"
DEFAULT_OUTPUT_ROOT = ROOT / "eval_runs"
AMBIGUITY_MARKER = "أكثر من آية"


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return data


def _ref(record: dict) -> tuple[int, int]:
    metadata = record.get("metadata") or {}
    return int(metadata["surah_number"]), int(metadata["ayah_number"])


def _ref_text(ref: tuple[int, int]) -> str:
    return f"{ref[0]}:{ref[1]}"


def _tafsir_reference(ref: tuple[int, int]) -> str:
    return f"تفسير ابن كثير {ref[0]}:{ref[1]}"


def _build_chunks(quran: list[dict], tafsir: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in quran:
        chunks.append({
            "id": record.get("id", ""),
            "source": "quran",
            "reference": record.get("reference", ""),
            "original_text": record.get("arabic_text", ""),
            "metadata": record.get("metadata") or {},
        })
    for record in tafsir:
        chunks.append({
            "id": record.get("id", ""),
            "source": "tafsir",
            "reference": record.get("reference", ""),
            "original_text": record.get("arabic_text", ""),
            "metadata": record.get("metadata") or {},
        })
    return chunks


def _duplicate_groups(records: Iterable[dict]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        text = record.get("arabic_text", "") or ""
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        groups[digest].append(_ref(record))
    return groups


def _normalize_for_coverage(text: str) -> str:
    return normalize_verse_text(text, expand_dagger_alef=False)


def _coverage(verse_text: str, tafsir_text: str) -> tuple[bool, float]:
    verse = _normalize_for_coverage(verse_text)
    tafsir = _normalize_for_coverage(tafsir_text)
    if not verse or not tafsir:
        return False, 0.0
    exact = verse in tafsir
    verse_tokens = set(verse.split())
    tafsir_tokens = set(tafsir.split())
    ratio = len(verse_tokens & tafsir_tokens) / max(1, len(verse_tokens))
    return exact, ratio


def _issue(
    *,
    severity: str,
    category: str,
    ref: tuple[int, int],
    detail: str,
    resolution_source: str = "",
    match_kind: str = "",
    failure_message: str = "",
    tafsir_id: str = "",
    tafsir_reference: str = "",
    group_size: int = 0,
    verse_exact_quote: bool | None = None,
    token_coverage: float | None = None,
) -> dict:
    return {
        "severity": severity,
        "category": category,
        "surah": ref[0],
        "ayah": ref[1],
        "reference": _ref_text(ref),
        "detail": detail,
        "resolution_source": resolution_source,
        "match_kind": match_kind,
        "failure_message": failure_message,
        "tafsir_id": tafsir_id,
        "tafsir_reference": tafsir_reference,
        "tafsir_group_size": group_size,
        "verse_exact_quote": verse_exact_quote,
        "token_coverage": (
            round(token_coverage, 4) if token_coverage is not None else None
        ),
    }


def audit_corpus(
    quran: list[dict], tafsir: list[dict],
) -> tuple[dict, list[dict], list[dict]]:
    quran_by_ref = {_ref(record): record for record in quran}
    tafsir_by_ref = {_ref(record): record for record in tafsir}
    groups = _duplicate_groups(tafsir)
    group_for_ref: dict[tuple[int, int], tuple[str, int]] = {}
    for digest, refs in groups.items():
        for ref in refs:
            group_for_ref[ref] = (digest, len(refs))

    quran_ids = Counter(record.get("id") for record in quran)
    tafsir_ids = Counter(record.get("id") for record in tafsir)
    duplicate_ids = sum(count - 1 for count in (*quran_ids.values(),
                                                *tafsir_ids.values())
                        if count > 1)

    failures: list[dict] = []
    warnings: list[dict] = []
    chunks = _build_chunks(quran, tafsir)

    explicit_ok = 0
    canonical_ok = 0
    canonical_ambiguous = 0
    canonical_no_match = 0
    exact_quotes = 0
    coverage_values: list[float] = []
    coverage_low = 0

    for ref in sorted(quran_by_ref):
        quran_record = quran_by_ref[ref]
        tafsir_record = tafsir_by_ref.get(ref)
        digest, group_size = group_for_ref.get(ref, ("", 0))
        tafsir_id = (tafsir_record or {}).get("id", "")
        tafsir_reference = (tafsir_record or {}).get("reference", "")
        common = {
            "resolution_source": "",
            "match_kind": "",
            "failure_message": "",
            "tafsir_id": tafsir_id,
            "tafsir_reference": tafsir_reference,
            "group_size": group_size,
        }

        if tafsir_record is None:
            failures.append(_issue(
                severity="fail",
                category="tafsir_missing",
                ref=ref,
                detail="canonical Quran ayah has no tafsir record",
                **common,
            ))
            continue

        if tafsir_reference != _tafsir_reference(ref):
            failures.append(_issue(
                severity="fail",
                category="tafsir_reference_mismatch",
                ref=ref,
                detail=(
                    f"expected {_tafsir_reference(ref)!r}, "
                    f"found {tafsir_reference!r}"
                ),
                **common,
            ))

        tafsir_metadata = tafsir_record.get("metadata") or {}
        if (int(tafsir_metadata.get("surah_number", -1)) != ref[0]
                or int(tafsir_metadata.get("ayah_number", -1)) != ref[1]):
            failures.append(_issue(
                severity="fail",
                category="tafsir_metadata_mismatch",
                ref=ref,
                detail=(
                    "tafsir metadata does not match its reference: "
                    f"{tafsir_metadata.get('surah_number')}:"
                    f"{tafsir_metadata.get('ayah_number')}"
                ),
                **common,
            ))

        tafsir_text = tafsir_record.get("arabic_text", "") or ""
        if not tafsir_text.strip():
            failures.append(_issue(
                severity="fail",
                category="tafsir_empty",
                ref=ref,
                detail="tafsir record has no text",
                **common,
            ))

        explicit = resolve_tafsir_reference(
            _ref_text(ref), None, chunks, None,
        )
        explicit_primary, _explicit_next, explicit_source = explicit
        if explicit_primary is None:
            failures.append(_issue(
                severity="fail",
                category="explicit_reference_unresolved",
                ref=ref,
                detail="explicit surah:ayah reference did not resolve",
                resolution_source=explicit_source,
                match_kind=explicit.match_kind,
                failure_message=explicit.failure_message,
                tafsir_id=tafsir_id,
                tafsir_reference=tafsir_reference,
                group_size=group_size,
            ))
        elif explicit_source != _ref_text(ref):
            failures.append(_issue(
                severity="fail",
                category="explicit_reference_wrong_source",
                ref=ref,
                detail=(
                    f"expected {_ref_text(ref)}, got {explicit_source}"
                ),
                resolution_source=explicit_source,
                match_kind=explicit.match_kind,
                failure_message=explicit.failure_message,
                tafsir_id=tafsir_id,
                tafsir_reference=tafsir_reference,
                group_size=group_size,
            ))
        else:
            explicit_ok += 1

        exact, ratio = _coverage(
            quran_record.get("arabic_text", ""), tafsir_text,
        )
        coverage_values.append(ratio)
        if exact:
            exact_quotes += 1
        if ratio < 0.5:
            coverage_low += 1
            warnings.append(_issue(
                severity="warn",
                category="low_token_overlap",
                ref=ref,
                detail=(
                    "tafsir block shares less than half of the verse tokens; "
                    "review for a genuine corpus gap"
                ),
                resolution_source=explicit_source,
                match_kind=explicit.match_kind,
                tafsir_id=tafsir_id,
                tafsir_reference=tafsir_reference,
                group_size=group_size,
                verse_exact_quote=exact,
                token_coverage=ratio,
            ))

        canonical = resolve_tafsir_reference(
            quran_record.get("arabic_text", ""), None, chunks, None,
        )
        canonical_primary, _canonical_next, canonical_source = canonical
        if canonical_primary is not None:
            if canonical_source == _ref_text(ref):
                canonical_ok += 1
            else:
                failures.append(_issue(
                    severity="fail",
                    category="canonical_wrong_source",
                    ref=ref,
                    detail=(
                        f"canonical verse text resolved to "
                        f"{canonical_source or '<none>'}"
                    ),
                    resolution_source=canonical_source,
                    match_kind=canonical.match_kind,
                    failure_message=canonical.failure_message,
                    tafsir_id=tafsir_id,
                    tafsir_reference=tafsir_reference,
                    group_size=group_size,
                    verse_exact_quote=exact,
                    token_coverage=ratio,
                ))
        elif AMBIGUITY_MARKER in canonical.failure_message:
            canonical_ambiguous += 1
            candidate_refs = {
                _ref_text(_ref(candidate))
                for candidate in canonical.candidates
            }
            if _ref_text(ref) not in candidate_refs:
                failures.append(_issue(
                    severity="fail",
                    category="canonical_ambiguous_without_expected_ref",
                    ref=ref,
                    detail=(
                        "ambiguous canonical text did not include the "
                        "expected ayah among its candidates"
                    ),
                    resolution_source=canonical_source,
                    match_kind=canonical.match_kind,
                    failure_message=canonical.failure_message,
                    tafsir_id=tafsir_id,
                    tafsir_reference=tafsir_reference,
                    group_size=group_size,
                    verse_exact_quote=exact,
                    token_coverage=ratio,
                ))
            else:
                warnings.append(_issue(
                    severity="warn",
                    category="canonical_ambiguous_requires_reference",
                    ref=ref,
                    detail=(
                        "canonical verse text is repeated in the Quran; an "
                        "explicit surah:ayah reference is required"
                    ),
                    resolution_source=canonical_source,
                    match_kind=canonical.match_kind,
                    failure_message=canonical.failure_message,
                    tafsir_id=tafsir_id,
                    tafsir_reference=tafsir_reference,
                    group_size=group_size,
                    verse_exact_quote=exact,
                    token_coverage=ratio,
                ))
        elif canonical.failure_message:
            failures.append(_issue(
                severity="fail",
                category="canonical_resolution_failure",
                ref=ref,
                detail="canonical verse text did not resolve",
                resolution_source=canonical_source,
                match_kind=canonical.match_kind,
                failure_message=canonical.failure_message,
                tafsir_id=tafsir_id,
                tafsir_reference=tafsir_reference,
                group_size=group_size,
                verse_exact_quote=exact,
                token_coverage=ratio,
            ))
        else:
            canonical_no_match += 1
            failures.append(_issue(
                severity="fail",
                category="canonical_no_match",
                ref=ref,
                detail="canonical verse text produced no resolution",
                resolution_source=canonical_source,
                match_kind=canonical.match_kind,
                failure_message=canonical.failure_message,
                tafsir_id=tafsir_id,
                tafsir_reference=tafsir_reference,
                group_size=group_size,
                verse_exact_quote=exact,
                token_coverage=ratio,
            ))

    duplicate_group_sizes = Counter(len(refs) for refs in groups.values())
    duplicate_records = sum(
        len(refs) for refs in groups.values() if len(refs) > 1
    )
    coverage_stats = {}
    if coverage_values:
        ordered = sorted(coverage_values)
        quartiles = statistics.quantiles(ordered, n=4)
        coverage_stats = {
            "min": round(ordered[0], 4),
            "p25": round(quartiles[0], 4),
            "median": round(statistics.median(ordered), 4),
            "p75": round(quartiles[2], 4),
            "max": round(ordered[-1], 4),
            "mean": round(statistics.mean(ordered), 4),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quran_records": len(quran),
        "tafsir_records": len(tafsir),
        "quran_references": len(quran_by_ref),
        "tafsir_references": len(tafsir_by_ref),
        "missing_tafsir": len(set(quran_by_ref) - set(tafsir_by_ref)),
        "extra_tafsir": len(set(tafsir_by_ref) - set(quran_by_ref)),
        "duplicate_ids": duplicate_ids,
        "explicit_reference_resolved": explicit_ok,
        "canonical_resolved": canonical_ok,
        "canonical_ambiguous": canonical_ambiguous,
        "canonical_no_match": canonical_no_match,
        "verse_exact_quote": exact_quotes,
        "token_coverage": coverage_stats,
        "low_token_overlap_lt_0_5": coverage_low,
        "tafsir_unique_texts": len(groups),
        "duplicate_tafsir_groups": sum(
            1 for refs in groups.values() if len(refs) > 1
        ),
        "duplicate_tafsir_records": duplicate_records,
        "duplicate_group_size_distribution": {
            str(size): count
            for size, count in sorted(duplicate_group_sizes.items())
        },
        "failures": len(failures),
        "warnings": len(warnings),
    }
    return summary, failures, warnings


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _report(summary: dict, failures: list[dict], warnings: list[dict]) -> str:
    lines = [
        "# Tafsir full-corpus coverage audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Totals",
        "",
        f"- Quran records: {summary['quran_records']}",
        f"- Tafsir records: {summary['tafsir_records']}",
        f"- Missing tafsir references: {summary['missing_tafsir']}",
        f"- Extra tafsir references: {summary['extra_tafsir']}",
        f"- Duplicate IDs: {summary['duplicate_ids']}",
        f"- Explicit references resolved: "
        f"{summary['explicit_reference_resolved']}/"
        f"{summary['quran_references']}",
        f"- Canonical text resolved: {summary['canonical_resolved']}",
        f"- Canonical text ambiguous (requires reference): "
        f"{summary['canonical_ambiguous']}",
        f"- Canonical text no match: {summary['canonical_no_match']}",
        "",
        "## Tafsir text shape",
        "",
        f"- Unique tafsir texts: {summary['tafsir_unique_texts']}",
        f"- Duplicated tafsir groups: "
        f"{summary['duplicate_tafsir_groups']}",
        f"- Records in duplicated groups: "
        f"{summary['duplicate_tafsir_records']}",
        f"- Exact verse quote present: {summary['verse_exact_quote']}",
        f"- Low token overlap (<0.5): "
        f"{summary['low_token_overlap_lt_0_5']}",
        f"- Token coverage stats: {summary['token_coverage']}",
        "",
        "## Result",
        "",
        f"- Failures: {summary['failures']}",
        f"- Warnings: {summary['warnings']}",
        "",
    ]
    if failures:
        lines.extend([
            "### First failures",
            "",
            "| Reference | Category | Detail |",
            "| --- | --- | --- |",
        ])
        for item in failures[:20]:
            detail = str(item["detail"]).replace("|", "\\|")
            lines.append(
                f"| {item['reference']} | {item['category']} | {detail} |"
            )
        lines.append("")
    if warnings:
        warning_counts = Counter(item["category"] for item in warnings)
        lines.extend([
            "### Warnings by category",
            "",
        ])
        for category, count in sorted(warning_counts.items()):
            lines.append(f"- {category}: {count}")
        lines.append("")
    lines.extend([
        "Machine-readable details: `failures.jsonl`, `warnings.jsonl`.",
        "",
    ])
    return "\n".join(lines)


def _output_dir(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"tafsir_coverage_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"tafsir_coverage_{stamp}_{suffix}"
        suffix += 1
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quran", type=Path, default=DEFAULT_QURAN)
    parser.add_argument("--tafsir", type=Path, default=DEFAULT_TAFSIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    quran = _load_json(args.quran)
    tafsir = _load_json(args.tafsir)
    summary, failures, warnings = audit_corpus(quran, tafsir)

    output = _output_dir(args.output_root, args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output / "failures.jsonl", failures)
    _write_jsonl(output / "warnings.jsonl", warnings)
    (output / "report.md").write_text(
        _report(summary, failures, warnings), encoding="utf-8",
    )

    print(f"audit output: {output}")
    print(
        f"failures={len(failures)} warnings={len(warnings)} "
        f"explicit_ok={summary['explicit_reference_resolved']}/"
        f"{summary['quran_references']} "
        f"ambiguous={summary['canonical_ambiguous']}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
