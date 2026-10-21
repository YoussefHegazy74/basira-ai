"""Regression tests for live-eval rubric false negatives."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _harness():
    path = (
        Path(__file__).resolve().parents[1]
        / "eval_runs/live_2026-10-20_autonomous/run_live_eval.py"
    )
    if not path.is_file():
        pytest.skip("local live-eval harness archive is not present")
    spec = importlib.util.spec_from_file_location("live_eval_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(module, question_id, name, answer):
    rubric = next(
        c for c in module.RUBRICS[question_id] if c["name"] == name
    )
    return module.check_fact(module.nz(answer), rubric)


def test_aqiqa_days_accepts_definite_article_form():
    h = _harness()
    result = _check(
        h,
        "aqiqa_multi",
        "أيام الذبح",
        "تذبح يوم السابع، وإلا فالرابع عشر، وإلا فالواحد والعشرين.",
    )
    assert result["passed"] is True


def test_aqiqa_days_still_rejects_missing_day():
    h = _harness()
    result = _check(
        h,
        "aqiqa_multi",
        "أيام الذبح",
        "تذبح يوم السابع، وإلا فالرابع عشر.",
    )
    assert result["passed"] is False


def test_ghush_ruling_accepts_evidence_backed_wording():
    h = _harness()
    result = _check(
        h,
        "ghush_buyer_option",
        "الغش محرم",
        "حكم الغش في البيع هو الحرمة.",
    )
    assert result["passed"] is True


def test_ghush_ruling_still_rejects_contradiction():
    h = _harness()
    result = _check(
        h,
        "ghush_buyer_option",
        "الغش محرم",
        "الغش في البيع حلال.",
    )
    assert result["passed"] is False


def _metrics_row(h, *, qid, set_name, answerable, answer, retrieved,
                 checks=None, phrase=None, evidence_ids=None):
    from types import SimpleNamespace

    q = {
        "id": qid,
        "set": set_name,
        "answerable": answerable,
        "checks": checks or [],
        "phrase": phrase or "",
        "evidence_ids": evidence_ids or [],
    }
    state = SimpleNamespace(
        retrieved=retrieved,
        raw_answer=answer,
        ground_log="",
    )
    result = {"answer": answer, "sources": []}
    byid = {c["id"]: c for c in retrieved}
    return h.answer_metrics(q, state, result, byid)


def test_heldout_source_phrase_is_not_answer_accuracy():
    h = _harness()
    chunk = {"id": "gold_1", "original_text": "الراحمون يرحمهم الرحمن"}
    row = _metrics_row(
        h,
        qid="heldout_x",
        set_name="heldout",
        answerable=True,
        answer="الراحمون يرحمهم الرحمن (مصدر 1).",
        retrieved=[chunk],
        phrase="الراحمون يرحمهم الرحمن",
    )
    assert row["answer_pass"] is None
    assert row["source_citation_pass"] is True
    assert row["phrase_supported_by_cited_source"] is True


def test_heldout_source_pass_rejects_unsupported_extra_claim():
    h = _harness()
    chunk = {"id": "gold_1", "original_text": "الراحمون يرحمهم الرحمن"}
    row = _metrics_row(
        h,
        qid="heldout_unsupported",
        set_name="heldout",
        answerable=True,
        answer=(
            "الراحمون يرحمهم الرحمن (مصدر 1). "
            "الجنة ثمانية أبواب (مصدر 1)."
        ),
        retrieved=[chunk],
        phrase="الراحمون يرحمهم الرحمن",
    )
    assert row["phrase_supported_by_cited_source"] is True
    assert row["unsupported_final_count"] == 1
    assert row["source_citation_pass"] is False


def test_heldout_metric_requires_cited_chunk():
    h = _harness()
    chunk = {"id": "other", "original_text": "مصدر آخر"}
    row2 = _metrics_row(
        h,
        qid="heldout_y",
        set_name="heldout",
        answerable=True,
        answer="الراحمون يرحمهم الرحمن (مصدر 1).",
        retrieved=[chunk],
        phrase="الراحمون يرحمهم الرحمن",
    )
    assert row2["answer_pass"] is None
    assert row2["source_citation_pass"] is False


def test_unsupported_primary_answer_is_not_a_pass():
    h = _harness()
    chunk = {"id": "other", "original_text": "مصدر آخر لا يذكر الحكم"}
    row = _metrics_row(
        h,
        qid="primary_x",
        set_name="primary",
        answerable=True,
        answer="الحكم كذا وكذا (مصدر 1).",
        retrieved=[chunk],
        checks=[{"name": "الحكم", "any_of": ["الحكم"], "optional": False}],
    )
    assert row["fact_pass"] is True
    assert row["unsupported_final_count"] == 1
    assert row["answer_pass"] is False


def test_manual_ambiguous_answer_is_not_a_false_pass():
    h = _harness()
    chunk = {"id": "c1", "original_text": "السنن الرواتب والوتر"}
    row = _metrics_row(
        h,
        qid="sunan_ambiguous",
        set_name="primary",
        answerable=True,
        answer="السنن الرواتب والوتر (مصدر 1).",
        retrieved=[chunk],
        checks=h.RUBRICS["sunan_ambiguous"],
    )
    assert row["manual_review"] is True
    assert row["answer_pass"] is None


def test_reset_run_outputs_truncates_append_targets(tmp_path):
    h = _harness()
    (tmp_path / "results.jsonl").write_text("old\n", encoding="utf-8")
    (tmp_path / "api_calls.jsonl").write_text("old\n", encoding="utf-8")
    h.reset_run_outputs(tmp_path)
    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "api_calls.jsonl").read_text(encoding="utf-8") == ""


def _aggregate_row(*, qid, set_name, answerable, answer_pass,
                   source_citation_pass=None, manual=False):
    return {
        "question": {
            "id": qid,
            "set": set_name,
            "answerable": answerable,
            "source_types": [],
        },
        "answer": {
            "answer_pass": answer_pass,
            "source_citation_pass": source_citation_pass,
            "manual_review": manual,
            "fact_pass": None,
            "refusal": False,
            "abstention_ok": True,
            "citation_validity": 1.0,
            "citation_support_rate": 1.0,
            "citation_gold_precision": 0.0,
            "citation_gold_recall": 0.0,
            "unsupported_final_count": 0,
            "dropped_sentences": 0,
            "ground_retention": 1.0,
        },
        "retrieval": {
            "in_context": True,
            "context_full": True,
            "context_coverage": 1.0,
            "pool_rr": 1.0,
            "sem_rr": 1.0,
            "pool_recall": {"@1": True},
            "sem_recall": {"@1": True},
            "context_recall": {"@1": True},
        },
        "timing": {"retrieve_s": 1.0, "total_s": 1.0},
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_tokens": 0,
            "cost_usd": 0.0,
        },
    }


def test_aggregate_excludes_heldout_and_manual_from_answer_rate():
    h = _harness()
    rows = [
        _aggregate_row(qid="p_pass", set_name="primary",
                       answerable=True, answer_pass=True),
        _aggregate_row(qid="p_fail", set_name="primary",
                       answerable=True, answer_pass=False),
        _aggregate_row(qid="manual", set_name="primary",
                       answerable=True, answer_pass=None, manual=True),
        _aggregate_row(qid="heldout", set_name="heldout",
                       answerable=True, answer_pass=None,
                       source_citation_pass=True),
        _aggregate_row(qid="unanswerable", set_name="primary",
                       answerable=False, answer_pass=True),
    ]
    agg = h.aggregate(rows)
    assert agg["answerable_scored"] == 2
    assert agg["answer_pass_count"] == 1
    assert agg["answer_pass_rate"] == 0.5
    assert agg["task_scored"] == 3
    assert agg["task_pass_count"] == 2
    assert agg["heldout_source_citation_count"] == 1
    assert agg["heldout_source_citation_rate"] == 1.0
    assert agg["manual_review"] == ["manual"]


def test_refusal_prefix_with_substantive_claim_is_not_refusal():
    h = _harness()
    chunk = {"id": "c1", "original_text": "ليلة القدر في رمضان"}
    row = _metrics_row(
        h,
        qid="unanswerable_mixed",
        set_name="primary",
        answerable=False,
        answer=(
            "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال. "
            "ليلة القدر في رمضان (مصدر 1)."
        ),
        retrieved=[chunk],
    )
    assert row["refusal"] is False
    assert row["abstention_ok"] is False
    assert row["answer_pass"] is False


def test_invalid_citation_cannot_pass_primary_answer():
    h = _harness()
    chunk = {"id": "c1", "original_text": "الحكم كذا وكذا"}
    row = _metrics_row(
        h,
        qid="primary_invalid_citation",
        set_name="primary",
        answerable=True,
        answer="الحكم كذا وكذا (مصدر 1) (مصدر 9).",
        retrieved=[chunk],
        checks=[{"name": "الحكم", "any_of": ["الحكم"], "optional": False}],
    )
    assert row["citation_validity"] == 0.5
    assert row["answer_pass"] is False
