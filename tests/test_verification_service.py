"""Regression tests for the independent verification service.

These tests exercise the production matching and result-building code in
``src/rag/verify_full.py`` and ``src/rag/tafsir.py`` against the real
canonical Quran corpus.  No test mocks the quotation matcher itself; only the
network model is replaced where an interpretation assessment is required.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag import post_analysis as PA
from src.rag import verify_full as V

ROOT = Path(__file__).resolve().parents[1]
QURAN_PATH = ROOT / "data" / "raw" / "quran" / "quran_processed.json"
TAFSIR_PATH = ROOT / "data" / "raw" / "tafsir" / "tafsir_processed.json"

AN_NISA_43 = "النساء:43"
PARTIAL = "ولا تقربوا الصلاة"


@pytest.fixture(scope="module")
def quran_chunks() -> list[dict]:
    records = json.loads(QURAN_PATH.read_text(encoding="utf-8"))
    return [
        {
            "id": record["id"],
            "source": "quran",
            "reference": record["reference"],
            "original_text": record["arabic_text"],
            "metadata": record["metadata"],
        }
        for record in records
    ]


@pytest.fixture(scope="module")
def corpus(quran_chunks: list[dict]) -> list[dict]:
    chunks = list(quran_chunks)
    records = json.loads(TAFSIR_PATH.read_text(encoding="utf-8"))
    for record in records:
        if record.get("reference") == "تفسير ابن كثير 4:43":
            chunks.append({
                "id": record["id"],
                "source": "tafsir",
                "reference": record["reference"],
                "original_text": record["arabic_text"],
                "metadata": record.get("metadata", {}),
            })
            break
    return chunks


def canonical_verse(quran_chunks: list[dict]) -> dict:
    return next(
        chunk for chunk in quran_chunks if chunk["reference"] == AN_NISA_43
    )


class FakeChat:
    def __init__(self, payload: dict | str):
        self.payload = (
            payload if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        self.called = False
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.called = True
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.payload)
            )]
        )


class ExplodingChat:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        raise AssertionError("the model must not be called on this path")


def _verify(text: str, explanation: str = "", corpus=None, client=None):
    return V.verify_full(
        text,
        explanation,
        object(),
        corpus,
        client,
        debug=False,
    )


def test_partial_quotation_returns_full_canonical_verse(corpus, quran_chunks):
    result = _verify(PARTIAL, corpus=corpus)

    verse = canonical_verse(quran_chunks)
    assert result["quotation"]["found"] is True
    assert result["quotation"]["is_partial"] is True
    assert result["quotation"]["is_complete"] is False
    assert result["quotation"]["match_type"] == "partial"
    assert result["quotation"]["reference"] == AN_NISA_43
    assert result["quotation"]["source"] == "quran"
    assert result["quotation"]["surah_name"] == "النساء"
    assert result["quotation"]["surah_number"] == 4
    assert result["quotation"]["ayah_number"] == 43
    assert result["quotation"]["canonical_text"] == verse["original_text"]
    assert result["full_text"] == verse["original_text"]
    assert result["text_status"] == "صحيح"
    assert result["is_truncated"] is True
    assert result["verification_status"] == "partial"
    assert result["source"].startswith("القرآن الكريم")
    assert AN_NISA_43 in result["source"]


@pytest.mark.parametrize(
    "text",
    [
        "وَلَا تَقْرَبُوا الصَّلَاةَ",
        "ولا تقربوا الصلاه",
        "وَلَا تَقْرَبُوا الصَّلَاةَ.",
        "ولا   تقربوا\nالصلاة",
        "ولا تقربوا الصلاة ـــ",
        "قال تعالى: ولا تقربوا الصلاة",
    ],
)
def test_quotation_variants_match_the_same_verse(corpus, text):
    result = _verify(text, corpus=corpus)
    assert result["quotation"]["found"] is True
    assert result["quotation"]["reference"] == AN_NISA_43
    assert result["quotation"]["is_partial"] is True
    assert result["verification_status"] == "partial"


def test_exact_full_verse_is_complete(corpus, quran_chunks):
    verse = canonical_verse(quran_chunks)
    result = _verify(verse["original_text"], corpus=corpus)

    assert result["quotation"]["found"] is True
    assert result["quotation"]["match_type"] == "exact"
    assert result["quotation"]["is_complete"] is True
    assert result["is_truncated"] is False
    assert result["verification_status"] == "verified"
    assert result["full_text"] == verse["original_text"]
    assert result["quotation"]["reference"] == AN_NISA_43


def test_no_interpretation_is_not_reported_as_a_source_absence(corpus):
    result = _verify(PARTIAL, explanation="", corpus=corpus)

    assert result["interpretation"]["provided"] is False
    assert result["interpretation"]["status"] == "not_provided"
    assert result["interpretation"]["label"] == "لم يتم تقديم تفسير لتقييمه"
    assert result["explanation_status"] == "لم يتم تقديم تفسير لتقييمه"
    assert result["explanation_verdict"] == ""
    assert result["text_status"] == "صحيح"
    assert result["quotation"]["found"] is True


def test_no_model_call_is_needed_for_a_quotation_without_explanation(corpus):
    result = _verify(PARTIAL, corpus=corpus, client=ExplodingChat())
    assert result["quotation"]["reference"] == AN_NISA_43


def test_misleading_interpretation_is_separate_from_the_quotation(corpus):
    chat = FakeChat({
        "explanation_status": "مخالف",
        "explanation_verdict": (
            "الآية نهت عن الصلاة في حال السكر، وليست نهياً عن الصلاة نفسها."
        ),
        "correct_explanation": "",
    })
    result = _verify(
        PARTIAL,
        explanation="الآية تدل على أن الصلاة محرمة على المسلمين.",
        corpus=corpus,
        client=chat,
    )

    assert chat.called is True
    assert result["quotation"]["found"] is True
    assert result["quotation"]["reference"] == AN_NISA_43
    assert result["text_status"] == "صحيح"
    assert result["interpretation"]["provided"] is True
    assert result["interpretation"]["status"] == "contradicted"
    assert result["explanation_status"] == "مخالف"
    assert "التفسير غير مدعوم" in result["status_label"]
    assert result["danger_level"] == "عالية"


def test_supported_interpretation_is_reported_as_supported(corpus):
    chat = FakeChat({
        "explanation_status": "موافق",
        "explanation_verdict": "المعنى موافق للسياق.",
        "correct_explanation": "",
    })
    result = _verify(
        PARTIAL,
        explanation="الآية تنهى عن الاقتراب من الصلاة في حال السكر حتى يعلم المصلي ما يقول.",
        corpus=corpus,
        client=chat,
    )

    assert result["interpretation"]["status"] == "supported"
    assert result["explanation_status"] == "موافق"
    assert result["text_status"] == "صحيح"
    assert result["danger_level"] == "متوسطة"  # partial quotation


def test_retrieval_failure_is_an_explicit_error(monkeypatch, corpus):
    def explode(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(V, "_dual_search", explode)
    result = _verify(
        "نص غير قرآني للاختبار",
        corpus=corpus,
        client=ExplodingChat(),
    )

    assert result["verification_status"] == "error"
    assert result["text_status"] == "خطأ في التحقق"
    assert result["retrieval_ok"] is False
    assert result["verification_error"]
    assert result["full_text"] == ""
    assert result["source"] == ""
    assert result["danger_level"] == "غير محسوبة"
    assert result["danger_level_applicable"] is False
    assert "غير موجود" not in result["status_label"]


def test_empty_retrieval_is_inconclusive_not_false(monkeypatch, corpus):
    monkeypatch.setattr(V, "_dual_search", lambda *args, **kwargs: [])
    result = _verify(
        "نص غير موجود في المصادر",
        corpus=corpus,
        client=ExplodingChat(),
    )

    assert result["verification_status"] == "inconclusive"
    assert result["text_status"] == "غير مؤكد"
    assert result["quotation"]["found"] is False
    assert result["source"] == ""
    assert result["full_text"] == ""
    assert result["danger_level_applicable"] is False
    assert "لا يعني بالضرورة" in result["status_explanation"]


def test_fabricated_quotation_is_not_attributed(monkeypatch, corpus):
    monkeypatch.setattr(V, "_dual_search", lambda *args, **kwargs: [])
    text = "واقتربوا إلى الصلاة بخشوع"
    result = _verify(text, corpus=corpus, client=ExplodingChat())

    assert result["quotation"]["found"] is False
    assert result["quotation"]["reference"] == ""
    assert result["quotation"]["canonical_text"] == ""
    assert result["source"] == ""
    assert result["verification_status"] == "inconclusive"


def test_contradiction_requires_retrieved_evidence(monkeypatch, corpus):
    hadith = {
        "id": "h-test",
        "source": "hadith",
        "reference": "صحيح البخاري 1",
        "original_text": "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات",
        "score": 0.92,
    }
    monkeypatch.setattr(
        V, "_dual_search", lambda *args, **kwargs: [hadith],
    )
    chat = FakeChat({
        "text_status": "موضوع",
        "is_truncated": "لا",
        "missing_part": "",
        "full_text": "",
        "source": "صحيح البخاري 1",
        "danger_level": "عالية",
        "explanation_status": "لا يوجد",
        "explanation_verdict": "",
        "correct_explanation": "",
    })
    result = _verify(
        "قال رسول الله إنما الأعمال بالنيات",
        corpus=corpus,
        client=chat,
    )

    assert result["verification_status"] == "contradicted"
    assert result["text_status"] == "موضوع"
    assert result["source"] == ""
    assert result["quotation"]["reference"] == ""
    assert any(
        item["reference"] == "صحيح البخاري 1"
        for item in result["evidence"]
    )


def test_inconclusive_semantic_result_does_not_attribute_a_candidate(monkeypatch, corpus):
    candidate = {
        "id": "q-test",
        "source": "quran",
        "reference": "المؤمنون:2",
        "original_text": "نص قرآني قريب دلالياً",
        "score": 0.44,
    }
    monkeypatch.setattr(
        V, "_dual_search", lambda *args, **kwargs: [candidate],
    )
    chat = FakeChat({
        "text_status": "غير موجود",
        "is_truncated": "لا",
        "missing_part": "",
        "full_text": "",
        "source": "المؤمنون:2",
        "danger_level": "منخفضة",
        "explanation_status": "لا يوجد",
        "explanation_verdict": "",
        "correct_explanation": "",
    })
    result = _verify(
        "نص غير موجود فعلاً في المصادر",
        corpus=corpus,
        client=chat,
    )

    assert result["verification_status"] == "inconclusive"
    assert result["text_status"] == "غير مؤكد"
    assert result["source"] == ""
    assert result["quotation"]["found"] is False
    assert result["quotation"]["reference"] == ""


def test_completeness_and_interpretation_fields_never_contradict(corpus):
    no_explanation = _verify(PARTIAL, corpus=corpus)
    assert no_explanation["quotation"]["is_partial"] is True
    assert no_explanation["is_truncated"] is True
    assert no_explanation["interpretation"]["status"] == "not_provided"
    assert no_explanation["interpretation"]["label"] == (
        "لم يتم تقديم تفسير لتقييمه"
    )

    chat = FakeChat({
        "explanation_status": "مخالف",
        "explanation_verdict": "التفسير لا يوافق السياق.",
        "correct_explanation": "",
    })
    with_interpretation = _verify(
        PARTIAL,
        explanation="تفسير غير صحيح",
        corpus=corpus,
        client=chat,
    )
    assert with_interpretation["text_status"] == "صحيح"
    assert with_interpretation["interpretation"]["status"] == "contradicted"
    assert "اقتباس جزئي" in with_interpretation["status_label"]
    assert "التفسير غير مدعوم" in with_interpretation["status_label"]


def test_ambiguous_fragment_is_not_forced_to_one_verse(corpus):
    result = _verify("الصلاة", corpus=corpus, client=ExplodingChat())
    assert result["quotation"]["ambiguous"] is True
    assert result["quotation"]["found"] is False
    assert result["verification_status"] == "inconclusive"
    assert result["quotation"]["reference"] == ""


# ---------------------------------------------------------------------------
# Audit regressions: ambiguous Quran wording and omitted context
# ---------------------------------------------------------------------------


def test_repeated_quran_phrase_is_not_forced_to_one_verse(corpus):
    result = _verify(
        "فبأي آلاء ربكما تكذبان",
        corpus=corpus,
        client=ExplodingChat(),
    )

    assert result["verification_status"] == "inconclusive"
    assert result["quotation"]["ambiguous"] is True
    assert result["quotation"]["found"] is False
    assert result["quotation"]["reference"] == ""
    assert result["quotation"]["candidates"]


def test_post_omitted_text_identifies_the_surrounding_verse(corpus):
    post = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = V.verify_full(post, None, object(), corpus, None)
    first = result["components"][0]
    omitted = PA._normalize(first["completeness"]["omitted_text"])

    assert "وانتم سكاري" in omitted
    assert "حتي تعلموا" in omitted
    assert "تقربوا الصلاه" not in omitted
