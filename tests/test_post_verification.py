"""End-to-end tests for complete-post verification.

The source-identity, completeness, grading, and result-assembly code is the
production code in ``src/rag/post_analysis.py``.  Only the model's
interpretive judgment is replaced by a deterministic fake, because that
judgment depends on an external API.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.nlp.preprocessor import normalize
from src.rag import post_analysis as PA
from src.rag import verify_full as V

ROOT = Path(__file__).resolve().parents[1]
QURAN_PATH = ROOT / "data" / "raw" / "quran" / "quran_processed.json"
HADITH_PATH = ROOT / "data" / "raw" / "hadith" / "hadith_processed.json"
TAFSIR_PATH = ROOT / "data" / "raw" / "tafsir" / "tafsir_processed.json"
FIQH_PATH = ROOT / "data" / "raw" / "fiqh" / "fiqh_processed.json"


def _make_chunk(record: dict) -> dict:
    return {
        "id": record.get("id") or f"tmp_{record.get('reference')}",
        "source": record.get("source") or "tafsir",
        "reference": record.get("reference", ""),
        "original_text": record.get("arabic_text") or record.get("original_text", ""),
        "clean_text": normalize(
            record.get("arabic_text") or record.get("original_text", "")
        ),
        "metadata": record.get("metadata", {}),
    }


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    quran = json.loads(QURAN_PATH.read_text(encoding="utf-8"))
    hadith = json.loads(HADITH_PATH.read_text(encoding="utf-8"))
    tafsir = json.loads(TAFSIR_PATH.read_text(encoding="utf-8"))
    fiqh = json.loads(FIQH_PATH.read_text(encoding="utf-8"))

    chunks: list[dict] = []
    for record in quran:
        if record["reference"] in {
            "النساء:43", "البقرة:255", "الإسراء:32",
        }:
            chunks.append(_make_chunk(record))
    for record in hadith:
        if record["reference"] in {
            "صحيح البخاري:1",
            "جامع الترمذي:2119",
            "سنن أبي داود:2202",
            "سنن ابن ماجه:3965",
            "صحيح مسلم:2568",
        }:
            chunks.append(_make_chunk(record))
    for record in tafsir:
        if record["reference"] in {
            "تفسير ابن كثير 4:43",
            "تفسير ابن كثير 4:77",
            "تفسير ابن كثير 9:34",
            "تفسير ابن كثير 5:32",
            "تفسير ابن كثير 44:1",
            "تفسير ابن كثير 17:32",
        }:
            chunks.append(_make_chunk(record))
    for record in fiqh:
        if record["reference"] == (
            "فقه السنة، سيد سابق، الطهارة - النجاسة (1/6)"
        ):
            chunks.append(_make_chunk(record))
    return chunks


class FakeAssessmentClient:
    """Deterministic stand-in for the interpretation-assessment model."""

    def __init__(self, forced: dict | None = None):
        self.forced = forced or {}
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][-1]["content"]
        _, _, payload_text = user.partition("العناصر:\n")
        payload = json.loads(payload_text)
        components = []
        for item in payload:
            claim = item.get("claim_or_explanation", "")
            completeness = item.get("completeness", {})
            status = completeness.get("status", "unknown")
            if status == "complete":
                completeness_assessment = "complete"
            elif status in {"partial_not_material", "partial_material", "altered"}:
                completeness_assessment = "partial_not_material"
            else:
                completeness_assessment = "unknown"

            if item["id"] in self.forced:
                forced = self.forced[item["id"]]
                completeness_assessment = forced.get(
                    "completeness_assessment", completeness_assessment,
                )
                interpretation_status = forced.get(
                    "interpretation_status", "inconclusive",
                )
                argument_status = forced.get(
                    "argument_status", "inconclusive",
                )
                explanation = forced.get("explanation", "")
                correction = forced.get("correction", "")
                severity = forced.get("severity", "medium")
            elif not claim:
                interpretation_status = "not_provided"
                argument_status = "not_assessed"
                explanation = ""
                correction = ""
                severity = "none"
            elif any(marker in claim for marker in (
                "تكفي دون عمل", "محرمة على المسلمين", "يجوز ترك الصلاة",
                "مش على أفعالك", "على نيتك بس", "يقدر يعمل أي حاجة",
                "مش مهم الفعل",
            )):
                completeness_assessment = "partial_material"
                interpretation_status = "contradicted"
                argument_status = "misleading"
                explanation = (
                    "الاقتباس صحيح في نفسه، لكن الاستنتاج لا تدعمه المصادر."
                )
                correction = (
                    "المصدر يثبت معنى أضيق من الاستنتاج المذكور؛ راجع النص "
                    "الكامل والسياق."
                )
                severity = "high"
            elif any(marker in claim for marker in (
                "ثواب العمل", "حال السكر", "النهي متعلق", "ثبت ذلك",
            )):
                interpretation_status = "supported"
                argument_status = "supported"
                explanation = "الشرح والاستنتاج متوافقان مع الدليل المتاح."
                correction = ""
                severity = "low"
            else:
                interpretation_status = "supported"
                argument_status = "supported"
                explanation = "لا يظهر تعارض مع الدليل المتاح."
                correction = ""
                severity = "low"
            components.append({
                "id": item["id"],
                "completeness_assessment": completeness_assessment,
                "interpretation_status": interpretation_status,
                "argument_status": argument_status,
                "explanation": explanation,
                "correction": correction,
                "severity": severity,
            })
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {"components": components}, ensure_ascii=False,
                    )
                )
            )]
        )


def analyze(text: str, corpus, client=None, chunks=None):
    return V.verify_full(
        text,
        None,
        object(),
        chunks if chunks is not None else corpus,
        client or FakeAssessmentClient(),
    )


def component(result, index=0):
    return result["components"][index]


def test_complete_hadith_with_supported_explanation(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى، "
        "فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها فهجرته إلى "
        "ما هاجر إليه». وثواب العمل يتوقف على النية."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert result["overall_status"] == "verified"
    assert first["type"] == "hadith_quote"
    assert first["source_identity"]["status"] == "verified"
    assert first["reference"] == "صحيح البخاري:1"
    assert first["completeness"]["status"] == "complete"
    assert first["authenticity"]["status"] == "sahih_collection"
    assert first["interpretation"]["status"] == "supported"
    assert first["argument"]["status"] == "supported"


def test_genuine_hadith_with_misleading_explanation(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "فالنية وحدها تكفي دون عمل."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "partial"
    assert first["authenticity"]["status"] == "sahih_collection"
    assert first["interpretation"]["status"] == "contradicted"
    assert first["argument"]["status"] == "misleading"
    assert result["overall_status"] == "misleading"
    assert "التفسير" in result["status_explanation"] or "استدلال" in result["status_explanation"]


def test_hadith_excerpt_omits_material_qualification(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_material",
            "interpretation_status": "contradicted",
            "argument_status": "misleading",
            "explanation": "الاستثناء في المتن يقيد الحكم المطلق.",
            "correction": "",
            "severity": "high",
        }
    }
    text = (
        "قال رسول الله ﷺ: «لا تخصوا يوم الجمعة بصيام من بين الأيام». "
        "وهذا يجوز دائمًا."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] == "partial_material"
    assert first["completeness"]["is_material"] is True
    assert "الا" in PA._normalize(first["completeness"]["omitted_text"])
    assert result["overall_status"] == "misleading"


def test_partial_hadith_quotation_that_is_not_misleading(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». "
        "وثواب العمل يتوقف على النية."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["completeness"]["is_material"] is False
    assert first["interpretation"]["status"] == "supported"
    assert result["overall_status"] == "verified"


def test_fabricated_hadith_with_documented_grading(corpus):
    text = (
        "قال رسول الله ﷺ: «الحسنات من الله والسيئات من أنفسنا». "
        "وهذا أصل في أن العبد يخلق أفعاله."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["type"] == "hadith_quote"
    assert first["authenticity"]["status"] == "mawdu"
    assert "موضوعة" in first["authenticity"]["label"]
    assert first["reference"] == "تفسير ابن كثير 4:77"
    assert first["canonical_label"] == "المقطع الذي ورد فيه النص والتقييم"
    assert result["overall_status"] == "misleading"


def test_weak_hadith_is_not_reported_as_fabricated(corpus):
    text = (
        "قال رسول الله ﷺ: «تبا للذهب وتبا للفضة». "
        "وهذا يدل على تحريم اقتناء الذهب مطلقًا."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["authenticity"]["status"] == "daif"
    assert "ضعيفة" in first["authenticity"]["label"]
    assert "موضوعة" not in first["authenticity"]["label"]
    assert result["overall_status"] in {"partially_verified", "misleading"}


def test_absent_hadith_is_not_classified_as_fabricated(corpus):
    text = (
        "قال رسول الله ﷺ: «من قرأ سورة القمر في ليلة الجمعة غفر له». "
        "وهذا حديث صحيح بلا شك."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "not_found"
    assert first["authenticity"]["status"] == "not_found"
    assert "لا يعني" in first["authenticity"]["assessment"]
    assert result["overall_status"] == "inconclusive"


def test_quran_partial_quote_used_misleadingly(corpus):
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["type"] == "quran_quote"
    assert first["source_identity"]["status"] == "partial"
    assert first["completeness"]["status"] == "partial_material"
    assert first["interpretation"]["status"] == "contradicted"
    assert result["overall_status"] == "misleading"
    assert first["canonical_text"]


def test_correct_quran_quote_with_unsupported_conclusion(corpus):
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "والمعنى أن الصلاة محرمة على المسلمين، ويجوز ترك الصلاة."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "partial"
    assert first["interpretation"]["status"] == "contradicted"
    assert result["overall_status"] == "misleading"


def test_valid_quran_excerpt_without_material_omission(corpus):
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "والنهي متعلق بحال السكر حتى يعلم المصلي ما يقول."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["interpretation"]["status"] == "supported"
    assert result["overall_status"] == "verified"


def test_seerah_claim_with_documented_reliability_concern(corpus):
    text = (
        "كان أمير السرية كرز بن جابر الفهري، وقد ثبت ذلك في السيرة."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["type"] == "seerah_claim"
    assert first["source_identity"]["status"] in {"partial", "verified"}
    assert first["authenticity"]["status"] == "daif"
    assert "ضعف" in first["authenticity"]["assessment"]
    assert result["overall_status"] == "partially_verified"


def test_seerah_claim_with_insufficient_data(corpus):
    text = (
        "غزوة القمر وقعت في رمضان من السنة الثالثة، وقد ثبت ذلك."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "not_found"
    assert first["authenticity"]["status"] == "not_found"
    assert "لا يعني" in first["authenticity"]["assessment"]
    assert result["overall_status"] == "inconclusive"


def test_multiple_claims_are_reported_individually(corpus):
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "والنهي متعلق بحال السكر حتى يعلم المصلي ما يقول. "
        "وقال رسول الله ﷺ: «الحسنات من الله والسيئات من أنفسنا». "
        "وهذا أصل في أن العبد يخلق أفعاله."
    )
    result = analyze(text, corpus)

    assert result["component_count"] == 2
    assert component(result, 0)["type"] == "quran_quote"
    assert component(result, 0)["interpretation"]["status"] == "supported"
    assert component(result, 1)["authenticity"]["status"] == "mawdu"
    assert result["overall_status"] == "misleading"


def test_interpretive_disagreement_is_not_forced_into_falsehood(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "disputed",
            "argument_status": "disputed",
            "explanation": "المعنى محل خلاف بين المفسرين.",
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "والمعنى أن الصلاة محرمة على المسلمين."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["interpretation"]["status"] == "disputed"
    assert first["argument"]["status"] == "disputed"
    assert result["overall_status"] == "partially_verified"
    assert "خلاف" in result["status_explanation"]


def test_retrieval_failure_is_an_error_not_falsity(monkeypatch, corpus):
    def explode(*args, **kwargs):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(PA, "_bm25_search", explode)
    result = analyze(
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». وهذا يثبت ما نقول.",
        corpus,
        FakeAssessmentClient(),
    )

    assert result["overall_status"] == "error"
    assert result["retrieval_ok"] is False
    assert result["text_status"] == "خطأ في التحقق"
    assert "غير موجود" not in result["status_label"]


def test_source_backed_correction_includes_reference(corpus):
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["interpretation"]["correction"]
    assert first["reference"] == "النساء:43"
    assert any(
        item.get("reference") == "النساء:43"
        for item in first["evidence"]
    )


def test_authenticity_context_and_interpretation_are_separate(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "فالنية وحدها تكفي دون عمل."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["authenticity"]["status"] == "sahih_collection"
    assert first["completeness"]["status"] == "partial_not_material"
    assert first["interpretation"]["status"] == "contradicted"
    assert result["text_status"] == "صحيح"  # quotation authenticity
    assert result["overall_status"] == "misleading"  # argument


def test_complete_post_routes_through_single_input(corpus):
    text = (
        "منشور للاختبار: قال رسول الله ﷺ: «إنما الأعمال بالنيات». "
        "والنية وحدها تكفي دون عمل. "
        "وهذا هو المطلوب من كل مسلم."
    )
    result = analyze(text, corpus)

    assert result["analysis_mode"] == "post"
    assert result["component_count"] >= 1
    assert result["input_text"] == text
    assert result["overall_status"] == "misleading"


def test_multi_collection_hadith_reports_multiple_references(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "وثواب العمل يتوقف على النية."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "partial"
    assert "أكثر من مصدر" in first["source_identity"]["label"]
    assert len(first.get("additional_references", [])) >= 1
    assert any(
        item.get("role") == "additional_reference"
        for item in first["evidence"]
    )


def test_omitted_qualification_is_flagged_when_argument_is_unsupported(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "contradicted",
            "argument_status": "unsupported",
            "explanation": "الاستنتاج لا يدعمه النص.",
            "correction": "",
            "severity": "high",
        }
    }
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] == "partial_material"
    assert first["completeness"]["is_material"] is True
    assert "قيد" in first["completeness"]["explanation"]


# ---------------------------------------------------------------------------
# Audit regressions: attribution, baseline inheritance, deterministic facts
# ---------------------------------------------------------------------------


def _chunk_by_reference(corpus: list[dict], reference: str) -> dict:
    return next(chunk for chunk in corpus if chunk["reference"] == reference)


def test_external_grading_is_not_borrowed_from_an_adjacent_narration(corpus):
    """The fiqh passage grades the second narration, not the first."""
    first = "ما قطع من البهيمة وهي حية فهو ميتة"
    second = "أحل لنا ميتتان ودمان"

    first_evidence = PA.find_external_grading(first, corpus)
    second_evidence = PA.find_external_grading(second, corpus)

    assert not any(
        item["status"] in {"daif", "mawdu", "munkar"}
        for item in first_evidence
    )
    assert any(item["status"] == "daif" for item in second_evidence)


def test_altered_hadith_does_not_inherit_sahihain_baseline(corpus):
    bukhari = _chunk_by_reference(corpus, "صحيح البخاري:1")
    matn = (bukhari.get("metadata") or {}).get("matn", "").strip()
    text = (
        "قال رسول الله ﷺ: «"
        + matn
        + " وكتب»."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "altered"
    assert first["completeness"]["status"] == "altered"
    assert first["completeness"]["is_material"] is True
    assert first["authenticity"]["status"] == "not_recorded"
    assert "صحيح البخاري" not in first["authenticity"]["label"]
    assert result["overall_status"] == "misleading"


def test_model_cannot_mark_a_partial_quote_complete(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "complete",
            "interpretation_status": "supported",
            "argument_status": "supported",
            "explanation": "لا يوجد نقص.",
            "correction": "",
            "severity": "low",
        }
    }
    text = "قال الله تعالى: «ولا تقربوا الصلاة». والنهي متعلق بحال السكر."
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] != "complete"
    assert first["completeness"]["status"] in {
        "partial_not_material", "partial_material",
    }


def test_model_cannot_mark_an_exact_quote_altered(corpus):
    bukhari = _chunk_by_reference(corpus, "صحيح البخاري:1")
    matn = (bukhari.get("metadata") or {}).get("matn", "").strip()
    forced = {
        "c1": {
            "completeness_assessment": "altered",
            "interpretation_status": "supported",
            "argument_status": "supported",
            "explanation": "النص مختلف.",
            "correction": "",
            "severity": "high",
        }
    }
    result = analyze(
        "قال رسول الله ﷺ: «" + matn + "».",
        corpus,
        FakeAssessmentClient(forced),
    )
    first = component(result)

    assert first["source_identity"]["status"] == "verified"
    assert first["completeness"]["status"] == "complete"
    assert first["completeness"]["is_material"] is False
    assert first["authenticity"]["status"] == "sahih_collection"


def test_query_specific_retrieval_failure_becomes_an_error(monkeypatch, corpus):
    def flaky(tokens, chunks, top_k=20):
        if tokens == ["اختبار"]:
            return []
        raise RuntimeError("query-specific retrieval failure")

    monkeypatch.setattr(PA, "_bm25_search", flaky)
    result = analyze(
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». وهذا يثبت ما نقول.",
        corpus,
        FakeAssessmentClient(),
    )

    assert result["overall_status"] == "error"
    assert result["retrieval_ok"] is False
    assert result["components"] == []
    assert "غير موجود" not in result["status_label"]


def test_multiline_hadith_with_inner_quotes_stays_one_component(corpus):
    bukhari = _chunk_by_reference(corpus, "صحيح البخاري:1")
    matn = (bukhari.get("metadata") or {}).get("matn", "").strip()
    text = "قال رسول الله ﷺ: «" + matn + "»."

    components = PA.extract_components(text)

    assert len(components) == 1
    assert components[0]["type"] == "hadith_quote"
    assert "الاعمال" in PA._normalize(
        components[0]["submitted_excerpt"]
    )
    assert len(components[0]["submitted_excerpt"]) > 100


def test_malformed_model_output_does_not_create_a_misleading_verdict(corpus):
    class MalformedClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="not json")
                )]
            )

    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = analyze(text, corpus, MalformedClient())
    first = component(result)

    assert result["overall_status"] in {
        "partially_verified", "inconclusive",
    }
    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "inconclusive"
    assert first["interpretation"]["correction"] == ""
    assert result["overall_status"] != "misleading"


def test_model_cannot_contradict_a_narration_with_no_matched_source(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "unknown",
            "interpretation_status": "contradicted",
            "argument_status": "misleading",
            "explanation": "لا يوجد دليل لكن الحكم قاطع.",
            "correction": "تصحيح بلا مصدر.",
            "severity": "high",
        }
    }
    text = (
        "قال رسول الله ﷺ: «من قرأ سورة القمر في ليلة الجمعة غفر له». "
        "وهذا أصل عظيم."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["source_identity"]["status"] == "not_found"
    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "inconclusive"
    assert first["interpretation"]["correction"] == ""
    assert result["overall_status"] == "inconclusive"


def test_model_correction_with_unverified_citation_is_dropped(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_material",
            "interpretation_status": "contradicted",
            "argument_status": "misleading",
            "explanation": "الاستنتاج غير مدعوم.",
            "correction": "الصواب في النساء:44 أن الصلاة لا تُترك.",
            "severity": "high",
        }
    }
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذا يعني أن الصلاة محرمة على المسلمين كلهم."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["interpretation"]["status"] == "contradicted"
    assert "النساء:44" not in first["interpretation"]["correction"]
    assert "النساء:43" in first["interpretation"]["correction"]


def test_mid_narration_phrase_shared_across_collections_is_ambiguous():
    records = json.loads(HADITH_PATH.read_text(encoding="utf-8"))
    refs = {
        "سنن أبي داود:1",
        "صحيح البخاري:3",
        "صحيح البخاري:5",
        "صحيح البخاري:40",
    }
    chunks = [
        _make_chunk(record)
        for record in records
        if record["reference"] in refs
    ]

    match = PA.match_hadith_quote(
        "صلى الله عليه وسلم كان", chunks,
    )

    assert match["found"] is False
    assert match["ambiguous"] is True


def test_absent_narration_does_not_inherit_a_nearby_weak_grading(corpus):
    quote = "من قرأ سورة القمر في ليلة الجمعة غفر له"
    evidence = PA.find_external_grading(quote, corpus)

    assert not any(item["status"] == "daif" for item in evidence)

    text = (
        "قال رسول الله ﷺ: «" + quote + "». "
        "وهذا أصل عظيم."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert first["source_identity"]["status"] == "not_found"
    assert first["authenticity"]["status"] == "not_found"
    assert "ضعيفة" not in first["authenticity"]["label"]
    assert result["overall_status"] == "inconclusive"


def test_rhetorical_intro_and_conclusion_stay_with_the_hadith_component(corpus):
    text = (
        "ناس كتير فاهمة الدين غلط!\n\n"
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى».\n\n"
        "الخلاصة: المهم إن قلبك نضيف، وربنا بيحاسبك على نيتك بس، "
        "مش على أفعالك."
    )
    extracted = PA.extract_components(text)

    assert len(extracted) == 1
    assert extracted[0]["type"] == "hadith_quote"
    assert "ناس كتير" in extracted[0]["preface"]
    assert "ناس كتير" not in extracted[0]["claim_text"]
    assert "قلبك نضيف" in extracted[0]["claim_text"]

    result = analyze(text, corpus)
    first = component(result)

    assert result["component_count"] == 1
    assert first["source_identity"]["status"] == "partial"
    assert first["authenticity"]["status"] == "sahih_collection"
    assert first["completeness"]["status"] == "partial_not_material"
    assert first["completeness"]["label"] == "اقتباس ناقص غير مؤثر"
    assert "محرّف" not in first["completeness"]["label"]
    assert first["argument"]["status"] in {"unsupported", "misleading"}
    assert result["overall_status"] == "misleading"
    assert "سياق ناقص" not in result["status_label"]
    assert (
        "غير مدعوم" in result["status_label"]
        or "مضلّل" in result["status_label"]
    )
    assert "صحيح البخاري:1" in first["interpretation"]["correction"]
    assert "الأفعال معًا" not in first["interpretation"]["correction"]
    assert first["evidence"]
    assert any(
        item["reference"] == "صحيح البخاري:1"
        for item in first["evidence"]
    )


def test_disputed_interpretation_requires_actual_disagreement_evidence(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "disputed",
            "argument_status": "disputed",
            "explanation": "المعنى محل خلاف بين المفسرين.",
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». "
        "وثواب العمل يتوقف على النية."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "inconclusive"
    assert "خلاف" not in first["interpretation"]["label"]
    assert "خلاف" not in first["interpretation"]["explanation"]
    assert "متنازع" not in first["interpretation"]["explanation"]
    assert "خلاف" not in result["status_explanation"]
    assert result["overall_status"] == "partially_verified"


def test_unsupported_non_absolute_claim_does_not_mark_partial_quote_material(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": "لا دليل على تحديد مكان النزول.",
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «ولا تقربوا الصلاة». "
        "وهذه الآية نزلت في المدينة."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["completeness"]["is_material"] is False
    assert first["completeness"]["label"] == "اقتباس ناقص غير مؤثر"


def test_every_counted_evidence_entry_has_readable_text(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "فالنية وحدها تكفي دون عمل."
    )
    result = analyze(text, corpus)
    first = component(result)
    evidence = first["evidence"]

    assert len(evidence) == len([
        item for item in evidence if item.get("text", "").strip()
    ])
    assert len(evidence) >= 2
    refs = {item.get("reference") for item in evidence}
    assert "صحيح البخاري:1" in refs
    assert "سنن أبي داود:2202" in refs
    assert all(item.get("text", "").strip() for item in evidence)


def test_non_material_partial_quote_keeps_argument_assessment_independent(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": "الأدلة لا تكفي لتقييم الشرح.",
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "فالنية وحدها تكفي دون عمل."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["completeness"]["is_material"] is False
    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "unsupported"
    assert first["interpretation"]["label"] != first["argument"]["label"]
    assert "سياق ناقص" not in result["status_label"]
    assert "استدلال غير مدعوم" in result["status_label"]
    assert "صحيح البخاري:1" in first["interpretation"]["correction"]


def test_partial_bukhari_quote_supports_narrow_claim_but_not_broad_claim(corpus):
    narrow = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». "
        "الحديث ده بيبين أهمية النية في الأعمال."
    )
    broad = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "إذن، أي شخص نيته كويسة يقدر يعمل أي حاجة، ومش مهم الفعل نفسه "
        "طالما قلبه سليم."
    )

    narrow_result = analyze(narrow, corpus)
    broad_result = analyze(broad, corpus)
    narrow_component = component(narrow_result)
    broad_component = component(broad_result)

    assert narrow_component["completeness"]["status"] == "partial_not_material"
    assert broad_component["completeness"]["status"] == "partial_not_material"
    assert narrow_component["argument"]["status"] == "supported"
    assert broad_component["argument"]["status"] in {
        "unsupported", "misleading",
    }
    assert narrow_result["overall_status"] == "verified"
    assert broad_result["overall_status"] == "misleading"
    assert "الدعوى الضيقة" in narrow_component["completeness"]["materiality_basis"]
    assert "الدعوى أوسع" in broad_component["completeness"]["materiality_basis"]


def test_zina_verse_interpretation_is_inconclusive_not_disputed(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": "الآية تنهى عن الزنى، ولا تثبت تحريماً عاماً لكل تعامل.",
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «وَلَا تَقْرَبُوا الزِّنَىٰ». "
        "إذن، الآية دي دليل على تحريم أي تعامل أو كلام بين رجل وامرأة "
        "في كل الظروف، من غير أي استثناء."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["type"] == "quran_quote"
    assert first["reference"] == "الإسراء:32"
    assert first["source_identity"]["status"] == "partial"
    assert first["authenticity"]["status"] == "not_applicable"
    assert first["completeness"]["status"] == "partial_not_material"
    assert "فَٰحِشَة" in first["completeness"]["omitted_text"]
    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "unsupported"
    assert "خلاف" not in first["interpretation"]["label"]
    assert "خلاف" not in first["interpretation"]["explanation"]
    assert "متنازع" not in first["interpretation"]["explanation"]
    assert "خلاف" not in result["status_explanation"]
    assert result["overall_status"] == "misleading"
    assert any(
        item["reference"] == "تفسير ابن كثير 17:32"
        and item["text"].strip()
        for item in first["evidence"]
    )


def test_unidentified_hadith_claim_is_one_unresolved_claim(corpus):
    text = (
        "الحديث ده موضوع ومكذوب باتفاق العلماء، ومفيش أي خلاف في حكمه.\n\n"
        "لكن المنشور مش كاتب نص الحديث ولا اسم الكتاب ولا مصدر الحكم."
    )
    result = analyze(text, corpus)
    first = component(result)

    assert result["component_count"] == 1
    assert first["type"] == "claim"
    assert first["type_label"] == "حكم على حديث غير محدد"
    assert first["source_identity"]["status"] == "not_found"
    assert first["source_identity"]["label"] == "لا يمكن تحديد الحديث"
    assert first["completeness"]["status"] == "unknown"
    assert first["authenticity"]["status"] == "not_found"
    assert first["authenticity"]["label"] == "لا يمكن تقييم درجة الحديث"
    assert "لا يعني أنه موضوع" in first["authenticity"]["assessment"]
    assert first["interpretation"]["status"] == "inconclusive"
    assert first["argument"]["status"] == "inconclusive"
    assert first["evidence"] == []
    assert "نص الحديث" in first["missing_information"]
    assert "اسم الكتاب أو المصدر" in first["missing_information"]
    assert "رقم الحديث أو المرجع" in first["missing_information"]
    assert result["overall_status"] == "inconclusive"
    assert "لا يمكن تحديد الحديث" in result["status_label"]
    assert result["danger_level"] == "غير محسوبة"
    assert result["danger_level_applicable"] is False
    assert "موضوع" not in result["text_status"]


def test_single_general_hadith_authenticity_claim_is_not_a_quotation(corpus):
    result = analyze("الحديث ده صحيح باتفاق العلماء.", corpus)
    first = component(result)

    assert result["component_count"] == 1
    assert first["type"] == "claim"
    assert first["source_identity"]["label"] == "لا يمكن تحديد الحديث"
    assert first["authenticity"]["status"] == "not_found"
    assert result["overall_status"] == "inconclusive"


def test_hadith_claim_without_book_reference_is_unresolved(corpus):
    result = analyze(
        "الحديث رقم 5 موضوع، ومفيش مصدر مذكور في المنشور.",
        corpus,
    )
    first = component(result)

    assert result["component_count"] == 1
    assert first["type"] == "claim"
    assert first["source_identity"]["label"] == "لا يمكن تحديد الحديث"
    assert first["authenticity"]["status"] == "not_found"
    assert "مصدر" in " ".join(first["missing_information"])


def test_narrow_supported_interpretation_promotes_argument_without_model_guess(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "supported",
            "argument_status": "inconclusive",
            "explanation": "الشرح الضيق متوافق مع النص.",
            "correction": "",
            "severity": "low",
        }
    }
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات». "
        "الحديث ده بيبين أهمية النية في الأعمال."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["interpretation"]["status"] == "supported"
    assert first["argument"]["status"] == "supported"
    assert result["overall_status"] == "verified"


def test_model_dispute_claim_is_removed_when_evidence_does_not_establish_it(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": (
                "التفسير الذي يربط الآية بتحريم أي تعامل بين رجل وامرأة "
                "يعتبر تفسيراً متنازعاً عليه."
            ),
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «وَلَا تَقْرَبُوا الزِّنَىٰ». "
        "إذن، الآية دي دليل على تحريم أي تعامل أو كلام بين رجل وامرأة "
        "في كل الظروف، من غير أي استثناء."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert first["interpretation"]["status"] == "inconclusive"
    assert first["interpretation"]["label"] == "غير محسوم — الأدلة غير كافية"
    assert "متنازع" not in first["interpretation"]["explanation"]
    assert "خلاف" not in first["interpretation"]["explanation"]
    assert "خلاف" not in result["explanation_verdict"]
    assert "خلاف" not in result["components"][0]["interpretation"]["explanation"]
    assert "خلاف" not in result["status_explanation"]


def test_partial_quote_materiality_text_does_not_contradict_status(corpus):
    text = (
        "قال رسول الله ﷺ: «إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى». "
        "إذن أي شخص نيته كويسة يقدر يعمل أي حاجة، ومش مهم الفعل نفسه "
        "طالما قلبه سليم."
    )
    result = analyze(text, corpus)
    first = component(result)
    explanation = first["completeness"]["explanation"]

    assert first["completeness"]["status"] == "partial_not_material"
    assert first["completeness"]["is_material"] is False
    assert "ناقص نصًّا" in explanation
    assert "النص الكامل" in explanation
    assert "يؤثر على فهم المعنى الكامل" not in explanation
    assert "الدعوى أوسع" in first["completeness"]["materiality_basis"]


def test_unlocated_claim_about_other_texts_is_not_repeated(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": (
                "هناك نصوص أخرى تستثني بعض الحالات من هذا التحريم العام."
            ),
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «وَلَا تَقْرَبُوا الزِّنَىٰ». "
        "إذن الآية دي دليل على تحريم أي تعامل أو كلام بين رجل وامرأة "
        "في كل الظروف، من غير أي استثناء."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert "نصوص أخرى" not in first["argument"]["explanation"]
    assert "نصوص أخرى" not in first["interpretation"]["explanation"]
    assert "الأدلة المستردة لا تحدد" in first["argument"]["explanation"]
    assert first["argument"]["status"] == "unsupported"


def test_argument_must_not_claim_unretrieved_texts_establish_exceptions(corpus):
    forced = {
        "c1": {
            "completeness_assessment": "partial_not_material",
            "interpretation_status": "inconclusive",
            "argument_status": "unsupported",
            "explanation": (
                "الاستدلال على تحريم أي تعامل بين الجنسين غير مدعوم "
                "بالأدلة، حيث أن النصوص الأخرى تشير إلى وجود استثناءات."
            ),
            "correction": "",
            "severity": "medium",
        }
    }
    text = (
        "قال الله تعالى: «وَلَا تَقْرَبُوا الزِّنَىٰ». "
        "إذن الآية دي دليل على تحريم أي تعامل أو كلام بين رجل وامرأة "
        "في كل الظروف، من غير أي استثناء."
    )
    result = analyze(text, corpus, FakeAssessmentClient(forced))
    first = component(result)

    assert "النصوص الأخرى" not in first["argument"]["explanation"]
    assert "استثناء" in first["argument"]["explanation"]
    assert "لا تحدد استثناءات" in first["argument"]["explanation"]
