"""Tests for Basira's website-help routing and product answers."""
from __future__ import annotations

import pytest

from src.rag import pipeline
from src.rag.website_help import (
    WEBSITE_HELP_SOURCE,
    build_website_answer,
    classify_website_intent,
)


class ExplodingChat:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise AssertionError("website help must not call the model")


WEBSITE_QUESTIONS = [
    "ماذا تفعل وكيف تساعدني؟",
    "إيه الفرق بين المساعد المعرفي والتحقق من المعلومات؟",
    "إزاي أستخدم خدمة التحقق من المعلومات؟",
    "إزاي أبحث عن تفسير آية؟",
    "أوصل للمحادثات القديمة منين؟",
    "إيه الخدمات اللي موجودة في الموقع؟",
    "إزاي أغير المظهر؟",
    "عايز أعرف الحد الأقصى لطول الرسالة",
]

RELIGIOUS_QUESTIONS = [
    "ما حكم الصلاة؟",
    "ما تفسير آية الكرسي؟",
    "كيف أصلي؟",
    "هل يجوز الجمع بين الصلاتين؟",
    "ما هي شروط الوضوء؟",
]


@pytest.mark.parametrize("question", WEBSITE_QUESTIONS)
def test_website_questions_are_classified_as_website_help(question):
    intent = classify_website_intent(question)
    assert intent["is_website"] is True
    assert intent["is_mixed"] is False
    assert intent["topics"]


@pytest.mark.parametrize("question", RELIGIOUS_QUESTIONS)
def test_religious_questions_are_not_hijacked(question):
    intent = classify_website_intent(question)
    assert intent["is_website"] is False


def test_mixed_question_is_marked_mixed():
    intent = classify_website_intent(
        "ما حكم الصلاة وكيف أستخدم التحقق من المعلومات؟"
    )
    assert intent["is_website"] is True
    assert intent["is_mixed"] is True


def test_overview_mentions_the_actual_services_and_sources():
    answer = build_website_answer("ماذا تفعل وكيف تساعدني؟")
    assert "بصيرة" in answer
    assert "القرآن" in answer
    assert "الحديث" in answer
    assert "التحقق من المعلومات" in answer
    assert "البحث في التفسير" in answer
    assert "المصادر" in answer


def test_difference_answer_keeps_services_separate():
    answer = build_website_answer(
        "إيه الفرق بين المساعد المعرفي والتحقق من المعلومات؟"
    )
    assert "خدمة حوارية" in answer
    assert "خدمة مستقلة" in answer
    assert "منفصلتان" in answer
    assert "لا تُستخدم إجابة المساعد كدليل" in answer


def test_tafsir_navigation_answer_has_real_steps():
    answer = build_website_answer("إزاي أبحث عن تفسير آية؟")
    assert "الشريط العلوي" in answer
    assert "البحث في التفسير" in answer
    assert "النساء:43" in answer
    assert "ابحث عن التفسير" in answer


def test_history_answer_points_to_the_chat_sidebar():
    answer = build_website_answer("أوصل للمحادثات القديمة منين؟")
    assert "الشريط الجانبي" in answer
    assert "محادثة جديدة" in answer
    assert "البحث في المحادثات" in answer or "بحث" in answer
    assert "تغيير الاسم" in answer


def test_generate_returns_website_answer_without_model_or_retrieval():
    result = pipeline.generate(
        "ماذا تفعل وكيف تساعدني؟",
        None,
        [],
        ExplodingChat(),
    )
    assert result["mode"] == "website_help"
    assert result["sources"] == [WEBSITE_HELP_SOURCE]
    assert "بصيرة" in result["answer"]
    assert "التحقق من المعلومات" in result["answer"]


def test_generate_does_not_hijack_a_religious_question(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_resolve_chat_tafsir",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda *args, **kwargs: [],
    )
    result = pipeline.generate(
        "ما حكم الصلاة؟",
        None,
        [],
        ExplodingChat(),
    )
    assert result["answer"] == pipeline._RELIGIOUS_REFUSAL
    assert result["sources"] == []
    assert result.get("mode") != "website_help"


def test_mixed_question_keeps_both_parts(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "_resolve_chat_tafsir",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda *args, **kwargs: [],
    )
    result = pipeline.generate(
        "ما حكم الصلاة وكيف أستخدم التحقق من المعلومات؟",
        None,
        [],
        ExplodingChat(),
    )
    assert result["mode"] == "mixed"
    assert "التحقق من المعلومات" in result["answer"]
    assert "الجزء الشرعي" in result["answer"]
    assert WEBSITE_HELP_SOURCE in result["sources"]


FIQH_QUESTIONS = [
    "ما حكم الوقف في الإسلام؟",
    "كيف ينعقد الوقف؟ وما الفرق بين الصريح والكناية؟",
    (
        "ما حكم الوقف في الإسلام؟ وضّح كيف ينعقد الوقف وما الفرق بين "
        "الصريح والكناية وفق المصدر المتاح."
    ),
    "ما حكم صيام يوم الجمعة منفردًا؟",
]


@pytest.mark.parametrize("question", FIQH_QUESTIONS)
def test_fiqh_questions_are_not_routed_to_website_help(question):
    intent = classify_website_intent(question)
    assert intent["is_website"] is False
    assert intent["topics"] == []


def test_genuine_chat_navigation_question_still_returns_the_guide():
    question = "كيف أبدأ محادثة جديدة في المساعد المعرفي؟"
    intent = classify_website_intent(question)
    assert intent["is_website"] is True
    result = pipeline.generate(question, None, [], ExplodingChat())
    assert result["mode"] == "website_help"
    assert result["sources"] == [WEBSITE_HELP_SOURCE]
    assert "محادثة جديدة" in result["answer"]
    assert "الشريط الجانبي" in result["answer"]


class FiqhChat:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {
                    "content": (
                        "ينعقد الوقف بالقول أو الفعل الدال عليه، وينقسم "
                        "القول إلى صريح وكناية (مصدر 1)."
                    ),
                })(),
            })()],
        })()


def test_waqf_fiqh_question_uses_corpus_evidence_not_website_guide(monkeypatch):
    chunk = {
        "id": "waqf-endowment",
        "source": "fiqh",
        "reference": "فقه السنة، سيد سابق، الوقف - إنعقاد الوقف",
        "original_text": (
            "ينعقد الوقف بالقول أو الفعل الدال عليه، وينقسم القول إلى "
            "صريح وكناية."
        ),
        "_gate_passed": True,
        "score": 0.9,
    }
    monkeypatch.setattr(
        pipeline, "retrieve", lambda *args, **kwargs: [chunk],
    )
    monkeypatch.setattr(
        pipeline, "_resolve_chat_tafsir",
        lambda *args, **kwargs: (None, None),
    )
    result = pipeline.generate(
        (
            "ما حكم الوقف في الإسلام؟ وضّح كيف ينعقد الوقف وما الفرق بين "
            "الصريح والكناية وفق المصدر المتاح."
        ),
        None,
        [],
        FiqhChat(),
    )

    assert result.get("mode") != "website_help"
    assert "الوقف" in result["answer"]
    assert "صريح وكناية" in result["answer"]
    assert result["sources"] == ["fiqh — فقه السنة، سيد سابق، الوقف - إنعقاد الوقف"]
    assert WEBSITE_HELP_SOURCE not in result["sources"]
