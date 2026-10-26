"""Regression tests for context-aware chatbot follow-ups."""
from __future__ import annotations

import re

import pytest

from src.nlp.preprocessor import normalize
from src.rag import pipeline as P
from src.vector_db.faiss_store import build_index

FASTING_FIQH = (
    "لا يجوز تخصيص يوم الجمعة بصيام منفرد إلا إذا صام قبله أو بعده."
)
FASTING_HADITH = (
    "لا تخصوا يوم الجمعة بصيام من بين الأيام إلا أن يكون في صوم "
    "يصومه أحدكم."
)
WAQF = "لا يصح الوقف إلا على معين أو جهة بر."
GHASB = "الدفاع عن المال المغصوب جائز، والغصب حرام."


def _chunk(cid: str, source: str, reference: str, text: str, vec):
    return {
        "id": cid,
        "source": source,
        "reference": reference,
        "original_text": text,
        "clean_text": normalize(text),
        "metadata": {},
        "embedding": vec,
    }


def _corpus():
    chunks = [
        _chunk(
            "f1", "fiqh",
            "فقه السنة، سيد سابق، الصيام - الأيام المنهي عن صيامها (1/3)",
            FASTING_FIQH, [1.0, 0.0, 0.0],
        ),
        _chunk(
            "h1", "hadith", "صحيح مسلم:2568",
            FASTING_HADITH, [0.9, 0.1, 0.0],
        ),
        _chunk(
            "u1", "fiqh",
            "فقه السنة، سيد سابق، الوقف - ما يصح وقفه وما لا يصح",
            WAQF, [0.0, 1.0, 0.0],
        ),
        _chunk(
            "u2", "fiqh",
            "فقه السنة، سيد سابق، الغصب - الدفاع عن المال",
            GHASB, [0.0, 1.0, 0.0],
        ),
    ]
    return build_index(chunks)


class FakeChat:
    def __init__(self):
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][-1]["content"]
        match = re.search(
            r"\[مصدر (\d+)\][^\n]*\n([^\n]+)", prompt,
        )
        if match:
            index = int(match.group(1))
            text = match.group(2).strip()
            if "الجمعة" in text or "الوقف" in text:
                return type("Response", (), {
                    "choices": [type("Choice", (), {
                        "message": type("Message", (), {
                            "content": f"{text} (مصدر {index}).",
                        })(),
                    })()],
                })()
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {
                    "content": "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.",
                })(),
            })()],
        })()


@pytest.fixture()
def corpus(monkeypatch):
    index, chunks = _corpus()

    def fake_embed(text, client):
        n = normalize(text)
        if any(word in n for word in ("صيام", "صوم", "الجمعه", "اصومه")):
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(P, "embed_text", fake_embed)
    monkeypatch.setattr(P, "_rewrite_query", lambda query, client: [])
    monkeypatch.setattr(
        P, "_resolve_chat_tafsir", lambda *args, **kwargs: (None, None),
    )
    return index, chunks


def test_exact_two_turn_followup_uses_fasting_context(corpus):
    index, chunks = corpus
    client = FakeChat()
    q1 = "ما حكم صيام يوم الجمعة"
    q2 = "يعني مينفعش أصومه لوحده؟"

    assert P._is_followup_query(q1) is False
    assert P._is_followup_query(q2) is True
    assert P._history_aware_query(q2, [{"role": "user", "content": q1}]) == q1

    turn1 = P.generate(q1, index, chunks, client)
    assert "لم أجد في المصادر" not in turn1["answer"]
    assert any("الصيام" in source for source in turn1["sources"])

    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": turn1["answer"]},
    ]
    turn2 = P.generate(q2, index, chunks, client, chat_history=history)

    assert "لم أجد في المصادر" not in turn2["answer"]
    assert "الجمعة" in turn2["answer"] or "صيام" in turn2["answer"]
    assert turn2["sources"]
    assert all(
        "الوقف" not in source and "الغصب" not in source
        for source in turn2["sources"]
    )


def test_unrelated_followup_results_are_retried_with_context(monkeypatch, corpus):
    index, chunks = corpus
    calls = []

    unrelated = [
        {**chunks[2], "_gate_passed": True, "score": 0.9},
        {**chunks[3], "_gate_passed": True, "score": 0.8},
    ]
    relevant = [
        {**chunks[0], "_gate_passed": True, "score": 0.9},
        {**chunks[1], "_gate_passed": True, "score": 0.8},
    ]

    def fake_retrieve(query, index, chunks, client, **kwargs):
        calls.append(kwargs.get("retrieval_query", query))
        return unrelated if len(calls) == 1 else relevant

    monkeypatch.setattr(P, "retrieve", fake_retrieve)
    history = [
        {"role": "user", "content": "ما حكم صيام يوم الجمعة"},
        {"role": "assistant", "content": FASTING_FIQH + " (مصدر 1)."},
    ]
    result = P.generate(
        "يعني مينفعش أصومه لوحده؟",
        index, chunks, FakeChat(), chat_history=history,
    )

    assert len(calls) == 2
    assert "صيام" in calls[1] or "الجمعة" in calls[1]
    assert result["sources"]
    assert all(
        "الوقف" not in source and "الغصب" not in source
        for source in result["sources"]
    )


def test_unrelated_followup_results_end_in_insufficient_evidence(monkeypatch, corpus):
    index, chunks = corpus
    unrelated = [
        {**chunks[2], "_gate_passed": True, "score": 0.9},
        {**chunks[3], "_gate_passed": True, "score": 0.8},
    ]
    monkeypatch.setattr(
        P, "retrieve", lambda *args, **kwargs: unrelated,
    )
    history = [
        {"role": "user", "content": "ما حكم صيام يوم الجمعة"},
        {"role": "assistant", "content": FASTING_FIQH + " (مصدر 1)."},
    ]
    result = P.generate(
        "يعني مينفعش أصومه لوحده؟",
        index, chunks, FakeChat(), chat_history=history,
    )

    assert result["answer"] == P._RELIGIOUS_REFUSAL
    assert result["sources"] == []


def test_standalone_religious_question_is_not_rewritten_as_followup(corpus):
    index, chunks = corpus
    q = "ما حكم صيام يوم الجمعة"

    assert P._is_followup_query(q) is False
    assert P._history_aware_query(
        q, [{"role": "user", "content": "ما حكم الوقف؟"}],
    ) == q

    result = P.generate(q, index, chunks, FakeChat())
    assert "لم أجد في المصادر" not in result["answer"]
    assert any("الصيام" in source for source in result["sources"])


def test_non_fasting_followup_is_not_blocked_by_the_strict_day_guard(corpus):
    index, chunks = corpus
    history = [
        {"role": "user", "content": "ما حكم الوقف؟"},
        {"role": "assistant", "content": WAQF + " (مصدر 1)."},
    ]
    result = P.generate(
        "وما شروطه؟",
        index, chunks, FakeChat(), chat_history=history,
    )

    assert P._followup_required_evidence_keys("ما حكم الوقف؟") == set()
    assert result["answer"] != P._RELIGIOUS_REFUSAL
    assert result["sources"]


class ExplodingChat:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise AssertionError("ambiguous follow-up must not call the model")


def test_deictic_action_followup_without_identified_action_asks_clarification():
    history = [
        {"role": "user", "content": "ما حكم الوقف في الإسلام؟"},
        {
            "role": "assistant",
            "content": "الوقف هو حبس الأصل وتسبيل الثمرة. (مصدر 1).",
        },
    ]
    result = P.generate(
        "ما حكمه لو عملت كده؟",
        None, [], ExplodingChat(), chat_history=history,
    )

    assert result["answer"] == P._AMBIGUOUS_ACTION_CLARIFICATION
    assert result["sources"] == []
    assert "كده" in result["answer"]
    assert "الوقف" not in result["answer"]


def test_deictic_action_followup_does_not_invent_waqf_answer(corpus):
    index, chunks = corpus
    history = [
        {"role": "user", "content": "ما حكم الوقف في الإسلام؟"},
        {
            "role": "assistant",
            "content": "الوقف هو حبس الأصل وتسبيل الثمرة. (مصدر 1).",
        },
    ]
    result = P.generate(
        "ما حكمه لو عملت كده؟",
        index, chunks, ExplodingChat(), chat_history=history,
    )

    assert result["answer"] == P._AMBIGUOUS_ACTION_CLARIFICATION
    assert result["sources"] == []


QURAN = {
    "id": "quran_1_2",
    "source": "quran",
    "reference": "الفاتحة:2",
    "original_text": "الحمد لله رب العالمين",
}
TAFSIR = {
    "id": "tafsir_ibnkathir_1_2",
    "source": "tafsir",
    "reference": "تفسير ابن كثير 1:2",
    "original_text": "الحمد لله ثناء أثنى به على نفسه",
}


HADITH_NAWI = (
    "قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: "
    "\"إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى، "
    "فَمَنْ كَانَتْ هِجْرَتُهُ إِلَى دُنْيَا يُصِيبُهَا فَهِجْرَتُهُ إِلَى مَا هَاجَرَ إِلَيْهِ\""
)


class RefusalChat:
    def __init__(self):
        self.chat = self
        self.completions = self
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {
                    "content": "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.",
                })(),
            })()],
        })()


def test_hadith_text_request_recovers_verbatim_source_after_false_refusal(monkeypatch):
    hadith = {
        "id": "h-nawawi",
        "source": "hadith",
        "reference": "صحيح البخاري:1",
        "original_text": HADITH_NAWI,
        "clean_text": normalize(HADITH_NAWI),
        "metadata": {"matn": HADITH_NAWI},
        "_gate_passed": True,
        "score": 0.9,
    }

    def fake_retrieve(*args, **kwargs):
        return [hadith]

    monkeypatch.setattr(P, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        P, "_resolve_chat_tafsir", lambda *args, **kwargs: (None, None),
    )
    result = P.generate(
        "ما حديث «إنما الأعمال بالنيات»؟ اذكر نص الحديث والمصدر المتاح "
        "في قاعدة البيانات، ووضّح دلالته الأساسية.",
        None, [], RefusalChat(),
    )

    assert "لم أجد في المصادر" not in result["answer"]
    assert "نص الحديث" in result["answer"]
    assert normalize("إنما الأعمال بالنيات") in normalize(result["answer"])
    assert "صحيح البخاري:1" in result["answer"]
    assert "الدلالة الأساسية" in result["answer"]
    assert any("صحيح البخاري:1" in source for source in result["sources"])


def test_quote_only_verse_sentence_is_supported_by_verbatim_citation():
    verse = "فَإِنَّ مَعَ الْعُسْرِ يُسْرًا"
    retrieved = [{
        "id": "q-94-6",
        "source": "quran",
        "reference": "الشرح:6",
        "original_text": verse,
        "metadata": {"surah_name": "الشرح", "ayah_number": 6},
    }]
    sentence = (
        f'قال الله تعالى: "{verse}" (الشرح: 6) (مصدر 1).'
    )
    grounded = P._ground_answer(sentence, retrieved)
    assert "لم أجد في المصادر" not in grounded
    assert verse in grounded


def test_seerah_event_outline_orders_retrieved_sections_and_cites_them():
    chunks = [
        {
            "id": "sira-20",
            "source": "sira",
            "reference": "سيرة ابن هشام، هجرة الرسول صلى الله عليه وسلم (20/30)",
            "original_text": (
                "(بِنَاءُ مَسْجِدِ الْمَدِينَةِ): "
                "فَأَمَرَ بِهِ رَسُولُ اللَّهِ أَنْ يُبْنَى مَسْجِدًا، "
                "وَنَزَلَ عَلَى أَبِي أَيُّوبَ حَتَّى بَنَى مَسْجِدَهُ."
            ),
        },
        {
            "id": "sira-1",
            "source": "sira",
            "reference": "سيرة ابن هشام، هجرة الرسول صلى الله عليه وسلم (1/30)",
            "original_text": (
                "(تَأَخُّرُ عَلِيٍّ وَأَبِي بَكْرٍ فِي الْهِجْرَةِ): "
                "وَأَقَامَ رَسُولُ اللَّهِ بِمَكَّةَ بَعْدَ أَصْحَابِهِ "
                "يَنْتَظِرُ أَنْ يُؤْذَنَ لَهُ فِي الْهِجْرَةِ. "
                "(اجْتِمَاعُ الْمَلَأِ مِنْ قُرَيْشٍ): "
                "وَلَمَّا رَأَتْ قُرَيْشٌ أَنَّ رَسُولَ اللَّهِ قَدْ "
                "صَارَتْ لَهُ شِيعَةٌ اجْتَمَعُوا فِي دَارِ النَّدْوَةِ."
            ),
        },
        {
            "id": "sira-23",
            "source": "sira",
            "reference": "سيرة ابن هشام، هجرة الرسول صلى الله عليه وسلم (23/30)",
            "original_text": (
                "(تَلَاحُقُ الْمُهَاجِرِينَ): "
                "وَتَلَاحَقَ الْمُهَاجِرُونَ إلَى رَسُولِ اللَّهِ، "
                "فَلَمْ يَبْقَ بِمَكَّةَ مِنْهُمْ أَحَدٌ."
            ),
        },
    ]
    outline = P._seerah_event_outline(
        "ما أهم أحداث الهجرة النبوية من مكة إلى المدينة؟ "
        "رتّب الأحداث واشرحها اعتمادًا على مصادر السيرة المتاحة.",
        chunks,
    )
    assert outline
    lines = outline.splitlines()
    event_lines = [line for line in lines if line[:1].isdigit()]
    assert len(event_lines) == 4
    assert "تَأَخُّرُ عَلِيٍّ" in event_lines[0]
    assert "اجْتِمَاعُ الْمَلَأِ مِنْ قُرَيْشٍ" in event_lines[1]
    assert "بِنَاءُ مَسْجِدِ الْمَدِينَةِ" in event_lines[2]
    assert "تَلَاحُقُ الْمُهَاجِرِينَ" in event_lines[3]
    assert all("(مصدر" in line for line in event_lines)
    assert "المكوث في الغار" in outline
    assert "القدوم إلى قباء" in outline
    assert "لا تُذكر هنا" in outline


def test_fasting_detail_appends_full_source_qualification():
    source = (
        "يوم الجمعة عيد أسبوعي للمسلمين، ولذلك نهى الشارع عن صيامه. "
        "وذهب الجمهور: إلى أن النهي للكراهة لا للتحريم إلا إذا صام يوما "
        "قبله، أو يوما بعده، أو وافق عادة له، أو كان يوم عرفة، أو عاشوراء. "
        "فإنه حينئذ لا يكره صيامه."
    )
    chunk = {
        "id": "fasting-fiqh",
        "source": "fiqh",
        "reference": "فقه السنة، سيد سابق، الصيام - الأيام المنهي عن صيامها (1/3)",
        "original_text": source,
    }
    query = (
        "ما حكم صيام يوم الجمعة منفردًا؟ اذكر الأدلة والتفصيل الموجود "
        "في المصادر، ولا تختزل المسألة لو كان فيها تفصيل."
    )
    answer = (
        'جاء في الحديث: "لا تصوموا يوم الجمعة، إلا وقبله يوم، أو بعده يوم" '
        "(مصدر 1)."
    )
    augmented = P._augment_answer_completeness(query, answer, [chunk])

    assert "وذهب الجمهور" in augmented
    assert "وافق عادة له" in augmented
    assert "عرفة" in augmented
    assert "عاشوراء" in augmented


class QuoteOnlyChat:
    def __init__(self, quote: str):
        self.quote = quote
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {
                    "content": f'يقول الحديث: "{self.quote}" (مصدر 1).',
                })(),
            })()],
        })()


def test_hadith_explanation_request_adds_explanation_and_labelled_example(monkeypatch):
    hadith = {
        "id": "h-nawawi-explain",
        "source": "hadith",
        "reference": "صحيح البخاري:1",
        "original_text": HADITH_NAWI,
        "clean_text": normalize(HADITH_NAWI),
        "metadata": {"matn": HADITH_NAWI},
        "_gate_passed": True,
        "score": 0.9,
    }

    def fake_retrieve(*args, **kwargs):
        return [hadith]

    monkeypatch.setattr(P, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        P, "_resolve_chat_tafsir", lambda *args, **kwargs: (None, None),
    )
    result = P.generate(
        "اشرح حديث إنما الأعمال بالنيات، واذكر مثالًا يوضح معنى النية "
        "دون أن تنسب المثال إلى الحديث.",
        None, [], QuoteOnlyChat("إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى"),
    )

    answer = result["answer"]
    assert "الشرح" in answer
    assert normalize("النية") in normalize(answer)
    assert "مثال توضيحي" in answer
    assert "ليس من نص الحديث" in answer
    assert "(مصدر" in answer


def test_tafsir_meaning_intent_accepts_ma_almaqsud(monkeypatch):
    from src.rag.tafsir import TafsirResolution
    from src.rag import tafsir as tafsir_module

    monkeypatch.setattr(
        tafsir_module, "resolve_tafsir_reference",
        lambda *args, **kwargs: TafsirResolution(
            TAFSIR, None, "1:2", quran_chunk=QURAN, match_kind="exact",
        ),
    )
    anchors, clarification = P._resolve_chat_tafsir(
        "ما المقصود بقوله تعالى «الحمد لله رب العالمين»؟",
        None, [], None,
    )
    assert clarification == ""
    assert [c["id"] for c in anchors] == ["quran_1_2", "tafsir_ibnkathir_1_2"]


def test_hadith_explanation_filters_unrelated_tafsir_context(monkeypatch):
    hadith = {
        "id": "h-nawawi-filter",
        "source": "hadith",
        "reference": "صحيح البخاري:1",
        "original_text": HADITH_NAWI,
        "clean_text": normalize(HADITH_NAWI),
        "metadata": {"matn": HADITH_NAWI},
        "_gate_passed": True,
    }
    unrelated = {
        "id": "tafsir-unrelated",
        "source": "tafsir",
        "reference": "تفسير ابن كثير 17:13",
        "original_text": "كلام في آية أخرى لا يشرح الحديث.",
        "_gate_passed": True,
    }

    monkeypatch.setattr(
        P, "retrieve", lambda *args, **kwargs: [hadith, unrelated],
    )
    monkeypatch.setattr(
        P, "_resolve_chat_tafsir", lambda *args, **kwargs: (None, None),
    )
    result = P.generate(
        "اشرح حديث إنما الأعمال بالنيات، واذكر مثالًا يوضح معنى النية.",
        None, [], QuoteOnlyChat("إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى"),
    )

    assert result["sources"] == ["hadith — صحيح البخاري:1"]


def test_requested_distinction_adds_both_source_terms():
    chunk = {
        "id": "waqf-endowment-distinction",
        "source": "fiqh",
        "reference": "فقه السنة، سيد سابق، الوقف - إنعقاد الوقف",
        "original_text": (
            "فالصريح: مثل قول الواقف: وقفت وحبست وسبلت وأبدت. "
            "والكناية: كأن يقول: تصدقت ناويا به الوقف."
        ),
    }
    query = "كيف ينعقد الوقف؟ وما الفرق بين الصريح والكناية؟"
    answer = "الوقف ينعقد بالقول أو الفعل (مصدر 1)."
    augmented = P._augment_answer_completeness(query, answer, [chunk])

    assert "الصريح" in augmented
    assert "الكناية" in augmented
    assert "(مصدر 1)" in augmented
