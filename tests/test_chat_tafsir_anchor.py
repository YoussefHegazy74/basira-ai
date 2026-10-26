"""Chat-side Tafsir intent resolution and anchor injection."""
from src.rag import pipeline
from src.rag import tafsir as tafsir_module
from src.rag.tafsir import TafsirResolution


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


def _resolution(monkeypatch, resolution):
    monkeypatch.setattr(
        tafsir_module, "resolve_tafsir_reference",
        lambda *args, **kwargs: resolution,
    )


def test_chat_tafsir_exact_meaning_question_gets_both_anchors(monkeypatch):
    _resolution(monkeypatch, TafsirResolution(
        TAFSIR, None, "1:2", quran_chunk=QURAN, match_kind="exact",
    ))
    anchors, clarification = pipeline._resolve_chat_tafsir(
        "ما معنى الحمد لله رب العالمين؟", None, [], None)
    assert [c["id"] for c in anchors] == ["quran_1_2", "tafsir_ibnkathir_1_2"]
    assert clarification == ""


def test_chat_tafsir_whole_surah_returns_clarification(monkeypatch):
    _resolution(monkeypatch, TafsirResolution(
        None, None, "", failure_message="يرجى تحديد رقم الآية المطلوبة.",
    ))
    anchors, clarification = pipeline._resolve_chat_tafsir(
        "تفسير سورة الإخلاص", None, [], None)
    assert anchors == []
    assert "رقم الآية" in clarification


def test_chat_tafsir_does_not_hijack_generic_meaning_question(monkeypatch):
    _resolution(monkeypatch, TafsirResolution(
        None, None, "", failure_message="ambiguous", match_kind="partial",
        candidates=(QURAN,),
    ))
    anchors, clarification = pipeline._resolve_chat_tafsir(
        "ما معنى الصبر؟", None, [], None)
    assert anchors == []
    assert clarification == ""


def test_chat_tafsir_anchor_injection_bypasses_diversity_cap():
    pool = [{"id": "other", "source": "hadith", "score": 0.9}]
    augmented, injected = pipeline._inject_tafsir_anchors(
        [QURAN, TAFSIR], pool)
    assert injected == {"quran_1_2", "tafsir_ibnkathir_1_2"}
    by_id = {c["id"]: c for c in augmented}
    assert by_id["quran_1_2"]["_injected"] is True
    assert by_id["tafsir_ibnkathir_1_2"]["_injected"] is True
    assert by_id["quran_1_2"]["score"] == pipeline._SURAH_INJECT_SCORE


class _FakeChat:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {
                    "content": (
                        'قال الله تعالى: "الحمد لله رب العالمين" '
                        "(الفاتحة: 2) (مصدر 1)."
                    ),
                })(),
            })()],
        })()


def test_tafsir_generate_keeps_only_the_resolved_verse_and_commentary(monkeypatch):
    unrelated = {
        "id": "tafsir_unrelated",
        "source": "tafsir",
        "reference": "تفسير ابن كثير 9:85",
        "original_text": "كلام في آية أخرى لا علاقة له بالسؤال.",
        "_gate_passed": True,
    }
    quran = {**QURAN, "_gate_passed": True}
    tafsir = {**TAFSIR, "_gate_passed": True}

    monkeypatch.setattr(
        pipeline, "retrieve",
        lambda *args, **kwargs: [quran, tafsir, unrelated],
    )
    monkeypatch.setattr(
        pipeline, "_resolve_chat_tafsir",
        lambda *args, **kwargs: ([QURAN, TAFSIR], ""),
    )
    result = pipeline.generate(
        "ما معنى قوله تعالى: «الحمد لله رب العالمين»؟ اذكر السورة ورقم الآية.",
        None, [], _FakeChat(),
    )

    assert result["sources"] == [
        "quran — الفاتحة:2",
        "tafsir — تفسير ابن كثير 1:2",
    ]
    assert not any("9:85" in source for source in result["sources"])


def test_tafsir_meaning_fallback_adds_source_sentence_when_model_omits_it():
    tafsir = {
        **TAFSIR,
        "original_text": (
            "وَقَوْلُهُ: ﴿فَإِنَّ مَعَ الْعُسْرِ يُسْرًا﴾ أَخْبَرَ تَعَالَى "
            "أَنَّ مَعَ الْعُسْرِ يوجَدُ الْيُسْرُ، ثُمَّ أَكَّدَ هَذَا "
            "الْخَبَرَ."
        ),
    }
    found = pipeline._find_tafsir_meaning_sentence(
        "ما معنى قوله تعالى: «إِنَّ مَعَ الْعُسْرِ يُسْرًا»؟",
        [tafsir],
    )
    assert found is not None
    assert found[0] == 1
    assert "أَخْبَرَ تَعَالَى" in found[1]
    assert "يوجَدُ الْيُسْرُ" in found[1]
