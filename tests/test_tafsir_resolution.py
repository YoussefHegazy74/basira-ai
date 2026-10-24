"""Regression tests for canonical Quran verse resolution in the Tafsir flow."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.nlp.preprocessor import normalize_verse_text
from src.rag.pipeline import ground_text
from src.rag.tafsir import (
    _TAFSIR_INVALID_REFERENCE,
    _TAFSIR_MISSING,
    _TAFSIR_REFUSAL,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_RETRY,
    _generate_tafsir,
    get_tafsir,
    resolve_tafsir_reference,
)

ROOT = Path(__file__).resolve().parents[1]
QURAN_PATH = ROOT / "data" / "raw" / "quran" / "quran_processed.json"
TAFSIR_PATH = ROOT / "data" / "raw" / "tafsir" / "tafsir_processed.json"


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    records = json.loads(QURAN_PATH.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for record in records:
        metadata = record["metadata"]
        chunks.append({
            "id": record["id"],
            "source": "quran",
            "reference": record["reference"],
            "original_text": record["arabic_text"],
            "metadata": metadata,
        })
    for record in records:
        metadata = record["metadata"]
        surah = metadata["surah_number"]
        ayah = metadata["ayah_number"]
        chunks.append({
            "id": f"tafsir_{record['id']}",
            "source": "tafsir",
            "reference": f"تفسير ابن كثير {surah}:{ayah}",
            "original_text": f"تفسير تجريبي للآية {surah}:{ayah}",
            "metadata": metadata,
        })
    return chunks


def resolve(text: str, corpus: list[dict]):
    return resolve_tafsir_reference(text, None, corpus, None)


def candidate_refs(resolution) -> list[str]:
    return [
        f"{candidate['metadata']['surah_number']}:"
        f"{candidate['metadata']['ayah_number']}"
        for candidate in resolution.candidates
    ]


def test_verse_normalization_handles_uthmani_tatweel_and_digits():
    uthmani = "بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ"
    assert normalize_verse_text(
        uthmani, expand_dagger_alef=False,
    ) == "بسم الله الرحمن الرحيم"
    assert normalize_verse_text(uthmani) == "بسم الله الرحمان الرحيم"
    assert normalize_verse_text("قـــل هــو الله أحـــد") == "قل هو الله احد"
    assert normalize_verse_text("١١٢:١") == "112 1"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ", "1:1"),
        ("بسم الله الرحمن الرحيم", "1:1"),
        ("الحمد لله رب العالمين", "1:2"),
        ("إياك نعبد وإياك نستعين", "1:5"),
        ("اهدنا الصراط المستقيم", "1:6"),
        ("قل هو الله أحد", "112:1"),
        ("قـــل هــو الله أحـــد", "112:1"),
        ("  قل   هو   الله   أحد  ", "112:1"),
        ("إنا أعطيناك الكوثر", "108:1"),
        ("قل أعوذ برب الفلق", "113:1"),
    ],
)
def test_exact_and_variant_verses_resolve(text, expected, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == expected
    assert resolution.failure_message == ""
    assert resolution.quran_chunk is not None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("قل هو الله", "112:1"),
        ("من ذا الذي يشفع عنده إلا بإذنه", "2:255"),
        ("وسع كرسيه السماوات والأرض", "2:255"),
        ("تبارك الذي بيده الملك", "67:1"),
        ("إنا لله وإنا إليه راجعون", "2:156"),
    ],
)
def test_unique_partial_verse_resolves(text, expected, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == expected
    assert resolution.failure_message == ""
    assert resolution.match_kind == "partial"


def test_ambiguous_partial_text_is_reported_not_guessed(corpus):
    resolution = resolve("وهو العلي العظيم", corpus)
    assert resolution[0] is None
    assert "أكثر من آية" in resolution.failure_message
    refs = candidate_refs(resolution)
    assert "2:255" in refs
    assert "42:4" in refs
    assert len(refs) == 2


def test_repeated_verse_text_is_reported_as_ambiguous(corpus):
    resolution = resolve("فبأي آلاء ربكما تكذبان", corpus)
    assert resolution[0] is None
    assert "أكثر من آية" in resolution.failure_message
    assert len(resolution.candidates) == 31
    assert "55:13" in candidate_refs(resolution)


def test_shared_opening_phrase_is_reported_as_ambiguous(corpus):
    resolution = resolve("يا أيها الذين آمنوا", corpus)
    assert resolution[0] is None
    assert "أكثر من آية" in resolution.failure_message
    assert len(resolution.candidates) > 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("112:1", "112:1"),
        ("سورة الإخلاص آية 1", "112:1"),
        ("الفاتحة:2", "1:2"),
        ("البقرة 255", "2:255"),
        ("الآية 255 من سورة البقرة", "2:255"),
        ("آية الكرسي", "2:255"),
        ("Al-Ikhlas 1", "112:1"),
    ],
)
def test_explicit_references_resolve(text, expected, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == expected
    assert resolution.match_kind == "reference"


def test_invalid_reference_is_rejected(corpus):
    resolution = resolve("112:999", corpus)
    assert resolution[0] is None
    assert resolution.failure_message == _TAFSIR_INVALID_REFERENCE


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ما تفسير آية قل هو الله أحد؟", "112:1"),
        ("ما تفسير قوله تعالى قل هو الله أحد", "112:1"),
        ("اشرح قوله تعالى الحمد لله رب العالمين", "1:2"),
        ("تفسير آية الذين يأكلون أموال اليتامى ظلما", "4:10"),
        ("الآية التي نزلت في الذين يأكلون أموال اليتامى", "4:10"),
        ("ما تفسير سورة الإخلاص آية 1", "112:1"),
        ("ابحث عن تفسير الآية 255 من سورة البقرة", "2:255"),
        ("تفسير الآية 7 من سورة الأنفال", "8:7"),
        ("ايه 102:1", "102:1"),
        ("سورة التكاثر الاية الاولى", "102:1"),
        ("تفسير قوله تعالى: ﴿قُلْ هُوَ اللَّهُ أَحَدٌ﴾", "112:1"),
    ],
)
def test_natural_language_requests_resolve(text, expected, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == expected
    assert resolution.failure_message == ""


def test_tafsir_alias_resolves_inside_a_sentence(corpus):
    resolution = resolve("فسر لي آية الكرسي", corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == "2:255"
    assert resolution.match_kind == "reference"


def test_surah_only_request_asks_for_a_specific_ayah(corpus):
    resolution = resolve("تفسير سورة الإخلاص", corpus)
    assert resolution[0] is None
    assert "سورة الإخلاص" in resolution.failure_message
    assert "من 1 إلى 4" in resolution.failure_message


def test_ayah_without_surah_asks_for_the_surah(corpus):
    resolution = resolve("ما تفسير الآية 255", corpus)
    assert resolution[0] is None
    assert "اسم السورة" in resolution.failure_message


def test_natural_language_framing_does_not_break_ambiguity(corpus):
    resolution = resolve("ما تفسير آية فبأي آلاء ربكما تكذبان", corpus)
    assert resolution[0] is None
    assert "أكثر من آية" in resolution.failure_message
    assert len(resolution.candidates) == 31


def test_unknown_text_is_rejected(corpus):
    resolution = resolve("آية غير موجودة إطلاقاً اختبار فقط", corpus)
    assert resolution[0] is None
    assert resolution.failure_message == ""


def test_canonical_text_is_preserved_for_successful_match(corpus):
    resolution = resolve("قل هو الله أحد", corpus)
    canonical = resolution.quran_chunk
    assert canonical is not None
    assert canonical["metadata"]["surah_number"] == 112
    assert canonical["metadata"]["ayah_number"] == 1
    assert canonical["reference"] == "الإخلاص:1"
    assert normalize_verse_text(
        canonical["original_text"], expand_dagger_alef=False,
    ) == "قل هو الله احد"


def test_get_tafsir_reports_ambiguity_without_calling_model(corpus):
    client = Mock()
    result = get_tafsir("وهو العلي العظيم", None, corpus, client)
    assert set(result) == {"tafsir", "context", "source", "input_ayah"}
    assert "أكثر من آية" in result["tafsir"]
    assert "2:255" in result["tafsir"]
    assert "42:4" in result["tafsir"]
    assert result["source"] == ""
    assert result["context"] == ""
    client.chat.completions.create.assert_not_called()


def test_get_tafsir_unknown_text_uses_refusal_without_calling_model(corpus):
    client = Mock()
    result = get_tafsir("آية غير موجودة إطلاقاً اختبار فقط", None, corpus, client)
    assert result == {
        "tafsir": _TAFSIR_REFUSAL,
        "context": "",
        "source": "",
        "input_ayah": "آية غير موجودة إطلاقاً اختبار فقط",
    }
    client.chat.completions.create.assert_not_called()


def _tafsir_grounding_fixture():
    primary = {
        "id": "t1",
        "source": "tafsir",
        "reference": "تفسير ابن كثير 113:1",
        "original_text": "قال ابن عباس الفلق هو الصبح نور الصباح",
        "metadata": {},
    }
    corpus = [primary] + [
        {
            "id": f"f{i}",
            "source": "fiqh",
            "reference": "مرجع تجريبي",
            "original_text": (
                "نص تجريبي عام كلمهنادره" if i == 0 else "نص تجريبي عام"
            ),
            "metadata": {},
        }
        for i in range(600)
    ]
    return primary, corpus


def _json_client(payload: str) -> Mock:
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
    )
    return client


def test_ground_text_ignore_rare_allows_existing_rare_tokens():
    primary, corpus = _tafsir_grounding_fixture()
    answer = "قال ابن عباس الفلق هو الصبح نور الصباح كلمهنادره."
    assert ground_text(
        answer, [primary], corpus, refusal="REF",
    ) == "REF"
    assert ground_text(
        answer, [primary], corpus, refusal="REF", ignore_rare=True,
    ) == answer


def test_ground_text_ignore_rare_rejects_tokens_absent_from_corpus():
    primary, corpus = _tafsir_grounding_fixture()
    answer = "قال ابن عباس الفلق هو الصبح نور الصباح كلمهغريبه."
    assert ground_text(
        answer, [primary], corpus, refusal="REF", ignore_rare=True,
    ) == "REF"


def test_ground_text_ignore_rare_allows_weak_skeleton_paraphrase():
    primary = {
        "id": "t1",
        "source": "tafsir",
        "reference": "تفسير ابن كثير 102:1",
        "original_text": "قال ابن عباس الفلق هو الصبح نور الصباح وصرتم",
        "metadata": {},
    }
    corpus = [primary] + [
        {
            "id": f"f{i}",
            "source": "fiqh",
            "reference": "مرجع تجريبي",
            "original_text": "نص تجريبي عام",
            "metadata": {},
        }
        for i in range(600)
    ]
    answer = "قال ابن عباس الفلق هو الصبح نور الصباح وصيرتم."
    assert ground_text(
        answer, [primary], corpus, refusal="REF",
    ) == "REF"
    assert ground_text(
        answer, [primary], corpus, refusal="REF", ignore_rare=True,
    ) == answer


def test_generate_tafsir_uses_relaxed_grounding_before_refusing():
    primary, corpus = _tafsir_grounding_fixture()
    answer = "قال ابن عباس الفلق هو الصبح نور الصباح كلمهنادره."
    client = _json_client(json.dumps({
        "tafsir": answer,
        "context": "",
    }, ensure_ascii=False))

    tafsir, context = _generate_tafsir(
        client,
        SYSTEM_PROMPT,
        "الآية المدخلة: قل أعوذ برب الفلق",
        [primary],
        corpus,
    )

    assert tafsir == answer
    assert context == ""


def test_uthmani_alef_maqsura_with_dagger_alef_variants():
    uthmani = "أَلۡهَىٰكُمُ ٱلتَّكَاثُرُ"
    assert normalize_verse_text(
        uthmani, alef_maqsura_mode="alef", expand_dagger_alef=False,
    ) == "الهاكم التكاثر"
    assert normalize_verse_text(
        "هُدَىٰ", alef_maqsura_mode="alef", expand_dagger_alef=False,
    ) == "هدا"
    assert normalize_verse_text(
        "هُدَىٰ", alef_maqsura_mode="keep", expand_dagger_alef=False,
    ) == "هدي"


@pytest.mark.parametrize(
    "text",
    [
        "الهاكم التكاثر",
        "ألهاكم التكاثر",
        "أَلْهَاكُمُ التَّكَاثُرُ",
    ],
)
def test_at_takathur_opening_resolves_to_canonical_verse(text, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == "102:1"
    assert resolution.match_kind == "exact"
    assert resolution.failure_message == ""
    canonical = resolution.quran_chunk
    assert canonical is not None
    assert canonical["reference"] == "التكاثر:1"
    real = next(c for c in corpus if c["id"] == "quran_102_1")
    assert canonical["original_text"] == real["original_text"]


@pytest.mark.parametrize("text", ["الهاكم", "ألهاكم", "اَلْهَاكُمُ"])
def test_at_takathur_opening_partial_resolves(text, corpus):
    resolution = resolve(text, corpus)
    primary, _next, source = resolution
    assert primary is not None
    assert source == "102:1"
    assert resolution.match_kind == "partial"


def test_valid_verse_without_tafsir_is_distinct_from_not_found(corpus):
    filtered = [
        chunk
        for chunk in corpus
        if not (
            chunk.get("source") == "tafsir"
            and (chunk.get("metadata") or {}).get("surah_number") == 102
            and (chunk.get("metadata") or {}).get("ayah_number") == 1
        )
    ]
    resolution = resolve_tafsir_reference(
        "الهاكم التكاثر", None, filtered, None,
    )
    primary, _next, source = resolution
    assert primary is None
    assert source == "102:1"
    assert resolution.failure_message == _TAFSIR_MISSING

    client = Mock()
    result = get_tafsir("الهاكم التكاثر", None, filtered, client)
    assert result == {
        "tafsir": _TAFSIR_MISSING,
        "context": "",
        "source": "102:1",
        "input_ayah": "الهاكم التكاثر",
    }
    client.chat.completions.create.assert_not_called()


def test_system_prompt_separates_ayah_tafsir_from_surah_context():
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_RETRY):
        assert "الآية المطلوبة" in prompt
        assert "'context'" in prompt
        assert "سياق عام" in prompt


def test_surah_context_is_returned_separately_from_ayah_tafsir():
    primary, corpus = _tafsir_grounding_fixture()
    client = _json_client(json.dumps({
        "tafsir": "قال ابن عباس الفلق هو الصبح",
        "context": "قال ابن عباس الفلق هو الصبح نور الصباح",
    }, ensure_ascii=False))

    tafsir, context = _generate_tafsir(
        client,
        SYSTEM_PROMPT,
        "الآية المطلوبة فقط: قل أعوذ برب الفلق",
        [primary],
        corpus,
    )

    assert tafsir == "قال ابن عباس الفلق هو الصبح"
    assert context == "قال ابن عباس الفلق هو الصبح نور الصباح"


def test_model_failure_is_not_converted_to_not_found(corpus):
    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    raise RuntimeError("upstream model unavailable")

    with pytest.raises(RuntimeError, match="upstream model unavailable"):
        get_tafsir("الهاكم التكاثر", None, corpus, BrokenClient())


@pytest.fixture(scope="module")
def real_8_7_corpus() -> list[dict]:
    quran = json.loads(QURAN_PATH.read_text(encoding="utf-8"))
    tafsir = json.loads(TAFSIR_PATH.read_text(encoding="utf-8"))
    wanted = {(8, 7), (8, 8)}
    chunks: list[dict] = []
    for record in quran:
        metadata = record["metadata"]
        if (metadata["surah_number"], metadata["ayah_number"]) in wanted:
            chunks.append({
                "id": record["id"],
                "source": "quran",
                "reference": record["reference"],
                "original_text": record["arabic_text"],
                "metadata": metadata,
            })
    for record in tafsir:
        metadata = record["metadata"]
        if (metadata["surah_number"], metadata["ayah_number"]) in wanted:
            chunks.append({
                "id": record["id"],
                "source": "tafsir",
                "reference": record["reference"],
                "original_text": record["arabic_text"],
                "metadata": metadata,
            })
    return chunks


def test_8_7_inversion_in_quran_quote_is_not_refused(real_8_7_corpus):
    payload = json.dumps({
        "tafsir": (
            "وَمَعْنَى قَوْلِهِ تَعَالَى "
            "﴿وَتَوَدُّونَ أَنْ غَيْرَ ذَاتِ الشَّوْكَةِ تَكُونُ لَكُمْ﴾ "
            "أَيْ: يُحِبُّونَ أَنَّْ الطَّائِفَةَ الَّتِي لَا حَدَّ لَهَا "
            "وَلَا مَنَعَةَ وَلَا قِتَالَ، تَكُونُ لَهُمْ وَهِيَ الْعِيرُ."
        ),
        "context": "",
    }, ensure_ascii=False)
    client = _json_client(payload)

    result = get_tafsir(
        "وإذ يعدكم الله إحدى الطائفتين أنها لكم",
        None,
        real_8_7_corpus,
        client,
    )

    assert result["source"] == "8:7"
    assert result.get("source_scope") == "8:7–8:8"
    assert result["tafsir"] != _TAFSIR_REFUSAL
    assert "غير ذات الشوكه" in normalize_verse_text(
        result["tafsir"], expand_dagger_alef=False,
    )


def test_duplicate_commentary_scope_marks_only_contiguous_ranges(
    real_8_7_corpus,
):
    resolution = resolve_tafsir_reference(
        "تفسير الآية 7 من سورة الأنفال", None, real_8_7_corpus, None,
    )
    assert resolution.source_scope == (8, 7, 8)


def test_unique_commentary_has_no_source_scope(corpus):
    resolution = resolve("قل هو الله أحد", corpus)
    assert resolution.source_scope is None


def test_quran_quote_polarity_exemption_is_scoped_to_tafsir(real_8_7_corpus):
    source = next(
        chunk for chunk in real_8_7_corpus
        if chunk["id"] == "tafsir_ibnkathir_8_7"
    )
    answer = (
        "وَمَعْنَى قَوْلِهِ تَعَالَى "
        "﴿وَتَوَدُّونَ أَنْ غَيْرَ ذَاتِ الشَّوْكَةِ تَكُونُ لَكُمْ﴾ "
        "أَيْ: يُحِبُّونَ أَنَّْ الطَّائِفَةَ الَّتِي لَا حَدَّ لَهَا "
        "وَلَا مَنَعَةَ وَلَا قِتَالَ، تَكُونُ لَهُمْ وَهِيَ الْعِيرُ."
    )

    chat_default = ground_text(answer, [source], real_8_7_corpus)
    tafsir_scoped = ground_text(
        answer, [source], real_8_7_corpus, quran_quotes=True,
    )

    assert chat_default == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
    assert tafsir_scoped != chat_default
    assert "غير ذات الشوكه" in normalize_verse_text(
        tafsir_scoped, expand_dagger_alef=False,
    )
