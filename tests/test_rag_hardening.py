"""Regression tests for grounding/citation and embedding-failure hardening."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.rag import pipeline
from src.embeddings import embedder

REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."


@pytest.fixture
def retrieved():
    return [{
        "id": "test-1",
        "original_text": "الصلاة واجبة على المسلمين خمس مرات كل يوم",
        "source": "quran",
        "reference": "test",
    }]


def test_grounding_preserves_valid_citations(retrieved):
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1)."
    result = pipeline._ground_answer(answer, retrieved)
    assert "(مصدر 1)" in result


def test_grounding_rejects_invalid_explicit_citation(retrieved):
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 9)."
    result = pipeline._ground_answer(answer, retrieved)
    assert "الصلاة واجبة" not in result
    assert REFUSAL in result


def test_grounding_rejects_mixed_valid_and_invalid_citations(retrieved):
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1) (مصدر 9)."
    result = pipeline._ground_answer(answer, retrieved)
    assert "الصلاة واجبة" not in result
    assert REFUSAL in result


def test_generate_never_skips_grounding_for_refusal_prefix(retrieved):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال. ادعاء غير مدعوم إطلاقًا."
        ))]
    )
    source = dict(retrieved[0], score=1.0, _gate_passed=True)
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]), \
         patch.object(pipeline, "_ground_answer", return_value="filtered") as ground:
        result = pipeline.generate("سؤال", object(), retrieved, client)
    ground.assert_called_once()
    assert result["answer"] == "filtered"


def test_generate_preserves_valid_citation(retrieved):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1)."
        ))]
    )
    source = dict(retrieved[0], score=1.0, _gate_passed=True)
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate("سؤال", object(), [source], client)
    assert "الصلاة واجبة" in result["answer"]
    assert "(مصدر 1)" in result["answer"]


def test_generate_grounds_refusal_prefixed_unsupported_claim(retrieved):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=(
                "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال. "
                "غزوة بدر كانت في شهر شوال (مصدر 1)."
            )
        ))]
    )
    source = dict(retrieved[0], score=1.0, _gate_passed=True)
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate("سؤال", object(), [source], client)
    assert result["answer"] == REFUSAL


def _chunk(chunk_id: str, text: str) -> dict:
    return {"id": chunk_id, "clean_text": text}


def _client_with(vectors: list) -> Mock:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=vector) for vector in vectors]
    )
    return client


def test_embedding_api_failure_aborts_without_zero_vector():
    chunk = _chunk("c1", "نص صالح للاختبار")
    client = Mock()
    client.embeddings.create.side_effect = OSError("simulated API failure")
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks([chunk], client, batch_size=1)
    assert "embedding" not in chunk


def test_embedding_response_count_mismatch_aborts():
    chunks = [
        _chunk("c1", "النص الأول للاختبار"),
        _chunk("c2", "النص الثاني للاختبار"),
    ]
    client = _client_with([[0.1] * embedder.DIM])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks(chunks, client, batch_size=2)
    assert all("embedding" not in chunk for chunk in chunks)


def test_embedding_wrong_dimension_aborts_before_commit():
    chunks = [
        _chunk("c1", "النص الأول للاختبار"),
        _chunk("c2", "النص الثاني للاختبار"),
    ]
    client = _client_with([
        [0.1] * embedder.DIM,
        [0.1] * (embedder.DIM - 1),
    ])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks(chunks, client, batch_size=2)
    assert all("embedding" not in chunk for chunk in chunks)


def test_embedding_non_numeric_vector_aborts():
    chunk = _chunk("c1", "نص صالح للاختبار")
    client = _client_with([[0.1] * (embedder.DIM - 1) + ["bad"]])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks([chunk], client, batch_size=1)
    assert "embedding" not in chunk


def test_embedding_nan_vector_aborts():
    chunk = _chunk("c1", "نص صالح للاختبار")
    client = _client_with([[0.1] * (embedder.DIM - 1) + [float("nan")]])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks([chunk], client, batch_size=1)
    assert "embedding" not in chunk


def test_embedding_missing_vector_aborts():
    chunk = _chunk("c1", "نص صالح للاختبار")
    client = _client_with([None])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks([chunk], client, batch_size=1)
    assert "embedding" not in chunk


def test_embedding_empty_text_aborts_without_zero_vector():
    chunk = _chunk("c1", "   ")
    client = _client_with([[0.1] * embedder.DIM])
    with pytest.raises(RuntimeError, match="aborting to prevent a partial/zero-vector index"):
        embedder.embed_chunks([chunk], client, batch_size=1)
    client.embeddings.create.assert_not_called()
    assert "embedding" not in chunk


def test_embedding_valid_batch_commits_vectors():
    chunks = [
        _chunk("c1", "النص الأول للاختبار"),
        _chunk("c2", "النص الثاني للاختبار"),
    ]
    client = _client_with([[0.25] * embedder.DIM, [0.5] * embedder.DIM])
    result = embedder.embed_chunks(chunks, client, batch_size=2)
    assert result[0]["embedding"] == [0.25] * embedder.DIM
    assert result[1]["embedding"] == [0.5] * embedder.DIM


def test_embedding_rejects_invalid_batch_arguments():
    client = _client_with([])
    with pytest.raises(ValueError):
        embedder.embed_chunks([], client, batch_size=0)
    with pytest.raises(ValueError):
        embedder.embed_chunks([], client, start_batch=-1)


def _rag_chunk(chunk_id: str, text: str, source: str = "fiqh") -> dict:
    return {
        "id": chunk_id,
        "source": source,
        "reference": "test",
        "original_text": text,
        "clean_text": text,
        "metadata": {},
    }


def _badr_case():
    source = _rag_chunk(
        "badr_src",
        "كان أول صدام مسلح بين المسلمين والمشركين يوم السابع عشر من رمضان "
        "من السنة الثانية من الهجرة في بدر",
    )
    corpus = [source] + [_rag_chunk(f"b{i}", "غزوه بدر") for i in range(399)]
    return source, corpus


def _relation_case():
    fire = _rag_chunk(
        "fire_src",
        "إن الذين يأكلون أموال اليتامى ظلما إنما يأكلون في بطونهم نارا "
        "وسيصلون سعيرا",
        source="quran",
    )
    paradise = _rag_chunk(
        "paradise_src",
        "إن الذين آمنوا وعملوا الصالحات يدخلون الجنة",
        source="quran",
    )
    corpus = [fire, paradise] + [
        _rag_chunk(f"r{i}", "دخل الجنه والنار") for i in range(600)
    ]
    return fire, paradise, corpus


def test_battle_term_canon_includes_sadama():
    assert (
        pipeline._gkey("صدام")
        == pipeline._gkey("غزوه")
        == pipeline._gkey("وقعت")
    )


def test_badr_year_paraphrase_is_grounded():
    source, corpus = _badr_case()
    answer = (
        "وقعت غزوة بدر في السابع عشر من رمضان من السنة الثانية للهجرة "
        "(مصدر 1)."
    )
    assert pipeline._ground_answer(answer, [source], corpus) != REFUSAL


def test_badr_year_wrong_month_still_rejected():
    source, corpus = _badr_case()
    answer = (
        "وقعت غزوة بدر في السابع عشر من شوال من السنة الثانية للهجرة "
        "(مصدر 1)."
    )
    assert pipeline._ground_answer(answer, [source], corpus) == REFUSAL


def test_badr_year_wrong_place_still_rejected():
    source, corpus = _badr_case()
    answer = (
        "وقعت غزوة الخندق في السابع عشر من رمضان من السنة الثانية للهجرة "
        "(مصدر 1)."
    )
    assert pipeline._ground_answer(answer, [source], corpus) == REFUSAL


def test_relation_inversion_paradise_vs_fire_rejected():
    fire, _, corpus = _relation_case()
    answer = "الذين يأكلون أموال اليتامى ظلما يدخلون الجنة (مصدر 1)."
    assert pipeline._ground_answer(answer, [fire], corpus) == REFUSAL


def test_relation_same_destination_still_accepted():
    fire, _, corpus = _relation_case()
    answer = "الذين يأكلون أموال اليتامى ظلما يدخلون النار (مصدر 1)."
    assert pipeline._ground_answer(answer, [fire], corpus) != REFUSAL


def test_negated_paradise_claim_not_treated_as_contradiction():
    fire, _, corpus = _relation_case()
    answer = "الذين يأكلون أموال اليتامى ظلما لا يدخلون الجنة (مصدر 1)."
    assert pipeline._ground_answer(answer, [fire], corpus) != REFUSAL


def test_reverse_relation_inversion_rejected():
    _, paradise, corpus = _relation_case()
    answer = "الذين آمنوا وعملوا الصالحات يدخلون النار (مصدر 1)."
    assert pipeline._ground_answer(answer, [paradise], corpus) == REFUSAL


def test_date_intent_requires_date_near_event():
    near = _rag_chunk(
        "near",
        "فكانت وقعة بدر يوم الجمعة صبيحة سبع عشرة من شهر رمضان",
        source="sira",
    )
    far = _rag_chunk(
        "far",
        "ذكر بدر في هذا الباب " + "كلام " * 12 + "ثم ذكر فضل الجمعة",
        source="sira",
    )
    candidates = [
        {"id": "near", "source": "sira", "reference": "سيرة", "score": 0.1,
         "original_text": near["original_text"]},
        {"id": "far", "source": "sira", "reference": "سيرة", "score": 0.1,
         "original_text": far["original_text"]},
    ]
    out = {c["id"]: c for c in pipeline._apply_lexical_boost(
        candidates, "متى كانت غزوة بدر؟")}
    assert out["near"]["score"] > out["far"]["score"]
    assert out["near"].get("_date_intent_match") is True
    assert out["far"].get("_date_intent_match") is None


def test_date_query_retrieves_event_synonym_source(monkeypatch):
    target = _rag_chunk(
        "sira_synonym",
        "فكانت وقعة بدر يوم الجمعة صبيحة سبع عشرة من شهر رمضان.",
        source="sira",
    )
    distractors = [
        _rag_chunk(f"d{i}", "غزوة بدر في السنة الثانية من الهجرة", source="sira")
        for i in range(30)
    ]
    corpus = [target] + distractors

    monkeypatch.setattr(pipeline, "embed_text", lambda text, client: 1.0)
    monkeypatch.setattr(
        pipeline, "search",
        lambda emb, index, chunks, top_k=20: [
            dict(c, score=0.5 - i * 0.01)
            for i, c in enumerate(distractors[:top_k])
        ],
    )
    monkeypatch.setattr(pipeline, "_rewrite_query", lambda q, client: [])
    monkeypatch.setattr(pipeline, "USE_QUERY_REWRITE", False)
    selected = pipeline.retrieve(
        "متى غزوة بدر؟", object(), corpus, Mock(), top_k=8,
    )
    assert "sira_synonym" in {c["id"] for c in selected}


def test_ghusl_completeness_adds_source_timing_sentence():
    fiqh58 = _rag_chunk(
        "fiqh58",
        "غسل الجمعة مستحب، أمر الشارع بالغسل وأكده، ووقت الغسل "
        "يمتد من طلوع الفجر إلى صلاة الجمعة.",
    )
    fiqh59 = _rag_chunk(
        "fiqh59",
        "ويخرج وقت الغسل بالفراغ من الصلاة فمن اغتسل بعد الصلاة "
        "لا يكون غسلا للجمعة.",
    )
    answer = (
        "الغسل يوم الجمعة مستحب، وقد أمر الشارع بالغسل وأكده "
        "(مصدر 1). وقت الغسل يمتد من طلوع الفجر إلى صلاة الجمعة "
        "(مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [fiqh58, fiqh59],
    )
    assert "بالفراغ من الصلاة" in out


def test_ghusl_completeness_adds_explicit_recommendation():
    fiqh58 = _rag_chunk(
        "fiqh58",
        "غسل الجمعة مستحب، أمر الشارع بالغسل وأكده، والمراد بالوجوب "
        "تأكيد استحبابه.",
    )
    fiqh59 = _rag_chunk(
        "fiqh59", "ويخرج وقت الغسل بالفراغ من الصلاة.",
    )
    answer = (
        'حكم الغسل يوم الجمعة هو أنه "واجب على كل محتلم"، وقد ذكر '
        'الشافعي أن الأمر بالغسل على الاختيار لا على الوجوب (مصدر 1).'
    )
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [fiqh58, fiqh59],
    )
    assert "مستحب" in out
    assert "بالفراغ من الصلاة" in out


def test_date_priority_keeps_one_chunk_per_source_type():
    pool = [
        {"id": f"t{i}", "source": "tafsir", "score": 1.0 - i * 0.01,
         "_date_intent_match": True} for i in range(5)
    ]
    pool.append({
        "id": "sira_date", "source": "sira", "score": 0.7,
        "_date_intent_match": True,
    })
    ids = pipeline._select_date_priority_ids(pool)
    assert "t0" in ids
    assert "sira_date" in ids
    assert not any(i.startswith("t") and i != "t0" for i in ids)


def test_date_intent_qualifier_guard_suppresses_other_badr_sense():
    candidates = [{
        "id": "badr_akhira",
        "source": "sira",
        "reference": "غزوة بدر الآخرة في شعبان",
        "score": 0.1,
        "original_text": "غزوة بدر الآخرة في شعبان سنة أربع",
    }]
    out = pipeline._apply_lexical_boost(
        candidates, "متى كانت غزوة بدر؟")[0]
    assert out.get("_date_intent_match") is None
    assert out["score"] < 0.6


def test_priority_selection_keeps_semantic_and_lexical_guarantees():
    pool = [
        {"id": "f1", "source": "fiqh", "score": 6.0},
        {"id": "f2", "source": "fiqh", "score": 5.0},
        {"id": "f3", "source": "fiqh", "score": 4.0},
        {"id": "f4", "source": "fiqh", "score": 3.0},
        {"id": "f5", "source": "fiqh", "score": 2.0},
        {"id": "q1", "source": "quran", "score": 0.5},
    ]
    selected = pipeline._select_diverse(
        pool, 4, priority_ids={"f5", "q1"})
    ids = {c["id"] for c in selected}
    assert {"f5", "q1"} <= ids
    assert len(ids) == 4


def test_discourse_connector_does_not_drop_supported_sentence():
    source = _rag_chunk(
        "discourse_src", "ثبت استحباب صيام ست من شوال")
    corpus = [source] + [_rag_chunk(f"d{i}", "نص آخر للاختبار")
                         for i in range(400)]
    answer = "وقد ثبت استحباب صيام ست من شوال (مصدر 1)."
    assert pipeline._ground_answer(answer, [source], corpus) != REFUSAL


def test_mixed_source_polarity_is_rejected_as_contradictory():
    source = _rag_chunk(
        "mixed_pol",
        "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج "
        "وصيام يوم عرفة مستحب لغير الحاج",
    )
    corpus = [source] + [_rag_chunk(f"m{i}", "نص آخر للاختبار")
                         for i in range(400)]
    answer = "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1)."
    assert pipeline._ground_answer(answer, [source], corpus) == REFUSAL


def test_unmixed_inverted_source_remains_supported():
    source = _rag_chunk(
        "inverted_pol",
        "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج",
    )
    corpus = [source] + [_rag_chunk(f"u{i}", "نص آخر للاختبار")
                         for i in range(400)]
    answer = "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1)."
    assert pipeline._ground_answer(answer, [source], corpus) != REFUSAL


def test_grounding_ignores_quran_ayah_reference_digits():
    source = _rag_chunk(
        "q_orphan",
        "إن الذين يأكلون أموال اليتامى ظلما إنما يأكلون في بطونهم نارا "
        "وسيصلون سعيرا",
        source="quran",
    )
    corpus = [source] + [
        _rag_chunk(f"q{i}", "جزاء ذلك سنة نص آخر للاختبار") for i in range(399)
    ]
    answer = (
        'جزاء الذين يأكلون أموال اليتامى أنهم "إنما يأكلون في بطونهم '
        'نارا وسيصلون سعيرا" (النساء: 10) (مصدر 1).'
    )
    result = pipeline._ground_answer(answer, [source], corpus)
    assert REFUSAL not in result
    assert "سعيرا" in result


def test_grounding_accepts_supported_quote_with_morphological_framing():
    source = _rag_chunk(
        "fiqh_qard",
        'كل قرض جر نفعا فهو ربا، ولا يحل للمقرض أن يشترط زيادة.',
    )
    corpus = [source] + [
        _rag_chunk(f"f{i}", "ذلك سنة جزاء نص آخر للاختبار") for i in range(399)
    ]
    answer = (
        'إن القرض الذي يجر نفعا يعتبر ربا، حيث ينص على ذلك: '
        '"كل قرض جر نفعا فهو ربا" (مصدر 1).'
    )
    result = pipeline._ground_answer(answer, [source], corpus)
    assert REFUSAL not in result
    assert "كل قرض جر نفعا فهو ربا" in result


def test_grounding_links_wajib_and_wujub_ruling_family():
    source = _rag_chunk(
        "fiqh_aqiqa",
        "والعقيقة سنة مؤكدة ولو كان الاب معسرا، فعلها الرسول، صلى "
        "الله عليه وسلم، وفعلها أصحابه، ويرى وجوبها الليث وداود "
        "الظاهري.",
    )
    answer = (
        "حكم العقيقة هو أنها سنة مؤكدة، وقد فعلها الرسول صلى الله "
        "عليه وسلم وأصحابه، ويرى بعض العلماء مثل الليث وداود الظاهري "
        "أنها واجبة (مصدر 1)."
    )
    result = pipeline._ground_answer(answer, [source], None)
    assert REFUSAL not in result
    assert "سنة مؤكدة" in result
    assert "أنها واجبة" in result
    assert pipeline._gkey("واجبه") == pipeline._gkey("وجوبها")


def test_quote_citation_prefers_primary_hadith_without_variant_query():
    darimi = _rag_chunk(
        "darimi_quote",
        'قال رسول الله صلى الله عليه وسلم: "من كان يؤمن بالله واليوم الآخر فليكرم ضيفه".',
        source="hadith",
    )
    darimi["reference"] = "سنن الدارمي:1"
    bukhari = _rag_chunk(
        "bukhari_quote",
        'قال رسول الله صلى الله عليه وسلم: "من كان يؤمن بالله واليوم الآخر فليكرم ضيفه".',
        source="hadith",
    )
    bukhari["reference"] = "صحيح البخاري:1"
    answer = (
        'قال رسول الله صلى الله عليه وسلم: "من كان يؤمن بالله واليوم '
        'الآخر فليكرم ضيفه" (مصدر 1).'
    )
    out = pipeline._repair_citation_support(answer, [darimi, bukhari])
    assert "(مصدر 2)" in out
    assert "(مصدر 1)" not in out


def test_quote_citation_keeps_secondary_source_when_primary_has_scattered_words():
    darimi = _rag_chunk(
        "darimi_exact",
        'قال رسول الله صلى الله عليه وسلم: "فليكرم جاره".',
        source="hadith",
    )
    darimi["reference"] = "سنن الدارمي:1"
    bukhari = _rag_chunk(
        "bukhari_scattered",
        'قال رسول الله صلى الله عليه وسلم: "فليكرم ضيفه"، '
        'ومن كان يؤمن بالله واليوم الآخر فلا يؤذ جاره".',
        source="hadith",
    )
    bukhari["reference"] = "صحيح البخاري:1"
    answer = 'قال رسول الله صلى الله عليه وسلم: "فليكرم جاره" (مصدر 1).'
    out = pipeline._repair_citation_support(answer, [darimi, bukhari])
    assert "(مصدر 1)" in out
    assert "(مصدر 2)" not in out


def test_grounding_keeps_supported_isna_item_with_framing_word():
    source = _rag_chunk(
        "fiqh_isha",
        "تقدم من الاحاديث ما يدل على سنية الركعتين بعد العشاء.",
    )
    corpus = [source] + [
        _rag_chunk(f"i{i}", "سنة ذلك جزاء نص آخر للاختبار") for i in range(399)
    ]
    answer = "سنية العشاء: هناك أحاديث تدل على سنية الركعتين بعد العشاء (مصدر 1)."
    result = pipeline._ground_answer(answer, [source], corpus)
    assert REFUSAL not in result
    assert "سنية العشاء" in result


def test_rawatib_paraphrase_phrase_alias_expands_scoped_trigram():
    tokens = ["صلوات", "مسنونه", "رواتب"]
    rewrites = pipeline._phrase_alias_rewrites(tokens)
    assert ["سنن", "راتبه", "رواتب"] in rewrites


def test_fiqh_bm25_cap_keeps_decisive_tail_passage():
    source = _rag_chunk(
        "long_fiqh",
        ("حديث في فضل الصلاة " * 30) + "الوتر سنة مؤكدة حث عليه النبي",
    )
    corpus = [source] + [
        _rag_chunk(f"long{i}", "نص آخر للاختبار") for i in range(30)
    ]
    pipeline._bm25_cache.clear()
    hits = pipeline._bm25_search(
        ["وتر", "سنه", "مؤكده"], corpus, top_k=5
    )
    assert hits
    assert hits[0]["id"] == "long_fiqh"


def test_grounding_accepts_form_v_variant_liyaqwa():
    source = _rag_chunk(
        "arafah_strength",
        "ثبت عنه أنه نهى عن صوم يوم عرفة بعرفات ليتقوى على الدعاء",
    )
    corpus = [source] + [
        _rag_chunk(f"a{i}", "نص آخر للاختبار") for i in range(400)
    ]
    answer = (
        "ثبت أنه نهى عن صوم يوم عرفة بعرفات ليقوى على الدعاء (مصدر 1)."
    )
    assert pipeline._ground_answer(answer, [source], corpus) != REFUSAL


def test_system_prompt_requires_body_citation_over_title():
    assert "استشهد بالمصدر الذي يذكر المعلومة في متنه" in pipeline.SYSTEM_MESSAGE


def test_section_context_adds_next_fiqh_paragraph():
    parent = _rag_chunk(
        "sec_parent", "غسل الجمعة أمر الشارع به وأكده",
    )
    parent["metadata"] = {
        "book": "فقه السنة", "chapter": "الطهارة",
        "subheading": "الأغسال المستحبة", "chunk": 1, "total_chunks": 2,
    }
    child = _rag_chunk(
        "sec_child", "ويخرج وقت الغسل بالفراغ من الصلاة",
    )
    child["metadata"] = {
        "book": "فقه السنة", "chapter": "الطهارة",
        "subheading": "الأغسال المستحبة", "chunk": 2, "total_chunks": 2,
    }
    other = _rag_chunk("sec_other", "حديث آخر", source="hadith")
    out = pipeline._expand_section_context(
        [parent, other], [parent, child, other], ["غسل", "جمعه"]
    )
    assert [c["id"] for c in out] == ["sec_parent", "sec_child"]


def test_section_context_skips_unrelated_next_paragraph():
    parent = _rag_chunk(
        "sec_parent2", "غسل الجمعة أمر الشارع به وأكده",
    )
    parent["metadata"] = {
        "book": "فقه السنة", "chapter": "الطهارة",
        "subheading": "الأغسال المستحبة", "chunk": 1, "total_chunks": 2,
    }
    child = _rag_chunk("sec_child2", "باب آخر لا علاقة له")
    child["metadata"] = {
        "book": "فقه السنة", "chapter": "الطهارة",
        "subheading": "الأغسال المستحبة", "chunk": 2, "total_chunks": 2,
    }
    other = _rag_chunk("sec_other2", "حديث آخر", source="hadith")
    out = pipeline._expand_section_context(
        [parent, other], [parent, child, other], ["غسل", "جمعه"]
    )
    assert [c["id"] for c in out] == ["sec_parent2", "sec_other2"]


def test_terminology_anchor_finds_rawatib_list_chunk():
    source = _rag_chunk(
        "rawatib_list",
        "التطوع المقيد يسمى السنن الراتبة، ويشمل سنة الفجر والظهر "
        "والعصر والمغرب والعشاء في الصلاة",
    )
    corpus = [source] + [
        _rag_chunk(f"r{i}", "نص آخر للاختبار") for i in range(20)
    ]
    pipeline._bm25_cache.clear()
    assert pipeline._terminology_anchor_ids(["سنن"], corpus) == {"rawatib_list"}


def test_priority_must_ids_are_not_trimmed_by_score():
    candidates = [
        {"id": "high1", "source": "hadith", "score": 0.9},
        {"id": "high2", "source": "tafsir", "score": 0.8},
        {"id": "low_must", "source": "fiqh", "score": 0.1},
    ]
    selected = pipeline._select_diverse(
        candidates, 2,
        priority_ids={"high1", "high2", "low_must"},
        must_ids={"low_must"},
    )
    assert {c["id"] for c in selected} == {"low_must", "high1"}


def test_dual_rakatan_form_matches_family_source():
    assert (
        pipeline._match_key("ركعتان")
        == pipeline._match_key("ركعتا")
        == pipeline._match_key("ركعتين")
    )
    source = _rag_chunk(
        "rawatib_dual",
        "وكان يصلي ركعتين قبل الفجر، وقال النبي صلى الله عليه وسلم: "
        "ركعتا الفجر خير من الدنيا وما فيها.",
    )
    corpus = [source] + [
        _rag_chunk(f"rawatib_bg{i}", "نص آخر للاختبار") for i in range(400)
    ]
    answer = (
        '**ركعتان قبل الفجر**: قال النبي صلى الله عليه وسلم: '
        '"ركعتا الفجر خير من الدنيا وما فيها" (مصدر 1).'
    )
    result = pipeline._ground_answer(answer, [source], corpus)
    assert result != REFUSAL
    assert "ركعتان قبل الفجر" in result


def test_negation_bridges_short_preposition_to_target():
    source = _rag_chunk(
        "ghusl_negation",
        "وقال الشافعي إن الأمر بالغسل يوم الجمعة للاختيار لا على الوجوب.",
    )
    corpus = [source] + [
        _rag_chunk(f"ghusl_bg{i}", "نص آخر للاختبار") for i in range(400)
    ]
    answer = (
        "قال الشافعي إن الأمر بالغسل يوم الجمعة للاختيار وليس للوجوب "
        "(مصدر 1)."
    )
    result = pipeline._ground_answer(answer, [source], corpus)
    assert result != REFUSAL
    assert "وليس للوجوب" in result


def test_completeness_preserves_ghusl_qualification_and_timing():
    source = _rag_chunk(
        "ghusl_complete",
        "غسل يوم الجمعة واجب على كل محتلم. والمراد بالوجوب تأكيد استحبابه. "
        "أمر الشارع بالغسل وأكده. ويخرج وقت الغسل بالفراغ من الصلاة.",
    )
    source["reference"] = "فقه السنة - الأغسال المستحبة (1/3)"
    answer = "غسل يوم الجمعة واجب على كل محتلم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [source],
    )
    assert "مستحب" in out
    assert "الشارع" in out
    assert "تاكيد استحبابه" in out
    assert "بالفراغ من الصلاة" in out
    assert "(مصدر 1)" in out


def test_completeness_does_not_invent_ghusl_qualification():
    source = _rag_chunk(
        "ghusl_plain", "غسل يوم الجمعة واجب على كل محتلم.",
    )
    answer = "غسل يوم الجمعة واجب على كل محتلم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [source],
    )
    assert out == answer


def test_completeness_uses_full_rawatib_source_list():
    source = _rag_chunk(
        "rawatib_complete",
        "التطوع المقيد يسمى السنن الراتبة، ويشمل سنة الفجر والظهر "
        "والعصر والمغرب والعشاء.",
    )
    answer = (
        "السنن المؤكدة تشمل الفجر والعصر والمغرب والعشاء (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "ما هي السنن المؤكدة في الصلاة؟", answer, [source],
    )
    assert "الظهر" in out
    assert "سنة الفجر والظهر والعصر" in out


def test_completeness_does_not_add_unmentioned_list_item():
    source = _rag_chunk(
        "rawatib_partial",
        "التطوع المقيد يسمى السنن الراتبة، ويشمل سنة الفجر والعصر "
        "والمغرب والعشاء.",
    )
    answer = (
        "السنن المؤكدة تشمل الفجر والعصر والمغرب والعشاء (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "ما هي السنن المؤكدة في الصلاة؟", answer, [source],
    )
    assert "الظهر" not in out


def test_completeness_prepends_source_qadr_definition():
    source = _rag_chunk(
        "qadr_definition",
        "يخبر الله تعالى أنه أنزل القرآن ليلة القدر، وهي الليلة "
        "المباركة، وهي من شهر رمضان.",
        source="tafsir",
    )
    answer = "فضلها عظيم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما هي ليلة القدر؟", answer, [source],
    )
    assert out.startswith("يخبر الله")
    assert "الليلة المباركة" in out
    assert "من شهر رمضان" in out
    assert "(مصدر 1)" in out


def test_completeness_does_not_invent_missing_kahf_fact():
    source = _rag_chunk(
        "kahf_context",
        "قصة أصحاب الكهف والرقيم، وفتية في الكهف.",
        source="tafsir",
    )
    answer = "أما الكهف فهو الغار (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما قصة أصحاب الكهف؟", answer, [source],
    )
    assert "الكهف والرقيم" in out
    assert "آمنوا" not in out


def test_generate_applies_source_completeness_after_grounding():
    text = (
        "غسل يوم الجمعة واجب على كل محتلم. والمراد بالوجوب تأكيد استحبابه. "
        "أمر الشارع بالغسل وأكده. ويخرج وقت الغسل بالفراغ من الصلاة."
    )
    source = _rag_chunk("ghusl_generate", text)
    source["reference"] = "فقه السنة - الأغسال المستحبة (1/3)"
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="غسل يوم الجمعة واجب على كل محتلم (مصدر 1)."
        ))]
    )
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "ما حكم الغسل يوم الجمعة؟", object(), [source], client,
        )
    assert "مستحب" in result["answer"]
    assert "بالفراغ من الصلاة" in result["answer"]


def test_completeness_cites_the_source_that_contains_qadr_definition():
    unrelated = _rag_chunk(
        "qadr_unrelated", "ليلة القدر خير من ألف شهر.",
        source="quran",
    )
    definition = _rag_chunk(
        "qadr_definition_2",
        "يخبر الله تعالى أنه أنزل القرآن ليلة القدر، وهي الليلة "
        "المباركة، وهي من شهر رمضان.",
        source="tafsir",
    )
    answer = "فضلها عظيم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما هي ليلة القدر؟", answer, [unrelated, definition],
    )
    assert out.startswith("يخبر الله")
    assert out.split("\n\n")[0].endswith("(مصدر 2).")


def test_lexical_variant_query_uses_corpus_surface_forms():
    assert pipeline._lexical_variant_query(["غش", "مسلمين"]) == [
        "غشنا", "مسلمين",
    ]
    assert pipeline._lexical_variant_query([
        "معيار", "كمال", "ايمان", "معامله", "اخ",
    ]) == [
        "معيار", "كمال", "يؤمن", "معامله", "لاخيه",
    ]
    assert pipeline._lexical_variant_query(["يرحمه", "له", "ناس"]) == [
        "يرحمهم", "له", "ناس",
    ]
    assert pipeline._lexical_variant_query(["صلاه", "فجر"]) == []


def test_phrase_new_variant_requires_a_literal_to_variant_replacement():
    assert pipeline._phrase_uses_new_variant(
        ["شديد", "حقا"], ["قوي", "حقا"],
    )
    assert not pipeline._phrase_uses_new_variant(
        ["قوي", "حقا"], ["قوي", "حقا"],
    )
    assert not pipeline._phrase_uses_new_variant(
        ["شديد", "حقا"], ["قوي", "حقا", "شديد"],
    )


def test_narrative_quran_anchors_include_first_and_next_ayah():
    tafsir_t9 = _rag_chunk(
        "t9", "قصة أصحاب الكهف والرقيم", source="tafsir",
    )
    tafsir_t10 = _rag_chunk(
        "t10", "قصة أصحاب الكهف والرقيم", source="tafsir",
    )
    tafsir_t9["reference"] = "تفسير ابن كثير 18:9"
    tafsir_t10["reference"] = "تفسير ابن كثير 18:12"
    quran_q9 = _rag_chunk(
        "q9", "أم حسبت أن أصحب الكهف والرقيم كانوا آياتنا عجبا",
        source="quran",
    )
    quran_q13 = _rag_chunk(
        "q13", "إنهم فتية آمنوا بربهم وزدناهم هدى", source="quran",
    )
    quran_q9["metadata"] = {"surah_number": 18, "ayah_number": 9}
    quran_q13["metadata"] = {"surah_number": 18, "ayah_number": 13}
    corpus = [tafsir_t9, tafsir_t10, quran_q9, quran_q13]
    anchors = pipeline._narrative_quran_anchors(
        [tafsir_t9, tafsir_t10], corpus, ["قصه", "اصحاب"],
    )
    assert [c["id"] for c in anchors] == ["q9", "q13"]
    assert pipeline._narrative_quran_anchors(
        [tafsir_t9, tafsir_t10], corpus, ["صلاة", "فجر"],
    ) == []


def test_narrative_quran_expansion_keeps_both_anchors():
    tafsir_t9 = _rag_chunk(
        "t9", "قصة أصحاب الكهف والرقيم", source="tafsir",
    )
    tafsir_t10 = _rag_chunk(
        "t10", "قصة أصحاب الكهف والرقيم", source="tafsir",
    )
    tafsir_t9["reference"] = "تفسير ابن كثير 18:9"
    tafsir_t10["reference"] = "تفسير ابن كثير 18:12"
    quran_q9 = _rag_chunk(
        "q9", "أم حسبت أن أصحب الكهف والرقيم", source="quran",
    )
    quran_q13 = _rag_chunk(
        "q13", "إنهم فتية آمنوا بربهم", source="quran",
    )
    quran_q9["metadata"] = {"surah_number": 18, "ayah_number": 9}
    quran_q13["metadata"] = {"surah_number": 18, "ayah_number": 13}
    selected = [
        dict(tafsir_t9, score=1.0),
        dict(tafsir_t10, score=0.9),
        {"id": "low1", "source": "sira", "score": 0.1},
        {"id": "low2", "source": "fiqh", "score": 0.2},
    ]
    corpus = [tafsir_t9, tafsir_t10, quran_q9, quran_q13]
    out = pipeline._expand_narrative_quran_context(
        selected, corpus, ["قصه", "اصحاب"], 4,
    )
    ids = [c["id"] for c in out]
    assert "q9" in ids and "q13" in ids
    assert len(ids) == 4


def test_retrieve_promotes_lexical_variant_evidence(monkeypatch):
    gold = _rag_chunk(
        "gold_ghish",
        "من حمل علينا السلاح فليس منا ومن غشنا فليس منا",
        source="hadith",
    )
    distractors = [
        _rag_chunk(f"d{i}", "نص آخر عن الغش والمسلمين", source="hadith")
        for i in range(12)
    ]
    corpus = [gold] + distractors
    semantic = [
        {**c, "score": 0.50}
        for c in distractors[:6]
    ]
    monkeypatch.setattr(pipeline, "_LEXICAL_VARIANT_ANCHOR_MIN_SCORE", 0.0)
    monkeypatch.setattr(pipeline, "embed_text", lambda text, client: [0.1])
    monkeypatch.setattr(
        pipeline, "search", lambda emb, index, chunks, top_k=20: semantic,
    )
    monkeypatch.setattr(pipeline, "_rewrite_query", lambda q, client: [])
    selected = pipeline.retrieve(
        "ما حكم من غش المسلمين؟", object(), corpus, Mock(), top_k=3,
    )
    assert "gold_ghish" in {c["id"] for c in selected}


def test_retrieve_promotes_synonym_rewrite_anchor(monkeypatch):
    gold = _rag_chunk(
        "gold_shadid",
        "إنما الشديد الذي يملك نفسه عند الغضب",
        source="hadith",
    )
    distractors = [
        _rag_chunk(f"r{i}", "نص آخر عن القوة", source="hadith")
        for i in range(12)
    ]
    corpus = [gold] + distractors
    semantic = [{**c, "score": 0.50} for c in distractors[:6]]

    def fake_embed(text, client):
        return 2.0 if "الشديد" in text else 1.0

    def fake_search(emb, index, chunks, top_k=20):
        if emb == 2.0:
            return [{**gold, "score": 0.60}]
        return semantic

    monkeypatch.setattr(pipeline, "embed_text", fake_embed)
    monkeypatch.setattr(pipeline, "search", fake_search)
    monkeypatch.setattr(
        pipeline, "_rewrite_query", lambda q, client: ["من هو الشديد حقا"],
    )
    selected = pipeline.retrieve(
        "من هو القوي حقا؟", object(), corpus, Mock(), top_k=3,
    )
    assert "gold_shadid" in {c["id"] for c in selected}


def test_kahf_evidence_includes_the_ayah_supplying_second_fact():
    from src.rag.eval_set import EVAL_QUESTIONS

    kahf = next(q for q in EVAL_QUESTIONS if q["id"] == "kahf_story")
    assert "quran_18_13" in kahf["evidence_ids"]


def test_citation_repair_replaces_title_only_support():
    title_only = _rag_chunk(
        "title_only",
        "ثم تيمموا مكانا ضيقا من الخندق وخرج علي بن أبي طالب.",
        source="sira",
    )
    title_only["reference"] = "سيرة ابن هشام، غزوة الخندق في شوال سنة خمس"
    body_support = _rag_chunk(
        "body_support",
        "ثم كانت غزوة الخندق في شوال سنة خمس.",
        source="sira",
    )
    answer = "كانت غزوة الخندق في شوال سنة خمس (مصدر 1)."
    out = pipeline._repair_citation_support(
        answer, [title_only, body_support],
    )
    assert "(مصدر 2)" in out
    assert "(مصدر 1)" not in out


def test_citation_repair_keeps_body_supported_citation():
    source = _rag_chunk(
        "body_ok",
        "ثم كانت غزوة الخندق في شوال سنة خمس.",
        source="sira",
    )
    answer = "كانت غزوة الخندق في شوال سنة خمس (مصدر 1)."
    assert pipeline._repair_citation_support(answer, [source]) == answer


def test_variant_citation_prefers_primary_hadith_when_quote_matches():
    darimi = _rag_chunk(
        "darimi_quote",
        'قال رسول الله صلى الله عليه وسلم: "من غشنا فليس منا".',
        source="hadith",
    )
    darimi["reference"] = "سنن الدارمي:1"
    muslim = _rag_chunk(
        "muslim_quote",
        'قال رسول الله صلى الله عليه وسلم: "من غشنا فليس منا".',
        source="hadith",
    )
    muslim["reference"] = "صحيح مسلم:1"
    answer = 'قال رسول الله صلى الله عليه وسلم: "من غشنا فليس منا" (مصدر 1).'
    out = pipeline._repair_citation_support(
        answer, [darimi, muslim], "ما حكم من غش المسلمين؟",
    )
    assert "(مصدر 2)" in out
    assert "(مصدر 1)" not in out


def test_variant_refusal_recovers_primary_quoted_hadith():
    darimi = _rag_chunk(
        "darimi_variant",
        'قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: '
        '"لَا يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لِأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ".',
        source="hadith",
    )
    darimi["reference"] = "سنن الدارمي:1"
    bukhari = _rag_chunk(
        "bukhari_variant",
        'قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: '
        '"\u200f لَا يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لِأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ \u200f".',
        source="hadith",
    )
    bukhari["reference"] = "صحيح البخاري:13"
    out = pipeline._augment_answer_completeness(
        "ما معيار كمال الإيمان في معاملة الأخ؟",
        REFUSAL,
        [darimi, bukhari],
    )
    assert out != REFUSAL
    assert "(مصدر 2)" in out
    assert "يومن" in pipeline._fold_ar(pipeline.normalize(out))


def test_variant_quote_requires_exact_variant_surface_not_just_stem():
    stem_only = _rag_chunk(
        "stem_only",
        'قال رسول الله صلى الله عليه وسلم: "لا يرحم الله من لا يرحم الناس".',
        source="hadith",
    )
    stem_only["reference"] = "صحيح البخاري:1"
    exact = _rag_chunk(
        "exact_variant",
        'قال رسول الله صلى الله عليه وسلم: "'
        'الراحمون يرحمهم الرحمن ارحموا أهل الأرض يرحمكم من في السماء".',
        source="hadith",
    )
    exact["reference"] = "صحيح البخاري:2"
    found = pipeline._find_variant_quote(
        "من الذي يرحمه الله من الناس؟", [stem_only, exact],
    )
    assert found is not None
    index, quote = found
    assert index == 2
    assert "يرحمهم" in pipeline.normalize(quote)


def test_variant_quote_is_appended_when_answer_omits_the_source_wording():
    source = _rag_chunk(
        "variant_append",
        'قَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ: '
        '"لَيْسَ الشَّدِيدُ بِالصُّرَعَةِ، إِنَّمَا الشَّدِيدُ الَّذِي '
        'يَمْلِكُ نَفْسَهُ عِنْدَ الْغَضَبِ".',
        source="hadith",
    )
    source["reference"] = "صحيح البخاري:5882"
    answer = "القوي هو الله سبحانه وتعالى (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "من هو القوي حقا؟", answer, [source],
    )
    assert "الشديد" in pipeline.normalize(out)
    assert out.count("(مصدر 1)") >= 2


def test_variant_refusal_does_not_recover_without_variant_evidence():
    source = _rag_chunk(
        "no_variant",
        "قال رسول الله صلى الله عليه وسلم: \"اتقوا النار ولو بشق تمرة\".",
        source="hadith",
    )
    out = pipeline._augment_answer_completeness(
        "ما معيار كمال الإيمان في معاملة الأخ؟", REFUSAL, [source],
    )
    assert out == REFUSAL


def test_story_completeness_adds_aligned_next_quran_ayah():
    tafsir = _rag_chunk(
        "tafsir_18_10",
        "هذا إخبار عن قصة أصحاب الكهف والرقيم على سبيل الإجمال.",
        source="tafsir",
    )
    tafsir["reference"] = "تفسير ابن كثير 18:10"
    q9 = _rag_chunk(
        "quran_18_9",
        "أم حسبت أن أصحاب الكهف والرقيم كانوا من آياتنا عجبا",
        source="quran",
    )
    q9["metadata"] = {"surah_number": 18, "ayah_number": 9}
    q13 = _rag_chunk(
        "quran_18_13",
        "نحن نقص عليك نبأهم بالحق إنهم فتية آمنوا بربهم وزدناهم هدى",
        source="quran",
    )
    q13["metadata"] = {"surah_number": 18, "ayah_number": 13}
    answer = "أما الكهف فهو الغار (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما قصة أصحاب الكهف؟", answer, [tafsir, q9, q13],
    )
    assert "فتيه" in pipeline._fold_ar(pipeline.normalize(out))
    assert "(مصدر 3)" in out


def test_grounding_expands_multi_source_citation_clusters():
    source = _rag_chunk(
        "multi_source",
        "الصلاة واجبة على المسلمين خمس مرات كل يوم",
        source="quran",
    )
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1، مصدر 2)."
    out = pipeline._ground_answer(answer, [source, source])
    assert "(مصدر 1) (مصدر 2)" in out
    assert "خمس مرات" in out


def test_grounding_rejects_invalid_number_in_citation_cluster():
    source = _rag_chunk(
        "cluster_source",
        "الصلاة واجبة على المسلمين خمس مرات كل يوم",
        source="quran",
    )
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1، مصدر 9)."
    assert pipeline._ground_answer(answer, [source]) == REFUSAL


def test_grounding_drops_list_marker_only_fragments():
    source = _rag_chunk(
        "list_source",
        "الصلاة واجبة على المسلمين خمس مرات كل يوم",
        source="quran",
    )
    answer = "1. الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1). 2."
    out = pipeline._ground_answer(answer, [source])
    assert "خمس مرات" in out
    assert "2." not in out


def test_grounding_drops_short_unsupported_cited_claim():
    source = _rag_chunk(
        "short_claim_source",
        "القوة الحقيقية هي التقوى والخشوع لله تعالى.",
        source="hadith",
    )
    answer = "التقوى والخوف من الله (مصدر 1)."
    out = pipeline._ground_answer(answer, [source])
    assert "الخوف" not in out


def test_grounding_drops_numbered_list_lead_in():
    source = _rag_chunk(
        "list_intro_source",
        "السنن الراتبة تشمل سنة الفجر والظهر والعصر والمغرب والعشاء.",
        source="fiqh",
    )
    answer = (
        "السنن الراتبة تشمل: 1. سنة الفجر والظهر والعصر "
        "والمغرب والعشاء (مصدر 1)."
    )
    out = pipeline._ground_answer(answer, [source])
    assert "تشمل: 1" not in out
    assert "سنة الفجر" in out


def test_generate_adds_citation_to_grounded_uncited_sentence():
    source = _rag_chunk(
        "uncited_generate",
        "الصلاة واجبة على المسلمين خمس مرات كل يوم",
        source="quran",
    )
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="الصلاة واجبة على المسلمين خمس مرات كل يوم."
        ))]
    )
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate("سؤال", object(), [source], client)
    assert "خمس مرات" in result["answer"]
    assert "(مصدر 1)" in result["answer"]


def test_completeness_adds_ruling_when_answer_omits_it():
    source = _rag_chunk(
        "ghusl_no_ruling",
        "أمر الشارع بالغسل وأكده. والمراد بالوجوب تأكيد استحبابه. "
        "ويخرج وقت الغسل بالفراغ من الصلاة.",
        source="fiqh",
    )
    source["reference"] = "فقه السنة - الأغسال المستحبة (1/3)"
    answer = "ويخرج وقت الغسل بالفراغ من الصلاة (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [source],
    )
    assert "مستحب" in out
    assert "امر الشارع" in pipeline._fold_ar(pipeline.normalize(out))


def test_completeness_adds_command_when_answer_only_says_mustahabb():
    source = _rag_chunk(
        "ghusl_command",
        "أمر الشارع بالغسل وأكده. والمراد بالوجوب تأكيد استحبابه. "
        "ويخرج وقت الغسل بالفراغ من الصلاة.",
        source="fiqh",
    )
    source["reference"] = "فقه السنة - الأغسال المستحبة (1/3)"
    answer = "الغسل يوم الجمعة مستحب (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم الغسل يوم الجمعة؟", answer, [source],
    )
    keys = pipeline._source_text_keys(out)
    assert pipeline._gkey("امر") in keys
    assert pipeline._gkey("اكد") in keys


def test_generate_drops_question_echoed_by_model():
    source = _rag_chunk(
        "echo_source",
        "الوتر سنة مؤكدة حث عليه الرسول ورغب فيه",
        source="fiqh",
    )
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="ما حكم صلاة الوتر؟ الوتر سنة مؤكدة (مصدر 1)."
        ))]
    )
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "ما حكم صلاة الوتر؟", object(), [source], client,
        )
    assert result["answer"].startswith("الوتر سنة مؤكدة")
    assert "ما حكم صلاة الوتر؟" not in result["answer"]


def test_generate_repairs_title_only_citation_before_returning(retrieved):
    title_only = _rag_chunk(
        "gen_title",
        "ثم تيمموا مكانا ضيقا من الخندق وخرج علي بن أبي طالب.",
        source="sira",
    )
    title_only["reference"] = "سيرة ابن هشام، غزوة الخندق في شوال سنة خمس"
    body_support = _rag_chunk(
        "gen_body",
        "ثم كانت غزوة الخندق في شوال سنة خمس.",
        source="sira",
    )
    for source in (title_only, body_support):
        source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="كانت غزوة الخندق في شوال سنة خمس (مصدر 1)."
        ))]
    )
    with patch.object(
        pipeline, "retrieve", return_value=[title_only, body_support],
    ), patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "في أي سنة كانت غزوة الخندق؟", object(),
            [title_only, body_support], client,
        )
    assert "(مصدر 2)" in result["answer"]


def test_completeness_definition_markers_do_not_apply_outside_qadr():
    source = _rag_chunk(
        "witr_blessed",
        "وقال النووي: إسناده صحيح في مسألة الوتر المبارك، "
        "وهذا مذهب جمهور العلماء.",
        source="fiqh",
    )
    answer = "الوتر سنة مؤكدة (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما هي السنن المؤكدة في الصلاة وما حكم صلاة الوتر؟",
        answer, [source],
    )
    assert out == answer
    assert "النووي" not in out


def test_prayer_quote_guard_realigns_inline_heading_to_quote_time():
    answer = (
        '**ركعتان قبل الظهر**: '
        '"رحم الله امرأ صلى قبل العصر أربعا" (مصدر 1).'
    )
    out = pipeline._fix_mismatched_prayer_quotes(answer)
    assert "**ركعتان قبل العصر**" in out
    assert "قبل الظهر" not in out
    assert "رحم الله امرأ صلى قبل العصر" in out


def test_prayer_quote_guard_drops_separate_conflicting_quote():
    answer = (
        "**سنة الظهر**: وهي أربع ركعات قبل الظهر وركعتان بعده (مصدر 1). "
        'وقد جاء في الحديث: "رحم الله امرأ صلى قبل العصر أربعا" '
        "(مصدر 1). **سنة العصر**: وهي ركعتان أو أربع قبل صلاة العصر "
        "(مصدر 1)."
    )
    out = pipeline._fix_mismatched_prayer_quotes(answer)
    assert "سنة الظهر" in out
    assert "سنة العصر" in out
    assert "رحم الله امرأ صلى قبل العصر" not in out
    assert "قبل الظهر" in out


def test_prayer_quote_guard_preserves_matching_quote():
    answer = (
        '**سنة العصر**: "رحم الله امرأ صلى قبل العصر أربعا" (مصدر 1).'
    )
    out = pipeline._fix_mismatched_prayer_quotes(answer)
    assert out == answer


def test_generate_never_pairs_cross_prayer_heading_and_quote():
    source = _rag_chunk(
        "rawatib_quote",
        "ركعتان قبل الظهر. رحم الله امرأ صلى قبل العصر أربعا.",
        source="fiqh",
    )
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=(
                '**ركعتان قبل الظهر**: '
                '"رحم الله امرأ صلى قبل العصر أربعا" (مصدر 1).'
            )
        ))]
    )
    with patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "ما هي السنن المؤكدة في الصلاة؟", object(), [source], client,
        )
    assert "**ركعتان قبل العصر**" in result["answer"]
    assert "قبل الظهر" not in result["answer"]
