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


def test_grounding_preserves_citation_at_end_without_period(retrieved):
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم (مصدر 1)"
    result = pipeline._ground_answer(answer, retrieved)
    assert "(مصدر 1)" in result
    assert result.endswith("(مصدر 1)")


def test_add_missing_citation_terminates_sentence(retrieved):
    answer = "الصلاة واجبة على المسلمين خمس مرات كل يوم"
    result = pipeline._add_missing_citations(answer, retrieved)
    assert "(مصدر 1)" in result
    assert result.endswith(".")


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
    # The source's explicit recommendation wording is "تأكيد استحبابه".
    assert "استحبابه" in out
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


def test_priority_selection_reserves_top_fused_against_oversized_priority():
    pool = [
        {"id": f"f{i}", "source": "fiqh", "score": 10.0 - i}
        for i in range(1, 11)
    ]
    selected = pipeline._select_diverse(
        pool, 8, priority_ids={f"f{i}" for i in range(5, 11)})
    ids = [c["id"] for c in selected]
    # The fused top-4 reserve must survive even though the priority set is
    # larger than the context, and the remaining slots still come from it.
    assert {"f1", "f2", "f3", "f4"} <= set(ids)
    assert len(ids) == 8
    assert any(cid in {f"f{i}" for i in range(5, 11)} for cid in ids)


def test_priority_selection_keeps_low_ranked_must_chunks():
    pool = [
        {"id": f"h{i}", "source": "hadith", "score": 10.0 - i}
        for i in range(1, 9)
    ] + [
        {"id": "sira_440", "source": "sira", "score": 0.1},
    ]
    selected = pipeline._select_diverse(
        pool, 8,
        priority_ids={f"h{i}" for i in range(1, 9)},
        must_ids={"sira_440"},
    )
    ids = {c["id"] for c in selected}
    # A must chunk at pool rank 9 cannot be starved by a larger generic
    # priority set (this was the second-Aqaba-pledge regression).
    assert "sira_440" in ids
    assert len(ids) == 8


def test_lexical_must_ids_only_for_fully_covered_query():
    lex = [{"id": "fiqh_512"}, {"id": "quran_72_26"}]
    assert pipeline._lexical_must_ids(
        lex, ["كفاره", "غيبه"], 2,
    ) == {"fiqh_512"}
    # A partial lexical match is not promoted.
    assert pipeline._lexical_must_ids(
        lex, ["كفاره", "غيبه", "اخري"], 2,
    ) == set()
    # Date questions need multi-chunk coverage and are not single-promoted.
    assert pipeline._lexical_must_ids(
        lex, ["غزوه", "بدر"], 2, asks_date=True,
    ) == set()


def test_priority_selection_keeps_lexical_must_against_priority_flood():
    pool = [
        {"id": f"h{i}", "source": "hadith", "score": 10.0 - i}
        for i in range(1, 9)
    ] + [
        {"id": "fiqh_512", "source": "fiqh", "score": 0.1},
    ]
    selected = pipeline._select_diverse(
        pool, 8,
        priority_ids={f"h{i}" for i in range(1, 9)},
        must_ids={"fiqh_512"},
    )
    assert "fiqh_512" in {c["id"] for c in selected}


def test_seerah_title_index_includes_internal_passage_heading():
    chunks = [
        {
            "id": "sira_farewell",
            "source": "sira",
            "reference": "سيرة ابن هشام، موافاة على في قفوله من اليمن رسول الله في الحج (2/5)",
            "original_text": "(خُطْبَةُ الرَّسُولِ فِي حَجَّةِ الْوَدَاعِ): قَالَ ابْنُ إسْحَاقَ",
        },
        {
            "id": "sira_other",
            "source": "sira",
            "reference": "سيرة ابن هشام، حجة الوداع (1/3)",
            "original_text": "قَالَ ابْنُ إسْحَاقَ: فَلَمَّا دَخَلَ عَلَى رَسُولِ اللَّهِ",
        },
    ]
    candidates = pipeline._seerah_title_candidates(
        "ماذا أوصى النبي في خطبة حجة الوداع؟",
        ["خطبه", "حجه", "وداع", "اوصي"],
        chunks,
    )
    assert candidates
    assert candidates[0]["id"] == "sira_farewell"


def test_heading_fallback_keeps_supported_body_without_unsupported_label():
    source = _rag_chunk(
        "farewell",
        'قال النبي صلى الله عليه وسلم: "إن دماءكم وأموالكم عليكم حرام '
        'إلى أن تلقوا ربكم".',
    )
    corpus = [source] + [
        _rag_chunk(f"farewell_bg{i}", "نص آخر للاختبار") for i in range(400)
    ]
    answer = (
        '**تحريم الأموال والدماء**: قال النبي صلى الله عليه وسلم: '
        '"إن دماءكم وأموالكم عليكم حرام إلى أن تلقوا ربكم" (مصدر 1).'
    )
    result = pipeline._ground_with_heading_fallback(
        answer, [source], corpus)
    assert "دماءكم وأموالكم" in result
    assert "**" not in result
    assert "تحريم" not in result


def test_tafsir_completeness_prepends_opening_meaning_clause():
    source = _rag_chunk(
        "tafsir_hajj",
        "يخبر تعالى أن أول بيت وضع للناس أي: لعموم الناس، لعبادتهم "
        "ونسكهم، يطوفون به ويصلون إليه.",
        source="tafsir",
    )
    answer = "الحج واجب على من استطاع إليه سبيلا (مصدر 1)."
    result = pipeline._augment_answer_completeness(
        "ما تفسير آية حج البيت من استطاع إليه سبيلا؟", answer, [source])
    assert "أول بيت وضع للناس" in result
    assert result.startswith("يخبر تعالى")


def test_qasr_completeness_appends_two_rakahs():
    source = _rag_chunk(
        "qasr",
        "المسافر يقصر الصلاة. وقال أنس: صليت الظهر بالمدينة أربعا "
        "وبذي الحليفة ركعتين.",
    )
    answer = "المسافر يقصر الصلاة إذا خرج من بلده (مصدر 1)."
    result = pipeline._augment_answer_completeness(
        "متى يقصر المسافر الصلاة؟", answer, [source])
    assert "ركعتين" in result


def test_strip_al_keeps_lexical_allah_forms():
    assert pipeline._strip_al("الله") == "الله"
    assert pipeline._strip_al("لله") == "لله"
    assert pipeline._strip_al("الرواتب") == "رواتب"
    assert pipeline._strip_al("والرواتب") == "رواتب"


def test_exact_quote_citation_is_repaired_to_the_containing_chunk():
    wrong = _rag_chunk("wrong", "نص آخر لا يحتوي الاقتباس")
    right = _rag_chunk(
        "right", 'قال: "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"')
    sentence = 'قال: "الحمد لله رب العالمين" (مصدر 1).'
    repaired = pipeline._repair_exact_quote_citation(
        sentence, [wrong, right])
    assert "(مصدر 2)" in repaired
    assert "(مصدر 1)" not in repaired


def test_grounding_accepts_verbatim_quote_with_orthographic_variants():
    source = _rag_chunk(
        "quote_ok",
        'قَالَ: "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"',
    )
    answer = 'قال: "الحمد لله رب العالمين" (مصدر 1).'
    assert pipeline._quoted_spans_exact(answer, source) is True
    assert pipeline._ground_answer(answer, [source]) != REFUSAL


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
    assert "استحبابه" in out
    assert "الشارع" in out
    assert "تأكيد استحبابه" in out
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
    assert "استحبابه" in result["answer"]
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


def test_citation_inside_quote_is_moved_after_closing_quote():
    answer = 'قال النبي ﷺ: "أي آية من كتاب الله معك أعظم (مصدر 2)؟"'
    out = pipeline._move_citations_out_of_quotes(answer)
    assert 'أعظم؟" (مصدر 2)' in out
    assert "(مصدر 2)؟\"" not in out


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
    # The source's own explicit wording is "تأكيد استحبابه"; accepting that
    # wording is more faithful than forcing the surface form "مستحب".
    assert "استحبابه" in out
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


def test_grounding_rejects_unsupported_construct_name_pair():
    """A name like "أم الفضل" needs the full idafa pair in the source.

    The source contains the common noun "فضل" (virtue), which used to satisfy
    the follower-only entity check and let a fabricated name survive.
    """
    source = {
        "id": "test-breastfeeding",
        "original_text": (
            "أرضعت امرأة النبي صلى الله عليه وسلم، وفضل الرضاع عظيم."
        ),
        "source": "hadith",
        "reference": "test",
    }
    answer = "أرضعت النبي صلى الله عليه وسلم أم الفضل (مصدر 1)."
    result = pipeline._ground_answer(answer, [source])
    assert result == REFUSAL


def test_grounding_accepts_supported_name_after_label_anchor():
    source = {
        "id": "test-halima",
        "original_text": (
            "أرضعت النبي صلى الله عليه وسلم امرأة تدعى حليمة السعدية."
        ),
        "source": "sira",
        "reference": "test",
    }
    answer = "أرضعت النبي صلى الله عليه وسلم امرأة تدعى حليمة السعدية (مصدر 1)."
    assert pipeline._ground_answer(answer, [source]) != REFUSAL


def test_grounding_rejects_unsupported_name_even_in_lenient_mode():
    source = {
        "id": "test-foster",
        "original_text": "أرضعت النبي صلى الله عليه وسلم امرأة من بني سعد.",
        "source": "sira",
        "reference": "test",
    }
    answer = "أرضعت النبي صلى الله عليه وسلم امرأة تدعى حليمة السعدية (مصدر 1)."
    assert pipeline._ground_answer(answer, [source], ignore_rare=True) == REFUSAL


def test_ordinal_label_does_not_require_matching_numeral():
    """Sequence labels are not quantities.

    "بيعة العقبة الثانية" and the source's "بيعة العقبة الأخيرة" name the
    same event; the strict digit guard must still verify the real count (73)
    without demanding a literal "2" from the source.
    """
    source = {
        "id": "test-aqaba",
        "original_text": (
            "وكانوا ثلاثة وسبعين رجلا وامرأتين في بيعة العقبة الأخيرة."
        ),
        "source": "sira",
        "reference": "أمر العقبة الثانية (4/12)",
    }
    answer = (
        'كان عدد من بايع النبي في بيعة العقبة الثانية '
        '"ثلاثة وسبعين رجلاً وامرأتين" (مصدر 1).'
    )
    assert pipeline._ground_answer(answer, [source]) != REFUSAL


def test_quote_fallback_recovers_supported_paraphrase():
    """A paraphrase in quotation marks must not discard a supported claim."""
    source = {
        "id": "test-tayammum",
        "original_text": (
            "سقط عقد عائشة فتخلفت لالتماسه فانطلق أبو بكر إليها "
            "فتغيظ عليها في حبسها الناس فأنزل الله الرخصة في التيمم."
        ),
        "source": "hadith",
        "reference": "test",
    }
    answer = (
        'شرع التيمم عندما "سقط عقد عائشة فتخلفت للالتماس، '
        'فانطلق أبو بكر إليها فغضب عليها في حبسها الناس، '
        'فأنزل الله الرخصة في التيمم" (مصدر 1).'
    )
    out = pipeline._ground_with_heading_fallback(answer, [source], None)
    assert out != REFUSAL
    assert "التيمم" in out


def test_tayammum_story_completeness_restores_lost_necklace_clause():
    story = _rag_chunk(
        "tayammum_story",
        "خرجنا مع عائشة في بعض الأسفار، حتى إذا كنا بالبيداء "
        "انقطع عقد لي، وليسوا على ماء.",
        source="hadith",
    )
    disease = _rag_chunk(
        "tayammum_disease",
        "ويجوز التيمم إذا كان به جراحة أو مرض وخاف من استعمال الماء.",
        source="fiqh",
    )
    answer = (
        "أقام النبي صلى الله عليه وسلم على التماس العقد، وليسوا على ماء، "
        "فأنزل الله آية التيمم (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "كيف شرع التيمم في سفر عائشة؟", answer, [story, disease],
    )
    # A narrative tayammum question needs the initiating event; merely
    # "التماس العقد" (searching for it) does not say the necklace was lost.
    # The disease cause belongs to the ruling question and must not be mixed.
    assert "انقطع عقد" in out
    assert "جراحة" not in out


def test_tayammum_story_refusal_recovers_source_narrative():
    source = _rag_chunk(
        "tayammum_full_story",
        "خرجنا مع عائشة في بعض الأسفار حتى إذا كنا بالبيداء انقطع عقد لي، "
        "وأقام الناس معه وليسوا على ماء، فأنزل الله تعالى آية التيمم.",
        source="fiqh",
    )
    out = pipeline._augment_answer_completeness(
        "كيف شرع التيمم في سفر عائشة؟", REFUSAL, [source],
    )
    assert out != REFUSAL
    assert "انقطع عقد" in out
    assert "ليسوا على ماء" in out
    assert "آية التيمم" in out


def test_quote_fallback_keeps_rejecting_unsupported_claim():
    source = {
        "id": "test-unrelated",
        "original_text": "الصلاة عمود الإسلام.",
        "source": "hadith",
        "reference": "test",
    }
    answer = '"سقط عقد عائشة فأنزل الله الرخصة في التيمم" (مصدر 1).'
    assert pipeline._ground_with_heading_fallback(
        answer, [source], None,
    ) == REFUSAL


def _quran_topic_chunks():
    return [
        {
            "id": "quran_5_6", "source": "quran", "reference": "المائدة:6",
            "original_text": "فاغسلوا وجوهكم وأيديكم إلى المرافق.",
        },
        {
            "id": "quran_2_275", "source": "quran", "reference": "البقرة:275",
            "original_text": "وأحل الله البيع وحرم الربا.",
        },
        {
            "id": "quran_17_23", "source": "quran", "reference": "الإسراء:23",
            "original_text": "وبالوالدين إحسانا.",
        },
    ]


def test_quran_topic_anchor_wudu():
    chunks = _quran_topic_chunks()
    anchor = pipeline._resolve_quran_topic_anchor(
        "كيف يتوضأ المسلم قبل الصلاة؟", chunks,
    )
    assert anchor is not None and anchor["id"] == "quran_5_6"


def test_quran_topic_anchor_riba_and_parents():
    chunks = _quran_topic_chunks()
    assert pipeline._resolve_quran_topic_anchor(
        "ما حكم الربا في القرآن الكريم؟", chunks,
    )["id"] == "quran_2_275"
    assert pipeline._resolve_quran_topic_anchor(
        "كيف أمر الله بالإحسان إلى الوالدين؟", chunks,
    )["id"] == "quran_17_23"


def test_quran_topic_anchor_excluded_for_tafsir_and_unrelated():
    chunks = _quran_topic_chunks()
    assert pipeline._resolve_quran_topic_anchor(
        "ما تفسير آية الربا؟", chunks,
    ) is None
    assert pipeline._resolve_quran_topic_anchor(
        "كيف أستخدم بايثون لتحليل البيانات؟", chunks,
    ) is None


def test_quran_topic_anchor_excludes_arafah_fasting():
    chunks = [
        _rag_chunk(
            "quran_2_183",
            "كتب عليكم الصيام كما كتب على الذين من قبلكم",
            source="quran",
        ),
    ]
    assert pipeline._resolve_quran_topic_anchor(
        "ما حكم صيام يوم عرفة؟", chunks,
    ) is None
    assert pipeline._resolve_quran_topic_anchor(
        "لماذا فرض الله الصيام على المؤمنين؟", chunks,
    )["id"] == "quran_2_183"


def test_core_hadith_anchor_eight_gates():
    chunks = [_rag_chunk(
        "hadith_darimi_39619",
        "للجنة ثمانية أبواب",
        source="hadith",
    )]
    anchor = pipeline._resolve_core_hadith_anchor(
        "كم عدد أبواب الجنة؟", chunks,
    )
    assert anchor is not None and anchor["id"] == "hadith_darimi_39619"


@pytest.mark.parametrize("followup", [
    "أيوة يعني حلال ولا حرام؟",
    "ليه؟",
    "ممكن توضح أكتر؟",
    "طب إيه الحكم في الحالة دي؟",
    "والرأي التاني إيه؟",
])
def test_history_aware_query_covers_egyptian_and_msa_followups(followup):
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {"role": "assistant", "content": "الحكم مبني على حديث إعفاء اللحية."},
    ]
    expanded = pipeline._history_aware_query(followup, history)
    # Pure clarifications retrieve the prior substantive question only;
    # generic polarity/discourse words must not dilute the query.
    assert expanded == "حكم اللحية؟"


@pytest.mark.parametrize("question", [
    "ما حكم الربا؟",
    "كيف يتوضأ المسلم؟",
    "ما هي شروط وجوب الحج؟",
    "من هو الرسول؟",
    "ما هو القرآن؟",
])
def test_history_aware_query_does_not_hijack_new_topic(question):
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {"role": "assistant", "content": "الحكم مبني على حديث إعفاء اللحية."},
    ]
    assert pipeline._history_aware_query(question, history) == question


def test_history_aware_query_resolves_attached_pronoun_without_conjunction():
    history = [
        {"role": "user", "content": "ما حكم صيام يوم عرفة؟"},
        {"role": "assistant", "content": "صيام يوم عرفة مستحب لغير الحاج."},
    ]
    expanded = pipeline._history_aware_query("ما حكمه للحاج؟", history)
    assert expanded.startswith("ما حكم صيام يوم عرفة؟")
    assert "ما حكمه للحاج؟" in expanded


def test_history_aware_query_resolves_demonstrative_with_qualifier():
    history = [
        {"role": "user", "content": "ما حكم صيام يوم عرفة؟"},
        {"role": "assistant", "content": "صيام يوم عرفة مستحب لغير الحاج."},
    ]
    expanded = pipeline._history_aware_query("هل هذا الحديث صحيح؟", history)
    assert expanded.startswith("ما حكم صيام يوم عرفة؟")
    assert "هل هذا الحديث صحيح؟" in expanded


def test_history_aware_query_walks_back_past_followups():
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {"role": "assistant", "content": "الحكم مبني على حديث إعفاء اللحية."},
        {"role": "user", "content": "أيوة يعني حلال ولا حرام؟"},
        {"role": "assistant", "content": "الإعفاء هو الحكم."},
    ]
    expanded = pipeline._history_aware_query("ليه؟", history)
    assert expanded == "حكم اللحية؟"


def test_history_aware_query_evidence_request_targets_previous_claim():
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {
            "role": "assistant",
            "content": (
                "النهي عن صيام يوم الجمعة منفردًا للكراهة لا للتحريم "
                "(مصدر 1). ومن وافق عادة له فلا يكره صيامه (مصدر 1)."
            ),
        },
    ]
    expanded = pipeline._history_aware_query("إيه الدليل؟", history)
    assert "عادة" in expanded
    assert "صيام" in expanded


def test_history_aware_query_handles_malformed_history():
    assert pipeline._history_aware_query("ليه؟", []) == "ليه؟"
    assert pipeline._history_aware_query("ليه؟", None) == "ليه؟"
    assert pipeline._history_aware_query("ليه؟", [None, {}, {"role": "user"}]) == "ليه؟"


def test_history_aware_query_expands_short_followup():
    history = [
        {"role": "user", "content": "ما حكم صيام يوم عرفة؟"},
        {"role": "assistant", "content": "صيام يوم عرفة مستحب لغير الحاج."},
    ]
    expanded = pipeline._history_aware_query("وما حكمه للحاج؟", history)
    assert expanded.startswith("ما حكم صيام يوم عرفة؟")
    assert "وما حكمه للحاج؟" in expanded


def test_history_aware_query_leaves_standalone_question_unchanged():
    history = [
        {"role": "user", "content": "ما حكم صيام يوم عرفة؟"},
    ]
    question = "ما هي شروط وجوب الحج ومن يجب عليه؟"
    assert pipeline._history_aware_query(question, history) == question
    assert pipeline._history_aware_query("وما حكمه؟", None) == "وما حكمه؟"


def test_retrieve_rewrites_and_boosts_resolved_followup_topic(monkeypatch):
    rewritten = []
    boosted = []
    monkeypatch.setattr(
        pipeline, "_resolve_chat_tafsir", lambda q, i, c, cl: ([], ""),
    )
    monkeypatch.setattr(pipeline, "_resolve_surah_query", lambda q, c: None)
    monkeypatch.setattr(
        pipeline, "_resolve_quran_topic_anchor", lambda q, c: None,
    )
    monkeypatch.setattr(
        pipeline, "_resolve_core_hadith_anchor", lambda q, c: None,
    )
    monkeypatch.setattr(pipeline, "embed_text", lambda text, client: [0.1])
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_bm25_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        pipeline, "_rewrite_query",
        lambda q, client: rewritten.append(q) or [],
    )
    monkeypatch.setattr(
        pipeline, "_apply_lexical_boost",
        lambda pool, q: boosted.append(q) or pool,
    )
    pipeline.retrieve(
        "ليه؟", object(), [_rag_chunk("topic", "نص فقهي")], Mock(),
        retrieval_query="ما حكم الربا؟",
    )
    assert rewritten == ["ما حكم الربا؟"]
    assert boosted == ["ما حكم الربا؟"]


def test_retrieve_embeds_resolved_followup_topic(monkeypatch):
    texts = []
    monkeypatch.setattr(
        pipeline, "_resolve_chat_tafsir", lambda q, i, c, cl: ([], ""),
    )
    monkeypatch.setattr(pipeline, "_resolve_surah_query", lambda q, c: None)
    monkeypatch.setattr(
        pipeline, "_resolve_quran_topic_anchor", lambda q, c: None,
    )
    monkeypatch.setattr(
        pipeline, "_resolve_core_hadith_anchor", lambda q, c: None,
    )
    monkeypatch.setattr(
        pipeline, "embed_text",
        lambda text, client: texts.append(text) or [0.1],
    )
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_bm25_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_rewrite_query", lambda q, client: [])
    pipeline.retrieve(
        "ليه؟", object(), [_rag_chunk("topic", "نص فقهي")], Mock(),
        retrieval_query="ما حكم الربا؟",
    )
    assert len(texts) == 2
    assert all("الربا" in text for text in texts)
    assert all("ليه" not in text for text in texts)


def test_followup_fallback_prefers_reason_sentence_for_why_question():
    source = _rag_chunk(
        "riba-reason-source",
        "والسبب في تحريمه ما فيه من ضرر عظيم، وأنه يسبب العداوة بين الأفراد.",
        source="fiqh",
    )
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم الربا؟"},
        {
            "role": "assistant",
            "content": (
                "حكم الربا هو التحريم (مصدر 1). "
                "والسبب في تحريمه ما فيه من ضرر عظيم، "
                "وأنه يسبب العداوة بين الأفراد (مصدر 1)."
            ),
        },
    ]
    fallback = pipeline._followup_fallback_answer(
        "ليه؟", history, [source], [source],
    )
    assert fallback is not None
    assert "السبب" in fallback
    assert "العداوة" in fallback
    assert pipeline._answer_citation_support_ok(
        fallback, [source], threshold=0.65,
    )


def test_retrieve_uses_resolved_query_for_topic_anchors(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pipeline, "_resolve_chat_tafsir", lambda q, i, c, cl: ([], ""),
    )
    monkeypatch.setattr(pipeline, "_resolve_surah_query", lambda q, c: None)
    monkeypatch.setattr(
        pipeline, "_resolve_quran_topic_anchor",
        lambda q, c: seen.append(q) or None,
    )
    monkeypatch.setattr(
        pipeline, "_resolve_core_hadith_anchor",
        lambda q, c: seen.append(q) or None,
    )
    monkeypatch.setattr(pipeline, "embed_text", lambda text, client: [0.1])
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_bm25_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_rewrite_query", lambda q, client: [])
    pipeline.retrieve(
        "ليه؟", object(), [_rag_chunk("topic", "نص فقهي")], Mock(),
        retrieval_query="ما حكم الربا؟",
    )
    assert seen
    assert all(query == "ما حكم الربا؟" for query in seen)


def test_generate_resolves_tafsir_from_history_aware_query():
    seen = []

    def fake_resolve(query, index, chunks, client):
        seen.append(query)
        return [], ""

    source = _rag_chunk(
        "riba-source", "الربا محرم في الإسلام.", source="fiqh",
    )
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="الربا محرم في الإسلام (مصدر 1)."
        ))]
    )
    history = [
        {"role": "user", "content": "ما حكم الربا؟"},
        {"role": "assistant", "content": "الربا محرم (مصدر 1)."},
    ]
    with patch.object(pipeline, "_resolve_chat_tafsir", fake_resolve), \
         patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "ليه؟", object(), [source], client, chat_history=history,
        )
    assert result["answer"]
    assert seen == ["ما حكم الربا؟"]


def test_generate_passes_history_aware_query_to_retrieval():
    source = _rag_chunk(
        "history-source",
        "صيام يوم عرفة مستحب لغير الحاج.",
        source="fiqh",
    )
    source.update(score=1.0, _gate_passed=True)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="صيام يوم عرفة مستحب لغير الحاج (مصدر 1)."
        ))]
    )
    history = [
        {"role": "user", "content": "ما حكم صيام يوم عرفة؟"},
        {"role": "assistant", "content": "مستحب لغير الحاج."},
    ]
    with patch.object(pipeline, "retrieve", return_value=[source]) as retrieve_mock, \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "وما حكمه للحاج؟", object(), [source], client,
            chat_history=history,
        )
    assert result["answer"]
    retrieval_query = retrieve_mock.call_args.kwargs.get("retrieval_query")
    assert retrieval_query.startswith("ما حكم صيام يوم عرفة؟")
    assert "وما حكمه للحاج؟" in retrieval_query


def test_build_prompt_keeps_history_and_current_question_single():
    chunk = _rag_chunk("prompt-source", "الصيام عبادة مشروعة.")
    history = [
        {"role": "user", "content": "سؤال سابق"},
        {"role": "assistant", "content": "جواب سابق"},
    ]
    messages = pipeline.build_prompt(
        "سؤال حالي", [chunk], chat_history=history,
    )
    assert messages[0]["role"] == "system"
    assert any(
        message["role"] == "user" and message["content"] == "سؤال سابق"
        for message in messages
    )
    assert any(
        message["role"] == "assistant" and message["content"] == "جواب سابق"
        for message in messages
    )
    final_user = messages[-1]
    assert final_user["role"] == "user"
    assert final_user["content"].count("سؤال حالي") == 1
    assert "سؤال سابق" not in final_user["content"]


def test_followup_fallback_reuses_prior_grounded_ruling():
    source = _rag_chunk(
        "beard-source",
        "أمر النبي صلى الله عليه وسلم بإحفاء الشوارب وإعفاء اللحية.",
        source="hadith",
    )
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {
            "role": "assistant",
            "content": (
                "حكم اللحية هو الإعفاء، وقد ورد في الحديث "
                "أن النبي صلى الله عليه وسلم أمر بإعفاء اللحية (مصدر 1)."
            ),
        },
    ]
    fallback = pipeline._followup_fallback_answer(
        "أيوة يعني حلال ولا حرام؟", history, [source], [source],
    )
    assert fallback is not None
    assert "الإعفاء" in fallback
    assert "(مصدر 1)" in fallback
    assert pipeline._answer_citation_support_ok(fallback, [source])


def test_followup_fallback_does_not_fire_for_unrelated_question():
    source = _rag_chunk(
        "beard-source",
        "أمر النبي صلى الله عليه وسلم بإعفاء اللحية.",
        source="hadith",
    )
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {"role": "assistant", "content": "حكم اللحية هو الإعفاء (مصدر 1)."},
    ]
    assert pipeline._followup_fallback_answer(
        "ما حكم الربا؟", history, [source], [source],
    ) is None


FRIDAY_FIQH_TEXT = (
    "النهي عن صوم يوم الجمعة منفردا: ذهب الجمهور إلى أن النهي "
    "للكراهة لا للتحريم، إلا إذا صام يوما قبله أو يوما بعده، أو "
    "وافق عادة له، أو كان يوم عرفة أو عاشوراء، فإنه حينئذ لا يكره "
    "صيامه."
)
FRIDAY_PRIOR_ANSWER = (
    "حكم صيام يوم الجمعة هو النهي عن صيامه منفردًا (مصدر 1). "
    "الجمهور من العلماء يرون أن النهي للكراهة لا للتحريم، إلا إذا "
    "صام يومًا قبله أو يومًا بعده، أو وافق عادة له، أو كان يوم "
    "عرفة أو عاشوراء، فإنه حينئذ لا يكره صيامه (مصدر 1)."
)


def _generate_with_mock(
    query, source, answer, history=None, retrieve_mock=None,
):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    with patch.object(
        pipeline, "_resolve_chat_tafsir", return_value=([], ""),
    ), patch.object(
        pipeline, "retrieve", return_value=[source],
    ) as mocked, patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            query, object(), [source], client, chat_history=history,
        )
    if retrieve_mock is not None:
        retrieve_mock.append(mocked.call_args.kwargs.get("retrieval_query"))
    return result


def test_legal_modality_extraction_distinguishes_families():
    assert "obligation" in pipeline._positive_legal_families(
        "فعليه أن يصوم"
    )
    assert "obligation" not in pipeline._positive_legal_families("لا يجب")
    assert "prohibition" in pipeline._positive_legal_families("لا يجوز")
    assert "prohibition" not in pipeline._positive_legal_families("يكره")


def test_friday_followup_rejects_unconditional_obligation():
    source = _rag_chunk("friday_fiqh", FRIDAY_FIQH_TEXT)
    source["reference"] = "فقه السنة، سيد سيد سابق، الصيام - الأيام المنهي عن صيامها (1/3)"
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {"role": "assistant", "content": FRIDAY_PRIOR_ANSWER},
    ]
    seen = []
    result = _generate_with_mock(
        "يعني لازم أصوم يوم قبله أو بعده؟",
        source,
        "نعم، يجب صيام يوم الجمعة مع يوم قبله أو يوم بعده (مصدر 1).",
        history=history,
        retrieve_mock=seen,
    )
    answer = result["answer"]
    assert seen == ["ما حكم صيام الجمعة؟"]
    assert "لا توجب" in answer
    assert "الكراهة" in answer or "لا يكره" in answer
    assert "obligation" not in pipeline._positive_legal_families(answer)
    assert "(مصدر 1)" in answer


def test_day_qualified_fast_filters_prayer_and_general_ramadan_verses():
    prayer = _rag_chunk(
        "quran_62_9",
        "يا أيها الذين آمنوا إذا نودي للصلاة من يوم الجمعة فاسعوا إلى ذكر الله",
        source="quran",
    )
    ramadan = _rag_chunk(
        "quran_2_183",
        "يا أيها الذين آمنوا كتب عليكم الصيام كما كتب على الذين من قبلكم",
        source="quran",
    )
    fiqh = _rag_chunk("friday_fiqh_filter", FRIDAY_FIQH_TEXT)
    kept = pipeline._filter_joint_evidence(
        [prayer, ramadan, fiqh], "ما حكم صيام الجمعة؟",
    )
    assert [c["id"] for c in kept] == ["friday_fiqh_filter"]
    assert pipeline._filter_joint_evidence(
        [ramadan], "ما حكم صيام رمضان؟",
    )
    assert pipeline._filter_joint_evidence(
        [prayer], "ما حكم صلاة الجمعة؟",
    )
    assert not pipeline._filter_joint_evidence(
        [prayer], "ما حكم صيام الخميس؟",
    )


def test_quran_friday_prayer_not_anchored_for_fasting_question():
    prayer = _rag_chunk(
        "quran_62_9",
        "يا أيها الذين آمنوا إذا نودي للصلاة من يوم الجمعة فاسعوا إلى ذكر الله",
        source="quran",
    )
    prayer["reference"] = "الجمعة:9"
    assert pipeline._resolve_quran_topic_anchor(
        "ما حكم صيام الجمعة؟", [prayer],
    ) is None
    assert pipeline._resolve_quran_topic_anchor(
        "ما حكم صلاة الجمعة؟", [prayer],
    ) is not None


def test_friday_first_answer_keeps_fiqh_qualification():
    source = _rag_chunk("friday_full", FRIDAY_FIQH_TEXT)
    source.update(score=1.0, _gate_passed=True)
    answer = (
        "وذهب الجمهور إلى أن النهي للكراهة لا للتحريم، إلا إذا صام "
        "يوما قبله أو يوما بعده، أو وافق عادة له (مصدر 1)."
    )
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    with patch.object(
        pipeline, "_resolve_chat_tafsir", return_value=([], ""),
    ), patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "ما حكم صيام الجمعة؟", object(), [source], client,
        )
    assert "كراه" in result["answer"]
    assert "عادة" in result["answer"]


def test_habit_followup_uses_habit_exception_without_condition():
    source = _rag_chunk("friday_habit", FRIDAY_FIQH_TEXT)
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {"role": "assistant", "content": "لا يجب صيام يوم الجمعة (مصدر 1)."},
        {"role": "user", "content": "يعني لازم أصوم يوم قبله أو بعده؟"},
        {"role": "assistant", "content": "المصادر لا توجب ذلك (مصدر 1)."},
    ]
    out = pipeline._followup_fallback_answer(
        "طب لو أنا متعود أصوم يوم الجمعة؟",
        history, [source], [source],
    )
    assert out is not None
    assert "عادة" in out
    assert "بشرط" not in out
    assert "obligation" not in pipeline._positive_legal_families(out)


def test_unsupported_condition_is_not_grounded():
    source = _rag_chunk(
        "friday_condition",
        "أو وافق عادة له، أو كان يوم عرفة أو عاشوراء، فإنه حينئذ لا يكره صيامه.",
    )
    answer = (
        "إذا كنت متعودًا على صيام يوم الجمعة، فلا حرج في ذلك بشرط "
        "أن تصوم قبله يومًا أو بعده يومًا (مصدر 1)."
    )
    result = pipeline._ground_answer(answer, [source], [source])
    assert result == REFUSAL


def test_evidence_request_returns_hadith_plus_habit_evidence():
    source = _rag_chunk(
        "friday_evidence",
        'وفي الصحيحين من حديث جابر رضي الله عنه أن النبي صلى الله '
        'عليه وسلم قال: "لا تصوموا يوم الجمعة، إلا وقبله يوم، أو '
        'بعده يوم". وذهب الجمهور إلى أن النهي للكراهة لا للتحريم '
        'إلا إذا صام يوما قبله أو يوما بعده، أو وافق عادة له.',
    )
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {"role": "assistant", "content": "لا يجب صيام يوم الجمعة (مصدر 1)."},
        {"role": "user", "content": "طب لو أنا متعود أصوم يوم الجمعة؟"},
        {
            "role": "assistant",
            "content": (
                "وذهب الجمهور إلى أن النهي للكراهة لا للتحريم إلا "
                "إذا صام يوما قبله أو يوما بعده، أو وافق عادة له "
                "(مصدر 1)."
            ),
        },
    ]
    out = pipeline._followup_fallback_answer(
        "إيه الدليل؟", history, [source], [source],
    )
    assert out is not None
    assert "الصحيحين" in out or "قال" in out
    assert "عادة" in out
    assert "مصدر 1" in out


def test_evidence_request_rejects_unrelated_tashreeq_hadith():
    source = _rag_chunk(
        "friday_and_tashreeq",
        "وروى الطبراني أن لا تصوموا هذه الأيام فإنها أيام أكل وشرب. "
        "النهي عن صوم يوم الجمعة منفردا: وذهب الجمهور إلى أن النهي "
        "للكراهة لا للتحريم إلا إذا صام يوما قبله أو يوما بعده، أو "
        "وافق عادة له. وفي الصحيحين من حديث جابر أن النبي صلى الله "
        'عليه وسلم قال: "لا تصوموا يوم الجمعة، إلا وقبله يوم، أو '
        'بعده يوم".',
    )
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "يعني لازم أصوم يوم قبله أو بعده؟"},
        {"role": "assistant", "content": "المصادر لا توجب ذلك (مصدر 1)."},
        {"role": "user", "content": "طب لو أنا متعود أصوم يوم الجمعة؟"},
        {
            "role": "assistant",
            "content": (
                "وذهب الجمهور إلى أن النهي للكراهة لا للتحريم إلا "
                "إذا صام يوما قبله أو يوما بعده، أو وافق عادة له "
                "(مصدر 1)."
            ),
        },
    ]
    answer = (
        "وروى الطبراني أن لا تصوموا هذه الأيام فإنها أيام أكل وشرب "
        "(مصدر 1)."
    )
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    with patch.object(
        pipeline, "_resolve_chat_tafsir", return_value=([], ""),
    ), patch.object(pipeline, "retrieve", return_value=[source]), \
         patch.object(pipeline, "build_prompt", return_value=[]):
        result = pipeline.generate(
            "إيه الدليل؟", object(), [source], client,
            chat_history=history,
        )
    assert "أيام التشريق" not in result["answer"]
    assert (
        "الصحيحين" in result["answer"]
        or "لا تصوموا يوم الجمعة" in result["answer"]
    )
    assert "عادة" in result["answer"]


def test_ordinary_dislike_verb_is_not_a_legal_modality():
    assert "dislike" not in pipeline._positive_legal_families(
        "تعريف الغيبة هو ذكرك أخاك بما يكره"
    )
    assert "dislike" in pipeline._positive_legal_families(
        "يكره صيام يوم الجمعة منفردا"
    )


def test_condition_only_source_supports_travel_completeness():
    source = _rag_chunk(
        "travel_condition",
        "ذهب جمهور العلماء إلى أن قصر الصلاة يشرع بمفارقة الحضر "
        "والخروج من البلد وأن ذلك شرط ولا يتم حتى يدخل أو بيوتها.",
    )
    answer = (
        "المسافر يقصر الصلاة ما دام مسافرا، حتى لو أقام لحاجة "
        "ينتظر قضاءها (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "متى يقصر المسافر الصلاة؟", answer, [source],
    )
    assert "مفارقة الحضر" in out


def test_condition_list_completeness_includes_all_items():
    source = _rag_chunk(
        "hajj_conditions",
        "اتفق الفقهاء على أنه يشترط لوجوب الحج، الشرط الاتية: "
        "١ - الاسلام. ٢ - البلوغ. ٣ - العقل. ٤ - الحرية. "
        "٥ - الاستطاعة. فمن لم تتحقق فيه هذه الشروط، فلا يجب عليه الحج.",
    )
    answer = (
        "**الإسلام**: فلا يجب الحج على غير المسلم (مصدر 1). "
        "**البلوغ**: فلا يجب الحج على الصبي حتى يبلغ (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "ما شروط وجوب الحج؟", answer, [source],
    )
    assert "العقل" in out
    assert "الاستطاعة" in out


def test_tayammum_narrative_adds_missing_water_cause():
    source = _rag_chunk(
        "tayammum_water",
        "خرجنا مع النبي صلى الله عليه وسلم في بعض أسفاره حتى إذا كنا "
        "بالبيداء انقطع عقد لي، فأقام النبي صلى الله عليه وسلم على "
        "التماسه، وأقام الناس معه، وليسوا على ماء، وليس معهم ماء، "
        "فأنزل الله آية التيمم.",
        source="fiqh",
    )
    answer = (
        "خرجنا مع النبي صلى الله عليه وسلم في بعض أسفاره حتى إذا كنا "
        "بالبيداء انقطع عقد لي (مصدر 1). فأنزل الله آية التيمم (مصدر 1)."
    )
    out = pipeline._augment_answer_completeness(
        "كيف شرع التيمم في سفر عائشة؟", answer, [source],
    )
    assert "ليسوا على ماء" in out or "ليس معهم ماء" in out


def test_obligation_followup_after_recommendation_stays_non_obligatory():
    source = _rag_chunk("monday_fast", "صيام يوم الاثنين مستحب.")
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم صيام يوم الاثنين؟"},
        {"role": "assistant", "content": "صيام يوم الاثنين مستحب (مصدر 1)."},
    ]
    result = _generate_with_mock(
        "هل ده واجب؟", source,
        "نعم، يجب صيامه (مصدر 1).", history=history,
    )
    answer = result["answer"]
    assert "لا توجب" in answer
    assert "مستحب" in answer
    assert "obligation" not in pipeline._positive_legal_families(answer)


def test_prohibition_followup_preserves_qualified_dislike():
    source = _rag_chunk(
        "friday_dislike", "إفراد يوم الجمعة بالصوم مكروه عند الجمهور.",
    )
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم إفراد الجمعة بالصوم؟"},
        {
            "role": "assistant",
            "content": "إفراد يوم الجمعة بالصوم مكروه عند الجمهور (مصدر 1).",
        },
    ]
    result = _generate_with_mock(
        "يعني حرام؟", source, "نعم، حرام (مصدر 1).", history=history,
    )
    answer = result["answer"]
    assert "لا تحرم" in answer
    assert "مكروه" in answer
    assert "prohibition" not in pipeline._positive_legal_families(answer)


def test_disputed_followup_does_not_present_one_opinion_as_consensus():
    source = _rag_chunk(
        "friday_dispute",
        "إفراد يوم الجمعة بالصوم مكروه عند الجمهور، وذهب آخرون إلى تحريمه.",
    )
    source.update(score=1.0, _gate_passed=True)
    prior = (
        "إفراد يوم الجمعة بالصوم مكروه عند الجمهور، وذهب آخرون إلى "
        "تحريمه (مصدر 1)."
    )
    history = [
        {"role": "user", "content": "ما حكم إفراد الجمعة بالصوم؟"},
        {"role": "assistant", "content": prior},
    ]
    result = _generate_with_mock(
        "يعني حرام؟", source, "نعم، حرام (مصدر 1).", history=history,
    )
    answer = result["answer"]
    # The answer may state the disagreement directly instead of using the
    # generic lead, but it must not flatten the issue to "نعم، حرام".
    assert "آخرون" in answer or "المسألة محل خلاف" in answer
    assert "مكروه" in answer and "تحريم" in answer
    assert "نعم، حرام" not in answer


def test_hadith_wording_does_not_establish_obligation_by_itself():
    source = _rag_chunk(
        "friday_hadith",
        'قال رسول الله صلى الله عليه وسلم: "لا تصوموا يوم الجمعة '
        'إلا وقبله يوم أو بعده يوم".',
        source="hadith",
    )
    source["reference"] = "صحيح مسلم:2568"
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {
            "role": "assistant",
            "content": (
                'قال رسول الله صلى الله عليه وسلم: "لا تصوموا يوم '
                'الجمعة إلا وقبله يوم أو بعده يوم" (مصدر 1).'
            ),
        },
    ]
    result = _generate_with_mock(
        "يعني لازم أصوم يوم قبله أو بعده؟", source,
        "نعم، يجب صيام يوم قبله أو بعده (مصدر 1).",
        history=history,
    )
    answer = result["answer"]
    assert "لا توجب" in answer
    assert "لا تصوموا يوم الجمعة" in answer
    assert "obligation" not in pipeline._positive_legal_families(answer)


def test_explicit_obligation_is_not_made_hesitant():
    source = _rag_chunk("ramadan_obligation", "صيام رمضان واجب على كل مسلم بالغ.")
    source.update(score=1.0, _gate_passed=True)
    result = _generate_with_mock(
        "هل يجب صيام رمضان؟", source,
        "نعم، يجب صيام رمضان (مصدر 1).",
    )
    answer = result["answer"]
    assert "obligation" in pipeline._positive_legal_families(answer)
    assert "لا توجب" not in answer


def test_unrelated_new_question_does_not_inherit_legal_lead():
    source = _rag_chunk("ramadan_obligation2", "صيام رمضان واجب على كل مسلم بالغ.")
    source.update(score=1.0, _gate_passed=True)
    history = [
        {"role": "user", "content": "ما حكم صيام الجمعة؟"},
        {"role": "assistant", "content": FRIDAY_PRIOR_ANSWER},
    ]
    seen = []
    result = _generate_with_mock(
        "ما حكم صيام رمضان؟", source,
        "صيام رمضان واجب على كل مسلم بالغ (مصدر 1).",
        history=history, retrieve_mock=seen,
    )
    answer = result["answer"]
    assert seen == ["ما حكم صيام رمضان؟"]
    assert "لا توجب" not in answer
    assert "obligation" in pipeline._positive_legal_families(answer)


def test_citation_alignment_for_hadith_quote_and_fiqh_interpretation():
    hadith = _rag_chunk(
        "friday_quote_hadith",
        'قال رسول الله صلى الله عليه وسلم: "لا تصوموا يوم الجمعة '
        'إلا وقبله يوم أو بعده يوم".',
        source="hadith",
    )
    hadith["reference"] = "صحيح مسلم:2568"
    fiqh = _rag_chunk(
        "friday_interpretation",
        "النهي عن إفراد الجمعة بالصوم للكراهة لا للتحريم.",
    )
    answer = (
        'قال النبي صلى الله عليه وسلم: "لا تصوموا يوم الجمعة إلا '
        'وقبله يوم أو بعده يوم". النهي عن إفراد الجمعة بالصوم للكراهة '
        'لا للتحريم.'
    )
    out = pipeline._add_missing_citations(answer, [hadith, fiqh])
    assert out.index("(مصدر 1)") < out.index("(مصدر 2)")
    assert "لا للتحريم (مصدر 2)" in out


def test_build_prompt_adds_followup_context_note():
    chunk = _rag_chunk("followup-prompt-source", "إعفاء اللحية ورد في الحديث.")
    history = [
        {"role": "user", "content": "حكم اللحية؟"},
        {"role": "assistant", "content": "الحكم هو الإعفاء."},
    ]
    messages = pipeline.build_prompt(
        "أيوة يعني حلال ولا حرام؟", [chunk], chat_history=history,
    )
    final_user = messages[-1]["content"]
    assert "ملاحظة سياق" in final_user
    assert "حكم اللحية؟" in final_user
    assert final_user.count("أيوة يعني حلال ولا حرام؟") == 1


def test_core_hadith_anchor_names_of_allah():
    chunks = [_rag_chunk(
        "hadith_bukhari_2624",
        "إن لله تسعة وتسعين اسما مائة إلا واحدا من أحصاها دخل الجنة",
        source="hadith",
    )]
    anchor = pipeline._resolve_core_hadith_anchor(
        "كم عدد أسماء الله الحسنى؟", chunks,
    )
    assert anchor is not None and anchor["id"] == "hadith_bukhari_2624"


def _quran_anchor_chunks():
    verse = (
        "وقضى ربك ألا تعبدوا إلا إياه وبالوالدين إحسانا "
        "فلا تقل لهما أف ولا تنهرهما وقل لهما قولا كريما."
    )
    return [{
        "id": "quran_17_23", "source": "quran", "reference": "الإسراء:23",
        "original_text": verse,
        "metadata": {"surah_name": "الإسراء", "ayah_number": 23},
        "_injected": True,
    }]


def test_quran_anchor_completeness_recovers_false_refusal():
    chunks = _quran_anchor_chunks()
    out = pipeline._augment_answer_completeness(
        "كيف أمر الله بالإحسان إلى الوالدين؟", REFUSAL, chunks,
    )
    assert out != REFUSAL
    assert "وبالوالدين إحسانا" in out
    assert "(مصدر 1)" in out


def test_quran_anchor_completeness_adds_missing_verse_clause():
    chunks = _quran_anchor_chunks()
    partial = (
        'أمر الله بالإحسان إلى الوالدين في قوله: '
        '"وقضى ربك ألا تعبدوا إلا إياه وبالوالدين إحسانا" (مصدر 1).'
    )
    out = pipeline._augment_answer_completeness(
        "كيف أمر الله بالإحسان إلى الوالدين؟", partial, chunks,
    )
    assert "فلا تقل لهما أف" in out
    assert "وقل لهما قولا كريما" in out


def test_quran_anchor_completeness_skips_tafsir_intent():
    chunks = _quran_anchor_chunks()
    assert pipeline._quran_topic_anchor_verse(
        "ما تفسير آية الوالدين؟", chunks,
    ) is None


def test_quran_wudu_fragment_suppresses_duplicate_full_verse():
    verse = (
        "يا أيها الذين آمنوا إذا قمتم إلى الصلاة فاغسلوا وجوهكم "
        "وأيديكم إلى المرافق وامسحوا برؤوسكم وأرجلكم إلى الكعبين "
        "وإن كنتم جنبا فاطهروا"
    )
    chunk = {
        "id": "quran_5_6", "source": "quran", "reference": "المائدة:6",
        "original_text": verse,
        "metadata": {"surah_name": "المائدة", "ayah_number": 6},
        "_injected": True,
    }
    answer = (
        'قال الله تعالى: "يا أيها الذين آمنوا إذا قمتم إلى الصلاة '
        "فاغسلوا وجوهكم وأيديكم إلى المرافق وامسحوا برؤوسكم وأرجلكم "
        'إلى الكعبين" (المائدة:6) (مصدر 1).'
    )
    out = pipeline._augment_answer_completeness(
        "كيف يتوضأ المسلم قبل الصلاة؟", answer, [chunk],
    )
    assert "جنبا" not in out


def test_ayat_al_kursi_label_comes_from_tafsir_evidence():
    quran = {
        "id": "quran_2_255", "source": "quran", "reference": "البقرة:255",
        "original_text": "الله لا إله إلا هو الحي القيوم",
        "metadata": {"surah_number": 2, "ayah_number": 255},
        "_injected": True,
    }
    tafsir = {
        "id": "tafsir_ibnkathir_2_255", "source": "tafsir",
        "reference": "تفسير ابن كثير 2:255",
        "original_text": (
            "هذه آية الكرسي ولها شأن عظيم، قد صح الحديث عن رسول الله "
            "صلى الله عليه وسلم بأنها أفضل آية في كتاب الله."
        ),
    }
    assert pipeline._quran_topic_companions(
        quran, [quran, tafsir],
    )[0]["id"] == "tafsir_ibnkathir_2_255"
    answer = 'قال الله تعالى: "الله لا إله إلا هو الحي القيوم" (مصدر 1).'
    out = pipeline._augment_answer_completeness(
        "ما أعظم آية في القرآن الكريم؟", answer, [quran, tafsir],
    )
    assert "آية الكرسي" in out


def test_zakat_fitr_completeness_adds_measure_and_deadline():
    source = {
        "id": "fiqh-zakat", "source": "fiqh", "reference": "زكاة الفطر",
        "original_text": (
            "زكاة الفطر واجبة على كل مسلم. "
            "والواجب فيها صاع من طعام. "
            "أمرنا رسول الله صلى الله عليه وسلم بزكاة الفطر أن تؤدى "
            "قبل خروج الناس إلى الصلاة."
        ),
    }
    answer = "زكاة الفطر واجبة على كل مسلم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم زكاة الفطر ومتى تجب؟", answer, [source],
    )
    assert "صاع" in out
    assert "قبل خروج الناس" in out


def test_zakat_timing_snippet_does_not_begin_mid_quote():
    source = {
        "id": "fiqh-zakat-long", "source": "fiqh", "reference": "زكاة الفطر",
        "original_text": (
            'روى ابن عباس قال: "فرض رسول الله صلى الله عليه وسلم زكاة '
            "الفطر طهرة للصائم، من اللغو والرفث وطعمة للمساكين، من أداها "
            'قبل الصلاة، فهي زكاة مقبولة، ومن أداها بعد الصلاة، فهي صدقة '
            'من الصدقات".'
        ),
    }
    answer = "زكاة الفطر واجبة على كل مسلم (مصدر 1)."
    out = pipeline._augment_answer_completeness(
        "ما حكم زكاة الفطر ومتى تجب؟", answer, [source],
    )
    assert "من أداها قبل الصلاة" in out
    assert "من اللغو والرفث" not in out
    assert "(مصدر 1)" in out
