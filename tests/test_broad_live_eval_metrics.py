"""Regression tests for broad live-eval metric normalization."""
from __future__ import annotations

from src.rag import run_broad_live_eval as E


def test_nz_joins_uthmani_tanween_alef():
    # Uthmani writes the tanween alef as a separate token; the rubric is
    # written in modern Arabic and must still match.
    text = "وَقُل لَّهُمَا قَوۡلࣰ ا كَرِيمࣰ ا"
    normalized = E.nz(text)
    assert "قل لهما قولا كريما" in normalized


def test_nz_folds_uthmani_waw_alef_for_rubric_matching():
    normalized = E.nz("وَحَرَّمَ ٱلرِّبَوٰاْ")
    assert "حرم الربا" in normalized
    assert E.nz("حرم الربا") in normalized


def test_sentence_support_credits_exact_quote():
    chunk = {
        "id": "quran_5_6",
        "source": "quran",
        "original_text": (
            "فاغسلوا وجوهكم وأيديكم إلى المرافق وامسحوا برؤوسكم "
            "وأرجلكم إلى الكعبين."
        ),
    }
    sentence = (
        '**غسل الوجه**: لقوله تعالى: "فاغسلوا وجوهكم" '
        "(المائدة: 6) (مصدر 1)."
    )
    assert E._sentence_support(sentence, [chunk], [0]) >= 0.5


def test_sentence_support_still_rejects_unsupported_body():
    chunk = {
        "id": "quran_5_6",
        "source": "quran",
        "original_text": "فاغسلوا وجوهكم وأيديكم إلى المرافق.",
    }
    sentence = (
        'غسل الوجه واجب، وغسل القدمين سبع مرات، '
        'لقوله تعالى: "فاغسلوا وجوهكم" (مصدر 1).'
    )
    assert E._sentence_support(sentence, [chunk], [0]) < 0.6
