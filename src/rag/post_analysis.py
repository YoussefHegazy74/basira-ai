"""
Basira — Evidence-Based Post Analysis
=====================================
Extends the independent verification service from a single quotation to a
complete post: quotations, attributions, explanations, and conclusions.

The module keeps three kinds of work separate:

1. Deterministic source identity: lexical Quran matching, lexical Hadith
   matn matching, and lexical Seerah/historical passage matching.
2. Recorded authenticity/reliability evidence: direct grading statements in
   the Hadith record, or a source-backed grading/assessment found in the
   Tafsir/Fiqh/Sira passages that quote the same narration.
3. Interpretive assessment: whether the author's explanation and conclusion
   are supported by the retrieved evidence.  This is the only part delegated
   to the model, and it is explicitly prevented from changing the
   deterministic source facts.

No semantic similarity is treated as proof that a quotation exists.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import OrderedDict

from src.nlp.preprocessor import normalize_verse_text
from src.rag.pipeline import _bm25_search
from src.rag.tafsir import match_quran_quotation

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "quran": "القرآن",
    "hadith": "حديث",
    "tafsir": "تفسير",
    "fiqh": "فقه",
    "sira": "سيرة",
}

TYPE_LABELS = {
    "quran_quote": "اقتباس قرآني",
    "hadith_quote": "اقتباس حديثي",
    "seerah_claim": "خبر تاريخي / سيرة",
    "quote": "اقتباس",
    "claim": "دعوى",
    "conclusion": "استنتاج",
    "unidentified_hadith_claim": "حكم على حديث غير محدد",
}

GRADING_TERMS = (
    "موضوع", "مختلق", "ضعيف", "منكر", "لا يصح", "لم يثبت", "باطل",
    "حسن صحيح", "صحيح", "حسن", "ثابت", "متفق عليه",
)

_SEERAH_MARKERS = (
    "غزوة", "غزوه", "سرية", "سريه", "صلح الحديبية", "صلح الحديبيه",
    "فتح مكة", "هجرة النبي", "الهجرة إلى المدينة", "الهجره الي المدينه",
    "الاسراء", "الإسراء", "المعراج", "غزوة بدر", "غزوة أحد",
    "غزوة الخندق", "خيبر", "تبوك", "مؤتة", "مؤته", "حنين",
    "عام الفيل", "دار الارقم", "دار الأرقم", "وفاة النبي",
)

_ARGUMENT_MARKERS = (
    "اذن", "إذن", "لذلك", "فان", "فإن", "وهذا يدل", "يدل على",
    "معني", "معنى", "يعني", "اي ان", "أي أن", "الخلاصه", "الخلاصة",
    "الدرس", "يستفاد", "النتيجه", "النتيجة", "لان", "لأن",
    "يجب", "لا يجوز", "الواجب", "الحرام", "الحلال",
)

_ATTRIBUTION_HADITH = (
    "رسول الله", "النبي", "صلى الله عليه وسلم", "ﷺ", "روى", "أخرج",
    "الحديث", "حديث", "عن النبي", "عن رسول",
)
_ATTRIBUTION_QURAN = (
    "قال الله", "قال تعالى", "الآية", "الايه", "القرآن", "القران",
    "قوله تعالى",
)
_SEERAH_RE = re.compile("|".join(re.escape(marker) for marker in _SEERAH_MARKERS))

_QUOTE_PATTERNS = (
    re.compile(r"﴿([^﴾]{3,})﴾"),
    re.compile(r"«([^»]{3,})»"),
    re.compile(r"“([^”]{3,})”"),
    re.compile(r'"([^"\n]{3,})"'),
)

# Leading narration words that are not part of the matn.
_FRAMING_LEAD = (
    "حدثنا", "حدثني", "أخبرنا", "اخبرنا", "أنبأنا", "انبانا",
    "قال رسول الله", "قال النبي", "قال", "يقول", "سمعت", "عن",
    "أن", "ان", "رضي الله عنه", "رضي الله عنها", "صلى الله عليه وسلم",
    "ﷺ",
)

_ASSESSOR_NAMES = (
    "أبو عيسى", "ابو عيسى", "الترمذي", "الترمذى", "أبو داود", "ابو داود",
    "النسائي", "ابن ماجه", "أحمد", "احمد", "البخاري", "البخارى", "مسلم",
    "ابن حجر", "الألباني", "الالباني", "ابن الجوزي", "ابن تيمية",
    "ابن تيميه", "الذهبي", "النووي", "الزيلعي", "الهيثمي", "البيهقي",
    "الدارقطني", "ابن كثير", "الشافعي", "مالك",
)

_GRADING_PATTERNS = (
    ("hasan_sahih", re.compile(r"حسن\s+صحيح|حسنٌ\s+صحيح")),
    ("sahih", re.compile(r"(?:هذا\s+)?حديث\s+صحيح|صحيح\s+الإسناد|إسناده\s+صحيح")),
    ("hasan", re.compile(r"(?:هذا\s+)?حديث\s+حسن|حسن\s+الإسناد|إسناده\s+حسن")),
    ("mawdu", re.compile(r"حديث\s+موضوع|موضوع\s+مختلق|مختلق|وضعه|الموضوعات")),
    ("daif", re.compile(r"حديث\s+ضعيف|ضعيف\s+جدا|ضعفه\s+جدا|ضعفه|إسناده\s+ضعيف|اسناده\s+ضعيف|في\s+إسناده\s+ضعف|في\s+اسناده\s+ضعف")),
    ("munkar", re.compile(r"حديث\s+منكر|منكر\s+الحديث|خبر\s+منكر")),
    ("not_authentic", re.compile(r"لا\s+يصح|لم\s+يثبت|غير\s+ثابت|لا\s+أصل\s+له|لا\s+اصل\s+له")),
    ("batil", re.compile(r"حديث\s+باطل|باطل\s+لا\s+أصل|باطل\s+لا\s+اصل")),
)

_STATUS_LABELS = {
    "sahih_collection": "أخرجه المصدر الصحيح",
    "hasan_sahih": "حسن صحيح وفق المصدر المذكور",
    "sahih": "صحيح وفق المصدر المذكور",
    "hasan": "حسن وفق المصدر المذكور",
    "daif": "الرواية ضعيفة وفق المصدر المذكور",
    "mawdu": "الرواية موضوعة وفق المصدر المذكور",
    "munkar": "الرواية منكرة وفق المصدر المذكور",
    "not_authentic": "لا يصح وفق المصدر المذكور",
    "batil": "باطل وفق المصدر المذكور",
    "disputed": "توجد خلافات معتبرة في التقييم",
    "not_recorded": "لم تُسجَّل درجة حديثية مستقلة في قاعدة البيانات المتاحة",
    "not_applicable": "",
}

_COMPONENT_STATUS_LABELS = {
    "complete": "الاقتباس كامل",
    "partial_not_material": "اقتباس ناقص غير مؤثر",
    "partial_material": "اقتباس ناقص يحذف سياقاً مؤثراً",
    "altered": "اقتباس محرّف",
    "unknown": "تعذّر تقييم الاكتمال",
}

_INTERPRETATION_LABELS = {
    "supported": "الشرح مدعوم بالمصدر",
    "contradicted": "الشرح غير مدعوم بالمصدر",
    "partially_supported": "الشرح مدعوم جزئياً",
    "disputed": "توجد خلافات معتبرة في التفسير",
    "inconclusive": "غير محسوم — الأدلة غير كافية",
    "not_provided": "لم يتم تقديم تفسير لتقييمه",
}

_ARGUMENT_LABELS = {
    "supported": "الاستدلال مدعوم",
    "unsupported": "الاستدلال غير مدعوم بالمصدر",
    "overstated": "الاستدلال أقوى مما يثبته المصدر",
    "misleading": "استدلال مضلّل",
    "disputed": "استدلال محل خلاف",
    "inconclusive": "لم يتم حسم الاستدلال",
    "not_assessed": "لم يتم تقديم استدلال صريح لتقييمه",
}

_SOURCE_STATUS_LABELS = {
    "verified": "المصدر مطابق",
    "partial": "المصدر مطابق جزئياً",
    "approximate": "مطابقة تقريبية",
    "altered": "الاقتباس محرّف",
    "ambiguous": "تطابق غير حاسم",
    "not_found": "لم يُعثر عليه في قاعدة البيانات المتاحة",
    "inconclusive": "تعذّر حسم المصدر",
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return normalize_verse_text(text or "")


_FAST_MARKS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u200B-\u200F"
    r"\u202A-\u202E\u2066-\u2069\uFEFF\u0640]"
)
_FAST_TRANS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    "ى": "ي", "ی": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه", "ہ": "ه",
    "ک": "ك",
})


def _fast_normalize(text: str) -> str:
    """Fast normalisation for long-cache substring matching only."""
    value = (text or "").translate(_FAST_TRANS)
    value = _FAST_MARKS_RE.sub("", value)
    value = "".join(
        char if (char.isalnum() or char.isspace()) else " "
        for char in value
    )
    return " ".join(value.split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_normalize(text).split())


def _contains(text: str, markers) -> bool:
    normalized = _normalize(text)
    return any(_normalize(marker) in normalized for marker in markers)


def _strip_hadith_framing(text: str) -> str:
    """Remove isnad/framing around a hadith matn for matching only."""
    raw = (text or "").strip()
    if not raw:
        return ""
    # Keep the part after the last explicit prophetic attribution when the
    # remaining text is substantive.
    for marker in ("قال رسول الله", "قال النبي", "صلى الله عليه وسلم", "ﷺ"):
        index = raw.rfind(marker)
        if index >= 0:
            tail = raw[index + len(marker):]
            if len(_tokens(tail)) >= 3:
                raw = tail
                break
    raw = raw.strip(" \n\t:،.؛\"'«»“”")
    changed = True
    while changed:
        changed = False
        normalized = _normalize(raw)
        words = normalized.split()
        for lead in sorted(_FRAMING_LEAD, key=len, reverse=True):
            lead_words = _normalize(lead).split()
            if lead_words and words[:len(lead_words)] == lead_words:
                # Remove by raw word count; normalized and raw word counts
                # usually match for these short framing phrases.
                raw_words = raw.split()
                raw = " ".join(raw_words[len(lead_words):]).strip()
                changed = True
                break
    return raw.strip(" \n\t:،.؛\"'«»“”")


def _query_variants(text: str) -> set[tuple[str, ...]]:
    base = _tokens(_strip_hadith_framing(text))
    variants = {base}
    if len(base) > 1 and base[0] in {"و", "ف"}:
        variants.add(base[1:])
    if len(base) > 1 and base[0] in {"قال", "يقول", "عن", "ان", "أن"}:
        variants.add(base[1:])
    # A leading conjunction attached to the first word (ولا / فلم).
    if base and len(base[0]) > 2 and base[0][0] in {"و", "ف"}:
        variants.add((base[0][1:],) + base[1:])
    # Conjunctions can also be attached to later words ("تبا للذهب وتبا
    # للفضة" vs "تبا للذهب تبا للفضة").  This variant is only used for the
    # contiguous lexical comparison, never to rewrite the submitted text.
    stripped = tuple(
        token[1:]
        if len(token) > 2 and token[0] in {"و", "ف"}
        else token
        for token in base
    )
    if stripped != base:
        variants.add(stripped)
    return {variant for variant in variants if variant}


def _best_run(
    query: tuple[str, ...],
    source: tuple[str, ...],
) -> tuple[int, int, int]:
    """Return (matched_length, query_start, source_start)."""
    best = (0, 0, 0)
    if not query or not source:
        return best
    for q_start in range(len(query)):
        for s_start in range(len(source)):
            length = 0
            while (
                q_start + length < len(query)
                and s_start + length < len(source)
                and query[q_start + length] == source[s_start + length]
            ):
                length += 1
            if length > best[0]:
                best = (length, q_start, s_start)
    return best


def _best_match(
    query: set[tuple[str, ...]],
    source: tuple[str, ...],
) -> tuple[int, int, int, tuple[str, ...]]:
    best = (0, 0, 0, ())
    for variant in query:
        matched, q_start, s_start = _best_run(variant, source)
        if matched > best[0]:
            best = (matched, q_start, s_start, variant)
    return best


def _chunk_map(chunks: list[dict]) -> dict[str, dict]:
    return {
        chunk.get("id"): chunk
        for chunk in chunks
        if chunk.get("id")
    }


def _bm25_chunks(
    query_text: str,
    chunks: list[dict],
    *,
    sources: set[str],
    top_k: int = 80,
    extra_terms: tuple[str, ...] = (),
) -> list[dict]:
    query_tokens = list(_tokens(query_text)) + list(extra_terms)
    if not query_tokens:
        return []
    try:
        raw = _bm25_search(query_tokens, chunks, top_k=top_k)
    except Exception as error:
        raise RetrievalFailure("lexical retrieval failed") from error
    by_id = _chunk_map(chunks)
    results: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        chunk = by_id.get(item.get("id"))
        if chunk is None:
            continue
        if chunk.get("source") not in sources:
            continue
        marker = chunk.get("id")
        if marker in seen:
            continue
        seen.add(marker)
        enriched = dict(chunk)
        enriched["_bm25_score"] = item.get("score", 0.0)
        results.append(enriched)
    return results


class RetrievalFailure(RuntimeError):
    """Raised when the lexical retrieval subsystem itself fails."""


_TEXT_CACHE: dict[tuple, list[tuple[dict, str]]] = {}


def _cached_texts(
    chunks: list[dict],
    sources: set[str],
    require_terms: tuple[str, ...] = (),
) -> list[tuple[dict, str]]:
    """Return normalized texts for a source subset, built once per corpus.

    BM25 caps long Tafsir chunks at their first tokens, so a grading or a
    historical assessment near the end of a long passage would otherwise be
    invisible.  This cache is deliberately limited to the sources and terms
    needed by the caller.
    """
    key = (
        id(chunks),
        len(chunks),
        tuple(sorted(sources)),
        tuple(require_terms),
    )
    cached = _TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    entries: list[tuple[dict, str]] = []
    normalized_terms = tuple(_fast_normalize(term) for term in require_terms)
    for chunk in chunks:
        if chunk.get("source") not in sources:
            continue
        text = _fast_normalize(chunk.get("original_text", ""))
        if normalized_terms and not any(term in text for term in normalized_terms):
            continue
        entries.append((chunk, text))
    _TEXT_CACHE[key] = entries
    return entries


def _fast_query_variants(text: str) -> set[tuple[str, ...]]:
    base = tuple(_fast_normalize(text).split())
    variants = {base}
    if len(base) > 1 and base[0] in {"و", "ف"}:
        variants.add(base[1:])
    if base and len(base[0]) > 2 and base[0][0] in {"و", "ف"}:
        variants.add((base[0][1:],) + base[1:])
    stripped = tuple(
        token[1:]
        if len(token) > 2 and token[0] in {"و", "ف"}
        else token
        for token in base
    )
    if stripped != base:
        variants.add(stripped)
    return {variant for variant in variants if variant}


def _lexical_substring_candidates(
    text: str,
    cached: list[tuple[dict, str]],
    *,
    max_windows: int = 18,
    limit: int = 80,
) -> list[dict]:
    if not cached:
        return []
    variants = _fast_query_variants(text)
    windows: list[str] = []
    for variant in variants:
        if len(variant) < 4:
            continue
        for size in (7, 6, 5, 4):
            if len(variant) < size:
                continue
            for start in range(min(len(variant) - size + 1, 10)):
                windows.append(" ".join(variant[start:start + size]))
    windows = list(dict.fromkeys(windows))[:max_windows]
    if not windows:
        return []
    found: list[dict] = []
    for chunk, normalized in cached:
        if any(window in normalized for window in windows):
            found.append(chunk)
            if len(found) >= limit:
                break
    return found


def _excerpt_around_tokens(
    text: str,
    match_start: int,
    match_length: int,
    *,
    before: int = 45,
    after: int = 90,
) -> str:
    tokens = _normalize(text).split()
    start = max(0, match_start - before)
    end = min(len(tokens), match_start + match_length + after)
    return " ".join(tokens[start:end])


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the sentence boundaries containing text[start:end]."""
    separators = ".!؟\n؛"
    left = start
    while left > 0 and text[left - 1] not in separators:
        left -= 1
    right = end
    while right < len(text) and text[right] not in separators:
        right += 1
    if right < len(text):
        right += 1
    return left, right


def _sentence_text(text: str, start: int, end: int) -> str:
    return " ".join(text[start:end].split()).strip()


_RAW_WORD_RE = re.compile(r"\S+")


def _raw_omitted_text(
    canonical: str,
    source: tuple[str, ...],
    start: int,
    matched: int,
) -> str:
    """Return omitted text from the original wording, not normalized tokens.

    Normalization strips hamza and diacritics and can split Uthmani words,
    which made the displayed omission look malformed.  Map each normalized
    token back to its original word and slice the original string instead.
    """
    spans: list[tuple[int, int]] = []
    for match in _RAW_WORD_RE.finditer(canonical or ""):
        parts = _normalize(match.group()).split()
        if len(parts) != 1:
            return ""
        spans.append((match.start(), match.end()))
    if len(spans) != len(source):
        return ""

    def join(items: list[tuple[int, int]]) -> str:
        return " ".join(canonical[start:end] for start, end in items).strip()

    before = join(spans[:start])
    after = join(spans[start + matched:])
    parts = [part for part in (before, after) if part]
    if len(parts) == 2:
        return f"{parts[0]} … {parts[1]}"
    return parts[0] if parts else ""


def _omitted_text(canonical: str, excerpt: str) -> str:
    """Return the canonical text outside the quoted contiguous segment."""
    source = _tokens(canonical)
    matched, _q_start, source_start, _variant = _best_match(
        _query_variants(excerpt), source,
    )
    if matched <= 0:
        return " ".join(source)
    raw = _raw_omitted_text(canonical, source, source_start, matched)
    if raw:
        return raw
    omitted = source[:source_start] + source[source_start + matched:]
    return " ".join(omitted).strip()


# ---------------------------------------------------------------------------
# Hadith matching
# ---------------------------------------------------------------------------

def _hadith_matn(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    matn = metadata.get("matn") or chunk.get("original_text") or ""
    return _strip_hadith_framing(matn)


def match_hadith_quote(
    excerpt: str,
    chunks: list[dict],
    *,
    top_k: int = 100,
) -> dict:
    """Find a hadith matn by lexical exact/partial/fuzzy matching.

    BM25 is used only to shortlist candidates from the existing corpus.  The
    identity decision is made by a contiguous token comparison against each
    candidate's recorded matn.  A semantic score never decides existence.
    """
    empty = {
        "found": False,
        "ambiguous": False,
        "match_type": "none",
        "is_complete": False,
        "is_partial": False,
        "coverage": 0.0,
        "matched_tokens": 0,
        "query_tokens": 0,
        "canonical_text": "",
        "matn": "",
        "reference": "",
        "source": "hadith",
        "book": "",
        "omitted_text": "",
        "chunk": None,
        "candidates": [],
    }
    query = _query_variants(excerpt)
    if not query:
        return empty
    query_len = max(len(variant) for variant in query)

    candidates = _bm25_chunks(
        excerpt,
        chunks,
        sources={"hadith"},
        top_k=top_k,
        extra_terms=("حديث",),
    )
    # Small corpora (tests and offline evaluation) can be scanned fully.
    hadith_chunks = [c for c in chunks if c.get("source") == "hadith"]
    if len(hadith_chunks) <= 5000:
        seen = {chunk.get("id") for chunk in candidates}
        for chunk in hadith_chunks:
            if chunk.get("id") not in seen:
                candidates.append(chunk)
                seen.add(chunk.get("id"))

    scored: list[tuple[float, int, dict, tuple[str, ...], int]] = []
    for chunk in candidates:
        matn_raw = _hadith_matn(chunk)
        source = _tokens(matn_raw)
        if not source:
            continue
        matched, _q_start, s_start, variant = _best_match(query, source)
        coverage = matched / max(1, len(variant))
        score = coverage + min(matched, 20) / 1000.0
        if matched:
            scored.append(
                (score, matched, chunk, variant, len(source), s_start)
            )

    if not scored:
        # Bounded fuzzy fallback.  It is labelled approximate, never exact.
        fuzzy: list[tuple[float, dict]] = []
        for chunk in candidates:
            source = _tokens(_hadith_matn(chunk))
            for variant in query:
                if len(variant) < 6 or len(source) < 6:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, variant, source, autojunk=False,
                ).ratio()
                if ratio >= 0.90:
                    fuzzy.append((ratio, chunk))
        fuzzy.sort(key=lambda item: item[0], reverse=True)
        if fuzzy:
            ratio, chunk = fuzzy[0]
            matn = _hadith_matn(chunk)
            metadata = chunk.get("metadata") or {}
            return {
                **empty,
                "found": True,
                "match_type": "approximate",
                "is_partial": False,
                "coverage": round(ratio, 4),
                "matched_tokens": query_len,
                "query_tokens": query_len,
                "canonical_text": matn,
                "matn": matn,
                "reference": chunk.get("reference", ""),
                "book": metadata.get("book_name_ar", ""),
                "chunk": chunk,
            }
        return empty

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_matched, best_chunk, best_variant, source_len, _ = scored[0]
    best_coverage = best_matched / max(1, len(best_variant))
    tied = [
        item for item in scored
        if abs(item[0] - best_score) < 0.005
    ]
    prefix_match = all(item[5] <= 2 for item in tied)
    if len(tied) > 1 and (best_matched < 3 or not prefix_match):
        return {
            **empty,
            "ambiguous": True,
            "match_type": "ambiguous",
            "query_tokens": query_len,
            "candidates": [
                {
                    "reference": item[2].get("reference", ""),
                    "text": _hadith_matn(item[2]),
                }
                for item in tied[:8]
            ],
        }
    if len(tied) > 1:
        # The same short matn is transmitted in several collections.  This
        # is a multi-reference match, not an ambiguous one.  Prefer a
        # collection whose inclusion itself carries authenticity weight.
        def _preference(item):
            book = (item[2].get("metadata") or {}).get("book_name_ar", "")
            return (
                0 if book in {"صحيح البخاري", "صحيح مسلم"} else 1,
                item[2].get("reference", ""),
            )

        tied.sort(key=_preference)
        best_score, best_matched, best_chunk, best_variant, source_len, _ = tied[0]
        best_coverage = best_matched / max(1, len(best_variant))
        metadata = best_chunk.get("metadata") or {}
        matn = _hadith_matn(best_chunk)
        complete = (
            best_coverage >= 0.999
            and best_matched >= source_len
            and len(best_variant) <= source_len + 2
        )
        references = list(dict.fromkeys(
            item[2].get("reference", "") for item in tied
            if item[2].get("reference")
        ))
        unmatched = max(0, len(best_variant) - best_matched)
        return {
            **empty,
            "found": True,
            "match_type": (
                "exact" if complete
                else "altered" if unmatched > 0
                else "partial"
            ),
            "is_complete": complete,
            "is_partial": not complete and unmatched == 0,
            "unmatched_tokens": unmatched,
            "coverage": round(best_coverage, 4),
            "matched_tokens": best_matched,
            "query_tokens": query_len,
            "canonical_text": matn,
            "matn": matn,
            "reference": references[0] if references else "",
            "references": references,
            "book": metadata.get("book_name_ar", ""),
            "omitted_text": "" if complete else matn,
            "chunk": best_chunk,
            "candidates": [
                {
                    "reference": item[2].get("reference", ""),
                    "text": _hadith_matn(item[2]),
                }
                for item in tied[:8]
            ],
        }

    metadata = best_chunk.get("metadata") or {}
    matn = _hadith_matn(best_chunk)
    complete = (
        best_coverage >= 0.999
        and best_matched >= source_len
        and len(best_variant) <= source_len + 2
    )
    is_partial = not complete
    found = (
        (best_coverage >= 0.80 and best_matched >= min(3, len(best_variant)))
        or (best_coverage >= 0.999 and best_matched >= 2)
    )
    if not found:
        return empty

    unmatched = max(0, len(best_variant) - best_matched)
    return {
        **empty,
        "found": True,
        "match_type": (
            "exact" if complete
            else "altered" if unmatched > 0
            else "partial"
        ),
        "is_complete": complete,
        "is_partial": is_partial and unmatched == 0,
        "unmatched_tokens": unmatched,
        "coverage": round(best_coverage, 4),
        "matched_tokens": best_matched,
        "query_tokens": query_len,
        "canonical_text": matn,
        "matn": matn,
        "reference": best_chunk.get("reference", ""),
        "book": metadata.get("book_name_ar", ""),
        "omitted_text": "" if complete else matn,
        "chunk": best_chunk,
    }


# ---------------------------------------------------------------------------
# Grading / reliability evidence
# ---------------------------------------------------------------------------

_NEW_NARRATION_MARKERS = (
    "لحديث", "حدثنا", "أخبرنا", "اخبرنا", "وعن ", "وروى ",
)

_GRADING_CONTEXT_MARKERS = (
    "رواه", "رواة", "قال الترمذي", "قال أبو عيسى", "قال ابو عيسى",
    "قال أبو داود", "قال ابو داود", "قال النسائي", "قال ابن ماجه",
    "قال الحافظ", "قال ابن", "قال الألباني", "قال الالباني",
    "الحديث", "إسناد", "اسناد", "ضعيف", "موضوع", "مختلق",
    "منكر", "لا يصح", "لم يثبت", "حسن", "صحيح", "باطل",
)


def _norm_contains(text: str, marker: str) -> bool:
    return _normalize(marker) in _normalize(text)


def _grading_window_for_match(
    raw: str,
    windows: list[str],
) -> str:
    """Return only the source sentence(s) attached to the matched narration.

    A long Tafsir chunk can quote several narrations and then grade one of
    them.  Searching a fixed +/-900-character window would let a grading for
    narration B leak onto narration A.  This helper locates the raw sentence
    containing the matched phrase and, at most, one immediately following
    grading/reference sentence that does not start a new narration.
    """
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!؟؛])\s+|\n+", raw)
        if sentence.strip()
    ]
    selected_index = None
    for index, sentence in enumerate(sentences):
        fast = _fast_normalize(sentence)
        if any(window in fast for window in windows):
            selected_index = index
            break
    if selected_index is None:
        # No single sentence contains the full lexical window (punctuation
        # may split it).  Choose the sentence with the strongest token overlap
        # with the matched matn; if even that is weak, return no evidence
        # rather than falling back to an unrestricted nearby window.
        query_tokens = set(" ".join(windows).split())
        best_overlap = 0
        best_index = None
        for index, sentence in enumerate(sentences):
            overlap = len(
                query_tokens & set(_fast_normalize(sentence).split())
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is None or best_overlap < 4:
            return ""
        selected_index = best_index

    parts = [sentences[selected_index]]
    # Follow the same narration through its continuation and any immediately
    # following reference/grading sentences.  Stop at an explicit new
    # narration marker or after a bounded number of sentences.
    for next_sentence in sentences[selected_index + 1:selected_index + 13]:
        if any(_norm_contains(next_sentence, marker)
               for marker in _NEW_NARRATION_MARKERS):
            break
        parts.append(next_sentence)
        if len(" ".join(parts)) > 3500:
            break
    return " ".join(parts)


def _assessor_near(text: str, position: int) -> str:
    window = text[max(0, position - 260):position]
    found: list[tuple[int, str]] = []
    for name in _ASSESSOR_NAMES:
        index = window.rfind(name)
        if index >= 0:
            found.append((index, name))
    if not found:
        return ""
    return max(found, key=lambda item: item[0])[1]


def _gradings_from_text(text: str) -> list[tuple[str, str, str]]:
    """Return all distinct grading assessments found in a text window."""
    raw = text or ""
    results: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for status, pattern in _GRADING_PATTERNS:
        for match in pattern.finditer(raw):
            window = raw[max(0, match.start() - 100):match.end() + 120]
            # A bare "صحيح" is usually a book title; require a hadith word
            # nearby unless the phrase itself is a grading phrase.
            if status in {"sahih", "hasan"} and not re.search(
                r"حديث|خبر|رواية|إسناد|اسناد", window,
            ):
                continue
            left, right = _sentence_around(raw, match.start(), match.end())
            sentence = _sentence_text(raw, left, right)[:700]
            assessor = _assessor_near(raw, match.start())
            key = (status, sentence[:100])
            if key in seen:
                continue
            seen.add(key)
            results.append((status, sentence, assessor))
    # "حسن صحيح" already carries the combined ruling; do not also report a
    # separate generic "صحيح" from the same phrase.
    if any(status == "hasan_sahih" for status, _s, _a in results):
        results = [
            item for item in results
            if item[0] not in {"sahih", "hasan"}
        ]
    return results


def _grading_from_text(text: str) -> tuple[str, str, str] | None:
    """Return the first (status, assessment_sentence, assessor) found."""
    results = _gradings_from_text(text)
    return results[0] if results else None


def extract_direct_grading(chunk: dict, matn: str) -> dict | None:
    metadata = chunk.get("metadata") or {}
    text = " ".join(
        part for part in (matn, chunk.get("original_text", "")) if part
    )
    found = _grading_from_text(text)
    if found is None:
        return None
    status, sentence, assessor = found
    if status == "mawdu":
        label = "الرواية موضوعة وفق المصدر المذكور"
    elif status == "daif":
        label = "الرواية ضعيفة وفق المصدر المذكور"
    elif status == "hasan_sahih":
        label = "حسن صحيح وفق المصدر المذكور"
    else:
        label = _STATUS_LABELS.get(status, "تقييم مذكور في المصدر")
    return {
        "status": status,
        "label": label,
        "source": chunk.get("reference", ""),
        "book": metadata.get("book_name_ar", ""),
        "assessment": sentence,
        "assessor": assessor,
        "evidence_type": "hadith_text",
    }


def find_external_grading(
    matn: str,
    chunks: list[dict],
    *,
    top_k: int = 50,
) -> list[dict]:
    """Find source-backed grading statements in Tafsir/Fiqh/Sira passages."""
    if not matn:
        return []
    query_groups = (
        ("موضوع", "مختلق"),
        ("ضعيف", "منكر", "لا", "يصح"),
        ("حسن", "صحيح"),
    )
    candidates: list[dict] = []
    seen: set[str] = set()
    for terms in query_groups:
        for chunk in _bm25_chunks(
            matn,
            chunks,
            sources={"tafsir", "fiqh", "sira"},
            top_k=top_k,
            extra_terms=terms,
        ):
            if chunk.get("id") not in seen:
                seen.add(chunk.get("id"))
                candidates.append(chunk)

    # BM25 caps long Tafsir chunks; scan the cached grading-bearing passages
    # directly so an assessment near the end of a long passage is not missed.
    cached = _cached_texts(
        chunks, {"tafsir", "fiqh", "sira"}, GRADING_TERMS,
    )
    for chunk in _lexical_substring_candidates(matn, cached):
        if chunk.get("id") not in seen:
            seen.add(chunk.get("id"))
            candidates.append(chunk)

    matn_tokens = _tokens(matn)
    if not matn_tokens:
        return []
    # Distinctive windows from the matn, used to prove that the passage is
    # actually quoting this narration and not merely discussing a theme.
    windows: list[str] = []
    for variant in _query_variants(matn):
        if len(variant) < 4:
            continue
        for size in (7, 5, 4):
            if len(variant) < size:
                continue
            for start in range(0, min(len(variant) - size + 1, 12)):
                windows.append(" ".join(variant[start:start + size]))
    windows = list(dict.fromkeys(windows))[:40]
    if not windows:
        windows = [" ".join(matn_tokens[:6])]

    # Require the passage to contain the opening of the same narration, not
    # only a generic tail such as "في ليلة الجمعة غفر له".  This prevents a
    # different Surah/narration from inheriting a nearby grading.
    anchors: list[str] = []
    for variant in _query_variants(matn):
        if len(variant) >= 4:
            anchors.append(" ".join(variant[:3]))
    anchors = list(dict.fromkeys(anchors))
    if not anchors:
        return []

    evidence: list[dict] = []
    seen_refs: set[str] = set()
    for chunk in candidates:
        normalized_text = _normalize(chunk.get("original_text", ""))
        if not any(anchor in normalized_text for anchor in anchors):
            continue
        positions = [
            normalized_text.find(window)
            for window in windows
        ]
        positions = [position for position in positions if position >= 0]
        if not positions:
            continue
        window_text = chunk.get("original_text", "")
        raw_window = _grading_window_for_match(
            window_text, windows,
        )
        found_items = _gradings_from_text(raw_window)
        if not found_items:
            continue
        matched_excerpt = " ".join(raw_window.split())[:1400]
        for status, sentence, assessor in found_items:
            key = sentence[:180]
            if key in seen_refs:
                continue
            seen_refs.add(key)
            evidence.append({
                "status": status,
                "label": _STATUS_LABELS.get(status, "تقييم مذكور في المصدر"),
                "source": chunk.get("reference", ""),
                "book": (chunk.get("metadata") or {}).get("book", ""),
                "assessment": sentence,
                "matched_excerpt": matched_excerpt,
                "assessor": assessor,
                "evidence_type": "external_assessment",
            })
            if len(evidence) >= 6:
                break
        if len(evidence) >= 6:
            break
    return evidence


def _merge_grading(direct: dict | None, external: list[dict]) -> dict:
    if direct is None and not external:
        return {
            "status": "not_recorded",
            "label": _STATUS_LABELS["not_recorded"],
            "source": "",
            "assessment": "",
            "assessor": "",
            "evidence": [],
        }
    items = []
    if direct is not None:
        items.append(direct)
    items.extend(external)
    statuses = {item["status"] for item in items}
    # Conflicting authentic/weak/fabricated assessments are reported as
    # disagreement, not resolved by picking the most convenient one.
    hard = statuses & {"mawdu", "daif", "munkar", "not_authentic", "batil"}
    positive = statuses & {"hasan_sahih", "sahih", "hasan"}
    if hard and positive:
        status = "disputed"
        label = _STATUS_LABELS["disputed"]
    else:
        status = items[0]["status"]
        label = items[0]["label"]
    return {
        "status": status,
        "label": label,
        "source": items[0].get("source", ""),
        "assessment": items[0].get("assessment", ""),
        "assessor": items[0].get("assessor", ""),
        "evidence": items,
    }


def _collection_baseline(chunk: dict) -> dict:
    metadata = chunk.get("metadata") or {}
    book = metadata.get("book_name_ar", "")
    reference = chunk.get("reference", "")
    if book in {"صحيح البخاري", "صحيح مسلم"}:
        return {
            "status": "sahih_collection",
            "label": f"أخرجه {book}",
            "source": reference,
            "assessment": (
                "النص مخرج في أحد الصحيحين؛ لا تتوفر درجة حديثية مستقلة "
                "مسجلة في قاعدة البيانات الحالية."
            ),
            "assessor": "",
            "evidence": [],
        }
    return {
        "status": "not_recorded",
        "label": _STATUS_LABELS["not_recorded"],
        "source": reference,
        "assessment": (
            "قاعدة البيانات الحالية تحتوي متن الحديث ومرجعه، لكنها لا "
            "تحتوي حقل درجة حديثية مستقلاً لهذا الإسناد."
        ),
        "assessor": "",
        "evidence": [],
    }


def assess_hadith(chunk: dict, matn: str, chunks: list[dict]) -> dict:
    direct = extract_direct_grading(chunk, matn)
    if direct is not None:
        return _merge_grading(direct, [])
    book = (chunk.get("metadata") or {}).get("book_name_ar", "")
    if book in {"صحيح البخاري", "صحيح مسلم"}:
        # Inclusion in one of the two Sahih collections is itself the
        # strongest available authenticity signal in this corpus.  External
        # passages are attached as review notes, but a stray assessment in a
        # long Tafsir passage cannot silently overturn that baseline.
        baseline = _collection_baseline(chunk)
        external = find_external_grading(matn, chunks)
        if external:
            baseline["label"] += " — توجد ملاحظات خارجية للمراجعة"
            baseline["external_notes"] = external
            baseline["evidence"] = external[:3]
        return baseline
    external = find_external_grading(matn, chunks)
    if external:
        return _merge_grading(None, external)
    return _collection_baseline(chunk)


# ---------------------------------------------------------------------------
# Seerah / historical matching
# ---------------------------------------------------------------------------

def match_historical_claim(
    claim: str,
    chunks: list[dict],
    *,
    top_k: int = 60,
) -> dict:
    empty = {
        "found": False,
        "ambiguous": False,
        "match_type": "none",
        "coverage": 0.0,
        "matched_tokens": 0,
        "canonical_text": "",
        "reference": "",
        "source": "",
        "chunk": None,
        "candidates": [],
    }
    query = _query_variants(claim)
    if not query:
        return empty
    candidates = _bm25_chunks(
        claim,
        chunks,
        sources={"sira", "tafsir", "fiqh"},
        top_k=top_k,
        extra_terms=("غزوة", "سيرة", "هجرة", "فتح"),
    )
    seen = {chunk.get("id") for chunk in candidates}
    cached_sira = _cached_texts(chunks, {"sira"})
    for chunk in _lexical_substring_candidates(claim, cached_sira):
        if chunk.get("id") not in seen:
            seen.add(chunk.get("id"))
            candidates.append(chunk)
    cached_assessments = _cached_texts(
        chunks, {"tafsir", "fiqh"}, GRADING_TERMS,
    )
    for chunk in _lexical_substring_candidates(claim, cached_assessments):
        if chunk.get("id") not in seen:
            seen.add(chunk.get("id"))
            candidates.append(chunk)

    scored = []
    for chunk in candidates:
        source = _tokens(chunk.get("original_text", ""))
        matched, _q_start, s_start, variant = _best_match(query, source)
        coverage = matched / max(1, len(variant))
        if matched >= 4 or (len(variant) <= 4 and coverage >= 0.999):
            scored.append((coverage + min(matched, 20) / 1000, matched, chunk, s_start, variant, source))
    if not scored:
        return empty
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0]
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.005:
        return {
            **empty,
            "ambiguous": True,
            "match_type": "ambiguous",
            "candidates": [
                {
                    "reference": item[2].get("reference", ""),
                    "text": item[2].get("original_text", "")[:500],
                }
                for item in scored[:5]
            ],
        }
    coverage, matched, chunk, s_start, variant, source = best
    raw = chunk.get("original_text", "")
    raw_position = int(len(raw) * s_start / max(1, len(source)))
    excerpt = raw[max(0, raw_position - 800):raw_position + 1600]
    return {
        "found": True,
        "ambiguous": False,
        "match_type": "partial" if matched < len(source) else "exact",
        "coverage": round(min(1.0, coverage), 4),
        "matched_tokens": matched,
        "canonical_text": excerpt,
        "reference": chunk.get("reference", ""),
        "source": chunk.get("source", ""),
        "chunk": chunk,
        "candidates": [],
    }


def assess_reliability(
    claim: str,
    chunks: list[dict],
    matched: dict,
) -> dict:
    if not matched.get("found"):
        return {
            "status": "not_found",
            "label": "لم يُعثر عليه في قاعدة البيانات المتاحة",
            "source": "",
            "assessment": (
                "عدم وجود الخبر في قاعدة البيانات الحالية لا يعني أنه "
                "مكذوب؛ لا تتوفر أدلة كافية للحكم على ثبوته."
            ),
            "assessor": "",
            "evidence": [],
        }
    chunk = matched.get("chunk") or {}
    text = matched.get("canonical_text") or chunk.get("original_text", "")
    found = _grading_from_text(text)
    if found is not None:
        status, sentence, assessor = found
        return {
            "status": status,
            "label": _STATUS_LABELS.get(status, "توجد ملاحظات على الثبوت"),
            "source": chunk.get("reference", ""),
            "assessment": sentence,
            "assessor": assessor,
            "evidence": [{
                "status": status,
                "label": _STATUS_LABELS.get(status, "ملاحظة"),
                "source": chunk.get("reference", ""),
                "assessment": sentence,
                "assessor": assessor,
            }],
        }
    # Look for an explicit editorial/reliability note near the matching text.
    for term in ("ضعيف", "منكر", "لا يصح", "لم يثبت", "مرسل", "منقطع", "اختلف"):
        if term in _normalize(text):
            left = _normalize(text).find(term)
            raw_position = int(len(text) * left / max(1, len(_normalize(text))))
            window = text[max(0, raw_position - 400):raw_position + 500]
            return {
                "status": "documented_concern",
                "label": "توجد ملاحظات على ثبوت الخبر في المصدر",
                "source": chunk.get("reference", ""),
                "assessment": " ".join(window.split())[:700],
                "assessor": "",
                "evidence": [],
            }
    return {
        "status": "not_assessed",
        "label": "المصدر يسرد الخبر دون تقييم إسنادي صريح في المقطع المتاح",
        "source": chunk.get("reference", ""),
        "assessment": (
            "المصدر التاريخي المتاح يذكر الخبر، لكنه لا يقدم في هذا "
            "المقطع حكماً إسنادياً صريحاً يمكن الاعتماد عليه."
        ),
        "assessor": "",
        "evidence": [],
    }


# ---------------------------------------------------------------------------
# Post component extraction
# ---------------------------------------------------------------------------

def _quote_spans(paragraph: str) -> list[dict]:
    spans: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(paragraph):
            start, end = match.span()
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            quote = match.group(1).strip()
            if len(_tokens(quote)) < 2:
                continue
            occupied.append((start, end))
            spans.append({
                "start": start,
                "end": end,
                "quote": quote,
                "marker": match.group(0)[0],
            })
    spans.sort(key=lambda item: item["start"])
    return spans


def _clean_author_context(context: str) -> str:
    cleaned = context or ""
    cleaned = re.sub(
        r"(?:قال|يقول|روى|أخرج|عن|سمعت)\s+"
        r"(?:رسول\s+الله|النبي|النبي\s+ﷺ|الله\s+تعالى|تعالى|الله)\s*"
        r"(?:صلى\s+الله\s+عليه\s+وسلم)?\s*[:،.؛-]*",
        " ",
        cleaned,
    )
    cleaned = cleaned.replace("﴿", " ").replace("﴾", " ")
    cleaned = cleaned.replace("ﷺ", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :،.؛-")
    cleaned = re.sub(r"^[وف]\s+", "", cleaned)
    return cleaned


_HADITH_AUTHENTICITY_MARKERS = (
    "موضوع", "مكذوب", "مكذوبه", "صحيح", "ضعيف", "حسن", "ثابت",
    "باتفاق العلماء", "متفق عليه", "لا خلاف", "مشهور", "منكر",
)
_HADITH_META_MARKERS = (
    "نص الحديث", "اسم الكتاب", "مصدر", "سند", "اسناد", "رقم الحديث",
    "المرجع", "المصدر",
)


def _has_identifiable_hadith_narration(text: str) -> bool:
    """True only for an actual narration or a usable narration reference."""
    normalized = _normalize(text)
    for marker in (
        "قال رسول الله", "قال النبي", "عن النبي", "عن رسول الله",
        "صلى الله عليه وسلم", "ﷺ",
    ):
        needle = _normalize(marker)
        index = normalized.find(needle)
        if index < 0:
            continue
        tail = normalized[index + len(needle):]
        tail_tokens = [
            token for token in tail.split()
            if token not in {
                "صلى", "الله", "عليه", "وسلم", "قال", "ان", "أن",
                "عن", "رسول", "النبي",
            }
        ]
        if len(tail_tokens) >= 4:
            return True
    return False


def _is_unidentified_hadith_claim(text: str) -> bool:
    """A claim *about* a Hadith without its matn or a usable reference."""
    normalized = _normalize(text)
    if "حديث" not in normalized and "روايه" not in normalized and "رواية" not in normalized:
        return False
    has_authenticity_claim = any(
        marker in normalized for marker in _HADITH_AUTHENTICITY_MARKERS
    )
    has_meta_gap = any(
        marker in normalized for marker in _HADITH_META_MARKERS
    )
    if not (has_authenticity_claim or has_meta_gap):
        return False
    return not _has_identifiable_hadith_narration(text)


def _classify_source(text: str) -> str:
    if "﴿" in text or _contains(text, _ATTRIBUTION_QURAN):
        return "quran_quote"
    if _contains(text, _ATTRIBUTION_HADITH):
        return "hadith_quote"
    if _SEERAH_RE.search(_normalize(text)):
        return "seerah_claim"
    return "quote"


def _component_skeleton(
    quote: str,
    context: str,
    paragraph_index: int,
    order: int,
) -> dict:
    source_type = _classify_source(f"{context} {quote}")
    return {
        "id": f"c{order}",
        "type": source_type,
        "type_label": TYPE_LABELS.get(source_type, "عنصر"),
        "submitted_excerpt": quote,
        "author_context": _clean_author_context(context),
        "claim_text": "",
        "source_identity": {
            "status": "inconclusive",
            "label": _SOURCE_STATUS_LABELS["inconclusive"],
        },
        "canonical_text": "",
        "canonical_label": "النص الكامل من المصدر",
        "reference": "",
        "source_name": "",
        "completeness": {
            "status": "unknown",
            "label": _COMPONENT_STATUS_LABELS["unknown"],
            "is_material": None,
            "omitted_text": "",
            "explanation": "",
            "materiality_basis": "",
        },
        "authenticity": {
            "status": "not_applicable",
            "label": "",
            "source": "",
            "assessment": "",
            "assessor": "",
            "evidence": [],
        },
        "interpretation": {
            "status": "not_provided",
            "label": _INTERPRETATION_LABELS["not_provided"],
            "explanation": "",
            "correction": "",
            "basis": "not_assessed",
        },
        "argument": {
            "status": "not_assessed",
            "label": _ARGUMENT_LABELS["not_assessed"],
            "explanation": "",
            "basis": "not_assessed",
        },
        "evidence": [],
        "severity": "none",
        "paragraph_index": paragraph_index,
        "_match": None,
    }


def _is_rhetorical(text: str) -> bool:
    """Return True for a short rhetorical framing sentence.

    These sentences introduce a post but cannot be checked against a
    religious source on their own.  They must not become standalone source
    lookups.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    normalized = _normalize(raw)
    tokens = normalized.split()
    if len(tokens) > 22:
        return False
    rhetorical_markers = (
        "ناس كتير", "ناس كثير", "فاهمه الدين غلط", "فاهمين الدين غلط",
        "يا ناس", "للاسف", "بصراحه", "تخيل", "مش معقول", "عجيب",
        "غريب", "انتبه", "اسمع",
    )
    content_markers = (
        "واجب", "حرام", "حلال", "يجوز", "سنه", "بدعه", "فرض", "حكم",
        "حديث", "ايه", "سوره", "صلي", "النبي", "الله", "قران",
        "زكاه", "صوم", "حج", "صلاه", "وضوء", "طلاق", "ميراث",
    )
    if any(marker in normalized for marker in content_markers):
        return False
    if any(marker in normalized for marker in rhetorical_markers):
        return True
    if raw.endswith(("!", "؟")) and len(tokens) <= 12:
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!؟])\s+|\n+", text or "")
        if sentence.strip()
    ]


def extract_components(text: str) -> list[dict]:
    """Extract quotations and group their related claims.

    Quotations remain independent components so each can be verified
    separately.  Rhetorical introductions and explanatory/concluding
    sentences are attached to the quotation they belong to instead of being
    treated as standalone religious claims.
    """
    text = text or ""
    spans = _quote_spans(text)
    components: list[dict] = []
    order = 0

    if spans:
        for index, span in enumerate(spans):
            order += 1
            quote_start, quote_end = span["start"], span["end"]
            sentence_start, _sentence_end = _sentence_around(
                text, quote_start, quote_end,
            )
            before_start = 0 if index == 0 else sentence_start
            before_text = text[before_start:quote_start]

            if index + 1 < len(spans):
                next_span = spans[index + 1]
                next_sentence_start, _ = _sentence_around(
                    text, next_span["start"], next_span["end"],
                )
                after_end = max(quote_end, next_sentence_start)
            else:
                after_end = len(text)
            after_text = text[quote_end:after_end]

            context = before_text + " " + after_text
            component = _component_skeleton(
                span["quote"], context, 0, order,
            )

            preface_parts: list[str] = []
            claim_parts: list[str] = []
            for sentence in _split_sentences(before_text):
                if _is_rhetorical(sentence):
                    preface_parts.append(sentence)
                else:
                    cleaned = _clean_author_context(sentence)
                    if cleaned:
                        claim_parts.append(cleaned)
            cleaned_after = _clean_author_context(after_text)
            if cleaned_after:
                claim_parts.append(cleaned_after)

            component["preface"] = " ".join(preface_parts).strip()
            component["author_context"] = " ".join(claim_parts).strip()
            component["claim_text"] = " ".join(claim_parts).strip()
            components.append(component)
        return components

    # No quotations: keep one grouped claim component for a post that is
    # entirely explanatory, but still split when a new source attribution
    # starts a distinct claim.
    preface_parts = []
    current: dict | None = None
    for sentence in _split_sentences(text):
        if _is_rhetorical(sentence):
            preface_parts.append(sentence)
            continue
        kind = _classify_source(sentence)
        if kind == "hadith_quote" and not _has_identifiable_hadith_narration(sentence):
            kind = "claim"
        unidentified = _is_unidentified_hadith_claim(sentence)
        if current is not None and kind in {"claim", "quote"}:
            current["claim_text"] = (
                current.get("claim_text", "") + " " + sentence
            ).strip()
            current["author_context"] = current["claim_text"]
            if unidentified:
                current["unresolved_kind"] = "unidentified_hadith_claim"
                current["type"] = "claim"
                current["type_label"] = TYPE_LABELS[
                    "unidentified_hadith_claim"
                ]
                current["missing_information"] = [
                    "نص الحديث",
                    "اسم الكتاب أو المصدر",
                    "رقم الحديث أو المرجع",
                ]
            continue
        order += 1
        current = _component_skeleton(sentence, "", 0, order)
        if kind in {"claim", "quote"}:
            current["type"] = "claim"
            current["type_label"] = TYPE_LABELS["claim"]
            current["source_identity"] = {
                "status": "not_found",
                "label": _SOURCE_STATUS_LABELS["not_found"],
            }
        if unidentified:
            current["unresolved_kind"] = "unidentified_hadith_claim"
            current["type"] = "claim"
            current["type_label"] = TYPE_LABELS[
                "unidentified_hadith_claim"
            ]
            current["missing_information"] = [
                "نص الحديث",
                "اسم الكتاب أو المصدر",
                "رقم الحديث أو المرجع",
            ]
            current["source_identity"] = {
                "status": "not_found",
                "label": "لا يمكن تحديد الحديث",
            }
        current["preface"] = " ".join(preface_parts).strip()
        current["claim_text"] = sentence
        current["author_context"] = sentence
        preface_parts = []
        components.append(current)
    return components


# ---------------------------------------------------------------------------
# Source resolution per component
# ---------------------------------------------------------------------------

def _tafsir_evidence_excerpt(text: str, limit: int = 1400) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    normalized = _normalize(clean)
    for marker in ("اختلف", "قولين", "قولان", "وقال آخرون", "الجمهور", "خلاف"):
        position = normalized.find(marker)
        if position >= 0:
            raw_position = int(len(clean) * position / max(1, len(normalized)))
            start = max(0, raw_position - 500)
            end = min(len(clean), raw_position + 1200)
            return clean[start:end]
    return clean[:limit]


def _clean_evidence_text(text: str) -> str:
    return " ".join(
        (text or "")
        .replace("\u200e", " ")
        .replace("\u200f", " ")
        .split()
    )


def _find_tafsir_evidence(chunks: list[dict], surah, ayah) -> dict | None:
    if surah is None or ayah is None:
        return None
    wanted = f"تفسير ابن كثير {surah}:{ayah}"
    for chunk in chunks:
        if chunk.get("source") != "tafsir":
            continue
        if chunk.get("reference") == wanted:
            return chunk
    for chunk in chunks:
        if chunk.get("source") != "tafsir":
            continue
        metadata = chunk.get("metadata") or {}
        if (
            metadata.get("surah_number") == surah
            and metadata.get("ayah_number") == ayah
        ):
            return chunk
    return None


_DISAGREEMENT_ASSERTION_MARKERS = (
    "متنازع", "محل خلاف", "خلاف تفسيري", "اختلاف تفسيري",
    "عليه خلاف", "توجد خلافات", "قولان", "قولين", "مذهبان",
    "أقوال مختلفة",
)


def _asserts_disagreement(reason: str) -> bool:
    normalized = _normalize(reason or "")
    if not normalized:
        return False
    for negation in (
        "لا خلاف", "لا يوجد خلاف", "لا توجد خلافات", "لم يثبت خلاف",
        "دون خلاف", "ليس محل خلاف", "غير متنازع",
    ):
        if negation in normalized:
            return False
    return any(
        _normalize(marker) in normalized
        for marker in _DISAGREEMENT_ASSERTION_MARKERS
    )


_OTHER_TEXT_CLAIM_MARKERS = (
    "نصوص اخرى", "النصوص الاخري", "نص اخر", "النص الاخر",
    "احاديث اخرى", "الاحاديث الاخري", "حديث اخر", "الحديث الاخر",
    "ايات اخرى", "الايات الاخري", "ايه اخرى", "الايه الاخري",
    "آيات أخرى", "النصوص الأخرى", "أحاديث أخرى", "الحديث الآخر",
    "نصوص أخرى", "نص آخر", "حديث آخر", "آية أخرى",
)


def _asserts_unlocated_other_text(
    reason: str,
    component: dict,
) -> bool:
    """True when a reason relies on unnamed other texts.

    If the reason names a retrieved reference, the claim can be audited and
    is left alone.  Otherwise the explanation must not imply that unseen
    texts establish exceptions or a different ruling.
    """
    normalized = _normalize(reason or "")
    if not any(
        _normalize(marker) in normalized
        for marker in _OTHER_TEXT_CLAIM_MARKERS
    ):
        return False
    for item in component.get("evidence", []):
        reference = _normalize(str(item.get("reference") or ""))
        if not reference:
            continue
        if reference in normalized:
            return False
        core = re.sub(r"\d+.*$", "", reference).strip()
        if core and core in normalized:
            return False
    return True


def _has_explicit_disagreement(component: dict, reason: str) -> bool:
    evidence_text = " ".join(
        _normalize(item.get("text", ""))
        for item in component.get("evidence", [])
        if item.get("text")
    )
    reason_text = _normalize(reason or "")
    evidence_markers = (
        "اختلف", "قولان", "قولين", "وقال آخرون", "الجمهور",
        "على قولين", "مذهبان", "خلاف",
    )
    reason_markers = (
        "اختلاف", "خلاف", "قول", "مذهب", "جمهور", "رأي", "آخر",
    )
    return (
        any(marker in evidence_text for marker in evidence_markers)
        and any(marker in reason_text for marker in reason_markers)
    )


def _resolve_component(
    component: dict,
    chunks: list[dict],
) -> dict:
    if component.get("unresolved_kind") == "unidentified_hadith_claim":
        component["source_identity"] = {
            "status": "not_found",
            "label": "لا يمكن تحديد الحديث",
        }
        component["completeness"] = {
            "status": "unknown",
            "label": _COMPONENT_STATUS_LABELS["unknown"],
            "is_material": False,
            "omitted_text": "",
            "explanation": (
                "لا يوجد نص حديث محدد لمقارنته بالمصدر، لذلك لا يمكن "
                "تقييم الاكتمال."
            ),
            "materiality_basis": (
                "لم يتم تقديم نص الحديث، فلا يمكن تقييم أثر أي جزء محذوف."
            ),
        }
        component["authenticity"] = {
            "status": "not_found",
            "label": "لا يمكن تقييم درجة الحديث",
            "source": "",
            "assessment": (
                "لم يذكر المنشور نص الحديث ولا اسم الكتاب ولا رقم الحديث. "
                "لا يمكن نسبة درجة صحة أو وضع إلى حديث غير محدد، وعدم "
                "العثور عليه لا يعني أنه موضوع."
            ),
            "assessor": "",
            "evidence": [],
        }
        component["interpretation"] = {
            "status": "inconclusive",
            "label": "غير محسوم — لا يمكن تحديد الحديث",
            "explanation": (
                "لا يمكن تقييم شرح حديث لم يتم تحديد نصه أو مرجعه."
            ),
            "correction": "",
            "basis": "deterministic",
        }
        component["argument"] = {
            "status": "inconclusive",
            "label": "لا يمكن الحكم على الاستدلال بدون تحديد الحديث",
            "explanation": (
                "الحكم على صحة نسبة الحديث أو معنى الاستدلال يتطلب نص "
                "الحديث أو مرجعاً يمكن التحقق منه."
            ),
            "basis": "deterministic",
        }
        component["evidence"] = []
        component["_match"] = None
        return component

    excerpt = component.get("submitted_excerpt", "")
    source_type = component.get("type", "claim")
    quran = None
    hadith = None
    historical = None

    if source_type in {"quran_quote", "quote"}:
        try:
            quran = match_quran_quotation(excerpt, chunks)
        except RetrievalFailure:
            raise
        except Exception:
            quran = None
    if not quran or not quran.get("found"):
        if source_type in {
            "hadith_quote", "quote", "seerah_claim", "quran_quote",
        }:
            try:
                hadith = match_hadith_quote(excerpt, chunks)
            except RetrievalFailure:
                raise
            except Exception:
                hadith = None
    if (
        source_type in {"seerah_claim", "quote"}
        and (quran is None or not quran.get("found"))
        and (hadith is None or not hadith.get("found"))
    ):
        try:
            historical = match_historical_claim(excerpt, chunks)
        except RetrievalFailure:
            raise
        except Exception:
            historical = None

    if quran and quran.get("found"):
        component["type"] = "quran_quote"
        component["type_label"] = TYPE_LABELS["quran_quote"]
        component["_match"] = quran
        component["source_name"] = "quran"
        component["canonical_text"] = quran.get("canonical_text", "")
        component["reference"] = quran.get("reference", "")
        component["source_identity"] = {
            "status": (
                "verified" if quran.get("is_complete")
                else "partial" if quran.get("match_type") == "partial"
                else "approximate"
            ),
            "label": (
                "مطابق للآية كاملة" if quran.get("is_complete")
                else "جزء من آية مطابقة" if quran.get("match_type") == "partial"
                else "مطابقة تقريبية"
            ),
        }
        component["completeness"] = {
            "status": "complete" if quran.get("is_complete") else "partial_not_material",
            "label": (
                _COMPONENT_STATUS_LABELS["complete"]
                if quran.get("is_complete")
                else _COMPONENT_STATUS_LABELS["partial_not_material"]
            ),
            "is_material": None if not quran.get("is_complete") else False,
            "omitted_text": (
                "" if quran.get("is_complete")
                else _omitted_text(
                    quran.get("canonical_text", ""), excerpt,
                )
            ),
            "explanation": (
                "الاقتباس يطابق النص القرآني كاملاً."
                if quran.get("is_complete")
                else "الاقتباس جزء حرفي من الآية، ويظهر النص الكامل للمراجعة."
            ),
        }
        component["authenticity"] = {
            "status": "not_applicable",
            "label": "نص قرآني ثابت في المصحف",
            "source": quran.get("reference", ""),
            "assessment": "",
            "assessor": "",
            "evidence": [],
        }
        component["evidence"] = [{
            "source": "quran",
            "reference": quran.get("reference", ""),
            "text": quran.get("canonical_text", ""),
            "role": "canonical",
        }]
        tafsir_chunk = _find_tafsir_evidence(
            chunks, quran.get("surah_number"), quran.get("ayah_number"),
        )
        if tafsir_chunk is not None:
            tafsir_text = _tafsir_evidence_excerpt(
                tafsir_chunk.get("original_text") or "",
            )
            component["evidence"].append({
                "source": "tafsir",
                "reference": tafsir_chunk.get("reference", ""),
                "text": tafsir_text,
                "role": "interpretation",
            })
    elif hadith and hadith.get("found"):
        component["type"] = "hadith_quote"
        component["type_label"] = TYPE_LABELS["hadith_quote"]
        component["_match"] = hadith
        component["source_name"] = "hadith"
        component["canonical_text"] = hadith.get("matn", "")
        component["reference"] = hadith.get("reference", "")
        references = hadith.get("references") or []
        match_type = hadith.get("match_type", "partial")
        attributable = (
            match_type in {"exact", "partial"}
            and not hadith.get("unmatched_tokens")
        )
        component["additional_references"] = references[1:]
        if attributable:
            component["source_identity"] = {
                "status": (
                    "verified" if hadith.get("is_complete") else "partial"
                ),
                "label": (
                    "مطابق لمتن الحديث" if hadith.get("is_complete")
                    else "جزء من متن حديث مطابق"
                )
                + (
                    " — ورد في أكثر من مصدر" if len(references) > 1 else ""
                ),
            }
            component["completeness"] = {
                "status": (
                    "complete" if hadith.get("is_complete")
                    else "partial_not_material"
                ),
                "label": (
                    _COMPONENT_STATUS_LABELS["complete"]
                    if hadith.get("is_complete")
                    else _COMPONENT_STATUS_LABELS["partial_not_material"]
                ),
                "is_material": (
                    None if not hadith.get("is_complete") else False
                ),
                "omitted_text": (
                    "" if hadith.get("is_complete")
                    else _omitted_text(hadith.get("matn", ""), excerpt)
                ),
                "explanation": (
                    "الاقتباس يطابق متن الحديث كما هو مسجل."
                    if hadith.get("is_complete")
                    else "الاقتباس جزء من متن الحديث، ويظهر المتن الكامل للمراجعة."
                ),
            }
            component["authenticity"] = assess_hadith(
                hadith.get("chunk") or {}, hadith.get("matn", ""), chunks,
            )
        else:
            # A partial/approximate match that contains submitted words
            # outside the canonical matn is not an exact narration.  Do not
            # let the collection name (e.g. Sahih al-Bukhari) authenticate a
            # different wording.
            component["source_identity"] = {
                "status": (
                    "altered" if match_type == "altered"
                    else "approximate"
                ),
                "label": (
                    "النص يختلف عن المتن المسجل"
                    if match_type == "altered"
                    else "مطابقة تقريبية لمتن مسجل"
                )
                + (
                    " — ورد في أكثر من مصدر" if len(references) > 1 else ""
                ),
            }
            component["completeness"] = {
                "status": "altered",
                "label": _COMPONENT_STATUS_LABELS["altered"],
                "is_material": True,
                "omitted_text": _omitted_text(
                    hadith.get("matn", ""), excerpt,
                ),
                "explanation": (
                    "النص المقدم يحتوي ألفاظاً لا تطابق المتن المسجل. "
                    "لا تنسب درجة المصدر إلى الصياغة المختلفة قبل مراجعة "
                    "المتن الأصلي."
                ),
            }
            component["authenticity"] = {
                "status": "not_recorded",
                "label": "لم تُنسب درجة الحديث لهذا النص المختلف",
                "source": hadith.get("reference", ""),
                "assessment": (
                    "درجة المصدر تخص المتن المسجل، ولا يصح نقلها إلى "
                    "صياغة تحتوي ألفاظاً إضافية أو مختلفة."
                ),
                "assessor": "",
                "evidence": [],
            }
        component["evidence"] = [{
            "source": "hadith",
            "reference": hadith.get("reference", ""),
            "text": hadith.get("matn", ""),
            "role": "canonical",
        }]
        candidate_texts = {
            str(candidate.get("reference", "")): str(candidate.get("text", ""))
            for candidate in (hadith.get("candidates") or [])
        }
        for extra_reference in references[1:4]:
            component["evidence"].append({
                "source": "hadith",
                "reference": extra_reference,
                "text": _clean_evidence_text(
                    candidate_texts.get(extra_reference)
                    or hadith.get("matn", "")
                ),
                "role": "additional_reference",
            })
        for item in component["authenticity"].get("evidence", [])[:3]:
            component["evidence"].append({
                "source": "grading",
                "reference": item.get("source", ""),
                "text": item.get("assessment", ""),
                "role": "grading",
            })
    elif (
        source_type == "hadith_quote"
        and not (hadith and hadith.get("found"))
    ):
        # The narration may be absent from the Hadith collection while a
        # Tafsir/Fiqh passage quotes it together with a documented grading.
        external = find_external_grading(excerpt, chunks)
        if external:
            merged = _merge_grading(None, external)
            component["type"] = "hadith_quote"
            component["type_label"] = TYPE_LABELS["hadith_quote"]
            component["source_name"] = "hadith_external"
            component["canonical_text"] = (
                external[0].get("matched_excerpt")
                or external[0].get("assessment", "")
            )
            component["canonical_label"] = "المقطع الذي ورد فيه النص والتقييم"
            component["reference"] = external[0].get("source", "")
            component["source_identity"] = {
                "status": "partial",
                "label": "ورد النص في مصدر تفسيري/فقهي لا في قاعدة الحديث",
            }
            component["completeness"] = {
                "status": "unknown",
                "label": _COMPONENT_STATUS_LABELS["unknown"],
                "is_material": None,
                "omitted_text": "",
                "explanation": (
                    "لم يُعثر على متن الحديث كاملاً في قاعدة الحديث؛ "
                    "التقييم منقول من المصدر الذي ذكر النص."
                ),
            }
            component["authenticity"] = merged
            component["evidence"] = [
                {
                    "source": "grading",
                    "reference": item.get("source", ""),
                    "text": item.get("matched_excerpt") or item.get("assessment", ""),
                    "role": "grading",
                }
                for item in external[:3]
            ]
        else:
            component["source_identity"] = {
                "status": "not_found",
                "label": _SOURCE_STATUS_LABELS["not_found"],
            }
            component["completeness"] = {
                "status": "unknown",
                "label": _COMPONENT_STATUS_LABELS["unknown"],
                "is_material": None,
                "omitted_text": "",
                "explanation": (
                    "لم يُعثر على النص في قاعدة البيانات المتاحة. هذا لا "
                    "يعني بالضرورة أنه غير صحيح أو مكذوب."
                ),
            }
            component["authenticity"] = {
                "status": "not_found",
                "label": "لم يُعثر عليه في قاعدة البيانات المتاحة",
                "source": "",
                "assessment": (
                    "عدم العثور على الرواية في قاعدة البيانات الحالية لا "
                    "يعني بالضرورة أنها موضوعة، ولا يثبت أنها صحيحة؛ لا "
                    "تتوفر أدلة كافية للحكم."
                ),
                "assessor": "",
                "evidence": [],
            }
            component["evidence"] = []
    elif historical and historical.get("found"):
        component["type"] = "seerah_claim"
        component["type_label"] = TYPE_LABELS["seerah_claim"]
        component["_match"] = historical
        component["source_name"] = historical.get("source", "")
        component["canonical_text"] = historical.get("canonical_text", "")
        component["canonical_label"] = "المقطع المطابق من المصدر التاريخي"
        component["reference"] = historical.get("reference", "")
        component["source_identity"] = {
            "status": "partial" if historical.get("match_type") == "partial" else "verified",
            "label": "ورد الخبر في المصدر المتاح",
        }
        component["completeness"] = {
            "status": "partial_not_material",
            "label": _COMPONENT_STATUS_LABELS["partial_not_material"],
            "is_material": None,
            "omitted_text": "",
            "explanation": "تم العثور على مقطع مطابق من المصدر التاريخي.",
        }
        component["authenticity"] = assess_reliability(
            excerpt, chunks, historical,
        )
        component["evidence"] = [{
            "source": historical.get("source", ""),
            "reference": historical.get("reference", ""),
            "text": historical.get("canonical_text", ""),
            "role": "historical",
        }]
    else:
        component["source_identity"] = {
            "status": "not_found",
            "label": _SOURCE_STATUS_LABELS["not_found"],
        }
        component["completeness"] = {
            "status": "unknown",
            "label": _COMPONENT_STATUS_LABELS["unknown"],
            "is_material": None,
            "omitted_text": "",
            "explanation": (
                "لم يُعثر على النص في قاعدة البيانات المتاحة. هذا لا يعني "
                "بالضرورة أنه غير صحيح أو مكذوب."
            ),
        }
        if component.get("type") == "seerah_claim":
            component["authenticity"] = {
                "status": "not_found",
                "label": "لم يُعثر عليه في قاعدة البيانات المتاحة",
                "source": "",
                "assessment": (
                    "عدم وجود الخبر في قاعدة البيانات الحالية لا يعني أنه "
                    "مكذوب؛ لا تتوفر أدلة كافية للحكم على ثبوته."
                ),
                "assessor": "",
                "evidence": [],
            }
        component["evidence"] = []

    if component.get("claim_text"):
        component["interpretation"] = {
            "status": "inconclusive",
            "label": _INTERPRETATION_LABELS["inconclusive"],
            "explanation": "",
            "correction": "",
        }
        component["argument"] = {
            "status": "inconclusive",
            "label": _ARGUMENT_LABELS["inconclusive"],
            "explanation": "",
        }
    return component


# ---------------------------------------------------------------------------
# Model-assisted interpretation and argument assessment
# ---------------------------------------------------------------------------

POST_SYSTEM_PROMPT = (
    "أنت مراجع نقدي للمحتوى الديني. لديك منشور كامل وعناصره، ومصادرها "
    "المحسوبة مسبقاً. مهمتك تقييم التفسير والاستدلال فقط.\n\n"
    "قواعد صارمة:\n"
    "- لا تغيّر حكم وجود النص أو مرجعه أو درجة الحديث؛ هذه حقائق محسوبة "
    "وصلت إليك في الحقول.\n"
    "- لا تخترع آية أو حديثاً أو مرجعاً أو عالماً.\n"
    "- الاقتباس الناقص ليس مضللاً تلقائياً. احكم على تأثير النقص في سياق "
    "دعوى المؤلف. إذا كان الاستنتاج يعتمد على إطلاق أو تحريم أو حكم عام "
    "يقيّده السياق المحذوف فاجعل completeness_assessment=partial_material.\n"
    "- إذا كان الاختلاف تفسيرياً معتبراً فاستخدم disputed، ولا تعتبر مجرد "
    "المخالفة تفسيراً باطلاً.\n"
    "- إذا لم تكف الأدلة فاستخدم inconclusive.\n"
    "- إذا لم يقدم المؤلف تفسيراً صريحاً فاستخدم not_provided.\n"
    "- إذا لم يقدم استدلالاً صريحاً فاستخدم not_assessed.\n"
    "- التصحيح يجب أن يستند إلى الأدلة المقدمة فقط.\n\n"
    "أعد JSON فقط:\n"
    "{\n"
    '  "components": [\n'
    "    {\n"
    '      "id": "c1",\n'
    '      "completeness_assessment": '
    '"complete|partial_not_material|partial_material|altered|unknown",\n'
    '      "interpretation_status": '
    '"supported|contradicted|partially_supported|disputed|inconclusive|not_provided",\n'
    '      "argument_status": '
    '"supported|unsupported|overstated|misleading|disputed|inconclusive|not_assessed",\n'
    '      "completeness_explanation": "سبب حكم الاكتمال أو فارغ",\n'
    '      "interpretation_explanation": "سبب حكم الشرح أو فارغ",\n'
    '      "argument_explanation": "سبب حكم الاستدلال أو فارغ",\n'
    '      "explanation": "ملخص الحكم في جملتين",\n'
    '      "correction": "تصحيح مستند إلى الأدلة أو فارغ",\n'
    '      "severity": "none|low|medium|high"\n'
    "    }\n"
    "  ]\n"
    "}"
)

_COMPLETENESS_VALUES = {
    "complete", "partial_not_material", "partial_material", "altered", "unknown",
}
_INTERPRETATION_VALUES = {
    "supported", "contradicted", "partially_supported", "disputed",
    "inconclusive", "not_provided",
}
_ARGUMENT_VALUES = {
    "supported", "unsupported", "overstated", "misleading", "disputed",
    "inconclusive", "not_assessed",
}


def _evidence_payload(component: dict) -> list[dict]:
    seen = set()
    payload = []
    for item in component.get("evidence", []):
        key = (item.get("reference"), item.get("text", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        payload.append({
            "source": item.get("source", ""),
            "reference": item.get("reference", ""),
            "text": (item.get("text", "") or "")[:1500],
            "role": item.get("role", ""),
        })
    return payload[:4]


def assess_with_model(
    post_text: str,
    components: list[dict],
    client,
    *,
    debug: bool = False,
) -> dict[str, dict]:
    if client is None:
        return {}
    payload = []
    for component in components:
        payload.append({
            "id": component["id"],
            "type": component["type_label"],
            "excerpt": component.get("submitted_excerpt", "")[:2500],
            "claim_or_explanation": component.get("claim_text", "")[:2500],
            "source_reference": component.get("reference", ""),
            "canonical_text": component.get("canonical_text", "")[:3000],
            "source_identity": component.get("source_identity", {}),
            "completeness": component.get("completeness", {}),
            "authenticity": {
                key: value
                for key, value in component.get("authenticity", {}).items()
                if key != "evidence"
            },
            "evidence": _evidence_payload(component),
        })
    user_message = (
        "المنشور الكامل:\n"
        + (post_text or "")[:6000]
        + "\n\nالعناصر:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": POST_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if debug:
            print(f"[post_analysis] model assessment:\n{raw}")
        parsed = json.loads(raw)
    except Exception as error:
        if debug:
            print(f"[post_analysis] assessment failed: {error}")
        return {}
    results = {}
    for item in parsed.get("components", []):
        if not isinstance(item, dict):
            continue
        component_id = item.get("id")
        if component_id not in {component["id"] for component in components}:
            continue
        results[component_id] = item
    return results


_QUALIFYING_MARKERS = (
    "حتي", "حتى", "الا", "إلا", "اذا", "إذا", "بشرط", "في حال",
    "علي سفر", "على سفر", "مرضي", "مرضى", "جنبا", "جنب",
    "وانتم سكارى", "وأنتم سكارى", "غير", "لكن", "حتي تعلموا",
    "حتى تعلموا",
)

_ABSOLUTE_CLAIM_MARKERS = (
    "وحدها", "وحده", "فقط", "بس", "دون", "بدون", "دائم", "دائما", "ابدا",
    "مطلق", "كل", "جميع", "مهما", "بغض النظر", "لا يشترط",
    "يكفي", "تكفي", "مش على", "ليس على", "غير محاسب",
)


def _claim_is_absolute(claim: str) -> bool:
    normalized = _normalize(claim or "")
    markers = (
        "وحدها", "وحده", "فقط", "بس", "دون", "بدون", "دائم", "ابدا",
        "مطلق", "كل", "جميع", "مهما", "بغض النظر", "لا يشترط",
        "يكفي", "تكفي", "مش على", "ليس على", "غير محاسب",
        "اي حاجه", "أي حاجة", "اي شيء", "أي شيء",
    )
    return any(marker in normalized for marker in markers)


def _material_omission_supported(component: dict) -> bool:
    completeness = component.get("completeness", {})
    omitted = _normalize(completeness.get("omitted_text", ""))
    claim = _normalize(component.get("claim_text", ""))
    return bool(
        omitted
        and any(_normalize(marker) in omitted for marker in _QUALIFYING_MARKERS)
        and any(_normalize(marker) in claim for marker in _ABSOLUTE_CLAIM_MARKERS)
    )


def _apply_materiality_heuristic(components: list[dict]) -> None:
    """Flag an omission as material when it removes a qualification that the
    author's own unsupported conclusion depends on.

    This is not a verdict by itself: the interpretation/argument assessment
    must already be contradicted/unsupported, and the omitted canonical text
    must contain an explicit qualifier.  A partial quotation with a supported
    conclusion is left as ``partial_not_material``.
    """
    for component in components:
        completeness = component.get("completeness", {})
        if completeness.get("status") != "partial_not_material":
            continue
        if not _material_omission_supported(component):
            continue
        interpretation = component.get("interpretation", {}).get("status")
        argument = component.get("argument", {}).get("status")
        if interpretation not in {"contradicted"} and argument not in {
            "unsupported", "misleading", "overstated",
        }:
            continue
        completeness["status"] = "partial_material"
        completeness["label"] = _COMPONENT_STATUS_LABELS["partial_material"]
        completeness["is_material"] = True
        note = (
            "النص المحذوف يحتوي قيداً أو تفصيلاً مؤثراً في الاستنتاج "
            "المذكور؛ لذلك لا يصح التعامل مع الاقتباس كأنه مستقل عن سياقه."
        )
        if note not in (completeness.get("explanation") or ""):
            completeness["explanation"] = (
                (completeness.get("explanation") or "").strip()
                + " " + note
            ).strip()


_CORRECTION_BOOK_NAMES = (
    "صحيح البخاري", "صحيح مسلم", "سنن أبي داود", "جامع الترمذي",
    "سنن النسائي", "سنن ابن ماجه", "موطأ مالك", "مسند أحمد",
    "تفسير ابن كثير",
)


def _correction_citations_supported(
    correction: str,
    evidence: list[dict],
) -> bool:
    """Reject a model correction that cites a reference not in evidence."""
    references = " ".join(
        _normalize(item.get("reference", ""))
        for item in evidence
        if item.get("reference")
    )
    if not references:
        # No reference-bearing evidence: allow only a correction without a
        # citation, and never a named book/collection.
        return not any(
            _normalize(name) in _normalize(correction)
            for name in _CORRECTION_BOOK_NAMES
        )
    evidence_numbers = {
        tuple(re.findall(r"\d+", item.get("reference", "")))
        for item in evidence
        if item.get("reference")
    }
    for numeric in re.findall(
        r"\d{1,3}\s*[:：]\s*\d{1,3}", correction,
    ):
        numbers = tuple(re.findall(r"\d+", numeric))
        if numbers and numbers not in evidence_numbers:
            return False
    # Arabic Surah/verse citations such as "النساء:44" have only one digit
    # run, so check the named reference explicitly.
    for name, ayah in re.findall(
        r"([\u0621-\u064A]{2,})\s*[:：]\s*(\d{1,3})", correction,
    ):
        normalized_name = _normalize(name)
        matching_refs = [
            item.get("reference", "")
            for item in evidence
            if normalized_name in _normalize(item.get("reference", ""))
        ]
        if not matching_refs:
            return False
        if not any(
            ayah in re.findall(r"\d+", ref) for ref in matching_refs
        ):
            return False
    for name in _CORRECTION_BOOK_NAMES:
        if _normalize(name) in _normalize(correction):
            if _normalize(name) not in references:
                return False
    return True


def _correction_text_supported(
    correction: str,
    evidence: list[dict],
) -> bool:
    """Require correction wording to overlap the displayed evidence."""
    tokens = {
        token for token in _normalize(correction).split()
        if len(token) >= 3
    }
    if not tokens:
        return False
    evidence_text = _normalize(" ".join(
        item.get("text", "") for item in evidence
    ))
    overlap = sum(1 for token in tokens if token in evidence_text)
    return overlap / len(tokens) >= 0.5


def _deterministic_correction(component: dict) -> str:
    """Build a short correction that only states what the evidence shows."""
    reference = (component.get("reference") or "المصدر المتاح").strip()
    authenticity = component.get("authenticity") or {}
    completeness = component.get("completeness") or {}
    argument = component.get("argument") or {}
    auth_status = authenticity.get("status")
    if auth_status in {"mawdu", "munkar", "not_authentic", "batil"}:
        return (
            f"المصدر {reference} يصف الرواية بأنها موضوعة أو منكرة؛ "
            "لا يصح بناء استدلال عليها."
        )
    if auth_status == "daif":
        return (
            f"المصدر {reference} يصف الرواية بأنها ضعيفة؛ لا يصح بناء "
            "حكم قاطع عليها."
        )
    if (
        completeness.get("status") == "partial_material"
        and completeness.get("omitted_text")
    ):
        omitted = " ".join(
            str(completeness.get("omitted_text")).split()
        )[:180]
        return (
            f"المصدر {reference} يوضح أن النص المحذوف يشمل: «{omitted}». "
            "هذا السياق مطلوب قبل اعتماد الاستنتاج."
        )
    if argument.get("status") in {
        "unsupported", "misleading", "overstated",
    }:
        return (
            f"المصدر {reference} يثبت النص المقتبس فقط، ولا يثبت النتيجة "
            "التي ذكرها المؤلف. راجع النص الكامل أعلاه قبل اعتماد الاستنتاج."
        )
    return ""


def _merge_model_assessment(
    components: list[dict],
    assessments: dict[str, dict],
) -> None:
    for component in components:
        if component.get("unresolved_kind") == "unidentified_hadith_claim":
            # The deterministic unresolved result must not be overwritten by
            # a model judgment about an unidentified narration.
            continue
        item = assessments.get(component["id"])
        if not item:
            continue
        completeness = component.get("completeness", {})
        status = item.get("completeness_assessment")
        deterministic_status = completeness.get("status")
        # The model may assess whether a known omission is material, but it
        # must not overwrite the lexical completeness decision.  Otherwise a
        # model could call a partial quotation "complete" or an exact
        # quotation "altered" without any new source evidence.
        allowed_completeness = {
            "partial_not_material": {
                "partial_not_material", "partial_material",
            },
            "partial_material": {
                "partial_not_material", "partial_material",
            },
            "altered": {"altered", "partial_material"},
            "complete": {"complete"},
            "unknown": {"unknown"},
        }.get(deterministic_status, set())
        if status in _COMPLETENESS_VALUES and status in allowed_completeness:
            if (
                status == "partial_material"
                and not _material_omission_supported(component)
            ):
                status = "partial_not_material"
            completeness["status"] = status
            completeness["label"] = _COMPONENT_STATUS_LABELS[status]
            completeness["is_material"] = status in {
                "partial_material", "altered",
            }
        summary = str(item.get("explanation") or "").strip()
        completeness_reason = str(
            item.get("completeness_explanation") or summary
        ).strip()
        interpretation_reason = str(
            item.get("interpretation_explanation") or summary
        ).strip()
        argument_reason = str(
            item.get("argument_explanation") or summary
        ).strip()
        if completeness_reason:
            completeness["explanation"] = completeness_reason

        has_source_evidence = bool(component.get("evidence"))
        interpretation_status = item.get("interpretation_status")
        if (
            interpretation_status == "not_provided"
            and component.get("claim_text")
        ):
            # The author did provide an explanation/claim; "not provided"
            # would under-report it.
            interpretation_status = "inconclusive"
        if not has_source_evidence:
            interpretation_reason = ""
            argument_reason = ""
        if not has_source_evidence and interpretation_status not in {
            "inconclusive", "not_provided",
        }:
            # A model judgment about a narration that has no matched source
            # must not become a contradiction/misleading verdict.
            interpretation_status = "inconclusive"
        if (
            interpretation_status == "disputed"
            and not _has_explicit_disagreement(
                component, interpretation_reason,
            )
        ):
            interpretation_status = "inconclusive"
        if (
            interpretation_status != "disputed"
            and _asserts_disagreement(interpretation_reason)
        ):
            # The model asserted a scholarly dispute but the retrieved
            # evidence does not establish one.  Keep the text consistent with
            # the inconclusive status instead of repeating the claim.
            interpretation_reason = (
                "الأدلة المتاحة غير كافية للحكم على صحة هذا التفسير أو "
                "نفيه، ولم يظهر فيها ما يثبت وجود تباين معتبر في تفسير "
                "النص."
            )
        if (
            interpretation_status != "supported"
            and _asserts_unlocated_other_text(
                interpretation_reason, component,
            )
        ):
            interpretation_reason = (
                "الأدلة المستردة لا تحدد نصوصاً أخرى أو استثناءات تدعم "
                "هذا التفسير. المصدر المقتبس لا يثبت النتيجة الأوسع."
            )
        model_correction = str(item.get("correction") or "").strip()
        if interpretation_status in _INTERPRETATION_VALUES:
            component["interpretation"] = {
                "status": interpretation_status,
                "label": _INTERPRETATION_LABELS[interpretation_status],
                "explanation": interpretation_reason,
                "correction": "",
                "basis": "model_assessment",
            }
        argument_status = item.get("argument_status")
        if (
            argument_status == "not_assessed"
            and component.get("claim_text")
        ):
            argument_status = "inconclusive"
        if not has_source_evidence and argument_status not in {
            "inconclusive", "not_assessed",
        }:
            argument_status = "inconclusive"
        if (
            argument_status == "disputed"
            and not _has_explicit_disagreement(
                component, argument_reason,
            )
        ):
            argument_status = "inconclusive"
        if (
            argument_status != "disputed"
            and _asserts_disagreement(argument_reason)
        ):
            argument_reason = (
                "الأدلة المتاحة غير كافية للحكم على مدى صحة هذا "
                "الاستدلال."
            )
        if (
            argument_status not in {"supported"}
            and _asserts_unlocated_other_text(argument_reason, component)
        ):
            argument_reason = (
                "المصدر المقتبس لا يثبت النتيجة الأوسع، والأدلة "
                "المستردة لا تحدد استثناءات تدعمها."
            )
        contradiction_markers = (
            "يتعارض", "تعارض", "مخالف", "غير صحيح", "لا يدعم",
            "يناقض", "مضلل", "خطأ",
        )
        if interpretation_status == "inconclusive" and any(
            marker in _normalize(interpretation_reason)
            for marker in contradiction_markers
        ):
            interpretation_reason = ""
        if argument_status == "inconclusive" and any(
            marker in _normalize(argument_reason)
            for marker in contradiction_markers
        ):
            argument_reason = ""
        if "interpretation" in component:
            component["interpretation"]["explanation"] = interpretation_reason
        argument_basis = "model_assessment"
        if (
            argument_status == "inconclusive"
            and interpretation_status == "supported"
            and component.get("claim_text")
            and not _claim_is_absolute(component.get("claim_text", ""))
            and len(component.get("claim_text", "").split()) <= 25
        ):
            argument_status = "supported"
            argument_reason = (
                "الدعوى المذكورة ضيقة ومتوافقة مع الشرح المدعوم بالمصدر."
            )
            argument_basis = "interpretation_supported"
        if argument_status in _ARGUMENT_VALUES:
            component["argument"] = {
                "status": argument_status,
                "label": _ARGUMENT_LABELS[argument_status],
                "explanation": argument_reason,
                "basis": argument_basis,
            }
        correction = ""
        if has_source_evidence:
            if (
                model_correction
                and _correction_citations_supported(
                    model_correction, component.get("evidence", []),
                )
                and _correction_text_supported(
                    model_correction, component.get("evidence", []),
                )
            ):
                correction = model_correction
            else:
                correction = _deterministic_correction(component)
        component["interpretation"]["correction"] = correction
        if completeness.get("status") == "partial_not_material":
            if argument_status in {
                "unsupported", "misleading", "overstated",
            }:
                completeness["materiality_basis"] = (
                    "النقص لا يغيّر ألفاظ الاقتباس نفسه، لكن الدعوى أوسع "
                    "من دلالة المصدر؛ الحكم على الاستدلال منفصل وغير مدعوم."
                )
            elif (
                interpretation_status == "supported"
                or argument_status == "supported"
            ):
                completeness["materiality_basis"] = (
                    "النقص لا يؤثر في الدعوى الضيقة التي ذكرها المؤلف؛ "
                    "الاقتباس يدعم الدعوى المذكورة دون تجاوزها."
                )
            else:
                completeness["materiality_basis"] = (
                    "لم تثبت المصادر أن النقص يغيّر الدعوى أو يحد منها."
                )
        elif completeness.get("status") == "partial_material":
            completeness["materiality_basis"] = (
                "المصدر يذكر قيداً أو تفصيلاً محذوفاً يؤثر في الدعوى."
            )
        elif completeness.get("status") == "complete":
            completeness["materiality_basis"] = "الاقتباس كامل."
        elif completeness.get("status") == "altered":
            completeness["materiality_basis"] = (
                "النص المقدم يختلف عن ألفاظ المصدر."
            )
        if completeness.get("status") == "partial_not_material":
            # The model's completeness text often says the omission affects
            # the "full meaning" even when the materiality decision is that
            # it does not affect this author's claim.  Those two statements
            # cannot coexist in one result.
            completeness["explanation"] = (
                "الاقتباس ناقص نصًّا؛ الجزء المقتبس مطابق للمصدر، "
                "والنص الكامل معروض للمراجعة."
            )
        severity = item.get("severity")
        if severity in {"none", "low", "medium", "high"}:
            component["severity"] = severity


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _legacy_from_component(component: dict) -> dict:
    source_status = component.get("source_identity", {}).get("status")
    completeness = component.get("completeness", {})
    authenticity = component.get("authenticity", {})
    interpretation = component.get("interpretation", {})
    match = component.get("_match") or {}
    source_name = component.get("source_name", "")

    if source_status in {"verified", "partial", "approximate"}:
        if source_name == "quran":
            text_status = "صحيح"
        elif source_name == "hadith":
            auth_status = authenticity.get("status")
            text_status = (
                "صحيح" if auth_status in {
                    "sahih_collection", "hasan_sahih", "sahih", "hasan",
                }
                else "ضعيف" if auth_status == "daif"
                else "موضوع" if auth_status in {"mawdu", "munkar", "not_authentic", "batil"}
                else "غير مؤكد"
            )
        else:
            text_status = "غير مؤكد"
    else:
        text_status = "غير مؤكد"

    explanation_status = {
        "supported": "موافق",
        "contradicted": "مخالف",
        "partially_supported": "جزئياً موافق",
        "disputed": "غير مؤكد",
        "inconclusive": "غير مؤكد",
        "not_provided": "لم يتم تقديم تفسير لتقييمه",
    }.get(interpretation.get("status"), "غير مؤكد")

    quotation = {
        "found": source_status in {"verified", "partial", "approximate"},
        "ambiguous": source_status == "ambiguous",
        "match_type": match.get("match_type", "none"),
        "is_partial": bool(completeness.get("status") in {
            "partial_not_material", "partial_material", "altered",
        }),
        "is_complete": bool(completeness.get("status") == "complete"),
        "coverage": match.get("coverage", 0.0),
        "matched_tokens": match.get("matched_tokens", 0),
        "query_tokens": match.get("query_tokens", 0),
        "canonical_text": component.get("canonical_text", ""),
        "reference": component.get("reference", ""),
        "source": source_name,
        "surah_name": (match.get("surah_name") or ""),
        "surah_number": match.get("surah_number"),
        "ayah_number": match.get("ayah_number"),
        "candidates": match.get("candidates", []),
    }
    return {
        "text_status": text_status,
        "is_truncated": quotation["is_partial"],
        "missing_part": completeness.get("omitted_text", ""),
        "full_text": component.get("canonical_text", ""),
        "source": component.get("reference", ""),
        "danger_level": "غير محسوبة",
        "explanation_status": explanation_status,
        "explanation_verdict": interpretation.get("explanation", ""),
        "correct_explanation": interpretation.get("correction", ""),
        "input_text": "",
        "input_explanation": "",
        "verification_status": source_status if source_status in {
            "verified", "partial", "approximate", "inconclusive",
        } else "inconclusive",
        "status_label": component.get("type_label", "عنصر"),
        "status_explanation": "",
        "quotation": quotation,
        "interpretation": {
            "provided": bool(component.get("claim_text")),
            "status": interpretation.get("status", "inconclusive"),
            "label": interpretation.get("label", ""),
            "assessment": interpretation.get("explanation", ""),
            "correction": interpretation.get("correction", ""),
            "error": "",
        },
        "evidence": component.get("evidence", []),
        "danger_level_meaning": (
            "درجة الخطورة تقدير لاحتمال أن يؤدي الاقتباس أو تفسيره إلى فهم "
            "مضلل، وليست درجة ثقة في صحة النتيجة."
        ),
        "danger_level_applicable": False,
        "retrieval_ok": True,
        "verification_error": "",
    }


def _overall_status(components: list[dict]) -> tuple[str, str, str, str]:
    source_components = [
        component for component in components
        if component.get("source_identity", {}).get("status") in {
            "verified", "partial", "approximate",
        }
    ]
    issues: list[str] = []
    warnings: list[str] = []
    issue_kinds: list[str] = []
    for component in components:
        completeness = component.get("completeness", {})
        interpretation = component.get("interpretation", {})
        argument = component.get("argument", {})
        authenticity = component.get("authenticity", {})
        if completeness.get("status") == "altered":
            issues.append("النص مختلف عن المصدر")
            issue_kinds.append("altered")
        elif completeness.get("is_material"):
            issues.append("سياق محذوف يؤثر في الاستدلال")
            issue_kinds.append("context")
        if interpretation.get("status") == "contradicted":
            issues.append("شرح غير مدعوم بالمصدر")
            issue_kinds.append("interpretation")
        if argument.get("status") == "misleading":
            issues.append("استدلال مضلّل")
            issue_kinds.append("argument_misleading")
        elif argument.get("status") in {"unsupported", "overstated"}:
            issues.append("استدلال غير مدعوم بالمصدر")
            issue_kinds.append("argument_unsupported")
        if authenticity.get("status") in {"mawdu", "munkar", "not_authentic", "batil"}:
            issues.append("رواية موضوعة أو منكرة وفق المصدر")
            issue_kinds.append("authenticity")
        elif authenticity.get("status") == "daif":
            warnings.append("رواية ضعيفة وفق المصدر")
        elif authenticity.get("status") == "disputed":
            warnings.append("خلاف في تقييم الرواية")
        if interpretation.get("status") == "disputed" or argument.get("status") == "disputed":
            warnings.append("خلاف تفسيري معتبر")
        if component.get("source_identity", {}).get("status") == "not_found":
            warnings.append("عنصر لم يُعثر له على مصدر")

    unresolved = [
        component for component in components
        if component.get("unresolved_kind") == "unidentified_hadith_claim"
    ]
    if unresolved and not source_components:
        return (
            "inconclusive",
            "غير محسوم — لا يمكن تحديد الحديث",
            "المنشور يذكر حكماً على حديث دون نصه أو مرجعه؛ لا يمكن تقييم "
            "درجته أو نسبة الاتفاق إليه. عدم العثور على الحديث لا يعني "
            "أنه موضوع.",
            "غير محسوبة",
        )

    if issues:
        label_parts: list[str] = []
        if "authenticity" in issue_kinds:
            label_parts.append("رواية موضوعة أو منكرة")
        if "altered" in issue_kinds:
            label_parts.append("نص مختلف عن المصدر")
        elif "context" in issue_kinds:
            label_parts.append("سياق ناقص")
        if "argument_misleading" in issue_kinds:
            label_parts.append("استدلال مضلّل")
        elif (
            "argument_unsupported" in issue_kinds
            and "interpretation" in issue_kinds
        ):
            label_parts.append("شرح واستدلال غير مدعومين")
        elif "argument_unsupported" in issue_kinds:
            label_parts.append("استدلال غير مدعوم")
        elif "interpretation" in issue_kinds:
            label_parts.append("شرح غير مدعوم")
        label_parts = list(OrderedDict.fromkeys(label_parts))[:2]
        label = "تحذير — " + " و".join(label_parts)
        explanation = "؛ ".join(OrderedDict.fromkeys(issues))
        if warnings:
            explanation += ". " + "؛ ".join(OrderedDict.fromkeys(warnings))
        return "misleading", label, explanation, "عالية"
    if not source_components:
        return (
            "inconclusive",
            "غير محسوم — الأدلة غير كافية",
            "لم يُعثر على مصدر مطابق كافٍ لعناصر المنشور. عدم العثور على "
            "دليل لا يعني بالضرورة أن المحتوى مكذوب.",
            "غير محسوبة",
        )
    if warnings:
        return (
            "partially_verified",
            "تحقق جزئي — بعض العناصر غير محسومة",
            "؛ ".join(OrderedDict.fromkeys(warnings)),
            "متوسطة",
        )
    if all(
        component.get("source_identity", {}).get("status") in {
            "verified", "partial", "approximate",
        }
        and component.get("interpretation", {}).get("status") in {
            "supported", "not_provided",
        }
        and component.get("argument", {}).get("status") in {
            "supported", "not_assessed",
        }
        for component in components
        if component.get("submitted_excerpt")
    ):
        return (
            "verified",
            "تم التحقق — المصادر والاستدلال مدعومة",
            "العناصر المنسوبة إلى المصادر مطابقة أو جزئية غير مضللة، "
            "والشرح والاستدلال مدعومان بالأدلة المتاحة.",
            "منخفضة",
        )
    return (
        "partially_verified",
        "تحقق جزئي — بعض العناصر غير محسومة",
        "تم التحقق من بعض العناصر، وبقيت عناصر أخرى غير محسومة أو غير "
        "مدعومة بالأدلة المتاحة.",
        "متوسطة",
    )


def _post_error_result(
    text: str,
    explanation: str,
    message: str,
) -> dict:
    return {
        "text_status": "خطأ في التحقق",
        "is_truncated": False,
        "missing_part": "",
        "full_text": "",
        "source": "",
        "danger_level": "غير محسوبة",
        "explanation_status": "تعذّر التقييم",
        "explanation_verdict": "",
        "correct_explanation": "",
        "input_text": text,
        "input_explanation": explanation or "",
        "verification_status": "error",
        "status_label": "تعذّر إكمال التحقق",
        "status_explanation": (
            "تعذّر البحث في المصادر أو تحليل المنشور. لم يتم إصدار حكم "
            "على المحتوى."
        ),
        "quotation": {
            "found": False, "ambiguous": False, "match_type": "none",
            "is_partial": False, "is_complete": False, "coverage": 0.0,
            "matched_tokens": 0, "query_tokens": 0,
            "canonical_text": "", "reference": "", "source": "",
            "surah_name": "", "surah_number": None, "ayah_number": None,
            "candidates": [],
        },
        "interpretation": {
            "provided": bool(explanation),
            "status": "error",
            "label": "تعذّر تقييم التفسير",
            "assessment": "",
            "correction": "",
            "error": message,
        },
        "evidence": [],
        "danger_level_meaning": (
            "درجة الخطورة تقدير لاحتمال الفهم المضلل، وليست درجة ثقة."
        ),
        "danger_level_applicable": False,
        "retrieval_ok": False,
        "verification_error": message,
        "components": [],
        "overall_status": "error",
        "component_count": 0,
        "analysis_mode": "post",
    }


def analyze_post(
    text: str,
    explanation: str | None = None,
    index=None,
    chunks: list[dict] | None = None,
    client=None,
    *,
    top_k: int = 8,
    debug: bool = False,
) -> dict | None:
    """Analyze a complete post and return a structured multi-component result.

    ``None`` means the caller should use the legacy single-quote path.
    """
    chunks = chunks or []
    post_text = (text or "").strip()
    if not post_text:
        return None

    # Verify that the retrieval subsystem itself is healthy.  A failed
    # retrieval is an error state, not evidence that the post is false.
    if chunks:
        try:
            _bm25_search(["اختبار"], chunks, top_k=1)
        except Exception as error:
            if debug:
                print(f"[post_analysis] retrieval probe failed: {error}")
            return _post_error_result(
                text, explanation or "",
                "تعذّر البحث في المصادر. لم يتم إصدار حكم على المحتوى.",
            )

    components = extract_components(post_text)
    if not components:
        return None

    for component in components:
        _resolve_component(component, chunks)

    assessments = assess_with_model(post_text, components, client, debug=debug)
    _merge_model_assessment(components, assessments)
    _apply_materiality_heuristic(components)

    status, status_label, status_explanation, danger = _overall_status(components)

    primary = next(
        (
            component for component in components
            if component.get("source_identity", {}).get("status") in {
                "verified", "partial", "approximate",
            }
        ),
        components[0],
    )
    legacy = _legacy_from_component(primary)
    legacy["input_text"] = text
    legacy["input_explanation"] = explanation or ""
    legacy["verification_status"] = status
    legacy["status_label"] = status_label
    legacy["status_explanation"] = status_explanation
    legacy["danger_level"] = danger
    legacy["danger_level_applicable"] = danger != "غير محسوبة"
    legacy["components"] = components
    legacy["overall_status"] = status
    legacy["component_count"] = len(components)
    legacy["analysis_mode"] = "post"
    legacy["retrieval_ok"] = True
    legacy["verification_error"] = ""
    # Aggregate evidence for the top-level panel.
    evidence = []
    seen = set()
    for component in components:
        for item in component.get("evidence", [])[:2]:
            key = (item.get("reference"), item.get("text", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    legacy["evidence"] = evidence[:8]
    return legacy


def looks_like_post(text: str) -> bool:
    """Return True when the input has more than one claim/quote component."""
    if not text:
        return False
    quote_count = sum(len(pattern.findall(text)) for pattern in _QUOTE_PATTERNS)
    attribution = _contains(text, _ATTRIBUTION_HADITH + _ATTRIBUTION_QURAN)
    sentences = [
        sentence for sentence in re.split(r"(?<=[.!؟])\s+|\n+", text)
        if sentence.strip()
    ]
    if quote_count >= 2:
        return True
    if attribution and len(sentences) >= 2:
        return True
    if attribution and _contains(text, _ARGUMENT_MARKERS):
        return True
    if quote_count >= 1 and _contains(text, _ARGUMENT_MARKERS) and len(text) > 80:
        return True
    if "\n" in text and len(sentences) >= 2 and len(text) > 120:
        return True
    if len(sentences) >= 3 and _contains(text, _ARGUMENT_MARKERS):
        return True
    if _SEERAH_RE.search(_normalize(text)):
        return True
    if _is_unidentified_hadith_claim(text):
        return True
    return False
