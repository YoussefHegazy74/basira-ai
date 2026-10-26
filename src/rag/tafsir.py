"""
Basira RAG — Tafsir Retrieval
================================
Resolves a Quranic verse against the canonical Quran corpus, retrieves the
matching Tafsir Ibn Kathir chunk, and then summarises the result using
GPT-4o-mini.  Verse resolution is deterministic and lexical; embeddings are
not treated as proof that a verse was found.
"""

from __future__ import annotations

import difflib
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import faiss

from src.embeddings.embedder import get_client
from src.nlp.preprocessor import normalize_verse_text
from src.rag.pipeline import ground_text

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

# Minimum cosine-similarity score for a Quran chunk match to be trusted.
# Real ayahs score ≥ 0.72; fabricated text scores ~0.44.
_QURAN_SCORE_THRESHOLD = 0.55

# Canonical refusal returned when no tafsir chunk is found and when
# post-generation grounding rejects the generated prose (RAG-02).  Matches
# the long-standing no-match response byte for byte.
_TAFSIR_REFUSAL = "لم أجد تفسير هذه الآية في قاعدة البيانات."
_TAFSIR_AMBIGUOUS = (
    "النص يطابق أكثر من آية. يرجى تحديد السورة ورقم الآية."
)
_TAFSIR_INVALID_REFERENCE = (
    "المرجع المطلوب غير موجود. تحقق من رقم السورة والآية."
)
_TAFSIR_MISSING = (
    "تم العثور على الآية في القرآن الكريم، لكن لا يتوفر لها تفسير "
    "في قاعدة البيانات الحالية."
)
_MAX_AMBIGUITY_CANDIDATES = 8

_REFERENCE_ALIASES = {
    "ايهالكرسي": (2, 255),
    "ايةالكرسي": (2, 255),
}

# A tafsir request is often phrased as a sentence rather than a bare verse.
# These patterns strip the request framing from normalized text.  They are
# deliberately conservative: standalone verse words such as "من" or "الله"
# are never removed unless they are part of an unambiguous framing phrase.
_REQUEST_PREFIX_PATTERNS = (
    re.compile(r"^(?:من\s+فضلك|لو\s+سمحت)\s+"),
    re.compile(
        r"^(?:ارجو|أرجو|اريد|أريد|ابحث\s+عن|أبحث\s+عن|"
        r"ابحث\s+لي\s+عن|أبحث\s+لي\s+عن|بحث\s+عن)\s+"
    ),
    re.compile(r"^(?:ما\s+هو|ما\s+هي|ماذا|ما|هل)\s+"),
    re.compile(
        r"^(?:تفسير|فسر|اشرح|وضح|بين|معني|شرح)\s+"
        r"(?:لي|لنا)?\s*"
    ),
    re.compile(r"^(?:الايه|ايه|الايات|ايات)\s+"),
    re.compile(
        r"^(?:قوله|قول|كلام)\s+(?:الله\s+)?"
        r"(?:تعالي|عز\s+وجل|جل\s+جلاله)?\s*"
    ),
    re.compile(
        r"^(?:قال|يقول)\s+(?:الله\s+)?"
        r"(?:تعالي|عز\s+وجل|جل\s+جلاله)\s*"
    ),
    re.compile(r"^(?:تعالي|عز\s+وجل|جل\s+جلاله)\s+"),
    re.compile(
        r"^(?:التي|الذي|اللذين|اللتين)\s+"
        r"(?:نزلت|نزل|انزلت|أنزلت|وردت|جاءت)\s+(?:في|عن)\s+"
    ),
    re.compile(r"^(?:ال)?(?:قران\s+الكريم|قران)\s+"),
)

_QUOTE_PATTERNS = (
    re.compile(r"﴿([^﴾]+)﴾"),
    re.compile(r"«([^»]+)»"),
    re.compile(r"“([^”]+)”"),
    re.compile(r'"([^"]+)"'),
    re.compile(r"\(([^)]+)\)"),
)

# Arabic-Indic digits are already translated by normalize_verse_text; this
# translation is only for references parsed from the raw user input.
_REFERENCE_DIGITS_TRANS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_COLON_REFERENCE_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*[:：]\s*(\d{1,3})(?!\d)"
)

_ARABIC_ORDINALS = {
    "الاول": 1,
    "الاولي": 1,
    "الثاني": 2,
    "الثانيه": 2,
    "الثالث": 3,
    "الثالثه": 3,
    "الرابع": 4,
    "الرابعه": 4,
    "الخامس": 5,
    "الخامسه": 5,
    "السادس": 6,
    "السادسه": 6,
    "السابع": 7,
    "السابعه": 7,
    "الثامن": 8,
    "الثامنه": 8,
    "التاسع": 9,
    "التاسعه": 9,
    "العاشر": 10,
    "العاشره": 10,
}

_AYAH_NUMBER_RE = re.compile(
    r"(?:الايه|ايه|الايات|ايات)\s+(?:رقم\s+)?(\d{1,3})(?!\d)"
)
_AYAH_ORDINAL_RE = re.compile(
    r"(?:الايه|ايه|الايات|ايات)\s+(?:رقم\s+)?"
    r"(الاولي|الاول|الثاني|الثانيه|الثالث|الثالثه|"
    r"الرابع|الرابعه|الخامس|الخامسه|السادس|السادسه|"
    r"السابع|السابعه|الثامن|الثامنه|التاسع|التاسعه|"
    r"العاشر|العاشره)(?!\w)"
)
_SURAH_MARKER_RE = re.compile(r"(?:^|\s)(?:سوره|سورة)(?:\s+|$)")
_MAX_SURAH_NAME_TOKENS = 5

# ---------------------------------------------------------------------------
# Module-level caches (built per chunk list)
# ---------------------------------------------------------------------------
# Key:   id(chunks)
# Value: (chunks, data).  Keeping a strong reference to the list prevents
#        CPython from reusing its id for a different list after garbage
#        collection, which would otherwise return another corpus's map.
_tafsir_map_cache: dict[int, tuple[list[dict], dict[str, dict]]] = {}
_verse_map_cache: dict[int, tuple[list[dict], dict]] = {}
_tafsir_scope_cache: dict[
    int, tuple[list[dict], dict[int, tuple[int, int, int]]]
] = {}
_MAX_CACHE_ENTRIES = 4


def _cache_lookup(cache: dict, chunks: list[dict]):
    entry = cache.get(id(chunks))
    if entry is not None and entry[0] is chunks:
        return entry[1]
    return None


def _cache_store(cache: dict, chunks: list[dict], value) -> None:
    cache[id(chunks)] = (chunks, value)
    while len(cache) > _MAX_CACHE_ENTRIES:
        cache.pop(next(iter(cache)))



class TafsirResolution(tuple):
    """Tuple-compatible resolution result with match metadata."""

    def __new__(
        cls,
        primary: Optional[dict],
        next_chunk: Optional[dict],
        source: str,
        *,
        failure_message: str = "",
        candidates=(),
        quran_chunk: Optional[dict] = None,
        match_kind: str = "",
        source_scope: tuple[int, int, int] | None = None,
    ):
        result = super().__new__(cls, (primary, next_chunk, source))
        result.failure_message = failure_message
        result.candidates = tuple(candidates)
        result.quran_chunk = quran_chunk
        result.match_kind = match_kind
        result.source_scope = source_scope
        return result


def _get_tafsir_map(chunks: list[dict]) -> dict[str, dict]:
    """Return (and cache) a {reference -> chunk} dict for tafsir chunks."""
    cached = _cache_lookup(_tafsir_map_cache, chunks)
    if cached is not None:
        return cached
    tmap: dict[str, dict] = {}
    for c in chunks:
        if c.get("source") == "tafsir":
            ref = c.get("reference", "")
            if ref:
                tmap[ref] = c
    _cache_store(_tafsir_map_cache, chunks, tmap)
    return tmap


def _get_tafsir_scope_map(
    chunks: list[dict],
) -> dict[int, tuple[int, int, int]]:
    """Map chunk id to its duplicated-commentary verse range.

    The Tafsir dataset repeats one whole passage for every verse it covers.
    These records therefore carry a verse-specific ``reference`` even though
    the commentary itself is not verse-specific.  For byte-identical groups
    of two or more *contiguous* verses in the same surah, return
    ``(surah, first_ayah, last_ayah)`` keyed by ``id(chunk)``.  Unique
    records are intentionally absent.

    This does not alter the corpus or claim where a passage truly begins; it
    only records which verses the source file assigns the same text to.
    """
    cached = _cache_lookup(_tafsir_scope_cache, chunks)
    if cached is not None:
        return cached

    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        if chunk.get("source") != "tafsir":
            continue
        text = " ".join((chunk.get("original_text") or "").split())
        if text:
            groups.setdefault(text, []).append(chunk)

    scopes: dict[int, tuple[int, int, int]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        refs: list[tuple[int, int, dict]] = []
        for chunk in group:
            metadata = chunk.get("metadata") or {}
            try:
                refs.append((
                    int(metadata["surah_number"]),
                    int(metadata["ayah_number"]),
                    chunk,
                ))
            except (KeyError, TypeError, ValueError):
                refs = []
                break
        if not refs:
            continue
        refs.sort(key=lambda item: (item[0], item[1]))
        surahs = {surah for surah, _ayah, _chunk in refs}
        ayahs = [ayah for _surah, ayah, _chunk in refs]
        if len(surahs) != 1:
            continue
        if ayahs != list(range(ayahs[0], ayahs[-1] + 1)):
            continue
        scope = (refs[0][0], refs[0][1], refs[-1][1])
        for _surah, _ayah, chunk in refs:
            scopes[id(chunk)] = scope

    _cache_store(_tafsir_scope_cache, chunks, scopes)
    return scopes



SYSTEM_PROMPT = (
    "أنت متخصص في تفسير القرآن الكريم. اعتمد حصراً على مقاطع تفسير ابن "
    "كثير المقدمة:\n"
    "1. اشرح الآية المطلوبة فقط، ولا تشرح آية أخرى ولا تدمج معاني آيات "
    "أخرى من السورة في حقل 'tafsir'.\n"
    "2. إذا كان المقطع يفسّر سورة كاملة، فاختر منه ما يتعلق بالآية "
    "المطلوبة تحديداً. لا تنسب معنى آية أخرى إلى الآية المطلوبة.\n"
    "3. ضع في حقل 'context' فقط السياق أو سبب النزول الوارد صراحة في "
    "المصدر. إذا كان السياق عاماً للسورة فبيّن أنه سياق عام، ولا تجعله "
    "بديلاً عن تفسير الآية أو مزيجاً معه.\n"
    "4. استخدم ألفاظ المصدر نفسها قدر الإمكان، ولا تستبدل كلمة واردة فيه "
    "بمرادف غير موجود في المقاطع.\n"
    "لا تجاوب من معرفتك الخاصة. إذا لم تجد الآية في المصادر، "
    "قل: 'لم أجد تفسير هذه الآية في قاعدة البيانات'.\n"
    "أجب بالعربية فقط في شكل JSON بهذين المفتاحين فقط:\n"
    "{\n"
    "  'tafsir': 'نص التفسير',\n"
    "  'context': 'السياق وسبب النزول أو فارغ'\n"
    "}"
)

SYSTEM_PROMPT_RETRY = (
    SYSTEM_PROMPT
    + "\nتنبيه: تم التحقق من وجود الآية في المصادر المرفقة. "
      "لا تستخدم عبارة 'لم أجد' إلا إذا كانت المقاطع لا تتضمن "
      "تفسير الآية إطلاقاً. لخّص ما ورد عن الآية المطلوبة فقط، "
      "ولا تدرج معاني آيات أخرى من السورة."
)


def load_faiss_index(
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a FAISS index and its associated chunks from disk.

    Args:
        index_path:  Path to the FAISS index file.
        chunks_path: Path to the pickled chunks file.

    Returns:
        A (faiss.IndexFlatIP, chunks) tuple.
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[tafsir] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[tafsir] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


def get_key(d: dict, *keys: str, default: str = ""):
    """Return the value of the first key found in *d*, else *default*."""
    for key in keys:
        if key in d:
            return d[key]
    return default


def _flatten_source(source_val) -> str:
    """Normalise a source value that may be a dict into a flat string.

    GPT sometimes returns the source as a nested object, e.g.
    ``{"رقم_السورة": "…", "رقم_الآية": "…"}``.
    """
    if isinstance(source_val, dict):
        surah = source_val.get("رقم_السورة", source_val.get("surah", ""))
        ayah = source_val.get("رقم_الآية", source_val.get("ayah", ""))
        if surah or ayah:
            return f"{surah}:{ayah}" if ayah else str(surah)
        # Fallback: book + number style
        book = source_val.get("اسم_الكتاب", source_val.get("book", ""))
        number = source_val.get("رقم_الحديث", source_val.get("number", ""))
        return f"{book}:{number}" if number else book
    return str(source_val) if source_val else ""


_PUNCTUATION_SPACE_RE = re.compile(r"\s+([،؛:.!?؟])")


def _clean_generated_text(text: str) -> str:
    """Remove sentence-split whitespace artifacts before punctuation."""
    return _PUNCTUATION_SPACE_RE.sub(r"\1", text or "").strip()


def _filter_tafsir(results: list[dict]) -> list[dict]:
    """Keep only chunks whose source is 'tafsir'."""
    return [r for r in results if r.get("source") == "tafsir"]


_ORPHAN_TOKENS = {"ا", "ي"}


def _merge_orphan_tokens(tokens: list[str]) -> list[str]:
    """Rejoin Uthmani orthographic orphan alef/yeh tokens to the prior word."""
    merged: list[str] = []
    for token in tokens:
        if token in _ORPHAN_TOKENS and merged:
            merged[-1] += token
        else:
            merged.append(token)
    return merged


def _token_variants(
    text: str,
    *,
    strip_leading_conjunction: bool = False,
) -> set[tuple[str, ...]]:
    """Return normalized token variants used for verse matching.

    ``strip_leading_conjunction`` is for *query* text only.  A quotation
    often begins with a conjunction (``ولا تقربوا...``) while the canonical
    verse begins with the bare word (``لا تقربوا...``).  Applying the same
    fold to canonical verses would make ``الحمد لله رب العالمين`` collide
    with ``والحمد لله رب العالمين`` (1:2 and 37:182), so verse variants keep
    their original leading word.
    """
    variants: set[tuple[str, ...]] = set()
    for expand in (True, False):
        for alef_maqsura in ("keep", "alef"):
            normalized = normalize_verse_text(
                text,
                expand_dagger_alef=expand,
                alef_maqsura_mode=alef_maqsura,
            )
            tokens = tuple(_merge_orphan_tokens(normalized.split()))
            if not tokens:
                continue
            variants.add(tokens)
            if len(tokens) > 1 and tokens[0] == "يا":
                variants.add((tokens[0] + tokens[1],) + tokens[2:])
            if (
                strip_leading_conjunction
                and len(tokens[0]) > 2
                and tokens[0][0] in ("و", "ف")
            ):
                variants.add((tokens[0][1:],) + tokens[1:])
    return variants


def _lookup_key(value: str) -> str:
    return normalize_verse_text(value).lower().replace(" ", "")


def _english_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _name_aliases(value: str, *, english: bool = False) -> set[str]:
    aliases: set[str] = set()
    if english:
        key = _english_lookup_key(value)
        if key:
            aliases.add(key)
            if key.startswith("al") and len(key) > 4:
                aliases.add(key[2:])
        return aliases

    key = _lookup_key(value)
    if key:
        aliases.add(key)
        if key.startswith("ال") and len(key) > 2:
            aliases.add(key[2:])
    return aliases


def _get_verse_map(chunks: list[dict]) -> dict:
    """Build (and cache) normalized Quran lookup structures for *chunks*."""
    cached = _cache_lookup(_verse_map_cache, chunks)
    if cached is not None:
        return cached

    by_ref: dict[tuple[int, int], dict] = {}
    by_exact: dict[str, list[dict]] = {}
    entries: list[tuple[dict, set[tuple[str, ...]]]] = []
    alias_to_surahs: dict[str, set[int]] = {}
    surah_totals: dict[int, int] = {}
    surah_names: dict[int, str] = {}
    trigrams: dict[str, list[tuple[dict, tuple[str, ...], int]]] = {}

    for chunk in chunks:
        if chunk.get("source") != "quran":
            continue
        metadata = chunk.get("metadata") or {}
        try:
            surah_num = int(metadata["surah_number"])
            ayah_num = int(metadata["ayah_number"])
        except (KeyError, TypeError, ValueError):
            continue

        by_ref[(surah_num, ayah_num)] = chunk
        surah_name_ar = metadata.get("surah_name")
        if surah_name_ar and surah_num not in surah_names:
            surah_names[surah_num] = str(surah_name_ar)
        total = metadata.get("total_ayahs_in_surah")
        if isinstance(total, int):
            surah_totals[surah_num] = max(
                surah_totals.get(surah_num, 0), total,
            )
        else:
            surah_totals[surah_num] = max(
                surah_totals.get(surah_num, 0), ayah_num,
            )

        for name, is_english in (
            (metadata.get("surah_name"), False),
            (metadata.get("surah_name_en"), True),
        ):
            if not name:
                continue
            for alias in _name_aliases(str(name), english=is_english):
                alias_to_surahs.setdefault(alias, set()).add(surah_num)

        variants = _token_variants(chunk.get("original_text", "") or "")
        entries.append((chunk, variants))
        for variant in variants:
            by_exact.setdefault(" ".join(variant), []).append(chunk)
            for position in range(len(variant) - 2):
                key = " ".join(variant[position:position + 3])
                trigrams.setdefault(key, []).append(
                    (chunk, variant, position)
                )

    for key, candidates in by_exact.items():
        seen: set[int] = set()
        unique: list[dict] = []
        for candidate in candidates:
            marker = id(candidate)
            if marker not in seen:
                seen.add(marker)
                unique.append(candidate)
        by_exact[key] = unique

    names = {
        key: next(iter(surahs))
        for key, surahs in alias_to_surahs.items()
        if len(surahs) == 1
    }

    verse_map = {
        "by_ref": by_ref,
        "by_exact": by_exact,
        "entries": entries,
        "names": names,
        "surah_names": surah_names,
        "surah_totals": surah_totals,
        "trigrams": trigrams,
    }
    _cache_store(_verse_map_cache, chunks, verse_map)
    return verse_map


def _lookup_surah(name: str, verse_map: dict) -> Optional[int]:
    cleaned = re.sub(r"^(?:سوره|سورة)\s+", "", name).strip()
    for key in (_lookup_key(cleaned), _english_lookup_key(cleaned)):
        if key and key in verse_map["names"]:
            return verse_map["names"][key]
    return None


def _lookup_surah_prefix(text: str, verse_map: dict) -> Optional[int]:
    """Resolve the longest surah-name prefix of *text*."""
    tokens = text.split()
    for count in range(
        min(len(tokens), _MAX_SURAH_NAME_TOKENS), 0, -1,
    ):
        surah = _lookup_surah(" ".join(tokens[:count]), verse_map)
        if surah is not None:
            return surah
    return None


def _lookup_surah_suffix(text: str, verse_map: dict) -> Optional[int]:
    """Resolve the longest surah-name suffix of *text*."""
    tokens = text.split()
    for count in range(
        min(len(tokens), _MAX_SURAH_NAME_TOKENS), 0, -1,
    ):
        surah = _lookup_surah(" ".join(tokens[-count:]), verse_map)
        if surah is not None:
            return surah
    return None


def _strip_request_framing(text: str) -> str:
    """Remove natural-language framing from a tafsir request.

    ``"ما تفسير قوله تعالى: قل هو الله أحد"`` becomes
    ``"قل هو الله احد"``.  The original text is always matched first, so
    stripping can never change the result for a bare canonical verse.
    """
    normalized = normalize_verse_text(text)
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        for pattern in _REQUEST_PREFIX_PATTERNS:
            stripped = pattern.sub("", normalized, count=1).strip()
            if stripped and stripped != normalized:
                normalized = stripped
                break
    return normalized


def _extract_quoted_texts(text: str) -> list[str]:
    """Return normalized Quran-sized fragments enclosed in quotes."""
    quoted: list[str] = []
    seen: set[str] = set()
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(text or ""):
            normalized = normalize_verse_text(match.group(1))
            if len(normalized.split()) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            quoted.append(normalized)
    quoted.sort(key=lambda value: len(value.split()), reverse=True)
    return quoted


def _surah_mentions(
    normalized: str, verse_map: dict,
) -> list[tuple[int, int]]:
    """Return ``(surah_number, end_token_index)`` for explicit surah names."""
    tokens = normalized.split()
    mentions: list[tuple[int, int]] = []
    seen_surahs: set[int] = set()
    for index, token in enumerate(tokens):
        if token not in {"سوره", "سورة", "surah", "surat"}:
            continue
        if index + 1 >= len(tokens):
            continue
        for count in range(
            min(_MAX_SURAH_NAME_TOKENS, len(tokens) - index - 1), 0, -1,
        ):
            name = " ".join(tokens[index + 1:index + 1 + count])
            surah = _lookup_surah(name, verse_map)
            if surah is None:
                continue
            if surah not in seen_surahs:
                seen_surahs.add(surah)
                mentions.append((surah, index + 1 + count))
            break
    return mentions


def _mentioned_ayah_numbers(normalized: str) -> list[int]:
    """Return distinct explicit ayah numbers/ordinals mentioned in *text*."""
    numbers: list[int] = []
    for match in _AYAH_NUMBER_RE.finditer(normalized):
        numbers.append(int(match.group(1)))
    for match in _AYAH_ORDINAL_RE.finditer(normalized):
        ordinal = _ARABIC_ORDINALS.get(match.group(1))
        if ordinal is not None:
            numbers.append(ordinal)
    unique: list[int] = []
    for number in numbers:
        if number not in unique:
            unique.append(number)
    return unique


def _surah_scope_message(surah: int, verse_map: dict) -> str:
    total = verse_map["surah_totals"].get(surah, 0)
    name = verse_map.get("surah_names", {}).get(surah, "")
    label = f"سورة {name}" if name else f"السورة رقم {surah}"
    return (
        f"تم التعرف على {label} ({surah}). يرجى تحديد رقم الآية المطلوبة "
        f"(من 1 إلى {total}) للحصول على تفسيرها."
    )


def _needs_surah_message() -> str:
    return (
        "يرجى تحديد اسم السورة مع رقم الآية، مثل: سورة البقرة آية 255."
    )


def _multiple_ayahs_message(surah: int, verse_map: dict) -> str:
    name = verse_map.get("surah_names", {}).get(surah, "")
    label = f"سورة {name}" if name else f"السورة رقم {surah}"
    return (
        f"تم التعرف على {label}، لكن الطلب يتضمن أكثر من رقم آية. "
        "يرجى تحديد آية واحدة."
    )


def _validate_reference(
    surah: int, ayah: int, verse_map: dict,
) -> tuple[Optional[tuple[int, int]], str]:
    total = verse_map["surah_totals"].get(surah)
    if total is None or not (1 <= ayah <= total):
        return None, _TAFSIR_INVALID_REFERENCE
    if (surah, ayah) not in verse_map["by_ref"]:
        return None, _TAFSIR_INVALID_REFERENCE
    return (surah, ayah), ""


def _match_full_reference(
    normalized: str, verse_map: dict,
) -> tuple[Optional[tuple[int, int]], str]:
    """Parse a reference that makes up the entire normalized string."""
    patterns = (
        (r"^(\d{1,3})\s+(\d{1,3})$", "numeric"),
        (
            r"^(?:سوره|سورة)\s+(\d{1,3})"
            r"(?:\s+(?:الايه|ايه|رقم))?\s+(\d{1,3})$",
            "numeric",
        ),
        (
            r"^(?:الايه|ايه)\s+(\d{1,3})\s+(?:من|في)\s+"
            r"(?:سوره|سورة)\s+(.+)$",
            "ayah_name",
        ),
        (
            r"^(\d{1,3})\s+(?:من|في)\s+(?:سوره|سورة)\s+(.+)$",
            "ayah_name",
        ),
        (
            r"^(?:سوره|سورة)\s+(.+?)\s+(?:الايه|ايه)\s+"
            r"(\d{1,3})$",
            "name_ayah",
        ),
        (r"^(.+?)\s+(?:الايه|ايه)\s+(\d{1,3})$", "name_ayah"),
        (r"^(?:سوره|سورة)\s+(.+?)\s+(\d{1,3})$", "name_ayah"),
        (r"^(.+?)\s+(\d{1,3})$", "name_ayah"),
    )

    for pattern, kind in patterns:
        match = re.match(pattern, normalized)
        if not match:
            continue
        if kind == "numeric":
            surah, ayah = int(match.group(1)), int(match.group(2))
            return _validate_reference(surah, ayah, verse_map)
        if kind == "ayah_name":
            ayah = int(match.group(1))
            surah = _lookup_surah(match.group(2), verse_map)
        else:
            surah = _lookup_surah_prefix(match.group(1), verse_map)
            ayah = int(match.group(2))
        if surah is None:
            continue
        return _validate_reference(surah, ayah, verse_map)

    return None, ""


def _resolve_explicit_reference(
    text: str, verse_map: dict,
) -> tuple[Optional[tuple[int, int]], str]:
    """Parse a surah/ayah reference anywhere in a natural-language request."""
    normalized = normalize_verse_text(text)
    if not normalized:
        return None, ""

    compact = normalized.replace(" ", "")
    for alias, reference in _REFERENCE_ALIASES.items():
        if alias in compact:
            return _validate_reference(*reference, verse_map)

    raw = (text or "").translate(_REFERENCE_DIGITS_TRANS)
    colon_match = _COLON_REFERENCE_RE.search(raw)
    if colon_match:
        return _validate_reference(
            int(colon_match.group(1)),
            int(colon_match.group(2)),
            verse_map,
        )

    for candidate in (normalized, _strip_request_framing(text)):
        reference, failure = _match_full_reference(candidate, verse_map)
        if failure:
            return None, failure
        if reference is not None:
            return reference, ""

    mentions = _surah_mentions(normalized, verse_map)
    ayah_numbers = _mentioned_ayah_numbers(normalized)

    # A number immediately after the surah name ("سورة البقرة 255") is an
    # ayah number even though the word "آية" is absent.
    tokens = normalized.split()
    for _surah, end_index in mentions:
        if end_index < len(tokens) and tokens[end_index].isdigit():
            number = int(tokens[end_index])
            if number not in ayah_numbers:
                ayah_numbers.append(number)

    if len(mentions) == 1 and len(ayah_numbers) == 1:
        return _validate_reference(
            mentions[0][0], ayah_numbers[0], verse_map,
        )

    return None, ""


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[int] = set()
    unique: list[dict] = []
    for chunk in chunks:
        marker = id(chunk)
        if marker not in seen:
            seen.add(marker)
            unique.append(chunk)
    return unique


def _sort_chunks(chunks: list[dict]) -> list[dict]:
    def key(chunk: dict) -> tuple[int, int]:
        metadata = chunk.get("metadata") or {}
        try:
            return (
                int(metadata.get("surah_number", 0)),
                int(metadata.get("ayah_number", 0)),
            )
        except (TypeError, ValueError):
            return (0, 0)
    return sorted(chunks, key=key)


def _ambiguity_message(candidates: list[dict]) -> str:
    refs: list[str] = []
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        surah = metadata.get("surah_number")
        ayah = metadata.get("ayah_number")
        if surah is not None and ayah is not None:
            refs.append(f"{surah}:{ayah}")
    shown = refs[:_MAX_AMBIGUITY_CANDIDATES]
    remaining = len(refs) - len(shown)
    suffix = f" وغيرها ({remaining} آية)" if remaining > 0 else ""
    if not shown:
        return _TAFSIR_AMBIGUOUS
    return f"{_TAFSIR_AMBIGUOUS} المرشحات: {'، '.join(shown)}{suffix}."


def _exact_verse_match(text: str, verse_map: dict) -> list[dict]:
    exact: list[dict] = []
    for variant in _token_variants(text):
        exact.extend(verse_map["by_exact"].get(" ".join(variant), []))
    if exact:
        return _sort_chunks(_dedupe_chunks(exact))

    # Only when the original wording matches no complete verse do we try the
    # conjunction-stripped quotation form.  This lets "ولا تقربوا الصلاة"
    # reach 4:43 without making "والحمد لله رب العالمين" (37:182) collide
    # with the bare "الحمد لله رب العالمين" (1:2).
    for variant in _token_variants(text, strip_leading_conjunction=True):
        exact.extend(verse_map["by_exact"].get(" ".join(variant), []))
    return _sort_chunks(_dedupe_chunks(exact))


def _short_partial_verse_match(text: str, verse_map: dict) -> list[dict]:
    """Match a one- or two-word fragment against the verse corpus."""
    variants = _token_variants(text, strip_leading_conjunction=True)
    if not variants:
        return []
    if any(len(variant) > 2 for variant in variants):
        return []
    wrapped = {f" {' '.join(variant)} " for variant in variants}
    partial: list[dict] = []
    for chunk, chunk_variants in verse_map["entries"]:
        for chunk_variant in chunk_variants:
            wrapped_chunk = f" {' '.join(chunk_variant)} "
            if any(query in wrapped_chunk for query in wrapped):
                partial.append(chunk)
                break
    return _sort_chunks(_dedupe_chunks(partial))


def _longest_ngram_match(
    texts: list[str], verse_map: dict,
) -> tuple[list[dict], str]:
    """Match the longest contiguous phrase shared with a canonical verse.

    This makes a natural-language request such as ``"ما تفسير آية الذين
    يأكلون أموال اليتامى"`` resolve to 4:10 without treating the request
    words as Quranic text.  Ties are reported as ambiguity; the resolver
    never guesses between verses.
    """
    best_length = 0
    best_chunks: list[dict] = []
    best_phrase = ""
    seen_chunks: set[int] = set()

    for text in texts:
        for tokens in _token_variants(
            text, strip_leading_conjunction=True,
        ):
            for start in range(len(tokens) - 2):
                trigram = tokens[start:start + 3]
                for chunk, verse_tokens, position in verse_map[
                    "trigrams"
                ].get(" ".join(trigram), ()):
                    if verse_tokens[position:position + 3] != trigram:
                        continue
                    limit = min(
                        len(tokens) - start,
                        len(verse_tokens) - position,
                    )
                    length = 3
                    while (
                        length < limit
                        and tokens[start + length]
                        == verse_tokens[position + length]
                    ):
                        length += 1
                    if length < best_length:
                        continue
                    marker = id(chunk)
                    if length > best_length:
                        best_length = length
                        best_chunks = [chunk]
                        seen_chunks = {marker}
                        best_phrase = " ".join(tokens[start:start + length])
                    elif marker not in seen_chunks:
                        seen_chunks.add(marker)
                        best_chunks.append(chunk)

    if best_length < 3:
        return [], ""
    return _sort_chunks(_dedupe_chunks(best_chunks)), best_phrase


def _match_verse_text(
    text: str,
    verse_map: dict,
    *,
    quoted_texts: tuple[str, ...] = (),
) -> tuple[list[dict], str]:
    """Match a user's request against canonical Quran text.

    The original normalized text is tried first, so stripping request
    framing can never turn an ambiguous canonical verse into a false unique
    match.  Quoted fragments and a framing-stripped candidate are tried
    afterwards, followed by longest-contiguous-phrase matching.
    """
    normalized = normalize_verse_text(text)
    stripped = _strip_request_framing(text)

    candidates: list[str] = []
    for candidate in (normalized, *quoted_texts, stripped):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        exact = _exact_verse_match(candidate, verse_map)
        if exact:
            return exact, "exact"

    for candidate in candidates:
        partial = _short_partial_verse_match(candidate, verse_map)
        if partial:
            return partial, "partial"

    partial, _phrase = _longest_ngram_match(candidates, verse_map)
    if partial:
        return partial, "partial"
    return [], ""


def _best_contiguous_match(
    query_variants: set[tuple[str, ...]],
    verse_variants: set[tuple[str, ...]],
) -> tuple[int, int, int, bool]:
    """Return (matched_len, query_len, verse_start, covers_whole_verse).

    Only a contiguous sequence of query tokens counts.  A quotation may be a
    fragment of a verse, but re-ordered or scattered words are not treated as
    a verbatim quotation.
    """
    best_len = 0
    best_query_len = 0
    best_start = 0
    covers_whole = False

    for query in query_variants:
        if not query:
            continue
        for verse in verse_variants:
            if len(query) > len(verse):
                continue
            for start in range(len(verse) - len(query) + 1):
                if verse[start:start + len(query)] != query:
                    continue
                whole = len(query) == len(verse)
                if len(query) > best_len or (whole and not covers_whole):
                    best_len = len(query)
                    best_query_len = len(query)
                    best_start = start
                    covers_whole = whole
                break

    return best_len, best_query_len, best_start, covers_whole


def _fuzzy_verse_match(
    text: str,
    verse_map: dict,
) -> list[tuple[dict, float]]:
    """Find a tightly bounded approximate verse match.

    This is only a fallback after exact and verbatim-partial matching have
    failed.  It requires at least four query tokens and a token-sequence
    similarity of 0.90, so a single ordinary word can never be attributed to
    a verse and a re-ordered phrase is not mistaken for a quotation.
    """
    query_variants = {
        variant
        for variant in _token_variants(
            text, strip_leading_conjunction=True,
        )
        if len(variant) >= 4
    }
    if not query_variants:
        return []

    matches: list[tuple[dict, float]] = []
    for chunk, verse_variants in verse_map["entries"]:
        best_ratio = 0.0
        for query in query_variants:
            for verse in verse_variants:
                if abs(len(verse) - len(query)) > max(2, len(query) // 5):
                    continue
                ratio = difflib.SequenceMatcher(
                    None, query, verse, autojunk=False,
                ).ratio()
                best_ratio = max(best_ratio, ratio)
        if best_ratio >= 0.90:
            matches.append((chunk, best_ratio))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def match_quran_quotation(
    text: str,
    chunks: list[dict],
    *,
    debug: bool = False,
) -> dict:
    """Resolve *text* against the canonical Quran corpus.

    The result is deterministic and lexical.  Semantic similarity is never
    consulted here, so a result of ``found=True`` always points to an actual
    canonical verse record.  ``is_complete`` is true only when the submitted
    text covers the whole verse; a verbatim fragment is reported as a partial
    quotation with its exact reference.
    """
    verse_map = _get_verse_map(chunks)
    normalized = normalize_verse_text(text)
    quoted = tuple(_extract_quoted_texts(text))
    candidates, match_kind = _match_verse_text(
        text, verse_map, quoted_texts=quoted,
    )

    base = {
        "found": False,
        "ambiguous": False,
        "match_type": "none",
        "is_partial": False,
        "is_complete": False,
        "coverage": 0.0,
        "matched_tokens": 0,
        "query_tokens": len(normalized.split()),
        "canonical_text": "",
        "reference": "",
        "source": "",
        "surah_name": "",
        "surah_number": None,
        "ayah_number": None,
        "chunk": None,
        "candidates": [],
    }

    if len(candidates) > 1:
        return {
            **base,
            "ambiguous": True,
            "match_type": "ambiguous",
            "candidates": [
                {
                    "reference": candidate.get("reference", ""),
                    "source": candidate.get("source", ""),
                    "text": candidate.get("original_text", ""),
                }
                for candidate in candidates[:_MAX_AMBIGUITY_CANDIDATES]
            ],
        }

    if len(candidates) == 1:
        chunk = candidates[0]
        query_variants: set[tuple[str, ...]] = set()
        for candidate_text in (
            normalized,
            *quoted,
            _strip_request_framing(text),
        ):
            if candidate_text:
                query_variants.update(_token_variants(
                    candidate_text, strip_leading_conjunction=True,
                ))
        verse_variants = _token_variants(
            chunk.get("original_text", "") or "",
        )
        matched, query_len, _start, complete = _best_contiguous_match(
            query_variants, verse_variants,
        )
        if matched <= 0:
            return base

        metadata = chunk.get("metadata") or {}
        return {
            **base,
            "found": True,
            "match_type": "exact" if complete else "partial",
            "is_partial": not complete,
            "is_complete": complete,
            "coverage": (
                round(query_len / matched, 4) if matched else 0.0
            ),
            "matched_tokens": matched,
            "query_tokens": query_len,
            "canonical_text": chunk.get("original_text", ""),
            "reference": chunk.get("reference", ""),
            "source": chunk.get("source", "quran"),
            "surah_name": metadata.get("surah_name", ""),
            "surah_number": metadata.get("surah_number"),
            "ayah_number": metadata.get("ayah_number"),
            "chunk": chunk,
        }

    fuzzy = _fuzzy_verse_match(text, verse_map)
    if fuzzy:
        best_ratio = fuzzy[0][1]
        top = [item for item in fuzzy if item[1] == best_ratio]
        if len(top) > 1:
            return {
                **base,
                "ambiguous": True,
                "match_type": "ambiguous",
                "candidates": [
                    {
                        "reference": chunk.get("reference", ""),
                        "source": chunk.get("source", ""),
                        "text": chunk.get("original_text", ""),
                    }
                    for chunk, _ratio in top[:_MAX_AMBIGUITY_CANDIDATES]
                ],
            }
        chunk, ratio = top[0]
        metadata = chunk.get("metadata") or {}
        return {
            **base,
            "found": True,
            "match_type": "approximate",
            "is_partial": True,
            "is_complete": False,
            "coverage": round(ratio, 4),
            "matched_tokens": len(normalized.split()),
            "query_tokens": len(normalized.split()),
            "canonical_text": chunk.get("original_text", ""),
            "reference": chunk.get("reference", ""),
            "source": chunk.get("source", "quran"),
            "surah_name": metadata.get("surah_name", ""),
            "surah_number": metadata.get("surah_number"),
            "ayah_number": metadata.get("ayah_number"),
            "chunk": chunk,
        }

    if debug:
        print(f"[DEBUG] No canonical Quran quotation for {text!r}")
    return base


def _build_resolution(
    quran_chunk: dict,
    match_kind: str,
    chunks: list[dict],
    *,
    debug: bool = False,
) -> TafsirResolution:
    metadata = quran_chunk.get("metadata") or {}
    try:
        surah_num = int(metadata["surah_number"])
        ayah_num = int(metadata["ayah_number"])
    except (KeyError, TypeError, ValueError):
        return TafsirResolution(None, None, "", quran_chunk=quran_chunk)

    tafsir_map = _get_tafsir_map(chunks)
    primary = tafsir_map.get(f"تفسير ابن كثير {surah_num}:{ayah_num}")
    if primary is None:
        return TafsirResolution(
            None,
            None,
            f"{surah_num}:{ayah_num}",
            failure_message=_TAFSIR_MISSING,
            quran_chunk=quran_chunk,
            match_kind=match_kind,
        )

    primary_metadata = primary.get("metadata") or {}
    if (
        primary_metadata.get("surah_number") not in (None, surah_num)
        or primary_metadata.get("ayah_number") not in (None, ayah_num)
    ):
        return TafsirResolution(
            None,
            None,
            f"{surah_num}:{ayah_num}",
            failure_message=_TAFSIR_MISSING,
            quran_chunk=quran_chunk,
            match_kind=match_kind,
        )

    source_str = f"{surah_num}:{ayah_num}"
    total = metadata.get("total_ayahs_in_surah")
    next_chunk = None
    if not isinstance(total, int) or ayah_num < total:
        next_chunk = tafsir_map.get(
            f"تفسير ابن كثير {surah_num}:{ayah_num + 1}"
        )

    if debug:
        print(
            f"[DEBUG] Resolved {source_str} via {match_kind}; "
            f"tafsir={primary.get('reference')!r}"
        )

    return TafsirResolution(
        primary,
        next_chunk,
        source_str,
        quran_chunk=quran_chunk,
        match_kind=match_kind,
        source_scope=_get_tafsir_scope_map(chunks).get(id(primary)),
    )


# ---------------------------------------------------------------------------
# Public helper: resolve_tafsir_reference
# ---------------------------------------------------------------------------

def resolve_tafsir_reference(
    ayah_text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    debug: bool = False,
) -> TafsirResolution:
    """Identify the correct tafsir chunk for *ayah_text*.

    Resolution is lexical and deterministic: explicit references are parsed
    first, then the normalized verse text is matched exactly or as a
    contiguous phrase.  Ambiguous matches fail with candidate references
    instead of silently choosing one verse.  Embeddings are not used as proof
    that a verse was found.

    Returns a tuple-compatible ``(primary, next_chunk, source_str)`` result
    with extra attributes: ``failure_message``, ``candidates``,
    ``quran_chunk`` and ``match_kind``.
    """
    verse_map = _get_verse_map(chunks)

    reference, failure = _resolve_explicit_reference(ayah_text, verse_map)
    if failure:
        return TafsirResolution(None, None, "", failure_message=failure)
    if reference is not None:
        quran_chunk = verse_map["by_ref"].get(reference)
        if quran_chunk is None:
            return TafsirResolution(
                None, None, "", failure_message=_TAFSIR_INVALID_REFERENCE,
            )
        return _build_resolution(
            quran_chunk, "reference", chunks, debug=debug,
        )

    candidates, match_kind = _match_verse_text(
        ayah_text,
        verse_map,
        quoted_texts=tuple(_extract_quoted_texts(ayah_text)),
    )
    if len(candidates) > 1:
        return TafsirResolution(
            None,
            None,
            "",
            failure_message=_ambiguity_message(candidates),
            candidates=candidates,
            match_kind=match_kind,
        )
    if len(candidates) == 1:
        return _build_resolution(
            candidates[0], match_kind, chunks, debug=debug,
        )

    normalized = normalize_verse_text(ayah_text)
    mentions = _surah_mentions(normalized, verse_map)
    ayah_numbers = _mentioned_ayah_numbers(normalized)
    if len(mentions) == 1:
        surah = mentions[0][0]
        failure = (
            _multiple_ayahs_message(surah, verse_map)
            if ayah_numbers
            else _surah_scope_message(surah, verse_map)
        )
        return TafsirResolution(None, None, "", failure_message=failure)
    if ayah_numbers:
        return TafsirResolution(
            None, None, "", failure_message=_needs_surah_message(),
        )

    if debug:
        print(f"[DEBUG] No lexical Quran match for {ayah_text!r}")
    return TafsirResolution(None, None, "")


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def _generate_tafsir(
    client,
    system_prompt: str,
    user_message: str,
    context_chunks: list[dict],
    chunks: list[dict],
    *,
    debug: bool = False,
) -> tuple[str, str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_answer = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    parse_failed = False
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        answer = {
            "tafsir": "لم أجد تفسير هذه الآية في قاعدة البيانات",
            "context": "",
        }
        parse_failed = True

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    tafsir_text = get_key(
        answer,
        "tafsir", "التفسير", "نص_التفسير", "تفسير",
        default="لم أجد تفسير هذه الآية في قاعدة البيانات",
    )
    context_text = get_key(
        answer,
        "context", "السياق", "سبب_النزول", "السياق_وسبب_النزول",
        "السياق وسبب النزول",
    )

    if not isinstance(tafsir_text, str):
        tafsir_text = "" if tafsir_text is None else str(tafsir_text)
    if not isinstance(context_text, str):
        context_text = "" if context_text is None else str(context_text)
    if parse_failed:
        return _clean_generated_text(tafsir_text), _clean_generated_text(
            context_text
        )

    grounded_tafsir = ground_text(
        tafsir_text, context_chunks, chunks, refusal=_TAFSIR_REFUSAL,
        quran_quotes=True,
    )
    if grounded_tafsir == _TAFSIR_REFUSAL:
        relaxed = ground_text(
            tafsir_text,
            context_chunks,
            chunks,
            refusal=_TAFSIR_REFUSAL,
            ignore_rare=True,
            quran_quotes=True,
        )
        if relaxed != _TAFSIR_REFUSAL:
            grounded_tafsir = relaxed

    if grounded_tafsir == _TAFSIR_REFUSAL or not context_text.strip():
        return _clean_generated_text(grounded_tafsir), ""
    grounded_context = ground_text(
        context_text, context_chunks, chunks, refusal="",
        quran_quotes=True,
    )
    return _clean_generated_text(grounded_tafsir), _clean_generated_text(
        grounded_context
    )


def get_tafsir(
    ayah_text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Retrieve Tafsir Ibn Kathir for a Quranic verse.

    Pipeline:
        1. Resolve the verse lexically against the canonical Quran corpus.
        2. Look up the exact tafsir chunk by its surah:ayah reference.
        3. Send only that chunk (plus the next one) to GPT-4o-mini.
        4. Compute 'source' from the resolved verse — never from GPT output.
        5. Return a structured tafsir result.

    Args:
        ayah_text: The Quranic verse text to look up.
        index:     A loaded FAISS index.
        chunks:    The chunk dicts aligned with the index.
        client:    An authenticated OpenAI client.
        top_k:     (unused — kept for interface compatibility)
        debug:     If True, print raw GPT response and top FAISS hits.

    Returns:
        Dict with keys: tafsir, context, source, input_ayah.
    """
    resolution = resolve_tafsir_reference(
        ayah_text, index, chunks, client, debug=debug
    )
    primary, next_chunk, source_str = resolution
    failure_message = getattr(resolution, "failure_message", "")
    quran_chunk = getattr(resolution, "quran_chunk", None)
    source_scope = getattr(resolution, "source_scope", None)

    # Nothing found, an ambiguous text, or a verse without a tafsir record
    if primary is None:
        return {
            "tafsir": failure_message or _TAFSIR_REFUSAL,
            "context": "",
            "source": source_str,
            "input_ayah": ayah_text,
        }

    # Build context from only the primary (+ next) chunk(s).  Duplicated
    # contiguous commentary is sent once; repeating the byte-identical text
    # adds no evidence and encourages cross-verse blending.
    context_chunks = [primary]
    if next_chunk is not None:
        primary_text = " ".join(
            (primary.get("original_text") or "").split()
        )
        next_text = " ".join(
            (next_chunk.get("original_text") or "").split()
        )
        if next_text and next_text != primary_text:
            context_chunks.append(next_chunk)

    sources_text = "\n\n".join(
        f"[مصدر {i + 1}]\n"
        f"النص: {c.get('original_text', '')}\n"
        f"المرجع: {c.get('reference', '')}"
        for i, c in enumerate(context_chunks)
    )

    scope_note = ""
    if source_scope and source_scope[2] > source_scope[1]:
        scope_surah, scope_first, scope_last = source_scope
        scope_note = (
            "ملاحظة تحقق من المصدر: نص التفسير المرفق مسجل في ملف "
            f"المصدر للآيات {scope_surah}:{scope_first}–"
            f"{scope_surah}:{scope_last}، وليس حصراً للآية "
            f"{source_str}. لا تنسب ما يخص آية أخرى إلى الآية المطلوبة.\n"
        )

    canonical_text = ""
    if quran_chunk is not None:
        canonical_text = quran_chunk.get("original_text", "") or ""
    target_text = canonical_text or ayah_text
    canonical_block = (
        f"الآية المرجعية ({source_str}): {canonical_text}\n"
        if canonical_text else ""
    )

    user_message = (
        f"الآية المدخلة: {ayah_text}\n"
        f"المرجع الذي تم التحقق منه: {source_str}\n"
        f"{canonical_block}\n"
        f"الآية المطلوب تفسيرها فقط: {target_text}\n"
        "لا تشرح أي آية أخرى من السورة، ولا تدرج معنى آية أخرى في "
        "حقل 'tafsir'. ما يتعلق بالسورة عموماً اجعله في 'context' فقط "
        "إن ورد صراحة في المصدر. استخدم ألفاظ المصدر نفسها قدر الإمكان، "
        "ولا تستبدل كلمة واردة فيه بمرادف غير موجود في المقاطع.\n"
        f"{scope_note}\n"
        f"مقاطع التفسير المسترجعة:\n{sources_text}\n\n"
        "تنبيه: تم التحقق من وجود الآية في المصادر المرفقة."
    )

    grounded_tafsir, grounded_context = _generate_tafsir(
        client,
        SYSTEM_PROMPT,
        user_message,
        context_chunks,
        chunks,
        debug=debug,
    )

    if grounded_tafsir == _TAFSIR_REFUSAL and quran_chunk is not None:
        retry_tafsir, retry_context = _generate_tafsir(
            client,
            SYSTEM_PROMPT_RETRY,
            user_message,
            context_chunks,
            chunks,
            debug=debug,
        )
        if retry_tafsir != _TAFSIR_REFUSAL:
            grounded_tafsir = retry_tafsir
            grounded_context = retry_context

    result = {
        "tafsir": grounded_tafsir,
        "context": grounded_context,
        "source": source_str,
        "input_ayah": ayah_text,
    }
    if source_scope and source_scope[2] > source_scope[1]:
        scope_surah, scope_first, scope_last = source_scope
        result["source_scope"] = (
            f"{scope_surah}:{scope_first}–{scope_surah}:{scope_last}"
        )
    return result


if __name__ == "__main__":
    # --- Load resources ---
    client = get_client()
    index, chunks = load_faiss_index()

    # --- Diagnostic: print one quran chunk and one tafsir chunk ---
    print("=== DIAGNOSTIC: chunk field names ===")
    quran_sample = next(c for c in chunks if c.get("source") == "quran")
    tafsir_sample = next(c for c in chunks if c.get("source") == "tafsir")
    print("Quran chunk keys:  ", list(quran_sample.keys()))
    print("Quran metadata:    ", quran_sample.get("metadata"))
    print("Tafsir chunk keys: ", list(tafsir_sample.keys()))
    print("Tafsir metadata:   ", tafsir_sample.get("metadata"))
    print("==================")

    # --- Test cases ---
    test_inputs = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "إياك نعبد وإياك نستعين",
        "قل هو الله أحد",
        "اهدنا الصراط المستقيم",
        "آية غير موجودة إطلاقاً اختبار فقط",
    ]

    for ayah in test_inputs:
        print(f"\n{'=' * 60}")
        print(f"الآية المدخلة: {ayah}")
        print("-" * 60)

        result = get_tafsir(ayah, index, chunks, client, debug=True)

        print(f"التفسير:       {result['tafsir'][:120]}{'…' if len(result.get('tafsir','')) > 120 else ''}")
        print(f"السياق:        {result['context'][:80] if result.get('context') else ''}")
        print(f"المصدر:        {result['source']}")
        print(f"الآية المدخلة: {result['input_ayah']}")
        print("=" * 60)
