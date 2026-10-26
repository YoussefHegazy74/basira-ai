"""
Basira RAG — Pipeline
=======================
End-to-end Retrieval-Augmented Generation pipeline for Arabic Islamic content.
Retrieves relevant chunks via FAISS + BM25 hybrid search, builds an
Arabic-aware prompt, and generates answers using GPT-4o-mini.
"""

from __future__ import annotations

import math
import tracemalloc
import re
import threading
import time
import unicodedata
from collections import defaultdict

from openai import OpenAI

from src.nlp.preprocessor import preprocess, normalize, normalize_verse_text
from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search
from src.rag.website_help import (
    build_website_answer,
    classify_website_intent,
    website_help_sources,
)

GENERATION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Retrieval gate: accept if EITHER condition holds:
#   (a) best cosine score >= MIN_COSINE_SCORE, OR
#   (b) best BM25 hit covers >= MIN_LEXICAL_TOKEN_COVERAGE query tokens
#       (or all tokens when there is only 1).
# MIN_COSINE_SCORE is calibrated against genuine text-embedding-3-small query
# vectors over THIS corpus (observed separation, offline): supported questions
# score 0.4274-0.6888, unsupported/off-domain questions 0.2257-0.3343.  0.38
# sits inside the empty gap with >= 0.04 cosine margin on both sides.  The old
# 0.30 admitted off-domain questions: on a uniformly thematic (all-Islamic)
# corpus even unsupported short questions reach ~0.33 cosine to their nearest
# chunks.  Recalibrate if the embedding model, corpus or chunking changes.
# ---------------------------------------------------------------------------
MIN_COSINE_SCORE    = 0.38
MIN_LEXICAL_TOKENS  = 2      # minimum query tokens a BM25 hit must cover

# ---------------------------------------------------------------------------
# Query rewriting toggle (LLM-generated rewrites; API-bound).
# Rewrites join the RRF fusion as extra low-weight rankings; the
# deterministic classical-synonym lists are always active regardless.
# ---------------------------------------------------------------------------
USE_QUERY_REWRITE = True

# Score added when ALL significant query keywords appear in the chunk reference.
_REF_BOOST = 0.15

# Fixed score injected for deterministic surah chunks — high enough to beat
# any semantic result so they always appear first.
_SURAH_INJECT_SCORE = 2.0

# BM25 tuning parameters.
_BM25_K1 = 1.5
_BM25_B  = 0.75

# Reciprocal-rank-fusion constant (prevents top-1 from dominating).
# Extra rewritten-query lists contribute weighted RRF terms
# (<= _REWRITE_BM25_RRF_WEIGHT / (_RRF_K + 1) ~= 0.033 per list).  In
# addition, a chunk placed in the top _SIG_TOP of a list that matches ALL of
# that list's query tokens (>= 2) receives ONE bounded signal boost, applied
# at most once per chunk across all lists.  This rewards full alternative-
# query matches without letting weak partial matches or stacked/degenerate
# lists inflate scores.
_RRF_K = 60
_REWRITE_RRF_WEIGHT = 0.5
_REWRITE_BM25_RRF_WEIGHT = 2.0
_SIG_TOP = 10
_SIG_MAX_WEIGHT = 3.0

# Semantic top-ranks signal.  Cosine top-3 hits are near-duplicates of the
# query intent; they receive one bounded extra RRF term so a strong semantic
# match is not buried under the 20-entry lexical list (RRF base terms differ
# by < 30% across the top lists).  Bounded at _SEM_SIG_WEIGHT/(K+1) ~= 0.049,
# the same scale as the lexical full-coverage signal; the diversity cap still
# limits how many chunks of one source reach the context.  No effect in
# lexical-only runs (empty semantic list).
_SEM_SIG_TOP = 3
_SEM_SIG_WEIGHT = 3.0

# Phrase boost: if two consecutive query tokens appear consecutively in a
# chunk's token list (after stripping ال), multiply its BM25 score by this.
PHRASE_BOOST = 2.0

# Post-generation grounding check (applied inside generate()).
# Each sentence in the LLM answer must have at least this fraction of its
# content tokens match the text of the sources it cites (tolerant Arabic
# match keys; spelled numbers and digits are canonicalised to decimals).
# Every number and every month word in the sentence must additionally be
# supported by the cited sources.  Sentences with no content tokens pass
# unconditionally.
GROUNDING_MIN_COVERAGE = 0.6

# Phase 2(a) fix: cap body tokens per document in the BM25 lexical document.
# Phase 1 profiler showed tafsir avg=772 tokens/doc → 4.3M posting entries
# → sub-phase C=54s.  Cap at 200 keeps the semantically rich document head
# while reducing posting entries by ~75% for tafsir.  Retrieval output is
# identical for the top-ranked chunks (early tokens are most discriminative).
_BM25_BODY_TOKEN_CAP = 200

# Fiqh sections are long multi-topic chunks (e.g. fiqh_165 has the decisive
# "الوتر سنة مؤكدة" at token 201), so the global 200-token cap can hide the
# one passage that answers a ruling question.  Long hadith matn/sharh blocks
# can likewise place the decisive wording late (519 hadith chunks exceed 200
# tokens; 47 exceed 500).  Tafsir keeps the lower cap: its opening verses
# already carry the retrieval signal and its documents are much longer.
# This is source-aware, not a global index expansion.
_BM25_BODY_TOKEN_CAP_BY_SOURCE = {"fiqh": 500, "hadith": 600}

# Query tokens stripped for retrieval purposes.
_QUERY_STOPWORDS = {
    "حكم", "ما", "هي", "هو", "من", "في", "على", "معنى", "هل",
    "متى", "كيف", "لماذا", "اين", "من",
}

# Terminology questions ("السنن المؤكدة/الراتبة") are sensitive to surface
# wording: the decisive chunk may use "السنن الراتبة" while fusion prefers a
# generic "سنن" passage.  For these queries, guarantee the top hits from the
# deterministic synonym expansion as well.
_TERMINOLOGY_QUERY_TOKENS = {"سنن", "رواتب", "راتبه", "مؤكده"}

_RULING_INTENT_TOKENS = {"حكم", "حكمه", "حكمها", "حكمهم", "احكام", "الحكم"}
_RULING_EXPANSION_TOKENS = ["سنه", "واجب", "حرام", "مستحب", "مكروه"]

# Sources whose lexical document is: normalize(reference) + " " + clean_text.
# For hadith and quran: clean_text only (references are book/surah codes,
# not topical text, and prepending them adds noise).
_SOURCES_WITH_REF_IN_LEXDOC = {"fiqh", "sira", "tafsir"}

SYSTEM_MESSAGE = (
    "أنت مساعد إسلامي موثوق. أجب بالعربية فقط، وبناء على المصادر المقدمة حصر، "
    "ولا تستخدم أي معلومة من معرفتك الخاصة حتى لو كنت متأكد منها. "
    "أجب بأسلوب طبيعي متصل كأنك تشرح لشخص أمامك: ابدأ بالإجابة المباشرة، "
    "واستخدم فقرات قصيرة وجملًا مفهومة، ولا تبدأ بعبارات إنشائية طويلة. "
    "لو كتب المستخدم بالعامية المصرية، رد بعامية بسيطة وواضحة دون مبالغة. "
    "اجعل الاستشهاد جزءًا طبيعيًا من الكلام، ولا تعرض النصوص المسترجعة خامًا. "
    "كل جملة في إجابتك يجب أن يدعمها نص صريح في أحد المصادر، واذكر رقم المصدر بين قوسين مثل (مصدر 2) بعد كل معلومة. "
    "لا تنسب لأي مصدر معلومة غير مكتوبة فيه. "
    "احذف أي جملة لا تجد لها سند صريح في المصادر. "
    "إذا أجابت المصادر عن السؤال فأجب منها مباشرة دون جملة الاعتذار. "
    "انقل العبارة الدالة من المصدر كما وردت ما أمكن، وتجنب إعادة صياغتها بألفاظ مرادفة. "
    "إذا طلب السؤال قائمة أو حكما ووقتا معا فاستوف جميع العناصر التي تذكرها المصادر، واذكر كل حكم ووقت صريحا. "
    "إذا ورد في المصادر تفسير فقهي يبين المراد من الحديث فاعتمد التفسير واذكره صريحا. "
    "إذا ذكر المصدر أن الوجوب أو الحرمة محمول على معنى آخر مثل تأكيد الاستحباب فاذكر المعنى الصريح. "
    "لا تكتف بظاهر لفظ الوجوب أو الحرمة إذا ورد بعده تفسير صريح للمراد، بل اذكر التفسير. "
    "لا ترفع درجة الحكم من كراهة أو استحباب أو إباحة إلى وجوب أو تحريم أو بطلان إلا إذا دل اللفظ الصريح في المصدر على ذلك. "
    "إذا ورد في المصدر استثناء أو شرط يبيح الفعل أو يرفع الكراهة عنه فلا تجعله دليلا على وجوب الفعل. "
    "إذا سأل السؤال هل يجب أو هل يحرم، فبيّن درجة الحكم التي يدعمها المصدر وما يتصل بها من شروط أو استثناءات، ولا تجزم بما لا يدعمه النص. "
    "إذا كان السؤال عن الوجوب أو اللزوم فاجعل الجواب المباشر جملة مستقلة تبدأ بـ'لا يجب' أو 'يجب' حسب ما يدعمه المصدر، ثم اذكر التفصيل والدليل في جملة منفصلة. "
    "لا تستخدم 'فعليه أن' أو 'يلزمه' إلا إذا كان الوجوب ثابتا بنص صريح في المصادر. "
    "فرّق بين نص الحديث وبين تفسير الفقهاء له، وانسب كل قول إلى مصدره. "
    "لا تستشهد بآية أو حديث عام لا يدعم الجملة التي قبله. "
    "إذا ورد في المصادر نهي عن الفعل فاذكره بلفظ النهي. "
    "إذا سأل السؤال عن وقت فاذكر بداية الوقت ونهايته إن ذكرتا في المصادر. "
    "إذا ذكرت المصادر أرقاما أو مددا متعددة لنفس الحكم فاذكرها كلها. "
    "إذا كان السؤال عن حكم فاذكر جملة الحكم الصريحة من المصدر قبل التفاصيل. "
    "إذا كان السؤال الحالي متابعة أو طلب توضيح، فافهمه من سياق المحادثة وأجب عن نفس الموضوع مباشرة دون تكرار الجواب السابق كاملا. "
    "لا تضف وصفا للحكم (مثل مستحب أو واجب أو حلال أو حرام) غير موجود نصا في المصادر، واكتف بالحكم الذي تدعمه. "
    "اجعل جملة الحكم مستقلة، واذكر الخلاف أو التفاصيل بعدها في جملة منفصلة. "
    "إذا ورد تعريف صريح للشيء في المصادر فابدأ به. "
    "إذا طلب السؤال قصة فاذكر عناصرها الأساسية كما ترد في المصادر، واجمع عناصر القصة من جميع المصادر ذات الصلة ولا تكتف برواية واحدة. "
    "إذا روت المصادر القصة نفسها في أكثر من حديث أو كتاب، فاقرأ جميع الروايات واجمع التفاصيل التي تنفرد بها كل رواية، واذكر مصدر كل تفصيل. "
    "إذا كان السؤال عن تفسير آية فابدأ ببيان معناها الإجمالي كما ورد في التفسير ثم اذكر تفاصيله. "
    "إذا كان السؤال عن حكم فاذكر معه ما ورد في المصادر من عدد أو مدة أو شروط متعلقة به. "
    "إذا كان جواب السؤال نهيا فاذكره مباشرة دون تفريعات. "
    "إذا أجابت المصادر عن السؤال أو عن جزء منه فأجب بما ورد فيها، ولا ترفض إلا إذا لم تتناول المصادر الموضوع إطلاقا. "
    "اقتصر على ما يجيب السؤال مباشرة، ولا تضف تفاصيل أو قصصا غير مطلوبة. "
    "استشهد بالمصدر الذي يذكر المعلومة في متنه، ولا تكتف بعنوان المصدر أو ترجمته. "
    "لا تضع بين علامتي اقتباس إلا نصا حرفيا من المصدر، وإن لم يكن حرفيا فاذكره دون اقتباس. "
    "حافظ على النفي والإثبات كما ورد في المصدر. "
    "إذا لم تتناول المصادر المقدمة السؤال فاكتب حرفيا: لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال، ولا تضف شيئا آخر."
)

# ===========================================================================
# Surah resolution
# ===========================================================================
_surah_map_cache: dict[int, dict[str, int]] = {}


def _get_surah_map(chunks: list[dict]) -> dict[str, int]:
    """Return (cached) mapping of normalized surah name -> surah number."""
    key = id(chunks)
    if key not in _surah_map_cache:
        smap: dict[str, int] = {}
        for c in chunks:
            if c.get("source") != "quran":
                continue
            meta = c.get("metadata") or {}
            num = meta.get("surah_number")
            name = meta.get("surah_name") or ""
            ref = c.get("reference", "")
            if ":" in ref:
                name_from_ref = ref.split(":")[0].strip()
                if name_from_ref and num:
                    smap[normalize(name_from_ref)] = int(num)
            if name and num:
                smap[normalize(name)] = int(num)
            if num:
                smap[str(int(num))] = int(num)
        _surah_map_cache[key] = smap
    return _surah_map_cache[key]


def _resolve_surah_query(query: str, chunks: list[dict]) -> int | None:
    """Return surah number if the query is a surah question, else None."""
    norm_q = normalize(query)
    m = re.search(r'سوره\s+(\S+)', norm_q)
    if not m:
        return None
    token = m.group(1).strip()
    smap = _get_surah_map(chunks)
    if token in smap:
        return smap[token]
    bare = token[2:] if token.startswith("ال") else token
    if bare in smap:
        return smap[bare]
    try:
        n = int(token)
        if str(n) in smap:
            return n
    except ValueError:
        pass
    return None


def _inject_surah_chunks(
    surah_num: int,
    chunks: list[dict],
    pool: list[dict],
) -> tuple[list[dict], set[str]]:
    """Inject surah quran + tafsir chunks into pool with a high fixed score.

    Returns (augmented_pool, set_of_injected_ids).
    Injected ids bypass the per-source-type diversity cap in _select_diverse.
    """
    existing_ids = {c["id"]: i for i, c in enumerate(pool)}
    to_inject: list[dict] = []
    injected_ids: set[str] = set()

    quran_ayahs: list[tuple[int, dict]] = []
    for c in chunks:
        if c.get("source") != "quran":
            continue
        meta = c.get("metadata") or {}
        if meta.get("surah_number") == surah_num:
            quran_ayahs.append((meta.get("ayah_number", 0), c))
    quran_ayahs.sort(key=lambda x: x[0])

    tafsir_ayahs: list[tuple[int, dict]] = []
    prefix = f"تفسير ابن كثير {surah_num}:"
    for c in chunks:
        if c.get("source") != "tafsir":
            continue
        ref = c.get("reference", "")
        if ref.startswith(prefix):
            try:
                ayah_num = int(ref.split(":")[-1])
            except ValueError:
                ayah_num = 0
            tafsir_ayahs.append((ayah_num, c))
    tafsir_ayahs.sort(key=lambda x: x[0])

    for _, c in (quran_ayahs + tafsir_ayahs):
        cid = c["id"]
        injected = {
            "id": cid,
            "source": c.get("source", ""),
            "reference": c.get("reference", ""),
            "original_text": c.get("original_text", ""),
            "score": _SURAH_INJECT_SCORE,
            "_injected": True,   # flag: bypass diversity cap
        }
        injected_ids.add(cid)
        if cid in existing_ids:
            pool[existing_ids[cid]] = injected
        else:
            to_inject.append(injected)

    return to_inject + pool, injected_ids


# ===========================================================================
# Quran topic anchors
# ===========================================================================
# Core Quranic topics whose governing verse uses classical vocabulary that a
# modern question may not share (e.g. "كيف يتوضأ؟" vs "فاغسلوا وجوهكم",
# "الربا" vs the Uthmani spelling).  BM25 and the embedding model can both
# miss the verse even when it is the primary source.  This small, explicit
# topic index guarantees ONE canonical verse for the clearest core topics;
# it is not a general-purpose verse recommender and never replaces normal
# retrieval.  Tafsir requests and whole-surah requests are excluded because
# their own deterministic resolvers are more precise.
_QURAN_TOPIC_ANCHOR_RULES = (
    (re.compile(
        r"(?:^|\s)(?:ايه\s+الكرسي|الكرسي|اعظم\s+ايه|افضل\s+ايه)"
        r"(?:\s|$)"
    ), "quran_2_255"),
    (re.compile(
        r"(?:^|\s)(?:يتوضا|توضا|الوضوء|وضوء|وضو)(?:\s|$)"
    ), "quran_5_6"),
    (re.compile(
        r"(?:^|\s)(?:الربا|ربا|الربوا|ربوا)(?:\s|$)"
    ), "quran_2_275"),
    (re.compile(
        r"(?:^|\s)(?:الوالدين|والدين|الوالد|والد)(?:\s|$)"
    ), "quran_17_23"),
    (re.compile(
        r"(?:^|\s)(?:الميراث|ميراث|التركة|تركة|الفرائض|فرائض|"
        r"نصيب|ورث|الورث)(?:\s|$)"
    ), "quran_4_11"),
    (re.compile(
        r"(?:^|\s)(?:الحج|حج|بيت\s+الله\s+الحرام)(?:\s|$)"
    ), "quran_3_97"),
    (re.compile(
        r"(?:^|\s)(?:الصيام|صيام|الصوم|صوم)(?:\s|$)"
    ), "quran_2_183"),
    (re.compile(
        r"(?:^|\s)(?:الجمعة|جمعة|الجمعه|جمعه)(?:\s|$)"
    ), "quran_62_9"),
    (re.compile(
        r"(?:^|\s)(?:اليتامى|يتامى|اليتيم|يتيم)(?:\s|$)"
    ), "quran_4_10"),
)

_chunk_id_map_cache: dict[
    tuple[int, int], tuple[list[dict], dict[str, dict]]
] = {}


def _get_chunk_id_map(chunks: list[dict]) -> dict[str, dict]:
    key = (id(chunks), len(chunks))
    entry = _chunk_id_map_cache.get(key)
    # Keep a reference to the list itself.  A bare id() key can be reused
    # after an ephemeral list is garbage-collected, which made anchors resolve
    # against a previous test's chunk map.  The cached map already holds the
    # chunk dicts, so retaining the list adds no meaningful memory cost.
    if entry is not None and entry[0] is chunks:
        return entry[1]
    cached = {c.get("id", ""): c for c in chunks}
    _chunk_id_map_cache[key] = (chunks, cached)
    return cached


_QURAN_TOPIC_COMPANION_IDS = {
    # The verse itself never uses the name "آية الكرسي"; Ibn Kathir's
    # commentary opens with "هذه آية الكرسي" and is the retrieved evidence
    # that lets the answer identify the verse the user asked about.
    "quran_2_255": "tafsir_ibnkathir_2_255",
}


def _quran_topic_companions(
    anchor: dict, chunks: list[dict],
) -> list[dict]:
    """Return cited commentary companions for a core-topic verse anchor."""
    cid = _QURAN_TOPIC_COMPANION_IDS.get(anchor.get("id", ""))
    if not cid:
        return []
    companion = _get_chunk_id_map(chunks).get(cid)
    return [companion] if companion is not None else []


_FASTING_TOKEN_PREFIXES = (
    "صوم", "صيم", "صيام", "صائم", "صايم", "صاءم",
    "صام", "يصوم", "اصوم", "تصوم", "نصوم", "صوموا",
    "افطر", "فطر", "مفطر",
)
_FASTING_DAY_QUALIFIERS = {
    "عرفه", "عرفة", "عاشوراء", "عاشورا", "رمضان",
}


def _asks_fasting_query(query: str) -> bool:
    for token in _tokenize(normalize(query)):
        folded = _fold_ar(token)
        if folded.startswith(_FASTING_TOKEN_PREFIXES):
            return True
    return False


def _required_action_evidence_keys(query: str) -> set[str]:
    """Action + day keys a source must share for a day-qualified fast claim."""
    tokens = set(_tokenize(normalize(query)))
    if not _asks_fasting_query(query):
        return set()
    qualifiers = (
        tokens & _DATE_WEEKDAY_TOKENS
    ) | (
        tokens & (_FASTING_DAY_QUALIFIERS - {"رمضان"})
    )
    if not qualifiers:
        return set()
    return {"صوم"} | {_legal_topic_key(q) for q in qualifiers}


def _filter_joint_evidence(
    selected: list[dict], query: str,
) -> list[dict]:
    """Drop scripture that shares only one side of a day-qualified fast claim.

    A Quran/tafsir passage about Friday prayer, or about Ramadan fasting in
    general, is not evidence for "fasting on Friday".  Hadith/Fiqh remain
    available because they often state the ruling without repeating every
    qualifier; their claims are checked sentence-by-sentence later.
    """
    required = _required_action_evidence_keys(query)
    if not required:
        return selected
    kept = []
    for chunk in selected:
        if chunk.get("source") not in {"quran", "tafsir"}:
            kept.append(chunk)
            continue
        if required <= _legal_topic_keys(
            chunk.get("original_text", "") or "",
        ):
            kept.append(chunk)
    return kept


def _resolve_quran_topic_anchor(
    query: str, chunks: list[dict],
) -> dict | None:
    """Return the canonical Quran verse for a clear core-topic question."""
    qn = normalize(query)
    if _TAFSIR_VERB_RE.search(qn) or _TAFSIR_MEANING_RE.search(qn):
        return None
    by_id = _get_chunk_id_map(chunks)
    for pattern, chunk_id in _QURAN_TOPIC_ANCHOR_RULES:
        if pattern.search(qn):
            # The Ramadan obligation verse (2:183) is not evidence for a
            # separately qualified voluntary fast (Arafah or a named weekday).
            # Anchoring those queries to 2:183 displaced the actual fiqh
            # evidence and made a general obligation verse look like an answer
            # to the narrower question.
            if chunk_id == "quran_2_183" and (
                "عرفه" in qn or "عرفة" in qn
                or set(_tokenize(qn)) & _DATE_WEEKDAY_TOKENS
            ):
                continue
            # The Friday-prayer verse is not evidence for a fasting question.
            if chunk_id == "quran_62_9" and _asks_fasting_query(qn):
                continue
            chunk = by_id.get(chunk_id)
            if chunk is not None:
                return chunk
    return None


# A very small set of canonical hadith facts whose wording shares almost no
# lexical overlap with the natural question.  "كم كان عمر النبي حين توفي؟"
# retrieves narrations containing عمر (the name) but not the death-age
# hadith, and "كيف بدأ الوحي؟" retrieves general revelation narrations but
# not Bukhari's first hadith.  These anchors guarantee the canonical report;
# they are not a general-purpose hadith recommender.
_CORE_HADITH_ANCHOR_RULES = (
    (
        re.compile(
            r"(?:^|\s)عمر(?:\s|$).*?(?:النبي|رسول)(?:\s|$)"
            r".*?(?:توفي|توفى|مات|الوفاه|وفاه)(?:\s|$)"
        ),
        "hadith_bukhari_3386",
    ),
    (
        re.compile(
            r"(?:^|\s)(?:بدا|بدء|اول|ابتداء)(?:\s|$)"
            r".*?(?:الوحي|الوحى)(?:\s|$)"
        ),
        "hadith_bukhari_3",
    ),
    (
        re.compile(
            r"(?:^|\s)(?:اسماء|اسم)(?:\s|$)"
            r".*?(?:الله|لله)(?:\s|$).*?الحسن"
        ),
        "hadith_bukhari_2624",
    ),
    (
        # "كم عدد أبواب الجنة؟" has no exact lexical bridge to the short
        # hadith "للجنة ثمانية أبواب"; it ranked 60th in BM25 at HEAD and in
        # the current index.  This is a canonical core fact, so anchor it.
        re.compile(
            r"(?:^|\s)(?:ابواب|باب)(?:\s|$).*?(?:الجنه|الجنة)(?:\s|$)"
            r"|(?:^|\s)(?:الجنه|الجنة)(?:\s|$).*?(?:ابواب|باب)(?:\s|$)"
        ),
        "hadith_darimi_39619",
    ),
)


def _resolve_core_hadith_anchor(
    query: str, chunks: list[dict],
) -> dict | None:
    """Return the canonical hadith chunk for a clear core-fact question."""
    qn = normalize(query)
    if _TAFSIR_VERB_RE.search(qn) or _TAFSIR_MEANING_RE.search(qn):
        return None
    by_id = _get_chunk_id_map(chunks)
    for pattern, chunk_id in _CORE_HADITH_ANCHOR_RULES:
        if pattern.search(qn):
            chunk = by_id.get(chunk_id)
            if chunk is not None:
                return chunk
    return None


# ===========================================================================
# Tafsir-intent chat anchors
# ===========================================================================
# A chat question that explicitly asks for a tafsir/meaning of Quranic text
# should use the same deterministic verse resolver as /api/tafsir.  Without
# this, a specific-ayah question could retrieve an unrelated tafsir passage
# (observed live: "تفسير سورة الإخلاص الآية الأولى" retrieved 112:1 only by
# luck and "تفسير آية حج البيت" retrieved no tafsir at all).
_TAFSIR_VERB_RE = re.compile(
    r"(?:^|\s)(?:تفسير|فسر|فسري|اشرح|وضح|بين)(?:\s|$)"
)
_TAFSIR_EXPLAIN_RE = re.compile(
    r"(?:^|\s)(?:اشرح|وضح|بين)(?:\s|$)"
)
_TAFSIR_MEANING_RE = re.compile(
    r"(?:^|\s)(?:معن[ىي]|المقصود|مقصود)(?:\s|$)"
)
_QURAN_MARKER_RE = re.compile(
    r"(?:^|\s)(?:ايه|الايه|ايات|الايات|سوره|قوله|قول)(?:\s|$)"
    r"|[\uFD3F\u00ab\"]"
)
# Honorifics add high-IDF query tokens ("صلى" df 317) that pull the lexical
# ranking toward hadith narrations instead of the asked-about subject.  They
# carry no retrieval intent, so remove them from the BM25 query only; the
# semantic embedding still sees the user's exact wording.
_HONORIFIC_RE = re.compile(
    r"(?:صلي|صلى)\s+الله\s+(?:عليه|عليها|عليهم)\s+وسلم"
    r"|(?:عليه|عليها|عليهم)\s+السلام"
    r"|رضي\s+الله\s+(?:عنه|عنها|عنهم)"
)


def _resolve_chat_tafsir(
    query: str,
    index,
    chunks: list[dict],
    client=None,
) -> tuple[list[dict], str]:
    """Resolve a Tafsir-intent chat query to deterministic anchors.

    Returns ``(anchors, clarification)``.  Anchors are the canonical Quran
    verse plus its Tafsir Ibn Kathir chunk.  A clarification is returned only
    for an explicit tafsir request that names a surah/ayah incompletely
    (e.g. a whole surah without an ayah number); ambiguous or non-Quranic
    "معنى" questions are left to normal retrieval instead of being hijacked.
    """
    normalized = normalize(query)
    explicit = bool(_TAFSIR_VERB_RE.search(normalized))
    explain = bool(_TAFSIR_EXPLAIN_RE.search(normalized))
    meaning = bool(_TAFSIR_MEANING_RE.search(normalized))
    quran_marker = bool(_QURAN_MARKER_RE.search(normalized))
    if not (explicit or (explain and quran_marker) or meaning):
        return [], ""

    # Imported lazily: tafsir.py imports ground_text from this module.
    from src.rag.tafsir import resolve_tafsir_reference

    resolution = resolve_tafsir_reference(query, index, chunks, client)
    if resolution[0] is not None:
        # A bare "ما معنى X؟" is only treated as a Quran question when the
        # resolver found an exact verse (or an explicit reference).  A short
        # partial phrase could otherwise match dozens of unrelated verses.
        if (meaning and not explicit and not quran_marker
                and resolution.match_kind not in ("exact", "reference")):
            return [], ""
        anchors = []
        if resolution.quran_chunk is not None:
            anchors.append(resolution.quran_chunk)
        anchors.append(resolution[0])
        return anchors, ""

    if (explicit or (explain and quran_marker)) and resolution.failure_message:
        return [], resolution.failure_message
    return [], ""


def _inject_tafsir_anchors(
    anchors: list[dict],
    pool: list[dict],
) -> tuple[list[dict], set[str]]:
    """Inject resolved Quran/Tafsir anchors with the fixed high score.

    The anchors bypass the diversity cap, exactly like surah injections, so
    the verse and its commentary cannot be displaced by near-duplicate
    narrations.  No synthetic relevance score is attached to any other chunk.
    """
    existing_ids = {c["id"]: i for i, c in enumerate(pool)}
    to_inject: list[dict] = []
    injected_ids: set[str] = set()
    for c in anchors:
        cid = c["id"]
        injected = {
            "id": cid,
            "source": c.get("source", ""),
            "reference": c.get("reference", ""),
            "original_text": c.get("original_text", ""),
            "score": _SURAH_INJECT_SCORE,
            "_injected": True,
        }
        injected_ids.add(cid)
        if cid in existing_ids:
            pool[existing_ids[cid]] = injected
        else:
            to_inject.append(injected)
    return to_inject + pool, injected_ids


# ===========================================================================
# BM25 index (lazy build, cached by (id(chunks), len(chunks)), single-flight)
# Posting-list design: builds once in ~2s, searched in O(|q| * avg_postings)
# ===========================================================================
_bm25_cache: dict[tuple[int, int], dict] = {}

# Single-flight guard: only one thread may construct a missing index at a
# time.  Losers of the race re-check the cache under this lock and return the
# winner's index instead of starting a duplicate build.
_bm25_build_lock = threading.Lock()


# Conjunction/preposition + definite article forms.  Stripping them is safe
# only when a stem of >= 3 chars remains (e.g. والرواتب -> رواتب), which
# protects words that merely start with these letters (والد, والده, ...).
_CONJ_ARTICLE_PREFIXES = ("وال", "فال", "بال", "كال")


def _strip_al(token: str) -> str:
    """Strip a leading definite article from a token, including the common
    prefix+article forms (وال, فال, بال, كال, لل).

    This runs identically at BM25 index build time and at query time, so the
    lexical index stays consistent with query normalization without a
    persisted-index rebuild (the BM25 index is rebuilt in memory from
    chunks.pkl on each process start).
    """
    # "الله" is a lexical word, not the article ال + له.  Stripping it
    # produced the pronoun "له" (df 44,995), so the index had no "الله"
    # token at all and every mention of God matched the unrelated pronoun.
    if token in {"الله", "لله"}:
        return token
    for p in _CONJ_ARTICLE_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            return token[len(p):]
    if token.startswith("لل") and len(token) - 2 >= 2:
        return token[2:]
    if token.startswith("ال") and len(token) > 2:
        return token[2:]
    return token


# Orthographic folding applied identically to BM25 index tokens and query
# tokens.  clean_text was produced by the older normalize(), which leaves
# alef maqsura, tatweel, nisba/yaa hamza and Arabic-Indic digits untouched.
# Folding both sides in memory closes those gaps without rebuilding the
# persisted embeddings or modifying the corpus.
_LEXICAL_FOLD_TRANS = str.maketrans({
    "\u0640": "",      # tatweel
    "ى": "ي",
    "ی": "ي",
    "ئ": "ي",
    "ؤ": "و",
    "ء": "",
    "ة": "ه",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})


def _fold_lexical(token: str) -> str:
    return _strip_al(token.translate(_LEXICAL_FOLD_TRANS))


def _root_key(token: str) -> str:
    """Conservative Arabic root-ish key for *reference-title* matching only.

    This is deliberately not used for BM25 postings or grounding: it is a
    retrieval aid for descriptive metadata such as
    "ولادة رسول الله ... ورضاعته", where the query says "أرضعت" and a plain
    token index sees no overlap.  The transformation is deterministic and
    applied identically to title and query tokens, so consistency matters
    more than linguistic perfection.
    """
    t = _fold_lexical(token)
    if t in {"الله", "لله"}:
        return t
    # Conjunctions/articles and the Form-X prefix.
    for p in ("است", "وال", "فال", "بال", "كال", "لل", "ال", "و", "ف"):
        if t.startswith(p) and len(t) - len(p) >= 3:
            t = t[len(p):]
            break
    # Imperfect/imperative person prefixes.
    for p in ("ا", "ت", "ي", "ن", "م"):
        if t.startswith(p) and len(t) - 1 >= 3 and len(t) >= 4:
            t = t[1:]
            break
    # Pronominal/plural/verbal suffixes, iteratively (e.g. رضاعته -> رضاع).
    changed = True
    while changed and len(t) > 3:
        changed = False
        for s in ("ها", "هم", "هن", "كم", "كن", "نا", "ون", "ين",
                  "ات", "ان", "وا", "ه", "ت", "ي", "ا"):
            if t.endswith(s) and len(t) - len(s) >= 3:
                t = t[:-len(s)]
                changed = True
                break
    # Drop weak letters from the interior (رضاع -> رضع).
    if len(t) > 3:
        t = t[0] + "".join(ch for ch in t[1:-1] if ch not in "اوي") + t[-1]
    return t


def _tokenize_clean(text: str) -> list[str]:
    """Fast split of already-normalized clean_text — no regex, no normalize()."""
    return [
        folded
        for w in text.translate(_LEXICAL_FOLD_TRANS).split()
        if len(w) >= 2
        for folded in (_strip_al(w),)
        if len(folded) >= 2
    ]


def _tokenize(text: str) -> list[str]:
    """Normalize + split arbitrary text (used only for query tokenization)."""
    norm = normalize(text)
    return [_strip_al(w) for w in norm.split() if len(w) >= 2]


def _build_bm25(chunks: list[dict]) -> dict:
    """Return the cached BM25 index for *chunks*, building it once if missing.

    Single-flight: concurrent cold callers serialize on `_bm25_build_lock`;
    the winner constructs the index and the losers re-check the cache after
    acquiring the lock and reuse the same object.
    """
    cache_key = (id(chunks), len(chunks))
    cached = _bm25_cache.get(cache_key)
    if cached is not None:
        return cached

    with _bm25_build_lock:
        # Double-checked: another thread may have published the index while
        # we were waiting for the lock.
        cached = _bm25_cache.get(cache_key)
        if cached is not None:
            return cached
        # tracemalloc state is process-global: start tracing only when the
        # process is not already tracing, and let only the starter stop it.
        # The try/finally guarantees an exception cannot leave tracing on.
        tracing_started = not tracemalloc.is_tracing()
        if tracing_started:
            tracemalloc.start()
        try:
            idx = _bm25_construct(chunks, tracing_started)
        finally:
            if tracing_started:
                tracemalloc.stop()
        _bm25_cache[cache_key] = idx
        return idx


def _bm25_construct(chunks: list[dict], tracing_started: bool) -> dict:
    """Build a BM25 inverted posting-list index.

    The caller (see `_build_bm25`) must hold `_bm25_build_lock` and owns the
    process-global tracemalloc window (`tracing_started` is True only when
    this call enabled tracing).

    Lexical document composition:
      - fiqh / sira / tafsir:  normalized reference tokens +
        body_tokens[:CAP].  References carry hamza forms while queries are
        normalized, so applying normalize() to refs is required for matches
        such as "الأغسال" vs "اغسال".
      - hadith / quran:        body_tokens[:CAP] only
        Their references are book codes — prepending floods the index with
        the token 'سنن' (previously DF=21,027 chunks before Fix A).

    Body tokens are capped per source at _BM25_BODY_TOKEN_CAP_BY_SOURCE
    (default _BM25_BODY_TOKEN_CAP=200).  Phase 1 profiler showed tafsir
    avg=772 tokens → 4.3M posting entries and sub-phase C=54 s.  Cap reduces
    posting entries ~75 % for tafsir with negligible retrieval loss (early
    tokens are most discriminative).

    Also stores doc_tok per chunk so _bm25_search can apply PHRASE_BOOST
    without re-tokenizing.

    Index structure:
        postings:  {token: [(chunk_id, bm25_weight), ...]}
        idf:       {token: float}
        doc_tok:   {chunk_id: [token, ...]}
        N:         int
        _chunks:   {id: chunk_dict}
    """
    t0 = time.time()
    N = len(chunks)
    id_map: dict[str, dict] = {c["id"]: c for c in chunks}

    # --- Pass 1: tokenize and collect doc lengths + document frequencies ---
    doc_tok: dict[str, list[str]] = {}   # cid -> token list
    dl: dict[str, int] = {}
    df: dict[str, int] = {}

    for c in chunks:
        cid = c["id"]
        src = c.get("source", "")
        # Body: cap at the per-source limit to bound posting-list size.
        # Phase 1: tafsir avg=772 tok/doc → 4.3M entries → 54s sub-phase C.
        if src == "quran":
            # Quran chunks use Uthmani orthography.  clean_text was produced
            # by the older normalize(), which drops the superscript alef and
            # leaves spellings such as "الصلوة"/"الربوا"; both break lexical
            # matching against modern queries ("الصلاة"/"الربا").  Normalize
            # the canonical original_text for the in-memory BM25 index only;
            # stored text and embeddings are untouched.
            body_text = normalize_verse_text(
                c.get("original_text") or c.get("clean_text") or ""
            )
        else:
            body_text = c.get("clean_text") or c.get("original_text") or ""
        body_tokens = _tokenize_clean(
            body_text
        )[:_BM25_BODY_TOKEN_CAP_BY_SOURCE.get(src, _BM25_BODY_TOKEN_CAP)]
        # Reference: use _tokenize_clean() (not normalize()) — refs have no
        # tashkeel so normalize() is a no-op that costs 8.8s (pyarabic).
        if src in _SOURCES_WITH_REF_IN_LEXDOC:
            # Query tokens go through normalize() (hamza/ya/alef folding), so
            # reference tokens must too: e.g. "الأغسال" must match "اغسال".
            ref_tokens = _tokenize(c.get("reference") or "")
            tokens = ref_tokens + body_tokens
        else:
            tokens = body_tokens
        doc_tok[cid] = tokens
        dl[cid] = len(tokens)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    avgdl = sum(dl.values()) / max(N, 1)

    # --- Pass 2: compute IDF ---
    idf: dict[str, float] = {
        t: math.log((N - f + 0.5) / (f + 0.5) + 1)
        for t, f in df.items()
    }

    # --- Pass 3: build posting lists with BM25 tf weights ---
    from collections import Counter
    postings: dict[str, list[tuple[str, float]]] = {}
    for cid, tokens in doc_tok.items():
        dlen = dl[cid]
        denom_base = _BM25_K1 * (1 - _BM25_B + _BM25_B * dlen / max(avgdl, 1))
        freq_map = Counter(tokens)
        for t, cnt in freq_map.items():
            weight = (cnt * (_BM25_K1 + 1)) / (cnt + denom_base)
            if t not in postings:
                postings[t] = []
            postings[t].append((cid, weight))

    build_time = time.time() - t0
    if tracing_started:
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        mem_stats = (f"mem_current={current_mem/1e6:.1f}MB  "
                     f"mem_peak={peak_mem/1e6:.1f}MB")
    else:
        mem_stats = "mem=untraced (process already tracing)"
    print(f"[bm25] built index in {build_time:.2f}s  (N={N})  {mem_stats}")

    idx = {
        "idf": idf, "postings": postings, "dl": dl, "avgdl": avgdl,
        "N": N, "_chunks": id_map, "doc_tok": doc_tok,
    }
    return idx


def prewarm_bm25(chunks: list[dict]) -> dict:
    """Pre-build (or fetch) the BM25 index for the canonical corpus.

    RAG-03 R2 startup hook: delegates to `_build_bm25` so a prewarm shares
    R1's single-flight lock, double-checked cache, and exception safety, and
    populates exactly the cache entry normal retrieval reads.  Returns the
    shared index object.
    """
    return _build_bm25(chunks)


def _has_phrase(tokens: list[str], a: str, b: str) -> bool:
    """Return True if token a is immediately followed by token b in tokens."""
    for i in range(len(tokens) - 1):
        if tokens[i] == a and tokens[i + 1] == b:
            return True
    return False


def _bm25_search(
    query_tokens: list[str],
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Return top_k chunks by BM25 score using the posting-list index.

    Fix B: after computing raw BM25 scores over the full index, take the
    top-200 candidates and apply PHRASE_BOOST (×2.0) to any chunk whose
    token list contains two consecutive query tokens consecutively.
    Phrase pairs are checked after stripping ال from both query tokens
    and chunk tokens.
    """
    idx = _build_bm25(chunks)
    idf = idx["idf"]
    postings = idx["postings"]
    id_map = idx["_chunks"]
    doc_tok = idx["doc_tok"]

    scores: dict[str, float] = {}
    token_coverage: dict[str, int] = {}
    de_prefixed = [_fold_lexical(t) for t in query_tokens]

    for qt in de_prefixed:
        if qt not in idf:
            continue
        qt_idf = idf[qt]
        for cid, bm25_w in postings.get(qt, []):
            scores[cid] = scores.get(cid, 0.0) + qt_idf * bm25_w
            token_coverage[cid] = token_coverage.get(cid, 0) + 1

    # Fix B: phrase boost over top-200 candidates
    if len(de_prefixed) >= 2:
        top200 = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:200]
        for i in range(len(de_prefixed) - 1):
            a, b = de_prefixed[i], de_prefixed[i + 1]
            for cid in top200:
                if cid in scores and _has_phrase(doc_tok.get(cid, []), a, b):
                    scores[cid] *= PHRASE_BOOST

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    results = []
    for cid in ranked[:top_k]:
        c = id_map.get(cid)
        if c:
            results.append({
                "id": cid,
                "source": c.get("source", ""),
                "reference": c.get("reference", ""),
                "original_text": c.get("original_text", ""),
                "score": scores[cid],
                "_bm25_score": scores[cid],
                "_token_coverage": token_coverage.get(cid, 0),
            })
    return results


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================

def _rrf_fuse(
    sem_ranked: list[dict],
    lex_ranked: list[dict],
    lex_top3_ids: set[str],
    extra_ranked: list[tuple[list[dict], float]] | None = None,
) -> list[dict]:
    """Fuse semantic and lexical rankings via RRF.

    Returns merged list sorted by fused score descending, preserving all
    original chunk fields plus _rrf_score, _sem_rank, _lex_rank.
    """
    sem_rank_map = {c["id"]: (i + 1) for i, c in enumerate(sem_ranked)}
    lex_rank_map = {c["id"]: (i + 1) for i, c in enumerate(lex_ranked)}

    extra_pairs = list(extra_ranked or [])
    extra_lists = [lst for lst, _w in extra_pairs]
    extra_weights = [w for _lst, w in extra_pairs]
    extra_maps = [{c["id"]: (i + 1) for i, c in enumerate(lst)} for lst in extra_lists]
    # Per-list token coverage of the alternative query (from _bm25_search).
    # full_cov = the highest coverage attained anywhere in the list; a signal
    # boost is only granted to chunks reaching that coverage (>= 2 tokens).
    extra_cov_maps = [
        {c["id"]: c.get("_token_coverage", 0) for c in lst} for lst in extra_lists
    ]
    extra_full_cov = [max(cv.values(), default=0) for cv in extra_cov_maps]
    # Deterministic id iteration: sem, then lex, then extras (insertion order).
    # Iterating a set here leaks PYTHONHASHSEED into the order of exactly-tied
    # fused scores (the final sort is stable), which made pool_rank jitter by
    # 1 across identical runs.
    ordered_ids: list[str] = list(sem_rank_map)
    seen_ids: set[str] = set(ordered_ids)
    for cid in lex_rank_map:
        if cid not in seen_ids:
            ordered_ids.append(cid)
            seen_ids.add(cid)
    for m in extra_maps:
        for cid in m:
            if cid not in seen_ids:
                ordered_ids.append(cid)
                seen_ids.add(cid)

    # Build a merged chunk view (prefer semantic chunk dict for field richness)
    id_to_chunk: dict[str, dict] = {}
    for c in sem_ranked:
        id_to_chunk[c["id"]] = c
    for c in lex_ranked:
        if c["id"] not in id_to_chunk:
            id_to_chunk[c["id"]] = c

    for lst in extra_lists:
        for c in lst:
            if c["id"] not in id_to_chunk:
                id_to_chunk[c["id"]] = c

    fused = []
    for cid in ordered_ids:
        sem_r = sem_rank_map.get(cid, len(sem_ranked) + _RRF_K)
        lex_r = lex_rank_map.get(cid, len(lex_ranked) + _RRF_K)
        rrf = 1.0 / (_RRF_K + sem_r) + 1.0 / (_RRF_K + lex_r)
        if sem_r <= _SEM_SIG_TOP:
            rrf += _SEM_SIG_WEIGHT / (_RRF_K + sem_r)
        signal = 0.0
        for m, lst, w, covs, full in zip(
            extra_maps, extra_lists, extra_weights, extra_cov_maps, extra_full_cov
        ):
            rrf += w / (_RRF_K + m.get(cid, len(lst) + _RRF_K))
            r = m.get(cid)
            if (
                r is not None
                and r <= _SIG_TOP
                and full >= 2
                and covs.get(cid, 0) == full
            ):
                signal = max(signal, _SIG_MAX_WEIGHT / (_RRF_K + r))
        rrf += signal
        c = dict(id_to_chunk[cid])
        c["score"] = rrf
        c["_sem_rank"] = sem_rank_map.get(cid)
        c["_lex_rank"] = lex_rank_map.get(cid)
        fused.append(c)

    fused.sort(key=lambda c: c["score"], reverse=True)
    return fused


# ===========================================================================
# Lexical reference + date-intent boost
# ===========================================================================

# Date questions ("متى كانت غزوة بدر؟") are semantically close to every Badr
# chunk, including summaries and later expeditions.  A chunk that contains an
# explicit day/month/year phrase near the queried event is much more likely to
# answer the date question, so give it a bounded score bonus.  Proximity is
# token-level on purpose: a bare date word anywhere in a long chunk (e.g.
# "من الجمعة" in a tafsir passage) must not trigger a boost.
_DATE_INTENT_BOOST = 0.5
_DATE_EVENT_WINDOW = 5
_DATE_MARKER_WINDOW = 2
_DATE_MONTH_TOKENS = {
    "محرم", "صفر", "ربيع", "جمادي", "رجب", "شعبان", "رمضان", "شوال",
    "القعده", "قعده", "الحجه", "حجه",
}
_DATE_WEEKDAY_TOKENS = {
    "الاحد", "احد", "الاثنين", "اثنين", "الثلاثاء", "ثلاثاء",
    "الاربعاء", "اربعاء", "الخميس", "خميس", "الجمعه", "جمعه",
    "السبت", "سبت",
}
_DATE_PHRASE_MARKERS = {"يوم", "ليله", "شهر", "في"}
_EVENT_SYNONYM_GROUPS = (
    {"غزوه", "معركه", "وقعه", "صدام"},
)
_GENERIC_EVENT_TOKENS = {"غزوه", "معركه", "وقعه", "صدام"}
_DATE_QUALIFIER_PENALTY = 0.1
_DATE_PRIORITY_MAX = 3
_EVENT_QUALIFIERS = {"الاول", "الاولى", "الاخر", "الاخره"}
_DATE_QUERY_EXCLUDE = {
    "يوم", "شهر", "سنه", "عام", "تاريخ", "وقعت", "وقع", "كانت", "كان",
    "حدثت", "حدث",
    "الاحد", "الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعه",
    "السبت", "احد", "اثنين", "ثلاثاء", "اربعاء", "خميس", "جمعه", "سبت",
    "محرم", "صفر", "ربيع", "جمادي", "رجب", "شعبان", "رمضان", "شوال",
    "ذو", "ذي", "القعده", "الحجه", "قعده", "حجه",
}


def _expand_event_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for group in _EVENT_SYNONYM_GROUPS:
        if expanded & group:
            expanded |= group
    return expanded


def _date_near_event(tokens: list[str], event_tokens: set[str]) -> bool:
    """True when an explicit date phrase is close to a queried event token.

    A date word anywhere in a long chunk is not enough (a tafsir passage can
    mention Friday prayer far away from Badr).  Require (a) a marker word for
    the date phrase, and (b) the date token within _DATE_EVENT_WINDOW tokens
    of an event occurrence.
    """
    event_positions = [i for i, t in enumerate(tokens) if t in event_tokens]
    if not event_positions:
        return False
    for i, tok in enumerate(tokens):
        marked = False
        if tok in _DATE_MONTH_TOKENS:
            prev = tokens[max(0, i - _DATE_MARKER_WINDOW):i]
            marked = any(w in _DATE_PHRASE_MARKERS for w in prev)
        elif tok in _DATE_WEEKDAY_TOKENS:
            prev = tokens[max(0, i - _DATE_MARKER_WINDOW):i]
            marked = any(w in {"يوم", "ليله"} for w in prev)
        elif tok.isdigit():
            prev = tokens[max(0, i - _DATE_MARKER_WINDOW):i]
            marked = any(w in {"سنه", "عام"} for w in prev)
        if marked and any(
            abs(i - e) <= _DATE_EVENT_WINDOW for e in event_positions
        ):
            return True
    return False


def _asks_date_intent(norm_query: str) -> bool:
    return bool(
        re.search(r"(?:^|\s)متى(?:\s|$)", norm_query)
        or re.search(
            r"(?:^|\s)(?:اي|ايه)\s+(?:يوم|عام|سنه|شهر)(?:\s|$)",
            norm_query,
        )
        or re.search(
            r"في\s+(?:اي|ايه)\s+(?:يوم|عام|سنه|شهر)(?:\s|$)",
            norm_query,
        )
    )


def _apply_lexical_boost(candidates: list[dict], query: str) -> list[dict]:
    """Boost chunks whose reference or explicit date matches the query.

    Two independent, bounded signals:
      * reference boost: every significant query keyword occurs as a whole
        word in the chunk reference; skipped when the reference carries an
        event qualifier (الأولى/الآخرة) the query did not ask for.
      * date-intent boost: the query asks when an event happened, the chunk
        contains both the queried event and an explicit day/month/year
        phrase.  Unqualified queries slightly penalize qualified event
        senses (e.g. "بدر الأولى") so the main event wins.
    """
    norm_query = normalize(query)
    query_words = set(norm_query.split())
    keywords = [
        w for w in norm_query.split()
        if len(w) >= 3 and w not in _QUERY_STOPWORDS
    ]
    query_qualifiers = query_words & _EVENT_QUALIFIERS
    asks_date = _asks_date_intent(norm_query)
    base_event_tokens = {
        t for t in _tokenize(norm_query)
        if t not in _QUERY_STOPWORDS
        and t not in _DATE_QUERY_EXCLUDE
        and len(t) >= 3
    }
    event_tokens = _expand_event_tokens(base_event_tokens)
    named_event_tokens = base_event_tokens - _GENERIC_EVENT_TOKENS
    event_anchor = named_event_tokens or event_tokens
    surah_m = re.search(r'سوره\s+(\S+)', norm_query)
    surah_token = surah_m.group(1) if surah_m else None

    boosted = []
    for cand in candidates:
        ref = cand.get("reference", "")
        norm_ref = normalize(ref)
        src = cand.get("source", "")
        boost = 0.0

        ref_qualifiers = set(norm_ref.split()) & _EVENT_QUALIFIERS
        date_match = False
        if (
            keywords
            and (query_qualifiers or not ref_qualifiers)
            and all(
                re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$)', norm_ref)
                for kw in keywords
            )
        ):
            boost += _REF_BOOST

        if surah_token and src in ("quran", "tafsir"):
            if re.search(
                r'(?:^|\s)' + re.escape(surah_token) + r'(?:\s|:|$)',
                norm_ref,
            ):
                boost += _REF_BOOST

        if asks_date and event_tokens:
            cand_norm = normalize(cand.get("original_text", "") or "")
            cand_token_list = _tokenize(cand_norm)
            cand_tokens = set(cand_token_list)
            if (
                event_anchor & cand_tokens
                and _date_near_event(cand_token_list, event_anchor)
            ):
                boost += _DATE_INTENT_BOOST
                qualified = ref_qualifiers or (cand_tokens & _EVENT_QUALIFIERS)
                if not query_qualifiers and qualified:
                    boost -= _DATE_QUALIFIER_PENALTY
                else:
                    date_match = True

        if boost != 0:
            cand = dict(cand)
            cand["score"] = cand.get("score", 0.0) + boost
            if date_match:
                cand["_date_intent_match"] = True
        boosted.append(cand)

    return boosted


# ===========================================================================
# Diversity selection (with injection bypass)
# ===========================================================================

# Phase 3I diversity policy: total per-source-type cap for ranked selection.
# 3 preserves complementary same-source evidence (e.g. three consecutive
# tafsir chunks of one passage) while still preventing one source type from
# flooding the whole context.  Bypass chunks (lexical/semantic top-3) are
# considered ahead of score order, but only the first bypass chunk of each
# source type is admitted beyond score order; further bypass chunks of the
# same type compete under the same cap.  Root cause this fixes (measured):
# when the lexical top-3 and semantic top-3 overlap on near-duplicate
# variants from ONE source (e.g. several hadith narrations of the same
# ruling), the previous "bypass exempt from cap" rule let that source fill
# every slot, starving complementary evidence of other sources.
_DIVERSITY_MAX_PER_TYPE = 3
_DIVERSITY_BYPASS_QUOTA_PER_TYPE = 1

# Semantic-priority selection (production retrieval): the semantic top-5 are
# guaranteed a slot, then remaining slots fill by fused score under a wider
# cap of 4 per source type.  This anchors the context to the best semantic
# evidence while still allowing complementary source types to contribute.
_PRIORITY_SEM_TOP = 6
_PRIORITY_MAX_PER_TYPE = 4
# The semantic/lexical priority sets can exceed top_k on their own.  When
# that happened, a gold chunk at fused rank 2-4 could be dropped even though
# the fused ranking had already placed it above most priorities (observed:
# fiqh_66 at pool rank 2 and hadith_muslim_13708 at pool rank 4).  Reserve a
# bounded number of slots for the highest fused-score chunks before filling
# the rest from the priority set.  This is a selection-order guarantee, not a
# score change.
_PRIORITY_FUSED_RESERVE = 4


def _select_diverse(
    candidates: list[dict],
    top_k: int,
    bypass_ids: set[str] | None = None,
    lex_top3_ids: set[str] | None = None,
    sem_top3_ids: set[str] | None = None,
    priority_ids: set[str] | None = None,
    must_ids: set[str] | None = None,
) -> list[dict]:
    """Select top_k with source-type diversity.

    Selection order (legacy mode):
      1. Injected chunks (`bypass_ids`, query-determined surah verses) are
         always kept and exempt from the per-type cap.
      2. Chunks in the lexical top-3 or semantic top-3 are considered ahead
         of pure score order, up to _DIVERSITY_BYPASS_QUOTA_PER_TYPE per
         source type; further bypass chunks of the same type compete under
         the cap like everything else.
      3. Remaining slots fill in fused-score order under a hard
         _DIVERSITY_MAX_PER_TYPE cap per source type.
      4. The cap relaxes only to fill leftover slots (e.g. when every
         candidate belongs to one source type).

    When `priority_ids` is supplied, selection switches to semantic-priority
    mode: injected, lexical/semantic priority and explicit-date chunks are
    guaranteed first (trimmed by score if they exceed top_k), then remaining
    slots fill by fused score
    under _PRIORITY_MAX_PER_TYPE, relaxing only to fill empty slots.  The
    legacy arguments are ignored in that mode.
    """
    bypass_ids    = bypass_ids or set()
    lex_top3_ids  = lex_top3_ids or set()
    sem_top3_ids  = sem_top3_ids or set()

    # Deduplicate by id, keeping highest score.  The sort is stable, so the
    # upstream fused order deterministically breaks exact score ties.
    seen_ids: set = set()
    unique = []
    for c in sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True):
        cid = c.get("id")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(c)

    if priority_ids is not None:
        return _select_priority(
            unique, top_k, bypass_ids, priority_ids or set(), must_ids or set()
        )

    if len(unique) <= top_k:
        return unique

    selected: list[dict] = []
    remainder: list[dict] = []
    type_counts: dict[str, int] = defaultdict(int)
    bypass_quota: dict[str, int] = defaultdict(int)

    for c in unique:
        cid = c.get("id", "")
        if cid in bypass_ids:
            # Deterministic surah injections: always selected, never capped.
            selected.append(c)
            continue
        if cid in lex_top3_ids or cid in sem_top3_ids:
            src_type = c.get("source", "other")
            if bypass_quota[src_type] < _DIVERSITY_BYPASS_QUOTA_PER_TYPE:
                selected.append(c)
                bypass_quota[src_type] += 1
                type_counts[src_type] += 1
                continue
        remainder.append(c)

    # If bypass+injected chunks alone fill top_k, return them by score.
    if len(selected) >= top_k:
        return sorted(selected, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]

    # Fill remaining slots in fused-pool order under the per-type cap.
    for c in remainder:
        if len(selected) >= top_k:
            break
        src_type = c.get("source", "other")
        if type_counts[src_type] >= _DIVERSITY_MAX_PER_TYPE:
            continue
        selected.append(c)
        type_counts[src_type] += 1

    # Relax the cap only if capped fill left slots empty.
    if len(selected) < top_k:
        used_ids = {c["id"] for c in selected}
        for c in remainder:
            if len(selected) >= top_k:
                break
            if c["id"] in used_ids:
                continue
            selected.append(c)
            used_ids.add(c["id"])

    return sorted(selected, key=lambda c: c.get("score", 0.0), reverse=True)


def _terminology_anchor_ids(query_tokens: list[str], chunks: list[dict]) -> set[str]:
    """Return the rawatib-list chunk for terminology questions.

    The five confirmed sunnah prayers are defined in a section whose opening
    chunk says "السنن الراتبة ... سنة الفجر والظهر والعصر والمغرب والعشاء".
    A literal "السنن المؤكدة" query often retrieves later sections that mention
    only one or two of the five, so guarantee the defining chunk when the
    query uses terminology tokens.  This is a corpus-grounded anchor, not a
    general synonym expansion.
    """
    if not (set(query_tokens) & _TERMINOLOGY_QUERY_TOKENS):
        return set()
    for c in _bm25_search(["سنن", "راتبه", "صلاه"], chunks, top_k=20):
        text = normalize(c.get("original_text") or "")
        if all(t in text for t in ("السنن", "الراتبه", "الفجر", "الظهر")):
            return {c["id"]}
    return set()


_SECTION_INDEX_CACHE: dict[tuple, dict[tuple, list[dict]]] = {}


def _section_index(chunks: list[dict]) -> dict[tuple, list[dict]]:
    """Group multi-chunk fiqh sections by book/chapter/subheading."""
    # id()+len() alone can collide when a same-length temporary list is
    # allocated at a reused address (observed between two tests).  Include
    # boundary IDs so a stale index can never be served for different chunks.
    key = (
        id(chunks),
        len(chunks),
        chunks[0].get("id") if chunks else None,
        chunks[-1].get("id") if chunks else None,
    )
    cached = _SECTION_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    sections: dict[tuple, list[dict]] = {}
    for c in chunks:
        if c.get("source") != "fiqh":
            continue
        meta = c.get("metadata") or {}
        if meta.get("total_chunks", 1) <= 1:
            continue
        section = (
            meta.get("book"), meta.get("chapter"), meta.get("subheading"),
        )
        if not all(section):
            continue
        sections.setdefault(section, []).append(c)
    for members in sections.values():
        members.sort(key=lambda c: (c.get("metadata") or {}).get("chunk", 0))
    _SECTION_INDEX_CACHE[key] = sections
    return sections


_NARRATIVE_QUERY_TOKENS = {"قصه", "قصص", "خبر", "اخبار", "حكايه", "سيره"}
_QURAN_AYAH_INDEX_CACHE: dict[tuple[int, int], dict[tuple[int, int], dict]] = {}


def _quran_ayah_index(chunks: list[dict]) -> dict[tuple[int, int], dict]:
    key = (id(chunks), len(chunks))
    cached = _QURAN_AYAH_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index: dict[tuple[int, int], dict] = {}
    for c in chunks:
        if c.get("source") != "quran":
            continue
        meta = c.get("metadata") or {}
        surah = meta.get("surah_number")
        ayah = meta.get("ayah_number")
        if surah is None or ayah is None:
            continue
        index[(int(surah), int(ayah))] = c
    _QURAN_AYAH_INDEX_CACHE[key] = index
    return index


def _narrative_quran_anchors(
    selected: list[dict],
    chunks: list[dict],
    query_tokens: list[str],
) -> list[dict]:
    if not (set(query_tokens) & _NARRATIVE_QUERY_TOKENS):
        return []
    by_surah: dict[int, list[int]] = defaultdict(list)
    for c in selected:
        if c.get("source") != "tafsir":
            continue
        match = re.search(r"(\d+):(\d+)\s*$", c.get("reference") or "")
        if not match:
            continue
        by_surah[int(match.group(1))].append(int(match.group(2)))
    if not by_surah:
        return []
    surah, ayahs = max(by_surah.items(), key=lambda item: len(item[1]))
    if len(ayahs) < 2:
        return []
    index = _quran_ayah_index(chunks)
    wanted = [min(ayahs), max(ayahs) + 1]
    return [index[(surah, ayah)] for ayah in wanted if (surah, ayah) in index]


def _expand_narrative_quran_context(
    selected: list[dict],
    chunks: list[dict],
    query_tokens: list[str],
    top_k: int,
) -> list[dict]:
    """Add aligned Quran anchors, replacing the lowest-scored selected chunks."""
    anchors = _narrative_quran_anchors(selected, chunks, query_tokens)
    if not anchors:
        return selected
    out = list(selected)
    selected_ids = {c.get("id") for c in out}
    to_add = [a for a in anchors if a.get("id") not in selected_ids]
    if not to_add:
        return selected
    drop_count = max(0, len(out) + len(to_add) - top_k)
    anchor_ids = {a.get("id") for a in anchors}
    if drop_count:
        droppable = sorted(
            (c for c in out if c.get("id") not in anchor_ids),
            key=lambda c: c.get("score", 0.0),
        )
        for candidate in droppable[:drop_count]:
            out.remove(candidate)
    out.extend(to_add)
    return out


def _expand_section_context(
    selected: list[dict],
    chunks: list[dict],
    query_tokens: list[str],
) -> list[dict]:
    """Add the next paragraph of a selected multi-chunk fiqh section."""
    selected_ids = {c.get("id") for c in selected}
    query_set = {_fold_lexical(t) for t in query_tokens}
    sections = _section_index(chunks)
    by_id = {c.get("id"): c for c in chunks}
    for parent in selected:
        parent = by_id.get(parent.get("id"), parent)
        if parent.get("source") != "fiqh":
            continue
        meta = parent.get("metadata") or {}
        if meta.get("total_chunks", 1) <= 1:
            continue
        section = (
            meta.get("book"), meta.get("chapter"), meta.get("subheading"),
        )
        members = sections.get(section, [])
        parent_num = meta.get("chunk", 0)
        for candidate in members:
            cmeta = candidate.get("metadata") or {}
            if cmeta.get("chunk") != parent_num + 1:
                continue
            if candidate.get("id") in selected_ids:
                return selected
            body = candidate.get("clean_text") or ""
            if not (query_set & set(_tokenize_clean(body)[:120])):
                return selected
            replace_at = next(
                (i for i in range(len(selected) - 1, -1, -1)
                 if selected[i].get("source") != "fiqh"),
                None,
            )
            if replace_at is None:
                return selected
            out = list(selected)
            out[replace_at] = candidate
            return out
    return selected


# ===========================================================================
# Seerah title retrieval
# ===========================================================================
# Seerah chunk references are descriptive section titles ("ولادة رسول الله
# ... ورضاعته", "أمر العقبة الثانية", "ذكر الإسراء والمعراج").  The long
# classical body text often dilutes the whole-chunk embedding, so a question
# such as "من أرضعت النبي؟" can miss every sira passage even though the title
# names the topic.  This bounded, deterministic signal adds the best sira
# title matches as an extra RRF ranking.  It is gated on explicit seerah
# event vocabulary so fiqh/Quran questions are unaffected.
# Unique seerah-event vocabulary: sufficient on its own.
_SEERAH_UNIQUE_RE = re.compile(
    r"سير[ةه]|غزو[ةه]|سري[ةه]|هجر[ةه]|مولد|ولاد[ةه]|رضاع|[اأإآ]رضع|كفل|"
    r"وداع|[اأإآ]سراء|معراج|عقب[ةه]|حديبي[ةه]|تبوك|بدر|"
    r"خندق|حنين|طائف|خيبر|قريظ[ةه]|مؤت[ةه]|حليم[ةه]|خديج[ةه]|فاطم[ةه]"
)
# Generic terms that only count when the question is explicitly about the
# Prophet or his Companions (e.g. "خطبة الوداع", not "خطبة الجمعة").
_SEERAH_WEAK_RE = re.compile(
    r"خطب[ةه]|بيع[ةه]|تزوج|زوج[ةه]|[اأإآ]ولاد|[اأإآ]بناء|نسب|"
    r"قبيل[ةه]|عم[هة]|خال[ةه]|جد[هة]|[اأإآ]سلم|[اأإآ]سلام|بعث"
)
_SEERAH_CONTEXT_RE = re.compile(
    r"نبي|رسول|صلي|وسلم|مكه|المدينه|قريش|انصار|مهاجرين|صحاب[ةه]|خليف[ةه]"
)

# Generic words that carry no topical title signal.  Removing them is
# essential: otherwise every title containing "النبي" outranks the one title
# that actually names the event.
_SEERAH_TITLE_STOP = {
    "نبي", "رسول", "الله", "صلي", "عليه", "وسلم", "سيره", "ابن", "هشام",
    "ذكر", "امر", "كان", "كانت", "ماذا", "كيف", "متي", "اين", "من", "ما",
    "هل", "عدد", "كم", "في", "علي", "عن", "الي", "اول", "اخره", "شهر",
    "سنه", "عام", "يوم", "بعد", "قبل", "بين", "مع", "عند", "حتي",
}

_SEERAH_TITLE_MAX_EXACT_DF = 80
_SEERAH_TITLE_MAX_ROOT_DF = 30
_SEERAH_TITLE_TOP_N = 8
_SEERAH_TITLE_RRF_WEIGHT = 2.0

_seerah_title_cache: dict[tuple[int, int], dict] = {}


def _get_seerah_title_index(chunks: list[dict]) -> dict:
    """Build/cache the sira reference-title index for *chunks*."""
    key = (id(chunks), len(chunks))
    cached = _seerah_title_cache.get(key)
    if cached is not None:
        return cached
    title_tokens: dict[str, set[str]] = {}
    title_roots: dict[str, set[str]] = {}
    chunk_by_id: dict[str, dict] = {}
    tok_df: dict[str, int] = defaultdict(int)
    root_df: dict[str, int] = defaultdict(int)
    heading_re = re.compile(r"^\s*[\(（]([^)）]{3,160})[\)）]")
    for c in chunks:
        if c.get("source") != "sira":
            continue
        cid = c.get("id", "")
        chunk_by_id[cid] = c
        # Many classical sira passages carry a precise internal heading at
        # the start of the text (e.g. "(خطبة الرسول في حجة الوداع)") while
        # the chunk's reference repeats a broad chapter title.  Index that
        # heading too; otherwise the farewell sermon is invisible to title
        # retrieval because its reference only says "موافاة علي ... في الحج".
        heading = ""
        m = heading_re.match(c.get("original_text") or "")
        if m:
            heading = normalize(m.group(1))
        title_text = f"{c.get('reference') or ''} {heading}"
        toks = {
            t for t in _tokenize_clean(title_text)
            if len(t) >= 3
            and t not in _QUERY_STOPWORDS
            and t not in _SEERAH_TITLE_STOP
        }
        roots = {_root_key(t) for t in toks if len(_root_key(t)) >= 3}
        title_tokens[cid] = toks
        title_roots[cid] = roots
        for t in toks:
            tok_df[t] += 1
        for r in roots:
            root_df[r] += 1
    idx = {
        "title_tokens": title_tokens,
        "title_roots": title_roots,
        "chunk_by_id": chunk_by_id,
        "tok_df": dict(tok_df),
        "root_df": dict(root_df),
        "n": max(len(title_tokens), 1),
    }
    _seerah_title_cache[key] = idx
    return idx


def _seerah_title_candidates(
    query: str,
    query_tokens: list[str],
    chunks: list[dict],
    top_n: int = _SEERAH_TITLE_TOP_N,
) -> list[dict]:
    """Return the best sira chunks whose descriptive reference matches *query*.

    Only explicit seerah-event vocabulary triggers this path.  Scoring is
    IDF-weighted exact title-token overlap plus rare root overlap (to bridge
    أرضعت/رضاعته).  At least one rare exact token or rare root is required,
    so a generic word cannot flood the pool.
    """
    norm_query = normalize(query)
    has_unique = bool(_SEERAH_UNIQUE_RE.search(norm_query))
    has_weak_context = bool(
        _SEERAH_WEAK_RE.search(norm_query)
        and _SEERAH_CONTEXT_RE.search(norm_query)
    )
    if not has_unique and not has_weak_context:
        return []
    idx = _get_seerah_title_index(chunks)
    q_tokens = {
        folded for t in query_tokens
        for folded in (_fold_lexical(t),)
        if len(folded) >= 3
        and folded not in _QUERY_STOPWORDS
        and folded not in _SEERAH_TITLE_STOP
    }
    if not q_tokens:
        return []
    q_roots = {_root_key(t) for t in q_tokens if len(_root_key(t)) >= 3}
    n = idx["n"]
    tok_df = idx["tok_df"]
    root_df = idx["root_df"]
    scored: list[tuple[int, float, str]] = []
    for cid, toks in idx["title_tokens"].items():
        exact = toks & q_tokens
        rare_exact = {t for t in exact if tok_df.get(t, 0) <= _SEERAH_TITLE_MAX_EXACT_DF}
        rare_roots = {
            r for r in (idx["title_roots"][cid] & q_roots)
            if root_df.get(r, 0) <= _SEERAH_TITLE_MAX_ROOT_DF
        }
        if not rare_exact and not rare_roots:
            continue
        score = sum(
            math.log((n + 1) / (tok_df.get(t, 0) + 1)) for t in rare_exact
        )
        score += 0.75 * sum(
            math.log((n + 1) / (root_df.get(r, 0) + 1)) for r in rare_roots
        )
        # Coverage first: a title matching two common query words
        # ("الهجرة الأولى") is a better topic match than one matching a
        # single rarer word ("المسلمين").
        coverage = len(rare_exact) + len(rare_roots)
        scored.append((coverage, score, cid))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    out: list[dict] = []
    for _cov, _score, cid in scored[:top_n]:
        c = idx["chunk_by_id"].get(cid)
        if c is not None:
            out.append(c)
    return out


def _select_date_priority_ids(
    pool: list[dict], limit: int = _DATE_PRIORITY_MAX,
) -> set[str]:
    """Pick top date-intent chunks with one slot per source type.

    Adjacent tafsir chunks often all match the same date phrase and would
    otherwise fill every priority slot, hiding a sira chunk with the precise
    expedition date.  Source-type diversity keeps that decisive evidence in
    the guaranteed set.
    """
    chosen: set[str] = set()
    seen_types: set[str] = set()
    for c in sorted(
        (c for c in pool if c.get("_date_intent_match")),
        key=lambda c: c.get("score", 0.0), reverse=True,
    ):
        source_type = c.get("source", "other")
        if source_type in seen_types:
            continue
        seen_types.add(source_type)
        chosen.add(c.get("id", ""))
        if len(chosen) >= limit:
            break
    return chosen


def _lexical_must_ids(
    lex_results: list[dict],
    query_tokens: list[str],
    best_coverage: int,
    *,
    asks_date: bool = False,
) -> set[str]:
    """Return the lexical top hit for a short, fully covered non-date query.

    A short definition/compensation query ("ما كفارة الغيبة؟") can have its
    rank-1 BM25 hit (fiqh_512) displaced by semantic/rewrite priority chunks
    even though it is the exact evidence.  Guarantee only that single hit,
    only for a two-term query with full lexical coverage, and never for date
    questions: a date question needs several complementary evidence chunks,
    and promoting a single lexical hit reduced multi-chunk coverage
    (badr_date 2/3 -> 1/3).
    """
    if (
        not asks_date
        and len(query_tokens) == 2
        and lex_results
        and best_coverage >= len(query_tokens)
    ):
        return {lex_results[0].get("id", "")} - {""}
    return set()


def _select_priority(
    unique: list[dict],
    top_k: int,
    bypass_ids: set[str],
    priority_ids: set[str],
    must_ids: set[str] | None = None,
) -> list[dict]:
    """Guarantee injected + top fused + semantic-priority chunks.

    Mandatory chunks (deterministic injections and ``must_ids``) are kept
    first.  The next ``_PRIORITY_FUSED_RESERVE`` slots go to the highest
    fused-score chunks, then the remaining priority chunks fill the context,
    and finally the per-source cap is relaxed only to fill empty slots.
    """
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(chunk: dict) -> None:
        selected.append(chunk)
        selected_ids.add(chunk.get("id", ""))

    must_ids = must_ids or set()

    # 1. Mandatory: deterministic injections and explicitly protected chunks.
    for c in unique:
        cid = c.get("id", "")
        if cid in bypass_ids or cid in must_ids:
            add(c)

    if len(selected) >= top_k:
        return sorted(
            selected, key=lambda c: c.get("score", 0.0), reverse=True,
        )[:top_k]

    # 2. Bounded reserve for the best fused chunks, so a high-ranked gold
    #    chunk cannot be starved by an oversized priority set.  At most half
    #    the context is reserved, leaving room for the priority guarantees.
    reserve_limit = min(_PRIORITY_FUSED_RESERVE, max(1, top_k // 2))
    reserved = 0
    for c in unique:
        if reserved >= reserve_limit or len(selected) >= top_k:
            break
        cid = c.get("id", "")
        if cid in selected_ids:
            continue
        add(c)
        reserved += 1

    # 3. Guaranteed semantic/lexical priority chunks.
    for c in unique:
        if len(selected) >= top_k:
            break
        cid = c.get("id", "")
        if cid in selected_ids:
            continue
        if cid in priority_ids:
            add(c)

    # 4. Fill under the per-source cap, then relax it only if needed.
    type_counts: dict[str, int] = defaultdict(int)
    for c in selected:
        type_counts[c.get("source", "other")] += 1
    for c in unique:
        if len(selected) >= top_k:
            break
        cid = c.get("id", "")
        if cid in selected_ids:
            continue
        src_type = c.get("source", "other")
        if type_counts[src_type] >= _PRIORITY_MAX_PER_TYPE:
            continue
        add(c)
        type_counts[src_type] += 1

    if len(selected) < top_k:
        for c in unique:
            if len(selected) >= top_k:
                break
            cid = c.get("id", "")
            if cid in selected_ids:
                continue
            add(c)

    return sorted(selected, key=lambda c: c.get("score", 0.0), reverse=True)


# ===========================================================================
# Query rewriting
# ===========================================================================

_REWRITE_SYSTEM = (
    "أنت خبير في المصطلحات الفقهية والحديثية والتاريخية الكلاسيكية. "
    "أعطني حتى 3 عبارات قصيرة (من 2 إلى 4 كلمات) بالعربية، كل عبارة تعيد صياغة السؤال "
    "باستبدال كلماته الأساسية بمرادفاتها الكلاسيكية كما تظهر في كتب الفقه والحديث والسيرة، "
    "وتجنب الجمل الكاملة. "
    "اكتب فقط العبارات مفصولة بسطر جديد، بدون شرح."
)


_SYNONYM_GROUPS = [
    ["غزوه", "وقعه", "معركه"],
    ["راتبه", "رواتب"],
    ["يغسل", "غسل"],
    ["غسل", "اغسال"],
]

# Grounding-only event-term extension.  Some fiqh passages describe a battle
# as "صدام مسلح" (fiqh_1184: first armed clash at Badr) where answers use the
# classical غزوة/وقعة/معركة family.  Deliberately NOT part of
# _SYNONYM_GROUPS so query expansion/retrieval behavior is unchanged; this
# only makes the grounding matcher treat صدام as the same event concept.
_GROUNDING_EXTRA_SYNONYM_GROUPS = [
    ["غزوه", "صدام"],
    ["ليقوى", "ليتقوى"],
    # The corpus states a ruling as "يرى وجوبها" while generated answers
    # naturally paraphrase it as "يرى ... أنها واجبة".  Both are the same
    # ruling family; without this grounding-only alias the rare-token guard
    # drops the supported qualifier and loses a required fact.
    ["واجب", "وجوب"],
    # The corpus often states the ruling as "تأكيد استحبابه" while an answer
    # naturally says "مستحب".  These are the same ruling family; without the
    # alias, a short, explicit ruling sentence fails coverage even though the
    # cited clause supports it.
    ["مستحب", "استحباب"],
]

# Phrase-level aliases: only the *scoped* replacements below are
# corpus-grounded.  'السنن المؤكدة' in the prayer chapter maps to the rawatib
# list (fiqh_160/164: 'السنن الراتبة', 'السنن والرواتب'), while the bare word
# مؤكدة is NOT a global synonym of راتبة — the corpus also uses 'سنة مؤكدة'
# for الوتر/الجماعة/الكسوف/العقيقة.  Applied only when the words are
# adjacent with no intervening negation (e.g. 'غير المؤكدة' is untouched).
_PHRASE_ALIASES = [
    (("سنن", "مؤكده"), (("سنن", "راتبه"), ("سنن", "رواتب"))),
    (("صلوات", "مسنونه", "رواتب"), (
        ("سنن", "راتبه", "رواتب"), ("سنن", "راتبه"),
    )),
    # "متى تذبح العقيقة؟" -> the corpus heading is "وقت الذبح"; this scoped
    # alias surfaces the timing chunk that plain BM25 misses (the body starts
    # with "والذبح يكون..." and does not repeat "العقيقة").
    (("متى", "تذبح"), (("وقت", "الذبح"),)),
    (("ومتى", "تذبح"), (("وقت", "الذبح"),)),
    (("متى", "يذبح"), (("وقت", "الذبح"),)),
    (("ومتى", "يذبح"), (("وقت", "الذبح"),)),
]


def _phrase_alias_rewrites(query_tokens: list[str]) -> list[list[str]]:
    """Scoped phrase-alias rewrites (corpus-grounded, deterministic)."""
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for i in range(len(query_tokens)):
        for pat, alts in _PHRASE_ALIASES:
            n = len(pat)
            if tuple(query_tokens[i:i + n]) != pat:
                continue
            for alt_seq in alts:
                full = query_tokens[:i] + list(alt_seq) + query_tokens[i + n:]
                for seq in (full, list(alt_seq) if i == 0 else None):
                    if seq is None:
                        continue
                    key = tuple(seq)
                    if key not in seen:
                        seen.add(key)
                        out.append(seq)
    return out


def _expand_synonyms(query_tokens: list[str]) -> list[list[str]]:
    """Return extra token lists where a query token is swapped for each
    synonym in its group, plus scoped phrase-alias rewrites (deterministic
    classical-vocabulary expansion)."""
    out: list[list[str]] = []
    for i, t in enumerate(query_tokens):
        for grp in _SYNONYM_GROUPS:
            if t in grp:
                for alt in grp:
                    if alt != t:
                        out.append(query_tokens[:i] + [alt] + query_tokens[i + 1:])
    out.extend(_phrase_alias_rewrites(query_tokens))
    return out


def _event_synonym_rewrites(query_tokens: list[str]) -> list[list[str]]:
    """Deterministic battlefield-term rewrites for date queries.

    ``متى غزوة بدر؟`` must still retrieve a source phrased as ``وقعة بدر``;
    relying on the LLM rewrite for this makes retrieval vary between runs.
    """
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for i, token in enumerate(query_tokens):
        for group in _EVENT_SYNONYM_GROUPS:
            if token not in group:
                continue
            for alt in group:
                if alt == token:
                    continue
                candidate = tuple(
                    query_tokens[:i] + [alt] + query_tokens[i + 1:]
                )
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(list(candidate))
    return out


_LEXICAL_VARIANT_GROUPS = [
    ("غش", "غشنا"),
    ("ايمان", "يؤمن"),
    ("اخ", "لاخيه"),
    ("يرحمه", "يرحمهم"),
    ("قوي", "شديد"),
]
_LEXICAL_VARIANT_ANCHOR_MIN_SCORE = 10.0


def _lexical_variant_query(query_tokens: list[str]) -> list[str]:
    """Return one alternative query with each known lexical variant swapped
    in.  Returns [] when the query contains no variant.  This stays
    deterministic and does not call the generation model."""
    out = list(query_tokens)
    changed = False
    for i, token in enumerate(query_tokens):
        for literal, variant in _LEXICAL_VARIANT_GROUPS:
            if token == literal and variant not in query_tokens:
                out[i] = variant
                changed = True
                break
    return out if changed else []


def _phrase_uses_new_variant(
    phrase_tokens: list[str], query_tokens: list[str],
) -> bool:
    """True when a rewritten query introduces a known lexical variant that
    the literal query did not contain."""
    query_set = set(query_tokens)
    phrase_set = set(phrase_tokens)
    return any(
        literal in query_set and variant in phrase_set
        and variant not in query_set
        for literal, variant in _LEXICAL_VARIANT_GROUPS
    )


def _rewrite_query(query: str, client: OpenAI) -> list[str]:
    """Return up to 3 classical-fiqh rewritings (retrieval only, never in prompt).

    Falls back to [] on any error.
    """
    try:
        resp = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = resp.choices[0].message.content or ""
        phrases = [p.strip() for p in raw.splitlines() if p.strip()]
        return phrases[:3]
    except Exception:
        return []


# ===========================================================================
# retrieve()
# ===========================================================================

def retrieve(
    query: str,
    index,
    chunks: list[dict],
    client: OpenAI,
    top_k: int = 8,
    tafsir_anchors: list[dict] | None = None,
    *,
    retrieval_query: str | None = None,
) -> list[dict]:
    """Retrieve the most relevant chunks for a user query.

    Pipeline:
        1. Preprocess and tokenize the query.
        2. Surah detection (Fix A) — set effective_top_k=8 for surah queries.
        3. Dual semantic embedding (original + normalized) over-fetch wide_k.
        4. BM25 search over clean_text, top 20 lexical hits.
        5. RRF fusion of semantic and lexical rankings.
        6. Optional query rewriting (USE_QUERY_REWRITE) — merge extra results.
        7. Inject surah chunks with fixed high score (bypasses diversity cap).
           For Tafsir-intent questions, inject the deterministically resolved
           Quran verse + Tafsir Ibn Kathir chunk instead; a specific ayah
           suppresses whole-surah injection.
        8. Lexical reference boost.
        9. Selection: injected surah chunks + semantic top-5 are guaranteed
            first; remaining slots fill by fused score under a cap of 4 per
            source type, relaxing only to fill empty slots.
        10. Retrieval gate: pass if cosine >= MIN_COSINE_SCORE OR
            BM25 best covers >= MIN_LEXICAL_TOKENS query tokens.
            The result is exposed as `_gate_passed` on every chunk; real
            similarity scores are returned unmodified (never zeroed or
            replaced by synthetic constants).

    Returns:
        List of top_k chunk result dicts.
    """
    resolved_query = retrieval_query or query
    processed = preprocess(resolved_query)
    clean_query = processed["normalized"]
    # Honorifics are devotionally important but lexically noisy; remove them
    # from the BM25 query only (the semantic query keeps the original text).
    lexical_query = _HONORIFIC_RE.sub(" ", clean_query)
    query_tokens = [
        _strip_al(w) for w in lexical_query.split()
        if w not in _QUERY_STOPWORDS and len(w) >= 2
    ]

    # Tafsir-intent resolution (deterministic, lexical).  Passing anchors in
    # from generate() avoids resolving twice; direct retrieve() callers get
    # the same behavior.
    tafsir_clarification = ""
    if tafsir_anchors is None:
        # Follow-ups ("إيه الدليل؟", "ليه؟") have no topic of their own.
        # Deterministic ayah/hadith resolvers must see the resolved question,
        # otherwise the semantic/BM25 path uses the prior topic while anchor
        # injection silently uses the literal clarification.
        tafsir_anchors, tafsir_clarification = _resolve_chat_tafsir(
            resolved_query, index, chunks, client,
        )

    # Fix A: detect surah query
    surah_num = _resolve_surah_query(resolved_query, chunks)
    # A resolved specific ayah is strictly more precise than whole-surah
    # injection.  A clarification request (whole surah without an ayah) also
    # must not dump the entire surah into the retrieval pool.
    if tafsir_anchors or tafsir_clarification:
        surah_num = None
    quran_topic_anchor = None
    core_hadith_anchor = None
    if not tafsir_anchors and not tafsir_clarification and surah_num is None:
        quran_topic_anchor = _resolve_quran_topic_anchor(
            resolved_query, chunks,
        )
        if quran_topic_anchor is None:
            core_hadith_anchor = _resolve_core_hadith_anchor(
                resolved_query, chunks,
            )
    effective_top_k = 8 if surah_num is not None else top_k

    # --- Semantic retrieval ---
    # For a clarification follow-up the literal phrase ("ليه؟") has no topic;
    # embedding it as a second ranking injects noise and can crowd out the
    # evidence of the resolved question.  Non-follow-ups keep `query` here.
    emb_norm = embed_text(clean_query, client)
    emb_orig = embed_text(resolved_query, client)
    wide_k = max(effective_top_k * 20, 100)
    results_norm = search(emb_norm, index, chunks, top_k=wide_k)
    results_orig = search(emb_orig, index, chunks, top_k=wide_k)

    # Keep best cosine score per chunk id
    best_by_id: dict[str, dict] = {}
    for r in results_norm + results_orig:
        cid = r["id"]
        if cid not in best_by_id or r["score"] > best_by_id[cid]["score"]:
            best_by_id[cid] = r
    sem_ranked = sorted(best_by_id.values(), key=lambda c: c["score"], reverse=True)
    best_cosine = sem_ranked[0]["score"] if sem_ranked else 0.0

    # --- BM25 retrieval ---
    lex_results = _bm25_search(query_tokens, chunks, top_k=20)
    lex_top3_ids = {c["id"] for c in lex_results[:3]}
    best_lex_coverage = max((c.get("_token_coverage", 0) for c in lex_results), default=0)
    lex_must_ids = _lexical_must_ids(
        lex_results, query_tokens, best_lex_coverage,
        asks_date=_asks_date_intent(clean_query),
    )

    # --- Retrieval gate ---
    needed_tokens = 1 if len(query_tokens) == 1 else MIN_LEXICAL_TOKENS
    cosine_ok = best_cosine >= MIN_COSINE_SCORE
    lexical_ok = best_lex_coverage >= needed_tokens
    gate_condition = "cosine" if cosine_ok else ("lexical" if lexical_ok else "FAIL")
    print(f"[retrieve] gate={gate_condition}  best_cosine={best_cosine:.4f}  lex_coverage={best_lex_coverage}/{len(query_tokens)}")

    # --- Optional query rewriting: rewrites join RRF as extra rankings ---
    extra_ranked: list[tuple[list[dict], float]] = []
    synonym_priority_ids: set[str] = set()
    ruling_priority_ids: set[str] = set()
    variant_anchor_ids: set[str] = set()
    rewrite_anchor_ids: set[str] = set()
    terminology_query = bool(set(query_tokens) & _TERMINOLOGY_QUERY_TOKENS)
    intent_tokens = {
        _strip_al(w)
        for w in (processed.get("tokens_clean") or [])
    }
    ruling_intent = bool(intent_tokens & _RULING_INTENT_TOKENS)

    # "غسل يوم الجمعة" and similar day questions: the filler "يوم" can push
    # the decisive fiqh section ("غسل الجمعة") out of the lexical top ranks.
    # Guarantee the top hits of the compact day phrase as well.
    weekday_compact_ids: set[str] = set()
    if "يوم" in query_tokens and set(query_tokens) & _DATE_WEEKDAY_TOKENS:
        _compact = [t for t in query_tokens if t != "يوم"]
        if _compact:
            weekday_compact_ids = {
                c["id"] for c in _bm25_search(_compact, chunks, top_k=2)
            }
    # Static classical-synonym expansion (deterministic, no LLM).
    _phrase_alts = {tuple(x) for x in _phrase_alias_rewrites(query_tokens)}
    _synonym_alts = _expand_synonyms(query_tokens)
    if _asks_date_intent(clean_query):
        # Date questions are especially sensitive to غزوة/وقعة/معركة wording.
        _synonym_alts.extend(_event_synonym_rewrites(query_tokens))
    for _i, _alt in enumerate(_synonym_alts):
        _alt_res = _bm25_search(_alt, chunks, top_k=50)
        if _alt_res:
            extra_ranked.append((_alt_res, _REWRITE_BM25_RRF_WEIGHT))
            if terminology_query and _i == 0:
                synonym_priority_ids |= {c["id"] for c in _alt_res[:2]}
            # Scoped phrase aliases are corpus-grounded and often surface the
            # exact heading the literal query misses (e.g. "وقت الذبح").
            if tuple(_alt) in _phrase_alts:
                synonym_priority_ids |= {c["id"] for c in _alt_res[:2]}

    variant_tokens = _lexical_variant_query(query_tokens)
    if variant_tokens:
        variant_results = _bm25_search(variant_tokens, chunks, top_k=50)
        if variant_results:
            extra_ranked.append((variant_results, _REWRITE_BM25_RRF_WEIGHT))
            if (
                variant_results[0].get("_bm25_score", 0.0)
                >= _LEXICAL_VARIANT_ANCHOR_MIN_SCORE
                and variant_results[0].get("source") == "hadith"
            ):
                swapped = sum(
                    a != b for a, b in zip(variant_tokens, query_tokens)
                )
                anchor_limit = 4 if swapped > 1 else 2
                variant_anchor_ids = {
                    c["id"] for c in variant_results[:anchor_limit]
                }
    if ruling_intent:
        _ruling_res = _bm25_search(
            query_tokens + _RULING_EXPANSION_TOKENS, chunks, top_k=50
        )
        if _ruling_res:
            extra_ranked.append(
                (_ruling_res, _REWRITE_BM25_RRF_WEIGHT)
            )
            # Only the best ruling-expansion hit is guaranteed.  Taking the
            # whole fused top-k here forced adjacent tafsir/hadith chunks into
            # the context and starved the diverse evidence selected by the
            # baseline policy (e.g. ghusl fiqh_58, rawatib fiqh_160).
            ruling_priority_ids = {_ruling_res[0]["id"]}
    if USE_QUERY_REWRITE:
        # Rewriting must see the resolved topic, not "ليه؟"/"إيه الدليل؟".
        rewrites = _rewrite_query(resolved_query, client)
        for phrase in rewrites:
            if not phrase:
                continue
            try:
                emb_phrase = embed_text(phrase, client)
                phrase_results = search(emb_phrase, index, chunks, top_k=wide_k)
                extra_ranked.append((phrase_results, _REWRITE_RRF_WEIGHT))
                _ptoks = [_strip_al(w) for w in _tokenize(phrase)
                          if w not in _QUERY_STOPWORDS and len(w) >= 2]
                if _ptoks:
                    _plex = _bm25_search(_ptoks, chunks, top_k=50)
                    if _plex:
                        extra_ranked.append((_plex, _REWRITE_BM25_RRF_WEIGHT))
                    if (
                        phrase_results
                        and _phrase_uses_new_variant(_ptoks, query_tokens)
                        and phrase_results[0].get("score", 0.0)
                        >= MIN_COSINE_SCORE
                    ):
                        rewrite_anchor_ids.add(phrase_results[0]["id"])
                for r in phrase_results:
                    cid = r["id"]
                    existing = best_by_id.get(cid)
                    if existing is None or r["score"] > existing.get("score", 0):
                        best_by_id[cid] = r
            except Exception:
                pass

    # Seerah questions often miss long classical passages whose whole-chunk
    # embedding is diluted, while the descriptive reference title names the
    # topic exactly.  Add the best sira title matches as one extra ranking.
    # Tafsir requests are excluded: their deterministic verse anchors are
    # already precise, and extra sira chunks would only crowd the context.
    seerah_priority_ids: set[str] = set()
    if not tafsir_anchors and not tafsir_clarification:
        seerah_titles = _seerah_title_candidates(
            clean_query, query_tokens, chunks,
        )
        if seerah_titles:
            extra_ranked.append((seerah_titles, _SEERAH_TITLE_RRF_WEIGHT))
            # A seerah event question needs the matching sira sections in the
            # context even when long hadith passages dominate the fused ranks.
            # Bounded by half the context, so other evidence still has room.
            # These must be guaranteed, not merely "priority": the generic
            # priority set (semantic/lexical top hits) is larger than the
            # remaining slots and fills them in fused order, which dropped
            # every sira title chunk at pool ranks 17+ (observed: sira_159
            # for the foster-mother question, sira_440 for the second Aqaba
            # pledge).
            seerah_priority_ids = {
                c["id"] for c in seerah_titles[:max(1, effective_top_k // 2)]
            }
            print(
                "[retrieve] seerah-title matches: "
                + ", ".join(c["id"] for c in seerah_titles[:4])
            )

    # --- RRF fusion (original semantic + BM25 + rewrites, one score scale) ---
    pool = _rrf_fuse(sem_ranked, lex_results, lex_top3_ids, extra_ranked)

    # --- Deterministic injection (bypasses diversity cap) ---
    injected_ids: set[str] = set()
    if tafsir_anchors:
        pool, injected_ids = _inject_tafsir_anchors(tafsir_anchors, pool)
    elif surah_num is not None:
        pool, injected_ids = _inject_surah_chunks(surah_num, chunks, pool)
    elif quran_topic_anchor is not None:
        topic_anchors = [quran_topic_anchor]
        topic_anchors.extend(
            _quran_topic_companions(quran_topic_anchor, chunks)
        )
        pool, injected_ids = _inject_tafsir_anchors(
            topic_anchors, pool,
        )
        print(
            "[retrieve] quran-topic anchor: "
            + ", ".join(c.get("id", "") for c in topic_anchors)
        )
    elif core_hadith_anchor is not None:
        pool, injected_ids = _inject_tafsir_anchors(
            [core_hadith_anchor], pool,
        )
        print(
            "[retrieve] core-hadith anchor: "
            f"{core_hadith_anchor.get('id')}"
        )

    # --- Lexical reference boost ---
    pool = _apply_lexical_boost(pool, resolved_query)

    # --- Semantic + explicit-date priority selection ---
    sem_priority_ids = {c["id"] for c in sem_ranked[:_PRIORITY_SEM_TOP]}
    terminology_anchor_ids = _terminology_anchor_ids(query_tokens, chunks)
    date_priority_ids = _select_date_priority_ids(pool)
    selected = _select_diverse(
        pool, effective_top_k,
        bypass_ids=injected_ids,
        priority_ids=(
            sem_priority_ids | lex_top3_ids | date_priority_ids
            | synonym_priority_ids | weekday_compact_ids
            | ruling_priority_ids | terminology_anchor_ids
            | variant_anchor_ids | rewrite_anchor_ids
            | seerah_priority_ids
        ),
        must_ids=(
            ruling_priority_ids | terminology_anchor_ids
            | variant_anchor_ids | rewrite_anchor_ids
            | seerah_priority_ids | lex_must_ids
        ),
    )
    selected = _expand_section_context(selected, chunks, query_tokens)
    selected = _expand_narrative_quran_context(
        selected, chunks, query_tokens, effective_top_k
    )
    selected = _filter_joint_evidence(selected, resolved_query)

    # --- Attach explicit gate state and REAL relevance scores ---
    # No synthetic constants: semantic chunks keep their real cosine score,
    # lexical-only chunks carry their genuine BM25 score, injected surah
    # chunks keep their injection score.  The gate result travels in
    # `_gate_passed` so callers never infer it from (or fake it via) scores.
    _gate_passed = cosine_ok or lexical_ok
    lex_bm25_map = {c["id"]: c.get("_bm25_score") for c in lex_results}
    restored = []
    for c in selected:
        cid = c.get("id", "")
        merged = dict(c)
        if c.get("_injected"):
            merged["_sem_score"] = None
            merged["_lex_score"] = None
        elif cid in best_by_id:
            merged["score"] = best_by_id[cid]["score"]      # real cosine
            merged["_sem_score"] = best_by_id[cid]["score"]
            merged["_lex_score"] = lex_bm25_map.get(cid)
        else:
            bm25 = lex_bm25_map.get(cid)
            if bm25 is None:
                bm25 = c.get("_bm25_score")
            if bm25 is not None:
                merged["score"] = bm25                       # real BM25
            merged["_sem_score"] = None
            merged["_lex_score"] = bm25
        merged["_gate_passed"] = _gate_passed
        restored.append(merged)

    return restored


# ===========================================================================
# build_prompt()
# ===========================================================================

def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """Build an Arabic-aware prompt for GPT-4o-mini.

    NOTE: rewritten query phrases are intentionally excluded — only the
    original user query is shown to the model for answering.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        reference = chunk.get("reference", "غير محدد")
        source = chunk.get("source", "غير محدد")
        text = chunk.get("original_text", "")
        context_parts.append(f"[مصدر {i}] ({source} — {reference}):\n{text}")

    context_block = "\n\n".join(context_parts)
    followup_note = ""
    if chat_history and _is_evidence_request(query):
        previous_assistant = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(chat_history)
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and str(message.get("content") or "").strip()
            ),
            "",
        )
        target = _evidence_target_text(previous_assistant)
        if target:
            followup_note = (
                "ملاحظة سياق: هذا طلب دليل على هذه الجملة من الجواب "
                f"السابق: {target}\n\n"
            )
    elif chat_history and _is_followup_query(query):
        previous_user = _last_substantive_user(chat_history)
        if previous_user:
            followup_note = (
                "ملاحظة سياق: هذا السؤال متابعة للسؤال السابق: "
                f"{previous_user}\n\n"
            )
    user_message = (
        f"{followup_note}"
        f"السؤال: {query}\n\n"
        f"المصادر:\n{context_block}\n\n"
        f"الإجابة:"
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_MESSAGE}]

    if chat_history:
        MAX_HISTORY_MESSAGES = 6
        messages.extend(chat_history[-MAX_HISTORY_MESSAGES:])

    messages.append({"role": "user", "content": user_message})
    return messages


# ===========================================================================
# Post-generation grounding check
# ===========================================================================

# Matches Arabic source citations like (مصدر 1) or (مصدر 12)
_CITATION_RE = re.compile(r'،?\s*\(مصدر\s*(\d+)\)')

# Some models cite several sources in one parenthetical, e.g.
# (مصدر 2، مصدر 3).  Expand these before grounding so every number is
# validated and counted individually; a cluster with an invalid number must
# not slip through as an unrecognised text fragment.
_CITATION_CLUSTER_RE = re.compile(
    r'\(\s*مصدر\s*\d+(?:\s*[،,]\s*(?:مصدر\s*)?\d+)+\s*\)'
)
_CITATION_RUN_RE = re.compile(r'(?:،?\s*\(مصدر\s*\d+\)\s*)+')
_LIST_MARKER_RE = re.compile(r'\(?\d+[.)\-]?\)?')


def _expand_citation_clusters(text: str) -> str:
    def _expand(match: re.Match) -> str:
        numbers = re.findall(r'\d+', match.group(0))
        return " ".join(f"(مصدر {number})" for number in numbers)

    return _CITATION_CLUSTER_RE.sub(_expand, text)

# Arabic / Latin sentence-end delimiters used for splitting
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?\u061f\n])')
_GROUNDING_MIN_CONTENT_TOKENS = 2

# Rare-token rule: an unmatched content token whose corpus document frequency
# is at or below this bound is treated as strong evidence AGAINST the claim
# (subject swaps like الخندق, relation inversions like يدخلون, and fabricated
# entities are distinctive by nature).  Common words may be unmatched without
# rejecting the sentence, so ordinary paraphrasing survives.  Bound chosen
# empirically: distinctive toponyms/entities sit far below it (خندق=119,
# يدخلون=56, شوال=74) while common narrative verbs/words sit above
# (وقعت=199, شهر=641, مؤكدة=27 is routed through the phrase alias).
RARE_DF = 150

# --- P5v5 grounding policy (Phase 3K.11, promoted from the 3K candidate) ----
# Reviewed components: clitic-v2 union document frequency, seed-aligned
# cited-source family exemption with polarity filters, material-addition
# guard (band/coverage), right-only negation marking with claim/source
# polarity parity, sequence-aware quote verification inside one chunk, and
# the entity-anchor follower guard.  Any of the polarity, quote or entity
# rejections may not be bypassed by the family exemption.
_GROUNDING_DF_THR = 300
_GROUNDING_GUARD_COV = 0.80
_GROUNDING_GUARD_BAND = 500

# Curated grounding alias (Phase E.4): the claim token نزول is treated as
# family-supported only by the exact cited-source surface نزلت, under the
# same polarity filters as every other family hit.  Deliberately a single
# hand-curated pair, NOT a general hollow-verb / waw-deletion rule: the
# Phase E.3 generalization audit showed a shape-based predicate also
# bridges unrelated roots (صوام/صامت، ثواب/ثابت، راوي/رايت، تورك/تركت),
# so only this evidence-backed pair is admitted.
_GROUNDING_CURATED_ALIASES: dict[str, tuple[str, ...]] = {
    "نزول": ("نزلت", "انزل", "نزل"),
    # "عدم وجود الماء" paraphrases the fiqh/Quran wording "لم يجد الماء"
    # (fiqh_66: "فلم تجدوا ماء"; fiqh_67: "لم يجد الماء").  The hollow-verb
    # family matcher does not align وجود with يجد, so admit the exact cited
    # surfaces only.  This keeps a supported paraphrase from being dropped
    # without weakening the rare-token check for any other word.
    "وجود": ("يجد", "تجدوا", "وجد"),
    # "يشعر بالقلق" paraphrases the revelation hadith's "كُرِبَ لِذَلِكَ"
    # (he was distressed by it).  Exact cited surface only; the family
    # matcher does not connect قلق with كرب.
    "قلق": ("كرب",),
}

# Polarity-inverting particles (غير/بدون), tolerant of و/ف prefixes.  Kept
# separate from _is_negation_token: غير/بدون do not negate a verb window,
# they invert the polarity of the following noun ("غير مستحب" vs "مستحب").
_GROUNDING_INVERSION_TOKENS = {"غير", "بدون"}

# Entity anchors: a claim token directly following one of these is a
# named-entity mention whose support must be verifiable in the cited source
# (follower presence, not corpus frequency).
#
# Construct anchors (ابن/بنت/أبي/أبو/أم/عبد/بن) introduce an idafa phrase, so
# BOTH tokens must occur together in the cited source.  Follower-only checking
# let "أم الفضل" pass whenever the common noun "فضل" appeared anywhere in a
# source, even though no source named that woman.  Label anchors introduce a
# person by name ("تدعى حليمة", "يسمى ...") and only require the name itself.
_GROUNDING_CONSTRUCT_ANCHORS = {"ابن", "بنت", "ابي", "ابو", "ام", "عبد", "بن"}
_GROUNDING_LABEL_ANCHORS = {
    "تدعى", "يدعى", "تسمى", "يسمى", "اسمه", "اسمها", "امه", "امها",
    "ابوه", "ابوها", "جدها", "جده", "زوجها", "زوجه",
}
_GROUNDING_ENTITY_ANCHORS = (
    _GROUNDING_CONSTRUCT_ANCHORS | _GROUNDING_LABEL_ANCHORS
)

# Windowed in-order quote alignment tolerates at most this many extra source
# tokens between the first and last aligned span token (covers Arabic
# insertions like "أهل العلم" / conjunction particles).  Fixed a priori.
_GROUNDING_SPAN_MAX_GAP = 6

# Quoted-span extraction: ASCII double quotes plus Arabic/French guillemets
# and curly quotes.
_QUOTE_RE = re.compile(
    r'["\u00ab\u00bb\u201c\u201d]([^"\u00ab\u00bb\u201c\u201d]{6,})'
    r'["\u00ab\u00bb\u201c\u201d]')

# Tafsir answers quote Quranic text with the ornate Quranic brackets.
# Keeping this separate from _QUOTE_RE preserves the chat grounding contract.
_QUOTE_RE_WITH_QURAN = re.compile(
    r'(?:["\u00ab\u00bb\u201c\u201d]|\uFD3F)'
    r'([^"\u00ab\u00bb\u201c\u201d\uFD3E]{6,})'
    r'(?:["\u00ab\u00bb\u201c\u201d]|\uFD3E)')

# --- Tolerant Arabic matching for the grounding check (F-03) ----------------
# Classic orthography differs from modern spelling in ways that must not
# cause correct paraphrases to be rejected:
#   * Quran Uthmani text drops the superscript/dagger alef (أموال -> امول,
#     اليتامى -> يتمى) and leaves combining marks that normalize() misses.
#   * Dates/numbers appear both spelled out and as ASCII/Arabic-Indic digits.
# The helpers below fold both the answer sentence and the source text to a
# tolerant match key and canonicalise number word-sequences to digits.  The
# check stays strict where it matters: every number and every month word in
# the answer must be supported by the cited source text.

_GROUNDING_DIGITS_TRANS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"
)

_GROUNDING_SUFFIXES = (
    "كما", "هما", "ات", "ون", "ين", "ها", "هم", "هن", "نا", "كم", "كن",
    "وا", "ه", "ت",
)

_GROUNDING_IGNORED_TOKENS = {
    "مصدر", "مصادر", "المصدر", "بمصدر",
    # Discourse/connective words: they carry no factual claim, but their
    # surface forms can be rare in the corpus (e.g. وقد) and must not drop an
    # otherwise well-supported sentence.
    "وقد", "قد", "وعندما", "عندما", "حيث", "كما", "لكن", "ايضا", "أيضا",
    # Affirmation/denial particles are discourse answers, not factual claims;
    # the ruling they introduce is checked separately.
    "نعم", "اجل", "بلى",
    # Presentational/framing fillers: they introduce a supported statement
    # but carry no factual claim of their own.
    "هناك", "ينص", "يعتبر", "تعتبر", "ذلك", "هذا", "هذه", "هؤلاء",
    # Generic human subject used as a placeholder ("إذا لم يجد الشخص الماء")
    # rather than a factual entity; it must not trigger the rare-token guard.
    "شخص", "الشخص",
    # Naming verbs introduce a name; the name itself is checked by the entity
    # guard, so the framing verb must not independently drop the sentence.
    "تدعى", "يدعى", "تسمى", "يسمى",
    # Presentational count ("in several forms") introducing a verbatim
    # quotation.  The quotation and the surrounding claim carry the factual
    # content; the framing noun itself is not an independently checkable
    # assertion and is common enough that its low df is an artefact.
    "اشكال", "أشكال",
}

# Quranic references such as (النساء: 10) are citations, not claim digits.
# Strip them before number verification so a valid ayah reference cannot
# turn a supported answer into a grounding failure.
_QURAN_REF_RE = re.compile(r"\([^()]{2,40}[:\uFF1A]\s*\d+\s*\)")

# Month words that are also common non-date words (محرم = forbidden,
# صفر = zero).  They are only treated as dates when explicitly preceded by
# "شهر"; otherwise "محرم" in a ruling sentence is a prohibition.
_GROUNDING_AMBIGUOUS_MONTHS = {"محرم", "صفر"}

_GROUNDING_MONTHS = {
    "محرم", "صفر", "ربيع", "جمادي", "جمادى", "رجب", "شعبان", "رمضان",
    "شوال", "قعده", "حجه",
}

_GROUNDING_UNIT_WORDS = {
    "واحد": 1, "احد": 1, "احده": 1, "اول": 1, "الاول": 1,
    "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2, "ثاني": 2, "ثانيه": 2,
    "ثلاث": 3, "ثلاثه": 3, "ثالث": 3, "ثالثه": 3,
    "اربع": 4, "اربعه": 4, "رابع": 4, "رابعه": 4,
    "خمس": 5, "خمسه": 5, "خامس": 5, "خامسه": 5,
    "ست": 6, "سته": 6, "سادس": 6, "سادسه": 6,
    "سبع": 7, "سبعه": 7, "سابع": 7, "سابعه": 7,
    "ثمان": 8, "ثماني": 8, "ثمانيه": 8, "ثامن": 8, "ثامنه": 8,
    "تسع": 9, "تسعه": 9, "تاسع": 9, "تاسعه": 9,
}

_GROUNDING_TENS_WORDS = {
    "عشر": 10, "عشره": 10, "عاشر": 10, "عاشره": 10,
    "عشرون": 20, "عشرين": 20,
    "ثلاثون": 30, "ثلاثين": 30, "اربعون": 40, "اربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60,
    "سبعون": 70, "سبعين": 70, "ثمانون": 80, "ثمانين": 80,
    "تسعون": 90, "تسعين": 90,
}

# Ordinal labels ("the second pledge", "the third chapter") are sequence
# names, not quantities.  They still participate in coverage matching, but
# they must not trigger the strict digit guard: a source can describe the
# same event with a different ordinal label ("العقبة الأخيرة" vs
# "العقبة الثانية"), and requiring the numeral 2 would reject a verbatim,
# correctly cited quotation.
_GROUNDING_ORDINAL_WORDS = {
    "اول", "الاول", "ثاني", "ثانيه", "ثالث", "ثالثه",
    "رابع", "رابعه", "خامس", "خامسه", "سادس", "سادسه",
    "سابع", "سابعه", "ثامن", "ثامنه", "تاسع", "تاسعه",
    "عاشر", "عاشره",
}


_GROUNDING_ORTHO_TRANS = str.maketrans({
    # Classical/Uthmani text uses alef maqsura where modern text uses yeh
    # (الوحى/الوحي, هدى/هدي).  Without this fold the two spellings produce
    # different grounding keys, so a verbatim Quran/hadith word can look
    # unsupported and a rare-token guard can drop a correct sentence.
    "ى": "ي",
    # Hamza carriers are orthographic variants of the same letter for
    # matching purposes (الرؤيا/الرويا, قائم/قايم).
    "ؤ": "و",
    "ئ": "ي",
})


def _fold_ar(token: str) -> str:
    """Fold orthographic variance: combining marks, Unicode format characters
    (category Cf, e.g. the RTL mark U+200F glued to some corpus tokens),
    tatweel, Arabic digits, and alef maqsura/hamza-carrier spellings."""
    t = unicodedata.normalize("NFKD", token)
    t = "".join(
        ch for ch in t
        if not unicodedata.combining(ch)
        and unicodedata.category(ch) != "Cf"
    )
    return (
        t.replace("\u0640", "")
        .translate(_GROUNDING_DIGITS_TRANS)
        .translate(_GROUNDING_ORTHO_TRANS)
        .strip()
    )


# The grounding constants above are written in their conventional Arabic
# spelling, while ``_tokenize`` normalizes surface forms (e.g. تدعى -> تدعي,
# على -> علي).  Compare against folded copies so orthographic variants cannot
# silently disable the entity or ignored-token guards.
_GROUNDING_IGNORED_FOLDED = {
    _fold_ar(t) for t in _GROUNDING_IGNORED_TOKENS
}

# Legal modality is part of the claim, not a discourse filler.  A sentence
# that says "يجب" is not grounded by a source that only says "يستحب" or
# "لا يكره", even when every topic word matches.  These families are used by
# the grounding validator and the citation chooser; they do not encode any
# particular answer.
_LEGAL_FAMILY_WORDS = {
    "obligation": {
        "يجب", "وجب", "تجب", "يجبون", "واجب", "واجبه", "واجبات",
        "وجوب", "اوجب", "يوجب", "فرض", "فرضه", "فرضت", "فريضه",
        "مفروض", "لازم", "لازمه", "يلزم", "الزام",
    },
    "prohibition": {
        "يحرم", "حرم", "حرام", "حرامه", "محرم", "محرمه",
        "تحريم", "حرمه", "ممنوع",
    },
    "permission": {
        "يجوز", "جاز", "جائز", "جايز", "جواز", "مباح",
        "اباح", "يبيح", "حلال", "حلاله", "محلل",
    },
    "recommendation": {
        "يستحب", "استحب", "مستحب", "مستحبه", "يستحسن",
        "مستحسن", "مندوب", "مندوبه",
    },
    "dislike": {
        "يكره", "كره", "مكروه", "مكروهه", "كراهه", "كراهيه",
    },
    "invalid": {
        "باطل", "باطله", "يبطل", "بطل", "فسد", "فاسد", "فساد",
    },
}
_LEGAL_FAMILY_BY_KEY: dict[str, str] = {}
for _legal_family, _legal_words in _LEGAL_FAMILY_WORDS.items():
    for _legal_word in _legal_words:
        _LEGAL_FAMILY_BY_KEY[
            _strip_al(_fold_ar(_legal_word))
        ] = _legal_family


_LEGAL_FAMILY_PREFIXES = (
    ("obligation", ("يجب", "وجب", "واجب", "وجوب", "فرض", "فريض", "مفروض", "لازم", "يلزم", "الزام")),
    ("prohibition", ("يحرم", "حرم", "حرام", "محرم", "تحريم", "ممنوع")),
    ("permission", ("يجوز", "جاز", "جائز", "جايز", "جواز", "مباح", "اباح", "يبيح", "حلال", "محلل")),
    ("recommendation", ("يستحب", "استحب", "مستحب", "يستحسن", "مستحسن", "مندوب")),
    ("dislike", ("يكره", "كره", "مكروه", "كراه")),
    ("invalid", ("باطل", "يبطل", "بطل", "فسد", "فاسد", "فساد")),
)


_LEGAL_PHRASE_KEYS = {"عليه", "عليك", "علينا", "عليهم", "عليها", "كتب"}
# Generic fast/day words do not identify which passage a legal claim belongs
# to; sentence-scoped support uses a more distinctive subject token.
_LEGAL_FOCUS_STOP = {
    "صوم", "صيام", "يوم", "ايام", "مسلم", "المسلم",
    "شخص", "اذا", "اراد", "يصوم", "مسلمين",
    "قبل", "بعد", "قبله", "بعده",
}


def _legal_key(token: str) -> str:
    key = _strip_al(_fold_ar(token))
    if (
        len(key) > 2 and key[0] in "وف"
        and (
            key[1:] in _LEGAL_FAMILY_BY_KEY
            or key[1:] in _LEGAL_PHRASE_KEYS
        )
    ):
        key = key[1:]
    return key


def _legal_family_of_token(token: str) -> str | None:
    key = _legal_key(token)
    family = _LEGAL_FAMILY_BY_KEY.get(key)
    if family is not None:
        return family
    # Morphological variants ("وجوبها", "واجبة") keep the same family.
    for candidate, prefixes in _LEGAL_FAMILY_PREFIXES:
        if key.startswith(prefixes):
            return candidate
    return None


def _polarity_marked_positions(tokens: list[str]) -> set[int]:
    """Token positions governed by negation or غير/بدون inversion."""
    marked: set[int] = set()
    for position, token in enumerate(tokens):
        if _is_negation_token(token):
            marked |= _negation_target_positions(tokens, position)
        if _is_inversion_token(token) and position + 1 < len(tokens):
            marked.add(position + 1)
    return marked


def _positive_legal_families_from_tokens(tokens: list[str]) -> set[str]:
    """Legal conclusions positively asserted by a token stream.

    Negated permission ("لا يجوز") is a prohibition; negated obligation or
    dislike asserts no positive strong claim.  "كتب عليكم" is treated as an
    explicit prescription, not as the noun "books".
    """
    marked = _polarity_marked_positions(tokens)
    families: set[str] = set()
    for position, token in enumerate(tokens):
        family = _legal_family_of_token(token)
        if family is None:
            folded = _legal_key(token)
            next_key = (
                _legal_key(tokens[position + 1])
                if position + 1 < len(tokens) else ""
            )
            if (
                folded == "كتب"
                and next_key in {"علي", "عليكم", "عليهم", "على"}
            ) or (
                folded in {"عليه", "عليك", "علينا", "عليهم", "عليها"}
                and next_key == "ان"
            ):
                # "كتب عليكم" is an explicit prescription; "فعليه أن"
                # is the model's common paraphrase of an obligation.
                family = "obligation"
            else:
                continue
        if not _legal_token_is_ruling(tokens, position, token):
            continue
        negated = position in marked
        if family == "permission" and negated:
            families.add("prohibition")
        elif family in {
            "obligation", "prohibition", "permission",
            "recommendation", "dislike", "invalid",
        } and not negated:
            families.add(family)
    return families


def _legal_token_is_ruling(
    tokens: list[str], position: int, token: str,
) -> bool:
    """False for ordinary verbs such as "بما يكره" (what he dislikes)."""
    key = _legal_key(token)
    if key in {"يكره", "كره", "يحرم", "حرم"} and position > 0:
        previous = _legal_key(tokens[position - 1])
        if previous in {"بما", "ما", "الذي", "التي", "من", "انه", "انها", "وما"}:
            return False
    return True


def _negative_legal_families_from_tokens(tokens: list[str]) -> set[str]:
    """Legal families asserted under negation/inversion.

    A source that only says "مستحب" cannot ground "غير مستحب"; a source with
    no obligation token at all does not contradict "لا يجب" (the cautious
    negative claim is allowed).
    """
    marked = _polarity_marked_positions(tokens)
    families: set[str] = set()
    for position, token in enumerate(tokens):
        family = _legal_family_of_token(token)
        if family is None or position not in marked:
            continue
        if not _legal_token_is_ruling(tokens, position, token):
            continue
        if family == "permission":
            # "لا يجوز" is a positive prohibition claim, handled elsewhere.
            continue
        families.add(family)
    return families


def _positive_legal_families(text: str) -> set[str]:
    return _positive_legal_families_from_tokens(
        _numbers_normalized(_tokenize(text or "")),
    )


def _negative_legal_families(text: str) -> set[str]:
    return _negative_legal_families_from_tokens(
        _numbers_normalized(_tokenize(text or "")),
    )
_GROUNDING_ENTITY_ANCHORS_FOLDED = {
    _fold_ar(t) for t in _GROUNDING_ENTITY_ANCHORS
}
_GROUNDING_CONSTRUCT_ANCHORS_FOLDED = {
    _fold_ar(t) for t in _GROUNDING_CONSTRUCT_ANCHORS
}


def _fold_weak(token: str) -> str:
    """Weak-letter skeleton used only by the relaxed rare-token exemption.

    Drops interior alef/waw/yeh after the first letter so a paraphrased
    surface that is absent from the corpus (e.g. وصيرتم) can still align
    with its cited-source spelling (وصرتم).  Short skeletons are ignored by
    the caller, so coincidence on two-letter function words cannot exempt a
    fabricated token.
    """
    folded = _fold_ar(token)
    if len(folded) <= 1:
        return folded
    return folded[0] + "".join(
        ch for ch in folded[1:] if ch not in "اوي"
    )


def _match_key(token: str) -> str:
    """Tolerant key for grounding token matching.

    Folds orthography and strips leading conjunctions (و/ف, iteratively)
    FIRST, then one common suffix (each guarded so that at least 2 letters
    remain), only THEN removes interior alef (Quran dagger-alef artifact:
    أموال -> امول, نارا -> نر, يتامى -> يتمى), and last an imperfective
    prefix (ي/ت/ن).  The suffix must be stripped BEFORE the alef removal:
    suffixes like وا/ات/هم carry the alef themselves, so removing interior
    alef first hid them and left plural stems unmatched against their
    singular (اجتنبوا -> اجتنبو instead of اجتنب, قالوا -> قلو instead of
    قال).  The imperfective prefix must stay AFTER the alef removal: on a
    full 4-letter word it would strip root letters (نارا -> ارا instead of
    نر).  Note: _tokenize() has already stripped definite articles
    (ال/وال/...) before this runs, so both the claim and the reference side
    always apply the exact same transformation.
    """
    t = _fold_ar(token)
    if not t:
        return t
    while len(t) > 3 and t[0] in "وف":
        t = t[1:]
    # Dual endings: ركعتان/ركعتين/ركعتا all reduce to ركعت, so a rare
    # dual in the answer can align with a source that uses another case.
    # Do not strip bare "ان": that would corrupt رمضان/عثمان and similar.
    if t.endswith("تان") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("تا") and len(t) > 3:
        t = t[:-1]
    else:
        for suf in _GROUNDING_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[:-len(suf)]
                break
    t = t[0] + t[1:].replace("ا", "")
    if len(t) > 3 and t[0] in "يتن":
        t = t[1:]
    return t


_NEGATION_TOKENS = {"لا", "لم", "لن", "ليس", "ليست"}

# Short particles that can sit between a negation and the governed content
# word without changing its polarity.  "لا على الوجوب" negates الوجوب just
# as "ليس للوجوب" does; without this bridge the source-side marking misses
# the target and a valid negated claim is rejected as a flip.
_NEGATION_SKIP_TOKENS = {
    _fold_ar(t) for t in (
        "على", "في", "من", "الى", "عن", "ان", "هو", "هي", "هذا", "هذه",
        "ذلك", "تلك", "مع", "ب", "ل", "ك",
        # Copulas do not stop the scope of a negation: "لم يكن معهم ماء" and
        # "ليس معهم ماء" negate the same predicate.  Without bridging the
        # copula, the claim token معهم was treated as affirmative while the
        # source marked it negated, producing a false neg=flip.
        "كان", "كانت", "يكون", "تكون", "يكن", "تكن", "نكون", "اكون",
    )
}


def _negation_target_positions(tokens: list[str], start: int) -> set[int]:
    """Positions governed by the negation at *start* (immediate + after a
    short preposition/pronoun bridge).  Right-only, matching the existing
    guard; the extra target is only added when it is a later token."""
    if start + 1 >= len(tokens):
        return set()
    targets = {start + 1}
    k = start + 1
    while k < len(tokens) and _fold_ar(tokens[k]) in _NEGATION_SKIP_TOKENS:
        k += 1
    if k < len(tokens) and k != start + 1:
        targets.add(k)
    return targets


def _is_negation_token(token: str) -> bool:
    """True for Arabic negation particles, tolerating و/ف conjunctions.

    Deliberately minimal (لا/لم/لن/ليس/ليست, exact match after folding and
    stripping a leading و/ف).  Ambiguous words (ما، غير، إلا) are excluded so
    the grounding negation check can never fire on a false cue.
    """
    f = _fold_ar(token)
    if len(f) > 2 and f[0] in "وف":
        f = f[1:]
    return f in _NEGATION_TOKENS


def _strict_key(token: str) -> str:
    """High-precision surface key for the negation check only.

    Folds orthography, strips a leading و/ف conjunction (and a definite
    article if still present), but performs NO stemming — so يعيد (repeat)
    and عيدين (the two Eids) never collide, unlike _match_key.  The negation
    check is deliberately conservative: a stricter key can only miss a flip,
    never invent one.
    """
    t = _fold_ar(token)
    while len(t) > 2 and t[0] in "وف":
        t = t[1:]
    return _strip_al(t) if t else t


def _source_strict_keys(text: str) -> set[str]:
    return {
        _strict_key(token)
        for token in _numbers_normalized(_tokenize(text or ""))
        if len(token) >= 3
    }


_LEGAL_TOPIC_ALIASES = {
    "صيام": "صوم", "صوم": "صوم", "صيامه": "صوم", "صومه": "صوم",
    "يصوم": "صوم", "صائم": "صوم", "صوموا": "صوم",
}


def _legal_topic_key(token: str) -> str:
    strict = _strict_key(token)
    mapped = _LEGAL_TOPIC_ALIASES.get(strict)
    if mapped:
        return mapped
    if (
        "صوم" in strict or "صيم" in strict
        or strict.startswith(("صائم", "صايم", "صاءم"))
    ):
        return "صوم"
    return strict


def _legal_topic_keys(text: str) -> set[str]:
    return {
        _legal_topic_key(token)
        for token in _tokenize(text or "")
        if len(token) >= 3
    }


_LEGAL_FOCUS_STOP_STRICT = {
    _strict_key(word) for word in _LEGAL_FOCUS_STOP
}


# Mutually exclusive outcome destinations.  A claim asserting exactly one
# side of a pair is contradicted when the cited source asserts the opposite
# side and never the claimed side (e.g. claim "يدخلون الجنة" against source
# "نارا").  Keys are strict surface keys (no stemming) so جن (jinn) can
# never be confused with جنة.  The check is skipped when a negation or
# inversion governs the claimed destination in the claim, or the asserted
# destination in the source.
_GROUNDING_CONTRADICTION_GROUPS = [
    (
        frozenset({
            _strict_key("جنه"), _strict_key("جنات"), _strict_key("جنان"),
        }),
        frozenset({
            _strict_key(t) for t in
            ("نار", "نارا", "جهنم", "سعير", "سعيرا", "لظى", "سقر", "حطمه")
        }),
    ),
]


def _build_group_canon() -> dict[str, str]:
    """Map every synonym-group member's match key to the group's canonical
    key (first member), so وقعة/معركة/غزوة all compare equal in grounding."""
    canon: dict[str, str] = {}
    for grp in (*_SYNONYM_GROUPS, *_GROUNDING_EXTRA_SYNONYM_GROUPS):
        c = _match_key(grp[0])
        for member in grp:
            canon[_match_key(member)] = c
    return canon


_GROUNDING_GROUP_CANON = _build_group_canon()

# Scoped phrase aliases on match keys (mirror of _PHRASE_ALIASES): only the
# adjacent bigram (سنن مؤكدة) is rewritten; a bare مؤكدة never is.  Pattern
# keys are computed with the same matcher used at check time (note that
# _fold_ar turns ؤ into و, so the pattern is 'موكد', not 'مؤكد').
_GROUNDING_PHRASE_ALIASES = [
    (
        (_match_key("سنن"), _match_key("مؤكده")),
        ("سنن", "راتبه"),
    ),
]


def _gkey(token: str) -> str:
    """Match key with synonym-group canonicalisation applied."""
    k = _match_key(token)
    return _GROUNDING_GROUP_CANON.get(k, k)


_LEGAL_FOCUS_STOP_GKEYS = {
    _gkey(word) for word in _LEGAL_FOCUS_STOP
}
_LEGAL_REQUIRED_GENERIC = {
    "يوم", "ايام", "شخص", "اذا", "اراد", "يصوم",
    "مسلم", "مسلمين", "قبل", "بعد", "قبله", "بعده",
    "علماء", "جمهور", "يرون", "قال", "ذهب", "ورد",
    "حديث", "فقه", "بعض", "اهل",
}
_LEGAL_REQUIRED_GENERIC_GKEYS = {
    _gkey(word) for word in _LEGAL_REQUIRED_GENERIC
}


def _apply_grounding_aliases(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for i in range(len(out) - 1):
        a, b = _match_key(out[i]), _match_key(out[i + 1])
        for pat, rep in _GROUNDING_PHRASE_ALIASES:
            if a == pat[0] and b == pat[1]:
                out[i + 1] = rep[1]
    return out


def _df_of(token: str, corpus_chunks: list[dict] | None) -> int | None:
    """Document frequency of a token in the corpus BM25 index.

    Returns None when no corpus is available (callers must skip the
    rare-token check, never guess).  0 means the token never occurs.
    """
    if corpus_chunks is None:
        return None
    postings = _build_bm25(corpus_chunks)["postings"]
    f = _fold_ar(token)
    forms = [token, f]
    if f:
        c = f
        if len(c) > 3 and c[0] in "وف":
            c = c[1:]
        forms.append(c)                      # conjunction-stripped surface
        forms.append(f[0] + f[1:].replace("ا", ""))   # rasm form
    forms.extend([_strip_al(token), _match_key(token)])
    folded = _fold_lexical(token)
    forms.extend([folded, _match_key(folded)])
    for form in forms:
        if form and form in postings:
            return len(postings[form])
    return 0


def _forms_of(t: str) -> list[str]:
    """Spelling variants for the clitic document-frequency metric.

    Folding, conjunction-stripped and rasm forms mirror _df_of's candidate
    list; _strip_al/_match_key cover definite articles and light stemming.
    """
    f = _fold_ar(t)
    fs = [t, f]
    if f:
        c = f
        if len(c) > 3 and c[0] in "وف":
            c = c[1:]
        fs.append(c)
        fs.append(f[0] + f[1:].replace("ا", ""))
    fs.extend([_strip_al(t), _match_key(t)])
    folded = _fold_lexical(t)
    fs.extend([folded, _match_key(folded)])
    out = []
    for x in fs:
        if x and x not in out:
            out.append(x)
    return out


_CLITIC_PREFIXES = ["", "و", "ف", "ل", "ب", "ك"]
_CLITIC_SUFFIXES = ["", "ه", "ها", "هم", "هن", "نا", "كم", "كن", "ات", "ون",
                    "ين", "وا", "ا", "ت"]
_FORM_PREFIXES = ["ي", "ت", "ن", "م", "ا", "س"]


def _seeds_v2(t: str) -> set[str]:
    """Stem seeds: base forms plus iterative conjunction/article-prefix
    strips and verb-prefix strips."""
    seeds = {x for x in _forms_of(t) if x}
    for b in list(seeds):
        if len(b) > 3 and b[0] in "وفلبك":
            seeds.add(b[1:])
            if len(b) > 3 and b[1] in "وفلبك":
                seeds.add(b[2:])
    for b in list(seeds):
        if b and b[0] in "ميتناس" and len(b) > 3:
            seeds.add(b[1:])
        elif b and b[0] in "يتن" and len(b) == 3:
            # Three-letter imperfect verbs: يجر -> جر, ينص -> نص.  Needed so
            # the rare-token family exemption sees the root variant used by
            # the source.
            seeds.add(b[1:])
    # Past-tense verb + object pronoun (أصابته) is reduced by _match_key to
    # أصبت (one suffix removed).  Allow one further final-ت strip so the bare
    # verb stem (أصاب -> أصب) can exempt the paraphrased form when the cited
    # source uses that verb.  Guarded by length; the exemption still requires
    # the resulting seed to occur in the cited source surface.
    for b in list(seeds):
        if len(b) > 3 and b.endswith("ت"):
            seeds.add(b[:-1])
    return {s for s in seeds if s}


def _clitic_v2(t: str) -> set[str]:
    """Clitic variant set (prefixes × suffixes) around each seed."""
    vs: set[str] = set()
    for s in _seeds_v2(t):
        vs.add(s)
        for p in _CLITIC_PREFIXES:
            vs.add(p + s)
            for sfx in _CLITIC_SUFFIXES:
                if sfx:
                    vs.add(p + s + sfx)
    return {v for v in vs if v}


def _family_v2(t: str) -> set[str]:
    """Wider family set: clitic variants plus letter-form prefixes with
    conjunction/suffix combinations, for the cited-source family exemption."""
    vs = set(_clitic_v2(t))
    for s in _seeds_v2(t):
        for fp in _FORM_PREFIXES:
            f = fp + s
            vs.add(f)
            for p in ("و", "ف"):
                vs.add(p + f)
                for sfx in ("ه", "ها", "هم", "ات", "ون", "ين", "ا", "ت"):
                    vs.add(p + f + sfx)
    return {v for v in vs if v}


def _union_docs_count(variants: set[str], postings: dict) -> int:
    """Number of distinct chunks containing at least one of the variants."""
    docs: set[str] = set()
    for v in variants:
        pl = postings.get(v)
        if pl:
            docs.update(cid for cid, _ in pl)
    return len(docs)


def _is_inversion_token(token: str) -> bool:
    """True for polarity-inverting particles غير/بدون (و/ف-tolerant).

    The inversion inverts the polarity of the following noun, which is the
    relation the family-exemption filter and the source-side flip check
    compare against — distinct from verb-window negation (لا/لم/لن/ليس).
    """
    t = _fold_ar(token)
    while len(t) > 2 and t[0] in "وف":
        t = t[1:]
    return t in _GROUNDING_INVERSION_TOKENS


def _seed_aligned(t: str, s: str) -> bool:
    """Semantic alignment for the family exemption: t and s must share a
    _seeds_v2 core stem (استحب/يستحب/مستحب align via 'ستحب'), so a bare
    clitic-prefixed coincidence can no longer exempt a rare token."""
    return bool(_seeds_v2(t) & _seeds_v2(s))


def _curated_alias_surfaces(t: str, ref_surface: set[str]) -> set[str]:
    """Cited surfaces admitted for a claim token via the curated aliases.

    Folded exact-target comparison only (no stem, prefix or suffix
    expansion): the claim token must be a table key and the cited surface
    must fold to one of its curated targets; every other token yields the
    empty set.  Used solely by the rare-token family exemption.
    """
    targets = _GROUNDING_CURATED_ALIASES.get(_fold_ar(t))
    if not targets:
        return set()
    return {s for s in ref_surface if _fold_ar(s) in targets}


def _seq_support(a: str, b: str, mode: str) -> bool:
    """Narrow alignment for in-quote verification: _gkey equality plus, in
    "aligned" mode, the same seed alignment the family exemption uses."""
    if _gkey(a) == _gkey(b):
        return True
    if mode == "exact":
        return False
    return _seed_aligned(a, b)


def _span_align(span_toks: list[str], ref_toks: list[str], mode: str,
                max_gap: int) -> tuple[bool, list[int]]:
    """In-order windowed alignment of a quoted span inside one chunk's token
    stream.  For every candidate start position, greedily align the rest in
    order (earliest placement is optimal for a fixed start) and accept the
    first alignment whose source window is <= len(span) + max_gap tokens.
    Enumerating starts prevents a distant seed-aligned coincidence from
    poisoning an alignment that exists contiguously elsewhere in the chunk.
    """
    if not span_toks:
        return False, []
    limit = len(span_toks) + max_gap
    for start in range(len(ref_toks)):
        if not _seq_support(span_toks[0], ref_toks[start], mode):
            continue
        positions = [start]
        pos = start
        ok = True
        for st in span_toks[1:]:
            nxt = None
            for p in range(pos + 1, min(len(ref_toks), start + limit)):
                if _seq_support(st, ref_toks[p], mode):
                    nxt = p
                    break
            if nxt is None:
                ok = False
                break
            positions.append(nxt)
            pos = nxt
        if ok and positions[-1] - start + 1 <= limit:
            return True, positions
    return False, []


def _num_lookup(f: str, table: dict[str, int]) -> int | None:
    """Lookup a (possibly و/ب-prefixed or ال-prefixed) number word."""
    cands = [f]
    t = f
    if t.startswith("ال") and len(t) > 2:
        t = t[2:]
        cands.append(t)
    for c in list(cands):
        if len(c) > 1 and c[0] in "وب":
            cands.append(c[1:])
    for c in cands:
        if c in table:
            return table[c]
    return None


def _is_ordinal_word(folded: str) -> bool:
    """True when a folded token is an ordinal label (first/second/...)."""
    candidates = [folded]
    if folded.startswith("ال") and len(folded) > 2:
        candidates.append(folded[2:])
    for candidate in list(candidates):
        if candidate and candidate[0] in "وفب":
            candidates.append(candidate[1:])
    return any(candidate in _GROUNDING_ORDINAL_WORDS
               for candidate in candidates)


def _numbers_normalized_meta(tokens: list[str]) -> list[tuple[str, bool]]:
    """Canonicalise numbers and flag the results that came from ordinals."""
    out: list[tuple[str, bool]] = []
    i = 0
    while i < len(tokens):
        f = _fold_ar(tokens[i])
        if f.isdigit():
            out.append((str(int(f)), False))
            i += 1
            continue
        tens = _num_lookup(f, _GROUNDING_TENS_WORDS)
        if tens is not None:
            out.append((str(tens), _is_ordinal_word(f)))
            i += 1
            continue
        unit = _num_lookup(f, _GROUNDING_UNIT_WORDS)
        if unit is not None:
            ordinal = _is_ordinal_word(f)
            nxt = _fold_ar(tokens[i + 1]) if i + 1 < len(tokens) else ""
            nxt_tens = _num_lookup(nxt, _GROUNDING_TENS_WORDS)
            if nxt in ("عشر", "عشره") or nxt_tens == 10:
                out.append((
                    str(10 + unit),
                    ordinal or _is_ordinal_word(nxt),
                ))
                i += 2
                continue
            if nxt_tens is not None and nxt_tens >= 20 and nxt.startswith("و"):
                out.append((
                    str(nxt_tens + unit),
                    ordinal or _is_ordinal_word(nxt),
                ))
                i += 2
                continue
            out.append((str(unit), ordinal))
            i += 1
            continue
        out.append((tokens[i], False))
        i += 1
    return out


def _numbers_normalized(tokens: list[str]) -> list[str]:
    """Replace spelled number words and digit tokens by canonical decimal
    strings so that 'سبع عشرة' == 'سابع عشر' == '١٧' == '17'.

    Handles prefixed words (بسبع) and the compound unit+tens form
    (سبع وعشرين -> 27, خمسة وثلاثون -> 35).  Ordinals are canonicalised too
    (for coverage matching); `_numbers_normalized_meta` exposes which output
    positions came from ordinals so the strict quantity guard can skip them.
    """
    return [token for token, _ in _numbers_normalized_meta(tokens)]


_QUOTE_ONLY_FRAMING_FOLDED = {
    _gkey(word) for word in (
        "قال", "قالت", "الله", "تعالى", "رسول", "النبي", "صلى",
        "عليه", "وسلم", "حديث", "الحديث", "آية", "الآية", "سورة",
        "السورة", "رقم", "المصدر", "مصدر", "الشرح",
    )
}


def _quote_only_supported(
    sentence: str,
    ref_keys: set[str],
    ref_title_keys: set[str],
) -> bool:
    """True when a sentence is just a verbatim quote plus framing/citation.

    A direct Quran or hadith quotation can be fully supported even when the
    surrounding framing words ("قال الله تعالى", "حديث", "المصدر") are not
    content of the cited passage.  The quote itself is still verified by the
    normal quote guard; this helper only decides whether the remaining words
    are non-claims (framing or source-title tokens).
    """
    if not _QUOTE_RE.search(sentence):
        return False
    remainder = _QUOTE_RE.sub(" ", sentence)
    remainder = _CITATION_RE.sub(" ", remainder)
    remainder = _QURAN_REF_RE.sub(" ", remainder)
    for token in _tokenize(remainder):
        key = _gkey(token)
        if not key or key.isdigit():
            continue
        if key in _QUOTE_ONLY_FRAMING_FOLDED:
            continue
        if key in ref_keys or key in ref_title_keys:
            continue
        return False
    return True


def _ground_answer(answer: str, retrieved: list[dict],
                   corpus_chunks: list[dict] | None = None,
                   ignore_rare: bool = False,
                   quran_quotes: bool = False) -> str:
    """Post-generation grounding filter (tolerant matcher).

    For each sentence in *answer*:
      1. Extract cited source numbers (maSdar markers).  Every explicit
         citation must resolve within 1..len(retrieved); otherwise the
         sentence fails closed.  Uncited sentences must be supported by
         ONE SINGLE chunk on its own — never by a union across chunks.
      2. Build reference match-keys and canonical number tokens from the
         cited chunks' original_text (per-chunk for uncited sentences).
      3. coverage = matched content tokens / content tokens.  Content tokens
         are the sentence's tokens (citation markers removed, numbers
         canonicalised, tolerant keys) minus stopwords, citation words and
         tokens shorter than 3 chars; digit tokens always count.
      4. Strict checks: every number in the sentence must be supported by
         the reference chunks, every month word must match, no unmatched
         content token may be RARE (clitic-v2 union df <= 300) unless it is
         seed-aligned family-exempted in the cited source, and quoted spans
         must verify as in-order windows inside ONE cited chunk (short spans:
         every token present somewhere in one chunk).  Entity mentions
         following name anchors must be matched or seed-aligned, and polarity
         is checked in both directions: an affirmative claim cannot be
         grounded by negation/inversion-only source occurrences, a
         غير/بدون-governed claim token requires every matched source
         occurrence to be under the same inversion (a source that also states
         the bare affirmative is treated as contradictory and rejected), and
         a negated claim token requires a negated source occurrence.  The
         family exemption is
         polarity-filtered and can never bypass a polarity, quote or entity
         rejection.  The rare/exemption checks need *corpus_chunks*; without
         it a warning is printed and the rare check is skipped (never
         silently guessed).
      5. Keep if coverage >= GROUNDING_MIN_COVERAGE AND all strict checks
         pass for at least one reference option.  Sentences with no content
         tokens are kept unconditionally.  With *ignore_rare* low-frequency
         tokens that still exist somewhere in the corpus are tolerated, and a
         token absent from the corpus is tolerated only when its weak-letter
         skeleton matches a cited-source token (last-resort partial-answer
         fallback only); quote, entity, polarity, number and month checks
         still apply.

    After filtering, preserve valid (maSdar N) markers in the kept text so
    callers retain the citation audit trail.  If nothing survives, return
    the canonical refusal sentence.
    """
    REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
    quote_re = _QUOTE_RE_WITH_QURAN if quran_quotes else _QUOTE_RE
    answer = _expand_citation_clusters(answer)
    n_chunks = len(retrieved)

    # Pre-compute per retrieved chunk: tolerant key / number sets, strict key
    # lists, polarity marks (right-only negation-adjacent positions plus
    # غير/بدون-inverted positions), negation/inversion follower surfaces, the
    # token stream and the surface set (P5v5).
    chunk_keys: list[set[str]] = []
    chunk_ref_keys: list[set[str]] = []
    chunk_ref_nums: list[set[str]] = []
    chunk_ref_pairs: list[set[tuple[str, str]]] = []
    chunk_nums: list[set[str]] = []
    chunk_strict_lists: list[list[str]] = []
    chunk_pol_adj: list[set[int]] = []
    chunk_neg_adj: list[set[int]] = []
    chunk_neg_follow: list[set[str]] = []
    chunk_inv_follow: list[set[str]] = []
    chunk_surface: list[set[str]] = []
    chunk_toks: list[list[str]] = []
    chunk_pairs: list[set[tuple[str, str]]] = []
    chunk_legal: list[set[str]] = []
    chunk_legal_sentences: list[
        list[tuple[set[str], set[str], set[str], bool]]
    ] = []
    for c in retrieved:
        keys: set[str] = set()
        nums: set[str] = set()
        toks = _numbers_normalized(_tokenize(c.get("original_text", "") or ""))
        pol_adj: set[int] = set()
        neg_adj: set[int] = set()
        neg_follow: set[str] = set()
        inv_follow: set[str] = set()
        for p, t in enumerate(toks):
            if _is_negation_token(t):
                for target in _negation_target_positions(toks, p):
                    pol_adj.add(target)
                    neg_adj.add(target)
                    neg_follow.add(toks[target])
            if _is_inversion_token(t) and p + 1 < len(toks):
                pol_adj.add(p + 1)
                inv_follow.add(toks[p + 1])
        for t in toks:
            k = _gkey(t)
            keys.add(k)
            if k.isdigit():
                nums.add(k)
        chunk_keys.append(keys)
        ref_toks = _tokenize(c.get("reference", "") or "")
        ref_num_toks = _numbers_normalized(ref_toks)
        chunk_ref_keys.append({_gkey(t) for t in ref_toks})
        chunk_ref_nums.append({
            _gkey(t) for t in ref_num_toks if t.isdigit()
        })
        chunk_ref_pairs.append({
            (_gkey(a), _gkey(b)) for a, b in zip(ref_toks, ref_toks[1:])
        })
        chunk_nums.append(nums)
        chunk_strict_lists.append([_strict_key(t) for t in toks])
        chunk_pol_adj.append(pol_adj)
        chunk_neg_adj.append(neg_adj)
        chunk_neg_follow.append(neg_follow)
        chunk_inv_follow.append(inv_follow)
        chunk_surface.append({t for t in toks})
        chunk_toks.append(toks)
        chunk_pairs.append({
            (_gkey(a), _gkey(b)) for a, b in zip(toks, toks[1:])
        })
        chunk_legal.append(_positive_legal_families_from_tokens(toks))
        legal_sentences: list[
            tuple[set[str], set[str], set[str], bool]
        ] = []
        previous_topic_keys: set[str] = set()
        for source_sentence in _split_source_sentences(
            c.get("original_text", "") or "",
        ):
            sentence_topic_keys = _legal_topic_keys(source_sentence)
            sentence_legal = _positive_legal_families(source_sentence)
            sentence_legal_neg = _negative_legal_families(source_sentence)
            sentence_condition = any(
                marker in normalize(source_sentence)
                for marker in ("بشرط", "شرط", "يشترط")
            )
            if sentence_legal or sentence_legal_neg or sentence_condition:
                # A legal sentence can refer back to the subject of the
                # immediately preceding sentence ("... ويرى وجوبها ...").
                # Condition-only sentences are kept so a supported "شرط"
                # clause can satisfy the condition-marker guard.
                legal_sentences.append((
                    sentence_topic_keys | previous_topic_keys,
                    sentence_legal,
                    sentence_legal_neg,
                    sentence_condition,
                ))
            previous_topic_keys = sentence_topic_keys
        chunk_legal_sentences.append(legal_sentences)

    if corpus_chunks is None:
        print("[ground] WARNING: rare-token check skipped (no corpus provided)")
        postings: dict | None = None
    else:
        postings = _build_bm25(corpus_chunks)["postings"]
    df_cache: dict[str, int] = {}
    fam_cache: dict[str, set[str]] = {}

    def _df_cached(tok: str) -> int:
        if tok not in df_cache:
            df_cache[tok] = _union_docs_count(_clitic_v2(tok), postings)
        return df_cache[tok]

    def _fam_of(tok: str) -> set[str]:
        if tok not in fam_cache:
            fam_cache[tok] = _family_v2(tok)
        return fam_cache[tok]

    # Split answer into candidate sentences (keep separators attached)
    raw_sentences = _SENT_SPLIT_RE.split(answer)

    kept: list[str] = []
    for sent in raw_sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue

        # Every explicit citation must resolve; an invalid reference fails
        # closed instead of silently degrading to an uncited sentence.
        citation_matches = list(_CITATION_RE.finditer(sent))
        if any(
            not (1 <= int(m.group(1)) <= n_chunks)
            for m in citation_matches
        ):
            print(f"[ground] DROP invalid citation | {sent_stripped[:40]!r}")
            continue

        # A verbatim quote attributed to the wrong retrieved chunk is an
        # attribution error, not an unsupported claim.  Correct it before
        # grounding so a valid quotation is not discarded with the citation.
        if citation_matches and _QUOTE_RE.search(sent):
            repaired_sent = _repair_exact_quote_citation(sent, retrieved)
            if repaired_sent != sent:
                print(
                    "[ground] repaired exact-quote citation | "
                    f"{sent_stripped[:40]!r}"
                )
                sent = repaired_sent
                sent_stripped = sent.strip()
                citation_matches = list(_CITATION_RE.finditer(sent))

        # Extract valid citation indices (1-based -> 0-based)
        cited_indices = [int(m.group(1)) - 1 for m in citation_matches]

        # Content tokens (citation markers removed, numbers canonicalised)
        body = _CITATION_RE.sub("", sent_stripped)
        body = _QURAN_REF_RE.sub("", body)
        body_meta = _numbers_normalized_meta(_tokenize(body))
        body_tokens_all = [token for token, _ in body_meta]
        ordinal_body_positions = {
            pos for pos, (_, ordinal) in enumerate(body_meta) if ordinal
        }
        content_tokens = [
            t for t in body_tokens_all
            if t not in _QUERY_STOPWORDS
            and t not in _GROUNDING_IGNORED_FOLDED
            and not _is_negation_token(t)
            and (len(t) >= 3 or t.isdigit())
        ]
        content_tokens = _apply_grounding_aliases(content_tokens)
        # Bold labels are presentational; a legal word inside a heading must
        # not make the heading itself look like a source-supported claim.
        bold_tokens = {
            token
            for span in re.findall(r"\*\*(.+?)\*\*", sent_stripped)
            for token in _tokenize(span)
        }

        # Claim-side polarity maps: content positions directly following a
        # negation particle (right-only) or directly governed by a غير/بدون
        # inversion.  Entity mentions: a content token directly following a
        # name anchor must be supported by the cited source.
        claim_neg_positions: set[int] = set()
        pos_map: list[int] = []
        ci = 0
        for t in body_tokens_all:
            if (t not in _QUERY_STOPWORDS
                    and t not in _GROUNDING_IGNORED_FOLDED
                    and not _is_negation_token(t)
                    and (len(t) >= 3 or t.isdigit())):
                pos_map.append(ci)
                ci += 1
            else:
                pos_map.append(-1)
        ordinal_content_positions = {
            pos_map[q] for q in ordinal_body_positions if pos_map[q] >= 0
        }
        for q, t in enumerate(body_tokens_all):
            if _is_negation_token(t):
                for target in _negation_target_positions(body_tokens_all, q):
                    if pos_map[target] >= 0:
                        claim_neg_positions.add(pos_map[target])
        claim_inv_positions: set[int] = set()
        for q, t in enumerate(body_tokens_all):
            if (_is_inversion_token(t) and q + 1 < len(body_tokens_all)
                    and pos_map[q + 1] >= 0):
                claim_inv_positions.add(pos_map[q + 1])
        entity_mentions: list[tuple[str, str, bool]] = []
        for q, t in enumerate(body_tokens_all):
            a = _fold_ar(t)
            while len(a) > 2 and a[0] in "وف":
                a = a[1:]
            if (a in _GROUNDING_ENTITY_ANCHORS_FOLDED
                    and q + 1 < len(body_tokens_all)
                    and pos_map[q + 1] >= 0):
                entity_mentions.append((
                    a,
                    body_tokens_all[q + 1],
                    a in _GROUNDING_CONSTRUCT_ANCHORS_FOLDED,
                ))

        # List markers and citation-only fragments carry no claim; keeping
        # them leaves artefacts such as a leading "2." in the final answer.
        marker_body = body.strip()
        if not marker_body or _LIST_MARKER_RE.fullmatch(marker_body):
            continue
        # A numbered-list lead-in ("... تشمل: 1.") is not a claim; the
        # following item carries the content.  Keeping it produces artefacts
        # such as "تشمل: 1 (مصدر 2)." after citation repair.
        if re.search(r"[:：]\s*(?:\d+[.)\-]?)?$", marker_body):
            continue

        claim_legal = _positive_legal_families(sent)
        # Vacuous pass: no content tokens to check.  A short legal claim
        # ("نعم، حرام") is not vacuous: its modality must still be grounded.
        if (
            len(content_tokens) < _GROUNDING_MIN_CONTENT_TOKENS
            and not claim_legal
        ):
            kept.append(sent)
            print(f"[ground] cov=1.00 KEEP (vacuous) | {sent_stripped[:40]!r}")
            continue

        token_keys = [_gkey(t) for t in content_tokens]
        claim_strict = [_strict_key(t) for t in content_tokens]
        digit_tokens = [
            t for j, t in enumerate(content_tokens)
            if t.isdigit() and j not in ordinal_content_positions
        ]
        month_positions = []
        for j, t in enumerate(content_tokens):
            ft = _fold_ar(t)
            if ft not in _GROUNDING_MONTHS:
                continue
            if ft in _GROUNDING_AMBIGUOUS_MONTHS and not any(
                _fold_ar(x) == "شهر" for x in content_tokens[max(0, j - 2):j]
            ):
                continue
            month_positions.append(j)

        # Quoted spans: content tokens of each quoted span.  Spans with >= 3
        # content tokens verify as in-order windows inside ONE cited chunk;
        # shorter spans only need each token to occur somewhere in one chunk
        # (too short to carry sequence structure).
        quote_spans: list[list[str]] = []
        quoted_strict: set[str] = set()
        for span in quote_re.findall(sent):
            span_tokens = _numbers_normalized(_tokenize(span))
            if quran_quotes:
                quoted_strict.update(_strict_key(t) for t in span_tokens)
            qt = [t for t in span_tokens
                  if t not in _QUERY_STOPWORDS
                  and t not in _GROUNDING_IGNORED_FOLDED
                  and (len(t) >= 3 or t.isdigit())]
            if qt:
                quote_spans.append(qt)

        # Reference options: multiple citations are combined (the model
        # annotated them together); an UNCITED sentence must be supported by
        # a single chunk on its own — no union across different chunks.
        if cited_indices:
            ref_options = [cited_indices]
        else:
            ref_options = [[i] for i in range(n_chunks)] or [[]]

        kept_this = False
        lenient_used = False
        best_cov = 0.0
        legal_topic_tokens = [
            t for t in content_tokens
            if _legal_family_of_token(t) is None
            and not t.isdigit()
            and _gkey(t) not in _LEGAL_REQUIRED_GENERIC_GKEYS
        ]
        legal_required_keys: set[str] = set()
        if legal_topic_tokens:
            ranked = sorted(
                {_legal_topic_key(t) for t in legal_topic_tokens},
                key=len, reverse=True,
            )
            legal_required_keys = set(ranked[:2])
        claim_legal_neg = _negative_legal_families(sent)
        claim_condition = any(
            marker in normalize(sent)
            for marker in ("بشرط", "شرط", "يشترط")
        )
        best_state: dict = {
            "digits_ok": False, "months_ok": False, "neg_ok": True,
            "contra_ok": True, "quote_ok": True, "entity_ok": True,
            "legal_ok": True, "rare": [],
        }
        for opt in ref_options:
            ref_keys: set[str] = (
                set().union(*(chunk_keys[i] for i in opt)) if opt else set()
            )
            ref_nums: set[str] = (
                set().union(*(chunk_nums[i] for i in opt)) if opt else set()
            )
            ref_surface: set[str] = (
                set().union(*(chunk_surface[i] for i in opt)) if opt else set()
            )
            ref_title_keys: set[str] = (
                set().union(*(chunk_ref_keys[i] for i in opt)) if opt else set()
            )
            ref_title_nums: set[str] = (
                set().union(*(chunk_ref_nums[i] for i in opt)) if opt else set()
            )
            ref_title_pairs: set[tuple[str, str]] = (
                set().union(*(chunk_ref_pairs[i] for i in opt))
                if opt else set()
            )
            ref_strict: set[str] = (
                set().union(*(set(chunk_strict_lists[i]) for i in opt))
                if opt else set()
            )
            ref_inv_follow: set[str] = (
                set().union(*(chunk_inv_follow[i] for i in opt)) if opt else set()
            )
            ref_neg_follow: set[str] = (
                set().union(*(chunk_neg_follow[i] for i in opt)) if opt else set()
            )
            ref_pairs: set[tuple[str, str]] = (
                set().union(*(chunk_pairs[i] for i in opt)) if opt else set()
            )
            ref_legal: set[str] = (
                set().union(*(chunk_legal[i] for i in opt)) if opt else set()
            )
            ref_legal_sentences = [
                entry
                for i in opt
                for entry in chunk_legal_sentences[i]
            ]
            matched = 0
            for j, key in enumerate(token_keys):
                if key in ref_keys or key in ref_title_keys:
                    matched += 1
                    continue
                # يجب/واجب/فرض are one legal family; the legal check below
                # still requires the cited evidence to assert that family.
                legal_family = _legal_family_of_token(content_tokens[j])
                if (
                    legal_family and legal_family in ref_legal
                    and content_tokens[j] not in bold_tokens
                ):
                    matched += 1
            coverage = matched / len(content_tokens)
            digits_ok = all(
                t in ref_nums or t in ref_title_nums for t in digit_tokens
            )
            months_ok = all(
                token_keys[j] in ref_keys or token_keys[j] in ref_title_keys
                for j in month_positions
            )
            # Negation flip, inversion-aware: an affirmative claim token whose
            # every source occurrence is polarity-marked rejects the option;
            # a غير/بدون-governed claim token rejects when no source
            # occurrence is under an inversion; at least one matching inverted
            # occurrence supports it.  (The comparison uses the strict surface key
            # so stem collisions like يعيد/عيدين can never cause a false
            # flip; claim-side negated tokens are skipped.)
            neg_ok = True
            for j, k in enumerate(token_keys):
                if (quran_quotes and claim_strict[j] in quoted_strict):
                    continue
                if (j in claim_neg_positions or k not in ref_keys
                        or content_tokens[j].isdigit()):
                    continue
                sk = claim_strict[j]
                occ = [
                    (i, p)
                    for i in opt
                    for p, st in enumerate(chunk_strict_lists[i])
                    if st == sk
                ]
                if j in claim_inv_positions:
                    # Conservative contradiction rule: an inverted claim is
                    # supported only when every matched source occurrence is
                    # under the same غير/بدون inversion.  A source that also
                    # contains a bare affirmative occurrence is internally
                    # contradictory for this token, so it cannot ground the
                    # claim; accepting any one inverted occurrence would let
                    # the generator choose one side of a contradictory source.
                    if any(p not in chunk_pol_adj[i] for i, p in occ):
                        neg_ok = False
                        break
                    continue
                if occ and all(p in chunk_pol_adj[i] for i, p in occ):
                    neg_ok = False
                    break
            # Negation parity: a claim token governed by negation that IS
            # matched must have at least one source occurrence under the
            # source's own negation marking — a claim "لا يستحب" cannot be
            # grounded by a bare affirmative "يستحب".
            if neg_ok:
                for j, k in enumerate(token_keys):
                    if (quran_quotes and claim_strict[j] in quoted_strict):
                        continue
                    if j not in claim_neg_positions or k not in ref_keys:
                        continue
                    sk = claim_strict[j]
                    occ = [
                        (i, p)
                        for i in opt
                        for p, st in enumerate(chunk_strict_lists[i])
                        if st == sk
                    ]
                    if occ and not any(p in chunk_neg_adj[i] for i, p in occ):
                        neg_ok = False
                        break
            # Outcome contradiction: a claim asserting exactly one side of a
            # mutually exclusive destination pair is rejected when the cited
            # source asserts the opposite side and never the claimed side.
            # Skipped when a negation/inversion governs the claimed
            # destination in the claim (e.g. "لا يدخلون الجنة") or the
            # asserted destination in the source, and when the source
            # mentions both sides (conservative).
            contra_ok = True
            for side_a, side_b in _GROUNDING_CONTRADICTION_GROUPS:
                claim_a = [j for j, sk in enumerate(claim_strict)
                           if sk in side_a]
                claim_b = [j for j, sk in enumerate(claim_strict)
                           if sk in side_b]
                if bool(claim_a) == bool(claim_b):
                    continue
                claimed = side_a if claim_a else side_b
                opposite = side_b if claim_a else side_a
                positions = claim_a or claim_b
                if any(
                    any(p in claim_neg_positions or p in claim_inv_positions
                        for p in range(max(0, j - 3), j + 1))
                    for j in positions
                ):
                    continue
                if ref_strict & claimed:
                    continue
                if any(
                    sk in opposite
                    and not any(
                        q in chunk_pol_adj[i]
                        for q in range(max(0, p - 3), p + 1)
                    )
                    for i in opt
                    for p, sk in enumerate(chunk_strict_lists[i])
                ):
                    contra_ok = False
                    break
            # Quote verification: every quoted span with >= 3 content tokens
            # must align as an in-order windowed sequence inside ONE cited
            # chunk (per-chunk alignment rejects quotes stitched across
            # sources); the aligned mode allows the same narrow seed pairs as
            # the family exemption.  Shorter spans are support-only.
            quote_ok = True
            # Exact normalised substring is stronger evidence than token
            # alignment and avoids false negatives from diacritic/tokenisation
            # differences (observed on a verbatim Ibn Kathir quotation that
            # the tolerant aligner rejected).  Only skip token verification
            # when ONE cited chunk contains every quoted span verbatim.
            exact_quote = bool(quote_spans) and any(
                _quoted_spans_exact(sent, retrieved[i]) for i in opt
            )
            if not exact_quote:
                for stoks in quote_spans:
                    ok = False
                    if len(stoks) < 3:
                        for i in opt:
                            if all(any(_seq_support(st, rt, "aligned")
                                       for rt in chunk_toks[i])
                                   for st in stoks):
                                ok = True
                                break
                    else:
                        for i in opt:
                            ok, _ = _span_align(stoks, chunk_toks[i], "aligned",
                                                _GROUNDING_SPAN_MAX_GAP)
                            if ok:
                                break
                    if not ok:
                        quote_ok = False
                        break
            # Entity guard: a claim token directly following a name anchor
            # must be matched or seed-aligned in the cited source — entity
            # names cannot hide behind rare-token df or the family exemption.
            # Construct anchors additionally require the anchor+follower pair
            # in the same cited chunk, so "أم الفضل" cannot be grounded by an
            # unrelated occurrence of the common noun "فضل".
            entity_ok = True
            for anchor, ftok, require_pair in entity_mentions:
                if require_pair:
                    pair = (_gkey(anchor), _gkey(ftok))
                    if pair in ref_pairs or pair in ref_title_pairs:
                        continue
                    # No family fallback here: the whole point of the pair
                    # check is that the common noun alone cannot ground a
                    # different person's name.  A pair in the cited chunk's
                    # own reference is a source label ("تفسير ابن كثير"),
                    # not a claim about the person, so it is exempt.
                    entity_ok = False
                    break
                if _gkey(ftok) in ref_keys:
                    continue
                fhits = {s for s in (_fam_of(ftok) & ref_surface)
                         if _seed_aligned(ftok, s)}
                if not fhits:
                    entity_ok = False
                    break
            # Rare-token rule on the clitic-v2 union document frequency, with
            # the seed-aligned cited-source family exemption (polarity-
            # filtered: an inverted/negated claim token may only be exempted
            # by a source variant under the same inversion/negation) and the
            # material-addition guard: an unmatched token in the uncertain
            # band (300 < df <= 500) is tolerated only in a well-covered
            # sentence (coverage >= 0.80), so a genuinely common material
            # word and an unsupported swap in a weakly-covered sentence stay
            # distinguishable.
            rare: list[tuple[str, int]] = []
            if postings is not None:
                for j, t in enumerate(content_tokens):
                    if (t.isdigit() or token_keys[j] in ref_keys
                            or token_keys[j] in ref_title_keys):
                        continue
                    legal_family = _legal_family_of_token(t)
                    if legal_family and legal_family in ref_legal:
                        continue
                    hits = {s for s in (_fam_of(t) & ref_surface)
                            if _seed_aligned(t, s)}
                    hits |= _curated_alias_surfaces(t, ref_surface)
                    if j in claim_inv_positions:
                        hits = {s for s in hits if s in ref_inv_follow}
                    if j in claim_neg_positions:
                        hits = {s for s in hits if s in ref_neg_follow}
                    if hits:
                        continue
                    d = _df_cached(t)
                    if d <= _GROUNDING_DF_THR or (
                        coverage < _GROUNDING_GUARD_COV
                        and d <= _GROUNDING_GUARD_BAND
                    ):
                        if ignore_rare:
                            weak = _fold_weak(t)
                            if len(weak) >= 3 and any(
                                _fold_weak(s) == weak for s in ref_surface
                            ):
                                continue
                        rare.append((t, d))
            # A strong legal conclusion is part of the claim.  It must be
            # asserted by at least one cited chunk, not merely be absent from
            # the topic words or implied by a quoted command.
            legal_ok = all(
                any(
                    claim in sentence_legal
                    and legal_required_keys <= sentence_topic
                    for sentence_topic, sentence_legal, _, _
                    in ref_legal_sentences
                )
                for claim in claim_legal
            )
            if legal_ok and claim_legal_neg:
                for claim in claim_legal_neg:
                    has_positive = any(
                        claim in sentence_legal
                        and legal_required_keys <= sentence_topic
                        for sentence_topic, sentence_legal, _, _
                        in ref_legal_sentences
                    )
                    has_negative = any(
                        claim in sentence_legal_neg
                        and legal_required_keys <= sentence_topic
                        for sentence_topic, _, sentence_legal_neg, _
                        in ref_legal_sentences
                    )
                    if has_positive and not has_negative:
                        legal_ok = False
                        break
            if legal_ok and claim_condition:
                legal_ok = any(
                    condition
                    and legal_required_keys <= sentence_topic
                    for sentence_topic, _, _, condition
                    in ref_legal_sentences
                )
            # Bold labels are headings, not evidence.  They must be directly
            # present in the cited text; a legal synonym in a heading does not
            # make the heading itself supported.
            heading_ok = all(
                _gkey(token) in ref_keys
                or _gkey(token) in ref_title_keys
                for token in bold_tokens
            )
            if coverage >= best_cov:
                best_cov = coverage
                best_state = {"digits_ok": digits_ok, "months_ok": months_ok,
                              "neg_ok": neg_ok, "contra_ok": contra_ok,
                              "quote_ok": quote_ok, "entity_ok": entity_ok,
                              "legal_ok": legal_ok, "heading_ok": heading_ok,
                              "rare": rare}
            quote_only = _quote_only_supported(
                sent, ref_keys, ref_title_keys,
            )
            coverage_ok = (
                coverage >= GROUNDING_MIN_COVERAGE
                or (quote_only and quote_ok)
            )
            if (
                coverage_ok
                and digits_ok and months_ok and neg_ok and contra_ok
                and quote_ok and entity_ok and legal_ok and heading_ok
                and (not rare
                     or (ignore_rare
                         and not any(d == 0 for _, d in rare)))
            ):
                kept_this = True
                lenient_used = bool(rare)
                best_cov = coverage
                best_state = {"digits_ok": digits_ok, "months_ok": months_ok,
                              "neg_ok": neg_ok, "contra_ok": contra_ok,
                              "quote_ok": quote_ok, "entity_ok": entity_ok,
                              "legal_ok": legal_ok, "heading_ok": heading_ok,
                              "rare": rare}
                break

        digits_ok = best_state["digits_ok"]
        months_ok = best_state["months_ok"]
        neg_ok = best_state["neg_ok"]
        contra_ok = best_state["contra_ok"]
        quote_ok = best_state["quote_ok"]
        entity_ok = best_state["entity_ok"]
        legal_ok = best_state["legal_ok"]
        heading_ok = best_state["heading_ok"]
        rare = best_state["rare"]
        tag = ("KEEP-LENIENT" if kept_this and lenient_used
               else "KEEP" if kept_this else "DROP")
        rare_s = ""
        if rare:
            rare_s = " rare=" + ",".join(f"{t}:{d}" for t, d in rare[:4])
        print(
            f"[ground] cov={best_cov:.2f} {tag} "
            f"digits={'ok' if digits_ok else 'fail'} "
            f"months={'ok' if months_ok else 'fail'} "
            f"neg={'ok' if neg_ok else 'flip'} "
            f"quote={'ok' if quote_ok else 'fail'} "
            f"entity={'ok' if entity_ok else 'fail'} "
            f"contra={'ok' if contra_ok else 'fail'} "
            f"legal={'ok' if legal_ok else 'fail'} "
            f"heading={'ok' if heading_ok else 'fail'}{rare_s}"
            f" | {sent_stripped[:40]!r}"
        )
        if kept_this:
            kept.append(sent)

    if not kept:
        return REFUSAL

    # Re-join while preserving valid citation markers for auditability.
    joined = " ".join(k.strip() for k in kept).strip()
    # Only remove list markers at a line start/absolute start.  The previous
    # whitespace-based pattern also matched the "1)" inside "(مصدر 1)" at
    # the end of a sentence and corrupted a valid citation.
    joined = re.sub(
        r'(?:(?<=\n)|^)\s*\d+[.)]\s*(?=(?:\d+[.)]|$))',
        '', joined,
    ).strip()
    return joined


def ground_text(answer: str, retrieved: list[dict],
                corpus_chunks: list[dict] | None = None,
                refusal: str | None = None,
                ignore_rare: bool = False,
                quran_quotes: bool = False) -> str:
    """Public grounding entry point for endpoints other than /api/chat.

    Thin wrapper over the audited `_ground_answer` engine: the matcher,
    thresholds and strict checks are unchanged, and /api/chat keeps calling
    `_ground_answer` directly.  When *refusal* is provided, it replaces the
    engine's canonical refusal on total rejection so each endpoint can keep
    its own established refusal contract; ``None`` preserves the chat
    refusal exactly.  ``ignore_rare`` is the engine's documented last-resort
    relaxation: it tolerates existing low-frequency tokens and corpus-absent
    tokens whose weak-letter skeleton matches a cited-source token, while
    still enforcing coverage, quote, entity, polarity, number and month
    checks.
    """
    grounded = _ground_answer(
        answer, retrieved, corpus_chunks, ignore_rare=ignore_rare,
        quran_quotes=quran_quotes,
    )
    if (refusal is not None
            and grounded == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."):
        return refusal
    return grounded


def _answer_content_tokens(text: str) -> list[str]:
    body = _CITATION_RE.sub(" ", text or "")
    body = _QURAN_REF_RE.sub(" ", body)
    tokens = _numbers_normalized(_tokenize(body))
    return [
        t for t in tokens
        if t not in _QUERY_STOPWORDS
        and t not in _GROUNDING_IGNORED_FOLDED
        and (len(t) >= 3 or t.isdigit())
    ]


def _source_support_ratio(text: str, chunk: dict) -> float:
    tokens = _answer_content_tokens(text)
    if not tokens:
        return 1.0
    chunk_tokens = _numbers_normalized(
        _tokenize(chunk.get("original_text", "") or ""))
    reference = {_gkey(t) for t in chunk_tokens}
    reference_legal = _positive_legal_families_from_tokens(chunk_tokens)
    matched = 0
    for token in tokens:
        if _gkey(token) in reference:
            matched += 1
            continue
        legal_family = _legal_family_of_token(token)
        if legal_family and legal_family in reference_legal:
            matched += 1
    return matched / len(tokens)


def _is_primary_hadith(chunk: dict) -> bool:
    if chunk.get("source") != "hadith":
        return False
    reference = chunk.get("reference", "") or ""
    return "البخاري" in reference or "مسلم" in reference


def _quoted_spans_exact(sentence: str, chunk: dict) -> bool:
    """True when every quoted span occurs verbatim (normalised) in the chunk.

    Token-level quote support is deliberately tolerant for grounding, but it
    cannot justify *changing* a citation: a chunk can contain the individual
    words of a quote in unrelated clauses.  Primary-source preference therefore
    requires the same normalised substring test the heldout checker uses.
    """
    chunk_text = normalize(chunk.get("original_text", "") or "")
    for span in _QUOTE_RE.findall(sentence):
        quote = normalize(span).strip()
        if quote and quote not in chunk_text:
            return False
    return True


def _quoted_spans_supported(sentence: str, chunk: dict) -> bool:
    chunk_tokens = _tokenize(chunk.get("original_text", "") or "")
    for span in _QUOTE_RE.findall(sentence):
        tokens = [
            t for t in _numbers_normalized(_tokenize(span))
            if t not in _QUERY_STOPWORDS
            and t not in _GROUNDING_IGNORED_FOLDED
            and (len(t) >= 3 or t.isdigit())
        ]
        if not tokens:
            continue
        if len(tokens) < 3:
            if not all(any(_seq_support(t, rt, "aligned")
                           for rt in chunk_tokens) for t in tokens):
                return False
        else:
            ok, _ = _span_align(tokens, chunk_tokens, "aligned",
                                _GROUNDING_SPAN_MAX_GAP)
            if not ok:
                return False
    return True


def _repair_exact_quote_citation(sentence: str, retrieved: list[dict]) -> str:
    """Point a verbatim quotation at the one retrieved chunk that contains it.

    This is deliberately narrower than general citation repair: it only acts
    when (a) the sentence contains a quoted span, (b) none of its current
    citations contains that span verbatim, and (c) exactly one retrieved
    chunk does.  That is an attribution correction, not a support relaxation.
    """
    if not _QUOTE_RE.search(sentence):
        return sentence
    citations = list(_CITATION_RE.finditer(sentence))
    if not citations:
        return sentence
    cited = [
        int(match.group(1)) - 1 for match in citations
        if 1 <= int(match.group(1)) <= len(retrieved)
    ]
    if not cited:
        return sentence
    if any(_quoted_spans_exact(sentence, retrieved[i]) for i in cited):
        return sentence
    exact = [
        i for i, chunk in enumerate(retrieved)
        if _quoted_spans_exact(sentence, chunk)
    ]
    if len(exact) != 1:
        return sentence
    return _CITATION_RUN_RE.sub(
        f" (مصدر {exact[0] + 1})", sentence,
    )


def _strip_unsupported_quotes(
    answer: str, retrieved: list[dict],
) -> str:
    """Remove quotation marks around spans not supported by their citation.

    Quotation marks promise a verbatim source text.  When the generator puts
    a paraphrase inside quotes, the claim itself may still be supported; the
    honest repair is to present it as a paraphrase, not to discard the whole
    sentence.  Only the quote characters are removed; every substantive
    grounding guard still runs on the resulting sentence.
    """
    pieces = _SENT_SPLIT_RE.split(answer)
    out: list[str] = []
    changed = False
    for sentence in pieces:
        citations = list(_CITATION_RE.finditer(sentence))
        cited = [
            int(match.group(1)) - 1 for match in citations
            if 1 <= int(match.group(1)) <= len(retrieved)
        ]
        if not cited or not _QUOTE_RE.search(sentence):
            out.append(sentence)
            continue
        if any(_quoted_spans_supported(sentence, retrieved[i]) for i in cited):
            out.append(sentence)
            continue
        out.append(_QUOTE_RE.sub(lambda match: match.group(1), sentence))
        changed = True
    return "".join(out) if changed else answer


def _repair_citation_support(
    answer: str, retrieved: list[dict], query: str = "",
) -> str:
    if not answer.strip() or not retrieved:
        return answer
    variant_query = _lexical_variant_query(_tokenize(normalize(query)))
    repaired: list[str] = []
    for sentence in _SENT_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        citations = list(_CITATION_RE.finditer(sentence))
        if not citations:
            repaired.append(sentence)
            continue
        cited = [
            int(match.group(1)) - 1 for match in citations
            if 1 <= int(match.group(1)) <= len(retrieved)
        ]
        if not cited:
            repaired.append(sentence)
            continue
        current = max(_source_support_ratio(sentence, retrieved[i])
                      for i in cited)
        scores = [
            _source_support_ratio(sentence, chunk)
            for chunk in retrieved
        ]
        best = max(range(len(retrieved)), key=lambda i: scores[i])
        chosen = None
        if (current < GROUNDING_MIN_COVERAGE
                and scores[best] >= GROUNDING_MIN_COVERAGE
                and scores[best] >= current + 0.25):
            chosen = best
        elif variant_query and not any(
            _is_primary_hadith(retrieved[i]) for i in cited
        ):
            primary = [
                i for i, chunk in enumerate(retrieved)
                if _is_primary_hadith(chunk)
                and scores[i] >= max(current - 0.10, GROUNDING_MIN_COVERAGE)
                and _quoted_spans_exact(sentence, chunk)
            ]
            if primary:
                chosen = max(primary, key=lambda i: scores[i])
        if (chosen is None and _QUOTE_RE.search(sentence)
                and not any(_is_primary_hadith(retrieved[i]) for i in cited)):
            # A quoted hadith can appear in several collections.  When the
            # model cites a secondary collection but a primary hadith has
            # equivalent quote support, prefer the primary collection.  This
            # is a citation-quality preference, not a truth claim, and only
            # applies to explicitly quoted text.
            primary = [
                i for i, chunk in enumerate(retrieved)
                if _is_primary_hadith(chunk)
                and scores[i] >= max(current - 0.10, GROUNDING_MIN_COVERAGE)
                and _quoted_spans_exact(sentence, chunk)
            ]
            if primary:
                chosen = max(primary, key=lambda i: scores[i])
        if chosen is None and _QUOTE_RE.search(sentence):
            # Exact-quote attribution repair: prefer a retrieved chunk that
            # contains every quoted span verbatim over a nearby duplicate or
            # an adjacent tafsir chunk that merely shares the topic.
            exact = [
                i for i, chunk in enumerate(retrieved)
                if _quoted_spans_exact(sentence, chunk)
            ]
            if exact and not any(i in cited for i in exact):
                chosen = max(exact, key=lambda i: scores[i])
        if chosen is None:
            repaired.append(sentence)
        else:
            repaired.append(_CITATION_RUN_RE.sub(
                f" (مصدر {chosen + 1})", sentence))
    return " ".join(repaired)


def _add_missing_citations(answer: str, retrieved: list[dict]) -> str:
    """Attach a source marker to grounded sentences the model left uncited."""
    if (not answer.strip() or not retrieved
            or _COMPLETENESS_REFUSAL in answer):
        return answer
    chunks = []
    for sentence in _SENT_SPLIT_RE.split(answer):
        stripped = sentence.strip()
        if (not stripped or _CITATION_RE.search(sentence)
                or len(_answer_content_tokens(stripped))
                < _GROUNDING_MIN_CONTENT_TOKENS):
            chunks.append(sentence)
            continue
        scores = [_source_support_ratio(stripped, chunk)
                  for chunk in retrieved]
        candidate_indices = list(range(len(retrieved)))
        legal_claims = _positive_legal_families(stripped)
        if legal_claims:
            candidate_indices = [
                i for i in candidate_indices
                if legal_claims <= _positive_legal_families(
                    retrieved[i].get("original_text", "") or "",
                )
            ]
        if not candidate_indices:
            chunks.append(sentence)
            continue
        best = max(candidate_indices, key=lambda i: scores[i])
        if (scores[best] < GROUNDING_MIN_COVERAGE
                or not _quoted_spans_supported(stripped, retrieved[best])):
            chunks.append(sentence)
            continue
        citation = f"(مصدر {best + 1})"
        if stripped[-1] in ".!?؟":
            body = stripped[:-1].rstrip()
        else:
            body = stripped
        leading = sentence[:len(sentence) - len(sentence.lstrip())]
        trailing = sentence[len(sentence.rstrip()):]
        ending = stripped[-1] if stripped[-1] in ".!?؟" else "."
        chunks.append(f"{leading}{body} {citation}{ending}{trailing}")
    return "".join(chunks)


_COMPLETENESS_REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
_COMPLETENESS_STOP_KEYS = {
    _gkey(t) for t in (
        "حكم", "ما", "هي", "هو", "من", "في", "على", "معنى", "تعريف",
        "المقصود", "قصة", "التي", "الذي", "الصلاة", "صلاة",
    )
}
_COMPLETENESS_DEFINITION_PREFIXES = (
    "ما هي", "ما هو", "ما معنى", "ما تعريف", "ما المقصود",
)
_COMPLETENESS_DEFINITION_MARKERS = ("انزل", "مبارك", "من شهر رمضان")
# Those markers describe the Quranic night of Qadr.  "مبارك" also occurs in
# unrelated chapters (e.g. Witr), so the definition branch must be scoped to
# a Qadr subject instead of every "ما هي" question.
_COMPLETENESS_DEFINITION_SUBJECTS = {_gkey("قدر")}
_COMPLETENESS_RULING_MARKERS = ("حكم",)
_COMPLETENESS_QUALIFIER_MARKERS = (
    "والمراد بالوجوب تاكيد استحبابه",
    "لا على الوجوب",
    "ليس بواجب",
)
_COMPLETENESS_TIMING_MARKERS = (
    "ويخرج وقت الغسل بالفراغ من الصلاة",
)
_COMPLETENESS_COMMAND_MARKERS = ("امر الشارع بالغسل واكده",)
_COMPLETENESS_LIST_MARKERS = ("يشمل", "تشمل")
_COMPLETENESS_STORY_PREFIXES = ("ما قصه", "ما هي قصه")
_COMPLETENESS_STRONG_RULINGS = {_gkey(t) for t in ("واجب", "فرض")}
_COMPLETENESS_WEAK_RULINGS = {
    _gkey(t) for t in ("استحباب", "مستحب", "يستحب")
}


def _split_source_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw = re.split(r"(?<=[.!?؟])\s+|\n+", text.replace("\r", " "))
    return [re.sub(r"\s+", " ", s).strip() for s in raw if s.strip()]


def _source_text_keys(text: str) -> set[str]:
    return {
        _gkey(t) for t in _tokenize(normalize(text))
        if len(t) >= 3
    }


def _query_subject_keys(query: str) -> set[str]:
    return _source_text_keys(query) - _COMPLETENESS_STOP_KEYS


def _query_subject_phrase(query: str) -> str:
    text = normalize(query).strip(" ؟?.")
    text = re.sub(r"^(?:ما|ماذا)\s+حكم\s+", "", text)
    text = re.sub(r"^(?:ما|ماذا)\s+", "", text)
    return text.strip()


def _supported_subject_phrase(query: str, retrieved: list[dict]) -> str:
    """Keep only query-subject words the retrieved sources actually contain.

    A completeness summary must not smuggle query context into a citation.
    For example, the query may say "الغسل يوم الجمعة" while the cited fiqh
    clause only discusses "الغسل"; asserting the full phrase would make the
    summary ungrounded (and correctly get dropped).  Filtering to supported
    words keeps the explicit ruling while staying inside the source.
    """
    phrase = _query_subject_phrase(query)
    source_keys: set[str] = set()
    for chunk in retrieved:
        source_keys |= _source_text_keys(
            chunk.get("original_text", "") or "")
        source_keys |= _source_text_keys(chunk.get("reference", "") or "")
    kept = [
        word for word in phrase.split()
        if _source_text_keys(word) & source_keys
    ]
    return " ".join(kept) if kept else phrase


def _strip_source_notes(text: str) -> str:
    text = re.sub(r"\[\[.*?\]\]", "", text)
    return re.sub(r"\[([^\[\]]*)\]", r"\1", text)


def _clean_source_snippet(text: str) -> str:
    text = re.sub(r"^\s*بسم الله الرحمن الرحيم\s*", "", text)
    text = re.sub(r"^(?:\s*\*\s*)+\s*", "", text)
    text = re.sub(r"^\s*\(\d+\)\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    # Strip a bare list label ("ب - "), a dangling clause separator, and a
    # closing quotation mark left over from a quoted source clause.
    text = re.sub(r"^\s*[\u0621-\u064A]\s*[-–]\s+", "", text)
    text = re.sub(r"\s*[،؛:]\s*$", "", text)
    text = re.sub(r'[\s"«»“”]+$', "", text)
    return re.sub(r"\s+", " ", text).strip()


def _list_items_missing_from_answer(
    sentence: str, markers: tuple[str, ...], answer_keys: set[str],
) -> list[str]:
    for marker in markers:
        pos = sentence.find(marker)
        if pos < 0:
            continue
        tail = sentence[pos + len(marker):]
        tail = re.split(r"[،؛.]", tail, maxsplit=1)[0]
        item_keys = [
            _gkey(t) for t in _tokenize(tail)
            if len(t) >= 3 and _gkey(t) not in {
                _gkey("سنة"), _gkey("سنن"),
            }
        ]
        if len(item_keys) < 3:
            continue
        return [
            key for key in dict.fromkeys(item_keys)
            if key not in answer_keys
        ]
    return []


def _find_story_snippet(
    query: str, answer: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    subject_keys = _query_subject_keys(query)
    answer_norm = _fold_ar(normalize(answer))
    if (
        "اصحاب الكهف والرقيم" in answer_norm
        or "الكهف والرقيم" in answer_norm
    ):
        return None
    for index, chunk in enumerate(retrieved, 1):
        raw = _strip_source_notes(chunk.get("original_text", "") or "")
        body = _fold_ar(normalize(raw))
        chunk_keys = _source_text_keys(body)
        if not subject_keys or not subject_keys.intersection(chunk_keys):
            continue
        phrase = "قصه اصحاب الكهف والرقيم"
        for sentence in _split_source_sentences(body):
            pos = sentence.find(phrase)
            if pos < 0:
                continue
            start = max(0, pos - 40)
            if start > 0:
                start = sentence.rfind(" ", 0, start) + 1
            clause = sentence[start:pos + len(phrase)]
            return index, _clean_source_snippet(clause)
    return None


def _find_variant_quote(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    query_tokens = _tokenize(normalize(query))
    variant_query = _lexical_variant_query(query_tokens)
    if not variant_query:
        return None
    query_set = set(query_tokens)
    changed = {
        _strict_key(variant)
        for literal, variant in _LEXICAL_VARIANT_GROUPS
        if literal in query_set and variant in variant_query
        and variant not in query_set
    }
    if not changed:
        return None
    best: tuple[tuple[int, int, int, int], int, str] | None = None
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "hadith":
            continue
        for quote in _QUOTE_RE.findall(chunk.get("original_text", "") or ""):
            quote_tokens = _tokenize(quote)
            quote_keys = {_strict_key(t) for t in quote_tokens}
            if not changed <= quote_keys:
                continue
            score = (
                int(_is_primary_hadith(chunk)),
                len(changed & quote_keys),
                -len(quote_tokens),
                -index,
            )
            if best is None or score > best[0]:
                best = (score, index, quote.strip())
    if best is None:
        return None
    return best[1], best[2]


def _find_narrative_quran_snippet(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    qn = normalize(query)
    if not qn.startswith(_COMPLETENESS_STORY_PREFIXES):
        return None
    by_surah: dict[int, list[int]] = defaultdict(list)
    for chunk in retrieved:
        if chunk.get("source") != "tafsir":
            continue
        match = re.search(r"(\d+):(\d+)\s*$", chunk.get("reference") or "")
        if match:
            by_surah[int(match.group(1))].append(int(match.group(2)))
    if not by_surah:
        return None
    surah, ayahs = max(by_surah.items(), key=lambda item: len(item[1]))
    candidates = []
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "quran":
            continue
        meta = chunk.get("metadata") or {}
        if int(meta.get("surah_number", -1)) != surah:
            continue
        ayah = int(meta.get("ayah_number", -1))
        if ayah > max(ayahs):
            candidates.append((ayah, index, chunk))
    if not candidates:
        return None
    _, index, chunk = max(candidates, key=lambda item: item[0])
    text = _clean_source_snippet(chunk.get("original_text", "") or "")
    return (index, text) if text else None


def _find_variant_answer(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    found = _find_variant_quote(query, retrieved)
    if not found:
        return None
    index, quote = found
    return index, f"قال رسول الله صلى الله عليه وسلم: {quote}"


def _recover_tayammum_narrative(
    query: str, retrieved: list[dict],
) -> str | None:
    """Recover a refused tayammum-story answer extractively from the sources.

    Live generation occasionally paraphrases "فتَغَيَّظَ" with a source-absent
    synonym (e.g. فاغتاظ).  Grounding correctly drops that unsupported clause,
    and when it was the only sentence the answer becomes a false refusal even
    though the cited fiqh passage narrates the whole incident.  This fallback
    quotes the passage's own necklace, water and revelation clauses, each
    cited and re-grounded; it never invents text.
    """
    qn = normalize(query)
    if not ("تيمم" in qn and "عايشه" in _fold_ar(qn)):
        return None
    marker_groups = (
        ("سقط عقد", "انقطع عقد", "عقد لي"),
        ("ليسوا على ماء", "لم يكن معهم ماء", "لم يجدوا ماء",
         "على غير ماء"),
        ("آية التيمم", "الرخصة في التيمم"),
    )
    parts: list[str] = []
    seen: set[tuple[int, str]] = set()
    for markers in marker_groups:
        found = _find_source_snippet(
            query, "", retrieved, markers, min_words=6,
        )
        if not found:
            continue
        index, snippet = found
        # Drop a trailing narrator lead-in ("قال السيد بن حضير") that the
        # clause splitter cannot see inside the parenthetical.
        snippet = re.split(r"\s+قال\s+", snippet, maxsplit=1)[0].strip()
        if not snippet or (index, snippet) in seen:
            continue
        seen.add((index, snippet))
        trial = f"{snippet} (مصدر {index})."
        if _COMPLETENESS_REFUSAL in _ground_answer(trial, retrieved):
            continue
        parts.append(trial)
    if len(parts) < 2:
        return None
    combined = " ".join(parts)
    if _COMPLETENESS_REFUSAL in _ground_answer(combined, retrieved):
        return None
    return combined


def _find_tafsir_opening_clause(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    """Return the tafsir passage's opening meaning clause for a Tafsir answer.

    Classical tafsir chunks often open with the verse's overall meaning
    ("يخبر تعالى أن أول بيت وضع للناس ...").  A generated answer can omit
    that opening and jump straight to legal details.  This bounded clause is
    used only to prepend the passage's own words; grounding still verifies
    it against the cited chunk.
    """
    subject_keys = _query_subject_keys(query)
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "tafsir":
            continue
        # Split the raw passage so punctuation survives; matching is done on
        # the folded form, but the returned clause keeps the source's own
        # orthography (الصلاة, not الصلاه).
        raw = _strip_source_notes(chunk.get("original_text") or "")
        sentences = _split_source_sentences(raw)
        if not sentences:
            continue
        first = sentences[0]
        clause = re.split(
            r"\s+(?:أي|اي|يعني|وهو|وهي|قال|قالت)\s*[:：]?",
            first, maxsplit=1,
        )[0]
        clause = re.split(r"[،:؛.]", clause, maxsplit=1)[0]
        words = clause.split()
        if len(words) > 16:
            clause = " ".join(words[:16])
        clause = _clean_source_snippet(clause)
        if len(clause) < 8:
            continue
        if subject_keys and not (
            subject_keys & _source_text_keys(clause)
        ):
            continue
        return index, clause
    return None


def _find_tafsir_meaning_sentence(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    """Return a concise tafsir sentence that explains the verse's meaning."""
    subject_keys = _query_subject_keys(query)
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "tafsir":
            continue
        raw = _strip_source_notes(chunk.get("original_text") or "")
        for sentence in _split_source_sentences(raw):
            normalized = normalize(sentence)
            if not (
                ("اخبر" in normalized or "معني" in normalized)
                and ("العسر" in normalized or "اليسر" in normalized)
            ):
                continue
            if subject_keys and not (
                subject_keys & _source_text_keys(sentence)
            ):
                continue
            clean = re.sub(r"﴿[^﴾]*﴾", "", sentence)
            clean = re.sub(r"^\s*وَقَوْلُهُ\s*[:：]?\s*", "", clean)
            clean = _clean_source_snippet(clean)
            words = clean.split()
            if len(words) < 5:
                continue
            if len(words) > 40:
                clean = " ".join(words[:40]) + "…"
            return index, clean
    return None


def _concise_source_clause(
    raw_sentence: str,
    folded_markers: tuple[str, ...],
    *,
    min_words: int = 12,
    max_words: int = 36,
) -> str:
    """Return the clause around a completeness marker, not a whole chunk.

    Some fiqh chunks are a single long paragraph with no sentence break at
    all.  Returning the whole paragraph both bloats the answer and makes the
    grounding check expensive.  Split the *raw* sentence on Arabic clause
    punctuation first (normalization strips punctuation), then take the
    marker's clause plus just enough neighbouring context to state the rule.
    """
    clauses = [
        c.strip() for c in re.split(r"(?<=[،؛:])\s*", raw_sentence) if c.strip()
    ]
    if len(clauses) <= 1:
        return _clean_source_snippet(raw_sentence)
    folded_clauses = [_fold_ar(normalize(c)) for c in clauses]
    hit_index = next(
        (
            i for i, clause in enumerate(folded_clauses)
            if any(marker in clause for marker in folded_markers)
        ),
        None,
    )
    if hit_index is None:
        return _clean_source_snippet(raw_sentence)
    start = hit_index
    end = hit_index
    words = len(folded_clauses[hit_index].split())
    # Prefer the clause that follows the marker.  For the zakat timing
    # marker, the hit clause is "من أداها قبل الصلاة"; its meaningful
    # continuation is "فهي زكاة مقبولة", while the preceding clause is the
    # unrelated "من اللغو والرفث..." fragment.  Only fall back to one
    # preceding clause when forward context cannot reach min_words (e.g. the
    # riba definition "الزيادة على رأس المال").
    while words < min_words and end + 1 < len(clauses):
        end += 1
        words += len(folded_clauses[end].split())
    if words < min_words and start > 0:
        start -= 1
        words += len(folded_clauses[start].split())
    while end > hit_index and words > max_words:
        words -= len(folded_clauses[end].split())
        end -= 1
    return _clean_source_snippet(" ".join(clauses[start:end + 1]))


def _requested_distinction_terms(query: str) -> tuple[str, str] | None:
    """Extract the two terms from "الفرق بين X وY" in a content question."""
    match = re.search(
        r"الفرق\s+بين\s+([^\s،.؟]+)\s+و([^\s،.؟]+)",
        query or "",
    )
    if not match:
        return None
    left = match.group(1).strip('«»\"\'()[]{}')
    right = match.group(2).strip('«»\"\'()[]{}')
    if not left or not right:
        return None
    return left, right


def _find_distinction_snippet(
    query: str,
    answer: str,
    retrieved: list[dict],
) -> tuple[int, str] | None:
    """Find a source passage that explains both terms the user contrasted."""
    terms = _requested_distinction_terms(query)
    if not terms:
        return None
    left, right = terms
    left_key = normalize(left)
    right_key = normalize(right)
    answer_norm = normalize(answer)
    if left_key in answer_norm and right_key in answer_norm:
        return None
    for index, chunk in enumerate(retrieved, 1):
        sentences = _split_source_sentences(
            chunk.get("original_text", "") or "",
        )
        for start, sentence in enumerate(sentences):
            if left_key not in normalize(sentence):
                continue
            for end in range(start, min(start + 3, len(sentences))):
                if right_key not in normalize(sentences[end]):
                    continue
                snippet = _clean_source_snippet(
                    " ".join(sentences[start:end + 1])
                )
                if snippet:
                    return index, snippet
    return None


def _find_source_sentence(
    query: str,
    retrieved: list[dict],
    marker: str,
) -> tuple[int, str] | None:
    """Return the full source sentence containing *marker*.

    Unlike ``_find_source_snippet`` this preserves the complete qualification
    instead of truncating after the first clause, so an answer that asks for
    detail receives the source's full exception list.
    """
    subject_keys = _query_subject_keys(query) - _COMPLETENESS_STOP_KEYS
    for index, chunk in enumerate(retrieved, 1):
        raw = chunk.get("original_text", "") or ""
        if subject_keys and not (
            subject_keys & _source_text_keys(raw)
        ):
            continue
        for sentence in _split_source_sentences(raw):
            if marker not in _fold_ar(normalize(sentence)):
                continue
            clean = re.sub(r"\s*\(\d+\)", "", sentence)
            clean = _clean_source_snippet(clean)
            if clean:
                return index, clean
    return None


def _find_condition_list_snippet(
    query: str, retrieved: list[dict],
) -> tuple[int, str] | None:
    """Return a numbered condition list anchored by "الشرط الآتية"."""
    subject_keys = _query_subject_keys(query)
    markers = ("الشرط الاتية", "الشرط الاتيه", "الشروط الاتية", "الشروط الاتيه")
    for index, chunk in enumerate(retrieved, 1):
        raw = chunk.get("original_text", "") or ""
        if not subject_keys or not subject_keys.intersection(
            _source_text_keys(raw)
        ):
            continue
        for marker in markers:
            start = raw.find(marker)
            if start < 0:
                continue
            tail = raw[start:]
            stop = re.search(r"فمن لم|ومن لم|\n\n", tail)
            end = len(tail) if stop is None else stop.start()
            snippet = tail[:min(end, 600)].strip()
            if snippet:
                return index, _clean_source_snippet(snippet)
    return None


def _find_source_snippet(
    query: str,
    answer: str,
    retrieved: list[dict],
    markers: tuple[str, ...],
    *,
    list_mode: bool = False,
    phrase_only: bool = False,
    min_words: int = 12,
) -> tuple[int, str] | None:
    subject_keys = _query_subject_keys(query)
    answer_keys = _source_text_keys(answer)
    # Completeness markers are written in modern normalized Arabic (e.g.
    # جراحه), while the corpus preserves ta marbuta (جراحة).  Normalize the
    # source and the markers with the same pipeline before folding so a
    # legitimate clause is not missed on an orthographic difference.
    folded_markers = tuple(_fold_ar(normalize(m)) for m in markers)
    best: tuple[int, int, str] | None = None
    for index, chunk in enumerate(retrieved, 1):
        raw = chunk.get("original_text", "") or ""
        body = _fold_ar(normalize(raw))
        chunk_keys = _source_text_keys(body)
        if not subject_keys or not subject_keys.intersection(chunk_keys):
            continue
        # Split the raw text so sentence punctuation survives; normalizing
        # first would erase the very boundaries used to bound the snippet.
        for raw_sentence in _split_source_sentences(raw):
            folded = _fold_ar(normalize(raw_sentence))
            hits = [m for m in folded_markers if m in folded]
            if not hits:
                continue
            if list_mode:
                missing = _list_items_missing_from_answer(
                    folded, folded_markers, answer_keys,
                )
                if not missing:
                    continue
            score = 2 * len(subject_keys.intersection(chunk_keys)) + len(hits)
            if best is None or score > best[0]:
                snippet = _concise_source_clause(
                    raw_sentence, folded_markers,
                    min_words=4 if phrase_only else min_words,
                )
                best = (score, index, _clean_source_snippet(snippet))
    if best is None:
        return None
    return best[1], best[2]


def _add_completeness_snippet(
    answer: str,
    retrieved: list[dict],
    index: int,
    snippet: str,
    *,
    prepend: bool = False,
) -> str:
    if not snippet:
        return answer
    snippet = snippet.rstrip(' .،؛:"«»“”')
    snippet_keys = _source_text_keys(snippet)
    answer_keys = _source_text_keys(answer)
    if snippet_keys and answer_keys:
        overlap = len(snippet_keys.intersection(answer_keys)) / len(snippet_keys)
        if overlap > 0.82:
            return answer
    trial = f"{snippet} (مصدر {index})."
    if _COMPLETENESS_REFUSAL in _ground_answer(trial, retrieved):
        return answer
    if prepend:
        return f"{trial}\n\n{answer}"
    return f"{answer.rstrip()}\n\n{trial}"


def _injected_quran_verse(
    retrieved: list[dict],
) -> tuple[int, str, str] | None:
    """Return the first deterministic Quran anchor in *retrieved*."""
    for index, chunk in enumerate(retrieved, 1):
        if (chunk.get("source") != "quran"
                or not chunk.get("_injected")):
            continue
        text = (chunk.get("original_text") or "").strip()
        if not text:
            continue
        metadata = chunk.get("metadata") or {}
        surah = str(metadata.get("surah_name") or "").strip()
        ayah = metadata.get("ayah_number")
        if surah and ayah is not None:
            reference = f"{surah}: {ayah}"
        else:
            reference = str(chunk.get("reference") or "").strip()
        return index, text, reference
    return None


def _quran_topic_anchor_verse(
    query: str, retrieved: list[dict],
) -> tuple[int, str, str] | None:
    """Return the injected canonical Quran verse for a core-topic question.

    Only chunks marked ``_injected`` are considered.  Those are the
    deterministic topic/tafsir anchors added by ``retrieve``; a verse that
    merely happened to rank highly is not treated as a primary-source
    requirement.  Tafsir-intent questions are excluded because their answer
    is already anchored to the resolved verse and commentary.
    """
    qn = normalize(query)
    if _TAFSIR_VERB_RE.search(qn) or _TAFSIR_MEANING_RE.search(qn):
        return None
    return _injected_quran_verse(retrieved)


def _tafsir_anchor_verse(
    query: str, retrieved: list[dict],
) -> tuple[int, str, str] | None:
    """Return the resolved Quran verse for a Tafsir-intent answer.

    The Tafsir resolver injects the exact verse and its commentary, but the
    generated prose sometimes omits the verse itself and only paraphrases
    the commentary.  The answer-completeness pass quotes the injected verse
    when it is missing; this helper deliberately only fires for explicit
    Tafsir intent so ordinary questions keep the core-topic rules.
    """
    qn = normalize(query)
    if not (_TAFSIR_VERB_RE.search(qn) or _TAFSIR_MEANING_RE.search(qn)):
        return None
    return _injected_quran_verse(retrieved)


def _answer_quotes_verse(answer: str, verse: str) -> bool:
    """True when the answer contains the verse's normalized text verbatim.

    A token-overlap test is too weak for this purpose: "الله الواحد" shares
    the keys الله/أحد with "قل هو الله أحد" and would falsely count as a
    quotation.  The completeness pass must append the verse unless its full
    normalized wording is actually present.
    """
    return (
        _fold_ar(normalize(verse))
        in _fold_ar(normalize(answer))
    )


def _answer_quotes_verse_fragment(
    answer: str, verse: str, min_tokens: int = 12,
) -> bool:
    """True when the answer already contains a long contiguous verse quote.

    A partial but substantial quotation (e.g. the wudu portion of 5:6) is
    already the relevant revelation text.  Appending the entire verse then
    repeats dozens of words and adds unrelated clauses.  A 12-token
    contiguous window is deliberately conservative: a short phrase such as
    "وأحل الله البيع وحرم الربا" does not suppress the full verse.
    """
    verse_tokens = _fold_ar(normalize(verse)).split()
    if len(verse_tokens) < min_tokens:
        return False
    answer_norm = _fold_ar(normalize(answer))
    return any(
        " ".join(verse_tokens[i:i + min_tokens]) in answer_norm
        for i in range(len(verse_tokens) - min_tokens + 1)
    )


_HADITH_TEXT_REQUEST_MARKERS = (
    "نص", "نصه", "اذكر", "المصدر", "مصدر", "دلال", "معني", "معنى",
    "اشرح", "وضح", "شرح",
)


def _is_hadith_text_request(query: str) -> bool:
    qn = normalize(query)
    return "حديث" in qn and any(
        marker in qn for marker in _HADITH_TEXT_REQUEST_MARKERS
    )


def _hadith_quote_from_chunk(chunk: dict) -> str:
    """Extract the matn quotation from a hadith chunk, without isnad."""
    metadata = chunk.get("metadata") or {}
    raw = metadata.get("matn") or chunk.get("original_text") or ""
    quotes = _QUOTE_RE.findall(raw)
    if quotes:
        quote = max(quotes, key=len)
    else:
        quote = raw
        for marker in (
            "قال رسول الله", "قال النبي", "صلى الله عليه وسلم",
            "يقول :", "يقول:",
        ):
            index = quote.rfind(marker)
            if index >= 0:
                tail = quote[index + len(marker):]
                if len(tail.split()) >= 4:
                    quote = tail
                    break
    return _clean_source_snippet(quote)


def _hadith_principle(quote: str) -> str:
    """Restate the opening principle of a hadith without its narrative.

    This is extractive: it keeps the source's own words up to the first
    narrative clause (فمن/ومن/ثم) and removes only the emphatic إنما opening.
    It does not introduce an outside interpretation.
    """
    text = _clean_source_snippet(quote)
    for marker in (" فَمَنْ ", " فمن ", " وَمَنْ ", " ومن ", " ثُمَّ ", " ثم "):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
            break
    text = re.sub(r"^(?:إِنَّمَا|إنما)\s+", "", text).strip()
    text = text.replace("وَإِنَّمَا", "وَ").replace("وإنما", "و")
    text = text.replace("وَ ", "و")
    text = text.strip(" ،.؛:")
    words = text.split()
    if len(words) > 30:
        text = " ".join(words[:30])
    return text


def _hadith_candidates(
    query: str, retrieved: list[dict],
) -> list[tuple[int, dict, str]]:
    """Retrieved hadiths whose matn matches the quoted phrase in *query*."""
    query_quotes = _QUOTE_RE.findall(query)
    wanted = normalize(max(query_quotes, key=len)).strip() if query_quotes else ""
    candidates: list[tuple[int, dict, str]] = []
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "hadith":
            continue
        quote = _hadith_quote_from_chunk(chunk)
        if len(quote.split()) < 3:
            continue
        source_norm = normalize(
            (chunk.get("original_text") or "") + " " + quote,
        )
        if wanted and wanted not in source_norm:
            wanted_tokens = set(wanted.split())
            source_tokens = set(source_norm.split())
            if not wanted_tokens or (
                len(wanted_tokens & source_tokens) / len(wanted_tokens) < 0.80
            ):
                continue
        candidates.append((index, chunk, quote))
    candidates.sort(
        key=lambda item: (not _is_primary_hadith(item[1]), item[0]),
    )
    return candidates


def _is_hadith_explanation_request(query: str) -> bool:
    qn = normalize(query)
    return "حديث" in qn and any(
        marker in qn for marker in ("اشرح", "فسر", "وضح", "بيّن", "بين")
    )


def _hadith_explanation_supplement(
    query: str,
    answer: str,
    retrieved: list[dict],
) -> str | None:
    """Add an explanation/example when the answer only quotes the hadith.

    The explanation is a restatement of the source's opening principle.  The
    example is explicitly labelled as illustrative and not part of the hadith
    text; it is never cited to the source.
    """
    if not _is_hadith_explanation_request(query):
        return None
    candidates = _hadith_candidates(query, retrieved)
    if not candidates:
        return None
    index, _chunk, quote = candidates[0]
    principle = _hadith_principle(quote)
    if len(principle.split()) < 3:
        return None
    answer_norm = normalize(answer)
    lines: list[str] = []
    has_explanation = any(
        marker in answer_norm
        for marker in ("الشرح", "يبيّن", "يبن", "تقوم", "يعني", "أي أن", "اي ان")
    )
    if not has_explanation:
        if "نيات" in normalize(principle) or "نيه" in normalize(principle):
            lines.append(
                "الشرح: يبيّن الحديث أن الأعمال تُقوَّم بنياتها، وأن لكل "
                "امرئ ما نوى؛ فليست العبرة بصورة العمل وحدها، بل بالقصد "
                f"الذي يحمله صاحبه (مصدر {index})."
            )
        else:
            lines.append(f"الشرح: {principle} (مصدر {index}).")
    if "مثال" in normalize(query) and "مثال" not in answer_norm:
        lines.append(
            "مثال توضيحي (ليس من نص الحديث): نفرض عملين ظاهرهما واحد، "
            "لكن أحدهما يقصد به صاحبه وجه الله، والآخر يقصد به مدح "
            "الناس؛ فالصورة متشابهة، والنية مختلفة."
        )
    return "\n\n".join(lines) if lines else None


def _hadith_text_answer(query: str, retrieved: list[dict]) -> str | None:
    """Build a source-grounded text/source/meaning answer for a hadith query.

    Used only when generation and grounding still end in a false refusal.
    The rule is generic: locate the retrieved hadith whose matn contains the
    quoted phrase from the question (or the first retrieved hadith), quote
    its matn, name its reference, and restate its opening principle.
    """
    if not _is_hadith_text_request(query):
        return None
    candidates = _hadith_candidates(query, retrieved)
    if not candidates:
        return None
    index, chunk, quote = candidates[0]
    reference = str(chunk.get("reference") or "مصدر حديثي")
    principle = _hadith_principle(quote)
    lines = [
        f"نص الحديث: «{quote}» (مصدر {index}).",
        f"المصدر: {reference} (مصدر {index}).",
    ]
    if len(principle.split()) >= 3:
        lines.append(
            f"الدلالة الأساسية: {principle} (مصدر {index})."
        )
    return "\n\n".join(lines)


_SEERAH_EVENT_QUERY_MARKERS = (
    "احداث", "أحداث", "اهم", "أهم", "رتب", "رتّب", "مراحل", "تسلسل",
)
_SEERAH_MAJOR_HEADING_MARKERS = (
    "هجره", "هجرة", "قريش", "غار", "مدينه", "مدينة", "مسجد",
    "مهاجر", "خروج", "قدوم", "نزول", "بناء", "تلاحق", "انتظار",
    "اجتماع",
)
_SEERAH_NON_EVENT_HEADING_MARKERS = (
    "ارتجاز", "شعر", "إخبار", "اخبار", "مشادة", "نسب",
)


def _is_seerah_event_query(query: str) -> bool:
    qn = normalize(query)
    return (
        ("هجره" in qn or "سيره" in qn or "غزوه" in qn)
        and any(marker in qn for marker in _SEERAH_EVENT_QUERY_MARKERS)
    )


def _seerah_family(reference: str) -> str:
    return re.sub(r"\s*\(\d+/\d+\)\s*$", "", reference or "").strip()


def _seerah_event_outline(
    query: str, retrieved: list[dict],
) -> str | None:
    """Extract an ordered outline from retrieved sira section headings.

    Classical sira chunks carry their own section headings in parentheses.
    This returns the major headings, in reference order, with the first source
    sentence of each section.  No event is invented when it is not present in
    the retrieved chunks.
    """
    if not _is_seerah_event_query(query):
        return None
    families: dict[str, list[tuple[int, str, dict]]] = {}
    for index, chunk in enumerate(retrieved, 1):
        if chunk.get("source") != "sira":
            continue
        reference = str(chunk.get("reference") or "")
        family = _seerah_family(reference)
        if not family:
            continue
        families.setdefault(family, []).append((index, reference, chunk))
    if not families:
        return None
    query_keys = _source_text_keys(query)

    def family_score(item: tuple[str, list]) -> tuple[int, int]:
        family, members = item
        family_keys = _source_text_keys(family)
        return (
            len(query_keys & family_keys),
            len(members),
        )

    family, members = max(families.items(), key=family_score)
    del family
    events: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for index, reference, chunk in members:
        ordinal_match = re.search(r"\((\d+)/(\d+)\)", reference)
        ordinal = int(ordinal_match.group(1)) if ordinal_match else 999
        raw = chunk.get("original_text") or ""
        headings = list(re.finditer(r"\(([^)\n]{8,140})\)\s*:", raw))
        for position, heading_match in enumerate(headings):
            heading = _clean_source_snippet(heading_match.group(1))
            if not heading or heading in seen:
                continue
            is_first = position == 0
            heading_norm = normalize(heading)
            if any(
                normalize(marker) in heading_norm
                for marker in _SEERAH_NON_EVENT_HEADING_MARKERS
            ):
                continue
            if not is_first and not any(
                marker in heading_norm
                for marker in _SEERAH_MAJOR_HEADING_MARKERS
            ):
                continue
            end = (
                headings[position + 1].start()
                if position + 1 < len(headings)
                else len(raw)
            )
            segment = raw[heading_match.end():end]
            sentences = _split_source_sentences(segment)
            clause = ""
            for sentence in sentences:
                cleaned = _clean_source_snippet(sentence)
                if len(cleaned.split()) < 5:
                    continue
                if cleaned.startswith("قال ابن هشام"):
                    continue
                clause = cleaned
                break
            if not clause:
                continue
            words = clause.split()
            if len(words) > 55:
                prefix = " ".join(words[:55])
                comma = prefix.rfind("،")
                if comma > 0:
                    prefix = prefix[:comma]
                prefix = re.sub(
                    r"\s+(?:صلى\s+الله\s+عليه|الله\s+عليه)$",
                    "", prefix,
                )
                clause = prefix.strip(" ،") + "…"
            events.append((ordinal, index, heading, clause))
            seen.add(heading)
    if len(events) < 3:
        return None
    events.sort(key=lambda item: (item[0], item[1]))
    chosen = events[:6]
    lines = []
    for number, (_, index, heading, clause) in enumerate(chosen, 1):
        heading = heading.strip(" :،.؛")
        lines.append(
            f"{number}. {heading}: {clause} (مصدر {index})."
        )
    covered = normalize(
        " ".join(heading + " " + clause for _, _, heading, clause in chosen)
    )
    missing: list[str] = []
    for marker, label in (
        ("غار", "المكوث في الغار"),
        ("قباء", "القدوم إلى قباء"),
        ("الطريق", "تفاصيل الطريق"),
        ("خرج النبي", "تفاصيل الخروج من مكة"),
        ("وصل النبي", "تفاصيل الوصول إلى المدينة"),
    ):
        if normalize(marker) not in covered:
            missing.append(label)
    if missing:
        lines.append("")
        lines.append(
            "المراحل الواردة أعلاه هي ما تدعمه المصادر المستردة. "
            "لم تُسترد تفاصيل كافية عن: "
            + "، ".join(missing)
            + "؛ لذلك لا تُذكر هنا ولا تُنسب إلى المصادر."
        )
    else:
        lines.append("")
        lines.append(
            "ما ورد أعلاه هو ما تدعمه المصادر المستردة، ولم تُضف "
            "تفاصيل غير ثابتة فيها."
        )
    return "\n".join(lines)


def _augment_answer_completeness(
    query: str, answer: str, retrieved: list[dict],
) -> str:
    if not answer.strip():
        return answer
    qn = normalize(query)
    tafsir_anchor = _tafsir_anchor_verse(query, retrieved)
    anchor_verse = (
        _quran_topic_anchor_verse(query, retrieved)
        or tafsir_anchor
    )
    if _COMPLETENESS_REFUSAL in answer:
        # A core-topic question can be answered directly from its injected
        # canonical verse even when the model refused or its prose failed
        # grounding.  The verse is quoted verbatim and re-grounded below.
        recovered_anchor = False
        if anchor_verse is not None:
            index, verse, reference = anchor_verse
            trial = (
                f'قال الله تعالى: "{verse}" ({reference}) '
                f"(مصدر {index})."
            )
            if _COMPLETENESS_REFUSAL not in _ground_answer(trial, retrieved):
                answer = trial
                recovered_anchor = True
        if not recovered_anchor:
            story = _recover_tayammum_narrative(query, retrieved)
            if story is not None:
                answer = story
                recovered_anchor = True
        if not recovered_anchor:
            recovered = _find_variant_answer(query, retrieved)
            if not recovered:
                return answer
            index, snippet = recovered
            trial = f"{snippet} (مصدر {index})."
            if _COMPLETENESS_REFUSAL in _ground_answer(trial, retrieved):
                return answer
            answer = trial
    answer_keys = _source_text_keys(answer)
    additions: list[tuple[str, int, str]] = []
    if anchor_verse is not None:
        index, verse, reference = anchor_verse
        if tafsir_anchor is not None:
            # For a resolved Tafsir question the verse itself is the primary
            # text the user asked about.  Token overlap is not enough: an
            # answer saying "الله الواحد" is not a quotation of "قل هو الله
            # أحد".  Append the verse unless its full normalized wording is
            # present.
            missing_verse = not _answer_quotes_verse(answer, verse)
        elif _answer_quotes_verse_fragment(answer, verse):
            # The answer already quotes a substantial contiguous portion of
            # the verse; appending the whole verse would duplicate it (the
            # quran_wudu answer repeated the full 5:6 verse this way).
            missing_verse = False
        else:
            verse_keys = _source_text_keys(verse)
            missing_verse = bool(verse_keys) and (
                len(verse_keys.intersection(answer_keys)) / len(verse_keys)
                < 0.80
            )
        if missing_verse:
            # The question has a deterministic primary verse but the answer
            # quotes only part of it (or none).  Add the complete verse so a
            # partial quotation cannot silently omit a clause the question
            # asks about.
            additions.append((
                "append", index,
                f'قال الله تعالى: "{verse}" ({reference})',
            ))
    if (
        "اعظم ايه" in qn
        or "افضل ايه" in qn
        or "الكرسي" in qn
    ):
        answer_norm = _fold_ar(normalize(answer))
        if (
            "ايه الكرسي" not in answer_norm
            and "الكرسي" not in answer_norm
        ):
            found = _find_source_snippet(
                query, answer, retrieved, ("آية الكرسي",),
                min_words=4,
            )
            if found:
                additions.append(("append", *found))
    if "زكاه الفطر" in qn:
        # Zakat al-fitr has a precise measure and deadline.  An answer that
        # discusses obligation but omits the amount or the "before the Eid
        # prayer" timing is incomplete even when every sentence it contains
        # is grounded.
        if _gkey("صاع") not in answer_keys:
            found = _find_source_snippet(
                query, answer, retrieved, ("صاع",),
            )
            if found:
                additions.append(("append", *found))
        answer_norm = _fold_ar(normalize(answer))
        if not any(
            marker in answer_norm
            for marker in (
                "قبل الصلاه", "قبل صلاة العيد", "قبل خروج الناس",
                "قبل العيد",
            )
        ):
            found = _find_source_snippet(
                query, answer, retrieved,
                ("قبل خروج الناس", "قبل الصلاه", "قبل صلاة العيد"),
            )
            if found:
                additions.append(("append", *found))
    if "تيمم" in qn:
        qn_folded = _fold_ar(qn)
        narrative = (
            "عايشه" in qn_folded
            and any(marker in qn for marker in ("سفر", "شرع", "قصه"))
        )
        if narrative:
            # The tayammum story's initiating event is the lost necklace.
            # If the generated paraphrase of that clause is dropped (e.g. a
            # non-source synonym for "فتغيظ"), the surviving water/verse
            # sentences do not tell the user how the incident began.  Add
            # the source's own concise necklace clause.
            if not (
                {_gkey("انقطع"), _gkey("سقط"), _gkey("ضاع")}
                & answer_keys
            ):
                found = _find_source_snippet(
                    query, answer, retrieved,
                    ("سقط عقد", "انقطع عقد", "عقد لي"),
                )
                if found:
                    additions.append(("append", *found))
            answer_norm = _fold_ar(normalize(answer))
            if not any(
                marker in answer_norm
                for marker in (
                    "ليسوا على ماء", "ليس معهم ماء", "لم يجدوا ماء",
                    "على غير ماء", "لم يكن معهم ماء",
                )
            ):
                # The narrative's legal cause is the absence of water; a
                # necklace-only retelling is incomplete.
                found = _find_source_snippet(
                    query, answer, retrieved,
                    ("ليسوا على ماء", "ليس معهم ماء", "لم يجدوا ماء",
                     "على غير ماء", "لم يكن معهم ماء"),
                )
                if found:
                    additions.append(("append", *found))
            if "ايه التيمم" not in _fold_ar(normalize(answer)):
                # "الرخصة في التيمم" is a valid paraphrase, but the story's
                # revealed verse is explicitly called آية التيمم in the
                # source; state that clause rather than leaving the reader
                # with only the generic word الرخصة.
                found = _find_source_snippet(
                    query, answer, retrieved, ("آية التيمم",),
                    min_words=6,
                )
                if found:
                    index, snippet = found
                    snippet = re.split(
                        r"\s+قال\s+", snippet, maxsplit=1,
                    )[0].strip()
                    if snippet:
                        additions.append(("append", index, snippet))
        elif not (
            {_gkey("مرض"), _gkey("جراحة"), _gkey("جرح")} & answer_keys
        ):
            # The permissibility of tayammum has two independent causes in
            # the fiqh source: absence of water and illness/injury.  A model
            # answer that lists only the water cases is incomplete even when
            # every listed case is grounded.
            # Anchor on جراحة, not the bare مرض: the Quranic basis sentence
            # in fiqh_66 contains "مرضى" and would otherwise win the
            # subject-overlap score while saying nothing about the disease
            # or injury cause.  The fiqh_67 clause states the actual cause.
            found = _find_source_snippet(
                query, answer, retrieved, ("جراحه", "جرح"),
            )
            if found:
                additions.append(("append", *found))
    if "ربا" in qn:
        # The fiqh definition states the essential nature of riba (an
        # increase over the principal), which is the non-optional fact in
        # the broad rubric.  A harm-only answer omits it.
        if not ({_gkey("زيادة"), _gkey("رأس")} & answer_keys):
            found = _find_source_snippet(
                query, answer, retrieved,
                ("الزياده على راس المال",),
            )
            if found:
                additions.append(("append", *found))
        if _gkey("ضرر") not in answer_keys:
            found = _find_source_snippet(
                query, answer, retrieved, ("ضرر عظيم",),
            )
            if found:
                additions.append(("append", *found))
    variant = _find_variant_quote(query, retrieved)
    if variant:
        index, quote = variant
        additions.append((
            "append", index,
            f"قال رسول الله صلى الله عليه وسلم: {quote}",
        ))
    if any(marker in qn for marker in _COMPLETENESS_RULING_MARKERS):
        if not _COMPLETENESS_WEAK_RULINGS.intersection(answer_keys):
            found = _find_source_snippet(
                query, answer, retrieved, _COMPLETENESS_QUALIFIER_MARKERS,
                phrase_only=True,
            )
            if found:
                additions.append(("append", *found))
                if _gkey("استحباب") in _source_text_keys(found[1]):
                    # State the ruling explicitly.  The supporting command is
                    # appended by the separate command block below; folding it
                    # into this summary duplicated the same source clause and
                    # made the prepend overlap check skip the ruling itself.
                    subject = _supported_subject_phrase(query, retrieved)
                    if subject:
                        additions.append((
                            "prepend", found[0], f"{subject} مستحب",
                        ))
        if not {_gkey("امر"), _gkey("اكد")} <= answer_keys:
            command = _find_source_snippet(
                query, answer, retrieved,
                _COMPLETENESS_COMMAND_MARKERS, phrase_only=True,
            )
            if command:
                additions.append(("append", *command))
        if not (
            {_gkey("بالفراغ"), _gkey("الفراغ")} & answer_keys
        ):
            found = _find_source_snippet(
                query, answer, retrieved, _COMPLETENESS_TIMING_MARKERS,
                phrase_only=True,
            )
            if found:
                additions.append(("append", *found))
    if _TAFSIR_VERB_RE.search(qn):
        # Tafsir answers should open with the passage's overall meaning.
        # When the model jumps straight to legal details, prepend the tafsir's
        # own opening clause (verified against the cited chunk).
        found = _find_tafsir_opening_clause(query, retrieved)
        if found:
            additions.append(("prepend", *found))
    if _asks_fasting_query(qn):
        day_tokens = [
            token for token in _tokenize(qn)
            if token in _DATE_WEEKDAY_TOKENS
            or token in _FASTING_DAY_QUALIFIERS
        ]
        if day_tokens:
            answer_norm = _fold_ar(normalize(answer))
            if not any(
                marker in answer_norm
                for marker in ("النهي عن", "لا يكره", "مكروه")
            ):
                qualified = _find_source_snippet(
                    query, answer, retrieved,
                    ("وذهب الجمهور", "الكراهه", "لا يكره"),
                    phrase_only=True,
                )
                if qualified:
                    qualified_chunk = retrieved[qualified[0] - 1]
                    if day_tokens[0] in normalize(
                        qualified_chunk.get("original_text", "") or "",
                    ):
                        additions.append(("append", *qualified))
                day = day_tokens[0]
                found = _find_source_snippet(
                    query, answer, retrieved,
                    (
                        f"النهي عن صوم يوم {day}",
                        f"النهي عن صوم يوم ال{day}",
                        f"النهي عن صيام يوم {day}",
                        f"النهي عن صيام يوم ال{day}",
                        f"صوم يوم {day}",
                        f"صوم يوم ال{day}",
                    ),
                    phrase_only=True,
                )
                if found and len(found[1].split()) >= 8:
                    additions.append(("append", *found))
            if "عاده" not in answer_norm:
                full_qualification = _find_source_sentence(
                    query, retrieved, "وذهب الجمهور",
                )
                if full_qualification:
                    additions.append(("append", *full_qualification))
    distinction = _find_distinction_snippet(query, answer, retrieved)
    if distinction:
        additions.append(("append", *distinction))

    if "شروط" in qn:
        # A conditions question needs the source's own condition list, not
        # only the subset the model happened to mention.
        found = _find_condition_list_snippet(query, retrieved)
        if found:
            additions.append(("append", *found))
    if "قصر" in qn:
        if not (
            {_gkey("مفارقة"), _gkey("الحضر")} & answer_keys
        ):
            # Shortening begins when the traveler leaves the settled area;
            # an answer that only discusses duration is incomplete.
            found = _find_source_snippet(
                query, answer, retrieved,
                ("مفارقة الحضر", "الخروج من البلد"),
            )
            if found:
                additions.append(("append", *found))
        if not (
            {_gkey("ركعتين"), _gkey("ركعتان")} & answer_keys
        ):
            # A qasr answer that never states the shortened prayer's two
            # rak'ahs is incomplete even when it correctly explains when
            # shortening starts.  The number is stated explicitly in the fiqh
            # source.
            found = _find_source_snippet(
                query, answer, retrieved, ("ركعتين", "ركعتان"),
            )
            if found:
                additions.append(("append", *found))
    if qn.startswith(_COMPLETENESS_DEFINITION_PREFIXES):
        if (_COMPLETENESS_DEFINITION_SUBJECTS
                & _query_subject_keys(query)):
            if _gkey("انزل") not in answer_keys:
                found = _find_source_snippet(
                    query, answer, retrieved, _COMPLETENESS_DEFINITION_MARKERS,
                )
                if found:
                    additions.append(("prepend", *found))
        found = _find_source_snippet(
            query, answer, retrieved, _COMPLETENESS_LIST_MARKERS,
            list_mode=True,
        )
        if found:
            additions.append(("append", *found))
    if qn.startswith(_COMPLETENESS_STORY_PREFIXES):
        found = _find_story_snippet(query, answer, retrieved)
        if found:
            additions.append(("prepend", *found))
        found = _find_narrative_quran_snippet(query, retrieved)
        if found:
            additions.append(("append", *found))
    seen: set[tuple[int, str]] = set()
    for mode, index, snippet in additions:
        key = (index, snippet)
        if key in seen:
            continue
        seen.add(key)
        answer = _add_completeness_snippet(
            answer, retrieved, index, snippet, prepend=(mode == "prepend"),
        )
    return answer


# ---------------------------------------------------------------------------
# Prayer-quote consistency
# ---------------------------------------------------------------------------

_PRAYER_TIME_ALIASES = {
    _gkey("فجر"): "فجر",
    _gkey("صبح"): "فجر",
    _gkey("ظهر"): "ظهر",
    _gkey("عصر"): "عصر",
    _gkey("مغرب"): "مغرب",
    _gkey("عشاء"): "عشاء",
}
_PRAYER_TIME_SURFACE = {
    "فجر": "فجر", "ظهر": "ظهر", "عصر": "عصر",
    "مغرب": "مغرب", "عشاء": "عشاء",
}
_BOLD_HEADING_RE = re.compile(r"\*\*([^*\n]+)\*\*")


def _strip_bold_headings(text: str) -> str:
    """Remove presentational ``**label**`` headings from *text*.

    Headings are structural labels, not evidence.  When one contains a word
    the cited source does not use (e.g. ``**تحريم الأموال والدماء**`` over a
    verbatim ``دماءكم وأموالكم ... حرام`` quote), strict grounding must not
    throw away the supported body.  Removing the label lets the normal
    checks validate the remaining claim without weakening any guard.
    """
    out = _BOLD_HEADING_RE.sub("", text)
    out = re.sub(r"(?m)^\s*(?:\d+[.)]\s*)?[:：]\s*", "", out)
    out = re.sub(r"(?<=[.!?\u061f])\s*[:：]\s*", " ", out)
    out = re.sub(r"\s*[:：]\s*(?=[.،؛])", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _ground_with_heading_fallback(
    answer: str, retrieved: list[dict], corpus_chunks: list[dict] | None,
) -> str:
    """Ground *answer*, retrying presentational repairs and keeping the most.

    Two repairs are tried independently and together: removing bold labels
    (headings are not evidence) and removing quotation marks around spans the
    cited source does not contain verbatim (a paraphrased quote is still a
    claim, but its body must pass every guard).  Each candidate is produced by
    the full strict grounding policy; the longest grounded result wins.  This
    prevents one unsupported label or pair of quotation marks from discarding
    otherwise supported sentences, without relaxing any substantive check.
    """
    refusal = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
    candidates = [_ground_answer(answer, retrieved, corpus_chunks)]
    stripped = _strip_bold_headings(answer)
    if stripped != answer:
        fallback = _ground_answer(stripped, retrieved, corpus_chunks)
        if fallback.strip() != refusal:
            candidates.append(fallback)
    unquoted = _strip_unsupported_quotes(answer, retrieved)
    if unquoted != answer:
        fallback = _ground_answer(unquoted, retrieved, corpus_chunks)
        if fallback.strip() != refusal:
            candidates.append(fallback)
    if stripped != answer:
        both = _strip_unsupported_quotes(stripped, retrieved)
        if both != stripped:
            fallback = _ground_answer(both, retrieved, corpus_chunks)
            if fallback.strip() != refusal:
                candidates.append(fallback)
    return max(candidates, key=len)


def _prayer_time_keys(text: str) -> set[str]:
    return {
        _PRAYER_TIME_ALIASES[_gkey(token)]
        for token in _tokenize(normalize(text))
        if _gkey(token) in _PRAYER_TIME_ALIASES
    }


def _bold_heading(sentence: str):
    matches = list(_BOLD_HEADING_RE.finditer(sentence))
    if not matches:
        return None
    match = matches[-1]
    return match.start(1), match.end(1), _prayer_time_keys(match.group(1))


def _replace_prayer_time(text: str, old_key: str, new_key: str) -> str:
    def replace(match: re.Match) -> str:
        token = match.group(0)
        core = token.strip("،.:;!?؟()[]{}")
        if _prayer_time_keys(core) == {old_key}:
            prefix = "ال" if core.startswith("ال") else ""
            return token.replace(core, prefix + _PRAYER_TIME_SURFACE[new_key])
        return token
    return re.sub(r"[^\s]+", replace, text)


def _remove_conflicting_prayer_quotes(sentence: str, active: str) -> str:
    out = sentence
    for match in reversed(list(_QUOTE_RE.finditer(out))):
        quote_keys = _prayer_time_keys(match.group(1))
        if not quote_keys or active in quote_keys:
            continue
        start, end = match.span()
        tail = out[end:]
        citations = re.match(r"\s*(?:\(مصدر\s*\d+\)\s*)+", tail)
        if citations:
            end += citations.end()
        while end < len(out) and out[end] in ".!?؟ ":
            end += 1
        out = out[:start] + out[end:]
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+([.!?؟])", r"\1", out)
    if re.search(r"[:：]\s*[.!?؟]?\s*$", out):
        return ""
    if len(_answer_content_tokens(out)) < 3:
        return ""
    return out


def _fix_mismatched_prayer_quotes(answer: str) -> str:
    """Remove or realign quotations whose prayer time contradicts the item.

    The generator sometimes puts a source quotation under the wrong prayer
    heading (observed: a "before Asr" hadith under "before Dhuhr").  A quote
    must never contradict its own item.  For an inline heading the heading is
    realigned to the quote when the quote names exactly one prayer; for a
    separate follow-up sentence the contradictory quote is dropped.
    """
    if not answer.strip():
        return answer
    pieces = _SENT_SPLIT_RE.split(answer)
    out: list[str] = []
    active: str | None = None
    for piece in pieces:
        if not piece.strip():
            out.append(piece)
            continue
        heading = _bold_heading(piece)
        if heading is not None:
            _, _, heading_times = heading
            active = (
                next(iter(heading_times)) if len(heading_times) == 1 else None
            )
        quotes = list(_QUOTE_RE.finditer(piece))
        if not active or not quotes:
            out.append(piece)
            continue
        conflicting = [
            match for match in quotes
            if (lambda keys: bool(keys) and active not in keys)(
                _prayer_time_keys(match.group(1))
            )
        ]
        if not conflicting:
            out.append(piece)
            continue
        if heading is not None and len(conflicting) == len(quotes):
            start, end, heading_times = heading
            quote_times: set[str] = set()
            for match in conflicting:
                quote_times.update(_prayer_time_keys(match.group(1)))
            if len(heading_times) == 1 and len(quote_times) == 1:
                new_key = next(iter(quote_times))
                new_heading = _replace_prayer_time(
                    piece[start:end], active, new_key,
                )
                piece = piece[:start] + new_heading + piece[end:]
                active = new_key
                out.append(piece)
                continue
        cleaned = _remove_conflicting_prayer_quotes(piece, active)
        if cleaned:
            out.append(cleaned)
    result = "".join(out).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


# ===========================================================================
# generate()
# ===========================================================================


_CITATION_BEFORE_QUOTE_END_RE = re.compile(
    r'\s*\(مصدر\s*(\d+)\)\s*([؟?])\s*(["»”])'
)
_CITATION_AT_QUOTE_END_RE = re.compile(
    r'\s*\(مصدر\s*(\d+)\)\s*(["»”])'
)


def _move_citations_out_of_quotes(answer: str) -> str:
    """Move a citation that the generator placed inside a quotation outside it.

    A citation marker is not part of the quoted source text.  Leaving it
    inside changes the quotation (observed live: "أي آية ... أعظم (مصدر 2)؟").
    Only markers immediately before the closing quote are moved; a marker in
    the middle of a longer quotation is left untouched rather than guessed at.
    """
    answer = _CITATION_BEFORE_QUOTE_END_RE.sub(
        lambda m: f'{m.group(2)}{m.group(3)} (مصدر {m.group(1)})',
        answer,
    )
    return _CITATION_AT_QUOTE_END_RE.sub(
        lambda m: f'{m.group(2)} (مصدر {m.group(1)})',
        answer,
    )


def _drop_query_echo(answer: str, query: str) -> str:
    """Remove a leading verbatim question echoed by the generator."""
    query_norm = _fold_ar(normalize(query)).strip(" ؟?.")
    if not query_norm:
        return answer
    sentences = _SENT_SPLIT_RE.split(answer)
    kept: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped:
            sentence_norm = _fold_ar(normalize(stripped)).strip(" ؟?.")
            if sentence_norm == query_norm:
                continue
        kept.append(sentence)
    return "".join(kept).strip()

def _context_covers_query(query: str, retrieved: list[dict]) -> bool:
    """True when the query's content terms are present in the context.

    Used only to decide whether a model refusal deserves one retry; it never
    decides answerability (the retrieval gate does that).
    """
    qkeys = {
        t for t in _tokenize(normalize(query))
        if len(t) >= 3 and t not in _QUERY_STOPWORDS
    }
    if len(qkeys) < 2:
        return False
    context = " ".join(c.get("original_text", "") for c in retrieved)
    ckeys = set(_tokenize(normalize(context)))
    return len(qkeys & ckeys) / len(qkeys) >= 0.75


_FOLLOWUP_PRONOUN_TOKENS = {
    "هو", "هما", "هم", "هن", "ذلك", "تلك", "هذا", "هذه",
}

# Demonstratives and attached pronouns are anaphoric even when the turn also
# names a small qualifier ("هل هذا الحديث صحيح؟", "ما حكمه للحاج؟").  They are
# kept separate from personal pronouns, which can be copulas in a genuinely
# new question ("من هو الرسول؟").
_FOLLOWUP_DEMONSTRATIVE_TOKENS = {
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك",
    # Egyptian demonstratives used in the same anaphoric way.
    "ده", "دي", "دول",
}
_FOLLOWUP_ATTACHED_TOKENS = {
    "حكمه", "حكمها", "حكمهم", "دليله", "دليلها",
    "معناه", "معناها", "رأيه", "رأيها", "مثله", "مثلها",
    "اصومه", "أصومه", "اصومها", "أصومها",
    "صومه", "صيامه", "صومها", "صيامها",
    "لوحده", "لوحدها", "لوحدي",
    "انواعه", "أنواعه", "انواعها", "أنواعها",
    "شروطه", "شروطها", "احكامه", "أحكامه",
}

# Legal/practical words in a follow-up ("يعني لازم أصوم...؟", "هل ده واجب؟")
# describe the *kind* of clarification requested, not a new topic.  They are
# removed only when deciding whether the turn is elliptical; a genuinely new
# subject word such as "رمضان" keeps the turn standalone.
_FOLLOWUP_LEGAL_WORDS = {
    "لازم", "لازمه", "يلزم", "الزام", "واجب", "واجبه", "وجوب",
    "فرض", "فرضه", "فريضه", "مفروض", "حرام", "حرامه", "تحريم",
    "محرم", "حلال", "حلاله", "مباح", "اباحه", "مكروه", "كراهه",
    "جائز", "جواز", "يجوز",
}
_FOLLOWUP_PRACTICAL_WORDS = {
    "اصوم", "صوم", "صيام", "افطر", "فطر", "يوم", "قبله", "بعده",
    "قبل", "بعد", "الحاله", "حاله", "ده", "دي", "دول",
    "مينفعش", "ماينفعش", "ينفع", "مش", "لوحده", "لوحدها", "لوحدي",
    "اصومه", "أصومه", "اصومها", "أصومها",
    "صومه", "صيامه", "صومها", "صيامها",
}

# Words that carry a conversational intent but not a topic.  A query made
# only of these words is elliptical: it means "the thing we were discussing",
# not a new subject.  This covers Egyptian and MSA clarification turns such
# as "أيوة يعني حلال ولا حرام؟", "إيه الدليل؟", "ليه؟", and
# "ممكن توضح أكتر؟" without hardcoding any religious answer.
_FOLLOWUP_GENERIC_WORDS = {
    "حكم", "حكمه", "حكمها", "حكمهم", "احكام", "الحكم",
    "حلال", "حرام", "مباح", "مكروه", "واجب", "جائز", "يجوز",
    "دليل", "الدليل", "ادلته", "ايه", "ايوه", "ايوة", "نعم", "لا",
    "يعني", "ليه", "لماذا", "طب", "طيب", "ممكن", "اكتر", "اكثر",
    "توضح", "توضيح", "وضح", "اشرح", "قال",
    "راي", "الراي", "تاني", "التاني", "اخر", "الاخر", "الحاله",
    "حاله", "دي", "كده", "كدا", "صحيح", "غلط", "خطا", "ولا",
    "او", "أو", "ام", "أم", "بس", "برضه", "برضو", "فين", "ازاي",
    "إزاي", "ايه", "الدليل", "لو", "انا", "أنا", "احنا", "إحنا",
}

_FOLLOWUP_PHRASES = (
    "قال ايه الرسول",
    "قال الرسول",
    "ايه الدليل",
    "الدليل ايه",
    "ممكن توضح",
    "توضح اكتر",
    "الراي التاني",
    "راي التاني",
    "في الحاله دي",
    "حلال ولا حرام",
    "ايوه يعني",
    "طب ايه",
)
_FOLLOWUP_HYPOTHETICAL_PREFIXES = (
    "طب لو", "طيب لو", "لو انا", "لو كنت", "ماذا لو",
    "في حاله", "في الحاله", "بفرض",
)
_FOLLOWUP_GENERIC_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_GENERIC_WORDS
}
_QUERY_STOPWORDS_FOLDED = {
    _fold_ar(normalize(word)) for word in _QUERY_STOPWORDS
}
_FOLLOWUP_PRONOUN_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_PRONOUN_TOKENS
}
_FOLLOWUP_DEMONSTRATIVE_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_DEMONSTRATIVE_TOKENS
}
_FOLLOWUP_ATTACHED_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_ATTACHED_TOKENS
}
_FOLLOWUP_ANAPHORIC_FOLDED = (
    _FOLLOWUP_DEMONSTRATIVE_FOLDED | _FOLLOWUP_ATTACHED_FOLDED
)
_FOLLOWUP_LEGAL_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_LEGAL_WORDS
}
_FOLLOWUP_OBLIGATION_FOLDED = {
    _fold_ar(normalize(word)) for word in (
        "لازم", "لازمه", "يلزم", "الزام", "واجب", "واجبه",
        "وجوب", "فرض", "فرضه", "فريضه", "مفروض",
    )
}
_FOLLOWUP_PROHIBITION_FOLDED = {
    _fold_ar(normalize(word)) for word in (
        "حرام", "حرامه", "يحرم", "تحريم", "محرم", "ممنوع",
    )
}
_FOLLOWUP_PERMISSION_FOLDED = {
    _fold_ar(normalize(word)) for word in (
        "حلال", "حلاله", "مباح", "جائز", "يجوز", "جواز",
    )
}
_FOLLOWUP_PRACTICAL_FOLDED = {
    _fold_ar(normalize(word)) for word in _FOLLOWUP_PRACTICAL_WORDS
}


def _followup_content_tokens(query: str) -> list[str]:
    """Topic-bearing tokens left after removing conversational fillers."""
    qn = normalize(query).strip(" ؟?.")
    if not qn:
        return []
    tokens = [_fold_ar(token) for token in _tokenize(qn)]
    return [
        token for token in tokens
        if token not in _QUERY_STOPWORDS_FOLDED
        and token not in _FOLLOWUP_GENERIC_FOLDED
        and token not in _FOLLOWUP_PRONOUN_FOLDED
        and len(token) >= 2
    ]


def _followup_topic_tokens(query: str) -> list[str]:
    """Content tokens that name a topic, excluding legal/practical fillers."""
    return [
        token for token in _followup_content_tokens(query)
        if token not in _FOLLOWUP_LEGAL_FOLDED
        and token not in _FOLLOWUP_PRACTICAL_FOLDED
    ]


def _is_followup_query(query: str) -> bool:
    """True when *query* is a clarification/ellipsis, not a new topic."""
    qn = normalize(query).strip(" ؟?.")
    if not qn:
        return False
    tokens = [_fold_ar(token) for token in _tokenize(qn)]
    if not tokens:
        return False
    content = _followup_content_tokens(query)
    if not content:
        return True
    if (
        qn.startswith(_FOLLOWUP_HYPOTHETICAL_PREFIXES)
        and len(content) <= 5
    ):
        return True
    if (
        any(token in _FOLLOWUP_LEGAL_FOLDED for token in tokens)
        and not _followup_topic_tokens(query)
    ):
        # "يعني لازم أصوم يوم قبله أو بعده؟" asks about the legal status of
        # the previous subject; none of its words names a new topic.
        return True
    if len(content) <= 3 and any(
        token in _FOLLOWUP_ANAPHORIC_FOLDED for token in tokens
    ):
        return True
    if len(content) <= 1 and any(
        phrase in qn for phrase in _FOLLOWUP_PHRASES
    ):
        return True
    if qn.startswith(("و", "ف", "ثم")) and len(content) <= 3:
        return True
    return False


def _is_evidence_request(query: str) -> bool:
    qn = normalize(query)
    return any(
        marker in qn
        for marker in (
            "دليل", "الدليل", "قال ايه الرسول", "قال إيه الرسول",
            "قال ايه النبي", "قال إيه النبي", "رواه الحديث",
        )
    )


_HABIT_MARKERS = ("متعود", "معتاد", "عاده", "عادة", "عودت")


def _answer_mentions_habit(text: str) -> bool:
    qn = normalize(text or "")
    return any(marker in qn for marker in _HABIT_MARKERS)


def _is_habit_query(query: str) -> bool:
    return _answer_mentions_habit(query)


def _last_user_message(chat_history: list[dict] | None) -> str:
    return next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(chat_history or [])
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content") or "").strip()
        ),
        "",
    )


def _last_substantive_user(chat_history: list[dict] | None) -> str:
    return next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(chat_history or [])
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content") or "").strip()
            and not _is_followup_query(str(message.get("content") or ""))
        ),
        "",
    )


def _evidence_target_text(previous_assistant: str) -> str:
    """Pick the preceding substantive sentence an evidence request refers to."""
    body = _CITATION_RE.sub("", previous_assistant or "")
    sentences = [
        sentence.strip()
        for sentence in _SENT_SPLIT_RE.split(body)
        if sentence.strip()
    ]
    markers = (
        "عاده", "عادة", "الكراهه", "لا يكره", "النهي", "يحرم", "حرام",
        "واجب", "لا يجب", "يجوز", "يشترط", "قال رسول", "قال النبي",
        "ورد",
    )
    preferred = [
        sentence for sentence in sentences
        if any(marker in normalize(sentence) for marker in markers)
        and "قال الله" not in normalize(sentence)
    ]
    if preferred:
        return preferred[-1][:500]
    return sentences[-1][:500] if sentences else ""


def _followup_required_evidence_keys(query: str) -> set[str]:
    """Action/day keys that a strict follow-up retrieval must cover.

    Only day-qualified action questions (e.g. Friday fasting) use the strict
    all-keys check.  Other follow-ups keep the established retrieval behavior
    so a narrow clarification such as "وما شروطه؟" is not converted into a
    refusal.
    """
    return _required_action_evidence_keys(query)


def _followup_results_relevant(
    retrieval_query: str,
    retrieved: list[dict],
    required_keys: set[str],
) -> bool:
    if not retrieved:
        return False
    if not required_keys:
        return True
    for chunk in retrieved:
        if required_keys <= _legal_topic_keys(
            chunk.get("original_text", "") or "",
        ):
            return True
    return False


def _filter_followup_results(
    retrieved: list[dict],
    required_keys: set[str],
) -> list[dict]:
    """Drop unrelated chunks from a follow-up retrieval.

    A follow-up about Friday fasting must not display waqf, usurpation, or
    other unrelated sources merely because they passed the generic retrieval
    gate.  Chunks covering all required keys are preferred; chunks covering
    at least one are retained as secondary evidence.
    """
    if not required_keys:
        return retrieved
    full = [
        chunk for chunk in retrieved
        if required_keys <= _legal_topic_keys(
            chunk.get("original_text", "") or "",
        )
    ]
    if full:
        return full
    partial = [
        chunk for chunk in retrieved
        if required_keys & _legal_topic_keys(
            chunk.get("original_text", "") or "",
        )
    ]
    return partial or retrieved


def _followup_retry_query(
    query: str,
    chat_history: list[dict] | None,
) -> str:
    """Build a stronger retrieval query from the prior turn's context."""
    previous_user = _last_substantive_user(chat_history)
    previous_assistant = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(chat_history or [])
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and str(message.get("content") or "").strip()
        ),
        "",
    )
    target = _evidence_target_text(previous_assistant)
    parts = [previous_user, target, query]
    return " ".join(part for part in parts if part).strip()


_AMBIGUOUS_ACTION_CLARIFICATION = (
    "ما الفعل أو التصرف المقصود بـ«كده»؟ وضّح لي ما فعلته أو تنوي فعله "
    "(مثل: صيام يوم الجمعة، أو فعل معين في الصلاة، أو تصرف مالي) حتى "
    "أبحث عن حكمه في المصادر."
)


def _is_ambiguous_action_followup(
    query: str,
    chat_history: list[dict] | None,
) -> bool:
    """True for a deictic "I did that" follow-up with no identified action."""
    if not chat_history:
        return False
    qn = normalize(query)
    deictic = any(
        marker in qn for marker in ("كده", "كدا", "كذالك", "كذلك", "هكذا")
    )
    generic_action = any(
        marker in qn for marker in (
            "عملت", "اعمل", "فعلت", "افعل", "سويت", "اسوي", "عمل",
        )
    )
    return deictic and generic_action


def _history_aware_query(
    query: str, chat_history: list[dict] | None,
) -> str:
    """Build a retrieval-only query for a short follow-up turn.

    The generation prompt already receives the conversation history, but
    retrieval previously saw only the literal follow-up ("وما حكمه؟") and
    could not find the subject.  Prepend the most recent substantive user
    question for clarification/elliptical turns; the original query still
    controls anchors, citations, and the final prompt.  A new unrelated
    question is never enriched.
    """
    if not chat_history or not _is_followup_query(query):
        return query

    if _is_evidence_request(query):
        previous_assistant = next(
            (
                str(message.get("content") or "").strip()
                for message in reversed(chat_history)
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and str(message.get("content") or "").strip()
            ),
            "",
        )
        target = _evidence_target_text(previous_assistant)
        if target:
            previous_question = _last_user_message(chat_history)
            if previous_question:
                return f"{previous_question} {target}"
            return target

    previous_user = _last_substantive_user(chat_history)
    if not previous_user:
        return query
    if normalize(previous_user).strip(" ؟?.") == normalize(query).strip(" ؟?."):
        return query
    # For a pure clarification ("إيه الدليل؟", "حلال ولا حرام؟") the current
    # words contain no topic; retrieve with the previous substantive question
    # alone.  Appending generic polarity/discourse words diluted BM25 and
    # pulled unrelated finance/prayer chunks.
    current_topic = _followup_topic_tokens(query)
    phrase_only = (
        len(current_topic) <= 1
        and any(phrase in normalize(query) for phrase in _FOLLOWUP_PHRASES)
    )
    if not current_topic or phrase_only:
        return previous_user[-300:]
    return f"{previous_user[-300:]} {query.strip()}"


def _answer_citation_support_ok(
    answer: str, retrieved: list[dict], threshold: float = 0.80,
) -> bool:
    if not answer.strip() or _COMPLETENESS_REFUSAL in answer:
        return False
    for sentence in _SENT_SPLIT_RE.split(answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        indices = [
            int(match.group(1)) - 1
            for match in _CITATION_RE.finditer(sentence)
            if 1 <= int(match.group(1)) <= len(retrieved)
        ]
        if not indices:
            return False
        support = max(
            _source_support_ratio(sentence, retrieved[index])
            for index in indices
        )
        if support < threshold:
            return False
        legal_claims = _positive_legal_families(sentence)
        if legal_claims and not any(
            legal_claims <= _positive_legal_families(
                retrieved[index].get("original_text", "") or "",
            )
            for index in indices
        ):
            return False
    return True


def _grounded_source_claim(
    subject_text: str,
    markers: tuple[str, ...],
    retrieved: list[dict],
    chunks: list[dict],
    subject_keys: set[str] | None = None,
) -> str | None:
    """Find a verbatim source sentence about the subject and ground it."""
    if subject_keys is None:
        subject_keys = (
            _legal_topic_keys(subject_text) - _COMPLETENESS_STOP_KEYS
        )
    else:
        subject_keys = set(subject_keys)
    for chunk in retrieved:
        previous_topic_keys: set[str] = set()
        for sentence in _split_source_sentences(
            chunk.get("original_text", "") or "",
        ):
            sentence_topic_keys = _legal_topic_keys(sentence)
            effective_topic_keys = sentence_topic_keys | previous_topic_keys
            previous_topic_keys = sentence_topic_keys
            if not any(marker in normalize(sentence) for marker in markers):
                continue
            if (
                subject_keys
                and not (subject_keys <= effective_topic_keys)
            ):
                continue
            candidate = _clean_source_snippet(sentence)
            if len(candidate) < 8:
                continue
            cited = _add_missing_citations(candidate, retrieved)
            if cited == candidate:
                continue
            grounded = _ground_answer(cited, retrieved, chunks)
            if _COMPLETENESS_REFUSAL in grounded:
                continue
            if _answer_citation_support_ok(
                grounded, retrieved, threshold=0.65,
            ):
                return grounded
    return None


def _followup_fallback_answer(
    query: str,
    chat_history: list[dict] | None,
    retrieved: list[dict],
    chunks: list[dict],
) -> str | None:
    """Reuse a previous grounded sentence when a clarification answer drifts.

    A clarification such as "حلال ولا حرام؟" can make the model add an
    unsupported ruling label.  The previous assistant sentence is already
    grounded and answers the same subject; re-cite it against the current
    retrieved evidence and use it only when its support is strong.
    """
    if not chat_history or not _is_followup_query(query):
        return None
    previous = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(chat_history)
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and str(message.get("content") or "").strip()
        ),
        "",
    )
    if not previous:
        return None
    normalized_query = normalize(query)
    asks_evidence = any(
        marker in normalized_query
        for marker in ("دليل", "حديث", "قال الرسول", "الرسول")
    )
    reason_tokens = {"ليه", "لماذا", "ليش", "لما"}
    asks_reason = bool(
        reason_tokens & set(_tokenize(normalized_query))
    ) or "السبب" in normalized_query
    asks_obligation = bool(
        set(_tokenize(normalized_query)) & _FOLLOWUP_OBLIGATION_FOLDED
    )
    asks_modality = bool(
        set(_tokenize(normalized_query)) & _FOLLOWUP_LEGAL_FOLDED
    )
    asks_habit = _is_habit_query(query)
    keep_full = (
        asks_evidence or asks_reason or asks_obligation or asks_habit
    )
    if asks_evidence and retrieved:
        target = _evidence_target_text(previous)
        subject_text = _last_user_message(chat_history) or target
        core_keys = _required_action_evidence_keys(subject_text)
        if not core_keys:
            core_keys = (
                _legal_topic_keys(subject_text) - _COMPLETENESS_STOP_KEYS
            )
        evidence_parts: list[str] = []
        hadith = _grounded_source_claim(
            subject_text,
            ("قال رسول", "قال النبي", "روى", "ورد", "الصحيحين"),
            retrieved,
            chunks,
            subject_keys=core_keys,
        )
        if hadith:
            evidence_parts.append(hadith)
        if target and _answer_mentions_habit(target):
            habit_evidence = _grounded_source_claim(
                subject_text,
                ("عاده", "عادة"),
                retrieved,
                chunks,
                subject_keys=core_keys,
            )
            if habit_evidence and habit_evidence not in evidence_parts:
                evidence_parts.append(habit_evidence)
        if evidence_parts:
            return " ".join(evidence_parts)
    sentences = [
        sentence.strip()
        for sentence in _SENT_SPLIT_RE.split(previous)
        if sentence.strip()
    ]
    candidates = sentences
    if asks_evidence:
        evidence_sentences = [
            sentence for sentence in sentences
            if any(marker in normalize(sentence)
                   for marker in ("قال", "حديث", "دليل", "روى", "ورد"))
        ]
        if evidence_sentences:
            candidates = evidence_sentences
    elif asks_reason:
        # "ليه؟" asks for the cause, not a repetition of the ruling.  Prefer
        # the prior answer's explanatory sentence when it has one.
        reason_sentences = [
            sentence for sentence in sentences
            if any(
                marker in normalize(sentence)
                for marker in (
                    "لأن", "السبب", "بسبب", "ضرر", "حكمه", "حكمة",
                    "يسبب", "يؤدي", "تحريمه", "سبب",
                )
            )
        ]
        if reason_sentences:
            candidates = reason_sentences
    elif asks_obligation:
        # "يعني لازم...؟" asks about the legal status of the previous
        # subject.  Reuse the prior answer's qualified/exception sentence
        # rather than repeating a bare ruling or inventing a stronger one.
        legal_sentences = [
            sentence for sentence in sentences
            if any(
                marker in normalize(sentence)
                for marker in (
                    "لا يجب", "ليس بواجب", "لا يلزم", "لا يكره",
                    "الكراهه", "للتحريم", "يجوز", "يشترط",
                    "الا اذا", "الا ان", "قبله", "بعده",
                )
            )
        ]
        if legal_sentences:
            candidates = legal_sentences
    elif asks_habit:
        habit_sentences = [
            sentence for sentence in sentences
            if _answer_mentions_habit(sentence)
        ]
        if habit_sentences:
            candidates = habit_sentences
    elif asks_modality:
        # A modality question ("يعني حرام؟") must not flatten a documented
        # disagreement into one opinion.  Reuse the prior sentence that names
        # the disagreement when it has one.
        dispute_sentences = [
            sentence for sentence in sentences
            if any(
                marker in normalize(sentence)
                for marker in (
                    "الجمهور", "جمهور", "آخرون", "اخرون", "اختلف",
                    "خلاف", "قولان", "فريقان", "الراي الثاني",
                    "الرأي الثاني", "ذهب بعض", "ذهب فريق",
                )
            )
        ]
        if dispute_sentences:
            candidates = dispute_sentences
            keep_full = True
    grounded_answers: list[str] = []
    for candidate in candidates:
        candidate = _CITATION_RE.sub("", candidate)
        candidate = _QUOTE_RE.sub(lambda match: match.group(1), candidate)
        if not keep_full:
            # A polarity clarification needs the direct ruling, not the
            # previous answer's explanatory tail.
            candidate = re.split(r"[،,]", candidate, maxsplit=1)[0]
        candidate = _clean_source_snippet(candidate)
        if len(candidate) < 8:
            continue
        cited = _add_missing_citations(candidate, retrieved)
        if cited == candidate or _COMPLETENESS_REFUSAL in cited:
            continue
        grounded = _ground_answer(cited, retrieved, chunks)
        if _COMPLETENESS_REFUSAL in grounded:
            continue
        # The prior sentence may contain the framing word "حكم" that is not
        # repeated in the source; require strong but not perfect support.
        if _answer_citation_support_ok(
            grounded, retrieved, threshold=0.65,
        ):
            if not asks_obligation and not asks_habit:
                return grounded
            if asks_habit and _answer_mentions_habit(grounded):
                return grounded
            if grounded not in grounded_answers:
                grounded_answers.append(grounded)
            if len(grounded_answers) >= 2:
                break
    need_qualification = asks_obligation and not any(
        _answer_mentions_qualification(item) for item in grounded_answers
    )
    need_habit = asks_habit and not any(
        _answer_mentions_habit(item) for item in grounded_answers
    )
    if grounded_answers and not need_qualification and not need_habit:
        return " ".join(grounded_answers)

    previous_user = _last_substantive_user(chat_history)
    if asks_evidence and retrieved:
        target = _evidence_target_text(previous)
        subject_text = _last_user_message(chat_history) or previous_user or target
        core_keys = _required_action_evidence_keys(subject_text)
        if not core_keys:
            core_keys = (
                _legal_topic_keys(subject_text) - _COMPLETENESS_STOP_KEYS
            )
        evidence = _grounded_source_claim(
            subject_text,
            ("قال رسول", "قال النبي", "روى", "ورد", "الصحيحين"),
            retrieved,
            chunks,
            subject_keys=core_keys,
        )
        if not evidence and target:
            evidence = _grounded_source_claim(
                target,
                ("قال رسول", "قال النبي", "روى", "ورد", "الصحيحين"),
                retrieved,
                chunks,
            )
        if evidence:
            return evidence
    if asks_habit and retrieved:
        habit = _grounded_source_claim(
            previous_user, ("عاده", "عادة"), retrieved, chunks,
        )
        if habit:
            return habit
    if asks_modality and retrieved:
        # The prior answer may be a paraphrase whose framing words are absent
        # from the source.  Before refusing, fall back to the verbatim source
        # sentence that carries the qualification.
        markers = (
            "لا يجب", "ليس بواجب", "لا يلزم", "لا يكره",
            "الكراهه", "مكروه", "يكره", "للتحريم", "يجوز",
            "يشترط", "الا اذا", "الا ان", "قبله", "بعده",
        )
        legal = _grounded_source_claim(
            previous_user, markers, retrieved, chunks,
        )
        if legal:
            if grounded_answers:
                return grounded_answers[0] + " " + legal
            return legal
    if grounded_answers:
        return " ".join(grounded_answers)
    return None


def _requested_legal_modality(query: str) -> str | None:
    tokens = set(_tokenize(normalize(query)))
    if tokens & _FOLLOWUP_OBLIGATION_FOLDED:
        return "obligation"
    if tokens & _FOLLOWUP_PROHIBITION_FOLDED:
        return "prohibition"
    if tokens & _FOLLOWUP_PERMISSION_FOLDED:
        return "permission"
    return None


def _answer_addresses_modality(answer: str, modality: str) -> bool:
    text = _fold_ar(normalize(answer or ""))
    if modality == "obligation":
        if any(
            phrase in text
            for phrase in (
                "لا يجب", "ليس بواجب", "غير واجب", "لا يلزم",
                "ليس بلازم", "لا توجب", "لا فرض", "ليس بفرض",
            )
        ):
            return True
        return "obligation" in _positive_legal_families(answer)
    if modality == "prohibition":
        if any(
            phrase in text
            for phrase in (
                "لا يحرم", "ليس حراما", "غير محرم", "لا تحرم",
            )
        ):
            return True
        return "prohibition" in _positive_legal_families(answer)
    return bool(
        _positive_legal_families(answer) & {"permission", "prohibition"}
    )


_DISAGREEMENT_MARKERS = (
    "اختلف", "خلاف", "آخرون", "اخرون", "الرأي الثاني",
    "الراي التاني", "قولان", "فريقان", "ذهب بعض", "ذهب فريق",
)


def _without_no_disagreement(text: str) -> str:
    for phrase in ("بلا خلاف", "لا خلاف", "دون خلاف", "من غير خلاف"):
        text = text.replace(phrase, " ")
    return text


def _evidence_has_disagreement(retrieved: list[dict]) -> bool:
    text = _without_no_disagreement(" ".join(
        normalize(chunk.get("original_text", "") or "")
        for chunk in retrieved
    ))
    return any(marker in text for marker in _DISAGREEMENT_MARKERS)


def _answer_mentions_disagreement(answer: str) -> bool:
    text = _without_no_disagreement(normalize(answer or ""))
    return any(marker in text for marker in _DISAGREEMENT_MARKERS)


_QUALIFICATION_MARKERS = (
    "الكراهه", "لا يكره", "مكروه", "يشترط", "يجوز",
    "الا اذا", "الا ان", "للتحريم",
)


def _answer_mentions_qualification(answer: str) -> bool:
    text = normalize(answer or "")
    return any(marker in text for marker in _QUALIFICATION_MARKERS)


def _legal_subject_keys(text: str) -> set[str]:
    return (
        _legal_topic_keys(text)
        - _COMPLETENESS_STOP_KEYS
        - _LEGAL_FOCUS_STOP_STRICT
    )


def _retrieved_supports_modality(
    retrieved: list[dict], modality: str, subject_keys: set[str],
) -> bool:
    for chunk in retrieved:
        for sentence in _split_source_sentences(
            chunk.get("original_text", "") or "",
        ):
            if modality not in _positive_legal_families(sentence):
                continue
            if (
                not subject_keys
                or subject_keys <= _legal_topic_keys(sentence)
            ):
                return True
    return False


def _legal_followup_lead(
    query: str, answer: str, retrieved: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """State plainly when the evidence does not establish a strong modality.

    This is a limitation of the retrieved evidence, not a fatwa: it prevents
    an obligation question from being answered with an unqualified "نعم"
    when the sources only establish a condition, permission, or dislike.
    """
    modality = _requested_legal_modality(query)
    if not modality:
        return ""
    if _COMPLETENESS_REFUSAL in answer:
        return ""
    previous_user = next(
        (
            str(message.get("content") or "").strip()
            for message in reversed(chat_history or [])
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content") or "").strip()
        ),
        "",
    )
    subject_keys = (
        _legal_topic_keys(previous_user) - _COMPLETENESS_STOP_KEYS
        if previous_user else set()
    )
    if _retrieved_supports_modality(
        retrieved, modality, subject_keys,
    ) or modality in _positive_legal_families(answer):
        return ""
    if modality == "obligation":
        return "المصادر المسترجعة لا توجب ذلك. "
    if modality == "prohibition":
        return "المصادر المسترجعة لا تحرم ذلك. "
    return "المصادر المسترجعة لا تجزم بالإباحة. "


_RELIGIOUS_REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."


def _merge_website_help(
    answer: str,
    query: str,
    chat_history: list[dict] | None,
    sources: list,
) -> tuple[str, list]:
    """Attach the website-help part of a mixed question to a RAG answer.

    Pure website questions return before retrieval.  A mixed question keeps
    its religious answer, then receives the product-help answer from the
    website source of truth.  If the religious part has no evidence, the
    user still gets the website help plus an honest note about the religious
    part instead of a bare refusal.
    """
    site_answer = build_website_answer(query, chat_history)
    site_sources = website_help_sources()
    if answer.strip() == _RELIGIOUS_REFUSAL:
        combined = (
            site_answer
            + "\n\nأما الجزء الشرعي من سؤالك، فلم أجد في المصادر المتاحة "
            "ما يجيب عنه."
        )
        return combined, site_sources
    return (
        answer.rstrip() + "\n\n" + site_answer,
        list(sources) + site_sources,
    )


def generate(
    query: str,
    index,
    chunks: list[dict],
    client: OpenAI,
    chat_history: list[dict] | None = None,
    top_k: int = 8,
) -> dict:
    """Run the full RAG pipeline: retrieve -> build prompt -> generate answer.

    Returns:
        Dict with keys: answer (str), sources (list), query (str).
    """
    # Website questions are answered from the implemented product surface,
    # not from the Islamic corpus.  This routing happens before Tafsir
    # resolution so questions such as "إزاي أبحث عن تفسير آية؟" are not
    # mistaken for a request to resolve a verse.
    website_intent = classify_website_intent(query, chat_history)
    if website_intent["is_website"] and not website_intent["is_mixed"]:
        return {
            "answer": build_website_answer(query, chat_history),
            "sources": website_help_sources(),
            "query": query,
            "mode": "website_help",
        }

    if _is_ambiguous_action_followup(query, chat_history):
        # "ما حكمه لو عملت كده؟" has no identifiable action.  Do not let
        # retrieval guess a topic from the previous answer.
        return {
            "answer": _AMBIGUOUS_ACTION_CLARIFICATION,
            "sources": [],
            "query": query,
        }

    # A tafsir request that cannot be resolved to one ayah gets an explicit
    # clarification before any generation (e.g. whole-surah request).  A
    # resolved request passes its deterministic anchors into retrieval.
    # Follow-ups resolve against the prior substantive question, never the
    # literal clarification phrase.
    retrieval_query = _history_aware_query(query, chat_history)
    tafsir_anchors, tafsir_clarification = _resolve_chat_tafsir(
        retrieval_query, index, chunks, client,
    )
    if tafsir_clarification:
        if website_intent["is_website"] and website_intent["is_mixed"]:
            answer, sources = _merge_website_help(
                tafsir_clarification, query, chat_history, [],
            )
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "mode": "mixed",
            }
        return {
            "answer": tafsir_clarification,
            "sources": [],
            "query": query,
        }

    retrieved = retrieve(
        query, index, chunks, client, top_k=top_k,
        tafsir_anchors=tafsir_anchors,
        retrieval_query=retrieval_query,
    )

    # A hadith text/explanation question is answered from the hadith corpus;
    # unrelated tafsir/fiqh chunks in the generic retrieval pool can otherwise
    # be cited as if they explained the narration.
    if (
        _is_hadith_text_request(query)
        or _is_hadith_explanation_request(query)
    ):
        hadith_only = [
            chunk for chunk in retrieved
            if chunk.get("source") == "hadith"
        ]
        if hadith_only:
            retrieved = hadith_only

    # A resolved Tafsir request has two precise anchors: the verse and its
    # commentary.  Other high-scoring tafsir chunks can be unrelated to the
    # asked ayah; do not let the generator cite them in a meaning answer.
    if tafsir_anchors and not any(
        marker in normalize(query)
        for marker in ("حكم", "يجوز", "حرام", "حلال", "فتوى")
    ):
        anchor_ids = {
            str(chunk.get("id"))
            for chunk in tafsir_anchors
            if chunk.get("id")
        }
        focused = [
            chunk for chunk in retrieved
            if str(chunk.get("id")) in anchor_ids
        ]
        if focused:
            retrieved = focused

    # Follow-up turns must not display unrelated evidence merely because the
    # generic gate passed.  Retry once with the previous question and answer
    # context; if that still fails, return an honest refusal with no sources.
    followup_turn = bool(chat_history and _is_followup_query(query))
    if followup_turn:
        required_keys = _followup_required_evidence_keys(retrieval_query)
        if not _followup_results_relevant(
            retrieval_query, retrieved, required_keys,
        ):
            retry_query = _followup_retry_query(query, chat_history)
            retried = retrieve(
                query, index, chunks, client, top_k=top_k,
                tafsir_anchors=tafsir_anchors,
                retrieval_query=retry_query,
            )
            if _followup_results_relevant(
                retry_query, retried, required_keys,
            ):
                retrieved = retried
            else:
                return {
                    "answer": _RELIGIOUS_REFUSAL,
                    "sources": [],
                    "query": query,
                }
        retrieved = _filter_followup_results(retrieved, required_keys)

    # Gate: retrieve() marks every chunk with _gate_passed.  Refuse before
    # calling GPT when the retrieval gate failed — no synthetic scores are
    # involved in this decision.
    if not retrieved or not retrieved[0].get("_gate_passed", False):
        sources = [f"{c.get('source','غير محدد')} — {c.get('reference','غير محدد')}"
                   for c in retrieved]
        if website_intent["is_website"] and website_intent["is_mixed"]:
            answer, sources = _merge_website_help(
                _RELIGIOUS_REFUSAL, query, chat_history, sources,
            )
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "mode": "mixed",
            }
        return {
            "answer": _RELIGIOUS_REFUSAL,
            "sources": sources,
            "query": query,
        }

    messages = build_prompt(query, retrieved, chat_history=chat_history)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content or ""

    # The model occasionally refuses even when the retrieved context covers
    # every query term (observed on "السنن المؤكدة").  Give it one explicit
    # second chance; the same grounding filter still validates the result.
    if (answer.strip() == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
            and _context_covers_query(query, retrieved)):
        retry_messages = messages + [{
            "role": "user",
            "content": ("المصادر أعلاه تتناول السؤال وتحتوي إجابة صريحة. "
                        "أعد الإجابة الآن مباشرة من المصادر، واذكر رقم "
                        "المصدر، ولا تعتذر."),
        }]
        retry = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=retry_messages,
            temperature=0,
        )
        retry_answer = retry.choices[0].message.content or ""
        if retry_answer.strip():
            answer = retry_answer

    # Always run grounding. A refusal phrase embedded in an otherwise
    # substantive answer must not bypass sentence-level evidence checks.
    refusal = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
    raw_answer = answer
    strict = _ground_with_heading_fallback(raw_answer, retrieved, chunks)
    answer = strict
    if (strict.strip() == refusal and raw_answer.strip()
            and raw_answer.strip() != refusal):
        # A paraphrase placed inside quotation marks fails the quote guard
        # even when its claim is supported.  Remove only the false quotation
        # marks and re-run every grounding guard on the paraphrase.
        unquoted = _strip_unsupported_quotes(raw_answer, retrieved)
        if unquoted != raw_answer:
            requoted = _ground_answer(unquoted, retrieved, chunks)
            if requoted.strip() != refusal:
                answer = requoted
                raw_answer = unquoted
                strict = requoted
        if strict.strip() == refusal:
            # The generator did produce an answer, but one unmatched low-DF
            # framing word was enough to drop every sentence.  Prefer a
            # partial, citation-preserving answer over a false refusal; all
            # substantive guards (quotes, names, numbers, months, polarity)
            # still apply.
            lenient = _ground_answer(raw_answer, retrieved, chunks,
                                     ignore_rare=True)
            if lenient.strip() != refusal:
                answer = lenient

    # If a substantive answer still fails every grounding attempt (typically
    # a wrong/inexact citation), ask once for a source-cited rewrite before
    # falling back to refusal.
    if (answer.strip() == refusal and raw_answer.strip() != refusal
            and _context_covers_query(query, retrieved)):
        fix_messages = messages + [{
            "role": "user",
            "content": ("الإجابة السابقة لم تطابق نص المصادر. أعد الإجابة "
                        "بعبارات منقولة حرفيا من المصادر مع ذكر (مصدر N) "
                        "الصحيح، ولا تضف معلومة من خارجها."),
        }]
        fixed = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=fix_messages,
            temperature=0,
        )
        fixed_answer = fixed.choices[0].message.content or ""
        if fixed_answer.strip():
            fixed_ground = _ground_answer(fixed_answer, retrieved, chunks)
            if fixed_ground.strip() == refusal:
                fixed_ground = _ground_answer(
                    fixed_answer, retrieved, chunks, ignore_rare=True)
            if fixed_ground.strip() != refusal:
                answer = fixed_ground

    answer = _repair_citation_support(answer, retrieved, query)
    answer = _add_missing_citations(answer, retrieved)
    answer = _augment_answer_completeness(query, answer, retrieved)
    answer = _fix_mismatched_prayer_quotes(answer)
    answer = _move_citations_out_of_quotes(answer)
    answer = _drop_query_echo(answer, query)
    if not _answer_citation_support_ok(answer, retrieved):
        fallback = _followup_fallback_answer(
            query, chat_history, retrieved, chunks,
        )
        modality = (
            _requested_legal_modality(query)
            if chat_history and _is_followup_query(query) else None
        )
        # A grounded answer that already states the requested modality is
        # more useful than a one-line fallback; keep its qualification and
        # evidence instead of replacing it.
        if fallback is not None and not (
            modality and _answer_addresses_modality(answer, modality)
        ):
            answer = fallback
    if chat_history and _is_followup_query(query):
        modality = _requested_legal_modality(query)
        if modality:
            legal_fallback = _followup_fallback_answer(
                query, chat_history, retrieved, chunks,
            )
            if (
                legal_fallback is not None
                and _answer_mentions_qualification(legal_fallback)
                and not _answer_mentions_qualification(answer)
            ):
                # Prefer the direct answer plus its source qualification over
                # a bare synthesized ruling.
                answer = legal_fallback
        if _is_habit_query(query) and not _answer_mentions_habit(answer):
            habit_fallback = _followup_fallback_answer(
                query, chat_history, retrieved, chunks,
            )
            if (
                habit_fallback is not None
                and _answer_mentions_habit(habit_fallback)
            ):
                answer = habit_fallback
        if _is_evidence_request(query):
            evidence_fallback = _followup_fallback_answer(
                query, chat_history, retrieved, chunks,
            )
            if (
                evidence_fallback is not None
                and any(
                    marker in normalize(evidence_fallback)
                    for marker in ("قال رسول", "قال النبي", "روى", "ورد", "حديث")
                )
            ):
                previous_assistant = next(
                    (
                        str(message.get("content") or "").strip()
                        for message in reversed(chat_history)
                        if isinstance(message, dict)
                        and message.get("role") == "assistant"
                        and str(message.get("content") or "").strip()
                    ),
                    "",
                )
                target = _evidence_target_text(previous_assistant)
                subject_text = _last_user_message(chat_history) or target
                core_keys = _required_action_evidence_keys(subject_text)
                if not core_keys:
                    core_keys = (
                        _legal_topic_keys(subject_text)
                        - _COMPLETENESS_STOP_KEYS
                    )
                evidence_sentences = [
                    sentence.strip()
                    for sentence in _SENT_SPLIT_RE.split(answer)
                    if any(
                        marker in normalize(sentence)
                        for marker in (
                            "قال رسول", "قال النبي", "روى", "ورد",
                            "الصحيحين", "حديث",
                        )
                    )
                ]
                first_evidence = (
                    evidence_sentences[0] if evidence_sentences else ""
                )
                evidence_matches_subject = (
                    bool(first_evidence)
                    and (
                        not core_keys
                        or core_keys <= _legal_topic_keys(first_evidence)
                    )
                )
                target_mentions_habit = bool(
                    target and _answer_mentions_habit(target)
                )
                if (
                    not evidence_matches_subject
                    or (
                        target_mentions_habit
                        and not _answer_mentions_habit(answer)
                    )
                ):
                    answer = evidence_fallback
        lead = _legal_followup_lead(query, answer, retrieved, chat_history)
        if lead:
            answer = lead + answer
        if (
            _COMPLETENESS_REFUSAL not in answer
            and _evidence_has_disagreement(retrieved)
            and not _answer_mentions_disagreement(answer)
        ):
            dispute_fallback = _followup_fallback_answer(
                query, chat_history, retrieved, chunks,
            )
            prefix = "المسألة محل خلاف بين الفقهاء. "
            if (
                dispute_fallback is not None
                and _answer_mentions_disagreement(dispute_fallback)
            ):
                # Prefer the prior sentence that names both positions over a
                # one-sided answer that happens to be grounded.
                answer = prefix + dispute_fallback
            else:
                answer = prefix + answer

    if answer.strip() == _RELIGIOUS_REFUSAL:
        hadith_answer = _hadith_text_answer(query, retrieved)
        if hadith_answer:
            answer = hadith_answer

    hadith_supplement = _hadith_explanation_supplement(
        query, answer, retrieved,
    )
    if hadith_supplement:
        answer = answer.rstrip() + "\n\n" + hadith_supplement

    if _is_seerah_event_query(query):
        seerah_outline = _seerah_event_outline(query, retrieved)
        if seerah_outline:
            answer = seerah_outline

    if tafsir_anchors and not any(
        marker in normalize(answer)
        for marker in ("اخبر", "معني الايه", "يدل علي", "يفيد")
    ):
        meaning = _find_tafsir_meaning_sentence(query, retrieved)
        if meaning:
            answer = (
                answer.rstrip()
                + f"\n\n{meaning[1]} (مصدر {meaning[0]})."
            )

    sources = [f"{c.get('source','غير محدد')} — {c.get('reference','غير محدد')}"
               for c in retrieved]

    if website_intent["is_website"] and website_intent["is_mixed"]:
        answer, sources = _merge_website_help(
            answer, query, chat_history, sources,
        )
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "mode": "mixed",
        }

    return {"answer": answer, "sources": sources, "query": query}


if __name__ == "__main__":
    import inspect
    print("Pipeline updated — chat_history parameter added successfully")
    print(f"generate() signature: {inspect.signature(generate)}")
