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
import time
import unicodedata
from collections import defaultdict

from openai import OpenAI

from src.nlp.preprocessor import preprocess, normalize
from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search

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

# Query tokens stripped for retrieval purposes.
_QUERY_STOPWORDS = {
    "حكم", "ما", "هي", "هو", "من", "في", "على", "معنى", "هل",
    "متى", "كيف", "لماذا", "اين", "من",
}

# Sources whose lexical document is: normalize(reference) + " " + clean_text.
# For hadith and quran: clean_text only (references are book/surah codes,
# not topical text, and prepending them adds noise).
_SOURCES_WITH_REF_IN_LEXDOC = {"fiqh", "sira", "tafsir"}

SYSTEM_MESSAGE = (
    "أنت مساعد إسلامي موثوق. أجب بالعربية فقط، وبناء على المصادر المقدمة حصر، "
    "ولا تستخدم أي معلومة من معرفتك الخاصة حتى لو كنت متأكد منها. "
    "كل جملة في إجابتك يجب أن يدعمها نص صريح في أحد المصادر، واذكر رقم المصدر بين قوسين مثل (مصدر 2) بعد كل معلومة. "
    "لا تنسب لأي مصدر معلومة غير مكتوبة فيه. "
    "احذف أي جملة لا تجد لها سند صريح في المصادر. "
    "إذا أجابت المصادر عن السؤال فأجب منها مباشرة دون جملة الاعتذار. "
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
# BM25 index (lazy build, cached by (id(chunks), len(chunks)))
# Posting-list design: builds once in ~2s, searched in O(|q| * avg_postings)
# ===========================================================================
_bm25_cache: dict[tuple[int, int], dict] = {}


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
    for p in _CONJ_ARTICLE_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            return token[len(p):]
    if token.startswith("لل") and len(token) - 2 >= 2:
        return token[2:]
    if token.startswith("ال") and len(token) > 2:
        return token[2:]
    return token


def _tokenize_clean(text: str) -> list[str]:
    """Fast split of already-normalized clean_text — no regex, no normalize()."""
    return [_strip_al(w) for w in text.split() if len(w) >= 2]


def _tokenize(text: str) -> list[str]:
    """Normalize + split arbitrary text (used only for query tokenization)."""
    norm = normalize(text)
    return [_strip_al(w) for w in norm.split() if len(w) >= 2]


def _build_bm25(chunks: list[dict]) -> dict:
    """Build a BM25 inverted posting-list index.

    Lexical document composition:
      - fiqh / sira / tafsir:  _tokenize_clean(reference) + body_tokens[:CAP]
        Topically meaningful references (chapter headings, event names) give
        BM25 recall on references.  normalize() is NOT called on refs because
        stored refs have no tashkeel — _tokenize_clean() suffices and is 8×
        faster (Phase 1 sub-phase A was 8.8 s with normalize(); drops to ~1 s).
      - hadith / quran:        body_tokens[:CAP] only
        Their references are book codes — prepending floods the index with
        the token 'سنن' (previously DF=21,027 chunks before Fix A).

    Body tokens are capped at _BM25_BODY_TOKEN_CAP (200) per document.
    Phase 1 profiler showed tafsir avg=772 tokens → 4.3M posting entries
    and sub-phase C=54 s.  Cap reduces posting entries ~75 % for tafsir
    with negligible retrieval loss (early tokens are most discriminative).

    Also stores doc_tok per chunk so _bm25_search can apply PHRASE_BOOST
    without re-tokenizing.

    Index structure:
        postings:  {token: [(chunk_id, bm25_weight), ...]}
        idf:       {token: float}
        doc_tok:   {chunk_id: [token, ...]}
        N:         int
        _chunks:   {id: chunk_dict}
    """
    cache_key = (id(chunks), len(chunks))
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    tracemalloc.start()
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
        # Body: cap at _BM25_BODY_TOKEN_CAP to bound posting-list size.
        # Phase 1: tafsir avg=772 tok/doc → 4.3M entries → 54s sub-phase C.
        body_tokens = _tokenize_clean(
            c.get("clean_text") or c.get("original_text") or ""
        )[:_BM25_BODY_TOKEN_CAP]
        # Reference: use _tokenize_clean() (not normalize()) — refs have no
        # tashkeel so normalize() is a no-op that costs 8.8s (pyarabic).
        if src in _SOURCES_WITH_REF_IN_LEXDOC:
            ref_tokens = _tokenize_clean(c.get("reference") or "")
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
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"[bm25] built index in {build_time:.2f}s  (N={N})  "
        f"mem_current={current_mem/1e6:.1f}MB  mem_peak={peak_mem/1e6:.1f}MB"
    )

    idx = {
        "idf": idf, "postings": postings, "dl": dl, "avgdl": avgdl,
        "N": N, "_chunks": id_map, "doc_tok": doc_tok,
    }
    _bm25_cache[cache_key] = idx
    return idx


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
    de_prefixed = [_strip_al(t) for t in query_tokens]

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
# Lexical reference boost
# ===========================================================================

def _apply_lexical_boost(candidates: list[dict], query: str) -> list[dict]:
    """Boost candidates whose normalized reference contains all query keywords."""
    norm_query = normalize(query)
    keywords = [
        w for w in norm_query.split()
        if len(w) >= 3 and w not in _QUERY_STOPWORDS
    ]
    surah_m = re.search(r'سوره\s+(\S+)', norm_query)
    surah_token = surah_m.group(1) if surah_m else None

    boosted = []
    for cand in candidates:
        ref = cand.get("reference", "")
        norm_ref = normalize(ref)
        src = cand.get("source", "")
        boost = 0.0

        if keywords and all(
            re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$)', norm_ref)
            for kw in keywords
        ):
            boost += _REF_BOOST

        if surah_token and src in ("quran", "tafsir"):
            if re.search(
                r'(?:^|\s)' + re.escape(surah_token) + r'(?:\s|:|$)',
                norm_ref,
            ):
                boost += _REF_BOOST

        if boost > 0:
            cand = dict(cand)
            cand["score"] = cand.get("score", 0.0) + boost
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


def _select_diverse(
    candidates: list[dict],
    top_k: int,
    bypass_ids: set[str] | None = None,
    lex_top3_ids: set[str] | None = None,
    sem_top3_ids: set[str] | None = None,
) -> list[dict]:
    """Select top_k with source-type diversity.

    Selection order:
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
]

# Phrase-level aliases: only the *scoped* bigram replacement below is
# corpus-grounded.  'السنن المؤكدة' in the prayer chapter maps to the rawatib
# list (fiqh_160/164: 'السنن الراتبة', 'السنن والرواتب'), while the bare word
# مؤكدة is NOT a global synonym of راتبة — the corpus also uses 'سنة مؤكدة'
# for الوتر/الجماعة/الكسوف/العقيقة.  Applied only when the two words are
# adjacent with no intervening negation (e.g. 'غير المؤكدة' is untouched).
_PHRASE_ALIASES = [
    (("سنن", "مؤكده"), (("سنن", "راتبه"), ("سنن", "رواتب"))),
]


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
    for i in range(len(query_tokens) - 1):
        for pat, alts in _PHRASE_ALIASES:
            if query_tokens[i] == pat[0] and query_tokens[i + 1] == pat[1]:
                for alt_seq in alts:
                    out.append(query_tokens[:i] + list(alt_seq) + query_tokens[i + 2:])
    return out


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
    top_k: int = 5,
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
        8. Lexical reference boost.
        9. Source-type diverse selection (Phase 3I policy: injected surah
            chunks exempt; lexical/semantic top-3 admitted ahead of score
            order with a quota of 1 per source type; total cap 3 per source
            type during score-order fill; relaxes only to fill empty slots).
        10. Retrieval gate: pass if cosine >= MIN_COSINE_SCORE OR
            BM25 best covers >= MIN_LEXICAL_TOKENS query tokens.
            The result is exposed as `_gate_passed` on every chunk; real
            similarity scores are returned unmodified (never zeroed or
            replaced by synthetic constants).

    Returns:
        List of top_k chunk result dicts.
    """
    processed = preprocess(query)
    clean_query = processed["normalized"]
    query_tokens = [
        _strip_al(w) for w in (processed.get("tokens_clean") or clean_query.split())
        if w not in _QUERY_STOPWORDS and len(w) >= 2
    ]

    # Fix A: detect surah query
    surah_num = _resolve_surah_query(query, chunks)
    effective_top_k = 8 if surah_num is not None else top_k

    # --- Semantic retrieval ---
    emb_norm = embed_text(clean_query, client)
    emb_orig = embed_text(query, client)
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

    # --- Retrieval gate ---
    needed_tokens = 1 if len(query_tokens) == 1 else MIN_LEXICAL_TOKENS
    cosine_ok = best_cosine >= MIN_COSINE_SCORE
    lexical_ok = best_lex_coverage >= needed_tokens
    gate_condition = "cosine" if cosine_ok else ("lexical" if lexical_ok else "FAIL")
    print(f"[retrieve] gate={gate_condition}  best_cosine={best_cosine:.4f}  lex_coverage={best_lex_coverage}/{len(query_tokens)}")

    # --- Optional query rewriting: rewrites join RRF as extra rankings ---
    extra_ranked: list[tuple[list[dict], float]] = []
    # Static classical-synonym expansion (deterministic, no LLM).
    for _alt in _expand_synonyms(query_tokens):
        _alt_res = _bm25_search(_alt, chunks, top_k=50)
        if _alt_res:
            extra_ranked.append((_alt_res, _REWRITE_BM25_RRF_WEIGHT))
    if USE_QUERY_REWRITE:
        rewrites = _rewrite_query(query, client)
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
                for r in phrase_results:
                    cid = r["id"]
                    existing = best_by_id.get(cid)
                    if existing is None or r["score"] > existing.get("score", 0):
                        best_by_id[cid] = r
            except Exception:
                pass

    # --- RRF fusion (original semantic + BM25 + rewrites, one score scale) ---
    pool = _rrf_fuse(sem_ranked, lex_results, lex_top3_ids, extra_ranked)

    # --- Surah injection (bypasses diversity cap) ---
    injected_ids: set[str] = set()
    if surah_num is not None:
        pool, injected_ids = _inject_surah_chunks(surah_num, chunks, pool)

    # --- Lexical reference boost ---
    pool = _apply_lexical_boost(pool, query)

    # --- Source-type diverse selection (lexical/semantic top-3 bypass) ---
    sem_top3_ids = {c["id"] for c in sem_ranked[:3]}
    selected = _select_diverse(
        pool, effective_top_k,
        bypass_ids=injected_ids,
        lex_top3_ids=lex_top3_ids,
        sem_top3_ids=sem_top3_ids,
    )

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
    user_message = (
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

# Arabic / Latin sentence-end delimiters used for splitting
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?\u061f\n])')
_GROUNDING_MIN_CONTENT_TOKENS = 3

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
# so only this evidence-backed pair is admitted.  See the Phase E.1
# false-rejection analysis and Phase E.3 audit reports.
_GROUNDING_CURATED_ALIASES: dict[str, tuple[str, ...]] = {
    "نزول": ("نزلت",),
}

# Polarity-inverting particles (غير/بدون), tolerant of و/ف prefixes.  Kept
# separate from _is_negation_token: غير/بدون do not negate a verb window,
# they invert the polarity of the following noun ("غير مستحب" vs "مستحب").
_GROUNDING_INVERSION_TOKENS = {"غير", "بدون"}

# Entity anchors: a claim token directly following one of these is a
# named-entity mention whose support must be verifiable in the cited source
# (follower presence, not corpus frequency).
_GROUNDING_ENTITY_ANCHORS = {"ابن", "بنت", "ابي", "ابو", "ام", "عبد", "بن"}

# Windowed in-order quote alignment tolerates at most this many extra source
# tokens between the first and last aligned span token (covers Arabic
# insertions like "أهل العلم" / conjunction particles).  Fixed a priori.
_GROUNDING_SPAN_MAX_GAP = 6

# Quoted-span extraction: ASCII double quotes plus Arabic/French guillemets
# and curly quotes.
_QUOTE_RE = re.compile(
    r'["\u00ab\u00bb\u201c\u201d]([^"\u00ab\u00bb\u201c\u201d]{6,})'
    r'["\u00ab\u00bb\u201c\u201d]')

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

_GROUNDING_IGNORED_TOKENS = {"مصدر", "مصادر", "المصدر", "بمصدر"}

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


def _fold_ar(token: str) -> str:
    """Fold orthographic variance: combining marks, Unicode format characters
    (category Cf, e.g. the RTL mark U+200F glued to some corpus tokens),
    tatweel, Arabic digits."""
    t = unicodedata.normalize("NFKD", token)
    t = "".join(
        ch for ch in t
        if not unicodedata.combining(ch)
        and unicodedata.category(ch) != "Cf"
    )
    return t.replace("\u0640", "").translate(_GROUNDING_DIGITS_TRANS).strip()


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
    for suf in _GROUNDING_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 2:
            t = t[:-len(suf)]
            break
    t = t[0] + t[1:].replace("ا", "")
    if len(t) > 3 and t[0] in "يتن":
        t = t[1:]
    return t


_NEGATION_TOKENS = {"لا", "لم", "لن", "ليس", "ليست"}


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


def _build_group_canon() -> dict[str, str]:
    """Map every synonym-group member's match key to the group's canonical
    key (first member), so وقعة/معركة/غزوة all compare equal in grounding."""
    canon: dict[str, str] = {}
    for grp in _SYNONYM_GROUPS:
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


def _numbers_normalized(tokens: list[str]) -> list[str]:
    """Replace spelled number words and digit tokens by canonical decimal
    strings so that 'سبع عشرة' == 'سابع عشر' == '١٧' == '17'.

    Handles prefixed words (بسبع) and the compound unit+tens form
    (سبع وعشرين -> 27, خمسة وثلاثون -> 35).
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        f = _fold_ar(tokens[i])
        if f.isdigit():
            out.append(str(int(f)))
            i += 1
            continue
        tens = _num_lookup(f, _GROUNDING_TENS_WORDS)
        if tens is not None:
            out.append(str(tens))
            i += 1
            continue
        unit = _num_lookup(f, _GROUNDING_UNIT_WORDS)
        if unit is not None:
            nxt = _fold_ar(tokens[i + 1]) if i + 1 < len(tokens) else ""
            nxt_tens = _num_lookup(nxt, _GROUNDING_TENS_WORDS)
            if nxt in ("عشر", "عشره") or nxt_tens == 10:
                out.append(str(10 + unit))
                i += 2
                continue
            if nxt_tens is not None and nxt_tens >= 20 and nxt.startswith("و"):
                out.append(str(nxt_tens + unit))
                i += 2
                continue
            out.append(str(unit))
            i += 1
            continue
        out.append(tokens[i])
        i += 1
    return out


def _ground_answer(answer: str, retrieved: list[dict],
                   corpus_chunks: list[dict] | None = None) -> str:
    """Post-generation grounding filter (tolerant matcher).

    For each sentence in *answer*:
      1. Extract cited source numbers (maSdar markers).  Keep only those
         within 1..len(retrieved).  Uncited sentences must be supported by
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
         غير/بدون-governed claim token conflicts with bare-affirmative source
         occurrences (mixed sets stay conservative), and a negated claim token
         requires a negated source occurrence.  The family exemption is
         polarity-filtered and can never bypass a polarity, quote or entity
         rejection.  The rare/exemption checks need *corpus_chunks*; without
         it a warning is printed and the rare check is skipped (never
         silently guessed).
      5. Keep if coverage >= GROUNDING_MIN_COVERAGE AND all strict checks
         pass for at least one reference option.  Sentences with no content
         tokens are kept unconditionally.

    After filtering, strip all (maSdar N) markers from the kept text.
    If nothing survives, return the canonical refusal sentence.
    """
    REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
    n_chunks = len(retrieved)

    # Pre-compute per retrieved chunk: tolerant key / number sets, strict key
    # lists, polarity marks (right-only negation-adjacent positions plus
    # غير/بدون-inverted positions), negation/inversion follower surfaces, the
    # token stream and the surface set (P5v5).
    chunk_keys: list[set[str]] = []
    chunk_nums: list[set[str]] = []
    chunk_strict_lists: list[list[str]] = []
    chunk_pol_adj: list[set[int]] = []
    chunk_neg_adj: list[set[int]] = []
    chunk_neg_follow: list[set[str]] = []
    chunk_inv_follow: list[set[str]] = []
    chunk_surface: list[set[str]] = []
    chunk_toks: list[list[str]] = []
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
                if p + 1 < len(toks):
                    pol_adj.add(p + 1)
                    neg_adj.add(p + 1)
                    neg_follow.add(toks[p + 1])
            if _is_inversion_token(t) and p + 1 < len(toks):
                pol_adj.add(p + 1)
                inv_follow.add(toks[p + 1])
        for t in toks:
            k = _gkey(t)
            keys.add(k)
            if k.isdigit():
                nums.add(k)
        chunk_keys.append(keys)
        chunk_nums.append(nums)
        chunk_strict_lists.append([_strict_key(t) for t in toks])
        chunk_pol_adj.append(pol_adj)
        chunk_neg_adj.append(neg_adj)
        chunk_neg_follow.append(neg_follow)
        chunk_inv_follow.append(inv_follow)
        chunk_surface.append({t for t in toks})
        chunk_toks.append(toks)

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

        # Extract valid citation indices (1-based -> 0-based)
        cited_indices: list[int] = []
        for m in _CITATION_RE.finditer(sent):
            n = int(m.group(1))
            if 1 <= n <= n_chunks:
                cited_indices.append(n - 1)

        # Content tokens (citation markers removed, numbers canonicalised)
        body = _CITATION_RE.sub("", sent_stripped)
        body_tokens_all = _numbers_normalized(_tokenize(body))
        content_tokens = [
            t for t in body_tokens_all
            if t not in _QUERY_STOPWORDS
            and t not in _GROUNDING_IGNORED_TOKENS
            and (len(t) >= 3 or t.isdigit())
        ]
        content_tokens = _apply_grounding_aliases(content_tokens)

        # Claim-side polarity maps: content positions directly following a
        # negation particle (right-only) or directly governed by a غير/بدون
        # inversion.  Entity mentions: a content token directly following a
        # name anchor must be supported by the cited source.
        claim_neg_positions: set[int] = set()
        pos_map: list[int] = []
        ci = 0
        for t in body_tokens_all:
            if (t not in _QUERY_STOPWORDS
                    and t not in _GROUNDING_IGNORED_TOKENS
                    and (len(t) >= 3 or t.isdigit())):
                pos_map.append(ci)
                ci += 1
            else:
                pos_map.append(-1)
        for q, t in enumerate(body_tokens_all):
            if _is_negation_token(t):
                nb = q + 1
                if nb < len(body_tokens_all) and pos_map[nb] >= 0:
                    claim_neg_positions.add(pos_map[nb])
        claim_inv_positions: set[int] = set()
        for q, t in enumerate(body_tokens_all):
            if (_is_inversion_token(t) and q + 1 < len(body_tokens_all)
                    and pos_map[q + 1] >= 0):
                claim_inv_positions.add(pos_map[q + 1])
        entity_followers: list[str] = []
        for q, t in enumerate(body_tokens_all):
            a = _fold_ar(t)
            while len(a) > 2 and a[0] in "وف":
                a = a[1:]
            if (a in _GROUNDING_ENTITY_ANCHORS and q + 1 < len(body_tokens_all)
                    and pos_map[q + 1] >= 0):
                entity_followers.append(body_tokens_all[q + 1])

        # Vacuous pass: no content tokens to check
        if len(content_tokens) < _GROUNDING_MIN_CONTENT_TOKENS:
            kept.append(sent)
            print(f"[ground] cov=1.00 KEEP (vacuous) | {sent_stripped[:40]!r}")
            continue

        token_keys = [_gkey(t) for t in content_tokens]
        claim_strict = [_strict_key(t) for t in content_tokens]
        digit_tokens = [t for t in content_tokens if t.isdigit()]
        month_positions = [
            j for j, t in enumerate(content_tokens)
            if _fold_ar(t) in _GROUNDING_MONTHS
        ]

        # Quoted spans: content tokens of each quoted span.  Spans with >= 3
        # content tokens verify as in-order windows inside ONE cited chunk;
        # shorter spans only need each token to occur somewhere in one chunk
        # (too short to carry sequence structure).
        quote_spans: list[list[str]] = []
        for span in _QUOTE_RE.findall(sent):
            qt = [t for t in _numbers_normalized(_tokenize(span))
                  if t not in _QUERY_STOPWORDS
                  and t not in _GROUNDING_IGNORED_TOKENS
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
        best_cov = 0.0
        best_state: dict = {
            "digits_ok": False, "months_ok": False, "neg_ok": True,
            "quote_ok": True, "entity_ok": True, "rare": [],
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
            ref_inv_follow: set[str] = (
                set().union(*(chunk_inv_follow[i] for i in opt)) if opt else set()
            )
            ref_neg_follow: set[str] = (
                set().union(*(chunk_neg_follow[i] for i in opt)) if opt else set()
            )
            matched = sum(1 for k in token_keys if k in ref_keys)
            coverage = matched / len(content_tokens)
            digits_ok = all(t in ref_nums for t in digit_tokens)
            months_ok = all(token_keys[j] in ref_keys for j in month_positions)
            # Negation flip, inversion-aware: an affirmative claim token whose
            # every source occurrence is polarity-marked rejects the option;
            # a غير/بدون-governed claim token rejects only when a source
            # occurrence is bare-affirmative (source negation or the same
            # inversion = matching negative polarity).  Mixed occurrence sets
            # stay conservative.  (The comparison uses the strict surface key
            # so stem collisions like يعيد/عيدين can never cause a false
            # flip; claim-side negated tokens are skipped.)
            neg_ok = True
            for j, k in enumerate(token_keys):
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
            # Quote verification: every quoted span with >= 3 content tokens
            # must align as an in-order windowed sequence inside ONE cited
            # chunk (per-chunk alignment rejects quotes stitched across
            # sources); the aligned mode allows the same narrow seed pairs as
            # the family exemption.  Shorter spans are support-only.
            quote_ok = True
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
            entity_ok = True
            for ftok in entity_followers:
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
            # by a source variant under the same inversion/negation), the
            # Phase E.4 curated single-pair alias (same polarity filters),
            # and the material-addition guard: an unmatched token in the
            # uncertain band (300 < df <= 500) is tolerated only in a
            # well-covered sentence (coverage >= 0.80), so a genuinely common
            # material word and an unsupported swap in a weakly-covered
            # sentence stay distinguishable.
            rare: list[tuple[str, int]] = []
            if postings is not None:
                for j, t in enumerate(content_tokens):
                    if t.isdigit() or token_keys[j] in ref_keys:
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
                        rare.append((t, d))
            if coverage >= best_cov:
                best_cov = coverage
                best_state = {"digits_ok": digits_ok, "months_ok": months_ok,
                              "neg_ok": neg_ok, "quote_ok": quote_ok,
                              "entity_ok": entity_ok, "rare": rare}
            if (
                coverage >= GROUNDING_MIN_COVERAGE
                and digits_ok and months_ok and neg_ok and quote_ok
                and entity_ok and not rare
            ):
                kept_this = True
                best_cov = coverage
                best_state = {"digits_ok": digits_ok, "months_ok": months_ok,
                              "neg_ok": neg_ok, "quote_ok": quote_ok,
                              "entity_ok": entity_ok, "rare": rare}
                break

        digits_ok = best_state["digits_ok"]
        months_ok = best_state["months_ok"]
        neg_ok = best_state["neg_ok"]
        quote_ok = best_state["quote_ok"]
        entity_ok = best_state["entity_ok"]
        rare = best_state["rare"]
        tag = "KEEP" if kept_this else "DROP"
        rare_s = ""
        if rare:
            rare_s = " rare=" + ",".join(f"{t}:{d}" for t, d in rare[:4])
        print(
            f"[ground] cov={best_cov:.2f} {tag} "
            f"digits={'ok' if digits_ok else 'fail'} "
            f"months={'ok' if months_ok else 'fail'} "
            f"neg={'ok' if neg_ok else 'flip'} "
            f"quote={'ok' if quote_ok else 'fail'} "
            f"entity={'ok' if entity_ok else 'fail'}{rare_s}"
            f" | {sent_stripped[:40]!r}"
        )
        if kept_this:
            kept.append(sent)

    if not kept:
        return REFUSAL

    # Re-join and strip all citation markers from the final text
    joined = " ".join(k.strip() for k in kept).strip()
    joined = re.sub(r'(?:(?<=\s)|^)\d+[.)]\s*(?=(?:\d+[.)]|$))', '', joined).strip()
    cleaned = _CITATION_RE.sub("", joined).strip()
    return cleaned


# ===========================================================================
# generate()
# ===========================================================================

def generate(
    query: str,
    index,
    chunks: list[dict],
    client: OpenAI,
    chat_history: list[dict] | None = None,
    top_k: int = 5,
) -> dict:
    """Run the full RAG pipeline: retrieve -> build prompt -> generate answer.

    Returns:
        Dict with keys: answer (str), sources (list), query (str).
    """
    retrieved = retrieve(query, index, chunks, client, top_k=top_k)

    # Gate: retrieve() marks every chunk with _gate_passed.  Refuse before
    # calling GPT when the retrieval gate failed — no synthetic scores are
    # involved in this decision.
    if not retrieved or not retrieved[0].get("_gate_passed", False):
        sources = [f"{c.get('source','غير محدد')} — {c.get('reference','غير محدد')}"
                   for c in retrieved]
        return {
            "answer": "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.",
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

    # Post-generation grounding check: filter unsupported sentences,
    # then strip citation markers before returning to the caller.
    REFUSAL_PREFIX = "لم أجد في المصادر المتاحة"
    if REFUSAL_PREFIX not in answer:
        answer = _ground_answer(answer, retrieved, chunks)

    sources = [f"{c.get('source','غير محدد')} — {c.get('reference','غير محدد')}"
               for c in retrieved]

    return {"answer": answer, "sources": sources, "query": query}


if __name__ == "__main__":
    import inspect
    print("Pipeline updated — chat_history parameter added successfully")
    print(f"generate() signature: {inspect.signature(generate)}")
