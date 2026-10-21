"""Offline unit tests for the Basira RAG retrieval/grounding internals.

Runs entirely offline (no API, no network) with synthetic data.  These tests
pin down the mechanics that the eval harness measures statistically:

  * RRF semantic top-3 signal: bounded, only for ranks <= 3, absent when the
    semantic list is empty (lexical-only runs must be unaffected).
  * Diversity selection (Phase 3I policy): per-source-type cap 3 during the
    capped fill, pool-order fill, relaxation when slots remain, injected
    bypass exempt from the cap, lex/sem top-3 admitted ahead of score order
    with a quota of 1 per source type (further same-type bypass chunks
    compete under the cap), determinism.
  * Retrieval gate: cosine pass/fail and lexical-coverage pass/fail through
    the REAL FAISS search path on a tiny synthetic index (real vectors, real
    cosine scores — no synthetic score constants).
  * Terminology expansion: synonym groups and the scoped phrase alias
    (سنن مؤكدة -> سنن راتبة), with NO global مؤكدة -> راتبة mapping.
  * Grounding: match-key equivalences, number canonicalisation, scoped
    phrase alias, wrong month/day rejection, subject-swap rare-token
    rejection (with corpus df supplied), relation-inversion rejection, and
    per-chunk support for uncited sentences (no union across chunks).
  * Variant-generator ordering: plural/verbal suffixes (وا/ون/ين) and
    possessives are stripped BEFORE rasm alef removal, so اجتنبوا grounds
    against اجتنب and بطونهم against بطون (grounding replay included).
  * Embedding interface: model/dimension constants, request pass-through,
    token truncation, query normalization in search(), fail-fast on encoder
    errors.
  * Generation branches (deterministic, NO LLM): gate-failure refusal
    without an answering call, prompt construction (question, sources,
    refusal instruction, citation form), model/temperature, grounding
    fallback to refusal, refusal pass-through, rewrite-stage degradation.
    These validate the surrounding pipeline logic only — they do NOT
    validate the real LLM's behavior.
  * Multi-evidence selection: same-type and cross-type evidence pairs reach
    the final context together under a mixed query vector; guarantee-
    overflow behavior for lex/sem top-3 (same-type guarantees capped at 3,
    complementary source admitted, weakest same-type guarantees dropped).
  * Cached query-embedding file interface (shape/dimension validation,
    mean-vector normalization).
  * Grounding negation-flip check: negation particle detection (only
    لا/لم/لن/ليس/ليست, tolerating و/ف), strict surface key (no stemming
    collisions), affirmative-flip rejection, same-polarity claims kept,
    negation elsewhere in the chunk not poisoning other tokens.
  * RRF determinism: exactly-tied fused scores keep insertion order (no
    PYTHONHASHSEED-dependent tie order / pool_rank jitter).
  * Curated grounding alias (Phase E.4): the single audited pair
    نزول -> نزلت is admitted for the rare-token family exemption only —
    exact folded match, no bare-stem/reverse/suffix expansion, no effect
    on quote verification or the entity guard, and the Phase E.3 collision
    falsifiers (صوام/صامت، راوي/رايت، ثواب/ثابت، تورك/تركت) stay dropped.
  * BM25 single-flight initialization (RAG-03 R1): concurrent cold callers
    build exactly one index, share the cached object, recover from failed
    builds (no poisoned entry, lock released), and never stop a tracemalloc
    window they did not start.
  * Startup prewarming and readiness (RAG-03 R2): ASGI lifespan startup
    prewarms the canonical corpus through the same R1 cache exactly once,
    readiness is false before initialization and after shutdown, becomes
    true only after a successful prewarm, a failed prewarm never reports
    ready (fail-fast startup, released lock, no poisoned cache), and
    repeated lifecycle calls reuse the prewarmed index without rebuilding.
  * RAG-02 endpoint validation: tafsir/context grounded against the exact
    retrieved evidence with the endpoint refusal on rejection; verify
    correct_explanation grounded, fabricated citations replaced by
    retrieved references, enum-like fields validated with safe defaults,
    empty retrieval forced to غير موجود, deterministic truncation
    precedence preserved, and the /api/chat grounding path unchanged.

Usage:
    PYTHONDONTWRITEBYTECODE=1 venv/bin/python src/rag/offline_tests.py

Exit code 0 when all tests pass, 1 otherwise.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import time
import tracemalloc
from types import SimpleNamespace

sys.path.insert(0, ".")

from fastapi.testclient import TestClient  # noqa: E402

import src.rag.model_api as M  # noqa: E402
import src.rag.pipeline as P  # noqa: E402
import src.rag.tafsir as G  # noqa: E402
import src.rag.verify_full as V  # noqa: E402
from src.embeddings.embedder import (  # noqa: E402
    DIM,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    embed_text,
    truncate_to_tokens,
)
from src.vector_db.faiss_store import build_index  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [ok  ] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}")


def _chunk(cid: str, ctype: str, text: str, vec=None, ref: str = "مرجع") -> dict:
    c = {
        "id": cid,
        "source": ctype,
        "reference": ref,
        "original_text": text,
        "clean_text": text,
        "metadata": {},
    }
    if vec is not None:
        c["embedding"] = vec
    return c


# ---------------------------------------------------------------------------
# 1. RRF semantic signal
# ---------------------------------------------------------------------------

def test_rrf_sem_signal() -> None:
    print("\n[1] RRF semantic top-3 signal")
    sem = [{"id": f"s{i}"} for i in range(1, 6)]
    lex: list[dict] = []
    fused = {c["id"]: c["score"] for c in P._rrf_fuse(sem, lex, set())}
    # ranks 1..3 receive +3/(K+r); ranks 4..5 only the base 1/(K+r).
    # Chunks absent from the lexical list get lex_term = 1/(2*_RRF_K).
    lex_missing = 1.0 / (2 * P._RRF_K)
    check("s1 ranked first", max(fused, key=fused.get) == "s1")
    check("top3 boosted above rank4+",
          min(fused[f"s{i}"] for i in (1, 2, 3)) > max(fused[f"s{i}"] for i in (4, 5)))
    boost1 = fused["s1"] - 1.0 / (P._RRF_K + 1) - lex_missing
    check("boost bounded by _SEM_SIG_WEIGHT/(K+1)",
          abs(boost1 - P._SEM_SIG_WEIGHT / (P._RRF_K + 1)) < 1e-12)

    # lexical-only run: empty semantic list must contribute no boost term
    lex2 = [{"id": f"l{i}"} for i in range(1, 4)]
    fused2 = {c["id"]: c["score"] for c in P._rrf_fuse([], lex2, set())}
    base = 1.0 / (2 * P._RRF_K) + 1.0 / (P._RRF_K + 1)
    check("no sem signal without semantic list", abs(fused2["l1"] - base) < 1e-12)

    # Deterministic tie order: two chunks tied at the same fused score (one per
    # equal-weight extra list) must come out in insertion order, never in
    # PYTHONHASHSEED-dependent set order (regression: pool_rank jitter).
    tied = P._rrf_fuse(
        [], [], set(),
        [([{"id": "ya"}], 1.0), ([{"id": "yb"}], 1.0)],
    )
    check("tied fused scores keep insertion order",
          [c["id"] for c in tied] == ["ya", "yb"])
    check("tied scores are actually equal",
          abs(tied[0]["score"] - tied[1]["score"]) < 1e-12)


# ---------------------------------------------------------------------------
# 2. Diversity selection
# ---------------------------------------------------------------------------

def _mk(cid: str, ctype: str, score: float) -> dict:
    return {"id": cid, "source": ctype, "score": score}


def test_selection() -> None:
    print("\n[2] Diversity selection")
    pool = [
        _mk("f1", "fiqh", 4.0), _mk("f2", "fiqh", 3.0),
        _mk("f3", "fiqh", 2.9), _mk("f4", "fiqh", 2.8),
        _mk("h1", "hadith", 2.7), _mk("h2", "hadith", 2.6),
    ]
    selected = P._select_diverse(list(pool), 4)
    ids = {c["id"] for c in selected}
    check("per-source cap 3: fourth fiqh excluded, hadith admitted",
          ids == {"f1", "f2", "f3", "h1"})

    # lex-top3 chunk is admitted ahead of score order, but bypass chunks
    # consume the per-type cap too (f4 cannot follow f3 into the same quota).
    selected = P._select_diverse(list(pool), 4, lex_top3_ids={"f3", "f4"})
    ids = {c["id"] for c in selected}
    check("lex-top3 admitted first, cap enforced beyond it",
          ids == {"f1", "f2", "f3", "h1"})

    selected = P._select_diverse(list(pool), 3, sem_top3_ids={"f4"})
    ids = {c["id"] for c in selected}
    check("sem-top3 bypass: f4 selected despite fiqh cap",
          "f4" in ids and len(ids) == 3)

    # all one type: cap never shrinks the result (relaxation fills slots)
    pool_same = [_mk(f"x{i}", "fiqh", 10.0 - i) for i in range(6)]
    selected = P._select_diverse(list(pool_same), 5)
    check("relaxation fills to top_k when slots remain", len(selected) == 5)

    # Regression (Phase 3I root cause): when both top-3 lists are dominated
    # by near-duplicate variants of ONE source type, a complementary
    # lower-ranked chunk of another source type must still get a slot.
    flood = [_mk(f"h{i}", "hadith", 5.0 - i) for i in range(1, 6)]
    flood.append(_mk("f1", "fiqh", 0.5))
    selected = P._select_diverse(
        list(flood), 4, lex_top3_ids={"h1", "h2", "h3"}
    )
    ids = {c["id"] for c in selected}
    check("same-type top-3 flood keeps complementary source slot",
          ids == {"h1", "h2", "h3", "f1"})

    # A third same-source chunk of the same passage must survive (cap 3),
    # while a fourth must not; the complementary source keeps its slot.
    deep = [
        _mk("t1", "tafsir", 5.0), _mk("t2", "tafsir", 4.0),
        _mk("t3", "tafsir", 3.9), _mk("t4", "tafsir", 3.8),
        _mk("s1", "sira", 1.0),
    ]
    selected = P._select_diverse(list(deep), 4, sem_top3_ids={"t1", "t2", "t3"})
    ids = {c["id"] for c in selected}
    check("cap 3 keeps third same-type evidence, excludes fourth",
          "t3" in ids and "t4" not in ids and "s1" in ids)

    # Injected surah chunks are query-determined and exempt from the cap.
    inj = [_mk(f"q{i}", "quran", 9.0 - i) for i in range(4)]
    inj += [_mk("h1", "hadith", 2.0), _mk("h2", "hadith", 1.0)]
    selected = P._select_diverse(
        list(inj), 5, bypass_ids={"q0", "q1", "q2", "q3"}
    )
    ids = {c["id"] for c in selected}
    check("injected chunks exempt from per-type cap",
          {"q0", "q1", "q2", "q3"} <= ids)

    # Deterministic output for identical input (stable sort + fixed order).
    r1 = [c["id"] for c in P._select_diverse(
        list(flood), 4, lex_top3_ids={"h1", "h2", "h3"})]
    r2 = [c["id"] for c in P._select_diverse(
        list(flood), 4, lex_top3_ids={"h1", "h2", "h3"})]
    check("selection deterministic across runs", r1 == r2)


# ---------------------------------------------------------------------------
# 3. Retrieval gate through the REAL FAISS path (synthetic vectors)
# ---------------------------------------------------------------------------

def test_gate_real_faiss() -> None:
    print("\n[3] Retrieval gate (real FAISS, synthetic vectors)")
    # Three axis-aligned embeddings in 3D.
    chunks = [
        _chunk("c0", "fiqh", "محتوى الفقه", vec=[1.0, 0.0, 0.0]),
        _chunk("c1", "fiqh", "محتوى الفقه", vec=[0.0, 1.0, 0.0]),
        _chunk("c2", "hadith", "محتوى الحديث", vec=[0.0, 0.0, 1.0]),
    ]
    index, chunks = build_index(chunks)

    orig_embed = P.embed_text
    orig_search = P.search
    try:
        # --- cosine pass: query embeds onto c0; lexical coverage is 0
        # (the query token does not exist in the tiny corpus).
        P.embed_text = lambda text, client: [1.0, 0.0, 0.0]
        res = P.retrieve("زقزق", index, chunks, None, top_k=3)
        check("cosine gate passes on true self-match",
              res and res[0].get("_gate_passed") is True)
        c0 = next(c for c in res if c["id"] == "c0")
        check("real cosine score attached (no synthetic constant)",
              c0["score"] > 0.99 and c0.get("_sem_score", 0) > 0.99)

        # --- gate FAIL: low cosine (all components < 0.3 after norm) and no
        # lexical coverage for the gibberish token.
        P.embed_text = lambda text, client: [-0.9, 0.1, 0.2]
        res = P.retrieve("زقزق", index, chunks, None, top_k=3)
        check("gate fails with low cosine and no lexical coverage",
              bool(res) and all(c.get("_gate_passed") is False for c in res))

        # --- lexical pass: token exists in corpus, embedding irrelevant.
        P.embed_text = lambda text, client: [-0.9, 0.1, 0.2]
        res = P.retrieve("محتوى", index, chunks, None, top_k=3)
        check("lexical-coverage gate passes despite low cosine",
              bool(res) and res[0].get("_gate_passed") is True)

        # --- cosine-threshold calibration (Phase 3I): the gate must refuse
        # controlled borderline cosines with zero lexical coverage.  0.3343 is
        # the measured worst unsupported question, 0.4274 the measured
        # weakest supported question on the current corpus (genuine vectors).
        def _fake_search(emb, index, chunks, top_k, score):
            return [{"id": "c0", "source": "fiqh", "reference": "",
                     "original_text": "x", "score": score}]

        try:
            for score, expect in ((0.3343, False), (0.35, False),
                                  (0.4274, True), (0.45, True)):
                P.search = lambda e, i, c, top_k, s=score: _fake_search(e, i, c, top_k, s)
                res = P.retrieve("زقزق", index, chunks, None, top_k=1)
                check(
                    f"gate {'passes' if expect else 'refuses'} at cosine {score} "
                    f"with no lexical coverage",
                    bool(res) and all(c.get("_gate_passed") is expect for c in res),
                )
        finally:
            P.search = orig_search
    finally:
        P.embed_text = orig_embed
        P.search = orig_search


# ---------------------------------------------------------------------------
# 4. Terminology expansion (synonym groups + scoped phrase alias)
# ---------------------------------------------------------------------------

def test_terminology() -> None:
    print("\n[4] Terminology expansion")
    exp = P._expand_synonyms(["سنن", "مؤكده", "صلاه"])
    check("phrase alias: سنن مؤكدة -> سنن راتبة",
          ["سنن", "راتبه", "صلاه"] in exp)
    check("phrase alias: plural variant",
          ["سنن", "رواتب", "صلاه"] in exp)

    exp = P._expand_synonyms(["مؤكده"])
    check("NO global مؤكده -> راتبه mapping",
          not exp or all("راتبه" not in e and "رواتب" not in e for e in exp))

    exp = P._expand_synonyms(["غزوه"])
    check("غزوة expands to وقعة/معركة",
          ["وقعه"] in exp and ["معركه"] in exp)

    exp = P._expand_synonyms(["رواتب"])
    check("رواتب expands to راتبة", ["راتبه"] in exp)


# ---------------------------------------------------------------------------
# 5. Grounding
# ---------------------------------------------------------------------------

def _corpus() -> list[dict]:
    # Synthetic corpus whose BM25 doc frequencies drive the rare-token rule.
    corpus = [
        _chunk("k_badr", "sira", "فكانت وقعه بدر يوم الجمعه صبيحه سبع عشره من شهر رمضان"),
        _chunk("k_khandaq", "sira", "غزوه الخندق في شوال في السنه الخامسه"),
        _chunk("k_rawatib", "fiqh", "ويسمي السنن الراتبه ويشمل سنه الفجر والظهر والعصر والمغرب والعشاء"),
        _chunk("k_jannah", "hadith", "للجنه ثمانيه ابواب"),
        _chunk("k_two", "fiqh", "الوتر سنه مؤكده حث عليه الرسول ورغب فيه"),
    ]
    # Pad with filler docs so common tokens get higher df (rare tokens stay
    # low).  RARE_DF=150: with a tiny corpus every token is "rare" unless it
    # appears in several docs, which is exactly the synthetic behaviour we
    # want: only tokens KNOWN to be common-ish survive unmatched.
    for i in range(5):
        corpus.append(_chunk(f"f{i}", "fiqh", "بدر رمضان شهر الجمعه يوم صلاه فقه حديث سنه"))
    corpus.append(_chunk("k_quran", "quran", "الذين ياكلون امول يتمى ظلما انما ياكلون في بطونهم نارا وسيصلون سعيرا"))
    return corpus


def test_grounding() -> None:
    print("\n[5] Grounding")
    corpus = _corpus()
    by_id = {c["id"]: c for c in corpus}

    out = P._ground_answer(
        "كانت غزوة بدر يوم الجمعة في السابع عشر من شهر رمضان (مصدر 1).",
        [by_id["k_badr"]], corpus,
    )
    check("correct Badr paraphrase kept", "بدر" in out)

    out = P._ground_answer(
        "وقعت غزوة بدر يوم الجمعة في شهر شوال (مصدر 1).",
        [by_id["k_badr"]], corpus,
    )
    check("wrong month rejected", out == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.")

    out = P._ground_answer(
        "السنن المؤكدة تشمل سنة الفجر والظهر والعصر والمغرب والعشاء (مصدر 1).",
        [by_id["k_rawatib"]], corpus,
    )
    check("scoped phrase alias keeps rawatib paraphrase", "الفجر" in out)

    out = P._ground_answer(
        "الوتر سنة مؤكدة حث عليه الرسول (مصدر 1).",
        [by_id["k_two"]], corpus,
    )
    check("bare مؤكدة still matches its own topic", "الوتر" in out)

    out = P._ground_answer(
        "غزوة الخندق كانت يوم الجمعة في رمضان (مصدر 1).",
        [by_id["k_badr"]], corpus,
    )
    check("subject swap (الخندق) rejected despite shared words",
          out == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.")

    out = P._ground_answer(
        "الذين يأكلون أموال اليتامى ظلما يدخلون الجنة (مصدر 1).",
        [by_id["k_quran"]], corpus,
    )
    check("relation inversion (يدخلون الجنة) rejected", "يدخلون" not in out and "الجنة" not in out)

    out = P._ground_answer(
        "عدد أبواب الجنة سبعون بابا (مصدر 1).",
        [by_id["k_jannah"]], corpus,
    )
    check("wrong number rejected", "سبعون" not in out)

    # Uncited sentence: must be supported by ONE chunk, never a union.
    out = P._ground_answer(
        "الوتر سنة مؤكدة وغزوة بدر كانت في رمضان.",
        [by_id["k_two"], by_id["k_badr"]], corpus,
    )
    check("uncited sentence with split support rejected",
          out == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.")

    out = P._ground_answer(
        "الوتر سنة مؤكدة حث عليها الرسول.",
        [by_id["k_two"], by_id["k_jannah"]], corpus,
    )
    check("uncited sentence supported by one chunk accepted", "الوتر" in out)


# ---------------------------------------------------------------------------
# 6. Matcher / number canonicalisation regressions
# ---------------------------------------------------------------------------

def test_matcher() -> None:
    print("\n[6] Matcher + numbers")
    check("وقعت == وقعه", P._match_key("وقعت") == P._match_key("وقعه"))
    check("ووقعت == وقعت (iterative conjunction strip)",
          P._match_key("ووقعت") == P._match_key("وقعت"))
    check("نارا == نرا (dagger-alef rasm)",
          P._match_key("نارا") == P._match_key("نرا"))
    check("يتامى == يتمى (dagger-alef rasm)",
          P._match_key("يتامى") == P._match_key("يتمى"))
    check("synonym group keys canonicalise (وقعه==غزوه)",
          P._gkey("وقعه") == P._gkey("غزوه"))
    check("اجتنبوا == اجتنب (suffix before rasm)",
          P._match_key("اجتنبوا") == P._match_key("اجتنب") == "اجتنب")
    check("بطونهم -> بطون (suffix before rasm)",
          P._match_key("بطونهم") == "بطون")
    check("قالوا == قال (plural وا)",
          P._match_key("قالوا") == P._match_key("قال"))
    check("يدخلون == يدخل (plural ون)",
          P._match_key("يدخلون") == P._match_key("يدخل"))
    check("يعبدون == يعبد (prefix + plural ون)",
          P._match_key("يعبدون") == P._match_key("يعبد"))

    check("spelled 17", P._numbers_normalized(["سبع", "عشره"]) == ["17"])
    check("compound 27", P._numbers_normalized(["بسبع", "وعشرين"]) == ["27"])
    check("compound 25", P._numbers_normalized(["خمسه", "وعشرون"]) == ["25"])
    check("Arabic-Indic digits", P._numbers_normalized(["١٧"]) == ["17"])
    check("single tens", P._numbers_normalized(["ثمانون"]) == ["80"])


# ---------------------------------------------------------------------------
# 7. Embedding interface (offline validation of the API boundary)
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink.update(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25] * DIM)]
        )


class _FakeEmbedClient:
    def __init__(self):
        self.sink: dict = {}
        self.embeddings = _FakeEmbeddings(self.sink)


def test_embedding_interface() -> None:
    print("\n[7] Embedding interface")
    check("model constant is text-embedding-3-small",
          EMBEDDING_MODEL == "text-embedding-3-small")
    check("dimension constant is 1536 (matches index)", DIM == 1536)

    client = _FakeEmbedClient()
    vec = embed_text("الدين النصيحة", client)
    check("embed_text passes model id", client.sink.get("model") == EMBEDDING_MODEL)
    check("embed_text passes input text through", client.sink.get("input") == "الدين النصيحة")
    check("embed_text returns the response vector (len 1536)",
          isinstance(vec, list) and len(vec) == DIM)

    short = "الدين النصيحة"
    check("truncate_to_tokens leaves short text intact",
          truncate_to_tokens(short) == short)
    long_text = ("كلمة " * (MAX_TOKENS + 500)).strip()
    trunc = truncate_to_tokens(long_text)
    from src.embeddings.embedder import _encoding  # noqa: PLC0415
    check("truncate_to_tokens respects the token budget",
          len(_encoding.encode(trunc)) <= MAX_TOKENS)

    # Query normalization assumption: search() L2-normalizes the query, so
    # unnormalized vectors still yield cosine scores against the index.
    chunks = [
        _chunk("n0", "fiqh", "نص", vec=[1.0, 0.0, 0.0]),
        _chunk("n1", "fiqh", "نص", vec=[0.0, 1.0, 0.0]),
    ]
    index, chunks = build_index(chunks)
    res = P.search([5.0, 0.0, 0.0], index, chunks, top_k=2)
    check("unnormalized query normalized inside search (cos=1.0)",
          abs(res[0]["score"] - 1.0) < 1e-5)
    res = P.search([1.0, 1.0, 0.0], index, chunks, top_k=2)
    check("diagonal query yields cos=1/sqrt(2) against both",
          all(abs(r["score"] - 0.70710678) < 1e-5 for r in res))

    # Error handling: an encoder failure aborts retrieval (fail-fast, no
    # silent fallback to lexical-only), documented behavior.
    orig = P.embed_text

    def _boom(text, client):
        raise RuntimeError("simulated embeddings API failure")

    P.embed_text = _boom
    try:
        try:
            P.retrieve("زقزق", index, chunks, None, top_k=2)
            raised = False
        except RuntimeError:
            raised = True
        check("encoder failure propagates out of retrieve (fail-fast)", raised)
    finally:
        P.embed_text = orig


# ---------------------------------------------------------------------------
# 8. Answer-generation branches (deterministic, no LLM; NOT validating the
#    LLM's real behavior — only the surrounding pipeline logic)
# ---------------------------------------------------------------------------

REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."


class _FakeChat:
    def __init__(self, sink, answer, explode=False):
        self._sink = sink
        self._answer = answer
        self._explode = explode
        self.called = False
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.called = True
        if self._explode:
            raise AssertionError("LLM must not be called on the refusal path")
        self._sink.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._answer)
            )]
        )


def _gen_corpus() -> tuple:
    chunks = [
        _chunk("g0", "fiqh", "محتوى الفقه في الطهارة", vec=[1.0, 0.0, 0.0]),
        _chunk("g1", "fiqh", "محتوى الفقه في الصلاة", vec=[0.0, 1.0, 0.0]),
        _chunk("g2", "hadith", "محتوى الحديث في الطهارة", vec=[0.0, 0.0, 1.0]),
    ]
    index, chunks = build_index(chunks)
    return index, chunks


def test_generation_branches() -> None:
    print("\n[8] Generation branches (deterministic, no LLM)")
    index, chunks = _gen_corpus()
    orig_embed = P.embed_text
    orig_rewrite = P._rewrite_query
    try:
        # Query rewriting is an API stage of its own: isolate it so these
        # tests measure the ANSWERING stage only (the harness does the same).
        P._rewrite_query = lambda query, client: []

        # --- gate failure => refusal WITHOUT calling generation ----------
        P.embed_text = lambda text, client: [-0.9, 0.1, 0.2]
        chat = _FakeChat({}, "", explode=True)
        out = P.generate("زقزق", index, chunks, chat)
        check("gate failure returns the canonical refusal", out["answer"] == REFUSAL)
        check("answering LLM call is skipped on the refusal path",
              chat.called is False)

        # --- gate pass => prompt construction + model + temperature ------
        P.embed_text = lambda text, client: [1.0, 0.0, 0.0]
        sink: dict = {}
        chat = _FakeChat(sink, "محتوى الفقه في الطهارة (مصدر 1).")
        out = P.generate("زقزق", index, chunks, chat)
        check("LLM called exactly once", chat.called is True)
        check("generation model passed", sink.get("model") == P.GENERATION_MODEL)
        check("temperature=0", sink.get("temperature") == 0)
        msgs = sink.get("messages") or []
        check("system message first, user message last",
              msgs and msgs[0]["role"] == "system" and msgs[-1]["role"] == "user")
        user_msg = msgs[-1]["content"] if msgs else ""
        check("prompt contains the question", "السؤال: زقزق" in user_msg)
        check("prompt contains retrieved source text",
              "محتوى الفقه في الطهارة" in user_msg)
        check("prompt cites sources in [مصدر N] form", "[مصدر 1]" in user_msg)
        check("prompt carries the refusal instruction",
              any("لم أجد في المصادر المتاحة" in (m.get("content") or "")
                  for m in msgs))
        check("grounded answer survives unchanged",
              "الطهارة" in out["answer"])
        check("sources list uses 'source — reference'",
              out["sources"][0] == "fiqh — مرجع")

        # --- unsupported claim => post-grounding refusal ------------------
        chat = _FakeChat({}, "غزوة بدر كانت في شهر شوال (مصدر 1).")
        out = P.generate("زقزق", index, chunks, chat)
        check("unsupported generated claim becomes refusal",
              out["answer"] == REFUSAL)

        # --- model-level refusal passes through untouched -----------------
        passthrough = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
        chat = _FakeChat({}, passthrough)
        out = P.generate("زقزق", index, chunks, chat)
        check("model refusal passes through grounding", out["answer"] == passthrough)

        # --- rewrite stage degrades to [] when the client fails -----------
        P._rewrite_query = orig_rewrite
        exploding = _FakeChat({}, "", explode=True)
        check("_rewrite_query degrades to [] on client failure",
              P._rewrite_query("سؤال", exploding) == [])
    finally:
        P.embed_text = orig_embed
        P._rewrite_query = orig_rewrite


# ---------------------------------------------------------------------------
# 9. Multi-evidence selection regressions (Task C)
# ---------------------------------------------------------------------------

def test_multi_evidence_selection() -> None:
    print("\n[9] Multi-evidence selection")
    orig = P.embed_text
    try:
        # same-source-type pair (cap stress), mixed query vector
        chunks = [
            _chunk("e1", "fiqh", "دليل الغسل يوم الجمعة", vec=[1.0, 0.0, 0.0, 0.0]),
            _chunk("e2", "fiqh", "وقت الغسل يوم الجمعة", vec=[0.0, 1.0, 0.0, 0.0]),
            _chunk("n1", "fiqh", "نص عادي ا", vec=[0.0, 0.0, 1.0, 0.0]),
            _chunk("n2", "fiqh", "نص عادي ب", vec=[0.0, 0.0, 0.0, 1.0]),
            _chunk("n3", "hadith", "نص عادي ج", vec=[0.3, 0.3, 0.9, 0.0]),
            _chunk("n4", "hadith", "نص عادي د", vec=[0.3, 0.3, 0.0, 0.9]),
        ]
        index, chunks = build_index(chunks)
        mixed = [0.70710678, 0.70710678, 0.0, 0.0]
        P.embed_text = lambda text, client: mixed
        sel = P.retrieve("زقزق", index, chunks, None, top_k=5)
        sel_ids = {c["id"] for c in sel}
        check("same-type evidence pair reaches final context together",
              {"e1", "e2"} <= sel_ids)

        chunks = [
            _chunk("x1", "fiqh", "الدليل الفقهي", vec=[1.0, 0.0, 0.0, 0.0]),
            _chunk("x2", "hadith", "الدليل الحديثي", vec=[0.0, 1.0, 0.0, 0.0]),
            _chunk("y1", "fiqh", "نص عادي ه", vec=[0.0, 0.0, 1.0, 0.0]),
            _chunk("y2", "hadith", "نص عادي و", vec=[0.0, 0.0, 0.0, 1.0]),
            _chunk("y3", "quran", "نص عادي ز", vec=[0.3, 0.3, 0.9, 0.0]),
            _chunk("y4", "tafsir", "نص عادي ح", vec=[0.3, 0.3, 0.0, 0.9]),
        ]
        index, chunks = build_index(chunks)
        P.embed_text = lambda text, client: mixed
        sel = P.retrieve("زقزق", index, chunks, None, top_k=5)
        sel_ids = {c["id"] for c in sel}
        check("cross-type evidence pair reaches final context together",
              {"x1", "x2"} <= sel_ids)
    finally:
        P.embed_text = orig

    # guarantee-overflow behavior (Phase 3I policy): when more chunks carry a
    # survival guarantee (lex/sem top-3) than top_k permits and they all
    # belong to ONE source type, same-type guarantees are capped at
    # _DIVERSITY_MAX_PER_TYPE; a complementary non-guaranteed chunk of
    # another source type takes a slot, and the weakest same-type
    # guarantees are dropped (score order preserved among equal guarantees).
    pool = [_mk(f"G{i}", "fiqh", float(7 - i)) for i in range(1, 7)]
    pool.append(_mk("N1", "hadith", 99.0))
    sel = P._select_diverse(
        pool, 5,
        lex_top3_ids={"G1", "G2", "G3"},
        sem_top3_ids={"G4", "G5", "G6"},
    )
    ids = {c["id"] for c in sel}
    check("same-type guarantee overflow: cap kept, complementary type admitted",
          ids == {"G1", "G2", "G3", "G4", "N1"})
    check("overflow drops the weakest same-type guarantees (G5, G6)",
          "G5" not in ids and "G6" not in ids)


# ---------------------------------------------------------------------------
# 10. Cached query-embedding file interface (Task B)
# ---------------------------------------------------------------------------

def test_query_embedding_file() -> None:
    print("\n[10] Cached query-embedding file")
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from src.rag.offline_eval import _load_query_embeddings, _mean_vector  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="basira_offline_tests_") as tmpdir:
        path = os.path.join(tmpdir, "offline_tests_query_embeddings.json")
        good = {
            "q1": {"clean": [0.1, 0.2, 0.3, 0.4], "model": "test"},
            "q2": {"clean": [1.0, 0.0, 0.0, 0.0], "raw": [0.0, 1.0, 0.0, 0.0]},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(good, fh)
        loaded = _load_query_embeddings(path, dim=4)
        check("loader accepts valid file", set(loaded) == {"q1", "q2"})
        check("raw falls back to clean", loaded["q1"]["raw"] == loaded["q1"]["clean"])
        check("explicit raw preserved", loaded["q2"]["raw"] == [0.0, 1.0, 0.0, 0.0])

        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"bad": {"clean": [1.0, 2.0]}}, fh)
        try:
            _load_query_embeddings(path, dim=4)
            dim_err = False
        except ValueError as exc:
            dim_err = "bad" in str(exc) and "4 numbers" in str(exc)
        check("wrong dimension rejected with the question id named", dim_err)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"bad": {"raw": [1.0, 2.0, 3.0, 4.0]}}, fh)
        try:
            _load_query_embeddings(path, dim=4)
            missing_err = False
        except ValueError as exc:
            missing_err = "clean" in str(exc)
        check("missing clean vector rejected", missing_err)

        mean = _mean_vector([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        check("mean vector is L2-normalized",
              abs(mean[0] - 0.70710678) < 1e-6 and abs(mean[1] - 0.70710678) < 1e-6)
        check("zero-vector guard", _mean_vector([[0.0, 0.0], [0.0, 0.0]]) == [0.0, 0.0])


# ---------------------------------------------------------------------------
# 11. Grounding negation-flip check (Task E)
# ---------------------------------------------------------------------------

def test_negation_guard() -> None:
    print("\n[11] Grounding negation-flip check")
    sng = [{"id": "s1", "original_text": (
        "ولا يعيد الغسل ويخرج وقت الغسل بالفراغ الصلاه "
        "فمن اغتسل بعد الصلاه يكون غسلا للجمعه"
    )}]

    check("_is_negation_token accepts bare particles",
          all(P._is_negation_token(t) for t in ("لا", "لم", "لن", "ليس", "ليست")))
    check("_is_negation_token strips و/ف conjunctions",
          all(P._is_negation_token(t) for t in ("ولا", "فلا", "ولم", "فلم", "وليس")))
    check("_is_negation_token rejects ambiguous cues",
          not any(P._is_negation_token(t) for t in ("لكن", "ما", "غير", "إلا", "لماذا")))

    check("_strict_key does not collide يعيد/عيدين",
          P._strict_key("يعيد") != P._strict_key("عيدين"))
    check("_strict_key strips definite article", P._strict_key("الغسل") == "غسل")
    check("_strict_key folds conjunctions", P._strict_key("ولا") == "لا")

    flip = P._ground_answer("من اغتسل بعد الصلاة يعيد الغسل (مصدر 1).", sng)
    check("affirmative flip claim is dropped",
          flip.strip() == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.")
    mirror = P._ground_answer("من اغتسل بعد الصلاة لا يعيد الغسل (مصدر 1).", sng)
    check("same-polarity negated claim is kept",
          "يعيد الغسل" in mirror and "لا" in mirror)
    clean = P._ground_answer(
        "من اغتسل بعد الصلاة يخرج وقت الغسل بالفراغ (مصدر 1).", sng)
    check("claim unaffected by the guard is kept", "يخرج وقت الغسل" in clean)

    sng2 = [{"id": "s2", "original_text": "الصلاة ليست واجبة على المسافر"}]
    lem = P._ground_answer("الصلاة واجبة على المسافر (مصدر 1).", sng2)
    check("ليست negates the following token like ليس",
          lem.strip() == "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال.")


# ---------------------------------------------------------------------------
# 12. Unicode format-character (Cf) hygiene (Phase 3K.1, H1)
# ---------------------------------------------------------------------------

def test_cf_format_chars() -> None:
    print("\n[12] Unicode format-character (Cf) hygiene")
    # Phase 3J orphan_wealth: retrieved source chunks carry U+200F
    # (RIGHT-TO-LEFT MARK) glued to tokens (e.g. 'ظلما\u200f') while the
    # model-side claim token never does; the fold must drop Cf characters.
    check("_fold_ar strips trailing U+200F",
          P._fold_ar("ظلما\u200f") == P._fold_ar("ظلما"))
    check("_fold_ar strips leading U+200F",
          P._fold_ar("\u200fان") == P._fold_ar("ان"))
    check("_match_key tolerates U+200F glue",
          P._match_key("ظلما\u200f") == P._match_key("ظلما"))
    check("_gkey tolerates U+200F glue",
          P._gkey("ظلما\u200f") == P._gkey("ظلما"))

    # Deterministic replay of the recorded orphan_wealth matcher failure:
    # the recorded claim sentence must ground against a source whose token
    # is U+200F-glued.  Without Cf folding the claim token 'ظلما' is
    # unmatched, falls back to df=0 and trips the rare-token rule (DROP);
    # with the fix it matches and the sentence is kept.
    src = _chunk(
        "k_cf", "quran",
        "إِنَّ الَّذِينَ يَأْكُلُونَ أَمْوَالَ الْيَتَامَى ظُلْمًا\u200f "
        "إِنَّمَا يَأْكُلُونَ فِي بُطُونِهِمْ نَارًا",
    )
    claim = ("إن الذين يأكلون أموال اليتامى ظلمًا إنما يأكلون في "
             "بطونهم نارًا (مصدر 1).")
    out = P._ground_answer(claim, [src], [src])
    check("U+200F-glued source token still grounds the claim",
          out.strip() != REFUSAL and "ظلمًا" in out)


# ---------------------------------------------------------------------------
# 13. Variant generator: suffix stripping before rasm (asymmetry fix)
# ---------------------------------------------------------------------------

def test_variant_suffix_order() -> None:
    print("\n[13] Variant generator: suffix/rasm ordering")
    # Suffixes like وا/ات carry an alef themselves.  With rasm removal running
    # first (اجتنبوا -> اجتنبو) the suffix no longer matched and the plural
    # stem never reached the singular (اجتنب), which made the rare-token rule
    # reject supported paraphrases.  Suffixes must be stripped first.
    check("اجتنبوا -> اجتنب", P._match_key("اجتنبوا") == "اجتنب")
    check("بطونهم -> بطون", P._match_key("بطونهم") == "بطون")
    check("قالوا key == قال key", P._match_key("قالوا") == P._match_key("قال"))
    check("يدخلون key == يدخل key",
          P._match_key("يدخلون") == P._match_key("يدخل"))
    check("يتامى rasm fold intact", P._match_key("يتامى") == P._match_key("يتمى"))
    check("نارا rasm fold intact", P._match_key("نارا") == P._match_key("نرا"))
    check("وقعت == وقعه still holds",
          P._match_key("وقعت") == P._match_key("وقعه"))

    # Grounding replay of the recorded artifact-nasai failure: the claim uses
    # the plural اجتنبوا while the cited chunk has the singular اجتنب.
    # Before the fix the unmatched token fell to df=20 and tripped the
    # rare-token rule (DROP); now the keys coincide and the sentence is kept.
    src = _chunk(
        "k_sfx", "hadith",
        "عن ابن عباس قال لما نزلت هذه الآية ولا تقربوا مال اليتيم إلا "
        "بالتي هي أحسن قال اجتنب الناس مال اليتيم وطعامه",
    )
    claim = "قال ابن عباس إن الناس اجتنبوا مال اليتيم (مصدر 1)."
    out = P._ground_answer(claim, [src], [src])
    check("اجتنبوا claim grounds against اجتنب chunk",
          out.strip() != REFUSAL and "اجتنبوا" in out)


# ---------------------------------------------------------------------------
# 14. Grounding P5v5 policy: clitic df, family exemption, polarity parity,
#     quote verification, entity guard, add-guard (Phase 3K.11)
# ---------------------------------------------------------------------------

def test_p5v5_grounding_policy() -> None:
    print("\n[14] Grounding P5v5 policy")

    check("clitic_v2 links مثقال -> ثقال", "ثقال" in P._clitic_v2("مثقال"))
    check("family_v2 links مستحب -> استحب", "استحب" in P._family_v2("مستحب"))
    check("_seed_aligned aligns مستحب/استحب",
          P._seed_aligned("مستحب", "استحب"))
    check("_seed_aligned rejects مسعود/عباس",
          not P._seed_aligned("مسعود", "عباس"))
    check("_is_inversion_token accepts غير/وغير/بدون",
          all(P._is_inversion_token(t) for t in ("غير", "وغير", "بدون")))
    check("_is_inversion_token rejects لغير",
          not P._is_inversion_token("لغير"))
    ok, _ = P._span_align(["قال", "رسول", "الله"],
                          ["ثم", "قال", "رسول", "الله"], "exact", 6)
    check("_span_align finds in-order window", ok)
    ok, _ = P._span_align(["قال", "رسول", "الله"],
                          ["الله", "رسول", "قال"], "exact", 6)
    check("_span_align rejects reversed order", not ok)

    # Clitic union df metric and the material-addition guard.  مثقال is
    # unmatched in the cited chunk; its clitic variant ثقال fills the corpus.
    src_r = _chunk("p5_r", "fiqh", "قال اهل العلم زيد في رمضان")
    claim_r = "قال أهل العلم زيد مثقال في رمضان (مصدر 1)."
    src_g = _chunk("p5_g", "fiqh", "قال اهل العلم في رمضان")
    corpus_rescue = [src_r] + [_chunk(f"p5_rc{i}", "fiqh", "زيد ثقال")
                               for i in range(700)]
    postings = P._build_bm25(corpus_rescue)["postings"]
    check("union df counts variant documents (700)",
          P._union_docs_count(P._clitic_v2("مثقال"), postings) == 700)
    out = P._ground_answer(claim_r, [src_r], corpus_rescue)
    check("clitic union df beyond band tolerated (rescue)", "مثقال" in out)
    corpus_band = ([src_r]
                   + [_chunk(f"p5_bc{i}", "fiqh", "زيد ثقال")
                      for i in range(350)]
                   + [_chunk(f"p5_bz{i}", "fiqh", "زيد")
                      for i in range(350)])
    out = P._ground_answer(claim_r, [src_r], corpus_band)
    check("band token tolerated in well-covered sentence", "مثقال" in out)
    out = P._ground_answer(claim_r, [src_g], corpus_band)
    check("band token flagged in weakly-covered sentence",
          out.strip() == REFUSAL)

    # Polarity: dual inversion, inverted-vs-affirmative conflicts (both
    # directions), mixed occurrence sets, family exemption under compatible
    # inversion only, and the بدون control.
    pol_fixtures = [
        ("dual inversion kept (غير both sides)",
         "قال أهل العلم إن الإفراط في الصيام غير مستحب (مصدر 1).",
         "قال أهل العلم إن الإفراط في الصيام غير مستحب", True),
        ("arafah dual inversion with لغير kept",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج", True),
        ("inverted claim vs affirmative source rejected",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة مستحب لغير الحاج", False),
        ("affirmative claim vs inverted source rejected",
         "قال أهل العلم إن صيام يوم عرفة مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج", False),
        ("mixed source occurrence polarities rejected",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج وصيام يوم "
         "عرفة مستحب لغير الحاج", False),
        ("family variant under compatible inversion kept",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة غير استحب لغير الحاج", True),
        ("family variant without inversion rejected",
         "قال أهل العلم إن صيام يوم عرفة غير مستحب لغير الحاج (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة استحب لغير الحاج", False),
        ("بدون same polarity kept",
         "قال أهل العلم إن صيام يوم عرفة مستحب بدون عذر (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة مستحب بدون عذر", True),
        ("بدون conflict rejected",
         "قال أهل العلم إن صيام يوم عرفة مستحب بدون عذر (مصدر 1).",
         "قال أهل العلم إن صيام يوم عرفة مستحب مع العذر", False),
    ]
    for name, claim, src_text, kept in pol_fixtures:
        src_c = _chunk("p5_pol", "hadith", src_text)
        out = P._ground_answer(claim, [src_c], [src_c])
        check(name, (out.strip() != REFUSAL) == kept)

    # Quote verification: verbatim and truncated-span quotes pass; middle
    # substitution (even with a corpus-common substitute), reordering and
    # cross-source stitching are rejected.
    src_q = _chunk("p5_q", "hadith",
                   "قال رسول الله إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
    out = P._ground_answer(
        'قال رسول الله "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى" '
        "(مصدر 1).", [src_q], [src_q])
    check("verbatim quote kept", "بالنيات" in out)
    out = P._ground_answer(
        'قال رسول الله "إنما الأعمال بالنيات وإنما لكل امرئ" (مصدر 1).',
        [src_q], [src_q])
    check("quote truncation on a source word kept", "بالنيات" in out)
    corpus_sub = [src_q] + [_chunk(f"p5_qc{i}", "fiqh", "بالقصد فقه")
                            for i in range(350)]
    out = P._ground_answer(
        'قال رسول الله "إنما الأعمال بالقصد وإنما لكل امرئ ما نوى" '
        "(مصدر 1).", [src_q], corpus_sub)
    check("quote middle substitution rejected despite common substitute",
          out.strip() == REFUSAL)
    out = P._ground_answer(
        'قال رسول الله "وإنما لكل امرئ ما نوى إنما الأعمال بالنيات" '
        "(مصدر 1).", [src_q], [src_q])
    check("reordered quote rejected", out.strip() == REFUSAL)
    src_q1 = _chunk("p5_q1", "hadith", "قال رسول الله إنما الأعمال بالنيات")
    src_q2 = _chunk("p5_q2", "hadith", "وإنما لكل امرئ ما نوى عند الله")
    out = P._ground_answer(
        'قال: "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى" (مصدر 1) '
        "(مصدر 2).", [src_q1, src_q2], [src_q1, src_q2])
    check("cross-source stitched quote rejected", out.strip() == REFUSAL)

    # Entity guard: a matched anchor follower passes; a swapped follower
    # whose df is above every rare threshold is still rejected.
    src_e = _chunk("p5_e", "hadith",
                   "قال ابن عباس إن صيام يوم عرفة مستحب لغير الحاج")
    corpus_ent = [src_e] + [_chunk(f"p5_ei{i}", "fiqh", "مسعود فقه")
                            for i in range(350)]
    out = P._ground_answer(
        "قال ابن عباس إن صيام يوم عرفة مستحب (مصدر 1).", [src_e],
        corpus_ent)
    check("matched entity follower kept", "عباس" in out)
    out = P._ground_answer(
        "قال ابن مسعود إن صيام يوم عرفة مستحب (مصدر 1).", [src_e],
        corpus_ent)
    check("entity-anchor substitution rejected (despite common df)",
          out.strip() == REFUSAL)

    # Right-only negation marking: a token sitting directly BEFORE a
    # negation particle is no longer flip-marked (the old two-sided window
    # artifact), while the negated token itself still flips.
    src_n = _chunk("p5_n", "hadith",
                   "الآية ولا تقربوا مال اليتيم إلا بالتي هي أحسن")
    out = P._ground_answer("الآية ولا تقربوا مال اليتيم (مصدر 1).",
                           [src_n], [src_n])
    check("token before negation is not flip-marked (right-only)",
          "الآية" in out)

    # neg-window-fp residual: the frozen-set sentence whose نزول is only
    # supported as نزلت in the cited chunk.  On this micro-corpus the
    # sentence still drops because كان/يرى/بعد are df-0 rare there, but the
    # Phase E.4 curated alias must remove نزول itself from the blocker list
    # (نزلت is present in the cited source).  Pinned so any alias or
    # threshold drift shows up here.
    src_nw = _chunk(
        "p5_nw", "hadith",
        "عن ابن عباس قال لما نزلت هذه الآية ولا تقربوا مال اليتيم إلا "
        "بالتي هي أحسن قال اجتنب الناس مال اليتيم وطعامه")
    corpus_nw = [src_nw] + [_chunk(f"p5_nwi{i}", "fiqh", "نزول فقه")
                            for i in range(238)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = P._ground_answer(
            "كان ابن عباس يرى اجتناب مال اليتيم بعد نزول الآية (مصدر 1).",
            [src_nw], corpus_nw)
    log_nw = next((ln for ln in buf.getvalue().splitlines()
                   if ln.startswith("[ground]")), "")
    check("neg-window-fp micro-corpus still dropped (other rare tokens)",
          out.strip() == REFUSAL)
    check("curated alias removes نزول from the blockers only",
          "نزول" not in log_nw
          and all(tok in log_nw for tok in ("كان", "يرى", "بعد")))

    # Multi-sentence independence: one dropped sentence never suppresses a
    # supported neighbour in the same answer.
    corpus = _corpus()
    by_id = {c["id"]: c for c in corpus}
    out = P._ground_answer(
        "غزوة الخندق كانت يوم الجمعة في رمضان (مصدر 1). "
        "الوتر سنة مؤكدة حث عليها الرسول (مصدر 2).",
        [by_id["k_badr"], by_id["k_two"]], corpus)
    check("unsupported sentence dropped, supported neighbour kept",
          "الوتر" in out and "الخندق" not in out)


# ---------------------------------------------------------------------------

def test_grounding_curated_alias() -> None:
    print("\n[15] Grounding curated alias (نزول -> نزلت)")

    # The single audited pair: an exact folded table, no stem/prefix/suffix
    # expansion, and no reverse direction (Phase E.4; evidence in the
    # Phase E.1 false-rejection analysis and Phase E.3 audit reports).
    check("curated alias table holds exactly the audited pair",
          P._GROUNDING_CURATED_ALIASES == {"نزول": ("نزلت",)})
    check("alias hits select only the exact folded target",
          P._curated_alias_surfaces("نزول", {"نزلت", "نزل", "الرحمة"})
          == {"نزلت"})
    check("alias is not reverse-matching",
          P._curated_alias_surfaces("نزلت", {"نزول"}) == set())
    check("other hollow pairs stay unaliased",
          P._curated_alias_surfaces("دخول", {"دخلت"}) == set())

    # Positive target: نزول is the only unmatched content token and the
    # cited chunk contains نزلت, so the alias must remove it from the
    # rare-token blockers and keep the sentence with every guard ok.
    src_pos = _chunk("p5_alias", "hadith",
                     "لما نزلت آية الرحمة اجتنب الناس مال اليتيم")
    claim_pos = "نزول آية الرحمة اجتنب الناس مال اليتيم (مصدر 1)."
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = P._ground_answer(claim_pos, [src_pos], [src_pos])
    log_pos = next((ln for ln in buf.getvalue().splitlines()
                    if ln.startswith("[ground]")), "")
    check("curated alias flips the sole-نزول fixture to KEEP",
          out.strip() != REFUSAL)
    check("alias exemption leaves every other guard ok",
          "digits=ok months=ok neg=ok quote=ok entity=ok" in log_pos
          and "rare=" not in log_pos)

    # Exact-match boundaries: only the exact source surface نزلت activates;
    # bare نزل, the x+ة form نزلة, the reverse direction and any other
    # hollow-looking pair must all stay blocked.
    src_bare = _chunk("p5_alias_bare", "hadith",
                      "لما نزل آية الرحمة اجتنب الناس مال اليتيم")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = P._ground_answer(claim_pos, [src_bare], [src_bare])
    log_bare = next((ln for ln in buf.getvalue().splitlines()
                     if ln.startswith("[ground]")), "")
    check("bare نزل source does not activate the alias",
          out.strip() == REFUSAL and "rare=نزول" in log_bare)
    src_ta = _chunk("p5_alias_ta", "hadith",
                    "لما نزلة آية الرحمة اجتنب الناس مال اليتيم")
    out = P._ground_answer(claim_pos, [src_ta], [src_ta])
    check("x+ة suffix (نزلة) does not activate the alias",
          out.strip() == REFUSAL)
    src_rev = _chunk("p5_alias_rev", "hadith",
                     "لما نزول آية الرحمة اجتنب الناس مال اليتيم")
    out = P._ground_answer(
        "نزلت آية الرحمة اجتنب الناس مال اليتيم (مصدر 1).",
        [src_rev], [src_rev])
    check("reverse direction is not aliased",
          out.strip() == REFUSAL)
    src_dukhul = _chunk("p5_alias_dukhul", "hadith",
                        "لما دخلت آية الرحمة اجتنب الناس مال اليتيم")
    out = P._ground_answer(
        "دخول آية الرحمة اجتنب الناس مال اليتيم (مصدر 1).",
        [src_dukhul], [src_dukhul])
    check("other hollow pair (دخول/دخلت) stays unaliased",
          out.strip() == REFUSAL)

    # Adversarial falsifiers adapted from the Phase E.3 audit: the four
    # collision cases the generalized predicate false-accepted, plus the
    # negative controls for unsupported evidence, extra material, negation
    # polarity, quote verification, the entity guard and broader claims.
    falsifiers = [
        ("F2 unsupported hollow (نزول, no نزلت anywhere)",
         "نزول غزير للمطر (مصدر 1).",
         "المطر غزير جدا في الشتاء"),
        ("F2b bare-stem source (ينزل) not bridged",
         "نزول غزير للمطر (مصدر 1).",
         "المطر ينزل غزيرا في الشتاء"),
        ("F3a صوام/صامت (silent) not bridged",
         "الصوام صامت في المسجد (مصدر 1).",
         "كان صامت لا يتكلم في المسجد"),
        ("F3b راوي/رايت not bridged",
         "الراوي رايت في المسجد (مصدر 1).",
         "رايت المسجد في المنام"),
        ("F3c ثواب/ثابت not bridged",
         "الثواب ثابت في العمل (مصدر 1).",
         "ثابت العمل بالدليل في المسجد"),
        ("F3d تورك/تركت not bridged",
         "التورك تركت المسجد (مصدر 1).",
         "تركت المسجد في الشتاء"),
        ("F4 extra unmatched material still dropped",
         "نزول المطر والصوام (مصدر 1).",
         "لما نزلت الرحمة على المطر"),
        ("F5 negated claim without source negation dropped",
         "لا نزول للمطر في الشتاء (مصدر 1).",
         "نزلت الرحمة على المطر في الشتاء"),
        ("F5 quoted span alignment unaffected by alias",
         "قال «نزول المطر بعد الشتاء» في المسجد (مصدر 1).",
         "لما نزلت الرحمة على المطر في الشتاء"),
        ("F5 entity swap still rejected",
         "قال ابن يعود في المسجد الأقصى (مصدر 1).",
         "قال في المسجد الأقصى عن الشتاء"),
        ("F6 broader unsupported passage dropped",
         "نزول تحريم الخمر سنة سبع (مصدر 1).",
         "لما نزلت الآية في تحريم مال اليتيم"),
    ]
    fals_logs: dict[str, str] = {}
    for name, claim, src_text in falsifiers:
        c = _chunk("p5_alias_fal", "hadith", src_text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = P._ground_answer(claim, [c], [c])
        fals_logs[name] = next(
            (ln for ln in buf.getvalue().splitlines()
             if ln.startswith("[ground]")), "")
        check(name, out.strip() == REFUSAL)

    # The alias must not leak into the other guarded decisions: the quote
    # span and the entity follower still fail their own checks, and the
    # negation parity filter still blocks the alias hit.
    check("quoted-span failure is the quote guard, not a rare token",
          "quote=fail" in fals_logs[
              "F5 quoted span alignment unaffected by alias"])
    check("entity failure is the entity guard, not a rare token",
          "entity=fail" in fals_logs["F5 entity swap still rejected"])
    check("negated نزول stays blocked by the polarity filter",
          "rare=نزول" in fals_logs[
              "F5 negated claim without source negation dropped"])


# ---------------------------------------------------------------------------
# 16. Refusal-prefix grounding regression (RAG-01)
# ---------------------------------------------------------------------------

def test_refusal_prefix_grounding() -> None:
    print("\n[16] Refusal-prefix grounding regression (RAG-01)")
    chunk = _chunk("rf0", "fiqh",
                   "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
    retrieved = [dict(chunk, score=0.91, _gate_passed=True)]
    corpus = [chunk]

    orig_retrieve = P.retrieve
    P.retrieve = lambda query, index, chunks, client, top_k=5: [
        dict(retrieved[0])]
    try:
        hallucination = "الصلاة لا تجب على المسلم في السفر مطلقا."
        supported = "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى."
        supported_cited = (supported[:-1] + " (مصدر 1).")
        hallucination_cited = (hallucination[:-1] + " (مصدر 1).")

        def answer_for(text: str) -> str:
            chat = _FakeChat({}, text)
            return P.generate("سؤال", None, corpus, chat)["answer"]

        # A. Unsupported content without the prefix is dropped.
        out = answer_for(hallucination)
        check("unsupported claim without refusal prefix is dropped",
              out == REFUSAL)

        # B. The same content behind the conventional refusal prefix must
        #    not skip grounding (the RAG-01 bypass).
        out = answer_for(REFUSAL + " " + hallucination)
        check("refusal prefix cannot skip the grounding filter",
              out == REFUSAL)
        check("smuggled claim is absent from the returned answer",
              hallucination not in out)

        # C. A bare refusal stays the canonical refusal.
        out = answer_for(REFUSAL)
        check("canonical refusal is preserved", out == REFUSAL)

        # D. A supported answer survives grounding and keeps its valid
        #    citation marker for auditability.
        out = answer_for(supported_cited)
        check("supported answer survives grounding",
              out == supported_cited)
        check("valid source citation remains traceable",
              "(مصدر 1)" in out)

        # E. Mixed supported/unsupported sentences with citations: keep only
        #    the supported sentence and its valid citation.
        out = answer_for(supported_cited + " " + hallucination_cited)
        check("mixed answer keeps the supported sentence",
              out == supported_cited)
        check("mixed answer drops the unsupported sentence",
              hallucination not in out)
    finally:
        P.retrieve = orig_retrieve


# ---------------------------------------------------------------------------
# 17. Single-flight BM25 initialization (RAG-03 R1)
# ---------------------------------------------------------------------------

def _bm25_synth_corpus(prefix: str, n: int = 60) -> list[dict]:
    return [
        _chunk(f"{prefix}{i}", "fiqh",
               f"نص تجريبي رقم {i} يشمل كلمات متكررة وبحثا لفظيا {i}")
        for i in range(n)
    ]


def test_bm25_single_flight() -> None:
    print("\n[17] Single-flight BM25 initialization (RAG-03 R1)")
    corpus = _bm25_synth_corpus("sf")
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    orig_tokenize = P._tokenize_clean
    first_call = {"seen": False}
    first_gate = threading.Lock()

    def slow_first_tokenize(text: str) -> list[str]:
        # Widen the race window: the first builder pauses after entering the
        # construction body, so without single-flight every other cold caller
        # has already passed the cache check before any index is published.
        with first_gate:
            is_first = not first_call["seen"]
            first_call["seen"] = True
        if is_first:
            time.sleep(0.3)
        return orig_tokenize(text)

    n_threads = 6
    barrier = threading.Barrier(n_threads)
    results: list[dict] = []
    errors: list[BaseException] = []
    result_gate = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait(timeout=15)
            idx = P._build_bm25(corpus)
            with result_gate:
                results.append(idx)
        except BaseException as exc:  # surfaced through `errors` below
            with result_gate:
                errors.append(exc)

    P._tokenize_clean = slow_first_tokenize
    try:
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        P._tokenize_clean = orig_tokenize

    check("all concurrent cold callers obtained an index",
          not errors and len(results) == n_threads)
    check("concurrent cold callers built exactly one index object",
          len({id(r) for r in results}) == 1)
    check("cache stores the shared index",
          bool(results) and P._bm25_cache.get(key) is results[0])

    first = P._bm25_cache.get(key)
    second = P._build_bm25(corpus)
    check("subsequent call is a cache hit (same object, no rebuild)",
          second is first)
    P._bm25_cache.pop(key, None)


def test_bm25_failure_recovery() -> None:
    print("\n[18] BM25 failed-build recovery (RAG-03 R1)")
    corpus = _bm25_synth_corpus("fr", n=20)
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    orig_tokenize = P._tokenize_clean
    calls = {"n": 0}

    def exploding_tokenize(text: str) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected tokenizer failure")
        return orig_tokenize(text)

    tracing_before = tracemalloc.is_tracing()
    P._tokenize_clean = exploding_tokenize
    raised = False
    try:
        P._build_bm25(corpus)
    except RuntimeError:
        raised = True
    finally:
        P._tokenize_clean = orig_tokenize

    check("injected build failure propagates", raised)
    check("failed build leaves no cache entry", key not in P._bm25_cache)
    acquired = P._bm25_build_lock.acquire(blocking=False)
    if acquired:
        P._bm25_build_lock.release()
    check("build lock is released after a failed build", acquired)
    check("failed build does not leave tracing enabled",
          tracemalloc.is_tracing() is tracing_before)

    idx = P._build_bm25(corpus)
    check("a later valid build succeeds after the failure",
          idx.get("N") == len(corpus))
    hits = P._bm25_search(["تجريبي"], corpus, top_k=3)
    check("recovered index serves BM25 searches", len(hits) > 0)
    P._bm25_cache.pop(key, None)


def test_bm25_tracing_ownership() -> None:
    print("\n[19] BM25 tracemalloc ownership (RAG-03 R1)")
    check("suite runs this test with tracing off",
          not tracemalloc.is_tracing())

    # Simulate a foreign tracer: tracing is already active when the build
    # runs, so the build must leave it running.
    corpus = _bm25_synth_corpus("tr", n=20)
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)
    tracemalloc.start()
    try:
        P._build_bm25(corpus)
        check("build leaves externally started tracing active",
              tracemalloc.is_tracing())
    finally:
        tracemalloc.stop()
        P._bm25_cache.pop(key, None)

    # Normal path: tracing off before the build, off again after it.
    corpus2 = _bm25_synth_corpus("tr2", n=20)
    key2 = (id(corpus2), len(corpus2))
    P._bm25_cache.pop(key2, None)
    P._build_bm25(corpus2)
    check("build stops tracing it started", not tracemalloc.is_tracing())
    P._bm25_cache.pop(key2, None)


def test_bm25_grounding_reuse() -> None:
    print("\n[20] Grounding reuses the single BM25 build (RAG-01 path)")
    chunk = _chunk("sfg0", "fiqh",
                   "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
    retrieved = [dict(chunk, score=0.91, _gate_passed=True)]
    corpus = [chunk]
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    orig_retrieve = P.retrieve
    P.retrieve = lambda query, index, chunks, client, top_k=5: [
        dict(retrieved[0])]
    try:
        chat = _FakeChat(
            {}, "الصلاة لا تجب على المسلم في السفر مطلقا (مصدر 1).")
        out = P.generate("سؤال", None, corpus, chat)
    finally:
        P.retrieve = orig_retrieve

    check("unsupported answer is grounded to the canonical refusal",
          out["answer"] == REFUSAL)
    built = P._bm25_cache.get(key)
    check("grounding built the BM25 index for the corpus", built is not None)
    reused = P._build_bm25(corpus)
    check("grounding index is reused (single build)",
          built is not None and reused is built)
    P._bm25_cache.pop(key, None)


# ---------------------------------------------------------------------------
# 21-23. Startup prewarming and readiness (RAG-03 R2)
# ---------------------------------------------------------------------------

def _r2_fake_resources(corpus: list[dict]):
    """Return an `_initialize_resources` replacement installing fake globals.

    The real loader reads the production FAISS index and chunks.pkl; these
    tests must stay offline, so the lifespan loads a synthetic corpus instead.
    """
    def _init() -> None:
        M.index = None
        M.chunks = corpus
        M.client = None
    return _init


def _r2_save_state() -> tuple:
    return (M._initialize_resources, P._bm25_construct, M._service_phase,
            M.index, M.chunks, M.client)


def _r2_restore_state(state: tuple) -> None:
    (M._initialize_resources, P._bm25_construct, M._service_phase,
     M.index, M.chunks, M.client) = state


def test_r2_startup_prewarm() -> None:
    print("\n[21] R2 startup prewarm builds the canonical corpus once")
    corpus = _bm25_synth_corpus("r2a", n=30)
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    state = _r2_save_state()
    orig_construct = P._bm25_construct
    calls = {"n": 0}

    def counting_construct(chunks: list[dict], tracing_started: bool) -> dict:
        calls["n"] += 1
        return orig_construct(chunks, tracing_started)

    M._service_phase = M._INITIALIZING
    M._initialize_resources = _r2_fake_resources(corpus)
    P._bm25_construct = counting_construct
    try:
        # Required initialization has not run yet: not ready, nothing built.
        probe = TestClient(M.app)
        r = probe.get("/ready")
        check("readiness is false while initialization is incomplete",
              r.status_code == 503 and r.json() == {"status": "initializing"})
        check("nothing was prewarmed before startup", calls["n"] == 0)

        with TestClient(M.app) as client:
            check("startup prewarmed the canonical corpus exactly once",
                  calls["n"] == 1)
            check("service is ready only after a successful prewarm",
                  M.service_phase() == "ready")
            r = client.get("/ready")
            check("GET /ready reports 200 ready",
                  r.status_code == 200 and r.json() == {"status": "ready"})
            r = client.get("/health")
            check("GET /health reports liveness",
                  r.status_code == 200 and r.json() == {"status": "ok"})
            built = P._bm25_cache.get(key)
            check("prewarmed index is cached under the canonical key",
                  built is not None)
            reused = P._build_bm25(corpus)
            check("post-startup lookup reuses the prewarmed index",
                  reused is built and calls["n"] == 1)
        check("shutdown records the stopped phase",
              M.service_phase() == "stopped")
    finally:
        _r2_restore_state(state)
        P._bm25_cache.pop(key, None)


def test_r2_prewarm_failure() -> None:
    print("\n[22] R2 prewarm failure fails startup and never reports ready")
    corpus = _bm25_synth_corpus("r2b", n=20)
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    state = _r2_save_state()
    tracing_before = tracemalloc.is_tracing()

    def exploding_construct(chunks: list[dict], tracing_started: bool) -> dict:
        raise RuntimeError("injected prewarm failure")

    M._service_phase = M._INITIALIZING
    M._initialize_resources = _r2_fake_resources(corpus)
    P._bm25_construct = exploding_construct
    try:
        raised = False
        try:
            with TestClient(M.app):
                pass
        except RuntimeError:
            raised = True
        check("prewarm failure aborts startup (fail-fast policy)", raised)
        check("failed startup never marks the service ready",
              M.service_phase() == "failed")

        r = TestClient(M.app).get("/ready")
        check("readiness reports 503 failed after a failed prewarm",
              r.status_code == 503 and r.json() == {"status": "failed"})
        check("failed prewarm left no cache entry", key not in P._bm25_cache)
        acquired = P._bm25_build_lock.acquire(blocking=False)
        if acquired:
            P._bm25_build_lock.release()
        check("failed prewarm released the build lock", acquired)
        check("failed prewarm left tracemalloc in its original state",
              tracemalloc.is_tracing() is tracing_before)
    finally:
        _r2_restore_state(state)
        P._bm25_cache.pop(key, None)


def test_r2_lifecycle_repeat_and_shutdown() -> None:
    print("\n[23] R2 repeated lifecycle calls reuse the prewarmed index")
    corpus = _bm25_synth_corpus("r2c", n=15)
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    state = _r2_save_state()
    orig_construct = P._bm25_construct
    calls = {"n": 0}

    def counting_construct(chunks: list[dict], tracing_started: bool) -> dict:
        calls["n"] += 1
        return orig_construct(chunks, tracing_started)

    M._service_phase = M._INITIALIZING
    M._initialize_resources = _r2_fake_resources(corpus)
    P._bm25_construct = counting_construct
    try:
        with TestClient(M.app) as client:
            check("first startup prewarms once", calls["n"] == 1)
            r = client.get("/ready")
            check("first startup reports ready",
                  r.status_code == 200 and r.json() == {"status": "ready"})
        check("first shutdown records the stopped phase",
              M.service_phase() == "stopped")
        r = TestClient(M.app).get("/ready")
        check("readiness is false after shutdown",
              r.status_code == 503 and r.json() == {"status": "stopped"})

        with TestClient(M.app) as client:
            check("repeated startup does not rebuild (cache reuse)",
                  calls["n"] == 1)
            check("second startup reports ready again",
                  M.service_phase() == "ready")
        check("second shutdown records the stopped phase",
              M.service_phase() == "stopped")
    finally:
        _r2_restore_state(state)
        P._bm25_cache.pop(key, None)


# ---------------------------------------------------------------------------
# 24-26. RAG-02 endpoint grounding/validation (tafsir + verify)
# ---------------------------------------------------------------------------

# Fixture texts mirror the shapes these endpoints actually see: an
# Ibn-Kathir-style tafsir excerpt pair, and a hadith verification case with a
# matn metadata entry for the deterministic truncation check.
_RAG02_TAFSIR_PRIMARY = (
    "هذه الآية أعظم آية في القرآن الكريم، وهي آية الكرسي، أخبر الله فيها عن "
    "وحدانيته وقيوميته، وأنه الحي الذي لا يموت، والقيوم الذي لا تأخذه سنة "
    "ولا نوم، وله ما في السماوات وما في الأرض"
)
_RAG02_TAFSIR_NEXT = (
    "ذكر ابن كثير أن آية الكرسي تجمع أصول التوحيد، وفيها إثبات العلم والقدرة "
    "والملك لله وحده، ومن قرأها في ليلته لم يزل عليه من الله حافظ"
)
# Morphological/clitic paraphrase (آية→آيات, وحدانيته→بوحدانيته, reordered):
# the realistic shape of a temperature-0 summariser constrained to the
# excerpts — must survive grounding.
_RAG02_TAFSIR_PARAPHRASE = (
    "تعد هذه الآية أعظم آيات القرآن، فقد أخبر الله فيها بوحدانيته "
    "وقيوميته، فهو الحي الذي لا يموت، والقيوم الذي لا تأخذه سنة ولا نوم"
)
# Distinctive fabricated claim (asbab al-nuzul + a hadith attribution that
# occurs nowhere in the supplied evidence) — must be rejected.
_RAG02_UNSAFE_CLAIM = (
    "وقد ذهب جمهور المفسرين إلى أن هذه الآية نزلت في المدينة، وروى "
    "البخاري أن سبب نزولها غزوة تبوك"
)

_RAG02_VERIFY_TEXT = "إنما الأعمال بالنيات"
_RAG02_VERIFY_EVIDENCE = (
    "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات وإنما لكل امرئ "
    "ما نوى"
)
_RAG02_VERIFY_MATN = "إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"


def _rag02_tafsir_fixtures() -> tuple[dict, dict]:
    primary = _chunk("t1", "tafsir", _RAG02_TAFSIR_PRIMARY,
                     ref="تفسير ابن كثير 2:255")
    nxt = _chunk("t2", "tafsir", _RAG02_TAFSIR_NEXT,
                 ref="تفسير ابن كثير 2:256")
    return primary, nxt


def _rag02_high_df_corpus(evidence: list[dict], prefix: str,
                          n: int = 600) -> list[dict]:
    """Corpus whose filler docs make paraphrase connectives high-df.

    The grounding engine's rare-token bound is document-frequency based; in
    the production corpus these words are extremely common, so tests must
    not classify them as rare.  The actual grounding evidence remains the
    two retrieved excerpts.
    """
    vocab = "تعد فقد فهو وهو وقد آيات أعظم فيها والقيوم بوحدانيته"
    corpus = [dict(c) for c in evidence]
    for i in range(n):
        corpus.append(_chunk(f"{prefix}f{i}", "fiqh",
                             f"{vocab} نص تجريبي رقم {i}"))
    return corpus


def _rag02_verify_fixtures() -> tuple[dict, dict]:
    hadith = _chunk("h1", "hadith", _RAG02_VERIFY_EVIDENCE,
                    ref="صحيح البخاري 1")
    hadith["metadata"] = {"matn": _RAG02_VERIFY_MATN}
    quran = _chunk("q1", "quran", "لا يكلف الله نفسا إلا وسعها",
                   ref="البقرة 286")
    return hadith, quran


def test_rag02_tafsir_grounding() -> None:
    print("\n[24] RAG-02 tafsir post-generation grounding")
    primary, nxt = _rag02_tafsir_fixtures()
    corpus = _rag02_high_df_corpus([primary, nxt], "rag02t")
    key = (id(corpus), len(corpus))
    P._bm25_cache.pop(key, None)

    orig_resolve = G.resolve_tafsir_reference
    try:
        def run(payload, resolve=(primary, nxt, "2:255"), explode=False):
            G.resolve_tafsir_reference = (
                lambda ayah_text, index, chunks, client, debug=False: resolve
            )
            chat = _FakeChat({}, payload, explode=explode)
            try:
                return G.get_tafsir("آية اختبار", object(), corpus, chat), chat
            finally:
                G.resolve_tafsir_reference = orig_resolve

        out, _ = run(json.dumps(
            {"tafsir": _RAG02_TAFSIR_PRIMARY, "context": _RAG02_TAFSIR_NEXT},
            ensure_ascii=False))
        check("supported tafsir prose is retained",
              out["tafsir"] == _RAG02_TAFSIR_PRIMARY)
        check("supported context is retained",
              out["context"] == _RAG02_TAFSIR_NEXT)
        check("source stays the server-derived chunk reference",
              out["source"] == "2:255")
        check("tafsir response keys unchanged",
              sorted(out) == ["context", "input_ayah", "source", "tafsir"])
        check("tafsir response values are all strings",
              all(isinstance(v, str) for v in out.values()))

        out, _ = run(json.dumps(
            {"tafsir": _RAG02_TAFSIR_PARAPHRASE, "context": ""},
            ensure_ascii=False))
        check("reasonable morphological paraphrase is not rejected",
              out["tafsir"] == _RAG02_TAFSIR_PARAPHRASE)

        # Falsifiable: pre-fix this returned the fabricated text verbatim.
        out, _ = run(json.dumps(
            {"tafsir": _RAG02_UNSAFE_CLAIM, "context": _RAG02_TAFSIR_NEXT},
            ensure_ascii=False))
        check("distinctive unsupported tafsir claim is rejected",
              out["tafsir"] == G._TAFSIR_REFUSAL)
        check("context is emptied when tafsir is rejected",
              out["context"] == "")

        out, _ = run(json.dumps(
            {"tafsir": _RAG02_TAFSIR_PRIMARY, "context": _RAG02_UNSAFE_CLAIM},
            ensure_ascii=False))
        check("safe tafsir survives an ungrounded context",
              out["tafsir"] == _RAG02_TAFSIR_PRIMARY)
        check("ungrounded context is dropped independently",
              out["context"] == "")

        out, chat = run("{}", resolve=(None, None, ""), explode=True)
        check("no-match returns the established refusal",
              out["tafsir"] == G._TAFSIR_REFUSAL and out["source"] == "")
        check("no-match path never calls the model", chat.called is False)

        out, _ = run("not json at all")
        check("malformed JSON keeps the established fallback",
              out["tafsir"] == "لم أجد تفسير هذه الآية في قاعدة البيانات"
              and out["context"] == "")
        check("malformed JSON still preserves the server-derived source",
              out["source"] == "2:255")
    finally:
        G.resolve_tafsir_reference = orig_resolve
        P._bm25_cache.pop(key, None)


def test_rag02_verify_validation() -> None:
    print("\n[25] RAG-02 verify post-generation validation")
    hadith, quran = _rag02_verify_fixtures()
    corpus = [hadith, quran]
    results = [
        {"id": "h1", "source": "hadith", "reference": "صحيح البخاري 1",
         "original_text": _RAG02_VERIFY_EVIDENCE, "score": 0.92},
        {"id": "q1", "source": "quran", "reference": "البقرة 286",
         "original_text": "لا يكلف الله نفسا إلا وسعها", "score": 0.71},
    ]
    orig_search = V._dual_search
    try:
        def run(payload, res=results, text=_RAG02_VERIFY_TEXT):
            V._dual_search = (
                lambda t, c, i, ch, top_k, debug=False: list(res)
            )
            chat = _FakeChat({}, payload)
            try:
                return V.verify_full(text, "شرح", object(), corpus, chat)
            finally:
                V._dual_search = orig_search

        def payload(**over):
            base = {
                "text_status": "صحيح", "is_truncated": "لا",
                "missing_part": "", "full_text": "", "source": "",
                "danger_level": "منخفضة", "explanation_status": "موافق",
                "explanation_verdict": "", "correct_explanation": "",
            }
            base.update(over)
            return json.dumps(base, ensure_ascii=False)

        out = run(payload(source="صحيح البخاري: ١"))
        check("valid retrieved citation is retained",
              out["source"] == "صحيح البخاري: ١")

        # Falsifiable: pre-fix returned the fabricated citation verbatim.
        out = run(payload(source="صحيح مسلم 9999"))
        check("fabricated citation is replaced by the top retrieved reference",
              out["source"] == "صحيح البخاري 1")

        out = run(payload(correct_explanation=_RAG02_VERIFY_MATN))
        check("grounded correct_explanation is retained",
              out["correct_explanation"] == _RAG02_VERIFY_MATN)

        # Falsifiable: pre-fix returned the ungrounded explanation verbatim.
        out = run(payload(correct_explanation=_RAG02_UNSAFE_CLAIM))
        check("ungrounded correct_explanation is neutralized",
              out["correct_explanation"] == "")

        out = run(payload(text_status="ممتاز", danger_level="خطر شديد",
                          explanation_status="ربما"))
        check("invalid text_status falls back to غير موجود",
              out["text_status"] == "غير موجود")
        check("invalid danger_level falls back to منخفضة",
              out["danger_level"] == "منخفضة")
        check("invalid explanation_status falls back to لا يوجد",
              out["explanation_status"] == "لا يوجد")

        verdict = "الشرح موافق لأن النص يثبت أن ثواب العمل بالنيات"
        out = run(payload(explanation_verdict=verdict))
        check("explanation_verdict passes through (evaluative field, by design)",
              out["explanation_verdict"] == verdict)

        out = run(payload(is_truncated="لا", missing_part="مزعوم",
                          full_text="نص مزعوم"))
        check("deterministic truncation overrides model output",
              out["is_truncated"] is True
              and out["full_text"] == _RAG02_VERIFY_MATN)
        check("deterministic missing part overrides model output",
              out["missing_part"] != "مزعوم")

        out = run(payload(is_truncated="نعم", missing_part="جزء",
                          full_text="نص كامل مزعوم"), text="الدين النصيحة")
        check("non-firing truncation case keeps the model values",
              out["is_truncated"] is True and out["missing_part"] == "جزء"
              and out["full_text"] == "نص كامل مزعوم")

        # Falsifiable: pre-fix returned text_status صحيح with no evidence.
        out = run(payload(text_status="صحيح", source="صحيح البخاري 1",
                          correct_explanation=_RAG02_VERIFY_MATN), res=[])
        check("empty retrieval forces a safe verdict",
              out["text_status"] == "غير موجود")
        check("empty retrieval yields an empty source", out["source"] == "")
        check("empty retrieval cannot ground prose",
              out["correct_explanation"] == "")

        out = run("not json", text="الدين النصيحة")
        check("malformed JSON yields the safe defaults",
              out["text_status"] == "غير موجود"
              and out["danger_level"] == "منخفضة"
              and out["explanation_status"] == "لا يوجد")
        check("malformed JSON keeps is_truncated boolean",
              isinstance(out["is_truncated"], bool))

        out = run(payload(source="صحيح البخاري: ١",
                          correct_explanation=_RAG02_VERIFY_MATN))
        check("verify response exposes exactly the 11 contract keys",
              set(out) == {
                  "text_status", "is_truncated", "missing_part", "full_text",
                  "source", "danger_level", "explanation_status",
                  "explanation_verdict", "correct_explanation",
                  "input_text", "input_explanation",
              })
        check("is_truncated is bool; all other values are strings",
              isinstance(out["is_truncated"], bool)
              and all(isinstance(out[k], str) for k in out
                      if k != "is_truncated"))
    finally:
        V._dual_search = orig_search


def test_rag02_ground_text_wrapper_and_chat_path() -> None:
    print("\n[26] RAG-02 ground_text wrapper and unchanged chat path")
    index, chunks = _gen_corpus()
    retrieved = [{
        "id": "g0", "source": "fiqh", "reference": "مرجع",
        "original_text": "محتوى الفقه في الطهارة", "score": 0.9,
    }]
    kept = "محتوى الفقه في الطهارة"
    rejected = "غزوة بدر كانت في شهر شوال."

    check("wrapper is behaviour-identical with refusal=None (kept text)",
          P.ground_text(kept, retrieved, chunks)
          == P._ground_answer(kept, retrieved, chunks))
    check("wrapper is behaviour-identical with refusal=None (rejected text)",
          P.ground_text(rejected, retrieved, chunks)
          == P._ground_answer(rejected, retrieved, chunks))
    check("wrapper maps total rejection to the caller refusal",
          P.ground_text(rejected, retrieved, chunks, refusal="س") == "س")
    check("wrapper never rewrites kept text",
          P.ground_text(kept, retrieved, chunks, refusal="س") == kept)

    orig_embed = P.embed_text
    orig_rewrite = P._rewrite_query
    orig_ground = P._ground_answer
    calls: dict = {"n": 0}
    try:
        P.embed_text = lambda text, client: [1.0, 0.0, 0.0]
        P._rewrite_query = lambda query, client: []

        def recorder(answer, retrieved, corpus_chunks=None):
            calls["n"] += 1
            calls["answer"] = answer
            return "SENTINEL"

        P._ground_answer = recorder
        chat = _FakeChat({}, "محتوى الفقه في الطهارة (مصدر 1).")
        out = P.generate("زقزق", index, chunks, chat)
    finally:
        P.embed_text = orig_embed
        P._rewrite_query = orig_rewrite
        P._ground_answer = orig_ground
    check("generate still routes through _ground_answer directly",
          calls["n"] == 1)
    check("chat answer is _ground_answer's return value",
          out["answer"] == "SENTINEL")
    check("chat grounding receives the freshly generated answer",
          calls.get("answer") == "محتوى الفقه في الطهارة (مصدر 1).")


# ---------------------------------------------------------------------------

def main() -> None:
    test_rrf_sem_signal()
    test_selection()
    test_gate_real_faiss()
    test_terminology()
    test_grounding()
    test_matcher()
    test_embedding_interface()
    test_generation_branches()
    test_multi_evidence_selection()
    test_query_embedding_file()
    test_negation_guard()
    test_cf_format_chars()
    test_variant_suffix_order()
    test_p5v5_grounding_policy()
    test_grounding_curated_alias()
    test_refusal_prefix_grounding()
    test_bm25_single_flight()
    test_bm25_failure_recovery()
    test_bm25_tracing_ownership()
    test_bm25_grounding_reuse()
    test_r2_startup_prewarm()
    test_r2_prewarm_failure()
    test_r2_lifecycle_repeat_and_shutdown()
    test_rag02_tafsir_grounding()
    test_rag02_verify_validation()
    test_rag02_ground_text_wrapper_and_chat_path()
    print(f"\n[offline_tests] {_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
