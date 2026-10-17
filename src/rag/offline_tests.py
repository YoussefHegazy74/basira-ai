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

Usage:
    PYTHONDONTWRITEBYTECODE=1 venv/bin/python src/rag/offline_tests.py

Exit code 0 when all tests pass, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import src.rag.pipeline as P  # noqa: E402
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
    from src.rag.offline_eval import _load_query_embeddings, _mean_vector  # noqa: PLC0415

    path = "/tmp/opencode/offline_tests_query_embeddings.json"
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

    # neg-window-fp residual: the documented false rejection of the frozen
    # set.  نزول is unmatched with union df 238 <= 300 (rare), so the
    # sentence stays dropped — pinned here so any threshold drift shows up.
    src_nw = _chunk(
        "p5_nw", "hadith",
        "عن ابن عباس قال لما نزلت هذه الآية ولا تقربوا مال اليتيم إلا "
        "بالتي هي أحسن قال اجتنب الناس مال اليتيم وطعامه")
    corpus_nw = [src_nw] + [_chunk(f"p5_nwi{i}", "fiqh", "نزول فقه")
                            for i in range(238)]
    out = P._ground_answer(
        "كان ابن عباس يرى اجتناب مال اليتيم بعد نزول الآية (مصدر 1).",
        [src_nw], corpus_nw)
    check("neg-window-fp residual fixture still dropped (documented FR)",
          out.strip() == REFUSAL)

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
    print(f"\n[offline_tests] {_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
