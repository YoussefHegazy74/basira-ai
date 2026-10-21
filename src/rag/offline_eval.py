"""Deterministic OFFLINE evaluation harness for the Basira RAG pipeline.

Runs without any external API / network access.

SEMANTIC MODES (mutually exclusive CLI flags; `--sem-vectors` preferred):

  * lexical (default): the semantic stage is stubbed to empty results and the
    LLM query-rewrite returns [].  All metrics are LEXICAL-ONLY lower bounds.
    Semantic retrieval is reported as unmeasured.

  * --sem-vectors: the GENUINE semantic path is exercised: the real
    data/processed/faiss.index is loaded and the REAL
    src.vector_db.faiss_store.search() runs over it with L2-normalized
    vectors inside the production retrieve() code.  The ONLY mocked step is
    the OpenAI query-encoding call (embed_text): each question uses the REAL
    stored embedding of an annotated corpus chunk (`sem_proxy_id`, default
    first evidence_id) as the query vector.  This is a transparent proxy
    fixture: it measures plumbing + retrieval ranking behavior against real
    corpus vectors, NOT the quality of the API embedding model.  Questions
    with no honest proxy (unanswerable) are SKIPPED in this mode because a
    fabricated proxy would force a perfect self-match (cosine 1.0) and make
    the gate/refusal behavior meaningless.

  * --sem-queries FILE: GENUINE query embeddings supplied by the operator
    (e.g. captured from an authorized API run).  Format:
        {"<question_id>": {"clean": [1536 floats],
                           "raw":   [1536 floats],   # optional, falls back
                           "model": "text-embedding-3-small",
                           "captured_at": "..."}}
    The harness validates shape/dimension only; it CANNOT verify provenance,
    so results are only as genuine as the file.  Questions missing an entry
    are skipped with a clear reason.

  * --sem-mixed: DIAGNOSTIC upper-bound proxy (label before sharing): the
    query vector is the L2-normalized MEAN of the annotated evidence
    embeddings — simulating a real question embedding that covers all
    evidence topics at once.  Still NOT a genuine query embedding; it is
    used to separate pipeline defects from query-representation limits.

  * --sem-fixture FILE: fully mocked semantic stage (cached ranks injected
    directly in place of search results).  No fixture ships by default.

Metrics per answerable question:
    orig_rank / exp_rank   lexical diagnostics (BM25 top-100 / synonym lists)
    sem_rank               best evidence rank in semantic top-100 (if mode
                           provides semantic results; else None)
    pool_rank              best evidence rank in the RRF-fused pool
    Recall@K               for semantic hits and for the fused pool
    in_context             evidence chunk present in final selected top-k
    coverage               fraction of annotated evidence chunks in context
    multi_chunk_full       ALL annotated evidence chunks in context
    gate                   retrieval gate state for the query

Grounding cases run directly against _ground_answer and are compared with
their annotated keep/drop expectation; an optional `absent` phrase list
verifies sentence-level citation scoping on kept outputs.

Usage:
    PYTHONDONTWRITEBYTECODE=1 venv/bin/python src/rag/offline_eval.py \
        [--sem-vectors] [--json /tmp/opencode/out.json]

Exit code 0 always; read the printed summary.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys

sys.path.insert(0, ".")

import src.rag.pipeline as P  # noqa: E402
from src.rag.eval_set import EVAL_QUESTIONS, GROUNDING_CASES  # noqa: E402

CHUNKS_PATH = "data/processed/chunks.pkl"
INDEX_PATH = "data/processed/faiss.index"

_ORIGINAL = {
    name: getattr(P, name)
    for name in ("embed_text", "search", "_rewrite_query", "_select_diverse")
}

REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."

K_VALUES = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Offline stubbing of API-bound stages
# ---------------------------------------------------------------------------

def _enrich(items: list[dict], chunk_by_id: dict[str, dict]) -> list[dict]:
    """Map minimal {id, score} fixtures to full chunk-shaped dicts."""
    out = []
    for it in items:
        cid = it.get("id")
        base = chunk_by_id.get(cid)
        if base is None:
            continue
        merged = dict(base)
        merged["score"] = float(it.get("score", 0.0))
        out.append(merged)
    return out


def _patch(sem_mode: str, current_proxy,
           fixture_items: list[dict] | None,
           chunk_by_id: dict[str, dict]) -> None:
    # Always start from production functions.
    P.search = _ORIGINAL["search"]
    P._rewrite_query = lambda query, client: []                    # no API
    if sem_mode in ("vectors", "mixed"):
        # Genuine FAISS search; only the query encoder is proxied.
        # Never falls back to the API-bound original.
        P.embed_text = lambda text, client: current_proxy
    elif sem_mode == "queries":
        # Operator-supplied (claimed genuine) query embeddings; the encoder
        # maps the exact query strings retrieve() passes (normalized and raw)
        # to the cached vectors.  Provenance is NOT verifiable here.
        _mapping = current_proxy

        def _cached_encoder(text: str, client):
            return _mapping.get(text, _mapping.get("*"))

        P.embed_text = _cached_encoder
    elif sem_mode == "fixture":
        P.embed_text = lambda text, client: None
        P.search = lambda emb, index, chunks, top_k: _enrich(
            list(fixture_items or []), chunk_by_id
        )[:top_k]
    else:  # lexical
        P.embed_text = lambda text, client: None
        P.search = lambda emb, index, chunks, top_k: []


def _unpatch() -> None:
    for name, fn in _ORIGINAL.items():
        setattr(P, name, fn)


def _load_sem_fixture(path: str | None) -> dict[str, list[dict]]:
    """Optional user-supplied cached semantic ranks (none shipped).

    Format: {"<question>": [{"id": "...", "score": 0.42}, ...]}
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_query_embeddings(path: str, dim: int) -> dict[str, dict]:
    """Load operator-supplied (claimed genuine) cached query embeddings.

    Validates structure and dimension ONLY.  Provenance cannot be verified:
    results are flagged accordingly in the JSON disclosure.

    Format: {"<question_id>": {"clean": [...], "raw": [...], ...}}
    Raises ValueError naming the offending entry.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("query embedding file must be a JSON object")
    out: dict[str, dict] = {}
    for qid, entry in data.items():
        if not isinstance(entry, dict) or "clean" not in entry:
            raise ValueError(f"{qid}: entry must be an object with a 'clean' vector")
        clean = entry["clean"]
        raw = entry.get("raw", clean)
        for name, vec in (("clean", clean), ("raw", raw)):
            if (
                not isinstance(vec, list)
                or len(vec) != dim
                or not all(isinstance(x, (int, float)) for x in vec[:8])
            ):
                raise ValueError(
                    f"{qid}: '{name}' vector must be a list of {dim} numbers "
                    f"(got {type(vec).__name__}, len="
                    f"{len(vec) if isinstance(vec, list) else 'n/a'})"
                )
        out[qid] = {"clean": clean, "raw": raw, "meta": {
            k: v for k, v in entry.items() if k not in ("clean", "raw")
        }}
    return out


def _mean_vector(vecs: list[list[float]]) -> list[float]:
    """L2-normalized component-wise mean (pure Python; no numpy import)."""
    n = len(vecs)
    dim = len(vecs[0])
    mean = [sum(v[i] for v in vecs) / n for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5
    if norm == 0.0:
        return mean
    return [x / norm for x in mean]


# ---------------------------------------------------------------------------
# Helpers replicating retrieve()'s exact query-token construction
# ---------------------------------------------------------------------------

def _query_tokens(question: str) -> list[str]:
    processed = P.preprocess(question)
    clean_query = processed["normalized"]
    return [
        P._strip_al(w)
        for w in (processed.get("tokens_clean") or clean_query.split())
        if w not in P._QUERY_STOPWORDS and len(w) >= 2
    ]


def _evidence_rank(chunks_list: list[dict], evidence_ids: set[str]) -> int | None:
    for i, c in enumerate(chunks_list, 1):
        if c.get("id") in evidence_ids:
            return i
    return None


def _recall_at(chunks_list: list[dict], evidence_ids: set[str]) -> dict[str, bool]:
    if not evidence_ids:
        return {}
    r = _evidence_rank(chunks_list, evidence_ids)
    return {f"recall@{k}": (r is not None and r <= k) for k in K_VALUES}


def _gate_state(retrieved: list[dict]) -> bool:
    if not retrieved:
        return False
    first = retrieved[0]
    if "_gate_passed" in first:
        return bool(first["_gate_passed"])
    # Legacy (pre-F-07) semantics: gate failure zeroes every score.
    return any(c.get("score", 0.0) > 0 for c in retrieved)


def _proxy_id(q: dict) -> str | None:
    pid = q.get("sem_proxy_id")
    if pid:
        return pid
    ev = q.get("evidence_ids") or []
    return ev[0] if ev else None


# ---------------------------------------------------------------------------
# Retrieval evaluation (production path, query encoder stubbed)
# ---------------------------------------------------------------------------

def run_retrieval_eval(
    chunks: list[dict],
    chunk_by_id: dict[str, dict],
    index,
    sem_mode: str,
    sem_fixture: dict[str, list[dict]],
    query_vectors: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for q in EVAL_QUESTIONS:
        evidence = set(q["evidence_ids"])
        tokens = _query_tokens(q["question"])

        orig_hits = P._bm25_search(tokens, chunks, top_k=100)
        orig_rank = _evidence_rank(orig_hits, evidence) if evidence else None

        exp_rank = None
        if evidence:
            for alt in P._expand_synonyms(tokens):
                alt_hits = P._bm25_search(alt, chunks, top_k=50)
                r = _evidence_rank(alt_hits, evidence)
                if r is not None and (exp_rank is None or r < exp_rank):
                    exp_rank = r

        proxy_pid = _proxy_id(q)
        proxy_vec = chunk_by_id[proxy_pid]["embedding"] if proxy_pid else None

        # --- Resolve the semantic query vector (and its honest label) ------
        qvec_clean = None
        qvec_raw = None
        sem_source = None
        skip_reason = None
        if sem_mode == "vectors":
            if proxy_vec is None:
                skip_reason = (
                    "no honest proxy vector for this question; a fabricated "
                    "proxy would force a perfect self-match and nullify the gate"
                )
            else:
                qvec_clean = qvec_raw = proxy_vec
                sem_source = f"proxy({proxy_pid})"
        elif sem_mode == "mixed":
            ev_vecs = [chunk_by_id[e]["embedding"] for e in q["evidence_ids"]
                       if e in chunk_by_id]
            if not ev_vecs:
                skip_reason = (
                    "no annotated evidence chunk to build the mixed diagnostic "
                    "vector from (unanswerable)"
                )
            else:
                qvec_clean = qvec_raw = _mean_vector(ev_vecs)
                sem_source = f"mixed(mean of {len(ev_vecs)} evidence)"
        elif sem_mode == "queries":
            entry = (query_vectors or {}).get(q["id"])
            if entry is None:
                skip_reason = (
                    "no cached query embedding in the operator file for this "
                    "question; refusing to substitute a proxy"
                )
            else:
                qvec_clean = entry["clean"]
                qvec_raw = entry.get("raw", entry["clean"])
                model = (entry.get("meta") or {}).get("model", "unknown")
                sem_source = f"file(model={model})"

        row: dict = {
            "id": q["id"],
            "category": q["category"],
            "answerable": q["answerable"],
            "tokens": tokens,
            "orig_rank": orig_rank,
            "exp_rank": exp_rank,
            "sem_proxy_id": proxy_pid,
            "sem_query_source": sem_source,
            "sem_rank": None,
            "evidence_total": len(evidence),
            "evidence_in_ctx": 0,
            "in_context": False,
            "coverage": 0.0,
            "multi_chunk_full": False,
            "gate": False,
            "selected_ids": [],
        }

        if skip_reason is not None:
            row["skipped"] = skip_reason
            rows.append(row)
            continue

        if sem_mode == "queries":
            processed = P.preprocess(q["question"])
            encoder = {
                processed["normalized"]: qvec_clean,
                q["question"]: qvec_raw,
                "*": qvec_clean,
            }
        else:
            encoder = qvec_clean
        _patch(sem_mode, encoder, sem_fixture.get(q["question"]), chunk_by_id)

        # Semantic-only ranking diagnostic (genuine search in all these modes).
        if evidence:
            sem_hits = P.search(qvec_clean, index, chunks, top_k=100)
            row["sem_rank"] = _evidence_rank(sem_hits, evidence)
            row["sem_recall"] = _recall_at(sem_hits, evidence)

        # Pool mode: neutralize selection so the full fused pool is returned.
        P._select_diverse = (
            lambda candidates, top_k, bypass_ids=None, lex_top3_ids=None,
            sem_top3_ids=None, priority_ids=None, must_ids=None: candidates
        )
        pool = P.retrieve(q["question"], index, chunks, None, top_k=5)
        pool_rank = _evidence_rank(pool, evidence) if evidence else None
        row["pool_rank"] = pool_rank
        row["pool_recall"] = _recall_at(pool, evidence) if evidence else {}

        # Selection mode: production behavior.
        P._select_diverse = _ORIGINAL["_select_diverse"]
        selected = P.retrieve(q["question"], index, chunks, None, top_k=5)
        selected_ids = [c.get("id") for c in selected]
        hits = len(evidence & set(selected_ids)) if evidence else 0
        row["selected_ids"] = selected_ids
        row["evidence_in_ctx"] = hits
        row["coverage"] = (hits / len(evidence)) if evidence else 0.0
        row["multi_chunk_full"] = bool(evidence) and hits == len(evidence)
        row["in_context"] = hits > 0
        row["gate"] = _gate_state(selected)
        rows.append(row)
        _unpatch()
    return rows


# ---------------------------------------------------------------------------
# Grounding evaluation
# ---------------------------------------------------------------------------

def run_grounding_eval(chunks: list[dict]) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    rows: list[dict] = []
    for case in GROUNDING_CASES:
        retrieved = [by_id[cid] for cid in case["chunk_ids"]]
        output = P._ground_answer(case["answer"], retrieved, chunks)
        kept = output.strip() != REFUSAL and output.strip() != ""
        behavior_ok = (kept and case["expect"] == "keep") or (
            not kept and case["expect"] == "drop"
        )
        absent = case.get("absent") or []
        absent_hits = [p for p in absent if p in output]
        ok = behavior_ok and not absent_hits
        rows.append({
            "id": case["id"],
            "expect": case["expect"],
            "kept": kept,
            "behavior_ok": behavior_ok,
            "absent_hits": absent_hits,
            "ok": ok,
            "why": case["why"],
            "output": output[:120],
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mrr(rows: list[dict], key: str) -> float:
    vals = [1.0 / r[key] for r in rows if r.get(key)]
    return (sum(vals) / len(vals)) if vals else 0.0


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    agg = {
        "n": n,
        "in_context": sum(1 for r in rows if r["in_context"]),
        "multi_chunk_full": sum(1 for r in rows if r["multi_chunk_full"]),
        "avg_coverage": round(sum(r["coverage"] for r in rows) / n, 3),
        "pool_mrr": round(_mrr(rows, "pool_rank"), 3),
        "gate_pass": sum(1 for r in rows if r["gate"]),
    }
    for k in K_VALUES:
        agg[f"pool_recall@{k}"] = sum(
            1 for r in rows if (r.get("pool_recall") or {}).get(f"recall@{k}")
        )
        agg[f"sem_recall@{k}"] = sum(
            1 for r in rows if (r.get("sem_recall") or {}).get(f"recall@{k}")
        )
    sem_rows = [r for r in rows if r.get("sem_rank")]
    agg["sem_mrr"] = round(_mrr(rows, "sem_rank"), 3)
    agg["sem_ranked_n"] = sum(1 for r in rows if r.get("sem_rank") is not None)
    agg["sem_hit_any"] = len(sem_rows)
    return agg


def _by_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("answerable") and not r.get("skipped"):
            cats.setdefault(r["category"], []).append(r)
    return {c: _agg(rs) for c, rs in sorted(cats.items())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_path", default=None,
                    help="optional path to dump raw results as JSON")
    ap.add_argument("--sem-fixture", dest="sem_fixture", default=None,
                    help="optional cached semantic ranks (fully mocked; not shipped)")
    ap.add_argument("--sem-vectors", dest="sem_vectors", action="store_true",
                    help="genuine FAISS search using real corpus chunk embeddings "
                         "as query proxies (query encoder still local-only)")
    ap.add_argument("--sem-mixed", dest="sem_mixed", action="store_true",
                    help="DIAGNOSTIC upper-bound proxy: query vector = normalized "
                         "mean of the annotated evidence embeddings")
    ap.add_argument("--sem-queries", dest="sem_queries", default=None,
                    help="operator-supplied (claimed genuine) cached query "
                         "embeddings JSON; format documented in module docstring")
    args = ap.parse_args()

    chosen = [name for name, on in (
        ("fixture", bool(args.sem_fixture)),
        ("vectors", args.sem_vectors),
        ("mixed", args.sem_mixed),
        ("queries", bool(args.sem_queries)),
    ) if on]
    if len(chosen) > 1:
        print(f"[eval] ERROR: semantic modes are mutually exclusive: {chosen}")
        sys.exit(2)
    sem_mode = chosen[0] if chosen else "lexical"

    with open(CHUNKS_PATH, "rb") as fh:
        chunks = pickle.load(fh)
    chunk_by_id = {c["id"]: c for c in chunks}
    print(f"[eval] loaded {len(chunks)} chunks")

    index = None
    if sem_mode in ("vectors", "mixed", "queries"):
        import faiss  # noqa: PLC0415
        index = faiss.read_index(INDEX_PATH)  # read-only load; file untouched
        assert index.ntotal == len(chunks), (
            f"index/chunks mismatch: {index.ntotal} vs {len(chunks)}"
        )
        print(f"[eval] loaded REAL faiss index: ntotal={index.ntotal} dim={index.d}")

    sem_fixture = _load_sem_fixture(args.sem_fixture)
    query_vectors: dict[str, dict] | None = None
    if sem_mode == "queries":
        try:
            query_vectors = _load_query_embeddings(args.sem_queries, index.d)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[eval] ERROR loading --sem-queries file: {exc}")
            sys.exit(2)
        print(f"[eval] loaded claim-verified-by-operator query embeddings for "
              f"{len(query_vectors)} question ids (dim={index.d}); provenance "
              f"cannot be verified by this harness")

    print(f"\n===== Retrieval evaluation (sem_mode={sem_mode}) =====")
    if sem_mode == "vectors":
        print("[eval] semantic stage: GENUINE faiss search; query encoding proxied "
              "by the real stored embedding of sem_proxy_id (annotated).")
    elif sem_mode == "mixed":
        print("[eval] semantic stage: GENUINE faiss search; query encoding is a "
              "DIAGNOSTIC MIXED PROXY (normalized mean of annotated evidence "
              "embeddings), NOT a genuine question embedding.")
    elif sem_mode == "queries":
        print("[eval] semantic stage: GENUINE faiss search with operator-supplied "
              "cached query embeddings; provenance is the operator's to verify.")
    header = (
        f"{'id':26s} {'orig':>5s} {'exp':>4s} {'sem':>5s} {'pool':>5s} "
        f"{'cov':>5s} {'ctx':>4s} {'gate':>5s}"
    )
    print(header)
    print("-" * len(header))
    retrieval_rows = run_retrieval_eval(
        chunks, chunk_by_id, index, sem_mode, sem_fixture, query_vectors
    )
    for r in retrieval_rows:
        cov = (
            f"{r['evidence_in_ctx']}/{r['evidence_total']}"
            if r.get("answerable") and not r.get("skipped") else "-"
        )
        if r.get("skipped"):
            print(f"{r['id']:26s} skipped: no honest proxy (unanswerable)")
            continue
        print(
            f"{r['id']:26s} {str(r['orig_rank'] or '-'):>5s} "
            f"{str(r['exp_rank'] or '-'):>4s} {str(r.get('sem_rank') or '-'):>5s} "
            f"{str(r['pool_rank'] or '-'):>5s} {cov:>5s} "
            f"{'YES' if r['in_context'] else 'no':>4s} "
            f"{'pass' if r['gate'] else 'FAIL':>5s}"
        )

    print("\n================ Grounding evaluation ================")
    grounding_rows = run_grounding_eval(chunks)
    g_ok = sum(1 for r in grounding_rows if r["ok"])
    for r in grounding_rows:
        status = "ok  " if r["ok"] else "FAIL"
        extra = f" absent_hits={r['absent_hits']}" if r["absent_hits"] else ""
        note = "" if r["behavior_ok"] else "  (behavior mismatch)"
        print(f"  [{status}] {r['id']:22s} expect={r['expect']:4s} "
              f"kept={str(r['kept']):5s}{note}{extra}")
    keep_rows = [r for r in grounding_rows if r["expect"] == "keep"]
    drop_rows = [r for r in grounding_rows if r["expect"] == "drop"]
    print(f"  grounding: {g_ok}/{len(grounding_rows)} cases behave as annotated")
    print(f"    keep cases accepted: {sum(1 for r in keep_rows if r['kept'])}/{len(keep_rows)}"
          f"  (false rejections: {sum(1 for r in keep_rows if not r['kept'])})")
    print(f"    drop cases rejected: {sum(1 for r in drop_rows if not r['kept'])}/{len(drop_rows)}"
          f"  (unsupported acceptances: {sum(1 for r in drop_rows if r['kept'])})")

    ans_rows = [r for r in retrieval_rows if r["answerable"] and not r.get("skipped")]
    skipped = [r for r in retrieval_rows if r.get("skipped")]
    una_rows = [r for r in retrieval_rows if not r["answerable"] and not r.get("skipped")]
    overall = _agg(ans_rows)
    print(f"\n[summary] sem_mode={sem_mode}")
    if ans_rows:
        print(
            f"  answerable: n={overall['n']}  in-context={overall['in_context']}  "
            f"full-coverage={overall['multi_chunk_full']}  avg_cov={overall['avg_coverage']}  "
            f"pool_MRR={overall['pool_mrr']}  gate_pass={overall['gate_pass']}"
        )
        print(
            "  pool Recall@K: "
            + "  ".join(f"@{k}={overall[f'pool_recall@{k}']}" for k in K_VALUES)
        )
        if sem_mode in ("vectors", "mixed", "queries"):
            print(
                "  sem  Recall@K: "
                + "  ".join(f"@{k}={overall[f'sem_recall@{k}']}" for k in K_VALUES)
                + f"  (sem_MRR={overall['sem_mrr']})"
            )
        for cat, a in _by_category(retrieval_rows).items():
            print(
                f"    [{cat:20s}] n={a['n']} ctx={a['in_context']} full={a['multi_chunk_full']} "
                f"avg_cov={a['avg_coverage']} pool_MRR={a['pool_mrr']} "
                f"pool@5={a['pool_recall@5']}"
                + (f" sem@5={a['sem_recall@5']}"
                   if sem_mode in ("vectors", "mixed", "queries") else "")
            )
    if una_rows:
        refused = sum(1 for r in una_rows if not r["gate"])
        print(f"  unanswerable: gate-fail (would refuse) {refused}/{len(una_rows)}")
    if skipped:
        print(f"  skipped rows: {len(skipped)} "
              f"(no honest query vector available for this mode)")

    if args.json_path:
        out = {
            "sem_mode": sem_mode,
            "proxy_disclosure": (
                "query encoding mocked (no API); sem-vectors uses the real stored "
                "embedding of an annotated corpus chunk as query vector"
                if sem_mode == "vectors" else
                "query encoding mocked (no API); sem-mixed uses the normalized "
                "mean of annotated evidence embeddings (DIAGNOSTIC upper-bound "
                "proxy, not a genuine query embedding)"
                if sem_mode == "mixed" else
                "query vectors read from an operator-supplied file; GENUINE only "
                "if the operator's capture is genuine (provenance not verifiable "
                "by this harness)"
                if sem_mode == "queries" else
                "semantic stage fully stubbed" if sem_mode == "lexical"
                else "semantic ranks injected from fixture file" if sem_mode == "fixture"
                else ""
            ),
            "retrieval": retrieval_rows,
            "grounding": grounding_rows,
            "overall": overall,
            "by_category": _by_category(retrieval_rows),
        }
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"[eval] raw results written to {args.json_path}")


if __name__ == "__main__":
    main()
