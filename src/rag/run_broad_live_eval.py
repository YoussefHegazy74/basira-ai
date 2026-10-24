#!/usr/bin/env python3
"""Instrumented live RAG evaluation.

Uses the production pipeline exactly: P.generate() -> retrieve/build_prompt/
chat.completions/_ground_answer.  The OpenAI client is wrapped only to record
usage, latency and query embeddings, and to enforce a hard spend cap with no
retries.  Results are written only under this directory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import pickle
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "eval_runs" / "broad_pending"
sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import src.rag.pipeline as P  # noqa: E402
from src.nlp.preprocessor import normalize, normalize_verse_text  # noqa: E402
from src.rag.broad_eval_set import BROAD_EVAL_QUESTIONS, BROAD_RUBRICS  # noqa: E402
from src.embeddings.embedder import EMBEDDING_MODEL, get_client  # noqa: E402
from src.vector_db.faiss_store import load_index  # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")
REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال."
CHUNKS_PATH = ROOT / "data/processed/chunks.pkl"
INDEX_PATH = ROOT / "data/processed/faiss.index"

# OpenAI list prices at plan time, USD per token.  Actual usage is recorded from
# the API; if billing differs, the hard cap still stops the run.
RATES = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0.0},
}
ANSWER_MAX_OUTPUT = 16_384  # gpt-4o-mini per-request output ceiling
CONTEXT_LIMIT = 128_000

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# The rubric patterns are written in modern Arabic.  normalize() folds hamza
# alef and ta marbuta, but leaves alef maqsura and the hamza carriers; a
# verbatim Quran quotation would then fail checks such as "إلى المرافق"
# (normalized الى) or "برؤوسكم" (normalized برءوسكم).  Fold those too so the
# evaluator compares orthography, not spelling variants.
_NZ_FOLD = str.maketrans({
    "ى": "ي", "ؤ": "و", "ئ": "ي", "ء": "",
})


def nz(text: str) -> str:
    # Apply the Quran-orthography fold first: Uthmani waw + superscript
    # alef (الربوا/الصلوة) must compare with modern الربا/الصلاة.  This is
    # evaluator normalization only; the stored Quran text is untouched.
    t = normalize(normalize_verse_text(text or "")).translate(_DIGITS)
    # normalize() does not strip a decomposed combining hamza (ا + U+0654)
    # that appears in some Quranic quotations; NFKD + combining-mark removal
    # makes those letters comparable to their precomposed forms.
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # Uthmani tanween is written as a standalone alef after the word
    # (قَوۡلࣰ ا).  After marks are removed it becomes "قول ا"; attach that
    # alef so rubric terms such as "قولا كريما" compare as written Arabic.
    t = re.sub(r"(?<=\S)\s+ا(?=\s|$)", "ا", t)
    # Canonicalise spelled numbers so case/orthography variants compare:
    # ثلاثة وسبعون / ثلاثة وسبعين -> 73, ثلاث وستون -> 63.  Arabic number
    # words inflect for case, and a rubric written in the nominative must
    # not reject a correct genitive answer.
    t = " ".join(P._numbers_normalized(t.split()))
    return t.translate(_NZ_FOLD)


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = RATES.get(model, {"input": 0.0, "output": 0.0})
    return prompt_tokens * rate["input"] + completion_tokens * rate["output"]


def pct(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


# ---------------------------------------------------------------------------
# Deterministic answer checks (rubrics live in broad_eval_set.py)
# ---------------------------------------------------------------------------

def check_fact(answer_norm: str, chk: dict) -> dict:
    # "من/في شهر رمضان" is a valid paraphrase of "من/في رمضان"; the
    # optional month word must not turn a correct date into a metric failure.
    answer_norm = re.sub(r"\b(من|في) شهر ", r"\1 ", answer_norm)
    kind = chk.get("kind", "terms")
    status = "fail"
    if kind == "ambiguous":
        topics = {
            "رواتب": ["راتبه", "رواتب", "فجر", "ظهر", "عصر", "مغرب", "عشاء"],
            "وتر": ["وتر"],
            "جماعة": ["جماعه"],
            "كسوف": ["كسوف"],
            "عقيقة": ["عقيقه"],
        }
        found = sorted({name for name, pats in topics.items()
                        if any(nz(p) in answer_norm for p in pats)})
        qualified = any(nz(x) in answer_norm for x in
                        ["توضيح", "أي باب", "اي باب", "تختلف", "يشمل اكثر من باب",
                         "اكثر من معنى", "مصطلح"])
        status = "manual"
        return {"name": chk["name"], "optional": chk["optional"],
                "status": status, "passed": None,
                "topics": found, "qualified": qualified}
    if kind == "aqiqa_days":
        days = [
            ["السابع", "سبع", "7"],
            ["الرابع عشر", "اربع عشر", "14"],
            ["الحادي والعشرين", "الحادي والعشرون",
             "الواحد والعشرين", "الواحد والعشرون",
             "واحد وعشرين", "21"],
        ]
        found = [any(nz(p) in answer_norm for p in group) for group in days]
        passed = all(found)
        return {"name": chk["name"], "optional": chk["optional"],
                "status": "pass" if passed else "fail", "passed": passed,
                "days": found}
    if chk.get("all_of"):
        passed = all(nz(p) in answer_norm for p in chk["all_of"])
    else:
        passed = any(nz(p) in answer_norm for p in chk.get("any_of", []))
    return {"name": chk["name"], "optional": chk["optional"],
            "status": "pass" if passed else "fail", "passed": passed}


# ---------------------------------------------------------------------------
# Budget / recording client
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class Recorder:
    def __init__(self, cap: float, stop_fraction: float):
        self.cap = cap
        self.stop_at = cap * stop_fraction
        self.spent = 0.0
        self.calls: list[dict] = []
        self.embeddings: dict[str, list[dict]] = {}
        self.current_qid = "prepare"
        self.fatal: str | None = None
        self.blocked: str | None = None

    def precheck(self, worst_case: float, label: str) -> None:
        if self.spent + worst_case > self.stop_at:
            self.blocked = (
                f"budget stop before {label}: spent=${self.spent:.6f}, "
                f"next_worst_case=${worst_case:.6f}, stop_at=${self.stop_at:.2f}"
            )
            raise BudgetExceeded(self.blocked)

    def commit(self, **rec) -> None:
        self.spent += rec["cost_usd"]
        rec["cumulative_usd"] = self.spent
        self.calls.append(rec)
        with open(RUN_DIR / "api_calls.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def failure(self, kind: str, model: str, exc: Exception,
                seconds: float, ambiguous: bool = True) -> None:
        self.fatal = f"{kind} API failure: {type(exc).__name__}: {exc}"
        rec = {
            "qid": self.current_qid, "kind": kind, "model": model,
            "error": self.fatal, "ambiguous_charge": ambiguous,
            "latency_s": round(seconds, 4), "cost_usd": 0.0,
        }
        self.calls.append(rec)
        with open(RUN_DIR / "api_calls.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def capture_embedding(self, texts: list[str], vectors: list[list[float]],
                          model: str) -> None:
        self.embeddings.setdefault(self.current_qid, []).append({
            "texts": texts, "model": model, "vectors": vectors,
        })


class _EmbeddingsProxy:
    def __init__(self, real, rec: Recorder):
        self._real = real
        self._rec = rec

    def create(self, **kwargs):
        model = kwargs.get("model", EMBEDDING_MODEL)
        raw = kwargs.get("input", "")
        texts = [raw] if isinstance(raw, str) else [str(x) for x in raw]
        est = sum(len(ENC.encode(t)) for t in texts)
        self._rec.precheck(est * RATES.get(model, {}).get("input", 0.0),
                           "embedding")
        t0 = time.perf_counter()
        try:
            resp = self._real.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._rec.failure("embedding", model, exc, time.perf_counter() - t0)
            raise
        elapsed = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        prompt_tokens = (getattr(usage, "prompt_tokens", None)
                         or getattr(usage, "total_tokens", None) or est)
        total_tokens = getattr(usage, "total_tokens", None) or prompt_tokens
        cost = cost_of(model, prompt_tokens, 0)
        vectors = [list(item.embedding) for item in resp.data]
        self._rec.capture_embedding(texts, vectors, model)
        self._rec.commit(
            qid=self._rec.current_qid, kind="embedding", model=model,
            response_model=getattr(resp, "model", None),
            prompt_tokens=prompt_tokens, completion_tokens=0,
            total_tokens=total_tokens, cost_usd=cost,
            latency_s=round(elapsed, 4), n_inputs=len(texts),
        )
        return resp


class _ChatCompletionsProxy:
    def __init__(self, real, rec: Recorder):
        self._real = real
        self._rec = rec

    def create(self, **kwargs):
        model = kwargs.get("model", P.GENERATION_MODEL)
        messages = kwargs.get("messages", [])
        max_tokens = kwargs.get("max_tokens")
        kind = "rewrite" if max_tokens is not None else "answer"
        est_in = sum(len(ENC.encode(m.get("content", ""))) + 4
                     for m in messages) + 2
        out_cap = int(max_tokens) if max_tokens is not None else ANSWER_MAX_OUTPUT
        self._rec.precheck(cost_of(model, est_in, out_cap), kind)
        t0 = time.perf_counter()
        try:
            resp = self._real.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._rec.failure(kind, model, exc, time.perf_counter() - t0)
            raise
        elapsed = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) or est_in
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None:
            content = ""
            if getattr(resp, "choices", None):
                content = resp.choices[0].message.content or ""
            completion_tokens = len(ENC.encode(content))
        total_tokens = getattr(usage, "total_tokens", None) or (
            prompt_tokens + completion_tokens)
        cost = cost_of(model, prompt_tokens, completion_tokens)
        finish = None
        if getattr(resp, "choices", None):
            finish = getattr(resp.choices[0], "finish_reason", None)
        self._rec.commit(
            qid=self._rec.current_qid, kind=kind, model=model,
            response_model=getattr(resp, "model", None),
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total_tokens, cost_usd=cost,
            latency_s=round(elapsed, 4), finish_reason=finish,
        )
        return resp


class RecordingClient:
    def __init__(self, real, rec: Recorder):
        self.embeddings = _EmbeddingsProxy(real.embeddings, rec)
        self.chat = SimpleNamespace(
            completions=_ChatCompletionsProxy(real.chat.completions, rec))


# ---------------------------------------------------------------------------
# Questions / provenance
# ---------------------------------------------------------------------------

def _holdout_entries() -> list[tuple[str, str]]:
    entries = []
    path = ROOT / "src/rag/holdout.txt"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or " | " not in line:
            continue
        q, phrase = line.split(" | ", 1)
        entries.append((q.strip(), phrase.strip()))
    return entries


def build_questions(chunks: list[dict]) -> list[dict]:
    byid = {c["id"]: c for c in chunks}
    out = []
    for q in BROAD_EVAL_QUESTIONS:
        item = dict(q)
        item["set"] = "broad"
        item["checks"] = BROAD_RUBRICS.get(q["id"], [])
        item["source_types"] = sorted({
            byid[e].get("source", "?") for e in q["evidence_ids"] if e in byid
        })
        out.append(item)
    return out


def prepare_questions(chunks: list[dict]) -> list[dict]:
    # Always rebuild from BROAD_EVAL_QUESTIONS + BROAD_RUBRICS.  A cached
    # questions.json silently froze earlier rubric patterns, so checker fixes
    # appeared in the rubric module but never reached the live run.
    path = RUN_DIR / "questions.json"
    questions = build_questions(chunks)
    path.write_text(json.dumps(questions, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return questions


# ---------------------------------------------------------------------------
# Instrumentation of the production path
# ---------------------------------------------------------------------------

def install_instrumentation(state: SimpleNamespace, raw_retrieve, raw_rrf,
                           raw_search, raw_prompt, raw_ground) -> None:
    def retrieve_wrap(*args, **kwargs):
        t0 = time.perf_counter()
        result = raw_retrieve(*args, **kwargs)
        state.retrieved = result
        state.retrieve_s = time.perf_counter() - t0
        return result

    def rrf_wrap(*args, **kwargs):
        result = raw_rrf(*args, **kwargs)
        state.fused_pool = result
        return result

    def search_wrap(*args, **kwargs):
        result = raw_search(*args, **kwargs)
        state.search_calls.append(result)
        return result

    def prompt_wrap(*args, **kwargs):
        result = raw_prompt(*args, **kwargs)
        state.messages = result
        return result

    def ground_wrap(*args, **kwargs):
        answer = args[0] if args else kwargs.get("answer", "")
        state.raw_answer = answer
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = raw_ground(*args, **kwargs)
        state.ground_log = buf.getvalue()
        state.grounded_answer = result
        return result

    P.retrieve = retrieve_wrap
    P._rrf_fuse = rrf_wrap
    P.search = search_wrap
    P.build_prompt = prompt_wrap
    P._ground_answer = ground_wrap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _first_rank(ids: list[str], gold: set[str]) -> int | None:
    for i, cid in enumerate(ids, 1):
        if cid in gold:
            return i
    return None


def _ids(items: list[dict]) -> list[str]:
    return [c.get("id", "") for c in items]


def _semantic_union(search_calls: list[list[dict]]) -> list[dict]:
    if len(search_calls) < 2:
        return []
    best: dict[str, dict] = {}
    for results in search_calls[:2]:
        for item in results:
            cid = item.get("id")
            if cid and (cid not in best or
                        item.get("score", 0) > best[cid].get("score", 0)):
                best[cid] = item
    return sorted(best.values(), key=lambda c: c.get("score", 0), reverse=True)


def retrieval_metrics(q: dict, state: SimpleNamespace) -> dict:
    gold = set(q.get("evidence_ids", []))
    retrieved = state.retrieved or []
    ctx_ids = _ids(retrieved)
    pool = state.fused_pool or []
    pool_ids = _ids(pool)
    sem = _semantic_union(state.search_calls)
    sem_ids = _ids(sem)

    def rec(ids: list[str], k: int) -> bool | None:
        if not gold:
            return None
        return bool(gold & set(ids[:k]))

    ctx_rank = _first_rank(ctx_ids, gold) if gold else None
    pool_rank = _first_rank(pool_ids, gold) if gold else None
    sem_rank = _first_rank(sem_ids, gold) if gold else None
    return {
        "gate_passed": bool(retrieved and retrieved[0].get("_gate_passed")),
        "context_ids": ctx_ids,
        "context_rank": ctx_rank,
        "context_rr": (1.0 / ctx_rank) if ctx_rank else 0.0,
        "context_coverage": (len(gold & set(ctx_ids)) / len(gold)) if gold else None,
        "context_full": (gold <= set(ctx_ids)) if gold else None,
        "in_context": (bool(gold & set(ctx_ids)) if gold else None),
        "pool_rank": pool_rank,
        "pool_rr": (1.0 / pool_rank) if pool_rank else 0.0,
        "pool_top20": [
            {"id": c.get("id"), "score": round(c.get("score", 0.0), 6)}
            for c in pool[:20]
        ],
        "sem_rank": sem_rank,
        "sem_rr": (1.0 / sem_rank) if sem_rank else 0.0,
        "sem_top20": [
            {"id": c.get("id"), "score": round(c.get("score", 0.0), 6)}
            for c in sem[:20]
        ],
        "pool_recall": {f"@{k}": rec(pool_ids, k) for k in (1, 3, 5, 10)},
        "sem_recall": {f"@{k}": rec(sem_ids, k) for k in (1, 3, 5, 10)},
        "context_recall": {f"@{k}": rec(ctx_ids, k) for k in (1, 3, 5)},
    }


def _content_tokens(text: str) -> list[str]:
    body = P._CITATION_RE.sub(" ", text or "")
    body = P._QURAN_REF_RE.sub(" ", body)
    # Bold labels are presentational headings, not claims.  The production
    # grounding engine strips them before checking; the support metric must
    # do the same or a label like "**غسل الوجه**" is counted as an
    # unsupported source token.
    body = re.sub(r"\*\*[^*]+\*\*", " ", body)
    toks = P._numbers_normalized(P._tokenize(body))
    return [
        t for t in toks
        if t not in P._QUERY_STOPWORDS
        and t not in P._GROUNDING_IGNORED_TOKENS
        and (len(t) >= 3 or t.isdigit())
    ]


def _support_ratio(text: str, chunk: dict) -> float:
    toks = _content_tokens(text)
    if not toks:
        return 1.0
    ref = {P._gkey(t) for t in P._numbers_normalized(
        P._tokenize(chunk.get("original_text", "") or ""))}
    return sum(1 for t in toks if P._gkey(t) in ref) / len(toks)


def _sentence_support(sentence: str, retrieved: list[dict],
                      cited_indices: list[int]) -> float:
    if cited_indices:
        chunks = [retrieved[i] for i in cited_indices
                  if 0 <= i < len(retrieved)]
    else:
        chunks = list(retrieved)
    if not chunks:
        return 0.0
    return max(_support_ratio(sentence, c) for c in chunks)


def answer_metrics(q: dict, state: SimpleNamespace, result: dict,
                   byid: dict[str, dict]) -> dict:
    answer = result.get("answer", "") or ""
    answer_norm = nz(answer)
    refusal = answer_norm.strip() == nz(REFUSAL)
    retrieved = state.retrieved or []

    facts = [check_fact(answer_norm, c) for c in q.get("checks", [])]
    mandatory = [f for f in facts
                 if not f.get("optional") and f.get("status") != "manual"]
    fact_pass = all(f["passed"] for f in mandatory) if mandatory else None

    citations = []
    for m in P._CITATION_RE.finditer(answer):
        idx = int(m.group(1))
        valid = 1 <= idx <= len(retrieved)
        chunk = retrieved[idx - 1] if valid else None
        citations.append({
            "index": idx,
            "valid": valid,
            "chunk_id": chunk.get("id") if chunk else None,
            "source": chunk.get("source") if chunk else None,
            "gold": bool(chunk and chunk.get("id") in set(q.get("evidence_ids", []))),
        })
    n_cit = len(citations)
    n_valid = sum(1 for c in citations if c["valid"])
    n_gold = sum(1 for c in citations if c["valid"] and c["gold"])
    gold = set(q.get("evidence_ids", []))

    raw_sents = [s.strip() for s in P._SENT_SPLIT_RE.split(state.raw_answer or "")
                 if s.strip()]
    grounded_sents = [s.strip() for s in P._SENT_SPLIT_RE.split(
        state.grounded_answer or "") if s.strip()]
    kept_sents = [s.strip() for s in P._SENT_SPLIT_RE.split(answer) if s.strip()]
    # Measure actual grounding attrition, not log lines: the rare-token
    # fallback can log the same sentence twice, and post-grounding
    # completeness repair can add source-backed sentences.
    dropped = max(0, len(raw_sents) - len(grounded_sents))

    unsupported = []
    if not refusal:
        for sent in kept_sents:
            idxs = [int(m.group(1)) - 1 for m in P._CITATION_RE.finditer(sent)]
            support = _sentence_support(sent, retrieved, idxs)
            if support < 0.6:
                unsupported.append({
                    "sentence": sent[:220],
                    "support": round(support, 3),
                    "cited": [i + 1 for i in idxs],
                })

    phrase_supported = None
    if q.get("set") == "heldout":
        phrase = nz(q.get("phrase", ""))
        phrase_supported = any(
            c["valid"] and phrase and phrase in nz(
                byid[c["chunk_id"]].get("original_text", ""))
            for c in citations if c["chunk_id"] in byid
        )

    manual_fact = any(f.get("status") == "manual" for f in facts)
    source_citation_pass = (
        bool(not refusal and phrase_supported and not unsupported)
        if q.get("set") == "heldout" else None
    )
    if q.get("answerable") is False:
        answer_pass = refusal
    elif q.get("set") == "heldout":
        # A heldout phrase is a retrieval/source-citation check only.  It is
        # not evidence that the generated answer is correct or complete, so
        # do not let it enter answer-accuracy denominators.
        answer_pass = None
    elif manual_fact:
        # Literal rubric checks cannot grade an intentionally ambiguous
        # question; require explicit manual review instead of a false pass.
        answer_pass = None
    else:
        # A primary answer passes only when its mandatory facts pass, it has
        # at least one citation, and every kept sentence has source support
        # at/above the grounding support threshold.
        answer_pass = bool(
            not refusal and fact_pass is not False and n_cit > 0
            and n_valid == n_cit and not unsupported
        )

    return {
        "refusal": refusal,
        "expected_refusal": not q.get("answerable", True),
        "abstention_ok": refusal == (not q.get("answerable", True)),
        "answer_pass": answer_pass,
        "fact_pass": fact_pass,
        "facts": facts,
        "manual_review": manual_fact,
        "phrase_supported_by_cited_source": phrase_supported,
        "source_citation_pass": source_citation_pass,
        "citation_count": n_cit,
        "citation_validity": (n_valid / n_cit) if n_cit else None,
        "citation_gold_precision": (n_gold / n_cit) if n_cit else None,
        "citation_gold_recall": (
            len({c["chunk_id"] for c in citations if c["valid"] and c["gold"]}
                & gold) / len(gold)
        ) if gold else None,
        "citation_support_rate": (
            (len(kept_sents) - len(unsupported)) / len(kept_sents)
        ) if kept_sents else None,
        "citations": citations,
        "raw_sentences": len(raw_sents),
        "grounded_sentences": len(grounded_sents),
        "kept_sentences": len(kept_sents),
        "postprocess_added_sentences": max(
            0, len(kept_sents) - len(grounded_sents)),
        "dropped_sentences": dropped,
        "ground_retention": (
            min(1.0, len(grounded_sents) / len(raw_sents))
            if raw_sents else None
        ),
        "unsupported_final_sentences": unsupported,
        "unsupported_final_count": len(unsupported),
        "sources": result.get("sources", []),
    }


# ---------------------------------------------------------------------------
# Run / report
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["question"]["answerable"]]
    unanswerable = [r for r in rows if not r["question"]["answerable"]]
    facts = [r["answer"]["fact_pass"] for r in answerable
             if r["answer"]["fact_pass"] is not None]
    answerable_scored = [
        r for r in answerable if r["answer"]["answer_pass"] is not None
    ]
    scored = [r for r in rows if r["answer"]["answer_pass"] is not None]
    heldout = [r for r in rows if r["question"].get("set") == "heldout"]
    heldout_source = [
        r for r in heldout
        if r["answer"].get("source_citation_pass") is not None
    ]
    manual = [r["question"]["id"] for r in rows
              if r["answer"].get("manual_review")]

    def rate(items: list, section: str, key: str) -> float | None:
        vals = [r for r in items if r[section].get(key) is not None]
        return round(
            sum(bool(r[section][key]) for r in vals) / len(vals), 4
        ) if vals else None

    recalls = {}
    for stage in ("pool_recall", "sem_recall", "context_recall"):
        for k in (rows[0]["retrieval"][stage].keys() if rows else []):
            vals = [r["retrieval"][stage][k] for r in rows
                    if r["retrieval"][stage].get(k) is not None]
            recalls[f"{stage}_{k}"] = (
                round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None
            )

    return {
        "n": len(rows),
        "answerable": len(answerable),
        "answerable_scored": len(answerable_scored),
        "unanswerable": len(unanswerable),
        "answer_pass_count": sum(
            bool(r["answer"]["answer_pass"]) for r in answerable_scored),
        "answer_pass_rate": rate(answerable, "answer", "answer_pass"),
        "task_pass_count": sum(bool(r["answer"]["answer_pass"]) for r in scored),
        "task_scored": len(scored),
        "task_pass_rate": rate(rows, "answer", "answer_pass"),
        "heldout_source_citation_count": sum(
            bool(r["answer"]["source_citation_pass"])
            for r in heldout_source),
        "heldout_source_citation_rate": rate(
            heldout, "answer", "source_citation_pass"),
        "manual_review": manual,
        "abstention_ok_rate": mean([1.0 if r["answer"]["abstention_ok"] else 0.0
                                    for r in rows]),
        "fact_pass_rate": (round(sum(bool(x) for x in facts) / len(facts), 4)
                           if facts else None),
        "false_refusal_rate": rate(answerable, "answer", "refusal"),
        "unanswerable_refusal_rate": rate(
            unanswerable, "answer", "refusal"),
        "in_context_rate": rate(answerable, "retrieval", "in_context"),
        "context_full_rate": rate(answerable, "retrieval", "context_full"),
        "mean_context_coverage": mean([r["retrieval"]["context_coverage"]
                                       for r in answerable]),
        "pool_mrr": mean([r["retrieval"]["pool_rr"] for r in answerable]),
        "sem_mrr": mean([r["retrieval"]["sem_rr"] for r in answerable]),
        **recalls,
        "citation_validity": mean([r["answer"]["citation_validity"] for r in rows]),
        "citation_support_rate": mean([r["answer"].get("citation_support_rate")
                                       for r in rows]),
        "citation_gold_precision": mean([r["answer"]["citation_gold_precision"]
                                         for r in rows]),
        "citation_gold_recall": mean([r["answer"]["citation_gold_recall"]
                                      for r in rows]),
        "unsupported_final_sentences": sum(
            r["answer"]["unsupported_final_count"] for r in rows),
        "dropped_sentences": sum(r["answer"]["dropped_sentences"] for r in rows),
        "ground_retention": mean([r["answer"]["ground_retention"] for r in rows]),
        "retrieve_latency_p50": pct([r["timing"]["retrieve_s"] for r in rows], 0.5),
        "retrieve_latency_p95": pct([r["timing"]["retrieve_s"] for r in rows], 0.95),
        "generate_latency_p50": pct([r["timing"]["total_s"] for r in rows], 0.5),
        "generate_latency_p95": pct([r["timing"]["total_s"] for r in rows], 0.95),
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
        "embedding_tokens": sum(r["usage"]["embedding_tokens"] for r in rows),
        "cost_usd": round(sum(r["usage"]["cost_usd"] for r in rows), 6),
    }


def _failure_reasons(q: dict, r: dict) -> list[str]:
    if r.get("error"):
        return [f"api/runtime error: {r['error']}"]
    a = r.get("answer", {})
    reasons: list[str] = []
    if q.get("answerable") is False:
        if not a.get("abstention_ok"):
            reasons.append("expected refusal but answer was produced")
        return reasons
    if a.get("refusal"):
        reasons.append("false refusal on answerable question")
    if a.get("fact_pass") is False:
        failed = [f.get("name") for f in a.get("facts", [])
                  if f.get("status") == "fail" and not f.get("optional")]
        reasons.append("failed facts: " + ", ".join(failed))
    if a.get("citation_count", 0) == 0:
        reasons.append("no citations")
    if a.get("citation_validity") is not None and a["citation_validity"] < 1.0:
        reasons.append("invalid citation indices")
    if a.get("unsupported_final_count"):
        reasons.append(f"unsupported final sentences: {a['unsupported_final_count']}")
    if q.get("set") == "heldout":
        if not a.get("phrase_supported_by_cited_source"):
            reasons.append("heldout phrase not supported by cited source")
    rtr = r.get("retrieval", {})
    if not rtr.get("gate_passed"):
        reasons.append("retrieval gate failed")
    elif q.get("evidence_ids") and rtr.get("in_context") is False:
        reasons.append("no gold evidence in context")
    if not reasons and a.get("answer_pass") is False:
        reasons.append("answer check failed")
    return reasons


def write_report(questions: list[dict], rows: list[dict], rec: Recorder,
                 started: float, args) -> None:
    sets = {
        "all": rows,
        "primary": [r for r in rows if r["question"]["set"] == "primary"],
        "heldout": [r for r in rows if r["question"]["set"] == "heldout"],
    }
    slices = {}
    for source in ("quran", "hadith", "tafsir", "fiqh", "sira"):
        selected = [r for r in rows if source in r["question"].get("source_types", [])]
        slices[source] = aggregate(selected) if selected else {"n": 0}
    slices["unanswerable"] = aggregate(
        [r for r in rows if not r["question"]["answerable"]])

    observed_generation = sorted({
        c.get("response_model") for c in rec.calls
        if c.get("kind") in ("answer", "rewrite") and c.get("response_model")
    })
    observed_embedding = sorted({
        c.get("response_model") for c in rec.calls
        if c.get("kind") == "embedding" and c.get("response_model")
    })
    summary = {
        "run_dir": str(RUN_DIR),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_s": round(time.time() - started, 3),
        "questions": len(questions),
        "executed": len(rows),
        "model": {
            "requested_generation_model": P.GENERATION_MODEL,
            "observed_generation_models": observed_generation,
            "embedding_model": EMBEDDING_MODEL,
            "observed_embedding_models": observed_embedding,
        },
        "pricing_usd_per_million_tokens": {
            name: {"input": round(rate["input"] * 1_000_000, 6),
                   "output": round(rate["output"] * 1_000_000, 6)}
            for name, rate in RATES.items()
        },
        "cost_basis": "estimated from recorded API token usage and configured list prices; not an invoice",
        "api": {
            "cap_usd": rec.cap,
            "stop_at_usd": rec.stop_at,
            "spent_usd": round(rec.spent, 6),
            "calls": len(rec.calls),
            "fatal": rec.fatal,
            "blocked": rec.blocked,
        },
        "aggregate": {name: aggregate(group) for name, group in sets.items()},
        "per_source": slices,
        "failures": [
            {
                "id": r["question"]["id"],
                "set": r["question"]["set"],
                "question": r["question"]["question"],
                "answer_pass": r["answer"]["answer_pass"],
                "source_citation_pass": r["answer"].get("source_citation_pass"),
                "fact_pass": r["answer"]["fact_pass"],
                "abstention_ok": r["answer"]["abstention_ok"],
                "in_context": r["retrieval"]["in_context"],
                "gate": r["retrieval"]["gate_passed"],
                "reasons": r.get("failure_reasons", []),
                "answer": r.get("final_answer", ""),
                "error": r.get("error"),
            }
            for r in rows
            if r.get("failure_reasons")
        ],
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN_DIR / "query_embeddings.json").write_text(
        json.dumps(rec.embeddings, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Basira live RAG evaluation",
        "",
        f"- Run: `{RUN_DIR}`",
        f"- Questions: {summary['questions']} (executed {summary['executed']})",
        f"- API calls: {summary['api']['calls']}; spent: "
        f"${summary['api']['spent_usd']:.6f} of ${summary['api']['cap_usd']:.2f}",
        f"- Fatal: {summary['api']['fatal'] or 'none'}",
        "",
        "## Aggregate",
        "",
    ]
    for name in ("all", "primary", "heldout"):
        a = summary["aggregate"][name]
        lines.append(
            f"- **{name}**: n={a['n']}, answer_pass={a['answer_pass_rate']} "
            f"({a['answer_pass_count']}/{a['answerable_scored']} scored answerable), "
            f"task_pass={a['task_pass_rate']} "
            f"({a['task_pass_count']}/{a['task_scored']}), "
            f"heldout_source_citation={a['heldout_source_citation_rate']} "
            f"({a['heldout_source_citation_count']}), "
            f"abstention={a['abstention_ok_rate']}, fact={a['fact_pass_rate']}, "
            f"in_context={a['in_context_rate']}, coverage={a['mean_context_coverage']}, "
            f"citation_validity={a['citation_validity']}, "
            f"citation_support={a['citation_support_rate']}, "
            f"citation_gold_precision={a['citation_gold_precision']}, "
            f"unsupported_final={a['unsupported_final_sentences']}, "
            f"manual={a['manual_review']}, "
            f"cost=${a['cost_usd']:.6f}"
        )
    lines += ["", "## Per source", ""]
    for source, a in slices.items():
        if not a.get("n"):
            continue
        lines.append(
            f"- **{source}**: n={a['n']}, answer_pass={a['answer_pass_rate']}, "
            f"in_context={a['in_context_rate']}, coverage={a['mean_context_coverage']}, "
            f"citation_gold_precision={a['citation_gold_precision']}, "
            f"unsupported_final={a['unsupported_final_sentences']}"
        )
    lines += ["", "## Failures", ""]
    if summary["failures"]:
        for f in summary["failures"]:
            lines.append(
                f"- `{f['id']}` pass={f['answer_pass']} fact={f['fact_pass']} "
                f"abstain={f['abstention_ok']} gate={f['gate']} "
                f"in_ctx={f['in_context']} reasons={f['reasons']}: "
                f"{f['answer'][:300]}"
            )
    else:
        lines.append("- none")
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


def reset_run_outputs(run_dir: Path = RUN_DIR) -> None:
    """Clear per-execution outputs so reruns cannot silently append."""
    for name in ("results.jsonl", "api_calls.jsonl"):
        path = run_dir / name
        if path.exists():
            path.write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--ids", default="", help="comma-separated question IDs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget", type=float, default=1.00)
    parser.add_argument("--stop-fraction", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--append", action="store_true",
        help="append to existing results/api_calls instead of starting clean",
    )
    parser.add_argument(
        "--run-dir", default="",
        help="output directory (default: eval_runs/broad_<UTC timestamp>)",
    )
    args = parser.parse_args()

    global RUN_DIR
    if args.run_dir:
        RUN_DIR = Path(args.run_dir)
    else:
        RUN_DIR = ROOT / "eval_runs" / (
            "broad_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    log = open(RUN_DIR / "run.log", "a", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
            return len(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log)

    index, chunks = load_index(str(INDEX_PATH), str(CHUNKS_PATH))
    byid = {c["id"]: c for c in chunks}
    questions = prepare_questions(chunks)
    print(f"[live] prepared {len(questions)} questions")
    if args.prepare_only:
        print("[live] prepare-only: no API calls")
        return 0

    wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
    if wanted:
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        print("[live] no questions selected")
        return 2
    if not args.append:
        reset_run_outputs()
        print("[live] resetting results.jsonl and api_calls.jsonl")

    with open(RUN_DIR / "hashes_before.txt", "w", encoding="utf-8") as fh:
        fh.write(f"chunks.pkl {hash_file(CHUNKS_PATH)}\n")
        fh.write(f"faiss.index {hash_file(INDEX_PATH)}\n")

    load_dotenv(ROOT / ".env")
    real = get_client().with_options(max_retries=0, timeout=120.0)
    rec = Recorder(args.budget, args.stop_fraction)
    client = RecordingClient(real, rec)
    state = SimpleNamespace(
        retrieved=[], fused_pool=[], search_calls=[], messages=[],
        raw_answer="", grounded_answer="", ground_log="", retrieve_s=0.0,
    )
    install_instrumentation(
        state, P.retrieve, P._rrf_fuse, P.search, P.build_prompt,
        P._ground_answer,
    )

    rows = []
    stop = False
    for i, q in enumerate(questions, 1):
        rec.current_qid = q["id"]
        state.retrieved = []
        state.fused_pool = []
        state.search_calls = []
        state.messages = []
        state.raw_answer = ""
        state.grounded_answer = ""
        state.ground_log = ""
        state.retrieve_s = 0.0
        print(f"\n[live {i}/{len(questions)}] {q['id']} | {q['question']}")
        t0 = time.perf_counter()
        error = None
        result = {"answer": "", "sources": [], "query": q["question"]}
        try:
            result = P.generate(
                q["question"], index, chunks, client, top_k=args.top_k)
            if rec.fatal:
                raise RuntimeError(rec.fatal)
        except BudgetExceeded as exc:
            error = str(exc)
            print(f"[live] STOP: {error}")
            stop = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            print(f"[live] STOP: {error}")
            stop = True
        elapsed = time.perf_counter() - t0

        call_slice = [c for c in rec.calls if c.get("qid") == q["id"]]
        usage = {
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in call_slice
                                 if c.get("kind") in ("answer", "rewrite")),
            "completion_tokens": sum(c.get("completion_tokens", 0)
                                     for c in call_slice),
            "embedding_tokens": sum(c.get("prompt_tokens", 0) for c in call_slice
                                    if c.get("kind") == "embedding"),
            "cost_usd": sum(c.get("cost_usd", 0.0) for c in call_slice),
        }
        r = {
            "question": q,
            "retrieval": retrieval_metrics(q, state),
            "answer": answer_metrics(q, state, result, byid) if not error else {
                "answer_pass": False, "abstention_ok": False, "fact_pass": None,
                "refusal": False, "unsupported_final_count": 0,
                "dropped_sentences": 0, "ground_retention": None,
                "citation_validity": None, "citation_gold_precision": None,
                "citation_gold_recall": None,
            },
            "raw_answer": state.raw_answer,
            "ground_log": state.ground_log,
            "final_answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "timing": {"retrieve_s": round(state.retrieve_s, 4),
                       "total_s": round(elapsed, 4)},
            "usage": usage,
            "error": error,
        }
        r["failure_reasons"] = _failure_reasons(q, r)
        rows.append(r)
        with open(RUN_DIR / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if error:
            break

    write_report(questions, rows, rec, started, args)
    with open(RUN_DIR / "hashes_after.txt", "w", encoding="utf-8") as fh:
        fh.write(f"chunks.pkl {hash_file(CHUNKS_PATH)}\n")
        fh.write(f"faiss.index {hash_file(INDEX_PATH)}\n")
    print(f"\n[live] done: {len(rows)} questions, ${rec.spent:.6f} spent")
    return 0 if not stop else 1


def hash_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
