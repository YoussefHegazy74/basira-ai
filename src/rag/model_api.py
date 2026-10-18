import os
import pickle
import sys
from contextlib import asynccontextmanager
from typing import Literal

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(RAG_DIR))
for path in (RAG_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from openai import OpenAI
from pydantic import BaseModel

from verify_full import verify_full
from tafsir import get_tafsir
from src.rag.pipeline import generate, prewarm_bm25


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ---------------------------------------------------------------------------
# RAG-03 R2 — startup initialization and readiness
#
# The FAISS index, the canonical corpus, and the model client are loaded once
# during ASGI lifespan startup (not at import time), and the BM25 index is
# then prewarmed on that exact corpus object so the first chat request is a
# cache hit instead of paying the cold build.  Uvicorn does not accept
# connections until this startup completes.
# ---------------------------------------------------------------------------

_INITIALIZING = "initializing"
_READY = "ready"
_FAILED = "failed"
_STOPPED = "stopped"

_service_phase = _INITIALIZING

# Loaded by _initialize_resources() during lifespan startup; the endpoint
# handlers below use these same long-lived objects for every request.
index = None
chunks = None
client = None


def service_phase() -> str:
    """Current service lifecycle phase (no internal details exposed)."""
    return _service_phase


def _initialize_resources() -> None:
    """Load the FAISS index, the canonical corpus, and the model client.

    Kept out of module import so importing this module (e.g. from offline
    tests) does not require the production data files.
    """
    global index, chunks, client
    index = faiss.read_index(os.path.join(DATA_DIR, "faiss.index"))
    with open(os.path.join(DATA_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    client = OpenAI()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Fail-fast startup: initialize resources, prewarm BM25, then serve.

    A prewarm or initialization failure is logged and re-raised, so uvicorn
    aborts startup instead of serving requests that could not be answered
    without paying the full cold build.  `_service_phase` records the failure
    and can never report ready afterwards.
    """
    global _service_phase
    try:
        _initialize_resources()
        prewarm_bm25(chunks)
    except Exception:
        _service_phase = _FAILED
        print("[startup] initialization failed — refusing to serve",
              file=sys.stderr)
        raise
    _service_phase = _READY
    print(f"[startup] corpus loaded (N={len(chunks)}) and BM25 index prewarmed")
    try:
        yield
    finally:
        _service_phase = _STOPPED


app = FastAPI(
    title="Basira AI API",
    lifespan=_lifespan,
)


class VerifyRequest(BaseModel):
    text: str
    explanation: str | None = None


class TafsirRequest(BaseModel):
    ayah: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None


def history_as_messages(chat_history: list[ChatMessage] | None) -> list[dict] | None:
    if chat_history is None:
        return None
    return [{"role": m.role, "content": m.content} for m in chat_history]


@app.get("/health")
def health():
    """Liveness: the process is up and serving."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    """Readiness: corpus loaded and BM25 index prewarmed.

    Never reports ready before required initialization completes, and never
    reports ready again after a failure or shutdown.  No internal details
    (errors, paths, credentials) are exposed.
    """
    if _service_phase != _READY:
        response.status_code = 503
    return {"status": _service_phase}


@app.post("/api/verify")
def verify(request: VerifyRequest):
    return verify_full(
        request.text,
        request.explanation,
        index,
        chunks,
        client,
    )


@app.post("/api/tafsir")
def tafsir(request: TafsirRequest):
    return get_tafsir(
        request.ayah,
        index,
        chunks,
        client,
    )


@app.post("/api/chat")
def chat(request: ChatRequest):
    return generate(
        request.query,
        index,
        chunks,
        client,
        history_as_messages(request.chat_history),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
