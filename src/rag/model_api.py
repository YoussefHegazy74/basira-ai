import os
import pickle
import sys
from typing import Literal

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(RAG_DIR))
for path in (RAG_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from verify_full import verify_full
from tafsir import get_tafsir
from pipeline import generate


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

app = FastAPI(
    title="Basira AI API",
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

index = faiss.read_index(os.path.join(DATA_DIR, "faiss.index"))

with open(os.path.join(DATA_DIR, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

client = OpenAI()


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
