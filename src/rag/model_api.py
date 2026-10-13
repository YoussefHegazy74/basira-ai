import pickle

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from verify_full import verify_full
from tafsir import get_tafsir
from pipeline import generate


load_dotenv()

app = FastAPI(
    title="Basira AI API",
)

index = faiss.read_index("data/processed/faiss.index")

with open("data/processed/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = OpenAI()


class VerifyRequest(BaseModel):
    text: str
    explanation: str | None = None


class TafsirRequest(BaseModel):
    ayah: str

class ChatMessage(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None

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
        request.chat_history
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )