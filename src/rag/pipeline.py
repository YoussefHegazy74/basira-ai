"""
Basira RAG — Pipeline
=======================
End-to-end Retrieval-Augmented Generation pipeline for Arabic Islamic content.
Retrieves relevant chunks via FAISS, builds an Arabic-aware prompt,
and generates answers using GPT-4o-mini.
"""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from src.nlp.preprocessor import preprocess
from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search

GENERATION_MODEL = "gpt-4o-mini"

SYSTEM_MESSAGE = (
    "أنت مساعد إسلامي موثوق. أجب بالعربية فقط بناءً على المصادر المقدمة. "
    "لا تتجاوز المصادر المذكورة."
)


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    client: OpenAI,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the most relevant chunks for a user query.

    Steps:
        1. Preprocess the query (normalisation + stopword removal).
        2. Embed the normalised text via OpenAI.
        3. Search the FAISS index for the closest vectors.

    Args:
        query:   Raw user question in Arabic.
        index:   A FAISS index built by faiss_store.build_index.
        chunks:  The chunk dicts aligned with the index.
        client:  An authenticated OpenAI client.
        top_k:   Number of results to return.

    Returns:
        List of top_k chunk result dicts from faiss_store.search.
    """
    processed = preprocess(query)
    clean_query = processed["normalized"]

    query_embedding = embed_text(clean_query, client)

    results = search(query_embedding, index, chunks, top_k=top_k)
    return results


def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """Build an Arabic-aware prompt for GPT-4o-mini.

    Constructs the system message and a user message that includes the
    original query and the retrieved source texts with their references.
    Optionally prepends recent chat history to give the model
    conversational context.

    Args:
        query:             The user's original question.
        retrieved_chunks:  List of result dicts from retrieve(), each with
                           'original_text', 'reference', and 'source'.
        chat_history:      Optional list of previous messages, each a dict
                           with 'role' ("user"/"assistant") and 'content'.
                           Only the last 3 exchanges (6 messages max) are
                           kept to control token cost.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        reference = chunk.get("reference", "غير محدد")
        source = chunk.get("source", "غير محدد")
        text = chunk.get("original_text", "")
        context_parts.append(
            f"[مصدر {i}] ({source} — {reference}):\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    user_message = (
        f"السؤال: {query}\n\n"
        f"المصادر:\n{context_block}\n\n"
        f"الإجابة:"
    )

    # --- Assemble messages -------------------------------------------------
    # 1. System message
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
    ]

    # 2. Chat history (limited to last 3 exchanges = 6 messages max)
    if chat_history:
        MAX_HISTORY_MESSAGES = 6  # 3 exchanges × 2 messages each
        limited_history = chat_history[-MAX_HISTORY_MESSAGES:]
        messages.extend(limited_history)

    # 3. Current user query with retrieved context
    messages.append({"role": "user", "content": user_message})

    return messages


def generate(
    query: str,
    index,
    chunks: list[dict],
    client: OpenAI,
    chat_history: list[dict] | None = None,
    top_k: int = 5,
) -> dict:
    """Run the full RAG pipeline: retrieve → build prompt → generate answer.

    Args:
        query:         Raw user question in Arabic.
        index:         A FAISS index built by faiss_store.build_index.
        chunks:        The chunk dicts aligned with the index.
        client:        An authenticated OpenAI client.
        chat_history:  Optional list of previous messages for multi-turn
                       conversation context.
        top_k:         Number of source chunks to retrieve.

    Returns:
        Dict with keys:
            - answer:  The generated response text from GPT-4o-mini.
            - sources: List of reference strings from retrieved chunks.
            - query:   The original user query.
    """
    retrieved = retrieve(query, index, chunks, client, top_k=top_k)

    messages = build_prompt(query, retrieved, chat_history=chat_history)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    sources = []
    for chunk in retrieved:
        ref = chunk.get("reference", "غير محدد")
        src = chunk.get("source", "غير محدد")
        sources.append(f"{src} — {ref}")

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
    }


if __name__ == "__main__":
    import inspect

    print("Pipeline updated — chat_history parameter added successfully")
    print(f"generate() signature: {inspect.signature(generate)}")
