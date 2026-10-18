"""
Basira RAG — Tafsir Retrieval
================================
Retrieves and presents the Tafsir Ibn Kathir for a given Quranic verse
by performing semantic search against a FAISS index of trusted sources
and then summarising the result using GPT-4o-mini.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss

from src.embeddings.embedder import get_client, embed_text
from src.vector_db.faiss_store import search
from src.nlp.preprocessor import preprocess
from src.rag.pipeline import ground_text

# Default paths for the FAISS index and chunks
INDEX_PATH = Path("data/processed/faiss.index")
CHUNKS_PATH = Path("data/processed/chunks.pkl")

# Minimum cosine-similarity score for a Quran chunk match to be trusted.
# Real ayahs score ≥ 0.72; fabricated text scores ~0.44.
_QURAN_SCORE_THRESHOLD = 0.55

# Canonical refusal returned when no tafsir chunk is found and when
# post-generation grounding rejects the generated prose (RAG-02).  Matches
# the long-standing no-match response byte for byte.
_TAFSIR_REFUSAL = "لم أجد تفسير هذه الآية في قاعدة البيانات."

# ---------------------------------------------------------------------------
# Module-level tafsir-map cache (built once per process, keyed by id(chunks))
# ---------------------------------------------------------------------------
# Key:   "تفسير ابن كثير {surah}:{ayah}"   (the chunk reference string)
# Value: the full chunk dict
_tafsir_map_cache: dict[int, dict[str, dict]] = {}


def _get_tafsir_map(chunks: list[dict]) -> dict[str, dict]:
    """Return (and cache) a {reference -> chunk} dict for tafsir chunks."""
    cache_key = id(chunks)
    if cache_key not in _tafsir_map_cache:
        tmap: dict[str, dict] = {}
        for c in chunks:
            if c.get("source") == "tafsir":
                ref = c.get("reference", "")
                if ref:
                    tmap[ref] = c
        _tafsir_map_cache[cache_key] = tmap
    return _tafsir_map_cache[cache_key]


SYSTEM_PROMPT = (
    "أنت متخصص في تفسير القرآن الكريم. بناءً على مقاطع تفسير ابن كثير المقدمة فقط:\n"
    "1. قدّم تفسير الآية المدخلة بأسلوب واضح وموجز\n"
    "2. اذكر السياق والسبب في النزول إن وُجد في المصادر\n"
    "لا تجاوب من معرفتك الخاصة. إذا لم تجد الآية في المصادر، "
    "قل: 'لم أجد تفسير هذه الآية في قاعدة البيانات'.\n"
    "أجب بالعربية فقط في شكل JSON بهذين المفتاحين فقط:\n"
    "{\n"
    "  'tafsir': 'نص التفسير',\n"
    "  'context': 'السياق وسبب النزول أو فارغ'\n"
    "}"
)


def load_faiss_index(
    index_path: str | Path = INDEX_PATH,
    chunks_path: str | Path = CHUNKS_PATH,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a FAISS index and its associated chunks from disk.

    Args:
        index_path:  Path to the FAISS index file.
        chunks_path: Path to the pickled chunks file.

    Returns:
        A (faiss.IndexFlatIP, chunks) tuple.
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[tafsir] Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    print(f"[tafsir] Loaded {len(chunks):,} chunks from {chunks_path}")
    return index, chunks


def get_key(d: dict, *keys: str, default: str = ""):
    """Return the value of the first key found in *d*, else *default*."""
    for key in keys:
        if key in d:
            return d[key]
    return default


def _flatten_source(source_val) -> str:
    """Normalise a source value that may be a dict into a flat string.

    GPT sometimes returns the source as a nested object, e.g.
    ``{"رقم_السورة": "…", "رقم_الآية": "…"}``.
    """
    if isinstance(source_val, dict):
        surah = source_val.get("رقم_السورة", source_val.get("surah", ""))
        ayah = source_val.get("رقم_الآية", source_val.get("ayah", ""))
        if surah or ayah:
            return f"{surah}:{ayah}" if ayah else str(surah)
        # Fallback: book + number style
        book = source_val.get("اسم_الكتاب", source_val.get("book", ""))
        number = source_val.get("رقم_الحديث", source_val.get("number", ""))
        return f"{book}:{number}" if number else book
    return str(source_val) if source_val else ""


def _filter_tafsir(results: list[dict]) -> list[dict]:
    """Keep only chunks whose source is 'tafsir'."""
    return [r for r in results if r.get("source") == "tafsir"]


# ---------------------------------------------------------------------------
# Public helper: resolve_tafsir_reference
# ---------------------------------------------------------------------------

def resolve_tafsir_reference(
    ayah_text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    debug: bool = False,
) -> tuple[Optional[dict], Optional[dict], str]:
    """Identify the correct tafsir chunk(s) for *ayah_text* via reference lookup.

    Steps:
      1. Dual-embedding search over the full index; filter to source=="quran"
         chunks to identify the ayah's surah and ayah number.
      2. Build/look up the module-level tafsir-map and retrieve the tafsir
         chunk whose reference is  "تفسير ابن كثير {surah}:{ayah}".
         If missing, fall back to the nearest previous ayah of the same surah.
      3. Also return the *next* tafsir chunk (same surah, ayah+1) for context.

    Returns:
        (primary_chunk, next_chunk, source_str)
        primary_chunk: the main tafsir chunk dict (or None if not found).
        next_chunk:    the immediately following tafsir chunk (or None).
        source_str:    "{surah}:{ayah}" derived from the chunk reference.
    """
    processed = preprocess(ayah_text)
    emb_norm = embed_text(processed["normalized"], client)
    emb_orig = embed_text(ayah_text, client)

    # Wide search so we can filter by source=="quran" after the fact
    wide_k = 200
    results_norm = search(emb_norm, index, chunks, top_k=wide_k)
    results_orig = search(emb_orig, index, chunks, top_k=wide_k)

    # Filter to quran source only
    quran_norm = [r for r in results_norm if r.get("source") == "quran"]
    quran_orig = [r for r in results_orig if r.get("source") == "quran"]

    best_quran = None
    if quran_orig and quran_norm:
        best_quran = (
            quran_orig[0]
            if quran_orig[0]["score"] > quran_norm[0]["score"]
            else quran_norm[0]
        )
    elif quran_norm:
        best_quran = quran_norm[0]
    elif quran_orig:
        best_quran = quran_orig[0]

    if debug:
        if best_quran:
            print(
                f"[DEBUG] Best quran match: ref={best_quran.get('reference')!r}  "
                f"score={best_quran.get('score'):.4f}"
            )
        else:
            print("[DEBUG] No quran chunk found")

    # Guard: score too low → fabricated / unknown ayah
    if best_quran is None or best_quran["score"] < _QURAN_SCORE_THRESHOLD:
        return None, None, ""

    # Look up the full chunk to get metadata (search() strips metadata)
    quran_chunk_full = next(
        (c for c in chunks if c.get("id") == best_quran["id"]), None
    )
    if quran_chunk_full is None:
        return None, None, ""

    meta = quran_chunk_full.get("metadata") or {}
    surah_num = meta.get("surah_number")
    ayah_num = meta.get("ayah_number")

    if surah_num is None or ayah_num is None:
        return None, None, ""

    if debug:
        print(f"[DEBUG] Resolved to surah={surah_num}, ayah={ayah_num}")

    # Build / retrieve the tafsir reference map
    tmap = _get_tafsir_map(chunks)

    # Direct lookup
    primary_ref = f"تفسير ابن كثير {surah_num}:{ayah_num}"
    primary = tmap.get(primary_ref)

    # Fallback: nearest previous ayah in the same surah that has a tafsir chunk
    if primary is None:
        for prev_ayah in range(ayah_num - 1, 0, -1):
            candidate_ref = f"تفسير ابن كثير {surah_num}:{prev_ayah}"
            if candidate_ref in tmap:
                primary = tmap[candidate_ref]
                if debug:
                    print(
                        f"[DEBUG] Tafsir not found for {primary_ref}; "
                        f"fell back to {candidate_ref}"
                    )
                break

    if primary is None:
        return None, None, ""

    # Parse source string from the chunk reference (never from GPT)
    # Reference format: "تفسير ابن كثير {surah}:{ayah}"
    primary_ref_str = primary.get("reference", "")
    # Extract "surah:ayah" part after the last space sequence
    source_str = primary_ref_str.split()[-1] if primary_ref_str else ""

    # Also grab the next chunk for context
    primary_meta = primary.get("metadata") or {}
    p_surah = primary_meta.get("surah_number", surah_num)
    p_ayah = primary_meta.get("ayah_number", ayah_num)
    next_ref = f"تفسير ابن كثير {p_surah}:{p_ayah + 1}"
    next_chunk = tmap.get(next_ref)

    return primary, next_chunk, source_str


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def get_tafsir(
    ayah_text: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client,
    top_k: int = 5,
    debug: bool = False,
) -> dict:
    """Retrieve Tafsir Ibn Kathir for a Quranic verse.

    Pipeline:
        1. Call resolve_tafsir_reference to identify the correct tafsir chunk
           by matching the ayah to the Quran index and looking up by reference.
        2. Send only that chunk (plus the next one) to GPT-4o-mini.
        3. Compute 'source' from the chunk reference — never from GPT output.
        4. Return a structured tafsir result.

    Args:
        ayah_text: The Quranic verse text to look up.
        index:     A loaded FAISS index.
        chunks:    The chunk dicts aligned with the index.
        client:    An authenticated OpenAI client.
        top_k:     (unused — kept for interface compatibility)
        debug:     If True, print raw GPT response and top FAISS hits.

    Returns:
        Dict with keys: tafsir, context, source, input_ayah.
    """
    primary, next_chunk, source_str = resolve_tafsir_reference(
        ayah_text, index, chunks, client, debug=debug
    )

    # Nothing found
    if primary is None:
        return {
            "tafsir": _TAFSIR_REFUSAL,
            "context": "",
            "source": "",
            "input_ayah": ayah_text,
        }

    # Build context from only the primary (+ next) chunk(s)
    context_chunks = [primary] + ([next_chunk] if next_chunk else [])
    sources_text = "\n\n".join(
        f"[مصدر {i + 1}]\n"
        f"النص: {c.get('original_text', '')}\n"
        f"المرجع: {c.get('reference', '')}"
        for i, c in enumerate(context_chunks)
    )

    user_message = (
        f"الآية المدخلة: {ayah_text}\n\n"
        f"مقاطع التفسير المسترجعة:\n{sources_text}"
    )

    # Call GPT-4o-mini with temperature=0 for determinism
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw_answer = response.choices[0].message.content

    if debug:
        print(f"\n[DEBUG] Raw GPT response:\n{raw_answer}")

    parse_failed = False
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        answer = {
            "tafsir": "لم أجد تفسير هذه الآية في قاعدة البيانات",
            "context": "",
        }
        parse_failed = True

    if debug:
        print(f"[DEBUG] Parsed keys: {list(answer.keys())}")

    tafsir_text = get_key(
        answer,
        "tafsir", "التفسير", "نص_التفسير", "تفسير",
        default="لم أجد تفسير هذه الآية في قاعدة البيانات",
    )
    context_text = get_key(
        answer,
        "context", "السياق", "سبب_النزول", "السياق_وسبب_النزول",
        "السياق وسبب النزول",
    )

    # RAG-02: ground the generated prose against the exact evidence sent to
    # the model (primary + next chunk); `chunks` only feeds the engine's
    # corpus-frequency checks.  Non-string model output is coerced so the
    # response keeps its string contract and grounding can never raise.
    if not isinstance(tafsir_text, str):
        tafsir_text = "" if tafsir_text is None else str(tafsir_text)
    if not isinstance(context_text, str):
        context_text = "" if context_text is None else str(context_text)
    if parse_failed:
        grounded_tafsir = tafsir_text
        grounded_context = context_text
    else:
        grounded_tafsir = ground_text(
            tafsir_text, context_chunks, chunks, refusal=_TAFSIR_REFUSAL,
        )
        if (grounded_tafsir == _TAFSIR_REFUSAL
                or not context_text.strip()):
            # Never keep ungrounded context next to a rejected tafsir.
            grounded_context = ""
        else:
            grounded_context = ground_text(
                context_text, context_chunks, chunks, refusal="",
            )

    return {
        "tafsir": grounded_tafsir,
        "context": grounded_context,
        # Source is always computed from the chunk reference — never from GPT
        "source": source_str,
        "input_ayah": ayah_text,
    }


if __name__ == "__main__":
    # --- Load resources ---
    client = get_client()
    index, chunks = load_faiss_index()

    # --- Diagnostic: print one quran chunk and one tafsir chunk ---
    print("=== DIAGNOSTIC: chunk field names ===")
    quran_sample = next(c for c in chunks if c.get("source") == "quran")
    tafsir_sample = next(c for c in chunks if c.get("source") == "tafsir")
    print("Quran chunk keys:  ", list(quran_sample.keys()))
    print("Quran metadata:    ", quran_sample.get("metadata"))
    print("Tafsir chunk keys: ", list(tafsir_sample.keys()))
    print("Tafsir metadata:   ", tafsir_sample.get("metadata"))
    print("==================")

    # --- Test cases ---
    test_inputs = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "إياك نعبد وإياك نستعين",
        "قل هو الله أحد",
        "اهدنا الصراط المستقيم",
        "آية غير موجودة إطلاقاً اختبار فقط",
    ]

    for ayah in test_inputs:
        print(f"\n{'=' * 60}")
        print(f"الآية المدخلة: {ayah}")
        print("-" * 60)

        result = get_tafsir(ayah, index, chunks, client, debug=True)

        print(f"التفسير:       {result['tafsir'][:120]}{'…' if len(result.get('tafsir','')) > 120 else ''}")
        print(f"السياق:        {result['context'][:80] if result.get('context') else ''}")
        print(f"المصدر:        {result['source']}")
        print(f"الآية المدخلة: {result['input_ayah']}")
        print("=" * 60)
