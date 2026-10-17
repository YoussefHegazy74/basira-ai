import time
from src.vector_db.faiss_store import load_index
from src.embeddings.embedder import get_client
from src.rag.pipeline import generate

def safe(s):
    return str(s).encode("utf-8", "replace").decode("utf-8", "replace")

REFUSAL = "لم أجد في المصادر المتاحة ما يجيب عن هذا السؤال"
index, chunks = load_index("data/processed/faiss.index", "data/processed/chunks.pkl")
client = get_client()

rows = []
for line in open("src/rag/holdout.txt", encoding="utf-8"):
    if "|" in line:
        q, ph = line.split("|", 1)
        rows.append((q.strip(), ph.strip()))

stats = {"fiqh": [0, 0], "hadith": [0, 0]}
for i, (q, ph) in enumerate(rows, 1):
    key = ph[:50]
    exp = next((c for c in chunks if key in (c.get("original_text") or "")), None)
    exp_ref = (exp.get("reference") or "").strip() if exp else None
    exp_id = exp["id"] if exp else None
    cat = "hadith" if exp and exp.get("source") == "hadith" else "fiqh"
    t0 = time.time()
    r = generate(q, index, chunks, client)
    dt = time.time() - t0
    ans = r["answer"]
    srcs = [str(s) for s in r["sources"]]
    src_has = bool(exp_ref) and any(s.split("—", 1)[-1].strip() == exp_ref for s in srcs)
    refused = REFUSAL in ans
    ok = (not refused) and src_has
    stats[cat][1] += 1
    stats[cat][0] += int(ok)
    print("=" * 70)
    print(safe(f"[{i}/{len(rows)}] {cat} | {'PASS' if ok else 'FAIL'} | {dt:.1f}s"))
    print(safe("Q: " + q))
    print(safe(f"expected: {exp_id} | {exp_ref} | in sources: {src_has} | refusal: {refused}"))
    print(safe("sources: " + " ; ".join(srcs)))
    print(safe("ANSWER:\n " + ans))

print("=" * 70)
print("fiqh  :", stats["fiqh"][0], "/", stats["fiqh"][1])
print("hadith:", stats["hadith"][0], "/", stats["hadith"][1])
