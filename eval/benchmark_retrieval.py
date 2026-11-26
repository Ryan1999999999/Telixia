"""Retrieval benchmark: compares chunk sizes (and dense vs hybrid) on the gold set.

    python eval/benchmark_retrieval.py                      # children 256 vs 512 tokens
    python eval/benchmark_retrieval.py --sizes 128 256 512  # add a third variant (slower)
    python eval/benchmark_retrieval.py --skip-index         # re-evaluate existing bench collections

For each chunk size:
  1. build_chunks.py --child-tokens N  -> data/processed/bench_N/*.jsonl
  2. embeddings: reused from a local cache (seeded from the main Qdrant collection),
     only chunks whose text is new are encoded with BGE-M3
  3. indexed into a separate collection telixia_bench_N (the main collection is untouched)
  4. the 10 gold questions are run in dense and hybrid mode

Metrics (a hit = expected document AND section start):
  Recall@5  share of questions whose answer is in the top 5
  MRR@10    mean of 1/rank of the first correct result (0 if not in the top 10)
  DocHit@5  same as Recall@5 but only the document has to match

Outputs: docs/benchmark_s2.md (report, committed) and eval/runs/benchmark_s2.json (raw).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLD = ROOT / "eval" / "gold" / "retrieval_s2.csv"
CACHE_DB = ROOT / "data" / "processed" / "embedding_cache.sqlite"
REPORT_MD = ROOT / "docs" / "benchmark_s2.md"
REPORT_JSON = ROOT / "eval" / "runs" / "benchmark_s2.json"
MAIN_COLLECTION = "telixia_docs"
TOP_K = 10
EMBED_BATCH = 8
UPSERT_BATCH = 64


# --------------------------------------------------------------------------- gold set
def norm(text: str) -> str:
    """Lowercase, no accents, simple quotes, single spaces: robust string comparison."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.lower().split())


def load_gold(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:          # utf-8-sig: tolerates a BOM
        rows = [r for r in csv.DictReader(f, delimiter=";") if (r.get("question") or "").strip()]
    for r in rows:
        r["question"] = r["question"].strip()
        r["expected_title"] = (r.get("expected_title") or "").strip()
        r["expected_section"] = (r.get("expected_section") or "").strip()
    return rows


def section_matches(actual: str, expected: str) -> bool:
    """'Article 5' matches 'Article 5. Retractation...' but not 'Article 50'."""
    a, e = norm(actual), norm(expected)
    if not e:
        return True
    if not a.startswith(e):
        return False
    rest = a[len(e):]
    return not (e[-1].isdigit() and rest[:1].isdigit())


def is_hit(payload: dict, gold: dict, doc_only: bool = False) -> bool:
    if norm(payload.get("title", "")) != norm(gold["expected_title"]):
        return False
    return doc_only or section_matches(payload.get("section", ""), gold["expected_section"])


def first_rank(payloads: list[dict], gold: dict, doc_only: bool = False) -> int | None:
    for rank, pl in enumerate(payloads, 1):
        if is_hit(pl, gold, doc_only):
            return rank
    return None


def metrics(ranks: list[int | None], doc_ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr@10": sum(1 / r for r in ranks if r and r <= TOP_K) / n,
        "dochit@5": sum(1 for r in doc_ranks if r and r <= 5) / n,
    }


# ------------------------------------------------------------------------- variants
def build_variant(size: int, overlap: int) -> Path:
    out = ROOT / "data" / "processed" / f"bench_{size}"
    print(f"\n=== Variant {size} tokens: chunking -> {out.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_chunks.py"),
                    "--child-tokens", str(size), "--overlap", str(overlap), "--out", str(out)],
                   check=True)
    return out


def index_variant(client, embedder_factory, cache, size: int, folder: Path) -> dict:
    from qdrant_client import models

    with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    missing = [c for c in chunks if cache.get(c["embed_text"]) is None]
    print(f"{len(chunks)} chunks, {len(chunks) - len(missing)} from cache, {len(missing)} to encode")
    t0 = time.time()
    if missing:
        embedder = embedder_factory()
        for i in range(0, len(missing), UPSERT_BATCH):
            batch = missing[i:i + UPSERT_BATCH]
            enc = embedder.encode([c["embed_text"] for c in batch], batch_size=EMBED_BATCH)
            cache.put_many([(c["embed_text"], e.dense, e.sparse_indices, e.sparse_values)
                            for c, e in zip(batch, enc)])
            done = min(i + UPSERT_BATCH, len(missing))
            print(f"  encoded {done}/{len(missing)} ({(time.time() - t0) / done:.2f} s/chunk)")
    encode_s = time.time() - t0

    name = f"telixia_bench_{size}"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(
            size=1024, distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i:i + UPSERT_BATCH]
        points = []
        for c in batch:
            dense, idx, val = cache.get(c["embed_text"])
            points.append(models.PointStruct(
                id=c["id"],
                vector={"dense": dense, "sparse": models.SparseVector(indices=idx, values=val)},
                payload={k: v for k, v in c.items() if k not in ("embed_text", "id")}))
        client.upsert(name, points=points, wait=True)
    tokens = [c["n_tokens"] for c in chunks]
    return {"collection": name, "chunks": len(chunks), "encoded": len(missing),
            "encode_s": round(encode_s), "avg_tokens": round(sum(tokens) / len(tokens)),
            "max_tokens": max(tokens)}


def search(client, name: str, enc, mode: str) -> tuple[list[dict], float]:
    from qdrant_client import models

    sparse = models.SparseVector(indices=enc.sparse_indices, values=enc.sparse_values)
    t0 = time.perf_counter()
    if mode == "dense":
        res = client.query_points(name, query=enc.dense, using="dense", limit=TOP_K, with_payload=True)
    else:
        res = client.query_points(
            name,
            prefetch=[models.Prefetch(query=enc.dense, using="dense", limit=50),
                      models.Prefetch(query=sparse, using="sparse", limit=50)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K, with_payload=True)
    return [p.payload or {} for p in res.points], (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- report
def write_report(results: dict, gold: list[dict], meta: dict) -> None:
    lines = [
        "# Retrieval benchmark (session 2 deliverable)",
        "",
        f"Date: {meta['date']} | gold set: `{GOLD.relative_to(ROOT).as_posix()}` "
        f"({len(gold)} questions, {sum(g.get('lang') == 'en' for g in gold)} in English) | "
        f"model: BGE-M3 | parents: {meta['parent_tokens']} tokens | overlap: {meta['overlap']} tokens",
        "",
        "A hit = expected document **and** expected section. DocHit@5 = document only.",
        "",
        "| Children | Mode | Recall@5 | MRR@10 | DocHit@5 | Chunks | Avg tokens | Search (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for size, r in results.items():
        for mode in ("dense", "hybrid"):
            m = r["metrics"][mode]
            lines.append(f"| {size} | {mode} | {m['recall@5']:.0%} | {m['mrr@10']:.2f} | "
                         f"{m['dochit@5']:.0%} | {r['index']['chunks']} | {r['index']['avg_tokens']} | "
                         f"{r['latency_ms'][mode]:.0f} |")
    lines += ["", "## Rank of the first correct result per question (- = not in the top 10)", ""]
    header = "| # | Question | " + " | ".join(f"{s} {m}" for s in results for m in ("dense", "hybrid")) + " |"
    lines += [header, "|" + "---|" * (2 + 2 * len(results))]
    for i, g in enumerate(gold):
        cells = []
        for s in results:
            for m in ("dense", "hybrid"):
                rank = results[s]["ranks"][m][i]
                cells.append(str(rank) if rank else "-")
        q = g["question"] if len(g["question"]) <= 70 else g["question"][:67] + "..."
        lines.append(f"| {i + 1} | {q} | " + " | ".join(cells) + " |")
    reading = "_To complete: which size and which mode win, and why._"
    if REPORT_MD.exists():                     # keep a Reading section written by hand
        previous = REPORT_MD.read_text(encoding="utf-8")
        if "## Reading" in previous and "_To complete" not in previous:
            reading = previous.split("## Reading", 1)[1].strip()
    lines += ["", "## Reading", "", reading, ""]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({"meta": meta, "results": results}, indent=2, ensure_ascii=False),
                           encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Telixia retrieval benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512], help="child sizes to compare")
    ap.add_argument("--overlap", type=int, default=40, help="same overlap for every size (fair comparison)")
    ap.add_argument("--skip-index", action="store_true", help="evaluate existing bench collections")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from telixia.core.config import get_settings
    from telixia.core.embeddings import Embedder
    from telixia.core.embedding_cache import EmbeddingCache, seed_from_qdrant

    gold = load_gold(GOLD)
    print(f"Gold set: {len(gold)} questions")
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=120)

    embedder_box: dict = {}

    def get_embedder():
        if "e" not in embedder_box:
            print("Loading BGE-M3...")
            embedder_box["e"] = Embedder(settings.embedding_model)
        return embedder_box["e"]

    cache = EmbeddingCache(CACHE_DB)
    if not args.skip_index:
        if cache.size() == 0:
            print(f"Seeding the embedding cache from '{MAIN_COLLECTION}'...")
            print(f"  {seed_from_qdrant(client, cache, MAIN_COLLECTION)} vectors copied")
        else:
            print(f"Embedding cache: {cache.size()} vectors")

    results = {}
    for size in args.sizes:
        if args.skip_index:
            stats_file = ROOT / "data" / "processed" / f"bench_{size}" / "chunks.jsonl"
            toks = ([json.loads(x)["n_tokens"] for x in stats_file.open(encoding="utf-8") if x.strip()]
                    if stats_file.exists() else [0])
            index_info = {"collection": f"telixia_bench_{size}",
                          "chunks": client.count(f"telixia_bench_{size}").count,
                          "encoded": 0, "encode_s": 0, "avg_tokens": round(sum(toks) / len(toks)),
                          "max_tokens": max(toks)}
        else:
            folder = build_variant(size, args.overlap)
            index_info = index_variant(client, get_embedder, cache, size, folder)
        results[size] = {"index": index_info, "ranks": {}, "doc_ranks": {}, "metrics": {},
                         "latency_ms": {}}

    print("\nEncoding the questions...")
    q_enc = get_embedder().encode([g["question"] for g in gold], batch_size=8)

    for size, r in results.items():
        for mode in ("dense", "hybrid"):
            ranks, doc_ranks, lat = [], [], []
            for g, enc in zip(gold, q_enc):
                payloads, ms = search(client, r["index"]["collection"], enc, mode)
                ranks.append(first_rank(payloads, g))
                doc_ranks.append(first_rank(payloads, g, doc_only=True))
                lat.append(ms)
            r["ranks"][mode], r["doc_ranks"][mode] = ranks, doc_ranks
            r["metrics"][mode] = metrics(ranks, doc_ranks)
            r["latency_ms"][mode] = sum(lat) / len(lat)

    print(f"\n{'size':>6} {'mode':<7} {'Recall@5':>9} {'MRR@10':>7} {'DocHit@5':>9} {'chunks':>7}")
    for size, r in results.items():
        for mode in ("dense", "hybrid"):
            m = r["metrics"][mode]
            print(f"{size:>6} {mode:<7} {m['recall@5']:>9.0%} {m['mrr@10']:>7.2f} "
                  f"{m['dochit@5']:>9.0%} {r['index']['chunks']:>7}")

    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "overlap": args.overlap,
            "parent_tokens": 1024, "sizes": args.sizes, "top_k": TOP_K}
    write_report(results, gold, meta)
    print(f"\nReport: {REPORT_MD.relative_to(ROOT)} | raw: {REPORT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()