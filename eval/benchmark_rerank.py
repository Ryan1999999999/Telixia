"""Reranking benchmark (session 3): does the cross-encoder fix the order, and at what cost?

    python eval/benchmark_rerank.py                                  # both models, 10 and 20 candidates
    python eval/benchmark_rerank.py --models BAAI/bge-reranker-v2-m3 --candidates 20

Pipelines compared on the gold set (15 questions):
    dense                       first stage only (session 2 baseline)
    hybrid                      dense + BM25 (language-aware) + BGE-M3 sparse, RRF k=60
    dense  -> rerank[model, N]  the N best dense chunks rescored by the cross-encoder
    hybrid -> rerank[model, N]  the N best fused chunks rescored by the cross-encoder

Metrics: Recall@5, MRR@10 (+ per question category) and the reranking latency per
question (mean and p95), plus a full latency budget per stage for the evaluation sheet.
Output: docs/benchmark_rerank_s3.md (a hand-written "Reading" section is kept on re-runs).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from benchmark_retrieval import GOLD, first_rank, load_gold, metrics  # noqa: E402

REPORT = ROOT / "docs" / "benchmark_rerank_s3.md"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
MODELS = ["BAAI/bge-reranker-v2-m3", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"]
LANES = ("dense", "bm25", "sparse")
TOP = 10                  # ranks are measured inside the final top 10


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))] if s else 0.0


def score_pipeline(gold, payload_lists) -> dict:
    """payload_lists[i] = final ordered payloads for question i."""
    ranks = [first_rank(pls[:TOP], g) for g, pls in zip(gold, payload_lists)]
    doc_ranks = [first_rank(pls[:TOP], g, doc_only=True) for g, pls in zip(gold, payload_lists)]
    m = metrics(ranks, doc_ranks)
    by_cat = {}
    for cat in sorted({g.get("category") or "all" for g in gold}):
        idx = [i for i, g in enumerate(gold) if (g.get("category") or "all") == cat]
        by_cat[cat] = metrics([ranks[i] for i in idx], [doc_ranks[i] for i in idx])
        by_cat[cat]["n"] = len(idx)
    return {"ranks": ranks, "metrics": m, "by_category": by_cat}


def short(pl: dict, width: int = 70) -> str:
    s = f"{pl.get('title')} / {pl.get('section')}"        # no "|": it would break markdown tables
    return s if len(s) <= width else s[:width - 3] + "..."


def write_report(gold, results, latency, budget, showcase, models_info):
    reading = "_To complete: does reranking fix the order, which model, which N, at what latency?_"
    if REPORT.exists():
        prev = REPORT.read_text(encoding="utf-8")
        if "## Reading" in prev and "_To complete" not in prev:
            reading = prev.split("## Reading", 1)[1].strip()
    cats = list(next(iter(results.values()))["by_category"])
    lines = [
        "# Reranking benchmark (session 3)", "",
        f"Date: {datetime.now():%Y-%m-%d %H:%M} | gold set: `{GOLD.relative_to(ROOT).as_posix()}` "
        f"({len(gold)} questions) | first stage: dense / hybrid (dense + BM25 language-aware + "
        f"BGE-M3 sparse, RRF k=60) | CPU", "",
        "| Pipeline | Recall@5 | MRR@10 | " + " | ".join(f"{c} MRR" for c in cats)
        + " | Rerank mean (ms) | Rerank p95 (ms) |",
        "|---|---|---|" + "---|" * len(cats) + "---|---|",
    ]
    for name, r in results.items():
        m = r["metrics"]
        lat = latency.get(name)
        lat_cells = f"{mean(lat):.0f} | {p95(lat):.0f}" if lat else "- | -"
        lines.append(f"| {name} | {m['recall@5']:.0%} | {m['mrr@10']:.2f} | "
                     + " | ".join(f"{r['by_category'][c]['mrr@10']:.2f}" for c in cats)
                     + f" | {lat_cells} |")
    lines += ["", "## Models", "", "| Model | Parameters | Load time (s) |", "|---|---|---|"]
    lines += [f"| `{m}` | {info['params']} | {info['load_s']:.1f} |" for m, info in models_info.items()]
    lines += ["", "## Latency budget of the best pipeline (mean per question, ms)", "",
              "| Stage | Measured (ms) | SLA (ms) | Status |", "|---|---|---|---|"]
    for stage, (ms, sla) in budget.items():
        status = "OK" if ms <= sla else "over SLA"
        lines.append(f"| {stage} | {ms:.1f} | < {sla} | {status} |")
    q, before, after = showcase
    lines += ["", "## Before / after reranking (evaluation sheet, section 2)", "",
              f"Question: \"{q}\"", "",
              "| Rank | Dense search only | Cosine | After reranking (best pipeline) | Rank before | Cross-encoder |",
              "|:-:|---|:-:|---|:-:|:-:|"]
    for i in range(5):
        b = before[i] if i < len(before) else ("", 0.0)
        a = after[i] if i < len(after) else ("", 0, 0.0)
        lines.append(f"| {i + 1} | {b[0]} | {b[1]:.4f} | {a[0]} | {a[1]} | {a[2]:.3f} |")
    lines += ["", "## Rank of the first correct result per question (- = not in the top 10)", "",
              "| # | Question | " + " | ".join(results) + " |", "|" + "---|" * (2 + len(results))]
    for i, g in enumerate(gold):
        qq = g["question"] if len(g["question"]) <= 55 else g["question"][:52] + "..."
        cells = [str(r["ranks"][i]) if r["ranks"][i] else "-" for r in results.values()]
        lines.append(f"| {i + 1} | {qq} | " + " | ".join(cells) + " |")
    lines += ["", "## Reading", "", reading, ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--candidates", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--showcase", type=int, default=8, help="gold question number for the before/after table")
    ap.add_argument("--int8", action="store_true", help="int8 dynamic quantization of the rerankers")
    ap.add_argument("--report", default=None, help="report path (default docs/benchmark_rerank_s3.md)")
    args = ap.parse_args()
    global REPORT
    if args.report:
        REPORT = ROOT / args.report

    from qdrant_client import QdrantClient
    from telixia.core.config import get_settings
    from telixia.core.embeddings import Embedder
    from telixia.retrieval.bm25 import BM25Index
    from telixia.retrieval.hybrid import HybridRetriever
    from telixia.retrieval.reranker import CrossEncoderReranker

    gold = load_gold(GOLD)
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=60)
    bm25 = BM25Index.from_jsonl(CHUNKS)
    print(f"Gold set: {len(gold)} questions. Loading BGE-M3...")
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(client, embedder, bm25, lane_limit=50)
    retriever.search("warm-up", lanes=LANES)

    n_max = max(args.candidates)
    first_stage, stage_ms = [], {"encode": [], "dense": [], "bm25": [], "sparse": [], "rrf": []}
    for g in gold:
        t0 = time.perf_counter()
        enc = embedder.encode([g["question"]], batch_size=1)[0]
        stage_ms["encode"].append((time.perf_counter() - t0) * 1000)
        dense = retriever.search(g["question"], top_n=n_max, lanes=("dense",), encoded=enc)
        cosine = dict(retriever.last_scores.get("dense", {}))
        hybrid = retriever.search(g["question"], top_n=n_max, lanes=LANES, encoded=enc)
        for stage in ("dense", "bm25", "sparse", "rrf"):
            if stage in hybrid.timings:
                stage_ms[stage].append(hybrid.timings[stage])
        first_stage.append({"dense": dense.candidates, "hybrid": hybrid.candidates, "cosine": cosine})

    results = {
        "dense": score_pipeline(gold, [[c.payload for c in fs["dense"]] for fs in first_stage]),
        "hybrid": score_pipeline(gold, [[c.payload for c in fs["hybrid"]] for fs in first_stage]),
    }
    latency, models_info, reranked_store = {}, {}, {}
    for model in args.models:
        print(f"\nLoading reranker {model}...")
        t0 = time.perf_counter()
        try:
            reranker = CrossEncoderReranker(model, int8=args.int8)
        except Exception as e:                       # e.g. download or DLL problem: skip, keep going
            print(f"  [skip] {model}: {e}")
            continue
        load_s = time.perf_counter() - t0
        reranker.score("warm-up", ["warm-up"])
        try:
            params = reranker.n_params
            params = f"{params / 1e6:.0f}M"
        except Exception:
            params = "?"
        models_info[model] = {"params": params, "load_s": load_s}
        tag = model.split("/")[-1] + (" int8" if args.int8 else "")
        for first in ("dense", "hybrid"):
            for n in args.candidates:
                name = f"{first} -> rerank[{tag}, {n}]"
                finals, lat = [], []
                for g, fs in zip(gold, first_stage):
                    out = reranker.rerank(g["question"], fs[first][:n], top_k=TOP)
                    lat.append(reranker.last_ms)
                    finals.append([r.candidate.payload for r in out])
                    reranked_store[(name, g["question"])] = out
                results[name] = score_pipeline(gold, finals)
                latency[name] = lat
                m = results[name]["metrics"]
                print(f"  {name:<58} Recall@5 {m['recall@5']:>4.0%}  MRR@10 {m['mrr@10']:.2f}  "
                      f"rerank mean {mean(lat):.0f} ms  p95 {p95(lat):.0f} ms")

    print(f"\n{'pipeline':<60} {'Recall@5':>9} {'MRR@10':>7}")
    for name, r in results.items():
        print(f"{name:<60} {r['metrics']['recall@5']:>9.0%} {r['metrics']['mrr@10']:>7.2f}")

    best = max((n for n in results if n in latency), key=lambda n: results[n]["metrics"]["mrr@10"],
               default=None)
    budget = {
        "1. Query encoding (BGE-M3)": (mean(stage_ms["encode"]), 30),
        "2. Dense search Qdrant (HNSW)": (mean(stage_ms["dense"]), 40),
        "3. Sparse search BM25 (in memory)": (mean(stage_ms["bm25"]) if stage_ms["bm25"] else 0.0, 15),
        "3b. Sparse search BGE-M3 (Qdrant)": (mean(stage_ms["sparse"]), 15),
        "4. RRF fusion": (mean(stage_ms["rrf"]), 5),
    }
    if best:
        budget[f"5. Cross-encoder reranking ({best.split('[', 1)[1].rstrip(']')})"] = (mean(latency[best]), 150)
        print(f"\nBest reranked pipeline (MRR): {best}")

    sq = gold[args.showcase - 1]["question"]
    fs = first_stage[args.showcase - 1]
    before = [(short(c.payload), fs["cosine"].get(c.id, 0.0)) for c in fs["dense"][:5]]
    after = []
    if best:
        after = [(short(r.candidate.payload), r.first_stage_rank, r.score)
                 for r in reranked_store[(best, sq)][:5]]
    write_report(gold, results, latency, budget, (sq, before, after), models_info)
    print(f"Report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()