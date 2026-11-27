"""Hybrid retrieval benchmark (session 3): which lanes and which RRF k work best.

    python eval/benchmark_hybrid.py                 # all modes, k = 60
    python eval/benchmark_hybrid.py --k 10 60 100   # also compare the RRF smoothing constant

Modes evaluated on the gold set (eval/gold/retrieval_s2.csv), main collection telixia_docs:
    dense                 BGE-M3 dense only (session 2 baseline)
    bm25                  BM25 only (exact terms)
    sparse                BGE-M3 learned sparse only
    rrf(dense+bm25)       the course pipeline
    rrf(dense+bm25_la)    same, language-aware: BM25 skipped for non-French questions
    rrf(dense+sparse)     same fusion with the learned sparse lane
    rrf(dense+bm25_la+sparse)

Gold-set column `category` (e.g. paraphrase / exact) gives results per question type.

Same metrics as eval/benchmark_retrieval.py (Recall@5, MRR@10, DocHit@5) + latency per lane.
Output: docs/benchmark_hybrid_s3.md (a hand-written "Reading" section is kept on re-runs).
"""

from __future__ import annotations

import argparse
import os
import sys
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

REPORT = ROOT / "docs" / "benchmark_hybrid_s3.md"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
SINGLE = ["dense", "bm25", "sparse"]
FUSIONS = [("dense", "bm25"), ("dense", "bm25_la"), ("dense", "sparse"), ("dense", "bm25_la", "sparse")]
LANE_LIMIT = 50          # each lane returns 50 candidates
CANDIDATES = 20          # the fused list handed to the reranker (Recall@20 = candidate recall)


def evaluate(gold, per_question, modes_k):
    from telixia.retrieval.fusion import rrf

    results = {}
    for name, lanes, k in modes_k:
        ranks, doc_ranks = [], []
        for g, (rankings, payloads) in zip(gold, per_question):
            if len(lanes) == 1:
                ids = rankings[lanes[0]][:CANDIDATES]
            else:
                ids = [d for d, _, _ in rrf({ln: rankings[ln] for ln in lanes}, k=k, limit=CANDIDATES)]
            pls = [payloads[i] for i in ids]
            ranks.append(first_rank(pls, g))
            doc_ranks.append(first_rank(pls, g, doc_only=True))
        m = metrics(ranks, doc_ranks)
        # candidate recall: is the answer among the chunks handed to the reranker?
        m["recall@20"] = sum(1 for r in ranks if r and r <= CANDIDATES) / max(len(ranks), 1)
        results[name] = {"ranks": ranks, "doc_ranks": doc_ranks, "metrics": m}
    return results


def add_categories(results, gold) -> list[str]:
    """Adds per-category metrics (column `category` of the gold set) to every mode."""
    cats = sorted({g.get("category") or "all" for g in gold})
    for r in results.values():
        r["by_category"] = {}
        for cat in cats:
            idx = [i for i, g in enumerate(gold) if (g.get("category") or "all") == cat]
            rk, dk = [r["ranks"][i] for i in idx], [r["doc_ranks"][i] for i in idx]
            m = metrics(rk, dk)
            m["recall@20"] = sum(1 for x in rk if x and x <= CANDIDATES) / max(len(rk), 1)
            m["n"] = len(idx)
            r["by_category"][cat] = m
    return cats


def write_report(results, gold, latency, ks):
    reading = "_To complete: which lanes help, which RRF k, and why._"
    if REPORT.exists():
        prev = REPORT.read_text(encoding="utf-8")
        if "## Reading" in prev and "_To complete" not in prev:
            reading = prev.split("## Reading", 1)[1].strip()
    lines = [
        "# Hybrid retrieval benchmark (session 3)", "",
        f"Date: {datetime.now():%Y-%m-%d %H:%M} | gold set: `{GOLD.relative_to(ROOT).as_posix()}` "
        f"({len(gold)} questions) | collection: telixia_docs (children 256 tokens) | "
        f"RRF k tested: {', '.join(map(str, ks))}", "",
        "Recall@20 = candidate recall: is the answer among the 20 chunks handed to the reranker?", "",
        "| Mode | Recall@5 | MRR@10 | DocHit@5 | Recall@20 |", "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r["metrics"]
        lines.append(f"| {name} | {m['recall@5']:.0%} | {m['mrr@10']:.2f} | {m['dochit@5']:.0%} "
                     f"| {m['recall@20']:.0%} |")
    cats = list(next(iter(results.values())).get("by_category", {}))
    if len(cats) > 1:
        lines += ["", "## By question category", "",
                  "| Mode | " + " | ".join(f"{c} R@5 | {c} MRR" for c in cats) + " |",
                  "|---|" + "---|---|" * len(cats)]
        for name, r in results.items():
            cells = " | ".join(f"{r['by_category'][c]['recall@5']:.0%} | {r['by_category'][c]['mrr@10']:.2f}"
                               for c in cats)
            lines.append(f"| {name} | {cells} |")
    lines += ["", "## Latency per lane (mean over the questions, ms)", "",
              "| Stage | Mean (ms) |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in latency.items()]
    lines += ["", "## Rank of the first correct result (- = not in the top 20)", "",
              "| # | Question | " + " | ".join(results) + " |",
              "|" + "---|" * (2 + len(results))]
    for i, g in enumerate(gold):
        q = g["question"] if len(g["question"]) <= 60 else g["question"][:57] + "..."
        cells = [str(r["ranks"][i]) if r["ranks"][i] else "-" for r in results.values()]
        lines.append(f"| {i + 1} | {q} | " + " | ".join(cells) + " |")
    lines += ["", "## Reading", "", reading, ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[60], help="RRF constants to compare")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from telixia.core.config import get_settings
    from telixia.core.embeddings import Embedder
    from telixia.retrieval.bm25 import BM25Index
    from telixia.retrieval.hybrid import HybridRetriever

    gold = load_gold(GOLD)
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=60)
    bm25 = BM25Index.from_jsonl(CHUNKS)
    print(f"Gold set: {len(gold)} questions | BM25: {len(bm25)} chunks, {len(bm25.idf)} terms")
    print("Loading BGE-M3...")
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(client, embedder, bm25, lane_limit=LANE_LIMIT, lang_aware=False)
    retriever.lane_rankings("warm-up", lanes=tuple(SINGLE))

    from telixia.ingestion.quality import detect_language
    per_question, lat = [], {"encode": [], "dense": [], "sparse": [], "bm25": []}
    for g in gold:
        rankings, payloads, timings = retriever.lane_rankings(g["question"], lanes=tuple(SINGLE))
        # language-aware variant: BM25 lane emptied when the question is not French
        rankings["bm25_la"] = rankings["bm25"] if detect_language(g["question"]) == "fr" else []
        per_question.append((rankings, payloads))
        for stage, ms in timings.items():
            lat[stage].append(ms)
    latency = {k: mean(v) for k, v in lat.items() if v}

    modes = [(lane, (lane,), 0) for lane in SINGLE]
    for lanes in FUSIONS:
        for k in args.k:
            suffix = "" if len(args.k) == 1 else f" k={k}"
            modes.append((f"rrf({'+'.join(lanes)}){suffix}", lanes, k))
    results = evaluate(gold, per_question, modes)
    cats = add_categories(results, gold)

    print(f"\n{'mode':<30} {'Recall@5':>9} {'MRR@10':>7} {'DocHit@5':>9} {'Recall@20':>10}")
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<30} {m['recall@5']:>9.0%} {m['mrr@10']:>7.2f} {m['dochit@5']:>9.0%} "
              f"{m['recall@20']:>10.0%}")
    if len(cats) > 1:
        print("\nBy question category (Recall@5 / MRR@10):")
        first = next(iter(results.values()))["by_category"]
        print(f"{'mode':<30}" + "".join(f"{cat + ' (' + str(first[cat]['n']) + 'q)':>22}" for cat in cats))
        for name, r in results.items():
            print(f"{name:<30}" + "".join(
                f"{r['by_category'][cat]['recall@5']:>13.0%} / {r['by_category'][cat]['mrr@10']:.2f}"
                for cat in cats))
    print("\nLatency (mean ms): " + " | ".join(f"{k} {v:.1f}" for k, v in latency.items()))
    write_report(results, gold, latency, args.k)
    print(f"Report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()