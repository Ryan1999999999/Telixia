"""Hybrid retrieval demo: each lane's top results, then the RRF fusion.

    python scripts/hybrid_test.py "What is the withdrawal period after the delivery of equipment?"
    python scripts/hybrid_test.py "forfait 500 Go 5G+" --doc-type RC --top-k 5
    python scripts/hybrid_test.py "..." --lanes dense bm25 sparse      # add the BGE-M3 sparse lane
    python scripts/hybrid_test.py --demo                               # prepared trap questions

Reading the output: in the fused list, "D:3 B:1" means rank 3 in the dense lane and
rank 1 in BM25; "-" means the lane did not return the chunk in its top 20.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from telixia.core.config import get_settings  # noqa: E402
from telixia.core.embeddings import Embedder  # noqa: E402
from telixia.retrieval.bm25 import BM25Index  # noqa: E402
from telixia.retrieval.hybrid import HybridRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("HybridTest")

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
SHORT = {"dense": "D", "bm25": "B", "sparse": "S"}
DEMO = [
    "What is the withdrawal period after the delivery of equipment bought remotely with an Orange subscription?",
    "Le prix de la location de la Livebox est-il inclus dans l'offre Livebox Max ?",
    "Le forfait 500 Go 5G+ est-il disponible sans engagement ?",
]


def label(pl: dict) -> str:
    return f"{pl.get('title')} | {pl.get('section')} | p.{pl.get('page')}"


def show(query, rankings, payloads, cands, timings, top_k, lanes, k, skipped=()):
    line = "=" * 92
    print(f"\n{line}\nQUESTION : {query}\n{line}")
    for lane in lanes:
        print(f"\n--- {lane.upper()} lane (top {top_k}) ---")
        for r, doc_id in enumerate(rankings.get(lane, [])[:top_k], 1):
            print(f"  {r}. {label(payloads[doc_id])}")
    print(f"\n--- RRF fusion k={k} (top {top_k}) ---")
    for r, c in enumerate(cands[:top_k], 1):
        ranks = " ".join(f"{SHORT[ln]}:{c.ranks.get(ln, '-')}" for ln in lanes)
        print(f"  {r}. RRF {c.rrf_score:.4f} [{ranks}] {label(c.payload)}")
        print(textwrap.indent(textwrap.fill(" ".join(str(c.payload.get('text', '')).split())[:220], 84),
                              "       "))
    if skipped:
        print(f"\n(lane skipped: {', '.join(skipped)} - question not in the corpus language, "
              f"BM25 is monolingual)")
    stages = " | ".join(f"{k} {v:.1f} ms" for k, v in timings.items())
    print(f"\nLatency: {stages} | retrieval total {sum(timings.values()):.0f} ms\n{line}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Telixia hybrid retrieval (dense + BM25 + RRF)")
    ap.add_argument("query", nargs="*")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lanes", nargs="+", default=["dense", "bm25"], choices=["dense", "bm25", "sparse"])
    ap.add_argument("--k", type=int, default=60, help="RRF smoothing constant")
    ap.add_argument("--doc-type")
    ap.add_argument("--offer-line", choices=["mobile", "internet"])
    ap.add_argument("--language", choices=["fr", "en"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-lang-aware", action="store_true", help="always run BM25, even cross-lingual")
    args = ap.parse_args()
    if not args.query and not args.demo:
        ap.error("give a question in quotes, or use --demo")

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=30)
    t0 = time.perf_counter()
    bm25 = BM25Index.from_jsonl(CHUNKS)
    logger.info(f"BM25 index: {len(bm25)} chunks, {len(bm25.idf)} terms, "
                f"built in {(time.perf_counter() - t0) * 1000:.0f} ms")
    t0 = time.perf_counter()
    embedder = Embedder(settings.embedding_model)
    logger.info(f"BGE-M3 loaded in {time.perf_counter() - t0:.1f}s")
    retriever = HybridRetriever(client, embedder, bm25, k=args.k, lang_aware=not args.no_lang_aware)

    where = {k: v for k, v in (("doc_type", args.doc_type), ("offer_line", args.offer_line),
                               ("language", args.language)) if v}
    retriever.search("warm-up", lanes=tuple(args.lanes))      # first call is slower (lazy init)
    for query in (DEMO if args.demo else [" ".join(args.query)]):
        res = retriever.search(query, top_n=20, where=where or None, lanes=tuple(args.lanes))
        show(query, res.rankings, res.payloads, res.candidates, res.timings, args.top_k, args.lanes, args.k,
             retriever.last_skipped)


if __name__ == "__main__":
    main()