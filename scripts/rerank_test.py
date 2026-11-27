"""Before / after reranking, side by side (session 3 demo, minute 2).

    python scripts/rerank_test.py "Le forfait 500 Go 5G+ est-il disponible sans engagement ?"
    python scripts/rerank_test.py --demo                    # the prepared trap questions
    python scripts/rerank_test.py "..." --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    python scripts/rerank_test.py "..." --candidates 10     # fewer pairs = faster reranking

BEFORE = dense search only (cosine score), the session 2 baseline.
AFTER  = hybrid retrieval (dense + BM25 + BGE-M3 sparse, RRF k=60) -> top-N candidates
         -> cross-encoder reranking -> top 5. "was #7" = rank before reranking.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
from telixia.retrieval.reranker import DEFAULT_MODEL, CrossEncoderReranker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("RerankTest")

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
LANES = ("dense", "bm25", "sparse")
DEMO = [
    "Le forfait 500 Go 5G+ est-il disponible sans engagement ?",
    "What is the withdrawal period after the delivery of equipment bought remotely with an Orange subscription?",
    "Quelle est la dur\u00e9e d'engagement de l'offre S\u00e9rie Sp\u00e9ciale Livebox Fibre ?",
]


def short(pl: dict, width: int = 62) -> str:
    s = f"{pl.get('title')} | {pl.get('section')}"
    return s if len(s) <= width else s[:width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Telixia: top 5 before / after cross-encoder reranking")
    ap.add_argument("query", nargs="*")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--candidates", type=int, default=20, help="first-stage candidates to rerank")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--int8", action="store_true", help="int8 quantized reranker (default: RERANKER_INT8 in .env)")
    args = ap.parse_args()
    if not args.query and not args.demo:
        ap.error("give a question in quotes, or use --demo")

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=30)
    bm25 = BM25Index.from_jsonl(CHUNKS)
    embedder = Embedder(settings.embedding_model)
    retriever = HybridRetriever(client, embedder, bm25)
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker(args.model, int8=True if args.int8 else None)
    logger.info(f"Reranker {args.model} loaded in {time.perf_counter() - t0:.1f}s")
    reranker.score("warm-up", ["warm-up"])                       # first call is slower
    embedder.encode(["warm-up"], batch_size=1)                  # avoids the 3.8 s cold start on Q1

    for query in (DEMO if args.demo else [" ".join(args.query)]):
        t0 = time.perf_counter()
        enc = embedder.encode([query], batch_size=1)[0]
        encode_ms = (time.perf_counter() - t0) * 1000
        before = retriever.search(query, top_n=5, lanes=("dense",), encoded=enc)
        cosine = dict(retriever.last_scores.get("dense", {}))
        res = retriever.search(query, top_n=args.candidates, lanes=LANES, encoded=enc)
        after = reranker.rerank(query, res.candidates, top_k=5)

        line = "=" * 142
        print(f"\n{line}\nQUESTION : {query}\n{line}")
        print(f"{'Rank':<5}{'BEFORE: dense only':<62}{'Cosine':>8}   "
              f"{'AFTER: hybrid + cross-encoder':<56}{'(rank before)':<14}{'Score':>6}")
        print("-" * 142)
        for i in range(5):
            b = before.candidates[i] if i < len(before.candidates) else None
            a = after[i] if i < len(after) else None
            left = f"{short(b.payload):<62}{cosine.get(b.id, 0):>8.4f}" if b else " " * 70
            right = (f"{short(a.candidate.payload, 54):<56}{'(was #' + str(a.first_stage_rank) + ')':<14}"
                     f"{a.score:>6.3f}" if a else "")
            print(f"{i + 1:<5}{left}   {right}")
        stages = " | ".join(f"{k} {v:.1f} ms" for k, v in res.timings.items() if k != "encode")
        total = encode_ms + sum(v for k, v in res.timings.items() if k != "encode") + reranker.last_ms
        skipped = " (BM25 skipped: question not in French)" if retriever.last_skipped else ""
        print(f"\nLatency: encode {encode_ms:.0f} ms | {stages} | rerank {len(res.candidates)} pairs "
              f"{reranker.last_ms:.0f} ms | total {total:.0f} ms{skipped}\n{line}")


if __name__ == "__main__":
    main()