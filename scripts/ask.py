"""Ask Telixia a question: retrieval -> reranking -> grounded answer with citations.

    python scripts/ask.py "Le forfait 500 Go 5G+ est-il disponible sans engagement ?"
    python scripts/ask.py --demo                  # 3 answerable questions + 1 out of scope
    python scripts/ask.py "..." --show-context    # also print the text sent to the LLM

Settings come from .env: LLM_PROVIDER / LLM_MODEL, RETRIEVAL_LANES, RERANK_CANDIDATES,
RERANKER_MODEL. The answer is streamed word by word; then come the numbered sources,
the compliance check and the latency of every stage.
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

from telixia.generation.pipeline import RagPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
for noisy in ("httpx", "httpcore", "sentence_transformers", "FlagEmbedding"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("Ask")

DEMO = [
    "Le forfait 500 Go 5G+ est-il disponible sans engagement ?",
    "What is the withdrawal period after the delivery of equipment bought remotely with an Orange subscription?",
    "Quelle est la dur\u00e9e d'engagement de l'offre S\u00e9rie Sp\u00e9ciale Livebox Fibre ?",
    "Quel est le prix du forfait 5G de SFR ?",          # out of scope: must be refused
]
STATUS = {
    "ok": "OK - every citation points to a real source",
    "refusal": "REFUSAL - the corpus does not cover the question",
    "no_citation": "WARNING - answer without citation (not verifiable)",
    "invalid_citation": "WARNING - cites a source that does not exist",
    "truncated": "WARNING - answer cut off (LLM_MAX_TOKENS reached)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Telixia: grounded answer with citations")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    if not args.question and not args.demo:
        ap.error("give a question in quotes, or use --demo")

    t0 = time.perf_counter()
    rag = RagPipeline()
    logger.info(f"Pipeline ready in {time.perf_counter() - t0:.1f}s | LLM {rag.llm.provider}:{rag.llm.model} "
                f"| lanes {'+'.join(rag.lanes)} | {rag.candidates} candidates -> rerank -> top {rag.top_k}")

    line = "=" * 110
    for question in (DEMO if args.demo else [" ".join(args.question)]):
        print(f"\n{line}\nQUESTION : {question}\n{line}\nANSWER   : ", end="", flush=True)
        for piece in rag.stream(question):
            print(piece, end="", flush=True)
        r = rag.last
        print("\n\nSOURCES")
        for s in r.sources:
            mark = "*" if s["n"] in r.check["cited"] else " "
            print(f" {mark}[{s['n']}] {s['title']} | {s['section']}  (page {s['page']}, "
                  f"rerank {s['rerank_score']:.3f})")
            if args.show_context:
                print("      " + s["text"][:600].replace("\n", "\n      ") + ("..." if len(s["text"]) > 600 else ""))
        print(f"\nCHECK    : {STATUS.get(r.check['status'], r.check['status'])}   (* = cited)")
        t = r.timings
        stages = " | ".join(f"{k} {t[k]:.0f}" for k in ("encode", "dense", "bm25", "sparse", "rrf", "rerank")
                            if k in t)
        usage = ""
        if r.usage is not None:
            usage = (f" | tokens in {getattr(r.usage, 'prompt_tokens', '?')} "
                     f"out {getattr(r.usage, 'completion_tokens', '?')}")
        print(f"LATENCY  : {stages} | LLM first token {t.get('ttft', 0):.0f} | generation "
              f"{t.get('generation', 0):.0f} | total {t.get('total', 0):.0f} ms{usage}\n{line}")


if __name__ == "__main__":
    main()