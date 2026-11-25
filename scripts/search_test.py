"""Top-k retrieval test on the Telixia collection (session 2 live demo).

    python scripts/search_test.py "Quel est le prix de l'offre 4G Home 200 Go ?"
    python scripts/search_test.py "frais de resiliation" --doc-type RC --top-k 5
    python scripts/search_test.py "debit 5G" --offer-line mobile --after 2025-01-01
    python scripts/search_test.py "Livebox" --mode hybrid          # dense | sparse | hybrid
    python scripts/search_test.py "..." --parent                   # show the parent section (LLM context)
    python scripts/search_test.py --demo                           # prepared FR/EN questions

Same model (BGE-M3) and same collection as scripts/ingest.py.
Uses client.query_points (client.search is deprecated in recent Qdrant clients).
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

os.environ.setdefault("TQDM_DISABLE", "1")             # hide FlagEmbedding progress bars
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from telixia.core.config import get_settings  # noqa: E402
from telixia.core.embeddings import Embedder  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SearchTest")
logging.getLogger("httpx").setLevel(logging.WARNING)      # hide one log line per HTTP call

COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "telixia_docs")

DEMO = [
    ("Quel est le prix de l'offre 4G Home 200 Go ?", {"doc_type": "RC"}),
    ("Combien coute la location de la Livebox par mois ?", {"category": "tarif"}),
    ("Quelles sont les conditions de resiliation d'un abonnement avec engagement ?", {}),
    ("What is the withdrawal period after buying a phone online?", {}),
    ("Quel est le debit maximum en 5G+ ?", {"offer_line": "mobile"}),
]


def build_filter(doc_type=None, offer_line=None, category=None, chunk_type=None, after=None):
    must = []
    for field, value in (("doc_type", doc_type), ("offer_line", offer_line),
                         ("category", category), ("chunk_type", chunk_type)):
        if value:
            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if after:
        must.append(models.FieldCondition(key="publication_date",
                                          range=models.DatetimeRange(gte=after)))
    return models.Filter(must=must) if must else None


def search(client, embedder, query: str, top_k: int, mode: str, flt):
    enc = embedder.encode([query], batch_size=1)[0]
    dense = enc.dense
    sparse = models.SparseVector(indices=enc.sparse_indices, values=enc.sparse_values)

    t0 = time.perf_counter()
    if mode == "dense":
        res = client.query_points(COLLECTION, query=dense, using="dense",
                                  query_filter=flt, limit=top_k, with_payload=True)
    elif mode == "sparse":
        res = client.query_points(COLLECTION, query=sparse, using="sparse",
                                  query_filter=flt, limit=top_k, with_payload=True)
    else:  # hybrid: dense + sparse candidates fused with Reciprocal Rank Fusion
        res = client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=flt, limit=50),
                models.Prefetch(query=sparse, using="sparse", filter=flt, limit=50),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    return res.points, latency_ms


def fetch_parent(settings, parent_id: str) -> str | None:
    import pg8000.dbapi as pg
    conn = pg.connect(**settings.pg_params())
    cur = conn.cursor()
    cur.execute("SELECT section, page_start, page_end, content FROM docs.parent_chunks "
                "WHERE parent_id = %s", (parent_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    section, p1, p2, content = row
    return f"[{section} | pages {p1}-{p2}]\n{content}"


def show(query, points, latency_ms, top_k, mode, flt_desc, settings=None, parent=False):
    line = "=" * 88
    print(f"\n{line}\nQUESTION : {query}")
    print(f"TOP-{top_k} | mode: {mode} | collection: {COLLECTION} | filter: {flt_desc or 'none'}"
          f" | search: {latency_ms:.0f} ms\n{line}")
    if not points:
        print("No result (filter too strict?)")
    score_name = "RRF" if mode == "hybrid" else ("Cosine" if mode == "dense" else "Sparse")
    for rank, p in enumerate(points, 1):
        pl = p.payload or {}
        print(f"\n[Rank {rank}] {score_name} {p.score:.4f} | {pl.get('title')} | {pl.get('section')}"
              f" | p.{pl.get('page')} | {pl.get('doc_type')} | {pl.get('chunk_type')}"
              f" | {pl.get('publication_date') or 'no date'}")
        excerpt = " ".join(str(pl.get("text", "")).split())[:300]
        print(textwrap.indent(textwrap.fill(excerpt, 84), "   "))
    if parent and points and settings:
        ctx = fetch_parent(settings, points[0].payload.get("parent_id"))
        if ctx:
            print("\n--- Parent section of rank 1 (what the LLM will receive) ---")
            print(textwrap.indent(ctx[:1500] + ("..." if len(ctx) > 1500 else ""), "   "))
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Telixia top-k retrieval test")
    ap.add_argument("query", nargs="*", help="question (in quotes)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="dense")
    ap.add_argument("--doc-type", help="RC, CGA, CGV, CGU, CS, CG, TARIFS, ASSURANCE...")
    ap.add_argument("--offer-line", choices=["mobile", "internet"])
    ap.add_argument("--category", help="contrat, tarif, assurance, formulaire")
    ap.add_argument("--chunk-type", choices=["text", "table"])
    ap.add_argument("--after", help="publication_date >= YYYY-MM-DD")
    ap.add_argument("--parent", action="store_true", help="show the parent section of rank 1")
    ap.add_argument("--demo", action="store_true", help="run the prepared demo questions")
    args = ap.parse_args()

    if not args.query and not args.demo:
        ap.error("give a question in quotes, or use --demo")

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=30)
    count = client.count(COLLECTION).count
    logger.info(f"Qdrant {settings.qdrant_url} | collection '{COLLECTION}': {count} points")

    t0 = time.perf_counter()
    embedder = Embedder(settings.embedding_model)
    logger.info(f"BGE-M3 loaded on {embedder.device} in {time.perf_counter() - t0:.1f}s")

    if args.demo:
        runs = [(q, f) for q, f in DEMO]
    else:
        runs = [(" ".join(args.query), {
            "doc_type": args.doc_type, "offer_line": args.offer_line,
            "category": args.category, "chunk_type": args.chunk_type, "after": args.after,
        })]

    for query, filters in runs:
        filters = {k: v for k, v in filters.items() if v}
        flt = build_filter(**filters)
        flt_desc = ", ".join(f"{k}={v}" for k, v in filters.items())
        logger.info(f"Question: '{query}'")
        points, ms = search(client, embedder, query, args.top_k, args.mode, flt)
        show(query, points, ms, args.top_k, args.mode, flt_desc, settings, args.parent)


if __name__ == "__main__":
    main()