"""Ingestion: chunks -> BGE-M3 (dense + sparse) -> Qdrant, documents & parents -> PostgreSQL.

    python scripts/build_chunks.py           # 1. PDFs -> data/processed/*.jsonl
    python scripts/ingest.py                 # 2. vectors + payload -> Qdrant, registry -> Postgres
    python scripts/ingest.py --recreate      #    drop and rebuild the collection
    python scripts/ingest.py --limit 200     #    quick test on the first 200 chunks

Idempotent: point ids are deterministic UUIDs (uuid5 of doc_id + chunk number), so
running it twice overwrites instead of duplicating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pg8000.dbapi as pg  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402
from tqdm import tqdm  # noqa: E402

from telixia.core.config import get_settings  # noqa: E402
from telixia.core.embeddings import Embedder  # noqa: E402

warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

PROCESSED = ROOT / "data" / "processed"
COLLECTION = "telixia_docs"
UPSERT_BATCH = 64            # points per Qdrant call (course recommendation: 64 or 128)
EMBED_BATCH = 8              # texts per forward pass of the model (CPU friendly)

KEYWORD_FIELDS = ["doc_id", "doc_type", "category", "offer_line", "chunk_type",
                  "country", "language", "source"]
PAYLOAD_DROP = {"embed_text", "id"}   # not needed in Qdrant (embed_text = context + text)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------- Qdrant
def ensure_collection(client: QdrantClient, dim: int, recreate: bool) -> None:
    exists = client.collection_exists(COLLECTION)
    if exists and recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        print(f"Collection '{COLLECTION}' created (dense {dim}d Cosine HNSW m=16 ef=100 + sparse)")
    # payload indexes = fast filtering inside the HNSW search
    for field in KEYWORD_FIELDS:
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "publication_date", models.PayloadSchemaType.DATETIME)
    client.create_payload_index(COLLECTION, "page", models.PayloadSchemaType.INTEGER)
    client.create_payload_index(COLLECTION, "is_synthetic", models.PayloadSchemaType.BOOL)


def index_chunks(client: QdrantClient, embedder: Embedder, chunks: list[dict]) -> None:
    start = time.time()
    with tqdm(total=len(chunks), desc="embed+upsert", unit="chunk") as bar:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i:i + UPSERT_BATCH]
            encoded = embedder.encode([c["embed_text"] for c in batch], batch_size=EMBED_BATCH)
            points = [
                models.PointStruct(
                    id=c["id"],
                    vector={
                        "dense": e.dense,
                        "sparse": models.SparseVector(indices=e.sparse_indices, values=e.sparse_values),
                    },
                    payload={k: v for k, v in c.items() if k not in PAYLOAD_DROP},
                )
                for c, e in zip(batch, encoded)
            ]
            client.upsert(COLLECTION, points=points, wait=True)
            bar.update(len(batch))
    elapsed = time.time() - start
    print(f"{len(chunks)} chunks indexed in {elapsed:.0f}s ({len(chunks) / elapsed:.1f} chunks/s)")


# ------------------------------------------------------------------------- PostgreSQL
def file_info(path: Path) -> tuple[str, int | None]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        import pymupdf
        with pymupdf.open(path) as d:
            return sha, d.page_count
    except Exception:
        return sha, None


def save_registry(settings, chunks: list[dict], parents: list[dict]) -> None:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    conn = pg.connect(**settings.pg_params())
    cur = conn.cursor()
    for doc_id, cs in by_doc.items():
        c0 = cs[0]
        file_path = next((p for p in (ROOT / "data" / "raw" / "docs").rglob(c0["source"])), None)
        sha, n_pages = file_info(file_path) if file_path else (None, None)
        cur.execute(
            """
            INSERT INTO docs.documents (doc_id, title, source_url, file_path, category, doc_type,
                country, publication_date, language, is_synthetic, sha256, n_pages, n_chunks)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s::date,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title, source_url = EXCLUDED.source_url,
                file_path = EXCLUDED.file_path, category = EXCLUDED.category,
                doc_type = EXCLUDED.doc_type, publication_date = EXCLUDED.publication_date,
                sha256 = EXCLUDED.sha256, n_pages = EXCLUDED.n_pages,
                n_chunks = EXCLUDED.n_chunks, ingested_at = now()
            """,
            (doc_id, c0["title"], c0["source_url"],
             file_path.relative_to(ROOT).as_posix() if file_path else c0["source"],
             c0["category"], c0["doc_type"], c0["country"], c0["publication_date"],
             c0["language"], c0["is_synthetic"], sha, n_pages, len(cs)),
        )
    docs_kept = set(by_doc)
    rows = [
        (p["parent_id"], p["doc_id"], p["section"], p["page_start"], p["page_end"], p["text"])
        for p in parents if p["doc_id"] in docs_kept
    ]
    cur.executemany(
        """
        INSERT INTO docs.parent_chunks (parent_id, doc_id, section, page_start, page_end, content)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (parent_id) DO UPDATE SET
            section = EXCLUDED.section, page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end, content = EXCLUDED.content
        """,
        rows,
    )
    conn.commit()
    conn.close()
    print(f"PostgreSQL: {len(by_doc)} documents, {len(rows)} parent chunks upserted")


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--limit", type=int, default=0, help="only the first N chunks (test)")
    args = ap.parse_args()

    settings = get_settings()
    chunks = load_jsonl(PROCESSED / "chunks.jsonl")
    parents = load_jsonl(PROCESSED / "parents.jsonl")
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"{len(chunks)} chunks, {len(parents)} parents loaded")

    save_registry(settings, chunks, parents)      # parents first: the LLM context store

    client = QdrantClient(url=settings.qdrant_url,
                          api_key=settings.qdrant_api_key.get_secret_value(), timeout=120)
    ensure_collection(client, settings.embedding_dim, args.recreate)

    print(f"Loading {settings.embedding_model} (first run downloads ~2.3 GB)...")
    embedder = Embedder(settings.embedding_model)
    print(f"Model on {embedder.device}")
    index_chunks(client, embedder, chunks)

    info = client.get_collection(COLLECTION)
    print(f"\nCollection '{COLLECTION}': {info.points_count} points, status {info.status}")
    print(f"Dashboard: {settings.qdrant_url}/dashboard#/collections/{COLLECTION}")


if __name__ == "__main__":
    main()