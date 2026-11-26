"""Embedding cache: never encode the same text twice.

sqlite file data/processed/embedding_cache.sqlite, key = sha1(embed_text),
value = dense vector (float32 bytes) + sparse vector (json). Used by scripts/ingest.py
and eval/benchmark_retrieval.py. On CPU, BGE-M3 needs ~0.4 s per chunk, so a full
re-index drops from ~38 min to seconds when only metadata or a few chunks changed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from array import array
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dense BLOB, "
                        "sparse_idx TEXT, sparse_val TEXT)")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        """(dense, sparse_indices, sparse_values) or None."""
        row = self.db.execute("SELECT dense, sparse_idx, sparse_val FROM emb WHERE key = ?",
                              (self.key(text),)).fetchone()
        if not row:
            return None
        dense = array("f")
        dense.frombytes(row[0])
        return list(dense), json.loads(row[1]), json.loads(row[2])

    def put_many(self, items) -> None:
        """items: iterable of (text, dense, sparse_indices, sparse_values)."""
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?)",
            [(self.key(t), array("f", d).tobytes(), json.dumps(list(i)), json.dumps(list(v)))
             for t, d, i, v in items])
        self.db.commit()

    def size(self) -> int:
        return self.db.execute("SELECT count(*) FROM emb").fetchone()[0]


def embed_text_of(payload: dict) -> str:
    """Rebuilds the exact text that was embedded (see scripts/build_chunks.py)."""
    return f"{payload.get('title')} | {payload.get('section')}\n{payload.get('text')}"


def seed_from_qdrant(client, cache: EmbeddingCache, collection: str) -> int:
    """Copies the vectors already stored in a Qdrant collection into the cache."""
    if not client.collection_exists(collection):
        return 0
    added, offset = 0, None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset,
                                       with_payload=True, with_vectors=True)
        items = []
        for p in points:
            vec = p.vector or {}
            sp = vec.get("sparse")
            if "dense" in vec and sp is not None:
                items.append((embed_text_of(p.payload or {}), vec["dense"], sp.indices, sp.values))
        cache.put_many(items)
        added += len(items)
        if offset is None:
            return added