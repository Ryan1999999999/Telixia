"""Hybrid retriever: dense lane (Qdrant, BGE-M3) + lexical lane (BM25) fused with RRF.

Optional third lane: BGE-M3 learned sparse vectors stored in Qdrant ("sparse").

    q --> encode (BGE-M3) --> dense top-20 (Qdrant HNSW) ---+
    |                                                       +--> RRF (k=60) --> top-N candidates
    +-----------------------> BM25 top-20 (in memory) ------+       (then: cross-encoder rerank)

Every stage is timed (ms) for the latency budget of the evaluation sheet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..ingestion.quality import detect_language
from .bm25 import BM25Index
from .fusion import rrf

LANES = ("dense", "bm25", "sparse")


@dataclass
class Candidate:
    id: str
    payload: dict
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)    # rank in each lane (1 = best)


@dataclass
class HybridResult:
    candidates: list[Candidate]                 # fused list, best first
    rankings: dict[str, list[str]]              # each lane's ids, best first
    payloads: dict[str, dict]                   # id -> payload
    timings: dict[str, float]                   # stage -> ms


def qdrant_filter(where: dict | None):
    if not where:
        return None
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                               for k, v in where.items()])


class HybridRetriever:
    def __init__(self, client, embedder, bm25: BM25Index, collection: str = "telixia_docs",
                 k: int = 60, lane_limit: int = 20, corpus_language: str = "fr",
                 lang_aware: bool = True):
        self.client, self.embedder, self.bm25 = client, embedder, bm25
        self.collection, self.k, self.lane_limit = collection, k, lane_limit
        # BM25 matches exact words, so it is monolingual: an English question only matches the
        # few English chunks and adds noise. When the question is not in the corpus language,
        # the lexical lane is skipped (the dense lane is cross-lingual).
        self.corpus_language, self.lang_aware = corpus_language, lang_aware
        self.last_skipped: list[str] = []

    def use_bm25(self, query: str) -> bool:
        return not self.lang_aware or detect_language(query, self.corpus_language) == self.corpus_language

    def lane_rankings(self, query: str, where: dict | None = None,
                      lanes: tuple[str, ...] = ("dense", "bm25"), encoded=None
                      ) -> tuple[dict[str, list[str]], dict[str, dict], dict[str, float]]:
        """Runs each lane: ({lane: [ids]}, {id: payload}, {stage: ms})."""
        from qdrant_client import models

        timings: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        rankings: dict[str, list[str]] = {}
        flt = qdrant_filter(where)

        if any(lane in lanes for lane in ("dense", "sparse")):
            t0 = time.perf_counter()
            enc = encoded or self.embedder.encode([query], batch_size=1)[0]
            timings["encode"] = (time.perf_counter() - t0) * 1000 if encoded is None else 0.0

        for lane in ("dense", "sparse"):
            if lane not in lanes:
                continue
            vector = enc.dense if lane == "dense" else models.SparseVector(
                indices=enc.sparse_indices, values=enc.sparse_values)
            t0 = time.perf_counter()
            res = self.client.query_points(self.collection, query=vector, using=lane,
                                           query_filter=flt, limit=self.lane_limit, with_payload=True)
            timings[lane] = (time.perf_counter() - t0) * 1000
            rankings[lane] = [str(p.id) for p in res.points]
            payloads.update({str(p.id): p.payload or {} for p in res.points})

        self.last_skipped = []
        if "bm25" in lanes and not self.use_bm25(query):
            self.last_skipped.append("bm25")
        elif "bm25" in lanes:
            t0 = time.perf_counter()
            hits = self.bm25.search(query, limit=self.lane_limit, where=where)
            timings["bm25"] = (time.perf_counter() - t0) * 1000
            rankings["bm25"] = [doc_id for doc_id, _ in hits]
            for doc_id, _ in hits:
                payloads.setdefault(doc_id, self.bm25.payload(doc_id))
        return rankings, payloads, timings

    def search(self, query: str, top_n: int = 20, where: dict | None = None,
               lanes: tuple[str, ...] = ("dense", "bm25"), k: int | None = None, encoded=None
               ) -> HybridResult:
        rankings, payloads, timings = self.lane_rankings(query, where, lanes, encoded)
        t0 = time.perf_counter()
        fused = rrf(rankings, k=k or self.k, limit=top_n)
        timings["rrf"] = (time.perf_counter() - t0) * 1000
        cands = [Candidate(d, payloads[d], s, r) for d, s, r in fused]
        return HybridResult(cands, rankings, payloads, timings)