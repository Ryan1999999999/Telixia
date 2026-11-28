"""RAG pipeline (session 3): question -> retrieval -> reranking -> grounded answer.

    question
      -> BGE-M3 encoding
      -> first stage: dense top-N (RETRIEVAL_LANES, RERANK_CANDIDATES in .env)
      -> cross-encoder reranking -> top 5 child chunks
      -> grouped by parent section (no duplicate text), parent text sent to the LLM
      -> LLM answer with [n] citations, or the fixed refusal sentence
      -> compliance check (citations valid? refusal?) + latency of every stage

Children (~256 tokens) are what we search: small, precise vectors. Parents (<= 1024
tokens, the whole section) are what the LLM reads: enough context to answer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..ingestion.quality import detect_language
from .llm import LLMClient
from .prompts import check_answer, system_prompt, user_prompt

ROOT = Path(__file__).resolve().parents[3]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
PARENTS = ROOT / "data" / "processed" / "parents.jsonl"


@dataclass
class RagAnswer:
    question: str
    language: str
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    check: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    usage: object = None


def load_parents(path: Path = PARENTS) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {p["parent_id"]: p for p in map(json.loads, f)}


def build_sources(reranked: list, parents: dict[str, dict], max_sources: int = 5) -> list[dict]:
    """One numbered source per parent section, in reranking order.

    Two children of the same section become one source (the LLM does not read the same
    text twice). The parent text is used when available, otherwise the child texts.
    """
    groups: dict[str, dict] = {}
    for r in reranked:
        pl = r.candidate.payload
        key = pl.get("parent_id") or r.candidate.id
        if key not in groups:
            if len(groups) == max_sources:
                continue
            parent = parents.get(key)
            groups[key] = {
                "n": len(groups) + 1, "parent_id": key, "doc_id": pl.get("doc_id"),
                "title": pl.get("title"), "section": pl.get("section"),
                "page": pl.get("page"), "source_url": pl.get("source_url"),
                "rerank_score": round(float(r.score), 4), "chunk_ids": [],
                "text": parent["text"] if parent else "", "_children": [],
                "from_parent": parent is not None,
            }
        g = groups[key]
        g["chunk_ids"].append(r.candidate.id)
        g["_children"].append(pl.get("text", ""))
    sources = []
    for g in groups.values():
        children = g.pop("_children")
        if not g["text"]:
            g["text"] = "\n...\n".join(children)
        sources.append(g)
    return sources


class RagPipeline:
    def __init__(self, lanes: tuple[str, ...] | None = None, candidates: int | None = None,
                 top_k: int = 5, reranker_model: str | None = None, llm: LLMClient | None = None):
        from qdrant_client import QdrantClient

        from ..core.config import get_settings
        from ..core.embeddings import Embedder
        from ..retrieval.bm25 import BM25Index
        from ..retrieval.hybrid import HybridRetriever
        from ..retrieval.reranker import DEFAULT_MODEL, CrossEncoderReranker

        settings = get_settings()
        self.lanes = lanes or tuple(os.getenv("RETRIEVAL_LANES", "dense").replace(" ", "").split(","))
        self.candidates = candidates or int(os.getenv("RERANK_CANDIDATES", "10"))
        self.top_k = top_k
        client = QdrantClient(url=settings.qdrant_url,
                              api_key=settings.qdrant_api_key.get_secret_value(), timeout=30)
        self.embedder = Embedder(settings.embedding_model)
        self.retriever = HybridRetriever(client, self.embedder, BM25Index.from_jsonl(CHUNKS))
        self.reranker = CrossEncoderReranker(reranker_model or DEFAULT_MODEL)
        self.parents = load_parents()
        self.llm = llm or LLMClient()
        self.last: RagAnswer | None = None
        # warm-up: the first call of each model is much slower
        self.embedder.encode(["warm-up"], batch_size=1)
        self.reranker.score("warm-up", ["warm-up"])

    def retrieve(self, question: str) -> tuple[list[dict], dict[str, float]]:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        enc = self.embedder.encode([question], batch_size=1)[0]
        timings["encode"] = (time.perf_counter() - t0) * 1000
        res = self.retriever.search(question, top_n=self.candidates, lanes=self.lanes, encoded=enc)
        timings.update({k: v for k, v in res.timings.items() if k != "encode"})
        reranked = self.reranker.rerank(question, res.candidates, top_k=self.top_k)
        timings["rerank"] = self.reranker.last_ms
        return build_sources(reranked, self.parents), timings

    def stream(self, question: str) -> Iterator[str]:
        """Yields the answer piece by piece; the full result is in self.last at the end."""
        t_start = time.perf_counter()
        lang = detect_language(question, default="fr")
        result = RagAnswer(question, lang)
        result.sources, result.timings = self.retrieve(question)
        messages = [{"role": "system", "content": system_prompt(lang)},
                    {"role": "user", "content": user_prompt(question, result.sources)}]
        t_llm = time.perf_counter()
        pieces = []
        for piece in self.llm.stream(messages):
            if not pieces:
                result.timings["ttft"] = (time.perf_counter() - t_llm) * 1000
            pieces.append(piece)
            yield piece
        result.timings["generation"] = (time.perf_counter() - t_llm) * 1000
        result.timings["total"] = (time.perf_counter() - t_start) * 1000
        result.answer = "".join(pieces).strip()
        result.check = check_answer(result.answer, len(result.sources))
        if getattr(self.llm, "last_finish", None) == "length" and result.check["status"] != "refusal":
            result.check["status"] = "truncated"
        result.usage = self.llm.last_usage
        self.last = result

    def ask(self, question: str) -> RagAnswer:
        for _ in self.stream(question):
            pass
        return self.last