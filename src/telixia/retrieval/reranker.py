"""Cross-encoder reranking: second stage of the retrieval pipeline.

Bi-encoder (BGE-M3, first stage): question and chunk are encoded *separately*, then
compared with a cosine. Fast (one vector per chunk, computed at ingestion) but coarse.

Cross-encoder (this module): question and chunk are read *together* by one transformer
(joint attention over both texts) which outputs one relevance score. Much more precise
(it sees that "sans engagement ?" is answered by the "Duree" section, not "Services"),
but it must run once per (question, chunk) pair at query time, so it is only applied
to the top-N candidates of the first stage.

Each candidate is scored with its context header "title | section" so the model can
tell apart the 64 contract summaries that share the same sentences.

Supported models (both multilingual, French included):
  BAAI/bge-reranker-v2-m3                       568M params, best quality (FlagEmbedding)
  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1    118M params, ~5x faster (sentence-transformers)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:                                   # reranker settings live in .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


@dataclass
class Reranked:
    candidate: object          # the first-stage candidate (payload available)
    score: float               # cross-encoder relevance (higher = better)
    first_stage_rank: int      # its rank before reranking (1 = best)


def candidate_text(payload: dict) -> str:
    """Same text as at ingestion: context header + chunk."""
    return f"{payload.get('title')} | {payload.get('section')}\n{payload.get('text', '')}"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512, batch_size: int = 16,
                 int8: bool | None = None):
        self.model_name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self.last_ms = 0.0
        if "bge-reranker" in model_name.lower():
            from FlagEmbedding import FlagReranker
            self._flag = FlagReranker(model_name, use_fp16=False)    # fp16 only helps on GPU
            self._st = None
        else:
            from sentence_transformers import CrossEncoder
            self._st = CrossEncoder(model_name, max_length=max_length)
            self._flag = None
        model = self._flag.model if self._flag is not None else self._st.model
        self.n_params = sum(p.numel() for p in model.parameters())     # counted before quantization
        self.int8 = (os.getenv("RERANKER_INT8", "0") == "1") if int8 is None else int8
        if self.int8:          # dynamic int8 quantization of the Linear layers: CPU speed-up
            import torch
            # in place: the Linear layers are swapped inside the existing model, so the
            # sentence-transformers / FlagEmbedding wrappers around it stay untouched
            torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        t0 = time.perf_counter()
        if self._flag is not None:
            scores = self._flag.compute_score(pairs, normalize=True, max_length=self.max_length,
                                              batch_size=self.batch_size)
            scores = [scores] if isinstance(scores, float) else list(scores)
        else:
            scores = [float(s) for s in self._st.predict(pairs, batch_size=self.batch_size)]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: list, top_k: int = 5,
               payload_of=lambda c: c.payload) -> list[Reranked]:
        """Re-sorts first-stage candidates by cross-encoder score; keeps the best top_k."""
        scores = self.score(query, [candidate_text(payload_of(c)) for c in candidates])
        ranked = sorted(
            (Reranked(c, s, i + 1) for i, (c, s) in enumerate(zip(candidates, scores))),
            key=lambda r: -r.score)
        return ranked[:top_k]