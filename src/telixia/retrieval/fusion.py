"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over methods m of  1 / (k + rank_m(d))

Why ranks and not scores: a cosine score lives in [0, 1] while a BM25 score is
unbounded (0 to +inf); adding them directly would let BM25 dominate. RRF only uses
positions, so no normalization is needed.

Why k = 60: k smooths the gap between top ranks. With k = 0, rank 1 is worth twice
rank 2 (1 vs 0.5) and one method alone decides. With k = 60, rank 1 vs rank 2 is
1/61 vs 1/62: a document ranked well by *both* lanes beats a document ranked first
by only one. 60 is the value from the original paper, robust across datasets
(checked on our gold set by eval/benchmark_hybrid.py --k 10 60 100).
"""

from __future__ import annotations

from collections import defaultdict


def rrf(rankings: dict[str, list[str]], k: int = 60, limit: int | None = None
        ) -> list[tuple[str, float, dict[str, int]]]:
    """rankings: {method: [doc ids, best first]} ->
    [(doc_id, rrf_score, {method: rank}), ...] sorted by rrf_score."""
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for method, ids in rankings.items():
        for rank, doc_id in enumerate(ids, 1):
            if method in ranks[doc_id]:          # duplicate id in one list: keep the best rank
                continue
            scores[doc_id] += 1.0 / (k + rank)
            ranks[doc_id][method] = rank
    ordered = sorted(scores, key=lambda d: (-scores[d], min(ranks[d].values())))
    if limit:
        ordered = ordered[:limit]
    return [(d, scores[d], ranks[d]) for d in ordered]