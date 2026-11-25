"""BGE-M3 embedder: one model, two representations.

- dense  : 1024-dim normalized vector (semantic similarity, Cosine)
- sparse : lexical weights {token_id: weight} (exact terms: "5G+", "Mobicarte", "42,99")
Used by the ingestion (documents) and by the search (queries), so both sides are
encoded with exactly the same model and settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Encoded:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", max_length: int = 1024):
        from FlagEmbedding import BGEM3FlagModel
        import torch

        self.use_fp16 = torch.cuda.is_available()          # fp16 only makes sense on GPU
        self.model = BGEM3FlagModel(model_name, use_fp16=self.use_fp16)
        self.max_length = max_length
        self.device = "cuda" if self.use_fp16 else "cpu"

    def encode(self, texts: list[str], batch_size: int = 8) -> list[Encoded]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        result = []
        for dense, weights in zip(out["dense_vecs"], out["lexical_weights"]):
            items = sorted((int(k), float(v)) for k, v in weights.items() if float(v) > 0)
            result.append(Encoded(
                dense=[float(x) for x in dense],
                sparse_indices=[k for k, _ in items],
                sparse_values=[v for _, v in items],
            ))
        return result