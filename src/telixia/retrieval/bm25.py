"""BM25 (Okapi) lexical index over the chunk files, implemented from scratch.

Why a lexical lane: dense vectors smooth out exact tokens ("5G+", "Mobicarte",
"42,99", "Article 5"). BM25 scores documents on exact term overlap, weighted by
how rare each term is (IDF) and normalized by document length.

    score(d, q) = sum over terms t of q:
        IDF(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

k1 = 1.5 (term-frequency saturation) and b = 0.75 (length normalization) are the
standard values. The index is built in memory from data/processed/chunks.jsonl at
startup (about a second for ~3,600 chunks) and searched with an inverted index.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {
    # French
    "le", "la", "les", "l", "de", "des", "du", "d", "et", "en", "un", "une", "pour", "par",
    "sur", "au", "aux", "dans", "est", "sont", "que", "qui", "quoi", "ou", "vous", "votre",
    "vos", "nous", "notre", "il", "elle", "ils", "ce", "cet", "cette", "ces", "se", "sa", "son",
    "ses", "leur", "leurs", "avec", "sans", "pas", "ne", "plus", "y", "a", "quel", "quelle",
    "quels", "quelles", "comment", "combien", "est-il", "t", "s", "qu", "c", "j", "m", "n",
    # English
    "the", "of", "and", "to", "in", "is", "for", "you", "your", "on", "with", "are", "be",
    "this", "by", "or", "at", "from", "as", "an", "what", "how", "do", "does", "i", "my", "it",
    "can", "which", "when", "after", "many",
}
# words and numbers; a trailing "+" is kept so that "5g+" and "5g" stay different terms
TOKEN = re.compile(r"[a-z0-9]+\+?")


def tokenize(text: str) -> list[str]:
    """Lowercase, remove accents, split on non-alphanumerics, drop stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in TOKEN.findall(text)
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    @classmethod
    def from_jsonl(cls, path: Path, text_field: str = "embed_text", **kw) -> "BM25Index":
        """Indexes the same text that was embedded (title | section + chunk text)."""
        index = cls(**kw)
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    payload = {k: v for k, v in c.items() if k not in ("embed_text", "id")}
                    index.add(c["id"], c.get(text_field) or c["text"], payload)
        index.finalize()
        return index

    def add(self, doc_id: str, text: str, payload: dict) -> None:
        tokens = tokenize(text)
        idx = len(self.ids)
        self.ids.append(doc_id)
        self.payloads.append(payload)
        self.doc_len.append(len(tokens))
        for term, tf in Counter(tokens).items():
            self.postings[term].append((idx, tf))

    def finalize(self) -> None:
        n = len(self.ids)
        self.avgdl = sum(self.doc_len) / max(n, 1)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def _allowed(self, idx: int, where: dict | None) -> bool:
        if not where:
            return True
        pl = self.payloads[idx]
        return all(pl.get(k) == v for k, v in where.items())

    def search(self, query: str, limit: int = 20, where: dict | None = None) -> list[tuple[str, float]]:
        """Top documents as (id, score). `where` = exact payload matches, e.g. {"doc_type": "RC"}."""
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl or 1.0
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in self.postings[term]:
                norm = k1 * (1 - b + b * self.doc_len[idx] / avgdl)
                scores[idx] += idf * tf * (k1 + 1) / (tf + norm)
        ranked = sorted((i for i in scores if self._allowed(i, where)), key=lambda i: -scores[i])
        return [(self.ids[i], scores[i]) for i in ranked[:limit]]

    def payload(self, doc_id: str) -> dict:
        if not hasattr(self, "_pos"):
            self._pos = {d: i for i, d in enumerate(self.ids)}
        return self.payloads[self._pos[doc_id]]

    def __len__(self) -> int:
        return len(self.ids)