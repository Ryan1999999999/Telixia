"""Chunk quality filter and language detection.

Why: some PDFs use fonts without a Unicode mapping, so their text extracts as symbols
("~o<inf>Y..."), and a few chunks are empty fill-in lines (". . . .") or stray labels.
Such chunks have meaningless vectors that still come out on top for vague queries.
They are dropped from the search index (the parent sections keep the full text).

Rules (a chunk is dropped when one applies):
  1. fewer than 3 readable words                      -> empty / gibberish / stray label
  2. more than 10% of its letters are not FR/EN letters -> broken font encoding
  3. text chunk (not a table) of 30+ words with no common FR or EN word -> gibberish
Tables are exempt from rule 3: a price grid naturally has almost no "le/la/de/the".
"""

from __future__ import annotations

FR_EN_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
FR_EN_CHARS |= set("\u00e0\u00e2\u00e4\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00ff\u0153\u00e6")
FR_EN_CHARS |= set("\u00c0\u00c2\u00c4\u00c7\u00c9\u00c8\u00ca\u00cb\u00ce\u00cf\u00d4\u00d6\u00d9\u00db\u00dc\u0152\u00c6")
LETTERS = FR_EN_CHARS - set("0123456789")

FR_STOP = {"le", "la", "les", "de", "des", "du", "et", "en", "un", "une", "pour", "par", "sur",
           "au", "aux", "dans", "est", "que", "qui", "ou", "vous", "votre", "avec", "sont", "pas"}
EN_STOP = {"the", "of", "and", "to", "a", "in", "is", "for", "you", "your", "on", "with", "are",
           "be", "this", "by", "or", "at", "from", "as", "not", "will", "any"}


def _words(text: str) -> list[str]:
    return [w.strip(".,;:()'\"!?").lower() for w in text.split()]


def low_quality_reason(text: str, chunk_type: str = "text") -> str | None:
    """Returns why the chunk should be dropped, or None if it is fine."""
    alnum = [c for c in text if c.isalnum()]
    weird = sum(1 for c in alnum if c not in FR_EN_CHARS) / max(len(alnum), 1)
    words = _words(text)
    readable = [w for w in words if len(w) >= 2 and all(c in LETTERS or c in "-'" for c in w)]
    if len(readable) < 3:
        return "too little readable text"
    if weird > 0.10:
        return "non FR/EN letters"
    if chunk_type != "table" and len(words) > 30:
        common = sum(1 for w in words if w in FR_STOP or w in EN_STOP) / len(words)
        if common < 0.02:
            return "no common words"
    return None


def detect_language(text: str, default: str = "fr") -> str:
    """'fr' or 'en', by counting very common words (enough for contract documents)."""
    words = _words(text)
    fr = sum(1 for w in words if w in FR_STOP)
    en = sum(1 for w in words if w in EN_STOP)
    if fr == en == 0:
        return default
    return "en" if en > fr else "fr"