"""Structure-aware Parent-Document chunking.

Strategy (one per document family, justified by scripts/inspect_corpus.py):

  family   doc types                      parent (sent to the LLM)        children (searched)
  -------  -----------------------------  ------------------------------  ---------------------------
  legal    CGA, CGV, CGU, CS, CG, ANNEXE  one "Article N" section          ~256-token windows,
  rc       RC (EU contract summary)       one EU template section          sentence-aligned, overlap
  paged    TARIFS, ASSURANCE, others      one page                         tables split by rows,
                                                                           header repeated

Parents are capped (~1024 tokens) and split on paragraph boundaries if longer.
Every child is embedded with a context header "title | section" so that the 64
contract summaries, which all share the same section names ("Prix"...), stay
distinguishable in vector space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .pdf_extract import Block

LEGAL = {"CGA", "CGV", "CGU", "CS", "CG", "ANNEXE"}
RC = {"RC"}

ARTICLE_HEADING = re.compile(r"^(article|art\.)\s*(\d+(?:\.\d+)*)\s*[.\-\u2013:)]?\s*(.*)$", re.I)
RC_HEADING = re.compile(
    r"^(services et \u00e9quipements|\u00e9quipements\b.*|d\u00e9bits? du service.*|prix\b.*"
    r"|dur\u00e9e,? renouvellement et r\u00e9siliation.*|fonctionnalit\u00e9s pour les utilisateurs finaux.*"
    r"|autres informations utiles.*)$",
    re.I,
)
DATE = re.compile(
    r"(?:actualis[\u00e9e]e? le|mis[e]? \u00e0 jour le|en vigueur (?:au|le|\u00e0 compter du)|applicables? (?:au|\u00e0 compter du))"
    r"\s*:?\s*(\d{1,2})[/.](\d{1,2})[/.](\d{4})",
    re.I,
)
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z\u00c0-\u00d6\u00d8-\u00dd0-9\u25a0\u2022\-(\"])")


def family_of(doc_type: str) -> str:
    if doc_type in LEGAL:
        return "legal"
    if doc_type in RC:
        return "rc"
    return "paged"


def find_publication_date(blocks: list[Block]) -> str | None:
    for b in blocks[:40]:                      # the date is printed near the top
        m = DATE.search(b.text.replace("\n", " "))
        if m:
            d, mth, y = m.groups()
            return f"{y}-{int(mth):02d}-{int(d):02d}"
    return None


@dataclass
class Section:
    title: str
    page_start: int
    page_end: int
    units: list[tuple[str, str]] = field(default_factory=list)   # (kind, text)


def _is_heading(line: str, family: str) -> str | None:
    """Returns the section title if the line starts a new section."""
    if len(line) > 160:
        return None
    if family == "legal":
        m = ARTICLE_HEADING.match(line)
        if m:
            return f"Article {m.group(2)}. {m.group(3)}".strip().rstrip(".")
    if family == "rc" and len(line.split()) <= 12 and RC_HEADING.match(line):
        return line.rstrip(" :")
    return None


def split_sections(blocks: list[Block], family: str) -> list[Section]:
    sections: list[Section] = []
    current: Section | None = None

    def start(title: str, page: int) -> Section:
        s = Section(title=title, page_start=page, page_end=page)
        sections.append(s)
        return s

    for b in blocks:
        if family == "paged" and (current is None or b.page != current.page_start):
            current = start(f"Page {b.page}", b.page)
        if current is None:
            current = start("Pr\u00e9ambule", b.page)

        if b.kind == "table":
            current.units.append(("table", b.text))
            current.page_end = b.page
            continue

        paragraph: list[str] = []
        just_started = False
        for line in b.text.split("\n"):
            title = _is_heading(line, family)
            if title:
                if paragraph:
                    current.units.append(("text", " ".join(paragraph)))
                    paragraph = []
                current = start(title, b.page)
                just_started = True
            elif just_started and len(line.split()) <= 3 and not line.endswith((".", ":")):
                current.title = f"{current.title} {line}"     # heading wrapped on 2 lines
                just_started = False
            else:
                paragraph.append(line)
                just_started = False
        if paragraph:
            current.units.append(("text", " ".join(paragraph)))
        current.page_end = b.page

    return [s for s in sections if s.units]


# ---------------------------------------------------------------------------
# packing helpers
# ---------------------------------------------------------------------------

def _split_long_text(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Paragraph -> sentences -> word windows, each <= budget tokens."""
    if count(text) <= budget:
        return [text]
    pieces: list[str] = []
    for sent in SENTENCE_END.split(text):
        if count(sent) <= budget:
            pieces.append(sent)
            continue
        words, buf = sent.split(), []
        for w in words:
            if buf and count(" ".join(buf + [w])) > budget:
                pieces.append(" ".join(buf))
                buf = []
            buf.append(w)
        if buf:
            pieces.append(" ".join(buf))
    return pieces


def _split_table(md: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Split a markdown table by rows, repeating the header in every piece."""
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]
    if count(md) <= budget or not rows:
        return [md]
    pieces, buf = [], []
    for row in rows:
        if buf and count("\n".join(header + buf + [row])) > budget:
            pieces.append("\n".join(header + buf))
            buf = []
        buf.append(row)
    if buf:
        pieces.append("\n".join(header + buf))
    return pieces


def pack(units: list[tuple[str, str]], budget: int, overlap: int,
         count: Callable[[str], int]) -> list[tuple[str, str]]:
    """Greedy packing of (kind, text) units into chunks <= budget tokens.

    Text chunks overlap by their last sentences (<= overlap tokens); tables never overlap
    and are never mixed with text, so a table chunk stays a clean markdown table.
    """
    atoms: list[tuple[str, str]] = []
    for kind, text in units:
        if kind == "table":
            atoms += [("table", t) for t in _split_table(text, budget, count)]
        else:
            atoms += [("text", t) for t in _split_long_text(text, budget, count)]

    chunks: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append(("text", " ".join(buf)))

    for kind, text in atoms:
        if kind == "table":
            flush()
            buf = []
            chunks.append(("table", text))
            continue
        if buf and count(" ".join(buf + [text])) > budget:
            flush()
            tail: list[str] = []                       # overlap: last sentences of previous chunk
            for sent in reversed(SENTENCE_END.split(" ".join(buf))):
                if count(" ".join([sent] + tail)) > overlap:
                    break
                tail.insert(0, sent)
            buf = tail
        buf.append(text)
    flush()
    return chunks


def make_parents(section: Section, parent_budget: int,
                 count: Callable[[str], int]) -> list[list[tuple[str, str]]]:
    """Split a section into one or more parents (lists of units) of <= parent_budget tokens."""
    units: list[tuple[str, str]] = []
    for kind, text in section.units:                   # a single huge paragraph/table is split first
        splitter = _split_table if kind == "table" else _split_long_text
        units += [(kind, t) for t in splitter(text, parent_budget, count)]
    parents, buf, size = [], [], 0
    for kind, text in units:
        n = count(text)
        if buf and size + n > parent_budget:
            parents.append(buf)
            buf, size = [], 0
        buf.append((kind, text))
        size += n
    if buf:
        parents.append(buf)
    return parents


def units_to_text(units: list[tuple[str, str]]) -> str:
    return "\n\n".join(text for _, text in units)