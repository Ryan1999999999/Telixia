"""PDF -> ordered list of blocks (text paragraphs and markdown tables), page by page.

Why not page.get_text("text")?  On price grids it outputs one table cell per line
("25,96 EUR/mois" / "jusqu'au" / "17/01/2027"), which destroys the row/column link.
Here tables are detected with PyMuPDF and serialized as markdown (header kept),
and text that lies inside a table area is not extracted twice.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SPACES = {"\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ", "\u200b": "", "\ufeff": ""}
QUOTES = {"\u2019": "'", "\u2018": "'", "\u00ab": '"', "\u00bb": '"', "\u201c": '"', "\u201d": '"'}
PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,3}\s*(/\s*\d{1,3})?\s*$", re.I)


@dataclass
class Block:
    page: int          # 1-based page number
    kind: str          # "text" | "table"
    text: str


def normalize(text: str) -> str:
    """Strict UTF-8 normalization: NFC, non-breaking spaces, typographic quotes, spaces."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in {**SPACES, **QUOTES}.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _clean_cell(cell) -> str:
    return normalize(str(cell)).replace("\n", " ").replace("|", "/") if cell else ""


def table_to_markdown(rows: list[list]) -> str:
    rows = [[_clean_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _cell_text(page, bbox) -> str:
    """Text of one cell, read with the normal text extractor limited to the cell area.

    table.extract() rebuilds cells character by character and, on these PDFs, loses the
    spaces and reorders characters ("4299EUR ," instead of "42,99 EUR"). Clipped text
    extraction keeps the words and numbers intact.
    """
    if bbox is None:
        return ""
    import pymupdf
    text = page.get_text("text", clip=pymupdf.Rect(bbox))
    return " ".join(normalize(text).split()).replace("|", "/")


def _table_rows(page, table) -> list[list[str]]:
    try:
        rows = [[_cell_text(page, cell) for cell in row.cells] for row in table.rows]
        if any(any(r) for r in rows):
            return rows
    except Exception:
        pass
    return table.extract()                           # fallback: PyMuPDF's own cell text


def _inside(bbox, area, tol: float = 2.0) -> bool:
    x0, y0, x1, y1 = bbox
    ax0, ay0, ax1, ay1 = area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return ax0 - tol <= cx <= ax1 + tol and ay0 - tol <= cy <= ay1 + tol


def extract_blocks(pdf_path) -> tuple[list[Block], int]:
    """Returns (blocks in reading order, number of pages)."""
    import pymupdf  # imported here so the pure-Python parts can be tested without it

    blocks: list[Block] = []
    with pymupdf.open(pdf_path) as doc:
        n_pages = doc.page_count
        for page in doc:
            items: list[tuple[float, float, Block]] = []
            table_areas = []
            try:
                for t in page.find_tables().tables:
                    rows = _table_rows(page, t)
                    n_cols = max((len(r) for r in rows), default=0)
                    if len(rows) < 2 or n_cols < 2:
                        continue                      # a framed paragraph, not a real table
                    md = table_to_markdown(rows)
                    if md:
                        table_areas.append(tuple(t.bbox))
                        items.append((t.bbox[1], t.bbox[0], Block(page.number + 1, "table", md)))
            except Exception:
                pass                                  # table detection failure: text only

            for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
                if btype != 0:                        # 1 = image block
                    continue
                if any(_inside((x0, y0, x1, y1), a) for a in table_areas):
                    continue                          # already inside a markdown table
                lines = [normalize(line) for line in text.splitlines()]
                lines = [ln for ln in lines if ln and not PAGE_NUMBER.match(ln)]
                if lines:
                    # keep line breaks for now: heading detection works line by line
                    items.append((y0, x0, Block(page.number + 1, "text", "\n".join(lines))))

            items.sort(key=lambda it: (round(it[0], 0), it[1]))
            blocks.extend(b for _, _, b in items)
    return blocks, n_pages