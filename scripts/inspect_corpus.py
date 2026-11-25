"""Analyses the PDF corpus before chunking.

    python scripts/inspect_corpus.py

For each document in manifest.csv: pages, words, empty pages (scanned?),
tables detected, "Article" headings. Writes:
    data/processed/corpus_stats.csv           one row per document
    data/processed/samples/<doc_type>.txt     text of one example per type
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "docs" / "manifest.csv"
OUT = ROOT / "data" / "processed"
SAMPLES = OUT / "samples"
ARTICLE = re.compile(r"^\s*(article|art\.)\s*\d+", re.I | re.M)


def analyse(pdf_path: Path) -> tuple[dict, str]:
    doc = fitz.open(pdf_path)
    words = empty = tables = articles = 0
    text_parts = []
    for page in doc:
        text = page.get_text("text")
        n = len(text.split())
        words += n
        if n < 20:
            empty += 1                 # almost no text: image / scanned / cover page
        try:
            tables += len(page.find_tables().tables)
        except Exception:
            pass
        articles += len(ARTICLE.findall(text))
        text_parts.append(f"\n===== page {page.number + 1} =====\n{text}")
    stats = {"pages": doc.page_count, "words": words, "empty_pages": empty,
             "tables": tables, "articles": articles}
    doc.close()
    return stats, "".join(text_parts)


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(encoding="utf-8") as f:
        docs = list(csv.DictReader(f))

    rows, by_type, sampled = [], defaultdict(list), set()
    for d in docs:
        path = ROOT / d["file_path"]
        if not path.exists():
            print(f"[missing] {d['doc_id']}")
            continue
        try:
            stats, text = analyse(path)
        except Exception as e:
            print(f"[error] {d['doc_id']}: {e}")
            continue
        rows.append({"doc_id": d["doc_id"], "doc_type": d["doc_type"], **stats})
        by_type[d["doc_type"]].append(stats)
        if d["doc_type"] not in sampled:          # keep the first example of each type
            (SAMPLES / f"{d['doc_type']}.txt").write_text(
                f"{d['title']}\n{d['file_path']}\n{text}", encoding="utf-8")
            sampled.add(d["doc_type"])

    with (OUT / "corpus_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} documents analysed\n")
    print(f"{'type':<11}{'docs':>5}{'pages':>7}{'avg p.':>8}{'words':>9}{'avg w.':>8}"
          f"{'tables':>8}{'articles':>10}{'empty p.':>10}")
    total = defaultdict(int)
    for t, lst in sorted(by_type.items(), key=lambda x: -len(x[1])):
        p = sum(s["pages"] for s in lst)
        w = sum(s["words"] for s in lst)
        tb = sum(s["tables"] for s in lst)
        ar = sum(s["articles"] for s in lst)
        em = sum(s["empty_pages"] for s in lst)
        print(f"{t:<11}{len(lst):>5}{p:>7}{p/len(lst):>8.1f}{w:>9}{w/len(lst):>8.0f}"
              f"{tb:>8}{ar:>10}{em:>10}")
        for k, v in (("docs", len(lst)), ("pages", p), ("words", w)):
            total[k] += v
    print(f"\nTOTAL: {total['docs']} docs, {total['pages']} pages, {total['words']:,} words"
          f" (~{int(total['words'] * 1.4):,} tokens)")
    print(f"Details: {(OUT / 'corpus_stats.csv').relative_to(ROOT)}")
    print(f"Samples: {SAMPLES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()