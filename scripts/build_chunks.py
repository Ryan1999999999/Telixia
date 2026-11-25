"""Builds parent and child chunks for the whole corpus (no embeddings yet).

    python scripts/build_chunks.py                 # full corpus
    python scripts/build_chunks.py --limit 5       # quick test on 5 documents

Output:
    data/processed/parents.jsonl   parent sections (sent to the LLM)
    data/processed/chunks.jsonl    child chunks + full payload (searched in Qdrant)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telixia.ingestion.chunking import (  # noqa: E402
    family_of, find_publication_date, make_parents, pack, split_sections, units_to_text,
)
from telixia.ingestion.pdf_extract import extract_blocks  # noqa: E402

MANIFEST = ROOT / "data" / "raw" / "docs" / "manifest.csv"
OUT = ROOT / "data" / "processed"
NAMESPACE = uuid.UUID("6f1c2d3e-0000-4000-8000-7e11c1a00001")   # fixed: same chunk -> same UUID

CHILD_TOKENS = 256
OVERLAP_TOKENS = 40
PARENT_TOKENS = 1024
BOILERPLATE_MIN_WORDS = 12          # short blocks (headings like "Prix") are never removed
BOILERPLATE_DOC_SHARE = 0.3         # block found in >= 30% of the docs of a type -> boilerplate

CATEGORY = {"TARIFS": "tarif", "ASSURANCE": "assurance", "FORMULAIRE": "formulaire"}


def get_counter(model_name: str):
    """Token counter using the real BGE-M3 tokenizer (falls back to a word estimate)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"Token counting with the {model_name} tokenizer")
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({e}); using words x 1.4")
        return lambda text: int(len(text.split()) * 1.4) + 1


def signature(text: str) -> str:
    """Block fingerprint: lowercase, digits removed (dates/page numbers vary)."""
    return hashlib.md5(re.sub(r"[\d\s]+", " ", text.lower()).strip().encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    count = get_counter(args.model)
    with MANIFEST.open(encoding="utf-8") as f:
        docs = list(csv.DictReader(f))
    if args.limit:
        docs = docs[: args.limit]

    # ---- pass 1: extraction --------------------------------------------------------
    extracted = {}
    for d in docs:
        path = ROOT / d["file_path"]
        if not path.exists():
            print(f"[missing] {d['doc_id']}")
            continue
        try:
            blocks, n_pages = extract_blocks(path)
        except Exception as e:                      # graceful: one bad PDF never stops the run
            print(f"[error] {d['doc_id']}: {e}")
            continue
        extracted[d["doc_id"]] = (d, blocks, n_pages)
    print(f"{len(extracted)} documents extracted")

    # ---- pass 2: boilerplate detection (same long block repeated across documents) ----
    docs_per_type = Counter(d["doc_type"] for d, _, _ in extracted.values())
    seen_in = defaultdict(set)
    for doc_id, (d, blocks, _) in extracted.items():
        for b in blocks:
            if b.kind == "text" and len(b.text.split()) >= BOILERPLATE_MIN_WORDS:
                seen_in[(d["doc_type"], signature(b.text))].add(doc_id)
    boilerplate = {
        key for key, ids in seen_in.items()
        if docs_per_type[key[0]] >= 5 and len(ids) / docs_per_type[key[0]] >= BOILERPLATE_DOC_SHARE
    }
    kept_once: set = set()

    # ---- pass 3: sections -> parents -> children --------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(Counter)
    with (OUT / "parents.jsonl").open("w", encoding="utf-8") as fp, \
         (OUT / "chunks.jsonl").open("w", encoding="utf-8") as fc:
        for doc_id, (d, blocks, n_pages) in extracted.items():
            doc_type = d["doc_type"]
            pub_date = find_publication_date(blocks)

            clean = []
            for b in blocks:
                key = (doc_type, signature(b.text))
                if b.kind == "text" and key in boilerplate:
                    if key in kept_once:            # keep the first occurrence in the corpus only
                        stats[doc_type]["boilerplate_removed"] += 1
                        continue
                    kept_once.add(key)
                clean.append(b)

            family = family_of(doc_type)
            sections = split_sections(clean, family)
            n_children = 0
            for s_idx, section in enumerate(sections):
                for p_idx, units in enumerate(make_parents(section, PARENT_TOKENS, count)):
                    parent_id = f"{doc_id}::s{s_idx:03d}p{p_idx}"
                    parent_text = units_to_text(units)
                    title = section.title + (f" (suite {p_idx + 1})" if p_idx else "")
                    fp.write(json.dumps({
                        "parent_id": parent_id, "doc_id": doc_id, "section": title,
                        "page_start": section.page_start, "page_end": section.page_end,
                        "n_tokens": count(parent_text), "text": parent_text,
                    }, ensure_ascii=False) + "\n")
                    stats[doc_type]["parents"] += 1

                    for kind, text in pack(units, CHILD_TOKENS, OVERLAP_TOKENS, count):
                        context = f"{d['title']} | {title}"
                        chunk_key = f"{doc_id}::c{n_children:04d}"
                        fc.write(json.dumps({
                            "id": str(uuid.uuid5(NAMESPACE, chunk_key)),
                            "chunk_id": chunk_key,
                            "parent_id": parent_id,
                            "doc_id": doc_id,
                            "title": d["title"],
                            "doc_type": doc_type,
                            "category": CATEGORY.get(doc_type, "contrat"),
                            "offer_line": d["folder"].split("_")[-1],     # internet | mobile
                            "section": title,
                            "chunk_type": kind,                            # text | table
                            "page": section.page_start,
                            "page_end": section.page_end,
                            "source": Path(d["file_path"]).name,
                            "source_url": d["source_url"],
                            "publication_date": pub_date,
                            "country": d["country"],
                            "language": d["language"],
                            "is_synthetic": d["is_synthetic"] == "True",
                            "chunk_index": n_children,
                            "n_tokens": count(text),
                            "text": text,
                            "embed_text": f"{context}\n{text}",
                        }, ensure_ascii=False) + "\n")
                        n_children += 1
                        stats[doc_type]["chunks"] += 1
                        stats[doc_type]["table_chunks"] += kind == "table"
                        stats[doc_type]["tokens"] += count(text)
            stats[doc_type]["docs"] += 1
            stats[doc_type]["dated"] += pub_date is not None
            stats[doc_type]["sections"] += len(sections)

    # ---- report ---------------------------------------------------------------------
    print(f"\n{'type':<11}{'docs':>5}{'dated':>6}{'sect.':>7}{'parents':>8}{'chunks':>8}"
          f"{'tables':>7}{'avg tok':>8}{'boiler.':>8}")
    tot = Counter()
    for t, s in sorted(stats.items(), key=lambda x: -x[1]["chunks"]):
        avg = s["tokens"] / s["chunks"] if s["chunks"] else 0
        print(f"{t:<11}{s['docs']:>5}{s['dated']:>6}{s['sections']:>7}{s['parents']:>8}"
              f"{s['chunks']:>8}{s['table_chunks']:>7}{avg:>8.0f}{s['boilerplate_removed']:>8}")
        tot.update(s)
    print(f"\nTOTAL: {tot['docs']} docs -> {tot['parents']} parents -> {tot['chunks']} chunks "
          f"({tot['table_chunks']} tables), {tot['boilerplate_removed']} boilerplate blocks removed")
    print(f"Output: {(OUT / 'parents.jsonl').relative_to(ROOT)}, {(OUT / 'chunks.jsonl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()