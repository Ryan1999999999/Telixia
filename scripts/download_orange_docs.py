"""Downloads the Orange contract PDFs listed on boutique.orange.fr and writes a manifest.

    python scripts/download_orange_docs.py

Output:
    data/raw/docs/contrats_internet/*.pdf
    data/raw/docs/contrats_mobile/*.pdf
    data/raw/docs/manifest.csv      (committed: makes the corpus reproducible)
"""

import csv
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "raw" / "docs"
PAGES = {
    "contrats_internet": "https://boutique.orange.fr/information/tarifs-et-contrats",
    "contrats_mobile": "https://boutique.orange.fr/information/tarifs-et-contrats/documents-contractuels-mobile-et-equipement",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Telixia - academic project)"}
PREFIX = re.compile(r"^\(\s*(CGV|CGA|CGU|CG|CS|RC)\s*\)\s*", re.I)
PDF_SUFFIX = re.compile(r"\s*\(?\s*pdf[^)]*\)?\s*$", re.I)


def slugify(text: str, maxlen: int = 70) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:maxlen] or "doc"


def doc_type_of(title: str) -> str:
    m = PREFIX.match(title)
    if m:
        return m.group(1).upper()
    t = title.lower()
    if "frais" in t:
        return "TARIFS"
    if "tarif" in t:
        return "TARIFS"
    if "assurance" in t:
        return "ASSURANCE"
    if "formulaire" in t:
        return "FORMULAIRE"
    if "conditions g" in t or "condition g" in t:
        return "CGV"
    if "home" in t:
        return "ANNEXE"
    return "AUTRE"


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)
    # retry on network drops / server errors: waits 2s, 4s, 8s, 16s, 32s
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    rows, seen_urls, used_ids = [], set(), set()

    for folder, page in PAGES.items():
        out_dir = DOCS / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        resp = session.get(page, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")  # bytes: lets BS4 read the page charset
        links = [a for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
        print(f"\n{folder}: {len(links)} PDF links found")

        for a in links:
            url = urljoin(page, a["href"])
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = " ".join(a.get_text(" ", strip=True).split())
            title = PDF_SUFFIX.sub("", title) or Path(urlparse(url).path).stem
            doc_type = doc_type_of(title)

            base_id = f"{folder.split('_')[1]}_{slugify(PREFIX.sub('', title))}"
            doc_id, i = base_id, 2
            while doc_id in used_ids:
                doc_id, i = f"{base_id}_{i}", i + 1
            used_ids.add(doc_id)

            path = out_dir / f"{doc_id}.pdf"
            if not path.exists():
                try:
                    r = session.get(url, timeout=60)
                    r.raise_for_status()
                    if not r.content.startswith(b"%PDF"):
                        print(f"  [skip] not a PDF: {title}")
                        continue
                    path.write_bytes(r.content)
                    print(f"  [ok]   {doc_type:<10} {title[:70]}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  [fail] {title[:70]}: {e}")
                    continue

            rows.append({
                "doc_id": doc_id,
                "title": title,
                "doc_type": doc_type,
                "category": "contrat",
                "folder": folder,
                "file_path": path.relative_to(ROOT).as_posix(),
                "source_url": url,
                "country": "FR",
                "language": "fr",
                "is_synthetic": False,
                "downloaded_at": date.today().isoformat(),
            })

    if not rows:
        print("\nNo documents downloaded - the page structure may have changed.")
        return
    manifest = DOCS / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} documents -> {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()