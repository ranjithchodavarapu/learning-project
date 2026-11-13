"""
loader.py — Loads documents and their security labels from the documents/ folder.

FILE FORMAT (plain .txt):
    title: Executive Compensation 2024
    classification: restricted
    department: hr
    ---
    ...document body...

LINE BY LINE:
  text.partition("\\n---\\n")
      Splits at the FIRST "---" line into (header, separator, body).
      partition (not split) because the body may itself contain "---".

  meta[key.strip().lower()] = value.strip()
      Parses "key: value" header lines. `line.split(":", 1)` splits only at the
      first colon so a title like "Q3: Forecast" keeps its second colon.

  Clearance.from_name(meta.get("classification", "restricted"))
      A document with NO classification header becomes RESTRICTED.
      Secure by default: forgetting to label a file hides it; it never leaks it.

  meta.get("department", "all")
      Missing department means "all". This is safe because the classification
      check still applies first.

  doc_id = path.stem
      "executive_salaries.txt" -> "executive_salaries". The file name is the ID.
"""

from pathlib import Path
from typing import List

from models import Clearance, Document


def parse_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition("\n---\n")

    meta = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()

    return Document(
        doc_id=path.stem,
        title=meta.get("title", path.stem),
        content=body.strip(),
        classification=Clearance.from_name(meta.get("classification", "restricted")),
        department=meta.get("department", "all").lower(),
    )


def load_documents(directory: str) -> List[Document]:
    folder = Path(directory)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {directory}")
    return [parse_document(p) for p in sorted(folder.glob("*.txt"))]
