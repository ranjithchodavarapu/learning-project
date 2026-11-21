"""
truth_kit.py — Tools for testing the system on YOUR OWN documents.

  python truth_kit.py template --inbox my_invoices --out eval/my_truth.json
  python truth_kit.py check    --truth eval/my_truth.json --inbox my_invoices
  python evaluate.py --inbox my_invoices --truth eval/my_truth.json --no-registry

WHY A KIT AND NOT AN AUTOMATIC ANSWER KEY:
  The "truth" is what each document ACTUALLY says, written by a person who read
  it. It must never be drafted by the system being tested: a truth file written
  by the model would contain the model's own mistakes, and the evaluation would
  report 100% while measuring nothing. (This project learned that: my stand-in
  model and my truth file shared a wrong assumption and agreed perfectly.)

WHAT EACH COMMAND DOES:
  template   lists your documents (a scanned bundle is listed as bundle.pdf#1,
             #2, ...) and writes a skeleton full of "TODO" for you to fill in.
             It reads the files only to find their names. It does NOT ask the
             model, and it prefills NOTHING.
  check      finds mistakes in a filled-in file BEFORE they distort the score:
             leftover TODOs, wrong status words, money written as "1,323.00"
             (evaluate.py needs plain 1323.00), dates not in YYYY-MM-DD, fields
             that make no sense for the document type, names that match no file,
             and values that do not appear in the document (usually a typo).
             Exit code 1 means "fix these first".

LINE BY LINE (non-obvious lines):
  _PLAIN_NUMBER = ^-?\\d+(\\.\\d+)?$
      evaluate.py compares money with Decimal(str(value)). "1,323.00" would crash
      it, and "$5" would be nonsense. The check insists on a plain number.

  isinstance(value, bool) checked before int
      In Python True is an int (True == 1). Without the check, "line_items": true
      would pass as a count of 1.

  _in_document() is a TYPO guard, not a score
      It reuses the pipeline's own matching (whole-token for identifiers, date
      renderings, number set). A truth value the matcher cannot find in the text is
      far more likely a typo than a deliberate paraphrase. It only warns, because
      you may have legitimately written a clause in your own words.

  BALANCE warnings (fewer than 10 bad / 20 good)
      "Missed bad" is a rate over BAD documents. With 3 bad documents, "0 missed"
      means almost nothing. The check tells you how many you have and how many to aim
      for. Make bad ones by editing COPIES of good invoices (change a total,
      repeat an invoice number, swap a bank account).
"""

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loader import list_documents, load_document
from models import DocumentError
from splitter import split_pages
from textutil import (date_renderings, mostly_grounded, normalize, normalize_account,
                      numbers_in_text, to_iso_date)
from validator import _Haystack

INVOICE_FIELDS = ["invoice_number", "vendor", "customer", "invoice_date", "due_date", "currency",
                  "subtotal", "tax", "total", "po_number", "bank_account", "line_items"]
CONTRACT_FIELDS = ["title", "parties", "effective_date", "end_date", "term", "payment_terms",
                   "termination", "governing_law", "auto_renewal"]
MONEY = {"subtotal", "tax", "total"}
DATES = {"invoice_date", "due_date", "effective_date", "end_date"}
CLAUSES = {"term", "payment_terms", "termination", "governing_law"}
STATUSES = {"accepted", "needs_review"}
DOC_TYPES = {"invoice", "contract", "unknown"}
MIN_BAD, MIN_GOOD = 10, 20
_PLAIN_NUMBER = re.compile(r"-?\d+(\.\d+)?")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

HELP = (
    "Fill in every TODO from what each document ACTUALLY says. status: accepted = a correct pipeline "
    "should pass it automatically; needs_review = a person should look (wrong total, duplicate, unknown "
    "vendor, ...). Use null for anything the document does not state: never guess. Money: plain numbers "
    "(1323.00, no commas or symbols). Dates: YYYY-MM-DD. Delete fields you do not want scored. "
    "For a contract change doc_type to contract and use: " + ", ".join(CONTRACT_FIELDS) + ". "
    "Remove this _help line when done (lines starting with _ are ignored anyway)."
)


# ── reading the folder ────────────────────────────────────────────────────
def document_texts(files) -> Dict[str, Optional[str]]:
    """result name -> text (None when unreadable). Bundles become name#1, name#2 ..."""
    out: Dict[str, Optional[str]] = {}
    for path in files:
        try:
            segments = split_pages(load_document(path).pages)
        except DocumentError:
            out[path.name] = None
            continue
        for k, segment in enumerate(segments, 1):
            out[path.name if len(segments) == 1 else f"{path.name}#{k}"] = segment.text
    return out


def build_template(names: List[str]) -> dict:
    template: dict = {"_help": HELP}
    for name in names:
        template[name] = {
            "status": "TODO: accepted or needs_review",
            "doc_type": "TODO: invoice, contract or unknown",
            "fields": {field: "TODO" for field in INVOICE_FIELDS},
        }
    return template


# ── checking a filled-in file ─────────────────────────────────────────────
def _field_problem(field: str, value) -> Optional[str]:
    if value is None:
        return None
    if field in MONEY:
        if isinstance(value, bool) or not _PLAIN_NUMBER.fullmatch(str(value)):
            return f"{value!r} is not a plain number (write 1323.00, not '1,323.00' or '$1323')"
    elif field in DATES:
        if not (isinstance(value, str) and _ISO_DATE.fullmatch(value) and to_iso_date(value)):
            return f"{value!r} is not a valid date in YYYY-MM-DD form"
    elif field == "line_items":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"{value!r} must be a whole number (how many line items), or null"
    elif field == "parties":
        if not (isinstance(value, list) and value and all(isinstance(p, str) and p.strip() for p in value)):
            return "must be a list of names, like [\"A Ltd\", \"B Inc\"]"
    elif field == "auto_renewal":
        if not isinstance(value, bool):
            return f"{value!r} must be true, false or null (null = the document does not say)"
    elif not (isinstance(value, str) and value.strip()):
        return f"{value!r} must be text, or null if the document does not state it"
    return None


def _in_document(field: str, value, text: str) -> bool:
    h = _Haystack(text, ocr=True)               # tolerant superset: only used to warn about typos
    if field in MONEY:
        return abs(Decimal(str(value))) in numbers_in_text(text)
    if field in DATES:
        return any(h.token(r) for r in date_renderings(value))
    if field in ("invoice_number", "po_number", "currency"):
        return h.token(str(value))
    if field == "bank_account":
        return normalize_account(str(value)) in normalize_account(text)
    if field == "parties":
        return all(h.substring(p) for p in value)
    if field in CLAUSES:
        return h.substring(str(value)) or mostly_grounded(str(value), normalize(text))
    if field in ("line_items", "auto_renewal"):
        return True                              # cannot be checked by looking for text
    return h.substring(str(value))


def check_truth(truth: dict, texts: Dict[str, Optional[str]]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    entries = {k: v for k, v in truth.items() if not k.startswith("_")}

    for name in sorted(set(entries) - set(texts)):
        errors.append(f"{name}: no such document in the folder(s) (typo? wrong bundle part number?)")
    for name in sorted(set(texts) - set(entries)):
        warnings.append(f"{name}: not in the truth file, so evaluate.py will ignore it")
    for name in sorted(n for n, t in texts.items() if t is None):
        warnings.append(f"{name}: could not be read (scanned file with no OCR engine?)")

    for name, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            errors.append(f"{name}: must be an object with status, doc_type and fields")
            continue
        if "TODO" in json.dumps(entry):
            errors.append(f"{name}: still contains TODO")
            continue
        if entry.get("status") not in STATUSES:
            errors.append(f"{name}: status must be 'accepted' or 'needs_review', not {entry.get('status')!r}")
        doc_type = entry.get("doc_type")
        if doc_type not in DOC_TYPES:
            errors.append(f"{name}: doc_type must be invoice, contract or unknown, not {doc_type!r}")
            continue

        fields = entry.get("fields", {})
        if not isinstance(fields, dict):
            errors.append(f"{name}: fields must be an object")
            continue
        allowed = {"invoice": INVOICE_FIELDS, "contract": CONTRACT_FIELDS}.get(doc_type, [])
        for field, value in fields.items():
            if field not in allowed:
                errors.append(f"{name}: '{field}' is not a field of a {doc_type}"
                              + (f" (use one of: {', '.join(allowed)})" if allowed else " (unknown documents have no fields)"))
                continue
            problem = _field_problem(field, value)
            if problem:
                errors.append(f"{name}: {field}: {problem}")
                continue
            text = texts.get(name)
            if text is not None and value is not None and not _in_document(field, value, text):
                warnings.append(f"{name}: {field} = {value!r} was not found in the document text (typo?)")

    good = sum(1 for e in entries.values() if isinstance(e, dict) and e.get("status") == "accepted")
    bad = sum(1 for e in entries.values() if isinstance(e, dict) and e.get("status") == "needs_review")
    if bad < MIN_BAD:
        warnings.append(f"only {bad} documents are expected to need review. That is too few to measure how many bad "
                        f"documents slip through (aim for {MIN_BAD}+). Make bad copies of good invoices by editing a "
                        "total, repeating an invoice number, or swapping a bank account.")
    if good < MIN_GOOD:
        warnings.append(f"only {good} documents are expected to be accepted (aim for {MIN_GOOD}+ to judge false reviews).")
    return errors, warnings


def structure_errors(truth: dict) -> List[str]:
    """Only the structural mistakes (TODOs, bad status words, bad field formats). Needs no
    document text, so evaluate.py can refuse a broken truth file BEFORE spending minutes
    running the model on it."""
    entries = [k for k in truth if not k.startswith("_")]
    return check_truth(truth, {name: None for name in entries})[0]


# ── command line ──────────────────────────────────────────────────────────
def _gather(folders) -> Dict[str, Optional[str]]:
    texts: Dict[str, Optional[str]] = {}
    for folder in folders:
        texts.update(document_texts(list_documents(folder)))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and check a truth file for your own documents")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("template", help="write a skeleton truth file for the documents in a folder")
    t.add_argument("--inbox", action="append", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--force", action="store_true", help="overwrite an existing file")
    c = sub.add_parser("check", help="find mistakes in a filled-in truth file")
    c.add_argument("--truth", required=True)
    c.add_argument("--inbox", action="append", required=True)
    args = parser.parse_args()

    try:
        texts = _gather(args.inbox)
    except DocumentError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    if args.command == "template":
        if not texts:
            print(f"ERROR: no documents found in {', '.join(args.inbox)}. Put your invoices there first "
                  "(README files are ignored).")
            sys.exit(2)
        out = Path(args.out)
        if out.exists() and not args.force:
            print(f"ERROR: {out} already exists. Use --force to overwrite it (this would erase your answers).")
            sys.exit(2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(build_template(sorted(texts)), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out} with {len(texts)} document(s). Fill in every TODO by READING each document, then run:\n"
              f"  python truth_kit.py check --truth {out} " + " ".join(f"--inbox {i}" for i in args.inbox))
        return

    try:
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {args.truth}: {exc}")
        sys.exit(2)
    errors, warnings = check_truth(truth, texts)
    for e in errors:
        print(f"ERROR   {e}")
    for w in warnings:
        print(f"warning {w}")
    if errors:
        print(f"\n{len(errors)} error(s) to fix before running evaluate.py.")
        sys.exit(1)
    print(f"\nNo errors ({len(warnings)} warning(s)). Now run:  python evaluate.py --truth {args.truth} "
          + " ".join(f"--inbox {i}" for i in args.inbox) + " --no-registry")


if __name__ == "__main__":
    main()
