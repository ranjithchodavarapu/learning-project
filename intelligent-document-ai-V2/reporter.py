"""
reporter.py — Writes the results to disk.

OUTPUTS (in outputs/):
  results/<file>.json   full detail for each document
  invoices.csv          one row per invoice, with its status
  report.md             counts, a table, and the REVIEW QUEUE first

LINE BY LINE (non-obvious lines):
  json.dump(..., default=str)
      json cannot serialise Decimal. default=str turns it into a string
      ("1323.00"). Strings keep money EXACT; JSON numbers become binary floats
      in most readers, which is the very problem Decimal avoids.

  ensure_ascii=False + encoding="utf-8"
      Keeps names like "Müller" readable in the file instead of "M\\u00fcller".

  re.sub(r"[^A-Za-z0-9._-]", "_", name)
      The output filename is derived from the INPUT filename, which is
      untrusted. This removes slashes and other characters so a file called
      "../../etc/x" cannot make us write outside outputs/.

  _cents(value)
      The CSV shows amounts with exactly two decimals. The model's JSON gives
      1323.0; an accounting import expects 1323.00. (The per-document JSON
      keeps the exact Decimal; only the CSV is rounded for display.)

  csv_safe(value)   (SPREADSHEET FORMULA INJECTION)
      A vendor name such as  =HYPERLINK("http://evil","click")  is harmless
      text in a document but becomes a live formula when someone opens the CSV
      in Excel. Cells starting with = + - @ (or tab/CR) get a leading
      apostrophe, which spreadsheets treat as "this is text". It is applied to
      TEXT fields only: amounts are Decimals written with str(), so a negative
      total like -150.00 is not mangled.

  report.md puts the review queue FIRST, because that is the part a human has to act on.
"""

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from models import DocResult, InvoiceData

_FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_STARTS) else text


def _cents(value) -> str:
    """Amounts in the CSV always show two decimals (1323.0 -> 1323.00)."""
    return "" if value is None else f"{value:.2f}"


def _safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", filename)


def write_outputs(results: List[DocResult], out_dir) -> Dict[str, Path]:
    out = Path(out_dir)
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        path = results_dir / (_safe_name(r.filename) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(r), f, indent=2, default=str, ensure_ascii=False)

    csv_path = out / "invoices.csv"
    _write_invoice_csv(results, csv_path)

    report_path = out / "report.md"
    report_path.write_text(_build_report(results), encoding="utf-8")

    return {"results": results_dir, "csv": csv_path, "report": report_path}


def _write_invoice_csv(results: List[DocResult], path: Path) -> None:
    columns = ["filename", "status", "invoice_number", "vendor", "customer",
               "invoice_date", "due_date", "currency", "subtotal", "tax", "total",
               "po_number", "bank_account", "errors"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in results:
            if not isinstance(r.data, InvoiceData):
                continue
            d = r.data
            writer.writerow([
                csv_safe(r.filename), r.status,
                csv_safe(d.invoice_number), csv_safe(d.vendor), csv_safe(d.customer),
                d.invoice_date or "", d.due_date or "", d.currency or "",
                _cents(d.subtotal), _cents(d.tax), _cents(d.total),
                csv_safe(d.po_number), csv_safe(d.bank_account),
                len(r.errors),
            ])


def _build_report(results: List[DocResult]) -> str:
    counts = {s: sum(1 for r in results if r.status == s)
              for s in ("accepted", "needs_review", "failed")}
    lines = [
        "# Document Processing Report", "",
        f"- Documents: {len(results)}",
        f"- Accepted: {counts['accepted']}",
        f"- Needs review: {counts['needs_review']}",
        f"- Failed: {counts['failed']}", "",
    ]

    queue = [r for r in results if r.status != "accepted"]
    lines += ["## Review queue", ""]
    if not queue:
        lines += ["Nothing needs attention.", ""]
    for r in queue:
        lines.append(f"### {r.filename}  ({r.status}, {r.doc_type})")
        lines += [f"- **{i.severity}** `{i.field}`: {i.message}" for i in r.issues]
        lines.append("")

    lines += ["## All documents", "",
              "| File | Type | Status | Errors | Warnings | Summary |",
              "|---|---|---|---|---|---|"]
    for r in results:
        summary = r.summary.replace("|", "/").replace("\n", " ")
        lines.append(f"| {r.filename} | {r.doc_type} | {r.status} | "
                     f"{len(r.errors)} | {len(r.warnings)} | {summary} |")
    return "\n".join(lines) + "\n"
