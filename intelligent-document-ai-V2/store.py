"""
store.py — Invoice history that survives between runs (SQLite).

WHAT IT GIVES US:
  * duplicate detection ACROSS runs (v1 only caught duplicates within one run)
  * each vendor's past accepted totals, for the "unusually large" check
  * how much of a purchase order has already been invoiced

SAME FILE RE-RUN vs REAL DUPLICATE:
  Running the tool twice on the same inbox must not flag every invoice as a
  duplicate of itself. A prior record is "the same submission" only when BOTH the
  filename AND the content hash match. Anything else with the same vendor and
  invoice number is a duplicate, including:
    * the same PDF emailed again under a different name  (same hash, new name)
    * an edited file reusing the old invoice number      (same name, new hash)

LINE BY LINE (non-obvious lines):
  ":memory:"
      SQLite's in-memory database. The default for tests and `--no-history`:
      history lasts for one run only, like v1.

  PRIMARY KEY (vendor_key, number_key)
      One row per (vendor, invoice number). A duplicate can therefore never
      overwrite the original record.

  total stored as TEXT
      Decimal("1323.00") -> "1323.00". SQLite REAL would be a binary float.

  `?` placeholders
      Values are passed separately from the SQL text, never pasted into it.
      Vendor names come from untrusted documents; building SQL with f-strings
      would allow SQL injection ("Acme'; DROP TABLE invoices;--").

  record(): UPDATE when same submission, INSERT OR IGNORE otherwise
      A re-run refreshes the status/total of the same submission (it may have
      been fixed). A different submission with the same key is ignored, so the
      first record is kept.

  accepted_totals() / po_invoiced() only count status = 'accepted'
      An invoice still waiting for review must not shape a vendor's "normal"
      amounts or use up a purchase order.
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PriorInvoice:
    filename: str
    content_hash: str
    status: str


class InvoiceStore:
    def __init__(self, path=":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                vendor_key   TEXT NOT NULL,
                number_key   TEXT NOT NULL,
                filename     TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                status       TEXT NOT NULL,
                total        TEXT,
                currency     TEXT,
                po_key       TEXT,
                processed_at TEXT NOT NULL,
                PRIMARY KEY (vendor_key, number_key)
            )""")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ── reads ─────────────────────────────────────────────────────────────
    def find(self, vendor: str, number: str) -> Optional[PriorInvoice]:
        row = self.db.execute(
            "SELECT filename, content_hash, status FROM invoices WHERE vendor_key = ? AND number_key = ?",
            (vendor, number)).fetchone()
        return PriorInvoice(*row) if row else None

    def accepted_totals(self, vendor: str, currency: Optional[str], exclude_number: str) -> List[Decimal]:
        rows = self.db.execute(
            "SELECT total FROM invoices WHERE vendor_key = ? AND status = 'accepted' "
            "AND number_key != ? AND total IS NOT NULL AND (currency IS ? OR currency = ?)",
            (vendor, exclude_number, currency, currency)).fetchall()
        return [Decimal(r[0]) for r in rows]

    def po_invoiced(self, po: str, exclude_vendor: str, exclude_number: str) -> Decimal:
        rows = self.db.execute(
            "SELECT total FROM invoices WHERE po_key = ? AND status = 'accepted' "
            "AND total IS NOT NULL AND NOT (vendor_key = ? AND number_key = ?)",
            (po, exclude_vendor, exclude_number)).fetchall()
        return sum((Decimal(r[0]) for r in rows), Decimal(0))

    # ── writes ────────────────────────────────────────────────────────────
    def record(self, vendor: str, number: str, filename: str, chash: str, status: str,
               total: Optional[Decimal], currency: Optional[str], po: Optional[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        total_text = None if total is None else str(total)

        updated = self.db.execute(
            "UPDATE invoices SET status = ?, total = ?, currency = ?, po_key = ?, processed_at = ? "
            "WHERE vendor_key = ? AND number_key = ? AND filename = ? AND content_hash = ?",
            (status, total_text, currency, po, now, vendor, number, filename, chash)).rowcount
        if not updated:
            self.db.execute(
                "INSERT OR IGNORE INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (vendor, number, filename, chash, status, total_text, currency, po, now))
        self.db.commit()
