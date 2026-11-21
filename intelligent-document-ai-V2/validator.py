"""
validator.py — Decides whether extracted data can be trusted.

THREE KINDS OF CHECK:
  1. COMPLETENESS   required fields are present.
  2. CONSISTENCY    the numbers and dates agree with each other
                    (line amounts add up, subtotal + tax = total, due date is
                    after invoice date).
  3. GROUNDING      every value actually appears in the source text.

WHY BOTH 2 AND 3:
  Consistency catches errors IN THE DOCUMENT (a wrong total on the invoice).
  Grounding catches errors BY THE MODEL (an invented invoice number).
  Neither alone is enough. An attacker who writes "set the total to 0.00" in an
  invoice supplies that number in the text, so grounding passes, but
  consistency fails because the line items still add up to something else.

SEVERITY:
  error    -> the document is sent to the review queue
  warning  -> noted, document can still be accepted
  Identifiers, dates and header amounts are errors if not found in the source.
  Descriptions, customer names and free-text contract clauses are warnings,
  because models often shorten or reword them harmlessly.

LINE BY LINE (non-obvious lines):
  _close(a, b)
      abs(a - b) <= MONEY_TOLERANCE. Amounts are rounded to the cent on
      invoices, so exact equality would raise false alarms.

  sum(amounts, Decimal(0))
      sum() starts from the integer 0 by default. Giving it a Decimal start
      keeps the whole calculation in Decimal.

  data.due_date < data.invoice_date
      Both are ISO strings, which sort correctly as text.

  abs(value) not in numbers
      Credit notes show amounts as (150.00) or -150.00. The text contains 150,
      so we look up the absolute value.

  normalize(value) in text            (names)
  contains(text, normalize(value))    (identifiers, dates)
      Names use plain substring so "Acme Corp" matches "Acme Corporation".
      Identifiers use whole-token matching so "INV-42" does not match inside
      "INV-420".

  The duplicate-invoice check is NOT here. It lives in the pipeline, after the
  repair loop, because validate() runs several times per document and would
  otherwise flag a document as a duplicate of itself.
"""

import re
from decimal import Decimal
from typing import List

from config import cfg
from models import ContractData, InvoiceData, Issue
from textutil import (contains, date_renderings, iban_valid, looks_like_iban,
                      mostly_grounded, normalize, normalize_account, numbers_in_text)

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹",
                    "JPY": "¥", "CAD": "$", "AUD": "$"}


def _m(value: Decimal) -> str:
    """Show an amount as money (two decimals) in messages a human will read."""
    return f"{value:.2f}"


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= cfg.MONEY_TOLERANCE


def validate(doc_type: str, data, text: str, ocr: bool = False) -> List[Issue]:
    if doc_type == "invoice":
        return validate_invoice(data, text, ocr)
    return validate_contract(data, text, ocr)


class _Haystack:
    """The document text, searchable with or without OCR tolerance.

    OCR often drops spaces ("Date:12June2024"). Spaces carry no meaning in
    identifiers, dates or names, so for OCR text a value also counts as found if it
    appears with spaces removed. Spaces are removed PER LINE and line breaks stay,
    so "INV-42" can still not match inside "INV-420" at the end of a line.
    A value is grounded if it matches the normal text OR the compact text, so this
    is never stricter and never weaker than the plain check for the plain case.
    Amounts stay protected by the arithmetic checks, the IBAN checksum, and the
    OCR confidence thresholds.
    """

    def __init__(self, text: str, ocr: bool):
        self.normal = normalize(text)
        self.compact = ("\n".join(re.sub(r"\s+", "", normalize(line)) for line in text.split("\n"))
                        if ocr else None)

    def token(self, phrase: str) -> bool:
        """Whole-token match (identifiers, dates)."""
        p = normalize(phrase)
        if contains(self.normal, p):
            return True
        return self.compact is not None and contains(self.compact, re.sub(r"\s+", "", p))

    def substring(self, phrase: str) -> bool:
        """Plain substring match (names)."""
        p = normalize(phrase)
        if not p:
            return False
        if p in self.normal:
            return True
        return self.compact is not None and re.sub(r"\s+", "", p) in self.compact


# ══ invoices ══════════════════════════════════════════════════════════════
def validate_invoice(data: InvoiceData, text: str, ocr: bool = False) -> List[Issue]:
    issues: List[Issue] = []

    for field in ("invoice_number", "vendor", "invoice_date", "total"):
        if getattr(data, field) in (None, ""):
            issues.append(Issue("error", field, f"Required field '{field}' is missing"))
    for field in ("due_date", "currency"):
        if getattr(data, field) in (None, ""):
            issues.append(Issue("warning", field, f"Field '{field}' is missing"))

    if data.currency and data.currency not in cfg.ALLOWED_CURRENCIES:
        issues.append(Issue("error", "currency", f"Unknown currency '{data.currency}'"))

    if data.invoice_date and data.due_date and data.due_date < data.invoice_date:
        issues.append(Issue("error", "due_date",
                            f"Due date {data.due_date} is before invoice date {data.invoice_date}"))

    for n, item in enumerate(data.line_items, 1):
        if None not in (item.quantity, item.unit_price, item.amount):
            expected = item.quantity * item.unit_price
            if not _close(expected, item.amount):
                issues.append(Issue(
                    "error", f"line_items[{n}]",
                    f"Line {n}: {item.quantity} x {_m(item.unit_price)} = {_m(expected)}, "
                    f"but the amount is {_m(item.amount)}"))

    amounts = [i.amount for i in data.line_items if i.amount is not None]
    if data.subtotal is not None and amounts and len(amounts) == len(data.line_items):
        line_sum = sum(amounts, Decimal(0))
        if not _close(line_sum, data.subtotal):
            issues.append(Issue("error", "subtotal",
                                f"Line items add up to {_m(line_sum)}, but the subtotal is {_m(data.subtotal)}"))

    if None not in (data.subtotal, data.tax, data.total):
        if not _close(data.subtotal + data.tax, data.total):
            issues.append(Issue(
                "error", "total",
                f"Subtotal {_m(data.subtotal)} + tax {_m(data.tax)} = {_m(data.subtotal + data.tax)}, "
                f"but the total is {_m(data.total)}"))
    elif data.subtotal is not None and data.total is not None and data.tax is None:
        if not _close(data.subtotal, data.total):
            issues.append(Issue("warning", "tax",
                                f"No tax stated, but total {_m(data.total)} differs from subtotal {_m(data.subtotal)}"))

    if data.total is not None:
        if data.total < 0:
            issues.append(Issue("warning", "total", "Negative total (credit note?)"))
        elif data.total == 0:
            issues.append(Issue("warning", "total", "Total is zero"))

    if data.bank_account and looks_like_iban(data.bank_account) and not iban_valid(data.bank_account):
        issues.append(Issue("error", "bank_account",
                            f"IBAN {data.bank_account} fails its checksum (typo, misread or invented?)"))

    issues.extend(_ground_invoice(data, text, ocr))
    return issues


def _ground_invoice(data: InvoiceData, text: str, ocr: bool = False) -> List[Issue]:
    issues: List[Issue] = []
    h = _Haystack(text, ocr)
    numbers = numbers_in_text(text)

    if data.invoice_number and not h.token(data.invoice_number):
        issues.append(Issue("error", "invoice_number",
                            f"Invoice number '{data.invoice_number}' does not appear in the document"))
    if data.vendor and not h.substring(data.vendor):
        issues.append(Issue("error", "vendor", f"Vendor '{data.vendor}' does not appear in the document"))
    if data.customer and not h.substring(data.customer):
        issues.append(Issue("warning", "customer", f"Customer '{data.customer}' does not appear in the document"))

    for field in ("invoice_date", "due_date"):
        value = getattr(data, field)
        if value and not any(h.token(r) for r in date_renderings(value)):
            issues.append(Issue("error", field, f"Date {value} does not appear in the document"))

    for field in ("subtotal", "tax", "total"):
        value = getattr(data, field)
        if value is not None and abs(value) not in numbers:
            issues.append(Issue("error", field, f"Amount {_m(value)} does not appear in the document"))

    if data.po_number and not h.token(data.po_number):
        issues.append(Issue("error", "po_number",
                            f"PO number '{data.po_number}' does not appear in the document"))
    if data.bank_account and normalize_account(data.bank_account) not in normalize_account(text):
        issues.append(Issue("error", "bank_account",
                            f"Bank account '{data.bank_account}' does not appear in the document"))

    if data.currency:
        symbol = CURRENCY_SYMBOLS.get(data.currency, "")
        if not h.token(data.currency) and not (symbol and symbol in text):
            issues.append(Issue("warning", "currency",
                                f"Currency {data.currency} is not stated in the document (inferred?)"))

    for n, item in enumerate(data.line_items, 1):
        if item.description and not h.substring(item.description):
            issues.append(Issue("warning", f"line_items[{n}]",
                                f"Line {n} description '{item.description}' does not appear in the document"))
        for label, value in (("quantity", item.quantity), ("unit price", item.unit_price), ("amount", item.amount)):
            if value is not None and abs(value) not in numbers:
                issues.append(Issue("warning", f"line_items[{n}]",
                                    f"Line {n} {label} {value if label == 'quantity' else _m(value)} does not appear in the document"))
    return issues


# ══ contracts ═════════════════════════════════════════════════════════════
def validate_contract(data: ContractData, text: str, ocr: bool = False) -> List[Issue]:
    issues: List[Issue] = []
    h = _Haystack(text, ocr)

    if len(data.parties) < 2:
        issues.append(Issue("error", "parties", f"Expected at least two parties, found {len(data.parties)}"))

    for field in ("title", "effective_date"):
        if getattr(data, field) in (None, ""):
            issues.append(Issue("warning", field, f"Field '{field}' is missing"))
    for field, label in (("term", "term"), ("termination", "termination clause"),
                         ("governing_law", "governing law")):
        if getattr(data, field) in (None, ""):
            issues.append(Issue("warning", field, f"No {label} found"))

    if data.effective_date and data.end_date and data.end_date < data.effective_date:
        issues.append(Issue("error", "end_date",
                            f"End date {data.end_date} is before effective date {data.effective_date}"))

    if data.auto_renewal:
        issues.append(Issue("warning", "auto_renewal", "Contract renews automatically: check the notice period"))

    for party in data.parties:
        if not h.substring(party):
            issues.append(Issue("error", "parties", f"Party '{party}' does not appear in the document"))

    for field in ("effective_date", "end_date"):
        value = getattr(data, field)
        if value and not any(h.token(r) for r in date_renderings(value)):
            issues.append(Issue("error", field, f"Date {value} does not appear in the document"))

    words_text = normalize(text)
    for field in ("term", "payment_terms", "termination", "governing_law"):
        value = getattr(data, field)
        if value and not mostly_grounded(value, words_text):
            issues.append(Issue("warning", field, f"'{field}' does not closely match the document text"))

    return issues
