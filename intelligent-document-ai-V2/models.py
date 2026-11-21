"""
models.py — Shared data structures and exceptions. Imports nothing from this
project, so it can never take part in a circular import.

LINE BY LINE (non-obvious lines):
  Decimal fields (subtotal, tax, total, ...)
      Money is stored as Decimal, not float. See config.py for why.

  Dates are stored as ISO strings ("2024-03-15"), not date objects.
      ISO strings sort correctly as plain text ("2024-03-15" < "2024-04-14"),
      serialise to JSON with no custom code, and are unambiguous.

  Optional[...] = None
      "Not found in the document" is represented by None, never by 0 or "".
      A missing total and a total of zero mean very different things.

  class Issue(severity, field, message)
      One problem found during processing.
        severity "error"   -> the document goes to the review queue
        severity "warning" -> noted, but the document can still be accepted

  class DocResult.status
        "accepted"     no errors
        "needs_review" at least one error; a human should look
        "failed"       nothing usable could be read or extracted

  @property errors / warnings
      Computed on demand. dataclasses.asdict() ignores properties, so they
      are not written into the JSON output twice.

  class DocumentError / ExtractionError
      Two exception types so callers can tell "the FILE is unusable"
      from "the MODEL's output is unusable" and handle each differently.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Union


class DocumentError(Exception):
    """The file cannot be read (too big, empty, encrypted, wrong type...)."""


class ExtractionError(Exception):
    """The model's output could not be turned into data."""


@dataclass
class Issue:
    severity: str          # "error" or "warning"
    field: str
    message: str


@dataclass
class LineItem:
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    amount: Optional[Decimal] = None


@dataclass
class InvoiceData:
    invoice_number: Optional[str] = None
    vendor: Optional[str] = None
    customer: Optional[str] = None
    invoice_date: Optional[str] = None      # ISO "YYYY-MM-DD"
    due_date: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[LineItem] = field(default_factory=list)
    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    total: Optional[Decimal] = None
    po_number: Optional[str] = None
    bank_account: Optional[str] = None      # IBAN or account number, as written


@dataclass
class ContractData:
    title: Optional[str] = None
    parties: List[str] = field(default_factory=list)
    effective_date: Optional[str] = None
    end_date: Optional[str] = None
    term: Optional[str] = None
    payment_terms: Optional[str] = None
    termination: Optional[str] = None
    governing_law: Optional[str] = None
    auto_renewal: Optional[bool] = None
    key_obligations: List[str] = field(default_factory=list)


@dataclass
class PageOcr:
    """OCR quality for one page. Confidence is 0-100."""
    page: int                       # 0-based page index within the file
    mean_confidence: float
    lines: int
    low_numeric_lines: int          # lines containing digits read with low confidence


@dataclass
class LoadedDocument:
    """A file's text, page by page, plus how much of it came from OCR."""
    pages: List[str]
    ocr_pages: Dict[int, PageOcr] = field(default_factory=dict)
    ocr_engine: Optional[str] = None

    @property
    def text(self) -> str:
        return "\f".join(self.pages)       # \f (form feed) marks a page break


@dataclass
class DocResult:
    filename: str
    status: str = "failed"
    doc_type: str = "unknown"
    classification_method: str = ""
    data: Optional[Union[InvoiceData, ContractData]] = None
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""
    repair_attempts: int = 0
    part: int = 1                           # which document inside the file (bundles)
    parts: int = 1
    ocr_engine: Optional[str] = None
    ocr_confidence: Optional[float] = None

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "warning"]
