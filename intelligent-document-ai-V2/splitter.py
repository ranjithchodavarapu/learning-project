"""
splitter.py — Finds the boundaries between invoices that share one file.

THE PROBLEM:
  Accounts payable often scans a stack of invoices into ONE PDF. v1 treated a
  file as one document and would have mashed three invoices into one extraction.

THE METHOD (heuristic, and deliberately conservative):
  An "invoice number" label that starts a line (for example
  "Invoice Number: INV-7") is read as that invoice's identity. When the number
  CHANGES, a new document begins. A second page repeating the SAME number is
  just a continuation page and does not split.
  The boundary is placed at the start of the new invoice's header: the top of
  its page if the label sits near the top, otherwise its "INVOICE" title line,
  otherwise the label line itself.

WHAT THIS CAN'T DO (so the README says it too):
  * It splits invoices only. Bundled contracts, or an invoice mixed with other
    document types, are not separated.
  * Invoices with no recognisable "Invoice Number" label stay as one document.
  * A wrong split is not silently trusted: every piece goes through the whole
    pipeline, and a piece that is not a valid invoice lands in review.

LINE BY LINE (non-obvious lines):
  (?im)^[ \\t]*(?:tax[ \\t]+)?invoice[ \\t]*(?:number|no\\.?|nr\\.?|#)...
      (?i) ignore case, (?m) ^ matches at the start of every line. Requiring the
      label at the START of a line means "Credit note for invoice no. INV-5" in
      the middle of a sentence does NOT count as a new invoice.

  (?=[A-Za-z0-9\\-/._]{0,30}\\d)
      A lookahead: the value after the label must contain at least one DIGIT.
      "Invoice number must be quoted on payment" would otherwise read the word
      "must" as an invoice number and split the document. Real invoice numbers
      virtually always contain a digit. (An all-letter number such as "INV-ABC"
      is therefore not recognised; documented in the README.)

  (?:^|(?<=\\f))
      "Start of a line" must ALSO include "right after a page break". In multiline
      mode ^ matches only after a newline, but the first line of a page follows a
      form-feed. Without this, a bundle whose pages START with "Invoice No. 55"
      would never split.

  _TITLE = ... [^\\n:#\\d]{0,30}$
      A title line is "INVOICE", "TAX INVOICE" or "INVOICE (Rechnung)". The
      remainder may not contain a colon, hash or digit, otherwise
      "Invoice Date: 2024-03-15" would be mistaken for a title and the boundary
      would land in the middle of a header.

  value = normalize_account(...)
      Uppercase, spaces and dashes removed, so "inv-7" and "INV 7" are the same.

  prev_end
      A boundary must lie after the previous label ends, so two invoices can
      never swallow each other's header.

  MIN_SEGMENT_CHARS = 60
      A fragment that short (a stray page number, a footer) is merged back into
      the previous document instead of becoming its own "invoice".

  text[:start].count("\\f")
      The page number of a character offset: how many page breaks precede it.
"""

import re
from dataclasses import dataclass
from typing import List

from textutil import normalize_account

MIN_SEGMENT_CHARS = 60
WINDOW = 300                        # how far back from a label we look for its header

_LABEL = re.compile(
    r"(?im)(?:^|(?<=\f))[ \t]*(?:tax[ \t]+)?invoice[ \t]*(?:number|no\.?|nr\.?|#)[ \t]*[:#]?[ \t]*"
    r"(?=[A-Za-z0-9\-/._]{0,30}\d)([A-Za-z0-9][A-Za-z0-9\-/._]{1,30})")
_TITLE = re.compile(r"(?im)(?:^|(?<=\f))[ \t]*(?:tax[ \t]+)?(?:invoice|rechnung|facture)\b[^\n:#\d]{0,30}$")


@dataclass
class Segment:
    text: str
    first_page: int
    last_page: int


def _header_start(text: str, label_start: int, prev_end: int) -> int:
    lo = max(prev_end, label_start - WINDOW)
    window = text[lo:label_start]

    page_break = window.rfind("\f")
    if page_break != -1:
        return lo + page_break + 1                       # top of the page the label is on

    titles = list(_TITLE.finditer(window))
    if titles:
        return lo + titles[-1].start()                   # the nearest "INVOICE" title line

    return text.rfind("\n", 0, label_start) + 1          # the label's own line


def split_pages(pages: List[str]) -> List[Segment]:
    text = "\f".join(pages)
    starts = [0]
    current = None
    prev_end = 0

    for match in _LABEL.finditer(text):
        value = normalize_account(match.group(1))
        if current is not None and value != current:
            starts.append(max(_header_start(text, match.start(), prev_end), starts[-1]))
        current = value
        prev_end = match.end()

    cuts = starts + [len(text)]
    pieces = [(cuts[i], cuts[i + 1]) for i in range(len(starts))]

    merged: List[List[int]] = []
    for start, end in pieces:
        if merged and len(text[start:end].strip()) < MIN_SEGMENT_CHARS:
            merged[-1][1] = end                            # too small to be a document
        else:
            merged.append([start, end])

    segments = []
    for start, end in merged:
        body = text[start:end].strip()
        if not body:
            continue
        segments.append(Segment(
            text=body.replace("\f", "\n"),
            first_page=text[:start].count("\f"),
            last_page=text[:max(start, end - 1)].count("\f"),
        ))
    return segments or [Segment(text.replace("\f", "\n").strip(), 0, max(len(pages) - 1, 0))]
