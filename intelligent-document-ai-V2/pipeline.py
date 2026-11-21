"""
pipeline.py — Runs one file (which may hold several documents) through every stage.

STAGES:
  1 load       read the file; OCR pages that have no text layer     (loader.py, ocr.py)
  2 split      separate several invoices that share one file        (splitter.py)
  then, for EACH document found:
  3 ocr check  low-confidence OCR sends the document to review
  4 scan       flag text aimed at the AI                            (security.py)
  5 classify   invoice / contract / unknown                         (classifier.py)
  6 extract    LLM fills a schema; one repair attempt if checks fail (extractor.py, validator.py)
  7 controls   duplicates, vendor, bank account, PO, history        (controls.py)
  8 summarize  template or LLM                                      (summarizer.py)
  9 record     remember the invoice so later runs can spot duplicates (store.py)

THE REPAIR LOOP IS SAFE BECAUSE OF GROUNDING (unchanged from v1):
  Any number the model invents to "fix" an inconsistent invoice is not in the
  source text, so the grounding check flags it and the document stays in review.

LINE BY LINE (non-obvious lines):
  process_file() returns a LIST
      One file can contain several documents (a scanned stack of invoices). A
      single-document file gives a list of one, with the plain file name. A
      bundle gives "bundle.pdf#1", "bundle.pdf#2", ...

  process(path)  (v1 API)
      Still returns one DocResult, but RAISES if the file holds several. A method
      that quietly returned only the first of three invoices would hide two.

  use_ocr = OCR_ENGINE != "none" or an engine was injected
      OCR=none means scanned files fail with a clear message instead of trying.

  except Exception around loading / a segment
      OCR engines and PDF parsers can crash in ways we cannot predict on
      hostile files. One bad document must become a `failed` row, not kill the
      batch. The exception type and message are kept in the issue.

  chash = content_hash(seg.text) BEFORE truncation
      Identity of a submission must not change if MAX_DOC_CHARS is changed
      between runs, or every old invoice would look like a new duplicate.

  OCR confidence is weighted by line count
      A 1-line cover sheet at 99% must not outvote a 40-line invoice at 70%.

  controls AFTER the repair loop
      They depend on external records, not on how well the model read the page.

  store.record(...) for every invoice with a vendor and number
      Even needs_review ones: a second copy of an unresolved invoice should
      still be flagged. Only `accepted` invoices count toward vendor history and
      purchase-order usage (see store.py).
"""

from pathlib import Path
from typing import List, Optional, Tuple

from classifier import classify
from config import cfg
from controls import duplicate_issue, invoice_keys, run_controls
from extractor import build_data, describe_previous, extract_raw
from llm import call_llm
from loader import list_documents, load_document
from models import (DocResult, DocumentError, ExtractionError, InvoiceData,
                    Issue, LoadedDocument)
from registry import Registry
from security import scan_document
from splitter import Segment, split_pages
from store import InvoiceStore, content_hash
from summarizer import summarize
from textutil import po_key
from validator import validate


class DocumentPipeline:
    def __init__(self, llm=call_llm, store: Optional[InvoiceStore] = None,
                 registry: Optional[Registry] = None, ocr_engine=None):
        self.llm = llm
        self.store = store if store is not None else InvoiceStore(":memory:")
        self.registry = registry if registry is not None else Registry()
        self.ocr_engine = ocr_engine

    # ── a file (possibly several documents) ───────────────────────────────
    def process_file(self, path) -> List[DocResult]:
        path = Path(path)
        use_ocr = cfg.OCR_ENGINE.lower() != "none" or self.ocr_engine is not None

        try:
            loaded = load_document(path, ocr_engine=self.ocr_engine, use_ocr=use_ocr)
        except DocumentError as exc:
            return [self._failure(path.name, "file", str(exc))]
        except Exception as exc:
            return [self._failure(path.name, "file", f"Could not read the file ({type(exc).__name__}: {exc})")]

        segments = split_pages(loaded.pages)
        if len(segments) > cfg.MAX_DOCS_PER_FILE:
            return [self._failure(path.name, "file",
                                  f"File seems to contain {len(segments)} documents; the limit is "
                                  f"{cfg.MAX_DOCS_PER_FILE}")]

        results = []
        for k, segment in enumerate(segments, 1):
            name = path.name if len(segments) == 1 else f"{path.name}#{k}"
            try:
                results.append(self._process_segment(name, segment, loaded, k, len(segments)))
            except Exception as exc:
                failed = self._failure(name, "internal", f"Unexpected error ({type(exc).__name__}: {exc})")
                failed.part, failed.parts = k, len(segments)
                results.append(failed)
        return results

    def process(self, path) -> DocResult:
        results = self.process_file(path)
        if len(results) != 1:
            raise ValueError(f"{path} contains {len(results)} documents; use process_file()")
        return results[0]

    def process_folder(self, folder) -> List[DocResult]:
        return [r for p in list_documents(folder) for r in self.process_file(p)]

    @staticmethod
    def _failure(name: str, field: str, message: str) -> DocResult:
        result = DocResult(filename=name)
        result.issues.append(Issue("error", field, message))
        return result

    # ── one document ──────────────────────────────────────────────────────
    def _process_segment(self, name: str, segment: Segment, loaded: LoadedDocument,
                         part: int, parts: int) -> DocResult:
        result = DocResult(filename=name, part=part, parts=parts)
        text = segment.text
        chash = content_hash(text)

        notes: List[Issue] = self._ocr_issues(result, loaded, segment)

        if len(text) > cfg.MAX_DOC_CHARS:
            text = text[:cfg.MAX_DOC_CHARS]
            notes.append(Issue(
                "error", "document",
                f"Document is longer than {cfg.MAX_DOC_CHARS} characters. Only the "
                "beginning was analysed, so values may be missing."))

        security_issues = scan_document(text)
        notes.extend(security_issues)

        result.doc_type, result.classification_method = classify(text, self.llm)

        if result.doc_type == "unknown":
            notes.append(Issue("error", "document_type",
                               "Could not tell whether this is an invoice or a contract"))
            result.issues = notes
            result.status = "needs_review"
            result.summary = "Unclassified document: " + " ".join(text.split())[:80]
            return result

        data, found, attempts = self._extract_with_repair(result.doc_type, text, ocr=bool(result.ocr_engine))
        result.data, result.repair_attempts = data, attempts
        result.issues = notes + found

        if data is None:
            return result                                   # status stays "failed"

        keys = None
        if isinstance(data, InvoiceData):
            keys = invoice_keys(data)
            prior = self.store.find(*keys) if keys else None
            duplicate = duplicate_issue(data, prior, name, chash)
            if duplicate:
                result.issues.append(duplicate)
            result.issues.extend(run_controls(data, self.registry, self.store, text))

        result.status = "needs_review" if result.errors else "accepted"
        result.summary = summarize(data, text, self.llm, use_llm=not security_issues)

        if keys:
            self.store.record(keys[0], keys[1], name, chash, result.status, data.total,
                              data.currency, po_key(data.po_number) if data.po_number else None)
        return result

    # ── OCR quality ───────────────────────────────────────────────────────
    def _ocr_issues(self, result: DocResult, loaded: LoadedDocument, segment: Segment) -> List[Issue]:
        pages = [loaded.ocr_pages[p] for p in range(segment.first_page, segment.last_page + 1)
                 if p in loaded.ocr_pages]
        if not pages:
            return []

        lines = sum(p.lines for p in pages) or 1
        confidence = sum(p.mean_confidence * p.lines for p in pages) / lines
        low_numeric = sum(p.low_numeric_lines for p in pages)
        result.ocr_engine, result.ocr_confidence = loaded.ocr_engine, round(confidence, 1)

        issues = [Issue("warning", "ocr",
                        f"Text was read by OCR ({loaded.ocr_engine}, mean confidence {confidence:.0f}%). "
                        "Check the amounts against the original image.")]
        if confidence < cfg.OCR_MIN_CONFIDENCE:
            issues.append(Issue("error", "ocr",
                                f"OCR confidence {confidence:.0f}% is below the minimum {cfg.OCR_MIN_CONFIDENCE:.0f}%"))
        if low_numeric:
            issues.append(Issue("error", "ocr",
                                f"{low_numeric} line(s) containing numbers were read with confidence below "
                                f"{cfg.OCR_MIN_NUMERIC_CONFIDENCE:.0f}%: a digit may be wrong"))
        if cfg.OCR_FORCE_REVIEW:
            issues.append(Issue("error", "ocr", "Policy: every OCR document must be reviewed by a person"))
        return issues

    # ── extraction with one repair attempt ────────────────────────────────
    def _extract_with_repair(self, doc_type: str, text: str, ocr: bool = False):
        feedback: Optional[Tuple[str, List[str]]] = None
        previous_raw: Optional[dict] = None
        last_good = None
        attempts = 0

        while True:
            try:
                raw = extract_raw(doc_type, text, self.llm, feedback)
                previous_raw = raw
                data, issues = build_data(doc_type, raw)
                issues = issues + validate(doc_type, data, text, ocr)
                last_good = (data, issues)
            except ExtractionError as exc:
                if last_good is not None:
                    data, issues = last_good
                    issues = issues + [Issue("warning", "extraction",
                                             f"Repair attempt failed ({exc}); kept the first extraction")]
                    return data, issues, attempts
                data, issues = None, [Issue("error", "extraction", str(exc))]

            errors = [i for i in issues if i.severity == "error"]
            if not errors or attempts >= cfg.MAX_REPAIR_ATTEMPTS:
                return data, issues, attempts

            attempts += 1
            feedback = (describe_previous(previous_raw), [i.message for i in errors])
