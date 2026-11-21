"""v2 tests: OCR assembly and engines, loader, multi-document pipeline, history across runs,
evaluation harness, reporter and CLI."""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import loader
import ocr
import pipeline as pipeline_module
from config import cfg
from evaluate import evaluate_results, field_matches, load_truth, render_markdown, run_evaluation
from loader import list_documents, load_document
from make_samples import BUNDLE, render_image, write_text_pdf
from models import DocResult, DocumentError, InvoiceData
from ocr import Box, OcrLine, OcrResult, OcrUnavailable, assemble_lines, get_engine
from pipeline import DocumentPipeline
from registry import PurchaseOrder, Registry, load_registry
from reporter import write_outputs
from store import InvoiceStore
from tests.helpers import (INBOX, INBOX_V2, REGISTRY, ROUTES, NoisyLLM, RoutedLLM,
                           ScriptedLLM, invoice_json)
from textutil import po_key, vendor_key

PROJECT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLEAN_TEXT = (INBOX / "invoice_clean.txt").read_text()
Image = pytest.importorskip("PIL.Image")

try:
    from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    HAVE_RAPIDOCR = True
except Exception:
    HAVE_RAPIDOCR = False
HAVE_TESSERACT = bool(shutil.which("tesseract")) and importlib.util.find_spec("pytesseract") is not None
needs_pdfium = pytest.mark.skipif(importlib.util.find_spec("pypdfium2") is None,
                                  reason="pypdfium2 not installed (needed to render scanned PDFs)")


class FakeOcr:
    """A stand-in OCR engine returning scripted text and confidence."""
    name = "fake-ocr"

    def __init__(self, texts, confidence=99.0, numeric=99.0, confidences=None):
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        self.confidences = confidences or [confidence]
        self.numeric, self.calls = numeric, 0

    def read_image(self, image):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        conf = self.confidences[min(self.calls, len(self.confidences) - 1)]
        self.calls += 1
        lines = [OcrLine(t, conf, self.numeric if any(c.isdigit() for c in t) else None, float(i))
                 for i, t in enumerate(text.splitlines()) if t.strip()]
        return OcrResult(lines)


def png(tmp_path, name="scan.png", size=(120, 80)):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return path


def image_pdf(tmp_path, name="scan.pdf", pages=1):
    path = tmp_path / name
    imgs = [Image.new("RGB", (200, 150), "white") for _ in range(pages)]
    imgs[0].save(path, "PDF", resolution=100, save_all=True, append_images=imgs[1:])
    return path


# ═══ OCR line assembly ═══════════════════════════════════════════════════
def box(x0, y0, x1, y1, text, conf=99.0):
    return Box(x0, y0, x1, y1, text, conf)


def test_table_cells_become_one_row():
    cells = [box(37, 408, 200, 441, "Office chairs"), box(428, 411, 440, 442, "4"),
             box(555, 411, 640, 440, "120.00"), box(696, 411, 780, 440, "480.00")]
    assert [l.text for l in assemble_lines(cells)] == ["Office chairs  4  120.00  480.00"]


def test_rows_come_out_top_to_bottom_whatever_the_input_order():
    boxes = [box(0, 200, 90, 230, "third"), box(0, 100, 90, 130, "first"),
             box(300, 150, 390, 180, "second-right"), box(0, 150, 90, 180, "second-left")]
    assert [l.text for l in assemble_lines(boxes)] == ["first", "second-left  second-right", "third"]


def test_close_boxes_get_one_space_far_boxes_two():
    lines = assemble_lines([box(0, 0, 60, 30, "Invoice"), box(68, 0, 140, 30, "Number:"), box(400, 0, 480, 30, "X-1")])
    assert lines[0].text == "Invoice Number:  X-1"


def test_a_taller_box_still_joins_its_row():
    lines = assemble_lines([box(0, 374, 60, 403, "Description"), box(394, 369, 430, 408, "Qty"), box(500, 374, 600, 401, "Price")])
    assert len(lines) == 1


def test_rows_far_apart_stay_separate():
    assert len(assemble_lines([box(0, 0, 80, 30, "a"), box(0, 40, 80, 70, "b")])) == 2


def test_blank_and_empty_input():
    assert assemble_lines([]) == [] and assemble_lines([box(0, 0, 5, 5, "   "), box(0, 9, 5, 14, "")]) == []


def test_line_confidence_and_numeric_confidence():
    line = assemble_lines([box(0, 0, 60, 30, "Total:", 99), box(100, 0, 180, 30, "1,323.00", 62), box(300, 0, 340, 30, "USD", 95)])[0]
    assert round(line.confidence, 1) == 85.3 and line.numeric_confidence == 62
    assert assemble_lines([box(0, 0, 60, 30, "Hello", 80)])[0].numeric_confidence is None


def test_the_worst_digit_part_counts_not_the_average():
    line = assemble_lines([box(0, 0, 60, 30, "12", 91), box(100, 0, 180, 30, "34", 99), box(300, 0, 340, 30, "words", 10)])[0]
    assert line.numeric_confidence == 91


def test_ocr_result_summary_and_page_ocr(monkeypatch):
    result = OcrResult([OcrLine("a 1", 90, 85.0, 0), OcrLine("b 2", 100, 95.0, 1), OcrLine("words", 80, None, 2)])
    assert result.text == "a 1\nb 2\nwords" and round(result.mean_confidence, 1) == 90.0
    assert result.page_ocr(3).low_numeric_lines == 1 and result.page_ocr(3).page == 3
    monkeypatch.setattr(cfg, "OCR_MIN_NUMERIC_CONFIDENCE", 99.0)
    assert result.page_ocr(0).low_numeric_lines == 2
    assert OcrResult([]).mean_confidence == 0.0 and OcrResult([]).text == ""


# ═══ engine selection ════════════════════════════════════════════════════
def test_engine_none_unknown_and_unavailable(monkeypatch):
    assert get_engine("none") is None
    with pytest.raises(OcrUnavailable, match="Unknown OCR engine"):
        get_engine("magic")
    monkeypatch.setattr(ocr, "_ENGINE_CACHE", {})
    boom = lambda: (_ for _ in ()).throw(RuntimeError("not installed"))
    monkeypatch.setitem(ocr._FACTORIES, "rapidocr", boom)
    monkeypatch.setitem(ocr._FACTORIES, "tesseract", boom)
    with pytest.raises(OcrUnavailable, match="pip install rapidocr-onnxruntime"):
        get_engine("auto")


def test_auto_falls_back_and_engines_are_cached(monkeypatch):
    monkeypatch.setattr(ocr, "_ENGINE_CACHE", {})
    created = []

    class Second:
        name = "second"

        def __init__(self):
            created.append(1)

    monkeypatch.setitem(ocr._FACTORIES, "rapidocr", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setitem(ocr._FACTORIES, "tesseract", Second)
    first = get_engine("auto")
    assert first.name == "second" and get_engine("auto") is first and len(created) == 1


# ═══ loader with OCR ═════════════════════════════════════════════════════
def test_image_is_read_by_ocr(tmp_path):
    engine = FakeOcr("Invoice Number: X-1\nTotal 5.00", confidence=97)
    doc = load_document(png(tmp_path), ocr_engine=engine)
    assert doc.pages == ["Invoice Number: X-1\nTotal 5.00"] and doc.ocr_engine == "fake-ocr"
    assert doc.ocr_pages[0].mean_confidence == 97 and doc.ocr_pages[0].lines == 2


def test_images_need_ocr_switched_on(tmp_path):
    with pytest.raises(DocumentError, match="need OCR, which is switched off"):
        load_document(png(tmp_path), use_ocr=False)


@needs_pdfium
def test_scanned_pdf_is_rendered_and_ocrd(tmp_path):
    engine = FakeOcr("Scanned text with enough characters")
    doc = load_document(image_pdf(tmp_path), ocr_engine=engine)
    assert doc.pages == ["Scanned text with enough characters"] and engine.calls == 1 and list(doc.ocr_pages) == [0]


def test_text_pdf_never_calls_ocr(tmp_path):
    p = tmp_path / "t.pdf"
    write_text_pdf(p, [["Invoice Number: INV-1 and some more text on the page"]])
    engine = FakeOcr("never used")
    doc = load_document(p, ocr_engine=engine)
    assert engine.calls == 0 and doc.ocr_pages == {} and doc.ocr_engine is None


@needs_pdfium
def test_mixed_pdf_only_ocrs_the_pages_without_text(tmp_path):
    p = tmp_path / "mixed.pdf"
    write_text_pdf(p, [["This page has a real text layer, long enough"], [""], ["Another real text page, also long enough"]])
    engine = FakeOcr("text read from the image of page two")
    doc = load_document(p, ocr_engine=engine)
    assert engine.calls == 1 and list(doc.ocr_pages) == [1]
    assert doc.pages[0].startswith("This page") and doc.pages[1] == "text read from the image of page two"


def test_scanned_pdf_with_ocr_off_gives_the_old_clear_error(tmp_path):
    with pytest.raises(DocumentError, match="no extractable text"):
        load_document(image_pdf(tmp_path), use_ocr=False)


def test_ocr_switched_off_in_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OCR_ENGINE", "none")
    with pytest.raises(DocumentError, match="switched off"):
        load_document(image_pdf(tmp_path))


def test_missing_ocr_engine_explains_what_to_install(tmp_path, monkeypatch):
    def nothing(*a, **k):
        raise OcrUnavailable("No OCR engine is available (nothing installed)")
    monkeypatch.setattr(loader, "get_engine", nothing)
    with pytest.raises(DocumentError, match="needs OCR. No OCR engine"):
        load_document(png(tmp_path))


def test_blank_scan_is_reported_not_accepted(tmp_path):
    with pytest.raises(DocumentError, match="no extractable text"):
        load_document(png(tmp_path), ocr_engine=FakeOcr(""))


def test_pdf_page_limit(tmp_path, monkeypatch):
    p = tmp_path / "long.pdf"
    write_text_pdf(p, [[f"page {i} with enough text to count"] for i in range(3)])
    monkeypatch.setattr(cfg, "MAX_PDF_PAGES", 2)
    with pytest.raises(DocumentError, match="limit is 2"):
        load_document(p)


def test_image_pixel_limit_checked_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(DocumentError, match="larger than 100 pixels"):
        load_document(png(tmp_path, size=(50, 50)), ocr_engine=FakeOcr("x"))


@needs_pdfium
def test_rendered_pdf_page_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(DocumentError, match="too large to OCR"):
        load_document(image_pdf(tmp_path), ocr_engine=FakeOcr("x"))


def test_corrupt_image_is_a_clean_error(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"this is not an image")
    with pytest.raises(DocumentError, match="could not be read"):
        load_document(p, ocr_engine=FakeOcr("x"))


def test_multi_frame_tiff_becomes_pages(tmp_path):
    frames = [Image.new("RGB", (60, 40), c) for c in ("white", "white", "white")]
    p = tmp_path / "multi.tiff"
    frames[0].save(p, save_all=True, append_images=frames[1:])
    doc = load_document(p, ocr_engine=FakeOcr(["page one", "page two", "page three"]))
    assert doc.pages == ["page one", "page two", "page three"] and sorted(doc.ocr_pages) == [0, 1, 2]


def test_image_suffixes_are_listed(tmp_path):
    for name in ("a.png", "b.JPG", "c.tif", "d.jpeg", "e.gif", "f.bmp", "g.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_documents(tmp_path)] == ["a.png", "b.JPG", "c.tif", "d.jpeg", "g.txt"]


# ═══ the real engines ════════════════════════════════════════════════════
@pytest.mark.skipif(not HAVE_RAPIDOCR, reason="rapidocr-onnxruntime not installed")
def test_rapidocr_reads_the_scanned_png():
    doc = load_document(INBOX_V2 / "scanned_invoice.png", ocr_engine=get_engine("rapidocr"))
    text = doc.text
    assert "INV-2024-0555" in text and "GB29 NWBK 6016 1331 9268 19" in text and "1,620.00" in text
    row = next(line for line in text.splitlines() if "Pallet" in line)
    assert "650.00" in row and "1,300.00" in row                  # the table row was rebuilt
    assert doc.ocr_pages[0].mean_confidence > 90 and doc.ocr_pages[0].low_numeric_lines == 0


@pytest.mark.skipif(not HAVE_RAPIDOCR, reason="rapidocr-onnxruntime not installed")
@needs_pdfium
def test_rapidocr_reads_the_image_only_pdf():
    doc = load_document(INBOX_V2 / "scanned_invoice.pdf", ocr_engine=get_engine("rapidocr"))
    assert "Globex Consulting GmbH" in doc.text and "1,785.00" in doc.text and list(doc.ocr_pages) == [0]


@pytest.mark.skipif(not HAVE_TESSERACT, reason="tesseract and pytesseract not installed")
def test_tesseract_reads_the_scanned_png():
    doc = load_document(INBOX_V2 / "scanned_invoice.png", ocr_engine=get_engine("tesseract"))
    assert "INV-2024-0555" in doc.text and "1,300.00" in doc.text and doc.ocr_engine == "tesseract"
    assert doc.ocr_pages[0].mean_confidence > 80


# ═══ pipeline: OCR confidence routing ════════════════════════════════════
def run_ocr(tmp_path, engine, llm=None, text_name="scan.png"):
    pipe = DocumentPipeline(llm or ScriptedLLM([invoice_json()]), ocr_engine=engine)
    return pipe.process(png(tmp_path, text_name))


def test_confident_ocr_is_accepted_with_a_warning(tmp_path):
    r = run_ocr(tmp_path, FakeOcr(CLEAN_TEXT, confidence=98, numeric=98))
    assert r.status == "accepted" and r.ocr_engine == "fake-ocr" and r.ocr_confidence == 98.0
    assert [i.field for i in r.warnings] == ["ocr"] and "Check the amounts" in r.warnings[0].message


def test_low_mean_confidence_goes_to_review(tmp_path):
    r = run_ocr(tmp_path, FakeOcr(CLEAN_TEXT, confidence=70, numeric=95))
    assert r.status == "needs_review" and any("below the minimum" in i.message for i in r.errors)


def test_a_low_confidence_number_goes_to_review_even_if_the_average_is_high(tmp_path):
    r = run_ocr(tmp_path, FakeOcr(CLEAN_TEXT, confidence=98, numeric=60))
    assert r.status == "needs_review" and any("a digit may be wrong" in i.message for i in r.errors)


def test_policy_can_force_every_ocr_document_to_review(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OCR_FORCE_REVIEW", True)
    r = run_ocr(tmp_path, FakeOcr(CLEAN_TEXT))
    assert r.status == "needs_review" and any("Policy" in i.message for i in r.errors)


def test_thresholds_are_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OCR_MIN_CONFIDENCE", 50.0)
    monkeypatch.setattr(cfg, "OCR_MIN_NUMERIC_CONFIDENCE", 50.0)
    assert run_ocr(tmp_path, FakeOcr(CLEAN_TEXT, confidence=60, numeric=60)).status == "accepted"


def test_ocr_that_drops_spaces_is_still_grounded():
    """Real RapidOCR output for scanned_invoice.pdf: 'Date:12June2024', 'T0TAL', ..."""
    garbled = (FIXTURES / "garbled_ocr_invoice.txt").read_text()
    reply = [json.dumps({"doc_type": "invoice"}), ROUTES["5503"]]
    with tempfile.TemporaryDirectory() as d:
        r = run_ocr(Path(d), FakeOcr(garbled, confidence=97, numeric=97), ScriptedLLM(reply))
    assert r.status == "accepted" and r.data.invoice_date == "2024-06-12" and r.data.total == Decimal("1785")


def test_the_same_garbled_text_is_not_trusted_when_it_did_not_come_from_ocr(tmp_path):
    p = tmp_path / "garbled.txt"
    p.write_text((FIXTURES / "garbled_ocr_invoice.txt").read_text())
    r = DocumentPipeline(ScriptedLLM([json.dumps({"doc_type": "invoice"}), ROUTES["5503"]])).process(p)
    assert r.status == "needs_review" and any(i.field == "invoice_date" for i in r.errors)


def test_ocr_off_gives_a_failed_result_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OCR_ENGINE", "none")
    r = DocumentPipeline(ScriptedLLM([invoice_json()])).process(png(tmp_path))
    assert r.status == "failed" and "switched off" in r.errors[0].message


@needs_pdfium
def test_each_document_in_a_scanned_bundle_gets_its_own_ocr_confidence(tmp_path):
    pages = ["\n".join(lines) for lines in BUNDLE]
    p = image_pdf(tmp_path, "bundle.pdf", pages=3)
    results = DocumentPipeline(RoutedLLM(), ocr_engine=FakeOcr(pages, confidences=[99, 60, 99])).process_file(p)
    assert [r.filename for r in results] == ["bundle.pdf#1", "bundle.pdf#2", "bundle.pdf#3"]
    assert [r.ocr_confidence for r in results] == [99.0, 60.0, 99.0]
    assert [r.status for r in results] == ["accepted", "needs_review", "accepted"]


# ═══ pipeline: bundles, failures, history ═════════════════════════════════
def test_bundle_pdf_becomes_three_documents():
    pipe = DocumentPipeline(RoutedLLM())
    results = pipe.process_file(INBOX_V2 / "bundle_three_invoices.pdf")
    assert [(r.filename, r.part, r.parts) for r in results] == [(f"bundle_three_invoices.pdf#{i}", i, 3) for i in (1, 2, 3)]
    assert [r.data.invoice_number for r in results] == ["B-2001", "B-2002", "B-2003"]
    assert all(r.status == "accepted" for r in results)


def test_process_refuses_to_hide_extra_documents():
    with pytest.raises(ValueError, match="contains 3 documents"):
        DocumentPipeline(RoutedLLM()).process(INBOX_V2 / "bundle_three_invoices.pdf")


def test_single_document_files_keep_their_plain_name():
    results = DocumentPipeline(ScriptedLLM([invoice_json()])).process_file(INBOX / "invoice_clean.txt")
    assert [(r.filename, r.part, r.parts) for r in results] == [("invoice_clean.txt", 1, 1)]


def test_too_many_documents_in_one_file(monkeypatch):
    monkeypatch.setattr(cfg, "MAX_DOCS_PER_FILE", 2)
    results = DocumentPipeline(RoutedLLM()).process_file(INBOX_V2 / "bundle_three_invoices.pdf")
    assert len(results) == 1 and results[0].status == "failed" and "limit is 2" in results[0].errors[0].message


def test_an_unexpected_crash_becomes_a_failed_row_not_a_dead_batch(monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("parser exploded")
    monkeypatch.setattr(pipeline_module, "classify", crash)
    results = DocumentPipeline(RoutedLLM()).process_file(INBOX_V2 / "bundle_three_invoices.pdf")
    assert len(results) == 3 and all(r.status == "failed" for r in results)
    assert "Unexpected error (RuntimeError: parser exploded)" in results[0].errors[0].message and results[2].part == 3


def test_a_crash_while_loading_becomes_a_failed_row(monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(pipeline_module, "load_document", crash)
    r = DocumentPipeline(RoutedLLM()).process(INBOX / "invoice_clean.txt")
    assert r.status == "failed" and "Could not read the file (RuntimeError: boom)" in r.errors[0].message


def simple_invoice(tmp_path, number, total, vendor="Northwind Supplies Ltd.", po=None, name=None):
    text = (f"Invoice Number: {number}\nInvoice Date: 2024-03-15\nFrom: {vendor}\nBill To: Contoso\n"
            f"Subtotal: {total:,.2f}\nTax: 0.00\nTotal Due: {total:,.2f}\nCurrency: USD\n" + (f"PO Number: {po}\n" if po else ""))
    path = tmp_path / (name or f"{number}.txt")
    path.write_text(text)
    reply = json.dumps({"invoice_number": number, "vendor": vendor, "customer": "Contoso", "invoice_date": "2024-03-15",
                        "currency": "USD", "line_items": [], "subtotal": float(total), "tax": 0.0, "total": float(total),
                        "po_number": po})
    return path, reply


def test_a_rerun_of_the_same_files_is_not_a_wave_of_duplicates(tmp_path):
    db = tmp_path / "h.db"
    path, reply = simple_invoice(tmp_path, "N-1", Decimal("100.00"))
    first = DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)
    second = DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)      # a new run
    assert first.status == second.status == "accepted" and not any("Duplicate" in i.message for i in second.issues)


def test_duplicates_are_caught_across_runs(tmp_path):
    db = tmp_path / "h.db"
    path, reply = simple_invoice(tmp_path, "N-1", Decimal("100.00"))
    DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)

    copy = tmp_path / "emailed_again.txt"                       # same content, different name
    copy.write_text(path.read_text())
    r = DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(copy)
    assert r.status == "needs_review" and any("already submitted as 'N-1.txt'" in i.message for i in r.errors)


def test_an_edited_file_reusing_an_invoice_number_is_a_duplicate(tmp_path):
    db = tmp_path / "h.db"
    path, reply = simple_invoice(tmp_path, "N-1", Decimal("100.00"))
    DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)
    path.write_text(path.read_text() + "\nRevised terms apply.\n")          # same name, new content
    r = DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)
    assert r.status == "needs_review" and any("Duplicate" in i.message for i in r.errors)


def test_vendor_history_builds_up_across_runs(tmp_path):
    db = tmp_path / "h.db"
    for number, total in (("H-1", "100.00"), ("H-2", "120.00"), ("H-3", "90.00")):
        path, reply = simple_invoice(tmp_path, number, Decimal(total))
        assert DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path).status == "accepted"
    path, reply = simple_invoice(tmp_path, "H-4", Decimal("400.00"))
    r = DocumentPipeline(ScriptedLLM([reply]), store=InvoiceStore(db)).process(path)
    assert r.status == "needs_review" and any("more than 3x" in i.message for i in r.errors)


def test_purchase_order_is_used_up_across_runs(tmp_path):
    db = tmp_path / "h.db"
    reg = Registry(vendors={}, purchase_orders={po_key("PO-7"): PurchaseOrder(
        "PO-7", vendor_key("Northwind Supplies Ltd."), Decimal("1000.00"), "USD")})
    p1, r1 = simple_invoice(tmp_path, "P-1", Decimal("700.00"), po="PO-7")
    p2, r2 = simple_invoice(tmp_path, "P-2", Decimal("400.00"), po="PO-7")
    assert DocumentPipeline(ScriptedLLM([r1]), store=InvoiceStore(db), registry=reg).process(p1).status == "accepted"
    r = DocumentPipeline(ScriptedLLM([r2]), store=InvoiceStore(db), registry=reg).process(p2)
    assert r.status == "needs_review" and any("exceeds purchase order PO-7" in i.message for i in r.errors)


def test_a_model_that_forgets_the_bank_account_cannot_switch_the_check_off():
    reg = load_registry(REGISTRY)
    forgetful = json.loads(ROUTES["A-1002"])
    forgetful["bank_account"] = None
    r = DocumentPipeline(ScriptedLLM([json.dumps(forgetful)]), registry=reg).process(INBOX_V2 / "acme_bank_changed.txt")
    assert r.status == "needs_review" and any("payment-redirection" in i.message for i in r.errors)


# ═══ the shipped samples, all of them ════════════════════════════════════
EXPECTED = {
    "contract_nda.txt": "accepted", "contract_services.txt": "accepted", "invoice_clean.txt": "accepted",
    "invoice_clean_resubmitted.txt": "needs_review", "invoice_european.txt": "accepted",
    "invoice_injection.txt": "needs_review", "invoice_math_error.txt": "needs_review", "memo_unknown.txt": "needs_review",
    "acme_bank_changed.txt": "needs_review", "acme_no_po.txt": "needs_review", "acme_ok.txt": "accepted",
    "acme_over_po.txt": "needs_review", "northwind_large.txt": "needs_review", "unknown_vendor.txt": "needs_review",
    "bundle_three_invoices.pdf#1": "accepted", "bundle_three_invoices.pdf#2": "accepted",
    "bundle_three_invoices.pdf#3": "accepted",
}


def test_every_sample_document_with_a_faithful_model():
    pipe = DocumentPipeline(RoutedLLM(), registry=load_registry(REGISTRY))
    results = {r.filename: r for r in pipe.process_folder(INBOX) + pipe.process_folder(INBOX_V2)}
    assert {k: v.status for k, v in results.items() if not k.startswith("scanned")} == EXPECTED
    assert set(results) - set(EXPECTED) == {"scanned_invoice.png", "scanned_invoice.pdf"}


@pytest.mark.skipif(not HAVE_RAPIDOCR, reason="rapidocr-onnxruntime not installed")
def test_scanned_samples_with_real_ocr_are_accepted():
    pipe = DocumentPipeline(RoutedLLM(), registry=load_registry(REGISTRY))
    results = {r.filename: r for r in pipe.process_folder(INBOX_V2)}
    for name in ("scanned_invoice.png", "scanned_invoice.pdf"):
        assert results[name].status == "accepted" and results[name].ocr_engine == "rapidocr", name
        assert results[name].ocr_confidence > 90


def test_sample_reasons_for_the_new_controls():
    pipe = DocumentPipeline(RoutedLLM(), registry=load_registry(REGISTRY))
    by = {r.filename: r for r in pipe.process_folder(INBOX) + pipe.process_folder(INBOX_V2)}
    why = lambda name: " | ".join(i.message for i in by[name].errors)
    assert "payment-redirection fraud" in why("acme_bank_changed.txt")
    assert "requires a purchase order" in why("acme_no_po.txt")
    assert "exceeds purchase order PO-2002" in why("acme_over_po.txt")
    assert "not on the approved vendor list" in why("unknown_vendor.txt")
    assert "auto-approval limit" in why("northwind_large.txt")
    assert by["acme_ok.txt"].errors == [] and by["acme_ok.txt"].data.po_number == "PO-1001"


# ═══ evaluation harness ══════════════════════════════════════════════════
@pytest.mark.parametrize("name, want, got, ok", [
    ("total", "1323.00", Decimal("1323.0"), True), ("total", "1323.00", Decimal("1323.01"), False),
    ("total", None, None, True), ("total", "5", None, False), ("tax", None, Decimal("0"), False),
    ("parties", ["B Ltd", "A Inc"], ["a inc", "B LTD"], True), ("parties", ["A"], ["A", "B"], False),
    ("bank_account", "DE89 3704 0044 0532 0130 00", "de89370400440532013000", True),
    ("bank_account", None, None, True), ("bank_account", "DE89", None, False),
    ("vendor", "Müller GmbH", "Mu\u0308ller  gmbh", True), ("vendor", "A", None, False),
    ("invoice_date", "2024-03-15", "2024-03-15", True), ("invoice_date", "2024-03-15", "2024-03-16", False),
    ("auto_renewal", True, True, True), ("auto_renewal", False, None, False), ("po_number", None, None, True),
    ("po_number", None, "PO-1", False), ("line_items", 3, [1, 2, 3], True), ("line_items", 3, [], False),
])
def test_field_matches(name, want, got, ok):
    assert field_matches(name, want, got) is ok


def result(name, status, doc_type="invoice", **fields):
    return DocResult(filename=name, status=status, doc_type=doc_type, data=InvoiceData(**fields) if fields else None)


def test_missed_bad_and_false_review_are_counted_separately():
    truth = {"good_ok": {"status": "accepted", "doc_type": "invoice"}, "good_held": {"status": "accepted", "doc_type": "invoice"},
             "bad_caught": {"status": "needs_review", "doc_type": "invoice"}, "bad_missed": {"status": "needs_review", "doc_type": "invoice"}}
    s = evaluate_results([result("good_ok", "accepted"), result("good_held", "needs_review"),
                          result("bad_caught", "needs_review"), result("bad_missed", "accepted")], truth)
    c = s["counts"]
    assert (c["graded"], c["status_ok"], c["false_review"], c["expected_good"], c["missed_bad"], c["expected_bad"]) == (4, 2, 1, 2, 1, 2)


def test_failed_documents_count_as_review_not_accepted():
    s = evaluate_results([result("d", "failed")], {"d": {"status": "needs_review", "doc_type": "invoice"}})
    assert s["counts"]["missed_bad"] == 0 and s["counts"]["status_ok"] == 0


def test_ocr_dependent_documents_are_shown_but_not_scored():
    truth = {"scan.png": {"status": "accepted", "doc_type": "invoice", "ocr_dependent": True, "fields": {"invoice_number": "A"}}}
    s = evaluate_results([result("scan.png", "needs_review", invoice_number="WRONG")], truth)
    assert s["counts"]["graded"] == 0 and s["counts"]["ungraded"] == 1 and s["counts"]["false_review"] == 0
    assert s["field_total"] == 0 and s["rows"][0]["scored"] is False


def test_a_missing_document_is_a_failed_expectation_not_a_skip():
    s = evaluate_results([], {"bundle#3": {"status": "accepted", "doc_type": "invoice"}})
    assert s["counts"]["missing"] == 1 and s["counts"]["false_review"] == 1 and s["rows"][0]["actual"] == "missing"


def test_unexpected_extra_documents_are_listed():
    s = evaluate_results([result("surprise", "accepted")], {})
    assert s["extra_documents"] == ["surprise"]


def test_field_accuracy_is_per_field():
    truth = {"d": {"status": "accepted", "doc_type": "invoice", "fields": {"invoice_number": "A", "vendor": "V", "total": "5"}}}
    s = evaluate_results([result("d", "accepted", invoice_number="A", vendor="WRONG", total=Decimal("5"))], truth)
    assert s["per_field"] == {"invoice_number": {"correct": 1, "total": 1}, "total": {"correct": 1, "total": 1},
                              "vendor": {"correct": 0, "total": 1}}
    assert round(s["field_accuracy"], 3) == 0.667 and s["rows"][0]["mismatches"] == [("vendor", "V", "WRONG")]


def test_markdown_report_lists_problems():
    s = evaluate_results([result("d", "accepted", vendor="X")], {"d": {"status": "needs_review", "doc_type": "invoice", "fields": {"vendor": "V"}}})
    s["model"] = "test-model"
    md = render_markdown(s)
    assert "Missed bad (bad document auto-accepted): 1 of 1" in md and "WRONG" in md and "wanted 'V', got 'X'" in md


def test_a_perfect_model_scores_perfectly():
    s = run_evaluation(RoutedLLM(), [INBOX, INBOX_V2], REGISTRY, load_truth(PROJECT / "eval" / "ground_truth.json"))
    c = s["counts"]
    assert c["status_ok"] == c["graded"] == 17 and c["missed_bad"] == 0 and c["false_review"] == 0
    assert s["field_accuracy"] == 1.0 and s["field_total"] == 74 and s["extra_documents"] == []


def test_a_mistake_prone_model_is_measured_but_never_gets_a_bad_document_accepted():
    s = run_evaluation(NoisyLLM(), [INBOX, INBOX_V2], REGISTRY, load_truth(PROJECT / "eval" / "ground_truth.json"))
    c = s["counts"]
    assert c["missed_bad"] == 0 and c["false_review"] == 2 and c["status_ok"] == 15
    assert s["field_accuracy"] < 1.0 and s["per_field"]["bank_account"]["correct"] < s["per_field"]["bank_account"]["total"]


def test_the_shipped_truth_file_covers_every_sample():
    truth = load_truth(PROJECT / "eval" / "ground_truth.json")
    assert set(truth) == set(EXPECTED) | {"scanned_invoice.png", "scanned_invoice.pdf"}
    assert all(not k.startswith("_") for k in truth)


# ═══ reporter ════════════════════════════════════════════════════════════
def test_csv_has_po_and_bank_columns_and_is_injection_safe(tmp_path):
    data = InvoiceData(invoice_number="N1", vendor="V", total=Decimal("5"), po_number="=PO", bank_account="+DE89")
    write_outputs([DocResult(filename="a.txt", status="accepted", doc_type="invoice", data=data)], tmp_path)
    row = next(csv.DictReader(open(tmp_path / "invoices.csv", encoding="utf-8")))
    assert row["po_number"] == "'=PO" and row["bank_account"] == "'+DE89" and list(row)[-3:] == ["po_number", "bank_account", "errors"]


def test_bundle_parts_get_safe_distinct_file_names_and_ocr_fields(tmp_path):
    results = [DocResult(filename=f"bundle.pdf#{i}", part=i, parts=3, ocr_engine="rapidocr", ocr_confidence=91.5) for i in (1, 2, 3)]
    write_outputs(results, tmp_path)
    names = sorted(p.name for p in (tmp_path / "results").glob("*.json"))
    assert names == ["bundle.pdf_1.json", "bundle.pdf_2.json", "bundle.pdf_3.json"]
    data = json.loads((tmp_path / "results" / "bundle.pdf_2.json").read_text())
    assert data["part"] == 2 and data["parts"] == 3 and data["ocr_engine"] == "rapidocr" and data["ocr_confidence"] == 91.5


# ═══ command line ════════════════════════════════════════════════════════
def cli(script, *args, **env):
    full = {**os.environ, "DB_PATH": str(Path(tempfile.mkdtemp()) / "h.db"), **env}
    return subprocess.run([sys.executable, script, *args], cwd=PROJECT, env=full, capture_output=True, text=True, timeout=120)


def test_cli_stops_on_a_broken_registry(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "a.txt").write_text("hello")
    proc = cli("main.py", "--skip-check", "--registry", str(bad), "--inbox", str(inbox), "--out", str(tmp_path / "o"))
    assert proc.returncode == 2 and "not valid JSON" in proc.stdout


def test_cli_creates_the_history_database_unless_told_not_to(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    (inbox / "a.txt").write_text(CLEAN_TEXT)
    db = tmp_path / "state" / "h.db"
    cli("main.py", "--skip-check", "--inbox", str(inbox), "--out", str(tmp_path / "o1"), DB_PATH=str(db), OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert db.exists()
    db2 = tmp_path / "state2" / "h.db"
    cli("main.py", "--skip-check", "--no-history", "--inbox", str(inbox), "--out", str(tmp_path / "o2"), DB_PATH=str(db2), OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert not db2.exists()


def test_cli_ocr_none_explains_why_a_scan_failed(tmp_path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    shutil.copy(INBOX_V2 / "scanned_invoice.png", inbox)
    out = tmp_path / "o"
    proc = cli("main.py", "--skip-check", "--ocr", "none", "--inbox", str(inbox), "--out", str(out), OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert proc.returncode == 0 and "switched off" in (out / "report.md").read_text()


def test_evaluate_cli_self_test_exits_zero_and_writes_reports(tmp_path):
    proc = cli("evaluate.py", "--fake", "faithful", "--out", str(tmp_path))
    assert proc.returncode == 0 and "Missed bad:" in proc.stdout
    report = json.loads((tmp_path / "eval_report.json").read_text())
    assert report["counts"]["missed_bad"] == 0 and (tmp_path / "eval_report.md").exists()


def test_evaluate_cli_exits_one_when_a_bad_document_is_accepted(tmp_path):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"invoice_clean.txt": {"status": "needs_review", "doc_type": "invoice"}}))
    proc = cli("evaluate.py", "--fake", "faithful", "--truth", str(truth), "--inbox", str(INBOX), "--out", str(tmp_path / "o"))
    assert proc.returncode == 1


def test_evaluate_cli_without_ollama_exits_two(tmp_path):
    proc = cli("evaluate.py", "--out", str(tmp_path), OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert proc.returncode == 2 and "Cannot reach Ollama" in proc.stdout


# ═══ scoring fixes after the first real-model run ════════════════════════
@pytest.mark.parametrize("name, want, got, ok", [
    ("governing_law", "State of Delaware", "the laws of the State of Delaware", True),    # the real model's answer
    ("governing_law", "England and Wales", "the laws of England and Wales", True),
    ("governing_law", "the laws of the State of Delaware", "State of Delaware", True),    # either direction
    ("governing_law", "State of Delaware", "STATE  of delaware", True),
    ("term", "12 months", "an initial term of 12 months", True),
    ("governing_law", "State of Delaware", "Singapore courts", False),
    ("governing_law", "State of Delaware", "State of New York", False),
    ("governing_law", "State of Delaware", None, False),
    ("governing_law", "State of Delaware", "", False),                                    # empty is not "contained"
    ("governing_law", None, None, True), ("governing_law", None, "State of Delaware", False),
    ("governing_law", "Delaware", "This Agreement is governed by the laws of the State of Delaware and "
                                  "the parties submit to its courts for all disputes arising hereunder", False),   # a pasted paragraph
])
def test_clause_fields_accept_reasonable_wording_but_not_wrong_or_empty_answers(name, want, got, ok):
    assert field_matches(name, want, got) is ok


@pytest.mark.parametrize("name, want, got", [
    ("vendor", "Acme", "Acme Corp"), ("vendor", "Acme Corp", "Acme"),                   # names stay exact
    ("invoice_number", "INV-1", "INV-12"), ("invoice_number", "INV-12", "INV-1"),
    ("title", "Services Agreement", "Master Services Agreement"),
    ("currency", "USD", "USDX"), ("po_number", "PO-1", "PO-10"),
])
def test_identifiers_and_names_stay_exact(name, want, got):
    assert field_matches(name, want, got) is False


def test_not_stated_means_null_not_false():
    assert field_matches("auto_renewal", None, None) is True
    assert field_matches("auto_renewal", None, False) is False          # inventing "no renewal" is a guess
    assert field_matches("auto_renewal", False, None) is False


def test_the_nda_really_never_mentions_renewal_so_null_is_the_right_truth():
    assert "renew" not in (INBOX / "contract_nda.txt").read_text().lower()
    assert load_truth(PROJECT / "eval" / "ground_truth.json")["contract_nda.txt"]["fields"]["auto_renewal"] is None
    assert json.loads(ROUTES["NON-DISCLOSURE"])["auto_renewal"] is None


def test_mismatches_are_shown_in_full_not_cut_off(monkeypatch):
    import evaluate
    from rich.console import Console
    long_got = "x" * 200
    s = evaluate_results([result("d", "accepted", vendor=long_got)],
                         {"d": {"status": "accepted", "doc_type": "invoice", "fields": {"vendor": "V"}}})
    s["model"] = "m"
    console = Console(width=120, record=True)
    monkeypatch.setattr(evaluate, "console", console)
    evaluate.print_summary(s)
    assert long_got in console.export_text().replace("\n", "").replace("│", "").replace(" ", "")
