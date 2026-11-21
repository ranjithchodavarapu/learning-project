"""
loader.py — Reads a document file into pages of plain text, defensively.

Documents are UNTRUSTED input. This module refuses anything it cannot read
safely and raises DocumentError with a clear reason.

WHAT'S NEW IN v2:
  * Scanned PDFs and image files (png, jpg, tif) are read with OCR.
  * Text is kept PAGE BY PAGE (joined with a form-feed "\\f"), so the splitter can
    separate several invoices that were scanned into one file.
  * Per-page OCR confidence is recorded, so low-quality scans go to review.

LINE BY LINE (non-obvious lines):
  read_document(path)    (the v1 function, unchanged in behaviour)
      Never runs OCR; returns text only. Kept so existing callers and tests keep
      working. New code uses load_document().

  need = pages with fewer than MIN_PAGE_TEXT_CHARS (20) characters
      A page with a text layer is read directly (exact, free). Only pages
      without one are rendered to an image and OCR'd. Mixed PDFs work.

  MAX_PDF_PAGES / MAX_IMAGE_PIXELS
      OCR cost grows with page count and pixel count. A 5,000-page PDF, or an
      image with a billion pixels, is a denial-of-service attempt or an accident.
      Both are refused before any heavy work.

  width_pt * scale * height_pt * scale
      The size of the rendered image in pixels, computed BEFORE rendering, so an
      absurd page size is rejected instead of exhausting memory.

  img.width * img.height checked right after Image.open()
      Image.open reads only the header; the pixels are not decoded yet, so the
      check is cheap and happens before the expensive convert().

  ImageSequence.Iterator
      A multi-page TIFF is a sequence of frames; each becomes a page.

  except (...) as exc: raise DocumentError(...)
      Corrupt files are normal in the real world. They become a clear
      `failed` status for that one document, not a crashed batch.
"""

from pathlib import Path
from typing import Dict, Iterator, List

from config import cfg
from models import DocumentError, LoadedDocument, PageOcr
from ocr import OcrUnavailable, get_engine

TEXT_SUFFIXES = {".txt", ".md"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}


def read_document(path) -> str:
    """v1 API: text only, never uses OCR."""
    return load_document(path, use_ocr=False).text


def load_document(path, ocr_engine=None, use_ocr: bool = True) -> LoadedDocument:
    path = Path(path)

    if not path.is_file():
        raise DocumentError(f"File not found: {path.name}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported file type '{path.suffix}'")
    if path.stat().st_size > cfg.MAX_FILE_BYTES:
        raise DocumentError(f"File is larger than {cfg.MAX_FILE_BYTES} bytes")

    if suffix == ".pdf":
        doc = _load_pdf(path, ocr_engine, use_ocr)
    elif suffix in IMAGE_SUFFIXES:
        doc = _load_image(path, ocr_engine, use_ocr)
    else:
        doc = LoadedDocument(pages=path.read_text(encoding="utf-8", errors="replace").split("\f"))

    if not any(page.strip() for page in doc.pages):
        raise DocumentError("Document contains no extractable text")
    return doc


# ── OCR plumbing ──────────────────────────────────────────────────────────
def _engine(ocr_engine):
    if ocr_engine is not None:
        return ocr_engine
    try:
        engine = get_engine()
    except OcrUnavailable as exc:
        raise DocumentError(f"This document needs OCR. {exc}")
    if engine is None:
        raise DocumentError("This document needs OCR, but OCR is switched off (OCR_ENGINE=none)")
    return engine


def _render_pdf_pages(path: Path, indices: List[int]) -> Iterator:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise DocumentError("Scanned PDFs need pypdfium2: pip install pypdfium2 pillow")

    scale = cfg.OCR_DPI / 72                    # PDF points are 1/72 inch
    try:
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in indices:
                page = pdf[i]
                width_pt, height_pt = page.get_size()
                if width_pt * scale * height_pt * scale > cfg.MAX_IMAGE_PIXELS:
                    raise DocumentError(f"Page {i + 1} is too large to OCR safely")
                yield page.render(scale=scale).to_pil()
        finally:
            pdf.close()
    except DocumentError:
        raise
    except Exception as exc:                    # pdfium raises its own error types
        raise DocumentError(f"PDF pages could not be rendered for OCR: {exc}")


# ── PDFs ──────────────────────────────────────────────────────────────────
def _load_pdf(path: Path, ocr_engine, use_ocr: bool) -> LoadedDocument:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        raise DocumentError("PDF support needs pypdf: pip install pypdf")

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentError("PDF is password-protected")
        if len(reader.pages) > cfg.MAX_PDF_PAGES:
            raise DocumentError(f"PDF has {len(reader.pages)} pages; the limit is {cfg.MAX_PDF_PAGES}")
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, AttributeError) as exc:
        raise DocumentError(f"PDF could not be read: {exc}")

    need = [i for i, text in enumerate(pages) if len(text.strip()) < cfg.MIN_PAGE_TEXT_CHARS]
    if not need:
        return LoadedDocument(pages=pages)

    if not use_ocr:
        if not any(p.strip() for p in pages):
            raise DocumentError("PDF has no extractable text (a scan? turn OCR on)")
        return LoadedDocument(pages=pages)

    engine = _engine(ocr_engine)
    ocr_pages: Dict[int, PageOcr] = {}
    for i, image in zip(need, _render_pdf_pages(path, need)):
        result = engine.read_image(image)
        pages[i] = result.text
        ocr_pages[i] = result.page_ocr(i)
    return LoadedDocument(pages=pages, ocr_pages=ocr_pages, ocr_engine=engine.name)


# ── images ────────────────────────────────────────────────────────────────
def _load_image(path: Path, ocr_engine, use_ocr: bool) -> LoadedDocument:
    if not use_ocr:
        raise DocumentError("Image files need OCR, which is switched off")
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        raise DocumentError("Image files need Pillow: pip install pillow")

    engine = _engine(ocr_engine)
    try:
        with Image.open(path) as img:
            if img.width * img.height > cfg.MAX_IMAGE_PIXELS:
                raise DocumentError(f"Image is larger than {cfg.MAX_IMAGE_PIXELS} pixels")
            frames = []
            for n, frame in enumerate(ImageSequence.Iterator(img)):
                if n >= cfg.MAX_PDF_PAGES:
                    raise DocumentError(f"Image has more than {cfg.MAX_PDF_PAGES} frames")
                frames.append(frame.convert("RGB"))
    except DocumentError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DocumentError(f"Image could not be read: {exc}")

    pages, ocr_pages = [], {}
    for i, frame in enumerate(frames):
        result = engine.read_image(frame)
        pages.append(result.text)
        ocr_pages[i] = result.page_ocr(i)
    return LoadedDocument(pages=pages, ocr_pages=ocr_pages, ocr_engine=engine.name)


def list_documents(folder) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise DocumentError(f"Folder not found: {folder}")
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                  and not p.name.lower().startswith("readme"))      # instructions, not a document
