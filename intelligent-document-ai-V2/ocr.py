"""
ocr.py — Reads text from page images, and measures how much to trust it.

TWO ENGINES, SAME INTERFACE:
  rapidocr    pip-only (rapidocr-onnxruntime), runs on CPU, needs no admin rights.
              The default when installed.
  tesseract   needs the `tesseract` program installed on the machine plus
              `pip install pytesseract`.
  `OCR_ENGINE=auto` tries rapidocr first, then tesseract.

WHY CONFIDENCE MATTERS MORE HERE THAN ANYWHERE:
  OCR can misread a digit (8 for 3, 1 for 7). The grounding check then searches
  the misread text and PASSES, because the wrong number really is in the text we
  produced. So OCR quality has to be measured separately: documents read with low
  confidence, especially on lines containing digits, are routed to a human
  (see pipeline.py). OCR can also be confidently wrong; that is a limitation.

LINE BY LINE (non-obvious lines):
  assemble_lines(boxes)
      OCR engines return each text REGION as its own box. A table row such as
      "Office chairs | 4 | 120.00 | 480.00" comes back as FOUR separate boxes.
      Without rebuilding the row, the extractor would see the cells in an
      arbitrary order and could pair a quantity with the wrong price.
      Steps: sort by vertical centre; start a new row when the centre jumps by
      more than 0.6x the median box height; sort each row left to right; join
      cells with two spaces when there is a wide gap, one space when close.

  numeric_confidence = min(...) over parts containing a digit
      A line's score is the average, but a line like "Total: 1,323.00" could
      average 97% while the "1,323.00" part is 62%. We keep the WORST digit-
      bearing part, because digits are what we pay money for.

  float(score) * 100
      RapidOCR reports 0-1; Tesseract reports 0-100. Everything is stored 0-100.

  _ENGINE_CACHE
      Loading an OCR model takes a second or two. Engines are created once.
"""

import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import cfg
from models import PageOcr


class OcrUnavailable(Exception):
    """No usable OCR engine on this machine."""


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    confidence: float                       # 0-100


@dataclass
class OcrLine:
    text: str
    confidence: float                       # mean over the line
    numeric_confidence: Optional[float]     # worst digit-bearing part, or None
    y: float


@dataclass
class OcrResult:
    lines: List[OcrLine]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        return sum(l.confidence for l in self.lines) / len(self.lines) if self.lines else 0.0

    def page_ocr(self, page_index: int) -> PageOcr:
        low = sum(1 for l in self.lines
                  if l.numeric_confidence is not None
                  and l.numeric_confidence < cfg.OCR_MIN_NUMERIC_CONFIDENCE)
        return PageOcr(page=page_index, mean_confidence=round(self.mean_confidence, 1),
                       lines=len(self.lines), low_numeric_lines=low)


def assemble_lines(boxes: List[Box]) -> List[OcrLine]:
    boxes = [b for b in boxes if b.text.strip()]
    if not boxes:
        return []

    heights = sorted(b.y1 - b.y0 for b in boxes)
    median_h = heights[len(heights) // 2] or 1.0

    boxes.sort(key=lambda b: (b.y0 + b.y1) / 2)
    rows: List[List[Box]] = []
    centres: List[float] = []
    for b in boxes:
        centre = (b.y0 + b.y1) / 2
        if rows and abs(centre - centres[-1]) <= 0.6 * median_h:
            rows[-1].append(b)
            centres[-1] = sum((x.y0 + x.y1) / 2 for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([b])
            centres.append(centre)

    lines: List[OcrLine] = []
    for row, centre in zip(rows, centres):
        row.sort(key=lambda b: b.x0)
        text = row[0].text.strip()
        for prev, cur in zip(row, row[1:]):
            gap = cur.x0 - prev.x1
            text += ("  " if gap > 1.2 * median_h else " ") + cur.text.strip()
        digit_parts = [b.confidence for b in row if re.search(r"\d", b.text)]
        lines.append(OcrLine(
            text=text,
            confidence=sum(b.confidence for b in row) / len(row),
            numeric_confidence=min(digit_parts) if digit_parts else None,
            y=centre,
        ))
    return lines


# ── engines ───────────────────────────────────────────────────────────────
class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR        # ImportError -> handled by get_engine
        self._engine = RapidOCR()

    def read_image(self, image) -> OcrResult:
        import numpy as np
        result, _ = self._engine(np.array(image.convert("RGB")))
        boxes = []
        for points, text, score in (result or []):
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            boxes.append(Box(min(xs), min(ys), max(xs), max(ys), text, float(score) * 100))
        return OcrResult(assemble_lines(boxes))


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        if not shutil.which("tesseract"):
            raise OcrUnavailable("the 'tesseract' program is not installed")
        self._pt = pytesseract

    def read_image(self, image) -> OcrResult:
        data = self._pt.image_to_data(image.convert("RGB"), output_type=self._pt.Output.DICT)
        boxes = []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if word.strip() and conf >= 0:               # -1 marks layout rows, not words
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                boxes.append(Box(x, y, x + w, y + h, word, conf))
        return OcrResult(assemble_lines(boxes))


_FACTORIES = {"rapidocr": RapidOcrEngine, "tesseract": TesseractEngine}
_ENGINE_CACHE: Dict[str, object] = {}


def get_engine(preference: Optional[str] = None):
    """Returns an engine, or None when OCR is switched off (OCR_ENGINE=none)."""
    pref = (preference or cfg.OCR_ENGINE).lower()
    if pref == "none":
        return None
    order = {"auto": ["rapidocr", "tesseract"], "rapidocr": ["rapidocr"],
             "tesseract": ["tesseract"]}.get(pref)
    if order is None:
        raise OcrUnavailable(f"Unknown OCR engine '{pref}' (use auto, rapidocr, tesseract or none)")

    problems = []
    for name in order:
        if name in _ENGINE_CACHE:
            return _ENGINE_CACHE[name]
        try:
            _ENGINE_CACHE[name] = _FACTORIES[name]()
            return _ENGINE_CACHE[name]
        except Exception as exc:                         # missing package, missing program, bad model
            problems.append(f"{name}: {exc}")
    raise OcrUnavailable("No OCR engine is available (" + "; ".join(problems) +
                         "). Install one with:  pip install rapidocr-onnxruntime pypdfium2 pillow")
