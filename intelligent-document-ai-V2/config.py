"""
config.py — Settings for the Intelligent Document AI.

LINE BY LINE (non-obvious lines):
  BASE_DIR = Path(__file__).parent
      Paths are anchored to this file, so the program works from any folder.

  MAX_REPAIR_ATTEMPTS = 1
      If validation finds errors, the LLM gets ONE chance to correct its
      extraction. More attempts cost more and tempt the model to "fix"
      numbers by inventing them. The grounding check (validator.py) is what
      makes even one repair safe.

  MAX_DOC_CHARS = 12000
      Longest document text sent to the model (about 3,000 tokens, which fits
      the 8,192-token context together with the schema). Longer documents are
      cut, and the pipeline FLAGS them for review, because the cut-off part
      might hold the totals.

  MAX_FILE_BYTES = 10_000_000
      Refuse files over 10 MB before reading them. Untrusted input must not
      be able to exhaust memory.

  MONEY_TOLERANCE = Decimal("0.01")
      Two amounts count as equal if they differ by at most one cent
      (rounding). We use Decimal, never float, for money: floats cannot
      represent 0.10 exactly, so 0.1 + 0.2 != 0.3 in float arithmetic.
      The string "0.01" (not the float 0.01) keeps the exact value.

  ALLOWED_CURRENCIES
      A frozenset of ISO 4217 codes the validator accepts. Anything else
      (a made-up "XYZ") is an error.
"""

import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # ── LLM (Ollama, local) ───────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
    TIMEOUT: int         = int(os.getenv("TIMEOUT", 180))

    # ── Folders ───────────────────────────────────────────────────────────
    INBOX_DIR: str  = os.getenv("INBOX_DIR", str(BASE_DIR / "inbox"))
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))

    # ── Pipeline behaviour ────────────────────────────────────────────────
    MAX_REPAIR_ATTEMPTS: int = int(os.getenv("MAX_REPAIR_ATTEMPTS", 1))
    MAX_DOC_CHARS: int       = int(os.getenv("MAX_DOC_CHARS", 12000))
    MAX_FILE_BYTES: int      = int(os.getenv("MAX_FILE_BYTES", 10_000_000))

    # ── Controls (v2): history, vendors, purchase orders ──────────────────
    DB_PATH: str       = os.getenv("DB_PATH", str(BASE_DIR / "data" / "history.db"))
    REGISTRY_PATH: str = os.getenv("REGISTRY_PATH", str(BASE_DIR / "data" / "registry.json"))
    APPROVAL_THRESHOLD: Decimal = Decimal(os.getenv("APPROVAL_THRESHOLD", "10000"))
    ANOMALY_FACTOR: Decimal     = Decimal(os.getenv("ANOMALY_FACTOR", "3"))
    ANOMALY_MIN_HISTORY: int    = int(os.getenv("ANOMALY_MIN_HISTORY", 3))

    # ── OCR (v2) ──────────────────────────────────────────────────────────
    OCR_ENGINE: str   = os.getenv("OCR_ENGINE", "auto")      # auto | rapidocr | tesseract | none
    OCR_DPI: int      = int(os.getenv("OCR_DPI", 200))
    OCR_MIN_CONFIDENCE: float         = float(os.getenv("OCR_MIN_CONFIDENCE", 85))
    OCR_MIN_NUMERIC_CONFIDENCE: float = float(os.getenv("OCR_MIN_NUMERIC_CONFIDENCE", 90))
    OCR_FORCE_REVIEW: bool = os.getenv("OCR_FORCE_REVIEW", "false").lower() in ("1", "true", "yes")
    MIN_PAGE_TEXT_CHARS: int = int(os.getenv("MIN_PAGE_TEXT_CHARS", 20))
    MAX_PDF_PAGES: int       = int(os.getenv("MAX_PDF_PAGES", 100))
    MAX_IMAGE_PIXELS: int    = int(os.getenv("MAX_IMAGE_PIXELS", 50_000_000))
    MAX_DOCS_PER_FILE: int   = int(os.getenv("MAX_DOCS_PER_FILE", 50))

    # ── Validation ────────────────────────────────────────────────────────
    MONEY_TOLERANCE: Decimal = Decimal(os.getenv("MONEY_TOLERANCE", "0.01"))
    ALLOWED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "INR", "CAD", "AUD", "JPY", "CHF"})


cfg = Config()
