"""
textutil.py — Parsing and matching helpers used by the extractor and validator.

THE BIG IDEA: GROUNDING
  An LLM can invent a plausible value ("INV-2024-0042" when the document says
  "INV-2024-0024"). To catch that, every extracted value is searched for in the
  SOURCE TEXT. A value that does not appear in the document was not read from
  it. These helpers make that search tolerant of formatting differences
  (1,225.00 vs 1.225,00, "15 March 2024" vs 2024-03-15) but strict about facts.

LINE BY LINE (non-obvious lines):
  normalize(text)
      NFKC folds look-alike forms (full-width letters, "ﬁ" ligature, composed
      vs decomposed "ü"). Category "Cf" characters (zero-width space, soft
      hyphen) are removed so invisible characters cannot hide or fake a match.
      casefold() is a stronger lower(): "Straße" and "STRASSE" compare equal.
      Both the document and the extracted value go through the same function,
      so they always compare like with like.

  to_decimal(value)
      Turns "$1,234.50", "€ 1.234,56", "(150.00)", 1234.5 ... into Decimal.
      * bool is checked first because in Python True is an int (True == 1).
      * floats go through str() first: Decimal(0.1) is 0.1000000000000000055...
        but Decimal("0.1") is exactly 0.1.
      * If BOTH "," and "." appear, whichever comes LAST is the decimal mark
        (1.234,56 is European, 1,234.56 is US).
      * A lone comma is a decimal mark when 1-2 digits follow ("12,5"), and a
        thousands mark when exactly 3 digit groups follow ("1,234").
      * Any leftover letter means "this is not a plain number" -> None.
        This stops "1e3" from silently becoming 13.
      * Limitation: a lone dot is always a decimal point, so a European
        "1.234" (meaning 1234) is read as 1.234.

  to_iso_date(value)
      Accepts ISO dates, "15 March 2024", "March 15, 2024", and "15.03.2024".
      Slash dates ("03/04/2024") are deliberately NOT accepted: they mean
      3 April in Europe and March 4 in the US, and guessing is how invoices
      get paid on the wrong day. Returns None instead.

  numbers_in_text(text)
      Every number in the document as a set of Decimals, so "is 1225.00 in the
      document?" is one set lookup. Equal Decimals hash equally (1225.00 and
      1225 are the same member).

  date_renderings(iso)
      Every common way the date could be WRITTEN, so a grounded date is found
      whether the document says "2024-03-15", "15.03.2024" or "March 15th, 2024".

  contains(text, phrase)
      Whole-token search: "5/3/2024" must NOT match inside "15/3/2024",
      and "INV-42" must not match inside "INV-420". The lookarounds
      (?<!\\w) and (?!\\w) mean "not preceded / followed by a word character".
"""

import json
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Set

from models import ExtractionError

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MONTH_ABBR = [m[:3] for m in MONTHS]

_CURRENCY_CODES = re.compile(r"\b(?:usd|eur|gbp|inr|cad|aud|jpy|chf)\b", re.I)
_NUMBER_TOKEN = re.compile(r"\d[\d.,]*\d|\d")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", text).strip().casefold()


# ── money ─────────────────────────────────────────────────────────────────
def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        number = Decimal(str(value))
    elif isinstance(value, str):
        number = _parse_money_string(value)
    else:
        return None
    if number is None or not number.is_finite():
        return None
    return number


def _parse_money_string(raw: str) -> Optional[Decimal]:
    s = unicodedata.normalize("NFKC", raw).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = _CURRENCY_CODES.sub("", s)
    s = re.sub(r"[$€£₹¥\s]", "", s).strip("()")
    if s.startswith("-"):
        negative, s = True, s[1:]
    if not s or re.search(r"[^\d.,]", s):
        return None

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")      # 1.234,56
        else:
            s = s.replace(",", "")                        # 1,234.56
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            s = s.replace(",", "")                        # 1,234 / 1,234,567
        elif re.fullmatch(r"\d+,\d{1,2}", s):
            s = s.replace(",", ".")                       # 12,5 / 1234,56
        else:
            return None

    if not re.fullmatch(r"\d+(\.\d+)?|\.\d+", s):
        return None
    try:
        number = Decimal(s)
    except InvalidOperation:
        return None
    return -number if negative else number


def numbers_in_text(text: str) -> Set[Decimal]:
    found: Set[Decimal] = set()
    for token in _NUMBER_TOKEN.findall(text):
        number = to_decimal(token)
        if number is not None:
            found.add(number)
    return found


# ── dates ─────────────────────────────────────────────────────────────────
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d %b %Y",
                 "%b %d, %Y", "%B %d %Y", "%d.%m.%Y")


def to_iso_date(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = unicodedata.normalize("NFKC", value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def date_renderings(iso: str) -> Set[str]:
    d = date.fromisoformat(iso)
    mm, dd, yyyy = f"{d.month:02d}", f"{d.day:02d}", str(d.year)
    out = {
        f"{yyyy}-{mm}-{dd}", f"{dd}/{mm}/{yyyy}", f"{mm}/{dd}/{yyyy}",
        f"{dd}.{mm}.{yyyy}", f"{dd}-{mm}-{yyyy}",
        f"{d.day}/{d.month}/{yyyy}", f"{d.month}/{d.day}/{yyyy}",
        f"{d.day}.{d.month}.{yyyy}",
    }
    for day in {str(d.day), dd, _ordinal(d.day)}:
        for month in (MONTHS[d.month - 1], MONTH_ABBR[d.month - 1]):
            out.add(f"{day} {month} {yyyy}")
            out.add(f"{month} {day}, {yyyy}")
            out.add(f"{month} {day} {yyyy}")
    return out


# ── matching ──────────────────────────────────────────────────────────────
def contains(text: str, phrase: str) -> bool:
    """Whole-token search. Both arguments must already be normalized."""
    if not phrase:
        return False
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def mostly_grounded(value: str, text: str, threshold: float = 0.6) -> bool:
    """For free-text fields: do most of the value's content words appear in the text?"""
    words = [w for w in re.findall(r"\w+", normalize(value)) if len(w) > 3]
    if not words:
        return True
    text_words = set(re.findall(r"\w+", text))
    return sum(w in text_words for w in words) / len(words) >= threshold


# ── model output ──────────────────────────────────────────────────────────
def parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ExtractionError("The model's reply contained no JSON object")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"The model's reply was not valid JSON: {exc}")

    if not isinstance(obj, dict):
        raise ExtractionError("The model's reply was JSON but not an object")
    return obj


# ── vendors, accounts, purchase orders (v2) ───────────────────────────────
def vendor_key(name: str) -> str:
    """'Northwind Supplies Ltd.' and 'NORTHWIND SUPPLIES LTD' get the same key."""
    return re.sub(r"[^a-z0-9]+", "", normalize(name))


def normalize_account(value: str) -> str:
    """Account numbers/IBANs are written with spaces: 'DE89 3704 ...'. Compare without them."""
    return re.sub(r"[\s\-.]", "", unicodedata.normalize("NFKC", value)).upper()


def po_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKC", value).upper())


_IBAN_SHAPE = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}")


def looks_like_iban(value: str) -> bool:
    return _IBAN_SHAPE.fullmatch(normalize_account(value)) is not None


def iban_valid(value: str) -> bool:
    """ISO 13616 mod-97 check: catches typos, OCR digit errors and invented numbers."""
    s = normalize_account(value)
    if not _IBAN_SHAPE.fullmatch(s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)      # A=10 ... Z=35
    return int(digits) % 97 == 1


# Standard IBAN lengths by country (a sample of common ones). Knowing the exact length
# lets us read "FR14 2004 1010 ..." with spaces without swallowing the next word, and
# makes accidental matches (1 chance in 97 per candidate) very rare.
IBAN_LENGTHS = {"AT": 20, "BE": 16, "CH": 21, "CZ": 24, "DE": 22, "DK": 18, "ES": 24,
                "FI": 18, "FR": 27, "GB": 22, "GR": 27, "HU": 28, "IE": 22, "IT": 27,
                "LU": 20, "NL": 18, "NO": 15, "PL": 28, "PT": 25, "SE": 24}


def find_ibans(text: str) -> Set[str]:
    """Every checksum-valid IBAN written in the text, normalised (no spaces).

    Independent of the LLM on purpose: the bank-change control must still work if
    the model forgets to extract the account.
    """
    upper = unicodedata.normalize("NFKC", text).upper()
    found: Set[str] = set()
    for match in re.finditer(r"(?<![A-Z0-9])[A-Z]{2}\d{2}", upper):
        length = IBAN_LENGTHS.get(match.group()[:2])
        if not length:
            continue
        chars, i = [], match.start()
        while i < len(upper) and len(chars) < length:
            ch = upper[i]
            if ch.isascii() and ch.isalnum():
                chars.append(ch)
            elif ch not in " -":                  # spaces/hyphens are separators; anything else ends it
                break
            i += 1
        ends_cleanly = i >= len(upper) or not (upper[i].isascii() and upper[i].isalnum())
        candidate = "".join(chars)
        if len(chars) == length and ends_cleanly and iban_valid(candidate):
            found.add(candidate)
    return found
