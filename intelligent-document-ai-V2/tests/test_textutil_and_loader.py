"""Tests for text utilities, JSON parsing and the document loader."""

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config import cfg
from loader import list_documents, read_document
from models import DocumentError, ExtractionError
from textutil import (_ordinal, contains, date_renderings, mostly_grounded,
                      normalize, numbers_in_text, parse_json_object, to_decimal,
                      to_iso_date)
from tests.helpers import INBOX, make_pdf


# ── money ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw, expected", [
    ("$1,234.50", "1234.50"), ("€ 1.234,56", "1234.56"), ("1,234.5", "1234.5"),
    ("(150.00)", "-150.00"), ("-1,000", "-1000"), ("12,5", "12.5"),
    ("1,234", "1234"), ("1,234,567", "1234567"), ("1234,56", "1234.56"),
    ("USD 1,200.00", "1200.00"), ("  42  ", "42"), ("0,00", "0.00"),
    (1234.5, "1234.5"), (0.1, "0.1"), (7, "7"), (Decimal("3.10"), "3.10"),
])
def test_to_decimal_parses(raw, expected):
    assert to_decimal(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [
    None, True, False, "", "abc", "1e3", "1.2.3", "12,34,56", "$", "--5",
    float("nan"), float("inf"), [], {}, "1,2,3",
])
def test_to_decimal_rejects(raw):
    assert to_decimal(raw) is None


def test_float_goes_through_str_not_binary_expansion():
    assert to_decimal(0.1) == Decimal("0.1")
    assert str(to_decimal(0.1)) == "0.1"


def test_decimal_math_is_exact():
    assert to_decimal("0.1") + to_decimal("0.2") == to_decimal("0.3")


def test_numbers_in_text_finds_all_formats():
    found = numbers_in_text("Total 1,323.00 and 1.234,56 EUR, tax 0,00, qty 4, 8%")
    assert {Decimal("1323"), Decimal("1234.56"), Decimal("0"), Decimal("4"), Decimal("8")} <= found


def test_numbers_in_text_ignores_trailing_punctuation():
    assert Decimal("1225") in numbers_in_text("Subtotal: $1,225.00.")


# ── dates ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw, expected", [
    ("2024-03-15", "2024-03-15"), ("15 March 2024", "2024-03-15"),
    ("March 15, 2024", "2024-03-15"), ("15.03.2024", "2024-03-15"),
    ("02 Apr 2024", "2024-04-02"), ("  2024-03-15 ", "2024-03-15"),
])
def test_to_iso_date_parses(raw, expected):
    assert to_iso_date(raw) == expected


@pytest.mark.parametrize("raw", [
    "03/04/2024",          # ambiguous: refuse to guess
    "2024-02-30", "2024-13-01", "tomorrow", "", None, 20240315, "15/13/2024",
])
def test_to_iso_date_rejects(raw):
    assert to_iso_date(raw) is None


def test_ordinals():
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 30)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "30th"]


@pytest.mark.parametrize("written", [
    "2024-03-05", "05/03/2024", "03/05/2024", "05.03.2024", "5 march 2024",
    "march 5, 2024", "march 5th, 2024", "5 mar 2024", "5/3/2024",
])
def test_date_renderings_cover_common_formats(written):
    assert written in date_renderings("2024-03-05")


def test_contains_is_whole_token():
    assert contains("due 15/3/2024 ok", "15/3/2024")
    assert not contains("due 15/3/2024 ok", "5/3/2024")
    assert not contains("ref inv-420", "inv-42")
    assert contains("ref inv-42.", "inv-42")
    assert not contains("anything", "")


def test_normalize_handles_unicode_tricks():
    assert normalize("Müller") == normalize("Mu\u0308ller")          # composed vs decomposed
    assert normalize("ig\u200bnore") == "ignore"                      # zero-width space
    assert normalize("ｉｇｎｏｒｅ") == "ignore"                          # full-width letters
    assert normalize("STRASSE") == normalize("Straße")                # casefold, not lower
    assert normalize("a \n\t b") == "a b"


def test_mostly_grounded():
    text = normalize("Either party may terminate this agreement with 30 days written notice")
    assert mostly_grounded("Either party may terminate with 30 days written notice", text)
    assert not mostly_grounded("Client owns all intellectual property forever", text)
    assert mostly_grounded("12", text)          # no content words: nothing to check


# ── parse_json_object ─────────────────────────────────────────────────────
def test_parse_plain_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('```\n{"a": 1}') == {"a": 1}            # missing closing fence


def test_parse_json_with_prose_around_it():
    assert parse_json_object('Sure! Here you go: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}


@pytest.mark.parametrize("bad", ["", "no json here", "{broken", "[1, 2]", '"text"', "{a: 1}", "}{"])
def test_parse_json_rejects(bad):
    with pytest.raises(ExtractionError):
        parse_json_object(bad)


# ── loader ────────────────────────────────────────────────────────────────
def test_read_text_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    assert read_document(p) == "hello world"


def test_read_text_with_bad_bytes_does_not_crash(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"caf\xe9 total 5")
    assert "total 5" in read_document(p)


def test_empty_and_whitespace_files_rejected(tmp_path):
    for content in ("", "   \n\t "):
        p = tmp_path / "e.txt"
        p.write_text(content)
        with pytest.raises(DocumentError, match="no extractable text"):
            read_document(p)


def test_unsupported_and_missing_files(tmp_path):
    (tmp_path / "x.exe").write_text("data")
    with pytest.raises(DocumentError, match="Unsupported"):
        read_document(tmp_path / "x.exe")
    with pytest.raises(DocumentError, match="not found"):
        read_document(tmp_path / "ghost.txt")
    with pytest.raises(DocumentError, match="not found"):
        read_document(tmp_path)                                         # a folder, not a file


def test_oversize_file_rejected_before_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MAX_FILE_BYTES", 10)
    p = tmp_path / "big.txt"
    p.write_text("x" * 50)
    with pytest.raises(DocumentError, match="larger than"):
        read_document(p)


def test_pdf_text_extracted(tmp_path):
    p = tmp_path / "inv.pdf"
    make_pdf(p, ["Invoice Number: INV-77", "Total Due: 1,500.00"])
    text = read_document(p)
    assert "INV-77" in text and "1,500.00" in text


def test_pdf_without_text_rejected(tmp_path):
    p = tmp_path / "scan.pdf"
    make_pdf(p, [""])
    with pytest.raises(DocumentError, match="no extractable text"):
        read_document(p)


def test_corrupt_pdf_rejected_cleanly(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"this is definitely not a pdf")
    with pytest.raises(DocumentError, match="could not be read"):
        read_document(p)


def test_encrypted_pdf_rejected(tmp_path):
    from pypdf import PdfReader, PdfWriter
    src = tmp_path / "plain.pdf"
    make_pdf(src, ["secret text"])
    writer = PdfWriter(clone_from=PdfReader(str(src)))
    try:
        writer.encrypt("pw", algorithm="RC4-128")
    except Exception as exc:                       # encryption backend not installed
        pytest.skip(f"cannot create encrypted PDF here: {exc}")
    enc = tmp_path / "enc.pdf"
    with open(enc, "wb") as f:
        writer.write(f)
    with pytest.raises(DocumentError, match="password"):
        read_document(enc)


def test_list_documents_filters_and_sorts(tmp_path):
    for name in ("b.txt", "a.md", "c.pdf", "d.exe", "e.docx"):
        (tmp_path / name).write_text("x")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in list_documents(tmp_path)] == ["a.md", "b.txt", "c.pdf"]


def test_list_documents_missing_folder(tmp_path):
    with pytest.raises(DocumentError, match="Folder not found"):
        list_documents(tmp_path / "nope")


def test_sample_inbox_is_present():
    names = [p.name for p in list_documents(INBOX)]
    assert len(names) == 8 and "invoice_injection.txt" in names
