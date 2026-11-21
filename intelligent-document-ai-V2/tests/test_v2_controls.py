"""v2 tests: registry, invoice history store, controls, IBAN helpers, OCR-tolerant grounding, splitter."""

import json
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config import cfg
from controls import duplicate_issue, invoice_keys, run_controls
from models import InvoiceData
from registry import PurchaseOrder, Registry, RegistryError, VendorRule, load_registry
from splitter import split_pages
from store import InvoiceStore, PriorInvoice, content_hash
from textutil import (find_ibans, iban_valid, looks_like_iban, normalize_account,
                      po_key, vendor_key)
from validator import _Haystack, validate_invoice

ACME = "Acme Industrial Supply"
ACME_IBAN = "FR1420041010050500013M02606"
OTHER_IBAN = "GB82WEST12345698765432"


def registry(**kw) -> Registry:
    return Registry(
        vendors={vendor_key(ACME): VendorRule(ACME, ACME_IBAN, kw.get("po_required", False))},
        purchase_orders={po_key("PO-1"): PurchaseOrder("PO-1", vendor_key(ACME), Decimal("1000.00"), "USD")},
    )


def inv(**kw) -> InvoiceData:
    base = dict(invoice_number="A-1", vendor=ACME, total=Decimal("100.00"), currency="USD")
    base.update(kw)
    return InvoiceData(**base)


def msgs(issues, field=None):
    return [i.message for i in issues if field is None or i.field == field]


def errors(issues):
    return [i for i in issues if i.severity == "error"]


# ── registry ──────────────────────────────────────────────────────────────
def test_missing_registry_file_means_no_controls(tmp_path):
    r = load_registry(tmp_path / "nope.json")
    assert not r.active and r.vendors == {} and r.purchase_orders == {}


def test_load_registry_parses_vendors_and_pos(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "vendors": [{"name": "Acme Industrial Supply", "bank_account": "FR14 2004 1010 0505 0001 3M02 606",
                     "po_required": True}, {"name": "Plain Vendor"}],
        "purchase_orders": [{"po_number": "po-9", "vendor": "Acme Industrial Supply", "amount": 5000.1, "currency": "usd"}]}))
    r = load_registry(p)
    rule = r.vendors[vendor_key("ACME industrial supply.")]
    assert rule.bank_account == ACME_IBAN and rule.po_required is True
    assert r.vendors[vendor_key("Plain Vendor")].bank_account is None
    po = r.purchase_orders["PO9"]
    assert po.amount == Decimal("5000.1") and po.currency == "USD" and r.active


@pytest.mark.parametrize("content, message", [
    ("{not json", "not valid JSON"), ("[1, 2]", "must be a JSON object"),
    ('{"vendors": [{"bank_account": "x"}]}', "non-empty 'name'"),
    ('{"vendors": ["Acme"]}', "non-empty 'name'"),
    ('{"purchase_orders": [{"po_number": "P1", "vendor": "V", "amount": "lots"}]}', "numeric amount"),
    ('{"purchase_orders": [{"po_number": "P1", "vendor": "V"}]}', "numeric amount"),
    ('{"purchase_orders": [{"po_number": "", "vendor": "V", "amount": 5}]}', "numeric amount"),
    ('{"purchase_orders": [{"po_number": "P1", "vendor": "V", "amount": "NaN"}]}', "numeric amount"),
])
def test_broken_registry_stops_the_run_with_a_clear_message(tmp_path, content, message):
    p = tmp_path / "r.json"
    p.write_text(content)
    with pytest.raises(RegistryError, match=message):
        load_registry(p)


def test_shipped_registry_is_valid():
    r = load_registry(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "registry.json"))
    assert len(r.vendors) == 4 and len(r.purchase_orders) == 2
    assert all(rule.bank_account is None or iban_valid(rule.bank_account) for rule in r.vendors.values())


# ── IBAN and key helpers ──────────────────────────────────────────────────
@pytest.mark.parametrize("iban", ["DE89 3704 0044 0532 0130 00", "GB29NWBK60161331926819",
                                  "FR14 2004 1010 0505 0001 3M02 606", "gb82 west 1234 5698 7654 32"])
def test_valid_ibans(iban):
    assert iban_valid(iban) and looks_like_iban(iban)


@pytest.mark.parametrize("iban", ["DE89 3704 0044 0532 0130 01", "DE00", "12345678", "", "GB29NWBK6016133192681"])
def test_invalid_ibans(iban):
    assert not iban_valid(iban)


def test_one_changed_digit_always_breaks_the_checksum():
    base = "DE89370400440532013000"
    for i in range(2, len(base)):
        for digit in "0123456789":
            if digit != base[i] and base[i].isdigit():
                assert not iban_valid(base[:i] + digit + base[i + 1:])


@pytest.mark.parametrize("text, expected", [
    ("Pay to IBAN: FR14 2004 1010 0505 0001 3M02 606 thanks", {ACME_IBAN}),
    ("IBAN GB82WEST12345698765432.", {OTHER_IBAN}),
    ("DE89-3704-0044-0532-0130-00", {"DE89370400440532013000"}),
    ("iban: gb29 nwbk 6016 1331 9268 19", {"GB29NWBK60161331926819"}),
    ("A: GB29 NWBK 6016 1331 9268 19 B: DE89 3704 0044 0532 0130 00",
     {"GB29NWBK60161331926819", "DE89370400440532013000"}),
    ("DE89 3704 0044 0532 0130 01", set()),             # bad checksum
    ("XDE89370400440532013000X", set()),                # inside a longer token
    ("DE89 3704 0044", set()),                          # too short
    ("ZZ89 3704 0044 0532 0130 00", set()),             # unknown country
    ("Invoice INV-2024-0042 total 1,323.00 ref AB12 CD34 EF56", set()),
    ("", set()),
])
def test_find_ibans(text, expected):
    assert find_ibans(text) == expected


def test_vendor_and_po_keys_ignore_punctuation_and_case():
    assert vendor_key("Northwind Supplies Ltd.") == vendor_key("NORTHWIND  SUPPLIES, LTD")
    assert vendor_key("Müller GmbH") == vendor_key("Mu\u0308ller GmbH")
    assert po_key("po-1001") == po_key("PO 1001") == "PO1001"
    assert normalize_account("de89 3704-0044.0532 0130 00") == "DE89370400440532013000"


# ── store ─────────────────────────────────────────────────────────────────
def store_with(*rows) -> InvoiceStore:
    s = InvoiceStore(":memory:")
    for r in rows:
        s.record(*r)
    return s


def row(vendor="v", number="n1", filename="f.txt", chash="h", status="accepted",
        total="100.00", currency="USD", po=None):
    return (vendor, number, filename, chash, status, Decimal(total) if total else None, currency, po)


def test_store_records_and_finds():
    s = store_with(row())
    assert s.find("v", "n1") == PriorInvoice("f.txt", "h", "accepted")
    assert s.find("v", "n2") is None and s.find("other", "n1") is None


def test_first_record_wins_against_a_different_submission():
    s = store_with(row(), row(filename="copy.txt", chash="h2", status="needs_review"))
    assert s.find("v", "n1") == PriorInvoice("f.txt", "h", "accepted")


def test_same_submission_updates_status_and_total():
    s = store_with(row(status="needs_review", total="100.00"))
    s.record(*row(status="accepted", total="150.00"))
    assert s.find("v", "n1").status == "accepted"
    assert s.accepted_totals("v", "USD", exclude_number="zzz") == [Decimal("150.00")]


def test_accepted_totals_filters():
    s = store_with(row(number="a", total="10.00"), row(number="b", total="20.00", status="needs_review"),
                   row(number="c", total="30.00", currency="EUR"), row(number="d", total="40.00"),
                   row(vendor="other", number="e", total="99.00"))
    assert sorted(s.accepted_totals("v", "USD", exclude_number="d")) == [Decimal("10.00")]
    assert sorted(s.accepted_totals("v", "USD", exclude_number="zzz")) == [Decimal("10.00"), Decimal("40.00")]
    assert s.accepted_totals("v", "EUR", exclude_number="zzz") == [Decimal("30.00")]


def test_po_invoiced_counts_only_accepted_and_excludes_itself():
    s = store_with(row(number="a", total="300.00", po="PO1"), row(number="b", total="200.00", po="PO1", status="needs_review"),
                   row(number="c", total="50.00", po="PO2"), row(number="d", total="100.00", po="PO1"))
    assert s.po_invoiced("PO1", "v", "zzz") == Decimal("400.00")
    assert s.po_invoiced("PO1", "v", "d") == Decimal("300.00")
    assert s.po_invoiced("PO9", "v", "zzz") == Decimal(0)


def test_store_survives_hostile_vendor_names():
    evil = "Acme'; DROP TABLE invoices;--"
    s = store_with(row(vendor=evil, number="x'--"))
    assert s.find(evil, "x'--") is not None
    assert s.find("v", "n1") is None
    s.record(*row(number="after"))                    # the table still exists
    assert s.find("v", "after") is not None


def test_store_persists_between_instances(tmp_path):
    path = tmp_path / "sub" / "h.db"                 # the folder does not exist yet
    InvoiceStore(path).record(*row())
    assert InvoiceStore(path).find("v", "n1") == PriorInvoice("f.txt", "h", "accepted")


def test_content_hash_is_stable_and_sensitive():
    assert content_hash("abc") == content_hash("abc") != content_hash("abd")


# ── duplicate semantics ───────────────────────────────────────────────────
def test_duplicate_identity_rules():
    d = inv()
    prior = PriorInvoice("a.txt", "H1", "accepted")
    assert duplicate_issue(d, None, "a.txt", "H1") is None                       # first time
    assert duplicate_issue(d, prior, "a.txt", "H1") is None                      # the same file, re-run
    assert "already submitted as 'a.txt'" in duplicate_issue(d, prior, "b.txt", "H1").message   # same PDF, new name
    assert duplicate_issue(d, prior, "a.txt", "H2") is not None                  # edited file, same number
    assert duplicate_issue(d, prior, "b.txt", "H2").severity == "error"


def test_invoice_keys_need_vendor_and_number():
    assert invoice_keys(inv()) == (vendor_key(ACME), "a-1")
    assert invoice_keys(inv(vendor=None)) is None and invoice_keys(inv(invoice_number=None)) is None


# ── controls ──────────────────────────────────────────────────────────────
def run(data, reg=None, store=None, text=""):
    return run_controls(data, reg or registry(), store or InvoiceStore(":memory:"), text)


def test_clean_invoice_passes_every_control():
    assert run(inv(po_number="PO-1", bank_account="FR14 2004 1010 0505 0001 3M02 606")) == []


def test_approval_limit(monkeypatch):
    assert any("auto-approval limit" in m for m in msgs(run(inv(total=Decimal("10000.00"))), "total"))
    assert not any("limit" in m for m in msgs(run(inv(total=Decimal("9999.99"), po_number="PO-1")), "total"))
    monkeypatch.setattr(cfg, "APPROVAL_THRESHOLD", Decimal("50"))
    assert any("limit" in m for m in msgs(run(inv(total=Decimal("50.00"))), "total"))


def test_unapproved_vendor_only_checked_once_a_list_exists():
    assert any("not on the approved vendor list" in m for m in msgs(run(inv(vendor="Stranger Ltd")), "vendor"))
    assert run(inv(vendor="Stranger Ltd"), reg=Registry()) == []                 # no registry: no vendor check


def test_vendor_name_variants_match_the_registry():
    assert run(inv(vendor="ACME INDUSTRIAL SUPPLY.", po_number="PO-1")) == []


def test_bank_account_mismatch_from_the_model():
    issues = run(inv(po_number="PO-1", bank_account="GB82 WEST 1234 5698 7654 32"))
    assert any("payment-redirection fraud" in m for m in msgs(issues, "bank_account"))
    assert OTHER_IBAN in msgs(issues, "bank_account")[0]


def test_bank_account_mismatch_found_even_if_the_model_forgot_it():
    text = "Please pay to IBAN: GB82 WEST 1234 5698 7654 32 (new account)"
    issues = run(inv(po_number="PO-1", bank_account=None), text=text)
    assert any("payment-redirection fraud" in m for m in msgs(issues, "bank_account"))


def test_matching_or_absent_bank_account_is_fine():
    assert run(inv(po_number="PO-1", bank_account="fr14 2004 1010 0505 0001 3m02 606")) == []
    assert run(inv(po_number="PO-1", bank_account=None), text="no bank details on this invoice") == []
    assert run(inv(po_number="PO-1"), text="Pay to FR14 2004 1010 0505 0001 3M02 606") == []


def test_no_bank_on_file_gives_a_warning_not_an_error():
    reg = Registry(vendors={vendor_key(ACME): VendorRule(ACME, None, False)})
    issues = run(inv(bank_account="DE89 3704 0044 0532 0130 00"), reg=reg)
    assert [i.severity for i in issues if i.field == "bank_account"] == ["warning"]


def test_po_unknown_wrong_vendor_wrong_currency():
    assert any("not in the registry" in m for m in msgs(run(inv(po_number="PO-404")), "po_number"))
    other = Registry(vendors={}, purchase_orders={po_key("PO-1"): PurchaseOrder("PO-1", "someoneelse", Decimal("1000"), "USD")})
    assert any("different vendor" in m for m in msgs(run(inv(po_number="PO-1"), reg=other), "po_number"))
    assert any("currency" in m for m in msgs(run(inv(po_number="PO-1", currency="EUR")), "currency"))


def test_po_amount_is_cumulative_across_invoices():
    store = store_with(row(vendor=vendor_key(ACME), number="earlier", total="700.00", po=po_key("PO-1")))
    assert errors(run(inv(total=Decimal("300.00"), po_number="PO-1"), store=store)) == []
    over = run(inv(total=Decimal("300.02"), po_number="PO-1"), store=store)   # 1000.02 > 1000.00 + 0.01 tolerance
    assert any("exceeds purchase order PO-1" in m and "700.00 already invoiced" in m for m in msgs(over, "total"))


def test_po_not_consumed_by_invoices_awaiting_review():
    store = store_with(row(vendor=vendor_key(ACME), number="pending", total="900.00", po=po_key("PO-1"), status="needs_review"))
    assert errors(run(inv(total=Decimal("900.00"), po_number="PO-1"), store=store)) == []


def test_po_required_vendor_without_po():
    issues = run(inv(), reg=registry(po_required=True))
    assert any("requires a purchase order" in m for m in msgs(issues, "po_number"))
    assert msgs(run(inv(), reg=registry(po_required=False)), "po_number") == []


def test_amount_far_above_vendor_history():
    store = store_with(*[row(vendor=vendor_key(ACME), number=f"h{i}", total=t) for i, t in enumerate(["100.00", "120.00", "90.00"])])
    big = run(inv(total=Decimal("361.00"), po_number="PO-1"), store=store)
    assert any("more than 3x" in m and "120.00" in m for m in msgs(big, "total"))
    assert msgs(run(inv(total=Decimal("360.00"), po_number="PO-1"), store=store), "total") == []


def test_history_check_needs_enough_history_and_matching_currency():
    two = store_with(row(vendor=vendor_key(ACME), number="a", total="10.00"), row(vendor=vendor_key(ACME), number="b", total="10.00"))
    assert msgs(run(inv(total=Decimal("900.00"), po_number="PO-1"), store=two), "total") == []
    eur = store_with(*[row(vendor=vendor_key(ACME), number=f"e{i}", total="10.00", currency="EUR") for i in range(5)])
    assert msgs(run(inv(total=Decimal("900.00"), po_number="PO-1"), store=eur), "total") == []


def test_invoice_without_vendor_or_total_does_not_crash():
    assert isinstance(run(InvoiceData()), list)
    assert isinstance(run(InvoiceData(vendor=ACME, total=None, po_number="PO-1")), list)


# ── validator: new fields ─────────────────────────────────────────────────
TEXT = "Invoice INV-9 from Acme dated 2024-06-01 PO Number: PO-1 Pay to IBAN: FR14 2004 1010 0505 0001 3M02 606 total 5.00"


def base_invoice(**kw):
    data = dict(invoice_number="INV-9", vendor="Acme", invoice_date="2024-06-01", total=Decimal("5"))
    data.update(kw)
    return InvoiceData(**data)


def test_po_number_and_bank_account_must_appear_in_the_document():
    ok = validate_invoice(base_invoice(po_number="PO-1", bank_account="FR14 2004 1010 0505 0001 3M02 606"), TEXT)
    assert errors(ok) == []
    bad = validate_invoice(base_invoice(po_number="PO-2", bank_account="DE89 3704 0044 0532 0130 00"), TEXT)
    assert {"po_number", "bank_account"} <= {i.field for i in errors(bad)}


def test_iban_with_a_wrong_checksum_is_an_error():
    text = TEXT.replace("3M02 606", "3M02 607")
    issues = validate_invoice(base_invoice(bank_account="FR14 2004 1010 0505 0001 3M02 607"), text)
    assert any("fails its checksum" in m for m in msgs(issues, "bank_account"))


def test_non_iban_account_numbers_skip_the_checksum():
    issues = validate_invoice(base_invoice(bank_account="12345678"), TEXT + " account 12345678")
    assert not any("checksum" in m for m in msgs(issues))


# ── OCR-tolerant grounding ────────────────────────────────────────────────
def test_haystack_plain_mode_is_strict_about_spaces():
    h = _Haystack("Date:12June2024\nRef INV-42", ocr=False)
    assert not h.token("12 june 2024") and h.token("inv-42")


def test_haystack_ocr_mode_ignores_dropped_spaces():
    h = _Haystack("Date:12June2024\nVendor:Globex Consulting GmbH\nRef INV-42", ocr=True)
    assert h.token("12 june 2024") and h.substring("Globex  Consulting GmbH")


def test_haystack_ocr_mode_keeps_line_boundaries():
    h = _Haystack("Ref INV-420\nnext line", ocr=True)
    assert not h.token("INV-42")                                      # not inside INV-420
    assert not _Haystack("see INV 420 now", ocr=True).token("INV-42")


def test_haystack_ocr_mode_is_never_weaker_than_plain():
    text = "Invoice Number: INV-2024-0555\nTotal due"
    assert _Haystack(text, ocr=False).token("INV-2024-0555") and _Haystack(text, ocr=True).token("INV-2024-0555")


def test_haystack_empty_and_missing_values():
    h = _Haystack("some text", ocr=True)
    assert not h.token("") and not h.substring("") and not h.substring("absent")


# ── splitter ──────────────────────────────────────────────────────────────
def invoice_page(number, vendor="Alpha"):
    return (f"INVOICE\n\nInvoice Number: {number}\nInvoice Date: 2024-03-15\nFrom: {vendor}\n"
            "Bill To: Contoso Labs Inc.\n\nOffice chairs 4 120.00 480.00\nSubtotal: 480.00\nTotal Due: 480.00\n")


def test_three_invoices_on_three_pages():
    segs = split_pages([invoice_page("INV-1"), invoice_page("INV-2", "Beta"), invoice_page("INV-3", "Gamma")])
    assert [(s.first_page, s.last_page) for s in segs] == [(0, 0), (1, 1), (2, 2)]
    assert "INV-2" in segs[1].text and "INV-1" not in segs[1].text and "Gamma" in segs[2].text


def test_three_invoices_on_one_page():
    segs = split_pages([invoice_page("INV-1") + "\n" + invoice_page("INV-2", "Beta") + "\n" + invoice_page("INV-3", "Gamma")])
    assert len(segs) == 3 and all(s.first_page == s.last_page == 0 for s in segs)
    assert segs[1].text.startswith("INVOICE") and "Beta" in segs[1].text


def test_continuation_pages_do_not_split():
    page2 = "Invoice Number: INV-9 (page 2)\nCarried forward 480.00\nPayment is due within thirty days of the date above."
    segs = split_pages([invoice_page("INV-9"), page2])
    assert len(segs) == 1 and (segs[0].first_page, segs[0].last_page) == (0, 1)


def test_a_second_page_without_a_label_stays_attached():
    assert len(split_pages([invoice_page("INV-8"), "Terms and conditions apply to every order placed with us."])) == 1


@pytest.mark.parametrize("sentence", [
    "Invoice no. in the sentence INV-4 does not count.",
    "Invoice number must be quoted on every payment.",
    "Invoice number is required for payment.",
    "Credit note for invoice no. INV-3 applied above.",
])
def test_sentences_mentioning_invoice_numbers_do_not_split(sentence):
    assert len(split_pages([invoice_page("INV-5") + "\n" + sentence + "\n"])) == 1


def test_numbers_must_contain_a_digit_to_count():
    assert len(split_pages([invoice_page("INV-ABC") + "\n" + invoice_page("INV-DEF")])) == 1


def test_numbering_styles_split():
    assert len(split_pages([invoice_page("7781") + "\n" + invoice_page("R-2024-118")])) == 2
    assert len(split_pages(["Invoice No. 55\nlots of text here to be long enough " * 3, "Tax Invoice #56\nmore text here to be long enough " * 3])) == 2


def test_same_number_returning_later_is_a_separate_document():
    assert len(split_pages([invoice_page("INV-1"), invoice_page("INV-2"), invoice_page("INV-1")])) == 3


def test_tiny_fragments_are_merged_back():
    segs = split_pages([invoice_page("INV-1") + "\nInvoice Number: INV-2\n"])
    assert len(segs) == 1


def test_no_label_means_one_document_and_empty_input_is_safe():
    assert len(split_pages(["Just some text without any label, long enough to matter here."])) == 1
    assert len(split_pages([""])) == 1 and len(split_pages([])) == 1


def test_sample_documents_are_not_split():
    inbox = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inbox")
    for name in os.listdir(inbox):
        assert len(split_pages(open(os.path.join(inbox, name), encoding="utf-8").read().split("\f"))) == 1, name


# ── IBAN length and boundaries (found by mutation testing) ────────────────
def make_iban(country: str, body: str) -> str:
    """Build an IBAN with correct mod-97 check digits for any body (even a wrong-length one)."""
    digits = "".join(str(int(ch, 36)) for ch in body + country + "00")
    return f"{country}{98 - int(digits) % 97:02d}{body}"


def test_an_iban_with_a_valid_checksum_but_the_wrong_length_is_not_accepted_from_text():
    too_short = make_iban("GB", "NWBK12345678")                 # GB IBANs are 22 characters, this is 16
    assert iban_valid(too_short)                                # the checksum alone would pass it
    # The text must END right after the IBAN: otherwise the scanner keeps reading the next
    # words, the checksum fails for that reason, and the LENGTH rule is never what rejects it.
    assert find_ibans(too_short) == set()
    assert find_ibans(f"IBAN: {too_short}") == set()
    assert find_ibans(f"IBAN: {too_short}\n") == set()
    assert find_ibans(f"Please pay {too_short} today") == set()


def test_a_full_length_constructed_iban_is_found():
    full = make_iban("GB", "NWBK" + "12345678901234")           # 4 + 18 = 22 characters
    assert iban_valid(full) and find_ibans(f"Pay to {full}.") == {full}


@pytest.mark.parametrize("text", ["DE89370400440532013000X", "DE89370400440532013000123", "DE89 3704 0044 0532 0130 00X1"])
def test_an_iban_must_end_cleanly(text):
    assert find_ibans(text) == set()


def test_an_iban_followed_by_punctuation_or_words_is_found():
    for text in ("IBAN DE89370400440532013000.", "(DE89 3704 0044 0532 0130 00)", "DE89370400440532013000\nnext line", "x DE89370400440532013000, thanks"):
        assert find_ibans(text) == {"DE89370400440532013000"}, text
