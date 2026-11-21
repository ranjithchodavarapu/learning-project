"""Tests for the real-invoice kit: template, truth checking, and evaluate --no-registry."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tests.helpers import INBOX, INBOX_V2
from truth_kit import (CONTRACT_FIELDS, INVOICE_FIELDS, build_template, check_truth,
                       document_texts, _field_problem)

PROJECT = Path(__file__).resolve().parent.parent
SHIPPED = json.loads((PROJECT / "eval" / "ground_truth.json").read_text(encoding="utf-8"))
CLEAN = (INBOX / "invoice_clean.txt").read_text()


def texts(**docs):
    return docs


def good(**fields):
    base = {"status": "accepted", "doc_type": "invoice", "fields": fields}
    return base


def check(truth, docs):
    return check_truth(truth, docs)


def has(messages, fragment):
    return any(fragment in m for m in messages)


# ── template ──────────────────────────────────────────────────────────────
def test_template_lists_every_document_and_bundle_parts():
    docs = document_texts([INBOX / "invoice_clean.txt", INBOX_V2 / "bundle_three_invoices.pdf"])
    template = build_template(sorted(docs))
    assert sorted(k for k in template if not k.startswith("_")) == [
        "bundle_three_invoices.pdf#1", "bundle_three_invoices.pdf#2", "bundle_three_invoices.pdf#3", "invoice_clean.txt"]


def test_template_prefills_nothing_from_the_documents_or_the_model():
    entry = build_template(["x.txt"])["x.txt"]
    assert entry["status"].startswith("TODO") and entry["doc_type"].startswith("TODO")
    assert set(entry["fields"]) == set(INVOICE_FIELDS) and set(entry["fields"].values()) == {"TODO"}
    assert "CLEAN" not in json.dumps(entry) and "1323" not in json.dumps(entry)


def test_an_unfilled_template_fails_the_check():
    errors, _ = check(build_template(["invoice_clean.txt"]), texts(**{"invoice_clean.txt": CLEAN}))
    assert has(errors, "still contains TODO")


# ── structure checks ──────────────────────────────────────────────────────
def test_a_correct_entry_passes_cleanly():
    truth = {"d": good(invoice_number="INV-2024-0042", vendor="Northwind Supplies Ltd.", total="1323.00", invoice_date="2024-03-15")}
    errors, warnings = check(truth, texts(d=CLEAN))
    assert errors == [] and not has(warnings, "was not found")


@pytest.mark.parametrize("entry, fragment", [
    ({"status": "ok", "doc_type": "invoice"}, "status must be"),
    ({"status": "accepted", "doc_type": "receipt"}, "doc_type must be"),
    ({"status": "accepted"}, "doc_type must be"),
    ({"status": "accepted", "doc_type": "invoice", "fields": {"governing_law": "X"}}, "not a field of a invoice"),
    ({"status": "accepted", "doc_type": "contract", "fields": {"total": "5"}}, "not a field of a contract"),
    ({"status": "accepted", "doc_type": "unknown", "fields": {"total": "5"}}, "unknown documents have no fields"),
    ({"status": "accepted", "doc_type": "invoice", "fields": "total=5"}, "fields must be an object"),
    ("just text", "must be an object"),
])
def test_structural_mistakes_are_errors(entry, fragment):
    errors, _ = check({"d": entry}, texts(d=CLEAN))
    assert has(errors, fragment)


@pytest.mark.parametrize("field, value", [
    ("total", "1,323.00"), ("total", "$1323"), ("total", "twelve"), ("total", True), ("tax", "5%"), ("subtotal", "1 225"),
    ("invoice_date", "15/03/2024"), ("invoice_date", "2024-3-15"), ("invoice_date", "2024-02-30"), ("due_date", 20240415),
    ("line_items", "3"), ("line_items", True), ("line_items", -1), ("line_items", 2.5),
    ("vendor", ""), ("vendor", "  "), ("vendor", 5), ("vendor", ["A"]),
])
def test_badly_formatted_values_are_errors(field, value):
    assert _field_problem(field, value) is not None


@pytest.mark.parametrize("field, value", [
    ("total", "1323.00"), ("total", 1323), ("total", "-150.5"), ("total", None), ("invoice_date", "2024-03-15"),
    ("invoice_date", None), ("line_items", 3), ("line_items", 0), ("vendor", "Acme"), ("po_number", None),
    ("parties", ["A Ltd", "B Inc"]), ("auto_renewal", True), ("auto_renewal", False), ("auto_renewal", None),
])
def test_well_formed_values_pass(field, value):
    assert _field_problem(field, value) is None


@pytest.mark.parametrize("value", ["yes", 1, "no", [], ["A", ""], "A Ltd"])
def test_contract_only_fields_are_validated_too(value):
    assert _field_problem("auto_renewal", value) is not None or _field_problem("parties", value) is not None


def test_money_must_be_plain_so_evaluate_does_not_crash(tmp_path):
    from evaluate import field_matches
    assert field_matches("total", "1323.00", __import__("decimal").Decimal("1323"))      # what the kit allows works
    with pytest.raises(Exception):
        field_matches("total", "1,323.00", __import__("decimal").Decimal("1323"))      # what the kit forbids would crash


# ── coverage checks ───────────────────────────────────────────────────────
def test_a_name_with_no_matching_file_is_an_error():
    errors, _ = check({"ghost.pdf": good()}, texts(real=CLEAN))
    assert has(errors, "ghost.pdf: no such document")


def test_a_file_missing_from_the_truth_is_only_a_warning():
    errors, warnings = check({}, texts(extra=CLEAN))
    assert errors == [] and has(warnings, "extra: not in the truth file")


def test_unreadable_documents_are_reported():
    _, warnings = check({}, {"scan.png": None})
    assert has(warnings, "scan.png: could not be read")


# ── typo guard ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("field, value", [
    ("invoice_number", "INV-2024-0024"), ("vendor", "Northwind Supply Ltd."), ("total", "1332.00"),
    ("invoice_date", "2024-03-16"), ("currency", "EUR"), ("bank_account", "DE89 3704 0044 0532 0130 00"),
])
def test_values_not_in_the_document_are_flagged_as_probable_typos(field, value):
    errors, warnings = check({"d": good(**{field: value})}, texts(d=CLEAN))
    assert errors == [] and has(warnings, f"{field} = {value!r} was not found")


def test_null_and_uncheckable_fields_do_not_trigger_the_typo_guard():
    truth = {"d": good(po_number=None, line_items=3, bank_account=None)}
    assert not has(check(truth, texts(d=CLEAN))[1], "was not found")


def test_contract_clauses_and_parties_are_checked_against_the_text():
    text = (INBOX / "contract_services.txt").read_text()
    ok = {"status": "accepted", "doc_type": "contract", "fields": {
        "parties": ["Fabrikam Ltd", "Northwind Supplies Ltd"], "governing_law": "the laws of the State of Delaware",
        "effective_date": "2025-01-01", "auto_renewal": True}}
    assert check({"c": ok}, texts(c=text))[0] == [] and not has(check({"c": ok}, texts(c=text))[1], "was not found")
    wrong = json.loads(json.dumps(ok))
    wrong["fields"]["parties"] = ["Fabrikam Ltd", "Phantom Inc"]
    assert has(check({"c": wrong}, texts(c=text))[1], "parties")


# ── balance guidance ──────────────────────────────────────────────────────
def test_too_few_bad_documents_are_called_out():
    truth = {f"d{i}": good() for i in range(25)}
    truth.update({f"b{i}": {"status": "needs_review", "doc_type": "invoice"} for i in range(3)})
    _, warnings = check(truth, {})
    assert has(warnings, "only 3 documents are expected to need review") and not has(warnings, "expected to be accepted")


def test_enough_of_both_kinds_gives_no_balance_warning():
    truth = {f"d{i}": good() for i in range(20)}
    truth.update({f"b{i}": {"status": "needs_review", "doc_type": "invoice"} for i in range(10)})
    assert not has(check(truth, {})[1], "only")


# ── the shipped truth file passes its own check ───────────────────────────
def test_the_shipped_truth_file_has_no_errors_against_the_samples():
    docs = {}
    for folder in (INBOX, INBOX_V2):
        docs.update(document_texts(sorted(p for p in folder.iterdir() if p.is_file())))
    errors, _ = check(SHIPPED, docs)
    assert errors == []


# ── command line ──────────────────────────────────────────────────────────
def run(*args, **env):
    return subprocess.run([sys.executable, *args], cwd=PROJECT, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=120)


def test_cli_template_then_check_flow(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text(CLEAN)
    out = tmp_path / "truth.json"
    made = run("truth_kit.py", "template", "--inbox", str(folder), "--out", str(out))
    assert made.returncode == 0 and out.exists() and "Fill in every TODO" in made.stdout
    assert run("truth_kit.py", "check", "--truth", str(out), "--inbox", str(folder)).returncode == 1       # TODOs remain

    truth = json.loads(out.read_text())
    truth["a.txt"] = good(invoice_number="INV-2024-0042", total="1323.00")
    out.write_text(json.dumps(truth))
    done = run("truth_kit.py", "check", "--truth", str(out), "--inbox", str(folder))
    assert done.returncode == 0 and "No errors" in done.stdout and "--no-registry" in done.stdout


def test_cli_template_refuses_to_overwrite_your_answers(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text(CLEAN)
    out = tmp_path / "truth.json"
    out.write_text("MY HARD WORK")
    proc = run("truth_kit.py", "template", "--inbox", str(folder), "--out", str(out))
    assert proc.returncode == 2 and "already exists" in proc.stdout and out.read_text() == "MY HARD WORK"
    assert run("truth_kit.py", "template", "--inbox", str(folder), "--out", str(out), "--force").returncode == 0


def test_cli_reports_a_missing_folder_and_unreadable_truth(tmp_path):
    assert run("truth_kit.py", "check", "--truth", "x.json", "--inbox", str(tmp_path / "nope")).returncode == 2
    folder = tmp_path / "d"
    folder.mkdir()
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    proc = run("truth_kit.py", "check", "--truth", str(bad), "--inbox", str(folder))
    assert proc.returncode == 2 and "cannot read" in proc.stdout


def test_evaluate_no_registry_switches_the_vendor_and_bank_checks_off(tmp_path):
    base = ["evaluate.py", "--fake", "faithful", "--inbox", str(INBOX), "--inbox", str(INBOX_V2), "--out", str(tmp_path)]
    with_registry = run(*base)
    without = run(*base, "--no-registry")
    assert with_registry.returncode == 0
    assert without.returncode == 1                       # acme_bank_changed etc. are accepted with no registry
    report = json.loads((tmp_path / "eval_report.json").read_text())
    assert report["counts"]["missed_bad"] >= 3


def test_the_real_invoice_folder_is_present_and_ignored_by_git():
    assert (PROJECT / "my_invoices" / "README.md").exists()
    ignore = (PROJECT / ".gitignore").read_text()
    assert "my_invoices/*" in ignore and "!my_invoices/README.md" in ignore and "eval/my_truth*.json" in ignore


# ── found by the first real use: an empty folder and an unfilled template ──
from truth_kit import structure_errors
from loader import list_documents


def test_readme_files_are_not_documents(tmp_path):
    for name in ("README.md", "readme.txt", "ReadMe.pdf", "invoice.txt", "notes.md"):
        (tmp_path / name).write_text("x")
    assert [p.name for p in list_documents(tmp_path)] == ["invoice.txt", "notes.md"]


def test_the_shipped_folders_list_only_real_documents():
    assert "README.md" not in [p.name for p in list_documents(PROJECT / "my_invoices")]
    assert list_documents(PROJECT / "my_invoices") == []


def test_structure_errors_needs_no_documents_and_finds_todos():
    assert has(structure_errors(build_template(["a.txt", "b.txt"])), "a.txt: still contains TODO")
    assert len(structure_errors(build_template(["a.txt", "b.txt"]))) == 2


def test_structure_errors_accepts_the_shipped_truth_and_ignores_missing_files():
    assert structure_errors(SHIPPED) == []                       # no documents were read, none are "missing"
    assert structure_errors({"never_existed.pdf": good(total="5.00")}) == []


def test_structure_errors_still_catches_bad_formats():
    bad = {"d": good(total="1,323.00", invoice_date="15/03/2024")}
    errors = structure_errors(bad)
    assert has(errors, "total") and has(errors, "invoice_date")


def test_template_on_an_empty_folder_says_so_instead_of_listing_the_readme(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "README.md").write_text("instructions")
    out = tmp_path / "t.json"
    proc = run("truth_kit.py", "template", "--inbox", str(folder), "--out", str(out))
    assert proc.returncode == 2 and "no documents found" in proc.stdout and not out.exists()


def test_evaluate_refuses_an_unfilled_template_before_running_anything(tmp_path):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(build_template(["invoice_clean.txt"])))
    proc = run("evaluate.py", "--fake", "faithful", "--truth", str(template), "--inbox", str(INBOX), "--out", str(tmp_path / "o"))
    assert proc.returncode == 2 and "nothing was evaluated" in proc.stdout and "still contains TODO" in proc.stdout
    assert not (tmp_path / "o" / "eval_report.json").exists()


def test_evaluate_refuses_before_checking_ollama(tmp_path):
    template = tmp_path / "t.json"
    template.write_text(json.dumps(build_template(["a.txt"])))
    proc = run("evaluate.py", "--truth", str(template), "--out", str(tmp_path), OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert proc.returncode == 2 and "nothing was evaluated" in proc.stdout and "Cannot reach Ollama" not in proc.stdout


def test_evaluate_reports_a_missing_or_unreadable_truth_file(tmp_path):
    assert "Cannot read the truth file" in run("evaluate.py", "--fake", "faithful", "--truth", str(tmp_path / "nope.json")).stdout
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    proc = run("evaluate.py", "--fake", "faithful", "--truth", str(bad))
    assert proc.returncode == 2 and "Cannot read the truth file" in proc.stdout


def test_evaluate_still_accepts_a_correct_truth_file(tmp_path):
    proc = run("evaluate.py", "--fake", "faithful", "--out", str(tmp_path))
    assert proc.returncode == 0 and "nothing was evaluated" not in proc.stdout
