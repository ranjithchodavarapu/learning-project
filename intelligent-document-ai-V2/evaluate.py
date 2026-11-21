"""
evaluate.py — Measures how well the pipeline (and the model behind it) does on
documents whose correct answers are known.

WHY THIS EXISTS:
  "It worked on my three examples" is not evidence. A document system must be
  judged on how often it is WRONG, and the two kinds of wrong are not equal:

    false review   a good document sent to a human            costs minutes
    missed bad     a bad document ACCEPTED automatically       costs money / trust

  The harness reports both, plus field-by-field extraction accuracy, and exits
  with code 1 if ANY bad document was accepted, so it can gate a deployment.

HOW TO RUN:
  python evaluate.py                    # real model via Ollama, both sample folders
  python evaluate.py --fake faithful    # self-test: a model that reads perfectly
  python evaluate.py --fake noisy       # self-test: a model that makes mistakes
  python evaluate.py --inbox my_invoices --truth eval/my_truth.json --no-registry   # your own invoices

TRUTH FILE (eval/ground_truth.json), one entry per result name:
  {"invoice_clean.txt": {"status": "accepted", "doc_type": "invoice",
                         "fields": {"invoice_number": "INV-1", "total": "1323.00"}}}
  "ocr_dependent": true  -> shown, but not scored (depends on the OCR engine version)
  A truth file with mistakes (leftover TODO, "1,323.00", a bad status word...) is REFUSED
  with exit code 2 before any model call: scoring it would give meaningless numbers.
  A bundle's parts are named "file.pdf#1", "file.pdf#2", ...

LINE BY LINE (non-obvious lines):
  Each run uses a FRESH in-memory invoice store.
      Otherwise yesterday's run would make today's invoices "duplicates", and the
      numbers would depend on history. Folders are processed in the order given,
      because duplicate and purchase-order checks depend on what came before.

  Decimal(str(expected)) == actual
      Money fields compare as exact Decimals: "1323.00" == Decimal("1323.0").

  FREE_TEXT fields (governing_law, term, payment_terms, termination)
      A clause can be copied as "State of Delaware" or "the laws of the State of
      Delaware"; both are right. These fields count as correct when one text
      CONTAINS the other AND the longer is at most 4x the shorter, so a whole
      pasted paragraph does not pass for a two-word answer. Identifiers, dates,
      amounts, names and bank accounts stay EXACT, because there a near-match is
      a wrong answer. (Added after the first real-model run showed the strict
      comparison marking correct clause wording as wrong.)

  null in the truth file means "the document does not say".
      A model that answers null for something the document never states is
      right; assuming "no renewal" because renewal is not mentioned is a guess.

  sorted(normalize(x) ...) for parties
      Order and capitalisation of a party list do not matter.

  missed_bad / expected_bad
      Of the documents that should NOT be auto-accepted, how many were? The
      number that matters most. Target: 0.

  A truth document that produced no result counts as "missing" (for example if
  bundle splitting found 2 invoices instead of 3). It is a failed expectation, not
  a silent skip.
"""

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from config import cfg
from llm import call_llm, check_ollama
from models import DocResult
from pipeline import DocumentPipeline
from registry import Registry, load_registry
from store import InvoiceStore
from textutil import normalize, normalize_account
from truth_kit import structure_errors

console = Console()
BASE = Path(__file__).resolve().parent
MONEY = {"subtotal", "tax", "total"}
FREE_TEXT = {"governing_law", "term", "payment_terms", "termination"}   # clauses: wording varies
MAX_LENGTH_RATIO = 4


def load_truth(path) -> Dict[str, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def field_matches(name: str, expected, actual) -> bool:
    if name == "line_items":
        return len(actual or []) == expected
    if name in MONEY:
        if expected is None or actual is None:
            return expected is None and actual is None
        return Decimal(str(expected)) == actual
    if name == "parties":
        return sorted(normalize(x) for x in expected) == sorted(normalize(x) for x in (actual or []))
    if name == "bank_account":
        if expected is None or actual is None:
            return expected is None and actual is None
        return normalize_account(expected) == normalize_account(actual)
    if name in FREE_TEXT:
        if expected is None or actual is None:
            return expected is None and actual is None
        want, got = normalize(str(expected)), normalize(str(actual))
        shorter, longer = sorted((want, got), key=len)
        # The length bound also rejects "" against real text (n <= 4 * 0 is false).
        return shorter in longer and len(longer) <= MAX_LENGTH_RATIO * len(shorter)
    if expected is None or isinstance(expected, bool):
        return expected == actual
    if actual is None:
        return False
    if name.endswith("_date"):
        return expected == actual
    return normalize(str(expected)) == normalize(str(actual))


def evaluate_results(results: List[DocResult], truth: Dict[str, dict]) -> dict:
    by_name = {r.filename: r for r in results}
    rows, per_field = [], {}
    counts = dict(graded=0, status_ok=0, type_ok=0, expected_good=0, false_review=0,
                  expected_bad=0, missed_bad=0, ungraded=0, missing=0)
    field_total = field_correct = 0

    for name, expected in truth.items():
        result = by_name.get(name)
        actual_status = result.status if result else "missing"
        row = {"name": name, "expected": expected["status"], "actual": actual_status,
               "scored": not expected.get("ocr_dependent"), "mismatches": []}

        if result is None:
            counts["missing"] += 1
        elif result.data is not None and "fields" in expected:
            for fname, want in expected["fields"].items():
                got = getattr(result.data, fname, None)
                ok = field_matches(fname, want, got)
                per_field.setdefault(fname, [0, 0])
                if row["scored"]:
                    per_field[fname][1] += 1
                    per_field[fname][0] += ok
                    field_total += 1
                    field_correct += ok
                if not ok:
                    row["mismatches"].append((fname, want, got if not isinstance(got, list) else len(got)))

        if not row["scored"]:
            counts["ungraded"] += 1
            rows.append(row)
            continue

        counts["graded"] += 1
        counts["status_ok"] += actual_status == expected["status"]
        counts["type_ok"] += bool(result) and result.doc_type == expected.get("doc_type", result.doc_type)
        if expected["status"] == "accepted":
            counts["expected_good"] += 1
            counts["false_review"] += actual_status != "accepted"
        else:
            counts["expected_bad"] += 1
            counts["missed_bad"] += actual_status == "accepted"
        rows.append(row)

    extra = sorted(set(by_name) - set(truth))
    return {"counts": counts, "rows": rows, "extra_documents": extra,
            "field_accuracy": (field_correct / field_total) if field_total else None,
            "field_total": field_total,
            "per_field": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(per_field.items())}}


def run_evaluation(llm, folders, registry_path, truth) -> dict:
    registry = load_registry(registry_path) if registry_path else Registry()      # None = no vendor/PO/bank checks
    pipeline = DocumentPipeline(llm, store=InvoiceStore(":memory:"), registry=registry)
    results = [r for folder in folders for r in pipeline.process_folder(folder)]
    summary = evaluate_results(results, truth)
    summary["model"] = getattr(llm, "model_name", cfg.OLLAMA_MODEL)
    return summary


def pct(a, b) -> str:
    return "n/a" if not b else f"{100 * a / b:.1f}%"


def render_markdown(s: dict) -> str:
    c = s["counts"]
    lines = [f"# Evaluation report ({s['model']})", "",
             f"- Documents scored: {c['graded']} (not scored, OCR-dependent: {c['ungraded']}, missing: {c['missing']})",
             f"- Correct decision: {pct(c['status_ok'], c['graded'])}",
             f"- **Missed bad (bad document auto-accepted): {c['missed_bad']} of {c['expected_bad']}**",
             f"- False review (good document sent to a human): {c['false_review']} of {c['expected_good']}",
             f"- Document type correct: {pct(c['type_ok'], c['graded'])}",
             f"- Field accuracy: {pct(round((s['field_accuracy'] or 0) * s['field_total']), s['field_total'])}"
             f" ({s['field_total']} fields)", "", "| Field | Correct |", "|---|---|"]
    lines += [f"| {k} | {v['correct']}/{v['total']} |" for k, v in s["per_field"].items()]
    lines += ["", "| Document | Expected | Actual | Result | Field mismatches |", "|---|---|---|---|---|"]
    for r in s["rows"]:
        ok = "not scored" if not r["scored"] else ("ok" if r["expected"] == r["actual"] else "WRONG")
        mism = "; ".join(f"{f}: wanted {w!r}, got {g!r}" for f, w, g in r["mismatches"]) or "-"
        lines.append(f"| {r['name']} | {r['expected']} | {r['actual']} | {ok} | {mism} |")
    return "\n".join(lines) + "\n"


def print_summary(s: dict) -> None:
    c = s["counts"]
    table = Table(title=f"Evaluation: {s['model']}")
    for col in ("Document", "Expected", "Actual", "Result", "Field mismatches"):
        table.add_column(col, overflow="fold")      # wrap long text; never cut it off with "..."
    for r in s["rows"]:
        if not r["scored"]:
            verdict = "[dim]not scored[/dim]"
        elif r["expected"] == r["actual"]:
            verdict = "[green]ok[/green]"
        else:
            verdict = "[red]WRONG[/red]"
        mism = "; ".join(f"{f}: wanted {w!r}, got {g!r}" for f, w, g in r["mismatches"]) or "-"
        table.add_row(r["name"], r["expected"], r["actual"], verdict, mism)
    console.print(table)

    miss = f"[red]{c['missed_bad']}[/red]" if c["missed_bad"] else "[green]0[/green]"
    console.print(f"\nCorrect decision:   {pct(c['status_ok'], c['graded'])}  ({c['status_ok']}/{c['graded']})")
    console.print(f"Missed bad:         {miss} of {c['expected_bad']}   <- the number that matters most")
    console.print(f"False review:       {c['false_review']} of {c['expected_good']} good documents sent to a human")
    console.print(f"Field accuracy:     {pct(round((s['field_accuracy'] or 0) * s['field_total']), s['field_total'])}"
                  f" over {s['field_total']} fields")
    if s["extra_documents"]:
        console.print(f"[dim]Not in the truth file (ignored): {', '.join(s['extra_documents'])}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the document pipeline against known answers")
    parser.add_argument("--truth", default=str(BASE / "eval" / "ground_truth.json"))
    parser.add_argument("--inbox", action="append", help="Folder to process (repeatable; default: both sample folders)")
    parser.add_argument("--registry", default=cfg.REGISTRY_PATH)
    parser.add_argument("--no-registry", action="store_true",
                        help="Skip vendor/PO/bank-account checks (use for your own invoices: the sample registry does not know your vendors)")
    parser.add_argument("--out", default=cfg.OUTPUT_DIR)
    parser.add_argument("--fake", choices=["faithful", "noisy"], help="Self-test with a stand-in model (no Ollama)")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    # Refuse a broken truth file up front: scoring an unfilled template (full of "TODO")
    # produces real-looking numbers that mean nothing, and wastes a full model run.
    try:
        truth = load_truth(args.truth)
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]Cannot read the truth file {args.truth}: {exc}[/red]", markup=True)
        sys.exit(2)
    problems = structure_errors(truth)
    if problems:
        console.print(f"The truth file {args.truth} has {len(problems)} problem(s); nothing was evaluated:", markup=False)
        for problem in problems[:20]:
            console.print(f"  {problem}", markup=False)
        console.print("Fix them (python truth_kit.py check --truth FILE --inbox FOLDER) and run again.", markup=False)
        sys.exit(2)

    if args.fake:
        from tests.helpers import NoisyLLM, RoutedLLM
        llm = RoutedLLM() if args.fake == "faithful" else NoisyLLM()
        llm.model_name = f"stand-in ({args.fake})"
    else:
        if not args.skip_check and not check_ollama():
            console.print(f"[red]Cannot reach Ollama at {cfg.OLLAMA_BASE_URL}.[/red] Start it with: ollama serve &")
            sys.exit(2)
        llm = call_llm

    folders = args.inbox or [str(BASE / "inbox"), str(BASE / "inbox_v2")]
    console.print(f"Evaluating {len(truth)} documents from {', '.join(map(str, folders))} ...")
    summary = run_evaluation(llm, folders, None if args.no_registry else args.registry, truth)

    print_summary(summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "eval_report.md").write_text(render_markdown(summary), encoding="utf-8")
    console.print(f"\nWritten: {out / 'eval_report.md'}, {out / 'eval_report.json'}")

    sys.exit(1 if summary["counts"]["missed_bad"] else 0)


if __name__ == "__main__":
    main()
