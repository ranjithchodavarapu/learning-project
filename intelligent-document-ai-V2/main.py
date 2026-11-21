"""
main.py — Entry point.

HOW TO RUN:
  python main.py                       # process everything in inbox/
  python main.py --file inbox/invoice_clean.txt
  python main.py --inbox some/folder --out some/output
  python main.py --skip-check          # do not test that Ollama is reachable
  python main.py --no-history          # do not read or write the invoice history database
  python main.py --db other.db --registry other.json
  python main.py --ocr none            # never run OCR (scanned files will fail clearly)

LINE BY LINE (non-obvious lines):
  check_ollama() BEFORE processing
      If Ollama is not running, every document would fail with a vague
      "model returned no output". One clear message up front is kinder.
      Exit code 2 means "could not start" so scripts can detect it.

  process_and_report(pipeline, files, out_dir)
      The function the CLI calls AND the tests call. Testing the same function
      the real command uses means the tests cover the real path.

  One shared DocumentPipeline for the whole batch
      It holds the invoice store (history database) and the registry (approved
      vendors, purchase orders), so every document is checked against the same
      records and against the documents processed before it.

  --no-history
      Uses an in-memory database: nothing is remembered after the run, and
      earlier runs are ignored. Useful for experiments and demos.

  STATUS_STYLE
      Colours: green accepted, yellow needs_review, red failed.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

from config import cfg
from llm import check_ollama
from loader import list_documents
from models import DocResult, DocumentError
from pipeline import DocumentPipeline
from registry import RegistryError, load_registry
from reporter import write_outputs
from store import InvoiceStore

console = Console()
STATUS_STYLE = {"accepted": "green", "needs_review": "yellow", "failed": "red"}


def build_pipeline(db_path, registry_path, llm=None) -> DocumentPipeline:
    registry = load_registry(registry_path)
    store = InvoiceStore(db_path)
    return DocumentPipeline(llm=llm, store=store, registry=registry) if llm else \
        DocumentPipeline(store=store, registry=registry)


def process_and_report(pipeline: DocumentPipeline, files: List[Path], out_dir) -> List[DocResult]:
    results = [r for f in files for r in pipeline.process_file(f)]   # one file may hold several documents
    write_outputs(results, out_dir)
    return results


def print_table(results: List[DocResult]) -> None:
    table = Table(title="Document processing results")
    for column in ("File", "Type", "Status", "Errors", "Warn", "Summary"):
        table.add_column(column)
    for r in results:
        color = STATUS_STYLE.get(r.status, "white")
        table.add_row(r.filename, r.doc_type, f"[{color}]{r.status}[/{color}]",
                      str(len(r.errors)), str(len(r.warnings)), r.summary[:70])
    console.print(table)

    queue = [r for r in results if r.status != "accepted"]
    if queue:
        console.print("\n[bold yellow]Review queue[/bold yellow]")
        for r in queue:
            console.print(f"[bold]{r.filename}[/bold] ({r.status})")
            for issue in r.errors:
                console.print(f"  [red]error[/red] {issue.field}: {issue.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent Document AI")
    parser.add_argument("--file", help="Process a single file")
    parser.add_argument("--inbox", default=cfg.INBOX_DIR, help="Folder of documents")
    parser.add_argument("--out", default=cfg.OUTPUT_DIR, help="Folder for results")
    parser.add_argument("--skip-check", action="store_true", help="Do not test Ollama first")
    parser.add_argument("--db", default=cfg.DB_PATH, help="Invoice history database")
    parser.add_argument("--registry", default=cfg.REGISTRY_PATH, help="Approved vendors and purchase orders (JSON)")
    parser.add_argument("--no-history", action="store_true", help="Do not persist invoice history")
    parser.add_argument("--ocr", choices=["auto", "rapidocr", "tesseract", "none"], help="OCR engine")
    args = parser.parse_args()
    if args.ocr:
        cfg.OCR_ENGINE = args.ocr

    if not args.skip_check and not check_ollama():
        console.print(f"[red]Cannot reach Ollama at {cfg.OLLAMA_BASE_URL}.[/red] "
                      "Start it with:  ollama serve &")
        sys.exit(2)

    try:
        files = [Path(args.file)] if args.file else list_documents(args.inbox)
    except DocumentError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    if not files:
        console.print(f"No documents found in {args.inbox}")
        return

    try:
        pipeline = build_pipeline(":memory:" if args.no_history else args.db, args.registry)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    console.print(f"Processing {len(files)} file(s) with {cfg.OLLAMA_MODEL} ...")
    results = process_and_report(pipeline, files, args.out)
    print_table(results)
    console.print(f"\nResults written to {args.out}  (report.md, invoices.csv, results/)")


if __name__ == "__main__":
    main()
