"""
main.py — Entry point for the LLM Evaluation Platform.

HOW TO RUN:
  python main.py                           # eval all configured models on general suite
  python main.py --suite test_suites/general.json
  python main.py --model llama3.1:8b
  python main.py --model llama3.1:70b --model llama3.1:8b  # compare two models
  python main.py --list-suites            # show available test suites

LINE BY LINE:
  --model                → can be specified multiple times
                           action="append" = builds a list
                           --model A --model B → args.model = ["A", "B"]
                           if not given: uses cfg.MODELS_TO_EVAL from .env

  --suite                → path to test suite JSON
                           default: test_suites/general.json

  --list-suites          → lists all .json files in TEST_SUITES_DIR

  for model in models:   → evaluate each model in sequence
                           could be parallelised but sequential is clearer
                           for comparing models: same test, same time

  generate_report(report)→ writes .md and .json files
                           returns (md_path, json_path)

  if len(reports) > 1:   → if multiple models: print comparison table
                           shows which model won on each metric

  _print_comparison()    → side-by-side comparison of all models
    sorted by pass_rate  → best model first
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from evaluator import evaluate
from reporter import generate_report
from config import cfg

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Platform — test accuracy, latency, hallucination, cost"
    )
    parser.add_argument(
        "--model", action="append", dest="models",
        help="Model to evaluate (can specify multiple times)",
    )
    parser.add_argument(
        "--suite", type=str,
        default=f"{cfg.TEST_SUITES_DIR}/general.json",
        help="Path to test suite JSON file",
    )
    parser.add_argument(
        "--list-suites", action="store_true",
        help="List available test suites and exit",
    )
    args = parser.parse_args()

    # ── List suites ───────────────────────────────────────────────────────
    if args.list_suites:
        suite_dir = Path(cfg.TEST_SUITES_DIR)
        suites = list(suite_dir.glob("*.json"))
        console.print(f"\n[bold]Available test suites in {suite_dir}:[/bold]")
        for s in suites:
            console.print(f"  • {s}")
        return

    # ── Determine which models to eval ────────────────────────────────────
    models = args.models if args.models else cfg.MODELS_TO_EVAL

    console.print(Panel(
        f"[bold blue]LLM Evaluation Platform[/bold blue]\n"
        f"Models: {', '.join(models)}\n"
        f"Suite:  {args.suite}",
        border_style="blue",
    ))

    # ── Run evaluation for each model ─────────────────────────────────────
    reports = []
    for model in models:
        report = evaluate(model, args.suite)
        md_path, json_path = generate_report(report)
        reports.append(report)
        console.print(f"[dim]Reports saved: {md_path}[/dim]")

    # ── Compare models if multiple ────────────────────────────────────────
    if len(reports) > 1:
        _print_comparison(reports)


def _print_comparison(reports) -> None:
    """Print a side-by-side comparison table when multiple models are evaluated."""
    console.print("\n")
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Pass Rate", style="bold")
    table.add_column("Avg Accuracy")
    table.add_column("Hallucinations")
    table.add_column("p95 Latency")
    table.add_column("Avg Tokens")

    # Sort by pass rate descending (best first)
    sorted_reports = sorted(reports, key=lambda r: r.pass_rate, reverse=True)

    for report in sorted_reports:
        s = report.summary
        acc   = f"{s.get('accuracy', {}).get('avg_score', 0):.2f}"
        hall  = str(s.get('hallucination', {}).get('failures', '?'))
        lat   = f"{s.get('latency', {}).get('p95', '?')}ms"
        tok   = str(s.get('cost', {}).get('avg_tokens_per_request', '?'))
        color = "green" if report.pass_rate >= 80 else "yellow"
        table.add_row(
            report.model,
            f"[{color}]{report.pass_rate}%[/{color}]",
            acc, hall, lat, tok,
        )

    console.print(table)


if __name__ == "__main__":
    main()
