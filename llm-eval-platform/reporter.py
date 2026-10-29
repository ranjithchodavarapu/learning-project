"""
reporter.py — Generates markdown and JSON evaluation reports.

WHAT IT DOES:
  Takes an EvalReport and writes two files:
  1. report_MODEL_TIMESTAMP.md   → human-readable markdown
  2. results_MODEL_TIMESTAMP.json → machine-readable JSON for CI/CD

WHY TWO FORMATS:
  Markdown:  engineers read it in GitHub, VS Code, or Obsidian
  JSON:      CI/CD pipelines parse it to auto-fail builds on regression

LINE BY LINE:
  _build_markdown()      → builds the report as a list of strings
                           then "\n".join(lines) at the end
                           this is more readable than one giant f-string

  "## " prefix           → markdown h2 heading
  "| col | col |"        → markdown table syntax
  "|---|---|"            → table header separator row
  "✓" / "✗"             → unicode check/cross for pass/fail

  dataclasses.asdict()   → converts a dataclass to a plain dict
                           works recursively on nested dataclasses
                           needed for json.dumps() which can't serialize dataclasses

  json.dumps(data, indent=2)
                         → pretty-print JSON with 2-space indentation
                         → indent=2 makes it human-readable in editors

  datetime.now().strftime(...)
                         → format: "2024-01-15_14-30-22"
                         → used in filename to avoid overwriting old reports

  Path(cfg.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
                         → creates reports/ folder if it doesn't exist
                         → parents=True: creates parent dirs too
                         → exist_ok=True: no error if already exists
"""

import json
import dataclasses
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from models import EvalReport, EvalResult
from config import cfg

console = Console()


def generate_report(report: EvalReport) -> tuple[str, str]:
    """
    Generate markdown and JSON reports from an EvalReport.

    Args:
        report: Complete EvalReport from the evaluator.

    Returns:
        Tuple of (markdown_path, json_path).
    """
    Path(cfg.REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # Timestamped filenames to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model = report.model.replace(":", "-").replace("/", "-")

    md_path   = Path(cfg.REPORTS_DIR) / f"report_{safe_model}_{timestamp}.md"
    json_path = Path(cfg.REPORTS_DIR) / f"results_{safe_model}_{timestamp}.json"

    # Write markdown
    md_content = _build_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Write JSON (convert dataclasses to dicts first)
    json_data = _report_to_dict(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Display summary in terminal
    _display_summary(report, str(md_path))

    return str(md_path), str(json_path)


def _build_markdown(report: EvalReport) -> str:
    """Build the full markdown report as a string."""
    lines = [
        f"# LLM Evaluation Report",
        f"",
        f"**Model:** `{report.model}`",
        f"**Test suite:** {report.test_suite}",
        f"**Date:** {report.timestamp}",
        f"**Pass rate:** {report.pass_rate}% ({report.passed_count}/{len(report.results)})",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
    ]

    # Summary table
    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tests | {len(report.results)} |",
        f"| Passed | {report.passed_count} |",
        f"| Failed | {len(report.results) - report.passed_count} |",
        f"| Pass rate | {report.pass_rate}% |",
    ]

    # Add summary stats if available
    if report.summary:
        lines.append(f"")
        lines.append(f"## Scorer Results")
        lines.append(f"")
        lines += [
            f"| Scorer | Score | Status |",
            f"|--------|-------|--------|",
        ]
        for scorer, stats in report.summary.items():
            if isinstance(stats, dict):
                score = stats.get("avg_score", stats.get("score", "N/A"))
                passed = stats.get("passed", True)
                status = "✓ Pass" if passed else "✗ Fail"
                if isinstance(score, float):
                    score = f"{score:.2f}"
                lines.append(f"| {scorer} | {score} | {status} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Test Results",
        f"",
        f"| # | Prompt | Accuracy | Latency | Hallucination | Overall |",
        f"|---|--------|----------|---------|---------------|---------|",
    ]

    for i, result in enumerate(report.results, 1):
        prompt_preview = result.test_case.prompt[:40] + "..."
        acc  = _score_cell(result.scores.get("accuracy"))
        lat  = _latency_cell(result.response.latency_ms)
        hall = _score_cell(result.scores.get("hallucination"))
        overall = "✓" if result.overall_passed else "✗"
        lines.append(f"| {i} | {prompt_preview} | {acc} | {lat} | {hall} | {overall} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Failed Tests",
        f"",
    ]

    failed = [r for r in report.results if not r.overall_passed]
    if not failed:
        lines.append("All tests passed!")
    else:
        for result in failed:
            lines += [
                f"### {result.test_case.prompt[:60]}",
                f"",
                f"**Expected:** {result.test_case.expected[:100]}",
                f"",
                f"**Got:** {result.response.response[:200]}",
                f"",
            ]
            for scorer, score_result in result.scores.items():
                if not score_result.passed:
                    lines.append(
                        f"- **{scorer}** failed: score {score_result.score:.2f}"
                    )
            lines.append("")

    return "\n".join(lines)


def _score_cell(score_result) -> str:
    """Format a score for the markdown table."""
    if score_result is None:
        return "N/A"
    icon = "✓" if score_result.passed else "✗"
    return f"{icon} {score_result.score:.2f}"


def _latency_cell(latency_ms: float) -> str:
    """Format latency for the markdown table."""
    if latency_ms > 5000:
        return f"⚠ {latency_ms:.0f}ms"
    return f"{latency_ms:.0f}ms"


def _report_to_dict(report: EvalReport) -> dict:
    """Convert EvalReport to a plain dict for JSON serialization."""
    return {
        "model": report.model,
        "test_suite": report.test_suite,
        "timestamp": report.timestamp,
        "pass_rate": report.pass_rate,
        "passed_count": report.passed_count,
        "total_count": len(report.results),
        "summary": report.summary,
        "results": [
            {
                "prompt": r.test_case.prompt,
                "expected": r.test_case.expected,
                "response": r.response.response,
                "latency_ms": r.response.latency_ms,
                "total_tokens": r.response.total_tokens,
                "overall_passed": r.overall_passed,
                "scores": {
                    name: {
                        "score": s.score,
                        "passed": s.passed,
                        "details": s.details,
                    }
                    for name, s in r.scores.items()
                },
            }
            for r in report.results
        ],
    }


def _display_summary(report: EvalReport, md_path: str) -> None:
    """Print a summary panel to the terminal."""
    color = "green" if report.pass_rate >= 80 else "yellow" if report.pass_rate >= 50 else "red"
    console.print(Panel(
        f"[bold]Model:[/bold] {report.model}\n"
        f"[bold]Tests:[/bold] {len(report.results)}\n"
        f"[bold]Pass rate:[/bold] [{color}]{report.pass_rate}%[/{color}] "
        f"({report.passed_count}/{len(report.results)})\n"
        f"[bold]Report:[/bold] {md_path}",
        title="[bold]Evaluation Complete[/bold]",
        border_style=color,
    ))
