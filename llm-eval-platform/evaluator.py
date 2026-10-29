"""
evaluator.py — Orchestrates runner + all scorers into one EvalReport.

WHAT IT DOES:
  1. Loads test cases from a JSON file
  2. Runs them against the model (runner.py)
  3. Scores each response (accuracy + hallucination per test)
  4. Scores the whole run (latency + cost across all tests)
  5. Builds an EvalReport with all results
  6. Passes to reporter.py

LINE BY LINE:
  load_test_suite(path)  → reads JSON file → list of TestCase objects
    json.load(f)         → parse JSON file → Python list of dicts
    TestCase(**item)     → unpack dict as keyword args to TestCase constructor
                           {"prompt": "...", "expected": "..."} → TestCase(prompt=..., expected=...)

  evaluate(model, suite_path)
    runner.run_suite(...)→ run all prompts, get list of LLMResponse
    scorers.accuracy.score(resp, tc)
                         → score each response individually
    scorers.latency.score(all_responses)
                         → score latency across the whole run
    scorers.cost.score(all_responses)
                         → score cost across the whole run
    scorers.hallucination.score(resp, tc)
                         → score each response individually

  overall_passed         → True only if accuracy AND hallucination pass
                           latency and cost are informational
                           (a slow but correct answer still passes)

  _build_summary()       → aggregates per-test scores into run-level stats
    accuracy_scores      → list of all accuracy scores
    sum(scores) / len    → mean accuracy across all tests
    hallucination_rate   → fraction of tests that hallucinated
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from models import TestCase, EvalResult, EvalReport
from runner import run_suite
from scorers import accuracy, latency, hallucination, cost
from config import cfg

console = Console()


def load_test_suite(path: str) -> List[TestCase]:
    """
    Load test cases from a JSON file.

    Expected format:
    [
      {"prompt": "...", "expected": "...", "category": "factual"},
      ...
    ]

    Args:
        path: Path to the JSON test suite file.

    Returns:
        List of TestCase objects.
    """
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Test suite not found: {path}")

    with open(suite_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append(TestCase(
            prompt=item["prompt"],
            expected=item["expected"],
            category=item.get("category", "general"),
            tags=item.get("tags", []),
        ))

    console.print(f"[green]✓ Loaded {len(test_cases)} test cases from {suite_path.name}[/green]")
    return test_cases


def evaluate(model: str, suite_path: str) -> EvalReport:
    """
    Run a complete evaluation of one model on one test suite.

    Steps:
    1. Load test cases
    2. Run all prompts (runner.py)
    3. Score each response (accuracy + hallucination)
    4. Score the full run (latency + cost)
    5. Build and return EvalReport

    Args:
        model:      Ollama model name to evaluate.
        suite_path: Path to JSON test suite file.

    Returns:
        EvalReport with all results and scores.
    """
    console.print(f"\n[bold blue]Evaluating: {model}[/bold blue]")
    console.print(f"[dim]Test suite: {suite_path}[/dim]")

    # Step 1: Load test cases
    test_cases = load_test_suite(suite_path)

    # Step 2: Run all prompts
    console.print(f"\n[blue]Running {len(test_cases)} prompts...[/blue]")
    responses = run_suite(model, test_cases)

    # Step 3: Score each response (per-test scorers)
    console.print(f"\n[blue]Scoring responses...[/blue]")
    eval_results = []

    for tc, resp in zip(test_cases, responses):
        # Accuracy — how well does response match expected?
        acc_score  = accuracy.score(resp, tc)

        # Hallucination — does it make up facts?
        hall_score = hallucination.score(resp, tc)

        # Overall pass = accuracy AND hallucination both pass
        # (latency and cost are informational)
        overall = acc_score.passed and hall_score.passed

        eval_results.append(EvalResult(
            test_case=tc,
            response=resp,
            scores={
                "accuracy":      acc_score,
                "hallucination": hall_score,
            },
            overall_passed=overall,
        ))

    # Step 4: Score the full run (aggregate scorers)
    lat_score  = latency.score(responses)
    cost_score = cost.score(responses)

    # Step 5: Build summary statistics
    summary = _build_summary(eval_results, lat_score, cost_score)

    return EvalReport(
        model=model,
        test_suite=Path(suite_path).name,
        results=eval_results,
        summary=summary,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _build_summary(
    results: List[EvalResult],
    lat_score,
    cost_score,
) -> dict:
    """
    Aggregate per-test scores into run-level summary statistics.

    Returns a dict with avg scores per scorer and overall stats.
    """
    if not results:
        return {}

    # Collect all accuracy scores
    acc_scores  = [r.scores["accuracy"].score for r in results]
    hall_scores = [r.scores["hallucination"].score for r in results]

    # Count failures
    acc_failures  = sum(1 for r in results if not r.scores["accuracy"].passed)
    hall_failures = sum(1 for r in results if not r.scores["hallucination"].passed)

    return {
        "accuracy": {
            "avg_score": round(sum(acc_scores) / len(acc_scores), 4),
            "passed": acc_failures == 0,
            "failures": acc_failures,
        },
        "hallucination": {
            "avg_score": round(sum(hall_scores) / len(hall_scores), 4),
            "passed": hall_failures == 0,
            "failures": hall_failures,
            "rate": round(hall_failures / len(results), 4),
        },
        "latency": {
            "score": lat_score.score,
            "passed": lat_score.passed,
            **lat_score.details,
        },
        "cost": {
            "score": cost_score.score,
            "passed": cost_score.passed,
            **cost_score.details,
        },
    }
