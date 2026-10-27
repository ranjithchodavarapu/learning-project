"""
evaluator.py — Scores the generated answer on multiple dimensions.

WHAT IT DOES:
  Takes the question + answer and asks the LLM to score it on:
  - accuracy:     is the information factually correct?
  - completeness: does it cover all aspects?
  - clarity:      is it easy to understand?
  - conciseness:  appropriately brief?
  - relevance:    directly answers what was asked?

  Returns an EvaluationScore with individual scores + overall average.

WHY LLM-AS-JUDGE:
  Using a second LLM call to evaluate the first LLM's output
  is called "LLM-as-judge" — a standard technique in AI evaluation.

  Alternatives:
  - Human evaluation: most accurate but slow and expensive
  - Rule-based: fast but can't judge quality of complex answers
  - LLM-as-judge: good balance — understands nuance, scales well

  The evaluator uses a LOWER temperature (0.1) than the generator (0.7).
  We want CONSISTENT, STRICT scoring — not creative evaluation.
  The same answer should get the same score each time.

LINE BY LINE:
  EVAL_SYSTEM_PROMPT     → "strict evaluator" persona
                           "strict" = don't give everyone 0.9 just to be nice
                           "Return ONLY JSON" = machine-parseable output

  EVAL_PROMPT            → includes the full evaluation rubric
    0.0-0.3: poor        → helps LLM calibrate what each score means
    0.4-0.6: adequate    → without rubric, LLM might score everything 0.8
    0.7-0.8: good        → explicit ranges = more consistent scoring
    0.9-1.0: excellent   → the rubric IS the calibration

  call_llm_json()        → returns JSON string from Ollama
  json.loads(raw)        → parses JSON string → Python dict
  data.get("scores", {}) → safe access — returns {} if key missing

  _parse_scores()        → converts raw JSON to EvaluationScore
    for dim in EVALUATION_DIMENSIONS:
                         → iterate configured dimensions
    float(scores_data.get(dim, 0.5))
                         → get score for this dimension
                         → default 0.5 if missing (neutral)
                         → float() converts "0.8" → 0.8

  overall = sum(values) / len(values)
                         → arithmetic mean of all dimension scores
                         → all dimensions weighted equally

  overall >= cfg.SCORE_THRESHOLD
                         → True = answer is good enough → stop retrying
                         → False = needs improvement → reflect and retry
"""

import json
from rich.console import Console
from rich.table import Table
from models import EvaluationScore
from llm import call_llm_json
from config import cfg

console = Console()


EVAL_SYSTEM_PROMPT = """You are a strict, objective evaluator of answer quality.
Score the given answer on multiple dimensions.
Be critical and honest — don't inflate scores.
Return ONLY a JSON object with scores."""


EVAL_PROMPT = """Evaluate this answer to the given question.

Question: {question}

Answer to evaluate:
{answer}

Score the answer on these dimensions (0.0 to 1.0):
{dimensions}

Scoring rubric:
  0.0-0.3: poor — major issues, largely wrong or unhelpful
  0.4-0.6: adequate — some issues but basically correct
  0.7-0.8: good — minor issues, mostly correct and helpful
  0.9-1.0: excellent — near perfect, complete and accurate

Return ONLY this JSON structure:
{{
  "scores": {{
    "accuracy": 0.0,
    "completeness": 0.0,
    "clarity": 0.0,
    "conciseness": 0.0,
    "relevance": 0.0
  }},
  "feedback": "One paragraph explaining the scores and main issues"
}}"""


def evaluate_answer(question: str, answer: str) -> EvaluationScore:
    """
    Score an answer on all configured evaluation dimensions.

    Args:
        question: The original question asked.
        answer:   The generated answer to evaluate.

    Returns:
        EvaluationScore with per-dimension scores, overall, and feedback.
    """
    console.print("[blue]Evaluating answer...[/blue]")

    # Format dimensions list for the prompt
    dimensions_text = "\n".join(
        f"- {dim}" for dim in cfg.EVALUATION_DIMENSIONS
    )

    user_prompt = EVAL_PROMPT.format(
        question=question,
        answer=answer,
        dimensions=dimensions_text,
    )

    raw = call_llm_json(EVAL_SYSTEM_PROMPT, user_prompt)

    if not raw:
        console.print("[yellow]Evaluator LLM failed — returning neutral score[/yellow]")
        return _neutral_score()

    score = _parse_scores(raw)
    _display_scores(score)
    return score


def _parse_scores(raw: str) -> EvaluationScore:
    """
    Parse LLM JSON response into an EvaluationScore.

    Handles malformed JSON gracefully — returns neutral score on failure.
    """
    # Strip markdown fences if LLM added them
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Score JSON parse failed: {e}[/yellow]")
        return _neutral_score()

    scores_data = data.get("scores", {})
    feedback    = data.get("feedback", "No feedback provided")

    # Extract score for each configured dimension
    dimensions = {}
    for dim in cfg.EVALUATION_DIMENSIONS:
        try:
            # float() converts string "0.8" to float 0.8
            dimensions[dim] = float(scores_data.get(dim, 0.5))
        except (ValueError, TypeError):
            dimensions[dim] = 0.5   # neutral default on conversion error

    # Overall = arithmetic mean of all dimension scores
    overall = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
    overall = round(overall, 4)

    return EvaluationScore(
        dimensions=dimensions,
        overall=overall,
        passed=overall >= cfg.SCORE_THRESHOLD,
        feedback=feedback,
    )


def _neutral_score() -> EvaluationScore:
    """
    Return a neutral 0.5 score when evaluation fails.
    0.5 = below threshold → triggers retry → better than silently passing.
    """
    dims = {dim: 0.5 for dim in cfg.EVALUATION_DIMENSIONS}
    return EvaluationScore(
        dimensions=dims,
        overall=0.5,
        passed=False,
        feedback="Evaluation failed — neutral score assigned",
    )


def _display_scores(score: EvaluationScore) -> None:
    """Pretty-print evaluation scores as a table."""
    table = Table(title="Evaluation Scores")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Status", style="dim")

    for dim, val in score.dimensions.items():
        color = "green" if val >= 0.75 else "yellow" if val >= 0.5 else "red"
        table.add_row(
            dim,
            f"[{color}]{val:.2f}[/{color}]",
            "✓" if val >= cfg.SCORE_THRESHOLD else "✗",
        )

    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{'green' if score.passed else 'red'}]{score.overall:.2f}[/bold]",
        "[green]PASS[/green]" if score.passed else "[red]FAIL[/red]",
    )

    console.print(table)
    console.print(f"[dim]Feedback: {score.feedback[:150]}...[/dim]")
