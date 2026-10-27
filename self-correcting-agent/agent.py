"""
agent.py — The main self-correcting loop.

WHAT IT DOES:
  Runs the generate → evaluate → reflect → retry loop until:
  - Score passes SCORE_THRESHOLD (success), OR
  - MAX_ATTEMPTS reached (best effort result)

THE LOOP:
  attempt 1:
    generate(question)           → answer_1
    evaluate(question, answer_1) → score_1
    if score_1.passed: DONE
    reflect(question, answer_1, score_1) → critique_1

  attempt 2:
    generate(question + critique_1) → answer_2
    evaluate(question, answer_2)    → score_2
    if score_2.passed: DONE
    reflect(question, answer_2, score_2) → critique_2

  attempt 3 (MAX_ATTEMPTS):
    generate(question + critique_2) → answer_3
    evaluate(question, answer_3)    → score_3
    DONE regardless of score (return best attempt)

LINE BY LINE:
  class SelfCorrectingAgent → wraps the whole loop in one object

  def run(question)        → main entry point
    attempts = []          → list of Attempt objects (full history)
    critique = ""          → empty on first attempt, filled after reflection

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
                           → range(1, 4) = [1, 2, 3] (1-indexed for display)

    current = Attempt(...)→ create attempt object for this cycle
    current.critique = critique
                           → attach critique from previous cycle
                           → empty string on attempt 1

    current.answer = generate_answer(current)
                           → call generator with the attempt object
                           → generator checks attempt.critique to decide
                             whether to use INITIAL_PROMPT or RETRY_PROMPT

    current.score = evaluate_answer(question, current.answer)
                           → call evaluator
                           → returns EvaluationScore with .passed bool

    attempts.append(current)
                           → save to history before checking pass/fail

    if current.score.passed:
        break              → score passed threshold → exit loop early

    if attempt_num < MAX_ATTEMPTS:
        critique = reflect_on_answer(...)
                           → only reflect if there's another attempt coming
                           → no point reflecting on the last attempt

  _select_best(attempts)   → picks the attempt with highest overall score
                           → might not be the last one
                           → if attempt 2 scores 0.85 but attempt 3 scores 0.78
                             (LLM sometimes gets worse) → return attempt 2

  console.rule(...)        → prints "────── Attempt 1 ──────" divider
  time.time()              → timing each attempt
"""

import time
from rich.console import Console
from rich.panel import Panel
from models import Attempt, AgentResult
from generator import generate_answer
from evaluator import evaluate_answer
from reflector import reflect_on_answer
from config import cfg

console = Console()


class SelfCorrectingAgent:
    """
    Agent that generates, evaluates, and self-corrects its answers.
    Keeps retrying until quality threshold is met or max attempts reached.
    """

    def run(self, question: str) -> AgentResult:
        """
        Run the full self-correction loop on a question.

        Args:
            question: The question or task to answer.

        Returns:
            AgentResult with best answer, scores, and full attempt history.
        """
        start = time.time()

        console.print(Panel(
            f"[bold blue]Self-Correcting AI Agent[/bold blue]\n"
            f"Question: [cyan]{question[:100]}{'...' if len(question) > 100 else ''}[/cyan]\n"
            f"Threshold: {cfg.SCORE_THRESHOLD} | Max attempts: {cfg.MAX_ATTEMPTS}",
            border_style="blue",
        ))

        attempts = []
        critique = ""   # empty on first attempt, filled after each reflection

        for attempt_num in range(1, cfg.MAX_ATTEMPTS + 1):

            console.rule(f"[bold]Attempt {attempt_num}/{cfg.MAX_ATTEMPTS}[/bold]")
            attempt_start = time.time()

            # Create attempt object for this cycle
            current = Attempt(
                attempt_number=attempt_num,
                question=question,
                critique=critique,   # carries critique from previous attempt
            )

            # ── Generate ──────────────────────────────────────────────────
            current.answer = generate_answer(current)

            # ── Evaluate ──────────────────────────────────────────────────
            current.score = evaluate_answer(question, current.answer)

            # Save to history
            attempts.append(current)

            attempt_elapsed = time.time() - attempt_start
            console.print(
                f"[dim]Attempt {attempt_num} score: "
                f"{current.score.overall:.2f} "
                f"({'PASS' if current.score.passed else 'FAIL'}) "
                f"in {attempt_elapsed:.0f}s[/dim]"
            )

            # ── Check if passed ───────────────────────────────────────────
            if current.score.passed:
                console.print(
                    f"[bold green]✓ Score {current.score.overall:.2f} "
                    f">= threshold {cfg.SCORE_THRESHOLD} — converged![/bold green]"
                )
                break

            # ── Reflect (only if more attempts remain) ────────────────────
            if attempt_num < cfg.MAX_ATTEMPTS:
                console.print(
                    f"[yellow]Score {current.score.overall:.2f} below threshold "
                    f"{cfg.SCORE_THRESHOLD} — reflecting...[/yellow]"
                )
                critique = reflect_on_answer(question, current.answer, current.score)
            else:
                console.print(
                    f"[yellow]Max attempts reached — returning best result[/yellow]"
                )

        # ── Select best attempt ───────────────────────────────────────────
        best = _select_best(attempts)
        elapsed = time.time() - start

        # ── Build result ──────────────────────────────────────────────────
        result = AgentResult(
            question=question,
            final_answer=best.answer,
            final_score=best.score,
            attempts=attempts,
            total_attempts=len(attempts),
            converged=best.score.passed,
        )

        # ── Display summary ───────────────────────────────────────────────
        console.rule("[bold green]Complete[/bold green]")
        console.print(Panel(
            f"[bold]Result:[/bold] {'Converged ✓' if result.converged else 'Best effort ⚠'}\n"
            f"[bold]Final score:[/bold] {result.final_score.overall:.2f}\n"
            f"[bold]Total attempts:[/bold] {result.total_attempts}\n"
            f"[bold]Time:[/bold] {elapsed:.0f}s\n\n"
            f"[bold]Score progression:[/bold]\n{result.score_history}",
            title="[bold]Summary[/bold]",
            border_style="green" if result.converged else "yellow",
        ))

        return result


def _select_best(attempts: list) -> Attempt:
    """
    Select the attempt with the highest overall score.

    Not always the last attempt — LLMs sometimes produce worse
    answers when over-correcting. Take the best from all attempts.

    Uses max() with a key function:
    max(attempts, key=lambda a: a.score.overall)
    → returns the Attempt object where a.score.overall is highest
    """
    return max(attempts, key=lambda a: a.score.overall)
