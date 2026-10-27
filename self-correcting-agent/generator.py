"""
generator.py — Generates answers (initial and improved retries).

WHAT IT DOES:
  Attempt 1: generates an answer to the bare question.
  Attempt 2+: generates an improved answer using the previous critique.

KEY DESIGN: critique-aware generation
  On retry, the LLM receives:
  - The original question
  - Its previous answer
  - The evaluator's scores
  - The reflector's critique (what was wrong and how to fix it)

  This is richer context than just the question — the LLM knows
  exactly what it did wrong and what to improve.

LINE BY LINE:
  SYSTEM_PROMPT          → "expert assistant who improves based on feedback"
                           The word "improves" is intentional — it primes
                           the LLM to think about improvement, not just answering

  INITIAL_PROMPT         → for attempt 1 — just the question
                           simple, no extra context yet

  RETRY_PROMPT           → for attempt 2+ — includes:
    {question}           → original question
    {previous_answer}    → what the LLM said last time
    {critique}           → what was wrong and needs fixing
    {scores}             → numerical scores per dimension
                           showing the LLM its own performance metrics

  "Address ALL the issues mentioned in the critique"
                         → explicit instruction — prevents the LLM from
                           fixing one issue and ignoring others

  temperature=0.7        → generation uses higher temperature than evaluation
                           we want thoughtful, varied answers not robotic ones
                           0.7 = creative but not hallucinating

  _format_scores()       → converts {"accuracy": 0.8, ...} to readable text
    f"{dim}: {score:.2f}"→ "accuracy: 0.80" (2 decimal places)
    "\n".join(...)        → one score per line
"""

from rich.console import Console
from models import Attempt
from llm import call_llm
from config import cfg

console = Console()


SYSTEM_PROMPT = """You are an expert assistant who gives accurate, clear, and complete answers.
When given feedback on a previous answer, you carefully address all the issues raised
and produce a significantly improved response."""


INITIAL_PROMPT = """Answer this question thoroughly and clearly:

{question}"""


RETRY_PROMPT = """You previously answered a question but received critical feedback.
Carefully address ALL the issues and write a much better answer.

Original question: {question}

Your previous answer:
{previous_answer}

Critique of your previous answer:
{critique}

Scores from evaluation (1.0 = perfect):
{scores}

Write an improved answer that directly addresses all the issues raised in the critique.
Do not repeat the same mistakes."""


def generate_answer(attempt: Attempt) -> str:
    """
    Generate an answer for this attempt.

    First attempt: answers the bare question.
    Subsequent attempts: improves using previous critique.

    Args:
        attempt: Current Attempt object with question and critique.

    Returns:
        Generated answer as plain string.
    """
    if attempt.attempt_number == 1 or not attempt.critique:
        # First attempt — no critique yet
        console.print("[blue]Generating initial answer...[/blue]")
        user_prompt = INITIAL_PROMPT.format(question=attempt.question)
    else:
        # Retry — use critique to improve
        console.print(
            f"[blue]Generating improved answer (attempt {attempt.attempt_number})...[/blue]"
        )
        # Format the scores from the previous attempt for the prompt
        prev_attempt_scores = attempt.score.dimensions
        scores_text = _format_scores(prev_attempt_scores)

        user_prompt = RETRY_PROMPT.format(
            question=attempt.question,
            previous_answer=attempt.answer,
            critique=attempt.critique,
            scores=scores_text,
        )

    answer = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)

    if not answer:
        console.print("[yellow]LLM returned empty answer[/yellow]")
        return "I was unable to generate an answer at this time."

    console.print(f"[green]✓ Answer generated ({len(answer.split())} words)[/green]")
    return answer


def _format_scores(dimensions: dict) -> str:
    """
    Format dimension scores as a readable list.

    Input:  {"accuracy": 0.8, "clarity": 0.6}
    Output: "accuracy: 0.80\nclarity: 0.60"
    """
    if not dimensions:
        return "No scores available"
    return "\n".join(
        f"{dim}: {score:.2f}" for dim, score in dimensions.items()
    )
