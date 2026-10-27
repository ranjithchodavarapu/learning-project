"""
reflector.py — Critiques the answer and creates an improvement plan.

WHAT IT DOES:
  Takes the question + answer + evaluation scores and produces:
  1. A diagnosis of exactly what's wrong
  2. A specific improvement plan for the next attempt

WHY A SEPARATE REFLECTION STEP:
  The evaluator gives SCORES (numbers + brief feedback).
  The reflector gives INSTRUCTIONS (specific, actionable critique).

  Scores tell you "clarity is 0.6".
  Reflection tells you "the answer uses jargon like 'homeomorphic' without
  explaining it — replace with plain language examples".

  The generator reads the reflection and knows EXACTLY what to fix.
  This is metacognition — thinking about your own thinking.

  This mirrors how humans improve:
  - Feedback: "your essay was unclear"
  - Reflection: "paragraph 2 has no topic sentence, conclusion
                 doesn't follow from the body, fix these specifically"

LINE BY LINE:
  REFLECT_SYSTEM_PROMPT  → "expert critic who helps improve writing"
                           critic persona = constructive, specific, actionable
                           "specific, actionable improvements" = key instruction
                           vague critique ("be clearer") is useless
                           specific critique ("use bullet points for the 3 steps") works

  REFLECT_PROMPT         → includes ALL context:
    question             → what was asked
    answer               → what was said
    scores               → numerical performance
    threshold            → what score we need to reach
    feedback             → evaluator's written notes
    dimensions with low scores → explicitly calls out weak areas

  "Focus especially on these low-scoring dimensions"
                         → directs the reflector's attention
                           without this, it might write generic advice
                           that ignores the specific weak areas

  _get_low_dimensions()  → finds dimensions scoring below threshold
    returns list of (dim_name, score) pairs
    sorted by score ascending (worst first)

  dim_list               → formatted as bullet points for the prompt
    "• accuracy: 0.60\n• clarity: 0.55"
    makes it easy for the LLM to scan which areas need most work

  temperature=0.3        → reflection needs structured thinking
                           lower than generation (0.7) — we want
                           focused, specific critique not creative variation
"""

from rich.console import Console
from models import EvaluationScore
from llm import call_llm
from config import cfg

console = Console()


REFLECT_SYSTEM_PROMPT = """You are an expert critic helping improve answer quality.
Analyse what's wrong with an answer and provide specific, actionable improvements.
Be direct and concrete — vague feedback is useless.
Focus on the most impactful changes."""


REFLECT_PROMPT = """Analyse this answer and provide a detailed improvement plan.

Question: {question}

Current answer:
{answer}

Evaluation scores (threshold to pass: {threshold}):
{scores_text}

Evaluator feedback:
{feedback}

Focus especially on these low-scoring dimensions:
{low_dimensions}

Provide:
1. DIAGNOSIS: What specific problems does this answer have?
2. IMPROVEMENT PLAN: Exactly what changes should be made?

Be specific — say "add an example of X" not "be clearer".
Say "remove the jargon in paragraph 2" not "simplify language".
The next writer will use your critique to rewrite the answer from scratch."""


def reflect_on_answer(
    question: str,
    answer: str,
    score: EvaluationScore,
) -> str:
    """
    Generate a critique and improvement plan for the current answer.

    Called when evaluation score is below threshold.
    The returned critique is passed to the generator for the next attempt.

    Args:
        question: The original question.
        answer:   The current answer that scored below threshold.
        score:    EvaluationScore with dimensions and feedback.

    Returns:
        Critique string — specific improvement instructions for the generator.
    """
    console.print("[blue]Reflecting on answer quality...[/blue]")

    # Format all scores as readable text
    scores_text = "\n".join(
        f"  {dim}: {val:.2f}" for dim, val in score.dimensions.items()
    )

    # Identify and highlight the worst-performing dimensions
    low_dims = _get_low_dimensions(score)
    if low_dims:
        dim_list = "\n".join(
            f"• {dim}: {val:.2f}" for dim, val in low_dims
        )
    else:
        dim_list = "All dimensions are close to threshold — make general improvements"

    user_prompt = REFLECT_PROMPT.format(
        question=question,
        answer=answer,
        threshold=cfg.SCORE_THRESHOLD,
        scores_text=scores_text,
        feedback=score.feedback,
        low_dimensions=dim_list,
    )

    critique = call_llm(REFLECT_SYSTEM_PROMPT, user_prompt, temperature=0.3)

    if not critique:
        # Fallback: generate a basic critique from the scores
        console.print("[yellow]Reflector LLM failed — using score-based critique[/yellow]")
        return _fallback_critique(score)

    console.print(f"[green]✓ Critique generated ({len(critique.split())} words)[/green]")
    return critique


def _get_low_dimensions(score: EvaluationScore) -> list:
    """
    Find dimensions scoring below the threshold.

    Returns list of (dimension_name, score) tuples,
    sorted by score ascending (worst first).
    """
    low = [
        (dim, val)
        for dim, val in score.dimensions.items()
        if val < cfg.SCORE_THRESHOLD
    ]
    # Sort worst first so prompt highlights most critical issues
    return sorted(low, key=lambda x: x[1])


def _fallback_critique(score: EvaluationScore) -> str:
    """
    Generate a basic critique when the LLM reflector fails.
    Uses scores to identify which areas to improve.
    """
    low_dims = _get_low_dimensions(score)

    if not low_dims:
        return "Make general improvements to the answer quality."

    issues = ", ".join(dim for dim, _ in low_dims)
    return (
        f"The answer needs improvement in: {issues}. "
        f"Overall score {score.overall:.2f} is below the required {cfg.SCORE_THRESHOLD}. "
        f"Rewrite the answer focusing on these areas: {issues}."
    )
