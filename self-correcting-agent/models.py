"""
models.py — Shared data structures for the self-correcting agent.

LINE BY LINE:
  @dataclass             → auto-generates __init__, __repr__ etc.
  field(default_factory) → each instance gets its OWN fresh list/dict

  class EvaluationScore  → scores from the evaluator for ONE attempt
    dimensions: dict     → {"accuracy": 0.8, "clarity": 0.9, ...}
                           one float per evaluation dimension
    overall: float       → average of all dimension scores
    passed: bool         → True if overall >= SCORE_THRESHOLD
    feedback: str        → evaluator's written explanation of the scores
                           used by the reflector to understand what to fix

  class Attempt          → one full generate→evaluate cycle
    attempt_number: int  → 1, 2, 3 (for display)
    question: str        → original question (same every attempt)
    answer: str          → what the LLM generated this attempt
    critique: str        → what the reflector said was wrong (from previous attempt)
                           empty on first attempt
    score: EvaluationScore → scores for this attempt's answer

  class AgentResult      → final output of the whole pipeline
    question: str        → original question
    final_answer: str    → the best answer produced
    final_score: EvaluationScore → scores for the final answer
    attempts: list       → history of ALL attempts (for analysis)
    total_attempts: int  → how many cycles ran
    converged: bool      → True if score passed threshold
                           False if hit MAX_ATTEMPTS without passing

  @property              → makes a method callable like an attribute
    result.summary       → not result.summary()
    computed on the fly, not stored

  chr(10)                → newline character
                           used inside f-strings where \\n is not allowed
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EvaluationScore:
    """Scores from the evaluator for one answer attempt."""
    dimensions: Dict[str, float] = field(default_factory=dict)
    overall: float = 0.0        # average of all dimension scores
    passed: bool = False        # True if overall >= SCORE_THRESHOLD
    feedback: str = ""          # written explanation of what's good/bad


@dataclass
class Attempt:
    """One complete generate → evaluate cycle."""
    attempt_number: int
    question: str
    answer: str = ""
    critique: str = ""          # reflector's feedback from previous attempt
    score: EvaluationScore = field(default_factory=EvaluationScore)


@dataclass
class AgentResult:
    """Final output of the full self-correcting pipeline."""
    question: str
    final_answer: str = ""
    final_score: EvaluationScore = field(default_factory=EvaluationScore)
    attempts: List[Attempt] = field(default_factory=list)
    total_attempts: int = 0
    converged: bool = False     # True if score passed threshold

    @property
    def summary(self) -> str:
        """One-line summary of the result for logging."""
        status = "converged" if self.converged else "max attempts reached"
        return (
            f"Attempts: {self.total_attempts} | "
            f"Final score: {self.final_score.overall:.2f} | "
            f"Status: {status}"
        )

    @property
    def score_history(self) -> str:
        """Show score progression across attempts."""
        lines = []
        for a in self.attempts:
            lines.append(
                f"  Attempt {a.attempt_number}: {a.score.overall:.2f} "
                f"({'PASS' if a.score.passed else 'FAIL'})"
            )
        return chr(10).join(lines)
