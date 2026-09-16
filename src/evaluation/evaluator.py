"""
Evaluator — M6: End-to-end RAG evaluation with RAGAS.

WHY EVALUATION IS NON-NEGOTIABLE:
  Without metrics, "the RAG system is working" means nothing.
  You can't improve what you don't measure.
  You can't detect regressions when you change chunking/models.

RAGAS METRICS EXPLAINED:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ RETRIEVAL METRICS                                                   │
  │                                                                     │
  │ Context Recall:                                                     │
  │   Were all facts needed to answer the question present in the       │
  │   retrieved chunks?                                                 │
  │   Score 1.0 = retrieved context covers 100% of the ground truth.   │
  │   Score 0.4 = retrieved context is missing 60% of needed facts.    │
  │                                                                     │
  │ Context Precision:                                                  │
  │   Of the chunks retrieved, how many were actually relevant?        │
  │   Score 1.0 = every retrieved chunk was useful.                    │
  │   Score 0.3 = 70% of retrieved chunks were noise.                  │
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │ GENERATION METRICS                                                  │
  │                                                                     │
  │ Faithfulness:                                                       │
  │   Is every claim in the answer supported by the retrieved context? │
  │   Score 1.0 = answer is fully grounded in the retrieved chunks.    │
  │   Score 0.5 = half the claims are hallucinated (not in context).   │
  │                                                                     │
  │ Answer Relevance:                                                   │
  │   Does the answer actually address the question?                   │
  │   Score 1.0 = answer fully addresses the question.                 │
  │   Score 0.2 = answer is off-topic.                                 │
  └─────────────────────────────────────────────────────────────────────┘

  RAGAS Score = harmonic mean of all four = overall system quality.
  Target for production: > 0.75 on all metrics.

TEST SET FORMAT (data/test_set.json):
  [
    {
      "question": "What is the main contribution of the paper?",
      "ground_truth": "The paper proposes a new attention mechanism..."
    },
    ...
  ]
  You write these by hand for your domain — 20-50 QA pairs is enough for
  meaningful signal.

RAGAS uses LLM-as-judge internally (calls GPT-4 to score faithfulness etc.)
so you need an API key. It does NOT require ground truth answers for all metrics
(faithfulness and answer relevance are reference-free).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table

console = Console()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """One row in the evaluation set."""
    question: str
    ground_truth: str         # human-written reference answer
    answer: str               # RAG system's answer
    contexts: List[str]       # retrieved chunks (text only)


@dataclass
class EvalResult:
    """Scores for one sample or the whole test set (averaged)."""
    faithfulness: float          # 0-1: is answer grounded in context?
    answer_relevance: float      # 0-1: does answer address the question?
    context_recall: float        # 0-1: does context cover the ground truth?
    context_precision: float     # 0-1: is retrieved context precise?

    @property
    def ragas_score(self) -> float:
        """Harmonic mean of all four metrics (overall quality)."""
        scores = [self.faithfulness, self.answer_relevance,
                  self.context_recall, self.context_precision]
        if any(s == 0 for s in scores):
            return 0.0
        n = len(scores)
        return round(n / sum(1 / s for s in scores), 4)


# ── Test set loader ───────────────────────────────────────────────────────────

def load_test_set(path: str = "data/test_set.json") -> List[dict]:
    """
    Load Q&A pairs from a JSON file.
    Format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Test set not found at {path}.\n"
            "Create data/test_set.json with your QA pairs."
        )
    with open(p) as f:
        return json.load(f)


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def evaluate_with_ragas(samples: List[EvalSample]) -> EvalResult:
    """
    Run RAGAS evaluation on a list of EvalSamples.

    Requires: pip install ragas
    Requires: OPENAI_API_KEY set (RAGAS uses GPT as judge internally).

    Args:
        samples: List of EvalSample (question + answer + contexts + ground_truth).

    Returns:
        EvalResult with averaged scores across all samples.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        raise ImportError(
            "Install evaluation dependencies:\n"
            "  pip install ragas datasets"
        )

    # RAGAS expects a HuggingFace Dataset with specific column names
    data = {
        "question":    [s.question for s in samples],
        "answer":      [s.answer for s in samples],
        "contexts":    [s.contexts for s in samples],
        "ground_truth":[s.ground_truth for s in samples],
    }
    dataset = Dataset.from_dict(data)

    console.print(f"[blue]Running RAGAS evaluation on {len(samples)} samples...[/blue]")
    console.print("[dim](This calls your LLM API to judge each answer — may take a minute)[/dim]")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    return EvalResult(
        faithfulness=round(result["faithfulness"], 4),
        answer_relevance=round(result["answer_relevancy"], 4),
        context_recall=round(result["context_recall"], 4),
        context_precision=round(result["context_precision"], 4),
    )


# ── Simple rule-based eval (no API needed) ────────────────────────────────────

def evaluate_simple(samples: List[EvalSample]) -> EvalResult:
    """
    Lightweight evaluation without calling an LLM API.
    Uses string overlap heuristics — good for fast iteration / CI.

    Metrics:
      - faithfulness:      fraction of answer sentences that overlap with context
      - answer_relevance:  1.0 if question words appear in answer (rough proxy)
      - context_recall:    fraction of ground_truth words in retrieved context
      - context_precision: fraction of retrieved context words in ground_truth
    """
    def token_overlap(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta)

    faithfulness_scores, relevance_scores, recall_scores, precision_scores = [], [], [], []

    for s in samples:
        full_context = " ".join(s.contexts)

        # Faithfulness: how much of the answer is covered by context
        faithfulness_scores.append(token_overlap(s.answer, full_context))

        # Answer relevance: how much of the question appears in the answer
        relevance_scores.append(token_overlap(s.question, s.answer))

        # Context recall: how much of the ground truth is in the context
        recall_scores.append(token_overlap(s.ground_truth, full_context))

        # Context precision: how much of the context is relevant to ground truth
        precision_scores.append(token_overlap(full_context, s.ground_truth))

    return EvalResult(
        faithfulness=round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        answer_relevance=round(sum(relevance_scores) / len(relevance_scores), 4),
        context_recall=round(sum(recall_scores) / len(recall_scores), 4),
        context_precision=round(sum(precision_scores) / len(precision_scores), 4),
    )


# ── Display ───────────────────────────────────────────────────────────────────

def display_eval_results(result: EvalResult, milestone: str = "current") -> None:
    """Pretty-print evaluation results to the terminal."""
    table = Table(title=f"RAG Evaluation — {milestone}")
    table.add_column("Metric", style="bold")
    table.add_column("Score", style="cyan")
    table.add_column("Interpretation", style="dim")
    table.add_column("Target", style="green")

    def color(score: float) -> str:
        s = f"{score:.4f}"
        if score >= 0.75:
            return f"[green]{s}[/green]"
        elif score >= 0.5:
            return f"[yellow]{s}[/yellow]"
        else:
            return f"[red]{s}[/red]"

    rows = [
        ("Faithfulness",       result.faithfulness,       "Answer grounded in context",   ">0.75"),
        ("Answer relevance",   result.answer_relevance,   "Answer addresses the question", ">0.75"),
        ("Context recall",     result.context_recall,     "Context covers ground truth",   ">0.75"),
        ("Context precision",  result.context_precision,  "Retrieval is not noisy",        ">0.75"),
        ("RAGAS score",        result.ragas_score,        "Overall system quality",        ">0.75"),
    ]
    for name, score, interp, target in rows:
        table.add_row(name, color(score), interp, target)

    console.print(table)
