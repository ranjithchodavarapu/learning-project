"""
scorers/cost.py — Measures token usage and estimates cost.

WHAT IT DOES:
  Counts tokens used across all responses and estimates:
  - Total tokens (prompt + completion)
  - Average tokens per request
  - Estimated cost in USD

WHY TRACK COST EVEN FOR FREE LOCAL MODELS:
  1. When you move from Ollama to a paid API (GPT-4, Claude API),
     you already know exactly what it will cost.
  2. Token count = proxy for response length = proxy for quality vs efficiency.
     A model that uses 3x tokens but gives same quality costs more to operate.
  3. Token limits: if avg tokens per request is 2000 and context is 4096,
     you have limited room for multi-turn conversations.

LINE BY LINE:
  total_tokens = sum(r.total_tokens for r in responses)
                         → generator expression: sum all token counts
                         → r.total_tokens = prompt_tokens + completion_tokens

  avg_tokens = total_tokens / len(responses)
                         → average per request

  estimated_cost = (total_tokens / 1000) * cfg.TOKEN_PRICE_PER_1K
                         → total_tokens / 1000 = thousands of tokens
                         → × price per 1K = total cost in USD
                         → e.g. 10000 tokens × $0.002/1K = $0.02

  score                  → for cost, lower token usage = higher score
                           score = 1.0 - (avg_tokens / TOKEN_BUDGET)
                           TOKEN_BUDGET = 2000 tokens = "reasonable" request
                           avg < 1000 → score > 0.5 (efficient)
                           avg > 2000 → score < 0.5 (expensive)
                           avg > 4000 → score = 0.0 (very expensive)

  passed = True always   → cost is informational, not a pass/fail gate
                           we never fail a model purely for using tokens
                           (unless TOKEN_PRICE_PER_1K is set)
"""

from typing import List
from models import LLMResponse, ScoreResult
from config import cfg

TOKEN_BUDGET = 2000   # "reasonable" tokens per request for scoring


def score(responses: List[LLMResponse]) -> ScoreResult:
    """
    Score token usage and estimate cost across all responses.

    Args:
        responses: All LLMResponse objects from the run.

    Returns:
        ScoreResult with token stats and cost estimate.
        Always passes — cost is informational.
    """
    if not responses:
        return ScoreResult(
            scorer="cost",
            score=1.0,
            passed=True,
            details={"error": "No responses to score"},
        )

    # Sum token counts (0 for responses that didn't return metadata)
    total_tokens      = sum(r.total_tokens for r in responses)
    prompt_tokens     = sum(r.prompt_tokens for r in responses)
    completion_tokens = sum(r.completion_tokens for r in responses)

    avg_tokens = total_tokens / len(responses) if responses else 0

    # Estimate cost: total_tokens / 1000 × price_per_1k
    estimated_cost = (total_tokens / 1000) * cfg.TOKEN_PRICE_PER_1K

    # Score: 1.0 = very efficient, 0.0 = very expensive
    # Based on avg tokens vs budget
    raw_score = 1.0 - (avg_tokens / (TOKEN_BUDGET * 2))
    final_score = round(max(0.0, min(1.0, raw_score)), 4)

    return ScoreResult(
        scorer="cost",
        score=final_score,
        passed=True,    # cost is informational — always passes
        details={
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_tokens_per_request": round(avg_tokens, 1),
            "estimated_cost_usd": round(estimated_cost, 6),
            "price_per_1k": cfg.TOKEN_PRICE_PER_1K,
            "request_count": len(responses),
        },
    )
