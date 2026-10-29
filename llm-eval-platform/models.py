"""
models.py — Shared data structures for the evaluation platform.

LINE BY LINE:
  class TestCase         → one test: a prompt + expected output
    prompt: str          → what we send to the LLM
    expected: str        → what we expect it to say
    category: str        → group for filtering/reporting (e.g. "factual", "math")
    tags: list           → optional labels (e.g. ["hard", "hallucination-prone"])

  class LLMResponse      → raw response from one LLM call
    model: str           → which model answered
    prompt: str          → what was asked
    response: str        → what it said
    latency_ms: float    → milliseconds to get the response
    prompt_tokens: int   → tokens in the prompt
    completion_tokens: int → tokens in the response
    total_tokens: int    → sum of above

  class ScoreResult      → scores for ONE response from ONE scorer
    scorer: str          → "accuracy", "latency", "hallucination", "cost"
    score: float         → 0.0 to 1.0 (higher = better)
    passed: bool         → did it pass the threshold?
    details: dict        → scorer-specific details
                           accuracy: {"method": "fuzzy", "ratio": 0.92}
                           latency:  {"p50": 1200, "p95": 2300}
                           hallucination: {"judgement": "no hallucination"}
                           cost:     {"total_tokens": 450, "cost_usd": 0.0}

  class EvalResult       → complete evaluation of ONE test case
    test_case: TestCase  → the original test
    response: LLMResponse → what the LLM said
    scores: dict         → {"accuracy": ScoreResult, "latency": ScoreResult, ...}
    overall_passed: bool → True only if ALL scorers passed

  class EvalReport       → full report for a complete eval run
    model: str           → model that was evaluated
    test_suite: str      → name of the test suite file
    results: list        → one EvalResult per test case
    summary: dict        → aggregated stats across all test cases
    timestamp: str       → when the eval ran

  @property passed_count → counts how many EvalResults passed overall
  @property pass_rate    → passed_count / total as percentage
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TestCase:
    """One test: a prompt with an expected output."""
    prompt: str
    expected: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Raw response from one LLM call including timing and token counts."""
    model: str
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ScoreResult:
    """Score from one scorer for one response."""
    scorer: str                          # "accuracy", "latency" etc.
    score: float                         # 0.0 to 1.0
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation of one test case."""
    test_case: TestCase
    response: LLMResponse
    scores: Dict[str, ScoreResult] = field(default_factory=dict)
    overall_passed: bool = False


@dataclass
class EvalReport:
    """Full report for one complete evaluation run."""
    model: str
    test_suite: str
    results: List[EvalResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    @property
    def passed_count(self) -> int:
        """Number of test cases that passed all scorers."""
        return sum(1 for r in self.results if r.overall_passed)

    @property
    def pass_rate(self) -> float:
        """Percentage of test cases that passed (0-100)."""
        if not self.results:
            return 0.0
        return round(self.passed_count / len(self.results) * 100, 1)
