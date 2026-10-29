"""
scorers/accuracy.py — Measures how accurate the LLM's response is.

WHAT IT DOES:
  Compares the LLM's response to the expected output using two methods:
  1. Exact match:  lowercased, stripped — did it say exactly the right thing?
  2. Fuzzy match:  SequenceMatcher ratio — how similar is it?

WHY TWO METHODS:
  Exact match is too strict for natural language.
  "The capital is Paris." vs "Paris is the capital of France." — both correct.
  Fuzzy match handles paraphrasing.

  SequenceMatcher from difflib:
  - Standard library (no install needed)
  - Returns ratio 0.0 to 1.0
  - "hello world" vs "hello world" → 1.0
  - "Paris" vs "The capital is Paris" → ~0.44 (Paris is in there)
  - "Paris" vs "London" → 0.0

LINE BY LINE:
  difflib.SequenceMatcher(None, a, b)
    None = no custom junk function (default)
    a, b = strings to compare
    .ratio() = similarity 0.0 to 1.0

  _normalize(text)       → lowercases and strips whitespace
                           so "Paris" == "paris " == "PARIS"

  exact_match            → 1.0 if normalized strings are equal, else 0.0

  fuzzy_score            → SequenceMatcher ratio on normalized strings

  score = max(exact_match, fuzzy_score)
                         → take the higher of the two
                         → exact match always wins if both are equal

  FUZZY_THRESHOLD        → minimum fuzzy ratio to count as "passed"
                           0.6 = at least 60% similar to expected
                           too low = false positives
                           too high = penalises valid paraphrases

  "contains" check       → if expected is short (< 50 chars), check if
                           the expected text appears inside the response
                           "Paris" appearing in a longer answer counts as correct
"""

import difflib
from models import LLMResponse, TestCase, ScoreResult

FUZZY_THRESHOLD = 0.6   # minimum similarity to count as correct


def score(response: LLMResponse, test_case: TestCase) -> ScoreResult:
    """
    Score the response against the expected output.

    Uses exact match, fuzzy match, and substring containment.
    Returns the highest score found.

    Args:
        response:  LLMResponse from the runner.
        test_case: TestCase with the expected output.

    Returns:
        ScoreResult with score 0.0-1.0 and match details.
    """
    actual   = _normalize(response.response)
    expected = _normalize(test_case.expected)

    # Method 1: Exact match (normalised)
    exact = 1.0 if actual == expected else 0.0

    # Method 2: Fuzzy similarity ratio
    fuzzy = difflib.SequenceMatcher(None, actual, expected).ratio()
    fuzzy = round(fuzzy, 4)

    # Method 3: Contains check for short expected outputs
    # "Paris" might appear in a longer correct answer
    contains = 0.0
    if len(expected) < 50 and expected in actual:
        contains = 0.85   # not perfect but clearly contains the right answer

    # Take the best score
    final_score = max(exact, fuzzy, contains)
    final_score = round(final_score, 4)

    # Determine which method "won"
    if exact == 1.0:
        method = "exact"
    elif contains == 0.85:
        method = "contains"
    else:
        method = "fuzzy"

    return ScoreResult(
        scorer="accuracy",
        score=final_score,
        passed=final_score >= FUZZY_THRESHOLD,
        details={
            "method": method,
            "exact_match": exact,
            "fuzzy_score": fuzzy,
            "contains_match": contains > 0,
            "expected_preview": test_case.expected[:80],
            "response_preview": response.response[:80],
        },
    )


def _normalize(text: str) -> str:
    """Lowercase and strip whitespace for fair comparison."""
    return text.lower().strip()
