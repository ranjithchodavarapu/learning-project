"""
config.py — Settings for the Self-Correcting AI Agent.

WHAT IS SELF-CORRECTION:
  A normal LLM answers once and stops.
  A self-correcting agent:
    1. Generates an answer
    2. Evaluates it on multiple dimensions (accuracy, completeness, clarity etc.)
    3. If score is below threshold → reflects on what's wrong
    4. Retries with the critique as additional context
    5. Repeats until score passes OR max attempts reached

  This mirrors how a human expert would:
  - Write a first draft
  - Review it critically
  - Rewrite based on the review
  - Repeat until satisfied

LINE BY LINE:
  SCORE_THRESHOLD        → minimum acceptable score (0.0 to 1.0)
                           0.8 = agent must get 80% quality before stopping
                           higher = more retries, higher quality, more LLM calls
                           0.7 is a good balance for most tasks

  MAX_ATTEMPTS           → hard limit on retries
                           prevents infinite loops if LLM keeps failing
                           3 = initial attempt + 2 retries = 4 total LLM calls
                           per stage (generate + evaluate + reflect = 3 calls per attempt)

  EVALUATION_DIMENSIONS  → what the evaluator scores on
                           each dimension gets a 0.0-1.0 score
                           final score = average of all dimensions
                           choosing good dimensions is the art of evaluation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── LLM ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
    TIMEOUT: int         = int(os.getenv("TIMEOUT", 120))

    # ── Self-correction settings ──────────────────────────────────────────
    # Score must reach this to stop retrying (0.0 to 1.0)
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", 0.75))

    # Maximum number of generate→evaluate→reflect cycles
    MAX_ATTEMPTS: int = int(os.getenv("MAX_ATTEMPTS", 3))

    # ── Evaluation dimensions ─────────────────────────────────────────────
    # What the evaluator scores the answer on
    EVALUATION_DIMENSIONS = [
        "accuracy",       # is the information correct?
        "completeness",   # does it cover all aspects of the question?
        "clarity",        # is it easy to understand?
        "conciseness",    # is it appropriately brief without missing key info?
        "relevance",      # does it directly answer what was asked?
    ]


cfg = Config()
