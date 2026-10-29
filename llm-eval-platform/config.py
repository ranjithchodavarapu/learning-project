"""
config.py — Settings for the LLM Evaluation Platform.

WHAT THIS PLATFORM DOES:
  Tests LLM responses on 4 dimensions:
  1. Accuracy    — does the answer match the expected output?
  2. Latency     — how fast does it respond? (p50, p95, p99 percentiles)
  3. Hallucination — does it make up facts not in the prompt?
  4. Cost        — how many tokens does it use? (× price per token)

  This is what production AI teams run before deploying a new model
  or prompt change. It answers: "Is this model/prompt safe to ship?"

LINE BY LINE:
  MODELS_TO_EVAL       → list of Ollama models to compare
                          you can eval multiple models in one run
                          results are compared side-by-side in the report

  LATENCY_PERCENTILES  → which percentiles to measure
                          p50 = median (typical speed)
                          p95 = 95th percentile (slow cases)
                          p99 = 99th percentile (worst case)
                          production SLAs are usually written as p95 or p99

  HALLUCINATION_THRESHOLD → score below this = hallucination detected
                            0-1 scale from the LLM-as-judge evaluator
                            0.5 = judge thinks there's a 50% chance of hallucination

  TOKEN_PRICE_PER_1K   → cost per 1000 tokens
                          used to estimate cost per query and per run
                          for local Ollama models this is $0 but we track
                          token usage for when you move to a paid API

  REPORTS_DIR          → where markdown and JSON reports are saved
  TEST_SUITES_DIR      → where test case JSON files are stored
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── Ollama ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    TIMEOUT: int         = int(os.getenv("TIMEOUT", 120))

    # ── Models to evaluate ────────────────────────────────────────────────
    # Comma-separated list: "llama3.1:70b,llama3.1:8b"
    _models_str: str = os.getenv("MODELS_TO_EVAL", "llama3.1:70b")
    MODELS_TO_EVAL: list = [m.strip() for m in _models_str.split(",")]

    # ── Latency ───────────────────────────────────────────────────────────
    LATENCY_PERCENTILES: list = [50, 95, 99]

    # ── Hallucination ─────────────────────────────────────────────────────
    # Score below this = hallucination flagged
    HALLUCINATION_THRESHOLD: float = float(
        os.getenv("HALLUCINATION_THRESHOLD", 0.5)
    )
    # Model used as judge for hallucination detection
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "llama3.1:70b")

    # ── Cost estimation ───────────────────────────────────────────────────
    # Price per 1000 tokens (set to 0 for local Ollama models)
    TOKEN_PRICE_PER_1K: float = float(os.getenv("TOKEN_PRICE_PER_1K", 0.0))

    # ── Output ────────────────────────────────────────────────────────────
    REPORTS_DIR: str     = os.getenv("REPORTS_DIR", "./reports")
    TEST_SUITES_DIR: str = os.getenv("TEST_SUITES_DIR", "./test_suites")


cfg = Config()
