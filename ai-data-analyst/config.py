"""
config.py — Central settings for the AI Data Analyst.

LINE BY LINE:
  import os              → built-in, reads environment variables
  from pathlib import Path → clean file path handling
  from dotenv import load_dotenv → reads .env file

  load_dotenv(Path(__file__).parent / ".env")
    __file__  = this file's path (config.py)
    .parent   = the folder containing it (ai-data-analyst/)
    / ".env"  = joins to make ai-data-analyst/.env

  class Config           → groups all settings in one object
  os.getenv("X", Y)     → read X from .env, use Y as default
  int(os.getenv(...))   → os.getenv returns string, int() converts it

  MAX_RETRIES            → if generated code fails, retry this many times
                           the executor catches errors and asks LLM to fix them
                           this is the self-correcting loop shown in the diagram

  SAMPLE_ROWS            → how many rows to show the LLM as context
                           more rows = better code but more tokens used
                           5-10 rows is enough for the LLM to understand the data

  cfg = Config()         → ONE instance, shared everywhere (Singleton pattern)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── LLM (Ollama — free, local) ────────────────────────────────────────
    LLM_PROVIDER: str    = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

    # ── Code generation ───────────────────────────────────────────────────
    # How many times to retry if generated code throws an error
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

    # How many sample rows to show the LLM so it understands the data shape
    SAMPLE_ROWS: int = int(os.getenv("SAMPLE_ROWS", 5))

    # ── Output ────────────────────────────────────────────────────────────
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./outputs")

    # ── Execution ─────────────────────────────────────────────────────────
    # Seconds before an LLM call times out
    TIMEOUT: int = int(os.getenv("TIMEOUT", 120))


cfg = Config()
