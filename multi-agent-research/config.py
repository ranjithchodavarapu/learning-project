"""
config.py — Central settings for the multi-agent research system.

LINE BY LINE:
  import os            → built-in Python, reads environment variables
  from pathlib import Path → cleaner file path handling than plain strings
  from dotenv import load_dotenv → reads .env file into os.environ

  load_dotenv(...)     → finds .env two levels up from this file
  __file__             → the path of THIS file (config.py)
  .parent              → the folder containing this file (multi-agent-research/)
  / ".env"             → joins to make multi-agent-research/.env

  class Config         → groups all settings in one object
  os.getenv("X", Y)   → reads X from .env, uses Y as default if not found
  : str / : int        → type hints — tells you what type each setting is
  int(os.getenv(...))  → os.getenv always returns string, int() converts it

  cfg = Config()       → ONE instance created at import time
                         every other file does: from config import cfg
                         this is the Singleton pattern
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── LLM (Ollama runs locally — free, no API key needed) ───────────────
    LLM_PROVIDER: str  = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str  = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

    # ── Agent behaviour ───────────────────────────────────────────────────
    # How many research subtopics the orchestrator breaks a topic into
    MAX_SUBTOPICS: int = int(os.getenv("MAX_SUBTOPICS", 3))

    # How many facts the researcher extracts per subtopic
    FACTS_PER_SUBTOPIC: int = int(os.getenv("FACTS_PER_SUBTOPIC", 5))

    # Max seconds to wait for one LLM call before timing out
    TIMEOUT: int = int(os.getenv("TIMEOUT", 120))

    # ── Output ────────────────────────────────────────────────────────────
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./outputs")


cfg = Config()
