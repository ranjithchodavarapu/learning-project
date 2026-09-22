"""
config.py — Settings for the AI Software Engineer Agent.

LINE BY LINE:
  import os              → reads environment variables
  from pathlib import Path → clean file path handling
  from dotenv import load_dotenv → reads .env file

  load_dotenv(...)       → finds .env in same folder as this file

  class Config:
    OLLAMA_MODEL         → which local LLM to use for code generation
                           llama3.1:70b is best for code tasks on H100
    MAX_DEBUG_ATTEMPTS   → how many times debugger retries fixing broken code
                           3 = try original + 3 fixes = 4 total attempts
    WORKSPACE_DIR        → where generated code files are saved
                           each task gets its own subfolder inside here
    SUPPORTED_LANGUAGES  → languages the agent can generate code in
                           used to validate user input

  cfg = Config()         → single shared instance (Singleton pattern)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── LLM ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
    TIMEOUT: int         = int(os.getenv("TIMEOUT", 180))

    # ── Agent behaviour ───────────────────────────────────────────────────
    # How many times the debugger retries fixing failing tests
    MAX_DEBUG_ATTEMPTS: int = int(os.getenv("MAX_DEBUG_ATTEMPTS", 3))

    # ── Output ────────────────────────────────────────────────────────────
    WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", "./workspace")

    # ── Supported languages ───────────────────────────────────────────────
    SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "bash"]


cfg = Config()
