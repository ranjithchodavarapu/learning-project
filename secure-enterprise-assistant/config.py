"""
config.py — Settings for the Secure Enterprise AI Assistant.

LINE BY LINE (security-relevant lines):
  BASE_DIR = Path(__file__).parent
      All default paths are anchored to this file, not to the current working
      directory. Running `python main.py` from another folder still finds
      the documents and writes the audit log in the right place.

  SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
      The key that signs login tokens. There is NO hardcoded fallback value.
      A hardcoded default ("change-me") is a classic vulnerability: anyone who
      reads the source can forge tokens. If .env has no key we generate a
      random one per process. Tokens then stop working after a restart, which
      is the safe failure mode.
      secrets.token_hex(32) = 32 random bytes from the OS CSPRNG, as 64 hex chars.
      (`random` is NOT safe for this. It is predictable. `secrets` is not.)

  TOKEN_TTL_SECONDS
      Tokens expire. A stolen token is only useful for this long.

  MAX_FAILED_LOGINS / LOCKOUT_SECONDS
      Brute-force protection. After 3 wrong passwords the account is locked
      for 5 minutes.

  PBKDF2_ITERATIONS
      How many times the password hash function is repeated. More iterations
      = slower for attackers trying billions of guesses. 100,000 is a
      reasonable minimum for PBKDF2-HMAC-SHA256.

  MAX_QUERY_CHARS
      Hard cap on question length. Very long inputs are a common way to hide
      injection payloads or exhaust the LLM context window.

  INJECTION_BLOCK_THRESHOLD = 0.6
      Risk score (0 to 1) at or above which a request is blocked.
      Weak signals score about 0.5, so a single weak signal warns but does
      not block. Two weak signals combined do block.

  TOP_K
      How many chunks to retrieve. Fewer chunks = less text an attacker can
      poison, and less data exposed per question.
"""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # ── LLM (Ollama, local) ───────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1:70b")
    TIMEOUT: int         = int(os.getenv("TIMEOUT", 120))

    # ── Authentication ────────────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    TOKEN_TTL_SECONDS: int  = int(os.getenv("TOKEN_TTL_SECONDS", 3600))
    MAX_FAILED_LOGINS: int  = int(os.getenv("MAX_FAILED_LOGINS", 3))
    LOCKOUT_SECONDS: int    = int(os.getenv("LOCKOUT_SECONDS", 300))
    PBKDF2_ITERATIONS: int  = int(os.getenv("PBKDF2_ITERATIONS", 100_000))

    # ── Guardrails ────────────────────────────────────────────────────────
    MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", 1000))
    INJECTION_BLOCK_THRESHOLD: float = float(
        os.getenv("INJECTION_BLOCK_THRESHOLD", 0.6)
    )

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 3))
    DOCS_DIR: str = os.getenv("DOCS_DIR", str(BASE_DIR / "documents"))

    # ── Audit ─────────────────────────────────────────────────────────────
    AUDIT_LOG: str = os.getenv("AUDIT_LOG", str(BASE_DIR / "logs" / "audit.jsonl"))


cfg = Config()
