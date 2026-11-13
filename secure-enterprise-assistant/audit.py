"""
audit.py — Tamper-evident audit log.

WHY AN AUDIT LOG:
  When something goes wrong (a leak, an attack, a complaint) the first
  question is "who asked what, and what did the system do?". Compliance
  regimes (SOC 2, ISO 27001, GDPR accountability) require this record.

WHY HASH-CHAINED:
  A plain log file can be edited after the fact to hide an attack.
  Here every entry stores the SHA-256 hash of the PREVIOUS entry, like a tiny
  blockchain. Change or delete any past line and every later hash no longer
  matches. verify() walks the file and reports the first broken line.
  (An attacker who can rewrite the WHOLE file can recompute the chain, so in
  production you also ship logs off-box. Here the chain gives tamper
  EVIDENCE, not tamper-proofing.)

LINE BY LINE:
  .jsonl = JSON Lines: one JSON object per line. Append-only, easy to grep,
  and a crash can corrupt at most the last line.

  self._lock = threading.Lock()
      Two requests writing at once could interleave and break the chain.
      The lock makes record() one-at-a-time.

  _load_last_hash()
      On start-up, read the last line so a restart continues the same chain
      instead of starting a new one.

  json.dumps(entry, sort_keys=True)
      sort_keys makes the serialisation deterministic. Hashing must always
      see the same bytes for the same entry.

  entry["hash"] = sha256(prev_hash + payload)
      Binds this entry to the one before it.

  record(event, user, **details)
      **details collects any extra keyword arguments into a dict, so call
      sites read naturally: record("input_blocked", user="carol", risk=0.9).

  NEVER LOG: passwords, tokens, API keys. Questions are cut to 200 chars at
  the call site because users sometimes paste secrets into prompts.

  verify()
      For each line: remove "hash", recompute it, compare with the stored one
      (constant-time), and check prev_hash equals the previous line's hash.
      Returns (True, None) or (False, line_number_of_first_problem).
"""

import hashlib
import hmac
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

GENESIS = "0" * 64   # the "previous hash" of the very first entry


def _digest(prev_hash: str, entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True)
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line).get("hash", last)
        return last

    def record(self, event: str, user: str = "anonymous", **details) -> dict:
        with self._lock:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "user": user,
                "details": details,
                "prev_hash": self._last_hash,
            }
            entry["hash"] = _digest(self._last_hash, entry)   # computed before "hash" exists
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._last_hash = entry["hash"]
            return entry

    def verify(self) -> Tuple[bool, Optional[int]]:
        if not self.path.exists():
            return True, None

        prev = GENESIS
        with open(self.path, "r", encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False, number

                stored = entry.pop("hash", "")
                if entry.get("prev_hash") != prev:
                    return False, number
                if not hmac.compare_digest(stored, _digest(prev, entry)):
                    return False, number
                prev = stored
        return True, None
