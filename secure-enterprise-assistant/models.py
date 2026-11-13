"""
models.py — Shared data structures. Imports nothing from this project,
so no other file can create a circular import with it.

LINE BY LINE:
  class Clearance(IntEnum)
      The four data classification levels, as ordered integers:
      PUBLIC=0 < INTERNAL=1 < CONFIDENTIAL=2 < RESTRICTED=3.
      IntEnum is the key choice. Because the levels are integers, the access
      rule is a plain comparison:  doc.classification <= user_clearance.
      No lookup table of "who may see what" that can fall out of sync.

  Clearance.from_name(name)
      Parses "confidential" -> Clearance.CONFIDENTIAL.
      An unknown or misspelled label returns RESTRICTED, not PUBLIC.
      This is FAIL CLOSED: when in doubt, deny. A typo in a document header
      must never accidentally make a secret document public.

  @dataclass(frozen=True) on Session
      frozen=True makes the object immutable. After login nobody can write
      `session.role = "admin"`. Python raises FrozenInstanceError. A session
      is a fact about the user at login time; it should not be editable.

  class User
      Stores password_hash and salt, NEVER the plain password.

  class Document / Chunk
      A document is the whole file plus its classification and owning
      department. A chunk is a small slice of a document used for retrieval.
      Every chunk carries its parent doc_id and classification, so the access
      check can always be re-verified at chunk level.

  class GuardResult
      allowed: pass or block.  risk: 0.0-1.0 score.  reasons: which rules
      fired (logged for the security team, never shown to the user).

  class OutputCheck
      Result of the output guard: allowed flag, the cleaned (PII-redacted)
      text, and reasons.

  class AssistantResponse
      What the caller gets back. `stage` records WHERE the pipeline stopped
      ("auth", "input_guard", "no_context", "output_guard", "ok"...), which
      is what the tests and the attack demo assert on.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List


class Clearance(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def from_name(cls, name: str) -> "Clearance":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            return cls.RESTRICTED   # fail closed


@dataclass
class User:
    username: str
    role: str
    department: str
    password_hash: str
    salt: str


@dataclass(frozen=True)
class Session:
    username: str
    role: str
    department: str
    expires_at: float


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    classification: Clearance
    department: str = "all"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    classification: Clearance


@dataclass
class GuardResult:
    allowed: bool
    risk: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class OutputCheck:
    allowed: bool
    cleaned: str = ""
    reasons: List[str] = field(default_factory=list)


@dataclass
class AssistantResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    blocked: bool = False
    stage: str = "ok"
    reason: str = ""
    risk: float = 0.0
