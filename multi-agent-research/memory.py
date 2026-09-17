"""
memory.py — Shared memory that all agents read and write.

WHY SHARED MEMORY EXISTS:
  In a single-agent system, one LLM call holds all context.
  In a multi-agent system, agents run separately — they need a
  central place to share what they found.

  Think of it like a shared whiteboard in a team:
  - Researcher writes facts onto it
  - Verifier reads those facts and marks which are confirmed
  - Summarizer reads confirmed facts and writes the report
  - Orchestrator reads everything to decide what to do next

LINE BY LINE:
  @dataclass           → auto-generates __init__, __repr__ etc.
                         without it you'd write def __init__(self, topic="",...) manually

  field(default_factory=list)
                       → each ResearchMemory gets its OWN empty list
                         if you wrote facts: list = [] instead,
                         ALL instances would share the SAME list (Python bug trap)

  topic: str = ""      → the research topic entered by the user
  subtopics: list      → list of strings — subtopics orchestrator planned
  facts: list          → list of dicts — each fact found by researcher
  verified_facts: list → subset of facts confirmed by verifier
  conflicts: list      → facts that contradict each other
  report: str          → final written report from summarizer
  status: str          → "planning" | "researching" | "verifying" | "summarizing" | "done"

  to_dict()            → converts the whole memory to a plain dict
                         used to show the LLM current state as JSON string

  add_fact()           → safe way to add a fact — checks for duplicates
                         agents call this instead of memory.facts.append()
                         because append() doesn't check for duplicates

  get_summary()        → quick text snapshot of what's been done so far
                         printed to terminal so you can watch agents work
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class ResearchMemory:
    """
    Single shared state object passed between all agents.
    Every agent receives this object, reads what others wrote,
    and writes their own findings back to it.
    """

    # ── Input ─────────────────────────────────────────────────────────────
    topic: str = ""                              # what the user wants researched

    # ── Orchestrator writes these ─────────────────────────────────────────
    subtopics: List[str] = field(default_factory=list)   # broken-down sub-questions
    research_plan: str = ""                              # orchestrator's plan as text

    # ── Researcher writes these ───────────────────────────────────────────
    # Each fact is a dict: {"subtopic": "...", "fact": "...", "source": "..."}
    facts: List[Dict[str, Any]] = field(default_factory=list)

    # ── Verifier writes these ─────────────────────────────────────────────
    # Confirmed facts — subset of facts that passed verification
    verified_facts: List[Dict[str, Any]] = field(default_factory=list)
    # Contradictions found between facts
    conflicts: List[str] = field(default_factory=list)
    # Verification notes for each fact
    verification_notes: List[str] = field(default_factory=list)

    # ── Summarizer writes this ────────────────────────────────────────────
    report: str = ""                             # the final research report

    # ── Pipeline tracking ─────────────────────────────────────────────────
    # Tracks which stage the pipeline is currently in
    status: str = "idle"
    errors: List[str] = field(default_factory=list)


    def to_dict(self) -> dict:
        """
        Convert memory to a plain Python dict.

        WHY: When we pass memory state to the LLM as context, we need
        a string. json.dumps(memory.to_dict()) gives a clean JSON string
        the LLM can read and reason about.
        """
        return {
            "topic": self.topic,
            "subtopics": self.subtopics,
            "research_plan": self.research_plan,
            "facts_count": len(self.facts),
            "facts": self.facts,
            "verified_facts_count": len(self.verified_facts),
            "verified_facts": self.verified_facts,
            "conflicts": self.conflicts,
            "status": self.status,
        }


    def to_json(self) -> str:
        """
        Convert to formatted JSON string.

        json.dumps(..., indent=2) → pretty-printed JSON
        indent=2 means 2 spaces per nesting level — readable by both humans and LLMs
        """
        return json.dumps(self.to_dict(), indent=2)


    def add_fact(self, subtopic: str, fact: str, source: str = "LLM reasoning") -> None:
        """
        Safely add a fact — prevents duplicates.

        WHY check for duplicates:
          The researcher agent runs once per subtopic.
          Without dedup, the same fact could appear 3 times
          if multiple subtopics overlap.

        any(...) → returns True if ANY element in the iterable is True
        f["fact"] == fact → checks if this exact fact text already exists
        """
        if not any(f["fact"] == fact for f in self.facts):
            self.facts.append({
                "subtopic": subtopic,
                "fact": fact,
                "source": source,
            })


    def get_summary(self) -> str:
        """
        Short status string printed to terminal during execution.
        Shows progress without dumping the whole memory object.
        """
        return (
            f"Topic: {self.topic}\n"
            f"Status: {self.status}\n"
            f"Subtopics: {len(self.subtopics)}\n"
            f"Facts found: {len(self.facts)}\n"
            f"Facts verified: {len(self.verified_facts)}\n"
            f"Conflicts: {len(self.conflicts)}\n"
            f"Report ready: {'Yes' if self.report else 'No'}"
        )
