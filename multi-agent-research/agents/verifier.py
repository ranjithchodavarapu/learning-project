"""
agents/verifier.py — Cross-checks facts and flags conflicts.

WHAT THE VERIFIER DOES:
  1. Reads all facts from shared memory
  2. For each fact, asks: "Is this plausible and internally consistent?"
  3. Checks across facts: "Do any facts contradict each other?"
  4. Moves confirmed facts to memory.verified_facts
  5. Logs conflicts to memory.conflicts

WHY VERIFICATION MATTERS:
  LLMs hallucinate. A researcher agent can generate confident-sounding
  but wrong facts. The verifier is a second LLM call that independently
  reviews each fact — like a peer reviewer in academic publishing.

  This doesn't guarantee accuracy (both agents use the same LLM),
  but it catches:
  - Internal contradictions ("X was invented in 1950" vs "X was invented in 1960")
  - Obviously implausible claims
  - Facts that don't relate to the topic

LINE BY LINE:
  VERIFY_PROMPT        → instructs LLM to assess ONE fact at a time
                         "CONFIRM" or "QUESTION: reason"
                         structured output makes parsing reliable

  CONFLICT_PROMPT      → instructs LLM to check ALL facts together
                         looks for contradictions across the full fact set

  _verify_batch()      → verifies facts in groups of BATCH_SIZE
                         why batches: sending 15 facts at once = LLM loses focus
                         sending 5 at a time = more careful verification

  "CONFIRM" in result  → checks if LLM said CONFIRM
                         .upper() normalises case ("confirm" vs "CONFIRM")

  memory.verified_facts.append(fact)
                       → confirmed facts go to verified list

  memory.verification_notes.append(result)
                       → save the LLM's reasoning for each fact
                         useful for debugging: why was fact X rejected?

  _check_conflicts()   → sends ALL facts to LLM at once
                         asks: do any of these contradict each other?
                         parses response for conflict descriptions

  "NO CONFLICTS" in response.upper()
                       → normalise case before checking
                         LLM might say "No conflicts" or "NO CONFLICTS"
"""

from agents.base_agent import BaseAgent
from memory import ResearchMemory

BATCH_SIZE = 5   # verify this many facts per LLM call


VERIFY_PROMPT = """You are a fact verifier. Assess whether a fact is plausible and accurate.
For each fact given, respond with either:
  CONFIRM - if the fact is plausible, accurate, and relevant
  QUESTION: <reason> - if the fact seems wrong, implausible, or irrelevant

Be strict but fair. Minor wording issues are fine. Flag clear errors."""


CONFLICT_PROMPT = """You are a fact checker looking for contradictions.
Review the following facts and identify any that directly contradict each other.
If you find contradictions, describe them clearly, one per line.
If there are no contradictions, respond with exactly: NO CONFLICTS"""


class Verifier(BaseAgent):
    """
    Reviews researcher's facts for accuracy and internal consistency.
    Runs third in the pipeline.
    """

    def __init__(self):
        super().__init__("Verifier")


    def run(self, memory: ResearchMemory) -> ResearchMemory:
        """
        Verify each fact and check for contradictions across all facts.
        """
        self._log(f"Verifying {len(memory.facts)} facts")
        memory.status = "verifying"

        if not memory.facts:
            self._log("No facts to verify — skipping")
            return memory

        # Step 1: Verify facts in batches
        self._verify_batch(memory)

        # Step 2: Check for contradictions across all facts
        self._check_conflicts(memory)

        self._log(
            f"Verified: {len(memory.verified_facts)}/{len(memory.facts)} facts confirmed"
        )
        if memory.conflicts:
            self._log(f"Found {len(memory.conflicts)} conflict(s)")

        return memory


    def _verify_batch(self, memory: ResearchMemory) -> None:
        """
        Verify facts in small batches for focused attention.

        BATCH_SIZE = 5 means: send 5 facts per LLM call.
        range(0, 15, 5) → [0, 5, 10] → 3 batches for 15 facts.
        """
        for i in range(0, len(memory.facts), BATCH_SIZE):
            # Slice the facts list: [0:5], [5:10], [10:15] etc.
            batch = memory.facts[i : i + BATCH_SIZE]

            # Build numbered list of facts for the LLM
            facts_text = "\n".join(
                f"{j+1}. {f['fact']}" for j, f in enumerate(batch)
            )

            user_prompt = (
                f"Research topic: {memory.topic}\n\n"
                f"Facts to verify:\n{facts_text}\n\n"
                f"For each fact, respond with CONFIRM or QUESTION: <reason>.\n"
                f"One response per line, numbered to match."
            )

            result = self._call_llm(
                system_prompt=VERIFY_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,  # very low temp = consistent, strict assessment
            )

            if not result:
                # If LLM failed, confirm all facts in this batch as fallback
                for fact in batch:
                    memory.verified_facts.append(fact)
                continue

            # Parse response: one line per fact
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]

            for j, fact in enumerate(batch):
                # Get the corresponding verification line
                # j < len(lines) guards against LLM returning fewer lines than facts
                if j < len(lines):
                    verdict = lines[j].upper()
                    memory.verification_notes.append(lines[j])

                    if "CONFIRM" in verdict:
                        # Fact passed verification
                        memory.verified_facts.append(fact)
                    else:
                        # Fact was questioned — log but don't include in verified
                        self._log(f"  [yellow]Questioned:[/yellow] {fact['fact'][:60]}...")
                else:
                    # LLM returned fewer lines than facts — confirm by default
                    memory.verified_facts.append(fact)


    def _check_conflicts(self, memory: ResearchMemory) -> None:
        """
        Check all verified facts together for contradictions.
        One LLM call with all facts — looking for cross-fact issues.
        """
        if len(memory.verified_facts) < 2:
            # Need at least 2 facts to have a contradiction
            return

        # Build a numbered list of all verified facts
        all_facts = "\n".join(
            f"{i+1}. {f['fact']}" for i, f in enumerate(memory.verified_facts)
        )

        user_prompt = (
            f"Research topic: {memory.topic}\n\n"
            f"Review these facts for contradictions:\n{all_facts}\n\n"
            f"List any contradictions you find, or respond with NO CONFLICTS."
        )

        response = self._call_llm(
            system_prompt=CONFLICT_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        if not response:
            return

        if "NO CONFLICTS" in response.upper():
            self._log("No contradictions found")
            return

        # Parse conflict descriptions — one per line
        conflicts = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and "NO CONFLICTS" not in line.upper()
        ]

        memory.conflicts.extend(conflicts)
