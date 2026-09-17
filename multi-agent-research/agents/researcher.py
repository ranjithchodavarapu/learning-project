"""
agents/researcher.py — Extracts facts for each subtopic.

WHAT THE RESEARCHER DOES:
  1. Reads the subtopics from shared memory
  2. For each subtopic, asks the LLM to generate key facts
  3. Parses the LLM's response into structured fact objects
  4. Saves all facts to shared memory for the verifier

ANALOGY: Like a PhD student assigned a chapter.
  Given "write about X", they research and bring back bullet-point facts.

KEY DESIGN DECISIONS:
  - Loops over subtopics, one LLM call per subtopic
    (better than one big call — more focused, less hallucination)
  - Facts are structured dicts: {subtopic, fact, source}
    (structured data is easier to verify and display)
  - Uses memory.add_fact() which deduplicates automatically

LINE BY LINE:
  for subtopic in memory.subtopics:
                       → iterates over the list orchestrator wrote
                         one LLM call per subtopic

  user_prompt includes the subtopic AND what topic it belongs to
                       → giving context reduces off-topic responses

  "List exactly {N} key facts"
                       → being specific about count = more consistent output
                         "list some facts" = unpredictable number

  _parse_facts()       → same pattern as _parse_subtopics in orchestrator
                         LLM returns numbered list → we parse to Python list

  memory.add_fact(subtopic, fact)
                       → add_fact() is defined in memory.py
                         checks for duplicates before appending

  memory.status = "researching"
                       → updates pipeline status in shared memory
"""

from agents.base_agent import BaseAgent
from memory import ResearchMemory
from config import cfg


SYSTEM_PROMPT = """You are a research specialist. Your job is to find key facts.
Given a subtopic, provide specific, accurate, and informative facts about it.
Each fact should be a complete sentence with concrete information.
Respond with a numbered list of facts only — no introductions or conclusions."""


class Researcher(BaseAgent):
    """
    Finds key facts for each subtopic planned by the orchestrator.
    Runs second in the pipeline.
    """

    def __init__(self):
        super().__init__("Researcher")


    def run(self, memory: ResearchMemory) -> ResearchMemory:
        """
        For each subtopic, extract key facts and save to memory.
        """
        self._log(f"Researching {len(memory.subtopics)} subtopics")
        memory.status = "researching"

        # Loop over each subtopic the orchestrator planned
        for i, subtopic in enumerate(memory.subtopics, 1):
            self._log(f"  [{i}/{len(memory.subtopics)}] Researching: {subtopic}")

            user_prompt = (
                f"Main research topic: {memory.topic}\n"
                f"Subtopic to research: {subtopic}\n\n"
                f"List exactly {cfg.FACTS_PER_SUBTOPIC} key facts about this subtopic.\n"
                f"Each fact must be a complete, specific sentence.\n"
                f"Number them 1 to {cfg.FACTS_PER_SUBTOPIC}."
            )

            response = self._call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,   # low temp = factual, not creative
            )

            if not response:
                memory.errors.append(f"Researcher: Failed to research '{subtopic}'")
                continue   # skip this subtopic, move to next

            # Parse the numbered list into individual fact strings
            facts = self._parse_facts(response)

            # Add each fact to shared memory (with deduplication)
            for fact in facts:
                memory.add_fact(
                    subtopic=subtopic,
                    fact=fact,
                    source=f"LLM research on: {subtopic}",
                )

            self._log(f"    Found {len(facts)} facts")

        self._log(f"Total facts collected: {len(memory.facts)}")
        return memory


    def _parse_facts(self, response: str) -> list:
        """
        Parse LLM numbered list into a Python list of fact strings.

        Input:  "1. AI was coined in 1956.\n2. Deep learning uses neural nets.\n"
        Output: ["AI was coined in 1956.", "Deep learning uses neural nets."]
        """
        facts = []

        for line in response.strip().split("\n"):
            line = line.strip()

            if not line:
                continue   # skip blank lines

            if line[0].isdigit():
                # "1. The fact text here" → split on first dot
                parts = line.split(".", 1)
                if len(parts) == 2:
                    fact = parts[-1].strip()
                    if fact:
                        facts.append(fact)
            elif line.startswith("-") or line.startswith("•"):
                # Handle bullet point format as fallback
                # "- The fact text" → remove the bullet
                fact = line.lstrip("-•").strip()
                if fact:
                    facts.append(fact)

        # If parsing failed, try using raw lines
        if not facts:
            facts = [l.strip() for l in response.split("\n") if l.strip()]

        return facts[:cfg.FACTS_PER_SUBTOPIC]
