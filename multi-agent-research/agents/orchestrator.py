"""
agents/orchestrator.py — Plans the research and breaks topic into subtopics.

WHAT THE ORCHESTRATOR DOES:
  1. Receives the research topic from the user
  2. Thinks about how to break it into focused subtopics
  3. Writes a research plan
  4. Saves subtopics to shared memory so the researcher knows what to find

ANALOGY: Like a research supervisor assigning chapters to PhD students.
  The supervisor doesn't do the research themselves — they plan it.

LINE BY LINE:
  class Orchestrator(BaseAgent):
                       → inherits from BaseAgent
                         gets _call_llm() and _log() for free
                         only needs to implement run()

  super().__init__("Orchestrator")
                       → calls BaseAgent.__init__ with name="Orchestrator"
                         sets self.name, self.model, self.base_url, self.timeout

  SYSTEM_PROMPT        → tells the LLM what role it's playing
                         "You are a research orchestrator" = persona
                         "respond with a numbered list" = output format instruction
                         Giving the LLM a clear role + format = better outputs

  memory.status = "planning"
                       → updates shared memory status
                         orchestrator running = "planning"
                         researcher running = "researching" etc.
                         lets you watch progress in real time

  user_prompt          → the actual task sent to the LLM
                         includes the topic AND how many subtopics we want
                         f-string interpolates cfg.MAX_SUBTOPICS (from .env)

  self._call_llm(...)  → inherited from BaseAgent
                         sends system + user prompt to Ollama
                         returns the response string

  _parse_subtopics()   → parses the LLM's numbered list into a Python list
                         "1. History of X\n2. Current state\n3. Future"
                         → ["History of X", "Current state", "Future"]

  line.strip()         → removes leading/trailing whitespace
  if not line: continue → skips blank lines
  line[0].isdigit()    → checks if line starts with a number (1, 2, 3...)
  line.split(".", 1)   → splits "1. History of X" into ["1", " History of X"]
                         the 1 means split at most once (stops at first dot)
  [-1]                 → takes the last element (the text after the number)
  .strip()             → removes the space before "History of X"

  memory.subtopics = subtopics
                       → writes the parsed list to shared memory
                         researcher will read this in its run() method

  memory.research_plan = response
                       → saves the full LLM response as the plan
                         used by summarizer for context later
"""

from agents.base_agent import BaseAgent
from memory import ResearchMemory
from config import cfg


SYSTEM_PROMPT = """You are a research orchestrator. Your job is to plan research.
Given a topic, break it into focused subtopics that together cover the subject completely.
Each subtopic should be specific enough to research individually.
Respond with a numbered list of subtopics only — no other text."""


class Orchestrator(BaseAgent):
    """
    Plans the research by breaking the topic into subtopics.
    Runs first in the pipeline — sets up work for all other agents.
    """

    def __init__(self):
        super().__init__("Orchestrator")


    def run(self, memory: ResearchMemory) -> ResearchMemory:
        """
        Break the topic into subtopics and save to memory.
        Called first by the pipeline.
        """
        self._log(f"Planning research on: '{memory.topic}'")
        memory.status = "planning"

        # Ask the LLM to break the topic into subtopics
        user_prompt = (
            f"Research topic: {memory.topic}\n\n"
            f"Break this into exactly {cfg.MAX_SUBTOPICS} focused subtopics "
            f"that together fully cover the topic.\n"
            f"Number them 1 to {cfg.MAX_SUBTOPICS}."
        )

        response = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,   # low temp = consistent, focused planning
        )

        if not response:
            # If LLM failed, create generic subtopics so pipeline can continue
            memory.subtopics = [
                f"Overview of {memory.topic}",
                f"Key concepts in {memory.topic}",
                f"Applications of {memory.topic}",
            ]
            memory.errors.append("Orchestrator: LLM failed, using default subtopics")
            self._log("LLM failed — using default subtopics")
        else:
            memory.subtopics = self._parse_subtopics(response)
            memory.research_plan = response
            self._log(f"Created {len(memory.subtopics)} subtopics")

        # Print what was planned
        for i, sub in enumerate(memory.subtopics, 1):
            self._log(f"  {i}. {sub}")

        return memory


    def _parse_subtopics(self, response: str) -> list:
        """
        Parse LLM numbered list output into a Python list.

        Input:  "1. History of AI\n2. Current state\n3. Future directions"
        Output: ["History of AI", "Current state", "Future directions"]
        """
        subtopics = []

        for line in response.strip().split("\n"):
            line = line.strip()

            if not line:
                # Skip blank lines
                continue

            if line[0].isdigit():
                # Line starts with a number → it's a subtopic
                # Split on first dot only: "1. History" → ["1", " History"]
                parts = line.split(".", 1)
                if len(parts) == 2:
                    subtopic = parts[-1].strip()  # take text after the number
                    if subtopic:
                        subtopics.append(subtopic)
            else:
                # Line doesn't start with number but isn't blank
                # Could be a subtopic without numbering — include it
                subtopics.append(line)

        # Fallback: if parsing failed entirely, use the raw response lines
        if not subtopics:
            subtopics = [l.strip() for l in response.split("\n") if l.strip()]

        # Respect MAX_SUBTOPICS limit
        return subtopics[:cfg.MAX_SUBTOPICS]
