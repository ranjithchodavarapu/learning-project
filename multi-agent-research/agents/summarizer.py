"""
agents/summarizer.py — Writes the final research report.

WHAT THE SUMMARIZER DOES:
  1. Reads all verified facts from shared memory
  2. Reads the subtopics for structure
  3. Asks the LLM to write a coherent research report
  4. Saves the report to memory.report
  5. Saves report to disk as a .txt file

WHY THE SUMMARIZER IS LAST:
  It has access to the full verified picture:
  - What subtopics were covered (from orchestrator)
  - What facts were found (from researcher)
  - Which facts are confirmed and which are contested (from verifier)
  - What conflicts exist (from verifier)

  A good summary can only be written AFTER all this is known.

LINE BY LINE:
  SYSTEM_PROMPT        → gives the LLM a "science journalist" persona
                         personas help — "write like a journalist" produces
                         better structured prose than "write a report"

  _build_context()     → assembles all memory into one big context string
                         this string is passed to the LLM as the user prompt
                         the LLM reads this and writes the report from it

  "## Verified Facts by Subtopic"
                       → groups facts by their subtopic for clarity
                         facts from "History of AI" are grouped together
                         makes the report structure logical

  groupby_subtopic     → dict where key = subtopic, value = list of facts
                         built with a loop rather than itertools.groupby
                         because groupby needs sorted data — dict is simpler here

  conflicts_section    → if verifier found conflicts, include them in context
                         the LLM can then note uncertainty in the report

  temperature=0.5      → higher than other agents — writing needs some creativity
                         0.0 = robotic, 0.7 = too creative/hallucination risk
                         0.5 = good balance for coherent prose

  _save_report()       → writes report to disk
                         Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
                         parents=True: creates parent dirs if needed
                         exist_ok=True: doesn't fail if dir already exists

  timestamp            → adds date+time to filename so reports don't overwrite
                         "research_2024-01-15_14-30-22.txt"
                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                         strftime = string format time (standard Python date formatting)
"""

from agents.base_agent import BaseAgent
from memory import ResearchMemory
from config import cfg
from pathlib import Path
from datetime import datetime


SYSTEM_PROMPT = """You are a science journalist writing a research report.
Given verified facts organized by subtopic, write a clear, well-structured report.
The report should:
  - Have a title and introduction
  - Cover each subtopic in its own section with a heading
  - Use the facts provided — do not add facts not in the list
  - Note any conflicts or uncertainties where they exist
  - End with a concise conclusion
Write in clear, professional prose — not bullet points."""


class Summarizer(BaseAgent):
    """
    Writes the final research report from verified facts.
    Runs last in the pipeline.
    """

    def __init__(self):
        super().__init__("Summarizer")


    def run(self, memory: ResearchMemory) -> ResearchMemory:
        """
        Write and save the final research report.
        """
        self._log("Writing final research report")
        memory.status = "summarizing"

        if not memory.verified_facts:
            self._log("No verified facts — using all facts as fallback")
            # If verifier found nothing, use raw facts anyway
            facts_to_use = memory.facts
        else:
            facts_to_use = memory.verified_facts

        # Build the full context string for the LLM
        context = self._build_context(memory, facts_to_use)

        # Generate the report
        report = self._call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=context,
            temperature=0.5,   # some creativity for good writing
        )

        if not report:
            # Fallback: build a basic report from facts without LLM
            report = self._build_fallback_report(memory, facts_to_use)
            memory.errors.append("Summarizer: LLM failed, built fallback report")

        memory.report = report
        memory.status = "done"

        # Save to disk
        filepath = self._save_report(memory)
        self._log(f"Report saved to: {filepath}")
        self._log(f"Report length: {len(report)} characters")

        return memory


    def _build_context(self, memory: ResearchMemory, facts: list) -> str:
        """
        Assemble all memory into a structured context string for the LLM.

        The LLM reads this and writes the report from it.
        Structure: topic → subtopics → facts grouped by subtopic → conflicts
        """
        lines = [
            f"# Research Topic: {memory.topic}",
            "",
            f"## Subtopics Covered",
        ]

        for i, sub in enumerate(memory.subtopics, 1):
            lines.append(f"{i}. {sub}")

        lines.append("")
        lines.append("## Verified Facts by Subtopic")
        lines.append("")

        # Group facts by their subtopic
        # groupby_subtopic: {"History of AI": [fact1, fact2], "Current state": [...]}
        groupby_subtopic: dict = {}
        for fact in facts:
            sub = fact.get("subtopic", "General")
            if sub not in groupby_subtopic:
                groupby_subtopic[sub] = []
            groupby_subtopic[sub].append(fact["fact"])

        # Add each subtopic's facts as a numbered list
        for subtopic, subtopic_facts in groupby_subtopic.items():
            lines.append(f"### {subtopic}")
            for i, fact in enumerate(subtopic_facts, 1):
                lines.append(f"{i}. {fact}")
            lines.append("")

        # Include conflicts if any were found
        if memory.conflicts:
            lines.append("## Conflicts and Uncertainties Found")
            for conflict in memory.conflicts:
                lines.append(f"- {conflict}")
            lines.append("")

        lines.append("---")
        lines.append("Write a complete research report based on the above.")

        # "\n".join(lines) → joins all lines with newline character
        return "\n".join(lines)


    def _build_fallback_report(self, memory: ResearchMemory, facts: list) -> str:
        """
        Build a basic report without LLM — used when LLM call fails.
        Formats facts into sections using plain string concatenation.
        """
        lines = [
            f"# Research Report: {memory.topic}",
            "",
            "## Overview",
            f"This report covers {len(memory.subtopics)} subtopics "
            f"with {len(facts)} verified facts.",
            "",
        ]

        # Group by subtopic
        groupby: dict = {}
        for fact in facts:
            sub = fact.get("subtopic", "General")
            if sub not in groupby:
                groupby[sub] = []
            groupby[sub].append(fact["fact"])

        for subtopic, subtopic_facts in groupby.items():
            lines.append(f"## {subtopic}")
            for fact in subtopic_facts:
                lines.append(f"- {fact}")
            lines.append("")

        return "\n".join(lines)


    def _save_report(self, memory: ResearchMemory) -> str:
        """
        Save the report to a text file with a timestamped filename.

        Returns the filepath string so we can log it.
        """
        # Create output directory if it doesn't exist
        # parents=True: creates intermediate dirs too (like mkdir -p)
        # exist_ok=True: no error if dir already exists
        output_dir = Path(cfg.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build filename: "research_AI_in_medicine_2024-01-15_14-30-22.txt"
        # topic[:30] → take first 30 chars of topic (avoid very long filenames)
        # .replace(" ", "_") → spaces → underscores (safe for filesystem)
        safe_topic = memory.topic[:30].replace(" ", "_").replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"research_{safe_topic}_{timestamp}.txt"

        filepath = output_dir / filename

        # Write the report text to the file
        # encoding="utf-8" → explicit encoding avoids issues on Windows
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(memory.report)

        return str(filepath)
