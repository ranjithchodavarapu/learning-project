"""
pipeline.py — Chains all agents in sequence.

WHAT THE PIPELINE DOES:
  Runs each agent in order:
  Orchestrator → Researcher → Verifier → Summarizer

  Each agent receives the shared memory object,
  does its job, and returns the updated memory.
  The next agent reads what the previous one wrote.

LINE BY LINE:
  agents: List[BaseAgent]
                       → typed list — only BaseAgent subclasses allowed
                         Python doesn't enforce this at runtime but IDEs do
                         helps catch bugs: "you passed a string, not an agent"

  self.agents = [Orchestrator(), Researcher(), Verifier(), Summarizer()]
                       → creates one instance of each agent
                         order matters — Orchestrator must run before Researcher
                         because Researcher reads memory.subtopics which Orchestrator writes

  for agent in self.agents:
                       → loops through agents in order
                         Python lists preserve insertion order (guaranteed)

  console.rule(...)    → prints a horizontal divider line in the terminal
                         [bold] makes the agent name bold
                         helps visually separate each agent's output

  memory = agent.run(memory)
                       → run returns the updated memory
                         reassign to memory so next agent gets the latest state
                         if we didn't reassign: next agent would get the OLD memory

  time.time()          → returns current time as float (seconds since Unix epoch)
  elapsed = end - start → time taken for this agent
  f"{elapsed:.1f}s"    → format with 1 decimal place: "12.3s"

  try / except         → if one agent crashes, log the error and continue
                         we don't want one bad agent to stop the whole pipeline
                         the report might be incomplete but it's better than nothing

  console.print(memory.get_summary())
                       → get_summary() is defined in memory.py
                         prints a quick status after every agent
                         shows facts found, verified etc. in real time
"""

import time
from rich.console import Console
from rich.panel import Panel
from memory import ResearchMemory
from agents.orchestrator import Orchestrator
from agents.researcher import Researcher
from agents.verifier import Verifier
from agents.summarizer import Summarizer

console = Console()


class ResearchPipeline:
    """
    Runs all agents in sequence on the shared memory object.
    This is the core of the multi-agent system.
    """

    def __init__(self):
        # Create agents in execution order
        # Order MUST be: Orchestrator → Researcher → Verifier → Summarizer
        self.agents = [
            Orchestrator(),
            Researcher(),
            Verifier(),
            Summarizer(),
        ]


    def run(self, topic: str) -> ResearchMemory:
        """
        Run the full research pipeline on a topic.

        Args:
            topic: The research topic entered by the user.

        Returns:
            ResearchMemory with all results — subtopics, facts, report.
        """
        # Create fresh memory for this research session
        memory = ResearchMemory(topic=topic)

        console.print(Panel(
            f"[bold blue]Multi-Agent Research System[/bold blue]\n"
            f"Topic: [cyan]{topic}[/cyan]",
            border_style="blue",
        ))

        # Run each agent in sequence
        for agent in self.agents:
            # Print divider showing which agent is running
            console.rule(f"[bold]{agent.name}[/bold]")

            start = time.time()

            try:
                # Pass memory in, get updated memory back
                memory = agent.run(memory)

            except Exception as e:
                # Agent crashed — log error but continue to next agent
                error_msg = f"{agent.name} crashed: {e}"
                console.print(f"[red]{error_msg}[/red]")
                memory.errors.append(error_msg)

            elapsed = time.time() - start
            console.print(f"[dim]{agent.name} finished in {elapsed:.1f}s[/dim]")

            # Print memory summary after each agent
            console.print(f"[dim]{memory.get_summary()}[/dim]")
            console.print()

        return memory
