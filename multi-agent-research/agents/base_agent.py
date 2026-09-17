"""
agents/base_agent.py — Base class that all agents inherit from.

WHY A BASE CLASS:
  All 4 agents (orchestrator, researcher, verifier, summarizer)
  need to call Ollama. Without a base class, you'd copy-paste
  the same _call_llm() function into every agent file.

  Instead:
    class BaseAgent:         ← has _call_llm(), shared logic
    class Researcher(BaseAgent): ← inherits _call_llm() for free
    class Verifier(BaseAgent):   ← same
    class Summarizer(BaseAgent): ← same

  This is the OOP principle: Don't Repeat Yourself (DRY)

LINE BY LINE:
  ABC, abstractmethod  → ABC = Abstract Base Class
                         @abstractmethod = subclasses MUST implement this method
                         if they don't, Python raises an error at import time

  self.name            → each agent has a name ("Researcher", "Verifier" etc.)
                         used in log messages so you can see which agent is speaking

  self.model           → which Ollama model to use (from config)
  self.base_url        → where Ollama is running (localhost:11434)
  self.timeout         → max seconds to wait for LLM response

  console = Console()  → rich library pretty terminal output
                         [blue]text[/blue] → prints in blue
                         [green]text[/green] → prints in green

  def _call_llm(...)   → the _ prefix means "private method"
                         convention: don't call this from outside the class
                         subclasses call it via self._call_llm(...)

  requests.post(...)   → sends HTTP POST to Ollama API
                         Ollama runs as a local web server
                         we communicate with it via HTTP requests

  payload              → the data we send to Ollama
    "model"            → which model (llama3.1:70b)
    "messages"         → list of {role, content} dicts
                         role = "system" (instructions) or "user" (the question)
    "stream": False    → wait for complete response (vs streaming token by token)
    "options"          → model parameters
      "temperature"    → 0 = deterministic, 0.7 = creative
      "num_ctx"        → context window size in tokens

  resp.raise_for_status() → raises exception if HTTP status is 4xx or 5xx
                            without this, a failed request returns silently

  resp.json()["message"]["content"]
                       → Ollama response structure:
                         {"message": {"role": "assistant", "content": "..."}}
                         we extract just the content string

  ConnectionError      → Ollama server isn't running
  Timeout              → took longer than self.timeout seconds
  Exception as e       → catch-all for any other error
    self.memory.errors.append(...)
                       → log error to shared memory so orchestrator can see it

  @abstractmethod
  def run(self, memory)→ every agent MUST implement run()
                         this is the main method the pipeline calls
                         each agent does different things in run()
"""

from abc import ABC, abstractmethod
import requests
from rich.console import Console
from memory import ResearchMemory
from config import cfg

console = Console()


class BaseAgent(ABC):
    """
    Abstract base class for all research agents.
    Provides shared LLM calling logic and logging.
    All agents inherit from this — never instantiate directly.
    """

    def __init__(self, name: str):
        # name identifies which agent is speaking in logs
        self.name = name
        self.model = cfg.OLLAMA_MODEL
        self.base_url = cfg.OLLAMA_BASE_URL
        self.timeout = cfg.TIMEOUT

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        Call Ollama and return the response text.

        Args:
            system_prompt: Instructions for the agent (its role and rules)
            user_prompt:   The actual task/question for this call
            temperature:   0.0 = focused/deterministic, 1.0 = creative/varied
                           Researchers use 0.3 (factual but slightly flexible)
                           Summarizers use 0.5 (needs some creative writing)

        Returns:
            The LLM's response as a plain string.
            Returns empty string on error (logged to memory).
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,   # 8k token context window
            },
        }

        console.print(f"[dim]{self.name} → calling {self.model}...[/dim]")

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

        except requests.exceptions.ConnectionError:
            msg = f"{self.name}: Cannot connect to Ollama. Run: ollama serve &"
            console.print(f"[red]{msg}[/red]")
            return ""

        except requests.exceptions.Timeout:
            msg = f"{self.name}: LLM call timed out after {self.timeout}s"
            console.print(f"[red]{msg}[/red]")
            return ""

        except Exception as e:
            msg = f"{self.name}: Unexpected error: {e}"
            console.print(f"[red]{msg}[/red]")
            return ""

    def _log(self, message: str) -> None:
        """
        Print a coloured log line showing which agent is speaking.
        [bold cyan]{self.name}[/bold cyan] → agent name in bold cyan
        """
        console.print(f"[bold cyan]{self.name}[/bold cyan]: {message}")

    @abstractmethod
    def run(self, memory: ResearchMemory) -> ResearchMemory:
        """
        Every agent must implement this method.
        Receives shared memory, does its job, returns updated memory.

        This is the contract: take memory in → process → return memory out.
        The pipeline calls agent.run(memory) for each agent in sequence.
        """
        pass
