"""
llm.py — Centralised LLM calling for the self-correcting agent.

Same pattern as project 4 but simpler — no code cleaning needed here
because we generate natural language answers, not Python code.

LINE BY LINE:
  def call_llm(system, user, temperature)
    system       → agent's role/instructions
    user         → the actual task
    temperature  → randomness control
                   generator uses 0.7 (some creativity for answers)
                   evaluator uses 0.1 (strict, consistent scoring)
                   reflector uses 0.3 (structured critique)

  def call_llm_json(system, user)
    → same as call_llm but instructs LLM to return JSON
    → used by evaluator which needs structured scores
    → adds "respond with ONLY valid JSON" to system prompt
    → returns raw string (caller parses with json.loads)
    → temperature=0.1 for consistent JSON structure

  WHY SEPARATE JSON FUNCTION:
    The evaluator needs machine-parseable scores like:
    {"accuracy": 0.8, "clarity": 0.9, "overall": 0.85}
    If we use the regular function, the LLM might return:
    "The accuracy is good, about 0.8, and clarity is 0.9..."
    JSON function forces structured output every time.
"""

import requests
from rich.console import Console
from config import cfg

console = Console()


def call_llm(system: str, user: str, temperature: float = 0.5) -> str:
    """
    Call Ollama and return plain text response.
    Returns empty string on any error.
    """
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=cfg.TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        console.print("[red]Cannot connect to Ollama. Run: ollama serve &[/red]")
        return ""
    except requests.exceptions.Timeout:
        console.print(f"[red]LLM timed out after {cfg.TIMEOUT}s[/red]")
        return ""
    except Exception as e:
        console.print(f"[red]LLM error: {e}[/red]")
        return ""


def call_llm_json(system: str, user: str) -> str:
    """
    Call Ollama expecting a JSON response.

    Appends JSON instruction to system prompt.
    Uses low temperature for consistent structure.
    Returns raw string — caller must parse with json.loads().
    """
    json_system = system + "\nRespond with ONLY valid JSON. No explanation, no markdown."

    return call_llm(json_system, user, temperature=0.1)
