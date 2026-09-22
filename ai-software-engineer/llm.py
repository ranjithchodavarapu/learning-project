"""
llm.py — Single place for all LLM calls in the project.

WHY ONE LLM MODULE:
  Projects 2 and 3 had each agent/file calling Ollama with
  its own copy of the HTTP POST code. That works but:
  - If Ollama's API changes, you fix it in 4 places
  - Harder to add retry logic, logging, or swap models

  Here we centralise all LLM calls into two functions:
  call_llm()      → plain text response
  call_llm_code() → strips markdown fences (same as project 3)

LINE BY LINE:
  import requests        → HTTP library for calling Ollama
  from config import cfg → shared config (model name, URL, timeout)

  def call_llm(system, user, temperature)
    system             → the agent's role/instructions
    user               → the actual task/question
    temperature        → how random the output is
                         0.0 = deterministic (code, planning)
                         0.5 = creative (architecture descriptions)

    payload            → dict sent as JSON to Ollama
    "stream": False    → wait for full response, not token-by-token
    "num_ctx": 8192    → context window size in tokens
                         8192 = 8k tokens ≈ ~6000 words

    resp.raise_for_status()
                       → raises HTTPError if status is 4xx/5xx
                         without this, failed requests return silently
    resp.json()["message"]["content"]
                       → Ollama response structure:
                         {"message": {"content": "..."}}

    except ConnectionError → Ollama not running → return ""
    except Timeout         → took too long → return ""
    except Exception       → anything else → return ""
    All return "" so callers check: if not response: handle_error()

  def call_llm_code(system, user, temperature)
    → calls call_llm() then strips markdown fences
    → same _clean_code() logic as project 3
    → for fence in ["```python", "```js", "```"]:
         if code.startswith(fence): strip it
    → if code.endswith("```"): strip closing fence
"""

import requests
from rich.console import Console
from config import cfg

console = Console()


def call_llm(
    system: str,
    user: str,
    temperature: float = 0.2,
) -> str:
    """
    Call Ollama and return plain text response.

    Args:
        system:      System prompt — agent role and rules.
        user:        User prompt — the actual task.
        temperature: 0.0 = deterministic, 1.0 = very random.

    Returns:
        Response text as string. Empty string on any error.
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


def call_llm_code(
    system: str,
    user: str,
    temperature: float = 0.1,
) -> str:
    """
    Call Ollama for code generation — strips markdown fences automatically.

    Uses lower default temperature (0.1) because code needs precision.
    Strips ```python, ```js, ``` fences that LLMs often add.

    Args:
        system:      System prompt.
        user:        User prompt.
        temperature: Lower = more precise code.

    Returns:
        Raw executable code as string (no markdown).
    """
    raw = call_llm(system, user, temperature)
    if not raw:
        return ""
    return _clean_code(raw)


def _clean_code(code: str) -> str:
    """
    Strip markdown code fences from LLM output.

    LLMs often wrap code in ```python...``` even when told not to.
    This strips all variants of opening and closing fences.

    Input:  "```python\\nimport os\\n```"
    Output: "import os"
    """
    code = code.strip()

    # Remove opening fence (try most specific first)
    for fence in ["```python", "```javascript", "```typescript",
                  "```bash", "```sh", "```js", "```ts", "```"]:
        if code.startswith(fence):
            code = code[len(fence):]
            break

    # Remove closing fence
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()
