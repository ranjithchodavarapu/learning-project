"""
llm.py — The only file that talks to Ollama.

LINE BY LINE:
  json_mode=True -> payload["format"] = "json"
      Ollama then constrains the model to emit valid JSON. We still parse
      defensively (textutil.parse_json_object) because "valid JSON" does not
      mean "the right fields".

  "temperature": 0 for JSON mode, 0.2 for prose
      Extraction must be repeatable: the same document should give the same
      fields. A little variety is fine for a written summary.

  Every failure returns ""
      The caller treats "" as "the model gave nothing" and handles it; an LLM
      outage never crashes the batch or leaks a stack trace.

  check_ollama()
      A pre-flight test the CLI runs once at start-up. Without it, a stopped
      Ollama makes EVERY document fail with a confusing "no output" message.

  The pipeline receives call_llm as a PARAMETER (dependency injection). Tests
  pass a scripted fake, so the whole pipeline is tested without Ollama.
"""

import requests

from config import cfg


def call_llm(system: str, user: str, json_mode: bool = False) -> str:
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0 if json_mode else 0.2, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=cfg.TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return ""
    except requests.exceptions.Timeout:
        return ""
    except Exception:
        return ""


def check_ollama() -> bool:
    try:
        return requests.get(cfg.OLLAMA_BASE_URL, timeout=3).status_code == 200
    except Exception:
        return False
