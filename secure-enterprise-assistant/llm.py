"""
llm.py — One function that calls Ollama. Returns "" on any failure.

LINE BY LINE:
  "temperature": 0.1
      Low randomness. A security assistant should answer the same way each
      time, and low temperature makes the model follow the "answer only from
      the documents" rule more faithfully.

  "num_ctx": 8192
      Context window in tokens. Enough for 3 chunks plus the rules.

  Three except blocks, each returning "":
      The assistant checks `if not raw` and shows a safe message. An LLM
      outage must never crash the request or leak a stack trace to the user.

  The assistant takes this function as a PARAMETER (dependency injection).
  Tests pass in a fake function instead, so the whole security pipeline is
  tested without Ollama running and with fully predictable "LLM" output.
"""

import requests

from config import cfg


def call_llm(system: str, user: str, temperature: float = 0.1) -> str:
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
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
        return ""
    except requests.exceptions.Timeout:
        return ""
    except Exception:
        return ""
