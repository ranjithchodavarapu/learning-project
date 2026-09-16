"""
query_rewriter.py — HyDE + multi-query expansion.
Supports ollama (local), anthropic, openai.
"""

from typing import List
from rich.console import Console

from src.config import cfg

console = Console()

HYDE_PROMPT = """Write a short passage (3-5 sentences) that directly answers the following question.
Write as if you are an expert and this passage would appear in a technical document.
Do NOT say "I don't know" — always write a plausible hypothetical answer.

Question: {query}

Passage:"""

MULTI_QUERY_PROMPT = """Generate {n} different phrasings of the following question.
Each phrasing should approach the topic from a slightly different angle.
Output ONLY the questions, one per line, no numbering or bullets.

Original question: {query}

Phrasings:"""


def _call_llm(prompt: str) -> str:
    if cfg.LLM_PROVIDER == "ollama":
        import requests
        payload = {
            "model": cfg.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        }
        try:
            resp = requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running. Run: ollama serve &")

    elif cfg.LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    elif cfg.LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return resp.content[0].text.strip()

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {cfg.LLM_PROVIDER}")


def generate_hyde_document(query: str) -> str:
    console.print("[dim]HyDE: generating hypothetical document...[/dim]")
    return _call_llm(HYDE_PROMPT.format(query=query))


def generate_multi_queries(query: str, n: int = 3) -> List[str]:
    console.print(f"[dim]Multi-query: generating {n} paraphrases...[/dim]")
    raw = _call_llm(MULTI_QUERY_PROMPT.format(query=query, n=n))
    queries = [q.strip() for q in raw.strip().split("\n") if q.strip()][:n]
    all_queries = [query] + queries
    console.print(f"[green]✓ {len(all_queries)} queries total[/green]")
    return all_queries


def rewrite_query(query: str, strategy: str = "both") -> dict:
    result = {"original": query}
    if strategy in ("hyde", "both"):
        result["hyde_doc"] = generate_hyde_document(query)
    if strategy in ("multi", "both"):
        result["multi_queries"] = generate_multi_queries(query, n=3)
    return result
