"""
generator.py — Supports ollama (local), anthropic, openai.
Switch provider by changing LLM_PROVIDER in .env
"""

from typing import List
from rich.console import Console

from src.config import cfg
from src.retrieval.retriever import RetrievedChunk

console = Console()

SYSTEM_PROMPT = """You are a precise and helpful knowledge assistant.
Answer the user's question using ONLY the context chunks provided below.
If the context does not contain enough information to answer, say:
"I don't have enough information in the provided documents to answer this."
Do not use any knowledge outside the provided context."""


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Chunk {i} | Source: {chunk.source}, Page: {chunk.page}, Score: {chunk.score}]\n"
            f"{chunk.text}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)
    return (
        f"CONTEXT:\n\n{context_str}\n\n---\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


def _call_ollama(system: str, user: str) -> str:
    import requests
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
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
        raise RuntimeError(
            "\nCannot connect to Ollama. Start it first:\n"
            "  ollama serve &\n"
            "Then verify: curl http://localhost:11434"
        )


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=cfg.LLM_MODEL,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0,
        max_tokens=1024,
    )
    return response.content[0].text.strip()


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context was retrieved. Please try a different question."

    user_message = build_prompt(query, chunks)
    console.print(f"[blue]Calling {cfg.LLM_PROVIDER} "
                  f"({'ollama: ' + cfg.OLLAMA_MODEL if cfg.LLM_PROVIDER == 'ollama' else cfg.LLM_MODEL})...[/blue]")

    if cfg.LLM_PROVIDER == "ollama":
        return _call_ollama(SYSTEM_PROMPT, user_message)
    elif cfg.LLM_PROVIDER == "openai":
        return _call_openai(SYSTEM_PROMPT, user_message)
    elif cfg.LLM_PROVIDER == "anthropic":
        return _call_anthropic(SYSTEM_PROMPT, user_message)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{cfg.LLM_PROVIDER}'\n"
            "Valid options: ollama | openai | anthropic"
        )
