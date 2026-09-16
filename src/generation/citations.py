"""
citations.py — Structured output with source attribution.
Supports ollama (local), anthropic, openai.
"""

import json
from dataclasses import dataclass, field
from typing import List

from rich.console import Console

from src.config import cfg
from src.retrieval.retriever import RetrievedChunk

console = Console()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Citation:
    chunk_index: int
    source_file: str
    page: int
    excerpt: str
    relevance_score: float


@dataclass
class CitedAnswer:
    answer: str
    citations: List[Citation] = field(default_factory=list)

    def pretty_print(self) -> str:
        lines = [self.answer, ""]
        if self.citations:
            lines.append("─── Sources ───")
            for i, c in enumerate(self.citations, 1):
                lines.append(
                    f"[{i}] {c.source_file} p.{c.page} "
                    f"(score: {c.relevance_score}) — \"{c.excerpt[:80]}...\""
                )
        return "\n".join(lines)


# ── Prompt ────────────────────────────────────────────────────────────────────

CITED_SYSTEM_PROMPT = """You are a precise knowledge assistant that always cites sources.
Answer the user's question using ONLY the provided context chunks.
You MUST respond with valid JSON in this EXACT format, nothing else:
{"answer": "your answer here", "cited_chunks": [1, 2]}

Rules:
- cited_chunks is a list of chunk numbers (1-indexed) you actually used
- Only include chunks you directly drew information from
- If you cannot answer, set answer to "Insufficient context." and cited_chunks to []
- Output ONLY the JSON object, no markdown, no explanation"""


def build_cited_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"[Chunk {i} | {chunk.source} p.{chunk.page} | score: {chunk.score}]\n"
            f"{chunk.text}"
        )
    context = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n\n{context}\n\n---\n\nQUESTION: {query}"


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str, chunks: List[RetrievedChunk]) -> CitedAnswer:
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        console.print(f"[yellow]JSON parse error: {e} — returning raw answer.[/yellow]")
        return CitedAnswer(answer=raw, citations=[])

    answer_text    = data.get("answer", "")
    cited_indices  = data.get("cited_chunks", [])

    citations = []
    for idx in cited_indices:
        i = idx - 1
        if 0 <= i < len(chunks):
            chunk = chunks[i]
            citations.append(Citation(
                chunk_index=i,
                source_file=chunk.source,
                page=chunk.page,
                excerpt=chunk.text[:200],
                relevance_score=chunk.score,
            ))
    return CitedAnswer(answer=answer_text, citations=citations)


# ── LLM callers ───────────────────────────────────────────────────────────────

def _cited_ollama(user_msg: str) -> str:
    import requests
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": CITED_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "format": "json",          # forces valid JSON from Ollama
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama not running. Run: ollama serve &")


def _cited_openai(user_msg: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": CITED_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _cited_anthropic(user_msg: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=cfg.LLM_MODEL,
        system=CITED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0,
        max_tokens=1024,
    )
    return resp.content[0].text


# ── Public interface ──────────────────────────────────────────────────────────

def generate_with_citations(
    query: str,
    chunks: List[RetrievedChunk],
) -> CitedAnswer:
    if not chunks:
        return CitedAnswer(answer="No context retrieved.", citations=[])

    user_msg = build_cited_prompt(query, chunks)
    console.print(f"[blue]Generating cited answer via {cfg.LLM_PROVIDER}...[/blue]")

    if cfg.LLM_PROVIDER == "ollama":
        raw = _cited_ollama(user_msg)
    elif cfg.LLM_PROVIDER == "openai":
        raw = _cited_openai(user_msg)
    elif cfg.LLM_PROVIDER == "anthropic":
        raw = _cited_anthropic(user_msg)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {cfg.LLM_PROVIDER}")

    result = _parse_response(raw, chunks)
    console.print(f"[green]✓ Answer with {len(result.citations)} citation(s)[/green]")
    return result
