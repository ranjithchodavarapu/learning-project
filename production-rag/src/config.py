"""
Central configuration — loaded once, imported everywhere.
Supports: openai | anthropic | ollama
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    # ── LLM provider ─────────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # ── Ollama (local — free) ─────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.1")

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str      = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ── Anthropic ─────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Embeddings (always local, no API cost) ────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Vector store ──────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    COLLECTION_NAME: str    = "rag_documents"

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 5))


cfg = Config()
