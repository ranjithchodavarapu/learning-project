"""
Tests for M1 — run with: pytest tests/test_m1.py -v

These tests use a small in-memory text fixture so they run without
any API keys or real documents. They verify the logic, not the LLM quality.
"""

import pytest
from langchain.schema import Document

from src.ingestion.chunker import chunk_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
by retrieving relevant information from external knowledge bases before generating responses.

The system works in two stages. First, during ingestion, documents are split into chunks,
embedded into dense vectors, and stored in a vector database. Second, at query time, the
user's question is embedded and used to find the most similar chunks via nearest-neighbour search.

RAG reduces hallucinations because the model is constrained to answer from retrieved context.
It also allows knowledge to be updated without retraining the model — you just re-ingest new docs.
""".strip()


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content=SAMPLE_TEXT,
            metadata={"source": "test.txt", "file_name": "test.txt", "file_type": "txt"},
        )
    ]


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_chunker_produces_chunks(sample_docs):
    chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1, "A long document should produce multiple chunks"


def test_chunks_inherit_metadata(sample_docs):
    chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert "file_name" in chunk.metadata, "Chunks should inherit parent metadata"


def test_chunk_size_respected(sample_docs):
    chunk_size = 150
    chunks = chunk_documents(sample_docs, chunk_size=chunk_size, chunk_overlap=10)
    for chunk in chunks:
        # Allow 20% overshoot (splitter prefers natural boundaries over hard limits)
        assert len(chunk.page_content) <= chunk_size * 1.2, (
            f"Chunk too large: {len(chunk.page_content)} chars"
        )


def test_chunk_index_assigned(sample_docs):
    chunks = chunk_documents(sample_docs, chunk_size=200, chunk_overlap=20)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "Chunk indices should be sequential"


def test_empty_document_list():
    chunks = chunk_documents([])
    assert chunks == [], "Empty input should return empty list"


# ── Prompt builder test ───────────────────────────────────────────────────────

def test_build_prompt_includes_query():
    from src.generation.generator import build_prompt
    from src.retrieval.retriever import RetrievedChunk

    chunks = [
        RetrievedChunk(
            text="RAG reduces hallucinations.",
            score=0.92,
            source="test.txt",
            page=1,
            chunk_index=0,
        )
    ]
    prompt = build_prompt("What is RAG?", chunks)
    assert "What is RAG?" in prompt
    assert "RAG reduces hallucinations" in prompt
    assert "Chunk 1" in prompt
