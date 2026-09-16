"""
Retriever — Step 1 of the retrieval pipeline (M1: dense-only).

What happens here:
  1. User query comes in as plain text.
  2. We embed it using the same model used at ingestion time.
  3. ChromaDB finds the top-k most similar chunks (cosine similarity).
  4. We return those chunks + their similarity scores.

This is "naive" retrieval — just dense vector similarity.
In M2 we'll add BM25 (keyword) and fuse both results (hybrid retrieval).

Key concept — cosine similarity:
  - 1.0  = identical meaning
  - 0.7+ = very relevant
  - 0.5  = loosely related
  - <0.3  = probably noise
"""

from dataclasses import dataclass
from typing import List

import chromadb
from rich.console import Console
from rich.table import Table

from src.config import cfg
from src.ingestion.embedder import get_chroma_collection

console = Console()


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its score and metadata."""
    text: str
    score: float          # cosine similarity (higher = more relevant)
    source: str           # file name
    page: int             # page number (PDF) or 0 for text files
    chunk_index: int      # position in the full document


def retrieve(
    query: str,
    top_k: int = cfg.TOP_K,
    collection: chromadb.Collection | None = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query:      The user's question in natural language.
        top_k:      Number of chunks to return (default from .env).
        collection: Optional pre-loaded collection (avoids re-opening ChromaDB).

    Returns:
        List of RetrievedChunk, sorted by relevance (highest score first).
    """
    if collection is None:
        collection = get_chroma_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "Vector store is empty. Run the ingestion pipeline first:\n"
            "  python -m src.ingestion.pipeline --file your_doc.pdf"
        )

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns distances (lower = closer), convert to similarity score
    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist   # cosine distance → cosine similarity
        chunks.append(
            RetrievedChunk(
                text=text,
                score=round(similarity, 4),
                source=meta.get("file_name", "unknown"),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", -1),
            )
        )

    return chunks   # already sorted best-first by ChromaDB


def display_results(query: str, chunks: List[RetrievedChunk]) -> None:
    """Pretty-print retrieval results to the terminal (dev/debug helper)."""
    table = Table(title=f"Top {len(chunks)} chunks for: '{query}'")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Source", style="green")
    table.add_column("Page", width=4)
    table.add_column("Text preview", style="white")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:120].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            str(chunk.score),
            chunk.source,
            str(chunk.page),
            preview,
        )

    console.print(table)
