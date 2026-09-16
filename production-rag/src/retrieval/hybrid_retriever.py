"""
Hybrid Retriever — M2: dense + BM25 + RRF in one call.

This replaces the M1 retriever for query time.
Ingestion pipeline is unchanged — we just add a BM25 build step after embedding.

Data flow:
  query
    ├─► Dense retriever (ChromaDB cosine sim) → top-20 ranked chunks
    ├─► BM25 retriever (keyword match)         → top-20 ranked chunks
    └─► RRF fusion                             → top-k final chunks

The final top-k chunks go to the same generator as M1 — nothing else changes.
"""

from typing import List

import chromadb
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

from src.config import cfg
from src.ingestion.embedder import get_chroma_collection
from src.retrieval.bm25_index import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.retriever import RetrievedChunk

console = Console()

# Fetch more candidates from each retriever before fusion
# (we'll trim to top_k after fusion)
CANDIDATES_PER_RETRIEVER = 20


def _dense_retrieve(
    query: str,
    collection: chromadb.Collection,
    n: int = CANDIDATES_PER_RETRIEVER,
) -> List[tuple[Document, float]]:
    """Run dense retrieval, return list of (Document, cosine_similarity)."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        doc = Document(page_content=text, metadata=meta)
        out.append((doc, round(1 - dist, 4)))   # distance → similarity
    return out


def hybrid_retrieve(
    query: str,
    bm25_index: BM25Index,
    collection: chromadb.Collection | None = None,
    top_k: int = cfg.TOP_K,
) -> List[RetrievedChunk]:
    """
    Full hybrid retrieval: dense + BM25 → RRF → top_k RetrievedChunks.

    Args:
        query:      User's natural language question.
        bm25_index: Pre-built BM25Index (built during ingestion).
        collection: ChromaDB collection (loaded once, reused).
        top_k:      Final number of chunks to return.

    Returns:
        List of RetrievedChunk sorted by RRF score (best first).
    """
    if collection is None:
        collection = get_chroma_collection()

    # ── 1. Dense retrieval ────────────────────────────────────────────────
    dense_results = _dense_retrieve(query, collection, n=CANDIDATES_PER_RETRIEVER)
    console.print(f"[dim]Dense: {len(dense_results)} candidates[/dim]")

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────
    bm25_results = bm25_index.query(query, top_k=CANDIDATES_PER_RETRIEVER)
    console.print(f"[dim]BM25:  {len(bm25_results)} candidates[/dim]")

    # ── 3. RRF fusion ─────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results],
        k=60,
        top_k=top_k,
    )
    console.print(f"[green]✓ Hybrid: {len(fused)} chunks after RRF fusion[/green]")

    # ── 4. Convert to RetrievedChunk (same interface as M1) ───────────────
    chunks = []
    for doc, rrf_score in fused:
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                score=rrf_score,
                source=doc.metadata.get("file_name", "unknown"),
                page=doc.metadata.get("page", 0),
                chunk_index=doc.metadata.get("chunk_index", -1),
            )
        )
    return chunks


def display_hybrid_results(query: str, chunks: List[RetrievedChunk]) -> None:
    """Pretty-print hybrid retrieval results."""
    table = Table(title=f"Hybrid top-{len(chunks)} for: '{query}'")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("RRF score", style="cyan", width=10)
    table.add_column("Source", style="green")
    table.add_column("Page", width=4)
    table.add_column("Preview", style="white")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:100].replace("\n", " ") + "..."
        table.add_row(str(i), str(chunk.score), chunk.source, str(chunk.page), preview)

    console.print(table)
