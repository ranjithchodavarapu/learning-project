"""
Embedder — Step 3 of the ingestion pipeline.

What embeddings are:
  - Dense vectors (lists of floats) that encode the *meaning* of text.
  - Similar meaning → vectors are close in vector space (high cosine similarity).
  - We embed every chunk once at ingestion time, then embed the query at retrieval time.

Model used: all-MiniLM-L6-v2 (runs fully locally, no API cost)
  - 384-dimensional vectors
  - Fast and good enough for most tasks
  - In M2+ we can swap to OpenAI text-embedding-3-small (1536 dims, better quality)

Storage: ChromaDB
  - Local persistent vector store (saved to disk at CHROMA_PERSIST_DIR)
  - Supports similarity search out of the box
  - In production you'd swap to Pinecone / Weaviate / Qdrant for scale
"""

from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document
from rich.console import Console

from src.config import cfg

console = Console()


def get_chroma_collection(collection_name: str = cfg.COLLECTION_NAME):
    """
    Returns (or creates) a ChromaDB collection.
    Persists to disk at CHROMA_PERSIST_DIR so data survives restarts.
    """
    client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)

    # SentenceTransformer runs locally — downloads model on first call (~90MB)
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine similarity (vs default L2)
    )
    return collection


def embed_and_store(
    chunks: List[Document],
    collection_name: str = cfg.COLLECTION_NAME,
    batch_size: int = 100,
) -> chromadb.Collection:
    """
    Embed each chunk and upsert into ChromaDB.

    Uses upsert (not add) so re-running on the same docs is idempotent —
    existing chunks get overwritten, not duplicated.

    Args:
        chunks:          Output of chunker.py
        collection_name: ChromaDB collection to write to
        batch_size:      Chunks per batch (avoids memory issues on large corpora)

    Returns:
        The ChromaDB collection (ready for querying)
    """
    collection = get_chroma_collection(collection_name)

    console.print(f"[blue]Embedding {len(chunks)} chunks in batches of {batch_size}...[/blue]")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # ChromaDB needs: ids, documents (text), metadatas
        ids = [f"chunk_{i + j}" for j, _ in enumerate(batch)]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]

        # Chroma's embedding function handles the actual embedding call
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        console.print(f"  [dim]Stored batch {i // batch_size + 1} ({len(batch)} chunks)[/dim]")

    console.print(
        f"[bold green]✓ Vector store ready — {collection.count()} chunks indexed[/bold green]"
    )
    return collection
