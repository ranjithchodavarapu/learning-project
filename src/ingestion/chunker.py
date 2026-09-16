"""
Chunker — Step 2 of the ingestion pipeline.

Why chunking matters:
  - LLMs have context window limits — you can't pass an entire PDF.
  - Embedding a full document loses precision — small focused chunks retrieve better.
  - Overlap between chunks prevents losing context at boundaries.

Strategy used here: RecursiveCharacterTextSplitter
  - Tries to split on paragraphs → sentences → words → characters (in order).
  - Much better than naive fixed-size splitting because it respects natural boundaries.

In M2 we'll add semantic chunking (splits based on topic shift, not character count).
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from src.config import cfg

console = Console()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = cfg.CHUNK_SIZE,
    chunk_overlap: int = cfg.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents:     Output of loader.py — full pages/files.
        chunk_size:    Max characters per chunk (default from .env).
        chunk_overlap: Characters shared between consecutive chunks (default from .env).

    Returns:
        List of chunked Documents, each inheriting the parent's metadata
        plus a new 'chunk_index' field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try these separators in order — stops when a split is small enough
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index so we can trace it back
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    console.print(
        f"[bold green]Chunked {len(documents)} docs → {len(chunks)} chunks[/bold green] "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
