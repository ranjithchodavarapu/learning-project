"""
Reranker — M3: Cross-encoder reranking.

THE PROBLEM WITH M2:
  Both dense embeddings and BM25 are BI-ENCODERS:
    - They encode query and document SEPARATELY, then compare.
    - Fast (embed once, search millions of chunks instantly).
    - But they miss subtle interactions between query and document words.

CROSS-ENCODER SOLUTION:
  - Takes (query, document) as a SINGLE INPUT.
  - Processes them TOGETHER — attention layers see both simultaneously.
  - Scores how well this specific doc answers this specific query.
  - Much more accurate — but too slow to run on millions of docs.

TWO-STAGE STRATEGY (industry standard):
  Stage 1 — Retrieve (fast):   Bi-encoder retrieves top-20 candidates.
  Stage 2 — Rerank (accurate): Cross-encoder scores only those 20, picks top-5.

  This gives you the speed of bi-encoders + accuracy of cross-encoders.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fine-tuned on MS MARCO passage ranking dataset (530k queries).
  - Returns a relevance logit — higher = more relevant.
  - Runs on CPU in ~50ms for 20 passages. GPU would be ~5ms.

ALTERNATIVE MODELS (drop-in replacements):
  - "cross-encoder/ms-marco-MiniLM-L-12-v2"  (larger, more accurate, slower)
  - "BAAI/bge-reranker-base"                  (multilingual)
  - Cohere Rerank API                          (best quality, paid)
"""

from typing import List

from sentence_transformers import CrossEncoder
from rich.console import Console

from src.retrieval.retriever import RetrievedChunk

console = Console()

# Loaded once — model download ~25MB on first run
_cross_encoder: CrossEncoder | None = None


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Lazy-load the cross-encoder (singleton — avoid reloading per query)."""
    global _cross_encoder
    if _cross_encoder is None:
        console.print(f"[dim]Loading cross-encoder: {model_name}[/dim]")
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        console.print("[green]✓ Cross-encoder loaded[/green]")
    return _cross_encoder


def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[RetrievedChunk]:
    """
    Rerank retrieved chunks using a cross-encoder.

    Args:
        query:      The user's original question.
        chunks:     Candidates from hybrid retrieval (typically top-20).
        top_k:      How many to keep after reranking (typically 5).
        model_name: Which cross-encoder to use.

    Returns:
        Top-k chunks sorted by cross-encoder score (best first).
        The .score field is replaced with the cross-encoder score.
    """
    if not chunks:
        return []

    encoder = get_cross_encoder(model_name)

    # Cross-encoder expects list of [query, passage] pairs
    pairs = [[query, chunk.text] for chunk in chunks]

    console.print(f"[blue]Reranking {len(chunks)} chunks with cross-encoder...[/blue]")
    scores = encoder.predict(pairs)   # returns numpy array of floats

    # Attach scores and sort
    scored_chunks = sorted(
        zip(chunks, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    result = []
    for chunk, score in scored_chunks[:top_k]:
        # Replace the RRF/cosine score with the cross-encoder score
        reranked = RetrievedChunk(
            text=chunk.text,
            score=round(float(score), 4),
            source=chunk.source,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
        )
        result.append(reranked)

    console.print(
        f"[green]✓ Reranked {len(chunks)} → {len(result)} chunks "
        f"(scores: {result[0].score:.3f} → {result[-1].score:.3f})[/green]"
    )
    return result
