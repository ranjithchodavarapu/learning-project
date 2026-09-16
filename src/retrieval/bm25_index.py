"""
BM25 Index — M2: Sparse (keyword) retrieval.

WHY BM25 exists alongside dense vectors:
  Dense vectors excel at SEMANTIC similarity:
    "What causes fever?" → finds chunks about "elevated body temperature" even
    without the word "fever" appearing.

  BM25 excels at EXACT MATCH:
    "GPT-4o tokenizer" → finds chunks containing exactly those words.
    Dense embeddings sometimes dilute rare proper nouns into generic concepts.

  Together they cover each other's blind spots. This is Hybrid Search.

HOW BM25 works (intuition):
  For a query term t in document d:
    score(t,d) = IDF(t) × (TF(t,d) × (k1+1)) / (TF(t,d) + k1 × (1 - b + b × |d|/avgdl))

  - IDF (Inverse Document Frequency): rare words score higher than common ones.
    "the" appears everywhere → low IDF. "BERT" appears rarely → high IDF.
  - TF (Term Frequency): how often the term appears in this document.
  - k1, b: tuning constants (k1=1.5, b=0.75 are standard defaults).
  - Length normalisation (b param): long docs don't dominate just from repetition.

This is the same algorithm Elasticsearch / Lucene use internally.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from rich.console import Console

console = Console()

BM25_CACHE_PATH = Path("./data/bm25_index.pkl")


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Good enough for most English text.
    In production: add stopword removal, stemming (e.g. nltk).
    """
    return text.lower().split()


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi with persistence and a clean query interface.
    Stores the same chunks as ChromaDB so scores are comparable.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Document] = []

    def build(self, chunks: List[Document]) -> None:
        """Build the BM25 index from a list of chunks. Call once after ingestion."""
        self.chunks = chunks
        tokenized = [tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        console.print(f"[green]✓ BM25 index built — {len(chunks)} documents[/green]")

    def save(self, path: Path = BM25_CACHE_PATH) -> None:
        """Persist index to disk so we don't rebuild on every run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        console.print(f"[dim]BM25 index saved → {path}[/dim]")

    def load(self, path: Path = BM25_CACHE_PATH) -> bool:
        """Load index from disk. Returns True if successful."""
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        console.print(f"[green]✓ BM25 index loaded — {len(self.chunks)} documents[/green]")
        return True

    def query(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Return top_k chunks by BM25 score.

        Returns:
            List of (Document, normalised_score) sorted best-first.
            Scores are normalised to [0, 1] so they're comparable to cosine similarity.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        tokens = tokenize(query)
        raw_scores = self.bm25.get_scores(tokens)

        # Normalise: divide by max score so range is [0, 1]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normed = raw_scores / max_score

        # Pair with chunks and sort descending
        scored = sorted(
            zip(self.chunks, normed.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]
