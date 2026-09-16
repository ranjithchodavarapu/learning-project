"""
Full RAG Pipeline — M1 through M6 wired together.

This is the production entrypoint. Every milestone adds one stage:

  Query
    │
    ▼  [M4] Query rewriting (HyDE + multi-query)
    │
    ▼  [M2] Hybrid retrieval (dense + BM25 + RRF)   ← top-20 candidates
    │
    ▼  [M3] Reranking (cross-encoder)                ← top-5 final chunks
    │
    ▼  [M5] Generation with citations
    │
    ▼  CitedAnswer (answer + sources)
    │
    ▼  [M6] Optional evaluation (if ground truth available)

You can enable/disable each stage via flags — useful for A/B testing
which milestones actually improve your specific domain.
"""

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

from src.config import cfg
from src.generation.citations import CitedAnswer, generate_with_citations
from src.generation.generator import generate_answer
from src.ingestion.embedder import get_chroma_collection
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.reranker import rerank
from src.retrieval.retriever import RetrievedChunk, retrieve

console = Console()


@dataclass
class PipelineConfig:
    """Toggle each milestone on/off for ablation studies."""
    use_query_rewriting: bool = True    # M4
    use_hybrid_search:   bool = True    # M2
    use_reranking:       bool = True    # M3
    use_citations:       bool = True    # M5
    hyde_strategy:       str  = "both"  # "hyde" | "multi" | "both"
    retrieval_candidates: int = 20      # chunks before reranking
    final_top_k:         int = 5        # chunks after reranking


class RAGPipeline:
    """
    The full production RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.load()               # load vector store + BM25 index
        result = pipeline.query("What is the main finding?")
        print(result.answer)
        for c in result.citations:
            print(c.source_file, c.page)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.collection = None
        self.bm25_index = BM25Index()
        self._loaded = False

    def load(self) -> None:
        """Load vector store and BM25 index. Call once before querying."""
        console.print("[dim]Loading ChromaDB collection...[/dim]")
        self.collection = get_chroma_collection()
        count = self.collection.count()
        console.print(f"[green]✓ ChromaDB: {count} chunks[/green]")

        if self.config.use_hybrid_search:
            loaded = self.bm25_index.load()
            if not loaded:
                console.print(
                    "[yellow]BM25 index not found. Run ingestion with --build-bm25 flag.[/yellow]\n"
                    "[dim]Falling back to dense-only retrieval.[/dim]"
                )
                self.config.use_hybrid_search = False

        self._loaded = True

    def query(self, user_query: str) -> CitedAnswer:
        """
        Run the full pipeline on a user query.

        Returns CitedAnswer if use_citations=True,
        otherwise wraps a plain string in a CitedAnswer with no citations.
        """
        if not self._loaded:
            raise RuntimeError("Call pipeline.load() before querying.")

        console.print(Panel(f"[bold]Query:[/bold] {user_query}", border_style="blue"))

        # ── M4: Query rewriting ───────────────────────────────────────────
        queries_to_retrieve = [user_query]

        if self.config.use_query_rewriting:
            console.rule("[bold]M4 — Query rewriting[/bold]")
            rewritten = rewrite_query(user_query, strategy=self.config.hyde_strategy)

            if "hyde_doc" in rewritten:
                # Use the HyDE document as the primary retrieval query
                queries_to_retrieve = [rewritten["hyde_doc"]]

            if "multi_queries" in rewritten:
                # Add multi-query expansions
                queries_to_retrieve += rewritten["multi_queries"]

            console.print(f"[dim]{len(queries_to_retrieve)} query variants to retrieve[/dim]")

        # ── M2: Hybrid retrieval ──────────────────────────────────────────
        console.rule("[bold]M2 — Hybrid retrieval[/bold]")

        if self.config.use_hybrid_search:
            # Retrieve for each query variant, fuse with RRF
            from src.retrieval.fusion import reciprocal_rank_fusion
            from langchain_core.documents import Document

            all_ranked: list = []
            for q in queries_to_retrieve:
                chunks = hybrid_retrieve(
                    q, self.bm25_index, self.collection,
                    top_k=self.config.retrieval_candidates,
                )
                # Convert back to (Document, score) for RRF
                as_docs = [
                    (Document(page_content=c.text, metadata={
                        "file_name": c.source, "page": c.page, "chunk_index": c.chunk_index
                    }), c.score)
                    for c in chunks
                ]
                all_ranked.append(as_docs)

            fused = reciprocal_rank_fusion(
                all_ranked, top_k=self.config.retrieval_candidates
            )
            candidates = [
                RetrievedChunk(
                    text=doc.page_content,
                    score=score,
                    source=doc.metadata.get("file_name", "unknown"),
                    page=doc.metadata.get("page", 0),
                    chunk_index=doc.metadata.get("chunk_index", -1),
                )
                for doc, score in fused
            ]
        else:
            # Fallback: dense-only, just use first query variant
            candidates = retrieve(
                queries_to_retrieve[0],
                top_k=self.config.retrieval_candidates,
                collection=self.collection,
            )

        console.print(f"[green]✓ {len(candidates)} candidates retrieved[/green]")

        # ── M3: Reranking ─────────────────────────────────────────────────
        console.rule("[bold]M3 — Reranking[/bold]")

        if self.config.use_reranking:
            final_chunks = rerank(
                user_query,   # always rerank against the ORIGINAL query
                candidates,
                top_k=self.config.final_top_k,
            )
        else:
            final_chunks = candidates[:self.config.final_top_k]

        # ── M5: Generation with citations ─────────────────────────────────
        console.rule("[bold]M5 — Generation with citations[/bold]")

        if self.config.use_citations:
            result = generate_with_citations(user_query, final_chunks)
        else:
            answer_text = generate_answer(user_query, final_chunks)
            result = CitedAnswer(answer=answer_text, citations=[])

        return result
