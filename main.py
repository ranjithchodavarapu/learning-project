"""
main.py — Production RAG Q&A (M1-M6 complete pipeline).

Run:
    python main.py                  # full pipeline (M2-M5 all on)
    python main.py --simple         # M1 only (naive RAG)
    python main.py --no-rewrite     # skip query rewriting (M4 off)
    python main.py --no-rerank      # skip reranking (M3 off)
    python main.py --eval           # run M6 evaluation after session
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.evaluation.evaluator import (
    EvalSample,
    display_eval_results,
    evaluate_simple,
    load_test_set,
)
from src.pipeline import PipelineConfig, RAGPipeline

console = Console()


def run_interactive(pipeline: RAGPipeline, run_eval: bool = False) -> None:
    console.print(Panel(
        "[bold blue]Production RAG System — M1-M6[/bold blue]\n"
        "[dim]All milestones active. Type 'exit' to quit.[/dim]",
        border_style="blue",
    ))

    session_samples = []

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query.strip():
            continue

        result = pipeline.query(query)

        console.rule("[bold]Answer[/bold]")
        console.print(f"\n{result.answer}\n")

        if result.citations:
            console.rule("[dim]Sources[/dim]")
            for i, c in enumerate(result.citations, 1):
                console.print(
                    f"  [{i}] [green]{c.source_file}[/green] p.{c.page} "
                    f"(score: {c.relevance_score}) — \"{c.excerpt[:80]}...\""
                )
        console.rule()

        if run_eval:
            gt = Prompt.ask("[dim]Ground truth (Enter to skip)[/dim]", default="")
            if gt:
                session_samples.append(EvalSample(
                    question=query,
                    ground_truth=gt,
                    answer=result.answer,
                    contexts=[c.excerpt for c in result.citations],
                ))

    if run_eval and session_samples:
        console.rule("[bold]M6 — Evaluation[/bold]")
        eval_result = evaluate_simple(session_samples)
        display_eval_results(eval_result, milestone="session")


def main():
    parser = argparse.ArgumentParser(description="Production RAG System")
    parser.add_argument("--simple",       action="store_true")
    parser.add_argument("--no-rewrite",   action="store_true")
    parser.add_argument("--no-rerank",    action="store_true")
    parser.add_argument("--no-hybrid",    action="store_true")
    parser.add_argument("--no-citations", action="store_true")
    parser.add_argument("--eval",         action="store_true")
    args = parser.parse_args()

    if args.simple:
        cfg = PipelineConfig(
            use_query_rewriting=False,
            use_hybrid_search=False,
            use_reranking=False,
            use_citations=False,
        )
    else:
        cfg = PipelineConfig(
            use_query_rewriting=not args.no_rewrite,
            use_hybrid_search=not args.no_hybrid,
            use_reranking=not args.no_rerank,
            use_citations=not args.no_citations,
        )

    pipeline = RAGPipeline(config=cfg)
    pipeline.load()
    run_interactive(pipeline, run_eval=args.eval)


if __name__ == "__main__":
    main()
