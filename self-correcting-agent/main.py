"""
main.py — Entry point for the Self-Correcting AI Agent.

HOW TO RUN:
  python main.py
  python main.py --question "Explain quantum entanglement simply"
  python main.py --question "What is recursion?" --threshold 0.85 --attempts 4
  python main.py --demo

LINE BY LINE:
  --threshold            → override SCORE_THRESHOLD from CLI
                           allows testing with different quality bars
                           without editing .env

  --attempts             → override MAX_ATTEMPTS from CLI
                           useful for quick tests (--attempts 1) or
                           high-quality runs (--attempts 5)

  args.threshold         → if provided, override cfg value at runtime
  cfg.SCORE_THRESHOLD = args.threshold
                         → mutates the shared config object
                         → all modules read from cfg → pick up the new value
                         → this works because cfg is a singleton

  DEMO_QUESTIONS         → interesting questions that benefit from self-correction
                           chosen because they have multiple dimensions to evaluate:
                           accuracy (factual), completeness (covers sub-topics),
                           clarity (accessible language), conciseness (not rambling)

  result.final_answer    → the best answer after self-correction
  result.score_history   → shows score progression across attempts
  result.converged       → did score pass threshold?

  Panel(result.final_answer)
                         → rich Panel draws a border around the answer
                         → makes it visually distinct from the process logs
"""

import argparse
import random
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from agent import SelfCorrectingAgent
from config import cfg

console = Console()

DEMO_QUESTIONS = [
    "Explain how transformers work in large language models",
    "What is the difference between supervised and unsupervised learning?",
    "How does the internet actually work, from typing a URL to seeing a webpage?",
    "Explain gradient descent in machine learning",
    "What is attention mechanism in neural networks?",
]


def main():
    # ── Parse CLI arguments ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Self-Correcting AI Agent — generates, evaluates, and improves answers"
    )
    parser.add_argument(
        "--question", type=str,
        help="Question to answer (if not given, prompts interactively)",
    )
    parser.add_argument(
        "--threshold", type=float,
        help=f"Quality threshold 0.0-1.0 (default: {cfg.SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--attempts", type=int,
        help=f"Max correction attempts (default: {cfg.MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with a preset example question",
    )
    args = parser.parse_args()

    # ── Override config from CLI ──────────────────────────────────────────
    if args.threshold is not None:
        cfg.SCORE_THRESHOLD = args.threshold
        console.print(f"[dim]Threshold set to: {cfg.SCORE_THRESHOLD}[/dim]")

    if args.attempts is not None:
        cfg.MAX_ATTEMPTS = args.attempts
        console.print(f"[dim]Max attempts set to: {cfg.MAX_ATTEMPTS}[/dim]")

    # ── Get question ──────────────────────────────────────────────────────
    if args.demo:
        question = random.choice(DEMO_QUESTIONS)
        console.print(f"[green]Demo question: {question}[/green]\n")

    elif args.question:
        question = args.question

    else:
        console.print("[bold blue]Self-Correcting AI Agent[/bold blue]")
        console.print(
            "[dim]The agent will generate an answer, score it, "
            "reflect on what's wrong, and retry until quality passes.[/dim]\n"
        )
        question = Prompt.ask("[bold cyan]Your question[/bold cyan]")

    if not question.strip():
        console.print("[red]No question provided. Exiting.[/red]")
        return

    # ── Run the agent ─────────────────────────────────────────────────────
    agent = SelfCorrectingAgent()
    result = agent.run(question.strip())

    # ── Display final answer ──────────────────────────────────────────────
    console.print(Panel(
        result.final_answer,
        title=f"[bold]Final Answer[/bold] (score: {result.final_score.overall:.2f})",
        border_style="green" if result.converged else "yellow",
    ))

    # ── Show dimension breakdown ──────────────────────────────────────────
    console.print("\n[bold]Final scores:[/bold]")
    for dim, val in result.final_score.dimensions.items():
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        color = "green" if val >= cfg.SCORE_THRESHOLD else "yellow"
        console.print(f"  {dim:12} [{color}]{bar}[/{color}] {val:.2f}")

    status = "converged" if result.converged else "best effort (max attempts reached)"
    console.print(f"\n[dim]Status: {status}[/dim]")
    console.print(f"[dim]Attempts: {result.total_attempts}[/dim]")


if __name__ == "__main__":
    main()
