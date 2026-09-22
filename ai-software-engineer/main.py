"""
main.py — Entry point for the AI Software Engineer Agent.

HOW TO RUN:
  python main.py
  python main.py --task "Build a calculator with add, subtract, multiply, divide"
  python main.py --task "Create a file organiser that sorts files by extension" --language python
  python main.py --demo   (runs a built-in example task)

LINE BY LINE:
  argparse               → standard Python CLI argument library
  --task                 → the software task to build
  --language             → programming language (default: python)
  --demo                 → run a preset example task

  args.demo              → if --demo: use a preset task
                           good for testing without thinking of a task

  Prompt.ask(...)        → rich interactive prompt
                           waits for user to type the task

  pipeline.run(task, language)
                         → runs the full 4-step pipeline
                         → returns ProjectResult

  result.workspace_path  → where the code was saved
  result.all_tests_passed→ did everything work?
  result.test_results    → list of TestResult objects
  result.debug_attempts  → how many debug cycles ran

  DEMO_TASKS             → list of example tasks for --demo mode
    random.choice(...)   → picks one randomly each time
"""

import argparse
import random
from rich.console import Console
from rich.prompt import Prompt
from pipeline import EngineerPipeline

console = Console()

DEMO_TASKS = [
    "Build a calculator with add, subtract, multiply, divide, and power functions",
    "Create a to-do list manager with add, remove, complete, and list operations",
    "Build a text statistics tool that counts words, sentences, and characters",
    "Create a simple number guessing game with hints and score tracking",
    "Build a unit converter for temperature, length, and weight",
]


def main():
    # ── Parse CLI arguments ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="AI Software Engineer Agent — describe a task, get working code"
    )
    parser.add_argument(
        "--task", type=str,
        help="Software task to build (natural language description)",
    )
    parser.add_argument(
        "--language", type=str, default="python",
        choices=["python", "javascript", "typescript", "bash"],
        help="Programming language (default: python)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a preset example task",
    )
    args = parser.parse_args()

    # ── Get task ──────────────────────────────────────────────────────────
    if args.demo:
        task = random.choice(DEMO_TASKS)
        console.print(f"[green]Demo task: {task}[/green]")

    elif args.task:
        task = args.task

    else:
        console.print("[bold blue]AI Software Engineer Agent[/bold blue]")
        console.print("[dim]Describe what you want to build — the agent will plan, code, test and debug it.[/dim]\n")
        task = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]")

    if not task.strip():
        console.print("[red]No task provided. Exiting.[/red]")
        return

    # ── Run the pipeline ──────────────────────────────────────────────────
    pipeline = EngineerPipeline()
    result = pipeline.run(task.strip(), args.language)

    # ── Show final summary ────────────────────────────────────────────────
    console.print(f"\n[bold]Workspace:[/bold] {result.workspace_path}")
    console.print(f"[bold]Files created:[/bold]")
    for f in result.files_written:
        console.print(f"  • {f}")

    if result.all_tests_passed:
        console.print("\n[bold green]✓ Project built successfully — all tests passing![/bold green]")
    else:
        console.print("\n[yellow]⚠ Project built — some tests still failing[/yellow]")
        console.print("[dim]Check the workspace folder and fix manually if needed[/dim]")


if __name__ == "__main__":
    main()
