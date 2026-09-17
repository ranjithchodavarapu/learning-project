"""
main.py — Entry point for the multi-agent research system.

HOW TO RUN:
  python main.py
  python main.py --topic "History of transformer neural networks"
  python main.py --topic "Climate change solutions" --subtopics 4 --facts 6

LINE BY LINE:
  argparse             → standard Python library for command-line arguments
                         lets users pass --topic "X" instead of editing code

  parser.add_argument("--topic", ...)
                       → optional argument — if not given, we ask interactively
                         type=str means it expects a string value
                         help= shows in --help output

  parser.add_argument("--subtopics", type=int, default=3)
                       → integer argument with default value of 3
                         --subtopics 5 → research 5 subtopics

  args = parser.parse_args()
                       → parses sys.argv (the command line) and returns namespace
                         args.topic = value of --topic
                         args.subtopics = value of --subtopics

  if args.subtopics:
      cfg.MAX_SUBTOPICS = args.subtopics
                       → override config at runtime from command line
                         config defaults come from .env
                         but CLI args override those defaults

  Prompt.ask(...)      → rich library interactive prompt
                         waits for user input in the terminal
                         prettier than input()

  while not topic.strip():
                       → loop until user enters non-empty topic
                         .strip() removes whitespace — "   " counts as empty

  pipeline = ResearchPipeline()
  memory = pipeline.run(topic)
                       → these two lines run the entire system

  memory.report        → the final report written by the summarizer
  memory.errors        → any errors that occurred during research

  console.print(Panel(memory.report, ...))
                       → Panel draws a border around the report
                         makes it visually distinct in the terminal

  if memory.errors:    → only print errors section if errors exist
                         clean runs show no errors section
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from pipeline import ResearchPipeline
from config import cfg

console = Console()


def main():
    # ── Parse command-line arguments ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System — researches any topic using AI agents"
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic (if not given, prompts interactively)",
    )
    parser.add_argument(
        "--subtopics",
        type=int,
        help=f"Number of subtopics to research (default: {cfg.MAX_SUBTOPICS})",
    )
    parser.add_argument(
        "--facts",
        type=int,
        help=f"Facts per subtopic (default: {cfg.FACTS_PER_SUBTOPIC})",
    )
    args = parser.parse_args()

    # ── Override config from CLI if provided ──────────────────────────────
    # This lets users customise without editing .env
    if args.subtopics:
        cfg.MAX_SUBTOPICS = args.subtopics
    if args.facts:
        cfg.FACTS_PER_SUBTOPIC = args.facts

    # ── Get research topic ────────────────────────────────────────────────
    topic = args.topic  # None if not provided via CLI

    if not topic:
        # Interactive mode — ask user in terminal
        console.print(Panel(
            "[bold blue]Multi-Agent Research System[/bold blue]\n"
            "[dim]Powered by Ollama — runs fully locally[/dim]",
            border_style="blue",
        ))
        # Loop until non-empty topic entered
        while not topic or not topic.strip():
            topic = Prompt.ask("[bold cyan]Enter research topic[/bold cyan]")

    # ── Run the pipeline ──────────────────────────────────────────────────
    pipeline = ResearchPipeline()
    memory = pipeline.run(topic.strip())

    # ── Display results ───────────────────────────────────────────────────
    console.rule("[bold green]Research Complete[/bold green]")

    if memory.report:
        console.print(Panel(
            memory.report,
            title=f"[bold]Research Report: {memory.topic}[/bold]",
            border_style="green",
        ))
    else:
        console.print("[red]No report generated — check errors below[/red]")

    # Show any errors that occurred
    if memory.errors:
        console.print("\n[yellow]Errors during research:[/yellow]")
        for error in memory.errors:
            console.print(f"  [red]•[/red] {error}")

    # Show final stats
    console.print(f"\n[dim]Stats:[/dim]")
    console.print(f"  [dim]Facts found: {len(memory.facts)}[/dim]")
    console.print(f"  [dim]Facts verified: {len(memory.verified_facts)}[/dim]")
    console.print(f"  [dim]Conflicts: {len(memory.conflicts)}[/dim]")


if __name__:
    main()
