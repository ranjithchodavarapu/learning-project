"""
pipeline.py — Chains all steps: profile → generate → execute → insights.

WHAT IT DOES:
  Runs the 4 steps in sequence for a given CSV file:
    Step 1: profile_csv()        → understand the data
    Step 2: generate_code()      → write Python analysis code
    Step 3: execute_with_retry() → run the code, fix errors automatically
    Step 4: generate_insights()  → write plain-English findings

LINE BY LINE:
  @dataclass
  class PipelineResult    → bundles all outputs from all 4 steps
    profile               → DataProfile (schema, stats, sample)
    code                  → the Python code that was generated and run
    result                → ExecutionResult (output, charts, attempts)
    report                → AnalysisReport (insights, report path)
    success               → did the whole pipeline complete?

  class DataAnalystPipeline → wraps all 4 steps into one .run() call
    def run(csv_path)     → main entry point
      Step 1: profile_csv()
              if fails: return early with success=False
              (we can't proceed without understanding the data)

      Step 2: generate_code()
              if no code: return early
              (nothing to execute)

      Step 3: execute_with_retry()
              if not result.success: log warning but continue
              (partial results are still useful for insights)

      Step 4: generate_insights()
              always runs — even if execution failed
              (LLM can comment on what it saw, even partial output)

  console.rule(...)       → prints a divider like "────── Step 1 ──────"
                           helps visually separate each step in terminal

  time.time()             → float seconds since Unix epoch
  elapsed = end - start   → duration of the whole pipeline
  f"{elapsed:.0f}s"       → format with 0 decimal places: "45s" not "45.3s"
"""

import time
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from data_profiler import DataProfile, profile_csv
from code_generator import generate_code
from executor import ExecutionResult, execute_with_retry
from insight_generator import AnalysisReport, generate_insights
from config import cfg

console = Console()


@dataclass
class PipelineResult:
    """All outputs from the full analysis pipeline."""
    profile: DataProfile
    code: str
    result: ExecutionResult
    report: AnalysisReport
    success: bool


class DataAnalystPipeline:
    """
    Runs the full AI data analysis pipeline on a CSV file.
    Chains: profiling → code generation → execution → insights.
    """

    def run(self, csv_path: str) -> PipelineResult:
        """
        Analyse a CSV file end-to-end.

        Args:
            csv_path: Path to the CSV file to analyse.

        Returns:
            PipelineResult with all outputs from all steps.
        """
        start = time.time()

        console.print(Panel(
            f"[bold blue]AI Data Analyst[/bold blue]\n"
            f"File: [cyan]{csv_path}[/cyan]",
            border_style="blue",
        ))

        # Validate file exists before doing anything
        if not Path(csv_path).exists():
            console.print(f"[red]File not found: {csv_path}[/red]")
            return None

        # ── Step 1: Profile the data ──────────────────────────────────────
        console.rule("[bold]Step 1 — Data Profiling[/bold]")
        try:
            profile = profile_csv(csv_path)
        except Exception as e:
            console.print(f"[red]Profiling failed: {e}[/red]")
            return None

        # ── Step 2: Generate analysis code ────────────────────────────────
        console.rule("[bold]Step 2 — Code Generation[/bold]")
        code = generate_code(profile)

        if not code:
            console.print("[red]Code generation failed — cannot continue[/red]")
            return None

        # Show the generated code (first 20 lines)
        lines = code.splitlines()
        preview = "\n".join(lines[:20])
        if len(lines) > 20:
            preview += f"\n... ({len(lines) - 20} more lines)"
        console.print(f"[dim]Generated code preview:[/dim]\n[dim]{preview}[/dim]")

        # ── Step 3: Execute with self-correction ──────────────────────────
        console.rule("[bold]Step 3 — Code Execution[/bold]")
        result = execute_with_retry(code)

        if not result.success:
            console.print(
                f"[yellow]⚠ Code failed after {cfg.MAX_RETRIES + 1} attempts — "
                f"continuing with partial results[/yellow]"
            )
        else:
            console.print(f"[green]✓ {len(result.charts_saved)} chart(s) saved[/green]")
            if result.output:
                console.print(f"[dim]Code output:[/dim]\n[dim]{result.output[:500]}[/dim]")

        # ── Step 4: Generate insights ─────────────────────────────────────
        console.rule("[bold]Step 4 — Insight Generation[/bold]")
        report = generate_insights(profile, result)

        elapsed = time.time() - start
        console.print(f"\n[bold green]✓ Analysis complete in {elapsed:.0f}s[/bold green]")

        # Print summary
        console.print(f"\n[dim]Summary:[/dim]")
        console.print(f"  [dim]Dataset: {profile.shape[0]:,} rows × {profile.shape[1]} columns[/dim]")
        console.print(f"  [dim]Code: {len(code.splitlines())} lines[/dim]")
        console.print(f"  [dim]Attempts: {result.attempts}[/dim]")
        console.print(f"  [dim]Charts: {len(result.charts_saved)}[/dim]")
        console.print(f"  [dim]Report: {report.report_path}[/dim]")

        return PipelineResult(
            profile=profile,
            code=code,
            result=result,
            report=report,
            success=result.success,
        )
