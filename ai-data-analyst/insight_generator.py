"""
insight_generator.py — LLM reads execution results and writes insights.

WHAT IT DOES:
  After the code runs and produces output (printed stats, charts),
  this module takes all that output and asks the LLM to:
  1. Summarise the key findings in plain English
  2. Highlight the most important patterns and anomalies
  3. Suggest follow-up questions or next steps
  4. Save everything as a markdown report

WHY A SEPARATE INSIGHT STEP:
  The code execution step produces RAW output:
  - Numbers, tables, statistics printed to stdout
  - Chart image files saved to disk

  Raw numbers are not insights. The insight step is where the LLM
  acts as a data analyst — it reads the numbers and says
  "the most interesting thing here is X because Y".

LINE BY LINE:
  @dataclass
  class AnalysisReport    → full report container
    topic: str            → what was analysed (filename)
    insights: str         → LLM-written insights in plain English
    code_output: str      → raw printed output from executing code
    charts: list          → paths to saved chart PNG files
    report_path: str      → path to saved markdown report

  SYSTEM_PROMPT           → "You are a senior data analyst"
                           gives LLM a persona that produces business-style insights
                           not just technical descriptions

  def generate_insights() → main function: profile + result → AnalysisReport
    context               → builds the context string for the LLM
                           includes: what the data is, what the code printed,
                           how many charts were made
    _call_llm()           → sends to Ollama, gets insights back
    _save_report()        → saves everything as a markdown file

  def _build_context()    → assembles all information for the LLM
    f-string              → multi-line string with all variables interpolated
    profile.shape         → "(1000, 8)" — rows × columns
    result.output[:3000]  → first 3000 chars of code output
                          → [:3000] truncates very long output
                          → LLMs have context limits — we can't send everything

  def _save_report()      → saves insights + metadata to a .md file
    datetime.now()        → current timestamp for filename
    .strftime(...)        → formats timestamp as "2024-01-15_14-30"
    open(..., "w")        → opens file for writing
    f.write(...)          → writes the markdown content

  "## Charts Generated"   → markdown heading (## = h2)
  f"- {chart}"            → markdown bullet point for each chart path
  This creates a proper markdown report the user can open in any editor
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List
import requests
from rich.console import Console
from rich.panel import Panel
from data_profiler import DataProfile
from executor import ExecutionResult
from config import cfg

console = Console()


@dataclass
class AnalysisReport:
    """Complete analysis report including insights and charts."""
    topic: str                          # filename analysed
    insights: str                       # LLM-written insights
    code_output: str                    # raw code execution output
    charts: List[str] = field(default_factory=list)
    report_path: str = ""               # path to saved .md file


SYSTEM_PROMPT = """You are a senior data analyst writing an executive summary.
Given the output of a data analysis script and information about the dataset,
write clear, actionable insights in plain English.
Focus on: key patterns, anomalies, trends, and business implications.
Structure your response with sections: Key Findings, Patterns & Trends, Anomalies, Recommendations.
Be specific — reference actual numbers from the output."""


def generate_insights(
    profile: DataProfile,
    result: ExecutionResult,
) -> AnalysisReport:
    """
    Generate plain-English insights from analysis results.

    Args:
        profile: DataProfile from data_profiler.py
        result:  ExecutionResult from executor.py

    Returns:
        AnalysisReport with insights, output, charts, and saved report path.
    """
    console.print("[blue]Generating insights...[/blue]")

    # Build context for the LLM
    context = _build_context(profile, result)

    # Ask LLM to write insights
    insights = _call_llm(SYSTEM_PROMPT, context)

    if not insights:
        insights = "Could not generate insights — LLM unavailable."
        console.print("[yellow]LLM unavailable — insights skipped[/yellow]")
    else:
        console.print("[green]✓ Insights generated[/green]")

    # Create the report object
    report = AnalysisReport(
        topic=Path(profile.path).name,
        insights=insights,
        code_output=result.output,
        charts=result.charts_saved,
    )

    # Save to markdown file
    report.report_path = _save_report(report, profile)

    # Display insights in terminal
    console.print(Panel(
        insights,
        title="[bold green]Data Insights[/bold green]",
        border_style="green",
    ))

    return report


def _build_context(profile: DataProfile, result: ExecutionResult) -> str:
    """
    Assemble all information into a context string for the LLM.

    Includes dataset overview, analysis output, and chart count.
    Truncates output to 3000 chars to stay within token limits.
    """
    # Truncate output if very long — LLMs have context window limits
    # [:3000] = first 3000 characters
    output_preview = result.output[:3000] if result.output else "No output captured."

    charts_info = (
        f"{len(result.charts_saved)} charts saved to disk"
        if result.charts_saved
        else "No charts generated"
    )

    return f"""DATASET OVERVIEW:
File: {profile.path}
Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns
Columns: {', '.join(profile.columns)}

ANALYSIS OUTPUT (from executed code):
{output_preview}

CHARTS: {charts_info}

Based on the above, write a comprehensive data analysis report with:
1. Key Findings (3-5 bullet points of the most important facts)
2. Patterns & Trends (what patterns exist in the data)
3. Anomalies (anything unexpected or worth investigating)
4. Recommendations (what to explore next or act on)"""


def _call_llm(system: str, user: str) -> str:
    """Call Ollama for insight generation."""
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,  # slightly creative for readable prose
            "num_ctx": 8192,
        },
    }

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=cfg.TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    except Exception as e:
        console.print(f"[red]Insight LLM error: {e}[/red]")
        return ""


def _save_report(report: AnalysisReport, profile: DataProfile) -> str:
    """
    Save the full report to a markdown file.

    Markdown format means it renders nicely in VS Code, GitHub, Obsidian etc.
    Returns the file path string.
    """
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped filename so reports don't overwrite each other
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_name = Path(profile.path).stem[:20].replace(" ", "_")
    filename = f"report_{safe_name}_{timestamp}.md"
    filepath = output_dir / filename

    # Build markdown content
    # ## = h2 heading in markdown
    # ** = bold in markdown
    # - = bullet point in markdown
    content = f"""# Data Analysis Report: {report.topic}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Dataset Overview
- **File:** {profile.path}
- **Shape:** {profile.shape[0]:,} rows × {profile.shape[1]} columns
- **Columns:** {', '.join(profile.columns)}

## Key Insights
{report.insights}

## Analysis Output
```
{report.code_output[:2000]}
```

## Charts Generated
{chr(10).join(f"- {chart}" for chart in report.charts) if report.charts else "- None"}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"[dim]Report saved: {filepath}[/dim]")
    return str(filepath)
