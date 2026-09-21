"""
executor.py — Safely runs generated Python code with a retry loop.

WHAT IT DOES:
  Takes the LLM-generated code string and executes it.
  If it fails, passes the error back to the LLM to fix, then retries.
  This is the self-correcting loop — up to MAX_RETRIES attempts.

WHY SAFE EXECUTION MATTERS:
  We're running LLM-generated code — we don't know what it contains.
  We use exec() with a restricted namespace to limit what it can do:
  - It can use pandas, matplotlib, numpy, os, pathlib
  - It CANNOT import subprocess, socket, or other dangerous modules
  - We capture stdout so we can show the LLM's print() output

  In a real production system you'd run this in a Docker container.
  For this project, restricting the namespace is enough.

LINE BY LINE:
  @dataclass
  class ExecutionResult     → structured container for execution outcomes
    success: bool           → did the code run without errors?
    output: str             → everything the code printed (captured stdout)
    error: str              → error message if it failed, "" if success
    charts_saved: list      → paths to PNG files the code saved
    attempts: int           → how many tries it took

  def execute_with_retry()  → main function: code string → ExecutionResult
    for attempt in range(MAX_RETRIES + 1):
                            → range(4) = [0, 1, 2, 3] = 4 attempts total
                            → attempt 0 = first try (original code)
                            → attempts 1,2,3 = retries with fixes

    result = _execute_once()→ try running the code once
    if result.success:      → it worked → return immediately
    code = fix_code(...)    → it failed → ask LLM to fix it
                            → loop continues with fixed code

  def _execute_once()       → runs code once, captures output and errors
    io.StringIO()           → in-memory text buffer (like a fake file)
    redirect_stdout(buffer) → temporarily redirects print() to our buffer
                            → everything the code prints goes to buffer
                            → not to the real terminal
    contextlib.redirect_stdout() → context manager for capturing stdout

    namespace               → dict of variables the code can access
                            → like a restricted Python environment
    "pd": pd               → the code can use pandas as pd
    "plt": plt             → the code can use matplotlib
    "__builtins__": __builtins__
                            → allow standard Python built-ins (print, len, etc.)
    "OUTPUT_DIR": cfg.OUTPUT_DIR
                            → the code knows where to save charts

    exec(code, namespace)  → executes the code string in the namespace
                            → any variables the code creates go into namespace
                            → any errors become exceptions we catch

    except Exception as e  → catches ANY error the code raises
    traceback.format_exc() → formats the full traceback as a string
                            → this is what we send to the LLM to fix

    _find_charts()         → looks in OUTPUT_DIR for any new PNG files
                            → returns list of paths to saved chart files

  Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
                           → creates output directory if it doesn't exist
                           → parents=True: creates intermediate dirs too
                           → exist_ok=True: no error if already exists

  glob("*.png")            → finds all .png files in output directory
                           → glob = "global" pattern matching
                           → *.png = any filename ending in .png
"""

import io
import traceback
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from config import cfg
from code_generator import fix_code

console = Console()


@dataclass
class ExecutionResult:
    """Result of executing the generated analysis code."""
    success: bool
    output: str             # captured stdout (print statements)
    error: str = ""         # error message if failed
    charts_saved: List[str] = field(default_factory=list)
    attempts: int = 1       # how many attempts it took


def execute_with_retry(code: str) -> ExecutionResult:
    """
    Execute generated code with automatic error correction.

    Tries MAX_RETRIES+1 times:
      Attempt 0: original code
      Attempt 1: code fixed by LLM after seeing error from attempt 0
      Attempt 2: code fixed again after seeing error from attempt 1
      ...

    Args:
        code: Python code string from code_generator.py

    Returns:
        ExecutionResult with success status, output, charts, attempts.
    """
    # Ensure output directory exists before code tries to save charts
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    current_code = code

    for attempt in range(cfg.MAX_RETRIES + 1):
        console.print(f"[blue]Executing code (attempt {attempt + 1}/{cfg.MAX_RETRIES + 1})...[/blue]")

        result = _execute_once(current_code, attempt + 1)

        if result.success:
            console.print(f"[green]✓ Code executed successfully on attempt {attempt + 1}[/green]")
            return result

        # Code failed — show error
        console.print(f"[yellow]Attempt {attempt + 1} failed: {result.error[:100]}...[/yellow]")

        # If we have retries left, ask LLM to fix the code
        if attempt < cfg.MAX_RETRIES:
            console.print("[blue]Asking LLM to fix the error...[/blue]")
            current_code = fix_code(current_code, result.error)

            if not current_code:
                console.print("[red]LLM could not fix the code[/red]")
                return result

    # All retries exhausted
    console.print(f"[red]Code failed after {cfg.MAX_RETRIES + 1} attempts[/red]")
    return result


def _execute_once(code: str, attempt: int) -> ExecutionResult:
    """
    Execute the code once and capture output and errors.

    Uses exec() with a restricted namespace and captured stdout.

    Args:
        code:    Python code string to execute.
        attempt: Attempt number (for the result).

    Returns:
        ExecutionResult for this single attempt.
    """
    # Capture everything the code prints
    # StringIO() = in-memory text buffer (like a fake terminal)
    output_buffer = io.StringIO()

    # Restricted namespace: what the generated code can use
    # This is our safety layer — the code can't import dangerous modules
    # because it can only use what's in this dict
    namespace = {
        "pd":           pd,         # pandas
        "plt":          plt,        # matplotlib
        "np":           np,         # numpy
        "Path":         Path,       # pathlib.Path
        "OUTPUT_DIR":   cfg.OUTPUT_DIR,
        "__builtins__": __builtins__,  # allow: print, len, range, etc.
    }

    try:
        # Redirect all print() output to our buffer
        # contextlib.redirect_stdout = context manager that swaps sys.stdout
        # while inside the 'with' block, print() writes to output_buffer
        # when the block exits, sys.stdout is restored to the terminal
        with contextlib.redirect_stdout(output_buffer):
            exec(code, namespace)

        # Get everything the code printed
        captured_output = output_buffer.getvalue()

        # Find any chart PNG files the code saved
        charts = _find_charts()

        return ExecutionResult(
            success=True,
            output=captured_output,
            charts_saved=charts,
            attempts=attempt,
        )

    except Exception:
        # Get the full error traceback as a string
        # traceback.format_exc() formats the current exception
        error_text = traceback.format_exc()

        return ExecutionResult(
            success=False,
            output=output_buffer.getvalue(),  # partial output before crash
            error=error_text,
            attempts=attempt,
        )


def _find_charts() -> List[str]:
    """
    Find all PNG chart files in the output directory.

    Returns list of absolute file paths as strings.
    glob("*.png") = find all files matching *.png pattern
    """
    output_dir = Path(cfg.OUTPUT_DIR)
    if not output_dir.exists():
        return []

    # sorted() = alphabetical order (chart_1.png, chart_2.png, ...)
    charts = sorted([str(p) for p in output_dir.glob("*.png")])
    return charts
