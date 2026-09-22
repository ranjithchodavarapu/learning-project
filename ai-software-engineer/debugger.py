"""
debugger.py — Reads test failures and fixes the code.

WHAT IT DOES:
  1. Reads test results from tester.py
  2. For each failing test, finds the corresponding source file
  3. Sends: original code + test file + error output → LLM
  4. LLM returns fixed code
  5. Saves fixed code to disk
  6. Returns the count of debug attempts

SELF-CORRECTING LOOP:
  The pipeline calls debugger → tester → debugger → tester
  until either all tests pass OR MAX_DEBUG_ATTEMPTS is reached.
  This is the same loop as project 3 but at file level not code level.

LINE BY LINE:
  SYSTEM_PROMPT          → "expert debugging assistant"
                           "fix the code so tests pass" = clear objective
                           "return ONLY the fixed code" = no explanation

  FIX_PROMPT             → template with:
    {task}               → original user task (context)
    {source_file}        → filename being fixed (e.g. "main.py")
    {source_code}        → current broken code
    {test_file}          → test filename (e.g. "test_main.py")
    {test_code}          → the actual test code (what must pass)
    {error_output}       → pytest error output (what's failing)

  _find_source_for_test()→ maps test file → source file
    "test_main.py" → "main.py"
    "test_utils.py" → "utils.py"
    strips "test_" prefix to find the source file name

  _read_file()           → reads a file from disk as a string
    open(path, "r")      → "r" = read mode
    f.read()             → reads entire file as one string
    returns "" if file doesn't exist (defensive)

  _save_fixed_code()     → overwrites the source file with fixed code
    open(path, "w")      → "w" = write mode (overwrites existing content)
    f.write(fixed_code)  → writes the entire fixed code string

  plan.files             → list of FileSpec objects from the plan
    for f in plan.files: → iterate all files
    if f.filename == source_filename:
                         → find the matching FileSpec
    f.content = fixed_code
                         → update the in-memory content too
                           (so later stages have the latest code)
"""

from pathlib import Path
from rich.console import Console
from models import ProjectPlan, TestResult
from llm import call_llm_code
from config import cfg

console = Console()


SYSTEM_PROMPT = """You are an expert debugging assistant.
You will be given source code, its test file, and the test failure output.
Fix the source code so that all tests pass.
Return ONLY the fixed source code — no explanations, no markdown fences."""


FIX_PROMPT = """Fix this code so the tests pass.

Task context: {task}
Source file: {source_file}

Current source code:
{source_code}

Test file ({test_file}):
{test_code}

Test failure output:
{error_output}

Analyse the errors and fix the source code.
Return ONLY the fixed {source_file} code."""


def debug_and_fix(
    plan: ProjectPlan,
    test_results: list,
) -> tuple[ProjectPlan, int]:
    """
    Fix code for all failing tests.

    For each failing TestResult:
    1. Find the source file that the test is testing
    2. Read source code + test code from disk
    3. Ask LLM to fix the source code
    4. Save fixed code to disk
    5. Update plan.files

    Args:
        plan:         ProjectPlan with project info.
        test_results: List of TestResult from tester.py.

    Returns:
        Tuple of (updated_plan, fixes_applied_count).
    """
    project_dir = Path(cfg.WORKSPACE_DIR) / plan.project_name
    fixes_applied = 0

    # Only process FAILING tests
    failing = [r for r in test_results if not r.passed]

    if not failing:
        console.print("[green]All tests passing — nothing to debug[/green]")
        return plan, 0

    console.print(f"[yellow]Debugging {len(failing)} failing test(s)...[/yellow]")

    for test_result in failing:
        console.print(f"[yellow]  Fixing for: {test_result.filename}[/yellow]")

        # Fix ALL non-test source files — error could be in any of them
        source_files = [
            f.filename for f in plan.files
            if not f.filename.startswith("test_")
            and "_test." not in f.filename
        ]

        test_path = project_dir / test_result.filename
        test_code = _read_file(test_path)

        for source_filename in source_files:
            source_path = project_dir / source_filename
            source_code = _read_file(source_path)

            if not source_code:
                continue

            user_prompt = FIX_PROMPT.format(
                task=plan.task,
                source_file=source_filename,
                source_code=source_code,
                test_file=test_result.filename,
                test_code=test_code,
                error_output=test_result.error[:2000],
            )

            fixed_code = call_llm_code(
                SYSTEM_PROMPT, user_prompt, temperature=0.1,
            )

            if not fixed_code:
                console.print(f"[red]  LLM failed to fix {source_filename}[/red]")
                continue

            _save_fixed_code(source_path, fixed_code)

            for f in plan.files:
                if f.filename == source_filename:
                    f.content = fixed_code
                    break

            fixes_applied += 1
            console.print(f"  [green]✓ Fixed {source_filename}[/green]")

    return plan, fixes_applied


def _find_source_for_test(test_filename: str, plan: ProjectPlan) -> str:
    """
    Map a test filename to its source filename.

    Strategies (tried in order):
    1. Strip "test_" prefix: "test_main.py" → "main.py"
    2. Strip "_test" suffix: "main_test.py" → "main.py"
    3. Look for any non-test file in the plan

    Returns source filename string, or "" if not found.
    """
    # Strategy 1: test_main.py → main.py
    if test_filename.startswith("test_"):
        candidate = test_filename[5:]   # remove "test_" (5 chars)
        for f in plan.files:
            if f.filename == candidate:
                return candidate

    # Strategy 2: main_test.py → main.py
    stem = Path(test_filename).stem   # "main_test" from "main_test.py"
    ext  = Path(test_filename).suffix  # ".py"
    if stem.endswith("_test"):
        candidate = stem[:-5] + ext   # remove "_test"
        for f in plan.files:
            if f.filename == candidate:
                return candidate

    # Strategy 3: return first non-test file
    for f in plan.files:
        if not f.filename.startswith("test_") and "_test." not in f.filename:
            return f.filename

    return ""


def _read_file(path: Path) -> str:
    """Read file contents as a string. Returns "" if file doesn't exist."""
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _save_fixed_code(path: Path, code: str) -> None:
    """Overwrite a file with fixed code."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
