"""
tester.py — Runs tests for the generated code.

WHAT IT DOES:
  1. Finds test files (test_*.py or *_test.py) in the project
  2. Runs them with pytest (Python) or node (JavaScript)
  3. Captures the output and pass/fail status
  4. Returns TestResult objects for each test file

WHY RUN REAL TESTS:
  LLM-generated code often has bugs.
  Running actual tests gives us a ground truth: pass or fail.
  This is the feedback signal the debugger uses to fix code.

LINE BY LINE:
  subprocess.run(...)    → runs a terminal command from Python
    ["pytest", filepath] → runs: pytest test_main.py
    capture_output=True  → captures stdout and stderr
    cwd=project_dir      → runs the command IN the project directory
                           so relative imports work correctly
    timeout=30           → kill the process after 30 seconds
                           (prevents infinite loops in generated code)
    text=True            → decode output as text (not bytes)

  result.returncode      → 0 = success, non-zero = failure
                           pytest returns 0 if all tests pass
                           pytest returns 1 if any test fails

  result.stdout          → everything printed to stdout (test output)
  result.stderr          → error messages (import errors, syntax errors)

  _find_test_files()     → finds files matching test_*.py pattern
    project_dir.glob("test_*.py")
                         → glob finds all files matching the pattern
                         → test_main.py, test_utils.py etc.
    project_dir.glob("*_test.py")
                         → also finds: main_test.py style

  subprocess.TimeoutExpired
                         → caught when test takes > 30 seconds
                         → returns failed TestResult with timeout message

  FileNotFoundError      → caught when pytest isn't installed
                         → returns failed TestResult with install instructions

  returncode == 0        → test passed
  any(r.passed ...)      → True if at least one TestResult passed
"""

import subprocess
from pathlib import Path
from typing import List
from rich.console import Console
from models import TestResult, ProjectPlan
from config import cfg

console = Console()

TEST_TIMEOUT = 30   # seconds before killing a test run


def run_tests(plan: ProjectPlan) -> List[TestResult]:
    """
    Run all test files in the project and return results.

    Args:
        plan: ProjectPlan with project_name to find the workspace folder.

    Returns:
        List of TestResult — one per test file found.
        Empty list if no test files found.
    """
    project_dir = Path(cfg.WORKSPACE_DIR) / plan.project_name
    test_files = _find_test_files(project_dir)

    if not test_files:
        console.print("[yellow]No test files found[/yellow]")
        return []

    console.print(f"[blue]Running {len(test_files)} test file(s)...[/blue]")

    results = []
    for test_file in test_files:
        console.print(f"[blue]  Running: {test_file.name}[/blue]")
        result = _run_single_test(test_file, project_dir)
        results.append(result)

        # Show result
        if result.passed:
            console.print(f"  [green]✓ {test_file.name} — PASSED[/green]")
        else:
            console.print(f"  [red]✗ {test_file.name} — FAILED[/red]")
            if result.error:
                console.print(f"  [dim red]{result.error[:200]}[/dim red]")

    passed = sum(1 for r in results if r.passed)
    console.print(
        f"[{'green' if passed == len(results) else 'yellow'}]"
        f"Tests: {passed}/{len(results)} passed[/]"
    )

    return results


def _run_single_test(test_file: Path, project_dir: Path) -> TestResult:
    """
    Run one test file and return a TestResult.

    Determines the test runner from the file extension:
    - .py  → pytest
    - .js  → node (basic) or jest (if available)

    Uses subprocess.run() to execute the command and capture output.
    """
    filename = test_file.name

    if test_file.suffix == ".py":
        cmd = ["python", "-m", "pytest", test_file.name, "-v", "--tb=short"]
    elif test_file.suffix in {".js", ".ts"}:
        cmd = ["node", str(test_file)]
    else:
        return TestResult(
            filename=filename,
            passed=False,
            error=f"Unsupported test file type: {test_file.suffix}",
        )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,   # capture stdout and stderr
            cwd=str(project_dir),  # run from the project folder
            timeout=TEST_TIMEOUT,  # kill after 30 seconds
            text=True,             # decode output as UTF-8 text
        )

        # pytest returns 0 = all tests passed, non-zero = failures/errors
        passed = result.returncode == 0
        output = result.stdout + result.stderr

        return TestResult(
            filename=filename,
            passed=passed,
            output=output,
            error="" if passed else output[-500:],  # last 500 chars of output
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            filename=filename,
            passed=False,
            error=f"Test timed out after {TEST_TIMEOUT}s — possible infinite loop",
        )

    except FileNotFoundError:
        return TestResult(
            filename=filename,
            passed=False,
            error="pytest not found. Install with: pip install pytest",
        )


def _find_test_files(project_dir: Path) -> List[Path]:
    """
    Find all test files in the project directory.

    Looks for:
    - test_*.py    (pytest convention, most common)
    - *_test.py    (alternative pytest convention)
    - test_*.js    (JavaScript)

    Returns list of Path objects, sorted alphabetically.
    """
    patterns = ["test_*.py", "*_test.py", "test_*.js", "*_test.js"]
    test_files = []

    for pattern in patterns:
        # glob() returns a generator — list() materialises it
        test_files.extend(project_dir.glob(pattern))

    # Remove duplicates (a file could match multiple patterns)
    # sorted() for consistent alphabetical order
    return sorted(set(test_files))
