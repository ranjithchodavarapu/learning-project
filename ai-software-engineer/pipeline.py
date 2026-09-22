"""
pipeline.py — Chains all 4 agents: plan → code → test → debug.

WHAT IT DOES:
  Runs the full software engineering loop:
    1. Plan:  break task into files
    2. Code:  generate each file
    3. Test:  run tests
    4. Debug: fix failures → retest (up to MAX_DEBUG_ATTEMPTS)

THE DEBUG LOOP:
  After initial code generation:
    attempt 1: run tests
    if fail: debug → retest
    attempt 2: run tests
    if fail: debug → retest
    attempt 3: run tests → final result (pass or fail)

  This mirrors how real developers work:
  write → test → fix → test → fix → test → done

LINE BY LINE:
  class EngineerPipeline → wraps the whole workflow in one object
    def run(task, language)
                         → main entry point

  time.time()            → record start time
  elapsed = time.time() - start
                         → compute total time taken

  console.rule(...)      → prints "────── Step N ──────" divider

  test_results = run_tests(plan)
                         → runs pytest on all test_*.py files
                         → returns list of TestResult objects

  all_pass = all(r.passed for r in test_results)
                         → generator expression: checks if every result passed
                         → all() = True only if ALL elements are True
                         → if test_results is empty: all() returns True

  for attempt in range(cfg.MAX_DEBUG_ATTEMPTS):
                         → debug loop: try up to 3 times
                         → attempt 0, 1, 2

  if all_pass or not test_results:
      break              → exit the debug loop early if tests pass
                         → or if no tests exist (no point debugging)

  plan, fixes = debug_and_fix(plan, test_results)
                         → fix failing code
  test_results = run_tests(plan)
                         → rerun tests with fixed code

  _write_readme()        → generates a README.md for the project
                         → uses LLM to describe what was built
                         → saved alongside the generated code

  ProjectResult(...)     → bundles all outputs into one object
    files_written        → list of file paths on disk
    test_results         → final test results after all debug attempts
    all_tests_passed     → bool: did we end with passing tests?
    debug_attempts       → how many debug cycles ran
"""

import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from models import ProjectPlan, ProjectResult
from planner import plan_task
from coder import generate_code
from tester import run_tests
from debugger import debug_and_fix
from llm import call_llm
from config import cfg

console = Console()

README_PROMPT = """Write a README.md for this project.

Task: {task}
Project name: {project_name}
Language: {language}
Files: {files}
Architecture: {structure}

Include:
- Project title and description
- How to install/setup
- How to run
- How to run tests
- Brief description of each file

Write clean markdown."""


class EngineerPipeline:
    """
    Full AI Software Engineer pipeline.
    Plans, writes, tests, and debugs code for any task.
    """

    def run(self, task: str, language: str = "python") -> ProjectResult:
        """
        Build a complete project for the given task.

        Args:
            task:     Natural language task description.
            language: Primary language ("python", "javascript" etc.)

        Returns:
            ProjectResult with all files, test results, and workspace path.
        """
        start = time.time()

        console.print(Panel(
            f"[bold blue]AI Software Engineer Agent[/bold blue]\n"
            f"Task: [cyan]{task}[/cyan]\n"
            f"Language: [cyan]{language}[/cyan]",
            border_style="blue",
        ))

        # ── Step 1: Plan ──────────────────────────────────────────────────
        console.rule("[bold]Step 1 — Planning[/bold]")
        plan = plan_task(task, language)

        # ── Step 2: Generate code ─────────────────────────────────────────
        console.rule("[bold]Step 2 — Code Generation[/bold]")
        plan = generate_code(plan)

        # ── Step 3: Run tests ─────────────────────────────────────────────
        console.rule("[bold]Step 3 — Testing[/bold]")
        test_results = run_tests(plan)

        # ── Step 4: Debug loop ────────────────────────────────────────────
        total_debug_attempts = 0
        all_pass = all(r.passed for r in test_results) if test_results else True

        if not all_pass and test_results:
            console.rule("[bold]Step 4 — Debug Loop[/bold]")

            for attempt in range(cfg.MAX_DEBUG_ATTEMPTS):
                if all_pass:
                    break

                console.print(
                    f"[yellow]Debug attempt {attempt + 1}/{cfg.MAX_DEBUG_ATTEMPTS}[/yellow]"
                )

                # Fix failing code
                plan, fixes = debug_and_fix(plan, test_results)
                total_debug_attempts += 1

                if fixes == 0:
                    console.print("[red]No fixes applied — stopping debug loop[/red]")
                    break

                # Rerun tests with fixed code
                console.print("[blue]Re-running tests...[/blue]")
                test_results = run_tests(plan)
                all_pass = all(r.passed for r in test_results)

                if all_pass:
                    console.print("[green]✓ All tests passing after fixes![/green]")
                    break
        else:
            console.print("[dim]Step 4 skipped — tests already passing[/dim]")

        # ── Generate README ───────────────────────────────────────────────
        console.print("[blue]Generating README...[/blue]")
        _write_readme(plan)

        # ── Collect results ───────────────────────────────────────────────
        project_dir  = Path(cfg.WORKSPACE_DIR) / plan.project_name
        files_written = [str(project_dir / f.filename) for f in plan.files]
        files_written.append(str(project_dir / "README.md"))

        elapsed = time.time() - start

        # ── Display summary ───────────────────────────────────────────────
        console.rule("[bold green]Complete[/bold green]")
        status = "[green]✓ All tests passing[/green]" if all_pass else "[yellow]⚠ Some tests failing[/yellow]"
        console.print(Panel(
            f"{status}\n"
            f"Time: {elapsed:.0f}s\n"
            f"Files: {len(files_written)}\n"
            f"Debug cycles: {total_debug_attempts}\n"
            f"Workspace: {project_dir}",
            title="[bold]Summary[/bold]",
            border_style="green" if all_pass else "yellow",
        ))

        return ProjectResult(
            plan=plan,
            files_written=files_written,
            test_results=test_results,
            all_tests_passed=all_pass,
            workspace_path=str(project_dir),
            debug_attempts=total_debug_attempts,
        )


def _write_readme(plan: ProjectPlan) -> None:
    """Generate and save a README.md for the project."""
    files_list = "\n".join(f"- {f.filename}: {f.description}" for f in plan.files)

    user_prompt = README_PROMPT.format(
        task=plan.task,
        project_name=plan.project_name,
        language=plan.language,
        files=files_list,
        structure=plan.structure,
    )

    readme = call_llm(
        "You are a technical writer. Write clear, concise README files.",
        user_prompt,
        temperature=0.3,
    )

    if readme:
        project_dir = Path(cfg.WORKSPACE_DIR) / plan.project_name
        with open(project_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme)
        console.print("[dim]README.md written[/dim]")
