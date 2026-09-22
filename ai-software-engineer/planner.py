"""
planner.py — Breaks a task description into a file-by-file plan.

WHAT IT DOES:
  Takes a task like "Build a REST API for a todo list"
  and produces a structured plan:
  - project name: "todo_api"
  - language: python
  - files to create:
      main.py      → Flask app entry point, routes
      models.py    → TodoItem dataclass
      storage.py   → in-memory storage layer
      test_api.py  → pytest tests for all endpoints

  This plan is passed to coder.py which generates each file.

WHY PLAN FIRST:
  Without a plan, the LLM writes one big monolithic file.
  With a plan, it writes focused single-responsibility files.
  Focused files = easier to test, debug, and extend.

LINE BY LINE:
  SYSTEM_PROMPT          → "senior software architect"
                           persona = structured, modular thinking
                           "respond in JSON" = machine-parseable output

  PLAN_PROMPT            → asks LLM for a specific JSON structure:
    {
      "project_name": "todo_api",
      "language": "python",
      "architecture": "Flask REST API...",
      "files": [
        {"filename": "main.py", "description": "..."},
        ...
      ]
    }

  call_llm() with "format": "json"
                           → we need structured output
                           → plain text would be hard to parse reliably

  _parse_plan()          → converts the JSON response into a ProjectPlan
    json.loads(raw)      → parses JSON string → Python dict
    data.get("files", [])→ .get() = safe access, returns [] if missing
    FileSpec(...)        → creates a FileSpec for each file entry

  _safe_name()           → converts task to a safe folder name
    "Build a REST API!"  → "build_a_rest_api"
    .lower()             → lowercase
    re.sub(r"[^a-z0-9]", "_", ...)
                         → replaces anything not a letter/number with _
    [:30]                → max 30 chars to keep folder names short
    .strip("_")          → removes leading/trailing underscores
"""

import json
import re
from rich.console import Console
from rich.panel import Panel
from models import FileSpec, ProjectPlan
from llm import call_llm
from config import cfg

console = Console()


SYSTEM_PROMPT = """You are a senior software architect.
Given a task, create a detailed project plan with specific files to implement.
Respond with ONLY valid JSON — no explanation, no markdown.
The JSON must have this exact structure:
{
  "project_name": "short_snake_case_name",
  "language": "python",
  "architecture": "brief description of the architecture",
  "files": [
    {"filename": "main.py", "description": "what this file does and contains"},
    {"filename": "utils.py", "description": "what this file does and contains"}
  ]
}"""


PLAN_PROMPT = """Create a project plan for this task:

Task: {task}
Language: {language}

Requirements:
- Use {language} as the primary language
- Break into 2-4 focused files (single responsibility)
- Include one test file named test_*.py (or test_*.js)
- Each file description must be detailed enough for a developer to implement it
- Keep the project simple and self-contained (no external services needed)
- project_name must be lowercase letters, numbers, and underscores only

Respond with ONLY the JSON object."""


def plan_task(task: str, language: str = "python") -> ProjectPlan:
    """
    Break a task description into a structured project plan.

    Args:
        task:     Natural language task description.
        language: Primary programming language.

    Returns:
        ProjectPlan with files to generate.
    """
    console.print(f"[blue]Planning: {task}[/blue]")

    user_prompt = PLAN_PROMPT.format(task=task, language=language)

    # Use higher temperature for planning — we want architectural thinking
    raw = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.3)

    if not raw:
        console.print("[yellow]LLM failed — using default plan[/yellow]")
        return _default_plan(task, language)

    plan = _parse_plan(raw, task, language)
    _display_plan(plan)
    return plan


def _parse_plan(raw: str, task: str, language: str) -> ProjectPlan:
    """
    Parse LLM JSON response into a ProjectPlan.

    Handles cases where LLM wraps JSON in markdown despite instructions.
    Falls back to default plan if parsing fails entirely.
    """
    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first line (```json or ```) and last line (```)
        clean = "\n".join(lines[1:-1])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        console.print(f"[yellow]JSON parse failed: {e} — using default plan[/yellow]")
        return _default_plan(task, language)

    # Extract files list — .get() returns [] if "files" key is missing
    files_data = data.get("files", [])
    files = []
    for f in files_data:
        files.append(FileSpec(
            filename=f.get("filename", "main.py"),
            description=f.get("description", "Main implementation file"),
            language=language,
        ))

    # Ensure we always have at least one file
    if not files:
        files = [FileSpec(
            filename="main.py",
            description="Main implementation",
            language=language,
        )]

    project_name = data.get("project_name", _safe_name(task))

    return ProjectPlan(
        task=task,
        project_name=project_name,
        language=language,
        files=files,
        structure=data.get("architecture", ""),
    )


def _default_plan(task: str, language: str) -> ProjectPlan:
    """
    Fallback plan when LLM fails — creates a basic two-file structure.
    Allows the pipeline to continue even if planning fails.
    """
    ext = "py" if language == "python" else "js"
    return ProjectPlan(
        task=task,
        project_name=_safe_name(task),
        language=language,
        files=[
            FileSpec("main." + ext, f"Main implementation for: {task}", language),
            FileSpec("test_main." + ext, f"Tests for main.{ext}", language),
        ],
        structure=f"Simple single-file implementation of: {task}",
    )


def _safe_name(task: str) -> str:
    """
    Convert task description to a safe folder/project name.

    "Build a REST API for todo list!" → "build_a_rest_api_for_todo_list"
    """
    name = task.lower()
    # Replace any character that is NOT a letter, digit, or space with underscore
    # [^a-z0-9 ] = anything not in this set
    name = re.sub(r"[^a-z0-9 ]", "", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove leading/trailing underscores, limit to 30 chars
    return name.strip("_")[:30]


def _display_plan(plan: ProjectPlan) -> None:
    """Pretty-print the plan to the terminal."""
    lines = [
        f"Project: {plan.project_name}",
        f"Language: {plan.language}",
        f"Architecture: {plan.structure}",
        "",
        "Files to generate:",
    ]
    for f in plan.files:
        lines.append(f"  • {f.filename} — {f.description[:60]}")

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Project Plan[/bold]",
        border_style="purple",
    ))
