"""
coder.py — Generates code for each file in the project plan.

WHAT IT DOES:
  Takes a ProjectPlan (list of FileSpec objects) and for each file:
  1. Builds a prompt with the file's description + context about other files
  2. Calls the LLM to write the code
  3. Saves the code to disk in the workspace folder
  4. Updates the FileSpec.content field

WHY ONE FILE AT A TIME:
  Generating all files in one LLM call results in confusion —
  the LLM loses track of which code belongs to which file.
  One call per file = focused, clean output per file.

WHY CONTEXT ABOUT OTHER FILES:
  File A might need to import from File B.
  The LLM needs to know what functions/classes exist in other files
  to write correct import statements and function calls.

LINE BY LINE:
  SYSTEM_PROMPT          → "expert software developer"
                           "respond with ONLY code" → no markdown, no explanation
                           this is critical — the tester runs code directly

  CODE_PROMPT            → template filled with:
    {task}               → original user task
    {filename}           → which file to write ("main.py")
    {description}        → what this file should contain
    {language}           → "python", "javascript" etc.
    {other_files}        → list of other files with their descriptions
                           helps LLM write correct imports

  call_llm_code()        → calls LLM and strips markdown fences
                           from llm.py (centralised LLM module)

  _save_file()           → writes code to disk
    project_dir          → workspace/{project_name}/
    Path(project_dir).mkdir(parents=True, exist_ok=True)
                         → creates the folder if it doesn't exist
    with open(filepath, "w") → writes the code string to a .py file

  _build_context()       → formats other files as context string
    for each file in plan.files:
      if f.filename != current_filename:
        show the filename and description
    LLM uses this to know what to import

  code.splitlines()      → count lines for the log message
  f"[green]✓ Generated {len(code.splitlines())} lines[/green]"
                         → shows how much code was written
"""

from pathlib import Path
from rich.console import Console
from models import FileSpec, ProjectPlan
from llm import call_llm_code
from config import cfg

console = Console()


SYSTEM_PROMPT = """You are an expert software developer.
Write complete, working, production-quality code.
Respond with ONLY the code — no explanations, no markdown fences, no comments outside the code.
The code must be complete and immediately runnable."""


CODE_PROMPT = """Write the complete implementation for this file.

Overall task: {task}
File to write: {filename}
What this file must do: {description}
Language: {language}

Other files in this project (for import reference):
{other_files}

Requirements:
- Write complete, working code
- Include proper imports
- Add docstrings to functions
- Handle edge cases (empty input, invalid data etc.)
- Keep it simple — no external dependencies beyond standard library
- If this is a test file, use pytest and test the main functionality

Write ONLY the {language} code. No explanations. No markdown."""


def generate_code(plan: ProjectPlan) -> ProjectPlan:
    """
    Generate code for every file in the plan.

    Iterates over plan.files, generates code for each,
    saves to disk, and updates plan.files[i].content.

    Args:
        plan: ProjectPlan from planner.py.

    Returns:
        Updated ProjectPlan with .content filled in for every file.
    """
    # Create project directory in workspace
    project_dir = Path(cfg.WORKSPACE_DIR) / plan.project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Generating {len(plan.files)} files...[/blue]")

    for i, file_spec in enumerate(plan.files, 1):
        console.print(
            f"[blue]  [{i}/{len(plan.files)}] Writing {file_spec.filename}...[/blue]"
        )

        # Build context about other files (for import statements)
        other_files = _build_context(plan, file_spec.filename)

        # Build the prompt
        user_prompt = CODE_PROMPT.format(
            task=plan.task,
            filename=file_spec.filename,
            description=file_spec.description,
            language=file_spec.language,
            other_files=other_files,
        )

        # Generate code
        code = call_llm_code(SYSTEM_PROMPT, user_prompt, temperature=0.1)

        if not code:
            console.print(f"[yellow]LLM failed for {file_spec.filename}[/yellow]")
            code = f"# Generation failed for {file_spec.filename}\n"

        # Save to disk
        filepath = _save_file(project_dir, file_spec.filename, code)

        # Store in the FileSpec for later stages
        plan.files[i - 1].content = code

        console.print(
            f"[green]  ✓ {file_spec.filename} "
            f"({len(code.splitlines())} lines) → {filepath}[/green]"
        )

    return plan


def _build_context(plan: ProjectPlan, current_filename: str) -> str:
    """
    Build a description of all OTHER files in the project.

    The LLM uses this to write correct import statements.
    We skip the current file (it doesn't need to import itself).

    Returns:
        Multi-line string like:
        "- main.py: Flask app entry point with routes
         - models.py: TodoItem dataclass"
    """
    lines = []
    for f in plan.files:
        if f.filename != current_filename:
            lines.append(f"- {f.filename}: {f.description}")

    if not lines:
        return "No other files in this project."

    return "\n".join(lines)


def _save_file(project_dir: Path, filename: str, code: str) -> str:
    """
    Write code to a file in the project directory.

    Creates the file at: workspace/{project_name}/{filename}
    Returns the full file path as a string.
    """
    filepath = project_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    return str(filepath)
