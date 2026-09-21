"""
code_generator.py — LLM writes Python code to analyse the data.

WHAT IT DOES:
  Takes the data profile and asks the LLM to write a complete
  Python script using pandas and matplotlib that:
  1. Loads the CSV
  2. Analyses it (aggregations, correlations, distributions)
  3. Creates charts (saved as PNG files)
  4. Prints summary statistics

WHY LLM-GENERATED CODE:
  Every CSV is different — different columns, different types, different questions.
  You can't write one fixed script that works for all datasets.
  Instead, the LLM reads the schema and writes custom code for THAT dataset.

  This is called "code-as-tool" — using LLM to generate executable code
  rather than just text answers.

SELF-CORRECTING LOOP:
  If the generated code throws an error when executed, we pass the
  error message back to the LLM and ask it to fix the code.
  This is the retry loop (MAX_RETRIES times).

LINE BY LINE:
  SYSTEM_PROMPT          → tells LLM its role: Python data analyst
                           "respond with ONLY Python code" = no markdown, no explanation
                           this is critical — the executor runs the code directly
                           if the LLM adds "Here's the code:" before it, exec() will fail

  CODE_TEMPLATE          → tells LLM exactly what the code must do:
                           1. import pandas and matplotlib
                           2. load the specific CSV path
                           3. do analysis
                           4. save charts to OUTPUT_DIR
                           5. print findings

  def generate_code()    → main function: profile → code string
    profile_to_prompt()  → formats the data profile as context for the LLM
    _call_llm()          → sends prompt to Ollama, gets code back
    _clean_code()        → strips markdown fences if LLM added them

  def fix_code()         → called when code throws an error
                           passes the original code + error message to LLM
                           LLM returns fixed code

  _call_llm()            → HTTP POST to Ollama
    requests.post(...)   → same pattern as multi-agent project
    resp.json()[...]     → extracts the text from Ollama response

  _clean_code()          → defensive parsing
    .lstrip("```python") → removes opening markdown fence if present
    .lstrip("```")       → removes plain opening fence
    .rstrip("```")       → removes closing fence
    .strip()             → removes leading/trailing whitespace
    WHY: LLMs sometimes wrap code in ```python ... ``` even when told not to
         The executor needs raw Python, not markdown-wrapped code

  temperature=0.2        → low temperature for code generation
                           we want precise, correct code not creative variation
                           0.0 would be fully deterministic but sometimes too rigid
                           0.2 allows slight variation while staying focused
"""

import requests
from rich.console import Console
from data_profiler import DataProfile, profile_to_prompt
from config import cfg

console = Console()


SYSTEM_PROMPT = """You are an expert Python data analyst.
Given a dataset profile, write a complete Python script that analyses the data.
Respond with ONLY executable Python code — no explanations, no markdown fences, no comments outside the code.
The code must be complete and runnable as-is."""


CODE_TEMPLATE = """Write a complete Python data analysis script for this dataset.

The script MUST:
1. Import pandas as pd and matplotlib.pyplot as plt
2. Load the CSV from: {csv_path}
3. Perform at least 3 of these analyses (choose what fits the data):
   - Distribution of numeric columns (histogram)
   - Correlation between numeric columns (scatter or heatmap)
   - Value counts of categorical columns (bar chart)
   - Time series trends if date column exists (line chart)
   - Top/bottom N rows by a numeric column (bar chart)
   - Group by a category and aggregate a numeric column (grouped bar)
4. Save each chart to: {output_dir}/chart_N.png (N = 1, 2, 3...)
   Use: plt.savefig('{output_dir}/chart_N.png', dpi=100, bbox_inches='tight')
   After saving each chart: plt.close()
5. Print key findings as text (averages, totals, top values etc.)

{data_profile}

Write ONLY the Python code. No explanations. No markdown."""


FIX_TEMPLATE = """The following Python code raised an error. Fix it.

ORIGINAL CODE:
{code}

ERROR:
{error}

Write ONLY the fixed Python code. No explanations. No markdown."""


def generate_code(profile: DataProfile) -> str:
    """
    Ask the LLM to write Python analysis code for this dataset.

    Args:
        profile: DataProfile from data_profiler.py

    Returns:
        Python code as a plain string (ready to execute).
    """
    console.print("[blue]Generating analysis code...[/blue]")

    # Build the prompt: template filled with actual values
    user_prompt = CODE_TEMPLATE.format(
        csv_path=profile.path,
        output_dir=cfg.OUTPUT_DIR,
        data_profile=profile_to_prompt(profile),
    )

    code = _call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.2)

    if not code:
        console.print("[red]LLM failed to generate code[/red]")
        return ""

    # Strip markdown fences if LLM added them despite instructions
    code = _clean_code(code)
    console.print(f"[green]✓ Generated {len(code.splitlines())} lines of code[/green]")
    return code


def fix_code(code: str, error: str) -> str:
    """
    Ask the LLM to fix code that threw an error.

    This is the self-correcting loop — called by executor.py
    when the generated code fails.

    Args:
        code:  The Python code that failed.
        error: The error message/traceback.

    Returns:
        Fixed Python code as a string.
    """
    console.print(f"[yellow]Asking LLM to fix error: {error[:80]}...[/yellow]")

    user_prompt = FIX_TEMPLATE.format(code=code, error=error)

    fixed = _call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.1)

    if not fixed:
        console.print("[red]LLM failed to fix code[/red]")
        return code   # return original if fix failed

    return _clean_code(fixed)


def _call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    """
    Call Ollama and return the response text.

    Same HTTP POST pattern as the multi-agent project.
    Returns empty string on any error.
    """
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
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

    except requests.exceptions.ConnectionError:
        console.print("[red]Cannot connect to Ollama. Run: ollama serve &[/red]")
        return ""
    except requests.exceptions.Timeout:
        console.print(f"[red]LLM call timed out after {cfg.TIMEOUT}s[/red]")
        return ""
    except Exception as e:
        console.print(f"[red]LLM error: {e}[/red]")
        return ""


def _clean_code(code: str) -> str:
    """
    Strip markdown code fences if LLM added them.

    LLMs sometimes wrap code in ```python ... ``` despite being told not to.
    The executor needs raw Python — markdown fences cause SyntaxError.

    Input:  "```python\\nimport pandas\\n```"
    Output: "import pandas"
    """
    code = code.strip()

    # Remove opening fence variants
    for fence in ["```python", "```py", "```"]:
        if code.startswith(fence):
            code = code[len(fence):]
            break

    # Remove closing fence
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()
