"""
data_profiler.py — Reads a CSV and extracts its schema + statistics.

WHAT IT DOES:
  Before the LLM can write code, it needs to understand the data:
  - What columns exist and what type they are (int, float, string, date)
  - How many rows there are
  - What the data looks like (sample rows)
  - Basic statistics (min, max, mean, nulls)

  This profile is passed to the LLM as context so it writes
  correct pandas code — right column names, right dtypes.

LINE BY LINE:
  import pandas as pd    → pandas is the standard Python data library
                           pd is the universal alias (convention)

  @dataclass             → auto-generates __init__, __repr__ etc.
  class DataProfile      → structured container for all profile info
    path: str            → original file path
    shape: tuple         → (rows, columns) e.g. (1000, 8)
    columns: list        → list of column name strings
    dtypes: dict         → {"age": "int64", "name": "object", ...}
    sample: str          → first N rows as a formatted string table
    stats: str           → describe() output as string
    nulls: dict          → {"age": 0, "salary": 23} — null count per column

  def profile_csv(path)  → main function, takes file path, returns DataProfile
    pd.read_csv(path)    → reads the CSV into a DataFrame
                           DataFrame = table with rows and columns in memory

  df.shape               → tuple: (number_of_rows, number_of_columns)
  df.columns.tolist()    → converts pandas Index to plain Python list
  df.dtypes.astype(str).to_dict()
                         → dtypes is a Series like: age → int64
                           .astype(str) converts each dtype to string
                           .to_dict() → {"age": "int64", "name": "object"}

  df.head(cfg.SAMPLE_ROWS)
                         → first N rows as DataFrame
  .to_string(index=False)→ formats as text table WITHOUT the row numbers
                           index=False removes the 0,1,2... column on the left

  df.describe(include="all")
                         → computes statistics for ALL columns
                           numeric: count, mean, std, min, 25%, 50%, 75%, max
                           text: count, unique, top, freq
  .to_string()           → converts the stats table to a text string

  df.isnull().sum()      → isnull() → True/False for each cell
                           .sum() → counts True per column = null count
  .to_dict()             → {"age": 0, "salary": 23}

  def profile_to_prompt(profile)
                         → formats the profile as a string the LLM can read
                           the LLM reads this and uses it to write pandas code
  f-string multiline     → triple-quote f-strings allow newlines inside
  \\n.join(...)          → joins list items with newlines between them
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from rich.console import Console
from rich.table import Table
from config import cfg

console = Console()


@dataclass
class DataProfile:
    """
    Complete profile of a CSV file.
    Passed to the LLM so it understands the data before writing code.
    """
    path: str                  # original CSV file path
    shape: Tuple[int, int]     # (rows, columns)
    columns: List[str]         # column names
    dtypes: Dict[str, str]     # column name → data type
    sample: str                # first N rows as text table
    stats: str                 # describe() output as text
    nulls: Dict[str, int]      # null count per column


def profile_csv(path: str) -> DataProfile:
    """
    Read a CSV file and extract its complete profile.

    Args:
        path: Path to the CSV file.

    Returns:
        DataProfile with all metadata the LLM needs.
    """
    path = str(path)
    console.print(f"[blue]Profiling: {path}[/blue]")

    # Load the CSV into a pandas DataFrame
    # sep="," is the default but explicit is better
    # low_memory=False prevents mixed-type warnings on large files
    df = pd.read_csv(path, sep=",", low_memory=False)

    console.print(f"[green]✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns[/green]")

    # Extract profile components
    shape   = df.shape
    columns = df.columns.tolist()
    dtypes  = df.dtypes.astype(str).to_dict()

    # Sample rows as a readable text table (no row index numbers)
    sample = df.head(cfg.SAMPLE_ROWS).to_string(index=False)

    # Descriptive statistics for all columns
    # include="all" covers both numeric and text columns
    stats = df.describe(include="all").to_string()

    # Count nulls per column — important for the LLM to know
    # (it might need to handle nulls in generated code)
    nulls = df.isnull().sum().to_dict()

    # Pretty-print a summary table to terminal
    _print_summary(path, shape, columns, dtypes, nulls)

    return DataProfile(
        path=path,
        shape=shape,
        columns=columns,
        dtypes=dtypes,
        sample=sample,
        stats=stats,
        nulls=nulls,
    )


def profile_to_prompt(profile: DataProfile) -> str:
    """
    Format the data profile as a string for the LLM.

    This string is included in the LLM prompt so it knows:
    - What columns to use and their types
    - The data shape (how many rows)
    - What the data looks like (sample)
    - Where nulls are (needs handling)

    Returns:
        Formatted multi-line string.
    """
    # Format null info — only show columns WITH nulls (cleaner prompt)
    null_info = {col: count for col, count in profile.nulls.items() if count > 0}

    null_text = (
        "\n".join(f"  {col}: {count} nulls" for col, count in null_info.items())
        if null_info
        else "  None"
    )

    # Format column types as a readable list
    col_types = "\n".join(
        f"  {col}: {dtype}" for col, dtype in profile.dtypes.items()
    )

    return f"""DATA PROFILE
============
File: {profile.path}
Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns

COLUMNS AND TYPES:
{col_types}

COLUMNS WITH NULL VALUES:
{null_text}

SAMPLE ROWS (first {cfg.SAMPLE_ROWS}):
{profile.sample}

STATISTICS:
{profile.stats}
"""


def _print_summary(path, shape, columns, dtypes, nulls) -> None:
    """Print a nice summary table to the terminal."""
    table = Table(title=f"Data Profile: {Path(path).name}")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Nulls", style="yellow")

    for col in columns:
        null_count = nulls.get(col, 0)
        null_str = str(null_count) if null_count > 0 else "-"
        table.add_row(col, dtypes.get(col, "?"), null_str)

    console.print(table)
    console.print(f"[dim]Total: {shape[0]:,} rows, {shape[1]} columns[/dim]")
