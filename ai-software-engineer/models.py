"""
models.py — Shared data structures used across all agent files.

WHY A SEPARATE MODELS FILE:
  All 4 agents (planner, coder, tester, debugger) need to
  pass data between each other. Having one shared models file means:
  - No circular imports (A imports B imports A = crash)
  - One place to see all data shapes
  - Easy to add fields without editing multiple files

LINE BY LINE:
  @dataclass             → auto-generates __init__, __repr__, __eq__
  field(default_factory) → each instance gets a fresh list/dict
                           never write items: list = [] in a dataclass

  class FileSpec         → specification for ONE file to be generated
    filename: str        → e.g. "main.py", "utils.py", "api.py"
    description: str     → what this file should do (for the LLM prompt)
    language: str        → "python", "javascript" etc.
    content: str         → the actual generated code (filled by coder.py)

  class ProjectPlan      → the full plan from planner.py
    task: str            → original user task description
    project_name: str    → safe folder name derived from task
    language: str        → primary language for the project
    files: List[FileSpec]→ list of files to generate
    structure: str       → human-readable description of the architecture

  class TestResult       → result of running tests for one file
    filename: str        → which file was tested
    passed: bool         → did all tests pass?
    output: str          → full test runner output
    error: str           → error message if failed

  class ProjectResult    → final output of the whole pipeline
    plan: ProjectPlan    → the architecture plan
    files_written: list  → paths to files saved to disk
    test_results: list   → one TestResult per file
    all_tests_passed: bool → True only if every test passed
    workspace_path: str  → path to the project folder
    debug_attempts: int  → how many debug cycles were needed
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FileSpec:
    """Specification for a single file to be generated."""
    filename: str           # e.g. "main.py"
    description: str        # what this file should do
    language: str = "python"
    content: str = ""       # filled in by coder.py


@dataclass
class ProjectPlan:
    """Full project plan produced by planner.py."""
    task: str               # original user task
    project_name: str       # safe folder name
    language: str           # primary language
    files: List[FileSpec] = field(default_factory=list)
    structure: str = ""     # architecture description


@dataclass
class TestResult:
    """Result of running tests for one file."""
    filename: str
    passed: bool
    output: str = ""        # stdout from test runner
    error: str = ""         # error message if failed


@dataclass
class ProjectResult:
    """Final output of the full pipeline."""
    plan: ProjectPlan
    files_written: List[str] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    all_tests_passed: bool = False
    workspace_path: str = ""
    debug_attempts: int = 0
