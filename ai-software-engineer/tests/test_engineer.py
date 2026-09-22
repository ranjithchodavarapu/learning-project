"""
Tests for the AI Software Engineer Agent.
Run: pytest tests/ -v
No Ollama needed — tests logic only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import FileSpec, ProjectPlan, TestResult, ProjectResult
from planner import _safe_name, _default_plan
from llm import _clean_code
from debugger import _find_source_for_test


# ── Model tests ───────────────────────────────────────────────────────────

def test_file_spec_defaults():
    f = FileSpec(filename="main.py", description="main file")
    assert f.language == "python"
    assert f.content == ""


def test_project_plan_empty_files():
    plan = ProjectPlan(task="test", project_name="test", language="python")
    assert plan.files == []


def test_test_result_default_error():
    r = TestResult(filename="test_main.py", passed=False)
    assert r.error == ""
    assert r.output == ""


def test_project_result_default():
    plan = ProjectPlan(task="t", project_name="t", language="python")
    r = ProjectResult(plan=plan)
    assert r.all_tests_passed is False
    assert r.debug_attempts == 0
    assert r.files_written == []


# ── Planner tests ─────────────────────────────────────────────────────────

def test_safe_name_basic():
    assert _safe_name("Build a calculator") == "build_a_calculator"


def test_safe_name_special_chars():
    result = _safe_name("Build REST API!")
    assert "!" not in result
    assert " " not in result


def test_safe_name_max_length():
    long_task = "a" * 100
    assert len(_safe_name(long_task)) <= 30


def test_safe_name_lowercase():
    result = _safe_name("BUILD A CALCULATOR")
    assert result == result.lower()


def test_default_plan_has_files():
    plan = _default_plan("Build a calculator", "python")
    assert len(plan.files) >= 1
    assert plan.language == "python"


def test_default_plan_has_test_file():
    plan = _default_plan("Build a calculator", "python")
    test_files = [f for f in plan.files if f.filename.startswith("test_")]
    assert len(test_files) >= 1


# ── LLM code cleaning tests ───────────────────────────────────────────────

def test_clean_python_fence():
    code = "```python\nimport os\n```"
    assert _clean_code(code) == "import os"


def test_clean_js_fence():
    code = "```javascript\nconsole.log('hi')\n```"
    assert _clean_code(code) == "console.log('hi')"


def test_clean_plain_fence():
    code = "```\nimport os\n```"
    assert _clean_code(code) == "import os"


def test_clean_no_fence():
    code = "import os\nprint('hello')"
    assert _clean_code(code) == code


def test_clean_strips_whitespace():
    code = "  \nimport os\n  "
    assert _clean_code(code) == "import os"


# ── Debugger tests ────────────────────────────────────────────────────────

def make_plan():
    return ProjectPlan(
        task="test task",
        project_name="test",
        language="python",
        files=[
            FileSpec("main.py", "main implementation"),
            FileSpec("utils.py", "utility functions"),
            FileSpec("test_main.py", "tests for main.py"),
        ],
    )


def test_find_source_test_prefix():
    plan = make_plan()
    result = _find_source_for_test("test_main.py", plan)
    assert result == "main.py"


def test_find_source_no_match_returns_first_non_test():
    plan = make_plan()
    # "test_unknown.py" won't match by name, falls back to first non-test file
    result = _find_source_for_test("test_unknown.py", plan)
    assert result in ["main.py", "utils.py"]


def test_find_source_empty_plan():
    plan = ProjectPlan(task="t", project_name="t", language="python")
    result = _find_source_for_test("test_main.py", plan)
    assert result == ""


# ── Integration: pipeline structure ──────────────────────────────────────

def test_project_result_stores_plan():
    plan = ProjectPlan(task="build calc", project_name="calc", language="python")
    result = ProjectResult(plan=plan, all_tests_passed=True)
    assert result.plan.task == "build calc"
    assert result.all_tests_passed is True
