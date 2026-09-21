"""
Tests for AI Data Analyst. Run: pytest tests/ -v
No Ollama or real CSV needed — tests logic only.
"""

import pytest
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_profiler import DataProfile, profile_to_prompt
from code_generator import _clean_code
from executor import ExecutionResult, _find_charts


# ── DataProfile tests ─────────────────────────────────────────────────────

def make_profile():
    return DataProfile(
        path="test.csv",
        shape=(100, 3),
        columns=["age", "salary", "name"],
        dtypes={"age": "int64", "salary": "float64", "name": "object"},
        sample="age  salary  name\n 25  50000   Alice",
        stats="       age  salary\ncount  100  100",
        nulls={"age": 0, "salary": 5, "name": 0},
    )


def test_profile_to_prompt_contains_columns():
    profile = make_profile()
    prompt = profile_to_prompt(profile)
    assert "age" in prompt
    assert "salary" in prompt
    assert "name" in prompt


def test_profile_to_prompt_shows_nulls():
    profile = make_profile()
    prompt = profile_to_prompt(profile)
    # salary has 5 nulls — should appear in the null section
    assert "salary" in prompt
    assert "5" in prompt


def test_profile_to_prompt_shows_shape():
    profile = make_profile()
    prompt = profile_to_prompt(profile)
    assert "100" in prompt   # row count
    assert "3" in prompt     # column count


def test_profile_no_nulls_shows_none():
    profile = make_profile()
    profile.nulls = {"age": 0, "salary": 0, "name": 0}
    prompt = profile_to_prompt(profile)
    assert "None" in prompt


# ── Code cleaning tests ───────────────────────────────────────────────────

def test_clean_code_removes_python_fence():
    raw = "```python\nimport pandas as pd\ndf = pd.read_csv('x.csv')\n```"
    cleaned = _clean_code(raw)
    assert cleaned == "import pandas as pd\ndf = pd.read_csv('x.csv')"


def test_clean_code_removes_plain_fence():
    raw = "```\nimport pandas\n```"
    cleaned = _clean_code(raw)
    assert cleaned == "import pandas"


def test_clean_code_leaves_clean_code_alone():
    raw = "import pandas as pd\ndf = pd.read_csv('x.csv')"
    cleaned = _clean_code(raw)
    assert cleaned == raw


def test_clean_code_strips_whitespace():
    raw = "  \nimport pandas\n  "
    cleaned = _clean_code(raw)
    assert cleaned == "import pandas"


# ── ExecutionResult tests ─────────────────────────────────────────────────

def test_execution_result_success():
    r = ExecutionResult(success=True, output="hello", charts_saved=["a.png"])
    assert r.success is True
    assert r.output == "hello"
    assert len(r.charts_saved) == 1


def test_execution_result_failure():
    r = ExecutionResult(success=False, output="", error="NameError: x")
    assert r.success is False
    assert "NameError" in r.error


def test_execute_simple_code():
    """Test that simple safe code executes correctly."""
    from executor import _execute_once
    code = "x = 1 + 1\nprint(f'Result: {x}')"
    result = _execute_once(code, attempt=1)
    assert result.success is True
    assert "Result: 2" in result.output


def test_execute_bad_code_returns_error():
    """Test that bad code returns failure with error message."""
    from executor import _execute_once
    code = "raise ValueError('test error')"
    result = _execute_once(code, attempt=1)
    assert result.success is False
    assert "ValueError" in result.error


def test_execute_captures_stdout():
    """Test that print() output is captured."""
    from executor import _execute_once
    code = "print('hello world')\nprint('second line')"
    result = _execute_once(code, attempt=1)
    assert result.success is True
    assert "hello world" in result.output
    assert "second line" in result.output
