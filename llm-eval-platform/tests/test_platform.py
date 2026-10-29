"""
Tests for the LLM Evaluation Platform.
Run: pytest tests/ -v
No Ollama needed — tests logic and scoring only.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import TestCase, LLMResponse, ScoreResult, EvalResult, EvalReport
from scorers import accuracy, latency, cost
from evaluator import load_test_suite


# ── Helpers ───────────────────────────────────────────────────────────────

def make_response(text, latency_ms=500, tokens=100):
    return LLMResponse(
        model="test-model", prompt="test", response=text,
        latency_ms=latency_ms, total_tokens=tokens,
        prompt_tokens=50, completion_tokens=50,
    )

def make_tc(prompt="test", expected="Paris"):
    return TestCase(prompt=prompt, expected=expected)


# ── Accuracy scorer ───────────────────────────────────────────────────────

def test_accuracy_exact_match():
    r = make_response("Paris")
    s = accuracy.score(r, make_tc(expected="Paris"))
    assert s.score == 1.0
    assert s.passed is True

def test_accuracy_case_insensitive():
    r = make_response("paris")
    s = accuracy.score(r, make_tc(expected="Paris"))
    assert s.score == 1.0

def test_accuracy_contains_match():
    r = make_response("The capital of France is Paris.")
    s = accuracy.score(r, make_tc(expected="Paris"))
    assert s.passed is True

def test_accuracy_wrong_answer():
    r = make_response("London")
    s = accuracy.score(r, make_tc(expected="Paris"))
    assert s.passed is False

def test_accuracy_fuzzy_similar():
    r = make_response("Central Processing Units")
    s = accuracy.score(r, make_tc(expected="Central Processing Unit"))
    assert s.score > 0.8


# ── Latency scorer ────────────────────────────────────────────────────────

def test_latency_fast_responses():
    responses = [make_response("x", latency_ms=500) for _ in range(10)]
    s = latency.score(responses)
    assert s.passed is True
    assert s.score > 0.5

def test_latency_slow_responses():
    responses = [make_response("x", latency_ms=8000) for _ in range(10)]
    s = latency.score(responses)
    assert s.passed is False

def test_latency_empty():
    s = latency.score([])
    assert s.passed is False

def test_latency_percentiles_computed():
    responses = [make_response("x", latency_ms=i*100) for i in range(1, 11)]
    s = latency.score(responses)
    assert "p50" in s.details
    assert "p95" in s.details

def test_latency_percentile_calculation():
    """p50 of [100, 200, 300, 400, 500] should be 300."""
    from scorers.latency import _percentile
    vals = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert _percentile(vals, 50) == 300.0


# ── Cost scorer ───────────────────────────────────────────────────────────

def test_cost_always_passes():
    responses = [make_response("x", tokens=5000) for _ in range(5)]
    s = cost.score(responses)
    assert s.passed is True   # cost is informational

def test_cost_tracks_total_tokens():
    responses = [make_response("x", tokens=100) for _ in range(5)]
    s = cost.score(responses)
    assert s.details["total_tokens"] == 500

def test_cost_avg_tokens():
    responses = [make_response("x", tokens=200) for _ in range(4)]
    s = cost.score(responses)
    assert s.details["avg_tokens_per_request"] == 200.0

def test_cost_zero_for_free_models():
    """With TOKEN_PRICE_PER_1K=0, estimated cost should be 0."""
    responses = [make_response("x", tokens=1000) for _ in range(3)]
    s = cost.score(responses)
    assert s.details["estimated_cost_usd"] == 0.0


# ── EvalReport ────────────────────────────────────────────────────────────

def make_eval_result(passed=True):
    tc = make_tc()
    resp = make_response("Paris")
    score_r = ScoreResult(scorer="accuracy", score=0.9, passed=passed)
    return EvalResult(
        test_case=tc, response=resp,
        scores={"accuracy": score_r},
        overall_passed=passed,
    )

def test_report_pass_rate():
    report = EvalReport(model="test", test_suite="general.json")
    report.results = [make_eval_result(True)] * 8 + [make_eval_result(False)] * 2
    assert report.pass_rate == 80.0

def test_report_passed_count():
    report = EvalReport(model="test", test_suite="general.json")
    report.results = [make_eval_result(True)] * 6 + [make_eval_result(False)] * 4
    assert report.passed_count == 6

def test_report_empty():
    report = EvalReport(model="test", test_suite="general.json")
    assert report.pass_rate == 0.0
    assert report.passed_count == 0


# ── Test suite loading ────────────────────────────────────────────────────

def test_load_test_suite(tmp_path):
    import json
    data = [
        {"prompt": "What is 2+2?", "expected": "4", "category": "math"},
        {"prompt": "Capital of France?", "expected": "Paris"},
    ]
    suite_file = tmp_path / "test.json"
    suite_file.write_text(json.dumps(data))

    cases = load_test_suite(str(suite_file))
    assert len(cases) == 2
    assert cases[0].prompt == "What is 2+2?"
    assert cases[1].category == "general"   # default category

def test_load_missing_suite():
    with pytest.raises(FileNotFoundError):
        load_test_suite("nonexistent_file.json")
