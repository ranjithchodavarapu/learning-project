"""
Tests for the Self-Correcting AI Agent.
Run: pytest tests/ -v
No Ollama needed — tests logic and data structures only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import EvaluationScore, Attempt, AgentResult
from evaluator import _parse_scores, _neutral_score
from reflector import _get_low_dimensions, _fallback_critique
from agent import _select_best
from config import cfg


# ── EvaluationScore tests ─────────────────────────────────────────────────

def test_evaluation_score_passed_true():
    score = EvaluationScore(
        dimensions={"accuracy": 0.9, "clarity": 0.8},
        overall=0.85,
        passed=True,
    )
    assert score.passed is True


def test_evaluation_score_passed_false():
    score = EvaluationScore(
        dimensions={"accuracy": 0.5, "clarity": 0.4},
        overall=0.45,
        passed=False,
    )
    assert score.passed is False


def test_neutral_score():
    score = _neutral_score()
    assert score.overall == 0.5
    assert score.passed is False
    assert all(v == 0.5 for v in score.dimensions.values())


# ── Score parsing tests ───────────────────────────────────────────────────

def test_parse_scores_valid_json():
    raw = '{"scores": {"accuracy": 0.8, "completeness": 0.7, "clarity": 0.9, "conciseness": 0.6, "relevance": 0.8}, "feedback": "Good answer"}'
    score = _parse_scores(raw)
    assert score.dimensions["accuracy"] == 0.8
    assert score.feedback == "Good answer"
    assert 0.0 < score.overall < 1.0


def test_parse_scores_invalid_json():
    score = _parse_scores("not valid json")
    assert score.overall == 0.5   # falls back to neutral


def test_parse_scores_missing_dimension():
    raw = '{"scores": {"accuracy": 0.8}, "feedback": "ok"}'
    score = _parse_scores(raw)
    # Missing dimensions should default to 0.5
    for dim in cfg.EVALUATION_DIMENSIONS:
        assert dim in score.dimensions


def test_parse_scores_overall_is_mean():
    raw = '{"scores": {"accuracy": 0.8, "completeness": 0.6, "clarity": 0.7, "conciseness": 0.9, "relevance": 0.5}, "feedback": "ok"}'
    score = _parse_scores(raw)
    expected = (0.8 + 0.6 + 0.7 + 0.9 + 0.5) / 5
    assert abs(score.overall - expected) < 0.001


# ── Reflector tests ───────────────────────────────────────────────────────

def make_score(vals: dict) -> EvaluationScore:
    overall = sum(vals.values()) / len(vals)
    return EvaluationScore(
        dimensions=vals,
        overall=overall,
        passed=overall >= cfg.SCORE_THRESHOLD,
        feedback="test feedback",
    )


def test_get_low_dimensions_finds_low():
    score = make_score({"accuracy": 0.4, "clarity": 0.9, "completeness": 0.5})
    low = _get_low_dimensions(score)
    low_names = [d for d, _ in low]
    assert "accuracy" in low_names
    assert "completeness" in low_names
    assert "clarity" not in low_names


def test_get_low_dimensions_sorted_worst_first():
    score = make_score({"accuracy": 0.3, "clarity": 0.5, "completeness": 0.4})
    low = _get_low_dimensions(score)
    scores = [s for _, s in low]
    assert scores == sorted(scores)   # ascending (worst first)


def test_fallback_critique_mentions_low_dims():
    score = make_score({"accuracy": 0.3, "clarity": 0.9, "completeness": 0.4})
    critique = _fallback_critique(score)
    assert "accuracy" in critique or "completeness" in critique


# ── Agent selection tests ─────────────────────────────────────────────────

def make_attempt(num, overall):
    a = Attempt(attempt_number=num, question="test")
    a.score = EvaluationScore(overall=overall, passed=overall >= cfg.SCORE_THRESHOLD)
    return a


def test_select_best_picks_highest():
    attempts = [
        make_attempt(1, 0.5),
        make_attempt(2, 0.9),
        make_attempt(3, 0.7),
    ]
    best = _select_best(attempts)
    assert best.attempt_number == 2
    assert best.score.overall == 0.9


def test_select_best_single_attempt():
    attempts = [make_attempt(1, 0.6)]
    best = _select_best(attempts)
    assert best.attempt_number == 1


# ── AgentResult tests ─────────────────────────────────────────────────────

def test_agent_result_summary():
    score = EvaluationScore(overall=0.85, passed=True)
    result = AgentResult(
        question="test",
        final_score=score,
        total_attempts=2,
        converged=True,
    )
    summary = result.summary
    assert "2" in summary
    assert "0.85" in summary


def test_agent_result_score_history():
    a1 = make_attempt(1, 0.6)
    a2 = make_attempt(2, 0.8)
    result = AgentResult(
        question="test",
        attempts=[a1, a2],
        total_attempts=2,
    )
    history = result.score_history
    assert "Attempt 1" in history
    assert "Attempt 2" in history
