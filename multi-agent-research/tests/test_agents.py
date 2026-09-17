"""
Tests for multi-agent research system.
Run with: pytest tests/ -v
No Ollama needed — all LLM calls are tested via logic only.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import ResearchMemory
from agents.orchestrator import Orchestrator
from agents.researcher import Researcher
from agents.verifier import Verifier


# ── Memory tests ──────────────────────────────────────────────────────────

def test_memory_add_fact_no_duplicates():
    """add_fact() should not add the same fact twice."""
    mem = ResearchMemory(topic="AI")
    mem.add_fact("History", "AI was coined in 1956.")
    mem.add_fact("History", "AI was coined in 1956.")   # duplicate
    assert len(mem.facts) == 1


def test_memory_add_different_facts():
    """add_fact() should add different facts."""
    mem = ResearchMemory(topic="AI")
    mem.add_fact("History", "AI was coined in 1956.")
    mem.add_fact("History", "Deep learning emerged in 2010s.")
    assert len(mem.facts) == 2


def test_memory_to_dict():
    """to_dict() should return a plain dict with expected keys."""
    mem = ResearchMemory(topic="AI")
    d = mem.to_dict()
    assert "topic" in d
    assert "subtopics" in d
    assert "facts" in d
    assert d["topic"] == "AI"


def test_memory_get_summary():
    """get_summary() should return a non-empty string."""
    mem = ResearchMemory(topic="AI")
    summary = mem.get_summary()
    assert "AI" in summary
    assert "Facts found" in summary


def test_memory_status_default():
    """Memory status should start as idle."""
    mem = ResearchMemory()
    assert mem.status == "idle"


# ── Orchestrator tests ────────────────────────────────────────────────────

def test_orchestrator_parse_numbered_list():
    """_parse_subtopics() should parse numbered lists correctly."""
    orch = Orchestrator()
    raw = "1. History of AI\n2. Current state\n3. Future directions"
    result = orch._parse_subtopics(raw)
    assert len(result) == 3
    assert result[0] == "History of AI"
    assert result[1] == "Current state"
    assert result[2] == "Future directions"


def test_orchestrator_parse_handles_blank_lines():
    """_parse_subtopics() should skip blank lines."""
    orch = Orchestrator()
    raw = "1. Topic one\n\n2. Topic two\n\n3. Topic three"
    result = orch._parse_subtopics(raw)
    assert len(result) == 3


def test_orchestrator_respects_max_subtopics():
    """_parse_subtopics() should not exceed MAX_SUBTOPICS."""
    orch = Orchestrator()
    raw = "\n".join(f"{i}. Topic {i}" for i in range(1, 10))
    from config import cfg
    result = orch._parse_subtopics(raw)
    assert len(result) <= cfg.MAX_SUBTOPICS


# ── Researcher tests ──────────────────────────────────────────────────────

def test_researcher_parse_numbered_facts():
    """_parse_facts() should parse numbered fact lists."""
    res = Researcher()
    raw = "1. AI was invented in 1956.\n2. Deep learning uses neural networks.\n3. GPT uses transformers."
    facts = res._parse_facts(raw)
    assert len(facts) == 3
    assert "AI was invented in 1956." in facts


def test_researcher_parse_bullet_points():
    """_parse_facts() should handle bullet point format too."""
    res = Researcher()
    raw = "- First fact here.\n- Second fact here.\n- Third fact here."
    facts = res._parse_facts(raw)
    assert len(facts) == 3


def test_researcher_parse_empty():
    """_parse_facts() should handle empty input gracefully."""
    res = Researcher()
    facts = res._parse_facts("")
    assert facts == []


# ── Verifier tests ────────────────────────────────────────────────────────

def test_verifier_check_conflicts_needs_two_facts():
    """_check_conflicts() should return early with fewer than 2 facts."""
    ver = Verifier()
    mem = ResearchMemory(topic="AI")
    mem.verified_facts = [{"fact": "Only one fact", "subtopic": "test"}]
    # Should not crash with 1 fact
    ver._check_conflicts(mem)
    assert mem.conflicts == []


# ── Pipeline integration test (no LLM) ───────────────────────────────────

def test_memory_flows_through_pipeline_structure():
    """Memory object should carry data between agents correctly."""
    mem = ResearchMemory(topic="Test topic")
    mem.subtopics = ["Subtopic 1", "Subtopic 2"]
    mem.add_fact("Subtopic 1", "Fact about subtopic 1")
    mem.add_fact("Subtopic 2", "Fact about subtopic 2")
    mem.verified_facts = mem.facts.copy()

    assert len(mem.subtopics) == 2
    assert len(mem.facts) == 2
    assert len(mem.verified_facts) == 2
    assert mem.topic == "Test topic"
