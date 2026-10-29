"""
scorers/hallucination.py — Detects hallucinations using LLM-as-judge.

WHAT IT DOES:
  For each response, asks a separate LLM to judge:
  "Does this response contain any information NOT supported by the prompt?"

  If the response makes claims beyond what was asked or what facts
  were provided, that's a hallucination.

WHY LLM-AS-JUDGE FOR HALLUCINATION:
  Hallucination detection is hard to do with rules.
  "The Eiffel Tower is in London" is factually wrong but rule-based
  detectors would need a fact database to catch it.

  An LLM judge can reason about whether a claim is plausible
  and whether it was justified by the prompt context.

  Limitation: the judge LLM can itself hallucinate.
  Mitigation: use a different model as judge than being tested.

LINE BY LINE:
  JUDGE_SYSTEM_PROMPT    → strict persona for the judge
                           "conservative" = flag anything uncertain
                           better false positive than missing real hallucination

  JUDGE_PROMPT           → asks judge to rate 0-10
                           10 = no hallucination at all
                           0 = completely fabricated
                           returns JSON for reliable parsing

  _extract_score()       → parses the judge's JSON response
    json.loads(clean)    → parse JSON string
    data.get("score", 5) → safe access, neutral 5 if missing
    / 10                 → convert 0-10 → 0.0-1.0

  cfg.HALLUCINATION_THRESHOLD
                         → score below this = hallucination flagged
                           default 0.5 = judge thinks >50% chance of hallucination
                           0.5/10 = score of 5 or below triggers flag

  "temperature": 0       → deterministic judging
                           same response should always get same verdict
"""

import json
import requests
from models import LLMResponse, TestCase, ScoreResult
from config import cfg


JUDGE_SYSTEM_PROMPT = """You are a strict hallucination detector for AI responses.
Your job is to identify if an AI response contains claims not supported by the prompt.
Be conservative — flag anything uncertain.
Respond with ONLY valid JSON."""


JUDGE_PROMPT = """Check this AI response for hallucinations.

Original prompt to the AI:
{prompt}

AI's response:
{response}

A hallucination is any claim in the response that:
- Was not stated or implied in the original prompt
- Cannot be reasonably inferred from context
- Adds "facts" beyond what was asked

Rate the response 0-10 where:
  10 = no hallucination, fully grounded in the prompt
  7-9 = minor additions but essentially accurate
  4-6 = some ungrounded claims
  0-3 = significant hallucinations

Return ONLY this JSON:
{{"score": 8, "reasoning": "brief explanation", "hallucinated_claims": ["claim1"]}}"""


def score(response: LLMResponse, test_case: TestCase) -> ScoreResult:
    """
    Use an LLM judge to detect hallucinations in the response.

    Args:
        response:  LLMResponse to check.
        test_case: TestCase with the original prompt context.

    Returns:
        ScoreResult where higher score = less hallucination.
    """
    # Skip error responses
    if response.response.startswith("ERROR:"):
        return ScoreResult(
            scorer="hallucination",
            score=0.0,
            passed=False,
            details={"error": "Response was an error"},
        )

    payload = {
        "model": cfg.JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": JUDGE_PROMPT.format(
                prompt=test_case.prompt,
                response=response.response,
            )},
        ],
        "stream": False,
        "options": {"temperature": 0},   # deterministic judging
    }

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=cfg.TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()

        hallucination_score, details = _extract_score(raw)

        return ScoreResult(
            scorer="hallucination",
            score=hallucination_score,
            passed=hallucination_score >= cfg.HALLUCINATION_THRESHOLD,
            details=details,
        )

    except Exception as e:
        # On failure: neutral score (0.5) — neither pass nor fail
        return ScoreResult(
            scorer="hallucination",
            score=0.5,
            passed=True,    # don't penalise for judge failure
            details={"error": str(e), "note": "Judge call failed — neutral score"},
        )


def _extract_score(raw: str) -> tuple[float, dict]:
    """
    Parse judge's JSON response into a 0.0-1.0 score.

    Returns (score, details_dict).
    Falls back to 0.5 on any parse error.
    """
    # Strip markdown fences
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])

    try:
        data = json.loads(clean)
        raw_score = float(data.get("score", 5))       # 0-10 scale
        normalized = round(raw_score / 10.0, 4)       # convert to 0-1

        return normalized, {
            "raw_score": raw_score,
            "reasoning": data.get("reasoning", ""),
            "hallucinated_claims": data.get("hallucinated_claims", []),
        }

    except (json.JSONDecodeError, ValueError):
        return 0.5, {"error": "Parse failed", "raw": raw[:200]}
