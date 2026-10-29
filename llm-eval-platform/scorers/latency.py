"""
scorers/latency.py — Measures and scores response latency.

WHAT IT DOES:
  Takes all latency measurements from the run and computes:
  - p50 (median):  half the responses were faster than this
  - p95:           95% of responses were faster than this
  - p99:           99% of responses were faster than this

  Then scores based on whether p95 is within the acceptable threshold.

WHY PERCENTILES MATTER:
  Average latency is misleading. If 95 responses take 200ms but
  5 take 5000ms, the average is ~450ms — looks ok but users
  experience terrible slowness 5% of the time.

  p95 is the standard production SLA metric because it covers
  "what most users experience" while capturing tail latency.

  A p95 of 3000ms (3 seconds) is bad UX.
  A p95 of 1000ms (1 second) is acceptable for most uses.
  A p95 of 500ms is good.

LINE BY LINE:
  sorted(latencies)      → sort ascending for percentile calculation
  _percentile(vals, p)   → numpy-free percentile calculation
    idx = p / 100 * (len-1)
                         → fractional index into sorted list
    lower = int(idx)     → index of lower value
    upper = lower + 1    → index of upper value
    frac = idx - lower   → fractional part
    return vals[lower] + frac * (vals[upper] - vals[lower])
                         → linear interpolation between lower and upper

  LATENCY_THRESHOLDS     → what counts as "good latency"
    p50: 2000ms          → median under 2 seconds = acceptable
    p95: 5000ms          → 95th percentile under 5 seconds = acceptable
    p99: 10000ms         → 99th percentile under 10 seconds = acceptable

  score = 1.0 - (p95 / LATENCY_THRESHOLDS["p95"])
                         → higher score = faster response
                         → p95 = 2500ms, threshold = 5000ms → score = 0.5
                         → p95 = 1000ms, threshold = 5000ms → score = 0.8
                         → clamp to [0, 1] with max(..., 0.0)
"""

from typing import List
from models import LLMResponse, TestCase, ScoreResult
from config import cfg

# Latency thresholds in milliseconds (what we consider "acceptable")
LATENCY_THRESHOLDS = {
    "p50":  2000,    # 2 second median
    "p95":  5000,    # 5 second p95
    "p99": 10000,    # 10 second p99
}


def score(responses: List[LLMResponse]) -> ScoreResult:
    """
    Score latency across all responses in the test run.

    Unlike other scorers, latency scoring looks at ALL responses
    together to compute percentiles, not one at a time.

    Args:
        responses: All LLMResponse objects from the run.

    Returns:
        ScoreResult with percentile stats and pass/fail.
    """
    if not responses:
        return ScoreResult(
            scorer="latency",
            score=0.0,
            passed=False,
            details={"error": "No responses to score"},
        )

    latencies = sorted(r.latency_ms for r in responses)

    # Compute configured percentiles
    percentiles = {}
    for p in cfg.LATENCY_PERCENTILES:
        percentiles[f"p{p}"] = round(_percentile(latencies, p), 1)

    # Score based on p95 vs threshold
    p95 = percentiles.get("p95", percentiles.get("p50", latencies[-1]))
    threshold = LATENCY_THRESHOLDS["p95"]

    # Score: 1.0 = at or below half the threshold (very fast)
    #        0.5 = exactly at threshold
    #        0.0 = twice the threshold or worse
    raw_score = 1.0 - (p95 / threshold)
    final_score = round(max(0.0, min(1.0, raw_score)), 4)

    return ScoreResult(
        scorer="latency",
        score=final_score,
        passed=p95 <= threshold,
        details={
            **percentiles,
            "min_ms": round(latencies[0], 1),
            "max_ms": round(latencies[-1], 1),
            "mean_ms": round(sum(latencies) / len(latencies), 1),
            "threshold_p95_ms": threshold,
            "sample_count": len(latencies),
        },
    )


def _percentile(sorted_vals: List[float], p: int) -> float:
    """
    Compute the p-th percentile of a sorted list without numpy.

    Uses linear interpolation between the two nearest values.
    This matches numpy's default interpolation method.

    Args:
        sorted_vals: Sorted list of floats.
        p:           Percentile (0-100).

    Returns:
        The p-th percentile value.
    """
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    # Fractional index into the sorted list
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lower = int(idx)                    # floor index
    upper = min(lower + 1, len(sorted_vals) - 1)  # ceiling index (clamped)
    frac  = idx - lower                 # fractional part (0.0 to 1.0)

    # Linear interpolation between lower and upper values
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])
