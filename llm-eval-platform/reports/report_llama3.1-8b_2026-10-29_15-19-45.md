# LLM Evaluation Report

**Model:** `llama3.1:8b`
**Test suite:** general.json
**Date:** 2026-10-29 15:19:45
**Pass rate:** 90.0% (9/10)

---

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 10 |
| Passed | 9 |
| Failed | 1 |
| Pass rate | 90.0% |

## Scorer Results

| Scorer | Score | Status |
|--------|-------|--------|
| accuracy | 0.79 | ✗ Fail |
| hallucination | 0.79 | ✗ Fail |
| latency | 0.00 | ✗ Fail |
| cost | 0.97 | ✓ Pass |

---

## Test Results

| # | Prompt | Accuracy | Latency | Hallucination | Overall |
|---|--------|----------|---------|---------------|---------|
| 1 | What is the capital of France?... | ✓ 0.85 | ⚠ 6183ms | ✓ 1.00 | ✓ |
| 2 | What is 15 multiplied by 8?... | ✓ 0.85 | 49ms | ✓ 1.00 | ✓ |
| 3 | What does CPU stand for?... | ✓ 0.85 | 140ms | ✓ 0.80 | ✓ |
| 4 | In Python, what does len() do?... | ✓ 0.85 | 1286ms | ✓ 0.80 | ✓ |
| 5 | What is the boiling point of water in Ce... | ✓ 0.85 | 100ms | ✓ 1.00 | ✓ |
| 6 | What year was Python programming languag... | ✓ 0.85 | 1045ms | ✓ 0.70 | ✓ |
| 7 | What does RAG stand for in AI?... | ✗ 0.28 | ⚠ 120101ms | ✗ 0.00 | ✗ |
| 8 | What is the time complexity of binary se... | ✓ 0.85 | 2326ms | ✓ 0.80 | ✓ |
| 9 | Name the three laws of robotics by Isaac... | ✓ 0.85 | 649ms | ✓ 0.80 | ✓ |
| 10 | What is the default port for HTTP?... | ✓ 0.85 | 102ms | ✓ 1.00 | ✓ |

---

## Failed Tests

### What does RAG stand for in AI?

**Expected:** Retrieval-Augmented Generation

**Got:** ERROR: Timed out after 120s

- **accuracy** failed: score 0.28
- **hallucination** failed: score 0.00
