# LLM Evaluation Report

**Model:** `llama3.1:70b`
**Test suite:** general.json
**Date:** 2026-10-29 15:10:23
**Pass rate:** 0.0% (0/10)

---

## Summary

| Metric | Value |
|--------|-------|
| Total tests | 10 |
| Passed | 0 |
| Failed | 10 |
| Pass rate | 0.0% |

## Scorer Results

| Scorer | Score | Status |
|--------|-------|--------|
| accuracy | 0.12 | ✗ Fail |
| hallucination | 0.00 | ✗ Fail |
| latency | 1.00 | ✓ Pass |
| cost | 1.00 | ✓ Pass |

---

## Test Results

| # | Prompt | Accuracy | Latency | Hallucination | Overall |
|---|--------|----------|---------|---------------|---------|
| 1 | What is the capital of France?... | ✗ 0.06 | 1ms | ✗ 0.00 | ✗ |
| 2 | What is 15 multiplied by 8?... | ✗ 0.00 | 0ms | ✗ 0.00 | ✗ |
| 3 | What does CPU stand for?... | ✗ 0.26 | 0ms | ✗ 0.00 | ✗ |
| 4 | In Python, what does len() do?... | ✗ 0.28 | 1ms | ✗ 0.00 | ✗ |
| 5 | What is the boiling point of water in Ce... | ✗ 0.00 | 0ms | ✗ 0.00 | ✗ |
| 6 | What year was Python programming languag... | ✗ 0.00 | 0ms | ✗ 0.00 | ✗ |
| 7 | What does RAG stand for in AI?... | ✗ 0.20 | 0ms | ✗ 0.00 | ✗ |
| 8 | What is the time complexity of binary se... | ✗ 0.15 | 0ms | ✗ 0.00 | ✗ |
| 9 | Name the three laws of robotics by Isaac... | ✗ 0.22 | 0ms | ✗ 0.00 | ✗ |
| 10 | What is the default port for HTTP?... | ✗ 0.00 | 0ms | ✗ 0.00 | ✗ |

---

## Failed Tests

### What is the capital of France?

**Expected:** Paris

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.06
- **hallucination** failed: score 0.00

### What is 15 multiplied by 8?

**Expected:** 120

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.00
- **hallucination** failed: score 0.00

### What does CPU stand for?

**Expected:** Central Processing Unit

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.26
- **hallucination** failed: score 0.00

### In Python, what does len() do?

**Expected:** Returns the number of items in an object

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.28
- **hallucination** failed: score 0.00

### What is the boiling point of water in Celsius?

**Expected:** 100

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.00
- **hallucination** failed: score 0.00

### What year was Python programming language created?

**Expected:** 1991

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.00
- **hallucination** failed: score 0.00

### What does RAG stand for in AI?

**Expected:** Retrieval-Augmented Generation

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.20
- **hallucination** failed: score 0.00

### What is the time complexity of binary search?

**Expected:** O(log n)

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.15
- **hallucination** failed: score 0.00

### Name the three laws of robotics by Isaac Asimov

**Expected:** 1. A robot may not injure a human being. 2. A robot must obey orders. 3. A robot must protect its ow

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.22
- **hallucination** failed: score 0.00

### What is the default port for HTTP?

**Expected:** 80

**Got:** ERROR: Cannot connect to Ollama

- **accuracy** failed: score 0.00
- **hallucination** failed: score 0.00
