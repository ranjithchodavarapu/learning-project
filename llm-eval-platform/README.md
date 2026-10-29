# LLM Evaluation Platform

Test any LLM on accuracy, latency, hallucination detection, and cost. Compare multiple models side-by-side. Generates markdown and JSON reports. Runs fully locally on your H100 using Ollama.

---

## How it works

```
test_suites/general.json   (your test cases: prompt + expected output)
    │
    ▼
runner.py          sends each prompt to Ollama, records latency + tokens
    │
    ├──► scorers/accuracy.py       exact + fuzzy match vs expected
    ├──► scorers/hallucination.py  LLM-as-judge: does it make up facts?
    ├──► scorers/latency.py        p50/p95/p99 across all responses
    └──► scorers/cost.py           token count + cost estimate
    │
    ▼
reporter.py        writes reports/report_MODEL_TIMESTAMP.md + .json
```

---

## Folder structure

```
llm-eval-platform/
├── evaluator.py           ← orchestrates runner + all scorers
├── runner.py              ← sends prompts to Ollama, measures latency
├── reporter.py            ← generates markdown + JSON reports
├── main.py                ← entry point
├── models.py              ← shared data structures
├── config.py              ← settings from .env
├── scorers/
│   ├── accuracy.py        ← exact + fuzzy match scoring
│   ├── latency.py         ← p50/p95/p99 percentiles
│   ├── hallucination.py   ← LLM-as-judge detection
│   └── cost.py            ← token usage + cost estimate
├── test_suites/
│   └── general.json       ← 10 built-in test cases
├── reports/               ← output reports saved here
├── tests/
│   └── test_platform.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd llm-eval-platform
pip install -r requirements.txt
cp .env.example .env
```

Ollama must be running:
```bash
ollama serve &
ollama list   # verify llama3.1:70b is available
```

---

## Usage

```bash
# Run with defaults (llama3.1:70b on general suite)
python main.py

# Specify model and suite
python main.py --model llama3.1:8b --suite test_suites/general.json

# Compare two models side-by-side
python main.py --model llama3.1:70b --model llama3.1:8b

# List available test suites
python main.py --list-suites

# Run tests (no Ollama needed)
pytest tests/ -v
```

### Creating your own test suite

Create a JSON file in `test_suites/`:
```json
[
  {
    "prompt": "What is the capital of France?",
    "expected": "Paris",
    "category": "factual",
    "tags": ["geography", "easy"]
  },
  {
    "prompt": "What does RAG stand for?",
    "expected": "Retrieval-Augmented Generation",
    "category": "AI"
  }
]
```

Then run:
```bash
python main.py --suite test_suites/my_suite.json
```

---

## Metrics explained

**Accuracy** — how well does the response match the expected output?
Uses exact match, fuzzy similarity (SequenceMatcher), and substring containment. Threshold: 0.6.

**Latency** — how fast does the model respond?
Measured with `time.perf_counter()` (nanosecond precision). Reports p50/p95/p99 across all test cases. p95 threshold: 5000ms.

**Hallucination** — does the model make up facts?
Uses LLM-as-judge: a second LLM call rates the response 0-10 for groundedness. Score below 0.5 = hallucination flagged.

**Cost** — how many tokens does it use?
Total tokens, average per request, estimated USD cost. Always passes (informational only). Set `TOKEN_PRICE_PER_1K` in `.env` for cost estimates with paid APIs.

---

## Reading reports

```
reports/
├── report_llama3.1-70b_2024-01-15_14-30.md   ← human readable
└── results_llama3.1-70b_2024-01-15_14-30.json ← for CI/CD parsing
```

JSON report structure:
```json
{
  "model": "llama3.1:70b",
  "pass_rate": 90.0,
  "summary": {
    "accuracy": {"avg_score": 0.92, "failures": 1},
    "latency": {"p50": 1200, "p95": 2800},
    "cost": {"total_tokens": 4500, "avg_tokens_per_request": 450}
  },
  "results": [...]
}
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `MODELS_TO_EVAL` | `llama3.1:70b` | Comma-separated models to test |
| `JUDGE_MODEL` | `llama3.1:70b` | Model used for hallucination judging |
| `HALLUCINATION_THRESHOLD` | `0.5` | Score below this = hallucination flagged |
| `TOKEN_PRICE_PER_1K` | `0.0` | Cost per 1000 tokens (0 for local Ollama) |
| `REPORTS_DIR` | `./reports` | Where to save reports |
| `TEST_SUITES_DIR` | `./test_suites` | Where test suite JSON files live |
