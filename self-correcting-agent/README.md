# Self-Correcting AI Agent

An agent that generates answers, scores them on 5 dimensions, reflects on what's wrong, and retries until quality passes the threshold. Runs fully locally on your H100 using Ollama.

---

## How it works

```
Question
    │
    ▼
generator.py      LLM generates answer (attempt 1: bare question)
    │
    ▼
evaluator.py      LLM scores on: accuracy, completeness, clarity,
    │             conciseness, relevance (0.0 to 1.0 each)
    │
    ├── score >= threshold → DONE (return answer)
    │
    ▼
reflector.py      LLM diagnoses problems + writes improvement plan
    │
    └── retry →  generator.py (attempt 2: question + critique)
                      │
                      ▼ (repeat up to MAX_ATTEMPTS times)
```

The key insight: the generator receives the **previous answer + the critique** on retry. It knows exactly what was wrong and what to fix — not just "try again".

---

## Folder structure

```
self-correcting-agent/
├── agent.py           ← main loop: generate → evaluate → reflect → retry
├── generator.py       ← produces initial and improved answers
├── evaluator.py       ← scores answer on 5 dimensions (LLM-as-judge)
├── reflector.py       ← critiques answer, writes improvement plan
├── models.py          ← shared data structures
├── llm.py             ← centralised Ollama calling
├── config.py          ← settings from .env
├── main.py            ← entry point
├── tests/
│   └── test_agent.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd self-correcting-agent
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
# Interactive
python main.py

# Direct question
python main.py --question "Explain how transformers work"

# Demo mode
python main.py --demo

# Custom threshold and attempts
python main.py --question "What is attention?" --threshold 0.85 --attempts 4

# Run tests (no Ollama needed)
pytest tests/ -v
```

### Example output

```
Attempt 1/3
Generating initial answer...
✓ Answer generated (187 words)
Evaluating answer...
┌─────────────────────────────────────────┐
│ Dimension    Score  Status               │
│ accuracy     0.82   ✓                   │
│ completeness 0.61   ✗                   │
│ clarity      0.79   ✓                   │
│ conciseness  0.88   ✓                   │
│ relevance    0.85   ✓                   │
│ Overall      0.79   FAIL                │
└─────────────────────────────────────────┘
Score 0.79 below threshold 0.75 — reflecting...

Attempt 2/3
Generating improved answer (attempt 2)...
Overall      0.84   PASS
✓ Score 0.84 >= threshold 0.75 — converged!

Final Answer (score: 0.84)
╭──────────────────────────────────────╮
│ Transformers work by...              │
╰──────────────────────────────────────╯

accuracy     ██████████ 0.88
completeness ████████░░ 0.82
clarity      █████████░ 0.86
```

---

## Key concepts

**Why self-correction works:** The generator on retry receives not just the question but its previous answer + the critique. This is richer context than just "try again" — the LLM knows exactly what was wrong.

**LLM-as-judge:** Using a second LLM call to evaluate the first. Alternatives are human evaluation (slow), rule-based (can't judge complex answers), or embedding similarity (misses factual errors). LLM-as-judge scales well and understands nuance.

**Temperature by role:**
- Generator: 0.7 (needs some creativity for good answers)
- Evaluator: 0.1 (strict, consistent scoring — same answer = same score)
- Reflector: 0.3 (structured critique, not creative variation)

**Why pick the best attempt (not the last):** LLMs sometimes over-correct and produce worse answers. `_select_best()` takes the attempt with the highest score across ALL attempts, not just the final one.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:70b` | LLM for all three roles |
| `SCORE_THRESHOLD` | `0.75` | Minimum score to stop retrying |
| `MAX_ATTEMPTS` | `3` | Max generate→evaluate→reflect cycles |
| `TIMEOUT` | `120` | Seconds before LLM call times out |
