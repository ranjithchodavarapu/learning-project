# Multi-Agent Research System

Four AI agents that collaborate to research any topic — orchestrator plans,
researcher finds facts, verifier checks them, summarizer writes the report.
Runs fully locally on your H100 using Ollama — no API cost.

---

## How it works

```
User topic
    │
    ▼
Orchestrator  → breaks topic into 3 subtopics
    │
    ▼
Researcher    → finds 5 facts per subtopic (15 facts total)
    │
    ▼
Verifier      → cross-checks facts, flags contradictions
    │
    ▼
Summarizer    → writes final report, saves to outputs/
```

All agents share one memory object — each agent reads what the previous
one wrote and adds its own findings.

---

## Folder structure

```
multi-agent-research/
├── agents/
│   ├── base_agent.py       ← shared LLM calling logic (all agents inherit)
│   ├── orchestrator.py     ← breaks topic into subtopics
│   ├── researcher.py       ← extracts facts per subtopic
│   ├── verifier.py         ← cross-checks and validates facts
│   └── summarizer.py       ← writes final report
├── memory.py               ← shared state between agents
├── pipeline.py             ← chains agents in sequence
├── config.py               ← settings from .env
├── main.py                 ← entry point
├── outputs/                ← saved research reports
├── tests/
│   └── test_agents.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd multi-agent-research
pip install -r requirements.txt
cp .env.example .env
# .env is already configured for Ollama — no changes needed
```

Ollama must be running with llama3.1:70b:
```bash
ollama serve &
ollama pull llama3.1:70b    # if not already pulled
```

---

## Usage

```bash
# Interactive mode — prompts for topic
python main.py

# Direct topic via CLI
python main.py --topic "History of transformer neural networks"

# Customise depth
python main.py --topic "Climate change" --subtopics 4 --facts 6

# Run tests (no Ollama needed)
pytest tests/ -v
```

### Example output

```
╭─────────────────────────────────────────────────────╮
│ Multi-Agent Research System                         │
│ Topic: History of transformer neural networks       │
╰─────────────────────────────────────────────────────╯

──────────── Orchestrator ────────────
Orchestrator: Planning research on: 'History of transformer neural networks'
Orchestrator: Created 3 subtopics
  1. Origins and invention of the transformer architecture
  2. Key milestones and model evolution
  3. Impact on modern AI applications

──────────── Researcher ────────────
Researcher: [1/3] Researching: Origins and invention...
Researcher: Found 5 facts
...

──────────── Verifier ────────────
Verifier: Verifying 15 facts
Verifier: Verified: 13/15 facts confirmed

──────────── Summarizer ────────────
Summarizer: Writing final research report
Summarizer: Report saved to: outputs/research_History_of_trans_2024-01-15.txt

╭─────────────────────── Research Report ───────────────────────╮
│ # History of Transformer Neural Networks                      │
│                                                               │
│ ## Introduction                                               │
│ The transformer architecture, introduced in 2017...           │
╰───────────────────────────────────────────────────────────────╯
```

---

## Key concepts

**Why multiple agents instead of one prompt?**
One big prompt to do everything = worse results. Focused agents with clear
roles = better quality. The orchestrator doesn't need to know how to write;
the summarizer doesn't need to know how to verify facts.

**Why shared memory?**
Agents run separately — they need a central place to share findings.
`ResearchMemory` is a Python dataclass passed between agents. Each agent
reads what previous agents wrote and adds its own output.

**Why verify facts?**
LLMs hallucinate. Running a second LLM call that independently reviews
each fact catches internal contradictions and obviously wrong claims.

**What is temperature?**
Controls randomness. 0.0 = deterministic (same answer every time).
1.0 = creative (varies). Each agent uses the right temperature for its role:
- Orchestrator: 0.3 (focused planning)
- Researcher: 0.3 (factual)
- Verifier: 0.1 (strict assessment)
- Summarizer: 0.5 (needs some creativity for good prose)

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:70b` | Model to use |
| `MAX_SUBTOPICS` | `3` | Subtopics per research topic |
| `FACTS_PER_SUBTOPIC` | `5` | Facts extracted per subtopic |
| `TIMEOUT` | `120` | Seconds before LLM call times out |
| `OUTPUT_DIR` | `./outputs` | Where reports are saved |
