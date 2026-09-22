# AI Software Engineer Agent

Give it a task → it plans, writes code, runs tests, fixes bugs automatically.
Runs fully locally on your H100 using Ollama — no API cost.

---

## How it works

```
Task description
    │
    ▼
planner.py      breaks task into files + architecture plan
    │
    ▼
coder.py        LLM writes code for each file
    │
    ▼
tester.py       runs pytest on generated test files
    │
    ▼
debugger.py     reads failures → fixes code → retests (up to 3 times)
    │
    ▼
workspace/{project}/   working code + passing tests + README
```

---

## Folder structure

```
ai-software-engineer/
├── planner.py         ← breaks task into file plan (JSON from LLM)
├── coder.py           ← generates code for each file
├── tester.py          ← runs pytest, captures pass/fail
├── debugger.py        ← reads errors, fixes code, retests
├── pipeline.py        ← chains all 4 steps
├── main.py            ← entry point
├── models.py          ← shared data structures (FileSpec, ProjectPlan etc.)
├── llm.py             ← shared Ollama calling logic
├── config.py          ← settings from .env
├── workspace/         ← generated projects saved here
│   └── project_name/
│       ├── main.py
│       ├── test_main.py
│       └── README.md
├── tests/
│   └── test_engineer.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd ai-software-engineer
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
# Interactive — prompts for task
python main.py

# Specify task directly
python main.py --task "Build a calculator with add, subtract, multiply, divide"

# Demo mode — random example task
python main.py --demo

# Different language
python main.py --task "Build a text counter" --language javascript

# Run tests (no Ollama needed)
pytest tests/ -v
```

### Example session

```
What do you want to build? Build a to-do list manager

Step 1 — Planning
╭──────────────── Project Plan ────────────────╮
│ Project: todo_list_manager                   │
│ Language: python                             │
│ Architecture: Simple CLI todo manager        │
│ Files:                                       │
│   • main.py — CLI entry point                │
│   • todo.py — TodoItem and TodoList classes  │
│   • test_todo.py — pytest tests              │
╰──────────────────────────────────────────────╯

Step 2 — Code Generation
  [1/3] Writing main.py... ✓ (45 lines)
  [2/3] Writing todo.py... ✓ (67 lines)
  [3/3] Writing test_todo.py... ✓ (38 lines)

Step 3 — Testing
  Running: test_todo.py
  ✓ test_todo.py — PASSED
Tests: 1/1 passed

Step 4 skipped — tests already passing

✓ Project built successfully — all tests passing!
Workspace: ./workspace/todo_list_manager
```

---

## Key concepts

**Why plan first?** Without planning, the LLM writes one big monolithic file. Planning forces single-responsibility files that are easier to test and debug.

**Why run real tests?** LLM-generated code often has bugs. Running actual pytest gives ground truth — pass or fail. This is the signal the debugger uses.

**How does the debug loop work?** After each test run, failing tests are sent to the debugger with their error output. The LLM reads the error and rewrites the source file. This repeats up to MAX_DEBUG_ATTEMPTS times.

**What is `subprocess.run()`?** Runs a terminal command from Python. `subprocess.run(["pytest", "test_main.py"])` is exactly like typing `pytest test_main.py` in the terminal. We capture the output and return code.

**Why `cwd=project_dir`?** Running pytest from the project directory ensures relative imports work. Running from elsewhere would break `import main` in test files.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:70b` | LLM for planning, coding, debugging |
| `MAX_DEBUG_ATTEMPTS` | `3` | Max debug cycles before giving up |
| `WORKSPACE_DIR` | `./workspace` | Where generated projects are saved |
| `TIMEOUT` | `180` | Seconds before LLM call times out |
