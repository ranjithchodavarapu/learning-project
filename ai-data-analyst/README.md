# AI Data Analyst

Upload any CSV → AI writes Python analysis code → executes it → saves charts → writes insights.
Runs fully locally on your H100 using Ollama — no API cost.

---

## How it works

```
CSV file
    │
    ▼
data_profiler.py    reads schema, column types, sample rows, statistics
    │
    ▼
code_generator.py   LLM writes custom pandas + matplotlib code for THIS dataset
    │
    ▼
executor.py         runs the code safely, auto-fixes errors (up to 3 retries)
    │
    ▼
insight_generator.py  LLM reads output → writes plain-English findings
    │
    ├── outputs/chart_1.png  ← saved charts
    ├── outputs/chart_2.png
    └── outputs/report_*.md  ← markdown report with insights
```

---

## Folder structure

```
ai-data-analyst/
├── data_profiler.py       ← reads CSV, extracts schema + stats
├── code_generator.py      ← LLM writes Python (pandas + matplotlib)
├── executor.py            ← runs code safely, self-corrects on error
├── insight_generator.py   ← LLM writes plain-English insights
├── pipeline.py            ← chains all 4 steps together
├── main.py                ← entry point
├── config.py              ← settings from .env
├── sample_data/           ← demo CSV created by --demo flag
├── outputs/               ← charts (PNG) and reports (MD) saved here
├── tests/
│   └── test_analyst.py
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd ai-data-analyst
pip install -r requirements.txt
cp .env.example .env
# .env already configured for Ollama — no changes needed
```

Ollama must be running:
```bash
ollama serve &
ollama list   # verify llama3.1:70b is available
```

---

## Usage

```bash
# Demo mode — generates sample sales data and analyses it
python main.py --demo

# Analyse your own CSV
python main.py --file your_data.csv

# Interactive mode — prompts for file path
python main.py

# Run tests (no Ollama needed)
pytest tests/ -v
```

### Example output

```
Step 1 — Data Profiling
✓ Loaded: 500 rows × 8 columns
┌─────────────────┬──────────┬───────┐
│ Column          │ Type     │ Nulls │
├─────────────────┼──────────┼───────┤
│ date            │ object   │ -     │
│ product         │ object   │ -     │
│ revenue         │ float64  │ -     │
│ units_sold      │ int64    │ -     │
│ customer_rating │ float64  │ -     │
│ discount_pct    │ object   │ -     │
└─────────────────┴──────────┴───────┘

Step 2 — Code Generation
✓ Generated 45 lines of code

Step 3 — Code Execution
✓ Code executed successfully on attempt 1
✓ 3 chart(s) saved

Step 4 — Insight Generation
╭──────────────── Data Insights ─────────────────╮
│ Key Findings:                                   │
│ • Laptop generates highest average revenue      │
│ • Q3 shows 23% revenue spike                   │
│ • West region consistently underperforms       │
│ ...                                             │
╰─────────────────────────────────────────────────╯

✓ Analysis complete in 87s
```

---

## Key concepts

**Why generate code instead of answering directly?**
Every CSV is different. Fixed code can't handle all datasets. The LLM reads the schema and writes custom code for YOUR data — right column names, right types, right analysis.

**What is the self-correcting loop?**
LLM-generated code sometimes has bugs. `executor.py` catches errors and passes them back to the LLM: "you wrote X, it failed with Y, fix it." This loop runs up to MAX_RETRIES times, making the system resilient.

**Why `matplotlib.use("Agg")`?**
Matplotlib normally opens a GUI window to display charts. On a server with no display (like your H100), this crashes. `Agg` = Anti-Grain Geometry — a non-interactive backend that renders to files instead of screens.

**What is `exec(code, namespace)`?**
`exec()` runs a string as Python code. The `namespace` dict controls what the code can access — a safety layer restricting what the LLM-generated code can import or use.

**Why `io.StringIO()` + `redirect_stdout`?**
The generated code uses `print()` to output findings. We redirect those prints to an in-memory buffer so we can capture them and pass them to the insight generator.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:70b` | LLM for code generation and insights |
| `MAX_RETRIES` | `3` | Retry attempts when generated code fails |
| `SAMPLE_ROWS` | `5` | Rows shown to LLM as context |
| `OUTPUT_DIR` | `./outputs` | Where charts and reports are saved |
| `TIMEOUT` | `120` | Seconds before LLM call times out |
