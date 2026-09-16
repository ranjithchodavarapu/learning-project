# Production RAG System

A complete, milestone-by-milestone implementation of a production-grade
Retrieval-Augmented Generation (RAG) system. Every stage is explained,
testable, and independently togglable.

## Milestone progress

| # | Milestone | What it adds | Status |
|---|-----------|-------------|--------|
| M1 | Naive RAG | Load → chunk → embed → retrieve → generate | ✅ |
| M2 | Hybrid search | BM25 + dense + RRF fusion | ✅ |
| M3 | Reranking | Cross-encoder reorder (top-20 → top-5) | ✅ |
| M4 | Query rewriting | HyDE + multi-query expansion | ✅ |
| M5 | Citations | Structured output with source attribution | ✅ |
| M6 | Evaluation | RAGAS metrics end-to-end | ✅ |

---

## Full architecture

```
                        ┌─────────────────────────────────────┐
                        │          INGESTION (run once)        │
                        │                                     │
                        │  PDF/TXT → Loader → Chunker         │
                        │                  ↓                  │
                        │           Embedder (local)          │
                        │          ↙           ↘              │
                        │    ChromaDB        BM25 index       │
                        └─────────────────────────────────────┘

Query
  │
  ▼ [M4] Query rewriter
  │       ├─ HyDE: LLM generates hypothetical answer → embed that
  │       └─ Multi-query: 3 paraphrases of the question
  │
  ▼ [M2] Hybrid retriever (per query variant)
  │       ├─ Dense:  ChromaDB cosine similarity → top-20
  │       ├─ Sparse: BM25 keyword match → top-20
  │       └─ RRF fusion → top-20 combined candidates
  │
  ▼ [M3] Cross-encoder reranker
  │       └─ Scores all (query, chunk) pairs together → top-5
  │
  ▼ [M5] Generator with citations
  │       └─ LLM answer + JSON citation list → CitedAnswer
  │
  ▼ [M6] Evaluator (optional, uses RAGAS)
          └─ faithfulness · answer relevance · context recall · precision
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/production-rag.git
cd production-rag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements_full.txt

cp .env.example .env
# Edit .env — add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## Usage

### 1 — Ingest documents

```bash
python -m src.ingestion.pipeline --file data/my_paper.pdf
# Builds ChromaDB + BM25 index automatically
```

### 2 — Ask questions (full pipeline)

```bash
python main.py
```

### 3 — Ablation testing (compare milestones)

```bash
python main.py --simple         # M1 only
python main.py --no-rewrite     # M1+M2+M3+M5 (skip HyDE)
python main.py --no-rerank      # M1+M2+M4+M5 (skip cross-encoder)
```

### 4 — Run evaluation (M6)

```bash
# Edit data/test_set.json with your QA pairs, then:
python main.py --eval
```

### 5 — Run tests

```bash
pytest tests/ -v
```

---

## What each milestone improves

| Metric | M1 Naive | M2 +Hybrid | M3 +Rerank | M4 +Rewrite |
|--------|----------|-----------|-----------|-------------|
| Context recall | ~0.55 | ~0.70 | ~0.70 | ~0.80 |
| Context precision | ~0.60 | ~0.65 | ~0.85 | ~0.85 |
| Faithfulness | ~0.70 | ~0.72 | ~0.80 | ~0.82 |
| Answer relevance | ~0.75 | ~0.76 | ~0.78 | ~0.88 |

*Approximate values — actual results depend heavily on your domain and documents.*

---

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Final chunks after reranking |

---

## Key concepts

**M2 — Why hybrid over dense-only?**
Dense embeddings miss exact matches (rare names, numbers, codes). BM25 catches them.
RRF fuses both without needing to tune score scales.

**M3 — Why rerank after retrieve?**
Bi-encoders (dense + BM25) embed query and doc *separately* — fast but imprecise.
Cross-encoders see *both together* — slow but much more accurate. Two-stage = best of both.

**M4 — Why HyDE works?**
A hypothetical answer and the real answer live in the same region of embedding space.
The question "What is X?" and the passage about X do not.

**M5 — Why citations matter?**
Without citations you can't verify claims, detect hallucinations, or audit the system.
Every production RAG needs provenance.

**M6 — RAGAS metrics?**
- **Faithfulness**: is every claim in the answer grounded in the retrieved context?
- **Answer relevance**: does the answer actually address the question?
- **Context recall**: does the retrieved context cover everything in the ground truth?
- **Context precision**: is retrieved context focused (low noise)?

---

## Git history

```
M1: Naive RAG baseline — load, chunk, embed, retrieve, generate
M2: Hybrid search — BM25 + dense retrieval + RRF fusion
M3: Reranking — cross-encoder reorder (ms-marco-MiniLM)
M4: Query rewriting — HyDE + multi-query expansion
M5: Citations — structured output with source attribution
M6: Evaluation — RAGAS metrics + simple heuristic eval
```
