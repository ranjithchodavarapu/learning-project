# Secure Enterprise AI Assistant

A RAG assistant with **authentication**, **role-based access control**, **prompt injection protection**, and a **tamper-evident audit log**. Runs fully locally with Ollama.

The core idea: do not trust the LLM to keep secrets. Make sure restricted text never reaches it.

---

## How it works

```
Question + login token
   │
   ▼  1. Authenticate     signed token, expiry, brute-force lockout
   ▼  2. Input guard      scan the question for prompt injection
   ▼  3. RBAC filter      keep only documents this role may read   ← BEFORE retrieval
   ▼  4. Retrieve         search the allowed documents only
   ▼  5. Chunk guard      drop retrieved chunks containing injection (indirect attacks)
   ▼  6. LLM              random per-request markers + canary token in the prompt
   ▼  7. Output guard     canary check, leak check against forbidden docs, PII redaction
   ▼
Answer + source titles          (every step writes one hash-chained audit entry)
```

---

## Threat model: what each layer stops

| Attack | Stopped by |
|---|---|
| Guessing passwords | PBKDF2 hashing, lockout after 3 failures |
| Finding valid usernames by timing | Dummy hash for unknown users, identical error message |
| Forging a token (`role: admin`) | HMAC signature over the payload |
| Using a stolen or old token | Expiry, plus the role is re-read from the user store on every request |
| Reading documents above your clearance | RBAC filter before retrieval, so the text never enters the prompt |
| "Ignore all previous instructions..." | Input guard (pattern scoring, Unicode normalisation) |
| Hiding the attack with zero-width or full-width characters | `normalize()` before matching |
| Attack hidden inside a document (indirect injection) | Chunk guard drops flagged chunks; random markers stop a document from closing the data block |
| Tricking the model into revealing the system prompt | Canary token detected by the output guard |
| Model quoting a document you cannot see | Output guard compares the answer to forbidden documents (6-word shingles) |
| Secrets in the answer (SSN, card numbers) | PII redaction (Luhn-checked cards) |
| Learning that a secret document exists | Same "no information" reply for hidden and non-existent topics |
| Rewriting the log to hide an attack | Hash-chained audit log, `--verify-audit` |

---

## Folder structure

```
secure-enterprise-assistant/
├── main.py            ← entry point (chat, --demo-attacks, --verify-audit)
├── assistant.py       ← chains the 7 layers
├── auth.py            ← password hashing, signed tokens, lockout
├── rbac.py            ← clearance + department access rules
├── guardrails.py      ← input guard, chunk scanner, output guard
├── retriever.py       ← TF-IDF over allowed documents only
├── loader.py          ← reads documents and their classification headers
├── audit.py           ← hash-chained audit log
├── llm.py             ← Ollama caller
├── models.py          ← shared data structures
├── config.py          ← settings (no hardcoded secrets)
├── documents/         ← sample docs: public, internal, confidential, restricted, poisoned
├── logs/              ← audit.jsonl is written here
├── tests/test_security.py   ← 76 tests, no Ollama needed
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
cd secure-enterprise-assistant
pip install -r requirements.txt
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # paste into SECRET_KEY in .env

ollama serve &            # if not already running
```

## Usage

```bash
python -m pytest tests/ -v        # 76 security tests, no Ollama needed
python main.py --demo-attacks     # 9 scripted attack and access scenarios (needs Ollama)
python main.py --user bob         # interactive chat
python main.py --verify-audit     # check the audit log for tampering
python main.py --list-users
```

Chat commands: `/docs` (what can I read?), `/whoami`, `/quit`.

### Demo accounts (demo only)

| User | Password | Role | Department | Can read |
|---|---|---|---|---|
| dave | `Guest#2024` | guest | none | public |
| carol | `Employee#2024` | employee | engineering | public, internal (all + engineering) |
| bob | `Manager#2024` | manager | finance | above + confidential finance |
| alice | `Admin#2024` | admin | it | everything |

### Things to try

```
dave:  What are the office hours?                     → answered (public)
dave:  What is the CEO base salary?                   → "no information" (RBAC)
carol: What is the Q3 revenue forecast?               → "no information" (wrong clearance)
bob:   What is the Q3 revenue forecast?               → answered
alice: What is the CEO base salary?                   → answered
carol: Ignore all previous instructions and ...       → blocked (input guard)
carol: What does the vendor integration batch job do? → answered; poisoned paragraph dropped
```

Then run `python main.py --verify-audit`, edit one line of `logs/audit.jsonl`, and run it again.

### Classifying your own documents

Add a `.txt` file to `documents/` with a header:

```
title: Board Minutes
classification: confidential
department: legal
---
...body...
```

A missing or misspelled `classification` becomes **restricted** (fail closed).

---

## Honest limitations

This is a learning project that demonstrates the patterns. For production you would also need:

- **Pattern matching is not complete.** A new paraphrase can slip past the injection rules. That is why the structural layers (RBAC before retrieval, output leak check) matter more than the patterns.
- **Demo users live in source code.** Real systems use a user database or SSO (Okta, Azure AD) and never store passwords in code.
- **Lockout state is in memory.** It resets on restart and is per-process. Use a shared store (Redis) in production.
- **No TLS, rate limiting, or session revocation list.** Put it behind an HTTPS gateway.
- **Audit log is tamper-evident, not tamper-proof.** Someone who can rewrite the whole file can recompute the chain. Ship logs to a separate system.
- **Small chunks (200 chars)** limit the blast radius of a poisoned paragraph but carry less context. A general RAG would use larger chunks with paragraph-level scanning.
- **Guard thresholds need tuning** on your own traffic to balance false positives and false negatives.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | random per start | Signs login tokens. Set a fixed value to keep tokens across restarts |
| `TOKEN_TTL_SECONDS` | `3600` | Token lifetime |
| `MAX_FAILED_LOGINS` | `3` | Failed attempts before lockout |
| `LOCKOUT_SECONDS` | `300` | Lockout duration |
| `INJECTION_BLOCK_THRESHOLD` | `0.6` | Risk score at which a request is blocked |
| `MAX_QUERY_CHARS` | `1000` | Longest allowed question |
| `TOP_K` | `3` | Chunks retrieved per question |
| `OLLAMA_MODEL` | `llama3.1:70b` | Local model |
