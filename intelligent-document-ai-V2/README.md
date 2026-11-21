# Intelligent Document AI (v2)

Drop invoices and contracts into a folder (text, PDF, or **scanned images**). The system reads each one, extracts structured data, **checks it against the document, against arithmetic, and against your company's own records**, summarizes it, and sorts everything into *accepted* or a *review queue* for a human. Runs locally with Ollama.

The core idea: **don't trust the LLM, don't trust the document, and don't trust the scan.** Every value must be found in the source, the numbers must add up, and the invoice must make sense against what the company already knows.

---

## What's new in v2

| v1 limitation | v2 |
|---|---|
| "Not tested on a real model" | `evaluate.py` scores any model against known answers and reports **bad documents accepted** (the number that matters) and **good documents sent to a human** |
| Scanned PDFs and images unsupported | OCR (RapidOCR or Tesseract) with **confidence routing**: low-confidence text goes to review |
| One file = one document | A scanned stack of invoices is split into separate documents |
| "A forgery with consistent numbers can't be caught" | Checked against **approved vendors, bank account on file, purchase orders, vendor history, an approval limit**, and duplicates **across runs** |

> **Not yet run against your real Ollama model.** All automated tests use scripted stand-ins (and the OCR tests use real OCR engines). Run `python evaluate.py` to measure the real model; see below.

---

## How it works

```
inbox/ (txt, md, pdf, png, jpg, tif)
   │
   ▼ 1 load      text layer when there is one; OCR for pages without
   ▼ 2 split     several invoices in one file become separate documents
   │
   ▼ for each document:
   ▼ 3 OCR check   low-confidence OCR (especially on digits) -> review
   ▼ 4 scan        text aimed at the AI ("ignore all previous instructions...")
   ▼ 5 classify    keyword rules first; the LLM only when unsure
   ▼ 6 extract     the LLM fills a JSON schema; one repair attempt if checks fail
   ▼ 7 validate    required fields, arithmetic, dates, "is every value in the source?"
   ▼ 8 controls    duplicate? approved vendor? bank account? purchase order? unusual amount?
   ▼ 9 summarize   template for invoices, LLM for contracts
   ▼ 10 record     remember the invoice so later runs can spot duplicates
   │
   ▼
accepted   |   needs_review   |   failed
   │
   ▼ outputs/results/*.json   outputs/invoices.csv   outputs/report.md
```

### Statuses

| Status | Meaning |
|---|---|
| `accepted` | No errors. Warnings may exist (for example "read by OCR", "contract auto-renews"). |
| `needs_review` | Data was extracted but a check failed. A human decides. |
| `failed` | Nothing usable: unreadable file, or the model never produced valid JSON. |

---

## The checks, in four layers

| Layer | Question | Catches |
|---|---|---|
| **Grounding** | Is every extracted value actually in the document? | the *model* inventing or mis-copying a value |
| **Consistency** | Do the numbers and dates agree? | wrong totals printed *on the document* |
| **Controls** (new) | Does it match what we already know? | forgeries that are internally consistent |
| **OCR quality** (new) | Was the text read reliably? | misread digits in scans |

### Controls (`controls.py`, configured by `data/registry.json`)

| Control | Triggers when | Real-world fraud it targets |
|---|---|---|
| Duplicate | same vendor + invoice number already submitted in **any** earlier run | paying twice |
| Approval limit | total at or above `APPROVAL_THRESHOLD` | large one-off payments |
| Approved vendor | vendor not in the registry | fake suppliers |
| Bank account | an account on the invoice differs from the one on file | **"we've changed banks"**, the classic redirection scam |
| Purchase order | missing (when the vendor requires one), unknown, another vendor's, wrong currency, or already used up | invoices for goods never ordered |
| History | total is more than 3x the largest earlier invoice from that vendor | inflated bills |

**The bank check does not depend on the model.** A plain regex plus the IBAN checksum scans the raw text for account numbers, so a model that forgets to extract the account cannot switch the check off. (The evaluation harness's "noisy model" does exactly this, and the invoice still goes to review.)

**Duplicates vs re-runs.** Running the tool twice on the same folder does not flag everything as duplicates: a record counts as "the same submission" only when file name **and** content match. The same PDF emailed again under a new name, or an edited file reusing an old invoice number, **is** flagged.

### The registry (`data/registry.json`)

```json
{
  "vendors": [
    {"name": "Acme Industrial Supply", "bank_account": "FR14 2004 1010 0505 0001 3M02 606", "po_required": true}
  ],
  "purchase_orders": [
    {"po_number": "PO-1001", "vendor": "Acme Industrial Supply", "amount": "5000.00", "currency": "USD"}
  ]
}
```

No registry file means no vendor/PO checks (the system still works as an extractor). A malformed registry stops the run with a clear message instead of silently running with no controls.

### OCR (`ocr.py`)

- **RapidOCR** (default): `pip install rapidocr-onnxruntime`. Runs on CPU, **needs no admin rights**.
- **Tesseract** (optional): needs the `tesseract` program plus `pip install pytesseract`.
- OCR engines return each table cell as a separate box. `assemble_lines()` rebuilds rows from box positions, otherwise a quantity could be paired with the wrong price.
- **Confidence routing:** a document goes to review if mean OCR confidence is below 85%, or **any line containing digits** is below 90% (the worst digit-bearing part counts, not the average). `OCR_FORCE_REVIEW=true` sends every scan to a person.
- **Whitespace tolerance:** OCR often drops spaces (`Date:12June2024`). For OCR text, a name or date also counts as found with spaces removed (per line, so `INV-42` still does not match inside `INV-420`).

> **OCR can be confidently wrong.** On the bundled `scanned_invoice.pdf`, RapidOCR returned `INV0ICENo.5503` and `T0TAL` with **97% confidence** and no low-confidence digit lines. Confidence is a signal, not a guarantee. That is why amounts are also protected by arithmetic, the IBAN checksum, and grounding. For high-value invoices set `OCR_FORCE_REVIEW=true`.

### Bundles (`splitter.py`)

A line starting with `Invoice Number:` / `Invoice No.` / `Invoice #` followed by a value **that contains a digit** marks an invoice's identity. When the number changes, a new document starts; a repeated number is a continuation page. Each piece goes through the whole pipeline separately and is named `bundle.pdf#1`, `bundle.pdf#2`, ...

---

## Measuring a model: `evaluate.py`

```bash
python evaluate.py                    # your real model, on all sample documents
python evaluate.py --fake faithful    # self-test: a model that reads perfectly
python evaluate.py --fake noisy       # self-test: a model that makes mistakes
```

It scores each document against `eval/ground_truth.json` and reports:

| Metric | Meaning |
|---|---|
| **Missed bad** | bad documents that were **auto-accepted**. Target: 0. The command exits with code 1 if this is above 0, so it can gate a deployment |
| False review | good documents sent to a human (costs minutes, not money) |
| Correct decision | overall accept/review accuracy |
| Field accuracy | per-field extraction accuracy (invoice number, vendor, total, ...) |

Measured on the bundled documents with stand-in models:

| Model | Correct decision | Missed bad | False review | Field accuracy |
|---|---|---|---|---|
| faithful stand-in | 17 / 17 | 0 of 9 | 0 of 8 | 74 / 74 |
| noisy stand-in (4 realistic mistakes) | 15 / 17 | **0 of 9** | 2 of 8 | 91.9% |

The noisy stand-in swaps two digits of an invoice number, misreads a total, "fixes" an invoice's printed total, and forgets a bank account. Every one still ends up in review. **These numbers describe the harness and the checks, not your real model. Run it to get those.**

**First run against `llama3.1:70b`** (the author's server): 17 / 17 correct decisions, 0 of 9 bad documents accepted, 0 of 8 good documents sent to review, field accuracy 95.9% (71 of 74). **All three mismatches were mistakes in the scoring, not in the model:**

| Document | Field | Truth file said | Model returned | Cause |
|---|---|---|---|---|
| `contract_services.txt` | governing_law | `State of Delaware` | `the laws of the State of Delaware` | comparison too strict |
| `contract_nda.txt` | governing_law | `England and Wales` | `the laws of England and Wales` | comparison too strict |
| `contract_nda.txt` | auto_renewal | `false` | `null` | truth file wrong: the NDA never mentions renewal, so `null` ("not stated") was the honest answer |

The fixes: clause fields (`governing_law`, `term`, `payment_terms`, `termination`) now count as correct when one text contains the other and the longer is at most 4x the shorter; names, identifiers, dates, amounts and bank accounts stay exact; the truth file says `null`. The model's output was not changed, so the first-run figures above are the real measurement and ****a re-run on the corrected scoring reported 74 / 74 on the real model** (17 / 17 correct decisions, 0 of 9 bad documents accepted, 0 of 8 good documents sent to review)**. The stand-in model had made the same wrong assumption as the truth file, which is why it scored a perfect 100% and only the real model exposed it.

> **Nine bad documents is a small test, and they are hand-made.** Passing them shows the checks work on these scenarios, not that the system is safe on your invoices. Before trusting it with real payments, add 20 to 30 anonymised real invoices and a truth file written from what each one says (use `null` for anything a document does not state).

Documents marked `"ocr_dependent": true` in the truth file are shown but not scored, because their outcome depends on the OCR engine version.

---

## Testing on your own invoices (the real test)

Everything above uses hand-made, tidy documents. To find out whether the system is safe on **your** invoices, use the kit. Full instructions: [`my_invoices/README.md`](my_invoices/README.md).

```bash
# 1. put 20+ good and 10+ deliberately bad invoices in my_invoices/  (git-ignored)
python truth_kit.py template --inbox my_invoices --out eval/my_truth.json
# 2. fill in every TODO yourself, by reading each document
python truth_kit.py check --truth eval/my_truth.json --inbox my_invoices
# 3. measure
python evaluate.py --truth eval/my_truth.json --inbox my_invoices --no-registry
```

- **You write the answers, never the model.** A truth file drafted by the system under test hides its mistakes (the stand-in model and the first truth file shared a wrong assumption and agreed perfectly).
- `check` catches leftover `TODO`s, money written as `"1,323.00"`, dates not in `YYYY-MM-DD`, wrong field names, files that don't exist, and **values that aren't in the document (usually typos)**. It also tells you if you have too few bad documents to measure "missed bad" (aim for 10+; make them by editing copies of good invoices).
- `evaluate.py` **refuses a truth file that has mistakes** (a leftover `TODO`, `"1,323.00"`, a bad status word) and stops with exit code 2 before running the model: scoring an unfilled template would give real-looking but meaningless numbers. `README` files inside a documents folder are ignored.
- `--no-registry` switches off the vendor / purchase-order / bank checks, since the sample registry doesn't know your vendors. Pass `--registry yours.json` to test those too.
- Real invoices contain personal and financial data. `my_invoices/*` and `eval/my_truth*.json` are git-ignored.

## Sample documents

`inbox/` (from v1) and `inbox_v2/` (new). With a faithful model:

| File | Expected | Why |
|---|---|---|
| `invoice_clean.txt` | accepted | consistent and grounded |
| `invoice_clean_resubmitted.txt` | needs_review | duplicate |
| `invoice_european.txt` | accepted | reads `1.234,56`, `15.03.2024`, `Müller` |
| `invoice_math_error.txt` | needs_review | printed total 2,170.60 but the parts add to 2,070.60 |
| `invoice_injection.txt` | needs_review | text addressed to an AI + unapproved vendor |
| `contract_services.txt` / `contract_nda.txt` | accepted | services one warns it auto-renews |
| `memo_unknown.txt` | needs_review | neither invoice nor contract |
| `acme_ok.txt` | accepted | matches PO, vendor and bank account on file |
| `acme_bank_changed.txt` | needs_review | **different bank account than on file** |
| `acme_over_po.txt` | needs_review | total 2,500 against a 1,200 purchase order |
| `acme_no_po.txt` | needs_review | this vendor requires a PO |
| `unknown_vendor.txt` | needs_review | not on the approved vendor list |
| `northwind_large.txt` | needs_review | 15,000: over the auto-approval limit and 3x the vendor's history |
| `bundle_three_invoices.pdf` | 3 x accepted | one PDF, three invoices, split into `#1 #2 #3` |
| `scanned_invoice.png` | accepted (OCR warning) | read by OCR, confidence ~98% |
| `scanned_invoice.pdf` | accepted (OCR warning) | image-only PDF; OCR garbles it but values still verify |

---

## Folder structure

```
intelligent-document-ai/
├── main.py          ← entry point
├── evaluate.py      ← measure a model against known answers          (new)
├── truth_kit.py     ← build and check a truth file for YOUR invoices  (new)
├── pipeline.py      ← all stages, repair loop, controls, bundles
├── loader.py        ← txt/md/pdf/images, OCR for pages without text
├── ocr.py           ← RapidOCR / Tesseract, row assembly, confidence  (new)
├── splitter.py      ← separate invoices that share a file             (new)
├── controls.py      ← vendor, bank, PO, history, limit, duplicates    (new)
├── registry.py      ← approved vendors + purchase orders from JSON    (new)
├── store.py         ← invoice history in SQLite                       (new)
├── security.py      ← flags text aimed at the AI
├── classifier.py    ← invoice / contract / unknown
├── extractor.py     ← prompts the LLM, builds typed data
├── validator.py     ← completeness, arithmetic, dates, grounding
├── summarizer.py    ← template or LLM summary
├── reporter.py      ← JSON, CSV, markdown report
├── textutil.py      ← money, dates, IBANs, matching
├── models.py  llm.py  config.py
├── make_samples.py  ← regenerates the PNG / scanned PDF / bundle PDF
├── data/registry.json     ← sample vendors and purchase orders
├── eval/ground_truth.json ← expected outcome for every sample
├── my_invoices/           ← drop your own invoices here (git-ignored); see its README
├── inbox/  inbox_v2/      ← sample documents
├── outputs/               ← results are written here
└── tests/                 ← 483 tests, no Ollama needed
```

---

## Setup

```bash
cd intelligent-document-ai
pip install -r requirements.txt
cp .env.example .env
ollama serve &              # if not already running
```

## Usage

```bash
python -m pytest tests/ -q                    # 483 tests, no Ollama needed (about 25 s)
python evaluate.py --fake faithful            # prove the harness works, no Ollama needed
python evaluate.py                            # measure YOUR model on the samples
python truth_kit.py template --inbox my_invoices --out eval/my_truth.json   # then test on YOUR invoices
python main.py                                # process inbox/
python main.py --inbox inbox_v2               # the v2 scenarios
python main.py --no-history                   # forget everything after the run (good for demos)
python main.py --ocr none                     # never run OCR
python main.py --db my.db --registry my.json
python make_samples.py                        # rebuild the PNG / PDF samples
```

Read `outputs/report.md` (review queue first). Import only `accepted` rows of `invoices.csv`.

### Upgrading from v1

- `pip install -r requirements.txt` (new: pypdfium2, pillow, rapidocr-onnxruntime).
- `python main.py` now **remembers invoices** in `data/history.db` and reads `data/registry.json`. Delete or rename the registry to turn the vendor/PO checks off; use `--no-history` for v1-style single-run behaviour.
- Processing the same file twice is no longer a "duplicate" (v1 flagged it). A different file name or different content is.
- `DocumentPipeline.process_file()` returns a list (a file may hold several documents). `process()` still works for single-document files and **raises** if given a bundle rather than silently dropping documents.
- The CSV gained `po_number` and `bank_account` columns.

---

## Honest limitations

**Still true from v1**
- **Not run against a real model during the build.** Use `evaluate.py`.
- **A forgery that matches a real purchase order, from a known vendor, with unchanged bank details, still can't be caught** by any automated check. Controls shrink the gap; they don't close it. High-value payments still need a person.
- **Ambiguous slash dates** (`03/04/2024`) are refused when parsed, but grounding accepts either day/month order, so a model that silently converts one to ISO could be wrong without being caught.
- **European `1.234` (dot as thousands) with no comma** is read as 1.234.
- **English month names only.**
- **Mixed currencies on one invoice** are not checked; there is **no currency conversion**, so the approval limit and PO amounts compare numbers in the invoice's own currency.

**OCR**
- **OCR can be confidently wrong** (see the note above). Confidence thresholds reduce the risk; they cannot remove it.
- Scanned **tables** depend on the engine's box detection; line items from scans are the least reliable part.
- Handwriting, very low resolution, and skewed pages are not handled.
- Page rendering uses `OCR_DPI` (default 200); very small print may need 300.

**Controls and history**
- **You must maintain the registry.** Vendor names are matched ignoring case and punctuation (`Ltd.` = `ltd`) but **not** abbreviations (`Ltd` vs `Limited`).
- The text scan for bank accounts recognises **IBANs from 20 countries** only. Other account formats (US routing + account number) are checked only if the model extracts them.
- An invoice that states **no** bank account is not flagged (many honest invoices omit it).
- A vendor reusing an invoice number in a later year is flagged as a duplicate.
- SQLite is single-process. Run one instance at a time. **The history database contains vendor names and amounts; protect it like the invoices themselves.**
- Vendor-history "unusually large" needs at least 3 accepted invoices in the same currency first.

**Bundles**
- Splits **invoices only** (not contracts or mixed bundles) and needs a recognisable `Invoice Number` label whose value contains a digit. An all-letter number such as `INV-ABC` is not recognised.
- A wrong split is not trusted blindly: every piece goes through full validation, and a piece that isn't a valid invoice lands in review.
- Duplicate documents inside one file are flagged by the normal duplicate check.

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:70b` | local model |
| `MAX_REPAIR_ATTEMPTS` | `1` | self-correction attempts per document |
| `MAX_DOC_CHARS` | `12000` | longer documents are cut and flagged |
| `MONEY_TOLERANCE` | `0.01` | amounts within this count as equal |
| `APPROVAL_THRESHOLD` | `10000` | totals at or above this always need a person |
| `ANOMALY_FACTOR` / `ANOMALY_MIN_HISTORY` | `3` / `3` | "unusually large" rule |
| `DB_PATH` / `REGISTRY_PATH` | `data/history.db`, `data/registry.json` | history and registry |
| `OCR_ENGINE` | `auto` | `auto`, `rapidocr`, `tesseract`, `none` |
| `OCR_MIN_CONFIDENCE` / `OCR_MIN_NUMERIC_CONFIDENCE` | `85` / `90` | confidence routing |
| `OCR_FORCE_REVIEW` | `false` | every OCR'd document goes to a person |
| `MAX_PDF_PAGES` / `MAX_IMAGE_PIXELS` / `MAX_DOCS_PER_FILE` | `100` / `50000000` / `50` | safety limits |
