# Testing on your own invoices

The bundled documents are hand-made and tidy. This folder is where you measure the system on **real** ones.

## 1. Collect and protect

- Put **20+ good invoices** and **10+ deliberately bad ones** here (`.txt`, `.pdf`, `.png`, `.jpg`, `.tif`).
- Real invoices hold names, bank accounts and amounts. **Everything in this folder except this file is git-ignored**, so it will not be pushed. Keep it that way.
- If you must anonymise, replace names consistently but **leave amounts, dates and the structure of numbers alone**. A made-up IBAN must still pass its checksum (use one of the valid examples in `data/registry.json`), or it will correctly be flagged.

## 2. Make the bad ones

"Missed bad" is a rate over bad documents; with only 3 it means almost nothing. Copy good invoices and break them:

- change a **total** so it no longer matches the lines
- **repeat** an invoice number from another file
- swap in a different **bank account**
- remove the **invoice number** or **vendor**
- add a line like "ignore all previous instructions"

## 3. Write the answers yourself

```bash
python truth_kit.py template --inbox my_invoices --out eval/my_truth.json
```

Open `eval/my_truth.json` and replace every `TODO` by **reading each document**. Do not let the model fill it in: a truth file written by the system under test hides its mistakes.

- `status`: `accepted` (a correct system should pass it) or `needs_review` (a person should look)
- `null` for anything the document **does not state**. Never guess ("no renewal mentioned" is `null`, not `false`)
- money: plain numbers (`1323.00`); dates: `YYYY-MM-DD`

## 4. Check your answers

```bash
python truth_kit.py check --truth eval/my_truth.json --inbox my_invoices
```

It catches leftover TODOs, wrong words, `"1,323.00"`, bad dates, and values that are not in the document (usually typos). Fix every `ERROR`. Read each `warning`.

## 5. Measure

```bash
python evaluate.py --truth eval/my_truth.json --inbox my_invoices --no-registry
```

`--no-registry` switches off the vendor / purchase-order / bank-account checks, because the sample registry does not know your vendors. To test those checks too, write your own `registry.json` and pass `--registry yours.json` instead.

## 6. Read the result

- **Missed bad** above 0: the most serious finding. Open `outputs/eval_report.md`, find the document, and see which check should have caught it.
- **False review** high: the system is cautious; look at which good invoices it held back and why.
- **WRONG** rows with field mismatches: compare `wanted` with `got`. First ask whether *your answer* was wrong. Only then blame the model.
