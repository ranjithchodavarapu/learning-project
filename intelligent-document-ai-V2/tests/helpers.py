"""Shared test helpers: fake LLMs, reply builders, and a minimal PDF generator."""

import json
from pathlib import Path

INBOX = Path(__file__).resolve().parent.parent / "inbox"
INBOX_V2 = Path(__file__).resolve().parent.parent / "inbox_v2"
REGISTRY = Path(__file__).resolve().parent.parent / "data" / "registry.json"


# ── fake LLMs ─────────────────────────────────────────────────────────────
class ScriptedLLM:
    """Returns json_replies in order for JSON calls (the last one repeats),
    and `summary` for prose calls. Records every call."""

    def __init__(self, json_replies=None, summary="A short summary."):
        self.json_replies = list(json_replies or [])
        self.summary = summary
        self.calls = []

    def __call__(self, system, user, json_mode=False):
        self.calls.append((system, user, json_mode))
        if not json_mode:
            return self.summary
        if not self.json_replies:
            return ""
        return self.json_replies.pop(0) if len(self.json_replies) > 1 else self.json_replies[0]

    @property
    def json_calls(self):
        return [c for c in self.calls if c[2]]

    @property
    def prose_calls(self):
        return [c for c in self.calls if not c[2]]


def invoice_json(**overrides) -> str:
    data = {
        "invoice_number": "INV-2024-0042", "vendor": "Northwind Supplies Ltd.",
        "customer": "Contoso Labs Inc.", "invoice_date": "2024-03-15",
        "due_date": "2024-04-14", "currency": "USD",
        "line_items": [
            {"description": "Office chairs", "quantity": 4, "unit_price": 120.00, "amount": 480.00},
            {"description": "Standing desks", "quantity": 2, "unit_price": 350.00, "amount": 700.00},
            {"description": "Delivery", "quantity": 1, "unit_price": 45.00, "amount": 45.00},
        ],
        "subtotal": 1225.00, "tax": 98.00, "total": 1323.00,
    }
    data.update(overrides)
    return json.dumps(data)


def contract_json(**overrides) -> str:
    data = {
        "title": "Services Agreement",
        "parties": ["Fabrikam Ltd", "Northwind Supplies Ltd"],
        "effective_date": "2025-01-01", "end_date": None,
        "term": "12 months, renews automatically for successive 12 month terms unless 60 days notice",
        "payment_terms": "Provider invoices monthly; Client pays within 30 days of receipt",
        "termination": "Either party may terminate with 30 days written notice",
        "governing_law": "State of Delaware", "auto_renewal": True,
        "key_obligations": ["Provider delivers warehouse management services"],
    }
    data.update(overrides)
    return json.dumps(data)


# What a FAITHFUL model would return for each sample document.
ROUTES = {
    "INV-2024-0042": invoice_json(),
    "INVOICE No. 7781": json.dumps({
        "invoice_number": "7781", "vendor": "Globex Consulting GmbH", "customer": "Initech Ltd",
        "invoice_date": "2024-04-02", "due_date": "2024-05-02", "currency": "EUR",
        "line_items": [
            {"description": "Consulting hours", "quantity": 10, "unit_price": 150.00, "amount": 1500.00},
            {"description": "Travel expenses", "quantity": 1, "unit_price": 240.00, "amount": 240.00}],
        "subtotal": 1740.00, "tax": 330.60, "total": 2170.60}),
    "R-2024-118": json.dumps({
        "invoice_number": "R-2024-118", "vendor": "Müller Software GmbH", "customer": "Fabrikam Ltd",
        "invoice_date": "2024-03-15", "due_date": "2024-04-14", "currency": "EUR",
        "line_items": [
            {"description": "Consulting", "quantity": 2, "unit_price": 500.00, "amount": 1000.00},
            {"description": "License", "quantity": 1, "unit_price": 234.56, "amount": 234.56}],
        "subtotal": 1234.56, "tax": 0.00, "total": 1234.56}),
    "INV-9001": json.dumps({
        "invoice_number": "INV-9001", "vendor": "Shady Parts LLC", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-05-01", "due_date": "2024-05-31", "currency": "USD",
        "line_items": [{"description": "Brake pads", "quantity": 10, "unit_price": 55.00, "amount": 550.00}],
        "subtotal": 550.00, "tax": 44.00, "total": 594.00}),
    "A-1001": json.dumps({
        "invoice_number": "A-1001", "vendor": "Acme Industrial Supply", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-03", "due_date": "2024-07-03", "currency": "USD",
        "line_items": [
            {"description": "Safety gloves", "quantity": 50, "unit_price": 12.00, "amount": 600.00},
            {"description": "Hard hats", "quantity": 20, "unit_price": 45.00, "amount": 900.00},
            {"description": "Delivery", "quantity": 1, "unit_price": 300.00, "amount": 300.00}],
        "subtotal": 1800.00, "tax": 0.00, "total": 1800.00,
        "po_number": "PO-1001", "bank_account": "FR14 2004 1010 0505 0001 3M02 606"}),
    "A-1002": json.dumps({
        "invoice_number": "A-1002", "vendor": "Acme Industrial Supply", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-05", "due_date": "2024-07-05", "currency": "USD",
        "line_items": [
            {"description": "Safety gloves", "quantity": 50, "unit_price": 12.00, "amount": 600.00},
            {"description": "Delivery", "quantity": 1, "unit_price": 300.00, "amount": 300.00}],
        "subtotal": 900.00, "tax": 0.00, "total": 900.00,
        "po_number": "PO-1001", "bank_account": "GB82 WEST 1234 5698 7654 32"}),
    "A-1003": json.dumps({
        "invoice_number": "A-1003", "vendor": "Acme Industrial Supply", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-07", "due_date": "2024-07-07", "currency": "USD",
        "line_items": [
            {"description": "Pallet jack", "quantity": 2, "unit_price": 1000.00, "amount": 2000.00},
            {"description": "Delivery", "quantity": 1, "unit_price": 500.00, "amount": 500.00}],
        "subtotal": 2500.00, "tax": 0.00, "total": 2500.00,
        "po_number": "PO-2002", "bank_account": "FR14 2004 1010 0505 0001 3M02 606"}),
    "A-1004": json.dumps({
        "invoice_number": "A-1004", "vendor": "Acme Industrial Supply", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-09", "due_date": "2024-07-09", "currency": "USD",
        "line_items": [{"description": "Safety goggles", "quantity": 10, "unit_price": 45.00, "amount": 450.00}],
        "subtotal": 450.00, "tax": 0.00, "total": 450.00,
        "po_number": None, "bank_account": "FR14 2004 1010 0505 0001 3M02 606"}),
    "Totally Legit Trading": json.dumps({
        "invoice_number": "INV-77", "vendor": "Totally Legit Trading Co", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-11", "due_date": "2024-07-11", "currency": "USD",
        "line_items": [{"description": "Consulting services", "quantity": 4, "unit_price": 160.00, "amount": 640.00}],
        "subtotal": 640.00, "tax": 0.00, "total": 640.00}),
    "INV-2024-0900": json.dumps({
        "invoice_number": "INV-2024-0900", "vendor": "Northwind Supplies Ltd.", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-12", "due_date": "2024-07-12", "currency": "USD",
        "line_items": [
            {"description": "Server hardware", "quantity": 3, "unit_price": 4500.00, "amount": 13500.00},
            {"description": "Installation", "quantity": 1, "unit_price": 1500.00, "amount": 1500.00}],
        "subtotal": 15000.00, "tax": 0.00, "total": 15000.00,
        "bank_account": "GB29 NWBK 6016 1331 9268 19"}),
    "B-2001": json.dumps({
        "invoice_number": "B-2001", "vendor": "Northwind Supplies Ltd.", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-14", "due_date": "2024-07-14", "currency": "USD",
        "line_items": [{"description": "Label printer", "quantity": 2, "unit_price": 150.00, "amount": 300.00}],
        "subtotal": 300.00, "tax": 0.00, "total": 300.00}),
    "B-2002": json.dumps({
        "invoice_number": "B-2002", "vendor": "Globex Consulting GmbH", "customer": "Initech Ltd",
        "invoice_date": "2024-06-14", "due_date": "2024-07-14", "currency": "EUR",
        "line_items": [{"description": "Consulting hours", "quantity": 5, "unit_price": 150.00, "amount": 750.00}],
        "subtotal": 750.00, "tax": 142.50, "total": 892.50}),
    "B-2003": json.dumps({
        "invoice_number": "B-2003", "vendor": "Northwind Supplies Ltd.", "customer": "Contoso Labs Inc.",
        "invoice_date": "2024-06-15", "due_date": "2024-07-15", "currency": "USD",
        "line_items": [{"description": "Toner cartridges", "quantity": 10, "unit_price": 40.00, "amount": 400.00}],
        "subtotal": 400.00, "tax": 0.00, "total": 400.00}),
    "INV-2024-0555": json.dumps({
        "invoice_number": "INV-2024-0555", "vendor": "Northwind Supplies Ltd.", "customer": "Fabrikam Ltd",
        "invoice_date": "2024-06-10", "due_date": "2024-07-10", "currency": "USD",
        "line_items": [
            {"description": "Pallet racking", "quantity": 2, "unit_price": 650.00, "amount": 1300.00},
            {"description": "Installation", "quantity": 1, "unit_price": 200.00, "amount": 200.00}],
        "subtotal": 1500.00, "tax": 120.00, "total": 1620.00,
        "bank_account": "GB29 NWBK 6016 1331 9268 19"}),
    "5503": json.dumps({
        "invoice_number": "5503", "vendor": "Globex Consulting GmbH", "customer": "Initech Ltd",
        "invoice_date": "2024-06-12", "due_date": "2024-07-12", "currency": "EUR",
        "line_items": [
            {"description": "Consulting hours", "quantity": 8, "unit_price": 150.00, "amount": 1200.00},
            {"description": "Report writing", "quantity": 1, "unit_price": 300.00, "amount": 300.00}],
        "subtotal": 1500.00, "tax": 285.00, "total": 1785.00}),
    "SERVICES AGREEMENT": contract_json(),
    "NON-DISCLOSURE": contract_json(
        title="Mutual Non-Disclosure Agreement",
        parties=["Contoso Labs Inc.", "Initech Ltd"], effective_date="2024-03-01",
        term="2 years from the Effective Date", payment_terms=None,
        termination="Either party may terminate at any time by written notice",
        governing_law="England and Wales", auto_renewal=None,
        key_obligations=["Keep confidential information secret"]),
}


class RoutedLLM:
    """Looks at the document in the prompt and answers like a faithful model."""

    def __init__(self, summary="Contract summary."):
        self.summary = summary
        self.calls = []

    def __call__(self, system, user, json_mode=False):
        self.calls.append((system, user, json_mode))
        if not json_mode:
            return self.summary
        if "classify business documents" in system.lower():
            return '{"doc_type": "invoice"}' if "5503" in user else '{"doc_type": "other"}'
        for marker, reply in ROUTES.items():
            if marker in user:
                return reply
        return ""


# ── minimal single-page PDF ───────────────────────────────────────────────
def make_pdf(path: Path, lines) -> None:
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({esc(l)}) Tj T*" for l in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(out)


# A model that makes realistic mistakes. Used to prove the evaluation harness notices them.
NOISE = {
    "INV-2024-0042":    {"invoice_number": "INV-2024-0024"},               # swaps two digits
    "R-2024-118":       {"subtotal": 1234.65, "total": 1234.65},           # misreads a digit
    "INVOICE No. 7781": {"total": 2070.60},                                # "fixes" the printed total
    "A-1002":           {"bank_account": None},                            # forgets the bank account
}


class NoisyLLM(RoutedLLM):
    def __call__(self, system, user, json_mode=False):
        reply = super().__call__(system, user, json_mode)
        if json_mode and "classify business documents" not in system.lower():
            for marker, changes in NOISE.items():
                if marker in user:
                    data = json.loads(reply)
                    data.update(changes)
                    return json.dumps(data)
        return reply
