"""
controls.py — Checks an invoice against facts OUTSIDE the document.

The validator can only ask "is this document consistent and did the model read
it correctly?". These controls ask "should we PAY it?". They are what turns
"a forgery whose numbers all agree can't be caught" into "most practical
forgeries are caught", because a forger can fake a document but not our records.

THE CHECKS (each produces an error, so the invoice goes to human review):
  duplicate        same vendor + invoice number already submitted (any earlier run)
  approval limit   total at or above APPROVAL_THRESHOLD always needs a person
  vendor           not on the approved vendor list
  bank account     differs from the account we have on file (the classic
                   "we've changed banks" payment-redirection fraud)
  purchase order   PO missing (when the vendor requires one), unknown, belongs to
                   another vendor, wrong currency, or already used up
  history          far above anything this vendor has billed before

These run AFTER the repair loop, not inside it. They depend on external state, not
on how well the model read the page, so asking the LLM to "repair" them would be
meaningless (and tempting it to change the data to dodge the check).

LINE BY LINE (non-obvious lines):
  rule = registry.vendors.get(vkey)
      None for an unknown vendor. Every later check that needs a rule is guarded
      by `if rule`, so one unknown vendor never causes a crash.

  `if registry.vendors and ...`
      The vendor list check only applies once a list exists. An empty registry
      means "controls not configured", not "every vendor is unapproved".

  Bank check uses the model's extraction AND find_ibans(text)
      find_ibans() is a plain regex plus the IBAN checksum, independent of the
      LLM. If the model forgets to extract the account, the controls still see
      it. Many honest invoices omit bank details altogether, so an ABSENT account
      is not flagged: only an account that differs from the one on file.

  already + total > po.amount + MONEY_TOLERANCE
      Invoices against the same PO accumulate. Two invoices of 3,000 against a
      5,000 PO: the second is the one that breaks it.

  total > ANOMALY_FACTOR * max(history)
      Needs at least ANOMALY_MIN_HISTORY accepted invoices in the same currency,
      so a vendor's first invoices are not judged against nothing.
"""

from decimal import Decimal
from typing import List, Optional

from config import cfg
from models import InvoiceData, Issue
from registry import Registry
from store import InvoiceStore, PriorInvoice
from textutil import find_ibans, normalize_account, po_key, vendor_key


def _m(value: Decimal) -> str:
    return f"{value:.2f}"


def invoice_keys(data: InvoiceData):
    """(vendor_key, number_key) or None when the invoice cannot be identified."""
    if not (data.vendor and data.invoice_number):
        return None
    return vendor_key(data.vendor), data.invoice_number.strip().casefold()


def duplicate_issue(data: InvoiceData, prior: Optional[PriorInvoice],
                    filename: str, chash: str) -> Optional[Issue]:
    if prior is None or (prior.filename == filename and prior.content_hash == chash):
        return None                                # first time, or the same file re-run
    return Issue("error", "invoice_number",
                 f"Duplicate: invoice {data.invoice_number} from {data.vendor} was already "
                 f"submitted as '{prior.filename}' (status: {prior.status})")


def run_controls(data: InvoiceData, registry: Registry, store: InvoiceStore, text: str = "") -> List[Issue]:
    issues: List[Issue] = []
    keys = invoice_keys(data)
    vkey = keys[0] if keys else (vendor_key(data.vendor) if data.vendor else None)
    number_key = keys[1] if keys else ""

    # approval limit
    if data.total is not None and data.total >= cfg.APPROVAL_THRESHOLD:
        issues.append(Issue("error", "total",
                            f"Total {_m(data.total)} is at or above the auto-approval limit "
                            f"of {_m(cfg.APPROVAL_THRESHOLD)}: manual approval required"))

    # approved vendor
    rule = registry.vendors.get(vkey) if vkey else None
    if registry.vendors and vkey and rule is None:
        issues.append(Issue("error", "vendor", f"Vendor '{data.vendor}' is not on the approved vendor list"))

    # bank account: the model's extraction PLUS every valid IBAN found in the raw text,
    # so a model that forgets to extract the account cannot switch this check off
    if rule:
        accounts = set(find_ibans(text))
        if data.bank_account:
            accounts.add(normalize_account(data.bank_account))
        if accounts and rule.bank_account is None:
            issues.append(Issue("warning", "bank_account",
                                f"No bank account on file for {rule.name} to compare with"))
        elif rule.bank_account is not None:
            wrong = sorted(a for a in accounts if a != rule.bank_account)
            if wrong:
                issues.append(Issue("error", "bank_account",
                                    f"Bank account {wrong[0]} on the invoice differs from the account on file "
                                    f"for {rule.name}: possible payment-redirection fraud. Confirm by phone "
                                    "using a number you already hold."))

    # purchase order
    if data.po_number:
        pkey = po_key(data.po_number)
        po = registry.purchase_orders.get(pkey)
        if registry.purchase_orders and po is None:
            issues.append(Issue("error", "po_number", f"Purchase order {data.po_number} is not in the registry"))
        elif po is not None:
            if vkey and po.vendor_key != vkey:
                issues.append(Issue("error", "po_number",
                                    f"Purchase order {po.po_number} belongs to a different vendor"))
            if po.currency and data.currency and po.currency != data.currency:
                issues.append(Issue("error", "currency",
                                    f"Invoice currency {data.currency} differs from PO currency {po.currency}"))
            if data.total is not None and vkey:
                already = store.po_invoiced(pkey, vkey, number_key)
                if already + data.total > po.amount + cfg.MONEY_TOLERANCE:
                    issues.append(Issue("error", "total",
                                        f"Total {_m(data.total)} plus {_m(already)} already invoiced "
                                        f"exceeds purchase order {po.po_number} ({_m(po.amount)})"))
    elif rule and rule.po_required:
        issues.append(Issue("error", "po_number",
                            f"{rule.name} requires a purchase order, but none is referenced"))

    # history
    if vkey and data.total is not None:
        history = store.accepted_totals(vkey, data.currency, number_key)
        if len(history) >= cfg.ANOMALY_MIN_HISTORY and data.total > cfg.ANOMALY_FACTOR * max(history):
            issues.append(Issue("error", "total",
                                f"Total {_m(data.total)} is more than {cfg.ANOMALY_FACTOR}x the largest "
                                f"previous invoice from this vendor ({_m(max(history))})"))
    return issues
