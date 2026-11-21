"""
registry.py — The company's own facts: approved vendors and purchase orders.

WHY THIS EXISTS:
  Reading a document can never reveal a forgery that is internally consistent;
  the document is the only evidence. Checking it against facts the COMPANY
  already holds can: is this vendor approved? Is there a purchase order for
  this amount? Is this the bank account we normally pay?

FILE FORMAT (data/registry.json):
  {
    "vendors": [
      {"name": "Acme Industrial Supply", "bank_account": "FR14 2004 ...", "po_required": true}
    ],
    "purchase_orders": [
      {"po_number": "PO-1001", "vendor": "Acme Industrial Supply", "amount": "5000.00", "currency": "USD"}
    ]
  }

BEHAVIOUR WHEN EMPTY OR MISSING:
  No registry file means no vendor/PO checks, so the project still works as a
  plain extractor. Checks switch on as you add data.

LINE BY LINE (non-obvious lines):
  Decimal(str(entry["amount"]))
      Money is read through str() so a JSON number 5000.1 becomes exactly
      5000.1, and "5000.00" stays exact. Floats would introduce binary noise.

  bank_account stored via normalize_account()
      "FR14 2004 1010 ..." and "FR1420041010..." compare equal.

  RegistryError
      A malformed registry stops the run with a clear message. Silently
      ignoring a broken control file would mean running with NO controls
      while believing they were on.
"""

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Optional

from textutil import normalize_account, po_key, vendor_key


class RegistryError(Exception):
    """The registry file exists but cannot be used."""


@dataclass
class VendorRule:
    name: str
    bank_account: Optional[str] = None      # normalised
    po_required: bool = False


@dataclass
class PurchaseOrder:
    po_number: str
    vendor_key: str
    amount: Decimal
    currency: Optional[str] = None


@dataclass
class Registry:
    vendors: Dict[str, VendorRule] = field(default_factory=dict)
    purchase_orders: Dict[str, PurchaseOrder] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.vendors or self.purchase_orders)


def load_registry(path) -> Registry:
    path = Path(path)
    if not path.is_file():
        return Registry()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RegistryError(f"Registry {path.name} is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise RegistryError(f"Registry {path.name} must be a JSON object")

    registry = Registry()

    for i, entry in enumerate(raw.get("vendors", []), 1):
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise RegistryError(f"vendors[{i}] needs a non-empty 'name'")
        account = entry.get("bank_account")
        registry.vendors[vendor_key(entry["name"])] = VendorRule(
            name=entry["name"].strip(),
            bank_account=normalize_account(account) if account else None,
            po_required=bool(entry.get("po_required", False)),
        )

    for i, entry in enumerate(raw.get("purchase_orders", []), 1):
        try:
            amount = Decimal(str(entry["amount"]))
            number, vendor = str(entry["po_number"]).strip(), str(entry["vendor"]).strip()
        except (KeyError, InvalidOperation, TypeError):
            raise RegistryError(f"purchase_orders[{i}] needs po_number, vendor and a numeric amount")
        if not number or not vendor or not amount.is_finite():
            raise RegistryError(f"purchase_orders[{i}] needs po_number, vendor and a numeric amount")
        registry.purchase_orders[po_key(number)] = PurchaseOrder(
            po_number=number, vendor_key=vendor_key(vendor), amount=amount,
            currency=(str(entry["currency"]).upper() if entry.get("currency") else None),
        )
    return registry
