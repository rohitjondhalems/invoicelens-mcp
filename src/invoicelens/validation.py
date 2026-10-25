"""
Deterministic checks on an extracted invoice — the code-level safety net.

An LLM can return perfectly-formed JSON that is arithmetically wrong: a misread
digit, a skipped line item, the tax amount copied into the total. Schema
validation can't see that; arithmetic can. Errors here are phrased so an agent
can act on them ("line items sum to X but subtotal is Y"), which is what lets
the LangGraph loop self-correct instead of passing bad numbers on.

Some invoices really are wrong (see samples/initech_*), so a failing report is a
finding to surface, not always an extraction bug.
"""
from __future__ import annotations

from invoicelens.models import Invoice, ValidationReport

TOLERANCE = 0.02  # currency units; absorbs per-line rounding


def _close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) <= max(tol, abs(b) * 1e-6)


def _fmt(v: float) -> str:
    return f"{v:,.2f}"


def validate_invoice(inv: Invoice) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    for field in ("vendor_name", "invoice_number", "invoice_date", "total", "currency"):
        if getattr(inv, field) in (None, ""):
            errors.append(f"Missing required field '{field}'.")

    if inv.currency and (len(inv.currency) != 3 or not inv.currency.isalpha() or not inv.currency.isupper()):
        errors.append(f"Currency '{inv.currency}' is not a 3-letter ISO 4217 code (e.g. EUR, USD).")

    for i, li in enumerate(inv.line_items, 1):
        if li.quantity is not None and li.unit_price is not None:
            expected = li.quantity * li.unit_price
            if not _close(expected, li.amount, TOLERANCE * max(1, li.quantity)):
                errors.append(f"Line {i} ('{li.description[:40]}'): quantity {li.quantity:g} x unit price "
                              f"{_fmt(li.unit_price)} = {_fmt(expected)}, but amount is {_fmt(li.amount)}.")

    if inv.line_items and inv.subtotal is not None:
        lines_sum = sum(li.amount for li in inv.line_items)
        if not _close(lines_sum, inv.subtotal, TOLERANCE * len(inv.line_items)):
            errors.append(f"Line items sum to {_fmt(lines_sum)} but subtotal is {_fmt(inv.subtotal)} "
                          f"(difference {_fmt(inv.subtotal - lines_sum)}).")
    elif not inv.line_items:
        warnings.append("No line items extracted.")

    if inv.subtotal is not None and inv.tax_amount is not None and inv.total is not None:
        if not _close(inv.subtotal + inv.tax_amount, inv.total):
            errors.append(f"Subtotal {_fmt(inv.subtotal)} + tax {_fmt(inv.tax_amount)} = "
                          f"{_fmt(inv.subtotal + inv.tax_amount)}, but total is {_fmt(inv.total)}.")

    if inv.subtotal is not None and inv.tax_rate is not None and inv.tax_amount is not None:
        expected_tax = inv.subtotal * inv.tax_rate / 100
        if not _close(expected_tax, inv.tax_amount, 0.05):
            errors.append(f"Tax at {inv.tax_rate:g}% of {_fmt(inv.subtotal)} is {_fmt(expected_tax)}, "
                          f"but tax amount is {_fmt(inv.tax_amount)}.")

    if inv.invoice_date and inv.due_date and inv.due_date < inv.invoice_date:
        errors.append(f"Due date {inv.due_date} is before invoice date {inv.invoice_date}.")

    if inv.total is not None and inv.total <= 0:
        warnings.append(f"Total is {_fmt(inv.total)} — a credit note, or a misread?")

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)
