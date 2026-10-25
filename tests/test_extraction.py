from datetime import date

import pytest

from invoicelens.extraction import FallbackExtractor, RuleExtractor, _parse_date
from invoicelens.models import Invoice, LineItem
from invoicelens.parsing import parse_pdf
from invoicelens.validation import validate_invoice

from .conftest import ACME, GLOBEX, INITECH, NORTHWIND, sample

EXPECTED = {
    ACME: dict(vendor_name="Acme Industrial Supplies GmbH", invoice_number="ACME-2026-0311",
               invoice_date=date(2026, 3, 11), due_date=date(2026, 4, 10), po_number="PO-88213",
               currency="EUR", subtotal=2831.5, tax_rate=19.0, tax_amount=537.99, total=3369.49, n_items=4),
    NORTHWIND: dict(vendor_name="Northwind Logistics Inc.", invoice_number="NWL-48820",
                    invoice_date=date(2026, 4, 22), due_date=date(2026, 5, 22), currency="USD",
                    subtotal=6987.8, tax_amount=559.02, total=7546.82, n_items=3),
    GLOBEX: dict(vendor_name="Globex Consulting Services LLP", invoice_number="GCS/25-26/0142",
                 invoice_date=date(2026, 5, 15), due_date=date(2026, 6, 14), currency="INR",
                 subtotal=550000.0, tax_rate=18.0, total=649000.0, n_items=2),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_rule_extractor_on_samples(name):
    inv = RuleExtractor().extract(parse_pdf(sample(name)))
    exp = dict(EXPECTED[name])
    assert len(inv.line_items) == exp.pop("n_items")
    for field, value in exp.items():
        assert getattr(inv, field) == value, field
    assert validate_invoice(inv).valid


def test_validator_catches_the_planted_initech_error():
    inv = RuleExtractor().extract(parse_pdf(sample(INITECH)))
    report = validate_invoice(inv)
    assert not report.valid
    assert report.errors == ["Line items sum to 5,700.00 but subtotal is 5,900.00 (difference 200.00)."]
    assert inv.subtotal == 5900.0          # reported as printed, not "fixed"


def _good() -> Invoice:
    return Invoice(vendor_name="V", invoice_number="1", invoice_date=date(2026, 1, 1),
                   due_date=date(2026, 2, 1), currency="EUR",
                   line_items=[LineItem(description="a", quantity=2, unit_price=10, amount=20),
                               LineItem(description="b", quantity=1, unit_price=5, amount=5)],
                   subtotal=25, tax_rate=20, tax_amount=5, total=30)


@pytest.mark.parametrize("change, message", [
    ({"total": 31}, "Subtotal 25.00 + tax 5.00 = 30.00, but total is 31.00"),
    ({"tax_amount": 4, "total": 29}, "Tax at 20% of 25.00 is 5.00, but tax amount is 4.00"),
    ({"due_date": date(2025, 12, 1)}, "Due date 2025-12-01 is before invoice date 2026-01-01"),
    ({"currency": "€"}, "not a 3-letter ISO 4217 code"),
    ({"invoice_number": None}, "Missing required field 'invoice_number'"),
])
def test_validator_rules(change, message):
    assert validate_invoice(_good()).valid
    report = validate_invoice(_good().model_copy(update=change))
    assert not report.valid and any(message in e for e in report.errors), report.errors


def test_line_item_arithmetic():
    inv = _good()
    inv.line_items[0].amount = 25
    inv.subtotal = 30
    inv.tax_amount, inv.total = 6, 36
    report = validate_invoice(inv)
    assert report.errors == ["Line 1 ('a'): quantity 2 x unit price 10.00 = 20.00, but amount is 25.00."]


@pytest.mark.parametrize("raw, month_first, expected", [
    ("2026-03-11", False, date(2026, 3, 11)),
    ("04/22/2026", True, date(2026, 4, 22)),
    ("22/04/2026", False, date(2026, 4, 22)),
    ("05/04/2026", False, date(2026, 4, 5)),      # ambiguous: day-first outside the US
    ("05/04/2026", True, date(2026, 5, 4)),
    ("15 May 2026", False, date(2026, 5, 15)),
    ("not a date", False, None),
])
def test_date_parsing(raw, month_first, expected):
    assert _parse_date(raw, month_first) == expected


def test_fallback_extractor_survives_llm_failure():
    class Broken:
        name = "azure:gpt-4o"

        def extract(self, doc, feedback=None):
            raise ConnectionError("network down")

    ex = FallbackExtractor(Broken(), RuleExtractor())
    inv = ex.extract(parse_pdf(sample(ACME)))
    assert inv.total == 3369.49 and ex.name == "rules (fallback: ConnectionError)"
