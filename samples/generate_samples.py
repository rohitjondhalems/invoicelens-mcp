"""
Generate synthetic invoice PDFs for demos and tests (no real company data).

    python samples/generate_samples.py            # writes samples/*.pdf

One invoice (Initech) has a deliberate arithmetic error — its stated subtotal
doesn't match its line items — so the validator has something real to catch.
Northwind has a second page of terms & conditions to exercise text chunking.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).parent


@dataclass
class SampleInvoice:
    filename: str
    vendor: str
    vendor_address: list[str]
    customer: str
    customer_address: list[str]
    invoice_number: str
    invoice_date: str          # printed as-is (formats vary on purpose)
    due_date: str
    currency: str
    tax_label: str
    tax_rate: float
    lines: list[tuple[str, float, float]]   # (description, qty, unit_price)
    po_number: str | None = None
    stated_subtotal: float | None = None    # override to inject an error
    terms: list[str] = field(default_factory=list)

    @property
    def subtotal(self) -> float:
        return round(sum(q * p for _, q, p in self.lines), 2)


def _money(v: float, cur: str) -> str:
    return f"{cur} {v:,.2f}"


SAMPLES = [
    SampleInvoice(
        filename="acme_supplies_2026-03.pdf",
        vendor="Acme Industrial Supplies GmbH",
        vendor_address=["Hafenstrasse 12", "20457 Hamburg, Germany"],
        customer="Polymer Works India Pvt Ltd",
        customer_address=["Plot 44, MIDC Taloja", "Navi Mumbai 410208, India"],
        invoice_number="ACME-2026-0311",
        invoice_date="2026-03-11",
        due_date="2026-04-10",
        currency="EUR",
        tax_label="VAT",
        tax_rate=19.0,
        po_number="PO-88213",
        lines=[
            ("Stainless steel ball valve DN50", 12, 145.00),
            ("PTFE gasket set, 100 pcs", 4, 89.50),
            ("Pressure gauge 0-16 bar", 6, 62.25),
            ("On-site calibration service (hours)", 3, 120.00),
        ],
    ),
    SampleInvoice(
        filename="northwind_logistics_2026-04.pdf",
        vendor="Northwind Logistics Inc.",
        vendor_address=["1200 Harbor Blvd", "Oakland, CA 94607, USA"],
        customer="Polymer Works India Pvt Ltd",
        customer_address=["Plot 44, MIDC Taloja", "Navi Mumbai 410208, India"],
        invoice_number="NWL-48820",
        invoice_date="04/22/2026",
        due_date="05/22/2026",
        currency="USD",
        tax_label="Sales tax",
        tax_rate=8.0,
        lines=[
            ("Ocean freight, 40ft container, Oakland to Nhava Sheva", 2, 3150.00),
            ("Customs documentation fee", 1, 275.00),
            ("Cargo insurance (0.3% of declared value)", 1, 412.80),
        ],
        terms=[
            "Payment terms. Payment is due within thirty (30) days of the invoice date. "
            "All amounts are payable in US dollars by wire transfer to the account shown "
            "on this invoice. Bank charges on the sender's side are borne by the customer.",
            "Late payment penalty. Invoices not paid by the due date accrue a late payment "
            "penalty of 1.5% per month on the outstanding balance, calculated daily from the "
            "due date until payment is received in full. Northwind Logistics may suspend "
            "further shipments while any invoice remains overdue by more than fifteen days.",
            "Liability. Carrier liability for loss or damage is limited as set out in the "
            "applicable bill of lading. Claims must be notified in writing within seven days "
            "of delivery, with photographs and a copy of the delivery receipt.",
            "Disputes. Any dispute regarding this invoice must be raised within ten business "
            "days of receipt. Undisputed portions remain payable on the original due date.",
        ],
    ),
    SampleInvoice(
        filename="globex_consulting_2026-05.pdf",
        vendor="Globex Consulting Services LLP",
        vendor_address=["9th Floor, Lotus Corporate Park", "Goregaon East, Mumbai 400063"],
        customer="Polymer Works India Pvt Ltd",
        customer_address=["Plot 44, MIDC Taloja", "Navi Mumbai 410208, India"],
        invoice_number="GCS/25-26/0142",
        invoice_date="15 May 2026",
        due_date="14 June 2026",
        currency="INR",
        tax_label="GST",
        tax_rate=18.0,
        lines=[
            ("Data platform assessment, phase 1", 1, 420000.00),
            ("ML model review workshop (days)", 2, 65000.00),
        ],
    ),
    SampleInvoice(
        filename="initech_software_2026-06.pdf",
        vendor="Initech Software Ltd",
        vendor_address=["4 Canal Street", "Manchester M1 3HE, United Kingdom"],
        customer="Polymer Works India Pvt Ltd",
        customer_address=["Plot 44, MIDC Taloja", "Navi Mumbai 410208, India"],
        invoice_number="INI-7731",
        invoice_date="2026-06-02",
        due_date="2026-07-02",
        currency="GBP",
        tax_label="VAT",
        tax_rate=20.0,
        lines=[
            ("Annual licence, document workflow suite (seats)", 25, 180.00),
            ("Premium support plan", 1, 1200.00),
        ],
        stated_subtotal=5900.00,   # true subtotal is 5,700.00 — deliberate error
    ),
]


def build_pdf(inv: SampleInvoice, path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Invoice {inv.invoice_number}", author=inv.vendor)
    story = [Paragraph(f"<b>{inv.vendor}</b>", styles["Title"])]
    story += [Paragraph(line, styles["Normal"]) for line in inv.vendor_address]
    story += [Spacer(1, 6 * mm), Paragraph("<b>INVOICE</b>", styles["Heading2"])]

    meta = [f"Invoice Number: {inv.invoice_number}",
            f"Invoice Date: {inv.invoice_date}",
            f"Due Date: {inv.due_date}"]
    if inv.po_number:
        meta.append(f"PO Number: {inv.po_number}")
    meta.append(f"Currency: {inv.currency}")
    story += [Paragraph(m, styles["Normal"]) for m in meta]

    story += [Spacer(1, 5 * mm), Paragraph("<b>Bill To:</b>", styles["Normal"]),
              Paragraph(inv.customer, styles["Normal"])]
    story += [Paragraph(line, styles["Normal"]) for line in inv.customer_address]
    story.append(Spacer(1, 6 * mm))

    rows = [["Description", "Qty", "Unit Price", "Amount"]]
    for desc, qty, price in inv.lines:
        rows.append([desc, f"{qty:g}", f"{price:,.2f}", f"{qty * price:,.2f}"])
    table = Table(rows, colWidths=[95 * mm, 15 * mm, 30 * mm, 32 * mm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F1FB")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [table, Spacer(1, 6 * mm)]

    subtotal = inv.stated_subtotal if inv.stated_subtotal is not None else inv.subtotal
    tax = round(subtotal * inv.tax_rate / 100, 2)
    story += [
        Paragraph(f"Subtotal: {_money(subtotal, inv.currency)}", styles["Normal"]),
        Paragraph(f"{inv.tax_label} ({inv.tax_rate:g}%): {_money(tax, inv.currency)}", styles["Normal"]),
        Paragraph(f"<b>Total Due: {_money(subtotal + tax, inv.currency)}</b>", styles["Normal"]),
        Spacer(1, 6 * mm),
        Paragraph("Thank you for your business. Please quote the invoice number with your payment.",
                  styles["Italic"]),
    ]
    if inv.terms:
        story += [PageBreak(), Paragraph("Terms and Conditions", styles["Heading2"])]
        for para in inv.terms:
            story += [Paragraph(para, styles["Normal"]), Spacer(1, 3 * mm)]
    doc.build(story)


def main(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for inv in SAMPLES:
        p = out_dir / inv.filename
        build_pdf(inv, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR):
        print("wrote", p)
