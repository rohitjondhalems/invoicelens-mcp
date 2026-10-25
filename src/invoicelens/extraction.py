"""
Invoice extraction: PDF content -> Invoice schema.

  LLMExtractor   Azure OpenAI / OpenAI structured outputs (response_format=Invoice),
                 temperature 0. Accepts `feedback` — validation errors from a
                 previous attempt — so an agent can ask for a corrected re-read.
  RuleExtractor  Regex + table-column mapping. Offline and deterministic; handles
                 conventional "Label: value" invoices like the samples. Used when no
                 LLM is configured, and as the fallback if the LLM call fails.

The extractor reports values AS PRINTED. It must not "fix" arithmetic — that is
the validator's job, and silently correcting a wrong invoice would hide a real
problem from the person paying it.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Protocol

from invoicelens.chunking import table_to_markdown
from invoicelens.config import Settings
from invoicelens.models import Invoice, LineItem
from invoicelens.parsing import ParsedDocument

log = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 40_000


class Extractor(Protocol):
    name: str

    def extract(self, doc: ParsedDocument, feedback: str | None = None) -> Invoice: ...


def document_as_text(doc: ParsedDocument) -> str:
    """Layout-preserving text for the LLM: blocks and markdown tables per page."""
    parts = []
    for page in doc.pages:
        parts.append(f"--- Page {page.page} ---")
        for el in page.elements():
            parts.append(table_to_markdown(el.rows) if hasattr(el, "rows") else el.text)
    text = "\n\n".join(parts)
    return text[:MAX_PROMPT_CHARS]


# ── LLM ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You extract structured data from invoices.

Rules:
- Report values exactly as printed. Do NOT recompute, correct or infer amounts; if the
  document's arithmetic is wrong, return the printed numbers anyway.
- If a field is not present, return null. Never invent values.
- Amounts are plain numbers: no currency symbols or thousands separators (1,234.50 -> 1234.5).
- Dates are ISO 8601 (YYYY-MM-DD). For ambiguous dd/mm vs mm/dd dates, use the issuer's
  country convention.
- currency is the ISO 4217 code (EUR, USD, GBP, INR ...), not a symbol.
- tax_rate is a percentage number (19 for 19%).
- line_items: one entry per row of the items table; amount is the printed line total."""


class LLMExtractor:
    def __init__(self, client, model: str, name: str):
        self._client, self._model, self.name = client, model, name

    def extract(self, doc: ParsedDocument, feedback: str | None = None) -> Invoice:
        user = f"Extract the invoice from this document:\n\n{document_as_text(doc)}"
        if feedback:
            user += ("\n\nA previous extraction failed these checks:\n" + feedback +
                     "\nRe-read the document carefully. If the document itself is "
                     "inconsistent, still report the printed values.")
        resp = self._client.chat.completions.parse(
            model=self._model, temperature=0, response_format=Invoice,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        )
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise RuntimeError(f"Model returned no parsable invoice (refusal: {msg.refusal!r})")
        return msg.parsed


# ── Rules ───────────────────────────────────────────────────────────────
_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP", "₹": "INR"}
_AMOUNT = r"(?:[A-Z]{3}|[€$£₹])?\s*(-?[\d][\d,]*(?:\.\d+)?)"
_MONTHS = "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _parse_date(s: str, month_first: bool) -> date | None:
    s = s.strip().rstrip(".")
    for fmt in _MONTHS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", s)
    if m:
        a, b, y = map(int, m.groups())
        if a > 12 or (b <= 12 and not month_first):
            a, b = b, a               # a was the day: convert to (month, day)
        try:
            return date(y, a, b)
        except ValueError:
            return None
    return None


def _labelled(text: str, label: str) -> str | None:
    m = re.search(rf"^\s*(?:{label})\s*[:#]\s*(.+?)\s*$", text, re.I | re.M)
    return m.group(1) if m else None


class RuleExtractor:
    name = "rules"

    def extract(self, doc: ParsedDocument, feedback: str | None = None) -> Invoice:
        first = doc.pages[0]
        text = "\n".join(b.text for p in doc.pages for b in p.blocks)
        lines = [l for b in first.blocks for l in b.lines]

        currency = _labelled(text, "currency")
        if not currency:
            m = re.search(r"\b(EUR|USD|GBP|INR|CHF|JPY|AUD|CAD|SGD|AED)\b", text)
            currency = m.group(1) if m else next((c for s, c in _SYMBOLS.items() if s in text), None)
        currency = currency.upper()[:3] if currency else None
        month_first = currency == "USD"

        inv = Invoice(currency=currency)
        inv.invoice_number = _labelled(text, r"invoice\s*(?:number|no\.?|#)")
        if d := _labelled(text, r"(?:invoice\s*)?date(?! due)"):
            inv.invoice_date = _parse_date(d, month_first)
        if d := _labelled(text, r"due\s*date|payment\s*due"):
            inv.due_date = _parse_date(d, month_first)
        inv.po_number = _labelled(text, r"p\.?o\.?\s*(?:number|no\.?|#)?")

        # Vendor: first line on the page; address: lines until the INVOICE title / first label
        if lines:
            inv.vendor_name = lines[0]
            addr = []
            for l in lines[1:]:
                if re.match(r"^(tax\s*)?invoice\b", l, re.I) or ":" in l:
                    break
                addr.append(l)
            inv.vendor_address = ", ".join(addr) or None
        for i, l in enumerate(lines):
            if re.match(r"^\s*bill\s*to\s*:?\s*$", l, re.I) and i + 1 < len(lines):
                inv.customer_name = lines[i + 1]
            elif m := re.match(r"^\s*bill\s*to\s*:\s*(.+)$", l, re.I):
                inv.customer_name = m.group(1)

        # Totals: scan lines that end in an amount
        for l in lines:
            low = l.lower()
            if m := re.search(rf"sub\s*-?\s*total\s*:?\s*{_AMOUNT}\s*$", l, re.I):
                inv.subtotal = _num(m.group(1))
            elif m := re.search(rf"^\s*(vat|gst|sales tax|tax)\b\s*(?:\((\d+(?:\.\d+)?)\s*%\))?\s*:?\s*{_AMOUNT}\s*$", l, re.I):
                inv.tax_label = m.group(1)
                inv.tax_rate = float(m.group(2)) if m.group(2) else None
                inv.tax_amount = _num(m.group(3))
            elif "total" in low and (m := re.search(rf"total(?: due| amount| payable)?\s*:?\s*{_AMOUNT}\s*$", l, re.I)):
                inv.total = _num(m.group(1))

        inv.line_items = self._line_items(doc)
        return inv

    @staticmethod
    def _line_items(doc: ParsedDocument) -> list[LineItem]:
        for page in doc.pages:
            for t in page.tables:
                header = [h.lower() for h in t.rows[0]]

                def col(*keys: str) -> int | None:
                    for k in keys:
                        for i, h in enumerate(header):
                            if k in h:
                                return i
                    return None

                c_desc = col("description", "item", "service", "particulars")
                c_amt = col("amount", "line total", "total")
                if c_desc is None or c_amt is None:
                    continue
                c_qty = col("qty", "quantity", "hours", "units")
                c_unit = col("unit price", "rate", "price")
                items = []
                for row in t.rows[1:]:
                    try:
                        amount = _num(row[c_amt])
                    except (ValueError, IndexError):
                        continue
                    def maybe(c):
                        try:
                            return _num(row[c]) if c is not None else None
                        except (ValueError, IndexError):
                            return None
                    items.append(LineItem(description=row[c_desc], quantity=maybe(c_qty),
                                          unit_price=maybe(c_unit), amount=amount))
                if items:
                    return items
        return []


class FallbackExtractor:
    """Try the LLM; on any failure, log it and use the rules instead."""

    def __init__(self, primary: Extractor, fallback: Extractor):
        self.primary, self.fallback = primary, fallback
        self.name = primary.name

    def extract(self, doc: ParsedDocument, feedback: str | None = None) -> Invoice:
        try:
            self.name = self.primary.name
            return self.primary.extract(doc, feedback)
        except Exception as e:
            log.warning("LLM extraction failed (%s); falling back to rules", e)
            self.name = f"{self.fallback.name} (fallback: {type(e).__name__})"
            return self.fallback.extract(doc, feedback)


def make_extractor(settings: Settings) -> Extractor:
    provider = settings.resolve(settings.extraction_provider)
    if provider == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(azure_endpoint=settings.azure_openai_endpoint,
                             api_key=settings.azure_openai_api_key,
                             api_version=settings.azure_openai_api_version, max_retries=2)
        llm = LLMExtractor(client, settings.azure_chat_deployment, f"azure:{settings.azure_chat_deployment}")
        return FallbackExtractor(llm, RuleExtractor())
    if provider == "openai":
        from openai import OpenAI

        llm = LLMExtractor(OpenAI(api_key=settings.openai_api_key, max_retries=2),
                           settings.openai_chat_model, f"openai:{settings.openai_chat_model}")
        return FallbackExtractor(llm, RuleExtractor())
    return RuleExtractor()
