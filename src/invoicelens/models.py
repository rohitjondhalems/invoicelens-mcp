"""Pydantic schemas shared by the service, MCP tools and REST API."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


# ── Invoice (the extraction target) ─────────────────────────────────────
class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float = Field(description="Line total as printed on the invoice")


class Invoice(BaseModel):
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    po_number: str | None = None
    currency: str | None = Field(None, description="ISO 4217 code, e.g. EUR, USD, INR")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax_label: str | None = Field(None, description="As printed: VAT, GST, Sales tax ...")
    tax_rate: float | None = Field(None, description="Percent, e.g. 19 for 19%")
    tax_amount: float | None = None
    total: float | None = None


class ValidationReport(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list, description="Must-fix problems: wrong arithmetic, missing required fields")
    warnings: list[str] = Field(default_factory=list, description="Worth a look but not necessarily wrong")


# ── Tool / endpoint results ─────────────────────────────────────────────
class IngestResult(BaseModel):
    doc_id: str = Field(description="Stable id (content hash) — pass to search or extract")
    filename: str
    collection: str
    pages: int
    chunks: int
    chunks_by_type: dict[str, int]
    replaced_existing: bool = Field(description="True if this exact file was already ingested and was re-indexed")
    embedder: str


class SearchResultItem(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_type: Literal["header", "table", "totals", "text"]
    section: str | None
    score: float = Field(description="Cosine similarity, higher is closer")
    text: str


class SearchResult(BaseModel):
    query: str
    collection: str
    results: list[SearchResultItem]


class ExtractResult(BaseModel):
    invoice: Invoice
    validation: ValidationReport
    extractor: str
    doc_id: str | None = None
    filename: str | None = None


class WebSearchItem(BaseModel):
    title: str
    url: str
    content: str = Field(description="Relevant snippet from the page")
    score: float | None = None
    published_date: str | None = None


class WebSearchResult(BaseModel):
    query: str
    topic: str
    provider: str
    retrieved_at: str
    answer: str | None = Field(None, description="Provider's short summary, when requested")
    results: list[WebSearchItem]


class GrowthPeriod(BaseModel):
    fiscal_year: str | None = None
    period_end: str | None = None
    revenue_growth: float | None = Field(None, description="Year-over-year, as a fraction: 0.12 = +12%")
    net_income_growth: float | None = None
    eps_growth: float | None = None


class CompanyFinancials(BaseModel):
    query: str
    symbol: str
    resolved_by: str = Field(description="'ticker' if given directly, else 'name search' — confirm the match")
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    price: float | None = None
    change_percent: float | None = Field(None, description="Today's price change, percent")
    market_cap: float | None = Field(None, description="In `currency`")
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    ceo: str | None = None
    website: str | None = None
    employees: int | None = None
    ipo_date: str | None = None
    is_actively_trading: bool | None = None
    description: str | None = None
    growth: list[GrowthPeriod] = Field(default_factory=list, description="Most recent fiscal years first")
    notes: list[str] = Field(default_factory=list)
    provider: str
    retrieved_at: str
    disclaimer: str = "Market data for information only; not investment advice. Prices may be delayed."


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunk_count: int
    ingested_at: str
