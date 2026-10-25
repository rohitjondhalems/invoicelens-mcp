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


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunk_count: int
    ingested_at: str
