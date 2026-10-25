"""
InvoiceLens MCP server — three tools over stdio or Streamable HTTP.

    invoicelens-mcp                          # stdio (Claude Desktop, Claude Code)
    invoicelens-mcp --transport http         # http://127.0.0.1:8765/mcp

Tools are thin adapters over InvoiceLensService; all real logic lives there.
Errors are raised as ToolError with a message that tells the calling agent
what to do next, since the agent — not a human — reads them first.
"""
from __future__ import annotations

import argparse
import logging
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from invoicelens.models import (CompanyFinancials, ExtractResult, IngestResult, SearchResult,
                                WebSearchResult)
from invoicelens.service import (ExternalDataError, InputError, InvoiceLensService, PDFError,
                                 StoreError)

INSTRUCTIONS = """InvoiceLens works with invoice PDFs, plus vendor research on the web.
- invoicelens_ingest_document: index a PDF so it becomes searchable. Returns a doc_id.
- invoicelens_search_documents: find relevant passages across ingested PDFs (semantic search).
  Use it to locate which document answers a question, then cite filename and page.
- invoicelens_extract_invoice: read ONE invoice into structured fields, with arithmetic checks.
  Pass a doc_id from search/ingest, or a PDF directly.
- invoicelens_company_financials: share price, market cap, sector and annual growth for a
  listed company (by name or ticker). Use this for numbers, not web search.
- invoicelens_web_search: current web results (news, growth, strategy) with source URLs.
If extraction returns validation.valid = false, call it again with `feedback` set to the
validation errors. If it still fails, report the discrepancy instead of guessing numbers:
some invoices really are wrong. Always state the source and retrieval date of market data."""

PdfBase64 = Annotated[str | None, Field(description="The PDF file content, base64-encoded. Use this when the server is remote.")]
PdfPath = Annotated[str | None, Field(description="Path to a PDF on the server's filesystem (local/stdio use only).")]
Collection = Annotated[str | None, Field(description="Index to use, e.g. 'default' or a per-client name. Lowercase letters, digits, '-', '_'.")]


def build_server(service: InvoiceLensService | None = None, host: str = "127.0.0.1",
                 port: int = 8765) -> FastMCP:
    mcp = FastMCP("invoicelens_mcp", instructions=INSTRUCTIONS, host=host, port=port,
                  stateless_http=True, json_response=True)
    _svc: dict[str, InvoiceLensService] = {}

    def svc() -> InvoiceLensService:
        # Lazy: start-up stays fast and config errors surface on first use, as a tool error
        if "s" not in _svc:
            _svc["s"] = service or InvoiceLensService()
        return _svc["s"]

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (InputError, PDFError, StoreError, ExternalDataError) as e:
            raise ToolError(str(e)) from e

    @mcp.tool(
        name="invoicelens_ingest_document",
        annotations=ToolAnnotations(title="Ingest invoice PDF", readOnlyHint=False,
                                    destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    def ingest_document(
        pdf_base64: PdfBase64 = None,
        path: PdfPath = None,
        filename: Annotated[str | None, Field(description="Display name, e.g. 'acme_march.pdf'. Defaults to the file name or PDF title.")] = None,
        collection: Collection = None,
    ) -> IngestResult:
        """Parse a PDF, split it with hybrid chunking (tables, header fields, totals and text kept
        as separate chunks), embed the chunks and store them in the FAISS index.

        Ingesting the same file again replaces its previous version (doc_id is a content hash).
        Returns the doc_id plus chunk counts by type."""
        s = svc()
        data, name = run(s.load_pdf, pdf_base64=pdf_base64, path=path)
        return run(s.ingest, data, filename or name, collection)

    @mcp.tool(
        name="invoicelens_search_documents",
        annotations=ToolAnnotations(title="Search ingested invoices", readOnlyHint=True,
                                    destructiveHint=False, idempotentHint=True, openWorldHint=False),
    )
    def search_documents(
        query: Annotated[str, Field(description="What to look for, in natural language, e.g. 'late payment penalty' or 'Acme invoice total'.", min_length=1, max_length=1000)],
        k: Annotated[int, Field(description="Number of passages to return.", ge=1, le=20)] = 5,
        doc_id: Annotated[str | None, Field(description="Restrict to one document.")] = None,
        chunk_types: Annotated[list[Literal["header", "table", "totals", "text"]] | None,
                               Field(description="Restrict to chunk types, e.g. ['totals'] for amounts or ['text'] for terms.")] = None,
        collection: Collection = None,
    ) -> SearchResult:
        """Semantic search over ingested invoices. Returns the top-k passages with doc_id,
        filename, page, chunk type and similarity score — cite filename and page in answers.
        Use the doc_id with invoicelens_extract_invoice to read a document's fields precisely."""
        return run(svc().search, query, k=k, collection=collection, doc_id=doc_id, chunk_types=chunk_types)

    @mcp.tool(
        name="invoicelens_extract_invoice",
        annotations=ToolAnnotations(title="Extract invoice fields", readOnlyHint=True,
                                    destructiveHint=False, idempotentHint=True, openWorldHint=True),
    )
    def extract_invoice(
        doc_id: Annotated[str | None, Field(description="An ingested document's doc_id (from ingest or search).")] = None,
        pdf_base64: PdfBase64 = None,
        path: PdfPath = None,
        feedback: Annotated[str | None, Field(description="Validation errors from a previous attempt, to guide a corrected re-read.")] = None,
        collection: Collection = None,
    ) -> ExtractResult:
        """Extract one invoice into structured fields (vendor, invoice number, dates, currency,
        line items, subtotal, tax, total) and check them: line items vs subtotal, subtotal + tax
        vs total, tax rate, dates.

        validation.valid = false means the numbers don't add up — either a misread (retry with
        `feedback`) or an invoice that is genuinely wrong. Values are reported as printed, never
        silently corrected."""
        s = svc()
        if doc_id:
            if pdf_base64 or path:
                raise ToolError("Pass either doc_id or a PDF (pdf_base64/path), not both.")
            return run(s.extract, doc_id=doc_id, collection=collection, feedback=feedback)
        data, name = run(s.load_pdf, pdf_base64=pdf_base64, path=path)
        return run(s.extract, pdf_bytes=data, feedback=feedback, filename=name)

    @mcp.tool(
        name="invoicelens_company_financials",
        annotations=ToolAnnotations(title="Company market data", readOnlyHint=True,
                                    destructiveHint=False, idempotentHint=True, openWorldHint=True),
    )
    def company_financials(
        company: Annotated[str, Field(description="Company name ('Microsoft') or ticker ('MSFT'). For non-US companies, a US/ADR ticker works best ('INFY' for Infosys).", min_length=1, max_length=120)],
        include_growth: Annotated[bool, Field(description="Also fetch up to 3 years of revenue, net income and EPS growth.")] = True,
    ) -> CompanyFinancials:
        """Share price, market capitalisation, sector/industry, CEO, employees and annual growth
        (revenue, net income, EPS; fractions, 0.12 = +12%) for a publicly listed company, from
        Financial Modeling Prep.

        Check `resolved_by`: if it was found by name search, confirm `name` matches what was
        meant. Private companies aren't covered. Quote `retrieved_at` with the numbers; prices may
        be delayed. For context or news, use invoicelens_web_search."""
        return run(svc().company_financials, company, include_growth=include_growth)

    @mcp.tool(
        name="invoicelens_web_search",
        annotations=ToolAnnotations(title="Web search", readOnlyHint=True,
                                    destructiveHint=False, idempotentHint=False, openWorldHint=True),
    )
    def web_search(
        query: Annotated[str, Field(description="Search query, e.g. 'Microsoft cloud revenue growth 2026'.", min_length=1, max_length=400)],
        topic: Annotated[Literal["general", "news", "finance"], Field(description="'news' for recent events, 'finance' for markets and companies.")] = "general",
        time_range: Annotated[Literal["day", "week", "month", "year"] | None, Field(description="Only results from this recent period.")] = None,
        max_results: Annotated[int, Field(ge=1, le=10)] = 5,
        include_domains: Annotated[list[str] | None, Field(description="Restrict to sites, e.g. ['reuters.com', 'sec.gov'].", max_length=20)] = None,
    ) -> WebSearchResult:
        """Search the web (Tavily) and return ranked results with title, URL, snippet and
        publication date. Use for recent news, growth narrative and strategy; cite the URLs.
        Don't take share prices or market caps from snippets — use
        invoicelens_company_financials for numbers."""
        return run(svc().web_search, query, topic=topic, time_range=time_range,
                   max_results=max_results, include_domains=include_domains)

    return mcp


def http_app(mcp: FastMCP, token: str | None):
    """Streamable HTTP ASGI app, with the optional bearer-token check."""
    from invoicelens.auth import BearerAuthMiddleware

    return BearerAuthMiddleware(mcp.streamable_http_app(), token)


def main() -> None:
    ap = argparse.ArgumentParser(description="InvoiceLens MCP server")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    # stdout is the protocol channel in stdio mode: log to stderr only
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    mcp = build_server(host=args.host, port=args.port)
    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        import uvicorn

        from invoicelens.config import get_settings
        uvicorn.run(http_app(mcp, get_settings().api_token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
