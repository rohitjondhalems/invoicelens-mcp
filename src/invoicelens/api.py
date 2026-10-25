"""
REST API + MCP in one process.

    invoicelens-serve                   # http://127.0.0.1:8765
      POST /ingest      multipart file upload            -> IngestResult
      POST /search      JSON {query, k, ...}             -> SearchResult
      POST /extract     multipart file, or form doc_id   -> ExtractResult
      GET  /documents                                    -> [DocumentInfo]
      GET  /company-financials?company=MSFT              -> CompanyFinancials (FMP)
      POST /web-search  JSON {query, topic, ...}         -> WebSearchResult (Tavily)
      GET  /health
      POST /mcp         the MCP endpoint (Streamable HTTP) — same tools as stdio
      GET  /docs        Swagger UI

Both halves share ONE InvoiceLensService instance, so a file uploaded through
/ingest is immediately searchable from an MCP client, and vice versa.
"""
from __future__ import annotations

import argparse
import contextlib
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from invoicelens import __version__
from invoicelens.auth import BearerAuthMiddleware
from invoicelens.mcp_server import build_server
from invoicelens.models import (CompanyFinancials, DocumentInfo, ExtractResult, IngestResult,
                                SearchResult, WebSearchResult)
from invoicelens.service import (ExternalDataError, InputError, InvoiceLensService, PDFError,
                                 StoreError)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000, examples=["late payment penalty"])
    k: int = Field(5, ge=1, le=20)
    doc_id: str | None = None
    chunk_types: list[Literal["header", "table", "totals", "text"]] | None = None
    collection: str | None = None


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400, examples=["Microsoft cloud revenue growth"])
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None
    max_results: int = Field(5, ge=1, le=10)
    include_domains: list[str] | None = None


def create_app(service: InvoiceLensService | None = None, host: str = "127.0.0.1") -> FastAPI:
    """`host` must be the address the server binds to: bound to localhost, the MCP SDK
    enables DNS-rebinding protection (only localhost Host headers accepted); bound to
    0.0.0.0 (Docker, cloud) it doesn't, so set INVOICELENS_API_TOKEN there."""
    service = service or InvoiceLensService()
    mcp = build_server(service, host=host)
    mcp_app = mcp.streamable_http_app()       # creates mcp.session_manager

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Mounted sub-apps don't get their own lifespan; run the MCP session manager here
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="InvoiceLens", version=__version__, lifespan=lifespan,
                  description="Invoice PDF ingestion (hybrid chunking + FAISS), semantic search and "
                              "validated extraction. The same tools are served over MCP at /mcp.")

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (InputError, PDFError) as e:
            raise HTTPException(422, str(e)) from e
        except StoreError as e:
            raise HTTPException(404 if "No document" in str(e) else 409, str(e)) from e
        except ExternalDataError as e:
            raise HTTPException(502, str(e)) from e

    async def read_upload(file: UploadFile) -> bytes:
        data = await file.read(service.settings.max_pdf_bytes + 1)
        if len(data) > service.settings.max_pdf_bytes:
            raise HTTPException(413, f"PDF exceeds {service.settings.max_pdf_bytes / 1e6:.0f} MB")
        return data

    @app.get("/health", tags=["info"])
    def health():
        return {"status": "ok", "version": __version__, "embedder": service.embedder.name,
                "extractor": service.extractor.name}

    @app.post("/ingest", response_model=IngestResult, tags=["ingestion"])
    async def ingest(file: UploadFile = File(..., description="Invoice PDF"),
                     collection: str | None = Form(None)):
        """Upload a PDF to parse, hybrid-chunk, embed and index into FAISS."""
        data = await read_upload(file)
        return call(service.ingest, data, file.filename, collection)

    @app.post("/search", response_model=SearchResult, tags=["retrieval"])
    def search(req: SearchRequest):
        """Semantic search over ingested documents."""
        return call(service.search, req.query, k=req.k, collection=req.collection,
                    doc_id=req.doc_id, chunk_types=req.chunk_types)

    @app.post("/extract", response_model=ExtractResult, tags=["extraction"])
    async def extract(file: UploadFile | None = File(None, description="Invoice PDF (or use doc_id)"),
                      doc_id: str | None = Form(None),
                      feedback: str | None = Form(None),
                      collection: str | None = Form(None)):
        """Extract structured invoice fields with arithmetic validation."""
        if (file is None) == (doc_id is None):
            raise HTTPException(422, "Send either a file or a doc_id.")
        if doc_id:
            return call(service.extract, doc_id=doc_id, collection=collection, feedback=feedback)
        data = await read_upload(file)
        return call(service.extract, pdf_bytes=data, feedback=feedback, filename=file.filename)

    @app.get("/documents", response_model=list[DocumentInfo], tags=["ingestion"])
    def documents(collection: str | None = None):
        return call(service.list_documents, collection)

    @app.get("/company-financials", response_model=CompanyFinancials, tags=["market data"])
    def company_financials(company: str, include_growth: bool = True):
        """Share price, market cap and growth for a listed company (Financial Modeling Prep)."""
        return call(service.company_financials, company, include_growth=include_growth)

    @app.post("/web-search", response_model=WebSearchResult, tags=["market data"])
    def web_search(req: WebSearchRequest):
        """Web search with source URLs (Tavily)."""
        return call(service.web_search, req.query, topic=req.topic, time_range=req.time_range,
                    max_results=req.max_results, include_domains=req.include_domains)

    app.mount("/", mcp_app)                   # serves /mcp; registered last so API routes win
    app.add_middleware(BearerAuthMiddleware, token=service.settings.api_token)
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="InvoiceLens REST API + MCP over HTTP")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(create_app(host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
