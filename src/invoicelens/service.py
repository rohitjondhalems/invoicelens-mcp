"""
The one core both interfaces call. MCP tools and REST endpoints are thin
adapters over these methods, so they cannot drift apart (a test checks this).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from invoicelens.chunking import HybridChunker
from invoicelens.config import Settings, get_settings
from invoicelens.embeddings import Embedder, make_embedder
from invoicelens.external import ExternalDataError, FMPFinancials, TavilySearch
from invoicelens.extraction import Extractor, make_extractor
from invoicelens.models import (CompanyFinancials, DocumentInfo, ExtractResult, IngestResult,
                                SearchResult, SearchResultItem, WebSearchResult)
from invoicelens.parsing import PDFError, parse_pdf
from invoicelens.store import StoreError, VectorStore, validate_collection
from invoicelens.validation import validate_invoice


class InputError(ValueError):
    """Bad caller input; the message tells the caller how to fix it."""


def doc_id_for(pdf_bytes: bytes) -> str:
    """Content hash: the same file always gets the same id, so re-ingest replaces."""
    return "doc_" + hashlib.sha256(pdf_bytes).hexdigest()[:16]


class InvoiceLensService:
    def __init__(self, settings: Settings | None = None,
                 embedder: Embedder | None = None, extractor: Extractor | None = None,
                 web: TavilySearch | None = None, market: FMPFinancials | None = None):
        self.settings = settings or get_settings()
        self.embedder = embedder or make_embedder(self.settings)
        self.extractor = extractor or make_extractor(self.settings)
        s = self.settings
        self.web = web or TavilySearch(s.tavily_api_key, timeout=s.external_timeout_s)
        self.market = market or FMPFinancials(s.fmp_api_key, timeout=s.external_timeout_s,
                                              cache_seconds=s.market_data_cache_s)
        self.chunker = HybridChunker(self.settings.chunk_max_tokens, self.settings.chunk_overlap_tokens)
        self._stores: dict[str, VectorStore] = {}
        self._lock = threading.Lock()

    # ── helpers ─────────────────────────────────────────────────────────
    def store(self, collection: str | None = None) -> VectorStore:
        name = validate_collection(collection or self.settings.default_collection)
        with self._lock:
            if name not in self._stores:
                self._stores[name] = VectorStore(self.settings.data_dir, name,
                                                 self.embedder.name, self.embedder.dim)
            return self._stores[name]

    def load_pdf(self, *, pdf_base64: str | None = None, path: str | None = None,
                 data: bytes | None = None) -> tuple[bytes, str | None]:
        """Resolve exactly one PDF source. Returns (bytes, filename-if-known)."""
        given = [x for x in (pdf_base64, path, data) if x]
        if len(given) != 1:
            raise InputError("Provide exactly one of: pdf_base64 (the PDF file, base64-encoded) "
                             "or path (a file on the server).")
        name = None
        if pdf_base64:
            try:
                data = base64.b64decode(pdf_base64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise InputError(f"pdf_base64 is not valid base64: {e}") from e
        elif path:
            if not self.settings.allow_local_paths:
                raise InputError("Reading server-side paths is disabled on this server. "
                                 "Send the file as pdf_base64 instead.")
            p = Path(path).expanduser()
            if not p.is_file():
                raise InputError(f"File not found on the server: {path}. If the PDF is on your "
                                 "machine and the server is remote, send it as pdf_base64.")
            data, name = p.read_bytes(), p.name
        if len(data) > self.settings.max_pdf_bytes:
            raise InputError(f"PDF is {len(data) / 1e6:.1f} MB; the limit is "
                             f"{self.settings.max_pdf_bytes / 1e6:.0f} MB.")
        return data, name

    # ── ingestion ───────────────────────────────────────────────────────
    def ingest(self, pdf_bytes: bytes, filename: str | None = None,
               collection: str | None = None) -> IngestResult:
        store = self.store(collection)
        parsed = parse_pdf(pdf_bytes, self.settings.max_pages)
        doc_id = doc_id_for(pdf_bytes)
        filename = filename or parsed.title or f"{doc_id}.pdf"

        chunks = self.chunker.chunk(parsed)
        vectors = self.embedder.embed([c.embed_text(filename) for c in chunks])
        replaced = store.add_document(
            doc_id,
            {"filename": filename, "pages": len(parsed.pages),
             "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
             "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            pdf_bytes,
            [{"page": c.page, "chunk_type": c.chunk_type, "section": c.section, "text": c.text}
             for c in chunks],
            vectors,
        )
        return IngestResult(doc_id=doc_id, filename=filename, collection=store.collection,
                            pages=len(parsed.pages), chunks=len(chunks),
                            chunks_by_type=dict(Counter(c.chunk_type for c in chunks)),
                            replaced_existing=replaced, embedder=self.embedder.name)

    # ── retrieval ───────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5, collection: str | None = None,
               doc_id: str | None = None, chunk_types: list[str] | None = None) -> SearchResult:
        if not query.strip():
            raise InputError("query is empty.")
        store = self.store(collection)
        qvec = self.embedder.embed([query])[0]
        hits = store.search(qvec, k=k, doc_id=doc_id, chunk_types=set(chunk_types) if chunk_types else None)
        return SearchResult(query=query, collection=store.collection, results=[
            SearchResultItem(doc_id=h.chunk.doc_id, filename=h.filename, page=h.chunk.page,
                             chunk_type=h.chunk.chunk_type, section=h.chunk.section,
                             score=round(h.score, 4), text=h.chunk.text)
            for h in hits])

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        store = self.store(collection)
        return [DocumentInfo(doc_id=d, filename=m["filename"], pages=m["pages"],
                             chunk_count=m["chunk_count"], ingested_at=m["ingested_at"])
                for d, m in sorted(store.documents.items(), key=lambda kv: kv[1]["ingested_at"])]

    # ── extraction ──────────────────────────────────────────────────────
    def extract(self, pdf_bytes: bytes | None = None, doc_id: str | None = None,
                collection: str | None = None, feedback: str | None = None,
                filename: str | None = None) -> ExtractResult:
        if (pdf_bytes is None) == (doc_id is None):
            raise InputError("Provide either a PDF (pdf_base64 / path) or a doc_id of an ingested document.")
        if doc_id:
            store = self.store(collection)
            pdf_bytes = store.get_pdf(doc_id)
            filename = store.documents[doc_id]["filename"]
        parsed = parse_pdf(pdf_bytes, self.settings.max_pages)
        invoice = self.extractor.extract(parsed, feedback=feedback)
        return ExtractResult(invoice=invoice, validation=validate_invoice(invoice),
                             extractor=self.extractor.name,
                             doc_id=doc_id or doc_id_for(pdf_bytes), filename=filename)


    # ── external data ───────────────────────────────────────────────────
    def web_search(self, query: str, topic: str = "general", time_range: str | None = None,
                   max_results: int = 5, include_domains: list[str] | None = None,
                   include_answer: bool = False) -> WebSearchResult:
        if not query.strip():
            raise InputError("query is empty.")
        return self.web.search(query, topic=topic, time_range=time_range, max_results=max_results,
                               include_domains=include_domains, include_answer=include_answer)

    def company_financials(self, company: str, include_growth: bool = True) -> CompanyFinancials:
        if not company.strip():
            raise InputError("company is empty: pass a name ('Microsoft') or a ticker ('MSFT').")
        return self.market.company(company, include_growth=include_growth)


__all__ = ["InvoiceLensService", "InputError", "PDFError", "StoreError", "ExternalDataError", "doc_id_for"]
