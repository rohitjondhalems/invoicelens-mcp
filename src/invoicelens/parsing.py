"""
PDF -> structured page content.

Each page yields two kinds of element, which the hybrid chunker treats differently:
  - tables:  extracted as rows of cells (pdfplumber's ruled-line detection)
  - blocks:  runs of text lines OUTSIDE any table, split where the vertical gap
             between lines is clearly larger than normal line spacing
             (i.e. paragraph / section boundaries as the layout shows them)

Text-only PDFs only: scanned images have no text layer and are rejected with an
actionable error rather than silently producing empty chunks.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pdfplumber


class PDFError(ValueError):
    """Raised for unreadable, oversized or image-only PDFs."""


@dataclass
class TextBlock:
    text: str
    lines: list[str]
    top: float = 0.0          # y position on the page, for reading order


@dataclass
class Table:
    rows: list[list[str]]
    top: float = 0.0


@dataclass
class PageContent:
    page: int                                   # 1-based
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    def elements(self) -> list[TextBlock | Table]:
        """Blocks and tables interleaved in reading (top-to-bottom) order."""
        return sorted([*self.blocks, *self.tables], key=lambda e: e.top)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


@dataclass
class ParsedDocument:
    pages: list[PageContent]
    title: str | None = None

    @property
    def full_text(self) -> str:
        parts = []
        for p in self.pages:
            parts.append(p.text)
            for t in p.tables:
                parts.append("\n".join(" | ".join(r) for r in t.rows))
        return "\n".join(x for x in parts if x)


def _clean_cell(c: object) -> str:
    return " ".join(str(c).split()) if c is not None else ""


def _group_lines(lines: list[dict], gap_factor: float = 1.6) -> list[TextBlock]:
    """Split lines into blocks where the gap to the previous line is unusually large."""
    if not lines:
        return []
    heights = sorted(l["bottom"] - l["top"] for l in lines)
    typical = heights[len(heights) // 2] or 10.0
    blocks, current = [], [lines[0]]
    for prev, line in zip(lines, lines[1:]):
        gap = line["top"] - prev["bottom"]
        if gap > typical * (gap_factor - 1.0) + 1.5:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    blocks.append(current)
    out = []
    for b in blocks:
        texts = [" ".join(l["text"].split()) for l in b if l["text"].strip()]
        if texts:
            out.append(TextBlock(text="\n".join(texts), lines=texts, top=b[0]["top"]))
    return out


def parse_pdf(data: bytes, max_pages: int = 200) -> ParsedDocument:
    if not data.startswith(b"%PDF"):
        raise PDFError("Not a PDF file (missing %PDF header). Send the raw PDF bytes, base64-encoded.")
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as e:  # pdfminer raises a zoo of exception types
        raise PDFError(f"Could not open PDF: {e}") from e

    with pdf:
        if len(pdf.pages) > max_pages:
            raise PDFError(f"PDF has {len(pdf.pages)} pages; the limit is {max_pages}.")
        title = (pdf.metadata or {}).get("Title") or None
        pages: list[PageContent] = []
        for p in pdf.pages:
            found = p.find_tables()
            tables = []
            for t in found:
                rows = [[_clean_cell(c) for c in row] for row in t.extract()]
                rows = [r for r in rows if any(r)]
                if rows:
                    tables.append(Table(rows=rows, top=t.bbox[1]))

            # Text outside table regions, so table content isn't chunked twice
            text_page = p
            for t in found:
                text_page = text_page.outside_bbox(t.bbox)
            lines = sorted(text_page.extract_text_lines(), key=lambda l: (round(l["top"]), l["x0"]))
            pages.append(PageContent(page=p.page_number, blocks=_group_lines(lines), tables=tables))

    if not any(pg.blocks or pg.tables for pg in pages):
        raise PDFError("PDF has no extractable text (likely a scanned image). OCR it first, "
                       "e.g. with ocrmypdf, then ingest the result.")
    return ParsedDocument(pages=pages, title=title)
