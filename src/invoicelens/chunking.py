"""
Hybrid chunking: structure first, size second.

A pure fixed-size splitter cuts invoices in the worst places — through the
middle of a line-item table, or splitting "Invoice Number:" from its value.
A pure structural splitter produces wildly uneven chunks (one 3-word heading,
one 2,000-token T&C page). This chunker does both, in order:

1. STRUCTURE — walk each page's elements in reading order and classify them:
     table    -> its own chunk, serialised as a markdown table. Never split
                 mid-row; an oversized table is split by rows with the header
                 row repeated in every piece, so each piece stands alone.
     totals   -> a key:value block mentioning subtotal/tax/total gets its own
                 chunk: it's what "how much is X?" queries need.
     header   -> consecutive compact blocks (vendor, address, invoice no/dates,
                 bill-to) are merged into one chunk, so the metadata a query
                 matches on stays together.
     heading  -> a short title line; not a chunk by itself, it becomes the
                 `section` of the chunks that follow.
     prose    -> paragraphs; passed to step 2.
2. SIZE — prose is packed paragraph by paragraph into windows of at most
   `max_tokens`, with `overlap_tokens` of sentence-level overlap between
   consecutive windows. A single paragraph longer than the window is split on
   sentence boundaries.

Every chunk also gets a contextual header (file, page, type, section) that is
embedded together with the text, so a chunk like "Payment is due within 30
days" is still findable by the vendor it belongs to.

Token counts are approximate (word/punctuation units, ~= tiktoken within 10-20%
for English). That's deliberate: exact tokenisation would need a downloaded
vocabulary, and window sizes don't need to be exact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from invoicelens.parsing import ParsedDocument, Table, TextBlock

ChunkType = Literal["header", "table", "totals", "text"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_KV_LINE_RE = re.compile(r"^[A-Za-z][\w .#/()%&-]{0,40}:\s*\S")
_TOTALS_RE = re.compile(r"\b(sub\s*total|total|tax|vat|gst|amount due|balance)\b", re.I)


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


@dataclass
class Chunk:
    index: int
    chunk_type: ChunkType
    page: int
    text: str
    section: str | None = None

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    def embed_text(self, filename: str) -> str:
        """Text sent to the embedding model: contextual header + content."""
        ctx = f"Document: {filename} | Page {self.page} | {self.chunk_type}"
        if self.section:
            ctx += f" | Section: {self.section}"
        return f"{ctx}\n{self.text}"


def table_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = [[c.replace("|", "\\|") for c in r] for r in rows]
    lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in esc[1:]]
    return "\n".join(lines)


def _is_heading(b: TextBlock) -> bool:
    if len(b.lines) != 1:
        return False
    line = b.lines[0]
    return (len(line.split()) <= 8 and not line.rstrip().endswith((".", ":", ","))
            and not _KV_LINE_RE.match(line) and not any(ch.isdigit() for ch in line))


def _is_kv(b: TextBlock) -> bool:
    return sum(bool(_KV_LINE_RE.match(l)) for l in b.lines) >= max(1, len(b.lines) / 2)


def _is_compact(b: TextBlock) -> bool:
    """Short label-like lines (names, addresses, IDs) rather than sentences."""
    return (len(b.lines) <= 6 and all(len(l.split()) <= 10 for l in b.lines)
            and len(_SENTENCE_RE.split(b.text)) < 2)


def _classify(b: TextBlock) -> str:
    if _is_kv(b):
        return "totals" if _TOTALS_RE.search(b.text) else "header"
    if _is_compact(b) and not _is_heading(b):
        return "header"
    if _is_heading(b):
        return "heading"
    return "prose"


class HybridChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ── public ──────────────────────────────────────────────────────────
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []

        def emit(kind: ChunkType, page: int, text: str, section: str | None):
            text = text.strip()
            if text:
                chunks.append(Chunk(len(chunks), kind, page, text, section))

        for page in doc.pages:
            section: str | None = None
            header_buf: list[str] = []
            prose_buf: list[str] = []
            pending_heading: str | None = None

            def flush_header():
                if header_buf:
                    emit("header", page.page, "\n".join(header_buf), section)
                    header_buf.clear()

            def flush_prose():
                if prose_buf:
                    for window in self._pack(prose_buf):
                        emit("text", page.page, window, section)
                    prose_buf.clear()

            for el in page.elements():
                if isinstance(el, Table):
                    pending_heading = None
                    flush_header(); flush_prose()
                    for piece in self._split_table(el.rows):
                        emit("table", page.page, piece, section)
                    continue
                kind = _classify(el)
                if kind == "heading":
                    flush_prose()
                    section = el.text
                    # A title right before/inside a header run (vendor name at the
                    # top, "INVOICE" between vendor and invoice no.) belongs in it.
                    if header_buf:
                        header_buf.append(el.text)
                    else:
                        pending_heading = el.text
                    continue
                if kind == "header" and pending_heading:
                    header_buf.append(pending_heading)
                pending_heading = None
                if kind == "totals":
                    flush_header(); flush_prose()
                    emit("totals", page.page, el.text, section)
                elif kind == "header":
                    flush_prose()
                    header_buf.append(el.text)
                    # keep header chunks bounded too
                    if count_tokens("\n".join(header_buf)) > self.max_tokens:
                        last = header_buf.pop()
                        flush_header()
                        header_buf.append(last)
                else:
                    flush_header()
                    prose_buf.append(el.text.replace("\n", " "))
            flush_header(); flush_prose()
        return chunks

    # ── size control ────────────────────────────────────────────────────
    def _pack(self, paragraphs: list[str]) -> list[str]:
        """Pack sentences into <= max_tokens windows with sentence-level overlap."""
        sentences: list[str] = []
        for para in paragraphs:
            parts = _SENTENCE_RE.split(para)
            sentences.extend(s.strip() for s in parts if s.strip())
            sentences.append("\n")  # paragraph boundary marker

        windows, current = [], []
        for s in sentences:
            if s == "\n":
                if current and current[-1] != "\n":
                    current.append("\n")
                continue
            if current and count_tokens(" ".join(current + [s])) > self.max_tokens:
                windows.append(current)
                current = self._overlap_tail(current)
            if count_tokens(s) > self.max_tokens:      # one giant sentence
                for piece in self._hard_split(s):
                    windows.append([piece])
                current = []
                continue
            current.append(s)
        if any(x != "\n" for x in current):
            windows.append(current)

        out = []
        for w in windows:
            text = " ".join(w).replace(" \n ", "\n\n").replace(" \n", "").replace("\n ", "\n")
            out.append(text.strip())
        return [t for t in out if t]

    def _overlap_tail(self, window: list[str]) -> list[str]:
        tail, n = [], 0
        for s in reversed([x for x in window if x != "\n"]):
            n += count_tokens(s)
            if n > self.overlap_tokens:
                break
            tail.insert(0, s)
        return tail

    def _hard_split(self, sentence: str) -> list[str]:
        words = sentence.split()
        step = max(1, int(self.max_tokens * 0.75))
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    def _split_table(self, rows: list[list[str]]) -> list[str]:
        md = table_to_markdown(rows)
        if count_tokens(md) <= self.max_tokens or len(rows) <= 2:
            return [md]
        header, body = rows[0], rows[1:]
        pieces, current = [], []
        for row in body:
            if current and count_tokens(table_to_markdown([header, *current, row])) > self.max_tokens:
                pieces.append(table_to_markdown([header, *current]))
                current = []
            current.append(row)
        if current:
            pieces.append(table_to_markdown([header, *current]))
        return pieces
