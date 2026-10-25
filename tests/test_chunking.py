import pytest

from invoicelens.chunking import HybridChunker, count_tokens
from invoicelens.parsing import PageContent, ParsedDocument, PDFError, Table, TextBlock, parse_pdf

from .conftest import ACME, ALL, NORTHWIND, sample


def test_parser_separates_tables_from_text():
    doc = parse_pdf(sample(ACME))
    page = doc.pages[0]
    assert len(page.tables) == 1
    assert page.tables[0].rows[0] == ["Description", "Qty", "Unit Price", "Amount"]
    # table cell text must not also appear in the text blocks (no double chunking)
    assert "PTFE gasket" not in page.text


def test_rejects_non_pdf_and_image_only():
    with pytest.raises(PDFError, match="Not a PDF"):
        parse_pdf(b"hello")


@pytest.mark.parametrize("name", ALL)
def test_every_invoice_gets_structural_chunks(name):
    chunks = HybridChunker().chunk(parse_pdf(sample(name)))
    types = [c.chunk_type for c in chunks]
    assert types[:3] == ["header", "table", "totals"]
    header = chunks[0].text
    assert "Invoice Number:" in header and "Bill To:" in header
    # vendor name (first line of the page) stays with the invoice metadata
    assert header.splitlines()[0] == parse_pdf(sample(name)).pages[0].blocks[0].lines[0]


def test_tables_are_never_split_mid_row():
    rows = [["Description", "Qty", "Amount"]] + [[f"Item number {i} with a long description", "1", "10.00"]
                                                 for i in range(60)]
    doc = ParsedDocument(pages=[PageContent(page=1, tables=[Table(rows=rows)])])
    chunks = HybridChunker(max_tokens=120, overlap_tokens=10).chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[0] == "| Description | Qty | Amount |"        # header repeated in every piece
        assert all(l.startswith("|") and l.endswith("|") for l in lines)
        assert c.token_count <= 120
    body = [l for c in chunks for l in c.text.splitlines()[2:]]
    assert len(body) == 60                                          # no row lost or duplicated


def test_long_prose_is_windowed_with_overlap():
    sentences = [f"Clause {i} says the supplier must deliver goods number {i} on time." for i in range(40)]
    para = " ".join(sentences)
    doc = ParsedDocument(pages=[PageContent(page=2, blocks=[TextBlock(text=para, lines=[para])])])
    chunks = HybridChunker(max_tokens=100, overlap_tokens=20).chunk(doc)
    assert len(chunks) > 3 and all(c.chunk_type == "text" for c in chunks)
    assert all(c.token_count <= 100 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence_of_a = a.text.rsplit("Clause", 1)[1]
        assert last_sentence_of_a in b.text                          # sentence-level overlap
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)                       # nothing dropped


def test_terms_page_gets_section_from_heading():
    chunks = HybridChunker().chunk(parse_pdf(sample(NORTHWIND)))
    terms = [c for c in chunks if c.page == 2]
    assert terms and all(c.section == "Terms and Conditions" for c in terms)
    assert any("late payment penalty" in c.text for c in terms)


def test_embed_text_carries_context():
    c = HybridChunker().chunk(parse_pdf(sample(ACME)))[2]
    assert c.embed_text("acme.pdf").startswith("Document: acme.pdf | Page 1 | totals")


def test_count_tokens_is_reasonable():
    assert 8 <= count_tokens("Payment is due within thirty (30) days.") <= 12
