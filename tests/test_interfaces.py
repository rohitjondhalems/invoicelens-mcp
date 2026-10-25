"""MCP tools and REST endpoints: behaviour, errors, auth, and parity between the two."""
import base64
import json

import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from invoicelens.api import create_app
from invoicelens.config import Settings
from invoicelens.mcp_server import build_server
from invoicelens.service import InvoiceLensService

from .conftest import ACME, INITECH, NORTHWIND, SAMPLES, sample


def b64(name: str) -> str:
    return base64.b64encode(sample(name)).decode()


async def call(server, tool, args):
    async with create_connected_server_and_client_session(server._mcp_server) as c:
        return await c.call_tool(tool, args)


async def test_mcp_lists_three_tools_with_schemas(service):
    async with create_connected_server_and_client_session(build_server(service)._mcp_server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"invoicelens_ingest_document", "invoicelens_search_documents",
                          "invoicelens_extract_invoice"}
    assert all(t.outputSchema for t in tools.values())
    assert tools["invoicelens_search_documents"].annotations.readOnlyHint is True
    assert tools["invoicelens_ingest_document"].annotations.readOnlyHint is False


async def test_mcp_ingest_search_extract_roundtrip(service):
    server = build_server(service)
    r = await call(server, "invoicelens_ingest_document", {"pdf_base64": b64(NORTHWIND), "filename": NORTHWIND})
    assert not r.isError and r.structuredContent["chunks_by_type"]["table"] == 1
    doc_id = r.structuredContent["doc_id"]

    r = await call(server, "invoicelens_search_documents", {"query": "late payment penalty", "k": 1})
    hit = r.structuredContent["results"][0]
    assert (hit["doc_id"], hit["page"], hit["section"]) == (doc_id, 2, "Terms and Conditions")

    r = await call(server, "invoicelens_extract_invoice", {"doc_id": doc_id})
    assert r.structuredContent["invoice"]["total"] == 7546.82
    assert r.structuredContent["validation"]["valid"] is True


async def test_mcp_extract_accepts_a_pdf_directly(service):
    r = await call(build_server(service), "invoicelens_extract_invoice", {"pdf_base64": b64(INITECH)})
    assert r.structuredContent["validation"]["valid"] is False


@pytest.mark.parametrize("tool, args, expect", [
    ("invoicelens_extract_invoice", {"doc_id": "doc_missing"}, "Use invoicelens_search_documents"),
    ("invoicelens_ingest_document", {}, "Provide exactly one of"),
    ("invoicelens_ingest_document", {"pdf_base64": "%%%"}, "not valid base64"),
    ("invoicelens_ingest_document", {"pdf_base64": base64.b64encode(b"hello").decode()}, "Not a PDF"),
    ("invoicelens_search_documents", {"query": "x", "collection": "Bad Name"}, "Invalid collection name"),
])
async def test_mcp_errors_are_actionable(service, tool, args, expect):
    r = await call(build_server(service), tool, args)
    assert r.isError and expect in r.content[0].text


async def test_server_paths_can_be_disabled(tmp_path):
    s = InvoiceLensService(Settings(data_dir=tmp_path, embedding_provider="local",
                                    extraction_provider="local", allow_local_paths=False, _env_file=None))
    r = await call(build_server(s), "invoicelens_ingest_document", {"path": str(SAMPLES / ACME)})
    assert r.isError and "Send the file as pdf_base64" in r.content[0].text


def test_rest_endpoints(service):
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ingest", files={"file": (ACME, sample(ACME), "application/pdf")})
        assert r.status_code == 200 and r.json()["chunks"] == 4
        doc_id = r.json()["doc_id"]
        r = client.post("/search", json={"query": "pressure gauge", "k": 1})
        assert r.json()["results"][0]["doc_id"] == doc_id
        r = client.post("/extract", data={"doc_id": doc_id})
        assert r.json()["invoice"]["invoice_number"] == "ACME-2026-0311"
        assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]
        assert client.post("/ingest", files={"file": ("x.pdf", b"nope", "application/pdf")}).status_code == 422
        assert client.post("/extract", data={"doc_id": "doc_missing"}).status_code == 404


def test_rest_and_mcp_return_identical_results(service):
    """Parity: both interfaces are adapters over one service, so outputs must match."""
    with TestClient(create_app(service), base_url="http://127.0.0.1:8765") as client:
        doc_id = client.post("/ingest", files={"file": (INITECH, sample(INITECH), "application/pdf")}).json()["doc_id"]
        rest = client.post("/extract", data={"doc_id": doc_id}).json()
        resp = client.post("/mcp", headers=MCP_HEADERS, json=mcp_call(
            "invoicelens_extract_invoice", {"doc_id": doc_id}))
        assert resp.status_code == 200
        mcp = resp.json()["result"]["structuredContent"]
    assert mcp == rest


MCP_HEADERS = {"accept": "application/json, text/event-stream"}


def mcp_call(name, arguments):
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}


@pytest.mark.parametrize("bind_host, request_host, expected", [
    ("127.0.0.1", "127.0.0.1:8765", 200),     # local client
    ("127.0.0.1", "evil.example.com", 421),   # DNS-rebinding attempt is blocked
    ("0.0.0.0", "invoicelens.internal", 200), # deployed: any Host, so use a bearer token
])
def test_mcp_host_protection_follows_bind_host(service, bind_host, request_host, expected):
    with TestClient(create_app(service, host=bind_host), base_url=f"http://{request_host}") as client:
        resp = client.post("/mcp", headers=MCP_HEADERS,
                           json=mcp_call("invoicelens_search_documents", {"query": "invoice"}))
    assert resp.status_code == expected


def test_bearer_token_guards_rest_and_mcp(tmp_path):
    s = InvoiceLensService(Settings(data_dir=tmp_path, embedding_provider="local", extraction_provider="local",
                                    api_token="s3cret", _env_file=None))
    with TestClient(create_app(s)) as client:
        assert client.get("/health").status_code == 200           # open for liveness probes
        assert client.post("/search", json={"query": "x"}).status_code == 401
        assert client.post("/mcp", json={}).status_code == 401
        ok = client.post("/search", json={"query": "x"}, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
