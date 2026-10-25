"""Tavily and FMP clients against mocked HTTP (httpx.MockTransport): offline, deterministic."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from invoicelens.api import create_app
from invoicelens.external import ExternalDataError, FMPFinancials, TavilySearch
from invoicelens.mcp_server import build_server
from invoicelens.service import InvoiceLensService

TAVILY_OK = {"query": "q", "answer": None, "response_time": 0.4, "results": [
    {"title": "Microsoft cloud revenue up", "url": "https://example.com/msft", "content": "Azure grew...",
     "score": 0.91, "published_date": "2026-10-20"},
    {"title": "Analysis", "url": "https://example.org/a", "content": "x" * 5000, "score": 0.7}]}

PROFILE = [{"symbol": "MSFT", "companyName": "Microsoft Corporation", "price": 512.3, "marketCap": 3.81e12,
            "currency": "USD", "exchange": "NASDAQ", "sector": "Technology", "industry": "Software",
            "country": "US", "ceo": "Satya Nadella", "website": "https://www.microsoft.com",
            "fullTimeEmployees": "228000", "ipoDate": "1986-03-13", "isActivelyTrading": True,
            "changePercentage": 1.2, "beta": 0.9, "description": "Microsoft develops software. " * 60}]
GROWTH = [{"date": "2026-06-30", "fiscalYear": "2026", "growthRevenue": 0.15, "growthNetIncome": 0.16, "growthEPS": 0.17},
          {"date": "2025-06-30", "fiscalYear": "2025", "growthRevenue": 0.14, "growthNetIncome": 0.12, "growthEPS": 0.13}]


def mock(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FMPServer:
    """Minimal fake of the FMP stable API; records requests."""

    def __init__(self, growth_status=200, search=None):
        self.requests: list[httpx.Request] = []
        self.growth_status = growth_status
        self.search = search if search is not None else [
            {"symbol": "MSF.DE", "name": "Microsoft Corporation", "exchange": "XETRA"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"}]

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        path = req.url.path.rsplit("/", 1)[-1]
        if path in ("search-name", "search-symbol"):
            return httpx.Response(200, json=self.search if path == "search-name" else [])
        if path == "profile":
            return httpx.Response(200, json=PROFILE if req.url.params["symbol"] == "MSFT" else [])
        if path == "income-statement-growth":
            return httpx.Response(self.growth_status, json=GROWTH if self.growth_status == 200 else {})
        return httpx.Response(404)


# ── Tavily ──────────────────────────────────────────────────────────────
def test_tavily_request_and_parsing():
    seen = {}

    def handler(req):
        seen["auth"], seen["body"] = req.headers["authorization"], json.loads(req.content)
        return httpx.Response(200, json=TAVILY_OK)

    r = TavilySearch("tvly-key", client=mock(handler)).search(
        "Microsoft growth", topic="finance", time_range="month", max_results=3, include_domains=["reuters.com"])
    assert seen["auth"] == "Bearer tvly-key"
    assert seen["body"] == {"query": "Microsoft growth", "topic": "finance", "search_depth": "basic",
                            "max_results": 3, "include_answer": False, "time_range": "month",
                            "include_domains": ["reuters.com"]}
    assert [x.url for x in r.results] == ["https://example.com/msft", "https://example.org/a"]
    assert r.results[0].published_date == "2026-10-20"
    assert len(r.results[1].content) == 1200                       # long snippets trimmed


@pytest.mark.parametrize("status, expect", [
    (401, "rejected the API key"), (429, "rate limit"), (500, "HTTP 500")])
def test_tavily_errors_are_actionable(status, expect):
    with pytest.raises(ExternalDataError, match=expect):
        TavilySearch("k", client=mock(lambda r: httpx.Response(status, text="err"))).search("q")


def test_tavily_without_key_explains_setup():
    with pytest.raises(ExternalDataError, match="INVOICELENS_TAVILY_API_KEY"):
        TavilySearch(None).search("q")


def test_tavily_network_failure():
    def boom(req):
        raise httpx.ConnectError("down")
    with pytest.raises(ExternalDataError, match="Could not reach Tavily"):
        TavilySearch("k", client=mock(boom)).search("q")


# ── FMP ─────────────────────────────────────────────────────────────────
def test_fmp_name_resolves_to_us_listing_and_parses():
    srv = FMPServer()
    r = FMPFinancials("fmp-key", client=mock(srv)).company("Microsoft")
    assert (r.symbol, r.resolved_by) == ("MSFT", "name search")   # NASDAQ preferred over XETRA
    assert (r.price, r.market_cap, r.currency, r.employees) == (512.3, 3.81e12, "USD", 228000)
    assert [g.fiscal_year for g in r.growth] == ["2026", "2025"] and r.growth[0].revenue_growth == 0.15
    assert len(r.description) <= 601 and r.description.endswith("…")
    assert "not investment advice" in r.disclaimer and r.retrieved_at
    assert all(req.headers["apikey"] == "fmp-key" for req in srv.requests)
    assert "apikey" not in str(srv.requests[0].url)                 # key not leaked into URLs/logs


def test_fmp_ticker_skips_search():
    srv = FMPServer()
    FMPFinancials("k", client=mock(srv)).company("MSFT")
    assert [r.url.path.rsplit("/", 1)[-1] for r in srv.requests] == ["profile", "income-statement-growth"]


def test_fmp_growth_plan_restriction_degrades_gracefully():
    r = FMPFinancials("k", client=mock(FMPServer(growth_status=402))).company("MSFT")
    assert r.market_cap == 3.81e12 and r.growth == []
    assert r.notes and "isn't included in the current plan" in r.notes[0]


def test_fmp_not_found_suggests_next_step():
    with pytest.raises(ExternalDataError, match="may be private.*invoicelens_web_search"):
        FMPFinancials("k", client=mock(FMPServer(search=[]))).company("Acme Industrial Supplies GmbH")


def test_fmp_cache_saves_free_plan_calls():
    srv = FMPServer()
    fmp = FMPFinancials("k", client=mock(srv), cache_seconds=600)
    fmp.company("Microsoft")
    first = len(srv.requests)
    fmp.company("Microsoft")
    assert len(srv.requests) == first == 3


def test_fmp_error_message_payload():
    bad = mock(lambda r: httpx.Response(200, json={"Error Message": "Invalid API KEY."}))
    with pytest.raises(ExternalDataError, match="Check INVOICELENS_FMP_API_KEY"):
        FMPFinancials("k", client=bad).company("MSFT")


# ── Exposed through MCP and REST ────────────────────────────────────────
@pytest.fixture
def online_service(settings):
    tav = TavilySearch("tvly", client=mock(lambda r: httpx.Response(200, json=TAVILY_OK)))
    return InvoiceLensService(settings, web=tav, market=FMPFinancials("k", client=mock(FMPServer())))


async def test_new_tools_over_mcp(online_service):
    async with create_connected_server_and_client_session(build_server(online_service)._mcp_server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert {"invoicelens_company_financials", "invoicelens_web_search"} <= set(tools)
        assert tools["invoicelens_web_search"].annotations.openWorldHint is True
        r = await c.call_tool("invoicelens_company_financials", {"company": "Microsoft"})
        assert r.structuredContent["symbol"] == "MSFT"
        r = await c.call_tool("invoicelens_web_search", {"query": "Microsoft growth", "topic": "news"})
        assert r.structuredContent["results"][0]["url"] == "https://example.com/msft"


async def test_missing_keys_are_tool_errors_not_crashes(service):
    async with create_connected_server_and_client_session(build_server(service)._mcp_server) as c:
        r = await c.call_tool("invoicelens_web_search", {"query": "x"})
        assert r.isError and "INVOICELENS_TAVILY_API_KEY" in r.content[0].text
        r = await c.call_tool("invoicelens_company_financials", {"company": "MSFT"})
        assert r.isError and "INVOICELENS_FMP_API_KEY" in r.content[0].text


def test_new_rest_endpoints(online_service, service):
    with TestClient(create_app(online_service)) as client:
        assert client.get("/company-financials", params={"company": "MSFT"}).json()["name"] == "Microsoft Corporation"
        assert client.post("/web-search", json={"query": "q"}).json()["provider"] == "tavily"
    with TestClient(create_app(service)) as client:                    # no keys configured
        r = client.get("/company-financials", params={"company": "MSFT"})
        assert r.status_code == 502 and "INVOICELENS_FMP_API_KEY" in r.json()["detail"]
