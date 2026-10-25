"""
External data: web search (Tavily) and company market data (Financial Modeling Prep).

Why two sources: web search returns prose snippets — fine for "how is the company
growing / any recent news", unreliable for numbers (stale prices, mixed currencies).
Share price, market cap and growth rates come from FMP as structured, dated fields.

Both clients:
  - take an injectable httpx.Client, so tests run against mocked HTTP, offline
  - turn provider failures into ExternalDataError with a message that says what to
    do (missing key, bad key, rate limit, plan restriction, not found)
  - FMP responses are cached (default 10 min): the free plan allows 250 calls/day,
    and one company lookup costs 2-3 calls.

FMP's free plan covers US-listed securities only. Non-US companies resolve through
their US listing (e.g. an ADR: Infosys -> INFY) or need a paid plan.
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

from invoicelens.models import (CompanyFinancials, GrowthPeriod, WebSearchItem, WebSearchResult)

TAVILY_URL = "https://api.tavily.com/search"
FMP_BASE = "https://financialmodelingprep.com/stable"
US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "NYSEARCA", "NYSE AMERICAN", "CBOE", "BATS", "OTC"}
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z]{1,3})?$")


class ExternalDataError(RuntimeError):
    """Provider failure; the message tells the caller what to do next."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _http_error(provider: str, key_env: str, resp: httpx.Response) -> ExternalDataError:
    code = resp.status_code
    if code in (401, 403) and provider == "Tavily":
        return ExternalDataError(f"Tavily rejected the API key (HTTP {code}). Check {key_env} on the server.")
    if code == 401:
        return ExternalDataError(f"{provider} rejected the API key (HTTP 401). Check {key_env} on the server.")
    if code in (402, 403):
        return ExternalDataError(f"{provider}: this data isn't included in the current plan (HTTP {code}). "
                                 "The free plan covers US-listed companies and basic endpoints only.")
    if code == 429:
        return ExternalDataError(f"{provider} rate limit reached (HTTP 429). Wait and retry, or reduce calls.")
    return ExternalDataError(f"{provider} request failed (HTTP {code}): {resp.text[:200]}")


# ── Web search ──────────────────────────────────────────────────────────
class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str | None, client: httpx.Client | None = None, timeout: float = 15.0):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, topic: Literal["general", "news", "finance"] = "general",
               time_range: Literal["day", "week", "month", "year"] | None = None,
               max_results: int = 5, include_domains: list[str] | None = None,
               include_answer: bool = False) -> WebSearchResult:
        if not self.api_key:
            raise ExternalDataError("Web search isn't configured on this server: set INVOICELENS_TAVILY_API_KEY "
                                    "(free key at https://app.tavily.com).")
        body: dict[str, Any] = {"query": query, "topic": topic, "search_depth": "basic",
                                "max_results": max_results, "include_answer": include_answer}
        if time_range:
            body["time_range"] = time_range
        if include_domains:
            body["include_domains"] = include_domains
        try:
            resp = self._client.post(TAVILY_URL, json=body,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as e:
            raise ExternalDataError(f"Could not reach Tavily: {type(e).__name__}. Try again shortly.") from e
        if resp.status_code != 200:
            raise _http_error("Tavily", "INVOICELENS_TAVILY_API_KEY", resp)
        data = resp.json()
        return WebSearchResult(
            query=query, topic=topic, provider=self.name, retrieved_at=_now(),
            answer=data.get("answer") or None,
            results=[WebSearchItem(title=r.get("title") or "", url=r.get("url") or "",
                                   content=(r.get("content") or "")[:1200], score=r.get("score"),
                                   published_date=r.get("published_date"))
                     for r in data.get("results", [])])


# ── Market data ─────────────────────────────────────────────────────────
def _first(d: dict, *keys: str):
    """FMP has renamed fields between API versions; accept any known spelling."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


class FMPFinancials:
    name = "fmp"

    def __init__(self, api_key: str | None, client: httpx.Client | None = None,
                 timeout: float = 15.0, cache_seconds: int = 600):
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._ttl = cache_seconds
        self._lock = threading.Lock()

    def _get(self, path: str, **params) -> Any:
        key = (path, tuple(sorted(params.items())))
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < self._ttl:
                return hit[1]
        try:
            resp = self._client.get(f"{FMP_BASE}/{path}", params=params, headers={"apikey": self.api_key})
        except httpx.HTTPError as e:
            raise ExternalDataError(f"Could not reach Financial Modeling Prep: {type(e).__name__}.") from e
        if resp.status_code != 200:
            raise _http_error("Financial Modeling Prep", "INVOICELENS_FMP_API_KEY", resp)
        data = resp.json()
        if isinstance(data, dict) and data.get("Error Message"):
            msg = data["Error Message"]
            if "key" in msg.lower():
                raise ExternalDataError(f"Financial Modeling Prep: {msg} Check INVOICELENS_FMP_API_KEY.")
            raise ExternalDataError(f"Financial Modeling Prep: {msg}")
        with self._lock:
            self._cache[key] = (time.monotonic(), data)
        return data

    def resolve_symbol(self, company: str) -> tuple[str, str]:
        """Name or ticker -> (symbol, how it was resolved). Prefers US listings (free-plan coverage)."""
        query = company.strip()
        if _TICKER_RE.match(query):
            return query, "ticker"
        matches = self._get("search-name", query=query) or []
        if not matches:
            matches = self._get("search-symbol", query=query) or []
        us = [m for m in matches if str(_first(m, "exchange", "exchangeShortName") or "").upper() in US_EXCHANGES]
        pool = us or matches
        if not pool:
            raise ExternalDataError(
                f"No listed company found for '{company}'. It may be private, or listed only outside the US "
                "(the free FMP plan covers US exchanges; try its US/ADR ticker). "
                "For news or context, use invoicelens_web_search instead.")
        exact = [m for m in pool if str(m.get("name", "")).lower().startswith(query.lower())]
        return (exact or pool)[0]["symbol"], "name search"

    def company(self, company: str, include_growth: bool = True) -> CompanyFinancials:
        if not self.api_key:
            raise ExternalDataError("Market data isn't configured on this server: set INVOICELENS_FMP_API_KEY "
                                    "(free key at https://site.financialmodelingprep.com).")
        symbol, resolved_by = self.resolve_symbol(company)
        profiles = self._get("profile", symbol=symbol) or []
        if not profiles:
            raise ExternalDataError(
                f"No profile for ticker '{symbol}'. If it isn't a US listing, the free FMP plan won't cover it; "
                "try the company name or its US/ADR ticker.")
        p = profiles[0]

        growth: list[GrowthPeriod] = []
        notes: list[str] = []
        if include_growth:
            try:
                rows = self._get("income-statement-growth", symbol=symbol, period="annual", limit=3) or []
                growth = [GrowthPeriod(
                    fiscal_year=str(_first(r, "fiscalYear", "calendarYear") or str(r.get("date", ""))[:4]),
                    period_end=r.get("date"),
                    revenue_growth=_first(r, "growthRevenue", "revenueGrowth"),
                    net_income_growth=_first(r, "growthNetIncome", "netIncomeGrowth"),
                    eps_growth=_first(r, "growthEPS", "growthEps", "epsgrowth"),
                ) for r in rows[:3]]
            except ExternalDataError as e:
                notes.append(f"Growth data unavailable: {e}")   # profile still useful on its own

        description = p.get("description") or None
        if description and len(description) > 600:
            description = description[:600].rsplit(" ", 1)[0] + "…"
        return CompanyFinancials(
            query=company, symbol=p.get("symbol", symbol), resolved_by=resolved_by,
            name=p.get("companyName"), exchange=_first(p, "exchange", "exchangeShortName"),
            currency=p.get("currency"), price=p.get("price"),
            change_percent=_first(p, "changePercentage", "changesPercentage"),
            market_cap=_first(p, "marketCap", "mktCap"), beta=p.get("beta"),
            sector=p.get("sector"), industry=p.get("industry"), country=p.get("country"),
            ceo=p.get("ceo"), website=p.get("website"),
            employees=_int_or_none(p.get("fullTimeEmployees")), ipo_date=p.get("ipoDate"),
            is_actively_trading=p.get("isActivelyTrading"), description=description,
            growth=growth, notes=notes, provider="Financial Modeling Prep", retrieved_at=_now())


def _int_or_none(v) -> int | None:
    try:
        return int(str(v).replace(",", "")) if v not in (None, "") else None
    except ValueError:
        return None
