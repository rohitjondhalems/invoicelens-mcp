"""Optional static bearer-token check for the HTTP transports (MCP and REST).

Enabled when INVOICELENS_API_TOKEN is set. Fine for a demo or an internal
service; for multi-user production use, put the server behind an OAuth-aware
gateway (e.g. Azure API Management / Entra ID) instead.
"""
from __future__ import annotations

import hmac
import json

OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class BearerAuthMiddleware:
    def __init__(self, app, token: str | None):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        got = headers.get(b"authorization", b"").decode()
        if not hmac.compare_digest(got, f"Bearer {self.token}"):
            body = json.dumps({"detail": "Missing or invalid bearer token"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        return await self.app(scope, receive, send)
