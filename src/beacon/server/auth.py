"""Optional bearer API-key middleware for the Beacon server (Task 01.4).

Rules:
- When no API key is configured (``settings.server.api_key`` is ``None``),
  auth is completely off: all requests pass through.
- When a key is configured:
  - ``GET /healthz`` is always exempt (liveness probe must never require auth).
  - Requests from localhost (127.x.x.x, ::1, or ``[::1]``) are always exempt
    per the local-first rule documented in the global constraints.
  - All other requests must present the correct key as
    ``Authorization: Bearer <key>``; missing or wrong keys receive a 401
    problem-details response.

Security note: the localhost exemption is decided from the actual TCP client
address (``request.client.host``), not from any spoofable header such as
``X-Forwarded-For`` or ``X-Real-IP``.
Operators fronting Beacon with a reverse proxy MUST configure the API key and
disable localhost-only access so the proxy's source IP is never treated as
localhost.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Paths that are always exempt from auth.
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset({"/healthz"})

# Content-type for problem-detail responses.
_PROBLEM_CONTENT_TYPE = "application/problem+json"


def _is_localhost(addr: str) -> bool:
    """Return True if *addr* is a loopback address.

    Accepts IPv4 (127.x.x.x), IPv6 (::1), and the bracketed form ([::1])
    that some ASGI servers produce.

    Args:
        addr: Client address string from ``request.client.host``.

    Returns:
        True when *addr* resolves to a loopback address.
    """
    # Strip IPv6 bracket notation if present.
    clean = addr.strip("[]")
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


def _unauthorized_response(detail: str = "Valid Bearer token required") -> JSONResponse:
    """Build a 401 RFC 9457 problem-details response.

    Args:
        detail: Human-readable explanation of the failure.

    Returns:
        JSONResponse with status 401 and Content-Type application/problem+json.
    """
    body = {
        "type": "https://beacon.example/errors/unauthorized",
        "title": "Unauthorized",
        "status": 401,
        "detail": detail,
    }
    return JSONResponse(
        content=body,
        status_code=401,
        headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
    )


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing the optional bearer API-key policy.

    Instantiated by ``create_app`` only when ``settings.server.api_key`` is
    set.  Tests can substitute a fake ``_is_localhost`` by monkey-patching the
    module-level function of the same name.

    Args:
        app: The ASGI application.
        api_key: The configured API key (plain string, already extracted from
            SecretStr by the caller).
    """

    def __init__(self, app: ASGIApp, *, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Check auth and delegate to the next layer.

        Args:
            request: Incoming HTTP request.
            call_next: ASGI next-layer callable.

        Returns:
            The downstream response, or a 401 problem response.
        """
        # /healthz is always exempt.
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        # Localhost is always exempt.
        # When the transport provides no client info, fail closed (0.0.0.0 is non-loopback).
        client_host = request.client.host if request.client else "0.0.0.0"  # noqa: S104
        if _is_localhost(client_host):
            return await call_next(request)

        # Validate Bearer token.
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _unauthorized_response("Authorization header with Bearer token required")

        provided_key = auth_header[len("Bearer "):]
        if provided_key != self._api_key:
            return _unauthorized_response("Invalid API key")

        return await call_next(request)
