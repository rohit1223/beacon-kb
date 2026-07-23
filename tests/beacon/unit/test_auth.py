"""Unit tests for the optional bearer API-key middleware (Task 01.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from starlette.datastructures import Headers

from beacon.server.auth import ApiKeyMiddleware, _is_localhost


class TestIsLocalhost:
    """Tests for _is_localhost helper."""

    def test_ipv4_loopback_127_0_0_1(self) -> None:
        """IPv4 loopback 127.0.0.1 is localhost."""
        assert _is_localhost("127.0.0.1") is True

    def test_ipv4_loopback_127_255_255_255(self) -> None:
        """IPv4 loopback 127.x.x.x range is localhost."""
        assert _is_localhost("127.255.255.255") is True

    def test_ipv6_loopback_double_colon_1(self) -> None:
        """IPv6 loopback ::1 is localhost."""
        assert _is_localhost("::1") is True

    def test_ipv6_loopback_bracketed(self) -> None:
        """IPv6 loopback [::1] (bracketed form) is localhost."""
        assert _is_localhost("[::1]") is True

    def test_ipv4_non_loopback(self) -> None:
        """IPv4 non-loopback is not localhost."""
        assert _is_localhost("192.168.1.1") is False

    def test_ipv4_all_zeros(self) -> None:
        """IPv4 0.0.0.0 (all zeros) is not localhost."""
        assert _is_localhost("0.0.0.0") is False

    def test_invalid_address(self) -> None:
        """Invalid address is not localhost."""
        assert _is_localhost("invalid") is False

    def test_empty_string(self) -> None:
        """Empty string is not localhost."""
        assert _is_localhost("") is False


class TestApiKeyMiddlewareClientNone:
    """Tests for ApiKeyMiddleware behavior when request.client is None."""

    @pytest.mark.asyncio
    async def test_middleware_enforces_auth_when_client_none(self) -> None:
        """Middleware enforces auth (fails closed) when request.client is None.

        Simulates a transport that supplies no client info by setting
        request.client to None. The middleware should treat this as a
        non-loopback address and require authentication.
        """
        app = FastAPI()
        middleware = ApiKeyMiddleware(app, api_key="test-key")

        # Create a mock request with client=None
        request = MagicMock(spec=Request)
        request.url.path = "/readyz"
        request.client = None
        request.headers = Headers({"Authorization": "Bearer invalid-key"})

        # Mock the next callable
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        # Dispatch should not call call_next; it should return 401
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 401
        # Ensure call_next was not called (auth failed before delegation)
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_middleware_passes_with_correct_key_when_client_none(self) -> None:
        """Middleware passes with correct key even when request.client is None."""
        app = FastAPI()
        middleware = ApiKeyMiddleware(app, api_key="test-key")

        # Create a mock request with client=None
        request = MagicMock(spec=Request)
        request.url.path = "/readyz"
        request.client = None
        request.headers = Headers({"Authorization": "Bearer test-key"})

        # Mock the next callable
        mock_response = MagicMock(status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        # Dispatch should call call_next and return its response
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 200
        call_next.assert_called_once()
