"""Integration tests for the web and sitemap connector (Task 02.2).

All tests are fully offline: every HTTP interaction is handled by a custom
transport that routes requests to in-memory fixture responses.
The transport records every request URL so tests can assert that the crawler
never visits disallowed paths or exceeds depth/page limits.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    ConnectorKind,
    FetchSuccess,
    TransientFailure,
)
from beacon.ingest.connectors.web import WebConnector

# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


class RecordingTransport(httpx.BaseTransport):
    """Sync transport that records every request URL and delegates to a handler."""

    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> None:
        self._handler = handler
        self.requested_urls: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requested_urls.append(str(request.url))
        return self._handler(request)


def _make_transport(
    routes: dict[str, httpx.Response],
) -> RecordingTransport:
    """Build a recording transport from a URL -> Response mapping."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in routes:
            return routes[url]
        return httpx.Response(404, text="Not found")

    return RecordingTransport(handler)


# ---------------------------------------------------------------------------
# Fixture HTML/robots/sitemap content
# ---------------------------------------------------------------------------

_ROBOTS_ALLOW_ALL = "User-agent: *\nDisallow:"
_ROBOTS_DISALLOW_ADMIN = "User-agent: *\nDisallow: /admin/"

_HOME_HTML = """<!DOCTYPE html>
<html><head><title>Home</title></head>
<body>
<a href="/about">About</a>
<a href="/admin/secret">Secret (disallowed)</a>
<a href="https://external.com/page">External (out-of-scope)</a>
</body></html>"""

_ABOUT_HTML = """<!DOCTYPE html>
<html><head><title>About</title></head>
<body><p>About page</p></body></html>"""

_DEEP_HTML = """<!DOCTYPE html>
<html><head><title>Deep</title></head>
<body><a href="/deeper">Deeper</a></body></html>"""

_DEEPER_HTML = """<!DOCTYPE html>
<html><head><title>Deeper</title></head>
<body><p>Too deep</p></body></html>"""

_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/about</loc></url>
  <url><loc>https://example.com/contact</loc></url>
</urlset>"""

_SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
</sitemapindex>"""

_SITEMAP_PAGES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page-a</loc></url>
  <url><loc>https://example.com/page-b</loc></url>
</urlset>"""

_PAGE_A_HTML = """<!DOCTYPE html><html><head><title>Page A</title></head><body/></html>"""
_PAGE_B_HTML = """<!DOCTYPE html><html><head><title>Page B</title></head><body/></html>"""
_CONTACT_HTML = """<!DOCTYPE html><html><head><title>Contact</title></head><body/></html>"""

_ADMIN_HTML = """<!DOCTYPE html><html><head><title>Admin</title></head><body/></html>"""


# ---------------------------------------------------------------------------
# TestWebConnectorEnumerate: crawl mode
# ---------------------------------------------------------------------------


class TestWebConnectorEnumerate:
    """WebConnector.enumerate() discovers pages by crawling start URLs."""

    def _routes_basic(self) -> dict[str, httpx.Response]:
        return {
            "https://example.com/robots.txt": httpx.Response(
                200, text=_ROBOTS_ALLOW_ALL, headers={"content-type": "text/plain"}
            ),
            "https://example.com/": httpx.Response(
                200, text=_HOME_HTML, headers={"content-type": "text/html; charset=utf-8"}
            ),
            "https://example.com/about": httpx.Response(
                200, text=_ABOUT_HTML, headers={"content-type": "text/html"}
            ),
        }

    def test_enumerates_start_url_and_links(self) -> None:
        """Crawl discovers home page and its same-origin links."""
        transport = _make_transport(self._routes_basic())
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/" in uris
        assert "https://example.com/about" in uris

    def test_connector_kind_is_web(self) -> None:
        transport = _make_transport(self._routes_basic())
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        assert all(e.connector_kind == ConnectorKind.WEB for e in entries)

    def test_external_links_excluded(self) -> None:
        """Links to a different origin are never followed."""
        routes = self._routes_basic()
        routes["https://external.com/robots.txt"] = httpx.Response(
            200, text=_ROBOTS_ALLOW_ALL
        )
        routes["https://external.com/page"] = httpx.Response(200, text=_ABOUT_HTML)
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=2,
            max_pages=20,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert not any("external.com" in u for u in uris)

    def test_depth_limit_zero_seed_only(self) -> None:
        """With max_depth=0 only the seed URL is visited."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(
                200, text=_ROBOTS_ALLOW_ALL
            ),
            "https://example.com/": httpx.Response(
                200, text=_DEEP_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/deeper": httpx.Response(
                200, text=_DEEPER_HTML, headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,  # seed only, no link following
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/" in uris
        assert "https://example.com/deeper" not in uris

    def test_depth_limit_one_follows_one_level(self) -> None:
        """With max_depth=1 the crawler follows links from seed but not from those pages."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200, text=_DEEP_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/deeper": httpx.Response(
                200,
                text='<html><a href="/deepest">x</a></html>',
                headers={"content-type": "text/html"},
            ),
            "https://example.com/deepest": httpx.Response(
                200, text="<html/>", headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/" in uris
        assert "https://example.com/deeper" in uris
        assert "https://example.com/deepest" not in uris

    def test_max_pages_truncates_deterministically(self) -> None:
        """When max_pages is reached no further pages are visited."""
        routes = self._routes_basic()
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=2,
            max_pages=1,  # only the seed
            transport=transport,
        )
        entries = conn.enumerate()
        assert len(entries) == 1

    def test_robots_disallow_excludes_path(self) -> None:
        """A disallowed path is never requested, not even during enumeration."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(
                200, text=_ROBOTS_DISALLOW_ADMIN
            ),
            "https://example.com/": httpx.Response(
                200, text=_HOME_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/about": httpx.Response(
                200, text=_ABOUT_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/admin/secret": httpx.Response(
                200, text=_ADMIN_HTML, headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=2,
            max_pages=20,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        # The admin URL must not be in the enumerated set.
        assert "https://example.com/admin/secret" not in uris
        # And it must never have been requested.
        assert not any("admin" in u for u in transport.requested_urls)

    def test_robots_fetch_failure_allows_crawl(self, caplog: pytest.LogCaptureFixture) -> None:
        """If robots.txt cannot be fetched (404), crawling proceeds (fail-open)."""
        import logging

        routes: dict[str, httpx.Response] = {
            "https://example.com/": httpx.Response(
                200, text=_ABOUT_HTML, headers={"content-type": "text/html"}
            ),
        }
        # robots.txt returns 404 (handled by default 404 handler)
        transport = _make_transport(routes)
        with caplog.at_level(logging.WARNING, logger="beacon.ingest.connectors.web"):
            conn = WebConnector(
                start_urls=["https://example.com/"],
                max_depth=0,
                max_pages=10,
                transport=transport,
            )
            entries = conn.enumerate()
        assert len(entries) == 1
        assert any("robots" in r.message.lower() for r in caplog.records)

    def test_duplicate_urls_deduped(self) -> None:
        """The same URL appearing in multiple links is only visited once."""
        html_with_dup = """<html><a href="/about">A</a><a href="/about">B</a></html>"""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200, text=html_with_dup, headers={"content-type": "text/html"}
            ),
            "https://example.com/about": httpx.Response(
                200, text=_ABOUT_HTML, headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        about_entries = [e for e in entries if "about" in e.uri]
        assert len(about_entries) == 1

    def test_media_type_from_content_type_header(self) -> None:
        """SourceEntry.media_type is derived from the Content-Type header."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200, text=_HOME_HTML, headers={"content-type": "text/html; charset=utf-8"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        assert entries[0].media_type == "text/html"

    def test_no_real_network_calls(self) -> None:
        """A transport that raises on any call confirms no real network is used."""

        def raise_always(request: httpx.Request) -> httpx.Response:
            raise AssertionError("Real network call detected!")

        transport = RecordingTransport(raise_always)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=1,
            transport=transport,
        )
        with pytest.raises(AssertionError, match="Real network call"):
            conn.enumerate()


# ---------------------------------------------------------------------------
# TestWebConnectorSitemap: sitemap mode
# ---------------------------------------------------------------------------


class TestWebConnectorSitemap:
    """WebConnector with sitemap_url enumerates pages from sitemap.xml."""

    def _sitemap_routes(self) -> dict[str, httpx.Response]:
        return {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/sitemap.xml": httpx.Response(
                200,
                text=_SITEMAP_XML,
                headers={"content-type": "application/xml"},
            ),
            "https://example.com/": httpx.Response(
                200, text=_HOME_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/about": httpx.Response(
                200, text=_ABOUT_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/contact": httpx.Response(
                200, text=_CONTACT_HTML, headers={"content-type": "text/html"}
            ),
        }

    def test_sitemap_enumerates_all_locs(self) -> None:
        """All <loc> entries in sitemap become seeds."""
        transport = _make_transport(self._sitemap_routes())
        conn = WebConnector(
            sitemap_url="https://example.com/sitemap.xml",
            max_depth=0,  # seed-only; no further crawl
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/" in uris
        assert "https://example.com/about" in uris
        assert "https://example.com/contact" in uris

    def test_sitemap_index_expanded(self) -> None:
        """A sitemap index is expanded recursively into its child sitemaps."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/sitemap-index.xml": httpx.Response(
                200,
                text=_SITEMAP_INDEX_XML,
                headers={"content-type": "application/xml"},
            ),
            "https://example.com/sitemap-pages.xml": httpx.Response(
                200,
                text=_SITEMAP_PAGES_XML,
                headers={"content-type": "application/xml"},
            ),
            "https://example.com/page-a": httpx.Response(
                200, text=_PAGE_A_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/page-b": httpx.Response(
                200, text=_PAGE_B_HTML, headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            sitemap_url="https://example.com/sitemap-index.xml",
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/page-a" in uris
        assert "https://example.com/page-b" in uris

    def test_sitemap_respects_max_pages(self) -> None:
        """Even in sitemap mode, max_pages is honored."""
        transport = _make_transport(self._sitemap_routes())
        conn = WebConnector(
            sitemap_url="https://example.com/sitemap.xml",
            max_depth=0,
            max_pages=2,
            transport=transport,
        )
        entries = conn.enumerate()
        assert len(entries) <= 2

    def test_sitemap_robots_disallow_respected(self) -> None:
        """Sitemap URLs disallowed by robots.txt are excluded."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(
                200, text=_ROBOTS_DISALLOW_ADMIN
            ),
            "https://example.com/sitemap.xml": httpx.Response(
                200,
                text="""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/</loc></url>
  <url><loc>https://example.com/admin/panel</loc></url>
</urlset>""",
                headers={"content-type": "application/xml"},
            ),
            "https://example.com/": httpx.Response(
                200, text=_HOME_HTML, headers={"content-type": "text/html"}
            ),
            "https://example.com/admin/panel": httpx.Response(
                200, text=_ADMIN_HTML, headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            sitemap_url="https://example.com/sitemap.xml",
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/admin/panel" not in uris
        assert not any("admin" in u for u in transport.requested_urls)

    def test_sitemap_and_start_urls_combined(self) -> None:
        """Providing both sitemap_url and start_urls merges seeds."""
        routes = self._sitemap_routes()
        routes["https://example.com/extra"] = httpx.Response(
            200, text="<html/>", headers={"content-type": "text/html"}
        )
        transport = _make_transport(routes)
        conn = WebConnector(
            sitemap_url="https://example.com/sitemap.xml",
            start_urls=["https://example.com/extra"],
            max_depth=0,
            max_pages=20,
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/extra" in uris
        assert "https://example.com/about" in uris


# ---------------------------------------------------------------------------
# TestWebConnectorFetch: fetch outcomes
# ---------------------------------------------------------------------------


class TestWebConnectorFetch:
    """WebConnector.fetch() returns the correct FetchResult discriminated union."""

    def _conn(self, routes: dict[str, httpx.Response]) -> WebConnector:
        transport = _make_transport(routes)
        return WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )

    def test_fetch_success_returns_bytes(self) -> None:
        body = b"<html>content</html>"
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/html"},
            ),
        })
        conn.enumerate()  # warm robots cache
        result = conn.fetch("https://example.com/")
        assert isinstance(result, FetchSuccess)
        assert result.content == body

    def test_fetch_success_content_hash_is_sha256(self) -> None:
        import hashlib

        body = b"<html>hash me</html>"
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200, content=body, headers={"content-type": "text/html"}
            ),
        })
        conn.enumerate()
        result = conn.fetch("https://example.com/")
        assert isinstance(result, FetchSuccess)
        assert result.content_hash == hashlib.sha256(body).hexdigest()

    def test_fetch_404_is_confirmed_deletion(self) -> None:
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(200, text=_HOME_HTML),
        })
        conn.enumerate()
        result = conn.fetch("https://example.com/gone")
        assert isinstance(result, ConfirmedDeletion)
        assert result.uri == "https://example.com/gone"

    def test_fetch_410_is_confirmed_deletion(self) -> None:
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(200, text=_HOME_HTML),
            "https://example.com/gone": httpx.Response(410, text="Gone"),
        })
        conn.enumerate()
        result = conn.fetch("https://example.com/gone")
        assert isinstance(result, ConfirmedDeletion)

    def test_fetch_5xx_is_transient_failure(self) -> None:
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(200, text=_HOME_HTML),
            "https://example.com/error": httpx.Response(503, text="Service unavailable"),
        })
        conn.enumerate()
        result = conn.fetch("https://example.com/error")
        assert isinstance(result, TransientFailure)
        assert result.uri == "https://example.com/error"

    def test_fetch_timeout_is_transient_failure(self) -> None:
        """A network timeout maps to TransientFailure."""

        def timeout_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/slow":
                raise httpx.TimeoutException("timeout", request=request)
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text=_ROBOTS_ALLOW_ALL)
            return httpx.Response(200, text=_HOME_HTML)

        transport = RecordingTransport(timeout_handler)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        conn.enumerate()
        result = conn.fetch("https://example.com/slow")
        assert isinstance(result, TransientFailure)
        assert "timeout" in result.reason.lower()

    def test_fetch_connection_error_is_transient(self) -> None:
        """A connection error maps to TransientFailure."""

        def conn_error_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text=_ROBOTS_ALLOW_ALL)
            if request.url.path == "/":
                return httpx.Response(200, text=_HOME_HTML)
            raise httpx.ConnectError("refused", request=request)

        transport = RecordingTransport(conn_error_handler)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        conn.enumerate()
        result = conn.fetch("https://example.com/unreachable")
        assert isinstance(result, TransientFailure)

    def test_fetch_media_type_from_header(self) -> None:
        """FetchSuccess.media_type is taken from the Content-Type header."""
        body = b"# Markdown"
        conn = self._conn({
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200, content=body, headers={"content-type": "text/markdown"}
            ),
        })
        conn.enumerate()
        result = conn.fetch("https://example.com/")
        assert isinstance(result, FetchSuccess)
        assert result.media_type == "text/markdown"

    def test_fetch_media_type_fallback_octet_stream(self) -> None:
        """When Content-Type is absent, fallback is application/octet-stream."""
        body = b"\x00\x01binary"
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200,
                content=body,
                headers={},  # no content-type
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        conn.enumerate()
        result = conn.fetch("https://example.com/")
        assert isinstance(result, FetchSuccess)
        assert result.media_type == "application/octet-stream"


# ---------------------------------------------------------------------------
# TestWebConnectorRobots: robots policy
# ---------------------------------------------------------------------------


class TestWebConnectorRobots:
    """Detailed robots.txt policy tests."""

    def test_user_agent_beacon_respected(self) -> None:
        """Beacon-specific Disallow blocks requests even when * allows."""
        robots = "User-agent: BeaconCrawler\nDisallow: /private/\n\nUser-agent: *\nAllow: /"
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=robots),
            "https://example.com/": httpx.Response(
                200,
                text='<html><a href="/private/page">x</a></html>',
                headers={"content-type": "text/html"},
            ),
            "https://example.com/private/page": httpx.Response(
                200, text="<html/>", headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            user_agent="BeaconCrawler",
            transport=transport,
        )
        entries = conn.enumerate()
        uris = {e.uri for e in entries}
        assert "https://example.com/private/page" not in uris
        assert not any("private" in u for u in transport.requested_urls)

    def test_robots_fetched_once_per_host(self) -> None:
        """robots.txt is fetched exactly once per origin even across many pages."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200,
                text='<html><a href="/a">a</a><a href="/b">b</a></html>',
                headers={"content-type": "text/html"},
            ),
            "https://example.com/a": httpx.Response(
                200, text="<html/>", headers={"content-type": "text/html"}
            ),
            "https://example.com/b": httpx.Response(
                200, text="<html/>", headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=10,
            transport=transport,
        )
        conn.enumerate()
        robots_requests = [u for u in transport.requested_urls if "robots.txt" in u]
        assert len(robots_requests) == 1


# ---------------------------------------------------------------------------
# TestCrossOriginRedirect: Fix 1 regression tests
# ---------------------------------------------------------------------------


class TestCrossOriginRedirect:
    """Cross-origin redirect handling: external host must never be contacted."""

    def test_cross_origin_redirect_skipped_external_never_requested(self) -> None:
        """A 301 to an external host is aborted; the external URL is never fetched."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "robots.txt" in url:
                return httpx.Response(200, text=_ROBOTS_ALLOW_ALL)
            if url == "https://example.com/redirected":
                # 301 to an external host.
                return httpx.Response(
                    301,
                    headers={"location": "https://evil.example.net/stolen"},
                )
            if url == "https://evil.example.net/stolen":
                # This must never be reached.
                raise AssertionError("External host was contacted!")
            return httpx.Response(404)

        transport = RecordingTransport(handler)
        conn = WebConnector(
            start_urls=["https://example.com/redirected"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()

        # The page is skipped (cross-origin redirect).
        assert entries == []
        # The external URL must never have been requested.
        assert not any("evil.example.net" in u for u in transport.requested_urls)

    def test_same_host_redirect_is_followed_and_uri_is_final_url(self) -> None:
        """A 301 within the same origin is followed; SourceEntry.uri is the final URL."""
        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/old": httpx.Response(
                301,
                headers={"location": "https://example.com/new"},
            ),
            "https://example.com/new": httpx.Response(
                200,
                text=_ABOUT_HTML,
                headers={"content-type": "text/html"},
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/old"],
            max_depth=0,
            max_pages=10,
            transport=transport,
        )
        entries = conn.enumerate()

        assert len(entries) == 1
        # The final canonical URL (after redirect) is stored in SourceEntry.uri.
        assert entries[0].uri == "https://example.com/new"

    def test_cross_origin_redirect_in_fetch_is_transient_failure(self) -> None:
        """fetch() of a URL that redirects cross-origin returns TransientFailure."""
        from beacon.ingest.connectors.base import TransientFailure

        def handler(request: httpx.Request) -> httpx.Response:
            if "robots.txt" in str(request.url):
                return httpx.Response(200, text=_ROBOTS_ALLOW_ALL)
            if str(request.url) == "https://example.com/moved":
                return httpx.Response(
                    302,
                    headers={"location": "https://other.example.org/page"},
                )
            raise AssertionError(f"Unexpected request: {request.url}")

        transport = RecordingTransport(handler)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=0,
            max_pages=1,
            transport=transport,
        )
        # Warm robots cache with a dummy enumerate that hits a 404 seed.
        # (or just call fetch directly; fetch initialises robots cache lazily)
        result = conn.fetch("https://example.com/moved")

        assert isinstance(result, TransientFailure)
        # The external URL must never have been contacted.
        assert not any("other.example.org" in u for u in transport.requested_urls)


# ---------------------------------------------------------------------------
# TestEnumerate5xxHandling: Fix 2 regression tests
# ---------------------------------------------------------------------------


class TestEnumerate5xxHandling:
    """5xx during enumeration logs a warning and skips without aborting the crawl."""

    def test_500_during_enumeration_skips_and_continues(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 500 response in enumerate is skipped; other pages are still visited."""
        import logging

        routes: dict[str, httpx.Response] = {
            "https://example.com/robots.txt": httpx.Response(200, text=_ROBOTS_ALLOW_ALL),
            "https://example.com/": httpx.Response(
                200,
                text='<html><a href="/broken">b</a><a href="/ok">ok</a></html>',
                headers={"content-type": "text/html"},
            ),
            "https://example.com/broken": httpx.Response(500, text="Server Error"),
            "https://example.com/ok": httpx.Response(
                200, text="<html/>", headers={"content-type": "text/html"}
            ),
        }
        transport = _make_transport(routes)
        conn = WebConnector(
            start_urls=["https://example.com/"],
            max_depth=1,
            max_pages=20,
            transport=transport,
        )
        with caplog.at_level(logging.WARNING, logger="beacon.ingest.connectors.web"):
            entries = conn.enumerate()

        uris = {e.uri for e in entries}
        # The 500 page is skipped.
        assert "https://example.com/broken" not in uris
        # The other pages are still crawled.
        assert "https://example.com/" in uris
        assert "https://example.com/ok" in uris
        # A warning was logged for the 500.
        assert any("500" in r.message for r in caplog.records)
