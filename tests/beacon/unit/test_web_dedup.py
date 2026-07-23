"""Web connector regression tests: redirect dedup, sitemap origin, robots.

Covers three branch-review findings:
- Redirect pairs ('/docs' + '/docs/') collapsing to the same final URL must
  yield a single SourceEntry, otherwise staged-count validation fails on
  every sync retry.
- Sitemap-derived seeds must be constrained to the sitemap's own origin.
- robots.txt must be re-checked on the post-redirect target before the final
  URL is used.
"""
from __future__ import annotations

import httpx

from beacon.ingest.connectors.web import WebConnector

_PAGE = b"<html><body><h1>Docs</h1><p>Some content here.</p></body></html>"


class _RedirectTransport(httpx.BaseTransport):
    """/docs 301-redirects to /docs/; both are same-origin."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path == "/docs":
            return httpx.Response(301, headers={"location": "/docs/"})
        if path == "/docs/":
            return httpx.Response(
                200, content=_PAGE, headers={"content-type": "text/html"}
            )
        return httpx.Response(404, text="Not found")


def test_redirect_pair_yields_single_source_entry() -> None:
    """Two seeds that redirect to the same final URL emit one SourceEntry."""
    with WebConnector(
        start_urls=["http://example.test/docs", "http://example.test/docs/"],
        max_depth=0,
        max_pages=10,
        transport=_RedirectTransport(),
    ) as connector:
        entries = connector.enumerate()

    uris = [e.uri for e in entries]
    assert len(uris) == len(set(uris)), f"Duplicate URIs emitted: {uris!r}"
    assert len(entries) == 1, (
        f"Expected 1 entry after redirect dedup, got {len(entries)}: {uris!r}"
    )
    assert entries[0].uri == "http://example.test/docs/"


def test_link_plus_redirect_to_same_final_url_syncs_once() -> None:
    """A page linking both '/docs' and '/docs/' still yields one entry."""

    class _LinkTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            if path in ("/", ""):
                html = (
                    b"<html><body>"
                    b'<a href="/docs">docs</a> <a href="/docs/">docs slash</a>'
                    b"</body></html>"
                )
                return httpx.Response(
                    200, content=html, headers={"content-type": "text/html"}
                )
            if path == "/docs":
                return httpx.Response(301, headers={"location": "/docs/"})
            if path == "/docs/":
                return httpx.Response(
                    200, content=_PAGE, headers={"content-type": "text/html"}
                )
            return httpx.Response(404, text="Not found")

    with WebConnector(
        start_urls=["http://example.test/"],
        max_depth=1,
        max_pages=10,
        transport=_LinkTransport(),
    ) as connector:
        entries = connector.enumerate()

    uris = [e.uri for e in entries]
    assert len(uris) == len(set(uris)), f"Duplicate URIs emitted: {uris!r}"
    assert "http://example.test/docs/" in uris


def test_sitemap_seeds_constrained_to_sitemap_origin() -> None:
    """Off-origin URLs listed in a sitemap are dropped from the seed set."""

    class _SitemapTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            if host != "example.test":
                # Any request to a foreign host is a test failure signal; the
                # connector must never contact it.  Return 200 so the failure
                # mode is a visible extra entry, not a skipped URL.
                return httpx.Response(
                    200, content=_PAGE, headers={"content-type": "text/html"}
                )
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            if path == "/sitemap.xml":
                xml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    "<url><loc>http://example.test/page1</loc></url>"
                    "<url><loc>http://evil.example/poison</loc></url>"
                    "</urlset>"
                )
                return httpx.Response(
                    200, text=xml, headers={"content-type": "application/xml"}
                )
            if path == "/page1":
                return httpx.Response(
                    200, content=_PAGE, headers={"content-type": "text/html"}
                )
            return httpx.Response(404, text="Not found")

    with WebConnector(
        sitemap_url="http://example.test/sitemap.xml",
        max_depth=0,
        max_pages=10,
        transport=_SitemapTransport(),
    ) as connector:
        entries = connector.enumerate()

    uris = [e.uri for e in entries]
    assert uris == ["http://example.test/page1"], (
        f"Sitemap seeds must be constrained to the sitemap origin; got {uris!r}"
    )


def test_robots_rechecked_on_post_redirect_target() -> None:
    """A redirect landing on a robots-disallowed path is not emitted."""

    class _RobotsRedirectTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(
                    200, text="User-agent: *\nDisallow: /private/\n"
                )
            if path == "/open":
                return httpx.Response(
                    301, headers={"location": "/private/secret"}
                )
            if path == "/private/secret":
                return httpx.Response(
                    200, content=_PAGE, headers={"content-type": "text/html"}
                )
            return httpx.Response(404, text="Not found")

    with WebConnector(
        start_urls=["http://example.test/open"],
        max_depth=0,
        max_pages=10,
        transport=_RobotsRedirectTransport(),
    ) as connector:
        entries = connector.enumerate()

    uris = [e.uri for e in entries]
    assert "http://example.test/private/secret" not in uris, (
        f"robots-disallowed post-redirect target must not be emitted; got {uris!r}"
    )
    assert entries == []
