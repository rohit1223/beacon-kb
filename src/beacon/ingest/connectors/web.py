"""Web and sitemap connector for the Beacon ingestion pipeline (Task 02.2).

Discovers and fetches web pages starting from seed URLs or a sitemap.xml,
restricted to the configured origin, bounded by depth and page count, and
respecting robots.txt.

Crawl algorithm:
    BFS from seed URLs.  Depth is measured from the seed (seed depth = 0).
    At each BFS level the crawler fetches the page, parses ``<a href>`` links,
    and enqueues same-origin, allowed, unseen links at depth+1 - provided
    depth+1 <= max_depth.  The page-count ceiling (max_pages) truncates the
    queue deterministically: no new URL is enqueued once the ceiling is reached.

Sitemap mode:
    When ``sitemap_url`` is given the XML is fetched and parsed.  If the root
    element is ``<sitemapindex>`` each ``<sitemap><loc>`` is fetched and parsed
    recursively (one level of nesting).  The resulting ``<url><loc>`` set
    becomes the seed list, which is then processed exactly like ``start_urls``.
    Both ``sitemap_url`` and ``start_urls`` may be given simultaneously; seeds
    are merged and deduplicated.

URL normalization (``canonicalize_url``):
    - Scheme and host are lowercased.
    - Default ports (80 for http, 443 for https) are removed.
    - Fragment identifier is stripped entirely.
    - Query parameters whose names appear in ``TRACKING_PARAMS`` are removed.
    - Remaining query parameters are sorted for deterministic identity.
    - Empty query string (no params remaining) is removed (no trailing ``?``).

Robots.txt policy:
    - robots.txt is fetched once per origin (scheme + host + port) and cached
      in ``self._robots_cache``.
    - On fetch failure (non-200 response, network error, timeout) the policy
      is fail-open: all paths are treated as allowed.  A WARNING is logged.
    - The configured ``user_agent`` is used for ``urllib.robotparser`` lookup.
      If not specified, ``BeaconCrawler`` is the default.
    - A URL that is disallowed by robots.txt is silently skipped during both
      enumeration (BFS and sitemap expansion) and fetch (TransientFailure
      would be wrong; skipping is the only safe behaviour since fetch is
      called with arbitrary URIs).

Transient vs deleted:
    - HTTP 404 or 410 -> ``ConfirmedDeletion``.
    - HTTP 5xx, ``httpx.TimeoutException``, ``httpx.TransportError`` ->
      ``TransientFailure``.
    - HTTP 2xx -> ``FetchSuccess`` with SHA-256 content hash.

Media-type detection (for ``FetchSuccess`` and ``SourceEntry``):
    - The ``Content-Type`` header value is stripped of parameters
      (e.g. ``text/html; charset=utf-8`` -> ``text/html``).
    - If the header is absent or empty, falls back to
      ``application/octet-stream``.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
import urllib.robotparser
from collections import deque

import httpx
from bs4 import BeautifulSoup

from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    ConnectorKind,
    FetchResult,
    FetchSuccess,
    SourceEntry,
    TransientFailure,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

# Query-parameter names that carry tracking/session state and must be removed
# to produce a stable canonical URL identity.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        # Google Analytics / UTM
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        # Google Ads
        "gclid",
        "gbraid",
        "wbraid",
        # Facebook
        "fbclid",
        # Generic referral
        "ref",
        "referrer",
        "source",
        # Microsoft / Bing
        "msclkid",
        # Mailchimp
        "mc_cid",
        "mc_eid",
    }
)

# Default ports per scheme; present in the URL but carry no info.
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def canonicalize_url(url: str) -> str:
    """Return the canonical form of *url*.

    Normalization steps (in order):
    1. Lowercase scheme and host.
    2. Remove default port for the scheme.
    3. Strip fragment.
    4. Remove tracking query parameters (names in ``TRACKING_PARAMS``).
    5. Sort remaining query parameters.
    6. Remove empty query string.

    The function is pure and idempotent:
    ``canonicalize_url(canonicalize_url(u)) == canonicalize_url(u)``.

    Args:
        url: An absolute URL string.

    Returns:
        Canonical URL string.
    """
    parsed = urllib.parse.urlparse(url)

    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
    port = parsed.port
    path = parsed.path
    # Strip fragment - always.
    # Only keep query params that are not tracking params.
    raw_query = parsed.query
    if raw_query:
        params = urllib.parse.parse_qsl(raw_query, keep_blank_values=True)
        filtered = [(k, v) for k, v in params if k.lower() not in TRACKING_PARAMS]
        # Sort for determinism.
        filtered.sort()
        query = urllib.parse.urlencode(filtered)
    else:
        query = ""

    # Remove default port.
    default_port = _DEFAULT_PORTS.get(scheme)
    if port is not None and port == default_port:
        port = None

    # Reconstruct netloc.
    if port is not None:
        netloc = f"{host}:{port}"
    else:
        netloc = host

    return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_media_type(content_type: str | None) -> str:
    """Strip parameters from a Content-Type value and return the type/subtype.

    Args:
        content_type: Raw ``Content-Type`` header value or ``None``.

    Returns:
        Media type string (e.g. ``"text/html"``), or
        ``"application/octet-stream"`` as fallback.
    """
    if not content_type:
        return "application/octet-stream"
    # Content-Type can be "text/html; charset=utf-8" - take the first segment.
    return content_type.split(";")[0].strip() or "application/octet-stream"


def _origin(url: str) -> str:
    """Return the scheme+host+port (origin) for *url*.

    Args:
        url: Absolute URL.

    Returns:
        Origin string, e.g. ``"https://example.com"`` or
        ``"https://example.com:8443"``.
    """
    p = urllib.parse.urlparse(url)
    if p.port and p.port != _DEFAULT_PORTS.get(p.scheme or ""):
        return f"{p.scheme}://{p.hostname}:{p.port}"
    return f"{p.scheme}://{p.hostname}"


def _is_same_origin(url: str, origin: str) -> bool:
    """Return True if *url* belongs to *origin*.

    Args:
        url: Absolute URL to check.
        origin: Origin string from ``_origin()``.

    Returns:
        True when the URL's scheme+host+port matches *origin*.
    """
    return _origin(url) == origin


def _resolve_href(base: str, href: str) -> str | None:
    """Resolve *href* against *base*, returning an absolute URL or None.

    Handles relative paths and absolute URLs.  Returns ``None`` for
    ``javascript:``, ``mailto:``, ``data:`` and other non-http(s) schemes.

    Args:
        base: Absolute URL of the page containing the link.
        href: Raw href attribute value.

    Returns:
        Absolute http(s) URL string, or ``None``.
    """
    href = href.strip()
    if not href:
        return None
    resolved = urllib.parse.urljoin(base, href)
    scheme = urllib.parse.urlparse(resolved).scheme.lower()
    if scheme not in ("http", "https"):
        return None
    return resolved


def _extract_links(html: bytes, base_url: str) -> list[str]:
    """Parse *html* and return absolute URLs from ``<a href>`` attributes.

    Args:
        html: Raw HTML bytes.
        base_url: Absolute URL of the page (used for resolving relative links).

    Returns:
        List of absolute URL strings (may contain duplicates).
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "")
        if not isinstance(href, str):
            continue
        resolved = _resolve_href(base_url, href)
        if resolved:
            links.append(resolved)
    return links


def _parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse a sitemap or sitemap index XML document.

    Returns:
        Tuple of (page_urls, child_sitemap_urls).
        - ``page_urls``: URLs from ``<url><loc>`` elements (urlset).
        - ``child_sitemap_urls``: URLs from ``<sitemap><loc>`` elements (sitemapindex).
    """
    try:
        soup = BeautifulSoup(xml_text, "xml")
    except Exception:
        return [], []

    page_urls: list[str] = []
    child_urls: list[str] = []

    # Check for sitemapindex first.
    if soup.find("sitemapindex"):
        for sitemap_tag in soup.find_all("sitemap"):
            loc = sitemap_tag.find("loc")
            if loc and loc.string:
                child_urls.append(loc.string.strip())
    else:
        # Standard urlset.
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            if loc and loc.string:
                page_urls.append(loc.string.strip())

    return page_urls, child_urls


# ---------------------------------------------------------------------------
# RobotsCache
# ---------------------------------------------------------------------------


class _RobotsCache:
    """Per-origin robots.txt fetcher and cache.

    robots.txt is fetched at most once per origin.  On failure the policy is
    fail-open (all paths allowed) and a WARNING is emitted.

    Args:
        client: httpx.Client used for fetching.
        user_agent: User-agent string for ``robotparser`` lookup.
    """

    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        # Maps origin -> RobotFileParser (or None for allow-all).
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def is_allowed(self, url: str) -> bool:
        """Return True if *url* is allowed by the origin's robots.txt.

        Args:
            url: Absolute URL to check.

        Returns:
            True when the URL is allowed or when robots.txt is unavailable
            (fail-open).
        """
        origin = _origin(url)
        if origin not in self._cache:
            self._cache[origin] = self._fetch(origin)
        parser = self._cache[origin]
        if parser is None:
            return True
        parsed = urllib.parse.urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return parser.can_fetch(self._user_agent, path)

    def _fetch(
        self, origin: str
    ) -> urllib.robotparser.RobotFileParser | None:
        """Fetch and parse robots.txt for *origin*.

        Returns:
            Parsed ``RobotFileParser`` on success, or ``None`` on failure
            (triggers allow-all / fail-open policy).
        """
        robots_url = f"{origin}/robots.txt"
        try:
            # follow_redirects=True is safe for robots.txt; origin-check is
            # not required because this is a standard well-known path.
            response = self._client.get(robots_url, follow_redirects=True)
        except httpx.TransportError as exc:
            log.warning(
                "robots.txt fetch failed for %s (%s); treating as allow-all",
                origin,
                exc,
            )
            return None
        except httpx.TimeoutException as exc:
            log.warning(
                "robots.txt fetch timed out for %s (%s); treating as allow-all",
                origin,
                exc,
            )
            return None

        if response.status_code != 200:
            log.warning(
                "robots.txt returned HTTP %d for %s; treating as allow-all",
                response.status_code,
                origin,
            )
            return None

        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser


# ---------------------------------------------------------------------------
# WebConnector
# ---------------------------------------------------------------------------


class WebConnector(Connector):
    """Crawl web pages starting from seed URLs or a sitemap.xml.

    The connector is pure: it takes config at construction time and performs
    no I/O until ``enumerate()`` is called.  All network I/O uses the
    injected ``transport`` (required; no default) so tests can use an offline
    mock transport.

    Args:
        start_urls:   Seed URLs to begin crawling from.  At least one of
                      ``start_urls`` or ``sitemap_url`` must be provided.
        sitemap_url:  URL of a sitemap.xml or sitemap index.  Its ``<loc>``
                      entries are merged into the seed set.
        max_depth:    Maximum BFS depth from the seed (seed = depth 0).
                      Links at depth ``max_depth`` are fetched but their
                      outbound links are not followed.  Default: 3.
        max_pages:    Maximum total pages to enumerate.  Truncates the BFS
                      queue deterministically once reached.  Default: 500.
        user_agent:   User-agent string sent in HTTP requests and used for
                      robots.txt lookup.  Default: ``"BeaconCrawler"``.
        transport:    httpx transport for all HTTP requests.  Required.
                      Pass an ``httpx.MockTransport`` or custom
                      ``httpx.BaseTransport`` subclass in tests.
        request_delay_seconds:
                      Per-origin delay between requests in seconds.  Default:
                      0.0 (no delay; callers wanting politeness should set
                      this).  Not enforced in offline/test mode.
    """

    DEFAULT_USER_AGENT = "BeaconCrawler"
    DEFAULT_MAX_DEPTH = 3
    DEFAULT_MAX_PAGES = 500

    # Maximum number of redirects to follow manually before giving up.
    _MAX_REDIRECTS = 5

    def __init__(
        self,
        *,
        start_urls: list[str] | None = None,
        sitemap_url: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_pages: int = DEFAULT_MAX_PAGES,
        user_agent: str = DEFAULT_USER_AGENT,
        transport: httpx.BaseTransport,
        request_delay_seconds: float = 0.0,
    ) -> None:
        if not start_urls and not sitemap_url:
            raise ValueError(
                "WebConnector requires at least one of start_urls or sitemap_url."
            )
        self._start_urls: list[str] = list(start_urls or [])
        self._sitemap_url = sitemap_url
        self._max_depth = max_depth
        self._max_pages = max_pages
        self._user_agent = user_agent
        self._request_delay = request_delay_seconds

        # Single shared client; redirects are handled manually so we can gate
        # cross-origin hops before issuing a request.  robots.txt fetches pass
        # follow_redirects=True explicitly (origin-check is not required there).
        self._client = httpx.Client(
            transport=transport,
            headers={"User-Agent": user_agent},
            follow_redirects=False,
        )

        # Robots cache is populated lazily during enumerate / fetch.
        self._robots_cache: _RobotsCache | None = None

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client and release connection pools."""
        self._client.close()

    def __enter__(self) -> WebConnector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connector interface
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Redirect helper
    # ------------------------------------------------------------------

    def _get_same_origin(
        self, url: str, origin: str
    ) -> httpx.Response | None:
        """Issue a GET to *url* with a manual same-origin redirect loop.

        Redirects are followed up to ``_MAX_REDIRECTS`` hops only when each
        Location target is the same origin as *origin*.  The first hop to an
        off-origin host is aborted - the redirect response to the external
        Location is never issued, so the external host is never contacted.

        Args:
            url:    Absolute URL to request.
            origin: Expected origin; all redirect targets must match.

        Returns:
            Final ``httpx.Response`` when the chain resolved within the same
            origin, or ``None`` if an off-origin redirect was detected or the
            hop limit was exceeded.

        Raises:
            httpx.TimeoutException: propagated from the underlying client.
            httpx.TransportError:   propagated from the underlying client.
        """
        current_url = url
        for _ in range(self._MAX_REDIRECTS + 1):
            response = self._client.get(current_url)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                # Resolve Location against the current URL (may be relative).
                next_url = urllib.parse.urljoin(current_url, location)
                if not _is_same_origin(next_url, origin):
                    log.info(
                        "enumerate: cross-origin redirect from %s to %s; skipping",
                        current_url,
                        next_url,
                    )
                    return None
                current_url = next_url
                continue
            return response
        log.info(
            "enumerate: too many redirects for %s; skipping",
            url,
        )
        return None

    # ------------------------------------------------------------------
    # Connector interface
    # ------------------------------------------------------------------

    def enumerate(self) -> list[SourceEntry]:
        """Discover all pages within limits and return their metadata.

        Returns:
            List of ``SourceEntry`` records, one per discovered page.
            Order is BFS traversal order (not sorted).

        Raises:
            Nothing: per-URL fetch failures during enumeration are silently
            skipped and logged so the caller gets a best-effort result.
        """
        if self._robots_cache is None:
            self._robots_cache = _RobotsCache(self._client, self._user_agent)

        # Build seed list.
        seeds: list[str] = []
        if self._sitemap_url:
            seeds.extend(self._expand_sitemap(self._sitemap_url))
        seeds.extend(self._start_urls)

        # Canonicalize and deduplicate seeds, preserving order.
        seen: set[str] = set()
        canonical_seeds: list[str] = []
        for url in seeds:
            c = canonicalize_url(url)
            if c not in seen:
                seen.add(c)
                canonical_seeds.append(c)

        entries: list[SourceEntry] = []
        # BFS queue: (canonical_url, depth)
        queue: deque[tuple[str, int]] = deque()
        for seed in canonical_seeds:
            queue.append((seed, 0))

        # Track all visited URLs to avoid re-queuing.
        visited: set[str] = {c for c, _ in queue}

        while queue and len(entries) < self._max_pages:
            url, depth = queue.popleft()
            origin = _origin(url)

            # Check robots.txt.
            if not self._robots_cache.is_allowed(url):
                log.debug("robots.txt disallows %s; skipping", url)
                continue

            # Fetch the page, following only same-origin redirects.
            try:
                response = self._get_same_origin(url, origin)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                log.warning("enumerate: transient failure fetching %s: %s", url, exc)
                continue

            # _get_same_origin returns None on cross-origin or too-many-hops.
            if response is None:
                continue

            # Derive the canonical URL from the final (post-redirect) URL so
            # same-host redirects store the correct identity in SourceEntry.uri.
            final_url = canonicalize_url(str(response.url))

            # Fix 2: check 5xx BEFORE 4xx so the branches are semantically
            # correct - 5xx is a transient server error, 4xx is a client error.
            if response.status_code >= 500:
                log.warning(
                    "enumerate: HTTP %d (transient) for %s; skipping",
                    response.status_code,
                    url,
                )
                continue
            if response.status_code in (404, 410):
                log.debug("enumerate: %d for %s; skipping", response.status_code, url)
                continue
            if response.status_code >= 400:
                log.warning(
                    "enumerate: HTTP %d for %s; skipping",
                    response.status_code,
                    url,
                )
                continue

            media_type = _extract_media_type(response.headers.get("content-type"))
            # Derive title from the final URL path.
            parsed = urllib.parse.urlparse(final_url)
            path = parsed.path.rstrip("/") or "/"
            title = path.split("/")[-1] or parsed.netloc

            entries.append(
                SourceEntry(
                    uri=final_url,
                    title=title,
                    connector_kind=ConnectorKind.WEB,
                    media_type=media_type,
                    metadata={"origin": _origin(final_url)},
                )
            )

            # Only follow links if depth allows.
            if depth < self._max_depth and len(entries) < self._max_pages:
                content_type = response.headers.get("content-type", "")
                if "html" in content_type:
                    final_origin = _origin(final_url)
                    links = _extract_links(response.content, final_url)
                    for link in links:
                        c = canonicalize_url(link)
                        if c in visited:
                            continue
                        if not _is_same_origin(c, final_origin):
                            continue
                        if not self._robots_cache.is_allowed(c):
                            continue
                        visited.add(c)
                        queue.append((c, depth + 1))

        return entries

    def fetch(self, uri: str) -> FetchResult:
        """Fetch raw content for *uri*.

        Args:
            uri: Canonical URL previously returned by ``enumerate()``.

        Returns:
            ``FetchSuccess`` with bytes and SHA-256 hash on 2xx.
            ``ConfirmedDeletion`` on HTTP 404 or 410.
            ``TransientFailure`` on HTTP 5xx, timeouts, connection errors, or
            when the server redirects to a different origin (an off-origin
            redirect is not a deletion; it is a transient/skip - the content
            may still exist at the original URI later).
        """
        # Ensure robots cache exists (fetch may be called without enumerate).
        if self._robots_cache is None:
            self._robots_cache = _RobotsCache(self._client, self._user_agent)

        origin = _origin(uri)
        try:
            response = self._get_same_origin(uri, origin)
        except httpx.TimeoutException as exc:
            return TransientFailure(
                uri=uri,
                reason=f"WebConnector.fetch: timeout fetching {uri!r}: {exc}",
            )
        except httpx.TransportError as exc:
            return TransientFailure(
                uri=uri,
                reason=f"WebConnector.fetch: connection error fetching {uri!r}: {exc}",
            )

        # None means cross-origin redirect or hop-limit exceeded.
        if response is None:
            return TransientFailure(
                uri=uri,
                reason=(
                    f"WebConnector.fetch: cross-origin or too-many-hops redirect"
                    f" for {uri!r}"
                ),
            )

        if response.status_code in (404, 410):
            return ConfirmedDeletion(uri=uri)
        if response.status_code >= 500:
            return TransientFailure(
                uri=uri,
                reason=(
                    f"WebConnector.fetch: HTTP {response.status_code} for {uri!r}"
                ),
            )
        if response.status_code >= 400:
            return TransientFailure(
                uri=uri,
                reason=(
                    f"WebConnector.fetch: HTTP {response.status_code} for {uri!r}"
                ),
            )
        if response.status_code >= 300:
            # With manual redirect loop, 3xx here means a non-redirect 3xx
            # (e.g. 304 Not Modified) - treat as transient.
            return TransientFailure(
                uri=uri,
                reason=(
                    f"WebConnector.fetch: unexpected 3xx"
                    f" {response.status_code} for {uri!r}"
                ),
            )

        content = response.content
        content_hash = hashlib.sha256(content).hexdigest()
        media_type = _extract_media_type(response.headers.get("content-type"))
        return FetchSuccess(
            content=content,
            content_hash=content_hash,
            media_type=media_type,
        )

    # ------------------------------------------------------------------
    # Sitemap helpers
    # ------------------------------------------------------------------

    def _expand_sitemap(self, sitemap_url: str) -> list[str]:
        """Fetch and expand a sitemap URL into a flat list of page URLs.

        Nesting is capped at one level per the brief: a root sitemapindex
        expands its direct <sitemap><loc> children, but any sitemapindex
        elements found inside those children are intentionally ignored.
        This is by design - deeper nesting is not part of the spec and would
        require unbounded recursion.

        A ``fetched_sitemaps`` set prevents cycles regardless of what the
        sitemap XML contains.

        Args:
            sitemap_url: URL of the sitemap to fetch.

        Returns:
            List of page URL strings.
        """
        # Track every sitemap URL we have already fetched to make cycles
        # structurally impossible.
        fetched_sitemaps: set[str] = set()

        try:
            response = self._client.get(sitemap_url, follow_redirects=True)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            log.warning("Could not fetch sitemap %s: %s", sitemap_url, exc)
            return []

        fetched_sitemaps.add(sitemap_url)

        if response.status_code != 200:
            log.warning(
                "Sitemap fetch returned HTTP %d for %s",
                response.status_code,
                sitemap_url,
            )
            return []

        page_urls, child_sitemap_urls = _parse_sitemap(response.text)

        # Expand child sitemaps (one level of nesting only - see docstring).
        for child_url in child_sitemap_urls:
            if child_url in fetched_sitemaps:
                log.debug("Sitemap cycle detected; skipping %s", child_url)
                continue
            fetched_sitemaps.add(child_url)
            child_pages, _ = _parse_sitemap(self._fetch_xml(child_url))
            page_urls.extend(child_pages)

        return page_urls

    def _fetch_xml(self, url: str) -> str:
        """Fetch XML content from *url*, returning empty string on failure.

        Args:
            url: URL to fetch.

        Returns:
            Response text or empty string.
        """
        try:
            response = self._client.get(url, follow_redirects=True)
            if response.status_code == 200:
                return response.text
            log.warning(
                "Child sitemap fetch returned HTTP %d for %s",
                response.status_code,
                url,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            log.warning("Child sitemap fetch failed for %s: %s", url, exc)
        return ""
