"""Table-driven unit tests for canonical URL normalization (Task 02.2).

Every row exercises one normalization rule: scheme lowercasing, fragment
stripping, tracking-parameter removal, trailing-slash consistency, and
fragment-only identity.  The function under test is
``beacon.ingest.connectors.web.canonicalize_url``.
"""

from __future__ import annotations

import pytest

from beacon.ingest.connectors.web import canonicalize_url

# ---------------------------------------------------------------------------
# Table-driven normalization cases
# ---------------------------------------------------------------------------

_CASES: list[tuple[str, str, str]] = [
    # (label, input, expected)
    # Scheme is lowercased.
    (
        "https scheme lowercased",
        "HTTPS://example.com/page",
        "https://example.com/page",
    ),
    # Host is lowercased.
    (
        "host lowercased",
        "https://EXAMPLE.COM/page",
        "https://example.com/page",
    ),
    # Fragment is stripped.
    (
        "fragment stripped",
        "https://example.com/page#section-1",
        "https://example.com/page",
    ),
    # Fragment stripped when no path.
    (
        "fragment stripped from root",
        "https://example.com/#top",
        "https://example.com/",
    ),
    # Common UTM tracking params are removed.
    (
        "utm_source removed",
        "https://example.com/page?utm_source=newsletter",
        "https://example.com/page",
    ),
    (
        "utm_medium removed",
        "https://example.com/page?utm_medium=email",
        "https://example.com/page",
    ),
    (
        "utm_campaign removed",
        "https://example.com/page?utm_campaign=spring",
        "https://example.com/page",
    ),
    (
        "utm_term removed",
        "https://example.com/page?utm_term=offer",
        "https://example.com/page",
    ),
    (
        "utm_content removed",
        "https://example.com/page?utm_content=button",
        "https://example.com/page",
    ),
    # Other known tracking params are removed.
    (
        "fbclid removed",
        "https://example.com/page?fbclid=ABC123",
        "https://example.com/page",
    ),
    (
        "gclid removed",
        "https://example.com/page?gclid=XYZ",
        "https://example.com/page",
    ),
    (
        "ref removed",
        "https://example.com/page?ref=homepage",
        "https://example.com/page",
    ),
    # Non-tracking query params are preserved.
    (
        "real query param preserved",
        "https://example.com/search?q=python",
        "https://example.com/search?q=python",
    ),
    # Mixed: tracking removed, real preserved, sorted.
    (
        "mixed params: tracking removed real preserved",
        "https://example.com/page?utm_source=g&q=python&utm_medium=cpc",
        "https://example.com/page?q=python",
    ),
    # Default port stripped (80 for http, 443 for https).
    (
        "default https port 443 stripped",
        "https://example.com:443/page",
        "https://example.com/page",
    ),
    (
        "default http port 80 stripped",
        "http://example.com:80/page",
        "http://example.com/page",
    ),
    # Non-default port preserved.
    (
        "non-default port preserved",
        "https://example.com:8443/page",
        "https://example.com:8443/page",
    ),
    # Trailing slash on root preserved.
    (
        "root path slash preserved",
        "https://example.com/",
        "https://example.com/",
    ),
    # No trailing slash added to path without one.
    (
        "path without trailing slash unchanged",
        "https://example.com/docs/guide",
        "https://example.com/docs/guide",
    ),
    # Empty query string collapsed (no trailing '?').
    (
        "empty query string removed",
        "https://example.com/page?",
        "https://example.com/page",
    ),
    # Both fragment and tracking param stripped together.
    (
        "fragment and tracking stripped together",
        "https://example.com/page?utm_source=x#section",
        "https://example.com/page",
    ),
    # Query params are sorted for determinism.
    (
        "query params sorted deterministically",
        "https://example.com/page?z=1&a=2",
        "https://example.com/page?a=2&z=1",
    ),
]


@pytest.mark.parametrize("label,url_in,expected", _CASES, ids=[c[0] for c in _CASES])
def test_canonicalize_url(label: str, url_in: str, expected: str) -> None:
    """Each normalization rule produces the canonical form."""
    result = canonicalize_url(url_in)
    assert result == expected, f"[{label}] got {result!r}, want {expected!r}"


# ---------------------------------------------------------------------------
# Additional property tests
# ---------------------------------------------------------------------------


def test_canonicalize_idempotent() -> None:
    """Applying canonicalize_url twice gives the same result as once."""
    url = "https://EXAMPLE.COM/path?utm_source=x&b=1&a=2#frag"
    once = canonicalize_url(url)
    twice = canonicalize_url(once)
    assert once == twice


def test_canonicalize_same_content_same_hash() -> None:
    """Two URLs that canonicalize to the same form compare equal."""
    a = canonicalize_url("HTTPS://example.com/page#s")
    b = canonicalize_url("https://example.com/page")
    assert a == b
