"""Media type resolution and parser-selection hints.

All functions in this module derive media types purely from the URI string
(specifically its file extension) without opening or reading any file.
This keeps discovery, loading, and parsing cleanly separated.

Importing this module performs no side effects.
"""

from __future__ import annotations

import mimetypes
import posixpath
import urllib.parse

# Supplement the platform MIME database with common documentation types that
# may be absent on minimal installs.
_EXTRA_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".adoc": "text/asciidoc",
    ".asciidoc": "text/asciidoc",
    ".ipynb": "application/x-ipynb+json",
    ".jsonl": "application/jsonlines",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "application/toml",
}

# Parser-selection hints: map from media type prefix to a short hint string
# that tells the parser layer which parser family to use.
_PARSER_HINTS: dict[str, str] = {
    "text/markdown": "markdown",
    "text/x-rst": "rst",
    "text/asciidoc": "asciidoc",
    "text/html": "html",
    "application/pdf": "pdf",
    "application/x-ipynb+json": "jupyter",
    "application/json": "json",
    "application/jsonlines": "jsonlines",
    "text/plain": "plaintext",
    "text/yaml": "yaml",
    "application/toml": "toml",
}

_FALLBACK_MEDIA_TYPE: str = "application/octet-stream"
_FALLBACK_HINT: str = "binary"

# Fallback parser hints by major MIME type (used when no exact key matches).
# text/* types that are not in _PARSER_HINTS get a neutral "plaintext" hint
# rather than inheriting whatever text/* key appears first in _PARSER_HINTS.
_MAJOR_TYPE_FALLBACK_HINTS: dict[str, str] = {
    "text": "plaintext",
}


def _extension_from_uri(uri: str) -> str:
    """Extract the lowercase file extension from a URI, including the leading dot.

    Returns an empty string if the URI path has no extension.
    """
    parsed = urllib.parse.urlparse(uri)
    path = parsed.path or uri
    _, ext = posixpath.splitext(path)
    return ext.lower()


def resolve_media_type(uri: str) -> str:
    """Return the MIME media type for the given URI.

    Resolution is based solely on the file extension extracted from the URI
    path.  No file I/O is performed; the file does not need to exist.

    Priority order:
    1. ``_EXTRA_TYPES`` table (documentation-specific overrides).
    2. ``mimetypes`` standard library database.
    3. ``"application/octet-stream"`` as the safe fallback.

    Args:
        uri: A URI string such as ``file:///docs/guide.md``.
            The path component's extension determines the media type.

    Returns:
        A non-empty MIME type string, e.g. ``"text/markdown"``.
    """
    ext = _extension_from_uri(uri)
    if ext in _EXTRA_TYPES:
        return _EXTRA_TYPES[ext]
    guessed, _ = mimetypes.guess_type(uri)
    return guessed if guessed else _FALLBACK_MEDIA_TYPE


def resolve_media_type_with_hint(uri: str) -> tuple[str, str]:
    """Return ``(media_type, parser_hint)`` for the given URI.

    The parser hint is a short lowercase string that the parser layer uses to
    select the right parser implementation without knowing the media type string
    format. Both values are derived from the URI extension without file I/O.

    Args:
        uri: A URI string such as ``file:///docs/guide.md``.

    Returns:
        A ``(media_type, hint)`` tuple where ``hint`` is one of the known
        parser family names (e.g. ``"markdown"``, ``"html"``, ``"pdf"``) or
        ``"binary"`` for unknown types.
    """
    media_type = resolve_media_type(uri)
    # Exact match first.
    hint = _PARSER_HINTS.get(media_type)
    if hint is None:
        # Major-type prefix fallback: use an explicit mapping so that unknown
        # text/* types receive a neutral "plaintext" hint rather than inheriting
        # the hint of whichever text/* entry happens to appear first in
        # _PARSER_HINTS (which would be misleading, e.g. text/csv -> "markdown").
        major = media_type.split("/")[0]
        hint = _MAJOR_TYPE_FALLBACK_HINTS.get(major)
    return media_type, hint if hint is not None else _FALLBACK_HINT
