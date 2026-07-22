"""Filesystem source connector.

Discovers files matching caller-supplied glob patterns and loads their text
content as :class:`~beacon_kb.models.RawDocument` records.

Design constraints (from the implementation plan):
- Discovery, loading, and parsing are strictly separated: this connector
  loads raw text/bytes and never parses or indexes content.
- No hardcoded extension list: the caller supplies ``patterns``.
- Glob discovery produces deterministic (sorted) output.
- Canonical source URIs are CWD-independent ``file://`` URIs built via
  :mod:`beacon_kb.ingestion.identity`.
- External citation links are derived from a configurable base URL mapping
  that never leaks local absolute paths.
- Errors identify the source and operation; they never expose credentials.
- No credential ownership: clients are injected by the caller, not created here.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.parse
import urllib.request

from beacon_kb.errors import IngestionError
from beacon_kb.ingestion.identity import make_file_source_uri
from beacon_kb.ingestion.media import resolve_media_type
from beacon_kb.models import (
    RawDocument,
    SourceId,
    make_revision_id,
    make_source_id,
)


class FilesystemConnector:
    """Discover and load files from a configured root directory.

    Args:
        root: Root directory to search for files.
        corpus: Logical corpus name used in canonical source IDs.
        patterns: Glob patterns relative to *root* (e.g. ``["**/*.md"]``).
            No default pattern is applied; callers must supply at least one.
        encoding: Text encoding to use when reading files (default ``"utf-8"``).
        pipeline_fingerprint: Optional stable string identifying the active
            pipeline configuration.  Used to build
            :class:`~beacon_kb.models.RevisionId` values.
        external_base_url: Optional base URL for citation link mapping.
            When provided, :meth:`external_url` returns a public URL derived
            from the file's path relative to *root*.  When absent, returns
            ``None``.

    Connector errors identify the source URI and the failing operation.
    They never include credentials or environment secrets.
    """

    def __init__(
        self,
        *,
        root: pathlib.Path | str,
        corpus: str,
        patterns: list[str],
        encoding: str = "utf-8",
        pipeline_fingerprint: str = "v1",
        external_base_url: str | None = None,
    ) -> None:
        self._root: pathlib.Path = pathlib.Path(root).resolve()
        self._corpus: str = corpus
        self._patterns: list[str] = patterns
        self._encoding: str = encoding
        self._pipeline_fingerprint: str = pipeline_fingerprint
        self._external_base_url: str | None = external_base_url

    # ------------------------------------------------------------------
    # Connector protocol
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        """Return a sorted list of ``file://`` URIs for all matched files.

        Deterministic: files are sorted lexicographically by their canonical
        URI so that repeated calls return identical lists regardless of
        filesystem enumeration order.

        Returns:
            Sorted list of canonical ``file://`` URI strings.

        Raises:
            :class:`~beacon_kb.errors.IngestionError` if the root directory
            is not accessible.
        """
        if not self._root.is_dir():
            raise IngestionError(
                f"FilesystemConnector.list_sources: root directory not accessible: "
                f"{self._root!s}"
            )
        uris: list[str] = []
        for pattern in self._patterns:
            for path in self._root.glob(pattern):
                if path.is_file():
                    uris.append(make_file_source_uri(path))
        # Deduplicate in case patterns overlap, then sort for determinism.
        return sorted(set(uris))

    def fetch(self, uri: str) -> RawDocument:
        """Load and return the raw text content for *uri*.

        Content is read as bytes and then strictly decoded with the configured
        encoding.  No parsing, indexing, or transformation is performed.

        Only text media types are accepted (``text/*`` plus known JSON/TOML
        variants).  Binary types (e.g. ``image/*``, ``application/pdf``) raise
        :class:`~beacon_kb.errors.IngestionError` immediately rather than
        silently mangling binary bytes into replacement characters.

        Args:
            uri: A canonical ``file://`` URI returned by :meth:`list_sources`.

        Returns:
            :class:`~beacon_kb.models.RawDocument` with the file's text content.

        Raises:
            :class:`~beacon_kb.errors.IngestionError` if the file cannot be
            read, has a binary media type, or cannot be decoded with the
            configured encoding.  The error message identifies the source URI
            and the failing operation.
        """
        path = self._uri_to_path(uri)
        if not path.exists():
            raise IngestionError(
                f"FilesystemConnector.fetch: file not found for URI {uri!r}"
            )
        if not path.is_file():
            raise IngestionError(
                f"FilesystemConnector.fetch: path is not a file for URI {uri!r}"
            )
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise IngestionError(
                f"FilesystemConnector.fetch: cannot read file for URI {uri!r}: {exc}"
            ) from exc

        media_type = resolve_media_type(uri)
        # Only attempt text decoding for recognised text/* media types.
        # Binary types (application/octet-stream, application/pdf, image/*, etc.)
        # are not decodable as text and must be rejected with a clear error rather
        # than silently mangled by an errors="replace" fallback.
        _text_type = media_type.startswith("text/") or media_type in (
            "application/x-ipynb+json",
            "application/json",
            "application/jsonlines",
            "application/toml",
        )
        if not _text_type:
            raise IngestionError(
                f"FilesystemConnector.fetch: binary or non-text media type "
                f"{media_type!r} for URI {uri!r}; text decoding is not supported "
                f"for this file type."
            )
        try:
            content = raw_bytes.decode(self._encoding, errors="strict")
        except (UnicodeDecodeError, LookupError) as exc:
            raise IngestionError(
                f"FilesystemConnector.fetch: cannot decode file as {self._encoding!r} "
                f"for URI {uri!r}: {exc}"
            ) from exc
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        # The Connector protocol contract requires str(source_id) == uri so
        # that callers can round-trip from URI to source_id without a corpus
        # lookup.  Use the canonical URI directly as the SourceId value.
        # Higher-level pipeline layers derive a corpus-scoped SourceId via
        # make_source_id(corpus=..., canonical_uri=uri) when they need one.
        source_id = SourceId(uri)
        # RevisionId is content-addressed: same content + pipeline -> same ID.
        logical_source_id = make_source_id(corpus=self._corpus, canonical_uri=uri)
        revision_id = make_revision_id(
            source_id=str(logical_source_id),
            content_hash=content_hash,
            pipeline_fingerprint=self._pipeline_fingerprint,
        )
        return RawDocument(
            source_id=source_id,
            revision_id=revision_id,
            content=content,
            media_type=media_type,
            encoding=self._encoding,
        )

    # ------------------------------------------------------------------
    # External citation link mapping
    # ------------------------------------------------------------------

    def external_url(self, uri: str) -> str | None:
        """Return an external citation URL for *uri*, or ``None``.

        When :attr:`external_base_url` is configured, the file's path
        relative to :attr:`root` is appended to the base URL.  This ensures
        that absolute local paths are never embedded in the returned URL.

        Args:
            uri: A canonical ``file://`` URI returned by :meth:`list_sources`.

        Returns:
            An HTTPS URL string, or ``None`` when no mapping is configured.
        """
        if self._external_base_url is None:
            return None
        path = self._uri_to_path(uri)
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            # Path outside root - fall back to filename only
            rel = pathlib.Path(path.name)
        # Use POSIX separators for URL construction; quote special chars.
        rel_posix = rel.as_posix()
        encoded = urllib.parse.quote(rel_posix, safe="/")
        base = self._external_base_url.rstrip("/")
        return f"{base}/{encoded}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uri_to_path(uri: str) -> pathlib.Path:
        """Convert a ``file://`` URI back to a :class:`pathlib.Path`.

        Args:
            uri: A ``file://`` URI string.

        Returns:
            Absolute :class:`pathlib.Path`.
        """
        parsed = urllib.parse.urlparse(uri)
        # url2pathname handles percent-decoding and platform path conversion.
        path_str = urllib.request.url2pathname(parsed.path)
        return pathlib.Path(path_str)
