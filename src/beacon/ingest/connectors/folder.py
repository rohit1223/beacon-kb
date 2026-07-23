"""Folder connector for the Beacon ingestion pipeline (Task 02.1).

Recursively discovers files matching include glob patterns (minus any that match
exclude globs) under a configured root directory. Each file becomes a source
with a stable canonical ``file://`` URI derived from its resolved absolute path.

Identity pair:
    The connector never derives source identity from path alone. The identity
    pair is (canonical URI, content hash). A file moved with identical content
    will produce the same content hash; the canonical URI will differ, so Task
    02.5 treats it as a new source and retires the old one. An in-place edit
    changes the content hash while the URI stays the same - Task 02.5 sees a
    changed source and re-ingests it.

Transient vs deleted:
    - A file that ``path.exists()`` returns False for after being in a previous
      ``enumerate()`` result -> ``ConfirmedDeletion``.
    - A file that ``path.exists()`` returns True but ``path.read_bytes()``
      raises ``PermissionError`` or another ``OSError`` -> ``TransientFailure``.

Media-type detection:
    Uses ``mimetypes.guess_type`` on the file name (extension-based). Falls back
    to ``'application/octet-stream'`` for unknown extensions. The detected type
    is stored in ``SourceEntry.media_type`` and passed to the parser.

LlamaIndex adapter seam (documented for future connectors):
    Later connectors (web, Confluence) may use LlamaIndex readers inside
    ``enumerate`` / ``fetch``. The pattern: call ``reader.load_data()``, map
    each returned ``Document`` to a ``SourceEntry``, and return a ``FetchSuccess``
    from its ``text`` field encoded as UTF-8 bytes. This connector does NOT use
    LlamaIndex (plain filesystem walking is more efficient and has no dependency).

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
import mimetypes
import pathlib
import urllib.parse
import urllib.request

from beacon.errors import IngestionError
from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    ConnectorKind,
    FetchResult,
    FetchSuccess,
    SourceEntry,
    TransientFailure,
)

# Ensure common media types are registered (some minimal environments lack them).
mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("text/markdown", ".markdown")


def _make_file_uri(path: pathlib.Path) -> str:
    """Return a stable canonical ``file://`` URI for *path*.

    The path is resolved to an absolute POSIX path before encoding so that
    two callers referencing the same file from different working directories
    produce identical URIs.

    Args:
        path: Filesystem path (absolute or relative).

    Returns:
        A stable ``file://`` URI string with percent-encoded special chars.
        Example: ``file:///home/user/docs/guide.md``
    """
    resolved = path.resolve()
    posix = resolved.as_posix()
    quoted = urllib.parse.quote(posix, safe=":/")
    return f"file://{quoted}"


def _uri_to_path(uri: str) -> pathlib.Path:
    """Convert a ``file://`` URI back to a :class:`pathlib.Path`.

    Args:
        uri: A ``file://`` URI string.

    Returns:
        Absolute :class:`pathlib.Path`.
    """
    parsed = urllib.parse.urlparse(uri)
    path_str = urllib.request.url2pathname(parsed.path)
    return pathlib.Path(path_str)


def _detect_media_type(path: pathlib.Path) -> str:
    """Detect the media type for *path* from its extension.

    Prefers extension-based detection. Falls back to
    ``'application/octet-stream'`` for unknown extensions.

    Args:
        path: File path.

    Returns:
        MIME type string (e.g. ``'text/markdown'``, ``'application/pdf'``).
    """
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


class FolderConnector(Connector):
    """Discover and fetch files from a configured root directory.

    Files matched by any ``include_globs`` pattern that are NOT matched by any
    ``exclude_globs`` pattern are included. Globs are relative to *root*.

    Args:
        root:          Root directory to walk.
        include_globs: List of glob patterns to include (e.g. ``['**/*.md']``).
                       Must contain at least one pattern.
        exclude_globs: Optional list of glob patterns to exclude.
    """

    def __init__(
        self,
        *,
        root: pathlib.Path | str,
        include_globs: list[str],
        exclude_globs: list[str] | None = None,
    ) -> None:
        self._root = pathlib.Path(root).resolve()
        self._include_globs = include_globs
        self._exclude_globs = exclude_globs or []

    def enumerate(self) -> list[SourceEntry]:
        """Recursively discover files and return their metadata.

        Returns:
            List of ``SourceEntry`` records. Order is unspecified; callers
            should sort by URI for determinism.

        Raises:
            ``IngestionError``: If *root* does not exist or is not a directory.
        """
        if not self._root.exists():
            raise IngestionError(
                f"FolderConnector.enumerate: root directory does not exist: {self._root}"
            )
        if not self._root.is_dir():
            raise IngestionError(
                f"FolderConnector.enumerate: root is not a directory: {self._root}"
            )

        # Collect excluded URIs first (expand all exclude globs).
        excluded_uris: set[str] = set()
        for pattern in self._exclude_globs:
            for path in self._root.glob(pattern):
                if path.is_file():
                    excluded_uris.add(_make_file_uri(path))

        # Collect included files, deduplicate (patterns may overlap).
        seen_uris: set[str] = set()
        entries: list[SourceEntry] = []
        for pattern in self._include_globs:
            for path in self._root.glob(pattern):
                if not path.is_file():
                    continue
                uri = _make_file_uri(path)
                if uri in excluded_uris or uri in seen_uris:
                    continue
                seen_uris.add(uri)
                media_type = _detect_media_type(path)
                entries.append(
                    SourceEntry(
                        uri=uri,
                        title=path.name,
                        connector_kind=ConnectorKind.FOLDER,
                        media_type=media_type,
                        metadata={"root": str(self._root)},
                    )
                )
        return entries

    def fetch(self, uri: str) -> FetchResult:
        """Fetch raw bytes for *uri*.

        Returns:
            ``FetchSuccess`` with bytes and SHA-256 hash on success.
            ``ConfirmedDeletion`` if the file does not exist.
            ``TransientFailure`` if the file exists but cannot be read (e.g. EACCES).
        """
        path = _uri_to_path(uri)
        if not path.exists():
            return ConfirmedDeletion(uri=uri)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return TransientFailure(
                uri=uri,
                reason=f"FolderConnector.fetch: cannot read {uri!r}: {exc}",
            )
        content_hash = hashlib.sha256(raw).hexdigest()
        media_type = _detect_media_type(path)
        return FetchSuccess(
            content=raw,
            content_hash=content_hash,
            media_type=media_type,
        )
