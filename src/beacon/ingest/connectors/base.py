"""Connector interface for the Beacon ingestion pipeline (Task 02.1).

The ``Connector`` abstract base class defines the two-method surface that every
source type must implement:

- ``enumerate()`` - discover all sources and return their metadata.
- ``fetch(uri)`` - retrieve raw bytes for a single source, returning a typed
  outcome that distinguishes success, transient failure, and confirmed deletion.

The transient-vs-deleted distinction is the load-bearing design point: Task 02.5
must never retire an indexed source on a transient I/O failure (e.g. a permission
error or temporary network hiccup). Only ``ConfirmedDeletion`` is evidence that
the source is gone.

LlamaIndex adapter seam:
    Later connectors (web, Confluence, GitHub) may wrap LlamaIndex readers.
    The adapter pattern is: instantiate the LlamaIndex reader, call its ``load_data``
    method inside ``enumerate``/``fetch``, and map the resulting ``Document`` objects
    to ``SourceEntry`` / ``FetchSuccess`` values. The ``Connector`` interface itself
    has NO dependency on LlamaIndex; the adapter lives in the connector implementation.

Importing this module performs no side effects.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Connector kind constants
# ---------------------------------------------------------------------------


class ConnectorKind:
    """String constants for each connector type.

    Used in ``SourceEntry.connector_kind`` and stored in the state DB
    ``sources.connector_kind`` column so the sync planner can dispatch by kind.
    """

    FOLDER: Literal["folder"] = "folder"
    UPLOAD: Literal["upload"] = "upload"
    WEB: Literal["web"] = "web"

    # Valid kind values for validation in routes.
    ALL: frozenset[str] = frozenset({"folder", "upload", "web"})


# ---------------------------------------------------------------------------
# Source entry (returned by enumerate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceEntry:
    """Metadata for a single discovered source (no content yet).

    ``uri`` is the stable canonical URI that uniquely identifies this source
    across connector runs. For folders it is a ``file://`` URI; for uploads an
    ``upload://<content_hash>`` URI; for web a normalized ``https://`` URI.

    ``title`` is a human-readable display name (filename stem, page title, etc.).

    ``connector_kind`` is one of the ``ConnectorKind`` string constants.

    ``media_type`` is the detected media type (e.g. ``'text/markdown'``).
    The connector sets this from the declared content type or file extension;
    it is stored in source metadata and passed to the parser in Task 02.3.

    ``metadata`` is an arbitrary dict for connector-specific fields (root path
    for folder connectors, origin URL for web connectors, etc.).
    """

    uri: str
    title: str
    connector_kind: str
    media_type: str
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fetch result: discriminated union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchSuccess:
    """The source was fetched successfully.

    ``content`` is the raw bytes of the document (not decoded).
    ``content_hash`` is the hex-encoded SHA-256 of ``content``.
    ``media_type`` is the detected/confirmed media type for this fetch
    (may differ from the enumerate-time value if the server revised it).
    """

    content: bytes
    content_hash: str
    media_type: str


@dataclass(frozen=True)
class TransientFailure:
    """Fetching the source failed, but the failure is temporary.

    Examples: permission denied on a readable file (EACCES on a suddenly
    restricted path), a network timeout for a web connector, a 503 from a
    remote API. Task 02.5 must keep the existing indexed content for this source
    and record a warning; it must NOT retire the source.

    ``uri`` is the canonical URI that failed.
    ``reason`` is a human-readable explanation (never contains credentials).
    """

    uri: str
    reason: str


@dataclass(frozen=True)
class ConfirmedDeletion:
    """The source no longer exists at the connector.

    Examples: the file has been deleted from disk (``path.exists()`` is False
    after ``path`` was present in a recent ``enumerate()``), a 404 on a URL
    that previously returned 200.

    Task 02.5 MUST retire the indexed chunks for this source upon seeing this
    outcome.

    ``uri`` is the canonical URI that no longer exists.
    """

    uri: str


# Discriminated union type alias used as the return type of ``Connector.fetch``.
FetchResult = FetchSuccess | TransientFailure | ConfirmedDeletion


# ---------------------------------------------------------------------------
# Connector ABC
# ---------------------------------------------------------------------------


class Connector(abc.ABC):
    """Abstract base class for all Beacon source connectors.

    Subclasses implement two methods:

    ``enumerate()``
        Walk the source set and return a list of ``SourceEntry`` records with
        metadata (URI, title, kind, media type). No content is fetched here.
        The list need not be sorted; callers sort by URI for determinism.

    ``fetch(uri)``
        Retrieve the raw content for a single URI previously returned by
        ``enumerate()``. Returns a ``FetchResult`` discriminated union:
        - ``FetchSuccess`` on success.
        - ``TransientFailure`` if the fetch fails transiently (I/O error,
          network error) but the file/resource probably still exists.
        - ``ConfirmedDeletion`` if the source is confirmed gone (file deleted,
          404 response, etc.).

    Connectors must not own credentials; callers inject any auth material at
    construction time. Error messages must never leak credentials or secrets.
    """

    @abc.abstractmethod
    def enumerate(self) -> list[SourceEntry]:
        """Return metadata for all currently discoverable sources.

        Returns:
            List of ``SourceEntry`` records. May be empty. Order is unspecified.

        Raises:
            ``beacon.errors.IngestionError`` if the entire source set is
            inaccessible (e.g. root directory does not exist). Per-source
            failures during enumeration should be silently skipped and logged
            rather than raised, to match the transient-failure contract.
        """

    @abc.abstractmethod
    def fetch(self, uri: str) -> FetchResult:
        """Fetch the raw content for *uri*.

        Args:
            uri: A canonical URI previously returned by ``enumerate()``.

        Returns:
            ``FetchSuccess`` with raw bytes and content hash on success.
            ``TransientFailure`` if the fetch fails transiently.
            ``ConfirmedDeletion`` if the source no longer exists.

        Must not raise ``IngestionError`` for per-source failures; those must
        be returned as ``TransientFailure`` or ``ConfirmedDeletion`` so Task
        02.5 can apply the correct handling per source.
        """
