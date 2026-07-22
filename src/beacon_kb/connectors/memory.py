"""In-memory source connector.

Supplies deterministic documents from a caller-supplied mapping for tests
and embedding applications.  No filesystem access, no credentials.

Design constraints (from the implementation plan):
- Caller owns the document map; the connector never reads from disk.
- Source IDs are content-addressed from the corpus name and URI.
- Revision IDs are content-addressed from the source ID and a SHA-256 of
  the document text.
- list_sources() is deterministic (sorted).
- Errors identify source and operation; never credentials.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

from beacon_kb.errors import IngestionError
from beacon_kb.ingestion.identity import PROVISIONAL_FINGERPRINT
from beacon_kb.ingestion.media import resolve_media_type
from beacon_kb.models import (
    RawDocument,
    SourceId,
    make_revision_id,
    make_source_id,
)

_DEFAULT_PIPELINE_FINGERPRINT: str = PROVISIONAL_FINGERPRINT


class MemoryConnector:
    """Supply deterministic in-memory documents for tests and embedding applications.

    Args:
        corpus: Logical corpus name used to build canonical source IDs.
        sources: Mapping from canonical URI string to document text.
            URIs can use any scheme (e.g. ``memory://``, ``fake://``).
        pipeline_fingerprint: Stable string identifying the active pipeline
            configuration.  Used in :class:`~beacon_kb.models.RevisionId`
            construction.
            Defaults to :data:`~beacon_kb.ingestion.identity.PROVISIONAL_FINGERPRINT`
            (``"unpinned"``).  The revision_id produced is a *provisional*
            content identity - it captures the content hash but not the full
            pipeline fingerprint.  The sync pipeline re-derives the
            authoritative revision_id with the real pipeline fingerprint.

    The connector is deterministic: identical inputs always produce identical
    IDs and document records across processes.
    """

    _BUILTIN_SOURCES: ClassVar[dict[str, str]] = {
        "memory://doc-1": "Content of in-memory document one.",
        "memory://doc-2": "Content of in-memory document two.",
    }

    def __init__(
        self,
        corpus: str = "default",
        sources: dict[str, str] | None = None,
        pipeline_fingerprint: str = _DEFAULT_PIPELINE_FINGERPRINT,
    ) -> None:
        self._corpus: str = corpus
        self._sources: dict[str, str] = (
            sources if sources is not None else dict(self._BUILTIN_SOURCES)
        )
        self._pipeline_fingerprint: str = pipeline_fingerprint

    # ------------------------------------------------------------------
    # Connector protocol
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        """Return a sorted list of canonical URIs for all in-memory documents.

        Returns:
            Sorted list of URI strings.
        """
        return sorted(self._sources.keys())

    def fetch(self, uri: str) -> RawDocument:
        """Return the :class:`~beacon_kb.models.RawDocument` for *uri*.

        Args:
            uri: A canonical URI returned by :meth:`list_sources`.

        Returns:
            :class:`~beacon_kb.models.RawDocument` with the in-memory content.

        Raises:
            :class:`~beacon_kb.errors.IngestionError` if *uri* is not present
            in the in-memory source map.
        """
        if uri not in self._sources:
            raise IngestionError(
                f"MemoryConnector.fetch: unknown URI {uri!r}. "
                f"Available: {sorted(self._sources.keys())}"
            )
        text = self._sources[uri]
        raw_bytes = text.encode("utf-8")
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        # The Connector protocol contract requires str(source_id) == uri.
        # Use the URI directly as the SourceId value here; callers that need
        # a corpus-scoped SourceId call make_source_id() separately.
        source_id = SourceId(uri)
        logical_source_id = make_source_id(corpus=self._corpus, canonical_uri=uri)
        revision_id = make_revision_id(
            source_id=str(logical_source_id),
            content_hash=content_hash,
            pipeline_fingerprint=self._pipeline_fingerprint,
        )
        media_type = resolve_media_type(uri)
        if media_type == "application/octet-stream":
            # Default to plain text for scheme-only URIs (e.g. memory://)
            media_type = "text/plain"
        return RawDocument(
            source_id=source_id,
            revision_id=revision_id,
            content=text,
            media_type=media_type,
            encoding="utf-8",
        )
