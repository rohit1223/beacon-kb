"""Sample in-memory connector for testing plugin discovery and Connector contract."""

from __future__ import annotations

from typing import ClassVar

from beacon_kb.errors import IngestionError
from beacon_kb.models import RawDocument, RevisionId, SourceId


class SampleMemoryConnector:
    """Minimal in-memory Connector implementation used as a sample plugin.

    Demonstrates how a third-party package implements the Connector protocol
    by importing only from beacon_kb.models and beacon_kb.errors.
    """

    _DEFAULT_SOURCES: ClassVar[dict[str, str]] = {
        "memory://sample-doc-1": "This is the first sample document for testing.",
        "memory://sample-doc-2": "This is the second sample document for testing.",
        "memory://sample-doc-3": "A third sample document with different content.",
    }

    def __init__(self, sources: dict[str, str] | None = None) -> None:
        self._sources: dict[str, str] = (
            sources if sources is not None else dict(self._DEFAULT_SOURCES)
        )

    def list_sources(self) -> list[str]:
        """Return sorted list of in-memory document URIs."""
        return sorted(self._sources.keys())

    def fetch(self, uri: str) -> RawDocument:
        """Return the RawDocument for the given URI.

        Raises:
            IngestionError: If the URI is not in the in-memory store.
        """
        if uri not in self._sources:
            raise IngestionError(f"SampleMemoryConnector: unknown URI {uri!r}")
        return RawDocument(
            source_id=SourceId(uri),
            revision_id=RevisionId(f"rev-{uri}"),
            content=self._sources[uri],
            media_type="text/plain",
        )
