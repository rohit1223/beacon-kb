"""Deterministic test fakes for Beacon ingestion."""
from __future__ import annotations

import hashlib

from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    FetchResult,
    FetchSuccess,
    SourceEntry,
    TransientFailure,
)
from beacon.ingest.embeddings import EmbedderMode, EmbeddingResult, compute_sparse

__all__ = [
    "FakeConnector",
    "FakeEmbedder",
    "SparseOnlyFakeEmbedder",
]


class FakeEmbedder:
    """Deterministic fake embedder for tests.

    Produces stable, dimension-stable dense vectors seeded from the SHA-256
    hash of the input text and sparse vectors via the production
    ``compute_sparse`` (stable hashed vocabulary, no ``hash()`` salt).
    Counts total calls and total texts embedded so that tests can assert
    that unchanged sources produce zero embed calls.

    Args:
        dimension:   Dense vector dimension (default 8).
        model_name:  Model name string reported via ``self.model_name``.
    """

    def __init__(self, dimension: int = 8, model_name: str = "fake-model") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.mode = EmbedderMode.CLOUD
        self.call_count = 0
        self.embed_count = 0

    @property
    def fingerprint_model_id(self) -> str:
        """Model identity string for the pipeline fingerprint (mode + name)."""
        return f"{self.mode.value}:{self.model_name}"

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts deterministically.

        Args:
            texts: List of input texts.

        Returns:
            List of EmbeddingResult with seeded dense and sparse vectors.
        """
        self.call_count += 1
        self.embed_count += len(texts)
        results = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            dense = [(h >> (i * 4) & 0xF) / 16.0 for i in range(self.dimension)]
            indices, values = compute_sparse(text)
            results.append(
                EmbeddingResult(dense=dense, sparse_indices=indices, sparse_values=values)
            )
        return results

    def embed_one(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            EmbeddingResult.
        """
        return self.embed([text])[0]

    def reset_counts(self) -> None:
        """Reset call and embed counters to zero."""
        self.call_count = 0
        self.embed_count = 0


class SparseOnlyFakeEmbedder:
    """Deterministic sparse-only embedder for tests (no dense vectors).

    Useful for route-level integration tests that need a real embedder seam
    but must not depend on any dense model.  The sparse vectors are produced
    by the production ``compute_sparse`` function, so they are stable and
    reproducible across test runs.

    Args:
        model_name: Model name string reported via ``self.model_name`` and the
                    fingerprint.  Defaults to ``"fake-sparse"``.
    """

    def __init__(self, model_name: str = "fake-sparse") -> None:
        self.dimension = 8
        self.mode = EmbedderMode.SPARSE_ONLY
        self.model_name = model_name

    @property
    def fingerprint_model_id(self) -> str:
        """Model identity string for the pipeline fingerprint."""
        return f"{self.mode.value}:{self.model_name}"

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts using sparse-only (BM25) vectors.

        Args:
            texts: List of input texts.

        Returns:
            List of EmbeddingResult with ``dense=None`` and sparse vectors.
        """
        return [
            EmbeddingResult(dense=None, sparse_indices=i, sparse_values=v)
            for i, v in (compute_sparse(t) for t in texts)
        ]


class FakeConnector(Connector):
    """A fake connector backed by an in-memory dict of {uri: content_bytes}.

    Supports marking URIs as transient (returns TransientFailure) and
    removing URIs (returns ConfirmedDeletion on fetch).

    Args:
        sources: Optional initial dict of {uri: content_bytes}.
    """

    def __init__(self, sources: dict[str, bytes] | None = None) -> None:
        self._sources: dict[str, bytes] = sources or {}
        self._transient_uris: set[str] = set()
        self.fetch_count = 0
        self.enumerate_count = 0

    def add_source(self, uri: str, content: bytes) -> None:
        """Add or update a source.

        Args:
            uri:     Canonical source URI.
            content: Raw content bytes.
        """
        self._sources[uri] = content

    def remove_source(self, uri: str) -> None:
        """Remove a source (subsequent fetch returns ConfirmedDeletion).

        Args:
            uri: Canonical source URI.
        """
        self._sources.pop(uri, None)

    def set_transient(self, uri: str) -> None:
        """Mark a URI as transiently unavailable.

        Args:
            uri: Canonical source URI.
        """
        self._transient_uris.add(uri)

    def clear_transient(self, uri: str) -> None:
        """Clear transient failure state for a URI.

        Args:
            uri: Canonical source URI.
        """
        self._transient_uris.discard(uri)

    def enumerate(self) -> list[SourceEntry]:
        """Return SourceEntry records for all current sources (sorted by URI)."""
        self.enumerate_count += 1
        entries = []
        for uri, _content in sorted(self._sources.items()):
            entries.append(
                SourceEntry(
                    uri=uri,
                    title=uri.split("/")[-1],
                    connector_kind="fake",
                    media_type="text/markdown",
                    metadata={},
                )
            )
        return entries

    def fetch(self, uri: str) -> FetchResult:
        """Fetch content for a URI.

        Returns TransientFailure if marked transient, ConfirmedDeletion if not
        in the sources dict, or FetchSuccess with SHA-256 hash otherwise.

        Args:
            uri: Canonical source URI.

        Returns:
            FetchResult variant.
        """
        self.fetch_count += 1
        if uri in self._transient_uris:
            return TransientFailure(uri=uri, reason="Simulated transient failure")
        if uri not in self._sources:
            return ConfirmedDeletion(uri=uri)
        content = self._sources[uri]
        content_hash = hashlib.sha256(content).hexdigest()
        return FetchSuccess(content=content, content_hash=content_hash, media_type="text/markdown")
