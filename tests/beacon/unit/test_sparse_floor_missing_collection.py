"""Tests that sparse-only mode raises BackendError on missing physical collection.

Bug being fixed: _check_dense_dimension (called only in dense branch) raised
BackendError for missing collection, but the sparse-only branch called the
executor directly and returned empty results silently.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from beacon.errors import BackendError
from beacon.ingest.embeddings import EmbedderMode, EmbeddingResult, compute_sparse
from beacon.retrieval.filters import FilterSpec
from beacon.retrieval.hybrid import HybridRetriever
from beacon.state.repo import CorpusState


class SparseOnlyFakeEmbedder:
    """Deterministic sparse-only embedder: dense is always None."""

    def __init__(self) -> None:
        self.dimension = 8
        self.mode = EmbedderMode.SPARSE_ONLY
        self.model_name = "fake-sparse"

    @property
    def fingerprint_model_id(self) -> str:
        return f"{self.mode.value}:{self.model_name}"

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        results = []
        for text in texts:
            indices, values = compute_sparse(text)
            results.append(
                EmbeddingResult(dense=None, sparse_indices=indices, sparse_values=values)
            )
        return results


def _make_retriever_with_missing_collection() -> HybridRetriever:
    """Build a HybridRetriever whose store resolves to a missing physical collection."""
    mock_store = MagicMock()
    mock_db = MagicMock()

    # Corpus is READY (live revision exists).
    with patch("beacon.retrieval.hybrid.derive_corpus_state", return_value=CorpusState.READY):
        retriever = HybridRetriever(
            store=mock_store,
            db=mock_db,
            embedder=SparseOnlyFakeEmbedder(),
        )

    # The physical collection does not exist.
    mock_store.resolve_alias.return_value = None
    mock_store.collection_info.return_value = None

    return retriever


class TestSparseFloorMissingCollection:
    def test_sparse_only_missing_collection_raises_backend_error(self) -> None:
        """Sparse-only missing collection must raise BackendError, not return []."""
        retriever = _make_retriever_with_missing_collection()
        with patch("beacon.retrieval.hybrid.derive_corpus_state", return_value=CorpusState.READY):
            with pytest.raises(BackendError) as excinfo:
                retriever.search("anything", FilterSpec(collection="test-col"), top_k=5)
        assert excinfo.value.kind.value == "backend"

    def test_sparse_only_missing_collection_not_silent_empty(self) -> None:
        """The fix must not silently return []. The exception must propagate."""
        retriever = _make_retriever_with_missing_collection()
        raised = False
        with patch("beacon.retrieval.hybrid.derive_corpus_state", return_value=CorpusState.READY):
            try:
                retriever.search("anything", FilterSpec(collection="test-col"), top_k=5)
                # Should not reach here
            except BackendError:
                raised = True
        assert raised, "BackendError must be raised, not silent empty list"
