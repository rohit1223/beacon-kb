"""Integration tests for hybrid retrieval against embedded Qdrant.

Covers: hybrid one-request search (dense + sparse prefetch fused with native
RRF), sparse-only degraded mode, readiness gating, embedding-mode mismatch,
determinism, and the zero-LLM-call cost contract.

The corpus is built through the real staged sync workflow (SyncEngine) so the
tests exercise the exact collections Task 02.5 produces.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client.http import models as qmodels
from tests.beacon.fakes import FakeConnector, FakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.errors import BackendError, ReadinessError
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.embeddings import EmbedderMode, EmbeddingResult, compute_sparse
from beacon.ingest.sync import SyncEngine
from beacon.retrieval.filters import FilterSpec
from beacon.retrieval.hybrid import Hit, HybridRetriever
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SyncJobRepo
from beacon.storage.payload import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from beacon.storage.qdrant import QdrantStore

COLLECTION = "hybrid-test"

DOCS = {
    "fake://widgets.md": (
        b"# Widget configuration\n\n"
        b"To configure the widget, open the settings panel and adjust the "
        b"widget alignment options until the layout looks right."
    ),
    "fake://gadgets.md": (
        b"# Gadget maintenance\n\n"
        b"Gadget maintenance requires periodic lubrication of the rotors "
        b"and inspection of the drive belts for wear."
    ),
    "fake://sprockets.md": (
        b"# Sprocket assembly\n\n"
        b"Sprocket assembly begins with aligning the teeth against the "
        b"chain guide before torquing the hub bolts."
    ),
}


class SparseOnlyFakeEmbedder:
    """Deterministic sparse-only embedder: dense is always ``None``."""

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


def _make_infra(tmp_path: Path) -> tuple[QdrantStore, StateDB, BeaconSettings]:
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    settings = BeaconSettings(
        qdrant=QdrantSettings(path=str(qdrant_path)),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )
    store = QdrantStore(settings)
    db = StateDB(db_path=str(tmp_path / "state.db"))
    return store, db, settings


def _sync_corpus(
    store: QdrantStore,
    db: StateDB,
    settings: BeaconSettings,
    embedder: FakeEmbedder | SparseOnlyFakeEmbedder,
) -> None:
    CollectionRepo(db).create(name=COLLECTION)
    connector = FakeConnector(dict(DOCS))
    job_id = "job-hybrid-1"
    SyncJobRepo(db).create(job_id=job_id, collection_name=COLLECTION)
    engine = SyncEngine(
        store=store,
        db=db,
        embedder=embedder,
        chunker_config=ChunkerConfig(),
        settings=settings,
    )
    engine.run_sync(collection_name=COLLECTION, connector=connector, job_id=job_id)


@pytest.fixture
def corpus(tmp_path: Path) -> Iterator[tuple[QdrantStore, StateDB, FakeEmbedder]]:
    store, db, settings = _make_infra(tmp_path)
    embedder = FakeEmbedder(dimension=8)
    _sync_corpus(store, db, settings, embedder)
    yield store, db, embedder
    store.close()
    db.close()


def test_hybrid_search_returns_ranked_child_hits(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """A natural-language query returns typed hits ranked by fused score."""
    store, db, embedder = corpus
    retriever = HybridRetriever(store=store, db=db, embedder=embedder)

    hits = retriever.search(
        "how do I configure the widget alignment",
        FilterSpec(collection=COLLECTION),
        top_k=5,
    )

    assert hits, "natural-language query must return hits"
    assert all(isinstance(h, Hit) for h in hits)
    scores = [h.fused_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    for hit in hits:
        assert hit.payload["kind"] == "child"
        assert hit.payload["chunk_text"]
        assert hit.rerank_score is None


def test_search_issues_exactly_one_qdrant_query_with_rrf_fusion(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """One query_points call: dense + sparse prefetch branches fused with RRF."""
    store, db, embedder = corpus
    retriever = HybridRetriever(store=store, db=db, embedder=embedder)

    client = store._client
    with patch.object(client, "query_points", wraps=client.query_points) as recorder:
        hits = retriever.search(
            "widget alignment settings",
            FilterSpec(collection=COLLECTION),
            top_k=5,
        )

    assert hits
    assert recorder.call_count == 1
    kwargs = recorder.call_args.kwargs
    prefetch = kwargs["prefetch"]
    assert len(prefetch) == 2
    usings = {branch.using for branch in prefetch}
    assert usings == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}
    fusion_query = kwargs["query"]
    assert isinstance(fusion_query, qmodels.FusionQuery)
    assert fusion_query.fusion == qmodels.Fusion.RRF
    # The compiled filter rides on every branch, not just the top level.
    for branch in prefetch:
        assert branch.filter is not None
    assert kwargs["query_filter"] is not None


def test_sparse_only_mode_returns_results_without_dense_vectors(
    tmp_path: Path,
) -> None:
    """With no dense vectors at all, search degrades to sparse-only and works."""
    store, db, settings = _make_infra(tmp_path)
    try:
        embedder = SparseOnlyFakeEmbedder()
        _sync_corpus(store, db, settings, embedder)
        retriever = HybridRetriever(store=store, db=db, embedder=embedder)

        client = store._client
        with patch.object(client, "query_points", wraps=client.query_points) as recorder:
            hits = retriever.search(
                "gadget maintenance lubrication",
                FilterSpec(collection=COLLECTION),
                top_k=5,
            )

        assert hits, "sparse-only search must still return results"
        assert recorder.call_count == 1
        kwargs = recorder.call_args.kwargs
        assert kwargs.get("prefetch") is None
        assert isinstance(kwargs["query"], qmodels.SparseVector)
        assert kwargs["using"] == SPARSE_VECTOR_NAME
        assert kwargs["query_filter"] is not None
        top_payload = hits[0].payload
        assert "gadget" in top_payload["chunk_text"].lower()
    finally:
        store.close()
        db.close()


def test_search_is_deterministic(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """The same query twice returns identical ids and scores in order."""
    store, db, embedder = corpus
    retriever = HybridRetriever(store=store, db=db, embedder=embedder)
    spec = FilterSpec(collection=COLLECTION)

    first = retriever.search("sprocket assembly chain guide", spec, top_k=5)
    second = retriever.search("sprocket assembly chain guide", spec, top_k=5)

    assert [(h.chunk_point_id, h.fused_score) for h in first] == [
        (h.chunk_point_id, h.fused_score) for h in second
    ]


def test_empty_collection_raises_readiness_error(tmp_path: Path) -> None:
    """Searching a registered but never-synced collection raises readiness."""
    store, db, _settings = _make_infra(tmp_path)
    try:
        CollectionRepo(db).create(name="empty-coll")
        retriever = HybridRetriever(store=store, db=db, embedder=FakeEmbedder(dimension=8))
        with pytest.raises(ReadinessError) as excinfo:
            retriever.search("anything", FilterSpec(collection="empty-coll"))
        assert excinfo.value.kind.value == "readiness"
    finally:
        store.close()
        db.close()


def test_building_collection_raises_readiness_error(tmp_path: Path) -> None:
    """Searching while a sync job is RUNNING raises a readiness error."""
    store, db, _settings = _make_infra(tmp_path)
    try:
        CollectionRepo(db).create(name="building-coll")
        jobs = SyncJobRepo(db)
        jobs.create(job_id="job-b1", collection_name="building-coll")
        jobs.set_running("job-b1")
        retriever = HybridRetriever(store=store, db=db, embedder=FakeEmbedder(dimension=8))
        with pytest.raises(ReadinessError):
            retriever.search("anything", FilterSpec(collection="building-coll"))
    finally:
        store.close()
        db.close()


def test_embedding_mode_mismatch_raises_backend_error(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """A query embedder with a different dense dimension is a typed backend error."""
    store, db, _embedder = corpus
    mismatched = FakeEmbedder(dimension=16)
    retriever = HybridRetriever(store=store, db=db, embedder=mismatched)

    with pytest.raises(BackendError) as excinfo:
        retriever.search("widget", FilterSpec(collection=COLLECTION))
    assert excinfo.value.kind.value == "backend"


def test_search_performs_zero_llm_calls(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """The search path never invokes an LLM, asserted with a counting fake."""
    litellm = pytest.importorskip("litellm")
    store, db, embedder = corpus
    retriever = HybridRetriever(store=store, db=db, embedder=embedder)

    with (
        patch.object(litellm, "completion") as completion_counter,
        patch.object(litellm, "embedding") as embedding_counter,
    ):
        hits = retriever.search(
            "widget alignment", FilterSpec(collection=COLLECTION), top_k=3
        )

    assert hits
    completion_counter.assert_not_called()
    embedding_counter.assert_not_called()


def test_search_with_rerank_disabled_never_imports_sentence_transformers(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """Rerank disabled (reranker=None): no sentence-transformers import at all."""
    store, db, embedder = corpus
    retriever = HybridRetriever(store=store, db=db, embedder=embedder, reranker=None)

    hits = retriever.search("widget", FilterSpec(collection=COLLECTION), top_k=3)

    assert hits
    assert "sentence_transformers" not in sys.modules
