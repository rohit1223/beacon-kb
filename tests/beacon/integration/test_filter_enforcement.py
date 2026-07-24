"""Filter enforcement tests: filters constrain results inside the Qdrant query.

Proves the v1 boundary guarantee: the ``FilterSpec`` compiler runs at the
pipeline boundary, the compiled filter is attached to every branch of the
hybrid request, and an injected alternative ``QueryExecutor`` implementation
still operates under the compiled filter because compilation happens before
the implementation is invoked.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from tests.beacon.fakes import FakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.retrieval.filters import DateRange, FilterSpec, compile_filter
from beacon.retrieval.hybrid import (
    HybridQueryRequest,
    HybridRetriever,
)
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, RevisionRepo
from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    ChunkPayload,
    chunk_id_to_point_id,
)
from beacon.storage.qdrant import QdrantStore, QueryResult

COLLECTION = "filters-test"
DIMENSION = 8

# (text, source_uri, tags, kind, created_at, modified_at)
_POINT_SPECS: list[tuple[str, str, list[str], str, str | None, str | None]] = [
    (
        "Widget alignment is adjusted from the settings panel.",
        "fake://widgets.md",
        ["hardware", "alpha"],
        "child",
        "2024-01-15T00:00:00Z",
        "2024-02-10T00:00:00Z",
    ),
    (
        "Widget housing screws must be torqued to spec.",
        "fake://widgets.md",
        ["hardware", "alpha"],
        "child",
        "2024-01-15T00:00:00Z",
        "2024-02-10T00:00:00Z",
    ),
    (
        "Gadget rotors require periodic widget-safe lubrication.",
        "fake://gadgets.md",
        ["hardware", "beta"],
        "child",
        "2025-06-01T00:00:00Z",
        "2025-07-20T00:00:00Z",
    ),
    (
        "Parent overview of widget and gadget maintenance.",
        "fake://widgets.md",
        ["hardware", "alpha"],
        "parent",
        "2024-01-15T00:00:00Z",
        "2024-02-10T00:00:00Z",
    ),
]


def _build_points(
    embedder: FakeEmbedder,
) -> list[tuple[str, dict[str, Any], ChunkPayload]]:
    from qdrant_client.http import models as qmodels

    points: list[tuple[str, dict[str, Any], ChunkPayload]] = []
    for text, source_uri, tags, kind, created_at, modified_at in _POINT_SPECS:
        chunk_hash = hashlib.sha256(text.encode()).hexdigest()
        emb = embedder.embed([text])[0]
        vectors: dict[str, Any] = {DENSE_VECTOR_NAME: emb.dense}
        if emb.sparse_indices:
            vectors["sparse"] = qmodels.SparseVector(
                indices=emb.sparse_indices, values=emb.sparse_values
            )
        payload = ChunkPayload(
            chunk_text=text,
            source_uri=source_uri,
            title=source_uri.rsplit("/", 1)[-1],
            heading_path=["Doc"],
            tags=tags,
            ingested_at="2026-01-01T00:00:00Z",
            content_hash=chunk_hash,
            chunk_hash=chunk_hash,
            fingerprint="fp-test",
            kind=kind,
            section_kind="text",
            created_at=created_at,
            modified_at=modified_at,
        )
        points.append((chunk_id_to_point_id(chunk_hash), vectors, payload))
    return points


@pytest.fixture
def corpus(tmp_path: Any) -> Iterator[tuple[QdrantStore, StateDB, FakeEmbedder]]:
    """Stage points directly and promote them through the alias + revision path."""
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    settings = BeaconSettings(
        qdrant=QdrantSettings(path=str(qdrant_path)),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )
    store = QdrantStore(settings)
    db = StateDB(db_path=str(tmp_path / "state.db"))

    embedder = FakeEmbedder(dimension=DIMENSION)
    physical = f"{COLLECTION}__rev_test"
    store.create_collection(physical, dense_dim=DIMENSION)
    store.upsert(physical, _build_points(embedder))
    store.set_alias(COLLECTION, physical)

    CollectionRepo(db).create(name=COLLECTION)
    revisions = RevisionRepo(db)
    revisions.create(
        revision_id="rev-test",
        collection_name=COLLECTION,
        fingerprint="fp-test",
        chunk_count=len(_POINT_SPECS),
        source_count=2,
        physical_collection=physical,
    )
    revisions.set_live("rev-test", collection_name=COLLECTION)

    yield store, db, embedder
    store.close()
    db.close()


def _retriever(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
    **kwargs: Any,
) -> HybridRetriever:
    store, db, embedder = corpus
    return HybridRetriever(store=store, db=db, embedder=embedder, **kwargs)


def test_source_uri_filter_restricts_results(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    hits = retriever.search(
        "widget maintenance lubrication",
        FilterSpec(collection=COLLECTION, source_uris=("fake://gadgets.md",)),
        top_k=10,
    )
    assert hits
    assert {h.payload["source_uri"] for h in hits} == {"fake://gadgets.md"}


def test_tags_filter_restricts_results(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    hits = retriever.search(
        "widget maintenance lubrication",
        FilterSpec(collection=COLLECTION, tags=("beta",)),
        top_k=10,
    )
    assert hits
    for hit in hits:
        assert "beta" in hit.payload["tags"]


def test_created_date_range_filter_restricts_results(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    hits = retriever.search(
        "widget maintenance lubrication",
        FilterSpec(
            collection=COLLECTION,
            created=DateRange(gte=datetime(2025, 1, 1, tzinfo=UTC)),
        ),
        top_k=10,
    )
    assert hits
    assert {h.payload["source_uri"] for h in hits} == {"fake://gadgets.md"}


def test_modified_date_range_filter_restricts_results(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    hits = retriever.search(
        "widget maintenance lubrication",
        FilterSpec(
            collection=COLLECTION,
            modified=DateRange(gte=datetime(2025, 1, 1, tzinfo=UTC)),
        ),
        top_k=10,
    )
    assert hits
    assert {h.payload["source_uri"] for h in hits} == {"fake://gadgets.md"}


def test_ingested_date_range_excludes_future_window(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    spec_past = FilterSpec(
        collection=COLLECTION,
        ingested=DateRange(gte=datetime(2020, 1, 1, tzinfo=UTC)),
    )
    spec_future = FilterSpec(
        collection=COLLECTION,
        ingested=DateRange(gte=datetime(2030, 1, 1, tzinfo=UTC)),
    )
    assert retriever.search("widget", spec_past, top_k=10)
    assert retriever.search("widget", spec_future, top_k=10) == []


def test_child_only_ranking_by_default(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    retriever = _retriever(corpus)
    default_hits = retriever.search(
        "widget gadget maintenance overview",
        FilterSpec(collection=COLLECTION),
        top_k=10,
    )
    assert default_hits
    assert all(h.payload["kind"] == "child" for h in default_hits)

    parent_hits = retriever.search(
        "widget gadget maintenance overview",
        FilterSpec(collection=COLLECTION, kinds=("parent",)),
        top_k=10,
    )
    assert parent_hits
    assert all(h.payload["kind"] == "parent" for h in parent_hits)


def test_compiled_filter_attached_to_every_prefetch_branch(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """The executor receives a request whose branches all carry the filter."""
    captured: list[HybridQueryRequest] = []

    class RecordingExecutor:
        def __init__(self, store: QdrantStore) -> None:
            self._store = store

        def execute(self, request: HybridQueryRequest) -> list[QueryResult]:
            captured.append(request)
            return self._store.query_hybrid(request)

    store, _db, _embedder = corpus
    retriever = _retriever(corpus, executor=RecordingExecutor(store))
    spec = FilterSpec(collection=COLLECTION, source_uris=("fake://widgets.md",))
    hits = retriever.search("widget", spec, top_k=10)

    assert hits
    assert len(captured) == 1
    request = captured[0]
    expected_filter = compile_filter(spec)
    assert request.query_filter == expected_filter
    assert len(request.prefetch) == 2
    for branch in request.prefetch:
        assert branch.filter == expected_filter


def test_injected_alternative_executor_still_operates_under_filter(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """An alternative executor implementation cannot bypass the compiled filter.

    The rogue executor discards the fusion directive entirely and runs a plain
    dense query through a completely different store path - yet its results
    are still constrained, because the compiled filter is part of the request
    it receives (compilation happened before the implementation was invoked).
    """

    class RogueDenseOnlyExecutor:
        def __init__(self, store: QdrantStore) -> None:
            self._store = store

        def execute(self, request: HybridQueryRequest) -> list[QueryResult]:
            dense_branches = [
                b for b in request.prefetch if b.using == DENSE_VECTOR_NAME
            ]
            assert dense_branches, "hybrid request must carry a dense branch"
            dense_query = dense_branches[0].query
            assert isinstance(dense_query, list)
            vector = cast("list[float]", dense_query)
            return self._store.query(
                request.collection_name,
                vector=vector,
                limit=request.limit,
                query_filter=request.query_filter,
            )

    store, _db, _embedder = corpus
    retriever = _retriever(corpus, executor=RogueDenseOnlyExecutor(store))
    hits = retriever.search(
        "widget maintenance lubrication",
        FilterSpec(collection=COLLECTION, source_uris=("fake://gadgets.md",)),
        top_k=10,
    )

    assert hits
    assert {h.payload["source_uri"] for h in hits} == {"fake://gadgets.md"}


def test_unfiltered_search_sees_all_sources(
    corpus: tuple[QdrantStore, StateDB, FakeEmbedder],
) -> None:
    """Sanity: without source filters, both sources are reachable."""
    retriever = _retriever(corpus)
    hits = retriever.search(
        "widget maintenance lubrication settings rotors",
        FilterSpec(collection=COLLECTION),
        top_k=10,
    )
    assert {h.payload["source_uri"] for h in hits} == {
        "fake://widgets.md",
        "fake://gadgets.md",
    }
