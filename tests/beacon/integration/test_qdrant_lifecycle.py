"""Integration tests for the Qdrant store layer and shadow-collection lifecycle.

All tests run against embedded (in-process) Qdrant local mode backed by tmp_path.
No network, no server required.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from beacon.config import BeaconSettings, QdrantSettings
from beacon.errors import BackendError
from beacon.storage.lifecycle import abort, begin_stage, promote
from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    ChunkPayload,
)
from beacon.storage.qdrant import QdrantStore

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path, qdrant_path: str | None = None) -> BeaconSettings:
    """Return BeaconSettings pointing at tmp_path for embedded Qdrant."""
    path = qdrant_path or str(tmp_path / "qdrant")
    return BeaconSettings(
        qdrant=QdrantSettings(url=None, path=path)
    )


def _store(tmp_path: Path) -> QdrantStore:
    return QdrantStore(_settings(tmp_path))


def _dense_vector(dim: int = 4) -> list[float]:
    return [0.1] * dim


def _point(
    dim: int = 4,
    chunk_text: str = "hello",
) -> tuple[str, dict[str, object], ChunkPayload]:
    point_id = str(uuid.uuid4())
    vectors: dict[str, object] = {
        DENSE_VECTOR_NAME: _dense_vector(dim),
    }
    payload = ChunkPayload(
        chunk_text=chunk_text,
        source_uri="file:///test.md",
        title="Test",
        heading_path=[],
        tags=["alpha"],
        created_at=None,
        modified_at=None,
        ingested_at="2026-01-01T00:00:00Z",
        content_hash="abc",
        chunk_hash="def",
        parent_chunk_id=None,
        fingerprint="ghi",
    )
    return point_id, vectors, payload


# ---------------------------------------------------------------------------
# QdrantStore mode selection
# ---------------------------------------------------------------------------


class TestQdrantStoreModeSelection:
    """The store selects embedded vs server mode based on config."""

    def test_embedded_mode_when_no_url(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.mode == "embedded"

    def test_server_mode_label_when_url_set(self, tmp_path: Path) -> None:
        settings = BeaconSettings(
            qdrant=QdrantSettings(url="http://localhost:6333", path=str(tmp_path / "qdrant"))
        )
        store = QdrantStore(settings)
        assert store.mode == "server"

    def test_embedded_store_creates_path_on_first_use(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.list_collections()
        qdrant_path = tmp_path / "qdrant"
        assert qdrant_path.exists()


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


class TestCollectionManagement:
    """Creating, listing, and deleting physical collections."""

    def test_list_collections_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.list_collections() == []

    def test_create_collection(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("test_col", dense_dim=4)
        assert "test_col" in store.list_collections()

    def test_delete_collection(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("test_col", dense_dim=4)
        store.delete_collection("test_col")
        assert "test_col" not in store.list_collections()

    def test_collection_info(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("info_col", dense_dim=4)
        info = store.collection_info("info_col")
        assert info is not None

    def test_create_collection_twice_does_not_raise(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("dup_col", dense_dim=4)
        store.create_collection("dup_col", dense_dim=4)

    def test_delete_nonexistent_collection_is_noop(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.delete_collection("does_not_exist")


# ---------------------------------------------------------------------------
# Alias management
# ---------------------------------------------------------------------------


class TestAliasManagement:
    """Alias resolution: logical name -> physical collection."""

    def test_resolve_alias_returns_none_when_missing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.resolve_alias("no_alias") is None

    def test_create_and_resolve_alias(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("phys_col", dense_dim=4)
        store.set_alias("my_alias", "phys_col")
        assert store.resolve_alias("my_alias") == "phys_col"

    def test_swap_alias_atomically(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("phys_a", dense_dim=4)
        store.create_collection("phys_b", dense_dim=4)
        store.set_alias("live_alias", "phys_a")
        store.set_alias("live_alias", "phys_b")
        assert store.resolve_alias("live_alias") == "phys_b"

    def test_delete_alias(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("phys_col", dense_dim=4)
        store.set_alias("temp_alias", "phys_col")
        store.delete_alias("temp_alias")
        assert store.resolve_alias("temp_alias") is None


# ---------------------------------------------------------------------------
# Upsert and query
# ---------------------------------------------------------------------------


class TestUpsertAndQuery:
    """Points upserted into a physical collection are queryable by name."""

    def test_upsert_single_point(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("q_col", dense_dim=4)
        pid, vectors, payload = _point()
        store.upsert("q_col", [(pid, vectors, payload)])

    def test_query_returns_results(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("q_col", dense_dim=4)
        pid, vectors, payload = _point(chunk_text="unique content")
        store.upsert("q_col", [(pid, vectors, payload)])
        results = store.query("q_col", vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5)
        assert len(results) >= 1
        assert results[0].payload is not None
        assert results[0].payload.get("chunk_text") == "unique content"

    def test_query_against_alias(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("phys_q", dense_dim=4)
        store.set_alias("alias_q", "phys_q")
        pid, vectors, payload = _point(chunk_text="aliased content")
        store.upsert("phys_q", [(pid, vectors, payload)])
        results = store.query("alias_q", vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5)
        assert len(results) >= 1

    def test_upsert_batch(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("batch_col", dense_dim=4)
        points = [_point(chunk_text=f"text {i}") for i in range(10)]
        store.upsert("batch_col", [(p[0], p[1], p[2]) for p in points])
        results = store.query(
            "batch_col", vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=10
        )
        assert len(results) == 10

    def test_query_empty_collection_returns_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.create_collection("empty_col", dense_dim=4)
        results = store.query(
            "empty_col", vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        assert results == []

    def test_query_nonexistent_alias_returns_empty(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        results = store.query(
            "ghost_alias", vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        assert results == []


# ---------------------------------------------------------------------------
# Lifecycle: begin_stage / promote / abort
# ---------------------------------------------------------------------------


class TestStagingInvisibility:
    """Points in a shadow collection are NOT visible through the alias before promote."""

    def test_staged_points_not_visible_through_alias(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"

        # Seed the live collection
        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")
        pid0, vec0, pay0 = _point(chunk_text="live content")
        store.upsert("corpus__v0", [(pid0, vec0, pay0)])

        # Begin a stage
        stage = begin_stage(store, logical, dense_dim=4)

        # Write to shadow
        pid1, vec1, pay1 = _point(chunk_text="shadow content")
        store.upsert(stage.shadow_collection, [(pid1, vec1, pay1)])

        # Query via alias: must NOT see shadow content
        results = store.query(
            logical, vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=10
        )
        texts = [r.payload.get("chunk_text") for r in results if r.payload]
        assert "shadow content" not in texts
        assert "live content" in texts

    def test_shadow_collection_exists_but_alias_points_elsewhere(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        stage = begin_stage(store, "mylogical", dense_dim=4)
        assert store.resolve_alias("mylogical") is None
        assert stage.shadow_collection in store.list_collections()


class TestPromote:
    """promote() atomically flips the alias; no observable gap."""

    def test_promote_alias_points_to_shadow(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"

        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")

        stage = begin_stage(store, logical, dense_dim=4)
        pid, vec, pay = _point(chunk_text="new content")
        store.upsert(stage.shadow_collection, [(pid, vec, pay)])

        promote(store, stage)

        assert store.resolve_alias(logical) == stage.shadow_collection

    def test_promote_new_content_queryable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"

        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")

        stage = begin_stage(store, logical, dense_dim=4)
        pid, vec, pay = _point(chunk_text="promoted content")
        store.upsert(stage.shadow_collection, [(pid, vec, pay)])

        promote(store, stage)

        results = store.query(
            logical, vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        texts = [r.payload.get("chunk_text") for r in results if r.payload]
        assert "promoted content" in texts

    def test_promote_old_collection_cleaned_up_or_orphaned(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        logical = "corpus"

        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")

        stage = begin_stage(store, logical, dense_dim=4)
        promote(store, stage)

        # The old physical collection should be gone or renamed; alias must not point to it
        assert store.resolve_alias(logical) != "corpus__v0"

    def test_promote_from_no_prior_alias(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "fresh_corpus"

        stage = begin_stage(store, logical, dense_dim=4)
        pid, vec, pay = _point(chunk_text="first content")
        store.upsert(stage.shadow_collection, [(pid, vec, pay)])
        promote(store, stage)

        assert store.resolve_alias(logical) == stage.shadow_collection
        results = store.query(
            logical, vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        assert len(results) == 1


class TestAbort:
    """abort() drops the shadow, leaves the prior alias serving unchanged."""

    def test_abort_drops_shadow_collection(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        stage = begin_stage(store, "corpus", dense_dim=4)
        assert stage.shadow_collection in store.list_collections()
        abort(store, stage)
        assert stage.shadow_collection not in store.list_collections()

    def test_abort_leaves_alias_untouched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"
        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")

        stage = begin_stage(store, logical, dense_dim=4)
        abort(store, stage)

        assert store.resolve_alias(logical) == "corpus__v0"

    def test_abort_prior_content_still_queryable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"
        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")
        pid, vec, pay = _point(chunk_text="live before abort")
        store.upsert("corpus__v0", [(pid, vec, pay)])

        stage = begin_stage(store, logical, dense_dim=4)
        abort(store, stage)

        results = store.query(
            logical, vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        texts = [r.payload.get("chunk_text") for r in results if r.payload]
        assert "live before abort" in texts


class TestFailureSemantics:
    """Simulated failure between write and promote - prior alias stays live."""

    def test_failure_before_promote_leaves_prior_serving(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        logical = "corpus"
        store.create_collection("corpus__v0", dense_dim=4)
        store.set_alias(logical, "corpus__v0")
        pid, vec, pay = _point(chunk_text="prior content")
        store.upsert("corpus__v0", [(pid, vec, pay)])

        stage = begin_stage(store, logical, dense_dim=4)
        shadow_name = stage.shadow_collection

        # Write to shadow but do NOT promote (simulate failure)
        pid2, vec2, pay2 = _point(chunk_text="new content that never promoted")
        store.upsert(shadow_name, [(pid2, vec2, pay2)])

        # Alias still points to old collection
        assert store.resolve_alias(logical) == "corpus__v0"

        # Prior content still queryable through alias
        results = store.query(
            logical, vector=_dense_vector(4), using=DENSE_VECTOR_NAME, limit=5
        )
        texts = [r.payload.get("chunk_text") for r in results if r.payload]
        assert "prior content" in texts

        # Shadow can be cleaned up manually (abort)
        abort(store, stage)
        assert shadow_name not in store.list_collections()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


class TestErrorTranslation:
    """Qdrant client exceptions must surface as typed BackendError."""

    def test_upsert_to_nonexistent_collection_raises_backend_error(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        pid, vectors, payload = _point()
        with pytest.raises(BackendError):
            store.upsert("nonexistent_collection_xyz", [(pid, vectors, payload)])
