"""Stage failure injection tests: parse, chunk, embed, stage write,
validation, and promote failures each leave the prior collection serving
through the alias and record a FAILED job with problem detail."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

import pytest
from tests.beacon.fakes import FakeConnector, FakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.errors import BackendError, IngestionError
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.sync import SyncEngine
from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    SyncJobRepo,
    SyncJobState,
)
from beacon.storage.qdrant import QdrantStore


def _make_everything(tmp_path: Path) -> tuple[QdrantStore, StateDB, BeaconSettings]:
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir(exist_ok=True)
    settings = BeaconSettings(
        qdrant=QdrantSettings(path=str(qdrant_path)),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )
    store = QdrantStore(settings)
    db = StateDB(db_path=str(tmp_path / "state.db"))
    return store, db, settings


def _setup_initial_sync(
    store: QdrantStore,
    db: StateDB,
    settings: BeaconSettings,
    collection_name: str,
) -> str:
    """Create the collection and run an initial sync. Returns the live alias target."""
    CollectionRepo(db).create(name=collection_name)
    connector = FakeConnector({"fake://doc1": b"# Hello\n\nInitial content."})
    job_id = "job-initial"
    SyncJobRepo(db).create(job_id=job_id, collection_name=collection_name)

    engine = SyncEngine(
        store=store,
        db=db,
        embedder=FakeEmbedder(dimension=8),
        chunker_config=ChunkerConfig(),
        settings=settings,
    )
    engine.run_sync(collection_name=collection_name, connector=connector, job_id=job_id)

    alias_target = store.resolve_alias(collection_name)
    assert alias_target is not None
    return alias_target


def _assert_failure_leaves_prior_serving(
    tmp_path: Path,
    *,
    collection_name: str,
    make_patch: Callable[
        [SyncEngine, FakeEmbedder], AbstractContextManager[object]
    ],
    match: str,
) -> None:
    """Shared scenario: initial sync, then a second sync with an injected
    failure; asserts FAILED job with problem detail and unchanged alias."""
    store, db, settings = _make_everything(tmp_path)
    try:
        prior_alias_target = _setup_initial_sync(store, db, settings, collection_name)

        # Change the content so the second sync must actually process it.
        connector = FakeConnector({"fake://doc1": b"# Updated\n\nChanged content."})
        job_id = "job-injected-failure"
        SyncJobRepo(db).create(job_id=job_id, collection_name=collection_name)

        embedder = FakeEmbedder(dimension=8)
        engine = SyncEngine(
            store=store,
            db=db,
            embedder=embedder,
            chunker_config=ChunkerConfig(),
            settings=settings,
        )

        with make_patch(engine, embedder):
            with pytest.raises((BackendError, IngestionError), match=match):
                engine.run_sync(
                    collection_name=collection_name,
                    connector=connector,
                    job_id=job_id,
                )

        # The job is FAILED with problem detail.
        job_row = SyncJobRepo(db).get(job_id)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.FAILED
        error_detail = json.loads(job_row["error_detail"])
        assert match.split()[0] in error_detail["message"]

        # The prior collection keeps serving through the alias.
        current_alias = store.resolve_alias(collection_name)
        assert current_alias == prior_alias_target
        assert store.collection_info(prior_alias_target) is not None

        # The shadow collection created during the failed sync must have been
        # cleaned up by the abort() call in the exception handler.  Any
        # physical collection whose name starts with the shadow prefix and is
        # NOT the prior alias target is an orphaned shadow - assert none exist.
        shadow_prefix = f"{collection_name}__rev_"
        all_collections = store.list_collections()
        orphaned_shadows = [
            c for c in all_collections
            if c.startswith(shadow_prefix) and c != prior_alias_target
        ]
        assert not orphaned_shadows, (
            f"Shadow collection(s) not cleaned up after abort: {orphaned_shadows}"
        )
    finally:
        store.close()
        db.close()


def test_rollback_on_parse_failure(tmp_path: Path) -> None:
    """Failure injected at the parse stage: FAILED job, prior alias serving."""

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch(
            "beacon.ingest.sync.parse",
            side_effect=IngestionError("Simulated parse failure"),
        )

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-parse",
        make_patch=make_patch,
        match="Simulated parse failure",
    )


def test_rollback_on_chunk_failure(tmp_path: Path) -> None:
    """Failure injected at the chunk stage: FAILED job, prior alias serving."""

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch(
            "beacon.ingest.sync.DocumentChunker.chunk",
            side_effect=BackendError("Simulated chunk failure"),
        )

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-chunk",
        make_patch=make_patch,
        match="Simulated chunk failure",
    )


def test_rollback_on_embed_failure(tmp_path: Path) -> None:
    """Failure injected at the embed stage: FAILED job, prior alias serving.

    An embed failure for a planned source must never promote a revision
    missing that source's chunks (the v1 data-loss regression class).
    """

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        def failing_embed(texts: list[str]) -> list[object]:
            raise BackendError("Simulated embed failure")

        embedder.embed = failing_embed  # type: ignore[method-assign, assignment]
        from contextlib import nullcontext

        return nullcontext()

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-embed",
        make_patch=make_patch,
        match="Simulated embed failure",
    )


def test_rollback_on_stage_write_failure(tmp_path: Path) -> None:
    """Failure injected at the shadow upsert: FAILED job, prior alias serving."""

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch.object(
            QdrantStore,
            "upsert",
            side_effect=BackendError("Simulated stage-write failure"),
        )

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-write",
        make_patch=make_patch,
        match="Simulated stage-write failure",
    )


def test_rollback_on_validation_failure(tmp_path: Path) -> None:
    """Failure at pre-promote validation: FAILED job, prior alias serving.

    Simulated by making the staged collection unreadable so the point-count
    check cannot pass; validation raises before any alias flip.
    """

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch.object(QdrantStore, "collection_info", return_value=None)

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-validate",
        make_patch=make_patch,
        match="Stage validation failed",
    )


def test_rollback_on_promote_failure(tmp_path: Path) -> None:
    """Failure injected at promote: FAILED job, prior alias serving."""

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch(
            "beacon.ingest.sync.promote",
            side_effect=BackendError("Simulated promote failure"),
        )

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-promote",
        make_patch=make_patch,
        match="Simulated promote failure",
    )


def test_rollback_on_stage_creation_failure(tmp_path: Path) -> None:
    """Failure injected at begin_stage: FAILED job, prior alias serving."""

    def make_patch(
        engine: SyncEngine, embedder: FakeEmbedder
    ) -> AbstractContextManager[object]:
        return patch(
            "beacon.ingest.sync.begin_stage",
            side_effect=BackendError("Simulated stage creation failure"),
        )

    _assert_failure_leaves_prior_serving(
        tmp_path,
        collection_name="test-rollback-stage",
        make_patch=make_patch,
        match="Simulated stage creation failure",
    )
