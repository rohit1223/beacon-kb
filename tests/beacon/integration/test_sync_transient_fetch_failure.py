"""Tests for transient fetch failures and confirmed deletions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.beacon.fakes import FakeConnector, FakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.errors import BackendError
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.connectors.base import TransientFailure
from beacon.ingest.sync import SyncEngine
from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    SourceRepo,
    SyncJobRepo,
    SyncJobState,
)
from beacon.storage.qdrant import QdrantStore


def _make_everything(tmp_path: Path) -> tuple[QdrantStore, StateDB, BeaconSettings]:
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    settings = BeaconSettings(
        qdrant=QdrantSettings(path=str(qdrant_path)),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )
    store = QdrantStore(settings)
    db = StateDB(db_path=str(tmp_path / "state.db"))
    return store, db, settings


def _run_sync(
    store: QdrantStore,
    db: StateDB,
    settings: BeaconSettings,
    connector: FakeConnector,
    collection_name: str,
    job_id: str,
) -> object:
    embedder = FakeEmbedder(dimension=8)
    engine = SyncEngine(
        store=store,
        db=db,
        embedder=embedder,
        chunker_config=ChunkerConfig(),
        settings=settings,
    )
    return engine.run_sync(
        collection_name=collection_name,
        connector=connector,
        job_id=job_id,
    )


def test_transient_failure_source_stays_active(tmp_path: Path) -> None:
    """Transient fetch failure: source stays active, sync report has warning."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-transient"
        CollectionRepo(db).create(name=collection_name)

        # Initial sync with both docs.
        connector = FakeConnector({
            "fake://doc1": b"# Hello\n\nStable document.",
            "fake://doc2": b"# World\n\nDocument that will become transient.",
        })

        job_id_1 = "job-trans-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)
        _run_sync(store, db, settings, connector, collection_name, job_id_1)

        # Second sync: doc2 has transient failure.
        connector.set_transient("fake://doc2")

        job_id_2 = "job-trans-2"
        SyncJobRepo(db).create(job_id=job_id_2, collection_name=collection_name)
        from beacon.ingest.sync import SyncReport
        report2 = _run_sync(store, db, settings, connector, collection_name, job_id_2)
        assert isinstance(report2, SyncReport)

        # Should have a transient failure warning.
        assert report2.transient_failures == 1
        assert any("fake://doc2" in w for w in report2.warnings)

        # doc2 should still be ACTIVE in DB (not retired).
        source_row = SourceRepo(db).get(
            collection_name=collection_name, canonical_uri="fake://doc2"
        )
        assert source_row is not None
        assert source_row["status"] == "active"

        # doc2's indexed chunks are still serving in the live collection.
        live_physical = store.resolve_alias(collection_name)
        assert live_physical is not None
        doc2_points = store.scroll_by_source_uri(live_physical, "fake://doc2")
        assert len(doc2_points) > 0

        # Job should be SUCCEEDED (transient failures don't fail the job).
        job_row = SyncJobRepo(db).get(job_id_2)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED
    finally:
        store.close()
        db.close()


def test_confirmed_deletion_retires_source(tmp_path: Path) -> None:
    """Confirmed deletion: source retired, sync report reflects deletion."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-confirmed-del"
        CollectionRepo(db).create(name=collection_name)

        connector = FakeConnector({
            "fake://doc1": b"# Keep\n\nDocument to keep.",
            "fake://doc2": b"# Delete\n\nDocument to delete.",
        })

        job_id_1 = "job-cdel-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)
        _run_sync(store, db, settings, connector, collection_name, job_id_1)

        # Remove doc2 from connector (simulates deletion).
        connector.remove_source("fake://doc2")

        job_id_2 = "job-cdel-2"
        SyncJobRepo(db).create(job_id=job_id_2, collection_name=collection_name)
        from beacon.ingest.sync import SyncReport
        report2 = _run_sync(store, db, settings, connector, collection_name, job_id_2)
        assert isinstance(report2, SyncReport)

        # Should report 1 deletion.
        assert report2.sources_deleted == 1

        # doc2 should be retired in DB.
        source_row = SourceRepo(db).get(
            collection_name=collection_name, canonical_uri="fake://doc2"
        )
        assert source_row is not None
        assert source_row["status"] == "retired"

        # doc1 unchanged.
        assert report2.sources_unchanged == 1

        job_row = SyncJobRepo(db).get(job_id_2)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED
    finally:
        store.close()
        db.close()


def test_planned_source_transient_during_processing_fails_sync(tmp_path: Path) -> None:
    """Planned source fetch returning TransientFailure during processing fails the sync.

    The planner sees doc1 as new (succeeds at planning time).  Before the
    engine's _process_source call the connector switches to returning
    TransientFailure for doc1.  The engine must raise BackendError, mark the
    job FAILED, and leave the prior alias target unchanged so the prior
    collection keeps serving.

    This guards the invariant documented in sync.py: promoting a revision
    that is missing a planned source silently drops indexed content from the
    live corpus and must never happen.
    """
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-planned-transient-processing"
        CollectionRepo(db).create(name=collection_name)

        # Initial sync with doc0 so there is a live alias to preserve.
        connector = FakeConnector({"fake://doc0": b"# Seed\n\nSeed document."})
        job_id_0 = "job-ptp-0"
        SyncJobRepo(db).create(job_id=job_id_0, collection_name=collection_name)
        _run_sync(store, db, settings, connector, collection_name, job_id_0)

        prior_alias_target = store.resolve_alias(collection_name)
        assert prior_alias_target is not None

        # Add doc1: the planner will classify it as new (sources_to_process).
        connector.add_source("fake://doc1", b"# New\n\nNew document.")

        # After planning succeeds, make doc1's fetch degrade to TransientFailure
        # so _process_source returns None.
        original_fetch = connector.fetch
        call_counts: dict[str, int] = {"doc1": 0}

        def patched_fetch(uri: str) -> object:
            if uri == "fake://doc1":
                call_counts["doc1"] += 1
                if call_counts["doc1"] > 1:
                    # Second call is from _process_source; return transient.
                    return TransientFailure(
                        uri=uri, reason="Degraded during processing"
                    )
            return original_fetch(uri)

        connector.fetch = patched_fetch  # type: ignore[assignment]

        job_id_1 = "job-ptp-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)

        with pytest.raises(BackendError, match=r"Planned source.*TransientFailure"):
            _run_sync(store, db, settings, connector, collection_name, job_id_1)

        # Job must be FAILED.
        job_row = SyncJobRepo(db).get(job_id_1)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.FAILED
        error_detail = json.loads(job_row["error_detail"])
        assert "TransientFailure" in error_detail["message"]

        # The alias is unchanged; the prior collection still serves.
        assert store.resolve_alias(collection_name) == prior_alias_target
        assert store.collection_info(prior_alias_target) is not None
    finally:
        store.close()
        db.close()
