"""Tests for fingerprint computation and migration behavior."""
from __future__ import annotations

import hashlib
from pathlib import Path

from beacon.ingest.fingerprint import compute_fingerprint


def _fp(
    *,
    parser_version: str = "docling-2.beacon-adapter-1",
    chunker_config_str: str = "v=base",
    model_name: str = "model",
    dimension: int = 8,
    schema_version: int = 1,
) -> str:
    """Typed wrapper with baseline defaults for per-component sensitivity tests."""
    return compute_fingerprint(
        parser_version=parser_version,
        chunker_config_str=chunker_config_str,
        model_name=model_name,
        dimension=dimension,
        schema_version=schema_version,
    )


def test_fingerprint_determinism() -> None:
    """Same inputs produce identical output across multiple calls."""
    chunker = "v=llama-index-0.14.beacon-chunker-1,parent=512,child=128,overlap=20"
    fp1 = _fp(chunker_config_str=chunker, model_name="fake-model")
    fp2 = _fp(chunker_config_str=chunker, model_name="fake-model")
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_fingerprint_sensitivity_parser_version() -> None:
    """Changing parser_version changes the fingerprint."""
    assert _fp() != _fp(parser_version="docling-3.beacon-adapter-1")


def test_fingerprint_sensitivity_chunker_config() -> None:
    """Changing chunker_config_str changes the fingerprint."""
    assert _fp() != _fp(chunker_config_str="v=different")


def test_fingerprint_sensitivity_model_name() -> None:
    """Changing model_name changes the fingerprint."""
    assert _fp(model_name="model-a") != _fp(model_name="model-b")


def test_fingerprint_sensitivity_dimension() -> None:
    """Changing dimension changes the fingerprint."""
    assert _fp() != _fp(dimension=16)


def test_fingerprint_sensitivity_schema_version() -> None:
    """Changing schema_version changes the fingerprint."""
    assert _fp(schema_version=1) != _fp(schema_version=2)


def test_incompatible_fingerprint_triggers_rebuild(tmp_path: Path) -> None:
    """When fingerprint drifts, plan_sync marks ALL sources as incompatible."""
    from tests.beacon.fakes import FakeConnector

    from beacon.ingest.planner import plan_sync
    from beacon.state.db import StateDB
    from beacon.state.repo import CollectionRepo, RevisionRepo, SourceRepo

    db = StateDB(db_path=str(tmp_path / "state.db"))
    collection_name = "test-fp-rebuild"
    CollectionRepo(db).create(name=collection_name)

    # Set up an initial live revision with old fingerprint.
    source_repo = SourceRepo(db)
    source_content = b"# Hello\n\nWorld"
    content_hash = hashlib.sha256(source_content).hexdigest()
    source_repo.upsert(
        collection_name=collection_name,
        canonical_uri="fake://doc1",
        connector_kind="fake",
        content_hash=content_hash,
    )

    # Create a live revision with an old fingerprint.
    rev_repo = RevisionRepo(db)
    rev_repo.create(
        revision_id="rev-old",
        collection_name=collection_name,
        fingerprint="old-fingerprint-abc123",
    )
    rev_repo.set_live("rev-old", collection_name=collection_name)

    # Set up connector with same content (no content change).
    connector = FakeConnector({"fake://doc1": source_content})

    # Plan with different fingerprint.
    plan = plan_sync(
        connector=connector,
        collection_name=collection_name,
        current_fingerprint="new-fingerprint-xyz789",
        source_repo=source_repo,
        db=db,
    )

    assert plan.fingerprint_drifted is True
    assert len(plan.sources_to_process) == 1
    assert plan.sources_to_process[0].action == "incompatible"
    assert len(plan.sources_unchanged) == 0

    db.close()


def test_fingerprint_drift_rebuilds_through_engine(tmp_path: Path) -> None:
    """A model change re-embeds every source into the shadow before one flip."""
    from tests.beacon.fakes import FakeConnector, FakeEmbedder

    from beacon.config import BeaconSettings, QdrantSettings, StateSettings
    from beacon.ingest.chunking import ChunkerConfig
    from beacon.ingest.sync import SyncEngine
    from beacon.state.db import StateDB
    from beacon.state.repo import CollectionRepo, RevisionRepo, SyncJobRepo

    (tmp_path / "qdrant").mkdir()
    settings = BeaconSettings(
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )
    from beacon.storage.qdrant import QdrantStore

    store = QdrantStore(settings)
    db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        collection_name = "test-fp-engine"
        CollectionRepo(db).create(name=collection_name)
        connector = FakeConnector({"fake://doc1": b"# Hello\n\nStable content."})

        SyncJobRepo(db).create(job_id="fp-job-1", collection_name=collection_name)
        engine1 = SyncEngine(
            store=store,
            db=db,
            embedder=FakeEmbedder(dimension=8, model_name="model-a"),
            chunker_config=ChunkerConfig(),
            settings=settings,
        )
        report1 = engine1.run_sync(
            collection_name=collection_name, connector=connector, job_id="fp-job-1"
        )

        # Second sync with a different embedding model: unchanged content must
        # still be fully re-embedded and rebuilt into a new physical collection.
        embedder_b = FakeEmbedder(dimension=8, model_name="model-b")
        SyncJobRepo(db).create(job_id="fp-job-2", collection_name=collection_name)
        engine2 = SyncEngine(
            store=store,
            db=db,
            embedder=embedder_b,
            chunker_config=ChunkerConfig(),
            settings=settings,
        )
        report2 = engine2.run_sync(
            collection_name=collection_name, connector=connector, job_id="fp-job-2"
        )

        assert report2.fingerprint != report1.fingerprint
        assert report2.sources_changed == 1  # incompatible counts as changed
        assert embedder_b.embed_count > 0  # actually re-embedded
        assert report2.physical_collection != report1.physical_collection

        live_rev = RevisionRepo(db).get_live(collection_name=collection_name)
        assert live_rev is not None
        assert live_rev["fingerprint"] == report2.fingerprint
        assert live_rev["physical_collection"] == report2.physical_collection
        assert store.resolve_alias(collection_name) == report2.physical_collection
    finally:
        store.close()
        db.close()
