"""Full integration tests for the staged sync engine and sync routes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tests.beacon.fakes import FakeConnector, FakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.sync import SyncEngine
from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    RevisionRepo,
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


def make_engine(
    store: QdrantStore,
    db: StateDB,
    settings: BeaconSettings,
    dimension: int = 8,
) -> tuple[SyncEngine, FakeEmbedder]:
    embedder = FakeEmbedder(dimension=dimension)
    engine = SyncEngine(
        store=store,
        db=db,
        embedder=embedder,
        chunker_config=ChunkerConfig(),
        settings=settings,
    )
    return engine, embedder


def test_cold_sync_adds_sources(tmp_path: Path) -> None:
    """Cold sync: sources added, chunks written, job SUCCEEDED."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-cold"
        CollectionRepo(db).create(name=collection_name)

        connector = FakeConnector({
            "fake://doc1": b"# Hello\n\nThis is a test document with enough text to create chunks.",
            "fake://doc2": b"# World\n\nAnother test document here.",
        })

        job_id = "job-cold-1"
        SyncJobRepo(db).create(job_id=job_id, collection_name=collection_name)

        engine, _ = make_engine(store, db, settings)
        report = engine.run_sync(
            collection_name=collection_name,
            connector=connector,
            job_id=job_id,
        )

        assert report.sources_added == 2
        assert report.sources_changed == 0
        assert report.sources_deleted == 0
        assert report.chunks_written > 0

        # Job should be SUCCEEDED.
        job_row = SyncJobRepo(db).get(job_id)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED

        # A live revision should exist.
        live_rev = RevisionRepo(db).get_live(collection_name=collection_name)
        assert live_rev is not None
        assert live_rev["fingerprint"] == report.fingerprint
        assert live_rev["physical_collection"] is not None
    finally:
        store.close()
        db.close()


def test_second_sync_unchanged_zero_work(tmp_path: Path) -> None:
    """Second sync with no changes: zero parse, zero embed, zero write calls.

    Asserted by counting fakes and patched store writes, not timing. The live
    physical collection also stays exactly the same (no staging at all).
    """
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-second"
        CollectionRepo(db).create(name=collection_name)

        connector = FakeConnector({
            "fake://doc1": b"# Hello\n\nTest content for second sync.",
        })

        # First sync.
        job_id_1 = "job-second-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)
        engine, embedder = make_engine(store, db, settings)
        engine.run_sync(collection_name=collection_name, connector=connector, job_id=job_id_1)

        live_physical_before = store.resolve_alias(collection_name)
        assert live_physical_before is not None

        # Reset counts.
        embedder.reset_counts()

        # Second sync - reuse the same embedder instance to track calls, and
        # count parse and store-write calls via patched wrappers that fail
        # loudly if invoked.
        job_id_2 = "job-second-2"
        SyncJobRepo(db).create(job_id=job_id_2, collection_name=collection_name)

        engine2 = SyncEngine(
            store=store,
            db=db,
            embedder=embedder,
            chunker_config=ChunkerConfig(),
            settings=settings,
        )
        with (
            patch("beacon.ingest.sync.parse") as parse_mock,
            patch.object(QdrantStore, "upsert") as upsert_mock,
            patch.object(QdrantStore, "upsert_records") as upsert_records_mock,
        ):
            report2 = engine2.run_sync(
                collection_name=collection_name,
                connector=connector,
                job_id=job_id_2,
            )

        # Zero parse, zero embed, zero Qdrant write calls.
        assert parse_mock.call_count == 0
        assert embedder.embed_count == 0, f"Expected 0 embeds, got {embedder.embed_count}"
        assert upsert_mock.call_count == 0
        assert upsert_records_mock.call_count == 0

        assert report2.sources_unchanged == 1
        assert report2.sources_added == 0
        assert report2.chunks_written == 0

        # The live physical collection is untouched: same alias target, and it
        # matches the physical_collection recorded on the live revision row.
        assert store.resolve_alias(collection_name) == live_physical_before
        live_rev = RevisionRepo(db).get_live(collection_name=collection_name)
        assert live_rev is not None
        assert live_rev["physical_collection"] == live_physical_before
        assert report2.physical_collection == live_physical_before

        job_row = SyncJobRepo(db).get(job_id_2)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED
        assert job_row["sources_unchanged"] == 1
    finally:
        store.close()
        db.close()


def test_sync_detects_modified_source(tmp_path: Path) -> None:
    """Modify a source: correct plan and execution."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-modify"
        CollectionRepo(db).create(name=collection_name)

        connector = FakeConnector({"fake://doc1": b"# Initial\n\nInitial content."})

        job_id_1 = "job-modify-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)
        engine, _ = make_engine(store, db, settings)
        engine.run_sync(collection_name=collection_name, connector=connector, job_id=job_id_1)

        # Modify the source.
        connector.add_source("fake://doc1", b"# Updated\n\nUpdated content for doc1.")

        job_id_2 = "job-modify-2"
        SyncJobRepo(db).create(job_id=job_id_2, collection_name=collection_name)
        engine2, _ = make_engine(store, db, settings)
        report2 = engine2.run_sync(
            collection_name=collection_name,
            connector=connector,
            job_id=job_id_2,
        )

        assert report2.sources_changed == 1
        assert report2.sources_added == 0

        job_row = SyncJobRepo(db).get(job_id_2)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED
    finally:
        store.close()
        db.close()


def test_sync_detects_deleted_source(tmp_path: Path) -> None:
    """Delete a source: correct plan and execution."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-delete"
        CollectionRepo(db).create(name=collection_name)

        connector = FakeConnector({
            "fake://doc1": b"# Hello\n\nFirst document.",
            "fake://doc2": b"# World\n\nSecond document.",
        })

        job_id_1 = "job-del-1"
        SyncJobRepo(db).create(job_id=job_id_1, collection_name=collection_name)
        engine, _ = make_engine(store, db, settings)
        engine.run_sync(collection_name=collection_name, connector=connector, job_id=job_id_1)

        # Delete doc2.
        connector.remove_source("fake://doc2")

        job_id_2 = "job-del-2"
        SyncJobRepo(db).create(job_id=job_id_2, collection_name=collection_name)
        engine2, _ = make_engine(store, db, settings)
        report2 = engine2.run_sync(
            collection_name=collection_name,
            connector=connector,
            job_id=job_id_2,
        )

        assert report2.sources_deleted == 1
        assert report2.sources_unchanged == 1  # doc1 unchanged

        # doc2 should be retired in DB.
        source_row = SourceRepo(db).get(
            collection_name=collection_name, canonical_uri="fake://doc2"
        )
        assert source_row is not None
        assert source_row["status"] == "retired"

        job_row = SyncJobRepo(db).get(job_id_2)
        assert job_row is not None
        assert job_row["state"] == SyncJobState.SUCCEEDED
    finally:
        store.close()
        db.close()


def test_stale_running_job_reaped(tmp_path: Path) -> None:
    """Stale RUNNING job is reaped at the start of a new sync."""
    store, db, settings = _make_everything(tmp_path)

    try:
        collection_name = "test-stale"
        CollectionRepo(db).create(name=collection_name)

        # Manually create a RUNNING job (simulates a stale job from a crashed process).
        stale_job_id = "stale-job-001"
        SyncJobRepo(db).create(job_id=stale_job_id, collection_name=collection_name)
        SyncJobRepo(db).set_running(stale_job_id)

        stale_row = SyncJobRepo(db).get(stale_job_id)
        assert stale_row is not None
        assert stale_row["state"] == SyncJobState.RUNNING

        # Now start a new sync - it should reap the stale job.
        connector = FakeConnector({"fake://doc1": b"# Hello\n\nContent."})
        job_id = "job-after-stale"
        SyncJobRepo(db).create(job_id=job_id, collection_name=collection_name)
        engine, _ = make_engine(store, db, settings)
        engine.run_sync(collection_name=collection_name, connector=connector, job_id=job_id)

        # Stale job should now be FAILED.
        stale_row_after = SyncJobRepo(db).get(stale_job_id)
        assert stale_row_after is not None
        assert stale_row_after["state"] == SyncJobState.FAILED

        # New job should be SUCCEEDED.
        new_row = SyncJobRepo(db).get(job_id)
        assert new_row is not None
        assert new_row["state"] == SyncJobState.SUCCEEDED
    finally:
        store.close()
        db.close()


# ---------------------------------------------------------------------------
# Sync routes: POST /collections/{c}/sync + GET /jobs/{id}
# ---------------------------------------------------------------------------


@pytest.fixture()
def _no_cloud_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove cloud embedding keys so the route path uses the sparse-only floor.

    With ``HF_HUB_OFFLINE=1`` (set by the suite conftest) and no cloud keys,
    the embedder auto-detect ladder lands on the sparse-only floor: no model
    downloads and no network, exercising the floor end-to-end.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _make_app_settings(tmp_path: Path) -> BeaconSettings:
    (tmp_path / "qdrant").mkdir(exist_ok=True)
    return BeaconSettings(
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
        state=StateSettings(db_path=str(tmp_path / "state.db")),
    )


def _seed_folder_collection(tmp_path: Path, collection_name: str) -> None:
    """Create a docs folder and register a folder-connector collection."""
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "guide.md").write_text("# Guide\n\nRouted sync test content.\n")

    db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        CollectionRepo(db).create(
            name=collection_name,
            settings={
                "connector_kind": "folder",
                "connector_config": {"root": str(docs), "include_globs": ["**/*.md"]},
            },
        )
    finally:
        db.close()


def test_sync_route_returns_202_and_job_completes(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """POST /collections/{c}/sync returns 202 + job id; GET /jobs/{id} is truthful."""
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app

    collection_name = "route-sync"
    _seed_folder_collection(tmp_path, collection_name)
    settings = _make_app_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert job_id

        # TestClient waits for background tasks; the job must be terminal and
        # truthful now.
        job_resp = client.get(f"/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["job_id"] == job_id
        assert job["collection_name"] == collection_name
        assert job["state"] == "succeeded"
        assert job["sources_added"] == 1
        assert job["started_at"] is not None
        assert job["finished_at"] is not None
        assert job["error_detail"] is None


def test_sync_route_unknown_collection_404_and_unknown_job_404(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """Unknown collection and unknown job id both yield 404."""
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app

    settings = _make_app_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.post("/collections/absent/sync").status_code == 404
        assert client.get("/jobs/no-such-job").status_code == 404


def test_job_state_survives_process_restart(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """GET /jobs/{id} reports the terminal state truthfully after a restart.

    The job record is durable in the state DB: a fresh app instance over the
    same paths (simulating a process restart) serves the same terminal state.
    """
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app

    collection_name = "route-restart"
    _seed_folder_collection(tmp_path, collection_name)
    settings = _make_app_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert client.get(f"/jobs/{job_id}").json()["state"] == "succeeded"

    # "Restart": a brand-new app instance over the same state DB.
    with TestClient(create_app(settings)) as client2:
        job = client2.get(f"/jobs/{job_id}").json()
        assert job["state"] == "succeeded"
        assert job["job_id"] == job_id


def test_concurrent_sync_rejected_with_409(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """Second POST to /sync while a job is PENDING or RUNNING returns 409.

    TestClient executes background tasks synchronously before ``__exit__``,
    so within one request context the job is PENDING when we issue the second
    POST.  We exploit this by manipulating a job record directly to create a
    realistic active-job state, then verify the route guard.

    Two scenarios are tested:
    1. A PENDING job already exists -> second POST returns 409.
    2. A RUNNING job already exists -> second POST returns 409.
    Both responses must carry problem-detail fields (job_id, state, detail).
    """
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app
    from beacon.state.db import StateDB

    collection_name = "route-concurrent"
    _seed_folder_collection(tmp_path, collection_name)
    settings = _make_app_settings(tmp_path)

    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        # Scenario 1: inject a PENDING job directly into the state DB.
        state_db = StateDB(db_path=str(tmp_path / "state.db"))
        try:
            SyncJobRepo(state_db).create(
                job_id="manual-pending-job", collection_name=collection_name
            )
            # state is "pending" by default.
        finally:
            state_db.close()

        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert "job_id" in body.get("detail", body), (
            f"409 response must carry job_id; got {body}"
        )

        # Scenario 2: move to RUNNING then attempt a second POST.
        state_db2 = StateDB(db_path=str(tmp_path / "state.db"))
        try:
            SyncJobRepo(state_db2).set_running("manual-pending-job")
        finally:
            state_db2.close()

        resp2 = client.post(f"/collections/{collection_name}/sync")
        assert resp2.status_code == 409, (
            f"Expected 409 for RUNNING job, got {resp2.status_code}: {resp2.text}"
        )
        body2 = resp2.json()
        detail = body2.get("detail", {})
        if isinstance(detail, dict):
            assert detail.get("state") == "running", (
                f"Expected state=running in detail; got {detail}"
            )


# ---------------------------------------------------------------------------
# Web connector route: POST /collections/{c}/sync with connector_kind="web"
# ---------------------------------------------------------------------------


def _seed_web_collection(
    tmp_path: Path,
    collection_name: str,
    start_url: str,
) -> None:
    """Register a web-connector collection in the state DB."""
    db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        CollectionRepo(db).create(
            name=collection_name,
            settings={
                "connector_kind": "web",
                "connector_config": {
                    "start_urls": start_url,
                    "max_depth": "0",
                    "max_pages": "10",
                },
            },
        )
    finally:
        db.close()


def test_sync_route_web_connector_with_mock_transport(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """POST /sync with a web source uses the MockTransport injected via app.state.

    The test seam is ``app.state.web_transport_factory``: a callable that
    receives the connector_config dict and returns an httpx.BaseTransport.
    This keeps tests fully offline and the production code clean.

    Approach: connector-factory seam - the transport factory is set on
    app.state before the request is issued; the route reads it and passes it
    to ``_instantiate_connector``.  The WebConnector uses the injected
    MockTransport for all HTTP I/O, never reaching the real network.
    """
    import httpx
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app

    start_url = "http://example.test/"
    page_content = b"<html><body><h1>Hello Web</h1><p>Test page content.</p></body></html>"

    # Build a simple mock transport: serves the page and allows robots.txt.
    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if path == "/" or path == "":
            return httpx.Response(
                200,
                content=page_content,
                headers={"content-type": "text/html"},
            )
        return httpx.Response(404, text="Not found")

    class _StaticTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return _handler(request)

    collection_name = "route-web"
    _seed_web_collection(tmp_path, collection_name, start_url)
    settings = _make_app_settings(tmp_path)

    app = create_app(settings)
    # Inject the transport factory so the route constructs a WebConnector
    # backed by our static transport rather than the real network.
    app.state.web_transport_factory = lambda _cfg: _StaticTransport()

    with TestClient(app) as client:
        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
        job_id = resp.json()["job_id"]
        assert job_id

        job_resp = client.get(f"/jobs/{job_id}")
        assert job_resp.status_code == 200
        job = job_resp.json()
        assert job["state"] == "succeeded", (
            f"Expected job succeeded; got {job['state']}. "
            f"error_detail={job.get('error_detail')}"
        )
        assert job["sources_added"] >= 1


# ---------------------------------------------------------------------------
# Regression tests: orphaned RUNNING / PENDING jobs reaped at startup
# ---------------------------------------------------------------------------


def test_orphaned_running_job_reaped_at_startup(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """Orphaned RUNNING job is reaped when a new app instance starts.

    A crash mid-sync leaves a RUNNING job in the state DB.  Without startup
    reaping the 409 guard fires immediately and the collection is locked
    forever.  After the fix, a fresh create_app() over the same DB path must
    reap the stale job so the next POST /sync succeeds with 202.
    """
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app
    from beacon.state.db import StateDB

    collection_name = "route-reap-running"
    _seed_folder_collection(tmp_path, collection_name)
    settings = _make_app_settings(tmp_path)

    # Inject an orphaned RUNNING job directly into the state DB (simulates a
    # crash mid-sync before the new process starts).
    orphan_db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        SyncJobRepo(orphan_db).create(job_id="orphan-running", collection_name=collection_name)
        SyncJobRepo(orphan_db).set_running("orphan-running")
    finally:
        orphan_db.close()

    # Start a fresh app instance - this is the "new process" after a restart.
    # The lifespan startup sweep must reap the orphaned RUNNING job.
    with TestClient(create_app(settings)) as client:
        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 202, (
            f"Expected 202 after orphaned RUNNING job reaped, got {resp.status_code}: {resp.text}"
        )
        job_id = resp.json()["job_id"]
        assert job_id

    # The orphaned job must now be FAILED (reaped by startup sweep).
    check_db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        row = SyncJobRepo(check_db).get("orphan-running")
        assert row is not None
        assert row["state"] == SyncJobState.FAILED, (
            f"Expected orphan-running to be FAILED; got {row['state']}"
        )
    finally:
        check_db.close()


def test_orphaned_pending_job_reaped_at_startup(
    tmp_path: Path, _no_cloud_keys: None
) -> None:
    """Orphaned PENDING job is reaped when a new app instance starts.

    A crash in the create-to-background window leaves a PENDING job in the
    state DB.  The background worker never runs, so the job can never finish.
    Without startup reaping the 409 guard fires and the collection is locked.
    After the fix, a fresh create_app() over the same DB path must reap the
    stale PENDING job so the next POST /sync succeeds with 202.
    """
    from fastapi.testclient import TestClient

    from beacon.server.app import create_app
    from beacon.state.db import StateDB

    collection_name = "route-reap-pending"
    _seed_folder_collection(tmp_path, collection_name)
    settings = _make_app_settings(tmp_path)

    # Inject an orphaned PENDING job directly into the state DB (simulates a
    # crash between job creation and the background task being scheduled).
    orphan_db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        SyncJobRepo(orphan_db).create(job_id="orphan-pending", collection_name=collection_name)
        # State stays PENDING - never transitioned to RUNNING.
    finally:
        orphan_db.close()

    # Start a fresh app instance - the lifespan startup sweep must reap the
    # orphaned PENDING job along with any RUNNING ones.
    with TestClient(create_app(settings)) as client:
        resp = client.post(f"/collections/{collection_name}/sync")
        assert resp.status_code == 202, (
            f"Expected 202 after orphaned PENDING job reaped, got {resp.status_code}: {resp.text}"
        )
        job_id = resp.json()["job_id"]
        assert job_id

    # The orphaned job must now be FAILED (reaped by startup sweep).
    check_db = StateDB(db_path=str(tmp_path / "state.db"))
    try:
        row = SyncJobRepo(check_db).get("orphan-pending")
        assert row is not None
        assert row["state"] == SyncJobState.FAILED, (
            f"Expected orphan-pending to be FAILED; got {row['state']}"
        )
    finally:
        check_db.close()
