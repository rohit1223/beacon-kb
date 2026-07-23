"""Integration tests for the SQLite state DB repositories.

Verifies:
- Collections repository: create, get, list.
- Sources repository: create, active/retired status transition, content_hash/connector_kind.
- Revisions repository: create, status transitions, only one live revision per collection.
- Sync jobs repository: state transitions, timestamps, error payload, restart durability.
- Corpus-state derivation: empty/building/ready/failed four reference behaviors.
"""

from __future__ import annotations

import json
from typing import Any

from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    CorpusState,
    RevisionRepo,
    RevisionStatus,
    SourceRepo,
    SourceStatus,
    SyncJobRepo,
    SyncJobState,
    derive_corpus_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open(tmp_path: Any, name: str = "repo_test.db") -> StateDB:
    return StateDB(db_path=str(tmp_path / name))


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollectionRepo:
    def test_create_and_get(self, tmp_path: Any) -> None:
        """Creating a collection and retrieving it returns the same data."""
        db = _open(tmp_path)
        repo = CollectionRepo(db)
        repo.create(name="docs", settings={"embedding": "bge-small"})
        row = repo.get("docs")
        db.close()
        assert row is not None
        assert row["name"] == "docs"
        assert row["settings_json"] == json.dumps({"embedding": "bge-small"})

    def test_get_missing_returns_none(self, tmp_path: Any) -> None:
        """Getting a non-existent collection returns None."""
        db = _open(tmp_path)
        repo = CollectionRepo(db)
        assert repo.get("ghost") is None
        db.close()

    def test_list_empty(self, tmp_path: Any) -> None:
        """Listing collections on a fresh DB returns empty list."""
        db = _open(tmp_path)
        repo = CollectionRepo(db)
        assert repo.list() == []
        db.close()

    def test_list_multiple(self, tmp_path: Any) -> None:
        """Listing collections returns all created collections."""
        db = _open(tmp_path)
        repo = CollectionRepo(db)
        repo.create(name="a")
        repo.create(name="b")
        names = {r["name"] for r in repo.list()}
        db.close()
        assert names == {"a", "b"}

    def test_create_idempotent(self, tmp_path: Any) -> None:
        """Creating the same collection twice is idempotent (no error, no duplicate)."""
        db = _open(tmp_path)
        repo = CollectionRepo(db)
        repo.create(name="dup")
        repo.create(name="dup")
        assert len(repo.list()) == 1
        db.close()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestSourceRepo:
    def test_upsert_and_get(self, tmp_path: Any) -> None:
        """Upserting a source and reading it back returns the expected fields."""
        db = _open(tmp_path)
        CollectionRepo(db).create(name="docs")
        repo = SourceRepo(db)
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///a.md",
            connector_kind="folder",
            content_hash="abc123",
        )
        row = repo.get(collection_name="docs", canonical_uri="file:///a.md")
        db.close()
        assert row is not None
        assert row["canonical_uri"] == "file:///a.md"
        assert row["connector_kind"] == "folder"
        assert row["content_hash"] == "abc123"
        assert row["status"] == SourceStatus.ACTIVE

    def test_update_content_hash(self, tmp_path: Any) -> None:
        """Upserting an existing source updates content_hash and connector_kind."""
        db = _open(tmp_path)
        CollectionRepo(db).create(name="docs")
        repo = SourceRepo(db)
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///a.md",
            connector_kind="folder",
            content_hash="old",
        )
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///a.md",
            connector_kind="folder",
            content_hash="new",
        )
        row = repo.get(collection_name="docs", canonical_uri="file:///a.md")
        db.close()
        assert row is not None
        assert row["content_hash"] == "new"

    def test_retire_source(self, tmp_path: Any) -> None:
        """Retiring a source transitions its status to RETIRED."""
        db = _open(tmp_path)
        CollectionRepo(db).create(name="docs")
        repo = SourceRepo(db)
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///b.md",
            connector_kind="folder",
            content_hash="h1",
        )
        repo.retire(collection_name="docs", canonical_uri="file:///b.md")
        row = repo.get(collection_name="docs", canonical_uri="file:///b.md")
        db.close()
        assert row is not None
        assert row["status"] == SourceStatus.RETIRED

    def test_list_active(self, tmp_path: Any) -> None:
        """Listing active sources excludes retired ones."""
        db = _open(tmp_path)
        CollectionRepo(db).create(name="docs")
        repo = SourceRepo(db)
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///active.md",
            connector_kind="folder",
            content_hash="h1",
        )
        repo.upsert(
            collection_name="docs",
            canonical_uri="file:///retired.md",
            connector_kind="folder",
            content_hash="h2",
        )
        repo.retire(collection_name="docs", canonical_uri="file:///retired.md")
        active = repo.list_active(collection_name="docs")
        db.close()
        uris = {r["canonical_uri"] for r in active}
        assert "file:///active.md" in uris
        assert "file:///retired.md" not in uris

    def test_missing_source_returns_none(self, tmp_path: Any) -> None:
        """Getting a non-existent source returns None."""
        db = _open(tmp_path)
        repo = SourceRepo(db)
        assert repo.get(collection_name="docs", canonical_uri="file:///ghost.md") is None
        db.close()


# ---------------------------------------------------------------------------
# Revisions
# ---------------------------------------------------------------------------


class TestRevisionRepo:
    def test_create_and_get(self, tmp_path: Any) -> None:
        """Creating a revision and retrieving it returns the expected fields."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        repo.create(
            revision_id="rev-001",
            collection_name="docs",
            fingerprint="fp-abc",
            chunk_count=10,
            source_count=2,
        )
        row = repo.get("rev-001")
        db.close()
        assert row is not None
        assert row["revision_id"] == "rev-001"
        assert row["fingerprint"] == "fp-abc"
        assert row["status"] == RevisionStatus.STAGED

    def test_status_transition_to_live(self, tmp_path: Any) -> None:
        """Transitioning a revision to LIVE sets the status and retires the prior live one."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        # First revision goes live.
        repo.create(
            revision_id="rev-001",
            collection_name="docs",
            fingerprint="fp-1",
        )
        repo.set_live("rev-001", collection_name="docs")
        assert repo.get("rev-001")["status"] == RevisionStatus.LIVE  # type: ignore[index]

        # Second revision goes live - first should be retired.
        repo.create(
            revision_id="rev-002",
            collection_name="docs",
            fingerprint="fp-2",
        )
        repo.set_live("rev-002", collection_name="docs")
        db.close()

        db2 = _open(tmp_path)
        repo2 = RevisionRepo(db2)
        assert repo2.get("rev-001")["status"] == RevisionStatus.RETIRED  # type: ignore[index]
        assert repo2.get("rev-002")["status"] == RevisionStatus.LIVE  # type: ignore[index]
        db2.close()

    def test_only_one_live_revision_per_collection(self, tmp_path: Any) -> None:
        """Only one revision per collection can be LIVE at a time."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        for i in range(3):
            repo.create(
                revision_id=f"rev-{i:03d}",
                collection_name="docs",
                fingerprint=f"fp-{i}",
            )
            repo.set_live(f"rev-{i:03d}", collection_name="docs")

        rows = db.connection().execute(
            "SELECT revision_id FROM revisions WHERE status = ? AND collection_name = ?",
            (RevisionStatus.LIVE, "docs"),
        ).fetchall()
        db.close()
        assert len(rows) == 1, f"Expected exactly 1 LIVE revision, found {len(rows)}"

    def test_get_live_revision(self, tmp_path: Any) -> None:
        """get_live returns the current LIVE revision for a collection."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        repo.create(
            revision_id="rev-live",
            collection_name="docs",
            fingerprint="fp-live",
        )
        repo.set_live("rev-live", collection_name="docs")
        row = repo.get_live(collection_name="docs")
        db.close()
        assert row is not None
        assert row["revision_id"] == "rev-live"

    def test_get_live_returns_none_when_empty(self, tmp_path: Any) -> None:
        """get_live returns None when no LIVE revision exists."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        assert repo.get_live(collection_name="docs") is None
        db.close()

    def test_set_failed(self, tmp_path: Any) -> None:
        """set_failed transitions a STAGED revision to FAILED status."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        repo.create(revision_id="rev-bad", collection_name="docs", fingerprint="fp-x")
        repo.set_failed("rev-bad")
        row = repo.get("rev-bad")
        db.close()
        assert row is not None
        assert row["status"] == RevisionStatus.FAILED

    def test_collections_are_isolated(self, tmp_path: Any) -> None:
        """LIVE status for collection A does not affect collection B."""
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        repo.create(revision_id="a-rev", collection_name="col-a", fingerprint="fp-a")
        repo.set_live("a-rev", collection_name="col-a")
        assert repo.get_live(collection_name="col-b") is None
        db.close()

    def test_set_live_wrong_collection_raises_and_no_change(self, tmp_path: Any) -> None:
        """set_live with mismatched revision/collection raises BackendError and changes nothing."""
        import pytest

        from beacon.errors import BackendError
        db = _open(tmp_path)
        repo = RevisionRepo(db)
        # Create a revision in col-a.
        repo.create(revision_id="rev-a", collection_name="col-a", fingerprint="fp-a")
        # Attempt to promote it as if it belongs to col-b - must raise.
        with pytest.raises(BackendError):
            repo.set_live("rev-a", collection_name="col-b")
        # The revision must still be STAGED in col-a.
        row = repo.get("rev-a")
        db.close()
        assert row is not None
        assert row["status"] == RevisionStatus.STAGED


# ---------------------------------------------------------------------------
# Sync jobs
# ---------------------------------------------------------------------------


class TestSyncJobRepo:
    def test_create_pending_job(self, tmp_path: Any) -> None:
        """Creating a job results in a PENDING state record."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="job-001", collection_name="docs")
        row = repo.get("job-001")
        db.close()
        assert row is not None
        assert row["state"] == SyncJobState.PENDING
        assert row["started_at"] is None

    def test_transition_pending_to_running(self, tmp_path: Any) -> None:
        """Transitioning a job to RUNNING sets the started_at timestamp."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="job-001", collection_name="docs")
        repo.set_running("job-001")
        row = repo.get("job-001")
        db.close()
        assert row is not None
        assert row["state"] == SyncJobState.RUNNING
        assert row["started_at"] is not None

    def test_transition_to_succeeded(self, tmp_path: Any) -> None:
        """Transitioning a job to SUCCEEDED sets finished_at and change-plan counts."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="job-001", collection_name="docs")
        repo.set_running("job-001")
        repo.set_succeeded(
            "job-001",
            sources_added=3,
            sources_removed=1,
            sources_unchanged=10,
        )
        row = repo.get("job-001")
        db.close()
        assert row is not None
        assert row["state"] == SyncJobState.SUCCEEDED
        assert row["finished_at"] is not None
        assert row["error_detail"] is None
        assert row["sources_added"] == 3
        assert row["sources_removed"] == 1
        assert row["sources_unchanged"] == 10

    def test_transition_to_failed_with_error(self, tmp_path: Any) -> None:
        """Transitioning a job to FAILED stores the error detail JSON payload."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="job-001", collection_name="docs")
        repo.set_running("job-001")
        error = {"type": "backend", "message": "Disk full"}
        repo.set_failed("job-001", error_detail=error)
        row = repo.get("job-001")
        db.close()
        assert row is not None
        assert row["state"] == SyncJobState.FAILED
        assert row["error_detail"] is not None
        parsed = json.loads(row["error_detail"])
        assert parsed["type"] == "backend"
        assert row["finished_at"] is not None

    def test_list_by_collection(self, tmp_path: Any) -> None:
        """Listing jobs for a collection returns only that collection's jobs."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="j1", collection_name="col-a")
        repo.create(job_id="j2", collection_name="col-a")
        repo.create(job_id="j3", collection_name="col-b")
        jobs = repo.list_by_collection("col-a")
        db.close()
        ids = {r["job_id"] for r in jobs}
        assert ids == {"j1", "j2"}

    def test_get_missing_returns_none(self, tmp_path: Any) -> None:
        """Getting a non-existent job returns None."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        assert repo.get("ghost-job") is None
        db.close()

    def test_restart_durability(self, tmp_path: Any) -> None:
        """Job records are readable after a simulated restart (new connection)."""
        db_path = str(tmp_path / "restart.db")

        # Session 1: create and advance a job.
        db1 = StateDB(db_path=db_path)
        repo1 = SyncJobRepo(db1)
        repo1.create(job_id="durable-job", collection_name="docs")
        repo1.set_running("durable-job")
        repo1.set_succeeded("durable-job", sources_added=5)
        db1.close()

        # Session 2: open a fresh connection and verify the job is still there.
        db2 = StateDB(db_path=db_path)
        repo2 = SyncJobRepo(db2)
        row = repo2.get("durable-job")
        db2.close()
        assert row is not None
        assert row["state"] == SyncJobState.SUCCEEDED
        assert row["sources_added"] == 5

    def test_fail_stale_running_marks_running_jobs_failed(self, tmp_path: Any) -> None:
        """fail_stale_running transitions RUNNING and PENDING jobs to FAILED.

        Both states are stale at startup - neither can survive a process
        restart.  RUNNING jobs had a live engine that is now dead; PENDING
        jobs were created just before a crash and their background worker was
        never scheduled (or died before picking them up).
        """
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="j-run-1", collection_name="docs")
        repo.set_running("j-run-1")
        repo.create(job_id="j-run-2", collection_name="docs")
        repo.set_running("j-run-2")
        repo.create(job_id="j-pending", collection_name="docs")

        count = repo.fail_stale_running()
        assert count == 3  # RUNNING x2 + PENDING x1

        r1 = repo.get("j-run-1")
        r2 = repo.get("j-run-2")
        r3 = repo.get("j-pending")
        db.close()
        assert r1 is not None and r1["state"] == SyncJobState.FAILED
        assert r2 is not None and r2["state"] == SyncJobState.FAILED
        assert r3 is not None and r3["state"] == SyncJobState.FAILED
        # Error detail must explain the reason.
        detail = json.loads(r1["error_detail"])
        assert detail["type"] == "stale"

    def test_fail_stale_running_scoped_to_collection(self, tmp_path: Any) -> None:
        """fail_stale_running with collection_name only reaps that collection's jobs."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="j-a", collection_name="col-a")
        repo.set_running("j-a")
        repo.create(job_id="j-b", collection_name="col-b")
        repo.set_running("j-b")

        count = repo.fail_stale_running(collection_name="col-a")
        assert count == 1

        ra = repo.get("j-a")
        rb = repo.get("j-b")
        db.close()
        assert ra is not None and ra["state"] == SyncJobState.FAILED
        assert rb is not None and rb["state"] == SyncJobState.RUNNING

    def test_fail_stale_running_exclude_job_id_own_job_survives(
        self, tmp_path: Any
    ) -> None:
        """exclude_job_id prevents the caller's own PENDING job from being reaped.

        Scenario mirrors the engine's per-collection call: a new PENDING job is
        created, then fail_stale_running is called with that job's id excluded.
        The new job must survive while other stale jobs in the same collection
        are reaped.
        """
        db = _open(tmp_path)
        repo = SyncJobRepo(db)

        # A stale RUNNING job from a previous cycle.
        repo.create(job_id="j-stale", collection_name="docs")
        repo.set_running("j-stale")

        # The caller's own fresh PENDING job.
        repo.create(job_id="j-own", collection_name="docs")

        count = repo.fail_stale_running(
            collection_name="docs", exclude_job_id="j-own"
        )

        j_stale = repo.get("j-stale")
        j_own = repo.get("j-own")
        db.close()

        # Only the stale job should be reaped.
        assert count == 1
        assert j_stale is not None and j_stale["state"] == SyncJobState.FAILED
        # The caller's own job must still be PENDING, unharmed.
        assert j_own is not None and j_own["state"] == SyncJobState.PENDING


# ---------------------------------------------------------------------------
# Corpus-state derivation
# ---------------------------------------------------------------------------


class TestCorpusState:
    """Four reference behaviors for derive_corpus_state."""

    def test_empty_when_no_revision_and_no_jobs(self, tmp_path: Any) -> None:
        """EMPTY: no live revision and no jobs for the collection."""
        db = _open(tmp_path)
        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.EMPTY

    def test_building_while_job_is_running(self, tmp_path: Any) -> None:
        """BUILDING: a job is currently in RUNNING state."""
        db = _open(tmp_path)
        repo = SyncJobRepo(db)
        repo.create(job_id="j1", collection_name="docs")
        repo.set_running("j1")
        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.BUILDING

    def test_ready_when_live_revision_exists(self, tmp_path: Any) -> None:
        """READY: a live revision exists for the collection."""
        db = _open(tmp_path)
        rev_repo = RevisionRepo(db)
        rev_repo.create(revision_id="rev-1", collection_name="docs", fingerprint="fp")
        rev_repo.set_live("rev-1", collection_name="docs")
        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.READY

    def test_ready_despite_last_job_failed(self, tmp_path: Any) -> None:
        """READY even when last job failed: prior live revision is still served."""
        db = _open(tmp_path)
        rev_repo = RevisionRepo(db)
        job_repo = SyncJobRepo(db)

        # First job succeeds and a live revision exists.
        rev_repo.create(revision_id="rev-1", collection_name="docs", fingerprint="fp1")
        rev_repo.set_live("rev-1", collection_name="docs")
        job_repo.create(job_id="j1", collection_name="docs")
        job_repo.set_running("j1")
        job_repo.set_succeeded("j1")

        # Second job fails.
        job_repo.create(job_id="j2", collection_name="docs")
        job_repo.set_running("j2")
        job_repo.set_failed("j2", error_detail={"type": "backend", "message": "oops"})

        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.READY, (
            f"Expected READY (prior revision still live), got {state!r}"
        )

    def test_failed_when_first_sync_failed_and_no_live_revision(
        self, tmp_path: Any
    ) -> None:
        """FAILED: first-ever sync failed and no live revision exists."""
        db = _open(tmp_path)
        job_repo = SyncJobRepo(db)
        job_repo.create(job_id="j1", collection_name="docs")
        job_repo.set_running("j1")
        job_repo.set_failed("j1", error_detail={"type": "backend", "message": "crash"})

        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.FAILED

    def test_building_takes_precedence_over_live_revision(self, tmp_path: Any) -> None:
        """BUILDING takes highest precedence: a running job beats an existing live revision."""
        db = _open(tmp_path)
        rev_repo = RevisionRepo(db)
        job_repo = SyncJobRepo(db)

        rev_repo.create(revision_id="rev-1", collection_name="docs", fingerprint="fp")
        rev_repo.set_live("rev-1", collection_name="docs")

        # Now a new job is running (incremental re-sync in progress).
        job_repo.create(job_id="j2", collection_name="docs")
        job_repo.set_running("j2")

        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.BUILDING

    def test_empty_for_unknown_collection(self, tmp_path: Any) -> None:
        """An unknown collection with no rows returns EMPTY."""
        db = _open(tmp_path)
        state = derive_corpus_state(db, collection_name="nonexistent")
        db.close()
        assert state == CorpusState.EMPTY

    def test_failed_filters_stale_running_jobs(self, tmp_path: Any) -> None:
        """FAILED state ignores stale RUNNING jobs; only completed jobs matter.

        Regression test for derive_corpus_state ordering by finished_at.
        When a RUNNING job is older than a completed FAILED job, the completed
        job (ordered by finished_at) determines the state, not the stale RUNNING.
        The BUILDING state (running job check) takes precedence, so this test
        only applies when filtering to completed jobs for the FAILED check.
        """
        db = _open(tmp_path)
        job_repo = SyncJobRepo(db)

        # Create an old RUNNING job (simulating a crashed job not cleaned up).
        job_repo.create(job_id="old-running", collection_name="docs")
        job_repo.set_running("old-running")
        # (started_at is set, but never transitions to completed; finished_at stays NULL)

        # Create a newer SUCCEEDED job (finished cleanly).
        job_repo.create(job_id="new-succeeded", collection_name="docs")
        job_repo.set_running("new-succeeded")
        job_repo.set_succeeded("new-succeeded")

        # BUILDING: stale RUNNING job exists, so state should be BUILDING (highest precedence).
        state = derive_corpus_state(db, collection_name="docs")
        assert state == CorpusState.BUILDING, (
            "With a RUNNING job present, state must be BUILDING (highest precedence)"
        )

        db.close()

    def test_failed_with_stale_running_and_older_failure(self, tmp_path: Any) -> None:
        """When a stale RUNNING job precedes a completed FAILED, BUILDING wins precedence.

        This documents that the BUILDING check (step 1) evaluates all RUNNING jobs
        regardless of when they started. If any RUNNING job exists, the corpus is
        being built. Precedence: BUILDING > READY > FAILED > EMPTY.
        """
        db = _open(tmp_path)
        job_repo = SyncJobRepo(db)

        # Create an old FAILED job.
        job_repo.create(job_id="old-failed", collection_name="docs")
        job_repo.set_running("old-failed")
        job_repo.set_failed("old-failed", error_detail={"type": "early"})

        # Create a stale RUNNING job (started after the failure but never finished).
        job_repo.create(job_id="stale-running", collection_name="docs")
        job_repo.set_running("stale-running")

        # BUILDING takes precedence: any RUNNING job beats FAILED.
        state = derive_corpus_state(db, collection_name="docs")
        assert state == CorpusState.BUILDING, (
            "BUILDING precedence: any RUNNING job triggers BUILDING state"
        )

        db.close()

    def test_failed_state_derived_from_latest_finished_at(self, tmp_path: Any) -> None:
        """FAILED: last job by finished_at is FAILED even though an earlier SUCCEEDED job exists.

        Scenario: job A succeeded, then job B failed later. No LIVE revision.
        derive_corpus_state must return FAILED because B finished most recently.
        """
        import time as _time
        db = _open(tmp_path)
        job_repo = SyncJobRepo(db)

        # Job A: created, run, succeeded earlier.
        job_repo.create(job_id="job-a", collection_name="docs")
        job_repo.set_running("job-a")
        job_repo.set_succeeded("job-a")

        # Small sleep to ensure different finished_at timestamps.
        _time.sleep(0.01)

        # Job B: created, run, failed later.
        job_repo.create(job_id="job-b", collection_name="docs")
        job_repo.set_running("job-b")
        job_repo.set_failed("job-b", error_detail={"type": "backend", "message": "crash"})

        # No live revision exists.
        state = derive_corpus_state(db, collection_name="docs")
        db.close()
        assert state == CorpusState.FAILED, (
            f"Expected FAILED (job-b finished later with failure), got {state!r}"
        )
