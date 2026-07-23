"""Typed repositories over the Beacon state DB tables.

Provides four repository classes and one pure derivation function:

- ``CollectionRepo`` - create/get/list logical corpus collections.
- ``SourceRepo`` - upsert/retire/list source records (canonical URI,
  connector kind, content hash, active/retired status).
- ``RevisionRepo`` - create revisions, transition staged -> live/failed,
  enforce single-live-per-collection invariant.
- ``SyncJobRepo`` - create jobs, drive pending -> running -> succeeded/failed
  state transitions with timestamps and error payloads.
- ``derive_corpus_state`` - pure function deriving per-collection corpus
  state from durable DB rows without any in-process cache.

All writes are transactional; timestamps are UTC ISO 8601 strings.
This module has no dependency on Qdrant or FastAPI.
"""

from __future__ import annotations

import json
import sqlite3
from enum import StrEnum
from typing import Any

from beacon.errors import BackendError
from beacon.state._util import _now_iso
from beacon.state.db import StateDB

# sqlite3.Row type alias used throughout this module.
Row = sqlite3.Row


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _fetchone(cursor: sqlite3.Cursor) -> Row | None:
    """Fetch one row from *cursor* with a typed return for mypy strict mode.

    sqlite3's type stubs declare fetchone() -> Any; this wrapper narrows that
    to Row | None so the repository methods satisfy --strict without casts
    scattered at every call site.

    Args:
        cursor: An executed sqlite3.Cursor.

    Returns:
        The first result row as sqlite3.Row, or None.
    """
    result: Row | None = cursor.fetchone()
    return result


# ---------------------------------------------------------------------------
# Status / state enumerations
# ---------------------------------------------------------------------------


class SourceStatus(StrEnum):
    """Status values for source records."""

    ACTIVE = "active"
    RETIRED = "retired"


class RevisionStatus(StrEnum):
    """Status values for revision records."""

    STAGED = "staged"
    LIVE = "live"
    FAILED = "failed"
    RETIRED = "retired"


class SyncJobState(StrEnum):
    """State values for sync job records."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CorpusState(StrEnum):
    """Derived per-collection corpus health state.

    Semantics (highest-precedence first):

    BUILDING - a job is currently RUNNING for this collection.
    READY    - a LIVE revision exists; corpus is searchable.
    FAILED   - last job failed and no LIVE revision exists.
    EMPTY    - no jobs and no LIVE revision.
    """

    EMPTY = "empty"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# CollectionRepo
# ---------------------------------------------------------------------------


class CollectionRepo:
    """Repository for collection records.

    Args:
        db: Open StateDB instance.
    """

    def __init__(self, db: StateDB) -> None:
        self._conn: sqlite3.Connection = db.connection()

    def create(self, *, name: str, settings: dict[str, Any] | None = None) -> None:
        """Create a collection if it does not already exist (idempotent).

        Args:
            name:     Logical collection name (primary key).
            settings: Optional settings dict stored as JSON.

        Raises:
            BackendError: On SQLite write failure.
        """
        settings_json = json.dumps(settings) if settings else "{}"
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO collections (name, created_at, settings_json)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, _now_iso(), settings_json),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"CollectionRepo.create failed: {exc}") from exc

    def get(self, name: str) -> sqlite3.Row | None:
        """Return the collection row for *name*, or None if not found.

        Args:
            name: Collection name.

        Returns:
            sqlite3.Row with columns name, created_at, settings_json; or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    "SELECT * FROM collections WHERE name = ?", (name,)
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"CollectionRepo.get failed: {exc}") from exc

    def list(self) -> list[sqlite3.Row]:
        """Return all collection rows ordered by name.

        Returns:
            List of sqlite3.Row records.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return self._conn.execute(
                "SELECT * FROM collections ORDER BY name"
            ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"CollectionRepo.list failed: {exc}") from exc


# ---------------------------------------------------------------------------
# SourceRepo
# ---------------------------------------------------------------------------


class SourceRepo:
    """Repository for source records.

    Sources represent canonical URIs discovered by connectors.  A source can
    be ACTIVE (included in the current revision) or RETIRED (no longer present
    in the connector's listing).

    Args:
        db: Open StateDB instance.
    """

    def __init__(self, db: StateDB) -> None:
        self._conn: sqlite3.Connection = db.connection()

    def upsert(
        self,
        *,
        collection_name: str,
        canonical_uri: str,
        connector_kind: str = "",
        content_hash: str = "",
        media_type: str | None = None,
    ) -> None:
        """Create or update a source record.

        If the source already exists its ``content_hash``, ``connector_kind``,
        ``media_type``, ``status`` (reset to ACTIVE), and ``updated_at`` are updated.

        Args:
            collection_name: Owning collection name.
            canonical_uri:   Unique canonical URI for this source.
            connector_kind:  Connector type string (e.g. 'folder', 'upload').
            content_hash:    SHA-256 of the raw content for dedupe.
            media_type:      MIME type for upload sources; None for connector-backed sources.

        Raises:
            BackendError: On SQLite write failure.
        """
        now = _now_iso()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO sources
                    (collection_name, canonical_uri, connector_kind, content_hash,
                     status, media_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection_name, canonical_uri) DO UPDATE SET
                    connector_kind = excluded.connector_kind,
                    content_hash   = excluded.content_hash,
                    media_type     = excluded.media_type,
                    status         = 'active',
                    updated_at     = excluded.updated_at
                """,
                (collection_name, canonical_uri, connector_kind, content_hash,
                 SourceStatus.ACTIVE, media_type, now, now),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SourceRepo.upsert failed: {exc}") from exc

    def retire(self, *, collection_name: str, canonical_uri: str) -> None:
        """Transition a source to RETIRED status.

        Args:
            collection_name: Owning collection name.
            canonical_uri:   Source URI to retire.

        Raises:
            BackendError: On SQLite write failure.
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                UPDATE sources SET status = ?, updated_at = ?
                WHERE collection_name = ? AND canonical_uri = ?
                """,
                (SourceStatus.RETIRED, _now_iso(), collection_name, canonical_uri),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SourceRepo.retire failed: {exc}") from exc

    def get(
        self, *, collection_name: str, canonical_uri: str
    ) -> sqlite3.Row | None:
        """Return the source row for the given collection + URI, or None.

        Args:
            collection_name: Owning collection name.
            canonical_uri:   Source URI.

        Returns:
            sqlite3.Row or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    "SELECT * FROM sources WHERE collection_name = ? AND canonical_uri = ?",
                    (collection_name, canonical_uri),
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"SourceRepo.get failed: {exc}") from exc

    def list_active(self, *, collection_name: str) -> list[sqlite3.Row]:
        """Return all ACTIVE source rows for *collection_name*.

        Args:
            collection_name: Owning collection name.

        Returns:
            List of sqlite3.Row records with status = ACTIVE.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return self._conn.execute(
                "SELECT * FROM sources WHERE collection_name = ? AND status = ?",
                (collection_name, SourceStatus.ACTIVE),
            ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"SourceRepo.list_active failed: {exc}") from exc


# ---------------------------------------------------------------------------
# RevisionRepo
# ---------------------------------------------------------------------------


class RevisionRepo:
    """Repository for revision records.

    Invariant: at most one revision per collection can have status = LIVE.
    ``set_live`` enforces this atomically by retiring the previous live
    revision in the same transaction.

    Args:
        db: Open StateDB instance.
    """

    def __init__(self, db: StateDB) -> None:
        self._conn: sqlite3.Connection = db.connection()

    def create(
        self,
        *,
        revision_id: str,
        collection_name: str,
        fingerprint: str = "",
        chunk_count: int = 0,
        source_count: int = 0,
        physical_collection: str | None = None,
    ) -> None:
        """Insert a new STAGED revision record.

        Args:
            revision_id:          Unique revision identifier.
            collection_name:      Owning collection name.
            fingerprint:          Combined pipeline + content hash.
            chunk_count:          Number of chunks in this revision (0 until set_live).
            source_count:         Number of sources in this revision.
            physical_collection:  Name of the Qdrant shadow collection for this revision.

        Raises:
            BackendError: On SQLite write failure.
        """
        now = _now_iso()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO revisions
                    (revision_id, collection_name, fingerprint, status,
                     chunk_count, source_count, physical_collection, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(revision_id) DO NOTHING
                """,
                (revision_id, collection_name, fingerprint, RevisionStatus.STAGED,
                 chunk_count, source_count, physical_collection, now, now),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"RevisionRepo.create failed: {exc}") from exc

    def set_live(self, revision_id: str, *, collection_name: str) -> None:
        """Promote a STAGED revision to LIVE and retire the previous LIVE one.

        Atomically:
        1. Retire any existing LIVE revision for this collection.
        2. Set *revision_id* to LIVE.

        Args:
            revision_id:     The revision to promote.
            collection_name: Owning collection name.

        Raises:
            BackendError: On SQLite write failure.
        """
        now = _now_iso()
        try:
            self._conn.execute("BEGIN")
            # Retire any currently LIVE revision for this collection.
            self._conn.execute(
                """
                UPDATE revisions SET status = ?, updated_at = ?
                WHERE collection_name = ? AND status = ?
                """,
                (RevisionStatus.RETIRED, now, collection_name, RevisionStatus.LIVE),
            )
            # Promote the given revision - must belong to this collection and be STAGED.
            cursor = self._conn.execute(
                """
                UPDATE revisions SET status = ?, updated_at = ?
                WHERE revision_id = ? AND collection_name = ? AND status = ?
                """,
                (RevisionStatus.LIVE, now, revision_id, collection_name, RevisionStatus.STAGED),
            )
            if cursor.rowcount == 0:
                _rollback(self._conn)
                raise BackendError(
                    f"RevisionRepo.set_live: revision {revision_id!r} not found in "
                    f"collection {collection_name!r} with status staged"
                )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"RevisionRepo.set_live failed: {exc}") from exc

    def set_failed(self, revision_id: str) -> None:
        """Transition a STAGED revision to FAILED status.

        Args:
            revision_id: The revision to mark as failed.

        Raises:
            BackendError: On SQLite write failure.
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "UPDATE revisions SET status = ?, updated_at = ? WHERE revision_id = ?",
                (RevisionStatus.FAILED, _now_iso(), revision_id),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"RevisionRepo.set_failed failed: {exc}") from exc

    def get(self, revision_id: str) -> sqlite3.Row | None:
        """Return the revision row for *revision_id*, or None if not found.

        Args:
            revision_id: Revision identifier.

        Returns:
            sqlite3.Row or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    "SELECT * FROM revisions WHERE revision_id = ?",
                    (revision_id,),
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"RevisionRepo.get failed: {exc}") from exc

    def get_live(self, *, collection_name: str) -> sqlite3.Row | None:
        """Return the current LIVE revision for *collection_name*, or None.

        Args:
            collection_name: Owning collection name.

        Returns:
            sqlite3.Row with status = LIVE, or None if no live revision exists.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    "SELECT * FROM revisions WHERE collection_name = ? AND status = ?",
                    (collection_name, RevisionStatus.LIVE),
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"RevisionRepo.get_live failed: {exc}") from exc


# ---------------------------------------------------------------------------
# SyncJobRepo
# ---------------------------------------------------------------------------


class SyncJobRepo:
    """Repository for sync job records.

    State machine: PENDING -> RUNNING -> SUCCEEDED or FAILED.

    Args:
        db: Open StateDB instance.
    """

    def __init__(self, db: StateDB) -> None:
        self._conn: sqlite3.Connection = db.connection()

    def create(self, *, job_id: str, collection_name: str) -> None:
        """Create a new PENDING sync job.

        Args:
            job_id:          Unique job identifier.
            collection_name: Owning collection name.

        Raises:
            BackendError: On SQLite write failure.
        """
        now = _now_iso()
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO sync_jobs
                    (job_id, collection_name, state, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO NOTHING
                """,
                (job_id, collection_name, SyncJobState.PENDING, now),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SyncJobRepo.create failed: {exc}") from exc

    def set_running(self, job_id: str) -> None:
        """Transition a PENDING job to RUNNING and record the start timestamp.

        Args:
            job_id: Job identifier.

        Raises:
            BackendError: On SQLite write failure.
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                UPDATE sync_jobs SET
                    state        = ?,
                    started_at   = ?,
                    error_detail = NULL,
                    finished_at  = NULL
                WHERE job_id = ?
                """,
                (SyncJobState.RUNNING, _now_iso(), job_id),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SyncJobRepo.set_running failed: {exc}") from exc

    def set_succeeded(
        self,
        job_id: str,
        *,
        sources_added: int = 0,
        sources_removed: int = 0,
        sources_unchanged: int = 0,
    ) -> None:
        """Transition a RUNNING job to SUCCEEDED and record change-plan counts.

        Args:
            job_id:             Job identifier.
            sources_added:      Count of sources added in this sync.
            sources_removed:    Count of sources removed in this sync.
            sources_unchanged:  Count of sources unchanged in this sync.

        Raises:
            BackendError: On SQLite write failure.
        """
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                UPDATE sync_jobs SET
                    state             = ?,
                    finished_at       = ?,
                    error_detail      = NULL,
                    sources_added     = ?,
                    sources_removed   = ?,
                    sources_unchanged = ?
                WHERE job_id = ?
                """,
                (
                    SyncJobState.SUCCEEDED,
                    _now_iso(),
                    sources_added,
                    sources_removed,
                    sources_unchanged,
                    job_id,
                ),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SyncJobRepo.set_succeeded failed: {exc}") from exc

    def set_failed(
        self,
        job_id: str,
        *,
        error_detail: dict[str, Any] | None = None,
    ) -> None:
        """Transition a RUNNING job to FAILED and store the error payload.

        Args:
            job_id:        Job identifier.
            error_detail:  Optional problem-details dict stored as JSON.

        Raises:
            BackendError: On SQLite write failure.
        """
        error_json: str | None = json.dumps(error_detail) if error_detail is not None else None
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                UPDATE sync_jobs SET
                    state        = ?,
                    finished_at  = ?,
                    error_detail = ?
                WHERE job_id = ?
                """,
                (SyncJobState.FAILED, _now_iso(), error_json, job_id),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SyncJobRepo.set_failed failed: {exc}") from exc

    def fail_stale_running(
        self,
        collection_name: str | None = None,
        *,
        exclude_job_id: str | None = None,
    ) -> int:
        """Mark all RUNNING and PENDING jobs as FAILED with a stale-process reason.

        RUNNING and PENDING jobs cannot survive a process restart: the sync
        engine that started or was about to start them is no longer alive.
        PENDING jobs are stale too - they were created just before a crash and
        will never be picked up by the background worker that died.

        This method is called at startup (once, across all collections) to
        drain all stale in-flight state before any new requests arrive, and at
        the beginning of each per-collection sync to clean up any stragglers
        missed by the startup sweep.

        Args:
            collection_name: When provided, restrict reaping to this collection.
                When None, reap stale RUNNING and PENDING jobs across all
                collections (startup sweep).
            exclude_job_id: When provided, skip this specific job ID so that a
                caller does not reap its own just-created PENDING job.

        Returns:
            The number of jobs transitioned to FAILED.

        Raises:
            BackendError: On SQLite write failure.
        """
        reason = json.dumps(
            {
                "type": "stale",
                "message": (
                    "Job was RUNNING or PENDING at process start; presumed dead."
                ),
            }
        )
        now = _now_iso()
        stale_states = (SyncJobState.RUNNING, SyncJobState.PENDING)
        try:
            self._conn.execute("BEGIN")
            if collection_name is not None:
                if exclude_job_id is not None:
                    cursor = self._conn.execute(
                        """
                        UPDATE sync_jobs SET
                            state        = ?,
                            finished_at  = ?,
                            error_detail = ?
                        WHERE state IN (?, ?) AND collection_name = ? AND job_id != ?
                        """,
                        (
                            SyncJobState.FAILED,
                            now,
                            reason,
                            *stale_states,
                            collection_name,
                            exclude_job_id,
                        ),
                    )
                else:
                    cursor = self._conn.execute(
                        """
                        UPDATE sync_jobs SET
                            state        = ?,
                            finished_at  = ?,
                            error_detail = ?
                        WHERE state IN (?, ?) AND collection_name = ?
                        """,
                        (SyncJobState.FAILED, now, reason, *stale_states, collection_name),
                    )
            else:
                if exclude_job_id is not None:
                    cursor = self._conn.execute(
                        """
                        UPDATE sync_jobs SET
                            state        = ?,
                            finished_at  = ?,
                            error_detail = ?
                        WHERE state IN (?, ?) AND job_id != ?
                        """,
                        (SyncJobState.FAILED, now, reason, *stale_states, exclude_job_id),
                    )
                else:
                    cursor = self._conn.execute(
                        """
                        UPDATE sync_jobs SET
                            state        = ?,
                            finished_at  = ?,
                            error_detail = ?
                        WHERE state IN (?, ?)
                        """,
                        (SyncJobState.FAILED, now, reason, *stale_states),
                    )
            count = cursor.rowcount
            self._conn.execute("COMMIT")
            return count
        except sqlite3.Error as exc:
            _rollback(self._conn)
            raise BackendError(f"SyncJobRepo.fail_stale_running failed: {exc}") from exc

    def get(self, job_id: str) -> sqlite3.Row | None:
        """Return the job row for *job_id*, or None if not found.

        Args:
            job_id: Job identifier.

        Returns:
            sqlite3.Row or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    "SELECT * FROM sync_jobs WHERE job_id = ?", (job_id,)
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"SyncJobRepo.get failed: {exc}") from exc

    def get_active(self, collection_name: str) -> sqlite3.Row | None:
        """Return a PENDING or RUNNING job for *collection_name*, or None.

        Used by the sync trigger route to detect concurrent syncs and respond
        with 409 Conflict before creating a second job.

        Args:
            collection_name: Owning collection name.

        Returns:
            sqlite3.Row for the active job, or None if none exists.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return _fetchone(
                self._conn.execute(
                    """
                    SELECT * FROM sync_jobs
                    WHERE collection_name = ?
                      AND state IN ('pending', 'running')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (collection_name,),
                )
            )
        except sqlite3.Error as exc:
            raise BackendError(f"SyncJobRepo.get_active failed: {exc}") from exc

    def list_by_collection(self, collection_name: str) -> list[sqlite3.Row]:
        """Return all sync jobs for *collection_name*, ordered by created_at desc.

        Args:
            collection_name: Owning collection name.

        Returns:
            List of sqlite3.Row records.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            return self._conn.execute(
                """
                SELECT * FROM sync_jobs
                WHERE collection_name = ?
                ORDER BY created_at DESC
                """,
                (collection_name,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(
                f"SyncJobRepo.list_by_collection failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Corpus-state derivation
# ---------------------------------------------------------------------------


def derive_corpus_state(db: StateDB, *, collection_name: str) -> CorpusState:
    """Derive the per-collection corpus state from durable DB rows.

    Precedence (highest to lowest):
    1. BUILDING - a RUNNING job exists for this collection.
    2. READY    - a LIVE revision exists (searchable even if last job failed).
    3. FAILED   - last completed job by finished_at has state FAILED and no LIVE revision.
    4. EMPTY    - no jobs and no LIVE revision.

    This function is pure over the database state: no in-process cache is
    consulted.  Callers may call it after a restart and get a correct answer.

    Args:
        db:              Open StateDB instance.
        collection_name: Collection to inspect.

    Returns:
        CorpusState enum value.

    Raises:
        BackendError: If the database cannot be queried.  A broken store is a
            real failure the caller must surface, not silently coerce to EMPTY.
    """
    conn = db.connection()
    try:
        # 1. BUILDING: any job currently RUNNING.
        running_row = conn.execute(
            "SELECT 1 FROM sync_jobs WHERE collection_name = ? AND state = ? LIMIT 1",
            (collection_name, SyncJobState.RUNNING),
        ).fetchone()
        if running_row is not None:
            return CorpusState.BUILDING

        # 2. READY: a LIVE revision exists (searchable regardless of last job).
        live_row = conn.execute(
            "SELECT 1 FROM revisions WHERE collection_name = ? AND status = ? LIMIT 1",
            (collection_name, RevisionStatus.LIVE),
        ).fetchone()
        if live_row is not None:
            return CorpusState.READY

        # 3. FAILED: last completed job by finished_at has state FAILED and no LIVE revision.
        last_job_row = conn.execute(
            """
            SELECT state FROM sync_jobs
            WHERE collection_name = ? AND state IN ('succeeded', 'failed')
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (collection_name,),
        ).fetchone()
        if last_job_row is not None and last_job_row["state"] == SyncJobState.FAILED:
            return CorpusState.FAILED

        # 4. EMPTY: no relevant state exists.
        return CorpusState.EMPTY

    except sqlite3.Error as exc:
        raise BackendError(
            f"derive_corpus_state failed for {collection_name!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rollback(conn: sqlite3.Connection) -> None:
    """Attempt to roll back a transaction, suppressing secondary errors."""
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass
