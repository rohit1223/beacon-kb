"""SQLite-backed transactional knowledge store for beacon-kb.

Design guarantees enforced here:
- ONE SQLite database holds corpora, revisions, chunks, FTS5 BM25 rows,
  embedding rows, build runs, fingerprints, and active-revision pointers.
- Embeddings live IN SQLite - no separate JSON or vector files.
- Manifest state persists in the database - no standalone JSON manifest.
- One promotion transaction controls visibility: staged revisions invisible
  until validation completes and the promotion transaction flips active pointers.
- Rollback leaves the prior active revision fully searchable.
- Failed index writes raise typed BackendError - never swallowed, never leave
  stores drifted.
- Vectors carry declared dimension and similarity direction; missing distance
  metadata NEVER defaults to zero.
- Restart recovery reconstructs readiness and active revisions from durable
  state, never in-memory counters.
- Corpus namespace isolation: two corpora with identical source paths never
  see each other's records.
- FTS5 capability is checked at startup with typed BackendError.

Importing this module performs no side effects.  SQLiteStore is NOT registered
as a built-in default (see registry/builtins.py): a store requires a concrete
db_path and vector_dim, so callers construct and register it explicitly.
"""

from __future__ import annotations

import datetime
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from beacon_kb.errors import BackendError
from beacon_kb.models import (
    Chunk,
    ChunkId,
    ChunkKind,
    CorpusId,
    Hit,
    Query,
    Revision,
    RevisionId,
    SectionId,
    SourceId,
    make_build_run_id,
)
from beacon_kb.storage.vector_math import (
    compute_similarity,
    decode_vector,
    encode_vector,
    validate_dimension,
    validate_similarity,
    validate_unit_norm,
)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_MIGRATION_DIR = Path(__file__).parent / "migrations"
_SENTINEL_DATE = "1970-01-01T00:00:00Z"


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


# Fenced code block delimiters (triple-backtick or triple-tilde).
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


def _extract_code_content(text: str) -> str:
    """Extract lines inside fenced code blocks from *text*.

    Returns a single string of code lines (joined by newline) for use as the
    FTS5 ``code`` column value.  Preamble and postamble prose are excluded so
    that identifier and function-name searches are boosted by this column.

    Chunk records carry no explicit is_code field; code-ness is derived here
    at FTS-index time by scanning for fenced blocks in the chunk text.

    Args:
        text: Raw chunk text possibly containing fenced code blocks.

    Returns:
        String of code content lines, or empty string if no fenced blocks.
    """
    code_lines: list[str] = []
    in_fence = False
    fence_char = ""
    for line in text.splitlines():
        stripped = line.strip()
        m = _FENCE_RE.match(stripped)
        if not in_fence:
            if m:
                in_fence = True
                fence_char = m.group(1)[0]
        elif m and m.group(1)[0] == fence_char:
            in_fence = False
        elif stripped:
            code_lines.append(stripped)
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any unapplied SQL migrations in version order.

    Reads migration files from ``storage/migrations/`` named
    ``<version>_<description>.sql`` (e.g. ``0001_initial.sql``).
    Records each applied version in the ``schema_migrations`` table.

    The FTS rebuild is deliberately NOT triggered from here.  Whether the
    ``chunks_fts`` table needs repopulating is decided from durable state on
    every open via :func:`_fts_rebuild_needed`, so a crash between the version
    commit and the rebuild self-heals on the next open instead of leaving a
    permanently empty FTS index.

    Args:
        conn: Open SQLite connection in autocommit or transaction mode.

    Raises:
        BackendError: If a migration file cannot be read or executed.
    """
    # Ensure the schema_migrations table exists before we query it.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT    NOT NULL
        )
        """
    )
    conn.commit()

    applied: set[int] = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    migration_files = sorted(_MIGRATION_DIR.glob("*.sql"))
    for mf in migration_files:
        # Extract version from filename prefix (e.g. "0001" -> 1).
        try:
            version = int(mf.stem.split("_")[0])
        except ValueError:
            continue

        if version in applied:
            continue

        try:
            sql = mf.read_text(encoding="utf-8")
        except OSError as exc:
            raise BackendError(
                f"Cannot read migration file {mf}: {exc}"
            ) from exc

        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, _now_iso()),
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise BackendError(
                f"Migration {version} failed: {exc}"
            ) from exc


def _fts_rebuild_needed(conn: sqlite3.Connection) -> bool:
    """Return True if ``chunks_fts`` must be repopulated from durable state.

    The rebuild decision is derived from durable state, NOT from "was migration
    0002 freshly applied this call".  Migration 0002 drops/recreates the FTS
    table and commits its version row in a transaction separate from the
    application-layer rebuild; a crash in between would otherwise leave the DB
    at version=2 with a permanently empty ``chunks_fts``.

    Rebuild is needed when there is at least one active chunk but the FTS index
    holds zero rows.  This is safe and idempotent: it fires only when the two
    are out of sync (post-migration or post-crash) and never when the FTS index
    is already populated or when there are simply no active chunks.

    Args:
        conn: Open SQLite connection with migrations already applied.

    Returns:
        True if the FTS index should be rebuilt from the active chunks.
    """
    try:
        active_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE active = 1"
        ).fetchone()[0]
        if not active_chunks:
            return False
        fts_rows = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    except sqlite3.Error:
        # If the tables are not yet present (fresh DB mid-migration) there is
        # nothing to rebuild.
        return False
    return bool(active_chunks) and not fts_rows


def _rebuild_fts_from_chunks(conn: sqlite3.Connection) -> None:
    """Repopulate ``chunks_fts`` from active chunks after migration 0002.

    Called once after migration 0002 is freshly applied because the FTS table
    was dropped and recreated (FTS5 does not support ALTER TABLE ADD COLUMN).
    Preserves existing single-column behaviour for old databases: rows are
    rebuilt with text, heading (parent_locator), and code (fenced-block
    content) columns populated.

    Args:
        conn: Open SQLite connection with migrations already applied.

    Raises:
        BackendError: On SQLite write failure during the rebuild.
    """
    try:
        rows = conn.execute(
            "SELECT chunk_id, corpus_id, text, parent_locator FROM chunks WHERE active = 1"
        ).fetchall()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM chunks_fts")
        for row in rows:
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, corpus_id, text, heading, code) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    row["chunk_id"],
                    row["corpus_id"],
                    row["text"],
                    row["parent_locator"],
                    _extract_code_content(row["text"]),
                ),
            )
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise BackendError(f"FTS rebuild after migration failed: {exc}") from exc


def _check_fts5(conn: sqlite3.Connection) -> None:
    """Raise BackendError if FTS5 is not available in this SQLite build.

    Args:
        conn: Open SQLite connection.

    Raises:
        BackendError: If FTS5 is not compiled in.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
    except sqlite3.OperationalError as exc:
        raise BackendError(
            f"SQLite FTS5 extension is not available in this build: {exc}. "
            f"beacon-kb requires SQLite compiled with FTS5 for BM25 sparse retrieval."
        ) from exc


# ---------------------------------------------------------------------------
# Chunk serialization / deserialization
# ---------------------------------------------------------------------------


def _chunk_to_row(chunk: Chunk, corpus_id: CorpusId) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk.id),
        "corpus_id": str(corpus_id),
        "source_id": str(chunk.source_id),
        "revision_id": str(chunk.revision_id),
        "section_id": str(chunk.section_id),
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "parent_locator": chunk.parent_locator,
        "kind": chunk.kind.value,
        "token_count": chunk.token_count,
        "prev_chunk_id": str(chunk.prev_chunk_id) if chunk.prev_chunk_id else None,
        "next_chunk_id": str(chunk.next_chunk_id) if chunk.next_chunk_id else None,
        "active": 1,
        "created_at": _now_iso(),
    }


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=ChunkId(row["chunk_id"]),
        source_id=SourceId(row["source_id"]),
        revision_id=RevisionId(row["revision_id"]),
        section_id=SectionId(row["section_id"]),
        text=row["text"],
        ordinal=row["ordinal"],
        parent_locator=row["parent_locator"],
        kind=ChunkKind(row["kind"]),
        token_count=row["token_count"],
        prev_chunk_id=ChunkId(row["prev_chunk_id"]) if row["prev_chunk_id"] else None,
        next_chunk_id=ChunkId(row["next_chunk_id"]) if row["next_chunk_id"] else None,
    )


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore:
    """Transactional SQLite-backed knowledge store.

    All writes go through staged revisions invisible to readers until a single
    promotion transaction flips the active pointers and FTS5 index rows.
    Rollback leaves any prior active revision fully searchable.

    Threading: this store owns exactly ONE SQLite connection and is
    SINGLE-THREADED.  The connection is opened WITHOUT
    ``check_same_thread=False``, so using a store instance from a thread other
    than the one that created it raises immediately rather than risking silent
    corruption from an unsynchronised shared connection.  Callers needing
    concurrent access must create one store per thread or serialise access
    externally.  A pooled/locked multi-threaded variant is tracked in
    ROADMAP.md ("pooled/locked store variant").

    Args:
        db_path:    Path to the SQLite database file.  Created if absent.
        vector_dim: Declared vector dimension for this store.  All embeddings
                    must match this dimension; mismatches raise BackendError
                    immediately (never silently stored or defaulted).

    Raises:
        BackendError: If FTS5 is not available or the schema migration fails.
    """

    def __init__(self, *, db_path: str, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise BackendError(
                f"vector_dim must be a positive integer, got {vector_dim}."
            )
        self._db_path = db_path
        self._vector_dim = vector_dim
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        """Open the SQLite connection and apply migrations.

        Returns:
            Open SQLite connection with row_factory set.

        Raises:
            BackendError: On FTS5 missing or migration failure.
        """
        try:
            # check_same_thread stays at its default (True): this store owns ONE
            # connection and is single-threaded.  Failing fast on cross-thread
            # use is preferable to silent corruption from an unsynchronised
            # shared connection.  A pooled/locked variant is tracked in
            # ROADMAP.md.
            conn = sqlite3.connect(self._db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as exc:
            raise BackendError(f"Cannot open SQLite database at {self._db_path!r}: {exc}") from exc

        _check_fts5(conn)
        _apply_migrations(conn)
        # Decide the FTS rebuild from DURABLE STATE at every open, not from
        # "migration 0002 was freshly applied this call".  Migration 0002 commits
        # its version row in a separate transaction from this rebuild; if the
        # process crashed between the two, the DB would be left at version=2 with
        # an empty chunks_fts forever.  Rebuilding whenever active chunks exist
        # but chunks_fts is empty is idempotent and self-heals that crash window.
        if _fts_rebuild_needed(conn):
            _rebuild_fts_from_chunks(conn)
        return conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    # Store protocol (upsert_chunks / delete_chunks / get_chunk)
    # These write directly to the active layer (used by callers that do
    # not need staged promotion, e.g. contract tests).
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Write or update chunk records in the active layer.

        Chunks written via this method are immediately visible to readers
        (active=1).  Use ``upsert_chunks_to_staging`` for the staged
        promotion workflow.

        Args:
            chunks: List of Chunk records to persist (may be empty).

        Raises:
            BackendError: On SQLite write failure.
        """
        if not chunks:
            return
        # Derive corpus_id from the first chunk's source_id (not stored on Chunk).
        # For direct upserts we use an empty corpus_id sentinel - callers using
        # the staged workflow always supply corpus_id explicitly.
        try:
            self._conn.execute("BEGIN")
            for chunk in chunks:
                self._conn.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, corpus_id, source_id, revision_id, section_id,
                         text, ordinal, parent_locator, kind, token_count,
                         prev_chunk_id, next_chunk_id, active, created_at)
                    VALUES
                        (:chunk_id, :corpus_id, :source_id, :revision_id, :section_id,
                         :text, :ordinal, :parent_locator, :kind, :token_count,
                         :prev_chunk_id, :next_chunk_id, :active, :created_at)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        text           = excluded.text,
                        token_count    = excluded.token_count,
                        prev_chunk_id  = excluded.prev_chunk_id,
                        next_chunk_id  = excluded.next_chunk_id,
                        active         = excluded.active
                    """,
                    _chunk_to_row(chunk, CorpusId("")),
                )
                # Keep FTS5 in sync for active chunks.
                self._conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (str(chunk.id),)
                )
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, corpus_id, text, heading, code) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        str(chunk.id),
                        "",
                        chunk.text,
                        # heading column: Chunk has no heading_text field; the
                        # parent_locator carries the heading path (e.g.
                        # 'install/quickstart') and is the minimal correct source.
                        chunk.parent_locator,
                        # code column: fenced-block content derived from the text.
                        _extract_code_content(chunk.text),
                    ),
                )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"upsert_chunks failed: {exc}") from exc

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunk records by ID (both active layer and FTS5).

        Args:
            chunk_ids: List of ChunkId strings to delete.

        Raises:
            BackendError: On SQLite write failure.
        """
        if not chunk_ids:
            return
        try:
            self._conn.execute("BEGIN")
            for cid in chunk_ids:
                self._conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (cid,))
                self._conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", (cid,))
                self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"delete_chunks failed: {exc}") from exc

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single active chunk by ID, or None if not found.

        Cross-corpus safety: this method looks up chunks by ID only, without a
        corpus filter.  Isolation between corpora relies on chunk IDs being
        corpus-scoped by construction - see ``make_chunk_id()`` in models.py,
        which hashes the corpus name into every chunk ID.
        A chunk ID from corpus A can never collide with one from corpus B,
        so callers do not need to supply a corpus_id for this lookup.

        Args:
            chunk_id: ChunkId string.

        Returns:
            Chunk record or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ? AND active = 1",
                (chunk_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_chunk failed: {exc}") from exc
        if row is None:
            return None
        return _row_to_chunk(row)

    # ------------------------------------------------------------------
    # Staged promotion workflow
    # ------------------------------------------------------------------

    def stage_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision: Revision,
        canonical_uri: str = "",
    ) -> None:
        """Record a revision as staged (invisible to readers).

        Args:
            corpus_id:     Corpus namespace for the revision.
            revision:      Revision record to persist.
            canonical_uri: The canonical URI of the source document.  When
                           omitted, the string form of ``revision.source_id``
                           is used as a fallback.  Callers should always pass
                           the original canonical URI so that
                           ``get_active_revision_id`` can look it up correctly
                           across corpora with identical paths.

        Raises:
            BackendError: On SQLite write failure.
        """
        effective_uri = canonical_uri if canonical_uri else str(revision.source_id)
        try:
            self._conn.execute("BEGIN")
            # Ensure source record exists.
            self._conn.execute(
                """
                INSERT INTO sources
                    (source_id, corpus_id, canonical_uri, media_type, title,
                     extra_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO NOTHING
                """,
                (
                    str(revision.source_id),
                    str(corpus_id),
                    effective_uri,
                    "text/plain",
                    "",
                    "{}",
                    _now_iso(),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO revisions
                    (revision_id, source_id, corpus_id, content_hash,
                     pipeline_fingerprint, byte_size, fetched_at_iso, staged, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(revision_id) DO NOTHING
                """,
                (
                    str(revision.id),
                    str(revision.source_id),
                    str(corpus_id),
                    revision.content_hash,
                    revision.pipeline_fingerprint,
                    revision.byte_size,
                    revision.fetched_at_iso,
                    _now_iso(),
                ),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"stage_revision failed: {exc}") from exc

    def upsert_chunks_to_staging(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
        chunks: list[Chunk],
    ) -> None:
        """Write chunks to the staging area (active=0, invisible to readers).

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision these chunks belong to.
            chunks:      Chunk records to stage.

        Raises:
            BackendError: On SQLite write failure.
        """
        if not chunks:
            return
        try:
            self._conn.execute("BEGIN")
            for chunk in chunks:
                row = _chunk_to_row(chunk, corpus_id)
                row["active"] = 0  # staged - invisible until promotion
                self._conn.execute(
                    """
                    INSERT INTO chunks
                        (chunk_id, corpus_id, source_id, revision_id, section_id,
                         text, ordinal, parent_locator, kind, token_count,
                         prev_chunk_id, next_chunk_id, active, created_at)
                    VALUES
                        (:chunk_id, :corpus_id, :source_id, :revision_id, :section_id,
                         :text, :ordinal, :parent_locator, :kind, :token_count,
                         :prev_chunk_id, :next_chunk_id, :active, :created_at)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        text           = excluded.text,
                        token_count    = excluded.token_count,
                        active         = excluded.active
                    """,
                    row,
                )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"upsert_chunks_to_staging failed: {exc}") from exc

    def promote_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Atomically promote a staged revision to active visibility.

        This is the single visibility boundary:
        - All chunks belonging to *revision_id* have active flipped to 1.
        - FTS5 rows are inserted for the newly active chunks.
        - The active_revision_pointer for the source is updated.
        - Chunks belonging to the PREVIOUS active revision for the same
          source are retired (active=0) and removed from FTS5.
        - Embeddings for the new revision are also activated.

        The entire promotion happens in one transaction so a partial failure
        leaves the previous revision still active.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to promote.

        Returns:
            The number of chunks retired from the PREVIOUS active revision that
            this promotion superseded (0 when there was no prior revision).

        Raises:
            BackendError: If the revision is not found or the transaction fails.
        """
        retired_count = 0
        try:
            self._conn.execute("BEGIN")

            # Look up the source_id for this revision.
            rev_row = self._conn.execute(
                "SELECT source_id FROM revisions WHERE revision_id = ? AND corpus_id = ?",
                (str(revision_id), str(corpus_id)),
            ).fetchone()
            if rev_row is None:
                self._conn.execute("ROLLBACK")
                raise BackendError(
                    f"Cannot promote revision {revision_id!r}: not found in corpus {corpus_id!r}."
                )

            source_id = rev_row["source_id"]

            # Look up the canonical_uri for this source.
            src_row = self._conn.execute(
                "SELECT canonical_uri FROM sources WHERE source_id = ? AND corpus_id = ?",
                (source_id, str(corpus_id)),
            ).fetchone()
            canonical_uri = src_row["canonical_uri"] if src_row else source_id

            # Find the currently active revision for this source (if any).
            ptr_row = self._conn.execute(
                """
                SELECT revision_id FROM active_revision_pointers
                WHERE corpus_id = ? AND canonical_uri = ?
                """,
                (str(corpus_id), canonical_uri),
            ).fetchone()
            old_revision_id: str | None = ptr_row["revision_id"] if ptr_row else None

            # Retire old revision's chunks and FTS5 rows.
            if old_revision_id is not None and old_revision_id != str(revision_id):
                old_chunk_ids = [
                    row["chunk_id"]
                    for row in self._conn.execute(
                        "SELECT chunk_id FROM chunks WHERE revision_id = ? AND corpus_id = ?",
                        (old_revision_id, str(corpus_id)),
                    ).fetchall()
                ]
                if old_chunk_ids:
                    retired_count = len(old_chunk_ids)
                    # Build parametrized IN clause: placeholders is only "?,?,?"
                    # (question marks), never user-supplied strings.
                    placeholders = ",".join("?" * len(old_chunk_ids))
                    self._conn.execute(
                        f"UPDATE chunks SET active = 0 WHERE chunk_id IN ({placeholders})",  # noqa: S608
                        old_chunk_ids,
                    )
                    self._conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",  # noqa: S608
                        old_chunk_ids,
                    )
                    self._conn.execute(
                        f"UPDATE embeddings SET active = 0 WHERE chunk_id IN ({placeholders})",  # noqa: S608
                        old_chunk_ids,
                    )

            # Activate new revision's chunks.
            new_chunk_rows = self._conn.execute(
                "SELECT * FROM chunks WHERE revision_id = ? AND corpus_id = ?",
                (str(revision_id), str(corpus_id)),
            ).fetchall()

            for row in new_chunk_rows:
                self._conn.execute(
                    "UPDATE chunks SET active = 1 WHERE chunk_id = ?",
                    (row["chunk_id"],),
                )
                # Insert into FTS5 (delete first to be idempotent).
                self._conn.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", (row["chunk_id"],)
                )
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, corpus_id, text, heading, code) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        row["chunk_id"],
                        str(corpus_id),
                        row["text"],
                        row["parent_locator"],
                        _extract_code_content(row["text"]),
                    ),
                )

            # Activate embeddings for new revision.
            self._conn.execute(
                "UPDATE embeddings SET active = 1 WHERE revision_id = ? AND corpus_id = ?",
                (str(revision_id), str(corpus_id)),
            )

            # Flip the active_revision_pointer.
            self._conn.execute(
                """
                INSERT INTO active_revision_pointers
                    (corpus_id, canonical_uri, revision_id, promoted_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(corpus_id, canonical_uri) DO UPDATE SET
                    revision_id = excluded.revision_id,
                    promoted_at = excluded.promoted_at
                """,
                (str(corpus_id), canonical_uri, str(revision_id), _now_iso()),
            )

            self._conn.execute("COMMIT")
            return retired_count
        except BackendError:
            raise
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"promote_revision failed: {exc}") from exc

    def rollback_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> None:
        """Discard a staged revision without affecting the active revision.

        Deletes all staged chunks and embeddings for *revision_id* and
        removes the revision record.  The active revision (if any) remains
        fully searchable.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to discard.

        Raises:
            BackendError: On SQLite write failure.
        """
        try:
            self._conn.execute("BEGIN")

            # Collect chunk IDs for this revision.
            chunk_ids = [
                row["chunk_id"]
                for row in self._conn.execute(
                    "SELECT chunk_id FROM chunks "
                    "WHERE revision_id = ? AND corpus_id = ? AND active = 0",
                    (str(revision_id), str(corpus_id)),
                ).fetchall()
            ]

            if chunk_ids:
                # Parametrized IN clause: placeholders is only "?,?,?" (question
                # marks), never user-supplied strings.
                placeholders = ",".join("?" * len(chunk_ids))
                self._conn.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )
                self._conn.execute(
                    f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )
                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )

            self._conn.execute(
                "DELETE FROM revisions WHERE revision_id = ? AND corpus_id = ?",
                (str(revision_id), str(corpus_id)),
            )

            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"rollback_revision failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    def upsert_embedding(
        self,
        *,
        corpus_id: CorpusId,
        chunk_id: ChunkId,
        revision_id: RevisionId,
        vector: list[float],
        model_name: str,
        dimension: int,
        similarity: str,
    ) -> None:
        """Store an embedding vector for a chunk.

        The vector is validated against the store's declared dimension and
        the similarity direction is validated as a known value before storage.
        Both raise BackendError immediately - never silently accept bad data.

        Args:
            corpus_id:   Corpus namespace.
            chunk_id:    The chunk this embedding belongs to.
            revision_id: The revision this embedding was produced for.
            vector:      Unit-normalized embedding vector.
            model_name:  Name/identifier of the embedding model.
            dimension:   Declared dimension (must equal ``self._vector_dim``).
            similarity:  Similarity direction ('cosine', 'dot', 'euclidean').

        Raises:
            BackendError: If the dimension mismatches, the similarity direction
                is unknown, or (when similarity='cosine') the vector is not
                unit-normalized.  Cosine scoring treats the dot product as the
                cosine, which only holds for unit vectors; a non-unit vector is
                rejected here so callers cannot silently store unnormalized
                embeddings that would corrupt similarity scores.
        """
        # Validate dimension against both the passed value and the store's dim.
        validate_dimension(vector, self._vector_dim)
        if dimension != self._vector_dim:
            raise BackendError(
                f"Embedding dimension declaration {dimension} does not match "
                f"store's declared vector_dim {self._vector_dim}."
            )
        validate_similarity(similarity)
        # cosine_similarity() assumes unit-normalized vectors (cosine = dot product
        # for unit vectors).  Reject non-unit vectors at write time so callers cannot
        # silently store unnormalized embeddings that would corrupt similarity scores.
        if similarity == "cosine":
            validate_unit_norm(vector)

        blob = encode_vector(vector)
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO embeddings
                    (chunk_id, corpus_id, revision_id, model_name, dimension,
                     similarity, vector_blob, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    vector_blob = excluded.vector_blob,
                    model_name  = excluded.model_name,
                    similarity  = excluded.similarity,
                    dimension   = excluded.dimension
                """,
                (
                    str(chunk_id),
                    str(corpus_id),
                    str(revision_id),
                    model_name,
                    dimension,
                    similarity,
                    blob,
                    _now_iso(),
                ),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"upsert_embedding failed: {exc}") from exc

    def get_embedding(self, chunk_id: str) -> list[float] | None:
        """Retrieve the active embedding vector for a chunk, or None.

        Args:
            chunk_id: ChunkId string.

        Returns:
            List of floats (the vector) or None if not found / not active.

        Raises:
            BackendError: On I/O or decode failure.
        """
        try:
            row = self._conn.execute(
                "SELECT vector_blob, dimension FROM embeddings WHERE chunk_id = ? AND active = 1",
                (chunk_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_embedding failed: {exc}") from exc
        if row is None:
            return None
        return decode_vector(row["vector_blob"], row["dimension"])

    # ------------------------------------------------------------------
    # FTS5 sparse retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: Query,
        *,
        weights: tuple[float, float, float] | None = None,
    ) -> list[Hit]:
        """BM25 sparse retrieval using FTS5 with optional per-column weights.

        The FTS5 table (migration 0002) indexes three columns separately:
        ``text`` (chunk body), ``heading`` (parent locator heading path), and
        ``code`` (fenced-block content).  When *weights* is provided, the
        bm25() ranking uses those per-column weights; UNINDEXED columns
        (chunk_id, corpus_id) always receive weight 0.  When *weights* is
        omitted, plain ``bm25(chunks_fts)`` is used, preserving the existing
        single-expression behaviour for all databases.

        Epic 03's sparse retriever adopts the per-column weights; this method
        only provides the store capability.

        Args:
            query:   Query record with text, corpus_id filter, and top_k.
            weights: Optional (w_text, w_heading, w_code) BM25 column weights.
                     Safe default is None (uniform FTS5 weighting).

        Returns:
            List of Hit records with sparse_score set, ordered descending.
            dense_score, fusion_score, and rerank_score are all None.

        Raises:
            BackendError: On index read failure.
        """
        bm25_expr = "bm25(chunks_fts)"
        if weights is not None:
            w_text, w_heading, w_code = (float(w) for w in weights)
            # Column order matches migration 0002: chunk_id(0), corpus_id(0),
            # text, heading, code.  Values are validated floats formatted by
            # Python - never raw user strings - so interpolation is safe.
            bm25_expr = f"bm25(chunks_fts, 0.0, 0.0, {w_text}, {w_heading}, {w_code})"
        try:
            if query.corpus_id is not None:
                rows = self._conn.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.source_id,
                        c.revision_id,
                        c.section_id,
                        c.text,
                        c.ordinal,
                        c.parent_locator,
                        c.kind,
                        c.token_count,
                        c.prev_chunk_id,
                        c.next_chunk_id,
                        {bm25_expr} AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND chunks_fts.corpus_id = ?
                      AND c.active = 1
                    ORDER BY score
                    LIMIT ?
                    """,  # noqa: S608 - bm25_expr is built from validated floats only
                    (query.text, str(query.corpus_id), query.top_k),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.source_id,
                        c.revision_id,
                        c.section_id,
                        c.text,
                        c.ordinal,
                        c.parent_locator,
                        c.kind,
                        c.token_count,
                        c.prev_chunk_id,
                        c.next_chunk_id,
                        {bm25_expr} AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND c.active = 1
                    ORDER BY score
                    LIMIT ?
                    """,  # noqa: S608 - bm25_expr is built from validated floats only
                    (query.text, query.top_k),
                ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"FTS5 sparse retrieve failed: {exc}") from exc

        hits: list[Hit] = []
        for row in rows:
            chunk = Chunk(
                id=ChunkId(row["chunk_id"]),
                source_id=SourceId(row["source_id"]),
                revision_id=RevisionId(row["revision_id"]),
                section_id=SectionId(row["section_id"]),
                text=row["text"],
                ordinal=row["ordinal"],
                parent_locator=row["parent_locator"],
                kind=ChunkKind(row["kind"]),
                token_count=row["token_count"],
                prev_chunk_id=ChunkId(row["prev_chunk_id"]) if row["prev_chunk_id"] else None,
                next_chunk_id=ChunkId(row["next_chunk_id"]) if row["next_chunk_id"] else None,
            )
            # BM25 in FTS5 returns negative values (lower = more relevant).
            # Negate so higher sparse_score = more relevant, consistent with contract.
            raw_bm25: float = row["score"]
            sparse_score = -raw_bm25  # FTS5 bm25() returns negative: negate to get higher=better
            hits.append(Hit(chunk=chunk, sparse_score=sparse_score))

        # Sort descending by sparse_score (higher is better).
        hits.sort(key=lambda h: h.sparse_score or 0.0, reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Dense (vector) retrieval
    # ------------------------------------------------------------------

    def dense_retrieve(
        self,
        *,
        query_vector: list[float],
        corpus_id: CorpusId | None,
        top_k: int,
        similarity: str,
    ) -> list[Hit]:
        """NumPy similarity search over active embeddings.

        All vectors are loaded from SQLite into memory and similarity is
        computed via NumPy.  This is acceptable for local/embedded deployments.

        Args:
            query_vector: Unit-normalized query embedding vector.
            corpus_id:    Corpus filter (None = all corpora).
            top_k:        Maximum number of hits to return.
            similarity:   Declared similarity direction.

        Returns:
            List of Hit records with dense_score set, ordered descending.
            sparse_score, fusion_score, and rerank_score are all None.

        Raises:
            BackendError: On I/O or similarity computation failure.
        """
        validate_dimension(query_vector, self._vector_dim)
        validate_similarity(similarity)

        try:
            if corpus_id is not None:
                emb_rows = self._conn.execute(
                    """
                    SELECT e.chunk_id, e.vector_blob, e.dimension
                    FROM embeddings e
                    WHERE e.corpus_id = ? AND e.active = 1
                    """,
                    (str(corpus_id),),
                ).fetchall()
            else:
                emb_rows = self._conn.execute(
                    "SELECT e.chunk_id, e.vector_blob, e.dimension "
                    "FROM embeddings e WHERE e.active = 1"
                ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"dense_retrieve embedding load failed: {exc}") from exc

        if not emb_rows:
            return []

        # Decode all vectors.
        chunk_ids: list[str] = []
        vectors: list[list[float]] = []
        for row in emb_rows:
            try:
                vec = decode_vector(row["vector_blob"], row["dimension"])
            except BackendError:
                continue  # Skip corrupt blobs rather than surfacing a partial result.
            chunk_ids.append(row["chunk_id"])
            vectors.append(vec)

        if not vectors:
            return []

        scores = compute_similarity(query_vector, vectors, similarity=similarity)

        # Rank by score descending and take top_k.
        ranked = sorted(zip(scores, chunk_ids, strict=True), key=lambda x: x[0], reverse=True)[
            :top_k
        ]

        # Load full chunk records for the top hits.
        hits: list[Hit] = []
        try:
            for score, cid in ranked:
                row = self._conn.execute(
                    "SELECT * FROM chunks WHERE chunk_id = ? AND active = 1", (cid,)
                ).fetchone()
                if row is None:
                    continue
                chunk = _row_to_chunk(row)
                hits.append(Hit(chunk=chunk, dense_score=float(score)))
        except sqlite3.Error as exc:
            raise BackendError(f"dense_retrieve chunk load failed: {exc}") from exc

        return hits

    # ------------------------------------------------------------------
    # Active revision pointer queries
    # ------------------------------------------------------------------

    def get_active_revision_id(
        self, *, corpus_id: CorpusId, canonical_uri: str
    ) -> RevisionId | None:
        """Return the active RevisionId for *canonical_uri* in *corpus_id*, or None.

        Args:
            corpus_id:     Corpus namespace.
            canonical_uri: Canonical source URI.

        Returns:
            Active RevisionId or None if no revision has been promoted.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                """
                SELECT revision_id FROM active_revision_pointers
                WHERE corpus_id = ? AND canonical_uri = ?
                """,
                (str(corpus_id), canonical_uri),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_active_revision_id failed: {exc}") from exc
        if row is None:
            return None
        return RevisionId(row["revision_id"])

    # ------------------------------------------------------------------
    # Manifest read-only query helpers
    # These methods expose typed public read paths for the IndexManifest
    # so that manifest.py never needs to reach into the private _conn.
    # ------------------------------------------------------------------

    def get_revision_hashes(
        self,
        *,
        revision_id: RevisionId,
        corpus_id: CorpusId,
    ) -> tuple[str, str] | None:
        """Return (content_hash, pipeline_fingerprint) for a revision, or None.

        Args:
            revision_id: Revision to look up.
            corpus_id:   Corpus namespace.

        Returns:
            (content_hash, pipeline_fingerprint) tuple or None if not found.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                """
                SELECT content_hash, pipeline_fingerprint
                FROM revisions
                WHERE revision_id = ? AND corpus_id = ?
                """,
                (str(revision_id), str(corpus_id)),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_revision_hashes failed: {exc}") from exc
        if row is None:
            return None
        return (row["content_hash"], row["pipeline_fingerprint"])

    def list_active_canonical_uris(self, *, corpus_id: CorpusId) -> list[str]:
        """Return all canonical URIs with an active revision in *corpus_id*.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            List of canonical URI strings (unsorted; callers sort if needed).

        Raises:
            BackendError: On I/O failure.
        """
        try:
            rows = self._conn.execute(
                "SELECT canonical_uri FROM active_revision_pointers WHERE corpus_id = ?",
                (str(corpus_id),),
            ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"list_active_canonical_uris failed: {exc}") from exc
        return [row["canonical_uri"] for row in rows]

    def list_active_revision_fingerprints(
        self, *, corpus_id: CorpusId
    ) -> list[tuple[str, str, str]]:
        """Return (revision_id, content_hash, pipeline_fingerprint) for all active revisions.

        Results are ordered by revision_id ascending so callers can produce a
        stable combined fingerprint without sorting.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            List of (revision_id, content_hash, pipeline_fingerprint) tuples.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT r.revision_id, r.content_hash, r.pipeline_fingerprint
                FROM active_revision_pointers arp
                JOIN revisions r ON r.revision_id = arp.revision_id
                WHERE arp.corpus_id = ?
                ORDER BY r.revision_id
                """,
                (str(corpus_id),),
            ).fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"list_active_revision_fingerprints failed: {exc}") from exc
        return [
            (row["revision_id"], row["content_hash"], row["pipeline_fingerprint"])
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Build run tracking
    # ------------------------------------------------------------------

    def create_build_run(
        self,
        *,
        corpus_id: CorpusId,
        pipeline_fingerprint: str,
        started_at_iso: str,
    ) -> str:
        """Create a new build run record and return its ID.

        Args:
            corpus_id:            Corpus namespace.
            pipeline_fingerprint: Hash of the pipeline configuration.
            started_at_iso:       ISO 8601 start time.

        Returns:
            BuildRunId string.

        Raises:
            BackendError: On SQLite write failure.
        """
        run_id = make_build_run_id(
            corpus=str(corpus_id),
            pipeline_fingerprint=pipeline_fingerprint,
            started_at_iso=started_at_iso,
        )
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                INSERT INTO build_runs
                    (build_run_id, corpus_id, pipeline_fingerprint, started_at_iso,
                     status, sources_scanned, sources_changed, chunks_added,
                     chunks_deleted, error_count, errors_json)
                VALUES (?, ?, ?, ?, 'running', 0, 0, 0, 0, 0, '[]')
                ON CONFLICT(build_run_id) DO NOTHING
                """,
                (str(run_id), str(corpus_id), pipeline_fingerprint, started_at_iso),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"create_build_run failed: {exc}") from exc
        return str(run_id)

    def finish_build_run(
        self,
        *,
        build_run_id: str,
        status: str,
        sources_scanned: int = 0,
        sources_changed: int = 0,
        chunks_added: int = 0,
        chunks_deleted: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        """Record the completion of a build run.

        Args:
            build_run_id:    BuildRunId string.
            status:          Final status ('success', 'failed', 'partial').
            sources_scanned: Count of sources examined.
            sources_changed: Count of sources that had changes.
            chunks_added:    Count of new chunks written.
            chunks_deleted:  Count of old chunks retired.
            errors:          List of error message strings (may be empty).

        Raises:
            BackendError: On SQLite write failure.
        """
        errs = errors or []
        try:
            self._conn.execute("BEGIN")
            self._conn.execute(
                """
                UPDATE build_runs SET
                    finished_at_iso  = ?,
                    status           = ?,
                    sources_scanned  = ?,
                    sources_changed  = ?,
                    chunks_added     = ?,
                    chunks_deleted   = ?,
                    error_count      = ?,
                    errors_json      = ?
                WHERE build_run_id = ?
                """,
                (
                    _now_iso(),
                    status,
                    sources_scanned,
                    sources_changed,
                    chunks_added,
                    chunks_deleted,
                    len(errs),
                    json.dumps(errs),
                    build_run_id,
                ),
            )
            self._conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"finish_build_run failed: {exc}") from exc

    def get_build_run(self, *, build_run_id: str) -> dict[str, Any] | None:
        """Retrieve build run metadata by ID, or None if not found.

        Args:
            build_run_id: BuildRunId string.

        Returns:
            Dict of run fields, or None.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                "SELECT * FROM build_runs WHERE build_run_id = ?",
                (build_run_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_build_run failed: {exc}") from exc
        if row is None:
            return None
        return dict(row)

    def get_latest_build_run(self, *, corpus_id: CorpusId) -> dict[str, Any] | None:
        """Return the most recent build run for *corpus_id*, or None if none exist.

        "Most recent" is defined by latest started_at_iso, with rowid as tiebreaker.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            Dict of run fields, or None if no build runs recorded for this corpus.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                """
                SELECT * FROM build_runs
                WHERE corpus_id = ?
                ORDER BY started_at_iso DESC, rowid DESC
                LIMIT 1
                """,
                (str(corpus_id),),
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"get_latest_build_run failed: {exc}") from exc
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Schema metadata
    # ------------------------------------------------------------------

    def count_active_chunks(self, *, corpus_id: CorpusId) -> int:
        """Return the number of active chunks for *corpus_id*.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            Non-negative integer count of active chunks.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE corpus_id = ? AND active = 1",
                (str(corpus_id),),
            ).fetchone()
        except Exception as exc:
            raise BackendError(f"count_active_chunks failed: {exc}") from exc
        return int(row["c"]) if row else 0

    def get_staged_chunks(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> list[Chunk]:
        """Return all staged (active=0) chunks for *revision_id* in *corpus_id*.

        Used by RevisionValidator to inspect staged data before promotion
        without reaching into the private _conn attribute.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            List of Chunk records with active=0 for this revision.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE revision_id = ? AND corpus_id = ? AND active = 0",
                (str(revision_id), str(corpus_id)),
            ).fetchall()
        except Exception as exc:
            raise BackendError(f"get_staged_chunks failed: {exc}") from exc
        return [_row_to_chunk(row) for row in rows]

    def get_staged_embedding_count(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Return the number of staged (active=0) embeddings for *revision_id*.

        Used by RevisionValidator to confirm every staged chunk received an
        embedding before promotion, guarding against a provider miscount that
        would otherwise promote chunks with missing vectors.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            Non-negative integer count of staged embeddings.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM embeddings "
                "WHERE revision_id = ? AND corpus_id = ? AND active = 0",
                (str(revision_id), str(corpus_id)),
            ).fetchone()
        except Exception as exc:
            raise BackendError(f"get_staged_embedding_count failed: {exc}") from exc
        return int(row["c"]) if row else 0

    def get_staged_embeddings(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> list[tuple[str, int]]:
        """Return (chunk_id, dimension) for every staged embedding of *revision_id*.

        Used by RevisionValidator to verify staged embedding dimensions match
        the expected vector dimension before promotion, without decoding the
        vector blobs.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            List of (chunk_id, dimension) tuples for staged (active=0) embeddings.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, dimension FROM embeddings "
                "WHERE revision_id = ? AND corpus_id = ? AND active = 0",
                (str(revision_id), str(corpus_id)),
            ).fetchall()
        except Exception as exc:
            raise BackendError(f"get_staged_embeddings failed: {exc}") from exc
        return [(row["chunk_id"], int(row["dimension"])) for row in rows]

    def retire_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Retire an active revision by setting its chunks inactive and removing its pointer.

        This is the inverse of promote_revision: it removes a source from the
        active search index without deleting the revision record itself.
        Used when a source is deleted from the connector between syncs.

        Steps performed atomically:
        1. Find all active chunks for *revision_id* in *corpus_id*.
        2. Set those chunks to active=0.
        3. Remove their FTS5 rows.
        4. Set embeddings for those chunks to active=0.
        5. Delete the active_revision_pointer for the source.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The active revision to retire.

        Returns:
            The number of active chunks retired (0 if the revision had none).

        Raises:
            BackendError: On SQLite write failure.
        """
        retired_count = 0
        try:
            self._conn.execute("BEGIN")

            # Find all active chunks for this revision.
            chunk_rows = self._conn.execute(
                "SELECT chunk_id FROM chunks"
                " WHERE revision_id = ? AND corpus_id = ? AND active = 1",
                (str(revision_id), str(corpus_id)),
            ).fetchall()
            chunk_ids = [row["chunk_id"] for row in chunk_rows]

            if chunk_ids:
                retired_count = len(chunk_ids)
                placeholders = ",".join("?" * len(chunk_ids))
                self._conn.execute(
                    f"UPDATE chunks SET active = 0 WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )
                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )
                self._conn.execute(
                    f"UPDATE embeddings SET active = 0 WHERE chunk_id IN ({placeholders})",  # noqa: S608
                    chunk_ids,
                )

            # Remove the active_revision_pointer for this source.
            rev_row = self._conn.execute(
                "SELECT source_id FROM revisions WHERE revision_id = ? AND corpus_id = ?",
                (str(revision_id), str(corpus_id)),
            ).fetchone()
            if rev_row:
                src_row = self._conn.execute(
                    "SELECT canonical_uri FROM sources WHERE source_id = ? AND corpus_id = ?",
                    (rev_row["source_id"], str(corpus_id)),
                ).fetchone()
                if src_row:
                    self._conn.execute(
                        "DELETE FROM active_revision_pointers "
                        "WHERE corpus_id = ? AND canonical_uri = ?",
                        (str(corpus_id), src_row["canonical_uri"]),
                    )

            self._conn.execute("COMMIT")
            return retired_count
        except sqlite3.Error as exc:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise BackendError(f"retire_revision failed: {exc}") from exc

    def schema_version(self) -> int:
        """Return the highest applied migration version number.

        Returns:
            Integer version number, or 0 if no migrations recorded.

        Raises:
            BackendError: On I/O failure.
        """
        try:
            row = self._conn.execute(
                "SELECT MAX(version) AS v FROM schema_migrations"
            ).fetchone()
        except sqlite3.Error as exc:
            raise BackendError(f"schema_version query failed: {exc}") from exc
        if row is None or row["v"] is None:
            return 0
        return int(row["v"])


# ---------------------------------------------------------------------------
# Registration note
#
# SQLiteStore is NOT registered as a built-in default (see registry/builtins.py
# for the rationale): a store needs a concrete db_path and vector_dim, so a
# throwaway default instance would be a footgun.  Callers construct a
# SQLiteStore explicitly and, if they want registry resolution, register it via
# precedence.register(group=groups.STORES, name="sqlite", instance=...).
# ---------------------------------------------------------------------------
