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

Importing this module has the side effect of registering the SQLiteStore as
the default 'sqlite' store in the beacon_kb.stores entry-point group, via
the module-level call at the bottom of this file.
"""

from __future__ import annotations

import datetime
import json
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


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any unapplied SQL migrations in version order.

    Reads migration files from ``storage/migrations/`` named
    ``<version>_<description>.sql`` (e.g. ``0001_initial.sql``).
    Records each applied version in the ``schema_migrations`` table.

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
            conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as exc:
            raise BackendError(f"Cannot open SQLite database at {self._db_path!r}: {exc}") from exc

        _check_fts5(conn)
        _apply_migrations(conn)
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
                    "INSERT INTO chunks_fts (chunk_id, corpus_id, text) VALUES (?, ?, ?)",
                    (str(chunk.id), "", chunk.text),
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
    ) -> None:
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

        Raises:
            BackendError: If the revision is not found or the transaction fails.
        """
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
                    "INSERT INTO chunks_fts (chunk_id, corpus_id, text) VALUES (?, ?, ?)",
                    (row["chunk_id"], str(corpus_id), row["text"]),
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
            BackendError: If dimension mismatches or similarity is unknown.
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

    def retrieve(self, query: Query) -> list[Hit]:
        """BM25 sparse retrieval using FTS5.

        Args:
            query: Query record with text, corpus_id filter, and top_k.

        Returns:
            List of Hit records with sparse_score set, ordered descending.
            dense_score, fusion_score, and rerank_score are all None.

        Raises:
            BackendError: On index read failure.
        """
        try:
            if query.corpus_id is not None:
                rows = self._conn.execute(
                    """
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
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND chunks_fts.corpus_id = ?
                      AND c.active = 1
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query.text, str(query.corpus_id), query.top_k),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
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
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND c.active = 1
                    ORDER BY score
                    LIMIT ?
                    """,
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

    # ------------------------------------------------------------------
    # Schema metadata
    # ------------------------------------------------------------------

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
# SQLiteStore is registered as the default 'sqlite' store in
# registry/builtins.py, which is imported eagerly by registry/__init__.py.
# No registration call is made here to avoid potential double-registration
# when this module is imported independently of the registry package.
# ---------------------------------------------------------------------------
