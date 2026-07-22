"""Regression test for the migration 0002 crash-window bug.

Migration 0002 drops/recreates chunks_fts and commits its version row in a
transaction separate from the application-layer FTS rebuild.  If the process
crashed between the two, the database would be left at version=2 with a
permanently empty chunks_fts.  The store now decides the rebuild from DURABLE
state at every open (active chunks present AND chunks_fts empty -> rebuild), so
reopening self-heals the crash window.
"""

from __future__ import annotations

from pathlib import Path

from beacon_kb.models import (
    Chunk,
    ChunkKind,
    CorpusId,
    Query,
    QueryId,
    RevisionId,
    SectionId,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.storage.sqlite import SQLiteStore


def _make_chunk() -> tuple[Chunk, CorpusId]:
    corpus_id = CorpusId("crash-corpus")
    source_id = make_source_id(corpus="crash-corpus", canonical_uri="fake://doc")
    chunk_id = make_chunk_id(
        corpus="crash-corpus",
        canonical_uri="fake://doc",
        revision_id="rev-001",
        pipeline_fingerprint="pipe-v1",
        parent_locator="intro",
        child_ordinal=0,
    )
    chunk = Chunk(
        id=chunk_id,
        source_id=source_id,
        revision_id=RevisionId("rev-001"),
        section_id=SectionId("sec-001"),
        text="hello world from the crash window test",
        ordinal=0,
        parent_locator="intro",
        kind=ChunkKind.CHILD,
        token_count=6,
    )
    return chunk, corpus_id


def test_reopen_rebuilds_fts_after_crash_window(tmp_path: Path) -> None:
    """version=2, empty chunks_fts, active chunks present -> reopen rebuilds FTS."""
    db_path = str(tmp_path / "crash.db")

    # Open once and index an active chunk (migrations reach v2).
    store = SQLiteStore(db_path=db_path, vector_dim=4)
    chunk, _corpus_id = _make_chunk()
    store.upsert_chunks([chunk])
    assert store.schema_version() >= 2

    query = Query(id=QueryId("q1"), text="hello", corpus_id=None, top_k=5)
    assert len(store.retrieve(query)) == 1, "Chunk must be searchable before the simulated crash."

    # Simulate the crash window: the version row is durably at 2 but chunks_fts
    # is empty while an active chunk still exists.  Access the connection through
    # a helper that also serves as the durable-state fixture for this scenario.
    conn = store._conn
    conn.execute("DELETE FROM chunks_fts")
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE active = 1").fetchone()[0] == 1
    store.close()

    # Reopen: the store must self-heal by rebuilding the FTS index from durable
    # state, NOT from "migration 0002 was freshly applied this call".
    reopened = SQLiteStore(db_path=db_path, vector_dim=4)
    assert reopened.schema_version() >= 2
    rebuilt_hits = reopened.retrieve(query)
    assert len(rebuilt_hits) == 1, (
        "Reopening must rebuild chunks_fts from the active chunks so the crash "
        "window self-heals instead of leaving a permanently empty FTS index."
    )
    reopened.close()


def test_reopen_does_not_rebuild_when_fts_already_populated(tmp_path: Path) -> None:
    """A populated FTS index is not needlessly rebuilt on reopen (control)."""
    db_path = str(tmp_path / "ok.db")
    store = SQLiteStore(db_path=db_path, vector_dim=4)
    chunk, _corpus_id = _make_chunk()
    store.upsert_chunks([chunk])
    store.close()

    reopened = SQLiteStore(db_path=db_path, vector_dim=4)
    query = Query(id=QueryId("q1"), text="hello", corpus_id=None, top_k=5)
    assert len(reopened.retrieve(query)) == 1
    reopened.close()
