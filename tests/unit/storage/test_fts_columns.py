"""Tests for FTS5 multi-column migration 0002 and weighted retrieve()."""

from __future__ import annotations

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
from beacon_kb.storage.sqlite import SQLiteStore, _extract_code_content


def _make_store() -> SQLiteStore:
    return SQLiteStore(db_path=":memory:", vector_dim=4)


def _make_chunk(
    text: str = "hello world",
    parent_locator: str = "intro",
    ordinal: int = 0,
) -> tuple[Chunk, CorpusId]:
    corpus_id = CorpusId("test-corpus")
    source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc")
    chunk_id = make_chunk_id(
        corpus="test-corpus",
        canonical_uri="fake://doc",
        revision_id="rev-001",
        pipeline_fingerprint="pipe-v1",
        parent_locator=parent_locator,
        child_ordinal=ordinal,
    )
    chunk = Chunk(
        id=chunk_id,
        source_id=source_id,
        revision_id=RevisionId("rev-001"),
        section_id=SectionId("sec-001"),
        text=text,
        ordinal=ordinal,
        parent_locator=parent_locator,
        kind=ChunkKind.CHILD,
        token_count=len(text.split()),
    )
    return chunk, corpus_id


def test_extract_code_content_empty() -> None:
    assert _extract_code_content("no code here") == ""


def test_extract_code_content_single_block() -> None:
    text = "Intro.\n```python\ndef foo(): pass\n```\nOutro."
    result = _extract_code_content(text)
    assert "def foo(): pass" in result
    assert "Intro." not in result


def test_extract_code_content_multiple_blocks() -> None:
    text = "A\n```\nblock1\n```\nB\n```\nblock2\n```\nC"
    result = _extract_code_content(text)
    assert "block1" in result
    assert "block2" in result


def test_schema_version_after_migration() -> None:
    store = _make_store()
    # After initialization, migration 0002 should be applied.
    version = store.schema_version()
    assert version >= 2


def test_retrieve_returns_hits_after_upsert() -> None:
    """retrieve() must still work after migration 0002."""
    store = _make_store()
    chunk, _corpus_id = _make_chunk("the quick brown fox")
    store.upsert_chunks([chunk])
    query = Query(id=QueryId("q1"), text="quick fox", corpus_id=None, top_k=5)
    hits = store.retrieve(query)
    assert len(hits) >= 1
    assert hits[0].chunk.id == chunk.id


def test_retrieve_with_explicit_weights() -> None:
    """retrieve() accepts optional weights tuple (text, heading, code)."""
    store = _make_store()
    chunk, _ = _make_chunk("documentation content here")
    store.upsert_chunks([chunk])
    query = Query(id=QueryId("q1"), text="documentation", corpus_id=None, top_k=5)
    # Should not raise; weights are a no-op if BM25 column weighting not supported.
    hits = store.retrieve(query, weights=(1.0, 0.5, 2.0))
    assert isinstance(hits, list)


def test_code_column_populated_from_fenced_blocks() -> None:
    """Chunks with fenced code blocks must populate the code FTS column."""
    store = _make_store()
    text = "Preamble.\n```python\ndef bar(): return 1\n```\nPostamble."
    chunk, _ = _make_chunk(text=text, parent_locator="api/reference")
    store.upsert_chunks([chunk])
    # Query for a term from inside the code block.
    query = Query(id=QueryId("q2"), text="def bar", corpus_id=None, top_k=5)
    hits = store.retrieve(query)
    assert any(h.chunk.id == chunk.id for h in hits)


def test_heading_column_populated_from_parent_locator() -> None:
    """parent_locator content must appear in the heading FTS column."""
    store = _make_store()
    chunk, _ = _make_chunk(text="body text only", parent_locator="installation/quickstart")
    store.upsert_chunks([chunk])
    query = Query(id=QueryId("q3"), text="installation quickstart", corpus_id=None, top_k=5)
    hits = store.retrieve(query)
    assert any(h.chunk.id == chunk.id for h in hits)


def test_old_database_upgraded_with_fts_rows_preserved(tmp_path: object) -> None:
    """A pre-0002 database must be migrated and its FTS index rebuilt.

    Simulates an existing installation: apply only migration 0001, insert an
    active chunk with an old-style single-text-column FTS row, then open
    SQLiteStore (which applies 0002 and rebuilds FTS) and verify the chunk is
    still searchable - including via the new heading column.
    """
    import pathlib
    import sqlite3

    db_path = str(pathlib.Path(str(tmp_path)) / "old.db")
    # Locate migration 0001 relative to the sqlite module.
    import beacon_kb.storage.sqlite as sqlite_mod

    migrations = pathlib.Path(sqlite_mod.__file__).parent / "migrations"
    sql_0001 = (migrations / "0001_initial.sql").read_text(encoding="utf-8")

    conn = sqlite3.connect(db_path)
    conn.executescript(sql_0001)
    conn.execute(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (1, '2026-01-01T00:00:00Z')"
    )
    conn.execute(
        """
        INSERT INTO chunks
            (chunk_id, corpus_id, source_id, revision_id, section_id, text,
             ordinal, parent_locator, kind, token_count, active, created_at)
        VALUES
            ('old-chunk-1', 'old-corpus', 'src-1', 'rev-1', 'sec-1',
             'legacy searchable content', 0, 'legacy/heading', 'child', 3, 1,
             '2026-01-01T00:00:00Z')
        """
    )
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, corpus_id, text) "
        "VALUES ('old-chunk-1', 'old-corpus', 'legacy searchable content')"
    )
    conn.commit()
    conn.close()

    # Opening the store applies migration 0002 and rebuilds the FTS index.
    store = SQLiteStore(db_path=db_path, vector_dim=4)
    assert store.schema_version() >= 2

    hits = store.retrieve(
        Query(id=QueryId("q-old"), text="legacy searchable", corpus_id=None, top_k=5)
    )
    assert any(str(h.chunk.id) == "old-chunk-1" for h in hits), (
        "Old chunk must remain searchable after the FTS rebuild."
    )
    # The rebuilt row must also carry the heading column from parent_locator.
    heading_hits = store.retrieve(
        Query(id=QueryId("q-old-h"), text="legacy heading", corpus_id=None, top_k=5)
    )
    assert any(str(h.chunk.id) == "old-chunk-1" for h in heading_hits)
    store.close()
