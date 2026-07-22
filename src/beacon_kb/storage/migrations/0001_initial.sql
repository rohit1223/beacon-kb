-- Migration 0001: Initial schema for beacon-kb transactional SQLite store.
--
-- Design principles:
-- - One database holds corpora, revisions, chunks, FTS5 BM25 rows, embedding
--   rows, build runs, fingerprints, and active-revision pointers.
-- - Corpus namespace isolation: corpus_id is a first-class column on every
--   content table so two corpora with identical source paths never share rows.
-- - Staged promotion model: staged rows are invisible until a single
--   promotion transaction flips active_revision_pointers.
-- - Embeddings live IN SQLite - no separate JSON or vector files.
-- - Manifest state persists in the database - no standalone JSON manifest.

-- ---------------------------------------------------------------------------
-- Schema version tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- Corpora
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS corpora (
    corpus_id   TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- Sources (canonical URIs within a corpus)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT    NOT NULL,
    corpus_id       TEXT    NOT NULL,
    canonical_uri   TEXT    NOT NULL,
    media_type      TEXT    NOT NULL DEFAULT 'text/plain',
    title           TEXT    NOT NULL DEFAULT '',
    extra_json      TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (source_id),
    UNIQUE (corpus_id, canonical_uri)
);

CREATE INDEX IF NOT EXISTS idx_sources_corpus ON sources (corpus_id);

-- ---------------------------------------------------------------------------
-- Revisions (content-hash + pipeline-fingerprint versioning)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS revisions (
    revision_id             TEXT    NOT NULL PRIMARY KEY,
    source_id               TEXT    NOT NULL,
    corpus_id               TEXT    NOT NULL,
    content_hash            TEXT    NOT NULL,
    pipeline_fingerprint    TEXT    NOT NULL,
    byte_size               INTEGER NOT NULL DEFAULT 0,
    fetched_at_iso          TEXT    NOT NULL DEFAULT '',
    staged                  INTEGER NOT NULL DEFAULT 1,  -- 1=staged, 0=retired
    created_at              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_revisions_source ON revisions (source_id, corpus_id);
CREATE INDEX IF NOT EXISTS idx_revisions_corpus ON revisions (corpus_id);

-- ---------------------------------------------------------------------------
-- Active revision pointers (one per corpus + canonical_uri)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS active_revision_pointers (
    corpus_id       TEXT    NOT NULL,
    canonical_uri   TEXT    NOT NULL,
    revision_id     TEXT    NOT NULL,
    promoted_at     TEXT    NOT NULL,
    PRIMARY KEY (corpus_id, canonical_uri)
);

-- ---------------------------------------------------------------------------
-- Chunks (retrieval units)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT    NOT NULL PRIMARY KEY,
    corpus_id           TEXT    NOT NULL,
    source_id           TEXT    NOT NULL,
    revision_id         TEXT    NOT NULL,
    section_id          TEXT    NOT NULL,
    text                TEXT    NOT NULL,
    ordinal             INTEGER NOT NULL DEFAULT 0,
    parent_locator      TEXT    NOT NULL DEFAULT '',
    kind                TEXT    NOT NULL DEFAULT 'child',
    token_count         INTEGER NOT NULL DEFAULT 0,
    prev_chunk_id       TEXT,
    next_chunk_id       TEXT,
    active              INTEGER NOT NULL DEFAULT 0,  -- 0=staged, 1=active/visible
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_revision   ON chunks (revision_id, corpus_id);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus_active ON chunks (corpus_id, active);
CREATE INDEX IF NOT EXISTS idx_chunks_source     ON chunks (source_id, corpus_id);

-- ---------------------------------------------------------------------------
-- FTS5 virtual table for BM25 sparse retrieval
-- ---------------------------------------------------------------------------

-- FTS5 table mirrors the active chunks.  Rows are inserted/deleted here
-- atomically together with chunk.active flag updates in the promotion
-- transaction.  This is a standalone FTS5 table (not content-referenced) so
-- that manual INSERT/DELETE in the promotion transaction work correctly.
-- content= and content_rowid= are intentionally absent.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id    UNINDEXED,
    corpus_id   UNINDEXED,
    text,
    tokenize='porter ascii'
);

-- ---------------------------------------------------------------------------
-- Embeddings (dense vector rows stored inside SQLite as BLOBs)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id        TEXT    NOT NULL,
    corpus_id       TEXT    NOT NULL,
    revision_id     TEXT    NOT NULL,
    model_name      TEXT    NOT NULL,
    dimension       INTEGER NOT NULL,
    similarity      TEXT    NOT NULL,  -- 'cosine', 'dot', 'euclidean'
    vector_blob     BLOB    NOT NULL,  -- numpy float32 array serialised with tobytes()
    active          INTEGER NOT NULL DEFAULT 0,  -- mirrors chunk visibility
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_corpus_active ON embeddings (corpus_id, active);
CREATE INDEX IF NOT EXISTS idx_embeddings_revision ON embeddings (revision_id, corpus_id);

-- ---------------------------------------------------------------------------
-- Build runs (incremental sync history)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS build_runs (
    build_run_id        TEXT    NOT NULL PRIMARY KEY,
    corpus_id           TEXT    NOT NULL,
    pipeline_fingerprint TEXT   NOT NULL,
    started_at_iso      TEXT    NOT NULL,
    finished_at_iso     TEXT,
    status              TEXT    NOT NULL DEFAULT 'running',
    sources_scanned     INTEGER NOT NULL DEFAULT 0,
    sources_changed     INTEGER NOT NULL DEFAULT 0,
    chunks_added        INTEGER NOT NULL DEFAULT 0,
    chunks_deleted      INTEGER NOT NULL DEFAULT 0,
    error_count         INTEGER NOT NULL DEFAULT 0,
    errors_json         TEXT    NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_build_runs_corpus ON build_runs (corpus_id, started_at_iso);
