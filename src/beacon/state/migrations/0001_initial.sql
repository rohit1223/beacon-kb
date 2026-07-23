-- Migration 0001: Initial schema for the Beacon server SQLite state DB.
--
-- Design notes:
-- - Chunk data lives in Qdrant; this DB holds only bookkeeping rows.
-- - schema_migrations bootstrapped by the Python runner before this file runs;
--   the CREATE TABLE IF NOT EXISTS here is kept for safety.
-- - All timestamps are UTC ISO 8601 strings.
-- - Foreign keys are enabled at connection time via PRAGMA foreign_keys=ON.

-- ---------------------------------------------------------------------------
-- Schema version tracking
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- Collections (logical corpus namespaces)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS collections (
    name            TEXT    NOT NULL PRIMARY KEY,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    settings_json   TEXT    NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- Sources (canonical URIs within a collection)
--
-- status: 'active' | 'retired'
-- content_hash: SHA-256 of the last-seen raw content, used for dedupe.
-- connector_kind: e.g. 'folder', 'upload', 'web'.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_name TEXT    NOT NULL,
    canonical_uri   TEXT    NOT NULL,
    connector_kind  TEXT    NOT NULL DEFAULT '',
    content_hash    TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'active',
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (collection_name, canonical_uri)
);

CREATE INDEX IF NOT EXISTS idx_sources_collection
    ON sources (collection_name, status);

-- ---------------------------------------------------------------------------
-- Revisions (per-collection staging/live/failed/retired records)
--
-- Exactly one revision per collection can have status = 'live'.
-- The Python layer enforces this invariant atomically.
--
-- status: 'staged' | 'live' | 'failed' | 'retired'
-- fingerprint: combined hash of pipeline configuration and content.
-- chunk_count: number of chunks produced (informational; 0 until set_live).
-- source_count: number of sources included in this revision.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS revisions (
    revision_id     TEXT    NOT NULL PRIMARY KEY,
    collection_name TEXT    NOT NULL,
    fingerprint     TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'staged',
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    source_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_revisions_collection
    ON revisions (collection_name, status);

-- ---------------------------------------------------------------------------
-- Sync jobs (incremental sync job history)
--
-- state: 'pending' | 'running' | 'succeeded' | 'failed'
-- error_detail: JSON problem-details payload on failure, NULL otherwise.
-- sources_added/removed/unchanged: change-plan summary counts.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sync_jobs (
    job_id              TEXT    NOT NULL PRIMARY KEY,
    collection_name     TEXT    NOT NULL,
    state               TEXT    NOT NULL DEFAULT 'pending',
    sources_added       INTEGER NOT NULL DEFAULT 0,
    sources_removed     INTEGER NOT NULL DEFAULT 0,
    sources_unchanged   INTEGER NOT NULL DEFAULT 0,
    error_detail        TEXT,
    started_at          TEXT,
    finished_at         TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sync_jobs_collection
    ON sync_jobs (collection_name, created_at);
