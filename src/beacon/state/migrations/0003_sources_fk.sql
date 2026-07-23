-- Migration 0003: Add FOREIGN KEY sources -> collections(name).
--
-- SQLite does not support ADD CONSTRAINT on an existing table.
-- The migration rebuilds the sources table with the FK, copies all existing
-- rows, drops the old table, and renames the new one.
--
-- PRAGMA foreign_keys must be OFF during the rebuild (required by SQLite when
-- renaming tables in the presence of FK relationships). The StateDB connection
-- has PRAGMA foreign_keys=ON, but executescript() issues an implicit COMMIT
-- before running and operates outside the connection's PRAGMA state, so we
-- explicitly disable and re-enable here for safety.

PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS sources_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_name TEXT    NOT NULL,
    canonical_uri   TEXT    NOT NULL,
    connector_kind  TEXT    NOT NULL DEFAULT '',
    content_hash    TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'active',
    media_type      TEXT    NULL     DEFAULT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (collection_name, canonical_uri),
    FOREIGN KEY (collection_name) REFERENCES collections(name)
);

INSERT INTO sources_new
    (id, collection_name, canonical_uri, connector_kind,
     content_hash, status, media_type, created_at, updated_at)
SELECT
    id, collection_name, canonical_uri, connector_kind,
    content_hash, status, NULL, created_at, updated_at
FROM sources;

DROP TABLE sources;

ALTER TABLE sources_new RENAME TO sources;

CREATE INDEX IF NOT EXISTS idx_sources_collection
    ON sources (collection_name, status);

PRAGMA foreign_keys = ON;
