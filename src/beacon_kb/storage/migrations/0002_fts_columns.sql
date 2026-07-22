-- Migration 0002: Extend FTS5 table with separately-weightable heading and code columns.
--
-- Design:
-- - Drop the existing chunks_fts table and recreate with three indexed columns:
--     text     - body text of the chunk
--     heading  - heading path (parent_locator) for heading-boosted retrieval
--     code     - fenced code content for identifier-boosted retrieval
-- - chunk_id and corpus_id remain UNINDEXED (administrative, not searched).
-- - Existing single-column behaviour is preserved: bm25(chunks_fts) still works.
-- - Weighted retrieval uses: bm25(chunks_fts, 0, 0, w_text, w_heading, w_code)
--   where the column order matches the CREATE VIRTUAL TABLE definition.
-- - Epic 03's sparse retriever will adopt the per-column weights after merge.
--
-- Note: FTS5 does not support ALTER TABLE to add columns; the table must be
-- dropped and recreated. All active FTS rows are repopulated by the
-- SQLiteStore._rebuild_fts_from_chunks() method called after this migration runs.
-- The migration itself only recreates the empty table; data backfill happens
-- in the application layer after migration is applied.

DROP TABLE IF EXISTS chunks_fts;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id    UNINDEXED,
    corpus_id   UNINDEXED,
    text,
    heading,
    code,
    tokenize='porter ascii'
);
