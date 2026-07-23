-- Migration 0002: Partial unique index to enforce at most one live revision per collection.
--
-- This is a DB-level backstop for the application-layer invariant enforced
-- in RevisionRepo.set_live. The partial index allows multiple non-live
-- revisions (staged/failed/retired) per collection but prevents more than one
-- live revision from existing simultaneously.

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_live_per_collection
    ON revisions(collection_name)
    WHERE status = 'live';
