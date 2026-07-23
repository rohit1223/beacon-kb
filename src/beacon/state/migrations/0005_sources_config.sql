-- Migration 0005: Add config_json and is_connector_definition to sources.
--
-- config_json stores the connector config dict as JSON for connector-definition
-- rows (rows inserted by POST /collections/{name}/sources).  NULL for content
-- sources discovered by the sync engine.
--
-- is_connector_definition is 1 for rows inserted by attach_source and 0 for all
-- other source rows.  The planner and the sync engine's retire pass use this
-- flag to skip definition rows during content classification so a definition
-- row is never retired as a "vanished" source, and the sync trigger route uses
-- it to look up which connector to instantiate for a collection.

ALTER TABLE sources ADD COLUMN config_json TEXT NULL;
ALTER TABLE sources ADD COLUMN is_connector_definition INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_sources_definition
    ON sources (collection_name, is_connector_definition);
