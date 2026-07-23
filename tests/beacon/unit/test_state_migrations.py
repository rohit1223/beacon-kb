"""Unit tests for the SQLite state DB migration runner.

Verifies migration application, idempotency, schema shape, and that opening
a fresh database applies all migrations while reopening applies none and
preserves data.
"""

from __future__ import annotations

from typing import Any

import pytest

from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open(tmp_path: Any, name: str = "state.db") -> StateDB:
    return StateDB(db_path=str(tmp_path / name))


# ---------------------------------------------------------------------------
# Migration application and idempotency
# ---------------------------------------------------------------------------


def test_fresh_db_applies_migrations(tmp_path: Any) -> None:
    """Opening a fresh database applies all migrations and records them."""
    db = _open(tmp_path)
    version = db.schema_version()
    db.close()
    assert version >= 1, f"Expected at least schema version 1, got {version}"


def test_reopen_applies_no_new_migrations(tmp_path: Any) -> None:
    """Reopening an already-migrated database applies no additional migrations."""
    db1 = _open(tmp_path)
    v1 = db1.schema_version()
    db1.close()

    db2 = _open(tmp_path)
    v2 = db2.schema_version()
    db2.close()

    assert v1 == v2, f"Schema version changed on reopen: {v1} -> {v2}"


def test_reopen_preserves_data(tmp_path: Any) -> None:
    """Reopening a migrated database preserves previously written data."""
    # Session 1: use the repo API to write data.
    db1 = _open(tmp_path)
    repo1 = CollectionRepo(db1)
    repo1.create(name="my-col", settings={"key": "value"})
    db1.close()

    # Session 2: reopen and verify the data persists.
    db2 = _open(tmp_path)
    repo2 = CollectionRepo(db2)
    row = repo2.get("my-col")
    db2.close()

    assert row is not None, "Data written in first session must survive restart"
    assert row["name"] == "my-col"


def test_migration_is_idempotent_on_repeated_open(tmp_path: Any) -> None:
    """Opening the same database five times never raises and version stays stable."""
    versions = []
    for _ in range(5):
        db = _open(tmp_path)
        versions.append(db.schema_version())
        db.close()
    assert len(set(versions)) == 1, f"Schema version must be stable; got {versions}"


# ---------------------------------------------------------------------------
# Schema shape - required tables exist
# ---------------------------------------------------------------------------


def _table_names(db: StateDB) -> set[str]:
    rows = db.connection().execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row["name"] for row in rows}


def test_schema_has_required_tables(tmp_path: Any) -> None:
    """All required tables are created by the initial migration."""
    db = _open(tmp_path)
    tables = _table_names(db)
    db.close()
    required = {"schema_migrations", "collections", "sources", "revisions", "sync_jobs"}
    missing = required - tables
    assert not missing, f"Missing tables after migration: {missing}"


def test_collections_table_columns(tmp_path: Any) -> None:
    """collections table has expected columns."""
    db = _open(tmp_path)
    info = db.connection().execute("PRAGMA table_info(collections)").fetchall()
    db.close()
    cols = {row["name"] for row in info}
    assert {"name", "created_at", "settings_json"}.issubset(cols)


def test_sources_table_columns(tmp_path: Any) -> None:
    """sources table has expected columns including media_type added in 0003."""
    db = _open(tmp_path)
    info = db.connection().execute("PRAGMA table_info(sources)").fetchall()
    db.close()
    cols = {row["name"] for row in info}
    expected = {
        "canonical_uri", "connector_kind", "content_hash",
        "status", "collection_name", "media_type",
    }
    assert expected.issubset(cols)


def test_revisions_table_columns(tmp_path: Any) -> None:
    """revisions table has expected columns including fingerprint and status."""
    db = _open(tmp_path)
    info = db.connection().execute("PRAGMA table_info(revisions)").fetchall()
    db.close()
    cols = {row["name"] for row in info}
    assert {"revision_id", "collection_name", "fingerprint", "status", "chunk_count"}.issubset(
        cols
    )


def test_sync_jobs_table_columns(tmp_path: Any) -> None:
    """sync_jobs table has expected columns including state, error_detail, timestamps."""
    db = _open(tmp_path)
    info = db.connection().execute("PRAGMA table_info(sync_jobs)").fetchall()
    db.close()
    cols = {row["name"] for row in info}
    assert {
        "job_id",
        "collection_name",
        "state",
        "error_detail",
        "started_at",
        "finished_at",
        "sources_added",
        "sources_removed",
        "sources_unchanged",
    }.issubset(cols)


# ---------------------------------------------------------------------------
# WAL mode and foreign keys
# ---------------------------------------------------------------------------


def test_wal_mode_enabled(tmp_path: Any) -> None:
    """The database must run in WAL journal mode."""
    db = _open(tmp_path)
    row = db.connection().execute("PRAGMA journal_mode").fetchone()
    db.close()
    assert row[0] == "wal", f"Expected WAL mode, got {row[0]!r}"


def test_foreign_keys_on(tmp_path: Any) -> None:
    """Foreign key enforcement must be ON."""
    db = _open(tmp_path)
    row = db.connection().execute("PRAGMA foreign_keys").fetchone()
    db.close()
    assert row[0] == 1, "Foreign keys must be enabled"


# ---------------------------------------------------------------------------
# schema_migrations tracking
# ---------------------------------------------------------------------------


def test_schema_migrations_records_version(tmp_path: Any) -> None:
    """schema_migrations table records at least one version row after migration."""
    db = _open(tmp_path)
    rows = db.connection().execute("SELECT version FROM schema_migrations").fetchall()
    db.close()
    assert rows, "schema_migrations must contain at least one row after migration"
    versions = [r["version"] for r in rows]
    assert 1 in versions, f"Version 1 must be recorded; found {versions}"


def test_no_qdrant_import(tmp_path: Any) -> None:
    """The state module must not import qdrant_client."""
    import ast
    import inspect

    import beacon.state.db

    src = inspect.getsource(beacon.state.db)
    tree = ast.parse(src)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    qdrant_imports = [imp for imp in imports if "qdrant" in imp.lower()]
    assert not qdrant_imports, f"state.db must not import qdrant; found: {qdrant_imports}"


def test_no_fastapi_import(tmp_path: Any) -> None:
    """The state module must not import fastapi."""
    import ast
    import inspect

    import beacon.state.db

    src = inspect.getsource(beacon.state.db)
    tree = ast.parse(src)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    fastapi_imports = [imp for imp in imports if "fastapi" in imp.lower()]
    assert not fastapi_imports, f"state.db must not import fastapi; found: {fastapi_imports}"


# ---------------------------------------------------------------------------
# Migration 0003: sources FK to collections
# ---------------------------------------------------------------------------


class TestMigration0003:
    """Migration 0003 adds FK from sources to collections."""

    def test_schema_version_reaches_3(self, tmp_path: Any) -> None:
        """After fresh open, schema version must be at least 3."""
        db = StateDB(db_path=str(tmp_path / "m3.db"))
        assert db.schema_version() == 3
        db.close()

    def test_fk_enforced_on_insert(self, tmp_path: Any) -> None:
        """Inserting a source with nonexistent collection_name must fail."""
        import sqlite3
        db = StateDB(db_path=str(tmp_path / "fk.db"))
        conn = db.connection()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO sources (collection_name, canonical_uri) VALUES (?, ?)",
                ("ghost_collection", "file:///ghost.md"),
            )
        db.close()

    def test_existing_sources_survive_migration(self, tmp_path: Any) -> None:
        """Rows in sources before 0003 survive migration and gain a NULL media_type column."""
        import pathlib
        import sqlite3 as _sqlite3

        db_path = str(tmp_path / "pre0003.db")

        # Build a 0002-era DB via raw sqlite3, applying only migrations 0001 and 0002.
        migrations_dir = (
            pathlib.Path(__file__).parents[3]
            / "src" / "beacon" / "state" / "migrations"
        )
        sql_0001 = (migrations_dir / "0001_initial.sql").read_text()
        sql_0002 = (migrations_dir / "0002_single_live_index.sql").read_text()

        raw_conn = _sqlite3.connect(db_path)
        raw_conn.row_factory = _sqlite3.Row
        raw_conn.execute("PRAGMA journal_mode=WAL")
        raw_conn.execute("PRAGMA foreign_keys=ON")
        # Bootstrap schema_migrations table and apply 0001 + 0002.
        raw_conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        raw_conn.executescript(sql_0001)
        raw_conn.executescript(sql_0002)
        raw_conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (1, "2000-01-01T00:00:00.000Z"),
        )
        raw_conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (2, "2000-01-01T00:00:00.000Z"),
        )

        # Insert a collections row (needed for the future FK) and a sources row.
        raw_conn.execute(
            "INSERT INTO collections (name) VALUES (?)", ("legacy-col",)
        )
        raw_conn.execute(
            "INSERT INTO sources (collection_name, canonical_uri, connector_kind) VALUES (?, ?, ?)",
            ("legacy-col", "file:///legacy.md", "folder"),
        )
        raw_conn.commit()
        raw_conn.close()

        # Open with StateDB - this triggers migration 0003 which adds media_type.
        db = StateDB(db_path=db_path)
        conn = db.connection()
        row = conn.execute(
            "SELECT * FROM sources WHERE canonical_uri = ?", ("file:///legacy.md",)
        ).fetchone()
        assert row is not None, "Legacy source row must survive migration 0003"
        assert row["collection_name"] == "legacy-col"
        assert row["connector_kind"] == "folder"
        # media_type must be present as a column and NULL for legacy rows.
        assert row["media_type"] is None, "Legacy rows must have NULL media_type after migration"
        db.close()
