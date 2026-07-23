"""SQLite state database for the Beacon server.

Opens the database with WAL journal mode and foreign-key enforcement, applies
numbered SQL migrations from the ``migrations/`` subdirectory in version order,
and exposes a thin typed connection helper.

Threading: this class owns exactly ONE SQLite connection and is single-threaded.
The connection is opened without ``check_same_thread=False``, so using an
instance from a thread other than the one that created it raises immediately
rather than risking silent corruption.
A pooled/locked variant is roadmapped in ROADMAP.md.

Importing this module performs no side effects.
This module has no dependency on Qdrant or FastAPI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from beacon.errors import BackendError
from beacon.state._util import _now_iso

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIGRATION_DIR = Path(__file__).parent / "migrations"


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any unapplied SQL migrations from the migrations directory.

    Migration files must be named ``<NNNN>_<description>.sql`` (e.g.
    ``0001_initial.sql``).  The numeric prefix determines the application
    order.  Each migration is applied in its own transaction and recorded in
    the ``schema_migrations`` table.  Re-running is idempotent: already-
    applied versions are skipped.

    Args:
        conn: Open SQLite connection.

    Raises:
        BackendError: If a migration file cannot be read or the SQL fails.
    """
    # Bootstrap the tracking table before querying it.
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
        row[0]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    migration_files = sorted(_MIGRATION_DIR.glob("*.sql"))
    for mf in migration_files:
        try:
            version = int(mf.stem.split("_")[0])
        except ValueError:
            continue

        if version in applied:
            continue

        try:
            sql = mf.read_text(encoding="utf-8")
        except OSError as exc:
            raise BackendError(f"Cannot read migration file {mf}: {exc}") from exc

        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, _now_iso()),
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise BackendError(f"Migration {version} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# StateDB
# ---------------------------------------------------------------------------


class StateDB:
    """Thin wrapper around a single SQLite connection with migrations applied.

    Instantiate once per process (or per thread); pass the instance to the
    repository classes in ``beacon.state.repo``.

    Args:
        db_path: Path to the SQLite file.  Created if absent; parent directories
                 must already exist.

    Raises:
        BackendError: If the database cannot be opened or a migration fails.
    """

    def __init__(self, *, db_path: str) -> None:
        self._db_path = db_path
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        """Open the database, configure PRAGMAs, and apply migrations.

        Returns:
            Open SQLite connection with row_factory set to sqlite3.Row.

        Raises:
            BackendError: On open or migration failure.
        """
        try:
            # check_same_thread stays at the default (True): single-threaded
            # semantics documented on the class.  A pooled variant is roadmapped.
            conn = sqlite3.connect(self._db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as exc:
            raise BackendError(
                f"Cannot open SQLite state database at {self._db_path!r}: {exc}"
            ) from exc

        _apply_migrations(conn)
        return conn

    def connection(self) -> sqlite3.Connection:
        """Return the underlying SQLite connection for direct use by repositories.

        Callers must not close this connection themselves; call ``StateDB.close()``
        instead.

        Returns:
            The open sqlite3.Connection.
        """
        return self._conn

    def schema_version(self) -> int:
        """Return the highest applied migration version number.

        Returns:
            Integer version, or 0 if no migrations have been recorded.

        Raises:
            BackendError: On SQLite read failure.
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

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
