"""Contract-test suite for the knowledge store (Store protocol).

This module verifies that the SQLiteStore satisfies the Store protocol
defined in beacon_kb.protocols.  It uses the StoreContract base class
imported from beacon_kb.testing, consistent with all other contract suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beacon_kb.protocols import Store
from beacon_kb.testing import StoreContract

# ---------------------------------------------------------------------------
# SQLiteStore implementation of StoreContract
# ---------------------------------------------------------------------------


class TestSQLiteStoreContract(StoreContract):
    """Bind StoreContract to SQLiteStore for pytest discovery."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        self._store = SQLiteStore(db_path=str(tmp_path / "contract.db"), vector_dim=16)

    def make_subject(self) -> Store:
        return self._store  # type: ignore[return-value]
