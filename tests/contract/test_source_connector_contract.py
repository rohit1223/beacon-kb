"""Contract tests binding ConnectorContract to FilesystemConnector and MemoryConnector.

Both connectors must pass the reusable ConnectorContract suite imported
from beacon_kb.testing.
"""

from __future__ import annotations

import pathlib

import pytest

from beacon_kb.protocols import Connector
from beacon_kb.testing import ConnectorContract

# ---------------------------------------------------------------------------
# FilesystemConnector contract
# ---------------------------------------------------------------------------


class TestFilesystemConnectorContract(ConnectorContract):
    """Bind ConnectorContract to FilesystemConnector."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: pathlib.Path) -> None:
        from beacon_kb.connectors.filesystem import FilesystemConnector

        (tmp_path / "doc1.md").write_text("# Document One\nContent one.")
        (tmp_path / "doc2.md").write_text("# Document Two\nContent two.")

        self._connector = FilesystemConnector(
            root=tmp_path,
            corpus="contract-test",
            patterns=["*.md"],
        )

    def make_subject(self) -> Connector:
        return self._connector  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MemoryConnector contract
# ---------------------------------------------------------------------------


class TestMemoryConnectorContract(ConnectorContract):
    """Bind ConnectorContract to MemoryConnector."""

    def make_subject(self) -> Connector:
        from beacon_kb.connectors.memory import MemoryConnector

        return MemoryConnector(  # type: ignore[return-value]
            corpus="contract-test",
            sources={
                "memory://contract-doc-1": "Contract document one content.",
                "memory://contract-doc-2": "Contract document two content.",
            },
        )
