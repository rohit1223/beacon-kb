"""Tests for SourceRepo config_json and is_connector_definition support.

Migration 0005 adds a ``config_json`` column (JSON connector config for
definition rows) and an ``is_connector_definition`` flag so that connector
definitions attached via POST /collections/{name}/sources are distinguishable
from content sources discovered by the sync engine.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SourceRepo


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[StateDB]:
    d = StateDB(db_path=str(tmp_path / "test.db"))
    CollectionRepo(d).create(name="col1")
    yield d
    d.close()


def test_upsert_with_config_json(db: StateDB) -> None:
    """Upsert stores config_json and the definition flag."""
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="folder://cfg1",
        connector_kind="folder",
        config_json='{"root": "/tmp/docs"}',
        is_connector_definition=True,
    )
    row = SourceRepo(db).get(collection_name="col1", canonical_uri="folder://cfg1")
    assert row is not None
    assert row["config_json"] == '{"root": "/tmp/docs"}'
    assert row["is_connector_definition"] == 1


def test_upsert_defaults_to_content_source(db: StateDB) -> None:
    """Default upsert produces a content-source row (no config, flag 0)."""
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="file:///tmp/doc.md",
        connector_kind="folder",
        content_hash="abc",
    )
    row = SourceRepo(db).get(collection_name="col1", canonical_uri="file:///tmp/doc.md")
    assert row is not None
    assert row["config_json"] is None
    assert row["is_connector_definition"] == 0


def test_list_active_excludes_definitions_by_default(db: StateDB) -> None:
    """list_active hides connector-definition rows unless asked otherwise."""
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="folder://cfg1",
        connector_kind="folder",
        config_json='{"root": "/tmp"}',
        is_connector_definition=True,
    )
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="file:///tmp/doc.md",
        connector_kind="folder",
    )
    rows = SourceRepo(db).list_active(collection_name="col1")
    uris = [r["canonical_uri"] for r in rows]
    assert "folder://cfg1" not in uris
    assert "file:///tmp/doc.md" in uris

    all_rows = SourceRepo(db).list_active(
        collection_name="col1", exclude_definitions=False
    )
    all_uris = [r["canonical_uri"] for r in all_rows]
    assert "folder://cfg1" in all_uris


def test_list_connector_definitions(db: StateDB) -> None:
    """list_connector_definitions returns only definition rows."""
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="folder://cfg1",
        connector_kind="folder",
        config_json='{"root": "/tmp"}',
        is_connector_definition=True,
    )
    SourceRepo(db).upsert(
        collection_name="col1",
        canonical_uri="file:///tmp/doc.md",
        connector_kind="folder",
    )
    defs = SourceRepo(db).list_connector_definitions(collection_name="col1")
    assert len(defs) == 1
    assert defs[0]["canonical_uri"] == "folder://cfg1"
    assert defs[0]["config_json"] == '{"root": "/tmp"}'
