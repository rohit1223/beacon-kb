"""Unit tests for the typed payload schema, named-vector constants, and index declarations.

These tests are pure Python - no Qdrant client, no I/O.
"""

from __future__ import annotations

from typing import Any

from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
    ChunkPayload,
)


class TestVectorNameConstants:
    """Named-vector constants must be stable strings consumed by Epic 02 and 03."""

    def test_dense_vector_name_is_str(self) -> None:
        assert isinstance(DENSE_VECTOR_NAME, str)
        assert len(DENSE_VECTOR_NAME) > 0

    def test_sparse_vector_name_is_str(self) -> None:
        assert isinstance(SPARSE_VECTOR_NAME, str)
        assert len(SPARSE_VECTOR_NAME) > 0

    def test_vector_names_are_distinct(self) -> None:
        assert DENSE_VECTOR_NAME != SPARSE_VECTOR_NAME


class TestChunkPayloadSchema:
    """ChunkPayload must carry all required fields per the brief."""

    def _minimal(self) -> dict[str, Any]:
        return {
            "chunk_text": "hello world",
            "source_uri": "file:///test.md",
            "title": "Test",
            "heading_path": [],
            "tags": [],
            "created_at": None,
            "modified_at": None,
            "ingested_at": "2026-01-01T00:00:00Z",
            "content_hash": "abc123",
            "chunk_hash": "def456",
            "parent_chunk_id": None,
            "fingerprint": "ghi789",
            "kind": "parent",
            "section_kind": "text",
        }

    def test_create_minimal_payload(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.chunk_text == "hello world"

    def test_source_uri_present(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.source_uri == "file:///test.md"

    def test_title_present(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.title == "Test"

    def test_heading_path_is_list(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert isinstance(payload.heading_path, list)

    def test_tags_is_list(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert isinstance(payload.tags, list)

    def test_dates_optional(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.created_at is None
        assert payload.modified_at is None

    def test_ingested_at_required(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.ingested_at == "2026-01-01T00:00:00Z"

    def test_content_hash_present(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.content_hash == "abc123"

    def test_chunk_hash_present(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.chunk_hash == "def456"

    def test_parent_chunk_id_optional(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.parent_chunk_id is None

    def test_parent_chunk_id_set(self) -> None:
        data = self._minimal()
        data["parent_chunk_id"] = "parent-001"
        payload = ChunkPayload(**data)
        assert payload.parent_chunk_id == "parent-001"

    def test_fingerprint_present(self) -> None:
        payload = ChunkPayload(**self._minimal())
        assert payload.fingerprint == "ghi789"

    def test_to_dict_roundtrip(self) -> None:
        payload = ChunkPayload(**self._minimal())
        d = payload.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_text"] == "hello world"
        assert d["source_uri"] == "file:///test.md"

    def test_to_dict_contains_all_fields(self) -> None:
        payload = ChunkPayload(**self._minimal())
        d = payload.to_dict()
        required_keys = {
            "chunk_text",
            "source_uri",
            "title",
            "heading_path",
            "tags",
            "created_at",
            "modified_at",
            "ingested_at",
            "content_hash",
            "chunk_hash",
            "parent_chunk_id",
            "fingerprint",
            "kind",
            "section_kind",
            "prev_chunk_id",
            "next_chunk_id",
        }
        assert required_keys.issubset(d.keys())


class TestPayloadIndexFields:
    """PAYLOAD_INDEX_FIELDS must include source_uri, tags, and date fields."""

    def test_payload_index_fields_is_list(self) -> None:
        assert isinstance(PAYLOAD_INDEX_FIELDS, list)
        assert len(PAYLOAD_INDEX_FIELDS) > 0

    def test_source_uri_indexed(self) -> None:
        field_names = [f[0] for f in PAYLOAD_INDEX_FIELDS]
        assert "source_uri" in field_names

    def test_tags_indexed(self) -> None:
        field_names = [f[0] for f in PAYLOAD_INDEX_FIELDS]
        assert "tags" in field_names

    def test_ingested_at_indexed(self) -> None:
        field_names = [f[0] for f in PAYLOAD_INDEX_FIELDS]
        assert "ingested_at" in field_names

    def test_created_at_indexed(self) -> None:
        field_names = [f[0] for f in PAYLOAD_INDEX_FIELDS]
        assert "created_at" in field_names

    def test_modified_at_indexed(self) -> None:
        field_names = [f[0] for f in PAYLOAD_INDEX_FIELDS]
        assert "modified_at" in field_names

    def test_each_entry_has_field_name_and_schema(self) -> None:
        for entry in PAYLOAD_INDEX_FIELDS:
            assert len(entry) == 2, f"Expected (name, schema), got {entry!r}"
            assert isinstance(entry[0], str)
