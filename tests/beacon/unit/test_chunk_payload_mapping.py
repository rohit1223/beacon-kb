"""Unit tests for Chunk-to-ChunkPayload round-trip mapping (Task 02.4).

Verifies that every field retrieval needs survives the conversion
from Chunk (the chunker's typed record) to ChunkPayload (the Qdrant
point payload schema) and that the mapping is complete and lossless.
"""

from __future__ import annotations

from beacon.ingest.chunking import (
    Chunk,
    ChunkerConfig,
    ChunkKind,
    DocumentChunker,
    chunk_to_payload,
)
from beacon.ingest.parsing import ParsedDocument, ParsedSection, SectionKind
from beacon.storage.payload import ChunkPayload


def _make_doc(*sections: ParsedSection, title: str = "Test Doc") -> ParsedDocument:
    return ParsedDocument(
        title=title,
        media_type="text/markdown",
        sections=tuple(sections),
        warnings=(),
        parser_version="docling-2.beacon-adapter-1",
    )


def _section(
    text: str,
    locator: str = "Root",
    heading_path: tuple[str, ...] = ("Root",),
    kind: SectionKind = SectionKind.TEXT,
    ordinal: int = 0,
) -> ParsedSection:
    return ParsedSection(
        locator=locator,
        heading_path=heading_path,
        heading=heading_path[-1] if heading_path else "",
        kind=kind,
        text=text,
        ordinal=ordinal,
    )


# ~180 tokens - long enough to force splits at child_chunk_size=64.
_LONG_TEXT = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu alpha bravo charlie delta "
    "echo foxtrot golf hotel india juliet kilo lima mike november oscar "
    "papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu"
)


def _get_chunks() -> list[Chunk]:
    doc = _make_doc(
        _section(
            _LONG_TEXT,
            locator="Guide/Install",
            heading_path=("Guide", "Install"),
        )
    )
    cfg = ChunkerConfig(parent_chunk_size=256, child_chunk_size=64, chunk_overlap=16)
    return DocumentChunker(
        collection="testcol",
        canonical_uri="file:///readme.md",
        content_hash="abc123def456",
        config=cfg,
    ).chunk(doc)


class TestChunkToPayloadMapping:
    """chunk_to_payload converts a Chunk to ChunkPayload without data loss."""

    def test_returns_chunk_payload(self) -> None:
        chunks = _get_chunks()
        assert chunks
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=["v2"],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123def456",
            fingerprint="fp001",
        )
        assert isinstance(payload, ChunkPayload)

    def test_chunk_text_preserved(self) -> None:
        chunks = _get_chunks()
        child = next(c for c in chunks if c.kind == ChunkKind.CHILD)
        payload = chunk_to_payload(
            child,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.chunk_text == child.text

    def test_heading_path_preserved(self) -> None:
        chunks = _get_chunks()
        child = next(c for c in chunks if c.kind == ChunkKind.CHILD)
        payload = chunk_to_payload(
            child,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.heading_path == list(child.heading_path)

    def test_parent_chunk_id_on_child(self) -> None:
        chunks = _get_chunks()
        child = next(c for c in chunks if c.kind == ChunkKind.CHILD)
        payload = chunk_to_payload(
            child,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.parent_chunk_id == child.parent_chunk_id

    def test_parent_chunk_id_none_on_parent(self) -> None:
        chunks = _get_chunks()
        parent = next(c for c in chunks if c.kind == ChunkKind.PARENT)
        payload = chunk_to_payload(
            parent,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.parent_chunk_id is None

    def test_source_uri_preserved(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.source_uri == "file:///readme.md"

    def test_title_preserved(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="My Title",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.title == "My Title"

    def test_tags_preserved(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=["python", "rag"],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.tags == ["python", "rag"]

    def test_fingerprint_preserved(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="my-fingerprint",
        )
        assert payload.fingerprint == "my-fingerprint"

    def test_content_hash_preserved(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="deadbeef",
            fingerprint="fp",
        )
        assert payload.content_hash == "deadbeef"

    def test_chunk_hash_is_chunk_id(self) -> None:
        """chunk_hash in payload is the deterministic chunk_id."""
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert payload.chunk_hash == chunks[0].chunk_id

    def test_kind_preserved(self) -> None:
        """kind field in payload matches the chunk's kind value."""
        chunks = _get_chunks()
        parent = next(c for c in chunks if c.kind == ChunkKind.PARENT)
        child = next(c for c in chunks if c.kind == ChunkKind.CHILD)
        parent_payload = chunk_to_payload(
            parent,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        child_payload = chunk_to_payload(
            child,
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert parent_payload.kind == "parent"
        assert child_payload.kind == "child"

    def test_section_kind_preserved(self) -> None:
        """section_kind in payload carries the SectionKind string value."""
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        # _LONG_TEXT section uses default TEXT kind.
        assert payload.section_kind == SectionKind.TEXT.value

    def test_prev_next_chunk_ids_on_child(self) -> None:
        """prev_chunk_id and next_chunk_id are threaded correctly."""
        chunks = _get_chunks()
        children = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(children) >= 2, "Need multiple children to test neighbor links"
        # Middle child should have both prev and next.
        if len(children) >= 3:
            mid = children[1]
            payload = chunk_to_payload(
                mid,
                source_uri="file:///readme.md",
                title="Guide",
                tags=[],
                ingested_at="2026-01-01T00:00:00Z",
                content_hash="abc123",
                fingerprint="fp",
            )
            assert payload.prev_chunk_id == mid.prev_chunk_id
            assert payload.next_chunk_id == mid.next_chunk_id
        # First child has no prev.
        first_payload = chunk_to_payload(
            children[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert first_payload.prev_chunk_id is None
        # Last child has no next.
        last_payload = chunk_to_payload(
            children[-1],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        assert last_payload.next_chunk_id is None

    def test_to_dict_round_trip(self) -> None:
        """payload.to_dict() produces a plain dict with all required keys."""
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
        )
        d = payload.to_dict()
        required = {
            "chunk_text",
            "source_uri",
            "title",
            "heading_path",
            "tags",
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
        assert required.issubset(d.keys())
        assert d["chunk_text"] == payload.chunk_text

    def test_to_dict_is_lossless_over_all_dataclass_fields(self) -> None:
        """Every ChunkPayload dataclass field appears in to_dict() with the
        same value: the mapping is complete and lossless by construction, and
        any field added to the dataclass without a to_dict() entry fails here.
        """
        import dataclasses

        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///readme.md",
            title="Guide",
            tags=["a"],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc123",
            fingerprint="fp",
            created_at="2025-01-01T00:00:00Z",
            modified_at="2025-06-01T00:00:00Z",
        )
        d = payload.to_dict()
        field_names = {f.name for f in dataclasses.fields(ChunkPayload)}
        assert set(d.keys()) == field_names, (
            f"to_dict() keys diverge from dataclass fields: "
            f"missing={field_names - set(d.keys())}, extra={set(d.keys()) - field_names}"
        )
        for name in field_names:
            assert d[name] == getattr(payload, name), f"Field {name!r} value lost in to_dict()"


class TestPayloadOptionalDates:
    def test_created_at_defaults_none(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///x.md",
            title="T",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="h",
            fingerprint="fp",
        )
        assert payload.created_at is None

    def test_created_at_set(self) -> None:
        chunks = _get_chunks()
        payload = chunk_to_payload(
            chunks[0],
            source_uri="file:///x.md",
            title="T",
            tags=[],
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="h",
            fingerprint="fp",
            created_at="2025-01-01T00:00:00Z",
            modified_at="2025-06-01T00:00:00Z",
        )
        assert payload.created_at == "2025-01-01T00:00:00Z"
        assert payload.modified_at == "2025-06-01T00:00:00Z"
