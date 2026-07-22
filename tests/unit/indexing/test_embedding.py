"""Unit tests for BatchEmbedder in indexing/embedding.py."""
from __future__ import annotations

import pytest

from beacon_kb.errors import BackendError
from beacon_kb.indexing.embedding import BatchEmbedder
from beacon_kb.models import (
    Chunk,
    ChunkKind,
    RevisionId,
    SectionId,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.testing import FakeEmbedder, FakeFailingEmbedder, FakeProgressObserver


def _make_chunk(ordinal: int = 0, text: str = "sample text") -> Chunk:
    source_id = make_source_id(corpus="test", canonical_uri="fake://doc")
    return Chunk(
        id=make_chunk_id(
            corpus="test",
            canonical_uri="fake://doc",
            revision_id="rev-001",
            pipeline_fingerprint="pipe-v1",
            parent_locator="intro",
            child_ordinal=ordinal,
        ),
        source_id=source_id,
        revision_id=RevisionId("rev-001"),
        section_id=SectionId("sec-001"),
        text=text,
        ordinal=ordinal,
        parent_locator="intro",
        kind=ChunkKind.CHILD,
        token_count=len(text.split()),
    )


def test_batch_embedder_respects_provider_batch_size() -> None:
    """BatchEmbedder must batch chunks according to provider.batch_size, never hardcoded."""
    provider = FakeEmbedder(dim=8, batch_size=3)
    embedder = BatchEmbedder(provider=provider)
    chunks = [_make_chunk(i, f"text {i}") for i in range(7)]
    results = embedder.embed_chunks(chunks)
    assert len(results) == 7
    for _chunk, vec in results:
        assert len(vec) == provider.dimension()


def test_batch_embedder_validates_dimension() -> None:
    """All returned vectors must match provider.dimension()."""
    provider = FakeEmbedder(dim=16, batch_size=4)
    embedder = BatchEmbedder(provider=provider)
    chunks = [_make_chunk(i) for i in range(5)]
    results = embedder.embed_chunks(chunks)
    dim = provider.dimension()
    for _, vec in results:
        assert len(vec) == dim, f"Expected dim {dim}, got {len(vec)}"


def test_batch_embedder_caches_by_chunk_id() -> None:
    """Re-embedding the same chunk must return the cached vector, not call provider again."""
    call_count = {"n": 0}

    class CountingProvider(FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            call_count["n"] += len(texts)
            return super().embed(texts)

    counting = CountingProvider(dim=8, batch_size=4)
    embedder = BatchEmbedder(provider=counting)
    chunk = _make_chunk(0, "unique text for cache")

    embedder.embed_chunks([chunk])
    embedder.embed_chunks([chunk])  # Second call - same chunk ID.

    assert call_count["n"] == 1, (
        "BatchEmbedder must serve repeated chunk IDs from cache, not call provider again."
    )


def test_batch_embedder_empty_list() -> None:
    provider = FakeEmbedder(dim=8, batch_size=4)
    embedder = BatchEmbedder(provider=provider)
    assert embedder.embed_chunks([]) == []


def test_batch_embedder_failing_provider_raises() -> None:
    provider = FakeFailingEmbedder(dim=8, batch_size=4)
    embedder = BatchEmbedder(provider=provider, max_retries=1)
    chunks = [_make_chunk(0)]
    with pytest.raises(BackendError):
        embedder.embed_chunks(chunks)


def test_batch_embedder_batch_size_never_hardcoded() -> None:
    """Changing provider.batch_size must change actual batching behaviour."""
    call_lengths: dict[int, list[int]] = {1: [], 10: []}

    def make_counting(batch_size: int) -> FakeEmbedder:
        class Counting(FakeEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                call_lengths[batch_size].append(len(texts))
                return super().embed(texts)

        return Counting(dim=8, batch_size=batch_size)

    chunks = [_make_chunk(i, f"text {i}") for i in range(5)]

    results1 = BatchEmbedder(provider=make_counting(1)).embed_chunks(chunks)
    results10 = BatchEmbedder(provider=make_counting(10)).embed_chunks(chunks)
    assert len(results1) == 5
    assert len(results10) == 5
    # batch_size=1 forces per-chunk calls; batch_size=10 coalesces all five.
    assert call_lengths[1] == [1, 1, 1, 1, 1]
    assert call_lengths[10] == [5]


def test_batch_embedder_emits_progress_events() -> None:
    """The embed stage emits start/progress/end with current/total and elapsed."""
    provider = FakeEmbedder(dim=8, batch_size=2)
    observer = FakeProgressObserver()
    embedder = BatchEmbedder(provider=provider, observer=observer)
    chunks = [_make_chunk(i, f"text {i}") for i in range(5)]
    embedder.embed_chunks(chunks)

    statuses = [e["status"] for e in observer.events]
    assert "start" in statuses
    assert "end" in statuses
    progress_events = [e for e in observer.events if e["status"] == "progress"]
    assert progress_events, "Expected at least one progress event"
    assert progress_events[-1]["current"] == 5
    for event in observer.events:
        assert event["stage"] == "embed"
        assert "elapsed_seconds" in event
        assert event["total"] == 5
