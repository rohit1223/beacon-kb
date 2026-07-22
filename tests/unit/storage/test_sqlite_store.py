"""Unit tests for the SQLite-backed transactional knowledge store.

Tests are organized by acceptance criteria from the task brief:
- Atomic promotion: staged revisions invisible until promotion.
- Rollback: prior active revision stays searchable after failed write.
- FTS5 and dimension checks at startup with typed BackendError.
- Corpus namespace isolation.
- Restart recovery: durable state survives close/reopen.
- No swallowed errors: failed writes raise BackendError.
- Store contract: registered as 'sqlite' in beacon_kb.stores group.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from beacon_kb.errors import BackendError
from beacon_kb.models import (
    Chunk,
    ChunkKind,
    CorpusId,
    Query,
    QueryId,
    Revision,
    RevisionId,
    SectionId,
    SourceId,
    make_chunk_id,
    make_revision_id,
    make_source_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    corpus: str = "test-corpus",
    uri: str = "fake://doc-1",
    revision_id: str = "rev-001",
    pipeline: str = "pipe-v1",
    ordinal: int = 0,
    text: str = "hello world",
    section_locator: str = "intro",
) -> Chunk:
    source_id = make_source_id(corpus=corpus, canonical_uri=uri)
    chunk_id = make_chunk_id(
        corpus=corpus,
        canonical_uri=uri,
        revision_id=revision_id,
        pipeline_fingerprint=pipeline,
        parent_locator=section_locator,
        child_ordinal=ordinal,
    )
    return Chunk(
        id=chunk_id,
        source_id=source_id,
        revision_id=RevisionId(revision_id),
        section_id=SectionId("sec-001"),
        text=text,
        ordinal=ordinal,
        parent_locator=section_locator,
        kind=ChunkKind.CHILD,
        token_count=len(text.split()),
    )


def _make_revision(
    *,
    source_id: SourceId,
    content_hash: str = "abc123",
    pipeline: str = "pipe-v1",
) -> Revision:
    rev_id = make_revision_id(
        source_id=str(source_id),
        content_hash=content_hash,
        pipeline_fingerprint=pipeline,
    )
    return Revision(
        id=rev_id,
        source_id=source_id,
        content_hash=content_hash,
        pipeline_fingerprint=pipeline,
        byte_size=100,
        fetched_at_iso="2024-01-01T00:00:00Z",
    )


def _unit_vec(dim: int, val: float = 1.0) -> list[float]:
    """Return a unit vector with all equal components."""
    v = [val] * dim
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def store(db_path: Path) -> Any:
    from beacon_kb.storage.sqlite import SQLiteStore

    s = SQLiteStore(db_path=str(db_path), vector_dim=16)
    yield s
    s.close()


@pytest.fixture
def store_dim4(tmp_path: Path) -> Any:
    from beacon_kb.storage.sqlite import SQLiteStore

    db = tmp_path / "dim4.db"
    s = SQLiteStore(db_path=str(db), vector_dim=4)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. Store protocol conformance
# ---------------------------------------------------------------------------


class TestStoreProtocol:
    def test_store_satisfies_store_protocol(self, store: Any) -> None:
        from beacon_kb.protocols import Store

        assert isinstance(store, Store)

    def test_upsert_chunks_returns_none(self, store: Any) -> None:
        chunk = _make_chunk()
        result = store.upsert_chunks([chunk])
        assert result is None

    def test_get_chunk_returns_chunk(self, store: Any) -> None:
        chunk = _make_chunk()
        store.upsert_chunks([chunk])
        found = store.get_chunk(str(chunk.id))
        assert found is not None
        assert found.id == chunk.id
        assert found.text == chunk.text

    def test_get_chunk_missing_returns_none(self, store: Any) -> None:
        result = store.get_chunk("nonexistent-id")
        assert result is None

    def test_delete_chunks_removes_records(self, store: Any) -> None:
        chunk = _make_chunk()
        store.upsert_chunks([chunk])
        store.delete_chunks([str(chunk.id)])
        assert store.get_chunk(str(chunk.id)) is None

    def test_upsert_chunks_idempotent(self, store: Any) -> None:
        chunk = _make_chunk(text="original text")
        store.upsert_chunks([chunk])
        # Upsert same chunk again - should not raise and not duplicate
        store.upsert_chunks([chunk])
        # Still retrievable
        found = store.get_chunk(str(chunk.id))
        assert found is not None


# ---------------------------------------------------------------------------
# 2. Revision staging and atomic promotion
# ---------------------------------------------------------------------------


class TestAtomicPromotion:
    def test_staged_revision_invisible_before_promotion(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        uri = "fake://doc-1"
        source_id = make_source_id(corpus="test-corpus", canonical_uri=uri)
        revision = _make_revision(source_id=source_id)

        # Stage a revision but do not promote
        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri=uri)
        chunk = _make_chunk(revision_id=str(revision.id))
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        # Chunk must not appear in active query results
        assert store.get_chunk(str(chunk.id)) is None

    def test_promotion_makes_revision_visible(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        uri = "fake://doc-1"
        source_id = make_source_id(corpus="test-corpus", canonical_uri=uri)
        revision = _make_revision(source_id=source_id)

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri=uri)
        chunk = _make_chunk(revision_id=str(revision.id))
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])
        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        # After promotion chunk must be visible
        found = store.get_chunk(str(chunk.id))
        assert found is not None
        assert found.id == chunk.id

    def test_rollback_leaves_prior_active_revision_searchable(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        uri = "fake://doc-1"
        source_id = make_source_id(corpus="test-corpus", canonical_uri=uri)

        # First revision - staged and promoted
        rev1 = _make_revision(source_id=source_id, content_hash="hash-v1")
        chunk1 = _make_chunk(revision_id=str(rev1.id), text="first revision content")
        store.stage_revision(corpus_id=corpus_id, revision=rev1, canonical_uri=uri)
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=rev1.id, chunks=[chunk1])
        store.promote_revision(corpus_id=corpus_id, revision_id=rev1.id)

        # Second revision - staged but rolled back
        rev2 = _make_revision(source_id=source_id, content_hash="hash-v2", pipeline="pipe-v2")
        chunk2 = _make_chunk(
            revision_id=str(rev2.id), text="second revision content", ordinal=1
        )
        store.stage_revision(corpus_id=corpus_id, revision=rev2, canonical_uri=uri)
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=rev2.id, chunks=[chunk2])
        store.rollback_revision(corpus_id=corpus_id, revision_id=rev2.id)

        # First revision's chunks still accessible; second revision invisible
        assert store.get_chunk(str(chunk1.id)) is not None
        assert store.get_chunk(str(chunk2.id)) is None


# ---------------------------------------------------------------------------
# 3. FTS5 sparse retrieval
# ---------------------------------------------------------------------------


class TestFTS5Retrieval:
    def test_sparse_retrieve_returns_hits(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id), text="python machine learning tutorial")

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])
        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        query = Query(id=QueryId("q1"), text="python machine learning", corpus_id=corpus_id)
        hits = store.retrieve(query)
        assert len(hits) > 0
        assert all(h.sparse_score is not None for h in hits)
        assert all(h.dense_score is None for h in hits)

    def test_sparse_retrieve_corpus_isolation(self, store: Any) -> None:
        """Retrieval must not return hits from another corpus."""
        corpus_a = CorpusId("corpus-a")
        corpus_b = CorpusId("corpus-b")

        for corpus, uri in [(corpus_a, "fake://a"), (corpus_b, "fake://b")]:
            sid = make_source_id(corpus=str(corpus), canonical_uri=uri)
            rev = _make_revision(source_id=sid, content_hash=f"hash-{corpus}")
            chunk = _make_chunk(
                corpus=str(corpus),
                uri=uri,
                revision_id=str(rev.id),
                text=f"unique content for {corpus}",
            )
            store.stage_revision(corpus_id=corpus, revision=rev, canonical_uri=uri)
            store.upsert_chunks_to_staging(corpus_id=corpus, revision_id=rev.id, chunks=[chunk])
            store.promote_revision(corpus_id=corpus, revision_id=rev.id)

        query_a = Query(id=QueryId("q1"), text="unique content", corpus_id=corpus_a)
        hits_a = store.retrieve(query_a)
        chunk_ids_a = {h.chunk.id for h in hits_a}

        query_b = Query(id=QueryId("q2"), text="unique content", corpus_id=corpus_b)
        hits_b = store.retrieve(query_b)
        chunk_ids_b = {h.chunk.id for h in hits_b}

        # No overlap in results
        assert chunk_ids_a.isdisjoint(chunk_ids_b)

    def test_sparse_retrieve_top_k_respected(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunks = [
            _make_chunk(
                revision_id=str(revision.id),
                text=f"document {i} about machine learning",
                ordinal=i,
            )
            for i in range(5)
        ]

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(
            corpus_id=corpus_id, revision_id=revision.id, chunks=chunks
        )
        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        query = Query(id=QueryId("q1"), text="machine learning", corpus_id=corpus_id, top_k=3)
        hits = store.retrieve(query)
        assert len(hits) <= 3

    def test_sparse_score_not_zero(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id), text="machine learning model training")

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])
        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        query = Query(id=QueryId("q1"), text="machine learning", corpus_id=corpus_id)
        hits = store.retrieve(query)
        assert len(hits) > 0
        # sparse_score must never be zero for a relevant match
        for hit in hits:
            assert hit.sparse_score is not None
            assert hit.sparse_score != 0.0


# ---------------------------------------------------------------------------
# 4. Vector (embedding) storage and retrieval
# ---------------------------------------------------------------------------


class TestVectorStorage:
    def test_upsert_and_retrieve_embedding(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        vec = _unit_vec(16)
        store.upsert_embedding(
            corpus_id=corpus_id,
            chunk_id=chunk.id,
            revision_id=revision.id,
            vector=vec,
            model_name="test-model",
            dimension=16,
            similarity="cosine",
        )

        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        # Retrieve embedding
        embedding = store.get_embedding(chunk_id=str(chunk.id))
        assert embedding is not None
        assert len(embedding) == 16

    def test_wrong_dimension_raises_backend_error(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        # Wrong dimension: store expects 16, we provide 8
        wrong_vec = _unit_vec(8)
        with pytest.raises(BackendError):
            store.upsert_embedding(
                corpus_id=corpus_id,
                chunk_id=chunk.id,
                revision_id=revision.id,
                vector=wrong_vec,
                model_name="test-model",
                dimension=8,
                similarity="cosine",
            )

    def test_dense_vector_search(self, store_dim4: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)

        chunks = [
            _make_chunk(
                revision_id=str(revision.id),
                text=f"chunk {i}",
                ordinal=i,
            )
            for i in range(3)
        ]

        store_dim4.stage_revision(
            corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1"
        )
        store_dim4.upsert_chunks_to_staging(
            corpus_id=corpus_id, revision_id=revision.id, chunks=chunks
        )

        # Each chunk gets a distinct unit vector
        vecs = [
            _unit_vec(4, v) for v in [1.0, 0.5, -1.0]
        ]
        for chunk, vec in zip(chunks, vecs, strict=True):
            store_dim4.upsert_embedding(
                corpus_id=corpus_id,
                chunk_id=chunk.id,
                revision_id=revision.id,
                vector=vec,
                model_name="test-model",
                dimension=4,
                similarity="cosine",
            )

        store_dim4.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        # Search with a query vector similar to vecs[0]
        query_vec = _unit_vec(4, 1.0)
        hits = store_dim4.dense_retrieve(
            query_vector=query_vec,
            corpus_id=corpus_id,
            top_k=3,
            similarity="cosine",
        )
        assert len(hits) > 0
        assert all(h.dense_score is not None for h in hits)
        assert all(h.sparse_score is None for h in hits)
        # Scores must be ordered descending
        scores = [h.dense_score for h in hits if h.dense_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_non_unit_cosine_vector_raises_backend_error(self, store: Any) -> None:
        """upsert_embedding must reject non-unit vectors for cosine similarity."""
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        # Vector with L2 norm != 1.0 - must be rejected for cosine similarity
        non_unit_vec = [1.0] * 16  # norm = 4.0, far from 1.0
        with pytest.raises(BackendError):
            store.upsert_embedding(
                corpus_id=corpus_id,
                chunk_id=chunk.id,
                revision_id=revision.id,
                vector=non_unit_vec,
                model_name="test-model",
                dimension=16,
                similarity="cosine",
            )

    def test_unit_cosine_vector_accepted(self, store: Any) -> None:
        """upsert_embedding must accept properly unit-normalized cosine vectors."""
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        unit_vec = _unit_vec(16)
        # Must not raise
        store.upsert_embedding(
            corpus_id=corpus_id,
            chunk_id=chunk.id,
            revision_id=revision.id,
            vector=unit_vec,
            model_name="test-model",
            dimension=16,
            similarity="cosine",
        )

    def test_non_unit_dot_vector_accepted(self, store: Any) -> None:
        """dot similarity does not require unit normalization - accepts any vector."""
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        # Non-unit vector: L2 norm != 1.0 - dot similarity permits this
        non_unit_vec = [0.5] * 16
        # Must not raise for dot similarity
        store.upsert_embedding(
            corpus_id=corpus_id,
            chunk_id=chunk.id,
            revision_id=revision.id,
            vector=non_unit_vec,
            model_name="test-model",
            dimension=16,
            similarity="dot",
        )

    def test_non_unit_euclidean_vector_accepted(self, store: Any) -> None:
        """euclidean similarity does not require unit normalization - accepts any vector."""
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id))

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])

        # Non-unit vector: L2 norm != 1.0 - euclidean similarity permits this
        non_unit_vec = [2.0] + [0.0] * 15
        # Must not raise for euclidean similarity
        store.upsert_embedding(
            corpus_id=corpus_id,
            chunk_id=chunk.id,
            revision_id=revision.id,
            vector=non_unit_vec,
            model_name="test-model",
            dimension=16,
            similarity="euclidean",
        )


# ---------------------------------------------------------------------------
# 5. Corpus namespace isolation
# ---------------------------------------------------------------------------


class TestCorpusNamespaceIsolation:
    def test_identical_uri_different_corpora_isolated(self, store: Any) -> None:
        """Two corpora with identical source URIs must never share records."""
        uri = "fake://shared-uri"

        corpus_a = CorpusId("corpus-a")
        corpus_b = CorpusId("corpus-b")

        for corpus in [corpus_a, corpus_b]:
            sid = make_source_id(corpus=str(corpus), canonical_uri=uri)
            rev = _make_revision(source_id=sid, content_hash=f"hash-{corpus}")
            chunk = _make_chunk(
                corpus=str(corpus),
                uri=uri,
                revision_id=str(rev.id),
                text=f"private data for {corpus}",
            )
            store.stage_revision(corpus_id=corpus, revision=rev, canonical_uri=uri)
            store.upsert_chunks_to_staging(corpus_id=corpus, revision_id=rev.id, chunks=[chunk])
            store.promote_revision(corpus_id=corpus, revision_id=rev.id)

        # Each corpus has its own active revision; active revision for corpus_a
        # must not be corpus_b's revision.
        active_a = store.get_active_revision_id(corpus_id=corpus_a, canonical_uri=uri)
        active_b = store.get_active_revision_id(corpus_id=corpus_b, canonical_uri=uri)
        assert active_a != active_b

    def test_delete_in_one_corpus_does_not_affect_other(self, store: Any) -> None:
        uri = "fake://shared-uri"
        corpus_a = CorpusId("corpus-a")
        corpus_b = CorpusId("corpus-b")

        chunks_by_corpus: dict[str, Chunk] = {}
        for corpus in [corpus_a, corpus_b]:
            sid = make_source_id(corpus=str(corpus), canonical_uri=uri)
            rev = _make_revision(source_id=sid, content_hash=f"hash-{corpus}")
            chunk = _make_chunk(
                corpus=str(corpus),
                uri=uri,
                revision_id=str(rev.id),
                text=f"content for {corpus}",
            )
            chunks_by_corpus[str(corpus)] = chunk
            store.stage_revision(corpus_id=corpus, revision=rev, canonical_uri=uri)
            store.upsert_chunks_to_staging(corpus_id=corpus, revision_id=rev.id, chunks=[chunk])
            store.promote_revision(corpus_id=corpus, revision_id=rev.id)

        # Delete corpus_a's chunk directly
        chunk_a = chunks_by_corpus[str(corpus_a)]
        store.delete_chunks([str(chunk_a.id)])

        # corpus_b's chunk must still be accessible
        chunk_b = chunks_by_corpus[str(corpus_b)]
        assert store.get_chunk(str(chunk_b.id)) is not None


# ---------------------------------------------------------------------------
# 6. Restart recovery (durable state)
# ---------------------------------------------------------------------------


class TestRestartRecovery:
    def test_active_revision_survives_reopen(self, db_path: Path) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id), text="durable content")

        # First session: write and promote
        s1 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        s1.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        s1.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])
        s1.promote_revision(corpus_id=corpus_id, revision_id=revision.id)
        s1.close()

        # Second session: reopen and verify data is there
        s2 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        found = s2.get_chunk(str(chunk.id))
        assert found is not None
        assert found.text == "durable content"
        s2.close()

    def test_build_run_status_survives_reopen(self, db_path: Path) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        corpus_id = CorpusId("test-corpus")

        s1 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        run_id = s1.create_build_run(
            corpus_id=corpus_id,
            pipeline_fingerprint="pipe-v1",
            started_at_iso="2024-01-01T00:00:00Z",
        )
        s1.finish_build_run(build_run_id=run_id, status="success", sources_scanned=5)
        s1.close()

        s2 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        run_info = s2.get_build_run(build_run_id=run_id)
        assert run_info is not None
        assert run_info["status"] == "success"
        assert run_info["sources_scanned"] == 5
        s2.close()

    def test_no_active_revision_on_fresh_store(self, store: Any) -> None:
        result = store.get_active_revision_id(
            corpus_id=CorpusId("empty-corpus"),
            canonical_uri="fake://nonexistent",
        )
        assert result is None


# ---------------------------------------------------------------------------
# 7. Error typing: no swallowed writes
# ---------------------------------------------------------------------------


class TestNoSwallowedErrors:
    def test_upsert_raises_backend_error_on_write_failure(self, tmp_path: Path) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        db = tmp_path / "readonly.db"
        s = SQLiteStore(db_path=str(db), vector_dim=16)
        # Close the connection and make the file read-only to simulate failure
        s.close()
        db.chmod(0o444)

        # Re-opening should fail or writes should raise BackendError
        try:
            s2 = SQLiteStore(db_path=str(db), vector_dim=16)
            chunk = _make_chunk()
            with pytest.raises(BackendError):
                s2.upsert_chunks([chunk])
            s2.close()
        except BackendError:
            pass  # Opening itself raising BackendError is also acceptable
        finally:
            db.chmod(0o644)  # Restore for cleanup

    def test_failed_write_does_not_drift_stores(self, store: Any) -> None:
        corpus_id = CorpusId("test-corpus")
        source_id = make_source_id(corpus="test-corpus", canonical_uri="fake://doc-1")
        revision = _make_revision(source_id=source_id)
        chunk = _make_chunk(revision_id=str(revision.id), text="stable content")

        store.stage_revision(corpus_id=corpus_id, revision=revision, canonical_uri="fake://doc-1")
        store.upsert_chunks_to_staging(corpus_id=corpus_id, revision_id=revision.id, chunks=[chunk])
        store.promote_revision(corpus_id=corpus_id, revision_id=revision.id)

        # Try to upsert an embedding with wrong dimension - must raise, not silently skip
        wrong_vec = _unit_vec(8)  # Store expects 16
        with pytest.raises(BackendError):
            store.upsert_embedding(
                corpus_id=corpus_id,
                chunk_id=chunk.id,
                revision_id=revision.id,
                vector=wrong_vec,
                model_name="test-model",
                dimension=8,
                similarity="cosine",
            )

        # Chunk is still accessible - store not drifted
        found = store.get_chunk(str(chunk.id))
        assert found is not None


# ---------------------------------------------------------------------------
# 8. Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def setup_method(self) -> None:
        """Re-run builtins registration in case a prior test cleared the registry."""
        # Some contract tests call precedence.clear_registry() in their
        # setup_method.  Re-importing builtins re-triggers _register_builtins().
        import importlib

        import beacon_kb.registry.builtins as _b

        importlib.reload(_b)

    def test_sqlite_store_registered_as_builtin(self) -> None:
        from beacon_kb.registry import groups
        from beacon_kb.registry import precedence as prec

        # The sqlite store should be registered as a builtin
        builtin = prec._builtins.get(groups.STORES)
        assert builtin is not None
        name, _instance = builtin
        assert name == "sqlite"

    def test_sqlite_store_resolves_via_registry(self, tmp_path: Path) -> None:
        """Resolve 'sqlite' store from the registry by name."""
        from beacon_kb import registry
        from beacon_kb.registry import groups
        from beacon_kb.storage.sqlite import SQLiteStore

        instance = registry.resolve(group=groups.STORES, name="sqlite", protocol=None)
        assert isinstance(instance, SQLiteStore)


# ---------------------------------------------------------------------------
# 9. Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_schema_version_recorded(self, db_path: Path) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        s = SQLiteStore(db_path=str(db_path), vector_dim=16)
        version = s.schema_version()
        assert version >= 1
        s.close()

    def test_migration_idempotent(self, db_path: Path) -> None:
        """Opening the same DB twice should not re-apply migrations."""
        from beacon_kb.storage.sqlite import SQLiteStore

        s1 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        v1 = s1.schema_version()
        s1.close()

        s2 = SQLiteStore(db_path=str(db_path), vector_dim=16)
        v2 = s2.schema_version()
        s2.close()

        assert v1 == v2
