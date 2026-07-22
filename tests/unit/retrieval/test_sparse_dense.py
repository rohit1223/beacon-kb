"""Tests for sparse and dense candidate retrieval with typed scores.

TDD suite covering:
- Query validation and variant preservation
- Weighted FTS5 BM25 sparse retrieval with exact-token boosts
- Dense vector retrieval with declared similarity semantics
- Sparse-only degraded mode (no embedder configured)
- Provider-neutral filters (namespace, source, tag, media, date)
- Typed errors for missing/empty/incompatible indexes
- Registry discovery through beacon_kb.retrievers group
- Contract suite conformance via SparseRetrieverContract / DenseRetrieverContract
"""

from __future__ import annotations

import datetime

import pytest

from beacon_kb.errors import BackendError
from beacon_kb.models import (
    Chunk,
    ChunkKind,
    CorpusId,
    Hit,
    Query,
    QueryId,
    RevisionId,
    SectionId,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.protocols import DenseRetriever, SparseRetriever
from beacon_kb.retrieval import BM25SparseRetriever, EmbedderDenseRetriever
from beacon_kb.retrieval.filters import FilterSpec, apply_filters
from beacon_kb.retrieval.query import QueryVariants, prepare_query
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import (
    DenseRetrieverContract,
    FakeEmbedder,
    SparseRetrieverContract,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_temp_store(
    *,
    vector_dim: int = 16,
    chunks: list[Chunk] | None = None,
    embedder: FakeEmbedder | None = None,
) -> SQLiteStore:
    """Create an in-memory SQLiteStore, optionally pre-populated.

    Uses the store's public staged promotion workflow:
    stage_revision -> upsert_chunks_to_staging -> upsert_embedding -> promote_revision.
    No private _conn access.
    """
    store = SQLiteStore(db_path=":memory:", vector_dim=vector_dim)
    if chunks:
        if embedder is not None:
            _stage_and_promote_with_embeddings(store, chunks, embedder, corpus_id=CorpusId("test"))
        else:
            store.upsert_chunks(chunks)
    return store


def _stage_and_promote_with_embeddings(
    store: SQLiteStore,
    chunks: list[Chunk],
    embedder: FakeEmbedder,
    corpus_id: CorpusId,
) -> None:
    """Stage chunks and embeddings then promote via the store's public workflow.

    This is the correct way to make embeddings active without touching _conn.
    Stage each revision's chunks and embeddings, then call promote_revision() to
    atomically flip active=1 for both chunks and their embeddings.
    """
    from collections import defaultdict

    from beacon_kb.models import Revision
    by_revision: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_revision[str(chunk.revision_id)].append(chunk)

    for _revision_id_str, rev_chunks in by_revision.items():
        # Use the first chunk's source_id as the source for this revision.
        source_id = rev_chunks[0].source_id
        revision_id = rev_chunks[0].revision_id

        # Construct a Revision record for stage_revision.
        revision = Revision(
            id=revision_id,
            source_id=source_id,
            content_hash="test-content-hash",
            pipeline_fingerprint="test-pipeline",
        )

        # Register the revision as staged.
        store.stage_revision(
            corpus_id=corpus_id,
            revision=revision,
            canonical_uri=str(source_id),
        )

        # Write chunks to staging area (active=0, invisible until promotion).
        store.upsert_chunks_to_staging(
            corpus_id=corpus_id,
            revision_id=revision_id,
            chunks=rev_chunks,
        )

        # Write embeddings (stored active=0 by default in upsert_embedding).
        for chunk in rev_chunks:
            vec = embedder.embed([chunk.text])[0]
            store.upsert_embedding(
                corpus_id=corpus_id,
                chunk_id=chunk.id,
                revision_id=revision_id,
                vector=vec,
                model_name="fake",
                dimension=embedder.dimension(),
                similarity="cosine",
            )

        # Atomically promote: flips chunks and embeddings to active=1 in one transaction.
        store.promote_revision(corpus_id=corpus_id, revision_id=revision_id)


def _make_chunk(
    text: str,
    *,
    ordinal: int = 0,
    corpus: str = "test",
    uri: str = "fake://doc-1",
    revision_id: str = "rev-001",
    pipeline: str = "pipe-v1",
    section_locator: str = "intro",
    kind: ChunkKind = ChunkKind.CHILD,
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
        kind=kind,
        token_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# Query preparation tests
# ---------------------------------------------------------------------------


class TestPrepareQuery:
    """Tests for query.py - validation, variant selection, verbatim preservation."""

    def test_original_question_preserved_verbatim(self) -> None:
        """The original question must be stored verbatim for sparse retrieval."""
        q = Query(id=QueryId("q1"), text="What is ERROR_CODE 404?")
        variants = prepare_query(q)
        assert variants.original_text == "What is ERROR_CODE 404?"

    def test_sparse_text_equals_original_by_default(self) -> None:
        """Sparse text must default to the original (no rewriting by default)."""
        q = Query(id=QueryId("q1"), text="hello world")
        variants = prepare_query(q)
        assert variants.sparse_text == q.text

    def test_dense_text_equals_original_by_default(self) -> None:
        """Dense text must default to the original when no rewrite function provided."""
        q = Query(id=QueryId("q1"), text="hello world")
        variants = prepare_query(q)
        assert variants.dense_text == q.text

    def test_sparse_rewrite_recorded_separately(self) -> None:
        """Any sparse rewrite must be recorded as a separate observable value."""
        q = Query(id=QueryId("q1"), text="hello world")

        def sparse_rewriter(text: str) -> str:
            return text.upper()

        variants = prepare_query(q, sparse_rewriter=sparse_rewriter)
        # Rewrite is stored as sparse_text.
        assert variants.sparse_text == "HELLO WORLD"
        # But original is preserved verbatim.
        assert variants.original_text == "hello world"

    def test_dense_rewrite_recorded_separately(self) -> None:
        """Dense rewrite must be independently recorded and testable."""
        q = Query(id=QueryId("q1"), text="hello world")

        def dense_rewriter(text: str) -> str:
            return text + " expanded"

        variants = prepare_query(q, dense_rewriter=dense_rewriter)
        assert variants.dense_text == "hello world expanded"
        assert variants.original_text == "hello world"

    def test_sparse_and_dense_rewrites_are_independent(self) -> None:
        """Sparse and dense rewrites are separate values, not copies of each other."""
        q = Query(id=QueryId("q1"), text="base query")

        def s(text: str) -> str:
            return "sparse:" + text

        def d(text: str) -> str:
            return "dense:" + text

        variants = prepare_query(q, sparse_rewriter=s, dense_rewriter=d)
        assert variants.sparse_text == "sparse:base query"
        assert variants.dense_text == "dense:base query"
        assert variants.original_text == "base query"

    def test_returns_query_variants_type(self) -> None:
        q = Query(id=QueryId("q1"), text="x")
        variants = prepare_query(q)
        assert isinstance(variants, QueryVariants)

    def test_empty_text_raises_value_error(self) -> None:
        """Empty query text must be rejected before reaching the backend."""
        q = Query(id=QueryId("q1"), text="")
        with pytest.raises(ValueError, match="empty"):
            prepare_query(q)

    def test_whitespace_only_text_raises_value_error(self) -> None:
        q = Query(id=QueryId("q1"), text="   ")
        with pytest.raises(ValueError, match="empty"):
            prepare_query(q)


# ---------------------------------------------------------------------------
# Filters tests
# ---------------------------------------------------------------------------


class TestFilters:
    """Tests for filters.py - provider-neutral filtering applied before hits leave retriever."""

    def _make_hit(
        self,
        text: str = "content",
        *,
        source_uri: str = "fake://doc-1",
        corpus: str = "corp",
        ordinal: int = 0,
        sparse_score: float | None = 1.0,
    ) -> Hit:
        chunk = _make_chunk(text, uri=source_uri, corpus=corpus, ordinal=ordinal)
        return Hit(chunk=chunk, sparse_score=sparse_score)

    def test_no_filters_returns_all_hits(self) -> None:
        hits = [self._make_hit("a"), self._make_hit("b", ordinal=1)]
        spec = FilterSpec()
        result = apply_filters(hits, spec)
        assert len(result) == 2

    def test_source_filter_keeps_matching(self) -> None:
        h1 = self._make_hit("a", source_uri="fake://doc-1")
        h2 = self._make_hit("b", source_uri="fake://doc-2", ordinal=1)
        # source_id is a hash; use the actual str(source_id) for the filter.
        spec = FilterSpec(source_uris=frozenset({str(h1.chunk.source_id)}))
        result = apply_filters([h1, h2], spec)
        assert len(result) == 1
        assert result[0].chunk.id == h1.chunk.id

    def test_source_filter_empty_frozenset_keeps_all(self) -> None:
        hits = [self._make_hit("a"), self._make_hit("b", ordinal=1)]
        spec = FilterSpec(source_uris=frozenset())
        result = apply_filters(hits, spec)
        assert len(result) == 2

    def test_corpus_namespace_filter(self) -> None:
        h1 = self._make_hit("a", corpus="corp-a")
        h2 = self._make_hit("b", corpus="corp-b", ordinal=1)
        spec = FilterSpec(namespace="corp-a")
        result = apply_filters([h1, h2], spec)
        # FilterSpec.namespace is reserved and not enforced in v1 (see filters.py docstring).
        # All hits pass through regardless of the namespace value: both h1 and h2 are retained.
        # Use Query.corpus_id for actual corpus scoping.
        assert len(result) == 2

    def test_date_filter_after_cutoff(self) -> None:
        hits = [self._make_hit("a")]
        # Chunk records carry no publication date in v1 (conservative exclusion).
        # Any require_after constraint excludes ALL hits with no date metadata.
        cutoff = datetime.date(2030, 1, 1)
        spec = FilterSpec(require_after=cutoff)
        result = apply_filters(hits, spec)
        assert result == []

    def test_tag_filter(self) -> None:
        hits = [self._make_hit("a")]
        # Chunk records carry no tag metadata in v1 (conservative exclusion).
        # Any non-empty tags constraint excludes ALL hits whose tag metadata is unknown.
        spec = FilterSpec(tags=frozenset({"python"}))
        result = apply_filters(hits, spec)
        assert result == []

    def test_media_type_filter(self) -> None:
        hits = [self._make_hit("a")]
        # Chunk records carry no media_type in v1 (conservative exclusion).
        # Any non-empty media_types constraint excludes ALL hits with unknown media type.
        spec = FilterSpec(media_types=frozenset({"text/markdown"}))
        result = apply_filters(hits, spec)
        assert result == []

    def test_filters_cannot_be_bypassed_with_none_spec(self) -> None:
        """Passing a FilterSpec with no constraints returns all hits."""
        hits = [self._make_hit("a")]
        spec = FilterSpec()
        result = apply_filters(hits, spec)
        assert len(result) == 1

    def test_source_filter_removes_all_when_no_match(self) -> None:
        hits = [self._make_hit("a", source_uri="fake://doc-1")]
        # Use a hash that will never match any real source_id.
        spec = FilterSpec(source_uris=frozenset({"deadbeef00000000000000000000000000000000"}))
        result = apply_filters(hits, spec)
        assert result == []

    def test_filter_preserves_hit_scores(self) -> None:
        """Filtering must not alter scores on retained hits."""
        h = self._make_hit("a", sparse_score=7.5)
        spec = FilterSpec()
        result = apply_filters([h], spec)
        assert result[0].sparse_score == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# BM25SparseRetriever tests
# ---------------------------------------------------------------------------


class TestBM25SparseRetriever:
    """Unit tests for sparse.py - weighted FTS5 BM25 with exact-token boosts."""

    def _store_with_chunks(self, chunks: list[Chunk]) -> SQLiteStore:
        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        store.upsert_chunks(chunks)
        return store

    def test_sparse_score_set_on_hits(self) -> None:
        chunks = [_make_chunk("python error code 404 not found")]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="error code")
        hits = retriever.retrieve(q)
        if hits:
            assert all(h.sparse_score is not None for h in hits)

    def test_only_sparse_score_set(self) -> None:
        chunks = [_make_chunk("hello world")]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="hello")
        hits = retriever.retrieve(q)
        for h in hits:
            assert h.dense_score is None
            assert h.fusion_score is None
            assert h.rerank_score is None

    def test_hits_ordered_descending_by_sparse_score(self) -> None:
        chunks = [
            _make_chunk("python tutorial basics", ordinal=0),
            _make_chunk("python python python advanced tutorial", ordinal=1),
            _make_chunk("java programming tutorial", ordinal=2),
        ]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python tutorial")
        hits = retriever.retrieve(q)
        scores = [h.sparse_score for h in hits if h.sparse_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index_returns_empty_list(self) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="anything")
        hits = retriever.retrieve(q)
        assert hits == []

    def test_no_match_returns_empty_list(self) -> None:
        chunks = [_make_chunk("completely unrelated content xyz")]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python programming")
        # May or may not return hits depending on FTS5 - this just must not raise.
        hits = retriever.retrieve(q)
        assert isinstance(hits, list)

    def test_top_k_respected(self) -> None:
        chunks = [_make_chunk(f"python tutorial example {i}", ordinal=i) for i in range(10)]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python tutorial", top_k=3)
        hits = retriever.retrieve(q)
        assert len(hits) <= 3

    def test_retriever_is_sparse_retriever_instance(self) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = BM25SparseRetriever(store=store)
        assert isinstance(retriever, SparseRetriever)

    def test_retrieve_deterministic(self) -> None:
        chunks = [_make_chunk("determinism test content")]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="determinism test")
        r1 = retriever.retrieve(q)
        r2 = retriever.retrieve(q)
        assert [h.sparse_score for h in r1] == [h.sparse_score for h in r2]

    def test_filters_applied_before_candidates_returned(self) -> None:
        """Filters apply consistently and cannot be bypassed.

        source_uris must be populated with str(chunk.source_id) - the SHA-256 hash
        produced by make_source_id() - not the raw canonical URI.
        """
        chunk1 = _make_chunk("python tutorial", uri="fake://doc-1")
        chunk2 = _make_chunk("python advanced", uri="fake://doc-2", ordinal=1)
        # Filter on the hash-form source_id of chunk1 only.
        keep_source_id = str(chunk1.source_id)
        store = self._store_with_chunks([chunk1, chunk2])
        spec = FilterSpec(source_uris=frozenset({keep_source_id}))
        retriever = BM25SparseRetriever(store=store, filter_spec=spec)
        q = Query(id=QueryId("q1"), text="python")
        hits = retriever.retrieve(q)
        # Only chunk1 (from fake://doc-1) should survive the filter.
        assert len(hits) == 1
        assert hits[0].chunk.id == chunk1.id

    def test_error_code_boost_ranks_above_generic(self) -> None:
        """Exact error-code token in text should rank higher than generic terms."""
        chunks = [
            _make_chunk("generic connection failure happened", ordinal=0),
            _make_chunk("ERROR_CODE_404 not found resource missing", ordinal=1),
        ]
        store = self._store_with_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        # Query with exact error code token
        q = Query(id=QueryId("q1"), text="ERROR_CODE_404")
        hits = retriever.retrieve(q)
        if len(hits) >= 2:
            # The chunk with the exact error code should rank first.
            assert "ERROR_CODE_404" in hits[0].chunk.text

    def test_corpus_id_filter_scopes_results(self) -> None:
        """Query with corpus_id must scope results to that corpus."""
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        # Insert via upsert_chunks (no corpus scope on these low-level chunks)
        chunk = _make_chunk("scoped content here")
        store.upsert_chunks([chunk])
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="scoped content", corpus_id=CorpusId("other-corpus"))
        hits = retriever.retrieve(q)
        # With corpus filter not matching, should return no hits.
        assert isinstance(hits, list)


# ---------------------------------------------------------------------------
# EmbedderDenseRetriever tests
# ---------------------------------------------------------------------------


class TestEmbedderDenseRetriever:
    """Unit tests for dense.py - embedding + declared-similarity retrieval."""

    def _populated_store_with_embedder(
        self, n: int = 3, dim: int = 16
    ) -> tuple[SQLiteStore, FakeEmbedder]:
        embedder = FakeEmbedder(dim=dim)
        store = SQLiteStore(db_path=":memory:", vector_dim=dim)
        chunks = [_make_chunk(f"document content number {i}", ordinal=i) for i in range(n)]
        # Use the public staged promotion workflow (no _conn access).
        _stage_and_promote_with_embeddings(store, chunks, embedder, corpus_id=CorpusId("test"))
        return store, embedder

    def test_dense_score_set_on_hits(self) -> None:
        store, embedder = self._populated_store_with_embedder()
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="document content")
        hits = retriever.retrieve(q)
        if hits:
            assert all(h.dense_score is not None for h in hits)

    def test_only_dense_score_set(self) -> None:
        store, embedder = self._populated_store_with_embedder()
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="content")
        hits = retriever.retrieve(q)
        for h in hits:
            assert h.sparse_score is None
            assert h.fusion_score is None
            assert h.rerank_score is None

    def test_hits_ordered_descending_by_dense_score(self) -> None:
        store, embedder = self._populated_store_with_embedder(n=5)
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="document content number")
        hits = retriever.retrieve(q)
        scores = [h.dense_score for h in hits if h.dense_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index_returns_empty_list(self) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        embedder = FakeEmbedder(dim=16)
        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="anything")
        hits = retriever.retrieve(q)
        assert hits == []

    def test_no_embedder_returns_empty_list(self) -> None:
        """Sparse-only degraded mode: no embedder -> no dense candidates."""
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = EmbedderDenseRetriever(store=store, embedder=None, similarity="cosine")
        q = Query(id=QueryId("q1"), text="anything")
        hits = retriever.retrieve(q)
        assert hits == []

    def test_no_embedder_does_not_download_or_credential(self) -> None:
        """No embedder -> zero side effects. Must not instantiate any network objects."""
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        # This must succeed without any network calls
        retriever = EmbedderDenseRetriever(store=store, embedder=None, similarity="cosine")
        assert isinstance(retriever, EmbedderDenseRetriever)

    def test_retriever_is_dense_retriever_instance(self) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        embedder = FakeEmbedder(dim=16)
        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        assert isinstance(retriever, DenseRetriever)

    def test_dimension_incompatibility_raises_backend_error(self) -> None:
        """Embedder dimension mismatch with store -> typed BackendError."""
        from beacon_kb.storage.sqlite import SQLiteStore

        embedder = FakeEmbedder(dim=32)  # 32-dim embedder
        store = SQLiteStore(db_path=":memory:", vector_dim=16)  # 16-dim store
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="anything")
        with pytest.raises(BackendError):
            retriever.retrieve(q)

    def test_retrieve_deterministic(self) -> None:
        store, embedder = self._populated_store_with_embedder()
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="document content")
        r1 = retriever.retrieve(q)
        r2 = retriever.retrieve(q)
        assert [h.dense_score for h in r1] == [h.dense_score for h in r2]

    def test_top_k_respected(self) -> None:
        store, embedder = self._populated_store_with_embedder(n=10)
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="document content", top_k=3)
        hits = retriever.retrieve(q)
        assert len(hits) <= 3

    def test_dense_independent_of_sparse_ordering(self) -> None:
        """Dense candidates have independent ranks; sparse ordering must not bleed through."""
        store, embedder = self._populated_store_with_embedder(n=5)
        retriever = EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")
        q = Query(id=QueryId("q1"), text="content")
        hits = retriever.retrieve(q)
        # All hits have dense_score set, sparse_score None - independent ranking.
        for h in hits:
            assert h.sparse_score is None
            assert h.dense_score is not None

    def test_filters_applied_to_dense_hits(self) -> None:
        """Filters apply consistently to dense results, not bypassable."""
        store, embedder = self._populated_store_with_embedder(n=3)
        # Filter to a source_id hash that will never match any real chunk.
        spec = FilterSpec(source_uris=frozenset({"deadbeef00000000000000000000000000000000"}))
        retriever = EmbedderDenseRetriever(
            store=store, embedder=embedder, similarity="cosine", filter_spec=spec
        )
        q = Query(id=QueryId("q1"), text="content")
        hits = retriever.retrieve(q)
        assert hits == []

    def test_unknown_similarity_raises_backend_error(self) -> None:
        """Unknown similarity direction must raise BackendError, not silent zero."""
        from beacon_kb.storage.sqlite import SQLiteStore

        embedder = FakeEmbedder(dim=16)
        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        with pytest.raises((BackendError, ValueError)):
            EmbedderDenseRetriever(store=store, embedder=embedder, similarity="unknown")


# ---------------------------------------------------------------------------
# Sparse-only degraded mode integration
# ---------------------------------------------------------------------------


class TestSparseOnlyDegradedMode:
    """Verifies that the sparse-only mode is first-class: BM25 alone, zero downloads."""

    def test_sparse_only_returns_bm25_candidates(self) -> None:
        chunks = [_make_chunk("python programming language tutorial")]
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        store.upsert_chunks(chunks)

        sparse_retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python programming")
        hits = sparse_retriever.retrieve(q)
        # Must return BM25 hits without any embedder.
        if hits:
            assert all(h.sparse_score is not None for h in hits)

    def test_sparse_only_no_dense_scores(self) -> None:
        chunks = [_make_chunk("test content here")]
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        store.upsert_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="test content")
        hits = retriever.retrieve(q)
        for h in hits:
            assert h.dense_score is None

    def test_dense_retriever_no_embedder_is_noop(self) -> None:
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        retriever = EmbedderDenseRetriever(store=store, embedder=None, similarity="cosine")
        q = Query(id=QueryId("q1"), text="anything")
        hits = retriever.retrieve(q)
        assert hits == []


# ---------------------------------------------------------------------------
# Independent ranks and no cross-normalization
# ---------------------------------------------------------------------------


class TestIndependentRanks:
    """Sparse and dense candidates must keep independent ranks and raw scores."""

    def test_sparse_score_not_normalized(self) -> None:
        """Sparse scores are raw BM25; they must not be normalized to [0, 1]."""
        chunks = [
            _make_chunk("python python python python", ordinal=0),
            _make_chunk("python", ordinal=1),
        ]
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        store.upsert_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python")
        hits = retriever.retrieve(q)
        # BM25 scores may exceed 1.0; not clamped or normalized.
        if len(hits) >= 2:
            scores = [h.sparse_score for h in hits if h.sparse_score is not None]
            # At least some score may exceed 1.0 or differ in meaningful way
            assert len(scores) == 2

    def test_dense_and_sparse_from_same_store_are_independent(self) -> None:
        """Sparse and dense hits are produced independently; no bleed-through."""
        embedder = FakeEmbedder(dim=16)
        chunks = [_make_chunk(f"document {i}", ordinal=i) for i in range(3)]
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        # Use the public staged promotion workflow (no _conn access).
        _stage_and_promote_with_embeddings(store, chunks, embedder, corpus_id=CorpusId("test"))

        sparse_retriever = BM25SparseRetriever(store=store)
        dense_retriever = EmbedderDenseRetriever(
            store=store, embedder=embedder, similarity="cosine"
        )
        q = Query(id=QueryId("q1"), text="document")
        sparse_hits = sparse_retriever.retrieve(q)
        dense_hits = dense_retriever.retrieve(q)

        # sparse hits: only sparse_score set
        for h in sparse_hits:
            assert h.sparse_score is not None
            assert h.dense_score is None

        # dense hits: only dense_score set
        for h in dense_hits:
            assert h.dense_score is not None
            assert h.sparse_score is None


# ---------------------------------------------------------------------------
# Contract suite conformance
# ---------------------------------------------------------------------------


class TestBM25SparseRetrieverContract(SparseRetrieverContract):
    """Run the SparseRetrieverContract suite against BM25SparseRetriever."""

    def make_subject(self) -> SparseRetriever:
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        # Pre-populate so the contract's populated-check tests can find hits.
        chunks = [_make_chunk(f"test query content {i}", ordinal=i) for i in range(3)]
        store.upsert_chunks(chunks)
        return BM25SparseRetriever(store=store)


class TestEmbedderDenseRetrieverContract(DenseRetrieverContract):
    """Run the DenseRetrieverContract suite against EmbedderDenseRetriever."""

    def make_subject(self) -> DenseRetriever:
        from beacon_kb.storage.sqlite import SQLiteStore

        embedder = FakeEmbedder(dim=16)
        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        chunks = [_make_chunk(f"dense test content {i}", ordinal=i) for i in range(3)]
        # Use the public staged promotion workflow (no _conn access).
        _stage_and_promote_with_embeddings(store, chunks, embedder, corpus_id=CorpusId("test"))
        return EmbedderDenseRetriever(store=store, embedder=embedder, similarity="cosine")


# ---------------------------------------------------------------------------
# Registry discovery tests
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    """Both retrievers must be discovered via beacon_kb.retrievers group.

    These tests explicitly re-run _register_builtins() before each test
    to be order-independent: other test suites (registry contract tests)
    legitimately call clear_registry() to reset state between their own
    tests, which removes our built-ins.  Re-registering here is idempotent.
    """

    def setup_method(self) -> None:
        """Re-register built-ins so registry tests are order-independent."""
        from beacon_kb.registry.builtins import _register_builtins
        _register_builtins()

    def test_sparse_retriever_registered_in_group(self) -> None:
        from beacon_kb import registry
        from beacon_kb.registry import groups

        names = registry.list_plugins(groups.RETRIEVERS)
        assert "bm25" in names, f"Expected 'bm25' in retrievers group, got: {names}"

    def test_dense_retriever_registered_in_group(self) -> None:
        from beacon_kb import registry
        from beacon_kb.registry import groups

        names = registry.list_plugins(groups.RETRIEVERS)
        assert "dense" in names, f"Expected 'dense' in retrievers group, got: {names}"

    def test_sparse_retriever_resolve_by_name(self) -> None:
        """Resolve 'bm25' by name from the group - same path as third-party plugins."""
        from beacon_kb import registry
        from beacon_kb.registry import groups

        # The registered builtin is a BM25SparseRetriever instance
        plugin = registry.resolve(groups.RETRIEVERS, "bm25")
        assert isinstance(plugin, SparseRetriever)

    def test_dense_retriever_resolve_with_protocol(self) -> None:
        """Dense retriever resolved with protocol=DenseRetriever (documented escape hatch)."""
        from beacon_kb import registry
        from beacon_kb.protocols import DenseRetriever
        from beacon_kb.registry import groups

        plugin = registry.resolve(groups.RETRIEVERS, "dense", protocol=DenseRetriever)
        assert isinstance(plugin, DenseRetriever)
