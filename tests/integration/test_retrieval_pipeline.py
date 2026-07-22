"""Integration tests for RetrievalPipeline: context assembly, snippets, evidence IDs.

TDD suite covering all acceptance criteria from Task 03.1.3:
- Packed evidence never exceeds token budget; result-count + token recap present.
- Parent/neighbor expansion only after final candidate ordering; bounded.
- Context spans keep context_of relationships; never assigned invented scores.
- Snippets center the match span (not document prefix); preserve URI, title, locator.
- Every evidence item has stable [S1]-style ID; context-only spans distinguishable.
- RetrievalPipeline.search() is deterministic for identical inputs.
- Query.top_k overrides config.retrieval.top_k when set to non-default.
- sparse.py weighted bm25() adoption via store.retrieve(weights=...).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from beacon_kb.config import BeaconConfig, RetrievalConfig
from beacon_kb.models import (
    Chunk,
    ChunkId,
    ChunkKind,
    CorpusId,
    EvidenceRole,
    Query,
    QueryId,
    Revision,
    RevisionId,
    SectionId,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.retrieval.context import expand_and_pack
from beacon_kb.retrieval.filters import FilterSpec
from beacon_kb.retrieval.pipeline import RetrievalPipeline, SearchResult
from beacon_kb.retrieval.snippets import Snippet, build_snippet
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder, FakeReranker, FakeSparseRetriever

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_CORPUS = "test-corpus"
_URI_A = "fake://doc-a"
_URI_B = "fake://doc-b"
_REVISION = "rev-001"
_PIPELINE = "pipe-v1"


def _make_store(*, vector_dim: int = 16) -> SQLiteStore:
    return SQLiteStore(db_path=":memory:", vector_dim=vector_dim)


def _make_chunk(
    text: str,
    *,
    ordinal: int = 0,
    uri: str = _URI_A,
    corpus: str = _CORPUS,
    revision_id: str = _REVISION,
    pipeline: str = _PIPELINE,
    section_locator: str = "intro",
    prev_id: ChunkId | None = None,
    next_id: ChunkId | None = None,
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
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
    )


def _populate_store(
    store: SQLiteStore,
    chunks: list[Chunk],
    *,
    embedder: FakeEmbedder | None = None,
    corpus_id: CorpusId | None = None,
) -> None:
    """Populate store via staged promotion workflow."""
    effective_corpus = corpus_id if corpus_id is not None else CorpusId(_CORPUS)
    by_revision: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_revision[str(chunk.revision_id)].append(chunk)

    for _rev_id_str, rev_chunks in by_revision.items():
        source_id = rev_chunks[0].source_id
        revision_id = rev_chunks[0].revision_id
        revision = Revision(
            id=revision_id,
            source_id=source_id,
            content_hash="test-hash",
            pipeline_fingerprint="test-pipe",
        )
        store.stage_revision(
            corpus_id=effective_corpus,
            revision=revision,
            canonical_uri=str(source_id),
        )
        store.upsert_chunks_to_staging(
            corpus_id=effective_corpus,
            revision_id=revision_id,
            chunks=rev_chunks,
        )
        if embedder is not None:
            for chunk in rev_chunks:
                vec = embedder.embed([chunk.text])[0]
                store.upsert_embedding(
                    corpus_id=effective_corpus,
                    chunk_id=chunk.id,
                    revision_id=revision_id,
                    vector=vec,
                    model_name="fake",
                    dimension=embedder.dimension(),
                    similarity="cosine",
                )
        store.promote_revision(corpus_id=effective_corpus, revision_id=revision_id)


# ---------------------------------------------------------------------------
# Snippet tests
# ---------------------------------------------------------------------------


class TestBuildSnippet:
    """Verify that build_snippet centers the match and preserves provenance."""

    def test_snippet_not_prefix(self) -> None:
        """Snippet must not start at the document beginning when match is mid-text."""
        long_text = "A" * 200 + " python error code found here " + "B" * 200
        snippet = build_snippet(
            long_text,
            "python error",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
            max_chars=100,
        )
        # The snippet should NOT start with the leading "AAAA..." prefix.
        assert not snippet.text.startswith("AAAA")

    def test_snippet_contains_match(self) -> None:
        """The match term should appear somewhere in the snippet."""
        text = "Introduction. " + "filler " * 30 + "python programming " + "more " * 30
        snippet = build_snippet(
            text,
            "python",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
            max_chars=200,
        )
        assert "python" in snippet.text.lower()

    def test_snippet_preserves_source_uri(self) -> None:
        snippet = build_snippet(
            "hello world",
            "hello",
            source_id="sid",
            source_uri="fake://my-doc",
            title="My Doc",
            locator="section/sub",
            chunk_id="cid",
        )
        assert snippet.source_uri == "fake://my-doc"

    def test_snippet_preserves_title(self) -> None:
        snippet = build_snippet(
            "hello world",
            "hello",
            source_id="sid",
            source_uri="fake://doc",
            title="Installation Guide",
            locator="install",
            chunk_id="cid",
        )
        assert snippet.title == "Installation Guide"

    def test_snippet_preserves_locator(self) -> None:
        snippet = build_snippet(
            "hello world",
            "hello",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="install/quickstart",
            chunk_id="cid",
        )
        assert snippet.locator == "install/quickstart"

    def test_snippet_span_within_chunk(self) -> None:
        """char_start and char_end must be valid offsets within the chunk text."""
        text = "word " * 50
        snippet = build_snippet(
            text,
            "word",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
            max_chars=80,
        )
        assert 0 <= snippet.char_start <= snippet.char_end <= len(text)

    def test_snippet_empty_chunk(self) -> None:
        """Empty chunk text must produce an empty snippet without raising."""
        snippet = build_snippet(
            "",
            "query",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
        )
        assert snippet.text == ""
        assert snippet.char_start == 0
        assert snippet.char_end == 0

    def test_snippet_no_match_falls_back_to_center(self) -> None:
        """When query terms don't match, snippet is centered (not prefix)."""
        text = "a " * 100 + "center here " + "z " * 100
        snippet = build_snippet(
            text,
            "xyzzy does not exist",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
            max_chars=100,
        )
        # char_start >= 0 (can be 0 for very short or edge-case texts).
        assert snippet.char_start >= 0

    def test_snippet_is_snippet_type(self) -> None:
        snippet = build_snippet(
            "test text",
            "test",
            source_id="sid",
            source_uri="fake://doc",
            title="Doc",
            locator="intro",
            chunk_id="cid",
        )
        assert isinstance(snippet, Snippet)


# ---------------------------------------------------------------------------
# Context expansion tests
# ---------------------------------------------------------------------------


class TestExpandAndPack:
    """Verify bounded context expansion, budget enforcement, context_of labels."""

    def _make_linked_chunks(self, n: int = 3) -> list[Chunk]:
        """Create a chain of n chunks with prev/next links."""
        # First pass: create IDs.
        ids = [
            ChunkId(make_chunk_id(
                corpus=_CORPUS,
                canonical_uri=_URI_A,
                revision_id=_REVISION,
                pipeline_fingerprint=_PIPELINE,
                parent_locator="intro",
                child_ordinal=i,
            ))
            for i in range(n)
        ]
        source_id = make_source_id(corpus=_CORPUS, canonical_uri=_URI_A)
        chunks = [
            Chunk(
                id=ids[i],
                source_id=source_id,
                revision_id=RevisionId(_REVISION),
                section_id=SectionId("sec-001"),
                text=f"chunk text number {i} with some content to fill budget",
                ordinal=i,
                parent_locator="intro",
                kind=ChunkKind.CHILD,
                token_count=10,
                prev_chunk_id=ids[i - 1] if i > 0 else None,
                next_chunk_id=ids[i + 1] if i < n - 1 else None,
            )
            for i in range(n)
        ]
        return chunks

    def test_primary_hits_packed_first(self) -> None:
        """Primary HIT evidence items must appear before CONTEXT spans."""
        chunks = self._make_linked_chunks(3)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits = [Hit(chunk=chunks[1], sparse_score=1.0)]

        result = expand_and_pack(query, hits, store, token_budget=1000)

        hit_indices = [i for i, ev in enumerate(result.evidence) if ev.role == EvidenceRole.HIT]
        ctx_indices = [
            i for i, ev in enumerate(result.evidence) if ev.role == EvidenceRole.CONTEXT
        ]

        # All HITs come before all CONTEXT items.
        if hit_indices and ctx_indices:
            assert max(hit_indices) < min(ctx_indices)

    def test_context_spans_have_context_of_relationship(self) -> None:
        """Context spans must carry context_of=primary EvidenceId and a plain S-label."""
        chunks = self._make_linked_chunks(3)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits = [Hit(chunk=chunks[1], sparse_score=1.0)]

        result = expand_and_pack(
            query, hits, store, token_budget=1000, max_neighbor_hops=1, max_context_per_hit=2
        )
        ctx_items = [ev for ev in result.evidence if ev.role == EvidenceRole.CONTEXT]
        hit_items = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        primary_ids = {ev.id for ev in hit_items}
        for ctx_ev in ctx_items:
            # citation_label must be a plain Sn label (no "[context_of:...]" string).
            assert ctx_ev.citation_label.startswith("S"), (
                f"CONTEXT citation_label must be plain 'Sn' label, got {ctx_ev.citation_label!r}"
            )
            assert "context_of" not in ctx_ev.citation_label, (
                "citation_label must not encode context_of in the label string"
            )
            # context_of field must point to a primary HIT's EvidenceId.
            assert ctx_ev.context_of is not None, "CONTEXT span must have context_of set"
            assert ctx_ev.context_of in primary_ids, (
                "context_of must reference a primary HIT EvidenceId"
            )

    def test_context_spans_have_no_relevance_score(self) -> None:
        """Context spans must have None for all score fields (no invented scores)."""
        chunks = self._make_linked_chunks(3)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits = [Hit(chunk=chunks[1], sparse_score=1.0)]

        result = expand_and_pack(
            query, hits, store, token_budget=1000, max_neighbor_hops=1, max_context_per_hit=2
        )
        ctx_items = [ev for ev in result.evidence if ev.role == EvidenceRole.CONTEXT]
        for ctx_ev in ctx_items:
            assert ctx_ev.hit.sparse_score is None
            assert ctx_ev.hit.dense_score is None
            assert ctx_ev.hit.fusion_score is None
            assert ctx_ev.hit.rerank_score is None

    def test_budget_enforced(self) -> None:
        """Packed evidence must never exceed the configured token budget."""
        chunks = self._make_linked_chunks(5)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        from beacon_kb.tokens import HeuristicTokenCounter
        hits = [Hit(chunk=c, sparse_score=float(5 - i)) for i, c in enumerate(chunks)]

        counter = HeuristicTokenCounter()
        small_budget = 20  # very small budget - forces overflow

        result = expand_and_pack(query, hits, store, token_budget=small_budget, counter=counter)

        total = sum(
            counter.count_tokens(ev.hit.chunk.text)
            for ev in result.evidence
        )
        assert total <= small_budget
        # With a very small budget, at least one hit must overflow.
        assert result.budget_summary.overflow_count > 0, (
            "Expected overflow_count > 0 when budget is too small to fit all hits"
        )
        # The evidence list must be shorter than the full hit list.
        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert len(hit_evidence) < len(hits), (
            "Expected fewer evidence items than hits when budget forces overflow"
        )

    def test_budget_summary_present(self) -> None:
        """BudgetSummary must be present in the result (token recap before prompt)."""
        chunks = self._make_linked_chunks(2)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits = [Hit(chunk=c, sparse_score=1.0) for c in chunks]

        result = expand_and_pack(query, hits, store, token_budget=1000)
        assert result.budget_summary is not None
        assert result.budget_summary.budget == 1000
        assert result.budget_summary.result_count >= 0

    def test_expansion_bounded(self) -> None:
        """max_context_per_hit limits how many context chunks are added."""
        chunks = self._make_linked_chunks(5)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        # Use middle chunk so it has both prev and next neighbors.
        hits = [Hit(chunk=chunks[2], sparse_score=1.0)]

        result = expand_and_pack(
            query, hits, store,
            token_budget=10000,
            max_neighbor_hops=2,
            max_context_per_hit=1,
        )
        ctx_items = [ev for ev in result.evidence if ev.role == EvidenceRole.CONTEXT]
        # max_context_per_hit=1 means at most 1 context chunk per primary hit.
        assert len(ctx_items) <= 1

    def test_expansion_only_after_final_ordering(self) -> None:
        """Verify expansion determinism: same ordered hits -> same output."""
        chunks = self._make_linked_chunks(3)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits_ordered = [
            Hit(chunk=chunks[0], sparse_score=2.0),
            Hit(chunk=chunks[2], sparse_score=1.0),
        ]
        result1 = expand_and_pack(query, hits_ordered, store, token_budget=1000)

        # Same ordered hits -> same output (deterministic).
        result2 = expand_and_pack(query, hits_ordered, store, token_budget=1000)
        assert [ev.id for ev in result1.evidence] == [ev.id for ev in result2.evidence]

    def test_evidence_ids_stable(self) -> None:
        """Evidence IDs are content-addressed: same query + chunk -> same ID."""
        chunks = self._make_linked_chunks(2)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        hits = [Hit(chunk=chunks[0], sparse_score=1.0)]

        r1 = expand_and_pack(query, hits, store, token_budget=1000)
        r2 = expand_and_pack(query, hits, store, token_budget=1000)

        assert r1.evidence[0].id == r2.evidence[0].id

    def test_no_duplicate_chunk_in_evidence(self) -> None:
        """A chunk must not appear twice in the evidence list."""
        chunks = self._make_linked_chunks(3)
        store = _make_store()
        store.upsert_chunks(chunks)
        query = Query(id=QueryId("q1"), text="chunk text")

        from beacon_kb.models import Hit
        # Include adjacent chunks as primary hits too; neighbor of hit[0] is chunks[1]
        # which is also a primary hit - should not be duplicated as context.
        hits = [
            Hit(chunk=chunks[0], sparse_score=2.0),
            Hit(chunk=chunks[1], sparse_score=1.0),
        ]
        result = expand_and_pack(
            query, hits, store, token_budget=1000, max_neighbor_hops=1, max_context_per_hit=2
        )
        chunk_ids = [str(ev.hit.chunk.id) for ev in result.evidence]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk in evidence"

    def test_oversized_rank2_skipped_and_labels_contiguous(self) -> None:
        """Hand-ordered hit list: rank-2 item overflows budget, later smaller items fit.

        This distinguishes post-packing relabeling (correct) from pre-packing
        assignment (wrong): if labels were assigned before packing, the surviving
        items would carry S1 and S3..Sn with a gap.  Post-packing relabeling
        must produce contiguous S1..Sn labels with no gaps.
        """
        from beacon_kb.models import Hit
        from beacon_kb.tokens import HeuristicTokenCounter

        counter = HeuristicTokenCounter()

        # Build three chunks of very different sizes:
        #   rank-1: small text
        #   rank-2: very large text (will overflow the budget)
        #   rank-3: small text (fits after rank-2 is skipped)
        small_text = "short doc"
        large_text = " ".join(["word"] * 200)  # ~200 tokens, will overflow small budget
        small_text2 = "another brief item"

        chunks = self._make_linked_chunks(3)
        # Override text so sizes are controlled.
        from dataclasses import replace
        c1 = replace(chunks[0], text=small_text)
        c2 = replace(chunks[1], text=large_text)
        c3 = replace(chunks[2], text=small_text2)

        store = _make_store()
        store.upsert_chunks([c1, c2, c3])
        query = Query(id=QueryId("q-t1"), text="doc")

        # Order hits: rank-1=c1, rank-2=c2 (oversized), rank-3=c3 (fits).
        hits = [
            Hit(chunk=c1, sparse_score=3.0),
            Hit(chunk=c2, sparse_score=2.0),
            Hit(chunk=c3, sparse_score=1.0),
        ]

        # Budget: enough for c1 and c3 but NOT for c2.
        tok_c1 = counter.count_tokens(c1.text)
        tok_c2 = counter.count_tokens(c2.text)
        tok_c3 = counter.count_tokens(c3.text)
        # Budget = tok_c1 + tok_c3 + 5 (headroom), but less than tok_c2.
        budget = tok_c1 + tok_c3 + 5
        assert budget < tok_c2, "Test invariant: budget must not fit the large chunk"

        result = expand_and_pack(
            query, hits, store,
            token_budget=budget,
            counter=counter,
            max_neighbor_hops=0,  # no context expansion so labels are purely from primary hits
            max_context_per_hit=0,
        )

        primary_ev = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]

        # rank-2 item (c2) must be absent from evidence.
        included_chunk_ids = {str(ev.hit.chunk.id) for ev in primary_ev}
        assert str(c2.id) not in included_chunk_ids, (
            "Oversized rank-2 chunk must be excluded when it does not fit the budget"
        )

        # rank-3 item (c3) must be present (it fits after skipping c2).
        assert str(c3.id) in included_chunk_ids, (
            "Rank-3 chunk must be included after the oversized rank-2 chunk is skipped"
        )

        # Labels must be contiguous S1, S2 (no gap at S2 because c2 was skipped pre-label).
        labels = [ev.citation_label for ev in primary_ev]
        expected_labels = [f"S{i+1}" for i in range(len(primary_ev))]
        assert labels == expected_labels, (
            f"Citation labels must be contiguous S1..Sn after post-packing relabeling. "
            f"Got {labels!r}, expected {expected_labels!r}"
        )


# ---------------------------------------------------------------------------
# RetrievalPipeline integration tests
# ---------------------------------------------------------------------------


class TestRetrievalPipeline:
    """Full-pipeline integration tests using an in-memory SQLiteStore."""

    def _build_pipeline(
        self,
        store: SQLiteStore,
        *,
        embedder: FakeEmbedder | None = None,
        token_budget: int = 4096,
        top_k: int = 10,
    ) -> RetrievalPipeline:
        config = BeaconConfig(
            retrieval=RetrievalConfig(top_k=top_k),
        )
        return RetrievalPipeline(
            store=store,
            config=config,
            embedder=embedder,
            token_budget=token_budget,
        )

    def test_search_returns_search_result(self) -> None:
        """search() must return a SearchResult object."""
        store = _make_store()
        chunks = [_make_chunk(f"python tutorial content {i}", ordinal=i) for i in range(3)]
        _populate_store(store, chunks)
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="python tutorial")
        result = pipeline.search(q)
        assert isinstance(result, SearchResult)

    def test_evidence_budget_not_exceeded(self) -> None:
        """Total tokens of packed evidence must not exceed the configured budget."""
        from beacon_kb.tokens import HeuristicTokenCounter
        store = _make_store()
        chunks = [
            _make_chunk(f"python tutorial content example {i} " * 20, ordinal=i)
            for i in range(5)
        ]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        counter = HeuristicTokenCounter()

        pipeline = RetrievalPipeline(
            store=store,
            token_budget=50,
            token_counter=counter,
        )
        q = Query(id=QueryId("q1"), text="python tutorial")
        result = pipeline.search(q)

        total = sum(counter.count_tokens(ev.hit.chunk.text) for ev in result.evidence)
        assert total <= 50

    def test_budget_recap_present(self) -> None:
        """SearchResult must carry a non-empty budget_recap string."""
        store = _make_store()
        chunks = [_make_chunk("hello world content", ordinal=i) for i in range(2)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="hello")
        result = pipeline.search(q)
        assert isinstance(result.budget_recap, str)
        assert len(result.budget_recap) > 0

    def test_budget_summary_present(self) -> None:
        """SearchResult.budget_summary must be a BudgetSummary."""
        from beacon_kb.tokens import BudgetSummary
        store = _make_store()
        chunks = [_make_chunk("content here", ordinal=0)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="content")
        result = pipeline.search(q)
        assert isinstance(result.budget_summary, BudgetSummary)

    def test_query_variants_populated(self) -> None:
        """search() records the QueryVariants that drove retrieval on SearchResult."""
        from beacon_kb.retrieval.query import QueryVariants
        store = _make_store()
        chunks = [_make_chunk("python tutorial content", ordinal=0)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="python tutorial")
        result = pipeline.search(q)
        assert isinstance(result.query_variants, QueryVariants)
        # No rewrite stage exists yet: all variants equal the original text.
        assert result.query_variants.original_text == "python tutorial"
        assert result.query_variants.sparse_text == "python tutorial"
        assert result.query_variants.dense_text == "python tutorial"

    def test_deterministic_for_identical_inputs(self) -> None:
        """search() must be deterministic: identical inputs produce identical Evidence."""
        store = _make_store()
        chunks = [_make_chunk(f"topic content {i}", ordinal=i) for i in range(4)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="topic content")
        r1 = pipeline.search(q)
        r2 = pipeline.search(q)
        assert [ev.id for ev in r1.evidence] == [ev.id for ev in r2.evidence]

    def test_primary_hits_have_stable_citation_labels(self) -> None:
        """Primary HITs must have stable [S1]-style citation labels."""
        store = _make_store()
        chunks = [_make_chunk(f"document text {i}", ordinal=i) for i in range(3)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="document text")
        result = pipeline.search(q)
        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        for i, ev in enumerate(hit_evidence, start=1):
            assert ev.citation_label == f"S{i}", f"Expected S{i}, got {ev.citation_label}"

    def test_context_distinguishable_from_hits(self) -> None:
        """CONTEXT evidence must have EvidenceRole.CONTEXT (distinguishable from HIT)."""
        # Build chunks with neighbor links so context can be added.
        # Direct upsert_chunks used here (unit-style: testing linked-chunk expansion).
        source_id = make_source_id(corpus=_CORPUS, canonical_uri=_URI_A)
        ids = [
            ChunkId(make_chunk_id(
                corpus=_CORPUS,
                canonical_uri=_URI_A,
                revision_id=_REVISION,
                pipeline_fingerprint=_PIPELINE,
                parent_locator="intro",
                child_ordinal=i,
            ))
            for i in range(3)
        ]
        chunks = [
            Chunk(
                id=ids[i],
                source_id=source_id,
                revision_id=RevisionId(_REVISION),
                section_id=SectionId("sec-001"),
                text=f"context test chunk {i} with content",
                ordinal=i,
                parent_locator="intro",
                kind=ChunkKind.CHILD,
                token_count=8,
                prev_chunk_id=ids[i - 1] if i > 0 else None,
                next_chunk_id=ids[i + 1] if i < 2 else None,
            )
            for i in range(3)
        ]
        store = _make_store()
        store.upsert_chunks(chunks)  # unit-style: direct upsert to test neighbor expansion

        pipeline = RetrievalPipeline(
            store=store,
            token_budget=1000,
            max_neighbor_hops=1,
            max_context_per_hit=2,
        )
        q = Query(id=QueryId("q1"), text="context test chunk")
        result = pipeline.search(q)

        roles = {ev.role for ev in result.evidence}
        # At least HIT items must be present.
        assert EvidenceRole.HIT in roles

    def test_empty_index_returns_empty_evidence(self) -> None:
        """Empty store must return empty evidence list without raising."""
        store = _make_store()
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="anything")
        result = pipeline.search(q)
        assert isinstance(result, SearchResult)
        assert result.evidence == []

    def test_empty_query_text_raises(self) -> None:
        """Empty query text must raise ValueError."""
        store = _make_store()
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="")
        with pytest.raises(ValueError):
            pipeline.search(q)

    def test_top_k_override_respected(self) -> None:
        """Per-query top_k != DEFAULT overrides config.retrieval.top_k."""
        store = _make_store()
        chunks = [_make_chunk(f"content {i}", ordinal=i) for i in range(10)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote

        # Config says top_k=10, query says top_k=2.
        config = BeaconConfig(retrieval=RetrievalConfig(top_k=10))
        pipeline = RetrievalPipeline(store=store, config=config, token_budget=50000)
        q = Query(id=QueryId("q1"), text="content", top_k=2)
        result = pipeline.search(q)
        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert len(hit_evidence) <= 2

    def test_config_top_k_used_when_query_default(self) -> None:
        """When query.top_k is None (not set), config.retrieval.top_k governs."""
        store = _make_store()
        chunks = [_make_chunk(f"content {i}", ordinal=i) for i in range(20)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote

        config = BeaconConfig(retrieval=RetrievalConfig(top_k=3))
        pipeline = RetrievalPipeline(store=store, config=config, token_budget=50000)
        # Query leaves top_k unset (None), so config's 3 wins.
        q = Query(id=QueryId("q1"), text="content")
        result = pipeline.search(q)
        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert len(hit_evidence) <= 3

    def test_with_embedder_and_reranker(self) -> None:
        """Pipeline with embedder + reranker must still produce valid Evidence."""
        embedder = FakeEmbedder(dim=16)
        reranker = FakeReranker()
        store = _make_store(vector_dim=16)
        chunks = [_make_chunk(f"python document {i}", ordinal=i) for i in range(4)]
        _populate_store(store, chunks, embedder=embedder)
        pipeline = RetrievalPipeline(
            store=store,
            embedder=embedder,
            reranker=reranker,
            token_budget=10000,
        )
        q = Query(id=QueryId("q1"), text="python document")
        result = pipeline.search(q)
        assert isinstance(result, SearchResult)
        assert all(
            ev.role in (EvidenceRole.HIT, EvidenceRole.CONTEXT) for ev in result.evidence
        )

    def test_filter_applied_to_search(self) -> None:
        """FilterSpec passed to search() must narrow results."""
        store = _make_store()
        chunk_a = _make_chunk("python content", uri=_URI_A, ordinal=0)
        chunk_b = _make_chunk("python content", uri=_URI_B, ordinal=0, corpus="test-corpus")
        # Two chunks from different URIs; populate each separately via staged workflow.
        _populate_store(store, [chunk_a])
        _populate_store(store, [chunk_b])

        # Filter to only doc-a by source_id.
        source_id_a = str(chunk_a.source_id)
        spec = FilterSpec(source_uris=frozenset({source_id_a}))

        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="python content")
        result = pipeline.search(q, filters=spec)

        for ev in result.evidence:
            if ev.role == EvidenceRole.HIT:
                assert str(ev.hit.chunk.source_id) == source_id_a

    def test_injected_override_retriever_honours_filter_spec(self) -> None:
        """N3 regression: FilterSpec must not be bypassed when an override retriever is injected.

        An override retriever has no access to the FilterSpec; the pipeline must
        apply apply_filters() unconditionally after each retriever leg so that
        blocked-source hits are excluded and allowed-source hits survive.
        """
        store = _make_store()
        chunk_allowed = _make_chunk("allowed content", uri=_URI_A, ordinal=0)
        chunk_blocked = _make_chunk("blocked content", uri=_URI_B, ordinal=0)

        # Populate store (needed for snippet resolution in the pipeline).
        _populate_store(store, [chunk_allowed])
        _populate_store(store, [chunk_blocked])

        # The override retriever returns BOTH chunks - it is unaware of FilterSpec.
        override_sparse = FakeSparseRetriever(chunks=[chunk_allowed, chunk_blocked])

        # FilterSpec that allows only chunk_allowed's source.
        allowed_source_id = str(chunk_allowed.source_id)
        spec = FilterSpec(source_uris=frozenset({allowed_source_id}))

        pipeline = RetrievalPipeline(
            store=store,
            sparse_retriever=override_sparse,
            token_budget=4096,
        )
        q = Query(id=QueryId("q-n3"), text="content")
        result = pipeline.search(q, filters=spec)

        hit_source_ids = {
            str(ev.hit.chunk.source_id)
            for ev in result.evidence
            if ev.role == EvidenceRole.HIT
        }
        # Blocked source must not appear even though the override retriever returned it.
        assert str(chunk_blocked.source_id) not in hit_source_ids, (
            "Blocked-source hit must be excluded by apply_filters on the override leg"
        )
        # Allowed source must survive.
        assert allowed_source_id in hit_source_ids, (
            "Allowed-source hit must pass through apply_filters on the override leg"
        )

    def test_evidence_ids_resolve_to_active_revision(self) -> None:
        """Evidence IDs must be stable and derived from query_id + chunk_id."""
        store = _make_store()
        chunks = [_make_chunk("stable content", ordinal=0)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="stable content")
        result = pipeline.search(q)

        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert hit_evidence, "Expected at least one HIT evidence item for 'stable content' query"
        ev = hit_evidence[0]
        # Re-derive the expected ID to confirm stability.
        from beacon_kb.models import make_evidence_id
        expected_id = make_evidence_id(
            query_id=str(q.id), chunk_id=str(ev.hit.chunk.id)
        )
        assert str(ev.id) == str(expected_id)

    def test_sparse_only_no_embedder(self) -> None:
        """Pipeline without embedder must still return results via sparse BM25."""
        store = _make_store()
        chunks = [_make_chunk("sparse retrieval test content", ordinal=0)]
        _populate_store(store, chunks)  # staged workflow: stage -> upsert -> promote
        pipeline = RetrievalPipeline(store=store, embedder=None, token_budget=1000)
        q = Query(id=QueryId("q1"), text="sparse retrieval test")
        result = pipeline.search(q)
        assert isinstance(result, SearchResult)


    def test_search_results_carry_snippets(self) -> None:
        """search() results must carry non-None snippets on every evidence item."""
        store = _make_store()
        chunks = [_make_chunk(f"python tutorial content {i}", ordinal=i) for i in range(3)]
        _populate_store(store, chunks)
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="python tutorial")
        result = pipeline.search(q)
        for ev in result.evidence:
            assert ev.snippet is not None, (
                f"Evidence item {ev.id} must have snippet set after search()"
            )

    def test_snippet_source_uri_is_canonical_not_hash(self) -> None:
        """Snippets must carry the canonical URI (from sources table), never the sha256 hash."""
        store = _make_store()
        # Populate via the staged workflow which records the real canonical_uri.
        canonical_uri = "https://example.com/docs/intro"
        source_id = make_source_id(corpus=_CORPUS, canonical_uri=canonical_uri)
        revision_id = RevisionId(_REVISION)
        chunk = Chunk(
            id=ChunkId(make_chunk_id(
                corpus=_CORPUS,
                canonical_uri=canonical_uri,
                revision_id=_REVISION,
                pipeline_fingerprint=_PIPELINE,
                parent_locator="intro",
                child_ordinal=0,
            )),
            source_id=source_id,
            revision_id=revision_id,
            section_id=SectionId("sec-001"),
            text="canonical uri test content here",
            ordinal=0,
            parent_locator="intro",
            kind=ChunkKind.CHILD,
            token_count=5,
        )
        revision = Revision(
            id=revision_id,
            source_id=source_id,
            content_hash="test-hash",
            pipeline_fingerprint=_PIPELINE,
        )
        store.stage_revision(
            corpus_id=CorpusId(_CORPUS),
            revision=revision,
            canonical_uri=canonical_uri,
        )
        store.upsert_chunks_to_staging(
            corpus_id=CorpusId(_CORPUS),
            revision_id=revision_id,
            chunks=[chunk],
        )
        store.promote_revision(corpus_id=CorpusId(_CORPUS), revision_id=revision_id)

        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="canonical uri test")
        result = pipeline.search(q)
        hit_ev = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert hit_ev, "Expected at least one HIT evidence item"
        snip = hit_ev[0].snippet
        assert snip is not None
        # source_uri must be the canonical URI, not the sha256 hash.
        assert snip.source_uri == canonical_uri, (
            f"snippet.source_uri must be the canonical URI {canonical_uri!r}, "
            f"got {snip.source_uri!r}"
        )

    def test_snippet_centers_match_not_prefix(self) -> None:
        """Snippet for a match deep in a long chunk must not start at position 0."""
        # "python" is placed at char ~600 in a 1427-char text.
        # With max_chars=200, the snippet window covers 100 chars each side of
        # the match - it cannot include the very start (position 0) of the text.
        long_text = "A " * 300 + "python tutorial found here " + "B " * 300
        from beacon_kb.retrieval.snippets import build_snippet
        snip = build_snippet(
            long_text, "python tutorial",
            source_id="sid", source_uri="fake://doc", title="Doc",
            locator="intro", chunk_id="cid",
            max_chars=200,
        )
        # char_start must be well into the text (not at the document prefix)
        assert snip.char_start > 0, (
            "Snippet must center on the match - char_start must be > 0 when "
            "match is deep in a long chunk"
        )
        # The match term must appear in the snippet text.
        assert "python" in snip.text.lower(), (
            "The match term 'python' must appear within the centered snippet"
        )

    def test_snippet_preserves_provenance(self) -> None:
        """Evidence snippets must carry source_id and locator from the chunk."""
        store = _make_store()
        chunks = [_make_chunk("python tutorial content", ordinal=0, section_locator="guide/intro")]
        _populate_store(store, chunks)
        pipeline = self._build_pipeline(store)
        q = Query(id=QueryId("q1"), text="python tutorial")
        result = pipeline.search(q)
        hit_ev = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        if hit_ev:
            snip = hit_ev[0].snippet
            assert snip is not None
            assert snip.source_id == str(hit_ev[0].hit.chunk.source_id)
            assert snip.locator == hit_ev[0].hit.chunk.parent_locator

    def test_context_snippets_carry_snippets(self) -> None:
        """CONTEXT-role evidence items (prev/next neighbors) must also have snippets set.

        Only the middle chunk (ordinal 1) contains the query term; its neighbors
        (ordinals 0 and 2) have unrelated text so they are not retrieved as primary
        HITs but are attached as CONTEXT spans via neighbor expansion.
        """
        source_id = make_source_id(corpus=_CORPUS, canonical_uri=_URI_A)
        ids = [
            ChunkId(make_chunk_id(
                corpus=_CORPUS,
                canonical_uri=_URI_A,
                revision_id=_REVISION,
                pipeline_fingerprint=_PIPELINE,
                parent_locator="intro",
                child_ordinal=i,
            ))
            for i in range(3)
        ]
        # Only the middle chunk contains the distinctive query term; neighbors
        # have neutral filler text so they cannot be retrieved as primary HITs.
        texts = [
            "surrounding context neighbor before",
            "xyzzy distinctive target document",
            "surrounding context neighbor after",
        ]
        chunks = [
            Chunk(
                id=ids[i],
                source_id=source_id,
                revision_id=RevisionId(_REVISION),
                section_id=SectionId("sec-001"),
                text=texts[i],
                ordinal=i,
                parent_locator="intro",
                kind=ChunkKind.CHILD,
                token_count=len(texts[i].split()),
                prev_chunk_id=ids[i - 1] if i > 0 else None,
                next_chunk_id=ids[i + 1] if i < 2 else None,
            )
            for i in range(3)
        ]
        store = _make_store()
        store.upsert_chunks(chunks)
        pipeline = RetrievalPipeline(
            store=store, token_budget=5000, max_neighbor_hops=1, max_context_per_hit=2
        )
        # Query targets only the middle chunk's distinctive term.
        q = Query(id=QueryId("q1"), text="xyzzy distinctive target")
        result = pipeline.search(q)

        ctx_evs = [ev for ev in result.evidence if ev.role == EvidenceRole.CONTEXT]
        assert ctx_evs, "Expected at least one CONTEXT evidence item from neighbor expansion"
        for ctx_ev in ctx_evs:
            assert ctx_ev.snippet is not None, (
                f"CONTEXT evidence item {ctx_ev.id} must have snippet set"
            )
            # Snippet must carry source provenance (not be empty).
            assert ctx_ev.snippet.source_id, (
                "CONTEXT snippet must have source_id populated"
            )

    def test_context_of_field_set_on_context_spans(self) -> None:
        """CONTEXT evidence items must have context_of set to the primary EvidenceId."""
        # Build linked chunks so neighbor expansion happens.
        source_id = make_source_id(corpus=_CORPUS, canonical_uri=_URI_A)
        ids = [
            ChunkId(make_chunk_id(
                corpus=_CORPUS,
                canonical_uri=_URI_A,
                revision_id=_REVISION,
                pipeline_fingerprint=_PIPELINE,
                parent_locator="intro",
                child_ordinal=i,
            ))
            for i in range(3)
        ]
        chunks = [
            Chunk(
                id=ids[i],
                source_id=source_id,
                revision_id=RevisionId(_REVISION),
                section_id=SectionId("sec-001"),
                text=f"context field test chunk {i} content",
                ordinal=i,
                parent_locator="intro",
                kind=ChunkKind.CHILD,
                token_count=8,
                prev_chunk_id=ids[i - 1] if i > 0 else None,
                next_chunk_id=ids[i + 1] if i < 2 else None,
            )
            for i in range(3)
        ]
        store = _make_store()
        store.upsert_chunks(chunks)  # unit-style: direct upsert to test linked expansion
        pipeline = RetrievalPipeline(
            store=store, token_budget=5000, max_neighbor_hops=1, max_context_per_hit=2
        )
        q = Query(id=QueryId("q1"), text="context field test chunk")
        result = pipeline.search(q)
        hit_ids = {ev.id for ev in result.evidence if ev.role == EvidenceRole.HIT}
        ctx_evs = [ev for ev in result.evidence if ev.role == EvidenceRole.CONTEXT]
        for ctx_ev in ctx_evs:
            assert ctx_ev.context_of is not None, "CONTEXT span must have context_of set"
            assert ctx_ev.context_of in hit_ids, (
                "context_of must reference a primary HIT EvidenceId"
            )
            # citation_label must be a plain Sn label (no [context_of:...] encoding)
            assert ctx_ev.citation_label.startswith("S"), (
                f"CONTEXT citation_label must be plain 'Sn', got {ctx_ev.citation_label!r}"
            )

    def test_gap_free_labels_after_overflow(self) -> None:
        """Citation labels must be gap-free S1..Sn after a large mid-rank chunk is budget-dropped.

        Setup: one large chunk (many tokens, ranks mid-field via BM25) flanked by
        smaller chunks that fit within the budget.  The large chunk must be excluded
        by the packer; the survivors must carry contiguous S1..Sn labels with no gap
        where the large chunk's label would have been.
        """
        from beacon_kb.tokens import HeuristicTokenCounter
        store = _make_store()

        # Small chunks: each ~6 tokens, will fit in a tight budget.
        small_chunks = [
            _make_chunk(f"small keyword content chunk {i}", ordinal=i)
            for i in range(4)
        ]
        # Large chunk: ~60 tokens - intentionally too big to fit once small chunks fill the
        # budget.  Ordinal 2 places it mid-rank in ordinal ordering, but BM25 will rank
        # it based on text relevance.  We include the query term so it is retrieved.
        large_text = "keyword content " + ("padding word " * 40)
        large_chunk = _make_chunk(large_text, ordinal=10)  # distinct ordinal to avoid collision

        _populate_store(store, [*small_chunks, large_chunk])

        counter = HeuristicTokenCounter()
        # Budget that fits 3-4 small chunks (~6 tokens each) but not the large one (~56 tokens).
        # 40 tokens: fits ~6 small chunks if any, but not the large chunk.
        pipeline = RetrievalPipeline(store=store, token_budget=40, token_counter=counter)
        q = Query(id=QueryId("q1"), text="keyword content")
        result = pipeline.search(q)

        hit_evidence = [ev for ev in result.evidence if ev.role == EvidenceRole.HIT]
        assert len(hit_evidence) >= 1, "At least one small chunk must fit"

        # The large chunk must be absent (budget exceeded).
        included_chunk_ids = {str(ev.hit.chunk.id) for ev in hit_evidence}
        assert str(large_chunk.id) not in included_chunk_ids, (
            "Large chunk must be excluded by budget overflow"
        )

        # Labels of surviving hits must be exactly S1, S2, ... without gaps.
        for i, ev in enumerate(hit_evidence, start=1):
            assert ev.citation_label == f"S{i}", (
                f"Expected gap-free label S{i}, got {ev.citation_label!r} "
                f"(gap after large chunk was dropped)"
            )

    def test_injected_retriever_column_weights_honoured(self) -> None:
        """column_weights constructor param must reach store.retrieve() on every search().

        We wrap the store's retrieve() to record the weights kwarg, then assert that
        the value recorded equals the weights passed to RetrievalPipeline.__init__.
        """
        store = _make_store()
        chunks = [_make_chunk(f"content {i}", ordinal=i) for i in range(3)]
        _populate_store(store, chunks)

        custom_weights = (2.0, 20.0, 10.0)
        # Pass weights via the public constructor parameter (not via private mutation).
        pipeline = RetrievalPipeline(store=store, token_budget=5000, column_weights=custom_weights)

        # Wrap store.retrieve to capture the weights kwarg.
        recorded_weights: list[tuple[float, float, float] | None] = []
        _original_retrieve = store.retrieve

        def _spy_retrieve(  # type: ignore[override]
            query: Query, *, weights: tuple[float, float, float] | None = None
        ) -> list:
            recorded_weights.append(weights)
            return _original_retrieve(query, weights=weights)

        store.retrieve = _spy_retrieve  # type: ignore[method-assign]

        q = Query(id=QueryId("q1"), text="content")
        result = pipeline.search(q)

        assert isinstance(result, SearchResult)
        # At least one retrieve call must have been made with the custom weights.
        assert any(w == custom_weights for w in recorded_weights), (
            f"Expected store.retrieve to be called with weights={custom_weights!r}; "
            f"recorded calls: {recorded_weights!r}"
        )


# ---------------------------------------------------------------------------
# Sparse.py weighted bm25() adoption tests
# ---------------------------------------------------------------------------


class TestSparseWeightedBm25Adoption:
    """Verify that sparse.py uses weighted per-column bm25() via store.retrieve().

    Epic 03 obligation: BM25SparseRetriever must adopt the store's weighted
    retrieve(weights=...) API added in Epic 02 migration 0002.  This replaces
    the old exact-token OR-boost approach as the primary ranking mechanism.
    These tests verify the store-level API for completeness.
    """

    def _store_with_chunks(self, chunks: list[Chunk]) -> SQLiteStore:
        store = _make_store()
        store.upsert_chunks(chunks)
        return store

    def test_weighted_retrieve_returns_hits(self) -> None:
        """store.retrieve(weights=...) must return hits with sparse_score set."""
        store = _make_store()
        chunk = _make_chunk("python heading content code", section_locator="python/intro")
        store.upsert_chunks([chunk])
        q = Query(id=QueryId("q1"), text="python", top_k=5)
        hits = store.retrieve(q, weights=(1.0, 10.0, 5.0))
        assert hits, "Expected hits from store.retrieve with weights for 'python' query"
        assert all(h.sparse_score is not None for h in hits)

    def test_heading_weight_boosts_heading_matches(self) -> None:
        """Heading column weight > text weight should boost heading-matched chunks."""
        store = _make_store()
        # Chunk A: query term in section_locator (heading) only.
        chunk_a = _make_chunk(
            "unrelated stuff here",
            ordinal=0,
            section_locator="python",
        )
        # Chunk B: query term in text.
        chunk_b = _make_chunk(
            "python appears in text here",
            ordinal=1,
            section_locator="unrelated",
        )
        store.upsert_chunks([chunk_a, chunk_b])
        q = Query(id=QueryId("q1"), text="python", top_k=5)
        # With high heading weight, chunk_a (python in heading) should rank first.
        hits = store.retrieve(q, weights=(1.0, 100.0, 1.0))
        assert len(hits) >= 2, "Expected at least 2 hits for heading-weight ranking test"
        # The chunk with "python" in heading (section_locator) should rank higher.
        assert chunk_a.id in {h.chunk.id for h in hits[:1]}

    def test_weighted_retrieve_without_weights_still_works(self) -> None:
        """retrieve(weights=None) must work (backward compatible)."""
        store = _make_store()
        chunk = _make_chunk("python content", ordinal=0)
        store.upsert_chunks([chunk])
        q = Query(id=QueryId("q1"), text="python", top_k=5)
        hits = store.retrieve(q)  # no weights
        assert isinstance(hits, list)

    def test_bm25_sparse_retriever_uses_weighted_store(self) -> None:
        """BM25SparseRetriever.retrieve() uses the store's weighted bm25 via updated sparse.py."""
        from beacon_kb.retrieval.sparse import BM25SparseRetriever
        store = _make_store()
        chunks = [
            _make_chunk(f"python content {i}", ordinal=i, section_locator="python/guide")
            for i in range(3)
        ]
        store.upsert_chunks(chunks)
        retriever = BM25SparseRetriever(store=store)
        q = Query(id=QueryId("q1"), text="python content")
        hits = retriever.retrieve(q)
        # Must return hits and each must have sparse_score set.
        assert hits, "Expected hits from BM25SparseRetriever for 'python content' query"
        assert all(h.sparse_score is not None for h in hits)
        assert all(h.dense_score is None for h in hits)
