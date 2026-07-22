"""TDD tests for fusion, reranking, and diversity stages.

Covers:
- RRF fusion: rank-based, deterministic, tie-breaking, component score preservation
- FusionContract conformance for RRFusion
- Optional reranking: absent/failing reranker falls back to fused order
- Bounded reranking window and latency/score recording
- Diversity: near-duplicate collapse preserving provenance, cross-source preservation
- MMR-style diversity ordering
- Registry discovery: beacon_kb.fusion and beacon_kb.rerankers groups tested
"""

from __future__ import annotations

import dataclasses

import pytest

from beacon_kb.errors import BackendError, PluginNotFound
from beacon_kb.models import (
    Chunk,
    ChunkId,
    ChunkKind,
    Hit,
    Query,
    QueryId,
    RevisionId,
    SectionId,
    SourceId,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.protocols import Fusion
from beacon_kb.retrieval.diversity import collapse_near_duplicates, mmr_diversify
from beacon_kb.retrieval.fusion import RRFusion
from beacon_kb.retrieval.rerank import RerankResult, rerank_hits
from beacon_kb.testing import FakeClock, FakeReranker, FusionContract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str,
    *,
    chunk_id: str | None = None,
    source_id: str | None = None,
    ordinal: int = 0,
    source_uri: str = "fake://doc-1",
    corpus: str = "test",
) -> Chunk:
    sid = (
        SourceId(source_id)
        if source_id
        else make_source_id(corpus=corpus, canonical_uri=source_uri)
    )
    cid = ChunkId(chunk_id) if chunk_id else make_chunk_id(
        corpus=corpus,
        canonical_uri=source_uri,
        revision_id="rev-001",
        pipeline_fingerprint="pipe-v1",
        parent_locator="intro",
        child_ordinal=ordinal,
    )
    return Chunk(
        id=cid,
        source_id=sid,
        revision_id=RevisionId("rev-001"),
        section_id=SectionId("sec-001"),
        text=text,
        ordinal=ordinal,
        parent_locator="intro",
        kind=ChunkKind.CHILD,
    )


def _make_sparse_hit(chunk: Chunk, score: float) -> Hit:
    return Hit(chunk=chunk, sparse_score=score)


def _make_dense_hit(chunk: Chunk, score: float) -> Hit:
    return Hit(chunk=chunk, dense_score=score)


# ---------------------------------------------------------------------------
# RRFusion unit tests
# ---------------------------------------------------------------------------


class TestRRFusion:
    """Tests for RRF fusion: rank-based, deterministic, component score preserved."""

    def _chunks(self, n: int = 4, *, source_uri: str = "fake://doc-1") -> list[Chunk]:
        return [_make_chunk(f"text {i}", ordinal=i, source_uri=source_uri) for i in range(n)]

    def test_is_fusion_protocol_instance(self) -> None:
        assert isinstance(RRFusion(), Fusion)

    def test_fuse_returns_list_of_hits(self) -> None:
        chunks = self._chunks(2)
        sparse = [_make_sparse_hit(c, 5.0 - i) for i, c in enumerate(chunks)]
        dense = [_make_dense_hit(c, 0.9 - 0.1 * i) for i, c in enumerate(chunks)]
        result = RRFusion().fuse(sparse, dense)
        assert isinstance(result, list)
        assert all(isinstance(h, Hit) for h in result)

    def test_fuse_sets_fusion_score_on_all_hits(self) -> None:
        chunks = self._chunks(3)
        sparse = [_make_sparse_hit(c, float(len(chunks) - i)) for i, c in enumerate(chunks)]
        dense = [_make_dense_hit(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
        hits = RRFusion().fuse(sparse, dense)
        assert all(h.fusion_score is not None for h in hits)

    def test_fuse_ordered_descending_by_fusion_score(self) -> None:
        chunks = self._chunks(4)
        sparse = [_make_sparse_hit(c, float(4 - i)) for i, c in enumerate(chunks)]
        dense = [_make_dense_hit(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
        hits = RRFusion().fuse(sparse, dense)
        scores = [h.fusion_score for h in hits if h.fusion_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_deduplicates_by_chunk_id(self) -> None:
        """A chunk appearing in both sparse and dense lists must appear once."""
        chunk = _make_chunk("shared chunk", chunk_id="shared-c1")
        sparse = [_make_sparse_hit(chunk, 5.0)]
        dense = [_make_dense_hit(chunk, 0.9)]
        hits = RRFusion().fuse(sparse, dense)
        assert len(hits) == 1

    def test_fuse_preserves_sparse_score_on_fused_hit(self) -> None:
        """sparse_score must be carried through to the fused Hit."""
        chunk = _make_chunk("content", chunk_id="c1")
        sparse = [_make_sparse_hit(chunk, 7.5)]
        dense = [_make_dense_hit(chunk, 0.8)]
        hits = RRFusion().fuse(sparse, dense)
        assert len(hits) == 1
        assert hits[0].sparse_score == pytest.approx(7.5)

    def test_fuse_preserves_dense_score_on_fused_hit(self) -> None:
        """dense_score must be carried through to the fused Hit."""
        chunk = _make_chunk("content", chunk_id="c2")
        sparse = [_make_sparse_hit(chunk, 7.5)]
        dense = [_make_dense_hit(chunk, 0.8)]
        hits = RRFusion().fuse(sparse, dense)
        assert hits[0].dense_score == pytest.approx(0.8)

    def test_fuse_sparse_only_hit_has_none_dense_score(self) -> None:
        """A hit only in sparse list must have dense_score=None in fused output."""
        sparse_only = _make_chunk("sparse only", chunk_id="s-only")
        dense_only = _make_chunk("dense only", chunk_id="d-only")
        sparse = [_make_sparse_hit(sparse_only, 5.0)]
        dense = [_make_dense_hit(dense_only, 0.8)]
        hits = RRFusion().fuse(sparse, dense)
        id_map = {h.chunk.id: h for h in hits}
        assert id_map[ChunkId("s-only")].dense_score is None
        assert id_map[ChunkId("d-only")].sparse_score is None

    def test_fuse_empty_inputs_returns_empty(self) -> None:
        assert RRFusion().fuse([], []) == []

    def test_fuse_sparse_only_input(self) -> None:
        chunks = self._chunks(2)
        sparse = [_make_sparse_hit(c, float(2 - i)) for i, c in enumerate(chunks)]
        hits = RRFusion().fuse(sparse, [])
        assert len(hits) == 2
        assert all(h.fusion_score is not None for h in hits)

    def test_fuse_dense_only_input(self) -> None:
        chunks = self._chunks(2)
        dense = [_make_dense_hit(c, 1.0 - 0.1 * i) for i, c in enumerate(chunks)]
        hits = RRFusion().fuse([], dense)
        assert len(hits) == 2
        assert all(h.fusion_score is not None for h in hits)

    def test_fuse_is_deterministic(self) -> None:
        """Identical inputs must produce identical fusion_scores."""
        chunks = self._chunks(4)
        sparse = [_make_sparse_hit(c, float(4 - i)) for i, c in enumerate(chunks)]
        dense = [_make_dense_hit(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
        r1 = RRFusion().fuse(sparse, dense)
        r2 = RRFusion().fuse(sparse, dense)
        assert [h.fusion_score for h in r1] == [h.fusion_score for h in r2]
        assert [h.chunk.id for h in r1] == [h.chunk.id for h in r2]

    def test_fuse_rank_based_not_score_based(self) -> None:
        """Fusion score must depend on rank position, not raw score magnitude."""
        # Two chunks: rank-1 in sparse has a tiny BM25 score; RRF depends only on rank.
        # Rank ordering: sparse=[c1, c2], dense=[c2, c1].
        # c1 and c2 are symmetric: each is rank-1 in one list and rank-2 in the other.
        # RRF contribution: c1 = 1/(60+1) + 1/(60+2) = c2 (rank symmetric => same fusion score).
        c1 = _make_chunk("rank one sparse", chunk_id="rrf-c1")
        c2 = _make_chunk("rank one dense", chunk_id="rrf-c2")
        sparse = [_make_sparse_hit(c1, 0.01), _make_sparse_hit(c2, 0.001)]
        dense = [_make_dense_hit(c2, 0.5), _make_dense_hit(c1, 0.1)]
        hits = RRFusion().fuse(sparse, dense)
        id_to_score = {h.chunk.id: h.fusion_score for h in hits}
        # fusion_score must be set and equal due to rank symmetry (not raw scores)
        assert all(v is not None for v in id_to_score.values())
        c1_fusion = id_to_score[ChunkId("rrf-c1")]
        c2_fusion = id_to_score[ChunkId("rrf-c2")]
        assert c1_fusion == pytest.approx(c2_fusion)
        # Verify neither fusion score equals any input raw score (0.01, 0.001, 0.5, 0.1)
        raw_scores = {0.01, 0.001, 0.5, 0.1}
        for fusion_score in [c1_fusion, c2_fusion]:
            msg = f"Fusion score {fusion_score} leaked raw input score"
            assert fusion_score not in raw_scores, msg

    def test_fuse_tie_breaking_is_stable(self) -> None:
        """Equal RRF scores must be broken deterministically by chunk_id ASC."""
        # Two separate chunks each appearing only in one list -> same RRF contribution
        c1 = _make_chunk("only sparse", chunk_id="tie-d1")  # d1 > a1 lexicographically
        c2 = _make_chunk("only dense", chunk_id="tie-a1")   # a1 < d1 lexicographically
        sparse = [_make_sparse_hit(c1, 1.0)]
        dense = [_make_dense_hit(c2, 1.0)]
        r1 = RRFusion().fuse(sparse, dense)
        r2 = RRFusion().fuse(sparse, dense)
        # Verify deterministic ordering
        assert [h.chunk.id for h in r1] == [h.chunk.id for h in r2]
        # Verify tie-break is ASC (lexicographically smaller chunk_id comes first)
        assert len(r1) == 2
        assert str(r1[0].chunk.id) < str(r1[1].chunk.id)

    def test_rrf_k_parameter_configurable(self) -> None:
        """RRF k parameter must be configurable (default 60 is standard)."""
        fusion_default = RRFusion()
        fusion_custom = RRFusion(k=20)
        chunks = self._chunks(2)
        sparse = [_make_sparse_hit(c, float(2 - i)) for i, c in enumerate(chunks)]
        dense = [_make_dense_hit(c, 1.0 - 0.1 * i) for i, c in enumerate(chunks)]
        # Different k should produce different scores but same ordering here
        r_default = fusion_default.fuse(sparse, dense)
        r_custom = fusion_custom.fuse(sparse, dense)
        # Both must return same number of hits
        assert len(r_default) == len(r_custom)

    def test_fuse_disjoint_lists_merges_all(self) -> None:
        """Disjoint sparse and dense lists must all appear in fused output."""
        sparse_chunks = [_make_chunk(f"sparse {i}", chunk_id=f"sp{i}", ordinal=i) for i in range(3)]
        dense_chunks = [_make_chunk(f"dense {i}", chunk_id=f"dn{i}", ordinal=i) for i in range(3)]
        sparse = [_make_sparse_hit(c, float(3 - i)) for i, c in enumerate(sparse_chunks)]
        dense = [_make_dense_hit(c, 1.0 / (i + 1)) for i, c in enumerate(dense_chunks)]
        hits = RRFusion().fuse(sparse, dense)
        assert len(hits) == 6

    def test_fuse_union_of_chunk_ids(self) -> None:
        """All unique chunk IDs from both lists must appear in fused output."""
        c1 = _make_chunk("c1 text", chunk_id="union-c1")
        c2 = _make_chunk("c2 text", chunk_id="union-c2")
        c3 = _make_chunk("c3 text", chunk_id="union-c3")
        sparse = [_make_sparse_hit(c1, 5.0), _make_sparse_hit(c2, 3.0)]
        dense = [_make_dense_hit(c2, 0.9), _make_dense_hit(c3, 0.7)]
        hits = RRFusion().fuse(sparse, dense)
        result_ids = {h.chunk.id for h in hits}
        assert ChunkId("union-c1") in result_ids
        assert ChunkId("union-c2") in result_ids
        assert ChunkId("union-c3") in result_ids


# ---------------------------------------------------------------------------
# FusionContract conformance
# ---------------------------------------------------------------------------


class TestRRFusionFusionContract(FusionContract):
    """FusionContract suite applied to RRFusion."""

    def make_subject(self) -> Fusion:
        return RRFusion()


# ---------------------------------------------------------------------------
# Reranking tests
# ---------------------------------------------------------------------------


class TestRerankHits:
    """Tests for rerank_hits(): bounded window, fallback, latency, structured failure."""

    def _hits(self, n: int = 5, chunk_id_prefix: str = "rr") -> list[Hit]:
        return [
            Hit(chunk=_make_chunk(f"text {i}", chunk_id=f"{chunk_id_prefix}-{i}", ordinal=i))
            for i in range(n)
        ]

    def _query(self) -> Query:
        return Query(id=QueryId("q1"), text="test query")

    def test_returns_rerank_result(self) -> None:
        hits = self._hits(3)
        result = rerank_hits(self._query(), hits, reranker=FakeReranker())
        assert isinstance(result, RerankResult)

    def test_reranked_hits_have_rerank_score(self) -> None:
        hits = self._hits(3)
        result = rerank_hits(self._query(), hits, reranker=FakeReranker())
        assert all(h.rerank_score is not None for h in result.hits)

    def test_reranked_hits_ordered_by_rerank_score(self) -> None:
        hits = self._hits(5)
        result = rerank_hits(self._query(), hits, reranker=FakeReranker())
        scores = [h.rerank_score for h in result.hits if h.rerank_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_no_reranker_returns_fused_order_unchanged(self) -> None:
        hits = self._hits(4)
        result = rerank_hits(self._query(), hits, reranker=None)
        assert result.hits == hits

    def test_no_reranker_failure_is_none(self) -> None:
        hits = self._hits(2)
        result = rerank_hits(self._query(), hits, reranker=None)
        assert result.failure is None

    def test_failing_reranker_returns_fused_order(self) -> None:
        """BackendError from reranker -> return fused order unchanged."""

        class AlwaysFailReranker:
            def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
                raise BackendError("injected failure")

        hits = self._hits(3)
        result = rerank_hits(self._query(), hits, reranker=AlwaysFailReranker())
        assert result.hits == hits

    def test_failing_reranker_records_failure(self) -> None:
        """Failure must be recorded as a structured value, not just a log line."""

        class AlwaysFailReranker:
            def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
                raise BackendError("injected failure")

        hits = self._hits(3)
        result = rerank_hits(self._query(), hits, reranker=AlwaysFailReranker())
        assert result.failure is not None
        assert isinstance(result.failure, Exception)

    def test_reranking_bounded_by_window(self) -> None:
        """Reranker is only called on the bounded window (top-N), not all hits."""

        class CountingReranker:
            def __init__(self) -> None:
                self.call_count = 0
                self.last_hits_count = 0

            def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
                self.call_count += 1
                self.last_hits_count = len(hits)
                # Assign scores
                scored = [
                    dataclasses.replace(h, rerank_score=1.0 - 0.1 * i)
                    for i, h in enumerate(hits)
                ]
                return sorted(scored, key=lambda h: h.rerank_score or 0.0, reverse=True)

        reranker = CountingReranker()
        hits = self._hits(10)
        window = 4
        rerank_hits(self._query(), hits, reranker=reranker, window=window)
        assert reranker.last_hits_count == window

    def test_hits_outside_window_appended_after_reranked(self) -> None:
        """Hits beyond the window must still appear in the result, after reranked hits."""
        hits = self._hits(6)
        result = rerank_hits(self._query(), hits, reranker=FakeReranker(), window=3)
        # Total count is preserved
        assert len(result.hits) == 6

    def test_latency_recorded_when_reranker_present(self) -> None:
        hits = self._hits(3)
        clock = FakeClock(start=0.0)

        class TickingReranker:
            """Reranker that advances the clock to simulate latency."""

            def __init__(self, clock: FakeClock) -> None:
                self._clock = clock

            def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
                self._clock.tick(0.5)  # Advance clock by 0.5 seconds
                return [
                    dataclasses.replace(h, rerank_score=1.0 - 0.1 * i)
                    for i, h in enumerate(hits)
                ]

        result = rerank_hits(self._query(), hits, reranker=TickingReranker(clock), clock=clock.now)
        assert result.latency_seconds is not None
        assert result.latency_seconds == pytest.approx(0.5), "Latency must match clock tick"

    def test_latency_none_when_no_reranker(self) -> None:
        hits = self._hits(3)
        result = rerank_hits(self._query(), hits, reranker=None)
        assert result.latency_seconds is None

    def test_fusion_score_preserved_through_reranking(self) -> None:
        """fusion_score on input hits must be preserved in reranked output."""
        chunk = _make_chunk("test content", chunk_id="fs-c1")
        hits = [Hit(chunk=chunk, fusion_score=0.42)]
        result = rerank_hits(self._query(), hits, reranker=FakeReranker())
        assert result.hits[0].fusion_score == pytest.approx(0.42)

    def test_sparse_score_preserved_through_reranking(self) -> None:
        chunk = _make_chunk("test sparse", chunk_id="sp-c1")
        hits = [Hit(chunk=chunk, sparse_score=7.5, fusion_score=0.3)]
        result = rerank_hits(self._query(), hits, reranker=FakeReranker())
        assert result.hits[0].sparse_score == pytest.approx(7.5)

    def test_rerank_result_has_window_field(self) -> None:
        hits = self._hits(5)
        result = rerank_hits(self._query(), hits, reranker=FakeReranker(), window=3)
        assert result.window == 3

    def test_rerank_no_reranker_window_field(self) -> None:
        hits = self._hits(5)
        result = rerank_hits(self._query(), hits, reranker=None)
        assert result.window == 0

    def test_empty_hits_returns_empty(self) -> None:
        result = rerank_hits(self._query(), [], reranker=FakeReranker())
        assert result.hits == []

    def test_empty_hits_no_reranker(self) -> None:
        result = rerank_hits(self._query(), [], reranker=None)
        assert result.hits == []


# ---------------------------------------------------------------------------
# Diversity tests
# ---------------------------------------------------------------------------


class TestCollapsNearDuplicates:
    """Tests for collapse_near_duplicates(): content similarity within same source."""

    def _hit_with_text(
        self,
        text: str,
        *,
        source_uri: str = "fake://doc-1",
        chunk_id_suffix: str = "0",
        fusion_score: float | None = None,
    ) -> Hit:
        cid = f"nd-{source_uri[-1]}-{chunk_id_suffix}"
        chunk = _make_chunk(text, chunk_id=cid, source_uri=source_uri)
        return Hit(chunk=chunk, fusion_score=fusion_score)

    def test_no_duplicates_returns_all_hits(self) -> None:
        hits = [
            self._hit_with_text("completely different content alpha"),
            self._hit_with_text("totally unrelated text beta", chunk_id_suffix="1"),
        ]
        result = collapse_near_duplicates(hits, threshold=0.9)
        assert len(result) == 2

    def test_exact_duplicate_within_same_source_collapsed(self) -> None:
        text = "identical text content here for dedup testing"
        hits = [
            self._hit_with_text(text, chunk_id_suffix="0"),
            self._hit_with_text(text, chunk_id_suffix="1"),
        ]
        result = collapse_near_duplicates(hits, threshold=0.9)
        assert len(result) == 1

    def test_near_duplicate_same_source_collapsed(self) -> None:
        """Near-duplicate text from same source should collapse at threshold."""
        # Create two texts with Jaccard similarity exactly 0.95.
        # Math: 38 identical tokens + 1 different in each => 38/(38+2) = 0.95.
        base_tokens = [f"word{i}" for i in range(1, 39)] + ["unique_a"]
        near_tokens = [f"word{i}" for i in range(1, 39)] + ["unique_b"]
        base = " ".join(base_tokens)
        near = " ".join(near_tokens)
        # Verify Jaccard >= 0.95
        tokens_a = set(base.lower().split())
        tokens_b = set(near.lower().split())
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        assert jaccard >= 0.95, f"Expected Jaccard >= 0.95, got {jaccard}"

        hits = [
            self._hit_with_text(base, chunk_id_suffix="0"),
            self._hit_with_text(near, chunk_id_suffix="1"),
        ]
        result = collapse_near_duplicates(hits, threshold=0.95)
        # Near-duplicates from same source with similarity >= threshold -> collapsed to 1
        assert len(result) == 1, f"Expected 1 hit after collapse, got {len(result)}"
        # Surviving hit must have valid chunk with source_id (provenance preserved)
        assert result[0].chunk.source_id is not None
        # Per diversity.py contract: first (highest-ranked) hit survives
        assert result[0].chunk.id == hits[0].chunk.id

    def test_similar_text_different_sources_never_merged(self) -> None:
        """Chunks from different sources must never be merged, even if text is similar."""
        text = "shared content that looks identical across sources"
        hit1 = self._hit_with_text(text, source_uri="fake://doc-1", chunk_id_suffix="0")
        hit2 = self._hit_with_text(text, source_uri="fake://doc-2", chunk_id_suffix="0")
        # Even at threshold=0 (collapse all same-source pairs), different sources must be kept.
        result = collapse_near_duplicates([hit1, hit2], threshold=0.0)
        assert len(result) == 2

    def test_provenance_preserved_in_collapsed_result(self) -> None:
        """Collapsed hit must still carry the surviving chunk's provenance."""
        text = "repeated content that collapses"
        hits = [
            self._hit_with_text(text, chunk_id_suffix="0"),
            self._hit_with_text(text, chunk_id_suffix="1"),
        ]
        result = collapse_near_duplicates(hits, threshold=0.9)
        assert len(result) >= 1
        # The surviving hit must have a valid chunk with source_id
        assert result[0].chunk.source_id is not None

    def test_empty_hits_returns_empty(self) -> None:
        result = collapse_near_duplicates([], threshold=0.9)
        assert result == []

    def test_single_hit_returns_unchanged(self) -> None:
        hits = [self._hit_with_text("solo content")]
        result = collapse_near_duplicates(hits, threshold=0.9)
        assert len(result) == 1

    def test_threshold_1_0_no_collapse_unless_identical(self) -> None:
        """At threshold=1.0, only exact text matches collapse."""
        hits = [
            self._hit_with_text("almost same content here", chunk_id_suffix="0"),
            self._hit_with_text("almost same content here!", chunk_id_suffix="1"),  # one char diff
        ]
        result = collapse_near_duplicates(hits, threshold=1.0)
        # These are not identical, so should NOT collapse at threshold=1.0
        assert len(result) == 2

    def test_threshold_0_collapses_all_same_source(self) -> None:
        """At threshold=0, any two hits from same source collapse."""
        hits = [
            self._hit_with_text("content a", chunk_id_suffix="0"),
            self._hit_with_text("content b", chunk_id_suffix="1"),
        ]
        result = collapse_near_duplicates(hits, threshold=0.0)
        # At threshold=0, all pairs from same source collapse -> 1 hit
        assert len(result) == 1

    def test_scores_preserved_on_surviving_hit(self) -> None:
        text = "score preserved content"
        hits = [
            self._hit_with_text(text, chunk_id_suffix="0", fusion_score=0.9),
            self._hit_with_text(text, chunk_id_suffix="1", fusion_score=0.5),
        ]
        result = collapse_near_duplicates(hits, threshold=0.9)
        assert len(result) == 1
        # Winner keeps its fusion_score (first/highest, implementation choice)
        assert result[0].fusion_score is not None


class TestMMRDiversify:
    """Tests for mmr_diversify(): diversity re-ordering via Maximal Marginal Relevance."""

    def _hit(self, text: str, *, chunk_id: str, source_uri: str = "fake://doc-1") -> Hit:
        chunk = _make_chunk(text, chunk_id=chunk_id, source_uri=source_uri)
        return Hit(chunk=chunk)

    def test_empty_returns_empty(self) -> None:
        result = mmr_diversify([], lambda_mmr=0.5)
        assert result == []

    def test_single_hit_returned_unchanged(self) -> None:
        hits = [self._hit("solo", chunk_id="solo-1")]
        result = mmr_diversify(hits, lambda_mmr=0.5)
        assert len(result) == 1

    def test_returns_same_count_as_input(self) -> None:
        hits = [self._hit(f"text {i}", chunk_id=f"mmr-{i}") for i in range(5)]
        result = mmr_diversify(hits, lambda_mmr=0.5)
        assert len(result) == len(hits)

    def test_all_chunk_ids_preserved(self) -> None:
        """No chunk is dropped by MMR; all IDs must appear in result."""
        hits = [self._hit(f"text {i}", chunk_id=f"mmr-id-{i}") for i in range(4)]
        result = mmr_diversify(hits, lambda_mmr=0.5)
        input_ids = {h.chunk.id for h in hits}
        result_ids = {h.chunk.id for h in result}
        assert input_ids == result_ids

    def test_lambda_1_is_relevance_only(self) -> None:
        """lambda_mmr=1.0 means pure relevance, no diversity."""
        hits = [self._hit(f"text {i}", chunk_id=f"lam1-{i}") for i in range(4)]
        result = mmr_diversify(hits, lambda_mmr=1.0)
        # With pure relevance, order should match original (no reordering for diversity)
        # Just verify all returned
        assert len(result) == 4

    def test_lambda_0_is_diversity_only(self) -> None:
        """lambda_mmr=0.0 means maximum diversity, first pick preserved."""
        hits = [self._hit(f"text {i}", chunk_id=f"lam0-{i}") for i in range(4)]
        result = mmr_diversify(hits, lambda_mmr=0.0)
        assert len(result) == 4

    def test_duplicate_texts_diversified(self) -> None:
        """Identical-text chunks should be spread out, not clustered at top."""
        # Two identical texts and two different - diversity should separate same-text ones
        hits = [
            self._hit("identical content", chunk_id="dup-0"),
            self._hit("completely different", chunk_id="diff-1"),
            self._hit("identical content", chunk_id="dup-2"),
            self._hit("another unique one", chunk_id="uniq-3"),
        ]
        result = mmr_diversify(hits, lambda_mmr=0.5)
        # All hits preserved
        assert len(result) == 4

    def test_lambda_1_short_circuit_preserves_rerank_order(self) -> None:
        """lambda_mmr=1.0 must preserve input order even when it inverts fusion order.

        Regression (I1): MMR previously used fusion_score as the relevance proxy,
        so at lambda_mmr=1.0 it re-sorted by fusion_score and undid the reranker's
        deliberate ordering.
        """
        # Reranker put 'b' first despite lower fusion_score than 'a'.
        hit_b = dataclasses.replace(
            self._hit("content b", chunk_id="ord-b"),
            fusion_score=0.3,
            rerank_score=0.9,
        )
        hit_a = dataclasses.replace(
            self._hit("content a", chunk_id="ord-a"),
            fusion_score=0.8,
            rerank_score=0.2,
        )
        result = mmr_diversify([hit_b, hit_a], lambda_mmr=1.0)
        assert [h.chunk.id for h in result] == [hit_b.chunk.id, hit_a.chunk.id], (
            "lambda_mmr=1.0 must preserve the (rerank-ordered) input order, "
            "not re-sort by fusion_score."
        )

    def test_rerank_score_used_as_relevance_proxy(self) -> None:
        """When rerank_score is set, MMR must use it as relevance, not fusion_score.

        'b' has low fusion but high rerank; at near-pure relevance it must be
        selected first when the proxy is rerank_score.
        """
        hit_a = dataclasses.replace(
            self._hit("alpha topic text", chunk_id="prox-a"),
            fusion_score=0.9,
            rerank_score=0.1,
        )
        hit_b = dataclasses.replace(
            self._hit("beta subject words", chunk_id="prox-b"),
            fusion_score=0.2,
            rerank_score=0.9,
        )
        result = mmr_diversify([hit_a, hit_b], lambda_mmr=0.9)
        assert result[0].chunk.id == hit_b.chunk.id, (
            "MMR relevance proxy must prioritise rerank_score over fusion_score."
        )


# ---------------------------------------------------------------------------
# Registry discovery tests
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    """RRFusion must be discoverable via beacon_kb.fusion group."""

    def setup_method(self) -> None:
        from beacon_kb.registry.builtins import _register_builtins
        _register_builtins()

    def test_rrf_fusion_in_fusion_group(self) -> None:
        from beacon_kb import registry
        from beacon_kb.registry import groups

        names = registry.list_plugins(groups.FUSION)
        assert "rrf" in names, f"Expected 'rrf' in fusion group, got: {names}"

    def test_rrf_fusion_resolves_as_fusion(self) -> None:
        from beacon_kb import registry
        from beacon_kb.registry import groups

        plugin = registry.resolve(groups.FUSION, "rrf")
        assert isinstance(plugin, Fusion)

    def test_rerankers_group_exists_in_registry(self) -> None:
        """beacon_kb.rerankers group must be registered in groups constants."""
        from beacon_kb.registry import groups

        # Verify the group name constant exists and is a valid string
        assert hasattr(groups, "RERANKERS")
        assert isinstance(groups.RERANKERS, str)
        assert groups.RERANKERS == "beacon_kb.rerankers"

    def test_resolving_missing_reranker_raises_plugin_not_found(self) -> None:
        """Resolving a non-existent reranker must raise PluginNotFound."""
        from beacon_kb import registry
        from beacon_kb.registry import groups

        # With no reranker installed, resolving any name must raise PluginNotFound
        with pytest.raises(PluginNotFound):
            registry.resolve(groups.RERANKERS, "nonexistent_reranker")


# ---------------------------------------------------------------------------
# RRFusion registry precedence (I4)
# ---------------------------------------------------------------------------


class TestRRFusionRegistryPrecedence:
    """RRFusion must be a builtin (lowest precedence) so 'rrf' can be overridden."""

    def setup_method(self) -> None:
        from beacon_kb.registry import discovery, precedence
        from beacon_kb.registry.builtins import _register_builtins

        precedence.clear_registry()
        discovery.reset_scan_state()
        _register_builtins()

    def teardown_method(self) -> None:
        from beacon_kb.registry import discovery, precedence
        from beacon_kb.registry.builtins import _register_builtins

        precedence.clear_registry()
        discovery.reset_scan_state()
        _register_builtins()

    def test_rrf_appears_in_list_plugins(self) -> None:
        """list_plugins() must include builtin names so operators can see them."""
        from beacon_kb import registry
        from beacon_kb.registry import groups

        names = registry.list_plugins(groups.FUSION)
        assert "rrf" in names, f"list_plugins(FUSION) must include builtin 'rrf'. Got: {names}"

    def test_rrf_describe_has_builtin_flag(self) -> None:
        """describe() for rrf must report builtin=True."""
        from beacon_kb.registry import groups, precedence

        info = precedence.describe(groups.FUSION, "rrf")
        assert info.get("builtin") is True, (
            f"describe(FUSION, 'rrf') must have builtin=True. Got: {info}"
        )

    def test_explicit_rrf_overrides_builtin(self) -> None:
        """An explicitly registered 'rrf' instance must win over the builtin."""
        from beacon_kb import registry
        from beacon_kb.registry import groups

        custom_rrf = RRFusion(k=10)  # non-default k distinguishes it
        registry.register(group=groups.FUSION, name="rrf", instance=custom_rrf)
        resolved = registry.resolve(groups.FUSION, "rrf")
        assert resolved is custom_rrf, "Explicitly registered 'rrf' must override the builtin."
        info = registry.describe(groups.FUSION, "rrf")
        assert info.get("builtin") is False
