"""Unit tests for generation.abstention - pre- and post-generation abstention policy."""

from __future__ import annotations

from beacon_kb.generation.abstention import (
    ABSTAIN_SENTINEL,
    is_post_abstain,
    should_pre_abstain,
)
from beacon_kb.models import Chunk, ChunkId, Hit, RevisionId, SectionId, SourceId


def _make_hit(
    *,
    sparse_score: float | None = None,
    dense_score: float | None = None,
    fusion_score: float | None = None,
    rerank_score: float | None = None,
    chunk_id: str = "c1",
) -> Hit:
    chunk = Chunk(
        id=ChunkId(chunk_id),
        source_id=SourceId("s"),
        revision_id=RevisionId("r"),
        section_id=SectionId("sec"),
        text="some content",
        ordinal=0,
        parent_locator="",
    )
    return Hit(
        chunk=chunk,
        sparse_score=sparse_score,
        dense_score=dense_score,
        fusion_score=fusion_score,
        rerank_score=rerank_score,
    )


class TestShouldPreAbstain:
    def test_empty_hits_always_abstains(self) -> None:
        assert should_pre_abstain([]) is True

    def test_empty_hits_abstains_regardless_of_threshold(self) -> None:
        assert should_pre_abstain([], abstain_threshold=0.0) is True
        assert should_pre_abstain([], abstain_threshold=0.5) is True
        assert should_pre_abstain([], abstain_threshold=1.0) is True

    def test_zero_threshold_never_abstains_with_hits(self) -> None:
        hit = _make_hit(sparse_score=0.001)
        assert should_pre_abstain([hit], abstain_threshold=0.0) is False

    def test_no_threshold_never_abstains_with_hits(self) -> None:
        hit = _make_hit(sparse_score=5.0)
        assert should_pre_abstain([hit]) is False

    def test_hit_above_threshold_does_not_abstain(self) -> None:
        hit = _make_hit(fusion_score=0.8)
        assert should_pre_abstain([hit], abstain_threshold=0.5) is False

    def test_all_hits_below_threshold_abstains(self) -> None:
        h1 = _make_hit(fusion_score=0.1, chunk_id="c1")
        h2 = _make_hit(fusion_score=0.2, chunk_id="c2")
        assert should_pre_abstain([h1, h2], abstain_threshold=0.5) is True

    def test_one_hit_above_threshold_no_abstain(self) -> None:
        h1 = _make_hit(fusion_score=0.1, chunk_id="c1")
        h2 = _make_hit(fusion_score=0.9, chunk_id="c2")
        assert should_pre_abstain([h1, h2], abstain_threshold=0.5) is False

    def test_score_priority_rerank_over_fusion(self) -> None:
        """rerank_score takes priority over fusion_score."""
        hit = _make_hit(rerank_score=0.9, fusion_score=0.1)
        # rerank says 0.9 >= 0.5 -> no abstain
        assert should_pre_abstain([hit], abstain_threshold=0.5) is False

    def test_score_priority_fusion_over_dense(self) -> None:
        hit = _make_hit(fusion_score=0.8, dense_score=0.1)
        assert should_pre_abstain([hit], abstain_threshold=0.5) is False

    def test_score_priority_dense_over_sparse(self) -> None:
        hit = _make_hit(dense_score=0.8, sparse_score=0.1)
        assert should_pre_abstain([hit], abstain_threshold=0.5) is False

    def test_hit_with_no_score_treated_as_zero(self) -> None:
        """Hits with no scores are treated as 0.0 - fails any threshold > 0."""
        hit = _make_hit()  # all scores None
        assert should_pre_abstain([hit], abstain_threshold=0.1) is True

    def test_hit_with_no_score_passes_zero_threshold(self) -> None:
        hit = _make_hit()
        assert should_pre_abstain([hit], abstain_threshold=0.0) is False


class TestIsPostAbstain:
    def test_empty_string_is_abstain(self) -> None:
        assert is_post_abstain("") is True

    def test_whitespace_only_is_abstain(self) -> None:
        assert is_post_abstain("   ") is True
        assert is_post_abstain("\n\t") is True

    def test_sentinel_string_is_abstain(self) -> None:
        assert is_post_abstain(ABSTAIN_SENTINEL) is True

    def test_sentinel_with_whitespace_is_abstain(self) -> None:
        assert is_post_abstain("  ABSTAIN  ") is True

    def test_normal_answer_is_not_abstain(self) -> None:
        assert is_post_abstain("The answer is 42 [S1].") is False

    def test_partial_sentinel_is_not_abstain(self) -> None:
        """Substring matches must NOT trigger abstention."""
        assert is_post_abstain("ABSTAINING from the subject") is False
        assert is_post_abstain("NOT ABSTAIN") is False

    def test_lowercase_abstain_is_not_abstain(self) -> None:
        """The sentinel is case-sensitive."""
        assert is_post_abstain("abstain") is False

    def test_abstain_sentinel_constant(self) -> None:
        assert ABSTAIN_SENTINEL == "ABSTAIN"
