"""Unit tests for the optional cross-encoder reranker.

Covers ordering, boundedness (never fetches new candidates), stability,
score attachment, and lazy-import behaviour, all with a fake scorer.
"""
from __future__ import annotations

import sys

import pytest

from beacon.errors import BackendError
from beacon.retrieval.hybrid import Hit
from beacon.retrieval.rerank import CrossEncoderReranker


class FakeScorer:
    """Deterministic scorer that returns pre-programmed scores in order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        return self._scores[: len(pairs)]


def _hit(point_id: str, text: str, fused: float) -> Hit:
    return Hit(
        chunk_point_id=point_id,
        payload={"chunk_text": text, "kind": "child"},
        fused_score=fused,
    )


def test_rerank_reorders_by_scorer_descending() -> None:
    hits = [
        _hit("a", "first text", 0.9),
        _hit("b", "second text", 0.8),
        _hit("c", "third text", 0.7),
    ]
    scorer = FakeScorer([0.1, 0.5, 0.9])
    reranker = CrossEncoderReranker(model_name="fake-model", scorer=scorer)

    reranked = reranker.rerank("query", hits)

    assert [h.chunk_point_id for h in reranked] == ["c", "b", "a"]
    assert [h.rerank_score for h in reranked] == [0.9, 0.5, 0.1]


def test_rerank_is_bounded_to_the_candidate_list() -> None:
    """Rerank never fetches new candidates: same ids in, same ids out."""
    hits = [_hit(pid, f"text {pid}", 1.0 - i / 10) for i, pid in enumerate("abcde")]
    scorer = FakeScorer([0.2, 0.4, 0.6, 0.8, 1.0])
    reranker = CrossEncoderReranker(model_name="fake-model", scorer=scorer)

    reranked = reranker.rerank("query", hits)

    assert len(reranked) == len(hits)
    assert {h.chunk_point_id for h in reranked} == {h.chunk_point_id for h in hits}
    assert len(scorer.calls) == 1
    assert scorer.calls[0] == [("query", h.payload["chunk_text"]) for h in hits]


def test_rerank_preserves_fused_score_and_attaches_rerank_score() -> None:
    hits = [_hit("a", "alpha", 0.42), _hit("b", "beta", 0.41)]
    scorer = FakeScorer([1.0, 2.0])
    reranker = CrossEncoderReranker(model_name="fake-model", scorer=scorer)

    reranked = reranker.rerank("query", hits)

    by_id = {h.chunk_point_id: h for h in reranked}
    assert by_id["a"].fused_score == 0.42
    assert by_id["a"].rerank_score == 1.0
    assert by_id["b"].fused_score == 0.41
    assert by_id["b"].rerank_score == 2.0


def test_rerank_is_stable_for_tied_scores() -> None:
    """Ties keep the fused (input) order: stable sort."""
    hits = [_hit("a", "one", 0.9), _hit("b", "two", 0.8), _hit("c", "three", 0.7)]
    scorer = FakeScorer([0.5, 0.5, 0.5])
    reranker = CrossEncoderReranker(model_name="fake-model", scorer=scorer)

    reranked = reranker.rerank("query", hits)

    assert [h.chunk_point_id for h in reranked] == ["a", "b", "c"]


def test_rerank_empty_hits_returns_empty_without_scoring() -> None:
    scorer = FakeScorer([])
    reranker = CrossEncoderReranker(model_name="fake-model", scorer=scorer)

    assert reranker.rerank("query", []) == []
    assert scorer.calls == []


def test_constructing_reranker_does_not_import_sentence_transformers() -> None:
    """The lazy import fires only on first use, never at construction."""
    CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert "sentence_transformers" not in sys.modules


def test_rerank_without_sentence_transformers_raises_backend_error() -> None:
    """With no injected scorer and no package installed, rerank fails typed."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed in this environment")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    with pytest.raises(BackendError) as excinfo:
        reranker.rerank("query", [_hit("a", "alpha", 0.9)])
    assert excinfo.value.kind.value == "backend"
