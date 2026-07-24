"""Optional cross-encoder reranking over fused candidates.

``CrossEncoderReranker`` satisfies the ``Reranker`` protocol from
``beacon.retrieval.hybrid``.  It is strictly bounded: it scores exactly the
``(query, chunk_text)`` pairs for the hits it is given and never fetches new
candidates.  Sorting is stable and fully deterministic given the same model
and inputs, so tied scores preserve the fused order.

sentence-transformers is imported lazily on first use only.  When reranking
is disabled (``HybridRetriever(reranker=None)``) this module's model path is
never touched and no import occurs.  A missing package at rerank time is a
typed ``BackendError``, not an ``ImportError`` leak.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol, cast

from beacon.errors import BackendError
from beacon.retrieval.hybrid import Hit

__all__ = ["CrossEncoderReranker", "Scorer"]


class Scorer(Protocol):
    """Scores ``(query, text)`` pairs; higher means more relevant."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score per pair, in input order."""
        ...


class _SentenceTransformersScorer:
    """Scorer backed by a sentence-transformers ``CrossEncoder`` model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                CrossEncoder,
            )
        except ImportError as exc:
            raise BackendError(
                "sentence-transformers is required for cross-encoder "
                "reranking. Install it with: pip install sentence-transformers"
            ) from exc
        try:
            self._model: Any = CrossEncoder(model_name)
        except Exception as exc:
            raise BackendError(
                f"Failed to load cross-encoder model {model_name!r}: {exc}"
            ) from exc

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch-score all pairs in one model call."""
        try:
            raw = self._model.predict(pairs)
        except Exception as exc:
            raise BackendError(f"Cross-encoder scoring failed: {exc}") from exc
        return [float(value) for value in raw]


class CrossEncoderReranker:
    """Bounded, deterministic reranker over an existing candidate list.

    Args:
        model_name: Cross-encoder model to load lazily on first use.
        scorer:     Optional injected scorer (tests use deterministic fakes);
                    when ``None`` a sentence-transformers scorer is created
                    lazily on the first ``rerank`` call.
    """

    def __init__(self, *, model_name: str, scorer: Scorer | None = None) -> None:
        self._model_name = model_name
        self._scorer = scorer

    def rerank(self, query_text: str, hits: Sequence[Hit]) -> list[Hit]:
        """Reorder ``hits`` by cross-encoder relevance, best first.

        Scores exactly one ``(query_text, chunk_text)`` pair per hit in a
        single batch, attaches ``rerank_score`` to each hit, and stable-sorts
        descending so ties keep the fused order.  The output is a permutation
        of the input: no candidate is added or dropped.

        Args:
            query_text: The user query.
            hits:       Fused candidates from the hybrid retriever.

        Returns:
            The same hits reordered, each with ``rerank_score`` set.

        Raises:
            BackendError: If the model cannot be imported, loaded, or scored.
        """
        if not hits:
            return []
        if self._scorer is None:
            self._scorer = _SentenceTransformersScorer(self._model_name)

        pairs = [
            (query_text, str(hit.payload.get("chunk_text", ""))) for hit in hits
        ]
        scores = self._scorer.score(pairs)
        scored = [
            replace(hit, rerank_score=score)
            for hit, score in zip(hits, scores, strict=True)
        ]
        # sorted() is stable: equal scores preserve the fused candidate order.
        # replace() guarantees rerank_score is always float, never None.
        return sorted(
            scored,
            key=lambda hit: cast(float, hit.rerank_score),
            reverse=True,
        )
