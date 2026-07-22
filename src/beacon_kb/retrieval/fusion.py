"""Reciprocal Rank Fusion (RRF) for combining sparse and dense hit lists.

Design rules enforced here:
- Rank-based fusion only: raw BM25 and cosine scores are diagnostics, never inputs.
- Deterministic tie-breaking: equal RRF scores are broken by chunk_id (lexicographic),
  so identical inputs always produce identical output ordering across processes.
- Component scores preserved: sparse_score and dense_score from input hits are carried
  through to the fused Hit unchanged so callers can inspect the original signals.
- fusion_score is set on every returned Hit; rerank_score is always None (set later).
- Empty inputs are legal and return an empty list.

RRF formula: score(d) = sum_{r in ranklists} 1 / (k + rank(d, r))
  where rank(d, r) is 1-based position in ranked list r (not in list -> contributes 0).

Tie-break key: (fusion_score DESC, chunk_id ASC) - fully deterministic.

Importing this module performs no side effects.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from beacon_kb.models import Hit


class RRFusion:
    """Reciprocal Rank Fusion combining sparse and dense hit lists.

    Score direction: fusion_score is higher for more relevant results (range > 0,
    bounded above by 2/k for a chunk that is rank-1 in both lists).

    Determinism: identical inputs always produce identical ordering.
    Tie-break: secondary sort by chunk_id ASC (lexicographic) after fusion_score DESC.

    Parameters:
        k: Smoothing constant (default 60, standard in RRF literature).
    """

    def __init__(self, *, k: int = 60) -> None:
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}")
        self._k: int = k

    def fuse(self, sparse_hits: list[Hit], dense_hits: list[Hit]) -> list[Hit]:
        """Merge and re-rank sparse and dense hits using Reciprocal Rank Fusion.

        RRF score for chunk d: sum over each ranked list of 1/(k + rank(d)),
        where rank(d) is the 1-based position in that list (absent -> 0 contribution).

        Duplicate chunk IDs are merged: the first occurrence carries the winning
        sparse_score/dense_score values.  When a chunk appears in both lists its
        sparse_score and dense_score are both preserved.

        Args:
            sparse_hits: Hits from a SparseRetriever, ordered by sparse_score DESC.
            dense_hits:  Hits from a DenseRetriever, ordered by dense_score DESC.

        Returns:
            List of Hit records with fusion_score set, ordered by fusion_score DESC.
            Tie-break: chunk_id ASC (lexicographic) for full determinism.
            sparse_score and dense_score from inputs are preserved.
            rerank_score is always None (set by the reranking stage).
        """
        k = self._k

        # Accumulate the best component scores and RRF contributions per chunk.
        # sparse_scores[chunk_id] = sparse_score (from the first occurrence)
        # dense_scores[chunk_id]  = dense_score  (from the first occurrence)
        sparse_scores: dict[str, float | None] = {}
        dense_scores: dict[str, float | None] = {}
        rrf_score: dict[str, float] = {}
        chunks: dict[str, Hit] = {}  # chunk_id -> representative Hit (for the Chunk object)

        # Process sparse list (rank is 1-based position in sparse_hits)
        for rank, hit in enumerate(sparse_hits, start=1):
            cid = str(hit.chunk.id)
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks:
                chunks[cid] = hit
                sparse_scores[cid] = hit.sparse_score
                dense_scores[cid] = hit.dense_score

        # Process dense list (rank is 1-based position in dense_hits)
        for rank, hit in enumerate(dense_hits, start=1):
            cid = str(hit.chunk.id)
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks:
                chunks[cid] = hit
                sparse_scores[cid] = hit.sparse_score
                dense_scores[cid] = hit.dense_score
            else:
                # Chunk was in sparse list; pick up dense_score from this hit.
                if dense_scores[cid] is None and hit.dense_score is not None:
                    dense_scores[cid] = hit.dense_score

        # Build fused Hits preserving component scores.
        result: list[Hit] = []
        for cid, base_hit in chunks.items():
            fused = dataclasses.replace(
                base_hit,
                sparse_score=sparse_scores[cid],
                dense_score=dense_scores[cid],
                fusion_score=rrf_score[cid],
                rerank_score=None,
            )
            result.append(fused)

        # Primary sort: fusion_score DESC. Tie-break: chunk_id ASC (deterministic).
        result.sort(key=lambda h: (-( h.fusion_score or 0.0), str(h.chunk.id)))
        return result
