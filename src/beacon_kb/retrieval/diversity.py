"""Content near-duplicate collapse and MMR-style diversity for fused hit lists.

Design rules enforced here:
- Near-duplicate collapse compares text content within the same source only.
  Chunks from different sources are never merged, even if their text is identical.
  This preserves cross-source provenance.
- Similarity is measured with a simple Jaccard coefficient over word-token sets.
  This is lightweight, deterministic, and has zero dependencies.
- The ``threshold`` parameter controls the minimum similarity above which two
  hits from the same source are considered near-duplicates.
- When collapsing, the first (highest-ranked) hit survives; others are dropped.
- All scores on the surviving hit are preserved unchanged.
- MMR (Maximal Marginal Relevance) diversifies a ranked list by re-ordering:
  each step picks the hit that maximises (lambda_mmr * relevance - (1-lambda_mmr) * redundancy).
  Relevance proxy: fusion_score or 1/(1+position) when no fusion_score.
  Redundancy: maximum Jaccard similarity to any already-selected hit.
  lambda_mmr=1.0 -> pure relevance (no reordering), lambda_mmr=0.0 -> pure diversity.
- MMR never drops hits; it only re-orders them.

Importing this module performs no side effects.
"""

from __future__ import annotations

from beacon_kb.models import Hit

# ---------------------------------------------------------------------------
# Internal text-similarity helper
# ---------------------------------------------------------------------------


def _jaccard(text_a: str, text_b: str) -> float:
    """Return Jaccard similarity of the word-token sets of *text_a* and *text_b*.

    Returns 1.0 for identical texts and 0.0 for texts with no common tokens.
    Case-insensitive; whitespace-split tokenisation.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collapse_near_duplicates(hits: list[Hit], *, threshold: float) -> list[Hit]:
    """Collapse content near-duplicates within the same source, preserving provenance.

    Two hits are near-duplicates if:
    1. They have the same source_id (same source document), AND
    2. Their text Jaccard similarity >= threshold.

    The first (highest-ranked) hit in each near-duplicate group survives.
    Hits from different sources are NEVER merged, even at threshold=0.0,
    ensuring every source's provenance is preserved in the output.

    All scores on the surviving hit are unchanged.  The output list preserves
    the relative order of surviving hits.

    Args:
        hits:      Fused (and optionally reranked) hit list.
        threshold: Minimum Jaccard similarity [0, 1] to consider two same-source
                   hits near-duplicates.  0.0 collapses all same-source pairs;
                   1.0 collapses only identical texts.

    Returns:
        Subset of *hits* with near-duplicates removed, in original relative order.
    """
    if not hits:
        return []

    # survivors[i] tracks which hits we keep.
    # We process hits in order; a later hit is dropped only if it is similar enough
    # to an ALREADY-KEPT hit from the SAME SOURCE.

    kept: list[Hit] = []
    kept_texts_by_source: dict[str, list[str]] = {}  # source_id -> [kept_texts]

    for hit in hits:
        source_key = str(hit.chunk.source_id)
        text = hit.chunk.text

        already_kept_texts = kept_texts_by_source.get(source_key, [])
        is_near_dup = False

        for kept_text in already_kept_texts:
            if _jaccard(text, kept_text) >= threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            kept.append(hit)
            already_kept_texts.append(text)
            kept_texts_by_source[source_key] = already_kept_texts

    return kept


def mmr_diversify(hits: list[Hit], *, lambda_mmr: float = 0.5) -> list[Hit]:
    """Re-order hits using Maximal Marginal Relevance (MMR).

    MMR greedily selects the next hit to maximise:
        lambda_mmr * relevance(h) - (1 - lambda_mmr) * max_sim(h, selected)

    where:
        relevance(h) = h.fusion_score if set, else 1 / (1 + original_rank)
        max_sim(h, selected) = max Jaccard similarity to any already-selected hit.

    lambda_mmr=1.0: pure relevance ordering (original order preserved).
    lambda_mmr=0.0: maximum diversity (similarity to selected hits penalised fully).

    No hits are dropped; all are re-ordered.

    Args:
        hits:       Hit list to diversify (may be empty).
        lambda_mmr: Trade-off between relevance and diversity [0, 1].

    Returns:
        Re-ordered copy of *hits* with all original hits present.
    """
    if not hits:
        return []

    if len(hits) == 1:
        return list(hits)

    # Assign relevance scores.
    relevance: list[float] = []
    for rank, hit in enumerate(hits):
        if hit.fusion_score is not None:
            relevance.append(hit.fusion_score)
        else:
            relevance.append(1.0 / (1.0 + rank))

    remaining_indices = list(range(len(hits)))
    selected: list[Hit] = []
    selected_texts: list[str] = []

    while remaining_indices:
        best_idx: int | None = None
        best_score = float("-inf")

        for i in remaining_indices:
            rel = relevance[i]
            text = hits[i].chunk.text

            if selected_texts:
                max_sim = max(_jaccard(text, st) for st in selected_texts)
            else:
                max_sim = 0.0

            mmr_score = lambda_mmr * rel - (1.0 - lambda_mmr) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        assert best_idx is not None
        selected.append(hits[best_idx])
        selected_texts.append(hits[best_idx].chunk.text)
        remaining_indices.remove(best_idx)

    return selected
