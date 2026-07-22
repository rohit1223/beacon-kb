"""Deterministic pre- and post-generation abstention policy.

PRE-generation abstention triggers BEFORE the generator is called:
  - No evidence (empty hit list) -> abstain immediately, no LLM call.
  - All hits below abstain_threshold (fusion/rerank score) -> abstain.

POST-generation abstention converts generator output to typed failure or
safe abstention when the model returns the sentinel word "ABSTAIN" or when
the answer_text is empty.

Both policies are pure functions with no side effects.

Importing this module performs no side effects.
"""

from __future__ import annotations

from beacon_kb.models import Hit

# ---------------------------------------------------------------------------
# Pre-generation policy
# ---------------------------------------------------------------------------

ABSTAIN_SENTINEL: str = "ABSTAIN"
"""Token the grounded prompt instructs the model to emit when it cannot answer."""


def should_pre_abstain(
    hits: list[Hit],
    *,
    abstain_threshold: float = 0.0,
) -> bool:
    """Return True if the answer path must abstain before calling the generator.

    Triggers:
      1. *hits* is empty.
      2. *abstain_threshold* > 0 and every hit's best available score falls
         below the threshold.  Score priority: rerank_score > fusion_score >
         dense_score > sparse_score.  Hits with no score at all are treated as
         score=0.0 (safest assumption).

    Args:
        hits:              Retrieved hits from the retrieval pipeline.
        abstain_threshold: Minimum acceptable score.  0.0 means no score gate.

    Returns:
        True when the generator must be skipped and a canned abstention returned.
    """
    if not hits:
        return True

    if abstain_threshold <= 0.0:
        return False

    for hit in hits:
        best: float = _best_score(hit)
        if best >= abstain_threshold:
            return False  # At least one hit clears the bar.

    return True  # All hits are below threshold.


def _best_score(hit: Hit) -> float:
    """Return the highest available score on a Hit, or 0.0 if none are set."""
    for score in (hit.rerank_score, hit.fusion_score, hit.dense_score, hit.sparse_score):
        if score is not None:
            return score
    return 0.0


# ---------------------------------------------------------------------------
# Post-generation policy
# ---------------------------------------------------------------------------


def is_post_abstain(answer_text: str) -> bool:
    """Return True if the generator's output represents a post-generation abstention.

    Triggers:
      - answer_text is empty (the generator produced no content).
      - answer_text stripped is exactly the sentinel string "ABSTAIN" (case-sensitive).

    Args:
        answer_text: Raw answer_text returned by the generator.

    Returns:
        True when the response must be converted to a safe abstention.
    """
    stripped = answer_text.strip()
    return stripped == "" or stripped == ABSTAIN_SENTINEL
