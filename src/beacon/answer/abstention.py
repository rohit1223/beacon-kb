"""Deterministic pre- and post-generation abstention policy (Task 03.3).

Ported from beacon_kb.generation.abstention onto the beacon EvidenceBundle.

PRE-generation abstention triggers BEFORE any provider call, with zero LLM
calls:
  - The bundle carries no primary HIT evidence (empty bundle, or only context
    spans) -> abstain immediately.
  - Every primary HIT score falls below the configured policy threshold ->
    abstain.  A HIT with no score is treated as score 0.0 (safest assumption).

The threshold is expressed against the RRF-scale scores produced by the
retrieval pipeline (Task 03.1): Qdrant native RRF fused scores are bounded by
2/k (approx 0.033 at k=60).  The default threshold is therefore 0.0 (gate off)
so normal fused scores never spuriously abstain - the v1 bug where a 0.5
default silenced every hybrid answer before the model was ever called.

POST-generation abstention converts provider output to a safe abstention when
the model returns the sentinel word "ABSTAIN" (exact, after stripping) or an
empty answer.

Both policies are pure functions with no side effects.

Importing this module performs no side effects.
"""

from __future__ import annotations

from beacon.models import EvidenceBundle, EvidenceRole

__all__ = [
    "ABSTAIN_SENTINEL",
    "is_post_abstain",
    "should_pre_abstain",
]

# ---------------------------------------------------------------------------
# Pre-generation policy
# ---------------------------------------------------------------------------

ABSTAIN_SENTINEL: str = "ABSTAIN"
"""Token the grounded prompt instructs the model to emit when it cannot answer."""


def should_pre_abstain(
    bundle: EvidenceBundle,
    *,
    abstain_threshold: float = 0.0,
) -> bool:
    """Return True if the answer path must abstain before calling the provider.

    Triggers:
      1. The bundle contains no primary HIT evidence (empty, or only context
         spans, which carry no relevance score and cannot ground an answer).
      2. ``abstain_threshold`` > 0 and every primary HIT score is below the
         threshold.  A HIT with ``score is None`` is treated as 0.0.

    The threshold is on the RRF-scale scores from retrieval; the default 0.0
    disables the gate so normal fused scores never spuriously abstain.

    Args:
        bundle:            Canonical evidence bundle from retrieval assembly.
        abstain_threshold: Minimum acceptable HIT score.  0.0 means no gate.

    Returns:
        True when the generator must be skipped and a canned abstention returned.
    """
    hit_scores = [
        (ev.score if ev.score is not None else 0.0)
        for ev in bundle.evidence
        if ev.role is EvidenceRole.HIT
    ]
    if not hit_scores:
        return True

    if abstain_threshold <= 0.0:
        return False

    return all(score < abstain_threshold for score in hit_scores)


# ---------------------------------------------------------------------------
# Post-generation policy
# ---------------------------------------------------------------------------


def is_post_abstain(answer_text: str) -> bool:
    """Return True if the provider output represents a post-generation abstention.

    Triggers:
      - ``answer_text`` is empty or whitespace-only (the model produced no content).
      - ``answer_text`` stripped is exactly the sentinel "ABSTAIN" (case-sensitive).

    Detection is exact, not substring, so a real answer that merely mentions the
    word is not silenced.

    Args:
        answer_text: Raw answer text returned by the provider.

    Returns:
        True when the response must be converted to a safe abstention.
    """
    stripped = answer_text.strip()
    return stripped == "" or stripped == ABSTAIN_SENTINEL
