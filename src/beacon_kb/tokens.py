"""Heuristic token counter and budget arithmetic helpers.

This module provides:
- ``HeuristicTokenCounter``: a default ``TokenCounter`` implementation using a
  simple word/4 heuristic for fast, dependency-free token estimation.
- Budget arithmetic helpers: ``compute_evidence_budget`` and
  ``summarize_budget`` that produce a result-count and token recap before
  prompt construction, satisfying the brief's requirement for an enforceable
  evidence budget.

The heuristic counter deliberately avoids importing any tokenizer library so
the module is safe to use without optional dependencies.  It satisfies the
``TokenCounter`` protocol from ``beacon_kb.protocols`` and falls back
to the heuristic for any model name.

Importing this module performs no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from beacon_kb.errors import BeaconError
from beacon_kb.protocols import TokenCounter

# ---------------------------------------------------------------------------
# Heuristic token counter
# ---------------------------------------------------------------------------

# Characters-per-token estimates per model family.
# These are conservative estimates; actual values vary by model.
_CHARS_PER_TOKEN: dict[str, float] = {
    "gpt-4": 4.0,
    "gpt-3.5": 4.0,
    "claude": 4.0,
    "gemini": 4.0,
    "llama": 4.0,
    "mistral": 4.0,
}

_DEFAULT_CHARS_PER_TOKEN: float = 4.0


def _chars_per_token(model: str) -> float:
    """Return estimated chars-per-token for the given model family.

    Falls back to the default heuristic if the model is unknown.

    Args:
        model: Model name string (e.g. 'gpt-4', 'claude-3-opus', '').

    Returns:
        Float chars-per-token ratio.
    """
    if not model:
        return _DEFAULT_CHARS_PER_TOKEN
    model_lower = model.lower()
    for prefix, cpt in _CHARS_PER_TOKEN.items():
        if prefix in model_lower:
            return cpt
    return _DEFAULT_CHARS_PER_TOKEN


class HeuristicTokenCounter:
    """Default heuristic TokenCounter satisfying the TokenCounter protocol.

    Uses a character-count / chars-per-token heuristic.  The default is
    4 characters per token, which is a common approximation for English text.

    This counter is intentionally fast and dependency-free.  It does not use
    a real tokenizer, so counts are approximate.  For production use where
    exact counts matter, replace this with a model-specific tokenizer via the
    ``beacon_kb.token_counters`` entry-point group.

    Error contract: never raises for an unknown model name.
    Unknown or empty model names always fall back to the heuristic.
    """

    def count_tokens(self, text: str, *, model: str = "") -> int:
        """Return the estimated token count for *text*.

        Uses ``ceil(len(text) / chars_per_token)`` where ``chars_per_token``
        defaults to 4.0 for unknown or empty model names.

        Args:
            text:  Input string to count tokens for.
            model: Optional model name; used to look up a chars-per-token ratio.

        Returns:
            Non-negative integer token count.  Empty strings return 0.
        """
        if not text:
            return 0
        cpt = _chars_per_token(model)
        return math.ceil(len(text) / cpt)


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetSummary:
    """Result of a budget computation for evidence selection.

    Attributes:
        result_count:       Number of evidence items that fit within the budget.
        total_tokens:       Total estimated token count for the selected items.
        remaining_tokens:   Tokens remaining after evidence selection.
        budget:             The original token budget.
        overflow_count:     Number of items that were excluded due to budget.
    """

    result_count: int
    total_tokens: int
    remaining_tokens: int
    budget: int
    overflow_count: int

    def __str__(self) -> str:
        """Return a human-readable recap of the budget computation."""
        return (
            f"Budget: {self.total_tokens}/{self.budget} tokens used "
            f"({self.result_count} results fit, {self.overflow_count} excluded, "
            f"{self.remaining_tokens} tokens remaining)"
        )


def compute_evidence_budget(
    texts: list[str],
    *,
    token_budget: int,
    counter: TokenCounter | None = None,
    model: str = "",
    overhead_tokens: int = 0,
) -> BudgetSummary:
    """Select as many evidence texts as fit within *token_budget*.

    Iterates through *texts* in order and accumulates token counts until the
    budget is exhausted.  Returns a ``BudgetSummary`` with the result count,
    total tokens used, remaining tokens, and overflow count.

    This function is the token recap step that must run before prompt
    construction.  Callers use ``BudgetSummary.result_count`` to slice
    the evidence list and ``BudgetSummary.remaining_tokens`` to budget the
    generator's ``max_output_tokens``.

    Args:
        texts:           Ordered list of evidence text strings.
        token_budget:    Maximum total token count for evidence.
        counter:         TokenCounter instance.  Defaults to HeuristicTokenCounter.
        model:           Model name passed to the counter.
        overhead_tokens: Fixed overhead to reserve (e.g. for system prompt).

    Returns:
        BudgetSummary with result_count, total_tokens, remaining_tokens,
        budget, and overflow_count populated.

    Raises:
        BeaconError: If token_budget is negative.
    """
    if token_budget < 0:
        raise BeaconError(
            f"token_budget must be non-negative. Got: {token_budget!r}."
        )
    if counter is None:
        counter = HeuristicTokenCounter()

    effective_budget = max(0, token_budget - overhead_tokens)
    total = 0
    count = 0
    for text in texts:
        tok = counter.count_tokens(text, model=model)
        if total + tok > effective_budget:
            break
        total += tok
        count += 1

    overflow = len(texts) - count
    remaining = effective_budget - total
    return BudgetSummary(
        result_count=count,
        total_tokens=total,
        remaining_tokens=remaining,
        budget=token_budget,
        overflow_count=overflow,
    )


def summarize_budget(summary: BudgetSummary) -> str:
    """Return a plain-text recap of the budget for logging or prompt preamble.

    Args:
        summary: BudgetSummary from :func:`compute_evidence_budget`.

    Returns:
        Human-readable single-line string suitable for logging or prompt preamble.
    """
    return str(summary)
