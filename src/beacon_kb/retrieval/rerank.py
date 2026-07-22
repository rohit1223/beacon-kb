"""Optional reranking stage over a bounded fused candidate window.

Design rules enforced here:
- The reranker is injected (protocol); this module contains no first-party model.
- Reranking applies only to the top ``window`` hits (bounded to control cost/latency).
- Hits outside the window are appended after the reranked window, in original order.
- Reranker absent (None) -> fused order returned unchanged; no latency recorded.
- Reranker fails (BackendError or any exception) -> fused order returned unchanged;
  the exception is stored in RerankResult.failure (structured, not a log line).
- Latency is measured with an injectable clock (default: time.monotonic).
  Tests pass FakeClock.now to avoid wall-clock dependency.
- Every component score from the fused input is preserved through reranking.

Importing this module performs no side effects.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from beacon_kb.models import Hit, Query
from beacon_kb.protocols import Reranker


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Structured result of the optional reranking stage.

    Fields:
        hits:             Final ordered Hit list (reranked window + tail, or fused order).
        window:           Number of hits passed to the reranker (0 if no reranker).
        latency_seconds:  Wall-clock seconds consumed by the reranker call
                          (None if reranker was absent or before the call succeeded).
        failure:          Exception raised by the reranker (None if no failure).
                          Stored as a structured Exception object for typed handling.
    """

    hits: list[Hit]
    window: int
    latency_seconds: float | None = None
    failure: Exception | None = None


def rerank_hits(
    query: Query,
    hits: list[Hit],
    *,
    reranker: Reranker | None,
    window: int = 50,
    clock: Callable[[], float] | None = None,
) -> RerankResult:
    """Optionally rerank the top ``window`` fused hits.

    Applies a bounded reranking window: only the first ``window`` hits are passed
    to the reranker.  Hits beyond the window are appended after the reranked
    segment in their original order.

    Best-effort policy: any exception from ``reranker.rerank()`` is caught,
    stored in ``RerankResult.failure``, and the original fused order is returned.
    The caller decides whether to surface, log, or ignore the failure.

    Args:
        query:    The original user query.
        hits:     Fused hits to (optionally) rerank, ordered by fusion_score DESC.
        reranker: An optional Reranker implementation.  None -> no reranking.
        window:   Maximum number of hits to pass to the reranker (default 50).
        clock:    Injectable monotonic clock (default time.monotonic).
                  Tests pass FakeClock.now to avoid wall-clock dependency.

    Returns:
        RerankResult with hits, window, latency_seconds, and failure.
    """
    _clock = clock if clock is not None else time.monotonic

    if reranker is None:
        return RerankResult(hits=list(hits), window=0, latency_seconds=None, failure=None)

    if not hits:
        return RerankResult(hits=[], window=0, latency_seconds=None, failure=None)

    # Partition into the bounded window and the tail.
    to_rerank = hits[:window]
    tail = hits[window:]

    t_start = _clock()
    try:
        reranked = reranker.rerank(query, to_rerank)
    except Exception as exc:  # best-effort: catch all, record, fall back to fused order
        return RerankResult(
            hits=list(hits),
            window=len(to_rerank),
            latency_seconds=None,
            failure=exc,
        )

    t_end = _clock()
    latency = t_end - t_start

    # Guard: reranker must return the same chunk IDs it received, including
    # duplicates.  Counter (multiset) equality catches both added/removed IDs
    # and duplicate-ID returns that set equality would miss.
    expected_ids = Counter(str(h.chunk.id) for h in to_rerank)
    returned_ids = Counter(str(h.chunk.id) for h in reranked)
    if expected_ids != returned_ids:
        return RerankResult(
            hits=list(hits),
            window=len(to_rerank),
            latency_seconds=latency,
            failure=ValueError(
                f"Reranker returned chunk IDs (multiset) that do not match input window. "
                f"Expected counts: {dict(expected_ids)!r}, "
                f"Returned counts: {dict(returned_ids)!r}. "
                f"Falling back to fused order."
            ),
        )

    final_hits = list(reranked) + list(tail)
    return RerankResult(
        hits=final_hits,
        window=len(to_rerank),
        latency_seconds=latency,
        failure=None,
    )
