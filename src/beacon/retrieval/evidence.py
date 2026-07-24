"""Evidence assembly: budgeted packing, post-ordering expansion, gap-free labels (Task 03.2).

This module converts final-ordered retrieval hits into an EvidenceBundle:

1. Pack primary hits in final order while the token budget remains.
   - Hits whose text exceeds the remaining budget are skipped; the next hit is
     tried.  Skipping never leaves a gap in the label sequence.
2. For each packed primary hit, expand context neighbors (prev/next chunks)
   using the caller-supplied fetch_chunk callable.
   - Expansion is bounded: max_neighbor_hops hops per direction, and
     max_context_per_hit total context spans per primary hit.
   - Expansion happens ONLY after the final candidate order is known (the
     caller must supply ordered_hits in final order before calling this
     function).
3. Assign stable, gap-free S1..Sn labels to packed evidence (primary hits
   then context spans) AFTER packing is complete.
4. Context spans carry context_of (the chunk_id of the primary hit they were
   expanded from) and no relevance score (score=None).
5. No chunk_id appears twice in a bundle.

Canonical chunk identity
------------------------
The canonical identifier throughout this module is the **hex chunk id** stored
in the payload field ``chunk_hash`` (a 64-character SHA-256 hex string).

- ``Evidence.chunk_id`` is always the hex chunk id.
- ``included_ids`` (dedup set) is keyed exclusively on hex chunk ids.
- Payload navigation fields ``prev_chunk_id`` / ``next_chunk_id`` also carry
  hex chunk ids, so the dedup set and the neighbor chain share the same key
  space and intersect correctly.

The Qdrant point id (UUID derived from the hex id via
``chunk_id_to_point_id``) is an internal storage detail.  It is NOT used as
an Evidence identifier; callers that need to address a Qdrant point must call
``chunk_id_to_point_id`` themselves.

The ``fetch_chunk`` seam accepts and returns hex chunk ids.  Implementations
that back-translate to a Qdrant point id for the actual fetch must do so
internally; the seam contract is hex-in / payload-out.

Token accounting uses a heuristic: ceil(len(text) / 4.0) characters per
token.  This is the same approach as beacon_kb's HeuristicTokenCounter and
is documented here as the module's stated heuristic.

All functions are pure over their inputs: no Qdrant, no SQLite, no LLM calls.
The fetch_chunk callable is the only IO seam.

Importing this module performs no side effects.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from beacon.models import (
    BudgetRecap,
    Evidence,
    EvidenceBundle,
    EvidenceRole,
    Snippet,
)
from beacon.retrieval.hybrid import Hit
from beacon.retrieval.snippets import build_snippet

__all__ = [
    "EvidenceRole",
    "assemble_evidence",
]

# ---------------------------------------------------------------------------
# Token heuristic
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN: float = 4.0
"""Heuristic: 4 characters per token (same as beacon_kb HeuristicTokenCounter)."""


def _count_tokens(text: str) -> int:
    """Return estimated token count for text using the module heuristic.

    Heuristic: ceil(len(text) / 4.0).  Never raises; empty strings return 0.
    """
    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Neighbor resolution
# ---------------------------------------------------------------------------


def _resolve_neighbors(
    chunk_id: str,
    payload: dict[str, Any],
    fetch_chunk: Callable[[str], dict[str, Any] | None],
    *,
    max_hops: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve prev/next neighbor chunks for one primary hit.

    Walks the prev_chunk_id / next_chunk_id chain up to max_hops in each
    direction.  Chunks that cannot be fetched are silently skipped.

    ID format contract: all chunk ids exchanged here are hex chunk ids
    (64-char SHA-256 strings, stored in payload ``chunk_hash``).  The
    ``fetch_chunk`` callable receives a hex chunk id and returns a payload
    dict (or None).

    Args:
        chunk_id:    Hex chunk id of the primary hit.
        payload:     Payload dict of the primary hit.
        fetch_chunk: Callable mapping hex chunk id -> payload dict | None.
        max_hops:    Maximum hops in each direction.

    Returns:
        List of (hex_chunk_id, payload) tuples for neighbors in document order
        (predecessors first, then successors relative to the primary hit).
    """
    neighbors: list[tuple[str, dict[str, Any]]] = []

    # Walk backward (prev_chunk_id chain).
    prev_chain: list[tuple[str, dict[str, Any]]] = []
    current_payload = payload
    for _ in range(max_hops):
        prev_id = current_payload.get("prev_chunk_id")
        if not prev_id:
            break
        prev_payload = fetch_chunk(str(prev_id))
        if prev_payload is None:
            break
        prev_chain.append((str(prev_id), prev_payload))
        current_payload = prev_payload

    # Reverse so predecessors appear in document order (oldest first).
    neighbors.extend(reversed(prev_chain))

    # Walk forward (next_chunk_id chain).
    current_payload = payload
    for _ in range(max_hops):
        next_id = current_payload.get("next_chunk_id")
        if not next_id:
            break
        next_payload = fetch_chunk(str(next_id))
        if next_payload is None:
            break
        neighbors.append((str(next_id), next_payload))
        current_payload = next_payload

    return neighbors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_evidence(
    hits: list[Hit],
    query_text: str,
    fetch_chunk: Callable[[str], dict[str, Any] | None],
    *,
    token_budget: int,
    max_neighbor_hops: int = 1,
    max_context_per_hit: int = 2,
) -> EvidenceBundle:
    """Assemble a token-budgeted, labeled EvidenceBundle from final-ordered hits.

    Steps:
    1. Pack primary hits in final order while budget remains; skip oversized hits
       and continue (do not stop on first overflow).
    2. Expand context neighbors for each packed primary hit within the remaining
       budget.  Expansion is bounded by max_neighbor_hops and max_context_per_hit.
    3. Assign gap-free S1..Sn labels to all packed evidence AFTER packing.
    4. Build match-centered Snippet for each evidence item from its payload.

    The caller must supply hits in final order (post-rerank); this function
    never reorders them.

    Args:
        hits:               Final-ordered primary hits from the retrieval pipeline.
        query_text:         Original user query text for snippet centering.
        fetch_chunk:        Callable mapping **hex chunk id** -> payload dict | None.
                            Used for neighbor expansion; may return None when the
                            neighbor is not available.  The hex chunk id is the
                            64-char SHA-256 string stored in payload ``chunk_hash``.
        token_budget:       Maximum total token count for all evidence.
        max_neighbor_hops:  Maximum prev/next hops per primary hit per direction.
        max_context_per_hit: Maximum context chunks added per primary hit.

    Returns:
        EvidenceBundle with gap-free labeled Evidence items and a BudgetRecap.
        Every ``Evidence.chunk_id`` is the hex chunk id (from payload
        ``chunk_hash``), never a Qdrant point UUID.
    """
    requested = len(hits)
    effective_budget = max(0, token_budget)

    # --- Phase 1: Pack primary hits ------------------------------------------
    # Skip hits whose text does not fit the remaining budget, but continue
    # checking subsequent hits (do not stop early).
    # Surviving hits collect (chunk_id, payload, fused_score); labels come later.

    surviving_primary: list[tuple[str, dict[str, Any], float]] = []
    included_ids: set[str] = set()
    token_tally = 0
    skipped = 0

    for hit in hits:
        payload = hit.payload
        # Canonical identity is the hex chunk id from payload["chunk_hash"].
        # Falling back to chunk_point_id (UUID) is a safety net for synthetic
        # test data that omits chunk_hash; in production every payload has it.
        chunk_id: str = str(payload.get("chunk_hash") or hit.chunk_point_id)
        text = payload.get("chunk_text", "")
        tok = _count_tokens(text)
        if token_tally + tok > effective_budget:
            skipped += 1
            continue
        token_tally += tok
        surviving_primary.append((chunk_id, payload, hit.fused_score))
        included_ids.add(chunk_id)

    packed_primary_count = len(surviving_primary)

    # --- Phase 2: Expand context neighbors -----------------------------------
    # Only for primary hits that fit the budget; expansion uses remaining budget.

    context_items: list[tuple[str, dict[str, Any], str]] = []  # (cid, payload, primary_cid)

    for primary_cid, primary_payload, _ in surviving_primary:
        neighbors = _resolve_neighbors(
            primary_cid,
            primary_payload,
            fetch_chunk,
            max_hops=max_neighbor_hops,
        )
        added_for_hit = 0
        for neighbor_cid, neighbor_payload in neighbors:
            if added_for_hit >= max_context_per_hit:
                break
            if neighbor_cid in included_ids:
                continue
            text = neighbor_payload.get("chunk_text", "")
            tok = _count_tokens(text)
            if token_tally + tok > effective_budget:
                continue
            token_tally += tok
            included_ids.add(neighbor_cid)
            added_for_hit += 1
            context_items.append((neighbor_cid, neighbor_payload, primary_cid))

    # --- Phase 3: Assign gap-free labels and build Evidence ------------------
    # Labels are assigned AFTER packing so skipped hits leave no gaps.

    all_evidence: list[Evidence] = []
    label_counter = 1

    # Primary hits first.
    for chunk_id, payload, fused_score in surviving_primary:
        snippet = _build_snippet_from_payload(chunk_id, payload, query_text)
        ev = Evidence(
            chunk_id=chunk_id,
            label=f"S{label_counter}",
            role=EvidenceRole.HIT,
            score=fused_score,
            context_of=None,
            snippet=snippet,
        )
        all_evidence.append(ev)
        label_counter += 1

    # Context spans after primary hits.
    for neighbor_cid, neighbor_payload, primary_cid in context_items:
        snippet = _build_snippet_from_payload(neighbor_cid, neighbor_payload, query_text)
        ev = Evidence(
            chunk_id=neighbor_cid,
            label=f"S{label_counter}",
            role=EvidenceRole.CONTEXT,
            score=None,
            context_of=primary_cid,
            snippet=snippet,
        )
        all_evidence.append(ev)
        label_counter += 1

    recap = BudgetRecap(
        requested=requested,
        packed=packed_primary_count,
        skipped=skipped,
        tokens_packed=token_tally,
        token_budget=token_budget,
    )

    return EvidenceBundle(evidence=all_evidence, recap=recap)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_snippet_from_payload(
    chunk_id: str,
    payload: dict[str, Any],
    query_text: str,
) -> Snippet:
    """Build a match-centered Snippet from a chunk payload dict.

    Provenance fields are read directly from the payload; source_uri and title
    are taken verbatim from the payload - they are real URIs and titles, never
    internal hash-form identifiers.

    Args:
        chunk_id:   Chunk identifier for the Snippet.
        payload:    Chunk payload dict (ChunkPayload.to_dict() shape).
        query_text: User query text for match centering.

    Returns:
        Snippet with match-centered text and real provenance.
    """
    chunk_text = payload.get("chunk_text", "")
    source_uri = payload.get("source_uri", "")
    title = payload.get("title", "")
    heading_path = payload.get("heading_path") or []
    locator = "/".join(heading_path) if heading_path else ""

    return build_snippet(
        chunk_text,
        query_text,
        source_uri=source_uri,
        title=title,
        heading_path=list(heading_path),
        locator=locator,
        chunk_id=chunk_id,
    )
