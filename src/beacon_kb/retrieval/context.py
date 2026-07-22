"""Bounded context assembly: neighbor and sibling-section expansion for evidence packing.

Design rules enforced here:

Parent identity note (CRITICAL - from ROADMAP and merge checklist):
  ChunkKind.PARENT records are NOT materialized in the store.
  The implicit parent of any child chunk is identified by the pair:
  (section_id, parent_locator) carried on each child chunk.
  "Parent expansion" means fetching sibling children that share the same
  section_id and parent_locator as a candidate - it does NOT mean fetching
  a PARENT record (none exists).  See ROADMAP.md and the ingestion/chunking.py
  module for the authoritative definition.

Context expansion rules:
- Neighbor expansion (prev/next) resolves prev_chunk_id / next_chunk_id via
  the store's get_chunk() after the final candidate ordering is determined.
- Parent expansion means fetching sibling children of the same section locator.
  In this implementation, sibling discovery relies on the store's retrieve() or
  get_chunk() because there is no list-by-section API.  We resolve prev/next
  chains up to `max_neighbor_hops` hops from each primary hit.
- Expansion ONLY occurs after final candidate ordering.
- Context spans keep context_of relationships - they reference the primary hit
  they were expanded from and NEVER receive invented relevance scores.
- Expansion is bounded: the total number of context chunks added per primary
  hit is capped by max_context_per_hit.
- Token budget is enforced: evidence is packed until the budget is exhausted.
  The budget enforcer runs before prompt construction and produces a
  BudgetSummary recap.
- Primary hits are always included before context spans (context spans are
  added only if budget allows after all primary hits are packed).

Query.top_k vs config.retrieval.top_k reconciliation decision (ROADMAP Epic 03):
  Per-query top_k overrides the config value when set to a non-default value.
  The pipeline uses query.top_k if it differs from the default (10), otherwise
  it falls back to config.retrieval.top_k.  This is documented at the call site
  in pipeline.py.  Context assembly respects whichever top_k the pipeline
  resolved to.

Importing this module performs no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from beacon_kb.models import (
    Evidence,
    EvidenceId,
    EvidenceRole,
    Hit,
    Query,
    make_evidence_id,
)
from beacon_kb.tokens import BudgetSummary, HeuristicTokenCounter

if TYPE_CHECKING:
    from beacon_kb.protocols import TokenCounter
    from beacon_kb.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# Context expansion parameters
# ---------------------------------------------------------------------------

# Default maximum neighbor hops to expand per primary hit (prev + next).
# Each hop resolves one prev_chunk_id or next_chunk_id via get_chunk().
# With max_neighbor_hops=1, a primary hit gains at most one predecessor and
# one successor.  With max_neighbor_hops=2, it may gain two predecessors and
# two successors.
_DEFAULT_MAX_NEIGHBOR_HOPS: int = 1

# Default maximum context chunks added per primary hit across all directions.
_DEFAULT_MAX_CONTEXT_PER_HIT: int = 2


@dataclass(frozen=True, slots=True)
class ContextExpansionResult:
    """Result of context expansion for a list of primary hits.

    Attributes:
        evidence:      Packed Evidence list (primary HITs then CONTEXT spans),
                       ordered by hit rank then context position.
        budget_summary: Token-budget recap to log or include before prompt
                        construction (satisfies the required result-count and
                        token recap step).
    """

    evidence: list[Evidence]
    budget_summary: BudgetSummary


def _resolve_neighbors(
    primary_hit: Hit,
    store: SQLiteStore,
    *,
    max_hops: int,
) -> list[Hit]:
    """Resolve prev/next neighbor chunks for one primary hit.

    Walks the prev_chunk_id / next_chunk_id chain up to *max_hops* in each
    direction.  Chunks that cannot be fetched (None from get_chunk) are
    silently skipped - the chain is considered terminated.

    Args:
        primary_hit: The primary Hit whose neighbors to resolve.
        store:       Store for get_chunk() lookups.
        max_hops:    Maximum number of hops in each direction.

    Returns:
        List of Hit records for neighbors in document order (predecessors
        then successors relative to primary_hit), each without any score
        (all score fields None) - scores are intentionally absent because
        neighbors are context, not primary hits.
    """
    neighbors: list[Hit] = []

    # Walk backward (prev_chunk_id chain).
    prev_chain: list[Hit] = []
    current_chunk = primary_hit.chunk
    for _ in range(max_hops):
        prev_id = current_chunk.prev_chunk_id
        if prev_id is None:
            break
        prev_chunk = store.get_chunk(str(prev_id))
        if prev_chunk is None:
            break
        # No scores: this is a context-only span.
        prev_chain.append(Hit(chunk=prev_chunk))
        current_chunk = prev_chunk

    # Reverse so predecessors appear in document order (oldest first).
    neighbors.extend(reversed(prev_chain))

    # Walk forward (next_chunk_id chain).
    current_chunk = primary_hit.chunk
    for _ in range(max_hops):
        next_id = current_chunk.next_chunk_id
        if next_id is None:
            break
        next_chunk = store.get_chunk(str(next_id))
        if next_chunk is None:
            break
        neighbors.append(Hit(chunk=next_chunk))
        current_chunk = next_chunk

    return neighbors


def expand_and_pack(
    query: Query,
    ordered_hits: list[Hit],
    store: SQLiteStore,
    *,
    token_budget: int,
    overhead_tokens: int = 0,
    counter: TokenCounter | None = None,
    model: str = "",
    max_neighbor_hops: int = _DEFAULT_MAX_NEIGHBOR_HOPS,
    max_context_per_hit: int = _DEFAULT_MAX_CONTEXT_PER_HIT,
) -> ContextExpansionResult:
    """Expand context neighbors and pack evidence under a token budget.

    Steps:
    1. Assign stable [S1]-style citation labels to primary hits in order.
    2. Pack primary hit Evidence items into the budget (primary hits first).
    3. For each primary hit that fit the budget, expand neighbors (bounded).
    4. Pack context Evidence items into the remaining budget.
    5. Compute and return BudgetSummary as the result-count + token recap.

    Context expansion ONLY occurs after final candidate ordering (ordered_hits
    must already be in final order before calling this function).

    Context spans use EvidenceRole.CONTEXT and carry context_of=primary_ev_id
    (the EvidenceId of the primary HIT) so callers can distinguish context from
    primary hits.  citation_label for CONTEXT spans is also a plain "S{n}" label
    in the sequential gap-free numbering.  Context spans never receive invented
    relevance scores - their Hit records have all score fields None.

    Args:
        query:              The query that produced the hits.
        ordered_hits:       Final ordered primary hits (best first).
        store:              Store for neighbor chunk lookups.
        token_budget:       Maximum total tokens for all evidence (primary + context).
        overhead_tokens:    Reserved token overhead (system prompt, etc.).
        counter:            TokenCounter instance; defaults to HeuristicTokenCounter.
        model:              Model name for the counter.
        max_neighbor_hops:  Maximum neighbor-chain hops per primary hit per direction.
        max_context_per_hit: Maximum context chunks added per primary hit.

    Returns:
        ContextExpansionResult with packed Evidence list and BudgetSummary.
    """
    if counter is None:
        counter = HeuristicTokenCounter()

    query_id = str(query.id)
    effective_budget = max(0, token_budget - overhead_tokens)

    # --- Phase 1: Pack primary hits into the budget --------------------------
    # Hits that exceed the budget are skipped (overflow). After all hits are
    # considered, surviving items are reassigned sequential gap-free S1..Sn
    # labels so the final evidence is always numbered without gaps.

    # Collect (hit, evidence_id) for surviving primary hits; labels come later.
    surviving_primary: list[tuple[Hit, EvidenceId]] = []
    token_tally = 0
    primary_overflow = 0

    for hit in ordered_hits:
        text = hit.chunk.text
        tok = counter.count_tokens(text, model=model)
        if token_tally + tok > effective_budget:
            # This hit does not fit; count as overflow.
            primary_overflow += 1
            continue
        token_tally += tok
        eid = make_evidence_id(query_id=query_id, chunk_id=str(hit.chunk.id))
        surviving_primary.append((hit, EvidenceId(eid)))

    # Assign gap-free sequential labels to surviving primary hits.
    primary_evidence: list[Evidence] = [
        Evidence(
            id=eid,
            hit=hit,
            citation_label=f"S{rank}",
            role=EvidenceRole.HIT,
        )
        for rank, (hit, eid) in enumerate(surviving_primary, start=1)
    ]

    # --- Phase 2: Expand neighbors for primary hits that fit -----------------

    # Track chunk IDs already included to avoid duplicates.
    included_ids: set[str] = {str(ev.hit.chunk.id) for ev in primary_evidence}

    context_evidence: list[Evidence] = []
    # Sequential label counter continues from where primary labels ended.
    next_label_n = len(primary_evidence) + 1

    for ev in primary_evidence:
        primary_hit = ev.hit
        primary_ev_id = ev.id  # used for context_of provenance

        neighbors = _resolve_neighbors(
            primary_hit,
            store,
            max_hops=max_neighbor_hops,
        )

        added_for_this_hit = 0
        for neighbor_hit in neighbors:
            if added_for_this_hit >= max_context_per_hit:
                break
            cid = str(neighbor_hit.chunk.id)
            if cid in included_ids:
                continue
            text = neighbor_hit.chunk.text
            tok = counter.count_tokens(text, model=model)
            if token_tally + tok > effective_budget:
                continue
            token_tally += tok
            included_ids.add(cid)
            added_for_this_hit += 1
            ctx_label = f"S{next_label_n}"
            next_label_n += 1
            ctx_eid = make_evidence_id(query_id=query_id, chunk_id=cid)
            context_evidence.append(
                Evidence(
                    id=EvidenceId(ctx_eid),
                    hit=neighbor_hit,
                    citation_label=ctx_label,
                    role=EvidenceRole.CONTEXT,
                    context_of=primary_ev_id,
                )
            )

    # Build final evidence list: primary HITs first, then CONTEXT spans.
    all_evidence = list(primary_evidence) + context_evidence

    # --- Phase 3: Build BudgetSummary ----------------------------------------
    # overflow_count tracks primary hits that did not fit the budget.
    # Context overflow is not counted here because context spans are additive
    # extras; the primary result_count and overflow_count capture the actionable
    # budget signal for callers deciding how many primary sources were included.

    result_count = len(primary_evidence)
    overflow_count = primary_overflow  # primary overflow only; see docstring above

    summary = BudgetSummary(
        result_count=result_count,
        total_tokens=token_tally,
        remaining_tokens=max(0, effective_budget - token_tally),
        budget=token_budget,
        overflow_count=overflow_count,
    )

    return ContextExpansionResult(evidence=all_evidence, budget_summary=summary)
