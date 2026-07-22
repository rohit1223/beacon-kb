"""Unified, deterministic RetrievalPipeline composing all retrieval stages.

This module exposes one entry point: ``RetrievalPipeline.search(query, filters)``.
It is the single retrieval primitive reused by answer() and investigate() so
citation logic is never duplicated across callers.

Pipeline stages (in order):
  1. Query policy  - prepare_query (validation + variant text).
  2. Sparse        - BM25 sparse retrieval (required; always runs).
  3. Dense         - embedding-based dense retrieval (optional; skipped when
                     no embedder is configured).
  4. Fusion        - RRF fusion of sparse + dense candidates.
  5. Rerank        - optional cross-encoder rerank over a bounded window.
  6. Diversity     - near-duplicate collapse + optional MMR re-ordering.
  7. Context       - bounded neighbor expansion after final ordering.
  8. Snippets      - match-centered snippet construction for each evidence item.

Design rules enforced here:
- Zero LLM calls anywhere in this pipeline.
- Deterministic for identical inputs (hash-stable IDs, RRF tie-break).
- per-query top_k overrides config when set to a non-default value.  Decision
  site: Query.top_k != DEFAULT_TOP_K means the caller supplied an explicit
  override; otherwise config.retrieval.top_k is used.  This resolves the
  ROADMAP item "Query.top_k vs config.retrieval.top_k reconciliation (Epic 03)".
- Packed evidence never exceeds the configured token budget.
- A result-count + token recap (BudgetSummary) is always computed before
  evidence is returned to the caller.
- Context spans use EvidenceRole.CONTEXT and are distinguishable from primary hits.
- Every Evidence item has a stable [S1]-style ID derived from query_id and chunk_id.

Importing this module performs no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from beacon_kb.config import BeaconConfig
from beacon_kb.models import Evidence, Query
from beacon_kb.retrieval.context import ContextExpansionResult, expand_and_pack
from beacon_kb.retrieval.dense import EmbedderDenseRetriever
from beacon_kb.retrieval.diversity import collapse_near_duplicates, mmr_diversify
from beacon_kb.retrieval.filters import FilterSpec
from beacon_kb.retrieval.fusion import RRFusion
from beacon_kb.retrieval.query import prepare_query
from beacon_kb.retrieval.rerank import rerank_hits
from beacon_kb.retrieval.snippets import build_snippet
from beacon_kb.retrieval.sparse import BM25SparseRetriever
from beacon_kb.tokens import BudgetSummary, summarize_budget

if TYPE_CHECKING:
    from beacon_kb.protocols import Embedder, Reranker, TokenCounter
    from beacon_kb.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# Query.top_k vs config.retrieval.top_k reconciliation
#
# Decision (ROADMAP Epic 03):
#   The default top_k on the Query model is 10 (matches models.py default).
#   When a caller sets Query.top_k to a value different from DEFAULT_QUERY_TOP_K,
#   that per-query value wins.  Otherwise the pipeline uses config.retrieval.top_k.
#
#   This keeps query callers in control (they can lower top_k for fast lookups)
#   without silently ignoring the operator-configured default.
# ---------------------------------------------------------------------------

_DEFAULT_QUERY_TOP_K: int = 10
"""Default top_k from models.Query; used as the "not overridden" sentinel."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of a RetrievalPipeline.search() call.

    Attributes:
        evidence:       Final packed Evidence list, primary HITs first then
                        CONTEXT spans.  Never exceeds the configured token budget.
        budget_summary: Result-count and token recap (produced before prompt
                        construction as required by the task spec).
        budget_recap:   Plain-text string of budget_summary (convenience for logging).
    """

    evidence: list[Evidence]
    budget_summary: BudgetSummary
    budget_recap: str


class RetrievalPipeline:
    """Deterministic hybrid retrieval pipeline.

    Composes query preparation, sparse BM25, dense embedding (optional),
    RRF fusion, optional rerank, diversity, bounded context expansion, and
    match-centered snippet construction into one deterministic search() call.

    This is the single retrieval primitive reused by answer() and investigate();
    citation logic is never duplicated.  Zero LLM calls inside this pipeline.

    Args:
        store:            SQLiteStore (read-only from this pipeline).
        config:           BeaconConfig driving top_k, budget, diversity params.
        embedder:         Optional Embedder.  None -> sparse-only degraded mode.
        reranker:         Optional Reranker.  None -> skip rerank stage.
        token_counter:    Optional TokenCounter.  None -> HeuristicTokenCounter.
        similarity:       Declared vector similarity direction (default 'cosine').
        dup_threshold:    Jaccard threshold for near-duplicate collapse (default 0.85).
        lambda_mmr:       MMR lambda trade-off [0,1] (default 1.0 = no reordering).
        rerank_window:    Maximum hits to pass to the reranker (default 50).
        token_budget:     Maximum token budget for packed evidence.  Overrides
                          config.answer.max_input_tokens when set.
        overhead_tokens:  Reserved tokens for system/user prompt overhead.
        max_neighbor_hops: Neighbor expansion depth per primary hit per direction.
        max_context_per_hit: Max context chunks per primary hit.
        column_weights:   Optional (w_text, w_heading, w_code) BM25 column weights
                          threaded to the sparse retriever.  None uses BM25SparseRetriever
                          defaults (text=1.0, heading=10.0, code=5.0).
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        config: BeaconConfig | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        token_counter: TokenCounter | None = None,
        similarity: str = "cosine",
        dup_threshold: float = 0.85,
        lambda_mmr: float = 1.0,
        rerank_window: int = 50,
        token_budget: int | None = None,
        overhead_tokens: int = 0,
        max_neighbor_hops: int = 1,
        max_context_per_hit: int = 2,
        column_weights: tuple[float, float, float] | None = None,
    ) -> None:
        self._store = store
        self._config = config if config is not None else BeaconConfig()
        self._embedder = embedder
        self._reranker = reranker
        self._token_counter = token_counter
        self._similarity = similarity
        self._dup_threshold = dup_threshold
        self._lambda_mmr = lambda_mmr
        self._rerank_window = rerank_window
        self._overhead_tokens = overhead_tokens
        self._max_neighbor_hops = max_neighbor_hops
        self._max_context_per_hit = max_context_per_hit
        self._column_weights = column_weights

        # Resolve effective token budget.
        if token_budget is not None:
            self._token_budget = token_budget
        else:
            self._token_budget = self._config.answer.max_input_tokens

        # Build sub-components.
        self._sparse = BM25SparseRetriever(store=store, column_weights=column_weights)
        self._dense = EmbedderDenseRetriever(
            store=store,
            embedder=embedder,
            similarity=similarity,
        )
        self._fusion = RRFusion()

    def _resolve_top_k(self, query: Query) -> int:
        """Resolve effective top_k: per-query value overrides config when set.

        Decision (documented per ROADMAP "Query.top_k vs config.retrieval.top_k
        reconciliation"):
          If query.top_k != DEFAULT_QUERY_TOP_K (10), the caller explicitly
          set a top_k override - use it.  Otherwise fall back to
          config.retrieval.top_k which is the operator-configured default.

        This allows individual queries to request fewer candidates for speed
        without silently ignoring the globally-configured default.
        """
        if query.top_k != _DEFAULT_QUERY_TOP_K:
            return query.top_k
        return self._config.retrieval.top_k

    def search(
        self,
        query: Query,
        filters: FilterSpec | None = None,
    ) -> SearchResult:
        """Run the full retrieval pipeline and return packed Evidence with a token recap.

        This method is deterministic: identical (query, filters) inputs always
        produce identical Evidence ordering and IDs within the same store state.

        Stages:
          1. Query policy (validation + text variants).
          2. Sparse BM25 retrieval.
          3. Dense embedding retrieval (skipped when no embedder).
          4. RRF fusion.
          5. Optional rerank over a bounded window.
          6. Near-duplicate collapse + optional MMR diversity.
          7. Bounded context expansion + token-budget packing (neighbor chunks
             after final ordering; never exceeds the configured budget).
          8. Match-centered snippet construction for each evidence item.

        Args:
            query:   Query record.  top_k overrides config when set explicitly.
            filters: Optional FilterSpec for source/tag/media/date filtering.
                     None means no additional filtering.

        Returns:
            SearchResult with evidence list, BudgetSummary, and budget_recap.

        Raises:
            ValueError: If query.text is empty or whitespace-only.
            BackendError: On store read failure.
        """
        # 1. Query policy - validate and prepare text variants.
        _ = prepare_query(query)  # raises ValueError on empty text

        # Apply per-query or config top_k.
        effective_top_k = self._resolve_top_k(query)
        effective_query = Query(
            id=query.id,
            text=query.text,
            corpus_id=query.corpus_id,
            top_k=effective_top_k,
        )

        # Apply filters to sub-retrievers if specified.
        # Re-use constructor-stored sparse/dense retrievers with the filter overlaid.
        # This ensures custom column_weights or other constructor-time retriever
        # configuration is honoured; the filter_spec is the only per-query override.
        filter_spec = filters if filters is not None else FilterSpec()
        sparse_retriever = BM25SparseRetriever(
            store=self._store,
            filter_spec=filter_spec,
            column_weights=self._column_weights,
        )
        dense_retriever = EmbedderDenseRetriever(
            store=self._store,
            embedder=self._embedder,
            similarity=self._similarity,
            filter_spec=filter_spec,
        )

        # 2. Sparse BM25 retrieval.
        sparse_hits = sparse_retriever.retrieve(effective_query)

        # 3. Dense embedding retrieval (empty list when no embedder configured).
        dense_hits = dense_retriever.retrieve(effective_query)

        # 4. RRF fusion.
        fused_hits = self._fusion.fuse(sparse_hits, dense_hits)

        # 5. Optional rerank over bounded window.
        rerank_result = rerank_hits(
            effective_query,
            fused_hits,
            reranker=self._reranker,
            window=self._rerank_window,
        )
        ranked_hits = rerank_result.hits

        # 6. Near-duplicate collapse + optional MMR diversity.
        deduped_hits = collapse_near_duplicates(ranked_hits, threshold=self._dup_threshold)
        diverse_hits = mmr_diversify(deduped_hits, lambda_mmr=self._lambda_mmr)

        # Trim to effective_top_k after diversity (diversity never drops hits but
        # we enforce the top_k ceiling before context expansion).
        candidate_hits = diverse_hits[:effective_top_k]

        # 7-8. Bounded context expansion + token budget packing.
        expansion: ContextExpansionResult = expand_and_pack(
            effective_query,
            candidate_hits,
            self._store,
            token_budget=self._token_budget,
            overhead_tokens=self._overhead_tokens,
            counter=self._token_counter,
            max_neighbor_hops=self._max_neighbor_hops,
            max_context_per_hit=self._max_context_per_hit,
        )

        # 8. Snippets - build match-centered snippet for each evidence item.
        # Snippets are constructed here (not in expand_and_pack) because snippet
        # building requires the query text, and the pipeline is the natural place
        # where both evidence and query are co-located.
        # Resolve canonical_uri and title from the store's sources table.  One
        # lookup per distinct source_id (memoised in _source_cache) to avoid
        # repeated round-trips for chunks from the same document.
        _source_cache: dict[str, tuple[str, str]] = {}
        snippeted_evidence: list[Evidence] = []
        for ev in expansion.evidence:
            chunk = ev.hit.chunk
            sid = str(chunk.source_id)
            if sid not in _source_cache:
                info = self._store.get_source_info(sid)
                if info is not None:
                    _source_cache[sid] = info
                else:
                    # Source row missing - use empty strings so the hash never
                    # leaks into source_uri (a hash is not a navigable URI).
                    _source_cache[sid] = ("", "")
            canonical_uri, title = _source_cache[sid]
            snip = build_snippet(
                chunk.text,
                effective_query.text,
                source_id=sid,
                source_uri=canonical_uri,
                title=title,
                locator=chunk.parent_locator,
                chunk_id=str(chunk.id),
            )
            # frozen=True: reconstruct with snippet attached.
            snippeted_evidence.append(
                Evidence(
                    id=ev.id,
                    hit=ev.hit,
                    citation_label=ev.citation_label,
                    role=ev.role,
                    context_of=ev.context_of,
                    snippet=snip,
                )
            )

        # Produce the plain-text recap (required before prompt construction).
        recap = summarize_budget(expansion.budget_summary)

        return SearchResult(
            evidence=snippeted_evidence,
            budget_summary=expansion.budget_summary,
            budget_recap=recap,
        )
