"""Hybrid retrieval: one Qdrant Query API request per search, filters enforced.

``HybridRetriever.search`` is the single search path Tasks 03.2 through 03.4
and Epic 05 reuse.  Per search it:

1. Gates on corpus readiness (typed ``ReadinessError`` for non-READY states).
2. Compiles the ``FilterSpec`` at the boundary via ``compile_filter``.
3. Embeds the query text once with the injected embedder (zero LLM calls by
   construction: this module has no LLM dependency at all).
4. Builds one ``HybridQueryRequest``: dense and sparse prefetch branches fused
   with Qdrant's native RRF, the compiled filter attached to *every* branch
   and to the top level; or the sparse-only degraded form when no dense query
   vector is available.
5. Hands the request to the ``QueryExecutor`` seam.  Because the filter is
   compiled into the request before any executor runs, an alternative
   executor implementation still operates under the filter.
6. Optionally reranks the fused candidates (bounded: the reranker only ever
   sees the candidate list) and returns typed ``Hit`` records.

Score direction: ``fused_score`` is the Qdrant RRF fused score (higher is
better); ``rerank_score`` is the cross-encoder relevance score (higher is
better), ``None`` unless a reranker ran.  The sparse component uses term
frequency only (no IDF); RRF fusion is rank-based so IDF assumptions are not
made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client.http import models as qmodels

from beacon.errors import BackendError, ReadinessError
from beacon.ingest.embeddings import Embedder
from beacon.retrieval.filters import FilterSpec, compile_filter
from beacon.state.db import StateDB
from beacon.state.repo import CorpusState, derive_corpus_state
from beacon.storage.payload import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from beacon.storage.qdrant import HybridQueryRequest, QdrantStore, QueryResult

DEFAULT_PREFETCH_LIMIT = 50
"""Candidates fetched per prefetch branch before fusion."""

__all__ = [
    "DEFAULT_PREFETCH_LIMIT",
    "Hit",
    "HybridQueryRequest",
    "HybridRetriever",
    "QdrantQueryExecutor",
    "QueryExecutor",
    "Reranker",
]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """One retrieval result with per-stage scores.

    Attributes:
        chunk_point_id: Qdrant point ID (UUID string derived from the chunk id).
        payload:        Full chunk payload (``ChunkPayload.to_dict`` shape).
        fused_score:    Qdrant fused (RRF) or single-branch score; higher is
                        better.
        rerank_score:   Cross-encoder relevance score, higher is better;
                        ``None`` when no reranker ran.
    """

    chunk_point_id: str
    payload: dict[str, Any]
    fused_score: float
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------


class QueryExecutor(Protocol):
    """Executes a fully compiled hybrid query request.

    The pipeline compiles the payload filter into the request *before* this
    seam is invoked, so implementations receive the filter on every branch
    and cannot widen the result set beyond it.
    """

    def execute(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Run the request and return scored results."""
        ...


class QdrantQueryExecutor:
    """Default executor: one ``query_points`` call through ``QdrantStore``."""

    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    def execute(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Run the request via ``QdrantStore.query_hybrid``."""
        return self._store.query_hybrid(request)


class Reranker(Protocol):
    """Reorders fused candidates; never fetches new ones."""

    def rerank(self, query_text: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return the same hits reordered with ``rerank_score`` attached."""
        ...


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """The single hybrid search path over a live Qdrant collection.

    Args:
        store:          Qdrant store (alias resolution + query execution).
        db:             State DB for corpus-readiness derivation.
        embedder:       Query-time embedder; must be configured the same way
                        as the embedder the collection was synced with.
        executor:       Query executor seam; defaults to the Qdrant executor.
        reranker:       Optional bounded reranker; ``None`` disables reranking
                        entirely (no model import happens).
        prefetch_limit: Candidates per prefetch branch before fusion.
    """

    def __init__(
        self,
        *,
        store: QdrantStore,
        db: StateDB,
        embedder: Embedder,
        executor: QueryExecutor | None = None,
        reranker: Reranker | None = None,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    ) -> None:
        self._store = store
        self._db = db
        self._embedder = embedder
        self._executor: QueryExecutor = (
            executor if executor is not None else QdrantQueryExecutor(store)
        )
        self._reranker = reranker
        self._prefetch_limit = prefetch_limit

    def search(
        self,
        query_text: str,
        filter_spec: FilterSpec,
        top_k: int = 10,
    ) -> list[Hit]:
        """Search the collection named by ``filter_spec.collection``.

        Args:
            query_text:  Natural-language query text.
            filter_spec: Typed constraints; compiled once at this boundary.
            top_k:       Maximum number of hits to return.

        Returns:
            Ranked ``Hit`` records, best first.

        Raises:
            ReadinessError: When the corpus is EMPTY, BUILDING, or FAILED.
            BackendError:   On Qdrant failures or embedding-mode mismatch
                            between the query embedder and the collection.
        """
        collection = filter_spec.collection
        state = derive_corpus_state(self._db, collection_name=collection)
        if state is not CorpusState.READY:
            raise ReadinessError(
                f"Collection {collection!r} is not searchable: corpus state is "
                f"{state.value!r} (a live revision is required)"
            )

        query_filter = compile_filter(filter_spec)

        # Hoist collection-existence check for BOTH dense and sparse-only modes.
        # Previously, _check_dense_dimension (called only in the dense branch) raised
        # BackendError when the physical collection was missing, but sparse-only
        # searches called the executor directly and returned empty results silently.
        # Now both modes fail with a typed BackendError when the live revision points
        # at a missing physical collection.
        physical = self._store.resolve_alias(collection)
        target = physical if physical is not None else collection
        if self._store.collection_info(target) is None:
            raise BackendError(
                f"Collection {collection!r} has a live revision but no "
                f"queryable Qdrant collection ({target!r})"
            )

        embedding = self._embedder.embed([query_text])[0]
        sparse: qmodels.SparseVector | None = None
        if embedding.sparse_indices:
            sparse = qmodels.SparseVector(
                indices=embedding.sparse_indices,
                values=embedding.sparse_values,
            )

        if embedding.dense is not None:
            self._check_dense_dimension(collection, len(embedding.dense))
            request = self._build_hybrid_request(
                collection=collection,
                dense=embedding.dense,
                sparse=sparse,
                query_filter=query_filter,
                top_k=top_k,
            )
        elif sparse is not None:
            request = HybridQueryRequest(
                collection_name=collection,
                prefetch=(),
                query=sparse,
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
            )
        else:
            # Blank query in sparse-only mode: nothing to search with.
            return []

        results = self._executor.execute(request)
        hits = [
            Hit(
                chunk_point_id=result.id,
                payload=result.payload or {},
                fused_score=result.score,
            )
            for result in results
        ]

        if self._reranker is not None:
            hits = self._reranker.rerank(query_text, hits)

        return hits[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_hybrid_request(
        self,
        *,
        collection: str,
        dense: list[float],
        sparse: qmodels.SparseVector | None,
        query_filter: qmodels.Filter | None,
        top_k: int,
    ) -> HybridQueryRequest:
        """Build the fused request with the filter on every prefetch branch."""
        branches: list[qmodels.Prefetch] = [
            qmodels.Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=self._prefetch_limit,
            )
        ]
        if sparse is not None:
            branches.append(
                qmodels.Prefetch(
                    query=sparse,
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=self._prefetch_limit,
                )
            )
        return HybridQueryRequest(
            collection_name=collection,
            prefetch=tuple(branches),
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            using=None,
            query_filter=query_filter,
            limit=top_k,
        )

    def _check_dense_dimension(self, collection: str, query_dim: int) -> None:
        """Reject a query whose dense dimension differs from the collection's.

        A dimension mismatch means the query embedder is not the embedding
        mode/model the collection was synced with; failing typed here beats
        silently returning wrong nearest neighbours.

        Raises:
            BackendError: On dimension mismatch or unresolvable collection.
        """
        physical = self._store.resolve_alias(collection)
        target = physical if physical is not None else collection
        info = self._store.collection_info(target)
        if info is None:
            raise BackendError(
                f"Collection {collection!r} has a live revision but no "
                f"queryable Qdrant collection ({target!r})"
            )
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            dense_params = vectors_config.get(DENSE_VECTOR_NAME)
            if dense_params is not None and dense_params.size != query_dim:
                raise BackendError(
                    f"Embedding-mode mismatch for collection {collection!r}: "
                    f"query dense dimension {query_dim} does not match the "
                    f"collection's dense dimension {dense_params.size}; the "
                    f"query embedder must match the mode/model the collection "
                    f"was synced with"
                )
