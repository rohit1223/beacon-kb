"""Dense vector retrieval with declared similarity semantics and sparse-only fallback.

Design rules enforced here:
- dense_score is the only score set on returned hits; sparse_score, fusion_score,
  and rerank_score remain None.
- Similarity direction is declared explicitly; unknown directions raise BackendError.
- Missing, empty, or dimension-incompatible index raises typed BackendError, never
  silently returns zero scores.
- Sparse-only degraded mode is first-class: when embedder is None, retrieve()
  returns an empty list immediately with zero downloads and zero credentials.
- Dense and sparse candidates keep independent ranks; no cross-normalization.
- Filters are applied consistently before candidates leave this retriever.

Importing this module performs no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beacon_kb.errors import BackendError
from beacon_kb.models import DEFAULT_TOP_K, Hit, Query
from beacon_kb.retrieval.filters import FilterSpec, apply_filters
from beacon_kb.storage.vector_math import validate_similarity

if TYPE_CHECKING:
    from beacon_kb.protocols import Embedder
    from beacon_kb.storage.sqlite import SQLiteStore


class EmbedderDenseRetriever:
    """Dense vector retriever backed by an Embedder and the store's embedding rows.

    Embeds the query through the injected Embedder, then retrieves candidates
    from the store's active embedding rows using a declared similarity semantic.
    When no embedder is configured, returns an empty list immediately -
    no downloads, no credentials, no network calls.

    Score direction: dense_score higher = more relevant (cosine similarity,
    typically [0, 1] for normalized vectors). Only dense_score is set;
    other score fields remain None.

    Dense and sparse candidates keep independent ranks. This retriever does
    not read or normalize sparse_score values.

    Args:
        store:       SQLiteStore instance (read-only via dense_retrieve()).
        embedder:    Embedder instance for query vectorization, or None for
                     sparse-only degraded mode.
        similarity:  Declared similarity direction ('cosine', 'dot', 'euclidean').
        filter_spec: Optional provider-neutral filter applied before returning hits.

    Raises:
        BackendError: If *similarity* is not a supported direction.
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        embedder: Embedder | None,
        similarity: str,
        filter_spec: FilterSpec | None = None,
    ) -> None:
        # Validate similarity direction eagerly - unknown direction is a typed error.
        validate_similarity(similarity)
        self._store: SQLiteStore = store
        self._embedder: Embedder | None = embedder
        self._similarity: str = similarity
        self._filter_spec: FilterSpec = filter_spec if filter_spec is not None else FilterSpec()

    def retrieve(self, query: Query) -> list[Hit]:
        """Return ranked hits using dense vector similarity retrieval.

        When no embedder is configured, returns an empty list immediately
        (sparse-only degraded mode: zero downloads, zero credentials).
        Filters are applied before candidates are returned.

        Args:
            query: Query record with text and optional corpus_id / top_k.

        Returns:
            List of Hit records ordered by dense_score descending (higher is better).
            Each Hit has dense_score set; sparse_score, fusion_score, and rerank_score
            are None.
            Empty list when no embedder is configured or when the index is empty.

        Raises:
            BackendError: On vector store failure or dimension incompatibility.
        """
        # Sparse-only degraded mode: no embedder -> no dense candidates.
        # Zero downloads, zero credentials, zero network calls.
        if self._embedder is None:
            return []

        # Embed the query text using the injected embedder.
        # BackendError from the embedder propagates typed (never swallowed).
        try:
            vectors = self._embedder.embed([query.text])
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Embedder failed during query vectorization: {exc}") from exc

        if not vectors or not vectors[0]:
            raise BackendError(
                "Embedder returned empty vector for query. "
                "Never default missing embedding to zero."
            )

        query_vector: list[float] = vectors[0]

        # Retrieve dense candidates from the store with typed error propagation.
        # Dimension incompatibility raises BackendError from the store.
        # query.top_k is int | None; dense_retrieve() requires int; fall back to DEFAULT_TOP_K.
        effective_top_k: int = query.top_k if query.top_k is not None else DEFAULT_TOP_K
        hits: list[Hit] = self._store.dense_retrieve(
            query_vector=query_vector,
            corpus_id=query.corpus_id,
            top_k=effective_top_k,
            similarity=self._similarity,
        )

        # Apply provider-neutral filters before returning.
        return apply_filters(hits, self._filter_spec)
