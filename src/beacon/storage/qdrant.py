"""Qdrant client wrapper with embedded/server mode selection and typed error translation.

``QdrantStore`` is constructed from ``BeaconSettings`` and selects the
operating mode at construction time:

- **Embedded** (default): ``QdrantClient(path=...)`` runs Qdrant in-process
  from a local path.  No server, no network.  Suitable for development and
  single-process deployments.
- **Server**: ``QdrantClient(url=..., api_key=...)`` connects to a remote
  Qdrant instance.

The selected mode is exposed via ``QdrantStore.mode`` for ``/readyz``
reporting.

All ``qdrant_client`` exceptions are caught and re-raised as typed
``BackendError`` instances so callers never see raw client exceptions.

Live reads always resolve the logical name through an alias.  Querying a
logical name with no alias returns an empty list rather than raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from beacon.config import BeaconSettings
from beacon.errors import BackendError
from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
    ChunkPayload,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class QueryResult:
    """Thin wrapper around a Qdrant ``ScoredPoint`` for typed access."""

    def __init__(self, point: qmodels.ScoredPoint) -> None:
        self._point = point

    @property
    def id(self) -> str:
        return str(self._point.id)

    @property
    def score(self) -> float:
        return float(self._point.score)

    @property
    def payload(self) -> dict[str, Any] | None:
        raw = self._point.payload
        if raw is None:
            return None
        return dict(raw)


# ---------------------------------------------------------------------------
# Hybrid query request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridQueryRequest:
    """One fully compiled Qdrant Query API request for a hybrid search.

    Built by the retrieval pipeline *after* the ``FilterSpec`` has been
    compiled: the payload filter is embedded in every prefetch branch and at
    the top level, so any executor implementation that runs this request
    operates under the compiled filter by construction.

    Two forms exist:

    - **Hybrid**: ``prefetch`` holds the dense and sparse branches and
      ``query`` is a ``FusionQuery`` (native RRF); ``using`` is ``None``.
    - **Sparse-only (degraded)**: ``prefetch`` is empty, ``query`` is the
      sparse query vector, and ``using`` names the sparse vector.

    Attributes:
        collection_name: Logical alias or physical collection to query.
        prefetch:        Prefetch branches (empty for sparse-only).
        query:           Fusion directive or the sparse query vector.
        using:           Named vector for the direct query form, else ``None``.
        query_filter:    Compiled payload filter applied inside Qdrant.
        limit:           Maximum number of fused results.
    """

    collection_name: str
    prefetch: tuple[qmodels.Prefetch, ...]
    query: qmodels.FusionQuery | qmodels.SparseVector
    using: str | None
    query_filter: qmodels.Filter | None
    limit: int


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class QdrantStore:
    """Thin typed wrapper around ``QdrantClient``.

    Responsibilities:
    - mode selection (embedded vs server)
    - collection CRUD
    - alias management (set, delete, resolve)
    - point upsert with batching
    - point query
    - backend-error translation

    Callers never write to a logical alias name directly; they write to the
    physical collection returned by ``lifecycle.begin_stage`` and read through
    the alias via ``query``.
    """

    def __init__(self, settings: BeaconSettings) -> None:
        self._settings = settings
        qdrant_cfg = settings.qdrant

        if qdrant_cfg.url is None:
            self._mode = "embedded"
            logger.info("Qdrant: embedded mode, path=%s", qdrant_cfg.path)
            self._client = QdrantClient(
                path=qdrant_cfg.path,
                timeout=int(qdrant_cfg.timeout),
            )
        else:
            self._mode = "server"
            api_key = (
                qdrant_cfg.api_key.get_secret_value() if qdrant_cfg.api_key else None
            )
            logger.info("Qdrant: server mode, url=%s", qdrant_cfg.url)
            self._client = QdrantClient(
                url=qdrant_cfg.url,
                api_key=api_key,
                timeout=int(qdrant_cfg.timeout),
            )

    @property
    def mode(self) -> str:
        """``'embedded'`` or ``'server'``; exposed for ``/readyz`` reporting."""
        return self._mode

    def close(self) -> None:
        """Close the underlying Qdrant client connection.

        For embedded mode, this closes the in-process connection gracefully.
        For server mode, this closes the HTTP session.
        Safe to call multiple times; the second and subsequent calls are no-ops.
        """
        try:
            self._client.close()
        except Exception as exc:
            # Log at debug level; suppress exceptions during cleanup to avoid
            # masking primary errors in the calling context.
            logger.debug("Failed to close Qdrant client: %s", exc)

    # ------------------------------------------------------------------
    # Collection CRUD
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return physical collection names known to Qdrant."""
        try:
            resp = self._client.get_collections()
            return [c.name for c in resp.collections]
        except Exception as exc:
            raise BackendError(f"Failed to list collections: {exc}") from exc

    def create_collection(
        self,
        name: str,
        dense_dim: int,
    ) -> None:
        """Create a physical collection with dense + sparse named vectors.

        Idempotent: if the collection already exists the call is a no-op.
        Payload indexes are created for all filterable fields declared in
        ``PAYLOAD_INDEX_FIELDS``.
        """
        try:
            if self._client.collection_exists(name):
                return
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR_NAME: qmodels.VectorParams(
                        size=dense_dim,
                        distance=qmodels.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(),
                },
            )
            for field_name, schema_type in PAYLOAD_INDEX_FIELDS:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
        except Exception as exc:
            raise BackendError(f"Failed to create collection '{name}': {exc}") from exc

    def delete_collection(self, name: str) -> None:
        """Delete a physical collection; no-op if the collection does not exist."""
        try:
            if not self._client.collection_exists(name):
                return
            self._client.delete_collection(name)
        except Exception as exc:
            raise BackendError(f"Failed to delete collection '{name}': {exc}") from exc

    def collection_info(self, name: str) -> qmodels.CollectionInfo | None:
        """Return collection info or ``None`` if the collection does not exist.

        Returns a ``qmodels.CollectionInfo`` raw model as a deliberate pass-through;
        see Epic 03 for planned typed filter DSL.
        """
        try:
            if not self._client.collection_exists(name):
                return None
            return self._client.get_collection(name)
        except Exception as exc:
            raise BackendError(f"Failed to get info for '{name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Alias management
    # ------------------------------------------------------------------

    def resolve_alias(self, logical_name: str) -> str | None:
        """Return the physical collection name for ``logical_name``, or ``None``.

        Iterates ``get_aliases()`` to find an alias whose ``alias_name``
        matches ``logical_name``.  This is the reliable approach for both
        embedded and server modes: ``get_collection_aliases(name)`` returns
        aliases *for* a physical collection, not *by* alias name.
        """
        try:
            all_aliases = self._client.get_aliases()
            for alias in all_aliases.aliases:
                if alias.alias_name == logical_name:
                    return alias.collection_name
            return None
        except Exception as exc:
            raise BackendError(f"Failed to resolve alias '{logical_name}': {exc}") from exc

    def set_alias(self, alias_name: str, collection_name: str) -> None:
        """Atomically create or retarget ``alias_name`` to ``collection_name``.

        Uses a single ``update_collection_aliases`` call with a
        ``CreateAliasOperation`` so the swap is atomic from Qdrant's
        perspective.
        """
        try:
            self._client.update_collection_aliases(
                change_aliases_operations=[
                    qmodels.CreateAliasOperation(
                        create_alias=qmodels.CreateAlias(
                            collection_name=collection_name,
                            alias_name=alias_name,
                        )
                    )
                ]
            )
        except Exception as exc:
            raise BackendError(
                f"Failed to set alias '{alias_name}' -> '{collection_name}': {exc}"
            ) from exc

    def delete_alias(self, alias_name: str) -> None:
        """Delete an alias; no-op if it does not exist."""
        try:
            self._client.update_collection_aliases(
                change_aliases_operations=[
                    qmodels.DeleteAliasOperation(
                        delete_alias=qmodels.DeleteAlias(alias_name=alias_name)
                    )
                ]
            )
        except Exception as exc:
            raise BackendError(f"Failed to delete alias '{alias_name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        points: list[tuple[str, dict[str, Any], ChunkPayload]],
        batch_size: int = 100,
    ) -> None:
        """Upsert ``points`` into ``collection_name`` in batches.

        Each element of ``points`` is ``(id, vectors_dict, payload)``.
        ``vectors_dict`` keys must match the named-vector names used when the
        collection was created (i.e. ``DENSE_VECTOR_NAME`` and optionally
        ``SPARSE_VECTOR_NAME``).

        Raises ``BackendError`` for any Qdrant failure.

        Note:
            Qdrant point IDs must be either UUID strings or unsigned 64-bit integers.
            String IDs that are not valid UUIDs will be rejected by the Qdrant client
            at runtime. Ensure all point IDs passed as the first element of each
            tuple are valid UUID strings or unsigned integers.
        """
        try:
            structs: list[qmodels.PointStruct] = [
                qmodels.PointStruct(
                    id=pid,
                    vector=vectors,
                    payload=payload.to_dict(),
                )
                for pid, vectors, payload in points
            ]
            for start in range(0, len(structs), batch_size):
                batch = structs[start : start + batch_size]
                self._client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                f"Failed to upsert into '{collection_name}': {exc}"
            ) from exc

    def upsert_records(
        self,
        collection_name: str,
        records: list[qmodels.Record],
        batch_size: int = 100,
    ) -> None:
        """Upsert previously scrolled ``Record`` objects into ``collection_name``.

        Used by the sync engine to carry over unchanged-source points from a
        prior physical collection into a shadow collection verbatim (same IDs,
        vectors, and payloads) without re-embedding.

        Args:
            collection_name: Physical collection to write into.
            records:         Records from ``scroll_by_source_uri`` (must carry
                             vectors and payloads).
            batch_size:      Points per upsert batch.

        Raises:
            BackendError: On any Qdrant failure.
        """
        try:
            structs = [
                qmodels.PointStruct(
                    id=rec.id,
                    vector=rec.vector,  # type: ignore[arg-type]
                    payload=rec.payload,
                )
                for rec in records
            ]
            for start in range(0, len(structs), batch_size):
                batch = structs[start : start + batch_size]
                self._client.upsert(collection_name=collection_name, points=batch)
        except Exception as exc:
            raise BackendError(
                f"Failed to upsert records into '{collection_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll_by_source_uri(
        self,
        collection_name: str,
        source_uri: str,
    ) -> list[qmodels.Record]:
        """Scroll all points for a given source_uri in collection_name.

        Used by the sync engine to carry over unchanged-source points from a
        prior collection into the shadow collection without re-embedding.

        Args:
            collection_name: Physical collection to scroll.
            source_uri:      Source URI to filter on (payload field ``source_uri``).

        Returns:
            List of ``qmodels.Record`` with full payload and vectors.

        Raises:
            BackendError: On Qdrant failure.
        """
        scroll_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="source_uri",
                    match=qmodels.MatchValue(value=source_uri),
                )
            ]
        )

        all_records: list[qmodels.Record] = []
        offset: Any = None
        try:
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                all_records.extend(records)
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            raise BackendError(
                f"Failed to scroll '{collection_name}'"
                f" for source_uri={source_uri!r}: {exc}"
            ) from exc
        return all_records

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        collection_name: str,
        vector: list[float],
        using: str = DENSE_VECTOR_NAME,
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: qmodels.Filter | None = None,
    ) -> list[QueryResult]:
        """Query ``collection_name`` (or logical alias) by ``vector``.

        If ``collection_name`` resolves through an alias to a physical
        collection, the query runs against that physical collection.
        If the logical name has no alias and no physical collection by that
        name exists, returns an empty list rather than raising.

        The ``query_filter`` parameter accepts ``qmodels.Filter`` as a deliberate
        pass-through; see Epic 03 for planned typed filter DSL.
        """
        # Resolve alias to physical collection
        physical = self.resolve_alias(collection_name)
        target = physical if physical is not None else collection_name

        try:
            if not self._client.collection_exists(target):
                return []
            results = self._client.query_points(
                collection_name=target,
                query=vector,
                using=using,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            return [QueryResult(p) for p in results.points]
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Failed to query '{collection_name}': {exc}") from exc

    def query_hybrid(self, request: HybridQueryRequest) -> list[QueryResult]:
        """Execute one hybrid Query API request against Qdrant.

        Issues exactly one ``query_points`` call.  For the hybrid form the
        request carries dense and sparse prefetch branches fused with a
        native ``FusionQuery``; for the sparse-only degraded form it carries
        a single sparse query with no prefetch.  In both forms the compiled
        payload filter inside ``request`` is applied by Qdrant itself.

        ``request.collection_name`` may be a logical alias or a physical
        collection name; aliases are resolved before querying.  A missing
        collection returns an empty list rather than raising, matching
        ``query``.

        Args:
            request: The fully compiled hybrid query request.

        Returns:
            Scored points wrapped as ``QueryResult`` records.

        Raises:
            BackendError: On any Qdrant failure.
        """
        physical = self.resolve_alias(request.collection_name)
        target = physical if physical is not None else request.collection_name

        try:
            if not self._client.collection_exists(target):
                return []
            results = self._client.query_points(
                collection_name=target,
                prefetch=list(request.prefetch) if request.prefetch else None,
                query=request.query,
                using=request.using,
                query_filter=request.query_filter,
                limit=request.limit,
                with_payload=True,
            )
            return [QueryResult(p) for p in results.points]
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                f"Failed hybrid query on '{request.collection_name}': {exc}"
            ) from exc
