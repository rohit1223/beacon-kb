"""Beacon storage layer: Qdrant store, shadow-collection lifecycle, and payload schema.

Public surface
--------------
``QdrantStore``
    Typed wrapper around ``qdrant-client`` with embedded/server mode selection
    and backend-error translation.  Constructed from ``BeaconSettings``.

``begin_stage``, ``promote``, ``abort``, ``StageHandle``
    Staged promotion protocol: write to a shadow collection, then atomically
    flip the alias so the live alias always points at a fully built collection.

``ChunkPayload``
    Typed payload dataclass stored alongside every Qdrant point.

``DENSE_VECTOR_NAME``, ``SPARSE_VECTOR_NAME``
    Named-vector constants shared by the ingestion (Epic 02) and retrieval
    (Epic 03) layers.

``PAYLOAD_INDEX_FIELDS``
    Filterable-field declarations created at collection time.
"""

from __future__ import annotations

from beacon.storage.lifecycle import StageHandle, abort, begin_stage, promote
from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
    ChunkPayload,
)
from beacon.storage.qdrant import QdrantStore, QueryResult

__all__ = [
    "DENSE_VECTOR_NAME",
    "PAYLOAD_INDEX_FIELDS",
    "SPARSE_VECTOR_NAME",
    "ChunkPayload",
    "QdrantStore",
    "QueryResult",
    "StageHandle",
    "abort",
    "begin_stage",
    "promote",
]
