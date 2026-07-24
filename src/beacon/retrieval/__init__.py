"""Retrieval pipeline package (Epic 03+).

Hybrid search, reranking, and parent/context expansion live here.

Public surface:

- ``FilterSpec`` / ``DateRange`` / ``compile_filter`` - typed filters and the
  single boundary compiler to Qdrant payload filters.
- ``HybridRetriever`` / ``Hit`` - the single search path and its typed result.
- ``QueryExecutor`` / ``QdrantQueryExecutor`` - the execution seam.
- ``Reranker`` / ``CrossEncoderReranker`` - optional bounded reranking.
"""

from __future__ import annotations

from beacon.retrieval.filters import DateRange, FilterSpec, compile_filter
from beacon.retrieval.hybrid import (
    DEFAULT_PREFETCH_LIMIT,
    Hit,
    HybridQueryRequest,
    HybridRetriever,
    QdrantQueryExecutor,
    QueryExecutor,
    Reranker,
)
from beacon.retrieval.rerank import CrossEncoderReranker, Scorer

__all__ = [
    "DEFAULT_PREFETCH_LIMIT",
    "CrossEncoderReranker",
    "DateRange",
    "FilterSpec",
    "Hit",
    "HybridQueryRequest",
    "HybridRetriever",
    "QdrantQueryExecutor",
    "QueryExecutor",
    "Reranker",
    "Scorer",
    "compile_filter",
]
