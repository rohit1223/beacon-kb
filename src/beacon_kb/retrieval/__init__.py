"""Retrieval package for beacon-kb hybrid sparse and dense candidate retrieval.

Exports the public surface of the retrieval package without importing providers.
Both BM25SparseRetriever and EmbedderDenseRetriever are registered as built-in
plugins in the beacon_kb.retrievers entry-point group.

Registry convention:
- The beacon_kb.retrievers group's default protocol maps to SparseRetriever.
  Resolving a SparseRetriever: ``registry.resolve(groups.RETRIEVERS, 'bm25')``
- Dense retriever requires an explicit protocol kwarg (documented escape hatch):
  ``registry.resolve(groups.RETRIEVERS, 'dense', protocol=DenseRetriever)``
  This is because the group's canonical protocol is SparseRetriever; passing
  ``protocol=DenseRetriever`` bypasses the automatic SparseRetriever check.

Importing this module performs no side effects beyond defining the public names.
"""

from __future__ import annotations

from beacon_kb.retrieval.dense import EmbedderDenseRetriever
from beacon_kb.retrieval.filters import FilterSpec, apply_filters
from beacon_kb.retrieval.query import QueryVariants, prepare_query
from beacon_kb.retrieval.sparse import BM25SparseRetriever

__all__ = [
    "BM25SparseRetriever",
    "EmbedderDenseRetriever",
    "FilterSpec",
    "QueryVariants",
    "apply_filters",
    "prepare_query",
]
