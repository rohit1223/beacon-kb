"""Transport-free retrieval pipeline: run_search_pipeline and helpers.

This module extracts the retrieval logic from the HTTP route layer so that
Epic 04/05 (MCP tool layer, agentic loop) can reuse the exact same pipeline
without importing FastAPI, Starlette, or any HTTP-specific code.

All arguments are plain Python objects: no FastAPI Request, no app.state.
The caller is responsible for constructing the store and DB seams.

Pipeline extraction status: DONE (Epic 03 branch review fix).

Importing this module performs no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from beacon.ingest.embeddings import Embedder
from beacon.models import EvidenceBundle
from beacon.retrieval.evidence import assemble_evidence
from beacon.retrieval.filters import FilterSpec
from beacon.retrieval.hybrid import HybridRetriever
from beacon.state.db import StateDB
from beacon.storage.payload import chunk_id_to_point_id
from beacon.storage.qdrant import QdrantStore

__all__ = [
    "TOKEN_BUDGET",
    "build_fetch_chunk",
    "run_search_pipeline",
]

TOKEN_BUDGET = 8192
"""Default evidence-assembly token budget shared by search and answer pipelines.

ROADMAP: promote to a configurable field (RetrievalSettings or ServerSettings)
so operators can tune the cap without a code change (Epic 06).
"""


def build_fetch_chunk(
    store: QdrantStore,
    collection: str,
) -> Callable[[str], dict[str, Any] | None]:
    """Return a callable mapping hex chunk_id -> payload dict | None.

    Implements the ``fetch_chunk`` seam of ``assemble_evidence``: the hex
    chunk id is translated to its Qdrant point UUID and retrieved through the
    store's typed boundary. Missing chunks return None (the neighbor is
    skipped); real Qdrant failures raise BackendError and surface as a
    502 problem response at the HTTP layer.

    Args:
        store:      QdrantStore to retrieve from.
        collection: Logical collection name (alias resolved inside the store).

    Returns:
        A callable suitable for the ``fetch_chunk`` argument of
        ``assemble_evidence``.
    """

    def fetch_chunk(hex_chunk_id: str) -> dict[str, Any] | None:
        point_id = chunk_id_to_point_id(hex_chunk_id)
        return store.retrieve_payload(collection, point_id)

    return fetch_chunk


def run_search_pipeline(
    *,
    state_db: StateDB,
    store: QdrantStore,
    embedder: Embedder,
    spec: FilterSpec,
    query_text: str,
    top_k: int,
    token_budget: int = TOKEN_BUDGET,
    max_neighbor_hops: int = 1,
    max_context_per_hit: int = 2,
) -> EvidenceBundle:
    """Run the full retrieval pipeline for one query and return the bundle.

    This is the single retrieval path shared by the HTTP routes, the MCP
    tool layer (Epic 04), and the agentic loop (Epic 05). It performs zero
    LLM calls and has no dependency on FastAPI, Starlette, or any transport.

    Args:
        state_db:            StateDB for corpus-readiness derivation.
        store:               QdrantStore for search and neighbor fetch.
        embedder:            Query-time embedder instance (Embedder protocol).
        spec:                Compiled FilterSpec (collection + filter constraints).
        query_text:          Natural-language query text (used for search and
                             snippet centering in assemble_evidence).
        top_k:               Maximum number of primary hits to return.
        token_budget:        Evidence-assembly token budget; defaults to TOKEN_BUDGET.
        max_neighbor_hops:   Maximum prev/next hops per primary hit per direction.
        max_context_per_hit: Maximum context chunks added per primary hit.

    Returns:
        The assembled, labeled, budget-bounded EvidenceBundle.

    Raises:
        ReadinessError: When the collection is not READY (no live revision).
        BackendError:   On Qdrant failures, embedding-mode mismatch, or
                        missing physical collection.
    """
    retriever = HybridRetriever(
        store=store,
        db=state_db,
        embedder=embedder,
    )

    hits = retriever.search(
        query_text=query_text,
        filter_spec=spec,
        top_k=top_k,
    )

    fetch_chunk = build_fetch_chunk(store, spec.collection)

    return assemble_evidence(
        hits,
        query_text,
        fetch_chunk,
        token_budget=token_budget,
        max_neighbor_hops=max_neighbor_hops,
        max_context_per_hit=max_context_per_hit,
    )
