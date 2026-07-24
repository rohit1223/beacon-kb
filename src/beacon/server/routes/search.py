"""POST /search endpoint: hybrid retrieval returning an EvidenceBundle (Task 03.4).

The search route performs zero LLM calls by construction: it runs the hybrid
retrieval pipeline and assembles evidence, but never touches the answer
pipeline.  Tests can verify this cost contract by inspecting the call count on
any injected LLM fake.

State seams (shared with /answer via ``build_evidence_bundle``):
- ``app.state.embedder``: query-time embedder.  ``None`` (the lifespan default)
  means an ``EmbedderProvider`` is constructed lazily from settings on first
  use; tests inject a deterministic fake after the lifespan starts.

The route is a thin adapter over the Task 03.1-03.3 pipeline functions
(``HybridRetriever.search`` and ``assemble_evidence``); it re-implements no
pipeline logic, so the in-process cost contracts hold identically over HTTP.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from beacon.ingest.embeddings import Embedder, EmbedderProvider
from beacon.models import EvidenceBundle
from beacon.retrieval.evidence import assemble_evidence
from beacon.retrieval.filters import DateRange, FilterSpec
from beacon.retrieval.hybrid import HybridRetriever
from beacon.server.telemetry import pipeline_span
from beacon.storage.payload import chunk_id_to_point_id
from beacon.storage.qdrant import QdrantStore

TOKEN_BUDGET = 8192
"""Evidence-assembly token budget shared by the search and answer routes."""

router = APIRouter(tags=["search"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class DateRangeFilter(BaseModel):
    """Inclusive datetime bounds for a payload date field.

    Both bounds are optional; omitting a bound means no constraint on that end.
    Datetimes must be ISO-8601 strings with timezone information (e.g.
    ``2025-01-01T00:00:00Z``).

    Attributes:
        gte: Earliest matching timestamp (inclusive), or ``None`` for no floor.
        lte: Latest matching timestamp (inclusive), or ``None`` for no ceiling.
    """

    gte: datetime | None = None
    lte: datetime | None = None

    def to_date_range(self) -> DateRange:
        """Convert to the internal ``DateRange`` used by ``FilterSpec``."""
        return DateRange(gte=self.gte, lte=self.lte)


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    collection: str
    """Logical collection name to search."""

    query: str
    """Natural-language query string."""

    top_k: int = Field(default=10, ge=1)
    """Maximum number of evidence items to return."""

    sources: list[str] = Field(default_factory=list)
    """Restrict results to these source URIs (OR semantics; empty means no filter)."""

    tags: list[str] = Field(default_factory=list)
    """Restrict results to chunks carrying any of these tags (empty means no filter)."""

    created: DateRangeFilter | None = None
    """Optional date range on the source document ``created_at`` field."""

    modified: DateRangeFilter | None = None
    """Optional date range on the source document ``modified_at`` field."""

    ingested: DateRangeFilter | None = None
    """Optional date range on the pipeline ``ingested_at`` field."""


# ---------------------------------------------------------------------------
# Shared retrieval pipeline (used by /search and /answer)
# ---------------------------------------------------------------------------


def _resolve_embedder(request: Request) -> Embedder:
    """Return the app's embedder, constructing the default provider lazily.

    Reads ``app.state.embedder`` (the injection seam set to ``None`` by the
    lifespan).  When unset, constructs an ``EmbedderProvider`` from settings
    and caches it on ``app.state`` so the mode-detection and any local model
    load happen once per process.
    """
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    if embedder is None:
        settings = request.app.state.settings
        embedder = EmbedderProvider(
            model_name=settings.models.embedding_model,
            dimension=settings.models.embedding_dimension,
        )
        request.app.state.embedder = embedder
    return embedder


def _make_fetch_chunk(
    store: QdrantStore,
    collection: str,
) -> Callable[[str], dict[str, Any] | None]:
    """Return a callable mapping hex chunk_id -> payload dict | None.

    Implements the ``fetch_chunk`` seam of ``assemble_evidence``: the hex
    chunk id is translated to its Qdrant point UUID and retrieved through the
    store's typed boundary.  Missing chunks return ``None`` (the neighbor is
    skipped); real Qdrant failures raise ``BackendError`` and surface as a
    502 problem response.

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


def build_evidence_bundle(
    request: Request,
    *,
    collection: str,
    query: str,
    top_k: int,
    sources: list[str],
    tags: list[str],
    created: DateRangeFilter | None = None,
    modified: DateRangeFilter | None = None,
    ingested: DateRangeFilter | None = None,
) -> EvidenceBundle:
    """Run the full retrieval pipeline for one query and return the bundle.

    This is the single retrieval path shared by POST /search and POST /answer:
    filter compilation, hybrid search, and budgeted evidence assembly.  It
    performs zero LLM calls.

    Args:
        request:    FastAPI request (provides ``app.state``).
        collection: Logical collection name.
        query:      Natural-language query text.
        top_k:      Maximum number of primary hits.
        sources:    Source-URI restriction (empty means none).
        tags:       Tag restriction (empty means none).
        created:    Optional date-range filter on the source ``created_at`` field.
        modified:   Optional date-range filter on the source ``modified_at`` field.
        ingested:   Optional date-range filter on the pipeline ``ingested_at`` field.

    Returns:
        The assembled, labeled, budget-bounded EvidenceBundle.

    Raises:
        ReadinessError: When the collection is not READY (handled as a 503
                        problem by the global error handlers).
        BackendError:   On Qdrant or embedding-mode failures (502 problem).
    """
    state_db = request.app.state.state_db
    qdrant_store: QdrantStore = request.app.state.qdrant_store
    embedder = _resolve_embedder(request)

    filter_spec = FilterSpec(
        collection=collection,
        source_uris=tuple(sources),
        tags=tuple(tags),
        created=created.to_date_range() if created is not None else None,
        modified=modified.to_date_range() if modified is not None else None,
        ingested=ingested.to_date_range() if ingested is not None else None,
    )

    retriever = HybridRetriever(
        store=qdrant_store,
        db=state_db,
        embedder=embedder,
    )

    with pipeline_span("retrieval", collection=collection):
        hits = retriever.search(
            query_text=query,
            filter_spec=filter_spec,
            top_k=top_k,
        )

    fetch_chunk = _make_fetch_chunk(qdrant_store, collection)

    with pipeline_span("evidence_assembly", collection=collection):
        return assemble_evidence(
            hits,
            query,
            fetch_chunk,
            token_budget=TOKEN_BUDGET,
        )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/search", status_code=200)
async def search(request: Request, body: SearchRequest) -> JSONResponse:
    """Hybrid search returning an EvidenceBundle.

    Performs zero LLM calls.  The embedder is taken from ``app.state.embedder``
    when present, otherwise constructed lazily from settings.

    Args:
        request: FastAPI request (provides app.state).
        body:    Validated search request.

    Returns:
        200 JSON response with the EvidenceBundle (labels, snippets,
        provenance, scores, and the budget recap).
        503 problem+json if the collection is not ready (ReadinessError).
    """
    bundle = build_evidence_bundle(
        request,
        collection=body.collection,
        query=body.query,
        top_k=body.top_k,
        sources=body.sources,
        tags=body.tags,
        created=body.created,
        modified=body.modified,
        ingested=body.ingested,
    )
    return JSONResponse(content=bundle.model_dump(mode="json"), status_code=200)
