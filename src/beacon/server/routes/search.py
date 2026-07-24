"""POST /search endpoint: hybrid retrieval returning an EvidenceBundle (Task 03.4).

The search route is a thin adapter over run_search_pipeline from
beacon.retrieval.pipeline. It performs zero LLM calls.

State seams:
- ``app.state.embedder``: query-time embedder. ``None`` (the lifespan default)
  means an ``EmbedderProvider`` is constructed lazily from settings on first
  use; tests inject a deterministic fake after the lifespan starts.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from beacon.ingest.embeddings import Embedder, EmbedderProvider
from beacon.models import EvidenceBundle
from beacon.retrieval.filters import DateRange, FilterSpec
from beacon.retrieval.pipeline import TOKEN_BUDGET, run_search_pipeline
from beacon.server.telemetry import pipeline_span

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

    top_k: int | None = Field(default=None, ge=1)
    """Maximum number of evidence items to return.

    When None, uses ``RetrievalSettings.top_k`` from server config (default 10).
    Explicit values override the config default.
    """

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
# Embedder resolution helper (shared by answer route)
# ---------------------------------------------------------------------------


def _resolve_embedder(request: Request) -> Embedder:
    """Return the app's embedder, constructing the default provider lazily.

    Reads ``app.state.embedder`` (the injection seam set to ``None`` by the
    lifespan). When unset, constructs an ``EmbedderProvider`` from settings
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


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/search", status_code=200, response_model=EvidenceBundle)
async def search(request: Request, body: SearchRequest) -> EvidenceBundle:
    """Hybrid search returning an EvidenceBundle.

    Performs zero LLM calls. The embedder is taken from ``app.state.embedder``
    when present, otherwise constructed lazily from settings.

    Args:
        request: FastAPI request (provides app.state).
        body:    Validated search request.

    Returns:
        200 JSON response with the EvidenceBundle (labels, snippets,
        provenance, scores, and the budget recap).
        503 problem+json if the collection is not ready (ReadinessError).
    """
    settings = request.app.state.settings
    resolved_top_k = body.top_k if body.top_k is not None else settings.retrieval.top_k

    state_db = request.app.state.state_db
    qdrant_store = request.app.state.qdrant_store
    embedder = _resolve_embedder(request)

    spec = FilterSpec(
        collection=body.collection,
        source_uris=tuple(body.sources),
        tags=tuple(body.tags),
        created=body.created.to_date_range() if body.created is not None else None,
        modified=body.modified.to_date_range() if body.modified is not None else None,
        ingested=body.ingested.to_date_range() if body.ingested is not None else None,
    )

    with pipeline_span("retrieval", collection=body.collection):
        with pipeline_span("evidence_assembly", collection=body.collection):
            bundle = run_search_pipeline(
                state_db=state_db,
                store=qdrant_store,
                embedder=embedder,
                spec=spec,
                query_text=body.query,
                top_k=resolved_top_k,
                token_budget=TOKEN_BUDGET,
            )

    return bundle
