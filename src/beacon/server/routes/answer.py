"""POST /answer endpoint: grounded answer with exactly one LLM call (Task 03.4).

The answer route retrieves an EvidenceBundle via run_search_pipeline, then
calls ``run_answer`` which makes exactly one LLM provider call (zero on
pre-abstention). Abstention is data, not an error: even when the pipeline
abstains, the response is HTTP 200 with ``abstained=true``.

State seam for the LLM client: the route reads ``app.state.llm_client``.
- ``None`` (the default set by the lifespan): a ``LiteLlmClient`` is created
  lazily, so importing this module has no provider dependency.
- Any other value: used directly as the LLM client.
This lets tests inject a deterministic counting fake through the production
wiring path after the lifespan starts::

    with TestClient(app) as c:
        app.state.llm_client = counting_fake
        r = c.post("/answer", json=...)

The route is a thin adapter over the transport-free pipeline in
``beacon.retrieval.pipeline``; it re-implements no pipeline logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from beacon.answer.generate import LiteLlmClient, LlmClient, run_answer
from beacon.models import AnswerResult
from beacon.retrieval.filters import FilterSpec
from beacon.retrieval.pipeline import TOKEN_BUDGET, run_search_pipeline
from beacon.server.routes.search import DateRangeFilter, _resolve_embedder
from beacon.server.telemetry import pipeline_span

router = APIRouter(tags=["answer"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class AnswerRequest(BaseModel):
    """Request body for POST /answer."""

    collection: str
    """Logical collection name to search."""

    query: str
    """Natural-language question to answer."""

    top_k: int | None = Field(default=None, ge=1)
    """Maximum number of evidence items to retrieve.

    When None, uses ``RetrievalSettings.top_k`` from server config (default 10).
    """

    sources: list[str] = Field(default_factory=list)
    """Restrict retrieval to these source URIs (OR semantics; empty means no filter)."""

    tags: list[str] = Field(default_factory=list)
    """Restrict retrieval to chunks carrying any of these tags (empty means no filter)."""

    created: DateRangeFilter | None = None
    """Optional date range on the source document ``created_at`` field."""

    modified: DateRangeFilter | None = None
    """Optional date range on the source document ``modified_at`` field."""

    ingested: DateRangeFilter | None = None
    """Optional date range on the pipeline ``ingested_at`` field."""

    model: str | None = Field(default=None)
    """Optional LLM model override. When None, uses the configured default model."""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/answer", status_code=200, response_model=AnswerResult)
async def answer(request: Request, body: AnswerRequest) -> AnswerResult:
    """Grounded answer with exactly one LLM call.

    Performs retrieval and evidence assembly identically to /search, then calls
    the answer pipeline. Abstention produces HTTP 200 with ``abstained=true``
    in the response body - it is data, not an error.

    Args:
        request: FastAPI request (provides app.state).
        body:    Validated answer request.

    Returns:
        200 JSON response with the AnswerResult (may be abstained).
        503 problem+json if the collection is not ready (ReadinessError).
        422 problem+json if the model cites a label absent from the bundle
        (CitationError).
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

    # Resolve the LLM client from the app-state seam (supports test injection).
    llm_client: LlmClient | None = getattr(request.app.state, "llm_client", None)
    if llm_client is None:
        llm_client = LiteLlmClient(timeout=settings.answer.llm_timeout_s)

    model = body.model if body.model is not None else settings.models.llm_model

    abstain_threshold = (
        settings.retrieval.score_threshold
        if settings.answer.abstain_when_uncertain
        else 0.0
    )

    with pipeline_span("answer_generation", collection=body.collection):
        result: AnswerResult = run_answer(
            bundle,
            body.query,
            llm_client,
            model=model,
            temperature=settings.answer.temperature,
            max_tokens=settings.answer.max_tokens,
            abstain_threshold=abstain_threshold,
        )

    return result
