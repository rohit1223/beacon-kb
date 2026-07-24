"""POST /answer endpoint: grounded answer with exactly one LLM call (Task 03.4).

The answer route retrieves an EvidenceBundle via the same shared pipeline as
/search (``build_evidence_bundle``), then calls ``run_answer`` which makes
exactly one LLM provider call (zero on pre-abstention).  Abstention is data,
not an error: even when the pipeline abstains, the response is HTTP 200 with
``abstained=true``.

State seam for the LLM client: the route reads ``app.state.llm_client``.
- ``None`` (the default set by the lifespan): a ``LiteLlmClient`` is created
  lazily, so importing this module has no provider dependency.
- Any other value: used directly as the LLM client.
This lets tests inject a deterministic counting fake through the production
wiring path after the lifespan starts::

    with TestClient(app) as c:
        app.state.llm_client = counting_fake
        r = c.post("/answer", json=...)

The route is a thin adapter over the Task 03.1-03.3 pipeline functions; it
re-implements no pipeline logic, so the in-process cost contracts (search
zero calls, answer exactly one) hold identically over HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from beacon.answer.generate import LiteLlmClient, LlmClient, run_answer
from beacon.models import AnswerResult
from beacon.server.routes.search import DateRangeFilter, build_evidence_bundle
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

    top_k: int = Field(default=10, ge=1)
    """Maximum number of evidence items to retrieve."""

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
    """Optional LLM model override.  When None, uses the configured default model."""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/answer", status_code=200)
async def answer(request: Request, body: AnswerRequest) -> JSONResponse:
    """Grounded answer with exactly one LLM call.

    Performs retrieval and evidence assembly identically to /search, then calls
    the answer pipeline.  Abstention produces HTTP 200 with ``abstained=true``
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

    # Resolve the LLM client from the app-state seam (supports test injection).
    llm_client: LlmClient | None = getattr(request.app.state, "llm_client", None)
    if llm_client is None:
        llm_client = LiteLlmClient()

    model = body.model if body.model is not None else settings.models.llm_model

    with pipeline_span("answer_generation", collection=body.collection):
        result: AnswerResult = run_answer(
            bundle,
            body.query,
            llm_client,
            model=model,
            temperature=settings.answer.temperature,
            max_tokens=settings.answer.max_tokens,
        )

    return JSONResponse(content=result.model_dump(mode="json"), status_code=200)
