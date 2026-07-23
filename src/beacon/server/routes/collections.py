"""Collections REST resource for the Beacon server (Task 01.5).

Routes:
    POST /collections       - Register a new logical collection (state-DB only).
    GET  /collections       - List all registered collections with corpus state.
    GET  /collections/{name} - Detail for a single collection with revision and
                               last-job summary.

Design invariants:
- Creating a collection performs NO Qdrant write; the physical Qdrant
  collection is created only at first sync staging (Epic 02).
- Collection names are validated against a conservative pattern:
  lowercase alphanumerics, dash, underscore, bounded to 64 characters.
- Duplicate names return 409 problem+json.
- Unknown names in the detail route return 404 problem+json.
- All error responses use Content-Type: application/problem+json.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from beacon.errors import ConflictError, NotFoundError
from beacon.models import (
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    LastJobSummary,
)
from beacon.problems import error_to_problem, problem_to_dict
from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    RevisionRepo,
    SyncJobRepo,
    derive_corpus_state,
)

_PROBLEM_CONTENT_TYPE = "application/problem+json"

router = APIRouter(prefix="/collections", tags=["collections"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_collection_response(
    db: StateDB,
    name: str,
    created_at: str,
    *,
    include_detail: bool = False,
) -> CollectionResponse:
    """Build a CollectionResponse from the state DB.

    Args:
        db:             Open StateDB instance.
        name:           Collection name.
        created_at:     ISO 8601 timestamp from the DB row.
        include_detail: When True, also fetches live revision and last job.

    Returns:
        CollectionResponse with corpus_state and optional detail fields.
    """
    corpus_state = derive_corpus_state(db, collection_name=name)

    live_revision: str | None = None
    last_job: LastJobSummary | None = None

    if include_detail:
        rev_repo = RevisionRepo(db)
        live_row = rev_repo.get_live(collection_name=name)
        if live_row is not None:
            live_revision = str(live_row["revision_id"])

        job_repo = SyncJobRepo(db)
        jobs = job_repo.list_by_collection(name)
        if jobs:
            j = jobs[0]
            last_job = LastJobSummary(
                job_id=str(j["job_id"]),
                state=str(j["state"]),
                created_at=str(j["created_at"]),
                finished_at=str(j["finished_at"]) if j["finished_at"] is not None else None,
            )

    return CollectionResponse(
        name=name,
        corpus_state=corpus_state.value,
        created_at=created_at,
        live_revision=live_revision,
        last_job=last_job,
    )


def _problem_response(body: dict[str, object], status: int) -> JSONResponse:
    """Wrap a problem-details dict in a JSONResponse with the correct content-type."""
    return JSONResponse(
        content=body,
        status_code=status,
        headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_collection(
    request: Request,
    body: CollectionCreateRequest,
) -> JSONResponse:
    """Register a new logical collection.

    Validates the name, inserts a row into the state DB, and returns the
    new collection with its initial (empty) corpus state.

    No Qdrant write is performed here; the physical Qdrant collection is
    created lazily at first sync staging.

    Args:
        request: The incoming FastAPI request.
        body:    Validated request body with the collection name.

    Returns:
        201 JSON response with collection detail.
        409 problem+json if the name already exists.
        422 problem+json if the name is invalid (caught by Pydantic).
    """
    db: StateDB = request.app.state.state_db
    repo = CollectionRepo(db)

    # Check for duplicate before inserting.
    existing = repo.get(body.name)
    if existing is not None:
        err = ConflictError(
            f"Collection {body.name!r} already exists."
        )
        problem = error_to_problem(err, instance=f"/collections/{body.name}")
        return _problem_response(problem_to_dict(problem), 409)

    # Insert the collection row (idempotent; duplicate is guarded above).
    repo.create(name=body.name)

    # Fetch the row back to get the canonical created_at timestamp.
    row = repo.get(body.name)
    if row is None:
        from beacon.errors import BackendError
        raise BackendError(
            f"Failed to retrieve newly created collection {body.name!r}"
        )

    response_obj = _build_collection_response(
        db,
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        include_detail=False,
    )
    return JSONResponse(
        content=response_obj.model_dump(mode="json"),
        status_code=201,
    )


@router.get("", status_code=200)
async def list_collections(request: Request) -> CollectionListResponse:
    """List all registered collections with per-collection corpus state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        200 JSON response with a list of collection summaries.
    """
    db: StateDB = request.app.state.state_db
    repo = CollectionRepo(db)
    rows = repo.list()

    items = [
        _build_collection_response(
            db,
            name=str(row["name"]),
            created_at=str(row["created_at"]),
            include_detail=False,
        )
        for row in rows
    ]
    return CollectionListResponse(items=items)


@router.get("/{name}", status_code=200)
async def get_collection(request: Request, name: str) -> JSONResponse:
    """Fetch detail for a single collection.

    Args:
        request: The incoming FastAPI request.
        name:    The collection name from the URL path.

    Returns:
        200 JSON response with full collection detail.
        404 problem+json if the name does not exist.
    """
    db: StateDB = request.app.state.state_db
    repo = CollectionRepo(db)
    row = repo.get(name)

    if row is None:
        err = NotFoundError(f"Collection {name!r} does not exist.")
        problem = error_to_problem(err, instance=f"/collections/{name}")
        return _problem_response(problem_to_dict(problem), 404)

    response_obj = _build_collection_response(
        db,
        name=str(row["name"]),
        created_at=str(row["created_at"]),
        include_detail=True,
    )
    return JSONResponse(
        content=response_obj.model_dump(mode="json"),
        status_code=200,
    )
