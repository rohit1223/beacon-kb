"""Health and readiness routes for the Beacon server (Task 01.4).

Routes:
    GET /healthz - Liveness probe: always 200 while the process is serving.
    GET /readyz  - Readiness probe: per-collection corpus state derived from
                   the state DB; 503 when any backend is unreachable.

The readiness endpoint reports each registered collection as one of:
    empty | building | ready | failed
as derived by ``beacon.state.repo.derive_corpus_state``.

A 503 problem-details response is returned when the state DB raises
``BackendError``, indicating that the backend is unreachable.
The endpoint also pings Qdrant and returns 503 if Qdrant is unreachable.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from beacon.errors import BackendError, ReadinessError
from beacon.problems import problem_to_dict
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, CorpusState, derive_corpus_state
from beacon.storage.qdrant import QdrantStore

_PROBLEM_CONTENT_TYPE = "application/problem+json"

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe.

    Returns 200 with ``{"status": "ok"}`` while the process is serving.
    This endpoint is intentionally simple: if the process can respond it is
    alive.  No backend checks are performed here.

    Returns:
        A JSON body ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness probe with per-collection corpus state.

    Queries the state DB for all registered collections and derives each
    collection's corpus state (empty / building / ready / failed).
    Also pings the Qdrant store to verify it is reachable.

    Returns:
        200 with ``{"status": "ok", "collections": {name: state, ...}, "qdrant": {...}}``
        when all backends are reachable.
        503 problem-details when the state DB or Qdrant is unreachable.

    Args:
        request: The incoming FastAPI request (used to access app state).
    """
    db: StateDB = request.app.state.state_db

    try:
        repo = CollectionRepo(db)
        rows = repo.list()
        collection_states: dict[str, str] = {}
        for row in rows:
            name: str = row["name"]
            state: CorpusState = derive_corpus_state(db, collection_name=name)
            collection_states[name] = state.value
    except BackendError as exc:
        # State DB is unreachable: convert to ReadinessError for problem handler.
        readiness_err = ReadinessError(f"State DB unreachable: {exc.message}")
        from beacon.problems import error_to_problem
        problem = error_to_problem(readiness_err, instance="/readyz")
        body = problem_to_dict(problem)
        return JSONResponse(
            content=body,
            status_code=503,
            headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
        )

    # Ping Qdrant to verify it is reachable.
    store: QdrantStore = request.app.state.qdrant_store
    try:
        store.list_collections()
    except BackendError as exc:
        readiness_err = ReadinessError(f"Qdrant unreachable: {exc.message}")
        from beacon.problems import error_to_problem
        problem = error_to_problem(readiness_err, instance="/readyz")
        body = problem_to_dict(problem)
        return JSONResponse(
            content=body,
            status_code=503,
            headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
        )

    return JSONResponse(
        content={
            "status": "ok",
            "collections": collection_states,
            "qdrant": {"mode": store.mode, "reachable": True},
        },
        status_code=200,
    )
