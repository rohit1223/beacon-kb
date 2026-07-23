"""Sync routes must return RFC 9457 problem+json for 404/409/422 errors.

Regression tests for the branch-review finding that sync routes raised bare
``HTTPException`` (FastAPI default JSON shape, ``application/json``) instead of
going through the uniform problem-details channel used everywhere else.
"""
from __future__ import annotations

from typing import Any

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.server.app import create_app
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SyncJobRepo


def _settings(tmp_path: Any) -> BeaconSettings:
    return BeaconSettings(
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )


def test_sync_404_returns_problem_json(tmp_path: Any) -> None:
    """POST /sync on an unknown collection returns 404 problem+json."""
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        r = c.post("/collections/no-such-collection/sync")
    assert r.status_code == 404
    assert "problem+json" in r.headers.get("content-type", ""), (
        f"Expected problem+json content-type, got {r.headers.get('content-type')}"
    )
    body = r.json()
    assert "type" in body
    assert "title" in body
    assert body.get("status") == 404
    assert "detail" in body
    assert "instance" in body


def test_job_404_returns_problem_json(tmp_path: Any) -> None:
    """GET /jobs/{id} on an unknown job returns 404 problem+json."""
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        r = c.get("/jobs/no-such-job")
    assert r.status_code == 404
    assert "problem+json" in r.headers.get("content-type", "")
    body = r.json()
    assert "type" in body
    assert "title" in body
    assert "detail" in body


def test_sync_409_returns_problem_json_with_extensions(tmp_path: Any) -> None:
    """POST /sync during an active job returns 409 problem+json.

    Top-level RFC 9457 fields plus job_id/state extension members.
    """
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        # Inject the collection and a PENDING job after startup so the
        # stale-job startup sweep does not reap it before the request.
        db = StateDB(db_path=str(tmp_path / "beacon.db"))
        try:
            CollectionRepo(db).create(
                name="col-conflict",
                settings={
                    "connector_kind": "folder",
                    "connector_config": {"root": str(tmp_path)},
                },
            )
            SyncJobRepo(db).create(
                job_id="pending-job", collection_name="col-conflict"
            )
        finally:
            db.close()

        r = c.post("/collections/col-conflict/sync")
    assert r.status_code == 409
    assert "problem+json" in r.headers.get("content-type", "")
    body = r.json()
    assert "type" in body, f"Missing 'type' in {body}"
    assert "title" in body, f"Missing 'title' in {body}"
    assert body.get("status") == 409
    assert isinstance(body.get("detail"), str), f"detail must be a string: {body}"
    # Extension members lifted to the top level (no hand-rolled nesting).
    assert body.get("job_id") == "pending-job"
    assert body.get("state") == "pending"


def test_sync_422_no_connector_returns_problem_json(tmp_path: Any) -> None:
    """POST /sync on a collection without a connector returns 422 problem+json."""
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    db = StateDB(db_path=str(tmp_path / "beacon.db"))
    try:
        CollectionRepo(db).create(name="col-noconnector")
    finally:
        db.close()

    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        r = c.post("/collections/col-noconnector/sync")
    assert r.status_code == 422
    assert "problem+json" in r.headers.get("content-type", "")
    body = r.json()
    assert "type" in body
    assert "title" in body
    assert body.get("status") == 422
    assert "detail" in body
