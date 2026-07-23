"""Documents and sources REST resources for the Beacon server (Task 02.1).

Routes:
    POST /documents                     - Multipart upload into the raw-document store,
                                          deduped by content hash, registered as an
                                          upload:// source.
    POST /collections/{name}/sources    - Attach a connector-backed source definition
                                          (folder root, future URL) to a collection.

Design invariants:
- Uploads are content-addressed: SHA-256 of the raw bytes is computed in the
  request handler; re-uploading identical content returns the existing source
  without writing new bytes.
- Uploads over ``ingest.max_upload_bytes`` are rejected with 413 problem+json
  without reading/buffering the remainder of the body.
- Uploaded bytes are stored under ``<data_dir>/uploads/<prefix2>/<hash>/content``
  using a two-level hash directory (prefix2 = first 2 hex chars).
- The ``__uploads__`` sentinel collection stores all upload sources.
  It is created on first use by calling ``CollectionRepo.create()`` directly,
  bypassing the normal name-validation rules.
- ``POST /collections/{name}/sources`` validates the collection exists (404 if not)
  and that the connector_kind is known (422 if not), then upserts a
  connector-definition source row: the config is stored as JSON in
  ``config_json`` and the canonical_uri is a stable hash-based identity
  (``{kind}://{sha256(config)[:16]}``).
- All error responses use Content-Type: application/problem+json.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import pathlib
import sqlite3

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

from beacon.errors import NotFoundError
from beacon.ingest.connectors import ConnectorKind
from beacon.models import DocumentUploadResponse, SourceAttachRequest, SourceResponse
from beacon.problems import error_to_problem, problem_to_dict
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SourceRepo

# Ensure markdown types are registered in this worker process too.
mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("text/markdown", ".markdown")

_PROBLEM_CONTENT_TYPE = "application/problem+json"
_UPLOADS_COLLECTION = "__uploads__"
_CHUNK_SIZE = 64 * 1024  # 64 KiB

router = APIRouter(tags=["documents"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _problem_response(body: dict[str, object], status: int) -> JSONResponse:
    """Wrap a problem-details dict in a JSONResponse with the correct content-type."""
    return JSONResponse(
        content=body,
        status_code=status,
        headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
    )


def _detect_media_type(filename: str | None, declared: str | None) -> str:
    """Detect media type from declared content-type or filename extension.

    Preference order:
    1. Declared content-type from the multipart part (if not generic).
    2. Extension-based detection via mimetypes.
    3. Fallback: ``application/octet-stream``.

    Args:
        filename: Original filename from the upload (may be None).
        declared: Content-Type from the multipart part (may be None).

    Returns:
        MIME type string.
    """
    if declared and declared not in ("application/octet-stream", "binary/octet-stream"):
        return declared.split(";")[0].strip()
    if filename:
        mime, _ = mimetypes.guess_type(filename)
        if mime:
            return mime
    return "application/octet-stream"


def _ensure_uploads_collection(db: StateDB) -> None:
    """Create the ``__uploads__`` sentinel collection if it does not exist.

    Bypasses name-validation (double-underscore is reserved for internal use)
    by calling ``CollectionRepo.create()`` directly.

    Args:
        db: Open StateDB instance.
    """
    repo = CollectionRepo(db)
    if repo.get(_UPLOADS_COLLECTION) is None:
        repo.create(name=_UPLOADS_COLLECTION)


def _store_bytes(data_dir: pathlib.Path, content_hash: str, raw: bytes) -> bool:
    """Write *raw* bytes to the content-addressed store.

    Directory layout: ``<data_dir>/uploads/<hash[:2]>/<hash>/content``.

    Args:
        data_dir:     Root data directory from settings.
        content_hash: Hex-encoded SHA-256 of *raw*.
        raw:          Raw bytes to persist.

    Returns:
        True if bytes were written, False if already stored.
    """
    prefix = content_hash[:2]
    dest_dir = data_dir / "uploads" / prefix / content_hash
    dest = dest_dir / "content"
    if dest.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return True


def _source_row_to_response(row: object) -> SourceResponse:
    """Convert a sqlite3.Row to a SourceResponse.

    Args:
        row: sqlite3.Row with sources table columns.

    Returns:
        SourceResponse instance.
    """
    r: sqlite3.Row = row  # type: ignore[assignment]
    return SourceResponse(
        id=int(r["id"]),
        collection_name=str(r["collection_name"]),
        canonical_uri=str(r["canonical_uri"]),
        connector_kind=str(r["connector_kind"]),
        content_hash=str(r["content_hash"]),
        status=str(r["status"]),
        created_at=str(r["created_at"]),
        media_type=str(r["media_type"]) if r["media_type"] is not None else None,
        config_json=str(r["config_json"]) if r["config_json"] is not None else None,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/documents", status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile,
) -> JSONResponse:
    """Store an uploaded file in the content-addressed raw-document store.

    Reads the multipart file in 64 KiB chunks. Rejects oversized uploads with
    413 before reading the remainder of the body. Deduplicates by SHA-256:
    re-uploading identical content returns 200 with ``stored=False``.

    Args:
        request: Incoming FastAPI request.
        file:    Uploaded file from the multipart form.

    Returns:
        201 JSON on first store, 200 JSON on dedupe, 413 problem+json on size error.
    """
    settings = request.app.state.settings
    db: StateDB = request.app.state.state_db

    max_bytes: int = settings.ingest.max_upload_bytes
    data_dir = pathlib.Path(settings.ingest.data_dir)

    # Stream-read in chunks; abort early on size overflow.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(size=_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            problem = {
                "type": "https://beacon.example/errors/ingestion",
                "title": "Ingestion Error",
                "status": 413,
                "detail": (
                    f"Upload exceeds the configured limit of {max_bytes} bytes."
                ),
                "kind": "ingestion",
            }
            return _problem_response(problem, 413)
        chunks.append(chunk)

    raw = b"".join(chunks)
    content_hash = hashlib.sha256(raw).hexdigest()
    canonical_uri = f"upload://{content_hash}"

    filename = file.filename or ""
    declared_ct = file.content_type or None
    media_type = _detect_media_type(filename, declared_ct)

    # Ensure the sentinel collection exists (cached per-process via app.state flag).
    if not getattr(request.app.state, "_uploads_collection_ensured", False):
        _ensure_uploads_collection(db)
        request.app.state._uploads_collection_ensured = True

    source_repo = SourceRepo(db)
    existing_row = source_repo.get(
        collection_name=_UPLOADS_COLLECTION,
        canonical_uri=canonical_uri,
    )

    if existing_row is not None:
        # Deduplicated: return the existing source without writing bytes.
        return JSONResponse(
            content=DocumentUploadResponse(
                source_id=int(existing_row["id"]),
                canonical_uri=canonical_uri,
                content_hash=content_hash,
                stored=False,
                media_type=(
                    str(existing_row["media_type"])
                    if existing_row["media_type"] is not None
                    else media_type
                ),
            ).model_dump(mode="json"),
            status_code=200,
        )

    # First upload: store bytes and register the source.
    _store_bytes(data_dir, content_hash, raw)
    source_repo.upsert(
        collection_name=_UPLOADS_COLLECTION,
        canonical_uri=canonical_uri,
        connector_kind=ConnectorKind.UPLOAD,
        content_hash=content_hash,
        media_type=media_type,
    )

    new_row = source_repo.get(
        collection_name=_UPLOADS_COLLECTION,
        canonical_uri=canonical_uri,
    )
    if new_row is None:
        from beacon.errors import BackendError
        raise BackendError("Failed to retrieve newly created upload source")

    return JSONResponse(
        content=DocumentUploadResponse(
            source_id=int(new_row["id"]),
            canonical_uri=canonical_uri,
            content_hash=content_hash,
            stored=True,
            media_type=media_type,
        ).model_dump(mode="json"),
        status_code=201,
    )


@router.post("/collections/{name}/sources", status_code=201)
async def attach_source(
    request: Request,
    name: str,
    body: SourceAttachRequest,
) -> JSONResponse:
    """Attach a connector-backed source definition to a collection.

    Validates the collection exists and the connector_kind is known. Stores
    the connector config as ``config_json`` on a definition row (flagged with
    ``is_connector_definition``) in the sources table; the sync trigger route
    reads this row to decide which connector to run.

    The canonical_uri is a stable opaque identity derived from a hash of the
    config (``{kind}://{sha256(config)[:16]}``), NOT the JSON blob itself, so
    the URI stays short and stable while the full config lives in config_json.
    Legacy rows that encoded the JSON blob in the canonical_uri are still
    parsed by the sync trigger route for backward compatibility.

    Args:
        request: Incoming FastAPI request.
        name:    Collection name from the URL path.
        body:    Validated request body with connector_kind and config.

    Returns:
        201 JSON SourceResponse on success.
        404 problem+json if the collection does not exist.
        422 problem+json if connector_kind is unknown (caught by Pydantic).
    """
    db: StateDB = request.app.state.state_db
    col_repo = CollectionRepo(db)

    row = col_repo.get(name)
    if row is None:
        err = NotFoundError(f"Collection {name!r} does not exist.")
        problem = error_to_problem(err, instance=f"/collections/{name}/sources")
        return _problem_response(problem_to_dict(problem), 404)

    # Deterministic JSON config (sorted keys) + stable hash-based identity.
    config_json = json.dumps(body.config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]
    canonical_uri = f"{body.connector_kind}://{config_hash}"

    source_repo = SourceRepo(db)
    source_repo.upsert(
        collection_name=name,
        canonical_uri=canonical_uri,
        connector_kind=body.connector_kind,
        content_hash="",
        config_json=config_json,
        is_connector_definition=True,
    )

    source_row = source_repo.get(
        collection_name=name,
        canonical_uri=canonical_uri,
    )
    if source_row is None:
        from beacon.errors import BackendError
        raise BackendError(f"Failed to retrieve newly attached source for {name!r}")

    return JSONResponse(
        content=_source_row_to_response(source_row).model_dump(mode="json"),
        status_code=201,
    )
