"""Sync trigger and job status routes."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Callable

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.connectors.base import Connector, ConnectorKind
from beacon.ingest.connectors.folder import FolderConnector
from beacon.ingest.connectors.web import WebConnector
from beacon.ingest.embeddings import EmbedderProvider
from beacon.ingest.sync import SyncEngine
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SourceRepo, SyncJobRepo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sync"])

# Type alias for the optional transport factory hook used in tests.
# When set on ``app.state.web_transport_factory``, the factory is called with
# the connector_config dict and must return an ``httpx.BaseTransport``.  In
# production this attribute is absent (or None) and a real ``httpx.Client`` is
# used by WebConnector.
WebTransportFactory = Callable[[dict[str, object]], httpx.BaseTransport]


def _instantiate_connector(
    connector_kind: str,
    connector_config: dict[str, object],
    *,
    web_transport_factory: WebTransportFactory | None = None,
) -> Connector:
    """Instantiate a connector from kind and config dict.

    For ``web`` connectors, ``web_transport_factory`` is called (when provided)
    to supply the ``httpx.BaseTransport`` injected into ``WebConnector``.  This
    seam keeps tests fully offline without any production-code hacks.

    The comma-separation convention for list-valued config fields (e.g.
    ``start_urls``, ``include_globs``) mirrors the folder connector and is
    documented in the source-attachment API contract.

    Args:
        connector_kind:        Kind string identifying the connector type.
        connector_config:      Dict of connector-specific configuration.
        web_transport_factory: Optional factory called with ``connector_config``
                               to produce an ``httpx.BaseTransport`` for the
                               ``WebConnector``.  When ``None`` (production),
                               ``WebConnector`` uses the real network.

    Returns:
        A Connector instance.

    Raises:
        HTTPException: 422 if the connector kind is unsupported or required
                       fields are missing.
    """
    if connector_kind == ConnectorKind.FOLDER:
        root = str(connector_config.get("root", "."))
        include_globs_raw = connector_config.get("include_globs", "**/*")
        exclude_globs_raw = connector_config.get("exclude_globs", "")

        include_globs: list[str]
        if isinstance(include_globs_raw, str):
            include_globs = [g.strip() for g in include_globs_raw.split(",") if g.strip()]
        elif isinstance(include_globs_raw, list):
            include_globs = [str(g) for g in include_globs_raw]
        else:
            include_globs = []

        exclude_globs: list[str]
        if isinstance(exclude_globs_raw, str):
            exclude_globs = [g.strip() for g in exclude_globs_raw.split(",") if g.strip()]
        elif isinstance(exclude_globs_raw, list):
            exclude_globs = [str(g) for g in exclude_globs_raw]
        else:
            exclude_globs = []

        if not include_globs:
            include_globs = ["**/*"]

        return FolderConnector(
            root=root,
            include_globs=include_globs,
            exclude_globs=exclude_globs or None,
        )

    if connector_kind == ConnectorKind.WEB:
        start_urls_raw = connector_config.get("start_urls", "")
        sitemap_url_raw = connector_config.get("sitemap_url", None)

        start_urls: list[str]
        if isinstance(start_urls_raw, str):
            start_urls = [u.strip() for u in start_urls_raw.split(",") if u.strip()]
        elif isinstance(start_urls_raw, list):
            start_urls = [str(u) for u in start_urls_raw]
        else:
            start_urls = []

        sitemap_url: str | None = str(sitemap_url_raw) if sitemap_url_raw else None

        if not start_urls and not sitemap_url:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Web connector requires at least one of "
                    "'start_urls' or 'sitemap_url' in connector_config"
                ),
            )

        max_depth = int(str(connector_config.get("max_depth", WebConnector.DEFAULT_MAX_DEPTH)))
        max_pages = int(str(connector_config.get("max_pages", WebConnector.DEFAULT_MAX_PAGES)))
        user_agent = str(
            connector_config.get("user_agent", WebConnector.DEFAULT_USER_AGENT)
        )

        transport: httpx.BaseTransport
        if web_transport_factory is not None:
            transport = web_transport_factory(connector_config)
        else:
            # Production path: use a real synchronous transport.
            transport = httpx.HTTPTransport()

        return WebConnector(
            start_urls=start_urls or None,
            sitemap_url=sitemap_url,
            max_depth=max_depth,
            max_pages=max_pages,
            user_agent=user_agent,
            transport=transport,
        )

    raise HTTPException(status_code=422, detail=f"Unsupported connector kind: {connector_kind!r}")


async def _run_sync_background(
    *,
    collection_name: str,
    job_id: str,
    connector_kind: str,
    connector_config: dict[str, object],
    store: object,
    settings: object,
    web_transport_factory: WebTransportFactory | None = None,
) -> None:
    """Background task that runs the sync engine in a worker thread.

    The state DB connection is single-threaded by design, so the worker opens
    its own ``StateDB`` from the configured path instead of reusing the app's
    request-thread connection.  The Qdrant store is reused: embedded mode
    locks its storage path, so a second store on the same path cannot be
    opened while the app holds one.

    If anything fails before or inside the engine, the job is marked FAILED
    so ``GET /jobs/{id}`` never reports a silently lost job.

    Args:
        collection_name:       Logical collection name.
        job_id:                Pre-created PENDING job identifier.
        connector_kind:        Connector kind string.
        connector_config:      Connector-specific config dict.
        store:                 QdrantStore instance owned by the app.
        settings:              BeaconSettings instance.
        web_transport_factory: Optional factory for ``WebConnector`` transport
                               (tests inject a ``MockTransport`` here; production
                               passes ``None`` to use the real network).
    """
    from beacon.config import BeaconSettings
    from beacon.storage.qdrant import QdrantStore

    def _sync() -> None:
        assert isinstance(store, QdrantStore)
        assert isinstance(settings, BeaconSettings)

        db = StateDB(db_path=settings.state.db_path)
        try:
            try:
                connector = _instantiate_connector(
                    connector_kind,
                    connector_config,
                    web_transport_factory=web_transport_factory,
                )
                # WebConnector owns an httpx.Client; close it when the sync
                # finishes so connection pools are released.  Connectors
                # without a close() method (e.g. FolderConnector) need no
                # cleanup, hence the conditional context.
                closer: contextlib.AbstractContextManager[object] = (
                    contextlib.closing(connector)
                    if hasattr(connector, "close")
                    else contextlib.nullcontext()
                )
                with closer:
                    embedder = EmbedderProvider(
                        model_name=settings.models.embedding_model,
                        dimension=settings.models.embedding_dimension,
                    )
                    engine = SyncEngine(
                        store=store,
                        db=db,
                        embedder=embedder,
                        chunker_config=ChunkerConfig(),
                        settings=settings,
                    )
                    engine.run_sync(
                        collection_name=collection_name,
                        connector=connector,
                        job_id=job_id,
                    )
            except Exception as exc:
                # The engine marks the job FAILED for pipeline errors; this
                # covers failures before the engine takes over (connector or
                # embedder construction) and is idempotent otherwise.
                SyncJobRepo(db).set_failed(
                    job_id,
                    error_detail={"message": str(exc), "type": type(exc).__name__},
                )
                raise
        finally:
            db.close()

    try:
        await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.error(
            "Background sync failed for collection %r job %r: %s",
            collection_name,
            job_id,
            exc,
        )


def _parse_config_dict(raw_json: str) -> dict[str, object]:
    """Parse a JSON string into a str-keyed config dict, tolerating bad input.

    Args:
        raw_json: JSON text expected to contain an object.

    Returns:
        Parsed dict with string keys, or an empty dict when the input is not
        valid JSON or not an object.
    """
    try:
        raw = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items()}


def _resolve_connector_definition(
    *,
    db: StateDB,
    collection_name: str,
    collection_row: sqlite3.Row,
) -> tuple[str, dict[str, object]]:
    """Resolve the connector kind and config for a collection.

    Resolution order:

    1. Connector-definition source rows (``is_connector_definition = 1``)
       written by ``POST /collections/{name}/sources``.  The config comes from
       ``config_json``; rows created before migration 0005 that encoded the
       config JSON in the canonical_uri (``kind://{json}``) are still parsed
       for backward compatibility.
    2. Legacy fallback: ``connector_kind`` / ``connector_config`` keys inside
       the collection row's ``settings_json``.

    Args:
        db:              Open StateDB instance.
        collection_name: Logical collection name.
        collection_row:  The collection's DB row (for the legacy fallback).

    Returns:
        Tuple of (connector_kind, connector_config).

    Raises:
        HTTPException: 422 when no connector definition can be resolved.
    """
    definition_rows = SourceRepo(db).list_connector_definitions(
        collection_name=collection_name
    )
    if definition_rows:
        # Deterministic choice: lowest row id (first attached definition).
        defn = definition_rows[0]
        connector_kind = str(defn["connector_kind"])
        config_json_str = defn["config_json"]
        if config_json_str:
            return connector_kind, _parse_config_dict(str(config_json_str))
        # Legacy row: canonical_uri was "{kind}://{json_blob}".
        canonical_uri = str(defn["canonical_uri"])
        prefix = f"{connector_kind}://"
        if canonical_uri.startswith(prefix):
            return connector_kind, _parse_config_dict(canonical_uri[len(prefix):])
        return connector_kind, {}

    # Legacy fallback: settings_json on the collection row.
    settings_json_str = collection_row["settings_json"]
    try:
        collection_settings: dict[str, object] = json.loads(settings_json_str or "{}")
    except (json.JSONDecodeError, TypeError):
        collection_settings = {}

    connector_kind_raw = collection_settings.get("connector_kind")
    if not connector_kind_raw:
        raise HTTPException(
            status_code=422,
            detail="Collection has no connector configured",
        )
    connector_config_raw = collection_settings.get("connector_config", {})
    connector_config: dict[str, object] = (
        {str(k): v for k, v in connector_config_raw.items()}
        if isinstance(connector_config_raw, dict)
        else {}
    )
    return str(connector_kind_raw), connector_config


@router.post("/collections/{collection_name}/sync", status_code=202)
async def trigger_sync(
    collection_name: str,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict[str, str]:
    """Trigger an asynchronous sync for the collection.

    Reads the connector definition from the collection's connector-definition
    source rows (written by ``POST /collections/{name}/sources``), creates a
    PENDING sync job, and starts the sync engine in the background.  When no
    definition row exists, falls back to the legacy ``settings_json`` field on
    the collection row for backward compatibility.

    Rejects with 409 (Conflict) when a PENDING or RUNNING job already exists
    for the collection so callers never queue multiple concurrent syncs.

    Args:
        collection_name:   Logical collection name from the URL path.
        background_tasks:  FastAPI background task handler.
        request:           Incoming HTTP request (provides app state).

    Returns:
        Dict with ``job_id`` for polling.

    Raises:
        HTTPException: 404 if the collection is not registered.
        HTTPException: 409 if a PENDING or RUNNING sync is already in progress.
        HTTPException: 422 if the collection has no connector configured.
    """
    db = request.app.state.state_db
    store = request.app.state.qdrant_store
    settings = request.app.state.settings

    # Check collection exists.
    collection_row = CollectionRepo(db).get(collection_name)
    if collection_row is None:
        raise HTTPException(status_code=404, detail=f"Collection {collection_name!r} not found")

    # Reject concurrent syncs: only one PENDING or RUNNING job per collection.
    # The dict detail carries the human-readable message plus RFC 9457
    # extension members (job_id, state); the HTTPException handler in
    # error_handlers lifts them to top-level problem+json fields, so no
    # hand-rolled problem body is built here.
    active_job = SyncJobRepo(db).get_active(collection_name)
    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "detail": (
                    f"A sync job ({active_job['job_id']!r}) is already"
                    f" {active_job['state']} for collection {collection_name!r}."
                    f" Wait for it to finish before starting a new sync."
                ),
                "job_id": active_job["job_id"],
                "state": active_job["state"],
            },
        )

    connector_kind, connector_config = _resolve_connector_definition(
        db=db,
        collection_name=collection_name,
        collection_row=collection_row,
    )

    # Optional test seam: a factory on app.state supplies the httpx transport
    # for WebConnector so tests can run fully offline.
    web_transport_factory: WebTransportFactory | None = getattr(
        request.app.state, "web_transport_factory", None
    )

    # Create job record.
    job_id = uuid.uuid4().hex
    SyncJobRepo(db).create(job_id=job_id, collection_name=collection_name)

    # Start background task.
    background_tasks.add_task(
        _run_sync_background,
        collection_name=collection_name,
        job_id=job_id,
        connector_kind=connector_kind,
        connector_config=connector_config,
        store=store,
        settings=settings,
        web_transport_factory=web_transport_factory,
    )

    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, object]:
    """Get the status of a sync job.

    Args:
        job_id:  Job identifier.
        request: Incoming HTTP request (provides app state).

    Returns:
        Dict with job state and counters.

    Raises:
        HTTPException: 404 if the job is not found.
    """
    db = request.app.state.state_db
    row = SyncJobRepo(db).get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    return {
        "job_id": row["job_id"],
        "collection_name": row["collection_name"],
        "state": row["state"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "sources_added": row["sources_added"],
        "sources_removed": row["sources_removed"],
        "sources_unchanged": row["sources_unchanged"],
        "error_detail": json.loads(row["error_detail"]) if row["error_detail"] else None,
    }
