"""App factory for the Beacon FastAPI server (Task 01.4).

``create_app(settings)`` is the single entry point for creating an isolated
FastAPI application instance.  Nothing is created at module import time: all
per-instance state (StateDB, settings) is constructed inside the factory and
stored on ``app.state`` via the lifespan context.

Isolation guarantee: two calls to ``create_app`` with different settings
produce two independent applications that share no store or DB state.

Usage::

    from beacon.config import BeaconSettings
    from beacon.server.app import create_app

    settings = BeaconSettings()
    app = create_app(settings)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from beacon.config import BeaconSettings
from beacon.server.error_handlers import register_error_handlers
from beacon.server.routes.health import router as health_router
from beacon.server.telemetry import instrument_app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context that initialises and tears down per-instance resources.

    Opens the state DB on startup and closes it on shutdown so that tests
    using ``TestClient`` get clean teardown without leaking file handles.

    The settings object is stored on ``app.state.settings`` by ``create_app``
    before the lifespan runs, so the lifespan can read it directly.

    Args:
        app: The FastAPI application being started.

    Yields:
        Control while the server is running.
    """
    from beacon.state.db import StateDB

    settings: BeaconSettings = app.state.settings

    # Ensure the parent directory for the state DB exists.
    db_path = settings.state.db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    state_db = StateDB(db_path=db_path)
    app.state.state_db = state_db

    try:
        yield
    finally:
        state_db.close()


def create_app(settings: BeaconSettings | None = None) -> FastAPI:
    """Create an isolated FastAPI application with full Beacon middleware.

    This factory:
    1. Constructs a fresh FastAPI app with the lifespan context.
    2. Stores *settings* on ``app.state`` so routes can read it via request.
    3. Registers all exception handlers (problem-details for every error type).
    4. Installs the optional API-key middleware (when key is configured).
    5. Mounts all route routers.
    6. Hooks in OTel instrumentation (no-op when not configured).

    Args:
        settings: Configuration for this app instance.  If ``None``, a default
            ``BeaconSettings()`` is constructed (reads from environment /
            ``.env``).

    Returns:
        A configured FastAPI application.  The lifespan has not yet been
        entered; call ``app`` from ``TestClient`` or ``uvicorn`` to start it.
    """
    if settings is None:
        settings = BeaconSettings()

    app = FastAPI(
        title="Beacon",
        description="Self-hosted knowledge-base RAG server",
        version="0.1.0",
        lifespan=_lifespan,
        # Disable the default 422 validation-error handler shape: we register
        # our own handler that emits problem+json.
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store settings on app.state so the lifespan and routes can access them.
    app.state.settings = settings

    # Register exception handlers.
    register_error_handlers(app)

    # Install optional API-key middleware.
    if settings.server.api_key is not None:
        _install_auth_middleware(app, settings)

    # Mount route routers.
    app.include_router(health_router)

    # Wire up OTel instrumentation (no-op when OTel not configured).
    instrument_app(app, settings=settings)

    return app


def _install_auth_middleware(app: FastAPI, settings: BeaconSettings) -> None:
    """Add the ApiKeyMiddleware to *app* using the configured key.

    Args:
        app: The FastAPI application.
        settings: The server settings containing the API key.
    """
    from beacon.server.auth import ApiKeyMiddleware

    api_key = settings.server.api_key
    if api_key is None:
        return

    plain_key = api_key.get_secret_value()
    app.add_middleware(ApiKeyMiddleware, api_key=plain_key)
