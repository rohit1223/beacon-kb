"""OpenTelemetry hooks for the Beacon server (Task 01.4).

Provides tracer acquisition and span helpers for pipeline stages.
All imports of the OTel SDK are guarded: when the SDK is not installed or
no exporter is configured the module falls back to the OTel no-op API,
producing zero overhead and zero warnings.

Design rules:
- ``get_tracer()`` always returns a usable tracer (no-op or real).
- ``pipeline_span()`` is a context manager that wraps a pipeline stage.
- ``instrument_app()`` installs FastAPI/ASGI middleware when an OTel SDK
  exporter is configured; it is a no-op otherwise.
- No OTel exporter package is required at import time.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from beacon.config import BeaconSettings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _NoOpSpan:
    """A no-op span that does nothing but satisfies the OTel Span interface."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ANN401
        """No-op set_attribute."""

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """No-op set_attributes."""

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None, timestamp: int | None = None
    ) -> None:
        """No-op add_event."""

    def add_link(self, context: Any, attributes: dict[str, Any] | None = None) -> None:  # noqa: ANN401
        """No-op add_link."""

    def set_status(self, status: Any) -> None:  # noqa: ANN401
        """No-op set_status."""

    def update_name(self, name: str) -> None:
        """No-op update_name."""

    def __enter__(self) -> _NoOpSpan:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ANN401
        """Context manager exit (no-op)."""


class _NoOpTracer:
    """A no-op tracer that does nothing but satisfies the OTel Tracer interface."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:  # noqa: ANN401
        """Return a no-op span."""
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:  # noqa: ANN401
        """Return a no-op span as the current span."""
        return _NoOpSpan()

# Tracer name for all Beacon server spans.
_TRACER_NAME = "beacon.server"

# Flag indicating whether the OpenTelemetry SDK is available.
# When False, all instrumentation functions become no-ops.
_OTEL_AVAILABLE = False
_otel_trace: Any = None

try:
    import opentelemetry.trace as otel_trace_mod

    _otel_trace = otel_trace_mod
    _OTEL_AVAILABLE = True
except ImportError:
    # OpenTelemetry SDK not installed; fall back to no-op behavior.
    pass


def get_tracer(name: str = _TRACER_NAME) -> Any:  # noqa: ANN401
    """Return an OpenTelemetry tracer for the given *name*.

    Returns the global no-op tracer if no OTel SDK is configured.
    The returned object is always usable as a tracer (supports
    ``start_as_current_span`` and ``start_span``).

    Args:
        name: Instrumentation scope name (usually the module name).

    Returns:
        An ``opentelemetry.trace.Tracer`` instance (or no-op equivalent).
    """
    if _OTEL_AVAILABLE and _otel_trace is not None:
        return _otel_trace.get_tracer(name)
    # Fall back to global no-op tracer when SDK is not installed.
    try:
        import opentelemetry.trace

        return opentelemetry.trace.get_tracer(name)
    except ImportError:
        # Should not reach here, but provide a safe fallback.
        return _NoOpTracer()


@contextmanager
def pipeline_span(
    stage: str,
    **attributes: str,
) -> Generator[Any, None, None]:
    """Context manager that wraps a pipeline stage in an OTel span.

    The span is a child of the current active span (if any).  When no OTel
    exporter is configured the span is a no-op; there is zero overhead beyond
    the context-manager entry/exit.

    Args:
        stage: Human-readable stage name (e.g. ``"ingestion"``, ``"retrieval"``).
        **attributes: Additional span string attributes (e.g. ``collection="docs"``).

    Yields:
        The active span (real or no-op).
    """
    tracer = get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(f"beacon.{stage}") as span:
        for key, value in attributes.items():
            with contextlib.suppress(Exception):
                span.set_attribute(f"beacon.{key}", value)
        yield span


def instrument_app(app: FastAPI, *, settings: BeaconSettings | None = None) -> None:
    """Install OTel FastAPI instrumentation if the SDK exporter is configured.

    This function is intentionally a no-op when the
    ``opentelemetry-instrumentation-fastapi`` package is absent or when the
    OpenTelemetry SDK is not installed.  Operators who want traces must install
    the required packages and configure an exporter via the standard OTel
    environment variables (``OTEL_EXPORTER_OTLP_ENDPOINT`` etc.).

    Args:
        app: The FastAPI application instance.
        settings: Optional BeaconSettings (reserved for future use).
    """
    if not _OTEL_AVAILABLE or _otel_trace is None:
        # OpenTelemetry SDK not available; instrumentation is a no-op.
        return

    try:
        # Guard imports: these packages are optional.
        import importlib

        fastapi_instrumentor_mod = importlib.import_module(
            "opentelemetry.instrumentation.fastapi"
        )
        otel_trace_mod = importlib.import_module("opentelemetry.sdk.trace")

        fastapi_instrumentor_cls = fastapi_instrumentor_mod.FastAPIInstrumentor
        tracer_provider_cls = otel_trace_mod.TracerProvider
        provider = _otel_trace.get_tracer_provider()

        # Only instrument if the provider is a real SDK provider (not the no-op
        # global default).  This avoids double-instrumentation in tests.
        if isinstance(provider, tracer_provider_cls):
            fastapi_instrumentor_cls.instrument_app(app)
    except (ImportError, ModuleNotFoundError):
        # OTel SDK or FastAPI instrumentor not installed - silently skip.
        pass
    except Exception as exc:
        # Log but never raise: telemetry must never break the server.
        logger.debug("OTel instrumentation skipped: %s", exc)
