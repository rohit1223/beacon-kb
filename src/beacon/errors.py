"""Typed error taxonomy for the Beacon server.

Each error family carries a stable machine-readable ``kind`` value so transport
layers (FastAPI exception handlers, MCP tool-error mappers) can dispatch on a
single field without inspecting error messages or class names.

Error kinds are transport-neutral - this module has no dependency on FastAPI,
httpx, or any network library so that Epic 04 can reuse the same taxonomy for
MCP tool errors.
"""

from __future__ import annotations

from enum import StrEnum


class BeaconErrorKind(StrEnum):
    """Stable machine-readable kind values for every error family."""

    READINESS = "readiness"
    BACKEND = "backend"
    INGESTION = "ingestion"
    CITATION = "citation"
    BUDGET = "budget"


class BeaconError(Exception):
    """Base class for all Beacon typed errors.

    Subclasses must set ``kind`` to one of the ``BeaconErrorKind`` values.
    """

    kind: BeaconErrorKind

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind.value!r}, message={self.message!r})"


class ReadinessError(BeaconError):
    """Raised when a required backend or dependency is not ready.

    Covers: Qdrant unreachable at startup, required model not loaded,
    embedding endpoint unavailable.

    Maps to HTTP 503 Service Unavailable.
    """

    kind = BeaconErrorKind.READINESS


class BackendError(BeaconError):
    """Raised when an external backend returns an unexpected error at runtime.

    Covers: Qdrant query failures, LiteLLM provider errors (network, rate-limit),
    database I/O errors.

    Maps to HTTP 502 Bad Gateway.
    """

    kind = BeaconErrorKind.BACKEND


class IngestionError(BeaconError):
    """Raised when a document cannot be parsed, chunked, or embedded.

    Always carries the ``source_uri`` of the offending document so callers can
    associate the error with the right source record.

    Maps to HTTP 422 Unprocessable Entity.
    """

    kind = BeaconErrorKind.INGESTION

    def __init__(self, message: str, *, source_uri: str = "") -> None:
        super().__init__(message)
        self.source_uri = source_uri

    def __repr__(self) -> str:
        return (
            f"IngestionError(kind={self.kind.value!r}, "
            f"source_uri={self.source_uri!r}, "
            f"message={self.message!r})"
        )


class CitationError(BeaconError):
    """Raised when an answer contains structurally invalid citations.

    Covers: citation labels that reference non-existent evidence slots, gap in
    the [S1][S2]... sequence, or an evidence item missing a required field.

    Maps to HTTP 422 Unprocessable Entity.
    """

    kind = BeaconErrorKind.CITATION


class BudgetError(BeaconError):
    """Raised when a cost or token budget is exceeded.

    The ``investigate`` pipeline raises this when the LangGraph loop hits the
    configured budget ceiling; the ``answer`` pipeline raises it when a single
    LLM call would exceed the remaining token budget.

    Maps to HTTP 402 Payment Required (closest semantic match for a budget gate).
    """

    kind = BeaconErrorKind.BUDGET


class AnswerError(BackendError):
    """Raised when the LLM call in the answer pipeline fails.

    Subclasses ``BackendError`` because the cause is always an external LLM
    provider failure; its ``kind`` is therefore ``backend``.
    """
