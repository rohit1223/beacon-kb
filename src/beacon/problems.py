"""RFC 9457 problem-details helpers for Beacon.

This module is transport-neutral: it defines the frozen problem model and a
pure mapping from error taxonomy to HTTP status, with no dependency on FastAPI
or any HTTP framework.
Transport layers (FastAPI exception handlers, MCP error converters) import
``error_to_problem`` and wrap the result in their own response types.

References:
    https://www.rfc-editor.org/rfc/rfc9457
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from beacon.errors import BeaconError, BeaconErrorKind

# ---------------------------------------------------------------------------
# Error-to-status mapping
#
# One explicit table: every new kind must be added here or mypy --strict
# will catch the missing coverage in exhaustive checks.
# ---------------------------------------------------------------------------

KIND_TO_STATUS: dict[BeaconErrorKind, int] = {
    BeaconErrorKind.READINESS: 503,
    BeaconErrorKind.BACKEND: 502,
    BeaconErrorKind.INGESTION: 422,
    BeaconErrorKind.CITATION: 422,
    BeaconErrorKind.BUDGET: 402,
    BeaconErrorKind.CONFLICT: 409,
    BeaconErrorKind.NOT_FOUND: 404,
}

# Default titles for each kind, shown in the ``title`` field.
_KIND_TITLES: dict[BeaconErrorKind, str] = {
    BeaconErrorKind.READINESS: "Service Not Ready",
    BeaconErrorKind.BACKEND: "Backend Error",
    BeaconErrorKind.INGESTION: "Ingestion Error",
    BeaconErrorKind.CITATION: "Citation Error",
    BeaconErrorKind.BUDGET: "Budget Exceeded",
    BeaconErrorKind.CONFLICT: "Conflict",
    BeaconErrorKind.NOT_FOUND: "Not Found",
}

# Base URI for the ``type`` field.
_TYPE_BASE = "https://beacon.example/errors"


class ProblemDetail(BaseModel):
    """RFC 9457 problem-details representation.

    ``kind`` is a Beacon extension field - it carries the machine-readable
    error kind so callers can dispatch without parsing ``type`` URIs.

    The model is frozen so instances are safely hashable and cannot be mutated
    after construction.
    """

    model_config = ConfigDict(frozen=True)

    type: str
    """A URI that identifies the problem type (RFC 9457 Section 3.1.1)."""

    title: str
    """Short human-readable summary of the problem type (RFC 9457 Section 3.1.2)."""

    status: int
    """HTTP status code (RFC 9457 Section 3.1.3)."""

    detail: str
    """Human-readable explanation of this specific occurrence (RFC 9457 Section 3.1.4)."""

    kind: BeaconErrorKind
    """Beacon extension: machine-readable error kind for programmatic dispatch."""

    instance: str | None = None
    """Optional URI reference identifying the specific occurrence (RFC 9457 Section 3.1.5)."""


def error_to_problem(
    error: BeaconError,
    *,
    instance: str | None = None,
) -> ProblemDetail:
    """Convert a ``BeaconError`` to an RFC 9457 ``ProblemDetail``.

    Args:
        error: The typed Beacon error to convert.
        instance: Optional URI identifying the specific request or resource
            that triggered the error (e.g. ``/collections/docs/sync``).

    Returns:
        A frozen ``ProblemDetail`` ready for serialization as
        ``application/problem+json``.
    """
    kind = error.kind
    status = KIND_TO_STATUS[kind]
    title = _KIND_TITLES[kind]
    problem_type = f"{_TYPE_BASE}/{kind.value}"

    return ProblemDetail(
        type=problem_type,
        title=title,
        status=status,
        detail=error.message,
        kind=kind,
        instance=instance,
    )


def problem_to_dict(problem: ProblemDetail) -> dict[str, Any]:
    """Serialize a ``ProblemDetail`` to a plain dict for logging.

    Uses ``model_dump`` with ``mode="json"`` so enum values are serialized
    as their string values, not the enum objects.
    """
    return problem.model_dump(mode="json")
