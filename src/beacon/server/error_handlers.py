"""FastAPI exception handlers that translate errors to RFC 9457 problem-details.

All error responses use ``Content-Type: application/problem+json``.

This module registers three handler categories:
1. ``BeaconError`` subclasses: mapped via ``problems.error_to_problem()``.
2. FastAPI/Pydantic ``RequestValidationError``: remapped to our problem shape.
3. Catch-all ``Exception``: generic 500 problem with no internal details.

The ``problems`` module remains FastAPI-free; this module owns the coupling
between the error taxonomy and FastAPI's ``exception_handler`` decorator.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from beacon.errors import BeaconError
from beacon.problems import KIND_TO_STATUS, error_to_problem, problem_to_dict

logger = logging.getLogger(__name__)

_PROBLEM_CONTENT_TYPE = "application/problem+json"

# Type URI and title for the generic internal-error problem.
_INTERNAL_ERROR_TYPE = "https://beacon.example/errors/internal-error"
_INTERNAL_ERROR_TITLE = "Internal Server Error"

# Type URI for validation errors.
_VALIDATION_ERROR_TYPE = "https://beacon.example/errors/validation-error"
_VALIDATION_ERROR_TITLE = "Request Validation Error"


def _problem_response(body: dict[str, object], status: int) -> JSONResponse:
    """Wrap a problem-details dict in a JSONResponse with the correct content-type.

    Args:
        body: Serialized problem-details dict.
        status: HTTP status code.

    Returns:
        JSONResponse with Content-Type: application/problem+json.
    """
    return JSONResponse(
        content=body,
        status_code=status,
        headers={"Content-Type": _PROBLEM_CONTENT_TYPE},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all Beacon exception handlers on *app*.

    Call this once from ``create_app`` after the app is constructed.

    Args:
        app: The FastAPI application to register handlers on.
    """

    @app.exception_handler(BeaconError)
    async def handle_beacon_error(
        request: Request, exc: BeaconError
    ) -> JSONResponse:
        """Map any BeaconError subclass to the correct problem-details status.

        Args:
            request: The incoming request.
            exc: The raised BeaconError.

        Returns:
            A problem-details JSONResponse.
        """
        problem = error_to_problem(exc, instance=str(request.url.path))
        status = KIND_TO_STATUS[exc.kind]
        body = problem_to_dict(problem)
        logger.debug("BeaconError handled: kind=%s status=%d", exc.kind, status)
        return _problem_response(body, status)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Remap FastAPI request-validation errors to problem-details shape.

        FastAPI's default validation error format uses a ``detail`` list of
        error objects.  This handler replaces that with a single problem-details
        response that is consistent with all other Beacon error responses.

        Args:
            request: The incoming request.
            exc: The Pydantic/FastAPI validation error.

        Returns:
            A 422 problem-details JSONResponse.
        """
        # Summarize individual errors into a single human-readable string.
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = " -> ".join(str(part) for part in first.get("loc", []))
            msg = first.get("msg", "validation failed")
            detail = f"{loc}: {msg}" if loc else msg
        else:
            detail = "Request validation failed"

        body: dict[str, object] = {
            "type": _VALIDATION_ERROR_TYPE,
            "title": _VALIDATION_ERROR_TITLE,
            "status": 422,
            "detail": detail,
            "instance": str(request.url.path),
        }
        return _problem_response(body, 422)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler: generic 500 with no internal details in the body.

        The full exception is logged server-side (for operators) but is never
        included in the response body (to avoid leaking internals to callers).

        Args:
            request: The incoming request.
            exc: The unexpected exception.

        Returns:
            A 500 problem-details JSONResponse.
        """
        logger.exception(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        body: dict[str, object] = {
            "type": _INTERNAL_ERROR_TYPE,
            "title": _INTERNAL_ERROR_TITLE,
            "status": 500,
            "detail": "An unexpected error occurred. Please try again later.",
            "instance": str(request.url.path),
        }
        return _problem_response(body, 500)
