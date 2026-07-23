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
from starlette.exceptions import HTTPException as StarletteHTTPException

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

# Type URI prefix for HTTPException-derived problems (RFC 9457 'type').
_HTTP_ERROR_TYPE_PREFIX = "https://beacon.example/errors/http-"

# Reason-phrase titles for common HTTPException status codes.
_STATUS_TITLES: dict[int, str] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

# RFC 9457 reserved top-level member names; anything else in a dict detail is
# treated as an extension member and lifted to the top level of the body.
_RESERVED_PROBLEM_MEMBERS = frozenset({"type", "title", "status", "detail", "instance"})


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

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Map any HTTPException to an RFC 9457 problem-details response.

        Routes may raise ``fastapi.HTTPException`` with either a plain string
        detail or a dict detail.  A dict detail contributes its ``detail`` key
        as the problem detail string; every other non-reserved key is lifted
        to the top level as an RFC 9457 extension member (e.g. ``job_id`` and
        ``state`` on the sync 409 conflict).

        Args:
            request: The incoming request.
            exc: The raised HTTPException (FastAPI's subclasses Starlette's).

        Returns:
            A problem-details JSONResponse with the exception's status code.
        """
        status = exc.status_code
        title = _STATUS_TITLES.get(status, f"HTTP {status}")

        extensions: dict[str, object] = {}
        if isinstance(exc.detail, dict):
            detail_str = str(exc.detail.get("detail", ""))
            extensions = {
                str(k): v
                for k, v in exc.detail.items()
                if str(k) not in _RESERVED_PROBLEM_MEMBERS
            }
        else:
            detail_str = str(exc.detail)

        body: dict[str, object] = {
            "type": f"{_HTTP_ERROR_TYPE_PREFIX}{status}",
            "title": title,
            "status": status,
            "detail": detail_str,
            "instance": str(request.url.path),
        }
        body.update(extensions)

        response = _problem_response(body, status)
        # Preserve any headers the exception carries (e.g. WWW-Authenticate,
        # Retry-After); Content-Type stays problem+json.
        if exc.headers:
            for header_name, header_value in exc.headers.items():
                if header_name.lower() != "content-type":
                    response.headers[header_name] = header_value
        return response

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
