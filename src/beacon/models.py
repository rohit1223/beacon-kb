"""Pydantic API schemas shared across the Beacon server.

Domain-specific route schemas for collection create/list/detail are defined
here as introduced by Task 01.5.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Collection name validation
# ---------------------------------------------------------------------------

# Conservative pattern: lowercase alphanumerics, dash, underscore, 1-64 chars.
_COLLECTION_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _validate_collection_name(name: str) -> str:
    """Validate a collection name against the allowed pattern.

    Args:
        name: The collection name to validate.

    Returns:
        The name unchanged if valid.

    Raises:
        ValueError: If the name does not match the allowed pattern.
    """
    if not name:
        raise ValueError(
            "Collection name must not be empty."
        )
    if not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            "Collection name must contain only lowercase alphanumerics, "
            "dashes, or underscores, start with an alphanumeric character, "
            "and be between 1 and 64 characters long."
        )
    return name


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CollectionCreateRequest(BaseModel):
    """Request body for POST /collections."""

    name: str
    """Logical collection name. Must match [a-z0-9][a-z0-9_-]{0,63}."""

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate the collection name against the allowed pattern."""
        return _validate_collection_name(v)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CollectionResponse(BaseModel):
    """Response body for a single collection (POST and GET /collections/{name})."""

    name: str
    """Logical collection name."""

    corpus_state: str
    """Derived corpus state: empty | building | ready | failed."""

    created_at: str
    """UTC ISO 8601 timestamp when the collection was registered."""

    live_revision: str | None = None
    """Revision ID of the current live revision, or null if none."""

    last_job: LastJobSummary | None = None
    """Summary of the most recent sync job, or null if no jobs exist."""


class LastJobSummary(BaseModel):
    """Summary of the most recent sync job for a collection."""

    job_id: str
    """Unique job identifier."""

    state: str
    """Job state: pending | running | succeeded | failed."""

    created_at: str
    """UTC ISO 8601 timestamp when the job was created."""

    finished_at: str | None = None
    """UTC ISO 8601 timestamp when the job finished, or null if not yet finished."""


class CollectionListResponse(BaseModel):
    """Response body for GET /collections."""

    items: list[CollectionResponse]
    """Ordered list of all registered collections."""


# Rebuild models to resolve forward references (LastJobSummary used in CollectionResponse).
CollectionResponse.model_rebuild()
