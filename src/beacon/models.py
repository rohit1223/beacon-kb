"""Pydantic API schemas shared across the Beacon server.

Domain-specific route schemas for collection create/list/detail are defined
here as introduced by Task 01.5.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

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
    if "__" in name:
        raise ValueError(
            "Collection name must not contain double underscores (__); "
            "that prefix is reserved for internal shadow collections."
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


# ---------------------------------------------------------------------------
# Ingestion request/response models (Task 02.1)
# ---------------------------------------------------------------------------


class SourceAttachRequest(BaseModel):
    """Request body for POST /collections/{name}/sources."""

    connector_kind: str
    """Connector type. Must be one of the registered kinds (folder, upload, web)."""

    config: dict[str, str] = Field(default_factory=dict)
    """Connector-specific configuration key-value pairs.

    For ``folder`` connectors: ``root`` (required), ``include_globs`` (optional,
    comma-separated glob patterns reconstructed by Task 02.5 as a list),
    ``exclude_globs`` (optional, comma-separated).
    List-valued fields use comma-separation as the wire format; Task 02.5
    will split on commas to reconstruct the list for FolderConnector.
    For ``upload`` connectors: no required keys; the URI is resolved from the
    content hash set by the upload route.
    """

    @field_validator("connector_kind")
    @classmethod
    def validate_connector_kind(cls, v: str) -> str:
        """Reject unknown connector kinds with a typed validation error."""
        from beacon.ingest.connectors import get_connector_kinds

        allowed = get_connector_kinds()
        if v not in allowed:
            raise ValueError(
                f"Unknown connector kind {v!r}. "
                f"Allowed values: {sorted(allowed)!r}."
            )
        return v


class SourceResponse(BaseModel):
    """Response body for a single source record."""

    id: int
    """Row ID in the state DB sources table."""

    collection_name: str
    """Owning logical collection name."""

    canonical_uri: str
    """Stable canonical URI identifying this source (file://, upload://, https://)."""

    connector_kind: str
    """Connector type that produced this source."""

    content_hash: str
    """Hex-encoded SHA-256 of the last fetched content. Empty string before first fetch."""

    status: str
    """Source lifecycle status: active | retired."""

    created_at: str
    """UTC ISO 8601 timestamp when the source was first registered."""

    media_type: str | None = None
    """MIME type for upload sources (e.g. 'text/markdown'). None for connector-backed sources."""


class DocumentUploadResponse(BaseModel):
    """Response body for POST /documents."""

    source_id: int
    """Row ID of the source record in the state DB."""

    canonical_uri: str
    """Content-addressed canonical URI for this upload (``upload://<sha256>``)."""

    content_hash: str
    """Hex-encoded SHA-256 of the uploaded bytes."""

    stored: bool
    """True if the bytes were written to disk on this request; False if this
    content hash was already stored (deduplicated)."""

    media_type: str
    """Detected media type for the uploaded file."""
