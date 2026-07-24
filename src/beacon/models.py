"""Pydantic API schemas shared across the Beacon server.

Domain-specific route schemas for collection create/list/detail are defined
here as introduced by Task 01.5.

Evidence and EvidenceBundle schemas (Task 03.2) are the canonical evidence
input for citation validation (Task 03.3) and the POST /search response shape
(Task 03.4).
"""

from __future__ import annotations

import re
from enum import StrEnum

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

    config_json: str | None = None
    """JSON-serialized connector config for connector-definition sources.

    Populated for rows created by POST /collections/{name}/sources; None for
    content sources discovered by the sync engine and for uploads."""


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


# ---------------------------------------------------------------------------
# Evidence schemas (Task 03.2)
# ---------------------------------------------------------------------------


class EvidenceRole(StrEnum):
    """Role of an evidence item in the bundle."""

    HIT = "hit"
    """Primary retrieval hit - ranked and scored by the retrieval pipeline."""

    CONTEXT = "context"
    """Context span added by neighbor expansion - no relevance score."""


class Snippet(BaseModel):
    """Match-centered text excerpt with provenance from the source payload.

    Attributes:
        text:         Extracted text excerpt, centered on the query match.
        source_uri:   Canonical source URI (never an internal hash).
        title:        Human-readable document title.
        heading_path: Ordered list of heading components from the payload.
        locator:      Structural locator (heading path or page number string).
        chunk_id:     Chunk identifier for traceability.
        char_start:   0-based start offset of ``text`` within the full chunk text.
        char_end:     0-based exclusive end offset of ``text`` within the full chunk text.
    """

    text: str
    source_uri: str
    title: str
    heading_path: list[str]
    locator: str
    chunk_id: str
    char_start: int
    char_end: int


class Evidence(BaseModel):
    """One evidence item in an EvidenceBundle.

    Primary hits carry a fused relevance score from the retrieval pipeline.
    Context spans (neighbor-expanded chunks) carry no score (``score=None``)
    and reference the primary hit they were expanded from via ``context_of``.

    Canonical identity
    ------------------
    ``chunk_id`` is always the **hex chunk id**: the 64-character SHA-256 hex
    string stored in the payload field ``chunk_hash``.  It is NOT a Qdrant
    point UUID.  This format is shared by payload navigation fields
    ``prev_chunk_id`` / ``next_chunk_id``, so the dedup set and the neighbor
    chain operate in the same key space and intersect correctly.

    Attributes:
        chunk_id:   Hex chunk id (64-char SHA-256 from payload ``chunk_hash``).
                    Never a Qdrant point UUID.
        label:      Stable, gap-free citation label (S1, S2, ...).
        role:       ``hit`` for primary retrieval results; ``context`` for
                    neighbor-expanded spans.
        score:      Fused retrieval score for HIT items; ``None`` for CONTEXT.
        context_of: hex chunk_id of the primary HIT this span was expanded from;
                    ``None`` for primary HITs.
        snippet:    Match-centered text excerpt with provenance.
    """

    chunk_id: str
    label: str
    role: EvidenceRole
    score: float | None = None
    context_of: str | None = None
    snippet: Snippet | None = None


class BudgetRecap(BaseModel):
    """Token budget accounting for one evidence assembly run.

    Attributes:
        requested:    Number of primary hits provided to the assembler.
        packed:       Number of primary hits that fit within the budget.
        skipped:      Number of primary hits excluded due to budget overflow.
        tokens_packed: Total heuristic token count of all packed evidence (primary + context).
        token_budget: The original token budget.
    """

    requested: int
    packed: int
    skipped: int
    tokens_packed: int
    token_budget: int


class EvidenceBundle(BaseModel):
    """The complete, budget-bounded evidence set for one search result.

    Primary hits are packed before context spans.
    Labels are stable, gap-free S1..Sn assigned only after budget packing.
    This is the canonical input for citation validation (Task 03.3) and
    the POST /search response shape (Task 03.4).

    Attributes:
        evidence: Ordered list of evidence items (primary HITs then CONTEXT spans).
        recap:    Token budget accounting for this assembly run.
    """

    evidence: list[Evidence] = Field(default_factory=list)
    recap: BudgetRecap
