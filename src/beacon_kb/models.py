"""Frozen domain records, enums, content-addressed typed IDs, and AgenticTrace.

All records in this module are immutable (frozen=True, slots=True) and carry
typed IDs derived from stable content inputs so identical content reproduces
identical IDs across processes.

Importing this module performs no side effects - no I/O, no logging handlers,
no network calls, no credential loading.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from typing import NewType

# ---------------------------------------------------------------------------
# Typed IDs - NewType wrappers over str for type-safe boundaries
# ---------------------------------------------------------------------------

CorpusId = NewType("CorpusId", str)
"""Identity of a named corpus (a collection of sources)."""

SourceId = NewType("SourceId", str)
"""Identity of a single source within a corpus."""

RevisionId = NewType("RevisionId", str)
"""Identity of a specific revision of a source, tied to content hash and pipeline."""

SectionId = NewType("SectionId", str)
"""Identity of a parsed section within a revision."""

ChunkId = NewType("ChunkId", str)
"""Identity of a retrieval chunk, content-addressed from corpus/source/revision/pipeline."""

BuildRunId = NewType("BuildRunId", str)
"""Identity of a single indexing build run."""

EvidenceId = NewType("EvidenceId", str)
"""Identity of an evidence item within an answer response."""

QueryId = NewType("QueryId", str)
"""Identity of a query issued against a knowledge base."""

TraceId = NewType("TraceId", str)
"""Identity of an agentic trace session."""


# ---------------------------------------------------------------------------
# Content-addressed ID constructors
# Deterministic: identical inputs reproduce identical IDs across processes.
# Uses SHA-256 (first 32 hex chars) over a canonical colon-delimited string.
# ---------------------------------------------------------------------------


def _sha256_id(parts: list[str]) -> str:
    """Return the first 32 hex characters of SHA-256 over canonical parts string."""
    canonical = ":".join(p.replace(":", "_") for p in parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def make_source_id(*, corpus: str, canonical_uri: str) -> SourceId:
    """Return a deterministic SourceId from corpus name and canonical URI.

    Identical inputs always reproduce the same ID across processes.
    """
    return SourceId(_sha256_id(["source", corpus, canonical_uri]))


def make_revision_id(
    *,
    source_id: str,
    content_hash: str,
    pipeline_fingerprint: str,
) -> RevisionId:
    """Return a deterministic RevisionId.

    Inputs: stable source ID, content hash of the raw document bytes,
    and pipeline fingerprint capturing chunker/embedder versions.
    Identical inputs always reproduce the same ID across processes.
    """
    return RevisionId(_sha256_id(["revision", source_id, content_hash, pipeline_fingerprint]))


def make_chunk_id(
    *,
    corpus: str,
    canonical_uri: str,
    revision_id: str,
    pipeline_fingerprint: str,
    parent_locator: str,
    child_ordinal: int,
) -> ChunkId:
    """Return a deterministic ChunkId from all stable chunk-identity inputs.

    Inputs: corpus, canonical source URI, revision ID, pipeline fingerprint,
    parent section locator, and ordinal position within the parent.
    Identical inputs always reproduce the same ID across processes.
    Never derived from random values.
    """
    return ChunkId(
        _sha256_id(
            [
                "chunk",
                corpus,
                canonical_uri,
                revision_id,
                pipeline_fingerprint,
                parent_locator,
                str(child_ordinal),
            ]
        )
    )


def make_build_run_id(
    *,
    corpus: str,
    pipeline_fingerprint: str,
    started_at_iso: str,
) -> BuildRunId:
    """Return a deterministic BuildRunId from corpus, pipeline fingerprint, and start time.

    Identical inputs always reproduce the same ID across processes.

    Note: ``started_at_iso`` is an intentional run-identity input.
    Callers must record the exact ISO 8601 string they pass here so they can
    reproduce the same BuildRunId later (e.g. to look up or deduplicate a run).
    """
    return BuildRunId(_sha256_id(["build_run", corpus, pipeline_fingerprint, started_at_iso]))


def make_section_id(
    *,
    source_id: str,
    revision_id: str,
    locator: str,
) -> SectionId:
    """Return a deterministic SectionId from source_id, revision_id, and locator.

    Inputs:
        source_id: The stable SourceId string for the parent source.
        revision_id: The RevisionId string for the specific revision being parsed.
        locator: The stable section locator (e.g. 'intro', 'intro/background').

    Identical inputs always reproduce the same ID across processes.
    """
    return SectionId(_sha256_id(["section", source_id, revision_id, locator]))


def make_evidence_id(
    *,
    query_id: str,
    chunk_id: str,
) -> EvidenceId:
    """Return a deterministic EvidenceId from query_id and chunk_id.

    Inputs:
        query_id: The stable QueryId string for the query that produced this evidence.
        chunk_id: The stable ChunkId string for the retrieved chunk.

    Identical inputs always reproduce the same ID across processes.
    """
    return EvidenceId(_sha256_id(["evidence", query_id, chunk_id]))


# ---------------------------------------------------------------------------
# Closed-vocabulary enums
# ---------------------------------------------------------------------------


class SyncStatus(enum.Enum):
    """Outcome of a sync/build run."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class ChunkKind(enum.Enum):
    """Hierarchical role of a chunk in a parent/child chunking scheme."""

    PARENT = "parent"
    """Larger context unit used for scoring and ranking."""

    CHILD = "child"
    """Smaller retrieval unit used for embedding and similarity search."""


class EvidenceRole(enum.Enum):
    """Role of an evidence item relative to the answer.

    HIT - directly retrieved and cited in the answer.
    CONTEXT - surrounding context included but not directly cited.
    """

    HIT = "hit"
    CONTEXT = "context"


class IngestionChange(enum.Enum):
    """Classification of a source change detected during incremental sync."""

    UNCHANGED = "unchanged"
    """Content and pipeline fingerprint match the active revision; skip."""

    NEW = "new"
    """Source has no active revision; ingest fully."""

    CHANGED = "changed"
    """Content hash differs from active revision; re-ingest."""

    DELETED = "deleted"
    """Source no longer returned by the connector; retire active revision."""

    INCOMPATIBLE = "incompatible"
    """Pipeline fingerprint changed; full re-ingest required regardless of content."""


class CorpusHealth(enum.Enum):
    """Health state of a corpus, derived exclusively from durable store state.

    State machine (precedence in order when multiple conditions apply):

    EMPTY
        No active revision exists AND no build runs have ever been recorded.
        The corpus has never been synced.

    BUILDING
        At least one build run is currently in-progress (status='running').
        May coexist with an active revision if an incremental sync is running.

    READY
        At least one active revision exists.
        Takes precedence over FAILED: if an active revision is present but the
        latest build run failed, the corpus is still READY because the prior
        active revision remains fully searchable.

    FAILED
        The latest build run has status='failed' AND no active revision exists.
        The corpus has never had a successful sync, or a failed sync wiped the
        last active revision.

    Transitions are derived by ``SyncEngine.health()`` from the durable store;
    they are never inferred from in-memory counters.
    """

    EMPTY = "empty"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Domain records - frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Corpus:
    """A named collection of sources with a shared retrieval configuration.

    Identity is tracked by CorpusId. Records are immutable.
    """

    id: CorpusId
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Source:
    """A single addressable document source within a corpus.

    canonical_uri is the stable, normalized identifier for the source
    (e.g. file:/// for filesystem, https:// for web, confluence:// for Confluence).
    SourceId is content-addressed from corpus + canonical_uri.

    extra holds arbitrary string key-value metadata as a tuple of (key, value) pairs.
    Pairs must be sorted by key for deterministic identity (so two Source records
    built from the same logical metadata compare equal regardless of insertion order).
    Use ``tuple(sorted(d.items()))`` to convert a dict before constructing.
    """

    id: SourceId
    corpus_id: CorpusId
    canonical_uri: str
    media_type: str
    title: str = ""
    extra: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Revision:
    """A specific revision of a source, tied to content hash and pipeline fingerprint.

    RevisionId is content-addressed from source_id, content_hash, and
    pipeline_fingerprint. Two revisions with the same content and pipeline
    produce the same RevisionId and share chunk-level identity.
    """

    id: RevisionId
    source_id: SourceId
    content_hash: str
    """SHA-256 or similar hash of the raw document bytes."""

    pipeline_fingerprint: str
    """Hash of the pipeline configuration (chunker params, embedder version, etc.)."""

    byte_size: int = 0
    fetched_at_iso: str = ""


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Raw fetched content of a source before parsing.

    Records are immutable. content holds the document body as a string
    (decoded from bytes for text formats; base64 for binary).
    """

    source_id: SourceId
    revision_id: RevisionId
    """Connector-supplied revision identity.

    IMPORTANT: The revision_id produced by a connector (e.g. FilesystemConnector
    or MemoryConnector) is a *provisional* content identity.  It captures the
    content hash, but the pipeline_fingerprint baked into it uses the connector's
    default sentinel (``PROVISIONAL_FINGERPRINT = "unpinned"``), NOT the full
    pipeline fingerprint.

    The sync pipeline ALWAYS re-derives the authoritative revision_id using the
    real pipeline fingerprint before staging or promoting any revision.  Callers
    that receive a RawDocument from a connector must not assume its revision_id
    is the final authoritative ID that will appear in the store.
    """

    content: str
    media_type: str
    encoding: str = "utf-8"


@dataclass(frozen=True, slots=True)
class Section:
    """A parsed, heading-delimited section within a document.

    locator is a stable string identifying this section (e.g. heading path).
    SectionId is content-addressed at the ingestion layer.
    """

    id: SectionId
    source_id: SourceId
    revision_id: RevisionId
    locator: str
    """Stable section locator (e.g. 'intro', 'intro/background')."""

    heading: str
    text: str
    ordinal: int
    """Zero-based position of this section within the document."""

    parent_locator: str = ""
    depth: int = 0


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval unit derived from a section.

    ChunkId is content-addressed from corpus, canonical_uri, revision_id,
    pipeline_fingerprint, parent_locator, and child_ordinal. This ensures
    that identical content with identical pipeline configuration always
    produces the same ChunkId across processes, enabling deterministic
    deduplication and cache invalidation.
    """

    id: ChunkId
    source_id: SourceId
    revision_id: RevisionId
    section_id: SectionId
    text: str
    ordinal: int
    """Zero-based position of this chunk within its parent section."""

    parent_locator: str
    """Stable locator of the parent section."""

    kind: ChunkKind = ChunkKind.CHILD
    token_count: int = 0
    prev_chunk_id: ChunkId | None = None
    next_chunk_id: ChunkId | None = None


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Captures the pipeline configuration hash for a build run.

    Used to detect when chunker or embedder config has changed, requiring
    full re-ingestion (IngestionChange.INCOMPATIBLE).
    """

    value: str
    """SHA-256 (or similar) over all pipeline-config parameters."""

    description: str = ""


@dataclass(frozen=True, slots=True)
class Query:
    """A user query issued against a knowledge base."""

    id: QueryId
    text: str
    corpus_id: CorpusId | None = None
    top_k: int = 10


@dataclass(frozen=True, slots=True)
class Hit:
    """A retrieved chunk with its retrieval scores.

    Score fields are separate and independently optional to preserve
    explicit direction and avoid conflating different scoring signals.
    Never defaulting missing scores to zero prevents false confidence.

    Score directions:
    - sparse_score: higher is better (BM25 or similar term-frequency score, range >= 0)
    - dense_score: higher is better (cosine similarity, typically range [-1, 1], often [0, 1])
    - fusion_score: higher is better (reciprocal rank fusion or weighted combination, range >= 0)
    - rerank_score: higher is better (cross-encoder logit or normalized relevance, range [0, 1])
    """

    chunk: Chunk
    sparse_score: float | None = None
    """BM25 or term-frequency score. Higher is more relevant. None if not retrieved via sparse."""

    dense_score: float | None = None
    """Cosine similarity score. Higher is more relevant. None if not retrieved via dense."""

    fusion_score: float | None = None
    """Reciprocal rank fusion or weighted-combination score. Higher is better. None if not fused."""

    rerank_score: float | None = None
    """Cross-encoder relevance score. Higher is more relevant. None if not reranked."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """A piece of evidence cited in an answer.

    citation_label is a stable [S1]-style identifier assigned at answer time
    so the same chunk always receives the same label within one answer.
    Evidence items are structured objects; the system never returns
    preformatted Markdown as an evidence record.
    """

    id: EvidenceId
    hit: Hit
    citation_label: str
    """Stable label e.g. 'S1', 'S2' for inline citation in the answer text."""

    role: EvidenceRole = EvidenceRole.HIT


@dataclass(frozen=True, slots=True)
class Citation:
    """Structured citation record linking a label to a source and chunk.

    Carries enough provenance information for the caller to render citations
    in any format (footnote, sidebar, inline link) without re-querying.
    """

    label: str
    """Stable citation label e.g. 'S1', 'S2'."""

    chunk_id: ChunkId
    source_id: SourceId
    canonical_uri: str
    excerpt: str
    """Short excerpt from the chunk text used as the citation snippet."""


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    """Structured result of the answer() pipeline stage.

    answer_text is a plain string (never preformatted Markdown).
    evidence is a tuple of structured Evidence records carrying stable
    [S1]-style citation labels that match inline references in answer_text.
    Records return structured objects; callers render presentation.
    """

    query_id: QueryId
    answer_text: str
    """Plain answer string, not preformatted Markdown. May contain inline [S1] references."""

    evidence: tuple[Evidence, ...]
    """Structured evidence records; each carries a stable citation_label."""

    citations: tuple[Citation, ...] = ()
    abstained: bool = False
    """True if the generator declined to answer due to insufficient evidence."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AgenticStep:
    """One reasoning step within an agentic investigation loop.

    Records are frozen so the completed trace is immutable.
    """

    step_index: int
    action: str
    """Human-readable action label (e.g. 'search', 'synthesize', 'clarify')."""

    input_tokens: int
    output_tokens: int
    tool_name: str = ""
    tool_input: str = ""
    observation: str = ""


@dataclass(frozen=True, slots=True)
class AgenticTrace:
    """Always-on trace record for a single agentic investigation session.

    Captured regardless of whether investigation succeeds or fails.
    Records are frozen; steps is a tuple of AgenticStep records.
    The trace is the sole place where agentic loop metadata lives;
    it is not embedded in AnswerResponse to keep records decoupled.
    """

    id: TraceId
    query_id: QueryId | None
    steps: tuple[AgenticStep, ...]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    stop_reason: str = ""
    """Why the loop terminated ('budget_exceeded', 'stop_condition', 'completed')."""


@dataclass(frozen=True, slots=True)
class SyncReport:
    """Summary record produced at the end of a full or incremental sync run.

    Records are immutable. errors is a tuple of string error messages
    (not exception objects) to keep the record serializable.

    build_run_id identifies the active build run; use it to look up the run
    in the store via ``store.get_build_run(build_run_id=str(report.build_run_id))``.

    timings holds (stage_name, elapsed_seconds) pairs for each pipeline stage
    that completed during this sync.  Stage names match the 'stage' key emitted
    to the ProgressObserver.

    warnings holds non-fatal warning messages collected during the sync.
    Each entry is a human-readable string (e.g. validation warnings, skipped
    enrichment).  Warnings do not prevent promotion; errors do.

    failed_sources holds the canonical URIs of sources that could not be
    ingested in this run (fetch failure, parse failure, staging failure, etc.).
    """

    build_run_id: BuildRunId
    corpus_id: CorpusId
    status: SyncStatus
    sources_scanned: int
    sources_changed: int
    chunks_added: int
    chunks_deleted: int
    errors: tuple[str, ...]
    duration_seconds: float = 0.0
    pipeline_fingerprint: str = ""
    warnings: tuple[str, ...] = ()
    """Non-fatal warnings collected during the sync (does not block promotion)."""

    timings: tuple[tuple[str, float], ...] = ()
    """Stage-name/elapsed-seconds pairs, one per completed pipeline stage.
    Frozen-friendly: stored as a tuple of (stage_name, seconds) pairs.
    """

    failed_sources: tuple[str, ...] = ()
    """Canonical URIs of sources that could not be fully ingested this run."""
