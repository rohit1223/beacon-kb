"""Runtime-checkable Protocol contracts for every beacon-kb pipeline stage and agentic strategy.

Every protocol in this module is:
- Decorated with @runtime_checkable so isinstance() checks work structurally.
- Documented with score direction, error contract, and determinism guarantees
  in its docstring and method docstrings.

No web-search flag appears on any protocol. The Generator protocol is strictly
limited to synthesis from provided context; silent web retrieval is forbidden.

StopCondition and Tool are defined here even though v1 ships no entry-point
group for them, ensuring later versions can add groups without changing the
contract surface.

Importing this module performs no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from beacon_kb.models import (
        AnswerResponse,
        Chunk,
        ChunkId,
        CorpusId,
        Evidence,
        Hit,
        Query,
        RawDocument,
        Revision,
        RevisionId,
        Section,
    )


# ---------------------------------------------------------------------------
# Pipeline stage protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Connector(Protocol):
    """Protocol for source connectors that discover and fetch raw documents.

    Score direction: N/A - connectors do not produce scores.
    Error contract: list_sources() raises IngestionError on connectivity failure.
    fetch() raises IngestionError if the URI cannot be retrieved.
    Determinism: list_sources() output order is unspecified; callers must sort.
    """

    def list_sources(self) -> list[str]:
        """Return a list of canonical URIs for all available sources.

        Returns an unordered list of stable canonical URI strings.
        Raises IngestionError on connectivity failure.
        """
        ...

    def fetch(self, uri: str) -> RawDocument:
        """Fetch and return the raw document for the given canonical URI.

        Args:
            uri: A canonical URI returned by list_sources().

        Returns:
            RawDocument with content, media_type, and provenance fields.

        Raises:
            IngestionError if the document cannot be fetched.
        """
        ...


@runtime_checkable
class Parser(Protocol):
    """Protocol for document parsers that convert raw documents into sections.

    Score direction: N/A - parsers do not produce scores.
    Error contract: parse() raises IngestionError on malformed or unsupported input.
    Determinism: given identical RawDocument, parse() must return identical sections.
    """

    def parse(self, doc: RawDocument) -> list[Section]:
        """Parse a raw document into an ordered list of sections.

        Args:
            doc: A RawDocument fetched by a Connector.

        Returns:
            Ordered list of Section records; may be empty for blank documents.

        Raises:
            IngestionError if the document cannot be parsed.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunkers that split sections into retrieval chunks.

    Score direction: N/A - chunkers do not produce scores.
    Error contract: chunk() raises IngestionError if chunking fails.
    Determinism: given identical Section and config, chunk() must return identical
    Chunk records with identical ChunkIds across processes.
    """

    def chunk(self, section: Section) -> list[Chunk]:
        """Split a section into an ordered list of retrieval chunks.

        Args:
            section: A Section from the parser.

        Returns:
            Ordered list of Chunk records with content-addressed ChunkIds.

        Raises:
            IngestionError if chunking fails.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers that convert text to dense vectors.

    Score direction: outputs are unit-normalized vectors; cosine similarity
    between them is higher for more semantically similar texts (range [-1, 1],
    typically [0, 1] after normalization).
    Error contract: embed() raises BackendError on provider failure; never silently
    returns zero vectors.
    Determinism: given identical texts and model config, embed() should return
    identical vectors. Providers that use stochastic inference must document this.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return dense embedding vectors for the given texts.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text, in the same order.
            Vectors should be unit-normalized (L2 norm ~ 1.0).

        Raises:
            BackendError if the embedding provider fails or returns malformed output.
        """
        ...

    def dimension(self) -> int:
        """Return the dimension of the embedding vectors produced by this embedder.

        Returns:
            Positive integer dimension.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return a stable identifier for the embedding model.

        Used in the pipeline fingerprint and stored alongside embeddings so a
        model change triggers re-ingestion.  Implementations should return a
        stable, non-empty string (e.g. ``'openai/text-embedding-3-small'`` or a
        fake's canonical name).  Older embedders that predate this property are
        handled by the pipeline via a documented duck-typed fallback.
        """
        ...

    @property
    def batch_size(self) -> int:
        """Return the provider-owned batch size hint.

        Batching strategy belongs to the embedding provider.
        Core pipeline code must read this attribute rather than hardcoding a
        batch size constant.
        Implementations should return a positive integer.
        """
        ...


@runtime_checkable
class Store(Protocol):
    """Protocol for chunk storage backends (sparse index + dense vector storage).

    Covers both the basic chunk write/read surface AND the staged promotion
    lifecycle used by SyncEngine and RevisionCoordinator.  All staged-lifecycle
    methods (stage_revision, upsert_chunks_to_staging, upsert_embedding,
    promote_revision, rollback_revision) are part of this contract.

    Score direction: N/A - the store does not produce scores directly.
    Error contract: all write methods raise BackendError on failure.
    read methods raise BackendError on I/O failure and return empty results
    (not raise) for missing records.
    Determinism: reads are deterministic given stable stored state.
    """

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Write or update chunk records in the store.

        Args:
            chunks: Non-empty list of Chunk records to persist.

        Raises:
            BackendError on storage failure.
        """
        ...

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunk records by ID.

        Args:
            chunk_ids: List of ChunkId strings to delete.

        Raises:
            BackendError on storage failure.
        """
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk by ID, or None if not found.

        Args:
            chunk_id: ChunkId string.

        Returns:
            Chunk record or None if the ID does not exist.

        Raises:
            BackendError on I/O failure.
        """
        ...

    # ------------------------------------------------------------------
    # Staged promotion lifecycle methods
    # These are required by SyncEngine and RevisionCoordinator.
    # ------------------------------------------------------------------

    def stage_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision: Revision,
        canonical_uri: str = "",
    ) -> None:
        """Record a revision as staged (invisible to readers until promoted).

        Args:
            corpus_id:     Corpus namespace for the revision.
            revision:      Revision record to persist.
            canonical_uri: Canonical source URI.  When omitted, the string
                           form of ``revision.source_id`` is used.

        Raises:
            BackendError on storage failure.
        """
        ...

    def upsert_chunks_to_staging(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
        chunks: list[Chunk],
    ) -> None:
        """Write chunks to the staging area (active=0, invisible to readers).

        Chunks staged here become visible only after promote_revision() is
        called for the same revision_id.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision these chunks belong to.
            chunks:      Chunk records to stage.

        Raises:
            BackendError on storage failure.
        """
        ...

    def upsert_embedding(
        self,
        *,
        corpus_id: CorpusId,
        chunk_id: ChunkId,
        revision_id: RevisionId,
        vector: list[float],
        model_name: str,
        dimension: int,
        similarity: str,
    ) -> None:
        """Store an embedding vector for a staged chunk.

        The vector is validated against the store's declared dimension.
        Staged embeddings become visible only after promote_revision() is called.

        Args:
            corpus_id:   Corpus namespace.
            chunk_id:    The chunk this embedding belongs to.
            revision_id: The staged revision this embedding was produced for.
            vector:      Unit-normalized embedding vector (for cosine similarity).
            model_name:  Name/identifier of the embedding model.
            dimension:   Declared dimension (must match the store's vector_dim).
            similarity:  Similarity direction ('cosine', 'dot', 'euclidean').

        Raises:
            BackendError if the dimension mismatches, the similarity direction is
                unknown, or (when similarity='cosine') the vector is not
                unit-normalized.  Cosine scoring assumes unit vectors, so a
                non-unit vector is rejected at write time rather than silently
                corrupting similarity scores.
        """
        ...

    def promote_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Atomically promote a staged revision to active visibility.

        Flips all staged chunks and embeddings for *revision_id* to active=1,
        updates FTS5 index rows, retires the previous active revision for the
        same source, and updates the active_revision_pointer.

        The entire promotion happens in one transaction; partial failure leaves
        the previous revision active.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to promote.

        Returns:
            The number of chunks retired from the previous active revision that
            this promotion superseded (0 when there was no prior revision).

        Raises:
            BackendError if the revision is not found or the transaction fails.
        """
        ...

    def rollback_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> None:
        """Discard a staged revision without affecting the active revision.

        Deletes all staged chunks, embeddings, and the revision record for
        *revision_id*.  The active revision (if any) remains fully searchable.

        Error contract: rollback failure is non-fatal in the coordinator;
        the original error is always reported and the rollback failure is
        swallowed to avoid masking it.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to discard.

        Raises:
            BackendError on SQLite write failure.
        """
        ...

    def get_active_revision_id(
        self, *, corpus_id: CorpusId, canonical_uri: str
    ) -> RevisionId | None:
        """Return the active RevisionId for *canonical_uri* in *corpus_id*, or None.

        Args:
            corpus_id:     Corpus namespace.
            canonical_uri: Canonical source URI.

        Returns:
            Active RevisionId or None if no revision has been promoted.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def get_revision_hashes(
        self,
        *,
        revision_id: RevisionId,
        corpus_id: CorpusId,
    ) -> tuple[str, str] | None:
        """Return (content_hash, pipeline_fingerprint) for a revision, or None.

        Args:
            revision_id: Revision to look up.
            corpus_id:   Corpus namespace.

        Returns:
            (content_hash, pipeline_fingerprint) tuple or None if not found.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def list_active_canonical_uris(self, *, corpus_id: CorpusId) -> list[str]:
        """Return all canonical URIs with an active revision in *corpus_id*.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            Unsorted list of canonical URI strings.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def list_active_revision_fingerprints(
        self, *, corpus_id: CorpusId
    ) -> list[tuple[str, str, str]]:
        """Return (revision_id, content_hash, pipeline_fingerprint) for all active revisions.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            List of (revision_id, content_hash, pipeline_fingerprint) tuples.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def create_build_run(
        self,
        *,
        corpus_id: CorpusId,
        pipeline_fingerprint: str,
        started_at_iso: str,
    ) -> str:
        """Create a new build run record and return its string ID.

        Args:
            corpus_id:            Corpus namespace.
            pipeline_fingerprint: Hash of the pipeline configuration.
            started_at_iso:       ISO 8601 start time.

        Returns:
            BuildRunId string.

        Raises:
            BackendError on storage failure.
        """
        ...

    def finish_build_run(
        self,
        *,
        build_run_id: str,
        status: str,
        sources_scanned: int = 0,
        sources_changed: int = 0,
        chunks_added: int = 0,
        chunks_deleted: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        """Record the completion of a build run.

        Args:
            build_run_id:    BuildRunId string.
            status:          Final status ('success', 'failed', 'partial').
            sources_scanned: Count of sources examined.
            sources_changed: Count of sources that had changes.
            chunks_added:    Count of new chunks written.
            chunks_deleted:  Count of old chunks retired.
            errors:          List of error message strings (may be empty).

        Raises:
            BackendError on storage failure.
        """
        ...

    def get_build_run(self, *, build_run_id: str) -> dict[str, Any] | None:
        """Retrieve build run metadata by ID, or None if not found.

        Args:
            build_run_id: BuildRunId string.

        Returns:
            Dict of run fields, or None.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def get_latest_build_run(self, *, corpus_id: CorpusId) -> dict[str, Any] | None:
        """Return the most recent build run for *corpus_id*, or None if none exist.

        "Most recent" is defined as the build run with the latest started_at_iso
        timestamp (rowid as tiebreaker for identical timestamps).

        Args:
            corpus_id: Corpus namespace to query.

        Returns:
            Dict of run fields for the latest build run, or None if no build
            runs have ever been recorded for this corpus.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def get_staged_chunks(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> list[Chunk]:
        """Return all staged (active=0) chunks for *revision_id* in *corpus_id*.

        Used by RevisionValidator to inspect staged data before promotion
        without accessing private implementation attributes.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            List of Chunk records with active=0 for this revision.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def get_staged_embedding_count(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Return the number of staged (active=0) embeddings for *revision_id*.

        Used by RevisionValidator to confirm every staged chunk received an
        embedding before promotion, guarding against a provider miscount that
        would otherwise promote chunks with missing vectors.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            Non-negative integer count of staged embeddings.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def get_staged_embeddings(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> list[tuple[str, int]]:
        """Return (chunk_id, dimension) for every staged embedding of *revision_id*.

        Used by RevisionValidator to verify staged embedding dimensions match
        the expected vector dimension before promotion.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The staged revision to inspect.

        Returns:
            List of (chunk_id, dimension) tuples for staged (active=0) embeddings.

        Raises:
            BackendError on I/O failure.
        """
        ...

    def schema_version(self) -> int:
        """Return the highest applied migration version number.

        Used by the sync pipeline to include the schema version in the pipeline
        fingerprint, so that schema changes trigger re-ingestion automatically.

        Returns:
            Integer version number (0 if no migrations applied).

        Raises:
            BackendError on I/O failure.
        """
        ...

    def retire_revision(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
    ) -> int:
        """Retire an active revision by setting its chunks inactive and removing its pointer.

        Used when a source is deleted from the connector between syncs.
        All active chunks and embeddings for *revision_id* are deactivated and
        removed from FTS5; the active_revision_pointer for the source is deleted.
        The revision record itself is preserved for audit/history purposes.

        Error contract: raises BackendError on storage failure.

        Args:
            corpus_id:   Corpus namespace.
            revision_id: The active revision to retire.

        Returns:
            The number of active chunks retired (0 if the revision had none).

        Raises:
            BackendError on SQLite write failure.
        """
        ...


@runtime_checkable
class SparseRetriever(Protocol):
    """Protocol for BM25 or term-frequency sparse retrievers.

    Score direction: sparse_score is higher for more relevant results (range >= 0,
    unbounded above for BM25). Never default missing scores to zero.
    Error contract: retrieve() raises BackendError on index read failure.
    Determinism: given identical query and index state, retrieve() returns identical
    hits in the same order.
    """

    def retrieve(self, query: Query) -> list[Hit]:
        """Return ranked hits using sparse (term-frequency) retrieval.

        Args:
            query: Query record with text and optional corpus_id filter.

        Returns:
            List of Hit records ordered by sparse_score descending (higher is better).
            Each Hit has sparse_score set; dense_score, fusion_score, and rerank_score
            are None.

        Raises:
            BackendError on index read failure.
        """
        ...


@runtime_checkable
class DenseRetriever(Protocol):
    """Protocol for dense vector similarity retrievers.

    Score direction: dense_score is higher for more relevant results (cosine similarity,
    typically range [0, 1] for normalized vectors). Never default missing scores to zero.
    Error contract: retrieve() raises BackendError on vector store failure.
    Determinism: given identical query embedding and index state, results are stable.
    """

    def retrieve(self, query: Query) -> list[Hit]:
        """Return ranked hits using dense vector similarity retrieval.

        Args:
            query: Query record with text and optional corpus_id filter.

        Returns:
            List of Hit records ordered by dense_score descending (higher is better).
            Each Hit has dense_score set; sparse_score, fusion_score, and rerank_score
            are None.

        Raises:
            BackendError on vector store failure.
        """
        ...


@runtime_checkable
class Fusion(Protocol):
    """Protocol for score fusion strategies combining sparse and dense hits.

    Score direction: fusion_score is higher for more relevant results (range >= 0).
    Reciprocal rank fusion (RRF) typically produces values in (0, 1].
    Never default missing component scores to zero when computing fusion.
    Error contract: fuse() is pure and must not raise unless inputs are malformed.
    Determinism: given identical hit lists, fuse() always returns identical order.
    """

    def fuse(self, sparse_hits: list[Hit], dense_hits: list[Hit]) -> list[Hit]:
        """Merge and re-rank sparse and dense hits into a single fused ranking.

        Args:
            sparse_hits: Hits from a SparseRetriever, ordered by sparse_score.
            dense_hits: Hits from a DenseRetriever, ordered by dense_score.

        Returns:
            List of Hit records with fusion_score set, ordered by fusion_score descending.
            Duplicate chunk IDs are merged; the hit with the highest fusion_score wins.

        Must not raise except for malformed input (empty dict, None chunk, etc.).
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for cross-encoder or LLM-based rerankers.

    Score direction: rerank_score is higher for more relevant results (range [0, 1]
    for normalized cross-encoder logit; may vary by provider).
    Error contract: rerank() raises BackendError on provider failure.
    Determinism: given identical query and hits, rerank() should return identical
    scores. Providers using sampling must document this.
    """

    def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
        """Rerank hits using a cross-encoder or LLM-based relevance model.

        Args:
            query: Query record with the original query text.
            hits: Candidate hits to rerank (typically top-k from fusion).

        Returns:
            List of Hit records with rerank_score set, ordered by rerank_score
            descending (higher is more relevant). Preserves all input hits.

        Raises:
            BackendError on provider failure.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for LLM-based answer generators.

    Score direction: N/A - generators produce text and evidence, not ranked scores.
    Error contract: generate() raises BackendError on provider failure.
    Abstention contract: if evidence is insufficient, set abstained=True in
    AnswerResponse rather than hallucinating.
    Abstained responses must carry an empty answer_text string (""); the
    generator must not synthesize content when it cannot ground the answer
    in evidence.
    Raise CitationError only if an explicit citation validation step fails.
    Determinism: generators are generally non-deterministic; callers must not
    assume identical outputs for identical inputs.

    IMPORTANT: This protocol exposes NO web-search flag of any kind.
    Generators synthesize answers strictly from the provided context.
    Silent web retrieval is forbidden by this contract.
    """

    def generate(
        self,
        query: Query,
        hits: list[Hit],
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> AnswerResponse:
        """Generate an answer from the query and retrieved hits.

        Args:
            query: The original user query.
            hits: Retrieved and optionally reranked hits to use as context.
            max_input_tokens: Hard budget for the context window (prompt tokens).
            max_output_tokens: Hard budget for the generated response.

        Returns:
            AnswerResponse with answer_text (plain string, not Markdown),
            structured evidence with stable [S1]-style citation_labels,
            and input_tokens / output_tokens counts.
            If evidence is insufficient, returns abstained=True with empty evidence.

        Raises:
            BackendError on provider failure.
            CitationError if citation validation explicitly fails.
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting utilities.

    Score direction: N/A - returns counts, not scores.
    Error contract: count_tokens() never raises for an unknown model name.
    When the model is unrecognised or empty, the implementation falls back to a
    heuristic (character-count / chars-per-token).
    Determinism: given identical text and model, count_tokens() is deterministic.
    """

    def count_tokens(self, text: str, *, model: str = "") -> int:
        """Return the token count for the given text.

        Args:
            text: Input string to count tokens for.
            model: Optional model name to use the model's tokenizer.
                   Falls back to a heuristic (word/4) if empty or unknown.

        Returns:
            Non-negative integer token count.
        """
        ...


@runtime_checkable
class ProgressObserver(Protocol):
    """Protocol for structured pipeline progress event observers.

    Score direction: N/A - observers receive events, not scores.
    Error contract: on_event() must not raise; implementations must swallow errors
    to avoid disrupting the pipeline.
    Determinism: N/A - observers are side-effect only.
    """

    def on_event(self, event: dict[str, Any]) -> None:
        """Receive a structured pipeline progress event.

        Args:
            event: Dict with at minimum 'stage' (str) and 'status' (str) keys.
                   May include 'count', 'total', 'message', 'elapsed_seconds'.

        Must not raise; implementations swallow internal errors.
        """
        ...


# ---------------------------------------------------------------------------
# Agentic strategy protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class QueryPlanner(Protocol):
    """Protocol for query planning strategies in the agentic loop.

    Score direction: N/A - produces sub-queries, not scores.
    Error contract: plan() raises AgenticError on planning failure.
    Determinism: given identical query and corpus list, plan() should return
    identical sub-queries. Planners using sampling must document this.
    """

    def plan(self, query: Query, corpus_ids: list[CorpusId]) -> list[Query]:
        """Decompose a complex query into a list of sub-queries.

        Args:
            query: The original user query to plan for.
            corpus_ids: Available corpus IDs the planner may target.

        Returns:
            Ordered list of sub-Query records; may include the original query unchanged.

        Raises:
            AgenticError on planning failure.
        """
        ...


@runtime_checkable
class EvidenceGrader(Protocol):
    """Protocol for evidence graders that score relevance of retrieved hits.

    Score direction: grade score is higher for more relevant evidence (range [0, 1]).
    Error contract: grade() raises AgenticError on grading failure.
    Determinism: given identical query and evidence, grade() should return identical
    scores. Graders using sampling must document this.
    """

    def grade(self, query: Query, evidence: list[Evidence]) -> list[tuple[Evidence, float]]:
        """Assign relevance scores to evidence items for the given query.

        Args:
            query: The original user query.
            evidence: List of Evidence items to grade.

        Returns:
            List of (Evidence, score) tuples ordered by score descending.
            score is in [0, 1]; higher means more relevant to the query.

        Raises:
            AgenticError on grading failure.
        """
        ...


@runtime_checkable
class CorpusRouter(Protocol):
    """Protocol for corpus routing strategies that select which corpus to query.

    Score direction: N/A - produces a corpus selection, not a ranked list.
    Error contract: route() raises AgenticError on routing failure.
    Determinism: given identical query and available corpus IDs, route() should
    return the same selection. Routers using sampling must document this.
    """

    def route(self, query: Query, corpus_ids: list[CorpusId]) -> list[CorpusId]:
        """Select the corpus or corpora to query for the given query.

        Args:
            query: The user query to route.
            corpus_ids: Available corpus IDs.

        Returns:
            Ordered list of CorpusId values to query; may be a subset of corpus_ids.

        Raises:
            AgenticError on routing failure.
        """
        ...


@runtime_checkable
class StopCondition(Protocol):
    """Protocol for agentic loop stop conditions.

    v1 ships no entry-point group for StopCondition; this protocol exists
    so the contract surface is stable when groups are added in later versions.

    Score direction: N/A - returns a boolean decision.
    Error contract: should_stop() must not raise; return True on internal errors
    to prevent unbounded loops.
    Determinism: given identical trace, should_stop() should be deterministic.
    """

    def should_stop(self, trace: Any) -> bool:
        """Return True if the agentic loop should terminate.

        Args:
            trace: AgenticTrace record capturing the current investigation state.
                   Typed as Any until the contract suite and fakes exchange real
                   AgenticTrace values; implementations should narrow to AgenticTrace.

        Returns:
            True if the loop should stop; False to continue.

        Must not raise; return True on internal errors to prevent unbounded loops.
        """
        ...


@runtime_checkable
class Tool(Protocol):
    """Protocol for agentic tools callable by the investigation loop.

    v1 ships no entry-point group for Tool; this protocol exists
    so the contract surface is stable when groups are added in later versions.

    Score direction: N/A - tools produce observations, not scores.
    Error contract: call() raises AgenticError on unrecoverable tool failure;
    returns a result string for recoverable errors (e.g. empty search results).
    Determinism: tools are generally not deterministic (search results may vary).
    """

    @property
    def name(self) -> str:
        """Stable, unique name for this tool (e.g. 'search', 'lookup', 'clarify')."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description used in the tool-selection prompt."""
        ...

    def call(self, input: str) -> str:
        """Execute the tool with the given input string.

        Args:
            input: Tool input as a plain string (may be JSON for structured tools).

        Returns:
            Tool observation as a plain string.

        Raises:
            AgenticError on unrecoverable failure.
        """
        ...


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for agentic session state persistence.

    Score direction: N/A - stores session state, not scores.
    Error contract: save() and load() raise BackendError on storage failure.
    Determinism: load() is deterministic given the same session_id and stored state.
    """

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist the session state for the given session ID.

        Args:
            session_id: Unique session identifier.
            state: Arbitrary JSON-serializable session state dict.

        Raises:
            BackendError on storage failure.
        """
        ...

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session state by ID.

        Args:
            session_id: Unique session identifier.

        Returns:
            Session state dict, or None if the session does not exist.

        Raises:
            BackendError on storage failure.
        """
        ...

    def delete(self, session_id: str) -> None:
        """Delete session state by ID.

        Args:
            session_id: Unique session identifier.

        Raises:
            BackendError on storage failure.
        """
        ...


# ---------------------------------------------------------------------------
# Public exports summary (for documentation; not enforced by __all__)
# ---------------------------------------------------------------------------
# Pipeline stage protocols:
#   Connector, Parser, Chunker, Embedder, Store,
#   SparseRetriever, DenseRetriever, Fusion, Reranker, Generator,
#   TokenCounter, ProgressObserver
# Agentic strategy protocols:
#   QueryPlanner, EvidenceGrader, CorpusRouter, StopCondition, Tool, SessionStore
