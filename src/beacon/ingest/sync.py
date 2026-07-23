"""Staged sync engine for the Beacon ingestion pipeline.

Executes a change plan as one staged operation: parse, chunk, and embed only
new/changed/incompatible sources; write all surviving chunks to a shadow
collection via the staged-promotion lifecycle; validate (point counts, vector
dimensions, fingerprint match); flip the alias; then update sources,
revisions, and the job record in the state DB.

Safety invariants:

- A fully unchanged sync short-circuits after planning: zero parse, zero
  embed, and zero Qdrant write calls; the live collection keeps serving.
- A failure at any stage (parse, chunk, embed, stage write, validation,
  promote) aborts the shadow stage, leaves the prior collection serving
  through the alias, and records a FAILED job with problem detail.
- A transient fetch failure never retires an indexed source; its points are
  carried over and a warning is recorded.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from qdrant_client.http import models as qmodels

from beacon.config import BeaconSettings
from beacon.errors import BackendError
from beacon.ingest.chunking import (
    ChunkerConfig,
    DocumentChunker,
    chunk_to_payload,
    chunker_config_str,
)
from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    FetchResult,
    FetchSuccess,
    TransientFailure,
)
from beacon.ingest.embeddings import Embedder, EmbeddingResult
from beacon.ingest.fingerprint import SCHEMA_VERSION, compute_fingerprint
from beacon.ingest.parsing import PARSER_VERSION, parse
from beacon.ingest.planner import ChangePlan, plan_sync
from beacon.state.db import StateDB
from beacon.state.repo import RevisionRepo, SourceRepo, SyncJobRepo
from beacon.storage.lifecycle import StageHandle, abort, begin_stage, promote
from beacon.storage.payload import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ChunkPayload,
    chunk_id_to_point_id,
)
from beacon.storage.qdrant import QdrantStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass
class SyncReport:
    """Summary of a completed sync run.

    Attributes:
        job_id:              Sync job identifier.
        collection_name:     Logical collection name.
        sources_added:       Count of new sources indexed.
        sources_changed:     Count of changed/incompatible sources re-indexed.
        sources_deleted:     Count of sources retired.
        sources_unchanged:   Count of sources carried over unchanged.
        transient_failures:  Count of sources with transient fetch errors.
        chunks_written:      Total points written into the shadow collection
                             (0 for a short-circuited unchanged sync).
        fingerprint:         Pipeline fingerprint for this revision.
        physical_collection: Qdrant physical collection serving this revision.
        warnings:            Non-fatal warnings collected during the run.
    """

    job_id: str
    collection_name: str
    sources_added: int
    sources_changed: int
    sources_deleted: int
    sources_unchanged: int
    transient_failures: int
    chunks_written: int
    fingerprint: str
    physical_collection: str
    warnings: list[str] = field(default_factory=list)


class SyncEngine:
    """Staged sync engine.

    Orchestrates the full incremental sync pipeline:

    1. Reap stale RUNNING jobs (controller obligation: jobs cannot survive
       the process).
    2. Compute the pipeline fingerprint from parser version, chunker config,
       embedding model identity, dimension, and schema version.
    3. Plan the change set via the planner.
    4. Short-circuit when nothing changed: zero parse/embed/write work.
    5. Begin a shadow stage and record its physical collection on the
       revision row.
    6. Process new/changed/incompatible sources through parse/chunk/embed;
       carry over unchanged and transiently-failing sources verbatim.
    7. Validate the staged collection, then promote it to live.
    8. Update DB state (revision live, job succeeded/failed).

    Any stage failure aborts the shadow so the prior alias target remains
    serving, and the job is recorded FAILED with problem detail.

    Args:
        store:          Open QdrantStore instance.
        db:             Open StateDB instance.
        embedder:       Any object satisfying the ``Embedder`` protocol
                        (``EmbedderProvider`` in production, deterministic
                        counting fakes in tests).
        chunker_config: ChunkerConfig for the DocumentChunker.
        settings:       BeaconSettings for configuration.
    """

    def __init__(
        self,
        *,
        store: QdrantStore,
        db: StateDB,
        embedder: Embedder,
        chunker_config: ChunkerConfig,
        settings: BeaconSettings,
    ) -> None:
        self._store = store
        self._db = db
        self._embedder = embedder
        self._chunker_config = chunker_config
        self._settings = settings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vectors(self, result: EmbeddingResult) -> dict[str, object]:
        """Build the named-vector dict for a single EmbeddingResult.

        Args:
            result: EmbeddingResult from the embedder.

        Returns:
            Dict with DENSE_VECTOR_NAME and/or SPARSE_VECTOR_NAME entries.
        """
        vectors: dict[str, object] = {}
        if result.dense is not None:
            vectors[DENSE_VECTOR_NAME] = result.dense
        if result.sparse_indices:
            vectors[SPARSE_VECTOR_NAME] = qmodels.SparseVector(
                indices=result.sparse_indices,
                values=result.sparse_values,
            )
        return vectors

    def _process_source(
        self,
        *,
        uri: str,
        content_hash: str,
        collection_name: str,
        fingerprint: str,
        connector: Connector,
        warnings: list[str],
    ) -> list[tuple[str, dict[str, object], ChunkPayload]] | TransientFailure | ConfirmedDeletion:
        """Fetch, parse, chunk, and embed a single source.

        A fetch that no longer succeeds (the source changed state between
        planning and processing) is recorded as a warning and the actual
        ``FetchResult`` variant is returned so the caller can include its
        type name in any diagnostic message; parse/chunk/embed errors
        propagate and fail the sync so the prior revision keeps serving.

        Args:
            uri:             Canonical source URI.
            content_hash:    Expected content hash (from the planning phase).
            collection_name: Logical collection name.
            fingerprint:     Pipeline fingerprint for payload stamping.
            connector:       Connector to fetch content from.
            warnings:        Mutable warning list for the sync report.

        Returns:
            List of (point_id, vectors_dict, ChunkPayload) tuples for upsert,
            or the failed ``FetchResult`` variant (``TransientFailure`` or
            ``ConfirmedDeletion``) when the fetch no longer succeeds - the
            caller must not record the planned content hash in that case.

        Raises:
            IngestionError: On parse failure.
            BackendError:   On embedding or storage failure.
        """
        result: FetchResult = connector.fetch(uri)
        if not isinstance(result, FetchSuccess):
            warnings.append(
                f"Source {uri!r} could not be fetched during processing: {result!r}"
            )
            logger.warning(
                "Source %r could not be fetched during processing: %r", uri, result
            )
            return result

        doc = parse(result.content, result.media_type, source_uri=uri)

        chunker = DocumentChunker(
            collection=collection_name,
            canonical_uri=uri,
            content_hash=content_hash,
            config=self._chunker_config,
        )
        chunks = chunker.chunk(doc)
        if not chunks:
            return []

        embeddings = self._embedder.embed([c.text for c in chunks])

        ingested_at = _now_iso()
        points: list[tuple[str, dict[str, object], ChunkPayload]] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            payload = chunk_to_payload(
                chunk,
                source_uri=uri,
                title=doc.title,
                tags=[],
                ingested_at=ingested_at,
                content_hash=content_hash,
                fingerprint=fingerprint,
            )
            points.append(
                (chunk_id_to_point_id(chunk.chunk_id), self._build_vectors(emb), payload)
            )
        return points

    def _copy_source_points(
        self,
        *,
        source_uri: str,
        prior_collection: str,
        shadow_collection: str,
    ) -> int:
        """Copy all points for one source from the prior collection verbatim.

        Args:
            source_uri:        Canonical source URI to copy.
            prior_collection:  Physical collection to read from.
            shadow_collection: Physical collection to write into.

        Returns:
            Count of points copied.

        Raises:
            BackendError: On Qdrant failure.
        """
        records = self._store.scroll_by_source_uri(prior_collection, source_uri)
        if records:
            self._store.upsert_records(shadow_collection, records)
        return len(records)

    def _validate_stage(
        self,
        stage: StageHandle,
        *,
        expected_points: int,
        fingerprint: str,
        revision_id: str,
    ) -> None:
        """Validate the staged collection before the alias flip.

        Checks, mirroring the legacy indexing validation:

        1. Staged point count equals the planned point count.
        2. The dense vector dimension of the staged collection matches the
           embedder's fingerprinted dimension.
        3. The revision row's fingerprint equals the computed fingerprint.

        Args:
            stage:           The active StageHandle.
            expected_points: Number of points the engine wrote to the shadow.
            fingerprint:     The computed pipeline fingerprint.
            revision_id:     The revision row created for this stage.

        Raises:
            BackendError: On any validation mismatch.
        """
        info = self._store.collection_info(stage.shadow_collection)
        if info is None:
            raise BackendError(
                f"Stage validation failed: shadow collection "
                f"{stage.shadow_collection!r} does not exist"
            )

        actual_points = info.points_count or 0
        if actual_points != expected_points:
            raise BackendError(
                f"Stage validation failed: staged point count {actual_points} "
                f"does not match planned count {expected_points}"
            )

        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            dense_params = vectors_config.get(DENSE_VECTOR_NAME)
            if dense_params is not None and dense_params.size != self._embedder.dimension:
                raise BackendError(
                    f"Stage validation failed: dense dimension "
                    f"{dense_params.size} does not match fingerprinted "
                    f"dimension {self._embedder.dimension}"
                )

        revision = RevisionRepo(self._db).get(revision_id)
        if revision is None or revision["fingerprint"] != fingerprint:
            raise BackendError(
                f"Stage validation failed: revision {revision_id!r} fingerprint "
                f"does not match the computed pipeline fingerprint"
            )

    @staticmethod
    def _transient_warnings(plan: ChangePlan) -> list[str]:
        """Render transient-failure classifications as report warnings."""
        return [
            f"Transient fetch failure for {tf.uri!r}: {tf.reason}"
            for tf in plan.transient_failures
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sync(
        self,
        *,
        collection_name: str,
        connector: Connector,
        job_id: str,
    ) -> SyncReport:
        """Run a full staged sync for the collection.

        Args:
            collection_name: Logical collection name.
            connector:       Connector to enumerate and fetch sources from.
            job_id:          Pre-created PENDING job identifier.

        Returns:
            SyncReport with counters and the pipeline fingerprint.

        Raises:
            Exception: Any pipeline failure is re-raised after aborting the
                stage and marking the job FAILED; the prior revision keeps
                serving through the alias.
        """
        job_repo = SyncJobRepo(self._db)

        # Obligation: reap stale RUNNING/PENDING jobs first; jobs cannot
        # survive the process that started them.  Exclude the caller's own
        # just-created PENDING job so it is not reaped before set_running.
        job_repo.fail_stale_running(collection_name, exclude_job_id=job_id)
        job_repo.set_running(job_id)

        stage: StageHandle | None = None
        revision_id = uuid.uuid4().hex

        try:
            # Fingerprint over every pipeline component, compared on every sync.
            fingerprint = compute_fingerprint(
                parser_version=PARSER_VERSION,
                chunker_config_str=chunker_config_str(self._chunker_config),
                model_name=self._embedder.fingerprint_model_id,
                dimension=self._embedder.dimension,
                schema_version=SCHEMA_VERSION,
            )

            source_repo = SourceRepo(self._db)
            plan = plan_sync(
                connector=connector,
                collection_name=collection_name,
                current_fingerprint=fingerprint,
                source_repo=source_repo,
                db=self._db,
            )

            # Short-circuit: nothing to process, delete, or rebuild while a
            # live revision exists means zero parse/chunk/embed/write work.
            live_rev = RevisionRepo(self._db).get_live(collection_name=collection_name)
            if (
                live_rev is not None
                and not plan.sources_to_process
                and not plan.sources_to_delete
                and not plan.fingerprint_drifted
            ):
                job_repo.set_succeeded(
                    job_id,
                    sources_added=0,
                    sources_removed=0,
                    sources_unchanged=len(plan.sources_unchanged),
                )
                return SyncReport(
                    job_id=job_id,
                    collection_name=collection_name,
                    sources_added=0,
                    sources_changed=0,
                    sources_deleted=0,
                    sources_unchanged=len(plan.sources_unchanged),
                    transient_failures=len(plan.transient_failures),
                    chunks_written=0,
                    fingerprint=fingerprint,
                    physical_collection=live_rev["physical_collection"] or "",
                    warnings=self._transient_warnings(plan),
                )

            stage = begin_stage(
                self._store, collection_name, dense_dim=self._embedder.dimension
            )

            # Record the physical Qdrant collection on the revision row so
            # state rows correlate with Qdrant collections directly.
            RevisionRepo(self._db).create(
                revision_id=revision_id,
                collection_name=collection_name,
                fingerprint=fingerprint,
                physical_collection=stage.shadow_collection,
            )

            chunks_written = 0
            warnings = self._transient_warnings(plan)

            # Process new/changed/incompatible sources. Errors here fail the
            # whole sync: promoting a revision missing a planned source would
            # silently drop indexed content from the live corpus.
            for sc in plan.sources_to_process:
                points = self._process_source(
                    uri=sc.uri,
                    content_hash=sc.content_hash,
                    collection_name=collection_name,
                    fingerprint=fingerprint,
                    connector=connector,
                    warnings=warnings,
                )
                if not isinstance(points, list):
                    # A planned source that cannot be fetched during processing
                    # means the live corpus would be promoted missing that
                    # source's content - a silent data gap. Fail the sync so
                    # the prior collection keeps serving intact content; the
                    # transient condition resolves on the next sync attempt.
                    result_variant = type(points).__name__
                    raise BackendError(
                        f"Planned source {sc.uri!r} returned {result_variant}"
                        f" during processing; aborting sync to prevent promoting"
                        f" an incomplete revision. Prior collection keeps serving."
                    )
                if points:
                    self._store.upsert(stage.shadow_collection, points)
                    chunks_written += len(points)
                source_repo.upsert(
                    collection_name=collection_name,
                    canonical_uri=sc.uri,
                    connector_kind=sc.connector_kind,
                    content_hash=sc.content_hash,
                    media_type=sc.media_type,
                )

            # Carry over unchanged sources verbatim (zero parse/embed work).
            for sc in plan.sources_unchanged:
                if stage.prior_collection is not None:
                    chunks_written += self._copy_source_points(
                        source_uri=sc.uri,
                        prior_collection=stage.prior_collection,
                        shadow_collection=stage.shadow_collection,
                    )

            # Preserve transiently-failing sources: never retire them. Their
            # prior-generation points can only be carried over while the
            # fingerprint is unchanged; under drift the vectors are
            # incompatible, so the source stays active and re-indexes on the
            # first successful fetch.
            for tf in plan.transient_failures:
                if stage.prior_collection is None:
                    continue
                if plan.fingerprint_drifted:
                    warnings.append(
                        f"Source {tf.uri!r} could not be carried over under the "
                        f"new pipeline fingerprint; it will be re-indexed when "
                        f"the fetch succeeds"
                    )
                    # Clear the stored content_hash: it was recorded under the
                    # old fingerprint and this revision carries none of the
                    # source's points.  Without the clear, the next successful
                    # fetch with unchanged content would classify UNCHANGED
                    # (hash match, fingerprint match after this promote) and
                    # the source would be silently dropped from the corpus
                    # forever.  With an empty hash the next fetch classifies
                    # CHANGED and re-indexes.
                    source_repo.upsert(
                        collection_name=collection_name,
                        canonical_uri=tf.uri,
                        connector_kind=tf.connector_kind,
                        content_hash="",
                        media_type=tf.media_type or None,
                    )
                    continue
                chunks_written += self._copy_source_points(
                    source_uri=tf.uri,
                    prior_collection=stage.prior_collection,
                    shadow_collection=stage.shadow_collection,
                )

            # Confirmed deletions: retire in the DB; their points are simply
            # absent from the shadow collection.
            for sd in plan.sources_to_delete:
                source_repo.retire(
                    collection_name=collection_name, canonical_uri=sd.uri
                )

            # Validate before the flip; then promote atomically.
            self._validate_stage(
                stage,
                expected_points=chunks_written,
                fingerprint=fingerprint,
                revision_id=revision_id,
            )
            promote(self._store, stage)

            RevisionRepo(self._db).set_live(
                revision_id, collection_name=collection_name
            )

            sources_added = sum(
                1 for sc in plan.sources_to_process if sc.action == "new"
            )
            sources_changed = sum(
                1
                for sc in plan.sources_to_process
                if sc.action in ("changed", "incompatible")
            )
            job_repo.set_succeeded(
                job_id,
                sources_added=sources_added + sources_changed,
                sources_removed=len(plan.sources_to_delete),
                sources_unchanged=len(plan.sources_unchanged),
            )

            return SyncReport(
                job_id=job_id,
                collection_name=collection_name,
                sources_added=sources_added,
                sources_changed=sources_changed,
                sources_deleted=len(plan.sources_to_delete),
                sources_unchanged=len(plan.sources_unchanged),
                transient_failures=len(plan.transient_failures),
                chunks_written=chunks_written,
                fingerprint=fingerprint,
                physical_collection=stage.shadow_collection,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error(
                "Sync failed for collection %r job %r: %s",
                collection_name,
                job_id,
                exc,
            )
            if stage is not None:
                try:
                    abort(self._store, stage)
                except Exception as abort_exc:
                    logger.error("Failed to abort stage: %s", abort_exc)

            try:
                RevisionRepo(self._db).set_failed(revision_id)
            except Exception as rev_exc:
                logger.debug("Could not mark revision as failed: %s", rev_exc)

            try:
                job_repo.set_failed(
                    job_id,
                    error_detail={"message": str(exc), "type": type(exc).__name__},
                )
            except Exception as job_exc:
                logger.debug("Could not mark job as failed: %s", job_exc)

            raise
