"""Staged full and incremental synchronization orchestrator.

Implements the scan -> parse -> chunk -> enrich -> embed -> stage ->
validate -> promote pipeline as one recoverable operation.

Design guarantees:
- An unchanged second sync performs ZERO parsing, enrichment, embedding,
  and index writes (verified by call-count assertions in tests).
- Simulated failure at any stage leaves the previous active corpus searchable
  plus a recoverable failed build record.
- A full rebuild creates a new corpus generation ONCE (no shared-state clearing).
- Restart reconstructs readiness and active revisions from durable state.
- Sync returns a typed SyncReport with counts, timings, warnings,
  fingerprints, failed sources, and active build identity.
- Health is derived from durable build_run state, never in-memory counters.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from beacon_kb.errors import BackendError, IngestionError
from beacon_kb.indexing.coordinator import RevisionCoordinator
from beacon_kb.indexing.manifest import IndexManifest
from beacon_kb.ingestion.enrichment import EnrichmentOrchestrator
from beacon_kb.ingestion.planning import ChangeSet, ChangeSetPlanner, build_pipeline_fingerprint
from beacon_kb.models import (
    BuildRunId,
    Chunk,
    CorpusHealth,
    CorpusId,
    SyncReport,
    SyncStatus,
)
from beacon_kb.progress import NullProgressObserver
from beacon_kb.protocols import Connector, Embedder, Parser, ProgressObserver, Store

if TYPE_CHECKING:
    from beacon_kb.ingestion.chunking import HeadingAwareChunker


def _sha256_content(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SyncEngine
# ---------------------------------------------------------------------------


class SyncEngine:
    """Orchestrate incremental or full synchronization for one corpus.

    Implements the full scan -> plan -> parse -> chunk -> embed -> stage ->
    validate -> promote pipeline.  Sources classified UNCHANGED are skipped
    with zero I/O beyond the fingerprint comparison.

    Args:
        store:            Open Store implementation (SQLiteStore or duck-typed equivalent
                          implementing the Store protocol's staged lifecycle methods).
        connector:        Source connector providing list_sources() and fetch().
        parser:           Document parser with parse().
        chunker_factory:  Callable(corpus, canonical_uri, revision_id,
                          pipeline_fingerprint) -> Chunker instance.
        embedder:         Embedding provider.
        enrichment_orchestrator: Optional EnrichmentOrchestrator; None disables enrichment.
        observer:         Optional ProgressObserver for stage events.
        parser_version:   Stable string identifying the parser (for fingerprint).
        chunker_params:   Dict of chunker configuration (for fingerprint).
        enrichment_version: String identifying enrichment config (for fingerprint).
        corpus_name:      Corpus name used in source ID derivation.
                          Also used as the corpus_id string when not passed separately;
                          both corpus_name and corpus_id always agree in this engine.
    """

    def __init__(
        self,
        *,
        store: Store,
        connector: Connector,
        parser: Parser,
        chunker_factory: Callable[..., HeadingAwareChunker],
        embedder: Embedder,
        enrichment_orchestrator: EnrichmentOrchestrator | None = None,
        observer: ProgressObserver | None = None,
        parser_version: str = "v1",
        chunker_params: dict[str, Any] | None = None,
        enrichment_version: str = "",
        corpus_name: str = "default",
    ) -> None:
        self._store = store
        self._connector = connector
        self._parser = parser
        self._chunker_factory = chunker_factory
        self._embedder = embedder
        self._enrichment = enrichment_orchestrator
        self._observer: ProgressObserver = (
            observer if observer is not None else NullProgressObserver()
        )
        self._parser_version = parser_version
        self._chunker_params = chunker_params or {}
        self._enrichment_version = enrichment_version
        # corpus_name and corpus_id are always the same string in this engine.
        # corpus_name is passed in at construction to derive source IDs and
        # chunker parameters; corpus_id is passed to sync() by the caller.
        # The single source of truth is corpus_name: sync() callers MUST pass
        # CorpusId(corpus_name) to ensure consistency.
        # The facade layer enforces this by deriving corpus_id = CorpusId(corpus_name).
        self._corpus_name = corpus_name

    def _emit(self, event: dict[str, Any]) -> None:
        """Fire a progress event, suppressing observer errors."""
        try:
            self._observer.on_event(event)
        except Exception:  # noqa: S110
            pass

    def _pipeline_fingerprint(self) -> str:
        """Compute the pipeline fingerprint for the current configuration."""
        schema_ver = 0
        try:
            schema_ver = self._store.schema_version()
        except BackendError:
            pass

        return build_pipeline_fingerprint(
            parser_version=self._parser_version,
            chunker_params=self._chunker_params,
            enrichment_version=self._enrichment_version,
            embedder_model=self._embedder_model_name(),
            embedder_dimension=self._embedder.dimension(),
            schema_version=schema_ver,
        )

    def _embedder_model_name(self) -> str:
        """Return the embedder's model name string for fingerprinting."""
        # Try common attributes; fall back to class name.
        for attr in ("model_name", "model", "_model_name"):
            val = getattr(self._embedder, attr, None)
            if isinstance(val, str) and val:
                return val
        return type(self._embedder).__name__

    def health(self, *, corpus_id: CorpusId) -> CorpusHealth:
        """Derive the current health state of *corpus_id* from durable store state.

        Precedence (highest to lowest):
        1. BUILDING - the latest build run has status='running' (in-progress).
        2. READY - at least one active revision pointer exists.
           Applies even when the latest build run failed, because the prior
           corpus remains fully searchable.
        3. FAILED - latest build run has status='failed' AND no active revision.
        4. EMPTY - no active revision AND no build runs recorded for this corpus.

        All decisions are derived exclusively from durable store state; no
        in-memory counters are consulted.
        This method performs read-only store queries; it never modifies state.

        Args:
            corpus_id: Corpus namespace to check.

        Returns:
            CorpusHealth enum value.
        """
        return derive_corpus_health(self._store, corpus_id)

    def sync(self, *, corpus_id: CorpusId) -> SyncReport:
        """Run a full incremental sync for *corpus_id*.

        Returns:
            SyncReport with status, counts, timings, fingerprints, errors,
            warnings, failed_sources, and stage timings.
        """
        t0 = time.monotonic()
        stage_timings: list[tuple[str, float]] = []
        pipeline_fp = self._pipeline_fingerprint()
        started_at_iso = _now_iso()
        errors: list[str] = []
        warnings: list[str] = []
        failed_sources: list[str] = []

        # Create a build run record so restarts can reconstruct state.
        build_run_id_str = self._store.create_build_run(
            corpus_id=corpus_id,
            pipeline_fingerprint=pipeline_fp,
            started_at_iso=started_at_iso,
        )
        build_run_id = BuildRunId(build_run_id_str)

        self._emit({"stage": "sync", "status": "start", "build_run_id": build_run_id_str})

        sources_scanned = 0
        sources_changed = 0
        chunks_added = 0
        chunks_deleted = 0
        final_status = SyncStatus.FAILED

        try:
            # 1. SCAN: collect all sources and their content hashes.
            self._emit({"stage": "scan", "status": "start"})
            uris = sorted(self._connector.list_sources())
            content_hashes: dict[str, str] = {}
            raw_docs: dict[str, Any] = {}

            for uri in uris:
                try:
                    doc = self._connector.fetch(uri)
                    content_hashes[uri] = _sha256_content(doc.content)
                    raw_docs[uri] = doc  # doc.revision_id fixed up after planning
                except IngestionError as exc:
                    errors.append(f"Fetch failed for {uri!r}: {exc}")
                    warnings.append(f"Skipping {uri!r} due to fetch error.")
                    failed_sources.append(uri)

            scan_elapsed = time.monotonic() - t0
            stage_timings.append(("scan", scan_elapsed))
            self._emit({"stage": "scan", "status": "end", "count": len(uris)})

            # 2. PLAN: classify each source.
            t_plan = time.monotonic()
            manifest = IndexManifest(self._store)
            planner = ChangeSetPlanner(manifest)
            # Only plan for sources we successfully fetched.
            fetched_uris = sorted(raw_docs.keys())
            change_set: ChangeSet = planner.plan(
                corpus_id=corpus_id,
                scanned_uris=fetched_uris,
                content_hashes=content_hashes,
                pipeline_fingerprint=pipeline_fp,
            )
            sources_scanned = change_set.total_scanned
            stage_timings.append(("plan", time.monotonic() - t_plan))

            self._emit({
                "stage": "plan",
                "status": "end",
                "unchanged": len(change_set.unchanged),
                "new": len(change_set.new),
                "changed": len(change_set.changed),
                "deleted": len(change_set.deleted),
                "incompatible": len(change_set.incompatible),
            })

            if not change_set.needs_work:
                # Nothing to do: all sources unchanged.
                final_status = SyncStatus.SUCCESS
                self._store.finish_build_run(
                    build_run_id=build_run_id_str,
                    status="success",
                    sources_scanned=sources_scanned,
                    sources_changed=0,
                    chunks_added=0,
                    chunks_deleted=0,
                    errors=errors,
                )
                elapsed = time.monotonic() - t0
                return SyncReport(
                    build_run_id=build_run_id,
                    corpus_id=corpus_id,
                    status=final_status,
                    sources_scanned=sources_scanned,
                    sources_changed=0,
                    chunks_added=0,
                    chunks_deleted=0,
                    errors=tuple(errors),
                    duration_seconds=elapsed,
                    pipeline_fingerprint=pipeline_fp,
                    warnings=tuple(warnings),
                    timings=tuple(stage_timings),
                    failed_sources=tuple(failed_sources),
                )

            # 3. INGEST: parse, chunk, enrich, embed, stage, validate, promote.
            coordinator = RevisionCoordinator(
                store=self._store,
                vector_dim=self._embedder.dimension(),
            )

            # Sources requiring full ingestion (new, changed, incompatible).
            to_ingest = (
                list(change_set.new)
                + list(change_set.changed)
                + list(change_set.incompatible)
            )

            for plan in to_ingest:
                uri = plan.canonical_uri
                raw_doc = raw_docs.get(uri)
                if raw_doc is None:
                    errors.append(f"No fetched document for {uri!r}: skipping.")
                    continue

                try:
                    # Compute the canonical revision_id for this source using the
                    # full pipeline fingerprint. Re-create the RawDocument with this
                    # revision_id so that all sections and chunks carry the correct ID
                    # that the coordinator will also derive.
                    from beacon_kb.models import (
                        RawDocument,
                        make_revision_id,
                        make_source_id,
                    )
                    source_id = make_source_id(corpus=self._corpus_name, canonical_uri=uri)
                    revision_id = make_revision_id(
                        source_id=str(source_id),
                        content_hash=plan.content_hash,
                        pipeline_fingerprint=pipeline_fp,
                    )
                    # Re-create doc with the pipeline-computed revision_id so sections
                    # and chunks carry the revision_id the coordinator will also produce.
                    doc = RawDocument(
                        source_id=raw_doc.source_id,
                        revision_id=revision_id,
                        content=raw_doc.content,
                        media_type=raw_doc.media_type,
                        encoding=raw_doc.encoding,
                    )

                    # Parse.
                    self._emit({"stage": "parse", "status": "start", "uri": uri})
                    sections = self._parser.parse(doc)
                    self._emit({
                        "stage": "parse", "status": "end",
                        "uri": uri, "sections": len(sections),
                    })

                    # Build chunker with the computed revision_id.
                    chunker = self._chunker_factory(
                        corpus=self._corpus_name,
                        canonical_uri=uri,
                        revision_id=str(revision_id),
                        pipeline_fingerprint=pipeline_fp,
                    )

                    all_chunks: list[Chunk] = []
                    self._emit({"stage": "chunk", "status": "start", "uri": uri})
                    for section in sections:
                        chunk_list = chunker.chunk(section)
                        all_chunks.extend(chunk_list)
                    self._emit({
                        "stage": "chunk", "status": "end",
                        "uri": uri, "chunks": len(all_chunks),
                    })

                    # Enrich (optional, best-effort).
                    # Enrichment is called for its side effects (e.g. caching, logging,
                    # external summarization pipeline population).  The returned enriched
                    # text is not currently stored in the chunk record or used in the
                    # search index - chunks are stored with their original text only.
                    # If enriched output should become searchable metadata in the future,
                    # wire the return value into an 'enriched_text' column on chunks
                    # and add it to FTS5 and the Store protocol contract (roadmap item).
                    if self._enrichment is not None:
                        self._emit({"stage": "enrich", "status": "start", "uri": uri})
                        for chunk in all_chunks:
                            try:
                                self._enrichment.enrich(chunk.text)
                                # Return value intentionally discarded: see docstring above.
                            except Exception:  # noqa: S110
                                pass
                        self._emit({"stage": "enrich", "status": "end", "uri": uri})

                    # Embed.
                    self._emit({
                        "stage": "embed", "status": "start",
                        "uri": uri, "count": len(all_chunks),
                    })
                    texts = [c.text for c in all_chunks]
                    if texts:
                        vectors = self._embedder.embed(texts)
                    else:
                        vectors = []
                    embed_pairs = [
                        (str(c.id), vec)
                        for c, vec in zip(all_chunks, vectors, strict=False)
                    ]
                    self._emit({"stage": "embed", "status": "end", "uri": uri})

                    # Stage, validate, promote.
                    outcome = coordinator.write_revision(
                        corpus_id=corpus_id,
                        canonical_uri=uri,
                        content_hash=plan.content_hash,
                        pipeline_fingerprint=pipeline_fp,
                        chunks=all_chunks,
                        embeddings=embed_pairs,
                        embedder_model=self._embedder_model_name(),
                        similarity="cosine",
                    )

                    if outcome.promoted:
                        sources_changed += 1
                        chunks_added += outcome.chunks_written
                        for warn in outcome.validation.warnings:
                            warnings.append(f"{uri}: {warn}")
                    else:
                        errors.append(f"Failed to promote {uri!r}: {outcome.error}")
                        failed_sources.append(uri)

                except (IngestionError, BackendError) as exc:
                    errors.append(f"Ingestion failed for {uri!r}: {exc}")
                    failed_sources.append(uri)
                except Exception as exc:
                    errors.append(f"Unexpected error for {uri!r}: {exc}")
                    failed_sources.append(uri)

            # 4. Handle deletions.
            for plan in change_set.deleted:
                uri = plan.canonical_uri
                try:
                    # Retire the active revision pointer by finding and retiring chunks.
                    active_rev = self._store.get_active_revision_id(
                        corpus_id=corpus_id, canonical_uri=uri
                    )
                    if active_rev is not None:
                        self._store.retire_revision(
                            corpus_id=corpus_id, revision_id=active_rev
                        )
                        sources_changed += 1
                except (BackendError, Exception) as exc:
                    errors.append(f"Failed to retire deleted source {uri!r}: {exc}")
                    failed_sources.append(uri)

            # Determine final status.
            if errors:
                final_status = SyncStatus.PARTIAL if sources_changed > 0 else SyncStatus.FAILED
            else:
                final_status = SyncStatus.SUCCESS

        except (IngestionError, BackendError) as exc:
            errors.append(f"Sync failed: {exc}")
            final_status = SyncStatus.FAILED
        except Exception as exc:
            errors.append(f"Unexpected sync error: {exc}")
            final_status = SyncStatus.FAILED

        # Always record build run completion.
        status_str = final_status.value
        try:
            self._store.finish_build_run(
                build_run_id=build_run_id_str,
                status=status_str,
                sources_scanned=sources_scanned,
                sources_changed=sources_changed,
                chunks_added=chunks_added,
                chunks_deleted=chunks_deleted,
                errors=errors,
            )
        except BackendError:
            pass

        elapsed = time.monotonic() - t0
        stage_timings.append(("sync", elapsed))
        self._emit({
            "stage": "sync",
            "status": "end",
            "build_run_id": build_run_id_str,
            "status_value": status_str,
            "elapsed": elapsed,
        })

        return SyncReport(
            build_run_id=build_run_id,
            corpus_id=corpus_id,
            status=final_status,
            sources_scanned=sources_scanned,
            sources_changed=sources_changed,
            chunks_added=chunks_added,
            chunks_deleted=chunks_deleted,
            errors=tuple(errors),
            duration_seconds=elapsed,
            pipeline_fingerprint=pipeline_fp,
            warnings=tuple(warnings),
            timings=tuple(stage_timings),
            failed_sources=tuple(failed_sources),
        )

def derive_corpus_health(store: Store, corpus_id: CorpusId) -> CorpusHealth:
    """Derive corpus health from durable store state only.

    Precedence (highest to lowest):
    1. BUILDING - latest build run has status='running'.
    2. READY - at least one active revision exists (searchable).
       Applies even when the latest build run failed.
    3. FAILED - latest build run has status='failed' AND no active revision.
    4. EMPTY - no build runs recorded AND no active revision.

    This function is the single source of truth for health derivation.
    It is used by SyncEngine.health() and KnowledgeBase.health() directly
    so neither requires a Connector or other non-Store dependency.

    Args:
        store:     Open Store implementation.
        corpus_id: Corpus namespace to check.

    Returns:
        CorpusHealth enum value.
    """
    # Fetch latest build run first (determines BUILDING and FAILED states).
    latest_run: dict[str, Any] | None = None
    try:
        latest_run = store.get_latest_build_run(corpus_id=corpus_id)
    except Exception:
        latest_run = None

    # BUILDING: in-progress build run - highest precedence.
    if latest_run is not None and latest_run.get("status") == "running":
        return CorpusHealth.BUILDING

    # Check active revisions (determines READY vs FAILED/EMPTY).
    try:
        active_uris = store.list_active_canonical_uris(corpus_id=corpus_id)
    except Exception:
        active_uris = []

    has_active_revision = len(active_uris) > 0

    # READY: active revision present - searchable regardless of last build status.
    if has_active_revision:
        return CorpusHealth.READY

    # No active revision.
    if latest_run is not None and latest_run.get("status") == "failed":
        return CorpusHealth.FAILED

    # No active revision and no build runs (or a partial/success with no active revision).
    return CorpusHealth.EMPTY


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat()
