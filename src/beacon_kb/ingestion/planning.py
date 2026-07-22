"""Source change planning for incremental synchronization.

Classifies each source revision as unchanged, new, changed, deleted, or
pipeline-incompatible by comparing fingerprints from the manifest against
the freshly scanned sources.

Design guarantees:
- Fingerprints include parser version, chunker params, enrichment config,
  embedding model, embedding dimension, and schema version.
  Content hashes alone are never sufficient.
- Fingerprints are compared on EVERY sync call, not just when content changes.
- Pure classification: no I/O, no network calls.
- Deleted sources are detected by comparing the current scan with the active
  canonical URIs already in the manifest.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from beacon_kb.models import CorpusId, IngestionChange

if TYPE_CHECKING:
    from beacon_kb.indexing.manifest import IndexManifest

# ---------------------------------------------------------------------------
# Pipeline fingerprint construction
# ---------------------------------------------------------------------------


def build_pipeline_fingerprint(
    *,
    parser_version: str = "",
    chunker_params: dict[str, Any] | None = None,
    enrichment_version: str = "",
    embedder_model: str = "",
    embedder_dimension: int = 0,
    schema_version: int = 0,
) -> str:
    """Return a deterministic fingerprint covering the full pipeline configuration.

    The fingerprint changes when ANY pipeline parameter changes, triggering
    INCOMPATIBLE classification for all active sources that used the old
    fingerprint.

    Args:
        parser_version:      Version string identifying the parser (e.g. 'markdown-v1').
        chunker_params:      Dict of chunker configuration parameters.
                             All values are coerced to strings for hashing.
        enrichment_version:  Version string for the enrichment model/config, or ''.
        embedder_model:      Embedding model name/identifier string.
        embedder_dimension:  Integer dimension of the embedding vectors.
        schema_version:      Integer schema migration version from the store.

    Returns:
        32-character hex string (SHA-256 prefix).
    """
    cp = chunker_params or {}
    chunker_parts = "|".join(f"{k}={v}" for k, v in sorted(cp.items()))
    canonical = (
        f"parser={parser_version}"
        f"|chunker={chunker_parts}"
        f"|enrichment={enrichment_version}"
        f"|embedder={embedder_model}"
        f"|dim={embedder_dimension}"
        f"|schema={schema_version}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Source plan record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """Classification of a single source within a sync pass.

    Attributes:
        canonical_uri:      Stable canonical URI for the source.
        change:             Detected change classification.
        content_hash:       SHA-256 of the freshly fetched document bytes,
                            or '' for deleted sources.
        pipeline_fingerprint: Pipeline fingerprint at planning time.
        active_revision_id: The currently active RevisionId string, or ''.
    """

    canonical_uri: str
    change: IngestionChange
    content_hash: str
    pipeline_fingerprint: str
    active_revision_id: str = ""


# ---------------------------------------------------------------------------
# ChangeSet - the full output of a planning pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChangeSet:
    """Full result of a sync planning pass for one corpus.

    Attributes:
        corpus_id:           Corpus namespace.
        pipeline_fingerprint: The pipeline fingerprint used during this planning pass.
        unchanged:           Sources with no change to content or pipeline.
        new:                 Sources with no active revision.
        changed:             Sources with changed content.
        deleted:             Sources no longer present in the connector scan.
        incompatible:        Sources whose pipeline fingerprint has changed.
    """

    corpus_id: CorpusId
    pipeline_fingerprint: str
    unchanged: tuple[SourcePlan, ...]
    new: tuple[SourcePlan, ...]
    changed: tuple[SourcePlan, ...]
    deleted: tuple[SourcePlan, ...]
    incompatible: tuple[SourcePlan, ...]

    @property
    def needs_work(self) -> bool:
        """Return True if any source requires ingestion or deletion."""
        return bool(self.new or self.changed or self.deleted or self.incompatible)

    @property
    def total_scanned(self) -> int:
        """Total number of sources scanned (live + deleted)."""
        return (
            len(self.unchanged)
            + len(self.new)
            + len(self.changed)
            + len(self.incompatible)
            + len(self.deleted)
        )

    @property
    def total_changed(self) -> int:
        """Number of sources that require action (new + changed + deleted + incompatible)."""
        return len(self.new) + len(self.changed) + len(self.deleted) + len(self.incompatible)


# ---------------------------------------------------------------------------
# ChangeSetPlanner
# ---------------------------------------------------------------------------


class ChangeSetPlanner:
    """Classify each source revision relative to the active manifest state.

    Uses the injected manifest to look up active revisions and compares
    each source's current content hash and pipeline fingerprint against
    the stored values.

    Args:
        manifest: An IndexManifest instance providing revision_status().
    """

    def __init__(self, manifest: IndexManifest) -> None:
        self._manifest = manifest

    def plan(
        self,
        *,
        corpus_id: CorpusId,
        scanned_uris: list[str],
        content_hashes: dict[str, str],
        pipeline_fingerprint: str,
        all_listed_uris: list[str] | None = None,
        failed_sources: list[str] | None = None,
    ) -> ChangeSet:
        """Classify all scanned sources and detect deletions.

        For each URI in *scanned_uris*, calls manifest.revision_status() to
        compare the candidate content hash and pipeline fingerprint against
        the active revision.  Sources in the active manifest but absent from
        the FULL connector listing (and not merely fetch-failed) are classified
        as DELETED.

        Fingerprints are compared on EVERY call regardless of whether the
        content hash changed - this ensures pipeline changes always trigger
        re-ingestion even for unchanged content.

        Deletion safety: a transient fetch failure must NEVER retire an indexed
        source.  Deletions are therefore planned against the FULL connector
        listing (*all_listed_uris*), with fetch-failed sources
        (*failed_sources*) explicitly excluded.  A source is retired only when
        it is truly absent from list_sources() this pass, not when it was
        listed but failed to fetch.

        Args:
            corpus_id:            Corpus namespace.
            scanned_uris:         Sorted list of canonical URIs that were
                                  successfully fetched this pass (candidates for
                                  ingestion classification).
            content_hashes:       Mapping from canonical_uri to content hash.
            pipeline_fingerprint: Pipeline fingerprint computed before scanning.
            all_listed_uris:      The FULL list of canonical URIs returned by the
                                  connector's list_sources() this pass.
                                  Deletions are computed against this set, not
                                  the (smaller) successfully-fetched set.
                                  Defaults to *scanned_uris* for backward
                                  compatibility when the caller does not
                                  distinguish listing from fetching.
            failed_sources:       Canonical URIs that were listed but failed to
                                  fetch this pass.  These are NEVER retired even
                                  though they are absent from *scanned_uris*.

        Returns:
            ChangeSet with all sources classified.
        """
        scanned_set = set(scanned_uris)
        listed_set = set(all_listed_uris) if all_listed_uris is not None else scanned_set
        failed_set = set(failed_sources) if failed_sources is not None else set()
        active_uris = set(self._manifest.list_active_uris(corpus_id=corpus_id))

        unchanged: list[SourcePlan] = []
        new_sources: list[SourcePlan] = []
        changed: list[SourcePlan] = []
        incompatible: list[SourcePlan] = []

        for uri in scanned_uris:
            content_hash = content_hashes.get(uri, "")
            status = self._manifest.revision_status(
                corpus_id=corpus_id,
                canonical_uri=uri,
                candidate_content_hash=content_hash,
                candidate_pipeline_fingerprint=pipeline_fingerprint,
            )
            active_rev_id = str(status.active_revision_id) if status.active_revision_id else ""
            plan = SourcePlan(
                canonical_uri=uri,
                change=status.change,
                content_hash=content_hash,
                pipeline_fingerprint=pipeline_fingerprint,
                active_revision_id=active_rev_id,
            )
            if status.change == IngestionChange.UNCHANGED:
                unchanged.append(plan)
            elif status.change == IngestionChange.NEW:
                new_sources.append(plan)
            elif status.change == IngestionChange.CHANGED:
                changed.append(plan)
            elif status.change == IngestionChange.INCOMPATIBLE:
                incompatible.append(plan)
            else:
                # DELETED is set below; if manifest returns it treat as new.
                new_sources.append(plan)

        # Sources active in the manifest but truly absent from the connector's
        # FULL listing are deleted.  Fetch-failed sources are excluded so a
        # transient fetch error never retires an indexed source.
        deleted_uris = active_uris - listed_set - failed_set
        deleted: list[SourcePlan] = [
            SourcePlan(
                canonical_uri=uri,
                change=IngestionChange.DELETED,
                content_hash="",
                pipeline_fingerprint=pipeline_fingerprint,
                active_revision_id="",
            )
            for uri in sorted(deleted_uris)
        ]

        return ChangeSet(
            corpus_id=corpus_id,
            pipeline_fingerprint=pipeline_fingerprint,
            unchanged=tuple(unchanged),
            new=tuple(new_sources),
            changed=tuple(changed),
            deleted=tuple(deleted),
            incompatible=tuple(incompatible),
        )
