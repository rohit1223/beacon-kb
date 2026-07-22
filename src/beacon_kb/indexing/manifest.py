"""Index manifest: fingerprint building and revision metadata from persisted state.

The manifest answers two questions from the durable SQLite state:
1. What is the pipeline fingerprint for the current active revision of each source?
2. Has anything changed since the last build run (content hash or pipeline)?

Design rules enforced here:
- Manifest state is ALWAYS read from the database - never from an in-memory
  counter or a standalone JSON file.
- Fingerprints are computed deterministically from content hash and pipeline
  configuration; identical inputs reproduce identical fingerprints.
- IngestionChange classification is pure (no I/O); I/O happens in the store.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from beacon_kb.errors import BackendError
from beacon_kb.models import CorpusId, IngestionChange, RevisionId
from beacon_kb.protocols import Store

# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def build_pipeline_fingerprint(*, chunker_params: dict[str, Any], embedder_version: str) -> str:
    """Return a deterministic fingerprint of the pipeline configuration.

    Inputs: chunker configuration parameters and embedder version string.
    Identical inputs always reproduce the same fingerprint across processes.

    Args:
        chunker_params:   Flat dict of chunker parameters (e.g. chunk_size,
                          chunk_overlap).  Values are coerced to strings.
        embedder_version: Stable version string for the embedder
                          (e.g. model name + version tag).

    Returns:
        32-character hex string (SHA-256 prefix).
    """
    parts = [f"{k}={v}" for k, v in sorted(chunker_params.items())]
    parts.append(f"embedder={embedder_version}")
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def build_content_fingerprint(content_hash: str, pipeline_fingerprint: str) -> str:
    """Return a combined fingerprint that changes when EITHER content OR pipeline changes.

    Used to detect INCOMPATIBLE changes: even when content is unchanged, a
    pipeline change requires full re-ingestion.

    Args:
        content_hash:         Hash of the raw document bytes.
        pipeline_fingerprint: Hash of the pipeline configuration.

    Returns:
        32-character hex string.
    """
    combined = f"{content_hash}:{pipeline_fingerprint}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Revision status record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RevisionStatus:
    """Describes the indexing status of one source URI within a corpus.

    Attributes:
        canonical_uri:        Stable canonical URI of the source.
        active_revision_id:   Currently active RevisionId, or None.
        content_hash:         Content hash of the active revision, or ''.
        pipeline_fingerprint: Pipeline fingerprint of the active revision, or ''.
        change:               Classification of change vs. candidate state.
    """

    canonical_uri: str
    active_revision_id: RevisionId | None
    content_hash: str
    pipeline_fingerprint: str
    change: IngestionChange


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


class IndexManifest:
    """Builds and validates index fingerprints from persisted database state.

    Reads ALL manifest state from the SQLiteStore - never from an in-memory
    counter or a standalone JSON file.  Restart recovery is therefore fully
    durable: opening a new SQLiteStore instance and building an IndexManifest
    reconstructs the exact same manifest as before the restart.

    Args:
        store: Open SQLiteStore instance to read manifest state from.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def revision_status(
        self,
        *,
        corpus_id: CorpusId,
        canonical_uri: str,
        candidate_content_hash: str,
        candidate_pipeline_fingerprint: str,
    ) -> RevisionStatus:
        """Classify the change state for one source URI.

        Reads the active revision from durable store state and compares it
        against the candidate content hash and pipeline fingerprint.

        Args:
            corpus_id:                        Corpus namespace.
            canonical_uri:                    Canonical source URI.
            candidate_content_hash:           Hash of the freshly fetched document.
            candidate_pipeline_fingerprint:   Hash of the current pipeline config.

        Returns:
            RevisionStatus with change classification.

        Raises:
            BackendError: If the store query fails.
        """
        active_rev_id = self._store.get_active_revision_id(
            corpus_id=corpus_id, canonical_uri=canonical_uri
        )

        if active_rev_id is None:
            return RevisionStatus(
                canonical_uri=canonical_uri,
                active_revision_id=None,
                content_hash="",
                pipeline_fingerprint="",
                change=IngestionChange.NEW,
            )

        # Load active revision metadata from the store using the public typed query.
        try:
            hashes = self._store.get_revision_hashes(
                revision_id=active_rev_id, corpus_id=corpus_id
            )
        except BackendError as exc:
            raise BackendError(
                f"IndexManifest.revision_status: cannot load revision {active_rev_id}: {exc}"
            ) from exc

        if hashes is None:
            # Active pointer exists but revision record is gone - treat as NEW.
            return RevisionStatus(
                canonical_uri=canonical_uri,
                active_revision_id=active_rev_id,
                content_hash="",
                pipeline_fingerprint="",
                change=IngestionChange.NEW,
            )

        active_content_hash, active_pipeline_fp = hashes

        # Classify the change.
        if candidate_pipeline_fingerprint != active_pipeline_fp:
            change = IngestionChange.INCOMPATIBLE
        elif candidate_content_hash != active_content_hash:
            change = IngestionChange.CHANGED
        else:
            change = IngestionChange.UNCHANGED

        return RevisionStatus(
            canonical_uri=canonical_uri,
            active_revision_id=active_rev_id,
            content_hash=active_content_hash,
            pipeline_fingerprint=active_pipeline_fp,
            change=change,
        )

    def list_active_uris(self, *, corpus_id: CorpusId) -> list[str]:
        """Return all canonical URIs with an active revision in *corpus_id*.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            Sorted list of canonical URI strings.

        Raises:
            BackendError: If the store query fails.
        """
        try:
            uris = self._store.list_active_canonical_uris(corpus_id=corpus_id)
        except BackendError as exc:
            raise BackendError(
                f"IndexManifest.list_active_uris failed: {exc}"
            ) from exc
        return sorted(uris)

    def corpus_fingerprint(self, *, corpus_id: CorpusId) -> str:
        """Return a combined fingerprint over ALL active revisions in *corpus_id*.

        Changing any active revision (content or pipeline) changes this
        fingerprint, making it suitable as a staleness check for downstream
        caches.

        Args:
            corpus_id: Corpus namespace.

        Returns:
            32-character hex fingerprint string.  Empty string if no active revisions.

        Raises:
            BackendError: If the store query fails.
        """
        try:
            entries = self._store.list_active_revision_fingerprints(corpus_id=corpus_id)
        except BackendError as exc:
            raise BackendError(
                f"IndexManifest.corpus_fingerprint failed: {exc}"
            ) from exc

        if not entries:
            return ""

        combined = "|".join(
            f"{revision_id}:{content_hash}:{pipeline_fp}"
            for revision_id, content_hash, pipeline_fp in entries
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
