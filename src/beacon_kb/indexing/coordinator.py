"""Revision write coordinator for staged index writes.

Coordinates sparse (FTS5), vector (embedding), and metadata writes inside
one revision transaction: stage -> validate -> promote.

Design guarantees:
- Chunks are written to staging (active=0) before any promotion.
- Embeddings are written to staging alongside chunks.
- Validation runs before promote_revision() is called.
- On validation failure or exception, rollback_revision() is called to
  discard the staged revision; the previous active revision remains searchable.
- A failed build is recorded in the build_run table with status='failed'.

Importing this module performs no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from beacon_kb.errors import BackendError
from beacon_kb.indexing.validation import RevisionValidator, ValidationResult
from beacon_kb.models import (
    Chunk,
    CorpusId,
    Revision,
    RevisionId,
    make_revision_id,
    make_source_id,
)
from beacon_kb.protocols import Store

# ---------------------------------------------------------------------------
# Staged revision outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RevisionOutcome:
    """Result of coordinating one staged revision write.

    Attributes:
        revision_id:      The RevisionId that was staged.
        promoted:         True if the revision was successfully promoted.
        chunks_written:   Number of chunks staged.
        chunks_retired:   Number of chunks retired from the previous active
                          revision that this promotion superseded (0 when there
                          was no prior revision or promotion did not occur).
        validation:       ValidationResult from pre-promotion checks.
        error:            Error message string, or '' if no error.
    """

    revision_id: RevisionId
    promoted: bool
    chunks_written: int
    validation: ValidationResult
    error: str = ""
    chunks_retired: int = 0


# ---------------------------------------------------------------------------
# RevisionCoordinator
# ---------------------------------------------------------------------------


class RevisionCoordinator:
    """Coordinate staged sparse + vector + metadata writes for one revision.

    Writes chunks to staging, writes embeddings to staging, validates,
    then promotes atomically.  On any failure, rolls back the staged revision
    so the previous active revision remains fully searchable.

    Args:
        store:     An open SQLiteStore for the corpus.
        vector_dim: Expected embedding vector dimension.
    """

    def __init__(self, *, store: Store, vector_dim: int) -> None:
        self._store = store
        self._vector_dim = vector_dim
        self._validator = RevisionValidator(store=store, vector_dim=vector_dim)

    def write_revision(
        self,
        *,
        corpus_id: CorpusId,
        canonical_uri: str,
        content_hash: str,
        pipeline_fingerprint: str,
        chunks: list[Chunk],
        embeddings: list[tuple[str, list[float]]],
        embedder_model: str,
        similarity: str = "cosine",
    ) -> RevisionOutcome:
        """Stage, validate, and promote one source revision.

        Steps:
        1. Compute a deterministic RevisionId from source_id, content_hash,
           and pipeline_fingerprint.
        2. stage_revision() - writes the revision record (active=0 for chunks).
        3. upsert_chunks_to_staging() - writes chunks with active=0.
        4. upsert_embedding() per chunk - writes embeddings with active=0.
        5. Validate counts, IDs, neighbor links, fingerprint consistency.
        6a. If validation passes: promote_revision().
        6b. If validation fails or any step raises: rollback_revision().

        Args:
            corpus_id:             Corpus namespace.
            canonical_uri:         Stable URI for the source document.
            content_hash:          SHA-256 of the DECODED document text (UTF-8),
                                   matching the sync pipeline's content-hash
                                   convention.  The single documented convention
                                   is: hash the decoded text, not the raw bytes.
            pipeline_fingerprint:  Full pipeline fingerprint string.
            chunks:                Ordered list of Chunk records to stage.
            embeddings:            List of (chunk_id_str, vector) pairs, one per chunk.
            embedder_model:        Embedding model name for the embedding table.
            similarity:            Similarity direction ('cosine', 'dot', 'euclidean').

        Returns:
            RevisionOutcome describing success or failure.
        """
        source_id = make_source_id(corpus=str(corpus_id), canonical_uri=canonical_uri)
        revision_id = make_revision_id(
            source_id=str(source_id),
            content_hash=content_hash,
            pipeline_fingerprint=pipeline_fingerprint,
        )
        revision = Revision(
            id=revision_id,
            source_id=source_id,
            content_hash=content_hash,
            pipeline_fingerprint=pipeline_fingerprint,
            byte_size=0,
            fetched_at_iso="",
        )

        error_msg = ""
        staged_ok = False

        try:
            # Stage the revision record.
            self._store.stage_revision(
                corpus_id=corpus_id,
                revision=revision,
                canonical_uri=canonical_uri,
            )
            staged_ok = True

            # Stage all chunks (active=0).
            if chunks:
                self._store.upsert_chunks_to_staging(
                    corpus_id=corpus_id,
                    revision_id=revision_id,
                    chunks=chunks,
                )

            # Stage all embeddings (active=0).
            embed_map: dict[str, list[float]] = dict(embeddings)
            for chunk in chunks:
                vec = embed_map.get(str(chunk.id))
                if vec is not None:
                    self._store.upsert_embedding(
                        corpus_id=corpus_id,
                        chunk_id=chunk.id,
                        revision_id=revision_id,
                        vector=vec,
                        model_name=embedder_model,
                        dimension=self._vector_dim,
                        similarity=similarity,
                    )

            # Pre-promotion validation.
            validation = self._validator.validate(
                corpus_id=corpus_id,
                revision_id=revision_id,
                pipeline_fingerprint=pipeline_fingerprint,
                expected_chunk_count=None,
            )

            if not validation.passed:
                # Validation failed: roll back the staged revision.
                self._store.rollback_revision(corpus_id=corpus_id, revision_id=revision_id)
                error_msg = f"Validation failed: {'; '.join(validation.errors)}"
                return RevisionOutcome(
                    revision_id=revision_id,
                    promoted=False,
                    chunks_written=len(chunks),
                    validation=validation,
                    error=error_msg,
                )

            # Promote: flip active pointers atomically.  The store returns the
            # number of chunks retired from the previous active revision that
            # this promotion superseded.
            retired = self._store.promote_revision(
                corpus_id=corpus_id, revision_id=revision_id
            )

            return RevisionOutcome(
                revision_id=revision_id,
                promoted=True,
                chunks_written=len(chunks),
                validation=validation,
                error="",
                chunks_retired=retired,
            )

        except BackendError as exc:
            error_msg = str(exc)
        except Exception as exc:
            error_msg = f"Unexpected error in write_revision: {exc}"

        # On any exception: roll back if we staged anything.
        if staged_ok:
            try:
                self._store.rollback_revision(
                    corpus_id=corpus_id, revision_id=revision_id
                )
            except BackendError:
                pass  # Rollback failure is secondary; report the original error.

        from beacon_kb.indexing.validation import ValidationResult

        empty_validation = ValidationResult(passed=False, errors=(error_msg,), warnings=())
        return RevisionOutcome(
            revision_id=revision_id,
            promoted=False,
            chunks_written=0,
            validation=empty_validation,
            error=error_msg,
        )
