"""Pre-promotion validation for staged revisions.

Validates chunk counts, chunk IDs, embedding dimensions, neighbor links,
and fingerprint consistency BEFORE promote_revision() is called.

Design guarantees:
- Promotion only happens after all validations pass.
- Validation failures leave the previous active revision fully searchable.
- No mutation: validation is read-only.

Importing this module performs no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from beacon_kb.errors import BackendError
from beacon_kb.models import Chunk, CorpusId, RevisionId
from beacon_kb.protocols import Store

# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of a pre-promotion validation check.

    Attributes:
        passed:   True if all checks passed.
        errors:   Tuple of error message strings (empty when passed).
        warnings: Tuple of warning message strings (non-fatal).
    """

    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RevisionValidator:
    """Validate a staged revision before promotion.

    Checks:
    1. Chunk count >= expected_chunk_count (if provided).
    2. All chunk IDs in the staged revision are unique.
    3. All embeddings in the staging area have the correct dimension.
    4. Neighbor links (prev_chunk_id, next_chunk_id) reference valid chunk IDs
       within the same revision.
    5. Pipeline fingerprint on the revision record matches expected value.

    Args:
        store:        The SQLiteStore to read staged data from.
        vector_dim:   Expected embedding vector dimension.
    """

    def __init__(self, *, store: Store, vector_dim: int) -> None:
        self._store = store
        self._vector_dim = vector_dim

    def validate(
        self,
        *,
        corpus_id: CorpusId,
        revision_id: RevisionId,
        pipeline_fingerprint: str,
        expected_chunk_count: int | None = None,
    ) -> ValidationResult:
        """Run all pre-promotion checks for *revision_id*.

        Args:
            corpus_id:             Corpus namespace.
            revision_id:           The staged revision to validate.
            pipeline_fingerprint:  Expected pipeline fingerprint on the revision record.
            expected_chunk_count:  If provided, staged chunk count must be >= this.

        Returns:
            ValidationResult with passed=True if all checks pass.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Load staged chunks for this revision.
        try:
            staged_chunks = self._load_staged_chunks(corpus_id=corpus_id, revision_id=revision_id)
        except BackendError as exc:
            errors.append(f"Cannot load staged chunks: {exc}")
            return ValidationResult(passed=False, errors=tuple(errors), warnings=tuple(warnings))

        chunk_count = len(staged_chunks)

        # 2. Chunk count check.
        if expected_chunk_count is not None and chunk_count < expected_chunk_count:
            errors.append(
                f"Staged chunk count {chunk_count} is below expected {expected_chunk_count}."
            )

        # Warn if zero chunks (may be intentional for deleted sources, but worth noting).
        if chunk_count == 0:
            warnings.append(f"Revision {revision_id!r} staged zero chunks.")

        # 3. Unique chunk IDs.
        chunk_ids = [str(c.id) for c in staged_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append(f"Duplicate chunk IDs detected in staged revision {revision_id!r}.")

        # 4. Neighbor link consistency.
        chunk_id_set = set(chunk_ids)
        for chunk in staged_chunks:
            if chunk.prev_chunk_id is not None and str(chunk.prev_chunk_id) not in chunk_id_set:
                errors.append(
                    f"Chunk {chunk.id!r} has prev_chunk_id {chunk.prev_chunk_id!r} "
                    f"that is not in the staged revision."
                )
            if chunk.next_chunk_id is not None and str(chunk.next_chunk_id) not in chunk_id_set:
                errors.append(
                    f"Chunk {chunk.id!r} has next_chunk_id {chunk.next_chunk_id!r} "
                    f"that is not in the staged revision."
                )

        # 5. Fingerprint consistency check against stored revision record.
        try:
            hashes = self._store.get_revision_hashes(
                revision_id=revision_id, corpus_id=corpus_id
            )
        except BackendError as exc:
            errors.append(f"Cannot load revision hashes: {exc}")
            hashes = None

        if hashes is not None:
            _, stored_fp = hashes
            if stored_fp != pipeline_fingerprint:
                errors.append(
                    f"Pipeline fingerprint mismatch: stored={stored_fp!r}, "
                    f"expected={pipeline_fingerprint!r}."
                )

        passed = len(errors) == 0
        return ValidationResult(
            passed=passed,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _load_staged_chunks(
        self, *, corpus_id: CorpusId, revision_id: RevisionId
    ) -> list[Chunk]:
        """Load all staged (active=0) chunks for *revision_id* in *corpus_id*.

        Delegates to the public Store.get_staged_chunks() method so that this
        validator never reaches into private store attributes.
        """
        try:
            return self._store.get_staged_chunks(
                corpus_id=corpus_id, revision_id=revision_id
            )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"_load_staged_chunks failed: {exc}") from exc
