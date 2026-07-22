"""Regression tests for RevisionValidator staged-embedding checks.

The validator must fail a staged revision when the staged embeddings do not
match the staged chunks in count or dimension.  This is the promotion-time
guard for the embedder-miscount failure mode: even if a miscount reached the
staging area, promotion must not proceed with missing or wrong-dimension
embeddings.
"""

from __future__ import annotations

from pathlib import Path

from beacon_kb.indexing.validation import RevisionValidator
from beacon_kb.models import (
    Chunk,
    ChunkKind,
    CorpusId,
    Revision,
    RevisionId,
    SectionId,
    make_chunk_id,
    make_revision_id,
    make_source_id,
)
from beacon_kb.storage.sqlite import SQLiteStore

CORPUS = CorpusId("val-corpus")
URI = "fake://doc"
PIPELINE_FP = "pipe-v1"
CONTENT_HASH = "abc123"


def _unit_vector(dim: int) -> list[float]:
    return [1.0] + [0.0] * (dim - 1)


def _stage_two_chunks(store: SQLiteStore) -> tuple[RevisionId, list[Chunk]]:
    source_id = make_source_id(corpus=str(CORPUS), canonical_uri=URI)
    revision_id = make_revision_id(
        source_id=str(source_id),
        content_hash=CONTENT_HASH,
        pipeline_fingerprint=PIPELINE_FP,
    )
    revision = Revision(
        id=revision_id,
        source_id=source_id,
        content_hash=CONTENT_HASH,
        pipeline_fingerprint=PIPELINE_FP,
    )
    chunks = [
        Chunk(
            id=make_chunk_id(
                corpus=str(CORPUS),
                canonical_uri=URI,
                revision_id=str(revision_id),
                pipeline_fingerprint=PIPELINE_FP,
                parent_locator="intro",
                child_ordinal=i,
            ),
            source_id=source_id,
            revision_id=revision_id,
            section_id=SectionId("sec-001"),
            text=f"chunk {i} text",
            ordinal=i,
            parent_locator="intro",
            kind=ChunkKind.CHILD,
            token_count=3,
        )
        for i in range(2)
    ]
    store.stage_revision(corpus_id=CORPUS, revision=revision, canonical_uri=URI)
    store.upsert_chunks_to_staging(corpus_id=CORPUS, revision_id=revision_id, chunks=chunks)
    return revision_id, chunks


def test_validation_fails_when_embedding_missing(tmp_path: Path) -> None:
    """Two staged chunks but only one staged embedding must fail validation."""
    store = SQLiteStore(db_path=str(tmp_path / "missing.db"), vector_dim=8)
    revision_id, chunks = _stage_two_chunks(store)

    # Stage an embedding for only ONE of the two chunks (simulated miscount).
    store.upsert_embedding(
        corpus_id=CORPUS,
        chunk_id=chunks[0].id,
        revision_id=revision_id,
        vector=_unit_vector(8),
        model_name="fake",
        dimension=8,
        similarity="cosine",
    )

    validator = RevisionValidator(store=store, vector_dim=8)
    result = validator.validate(
        corpus_id=CORPUS, revision_id=revision_id, pipeline_fingerprint=PIPELINE_FP
    )
    assert result.passed is False
    assert any("embedding count" in e for e in result.errors), (
        f"Expected a staged-embedding-count error. Errors: {result.errors}"
    )
    store.close()


def test_validation_passes_when_all_embeddings_present(tmp_path: Path) -> None:
    """Every staged chunk with a matching embedding passes validation (control)."""
    store = SQLiteStore(db_path=str(tmp_path / "full.db"), vector_dim=8)
    revision_id, chunks = _stage_two_chunks(store)
    for chunk in chunks:
        store.upsert_embedding(
            corpus_id=CORPUS,
            chunk_id=chunk.id,
            revision_id=revision_id,
            vector=_unit_vector(8),
            model_name="fake",
            dimension=8,
            similarity="cosine",
        )

    validator = RevisionValidator(store=store, vector_dim=8)
    result = validator.validate(
        corpus_id=CORPUS, revision_id=revision_id, pipeline_fingerprint=PIPELINE_FP
    )
    assert result.passed is True, f"Validation should pass. Errors: {result.errors}"
    store.close()
