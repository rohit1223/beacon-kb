"""Integration tests for sync failure and rollback scenarios.

Verifies that simulated failures at any stage leave the previous active
corpus searchable and produce a recoverable failed build record.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.errors import IngestionError
from beacon_kb.indexing.manifest import IndexManifest
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder, FakeFailingEmbedder

CORPUS = CorpusId("test-corpus")
DOC_A = "memory://doc-a"
CONTENT_A = "# Section A\n\nHello world from document A. Enough text for chunking here."
CONTENT_A_V2 = "# Updated A\n\nNew content for A. Different text for a second revision."


def _make_store(tmp_path: Any, vector_dim: int = 16) -> SQLiteStore:
    db = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db, vector_dim=vector_dim)


def _make_engine(
    store: SQLiteStore,
    sources: dict[str, str],
    *,
    corpus: str = "test-corpus",
    dim: int = 16,
    embedder: Any = None,
) -> SyncEngine:
    connector = MemoryConnector(corpus=corpus, sources=sources)
    parser = MarkdownParser()
    embed = embedder if embedder is not None else FakeEmbedder(dim=dim)

    def chunker_factory(
        corpus: str,
        canonical_uri: str,
        revision_id: str,
        pipeline_fingerprint: str,
    ) -> HeadingAwareChunker:
        return HeadingAwareChunker(
            corpus=corpus,
            canonical_uri=canonical_uri,
            revision_id=revision_id,
            pipeline_fingerprint=pipeline_fingerprint,
        )

    return SyncEngine(
        store=store,
        connector=connector,
        parser=parser,
        chunker_factory=chunker_factory,
        embedder=embed,
        corpus_name=corpus,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_chunks_count(store: SQLiteStore, corpus_id: CorpusId) -> int:
    """Return the number of active chunks for *corpus_id*."""
    return store.count_active_chunks(corpus_id=corpus_id)


def _active_uris(store: SQLiteStore, corpus_id: CorpusId) -> list[str]:
    """Return the active canonical URIs for *corpus_id*."""
    manifest = IndexManifest(store)
    return manifest.list_active_uris(corpus_id=corpus_id)


# ---------------------------------------------------------------------------
# Test: failure at embed stage leaves prior revision searchable
# ---------------------------------------------------------------------------


def test_embed_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When embedding fails, the prior active revision remains searchable."""
    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS

    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Second sync with updated content but a failing embedder.
    failing_embed = FakeFailingEmbedder(dim=16)
    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2}, embedder=failing_embed)
    report2 = engine2.sync(corpus_id=CORPUS)

    # The sync reports a failure or partial.
    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(report2.errors) > 0

    # The prior revision must still be active and searchable.
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0, "Prior active chunks must remain searchable after embed failure."
    assert DOC_A in _active_uris(store, CORPUS), "Prior active URI must remain after embed failure."
    store.close()


# ---------------------------------------------------------------------------
# Test: failure at parse stage leaves prior revision searchable
# ---------------------------------------------------------------------------


def test_parse_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When parsing fails, the prior active revision remains searchable."""
    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS

    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Second sync with a parser that fails.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2})

    class FailingParser:
        def parse(self, doc: Any) -> Any:
            raise IngestionError("Injected parse failure")

    engine2._parser = FailingParser()  # type: ignore[attr-defined]

    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(report2.errors) > 0

    # Prior revision must remain active.
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0
    assert DOC_A in _active_uris(store, CORPUS)
    store.close()


# ---------------------------------------------------------------------------
# Test: failed build run recorded in store
# ---------------------------------------------------------------------------


def test_failed_build_run_recorded(tmp_path: Any) -> None:
    """A failed sync produces a recoverable build run record in the store."""
    store = _make_store(tmp_path)

    # Sync with a failing embedder from the start.
    failing_embed = FakeFailingEmbedder(dim=16)
    engine = _make_engine(store, {DOC_A: CONTENT_A}, embedder=failing_embed)
    report = engine.sync(corpus_id=CORPUS)

    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    # Build run must be persisted.
    run = store.get_build_run(build_run_id=str(report.build_run_id))
    assert run is not None
    assert run["status"] in {"failed", "partial"}
    store.close()


# ---------------------------------------------------------------------------
# Test: staged chunks cleaned up after failure
# ---------------------------------------------------------------------------


def test_staged_chunks_cleaned_up_after_failure(tmp_path: Any) -> None:
    """After a promote_revision failure, rollback cleans all staged chunks.

    We inject a promote_revision failure (after staging succeeds) so that
    rollback_revision() is exercised.  After the rollback, get_staged_chunks()
    must return an empty list for the attempted revision.
    """
    from beacon_kb.errors import BackendError

    store = _make_store(tmp_path)

    # Capture the revision_id that will be attempted, then fail at promote.
    attempted_revision_id: list[Any] = []

    def capturing_failing_promote(**kwargs: Any) -> None:
        attempted_revision_id.append(kwargs.get("revision_id"))
        raise BackendError("Injected promote failure to test rollback cleanup")

    store.promote_revision = capturing_failing_promote  # type: ignore[method-assign]

    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(attempted_revision_id) >= 1, (
        "Expected promote_revision to have been called at least once."
    )

    # rollback_revision() must have removed all staged chunks for the attempted revision.
    rev_id = attempted_revision_id[0]
    staged = store.get_staged_chunks(corpus_id=CORPUS, revision_id=rev_id)
    assert len(staged) == 0, (
        f"Expected 0 staged chunks after rollback; got {len(staged)}. "
        "rollback_revision() must clean up all staged chunks on failure."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: connector failure leaves corpus empty but no crash
# ---------------------------------------------------------------------------


def test_connector_failure_returns_failed_report(tmp_path: Any) -> None:
    """A connector that fails list_sources() returns a FAILED SyncReport."""
    store = _make_store(tmp_path)

    class FailingConnector:
        def list_sources(self) -> list[str]:
            raise IngestionError("Injected connector failure")

        def fetch(self, uri: str) -> Any:
            raise IngestionError("Injected connector fetch failure")

    engine = _make_engine(store, {DOC_A: CONTENT_A})
    engine._connector = FailingConnector()  # type: ignore[attr-defined]

    report = engine.sync(corpus_id=CORPUS)
    assert report.status == SyncStatus.FAILED
    assert len(report.errors) > 0
    store.close()


# ---------------------------------------------------------------------------
# Test: chunker raises - prior corpus remains searchable
# ---------------------------------------------------------------------------


def test_chunker_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When the chunker raises, the prior active revision remains searchable
    and a recoverable failed build record exists."""
    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Second sync with updated content but a chunker that always raises.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2})

    class FailingChunkerFactory:
        def __call__(  # type: ignore[override]
            self, corpus: Any, canonical_uri: Any, revision_id: Any, pipeline_fingerprint: Any
        ) -> Any:
            class FailingChunker:
                def chunk(self, section: Any) -> Any:
                    raise IngestionError("Injected chunker failure")
            return FailingChunker()

    engine2._chunker_factory = FailingChunkerFactory()  # type: ignore[attr-defined]
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(report2.errors) > 0

    # Prior revision must remain active and searchable.
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0, "Prior active chunks must remain searchable after chunker failure."
    assert DOC_A in _active_uris(store, CORPUS)

    # A recoverable build run record must exist.
    run = store.get_build_run(build_run_id=str(report2.build_run_id))
    assert run is not None
    assert run["status"] in {"failed", "partial"}
    store.close()


# ---------------------------------------------------------------------------
# Test: store.stage_revision raises - prior corpus remains searchable
# ---------------------------------------------------------------------------


def test_stage_revision_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When store.stage_revision raises, the prior active revision remains searchable
    and a recoverable failed build record exists."""
    from beacon_kb.errors import BackendError

    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Wrap the store so that stage_revision raises on the second call.
    original_stage_revision = store.stage_revision
    call_count = [0]

    def failing_stage_revision(**kwargs: Any) -> None:
        call_count[0] += 1
        if call_count[0] >= 1:
            raise BackendError("Injected stage_revision failure")
        return original_stage_revision(**kwargs)

    store.stage_revision = failing_stage_revision  # type: ignore[method-assign]

    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2})
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(report2.errors) > 0

    # Prior revision must remain active and searchable.
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0, "Prior active chunks must remain after stage_revision failure."
    assert DOC_A in _active_uris(store, CORPUS)

    # A recoverable build run record must exist.
    run = store.get_build_run(build_run_id=str(report2.build_run_id))
    assert run is not None
    assert run["status"] in {"failed", "partial"}
    store.close()


# ---------------------------------------------------------------------------
# Test: store.promote_revision raises after successful staging+validation
# ---------------------------------------------------------------------------


def test_promote_revision_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When store.promote_revision raises after staging and validation succeed,
    the prior active revision remains searchable and a failed build record exists.
    This is the critical failure case: all pipeline work is done but the atomic
    commit to active visibility fails."""
    from beacon_kb.errors import BackendError

    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Patch promote_revision to raise on the second sync.
    def failing_promote(**kwargs: Any) -> None:
        raise BackendError("Injected promote_revision failure")

    store.promote_revision = failing_promote  # type: ignore[method-assign]

    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2})
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert len(report2.errors) > 0

    # Prior revision must remain active (promote failed so new revision never went live).
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0, (
        "Prior active chunks must remain after promote_revision failure. "
        "The staged revision was rolled back but the prior active corpus must survive."
    )
    assert DOC_A in _active_uris(store, CORPUS), (
        "Prior active URI must remain after promote_revision failure."
    )

    # A recoverable build run record must exist.
    run = store.get_build_run(build_run_id=str(report2.build_run_id))
    assert run is not None
    assert run["status"] in {"failed", "partial"}
    store.close()


# ---------------------------------------------------------------------------
# Test: validation failure (injected via bad chunk neighbor links)
# ---------------------------------------------------------------------------


def test_validation_failure_leaves_prior_revision_searchable(tmp_path: Any) -> None:
    """When validation fails (bad chunk neighbor links), the prior active revision
    remains searchable and a failed build record exists.

    This test uses content long enough to guarantee multiple chunks so that the
    injected bogus prev_chunk_id is always caught by the validator.
    """
    from beacon_kb.models import ChunkId

    store = _make_store(tmp_path)

    # First sync succeeds.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = _active_chunks_count(store, CORPUS)
    assert chunks_before > 0

    # Use content guaranteed to produce multiple chunks so that the bogus
    # prev_chunk_id link is always detected by validation.
    # 1500+ chars of distinct prose forces at least 2 child chunks even with
    # generous chunk size defaults.
    long_content = (
        "# Long Document\n\n"
        "First paragraph with enough text to fill a chunk. " * 15
        + "\n\n## Second Section\n\n"
        "Second paragraph with enough text to fill another chunk. " * 15
        + "\n\n## Third Section\n\n"
        "Third paragraph ensuring we have many chunks. " * 15
    )

    original_factory = engine1._chunker_factory  # type: ignore[attr-defined]

    def bad_chunker_factory(
        corpus: str,
        canonical_uri: str,
        revision_id: str,
        pipeline_fingerprint: str,
    ) -> HeadingAwareChunker:
        real_chunker = original_factory(
            corpus=corpus,
            canonical_uri=canonical_uri,
            revision_id=revision_id,
            pipeline_fingerprint=pipeline_fingerprint,
        )

        class ChunkerWithBadLinks:
            def chunk(self, section: Any) -> Any:
                chunks = real_chunker.chunk(section)
                if not chunks:
                    return chunks
                # Inject a bogus prev_chunk_id that doesn't exist in the revision.
                # This is guaranteed to fail validation because the id references
                # a non-existent chunk.
                import dataclasses
                first = chunks[0]
                bad_chunk = dataclasses.replace(
                    first,
                    prev_chunk_id=ChunkId("bogus-nonexistent-chunk-id"),
                )
                return [bad_chunk, *chunks[1:]]

        return ChunkerWithBadLinks()  # type: ignore[return-value]

    engine2 = _make_engine(store, {DOC_A: long_content})
    engine2._chunker_factory = bad_chunker_factory  # type: ignore[attr-defined]
    report2 = engine2.sync(corpus_id=CORPUS)

    # Validation must have caught the bad link - no escape hatch here because
    # long_content guarantees multiple chunks are produced.
    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}, (
        f"Expected FAILED or PARTIAL when validation detects bogus prev_chunk_id; "
        f"got {report2.status!r}. Errors: {report2.errors}"
    )

    # Prior revision must remain active.
    chunks_after = _active_chunks_count(store, CORPUS)
    assert chunks_after > 0, (
        "Prior active chunks must remain after validation failure."
    )
    assert DOC_A in _active_uris(store, CORPUS)
    store.close()
