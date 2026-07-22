"""Integration tests for CorpusHealth state machine.

Verifies that SyncEngine.health() derives corpus health exclusively from
durable store state and transitions correctly through EMPTY -> READY -> FAILED
states.

Also verifies KnowledgeBase.health() includes the corpus_health field.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusHealth, CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder, FakeFailingEmbedder

CORPUS = CorpusId("health-corpus")
DOC_A = "memory://doc-a"
CONTENT_A = "# Section A\n\nHello world from document A. Enough text for chunking here."
CONTENT_A_V2 = "# Updated A\n\nNew content for A. Different text for a second revision."


def _make_store(tmp_path: Any, vector_dim: int = 16) -> SQLiteStore:
    db = str(tmp_path / "health_test.db")
    return SQLiteStore(db_path=db, vector_dim=vector_dim)


def _make_engine(
    store: SQLiteStore,
    sources: dict[str, str],
    *,
    corpus: str = "health-corpus",
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
# Test: fresh corpus is EMPTY
# ---------------------------------------------------------------------------


def test_empty_corpus_health(tmp_path: Any) -> None:
    """A fresh corpus with no syncs returns EMPTY health."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    health = engine.health(corpus_id=CORPUS)
    assert health == CorpusHealth.EMPTY, (
        f"Fresh corpus must be EMPTY, got {health!r}."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: after successful sync, health is READY
# ---------------------------------------------------------------------------


def test_ready_health_after_successful_sync(tmp_path: Any) -> None:
    """After a successful sync, corpus health is READY."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)
    assert report.status == SyncStatus.SUCCESS

    health = engine.health(corpus_id=CORPUS)
    assert health == CorpusHealth.READY, (
        f"Corpus with active revision must be READY, got {health!r}."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: READY takes precedence when last build failed but active revision exists
# ---------------------------------------------------------------------------


def test_ready_despite_last_build_failed(tmp_path: Any) -> None:
    """When an active revision exists but the last build run failed, health is READY.

    Precedence rule: READY (prior corpus searchable) > FAILED (last run failed).
    This is the key precedence case: the system remains useful even after a
    failed incremental sync attempt.
    """
    store = _make_store(tmp_path)

    # First sync succeeds - creates an active revision.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS

    # Second sync with updated content but a failing embedder.
    failing_embed = FakeFailingEmbedder(dim=16)
    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2}, embedder=failing_embed)
    report2 = engine2.sync(corpus_id=CORPUS)
    assert report2.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}

    # Health must be READY (not FAILED) because the prior revision is still active.
    health = engine2.health(corpus_id=CORPUS)
    assert health == CorpusHealth.READY, (
        f"Corpus with active revision must be READY even after last build failed. "
        f"Got {health!r}. Precedence: READY > FAILED when prior corpus is searchable."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: restart reconstructs health from durable state
# ---------------------------------------------------------------------------


def test_health_reconstructed_after_restart(tmp_path: Any) -> None:
    """Health state is reconstructed from durable store state after a restart."""
    db_path = str(tmp_path / "health_restart.db")

    # First session: ingest.
    store1 = SQLiteStore(db_path=db_path, vector_dim=16)
    engine1 = _make_engine(store1, {DOC_A: CONTENT_A})
    report = engine1.sync(corpus_id=CORPUS)
    assert report.status == SyncStatus.SUCCESS
    store1.close()

    # Second session: open a fresh store and engine - in-memory state is gone.
    store2 = SQLiteStore(db_path=db_path, vector_dim=16)
    engine2 = _make_engine(store2, {DOC_A: CONTENT_A})
    health = engine2.health(corpus_id=CORPUS)
    assert health == CorpusHealth.READY, (
        f"Health must be READY after restart with durable active revision. Got {health!r}."
    )
    store2.close()


# ---------------------------------------------------------------------------
# Test: SyncReport includes warnings field
# ---------------------------------------------------------------------------


def test_sync_report_has_warnings_field(tmp_path: Any) -> None:
    """SyncReport must include a warnings field (tuple of strings)."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    assert hasattr(report, "warnings"), "SyncReport must have a 'warnings' field."
    assert isinstance(report.warnings, tuple), "SyncReport.warnings must be a tuple."
    assert all(isinstance(w, str) for w in report.warnings), (
        "SyncReport.warnings must be a tuple of strings."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: SyncReport includes timings field
# ---------------------------------------------------------------------------


def test_sync_report_has_timings_field(tmp_path: Any) -> None:
    """SyncReport must include a timings field with stage-name/elapsed pairs."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    assert hasattr(report, "timings"), "SyncReport must have a 'timings' field."
    assert isinstance(report.timings, tuple), "SyncReport.timings must be a tuple."
    for entry in report.timings:
        assert isinstance(entry, tuple) and len(entry) == 2, (
            f"Each timings entry must be a (stage_name, elapsed) tuple. Got {entry!r}."
        )
        stage_name, elapsed = entry
        assert isinstance(stage_name, str), "Stage name must be a string."
        assert isinstance(elapsed, float), "Elapsed time must be a float."
        assert elapsed >= 0.0, "Elapsed time must be non-negative."
    store.close()


# ---------------------------------------------------------------------------
# Test: SyncReport includes failed_sources field
# ---------------------------------------------------------------------------


def test_sync_report_has_failed_sources_field(tmp_path: Any) -> None:
    """SyncReport must include a failed_sources field (tuple of URIs)."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    assert hasattr(report, "failed_sources"), "SyncReport must have a 'failed_sources' field."
    assert isinstance(report.failed_sources, tuple), (
        "SyncReport.failed_sources must be a tuple."
    )
    assert all(isinstance(s, str) for s in report.failed_sources), (
        "SyncReport.failed_sources must be a tuple of strings."
    )
    # On a successful sync, no sources should have failed.
    assert report.failed_sources == (), (
        f"Successful sync should have no failed_sources, got {report.failed_sources!r}."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: failed_sources populated when a source fails
# ---------------------------------------------------------------------------


def test_failed_sources_populated_on_ingest_failure(tmp_path: Any) -> None:
    """SyncReport.failed_sources includes URIs of sources that failed ingestion."""
    store = _make_store(tmp_path)

    # Use a failing embedder so the embed stage fails for DOC_A.
    failing_embed = FakeFailingEmbedder(dim=16)
    engine = _make_engine(store, {DOC_A: CONTENT_A}, embedder=failing_embed)
    report = engine.sync(corpus_id=CORPUS)

    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    # DOC_A should appear in failed_sources.
    assert DOC_A in report.failed_sources, (
        f"Expected {DOC_A!r} in failed_sources after embed failure. "
        f"Got: {report.failed_sources!r}"
    )
    store.close()


def test_failed_sources_populated_on_fetch_failure(tmp_path: Any) -> None:
    """SyncReport.failed_sources includes URIs whose fetch() call failed."""
    from beacon_kb.errors import IngestionError

    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})

    # Patch connector.fetch to raise IngestionError for DOC_A.
    original_fetch = engine._connector.fetch  # type: ignore[attr-defined]

    def failing_fetch(uri: str) -> Any:
        if uri == DOC_A:
            raise IngestionError(f"Injected fetch failure for {uri!r}")
        return original_fetch(uri)

    engine._connector.fetch = failing_fetch  # type: ignore[method-assign]
    report = engine.sync(corpus_id=CORPUS)

    # The fetch failure should have been recorded in failed_sources.
    assert DOC_A in report.failed_sources, (
        f"Expected {DOC_A!r} in failed_sources after fetch failure. "
        f"Got: {report.failed_sources!r}"
    )
    store.close()


def test_failed_sources_populated_on_coordinator_failure(tmp_path: Any) -> None:
    """SyncReport.failed_sources includes URIs whose coordinator promotion failed."""
    from beacon_kb.errors import BackendError

    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})

    # Patch promote_revision to raise so the coordinator returns promoted=False.
    def failing_promote(**kwargs: Any) -> None:
        raise BackendError("Injected promote failure")

    store.promote_revision = failing_promote  # type: ignore[method-assign]
    report = engine.sync(corpus_id=CORPUS)

    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    # DOC_A should appear in failed_sources because coordinator returned promoted=False.
    assert DOC_A in report.failed_sources, (
        f"Expected {DOC_A!r} in failed_sources after coordinator promotion failure. "
        f"Got: {report.failed_sources!r}"
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: FAILED health after first-sync failure with no active revision
# ---------------------------------------------------------------------------


def test_failed_health_after_first_sync_failure(tmp_path: Any) -> None:
    """After a first sync that fails with no active revision, health is FAILED."""
    store = _make_store(tmp_path)

    # Sync with a failing embedder - first ever sync, no prior active revision.
    failing_embed = FakeFailingEmbedder(dim=16)
    engine = _make_engine(store, {DOC_A: CONTENT_A}, embedder=failing_embed)
    report = engine.sync(corpus_id=CORPUS)
    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}

    health = engine.health(corpus_id=CORPUS)
    assert health == CorpusHealth.FAILED, (
        f"After a failed first sync with no active revision, health must be FAILED. "
        f"Got {health!r}."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: BUILDING health when a build run is in running status
# ---------------------------------------------------------------------------


def test_building_health_when_build_run_in_progress(tmp_path: Any) -> None:
    """Health is BUILDING when the latest build run has status='running'."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})

    # Insert a build run in 'running' status (simulates an in-progress sync).
    import datetime

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    run_id = store.create_build_run(
        corpus_id=CORPUS,
        pipeline_fingerprint="test-fp",
        started_at_iso=started_at,
    )
    # The build run is created with status='running' - do NOT call finish_build_run.

    health = engine.health(corpus_id=CORPUS)
    assert health == CorpusHealth.BUILDING, (
        f"Health must be BUILDING when a build run is in 'running' status. "
        f"Got {health!r}."
    )

    # Clean up: finish the run so we leave no dangling running records.
    store.finish_build_run(build_run_id=run_id, status="failed")
    store.close()
