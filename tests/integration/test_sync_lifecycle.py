"""Integration tests for the full sync lifecycle.

Covers: empty corpus, full sync, unchanged second sync (zero work),
add/modify/delete source, and restart reconstruction.

All tests use in-memory SQLite and fake components. No real LLM or network.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.indexing.manifest import IndexManifest
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder, FakeProgressObserver

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Any, vector_dim: int = 16) -> SQLiteStore:
    """Return a fresh SQLiteStore backed by a temporary file."""
    db = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db, vector_dim=vector_dim)


def _make_engine(
    store: SQLiteStore,
    sources: dict[str, str],
    *,
    corpus: str = "test-corpus",
    dim: int = 16,
    observer: FakeProgressObserver | None = None,
) -> SyncEngine:
    """Return a configured SyncEngine backed by *store* and *sources*."""
    connector = MemoryConnector(corpus=corpus, sources=sources)
    parser = MarkdownParser()
    embedder = FakeEmbedder(dim=dim)

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
        embedder=embedder,
        observer=observer,
        corpus_name=corpus,
    )


CORPUS = CorpusId("test-corpus")
DOC_A = "memory://doc-a"
DOC_B = "memory://doc-b"
CONTENT_A = "# Section A\n\nHello world from document A. This has enough text to produce chunks."
CONTENT_B = "# Section B\n\nHello world from document B. This has enough text to produce chunks."
CONTENT_A_V2 = "# Section A Updated\n\nUpdated content for document A. More text here for chunking."


# ---------------------------------------------------------------------------
# Test: empty corpus health
# ---------------------------------------------------------------------------


def test_empty_corpus_no_active_revisions(tmp_path: Any) -> None:
    """A fresh corpus has no active revisions."""
    store = _make_store(tmp_path)
    manifest = IndexManifest(store)
    assert manifest.list_active_uris(corpus_id=CORPUS) == []
    store.close()


# ---------------------------------------------------------------------------
# Test: full sync on empty corpus
# ---------------------------------------------------------------------------


def test_full_sync_empty_corpus(tmp_path: Any) -> None:
    """Full sync on empty corpus ingests all sources and returns SUCCESS."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A, DOC_B: CONTENT_B})
    report = engine.sync(corpus_id=CORPUS)

    assert report.status == SyncStatus.SUCCESS
    assert report.sources_scanned == 2
    assert report.sources_changed == 2
    assert report.chunks_added > 0
    assert report.chunks_deleted == 0
    assert report.pipeline_fingerprint != ""
    assert report.build_run_id != ""
    assert report.duration_seconds >= 0.0
    assert len(report.errors) == 0

    # Active revisions should now exist.
    manifest = IndexManifest(store)
    active = manifest.list_active_uris(corpus_id=CORPUS)
    assert sorted(active) == sorted([DOC_A, DOC_B])
    store.close()


# ---------------------------------------------------------------------------
# Test: unchanged second sync - ZERO writes
# ---------------------------------------------------------------------------


def test_unchanged_second_sync_zero_writes(tmp_path: Any) -> None:
    """An unchanged second sync performs zero parsing, embedding, and index writes."""
    store = _make_store(tmp_path)
    sources = {DOC_A: CONTENT_A}
    engine = _make_engine(store, sources)

    # First sync ingests everything.
    report1 = engine.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    assert report1.sources_changed == 1

    # Track embed calls by wrapping the embedder.
    embed_call_count = [0]
    original_embed = engine._embedder.embed  # type: ignore[attr-defined]

    def counting_embed(texts: list[str]) -> list[list[float]]:
        embed_call_count[0] += 1
        return original_embed(texts)

    engine._embedder.embed = counting_embed  # type: ignore[method-assign]

    parse_call_count = [0]
    original_parse = engine._parser.parse  # type: ignore[attr-defined]

    def counting_parse(doc: Any) -> Any:
        parse_call_count[0] += 1
        return original_parse(doc)

    engine._parser.parse = counting_parse  # type: ignore[method-assign]

    # Second sync with identical sources.
    report2 = engine.sync(corpus_id=CORPUS)

    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 0
    assert report2.chunks_added == 0
    # Zero parsing and embedding should have occurred.
    assert parse_call_count[0] == 0, (
        f"Expected 0 parse calls on unchanged sync, got {parse_call_count[0]}"
    )
    assert embed_call_count[0] == 0, (
        f"Expected 0 embed calls on unchanged sync, got {embed_call_count[0]}"
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: add a new source
# ---------------------------------------------------------------------------


def test_add_new_source(tmp_path: Any) -> None:
    """Adding a new source in the second sync ingests only the new source."""
    store = _make_store(tmp_path)

    # First sync with only DOC_A.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    # Second sync adds DOC_B.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A, DOC_B: CONTENT_B})
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 1  # Only DOC_B is new.
    assert report2.sources_scanned == 2

    manifest = IndexManifest(store)
    active = manifest.list_active_uris(corpus_id=CORPUS)
    assert DOC_A in active
    assert DOC_B in active
    store.close()


# ---------------------------------------------------------------------------
# Test: modify an existing source
# ---------------------------------------------------------------------------


def test_modify_source(tmp_path: Any) -> None:
    """Modifying a source re-ingests it and updates the active revision."""
    store = _make_store(tmp_path)

    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    rev_id_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_id_before is not None

    # Sync with updated content.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A_V2})
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 1

    rev_id_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_id_after is not None
    # Revision should have changed.
    assert rev_id_after != rev_id_before
    store.close()


# ---------------------------------------------------------------------------
# Test: delete a source
# ---------------------------------------------------------------------------


def test_delete_source(tmp_path: Any) -> None:
    """Deleting a source removes it from the active index after the next sync."""
    store = _make_store(tmp_path)

    engine = _make_engine(store, {DOC_A: CONTENT_A, DOC_B: CONTENT_B})
    report1 = engine.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 2

    manifest = IndexManifest(store)
    assert DOC_B in manifest.list_active_uris(corpus_id=CORPUS)

    # Sync without DOC_B - it should be deleted.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A})
    report2 = engine2.sync(corpus_id=CORPUS)

    assert report2.status == SyncStatus.SUCCESS
    # DOC_B was deleted, DOC_A unchanged.
    assert report2.sources_changed == 1

    manifest2 = IndexManifest(store)
    active = manifest2.list_active_uris(corpus_id=CORPUS)
    assert DOC_A in active
    assert DOC_B not in active
    store.close()


# ---------------------------------------------------------------------------
# Test: restart reconstructs readiness from durable state
# ---------------------------------------------------------------------------


def test_restart_reconstructs_from_durable_state(tmp_path: Any) -> None:
    """After a process restart, readiness is reconstructed from the database."""
    db_path = str(tmp_path / "test.db")
    dim = 16

    # First session: ingest documents.
    store1 = SQLiteStore(db_path=db_path, vector_dim=dim)
    engine1 = _make_engine(store1, {DOC_A: CONTENT_A, DOC_B: CONTENT_B})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    store1.close()

    # Second session: open a new store instance on the same DB.
    store2 = SQLiteStore(db_path=db_path, vector_dim=dim)
    manifest = IndexManifest(store2)
    # Active revisions must be reconstructed from durable state, not in-memory counters.
    active = manifest.list_active_uris(corpus_id=CORPUS)
    assert sorted(active) == sorted([DOC_A, DOC_B])
    store2.close()


# ---------------------------------------------------------------------------
# Test: progress events emitted
# ---------------------------------------------------------------------------


def test_sync_emits_progress_events(tmp_path: Any) -> None:
    """Sync emits structured progress events through the injected observer."""
    store = _make_store(tmp_path)
    observer = FakeProgressObserver()
    engine = _make_engine(store, {DOC_A: CONTENT_A}, observer=observer)
    engine.sync(corpus_id=CORPUS)

    stages = {e.get("stage") for e in observer.events}
    assert "sync" in stages
    store.close()


# ---------------------------------------------------------------------------
# Test: SyncReport has all required fields
# ---------------------------------------------------------------------------


def test_sync_report_fields(tmp_path: Any) -> None:
    """SyncReport returned by sync() has all required typed fields."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    assert isinstance(report.build_run_id, str)
    assert isinstance(report.corpus_id, str)
    assert isinstance(report.status, SyncStatus)
    assert isinstance(report.sources_scanned, int)
    assert isinstance(report.sources_changed, int)
    assert isinstance(report.chunks_added, int)
    assert isinstance(report.chunks_deleted, int)
    assert isinstance(report.errors, tuple)
    assert isinstance(report.duration_seconds, float)
    assert isinstance(report.pipeline_fingerprint, str)
    store.close()


# ---------------------------------------------------------------------------
# Test: build run persisted
# ---------------------------------------------------------------------------


def test_build_run_persisted_in_store(tmp_path: Any) -> None:
    """A build run record is persisted in the store after sync completes."""
    store = _make_store(tmp_path)
    engine = _make_engine(store, {DOC_A: CONTENT_A})
    report = engine.sync(corpus_id=CORPUS)

    run = store.get_build_run(build_run_id=str(report.build_run_id))
    assert run is not None
    assert run["status"] in {"success", "partial", "failed"}
    store.close()


# ---------------------------------------------------------------------------
# Test: full rebuild creates new corpus generation
# ---------------------------------------------------------------------------


def test_full_rebuild_creates_new_generation(tmp_path: Any) -> None:
    """A pipeline fingerprint change triggers a full re-ingest (new generation)."""
    store = _make_store(tmp_path)

    # First sync with default pipeline.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    # Get active revision.
    rev_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_before is not None

    # Second engine with a different embedder (changes pipeline fingerprint).
    store2_engine = _make_engine(store, {DOC_A: CONTENT_A})
    # Manually patch the embedder model name to force a fingerprint change.
    store2_engine._embedder = FakeEmbedder(dim=16, seed=999)  # type: ignore[attr-defined]
    # Force a different embedder model name to change pipeline fingerprint.
    store2_engine._embedder_model_name = lambda: "different-model"  # type: ignore[method-assign]

    report2 = store2_engine.sync(corpus_id=CORPUS)
    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 1  # Incompatible = full re-ingest.

    rev_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    # A new revision was created and promoted.
    assert rev_after is not None
    assert rev_after != rev_before
    store.close()
