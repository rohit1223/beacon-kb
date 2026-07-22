"""Integration tests for multi-source sync scenarios.

Verifies mixed change sets: some sources unchanged, others new/changed/deleted.
Also verifies corpus isolation (two corpora with identical URIs never share records).
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
from beacon_kb.testing import FakeEmbedder

CORPUS_A = CorpusId("corpus-a")
CORPUS_B = CorpusId("corpus-b")

DOC_1 = "memory://doc-1"
DOC_2 = "memory://doc-2"
DOC_3 = "memory://doc-3"

TEXT_1 = "# Doc One\n\nDocument one content. Enough text for chunking here properly."
TEXT_2 = "# Doc Two\n\nDocument two content. Different content for variety."
TEXT_3 = "# Doc Three\n\nDocument three content. Third document in the test set."
TEXT_1_V2 = "# Doc One Updated\n\nUpdated content for document one. New revision here."


def _make_store(tmp_path: Any, vector_dim: int = 16) -> SQLiteStore:
    db = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db, vector_dim=vector_dim)


def _make_engine(
    store: SQLiteStore,
    sources: dict[str, str],
    *,
    corpus: str,
    dim: int = 16,
) -> SyncEngine:
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
        corpus_name=corpus,
    )


# ---------------------------------------------------------------------------
# Test: mixed change set (unchanged + changed + new + deleted)
# ---------------------------------------------------------------------------


def test_mixed_changeset_sync(tmp_path: Any) -> None:
    """A sync with mixed changes correctly handles each source classification."""
    store = _make_store(tmp_path)
    corpus = "test-corpus"
    corpus_id = CorpusId(corpus)

    # First sync: ingest DOC_1 and DOC_2.
    engine1 = _make_engine(store, {DOC_1: TEXT_1, DOC_2: TEXT_2}, corpus=corpus)
    report1 = engine1.sync(corpus_id=corpus_id)
    assert report1.status == SyncStatus.SUCCESS
    assert report1.sources_changed == 2

    # Second sync: DOC_1 changed, DOC_2 unchanged, DOC_3 added, DOC_2 not deleted.
    engine2 = _make_engine(
        store,
        {DOC_1: TEXT_1_V2, DOC_2: TEXT_2, DOC_3: TEXT_3},
        corpus=corpus,
    )
    report2 = engine2.sync(corpus_id=corpus_id)

    assert report2.status == SyncStatus.SUCCESS
    # DOC_1 changed + DOC_3 new = 2 changed.
    assert report2.sources_changed == 2
    # DOC_2 unchanged.
    assert report2.sources_scanned == 3

    manifest = IndexManifest(store)
    active = manifest.list_active_uris(corpus_id=corpus_id)
    assert DOC_1 in active
    assert DOC_2 in active
    assert DOC_3 in active
    store.close()


# ---------------------------------------------------------------------------
# Test: deletion with unchanged siblings
# ---------------------------------------------------------------------------


def test_deletion_with_unchanged_siblings(tmp_path: Any) -> None:
    """Deleting one source does not affect unchanged siblings."""
    store = _make_store(tmp_path)
    corpus = "test-corpus"
    corpus_id = CorpusId(corpus)

    # Ingest all three.
    engine1 = _make_engine(
        store, {DOC_1: TEXT_1, DOC_2: TEXT_2, DOC_3: TEXT_3}, corpus=corpus
    )
    report1 = engine1.sync(corpus_id=corpus_id)
    assert report1.sources_changed == 3

    # Remove DOC_3; DOC_1 and DOC_2 unchanged.
    engine2 = _make_engine(store, {DOC_1: TEXT_1, DOC_2: TEXT_2}, corpus=corpus)
    report2 = engine2.sync(corpus_id=corpus_id)

    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 1  # Only DOC_3 was deleted.

    manifest = IndexManifest(store)
    active = manifest.list_active_uris(corpus_id=corpus_id)
    assert DOC_1 in active
    assert DOC_2 in active
    assert DOC_3 not in active
    store.close()


# ---------------------------------------------------------------------------
# Test: corpus namespace isolation
# ---------------------------------------------------------------------------


def test_corpus_namespace_isolation(tmp_path: Any) -> None:
    """Two corpora with identical source URIs never share chunk records."""
    store = _make_store(tmp_path)

    # Sync the same URI in two different corpora.
    engine_a = _make_engine(store, {DOC_1: TEXT_1}, corpus="corpus-a")
    engine_b = _make_engine(store, {DOC_1: TEXT_2}, corpus="corpus-b")  # Different content!

    report_a = engine_a.sync(corpus_id=CORPUS_A)
    report_b = engine_b.sync(corpus_id=CORPUS_B)

    assert report_a.status == SyncStatus.SUCCESS
    assert report_b.status == SyncStatus.SUCCESS

    manifest_a = IndexManifest(store)
    manifest_b = IndexManifest(store)

    active_a = manifest_a.list_active_uris(corpus_id=CORPUS_A)
    active_b = manifest_b.list_active_uris(corpus_id=CORPUS_B)

    assert DOC_1 in active_a
    assert DOC_1 in active_b

    # Active revision IDs must be different (different content -> different revision).
    rev_a = store.get_active_revision_id(corpus_id=CORPUS_A, canonical_uri=DOC_1)
    rev_b = store.get_active_revision_id(corpus_id=CORPUS_B, canonical_uri=DOC_1)
    assert rev_a is not None
    assert rev_b is not None
    assert rev_a != rev_b, (
        "Two corpora with the same URI but different content must have different revisions."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: incremental sync is efficient (unchanged sources skip I/O)
# ---------------------------------------------------------------------------


def test_incremental_sync_skips_unchanged(tmp_path: Any) -> None:
    """Multiple unchanged sources are all skipped on second sync."""
    store = _make_store(tmp_path)
    corpus = "test-corpus"
    corpus_id = CorpusId(corpus)

    many_sources = {
        f"memory://doc-{i}": f"# Doc {i}\n\nContent {i} for testing purposes."
        for i in range(5)
    }

    # First sync ingests all.
    engine1 = _make_engine(store, many_sources, corpus=corpus)
    report1 = engine1.sync(corpus_id=corpus_id)
    assert report1.status == SyncStatus.SUCCESS
    assert report1.sources_changed == 5

    # Track parse calls.
    parse_calls = [0]
    orig_parse = engine1._parser.parse  # type: ignore[attr-defined]

    def counting_parse(doc: Any) -> Any:
        parse_calls[0] += 1
        return orig_parse(doc)

    engine1._parser.parse = counting_parse  # type: ignore[method-assign]

    # Second sync: all unchanged.
    report2 = engine1.sync(corpus_id=corpus_id)
    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 0
    assert parse_calls[0] == 0, (
        f"No parsing should occur for unchanged sources; got {parse_calls[0]} calls."
    )
    store.close()


# ---------------------------------------------------------------------------
# Test: consecutive add-change-delete cycle
# ---------------------------------------------------------------------------


def test_add_change_delete_cycle(tmp_path: Any) -> None:
    """A full add -> change -> delete cycle produces correct state at each step."""
    store = _make_store(tmp_path)
    corpus = "test-corpus"
    corpus_id = CorpusId(corpus)

    manifest = IndexManifest(store)

    # Step 1: Add DOC_1.
    e1 = _make_engine(store, {DOC_1: TEXT_1}, corpus=corpus)
    r1 = e1.sync(corpus_id=corpus_id)
    assert r1.sources_changed == 1
    assert DOC_1 in manifest.list_active_uris(corpus_id=corpus_id)

    # Step 2: Change DOC_1.
    e2 = _make_engine(store, {DOC_1: TEXT_1_V2}, corpus=corpus)
    r2 = e2.sync(corpus_id=corpus_id)
    assert r2.sources_changed == 1
    assert DOC_1 in manifest.list_active_uris(corpus_id=corpus_id)

    # Step 3: Delete DOC_1 (empty connector).
    e3 = _make_engine(store, {}, corpus=corpus)
    r3 = e3.sync(corpus_id=corpus_id)
    assert r3.sources_changed == 1  # Deletion counted.
    assert DOC_1 not in manifest.list_active_uris(corpus_id=corpus_id)

    store.close()
