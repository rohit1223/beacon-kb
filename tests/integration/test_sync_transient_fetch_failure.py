"""Regression tests for the transient-fetch-failure retirement bug.

A source that is still listed by the connector but fails to *fetch* on a given
sync must NEVER be retired.  Retirement is reserved for sources truly absent
from list_sources().  Previously, deletions were planned against the
successfully-fetched set only, so a transient fetch failure landed the source
in ``active_uris - scanned_set`` and it was wrongly retired.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.errors import IngestionError
from beacon_kb.indexing.manifest import IndexManifest
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, Query, QueryId, RawDocument, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder

CORPUS = CorpusId("transient-corpus")
DOC_A = "memory://a"
CONTENT_A = "# Doc A\n\nHello world from document A. Enough text here for a real chunk."


class _FetchFailingConnector:
    """Connector that lists its sources but fails to fetch selected URIs.

    Wraps a MemoryConnector: list_sources() is unchanged (the source is still
    listed), but fetch() raises IngestionError for URIs in ``fail_uris`` to
    simulate a transient fetch failure.
    """

    def __init__(self, inner: MemoryConnector, *, fail_uris: set[str]) -> None:
        self._inner = inner
        self._fail_uris = fail_uris

    def list_sources(self) -> list[str]:
        return self._inner.list_sources()

    def fetch(self, uri: str) -> RawDocument:
        if uri in self._fail_uris:
            raise IngestionError(f"Injected transient fetch failure for {uri!r}")
        return self._inner.fetch(uri)


def _chunker_factory(
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


def _make_engine(store: SQLiteStore, connector: Any) -> SyncEngine:
    return SyncEngine(
        store=store,
        connector=connector,
        parser=MarkdownParser(),
        chunker_factory=_chunker_factory,
        embedder=FakeEmbedder(dim=16),
        corpus_name=str(CORPUS),
    )


def test_transient_fetch_failure_does_not_retire_indexed_source(tmp_path: Any) -> None:
    """A fetch failure on a still-listed source must not retire it."""
    store = SQLiteStore(db_path=str(tmp_path / "transient.db"), vector_dim=16)

    # Sync 1: index memory://a successfully.
    connector = MemoryConnector(corpus=str(CORPUS), sources={DOC_A: CONTENT_A})
    report1 = _make_engine(store, connector).sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = store.count_active_chunks(corpus_id=CORPUS)
    assert chunks_before > 0

    # memory://a is searchable.
    query = Query(id=QueryId("q-hello"), text="Hello world", corpus_id=CORPUS)
    hits_before = store.retrieve(query)
    assert len(hits_before) > 0, "memory://a must be searchable after sync 1."

    # Sync 2: memory://a is still LISTED but fetch fails (transient).
    failing = _FetchFailingConnector(
        MemoryConnector(corpus=str(CORPUS), sources={DOC_A: CONTENT_A}),
        fail_uris={DOC_A},
    )
    report2 = _make_engine(store, failing).sync(corpus_id=CORPUS)

    # The report reflects the fetch failure.
    assert report2.status == SyncStatus.PARTIAL, (
        f"Expected PARTIAL after a transient fetch failure, got {report2.status}."
    )
    assert DOC_A in report2.failed_sources
    assert report2.chunks_deleted == 0, (
        "A transient fetch failure must not delete any chunks."
    )

    # memory://a must STILL be active and searchable - never retired.
    manifest = IndexManifest(store)
    assert DOC_A in manifest.list_active_uris(corpus_id=CORPUS), (
        "A transient fetch failure must NOT retire the previously indexed source."
    )
    assert store.count_active_chunks(corpus_id=CORPUS) == chunks_before

    hits_after = store.retrieve(query)
    assert len(hits_after) > 0, (
        "memory://a must remain searchable after a transient fetch failure."
    )
    store.close()


def test_true_deletion_still_retires_source(tmp_path: Any) -> None:
    """A source truly absent from list_sources() is still retired (control)."""
    store = SQLiteStore(db_path=str(tmp_path / "deletion.db"), vector_dim=16)

    connector = MemoryConnector(corpus=str(CORPUS), sources={DOC_A: CONTENT_A})
    report1 = _make_engine(store, connector).sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    chunks_before = store.count_active_chunks(corpus_id=CORPUS)
    assert chunks_before > 0

    # Sync 2: memory://a is GONE from the connector entirely.
    empty_connector = MemoryConnector(corpus=str(CORPUS), sources={})
    report2 = _make_engine(store, empty_connector).sync(corpus_id=CORPUS)

    manifest = IndexManifest(store)
    assert DOC_A not in manifest.list_active_uris(corpus_id=CORPUS), (
        "A source truly absent from list_sources() must be retired."
    )
    assert report2.chunks_deleted == chunks_before, (
        f"Retiring the source must report {chunks_before} deleted chunks, "
        f"got {report2.chunks_deleted}."
    )
    store.close()
