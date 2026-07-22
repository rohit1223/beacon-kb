"""Regression test: SyncReport.chunks_deleted reflects real retired-chunk counts.

Previously chunks_deleted was hard-wired to 0.  Both the promote-supersede path
(a content change retiring the previous revision's chunks) and the deletion path
(a source removed from the connector) must populate the count.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder

CORPUS = CorpusId("deleted-count-corpus")
DOC_A = "memory://a"
CONTENT_V1 = "# Doc A\n\nOriginal content for document A with enough words for a chunk."
CONTENT_V2 = "# Doc A v2\n\nCompletely rewritten content for document A, also chunkable."


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


def _make_engine(store: SQLiteStore, sources: dict[str, str]) -> SyncEngine:
    return SyncEngine(
        store=store,
        connector=MemoryConnector(corpus=str(CORPUS), sources=sources),
        parser=MarkdownParser(),
        chunker_factory=_chunker_factory,
        embedder=FakeEmbedder(dim=16),
        corpus_name=str(CORPUS),
    )


def test_chunks_deleted_on_content_supersede(tmp_path: Any) -> None:
    """A content change reports the retired previous-revision chunk count."""
    store = SQLiteStore(db_path=str(tmp_path / "supersede.db"), vector_dim=16)

    report1 = _make_engine(store, {DOC_A: CONTENT_V1}).sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    assert report1.chunks_deleted == 0, "First sync retires nothing."
    chunks_v1 = store.count_active_chunks(corpus_id=CORPUS)
    assert chunks_v1 > 0

    report2 = _make_engine(store, {DOC_A: CONTENT_V2}).sync(corpus_id=CORPUS)
    assert report2.status == SyncStatus.SUCCESS
    assert report2.chunks_added > 0
    assert report2.chunks_deleted == chunks_v1, (
        f"Superseding the revision must report {chunks_v1} retired chunks, "
        f"got {report2.chunks_deleted}."
    )

    # The build run row must persist the same count.
    run = store.get_build_run(build_run_id=str(report2.build_run_id))
    assert run is not None
    assert run["chunks_deleted"] == chunks_v1
    store.close()
