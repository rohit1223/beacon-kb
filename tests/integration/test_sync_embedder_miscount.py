"""Regression test: an embedder miscount must fail the source, never truncate.

SyncEngine routes embedding through BatchEmbedder, which validates that the
provider returns exactly one vector per text of the expected dimension.  A
provider that returns FEWER vectors than chunks must raise BackendError so the
source is recorded in failed_sources and no chunks are promoted without
embeddings - never silently truncated via zip(strict=False).
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

CORPUS = CorpusId("miscount-corpus")
DOC_A = "memory://a"
# Long enough to produce multiple chunks so a miscount is observable.
CONTENT_A = (
    "# Doc A\n\n"
    + "This is a fairly long paragraph that will be chunked into several pieces. "
    * 40
)


class _MiscountingEmbedder(FakeEmbedder):
    """FakeEmbedder that drops the last vector from every batch it returns."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = super().embed(texts)
        # Return one fewer vector than requested when there is more than one.
        return vectors[:-1] if len(vectors) > 1 else vectors


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


def _make_engine(store: SQLiteStore, embedder: Any) -> SyncEngine:
    return SyncEngine(
        store=store,
        connector=MemoryConnector(corpus=str(CORPUS), sources={DOC_A: CONTENT_A}),
        parser=MarkdownParser(),
        chunker_factory=_chunker_factory,
        embedder=embedder,
        corpus_name=str(CORPUS),
    )


def test_embedder_miscount_fails_source_without_promotion(tmp_path: Any) -> None:
    """Fewer vectors than chunks fails the source; nothing is promoted."""
    store = SQLiteStore(db_path=str(tmp_path / "miscount.db"), vector_dim=16)

    # Force the batch to hold >1 chunk so the miscount is triggered.
    embedder = _MiscountingEmbedder(dim=16, batch_size=64)
    report = _make_engine(store, embedder).sync(corpus_id=CORPUS)

    # The source must fail loudly.
    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert DOC_A in report.failed_sources, (
        "An embedder miscount must record the source in failed_sources."
    )
    assert report.chunks_added == 0, (
        "No chunks may be promoted when embeddings are missing."
    )

    # Nothing was promoted: no active chunks, no active pointer.
    assert store.count_active_chunks(corpus_id=CORPUS) == 0, (
        "A miscount must leave zero active chunks (no silent truncation)."
    )
    manifest = IndexManifest(store)
    assert DOC_A not in manifest.list_active_uris(corpus_id=CORPUS), (
        "The source must not become active when its embeddings are incomplete."
    )
    store.close()


def test_correct_embedder_count_promotes(tmp_path: Any) -> None:
    """Control: a correct embedder promotes the source normally."""
    store = SQLiteStore(db_path=str(tmp_path / "ok.db"), vector_dim=16)
    report = _make_engine(store, FakeEmbedder(dim=16, batch_size=64)).sync(corpus_id=CORPUS)
    assert report.status == SyncStatus.SUCCESS
    assert report.chunks_added > 0
    assert store.count_active_chunks(corpus_id=CORPUS) > 0
    store.close()
