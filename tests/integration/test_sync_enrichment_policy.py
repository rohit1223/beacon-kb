"""Regression tests for enrichment failure-policy propagation in SyncEngine.

The SyncEngine must honour the EnrichmentOrchestrator failure policy:
- 'best-effort': a failing enricher is swallowed and ingestion continues.
- 'raise':       a failing enricher propagates, failing that source.

Previously the SyncEngine wrapped enrichment in a bare ``except Exception: pass``
which neutered the 'raise' policy.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.indexing.manifest import IndexManifest
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.enrichment import EnrichmentOrchestrator
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder, FakeEnricher, FakeFailingEnricher

CORPUS = CorpusId("enrich-corpus")
DOC_A = "memory://a"
CONTENT_A = "# Doc A\n\nHello world from document A. Enough text here for a real chunk."


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


def _make_engine(store: SQLiteStore, orchestrator: EnrichmentOrchestrator) -> SyncEngine:
    return SyncEngine(
        store=store,
        connector=MemoryConnector(corpus=str(CORPUS), sources={DOC_A: CONTENT_A}),
        parser=MarkdownParser(),
        chunker_factory=_chunker_factory,
        embedder=FakeEmbedder(dim=16),
        enrichment_orchestrator=orchestrator,
        corpus_name=str(CORPUS),
    )


def test_enrichment_raise_policy_fails_source(tmp_path: Any) -> None:
    """Under 'raise' policy, a failing enricher fails the source loudly."""
    store = SQLiteStore(db_path=str(tmp_path / "raise.db"), vector_dim=16)
    orchestrator = EnrichmentOrchestrator(
        enricher=FakeFailingEnricher(), failure_policy="raise"
    )
    report = _make_engine(store, orchestrator).sync(corpus_id=CORPUS)

    assert report.status in {SyncStatus.FAILED, SyncStatus.PARTIAL}
    assert DOC_A in report.failed_sources, (
        "Under 'raise' policy a failing enricher must fail the source, not be swallowed."
    )
    # The source must not become active when enrichment raised.
    manifest = IndexManifest(store)
    assert DOC_A not in manifest.list_active_uris(corpus_id=CORPUS)
    store.close()


def test_enrichment_best_effort_policy_continues(tmp_path: Any) -> None:
    """Under 'best-effort' policy, a failing enricher does not stop ingestion."""
    store = SQLiteStore(db_path=str(tmp_path / "best.db"), vector_dim=16)
    orchestrator = EnrichmentOrchestrator(
        enricher=FakeFailingEnricher(), failure_policy="best-effort"
    )
    report = _make_engine(store, orchestrator).sync(corpus_id=CORPUS)

    assert report.status == SyncStatus.SUCCESS, (
        "Under 'best-effort' policy a failing enricher must be swallowed and ingestion succeed."
    )
    manifest = IndexManifest(store)
    assert DOC_A in manifest.list_active_uris(corpus_id=CORPUS)
    store.close()


def test_enrichment_success_promotes(tmp_path: Any) -> None:
    """Control: a working enricher does not interfere with promotion."""
    store = SQLiteStore(db_path=str(tmp_path / "ok.db"), vector_dim=16)
    orchestrator = EnrichmentOrchestrator(enricher=FakeEnricher(), failure_policy="raise")
    report = _make_engine(store, orchestrator).sync(corpus_id=CORPUS)
    assert report.status == SyncStatus.SUCCESS
    assert store.count_active_chunks(corpus_id=CORPUS) > 0
    store.close()
