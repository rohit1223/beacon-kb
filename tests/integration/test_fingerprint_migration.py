"""Integration tests for fingerprint migration and pipeline incompatibility.

Verifies that changing any pipeline parameter (parser version, chunker params,
enrichment version, embedder model, embedding dimension, schema version)
triggers INCOMPATIBLE classification and a full re-ingest.
"""

from __future__ import annotations

from typing import Any

from beacon_kb.connectors.memory import MemoryConnector
from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.ingestion.planning import build_pipeline_fingerprint
from beacon_kb.ingestion.sync import SyncEngine
from beacon_kb.models import CorpusId, SyncStatus
from beacon_kb.parsing.markdown import MarkdownParser
from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.testing import FakeEmbedder

CORPUS = CorpusId("test-corpus")
DOC_A = "memory://doc-a"
CONTENT_A = "# Test\n\nContent for fingerprint migration tests. Enough text for chunking."


def _make_store(tmp_path: Any, vector_dim: int = 16) -> SQLiteStore:
    db = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db, vector_dim=vector_dim)


def _make_engine(
    store: SQLiteStore,
    sources: dict[str, str],
    *,
    corpus: str = "test-corpus",
    dim: int = 16,
    parser_version: str = "markdown-v1",
    chunker_params: dict[str, Any] | None = None,
    enrichment_version: str = "",
    embedder_model_override: str | None = None,
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

    engine = SyncEngine(
        store=store,
        connector=connector,
        parser=parser,
        chunker_factory=chunker_factory,
        embedder=embedder,
        corpus_name=corpus,
        parser_version=parser_version,
        chunker_params=chunker_params or {},
        enrichment_version=enrichment_version,
    )
    if embedder_model_override is not None:
        engine._embedder_model_name = lambda: embedder_model_override  # type: ignore[method-assign]
    return engine


# ---------------------------------------------------------------------------
# Test: fingerprint is deterministic
# ---------------------------------------------------------------------------


def test_pipeline_fingerprint_deterministic() -> None:
    """build_pipeline_fingerprint returns the same value for identical inputs."""
    fp1 = build_pipeline_fingerprint(
        parser_version="md-v1",
        chunker_params={"max_tokens": 512, "overlap_tokens": 64},
        enrichment_version="",
        embedder_model="fake-v1",
        embedder_dimension=16,
        schema_version=2,
    )
    fp2 = build_pipeline_fingerprint(
        parser_version="md-v1",
        chunker_params={"max_tokens": 512, "overlap_tokens": 64},
        enrichment_version="",
        embedder_model="fake-v1",
        embedder_dimension=16,
        schema_version=2,
    )
    assert fp1 == fp2


def test_pipeline_fingerprint_changes_with_parser_version() -> None:
    """Fingerprint changes when parser_version changes."""
    fp1 = build_pipeline_fingerprint(parser_version="v1")
    fp2 = build_pipeline_fingerprint(parser_version="v2")
    assert fp1 != fp2


def test_pipeline_fingerprint_changes_with_chunker_params() -> None:
    """Fingerprint changes when chunker_params change."""
    fp1 = build_pipeline_fingerprint(chunker_params={"max_tokens": 512})
    fp2 = build_pipeline_fingerprint(chunker_params={"max_tokens": 256})
    assert fp1 != fp2


def test_pipeline_fingerprint_changes_with_embedder_model() -> None:
    """Fingerprint changes when embedder_model changes."""
    fp1 = build_pipeline_fingerprint(embedder_model="model-a")
    fp2 = build_pipeline_fingerprint(embedder_model="model-b")
    assert fp1 != fp2


def test_pipeline_fingerprint_changes_with_dimension() -> None:
    """Fingerprint changes when embedder_dimension changes."""
    fp1 = build_pipeline_fingerprint(embedder_dimension=16)
    fp2 = build_pipeline_fingerprint(embedder_dimension=32)
    assert fp1 != fp2


def test_pipeline_fingerprint_changes_with_schema_version() -> None:
    """Fingerprint changes when schema_version changes."""
    fp1 = build_pipeline_fingerprint(schema_version=1)
    fp2 = build_pipeline_fingerprint(schema_version=2)
    assert fp1 != fp2


def test_pipeline_fingerprint_includes_enrichment_version() -> None:
    """Fingerprint changes when enrichment_version changes."""
    fp1 = build_pipeline_fingerprint(enrichment_version="")
    fp2 = build_pipeline_fingerprint(enrichment_version="enrich-v2")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# Test: parser version change triggers INCOMPATIBLE
# ---------------------------------------------------------------------------


def test_parser_version_change_triggers_incompatible(tmp_path: Any) -> None:
    """Changing parser_version between syncs causes INCOMPATIBLE re-ingest."""
    store = _make_store(tmp_path)

    # First sync with parser-v1.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A}, parser_version="markdown-v1")
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.status == SyncStatus.SUCCESS
    assert report1.sources_changed == 1

    rev_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)

    # Second sync with parser-v2: must trigger INCOMPATIBLE.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A}, parser_version="markdown-v2")
    report2 = engine2.sync(corpus_id=CORPUS)
    assert report2.status == SyncStatus.SUCCESS
    assert report2.sources_changed == 1

    rev_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_after != rev_before, "Parser version change must produce a new revision."
    store.close()


# ---------------------------------------------------------------------------
# Test: chunker param change triggers INCOMPATIBLE
# ---------------------------------------------------------------------------


def test_chunker_param_change_triggers_incompatible(tmp_path: Any) -> None:
    """Changing chunker_params between syncs causes INCOMPATIBLE re-ingest."""
    store = _make_store(tmp_path)

    engine1 = _make_engine(store, {DOC_A: CONTENT_A}, chunker_params={"max_tokens": 512})
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    rev_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)

    engine2 = _make_engine(store, {DOC_A: CONTENT_A}, chunker_params={"max_tokens": 256})
    report2 = engine2.sync(corpus_id=CORPUS)
    assert report2.sources_changed == 1

    rev_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_after != rev_before
    store.close()


# ---------------------------------------------------------------------------
# Test: embedder model change triggers INCOMPATIBLE
# ---------------------------------------------------------------------------


def test_embedder_model_change_triggers_incompatible(tmp_path: Any) -> None:
    """Changing the embedder model between syncs causes INCOMPATIBLE re-ingest."""
    store = _make_store(tmp_path)

    engine1 = _make_engine(store, {DOC_A: CONTENT_A}, embedder_model_override="model-a")
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    rev_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)

    engine2 = _make_engine(store, {DOC_A: CONTENT_A}, embedder_model_override="model-b")
    report2 = engine2.sync(corpus_id=CORPUS)
    assert report2.sources_changed == 1

    rev_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_after != rev_before
    store.close()


# ---------------------------------------------------------------------------
# Test: fingerprints compared on every sync (not only on content change)
# ---------------------------------------------------------------------------


def test_fingerprints_compared_on_every_sync(tmp_path: Any) -> None:
    """Even when content is unchanged, a pipeline change triggers re-ingest."""
    store = _make_store(tmp_path)

    # First sync.
    engine1 = _make_engine(store, {DOC_A: CONTENT_A}, parser_version="v1")
    report1 = engine1.sync(corpus_id=CORPUS)
    assert report1.sources_changed == 1

    rev_before = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)

    # Second sync: SAME content, DIFFERENT parser version.
    # Content hash is unchanged but pipeline fingerprint changed.
    engine2 = _make_engine(store, {DOC_A: CONTENT_A}, parser_version="v2")
    report2 = engine2.sync(corpus_id=CORPUS)
    # Must re-ingest even though content is identical.
    assert report2.sources_changed >= 1

    rev_after = store.get_active_revision_id(corpus_id=CORPUS, canonical_uri=DOC_A)
    assert rev_after != rev_before, (
        "Fingerprint comparison must happen on every sync, not only on content change."
    )
    store.close()
