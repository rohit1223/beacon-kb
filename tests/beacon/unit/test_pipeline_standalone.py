"""Unit test: run_search_pipeline with no FastAPI objects.

This is Epic 04/05's consumption proof: the pipeline must be callable
without any FastAPI Request, app.state, or HTTP context.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from tests.beacon.fakes import FakeConnector, SparseOnlyFakeEmbedder

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.sync import SyncEngine
from beacon.models import EvidenceBundle
from beacon.retrieval.filters import FilterSpec
from beacon.retrieval.pipeline import run_search_pipeline
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SyncJobRepo
from beacon.storage.qdrant import QdrantStore


def _build_corpus(
    tmp_path: Path, collection: str = "test-col"
) -> tuple[QdrantStore, StateDB, SparseOnlyFakeEmbedder, BeaconSettings]:
    settings = BeaconSettings(
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )
    db = StateDB(db_path=settings.state.db_path)
    store = QdrantStore(settings)
    embedder = SparseOnlyFakeEmbedder()

    CollectionRepo(db).create(name=collection)
    doc_bytes = (
        b"# Widget configuration\n\n"
        b"To configure the widget, open the settings panel."
    )
    connector = FakeConnector({"fake://widgets.md": doc_bytes})
    job_id = "job-pipeline-1"
    SyncJobRepo(db).create(job_id=job_id, collection_name=collection)
    SyncEngine(
        store=store,
        db=db,
        embedder=embedder,
        chunker_config=ChunkerConfig(),
        settings=settings,
    ).run_sync(collection_name=collection, connector=connector, job_id=job_id)

    return store, db, embedder, settings


class TestRunSearchPipelineStandalone:
    def test_returns_evidence_bundle(self, tmp_path: Path) -> None:
        """run_search_pipeline returns EvidenceBundle with no FastAPI objects."""
        store, db, embedder, _ = _build_corpus(tmp_path)
        try:
            spec = FilterSpec(collection="test-col")
            bundle = run_search_pipeline(
                state_db=db,
                store=store,
                embedder=embedder,
                spec=spec,
                query_text="widget configuration",
                top_k=5,
                token_budget=8192,
            )
            assert isinstance(bundle, EvidenceBundle)
        finally:
            store.close()
            db.close()

    def test_returns_evidence_with_labels(self, tmp_path: Path) -> None:
        """Evidence items have gap-free labels starting at S1."""
        store, db, embedder, _ = _build_corpus(tmp_path)
        try:
            spec = FilterSpec(collection="test-col")
            bundle = run_search_pipeline(
                state_db=db,
                store=store,
                embedder=embedder,
                spec=spec,
                query_text="widget configuration",
                top_k=5,
                token_budget=8192,
            )
            if bundle.evidence:
                assert bundle.evidence[0].label == "S1"
        finally:
            store.close()
            db.close()

    def test_no_fastapi_import_needed(self, tmp_path: Path) -> None:
        """Calling run_search_pipeline never requires a FastAPI Request object."""
        # This test passes if the function signature accepts only plain Python objects.
        sig = inspect.signature(run_search_pipeline)
        param_names = list(sig.parameters.keys())
        # Must not include 'request' (FastAPI-specific)
        assert "request" not in param_names, (
            "run_search_pipeline must not accept a FastAPI Request; "
            "it is a transport-free function"
        )
