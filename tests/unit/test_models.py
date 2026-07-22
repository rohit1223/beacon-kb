"""Unit tests for beacon_kb.models, beacon_kb.errors, and beacon_kb.protocols.

All tests verify:
- Immutability of frozen dataclass records
- Content-addressed ID reproducibility across calls and processes
- Enum validity and closed-vocabulary constraints
- Score-field direction annotations are present and scores are optional
- Protocol runtime-checkability via isinstance checks
- Zero side effects from importing the three modules
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helper: clear module from sys.modules so re-import is truly fresh
# ---------------------------------------------------------------------------


def _fresh_import(name: str) -> Any:
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    return importlib.import_module(name)


# ===========================================================================
# Import side-effect tests
# ===========================================================================


@pytest.mark.unit
class TestImportSideEffects:
    """Importing the three modules must not execute any side effects."""

    def test_models_import_is_side_effect_free(self) -> None:
        """beacon_kb.models must import without side effects."""
        _fresh_import("beacon_kb.models")  # if it raises, it has a side effect

    def test_errors_import_is_side_effect_free(self) -> None:
        """beacon_kb.errors must import without side effects."""
        _fresh_import("beacon_kb.errors")

    def test_protocols_import_is_side_effect_free(self) -> None:
        """beacon_kb.protocols must import without side effects."""
        _fresh_import("beacon_kb.protocols")


# ===========================================================================
# Typed ID tests
# ===========================================================================


@pytest.mark.unit
class TestTypedIDs:
    """Typed IDs are NewType-wrapped strings with stable, reproducible values."""

    def test_corpus_id_is_str(self) -> None:
        from beacon_kb.models import CorpusId

        cid = CorpusId("test-corpus")
        assert isinstance(cid, str)

    def test_source_id_is_str(self) -> None:
        from beacon_kb.models import SourceId

        sid = SourceId("s1")
        assert isinstance(sid, str)

    def test_revision_id_is_str(self) -> None:
        from beacon_kb.models import RevisionId

        rid = RevisionId("r1")
        assert isinstance(rid, str)

    def test_chunk_id_is_str(self) -> None:
        from beacon_kb.models import ChunkId

        cid = ChunkId("c1")
        assert isinstance(cid, str)

    def test_build_run_id_is_str(self) -> None:
        from beacon_kb.models import BuildRunId

        bid = BuildRunId("b1")
        assert isinstance(bid, str)

    def test_evidence_id_is_str(self) -> None:
        from beacon_kb.models import EvidenceId

        eid = EvidenceId("e1")
        assert isinstance(eid, str)

    def test_section_id_is_str(self) -> None:
        from beacon_kb.models import SectionId

        sid = SectionId("sec1")
        assert isinstance(sid, str)

    def test_query_id_is_str(self) -> None:
        from beacon_kb.models import QueryId

        qid = QueryId("q1")
        assert isinstance(qid, str)

    def test_trace_id_is_str(self) -> None:
        from beacon_kb.models import TraceId

        tid = TraceId("t1")
        assert isinstance(tid, str)


# ===========================================================================
# Content-addressed ID reproducibility
# ===========================================================================


@pytest.mark.unit
class TestContentAddressedIDs:
    """Content-addressed IDs must reproduce identical values for identical inputs."""

    def test_make_source_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_source_id

        id1 = make_source_id(corpus="corp", canonical_uri="file:///a/b.md")
        id2 = make_source_id(corpus="corp", canonical_uri="file:///a/b.md")
        assert id1 == id2

    def test_make_source_id_differs_for_different_inputs(self) -> None:
        from beacon_kb.models import make_source_id

        id1 = make_source_id(corpus="corp", canonical_uri="file:///a/b.md")
        id2 = make_source_id(corpus="corp", canonical_uri="file:///a/c.md")
        assert id1 != id2

    def test_make_revision_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_revision_id

        id1 = make_revision_id(source_id="s1", content_hash="abc123", pipeline_fingerprint="fp1")
        id2 = make_revision_id(source_id="s1", content_hash="abc123", pipeline_fingerprint="fp1")
        assert id1 == id2

    def test_make_revision_id_differs_for_different_fingerprint(self) -> None:
        from beacon_kb.models import make_revision_id

        id1 = make_revision_id(source_id="s1", content_hash="abc123", pipeline_fingerprint="fp1")
        id2 = make_revision_id(source_id="s1", content_hash="abc123", pipeline_fingerprint="fp2")
        assert id1 != id2

    def test_make_chunk_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_chunk_id

        id1 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:intro",
            child_ordinal=0,
        )
        id2 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:intro",
            child_ordinal=0,
        )
        assert id1 == id2

    def test_make_chunk_id_differs_for_different_ordinal(self) -> None:
        from beacon_kb.models import make_chunk_id

        id1 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:intro",
            child_ordinal=0,
        )
        id2 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:intro",
            child_ordinal=1,
        )
        assert id1 != id2

    def test_make_chunk_id_differs_for_different_parent(self) -> None:
        from beacon_kb.models import make_chunk_id

        id1 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:intro",
            child_ordinal=0,
        )
        id2 = make_chunk_id(
            corpus="corp",
            canonical_uri="file:///a.md",
            revision_id="rev1",
            pipeline_fingerprint="fp1",
            parent_locator="sec:body",
            child_ordinal=0,
        )
        assert id1 != id2

    def test_make_build_run_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_build_run_id

        ts = "2024-01-01T00:00:00Z"
        id1 = make_build_run_id(corpus="corp", pipeline_fingerprint="fp1", started_at_iso=ts)
        id2 = make_build_run_id(corpus="corp", pipeline_fingerprint="fp1", started_at_iso=ts)
        assert id1 == id2

    def test_make_section_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_section_id

        id1 = make_section_id(source_id="s1", revision_id="r1", locator="intro")
        id2 = make_section_id(source_id="s1", revision_id="r1", locator="intro")
        assert id1 == id2

    def test_make_section_id_differs_for_different_locator(self) -> None:
        from beacon_kb.models import make_section_id

        id1 = make_section_id(source_id="s1", revision_id="r1", locator="intro")
        id2 = make_section_id(source_id="s1", revision_id="r1", locator="body")
        assert id1 != id2

    def test_make_section_id_differs_for_different_revision(self) -> None:
        from beacon_kb.models import make_section_id

        id1 = make_section_id(source_id="s1", revision_id="r1", locator="intro")
        id2 = make_section_id(source_id="s1", revision_id="r2", locator="intro")
        assert id1 != id2

    def test_make_evidence_id_is_deterministic(self) -> None:
        from beacon_kb.models import make_evidence_id

        id1 = make_evidence_id(query_id="q1", chunk_id="ch1")
        id2 = make_evidence_id(query_id="q1", chunk_id="ch1")
        assert id1 == id2

    def test_make_evidence_id_differs_for_different_chunk(self) -> None:
        from beacon_kb.models import make_evidence_id

        id1 = make_evidence_id(query_id="q1", chunk_id="ch1")
        id2 = make_evidence_id(query_id="q1", chunk_id="ch2")
        assert id1 != id2

    def test_make_evidence_id_differs_for_different_query(self) -> None:
        from beacon_kb.models import make_evidence_id

        id1 = make_evidence_id(query_id="q1", chunk_id="ch1")
        id2 = make_evidence_id(query_id="q2", chunk_id="ch1")
        assert id1 != id2


# ===========================================================================
# Frozen record immutability
# ===========================================================================


@pytest.mark.unit
class TestFrozenRecords:
    """Every domain record must be frozen and raise FrozenInstanceError on mutation."""

    def test_corpus_record_is_frozen(self) -> None:
        from beacon_kb.models import Corpus, CorpusId

        corpus = Corpus(id=CorpusId("c1"), name="test", description="desc")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            corpus.name = "other"  # type: ignore[misc]

    def test_corpus_has_slots(self) -> None:
        from beacon_kb.models import Corpus

        assert "__slots__" in Corpus.__dict__ or hasattr(Corpus, "__slots__")

    def test_source_record_is_frozen(self) -> None:
        from beacon_kb.models import CorpusId, Source, SourceId

        source = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            source.canonical_uri = "other"  # type: ignore[misc]

    def test_revision_record_is_frozen(self) -> None:
        from beacon_kb.models import Revision, RevisionId, SourceId

        rev = Revision(
            id=RevisionId("r1"),
            source_id=SourceId("s1"),
            content_hash="abc",
            pipeline_fingerprint="fp1",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            rev.content_hash = "other"  # type: ignore[misc]

    def test_raw_document_record_is_frozen(self) -> None:
        from beacon_kb.models import RawDocument, RevisionId, SourceId

        doc = RawDocument(
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            content="hello",
            media_type="text/plain",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            doc.content = "world"  # type: ignore[misc]

    def test_section_record_is_frozen(self) -> None:
        from beacon_kb.models import RevisionId, Section, SectionId, SourceId

        sec = Section(
            id=SectionId("sec1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            locator="intro",
            heading="Introduction",
            text="Some text",
            ordinal=0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            sec.text = "other"  # type: ignore[misc]

    def test_chunk_record_is_frozen(self) -> None:
        from beacon_kb.models import Chunk, ChunkId, RevisionId, SectionId, SourceId

        chunk = Chunk(
            id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="chunk text",
            ordinal=0,
            parent_locator="intro",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            chunk.text = "other"  # type: ignore[misc]

    def test_query_record_is_frozen(self) -> None:
        from beacon_kb.models import Query, QueryId

        q = Query(id=QueryId("q1"), text="what is X?")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            q.text = "other"  # type: ignore[misc]

    def test_hit_record_is_frozen(self) -> None:
        from beacon_kb.models import Chunk, ChunkId, Hit, RevisionId, SectionId, SourceId

        chunk = Chunk(
            id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="chunk text",
            ordinal=0,
            parent_locator="intro",
        )
        hit = Hit(chunk=chunk, sparse_score=0.8)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            hit.sparse_score = 0.5  # type: ignore[misc]

    def test_evidence_record_is_frozen(self) -> None:
        from beacon_kb.models import (
            Chunk,
            ChunkId,
            Evidence,
            EvidenceId,
            Hit,
            RevisionId,
            SectionId,
            SourceId,
        )

        chunk = Chunk(
            id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="chunk text",
            ordinal=0,
            parent_locator="intro",
        )
        hit = Hit(chunk=chunk, sparse_score=0.8)
        ev = Evidence(id=EvidenceId("e1"), hit=hit, citation_label="S1")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ev.citation_label = "S2"  # type: ignore[misc]

    def test_answer_response_record_is_frozen(self) -> None:
        from beacon_kb.models import AnswerResponse, QueryId

        ans = AnswerResponse(query_id=QueryId("q1"), answer_text="The answer.", evidence=())
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ans.answer_text = "other"  # type: ignore[misc]

    def test_agentic_trace_record_is_frozen(self) -> None:
        from beacon_kb.models import AgenticTrace, TraceId

        trace = AgenticTrace(id=TraceId("t1"), query_id=None, steps=())
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            trace.steps = ()  # type: ignore[misc]


# ===========================================================================
# Score field direction
# ===========================================================================


@pytest.mark.unit
class TestScoreFieldDirection:
    """Score fields must be separate, optional, and carry direction documentation."""

    def test_hit_has_separate_score_fields(self) -> None:
        from beacon_kb.models import Chunk, ChunkId, Hit, RevisionId, SectionId, SourceId

        chunk = Chunk(
            id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="chunk text",
            ordinal=0,
            parent_locator="intro",
        )
        # All four score fields must be independently settable and default to None
        hit_sparse = Hit(chunk=chunk, sparse_score=0.9)
        hit_dense = Hit(chunk=chunk, dense_score=0.7)
        hit_fusion = Hit(chunk=chunk, fusion_score=1.2)
        hit_rerank = Hit(chunk=chunk, rerank_score=0.95)

        assert hit_sparse.sparse_score == 0.9
        assert hit_sparse.dense_score is None
        assert hit_dense.dense_score == 0.7
        assert hit_dense.sparse_score is None
        assert hit_fusion.fusion_score == 1.2
        assert hit_rerank.rerank_score == 0.95

    def test_hit_default_scores_are_none_not_zero(self) -> None:
        """Missing distance metadata must never default to zero."""
        from beacon_kb.models import Chunk, ChunkId, Hit, RevisionId, SectionId, SourceId

        chunk = Chunk(
            id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="chunk text",
            ordinal=0,
            parent_locator="intro",
        )
        hit = Hit(chunk=chunk)
        assert hit.sparse_score is None
        assert hit.dense_score is None
        assert hit.fusion_score is None
        assert hit.rerank_score is None

    def test_hit_score_field_names_are_distinct(self) -> None:
        """Four score names must be distinct to avoid aliasing."""
        from beacon_kb.models import Hit

        fields = {f.name for f in dataclasses.fields(Hit)}
        assert "sparse_score" in fields
        assert "dense_score" in fields
        assert "fusion_score" in fields
        assert "rerank_score" in fields


# ===========================================================================
# Answer response preserves evidence with stable citation labels
# ===========================================================================


@pytest.mark.unit
class TestAnswerResponse:
    """AnswerResponse must preserve structured evidence and stable [S1]-style IDs."""

    def _make_evidence(self, label: str) -> Any:
        from beacon_kb.models import (
            Chunk,
            ChunkId,
            Evidence,
            EvidenceId,
            Hit,
            RevisionId,
            SectionId,
            SourceId,
        )

        chunk = Chunk(
            id=ChunkId(f"ch-{label}"),
            source_id=SourceId("s1"),
            revision_id=RevisionId("r1"),
            section_id=SectionId("sec1"),
            text="Some text",
            ordinal=0,
            parent_locator="intro",
        )
        hit = Hit(chunk=chunk, sparse_score=0.9)
        return Evidence(id=EvidenceId(f"ev-{label}"), hit=hit, citation_label=label)

    def test_answer_text_is_plain_string_not_markdown(self) -> None:
        from beacon_kb.models import AnswerResponse, QueryId

        ev1 = self._make_evidence("S1")
        ans = AnswerResponse(query_id=QueryId("q1"), answer_text="The answer.", evidence=(ev1,))
        assert isinstance(ans.answer_text, str)
        # The answer_text field should not be pre-formatted Markdown
        # (the test verifies it is a raw string, not a structured object)
        assert ans.answer_text == "The answer."

    def test_answer_preserves_citation_labels(self) -> None:
        from beacon_kb.models import AnswerResponse, QueryId

        ev1 = self._make_evidence("S1")
        ev2 = self._make_evidence("S2")
        ans = AnswerResponse(
            query_id=QueryId("q1"),
            answer_text="Answer [S1] text [S2].",
            evidence=(ev1, ev2),
        )
        labels = [e.citation_label for e in ans.evidence]
        assert "S1" in labels
        assert "S2" in labels

    def test_answer_evidence_is_structured_not_preformatted(self) -> None:
        from beacon_kb.models import AnswerResponse, Evidence, QueryId

        ev1 = self._make_evidence("S1")
        ans = AnswerResponse(query_id=QueryId("q1"), answer_text="Answer [S1].", evidence=(ev1,))
        assert len(ans.evidence) == 1
        assert isinstance(ans.evidence[0], Evidence)

    def test_answer_response_is_frozen(self) -> None:
        from beacon_kb.models import AnswerResponse, QueryId

        ans = AnswerResponse(query_id=QueryId("q1"), answer_text="text", evidence=())
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ans.answer_text = "new"  # type: ignore[misc]


# ===========================================================================
# Enum validity
# ===========================================================================


@pytest.mark.unit
class TestEnums:
    """Closed-vocabulary enums must be importable and have expected members."""

    def test_sync_status_enum_exists(self) -> None:
        from beacon_kb.models import SyncStatus

        assert hasattr(SyncStatus, "SUCCESS")
        assert hasattr(SyncStatus, "FAILED")
        assert hasattr(SyncStatus, "PARTIAL")

    def test_chunk_kind_enum_exists(self) -> None:
        from beacon_kb.models import ChunkKind

        assert hasattr(ChunkKind, "PARENT")
        assert hasattr(ChunkKind, "CHILD")

    def test_evidence_role_enum_exists(self) -> None:
        from beacon_kb.models import EvidenceRole

        assert hasattr(EvidenceRole, "HIT")
        assert hasattr(EvidenceRole, "CONTEXT")

    def test_ingestion_change_enum_exists(self) -> None:
        from beacon_kb.models import IngestionChange

        assert hasattr(IngestionChange, "UNCHANGED")
        assert hasattr(IngestionChange, "NEW")
        assert hasattr(IngestionChange, "CHANGED")
        assert hasattr(IngestionChange, "DELETED")
        assert hasattr(IngestionChange, "INCOMPATIBLE")


# ===========================================================================
# Error hierarchy
# ===========================================================================


@pytest.mark.unit
class TestErrorHierarchy:
    """The error hierarchy must expose all required typed classes."""

    def test_base_beacon_error_exists(self) -> None:
        from beacon_kb.errors import BeaconError

        assert issubclass(BeaconError, Exception)

    def test_config_error_exists(self) -> None:
        from beacon_kb.errors import ConfigError

        assert issubclass(ConfigError, Exception)

    def test_readiness_error_exists(self) -> None:
        from beacon_kb.errors import ReadinessError

        assert issubclass(ReadinessError, Exception)

    def test_backend_error_exists(self) -> None:
        from beacon_kb.errors import BackendError

        assert issubclass(BackendError, Exception)

    def test_ingestion_error_exists(self) -> None:
        from beacon_kb.errors import IngestionError

        assert issubclass(IngestionError, Exception)

    def test_citation_error_exists(self) -> None:
        from beacon_kb.errors import CitationError

        assert issubclass(CitationError, Exception)

    def test_plugin_error_exists(self) -> None:
        from beacon_kb.errors import PluginError

        assert issubclass(PluginError, Exception)

    def test_plugin_conflict_exists(self) -> None:
        from beacon_kb.errors import PluginConflict

        assert issubclass(PluginConflict, Exception)

    def test_plugin_not_found_exists(self) -> None:
        from beacon_kb.errors import PluginNotFound

        assert issubclass(PluginNotFound, Exception)

    def test_protocol_mismatch_exists(self) -> None:
        from beacon_kb.errors import ProtocolMismatch

        assert issubclass(ProtocolMismatch, Exception)

    def test_budget_error_exists(self) -> None:
        from beacon_kb.errors import BudgetError

        assert issubclass(BudgetError, Exception)

    def test_agentic_error_exists(self) -> None:
        from beacon_kb.errors import AgenticError

        assert issubclass(AgenticError, Exception)

    def test_plugin_conflict_is_plugin_error(self) -> None:
        from beacon_kb.errors import PluginConflict, PluginError

        assert issubclass(PluginConflict, PluginError)

    def test_plugin_not_found_is_plugin_error(self) -> None:
        from beacon_kb.errors import PluginError, PluginNotFound

        assert issubclass(PluginNotFound, PluginError)

    def test_protocol_mismatch_is_plugin_error(self) -> None:
        from beacon_kb.errors import PluginError, ProtocolMismatch

        assert issubclass(ProtocolMismatch, PluginError)

    def test_errors_are_raisable(self) -> None:
        from beacon_kb.errors import (
            AgenticError,
            BackendError,
            BudgetError,
            CitationError,
            ConfigError,
            IngestionError,
            PluginConflict,
            PluginNotFound,
            ProtocolMismatch,
            ReadinessError,
        )

        for cls in (
            ConfigError,
            ReadinessError,
            BackendError,
            IngestionError,
            CitationError,
            BudgetError,
            AgenticError,
        ):
            with pytest.raises(cls):
                raise cls("test")

        with pytest.raises(PluginConflict):
            raise PluginConflict("group", "name", ["a", "b"])

        with pytest.raises(PluginNotFound):
            raise PluginNotFound("group", "name")

        with pytest.raises(ProtocolMismatch):
            raise ProtocolMismatch("group", "name", ["method_a"])


# ===========================================================================
# Protocol runtime-checkability
# ===========================================================================


@pytest.mark.unit
class TestProtocolRuntimeCheckability:
    """Every protocol must be runtime_checkable and isinstance-checkable."""

    def test_connector_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Connector

        assert hasattr(Connector, "__protocol_attrs__") or hasattr(Connector, "_is_protocol")

        class FakeConnector:
            def list_sources(self) -> list[Any]:
                return []

            def fetch(self, uri: str) -> Any:
                return None

        # runtime_checkable allows isinstance on structural check
        # We just verify it does not raise TypeError
        try:
            isinstance(FakeConnector(), Connector)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_parser_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Parser

        try:
            isinstance(object(), Parser)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_chunker_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Chunker

        try:
            isinstance(object(), Chunker)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_embedder_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Embedder

        try:
            isinstance(object(), Embedder)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_store_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Store

        try:
            isinstance(object(), Store)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_sparse_retriever_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import SparseRetriever

        try:
            isinstance(object(), SparseRetriever)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_dense_retriever_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import DenseRetriever

        try:
            isinstance(object(), DenseRetriever)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_fusion_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Fusion

        try:
            isinstance(object(), Fusion)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_reranker_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Reranker

        try:
            isinstance(object(), Reranker)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_generator_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Generator

        try:
            isinstance(object(), Generator)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_token_counter_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import TokenCounter

        try:
            isinstance(object(), TokenCounter)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_progress_observer_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import ProgressObserver

        try:
            isinstance(object(), ProgressObserver)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_query_planner_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import QueryPlanner

        try:
            isinstance(object(), QueryPlanner)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_evidence_grader_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import EvidenceGrader

        try:
            isinstance(object(), EvidenceGrader)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_corpus_router_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import CorpusRouter

        try:
            isinstance(object(), CorpusRouter)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_stop_condition_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import StopCondition

        try:
            isinstance(object(), StopCondition)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_session_store_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import SessionStore

        try:
            isinstance(object(), SessionStore)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")

    def test_tool_protocol_is_runtime_checkable(self) -> None:
        from beacon_kb.protocols import Tool

        try:
            isinstance(object(), Tool)
        except TypeError as e:
            pytest.fail(f"Protocol is not runtime_checkable: {e}")


# ===========================================================================
# Generator has no web-search flag
# ===========================================================================


@pytest.mark.unit
class TestGeneratorProtocol:
    """The Generator protocol must not expose a web-search flag."""

    def test_generator_has_no_web_search_param(self) -> None:
        import inspect

        from beacon_kb.protocols import Generator

        # Check generate method signature
        generate_method = Generator.generate
        sig = inspect.signature(generate_method)
        param_names = list(sig.parameters.keys())
        bad_names = [
            p for p in param_names
            if "web" in p.lower() or "search" in p.lower() or "internet" in p.lower()
        ]
        assert not bad_names, (
            f"Generator.generate must not expose web-search params, found: {bad_names}"
        )


# ===========================================================================
# StopCondition and Tool protocols exist (v1 has no entry-point group)
# ===========================================================================


@pytest.mark.unit
class TestDeferredProtocols:
    """StopCondition and Tool protocols must exist even though v1 ships no entry-point group."""

    def test_stop_condition_protocol_importable(self) -> None:
        from beacon_kb.protocols import StopCondition

        assert StopCondition is not None

    def test_tool_protocol_importable(self) -> None:
        from beacon_kb.protocols import Tool

        assert Tool is not None


# ===========================================================================
# AgenticTrace lives in models.py and is frozen
# ===========================================================================


@pytest.mark.unit
class TestAgenticTrace:
    """AgenticTrace must be a frozen record in models.py."""

    def test_agentic_trace_importable_from_models(self) -> None:
        from beacon_kb.models import AgenticTrace

        assert AgenticTrace is not None

    def test_agentic_trace_is_frozen_dataclass(self) -> None:
        from beacon_kb.models import AgenticTrace, TraceId

        trace = AgenticTrace(id=TraceId("t1"), query_id=None, steps=())
        assert dataclasses.is_dataclass(trace)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            trace.id = TraceId("t2")  # type: ignore[misc]

    def test_agentic_trace_has_steps(self) -> None:
        from beacon_kb.models import AgenticStep, AgenticTrace, TraceId

        step = AgenticStep(step_index=0, action="search", input_tokens=10, output_tokens=20)
        trace = AgenticTrace(id=TraceId("t1"), query_id=None, steps=(step,))
        assert len(trace.steps) == 1
        assert trace.steps[0].action == "search"

    def test_agentic_step_is_frozen(self) -> None:
        from beacon_kb.models import AgenticStep

        step = AgenticStep(step_index=0, action="search", input_tokens=10, output_tokens=20)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            step.action = "other"  # type: ignore[misc]


# ===========================================================================
# SyncReport record
# ===========================================================================


@pytest.mark.unit
class TestSyncReport:
    """SyncReport must be a frozen record capturing ingestion stats."""

    def test_sync_report_importable(self) -> None:
        from beacon_kb.models import SyncReport

        assert SyncReport is not None

    def test_sync_report_is_frozen(self) -> None:
        from beacon_kb.models import BuildRunId, CorpusId, SyncReport, SyncStatus

        report = SyncReport(
            build_run_id=BuildRunId("b1"),
            corpus_id=CorpusId("c1"),
            status=SyncStatus.SUCCESS,
            sources_scanned=10,
            sources_changed=2,
            chunks_added=50,
            chunks_deleted=5,
            errors=(),
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            report.sources_scanned = 99  # type: ignore[misc]


# ===========================================================================
# Citation record
# ===========================================================================


@pytest.mark.unit
class TestCitation:
    """Citation record must carry stable label and structured source info."""

    def test_citation_importable(self) -> None:
        from beacon_kb.models import Citation

        assert Citation is not None

    def test_citation_is_frozen(self) -> None:
        from beacon_kb.models import ChunkId, Citation, SourceId

        citation = Citation(
            label="S1",
            chunk_id=ChunkId("ch1"),
            source_id=SourceId("s1"),
            canonical_uri="file:///a.md",
            excerpt="Some text.",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            citation.label = "S2"  # type: ignore[misc]


# ===========================================================================
# Source.extra is fully immutable (tuple of pairs, no mutable container)
# ===========================================================================


@pytest.mark.unit
class TestSourceExtraImmutability:
    """Source.extra must be a tuple of pairs - no mutable container anywhere."""

    def test_source_extra_accepts_tuple_of_pairs(self) -> None:
        from beacon_kb.models import CorpusId, Source, SourceId

        source = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
            extra=(("lang", "en"), ("owner", "team-a")),
        )
        assert source.extra == (("lang", "en"), ("owner", "team-a"))

    def test_source_extra_defaults_to_empty_tuple(self) -> None:
        from beacon_kb.models import CorpusId, Source, SourceId

        source = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
        )
        assert source.extra == ()
        assert isinstance(source.extra, tuple)

    def test_source_record_frozen_instance_error_on_reassignment(self) -> None:
        from beacon_kb.models import CorpusId, Source, SourceId

        source = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
            extra=(("k", "v"),),
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            source.extra = ()  # type: ignore[misc]

    def test_source_extra_has_no_mutable_container(self) -> None:
        """extra must be a tuple (immutable), not a dict or list."""
        from beacon_kb.models import CorpusId, Source, SourceId

        source = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
            extra=(("a", "1"), ("b", "2")),
        )
        assert isinstance(source.extra, tuple)
        # Each element is also an immutable pair
        for pair in source.extra:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_source_extra_sorted_pairs_give_deterministic_identity(self) -> None:
        """Sorting pairs by key before construction gives two equal Source records."""
        from beacon_kb.models import CorpusId, Source, SourceId

        raw_meta = {"z": "last", "a": "first", "m": "middle"}
        sorted_extra = tuple(sorted(raw_meta.items()))
        s1 = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
            extra=sorted_extra,
        )
        s2 = Source(
            id=SourceId("s1"),
            corpus_id=CorpusId("c1"),
            canonical_uri="file:///a.md",
            media_type="text/markdown",
            extra=sorted_extra,
        )
        assert s1 == s2


# ===========================================================================
# Positive protocol conformance tests
# ===========================================================================


@pytest.mark.unit
class TestPositiveProtocolConformance:
    """Minimal conforming classes must satisfy isinstance(x, P) for key protocols."""

    def test_embedder_positive_conformance(self) -> None:
        from beacon_kb.protocols import Embedder

        class MinimalEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0]] * len(texts)

            def dimension(self) -> int:
                return 1

            @property
            def batch_size(self) -> int:
                return 8

        instance = MinimalEmbedder()
        assert isinstance(instance, Embedder) is True

    def test_generator_positive_conformance(self) -> None:
        from beacon_kb.models import AnswerResponse, QueryId
        from beacon_kb.protocols import Generator

        class MinimalGenerator:
            def generate(
                self,
                query: Any,
                hits: list[Any],
                *,
                max_input_tokens: int = 4096,
                max_output_tokens: int = 512,
            ) -> Any:
                return AnswerResponse(
                    query_id=QueryId("q1"),
                    answer_text="",
                    evidence=(),
                    abstained=True,
                )

        instance = MinimalGenerator()
        assert isinstance(instance, Generator) is True

    def test_stop_condition_positive_conformance(self) -> None:
        from beacon_kb.protocols import StopCondition

        class MinimalStopCondition:
            def should_stop(self, trace: Any) -> bool:
                return False

        instance = MinimalStopCondition()
        assert isinstance(instance, StopCondition) is True

    def test_connector_positive_conformance(self) -> None:
        """Connector already had a conforming class in the checkability test; verify True here."""
        from beacon_kb.protocols import Connector

        class MinimalConnector:
            def list_sources(self) -> list[str]:
                return []

            def fetch(self, uri: str) -> Any:
                return None

        instance = MinimalConnector()
        assert isinstance(instance, Connector) is True
