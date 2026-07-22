"""Unit tests for beacon_kb.facade.KnowledgeBase shell.

Tests cover:
- Importing beacon_kb.facade does NOT import beacon_kb.agentic.
- Constructing KnowledgeBase does NOT import beacon_kb.agentic.
- investigate() raises ModuleNotFoundError (beacon_kb.agentic not yet implemented).
- Only calling investigate() triggers the lazy import attempt.
- search() raises ReadinessError when retrievers are not injected.
- answer() raises ReadinessError when generator is not injected.
- inspect() returns expected keys.
- health() returns 'degraded' when components are missing.
- health() returns expected structure.
- KnowledgeBase accepts injected components and composes them.
- Cost contracts: search() makes zero generator calls, answer() makes one.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from beacon_kb.config import BeaconConfig, RetrievalConfig
from beacon_kb.errors import ReadinessError
from beacon_kb.models import (
    AnswerResponse,
    Chunk,
    ChunkId,
    Hit,
    Query,
    QueryId,
    RevisionId,
    SectionId,
    SourceId,
)
from beacon_kb.version import PLUGIN_API_VERSION, __version__

# ===========================================================================
# Lazy import: facade must not import beacon_kb.agentic at module level
# ===========================================================================


@pytest.mark.unit
class TestLazyAgenticImport:
    """importing facade and constructing KnowledgeBase must not import beacon_kb.agentic."""

    def test_import_facade_does_not_import_agentic(self) -> None:
        """Import beacon_kb.facade; agentic must NOT be in sys.modules."""
        # Ensure beacon_kb.agentic is not already present
        for key in list(sys.modules):
            if key == "beacon_kb.agentic" or key.startswith("beacon_kb.agentic."):
                del sys.modules[key]

        import beacon_kb.facade  # noqa: F401 - side-effect test

        assert "beacon_kb.agentic" not in sys.modules

    def test_construct_knowledge_base_does_not_import_agentic(self) -> None:
        """Constructing KnowledgeBase must not import beacon_kb.agentic."""
        # Clear agentic from sys.modules
        for key in list(sys.modules):
            if key == "beacon_kb.agentic" or key.startswith("beacon_kb.agentic."):
                del sys.modules[key]

        from beacon_kb.facade import KnowledgeBase

        _kb = KnowledgeBase()
        assert "beacon_kb.agentic" not in sys.modules

    def test_investigate_raises_module_not_found_error(self) -> None:
        """investigate() must raise ModuleNotFoundError (agentic not yet implemented)."""
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        query = Query(id=QueryId("q1"), text="test query")
        with pytest.raises(ModuleNotFoundError):
            kb.investigate(query)

    def test_investigate_triggers_agentic_import_attempt(self) -> None:
        """Only investigate() attempts to import beacon_kb.agentic."""
        from beacon_kb.facade import KnowledgeBase

        # Clear agentic from sys.modules to ensure clean state
        for key in list(sys.modules):
            if key == "beacon_kb.agentic" or key.startswith("beacon_kb.agentic."):
                del sys.modules[key]

        kb = KnowledgeBase()
        assert "beacon_kb.agentic" not in sys.modules

        query = Query(id=QueryId("q1"), text="test")
        # This call must attempt the import and fail (agentic doesn't exist yet)
        with pytest.raises(ModuleNotFoundError):
            kb.investigate(query)


# ===========================================================================
# search() readiness
# ===========================================================================


@pytest.mark.unit
class TestSearchReadiness:
    """search() raises ReadinessError when required retrievers are missing."""

    def test_search_hybrid_missing_sparse_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        # hybrid mode requires sparse_retriever
        kb = KnowledgeBase()
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.search(query)
        assert "sparse_retriever" in str(exc_info.value)

    def test_search_dense_missing_dense_retriever_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        config = BeaconConfig(retrieval=RetrievalConfig(mode="dense"))
        kb = KnowledgeBase(config=config)
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.search(query)
        assert "dense_retriever" in str(exc_info.value)

    def test_search_sparse_mode_missing_sparse_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(config=config)
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.search(query)
        assert "sparse_retriever" in str(exc_info.value)


# ===========================================================================
# answer() readiness
# ===========================================================================


@pytest.mark.unit
class TestAnswerReadiness:
    """answer() raises ReadinessError when generator is not injected."""

    def test_answer_missing_generator_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        # Provide a sparse retriever so search() passes, but no generator
        sparse = MagicMock()
        sparse.retrieve.return_value = []
        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse)
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.answer(query)
        assert "generator" in str(exc_info.value)


# ===========================================================================
# search() cost contract: zero LLM calls
# ===========================================================================


@pytest.mark.unit
class TestSearchCostContract:
    """search() makes zero generator / LLM calls."""

    def _make_hit(self) -> Hit:
        chunk = Chunk(
            id=ChunkId("chunk1"),
            source_id=SourceId("src1"),
            revision_id=RevisionId("rev1"),
            section_id=SectionId("sec1"),
            text="test chunk",
            ordinal=0,
            parent_locator="intro",
        )
        return Hit(chunk=chunk, sparse_score=1.0)

    def test_search_does_not_call_generator(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        sparse.retrieve.return_value = [self._make_hit()]
        generator = MagicMock()
        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse, generator=generator)

        query = Query(id=QueryId("q1"), text="test")
        kb.search(query)

        generator.generate.assert_not_called()

    def test_search_returns_hits_up_to_top_k(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        hits = [self._make_hit() for _ in range(20)]
        sparse = MagicMock()
        sparse.retrieve.return_value = hits
        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse", top_k=5))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse)

        query = Query(id=QueryId("q1"), text="test")
        results = kb.search(query)
        assert len(results) <= 5


# ===========================================================================
# answer() cost contract: exactly one LLM call
# ===========================================================================


@pytest.mark.unit
class TestAnswerCostContract:
    """answer() makes exactly one generator call."""

    def _make_answer_response(self, query_id: str) -> AnswerResponse:
        return AnswerResponse(
            query_id=QueryId(query_id),
            answer_text="The answer is 42.",
            evidence=(),
        )

    def test_answer_calls_generator_exactly_once(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        sparse.retrieve.return_value = []
        generator = MagicMock()
        generator.generate.return_value = self._make_answer_response("q1")

        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse, generator=generator)

        query = Query(id=QueryId("q1"), text="what is the answer?")
        response = kb.answer(query)

        generator.generate.assert_called_once()
        assert response.answer_text == "The answer is 42."


# ===========================================================================
# inspect()
# ===========================================================================


@pytest.mark.unit
class TestInspect:
    """inspect() returns a structured config/component/version snapshot."""

    def test_inspect_returns_version_key(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.inspect()
        assert "version" in result
        assert result["version"] == __version__

    def test_inspect_returns_plugin_api_version(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.inspect()
        assert "plugin_api_version" in result
        assert result["plugin_api_version"] == PLUGIN_API_VERSION

    def test_inspect_returns_config_key(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.inspect()
        assert "config" in result
        assert "core" in result["config"]
        assert "retrieval" in result["config"]
        assert "answer" in result["config"]
        assert "agentic" in result["config"]

    def test_inspect_returns_components_key(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.inspect()
        assert "components" in result
        comps = result["components"]
        for expected in ("connector", "generator", "token_counter"):
            assert expected in comps

    def test_inspect_shows_none_for_missing_components(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.inspect()
        assert result["components"]["generator"] is None

    def test_inspect_shows_type_name_for_present_components(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        kb = KnowledgeBase(sparse_retriever=sparse)
        result = kb.inspect()
        assert result["components"]["sparse_retriever"] is not None


# ===========================================================================
# health()
# ===========================================================================


@pytest.mark.unit
class TestHealth:
    """health() returns readiness state for each component."""

    def test_health_degraded_when_no_components(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.health()
        assert result["status"] == "degraded"

    def test_health_returns_components_dict(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.health()
        assert "components" in result
        assert isinstance(result["components"], dict)

    def test_health_search_not_ready_without_retrievers(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        result = kb.health()
        assert result["search_ready"] is False

    def test_health_answer_not_ready_without_generator(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        dense = MagicMock()
        config = BeaconConfig(retrieval=RetrievalConfig(mode="hybrid"))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse, dense_retriever=dense)
        result = kb.health()
        assert result["answer_ready"] is False

    def test_health_status_ok_with_all_required_components(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        dense = MagicMock()
        generator = MagicMock()
        config = BeaconConfig(retrieval=RetrievalConfig(mode="hybrid"))
        kb = KnowledgeBase(
            config=config,
            sparse_retriever=sparse,
            dense_retriever=dense,
            generator=generator,
        )
        result = kb.health()
        assert result["status"] == "ok"
        assert result["answer_ready"] is True

    def test_health_sparse_mode_ok_without_dense(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        sparse = MagicMock()
        generator = MagicMock()
        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(config=config, sparse_retriever=sparse, generator=generator)
        result = kb.health()
        assert result["status"] == "ok"


# ===========================================================================
# No provider import at construction
# ===========================================================================


@pytest.mark.unit
class TestNoProviderImportAtConstruction:
    """Constructing KnowledgeBase must not import any provider module."""

    def test_construct_does_not_import_beacon_agentic(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        for key in list(sys.modules):
            if key == "beacon_kb.agentic" or key.startswith("beacon_kb.agentic."):
                del sys.modules[key]

        _kb = KnowledgeBase()
        assert "beacon_kb.agentic" not in sys.modules

    def test_import_beacon_kb_does_not_import_agentic(self) -> None:
        import beacon_kb  # noqa: F401

        assert "beacon_kb.agentic" not in sys.modules
