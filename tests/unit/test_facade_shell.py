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

from beacon_kb.errors import ReadinessError
from beacon_kb.models import (
    AnswerResponse,
    Query,
    QueryId,
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
    """search() raises ReadinessError when required components are missing."""

    def test_search_missing_store_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.search(query)
        assert "store" in str(exc_info.value).lower()

    def test_search_non_sqlite_store_raises_readiness_error(self) -> None:
        """search() requires a SQLiteStore, not any Store implementation."""
        # Inject a mock that passes the _require check but is not SQLiteStore.
        from unittest.mock import MagicMock

        from beacon_kb.facade import KnowledgeBase
        fake_store = MagicMock()
        kb = KnowledgeBase(store=fake_store)
        query = Query(id=QueryId("q1"), text="test")
        with pytest.raises(ReadinessError) as exc_info:
            kb.search(query)
        assert "SQLiteStore" in str(exc_info.value)

    def test_search_readiness_error_is_not_raised_when_store_present(self) -> None:
        """search() must not raise ReadinessError when a real SQLiteStore is injected."""
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        kb = KnowledgeBase(store=store)
        query = Query(id=QueryId("q1"), text="test")
        # Must not raise ReadinessError; empty store returns empty list.
        result = kb.search(query)
        assert isinstance(result, list)


# ===========================================================================
# answer() readiness
# ===========================================================================


@pytest.mark.unit
class TestAnswerReadiness:
    """answer() raises ReadinessError when generator is not injected."""

    def test_answer_missing_generator_raises_readiness_error(self) -> None:
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        kb = KnowledgeBase(store=store)  # no generator
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

    def _make_store(self) -> object:
        from beacon_kb.storage.sqlite import SQLiteStore
        return SQLiteStore(db_path=":memory:", vector_dim=16)

    def test_search_does_not_call_generator(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        store = self._make_store()
        generator = MagicMock()
        kb = KnowledgeBase(store=store, generator=generator)

        query = Query(id=QueryId("q1"), text="test")
        kb.search(query)

        generator.generate.assert_not_called()

    def test_search_returns_evidence_list(self) -> None:
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.models import Evidence

        store = self._make_store()
        kb = KnowledgeBase(store=store)

        query = Query(id=QueryId("q1"), text="test")
        results = kb.search(query)
        assert isinstance(results, list)
        # Empty store -> empty list (no crash)
        assert all(isinstance(item, Evidence) for item in results)


# ===========================================================================
# answer() cost contract: exactly one LLM call
# ===========================================================================


@pytest.mark.unit
class TestAnswerCostContract:
    """answer() makes at most one generator call (zero if abstention fires)."""

    def _make_store(self) -> object:
        from beacon_kb.storage.sqlite import SQLiteStore
        return SQLiteStore(db_path=":memory:", vector_dim=16)

    def test_answer_does_not_call_generator_more_than_once(self) -> None:
        from beacon_kb.facade import KnowledgeBase

        store = self._make_store()
        call_count = [0]

        class CountingGenerator:
            def generate(self, query, hits, *, max_input_tokens=4096, max_output_tokens=512):
                call_count[0] += 1
                return AnswerResponse(
                    query_id=query.id,
                    answer_text="The answer is 42.",
                    evidence=(),
                )

        kb = KnowledgeBase(store=store, generator=CountingGenerator())
        query = Query(id=QueryId("q1"), text="what is the answer?")
        response = kb.answer(query)

        assert call_count[0] <= 1, (
            f"answer() must call the generator at most once. Got {call_count[0]} calls."
        )
        assert isinstance(response, AnswerResponse)


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

    def test_health_search_not_ready_without_store(self) -> None:
        """search() requires a store; health() must report search_ready=False without one."""
        from beacon_kb.facade import KnowledgeBase

        kb = KnowledgeBase()  # no store
        result = kb.health()
        assert result["search_ready"] is False

    def test_health_search_ready_with_store_only(self) -> None:
        """Store-only KB -> search_ready=True (search() needs only the store)."""
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        kb = KnowledgeBase(store=store)
        result = kb.health()
        assert result["search_ready"] is True
        # answer_ready must be False without a generator.
        assert result["answer_ready"] is False

    def test_health_answer_not_ready_without_generator(self) -> None:
        """Store present but no generator -> answer_ready=False."""
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        kb = KnowledgeBase(store=store)  # no generator
        result = kb.health()
        assert result["answer_ready"] is False

    def test_health_status_ok_with_store_and_generator(self) -> None:
        """Store + generator -> status ok, answer_ready=True."""
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.storage.sqlite import SQLiteStore

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        generator = MagicMock()
        kb = KnowledgeBase(store=store, generator=generator)
        result = kb.health()
        assert result["status"] == "ok"
        assert result["answer_ready"] is True

    def test_health_no_store_reports_gap(self) -> None:
        """No store -> health reports search_ready=False (the gap)."""
        from beacon_kb.facade import KnowledgeBase

        generator = MagicMock()
        kb = KnowledgeBase(generator=generator)  # no store
        result = kb.health()
        assert result["search_ready"] is False
        assert result["answer_ready"] is False
        assert result["status"] == "degraded"


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


# ===========================================================================
# N2: injected components and config.retrieval.mode threading
# ===========================================================================


@pytest.mark.unit
class TestInjectedComponentsThreaded:
    """N2 fix: injected sparse/dense/fusion reach the RetrievalPipeline."""

    def test_injected_sparse_retriever_serves_search(self) -> None:
        """Custom injected SparseRetriever is actually called during search()."""
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.models import Hit, Query, QueryId
        from beacon_kb.storage.sqlite import SQLiteStore

        # A spy sparse retriever that records calls and returns an empty list.
        class SpySparseRetriever:
            def __init__(self) -> None:
                self.called = False

            def retrieve(self, query: Query) -> list[Hit]:
                self.called = True
                return []

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        spy = SpySparseRetriever()
        kb = KnowledgeBase(store=store, sparse_retriever=spy)

        query = Query(id=QueryId("q1"), text="test query")
        kb.search(query)

        assert spy.called, (
            "Injected sparse_retriever must be invoked during search(). "
            "If this fails, the facade is not threading the injected retriever "
            "into the RetrievalPipeline."
        )

    def test_mode_sparse_performs_no_embedder_calls(self) -> None:
        """config.retrieval.mode='sparse' -> no embedder calls during search()."""
        from beacon_kb.config import BeaconConfig, RetrievalConfig
        from beacon_kb.facade import KnowledgeBase
        from beacon_kb.models import Query, QueryId
        from beacon_kb.storage.sqlite import SQLiteStore

        class SpyEmbedder:
            def __init__(self) -> None:
                self.embed_calls = 0

            def embed(self, texts: list[str]) -> list[list[float]]:
                self.embed_calls += 1
                return [[0.0] * 16 for _ in texts]

            def dimension(self) -> int:
                return 16

        store = SQLiteStore(db_path=":memory:", vector_dim=16)
        spy_embedder = SpyEmbedder()
        config = BeaconConfig(retrieval=RetrievalConfig(mode="sparse"))
        kb = KnowledgeBase(store=store, embedder=spy_embedder, config=config)

        query = Query(id=QueryId("q1"), text="test")
        kb.search(query)

        assert spy_embedder.embed_calls == 0, (
            "config.retrieval.mode='sparse' must not call the embedder. "
            f"Got {spy_embedder.embed_calls} embed() calls."
        )
