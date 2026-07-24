"""Integration tests for POST /search and POST /answer endpoints (Task 03.4).

Cost-contract tests use a counting fake LLM client injected via the app.state
seam.  The seam works because the lifespan initialises ``app.state.llm_client``
to None before yielding, and the routes check ``getattr(app.state, 'llm_client',
None)`` at request time.  Tests set ``app.state.llm_client`` inside the
``TestClient`` context (after lifespan startup) and the next request picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient
from tests.beacon.fakes import FakeConnector, FakeEmbedder, SparseOnlyFakeEmbedder

from beacon.answer.generate import LlmResponse
from beacon.config import BeaconSettings, QdrantSettings, ServerSettings, StateSettings
from beacon.ingest.chunking import ChunkerConfig
from beacon.ingest.sync import SyncEngine
from beacon.server.app import create_app
from beacon.state.db import StateDB
from beacon.state.repo import CollectionRepo, SyncJobRepo
from beacon.storage.qdrant import QdrantStore

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class CountingLlmClient:
    """LLM client that counts calls and returns a canned response."""

    call_count: int = field(default=0, init=False)
    canned_text: str = field(default="Based on the evidence [S1].")

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LlmResponse:
        self.call_count += 1
        return LlmResponse(text=self.canned_text, input_tokens=10, output_tokens=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Any, *, api_key: str | None = None) -> BeaconSettings:
    server_kw: dict[str, Any] = {}
    if api_key is not None:
        server_kw["api_key"] = api_key
    return BeaconSettings(
        server=ServerSettings(**server_kw),
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )


def _build_ready_corpus(
    settings: BeaconSettings,
    collection: str = "test-col",
    docs: dict[str, bytes] | None = None,
    embedder: Any = None,
) -> None:
    """Sync a corpus into the given collection so it reaches READY state."""
    if docs is None:
        docs = {
            "fake://widgets.md": (
                b"# Widget configuration\n\n"
                b"To configure the widget, open the settings panel."
            ),
        }
    if embedder is None:
        embedder = FakeEmbedder(dimension=8)

    db = StateDB(db_path=settings.state.db_path)
    store = QdrantStore(settings)
    try:
        CollectionRepo(db).create(name=collection)
        connector = FakeConnector(docs)
        job_id = f"job-{collection}-1"
        SyncJobRepo(db).create(job_id=job_id, collection_name=collection)
        engine = SyncEngine(
            store=store,
            db=db,
            embedder=embedder,
            chunker_config=ChunkerConfig(),
            settings=settings,
        )
        engine.run_sync(collection_name=collection, connector=connector, job_id=job_id)
    finally:
        db.close()
        store.close()


# ---------------------------------------------------------------------------
# TestSearchZeroLLMCalls
# ---------------------------------------------------------------------------


class TestSearchZeroLLMCalls:
    """POST /search must never invoke an LLM."""

    def test_search_performs_zero_llm_calls(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/search",
                json={"collection": "test-col", "query": "widget configuration"},
            )
        assert r.status_code == 200
        assert counting.call_count == 0, (
            f"Expected 0 LLM calls from /search, got {counting.call_count}"
        )


# ---------------------------------------------------------------------------
# TestAnswerExactlyOneCall
# ---------------------------------------------------------------------------


class TestAnswerExactlyOneCall:
    """POST /answer must invoke the LLM exactly once when evidence is available."""

    def test_answer_performs_exactly_one_llm_call(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget configuration"},
            )
        assert r.status_code == 200
        assert counting.call_count == 1, (
            f"Expected exactly 1 LLM call from /answer, got {counting.call_count}"
        )


# ---------------------------------------------------------------------------
# TestAnswerZeroCallsOnPreAbstention
# ---------------------------------------------------------------------------


class TestAnswerZeroCallsOnPreAbstention:
    """POST /answer with empty bundle -> 0 LLM calls, abstained=True, HTTP 200."""

    def test_answer_empty_bundle_abstains_with_zero_calls(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)

        # Build a READY corpus, then restrict the search to a source URI that
        # does not exist.  The boundary-enforced filter yields zero hits, the
        # bundle is empty, and the pre-abstention gate fires with zero LLM
        # calls - all through the real production pipeline, no patching.
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={
                    "collection": "test-col",
                    "query": "widget configuration",
                    "sources": ["fake://does-not-exist.md"],
                },
            )

        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is True
        assert counting.call_count == 0, (
            f"Expected 0 LLM calls on pre-abstention, got {counting.call_count}"
        )


# ---------------------------------------------------------------------------
# TestReadinessProblem
# ---------------------------------------------------------------------------


class TestReadinessProblem:
    """Search on a non-ready collection -> 503 problem+json with kind=readiness."""

    def test_search_on_unready_collection_returns_503(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        # Create collection in DB but never sync - it stays EMPTY.
        db = StateDB(db_path=settings.state.db_path)
        CollectionRepo(db).create(name="empty-col")
        db.close()

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/search",
                json={"collection": "empty-col", "query": "anything"},
            )

        assert r.status_code == 503
        body = r.json()
        assert body.get("kind") == "readiness"
        assert r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# TestAbstentionIsData
# ---------------------------------------------------------------------------


class TestAbstentionIsData:
    """Abstained answer -> HTTP 200 with abstained=true in the body."""

    def test_abstained_answer_returns_200(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        # Use a client that returns the ABSTAIN sentinel.
        from beacon.answer.abstention import ABSTAIN_SENTINEL

        abstaining_client = CountingLlmClient(canned_text=ABSTAIN_SENTINEL)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = abstaining_client
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget configuration"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is True


# ---------------------------------------------------------------------------
# TestSearchResponse
# ---------------------------------------------------------------------------


class TestSearchResponse:
    """POST /search returns evidence bundle with expected structure."""

    def test_search_returns_evidence_bundle(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/search",
                json={"collection": "test-col", "query": "widget configuration", "top_k": 5},
            )

        assert r.status_code == 200
        body = r.json()
        assert "evidence" in body
        assert "recap" in body
        recap = body["recap"]
        assert "requested" in recap
        assert "packed" in recap
        assert "token_budget" in recap

    def test_search_evidence_has_labels_and_snippets(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/search",
                json={"collection": "test-col", "query": "widget configuration"},
            )

        body = r.json()
        assert body["evidence"], "Expected at least one evidence item for a matching query"
        ev = body["evidence"][0]
        assert ev["label"] == "S1"
        assert ev["role"] == "hit"
        assert ev["chunk_id"]
        snippet = ev["snippet"]
        assert snippet is not None
        assert snippet["source_uri"] == "fake://widgets.md"
        assert snippet["text"]


# ---------------------------------------------------------------------------
# TestAnswerCitesSources
# ---------------------------------------------------------------------------


class TestAnswerCitesSources:
    """POST /answer returns answer_text, citations, and evidence."""

    def test_answer_result_has_expected_fields(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient(canned_text="The widget settings are in S1 [S1].")
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget configuration"},
            )

        assert r.status_code == 200
        body = r.json()
        assert "answer_text" in body
        assert "citations" in body
        assert "evidence" in body
        assert "abstained" in body
        assert "diagnostics" in body


# ---------------------------------------------------------------------------
# TestAuthCoverage
# ---------------------------------------------------------------------------


class TestAuthCoverage:
    """When api_key is set, /search and /answer require auth."""

    def test_search_requires_auth_when_key_set(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path, api_key="my-secret")
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(
                "/search",
                json={"collection": "test-col", "query": "widget"},
            )
        assert r.status_code == 401

    def test_search_passes_with_correct_auth(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path, api_key="my-secret")
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/search",
                json={"collection": "test-col", "query": "widget"},
                headers={"Authorization": "Bearer my-secret"},
            )
        # The corpus is READY, so an authorized search must succeed outright.
        assert r.status_code == 200

    def test_answer_requires_auth_when_key_set(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path, api_key="my-secret")
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget"},
            )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# TestAnswerReadinessProblem
# ---------------------------------------------------------------------------


class TestAnswerReadinessProblem:
    """Answer on a non-ready collection -> 503 problem+json with kind=readiness."""

    def test_answer_on_unready_collection_returns_503(self, tmp_path: Any) -> None:
        settings = _settings(tmp_path)
        # Create collection in DB but never sync - it stays EMPTY.
        db = StateDB(db_path=settings.state.db_path)
        CollectionRepo(db).create(name="empty-col")
        db.close()

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "empty-col", "query": "anything"},
            )

        assert r.status_code == 503
        body = r.json()
        assert body.get("kind") == "readiness"
        assert r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# TestAbstentionReason
# ---------------------------------------------------------------------------


class TestAbstentionReason:
    """AnswerResult.reason is populated distinctly for pre- and post-abstention."""

    def test_pre_abstention_reason_string(self, tmp_path: Any) -> None:
        """Pre-abstention via empty bundle sets reason and carries a non-null evidence bundle."""
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient()
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={
                    "collection": "test-col",
                    "query": "widget configuration",
                    "sources": ["fake://does-not-exist.md"],
                },
            )

        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is True
        assert body["reason"] == "pre_abstention: no evidence above threshold"
        # Evidence bundle is present (though empty) on pre-abstention.
        assert body["evidence"] is not None

    def test_post_abstention_reason_string(self, tmp_path: Any) -> None:
        """Post-abstention via model sentinel sets reason and carries a non-null evidence bundle."""
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        from beacon.answer.abstention import ABSTAIN_SENTINEL

        app = create_app(settings)
        abstaining_client = CountingLlmClient(canned_text=ABSTAIN_SENTINEL)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = abstaining_client
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget configuration"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is True
        assert body["reason"] == "post_abstention: model declined"
        # Evidence bundle is non-null: post-abstention has real evidence hits.
        assert body["evidence"] is not None
        assert body["evidence"]["evidence"]  # at least one item

    def test_normal_answer_reason_is_none(self, tmp_path: Any) -> None:
        """A non-abstained answer carries reason=None."""
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        counting = CountingLlmClient(canned_text="The widget settings are in S1 [S1].")
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.llm_client = counting
            app.state.embedder = SparseOnlyFakeEmbedder()
            r = c.post(
                "/answer",
                json={"collection": "test-col", "query": "widget configuration"},
            )

        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is False
        assert body["reason"] is None


# ---------------------------------------------------------------------------
# TestDateRangeFilter
# ---------------------------------------------------------------------------


class TestDateRangeFilter:
    """Date-range filter wiring: passing modified date range through /search."""

    def test_search_with_future_ingested_date_returns_empty_evidence(
        self, tmp_path: Any
    ) -> None:
        """An ingested date filter with a future lower bound returns no evidence."""
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            # 2099-01-01 is in the future - no documents were ingested then.
            r = c.post(
                "/search",
                json={
                    "collection": "test-col",
                    "query": "widget configuration",
                    "ingested": {"gte": "2099-01-01T00:00:00Z"},
                },
            )

        assert r.status_code == 200
        body = r.json()
        # The future date filter should exclude all documents.
        assert body["evidence"] == []

    def test_search_with_past_ingested_date_returns_evidence(
        self, tmp_path: Any
    ) -> None:
        """An ingested date filter with a past lower bound still returns evidence."""
        settings = _settings(tmp_path)
        _build_ready_corpus(settings)

        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.embedder = SparseOnlyFakeEmbedder()
            # 2000-01-01 is well in the past - all documents pass the ingested filter.
            r = c.post(
                "/search",
                json={
                    "collection": "test-col",
                    "query": "widget configuration",
                    "ingested": {"gte": "2000-01-01T00:00:00Z"},
                },
            )

        assert r.status_code == 200
        body = r.json()
        assert body["evidence"]  # at least one result
