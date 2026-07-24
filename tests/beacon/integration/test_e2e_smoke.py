"""Default-config end-to-end smoke over pure REST (Task 03.4).

The full product loop, driven exclusively through the REST API with default
settings and no credentials:

  1. POST /collections                  - create the collection
  2. POST /collections/{name}/sources   - attach a folder connector
  3. POST /collections/{name}/sync      - staged sync (background task)
  4. GET  /jobs/{job_id}                - job terminal and succeeded
  5. GET  /readyz                       - collection reports ready
  6. POST /search                       - cited evidence with provenance
  7. POST /answer                       - grounded, cited answer (fake LLM)

Embeddings run on the production sparse-only floor: the suite conftest sets
``HF_HUB_OFFLINE=1`` and the ``_sparse_only_floor`` fixture removes cloud API
keys, so the auto-detect ladder lands on SPARSE_ONLY for both the sync worker
and the query path - no fake embedder anywhere, no model downloads, no
network.  The only injected fake is the LLM client for the answer step, wired
through the ``app.state.llm_client`` seam.

Natural-language recall: the fixture corpus holds four short documents on
clearly distinct topics; questions are phrased conversationally (not keyword
echoes of whole sentences) and the expected source must appear in the top
results.  Thresholds stay loose (top-3) so the assertions are robust in
sparse-only mode.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from beacon.answer.generate import LlmResponse
from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.server.app import create_app

# Path to the fixture corpus.
FIXTURE_CORPUS = Path(__file__).parent.parent / "fixtures" / "corpus"

COLLECTION = "smoke-kb"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class CitingLlmClient:
    """Deterministic LLM fake that returns an answer citing [S1]."""

    call_count: int = field(default=0, init=False)
    canned_text: str = field(default="Based on the documentation [S1].")

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _sparse_only_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove cloud embedding keys so auto-detect lands on the sparse-only floor.

    Combined with ``HF_HUB_OFFLINE=1`` from the suite conftest, both the sync
    worker's and the query path's ``EmbedderProvider`` select SPARSE_ONLY:
    the same production vector space on both sides, with no credentials.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def smoke_client(
    tmp_path: Path, _sparse_only_floor: None
) -> Any:
    """Drive the corpus to READY through pure REST and yield (app, client).

    Executes journey steps 1-5 (create, attach, sync, job succeeded, ready)
    with assertions at every step, so every test using this fixture exercises
    the full REST ingestion path before searching or answering.
    """
    settings = BeaconSettings(
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )

    # Materialize the fixture corpus in a temp folder for the folder connector.
    corpus_dir = tmp_path / "corpus"
    shutil.copytree(str(FIXTURE_CORPUS), str(corpus_dir))

    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        # Step 1: create the collection.
        r = client.post("/collections", json={"name": COLLECTION})
        assert r.status_code == 201, r.text

        # Step 2: attach the folder source.
        r = client.post(
            f"/collections/{COLLECTION}/sources",
            json={
                "connector_kind": "folder",
                "config": {"root": str(corpus_dir), "include_globs": "**/*.md"},
            },
        )
        assert r.status_code == 201, r.text

        # Step 3: trigger the sync (TestClient waits for background tasks).
        r = client.post(f"/collections/{COLLECTION}/sync")
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        # Step 4: the job must be terminal and succeeded.
        job = client.get(f"/jobs/{job_id}").json()
        assert job["state"] == "succeeded", job
        assert job["sources_added"] == 4

        # Step 5: readiness reports the collection as ready.
        r = client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["collections"][COLLECTION] == "ready"

        yield app, client


def _top_source_uris(client: TestClient, query: str, *, top_k: int = 3) -> list[str]:
    """POST /search and return the source URIs of the top evidence items."""
    r = client.post(
        "/search",
        json={"collection": COLLECTION, "query": query, "top_k": top_k},
    )
    assert r.status_code == 200, r.text
    return [
        ev["snippet"]["source_uri"]
        for ev in r.json()["evidence"]
        if ev.get("snippet")
    ]


# ---------------------------------------------------------------------------
# The full product loop
# ---------------------------------------------------------------------------


class TestFullJourneySmoke:
    """Create -> attach -> sync -> ready -> search -> cited answer, pure REST."""

    def test_search_returns_cited_evidence(self, smoke_client: Any) -> None:
        """Step 6: /search returns labeled evidence with real provenance."""
        _, client = smoke_client
        r = client.post(
            "/search",
            json={
                "collection": COLLECTION,
                "query": "How does Python handle concurrency with the global interpreter lock?",
                "top_k": 5,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["evidence"], "Expected at least one evidence item"
        first = body["evidence"][0]
        assert first["label"] == "S1"
        assert first["role"] == "hit"
        assert first["score"] is not None
        snippet = first["snippet"]
        assert snippet["text"]
        assert snippet["source_uri"].startswith("file://")
        assert snippet["title"]

        recap = body["recap"]
        assert recap["requested"] >= recap["packed"] > 0
        assert recap["token_budget"] > 0

    def test_answer_returns_grounded_cited_answer(self, smoke_client: Any) -> None:
        """Step 7: /answer returns a grounded answer whose citation resolves
        to the expected source, with exactly one (fake) LLM call."""
        app, client = smoke_client
        llm = CitingLlmClient()
        app.state.llm_client = llm

        r = client.post(
            "/answer",
            json={
                "collection": COLLECTION,
                "query": "How does Python handle concurrency with the global interpreter lock?",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["abstained"] is False
        assert body["answer_text"] == "Based on the documentation [S1]."
        assert llm.call_count == 1

        # The [S1] citation must resolve, via the canonical server-held
        # bundle, to the expected source document.
        assert body["citations"], "Expected at least one resolved citation"
        citation = body["citations"][0]
        assert citation["label"] == "S1"
        assert citation["source_uri"].endswith("python.md")
        assert citation["chunk_id"] == body["evidence"]["evidence"][0]["chunk_id"]

    def test_answer_respects_search_evidence_budget_recap(self, smoke_client: Any) -> None:
        """The answer response carries the same bundle shape /search produces."""
        app, client = smoke_client
        app.state.llm_client = CitingLlmClient()

        r = client.post(
            "/answer",
            json={"collection": COLLECTION, "query": "What are Python decorators used for?"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "recap" in body["evidence"]
        assert body["diagnostics"]["evidence_count"] > 0
        assert body["diagnostics"]["abstained"] is False


# ---------------------------------------------------------------------------
# Natural-language recall over the fixture corpus (sparse-only mode)
# ---------------------------------------------------------------------------


class TestNaturalLanguageRecall:
    """Conversational questions retrieve the expected source in the top results."""

    def test_python_concurrency_question_finds_python_doc(self, smoke_client: Any) -> None:
        _, client = smoke_client
        uris = _top_source_uris(
            client,
            "How does Python handle concurrency when the global interpreter "
            "lock prevents threads from running in parallel?",
        )
        assert any(uri.endswith("python.md") for uri in uris), uris

    def test_database_durability_question_finds_databases_doc(self, smoke_client: Any) -> None:
        _, client = smoke_client
        uris = _top_source_uris(
            client,
            "What guarantees do transactions give so committed changes survive crashes?",
        )
        assert any(uri.endswith("databases.md") for uri in uris), uris

    def test_kubernetes_update_question_finds_kubernetes_doc(self, smoke_client: Any) -> None:
        _, client = smoke_client
        uris = _top_source_uris(
            client,
            "How can I update a running application on a cluster without any downtime?",
        )
        assert any(uri.endswith("kubernetes.md") for uri in uris), uris

    def test_ml_overfitting_question_finds_ml_doc(self, smoke_client: Any) -> None:
        _, client = smoke_client
        uris = _top_source_uris(
            client,
            "Why does my model memorize the training data and how do I reduce overfitting?",
        )
        assert any(uri.endswith("ml.md") for uri in uris), uris

    def test_answer_to_database_question_cites_databases_doc(self, smoke_client: Any) -> None:
        """Recall holds through the answer path: the citation resolves to the
        topically correct source document."""
        app, client = smoke_client
        app.state.llm_client = CitingLlmClient()

        r = client.post(
            "/answer",
            json={
                "collection": COLLECTION,
                "query": (
                    "What guarantees do transactions give so committed "
                    "changes survive crashes?"
                ),
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["abstained"] is False
        assert body["citations"]
        assert body["citations"][0]["source_uri"].endswith("databases.md")
