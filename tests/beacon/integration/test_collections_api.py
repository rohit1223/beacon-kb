"""Integration tests for the collections REST resource (Task 01.5).

Covers:
- POST /collections: create, name validation (422), duplicate rejection (409).
- GET /collections: list with per-collection corpus state.
- GET /collections/{name}: detail with revision and last-job summary.
- No Qdrant write on collection creation (state-DB-only registration).
- Smoke test: embedded Qdrant + tmp state DB -> create -> list -> readyz.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from beacon.config import BeaconSettings, QdrantSettings, StateSettings
from beacon.server.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Any) -> BeaconSettings:
    """Build test settings backed by a tmp_path SQLite DB and embedded Qdrant."""
    return BeaconSettings(
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )


def _client(settings: BeaconSettings) -> TestClient:
    """Build a TestClient for *settings*."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /collections
# ---------------------------------------------------------------------------


class TestPostCollections:
    """POST /collections creates a collection and enforces name/duplicate rules."""

    def test_create_returns_201(self, tmp_path: Any) -> None:
        """A valid name returns 201 Created."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "my-docs"})
        assert r.status_code == 201

    def test_create_body_contains_name(self, tmp_path: Any) -> None:
        """Response body includes the collection name."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "my-docs"})
        body = r.json()
        assert body["name"] == "my-docs"

    def test_create_body_contains_corpus_state(self, tmp_path: Any) -> None:
        """Newly created collection reports corpus_state = 'empty'."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "my-docs"})
        body = r.json()
        assert body["corpus_state"] == "empty"

    def test_create_body_contains_created_at(self, tmp_path: Any) -> None:
        """Response body includes created_at timestamp string."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "my-docs"})
        body = r.json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_create_invalid_name_422(self, tmp_path: Any) -> None:
        """An invalid name (uppercase, spaces, etc.) returns 422 problem+json."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "Invalid Name!"})
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_create_name_too_long_422(self, tmp_path: Any) -> None:
        """A name exceeding the max length returns 422."""
        s = _settings(tmp_path)
        long_name = "a" * 65
        with _client(s) as c:
            r = c.post("/collections", json={"name": long_name})
        assert r.status_code == 422

    def test_create_empty_name_422(self, tmp_path: Any) -> None:
        """An empty name returns 422."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": ""})
        assert r.status_code == 422

    def test_create_name_with_dashes_and_underscores_valid(self, tmp_path: Any) -> None:
        """Names with dashes and underscores are valid."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": "my_docs-v2"})
        assert r.status_code == 201

    def test_create_duplicate_409(self, tmp_path: Any) -> None:
        """Creating the same collection twice returns 409 on the second attempt."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r1 = c.post("/collections", json={"name": "my-docs"})
            assert r1.status_code == 201
            r2 = c.post("/collections", json={"name": "my-docs"})
        assert r2.status_code == 409
        assert r2.headers["content-type"].startswith("application/problem+json")

    def test_create_no_qdrant_write(self, tmp_path: Any) -> None:
        """Creating a collection performs no Qdrant physical-collection write.

        The QdrantStore must not have a physical collection named after the
        logical collection at this point - state is DB-only until first sync.
        """
        s = _settings(tmp_path)
        app = create_app(s)
        with TestClient(app, raise_server_exceptions=False) as c:
            c.post("/collections", json={"name": "my-docs"})
            # Inspect Qdrant store directly via app state.
            store = app.state.qdrant_store
            physical_names = store.list_collections()
        # The logical name should not appear as a physical Qdrant collection.
        assert "my-docs" not in physical_names


# ---------------------------------------------------------------------------
# GET /collections
# ---------------------------------------------------------------------------


class TestGetCollections:
    """GET /collections lists all registered collections with corpus state."""

    def test_list_empty_returns_200(self, tmp_path: Any) -> None:
        """With no collections the endpoint returns 200 with an empty list."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/collections")
        assert r.status_code == 200

    def test_list_empty_body(self, tmp_path: Any) -> None:
        """No collections returns an empty items list."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/collections")
        body = r.json()
        assert "items" in body
        assert body["items"] == []

    def test_list_shows_created_collection(self, tmp_path: Any) -> None:
        """A collection created via POST appears in the list."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections")
        body = r.json()
        names = [item["name"] for item in body["items"]]
        assert "my-docs" in names

    def test_list_corpus_state_empty(self, tmp_path: Any) -> None:
        """A newly created collection reports corpus_state = 'empty' in the list."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections")
        body = r.json()
        item = next(i for i in body["items"] if i["name"] == "my-docs")
        assert item["corpus_state"] == "empty"

    def test_list_multiple_collections(self, tmp_path: Any) -> None:
        """Multiple collections all appear in the list."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "alpha"})
            c.post("/collections", json={"name": "beta"})
            r = c.get("/collections")
        body = r.json()
        names = [item["name"] for item in body["items"]]
        assert "alpha" in names
        assert "beta" in names


# ---------------------------------------------------------------------------
# GET /collections/{name}
# ---------------------------------------------------------------------------


class TestGetCollectionDetail:
    """GET /collections/{name} returns detail for a single collection."""

    def test_get_existing_returns_200(self, tmp_path: Any) -> None:
        """Fetching an existing collection returns 200."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections/my-docs")
        assert r.status_code == 200

    def test_get_nonexistent_returns_404(self, tmp_path: Any) -> None:
        """Fetching a non-existent collection returns 404 problem+json."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/collections/does-not-exist")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_get_detail_body_fields(self, tmp_path: Any) -> None:
        """Detail response includes name, corpus_state, created_at."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections/my-docs")
        body = r.json()
        assert body["name"] == "my-docs"
        assert body["corpus_state"] == "empty"
        assert "created_at" in body

    def test_get_detail_no_live_revision(self, tmp_path: Any) -> None:
        """A new collection with no revision has live_revision = null."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections/my-docs")
        body = r.json()
        assert body.get("live_revision") is None

    def test_get_detail_no_last_job(self, tmp_path: Any) -> None:
        """A new collection with no sync jobs has last_job = null."""
        s = _settings(tmp_path)
        with _client(s) as c:
            c.post("/collections", json={"name": "my-docs"})
            r = c.get("/collections/my-docs")
        body = r.json()
        assert body.get("last_job") is None


# ---------------------------------------------------------------------------
# Smoke test: embedded end-to-end walk
# ---------------------------------------------------------------------------


class TestSmokeWalk:
    """Boot embedded app, create collection, verify list + readyz consistency."""

    def test_smoke_create_list_readyz(self, tmp_path: Any) -> None:
        """Full walk: create -> list -> readyz shows collection as empty.

        This is the in-process smoke test required by the acceptance criteria.
        It boots the app with embedded Qdrant in a temp dir and walks the full
        create -> list -> readyz flow.
        """
        s = _settings(tmp_path)
        app = create_app(s)

        with TestClient(app, raise_server_exceptions=False) as c:
            # 1. Create a collection.
            r_create = c.post("/collections", json={"name": "smoke-test"})
            assert r_create.status_code == 201, r_create.text

            # 2. GET /collections shows it with corpus_state = empty.
            r_list = c.get("/collections")
            assert r_list.status_code == 200, r_list.text
            body_list = r_list.json()
            names = [item["name"] for item in body_list["items"]]
            assert "smoke-test" in names
            item = next(i for i in body_list["items"] if i["name"] == "smoke-test")
            assert item["corpus_state"] == "empty"

            # 3. GET /readyz reflects the empty collection.
            r_readyz = c.get("/readyz")
            assert r_readyz.status_code == 200, r_readyz.text
            body_readyz = r_readyz.json()
            assert "smoke-test" in body_readyz["collections"]
            assert body_readyz["collections"]["smoke-test"] == "empty"

            # 4. State is consistent: list and readyz agree.
            assert item["corpus_state"] == body_readyz["collections"]["smoke-test"]

    def test_smoke_no_qdrant_physical_collection_after_create(
        self, tmp_path: Any
    ) -> None:
        """Physical Qdrant collection is NOT created at registration time."""
        s = _settings(tmp_path)
        app = create_app(s)

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/collections", json={"name": "smoke-test"})
            assert r.status_code == 201

            # Inspect the Qdrant store - no physical collection should exist.
            store = app.state.qdrant_store
            physical = store.list_collections()

        assert "smoke-test" not in physical

    @pytest.mark.parametrize(
        "name",
        [
            "docs",
            "my-docs",
            "my_docs",
            "docs-v2",
            "a1b2c3",
            "abc123",
        ],
    )
    def test_valid_collection_names(self, tmp_path: Any, name: str) -> None:
        """Parametrized check that all valid name patterns are accepted."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": name})
        assert r.status_code == 201, f"Expected 201 for name {name!r}, got {r.status_code}"

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "UPPERCASE",
            "has space",
            "has.dot",
            "has/slash",
            "a" * 65,
        ],
    )
    def test_invalid_collection_names(self, tmp_path: Any, name: str) -> None:
        """Parametrized check that invalid names are rejected with 422."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/collections", json={"name": name})
        assert r.status_code == 422, f"Expected 422 for name {name!r}, got {r.status_code}"
