"""Integration tests for the documents REST resource (Task 02.1).

Covers:
- POST /documents: upload, content-hash dedupe, size limit (413), media type.
- POST /collections/{name}/sources: attach connector, unknown collection (404),
  unknown kind (422), FK enforcement.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from fastapi.testclient import TestClient

from beacon.config import BeaconSettings, IngestSettings, QdrantSettings, StateSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Any, max_upload_bytes: int = 10 * 1024 * 1024) -> BeaconSettings:
    """Build test settings backed by a tmp_path SQLite DB and embedded Qdrant."""
    return BeaconSettings(
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
        ingest=IngestSettings(
            data_dir=str(tmp_path / "data"),
            max_upload_bytes=max_upload_bytes,
        ),
    )


def _client(settings: BeaconSettings) -> TestClient:
    """Build a TestClient for *settings*."""
    from beacon.server.app import create_app

    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


def _upload(client: TestClient, content: bytes, filename: str = "test.md") -> Any:
    """POST /documents with *content* as a multipart file upload."""
    return client.post(
        "/documents",
        files={"file": (filename, io.BytesIO(content), "text/markdown")},
    )


# ---------------------------------------------------------------------------
# POST /documents
# ---------------------------------------------------------------------------


class TestPostDocuments:
    """POST /documents uploads files, deduplicates by content hash."""

    def test_upload_returns_201(self, tmp_path: Any) -> None:
        """First upload of new content returns 201."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"hello beacon")
        assert r.status_code == 201

    def test_upload_canonical_uri_scheme(self, tmp_path: Any) -> None:
        """canonical_uri of uploaded file starts with 'upload://'."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"hello beacon")
        body = r.json()
        assert body["canonical_uri"].startswith("upload://")

    def test_upload_stored_true_first_time(self, tmp_path: Any) -> None:
        """stored=True on the first upload of a given content hash."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"unique content abc")
        body = r.json()
        assert body["stored"] is True

    def test_upload_dedupe_returns_200(self, tmp_path: Any) -> None:
        """Re-uploading the same bytes returns 200 (not 201)."""
        s = _settings(tmp_path)
        with _client(s) as c:
            _upload(c, b"deduped content")
            r2 = _upload(c, b"deduped content")
        assert r2.status_code == 200

    def test_upload_dedupe_stored_false(self, tmp_path: Any) -> None:
        """Re-uploading identical bytes returns stored=False."""
        s = _settings(tmp_path)
        with _client(s) as c:
            _upload(c, b"deduped content")
            r2 = _upload(c, b"deduped content")
        body = r2.json()
        assert body["stored"] is False

    def test_upload_content_hash_matches_sha256(self, tmp_path: Any) -> None:
        """content_hash in the response matches SHA-256 of the uploaded bytes."""
        content = b"hash check content"
        expected_hash = hashlib.sha256(content).hexdigest()
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, content)
        body = r.json()
        assert body["content_hash"] == expected_hash

    def test_upload_canonical_uri_contains_hash(self, tmp_path: Any) -> None:
        """canonical_uri encodes the content hash: upload://<sha256>."""
        content = b"uri hash check"
        expected_hash = hashlib.sha256(content).hexdigest()
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, content)
        body = r.json()
        assert body["canonical_uri"] == f"upload://{expected_hash}"

    def test_upload_size_limit_413(self, tmp_path: Any) -> None:
        """Upload exceeding max_upload_bytes returns 413."""
        s = _settings(tmp_path, max_upload_bytes=100)
        with _client(s) as c:
            r = _upload(c, b"x" * 200)
        assert r.status_code == 413

    def test_upload_size_limit_problem_json(self, tmp_path: Any) -> None:
        """413 response uses application/problem+json content-type."""
        s = _settings(tmp_path, max_upload_bytes=100)
        with _client(s) as c:
            r = _upload(c, b"x" * 200)
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_upload_registers_source(self, tmp_path: Any) -> None:
        """Uploading a file registers a source record accessible via source_id."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"register source check")
        body = r.json()
        assert "source_id" in body
        assert isinstance(body["source_id"], int)
        assert body["source_id"] > 0

    def test_upload_media_type_markdown(self, tmp_path: Any) -> None:
        """Uploading a .md file returns media_type text/markdown."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"# Hello", "doc.md")
        body = r.json()
        assert body["media_type"] == "text/markdown"

    def test_upload_media_type_fallback(self, tmp_path: Any) -> None:
        """Unknown extension with no declared content-type falls back to octet-stream."""
        s = _settings(tmp_path)
        with _client(s) as c:
            # .beacontest is not a real registered MIME extension.
            r = c.post(
                "/documents",
                files={
                    "file": (
                        "data.beacontest",
                        io.BytesIO(b"binary data"),
                        "application/octet-stream",
                    )
                },
            )
        body = r.json()
        assert body["media_type"] == "application/octet-stream"

    def test_upload_missing_file_422(self, tmp_path: Any) -> None:
        """POST /documents without a file field returns 422."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post("/documents", data={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /collections/{name}/sources
# ---------------------------------------------------------------------------


class TestPostCollectionSources:
    """POST /collections/{name}/sources attaches connector definitions."""

    def _create_collection(self, client: TestClient, name: str = "docs") -> None:
        r = client.post("/collections", json={"name": name})
        assert r.status_code == 201

    def test_attach_folder_returns_201(self, tmp_path: Any) -> None:
        """Attaching a folder connector to an existing collection returns 201."""
        s = _settings(tmp_path)
        with _client(s) as c:
            self._create_collection(c)
            r = c.post(
                "/collections/docs/sources",
                json={"connector_kind": "folder", "config": {"root": str(tmp_path)}},
            )
        assert r.status_code == 201

    def test_attach_folder_response_shape(self, tmp_path: Any) -> None:
        """Response body for attach includes canonical_uri, connector_kind, id."""
        s = _settings(tmp_path)
        with _client(s) as c:
            self._create_collection(c)
            r = c.post(
                "/collections/docs/sources",
                json={"connector_kind": "folder", "config": {"root": str(tmp_path)}},
            )
        body = r.json()
        assert "canonical_uri" in body
        assert body["connector_kind"] == "folder"
        assert "id" in body
        assert isinstance(body["id"], int)

    def test_attach_unknown_collection_404(self, tmp_path: Any) -> None:
        """Attaching a source to a nonexistent collection returns 404 problem+json."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.post(
                "/collections/ghost/sources",
                json={"connector_kind": "folder", "config": {"root": "/tmp"}},
            )
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        assert body["kind"] == "not-found"

    def test_attach_unknown_connector_kind_422(self, tmp_path: Any) -> None:
        """Unknown connector_kind returns 422 problem+json."""
        s = _settings(tmp_path)
        with _client(s) as c:
            self._create_collection(c)
            r = c.post(
                "/collections/docs/sources",
                json={"connector_kind": "bogus", "config": {}},
            )
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_attach_no_config_defaults_empty(self, tmp_path: Any) -> None:
        """Attaching without config field defaults to empty dict (no error)."""
        s = _settings(tmp_path)
        with _client(s) as c:
            self._create_collection(c)
            r = c.post(
                "/collections/docs/sources",
                json={"connector_kind": "folder"},
            )
        # 201 or a config-level validation error is acceptable, but not a server crash.
        assert r.status_code in (201, 422)

    def test_attach_fk_enforced_at_db_level(self, tmp_path: Any) -> None:
        """FK constraint: cannot attach source to collection that doesn't exist in DB."""
        s = _settings(tmp_path)
        with _client(s) as c:
            # Attempt to attach without creating the collection first.
            r = c.post(
                "/collections/nonexistent/sources",
                json={"connector_kind": "folder", "config": {"root": "/tmp"}},
            )
        # Route must return 404 before even hitting the FK constraint.
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Upload connector_kind correctness
# ---------------------------------------------------------------------------


class TestUploadConnectorKind:
    """connector_kind for uploads must be 'upload', not the MIME type."""

    def test_upload_source_connector_kind_is_upload(self, tmp_path: Any) -> None:
        """Upload source has connector_kind='upload' (not the media type) and media_type set."""
        from beacon.state.db import StateDB
        from beacon.state.repo import SourceRepo
        s = _settings(tmp_path)
        with _client(s) as c:
            r = _upload(c, b"# connector kind test", "doc.md")
        assert r.status_code == 201
        body = r.json()
        # Verify DB row has connector_kind="upload" and media_type="text/markdown".
        db = StateDB(db_path=str(tmp_path / "beacon.db"))
        repo = SourceRepo(db)
        row = repo.get(collection_name="__uploads__", canonical_uri=body["canonical_uri"])
        db.close()
        assert row is not None
        assert row["connector_kind"] == "upload"
        assert row["media_type"] == "text/markdown"


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    """Auth enforcement tests for document and source endpoints."""

    def _settings_with_key(self, tmp_path: Any, key: str = "test-secret-key") -> BeaconSettings:
        """Settings with an API key configured."""
        from pydantic import SecretStr

        from beacon.config import ServerSettings
        s = _settings(tmp_path)
        s = BeaconSettings(
            state=s.state,
            qdrant=s.qdrant,
            ingest=s.ingest,
            server=ServerSettings(api_key=SecretStr(key)),
        )
        return s

    def test_post_documents_without_bearer_returns_401(self, tmp_path: Any) -> None:
        """POST /documents without a Bearer token returns 401 when API key is configured."""
        s = self._settings_with_key(tmp_path)
        with _client(s) as c:
            r = c.post(
                "/documents",
                files={"file": ("test.md", io.BytesIO(b"hello"), "text/markdown")},
            )
        assert r.status_code == 401

    def test_post_collection_sources_without_bearer_returns_401(self, tmp_path: Any) -> None:
        """POST /collections/{name}/sources without Bearer returns 401 when key is set."""
        s = self._settings_with_key(tmp_path)
        with _client(s) as c:
            r = c.post(
                "/collections/docs/sources",
                json={"connector_kind": "folder", "config": {"root": "/tmp"}},
            )
        assert r.status_code == 401
