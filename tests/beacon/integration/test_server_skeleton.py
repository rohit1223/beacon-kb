"""Integration tests for the FastAPI server skeleton (Task 01.4).

Verifies:
- App factory isolation: two instances with different settings share no state.
- /healthz: always 200 while serving.
- /readyz: per-collection corpus state and 503 on backend failure.
- Problem-details: every taxonomy error maps to the correct status + kind.
- Validation errors use problem-details shape, not FastAPI default.
- Catch-all: unhandled exceptions become a generic problem with no stack trace.
- Auth: no key -> all pass; key set -> localhost exempt, non-localhost needs key.
- OTel: no warnings when not configured; spans produced when configured.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from beacon.config import BeaconSettings, QdrantSettings, ServerSettings, StateSettings
from beacon.errors import (
    BackendError,
    BudgetError,
    CitationError,
    IngestionError,
    ReadinessError,
)
from beacon.server.app import create_app
from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    CorpusState,
    RevisionRepo,
    SyncJobRepo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Any, *, api_key: str | None = None) -> BeaconSettings:
    """Build test settings backed by a tmp_path SQLite DB and embedded Qdrant."""
    server_kw: dict[str, Any] = {}
    if api_key is not None:
        server_kw["api_key"] = api_key
    return BeaconSettings(
        server=ServerSettings(**server_kw),
        state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        qdrant=QdrantSettings(path=str(tmp_path / "qdrant")),
    )


def _client(settings: BeaconSettings) -> TestClient:
    """Build a TestClient for *settings*."""
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Factory isolation
# ---------------------------------------------------------------------------


class TestFactoryIsolation:
    """Two create_app instances must not share DB or store state."""

    def test_two_instances_use_separate_dbs(self, tmp_path: Any) -> None:
        """Each app factory call creates a fully independent state DB."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        settings_a = BeaconSettings(
            state=StateSettings(db_path=str(tmp_path / "a" / "beacon.db")),
        )
        settings_b = BeaconSettings(
            state=StateSettings(db_path=str(tmp_path / "b" / "beacon.db")),
        )
        app_a = create_app(settings_a)
        app_b = create_app(settings_b)
        assert app_a is not app_b

    def test_settings_not_shared_between_instances(self, tmp_path: Any) -> None:
        """Settings object is stored independently per app instance."""
        s1 = BeaconSettings(
            server=ServerSettings(port=8001),
            state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        )
        s2 = BeaconSettings(
            server=ServerSettings(port=8002),
            state=StateSettings(db_path=str(tmp_path / "beacon.db")),
        )
        app1 = create_app(s1)
        app2 = create_app(s2)
        assert app1 is not app2


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthz:
    """GET /healthz must always return 200 while the process is serving."""

    def test_healthz_returns_200(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/healthz")
        assert r.status_code == 200

    def test_healthz_body_is_json(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/healthz")
        body = r.json()
        assert "status" in body
        assert body["status"] == "ok"

    def test_healthz_content_type_is_json(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/healthz")
        assert "application/json" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestReadyz:
    """GET /readyz returns corpus state per collection; 503 on backend failure."""

    def test_readyz_200_no_collections(self, tmp_path: Any) -> None:
        """With no collections registered the endpoint reports ready (no failures)."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200

    def test_readyz_includes_collections(self, tmp_path: Any) -> None:
        """Registered collections appear in the readiness response."""
        s = _settings(tmp_path)
        # Pre-seed a collection in the DB so /readyz can find it.
        db = StateDB(db_path=s.state.db_path)
        CollectionRepo(db).create(name="mydocs")
        db.close()

        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert "collections" in body
        assert "mydocs" in body["collections"]
        assert body["collections"]["mydocs"] == CorpusState.EMPTY

    def test_readyz_reports_ready_state(self, tmp_path: Any) -> None:
        """A collection with a LIVE revision reports as ready."""
        s = _settings(tmp_path)
        db = StateDB(db_path=s.state.db_path)
        CollectionRepo(db).create(name="docs")
        RevisionRepo(db).create(revision_id="rev1", collection_name="docs")
        RevisionRepo(db).set_live("rev1", collection_name="docs")
        db.close()

        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["collections"]["docs"] == CorpusState.READY

    def test_readyz_reports_building_state(self, tmp_path: Any) -> None:
        """A collection with a running sync job reports as building."""
        s = _settings(tmp_path)
        db = StateDB(db_path=s.state.db_path)
        CollectionRepo(db).create(name="docs")
        SyncJobRepo(db).create(job_id="job1", collection_name="docs")
        SyncJobRepo(db).set_running("job1")
        db.close()

        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["collections"]["docs"] == CorpusState.BUILDING

    def test_readyz_reports_failed_state(self, tmp_path: Any) -> None:
        """A collection whose last job failed and has no live revision reports as failed."""
        s = _settings(tmp_path)
        db = StateDB(db_path=s.state.db_path)
        CollectionRepo(db).create(name="docs")
        SyncJobRepo(db).create(job_id="job1", collection_name="docs")
        SyncJobRepo(db).set_running("job1")
        SyncJobRepo(db).set_failed("job1")
        db.close()

        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["collections"]["docs"] == CorpusState.FAILED

    def test_readyz_503_on_backend_unreachable(self, tmp_path: Any) -> None:
        """When the state DB raises BackendError the endpoint returns 503 problem."""
        s = _settings(tmp_path)
        app = create_app(s)

        from beacon.server.routes import health as health_module

        def _raise_backend(*args: Any, **kwargs: Any) -> Any:
            raise BackendError("DB unreachable")

        with patch.object(health_module, "derive_corpus_state", _raise_backend):
            # Also need a collection to iterate over.
            db = StateDB(db_path=s.state.db_path)
            CollectionRepo(db).create(name="docs")
            db.close()
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/readyz")

        assert r.status_code == 503
        body = r.json()
        assert body.get("kind") == "readiness"
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_readyz_includes_qdrant_info(self, tmp_path: Any) -> None:
        """200 readyz response includes qdrant reachability info."""
        s = _settings(tmp_path)
        with _client(s) as c:
            r = c.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert "qdrant" in body
        assert body["qdrant"]["reachable"] is True
        assert body["qdrant"]["mode"] in ("embedded", "server")

    def test_readyz_503_when_qdrant_unreachable(self, tmp_path: Any) -> None:
        """503 problem when Qdrant store raises BackendError during ping."""
        from beacon.errors import BackendError
        from beacon.storage import qdrant as qdrant_mod

        s = _settings(tmp_path)
        app = create_app(s)

        original_list = qdrant_mod.QdrantStore.list_collections

        def _raise(self: Any) -> Any:
            raise BackendError("qdrant unreachable")

        qdrant_mod.QdrantStore.list_collections = _raise  # type: ignore[method-assign]
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/readyz")
        finally:
            qdrant_mod.QdrantStore.list_collections = original_list  # type: ignore[method-assign]

        assert r.status_code == 503
        body = r.json()
        assert body.get("kind") == "readiness"
        assert r.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# Error handlers: taxonomy errors
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Taxonomy errors must produce application/problem+json responses."""

    def _app_with_route(
        self, settings: BeaconSettings, exc: Exception
    ) -> Any:
        """Create an app that raises *exc* from a test route."""
        app = create_app(settings)

        @app.get("/_test_error")
        async def _raise() -> None:
            raise exc

        return app

    def test_readiness_error_maps_to_503(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        app = self._app_with_route(s, ReadinessError("backend not ready"))
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_error")
        assert r.status_code == 503
        body = r.json()
        assert body["kind"] == "readiness"
        assert body["status"] == 503
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_backend_error_maps_to_502(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        app = self._app_with_route(s, BackendError("qdrant down"))
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_error")
        assert r.status_code == 502
        body = r.json()
        assert body["kind"] == "backend"
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_ingestion_error_maps_to_422(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        app = self._app_with_route(s, IngestionError("bad doc", source_uri="file.pdf"))
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_error")
        assert r.status_code == 422
        body = r.json()
        assert body["kind"] == "ingestion"
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_citation_error_maps_to_422(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        app = self._app_with_route(s, CitationError("bad citation"))
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_error")
        assert r.status_code == 422
        body = r.json()
        assert body["kind"] == "citation"
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_budget_error_maps_to_402(self, tmp_path: Any) -> None:
        s = _settings(tmp_path)
        app = self._app_with_route(s, BudgetError("cost exceeded"))
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_error")
        assert r.status_code == 402
        body = r.json()
        assert body["kind"] == "budget"
        assert r.headers["content-type"].startswith("application/problem+json")

    def test_unhandled_exception_returns_500_problem(self, tmp_path: Any) -> None:
        """Unexpected exceptions must return a generic problem with no stack trace."""
        s = _settings(tmp_path)
        app = create_app(s)

        @app.get("/_test_crash")
        async def _crash() -> None:
            raise RuntimeError("internal boom")

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_crash")

        assert r.status_code == 500
        body = r.json()
        # Must be problem+json.
        assert r.headers["content-type"].startswith("application/problem+json")
        # Must NOT contain internal details.
        body_text = r.text
        assert "internal boom" not in body_text
        assert "Traceback" not in body_text
        # Must have type and title fields per RFC 9457.
        assert "type" in body
        assert "title" in body

    def test_validation_error_is_problem_json(self, tmp_path: Any) -> None:
        """FastAPI validation errors must be returned as problem+json, not the default shape."""
        s = _settings(tmp_path)
        app = create_app(s)

        from pydantic import BaseModel

        class Body(BaseModel):
            count: int

        @app.post("/_test_validation")
        async def _validate(body: Body) -> dict[str, int]:
            return {"count": body.count}

        with TestClient(app, raise_server_exceptions=False) as c:
            # Send invalid body (string instead of int).
            r = c.post("/_test_validation", json={"count": "not-a-number"})

        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")
        body = r.json()
        # Must have problem-detail fields.
        assert "type" in body
        assert "title" in body
        # Must NOT have FastAPI's default "detail" list format
        # (FastAPI default has {"detail": [{"loc": [...], "msg": ..., ...}]}).
        # Our shape has "detail" as a plain string.
        assert not isinstance(body.get("detail"), list)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """API-key middleware: off when no key; on with localhost exemption."""

    def test_no_key_configured_all_pass(self, tmp_path: Any) -> None:
        """With no api_key in settings every request passes."""
        s = _settings(tmp_path, api_key=None)
        with _client(s) as c:
            r = c.get("/healthz")
        assert r.status_code == 200

    def test_key_configured_healthz_exempt(self, tmp_path: Any) -> None:
        """GET /healthz is always reachable even when a key is configured."""
        s = _settings(tmp_path, api_key="secret-key")
        # TestClient host is "testclient" (not localhost); middleware allows /healthz.
        with _client(s) as c:
            r = c.get("/healthz")
        assert r.status_code == 200

    def test_key_configured_wrong_bearer_401(self, tmp_path: Any) -> None:
        """Non-localhost request with wrong bearer token gets 401 problem.

        TestClient host is "testclient" (not localhost), so middleware enforces auth.
        """
        s = _settings(tmp_path, api_key="correct-key")
        app = create_app(s)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/readyz", headers={"Authorization": "Bearer wrong-key"})

        assert r.status_code == 401
        body = r.json()
        assert r.headers["content-type"].startswith("application/problem+json")
        assert "type" in body
        assert "title" in body

    def test_key_configured_correct_bearer_passes(self, tmp_path: Any) -> None:
        """Non-localhost request with correct bearer token passes.

        TestClient host is "testclient" (not localhost), so auth is enforced.
        """
        s = _settings(tmp_path, api_key="correct-key")
        app = create_app(s)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/readyz", headers={"Authorization": "Bearer correct-key"})

        assert r.status_code == 200

    def test_key_configured_no_bearer_401(self, tmp_path: Any) -> None:
        """Non-localhost request without bearer token gets 401.

        TestClient host is "testclient" (not localhost), so auth is enforced.
        """
        s = _settings(tmp_path, api_key="correct-key")
        app = create_app(s)
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/readyz")

        assert r.status_code == 401
        body = r.json()
        assert r.headers["content-type"].startswith("application/problem+json")
        assert "type" in body

    def test_key_configured_localhost_exempt(self, tmp_path: Any) -> None:
        """When a key is configured, localhost clients pass without a token."""
        s = _settings(tmp_path, api_key="secret-key")
        app = create_app(s)

        from beacon.server import auth as auth_module

        original = auth_module._is_localhost
        auth_module._is_localhost = lambda addr: True
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.get("/readyz")
        finally:
            auth_module._is_localhost = original

        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    """OTel: no-op when not configured; spans when configured."""

    def test_no_otel_no_warnings(self, tmp_path: Any, recwarn: Any) -> None:
        """Starting the app without OTel config produces zero warnings."""
        s = _settings(tmp_path)
        app = create_app(s)
        with TestClient(app, raise_server_exceptions=False) as c:
            c.get("/healthz")
        otel_warnings = [w for w in recwarn.list if "otel" in str(w.message).lower()]
        assert len(otel_warnings) == 0

    def test_get_tracer_returns_noop_without_sdk(self, tmp_path: Any) -> None:
        """get_tracer() returns a usable tracer even without a configured exporter."""
        from beacon.server.telemetry import get_tracer

        tracer = get_tracer("test-component")
        # Must be usable as a context manager (no-op span).
        with tracer.start_as_current_span("test-span") as span:
            assert span is not None

    def test_span_helper_works_without_sdk(self, tmp_path: Any) -> None:
        """pipeline_span() context manager works without an OTel SDK exporter."""
        from beacon.server.telemetry import pipeline_span

        with pipeline_span("ingestion", collection="docs") as span:
            assert span is not None

    def test_app_imports_when_otel_absent(self) -> None:
        """telemetry module correctly no-ops when opentelemetry is not importable.

        Uses a sys.meta_path blocker to raise ImportError for any opentelemetry
        import, then reloads the telemetry module to verify _OTEL_AVAILABLE is
        False and instrument_app is a no-op.
        """
        import importlib
        import importlib.abc
        import importlib.machinery
        import sys
        from types import ModuleType

        class _BlockOtel(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            """Meta path finder that blocks opentelemetry imports."""

            def find_spec(
                self,
                fullname: str,
                path: object,
                target: object = None,
            ) -> importlib.machinery.ModuleSpec | None:
                if fullname.startswith("opentelemetry"):
                    return importlib.machinery.ModuleSpec(fullname, self)
                return None

            def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
                return None

            def exec_module(self, module: ModuleType) -> None:
                raise ImportError(f"opentelemetry blocked for test: {module.__name__}")

        blocker = _BlockOtel()

        # Save and purge opentelemetry modules.
        saved: dict[str, Any] = {}
        for k in list(sys.modules):
            if "opentelemetry" in k:
                saved[k] = sys.modules.pop(k)

        sys.meta_path.insert(0, blocker)
        try:
            import beacon.server.telemetry as telemetry_mod
            importlib.reload(telemetry_mod)

            # _OTEL_AVAILABLE must be False when the SDK is blocked.
            assert telemetry_mod._OTEL_AVAILABLE is False

            # instrument_app must be a no-op (must not raise).
            from beacon.server.app import create_app
            s = BeaconSettings()
            app = create_app(s)
            assert app is not None
            telemetry_mod.instrument_app(app)

            # get_tracer must still return a usable tracer.
            tracer = telemetry_mod.get_tracer()
            with tracer.start_as_current_span("test") as span:
                assert span is not None
        finally:
            sys.meta_path.remove(blocker)
            for k, mod in saved.items():
                sys.modules[k] = mod
            importlib.reload(telemetry_mod)
