"""Tests for BeaconSettings - defaults, env overrides, nesting, and secret redaction."""

from __future__ import annotations

import json
import os

import pytest

from beacon.config import BeaconSettings


class TestDefaults:
    """All fields must have local-first defaults; no credentials required."""

    def test_instantiates_with_no_env(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings is not None

    def test_server_default_host(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.server.host == "127.0.0.1"

    def test_server_default_port(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.server.port == 8000

    def test_server_auth_off_by_default(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.server.api_key is None

    def test_qdrant_default_is_local_path(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.qdrant.url is None
        assert settings.qdrant.path is not None

    def test_qdrant_no_api_key_by_default(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.qdrant.api_key is None

    def test_models_default_embedding(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.models.embedding_model is not None

    def test_retrieval_defaults_present(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.retrieval.top_k > 0

    def test_answer_defaults_present(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.answer.max_tokens > 0

    def test_investigate_defaults_present(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.investigate.max_iterations > 0

    def test_state_default_db_path(self, clean_env: None) -> None:
        settings = BeaconSettings()
        assert settings.state.db_path is not None


class TestEnvOverrides:
    """Every setting must be overridable via BEACON_-prefixed env vars."""

    def test_server_host_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_SERVER__HOST", "0.0.0.0")
        settings = BeaconSettings()
        assert settings.server.host == "0.0.0.0"

    def test_server_port_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_SERVER__PORT", "9000")
        settings = BeaconSettings()
        assert settings.server.port == 9000

    def test_qdrant_url_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_QDRANT__URL", "http://qdrant:6333")
        settings = BeaconSettings()
        assert settings.qdrant.url == "http://qdrant:6333"

    def test_retrieval_top_k_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_RETRIEVAL__TOP_K", "20")
        settings = BeaconSettings()
        assert settings.retrieval.top_k == 20

    def test_answer_max_tokens_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_ANSWER__MAX_TOKENS", "4096")
        settings = BeaconSettings()
        assert settings.answer.max_tokens == 4096


class TestSecretRedaction:
    """Secret-typed settings must not appear in repr or model dump."""

    def test_api_key_not_in_repr(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-api-key-12345"
        monkeypatch.setenv("BEACON_SERVER__API_KEY", secret)
        settings = BeaconSettings()
        r = repr(settings)
        assert secret not in r

    def test_qdrant_api_key_not_in_repr(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "qdrant-secret-key-99999"
        monkeypatch.setenv("BEACON_QDRANT__API_KEY", secret)
        settings = BeaconSettings()
        r = repr(settings)
        assert secret not in r

    def test_llm_api_key_not_in_repr(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "llm-secret-key-77777"
        monkeypatch.setenv("BEACON_MODELS__LLM_API_KEY", secret)
        settings = BeaconSettings()
        r = repr(settings)
        assert secret not in r

    def test_api_key_not_in_safe_dump(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-api-key-12345"
        monkeypatch.setenv("BEACON_SERVER__API_KEY", secret)
        settings = BeaconSettings()
        dump = settings.safe_dump()
        dump_str = json.dumps(dump)
        assert secret not in dump_str

    def test_qdrant_api_key_not_in_safe_dump(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "qdrant-secret-key-99999"
        monkeypatch.setenv("BEACON_QDRANT__API_KEY", secret)
        settings = BeaconSettings()
        dump = settings.safe_dump()
        dump_str = json.dumps(dump)
        assert secret not in dump_str

    def test_llm_api_key_not_in_safe_dump(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "llm-secret-key-77777"
        monkeypatch.setenv("BEACON_MODELS__LLM_API_KEY", secret)
        settings = BeaconSettings()
        dump = settings.safe_dump()
        dump_str = json.dumps(dump)
        assert secret not in dump_str


class TestSectionEnvIsolation:
    """Section models must be plain BaseModel so bare env vars don't leak in.

    Regression tests for the critical fix: sections must NOT inherit
    BaseSettings (which would independently scan the environment for their
    bare field names, e.g. PORT or PATH).
    """

    def test_bare_port_env_does_not_override_server_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PORT=80 in the environment must not change server.port from its default."""
        monkeypatch.setenv("PORT", "80")
        # Remove any BEACON_ vars that might legitimately set port.
        for key in list(os.environ):
            if key.startswith("BEACON_"):
                monkeypatch.delenv(key, raising=False)
        settings = BeaconSettings()
        assert settings.server.port == 8000

    def test_ambient_path_does_not_land_in_qdrant_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shell PATH variable must never appear as qdrant.path."""
        for key in list(os.environ):
            if key.startswith("BEACON_"):
                monkeypatch.delenv(key, raising=False)
        settings = BeaconSettings()
        # qdrant.path must be exactly the short relative default.
        assert settings.qdrant.path == "data/qdrant"

    def test_beacon_server_port_env_still_routes_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BEACON_SERVER__PORT must override server.port despite PORT also being set."""
        monkeypatch.setenv("PORT", "80")
        monkeypatch.setenv("BEACON_SERVER__PORT", "9000")
        for key in list(os.environ):
            if key.startswith("BEACON_") and key != "BEACON_SERVER__PORT":
                monkeypatch.delenv(key, raising=False)
        settings = BeaconSettings()
        assert settings.server.port == 9000
