"""Unit tests for beacon_kb.config_loader.

Tests cover:
- load_config() with no file (defaults only).
- load_config() with a valid TOML file.
- load_config() with a missing file raises ConfigError.
- Env-var overlay: BEACON_<SECTION>_<FIELD> overrides values.
- Layer merge order: defaults < TOML < env.
- Malformed TOML raises ConfigError with actionable message.
- resolve_secret() reads from os.environ.
- load_config_or_default() does not raise on missing file.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from beacon_kb.config_loader import load_config, load_config_or_default, resolve_secret
from beacon_kb.errors import ConfigError

# ===========================================================================
# Helpers
# ===========================================================================


def _write_toml(content: str) -> pathlib.Path:
    """Write *content* to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w")
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return pathlib.Path(tmp.name)


# ===========================================================================
# Defaults only (no file)
# ===========================================================================


@pytest.mark.unit
class TestLoadConfigNoFile:
    """load_config(path=None) returns all defaults."""

    def test_no_file_returns_defaults(self) -> None:
        cfg = load_config(path=None)
        assert cfg.core.log_level == "INFO"
        assert cfg.retrieval.mode == "hybrid"
        assert cfg.answer.max_input_tokens == 4096

    def test_no_file_beacon_config_is_frozen(self) -> None:
        import dataclasses

        cfg = load_config(path=None)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.core = cfg.core  # type: ignore[misc]


# ===========================================================================
# TOML file loading
# ===========================================================================


@pytest.mark.unit
class TestLoadConfigToml:
    """load_config() with a valid TOML file merges over defaults."""

    def test_loads_core_section(self) -> None:
        path = _write_toml('[core]\nlog_level = "DEBUG"\ncorpus_name = "docs"\n')
        try:
            cfg = load_config(path=path)
            assert cfg.core.log_level == "DEBUG"
            assert cfg.core.corpus_name == "docs"
            assert cfg.core.data_dir == ".beacon"  # default preserved
        finally:
            path.unlink()

    def test_loads_retrieval_section(self) -> None:
        path = _write_toml("[retrieval]\ntop_k = 20\nmode = \"dense\"\n")
        try:
            cfg = load_config(path=path)
            assert cfg.retrieval.top_k == 20
            assert cfg.retrieval.mode == "dense"
        finally:
            path.unlink()

    def test_loads_answer_section(self) -> None:
        path = _write_toml(
            "[answer]\ngenerator_name = \"openai\"\nmax_input_tokens = 8192\n"
        )
        try:
            cfg = load_config(path=path)
            assert cfg.answer.generator_name == "openai"
            assert cfg.answer.max_input_tokens == 8192
        finally:
            path.unlink()

    def test_loads_agentic_section(self) -> None:
        path = _write_toml("[agentic]\nmax_steps = 5\ntoken_budget = 16384\n")
        try:
            cfg = load_config(path=path)
            assert cfg.agentic.max_steps == 5
            assert cfg.agentic.token_budget == 16384
        finally:
            path.unlink()

    def test_loads_plugins_section(self) -> None:
        path = _write_toml(
            '[plugins]\nauto_discover = false\nextra_paths = ["/opt/plugins"]\n'
        )
        try:
            cfg = load_config(path=path)
            assert cfg.plugins.auto_discover is False
            assert "/opt/plugins" in cfg.plugins.extra_paths
        finally:
            path.unlink()

    def test_missing_file_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            load_config(path="/nonexistent/beacon-kb.toml")
        msg = str(exc_info.value)
        assert "not found" in msg.lower() or "nonexistent" in msg

    def test_malformed_toml_raises_config_error(self) -> None:
        path = _write_toml("[core\nbroken toml {{{\n")
        try:
            with pytest.raises(ConfigError) as exc_info:
                load_config(path=path)
            msg = str(exc_info.value)
            assert "toml" in msg.lower() or "parse" in msg.lower()
        finally:
            path.unlink()

    def test_invalid_value_in_toml_raises_config_error(self) -> None:
        path = _write_toml('[core]\nlog_level = "TRACE"\n')
        try:
            with pytest.raises(ConfigError) as exc_info:
                load_config(path=path)
            assert "core.log_level" in str(exc_info.value)
        finally:
            path.unlink()


# ===========================================================================
# Env-var overlay
# ===========================================================================


@pytest.mark.unit
class TestEnvVarOverlay:
    """BEACON_<SECTION>_<FIELD> env vars override config values."""

    def test_env_overrides_core_log_level(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_CORE_LOG_LEVEL": "DEBUG"})
        assert cfg.core.log_level == "DEBUG"

    def test_env_overrides_retrieval_top_k(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_RETRIEVAL_TOP_K": "25"})
        assert cfg.retrieval.top_k == 25

    def test_env_overrides_answer_temperature(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_ANSWER_TEMPERATURE": "0.7"})
        assert abs(cfg.answer.temperature - 0.7) < 1e-9

    def test_env_overrides_agentic_max_steps(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_AGENTIC_MAX_STEPS": "20"})
        assert cfg.agentic.max_steps == 20

    def test_env_overrides_plugins_auto_discover_false(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_PLUGINS_AUTO_DISCOVER": "false"})
        assert cfg.plugins.auto_discover is False

    def test_env_overrides_plugins_auto_discover_true(self) -> None:
        cfg = load_config(path=None, environ={"BEACON_PLUGINS_AUTO_DISCOVER": "1"})
        assert cfg.plugins.auto_discover is True

    def test_unknown_env_vars_ignored(self) -> None:
        """Non-BEACON_ vars must not affect config."""
        cfg = load_config(path=None, environ={"SOME_OTHER_VAR": "xyz"})
        assert cfg.core.log_level == "INFO"

    def test_env_invalid_int_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            load_config(path=None, environ={"BEACON_RETRIEVAL_TOP_K": "notanint"})

    def test_env_invalid_bool_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            load_config(path=None, environ={"BEACON_PLUGINS_AUTO_DISCOVER": "maybe"})


# ===========================================================================
# Layer merge order: defaults < TOML < env
# ===========================================================================


@pytest.mark.unit
class TestLayerMergeOrder:
    """Env overrides TOML, TOML overrides defaults."""

    def test_env_beats_toml(self) -> None:
        path = _write_toml('[core]\nlog_level = "WARNING"\n')
        try:
            cfg = load_config(path=path, environ={"BEACON_CORE_LOG_LEVEL": "ERROR"})
            assert cfg.core.log_level == "ERROR"  # env wins
        finally:
            path.unlink()

    def test_toml_beats_defaults(self) -> None:
        path = _write_toml('[retrieval]\ntop_k = 50\n')
        try:
            cfg = load_config(path=path)
            assert cfg.retrieval.top_k == 50  # TOML wins
        finally:
            path.unlink()

    def test_defaults_used_when_no_toml_or_env(self) -> None:
        cfg = load_config(path=None, environ={})
        assert cfg.retrieval.mode == "hybrid"


# ===========================================================================
# load_config_or_default
# ===========================================================================


@pytest.mark.unit
class TestLoadConfigOrDefault:
    """load_config_or_default() does not raise on missing file."""

    def test_missing_file_returns_defaults(self) -> None:
        cfg = load_config_or_default(path="/nonexistent/beacon-kb.toml")
        assert cfg.core.log_level == "INFO"

    def test_valid_file_is_loaded(self) -> None:
        path = _write_toml('[core]\nlog_level = "DEBUG"\n')
        try:
            cfg = load_config_or_default(path=path)
            assert cfg.core.log_level == "DEBUG"
        finally:
            path.unlink()


# ===========================================================================
# resolve_secret
# ===========================================================================


@pytest.mark.unit
class TestResolveSecret:
    """resolve_secret() reads the value of a named env var."""

    def test_resolve_set_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "super-secret-value")
        result = resolve_secret("MY_API_KEY")
        assert result == "super-secret-value"

    def test_resolve_unset_env_var_returns_none(self) -> None:
        result = resolve_secret("BEACON_NONEXISTENT_SECRET_12345")
        assert result is None

    def test_resolve_empty_name_returns_none(self) -> None:
        result = resolve_secret("")
        assert result is None
