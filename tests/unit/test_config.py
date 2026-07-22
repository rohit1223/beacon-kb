"""Unit tests for beacon_kb.config.

Tests cover:
- Frozen/immutable dataclasses
- Default construction (zero-arg) succeeds
- Validation on construction: invalid values raise ConfigError with key + fix
- Env-var-name-only enforcement: inline secrets are rejected
- build_config() construction from nested dicts
- Section isolation: one bad section doesn't silently skip others
"""

from __future__ import annotations

import dataclasses

import pytest

from beacon_kb.config import (
    AgenticConfig,
    AnswerConfig,
    BeaconConfig,
    CoreConfig,
    PluginsConfig,
    RetrievalConfig,
    build_config,
)
from beacon_kb.errors import ConfigError

# ===========================================================================
# Default construction
# ===========================================================================


@pytest.mark.unit
class TestDefaultConstruction:
    """Zero-arg construction of every config section must succeed."""

    def test_core_config_defaults(self) -> None:
        cfg = CoreConfig()
        assert cfg.corpus_name == "default"
        assert cfg.data_dir == ".beacon"
        assert cfg.log_level == "INFO"
        assert cfg.plugin_api_version == 1

    def test_retrieval_config_defaults(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.top_k == 10
        assert cfg.mode == "hybrid"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64

    def test_answer_config_defaults(self) -> None:
        cfg = AnswerConfig()
        assert cfg.max_input_tokens == 4096
        assert cfg.max_output_tokens == 512
        assert cfg.abstain_threshold == 0.0
        assert cfg.temperature == 0.0

    def test_agentic_config_defaults(self) -> None:
        cfg = AgenticConfig()
        assert cfg.max_steps == 10
        assert cfg.token_budget == 32768

    def test_plugins_config_defaults(self) -> None:
        cfg = PluginsConfig()
        assert cfg.auto_discover is True
        assert cfg.extra_paths == ()
        assert cfg.disabled == ()

    def test_beacon_config_defaults(self) -> None:
        cfg = BeaconConfig()
        assert isinstance(cfg.core, CoreConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.answer, AnswerConfig)
        assert isinstance(cfg.agentic, AgenticConfig)
        assert isinstance(cfg.plugins, PluginsConfig)


# ===========================================================================
# Frozen / immutable
# ===========================================================================


@pytest.mark.unit
class TestFrozen:
    """Config dataclasses must be immutable (frozen=True)."""

    def test_core_config_is_frozen(self) -> None:
        cfg = CoreConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.log_level = "DEBUG"  # type: ignore[misc]

    def test_retrieval_config_is_frozen(self) -> None:
        cfg = RetrievalConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.top_k = 99  # type: ignore[misc]

    def test_answer_config_is_frozen(self) -> None:
        cfg = AnswerConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.temperature = 1.0  # type: ignore[misc]

    def test_agentic_config_is_frozen(self) -> None:
        cfg = AgenticConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.max_steps = 99  # type: ignore[misc]

    def test_plugins_config_is_frozen(self) -> None:
        cfg = PluginsConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.auto_discover = False  # type: ignore[misc]

    def test_beacon_config_is_frozen(self) -> None:
        cfg = BeaconConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.core = CoreConfig(log_level="DEBUG")  # type: ignore[misc]


# ===========================================================================
# Validation: CoreConfig
# ===========================================================================


@pytest.mark.unit
class TestCoreConfigValidation:
    """CoreConfig validates its fields on construction."""

    def test_invalid_log_level_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            CoreConfig(log_level="VERBOSE")
        msg = str(exc_info.value)
        assert "core.log_level" in msg
        assert "Fix:" in msg

    def test_empty_corpus_name_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            CoreConfig(corpus_name="")
        msg = str(exc_info.value)
        assert "core.corpus_name" in msg

    def test_empty_data_dir_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            CoreConfig(data_dir="")
        msg = str(exc_info.value)
        assert "core.data_dir" in msg

    def test_zero_plugin_api_version_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            CoreConfig(plugin_api_version=0)
        msg = str(exc_info.value)
        assert "core.plugin_api_version" in msg

    def test_valid_log_levels_accepted(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = CoreConfig(log_level=level)
            assert cfg.log_level == level


# ===========================================================================
# Validation: RetrievalConfig
# ===========================================================================


@pytest.mark.unit
class TestRetrievalConfigValidation:
    """RetrievalConfig validates its fields on construction."""

    def test_zero_top_k_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            RetrievalConfig(top_k=0)
        assert "retrieval.top_k" in str(exc_info.value)

    def test_invalid_mode_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            RetrievalConfig(mode="bm42")
        msg = str(exc_info.value)
        assert "retrieval.mode" in msg
        assert "Fix:" in msg

    def test_zero_chunk_size_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            RetrievalConfig(chunk_size=0)
        assert "retrieval.chunk_size" in str(exc_info.value)

    def test_chunk_overlap_gte_chunk_size_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            RetrievalConfig(chunk_size=100, chunk_overlap=100)
        assert "chunk_overlap" in str(exc_info.value)

    def test_valid_modes_accepted(self) -> None:
        for mode in ("sparse", "dense", "hybrid"):
            cfg = RetrievalConfig(mode=mode)
            assert cfg.mode == mode


# ===========================================================================
# Validation: AnswerConfig
# ===========================================================================


@pytest.mark.unit
class TestAnswerConfigValidation:
    """AnswerConfig validates its fields on construction."""

    def test_zero_max_input_tokens_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(max_input_tokens=0)
        assert "answer.max_input_tokens" in str(exc_info.value)

    def test_zero_max_output_tokens_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(max_output_tokens=0)
        assert "answer.max_output_tokens" in str(exc_info.value)

    def test_abstain_threshold_out_of_range_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(abstain_threshold=1.5)
        assert "abstain_threshold" in str(exc_info.value)

    def test_negative_abstain_threshold_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(abstain_threshold=-0.1)
        assert "abstain_threshold" in str(exc_info.value)

    def test_temperature_out_of_range_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(temperature=2.5)
        assert "temperature" in str(exc_info.value)

    def test_boundary_values_accepted(self) -> None:
        cfg = AnswerConfig(abstain_threshold=0.0, temperature=0.0)
        assert cfg.abstain_threshold == 0.0
        cfg2 = AnswerConfig(abstain_threshold=1.0, temperature=2.0)
        assert cfg2.abstain_threshold == 1.0


# ===========================================================================
# Validation: AgenticConfig
# ===========================================================================


@pytest.mark.unit
class TestAgenticConfigValidation:
    """AgenticConfig validates its fields on construction."""

    def test_zero_max_steps_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AgenticConfig(max_steps=0)
        assert "agentic.max_steps" in str(exc_info.value)

    def test_zero_token_budget_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AgenticConfig(token_budget=0)
        assert "agentic.token_budget" in str(exc_info.value)


# ===========================================================================
# Secret enforcement: env-var name only
# ===========================================================================


@pytest.mark.unit
class TestSecretEnvVarEnforcement:
    """Secrets must be referenced by env-var NAME, not inline values."""

    def test_inline_openai_key_rejected(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(generator_api_key_env="sk-abcdefghijklmnopqrstuvwxyz1234567890")
        msg = str(exc_info.value)
        assert "inline secret" in msg.lower() or "env-var" in msg.lower() or "NAME" in msg

    def test_malformed_env_var_name_rejected(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            AnswerConfig(generator_api_key_env="my api key")
        assert "generator_api_key_env" in str(exc_info.value)

    def test_valid_env_var_name_accepted(self) -> None:
        cfg = AnswerConfig(generator_api_key_env="OPENAI_API_KEY")
        assert cfg.generator_api_key_env == "OPENAI_API_KEY"

    def test_long_all_caps_alphanum_env_var_name_accepted(self) -> None:
        """A 40+ char all-caps alphanumeric env-var name must not be flagged
        as an inline secret (regression for false-positive in check order)."""
        long_name = "OPENAICLAUDEDEFAULTMODELKEYNAME123456789"
        assert len(long_name) >= 40
        cfg = AnswerConfig(generator_api_key_env=long_name)
        assert cfg.generator_api_key_env == long_name

    def test_retrieval_embedder_key_validated(self) -> None:
        with pytest.raises(ConfigError):
            RetrievalConfig(embedder_api_key_env="bad key name")

    def test_agentic_planner_key_validated(self) -> None:
        with pytest.raises(ConfigError):
            AgenticConfig(planner_api_key_env="bad key name")

    def test_empty_env_var_name_allowed(self) -> None:
        """Empty string means 'not configured'; it must not raise."""
        cfg = AnswerConfig(generator_api_key_env="")
        assert cfg.generator_api_key_env == ""


# ===========================================================================
# build_config()
# ===========================================================================


@pytest.mark.unit
class TestBuildConfig:
    """build_config() constructs a BeaconConfig from a raw nested dict."""

    def test_empty_dict_uses_all_defaults(self) -> None:
        cfg = build_config({})
        assert isinstance(cfg, BeaconConfig)
        assert cfg.core.log_level == "INFO"
        assert cfg.retrieval.mode == "hybrid"

    def test_partial_override_preserved(self) -> None:
        cfg = build_config({"core": {"log_level": "DEBUG"}})
        assert cfg.core.log_level == "DEBUG"
        assert cfg.core.corpus_name == "default"  # other field still default

    def test_plugins_lists_converted_to_tuples(self) -> None:
        cfg = build_config({"plugins": {"extra_paths": ["/a", "/b"], "disabled": ["plugin1"]}})
        assert cfg.plugins.extra_paths == ("/a", "/b")
        assert cfg.plugins.disabled == ("plugin1",)

    def test_invalid_value_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            build_config({"core": {"log_level": "TRACE"}})
        assert "core.log_level" in str(exc_info.value)

    def test_unknown_keys_ignored(self) -> None:
        """Forward-compatible: unknown keys must not raise."""
        cfg = build_config({"core": {"log_level": "DEBUG", "future_field": "ignored"}})
        assert cfg.core.log_level == "DEBUG"

    def test_unknown_top_level_sections_ignored(self) -> None:
        cfg = build_config({"observability": {"tracer": "otel"}})
        assert cfg.core.log_level == "INFO"
