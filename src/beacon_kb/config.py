"""Frozen, validated configuration tree for beacon-kb.

This module defines the ``BeaconConfig`` dataclass hierarchy that mirrors the
``beacon-kb.toml`` schema used by both tool mode and library mode.

All config dataclasses are:
- ``frozen=True`` - immutable after construction.
- ``slots=True`` - memory-efficient, prevents accidental attribute addition.
- Validated on construction via ``__post_init__``.

Invalid values raise ``ConfigError`` that names the failing key and the fix.

Secrets (API keys, tokens) are referenced by environment-variable NAME only.
Inline secret values in config files are explicitly rejected.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from beacon_kb.errors import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Common patterns that look like actual secret values rather than env-var names
    re.compile(r"^sk-[A-Za-z0-9]{20,}$"),  # OpenAI-style key
    re.compile(r"^AIza[A-Za-z0-9_\-]{35}$"),  # Google API key
    re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$"),  # Base64-encoded token (40+ chars)
)

# Valid env-var names: uppercase letters, digits, underscores.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _require_env_var_name(value: str, key: str) -> None:
    """Raise ConfigError if *value* does not look like a bare env-var name.

    An env-var name must match ``[A-Z_][A-Z0-9_]*`` (uppercase letters, digits,
    underscores, no spaces).  Inline secret values are rejected.

    Check order: validate against _ENV_VAR_NAME_RE first.  A value that is a
    valid env-var name is accepted immediately - no secret-pattern scan is
    performed.  Only when the name check fails are the secret patterns consulted
    to choose between a "looks like an inline secret" diagnostic and a plain
    "must be an env-var NAME" diagnostic.

    Args:
        value: The string to validate.
        key:   Config key name for the error message.

    Raises:
        ConfigError: If the value contains an inline secret or is malformed.
    """
    if not value:
        return  # Empty means "not set"; handled elsewhere.
    # Accept immediately if it is a valid env-var name (no secret scan needed).
    if _ENV_VAR_NAME_RE.match(value):
        return
    # Name check failed - distinguish a likely inline secret from a plain typo.
    for pat in _SECRET_PATTERNS:
        if pat.match(value):
            raise ConfigError(
                f"'{key}' looks like an inline secret value. "
                f"Store the secret in an environment variable and set '{key}' "
                f"to the variable NAME (e.g. '{key} = \"MY_API_KEY\"')."
            )
    raise ConfigError(
        f"'{key}' must be an environment-variable NAME (uppercase letters, "
        f"digits, underscores, e.g. 'OPENAI_API_KEY'), not an inline value. "
        f"Got: {value!r}."
    )


def _require_positive(value: int | float, key: str) -> None:
    """Raise ConfigError if *value* is not strictly positive."""
    if value <= 0:
        raise ConfigError(
            f"'{key}' must be a positive number greater than 0. Got: {value!r}. "
            f"Fix: set a positive value, e.g. '{key} = 1'."
        )


def _require_non_negative(value: int | float, key: str) -> None:
    """Raise ConfigError if *value* is negative."""
    if value < 0:
        raise ConfigError(
            f"'{key}' must be a non-negative number (>= 0). Got: {value!r}. "
            f"Fix: set a non-negative value, e.g. '{key} = 0'."
        )


# ---------------------------------------------------------------------------
# Section: core
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """Core beacon-kb settings shared across all modes.

    Corresponds to the ``[core]`` section of ``beacon-kb.toml``.

    Attributes:
        corpus_name:   Human-readable name for the knowledge base corpus.
        data_dir:      Path (string) where beacon-kb stores its index and data.
        log_level:     Logging verbosity: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        plugin_api_version: Expected PLUGIN_API_VERSION; used to refuse
                       incompatible plugins at load time.
    """

    corpus_name: str = "default"
    data_dir: str = ".beacon"
    log_level: str = "INFO"
    plugin_api_version: int = 1

    def __post_init__(self) -> None:
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ConfigError(
                f"'core.log_level' must be one of {sorted(valid_log_levels)}. "
                f"Got: {self.log_level!r}. Fix: set log_level = \"INFO\"."
            )
        if not self.corpus_name or not self.corpus_name.strip():
            raise ConfigError(
                "'core.corpus_name' must be a non-empty string. "
                "Fix: set corpus_name = \"my-corpus\"."
            )
        if not self.data_dir or not self.data_dir.strip():
            raise ConfigError(
                "'core.data_dir' must be a non-empty path string. "
                "Fix: set data_dir = \".beacon\"."
            )
        _require_positive(self.plugin_api_version, "core.plugin_api_version")


# ---------------------------------------------------------------------------
# Section: retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Retrieval pipeline settings.

    Corresponds to the ``[retrieval]`` section of ``beacon-kb.toml``.

    Attributes:
        top_k:             Number of candidates to retrieve per query.
        mode:              Retrieval mode: 'sparse', 'dense', 'hybrid'.
        chunk_size:        Target token count for retrieval chunks.
        chunk_overlap:     Token overlap between adjacent chunks.
        embedder_name:     Plugin name for the embedder (entry-point key).
        embedder_api_key_env: Env-var NAME containing the embedder API key.
        reranker_name:     Plugin name for the reranker ('' to disable).
        reranker_api_key_env: Env-var NAME containing the reranker API key.
        fusion_name:       Plugin name for the fusion strategy.
    """

    top_k: int = 10
    mode: str = "hybrid"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedder_name: str = ""
    embedder_api_key_env: str = ""
    reranker_name: str = ""
    reranker_api_key_env: str = ""
    fusion_name: str = "rrf"

    def __post_init__(self) -> None:
        _require_positive(self.top_k, "retrieval.top_k")
        valid_modes = {"sparse", "dense", "hybrid"}
        if self.mode not in valid_modes:
            raise ConfigError(
                f"'retrieval.mode' must be one of {sorted(valid_modes)}. "
                f"Got: {self.mode!r}. Fix: set mode = \"hybrid\"."
            )
        _require_positive(self.chunk_size, "retrieval.chunk_size")
        _require_non_negative(self.chunk_overlap, "retrieval.chunk_overlap")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"'retrieval.chunk_overlap' ({self.chunk_overlap}) must be less than "
                f"'retrieval.chunk_size' ({self.chunk_size}). "
                f"Fix: set chunk_overlap to a value less than chunk_size."
            )
        if self.embedder_api_key_env:
            _require_env_var_name(self.embedder_api_key_env, "retrieval.embedder_api_key_env")
        if self.reranker_api_key_env:
            _require_env_var_name(self.reranker_api_key_env, "retrieval.reranker_api_key_env")


# ---------------------------------------------------------------------------
# Section: answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnswerConfig:
    """Answer generation settings.

    Corresponds to the ``[answer]`` section of ``beacon-kb.toml``.

    Attributes:
        generator_name:      Plugin name for the answer generator.
        generator_api_key_env: Env-var NAME containing the generator API key.
        model:               Model identifier string passed to the generator.
        max_input_tokens:    Hard budget for prompt tokens.
        max_output_tokens:   Hard budget for completion tokens.
        abstain_threshold:   Minimum evidence quality below which the generator
                             must abstain.  Range [0.0, 1.0].
        temperature:         Sampling temperature.  Range [0.0, 2.0].
    """

    generator_name: str = ""
    generator_api_key_env: str = ""
    model: str = ""
    max_input_tokens: int = 4096
    max_output_tokens: int = 512
    abstain_threshold: float = 0.5
    temperature: float = 0.0

    def __post_init__(self) -> None:
        _require_positive(self.max_input_tokens, "answer.max_input_tokens")
        _require_positive(self.max_output_tokens, "answer.max_output_tokens")
        if not (0.0 <= self.abstain_threshold <= 1.0):
            raise ConfigError(
                f"'answer.abstain_threshold' must be in [0.0, 1.0]. "
                f"Got: {self.abstain_threshold!r}. Fix: set abstain_threshold = 0.5."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigError(
                f"'answer.temperature' must be in [0.0, 2.0]. "
                f"Got: {self.temperature!r}. Fix: set temperature = 0.0."
            )
        if self.generator_api_key_env:
            _require_env_var_name(self.generator_api_key_env, "answer.generator_api_key_env")


# ---------------------------------------------------------------------------
# Section: agentic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgenticConfig:
    """Agentic investigation loop settings.

    Corresponds to the ``[agentic]`` section of ``beacon-kb.toml``.

    Attributes:
        max_steps:         Maximum number of reasoning steps per investigation.
        token_budget:      Total token budget for one agentic session.
        planner_name:      Plugin name for the query planner.
        grader_name:       Plugin name for the evidence grader.
        router_name:       Plugin name for the corpus router.
        session_store_name: Plugin name for the session-state store.
        planner_api_key_env: Env-var NAME containing the planner API key.
    """

    max_steps: int = 10
    token_budget: int = 32768
    planner_name: str = ""
    grader_name: str = ""
    router_name: str = ""
    session_store_name: str = ""
    planner_api_key_env: str = ""

    def __post_init__(self) -> None:
        _require_positive(self.max_steps, "agentic.max_steps")
        _require_positive(self.token_budget, "agentic.token_budget")
        if self.planner_api_key_env:
            _require_env_var_name(self.planner_api_key_env, "agentic.planner_api_key_env")


# ---------------------------------------------------------------------------
# Section: plugins
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PluginsConfig:
    """Plugin discovery and registration settings.

    Corresponds to the ``[plugins]`` section of ``beacon-kb.toml``.

    Attributes:
        auto_discover:      If True, scan installed entry-point groups on start.
        extra_paths:        Additional filesystem paths to search for plugins.
                            Stored as a tuple of strings.
        disabled:           Names of plugins to suppress even if discovered.
                            Stored as a tuple of strings.
    """

    auto_discover: bool = True
    extra_paths: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeaconConfig:
    """Root configuration tree for beacon-kb.

    Mirrors the full ``beacon-kb.toml`` schema.  All sections default to their
    respective section defaults so a zero-argument ``BeaconConfig()`` is always
    valid.

    Attributes:
        core:       ``[core]`` section.
        retrieval:  ``[retrieval]`` section.
        answer:     ``[answer]`` section.
        agentic:    ``[agentic]`` section.
        plugins:    ``[plugins]`` section.
    """

    core: CoreConfig = field(default_factory=CoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _coerce_section(cls: type[Any], raw: dict[str, Any]) -> Any:
    """Construct a config section dataclass from a raw dict.

    Ignores keys that are not fields of the dataclass (forward-compatible).
    Raises ConfigError with the failing key name and fix hint when validation
    fails inside __post_init__.

    Args:
        cls: The frozen dataclass class to construct.
        raw: Raw dict from TOML or env overlay.

    Returns:
        An instance of *cls*.

    Raises:
        ConfigError: On any validation failure.
    """
    import dataclasses as _dc

    field_names = {f.name for f in _dc.fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in field_names}
    return cls(**kwargs)


def build_config(raw: dict[str, Any]) -> BeaconConfig:
    """Construct and validate a BeaconConfig from a raw nested dict.

    Unknown top-level sections are ignored for forward compatibility.
    Each section is validated on construction; the first validation error
    terminates processing.

    Args:
        raw: Nested dict with optional keys 'core', 'retrieval', 'answer',
             'agentic', 'plugins'.  Each value is a flat dict of settings.

    Returns:
        Validated, frozen BeaconConfig.

    Raises:
        ConfigError: If any field value fails validation.
    """
    core = _coerce_section(CoreConfig, raw.get("core") or {})
    retrieval = _coerce_section(RetrievalConfig, raw.get("retrieval") or {})
    answer = _coerce_section(AnswerConfig, raw.get("answer") or {})
    agentic = _coerce_section(AgenticConfig, raw.get("agentic") or {})
    plugins_raw = raw.get("plugins") or {}
    # Convert lists to tuples for frozen compatibility.
    if "extra_paths" in plugins_raw:
        plugins_raw = dict(plugins_raw)
        plugins_raw["extra_paths"] = tuple(plugins_raw["extra_paths"])
    if "disabled" in plugins_raw:
        plugins_raw = dict(plugins_raw)
        plugins_raw["disabled"] = tuple(plugins_raw["disabled"])
    plugins = _coerce_section(PluginsConfig, plugins_raw)
    return BeaconConfig(
        core=core,
        retrieval=retrieval,
        answer=answer,
        agentic=agentic,
        plugins=plugins,
    )
