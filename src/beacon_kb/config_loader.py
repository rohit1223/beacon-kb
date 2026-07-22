"""Config loader: TOML file + env overlay, deterministic layer merge.

Loading order (later layers win):
  1. Built-in defaults (the zero-arg BeaconConfig defaults).
  2. TOML file at the given path (if it exists).
  3. Environment-variable overlay (reads the env vars NAMED in the config
     whose names match ``BEACON_<SECTION>_<KEY>`` or individual secret env
     vars referenced by name in the config).

Secrets are NEVER stored inline.  The loader reads the VALUE of an env var
whose NAME is stored in config (e.g. ``generator_api_key_env = "OPENAI_KEY"``
means the loader calls ``os.environ.get("OPENAI_KEY")`` at load time and
makes it available via the returned mapping --- the secret is not stored in
the config object itself).

Actionable diagnostics: every error names the failing key and the exact fix.

Importing this module performs no side effects.
"""

from __future__ import annotations

import dataclasses as _dc
import os
import pathlib
import tomllib
from collections.abc import Mapping
from typing import Any

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

# ---------------------------------------------------------------------------
# Env-var overlay prefix
# ---------------------------------------------------------------------------

_PREFIX = "BEACON_"
"""All env-var overrides for config values start with this prefix."""

# Mapping from (section, field) to the env-var override name.
# Format: BEACON_<SECTION>_<FIELD> where <SECTION> and <FIELD> are uppercase.
# Example: BEACON_CORE_LOG_LEVEL overrides core.log_level.


def _env_override_key(section: str, field: str) -> str:
    """Return the env-var name used to override a config field.

    Args:
        section: Config section name (e.g. 'core').
        field:   Field name within the section (e.g. 'log_level').

    Returns:
        Env-var name string, e.g. 'BEACON_CORE_LOG_LEVEL'.
    """
    return f"{_PREFIX}{section.upper()}_{field.upper()}"


# ---------------------------------------------------------------------------
# Type coercion for env-var strings
# ---------------------------------------------------------------------------


def _coerce_env_value(value: str, current: Any) -> Any:
    """Coerce an env-var string to the same type as *current*.

    Supports: bool, int, float, str, tuple.  Lists / tuples from env vars must
    be comma-separated strings.

    Args:
        value:   Raw env-var string value.
        current: Current field value whose type to match.

    Returns:
        Coerced value.

    Raises:
        ConfigError: If coercion fails.
    """
    if isinstance(current, bool):
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(
            f"Expected boolean env-var value (true/false/1/0). Got: {value!r}."
        )
    if isinstance(current, int):
        try:
            return int(value)
        except ValueError:
            raise ConfigError(
                f"Expected integer env-var value. Got: {value!r}."
            ) from None
    if isinstance(current, float):
        try:
            return float(value)
        except ValueError:
            raise ConfigError(
                f"Expected float env-var value. Got: {value!r}."
            ) from None
    if isinstance(current, tuple):
        # Comma-separated list.
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(items)
    # Fallback: str
    return value


# ---------------------------------------------------------------------------
# Deep merge helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge *overlay* into *base*, returning a new dict.

    Nested dicts are merged recursively.  Scalar values in *overlay* replace
    those in *base*.  Keys absent from *overlay* are preserved from *base*.

    Args:
        base:    Base dict (defaults or previous layer).
        overlay: Overlay dict (higher-priority layer).

    Returns:
        New merged dict (does not mutate inputs).
    """
    merged: dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# Env overlay extraction
# ---------------------------------------------------------------------------

_SECTION_FIELDS: dict[str, list[str]] = {
    "core": [
        "corpus_name",
        "data_dir",
        "log_level",
        "plugin_api_version",
    ],
    "retrieval": [
        "top_k",
        "mode",
        "chunk_size",
        "chunk_overlap",
        "embedder_name",
        "embedder_api_key_env",
        "reranker_name",
        "reranker_api_key_env",
        "fusion_name",
    ],
    "answer": [
        "generator_name",
        "generator_api_key_env",
        "model",
        "max_input_tokens",
        "max_output_tokens",
        "abstain_threshold",
        "temperature",
    ],
    "agentic": [
        "max_steps",
        "token_budget",
        "planner_name",
        "grader_name",
        "router_name",
        "session_store_name",
        "planner_api_key_env",
    ],
    "plugins": [
        "auto_discover",
        "extra_paths",
        "disabled",
    ],
}


def _extract_env_overlay(
    defaults_raw: dict[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an env-var overlay dict by scanning known BEACON_* variables.

    Each field in each section is probed against its ``BEACON_<SECTION>_<FIELD>``
    env var.  Only variables that are set are included in the overlay.

    The type of the env-var string is coerced to match the default value type.

    Args:
        defaults_raw: Nested dict of default field values (section -> field -> value).
        environ:      Optional mapping to read env vars from.  Defaults to
                      ``os.environ``.  Passing a dict here avoids any mutation
                      of the real process environment.

    Returns:
        Overlay dict with the same structure as a TOML raw dict.  May be empty
        if no matching env vars are set.

    Raises:
        ConfigError: If an env-var value cannot be coerced to the expected type.
    """
    env: Mapping[str, str] = environ if environ is not None else os.environ
    overlay: dict[str, Any] = {}
    for section, fields in _SECTION_FIELDS.items():
        section_defaults = defaults_raw.get(section, {})
        section_overlay: dict[str, Any] = {}
        for fname in fields:
            env_key = _env_override_key(section, fname)
            env_val = env.get(env_key)
            if env_val is not None:
                current = section_defaults.get(fname)
                try:
                    coerced = _coerce_env_value(env_val, current)
                except ConfigError as exc:
                    raise ConfigError(
                        f"Env var '{env_key}' (for '{section}.{fname}'): {exc}"
                    ) from exc
                section_overlay[fname] = coerced
        if section_overlay:
            overlay[section] = section_overlay
    return overlay


# ---------------------------------------------------------------------------
# Defaults extraction
# ---------------------------------------------------------------------------

def _fields_to_dict(instance: Any) -> dict[str, Any]:
    return {f.name: getattr(instance, f.name) for f in _dc.fields(instance)}


# Module-level constant: default config sections as a raw nested dict.
# Safe to share because all section dataclasses are frozen.
_DEFAULTS_RAW: dict[str, Any] = {
    "core": _fields_to_dict(CoreConfig()),
    "retrieval": _fields_to_dict(RetrievalConfig()),
    "answer": _fields_to_dict(AnswerConfig()),
    "agentic": _fields_to_dict(AgenticConfig()),
    "plugins": _fields_to_dict(PluginsConfig()),
}


def _defaults_as_raw() -> dict[str, Any]:
    """Return the default field values from each config section as a nested dict.

    Returns a reference to the module-level ``_DEFAULTS_RAW`` constant.
    Callers must not mutate the returned dict.

    Returns:
        Nested dict mirroring the TOML structure with default values.
    """
    return _DEFAULTS_RAW


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(
    path: str | pathlib.Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BeaconConfig:
    """Load and return a validated BeaconConfig using the three-layer merge.

    Layers (later wins):
      1. Built-in defaults.
      2. TOML file at *path* (skipped if *path* is None or does not exist).
      3. Env-var overlay (``BEACON_<SECTION>_<FIELD>``).

    The function is intentionally pure: it reads only the file at *path* and
    the environment (or the *environ* override for testing).  No network I/O,
    no credential storage.

    Args:
        path:    Path to the ``beacon-kb.toml`` config file.  If None, only
                 defaults and env overlay are used.
        environ: Optional env-var mapping override (for testing).  If None the
                 real ``os.environ`` is used.

    Returns:
        Validated, frozen BeaconConfig.

    Raises:
        ConfigError: If the TOML is malformed, any value fails validation, or
                     an env-var cannot be coerced to the expected type.
    """
    # Layer 1: defaults
    raw = _defaults_as_raw()

    # Layer 2: TOML file
    if path is not None:
        file_path = pathlib.Path(path)
        if file_path.exists():
            try:
                with file_path.open("rb") as fh:
                    toml_raw = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(
                    f"Failed to parse TOML config at '{file_path}': {exc}. "
                    f"Fix: ensure the file is valid TOML."
                ) from exc
            raw = _deep_merge(raw, toml_raw)
        else:
            raise ConfigError(
                f"Config file not found: '{file_path}'. "
                f"Fix: create the file or pass path=None to use defaults."
            )

    # Layer 3: env overlay
    env_overlay = _extract_env_overlay(raw, environ=environ)

    if env_overlay:
        raw = _deep_merge(raw, env_overlay)

    return build_config(raw)


def load_config_or_default(
    path: str | pathlib.Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BeaconConfig:
    """Like :func:`load_config` but returns defaults if the file does not exist.

    Unlike :func:`load_config`, a missing file is not an error; only
    validation failures raise.

    Args:
        path:    Path to the ``beacon-kb.toml`` file (may be absent).
        environ: Optional env-var mapping override (for testing).

    Returns:
        Validated BeaconConfig (possibly all defaults).

    Raises:
        ConfigError: On TOML parse or validation failure.
    """
    if path is not None and not pathlib.Path(path).exists():
        path = None
    return load_config(path, environ=environ)


def resolve_secret(env_var_name: str) -> str | None:
    """Read and return the value of the named env var at call time.

    This is the canonical way to resolve a secret referenced by name in
    config (e.g. ``config.answer.generator_api_key_env``).  The value is
    never cached; each call reads the current env.

    Args:
        env_var_name: The environment-variable name (e.g. 'OPENAI_API_KEY').

    Returns:
        The secret string, or None if the variable is not set.
    """
    if not env_var_name:
        return None
    return os.environ.get(env_var_name)
