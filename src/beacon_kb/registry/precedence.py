"""Deterministic precedence resolver for the beacon-kb plugin registry.

Resolution order (first match wins):
  1. Explicit instance - caller passed an object directly via ``register()``.
  2. Config-named plugin - ``name`` matches an explicit registered name.
  3. Sole entry-point default - group has exactly one discovered entry point
     and no name was requested (or name matches).
  4. Built-in default - the group's registered built-in.

Any other outcome raises a typed error:
- ``PluginConflict``  - two entry points share the same name in a group.
- ``PluginNotFound``  - requested name is not installed / registered.
- ``ProtocolMismatch`` - resolved object fails runtime protocol check.
- ``PluginError``     - incompatible PLUGIN_API_VERSION or bad capability.

Importing this module performs no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import EntryPoint
from typing import Any

from beacon_kb.errors import PluginConflict, PluginError, PluginNotFound, ProtocolMismatch
from beacon_kb.registry import discovery
from beacon_kb.version import PLUGIN_API_VERSION

# ---------------------------------------------------------------------------
# In-memory registry store (module-level, populated by register())
# ---------------------------------------------------------------------------

# Explicit instances registered by the application or built-in initialiser.
_explicit: dict[str, dict[str, Any]] = {}
"""Mapping of group -> { name -> instance }."""

# Built-in defaults per group.
_builtins: dict[str, tuple[str, Any]] = {}
"""Mapping of group -> (name, instance) for the registered built-in."""


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register(group: str, name: str, instance: Any) -> None:
    """Register an explicit plugin instance under *group* / *name*.

    This is how built-ins and application-level overrides are registered.
    Replaces any previous registration for the same group/name pair.

    Args:
        group:    Entry-point group string.
        name:     Plugin name (must be a non-empty string).
        instance: The plugin object to register.

    Raises:
        ValueError: If *name* is empty.
    """
    if not name:
        raise ValueError("Plugin name must be a non-empty string.")
    _explicit.setdefault(group, {})[name] = instance


def register_builtin(group: str, name: str, instance: Any) -> None:
    """Register a built-in default for *group*.

    The built-in is the lowest-precedence fallback: it is used only when
    no explicit instance and no entry-point plugin satisfy the request.

    Only one built-in per group is supported.  Calling this again for the
    same group replaces the previous built-in.

    Args:
        group:    Entry-point group string.
        name:     Canonical name for the built-in plugin.
        instance: The built-in plugin object.
    """
    _builtins[group] = (name, instance)


def clear_registry() -> None:
    """Remove all registered instances and built-ins.

    Intended for use in tests only.  Do not call in production code.
    """
    _explicit.clear()
    _builtins.clear()


def list_registered(group: str) -> list[str]:
    """Return the names of all explicitly registered plugins for *group*.

    Does not trigger entry-point discovery.

    Args:
        group: Entry-point group string.

    Returns:
        List of registered plugin names (may be empty).
    """
    return list(_explicit.get(group, {}).keys())


def describe(group: str, name: str) -> dict[str, Any]:
    """Return a description dict for the registered plugin *name* in *group*.

    Args:
        group: Entry-point group string.
        name:  Plugin name.

    Returns:
        Dict with at least ``group``, ``name``, and ``instance`` keys.

    Raises:
        PluginNotFound: If the name is not registered explicitly.
    """
    group_reg = _explicit.get(group, {})
    if name not in group_reg:
        raise PluginNotFound(group=group, name=name)
    return {
        "group": group,
        "name": name,
        "instance": group_reg[name],
    }


# ---------------------------------------------------------------------------
# Capability validation
# ---------------------------------------------------------------------------


def _validate_capability(group: str, name: str, instance: Any) -> None:
    """Check capability metadata declared on *instance* before use.

    Validates:
    - ``plugin_api_version`` attribute against ``PLUGIN_API_VERSION``.
      A plugin that declares a *lower major version* is rejected.
    - Any other capability metadata checks can be layered here in later epics.

    Args:
        group:    Entry-point group for error messages.
        name:     Plugin name for error messages.
        instance: The resolved plugin object.

    Raises:
        PluginError: If the plugin declares an incompatible API version.
    """
    declared_version = getattr(instance, "plugin_api_version", None)
    if declared_version is not None:
        if not isinstance(declared_version, int):
            raise PluginError(
                f"Plugin '{name}' in group '{group}' declared plugin_api_version="
                f"{declared_version!r} which is not an integer."
            )
        if declared_version < PLUGIN_API_VERSION:
            raise PluginError(
                f"Plugin '{name}' in group '{group}' targets plugin_api_version="
                f"{declared_version} but beacon-kb requires >= {PLUGIN_API_VERSION}. "
                f"Update or uninstall the plugin."
            )


def _validate_declared_capabilities(
    group: str,
    name: str,
    instance: Any,
    required: Mapping[str, object] | None,
) -> None:
    """Check *instance*'s declared capability metadata against *required* values.

    For each ``(key, expected)`` pair in *required*, the plugin's declared
    value is looked up as an attribute of the same name.  Zero-argument
    callables (e.g. ``Embedder.dimension()``) are invoked to obtain the
    declared value.  A plugin that does not declare the capability is
    accepted; a plugin whose declared value differs from the configured
    value is rejected before any indexing begins.

    Args:
        group:    Entry-point group for error messages.
        name:     Plugin name for error messages.
        instance: The resolved plugin object.
        required: Mapping of capability name to the config-required value,
                  or None to skip the check.

    Raises:
        PluginError: If a declared capability conflicts with *required*.
    """
    if not required:
        return
    for key, expected in required.items():
        if not hasattr(instance, key):
            continue  # Capability not declared; nothing to conflict with.
        declared = getattr(instance, key)
        if callable(declared):
            try:
                declared = declared()
            except TypeError:
                continue  # Not a zero-argument declaration; skip.
        if declared != expected:
            raise PluginError(
                f"Plugin '{name}' in group '{group}' declares capability "
                f"{key}={declared!r} which conflicts with the configured "
                f"value {expected!r}. Fix the configuration or choose a "
                f"compatible plugin."
            )


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------


def _validate_protocol(
    group: str, name: str, instance: Any, protocol: type | None
) -> None:
    """Check *instance* satisfies *protocol* via isinstance().

    Args:
        group:    Entry-point group for error messages.
        name:     Plugin name for error messages.
        instance: The resolved plugin object.
        protocol: A runtime_checkable Protocol class, or None to skip.

    Raises:
        ProtocolMismatch: If ``isinstance(instance, protocol)`` is False.
    """
    if protocol is None:
        return
    if not isinstance(instance, protocol):
        # Collect missing members for a helpful error message.
        missing = [
            attr
            for attr in dir(protocol)
            if not attr.startswith("_") and not hasattr(instance, attr)
        ]
        raise ProtocolMismatch(group=group, name=name, missing_members=missing)


# ---------------------------------------------------------------------------
# Entry point loading
# ---------------------------------------------------------------------------


def _load_entry_point(ep: EntryPoint) -> Any:
    """Load an entry point and instantiate it if the value is a class.

    Entry points may resolve to either a class or a pre-built instance.
    When the resolved value is a type (i.e. a class), it is instantiated
    with no arguments to produce the plugin object.

    Args:
        ep: The ``EntryPoint`` to load.

    Returns:
        The plugin instance.
    """
    value = ep.load()
    if isinstance(value, type):
        return value()
    return value


def _distribution_name(ep: EntryPoint) -> str:
    """Return the distribution name for an entry point, or its value string."""
    try:
        dist = ep.dist
        if dist is not None:
            name_val = dist.metadata["Name"]
            if name_val is not None:
                return str(name_val)
    except Exception:
        pass
    return ep.value


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------


def resolve(
    group: str,
    name: str | None = None,
    *,
    protocol: type | None = None,
    capabilities: Mapping[str, object] | None = None,
) -> Any:
    """Resolve a plugin for *group* using the fixed precedence order.

    Precedence (first match wins):
      1. Explicit instance with the given *name* (or the only registered one
         when *name* is None and exactly one is registered).
      2. Config-named entry-point: scans installed entry points and loads the
         one whose name matches *name*.
      3. Sole entry-point default: when *name* is None and exactly one entry
         point is installed in the group.
      4. Built-in default for the group.

    Args:
        group:        Entry-point group string (use constants from ``groups``).
        name:         Requested plugin name.  ``None`` means "use the default".
        protocol:     Optional runtime_checkable Protocol class to validate
                      the resolved object against.
        capabilities: Optional mapping of capability name to the value the
                      configuration requires (e.g. ``{"dimension": 768}``).
                      A plugin whose declared capability conflicts is rejected
                      with a typed ``PluginError`` before use.

    Returns:
        The resolved plugin object.

    Raises:
        PluginConflict:    Two entry points share the same name in *group*.
        PluginNotFound:    Requested *name* is not installed / registered;
                           the error lists the group and installed names.
        ProtocolMismatch:  Resolved object does not satisfy *protocol*.
        PluginError:       Incompatible ``plugin_api_version`` or a declared
                           capability that conflicts with configuration.
    """

    def _validate_and_return(found_name: str, instance: Any) -> Any:
        _validate_capability(group, found_name, instance)
        _validate_declared_capabilities(group, found_name, instance, capabilities)
        _validate_protocol(group, found_name, instance, protocol)
        return instance

    # 1. Explicit instance lookup.
    group_explicit = _explicit.get(group, {})
    if name is not None:
        if name in group_explicit:
            return _validate_and_return(name, group_explicit[name])
    else:
        # name=None: use the sole registered instance if there is exactly one.
        if len(group_explicit) == 1:
            found_name, instance = next(iter(group_explicit.items()))
            return _validate_and_return(found_name, instance)

    # 2 & 3. Entry-point discovery (lazy - triggers on first call).
    ep_map: dict[str, list[EntryPoint]] = discovery.scan_group(group)

    # Detect conflicts before selection - never last-installed-wins.
    for ep_name, eps in ep_map.items():
        if len(eps) > 1 and (name is None or ep_name == name):
            providers = [_distribution_name(ep) for ep in eps]
            raise PluginConflict(group=group, name=ep_name, providers=providers)

    if name is not None:
        # 2. Config-named entry-point.
        if name in ep_map:
            return _validate_and_return(name, _load_entry_point(ep_map[name][0]))
    else:
        # 3. Sole entry-point default.
        if len(ep_map) == 1:
            sole_name, eps = next(iter(ep_map.items()))
            return _validate_and_return(sole_name, _load_entry_point(eps[0]))

    # 4. Built-in default.
    if group in _builtins:
        builtin_name, instance = _builtins[group]
        if name is None or name == builtin_name:
            return _validate_and_return(builtin_name, instance)

    # Nothing matched - report the group and the names that ARE available.
    installed: set[str] = set(ep_map.keys()) | set(group_explicit.keys())
    if group in _builtins:
        installed.add(_builtins[group][0])
    raise PluginNotFound(
        group=group,
        name=name or "default",
        installed=sorted(installed),
    )
