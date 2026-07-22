"""Plugin registry facade for beacon-kb.

Exposes the public API for resolving, registering, listing, and describing
plugins by group and name.

Lazy discovery: entry points are scanned only on the first call to
``resolve()``.  Importing this module does NOT scan entry points.
Use ``discovery.has_scanned()`` to assert this in tests.

Importing this module triggers built-in registration (``builtins.py``)
but does NOT trigger entry-point discovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from beacon_kb.registry import builtins as _builtins_module  # noqa: F401
from beacon_kb.registry import discovery, groups, precedence

# Register first-party built-ins through the shared path.
# This import has the side-effect of calling _register_builtins(), which
# is intentional and expected.

__all__ = [
    "describe",
    "discovery",
    "groups",
    "list_plugins",
    "precedence",
    "register",
    "resolve",
]


def resolve(
    group: str,
    name: str | None = None,
    *,
    protocol: type | None = None,
    capabilities: Mapping[str, object] | None = None,
) -> Any:
    """Resolve a plugin for *group* using the fixed precedence order.

    Precedence (first match wins):
      1. Explicit instance registered via ``register()``.
      2. Config-named entry-point (name match in installed entry points).
      3. Sole entry-point default (no name requested, one entry point installed).
      4. Built-in default for the group.

    When *protocol* is not supplied, the group's canonical protocol is looked
    up via ``groups.get_protocol_for_group()`` and used automatically.
    This guarantees that ProtocolMismatch is raised for any resolved object
    that does not satisfy the group's contract, regardless of whether the
    caller explicitly passes a protocol.
    Callers that need to skip protocol validation should pass ``protocol=None``
    and suppress the automatic lookup by using ``precedence.resolve()`` directly.

    Args:
        group:        Entry-point group (use ``groups.*`` constants).
        name:         Requested plugin name.  ``None`` means "use default".
        protocol:     Optional runtime_checkable Protocol to validate the result.
                      When omitted, the group's canonical protocol is used.
        capabilities: Optional mapping of capability name to the config-required
                      value (e.g. ``{"dimension": 768}``); conflicting plugins
                      are rejected with a typed ``PluginError`` before use.

    Returns:
        The resolved plugin object.

    Raises:
        PluginConflict:    Two plugins share the same name in this group.
        PluginNotFound:    Requested name is not installed / registered;
                           the error lists the group and installed names.
        ProtocolMismatch:  Resolved object does not satisfy the group protocol
                           (or the explicitly supplied *protocol*).
        PluginError:       Incompatible plugin_api_version or a declared
                           capability that conflicts with configuration.
    """
    effective_protocol: type | None = protocol
    if effective_protocol is None:
        effective_protocol = groups.get_protocol_for_group(group)
    return precedence.resolve(
        group=group, name=name, protocol=effective_protocol, capabilities=capabilities
    )


def register(group: str, name: str, instance: Any) -> None:
    """Register an explicit plugin instance under *group* / *name*.

    Overwrites any previous registration for the same group/name pair.

    Args:
        group:    Entry-point group string.
        name:     Plugin name (must be non-empty).
        instance: The plugin object to register.
    """
    precedence.register(group=group, name=name, instance=instance)


def list_plugins(group: str) -> list[str]:
    """Return the names of all explicitly registered plugins for *group*.

    Does not trigger entry-point discovery.  For installed entry-point
    names, call ``discovery.scan_group(group)`` directly.

    Returns only explicitly registered names, not entry-point names.

    Args:
        group: Entry-point group string.

    Returns:
        List of registered plugin names (may be empty).
    """
    return precedence.list_registered(group=group)


def describe(group: str, name: str) -> dict[str, Any]:
    """Return a description dict for the registered plugin *name* in *group*.

    Args:
        group: Entry-point group string.
        name:  Plugin name.

    Returns:
        Dict with at least ``group``, ``name``, and ``instance`` keys.

    Raises:
        PluginNotFound: If the name is not explicitly registered.
    """
    return precedence.describe(group=group, name=name)
