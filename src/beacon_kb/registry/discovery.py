"""Lazy entry-point discovery for beacon-kb plugins.

Discovery is intentionally deferred: ``importlib.metadata.entry_points``
is called only on the *first* call to ``scan_group()``.  Until that call
``has_scanned()`` returns ``False`` so tests can assert that merely
importing the registry does not trigger a scan.

An installed-but-unused plugin is never imported (its module is loaded
only when its entry point is explicitly resolved via ``load()``).

``scan_group()`` returns a mapping of name -> list[EntryPoint] for the
requested group.  Duplicate names are intentionally preserved so the
precedence resolver can raise ``PluginConflict``.

Importing this module performs no side effects.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points

# ---------------------------------------------------------------------------
# Module-level state (lazy)
# ---------------------------------------------------------------------------

_scanned: bool = False
"""True after the first call to ``scan_group()``."""


def has_scanned() -> bool:
    """Return True if entry-point discovery has been triggered at least once.

    This predicate stays False until the first call to ``scan_group()``,
    allowing tests to assert that importing ``beacon_kb.registry`` does
    not eagerly scan entry points.
    """
    return _scanned


def reset_scan_state() -> None:
    """Reset the scan flag to False.

    Intended for use in tests only.  Do not call in production code.
    """
    global _scanned
    _scanned = False


def scan_group(group: str) -> dict[str, list[EntryPoint]]:
    """Scan the given entry-point group and return a name -> [EntryPoint] map.

    Marks discovery as having run (``has_scanned()`` becomes True after
    the first call to this function for any group).

    Each key is an entry-point name; the value is a list of all ``EntryPoint``
    objects that registered under that name.  When the list has length > 1
    there is a conflict; the caller is responsible for raising
    ``PluginConflict``.

    Plugin modules are NOT imported here; call ``entry_point.load()``
    explicitly to load a specific plugin.

    Args:
        group: The entry-point group string (e.g. ``beacon_kb.connectors``).

    Returns:
        Dict mapping each entry-point name to its list of ``EntryPoint``
        objects found in the given group.
    """
    global _scanned
    _scanned = True

    eps = entry_points(group=group)
    result: dict[str, list[EntryPoint]] = {}
    for ep in eps:
        result.setdefault(ep.name, []).append(ep)
    return result
