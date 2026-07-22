"""Register first-party (built-in) components through the shared registry path.

Every built-in is registered via the same ``register`` and
``register_builtin`` calls that a third-party plugin would use.  There is
no privileged code path.

This module is imported eagerly by ``registry/__init__.py`` so the
built-ins are always available before any user code calls ``resolve()``.

Importing this module performs no side effects beyond registering the
built-ins into the in-memory registry (which is expected behaviour).
"""

from __future__ import annotations

from beacon_kb.registry import groups, precedence
from beacon_kb.tokens import HeuristicTokenCounter


def _register_builtins() -> None:
    """Register all first-party components into the shared registry."""
    # TOKEN_COUNTERS: HeuristicTokenCounter is the built-in default.
    precedence.register_builtin(
        group=groups.TOKEN_COUNTERS,
        name="heuristic",
        instance=HeuristicTokenCounter(),
    )


_register_builtins()
