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

    # STORES: SQLiteStore is the built-in default store.
    # Import is deferred to avoid circular imports at module load time.
    from beacon_kb.storage.sqlite import SQLiteStore

    precedence.register_builtin(
        group=groups.STORES,
        name="sqlite",
        instance=SQLiteStore(db_path=":memory:", vector_dim=16),
    )

    # CONNECTORS: Register first-party connectors.
    # Import is deferred to avoid circular imports at module load time.
    from beacon_kb.connectors.memory import MemoryConnector

    precedence.register_builtin(
        group=groups.CONNECTORS,
        name="memory",
        instance=MemoryConnector(),
    )
    # FilesystemConnector is NOT registered as a built-in default.
    # It requires caller-supplied root directory, corpus name, and glob
    # patterns at construction time; a default instance would bind to the
    # process CWD at import time and silently produce CWD-sensitive results.
    # Callers must construct and register a FilesystemConnector explicitly:
    #
    #   from beacon_kb.connectors.filesystem import FilesystemConnector
    #   from beacon_kb.registry import precedence, groups
    #   precedence.register(
    #       group=groups.CONNECTORS,
    #       name="filesystem",
    #       instance=FilesystemConnector(root=..., corpus=..., patterns=[...]),
    #   )


_register_builtins()
