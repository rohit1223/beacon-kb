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

    _store = SQLiteStore(db_path=":memory:", vector_dim=16)
    precedence.register_builtin(
        group=groups.STORES,
        name="sqlite",
        instance=_store,
    )

    # RETRIEVERS: sparse and dense built-in retrievers.
    #
    # Convention: the beacon_kb.retrievers group's default protocol maps to
    # SparseRetriever (see registry/groups.py).  Anyone resolving a dense
    # retriever must pass ``protocol=DenseRetriever`` explicitly:
    #   registry.resolve(groups.RETRIEVERS, "dense", protocol=DenseRetriever)
    # This documented escape hatch bypasses the automatic SparseRetriever check.
    # Do NOT change the group-protocol map; document the convention here only.
    #
    # Both retrievers are registered via ``register()`` (the explicit path) so
    # that ``list_plugins()`` and ``resolve(group, name)`` both work - the same
    # path a third-party plugin would use.  register_builtin() is not used here
    # because the built-in slot holds only one entry per group; explicit
    # registration supports multiple named plugins in the same group.
    from beacon_kb.retrieval.dense import EmbedderDenseRetriever
    from beacon_kb.retrieval.sparse import BM25SparseRetriever

    precedence.register(
        group=groups.RETRIEVERS,
        name="bm25",
        instance=BM25SparseRetriever(store=_store),
    )

    # Dense retriever with no embedder (sparse-only degraded mode default).
    # Callers that supply an embedder construct EmbedderDenseRetriever directly.
    precedence.register(
        group=groups.RETRIEVERS,
        name="dense",
        instance=EmbedderDenseRetriever(store=_store, embedder=None, similarity="cosine"),
    )


_register_builtins()
