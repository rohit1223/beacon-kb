"""Canonical entry-point group name constants and protocol map for beacon-kb.

Each constant matches a ``[project.entry-points."<group>"]`` header in
``pyproject.toml`` exactly.  The ``get_protocol_for_group()`` function maps
every shipped group to its target runtime-checkable Protocol class.

Importing this module performs no side effects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Group name constants
# ---------------------------------------------------------------------------

CONNECTORS: str = "beacon_kb.connectors"
PARSERS: str = "beacon_kb.parsers"
CHUNKERS: str = "beacon_kb.chunkers"
EMBEDDERS: str = "beacon_kb.embedders"
STORES: str = "beacon_kb.stores"
RETRIEVERS: str = "beacon_kb.retrievers"
FUSION: str = "beacon_kb.fusion"
RERANKERS: str = "beacon_kb.rerankers"
GENERATORS: str = "beacon_kb.generators"
TOKEN_COUNTERS: str = "beacon_kb.token_counters"
PLANNERS: str = "beacon_kb.planners"
GRADERS: str = "beacon_kb.graders"
ROUTERS: str = "beacon_kb.routers"

ALL_GROUPS: tuple[str, ...] = (
    CONNECTORS,
    PARSERS,
    CHUNKERS,
    EMBEDDERS,
    STORES,
    RETRIEVERS,
    FUSION,
    RERANKERS,
    GENERATORS,
    TOKEN_COUNTERS,
    PLANNERS,
    GRADERS,
    ROUTERS,
)


def get_protocol_for_group(group: str) -> type | None:
    """Return the runtime-checkable Protocol class for *group*, or None.

    Defers the import of protocols to avoid circular-import issues and keep
    the group constants importable without loading all of beacon_kb.

    Args:
        group: One of the ``CONNECTORS``, ``PARSERS``, ... constants.

    Returns:
        The Protocol class, or None if the group has no protocol mapping.
    """
    from beacon_kb.protocols import (
        Chunker,
        Connector,
        CorpusRouter,
        Embedder,
        EvidenceGrader,
        Fusion,
        Generator,
        Parser,
        QueryPlanner,
        Reranker,
        SparseRetriever,
        Store,
        TokenCounter,
    )

    _group_protocols: dict[str, type] = {
        CONNECTORS: Connector,
        PARSERS: Parser,
        CHUNKERS: Chunker,
        EMBEDDERS: Embedder,
        STORES: Store,
        RETRIEVERS: SparseRetriever,  # default retriever protocol
        FUSION: Fusion,
        RERANKERS: Reranker,
        GENERATORS: Generator,
        TOKEN_COUNTERS: TokenCounter,
        PLANNERS: QueryPlanner,
        GRADERS: EvidenceGrader,
        ROUTERS: CorpusRouter,
    }
    return _group_protocols.get(group)
