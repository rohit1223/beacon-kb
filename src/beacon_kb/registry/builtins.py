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

    # STORES: no built-in default is registered.
    # A store cannot be constructed without a concrete db_path and vector_dim.
    # Registering a throwaway ``:memory:`` / 16-dim SQLiteStore here would mint
    # a live instance that, if resolved, silently reads/writes an isolated
    # in-memory database with the wrong dimension - a footgun.  We leave the
    # group empty so resolve() raises PluginNotFound (its message lists the
    # installed names).  Callers construct and register a store explicitly:
    #
    #   from beacon_kb.storage.sqlite import SQLiteStore
    #   from beacon_kb.registry import precedence, groups
    #   precedence.register(
    #       group=groups.STORES,
    #       name="sqlite",
    #       instance=SQLiteStore(db_path="/path/to/kb.db", vector_dim=768),
    #   )

    # RETRIEVERS: no built-in default is registered.
    # Both BM25SparseRetriever and EmbedderDenseRetriever require a concrete
    # SQLiteStore at construction time.  Registering them bound to a throwaway
    # ``:memory:`` / 16-dim store would silently yield an isolated in-memory
    # instance with the wrong dimension when resolved - the same footgun as the
    # store itself.  We leave the group empty so resolve() raises PluginNotFound.
    # Callers construct a real store first, then build and register the retrievers:
    #
    #   from beacon_kb.storage.sqlite import SQLiteStore
    #   from beacon_kb.retrieval.sparse import BM25SparseRetriever
    #   from beacon_kb.retrieval.dense import EmbedderDenseRetriever
    #   from beacon_kb.registry import precedence, groups
    #
    #   store = SQLiteStore(db_path="/path/to/kb.db", vector_dim=768)
    #   precedence.register(
    #       group=groups.RETRIEVERS,
    #       name="bm25",
    #       instance=BM25SparseRetriever(store=store),
    #   )
    #   precedence.register(
    #       group=groups.RETRIEVERS,
    #       name="dense",
    #       instance=EmbedderDenseRetriever(store=store, embedder=my_embedder, similarity="cosine"),
    #   )
    #
    # Note: resolving a dense retriever requires ``protocol=DenseRetriever`` explicitly
    # because the group's canonical protocol is SparseRetriever.  This documented
    # escape hatch bypasses the automatic SparseRetriever check:
    #   registry.resolve(groups.RETRIEVERS, "dense", protocol=DenseRetriever)

    # FUSION: RRFusion is the built-in rank-based fusion strategy.
    # Registered via register() (the explicit path) so list_plugins() returns it.
    # RRFusion is stateless and safe to register as a default.
    from beacon_kb.retrieval.fusion import RRFusion

    precedence.register(
        group=groups.FUSION,
        name="rrf",
        instance=RRFusion(),
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

    # CHUNKERS: no built-in default is registered.
    # A HeadingAwareChunker is bound to a specific corpus, canonical URI,
    # revision, and pipeline fingerprint at construction.  Registering a
    # ``__default__``-identity instance here would mint chunks with bogus
    # identity if it were ever resolved.  We leave the group empty so resolve()
    # raises PluginNotFound; the sync pipeline builds a fresh chunker per source
    # via its chunker_factory.  Callers that need registry resolution construct
    # and register their own instance explicitly:
    #
    #   from beacon_kb.ingestion.chunking import HeadingAwareChunker
    #   from beacon_kb.registry import precedence, groups
    #   precedence.register(
    #       group=groups.CHUNKERS,
    #       name="heading_aware",
    #       instance=HeadingAwareChunker(corpus=..., canonical_uri=...,
    #                                    revision_id=..., pipeline_fingerprint=...),
    #   )

    # PARSERS: MarkdownParser is the built-in default parser.
    # Import is deferred to avoid circular imports at module load time.
    from beacon_kb.parsing.markdown import MarkdownParser

    precedence.register_builtin(
        group=groups.PARSERS,
        name="markdown",
        instance=MarkdownParser(),
    )
    # HtmlParser and PdfParser are NOT registered as built-in defaults.
    # They depend on optional extras (beautifulsoup4/lxml and pypdf respectively).
    # Registering them at import time would either import the optional dependency
    # eagerly (breaking base-package installs) or require lazy-import wrappers
    # that the current registry has no factory mechanism for.
    # Callers must construct and register them explicitly:
    #
    #   from beacon_kb.parsing.html import HtmlParser
    #   from beacon_kb.registry import precedence, groups
    #   precedence.register(
    #       group=groups.PARSERS,
    #       name="html",
    #       instance=HtmlParser(),
    #   )
    #
    #   from beacon_kb.parsing.pdf import PdfParser
    #   precedence.register(
    #       group=groups.PARSERS,
    #       name="pdf",
    #       instance=PdfParser(),
    #   )


_register_builtins()
