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

    # CHUNKERS: HeadingAwareChunker is the built-in default chunker.
    # The default instance is registered with sentinel identity values so the
    # 'heading_aware' name is discoverable via resolve(); production callers
    # always construct their own HeadingAwareChunker with the live corpus,
    # canonical URI, revision, and pipeline fingerprint for their build run.
    # Import is deferred to avoid circular imports at module load time.
    from beacon_kb.ingestion.chunking import HeadingAwareChunker

    precedence.register_builtin(
        group=groups.CHUNKERS,
        name="heading_aware",
        instance=HeadingAwareChunker(
            corpus="__default__",
            canonical_uri="__default__",
            revision_id="__default__",
            pipeline_fingerprint="__default__",
            max_tokens=512,
            overlap_tokens=64,
        ),
    )

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
