"""Parser implementations for beacon-kb.

This package provides:
- ``parsing.base``: Shared ``ParseWarning``, ``ParseResult``, and
  section-construction helpers used by every parser.
- ``parsing.markdown``: Structure-aware Markdown parser (stdlib only).
- ``parsing.html``: HTML parser behind the ``html`` extra (beautifulsoup4 + lxml).
- ``parsing.pdf``: PDF parser behind the ``pdf`` extra (pypdf).

The Markdown parser is registered as a built-in plugin in
``registry/builtins.py``.

The HTML and PDF parsers have optional dependencies; their classes are
importable at any time but calling ``parse()`` without the extra installed
raises ``IngestionError``.  They are NOT registered as built-in instances at
import time because:
  - Registering them would require instantiating them, which is fine, but
    the registry comment convention documents them as "explicit registration"
    components to match the FilesystemConnector pattern.
  - This avoids any future confusion if dependencies ever trigger side effects
    at import time.

To register HtmlParser or PdfParser explicitly::

    from beacon_kb.parsing.html import HtmlParser
    from beacon_kb.registry import precedence, groups

    precedence.register(
        group=groups.PARSERS,
        name="html",
        instance=HtmlParser(),
    )

Importing this module performs no side effects beyond importing ``base``.
"""

from __future__ import annotations

from beacon_kb.parsing.base import ParseResult, ParseWarning, build_locator, make_section

__all__ = [
    "ParseResult",
    "ParseWarning",
    "build_locator",
    "make_section",
]
