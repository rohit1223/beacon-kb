"""Structure-aware HTML parser (requires the ``html`` extra).

This parser is behind an optional extra.
Import of ``beautifulsoup4`` and ``lxml`` is deferred to inside the ``parse()``
method so that importing *this module* never fails when the extra is absent.
The base-package import always succeeds; an ``IngestionError`` is raised only
when ``parse()`` is called without the extra installed.

Design rules:
- Generic extraction (headings, paragraphs, tables, code) is implemented here.
- Site-specific cleanup is delegated to a caller-supplied ``cleanup_hook``
  callable so that generic extraction stays separate from site-specific logic.
- NO lowercasing of content; original case is preserved throughout.
- Every emitted ``Section`` carries the source URI and a stable heading-path
  locator (or ``__root__`` for pre-heading content).

Importing this module performs no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from beacon_kb.errors import IngestionError
from beacon_kb.models import RawDocument, Section
from beacon_kb.parsing.base import (
    ParseResult,
    ParseWarning,
    build_locator,
    disambiguate_locator,
    make_section,
)

# Type alias for an optional cleanup hook.
CleanupHook = Callable[[Any], None]
"""A callable that receives a BeautifulSoup ``Tag`` object and mutates it
in place to strip site-specific boilerplate (nav bars, footers, cookie banners,
etc.) before the generic extractor runs.

The hook receives the root ``<body>`` tag (or the document root if no body is
found).  It must not return a value; it modifies the tree in place."""

_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# Tags whose subtrees are always skipped (no user-visible text content).
_INLINE_SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "noscript", "template"}
)

# Inline formatting tags that carry text but are not block containers.
# When these appear as direct children of a structural container, their text
# is consumed inline (with link-handling for <a>) rather than being silently
# dropped or recursed into as a container.
_INLINE_TEXT_TAGS: frozenset[str] = frozenset(
    {"span", "strong", "em", "b", "i", "u", "mark", "small", "abbr", "cite", "q"}
)

# Block-level content tags: each is consumed as a single text chunk without
# recursing into children.  This prevents double-emission of text that appears
# in both the block element (e.g. <p>) and its inline descendants (e.g. <a>).
_BLOCK_CONTENT_TAGS: frozenset[str] = frozenset(
    {"p", "li", "td", "th", "dd", "dt", "figcaption", "caption"}
)


def _heading_level(tag_name: str) -> int:
    """Return the integer level (1-6) of an HTML heading tag name."""
    return int(tag_name[1])


def _extract_text_with_inline_links(element: Any) -> str:
    """Extract text from a block-content element, substituting links inline.

    For each ``<a href="...">anchor text</a>`` inside *element*, the output
    replaces the anchor with ``anchor text [href]`` at its natural position
    rather than appending all hrefs at the end.  This keeps each URL adjacent
    to the anchor text it belongs to.

    Args:
        element: A BeautifulSoup ``Tag`` representing a block-content element.

    Returns:
        A string with inline link substitutions applied.
    """
    from bs4.element import NavigableString, Tag

    parts: list[str] = []
    for child in element.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "a":
                href = child.get("href", "")
                anchor_text = child.get_text(separator=" ", strip=True)
                if href and anchor_text:
                    parts.append(f"{anchor_text} [{href}]")
                elif anchor_text:
                    parts.append(anchor_text)
            else:
                # Inline formatting tag or other nested element: recurse.
                parts.append(_extract_text_with_inline_links(child))
    raw = " ".join(parts)
    # Collapse runs of whitespace that accumulate from the string joining.
    import re
    return re.sub(r" +", " ", raw).strip()


class HtmlParser:
    """Parse HTML documents into structured ``Section`` records.

    Requires the ``html`` extra (``beautifulsoup4`` + ``lxml``).
    Importing this class is always safe; the dependency is loaded lazily
    inside ``parse()`` to keep the base package dependency-free.

    Args:
        cleanup_hook: Optional callable that receives the root BeautifulSoup
            ``Tag`` and mutates it in place to remove site-specific elements
            before generic extraction.  Use this hook to strip navbars, footers,
            cookie banners, or other non-content elements without modifying this
            generic parser.
        parser_backend: The BeautifulSoup parser backend to use.
            Defaults to ``"lxml"`` (fastest, strict HTML5 parse).
            Pass ``"html.parser"`` for stdlib-only fallback when lxml is absent.
    """

    def __init__(
        self,
        *,
        cleanup_hook: CleanupHook | None = None,
        parser_backend: str = "lxml",
    ) -> None:
        self._cleanup_hook: CleanupHook | None = cleanup_hook
        self._parser_backend: str = parser_backend

    # ------------------------------------------------------------------
    # Parser protocol
    # ------------------------------------------------------------------

    def parse(self, doc: RawDocument) -> list[Section]:
        """Parse an HTML ``RawDocument`` into an ordered list of sections.

        Args:
            doc: A ``RawDocument`` with ``media_type == "text/html"`` (or any
                 HTML-like content; the parser does not reject unknown types).

        Returns:
            Ordered list of ``Section`` records.

        Raises:
            ``IngestionError`` if the ``html`` extra (beautifulsoup4 + lxml)
            is not installed, or if ``doc.content`` is not a string.
        """
        result = self.parse_with_warnings(doc)
        return result.sections

    def parse_with_warnings(self, doc: RawDocument) -> ParseResult:
        """Parse an HTML document and return sections plus any warnings.

        Identical to ``parse()`` but surfaces non-fatal issues as typed
        ``ParseWarning`` records.

        Args:
            doc: A ``RawDocument`` to parse.

        Returns:
            ``ParseResult`` with sections and warnings.

        Raises:
            ``IngestionError`` if the ``html`` extra is not installed or
            if ``doc.content`` is not a string.
        """
        try:
            from bs4 import BeautifulSoup, Tag
            from bs4.element import NavigableString
        except ModuleNotFoundError as exc:
            raise IngestionError(
                "HtmlParser requires the 'html' extra: "
                "install beacon-kb with `pip install beacon-kb[html]` or "
                "`uv add beacon-kb[html]`."
            ) from exc

        if not isinstance(doc.content, str):
            raise IngestionError(
                f"HtmlParser.parse: expected str content, "
                f"got {type(doc.content).__name__!r} for source {doc.source_id!r}"
            )

        soup = BeautifulSoup(doc.content, self._parser_backend)
        warnings: list[ParseWarning] = []

        # Site-specific cleanup runs first, before any extraction.
        root = soup.find("body") or soup
        if self._cleanup_hook is not None:
            self._cleanup_hook(root)

        sections: list[Section] = []
        current_lines: list[str] = []
        # heading_stack holds (level, disambiguated_title) pairs, maintained
        # so that siblings never become parent/child and children of a
        # disambiguated heading (e.g. Config[2]) carry the correct path.
        heading_stack: list[tuple[int, str]] = []
        current_level: int = 0
        current_heading: str = ""
        ordinal_counter: int = 0
        # Track how many times each raw locator has been emitted so that
        # duplicate headings (same text at the same depth) get distinct
        # locators and therefore distinct SectionIds.
        seen_locators: dict[str, int] = {}

        def _current_path() -> list[str]:
            return [title for _, title in heading_stack]

        def _flush() -> None:
            nonlocal ordinal_counter
            text = "\n".join(current_lines).strip()
            if not text and not current_heading:
                return
            # heading_stack already stores disambiguated leaf titles (set at
            # push-time), so build_locator gives us the correct full path.
            locator = build_locator(_current_path())
            parent_path = _current_path()[:-1] if heading_stack else []
            parent_locator = build_locator(parent_path) if parent_path else ""
            if not text and current_heading:
                warnings.append(
                    ParseWarning(
                        code="html_empty_section",
                        message=(
                            f"Section heading {current_heading!r} has no body text."
                        ),
                        locator=locator,
                    )
                )
            sections.append(
                make_section(
                    source_id=doc.source_id,
                    revision_id=doc.revision_id,
                    locator=locator,
                    heading=current_heading,
                    text=text,
                    ordinal=ordinal_counter,
                    parent_locator=parent_locator,
                    depth=current_level,
                )
            )
            ordinal_counter += 1

        # Walk the document in source order, collecting text per section.
        # We use a queue-based BFS that processes each block-level container
        # exactly once: when we encounter a block that we can extract text
        # from directly (e.g. <pre>, <p>, <li>), we call get_text() on it
        # and do NOT recurse into its children.  This prevents double-emission
        # of text that appears in both a parent element (e.g. <p>) and its
        # descendants (e.g. <a> inside the <p>).
        #
        # The traversal queue starts with the direct children of root.
        # For each element we either:
        #   a) consume it as a content block (and skip its subtree), OR
        #   b) treat it as a structural container and enqueue its children.

        queue: list[Any] = list(root.children)
        while queue:
            element = queue.pop(0)

            # Handle bare NavigableStrings (text nodes directly inside
            # structural containers like body, div, blockquote, etc.).
            if isinstance(element, NavigableString):
                text_content = str(element)
                if text_content.strip():
                    current_lines.append(text_content.strip())
                continue

            if not isinstance(element, Tag):
                continue
            tag_name = element.name
            if tag_name is None or tag_name in _INLINE_SKIP_TAGS:
                continue

            if tag_name in _HEADING_TAGS:
                _flush()
                current_lines = []
                level = _heading_level(tag_name)
                heading_text = element.get_text(separator=" ", strip=True)
                # Maintain heading_stack as (level, disambiguated_title) pairs.
                # Pop any entries at the same or deeper level, then push the new one.
                # This correctly handles siblings and level jumps.
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                # Disambiguate at push-time so that child headings inherit the
                # correct disambiguated segment (e.g. Config[2]) in their path.
                candidate_path = [title for _, title in heading_stack] + [heading_text]
                raw_locator = build_locator(candidate_path)
                disambiguated_locator = disambiguate_locator(raw_locator, seen_locators)
                # Extract the disambiguated leaf segment from the full locator.
                disambiguated_leaf = disambiguated_locator.split("/")[-1]
                heading_stack.append((level, disambiguated_leaf))
                current_level = level
                current_heading = heading_text
                # Headings are leaf-consumed; do not recurse into their children.

            elif tag_name == "pre":
                # Preserve code blocks verbatim via the <pre> element.
                # Do NOT recurse into <code> children to avoid double-emission.
                code_text = element.get_text(separator="\n", strip=False)
                if code_text.strip():
                    current_lines.append(code_text)

            elif tag_name == "code":
                # A bare <code> (not inside <pre>) - treat as inline code snippet.
                # Only reached when <code> is a direct descendant of a container
                # that was NOT already consumed as a <pre>.
                code_text = element.get_text(separator="\n", strip=False)
                if code_text.strip():
                    current_lines.append(code_text)

            elif tag_name in _BLOCK_CONTENT_TAGS:
                # Block-level content elements: extract all their text at once
                # (including any nested inline elements like <a>, <strong>, etc.)
                # with inline link substitution so each URL appears adjacent to
                # the anchor text it belongs to.
                text_content = _extract_text_with_inline_links(element)
                if text_content:
                    current_lines.append(text_content)

            elif tag_name == "a":
                # Bare anchor tag directly inside a structural container (not
                # wrapped in a <p> or other block-content element).
                href = element.get("href", "")
                anchor_text = element.get_text(separator=" ", strip=True)
                if anchor_text:
                    if href:
                        current_lines.append(f"{anchor_text} [{href}]")
                    else:
                        current_lines.append(anchor_text)

            elif tag_name in _INLINE_TEXT_TAGS:
                # Inline formatting tag directly inside a structural container.
                # Consume its text so it is not silently dropped.
                inline_text = element.get_text(separator=" ", strip=True)
                if inline_text:
                    current_lines.append(inline_text)

            else:
                # Structural container (div, section, article, ul, ol, table,
                # thead, tbody, tr, blockquote, figure, etc.): do not consume
                # as a single block; instead enqueue direct children so they
                # are processed individually.
                queue[:0] = list(element.children)

        # Warn if no headings were detected.
        if not any(s.depth > 0 for s in sections) and not heading_stack:
            warnings.append(
                ParseWarning(
                    code="html_missing_heading",
                    message=(
                        "No heading elements (h1-h6) found in HTML document; "
                        "all content emitted as '__root__' section."
                    ),
                    locator="__root__",
                )
            )

        # Flush the final accumulated section.
        _flush()

        return ParseResult(sections=sections, warnings=tuple(warnings))
