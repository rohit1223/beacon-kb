"""Structure-aware Markdown parser.

Parses Markdown documents into ordered ``Section`` records while preserving:
- Original case in all text, headings, and code.
- Fenced code blocks (triple-backtick and triple-tilde), including the
  language tag and body without any modification.
- ATX headings (``# H1``, ``## H2``, ..., ``###### H6``).
- Tables, links, and inline formatting as raw text.

Design rules:
- NO lowercasing anywhere.
- Content before the first heading is emitted as a ``__root__`` section
  rather than silently dropped.
- Each emitted ``Section`` carries a source URI (via ``source_id``) and at
  least one stable structural locator (the heading path or ``__root__``).

This parser has no optional dependencies; it uses stdlib only and is always
available in the base package.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re

from beacon_kb.errors import IngestionError
from beacon_kb.models import RawDocument, Section
from beacon_kb.parsing.base import (
    ParseResult,
    ParseWarning,
    build_locator,
    disambiguate_locator,
    make_section,
)

# ---------------------------------------------------------------------------
# ATX heading regex: captures level (1-6 #s) and the heading text.
# Setext headings (underline style) are intentionally NOT supported because
# they are rare in technical documentation and their level detection requires
# two-line lookahead, complicating the streaming parser.
# ---------------------------------------------------------------------------
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:#*)?\s*$")

# Fenced code block delimiters (triple-backtick or triple-tilde).
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})")


class MarkdownParser:
    """Parse Markdown documents into structured ``Section`` records.

    This parser is registered as the built-in ``"markdown"`` parser plugin.
    It requires no external dependencies.

    Usage::

        parser = MarkdownParser()
        sections = parser.parse(raw_doc)
    """

    # ------------------------------------------------------------------
    # Parser protocol
    # ------------------------------------------------------------------

    def parse(self, doc: RawDocument) -> list[Section]:
        """Parse a Markdown ``RawDocument`` into an ordered list of sections.

        Args:
            doc: A ``RawDocument`` with ``media_type`` in
                 ``{"text/markdown", "text/plain"}`` (or any text type;
                 the parser does not reject unknown media types so it can
                 serve as a fallback for plain-text content).

        Returns:
            Ordered list of ``Section`` records with heading paths as locators.
            A leading ``__root__`` section is emitted when content appears
            before the first ATX heading.

        Raises:
            ``IngestionError`` if ``doc.content`` is not a string.
        """
        result = self.parse_with_warnings(doc)
        return result.sections

    def parse_with_warnings(self, doc: RawDocument) -> ParseResult:
        """Parse a Markdown document and return sections plus any warnings.

        Identical to ``parse()`` but surfaces non-fatal issues as typed
        ``ParseWarning`` records rather than swallowing them silently.

        Args:
            doc: A ``RawDocument`` to parse.

        Returns:
            ``ParseResult`` with sections and warnings.

        Raises:
            ``IngestionError`` if ``doc.content`` is not a string.
        """
        if not isinstance(doc.content, str):
            raise IngestionError(
                f"MarkdownParser.parse: expected str content, "
                f"got {type(doc.content).__name__!r} for source {doc.source_id!r}"
            )

        lines = doc.content.splitlines(keepends=True)
        warnings: list[ParseWarning] = []
        sections: list[Section] = []

        # Accumulate text lines for the current section.
        current_lines: list[str] = []
        # Heading stack as (level, disambiguated_title) pairs.
        # Maintained so that siblings never become parent/child and children of
        # a disambiguated heading (e.g. Config[2]) carry the correct path.
        heading_stack: list[tuple[int, str]] = []
        # Current heading level (1-6) or 0 for the root (pre-first-heading) section.
        current_level: int = 0
        # Heading text of the current section (empty for root).
        current_heading: str = ""
        # Track whether we're inside a fenced code block.
        in_fence: bool = False
        fence_char: str = ""
        fence_min_len: int = 0
        # Track how many times each raw locator has been emitted so that
        # duplicate headings (same text at the same depth) get distinct
        # locators and therefore distinct SectionIds.
        seen_locators: dict[str, int] = {}

        def _current_path() -> list[str]:
            return [title for _, title in heading_stack]

        def _flush(ordinal: int) -> None:
            """Emit the accumulated section, resetting the buffer."""
            text = "".join(current_lines).rstrip("\n")
            if not text.strip() and not current_heading:
                # No content (or whitespace-only) and no heading: skip silently.
                return
            # The locator for the current section is built from heading_stack.
            # heading_stack already stores disambiguated leaf titles (set at
            # push-time), so build_locator gives us the correct full path
            # including any [N] suffix propagated from parent headings.
            locator = build_locator(_current_path())
            parent_path = _current_path()[:-1] if heading_stack else []
            parent_locator = build_locator(parent_path) if parent_path else ""
            sections.append(
                make_section(
                    source_id=doc.source_id,
                    revision_id=doc.revision_id,
                    locator=locator,
                    heading=current_heading,
                    text=text,
                    ordinal=ordinal,
                    parent_locator=parent_locator,
                    depth=current_level,
                )
            )

        ordinal_counter = 0

        for line in lines:
            stripped = line.rstrip("\n").rstrip("\r")

            # -----------------------------------------------------------
            # Fenced code block tracking (must happen before heading check
            # so that headings inside code blocks are NOT parsed).
            # -----------------------------------------------------------
            if in_fence:
                current_lines.append(line)
                # A closing fence must use the same character and be at least
                # as long as the opening fence.
                fence_match = _FENCE_OPEN_RE.match(stripped)
                if fence_match and fence_match.group(1)[0] == fence_char:
                    if len(fence_match.group(1)) >= fence_min_len:
                        in_fence = False
                continue

            fence_match = _FENCE_OPEN_RE.match(stripped)
            if fence_match:
                in_fence = True
                fence_char = fence_match.group(1)[0]
                fence_min_len = len(fence_match.group(1))
                current_lines.append(line)
                continue

            # -----------------------------------------------------------
            # ATX heading detection.
            # -----------------------------------------------------------
            heading_match = _ATX_HEADING_RE.match(stripped)
            if heading_match:
                # Flush current section before starting the new one.
                _flush(ordinal_counter)
                if current_lines or current_heading:
                    ordinal_counter += 1
                current_lines = []

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)

                # Maintain heading_stack as (level, disambiguated_title) pairs.
                # Pop any entries at the same or deeper level, then push the new one.
                # This correctly handles siblings and level jumps without creating
                # spurious parent/child relationships.
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
                continue

            # -----------------------------------------------------------
            # Regular content line.
            # -----------------------------------------------------------
            current_lines.append(line)

        # Flush the final section.
        _flush(ordinal_counter)

        return ParseResult(sections=sections, warnings=tuple(warnings))
