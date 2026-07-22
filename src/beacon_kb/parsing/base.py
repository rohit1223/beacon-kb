"""Shared section and provenance helpers for all beacon-kb parsers.

This module provides:
- ``ParseWarning``: a typed, structured warning record (never a bare string).
- ``ParseResult``: a container pairing sections with accumulated warnings.
- ``make_section``: a helper that constructs a ``Section`` record with a
  content-addressed ``SectionId`` from minimal caller-supplied fields.
- ``build_locator``: a helper that turns a heading path list into a stable
  slash-delimited locator string.

Importing this module performs no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from beacon_kb.models import (
    RevisionId,
    Section,
    SectionId,
    SourceId,
    make_section_id,
)

# ---------------------------------------------------------------------------
# Typed warning record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParseWarning:
    """A structured, typed warning emitted during document parsing.

    Warnings represent non-fatal issues encountered while parsing a document:
    ambiguous heading detection, possible header/footer misclassification, or
    content that could not be cleanly attributed to a section.

    Callers must never silently drop warnings; they must surface them via the
    ``ParseResult`` returned by each parser's ``parse()`` method.

    Attributes:
        code:    Short machine-readable code identifying the warning category.
                 Examples: ``"pdf_heading_heuristic"``, ``"html_missing_heading"``,
                 ``"md_empty_section"``.
        message: Human-readable description of the issue.
        locator: The section locator (heading path) where the warning occurred,
                 or an empty string if the position cannot be determined.
        details: Optional auxiliary string carrying parser-specific context
                 (e.g. the raw text line that triggered a PDF heuristic).
    """

    code: str
    message: str
    locator: str = ""
    details: str = ""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Container pairing a list of sections with accumulated parse warnings.

    Parsers that detect non-fatal issues (ambiguous PDF headings, missing HTML
    heading structure, etc.) return both the best-effort sections and the
    typed warnings so callers can decide how to handle each case.

    Attributes:
        sections: Ordered list of ``Section`` records produced by the parser.
        warnings: Tuple of typed ``ParseWarning`` records for any non-fatal issues.
    """

    sections: list[Section]
    warnings: tuple[ParseWarning, ...]


# ---------------------------------------------------------------------------
# Locator helpers
# ---------------------------------------------------------------------------


def disambiguate_locator(locator: str, seen_locators: dict[str, int]) -> str:
    """Return a deterministic, unique locator for repeated heading paths.

    When a document contains two sections with identical heading paths (e.g.
    two ``## Configuration`` headings at the same nesting level), their raw
    locators collide and a later store upsert would silently overwrite the
    earlier section.

    This helper assigns an ordinal suffix ``[2]``, ``[3]``, ... to any locator
    that has been seen before in the current parse, leaving the *first*
    occurrence unchanged (no suffix).

    ``seen_locators`` is a mutable dict mapping each raw locator to the count
    of times it has been emitted so far.  Callers must pass the same dict
    across every call within a single document parse.

    Args:
        locator:       The raw locator string produced by ``build_locator()``.
        seen_locators: Per-parse dict tracking emission counts.

    Returns:
        The original locator string for the first occurrence, or a string of
        the form ``"locator[N]"`` (N >= 2) for subsequent occurrences.
    """
    count = seen_locators.get(locator, 0) + 1
    seen_locators[locator] = count
    if count == 1:
        return locator
    return f"{locator}[{count}]"


def build_locator(heading_path: list[str]) -> str:
    """Return a stable slash-delimited locator from a heading path list.

    Each element of ``heading_path`` is a heading title (original case is
    preserved; no lowercasing is applied).  The first element is the document
    root heading; nested headings extend the path.

    An empty ``heading_path`` returns ``"__root__"`` to mark pre-heading
    content that belongs to no heading.

    Args:
        heading_path: Ordered list of heading titles from outermost to innermost.

    Returns:
        Slash-delimited locator string, e.g. ``"Installation/Basic Example"``.
    """
    if not heading_path:
        return "__root__"
    # Replace slashes inside heading text with underscores to avoid ambiguity
    # in the locator path, then join with "/".
    sanitised = [part.replace("/", "_") for part in heading_path]
    return "/".join(sanitised)


# ---------------------------------------------------------------------------
# Section construction helper
# ---------------------------------------------------------------------------


def make_section(
    *,
    source_id: SourceId,
    revision_id: RevisionId,
    locator: str,
    heading: str,
    text: str,
    ordinal: int,
    parent_locator: str = "",
    depth: int = 0,
) -> Section:
    """Construct a ``Section`` record with a content-addressed ``SectionId``.

    The ``SectionId`` is derived deterministically from ``source_id``,
    ``revision_id``, and ``locator`` via ``make_section_id()``, so identical
    content across runs produces the same ID.

    Args:
        source_id:      The ``SourceId`` of the parent source document.
        revision_id:    The ``RevisionId`` of the specific revision being parsed.
        locator:        The stable section locator (heading path or anchor).
        heading:        The section heading text (case preserved).
        text:           Full section text body (case preserved, code intact).
        ordinal:        Zero-based position of this section within the document.
        parent_locator: Locator of the parent section, or empty string for roots.
        depth:          Heading depth (0 = document root, 1 = H1, 2 = H2, ...).

    Returns:
        A frozen ``Section`` record.
    """
    section_id: SectionId = make_section_id(
        source_id=str(source_id),
        revision_id=str(revision_id),
        locator=locator,
    )
    return Section(
        id=section_id,
        source_id=source_id,
        revision_id=revision_id,
        locator=locator,
        heading=heading,
        text=text,
        ordinal=ordinal,
        parent_locator=parent_locator,
        depth=depth,
    )
