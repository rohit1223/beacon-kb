"""Docling parsing adapter: raw bytes plus media type to structured sections (Task 02.3).

This module wraps Docling behind a single ``parse()`` function that maps
``(content bytes, media_type)`` from a connector fetch into a typed
``ParsedDocument``: a document title plus ordered sections carrying heading
paths (ancestor headings), body text with case preserved, a section kind
(text, code, table, list), and locators (page number where the format
provides one, plus the Docling anchor reference).

Behavioral guarantees ported from the legacy ``beacon_kb`` parsers:

- Exactly-once extraction: every content item lands in exactly one section.
- Heading-path locators with ``[N]`` disambiguation for repeated paths, so
  two ``## Install`` headings never collide on the same locator.
- Preamble content before the first heading is kept under the ``__root__``
  locator instead of being dropped.
- Recoverable oddities (unrecognized element kinds, partial conversions)
  degrade gracefully and surface as typed ``ParseWarning`` records; they are
  never silently discarded.
- Case, code blocks, and tables are preserved verbatim; code and table
  sections carry their own kinds.

Strict-decode obligation: text media types (``text/*``) with invalid UTF-8
raise a typed ``IngestionError`` naming the source; unknown media types also
raise a typed error. Nothing silently produces an empty document.

Offline behavior: Markdown, HTML, and DOCX use Docling's declarative
backends and need no model artifacts. PDF conversion requires layout and
table-structure model weights that Docling downloads on first use; when those
artifacts are unavailable (for example in a fully offline environment with
``HF_HUB_OFFLINE=1``), the PDF route raises a typed ``IngestionError``
pointing at the Docling model-artifacts documentation instead of hanging.

``PARSER_VERSION`` captures the installed Docling major version plus this
adapter's own version; Task 02.5 folds it into the revision fingerprint so a
parser upgrade forces re-ingestion by construction.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import TYPE_CHECKING

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.code import CodeItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import (
    ListItem,
    SectionHeaderItem,
    TextItem,
    TitleItem,
)

from beacon.errors import IngestionError

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult

# ---------------------------------------------------------------------------
# Parser version
# ---------------------------------------------------------------------------

#: Version of this adapter's mapping logic. Bump whenever the section
#: structure, locator scheme, or warning semantics change.
_ADAPTER_VERSION = 1

_DOCLING_MAJOR = importlib.metadata.version("docling").split(".")[0]

#: Exported parser version: Docling major version plus adapter version.
#: Feeds the Task 02.5 revision fingerprint; any change to either component
#: makes previously ingested revisions incompatible by construction.
PARSER_VERSION = f"docling-{_DOCLING_MAJOR}.beacon-adapter-{_ADAPTER_VERSION}"

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SectionKind(StrEnum):
    """Kind of content a parsed section carries."""

    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"


@dataclass(frozen=True, slots=True)
class ParseWarning:
    """A structured, typed warning emitted during parsing.

    Attributes:
        code:    Short machine-readable code (e.g. ``"unhandled_item_kind"``,
                 ``"partial_conversion"``).
        message: Human-readable description of the issue.
        locator: Section locator where the warning occurred, or ``""`` when
                 the position cannot be attributed.
    """

    code: str
    message: str
    locator: str = ""


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One ordered section of a parsed document.

    Attributes:
        locator:      Unique slash-delimited heading-path locator. Repeated
                      paths get an ordinal suffix (``"Guide/Install[2]"``);
                      pre-heading content uses ``"__root__"``.
        heading_path: Ancestor headings from outermost to innermost, e.g.
                      ``("Guide", "Install")``. Empty for preamble content.
        heading:      Innermost heading text, or ``""`` for preamble content.
        kind:         Section content kind (text, code, table, list).
        text:         Section body with original case and content preserved.
        ordinal:      Zero-based position of this section in document order.
        page_no:      1-based page number for paginated formats (PDF), else
                      ``None``.
        anchor:       Docling self-reference of the first contributing item
                      (e.g. ``"#/texts/3"``), usable as an intra-document
                      anchor. ``""`` when unavailable.
    """

    locator: str
    heading_path: tuple[str, ...]
    heading: str
    kind: SectionKind
    text: str
    ordinal: int
    page_no: int | None = None
    anchor: str = ""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Structured result of parsing one raw document.

    Attributes:
        title:          Document title (first title or heading), or ``""``.
        media_type:     Normalized media type the document was parsed as.
        sections:       Ordered tuple of parsed sections.
        warnings:       Typed warnings for recoverable oddities; empty tuple
                        for a clean parse.
        parser_version: The ``PARSER_VERSION`` this result was produced with.
    """

    title: str
    media_type: str
    sections: tuple[ParsedSection, ...]
    warnings: tuple[ParseWarning, ...]
    parser_version: str


# ---------------------------------------------------------------------------
# Media-type routing
# ---------------------------------------------------------------------------

#: Supported media types and the Docling input format each routes to.
_MEDIA_TYPE_TO_FORMAT: dict[str, InputFormat] = {
    "text/markdown": InputFormat.MD,
    "text/x-markdown": InputFormat.MD,
    "text/plain": InputFormat.MD,
    "text/html": InputFormat.HTML,
    "application/xhtml+xml": InputFormat.HTML,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        InputFormat.DOCX
    ),
    "application/pdf": InputFormat.PDF,
}

_FORMAT_TO_EXTENSION: dict[InputFormat, str] = {
    InputFormat.MD: "md",
    InputFormat.HTML: "html",
    InputFormat.DOCX: "docx",
    InputFormat.PDF: "pdf",
}

_PDF_MODEL_HINT = (
    "PDF conversion requires Docling layout/table model artifacts; in offline "
    "environments pre-download them per the Docling documentation "
    "(https://docling-project.github.io/docling/usage/) and point Docling at "
    "the local artifacts cache."
)

# Lazily constructed converters, one per input format. Building a converter
# performs no model I/O; models are only loaded when a conversion actually
# runs, so the PDF entry is safe to create and only costs anything when a PDF
# is parsed.
# NOTE: not thread-safe; guard with a lock before concurrent ingestion workers
_converters: dict[InputFormat, DocumentConverter] = {}


def _get_converter(fmt: InputFormat) -> DocumentConverter:
    """Return a cached ``DocumentConverter`` restricted to *fmt*."""
    converter = _converters.get(fmt)
    if converter is None:
        converter = DocumentConverter(allowed_formats=[fmt])
        _converters[fmt] = converter
    return converter


def _normalize_media_type(media_type: str) -> str:
    """Strip parameters and whitespace from a media type ('text/html; charset=x')."""
    return media_type.split(";", 1)[0].strip().lower()


# ---------------------------------------------------------------------------
# Locator helpers
# ---------------------------------------------------------------------------

_ROOT_LOCATOR = "__root__"


def _base_locator(heading_path: tuple[str, ...]) -> str:
    """Return the raw slash-delimited locator for a heading path."""
    if not heading_path:
        return _ROOT_LOCATOR
    return "/".join(part.replace("/", "_") for part in heading_path)


def _disambiguate(locator: str, seen: dict[str, int]) -> str:
    """Return a unique locator, suffixing ``[N]`` (N >= 2) for repeats."""
    count = seen.get(locator, 0) + 1
    seen[locator] = count
    if count == 1:
        return locator
    return f"{locator}[{count}]"


def _page_of(item: TextItem | TableItem | CodeItem) -> int | None:
    """Return the 1-based page number of *item*, or ``None`` if unpaginated."""
    if item.prov:
        return item.prov[0].page_no
    return None


# ---------------------------------------------------------------------------
# Docling document -> sections
# ---------------------------------------------------------------------------


class _SectionBuilder:
    """Accumulates Docling items into ordered ``ParsedSection`` records.

    Consecutive plain-text items merge into one text section; consecutive
    list items merge into one list section; code and table items each become
    their own section. A heading change flushes any open run, so every item
    is extracted exactly once under the heading path that was active when it
    appeared.
    """

    def __init__(self) -> None:
        self.sections: list[ParsedSection] = []
        self.warnings: list[ParseWarning] = []
        self._heading_stack: list[tuple[int, str]] = []
        self._seen_locators: dict[str, int] = {}
        self._run_lines: list[str] = []
        self._run_kind: SectionKind | None = None
        self._run_page: int | None = None
        self._run_anchor: str = ""
        self.title: str = ""

    @property
    def heading_path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self._heading_stack)

    def push_heading(self, effective_level: int, text: str) -> None:
        """Enter a new heading: flush the open run and update the path stack."""
        self.flush()
        while self._heading_stack and self._heading_stack[-1][0] >= effective_level:
            self._heading_stack.pop()
        self._heading_stack.append((effective_level, text))
        if not self.title:
            self.title = text

    def add_run_item(self, kind: SectionKind, line: str, item: TextItem) -> None:
        """Append a text/list line to the current run, flushing on kind change."""
        if self._run_kind is not None and self._run_kind != kind:
            self.flush()
        if self._run_kind is None:
            self._run_kind = kind
            self._run_page = _page_of(item)
            self._run_anchor = item.self_ref
        self._run_lines.append(line)

    def add_standalone(
        self, kind: SectionKind, text: str, page_no: int | None, anchor: str
    ) -> None:
        """Emit a self-contained section (code block or table)."""
        self.flush()
        self._emit(kind, text, page_no, anchor)

    def add_warning(self, code: str, message: str) -> None:
        self.warnings.append(
            ParseWarning(
                code=code,
                message=message,
                locator=_base_locator(self.heading_path),
            )
        )

    def flush(self) -> None:
        """Close the open text/list run, emitting it as a section."""
        if self._run_kind is None:
            return
        text = "\n".join(self._run_lines)
        kind = self._run_kind
        page = self._run_page
        anchor = self._run_anchor
        self._run_lines = []
        self._run_kind = None
        self._run_page = None
        self._run_anchor = ""
        if text.strip():
            self._emit(kind, text, page, anchor)

    def _emit(self, kind: SectionKind, text: str, page_no: int | None, anchor: str) -> None:
        path = self.heading_path
        locator = _disambiguate(_base_locator(path), self._seen_locators)
        self.sections.append(
            ParsedSection(
                locator=locator,
                heading_path=path,
                heading=path[-1] if path else "",
                kind=kind,
                text=text,
                ordinal=len(self.sections),
                page_no=page_no,
                anchor=anchor,
            )
        )


def _build_sections(doc: DoclingDocument) -> _SectionBuilder:
    """Walk a ``DoclingDocument`` and map its items into sections.

    Heading nesting uses effective levels on one scale: a ``TitleItem`` is
    level 1 and a ``SectionHeaderItem`` is ``item.level + 1``, which keeps
    relative nesting correct both for formats that emit a title (Markdown H1,
    HTML ``<h1>``) and for formats that emit only section headers with their
    own 1-based levels (DOCX heading styles).
    """
    builder = _SectionBuilder()
    for item, _level in doc.iterate_items():
        if isinstance(item, TitleItem):
            builder.push_heading(1, item.text)
        elif isinstance(item, SectionHeaderItem):
            builder.push_heading(item.level + 1, item.text)
        elif isinstance(item, CodeItem):
            builder.add_standalone(
                SectionKind.CODE, item.text, _page_of(item), item.self_ref
            )
        elif isinstance(item, ListItem):
            if item.text.strip():
                builder.add_run_item(SectionKind.LIST, item.text, item)
        elif isinstance(item, TableItem):
            table_text = item.export_to_markdown(doc)
            if table_text.strip():
                builder.add_standalone(
                    SectionKind.TABLE, table_text, _page_of(item), item.self_ref
                )
        elif isinstance(item, TextItem):
            # Covers plain text plus any TextItem subclass without dedicated
            # handling (formulas, captions, ...): degrade to text content.
            if item.text.strip():
                builder.add_run_item(SectionKind.TEXT, item.text, item)
        else:
            builder.add_warning(
                "unhandled_item_kind",
                f"Skipped unrecognized document item of type "
                f"{type(item).__name__} (label={getattr(item, 'label', '?')}).",
            )
    builder.flush()
    return builder


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse(content: bytes, media_type: str, *, source_uri: str = "") -> ParsedDocument:
    """Parse raw document bytes into a structured ``ParsedDocument``.

    Args:
        content:    Raw document bytes as fetched by a connector.
        media_type: Declared media type (parameters such as ``charset`` are
                    ignored). Must be one of the supported types.
        source_uri: Canonical URI of the source, carried on any raised
                    ``IngestionError`` for attribution.

    Returns:
        A ``ParsedDocument`` with ordered sections, typed warnings, and the
        current ``PARSER_VERSION``. Parsing identical bytes twice yields
        identical results.

    Raises:
        IngestionError: If the media type is unsupported, a text document is
            not valid UTF-8 (strict-decode obligation: no silent replacement),
            the document is corrupt, or - for PDF - required Docling model
            artifacts are unavailable.
    """
    normalized = _normalize_media_type(media_type)
    fmt = _MEDIA_TYPE_TO_FORMAT.get(normalized)
    if fmt is None:
        supported = ", ".join(sorted(_MEDIA_TYPE_TO_FORMAT))
        raise IngestionError(
            f"Unsupported media type {normalized!r}; supported types: {supported}.",
            source_uri=source_uri,
        )

    if normalized.startswith("text/"):
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise IngestionError(
                f"Document declared as {normalized!r} could not be decoded as "
                f"strict UTF-8: {exc}.",
                source_uri=source_uri,
            ) from exc

    stream = DocumentStream(
        name=f"document.{_FORMAT_TO_EXTENSION[fmt]}",
        stream=BytesIO(content),
    )
    try:
        result: ConversionResult = _get_converter(fmt).convert(
            stream, raises_on_error=False
        )
    except Exception as exc:
        hint = f" {_PDF_MODEL_HINT}" if fmt is InputFormat.PDF else ""
        raise IngestionError(
            f"Docling conversion failed for media type {normalized!r}: {exc}.{hint}",
            source_uri=source_uri,
        ) from exc

    if result.status not in (
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ):
        errors = "; ".join(e.error_message for e in result.errors) or "unknown error"
        hint = f" {_PDF_MODEL_HINT}" if fmt is InputFormat.PDF else ""
        raise IngestionError(
            f"Docling could not convert document ({result.status.value}): "
            f"{errors}.{hint}",
            source_uri=source_uri,
        )

    builder = _build_sections(result.document)
    if result.status is ConversionStatus.PARTIAL_SUCCESS:
        errors = "; ".join(e.error_message for e in result.errors)
        builder.warnings.append(
            ParseWarning(
                code="partial_conversion",
                message=f"Docling reported a partial conversion: {errors or 'see logs'}.",
            )
        )

    return ParsedDocument(
        title=builder.title,
        media_type=normalized,
        sections=tuple(builder.sections),
        warnings=tuple(builder.warnings),
        parser_version=PARSER_VERSION,
    )
