"""Page-aware PDF parser (requires the ``pdf`` extra).

This parser is behind an optional extra.
Import of ``pypdf`` is deferred to inside the ``parse()`` method so that
importing *this module* never fails when the extra is absent.
The base-package import always succeeds; an ``IngestionError`` is raised only
when ``parse()`` is called without the extra installed.

Design rules:
- One ``Section`` per PDF page.
  Each section carries a ``page:<n>`` locator (1-indexed) as the stable
  structural locator, since PDF text streams lack reliable heading structure.
- Heuristic heading/header/footer classification (by font size, position, or
  line count) emits a typed ``ParseWarning`` with code
  ``"pdf_heading_heuristic"`` rather than silently guessing.
- NO lowercasing.
- Empty pages emit a ``ParseWarning`` with code ``"pdf_empty_page"`` rather
  than silently producing an empty section.

Importing this module performs no side effects.
"""

from __future__ import annotations

import io

from beacon_kb.errors import IngestionError
from beacon_kb.models import RawDocument, Section
from beacon_kb.parsing.base import ParseResult, ParseWarning, make_section

# Maximum line length (in characters) for a line to be considered a candidate
# page heading.  Lines longer than this threshold are too verbose to be titles.
_MAX_HEADING_LINE_CHARS: int = 120


class PdfParser:
    """Parse PDF documents into page-level ``Section`` records.

    Requires the ``pdf`` extra (``pypdf``).
    Importing this class is always safe; ``pypdf`` is loaded lazily inside
    ``parse()`` to keep the base package dependency-free.

    One ``Section`` is emitted per PDF page.
    Pages that appear to contain only a running header, footer, or page number
    (heuristic: fewer than ``min_body_chars`` non-whitespace characters) emit
    a ``ParseWarning`` with code ``"pdf_possible_header_footer"`` rather than
    silently producing a section with minimal content.

    Args:
        min_body_chars: Minimum number of non-whitespace characters required for
            a page to be classified as body content.
            Pages below this threshold trigger a typed warning.
            Defaults to 50.
    """

    def __init__(self, *, min_body_chars: int = 50) -> None:
        self._min_body_chars: int = min_body_chars

    # ------------------------------------------------------------------
    # Parser protocol
    # ------------------------------------------------------------------

    def parse(self, doc: RawDocument) -> list[Section]:
        """Parse a PDF ``RawDocument`` into an ordered list of page sections.

        The ``doc.content`` must be a base64-encoded string of the raw PDF
        bytes (as produced by connectors that read binary files).  If it is
        a plain string that starts with ``%PDF``, it is treated as raw PDF
        bytes decoded as Latin-1 (lossy but round-trippable).

        Args:
            doc: A ``RawDocument`` with ``media_type == "application/pdf"``.

        Returns:
            Ordered list of ``Section`` records, one per non-empty PDF page.

        Raises:
            ``IngestionError`` if the ``pdf`` extra (pypdf) is not installed,
            or if the document bytes cannot be parsed as a PDF.
        """
        result = self.parse_with_warnings(doc)
        return result.sections

    def parse_with_warnings(self, doc: RawDocument) -> ParseResult:
        """Parse a PDF document and return sections plus typed warnings.

        Identical to ``parse()`` but always surfaces non-fatal issues as
        typed ``ParseWarning`` records so callers can audit PDF extraction
        quality without relying on log output.

        Args:
            doc: A ``RawDocument`` to parse.

        Returns:
            ``ParseResult`` with sections and warnings.

        Raises:
            ``IngestionError`` if ``pypdf`` is not installed or if the PDF
            cannot be parsed.
        """
        try:
            import pypdf
        except ModuleNotFoundError as exc:
            raise IngestionError(
                "PdfParser requires the 'pdf' extra: "
                "install beacon-kb with `pip install beacon-kb[pdf]` or "
                "`uv add beacon-kb[pdf]`."
            ) from exc

        if not isinstance(doc.content, str):
            raise IngestionError(
                f"PdfParser.parse: expected str content, "
                f"got {type(doc.content).__name__!r} for source {doc.source_id!r}"
            )

        pdf_bytes = self._decode_content(doc)
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        except Exception as exc:
            raise IngestionError(
                f"PdfParser.parse: failed to open PDF for source "
                f"{doc.source_id!r}: {exc}"
            ) from exc

        warnings: list[ParseWarning] = []
        sections: list[Section] = []

        for page_index, page in enumerate(reader.pages):
            page_num = page_index + 1
            locator = f"page:{page_num}"

            try:
                page_text: str = page.extract_text() or ""
            except Exception as exc:
                warnings.append(
                    ParseWarning(
                        code="pdf_extraction_error",
                        message=f"Failed to extract text from page {page_num}: {exc}",
                        locator=locator,
                        details=str(exc),
                    )
                )
                continue

            # Warn on empty or near-empty pages rather than silently skipping.
            non_ws = len(page_text.replace(" ", "").replace("\n", "").replace("\t", ""))
            if non_ws == 0:
                warnings.append(
                    ParseWarning(
                        code="pdf_empty_page",
                        message=f"Page {page_num} extracted no text; page may be image-only.",
                        locator=locator,
                    )
                )
                continue

            if non_ws < self._min_body_chars:
                warnings.append(
                    ParseWarning(
                        code="pdf_possible_header_footer",
                        message=(
                            f"Page {page_num} has very few non-whitespace characters "
                            f"({non_ws} < {self._min_body_chars}); may be a running "
                            "header, footer, or page number rather than body content. "
                            "Review the extracted text."
                        ),
                        locator=locator,
                        details=page_text[:200],
                    )
                )
                # Still emit the section; the warning surfaces the uncertainty.
                # Callers can filter sections by locator or warning code.

            # Heuristic heading detection: first non-empty line may be a heading.
            # We surface this as a warning with code "pdf_heading_heuristic" so
            # callers know the heading field is a best-effort guess, not a
            # structurally authoritative value.
            lines = [ln for ln in page_text.splitlines() if ln.strip()]
            heading = ""
            if lines:
                candidate = lines[0].strip()
                # A line is treated as a possible heading if it is short enough
                # (< _MAX_HEADING_LINE_CHARS) and does not end with punctuation
                # that typically ends sentences.
                is_possible_heading = (
                    len(candidate) < _MAX_HEADING_LINE_CHARS
                    and not candidate.endswith((".", ",", ";", ":", "?", "!"))
                )
                if is_possible_heading and len(lines) > 1:
                    heading = candidate
                    warnings.append(
                        ParseWarning(
                            code="pdf_heading_heuristic",
                            message=(
                                f"Page {page_num}: first line used as heading via heuristic; "
                                "PDF lacks structural heading metadata. Verify accuracy."
                            ),
                            locator=locator,
                            details=candidate,
                        )
                    )

            sections.append(
                make_section(
                    source_id=doc.source_id,
                    revision_id=doc.revision_id,
                    locator=locator,
                    heading=heading,
                    text=page_text,
                    ordinal=page_index,
                    parent_locator="",
                    depth=0,
                )
            )

        return ParseResult(sections=sections, warnings=tuple(warnings))

    @staticmethod
    def _decode_content(doc: RawDocument) -> bytes:
        """Decode the ``doc.content`` string to raw PDF bytes.

        Supports two encodings:
        1. Base64: when the connector base64-encoded the binary bytes.
        2. Latin-1 pass-through: when ``doc.content`` starts with ``%PDF``
           (i.e. a raw PDF byte stream that was decoded with Latin-1 to allow
           storage as a string without information loss).

        Args:
            doc: The ``RawDocument`` whose content to decode.

        Returns:
            Raw PDF bytes.

        Raises:
            ``IngestionError`` if the content cannot be decoded to valid PDF bytes.
        """
        content = doc.content

        # Latin-1 pass-through: raw PDF byte string.
        if content.startswith("%PDF"):
            return content.encode("latin-1")

        # Base64-encoded bytes.
        import base64

        try:
            return base64.b64decode(content)
        except Exception as exc:
            raise IngestionError(
                f"PdfParser: cannot decode content for source {doc.source_id!r}. "
                "Expected base64-encoded PDF bytes or a '%PDF'-prefixed Latin-1 string."
            ) from exc
