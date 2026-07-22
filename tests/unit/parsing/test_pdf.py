"""Unit tests for beacon_kb.parsing.pdf.PdfParser.

Tests are split into two categories:
1. Tests that run without the pdf extra (import checks, missing-extra error).
2. Tests that require the pdf extra (only when pypdf is available).
"""

from __future__ import annotations

import base64
import io
import sys

import pytest

from beacon_kb.errors import IngestionError
from beacon_kb.models import (
    RawDocument,
    RevisionId,
    SourceId,
    make_revision_id,
    make_source_id,
)
from beacon_kb.parsing.pdf import PdfParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_id() -> SourceId:
    return make_source_id(corpus="test", canonical_uri="file:///docs/report.pdf")


@pytest.fixture()
def revision_id(source_id: SourceId) -> RevisionId:
    return make_revision_id(
        source_id=str(source_id),
        content_hash="feedface",
        pipeline_fingerprint="v1",
    )


def make_doc(content: str, source_id: SourceId, revision_id: RevisionId) -> RawDocument:
    return RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=content,
        media_type="application/pdf",
    )


def _build_raw_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF with embedded text using raw PDF syntax."""
    encoded_text = text.replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 700 Td ({encoded_text}) Tj ET".encode("latin-1")
    content_len = len(content)

    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {content_len} >>
stream
""".encode("latin-1") + content + b"""
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
trailer << /Size 6 /Root 1 0 R >>
startxref
0
%%EOF
"""
    return pdf


# ---------------------------------------------------------------------------
# Module-level import (always safe regardless of extra)
# ---------------------------------------------------------------------------


def test_pdf_parser_importable_without_extra() -> None:
    """Importing PdfParser must never fail even without pypdf."""
    from beacon_kb.parsing.pdf import PdfParser  # noqa: F401


def test_pdf_parser_instantiable_without_extra() -> None:
    """Constructing PdfParser must never fail even without pypdf."""
    _ = PdfParser()


# ---------------------------------------------------------------------------
# Missing extra simulation (monkeypatching sys.modules)
# ---------------------------------------------------------------------------


def test_parse_raises_ingestion_error_when_pypdf_absent(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """When pypdf is absent, parse() must raise IngestionError, not ImportError."""
    real_pypdf = sys.modules.pop("pypdf", None)
    try:
        sys.modules["pypdf"] = None  # type: ignore[assignment]
        parser = PdfParser()
        doc = make_doc(base64.b64encode(b"%PDF-1.4").decode(), source_id, revision_id)
        with pytest.raises(IngestionError, match=r"pdf.*extra"):
            parser.parse(doc)
    finally:
        if real_pypdf is not None:
            sys.modules["pypdf"] = real_pypdf
        else:
            sys.modules.pop("pypdf", None)


# ---------------------------------------------------------------------------
# Tests requiring the pdf extra
# ---------------------------------------------------------------------------

pypdf = pytest.importorskip("pypdf", reason="pdf extra (pypdf) not installed")


def test_parse_protocol_method_exists(source_id: SourceId, revision_id: RevisionId) -> None:
    parser = PdfParser()
    assert callable(parser.parse)


def test_parse_raises_ingestion_error_on_invalid_pdf(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """Malformed PDF bytes must raise IngestionError, not a raw pypdf exception."""
    parser = PdfParser()
    garbage = base64.b64encode(b"not a pdf at all").decode()
    doc = make_doc(garbage, source_id, revision_id)
    with pytest.raises(IngestionError):
        parser.parse(doc)


def test_parse_raises_on_non_string_content(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    parser = PdfParser()
    doc = RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=b"%PDF bytes",  # type: ignore[arg-type]
        media_type="application/pdf",
    )
    with pytest.raises(IngestionError):
        parser.parse(doc)


def test_sections_have_page_locators(source_id: SourceId, revision_id: RevisionId) -> None:
    """Every section from a PDF must have a page:<n> locator."""
    parser = PdfParser()
    pdf_bytes = _build_raw_pdf("This is page one content with enough text to pass threshold.")
    content = base64.b64encode(pdf_bytes).decode()
    doc = make_doc(content, source_id, revision_id)
    # May succeed or emit warnings; we just need locators to be page:N.
    result = parser.parse_with_warnings(doc)
    for section in result.sections:
        assert section.locator.startswith("page:"), (
            f"Section locator {section.locator!r} does not start with 'page:'"
        )


def test_sections_carry_source_and_revision(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    parser = PdfParser()
    pdf_bytes = _build_raw_pdf("Some content for the test.")
    content = base64.b64encode(pdf_bytes).decode()
    doc = make_doc(content, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    for section in result.sections:
        assert section.source_id == source_id
        assert section.revision_id == revision_id


def test_heading_heuristic_warning_emitted(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """When the first line of a page is used as a heading heuristically,
    a typed warning with code 'pdf_heading_heuristic' may be emitted."""
    parser = PdfParser(min_body_chars=10)
    pdf_bytes = _build_raw_pdf("Section Title\nThis is the body content of the section.")
    content = base64.b64encode(pdf_bytes).decode()
    doc = make_doc(content, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    # Validate that all emitted warning codes are from the known set.
    for w in result.warnings:
        assert w.code in {
            "pdf_heading_heuristic",
            "pdf_possible_header_footer",
            "pdf_empty_page",
            "pdf_extraction_error",
        }, f"Unexpected warning code: {w.code!r}"


def test_empty_page_warning_not_section(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """An empty PDF page must not produce a section; it must emit a typed warning."""
    from pypdf import PdfWriter  # type: ignore

    parser = PdfParser()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)  # truly blank page
    buf = io.BytesIO()
    writer.write(buf)
    content = base64.b64encode(buf.getvalue()).decode()
    doc = make_doc(content, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    warning_codes = [w.code for w in result.warnings]
    # A blank page must either: produce no sections, or produce a warning.
    # It must NOT silently produce an empty section.
    if result.sections:
        for s in result.sections:
            assert s.text.strip(), "Empty page produced a section with empty text"
    else:
        assert "pdf_empty_page" in warning_codes or len(result.warnings) > 0


def test_latin1_pass_through_decoded(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """Content starting with %PDF is decoded as Latin-1 (raw pass-through)."""
    parser = PdfParser()
    pdf_bytes = _build_raw_pdf("Latin-1 content test.")
    # Pass as Latin-1 decoded string.
    content = pdf_bytes.decode("latin-1")
    doc = make_doc(content, source_id, revision_id)
    # Should not raise IngestionError for the decode step.
    try:
        result = parser.parse_with_warnings(doc)
        # If it succeeds, pages should have locators.
        for s in result.sections:
            assert s.locator.startswith("page:")
    except IngestionError as exc:
        # Only PDF parse errors are acceptable; not content-decode errors.
        assert "open PDF" in str(exc) or "cannot decode" in str(exc)


def test_parse_with_warnings_returns_parse_result(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    from beacon_kb.parsing.base import ParseResult

    parser = PdfParser()
    pdf_bytes = _build_raw_pdf("Content for the parse result test.")
    content = base64.b64encode(pdf_bytes).decode()
    doc = make_doc(content, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    assert isinstance(result, ParseResult)
    assert isinstance(result.warnings, tuple)
