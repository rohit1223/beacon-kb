"""Integration tests for the Docling parsing adapter (Task 02.3).

Offline coverage is Markdown, HTML, and DOCX only: those routes use Docling's
declarative backends, which need no model weights and therefore run fully
offline.
PDF conversion requires layout/table model artifacts that Docling downloads on
first use; in a fully offline environment that download cannot succeed, so the
PDF test below is guarded by the ``BEACON_PDF_MODELS_AVAILABLE`` environment
flag and is exercised in CI with a pre-populated model cache.
The shared ``tests/beacon/conftest.py`` sets ``HF_HUB_OFFLINE`` and friends so
any accidental model fetch fails fast instead of hanging.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from beacon.errors import IngestionError
from beacon.ingest.parsing import (
    PARSER_VERSION,
    ParsedDocument,
    ParsedSection,
    ParseWarning,
    SectionKind,
    parse,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "docs"

MD_MEDIA = "text/markdown"
HTML_MEDIA = "text/html"
DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MEDIA = "application/pdf"


def _md_bytes() -> bytes:
    return (FIXTURES / "sample.md").read_bytes()


def _html_bytes() -> bytes:
    return (FIXTURES / "sample.html").read_bytes()


def _docx_bytes() -> bytes:
    return (FIXTURES / "sample.docx").read_bytes()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMarkdown:
    def test_returns_parsed_document(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA, source_uri="file:///sample.md")
        assert isinstance(doc, ParsedDocument)
        assert doc.title == "Guide"
        assert doc.parser_version == PARSER_VERSION
        assert doc.sections
        assert all(isinstance(s, ParsedSection) for s in doc.sections)

    def test_sections_are_ordered(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        assert [s.ordinal for s in doc.sections] == list(range(len(doc.sections)))

    def test_heading_paths_nested(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        paths = [s.heading_path for s in doc.sections]
        assert ("Guide",) in paths
        assert ("Guide", "Install") in paths
        assert ("Guide", "Install", "Linux") in paths
        assert ("Guide", "Install", "Windows") in paths
        assert ("Guide", "Configure") in paths

    def test_case_preserved(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        all_text = "\n".join(s.text for s in doc.sections)
        assert "Case PRESERVED here" in all_text

    def test_code_section_kind_and_preservation(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        code_sections = [s for s in doc.sections if s.kind == SectionKind.CODE]
        assert len(code_sections) == 1
        code = code_sections[0]
        assert 'def hello():\n    return "world"' in code.text
        assert code.heading_path == ("Guide", "Configure")

    def test_table_section_kind_and_preservation(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        table_sections = [s for s in doc.sections if s.kind == SectionKind.TABLE]
        assert len(table_sections) == 1
        table = table_sections[0]
        assert "Column A" in table.text
        assert "val1" in table.text
        assert "val4" in table.text
        assert table.heading_path == ("Guide", "Configure")

    def test_duplicate_heading_locator_disambiguation(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        install_text_sections = [
            s
            for s in doc.sections
            if s.heading_path == ("Guide", "Install") and s.kind == SectionKind.TEXT
        ]
        assert len(install_text_sections) == 2
        locators = [s.locator for s in install_text_sections]
        assert locators[0] == "Guide/Install"
        assert locators[1] == "Guide/Install[2]"
        assert "Duplicate heading" in install_text_sections[1].text

    def test_locators_are_unique(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        locators = [s.locator for s in doc.sections]
        assert len(locators) == len(set(locators))

    def test_preamble_before_first_heading_kept_as_root(self) -> None:
        content = b"Loose preamble text.\n\n# Heading\n\nBody text.\n"
        doc = parse(content, MD_MEDIA)
        root_sections = [s for s in doc.sections if s.heading_path == ()]
        assert len(root_sections) == 1
        assert root_sections[0].locator == "__root__"
        assert "Loose preamble text" in root_sections[0].text

    def test_list_section_kind(self) -> None:
        content = b"# Doc\n\n- item one\n- item two\n"
        doc = parse(content, MD_MEDIA)
        list_sections = [s for s in doc.sections if s.kind == SectionKind.LIST]
        assert len(list_sections) == 1
        assert "item one" in list_sections[0].text
        assert "item two" in list_sections[0].text

    def test_no_page_locators_for_markdown(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        assert all(s.page_no is None for s in doc.sections)

    def test_duplicate_heading_with_inline_code_locator_uniqueness(self) -> None:
        content = b"## Dup\ntext `code` more\n\n## Dup\ntext `code` more\n"
        doc = parse(content, MD_MEDIA)
        locators = [s.locator for s in doc.sections]
        assert len(locators) == len(set(locators)), "All locators must be unique"
        assert "Dup" in locators[0]
        assert "[2]" in locators[1], "Second duplicate heading must have [2] disambiguation"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHtml:
    def test_heading_paths_nested(self) -> None:
        doc = parse(_html_bytes(), HTML_MEDIA, source_uri="file:///sample.html")
        assert doc.title == "Guide"
        paths = [s.heading_path for s in doc.sections]
        assert ("Guide",) in paths
        assert ("Guide", "Install") in paths
        assert ("Guide", "Install", "Linux") in paths
        assert ("Guide", "Configure") in paths

    def test_link_text_preserved(self) -> None:
        doc = parse(_html_bytes(), HTML_MEDIA)
        all_text = "\n".join(s.text for s in doc.sections)
        assert "link" in all_text
        assert "CASE Preserved" in all_text

    def test_media_type_with_charset_parameter(self) -> None:
        doc = parse(_html_bytes(), "text/html; charset=utf-8")
        assert doc.sections


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDocx:
    def test_heading_paths_nested(self) -> None:
        doc = parse(_docx_bytes(), DOCX_MEDIA, source_uri="file:///sample.docx")
        paths = [s.heading_path for s in doc.sections]
        assert ("Guide",) in paths
        assert ("Guide", "Install") in paths
        assert ("Guide", "Install", "Linux") in paths
        assert ("Guide", "Install", "Windows") in paths
        assert ("Guide", "Configure") in paths

    def test_title_falls_back_to_first_heading(self) -> None:
        doc = parse(_docx_bytes(), DOCX_MEDIA)
        assert doc.title == "Guide"

    def test_case_preserved(self) -> None:
        doc = parse(_docx_bytes(), DOCX_MEDIA)
        all_text = "\n".join(s.text for s in doc.sections)
        assert "CASE Preserved" in all_text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDeterminism:
    def test_markdown_identical_bytes_identical_result(self) -> None:
        first = parse(_md_bytes(), MD_MEDIA)
        second = parse(_md_bytes(), MD_MEDIA)
        assert first == second

    def test_html_identical_bytes_identical_result(self) -> None:
        first = parse(_html_bytes(), HTML_MEDIA)
        second = parse(_html_bytes(), HTML_MEDIA)
        assert first == second

    def test_docx_identical_bytes_identical_result(self) -> None:
        first = parse(_docx_bytes(), DOCX_MEDIA)
        second = parse(_docx_bytes(), DOCX_MEDIA)
        assert first == second


# ---------------------------------------------------------------------------
# Error typing (strict-decode obligation)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestErrors:
    def test_invalid_utf8_markdown_raises_typed_error(self) -> None:
        bad = b"\xff\xfe# Not utf-8 \x80\x81"
        with pytest.raises(IngestionError) as exc_info:
            parse(bad, MD_MEDIA, source_uri="file:///bad.md")
        assert exc_info.value.source_uri == "file:///bad.md"
        assert "decode" in exc_info.value.message.lower()

    def test_invalid_utf8_html_raises_typed_error(self) -> None:
        bad = b"<html>\xff\xfe</html>"
        with pytest.raises(IngestionError) as exc_info:
            parse(bad, HTML_MEDIA, source_uri="file:///bad.html")
        assert exc_info.value.source_uri == "file:///bad.html"

    def test_unknown_media_type_raises_typed_error(self) -> None:
        with pytest.raises(IngestionError) as exc_info:
            parse(b"anything", "application/octet-stream", source_uri="file:///blob.bin")
        assert exc_info.value.source_uri == "file:///blob.bin"
        assert "application/octet-stream" in exc_info.value.message

    def test_corrupt_docx_raises_typed_error(self) -> None:
        with pytest.raises(IngestionError) as exc_info:
            parse(b"definitely not a zip archive", DOCX_MEDIA, source_uri="file:///bad.docx")
        assert exc_info.value.source_uri == "file:///bad.docx"

    def test_empty_content_raises_typed_error(self) -> None:
        with pytest.raises(IngestionError) as exc_info:
            parse(b"", DOCX_MEDIA, source_uri="file:///empty.docx")
        assert exc_info.value.source_uri == "file:///empty.docx"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWarnings:
    def test_unhandled_item_kind_degrades_with_typed_warning(self) -> None:
        content = b"# Doc\n\nText before.\n\n![an image](missing.png)\n\nText after.\n"
        doc = parse(content, MD_MEDIA)
        assert doc.warnings
        assert all(isinstance(w, ParseWarning) for w in doc.warnings)
        codes = {w.code for w in doc.warnings}
        assert "unhandled_item_kind" in codes
        all_text = "\n".join(s.text for s in doc.sections)
        assert "Text before" in all_text
        assert "Text after" in all_text

    def test_clean_document_has_no_warnings(self) -> None:
        doc = parse(_html_bytes(), HTML_MEDIA)
        assert doc.warnings == ()


# ---------------------------------------------------------------------------
# Parser version
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestParserVersion:
    def test_parser_version_exported_and_stable_format(self) -> None:
        assert isinstance(PARSER_VERSION, str)
        assert PARSER_VERSION.startswith("docling-")
        assert ".beacon-adapter-" in PARSER_VERSION

    def test_parsed_document_carries_parser_version(self) -> None:
        doc = parse(_md_bytes(), MD_MEDIA)
        assert doc.parser_version == PARSER_VERSION


# ---------------------------------------------------------------------------
# PDF (model-gated)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("BEACON_PDF_MODELS_AVAILABLE") != "1",
    reason=(
        "PDF conversion requires Docling layout/table model artifacts that are "
        "downloaded on first use; this environment runs fully offline. "
        "Set BEACON_PDF_MODELS_AVAILABLE=1 in CI with a pre-populated model "
        "cache to exercise the PDF path."
    ),
)
class TestPdf:
    """PDF conversion tests, gated on model availability.

    Docling's PDF pipeline needs layout and table-structure model weights.
    They are not present in the offline development environment, so these
    tests only run where ``BEACON_PDF_MODELS_AVAILABLE=1`` signals that a
    model cache is available (CI with cached artifacts).
    The PDF route itself ships as production code in
    ``beacon.ingest.parsing`` and is structurally identical to the other
    formats; only the conversion execution is gated.
    """

    def test_pdf_sections_carry_page_locators(self) -> None:
        content = (FIXTURES / "sample.pdf").read_bytes()
        doc = parse(content, PDF_MEDIA, source_uri="file:///sample.pdf")
        assert doc.sections
        assert any(s.page_no == 1 for s in doc.sections)
        assert any(s.page_no == 2 for s in doc.sections)

    def test_pdf_determinism(self) -> None:
        content = (FIXTURES / "sample.pdf").read_bytes()
        assert parse(content, PDF_MEDIA) == parse(content, PDF_MEDIA)
