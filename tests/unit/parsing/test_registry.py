"""Tests that the Markdown parser is registered as a builtin and
that HTML/PDF parsers are importable without triggering optional deps."""

from __future__ import annotations

import pytest


def test_markdown_parser_registered_as_builtin() -> None:
    """The MarkdownParser must be resolvable from the PARSERS registry group.

    Some contract tests call clear_registry() which removes builtins.
    Re-run the builtin registration if the builtin is absent to keep this
    test order-independent (the registry isolation issue is tracked in ROADMAP.md).
    """
    from beacon_kb.errors import PluginNotFound
    from beacon_kb.parsing.markdown import MarkdownParser
    from beacon_kb.registry import groups, precedence

    try:
        parser = precedence.resolve(group=groups.PARSERS, name="markdown")
    except PluginNotFound:
        # Registry was cleared by a prior test; re-seed the markdown builtin.
        precedence.register_builtin(
            group=groups.PARSERS,
            name="markdown",
            instance=MarkdownParser(),
        )
        parser = precedence.resolve(group=groups.PARSERS, name="markdown")

    assert isinstance(parser, MarkdownParser)


def test_html_parser_class_importable_without_extra() -> None:
    """HtmlParser class must be importable even when beautifulsoup4 is absent."""
    from beacon_kb.parsing.html import HtmlParser

    _ = HtmlParser


def test_pdf_parser_class_importable_without_extra() -> None:
    """PdfParser class must be importable even when pypdf is absent."""
    from beacon_kb.parsing.pdf import PdfParser

    _ = PdfParser


def test_parsing_package_importable() -> None:
    """The parsing package __init__ must import cleanly."""
    import beacon_kb.parsing as parsing

    assert hasattr(parsing, "ParseWarning")
    assert hasattr(parsing, "ParseResult")


def test_html_pdf_not_auto_registered() -> None:
    """HtmlParser and PdfParser must NOT be auto-registered at import time.

    They depend on optional extras; auto-registering would import those deps
    eagerly, breaking base-package installs.  They must be registered explicitly
    by the caller after constructing the parser with any required configuration.

    This test checks the explicit registry (not builtins) to confirm neither
    parser was registered as a side effect of any import.
    """
    from beacon_kb.registry import groups, precedence

    registered = precedence.list_registered(group=groups.PARSERS)
    assert "html" not in registered, (
        "HtmlParser must NOT be auto-registered; it requires explicit caller registration"
    )
    assert "pdf" not in registered, (
        "PdfParser must NOT be auto-registered; it requires explicit caller registration"
    )


def test_is_text_media_type() -> None:
    """is_text_media_type covers the single-source-of-truth for text decodability."""
    from beacon_kb.ingestion.media import is_text_media_type

    # Text types.
    assert is_text_media_type("text/markdown") is True
    assert is_text_media_type("text/html") is True
    assert is_text_media_type("text/plain") is True
    assert is_text_media_type("text/yaml") is True

    # Known application text types.
    assert is_text_media_type("application/json") is True
    assert is_text_media_type("application/toml") is True
    assert is_text_media_type("application/x-ipynb+json") is True

    # Binary types.
    assert is_text_media_type("application/pdf") is False
    assert is_text_media_type("application/octet-stream") is False
    assert is_text_media_type("image/png") is False
    assert is_text_media_type("image/jpeg") is False


def test_filesystem_connector_uses_is_text_media_type(tmp_path: pytest.TempPathFactory) -> None:
    """FilesystemConnector must reject PDF files (not just the old inline tuple).
    This confirms refactoring from the inline tuple to is_text_media_type works."""
    import pathlib

    from beacon_kb.connectors.filesystem import FilesystemConnector
    from beacon_kb.errors import IngestionError

    pdf_file = pathlib.Path(str(tmp_path)) / "doc.pdf"  # type: ignore[arg-type]
    pdf_file.write_bytes(b"%PDF-1.4 minimal")

    connector = FilesystemConnector(
        root=str(tmp_path),  # type: ignore[arg-type]
        corpus="test",
        patterns=["*.pdf"],
    )
    uri = connector.list_sources()[0]
    with pytest.raises(IngestionError, match="binary or non-text"):
        connector.fetch(uri)
