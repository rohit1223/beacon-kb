"""Unit tests for beacon_kb.parsing.html.HtmlParser.

Tests are split into two categories:
1. Tests that run without the html extra (import checks, missing-extra error).
2. Tests that require the html extra (only when beautifulsoup4 is available).
"""

from __future__ import annotations

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
from beacon_kb.parsing.html import HtmlParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_id() -> SourceId:
    return make_source_id(corpus="test", canonical_uri="file:///docs/page.html")


@pytest.fixture()
def revision_id(source_id: SourceId) -> RevisionId:
    return make_revision_id(
        source_id=str(source_id),
        content_hash="cafebabe",
        pipeline_fingerprint="v1",
    )


def make_doc(content: str, source_id: SourceId, revision_id: RevisionId) -> RawDocument:
    return RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=content,
        media_type="text/html",
    )


@pytest.fixture()
def parser() -> HtmlParser:
    return HtmlParser()


# ---------------------------------------------------------------------------
# Module-level import (always safe regardless of extra)
# ---------------------------------------------------------------------------


def test_html_parser_importable_without_extra() -> None:
    """Importing HtmlParser must never fail even without beautifulsoup4."""
    from beacon_kb.parsing.html import HtmlParser  # noqa: F401


def test_html_parser_instantiable_without_extra() -> None:
    """Constructing HtmlParser must never fail even without beautifulsoup4."""
    _ = HtmlParser()


# ---------------------------------------------------------------------------
# Missing extra simulation (monkeypatching sys.modules)
# ---------------------------------------------------------------------------


def test_parse_raises_ingestion_error_when_bs4_absent(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """When beautifulsoup4 is absent, parse() must raise IngestionError, not ImportError."""
    # Save the real module if installed.
    real_bs4 = sys.modules.pop("bs4", None)
    real_bs4_element = sys.modules.pop("bs4.element", None)
    try:
        # Inject a sentinel that raises ModuleNotFoundError on import.
        sys.modules["bs4"] = None  # type: ignore[assignment]
        parser = HtmlParser()
        doc = make_doc("<html><body><p>hello</p></body></html>", source_id, revision_id)
        with pytest.raises(IngestionError, match=r"html.*extra"):
            parser.parse(doc)
    finally:
        # Restore to avoid polluting other tests.
        if real_bs4 is not None:
            sys.modules["bs4"] = real_bs4
        else:
            sys.modules.pop("bs4", None)
        if real_bs4_element is not None:
            sys.modules["bs4.element"] = real_bs4_element


# ---------------------------------------------------------------------------
# Tests requiring the html extra
# ---------------------------------------------------------------------------

bs4 = pytest.importorskip("bs4", reason="html extra (beautifulsoup4) not installed")


def test_parse_protocol_method_exists(parser: HtmlParser) -> None:
    assert callable(parser.parse)


def test_empty_html_returns_no_sections(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("<html><body></body></html>", source_id, revision_id)
    sections = parser.parse(doc)
    # Empty body may produce zero sections or one empty __root__ section.
    # Either is acceptable; no exception should be raised.
    assert isinstance(sections, list)


def test_headings_split_sections(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = """
    <html><body>
        <h1>Introduction</h1>
        <p>Intro text.</p>
        <h2>Details</h2>
        <p>Detail text.</p>
    </body></html>
    """
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    headings = [s.heading for s in sections]
    assert "Introduction" in headings
    assert "Details" in headings


def test_case_preserved_in_headings(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = "<html><body><h1>MyHeading MixedCase</h1><p>body</p></body></html>"
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    assert any(s.heading == "MyHeading MixedCase" for s in sections)


def test_case_preserved_in_body_text(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = "<html><body><h1>H</h1><p>CamelCase AND UPPERCASE text.</p></body></html>"
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "CamelCase" in all_text
    assert "UPPERCASE" in all_text


def test_code_block_preserved(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = (
        "<html><body>"
        "<h1>Setup</h1>"
        "<pre><code>pip install beacon-kb</code></pre>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "pip install beacon-kb" in all_text


def test_links_captured(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = (
        "<html><body>"
        "<h1>Links</h1>"
        '<p>See <a href="https://example.com">docs</a>.</p>'
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "https://example.com" in all_text


def test_heading_locator_path(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = (
        "<html><body>"
        "<h1>Install</h1><p>text</p>"
        "<h2>Basic</h2><p>text</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    locators = {s.heading: s.locator for s in sections}
    assert "Install" in locators
    assert "Basic" in locators
    assert locators["Basic"].startswith("Install")


def test_sections_carry_source_and_revision(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = "<html><body><h1>X</h1><p>y</p></body></html>"
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    for s in sections:
        assert s.source_id == source_id
        assert s.revision_id == revision_id


def test_cleanup_hook_called_before_extraction(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """A cleanup_hook must remove site-specific elements before extraction."""
    removed: list[str] = []

    def hook(root: object) -> None:
        # Import here is safe because we've already skipped if bs4 is absent.
        from bs4 import Tag

        if isinstance(root, Tag):
            for nav in root.find_all("nav"):
                removed.append(nav.get_text())
                nav.decompose()

    parser = HtmlParser(cleanup_hook=hook)
    html = (
        "<html><body>"
        "<nav>Site Navigation Bar</nav>"
        "<h1>Content</h1>"
        "<p>Real content.</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    # The nav bar should have been removed before extraction.
    assert "Site Navigation Bar" not in all_text
    assert removed == ["Site Navigation Bar"]


def test_no_heading_warning_emitted(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """Documents without any headings should emit a typed html_missing_heading warning."""
    from beacon_kb.parsing.base import ParseResult

    parser = HtmlParser()
    html = "<html><body><p>Just a paragraph, no headings.</p></body></html>"
    doc = make_doc(html, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    assert isinstance(result, ParseResult)
    warning_codes = [w.code for w in result.warnings]
    assert "html_missing_heading" in warning_codes


def test_parse_raises_on_non_string_content(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=b"<html></html>",  # type: ignore[arg-type]
        media_type="text/html",
    )
    with pytest.raises(IngestionError):
        parser.parse(doc)


def test_section_ids_are_deterministic(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    html = "<html><body><h1>Section</h1><p>text</p></body></html>"
    doc = make_doc(html, source_id, revision_id)
    run1 = parser.parse(doc)
    run2 = parser.parse(doc)
    assert [s.id for s in run1] == [s.id for s in run2]


# ---------------------------------------------------------------------------
# Regression: duplicate-content bug (pre/code nesting, link-in-paragraph)
# ---------------------------------------------------------------------------


def test_pre_code_nesting_no_duplicate_content(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """<pre><code>text</code></pre> must emit the text exactly once, not twice."""
    html = (
        "<html><body>"
        "<h1>Setup</h1>"
        "<pre><code>pip install beacon-kb</code></pre>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert all_text.count("pip install beacon-kb") == 1, (
        f"Expected exactly 1 occurrence of the code snippet; got: {all_text!r}"
    )


def test_link_in_paragraph_no_duplicate_anchor_text(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """<p>See <a href="x">reference guide</a>.</p> must not emit anchor text twice."""
    # The anchor text "reference guide" must not appear in the URL so we can
    # assert on exact occurrence count without URL path contamination.
    html = (
        "<html><body>"
        "<h1>Links</h1>"
        '<p>See <a href="https://example.com/api">reference guide</a>.</p>'
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    # "reference guide" must appear exactly once as part of the paragraph text;
    # it must NOT be emitted a second time by the anchor handler.
    assert all_text.count("reference guide") == 1, (
        f"Expected 'reference guide' exactly once; got: {all_text!r}"
    )
    # The href URL must also be present so that citations can resolve links.
    assert "https://example.com/api" in all_text


# ---------------------------------------------------------------------------
# Regression: duplicate-heading SectionId collision
# ---------------------------------------------------------------------------


def test_duplicate_headings_yield_distinct_locators(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Two ## Configuration sections at the same depth must get distinct locators."""
    html = (
        "<html><body>"
        "<h1>Guide</h1><p>intro</p>"
        "<h2>Configuration</h2><p>first config section</p>"
        "<h2>Configuration</h2><p>second config section</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    # All locators must be distinct.
    assert len(locators) == len(set(locators)), (
        f"Duplicate locators detected: {locators}"
    )
    # The second Configuration section must carry an ordinal suffix.
    assert any("[2]" in loc for loc in locators), (
        f"Expected a '[2]' suffix for the duplicate heading; locators: {locators}"
    )


def test_duplicate_headings_yield_distinct_section_ids(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Distinct locators must produce distinct SectionIds (no silent upsert overwrite)."""
    html = (
        "<html><body>"
        "<h2>Configuration</h2><p>first</p>"
        "<h2>Configuration</h2><p>second</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids)), f"Duplicate SectionIds: {ids}"


def test_duplicate_heading_locators_deterministic_across_parses(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Duplicate-heading disambiguation must be deterministic across two parses."""
    html = (
        "<html><body>"
        "<h2>Configuration</h2><p>alpha</p>"
        "<h2>Configuration</h2><p>beta</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    run1 = [s.locator for s in parser.parse(doc)]
    run2 = [s.locator for s in parser.parse(doc)]
    assert run1 == run2, f"Locators differ across parses: {run1} vs {run2}"


# ---------------------------------------------------------------------------
# Regression: HTML traversal text-dropping (Fix round-2)
# ---------------------------------------------------------------------------


def test_loose_text_between_paragraphs_appears_once(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Bare text nodes between <p> elements must not be silently dropped."""
    html = (
        "<html><body>"
        "<h1>Section</h1>"
        "<p>First paragraph.</p>"
        "Loose text between paras."
        "<p>Second paragraph.</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert all_text.count("Loose text between paras.") == 1, (
        f"Expected 'Loose text between paras.' exactly once; got: {all_text!r}"
    )


def test_bare_text_div_not_dropped(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """A <div> containing only a bare text node must not be silently dropped."""
    html = (
        "<html><body>"
        "<h1>Section</h1>"
        "<div>Bare div text.</div>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "Bare div text." in all_text, (
        f"Expected 'Bare div text.' in section text; got: {all_text!r}"
    )


def test_span_in_div_not_dropped(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """A <span> directly inside a <div> must not be silently dropped."""
    html = (
        "<html><body>"
        "<h1>Section</h1>"
        "<div><span>Span text here.</span></div>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "Span text here." in all_text, (
        f"Expected 'Span text here.' in section text; got: {all_text!r}"
    )
    assert all_text.count("Span text here.") == 1, (
        f"Expected exactly 1 occurrence; got: {all_text!r}"
    )


def test_blockquote_text_not_dropped(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Text directly inside a <blockquote> must not be silently dropped."""
    html = (
        "<html><body>"
        "<h1>Section</h1>"
        "<blockquote>Quoted text here.</blockquote>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "Quoted text here." in all_text, (
        f"Expected 'Quoted text here.' in section text; got: {all_text!r}"
    )


def test_table_caption_not_dropped(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """A <caption> element inside a <table> must be captured exactly once."""
    html = (
        "<html><body>"
        "<h1>Section</h1>"
        "<table>"
        "<caption>Table Caption Title.</caption>"
        "<tr><td>data</td></tr>"
        "</table>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    assert "Table Caption Title." in all_text, (
        f"Expected caption text in sections; got: {all_text!r}"
    )
    assert all_text.count("Table Caption Title.") == 1, (
        f"Expected caption text exactly once; got: {all_text!r}"
    )


# ---------------------------------------------------------------------------
# Regression: heading hierarchy (Fix round-2)
# ---------------------------------------------------------------------------


def test_h2_rooted_doc_two_siblings_are_not_nested(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Two consecutive h2 headings must be siblings, not parent/child."""
    html = (
        "<html><body>"
        "<h2>Alpha</h2><p>a</p>"
        "<h2>Beta</h2><p>b</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    assert "Alpha" in locators, f"Alpha not found in {locators}"
    assert "Beta" in locators, f"Beta not found in {locators}"
    # Beta must NOT be nested under Alpha.
    assert not any("Alpha/Beta" in loc for loc in locators), (
        f"Beta is incorrectly nested under Alpha: {locators}"
    )


def test_duplicate_h2_second_gets_suffix(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Two h2 headings with the same title must yield distinct locators."""
    html = (
        "<html><body>"
        "<h2>Config</h2><p>first</p>"
        "<h2>Config</h2><p>second</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    assert any("[2]" in loc for loc in locators), (
        f"Expected '[2]' suffix for duplicate Config heading; locators: {locators}"
    )


def test_child_of_duplicate_heading_uses_disambiguated_parent(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """A child heading under the second duplicate must carry the Config[2] path."""
    html = (
        "<html><body>"
        "<h2>Config</h2><p>first</p>"
        "<h2>Config</h2><p>second</p>"
        "<h3>Sub</h3><p>child</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    sub = next((s for s in sections if s.heading == "Sub"), None)
    assert sub is not None, "Sub section not found"
    assert "Config[2]" in sub.locator, (
        f"Expected 'Config[2]' in Sub locator, got: {sub.locator!r}"
    )
    assert sub.parent_locator == "Config[2]", (
        f"Expected parent_locator='Config[2]', got: {sub.parent_locator!r}"
    )


def test_heading_hierarchy_deterministic_across_parses_round2(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Heading hierarchy disambiguation must be deterministic across two parses."""
    html = (
        "<html><body>"
        "<h2>Config</h2><p>a</p>"
        "<h2>Config</h2><p>b</p>"
        "<h3>Sub</h3><p>c</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    run1 = [s.locator for s in parser.parse(doc)]
    run2 = [s.locator for s in parser.parse(doc)]
    assert run1 == run2, f"Locators differ across parses: {run1} vs {run2}"


# ---------------------------------------------------------------------------
# Regression: multi-link inline adjacency (Fix round-2)
# ---------------------------------------------------------------------------


def test_multi_link_paragraph_hrefs_inline(
    parser: HtmlParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Multiple links in one paragraph must each appear with their href inline."""
    html = (
        "<html><body>"
        "<h1>Links</h1>"
        '<p>See <a href="https://a.com">first</a> and <a href="https://b.com">second</a>.</p>'
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = " ".join(s.text for s in sections)
    # Each anchor must appear adjacent to its own href.
    assert "first [https://a.com]" in all_text, (
        f"Expected 'first [https://a.com]' in text; got: {all_text!r}"
    )
    assert "second [https://b.com]" in all_text, (
        f"Expected 'second [https://b.com]' in text; got: {all_text!r}"
    )


# ---------------------------------------------------------------------------
# Regression: empty section warning (Fix round-2)
# ---------------------------------------------------------------------------


def test_empty_section_warning_emitted(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """A heading with no body text must emit an html_empty_section warning."""
    from beacon_kb.parsing.base import ParseResult

    parser = HtmlParser()
    html = (
        "<html><body>"
        "<h1>EmptySection</h1>"
        "<h2>HasContent</h2><p>text here</p>"
        "</body></html>"
    )
    doc = make_doc(html, source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    assert isinstance(result, ParseResult)
    warning_codes = [w.code for w in result.warnings]
    assert "html_empty_section" in warning_codes, (
        f"Expected 'html_empty_section' warning; got codes: {warning_codes}"
    )
    # The warning locator must reference the empty heading.
    empty_warnings = [w for w in result.warnings if w.code == "html_empty_section"]
    assert any("EmptySection" in w.locator for w in empty_warnings), (
        f"Expected warning locator to reference 'EmptySection'; got: {empty_warnings}"
    )
