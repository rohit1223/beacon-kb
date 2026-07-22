"""Unit tests for beacon_kb.parsing.markdown.MarkdownParser."""

from __future__ import annotations

import pathlib

import pytest

from beacon_kb.errors import IngestionError
from beacon_kb.models import (
    RawDocument,
    RevisionId,
    SourceId,
    make_revision_id,
    make_source_id,
)
from beacon_kb.parsing.markdown import MarkdownParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_MD = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "documents" / "sample.md"


@pytest.fixture()
def source_id() -> SourceId:
    return make_source_id(corpus="test", canonical_uri="file:///docs/guide.md")


@pytest.fixture()
def revision_id(source_id: SourceId) -> RevisionId:
    return make_revision_id(
        source_id=str(source_id),
        content_hash="deadbeef",
        pipeline_fingerprint="v1",
    )


def make_doc(content: str, source_id: SourceId, revision_id: RevisionId) -> RawDocument:
    return RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=content,
        media_type="text/markdown",
    )


@pytest.fixture()
def parser() -> MarkdownParser:
    return MarkdownParser()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_parser_has_parse_method(parser: MarkdownParser) -> None:
    assert callable(parser.parse)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------


def test_empty_document_returns_no_sections(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("", source_id, revision_id)
    sections = parser.parse(doc)
    assert sections == []


def test_blank_document_returns_no_sections(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("   \n\n   \n", source_id, revision_id)
    sections = parser.parse(doc)
    assert sections == []


def test_single_h1_heading(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("# Hello World\n\nSome text.", source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) == 1
    assert sections[0].heading == "Hello World"
    assert "Some text." in sections[0].text


def test_multiple_headings(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Section A\n\nContent A.\n\n## Section B\n\nContent B.\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) == 2
    headings = [s.heading for s in sections]
    assert "Section A" in headings
    assert "Section B" in headings


def test_pre_heading_content_emitted_as_root(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "Preamble text before any heading.\n\n# First Heading\n\nBody."
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) == 2
    root_section = sections[0]
    assert root_section.locator == "__root__"
    assert "Preamble text" in root_section.text


def test_ordinals_are_sequential(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# A\n\ntext\n\n# B\n\ntext\n\n# C\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    ordinals = [s.ordinal for s in sections]
    assert ordinals == sorted(ordinals)


def test_sections_carry_source_id(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("# H\n\ntext", source_id, revision_id)
    sections = parser.parse(doc)
    for s in sections:
        assert s.source_id == source_id


def test_sections_carry_revision_id(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("# H\n\ntext", source_id, revision_id)
    sections = parser.parse(doc)
    for s in sections:
        assert s.revision_id == revision_id


# ---------------------------------------------------------------------------
# Case preservation (the critical constraint)
# ---------------------------------------------------------------------------


def test_case_is_preserved_in_heading(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("# MyHeading with MixedCase\n\nBody.", source_id, revision_id)
    sections = parser.parse(doc)
    assert sections[0].heading == "MyHeading with MixedCase"


def test_case_is_preserved_in_body(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = make_doc("# H\n\nSome CamelCase Text AND UPPERCASE.", source_id, revision_id)
    sections = parser.parse(doc)
    assert "CamelCase" in sections[0].text
    assert "UPPERCASE" in sections[0].text


# ---------------------------------------------------------------------------
# Fenced code block preservation
# ---------------------------------------------------------------------------


def test_fenced_code_block_preserved(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Setup\n\n```bash\npip install beacon-kb\n```\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) == 1
    assert "pip install beacon-kb" in sections[0].text
    assert "```bash" in sections[0].text


def test_heading_inside_code_block_not_parsed(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """A # line inside a fenced code block must NOT become a heading."""
    md = "# Real Heading\n\n```\n# Not a heading\n```\n\nTrailing text."
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    # Only one section from the real heading; the code block # is not a heading.
    assert len(sections) == 1
    assert sections[0].heading == "Real Heading"
    assert "# Not a heading" in sections[0].text


def test_tilde_fenced_code_block(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Section\n\n~~~python\nprint('hello')\n~~~\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) == 1
    assert "print('hello')" in sections[0].text


# ---------------------------------------------------------------------------
# Heading path (locator) correctness
# ---------------------------------------------------------------------------


def test_nested_heading_locator(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Install\n\ntext\n\n## Basic\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    locators = {s.heading: s.locator for s in sections}
    assert locators["Install"] == "Install"
    assert locators["Basic"] == "Install/Basic"


def test_depth_set_correctly(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# H1\n\ntext\n\n## H2\n\ntext\n\n### H3\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    depths = {s.heading: s.depth for s in sections}
    assert depths["H1"] == 1
    assert depths["H2"] == 2
    assert depths["H3"] == 3


def test_parent_locator_set(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Parent\n\ntext\n\n## Child\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    child = next(s for s in sections if s.heading == "Child")
    assert child.parent_locator == "Parent"


# ---------------------------------------------------------------------------
# Tables and links pass through as text
# ---------------------------------------------------------------------------


def test_table_preserved_in_text(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Config\n\n| Key | Value |\n|-----|-------|\n| foo | bar |\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert "| foo | bar |" in sections[0].text


def test_link_preserved_in_text(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Links\n\nSee the [docs](https://example.com).\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    assert "[docs](https://example.com)" in sections[0].text


# ---------------------------------------------------------------------------
# Section IDs are deterministic
# ---------------------------------------------------------------------------


def test_section_ids_are_deterministic(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    md = "# Section\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    run1 = parser.parse(doc)
    run2 = parser.parse(doc)
    assert [s.id for s in run1] == [s.id for s in run2]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_parse_raises_on_non_string_content(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    doc = RawDocument(
        source_id=source_id,
        revision_id=revision_id,
        content=b"bytes not str",  # type: ignore[arg-type]
        media_type="text/markdown",
    )
    with pytest.raises(IngestionError):
        parser.parse(doc)


# ---------------------------------------------------------------------------
# Sample fixture regression
# ---------------------------------------------------------------------------


def test_sample_fixture_parsed(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """The sample fixture must produce multiple sections with case preserved."""
    content = _SAMPLE_MD.read_text(encoding="utf-8")
    doc = make_doc(content, source_id, revision_id)
    sections = parser.parse(doc)
    assert len(sections) >= 5, "Expected at least 5 sections from the sample fixture"
    headings = [s.heading for s in sections]
    # Check a few expected headings (case preserved).
    assert "Installation" in headings
    assert "Configuration" in headings
    # Ensure no section has an empty locator.
    for s in sections:
        assert s.locator, f"Section {s.ordinal!r} has empty locator"


def test_sample_fixture_code_blocks_preserved(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    content = _SAMPLE_MD.read_text(encoding="utf-8")
    doc = make_doc(content, source_id, revision_id)
    sections = parser.parse(doc)
    all_text = "\n".join(s.text for s in sections)
    assert "pip install beacon-kb" in all_text
    assert "```bash" in all_text


# ---------------------------------------------------------------------------
# Protocol: parse_with_warnings
# ---------------------------------------------------------------------------


def test_parse_with_warnings_returns_parse_result(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    from beacon_kb.parsing.base import ParseResult

    doc = make_doc("# H\n\ntext", source_id, revision_id)
    result = parser.parse_with_warnings(doc)
    assert isinstance(result, ParseResult)
    assert len(result.sections) == 1
    assert isinstance(result.warnings, tuple)


# ---------------------------------------------------------------------------
# Regression: duplicate-heading SectionId collision
# ---------------------------------------------------------------------------


def test_duplicate_headings_yield_distinct_locators(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Two ## Configuration sections under the same ## parent get distinct locators."""
    # Use an h1 parent so both h2 headings share the same path prefix,
    # making their raw locators identical ("Guide/Configuration" twice).
    md = (
        "# Guide\n\nIntro\n\n"
        "## Configuration\n\nfirst config\n\n"
        "## Configuration\n\nsecond config\n"
    )
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    assert len(locators) == len(set(locators)), (
        f"Duplicate locators detected: {locators}"
    )
    assert any("[2]" in loc for loc in locators), (
        f"Expected a '[2]' suffix for the duplicate heading; locators: {locators}"
    )


def test_duplicate_headings_yield_distinct_section_ids(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Distinct locators must produce distinct SectionIds (no silent upsert overwrite)."""
    md = (
        "# Guide\n\nIntro\n\n"
        "## Configuration\n\nfirst\n\n"
        "## Configuration\n\nsecond\n"
    )
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids)), f"Duplicate SectionIds: {ids}"


def test_duplicate_heading_locators_deterministic_across_parses(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Duplicate-heading disambiguation must be deterministic across two parses."""
    md = (
        "# Guide\n\nIntro\n\n"
        "## Configuration\n\nalpha\n\n"
        "## Configuration\n\nbeta\n"
    )
    doc = make_doc(md, source_id, revision_id)
    run1 = [s.locator for s in parser.parse(doc)]
    run2 = [s.locator for s in parser.parse(doc)]
    assert run1 == run2, f"Locators differ across parses: {run1} vs {run2}"


# ---------------------------------------------------------------------------
# Regression: heading hierarchy (Fix round-2)
# ---------------------------------------------------------------------------


def test_h2_rooted_doc_two_siblings_are_not_nested(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Two consecutive h2 headings with no h1 must be siblings, not parent/child."""
    md = "## Alpha\n\ntext a\n\n## Beta\n\ntext b\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    assert "Alpha" in locators, f"Alpha not found in {locators}"
    assert "Beta" in locators, f"Beta not found in {locators}"
    assert not any("Alpha/Beta" in loc for loc in locators), (
        f"Beta is incorrectly nested under Alpha: {locators}"
    )


def test_level_jump_h1_h3_h3_siblings(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """# A / ### B / ### D: B and D must be siblings under A, not nested."""
    md = "# A\n\ntext\n\n### B\n\ntext\n\n### D\n\ntext\n"
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    b = next((s for s in sections if s.heading == "B"), None)
    d = next((s for s in sections if s.heading == "D"), None)
    assert b is not None and d is not None, "B or D section missing"
    assert b.locator == "A/B", f"Expected 'A/B', got {b.locator!r}"
    assert d.locator == "A/D", f"Expected 'A/D', got {d.locator!r}"


def test_duplicate_h2_child_uses_disambiguated_parent(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """### Sub under second ## Config must have locator Config[2]/Sub."""
    md = (
        "## Config\n\na\n\n"
        "## Config\n\nb\n\n"
        "### Sub\n\nc\n"
    )
    doc = make_doc(md, source_id, revision_id)
    sections = parser.parse(doc)
    locators = [s.locator for s in sections]
    sub = next((s for s in sections if s.heading == "Sub"), None)
    assert sub is not None, f"Sub section not found; locators: {locators}"
    assert "Config[2]" in sub.locator, (
        f"Expected 'Config[2]' in Sub locator; got: {sub.locator!r}"
    )
    assert sub.parent_locator == "Config[2]", (
        f"Expected parent_locator='Config[2]'; got: {sub.parent_locator!r}"
    )


def test_heading_hierarchy_deterministic_across_parses_round2(
    parser: MarkdownParser, source_id: SourceId, revision_id: RevisionId
) -> None:
    """Heading hierarchy disambiguation must be stable across two parses."""
    md = (
        "## Config\n\na\n\n"
        "## Config\n\nb\n\n"
        "### Sub\n\nc\n"
    )
    doc = make_doc(md, source_id, revision_id)
    run1 = [s.locator for s in parser.parse(doc)]
    run2 = [s.locator for s in parser.parse(doc)]
    assert run1 == run2, f"Locators differ across parses: {run1} vs {run2}"
