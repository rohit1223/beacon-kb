"""Unit tests for beacon_kb.parsing.base helpers."""

from __future__ import annotations

import pytest

from beacon_kb.models import (
    RevisionId,
    Section,
    SourceId,
    make_revision_id,
    make_source_id,
)
from beacon_kb.parsing.base import (
    ParseResult,
    ParseWarning,
    build_locator,
    make_section,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_id() -> SourceId:
    return make_source_id(corpus="test", canonical_uri="file:///docs/guide.md")


@pytest.fixture()
def revision_id(source_id: SourceId) -> RevisionId:
    return make_revision_id(
        source_id=str(source_id),
        content_hash="abc123",
        pipeline_fingerprint="v1",
    )


# ---------------------------------------------------------------------------
# ParseWarning tests
# ---------------------------------------------------------------------------


def test_parse_warning_is_frozen() -> None:
    w = ParseWarning(code="test_code", message="test message")
    with pytest.raises(AttributeError):
        w.code = "new_code"  # type: ignore[misc]


def test_parse_warning_defaults() -> None:
    w = ParseWarning(code="x", message="y")
    assert w.locator == ""
    assert w.details == ""


def test_parse_warning_full() -> None:
    w = ParseWarning(code="pdf_heading_heuristic", message="msg", locator="page:1", details="raw")
    assert w.code == "pdf_heading_heuristic"
    assert w.message == "msg"
    assert w.locator == "page:1"
    assert w.details == "raw"


# ---------------------------------------------------------------------------
# ParseResult tests
# ---------------------------------------------------------------------------


def test_parse_result_is_frozen() -> None:
    result = ParseResult(sections=[], warnings=())
    with pytest.raises(AttributeError):
        result.sections = []  # type: ignore[misc]


def test_parse_result_holds_sections_and_warnings(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    section = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="intro",
        heading="Intro",
        text="text",
        ordinal=0,
    )
    warning = ParseWarning(code="test", message="msg")
    result = ParseResult(sections=[section], warnings=(warning,))
    assert len(result.sections) == 1
    assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# build_locator tests
# ---------------------------------------------------------------------------


def test_build_locator_empty_path() -> None:
    assert build_locator([]) == "__root__"


def test_build_locator_single_heading() -> None:
    assert build_locator(["Introduction"]) == "Introduction"


def test_build_locator_nested() -> None:
    assert build_locator(["Installation", "Basic Example"]) == "Installation/Basic Example"


def test_build_locator_preserves_case() -> None:
    """Case must be preserved; no lowercasing."""
    result = build_locator(["MyHeading", "SubTopic"])
    assert "MyHeading" in result
    assert "SubTopic" in result
    assert "myheading" not in result.lower() or result == "MyHeading/SubTopic"


def test_build_locator_sanitises_slash_in_heading() -> None:
    """Slashes inside a heading title are replaced with underscores."""
    locator = build_locator(["I/O Operations"])
    assert "/" not in locator.replace("I_O Operations", "")
    # The heading path separator should not be confused with the slash in the title.
    assert locator == "I_O Operations"


def test_build_locator_deep_nesting() -> None:
    path = ["A", "B", "C", "D"]
    assert build_locator(path) == "A/B/C/D"


# ---------------------------------------------------------------------------
# make_section tests
# ---------------------------------------------------------------------------


def test_make_section_returns_section(source_id: SourceId, revision_id: RevisionId) -> None:
    section = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="intro",
        heading="Introduction",
        text="Some text.",
        ordinal=0,
    )
    assert isinstance(section, Section)
    assert section.locator == "intro"
    assert section.heading == "Introduction"
    assert section.text == "Some text."
    assert section.ordinal == 0
    assert section.source_id == source_id
    assert section.revision_id == revision_id


def test_make_section_id_is_deterministic(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    """Identical inputs must produce the same SectionId."""
    s1 = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="intro",
        heading="Intro",
        text="text",
        ordinal=0,
    )
    s2 = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="intro",
        heading="Intro",
        text="text",
        ordinal=0,
    )
    assert s1.id == s2.id


def test_make_section_different_locators_differ(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    s1 = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="intro",
        heading="Intro",
        text="text",
        ordinal=0,
    )
    s2 = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="config",
        heading="Config",
        text="text",
        ordinal=1,
    )
    assert s1.id != s2.id


def test_make_section_depth_and_parent(
    source_id: SourceId, revision_id: RevisionId
) -> None:
    section = make_section(
        source_id=source_id,
        revision_id=revision_id,
        locator="install/basic",
        heading="Basic",
        text="text",
        ordinal=1,
        parent_locator="install",
        depth=2,
    )
    assert section.depth == 2
    assert section.parent_locator == "install"
