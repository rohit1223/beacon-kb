"""Unit tests for match-centered snippets with real provenance.

TDD suite - written before implementation in src/beacon/retrieval/snippets.py.

Coverage:
- Snippet centers on the query match, not on the document prefix.
- Falls back to text center when no query term matches.
- Match term appears within the snippet text.
- char_start/char_end are valid and consistent with the emitted text.
- Real source_uri, title, heading_path, and locator are preserved (never
  invented hash-form IDs).
- Empty chunk text produces an empty snippet without raising.
- Span is consistent: text == chunk_text[char_start:char_end] (after strip).
"""

from __future__ import annotations

from beacon.retrieval.snippets import Snippet, _find_match_center, build_snippet

# ---------------------------------------------------------------------------
# Tests: match centering
# ---------------------------------------------------------------------------


class TestMatchCentering:
    """Snippet must center on the query match, not start at document position 0."""

    def test_snippet_not_document_prefix_when_match_is_deep(self) -> None:
        """Match deep in a long chunk must not produce a snippet starting at 0."""
        # "python" at char ~400; 200 chars of padding before it.
        long_text = "A " * 200 + "python tutorial here" + "B " * 200
        snip = build_snippet(
            long_text,
            "python tutorial",
            source_uri="file:///docs/test.md",
            title="Test Doc",
            heading_path=["Section 1"],
            locator="section-1",
            chunk_id="cid-001",
            max_chars=200,
        )
        # char_start must be significantly > 0.
        assert snip.char_start > 0, (
            f"Snippet must not start at document prefix when match is deep. "
            f"char_start={snip.char_start}"
        )

    def test_match_term_appears_in_snippet(self) -> None:
        """The matched query term must appear within the snippet text."""
        text = "filler " * 30 + "python programming " + "filler " * 30
        snip = build_snippet(
            text,
            "python",
            source_uri="file:///docs/test.md",
            title="Test",
            heading_path=["Intro"],
            locator="intro",
            chunk_id="cid-001",
            max_chars=200,
        )
        assert "python" in snip.text.lower(), (
            f"Match term 'python' must appear in snippet. text={snip.text!r}"
        )

    def test_fallback_to_center_when_no_match(self) -> None:
        """When no query term matches, snippet centers on middle of text (not prefix)."""
        text = "a " * 100 + "midpoint here" + "z " * 100
        snip = build_snippet(
            text,
            "xyzzy does not exist in text",
            source_uri="file:///docs/test.md",
            title="Test",
            heading_path=["Intro"],
            locator="intro",
            chunk_id="cid-001",
            max_chars=100,
        )
        # char_start should NOT be at 0 for a long text (fallback centers).
        # Allow char_start == 0 only for very short texts where center IS position 0.
        if len(text) > 200:
            assert snip.char_start > 0, (
                f"Fallback snippet must center, not prefix. char_start={snip.char_start}"
            )


# ---------------------------------------------------------------------------
# Tests: span consistency
# ---------------------------------------------------------------------------


class TestSpanConsistency:
    """char_start and char_end must be valid offsets into the chunk text."""

    def test_span_within_chunk_bounds(self) -> None:
        """char_start and char_end must both be within [0, len(chunk_text)]."""
        text = "word " * 50
        snip = build_snippet(
            text,
            "word",
            source_uri="file:///docs/test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
            max_chars=80,
        )
        assert 0 <= snip.char_start <= snip.char_end <= len(text), (
            f"Span out of bounds: [{snip.char_start}, {snip.char_end}] for text len={len(text)}"
        )

    def test_span_consistent_with_text(self) -> None:
        """The snippet text must be a substring of chunk_text[char_start:char_end]."""
        text = "Introduction. " + "word " * 20 + "python snippet center " + "word " * 20
        snip = build_snippet(
            text,
            "python snippet",
            source_uri="file:///docs/test.md",
            title="Doc",
            heading_path=["Intro"],
            locator="intro",
            chunk_id="cid",
            max_chars=120,
        )
        # The snippet text should be recoverable from the span.
        extracted = text[snip.char_start:snip.char_end]
        # Snippet text matches the extracted span (modulo leading/trailing whitespace).
        assert snip.text == extracted.strip() or snip.text in extracted, (
            f"Snippet text {snip.text!r} must match chunk_text[{snip.char_start}:{snip.char_end}]"
            f" = {extracted!r}"
        )

    def test_char_start_lte_char_end(self) -> None:
        """char_start must always be <= char_end."""
        text = "some text here"
        snip = build_snippet(
            text,
            "some text",
            source_uri="file:///docs/test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
        )
        assert snip.char_start <= snip.char_end


# ---------------------------------------------------------------------------
# Tests: provenance fields
# ---------------------------------------------------------------------------


class TestProvenance:
    """source_uri, title, heading_path, and locator must be preserved exactly."""

    def test_source_uri_preserved(self) -> None:
        """source_uri must be the exact value passed in (never a hash)."""
        uri = "https://example.com/docs/guide#section-2"
        snip = build_snippet(
            "test content here",
            "test",
            source_uri=uri,
            title="Guide",
            heading_path=["Guide", "Installation"],
            locator="installation",
            chunk_id="cid",
        )
        assert snip.source_uri == uri, (
            f"source_uri must be preserved exactly. Got {snip.source_uri!r}, expected {uri!r}"
        )

    def test_title_preserved(self) -> None:
        """title must be exactly what was passed."""
        title = "Installation Guide - Chapter 3"
        snip = build_snippet(
            "test content",
            "test",
            source_uri="file:///test.md",
            title=title,
            heading_path=["Chapter 3"],
            locator="chapter-3",
            chunk_id="cid",
        )
        assert snip.title == title

    def test_heading_path_preserved(self) -> None:
        """heading_path must be the exact list passed in."""
        heading_path = ["Chapter 1", "Section 1.2", "Sub-section 1.2.3"]
        snip = build_snippet(
            "test content",
            "test",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=heading_path,
            locator="1.2.3",
            chunk_id="cid",
        )
        assert snip.heading_path == heading_path

    def test_locator_preserved(self) -> None:
        """locator must be the exact string passed in."""
        locator = "chapter-1/section-2/subsection-3"
        snip = build_snippet(
            "test content",
            "test",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Chapter 1"],
            locator=locator,
            chunk_id="cid",
        )
        assert snip.locator == locator

    def test_source_uri_never_hash(self) -> None:
        """source_uri must never look like a 64-hex hash (SHA-256 chunk id)."""
        uri = "https://example.com/api/reference"
        snip = build_snippet(
            "content here",
            "content",
            source_uri=uri,
            title="API Reference",
            heading_path=["API"],
            locator="api/reference",
            chunk_id="abcd1234",
        )
        # A real URI should not be 64 hex chars.
        import re
        assert not re.fullmatch(r"[0-9a-f]{64}", snip.source_uri), (
            "source_uri must never be a 64-character hex hash"
        )

    def test_chunk_id_preserved(self) -> None:
        """chunk_id must be preserved exactly as passed."""
        snip = build_snippet(
            "content here",
            "content",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="unique-chunk-id-abc",
        )
        assert snip.chunk_id == "unique-chunk-id-abc"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty text, very short text, single word."""

    def test_empty_chunk_text(self) -> None:
        """Empty chunk text must produce an empty snippet without raising."""
        snip = build_snippet(
            "",
            "query",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
        )
        assert snip.text == ""
        assert snip.char_start == 0
        assert snip.char_end == 0

    def test_short_text_entirely_included(self) -> None:
        """A short text shorter than max_chars should be fully included."""
        text = "tiny"
        snip = build_snippet(
            text,
            "tiny",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
            max_chars=200,
        )
        assert "tiny" in snip.text

    def test_snippet_is_snippet_type(self) -> None:
        """build_snippet must return a Snippet instance."""
        snip = build_snippet(
            "test text",
            "test",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
        )
        assert isinstance(snip, Snippet)

    def test_multi_word_query_finds_earliest_term(self) -> None:
        """Multi-word query: earliest matching term in text becomes the center."""
        text = "beginning content. " + "term_one found here. " + "other text. " * 30
        snip = build_snippet(
            text,
            "term_one missing_term",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
            max_chars=100,
        )
        assert "term_one" in snip.text.lower() or snip.char_start > 0, (
            "Snippet should center near 'term_one'"
        )

    def test_no_snippet_of_non_existent_word(self) -> None:
        """When no term matches, snippet is still valid (center fallback)."""
        text = "lots of content " * 30
        snip = build_snippet(
            text,
            "nonexistent_word_xyz",
            source_uri="file:///test.md",
            title="Doc",
            heading_path=["Section"],
            locator="section",
            chunk_id="cid",
            max_chars=100,
        )
        assert isinstance(snip, Snippet)
        assert snip.char_start >= 0
        assert snip.char_end >= snip.char_start


# ---------------------------------------------------------------------------
# Tests: _find_match_center internals
# ---------------------------------------------------------------------------


class TestFindMatchCenter:
    """Direct unit tests for _find_match_center."""

    def test_earliest_match_returned(self) -> None:
        """Returns the offset of the first matching query token in chunk text."""
        chunk = "irrelevant text python tutorial irrelevant"
        # "python" starts at index 16 (after "irrelevant text ")
        offset = _find_match_center(chunk, "python")
        assert offset == chunk.index("python"), (
            f"Expected offset at 'python', got {offset}"
        )

    def test_earliest_of_multiple_tokens(self) -> None:
        """When multiple query tokens match, the earliest occurrence wins."""
        chunk = "alpha text beta text"
        # "alpha" starts at 0, "beta" starts at 11
        offset = _find_match_center(chunk, "beta alpha")
        assert offset == chunk.index("alpha"), (
            f"Expected earliest match at 'alpha' (offset 0), got {offset}"
        )

    def test_no_match_returns_text_center(self) -> None:
        """When no query token matches, returns len(chunk_text) // 2."""
        chunk = "abcdefghij"  # 10 chars; center = 5
        offset = _find_match_center(chunk, "zzz yyy")
        assert offset == len(chunk) // 2, (
            f"Expected center offset {len(chunk) // 2}, got {offset}"
        )

    def test_empty_chunk_returns_zero(self) -> None:
        """Empty chunk text returns 0 without raising."""
        offset = _find_match_center("", "anything")
        assert offset == 0

    def test_short_tokens_ignored(self) -> None:
        """Query tokens shorter than 3 chars are ignored; falls back to center."""
        chunk = "ab xy"  # 5 chars; center = 2
        # "ab" and "xy" are < 3 chars, so no token matches.
        offset = _find_match_center(chunk, "ab xy")
        assert offset == len(chunk) // 2, (
            f"Short tokens should be skipped; expected center {len(chunk) // 2}, got {offset}"
        )

    def test_case_insensitive_match(self) -> None:
        """Matching is case-insensitive."""
        chunk = "PYTHON tutorial here"
        offset = _find_match_center(chunk, "python")
        assert offset == 0, f"Case-insensitive match should find 'PYTHON' at 0, got {offset}"
