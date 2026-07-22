"""Unit tests for HeadingAwareChunker."""
from __future__ import annotations

from beacon_kb.ingestion.chunking import HeadingAwareChunker
from beacon_kb.models import (
    ChunkKind,
    RevisionId,
    Section,
    make_chunk_id,
    make_section_id,
    make_source_id,
)
from beacon_kb.testing import ChunkerContract

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = "test-corpus"
URI = "fake://doc-1"
REVISION_ID = "rev-test-001"
PIPELINE = "pipe-v1"


def make_chunker(
    max_tokens: int = 50,
    overlap_tokens: int = 10,
) -> HeadingAwareChunker:
    return HeadingAwareChunker(
        corpus=CORPUS,
        canonical_uri=URI,
        revision_id=REVISION_ID,
        pipeline_fingerprint=PIPELINE,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def make_section(
    text: str,
    locator: str = "intro",
    heading: str = "Introduction",
    ordinal: int = 0,
) -> Section:
    source_id = make_source_id(corpus=CORPUS, canonical_uri=URI)
    revision_id = RevisionId(REVISION_ID)
    section_id = make_section_id(
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
    )


# ---------------------------------------------------------------------------
# ChunkerContract conformance
# ---------------------------------------------------------------------------


class TestHeadingAwareChunkerContract(ChunkerContract):
    """Run the reusable contract suite against HeadingAwareChunker."""

    def make_subject(self) -> HeadingAwareChunker:
        return make_chunker(max_tokens=30, overlap_tokens=5)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_short_section_produces_single_child_chunk() -> None:
    chunker = make_chunker(max_tokens=200)
    section = make_section("Short text that fits in one chunk.")
    chunks = chunker.chunk(section)
    child_chunks = [c for c in chunks if c.kind == ChunkKind.CHILD]
    assert len(child_chunks) == 1
    assert child_chunks[0].text == "Short text that fits in one chunk."


def test_chunk_ids_are_deterministic_across_processes() -> None:
    """Identical inputs must produce identical IDs in independent calls."""
    chunker1 = make_chunker(max_tokens=50, overlap_tokens=10)
    chunker2 = make_chunker(max_tokens=50, overlap_tokens=10)
    section = make_section("Word " * 40)
    ids1 = [c.id for c in chunker1.chunk(section)]
    ids2 = [c.id for c in chunker2.chunk(section)]
    assert ids1 == ids2


def test_chunk_ids_never_random() -> None:
    """Each chunk ID must match make_chunk_id with stable inputs."""
    chunker = make_chunker(max_tokens=50, overlap_tokens=10)
    section = make_section("Word " * 40)
    chunks = chunker.chunk(section)
    for ordinal, chunk in enumerate(c for c in chunks if c.kind == ChunkKind.CHILD):
        expected_id = make_chunk_id(
            corpus=CORPUS,
            canonical_uri=URI,
            revision_id=REVISION_ID,
            pipeline_fingerprint=PIPELINE,
            parent_locator=section.locator,
            child_ordinal=ordinal,
        )
        assert chunk.id == expected_id, (
            f"Chunk ID mismatch at ordinal {ordinal}: expected {expected_id}, got {chunk.id}"
        )


def test_real_token_overlap_between_consecutive_chunks() -> None:
    """Consecutive child chunks share boundary text measured in real tokens, not word count.

    Uses 60 unique multi-token words (``uniqueword{i:04d}`` = 14 chars = 4 heuristic
    tokens each, since ceil(14/4) = 4).
    With overlap_tokens=8 the token budget allows at most 2 such words of overlap.
    A word-slice implementation (``words[-overlap_tokens:]``) would produce 8 words
    of overlap (32 tokens), causing this test to fail.

    For each consecutive pair the test computes the ACTUAL LONGEST SHARED WINDOW -
    the longest word-sequence that is simultaneously a suffix of chunk[N] and a
    prefix of chunk[N+1] (observed from chunker output, not reconstructed from the
    configured budget).
    The token count of that observed window must be > 0 and <= overlap_tokens.
    """
    from beacon_kb.tokens import HeuristicTokenCounter

    overlap = 8
    # 60 unique words, each 14 chars = ceil(14/4) = 4 heuristic tokens.
    # With max_tokens=40 each chunk holds 10 such words (40 tokens / 4 per word).
    # With overlap_tokens=8 the token budget allows exactly 2 words of overlap.
    chunker = make_chunker(max_tokens=40, overlap_tokens=overlap)
    words = [f"uniqueword{i:04d}" for i in range(60)]
    section = make_section(" ".join(words))
    chunks = [c for c in chunker.chunk(section) if c.kind == ChunkKind.CHILD]
    assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

    counter = HeuristicTokenCounter()
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].text.split()
        curr_words = chunks[i].text.split()

        # Find the ACTUAL LONGEST shared window: the longest word-sequence that is
        # simultaneously a suffix of prev_words and a prefix of curr_words.
        # This is observed directly from chunker output - not reconstructed from
        # the configured budget - so a word-slice bug (8 words instead of 2) is
        # exposed because the shared window would be 8 words = 32 tokens > overlap.
        max_possible = min(len(prev_words), len(curr_words))
        shared_words: list[str] = []
        for win in range(max_possible, 0, -1):
            if prev_words[-win:] == curr_words[:win]:
                shared_words = curr_words[:win]
                break

        shared_tokens = counter.count_tokens(" ".join(shared_words)) if shared_words else 0

        assert shared_tokens > 0, (
            f"Chunk {i - 1} -> {i}: no shared boundary text found. "
            f"Prev tail: {prev_words[-3:]!r}, Curr head: {curr_words[:3]!r}"
        )
        assert shared_tokens <= overlap, (
            f"Chunk {i - 1} -> {i}: observed overlap is {shared_tokens} tokens "
            f"({len(shared_words)} words: {shared_words!r}) which exceeds the "
            f"token budget of {overlap}. This indicates word-level slicing instead "
            f"of token-budget overlap."
        )


def test_overlap_not_minimum_length() -> None:
    """overlap_tokens must NOT be treated as a minimum chunk length."""
    chunker = make_chunker(max_tokens=30, overlap_tokens=5)
    # A very short section - should produce a chunk shorter than max_tokens,
    # not be padded to overlap_tokens minimum.
    section = make_section("tiny")
    chunks = [c for c in chunker.chunk(section) if c.kind == ChunkKind.CHILD]
    assert len(chunks) == 1
    assert chunks[0].text == "tiny"


def test_fenced_code_block_not_split() -> None:
    """A fenced code block must not be split across two chunks."""
    chunker = make_chunker(max_tokens=15, overlap_tokens=3)
    text = (
        "Intro line before code.\n"
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n"
        "Outro line after code.\n"
    )
    section = make_section(text)
    chunks = [c for c in chunker.chunk(section) if c.kind == ChunkKind.CHILD]
    # Verify no chunk contains the opening ``` without the matching close.
    for chunk in chunks:
        if "```python" in chunk.text:
            after_open = chunk.text[chunk.text.index("```python") + 9:]
            assert "```" in after_open, (
                f"Chunk contains code-fence open but not close:\n{chunk.text!r}"
            )


def test_neighbor_links_set() -> None:
    """prev_chunk_id and next_chunk_id must link consecutive child chunks."""
    chunker = make_chunker(max_tokens=15, overlap_tokens=3)
    section = make_section("word " * 50)
    chunks = [c for c in chunker.chunk(section) if c.kind == ChunkKind.CHILD]
    assert len(chunks) >= 2

    for i, chunk in enumerate(chunks):
        if i > 0:
            assert chunk.prev_chunk_id == chunks[i - 1].id, (
                f"Chunk {i} prev_chunk_id does not point to chunk {i - 1}"
            )
        if i < len(chunks) - 1:
            assert chunk.next_chunk_id == chunks[i + 1].id, (
                f"Chunk {i} next_chunk_id does not point to chunk {i + 1}"
            )
    assert chunks[0].prev_chunk_id is None
    assert chunks[-1].next_chunk_id is None


def test_parent_locator_set_on_all_chunks() -> None:
    chunker = make_chunker()
    section = make_section("Some text.", locator="api/overview")
    chunks = chunker.chunk(section)
    for chunk in chunks:
        assert chunk.parent_locator == "api/overview"


def test_source_id_derived_via_make_source_id() -> None:
    """Chunk.source_id must equal make_source_id over corpus + canonical URI."""
    chunker = make_chunker()
    section = make_section("content")
    chunks = chunker.chunk(section)
    expected_source_id = make_source_id(corpus=CORPUS, canonical_uri=URI)
    for chunk in chunks:
        assert chunk.source_id == expected_source_id


def test_empty_section_returns_empty_list() -> None:
    chunker = make_chunker()
    section = make_section("")
    result = chunker.chunk(section)
    assert result == []


def test_token_count_field_populated() -> None:
    chunker = make_chunker(max_tokens=100)
    section = make_section("hello world this is a test")
    chunks = [c for c in chunker.chunk(section) if c.kind == ChunkKind.CHILD]
    for chunk in chunks:
        assert chunk.token_count > 0
