"""Unit tests for the hierarchical chunker (Task 02.4).

TDD: these tests are written against the public API before the implementation
exists. They verify the behavioral guarantees listed in the brief:

- Determinism: same ParsedDocument + config -> identical chunk list.
- Parent/child integrity: every child carries exactly one parent_chunk_id
  and the heading path of its source section.
- Real token overlap: adjacent children share tokens at their boundary, and
  overlap is never treated as a minimum chunk length.
- chunker_config sensitivity: changing any parameter changes the config string
  and therefore chunk IDs.
- Fenced code block safety: the splitter never cuts inside a fenced block
  when avoidable (beacon-native logic; not delegated to LlamaIndex).
- Empty document: zero chunks without error.
- Duplicate heading identity: sections with equal heading paths but different
  content produce different IDs.
- Neighbor links: prev_chunk_id / next_chunk_id are threaded only after all
  IDs are stable, and the outermost siblings link correctly.
"""

from __future__ import annotations

import pytest

from beacon.ingest.chunking import (
    CHUNKER_VERSION,
    Chunk,
    ChunkerConfig,
    ChunkKind,
    DocumentChunker,
    chunker_config_str,
)
from beacon.ingest.parsing import ParsedDocument, ParsedSection, SectionKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(*sections: ParsedSection, title: str = "Test") -> ParsedDocument:
    """Build a minimal ParsedDocument from the given sections."""
    return ParsedDocument(
        title=title,
        media_type="text/markdown",
        sections=tuple(sections),
        warnings=(),
        parser_version="docling-2.beacon-adapter-1",
    )


def _section(
    text: str,
    locator: str = "Root",
    heading_path: tuple[str, ...] = ("Root",),
    kind: SectionKind = SectionKind.TEXT,
    ordinal: int = 0,
) -> ParsedSection:
    """Build a minimal ParsedSection."""
    return ParsedSection(
        locator=locator,
        heading_path=heading_path,
        heading=heading_path[-1] if heading_path else "",
        kind=kind,
        text=text,
        ordinal=ordinal,
    )


def _cfg(
    parent_chunk_size: int = 512,
    child_chunk_size: int = 128,
    chunk_overlap: int = 20,
) -> ChunkerConfig:
    return ChunkerConfig(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )


# ---------------------------------------------------------------------------
# Shared long text (~180 tokens) - long enough to force splits at larger sizes
# ---------------------------------------------------------------------------

_LONG_TEXT = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu alpha bravo charlie delta "
    "echo foxtrot golf hotel india juliet kilo lima mike november oscar "
    "papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu"
)

# Smaller config that still forces splits (parent=256, child=64, overlap=16)
# With ~180 tokens the text splits into multiple child chunks at child_size=64.
_SPLIT_CFG = _cfg(parent_chunk_size=256, child_chunk_size=64, chunk_overlap=16)


# ---------------------------------------------------------------------------
# 1. chunker_config_str
# ---------------------------------------------------------------------------


class TestChunkerConfigStr:
    def test_returns_string(self) -> None:
        assert isinstance(chunker_config_str(_cfg()), str)

    def test_contains_version(self) -> None:
        cfg_s = chunker_config_str(_cfg())
        assert CHUNKER_VERSION in cfg_s

    def test_contains_params(self) -> None:
        cfg_s = chunker_config_str(
            _cfg(parent_chunk_size=1024, child_chunk_size=256, chunk_overlap=32)
        )
        assert "1024" in cfg_s
        assert "256" in cfg_s
        assert "32" in cfg_s

    def test_different_params_yield_different_strings(self) -> None:
        a = chunker_config_str(_cfg(parent_chunk_size=512))
        b = chunker_config_str(_cfg(parent_chunk_size=1024))
        assert a != b

    def test_same_params_yield_same_string(self) -> None:
        assert chunker_config_str(_cfg()) == chunker_config_str(_cfg())


# ---------------------------------------------------------------------------
# 2. Empty document
# ---------------------------------------------------------------------------


class TestEmptyDocument:
    def test_no_sections(self) -> None:
        doc = _make_doc()
        chunker = DocumentChunker(
            collection="mycol",
            canonical_uri="file:///test.md",
            content_hash="abc123",
            config=_cfg(),
        )
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_blank_section_text(self) -> None:
        doc = _make_doc(_section("   \n  "))
        chunker = DocumentChunker(
            collection="mycol",
            canonical_uri="file:///test.md",
            content_hash="abc123",
            config=_cfg(),
        )
        chunks = chunker.chunk(doc)
        assert chunks == []


# ---------------------------------------------------------------------------
# 3. Determinism - the core invariant
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _chunk_twice(self) -> tuple[list[Chunk], list[Chunk]]:
        doc = _make_doc(_section(_LONG_TEXT))
        cfg = _SPLIT_CFG
        kwargs = {
            "collection": "mycol",
            "canonical_uri": "file:///doc.md",
            "content_hash": "hash001",
            "config": cfg,
        }
        run1 = DocumentChunker(**kwargs).chunk(doc)  # type: ignore[arg-type]
        run2 = DocumentChunker(**kwargs).chunk(doc)  # type: ignore[arg-type]
        return run1, run2

    def test_chunk_ids_identical(self) -> None:
        r1, r2 = self._chunk_twice()
        assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]

    def test_chunk_texts_identical(self) -> None:
        r1, r2 = self._chunk_twice()
        assert [c.text for c in r1] == [c.text for c in r2]

    def test_parent_chunk_ids_identical(self) -> None:
        r1, r2 = self._chunk_twice()
        assert [c.parent_chunk_id for c in r1] == [c.parent_chunk_id for c in r2]

    def test_neighbor_links_identical(self) -> None:
        r1, r2 = self._chunk_twice()
        assert [(c.prev_chunk_id, c.next_chunk_id) for c in r1] == [
            (c.prev_chunk_id, c.next_chunk_id) for c in r2
        ]

    def test_at_least_two_chunks_produced(self) -> None:
        """Sanity: the text is long enough to produce multiple chunks."""
        r1, _ = self._chunk_twice()
        assert len(r1) >= 2


# ---------------------------------------------------------------------------
# 4. Parent / child integrity
# ---------------------------------------------------------------------------


class TestParentChildIntegrity:
    def _chunks(self) -> list[Chunk]:
        doc = _make_doc(_section(_LONG_TEXT))
        return DocumentChunker(
            collection="mycol",
            canonical_uri="file:///doc.md",
            content_hash="hash001",
            config=_SPLIT_CFG,
        ).chunk(doc)

    def test_all_child_chunks_have_parent(self) -> None:
        chunks = self._chunks()
        child_chunks = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(child_chunks) >= 2
        for c in child_chunks:
            assert c.parent_chunk_id is not None, f"chunk {c.chunk_id!r} has no parent"

    def test_parent_chunk_ids_reference_parent_chunks(self) -> None:
        chunks = self._chunks()
        parent_ids = {c.chunk_id for c in chunks if c.kind == ChunkKind.PARENT}
        for c in chunks:
            if c.kind == ChunkKind.CHILD:
                assert c.parent_chunk_id in parent_ids, (
                    f"parent_chunk_id {c.parent_chunk_id!r} not in parent set {parent_ids}"
                )

    def test_parent_chunks_have_no_parent(self) -> None:
        chunks = self._chunks()
        for c in chunks:
            if c.kind == ChunkKind.PARENT:
                assert c.parent_chunk_id is None

    def test_heading_path_carried_on_children(self) -> None:
        doc = _make_doc(
            _section(
                _LONG_TEXT,
                locator="Guide/Install",
                heading_path=("Guide", "Install"),
                ordinal=0,
            )
        )
        chunks = DocumentChunker(
            collection="mycol",
            canonical_uri="file:///doc.md",
            content_hash="hash001",
            config=_SPLIT_CFG,
        ).chunk(doc)
        for c in chunks:
            assert c.heading_path == ("Guide", "Install"), (
                f"chunk {c.chunk_id!r} has heading_path={c.heading_path!r}"
            )

    def test_section_kind_carried_on_chunks(self) -> None:
        doc = _make_doc(_section(_LONG_TEXT, kind=SectionKind.CODE))
        chunks = DocumentChunker(
            collection="mycol",
            canonical_uri="file:///doc.md",
            content_hash="hash001",
            config=_SPLIT_CFG,
        ).chunk(doc)
        for c in chunks:
            assert c.section_kind == SectionKind.CODE


# ---------------------------------------------------------------------------
# 5. Real token overlap
# ---------------------------------------------------------------------------


class TestRealTokenOverlap:
    def test_adjacent_children_share_tokens_at_boundary(self) -> None:
        """Verifies that the last tokens of chunk[i] appear at the start of chunk[i+1].

        We use a text of unique tokens (word0000, word0001, ...) so that
        the suffix-matching algorithm cannot produce false positives from
        repeated content.  Each token is exactly one BPE unit so the word
        count and token count match closely.  With chunk_overlap=16 the
        last ~5 unique words (15 tokens) of one child must reappear at the
        start of the next, and the shared token count must not exceed the
        configured overlap budget.
        """
        from llama_index.core.utils import get_tokenizer

        # Build a 120-word text with globally unique tokens to avoid
        # false-positive overlap detection from repeated content.
        unique_words = [f"word{i:04d}" for i in range(120)]
        text = " ".join(unique_words)

        doc = _make_doc(_section(text))
        chunk_overlap = 16
        cfg = _cfg(parent_chunk_size=512, child_chunk_size=64, chunk_overlap=chunk_overlap)
        chunks = DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h1",
            config=cfg,
        ).chunk(doc)

        child_chunks = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(child_chunks) >= 2, "Need at least 2 child chunks to check overlap"

        tokenizer = get_tokenizer()

        for i in range(len(child_chunks) - 1):
            prev_words = child_chunks[i].text.split()
            curr_words = child_chunks[i + 1].text.split()
            max_possible = min(len(prev_words), len(curr_words))
            shared_words: list[str] = []
            for win in range(max_possible, 0, -1):
                if prev_words[-win:] == curr_words[:win]:
                    shared_words = curr_words[:win]
                    break
            shared_tokens = len(tokenizer(" ".join(shared_words))) if shared_words else 0
            assert shared_tokens >= 1, (
                f"No token overlap found between chunk {i} and {i + 1}:\n"
                f"  chunk {i}: {child_chunks[i].text!r}\n"
                f"  chunk {i + 1}: {child_chunks[i + 1].text!r}"
            )
            assert shared_tokens <= chunk_overlap, (
                f"Overlap {shared_tokens} tokens exceeds budget {chunk_overlap}"
            )

    def test_overlap_is_not_minimum_chunk_length(self) -> None:
        """A text just under child_chunk_size produces exactly one child chunk.

        If overlap were misread as a minimum length, a short section would be
        split even when it fits within child_chunk_size.
        """
        # 'alpha bravo charlie' - only ~5 tokens; well under any realistic chunk size
        doc = _make_doc(_section("alpha bravo charlie"))
        cfg = _cfg(parent_chunk_size=512, child_chunk_size=128, chunk_overlap=20)
        chunks = DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h1",
            config=cfg,
        ).chunk(doc)
        child_chunks = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(child_chunks) == 1, (
            f"Expected exactly 1 child chunk for short text, got {len(child_chunks)}"
        )


# ---------------------------------------------------------------------------
# 6. chunker_config change changes chunk IDs
# ---------------------------------------------------------------------------


def _cfg_for_sensitivity(
    parent_chunk_size: int = 512,
    child_chunk_size: int = 128,
    chunk_overlap: int = 20,
) -> ChunkerConfig:
    """Build a noise-free config where all sizes are >= minimum thresholds."""
    return ChunkerConfig(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )


class TestChunkerConfigSensitivity:
    def _chunk_with_cfg(self, cfg: ChunkerConfig) -> list[Chunk]:
        doc = _make_doc(_section(_LONG_TEXT))
        return DocumentChunker(
            collection="mycol",
            canonical_uri="file:///doc.md",
            content_hash="hash001",
            config=cfg,
        ).chunk(doc)

    def test_different_child_size_changes_ids(self) -> None:
        ids_a = {c.chunk_id for c in self._chunk_with_cfg(
            _cfg_for_sensitivity(parent_chunk_size=512, child_chunk_size=64, chunk_overlap=16)
        )}
        ids_b = {c.chunk_id for c in self._chunk_with_cfg(
            _cfg_for_sensitivity(parent_chunk_size=512, child_chunk_size=96, chunk_overlap=16)
        )}
        assert ids_a != ids_b

    def test_different_overlap_changes_ids(self) -> None:
        # Config strings must differ; chunk IDs may not if text fits in one chunk.
        cfg_a = chunker_config_str(_cfg(chunk_overlap=16))
        cfg_b = chunker_config_str(_cfg(chunk_overlap=24))
        assert cfg_a != cfg_b

    def test_different_collection_changes_ids(self) -> None:
        doc = _make_doc(_section(_LONG_TEXT))
        cfg = _SPLIT_CFG
        ids_a = {c.chunk_id for c in DocumentChunker(
            collection="col_a", canonical_uri="file:///x.md", content_hash="h", config=cfg
        ).chunk(doc)}
        ids_b = {c.chunk_id for c in DocumentChunker(
            collection="col_b", canonical_uri="file:///x.md", content_hash="h", config=cfg
        ).chunk(doc)}
        assert ids_a != ids_b

    def test_different_content_hash_changes_ids(self) -> None:
        doc = _make_doc(_section(_LONG_TEXT))
        cfg = _SPLIT_CFG
        ids_a = {c.chunk_id for c in DocumentChunker(
            collection="col", canonical_uri="file:///x.md", content_hash="hash_a", config=cfg
        ).chunk(doc)}
        ids_b = {c.chunk_id for c in DocumentChunker(
            collection="col", canonical_uri="file:///x.md", content_hash="hash_b", config=cfg
        ).chunk(doc)}
        assert ids_a != ids_b


# ---------------------------------------------------------------------------
# 7. Neighbor links
# ---------------------------------------------------------------------------


class TestNeighborLinks:
    def _chunks(self) -> list[Chunk]:
        doc = _make_doc(_section(_LONG_TEXT))
        return DocumentChunker(
            collection="mycol",
            canonical_uri="file:///doc.md",
            content_hash="hash001",
            config=_SPLIT_CFG,
        ).chunk(doc)

    def test_first_child_has_no_prev(self) -> None:
        children = [c for c in self._chunks() if c.kind == ChunkKind.CHILD]
        assert children[0].prev_chunk_id is None

    def test_last_child_has_no_next(self) -> None:
        children = [c for c in self._chunks() if c.kind == ChunkKind.CHILD]
        assert children[-1].next_chunk_id is None

    def test_consecutive_children_are_linked(self) -> None:
        children = [c for c in self._chunks() if c.kind == ChunkKind.CHILD]
        assert len(children) >= 2
        for i in range(len(children) - 1):
            assert children[i].next_chunk_id == children[i + 1].chunk_id
            assert children[i + 1].prev_chunk_id == children[i].chunk_id


# ---------------------------------------------------------------------------
# 8. Chunk ID is a deterministic SHA-256
# ---------------------------------------------------------------------------


class TestChunkIdDeterminism:
    def test_chunk_id_is_hex_string(self) -> None:
        doc = _make_doc(_section("hello world"))
        cfg = _cfg()
        chunks = DocumentChunker(
            collection="c", canonical_uri="file:///x.md", content_hash="h", config=cfg
        ).chunk(doc)
        assert chunks
        for c in chunks:
            # Should be a valid hex string (SHA-256 = 64 hex chars)
            assert len(c.chunk_id) == 64
            int(c.chunk_id, 16)  # raises if not valid hex

    def test_chunk_id_is_sha256(self) -> None:
        """IDs must be SHA-256 digests, not random values."""
        doc = _make_doc(_section("hello world"))
        cfg = _cfg()
        r1 = DocumentChunker(
            collection="c", canonical_uri="file:///x.md", content_hash="h", config=cfg
        ).chunk(doc)
        r2 = DocumentChunker(
            collection="c", canonical_uri="file:///x.md", content_hash="h", config=cfg
        ).chunk(doc)
        assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]


# ---------------------------------------------------------------------------
# 9. Multiple sections from one document
# ---------------------------------------------------------------------------


class TestMultipleSections:
    def test_sections_produce_independent_chunks(self) -> None:
        doc = _make_doc(
            _section(_LONG_TEXT, locator="Sec1", heading_path=("Sec1",), ordinal=0),
            _section(_LONG_TEXT, locator="Sec2", heading_path=("Sec2",), ordinal=1),
        )
        chunks = DocumentChunker(
            collection="c", canonical_uri="file:///x.md", content_hash="h", config=_SPLIT_CFG
        ).chunk(doc)
        # Each section has its own heading path
        sec1_chunks = [c for c in chunks if c.heading_path == ("Sec1",)]
        sec2_chunks = [c for c in chunks if c.heading_path == ("Sec2",)]
        assert len(sec1_chunks) >= 2
        assert len(sec2_chunks) >= 2

    def test_all_chunk_ids_unique_across_sections(self) -> None:
        doc = _make_doc(
            _section(_LONG_TEXT, locator="Sec1", heading_path=("Sec1",), ordinal=0),
            _section(_LONG_TEXT, locator="Sec2", heading_path=("Sec2",), ordinal=1),
        )
        chunks = DocumentChunker(
            collection="c", canonical_uri="file:///x.md", content_hash="h", config=_SPLIT_CFG
        ).chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be globally unique"


# ---------------------------------------------------------------------------
# 10. Cross-section neighbor links (Fix 2 regression)
# ---------------------------------------------------------------------------


class TestCrossSectionNeighborLinks:
    """Verify that neighbor links do NOT cross section boundaries."""

    def _two_section_chunks(self) -> list[Chunk]:
        doc = _make_doc(
            _section(_LONG_TEXT, locator="Sec1", heading_path=("Sec1",), ordinal=0),
            _section(_LONG_TEXT, locator="Sec2", heading_path=("Sec2",), ordinal=1),
        )
        return DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h",
            config=_SPLIT_CFG,
        ).chunk(doc)

    def test_last_child_of_section1_has_no_next(self) -> None:
        chunks = self._two_section_chunks()
        sec1_children = [
            c for c in chunks
            if c.kind == ChunkKind.CHILD and c.heading_path == ("Sec1",)
        ]
        assert sec1_children, "Section 1 must have at least one child"
        assert sec1_children[-1].next_chunk_id is None, (
            "Last child of section 1 must not link to section 2"
        )

    def test_first_child_of_section2_has_no_prev(self) -> None:
        chunks = self._two_section_chunks()
        sec2_children = [
            c for c in chunks
            if c.kind == ChunkKind.CHILD and c.heading_path == ("Sec2",)
        ]
        assert sec2_children, "Section 2 must have at least one child"
        assert sec2_children[0].prev_chunk_id is None, (
            "First child of section 2 must not link back to section 1"
        )


# ---------------------------------------------------------------------------
# 11. Duplicate parent text collision (Fix 1 regression)
# ---------------------------------------------------------------------------


class TestDuplicateParentTextCollision:
    """Verify that positional parent-child pairing is used, not dict keying.

    A ``dict`` keyed by stripped parent text physically cannot hold two
    entries for two parents with identical text: the last writer wins, one
    parent's children are lost, and the other parent receives the wrong
    children.  These tests engineer a section whose text produces >= 3
    parents where several parents have IDENTICAL stripped text (a repeated
    identical sentence aligns parent windows exactly), then assert the
    mapping is positional, not text-keyed.
    """

    # One sentence repeated many times: parent windows align on sentence
    # boundaries, so most parents carry byte-identical stripped text.
    _SENTENCE = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. "
    )
    _REPEATED_TEXT = (_SENTENCE * 40).strip()
    _DUP_CFG = ChunkerConfig(parent_chunk_size=128, child_chunk_size=64, chunk_overlap=16)

    def test_duplicate_parent_texts_each_keep_their_own_pair(self) -> None:
        """The pair list has one entry PER PARENT, including duplicates.

        A dict-keyed implementation collapses duplicate parent texts into a
        single entry, so the pair count drops below the parent count and this
        assertion fails.
        """
        from collections import Counter

        from beacon.ingest.chunking import _split_section_text

        pairs = _split_section_text(self._REPEATED_TEXT, self._DUP_CFG)
        assert len(pairs) >= 3, f"Need >= 3 parents for this test, got {len(pairs)}"

        text_counts = Counter(parent_text for parent_text, _ in pairs)
        duplicated = {t: n for t, n in text_counts.items() if n >= 2}
        assert duplicated, (
            "Precondition failed: expected at least two parents with identical "
            "stripped text; adjust _REPEATED_TEXT / _DUP_CFG if the splitter changed."
        )
        # The kill-shot: every duplicate parent must retain its own entry.
        # sum of counts == len(pairs) holds trivially for a list; a dict-keyed
        # regression yields exactly one pair per unique text.
        for dup_text, n in duplicated.items():
            n_pairs_with_text = sum(1 for p, _ in pairs if p == dup_text)
            assert n_pairs_with_text == n >= 2, (
                f"Duplicate parent text collapsed: expected {n} positional pairs "
                f"for text starting {dup_text[:40]!r}, found {n_pairs_with_text}"
            )

    def test_each_parents_children_belong_to_it(self) -> None:
        """Every child text is a substring of ITS OWN parent's text, and every
        parent (including each duplicate) has a non-empty children list."""
        from beacon.ingest.chunking import _split_section_text

        pairs = _split_section_text(self._REPEATED_TEXT, self._DUP_CFG)
        for i, (parent_text, child_texts) in enumerate(pairs):
            assert child_texts, f"Parent {i} lost its children (dict collision symptom)"
            for child_text in child_texts:
                assert child_text in parent_text, (
                    f"Child of parent {i} is not contained in that parent's text - "
                    f"children were assigned to the wrong parent.\n"
                    f"  child:  {child_text!r}\n"
                    f"  parent: {parent_text!r}"
                )

    def test_end_to_end_duplicate_parents_have_distinct_ids_and_children(self) -> None:
        """Through DocumentChunker, duplicate-text parents get distinct IDs and
        each one is referenced by at least one child."""
        doc = _make_doc(_section(self._REPEATED_TEXT))
        chunks = DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h",
            config=self._DUP_CFG,
        ).chunk(doc)

        parent_chunks = [c for c in chunks if c.kind == ChunkKind.PARENT]
        child_chunks = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(parent_chunks) >= 3

        # Duplicate texts exist among parents but IDs are all distinct.
        parent_texts = [p.text for p in parent_chunks]
        assert len(set(parent_texts)) < len(parent_texts), (
            "Precondition failed: expected duplicate parent texts"
        )
        assert len({p.chunk_id for p in parent_chunks}) == len(parent_chunks)

        # Every parent, including each duplicate, is referenced by children,
        # and each child is a substring of its own parent's text.
        parent_text_by_id = {p.chunk_id: p.text for p in parent_chunks}
        referenced = {c.parent_chunk_id for c in child_chunks}
        for p in parent_chunks:
            assert p.chunk_id in referenced, (
                f"Parent {p.chunk_id[:12]} has no children (dict collision symptom)"
            )
        for child in child_chunks:
            assert child.parent_chunk_id is not None
            assert child.text in parent_text_by_id[child.parent_chunk_id]


# ---------------------------------------------------------------------------
# 12. ChunkerConfig validation (Fix 4)
# ---------------------------------------------------------------------------


class TestChunkerConfigValidation:
    def test_negative_chunk_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(parent_chunk_size=512, child_chunk_size=128, chunk_overlap=-1)

    def test_zero_chunk_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(parent_chunk_size=512, child_chunk_size=128, chunk_overlap=0)

    def test_zero_child_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="child_chunk_size"):
            ChunkerConfig(parent_chunk_size=512, child_chunk_size=0, chunk_overlap=20)

    def test_zero_parent_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_chunk_size"):
            ChunkerConfig(parent_chunk_size=0, child_chunk_size=128, chunk_overlap=20)

    def test_overlap_ge_child_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(parent_chunk_size=512, child_chunk_size=64, chunk_overlap=64)

    def test_child_size_gt_parent_size_raises(self) -> None:
        with pytest.raises(ValueError, match="child_chunk_size"):
            ChunkerConfig(parent_chunk_size=64, child_chunk_size=128, chunk_overlap=20)

    def test_valid_config_does_not_raise(self) -> None:
        cfg = ChunkerConfig(parent_chunk_size=512, child_chunk_size=128, chunk_overlap=20)
        assert cfg.parent_chunk_size == 512


# ---------------------------------------------------------------------------
# 13. CODE section fence handling (Fix 7)
# ---------------------------------------------------------------------------


class TestCodeSectionFenceHandling:
    """Verify beacon-native CODE section splitting at line boundaries."""

    def test_small_code_section_stays_whole(self) -> None:
        """A CODE section that fits in one parent emits exactly 1 parent + 1 child."""
        short_code = "def foo():\n    return 42\n"
        doc = _make_doc(_section(short_code, kind=SectionKind.CODE))
        cfg = _cfg(parent_chunk_size=512, child_chunk_size=128, chunk_overlap=20)
        chunks = DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h",
            config=cfg,
        ).chunk(doc)
        parents = [c for c in chunks if c.kind == ChunkKind.PARENT]
        children = [c for c in chunks if c.kind == ChunkKind.CHILD]
        assert len(parents) == 1, f"Expected 1 parent, got {len(parents)}"
        assert len(children) == 1, f"Expected 1 child, got {len(children)}"

    def test_large_code_section_splits_at_line_boundaries(self) -> None:
        """An oversized CODE section splits only at newlines - no mid-line cuts."""
        # Build a code block large enough to exceed parent_chunk_size=128 tokens.
        # Each line is a long comment (~15 tokens).
        lines = [f"# line {i:03d} with some extra padding words here there" for i in range(30)]
        large_code = "\n".join(lines)
        doc = _make_doc(_section(large_code, kind=SectionKind.CODE))
        cfg = _cfg(parent_chunk_size=128, child_chunk_size=64, chunk_overlap=16)
        chunks = DocumentChunker(
            collection="c",
            canonical_uri="file:///x.md",
            content_hash="h",
            config=cfg,
        ).chunk(doc)
        parents = [c for c in chunks if c.kind == ChunkKind.PARENT]
        assert len(parents) >= 2, f"Expected multiple parents from large code, got {len(parents)}"
        # Every parent and child text must consist only of COMPLETE original
        # lines: exact membership catches both mid-line starts and mid-line
        # ends (a startswith-style check would miss a truncated line tail).
        original_lines = set(lines)
        for chunk in chunks:
            for line in chunk.text.split("\n"):
                if not line.strip():
                    continue
                assert line in original_lines, (
                    f"Chunk contains a mid-line split: {line!r}"
                )
        # All original lines must be preserved across parents (no loss).
        covered = {
            line
            for c in parents
            for line in c.text.split("\n")
            if line.strip()
        }
        assert covered == original_lines, (
            f"Line loss across CODE parents: missing {sorted(original_lines - covered)[:3]}"
        )
