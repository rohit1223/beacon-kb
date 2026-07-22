"""Heading-aware parent/child chunker with real token overlap.

Design notes:
- Chunk identity uses make_chunk_id(corpus, canonical_uri, revision_id,
  pipeline_fingerprint, parent_locator, child_ordinal) - never random values.
- SourceId derivation: make_source_id(corpus=corpus, canonical_uri=canonical_uri)
  where canonical_uri is the connector's stable URI for the document.
  This matches the FilterSpec.source_uris hash contract: the source_id hash
  is computed from corpus + canonical_uri so that FilterSpec can reproduce the
  same ID without access to the original document object.
- Real token overlap: the last overlap_tokens tokens of chunk[i] are
  repeated as the first tokens of chunk[i+1]. This is measured by
  HeuristicTokenCounter.count_tokens(), not a character-length minimum.
- Fenced code blocks (``` ... ```) are never split: if a split boundary
  falls inside a block, the boundary is moved to after the closing fence.
- Neighbor links (prev_chunk_id, next_chunk_id) are threaded after all
  child chunk IDs are stable.

Importing this module performs no side effects.
"""
from __future__ import annotations

import re

from beacon_kb.errors import IngestionError
from beacon_kb.models import (
    Chunk,
    ChunkKind,
    Section,
    make_chunk_id,
    make_source_id,
)
from beacon_kb.tokens import HeuristicTokenCounter

# ---------------------------------------------------------------------------
# Fenced code block detection
# ---------------------------------------------------------------------------


def _build_fence_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Return list of (start_line, end_line) ranges for fenced code blocks.

    Args:
        lines: Text lines (from splitlines).

    Returns:
        List of (start, end) tuples inclusive, where lines[start..end] are
        inside a fenced code block.
    """
    fence_ranges: list[tuple[int, int]] = []
    in_fence = False
    fence_char: str = ""
    fence_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_fence:
            m = re.match(r"^(`{3,}|~{3,})", stripped)
            if m:
                in_fence = True
                fence_char = m.group(1)[0]
                fence_start = i
        else:
            m = re.match(r"^(`{3,}|~{3,})", stripped)
            if m and m.group(1)[0] == fence_char:
                fence_ranges.append((fence_start, i))
                in_fence = False

    # Unclosed fence: treat rest of file as fenced.
    if in_fence:
        fence_ranges.append((fence_start, len(lines) - 1))

    return fence_ranges


def _line_in_fence(line_idx: int, fence_ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line_idx <= end for start, end in fence_ranges)


def _split_by_words_with_overlap(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    counter: HeuristicTokenCounter,
) -> list[str]:
    """Split text at word boundaries with token overlap.

    Used as a fallback when a single line exceeds max_tokens.
    Produces word-level chunks with exact token overlap.

    Args:
        text:           Text to split.
        max_tokens:     Maximum tokens per chunk.
        overlap_tokens: Tokens to repeat at boundaries.
        counter:        Token counter instance.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        # Greedily accumulate words until we exceed max_tokens.
        end = start
        token_count = 0
        while end < len(words):
            word_tok = counter.count_tokens(words[end])
            if token_count + word_tok > max_tokens and end > start:
                break
            token_count += word_tok
            end += 1

        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        # Next chunk starts overlap_tokens tokens back from end.
        overlap_start = end
        overlap_count = 0
        while overlap_start > start and overlap_count < overlap_tokens:
            overlap_count += counter.count_tokens(words[overlap_start - 1])
            overlap_start -= 1

        start = overlap_start

    return chunks


def _split_preserving_fences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into token-bounded chunks, never splitting inside a fenced code block.

    Algorithm:
    1. Identify all fenced code block intervals [open_line, close_line] in the text.
    2. Walk lines, accumulating into a current chunk buffer.
    3. When the buffer exceeds max_tokens, find the safe split boundary:
       a. If the split point is inside a fence interval, push the boundary
          to after the closing fence line.
       b. Otherwise, split at the current position.
    4. The next chunk begins by repeating the last overlap_tokens tokens from
       the previous chunk.
    5. If a single line exceeds max_tokens, fall through to word-level splitting.

    Args:
        text:           Raw section text to split.
        max_tokens:     Maximum tokens per chunk (inclusive).
        overlap_tokens: Number of tokens to repeat at the start of each chunk.

    Returns:
        Ordered list of chunk text strings (may be empty if text is blank).
    """
    if not text or not text.strip():
        return []

    counter = HeuristicTokenCounter()

    # Fast path: entire text fits in one chunk.
    total_tokens = counter.count_tokens(text)
    if total_tokens <= max_tokens:
        return [text.strip()]

    # Check if text has no newlines (single long line) - fall through to word splitting.
    lines = text.splitlines(keepends=True)
    if len(lines) == 1:
        return _split_by_words_with_overlap(
            text.strip(), max_tokens, overlap_tokens, counter
        )

    fence_ranges = _build_fence_ranges([ln.rstrip("\n").rstrip("\r\n") for ln in lines])

    def _line_in_fence_local(line_idx: int) -> bool:
        return _line_in_fence(line_idx, fence_ranges)

    chunks: list[str] = []
    current_lines: list[tuple[int, str]] = []
    current_tokens = 0
    overlap_tail: list[str] = []

    def _flush(force_lines: list[tuple[int, str]] | None = None) -> None:
        nonlocal current_lines, current_tokens, overlap_tail
        target = force_lines if force_lines is not None else current_lines
        if not target:
            return
        chunk_text = "".join(ln for _, ln in target).rstrip()
        if chunk_text:
            chunks.append(chunk_text)
            # Build overlap_tail by walking words from the right and
            # accumulating until we hit the token budget.  This mirrors
            # the approach in _split_by_words_with_overlap and measures
            # real tokens via the injected counter rather than word count.
            if overlap_tokens > 0:
                words = chunk_text.split()
                tail_start = len(words)
                tail_count = 0
                while tail_start > 0 and tail_count < overlap_tokens:
                    tail_count += counter.count_tokens(words[tail_start - 1])
                    tail_start -= 1
                overlap_tail = words[tail_start:]
            else:
                overlap_tail = []
        current_lines = []
        current_tokens = 0

    line_records: list[tuple[int, str]] = list(enumerate(lines))
    i = 0
    while i < len(line_records):
        line_idx, line_text = line_records[i]
        line_tok = counter.count_tokens(line_text)

        if current_tokens + line_tok > max_tokens and current_lines:
            last_line_idx = current_lines[-1][0]
            in_fence_now = _line_in_fence_local(last_line_idx)
            if in_fence_now:
                # Find the end of the current fence block and absorb to that point.
                fence_end_line = last_line_idx
                for fstart, fend in fence_ranges:
                    if fstart <= last_line_idx <= fend:
                        fence_end_line = fend
                        break
                while i < len(line_records) and line_records[i][0] <= fence_end_line:
                    _, extra_line = line_records[i]
                    current_lines.append(line_records[i])
                    current_tokens += counter.count_tokens(extra_line)
                    i += 1
                _flush()
                if overlap_tail and i < len(line_records):
                    overlap_text = " ".join(overlap_tail) + "\n"
                    current_lines = [(-1, overlap_text)]
                    current_tokens = counter.count_tokens(overlap_text)
                continue
            else:
                _flush()
                if overlap_tail:
                    overlap_text = " ".join(overlap_tail) + "\n"
                    current_lines = [(-1, overlap_text)]
                    current_tokens = counter.count_tokens(overlap_text)
                # Re-process this line without incrementing i.
                continue

        current_lines.append((line_idx, line_text))
        current_tokens += line_tok
        i += 1

    _flush()
    return chunks


class HeadingAwareChunker:
    """Heading-aware parent/child chunker with real token overlap.

    Each call to chunk() processes one Section and returns a list of CHILD
    Chunk records with deterministic content-addressed IDs. Neighbor links
    (prev_chunk_id, next_chunk_id) are threaded across the returned list.

    SourceId is derived via::

        make_source_id(corpus=self._corpus, canonical_uri=self._canonical_uri)

    This matches the FilterSpec.source_uris hash contract: the source_id hash
    is computed from corpus + canonical_uri so that FilterSpec can reproduce
    the same ID from the connector's URI alone.

    Args:
        corpus:               Corpus name (used in make_chunk_id and make_source_id).
        canonical_uri:        Connector's stable URI for the document.
        revision_id:          Revision being indexed.
        pipeline_fingerprint: Hash of pipeline configuration (chunker params etc.).
        max_tokens:           Maximum tokens per child chunk (default 512).
        overlap_tokens:       Tokens to repeat at boundary between consecutive
                              child chunks. Must be less than max_tokens. (default 64)
    """

    def __init__(
        self,
        *,
        corpus: str,
        canonical_uri: str,
        revision_id: str,
        pipeline_fingerprint: str,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise IngestionError(
                f"overlap_tokens ({overlap_tokens}) must be less than max_tokens ({max_tokens})."
            )
        self._corpus = corpus
        self._canonical_uri = canonical_uri
        self._revision_id = revision_id
        self._pipeline_fingerprint = pipeline_fingerprint
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._counter = HeuristicTokenCounter()
        self._source_id = make_source_id(corpus=corpus, canonical_uri=canonical_uri)

    def chunk(self, section: Section) -> list[Chunk]:
        """Split a section into an ordered list of child Chunk records.

        Returns an empty list for blank sections. All returned chunks have
        kind=CHILD and carry content-addressed IDs. Neighbor links are
        threaded after ID assignment.

        Args:
            section: A Section record from the parser.

        Returns:
            Ordered list of Chunk records with stable content-addressed IDs.

        Raises:
            IngestionError: If chunking produces an internal inconsistency.
        """
        text = (section.text or "").strip()
        if not text:
            return []

        raw_chunks = _split_preserving_fences(
            text,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        if not raw_chunks:
            return []

        # Build child chunks with stable IDs (ordinal-based).
        chunks: list[Chunk] = []
        for ordinal, chunk_text in enumerate(raw_chunks):
            chunk_id = make_chunk_id(
                corpus=self._corpus,
                canonical_uri=self._canonical_uri,
                revision_id=self._revision_id,
                pipeline_fingerprint=self._pipeline_fingerprint,
                parent_locator=section.locator,
                child_ordinal=ordinal,
            )
            chunk = Chunk(
                id=chunk_id,
                source_id=self._source_id,
                revision_id=section.revision_id,
                section_id=section.id,
                text=chunk_text,
                ordinal=ordinal,
                parent_locator=section.locator,
                kind=ChunkKind.CHILD,
                token_count=self._counter.count_tokens(chunk_text),
                prev_chunk_id=None,
                next_chunk_id=None,
            )
            chunks.append(chunk)

        # Thread neighbor links by replacing frozen chunks with updated copies.
        linked: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            prev_id = chunks[i - 1].id if i > 0 else None
            next_id = chunks[i + 1].id if i < len(chunks) - 1 else None
            linked.append(
                Chunk(
                    id=chunk.id,
                    source_id=chunk.source_id,
                    revision_id=chunk.revision_id,
                    section_id=chunk.section_id,
                    text=chunk.text,
                    ordinal=chunk.ordinal,
                    parent_locator=chunk.parent_locator,
                    kind=chunk.kind,
                    token_count=chunk.token_count,
                    prev_chunk_id=prev_id,
                    next_chunk_id=next_id,
                )
            )

        return linked
