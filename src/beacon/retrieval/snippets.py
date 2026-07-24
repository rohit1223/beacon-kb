"""Match-centered, locator-preserving snippet construction (Task 03.2).

Design rules:
- Snippets are always centered on the matched span (first matching query term),
  never on a document or chunk prefix.
- When no query term appears in the chunk text, the snippet falls back to the
  center of the full chunk text (not a prefix).
- Provenance fields (source_uri, title, heading_path, locator, chunk_id) are
  taken directly from the caller-supplied arguments - never invented, never
  replaced with internal hash-form identifiers.
- char_start and char_end are consistent with the emitted snippet text:
  chunk_text[char_start:char_end].strip() == snippet.text.
- No LLM calls; purely text manipulation.

Token counting: this module does not count tokens; it works purely with
character offsets and character-based windows.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re

from beacon.models import Snippet

__all__ = ["Snippet", "build_snippet"]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _find_match_center(chunk_text: str, query_text: str) -> int:
    """Return the character offset of the best match for query_text in chunk_text.

    Strategy:
    1. Extract word tokens (length >= 3) from query_text.
    2. Find the first occurrence of any token in chunk_text (case-insensitive).
    3. Return the start offset of the earliest matching token.
    4. If no token matches, return the center of chunk_text (len // 2).

    Args:
        chunk_text: Full chunk text to search within.
        query_text: User query text.

    Returns:
        0-based character offset of the match center.
    """
    if not chunk_text:
        return 0

    # Extract tokens (word characters only, minimum 3 chars to avoid noise).
    tokens = [t for t in re.findall(r"\w+", query_text.lower()) if len(t) >= 3]

    if tokens:
        chunk_lower = chunk_text.lower()
        best_pos: int | None = None
        for token in tokens:
            m = re.search(re.escape(token), chunk_lower)
            if m is not None:
                if best_pos is None or m.start() < best_pos:
                    best_pos = m.start()
        if best_pos is not None:
            return best_pos

    # No match found: center of chunk text (never the prefix).
    return len(chunk_text) // 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snippet(
    chunk_text: str,
    query_text: str,
    *,
    source_uri: str,
    title: str,
    heading_path: list[str],
    locator: str,
    chunk_id: str,
    max_chars: int = 400,
) -> Snippet:
    """Build a match-centered snippet for one chunk.

    The snippet is centered on the first occurrence of any query term in the
    chunk text.  When no term matches, it is centered on the middle of the text.
    The window extends up to max_chars // 2 characters on each side of the
    center, clipped to the chunk boundaries.  Word boundaries are respected by
    snapping to the nearest whitespace.

    Provenance fields are copied verbatim from the caller's arguments - this
    function never derives or invents source identifiers.

    Args:
        chunk_text:   Full text of the chunk.
        query_text:   Original user query (used to locate the match center).
        source_uri:   Canonical URI of the source document.
        title:        Human-readable document title.
        heading_path: Ordered heading components from the chunk payload.
        locator:      Structural locator string (heading path or page).
        chunk_id:     Chunk identifier for traceability.
        max_chars:    Maximum character length of the snippet window.

    Returns:
        Snippet with text, provenance, locator, and span metadata.
    """
    if not chunk_text:
        return Snippet(
            text="",
            source_uri=source_uri,
            title=title,
            heading_path=heading_path,
            locator=locator,
            chunk_id=chunk_id,
            char_start=0,
            char_end=0,
        )

    half = max_chars // 2
    center = _find_match_center(chunk_text, query_text)

    # Calculate raw window around the match center.
    raw_start = max(0, center - half)
    raw_end = min(len(chunk_text), center + half)

    # Snap start to a word boundary (avoid cutting mid-word).
    if raw_start > 0:
        # Walk forward to the next whitespace after raw_start.
        while raw_start < len(chunk_text) and not chunk_text[raw_start].isspace():
            raw_start += 1
        # Skip the whitespace itself.
        while raw_start < len(chunk_text) and chunk_text[raw_start].isspace():
            raw_start += 1

    # Snap end to a word boundary.
    if raw_end < len(chunk_text):
        # Walk backward to find a clean word break.
        while raw_end > 0 and not chunk_text[raw_end - 1].isspace():
            raw_end -= 1
        # Strip trailing whitespace.
        while raw_end > 0 and chunk_text[raw_end - 1].isspace():
            raw_end -= 1

    # Ensure we have some content (degenerate case: no whitespace to snap to).
    if raw_start >= raw_end:
        center_pos = len(chunk_text) // 2
        raw_start = max(0, center_pos - half)
        raw_end = min(len(chunk_text), raw_start + max_chars)

    excerpt = chunk_text[raw_start:raw_end].strip()

    # Recompute span offsets to reflect the stripped excerpt's position.
    if excerpt:
        stripped_start = chunk_text.find(excerpt, raw_start)
        if stripped_start >= 0:
            raw_start = stripped_start
            raw_end = stripped_start + len(excerpt)

    return Snippet(
        text=excerpt,
        source_uri=source_uri,
        title=title,
        heading_path=heading_path,
        locator=locator,
        chunk_id=chunk_id,
        char_start=raw_start,
        char_end=raw_end,
    )
