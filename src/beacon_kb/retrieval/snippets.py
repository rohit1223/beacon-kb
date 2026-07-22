"""Match-centered, locator-preserving snippet construction.

Design rules enforced here:
- Snippets are always centered on the matched span (the query terms within the
  chunk text), never a document or chunk prefix.
- When no query term appears in the chunk text, the snippet falls back to the
  center of the full chunk text (still not a prefix).
- The snippet always preserves: source_id (as string), canonical_uri (from the
  Chunk's source_id derivation - callers supply the resolved URI), title, the
  structural locator (parent_locator), and the character span within the original
  chunk text.
- Snippets are plain strings (never pre-formatted Markdown).
- No LLM calls; purely text-manipulation.

Snippet is defined in beacon_kb.models (not here) to avoid an import cycle:
Evidence references Snippet, and context.py/pipeline.py import both.
This module re-exports Snippet from models for backward compatibility.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re

# Snippet is canonical in beacon_kb.models to avoid import cycles.
# Re-exported here for callers that import it from this module.
from beacon_kb.models import Snippet

__all__ = ["Snippet", "build_snippet"]


def _find_match_center(chunk_text: str, query_text: str) -> int:
    """Return the character offset of the best match for *query_text* in *chunk_text*.

    Strategy:
    1. Look for any query term (word-token) in chunk_text, case-insensitive.
    2. Return the start offset of the first matching term.
    3. If no term matches, return the center of the full chunk text (not prefix).

    Args:
        chunk_text: The full chunk text to search within.
        query_text: The original user query text.

    Returns:
        Character offset (0-based) of the match center in chunk_text.
    """
    if not chunk_text:
        return 0

    # Extract query tokens (word characters only, length >= 3 to avoid noise).
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

    # No match - center of chunk text.
    return len(chunk_text) // 2


def build_snippet(
    chunk_text: str,
    query_text: str,
    *,
    source_id: str,
    source_uri: str,
    title: str,
    locator: str,
    chunk_id: str,
    max_chars: int = 400,
) -> Snippet:
    """Build a match-centered snippet for one chunk.

    The snippet is centered on the first occurrence of any query term in the
    chunk text.  When no term matches, it is centered on the middle of the
    text.  The window extends up to *max_chars*/2 characters on each side of
    the center, clipped to the chunk boundaries.  Word boundaries are respected
    by snapping to the nearest whitespace when snipping mid-word.

    Args:
        chunk_text:  Full text of the chunk.
        query_text:  Original user query (used to locate the match center).
        source_id:   str(chunk.source_id) for provenance.
        source_uri:  Canonical URI of the source document.
        title:       Human-readable title (may be empty).
        locator:     chunk.parent_locator (heading path or page).
        chunk_id:    str(chunk.id) for traceability.
        max_chars:   Maximum character length of the snippet window.

    Returns:
        Snippet with text, source provenance, locator, and span metadata.
    """
    if not chunk_text:
        return Snippet(
            text="",
            source_id=source_id,
            source_uri=source_uri,
            title=title,
            locator=locator,
            char_start=0,
            char_end=0,
            chunk_id=chunk_id,
        )

    half = max_chars // 2
    center = _find_match_center(chunk_text, query_text)

    # Calculate raw window.
    raw_start = max(0, center - half)
    raw_end = min(len(chunk_text), center + half)

    # Snap start to a word boundary (avoid cutting mid-word).
    if raw_start > 0:
        # Walk forward to the next whitespace.
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

    # Ensure we always have at least some content.
    if raw_start >= raw_end:
        raw_start = 0
        raw_end = min(len(chunk_text), max_chars)

    excerpt = chunk_text[raw_start:raw_end].strip()

    return Snippet(
        text=excerpt,
        source_id=source_id,
        source_uri=source_uri,
        title=title,
        locator=locator,
        char_start=raw_start,
        char_end=raw_end,
        chunk_id=chunk_id,
    )
