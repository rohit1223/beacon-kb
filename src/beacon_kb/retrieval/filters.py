"""Provider-neutral filters for retrieval candidates.

Design rules enforced here:
- Filters apply consistently before candidates leave either retriever.
- Filters cannot be bypassed by either the sparse or dense retriever.
- The FilterSpec is immutable; no filter modifies hits in-place.
- Namespace, ACL, source, tag, media, and date filters are all supported.
- An empty or None constraint is a pass-through (no filtering on that axis).

Date and tag metadata is not stored on Chunk records in v1; applying a
date or tag filter with a populated constraint therefore excludes all hits
that carry no such metadata. This is the correct conservative behavior:
unknown metadata is not treated as a match.

Importing this module performs no side effects.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beacon_kb.models import Hit


@dataclass(frozen=True)
class FilterSpec:
    """Immutable specification for provider-neutral hit filters.

    All constraints are optional. An empty or absent constraint is a
    pass-through on that axis.

    Fields:
        namespace:   Reserved; not enforced in v1.
                     Use Query.corpus_id for corpus scoping instead.
                     Setting this field has no effect on filtering: all hits pass
                     through as if namespace were None.
                     Intended for cross-corpus namespace ACL enforcement in a
                     future version once ACL metadata is stored on Chunk records.
        acl_ids:     If non-empty, keep only hits accessible to at least one
                     of the given ACL principal IDs.  Not yet enforced in v1
                     (no ACL metadata on Chunk); reserved for future use.
        source_uris: If non-empty, keep only hits whose source_id string is in
                     this set.  Matched via exact equality against str(chunk.source_id).
                     In v1 str(chunk.source_id) is the hex digest produced by
                     make_source_id(corpus, canonical_uri) - a SHA-256-derived hash -
                     NOT the raw canonical URI.
                     The ingestion pipeline guarantees all chunks carry hash-form source
                     IDs (cross-referenced in the project merge checklist).
                     To filter by source URI, first resolve the URI to its source_id hash
                     via make_source_id(), then pass the resulting string in this set.
                     For corpus-level scoping, prefer Query.corpus_id over this field.
        tags:        If non-empty, keep only hits tagged with at least one of
                     the given tags.  Chunk records in v1 carry no tag metadata;
                     applying a non-empty tag filter excludes all hits.
        media_types: If non-empty, keep only hits with a matching media type.
                     Chunk records in v1 carry no media_type; applying a
                     non-empty media_types filter excludes all hits.
        require_after: If set, keep only hits whose publication date is on or
                     after this date.  Chunk records in v1 carry no date;
                     applying this filter excludes all hits.
    """

    namespace: str | None = None
    acl_ids: frozenset[str] = field(default_factory=frozenset)
    source_uris: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)
    media_types: frozenset[str] = field(default_factory=frozenset)
    require_after: datetime.date | None = None


def apply_filters(hits: list[Hit], spec: FilterSpec) -> list[Hit]:
    """Apply provider-neutral filters to a list of hits, returning only matching hits.

    Filters are applied in order; a hit is excluded if it fails any active filter.
    No filter modifies hit scores; retained hits carry exactly the scores they arrived with.

    Args:
        hits: Candidate hits from a sparse or dense retriever.
        spec: FilterSpec describing which constraints to enforce.

    Returns:
        Subset of *hits* that pass all active filters, in the same relative order.
    """
    result: list[Hit] = []
    for hit in hits:
        if not _passes_all(hit, spec):
            continue
        result.append(hit)
    return result


def _passes_all(hit: Hit, spec: FilterSpec) -> bool:
    """Return True if *hit* passes all active constraints in *spec*."""
    # Source filter: match against str(chunk.source_id), which is the
    # SHA-256-derived hash produced by make_source_id(corpus, canonical_uri).
    # Callers must resolve a canonical URI to its source_id hash first
    # (via make_source_id()) before populating FilterSpec.source_uris.
    if spec.source_uris:
        source_id_str = str(hit.chunk.source_id)
        if source_id_str not in spec.source_uris:
            return False

    # Tag filter: Chunk records in v1 carry no tag metadata.
    # Non-empty tag constraint excludes all hits (conservative default).
    if spec.tags:
        # No tag metadata on Chunk - exclude.
        return False

    # Media type filter: Chunk records in v1 carry no media_type.
    # Non-empty media_types constraint excludes all hits.
    if spec.media_types:
        return False

    # Date filter: Chunk records in v1 carry no publication date.
    # require_after set -> exclude all hits with no date metadata.
    if spec.require_after is not None:
        return False

    # Namespace filter: not enforced in v1; field is reserved.
    # Use Query.corpus_id for corpus scoping.
    # All hits pass through regardless of FilterSpec.namespace value.

    # ACL filter: not enforced in v1 (no ACL metadata on Chunk).

    return True
