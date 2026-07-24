"""Structural citation extraction, resolution, and validation (Task 03.3).

Ported from beacon_kb.generation.citations onto the beacon EvidenceBundle.

Citations are resolved against the canonical evidence bundle held by the server
- the labels the retrieval pipeline assigned - rather than from any content the
model echoes back.  A hostile model cannot smuggle fabricated evidence into the
response because validation never trusts model-supplied evidence; it only
resolves the model's ``[S#]`` labels against the server's bundle.

Validation rules:
  1. Every label referenced in the answer text (pattern ``[S<n>]``) must match a
     label on one of the Evidence items in the canonical bundle.
  2. An unresolvable label raises ``CitationError`` immediately and is never
     silently dropped.
  3. ``validate_no_unknown_evidence_ids`` is the structural subset gate: any
     cited chunk id not present in the bundle is rejected.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from beacon.errors import CitationError
from beacon.models import Citation, EvidenceBundle

__all__ = [
    "extract_cited_labels",
    "resolve_citations",
    "validate_no_unknown_evidence_ids",
]

# Pattern matching inline citation labels like [S1], [S12], [S999].
_CITATION_PATTERN: re.Pattern[str] = re.compile(r"\[S\d+\]")

_EXCERPT_MAX_CHARS = 200


def extract_cited_labels(answer_text: str) -> list[str]:
    """Return the ordered, de-duplicated citation labels found in *answer_text*.

    Matches the pattern ``[S<n>]`` where n is one or more digits.  Extraction
    tolerates adjacent punctuation and repeated citations; first-occurrence
    order is preserved and duplicates are removed.

    Args:
        answer_text: Plain answer string from the provider.

    Returns:
        List of label strings such as ``['S1', 'S3']`` (without brackets).
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        label = match.group()[1:-1]  # strip the brackets: '[S1]' -> 'S1'
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def resolve_citations(
    answer_text: str,
    bundle: EvidenceBundle,
) -> tuple[Citation, ...]:
    """Resolve and validate every citation label in *answer_text* against *bundle*.

    For each label found in the answer:
      - look up the matching Evidence item by its bundle label;
      - raise ``CitationError`` if the label is not present in the canonical
        bundle (the model cited evidence the server did not supply);
      - build a ``Citation`` record from the matched, server-held evidence.

    Citations are structural: the model returned labels; this code resolves them
    against the SAME canonical evidence the server holds.  Labels that reference
    evidence not in the bundle are rejected immediately.

    Args:
        answer_text: Plain answer text produced by the provider.
        bundle:      Canonical evidence bundle held by the server.

    Returns:
        Tuple of resolved Citation records (order matches first appearance).

    Raises:
        CitationError: If any label in the answer has no matching bundle item.
    """
    index = {ev.label: ev for ev in bundle.evidence}
    labels = extract_cited_labels(answer_text)

    citations: list[Citation] = []
    for label in labels:
        ev = index.get(label)
        if ev is None:
            available = sorted(index.keys())
            raise CitationError(
                f"Citation label '[{label}]' in the answer has no matching evidence "
                f"item in the canonical bundle.  Available labels: {available}.  "
                f"The model produced a citation that cannot be grounded in the "
                f"retrieved evidence; rejecting the response."
            )
        source_uri = ev.snippet.source_uri if ev.snippet is not None else ""
        excerpt = (
            ev.snippet.text[:_EXCERPT_MAX_CHARS].strip()
            if ev.snippet is not None and ev.snippet.text
            else ""
        )
        citations.append(
            Citation(
                label=label,
                chunk_id=ev.chunk_id,
                source_uri=source_uri,
                excerpt=excerpt,
            )
        )

    return tuple(citations)


def validate_no_unknown_evidence_ids(
    cited_ids: Iterable[str],
    available_ids: set[str],
) -> None:
    """Raise ``CitationError`` if any cited chunk id is not in *available_ids*.

    The structural subset gate: any chunk id the model references that was not
    part of the canonical retrieved evidence is rejected, so fabricated evidence
    cannot escape validation.

    Args:
        cited_ids:     Chunk ids the model referenced.
        available_ids: Set of hex chunk ids present in the canonical bundle.

    Raises:
        CitationError: On the first unknown chunk id encountered.
    """
    for cid in cited_ids:
        if cid not in available_ids:
            raise CitationError(
                f"The response references unknown evidence chunk id {cid!r}, "
                f"which is not in the canonical retrieved evidence for this query.  "
                f"Rejecting to prevent ungrounded citations from escaping validation."
            )
