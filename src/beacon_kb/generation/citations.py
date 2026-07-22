"""Structural citation resolution and validation.

Citations are resolved from a map of citation_label -> Evidence rather than
from free text so unknown or malformed labels cannot escape validation.

Validation rules:
  1. Every label referenced in *answer_text* (pattern ``[Sn]``) must match a
     citation_label on one of the Evidence items in the response.
  2. Every Citation produced references a chunk_id and source_id that exist
     in the same response's Evidence tuple.
  3. Unresolvable labels raise CitationError immediately and are never silently
     dropped.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re

from beacon_kb.errors import CitationError
from beacon_kb.models import (
    ChunkId,
    Citation,
    Evidence,
    EvidenceId,
    SourceId,
)

# Pattern matching inline citation labels like [S1], [S12], [S999].
_CITATION_PATTERN: re.Pattern[str] = re.compile(r"\[S\d+\]")


def extract_cited_labels(answer_text: str) -> list[str]:
    """Return ordered list of unique citation labels found in *answer_text*.

    Matches the pattern ``[Sn]`` where n is one or more digits.  Order of
    first occurrence is preserved; duplicates are removed.

    Args:
        answer_text: Plain answer string from the generator.

    Returns:
        List of label strings such as ['S1', 'S3'] (without brackets).
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        raw = match.group()          # e.g. '[S1]'
        label = raw[1:-1]            # strip brackets -> 'S1'
        if label not in seen:
            seen.add(label)
            result.append(label)
    return result


def build_evidence_index(evidence: tuple[Evidence, ...]) -> dict[str, Evidence]:
    """Build a label -> Evidence lookup map from a response's evidence tuple.

    Args:
        evidence: Tuple of Evidence items from an AnswerResponse.

    Returns:
        Dict mapping citation_label (e.g. 'S1') to its Evidence record.
    """
    return {ev.citation_label: ev for ev in evidence}


def resolve_citations(
    answer_text: str,
    evidence: tuple[Evidence, ...],
) -> tuple[Citation, ...]:
    """Resolve and validate all citations referenced in *answer_text*.

    For each label found in *answer_text*:
      - Looks up the corresponding Evidence item by citation_label.
      - Raises CitationError if the label does not match any Evidence item.
      - Constructs a Citation record from the matched Evidence.

    Citations are structural: the model returned labels; code resolves them
    against the SAME evidence items present in the response.  Free-text
    citations that reference unknown labels are rejected immediately.

    Args:
        answer_text: Plain answer text produced by the generator.
        evidence:    Tuple of Evidence items in the same AnswerResponse.

    Returns:
        Tuple of resolved Citation records (order matches first appearance).

    Raises:
        CitationError: If any label in answer_text has no matching Evidence.
    """
    index = build_evidence_index(evidence)
    labels = extract_cited_labels(answer_text)

    citations: list[Citation] = []
    for label in labels:
        ev = index.get(label)
        if ev is None:
            available = sorted(index.keys())
            raise CitationError(
                f"Citation label '[{label}]' in answer_text has no matching "
                f"Evidence item.  Available labels: {available}.  "
                f"This means the generator produced a citation that cannot be "
                f"grounded in the retrieved evidence - rejecting the response."
            )
        chunk = ev.hit.chunk
        excerpt = chunk.text[:200].strip()
        citations.append(
            Citation(
                label=label,
                chunk_id=ChunkId(str(chunk.id)),
                source_id=SourceId(str(chunk.source_id)),
                canonical_uri=str(chunk.source_id),  # resolved from snippet if available
                excerpt=excerpt,
            )
        )

    # Resolve canonical_uri from the snippet when available (better provenance).
    resolved: list[Citation] = []
    for cit in citations:
        ev = index[cit.label]
        canonical_uri = cit.canonical_uri
        if ev.snippet is not None and ev.snippet.source_uri:
            canonical_uri = ev.snippet.source_uri
        resolved.append(
            Citation(
                label=cit.label,
                chunk_id=cit.chunk_id,
                source_id=cit.source_id,
                canonical_uri=canonical_uri,
                excerpt=cit.excerpt,
            )
        )

    return tuple(resolved)


def validate_no_unknown_evidence_ids(
    cited_ids: list[EvidenceId],
    available_ids: set[EvidenceId],
) -> None:
    """Raise CitationError if any cited EvidenceId is not in *available_ids*.

    Used as an additional structural gate when the generator returns evidence
    IDs directly (rather than labels).  Unknown IDs cannot escape validation.

    Args:
        cited_ids:     EvidenceId values extracted from the generator's output.
        available_ids: Set of valid EvidenceIds from the same AnswerResponse.

    Raises:
        CitationError: On the first unknown EvidenceId encountered.
    """
    for eid in cited_ids:
        if eid not in available_ids:
            raise CitationError(
                f"Generated response references unknown EvidenceId {str(eid)!r}.  "
                f"This ID does not appear in the retrieved evidence for this query.  "
                f"Rejecting to prevent ungrounded citations from escaping validation."
            )
