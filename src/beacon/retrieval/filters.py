"""Typed retrieval filters and the single boundary compiler to Qdrant filters.

``FilterSpec`` is the only way callers express retrieval constraints, and
``compile_filter`` is the only path from a ``FilterSpec`` to a Qdrant payload
filter.  The hybrid pipeline compiles the spec *before* any query-executor
implementation is invoked and attaches the compiled filter to every branch of
the Qdrant request, so no retriever implementation can bypass it (the v1
``beacon_kb`` boundary-enforcement guarantee).

Child-only ranking is the default: ``FilterSpec.kinds`` defaults to
``("child",)`` so parent chunks never compete with their own children in the
ranked list.  Callers that need parent chunks opt in explicitly.

All filtered fields (``source_uri``, ``tags``, ``kind``, and the date fields)
are declared in ``beacon.storage.payload.PAYLOAD_INDEX_FIELDS`` and indexed at
collection-creation time, so filtering happens inside Qdrant on indexed
payload fields rather than via post-hoc scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from qdrant_client.http import models as qmodels

# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DateRange:
    """Inclusive datetime bounds for a payload date field.

    Attributes:
        gte: Earliest matching timestamp (inclusive), or ``None`` for no floor.
        lte: Latest matching timestamp (inclusive), or ``None`` for no ceiling.
    """

    gte: datetime | None = None
    lte: datetime | None = None

    def is_empty(self) -> bool:
        """Return ``True`` when neither bound is set."""
        return self.gte is None and self.lte is None


@dataclass(frozen=True)
class FilterSpec:
    """Typed retrieval constraints for one search.

    Attributes:
        collection:  Logical collection name to search (resolved through the
                     Qdrant alias to the live physical collection).
        source_uris: Restrict hits to these canonical source URIs (OR within
                     the tuple); empty means no source restriction.
        tags:        Restrict hits to chunks carrying any of these tags;
                     empty means no tag restriction.
        created:     Bounds on the source document ``created_at`` field.
        modified:    Bounds on the source document ``modified_at`` field.
        ingested:    Bounds on the pipeline ``ingested_at`` field.
        kinds:       Chunk kinds to rank; defaults to child-only ranking.
    """

    collection: str
    source_uris: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created: DateRange | None = None
    modified: DateRange | None = None
    ingested: DateRange | None = None
    kinds: tuple[str, ...] = field(default=("child",))


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def _date_condition(key: str, date_range: DateRange) -> qmodels.FieldCondition:
    """Compile one date range into a Qdrant datetime-range field condition."""
    return qmodels.FieldCondition(
        key=key,
        range=qmodels.DatetimeRange(gte=date_range.gte, lte=date_range.lte),
    )


def compile_filter(spec: FilterSpec) -> qmodels.Filter | None:
    """Compile a ``FilterSpec`` into a Qdrant payload filter.

    This is the single boundary compiler: every retrieval path must obtain
    its Qdrant filter from this function so user constraints are enforced
    inside the Qdrant query on indexed payload fields.

    Args:
        spec: The typed filter specification.

    Returns:
        A ``qmodels.Filter`` with one ``must`` condition per constraint, or
        ``None`` when the spec carries no constraints at all (only possible
        when ``kinds`` is explicitly emptied).
    """
    must: list[qmodels.Condition] = []

    if spec.kinds:
        must.append(
            qmodels.FieldCondition(
                key="kind",
                match=qmodels.MatchAny(any=list(spec.kinds)),
            )
        )
    if spec.source_uris:
        must.append(
            qmodels.FieldCondition(
                key="source_uri",
                match=qmodels.MatchAny(any=list(spec.source_uris)),
            )
        )
    if spec.tags:
        must.append(
            qmodels.FieldCondition(
                key="tags",
                match=qmodels.MatchAny(any=list(spec.tags)),
            )
        )
    if spec.created is not None and not spec.created.is_empty():
        must.append(_date_condition("created_at", spec.created))
    if spec.modified is not None and not spec.modified.is_empty():
        must.append(_date_condition("modified_at", spec.modified))
    if spec.ingested is not None and not spec.ingested.is_empty():
        must.append(_date_condition("ingested_at", spec.ingested))

    if not must:
        return None
    return qmodels.Filter(must=must)
