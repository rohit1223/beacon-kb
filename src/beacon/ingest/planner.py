"""Change planner for incremental sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    FetchSuccess,
    TransientFailure,
)
from beacon.state.db import StateDB
from beacon.state.repo import RevisionRepo, SourceRepo


@dataclass
class SourceClassification:
    """Classification of a single source URI during change planning.

    Attributes:
        uri:            Canonical URI for this source.
        action:         What the sync engine should do with this source.
        reason:         Human-readable explanation for the action.
        content_hash:   SHA-256 of the content; empty for deleted/transient.
        media_type:     MIME type of the source content.
        connector_kind: Kind string reported by the connector.
    """

    uri: str
    action: Literal["new", "changed", "deleted", "unchanged", "incompatible", "transient_failure"]
    reason: str
    content_hash: str
    media_type: str
    connector_kind: str


@dataclass
class ChangePlan:
    """Result of planning a sync run.

    Attributes:
        sources_to_process: Sources classified as new, changed, or incompatible.
        sources_unchanged:  Sources with unchanged content and fingerprint.
        sources_to_delete:  Sources confirmed as deleted by the connector.
        transient_failures: Sources that returned TransientFailure.
        fingerprint_drifted: True when the pipeline fingerprint changed vs the live revision.
    """

    sources_to_process: list[SourceClassification] = field(default_factory=list)
    sources_unchanged: list[SourceClassification] = field(default_factory=list)
    sources_to_delete: list[SourceClassification] = field(default_factory=list)
    transient_failures: list[SourceClassification] = field(default_factory=list)
    fingerprint_drifted: bool = False


def plan_sync(
    *,
    connector: Connector,
    collection_name: str,
    current_fingerprint: str,
    source_repo: SourceRepo,
    db: StateDB,
) -> ChangePlan:
    """Plan what needs to happen in a sync.

    Compares the connector's enumerated sources against the DB state to classify
    each source as new, changed, incompatible (fingerprint drift), unchanged,
    deleted, or transient failure.

    Sources in the DB that are no longer returned by the connector are classified
    as deleted.

    Args:
        connector:           Connector to enumerate and fetch from.
        collection_name:     Logical collection name.
        current_fingerprint: The fingerprint computed for the current pipeline config.
        source_repo:         SourceRepo for DB lookups.
        db:                  Open StateDB for RevisionRepo lookups.

    Returns:
        A ChangePlan with all sources classified.
    """
    plan = ChangePlan()

    # Check whether the fingerprint drifted vs the last live revision.
    live_rev = RevisionRepo(db).get_live(collection_name=collection_name)
    last_fingerprint = live_rev["fingerprint"] if live_rev is not None else ""
    plan.fingerprint_drifted = bool(last_fingerprint) and (last_fingerprint != current_fingerprint)

    # Enumerate all sources from connector.
    enumerated = connector.enumerate()
    enumerated_uris: set[str] = set()

    for entry in enumerated:
        enumerated_uris.add(entry.uri)

        # Fetch to get content_hash.
        result = connector.fetch(entry.uri)

        if isinstance(result, TransientFailure):
            plan.transient_failures.append(
                SourceClassification(
                    uri=entry.uri,
                    action="transient_failure",
                    reason=result.reason,
                    content_hash="",
                    media_type=entry.media_type,
                    connector_kind=entry.connector_kind,
                )
            )
            continue

        if isinstance(result, ConfirmedDeletion):
            plan.sources_to_delete.append(
                SourceClassification(
                    uri=entry.uri,
                    action="deleted",
                    reason="Connector returned ConfirmedDeletion",
                    content_hash="",
                    media_type=entry.media_type,
                    connector_kind=entry.connector_kind,
                )
            )
            continue

        # FetchSuccess path.
        assert isinstance(result, FetchSuccess)
        existing = source_repo.get(collection_name=collection_name, canonical_uri=entry.uri)

        if existing is None:
            action: Literal["new", "changed", "incompatible", "unchanged"] = "new"
            reason = "New source not previously seen"
        elif existing["status"] != "active":
            action = "new"
            reason = "Source previously retired; re-indexing"
        elif existing["content_hash"] != result.content_hash:
            action = "changed"
            reason = (
                f"Content hash changed: {existing['content_hash']!r}"
                f" -> {result.content_hash!r}"
            )
        elif plan.fingerprint_drifted:
            action = "incompatible"
            reason = (
                f"Pipeline fingerprint drifted: {last_fingerprint!r}"
                f" -> {current_fingerprint!r}"
            )
        else:
            action = "unchanged"
            reason = "Content hash and fingerprint unchanged"

        classification = SourceClassification(
            uri=entry.uri,
            action=action,
            reason=reason,
            content_hash=result.content_hash,
            media_type=result.media_type,
            connector_kind=entry.connector_kind,
        )

        if action in ("new", "changed", "incompatible"):
            plan.sources_to_process.append(classification)
        else:
            plan.sources_unchanged.append(classification)

    # Sources in DB but NOT enumerated by connector: confirm before retiring.
    # Absence from enumeration alone is NOT evidence of deletion (a per-source
    # enumeration hiccup must never retire an indexed source); the source is
    # retired only when a follow-up fetch returns ConfirmedDeletion or the
    # connector deliberately excludes a still-fetchable source.  A transient
    # fetch outcome keeps the source and records a warning classification.
    active_sources = source_repo.list_active(collection_name=collection_name)
    for row in active_sources:
        uri = row["canonical_uri"]
        if uri in enumerated_uris:
            continue

        confirm = connector.fetch(uri)
        if isinstance(confirm, TransientFailure):
            plan.transient_failures.append(
                SourceClassification(
                    uri=uri,
                    action="transient_failure",
                    reason=(
                        f"Not enumerated and fetch failed transiently: {confirm.reason}"
                    ),
                    content_hash="",
                    media_type=row["media_type"] or "",
                    connector_kind=row["connector_kind"],
                )
            )
            continue

        reason = (
            "Connector confirmed deletion after the source disappeared from "
            "enumeration"
            if isinstance(confirm, ConfirmedDeletion)
            else "Source deliberately excluded from connector enumeration "
            "(still fetchable)"
        )
        plan.sources_to_delete.append(
            SourceClassification(
                uri=uri,
                action="deleted",
                reason=reason,
                content_hash="",
                media_type=row["media_type"] or "",
                connector_kind=row["connector_kind"],
            )
        )

    return plan
