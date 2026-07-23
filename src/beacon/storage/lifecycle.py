"""Staged promotion protocol for Qdrant collections.

The protocol guarantees that a live alias always points at a fully built
collection.  New content is written to a SHADOW collection (invisible through
the alias) and only becomes live when ``promote()`` atomically flips the
alias.

Terminology
-----------
- **logical name**: the alias callers use to read (e.g. ``"corpus"``).
- **physical collection**: the actual Qdrant collection (e.g.
  ``"corpus__rev_<uuid>"``).
- **shadow collection**: the physical collection being built during a stage.

Lifecycle
---------
1. ``begin_stage(store, logical, dense_dim)`` - creates a new physical shadow
   collection named ``<logical>__rev_<uuid>`` and returns a ``StageHandle``.
2. Caller writes points into ``stage.shadow_collection`` via
   ``store.upsert(stage.shadow_collection, ...)``.
3. ``promote(store, stage)`` atomically flips the alias to the shadow
   collection, then schedules the previous physical collection for deletion.
   Deletion failures are logged and left as orphans - never un-promoted.
4. ``abort(store, stage)`` drops the shadow collection and leaves the alias
   (and its target collection) untouched.

Failure model
-------------
If the process crashes after ``begin_stage`` but before ``promote``, the
alias still points at the old collection.  The orphaned shadow collection can
be detected and cleaned up by a later sweep (Epic 08 adds the sweep).

If deletion of the old collection after promote fails, the orphan is logged
at WARNING level.  The alias already points at the new collection so the
failure has no user-visible impact.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from beacon.errors import BackendError
from beacon.storage.qdrant import QdrantStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageHandle:
    """Immutable token describing an in-progress staging operation.

    ``logical_name`` is the alias callers read from.
    ``shadow_collection`` is the physical collection being built.
    ``prior_collection`` is the physical collection the alias pointed to
    before this stage began (``None`` if the alias did not exist yet).
    """

    logical_name: str
    shadow_collection: str
    prior_collection: str | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def begin_stage(
    store: QdrantStore,
    logical_name: str,
    dense_dim: int,
) -> StageHandle:
    """Create a shadow collection and return a handle for the stage.

    The shadow collection is named ``<logical_name>__rev_<uuid4>`` so it is
    unique across concurrent stages and unambiguously tied to the logical name
    for diagnostic purposes.

    Args:
        store: An open ``QdrantStore`` instance.
        logical_name: The alias callers use to read (e.g. ``"corpus"``).
        dense_dim: Dense vector dimension; must match the embedding model.

    Returns:
        A ``StageHandle`` with the shadow collection name and prior alias
        target.

    Raises:
        BackendError: If the shadow collection cannot be created.
    """
    rev = uuid.uuid4().hex[:12]
    shadow = f"{logical_name}__rev_{rev}"
    prior = store.resolve_alias(logical_name)

    store.create_collection(shadow, dense_dim=dense_dim)
    logger.info(
        "Stage begun: logical=%r shadow=%r prior=%r",
        logical_name,
        shadow,
        prior,
    )
    return StageHandle(
        logical_name=logical_name,
        shadow_collection=shadow,
        prior_collection=prior,
    )


def promote(store: QdrantStore, stage: StageHandle) -> None:
    """Atomically flip the alias to the shadow collection.

    The alias is retargeted in a single ``update_collection_aliases`` call so
    there is no observable moment where the logical name resolves to nothing.

    After the alias flip, the prior physical collection (if any) is scheduled
    for deletion.  A deletion failure is logged at WARNING level and the
    orphan is left for a later sweep; the alias is never rolled back.

    Args:
        store: An open ``QdrantStore`` instance.
        stage: The ``StageHandle`` returned by ``begin_stage``.

    Raises:
        BackendError: If the alias flip fails.
    """
    # Atomic alias flip - this is the critical section.
    store.set_alias(stage.logical_name, stage.shadow_collection)
    logger.info(
        "Promoted: logical=%r -> shadow=%r (prior=%r)",
        stage.logical_name,
        stage.shadow_collection,
        stage.prior_collection,
    )

    # Best-effort cleanup of the prior physical collection.
    if stage.prior_collection is not None:
        _cleanup_orphan(store, stage.prior_collection)


def abort(store: QdrantStore, stage: StageHandle) -> None:
    """Drop the shadow collection; leave the alias and its target unchanged.

    Safe to call even if the shadow collection no longer exists (e.g. if a
    partial cleanup already ran).

    Args:
        store: An open ``QdrantStore`` instance.
        stage: The ``StageHandle`` returned by ``begin_stage``.

    Raises:
        BackendError: If the shadow collection delete fails unexpectedly.
    """
    store.delete_collection(stage.shadow_collection)
    logger.info(
        "Stage aborted: logical=%r shadow=%r dropped",
        stage.logical_name,
        stage.shadow_collection,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cleanup_orphan(store: QdrantStore, collection_name: str) -> None:
    """Delete an orphaned physical collection; log and swallow failures."""
    try:
        store.delete_collection(collection_name)
        logger.info("Cleaned up prior collection: %r", collection_name)
    except BackendError as exc:
        logger.warning(
            "Failed to delete prior collection %r (orphaned): %s",
            collection_name,
            exc,
        )
    except Exception as exc:  # pragma: no cover - unexpected
        logger.warning(
            "Unexpected error deleting prior collection %r (orphaned): %s",
            collection_name,
            exc,
        )
