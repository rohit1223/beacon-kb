"""Connector package for the Beacon ingestion pipeline (Task 02.1).

Public surface:
    - ``Connector`` - abstract base class.
    - ``SourceEntry`` - source metadata returned by ``enumerate()``.
    - ``FetchResult``, ``FetchSuccess``, ``TransientFailure``, ``ConfirmedDeletion``.
    - ``ConnectorKind`` - string constants and ``ALL`` frozenset.
    - ``get_connector_kinds()`` - returns the set of known kind strings.
"""

from __future__ import annotations

from beacon.ingest.connectors.base import (
    ConfirmedDeletion,
    Connector,
    ConnectorKind,
    FetchResult,
    FetchSuccess,
    SourceEntry,
    TransientFailure,
)

__all__ = [
    "ConfirmedDeletion",
    "Connector",
    "ConnectorKind",
    "FetchResult",
    "FetchSuccess",
    "SourceEntry",
    "TransientFailure",
    "get_connector_kinds",
]


def get_connector_kinds() -> frozenset[str]:
    """Return the frozenset of all known connector kind strings.

    Used by route validation to reject unknown kinds with a typed error.

    Returns:
        Frozenset of kind strings: ``{'folder', 'upload', 'web'}``.
    """
    return ConnectorKind.ALL
