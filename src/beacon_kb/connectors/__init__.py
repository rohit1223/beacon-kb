"""First-party source connector implementations for beacon-kb.

Connectors discover and load raw document bytes or text without parsing,
indexing, or owning credentials.  Injected clients are the caller's
responsibility.

Importing this package performs no side effects.
"""

from __future__ import annotations

from beacon_kb.connectors.filesystem import FilesystemConnector
from beacon_kb.connectors.memory import MemoryConnector

__all__ = [
    "FilesystemConnector",
    "MemoryConnector",
]
