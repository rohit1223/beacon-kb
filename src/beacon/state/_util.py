"""Private utilities for the beacon.state module.

Internal helpers for timestamp generation and database operations.
"""

from __future__ import annotations

import datetime


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()
