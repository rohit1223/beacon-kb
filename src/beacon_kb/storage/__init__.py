"""Storage backend implementations for beacon-kb.

This package provides:
- ``SQLiteStore``: The default transactional SQLite-backed knowledge store.
- ``vector_math``: Vector validation and NumPy similarity search utilities.

The ``SQLiteStore`` is registered as the default ``sqlite`` store in the
``beacon_kb.stores`` entry-point group when this package is imported.

Importing this module has the side effect of registering the SQLiteStore
as the built-in default store.
"""

from __future__ import annotations

from beacon_kb.storage.sqlite import SQLiteStore
from beacon_kb.storage.vector_math import (
    compute_similarity,
    decode_vector,
    encode_vector,
    validate_dimension,
    validate_similarity,
    validate_unit_norm,
)

__all__ = [
    "SQLiteStore",
    "compute_similarity",
    "decode_vector",
    "encode_vector",
    "validate_dimension",
    "validate_similarity",
    "validate_unit_norm",
]
