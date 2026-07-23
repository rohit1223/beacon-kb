"""Pipeline fingerprint for revision identity."""
from __future__ import annotations

import hashlib

SCHEMA_VERSION = 1


def compute_fingerprint(
    *,
    parser_version: str,
    chunker_config_str: str,
    model_name: str,
    dimension: int,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return a stable hex SHA-256 fingerprint for the pipeline configuration.

    Args:
        parser_version:    PARSER_VERSION string from parsing.py.
        chunker_config_str: Canonical chunker config string from chunking.py.
        model_name:        Embedding model name.
        dimension:         Dense embedding dimension.
        schema_version:    Internal schema version; bump to invalidate all fingerprints.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    payload = "\x00".join([
        parser_version,
        chunker_config_str,
        model_name,
        str(dimension),
        str(schema_version),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()
