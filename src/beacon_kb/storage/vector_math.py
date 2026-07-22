"""Vector validation and local NumPy similarity search with declared direction.

Design rules enforced here:
- Vectors carry a declared dimension and similarity direction.
- Missing distance metadata NEVER defaults to zero - the caller must supply
  an explicit similarity direction.
- Normalized vector validation is checked before storage.
- All similarity computations use NumPy for correctness and speed.

Importing this module performs no side effects.
"""

from __future__ import annotations

import struct
from typing import cast

import numpy as np

from beacon_kb.errors import BackendError

# Tolerance for unit-norm check (L2 norm must be within this of 1.0).
_NORM_TOLERANCE: float = 1e-4

# Supported similarity directions - never inferred, always declared.
SUPPORTED_SIMILARITY: frozenset[str] = frozenset({"cosine", "dot", "euclidean"})


def validate_dimension(vector: list[float], expected_dim: int) -> None:
    """Raise BackendError if *vector* has wrong length.

    Args:
        vector:       Input vector as a list of floats.
        expected_dim: Declared dimension for this store/model.

    Raises:
        BackendError: If ``len(vector) != expected_dim``.
    """
    if len(vector) != expected_dim:
        raise BackendError(
            f"Vector dimension mismatch: expected {expected_dim} but got {len(vector)}. "
            f"Ensure the embedder dimension matches the store's declared vector_dim. "
            f"Never infer dimension from an untyped metadata key."
        )


def validate_similarity(similarity: str) -> None:
    """Raise BackendError if *similarity* is not a known direction.

    Args:
        similarity: Declared similarity direction string.

    Raises:
        BackendError: If *similarity* is not in SUPPORTED_SIMILARITY.
    """
    if similarity not in SUPPORTED_SIMILARITY:
        raise BackendError(
            f"Unknown similarity direction {similarity!r}. "
            f"Must be one of {sorted(SUPPORTED_SIMILARITY)}. "
            f"Missing distance metadata must NEVER default to zero."
        )


def validate_unit_norm(vector: list[float], *, tolerance: float = _NORM_TOLERANCE) -> None:
    """Raise BackendError if *vector* is not unit-normalized.

    Args:
        vector:    Input vector as a list of floats.
        tolerance: Acceptable deviation from L2 norm 1.0.

    Raises:
        BackendError: If the L2 norm deviates from 1.0 by more than *tolerance*.
    """
    arr = np.array(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > tolerance:
        raise BackendError(
            f"Vector is not unit-normalized: L2 norm = {norm:.6f}, expected ~1.0 "
            f"(tolerance {tolerance}). Normalize the vector before storage."
        )


def encode_vector(vector: list[float]) -> bytes:
    """Encode a float list to a compact float32 byte string.

    Uses ``struct.pack`` over numpy to avoid a numpy dependency at decode time
    while keeping the encoding deterministic.

    Args:
        vector: List of floats to encode.

    Returns:
        Bytes of length ``len(vector) * 4`` (float32 little-endian).
    """
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes, dimension: int) -> list[float]:
    """Decode a float32 byte string back to a float list.

    Args:
        blob:      Bytes produced by ``encode_vector()``.
        dimension: Expected number of floats (used to validate length).

    Returns:
        List of floats of length *dimension*.

    Raises:
        BackendError: If the blob length does not match the declared dimension.
    """
    expected_bytes = dimension * 4
    if len(blob) != expected_bytes:
        raise BackendError(
            f"Embedding blob length {len(blob)} does not match declared dimension "
            f"{dimension} (expected {expected_bytes} bytes). Blob may be corrupt."
        )
    values = struct.unpack(f"<{dimension}f", blob)
    return list(values)


def cosine_similarity(query: list[float], candidates: list[list[float]]) -> list[float]:
    """Return cosine similarity between *query* and each candidate vector.

    All vectors are assumed to be unit-normalized (cosine = dot product).

    Args:
        query:      Query vector (unit-normalized).
        candidates: List of candidate vectors (unit-normalized).

    Returns:
        List of float similarity scores in the same order as *candidates*.
        Higher score means more similar. Range [-1, 1] for unit vectors.
    """
    if not candidates:
        return []
    q = np.array(query, dtype=np.float32)
    mat = np.array(candidates, dtype=np.float32)
    # For unit-normalized vectors, cosine similarity = dot product.
    scores: np.ndarray = mat @ q
    return cast(list[float], scores.tolist())


def dot_similarity(query: list[float], candidates: list[list[float]]) -> list[float]:
    """Return raw dot product between *query* and each candidate vector.

    Args:
        query:      Query vector.
        candidates: List of candidate vectors.

    Returns:
        List of float dot products. Higher is more similar.
    """
    if not candidates:
        return []
    q = np.array(query, dtype=np.float32)
    mat = np.array(candidates, dtype=np.float32)
    scores: np.ndarray = mat @ q
    return cast(list[float], scores.tolist())


def negative_euclidean_similarity(
    query: list[float], candidates: list[list[float]]
) -> list[float]:
    """Return negative Euclidean distance (higher = more similar) for each candidate.

    Args:
        query:      Query vector.
        candidates: List of candidate vectors.

    Returns:
        List of float scores. Higher (less negative) is more similar.
    """
    if not candidates:
        return []
    q = np.array(query, dtype=np.float32)
    mat = np.array(candidates, dtype=np.float32)
    diffs = mat - q
    distances: np.ndarray = np.linalg.norm(diffs, axis=1)
    return cast(list[float], (-distances).tolist())


def compute_similarity(
    query: list[float],
    candidates: list[list[float]],
    *,
    similarity: str,
) -> list[float]:
    """Dispatch to the correct similarity function based on the declared direction.

    Args:
        query:      Query vector.
        candidates: List of candidate vectors.
        similarity: Declared similarity direction ('cosine', 'dot', 'euclidean').

    Returns:
        List of float scores for each candidate (higher is more similar).

    Raises:
        BackendError: If *similarity* is not a supported direction.
    """
    validate_similarity(similarity)
    if similarity == "cosine":
        return cosine_similarity(query, candidates)
    if similarity == "dot":
        return dot_similarity(query, candidates)
    # euclidean
    return negative_euclidean_similarity(query, candidates)
