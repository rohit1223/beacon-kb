"""Embedding provider with auto-detect for cloud, local, and sparse-only modes.

Auto-detect ladder (config-driven, no probe calls, decided at construction):

1. CLOUD  - a recognised provider API key is present in the environment; dense
   embeddings are produced through LiteLLM.
2. LOCAL  - no cloud key, but local sentence-transformers embeddings are
   available (package installed and not blocked by ``HF_HUB_OFFLINE=1``).
3. SPARSE_ONLY - the floor: no dense vectors are produced and search runs
   sparse-only.

Sparse representations are computed for every chunk in ALL modes using a
BM25-style term-frequency vector over a stable hashed vocabulary.  Token
hashing uses SHA-256 (never Python's salted ``hash()``) so indices are
identical across processes - a sparse vector computed at query time in a new
process matches the vectors persisted at ingestion time.

The active mode is exposed via ``EmbedderProvider.mode`` for diagnostics, and
``fingerprint_model_id`` (mode + model name) plus ``dimension`` feed the
pipeline fingerprint so a mode change forces a rebuild instead of serving
mixed-generation vectors.

The chosen mode is logged once per process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from beacon.errors import IngestionError

logger = logging.getLogger(__name__)

VOCAB_SIZE = 30000

#: Environment variables that signal a cloud embedding provider is configured.
_CLOUD_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "LITELLM_API_KEY",
)

# Module-level guard so the chosen mode is logged once per process.
_mode_logged = False


class EmbedderMode(StrEnum):
    """Active embedding mode selected by the auto-detect ladder."""

    CLOUD = "cloud"
    LOCAL = "local"
    SPARSE_ONLY = "sparse_only"


@dataclass
class EmbeddingResult:
    """Embedding output for one text.

    Attributes:
        dense:          Dense vector, or ``None`` in sparse-only mode.
        sparse_indices: Sorted term indices of the sparse vector.
        sparse_values:  Values aligned with ``sparse_indices``.
    """

    dense: list[float] | None
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder(Protocol):
    """Structural interface the sync engine consumes from an embedder.

    This is the minimal surface actually used by ``SyncEngine``: batch
    embedding, the dense dimension for collection creation and stage
    validation, and the fingerprint identity string.  ``EmbedderProvider``
    satisfies it in production; deterministic test fakes satisfy it
    structurally without subclassing.
    """

    @property
    def dimension(self) -> int:
        """Dense vector dimension declared by configuration."""
        ...

    @property
    def fingerprint_model_id(self) -> str:
        """Model identity string for the pipeline fingerprint (mode + name)."""
        ...

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of EmbeddingResult, one per input text, in order.
        """
        ...


def _stable_token_index(token: str) -> int:
    """Map a token to a stable vocabulary index via SHA-256.

    Stable across processes and Python versions (no ``hash()`` salt).

    Args:
        token: Lowercased token.

    Returns:
        Integer index in ``[0, VOCAB_SIZE)``.
    """
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % VOCAB_SIZE


def compute_sparse(text: str) -> tuple[list[int], list[float]]:
    """Compute a TF-only (no IDF) sparse vector from term frequencies.

    The weight for each token is its **term frequency** (TF): count / total
    tokens.  There is intentionally **no IDF component** - IDF requires a
    document corpus to compute and would be non-deterministic at query time.
    The Epic 03 ranker is expected to apply BM25-style IDF weighting at
    retrieval time when re-ranking candidates; this layer only provides
    normalised per-document TF weights.

    **Vocabulary / collision behaviour:**
    Token strings are mapped to vocabulary indices via SHA-256 truncated to 8
    bytes and reduced modulo ``VOCAB_SIZE`` (30 000 slots).  Two distinct tokens
    can map to the same slot (SHA-256 truncation collision); the colliding
    tokens' frequencies are summed under that index.  The probability of a
    collision within a realistic document vocabulary is very low (birthday-
    paradox estimate: ~2% for a 100-term vocabulary over 30 000 slots), but
    callers must be aware that high-cardinality text may exhibit slight TF
    inflation for colliding token pairs.

    Args:
        text: Input text.

    Returns:
        Tuple of (sorted indices, aligned values).  Empty lists for blank text.
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return [], []

    freq: dict[int, int] = {}
    for token in tokens:
        idx = _stable_token_index(token)
        freq[idx] = freq.get(idx, 0) + 1

    total = len(tokens)
    indices = sorted(freq.keys())
    values = [freq[i] / total for i in indices]
    return indices, values


def _detect_mode() -> EmbedderMode:
    """Run the auto-detect ladder over environment configuration only.

    No network probe and no model load happens here.

    Returns:
        The selected ``EmbedderMode``.
    """
    if any(os.environ.get(var) for var in _CLOUD_KEY_ENV_VARS):
        return EmbedderMode.CLOUD

    local_allowed = os.environ.get("BEACON_OFFLINE_ALLOW_LOCAL_MODELS", "1") != "0"
    local_blocked_offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    local_installed = importlib.util.find_spec("sentence_transformers") is not None

    if local_allowed and local_installed and not local_blocked_offline:
        return EmbedderMode.LOCAL

    return EmbedderMode.SPARSE_ONLY


class EmbedderProvider:
    """Embedding provider with mode auto-detection at construction time.

    Mode selection priority (environment-driven, never probes the network):

    1. CLOUD - a recognised cloud API key is present in the environment.
    2. LOCAL - sentence-transformers is installed, not blocked by
       ``HF_HUB_OFFLINE=1``, and not disabled via
       ``BEACON_OFFLINE_ALLOW_LOCAL_MODELS=0``.
    3. SPARSE_ONLY - the floor: sparse vectors only, no dense embeddings.

    Args:
        model_name: The embedding model name (used for cloud and local modes).
        dimension:  Dense vector dimension declared by configuration.
    """

    def __init__(self, *, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.mode = _detect_mode()
        self._st_model: Any = None

        global _mode_logged
        if not _mode_logged:
            logger.info(
                "Embedding mode selected: %s (model=%r, dimension=%d)",
                self.mode.value,
                model_name,
                dimension,
            )
            _mode_logged = True

    @property
    def fingerprint_model_id(self) -> str:
        """Model identity string for the pipeline fingerprint.

        Includes the active mode so a mode change (e.g. an API key appearing)
        invalidates prior revisions instead of mixing vector generations.
        """
        return f"{self.mode.value}:{self.model_name}"

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of ``EmbeddingResult``, one per input, in order.

        Raises:
            IngestionError: If the active mode's backing library is missing or
                the embedding call fails.
        """
        if self.mode == EmbedderMode.CLOUD:
            return self._embed_cloud(texts)
        if self.mode == EmbedderMode.LOCAL:
            return self._embed_local(texts)
        return self._embed_sparse_only(texts)

    def embed_one(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Input text.

        Returns:
            ``EmbeddingResult`` for the text.
        """
        return self.embed([text])[0]

    def _embed_cloud(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed via LiteLLM against the configured cloud provider."""
        try:
            import litellm
        except ImportError as exc:
            raise IngestionError(
                "litellm is required for cloud embedding mode. "
                "Install it with: pip install litellm"
            ) from exc

        try:
            response = litellm.embedding(model=self.model_name, input=texts)
        except Exception as exc:
            raise IngestionError(
                f"Cloud embedding call failed for model {self.model_name!r}: {exc}"
            ) from exc

        results = []
        for i, text in enumerate(texts):
            dense = list(response.data[i]["embedding"])
            indices, values = compute_sparse(text)
            results.append(
                EmbeddingResult(dense=dense, sparse_indices=indices, sparse_values=values)
            )
        return results

    def _embed_local(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed via a local sentence-transformers model."""
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise IngestionError(
                "sentence-transformers is required for local embedding mode. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        if self._st_model is None:
            try:
                self._st_model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise IngestionError(
                    f"Failed to load local embedding model {self.model_name!r}: "
                    f"{exc}. In offline environments pre-download the model or "
                    "rely on the sparse-only floor."
                ) from exc

        vectors = self._st_model.encode(texts, convert_to_numpy=True)
        results = []
        for i, text in enumerate(texts):
            dense = list(map(float, vectors[i]))
            indices, values = compute_sparse(text)
            results.append(
                EmbeddingResult(dense=dense, sparse_indices=indices, sparse_values=values)
            )
        return results

    def _embed_sparse_only(self, texts: list[str]) -> list[EmbeddingResult]:
        """Sparse-only floor: term-frequency sparse vectors, no dense output."""
        results = []
        for text in texts:
            indices, values = compute_sparse(text)
            results.append(
                EmbeddingResult(dense=None, sparse_indices=indices, sparse_values=values)
            )
        return results
