"""Provider-neutral batched embedding pipeline stage.

Design:
- Batch size is ALWAYS read from Embedder.batch_size. Never hardcoded.
- Dimension is validated against provider.dimension() after each batch.
- Results are cached by chunk.id so re-processing the same chunk within
  one BatchEmbedder lifetime avoids redundant provider calls.
- Retry logic: up to max_retries attempts per batch before raising BackendError.
- The embed stage is a long stage: it emits start/progress/end events with
  current/total counts and elapsed time through the structured progress
  adapter (``beacon_kb.progress``) when an observer is injected.

Importing this module performs no side effects.
"""

from __future__ import annotations

import time

from beacon_kb.errors import BackendError
from beacon_kb.models import Chunk
from beacon_kb.progress import Clock, NullProgressObserver, make_stage_emitter
from beacon_kb.protocols import Embedder, ProgressObserver


class BatchEmbedder:
    """Drive batched embedding through the Embedder protocol.

    Reads provider.batch_size to determine how many texts to send per API call.
    Never hardcodes a batch size constant.

    Args:
        provider:    An Embedder-protocol-conforming instance.
        max_retries: How many times to retry a failing batch before raising.
                     Default is 3.
        observer:    Optional ProgressObserver receiving structured stage
                     events (start, progress with current/total, end,
                     elapsed_seconds).  Defaults to a no-op observer.
        clock:       Optional injectable clock for elapsed-time computation
                     in progress events (defaults to a monotonic wall clock).
    """

    def __init__(
        self,
        *,
        provider: Embedder,
        max_retries: int = 3,
        observer: ProgressObserver | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._provider = provider
        self._max_retries = max_retries
        self._observer: ProgressObserver = (
            observer if observer is not None else NullProgressObserver()
        )
        self._clock = clock
        self._cache: dict[str, list[float]] = {}

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Return (chunk, vector) pairs for all chunks.

        Chunks with IDs already in the cache are served without contacting
        the provider.  Remaining chunks are sent to the provider in batches
        of provider.batch_size.

        Dimension of each returned vector is validated to equal provider.dimension().
        If the provider returns a mismatched dimension, BackendError is raised.

        Args:
            chunks: Ordered list of Chunk records to embed.

        Returns:
            List of (Chunk, vector) pairs in the same order as the input.

        Raises:
            BackendError: If the provider fails after max_retries attempts,
                          or if any returned vector has the wrong dimension.
        """
        if not chunks:
            return []

        expected_dim = self._provider.dimension()
        batch_size = self._provider.batch_size  # Never hardcoded

        # Partition into cache hits and misses.
        results: dict[str, list[float]] = {}
        uncached: list[Chunk] = []

        for chunk in chunks:
            key = str(chunk.id)
            if key in self._cache:
                results[key] = self._cache[key]
            else:
                uncached.append(chunk)

        # Embed uncached chunks in provider-owned batches, emitting structured
        # progress (start / progress with current/total / end / elapsed) for
        # this long stage through the injected observer.
        done = len(chunks) - len(uncached)
        with make_stage_emitter(
            "embed",
            observer=self._observer,
            clock=self._clock,
            total=len(chunks),
        ) as emit:
            for batch_start in range(0, len(uncached), batch_size):
                batch = uncached[batch_start : batch_start + batch_size]
                texts = [c.text for c in batch]

                vector_batch: list[list[float]] | None = None
                last_exc: Exception | None = None

                for attempt in range(self._max_retries):
                    try:
                        vector_batch = self._provider.embed(texts)
                        break
                    except BackendError as exc:
                        last_exc = exc
                        if attempt < self._max_retries - 1:
                            # Placeholder - see ROADMAP.md "Embedding retry back-off".
                            time.sleep(0.0)

                if vector_batch is None:
                    raise BackendError(
                        f"Embedding provider failed after {self._max_retries} attempts: "
                        f"{last_exc}"
                    ) from last_exc

                if len(vector_batch) != len(batch):
                    raise BackendError(
                        f"Provider returned {len(vector_batch)} vectors for {len(batch)} texts."
                    )

                for chunk, vec in zip(batch, vector_batch, strict=True):
                    if len(vec) != expected_dim:
                        raise BackendError(
                            f"Provider returned vector of dimension {len(vec)} "
                            f"but expected {expected_dim} (chunk_id={chunk.id!r})."
                        )
                    self._cache[str(chunk.id)] = vec
                    results[str(chunk.id)] = vec

                done += len(batch)
                emit(current=done)

        # Reconstruct in original order.
        return [(chunk, results[str(chunk.id)]) for chunk in chunks]
