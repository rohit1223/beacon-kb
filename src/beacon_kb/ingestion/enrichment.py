"""Optional cached enrichment orchestration.

Enrichment is optional, cached by content + prompt + model_version, and
controlled by a failure policy.  Ingestion must always succeed regardless
of enrichment status.  Summaries, keywords, and FAQs produced here are
optional searchable metadata - never embedding prerequisites.

Importing this module performs no side effects.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol, runtime_checkable

from beacon_kb.errors import IngestionError


@runtime_checkable
class Enricher(Protocol):
    """Protocol for optional LLM enrichment providers.

    Score direction: N/A - enrichers produce text, not scores.
    Error contract: enrich() raises IngestionError on provider failure.
    Determinism: enrichers are generally non-deterministic; results are
    cached by content + prompt + model version at the orchestration layer.
    """

    def enrich(self, text: str, *, prompt: str = "") -> str:
        """Return enriched text (summary, keywords, FAQ) for *text*.

        Args:
            text:   The chunk text to enrich.
            prompt: Instructional prompt controlling the enrichment output.

        Returns:
            Enriched text string.

        Raises:
            IngestionError: On provider failure.
        """
        ...


def _cache_key(text: str, prompt: str, model_version: str) -> str:
    """Return a stable cache key for enrichment results.

    The key is SHA-256 over 'model_version:prompt:text' so that changes to
    any of the three inputs produce a different cache bucket.
    """
    canonical = f"{model_version}:{prompt}:{text}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EnrichmentOrchestrator:
    """Orchestrate optional LLM enrichment with in-memory caching and failure policy.

    Enrichment is completely optional: if *enricher* is None, every call to
    enrich() returns None immediately and no LLM is contacted.

    Results are cached by a SHA-256 key over (model_version + prompt + text)
    so that identical inputs never call the enricher twice within the lifetime
    of this orchestrator instance.

    Failure policy:
        'best-effort' (default): a failing enricher returns None and does not
            interrupt ingestion.
        'raise': a failing enricher propagates the original IngestionError so
            the caller can decide how to handle it.

    Args:
        enricher:        Any object with ``enrich(text: str, *, prompt: str) -> str``,
                         or None to disable enrichment entirely.
        prompt:          Prompt string passed to the enricher (default: '').
        model_version:   String identifying the enricher model/version for cache keying.
        failure_policy:  'best-effort' or 'raise'.

    Raises:
        ValueError: If failure_policy is not 'best-effort' or 'raise'.
    """

    def __init__(
        self,
        *,
        enricher: Enricher | None,
        prompt: str = "",
        model_version: str = "",
        failure_policy: Literal["best-effort", "raise"] = "best-effort",
    ) -> None:
        if failure_policy not in ("best-effort", "raise"):
            raise ValueError(
                f"failure_policy must be 'best-effort' or 'raise', got {failure_policy!r}."
            )
        self._enricher = enricher
        self._prompt = prompt
        self._model_version = model_version
        self._failure_policy = failure_policy
        self._cache: dict[str, str] = {}

    def enrich(self, text: str) -> str | None:
        """Return enriched text for *text*, or None if enrichment is disabled/failed.

        Caches results by SHA-256(model_version + prompt + text). A second call
        with identical inputs returns the cached result without calling the enricher.

        Args:
            text: The chunk text to enrich.

        Returns:
            Enriched string from the enricher, or None if enricher is None or
            if enrichment failed under 'best-effort' policy.

        Raises:
            IngestionError: Only when failure_policy='raise' and the enricher raises.
        """
        if self._enricher is None:
            return None

        key = _cache_key(text, self._prompt, self._model_version)
        if key in self._cache:
            return self._cache[key]

        try:
            result: str = self._enricher.enrich(text, prompt=self._prompt)
        except IngestionError:
            if self._failure_policy == "raise":
                raise
            return None
        except Exception as exc:
            if self._failure_policy == "raise":
                raise IngestionError(f"Enrichment failed: {exc}") from exc
            return None

        self._cache[key] = result
        return result
