"""Unit tests for EnrichmentOrchestrator."""
from __future__ import annotations

import pytest

from beacon_kb.errors import IngestionError
from beacon_kb.ingestion.enrichment import EnrichmentOrchestrator
from beacon_kb.testing import FakeEnricher, FakeFailingEnricher


def test_enrichment_disabled_returns_none() -> None:
    """When no enricher is provided, enrich() returns None."""
    orch = EnrichmentOrchestrator(enricher=None)
    result = orch.enrich("some text")
    assert result is None


def test_enrichment_returns_enriched_text() -> None:
    enricher = FakeEnricher(summaries={"hello world": "A greeting."})
    orch = EnrichmentOrchestrator(enricher=enricher, prompt="Summarize:", model_version="v1")
    result = orch.enrich("hello world")
    assert result == "A greeting."


def test_enrichment_cached_by_content_and_config() -> None:
    """Identical text + prompt + model_version must be served from cache."""
    enricher = FakeEnricher()
    orch = EnrichmentOrchestrator(enricher=enricher, prompt="Summarize:", model_version="v1")
    r1 = orch.enrich("unique text for cache test")
    r2 = orch.enrich("unique text for cache test")
    assert r1 == r2
    # The enricher should only have been called once (second call served from cache).
    assert len(enricher.calls) == 1


def test_enrichment_cache_misses_on_different_text() -> None:
    enricher = FakeEnricher()
    orch = EnrichmentOrchestrator(enricher=enricher, prompt="Summarize:", model_version="v1")
    orch.enrich("text A")
    orch.enrich("text B")
    assert len(enricher.calls) == 2


def test_enrichment_cache_misses_on_different_prompt() -> None:
    enricher = FakeEnricher()
    orch1 = EnrichmentOrchestrator(enricher=enricher, prompt="Summarize:", model_version="v1")
    orch2 = EnrichmentOrchestrator(enricher=enricher, prompt="Explain:", model_version="v1")
    orch1.enrich("same text")
    orch2.enrich("same text")
    assert len(enricher.calls) == 2


def test_enrichment_best_effort_policy_swallows_failure() -> None:
    """With failure_policy='best-effort', a failing enricher returns None."""
    enricher = FakeFailingEnricher()
    orch = EnrichmentOrchestrator(
        enricher=enricher,
        prompt="Summarize:",
        model_version="v1",
        failure_policy="best-effort",
    )
    result = orch.enrich("some text")
    assert result is None
    assert enricher.calls == 1


def test_enrichment_raise_policy_propagates_error() -> None:
    """With failure_policy='raise', a failing enricher propagates IngestionError."""
    enricher = FakeFailingEnricher()
    orch = EnrichmentOrchestrator(
        enricher=enricher,
        prompt="Summarize:",
        model_version="v1",
        failure_policy="raise",
    )
    with pytest.raises(IngestionError):
        orch.enrich("some text")


def test_ingestion_succeeds_with_failing_enricher_best_effort() -> None:
    """Validate that a full simulated ingestion loop completes with a failing enricher."""
    enricher = FakeFailingEnricher()
    orch = EnrichmentOrchestrator(
        enricher=enricher,
        failure_policy="best-effort",
    )
    texts = ["doc part 1", "doc part 2", "doc part 3"]
    results = [orch.enrich(t) for t in texts]
    assert all(r is None for r in results), "best-effort: all failures should return None"
    assert enricher.calls == 3
