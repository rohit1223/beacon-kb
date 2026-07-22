"""Run EmbedderContract against FakeEmbedder to prove the contract suite works."""
from __future__ import annotations

from beacon_kb.testing import EmbedderContract, FakeEmbedder


class TestFakeEmbedderContract(EmbedderContract):
    """Verify that FakeEmbedder satisfies the full EmbedderContract."""

    def make_subject(self) -> FakeEmbedder:
        return FakeEmbedder(dim=16, batch_size=4)
