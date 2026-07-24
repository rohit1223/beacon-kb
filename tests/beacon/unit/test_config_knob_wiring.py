"""Tests that config knobs are actually wired - no silent no-ops."""
from __future__ import annotations

from beacon.config import AnswerSettings, BeaconSettings, RetrievalSettings


class TestDeadKnobbsDeleted:
    def test_retrieval_settings_has_no_rerank_field(self) -> None:
        """RetrievalSettings.rerank was deleted - it must not exist."""
        assert not hasattr(RetrievalSettings(), "rerank"), (
            "RetrievalSettings.rerank was deleted (no-op knob); "
            "if you see this, the field was re-added without wiring."
        )

    def test_retrieval_settings_has_no_parent_expansion_field(self) -> None:
        """RetrievalSettings.parent_expansion was deleted - it must not exist."""
        assert not hasattr(RetrievalSettings(), "parent_expansion"), (
            "RetrievalSettings.parent_expansion was deleted (no-op knob); "
            "if you see this, the field was re-added without wiring."
        )


class TestLlmTimeoutWired:
    def test_answer_settings_has_llm_timeout_s(self) -> None:
        """AnswerSettings must have llm_timeout_s with a sensible default."""
        settings = AnswerSettings()
        assert hasattr(settings, "llm_timeout_s")
        assert isinstance(settings.llm_timeout_s, float)
        assert settings.llm_timeout_s > 0

    def test_llm_timeout_default_is_60(self) -> None:
        assert AnswerSettings().llm_timeout_s == 60.0


class TestTopKDefault:
    def test_retrieval_top_k_default_is_10(self) -> None:
        assert RetrievalSettings().top_k == 10


class TestSafeDumpContainsLlmTimeout:
    def test_safe_dump_includes_llm_timeout_s(self) -> None:
        settings = BeaconSettings()
        dump = settings.safe_dump()
        assert "llm_timeout_s" in dump["answer"], (
            "safe_dump must include llm_timeout_s under 'answer'"
        )

    def test_safe_dump_excludes_deleted_knobs(self) -> None:
        settings = BeaconSettings()
        dump = settings.safe_dump()
        assert "rerank" not in dump["retrieval"], (
            "safe_dump must not include deleted 'rerank' knob"
        )
        assert "parent_expansion" not in dump["retrieval"], (
            "safe_dump must not include deleted 'parent_expansion' knob"
        )
