"""Unit tests for beacon_kb.tokens.

Tests cover:
- HeuristicTokenCounter.count_tokens() returns non-negative integer.
- Empty string returns 0.
- Count grows with text length.
- Protocol conformance: HeuristicTokenCounter satisfies TokenCounter.
- compute_evidence_budget() selects items that fit within budget.
- compute_evidence_budget() returns correct result_count and overflow_count.
- compute_evidence_budget() with negative budget raises BeaconError.
- summarize_budget() returns a human-readable string with key values.
- BudgetSummary is frozen.
"""

from __future__ import annotations

import dataclasses

import pytest

from beacon_kb.errors import BeaconError
from beacon_kb.protocols import TokenCounter
from beacon_kb.tokens import (
    BudgetSummary,
    HeuristicTokenCounter,
    compute_evidence_budget,
    summarize_budget,
)

# ===========================================================================
# HeuristicTokenCounter
# ===========================================================================


@pytest.mark.unit
class TestHeuristicTokenCounter:
    """HeuristicTokenCounter satisfies the TokenCounter protocol."""

    def test_satisfies_token_counter_protocol(self) -> None:
        counter = HeuristicTokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_empty_string_returns_zero(self) -> None:
        counter = HeuristicTokenCounter()
        assert counter.count_tokens("") == 0

    def test_non_empty_string_returns_positive(self) -> None:
        counter = HeuristicTokenCounter()
        result = counter.count_tokens("Hello, world!")
        assert result > 0

    def test_count_is_non_negative(self) -> None:
        counter = HeuristicTokenCounter()
        for text in ["", "a", "hello world", "x" * 1000]:
            assert counter.count_tokens(text) >= 0

    def test_longer_text_has_more_tokens(self) -> None:
        counter = HeuristicTokenCounter()
        short = counter.count_tokens("hi")
        long = counter.count_tokens("hi " * 100)
        assert long > short

    def test_model_param_accepted(self) -> None:
        counter = HeuristicTokenCounter()
        result = counter.count_tokens("test text", model="gpt-4")
        assert result > 0

    def test_unknown_model_uses_heuristic(self) -> None:
        counter = HeuristicTokenCounter()
        result_unknown = counter.count_tokens("test text", model="unknown-model-xyz")
        result_default = counter.count_tokens("test text")
        assert result_unknown == result_default

    def test_returns_integer(self) -> None:
        counter = HeuristicTokenCounter()
        result = counter.count_tokens("some text")
        assert isinstance(result, int)


# ===========================================================================
# compute_evidence_budget
# ===========================================================================


@pytest.mark.unit
class TestComputeEvidenceBudget:
    """compute_evidence_budget() enforces token budgets over evidence lists."""

    def test_empty_list_returns_zero_results(self) -> None:
        summary = compute_evidence_budget([], token_budget=1000)
        assert summary.result_count == 0
        assert summary.total_tokens == 0
        assert summary.overflow_count == 0

    def test_all_items_fit_within_budget(self) -> None:
        texts = ["short text"] * 3
        summary = compute_evidence_budget(texts, token_budget=1000)
        assert summary.result_count == 3
        assert summary.overflow_count == 0

    def test_some_items_excluded_by_budget(self) -> None:
        counter = HeuristicTokenCounter()
        # Make texts long enough that only a few fit
        long_text = "word " * 200  # ~200 tokens
        texts = [long_text] * 10
        per_item = counter.count_tokens(long_text)
        budget = per_item * 3  # only 3 should fit
        summary = compute_evidence_budget(texts, token_budget=budget, counter=counter)
        assert summary.result_count <= 3
        assert summary.overflow_count >= 7

    def test_result_count_plus_overflow_equals_total(self) -> None:
        texts = ["text " * 50] * 5
        summary = compute_evidence_budget(texts, token_budget=500)
        assert summary.result_count + summary.overflow_count == len(texts)

    def test_total_tokens_within_budget(self) -> None:
        texts = ["hello world " * 10] * 10
        budget = 50
        summary = compute_evidence_budget(texts, token_budget=budget)
        assert summary.total_tokens <= budget

    def test_remaining_tokens_correct(self) -> None:
        texts = ["short"] * 2
        counter = HeuristicTokenCounter()
        per_item = counter.count_tokens("short")
        total_used = per_item * 2
        budget = 1000
        summary = compute_evidence_budget(texts, token_budget=budget, counter=counter)
        # remaining = budget - total_used (overhead=0)
        assert summary.remaining_tokens == budget - total_used

    def test_budget_attribute_matches_input(self) -> None:
        summary = compute_evidence_budget(["text"], token_budget=500)
        assert summary.budget == 500

    def test_negative_budget_raises_beacon_error(self) -> None:
        with pytest.raises(BeaconError):
            compute_evidence_budget(["text"], token_budget=-1)

    def test_overhead_tokens_reduces_effective_budget(self) -> None:
        texts = ["word " * 100] * 10
        counter = HeuristicTokenCounter()
        per_item = counter.count_tokens("word " * 100)
        budget = per_item * 5
        # With overhead = per_item * 2, effective budget = per_item * 3
        summary_no_overhead = compute_evidence_budget(
            texts, token_budget=budget, counter=counter
        )
        summary_with_overhead = compute_evidence_budget(
            texts, token_budget=budget, counter=counter, overhead_tokens=per_item * 2
        )
        assert summary_with_overhead.result_count <= summary_no_overhead.result_count

    def test_custom_counter_used(self) -> None:
        class FixedCounter:
            def count_tokens(self, text: str, *, model: str = "") -> int:
                return 10  # every text costs exactly 10 tokens

        texts = ["any text"] * 5
        summary = compute_evidence_budget(texts, token_budget=25, counter=FixedCounter())
        # 10+10=20 fits, 10+10+10=30 > 25 does not
        assert summary.result_count == 2
        assert summary.total_tokens == 20


# ===========================================================================
# BudgetSummary
# ===========================================================================


@pytest.mark.unit
class TestBudgetSummary:
    """BudgetSummary is a frozen dataclass."""

    def test_budget_summary_is_frozen(self) -> None:
        summary = BudgetSummary(
            result_count=3,
            total_tokens=100,
            remaining_tokens=900,
            budget=1000,
            overflow_count=7,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            summary.result_count = 0  # type: ignore[misc]

    def test_str_contains_key_values(self) -> None:
        summary = BudgetSummary(
            result_count=3,
            total_tokens=100,
            remaining_tokens=900,
            budget=1000,
            overflow_count=7,
        )
        text = str(summary)
        assert "100" in text  # total_tokens
        assert "1000" in text  # budget


# ===========================================================================
# summarize_budget
# ===========================================================================


@pytest.mark.unit
class TestSummarizeBudget:
    """summarize_budget() returns a human-readable single-line string."""

    def test_summarize_returns_string(self) -> None:
        summary = BudgetSummary(
            result_count=5,
            total_tokens=200,
            remaining_tokens=800,
            budget=1000,
            overflow_count=3,
        )
        result = summarize_budget(summary)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_contains_counts(self) -> None:
        summary = BudgetSummary(
            result_count=5,
            total_tokens=200,
            remaining_tokens=800,
            budget=1000,
            overflow_count=3,
        )
        result = summarize_budget(summary)
        assert "200" in result
        assert "1000" in result
