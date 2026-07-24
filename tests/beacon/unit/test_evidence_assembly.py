"""Unit tests for evidence assembly: packing, budgets, labels, context_of, dedup.

TDD suite - written before the implementation in src/beacon/retrieval/evidence.py.

Coverage:
- Primary hits pack before context spans.
- Token budget never exceeded; skipped hits excluded.
- Skipping an oversized mid-rank hit leaves labels contiguous (gap-free S1..Sn).
- context_of is a structured reference (the chunk_id string), not a string
  embedded in the label.
- context spans carry no relevance score (score=None).
- No chunk_id appears twice in one bundle.
- Expansion occurs only after hits are passed in final order (determinism).
- BudgetRecap fields are accurate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from beacon.models import EvidenceBundle
from beacon.retrieval.evidence import (
    EvidenceRole,
    assemble_evidence,
)

# ---------------------------------------------------------------------------
# Helpers - synthetic Hit-like dicts (no Qdrant needed)
# ---------------------------------------------------------------------------


def _make_payload(
    chunk_id: str,
    text: str,
    source_uri: str = "file:///docs/test.md",
    title: str = "Test Doc",
    heading_path: list[str] | None = None,
    parent_chunk_id: str | None = None,
    prev_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    kind: str = "child",
) -> dict[str, Any]:
    # chunk_hash is the canonical hex chunk id shared by navigation fields
    # (prev_chunk_id / next_chunk_id) and Evidence.chunk_id.  We use chunk_id
    # itself so that the dedup set and the neighbor chain share the same key
    # space in tests, matching the production invariant.
    return {
        "chunk_text": text,
        "source_uri": source_uri,
        "title": title,
        "heading_path": heading_path or ["Introduction"],
        "tags": [],
        "kind": kind,
        "section_kind": "text",
        "parent_chunk_id": parent_chunk_id,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "ingested_at": "2025-01-01T00:00:00Z",
        "content_hash": "abc123",
        "chunk_hash": chunk_id,
        "fingerprint": "fp001",
    }


def _fake_chunk_fetch(
    payloads: dict[str, dict[str, Any]],
) -> Callable[[str], dict[str, Any] | None]:
    """Return a callable that mimics a chunk lookup function."""

    def fetch(chunk_id: str) -> dict[str, Any] | None:
        return payloads.get(chunk_id)

    return fetch


# ---------------------------------------------------------------------------
# Tests: primary packing order
# ---------------------------------------------------------------------------


class TestPrimaryPackingOrder:
    """Primary HITs must appear before CONTEXT spans in the bundle."""

    def test_primary_before_context(self) -> None:
        """All HIT items must come before all CONTEXT items."""
        # Three chunks in a chain; middle is primary, neighbors become context.
        payloads = {
            "cA": _make_payload("cA", "alpha text neighbors", next_chunk_id="cB"),
            "cB": _make_payload("cB", "beta main hit text", prev_chunk_id="cA", next_chunk_id="cC"),
            "cC": _make_payload("cC", "gamma text neighbor", prev_chunk_id="cB"),
        }

        from beacon.retrieval.hybrid import Hit

        hits = [
            Hit(chunk_point_id="cB", payload=payloads["cB"], fused_score=1.0),
        ]

        bundle = assemble_evidence(
            hits=hits,
            query_text="beta main hit",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=2,
        )

        hit_positions = [i for i, e in enumerate(bundle.evidence) if e.role == EvidenceRole.HIT]
        ctx_positions = [i for i, e in enumerate(bundle.evidence) if e.role == EvidenceRole.CONTEXT]

        if hit_positions and ctx_positions:
            assert max(hit_positions) < min(ctx_positions), (
                "All primary HITs must appear before any CONTEXT span"
            )


# ---------------------------------------------------------------------------
# Tests: budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Token budget must never be exceeded; skipped items excluded from evidence."""

    def test_budget_not_exceeded(self) -> None:
        """Total token count of packed evidence must not exceed the budget."""
        # Each payload text is ~40 chars -> ~10 tokens (heuristic: chars/4).
        texts = ["word " * 8 for _ in range(5)]
        payloads = {
            f"c{i}": _make_payload(f"c{i}", texts[i]) for i in range(5)
        }

        from beacon.retrieval.hybrid import Hit

        hits = [
            Hit(chunk_point_id=f"c{i}", payload=payloads[f"c{i}"], fused_score=float(5 - i))
            for i in range(5)
        ]

        # Budget: enough for only 2 chunks.
        per_chunk_tokens = math.ceil(len(texts[0]) / 4)
        budget = per_chunk_tokens * 2 + 1  # room for exactly 2

        bundle = assemble_evidence(
            hits=hits,
            query_text="word",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=budget,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        # Use the recap's token_total for the assertion.
        assert bundle.recap.tokens_packed <= budget, (
            f"Token total {bundle.recap.tokens_packed} must not exceed budget {budget}"
        )

    def test_skipped_items_excluded(self) -> None:
        """Hits that exceed remaining budget must not appear in the bundle."""
        large_text = "big word " * 50  # ~450 chars -> ~113 tokens
        small_text = "small"
        payloads = {
            "big": _make_payload("big", large_text),
            "small": _make_payload("small", small_text),
        }

        from beacon.retrieval.hybrid import Hit

        hits = [
            Hit(chunk_point_id="big", payload=payloads["big"], fused_score=2.0),
            Hit(chunk_point_id="small", payload=payloads["small"], fused_score=1.0),
        ]

        # Budget only fits "small".
        bundle = assemble_evidence(
            hits=hits,
            query_text="small",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=5,  # only "small" (~2 tokens) fits
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        included_ids = {e.chunk_id for e in bundle.evidence}
        assert "big" not in included_ids, "Oversized chunk must be excluded from bundle"
        assert "small" in included_ids, "Small chunk must be included in bundle"


# ---------------------------------------------------------------------------
# Tests: gap-free labels after overflow
# ---------------------------------------------------------------------------


class TestGapFreeLabels:
    """Labels must be gap-free S1..Sn even when mid-rank items overflow."""

    def test_labels_contiguous_after_mid_rank_overflow(self) -> None:
        """S1, S2 must be assigned even when the rank-2 chunk overflows."""
        # rank-1: small (~5 tokens), rank-2: huge (will overflow), rank-3: small (~5 tokens)
        small1 = "short"
        large = "bigword " * 60  # ~120 tokens
        small2 = "brief"

        payloads = {
            "r1": _make_payload("r1", small1),
            "r2": _make_payload("r2", large),
            "r3": _make_payload("r3", small2),
        }

        from beacon.retrieval.hybrid import Hit

        hits = [
            Hit(chunk_point_id="r1", payload=payloads["r1"], fused_score=3.0),
            Hit(chunk_point_id="r2", payload=payloads["r2"], fused_score=2.0),
            Hit(chunk_point_id="r3", payload=payloads["r3"], fused_score=1.0),
        ]

        # Budget that fits r1 and r3 but not r2.
        tok_r1 = math.ceil(len(small1) / 4)
        tok_r3 = math.ceil(len(small2) / 4)
        tok_r2 = math.ceil(len(large) / 4)
        budget = tok_r1 + tok_r3 + 2
        assert budget < tok_r2, "Test invariant: budget must not fit the large chunk"

        bundle = assemble_evidence(
            hits=hits,
            query_text="short brief",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=budget,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        hit_ev = [e for e in bundle.evidence if e.role == EvidenceRole.HIT]
        assert len(hit_ev) == 2, f"Expected 2 packed hits, got {len(hit_ev)}"

        # Must not contain the large chunk.
        included_ids = {e.chunk_id for e in hit_ev}
        assert "r2" not in included_ids, "Oversized rank-2 chunk must be excluded"

        # Labels must be S1, S2 with no gap.
        labels = [e.label for e in hit_ev]
        assert labels == ["S1", "S2"], (
            f"Labels must be contiguous ['S1', 'S2'], got {labels!r}"
        )

    def test_labels_always_start_at_s1(self) -> None:
        """First packed hit always gets label S1."""
        payloads = {"c1": _make_payload("c1", "hello world")}
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="c1", payload=payloads["c1"], fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="hello",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=5000,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )
        assert bundle.evidence[0].label == "S1"

    def test_context_labels_continue_sequence(self) -> None:
        """Context spans get labels that continue from primary labels (S2, S3...)."""
        payloads = {
            "cA": _make_payload("cA", "alpha primary", next_chunk_id="cB"),
            "cB": _make_payload("cB", "beta neighbor", prev_chunk_id="cA"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="cA", payload=payloads["cA"], fused_score=1.0)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="alpha beta",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )

        labels = [e.label for e in bundle.evidence]
        for i, label in enumerate(labels, start=1):
            assert label == f"S{i}", (
                f"Expected S{i} at position {i - 1}, got {label!r}"
            )


# ---------------------------------------------------------------------------
# Tests: context_of is a structured field
# ---------------------------------------------------------------------------


class TestContextOf:
    """context_of must be a structured field (chunk_id string), not encoded in the label."""

    def test_context_of_set_on_context_spans(self) -> None:
        """Context spans must have context_of referencing the primary chunk_id."""
        payloads = {
            "primary": _make_payload("primary", "main content", next_chunk_id="neighbor"),
            "neighbor": _make_payload("neighbor", "neighbor content", prev_chunk_id="primary"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="primary", payload=payloads["primary"], fused_score=1.0)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="main",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )

        ctx_items = [e for e in bundle.evidence if e.role == EvidenceRole.CONTEXT]
        assert ctx_items, "Expected at least one CONTEXT span"
        for ctx_ev in ctx_items:
            assert ctx_ev.context_of is not None, "CONTEXT span must have context_of set"
            assert ctx_ev.context_of == "primary", (
                f"context_of must be the primary chunk_id 'primary', got {ctx_ev.context_of!r}"
            )

    def test_context_of_not_encoded_in_label(self) -> None:
        """The context_of value must not appear in the citation label string."""
        payloads = {
            "p1": _make_payload("p1", "primary text", next_chunk_id="c1"),
            "c1": _make_payload("c1", "context text", prev_chunk_id="p1"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="p1", payload=payloads["p1"], fused_score=1.0)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="text",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )

        ctx_items = [e for e in bundle.evidence if e.role == EvidenceRole.CONTEXT]
        for ctx_ev in ctx_items:
            assert "context_of" not in ctx_ev.label, (
                f"Label must not encode context_of: {ctx_ev.label!r}"
            )
            assert "p1" not in ctx_ev.label, (
                f"Label must not contain the context_of chunk_id: {ctx_ev.label!r}"
            )

    def test_context_spans_no_relevance_score(self) -> None:
        """Context spans must have no relevance score (score is None)."""
        payloads = {
            "p1": _make_payload("p1", "primary text", next_chunk_id="c1"),
            "c1": _make_payload("c1", "context neighbor", prev_chunk_id="p1"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="p1", payload=payloads["p1"], fused_score=1.5)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="text",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )

        ctx_items = [e for e in bundle.evidence if e.role == EvidenceRole.CONTEXT]
        assert ctx_items, "Expected at least one CONTEXT span"
        for ctx_ev in ctx_items:
            assert ctx_ev.score is None, (
                f"CONTEXT span must have score=None, got {ctx_ev.score!r}"
            )

    def test_primary_hits_have_score(self) -> None:
        """Primary HIT spans carry the fused_score from the Hit."""
        payloads = {"p1": _make_payload("p1", "primary text")}
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="p1", payload=payloads["p1"], fused_score=1.5)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="text",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=5000,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        hit_items = [e for e in bundle.evidence if e.role == EvidenceRole.HIT]
        assert hit_items
        assert hit_items[0].score == 1.5


# ---------------------------------------------------------------------------
# Tests: deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """No chunk_id may appear twice in a bundle."""

    def test_no_duplicate_chunk_ids(self) -> None:
        """If a neighbor is also a primary hit, it must not appear twice."""
        payloads = {
            "cA": _make_payload("cA", "alpha", next_chunk_id="cB"),
            "cB": _make_payload("cB", "beta", prev_chunk_id="cA"),
        }
        from beacon.retrieval.hybrid import Hit
        # Both cA and cB are primary hits; neighbor expansion of cA would add cB
        # but cB is already in primary hits, so it must not be duplicated.
        hits = [
            Hit(chunk_point_id="cA", payload=payloads["cA"], fused_score=2.0),
            Hit(chunk_point_id="cB", payload=payloads["cB"], fused_score=1.0),
        ]

        bundle = assemble_evidence(
            hits=hits,
            query_text="alpha beta",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=2,
        )

        chunk_ids = [e.chunk_id for e in bundle.evidence]
        assert len(chunk_ids) == len(set(chunk_ids)), (
            f"Duplicate chunk_ids found: {chunk_ids!r}"
        )


# ---------------------------------------------------------------------------
# Tests: expansion bounded and post-ordering
# ---------------------------------------------------------------------------


class TestExpansionBounds:
    """Expansion must be bounded by max_neighbor_hops and max_context_per_hit."""

    def test_max_context_per_hit_respected(self) -> None:
        """Context spans per primary hit must not exceed max_context_per_hit."""
        payloads = {
            "cA": _make_payload("cA", "alpha", prev_chunk_id="cPrev", next_chunk_id="cNext"),
            "cPrev": _make_payload("cPrev", "before alpha", next_chunk_id="cA"),
            "cNext": _make_payload("cNext", "after alpha", prev_chunk_id="cA"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="cA", payload=payloads["cA"], fused_score=1.0)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="alpha",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,  # only 1 context item per primary hit
        )

        ctx_items = [e for e in bundle.evidence if e.role == EvidenceRole.CONTEXT]
        assert len(ctx_items) <= 1, (
            f"Expected <= 1 context item per hit, got {len(ctx_items)}"
        )

    def test_max_neighbor_hops_respected(self) -> None:
        """Expansion must not exceed max_neighbor_hops hops in each direction."""
        # Chain of 5: p0 <-> p1 <-> p2 <-> p3 <-> p4
        payloads = {
            "p0": _make_payload("p0", "node zero", next_chunk_id="p1"),
            "p1": _make_payload("p1", "node one", prev_chunk_id="p0", next_chunk_id="p2"),
            "p2": _make_payload("p2", "node two main", prev_chunk_id="p1", next_chunk_id="p3"),
            "p3": _make_payload("p3", "node three", prev_chunk_id="p2", next_chunk_id="p4"),
            "p4": _make_payload("p4", "node four", prev_chunk_id="p3"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="p2", payload=payloads["p2"], fused_score=1.0)]

        bundle = assemble_evidence(
            hits=hits,
            query_text="node two",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,  # only 1 hop each direction -> p1 and p3
            max_context_per_hit=4,
        )

        ctx_ids = {e.chunk_id for e in bundle.evidence if e.role == EvidenceRole.CONTEXT}
        # With max_neighbor_hops=1 from p2: p1 and p3 are valid, p0 and p4 are NOT.
        assert "p0" not in ctx_ids, "p0 is 2 hops away, must not appear with max_hops=1"
        assert "p4" not in ctx_ids, "p4 is 2 hops away, must not appear with max_hops=1"

    def test_expansion_deterministic_for_same_inputs(self) -> None:
        """Identical hits and fetch callable must produce identical bundles."""
        payloads = {
            "cA": _make_payload("cA", "alpha text", next_chunk_id="cB"),
            "cB": _make_payload("cB", "beta text", prev_chunk_id="cA"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [Hit(chunk_point_id="cA", payload=payloads["cA"], fused_score=1.0)]

        b1 = assemble_evidence(
            hits=hits,
            query_text="alpha",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )
        b2 = assemble_evidence(
            hits=hits,
            query_text="alpha",
            fetch_chunk=_fake_chunk_fetch(payloads),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )

        assert [e.chunk_id for e in b1.evidence] == [e.chunk_id for e in b2.evidence], (
            "assemble_evidence must be deterministic for identical inputs"
        )


# ---------------------------------------------------------------------------
# Tests: BudgetRecap accuracy
# ---------------------------------------------------------------------------


class TestBudgetRecap:
    """EvidenceBundle.recap must accurately report requested, packed, skipped, tokens."""

    def test_recap_reports_all_fields(self) -> None:
        """BudgetRecap must have requested, packed, skipped, and tokens_packed."""
        payloads = {
            "c1": _make_payload("c1", "hello world small"),
            "c2": _make_payload("c2", "world " * 100),  # too large
        }
        from beacon.retrieval.hybrid import Hit
        hits = [
            Hit(chunk_point_id="c1", payload=payloads["c1"], fused_score=2.0),
            Hit(chunk_point_id="c2", payload=payloads["c2"], fused_score=1.0),
        ]

        bundle = assemble_evidence(
            hits=hits,
            query_text="hello",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=10,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        recap = bundle.recap
        assert hasattr(recap, "requested"), "recap must have 'requested' field"
        assert hasattr(recap, "packed"), "recap must have 'packed' field"
        assert hasattr(recap, "skipped"), "recap must have 'skipped' field"
        assert hasattr(recap, "tokens_packed"), "recap must have 'tokens_packed' field"

    def test_recap_counts_accurate(self) -> None:
        """packed + skipped == requested."""
        payloads = {
            "c1": _make_payload("c1", "small text"),
            "c2": _make_payload("c2", "enormous " * 100),
            "c3": _make_payload("c3", "tiny"),
        }
        from beacon.retrieval.hybrid import Hit
        hits = [
            Hit(chunk_point_id="c1", payload=payloads["c1"], fused_score=3.0),
            Hit(chunk_point_id="c2", payload=payloads["c2"], fused_score=2.0),
            Hit(chunk_point_id="c3", payload=payloads["c3"], fused_score=1.0),
        ]

        bundle = assemble_evidence(
            hits=hits,
            query_text="text",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=15,
            max_neighbor_hops=0,
            max_context_per_hit=0,
        )

        recap = bundle.recap
        assert recap.requested == 3
        assert recap.packed + recap.skipped == recap.requested

    def test_empty_hits_produces_empty_bundle(self) -> None:
        """Empty hit list must produce an empty bundle with zero recap counts."""
        bundle = assemble_evidence(
            hits=[],
            query_text="anything",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=2,
        )
        assert bundle.evidence == []
        assert bundle.recap.packed == 0
        assert bundle.recap.skipped == 0

    def test_bundle_is_evidence_bundle_type(self) -> None:
        """assemble_evidence must return an EvidenceBundle."""
        bundle = assemble_evidence(
            hits=[],
            query_text="anything",
            fetch_chunk=_fake_chunk_fetch({}),
            token_budget=5000,
        )
        assert isinstance(bundle, EvidenceBundle)


# ---------------------------------------------------------------------------
# Realistic-id integration: dedup across hex chunk ids
# ---------------------------------------------------------------------------


class TestRealisticIdDedup:
    """Dedup must work when chunk_point_id is a UUID and neighbor ids are hex.

    This test reproduces the production dedup bug described in review fix
    round 1: Hit.chunk_point_id holds the Qdrant point UUID, but payload
    prev_chunk_id / next_chunk_id hold 64-char hex chunk ids.  Before the fix,
    included_ids was keyed on UUIDs while neighbors were keyed on hex ids, so
    the sets never intersected and a neighbor that was also a primary hit
    appeared twice.

    The fix normalises to hex via payload["chunk_hash"].  This test MUST FAIL
    against the pre-fix code (where chunk_id = hit.chunk_point_id).
    """

    def test_neighbor_that_is_also_primary_hit_appears_exactly_once(self) -> None:
        """A neighbor whose hex id is also a primary hit must not be duplicated.

        Scenario:
        - hex_A and hex_B are two 64-char hex chunk ids.
        - Both are primary hits; their chunk_point_ids are UUIDs derived from hex.
        - hex_A's payload has next_chunk_id = hex_B (adjacent in document).
        - Without the fix, assemble_evidence would include hex_B twice:
          once as a primary hit and once as a neighbor of hex_A.
        - With the fix (canonical key = chunk_hash), hex_B is in included_ids
          from the primary pass and is correctly skipped during expansion.
        """
        from beacon.storage.payload import chunk_id_to_point_id

        # 64-char hex chunk ids (realistic format).
        hex_a = "a" * 64
        hex_b = "b" * 64

        # Qdrant point ids derived from hex ids (production mapping).
        point_id_a = chunk_id_to_point_id(hex_a)
        point_id_b = chunk_id_to_point_id(hex_b)

        # fetch_chunk is keyed on hex ids (the seam contract).
        payloads_by_hex: dict[str, dict[str, Any]] = {
            hex_a: _make_payload(
                hex_a,
                "chunk alpha text",
                next_chunk_id=hex_b,  # hex_b is the adjacent chunk
            ),
            hex_b: _make_payload(
                hex_b,
                "chunk beta text",
                prev_chunk_id=hex_a,
            ),
        }

        from beacon.retrieval.hybrid import Hit

        # Both chunks are primary hits; point_id is the Qdrant UUID.
        hits = [
            Hit(chunk_point_id=point_id_a, payload=payloads_by_hex[hex_a], fused_score=2.0),
            Hit(chunk_point_id=point_id_b, payload=payloads_by_hex[hex_b], fused_score=1.0),
        ]

        bundle = assemble_evidence(
            hits=hits,
            query_text="alpha beta",
            fetch_chunk=_fake_chunk_fetch(payloads_by_hex),
            token_budget=5000,
            max_neighbor_hops=1,
            max_context_per_hit=2,
        )

        chunk_ids = [e.chunk_id for e in bundle.evidence]

        # Each chunk must appear exactly once.
        assert len(chunk_ids) == len(set(chunk_ids)), (
            f"Duplicate chunk_ids in bundle: {chunk_ids!r}"
        )
        assert set(chunk_ids) == {hex_a, hex_b}, (
            f"Expected exactly {{hex_a, hex_b}}, got {set(chunk_ids)!r}"
        )
