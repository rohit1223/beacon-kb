"""Tests for Evidence.text: full chunk text reaches the prompt, not the snippet cap.

Branch-review fix 1: assemble_evidence budgets the full chunk_text but the
answer prompt was built from the 400-char snippet.  Evidence now carries the
full chunk text in ``text``; ``_build_messages`` builds the context block from
``ev.text``; ``BudgetRecap.tokens_packed`` therefore describes what the model
sees.
"""

from __future__ import annotations

import math
from typing import Any

from beacon.answer.generate import _build_messages
from beacon.models import (
    BudgetRecap,
    Evidence,
    EvidenceBundle,
    EvidenceRole,
    Snippet,
)
from beacon.retrieval.evidence import assemble_evidence
from beacon.retrieval.hybrid import Hit


def _make_payload(
    chunk_id: str,
    text: str,
    prev_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_text": text,
        "source_uri": "file:///test.md",
        "title": "Test",
        "heading_path": ["Section"],
        "tags": [],
        "kind": "child",
        "section_kind": "text",
        "parent_chunk_id": None,
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "ingested_at": "2025-01-01T00:00:00Z",
        "content_hash": "abc",
        "chunk_hash": chunk_id,
        "fingerprint": "fp001",
    }


# A chunk text clearly longer than the 400-char snippet cap, with a
# distinctive tail sentence positioned beyond the cap.
LONG_TEXT = (
    "This is the beginning of a long document section about widgets. "
    "It contains many sentences to ensure it exceeds the 400-character snippet window. "
    "The widget configuration panel allows you to set alignment, color, and size options. "
    "Adjustments are saved automatically when you close the panel. "
    "Advanced users can export the widget layout as a JSON preset for reuse. "
    "This distinctive tail sentence should only appear in ev.text, never in the snippet: "
    "Sentinel-Tail-XYZ-9999 is the key identifying phrase here."
)


def test_long_text_invariant() -> None:
    """Test invariant: LONG_TEXT must exceed the 400-char snippet cap."""
    assert len(LONG_TEXT) > 400


class TestEvidenceTextField:
    def test_ev_text_equals_full_chunk_text(self) -> None:
        """Evidence.text must equal the full chunk_text from the payload."""
        payload = _make_payload("c1", LONG_TEXT)
        hits = [Hit(chunk_point_id="c1", payload=payload, fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="widget",
            fetch_chunk=lambda _: None,
            token_budget=50000,
        )
        assert bundle.evidence, "Expected at least one evidence item"
        ev = bundle.evidence[0]
        assert ev.text == LONG_TEXT, (
            f"ev.text must be the full chunk_text; got length {len(ev.text)}"
        )

    def test_ev_text_longer_than_snippet_cap(self) -> None:
        """ev.text is the full chunk; ev.snippet.text is the capped display excerpt."""
        payload = _make_payload("c1", LONG_TEXT)
        hits = [Hit(chunk_point_id="c1", payload=payload, fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="widget",
            fetch_chunk=lambda _: None,
            token_budget=50000,
        )
        ev = bundle.evidence[0]
        assert ev.snippet is not None
        assert len(ev.text) > len(ev.snippet.text), (
            "ev.text must be longer than ev.snippet.text when the chunk "
            "exceeds the 400-char snippet cap"
        )

    def test_ev_text_contains_distinctive_tail(self) -> None:
        """The distinctive tail beyond 400 chars appears in ev.text."""
        payload = _make_payload("c1", LONG_TEXT)
        hits = [Hit(chunk_point_id="c1", payload=payload, fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="widget",
            fetch_chunk=lambda _: None,
            token_budget=50000,
        )
        ev = bundle.evidence[0]
        assert "Sentinel-Tail-XYZ-9999" in ev.text

    def test_context_span_text_also_set(self) -> None:
        """CONTEXT spans also get ev.text from their payload chunk_text."""
        neighbor_text = "Neighbor chunk " * 10 + " Neighbor-Sentinel-ABC"
        payloads = {
            "primary": _make_payload("primary", "main content", next_chunk_id="neighbor"),
            "neighbor": _make_payload("neighbor", neighbor_text, prev_chunk_id="primary"),
        }
        hits = [Hit(chunk_point_id="primary", payload=payloads["primary"], fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="main",
            fetch_chunk=lambda cid: payloads.get(cid),
            token_budget=50000,
            max_neighbor_hops=1,
            max_context_per_hit=1,
        )
        ctx_items = [e for e in bundle.evidence if e.role is EvidenceRole.CONTEXT]
        assert ctx_items, "Expected at least one CONTEXT span"
        assert ctx_items[0].text == neighbor_text


class TestRecapVsPromptConsistency:
    def test_recap_tokens_packed_reflects_full_text(self) -> None:
        """BudgetRecap.tokens_packed is based on full chunk_text, matching ev.text."""
        payload = _make_payload("c1", LONG_TEXT)
        hits = [Hit(chunk_point_id="c1", payload=payload, fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="widget",
            fetch_chunk=lambda _: None,
            token_budget=50000,
        )
        expected_tokens = math.ceil(len(LONG_TEXT) / 4)
        assert bundle.recap.tokens_packed == expected_tokens, (
            f"tokens_packed {bundle.recap.tokens_packed} must equal the full-text "
            f"token count {expected_tokens}"
        )
        # Consistency: the recap accounts exactly the text the prompt will carry.
        prompt_token_estimate = sum(
            math.ceil(len(ev.text) / 4) for ev in bundle.evidence
        )
        assert bundle.recap.tokens_packed == prompt_token_estimate, (
            "tokens_packed must describe the ev.text content the model sees"
        )


class TestBuildMessagesUsesFullText:
    """_build_messages must use ev.text (full chunk), not ev.snippet.text (cap)."""

    def test_full_text_reaches_prompt_not_snippet(self) -> None:
        """The distinctive tail phrase beyond 400 chars must appear in the prompt."""
        full_text = (
            "Widget configuration: open the settings panel. "
            "Alignment options control the layout. "
            "Additional context follows: " + "padding " * 60
            + " Sentinel-Prompt-ABCDE-9999 distinctive tail phrase."
        )
        assert len(full_text) > 400

        snippet = Snippet(
            text=full_text[:380],  # truncated - does NOT contain the tail
            source_uri="file:///test.md",
            title="Test",
            heading_path=[],
            locator="",
            chunk_id="c1",
            char_start=0,
            char_end=380,
        )
        ev = Evidence(
            chunk_id="c1",
            label="S1",
            role=EvidenceRole.HIT,
            score=0.9,
            context_of=None,
            snippet=snippet,
            text=full_text,
        )
        recap = BudgetRecap(
            requested=1,
            packed=1,
            skipped=0,
            tokens_packed=math.ceil(len(full_text) / 4),
            token_budget=50000,
        )
        bundle = EvidenceBundle(evidence=[ev], recap=recap)

        messages = _build_messages(bundle, "widget configuration")

        all_content = " ".join(m["content"] for m in messages)
        tail_phrase = "Sentinel-Prompt-ABCDE-9999"
        assert tail_phrase in all_content, (
            f"Distinctive tail phrase {tail_phrase!r} must appear in the built "
            f"prompt: _build_messages must use ev.text, not ev.snippet.text."
        )
        # The truncated snippet does not carry the tail - proving the prompt
        # content came from ev.text.
        assert tail_phrase not in snippet.text

    def test_assembled_bundle_end_to_end_prompt_carries_tail(self) -> None:
        """From raw hits through assembly to prompt: the full text survives."""
        payload = _make_payload("c1", LONG_TEXT)
        hits = [Hit(chunk_point_id="c1", payload=payload, fused_score=1.0)]
        bundle = assemble_evidence(
            hits=hits,
            query_text="widget configuration",
            fetch_chunk=lambda _: None,
            token_budget=50000,
        )
        messages = _build_messages(bundle, "widget configuration")
        all_content = " ".join(m["content"] for m in messages)
        assert "Sentinel-Tail-XYZ-9999" in all_content
