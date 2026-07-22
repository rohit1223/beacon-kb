"""Unit tests for generation.citations - structural citation resolution and validation."""

from __future__ import annotations

import pytest

from beacon_kb.errors import CitationError
from beacon_kb.generation.citations import (
    build_evidence_index,
    extract_cited_labels,
    resolve_citations,
    validate_no_unknown_evidence_ids,
)
from beacon_kb.models import (
    Chunk,
    ChunkId,
    Citation,
    Evidence,
    EvidenceId,
    EvidenceRole,
    Hit,
    RevisionId,
    SectionId,
    Snippet,
    SourceId,
    make_evidence_id,
)


def _make_chunk(chunk_id: str = "c1", text: str = "chunk content") -> Chunk:
    return Chunk(
        id=ChunkId(chunk_id),
        source_id=SourceId("source-1"),
        revision_id=RevisionId("rev-1"),
        section_id=SectionId("sec-1"),
        text=text,
        ordinal=0,
        parent_locator="intro",
    )


def _make_evidence(
    chunk_id: str = "c1",
    citation_label: str = "S1",
    query_id: str = "q1",
    snippet: Snippet | None = None,
) -> Evidence:
    chunk = _make_chunk(chunk_id)
    hit = Hit(chunk=chunk, sparse_score=1.0)
    eid = make_evidence_id(query_id=query_id, chunk_id=chunk_id)
    return Evidence(
        id=eid,
        hit=hit,
        citation_label=citation_label,
        role=EvidenceRole.HIT,
        snippet=snippet,
    )


class TestExtractCitedLabels:
    def test_no_labels_empty_text(self) -> None:
        assert extract_cited_labels("") == []

    def test_no_labels_plain_text(self) -> None:
        assert extract_cited_labels("The quick brown fox.") == []

    def test_single_label(self) -> None:
        assert extract_cited_labels("The sky is blue [S1].") == ["S1"]

    def test_multiple_labels(self) -> None:
        result = extract_cited_labels("First [S1] and second [S2] and third [S3].")
        assert result == ["S1", "S2", "S3"]

    def test_duplicate_labels_deduplicated(self) -> None:
        result = extract_cited_labels("See [S1] and also [S1] again.")
        assert result == ["S1"]

    def test_order_of_first_occurrence_preserved(self) -> None:
        result = extract_cited_labels("[S3] then [S1] then [S2] then [S3] again.")
        assert result == ["S3", "S1", "S2"]

    def test_multi_digit_labels(self) -> None:
        result = extract_cited_labels("Sources [S10] and [S123] confirm this.")
        assert result == ["S10", "S123"]

    def test_no_false_positives(self) -> None:
        """Patterns that should NOT match."""
        assert extract_cited_labels("[s1] lowercase") == []
        assert extract_cited_labels("[A1] not S prefix") == []
        assert extract_cited_labels("S1 without brackets") == []


class TestBuildEvidenceIndex:
    def test_empty_evidence(self) -> None:
        assert build_evidence_index(()) == {}

    def test_single_item(self) -> None:
        ev = _make_evidence("c1", "S1")
        index = build_evidence_index((ev,))
        assert "S1" in index
        assert index["S1"] is ev

    def test_multiple_items(self) -> None:
        ev1 = _make_evidence("c1", "S1")
        ev2 = _make_evidence("c2", "S2")
        ev3 = _make_evidence("c3", "S3")
        index = build_evidence_index((ev1, ev2, ev3))
        assert set(index.keys()) == {"S1", "S2", "S3"}


class TestResolveCitations:
    def test_no_citations_in_answer(self) -> None:
        ev = _make_evidence("c1", "S1")
        citations = resolve_citations("No inline references here.", (ev,))
        assert citations == ()

    def test_single_citation_resolved(self) -> None:
        ev = _make_evidence("c1", "S1")
        citations = resolve_citations("The answer is true [S1].", (ev,))
        assert len(citations) == 1
        cit = citations[0]
        assert isinstance(cit, Citation)
        assert cit.label == "S1"
        assert str(cit.chunk_id) == "c1"

    def test_multiple_citations_resolved(self) -> None:
        ev1 = _make_evidence("c1", "S1")
        ev2 = _make_evidence("c2", "S2")
        citations = resolve_citations("First [S1] and second [S2].", (ev1, ev2))
        assert len(citations) == 2
        labels = [c.label for c in citations]
        assert labels == ["S1", "S2"]

    def test_unknown_label_raises_citation_error(self) -> None:
        ev = _make_evidence("c1", "S1")
        with pytest.raises(CitationError) as exc_info:
            resolve_citations("See [S99] for details.", (ev,))
        assert "S99" in str(exc_info.value)
        assert "S1" in str(exc_info.value)  # available label shown in error

    def test_citation_error_is_typed(self) -> None:
        """CitationError must not be caught as a generic Exception in tests."""
        from beacon_kb.errors import CitationError
        ev = _make_evidence("c1", "S1")
        with pytest.raises(CitationError):
            resolve_citations("[S99]", (ev,))

    def test_snippet_uri_used_for_canonical_uri(self) -> None:
        snip = Snippet(
            text="chunk text",
            source_id="source-1",
            source_uri="https://example.com/doc",
            title="Doc Title",
            locator="intro",
            char_start=0,
            char_end=10,
            chunk_id="c1",
        )
        ev = _make_evidence("c1", "S1", snippet=snip)
        citations = resolve_citations("Evidence here [S1].", (ev,))
        assert citations[0].canonical_uri == "https://example.com/doc"

    def test_no_snippet_falls_back_to_source_id(self) -> None:
        ev = _make_evidence("c1", "S1")  # no snippet
        citations = resolve_citations("Evidence here [S1].", (ev,))
        assert citations[0].canonical_uri  # non-empty

    def test_duplicate_label_in_text_produces_one_citation(self) -> None:
        ev = _make_evidence("c1", "S1")
        citations = resolve_citations("[S1] and also [S1].", (ev,))
        assert len(citations) == 1

    def test_excerpt_truncated_to_200_chars(self) -> None:
        long_text = "x" * 300
        chunk = _make_chunk("c1", long_text)
        hit = Hit(chunk=chunk)
        eid = make_evidence_id(query_id="q1", chunk_id="c1")
        ev = Evidence(id=eid, hit=hit, citation_label="S1", role=EvidenceRole.HIT)
        citations = resolve_citations("[S1]", (ev,))
        assert len(citations[0].excerpt) <= 200


class TestValidateNoUnknownEvidenceIds:
    def test_all_known_ids_passes(self) -> None:
        eid1 = EvidenceId("e1")
        eid2 = EvidenceId("e2")
        validate_no_unknown_evidence_ids([eid1, eid2], {eid1, eid2})  # must not raise

    def test_unknown_id_raises_citation_error(self) -> None:
        eid_known = EvidenceId("e1")
        eid_unknown = EvidenceId("e-unknown")
        with pytest.raises(CitationError) as exc_info:
            validate_no_unknown_evidence_ids([eid_unknown], {eid_known})
        assert "e-unknown" in str(exc_info.value)

    def test_empty_cited_ids_passes(self) -> None:
        validate_no_unknown_evidence_ids([], {EvidenceId("e1")})  # must not raise

    def test_empty_available_ids_with_citation_raises(self) -> None:
        with pytest.raises(CitationError):
            validate_no_unknown_evidence_ids([EvidenceId("e1")], set())
