"""Unit tests for generation.answer - grounded answer orchestration."""

from __future__ import annotations

import pytest

from beacon_kb.config import AnswerConfig
from beacon_kb.errors import CitationError
from beacon_kb.generation.answer import AnswerDiagnostics, run_answer
from beacon_kb.generation.prompts import PROMPT_VERSION
from beacon_kb.models import (
    AnswerResponse,
    Chunk,
    ChunkId,
    Evidence,
    EvidenceRole,
    Hit,
    Query,
    QueryId,
    RevisionId,
    SectionId,
    SourceId,
    make_evidence_id,
)
from beacon_kb.retrieval.query import QueryVariants
from beacon_kb.testing import CountingGenerator, FakeGenerator, FakeProgressObserver


def _make_chunk(chunk_id: str = "c1", text: str = "content") -> Chunk:
    return Chunk(
        id=ChunkId(chunk_id),
        source_id=SourceId("s"),
        revision_id=RevisionId("r"),
        section_id=SectionId("sec"),
        text=text,
        ordinal=0,
        parent_locator="",
    )


def _make_hit(chunk_id: str = "c1", fusion_score: float = 0.8) -> Hit:
    return Hit(chunk=_make_chunk(chunk_id), fusion_score=fusion_score)


class AnswerWithCitationsGenerator:
    """Generator stub that returns an answer with inline [S1] citations."""

    def generate(
        self,
        query: Query,
        hits: list[Hit],
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> AnswerResponse:
        if not hits:
            return AnswerResponse(
                query_id=query.id,
                answer_text="",
                evidence=(),
                abstained=True,
                input_tokens=0,
                output_tokens=0,
            )
        eid = make_evidence_id(query_id=str(query.id), chunk_id=str(hits[0].chunk.id))
        ev = Evidence(
            id=eid,
            hit=hits[0],
            citation_label="S1",
            role=EvidenceRole.HIT,
        )
        return AnswerResponse(
            query_id=query.id,
            answer_text="The answer is documented in [S1].",
            evidence=(ev,),
            abstained=False,
            input_tokens=10,
            output_tokens=20,
        )


class UngroundedCitationGenerator:
    """Generator stub that returns answer_text referencing a non-existent label."""

    def generate(
        self,
        query: Query,
        hits: list[Hit],
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> AnswerResponse:
        eid = make_evidence_id(query_id=str(query.id), chunk_id=str(hits[0].chunk.id))
        ev = Evidence(
            id=eid,
            hit=hits[0],
            citation_label="S1",
            role=EvidenceRole.HIT,
        )
        return AnswerResponse(
            query_id=query.id,
            # [S99] is NOT in evidence - this must be rejected.
            answer_text="See [S99] for details.",
            evidence=(ev,),
            abstained=False,
            input_tokens=10,
            output_tokens=20,
        )


class AbstainSentinelGenerator:
    """Generator stub that returns the ABSTAIN sentinel string."""

    def generate(
        self,
        query: Query,
        hits: list[Hit],
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> AnswerResponse:
        eid = make_evidence_id(query_id=str(query.id), chunk_id=str(hits[0].chunk.id))
        ev = Evidence(
            id=eid,
            hit=hits[0],
            citation_label="S1",
            role=EvidenceRole.HIT,
        )
        return AnswerResponse(
            query_id=query.id,
            answer_text="ABSTAIN",
            evidence=(ev,),
            abstained=False,  # generator set False but text says ABSTAIN
            input_tokens=5,
            output_tokens=1,
        )


class TestPreAbstention:
    def test_empty_hits_abstains_without_generator_call(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        counting = CountingGenerator(FakeGenerator())
        resp, _diag = run_answer(query, counting, hits=[])
        assert resp.abstained is True
        assert resp.answer_text == ""
        assert counting.call_count == 0  # ZERO generator calls

    def test_empty_hits_returns_answer_response(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        resp, _diag = run_answer(query, FakeGenerator(), hits=[])
        assert isinstance(resp, AnswerResponse)
        assert resp.query_id == query.id

    def test_below_threshold_abstains_without_generator_call(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1", fusion_score=0.1)]
        config = AnswerConfig(abstain_threshold=0.5)
        counting = CountingGenerator(FakeGenerator())
        resp, _diag = run_answer(query, counting, hits=hits, config=config)
        assert resp.abstained is True
        assert counting.call_count == 0  # ZERO generator calls


class TestExactlyOneGeneratorCall:
    def test_normal_answer_calls_generator_exactly_once(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1", fusion_score=0.9)]
        counting = CountingGenerator(AnswerWithCitationsGenerator())
        run_answer(query, counting, hits=hits)
        assert counting.call_count == 1

    def test_post_abstain_still_exactly_one_call(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1", fusion_score=0.9)]
        counting = CountingGenerator(AbstainSentinelGenerator())
        run_answer(query, counting, hits=hits)
        assert counting.call_count == 1


class TestPostAbstention:
    def test_abstain_sentinel_triggers_abstention(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1", fusion_score=0.9)]
        resp, _diag = run_answer(query, AbstainSentinelGenerator(), hits=hits)
        assert resp.abstained is True
        assert resp.answer_text == ""

    def test_generator_abstained_flag_respected(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        gen = FakeGenerator(abstain=True)
        resp, _diag = run_answer(query, gen, hits=hits)
        assert resp.abstained is True
        assert resp.answer_text == ""


class TestCitationValidation:
    def test_valid_citations_resolved(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        resp, _diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert resp.abstained is False
        assert len(resp.citations) == 1
        assert resp.citations[0].label == "S1"

    def test_unknown_citation_raises_citation_error(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        with pytest.raises(CitationError):
            run_answer(query, UngroundedCitationGenerator(), hits=hits)

    def test_evidence_preserved_in_response(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        resp, _diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert len(resp.evidence) == 1


class TestDiagnostics:
    def test_diagnostics_returned(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _resp, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert isinstance(diag, AnswerDiagnostics)

    def test_diagnostics_prompt_version(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.prompt_version == PROMPT_VERSION

    def test_diagnostics_provider_type(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        gen = AnswerWithCitationsGenerator()
        _, diag = run_answer(query, gen, hits=hits)
        assert diag.provider_type == "AnswerWithCitationsGenerator"

    def test_diagnostics_no_secrets(self) -> None:
        """Diagnostics must NOT contain API keys or credentials."""
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        # Spot-check: none of the known secret patterns appear.
        diag_str = str(diag)
        assert "sk-" not in diag_str
        assert "password" not in diag_str.lower()
        assert "api_key" not in diag_str.lower()

    def test_diagnostics_token_counts(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.input_tokens >= 0
        assert diag.output_tokens >= 0

    def test_diagnostics_abstained_pre(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        _, diag = run_answer(query, FakeGenerator(), hits=[])
        assert diag.abstained is True

    def test_diagnostics_abstained_false_for_good_answer(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.abstained is False

    def test_diagnostics_elapsed_times_non_negative(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.elapsed_generation_s >= 0.0

    def test_diagnostics_elapsed_retrieval_passthrough(self) -> None:
        """elapsed_retrieval_s supplied by the caller is recorded verbatim."""
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(
            query, AnswerWithCitationsGenerator(), hits=hits, elapsed_retrieval_s=1.25
        )
        assert diag.elapsed_retrieval_s == 1.25

    def test_diagnostics_elapsed_retrieval_defaults_to_zero(self) -> None:
        """0.0 means 'not measured': run_answer() cannot time retrieval itself."""
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.elapsed_retrieval_s == 0.0

    def test_diagnostics_query_variants_recorded(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        variants = QueryVariants(
            original_text="what?", sparse_text="what sparse", dense_text="what dense"
        )
        _, diag = run_answer(
            query, AnswerWithCitationsGenerator(), hits=hits, query_variants=variants
        )
        assert diag.query_variants == (
            ("original", "what?"),
            ("sparse", "what sparse"),
            ("dense", "what dense"),
        )

    def test_diagnostics_query_variants_default_empty(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        _, diag = run_answer(query, AnswerWithCitationsGenerator(), hits=hits)
        assert diag.query_variants == ()

    def test_diagnostics_variants_and_timing_on_pre_abstain(self) -> None:
        """The pre-abstained diagnostics site records variants and retrieval time."""
        query = Query(id=QueryId("q1"), text="what?")
        variants = QueryVariants(
            original_text="what?", sparse_text="what?", dense_text="what?"
        )
        _, diag = run_answer(
            query,
            FakeGenerator(),
            hits=[],
            query_variants=variants,
            elapsed_retrieval_s=0.5,
        )
        assert diag.abstained is True
        assert diag.elapsed_retrieval_s == 0.5
        assert diag.query_variants == (
            ("original", "what?"),
            ("sparse", "what?"),
            ("dense", "what?"),
        )

    def test_diagnostics_variants_and_timing_on_post_abstain(self) -> None:
        """The post-abstained diagnostics site records variants and retrieval time."""
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1", fusion_score=0.9)]
        variants = QueryVariants(
            original_text="what?", sparse_text="what?", dense_text="what?"
        )
        _, diag = run_answer(
            query,
            AbstainSentinelGenerator(),
            hits=hits,
            query_variants=variants,
            elapsed_retrieval_s=0.25,
        )
        assert diag.abstained is True
        assert diag.elapsed_retrieval_s == 0.25
        assert diag.query_variants[0] == ("original", "what?")


class TestObserverEvents:
    def test_observer_receives_done_event(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        observer = FakeProgressObserver()
        run_answer(query, AnswerWithCitationsGenerator(), hits=hits, observer=observer)
        stages = [e.get("stage") for e in observer.events]
        assert "answer" in stages

    def test_observer_receives_abstained_event_on_pre_abstain(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        observer = FakeProgressObserver()
        run_answer(query, FakeGenerator(), hits=[], observer=observer)
        statuses = [e.get("status") for e in observer.events]
        assert "abstained" in statuses

    def test_none_observer_does_not_raise(self) -> None:
        query = Query(id=QueryId("q1"), text="what?")
        hits = [_make_hit("c1")]
        # Must not raise when observer is None.
        run_answer(query, AnswerWithCitationsGenerator(), hits=hits, observer=None)


class TestNoWebSearchFlag:
    def test_generator_protocol_has_no_web_search_param(self) -> None:
        """The Generator.generate() signature must NOT include a web_search parameter.

        This test enforces the binding hard constraint from the plan: no hidden
        web-search flag on the Generator protocol.
        """
        import inspect

        from beacon_kb.protocols import Generator
        sig = inspect.signature(Generator.generate)
        param_names = list(sig.parameters.keys())
        assert "web_search" not in param_names, (
            "Generator.generate() must NOT have a 'web_search' parameter.  "
            "Silent web retrieval is forbidden by the Generator protocol contract."
        )


class TestI2PostAbstentionSubsetCheck:
    """I2 fix: post-abstention path must also reject fabricated evidence."""

    def test_abstaining_hostile_generator_with_fabricated_evidence_is_rejected(
        self,
    ) -> None:
        """A generator that returns abstained=True with fabricated chunk IDs must raise
        CitationError, not silently pass through to the caller.

        This tests the post-abstention Stage 4 subset check added in the I2 fix.
        Without the fix, the fabricated evidence would escape validation whenever
        the generator triggers post-abstention.
        """
        fabricated_chunk = _make_chunk(chunk_id="fabricated-id-not-in-canonical")
        fabricated_ev = Evidence(
            id=make_evidence_id(query_id="q1", chunk_id="fabricated-id-not-in-canonical"),
            hit=Hit(chunk=fabricated_chunk, fusion_score=0.9),
            citation_label="S1",
            role=EvidenceRole.HIT,
        )

        class HostileAbstainGenerator:
            """Returns abstained=True with a fabricated evidence chunk ID."""

            def generate(
                self,
                query: Query,
                hits: list[Hit],
                *,
                max_input_tokens: int = 4096,
                max_output_tokens: int = 512,
            ) -> AnswerResponse:
                return AnswerResponse(
                    query_id=query.id,
                    answer_text="",
                    evidence=(fabricated_ev,),
                    abstained=True,
                    input_tokens=0,
                    output_tokens=0,
                )

        # Canonical evidence uses a REAL chunk ("c1"), not the fabricated one.
        real_chunk = _make_chunk(chunk_id="c1")
        real_ev = Evidence(
            id=make_evidence_id(query_id="q1", chunk_id="c1"),
            hit=Hit(chunk=real_chunk, fusion_score=0.9),
            citation_label="S1",
            role=EvidenceRole.HIT,
        )

        query = Query(id=QueryId("q1"), text="what?")
        hits = [Hit(chunk=real_chunk, fusion_score=0.9)]

        # run_answer with canonical evidence=[real_ev] must detect the fabrication.
        with pytest.raises(CitationError):
            run_answer(
                query,
                HostileAbstainGenerator(),
                hits=hits,
                evidence=[real_ev],
            )
