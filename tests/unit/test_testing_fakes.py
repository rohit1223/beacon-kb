"""Unit tests for beacon_kb.testing fakes and contract suites.

Tests verify:
- isinstance compatibility with runtime_checkable protocols
- Determinism under fixed seed
- EmbedderFake: batch_size param, unit-normalized vectors, correct dimension
- FakeGenerator: deterministic text and evidence
- FakeClock: controllable time, tick()
- FakeFailingEmbedder: raises BackendError
- Contract suites: importable and runnable against their fakes
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from beacon_kb.errors import BackendError, IngestionError
from beacon_kb.models import (
    AnswerResponse,
    Chunk,
    ChunkId,
    CorpusId,
    Evidence,
    Hit,
    Query,
    QueryId,
    RawDocument,
    RevisionId,
    SectionId,
    SourceId,
    make_evidence_id,
)
from beacon_kb.protocols import (
    Connector,
    CorpusRouter,
    DenseRetriever,
    Embedder,
    EvidenceGrader,
    Fusion,
    Generator,
    ProgressObserver,
    QueryPlanner,
    Reranker,
    SessionStore,
    SparseRetriever,
    StopCondition,
    TokenCounter,
    Tool,
)
from beacon_kb.testing import (
    ConnectorContract,
    CorpusRouterContract,
    DenseRetrieverContract,
    EmbedderContract,
    EvidenceGraderContract,
    FakeClock,
    FakeConnector,
    FakeCorpusRouter,
    FakeDenseRetriever,
    FakeEmbedder,
    FakeEvidenceGrader,
    FakeFailingEmbedder,
    FakeFusion,
    FakeGenerator,
    FakeProgressObserver,
    FakeQueryPlanner,
    FakeReranker,
    FakeSessionStore,
    FakeSparseRetriever,
    FakeStopCondition,
    FakeTokenCounter,
    FakeTool,
    FusionContract,
    GeneratorContract,
    ProgressObserverContract,
    QueryPlannerContract,
    RerankerContract,
    SessionStoreContract,
    SparseRetrieverContract,
    StopConditionContract,
    TokenCounterContract,
    ToolContract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str = "c1", text: str = "some text") -> Chunk:
    return Chunk(
        id=ChunkId(chunk_id),
        source_id=SourceId("fake://source"),
        revision_id=RevisionId("rev-1"),
        section_id=SectionId("sec-1"),
        text=text,
        ordinal=0,
        parent_locator="",
    )


def _make_hit(chunk_id: str = "c1") -> Hit:
    return Hit(chunk=_make_chunk(chunk_id))


def _make_query(text: str = "test query") -> Query:
    return Query(id=QueryId("q1"), text=text)


def _make_evidence(n: int = 2) -> list[Evidence]:
    return [
        Evidence(
            id=make_evidence_id(query_id="q1", chunk_id=f"c{i}"),
            hit=_make_hit(f"c{i}"),
            citation_label=f"S{i + 1}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# FakeClock
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeClock:
    def test_initial_time(self):
        clock = FakeClock(start=5.0)
        assert clock.now() == 5.0

    def test_default_start_is_zero(self):
        clock = FakeClock()
        assert clock.now() == 0.0

    def test_tick_increments(self):
        clock = FakeClock()
        clock.tick(2.5)
        assert clock.now() == 2.5

    def test_tick_default_delta(self):
        clock = FakeClock()
        clock.tick()
        assert clock.now() == 1.0

    def test_tick_accumulates(self):
        clock = FakeClock()
        clock.tick(1.0)
        clock.tick(1.0)
        assert clock.now() == 2.0

    def test_advance_to(self):
        clock = FakeClock(start=1.0)
        clock.advance_to(100.0)
        assert clock.now() == 100.0


# ---------------------------------------------------------------------------
# FakeConnector
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeConnector:
    def test_is_connector(self):
        assert isinstance(FakeConnector(), Connector)

    def test_list_sources_returns_strings(self):
        fc = FakeConnector()
        sources = fc.list_sources()
        assert isinstance(sources, list)
        assert all(isinstance(s, str) for s in sources)

    def test_list_sources_deterministic(self):
        fc = FakeConnector()
        assert fc.list_sources() == fc.list_sources()

    def test_list_sources_sorted(self):
        fc = FakeConnector()
        sources = fc.list_sources()
        assert sources == sorted(sources)

    def test_fetch_known_uri(self):
        fc = FakeConnector()
        uri = fc.list_sources()[0]
        doc = fc.fetch(uri)
        assert isinstance(doc, RawDocument)
        assert str(doc.source_id) == uri

    def test_fetch_unknown_uri_raises_ingestion_error(self):
        fc = FakeConnector()
        with pytest.raises(IngestionError):
            fc.fetch("unknown://nonexistent")

    def test_custom_sources(self):
        custom = {"mem://a": "hello", "mem://b": "world"}
        fc = FakeConnector(sources=custom)
        assert set(fc.list_sources()) == {"mem://a", "mem://b"}

    def test_fetch_content_matches(self):
        custom = {"mem://x": "custom content here"}
        fc = FakeConnector(sources=custom)
        doc = fc.fetch("mem://x")
        assert doc.content == "custom content here"


# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeEmbedder:
    def test_is_embedder(self):
        assert isinstance(FakeEmbedder(), Embedder)

    def test_dimension_matches_param(self):
        fe = FakeEmbedder(dim=32)
        assert fe.dimension() == 32

    def test_default_dimension(self):
        fe = FakeEmbedder()
        assert fe.dimension() == 16

    def test_embed_returns_correct_count(self):
        fe = FakeEmbedder()
        texts = ["a", "b", "c"]
        vecs = fe.embed(texts)
        assert len(vecs) == 3

    def test_embed_returns_correct_dim(self):
        fe = FakeEmbedder(dim=8)
        vecs = fe.embed(["hello"])
        assert len(vecs[0]) == 8

    def test_embed_unit_normalized(self):
        fe = FakeEmbedder(dim=16)
        texts = ["hello", "world", "test embedding normalization"]
        for vec in fe.embed(texts):
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-6, f"Vector not normalized: norm={norm}"

    def test_determinism_same_seed(self):
        fe = FakeEmbedder(seed=99)
        texts = ["alpha", "beta"]
        assert fe.embed(texts) == fe.embed(texts)

    def test_different_texts_different_vectors(self):
        fe = FakeEmbedder()
        v1 = fe.embed(["text one"])[0]
        v2 = fe.embed(["text two"])[0]
        assert v1 != v2

    def test_batch_size_attribute(self):
        fe = FakeEmbedder(batch_size=32)
        assert fe.batch_size == 32

    def test_default_batch_size(self):
        fe = FakeEmbedder()
        assert fe.batch_size == 8

    def test_embed_more_than_batch_size(self):
        fe = FakeEmbedder(batch_size=4)
        texts = [f"text-{i}" for i in range(10)]
        vecs = fe.embed(texts)
        assert len(vecs) == 10

    def test_embed_empty_list(self):
        fe = FakeEmbedder()
        assert fe.embed([]) == []

    def test_different_seeds_different_vectors(self):
        fe1 = FakeEmbedder(seed=1)
        fe2 = FakeEmbedder(seed=2)
        v1 = fe1.embed(["same text"])[0]
        v2 = fe2.embed(["same text"])[0]
        assert v1 != v2


# ---------------------------------------------------------------------------
# Cross-process hash-salt independence regression tests
# ---------------------------------------------------------------------------

# A small Python snippet run in a subprocess that prints the first element of
# the embedding vector for a fixed text and seed.  We run it twice under
# different PYTHONHASHSEED values and assert both outputs are identical.

_EMBED_SCRIPT = """
import sys
sys.path.insert(0, {src_path!r})
from beacon_kb.testing import FakeEmbedder
fe = FakeEmbedder(seed=42)
vec = fe.embed(["hello world"])[0]
print(repr(vec[0]))
"""

_SPARSE_SCRIPT = """
import sys
sys.path.insert(0, {src_path!r})
from beacon_kb.testing import FakeSparseRetriever
from beacon_kb.models import Chunk, ChunkId, SourceId, RevisionId, SectionId, Query, QueryId
chunks = [
    Chunk(
        id=ChunkId("c1"), source_id=SourceId("s"), revision_id=RevisionId("r"),
        section_id=SectionId("sec"), text="text", ordinal=0, parent_locator="",
    )
]
sr = FakeSparseRetriever(chunks=chunks, seed=42)
hits = sr.retrieve(Query(id=QueryId("q1"), text="test"))
print(repr(hits[0].sparse_score))
"""


def _run_script(script: str, hashseed: str) -> str:
    """Execute *script* in a subprocess with PYTHONHASHSEED=*hashseed*."""
    import os
    from pathlib import Path

    # Resolve src/ two directories above this test file (project root / src).
    src_path = str(Path(__file__).parent.parent.parent / "src")
    code = script.format(src_path=src_path)
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.unit
class TestHashSaltIndependence:
    """Fakes must produce identical outputs regardless of PYTHONHASHSEED."""

    def test_fake_embedder_cross_process_determinism(self):
        out_seed_0 = _run_script(_EMBED_SCRIPT, "0")
        out_seed_1 = _run_script(_EMBED_SCRIPT, "1")
        assert out_seed_0 == out_seed_1, (
            f"FakeEmbedder output differs across PYTHONHASHSEED values: "
            f"HASHSEED=0 -> {out_seed_0!r}, HASHSEED=1 -> {out_seed_1!r}"
        )

    def test_fake_sparse_retriever_cross_process_determinism(self):
        out_seed_0 = _run_script(_SPARSE_SCRIPT, "0")
        out_seed_1 = _run_script(_SPARSE_SCRIPT, "1")
        assert out_seed_0 == out_seed_1, (
            f"FakeSparseRetriever output differs across PYTHONHASHSEED values: "
            f"HASHSEED=0 -> {out_seed_0!r}, HASHSEED=1 -> {out_seed_1!r}"
        )


# ---------------------------------------------------------------------------
# FakeFailingEmbedder
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeFailingEmbedder:
    def test_is_embedder(self):
        assert isinstance(FakeFailingEmbedder(), Embedder)

    def test_embed_raises_backend_error(self):
        fe = FakeFailingEmbedder()
        with pytest.raises(BackendError):
            fe.embed(["anything"])

    def test_custom_message(self):
        fe = FakeFailingEmbedder(message="custom error msg")
        with pytest.raises(BackendError, match="custom error msg"):
            fe.embed(["x"])

    def test_dimension_still_works(self):
        fe = FakeFailingEmbedder(dim=64)
        assert fe.dimension() == 64


# ---------------------------------------------------------------------------
# FakeSparseRetriever
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeSparseRetriever:
    def test_is_sparse_retriever(self):
        assert isinstance(FakeSparseRetriever(), SparseRetriever)

    def test_retrieve_sets_sparse_score(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        sr = FakeSparseRetriever(chunks=chunks)
        hits = sr.retrieve(_make_query())
        assert all(h.sparse_score is not None for h in hits)

    def test_retrieve_other_scores_none(self):
        chunks = [_make_chunk("c1")]
        sr = FakeSparseRetriever(chunks=chunks)
        hits = sr.retrieve(_make_query())
        hit = hits[0]
        assert hit.dense_score is None
        assert hit.fusion_score is None
        assert hit.rerank_score is None

    def test_retrieve_ordered_descending(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        sr = FakeSparseRetriever(chunks=chunks)
        hits = sr.retrieve(_make_query())
        scores = [h.sparse_score for h in hits if h.sparse_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_empty_chunks(self):
        sr = FakeSparseRetriever(chunks=[])
        assert sr.retrieve(_make_query()) == []

    def test_retrieve_deterministic(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        sr = FakeSparseRetriever(chunks=chunks)
        q = _make_query()
        r1 = sr.retrieve(q)
        r2 = sr.retrieve(q)
        assert [h.sparse_score for h in r1] == [h.sparse_score for h in r2]


# ---------------------------------------------------------------------------
# FakeDenseRetriever
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeDenseRetriever:
    def test_is_dense_retriever(self):
        assert isinstance(FakeDenseRetriever(), DenseRetriever)

    def test_retrieve_sets_dense_score(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        dr = FakeDenseRetriever(chunks=chunks)
        hits = dr.retrieve(_make_query())
        assert all(h.dense_score is not None for h in hits)

    def test_retrieve_other_scores_none(self):
        chunks = [_make_chunk("c1")]
        dr = FakeDenseRetriever(chunks=chunks)
        hits = dr.retrieve(_make_query())
        hit = hits[0]
        assert hit.sparse_score is None
        assert hit.fusion_score is None
        assert hit.rerank_score is None

    def test_retrieve_ordered_descending(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        dr = FakeDenseRetriever(chunks=chunks)
        hits = dr.retrieve(_make_query())
        scores = [h.dense_score for h in hits if h.dense_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_deterministic(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        dr = FakeDenseRetriever(chunks=chunks)
        q = _make_query()
        r1 = dr.retrieve(q)
        r2 = dr.retrieve(q)
        assert [h.dense_score for h in r1] == [h.dense_score for h in r2]


# ---------------------------------------------------------------------------
# FakeFusion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeFusion:
    def test_is_fusion(self):
        assert isinstance(FakeFusion(), Fusion)

    def test_fuse_sets_fusion_score(self):
        sparse = [Hit(chunk=_make_chunk("c1"), sparse_score=5.0)]
        dense = [Hit(chunk=_make_chunk("c2"), dense_score=0.9)]
        ff = FakeFusion()
        hits = ff.fuse(sparse, dense)
        assert all(h.fusion_score is not None for h in hits)

    def test_fuse_deduplicates(self):
        chunk = _make_chunk("c1")
        sparse = [Hit(chunk=chunk, sparse_score=5.0)]
        dense = [Hit(chunk=chunk, dense_score=0.9)]
        ff = FakeFusion()
        hits = ff.fuse(sparse, dense)
        assert len(hits) == 1

    def test_fuse_ordered_descending(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        sparse = [Hit(chunk=c, sparse_score=float(i)) for i, c in enumerate(chunks[:2])]
        dense = [Hit(chunk=c, dense_score=float(i) / 10) for i, c in enumerate(chunks[2:])]
        ff = FakeFusion()
        hits = ff.fuse(sparse, dense)
        scores = [h.fusion_score for h in hits if h.fusion_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_fuse_empty_inputs(self):
        ff = FakeFusion()
        assert ff.fuse([], []) == []

    def test_fuse_deterministic(self):
        sparse = [Hit(chunk=_make_chunk("c1"), sparse_score=1.0)]
        dense = [Hit(chunk=_make_chunk("c2"), dense_score=0.5)]
        ff = FakeFusion()
        r1 = ff.fuse(sparse, dense)
        r2 = ff.fuse(sparse, dense)
        assert [h.fusion_score for h in r1] == [h.fusion_score for h in r2]

    def test_fuse_different_chunk_sets_different_scores(self):
        """Different chunk IDs must produce distinct score sequences."""
        ff = FakeFusion()
        sparse_a = [Hit(chunk=_make_chunk("a1"), sparse_score=1.0)]
        dense_a = [Hit(chunk=_make_chunk("a2"), dense_score=0.5)]
        sparse_b = [Hit(chunk=_make_chunk("b1"), sparse_score=1.0)]
        dense_b = [Hit(chunk=_make_chunk("b2"), dense_score=0.5)]
        hits_a = ff.fuse(sparse_a, dense_a)
        hits_b = ff.fuse(sparse_b, dense_b)
        scores_a = sorted(h.fusion_score or 0.0 for h in hits_a)
        scores_b = sorted(h.fusion_score or 0.0 for h in hits_b)
        assert scores_a != scores_b


# ---------------------------------------------------------------------------
# FakeReranker
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeReranker:
    def test_is_reranker(self):
        assert isinstance(FakeReranker(), Reranker)

    def test_rerank_sets_score(self):
        hits = [_make_hit(f"c{i}") for i in range(3)]
        fr = FakeReranker()
        result = fr.rerank(_make_query(), hits)
        assert all(h.rerank_score is not None for h in result)

    def test_rerank_score_in_range(self):
        hits = [_make_hit(f"c{i}") for i in range(5)]
        fr = FakeReranker()
        result = fr.rerank(_make_query(), hits)
        for h in result:
            assert h.rerank_score is not None
            assert 0.0 <= h.rerank_score <= 1.0

    def test_rerank_ordered_descending(self):
        hits = [_make_hit(f"c{i}") for i in range(5)]
        fr = FakeReranker()
        result = fr.rerank(_make_query(), hits)
        scores = [h.rerank_score for h in result if h.rerank_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_deterministic(self):
        hits = [_make_hit(f"c{i}") for i in range(3)]
        fr = FakeReranker()
        q = _make_query()
        r1 = fr.rerank(q, hits)
        r2 = fr.rerank(q, hits)
        assert [h.rerank_score for h in r1] == [h.rerank_score for h in r2]

    def test_rerank_empty(self):
        fr = FakeReranker()
        assert fr.rerank(_make_query(), []) == []

    def test_rerank_preserves_existing_scores(self):
        hit = Hit(chunk=_make_chunk("c1"), sparse_score=5.0, dense_score=0.9)
        fr = FakeReranker()
        result = fr.rerank(_make_query(), [hit])
        assert result[0].sparse_score == 5.0
        assert result[0].dense_score == 0.9


# ---------------------------------------------------------------------------
# FakeGenerator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeGenerator:
    def test_is_generator(self):
        assert isinstance(FakeGenerator(), Generator)

    def test_generate_returns_answer_response(self):
        fg = FakeGenerator()
        hits = [_make_hit()]
        result = fg.generate(_make_query(), hits)
        assert isinstance(result, AnswerResponse)

    def test_generate_query_id_matches(self):
        fg = FakeGenerator()
        q = _make_query("my question")
        hits = [_make_hit()]
        result = fg.generate(q, hits)
        assert result.query_id == q.id

    def test_generate_non_empty_answer(self):
        fg = FakeGenerator()
        hits = [_make_hit()]
        result = fg.generate(_make_query(), hits)
        assert not result.abstained
        assert result.answer_text != ""

    def test_generate_deterministic_text(self):
        fg = FakeGenerator()
        hits = [_make_hit()]
        q = _make_query("determinism check")
        r1 = fg.generate(q, hits)
        r2 = fg.generate(q, hits)
        assert r1.answer_text == r2.answer_text

    def test_generate_deterministic_token_count(self):
        fg = FakeGenerator()
        hits = [_make_hit()]
        q = _make_query("determinism check")
        r1 = fg.generate(q, hits)
        r2 = fg.generate(q, hits)
        assert r1.output_tokens == r2.output_tokens

    def test_generate_evidence_from_hits(self):
        fg = FakeGenerator()
        hits = [_make_hit(f"c{i}") for i in range(3)]
        result = fg.generate(_make_query(), hits)
        assert len(result.evidence) == 3

    def test_generate_max_3_evidence_items(self):
        fg = FakeGenerator()
        hits = [_make_hit(f"c{i}") for i in range(10)]
        result = fg.generate(_make_query(), hits)
        assert len(result.evidence) <= 3

    def test_generate_abstain_on_empty_hits(self):
        fg = FakeGenerator()
        result = fg.generate(_make_query(), [])
        assert result.abstained is True
        assert result.answer_text == ""

    def test_generate_abstain_flag(self):
        fg = FakeGenerator(abstain=True)
        hits = [_make_hit()]
        result = fg.generate(_make_query(), hits)
        assert result.abstained

    def test_generate_output_tokens_bounded(self):
        fg = FakeGenerator()
        hits = [_make_hit()]
        max_out = 20
        result = fg.generate(_make_query(), hits, max_output_tokens=max_out)
        assert result.output_tokens <= max_out

    def test_evidence_citation_labels(self):
        fg = FakeGenerator()
        hits = [_make_hit(f"c{i}") for i in range(3)]
        result = fg.generate(_make_query(), hits)
        labels = [ev.citation_label for ev in result.evidence]
        assert labels == ["S1", "S2", "S3"]

    def test_generate_tiny_max_output_tokens(self):
        """max_output_tokens smaller than 10 must not crash FakeGenerator."""
        fg = FakeGenerator()
        hits = [_make_hit()]
        result = fg.generate(_make_query(), hits, max_output_tokens=3)
        assert not result.abstained
        assert result.output_tokens <= 3


# ---------------------------------------------------------------------------
# FakeTokenCounter
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeTokenCounter:
    def test_is_token_counter(self):
        assert isinstance(FakeTokenCounter(), TokenCounter)

    def test_count_tokens_returns_int(self):
        ftc = FakeTokenCounter()
        result = ftc.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_word_count(self):
        ftc = FakeTokenCounter()
        assert ftc.count_tokens("hello world foo") == 3

    def test_count_tokens_empty(self):
        ftc = FakeTokenCounter()
        assert ftc.count_tokens("") == 0

    def test_count_tokens_model_ignored(self):
        ftc = FakeTokenCounter()
        assert ftc.count_tokens("one two", model="gpt-4") == 2


# ---------------------------------------------------------------------------
# FakeProgressObserver
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeProgressObserver:
    def test_is_progress_observer(self):
        assert isinstance(FakeProgressObserver(), ProgressObserver)

    def test_records_events(self):
        fpo = FakeProgressObserver()
        event = {"stage": "embed", "status": "done", "count": 10}
        fpo.on_event(event)
        assert len(fpo.events) == 1
        assert fpo.events[0] == event

    def test_multiple_events(self):
        fpo = FakeProgressObserver()
        for i in range(5):
            fpo.on_event({"stage": "chunk", "status": "progress", "count": i})
        assert len(fpo.events) == 5


# ---------------------------------------------------------------------------
# FakeQueryPlanner
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeQueryPlanner:
    def test_is_query_planner(self):
        assert isinstance(FakeQueryPlanner(), QueryPlanner)

    def test_plan_returns_n_subqueries(self):
        fqp = FakeQueryPlanner(n_subqueries=3)
        result = fqp.plan(_make_query(), [CorpusId("c1")])
        assert len(result) == 3

    def test_plan_subquery_texts_contain_original(self):
        q = Query(id=QueryId("q1"), text="original question")
        fqp = FakeQueryPlanner()
        result = fqp.plan(q, [])
        for sub in result:
            assert "original question" in sub.text

    def test_plan_deterministic(self):
        fqp = FakeQueryPlanner()
        q = _make_query("complex query")
        r1 = fqp.plan(q, [CorpusId("c1")])
        r2 = fqp.plan(q, [CorpusId("c1")])
        assert [s.text for s in r1] == [s.text for s in r2]

    def test_plan_zero_subqueries(self):
        fqp = FakeQueryPlanner(n_subqueries=0)
        result = fqp.plan(_make_query(), [])
        assert result == []

    def test_seed_accepted_but_unused(self):
        """seed= is accepted for interface uniformity; plans are text-derived, not RNG-driven."""
        fqp1 = FakeQueryPlanner(seed=1, n_subqueries=2)
        fqp2 = FakeQueryPlanner(seed=9999, n_subqueries=2)
        q = _make_query("same question")
        assert [s.text for s in fqp1.plan(q, [])] == [s.text for s in fqp2.plan(q, [])]


# ---------------------------------------------------------------------------
# FakeEvidenceGrader
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeEvidenceGrader:
    def test_is_evidence_grader(self):
        assert isinstance(FakeEvidenceGrader(), EvidenceGrader)

    def test_grade_returns_pairs(self):
        fg = FakeEvidenceGrader()
        evidence = _make_evidence(2)
        result = fg.grade(_make_query(), evidence)
        assert len(result) == 2
        for _ev, score in result:
            assert isinstance(score, float)

    def test_grade_score_in_range(self):
        fg = FakeEvidenceGrader()
        evidence = _make_evidence(3)
        result = fg.grade(_make_query(), evidence)
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_grade_ordered_descending(self):
        fg = FakeEvidenceGrader()
        evidence = _make_evidence(4)
        result = fg.grade(_make_query(), evidence)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_grade_deterministic(self):
        fg = FakeEvidenceGrader()
        evidence = _make_evidence(3)
        q = _make_query()
        r1 = fg.grade(q, evidence)
        r2 = fg.grade(q, evidence)
        assert [s for _, s in r1] == [s for _, s in r2]

    def test_grade_empty_evidence(self):
        fg = FakeEvidenceGrader()
        assert fg.grade(_make_query(), []) == []


# ---------------------------------------------------------------------------
# FakeCorpusRouter
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeCorpusRouter:
    def test_is_corpus_router(self):
        assert isinstance(FakeCorpusRouter(), CorpusRouter)

    def test_route_returns_all_by_default(self):
        fcr = FakeCorpusRouter()
        corpus_ids = [CorpusId("c1"), CorpusId("c2"), CorpusId("c3")]
        result = fcr.route(_make_query(), corpus_ids)
        assert result == corpus_ids

    def test_route_max_corpora(self):
        fcr = FakeCorpusRouter(max_corpora=2)
        corpus_ids = [CorpusId(f"c{i}") for i in range(5)]
        result = fcr.route(_make_query(), corpus_ids)
        assert len(result) == 2

    def test_route_returns_subset(self):
        fcr = FakeCorpusRouter(max_corpora=2)
        corpus_ids = [CorpusId("c1"), CorpusId("c2"), CorpusId("c3")]
        result = fcr.route(_make_query(), corpus_ids)
        assert all(cid in corpus_ids for cid in result)

    def test_route_deterministic(self):
        fcr = FakeCorpusRouter()
        corpus_ids = [CorpusId("c1"), CorpusId("c2")]
        q = _make_query()
        assert fcr.route(q, corpus_ids) == fcr.route(q, corpus_ids)

    def test_route_empty(self):
        fcr = FakeCorpusRouter()
        result = fcr.route(_make_query(), [])
        assert result == []


# ---------------------------------------------------------------------------
# FakeStopCondition
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeStopCondition:
    def test_stops_after_max_steps(self):
        fsc = FakeStopCondition(max_steps=3)
        results = [fsc.should_stop(None) for _ in range(4)]
        assert results == [False, False, True, True]

    def test_default_max_steps(self):
        fsc = FakeStopCondition()
        results = [fsc.should_stop(None) for _ in range(4)]
        # Default max_steps=3, so should stop at call 3
        assert results[2] is True

    def test_max_steps_one(self):
        fsc = FakeStopCondition(max_steps=1)
        assert fsc.should_stop(None) is True


# ---------------------------------------------------------------------------
# FakeTool
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeTool:
    def test_name_property(self):
        ft = FakeTool(name="my-tool")
        assert ft.name == "my-tool"

    def test_description_property(self):
        ft = FakeTool(description="A test tool.")
        assert ft.description == "A test tool."

    def test_call_returns_string(self):
        ft = FakeTool()
        result = ft.call("some input")
        assert isinstance(result, str)

    def test_call_echoes_input(self):
        ft = FakeTool(name="echo-tool")
        result = ft.call("hello world")
        assert "hello world" in result

    def test_call_is_deterministic(self):
        """FakeTool specifically is deterministic; verify equality of two calls."""
        ft = FakeTool()
        r1 = ft.call("identical input")
        r2 = ft.call("identical input")
        assert r1 == r2


# ---------------------------------------------------------------------------
# FakeSessionStore
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFakeSessionStore:
    def test_is_session_store(self):
        assert isinstance(FakeSessionStore(), SessionStore)

    def test_save_and_load(self):
        fss = FakeSessionStore()
        fss.save("session-1", {"key": "value"})
        loaded = fss.load("session-1")
        assert loaded == {"key": "value"}

    def test_load_missing_returns_none(self):
        fss = FakeSessionStore()
        assert fss.load("nonexistent") is None

    def test_delete_removes_session(self):
        fss = FakeSessionStore()
        fss.save("session-1", {"k": "v"})
        fss.delete("session-1")
        assert fss.load("session-1") is None

    def test_delete_nonexistent_is_ok(self):
        fss = FakeSessionStore()
        fss.delete("missing")  # should not raise

    def test_overwrite_session(self):
        fss = FakeSessionStore()
        fss.save("s1", {"a": 1})
        fss.save("s1", {"b": 2})
        assert fss.load("s1") == {"b": 2}


# ---------------------------------------------------------------------------
# Contract suites - run against fakes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConnectorContractOnFakeConnector(ConnectorContract):
    def make_subject(self) -> Connector:
        return FakeConnector()


@pytest.mark.unit
class TestEmbedderContractOnFakeEmbedder(EmbedderContract):
    def make_subject(self) -> Embedder:
        return FakeEmbedder(dim=16, batch_size=4)


@pytest.mark.unit
class TestSparseRetrieverContractOnFakeSparseRetriever(SparseRetrieverContract):
    def make_subject(self) -> SparseRetriever:
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        return FakeSparseRetriever(chunks=chunks)


@pytest.mark.unit
class TestDenseRetrieverContractOnFakeDenseRetriever(DenseRetrieverContract):
    def make_subject(self) -> DenseRetriever:
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        return FakeDenseRetriever(chunks=chunks)


@pytest.mark.unit
class TestFusionContractOnFakeFusion(FusionContract):
    def make_subject(self) -> Fusion:
        return FakeFusion()


@pytest.mark.unit
class TestTokenCounterContractOnFakeTokenCounter(TokenCounterContract):
    def make_subject(self) -> TokenCounter:
        return FakeTokenCounter()


@pytest.mark.unit
class TestProgressObserverContractOnFakeProgressObserver(ProgressObserverContract):
    def make_subject(self) -> ProgressObserver:
        return FakeProgressObserver()


@pytest.mark.unit
class TestSessionStoreContractOnFakeSessionStore(SessionStoreContract):
    def make_subject(self) -> SessionStore:
        return FakeSessionStore()


@pytest.mark.unit
class TestStopConditionContractOnFakeStopCondition(StopConditionContract):
    def make_subject(self) -> StopCondition:
        return FakeStopCondition(max_steps=3)


@pytest.mark.unit
class TestToolContractOnFakeTool(ToolContract):
    def make_subject(self) -> Tool:
        return FakeTool()


@pytest.mark.unit
class TestRerankerContractOnFakeReranker(RerankerContract):
    def make_subject(self) -> Reranker:
        return FakeReranker()


@pytest.mark.unit
class TestGeneratorContractOnFakeGenerator(GeneratorContract):
    def make_subject(self) -> Generator:
        return FakeGenerator()


@pytest.mark.unit
class TestQueryPlannerContractOnFakeQueryPlanner(QueryPlannerContract):
    def make_subject(self) -> QueryPlanner:
        return FakeQueryPlanner()


@pytest.mark.unit
class TestEvidenceGraderContractOnFakeGrader(EvidenceGraderContract):
    def make_subject(self) -> EvidenceGrader:
        return FakeEvidenceGrader()


@pytest.mark.unit
class TestCorpusRouterContractOnFakeRouter(CorpusRouterContract):
    def make_subject(self) -> CorpusRouter:
        return FakeCorpusRouter()
