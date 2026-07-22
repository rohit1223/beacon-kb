"""Deterministic fakes and reusable contract-test suites for beacon-kb.

Importing this module performs no side effects.
All fakes are deterministic under a fixed seed using ``random.Random(seed)``.
Contract test suites are abstract base classes with a ``make_subject()`` hook
that subclasses implement to provide the implementation under test.

Note: Contract suites for Parser, Chunker, and Store (``ParserContract``,
``ChunkerContract``, ``StoreContract``) are intentionally absent from this
module. Those suites will be added in Epic 02 together with the concrete
implementations they exercise. Adding skeleton suites here without a real
implementation to verify them would give false confidence.
"""

from __future__ import annotations

import abc
import hashlib
import math
import random
from typing import Any, ClassVar

from beacon_kb.errors import BackendError, IngestionError
from beacon_kb.models import (
    AnswerResponse,
    Chunk,
    ChunkId,
    CorpusId,
    Evidence,
    EvidenceRole,
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

_DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _unit_normalize(vec: list[float]) -> list[float]:
    """Return a copy of *vec* scaled to unit L2 norm."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        norm = 1.0
    return [x / norm for x in vec]


def _chunk_ids_digest(chunk_ids: list[ChunkId]) -> int:
    """Return a stable int digest of a sorted list of chunk IDs.

    Used by FakeFusion to mix input identity into the RNG seed so that
    different chunk sets produce different score sequences while identical
    calls remain reproducible.
    """
    key = ",".join(sorted(str(cid) for cid in chunk_ids))
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _text_digest(text: str) -> int:
    """Return a stable int digest of *text* using SHA-256.

    Unlike ``hash()``, this is not salted by CPython's per-process
    ``PYTHONHASHSEED``, so results are identical across interpreter runs.
    Only the lower 32 bits are returned to keep XOR arithmetic in the same
    range as the former ``hash(text) & 0xFFFF_FFFF`` pattern.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Deterministic fake implementations
# ---------------------------------------------------------------------------


class FakeClock:
    """Controllable clock for budget-arithmetic tests.

    Not a protocol implementation - a pure test helper.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._time: float = start

    def now(self) -> float:
        """Return the current fake time."""
        return self._time

    def tick(self, delta: float = 1.0) -> None:
        """Advance the clock by *delta* seconds."""
        self._time += delta

    def advance_to(self, t: float) -> None:
        """Set the clock to an absolute time *t*."""
        self._time = t


class FakeFailingEmbedder:
    """Embedder that always raises BackendError.

    Used to test failure and rollback paths without a real embedding provider.
    """

    def __init__(self, *, dim: int = 16, message: str = "injected failure") -> None:
        self._dim: int = dim
        self._message: str = message

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Always raise BackendError."""
        raise BackendError(self._message)

    def dimension(self) -> int:
        """Return the declared dimension."""
        return self._dim


class FakeConnector:
    """In-memory Connector fake.

    Deterministic under any seed - sources are fixed at construction time.
    Raises IngestionError for unknown URIs.
    """

    _BUILTIN_SOURCES: ClassVar[dict[str, str]] = {
        "fake://doc-1": "Content of document one.",
        "fake://doc-2": "Content of document two.",
    }

    def __init__(
        self,
        sources: dict[str, str] | None = None,
        *,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self._seed: int = seed
        self._sources: dict[str, str] = (
            sources if sources is not None else dict(self._BUILTIN_SOURCES)
        )

    def list_sources(self) -> list[str]:
        """Return sorted list of source URIs."""
        return sorted(self._sources.keys())

    def fetch(self, uri: str) -> RawDocument:
        """Return the RawDocument for *uri*.

        Raises:
            IngestionError: If *uri* is not in the in-memory store.
        """
        if uri not in self._sources:
            raise IngestionError(f"FakeConnector: unknown URI {uri!r}")
        return RawDocument(
            source_id=SourceId(uri),
            revision_id=RevisionId(f"rev-{uri}"),
            content=self._sources[uri],
            media_type="text/plain",
        )


class FakeEmbedder:
    """Deterministic Embedder fake.

    Produces unit-normalized vectors whose values depend on the text content
    and the seed, so identical inputs always produce identical outputs.
    The batch_size attribute exposes the provider-owned batch size; core
    logic must never hardcode this value.

    Construction params:
        dim: Dimensionality of output vectors (default 16).
        batch_size: Provider-side batch size hint (default 8).
        seed: Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        *,
        dim: int = 16,
        batch_size: int = 8,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self._dim: int = dim
        self._batch_size: int = batch_size
        self._seed: int = seed

    @property
    def batch_size(self) -> int:
        """Provider-owned batch size hint."""
        return self._batch_size

    def _embed_one(self, text: str) -> list[float]:
        rng = random.Random(self._seed ^ _text_digest(text))
        raw = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        return _unit_normalize(raw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-normalized vector per input text."""
        return [self._embed_one(t) for t in texts]

    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dim


class FakeSparseRetriever:
    """Deterministic SparseRetriever fake.

    Returns a fixed set of hits with sparse_score set; other scores are None.
    """

    def __init__(self, chunks: list[Chunk] | None = None, *, seed: int = _DEFAULT_SEED) -> None:
        self._chunks: list[Chunk] = chunks if chunks is not None else []
        self._seed: int = seed

    def retrieve(self, query: Query) -> list[Hit]:
        """Return hits with sparse_score set, ordered descending."""
        if not self._chunks:
            return []
        rng = random.Random(self._seed ^ _text_digest(query.text))
        scored = [
            Hit(chunk=c, sparse_score=rng.uniform(0.1, 10.0))
            for c in self._chunks
        ]
        return sorted(scored, key=lambda h: h.sparse_score or 0.0, reverse=True)


class FakeDenseRetriever:
    """Deterministic DenseRetriever fake.

    Returns a fixed set of hits with dense_score set; other scores are None.
    """

    def __init__(self, chunks: list[Chunk] | None = None, *, seed: int = _DEFAULT_SEED) -> None:
        self._chunks: list[Chunk] = chunks if chunks is not None else []
        self._seed: int = seed

    def retrieve(self, query: Query) -> list[Hit]:
        """Return hits with dense_score set, ordered descending."""
        if not self._chunks:
            return []
        rng = random.Random(self._seed ^ _text_digest(query.text))
        scored = [
            Hit(chunk=c, dense_score=rng.random())
            for c in self._chunks
        ]
        return sorted(scored, key=lambda h: h.dense_score or 0.0, reverse=True)


class FakeFusion:
    """Deterministic Fusion fake.

    Assigns a fusion_score to all returned hits, ordered descending.
    Different chunk sets produce different score sequences because a stable
    digest of the fused chunk IDs is mixed into the RNG seed; identical calls
    remain reproducible.
    """

    def __init__(self, *, seed: int = _DEFAULT_SEED) -> None:
        self._seed: int = seed

    def fuse(self, sparse_hits: list[Hit], dense_hits: list[Hit]) -> list[Hit]:
        """Merge sparse and dense hits, setting fusion_score on each."""
        seen: dict[ChunkId, Hit] = {}
        for hit in sparse_hits + dense_hits:
            if hit.chunk.id not in seen:
                seen[hit.chunk.id] = hit
        combined = list(seen.values())
        # Mix a digest of the input chunk IDs into the seed so distinct input
        # sets yield distinct score sequences while identical inputs stay stable.
        id_digest = _chunk_ids_digest(list(seen.keys()))
        rng = random.Random(self._seed ^ id_digest)
        result = [
            Hit(
                chunk=h.chunk,
                sparse_score=h.sparse_score,
                dense_score=h.dense_score,
                fusion_score=rng.random(),
            )
            for h in combined
        ]
        return sorted(result, key=lambda h: h.fusion_score or 0.0, reverse=True)


class FakeReranker:
    """Deterministic Reranker fake.

    rerank_score is in [0, 1] where higher is better, ordered descending.
    """

    def __init__(self, *, seed: int = _DEFAULT_SEED) -> None:
        self._seed: int = seed

    def rerank(self, query: Query, hits: list[Hit]) -> list[Hit]:
        """Assign deterministic rerank_score in [0, 1] to each hit."""
        rng = random.Random(self._seed ^ _text_digest(query.text))
        scored = [
            Hit(
                chunk=h.chunk,
                sparse_score=h.sparse_score,
                dense_score=h.dense_score,
                fusion_score=h.fusion_score,
                rerank_score=rng.random(),
            )
            for h in hits
        ]
        return sorted(scored, key=lambda h: h.rerank_score or 0.0, reverse=True)


class FakeGenerator:
    """Deterministic Generator fake.

    Returns canned answers with evidence derived from the top-3 hits.
    Set ``abstain=True`` to force AnswerResponse with abstained=True.
    """

    def __init__(self, *, seed: int = _DEFAULT_SEED, abstain: bool = False) -> None:
        self._seed: int = seed
        self._abstain: bool = abstain

    def generate(
        self,
        query: Query,
        hits: list[Hit],
        *,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 512,
    ) -> AnswerResponse:
        """Generate a deterministic answer from *query* and *hits*."""
        if self._abstain or not hits:
            return AnswerResponse(
                query_id=query.id,
                answer_text="",
                evidence=(),
                abstained=True,
                input_tokens=0,
                output_tokens=0,
            )
        rng = random.Random(self._seed ^ _text_digest(query.text))
        lo = min(10, max_output_tokens)
        token_count = rng.randint(lo, max(lo, min(50, max_output_tokens)))
        evidence_items: list[Evidence] = []
        for i, hit in enumerate(hits[:3]):
            eid = make_evidence_id(query_id=str(query.id), chunk_id=str(hit.chunk.id))
            ev = Evidence(
                id=eid,
                hit=hit,
                citation_label=f"S{i + 1}",
                role=EvidenceRole.HIT,
            )
            evidence_items.append(ev)
        answer_text = f"Answer to '{query.text}' based on {len(evidence_items)} sources."
        return AnswerResponse(
            query_id=query.id,
            answer_text=answer_text,
            evidence=tuple(evidence_items),
            abstained=False,
            input_tokens=len(query.text.split()),
            output_tokens=token_count,
        )


class FakeTokenCounter:
    """Deterministic TokenCounter fake using a simple word-based heuristic."""

    def count_tokens(self, text: str, *, model: str = "") -> int:
        """Return an approximate token count (word count)."""
        return len(text.split())


class FakeProgressObserver:
    """ProgressObserver fake that records all events for later inspection."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def on_event(self, event: dict[str, Any]) -> None:
        """Append *event* to the recorded events list."""
        self.events.append(event)


class FakeQueryPlanner:
    """Deterministic QueryPlanner fake.

    Decomposes each query into *n_subqueries* sub-queries by appending
    a ``[sub-N]`` suffix to the original query text.

    The ``seed`` constructor argument is accepted for interface uniformity with
    other fakes but is not used to drive plan variation: sub-query text is
    purely derived from the input query text and the sub-query index, making
    the output 100% deterministic without any RNG. If you need seeded plan
    variation, subclass and override ``plan()``.
    """

    def __init__(self, *, seed: int = _DEFAULT_SEED, n_subqueries: int = 2) -> None:
        # seed is intentionally unused; kept for interface uniformity only.
        self._n: int = n_subqueries

    def plan(self, query: Query, corpus_ids: list[CorpusId]) -> list[Query]:
        """Return a fixed number of sub-queries derived from *query*."""
        return [
            Query(
                id=QueryId(f"{query.id}-sub-{i}"),
                text=f"{query.text} [sub-{i}]",
                corpus_id=query.corpus_id,
            )
            for i in range(self._n)
        ]


class FakeEvidenceGrader:
    """Deterministic EvidenceGrader fake.

    Grade scores are in [0, 1] where higher means more relevant.
    Results are ordered by score descending.
    """

    def __init__(self, *, seed: int = _DEFAULT_SEED) -> None:
        self._seed: int = seed

    def grade(self, query: Query, evidence: list[Evidence]) -> list[tuple[Evidence, float]]:
        """Assign deterministic scores in [0, 1] to each evidence item."""
        rng = random.Random(self._seed ^ _text_digest(query.text))
        scored = [(ev, rng.random()) for ev in evidence]
        return sorted(scored, key=lambda x: x[1], reverse=True)


class FakeCorpusRouter:
    """Deterministic CorpusRouter fake.

    Returns all corpus_ids by default, or a leading slice of size *max_corpora*.
    """

    def __init__(
        self,
        *,
        seed: int = _DEFAULT_SEED,
        max_corpora: int | None = None,
    ) -> None:
        self._seed: int = seed
        self._max: int | None = max_corpora

    def route(self, query: Query, corpus_ids: list[CorpusId]) -> list[CorpusId]:
        """Return a deterministic subset of *corpus_ids*."""
        if self._max is None:
            return list(corpus_ids)
        return list(corpus_ids[: self._max])


class FakeStopCondition:
    """Deterministic StopCondition fake.

    Stops after *max_steps* calls to should_stop(), regardless of trace content.
    """

    def __init__(self, *, max_steps: int = 3) -> None:
        self._max_steps: int = max_steps
        self._calls: int = 0

    def should_stop(self, trace: Any) -> bool:
        """Return True after *max_steps* calls."""
        self._calls += 1
        return self._calls >= self._max_steps


class FakeTool:
    """Deterministic Tool fake that echoes its input back as the observation."""

    def __init__(self, *, name: str = "fake-tool", description: str = "A fake tool.") -> None:
        self._name: str = name
        self._description: str = description

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self._description

    def call(self, input: str) -> str:
        """Echo the input back as the tool observation."""
        return f"FakeTool({self._name}) received: {input}"


class FakeSessionStore:
    """In-memory SessionStore fake."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist *state* under *session_id*."""
        self._store[session_id] = dict(state)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Return the stored state for *session_id*, or None if absent."""
        stored = self._store.get(session_id)
        return dict(stored) if stored is not None else None

    def delete(self, session_id: str) -> None:
        """Remove the session state for *session_id* if it exists."""
        self._store.pop(session_id, None)


# ---------------------------------------------------------------------------
# Contract test base classes
# ---------------------------------------------------------------------------
# Subclass these in your plugin's test suite to verify protocol conformance.
# Each contract suite defines pytest test methods (starting with ``test_``).
# Implement the abstract ``make_subject()`` hook to return the object under test.
# ---------------------------------------------------------------------------


class ConnectorContract(abc.ABC):
    """Reusable contract-test suite for Connector implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Connector:
        """Return a fresh Connector instance for each test."""
        ...

    def test_is_connector_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Connector)

    def test_list_sources_returns_list_of_strings(self) -> None:
        subject = self.make_subject()
        sources = subject.list_sources()
        assert isinstance(sources, list)
        assert all(isinstance(s, str) for s in sources)

    def test_list_sources_is_deterministic(self) -> None:
        subject = self.make_subject()
        assert subject.list_sources() == subject.list_sources()

    def test_fetch_returns_raw_document(self) -> None:
        subject = self.make_subject()
        sources = subject.list_sources()
        if not sources:
            return
        doc = subject.fetch(sources[0])
        assert isinstance(doc, RawDocument)

    def test_fetch_source_id_matches_uri(self) -> None:
        subject = self.make_subject()
        sources = subject.list_sources()
        if not sources:
            return
        uri = sources[0]
        doc = subject.fetch(uri)
        assert str(doc.source_id) == uri

    def test_fetch_unknown_uri_raises(self) -> None:
        subject = self.make_subject()
        try:
            subject.fetch("nonexistent://uri-that-does-not-exist")
        except Exception as exc:
            # Accept any IngestionError class by name to handle test-suite
            # module-reload scenarios where the same exception class may have
            # two different identities in sys.modules.
            exc_name = type(exc).__name__
            if exc_name != "IngestionError":
                msg = f"Expected IngestionError, got {exc_name}"
                raise AssertionError(msg) from exc
        else:
            raise AssertionError("Expected IngestionError but no exception was raised")


class EmbedderContract(abc.ABC):
    """Reusable contract-test suite for Embedder implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Embedder:
        """Return a fresh Embedder instance for each test."""
        ...

    def test_is_embedder_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Embedder)

    def test_dimension_positive(self) -> None:
        subject = self.make_subject()
        assert subject.dimension() > 0

    def test_embed_returns_correct_shape(self) -> None:
        subject = self.make_subject()
        texts = ["hello", "world", "foo"]
        vecs = subject.embed(texts)
        assert len(vecs) == len(texts)
        dim = subject.dimension()
        for v in vecs:
            assert len(v) == dim

    def test_embed_unit_normalized(self) -> None:
        subject = self.make_subject()
        vecs = subject.embed(["test text for normalization"])
        v = vecs[0]
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-5

    def test_embed_deterministic(self) -> None:
        subject = self.make_subject()
        texts = ["determinism test"]
        assert subject.embed(texts) == subject.embed(texts)

    def test_embed_empty_list(self) -> None:
        subject = self.make_subject()
        assert subject.embed([]) == []

    def test_batching_provider_owned(self) -> None:
        """Core logic must not hardcode batch size - it comes from the provider.

        The Embedder contract requires implementations to expose a ``batch_size``
        attribute (int) that core pipeline code reads to drive batching decisions.
        Fakes and real providers alike must supply this attribute.
        """
        subject = self.make_subject()
        raw_batch_size = getattr(subject, "batch_size", None)
        assert isinstance(raw_batch_size, int), (
            "Embedder contract violation: subject must expose a 'batch_size' int attribute "
            "so pipeline code can read the provider-owned batch size rather than hardcoding it. "
            f"Got: {raw_batch_size!r}"
        )
        batch_size: int = raw_batch_size
        texts = [f"text-{i}" for i in range(batch_size + 3)]
        vecs = subject.embed(texts)
        assert len(vecs) == len(texts)


class SparseRetrieverContract(abc.ABC):
    """Reusable contract-test suite for SparseRetriever implementations."""

    @abc.abstractmethod
    def make_subject(self) -> SparseRetriever:
        """Return a fresh SparseRetriever instance for each test."""
        ...

    def test_is_sparse_retriever_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, SparseRetriever)

    def test_retrieve_sets_sparse_score(self) -> None:
        subject = self.make_subject()
        # The contract test relies on the subject returned by make_subject()
        # already being populated (as the FakeSparseRetriever test fixture is).
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        # If no hits are returned (empty index), skip score-field checks.
        if not hits:
            return
        assert all(h.sparse_score is not None for h in hits), (
            "SparseRetriever contract: every returned Hit must have sparse_score set."
        )

    def test_retrieve_only_sparse_score_set(self) -> None:
        """SparseRetriever must set only sparse_score; other score fields stay None."""
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        for h in hits:
            assert h.dense_score is None, (
                "SparseRetriever contract: dense_score must be None on sparse hits."
            )
            assert h.fusion_score is None, (
                "SparseRetriever contract: fusion_score must be None on sparse hits."
            )
            assert h.rerank_score is None, (
                "SparseRetriever contract: rerank_score must be None on sparse hits."
            )

    def test_retrieve_ordered_descending(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        scores = [h.sparse_score for h in hits if h.sparse_score is not None]
        assert scores == sorted(scores, reverse=True), (
            "SparseRetriever contract: hits must be ordered by sparse_score descending."
        )

    def test_retrieve_deterministic(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        r1 = subject.retrieve(query)
        r2 = subject.retrieve(query)
        assert [h.sparse_score for h in r1] == [h.sparse_score for h in r2], (
            "SparseRetriever contract: retrieve() must be deterministic for identical inputs."
        )

    def test_retrieve_empty_returns_list(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        result = subject.retrieve(query)
        assert isinstance(result, list)


class DenseRetrieverContract(abc.ABC):
    """Reusable contract-test suite for DenseRetriever implementations."""

    @abc.abstractmethod
    def make_subject(self) -> DenseRetriever:
        """Return a fresh DenseRetriever instance for each test."""
        ...

    def test_is_dense_retriever_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, DenseRetriever)

    def test_retrieve_sets_dense_score(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        if not hits:
            return
        assert all(h.dense_score is not None for h in hits), (
            "DenseRetriever contract: every returned Hit must have dense_score set."
        )

    def test_retrieve_only_dense_score_set(self) -> None:
        """DenseRetriever must set only dense_score; other score fields stay None."""
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        for h in hits:
            assert h.sparse_score is None, (
                "DenseRetriever contract: sparse_score must be None on dense hits."
            )
            assert h.fusion_score is None, (
                "DenseRetriever contract: fusion_score must be None on dense hits."
            )
            assert h.rerank_score is None, (
                "DenseRetriever contract: rerank_score must be None on dense hits."
            )

    def test_retrieve_ordered_descending(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        hits = subject.retrieve(query)
        scores = [h.dense_score for h in hits if h.dense_score is not None]
        assert scores == sorted(scores, reverse=True), (
            "DenseRetriever contract: hits must be ordered by dense_score descending."
        )

    def test_retrieve_deterministic(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        r1 = subject.retrieve(query)
        r2 = subject.retrieve(query)
        assert [h.dense_score for h in r1] == [h.dense_score for h in r2], (
            "DenseRetriever contract: retrieve() must be deterministic for identical inputs."
        )

    def test_retrieve_returns_list(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        result = subject.retrieve(query)
        assert isinstance(result, list)


class FusionContract(abc.ABC):
    """Reusable contract-test suite for Fusion implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Fusion:
        """Return a fresh Fusion instance for each test."""
        ...

    def _make_hits(
        self,
        chunk_ids: list[str],
        *,
        sparse: bool = False,
        dense: bool = False,
    ) -> list[Hit]:
        hits = []
        for cid in chunk_ids:
            chunk = Chunk(
                id=ChunkId(cid),
                source_id=SourceId("s"),
                revision_id=RevisionId("r"),
                section_id=SectionId("sec"),
                text=f"text for {cid}",
                ordinal=0,
                parent_locator="",
            )
            hit = Hit(
                chunk=chunk,
                sparse_score=1.0 if sparse else None,
                dense_score=0.5 if dense else None,
            )
            hits.append(hit)
        return hits

    def test_is_fusion_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Fusion)

    def test_fuse_sets_fusion_score(self) -> None:
        subject = self.make_subject()
        sparse = self._make_hits(["c1"], sparse=True)
        dense = self._make_hits(["c2"], dense=True)
        hits = subject.fuse(sparse, dense)
        assert all(h.fusion_score is not None for h in hits), (
            "Fusion contract: every returned Hit must have fusion_score set."
        )

    def test_fuse_deduplicates_by_chunk_id(self) -> None:
        subject = self.make_subject()
        sparse = self._make_hits(["c1"], sparse=True)
        dense = self._make_hits(["c1"], dense=True)
        hits = subject.fuse(sparse, dense)
        assert len(hits) == 1, (
            "Fusion contract: duplicate chunk IDs must be merged into one Hit."
        )

    def test_fuse_ordered_descending(self) -> None:
        subject = self.make_subject()
        sparse = self._make_hits(["c1", "c2"], sparse=True)
        dense = self._make_hits(["c3", "c4"], dense=True)
        hits = subject.fuse(sparse, dense)
        scores = [h.fusion_score for h in hits if h.fusion_score is not None]
        assert scores == sorted(scores, reverse=True), (
            "Fusion contract: hits must be ordered by fusion_score descending."
        )

    def test_fuse_deterministic(self) -> None:
        subject = self.make_subject()
        sparse = self._make_hits(["c1"], sparse=True)
        dense = self._make_hits(["c2"], dense=True)
        r1 = subject.fuse(sparse, dense)
        r2 = subject.fuse(sparse, dense)
        assert [h.fusion_score for h in r1] == [h.fusion_score for h in r2], (
            "Fusion contract: fuse() must be deterministic for identical inputs."
        )

    def test_fuse_empty_inputs(self) -> None:
        subject = self.make_subject()
        result = subject.fuse([], [])
        assert isinstance(result, list)
        assert result == []


class TokenCounterContract(abc.ABC):
    """Reusable contract-test suite for TokenCounter implementations."""

    @abc.abstractmethod
    def make_subject(self) -> TokenCounter:
        """Return a fresh TokenCounter instance for each test."""
        ...

    def test_is_token_counter_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, TokenCounter)

    def test_count_tokens_returns_non_negative_int(self) -> None:
        subject = self.make_subject()
        result = subject.count_tokens("hello world")
        assert isinstance(result, int), "TokenCounter contract: count_tokens must return int."
        assert result >= 0, "TokenCounter contract: count must be non-negative."

    def test_count_tokens_empty_string(self) -> None:
        subject = self.make_subject()
        result = subject.count_tokens("")
        assert isinstance(result, int)
        assert result >= 0

    def test_count_tokens_deterministic(self) -> None:
        subject = self.make_subject()
        text = "the quick brown fox"
        assert subject.count_tokens(text) == subject.count_tokens(text), (
            "TokenCounter contract: count_tokens() must be deterministic for identical inputs."
        )

    def test_count_tokens_longer_text_not_less(self) -> None:
        """Longer text must not produce a smaller token count than shorter text."""
        subject = self.make_subject()
        short = "hello"
        long_ = "hello world foo bar baz qux"
        assert subject.count_tokens(long_) >= subject.count_tokens(short), (
            "TokenCounter contract: token count must not decrease as text grows."
        )

    def test_count_tokens_model_kwarg_accepted(self) -> None:
        subject = self.make_subject()
        # Must not raise regardless of model name.
        result = subject.count_tokens("some text", model="unknown-model-xyz")
        assert isinstance(result, int)


class ProgressObserverContract(abc.ABC):
    """Reusable contract-test suite for ProgressObserver implementations."""

    @abc.abstractmethod
    def make_subject(self) -> ProgressObserver:
        """Return a fresh ProgressObserver instance for each test."""
        ...

    def test_is_progress_observer_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, ProgressObserver)

    def test_on_event_does_not_raise(self) -> None:
        """ProgressObserver contract: on_event() must never raise."""
        subject = self.make_subject()
        # Must swallow any internal errors; test with minimal and rich events.
        subject.on_event({"stage": "embed", "status": "done"})
        subject.on_event({"stage": "chunk", "status": "progress", "count": 5, "total": 10})
        subject.on_event({})  # Empty event must also be accepted silently.

    def test_on_event_accepts_arbitrary_dict(self) -> None:
        subject = self.make_subject()
        subject.on_event({"arbitrary_key": "arbitrary_value", "nested": {"x": 1}})

    def test_on_event_multiple_calls(self) -> None:
        subject = self.make_subject()
        for i in range(10):
            subject.on_event({"stage": "test", "status": "progress", "count": i})


class SessionStoreContract(abc.ABC):
    """Reusable contract-test suite for SessionStore implementations."""

    @abc.abstractmethod
    def make_subject(self) -> SessionStore:
        """Return a fresh SessionStore instance for each test."""
        ...

    def test_is_session_store_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, SessionStore)

    def test_save_and_load_round_trip(self) -> None:
        subject = self.make_subject()
        state = {"key": "value", "count": 42}
        subject.save("session-1", state)
        loaded = subject.load("session-1")
        assert loaded == state, (
            "SessionStore contract: load() must return the exact state saved by save()."
        )

    def test_load_missing_returns_none(self) -> None:
        subject = self.make_subject()
        result = subject.load("nonexistent-session-id")
        assert result is None, (
            "SessionStore contract: load() must return None for unknown session IDs."
        )

    def test_delete_removes_session(self) -> None:
        subject = self.make_subject()
        subject.save("s1", {"x": 1})
        subject.delete("s1")
        assert subject.load("s1") is None, (
            "SessionStore contract: load() must return None after delete()."
        )

    def test_delete_nonexistent_is_idempotent(self) -> None:
        subject = self.make_subject()
        subject.delete("missing-id")  # Must not raise.

    def test_save_overwrites_existing(self) -> None:
        subject = self.make_subject()
        subject.save("s1", {"a": 1})
        subject.save("s1", {"b": 2})
        loaded = subject.load("s1")
        assert loaded == {"b": 2}, (
            "SessionStore contract: second save() must overwrite the first."
        )

    def test_save_does_not_share_state_across_sessions(self) -> None:
        subject = self.make_subject()
        subject.save("s1", {"owner": "alice"})
        subject.save("s2", {"owner": "bob"})
        assert subject.load("s1") == {"owner": "alice"}
        assert subject.load("s2") == {"owner": "bob"}

    def test_deterministic_load(self) -> None:
        subject = self.make_subject()
        state = {"key": "determinism"}
        subject.save("s1", state)
        r1 = subject.load("s1")
        r2 = subject.load("s1")
        assert r1 == r2, (
            "SessionStore contract: load() must be deterministic for stable stored state."
        )


class StopConditionContract(abc.ABC):
    """Reusable contract-test suite for StopCondition implementations."""

    @abc.abstractmethod
    def make_subject(self) -> StopCondition:
        """Return a fresh StopCondition instance for each test."""
        ...

    def test_is_stop_condition_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, StopCondition)

    def test_should_stop_returns_bool(self) -> None:
        subject = self.make_subject()
        result = subject.should_stop(None)
        assert isinstance(result, bool), (
            "StopCondition contract: should_stop() must return bool."
        )

    def test_should_stop_does_not_raise(self) -> None:
        """StopCondition contract: should_stop() must never raise."""
        subject = self.make_subject()
        # Call multiple times with varied trace values.
        subject.should_stop(None)
        subject.should_stop({})
        subject.should_stop({"steps": 5, "hits": []})

    def test_eventually_stops(self) -> None:
        """A well-behaved StopCondition must eventually return True."""
        subject = self.make_subject()
        # Drive up to 1000 calls; any real implementation should stop well before.
        stopped = any(subject.should_stop({"step": i}) for i in range(1000))
        assert stopped, (
            "StopCondition contract: should_stop() must return True within 1000 calls "
            "to prevent unbounded loops."
        )


class ToolContract(abc.ABC):
    """Reusable contract-test suite for Tool implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Tool:
        """Return a fresh Tool instance for each test."""
        ...

    def test_is_tool_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Tool)

    def test_name_is_non_empty_string(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject.name, str) and subject.name, (
            "Tool contract: name must be a non-empty string."
        )

    def test_description_is_non_empty_string(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject.description, str) and subject.description, (
            "Tool contract: description must be a non-empty string."
        )

    def test_name_is_stable(self) -> None:
        subject = self.make_subject()
        assert subject.name == subject.name, (
            "Tool contract: name must be stable across accesses."
        )

    def test_call_returns_string(self) -> None:
        subject = self.make_subject()
        result = subject.call("test input")
        assert isinstance(result, str), (
            "Tool contract: call() must return a str observation."
        )

    def test_call_does_not_raise(self) -> None:
        """Tool contract: call() must not raise and must return str."""
        subject = self.make_subject()
        r1 = subject.call("identical input")
        r2 = subject.call("identical input")
        assert isinstance(r1, str), "Tool contract: call() must return str."
        assert isinstance(r2, str), "Tool contract: call() must return str."


class RerankerContract(abc.ABC):
    """Reusable contract-test suite for Reranker implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Reranker:
        """Return a fresh Reranker instance for each test."""
        ...

    def _make_hits(self, n: int = 5) -> list[Hit]:
        chunks = [
            Chunk(
                id=ChunkId(f"c{i}"),
                source_id=SourceId("s"),
                revision_id=RevisionId("r"),
                section_id=SectionId("sec"),
                text=f"chunk {i}",
                ordinal=i,
                parent_locator="",
            )
            for i in range(n)
        ]
        return [Hit(chunk=c) for c in chunks]

    def test_is_reranker_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Reranker)

    def test_rerank_sets_score_higher_better(self) -> None:
        subject = self.make_subject()
        # Use 5 hits so that with any reasonable RNG at least 2 distinct scores emerge.
        hits = self._make_hits(5)
        query = Query(id=QueryId("q1"), text="test query")
        result = subject.rerank(query, hits)
        assert len(result) == len(hits)
        raw_scores = [h.rerank_score for h in result]
        assert all(s is not None for s in raw_scores)
        scores: list[float] = [s for s in raw_scores if s is not None]
        # Verify at least 2 distinct scores exist so the ordering assertion is not a tautology.
        distinct_scores = set(scores)
        assert len(distinct_scores) >= 2, (
            "RerankerContract: need at least 2 distinct rerank_scores to verify ordering. "
            "Provide 5+ hits with distinct IDs/texts so the fake's RNG produces varied scores."
        )
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_rerank_deterministic(self) -> None:
        subject = self.make_subject()
        hits = self._make_hits(5)
        query = Query(id=QueryId("q1"), text="test query")
        r1 = subject.rerank(query, hits)
        r2 = subject.rerank(query, hits)
        assert [h.rerank_score for h in r1] == [h.rerank_score for h in r2]

    def test_rerank_empty_hits(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test query")
        result = subject.rerank(query, [])
        assert result == []


class GeneratorContract(abc.ABC):
    """Reusable contract-test suite for Generator implementations."""

    @abc.abstractmethod
    def make_subject(self) -> Generator:
        """Return a fresh Generator instance for each test."""
        ...

    def _make_hit(self, chunk_id: str = "c1", text: str = "context text") -> Hit:
        chunk = Chunk(
            id=ChunkId(chunk_id),
            source_id=SourceId("s"),
            revision_id=RevisionId("r"),
            section_id=SectionId("sec"),
            text=text,
            ordinal=0,
            parent_locator="",
        )
        return Hit(chunk=chunk)

    def test_is_generator_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, Generator)

    def test_generate_returns_answer_response(self) -> None:
        subject = self.make_subject()
        hits = [self._make_hit()]
        query = Query(id=QueryId("q1"), text="what is this?")
        result = subject.generate(query, hits)
        assert isinstance(result, AnswerResponse)
        assert result.query_id == query.id

    def test_generate_deterministic(self) -> None:
        subject = self.make_subject()
        hits = [self._make_hit()]
        query = Query(id=QueryId("q1"), text="what is this?")
        r1 = subject.generate(query, hits)
        r2 = subject.generate(query, hits)
        assert r1.answer_text == r2.answer_text
        assert r1.output_tokens == r2.output_tokens

    def test_generate_abstained_on_empty_hits(self) -> None:
        """Generator contract: empty hits must trigger abstention, not fabrication."""
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="what is this?")
        result = subject.generate(query, [])
        assert isinstance(result, AnswerResponse)
        assert result.query_id == query.id
        assert result.abstained is True, (
            "Generator contract: generate() with empty hits must set abstained=True. "
            "The generator must not fabricate content when no evidence is available."
        )
        # Abstained responses must carry an empty answer_text per the abstention
        # contract: the generator may not synthesize content when it cannot ground
        # the answer in evidence.
        assert result.answer_text == "", (
            "Generator contract: abstained response must have empty answer_text. "
            "The generator must not fabricate content when no evidence is available."
        )


class QueryPlannerContract(abc.ABC):
    """Reusable contract-test suite for QueryPlanner implementations."""

    @abc.abstractmethod
    def make_subject(self) -> QueryPlanner:
        """Return a fresh QueryPlanner instance for each test."""
        ...

    def test_is_planner_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, QueryPlanner)

    def test_plan_returns_queries(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="complex question")
        result = subject.plan(query, [CorpusId("c1"), CorpusId("c2")])
        assert isinstance(result, list)
        assert all(isinstance(q, Query) for q in result)

    def test_plan_deterministic(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="complex question")
        corpus_ids = [CorpusId("c1")]
        r1 = subject.plan(query, corpus_ids)
        r2 = subject.plan(query, corpus_ids)
        assert [q.text for q in r1] == [q.text for q in r2]

    def test_plan_empty_corpus(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="complex question")
        result = subject.plan(query, [])
        assert isinstance(result, list)


class EvidenceGraderContract(abc.ABC):
    """Reusable contract-test suite for EvidenceGrader implementations."""

    @abc.abstractmethod
    def make_subject(self) -> EvidenceGrader:
        """Return a fresh EvidenceGrader instance for each test."""
        ...

    def _make_evidence(self, query_id: str = "q1", n: int = 3) -> list[Evidence]:
        chunks = [
            Chunk(
                id=ChunkId(f"c{i}"),
                source_id=SourceId("s"),
                revision_id=RevisionId("r"),
                section_id=SectionId("sec"),
                text=f"evidence text {i}",
                ordinal=i,
                parent_locator="",
            )
            for i in range(n)
        ]
        hits = [Hit(chunk=c) for c in chunks]
        return [
            Evidence(
                id=make_evidence_id(query_id=query_id, chunk_id=str(h.chunk.id)),
                hit=h,
                citation_label=f"S{i + 1}",
            )
            for i, h in enumerate(hits)
        ]

    def test_is_grader_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, EvidenceGrader)

    def test_grade_returns_scored_pairs(self) -> None:
        subject = self.make_subject()
        evidence = self._make_evidence(n=1)
        query = Query(id=QueryId("q1"), text="test")
        result = subject.grade(query, evidence)
        assert len(result) == 1
        _ev, score = result[0]
        assert 0.0 <= score <= 1.0

    def test_grade_score_higher_is_better_ordering(self) -> None:
        subject = self.make_subject()
        evidence = self._make_evidence(n=3)
        query = Query(id=QueryId("q1"), text="test")
        result = subject.grade(query, evidence)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_grade_deterministic(self) -> None:
        subject = self.make_subject()
        evidence = self._make_evidence(n=3)
        query = Query(id=QueryId("q1"), text="test")
        r1 = subject.grade(query, evidence)
        r2 = subject.grade(query, evidence)
        assert [s for _, s in r1] == [s for _, s in r2]

    def test_grade_empty_evidence(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test")
        result = subject.grade(query, [])
        assert result == []


class CorpusRouterContract(abc.ABC):
    """Reusable contract-test suite for CorpusRouter implementations."""

    @abc.abstractmethod
    def make_subject(self) -> CorpusRouter:
        """Return a fresh CorpusRouter instance for each test."""
        ...

    def test_is_router_instance(self) -> None:
        subject = self.make_subject()
        assert isinstance(subject, CorpusRouter)

    def test_route_returns_subset_of_corpus_ids(self) -> None:
        subject = self.make_subject()
        corpus_ids = [CorpusId("c1"), CorpusId("c2"), CorpusId("c3")]
        query = Query(id=QueryId("q1"), text="test")
        result = subject.route(query, corpus_ids)
        assert isinstance(result, list)
        assert all(cid in corpus_ids for cid in result)

    def test_route_deterministic(self) -> None:
        subject = self.make_subject()
        corpus_ids = [CorpusId("c1"), CorpusId("c2")]
        query = Query(id=QueryId("q1"), text="test")
        r1 = subject.route(query, corpus_ids)
        r2 = subject.route(query, corpus_ids)
        assert r1 == r2

    def test_route_empty_corpus_ids(self) -> None:
        subject = self.make_subject()
        query = Query(id=QueryId("q1"), text="test")
        result = subject.route(query, [])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ConnectorContract",
    "CorpusRouterContract",
    "DenseRetrieverContract",
    "EmbedderContract",
    "EvidenceGraderContract",
    "FakeClock",
    "FakeConnector",
    "FakeCorpusRouter",
    "FakeDenseRetriever",
    "FakeEmbedder",
    "FakeEvidenceGrader",
    "FakeFailingEmbedder",
    "FakeFusion",
    "FakeGenerator",
    "FakeProgressObserver",
    "FakeQueryPlanner",
    "FakeReranker",
    "FakeSessionStore",
    "FakeSparseRetriever",
    "FakeStopCondition",
    "FakeTokenCounter",
    "FakeTool",
    "FusionContract",
    "GeneratorContract",
    "ProgressObserverContract",
    "QueryPlannerContract",
    "RerankerContract",
    "SessionStoreContract",
    "SparseRetrieverContract",
    "StopConditionContract",
    "TokenCounterContract",
    "ToolContract",
]
