"""KnowledgeBase facade - the primary entry point for beacon-kb users.

``KnowledgeBase`` is a thin composition shell that:
- Accepts injected components conforming to the protocols in ``beacon_kb.protocols``.
- Holds NO credential state and imports NO provider at construction or import time.
- Lazily imports ``beacon_kb.agentic`` ONLY inside ``investigate()`` so importing
  this module never triggers the agentic subpackage.
- Exposes exactly six methods: ``sync``, ``search``, ``answer``, ``investigate``,
  ``inspect``, and ``health``.

Cost contracts (from the plan):
- ``search()`` - zero LLM calls.
- ``answer()`` - exactly one LLM call (to the injected Generator).
- ``investigate()`` - budgeted loop (delegated to the lazy-loaded agentic module).

Methods raise ``ReadinessError`` when their required collaborators are not
yet injected, so the facade shell is useful even before all components are wired.

Importing this module performs no side effects and does NOT import
``beacon_kb.agentic``.  Verify with::

    python -c "import beacon_kb.facade, sys; assert 'beacon_kb.agentic' not in sys.modules"
"""

from __future__ import annotations

import importlib
from typing import Any

from beacon_kb.config import BeaconConfig
from beacon_kb.errors import ReadinessError
from beacon_kb.models import (
    AgenticTrace,
    AnswerResponse,
    ChunkId,
    Hit,
    Query,
    SyncReport,
)
from beacon_kb.progress import Clock, monotonic_clock
from beacon_kb.protocols import (
    Connector,
    DenseRetriever,
    Embedder,
    Fusion,
    Generator,
    Parser,
    ProgressObserver,
    Reranker,
    SparseRetriever,
    Store,
    TokenCounter,
)
from beacon_kb.tokens import HeuristicTokenCounter
from beacon_kb.version import PLUGIN_API_VERSION, __version__


class KnowledgeBase:
    """Composition root for the beacon-kb retrieval pipeline.

    Accepts injected components via ``__init__`` keyword arguments and
    orchestrates them across the six public API methods.  Each component
    is optional at construction; a ``ReadinessError`` is raised at call time
    if a required component is missing.

    No provider is imported by this class.  Credential state is never held.

    Args:
        config:          BeaconConfig.  Defaults to ``BeaconConfig()`` (all defaults).
        connector:       Source connector (Connector protocol).
        parser:          Document parser (Parser protocol).
        embedder:        Embedding provider (Embedder protocol).
        store:           Chunk storage backend (Store protocol).
        sparse_retriever: BM25 / term-frequency retriever (SparseRetriever protocol).
        dense_retriever:  Dense vector retriever (DenseRetriever protocol).
        fusion:          Score fusion strategy (Fusion protocol).
        reranker:        Cross-encoder reranker (Reranker protocol).
        generator:       LLM answer generator (Generator protocol).
        token_counter:   Token counter.  Defaults to HeuristicTokenCounter.
        observer:        Progress observer (ProgressObserver protocol).
        clock:           Injectable clock (``now() -> float``) used to time the
                         retrieval stage inside ``answer()``.  Defaults to a
                         real monotonic wall clock.
    """

    def __init__(
        self,
        *,
        config: BeaconConfig | None = None,
        connector: Connector | None = None,
        parser: Parser | None = None,
        embedder: Embedder | None = None,
        store: Store | None = None,
        sparse_retriever: SparseRetriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        fusion: Fusion | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        token_counter: TokenCounter | None = None,
        observer: ProgressObserver | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._config: BeaconConfig = config if config is not None else BeaconConfig()
        self._connector: Connector | None = connector
        self._parser: Parser | None = parser
        self._embedder: Embedder | None = embedder
        self._store: Store | None = store
        self._sparse_retriever: SparseRetriever | None = sparse_retriever
        self._dense_retriever: DenseRetriever | None = dense_retriever
        self._fusion: Fusion | None = fusion
        self._reranker: Reranker | None = reranker
        self._generator: Generator | None = generator
        self._token_counter: TokenCounter = (
            token_counter if token_counter is not None else HeuristicTokenCounter()
        )
        self._observer: ProgressObserver | None = observer
        self._clock: Clock = clock if clock is not None else monotonic_clock()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(self, event: dict[str, Any]) -> None:
        """Fire a progress event to the observer if one is registered."""
        if self._observer is not None:
            self._observer.on_event(event)

    def _require(self, component: Any, name: str) -> Any:
        """Return *component* or raise ReadinessError if it is None."""
        if component is None:
            raise ReadinessError(
                f"KnowledgeBase.{name} is not available: "
                f"the required component has not been injected. "
                f"Inject the component via the '{name}' constructor argument."
            )
        return component

    # ------------------------------------------------------------------
    # sync()
    # ------------------------------------------------------------------

    def sync(self) -> SyncReport:
        """Discover, fetch, parse, chunk, embed, and store all sources.

        Performs an incremental sync: sources unchanged since the last run
        are skipped.  Cost contract: zero LLM calls (embedder may call an
        embedding provider but that is not a generative LLM call).

        Returns:
            SyncReport with build run ID, corpus ID, status, and counts.

        Raises:
            ReadinessError: If connector, parser, embedder, or store are missing.
            IngestionError: If fetching or parsing fails.
            BackendError:   If the storage backend fails.
        """
        self._require(self._connector, "connector")
        self._require(self._parser, "parser")
        self._require(self._embedder, "embedder")
        self._require(self._store, "store")

        self._emit({"stage": "sync", "status": "started"})

        # Inline imports avoid circular deps: sync module only needed here.
        from beacon_kb.ingestion.chunking import HeadingAwareChunker
        from beacon_kb.ingestion.sync import SyncEngine
        from beacon_kb.models import CorpusId
        from beacon_kb.protocols import (
            Connector as _Connector,
        )
        from beacon_kb.protocols import (
            Embedder as _Embedder,
        )
        from beacon_kb.protocols import (
            Parser as _Parser,
        )
        from beacon_kb.protocols import (
            Store as _Store,
        )

        # _require guarantees these are non-None.
        # SyncEngine accepts any Store-protocol implementation; no SQLiteStore cast needed.
        store: _Store = self._store  # type: ignore[assignment]
        embedder: _Embedder = self._embedder  # type: ignore[assignment]
        connector: _Connector = self._connector  # type: ignore[assignment]
        parser: _Parser = self._parser  # type: ignore[assignment]

        corpus_name: str = self._config.core.corpus_name

        # Derive chunker params from config for fingerprinting.
        chunker_params: dict[str, Any] = {}

        def chunker_factory(
            corpus: str,
            canonical_uri: str,
            revision_id: str,
            pipeline_fingerprint: str,
        ) -> HeadingAwareChunker:
            return HeadingAwareChunker(
                corpus=corpus,
                canonical_uri=canonical_uri,
                revision_id=revision_id,
                pipeline_fingerprint=pipeline_fingerprint,
            )

        engine = SyncEngine(
            store=store,
            connector=connector,
            parser=parser,
            chunker_factory=chunker_factory,
            embedder=embedder,
            enrichment_orchestrator=None,
            observer=self._observer,
            corpus_name=corpus_name,
            chunker_params=chunker_params,
        )

        corpus_id = CorpusId(corpus_name)
        return engine.sync(corpus_id=corpus_id)

    # ------------------------------------------------------------------
    # search()
    # ------------------------------------------------------------------

    def search(self, query: Query) -> list[Hit]:
        """Retrieve relevant chunks for *query* with zero LLM calls.

        Runs the configured retrieval pipeline (sparse, dense, or hybrid) and
        optionally reranks the results.  No generative LLM is called.

        Cost contract: zero LLM calls.

        Args:
            query: Query record with text and optional corpus_id filter.

        Returns:
            Ordered list of Hit records (higher score = more relevant).

        Raises:
            ReadinessError: If the required retrievers are not injected.
            BackendError:   If the backend index read fails.
        """
        mode = self._config.retrieval.mode
        hits: list[Hit] = []

        if mode in {"sparse", "hybrid"}:
            sparse = self._require(self._sparse_retriever, "sparse_retriever")
            hits = sparse.retrieve(query)

        if mode in {"dense", "hybrid"}:
            dense = self._require(self._dense_retriever, "dense_retriever")
            dense_hits = dense.retrieve(query)
            if mode == "dense":
                hits = dense_hits
            elif self._fusion is not None:
                hits = self._fusion.fuse(hits, dense_hits)
            else:
                # No fusion: concatenate and dedup by chunk_id.
                seen: set[ChunkId] = {h.chunk.id for h in hits}
                hits = list(hits) + [h for h in dense_hits if h.chunk.id not in seen]

        if self._reranker is not None:
            hits = self._reranker.rerank(query, hits)

        top_k = self._config.retrieval.top_k
        return hits[:top_k]

    # ------------------------------------------------------------------
    # answer()
    # ------------------------------------------------------------------

    def answer(self, query: Query) -> AnswerResponse:
        """Retrieve evidence and generate an answer with exactly one LLM call.

        Runs the full retrieval pipeline (zero LLM calls) then passes the
        evidence through the generation stage (at most one LLM call).
        Pre-generation abstention fires deterministically when no evidence is
        available or when evidence falls below the configured quality threshold,
        skipping the LLM call entirely.

        Cost contract: zero LLM calls for retrieval; at most one LLM call for
        generation (skipped by pre-generation abstention when evidence is absent
        or below policy).

        Args:
            query: Query record with text and optional corpus_id filter.

        Returns:
            AnswerResponse with answer_text, evidence, and citation records.
            abstained=True with empty answer_text when no evidence is available
            or when the generator cannot ground the answer.

        Raises:
            ReadinessError: If the generator is not injected.
            ValueError:     If query.text is empty or whitespace-only.
            BackendError:   If retrieval or generation fails.
            CitationError:  If citation validation fails.
        """
        raw_generator = self._require(self._generator, "generator")
        generator: Generator = raw_generator

        # Inline imports keep the facade free of generation/retrieval imports
        # at module level.
        from beacon_kb.generation.answer import run_answer
        from beacon_kb.retrieval.query import prepare_query

        # Prepare the query variants that drive retrieval; recorded in
        # diagnostics.  The optional rewrite stage is not implemented yet
        # (arrives with the agentic layer), so prepare_query() with no
        # rewriters yields the variants that exist today.
        query_variants = prepare_query(query)

        self._emit({"stage": "answer", "status": "retrieving", "query_id": query.id})

        # Time the retrieval stage with the injectable clock so diagnostics
        # record a real elapsed_retrieval_s (run_answer cannot measure it:
        # it receives pre-computed hits).
        t_retrieval_start = self._clock.now()
        hits = self.search(query)
        elapsed_retrieval_s = self._clock.now() - t_retrieval_start

        self._emit(
            {
                "stage": "answer",
                "status": "generating",
                "query_id": query.id,
                "hit_count": len(hits),
            }
        )

        # Delegate to the generation stage: pre/post abstention + citation validation.
        response, _diag = run_answer(
            query,
            generator,
            hits,
            config=self._config.answer,
            observer=self._observer,
            query_variants=query_variants,
            elapsed_retrieval_s=elapsed_retrieval_s,
        )
        return response

    # ------------------------------------------------------------------
    # investigate()
    # ------------------------------------------------------------------

    def investigate(self, query: Query) -> AgenticTrace:
        """Run a multi-step agentic investigation loop for *query*.

        Lazily imports ``beacon_kb.agentic`` on the first call.  Importing
        ``beacon_kb.facade`` or constructing ``KnowledgeBase`` never triggers
        this import.

        Cost contract: budgeted loop (number of LLM calls is bounded by
        ``config.agentic.max_steps``).

        Args:
            query: Query record to investigate.

        Returns:
            AgenticTrace capturing every step of the investigation.

        Raises:
            ModuleNotFoundError: Until ``beacon_kb.agentic`` is implemented
                                 (Epic 04).
            ReadinessError:      If required agentic components are missing.
            BudgetError:         If the token or step budget is exceeded.
            AgenticError:        On unrecoverable loop failure.
        """
        # Lazy import: beacon_kb.agentic is not imported at module level.
        agentic = importlib.import_module("beacon_kb.agentic")
        return agentic.run_investigation(  # type: ignore[no-any-return]
            query=query,
            config=self._config.agentic,
            token_counter=self._token_counter,
        )

    # ------------------------------------------------------------------
    # inspect()
    # ------------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        """Return a structured snapshot of the knowledge base configuration.

        Returns configuration values, injected component names, and version
        metadata.  No I/O is performed.

        Returns:
            Dict with keys 'config', 'components', and 'version'.
        """
        components: dict[str, str | None] = {
            "connector": type(self._connector).__name__ if self._connector else None,
            "parser": type(self._parser).__name__ if self._parser else None,
            "embedder": type(self._embedder).__name__ if self._embedder else None,
            "store": type(self._store).__name__ if self._store else None,
            "sparse_retriever": (
                type(self._sparse_retriever).__name__ if self._sparse_retriever else None
            ),
            "dense_retriever": (
                type(self._dense_retriever).__name__ if self._dense_retriever else None
            ),
            "fusion": type(self._fusion).__name__ if self._fusion else None,
            "reranker": type(self._reranker).__name__ if self._reranker else None,
            "generator": type(self._generator).__name__ if self._generator else None,
            "token_counter": type(self._token_counter).__name__,
            "observer": type(self._observer).__name__ if self._observer else None,
        }
        return {
            "version": __version__,
            "plugin_api_version": PLUGIN_API_VERSION,
            "config": {
                "core": {
                    "corpus_name": self._config.core.corpus_name,
                    "log_level": self._config.core.log_level,
                    "data_dir": self._config.core.data_dir,
                },
                "retrieval": {
                    "mode": self._config.retrieval.mode,
                    "top_k": self._config.retrieval.top_k,
                },
                "answer": {
                    "model": self._config.answer.model,
                    "max_input_tokens": self._config.answer.max_input_tokens,
                    "max_output_tokens": self._config.answer.max_output_tokens,
                },
                "agentic": {
                    "max_steps": self._config.agentic.max_steps,
                    "token_budget": self._config.agentic.token_budget,
                },
            },
            "components": components,
        }

    # ------------------------------------------------------------------
    # health()
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return a health-check dict indicating readiness of each component.

        For each injected component, reports whether it is present.  Full
        health checks (ping, connectivity) are deferred to later epics.

        Returns:
            Dict with key 'status' ('ok' or 'degraded') and 'components' dict.
        """
        required_for_search = ["sparse_retriever", "dense_retriever"]

        components: dict[str, dict[str, Any]] = {}
        for name, component in [
            ("connector", self._connector),
            ("parser", self._parser),
            ("embedder", self._embedder),
            ("store", self._store),
            ("sparse_retriever", self._sparse_retriever),
            ("dense_retriever", self._dense_retriever),
            ("fusion", self._fusion),
            ("reranker", self._reranker),
            ("generator", self._generator),
            ("token_counter", self._token_counter),
            ("observer", self._observer),
        ]:
            components[name] = {
                "present": component is not None,
                "type": type(component).__name__ if component is not None else None,
            }

        # Determine overall status.
        mode = self._config.retrieval.mode
        required: list[str]
        if mode == "sparse":
            required = ["sparse_retriever"]
        elif mode == "dense":
            required = ["dense_retriever"]
        else:
            required = required_for_search

        search_ready = all(components[k]["present"] for k in required)
        answer_ready = search_ready and components["generator"]["present"]

        overall = "ok" if answer_ready else "degraded"

        # Derive corpus health from durable store state if store is present.
        corpus_health_value: str | None = None
        if self._store is not None:
            from beacon_kb.ingestion.sync import derive_corpus_health
            from beacon_kb.models import CorpusId

            corpus_id = CorpusId(self._config.core.corpus_name)
            corpus_health_value = derive_corpus_health(self._store, corpus_id).value

        return {
            "status": overall,
            "search_ready": search_ready,
            "answer_ready": answer_ready,
            "corpus_health": corpus_health_value,
            "components": components,
        }
