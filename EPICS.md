# Implementation Epics: beacon-kb Standalone Library

## Planning Envelope

**Objective:** Ship one standalone, pip-installable, modular agentic RAG library that installs with `pipx install beacon-kb` and works fully offline on first run with zero credentials and zero model downloads.

**Current state:** No implementation exists yet; this is a greenfield build against the approved design spec.

**Target package:** Root `pyproject.toml`, source under `src/beacon_kb`, distribution name `beacon-kb`, import name `beacon_kb`, `src/` layout, `requires-python >=3.11`.

**Included scope:** Frozen typed models and protocols, typed config and errors, an entry-point plugin registry, a transactional SQLite FTS5 plus NumPy store with staged atomic revision promotion, canonical source identity with filesystem and memory connectors, structure-aware Markdown/HTML/PDF parsers, parent/child chunking with real overlap, fingerprint-driven incremental sync, hybrid sparse/dense retrieval with RRF and optional rerank, bounded context assembly, grounded single-shot answers with validated citations and abstention, a base-install agentic subpackage as pure orchestration, a tool surface with an optional MCP adapter, a CLI with a doctor command, offline evaluation gates for core and agentic paths, remote and local provider adapters, web and confluence connectors, and release docs.

**Explicitly excluded scope:** No Jira, incident RCA, Slack crawling or triage, hosted service, browser UI, scheduled crawler, or web-search retriever inside this library.
No `agentic` extra; the agentic subpackage is base and its optionality is behavioral.
No `AgenticKnowledgeBase` wrapper class and no `mode` flag on `answer()`.
No `api/` package and no deep nested public-surface package.
No production-scale external vector engine beyond the store protocol and optional adapter seam.
No `stop_conditions` or `tools` entry-point group in v1.

**Input boundary:** The library consumes ordinary files, pages, and injected clients through connectors; it never owns credential lookup, remote crawling, or upstream knowledge-production pipelines.

**Implementation stance:** Design against the known RAG defects listed in the spec pitfalls, treating each as a hard requirement: non-atomic cross-index writes, mutable JSON vector stores, truncated JSON manifests, swallowed index-write failures, random chunk identity, overlap misread as minimum chunk length, mandatory per-chunk enrichment, hardcoded provider batch limits, content-only change detection, uncompared index versions, per-source state clearing on rebuild, lowercased or section-dropped parser output, entangled HTML cleanup, silent PDF heading misclassification, equal-weight field search, one query reused across sparse and dense, raw-score fusion, exact-ID-only dedup, unconditional neighbor expansion, skipped evidence recaps, free-text citations, mixed-stage answer methods, discarded structured evidence, and hidden web search in generation.

**Consumption modes:** The library serves a library mode through the typed `KnowledgeBase` facade with explicit component injection, and a tool mode through the `beacon-kb` and `bkb` CLI over a TOML config.

**Facade cost contracts:** `search()` performs zero LLM calls and returns fused, bounded, cited evidence; `answer()` performs exactly one LLM call over that evidence with validated citations and abstention; `investigate()` runs a budgeted agentic loop whose central budget caps LLM calls, retrievals, tokens, and wall-clock time and always returns an inspectable trace.

## Epic Overview

| # | Epic | Track | Depends On | Tasks |
|---|------|-------|------------|-------|
| 01 | Standalone Package, Contracts, and Registry | A | None | 5 |
| 02 | Local Storage, Ingestion, and Incremental Indexing | B | Epic 01 | 5 |
| 03 | Hybrid Retrieval, Context, and Grounded Answers | C | Epic 01; parts of Epic 02 | 4 |
| 04 | Agentic Layer | D | Epic 03; multi-corpus parts of Epic 02 | 4 |
| 05 | Tool Surface and MCP | E | Epic 03; Epic 04 | 1 |
| 06 | CLI Tool Mode | E | Epic 03; Epic 04; parts of Epic 01 | 1 |
| 07 | Evaluation, Provider Adapters, and Release | E | Epic 02; Epic 03; Epic 04; Epic 05; Epic 06 | 4 |

Track A establishes contracts and the registry and runs first.
Tracks B and C run in parallel after Track A, joining where retrieval needs canonical chunks and grounded answers need a complete indexed corpus.
Track D builds the agentic layer after Track C.
Track E closes out tool surface, CLI, and release, with Epic 07 gating the release.

---

## Epic 01: Standalone Package, Contracts, and Registry

**Track:** A
**Depends on:** None
**Summary:** Establish the standalone distribution, the full typed contract surface including agentic strategy protocols, typed config and errors, and the entry-point plugin registry.

### Feature 01.1: Foundation and Contracts

#### Task 01.1.1: Scaffold the standalone distribution and quality toolchain

**Size:** S | 8 files | 250 LOC
**Complexity rationale:** This task lays down a single PEP 621 `src`-layout package with the full extras matrix, dual console scripts, and entry-point group declarations, so the packaging surface is small but load-bearing for every later epic.
**Depends on:** None
**Description:** Author the root `pyproject.toml` with distribution `beacon-kb`, import `beacon_kb`, `src` layout, `requires-python >=3.11`, `py.typed`, the full extras matrix (html, pdf, web, confluence, remote, local, mcp, dev; no agentic extra), the `beacon-kb` and `bkb` console scripts at one main, the entry-point group declarations, Ruff/mypy/pytest/coverage config, and `.gitignore`.

#### Task 01.1.2: Define frozen domain models, typed IDs, errors, and all pipeline and agentic-strategy protocols

**Size:** L | 4 files | 700 LOC
**Complexity rationale:** Every downstream stage depends on frozen records, content-addressed typed IDs, and runtime-checkable protocols that each declare score direction, error contract, and determinism, so this contract surface must be complete and stable before any implementation.
**Depends on:** 01.1.1
**Description:** Define `models.py` frozen records and enums for corpus, source, revision, raw document, section, chunk, fingerprint, query, hit, evidence, citation, sync report, answer response, and the always-on AgenticTrace record with content-addressed typed IDs; `errors.py` with config, readiness, backend, ingestion, citation, plugin, budget, and agentic errors; and `protocols.py` runtime-checkable Protocols for connectors, parsers, chunkers, embedders, stores, sparse/dense retrievers, fusion, rerankers, generators, token counters, and progress observers, plus QueryPlanner, EvidenceGrader, CorpusRouter, StopCondition, and SessionStore, each stating score direction, error contract, and determinism.

#### Task 01.1.3: Define typed config, the loader, and the facade shell with PLUGIN_API_VERSION

**Size:** M | 5 files | 470 LOC
**Complexity rationale:** The TOML config must mirror the frozen dataclasses so tool mode and library mode share one validated model, and the facade shell must compose injected components without importing any provider, keeping import side-effect free.
**Depends on:** 01.1.2
**Description:** Author `config.py` frozen config tree (core, retrieval, answer, agentic, plugins) with validation, `config_loader.py` TOML plus env overlay with actionable diagnostics, `version.py` with `__version__` and `PLUGIN_API_VERSION`, `tokens.py` default heuristic counter and budget arithmetic, and `facade.py` shell exposing sync, search, answer, investigate, inspect, and health that composes injected components without importing providers.

### Feature 01.2: Plugin Registry

#### Task 01.2.1: Implement the entry-point plugin registry

**Size:** M | 5 files | 460 LOC
**Complexity rationale:** The registry is the single extensibility mechanism, so it must resolve components in one fixed precedence order, scan entry points lazily, and refuse conflicting, missing, mismatched, or incompatible plugins with typed errors before indexing begins.
**Depends on:** 01.1.3
**Description:** Author `registry/groups.py` canonical group constants and group-to-protocol map, `registry/discovery.py` lazy entry-point scanning and capability-metadata parsing with the PLUGIN_API_VERSION check, `registry/precedence.py` deterministic resolver and conflict detection raising PluginConflict, PluginNotFound, and ProtocolMismatch, and `registry/builtins.py` registering first-party components through the same path.

#### Task 01.2.2: Provide deterministic fakes and contract harnesses

**Size:** M | 4 files | 340 LOC
**Complexity rationale:** Deterministic fakes and reusable per-protocol contract suites make every stage testable without credentials and let any plugin author verify conformance, and a real installable sample distribution proves the out-of-tree entry-point path, so this harness underpins the entire test strategy.
**Depends on:** 01.1.2
**Description:** Author `testing.py` deterministic embedder, generator, reranker, planner, grader, router, clock, and failure-injection fakes plus reusable per-protocol contract-test suites that any plugin author runs, registry contract tests, and a real installable `tests/plugins/` sample distribution with its own entry point plus a test asserting discovery, precedence, and conflicts against `importlib.metadata`.

---

## Epic 02: Local Storage, Ingestion, and Incremental Indexing

**Track:** B
**Depends on:** Epic 01
**Summary:** Transactional local store, source connectors with canonical identity, structure-aware parsers, parent/child chunking, and staged fingerprint-driven sync.

### Feature 02.1: Transactional Local Store

#### Task 02.1.1: Implement the transactional SQLite store with staged atomic promotion

**Size:** L | 5 files | 600 LOC
**Complexity rationale:** One SQLite database must hold sparse, vector, and metadata state so a single promotion transaction controls visibility, staged writes stay invisible until promotion, and restart recovery keeps the prior active revision searchable.
**Depends on:** 01.2.1, 01.2.2
**Description:** Author `storage/sqlite.py`, `storage/vector_math.py`, `storage/migrations/0001_initial.sql`, and `indexing/manifest.py`; model corpora, revisions, chunks, FTS5 BM25, embedding rows, build runs, fingerprints, and active-revision pointers in one SQLite database with staged writes invisible until one promotion transaction, restart recovery, and store contract tests; register it as the default store.

### Feature 02.2: Sources and Parsing

#### Task 02.2.1: Implement source identity and filesystem and memory connectors

**Size:** M | 5 files | 380 LOC
**Complexity rationale:** A connector must normalize canonical identity and provenance without owning credentials, so discovery, canonical URIs, content-addressed IDs, and media resolution are separated from loading and parsing.
**Depends on:** 01.2.1, 01.2.2
**Description:** Author `ingestion/identity.py`, `ingestion/media.py`, `connectors/filesystem.py`, and `connectors/memory.py`; provide canonical URIs, content-addressed IDs, glob discovery, external-link mapping, media resolution, and connector contract tests; register them as first-party connector plugins.

#### Task 02.2.2: Implement structure-aware Markdown, HTML, and PDF parsers

**Size:** L | 6 files | 620 LOC
**Complexity rationale:** Parsers must preserve case, code, tables, links, headings, page numbers, anchors, and offsets while keeping site-specific cleanup behind hooks and emitting typed warnings rather than silently misclassifying structure.
**Depends on:** 02.2.1
**Description:** Author `parsing/base.py`, `parsing/markdown.py`, `parsing/html.py` (html extra), and `parsing/pdf.py` (pdf extra) emitting typed sections with heading paths, anchors, page/offset locators, code, tables, links, and warnings; preserve case; add safe fixtures; register them as parser plugins.

### Feature 02.3: Chunking, Embedding, and Sync

#### Task 02.3.1: Implement parent/child chunking, optional enrichment, and batched embeddings

**Size:** L | 5 files | 560 LOC
**Complexity rationale:** Chunking must derive deterministic parent/child/neighbor identity, implement real token overlap rather than a minimum length, keep enrichment optional and cached, and take batching from the injected provider.
**Depends on:** 02.2.2, 02.1.1
**Description:** Author `ingestion/chunking.py`, `ingestion/enrichment.py`, `indexing/embedding.py`, and `progress.py`; produce heading-aware parent/child chunks, real token overlap, deterministic parent/child/neighbor IDs, provider-owned batching, cached optional enrichment, and structured progress; register the default chunker.

#### Task 02.3.2: Implement staged full and incremental synchronization

**Size:** L | 5 files | 570 LOC
**Complexity rationale:** Sync must classify change from a full pipeline fingerprint, stage and validate all writes, promote atomically, roll back and recover from crashes, and report health as one recoverable operation so stores never drift apart.
**Depends on:** 02.3.1
**Description:** Author `ingestion/planning.py`, `ingestion/sync.py`, `indexing/coordinator.py`, and `indexing/validation.py`; classify change, invalidate on fingerprint drift, stage and promote, roll back, recover from crashes, expose EMPTY/BUILDING/READY/FAILED health and a typed SyncReport; wire the facade sync path.

---

## Epic 03: Hybrid Retrieval, Context, and Grounded Answers

**Track:** C
**Depends on:** Epic 01; parts of Epic 02
**Summary:** The zero-LLM search and one-LLM answer core the agentic loop reuses: hybrid retrieval, bounded context, and single-shot grounded answers.

### Feature 03.1: Hybrid Retrieval and Context

#### Task 03.1.1: Implement sparse and dense candidate retrieval with typed scores

**Size:** L | 6 files | 520 LOC
**Complexity rationale:** Sparse and dense retrievers must produce independent ranks with explicit score direction, preserve the original question for lexical precision, and support a first-class sparse-only degraded mode when no embedder is configured.
**Depends on:** 02.1.1
**Description:** Author `retrieval/query.py`, `retrieval/sparse.py`, `retrieval/dense.py`, and `retrieval/filters.py`; provide weighted FTS5 BM25 and dense vector retrieval with independent ranks and explicit score direction, original-question preservation, sparse-only degraded mode when no embedder is configured, consistent filters, and diagnostics; register them under retriever groups.

#### Task 03.1.2: Add RRF fusion, optional reranking, and diversity

**Size:** M | 3 files | 380 LOC
**Complexity rationale:** BM25 and cosine scores are not comparable, so fusion must be rank-based with deterministic tie-breaking, reranking must stay optional and bounded, and dedup must collapse near-duplicates while preserving provenance.
**Depends on:** 03.1.1
**Description:** Author `retrieval/fusion.py`, `retrieval/rerank.py`, and `retrieval/diversity.py`; implement rank-based RRF with deterministic tie-breaking, optional bounded reranking, near-duplicate collapse preserving provenance, and retained component scores; register them under fusion and reranker groups.

#### Task 03.1.3: Assemble bounded context and expose RetrievalPipeline

**Size:** M | 4 files | 360 LOC
**Complexity rationale:** Context must expand parents and neighbors only after final ordering under a token budget, keep `context_of` relationships, and expose one deterministic `RetrievalPipeline.search` reused by answer and investigate so citation logic is never duplicated.
**Depends on:** 03.1.2, 02.3.1
**Description:** Author `retrieval/context.py`, `retrieval/snippets.py`, and `retrieval/pipeline.py`; provide token-budgeted parent/neighbor expansion, match-centered snippets, stable [S1] evidence IDs distinguishing hits from context, and one `RetrievalPipeline.search(query, filters)` call reused by answer and investigate.

### Feature 03.2: Grounded Single-Shot Answers

#### Task 03.2.1: Generate grounded answers with validated citations and abstention

**Size:** L | 5 files | 560 LOC
**Complexity rationale:** The answer path must keep rewrite, retrieval, abstention, generation, and validation as separate stages, return evidence IDs and reject unknown ones, and preserve cited evidence in the response so the one-LLM contract stays verifiable.
**Depends on:** 03.1.3, 02.3.2
**Description:** Author `generation/answer.py`, `generation/citations.py`, `generation/prompts.py`, and `generation/abstention.py`; provide zero-LLM search and exactly-one-LLM answer, evidence-ID citation validation, untrusted-context delimiters, deterministic pre-generation abstention, versioned prompts, and complete diagnostics; wire the facade search and answer paths.

---

## Epic 04: Agentic Layer

**Track:** D
**Depends on:** Epic 03; multi-corpus parts of Epic 02
**Summary:** The base-install agentic subpackage as pure orchestration over the Epic 03 pipeline, each feature independently degradable, lazily imported by investigate().

### Feature 04.1: Budgeted Loop and Evidence Grading

#### Task 04.1.1: Implement budgets, stop conditions, and the always-on trace

**Size:** M | 5 files | 380 LOC
**Complexity rationale:** Central budget enforcement must cap LLM calls, retrievals, tokens, wall-clock time, and marginal gain so no strategy can run away, and the append-only trace must be deterministic under fakes and testable with no LLM.
**Depends on:** 03.1.3, 03.2.1
**Description:** Author `agentic/budget.py` with LLM-call, retrieval, token, wall-clock, and marginal-gain ceilings, `agentic/trace.py` append-only replayable AgenticTrace, and StopCondition default strategies; keep budget arithmetic and trace shape testable with no LLM.

#### Task 04.1.2: Implement evidence grading and the reflect loop

**Size:** L | 4 files | 560 LOC
**Complexity rationale:** The default grader must be a deterministic heuristic over retrieval-stage signals with zero LLM calls and an LLM grader as opt-in, and the loop must never exceed the budget, always emit a trace, and fall back to one retrieval when reflection is disabled.
**Depends on:** 04.1.1, 03.1.3
**Description:** Author `agentic/grading.py` keep/discard/re-retrieve verdicts with the deterministic heuristic default and an LLM grader opt-in adapter, and `agentic/loop.py` retrieve-reflect-refine over RetrievalPipeline that never exceeds the budget, always emits a trace, and falls back to one retrieval when reflection is disabled.

### Feature 04.2: Planning, Routing, Memory, Synthesis, and Facade

#### Task 04.2.1: Implement query planning and multi-corpus routing

**Size:** L | 4 files | 500 LOC
**Complexity rationale:** Planner and router must decompose queries and select corpora via registry-declared capabilities yet degrade to identity, so neither becomes a hidden default and both stay explicit config under `[agentic]`.
**Depends on:** 04.1.2, 01.2.1
**Description:** Author `agentic/planner.py` to decompose into an ordered subquery plan and `agentic/router.py` to score and select corpora per subquery via registry-declared capabilities; both degrade to identity (one subquery, all corpora) when disabled.

#### Task 04.2.2: Implement session memory, synthesis, engine, and facade investigate

**Size:** M | 5 files | 460 LOC
**Complexity rationale:** Synthesis must reuse `generation.answer` so citation validation and abstention behave identically to the single-shot path, and `investigate()` must lazily import the subpackage and be behaviorally identical to `answer()` with all features off.
**Depends on:** 04.2.1, 03.2.1
**Description:** Author `agentic/session.py` optional turn history and follow-up rewriting, `agentic/synthesis.py` cross-subquery evidence merge reusing generation, `agentic/orchestrator.py` AgenticEngine, and `facade.investigate(question, session, budget)` composing planner, router, loop, grader, session, and synthesis behind one lazily imported call; add an integration test asserting investigate with all features off is behaviorally identical to answer.

---

## Epic 05: Tool Surface and MCP

**Track:** E
**Depends on:** Epic 03; Epic 04
**Summary:** Framework-neutral tools with schemas defined once plus an optional MCP server behind the mcp extra.

### Feature 05.1: Tool Surface and MCP Server

#### Task 05.1.1: Implement the framework-neutral tool surface and the optional MCP server extra

**Size:** L | 5 files | 620 LOC
**Complexity rationale:** Tool schemas must be defined once and reused by in-process callables and the MCP server so there is one source of truth, and the MCP adapter must degrade gracefully when the extra is absent.
**Depends on:** 03.2.1, 04.2.2
**Description:** Author `tools/schema.py` tool schemas defined once (search, fetch_evidence, answer, investigate, list_corpora), `tools/surface.py` in-process callables mapping schemas to facade methods with structured results and redaction, and `tools/mcp.py` reusing the same schemas and callables behind the mcp extra with corpus resource listing and graceful absence when the extra is not installed; wire the serve-mcp command surface.

---

## Epic 06: CLI Tool Mode

**Track:** E
**Depends on:** Epic 03; Epic 04; parts of Epic 01
**Summary:** The CLI-first journey over a TOML config with doctor, stable exit codes, and journey tests.

### Feature 06.1: CLI Application and Commands

#### Task 06.1.1: Implement the CLI app, dispatch, renderers, config-driven init, and all commands including doctor and serve-mcp

**Size:** L | 6 files | 720 LOC
**Complexity rationale:** The CLI holds no business logic and calls only the facade, yet it must cover every command, a stable exit-code policy, three renderers, doctor messages that name the exact fix, and the shipped example configs, so the boundary layer is broad.
**Depends on:** 01.1.3, 02.3.2, 03.2.1, 04.2.2, 05.1.1
**Description:** Author `cli/app.py` dispatch with global flags (--config, --corpus, --json, --quiet, --no-color, --verbose) and the exit-code policy, `cli/render.py` human/plain/JSON renderers with progress, and `cli/commands.py` init scaffolding beacon-kb.toml with offline defaults, index with live progress, search zero-LLM, ask one-LLM, investigate printing the trace, inspect with the plugin capability report, doctor diagnosing config/extras/backend/credentials/readiness with exact-fix messages, plugins listing discovery and precedence, evaluate running gates, and serve-mcp when the mcp extra is present; ship the `examples/getting_started.toml` and `examples/multi_source.toml` reference configs; add CLI journey tests over a temp workspace.

---

## Epic 07: Evaluation, Provider Adapters, and Release

**Track:** E
**Depends on:** Epic 02; Epic 03; Epic 04; Epic 05; Epic 06
**Summary:** Offline quality gates for core and agentic paths, provider and remote-source adapters, docs, and the release.

### Feature 07.1: Evaluation and Quality Gates

#### Task 07.1.1: Build the offline evaluation and resilience suite with core gates

**Size:** L | 7 files | 560 LOC
**Complexity rationale:** Release readiness needs a versioned corpus and measurable gates covering retrieval quality, citation validity, abstention, and a full resilience matrix over restart, add/modify/delete, fingerprint migration, rollback, and concurrency.
**Depends on:** 02.3.2, 03.2.1
**Description:** Author `evaluation/metrics.py`, `evaluation/runner.py`, a safe versioned corpus.jsonl, and quality-gate tests enforcing Recall@5 >= 0.85, MRR@10 >= 0.75, citation validity 1.0, abstention accuracy >= 0.90, and a resilience matrix (restart, add/modify/delete, fingerprint migration, rollback, concurrency); wire an evaluate path.

#### Task 07.1.2: Add agentic evaluation and budget-regression gates

**Size:** M | 3 files | 340 LOC
**Complexity rationale:** The agentic path needs its own gates proving the loop never exceeds budget and that investigate never scores below the single-shot baseline, so agentic quality is measured rather than assumed.
**Depends on:** 07.1.1, 04.2.2
**Description:** Author `evaluation/agentic_metrics.py` for iterations, budget usage, grading precision, and answer gain, plus gates asserting the loop never exceeds budget and investigate never scores below the single-shot answer baseline on the labeled corpus.

### Feature 07.2: Adapters, Docs, and Release

#### Task 07.2.1: Add remote and local providers, web, and confluence connectors as plugins

**Size:** M | 4 files | 420 LOC
**Complexity rationale:** Optional adapters must satisfy the core protocols behind their extras with injected clients and env-var credentials, proving the extension path works without pulling heavy runtimes into the base install.
**Depends on:** 02.3.2, 03.2.1
**Description:** Author `providers/remote.py` remote embedding/generation/rerank behind the remote extra, `providers/local.py` local ONNX embedding and rerank behind the local extra, `connectors/web.py` (web extra), and `connectors/confluence.py` (confluence extra) with injected clients, env-var credentials, offline contract tests, and entry-point registration.

#### Task 07.2.2: Finalize extras hygiene, docs, benchmark, and release

**Size:** M | 13 files | 420 LOC
**Complexity rationale:** The release must verify extras isolation and wheel/sdist contents, complete every doc page and the examples tree, and run core and agentic gates so the shipped artifact matches the design and carries no secrets, caches, or local indexes.
**Depends on:** 07.1.2, 07.2.1, 05.1.1, 06.1.1
**Description:** Verify extras isolation and wheel/sdist contents (migrations included; no fixtures, caches, secrets, or local indexes), complete docs/ (architecture, configuration, cli, extending, plugins, agentic, operations, benchmark-report, quickstart), ship the `examples/` `library_quickstart.py` walk-through and `custom_plugin/` skeleton, run core and agentic gates, and record the release checklist and accepted trade-offs.

---

## Recommended Enhancements Included in the Plan

- Expose one typed `KnowledgeBase` facade with `search()`, `answer()`, and `investigate()` carrying hard cost contracts, with no wrapper class and no mode flag, so cost guarantees are legible at the API surface.
- Ship the agentic subpackage in the base install as pure orchestration with zero third-party dependencies, and lazily import it from `investigate()` so importing `beacon_kb` has no agentic import and no side effects.
- Keep sparse-only BM25 a first-class offline mode requiring no embedder, so the five-minute first run works with zero credentials and zero downloads, and activate dense retrieval only when an embedder is configured via the `local` or `remote` extra.
- Resolve every stage component through one entry-point group per stage with documented constant names, deterministic precedence, and `PluginConflict` on duplicate names, so extension is predictable and never silently shadowed.
- Discover plugins lazily on first resolution and register built-ins through the same registry path, so import has no side effects, unused plugins never import heavy dependencies, and no privileged code path can rot.
- Pin plugin compatibility with a single integer `PLUGIN_API_VERSION` and capability metadata, so the registry refuses incompatible plugins with a typed error before indexing begins.
- Store sparse, vector, and metadata state in one transactional SQLite database with staged writes and one atomic promotion, so partial writes are never query-visible and the prior active revision stays searchable on failure.
- Derive content-addressed identity from corpus, canonical source, revision, pipeline fingerprint, parent locator, and child ordinal, and include parser, chunker, enrichment, embedding model, embedding dimension, and schema versions in the fingerprint compared on every sync.
- Use rank-based RRF over independent sparse and dense ranks, preserve the original question for lexical precision, collapse near-duplicates while keeping provenance, and expand context only after final ordering under a token budget.
- Return typed evidence with stable IDs, validate every citation and reject unknown IDs, keep abstention deterministic, and render Markdown only at the CLI boundary.
- Enforce a central agentic budget with a graceful partial answer, keep the default evidence grader a deterministic heuristic with zero extra LLM calls, and always return an inspectable, deterministic `AgenticTrace`.
- Define tool schemas once and reuse them across in-process callables and the optional MCP server, and reference secrets by env-var name only in `beacon-kb.toml`.

## Deferred Beyond This Plan

- Confirm the specific lightweight ONNX embedding model shipped or fetched by the `local` extra and its wheel-size budget, since the offline dense-quality promise depends on it while the sparse-only mode is unaffected.
- Confirm the exact env-var names and provider SDK pins the `remote` extra targets before external release.
- Curate and approve the safe evaluation corpus labels before the recommended thresholds harden into release gates.
- Ship no `stop_conditions` or `tools` entry-point group in v1; the StopCondition and tool protocols may exist, but those strategies are configured by explicit instance only until a later version.
- Add no Jira, incident RCA, Slack crawling or triage, hosted service, browser UI, scheduled crawler, or web-search retriever inside this library.
- Add no `agentic` extra, no `AgenticKnowledgeBase` wrapper class, and no `mode` flag on `answer()`; agentic optionality stays behavioral through config and the explicit `investigate()` method.
- Add no `api/` package and no deep nested public-surface package, and add no production-scale external vector engine beyond the store protocol and optional adapter seam.
