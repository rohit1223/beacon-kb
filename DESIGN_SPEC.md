# beacon-kb Design Spec

Authoritative design for a standalone, pip-installable Python RAG library.
This spec synthesizes three architecture proposals under binding rulings and is the single source of truth for execution.
Distribution name `beacon-kb`, import name `beacon_kb`, `src/` layout, root `pyproject.toml`, Python 3.11 or newer.

## Goals

- Ship one standalone distribution that installs with `pipx install beacon-kb` and works fully offline on first run with zero credentials and zero model downloads.
- Expose one typed `KnowledgeBase` facade with three methods carrying hard cost contracts: `search()` performs zero LLM calls, `answer()` performs exactly one, `investigate()` runs a budgeted agentic loop.
- Preserve the deterministic core rigor: frozen typed models, transactional SQLite FTS5 plus NumPy storage with staged atomic revision promotion, fingerprint-driven incremental sync, hybrid retrieval with RRF and optional rerank, bounded context, validated citations with abstention, offline evaluation gates, and deterministic fakes.
- Make every pipeline stage a runtime-checkable `Protocol` and make every stage extensible through one entry-point plugin registry so third parties ship connectors, providers, stores, and strategies as separate pip packages without forking.
- Ship the agentic subpackage in the base install as pure orchestration over the core with zero third-party dependencies, where every agentic feature degrades independently and importing `beacon_kb` does not import the agentic subpackage.
- Offer a first-class CLI (`beacon-kb` and `bkb`) with a TOML config, stable exit codes, a doctor command, and error messages that name the exact fix.
- Ship an optional MCP adapter behind the `mcp` extra that reuses the single tool surface.

## Non-Goals

- No Jira, incident RCA, Slack crawling or triage, hosted service, browser UI, scheduled crawler, or web-search retriever inside this library.
- No `agentic` extra: the agentic subpackage is base, and optionality is behavioral through config and the explicit `investigate()` method.
- No `AgenticKnowledgeBase` wrapper class and no `mode` flag on `answer()`.
- No `api/` package and no deep nested public-surface package; prefer flat modules.
- No production-scale external vector engine beyond the store protocol and optional adapter seam.
- No `stop_conditions` or `tools` entry-point group in v1 (protocols may exist; groups are deferred).

## Module Layout

Flat modules are preferred (`models.py`, `protocols.py`, `errors.py`, `config.py`, facade file) and a module splits into a package only where it would exceed roughly 600 LOC.
Source connectors live in `connectors/`.
Model-provider adapters (embedders, generators, rerankers) live in `providers/`.

```text
beacon-kb/                              # repo root, standalone distribution
  pyproject.toml                            # PEP 621 metadata, extras matrix, entry-point groups, tool config
  README.md                                 # install modes, five-minute quickstart, extras table
  LICENSE
  CHANGELOG.md                              # auto-generated release notes, never hand-edited
  .gitignore                                # Python caches, venvs, wheels, sdists, local index dirs
  docs/
    quickstart.md                           # init/index/search/ask journey with copy-paste blocks
    architecture.md                         # boundaries, data flow, identity, storage, scoring, citation, security
    configuration.md                        # full beacon-kb.toml schema reference with defaults and env overrides
    cli.md                                  # every command, flag, exit code, output mode, doctor
    extending.md                            # protocol plus plugin-registry authoring, ship a plugin as a package
    plugins.md                              # entry-point group catalog, precedence, capability metadata, PLUGIN_API_VERSION
    agentic.md                              # agentic layer, budgets, stop conditions, trace, degradation matrix, tools, MCP
    operations.md                           # rebuild, fingerprint migration, backup, rollback, budget tuning
    benchmark-report.md                     # corpus, metrics, quality-gate results, accepted trade-offs
  examples/
    getting_started.toml                    # minimal single-source local config, no credentials
    multi_source.toml                       # filesystem plus web plus confluence-style config
    library_quickstart.py                   # library-mode ingest, search, answer, investigate via the facade
    custom_plugin/                          # minimal third-party plugin package skeleton registering a connector
  src/beacon_kb/
    __init__.py                             # curated public exports and __all__, no import-time side effects, no agentic import
    py.typed                                # PEP 561 typing marker
    version.py                              # single source of __version__ and PLUGIN_API_VERSION
    models.py                               # frozen domain records and enums, typed IDs, AgenticTrace record
    protocols.py                            # runtime-checkable Protocols for every pipeline stage and agentic strategy
    errors.py                               # typed error hierarchy including plugin, budget, and agentic errors
    config.py                               # typed config tree (core, retrieval, answer, agentic, plugins) with validation
    config_loader.py                        # TOML load, env-var overlay for secrets, layered merge, actionable diagnostics
    facade.py                               # KnowledgeBase facade: sync, search, answer, investigate, inspect, health
    progress.py                             # structured stage/progress events plus logging and TTY-neutral adapter
    tokens.py                               # default heuristic TokenCounter and budget arithmetic helpers
    testing.py                              # deterministic embedder/generator/reranker/planner/grader/router/clock/failure fakes and contract harnesses
    registry/
      __init__.py                           # registry facade: resolve, register, list, describe by group and name
      groups.py                             # canonical entry-point group name constants and group-to-protocol map
      discovery.py                          # lazy entry-point scanning, capability-metadata parsing, PLUGIN_API_VERSION check
      precedence.py                         # deterministic precedence resolver and conflict detection
      builtins.py                           # register first-party components through the same registry path
    storage/
      __init__.py                           # store exports and migration helpers
      sqlite.py                             # transactional SQLite store: FTS5 BM25, embedding rows, active-revision promotion
      vector_math.py                        # normalized-vector validation and local NumPy similarity search
      migrations/0001_initial.sql           # versioned local schema and indexes
    indexing/
      __init__.py                           # indexing package exports
      manifest.py                           # build and validate index fingerprints and revision metadata
      embedding.py                          # provider-neutral batching, validation, retry, cache through the Embedder protocol
      coordinator.py                        # coordinate sparse/vector/metadata writes inside one revision transaction
      validation.py                         # validate counts, IDs, dimensions, links, fingerprint consistency before promotion
    ingestion/
      __init__.py                           # ingestion package exports
      identity.py                           # canonical source URIs and stable content-addressed source/revision IDs
      media.py                              # media-type resolution and parser-selection hints
      chunking.py                           # heading-aware parent/child chunking, real overlap, stable IDs, neighbor links
      enrichment.py                         # optional cached enrichment orchestration with failure policy
      planning.py                           # deterministic change classification: unchanged/new/changed/deleted/incompatible
      sync.py                               # full and incremental sync: scan, parse, chunk, embed, stage, validate, promote
    parsing/
      __init__.py                           # parser package exports
      base.py                               # section/provenance helpers shared by parsers
      markdown.py                           # default Markdown parser preserving case, code, tables, links, headings
      html.py                               # HTML parser with generic extraction and cleanup hooks (html extra)
      pdf.py                                # page-aware PDF parser with page-level provenance (pdf extra)
    connectors/
      __init__.py                           # connector package exports
      filesystem.py                         # filesystem and glob source connector
      memory.py                             # in-memory source connector for tests and embedding apps
      web.py                                # web-page source connector (web extra)
      confluence.py                         # confluence-style page-tree connector with injected client (confluence extra)
    providers/
      __init__.py                           # provider-adapter package exports
      remote.py                             # remote embedding/generation/rerank provider adapters (remote extra)
      local.py                              # local ONNX embedding and rerank runtime adapters (local extra)
    retrieval/
      __init__.py                           # retrieval package exports
      query.py                              # query validation, sparse/dense variant selection, original-question preservation
      sparse.py                             # weighted FTS5 BM25 retriever with exact-token boosts
      dense.py                              # dense vector retriever with declared similarity semantics
      filters.py                            # provider-neutral namespace/ACL/source/tag/media/date filters
      fusion.py                             # Reciprocal Rank Fusion with deterministic tie-breaking
      rerank.py                             # optional reranker invocation over a bounded window
      diversity.py                          # near-duplicate collapse and optional MMR-style diversity
      context.py                            # bounded parent/neighbor expansion and evidence packing under a token budget
      snippets.py                           # match-centered, locator-preserving snippet construction
      pipeline.py                           # RetrievalPipeline.search: one deterministic path reused by answer and investigate
    generation/
      __init__.py                           # generation package exports
      answer.py                             # single-shot grounded answer: rewrite, retrieve, abstain, generate, validate
      citations.py                          # resolve and validate evidence IDs and citation locators
      prompts.py                            # versioned grounded prompts with untrusted-context delimiters
      abstention.py                         # configurable pre- and post-generation abstention policy
    agentic/                                # base install, pure orchestration, zero third-party deps, lazily imported by investigate()
      __init__.py                           # agentic subpackage exports, imported only on first investigate() call
      budget.py                             # budgets and stop conditions: LLM calls, retrievals, tokens, wall-clock, marginal gain
      trace.py                              # always-on append-only, replayable AgenticTrace steps
      grading.py                            # evidence grading: deterministic heuristic default, LLM grader opt-in adapter
      planner.py                            # query planning and decomposition, identity default when disabled
      router.py                             # multi-corpus routing, all-corpora default when disabled
      session.py                            # optional session memory: turns, carried evidence, follow-up rewriting
      loop.py                               # retrieve-reflect-refine controller over RetrievalPipeline under a budget
      synthesis.py                          # cross-subquery evidence merge and final answer assembly reusing generation
      orchestrator.py                       # AgenticEngine wiring planner, router, loop, grader, session, synthesis behind one call
    tools/
      __init__.py                           # tool-surface exports
      schema.py                             # tool schemas defined once: search, fetch_evidence, answer, investigate, list_corpora
      surface.py                            # in-process tool callables mapping schemas to facade methods with structured results
      mcp.py                                # thin MCP server reusing schema and surface (mcp extra)
    evaluation/
      __init__.py                           # evaluation package exports
      metrics.py                            # Recall@K, MRR, nDCG, citation validity, abstention, latency, cost metrics
      agentic_metrics.py                    # loop metrics: iterations, budget usage, grading precision, answer gain over baseline
      runner.py                             # run a versioned corpus, emit JSON plus Markdown summaries
    cli/
      __init__.py                           # console-script entry-point wiring for beacon-kb and bkb
      app.py                                # command dispatch, global flags, exit-code policy, color policy
      commands.py                           # init, index, search, ask, investigate, inspect, doctor, plugins, evaluate, serve-mcp
      render.py                             # human, plain, and JSON renderers at the CLI boundary only
  tests/
    conftest.py                             # deterministic fixtures, temp corpus/store, failure injection, markers
    unit/                                   # per-module unit tests
    contract/                               # protocol and plugin conformance tests for every stage and strategy
    integration/                            # sync lifecycle, rollback, retrieval-from-index, grounded answer, agentic loop
    plugins/                                # sample third-party package asserting discovery, precedence, conflicts
    cli/                                    # end-to-end CLI journey tests over a temp workspace
    evaluation/corpus.jsonl                 # safe labeled queries and expected evidence
    evaluation/test_quality_gates.py        # enforce committed offline thresholds, core and agentic
    performance/test_local_baseline.py      # record local indexing/search/loop timing and memory
    fixtures/documents/                     # safe md/html/pdf regression content
```

## Pipeline Architecture

The deterministic core is one linear pipeline.
The agentic loop is a scheduler that repeatedly calls `RetrievalPipeline.search` and reuses `generation.answer`, never a second retrieval stack.
The registry resolves every stage component from built-ins or plugins by the same path.

```text
                         beacon-kb.toml + env overlay
                                     |
                                     v
                          config_loader -> config
                                     |
                                     v
                         registry.resolve(group, name)
        explicit instance > config name > sole entry point > built-in default
                                     |
   +---------------------------------+---------------------------------+
   |                                                                   |
   v                          INGESTION + INDEXING                     |
 SourceConnector -> RawDocument -> Parser -> DocumentSection           |
   -> Chunker -> Chunk -> optional Enricher -> Embedder                |
   -> IndexCoordinator (stage) -> validate -> atomic promote           |
   -> transactional SQLite store (FTS5 BM25 + embedding rows)          |
                                     |                                  |
                                     v                                  |
                          RETRIEVAL PIPELINE (search)                   |
 SparseRetriever  DenseRetriever                                        |
        \            /                                                  |
         v          v                                                   |
          RRF fusion -> optional Reranker -> diversity                  |
              -> bounded context assembly -> snippets                   |
              -> Evidence[] with stable [S1] IDs   <-- search(): 0 LLM  |
                                     |                                  |
                                     v                                  |
                       GENERATION (single-shot answer)                 |
   optional rewrite -> RetrievalPipeline.search -> abstention gate      |
        -> grounded prompt (untrusted-context delimited)                |
        -> AnswerGenerator -> citation validation   <-- answer(): 1 LLM |
                                     |                                  |
                                     v                                  |
                    AGENTIC LOOP (investigate, budgeted)   <------------+
   planner.plan -> for each subquery:                                   
     router.route -> RetrievalPipeline.search -> grader.grade           
       -> reflect (gaps) -> refine (follow-up subqueries)               
     budget checked before every retrieval and every LLM call          
     stop on: budget exhausted | no marginal gain | all answered | confidence
   -> synthesis merge -> generation.answer -> validated citations       
   -> always-on AgenticTrace returned                                   
                                     |
                                     v
                    TOOL SURFACE (schema defined once)
        in-process callables  ----+----  MCP server (mcp extra)
```

Plugin registry position: every arrow that creates a component (connector, parser, chunker, embedder, store, retriever, fusion, reranker, generator, token counter, planner, grader, router) resolves through `registry.resolve`.
Built-ins register through `registry/builtins.py` on the same path as third-party plugins.

## Plugin Registry Design

The registry is the single extensibility mechanism.
A third-party pip package is a first-class extension, never a fork.

### Entry-point groups

Group names are constants in `registry/groups.py` and documented in `docs/plugins.md` as API surface protected by semver.

```text
beacon_kb.connectors       # SourceConnector implementations
beacon_kb.parsers          # Parser implementations
beacon_kb.chunkers         # Chunker implementations
beacon_kb.embedders        # Embedder implementations
beacon_kb.stores           # KnowledgeStore implementations
beacon_kb.retrievers       # SparseRetriever / DenseRetriever implementations
beacon_kb.fusion           # Fusion implementations
beacon_kb.rerankers        # Reranker implementations
beacon_kb.generators       # AnswerGenerator implementations
beacon_kb.token_counters   # TokenCounter implementations
beacon_kb.planners         # QueryPlanner implementations
beacon_kb.graders          # EvidenceGrader implementations
beacon_kb.routers          # CorpusRouter implementations
```

Deferred in v1: `stop_conditions` and `tools`.
The `StopCondition` and tool protocols may exist in `protocols.py`, but no entry-point group ships, so those strategies are configured by explicit instance only until a later version.

### Discovery

Discovery is lazy: `registry/discovery.py` scans entry points via `importlib.metadata.entry_points(group=...)` on first resolution, never at import.
Importing `beacon_kb` performs no plugin loading and no side effects.
An entry point is loaded only when its plugin is actually resolved, so an installed-but-unused plugin with a heavy dependency never imports that dependency.

### Precedence

`registry/precedence.py` resolves a component request in one fixed, documented order:

1. an explicit instance passed to the facade or builder wins (library mode);
2. a plugin named in config or the call, resolved by exact name within its group;
3. a single entry point in the group when the group defines a default and exactly one is installed;
4. the built-in default for that group.

Two entry points registering the same name in the same group raise `PluginConflict` at resolution with both distribution names; there is no last-installed-wins.
A named plugin that is requested but not installed raises `PluginNotFound` listing the group and the installed names.
A resolved object that does not satisfy the target `Protocol` raises `ProtocolMismatch` listing the missing members.

### Capability metadata and PLUGIN_API_VERSION

Each entry point may declare capability metadata: declared embedding dimension, supported media types, whether it needs network, and the plugin API line it targets.
`resolve.py` rejects an incompatible plugin (for example a store whose declared vector dimension conflicts with the configured embedder) with a typed error before indexing begins.
A single integer `PLUGIN_API_VERSION` lives in `version.py`.
A plugin may declare the integer it targets; the registry refuses a plugin built against an incompatible major line with a typed error.

## Agentic Layer Design

The agentic subpackage ships in the base install as pure orchestration over the core with zero third-party dependencies.
Optionality is behavioral: explicit config under `[agentic]` plus the explicit `investigate()` method.
There is no `agentic` extra.
Importing `beacon_kb` does not import `beacon_kb.agentic`; `facade.investigate()` lazily imports the subpackage on first call.

### Components

- `budget.py` caps LLM calls, retrievals, total tokens, wall-clock time, and minimum marginal gain per iteration; checks are central so no strategy can exceed them.
- `trace.py` holds the always-on append-only `AgenticTrace` returned by `investigate()`; it records every planner decision, retrieval, grading verdict, routing choice, and stop trigger, deterministic under fakes.
- `grading.py` implements `EvidenceGrader`; the default is a deterministic heuristic using retrieval-stage signals (fusion rank, reranker score, source diversity) with zero LLM calls; an LLM grader is an opt-in adapter.
- `planner.py` implements `QueryPlanner`; the default identity planner yields one subquery equal to the question, so planning is a no-op unless configured.
- `router.py` implements `CorpusRouter`; the default selects all corpora, so routing is transparent until configured.
- `session.py` implements optional session memory: turn history, carried evidence, follow-up rewriting; a stateless call passes no session.
- `loop.py` runs the retrieve-reflect-refine controller over `RetrievalPipeline.search`, always emitting a trace and never exceeding the budget; with reflection disabled it performs exactly one retrieval.
- `synthesis.py` merges cross-subquery evidence and assembles the final answer reusing `generation.answer` so citation validation and abstention behave identically to the single-shot path.
- `orchestrator.py` provides `AgenticEngine`, wiring planner, router, loop, grader, session, and synthesis behind one call for `investigate()`.

### Loop, budgets, and stop conditions

One iteration retrieves for the active subquery, grades the evidence, reflects to detect gaps or contradictions, and refines by emitting follow-up subqueries.
The budget is consulted before every retrieval and every LLM call.
The loop stops when a stop condition fires: budget exhausted, no marginal evidence gain over the last iteration, all subqueries answered, or a confidence threshold met.
Budget exhaustion produces a graceful partial answer carried through the trace, never a facade exception.

### Degradation matrix

| Feature | On | Off (default behavior) |
|---|---|---|
| planner | decompose into subqueries | one subquery equal to the question |
| router | select corpora per subquery | all corpora |
| grader | LLM grader adapter | deterministic heuristic, zero LLM calls |
| reflection loop | multi-iteration refine | one retrieval |
| session memory | follow-up rewriting from history | stateless single-turn |

With all agentic features off, `investigate()` is behaviorally identical to `answer()`.
A hard test enforces this equivalence.
No agentic feature is ever a hidden default; each is explicit config under `[agentic]`.

### Tool surface and MCP

Tool schemas are defined once in `tools/schema.py`: `search`, `fetch_evidence`, `answer`, `investigate`, `list_corpora`.
In-process callables in `tools/surface.py` map schemas to facade methods and are reused by the MCP server.
`tools/mcp.py` is a thin adapter behind the `mcp` extra of this package that reuses the same schemas and callables, so there is one source of truth for tool schemas.
The `serve-mcp` CLI command is available when the `mcp` extra is installed.

## CLI and Config Design

### Console scripts and commands

`pyproject.toml` declares two console scripts, `beacon-kb` and short alias `bkb`, both pointing at the same `main`.
The CLI imports only the facade and config; it holds no business logic and calls only `KnowledgeBase`.

```text
beacon-kb init         # scaffold a commented beacon-kb.toml with sane offline defaults
beacon-kb index        # sync sources into the local SQLite backend with live progress
beacon-kb search       # zero-LLM hybrid search
beacon-kb ask          # one-LLM grounded answer with citations (facade method is answer())
beacon-kb investigate  # budgeted agentic loop, prints the AgenticTrace
beacon-kb inspect      # corpus health, counts, fingerprints, active revision, plugin capability report
beacon-kb doctor       # diagnose config, extras, backend, credentials, readiness
beacon-kb plugins      # list discovered plugins with distribution, group, and precedence
beacon-kb evaluate     # run the offline corpus against quality gates
beacon-kb serve-mcp    # start the MCP server when the mcp extra is installed
```

The facade method is `answer()` but the CLI command is `ask`.
Global flags: `--config`, `--corpus`, `--json`, `--quiet`, `--no-color`, `--verbose`.

### Exit codes

- 0 success
- 1 usage or config error
- 2 readiness error (corpus not indexed)
- 3 backend error
- 4 abstention treated as error only when `--strict` is set

### Doctor and error ergonomics

`doctor` diagnoses config validity, installed extras, backend readiness, credential presence, and corpus readiness.
Error messages name the failing thing and the fix: a missing extra prints the exact `pip install 'beacon-kb[extra]'` command, a missing credential names the env var, and an unindexed corpus tells the user to run `beacon-kb index`.

### Config schema sketch

Config file name is `beacon-kb.toml`.
The TOML mirrors the frozen `config.py` dataclasses so tool mode and library mode share one validated model.
Secrets are referenced by env-var name only, never inline.

```toml
[corpus]
name = "default"
store = "sqlite"                 # a name in beacon_kb.stores, defaults to the built-in
storage_dir = ".beacon-kb"

[[sources]]
type = "filesystem"              # a name in beacon_kb.connectors
path = "./docs"
glob = "**/*.md"

[embedder]
provider = "none"                # sparse-only BM25 by default, no embedder required
# provider = "local"             # activates dense retrieval via the local ONNX runtime extra
# provider = "remote"            # activates dense retrieval via a remote provider extra
# api_key_env = "OPENAI_API_KEY" # secret referenced by env-var name only, never inline

[retrieval]
top_k = 8
fusion = "rrf"
rerank = false                   # opt-in, names a beacon_kb.rerankers plugin when true

[answer]
enabled = true                   # answer() makes exactly one LLM call when a generator is configured
generator = "remote"             # requires a configured generator
abstain_when_no_evidence = true

[agentic]
enabled = false                  # behavioral optionality, no extra
planner = "identity"             # off by default
router = "all"                   # off by default
grader = "heuristic"             # deterministic default, zero LLM calls
reflection = false               # off by default
session = false                  # off by default
max_iterations = 4
max_llm_calls = 6
token_budget = 8000
```

### Offline-first degraded mode

Sparse-only BM25 retrieval is a first-class mode requiring no embedder.
`pipx install -> init -> index -> search` works fully offline with zero credentials and zero model downloads.
Dense retrieval activates when an embedder is configured, either the `local` extra (lightweight ONNX embedding runtime) or a `remote` provider extra with env-var credentials.
`answer()` requires a configured generator.

## Extras Matrix

| Extra | Pulls in | Enables |
|---|---|---|
| (base) | stdlib, NumPy, SQLite FTS5, agentic subpackage | offline BM25 search, single-shot answer wiring, budgeted investigate orchestration |
| html | HTML parsing dependency | HTML parser |
| pdf | PDF parsing dependency | page-aware PDF parser |
| web | HTTP client dependency | web-page connector |
| confluence | page-tree client dependency | confluence-style connector |
| remote | remote provider SDKs | remote embedding, generation, and rerank providers |
| local | lightweight ONNX embedding and rerank runtime | local dense embedding and local rerank, no credentials |
| mcp | MCP server dependency | serve-mcp command and MCP adapter |
| dev | lint, type, test, build tooling | development and CI |

There is no `agentic` extra.

## Key Decisions

1. One `KnowledgeBase` facade with `search()`, `answer()`, `investigate()` and hard cost contracts, no wrapper class and no mode flag, because cost guarantees must be legible at the API surface.
2. Agentic subpackage ships in the base install as pure orchestration with zero third-party deps, because it adds no dependencies and optionality is behavioral through config and `investigate()`.
3. `investigate()` lazily imports `beacon_kb.agentic`, because importing `beacon_kb` must have no agentic import and no side effects.
4. Sparse-only BM25 is a first-class offline mode requiring no embedder, because the five-minute first run must work with zero credentials and zero downloads.
5. Dense retrieval activates only when an embedder is configured via the `local` or `remote` extra, because heavy runtimes stay out of the base install.
6. One entry-point group per stage under `beacon_kb.` with names as documented constants, because group names are permanent API once third parties register against them.
7. Deterministic precedence with `PluginConflict` on duplicate names, because resolution must be predictable and never silently shadow.
8. Lazy discovery on first resolution, never at import, because import must have no side effects and unused plugins must not import heavy dependencies.
9. Built-ins register through the same registry path as plugins, because dogfooding the extension path prevents a privileged code path that rots.
10. A single integer `PLUGIN_API_VERSION` that plugins may pin against, because the registry must refuse incompatible plugins with a typed error.
11. `RetrievalPipeline.search` is the one retrieval primitive the loop reuses, because the degradation path to `answer()` and `search()` stays exact and citation logic is not duplicated.
12. Central budget enforcement with a graceful partial answer, because no strategy may run away and exhaustion must not crash the facade.
13. Deterministic heuristic evidence grader default with an LLM grader as an opt-in adapter, because the default path stays zero extra LLM calls and testable.
14. Always-on typed `AgenticTrace` returned by `investigate()`, because agentic decisions must be inspectable and deterministic under fakes.
15. Flat modules with package splits only past roughly 600 LOC, connectors in `connectors/`, provider adapters in `providers/`, no `api/` package, because a small legible surface beats deep nesting.
16. Tool schemas defined once and reused by in-process callables and the MCP server, because there must be one source of truth for tool schemas.
17. Two console scripts `beacon-kb` and `bkb` at one `main`, CLI command `ask` for facade `answer()`, because the short alias and terminal verb serve the tool-mode user without a second code path.
18. Secrets referenced by env-var name only in `beacon-kb.toml`, because a shared or committed config must never leak credentials.

## Design Pitfalls to Avoid

Carried from the Current State pitfall bullets in the source plans; each is a hard requirement.

- Do not commit sparse documents independently from vector and metadata persistence; one transaction controls visibility.
- Do not mutate an in-memory vector store that rewrites a separate JSON file; store embeddings in the transactional SQLite database.
- Do not truncate and rewrite a standalone JSON manifest; persist fingerprints and build-run state durably.
- Do not swallow index-write failures so stores drift apart; fail with typed errors and keep the prior active revision searchable.
- Do not derive chunk identity from random IDs; derive from corpus, canonical source, revision, pipeline fingerprint, parent locator, and child ordinal.
- Do not treat the documented overlap parameter as a minimum chunk length; implement real token overlap.
- Do not make LLM enrichment mandatory per chunk; enrichment is optional, cached, and failure-policy controlled.
- Do not hardcode provider batch limits in core logic; batching comes from the injected provider.
- Do not detect change from raw content hashes alone; include parser, chunker, enrichment, embedding model, embedding dimension, and schema versions in the fingerprint.
- Do not write an index version that change analysis never compares; compare fingerprints on every sync.
- Do not clear shared state per source on full rebuild; create a new corpus generation once.
- Do not lowercase parser output or drop content not attached to a subheading; preserve case, code, tables, headings, links, page numbers, anchors, and offsets.
- Do not entangle generic HTML extraction with site-specific cleanup; keep cleanup behind hooks.
- Do not let PDF heuristics silently misclassify headings, headers, and footers; emit typed warnings.
- Do not convert paths straight into framework documents or hardcode a fixed extension list; normalize identity and provenance in the connector without owning credentials.
- Do not let one connector own auth lookup, remote fetch, child traversal, and conversion at once; separate discovery, loading, and parsing.
- Do not search every field with equal weight or default missing distance metadata to zero; use weighted fields and a typed score contract.
- Do not reuse one rewritten query for both sparse and dense retrieval; keep the original question for lexical precision and record any rewrite separately.
- Do not combine incomparable BM25 and cosine scores by fixed weighting; use rank-based RRF.
- Do not deduplicate by exact document ID only; collapse content near-duplicates while preserving provenance.
- Do not add previous and next chunks unconditionally or assign invented relevance to context; expand only after final ordering under a token budget and keep `context_of` relationships.
- Do not skip a result-count or token recap before prompt construction; enforce the evidence budget.
- Do not rely on the model to preserve free-text citations; return evidence IDs and reject unknown IDs.
- Do not mix rewrite, retrieval, formatting, and generation in one method; keep stages separated.
- Do not return answer text while discarding structured evidence; preserve cited evidence in the response.
- Do not silently enable web search inside generation; the generator protocol has no hidden web-search flag.

## Epic, Feature, and Task Graph

Seven epics, twenty-four tasks, roughly 11,750 LOC.
Sizes: S is up to roughly 250 LOC, M is roughly 250 to 500 LOC, L is roughly 500 to 750 LOC.
Tracks let independent epics run in parallel: contracts and registry first (Track A), then storage plus ingestion (Track B) and retrieval (Track C) in parallel, agentic after retrieval (Track D), and CLI, evaluation, and release last (Track E).

### Epic 01: Standalone Package, Contracts, and Registry

Track A. Depends on: None.
Establish the standalone distribution, the full typed contract surface including agentic strategy protocols, typed config and errors, and the entry-point plugin registry.

- Feature 01.1: Foundation and Contracts
  - Task 01.1.1: Scaffold the standalone distribution and quality toolchain. S, 8 files, 250 LOC, deps: None.
    Root `pyproject.toml` with distribution `beacon-kb`, import `beacon_kb`, `src` layout, `requires-python >=3.11`, `py.typed`, the full extras matrix (html, pdf, web, confluence, remote, local, mcp, dev; no agentic extra), the `beacon-kb` and `bkb` console scripts at one main, the entry-point group declarations, Ruff/mypy/pytest/coverage config, and `.gitignore`.
  - Task 01.1.2: Define frozen domain models, typed IDs, errors, and all pipeline and agentic-strategy protocols. L, 4 files, 700 LOC, deps: 01.1.1.
    `models.py` frozen records and enums for corpus, source, revision, raw document, section, chunk, fingerprint, query, hit, evidence, citation, sync report, answer response, and the always-on AgenticTrace record with content-addressed typed IDs; `errors.py` with config, readiness, backend, ingestion, citation, plugin, budget, and agentic errors; `protocols.py` runtime-checkable Protocols for connectors, parsers, chunkers, embedders, stores, sparse/dense retrievers, fusion, rerankers, generators, token counters, and progress observers, plus QueryPlanner, EvidenceGrader, CorpusRouter, StopCondition, and SessionStore, each stating score direction, error contract, and determinism.
  - Task 01.1.3: Define typed config, the loader, and the facade shell with PLUGIN_API_VERSION. M, 5 files, 470 LOC, deps: 01.1.2.
    `config.py` frozen config tree (core, retrieval, answer, agentic, plugins) with validation, `config_loader.py` TOML plus env overlay with actionable diagnostics, `version.py` with `__version__` and `PLUGIN_API_VERSION`, `tokens.py` default heuristic counter and budget arithmetic, and `facade.py` shell exposing sync, search, answer, investigate, inspect, health that composes injected components without importing providers.
- Feature 01.2: Plugin Registry
  - Task 01.2.1: Implement the entry-point plugin registry. M, 5 files, 460 LOC, deps: 01.1.3.
    `registry/groups.py` canonical group constants and group-to-protocol map, `registry/discovery.py` lazy entry-point scanning and capability-metadata parsing with the PLUGIN_API_VERSION check, `registry/precedence.py` deterministic resolver and conflict detection raising PluginConflict, PluginNotFound, and ProtocolMismatch, and `registry/builtins.py` registering first-party components through the same path.
  - Task 01.2.2: Provide deterministic fakes and contract harnesses. M, 2 files, 340 LOC, deps: 01.1.2.
    `testing.py` deterministic embedder, generator, reranker, planner, grader, router, clock, and failure-injection fakes plus reusable per-protocol contract-test suites that any plugin author runs, and registry contract tests.

### Epic 02: Local Storage, Ingestion, and Incremental Indexing

Track B. Depends on: 01.
Transactional local store, source connectors with canonical identity, structure-aware parsers, parent/child chunking, and staged fingerprint-driven sync.

- Feature 02.1: Transactional Local Store
  - Task 02.1.1: Implement the transactional SQLite store with staged atomic promotion. L, 5 files, 600 LOC, deps: 01.2.1, 01.2.2.
    `storage/sqlite.py`, `storage/vector_math.py`, `storage/migrations/0001_initial.sql`, `indexing/manifest.py`; corpora, revisions, chunks, FTS5 BM25, embedding rows, build runs, fingerprints, active-revision pointers in one SQLite database with staged writes invisible until one promotion transaction, restart recovery, and store contract tests; registered as the default store.
- Feature 02.2: Sources and Parsing
  - Task 02.2.1: Implement source identity and filesystem and memory connectors. M, 5 files, 380 LOC, deps: 01.2.1, 01.2.2.
    `ingestion/identity.py`, `ingestion/media.py`, `connectors/filesystem.py`, `connectors/memory.py`; canonical URIs, content-addressed IDs, glob discovery, external-link mapping, media resolution, connector contract tests; registered as first-party connector plugins.
  - Task 02.2.2: Implement structure-aware Markdown, HTML, and PDF parsers. L, 6 files, 620 LOC, deps: 02.2.1.
    `parsing/base.py`, `parsing/markdown.py`, `parsing/html.py` (html extra), `parsing/pdf.py` (pdf extra) emitting typed sections with heading paths, anchors, page/offset locators, code, tables, links, and warnings; case preserved; safe fixtures; registered as parser plugins.
- Feature 02.3: Chunking, Embedding, and Sync
  - Task 02.3.1: Implement parent/child chunking, optional enrichment, and batched embeddings. L, 5 files, 560 LOC, deps: 02.2.2, 02.1.1.
    `ingestion/chunking.py`, `ingestion/enrichment.py`, `indexing/embedding.py`, `progress.py`; heading-aware parent/child chunks, real token overlap, deterministic parent/child/neighbor IDs, provider-owned batching, cached optional enrichment, structured progress; registered as the default chunker.
  - Task 02.3.2: Implement staged full and incremental synchronization. L, 5 files, 570 LOC, deps: 02.3.1.
    `ingestion/planning.py`, `ingestion/sync.py`, `indexing/coordinator.py`, `indexing/validation.py`; change classification, fingerprint invalidation, staged promotion, rollback, crash recovery, EMPTY/BUILDING/READY/FAILED health, typed SyncReport; wires facade sync.

### Epic 03: Hybrid Retrieval, Context, and Grounded Answers

Track C. Depends on: 01; parts of 02.
The zero-LLM search and one-LLM answer core the agentic loop reuses.

- Feature 03.1: Hybrid Retrieval and Context
  - Task 03.1.1: Implement sparse and dense candidate retrieval with typed scores. L, 6 files, 520 LOC, deps: 02.1.1.
    `retrieval/query.py`, `retrieval/sparse.py`, `retrieval/dense.py`, `retrieval/filters.py`; weighted FTS5 BM25 and dense vector retrieval with independent ranks and explicit score direction, original-question preservation, sparse-only degraded mode when no embedder is configured, consistent filters, diagnostics; registered under retriever groups.
  - Task 03.1.2: Add RRF fusion, optional reranking, and diversity. M, 3 files, 380 LOC, deps: 03.1.1.
    `retrieval/fusion.py`, `retrieval/rerank.py`, `retrieval/diversity.py`; rank-based RRF with deterministic tie-breaking, optional bounded reranking, near-duplicate collapse preserving provenance, retained component scores; registered under fusion and reranker groups.
  - Task 03.1.3: Assemble bounded context and expose RetrievalPipeline. M, 4 files, 360 LOC, deps: 03.1.2, 02.3.1.
    `retrieval/context.py`, `retrieval/snippets.py`, `retrieval/pipeline.py`; token-budgeted parent/neighbor expansion, match-centered snippets, stable [S1] evidence IDs distinguishing hits from context, and one `RetrievalPipeline.search(query, filters)` call reused by answer and investigate.
- Feature 03.2: Grounded Single-Shot Answers
  - Task 03.2.1: Generate grounded answers with validated citations and abstention. L, 5 files, 560 LOC, deps: 03.1.3, 02.3.2.
    `generation/answer.py`, `generation/citations.py`, `generation/prompts.py`, `generation/abstention.py`; zero-LLM search and exactly-one-LLM answer, evidence-ID citation validation, untrusted-context delimiters, deterministic pre-generation abstention, versioned prompts, complete diagnostics; wires facade search and answer.

### Epic 04: Agentic Layer

Track D. Depends on: 03; multi-corpus parts of 02.
The base-install agentic subpackage as pure orchestration over the Epic 03 pipeline, each feature independently degradable, lazily imported by investigate().

- Feature 04.1: Budgeted Loop and Evidence Grading
  - Task 04.1.1: Implement budgets, stop conditions, and the always-on trace. M, 3 files, 380 LOC, deps: 03.1.3, 03.2.1.
    `agentic/budget.py` LLM-call, retrieval, token, wall-clock, and marginal-gain ceilings, `agentic/trace.py` append-only replayable AgenticTrace, and StopCondition default strategies; budget arithmetic and trace shape testable with no LLM.
  - Task 04.1.2: Implement evidence grading and the reflect loop. L, 4 files, 560 LOC, deps: 04.1.1, 03.1.3.
    `agentic/grading.py` keep/discard/re-retrieve verdicts with the deterministic heuristic default and an LLM grader opt-in adapter, and `agentic/loop.py` retrieve-reflect-refine over RetrievalPipeline that never exceeds the budget, always emits a trace, and falls back to one retrieval when reflection is disabled.
- Feature 04.2: Planning, Routing, Memory, Synthesis, and Facade
  - Task 04.2.1: Implement query planning and multi-corpus routing. L, 4 files, 500 LOC, deps: 04.1.2, 01.2.1.
    `agentic/planner.py` decompose into an ordered subquery plan and `agentic/router.py` score and select corpora per subquery via registry-declared capabilities; both degrade to identity (one subquery, all corpora) when disabled.
  - Task 04.2.2: Implement session memory, synthesis, engine, and facade investigate. M, 4 files, 460 LOC, deps: 04.2.1, 03.2.1.
    `agentic/session.py` optional turn history and follow-up rewriting, `agentic/synthesis.py` cross-subquery evidence merge reusing generation, `agentic/orchestrator.py` AgenticEngine, and `facade.investigate(question, session, budget)` composing planner, router, loop, grader, session, synthesis behind one lazily imported call; integration test asserts investigate with all features off is behaviorally identical to answer.

### Epic 05: Tool Surface and MCP

Track E. Depends on: 03; 04.
Framework-neutral tools defined once plus an optional MCP server.

- Feature 05.1: Tool Surface and MCP Server
  - Task 05.1.1: Implement the framework-neutral tool surface and the optional MCP server extra. L, 5 files, 620 LOC, deps: 03.2.1, 04.2.2.
    `tools/schema.py` tool schemas defined once (search, fetch_evidence, answer, investigate, list_corpora), `tools/surface.py` in-process callables mapping schemas to facade methods with structured results and redaction, and `tools/mcp.py` reusing the same schemas and callables behind the mcp extra with corpus resource listing and graceful absence when the extra is not installed; wires the serve-mcp command surface.

### Epic 06: CLI Tool Mode

Track E. Depends on: 03; 04; parts of 01.
The CLI-first journey over a TOML config with doctor and journey tests.

- Feature 06.1: CLI Application and Commands
  - Task 06.1.1: Implement the CLI app, dispatch, renderers, config-driven init, and all commands including doctor and serve-mcp. L, 4 files, 720 LOC, deps: 01.1.3, 02.3.2, 03.2.1, 04.2.2, 05.1.1.
    `cli/app.py` dispatch with global flags (--config, --corpus, --json, --quiet, --no-color, --verbose) and the exit-code policy, `cli/render.py` human/plain/JSON renderers with progress, and `cli/commands.py` init scaffolding beacon-kb.toml with offline defaults, index with live progress, search zero-LLM, ask one-LLM, investigate printing the trace, inspect with the plugin capability report, doctor diagnosing config/extras/backend/credentials/readiness with exact-fix messages, plugins listing discovery and precedence, evaluate running gates, and serve-mcp when the mcp extra is present; CLI journey tests over a temp workspace.

### Epic 07: Evaluation, Provider Adapters, and Release

Track E. Depends on: 02; 03; 04; 05; 06.
Offline quality gates for core and agentic paths, provider and remote-source adapters, docs, and release.

- Feature 07.1: Evaluation and Quality Gates
  - Task 07.1.1: Build the offline evaluation and resilience suite with core gates. L, 7 files, 560 LOC, deps: 02.3.2, 03.2.1.
    `evaluation/metrics.py`, `evaluation/runner.py`, safe versioned corpus.jsonl, quality-gate tests enforcing Recall@5 >= 0.85, MRR@10 >= 0.75, citation validity 1.0, abstention accuracy >= 0.90, and a resilience matrix (restart, add/modify/delete, fingerprint migration, rollback, concurrency); an evaluate path.
  - Task 07.1.2: Add agentic evaluation and budget-regression gates. M, 3 files, 340 LOC, deps: 07.1.1, 04.2.2.
    `evaluation/agentic_metrics.py` iterations, budget usage, grading precision, and answer gain, plus gates asserting the loop never exceeds budget and investigate never scores below the single-shot answer baseline on the labeled corpus.
- Feature 07.2: Adapters, Docs, and Release
  - Task 07.2.1: Add remote and local providers, web, and confluence connectors as plugins. M, 4 files, 420 LOC, deps: 02.3.2, 03.2.1.
    `providers/remote.py` remote embedding/generation/rerank behind the remote extra, `providers/local.py` local ONNX embedding and rerank behind the local extra, `connectors/web.py` (web extra), and `connectors/confluence.py` (confluence extra) with injected clients, env-var credentials, offline contract tests, and entry-point registration.
  - Task 07.2.2: Finalize extras hygiene, docs, benchmark, and release. M, 8 files, 420 LOC, deps: 07.1.2, 07.2.1, 05.1.1, 06.1.1.
    Verify extras isolation and wheel/sdist contents (migrations included; no fixtures, caches, secrets, or local indexes), complete docs/ (architecture, configuration, cli, extending, plugins, agentic, operations, benchmark-report, quickstart), run core and agentic gates, and record the release checklist and accepted trade-offs.

### Track summary

- Track A: Epic 01 (contracts and registry), runs first.
- Track B: Epic 02 (storage, ingestion, indexing), starts after 01.
- Track C: Epic 03 (retrieval and answers), runs in parallel with 02 after 01, joining 02 at chunking and sync boundaries.
- Track D: Epic 04 (agentic), starts after 03.
- Track E: Epics 05 (tools and MCP), 06 (CLI), and 07 (evaluation and release) close out after 03 and 04, with 07 gating release.

## Open Decisions

- Confirm the specific lightweight ONNX embedding model shipped or fetched by the `local` extra and its wheel-size budget, since the offline dense-quality promise depends on it; the sparse-only mode is unaffected.
- Confirm the exact env-var names and provider SDK pins the `remote` extra targets before external release.
- Curate and approve the safe evaluation corpus labels before the recommended thresholds harden into release gates.
