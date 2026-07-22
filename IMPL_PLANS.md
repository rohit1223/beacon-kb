# Implementation Plans: beacon-kb - Standalone Agentic RAG Library
*plans-task-gap-plan-create | 2026-07-22 | agentic-rag-library*

> Plans cover: Epic 01 Standalone Package, Contracts, and Registry; Epic 02 Local Storage, Ingestion, and Incremental Indexing; Epic 03 Hybrid Retrieval, Context, and Grounded Answers; Epic 04 Agentic Layer; Epic 05 Tool Surface and MCP; Epic 06 CLI Tool Mode; and Epic 07 Evaluation, Provider Adapters, and Release.
> Source: EPICS.md

## Scope Guardrails

- Build only the standalone, pip-installable `beacon-kb` library plus its package and repository documentation.
- Do not build Jira, incident RCA, Slack crawling or triage, a hosted service, a browser UI, a scheduled crawler, or a web-search retriever inside this library.
- Ship the agentic subpackage in the base install with zero third-party dependencies, and keep its optionality behavioral through config and the explicit `investigate()` method; add no `agentic` extra.
- Reference every secret by env-var name only in `beacon-kb.toml`, never inline, so a shared or committed config never leaks credentials.
- Add no `AgenticKnowledgeBase` wrapper class, no `mode` flag on `answer()`, no `api/` package, and no deep nested public-surface package.

## Target Architecture

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

---

## Task 01.1.1: Scaffold the standalone distribution and quality toolchain
Epic: 01 - Standalone Package, Contracts, and Registry | Feature: 01.1 - Foundation and Contracts
Size: S | 8 files | ~250 LOC | Track: A

### Current State

No `pyproject.toml`, import package, or quality toolchain exists yet.
No `.gitignore` covers Python caches, virtual environments, wheels, sdists, or local index directories.
No console scripts, extras matrix, or entry-point group declarations exist.
Pitfall to avoid: do not add an `agentic` extra, because the agentic subpackage ships in the base install and its optionality is behavioral through config and the explicit `investigate()` method.
Pitfall to avoid: do not create an `api/` package or any deep nested public-surface package, because a small flat module surface is the standard.

### Desired State

The repository root holds one PEP 621 distribution named `beacon-kb` with import name `beacon_kb`, a `src/` layout, and `requires-python = ">=3.11"`.
The base install pulls only stdlib, NumPy, and SQLite FTS5 and declares html, pdf, web, confluence, remote, local, mcp, and dev as optional extras with no `agentic` extra.
Two console scripts `beacon-kb` and `bkb` point at one `main` in `beacon_kb.cli`.
The `pyproject.toml` declares every entry-point group under `beacon_kb.` for connectors, parsers, chunkers, embedders, stores, retrievers, fusion, rerankers, generators, token_counters, planners, graders, and routers, and declares no `stop_conditions` or `tools` group.
Ruff, mypy strict on `src`, pytest with markers, and coverage each have one documented configuration section.
The package ships a `py.typed` marker and imports with no side effects.

### Gap Analysis

- Missing: PEP 621 build metadata, extras matrix, dual console scripts, entry-point group declarations, tool configuration, and ignore patterns.
- Missing: the `py.typed` marker and a curated `__init__.py` with `__all__` and no import-time side effects.
- Changes: none, this is greenfield scaffolding at the repository root.
- Blockers: none.

### Implementation Research

Follow the PEP 621 `[build-system]`, `[project]`, and `[tool]` layout for metadata, extras, scripts, and entry points.
Declare entry-point groups in `[project.entry-points]` tables keyed by the exact `beacon_kb.<stage>` names that later become registry constants.
Keep the base dependency set minimal so `pipx install beacon-kb` works fully offline with zero credentials and zero model downloads.
Configure mypy in strict mode over `src` and register pytest markers for unit, contract, integration, plugins, cli, evaluation, and performance suites.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `pyproject.toml` | Declare PEP 621 metadata, the extras matrix, dual console scripts at one main, entry-point group declarations, and Ruff/mypy/pytest/coverage configuration. |
| CREATE | `README.md` | Introduce install modes, the five-minute offline quickstart, and the extras table. |
| CREATE | `LICENSE` | Declare the distribution license. |
| CREATE | `.gitignore` | Ignore Python caches, virtual environments, wheels, sdists, coverage, and local index directories. |
| CREATE | `src/beacon_kb/__init__.py` | Establish the import package with curated exports and `__all__` and no import-time side effects or agentic import. |
| CREATE | `src/beacon_kb/py.typed` | Mark the package as typed under PEP 561. |
| CREATE | `tests/conftest.py` | Provide deterministic fixtures and register the pytest markers. |
| CREATE | `tests/unit/__init__.py` | Establish the unit-test package layout. |

### Acceptance Criteria

- [ ] `python -m build` produces both a wheel and an sdist from a clean checkout.
- [ ] Installing the base wheel pulls no html, pdf, web, confluence, remote, local, or mcp dependency.
- [ ] `pyproject.toml` declares no `agentic` extra and no `stop_conditions` or `tools` entry-point group.
- [ ] `pyproject.toml` declares both `beacon-kb` and `bkb` console scripts pointing at the same `main`.
- [ ] `pyproject.toml` declares one entry-point group per stage under `beacon_kb.` for connectors, parsers, chunkers, embedders, stores, retrievers, fusion, rerankers, generators, token_counters, planners, graders, and routers.
- [ ] Importing `beacon_kb` performs no network, filesystem, logging-handler, or credential side effect and does not import `beacon_kb.agentic`.
- [ ] `python -m ruff check .`, `python -m mypy src`, and an empty pytest collection all succeed.

### Validation Steps

```bash
python -m build
python -m ruff check .
python -m mypy src
python -m pytest --collect-only
python -c "import beacon_kb; import sys; assert 'beacon_kb.agentic' not in sys.modules"
python -m zipfile -l dist/*.whl
```

---

## Task 01.1.2: Define frozen domain models, typed IDs, errors, and all pipeline and agentic-strategy protocols
Epic: 01 - Standalone Package, Contracts, and Registry | Feature: 01.1 - Foundation and Contracts
Size: L | 4 files | ~700 LOC | Track: A

### Current State

No domain models, typed IDs, errors, or protocols exist yet.
Pitfall to avoid: do not derive chunk identity from random IDs; derive it from corpus, canonical source, revision, pipeline fingerprint, parent locator, and child ordinal so identity is content-addressed and reproducible.
Pitfall to avoid: do not search every field with equal weight or default missing distance metadata to zero; the score contract must carry explicit direction so sparse, dense, fusion, and rerank scores stay separable.
Pitfall to avoid: do not return answer text while discarding structured evidence; the answer response record must preserve cited evidence with stable IDs.
Pitfall to avoid: do not give the generator protocol a hidden web-search flag; the protocol surface must forbid silent web retrieval.

### Desired State

`models.py` defines frozen records and enums for corpus, source, revision, raw document, section, chunk, fingerprint, query, hit, evidence, citation, sync report, answer response, and the always-on `AgenticTrace`, each carrying content-addressed typed IDs.
Every score field states its direction and range, and evidence carries stable `[S1]`-style identity distinguishing hits from context.
`errors.py` defines a typed hierarchy covering config, readiness, backend, ingestion, citation, plugin, budget, and agentic errors, including `PluginConflict`, `PluginNotFound`, and `ProtocolMismatch`.
`protocols.py` defines runtime-checkable `Protocol` contracts for connectors, parsers, chunkers, embedders, stores, sparse and dense retrievers, fusion, rerankers, generators, token counters, and progress observers, plus `QueryPlanner`, `EvidenceGrader`, `CorpusRouter`, `StopCondition`, and `SessionStore`.
Each protocol states its score direction, error contract, and determinism guarantees, and the `StopCondition` and tool protocols exist even though no entry-point group ships for them in v1.

### Gap Analysis

- Missing: frozen typed records, content-addressed typed IDs, and enums for every pipeline boundary.
- Missing: a typed error hierarchy including plugin, budget, and agentic errors.
- Missing: runtime-checkable protocols for every pipeline stage and every agentic strategy with documented score direction, error contract, and determinism.
- Changes: none, this is greenfield contract definition.
- Blockers: Task 01.1.1 must establish packaging and the typing marker.

### Implementation Research

Model records as frozen dataclasses with typed ID newtypes so identity cannot be confused across corpus, source, revision, chunk, and evidence boundaries.
Derive content-addressed IDs from stable inputs so identical content and fingerprint reproduce identical IDs across processes.
Define protocols with `runtime_checkable` so the registry can raise `ProtocolMismatch` listing missing members, and keep every score field optional with a documented direction.
Include `StopCondition` and tool protocols now even though their entry-point groups are deferred, so later versions add groups without changing the contract surface.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/models.py` | Define frozen domain records, enums, content-addressed typed IDs, and the always-on `AgenticTrace` record. |
| CREATE | `src/beacon_kb/errors.py` | Define the typed error hierarchy including config, readiness, backend, ingestion, citation, plugin, budget, and agentic errors. |
| CREATE | `src/beacon_kb/protocols.py` | Define runtime-checkable protocols for every pipeline stage and every agentic strategy with documented score direction, error contract, and determinism. |
| CREATE | `tests/unit/test_models.py` | Verify immutability, content-addressed ID reproducibility, enum validity, score-field direction, and protocol runtime-checkability. |

### Acceptance Criteria

- [ ] Every corpus, source, section, chunk, evidence item, build run, and revision has a stable typed identifier, and identical content plus fingerprint reproduces identical chunk, parent, and neighbor IDs across processes.
- [ ] Sparse, dense, fusion, and rerank scores occupy separate optional fields, each with a documented direction and range.
- [ ] Every domain record is frozen and returns structured objects, never preformatted Markdown.
- [ ] The answer response record preserves cited evidence with stable IDs alongside the answer text.
- [ ] The error hierarchy exposes `PluginConflict`, `PluginNotFound`, `ProtocolMismatch`, budget errors, and agentic errors as distinct typed classes.
- [ ] Every pipeline and agentic-strategy protocol is `runtime_checkable` and documents its score direction, error contract, and determinism, and the generator protocol exposes no web-search flag.
- [ ] `StopCondition` and the tool protocols are defined even though v1 ships no entry-point group for them.
- [ ] Importing `beacon_kb.models`, `beacon_kb.errors`, and `beacon_kb.protocols` performs no side effect.

### Validation Steps

```bash
python -m pytest tests/unit/test_models.py
python -m mypy src
python -m ruff check src/beacon_kb/models.py src/beacon_kb/errors.py src/beacon_kb/protocols.py
```

---

## Task 01.1.3: Define typed config, the loader, and the facade shell with PLUGIN_API_VERSION
Epic: 01 - Standalone Package, Contracts, and Registry | Feature: 01.1 - Foundation and Contracts
Size: M | 5 files | ~470 LOC | Track: A

### Current State

No typed config tree, config loader, version module, token counter, or facade shell exists yet.
Pitfall to avoid: do not import providers at module import or hold credential state in the facade; the facade must compose injected components with no provider import so importing `beacon_kb` stays side-effect free.
Pitfall to avoid: do not reference secrets inline in config; secrets are referenced by env-var name only so a shared or committed config never leaks credentials.
Pitfall to avoid: do not skip a token or result recap before prompt construction; the token counter and budget arithmetic must make the evidence budget enforceable.

### Desired State

`config.py` defines a frozen config tree with core, retrieval, answer, agentic, and plugins sections that mirror the `beacon-kb.toml` schema and validate on construction.
`config_loader.py` loads TOML, overlays env vars for secrets by name only, merges layers deterministically, and emits actionable diagnostics that name the failing key and the fix.
`version.py` is the single source of `__version__` and the integer `PLUGIN_API_VERSION`.
`tokens.py` provides the default heuristic `TokenCounter` and budget arithmetic helpers.
`facade.py` exposes a `KnowledgeBase` shell with `sync`, `search`, `answer`, `investigate`, `inspect`, and `health` that composes injected components without importing any provider and lazily imports the agentic subpackage only inside `investigate()`.

### Gap Analysis

- Missing: a frozen, validated config tree mirroring the TOML schema for shared tool-mode and library-mode use.
- Missing: a TOML plus env-overlay loader with actionable diagnostics and env-var-name-only secret references.
- Missing: `version.py` with `__version__` and `PLUGIN_API_VERSION`, and `tokens.py` heuristic counter plus budget arithmetic.
- Missing: the facade shell wiring the three cost-contract methods over injected components.
- Changes: none, this is greenfield.
- Blockers: Task 01.1.2 must define the models, errors, and protocols the config and facade reference.

### Implementation Research

Mirror each frozen config dataclass to a TOML table so tool mode and library mode validate against one model.
Overlay secrets from environment variables named in config, never inline values, and raise config errors that name the missing key and the exact fix.
Keep the facade a thin composition shell: it accepts injected components conforming to the protocols and holds the cost contracts of `search()` at zero LLM calls, `answer()` at exactly one, and `investigate()` at a budgeted loop.
Import `beacon_kb.agentic` lazily inside `investigate()` so importing the facade never imports the agentic subpackage.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/config.py` | Define the frozen config tree with core, retrieval, answer, agentic, and plugins sections and validation. |
| CREATE | `src/beacon_kb/config_loader.py` | Load TOML, overlay env-var secrets by name, merge layers, and emit actionable diagnostics. |
| CREATE | `src/beacon_kb/version.py` | Provide `__version__` and the integer `PLUGIN_API_VERSION`. |
| CREATE | `src/beacon_kb/tokens.py` | Provide the default heuristic `TokenCounter` and budget arithmetic helpers. |
| CREATE | `src/beacon_kb/facade.py` | Implement the `KnowledgeBase` shell with sync, search, answer, investigate, inspect, and health composing injected components without importing providers. |

### Acceptance Criteria

- [ ] The config tree is frozen, validates on construction, and mirrors the `beacon-kb.toml` core, retrieval, answer, agentic, and plugins sections.
- [ ] The loader references secrets by env-var name only and never accepts inline secret values.
- [ ] Invalid config raises a typed config error that names the failing key and the fix.
- [ ] `version.py` exposes both `__version__` and an integer `PLUGIN_API_VERSION`.
- [ ] The heuristic `TokenCounter` and budget arithmetic support a result-count and token recap before prompt construction.
- [ ] The facade exposes sync, search, answer, investigate, inspect, and health, composes injected components, and imports no provider at construction.
- [ ] Constructing the facade and importing `beacon_kb` do not import `beacon_kb.agentic`; only calling `investigate()` triggers the lazy import.

### Validation Steps

```bash
python -m pytest tests/unit/test_config.py tests/unit/test_config_loader.py tests/unit/test_facade_shell.py
python -m mypy src
python -c "import beacon_kb.facade, sys; assert 'beacon_kb.agentic' not in sys.modules"
```

---

## Task 01.2.1: Implement the entry-point plugin registry
Epic: 01 - Standalone Package, Contracts, and Registry | Feature: 01.2 - Plugin Registry
Size: M | 5 files | ~460 LOC | Track: A

### Current State

No plugin registry, entry-point discovery, precedence resolver, or built-in registration path exists yet.
Pitfall to avoid: do not let discovery run at import or load unused plugins; discovery must be lazy on first resolution so an installed-but-unused plugin with a heavy dependency never imports that dependency.
Pitfall to avoid: do not resolve duplicate names by last-installed-wins; two entry points registering the same name in the same group must raise `PluginConflict` with both distribution names.
Pitfall to avoid: do not create a privileged code path for first-party components; built-ins register through the same registry path as third-party plugins so no privileged path can rot.

### Desired State

`registry/groups.py` holds canonical entry-point group name constants and a group-to-protocol map for every shipped group.
`registry/discovery.py` scans entry points lazily via `importlib.metadata.entry_points(group=...)` on first resolution, parses capability metadata, enforces the `PLUGIN_API_VERSION` check, and exposes a `has_scanned()` predicate that stays false until the first resolution so tests can prove discovery never runs at import.
`registry/precedence.py` resolves a request in one fixed order: an explicit instance wins, then a config-named plugin by exact name, then a sole entry point when the group defines a default, then the built-in default.
The resolver raises `PluginConflict` on duplicate names with both distributions, `PluginNotFound` listing the group and installed names, `ProtocolMismatch` listing missing members, and a typed capability error when declared metadata such as vector dimension conflicts with configuration or when the plugin targets an incompatible major API line.
`registry/builtins.py` registers first-party components through the same registry path as plugins.

### Gap Analysis

- Missing: canonical group constants and the group-to-protocol map.
- Missing: lazy entry-point discovery, capability-metadata parsing, and the `PLUGIN_API_VERSION` check.
- Missing: the deterministic precedence resolver and conflict detection raising typed errors.
- Missing: first-party registration through the shared registry path.
- Changes: none, this is greenfield.
- Blockers: Task 01.1.3 must define config, version, and the facade the registry serves.

### Implementation Research

Read entry points with `importlib.metadata.entry_points(group=...)` and load an entry point only when its plugin is actually resolved, never at import.
Encode precedence as one documented ordered resolver so resolution is predictable and never silently shadows a component.
Validate capability metadata such as declared embedding dimension, supported media types, network need, and the targeted `PLUGIN_API_VERSION` before indexing begins, rejecting incompatible plugins with typed errors.
Register built-ins through `registry/builtins.py` on the same `resolve` path so the extension path is dogfooded.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/registry/__init__.py` | Expose the registry facade to resolve, register, list, and describe by group and name. |
| CREATE | `src/beacon_kb/registry/groups.py` | Define canonical entry-point group constants and the group-to-protocol map. |
| CREATE | `src/beacon_kb/registry/discovery.py` | Scan entry points lazily, parse capability metadata, and enforce the `PLUGIN_API_VERSION` check. |
| CREATE | `src/beacon_kb/registry/precedence.py` | Implement the deterministic resolver and conflict detection raising `PluginConflict`, `PluginNotFound`, and `ProtocolMismatch`. |
| CREATE | `src/beacon_kb/registry/builtins.py` | Register first-party components through the same registry path as plugins. |

### Acceptance Criteria

- [ ] Discovery runs lazily on first resolution and never at import, and an installed-but-unused plugin with a heavy dependency never imports that dependency.
- [ ] Resolution follows the fixed order: explicit instance, then config-named plugin by exact name, then a sole entry point when the group defines a default, then the built-in default.
- [ ] Two entry points registering the same name in the same group raise `PluginConflict` naming both distributions, with no last-installed-wins.
- [ ] A requested but uninstalled plugin raises `PluginNotFound` listing the group and the installed names.
- [ ] A resolved object that does not satisfy the target protocol raises `ProtocolMismatch` listing the missing members.
- [ ] A plugin whose declared capability metadata conflicts with configuration, or which targets an incompatible major `PLUGIN_API_VERSION`, is rejected with a typed error before indexing begins.
- [ ] Built-in components register through the same registry path as third-party plugins with no privileged code path.

### Validation Steps

```bash
python -m pytest tests/unit/registry tests/contract/test_registry_contract.py
python -m mypy src
python -c "import beacon_kb.registry; from beacon_kb.registry import discovery; assert not discovery.has_scanned(), 'importing the registry must not eagerly scan entry points'"
```

---

## Task 01.2.2: Provide deterministic fakes and contract harnesses
Epic: 01 - Standalone Package, Contracts, and Registry | Feature: 01.2 - Plugin Registry
Size: M | 4 files | ~340 LOC | Track: A

### Current State

No deterministic fakes, reusable contract-test harnesses, or registry contract tests exist yet.
Pitfall to avoid: do not hardcode provider batch limits in test fakes; the embedder fake must expose provider-owned batching so contract suites verify batching comes from the injected provider.
Pitfall to avoid: do not inject synthetic score metadata that masks contract mismatches; fakes must return scores with the declared direction so contract suites test the real contract.

### Desired State

`testing.py` provides deterministic embedder, generator, reranker, planner, grader, router, clock, and failure-injection fakes plus reusable per-protocol contract-test suites that any plugin author runs.
The fakes produce reproducible outputs under a fixed seed so traces, gradings, and answers are deterministic, and the clock and failure-injection fakes let tests drive budget arithmetic and rollback without a real LLM.
`tests/contract/test_registry_contract.py` exercises the registry against the fakes to prove precedence, conflict detection, protocol mismatch, capability rejection, and lazy discovery.
`tests/plugins/sample_plugin/` is a real, separately installable sample distribution with its own `pyproject.toml` and a `[project.entry-points."beacon_kb.connectors"]` table registering a connector, proving the out-of-tree third-party plugin path against `importlib.metadata` entry points rather than in-repo fake objects.
`tests/plugins/test_sample_plugin_discovery.py` installs that sample distribution into the test environment and asserts registry discovery of its entry point, its precedence relative to the built-in connector, and a `PluginConflict` when it registers a name that collides with a built-in.

### Gap Analysis

- Missing: deterministic fakes for every provider and agentic-strategy protocol plus a clock and failure-injection fake.
- Missing: reusable per-protocol contract-test suites a plugin author can run against a candidate implementation.
- Missing: registry contract tests binding the harness to the resolver.
- Missing: a real installable sample third-party distribution proving entry-point discovery, precedence, and conflicts against `importlib.metadata` rather than in-repo fakes.
- Changes: none, this is greenfield.
- Blockers: Task 01.1.2 must define the protocols the fakes and harnesses target.

### Implementation Research

Build each fake to satisfy its protocol exactly and to be deterministic under a fixed seed so every downstream suite is reproducible.
Expose the embedder fake's batch size as provider-owned so the embedding contract suite verifies core logic never hardcodes a batch limit.
Package the contract suites as importable parametrizable test bases so a third-party plugin author drops in an implementation and runs the same conformance checks.
Drive budget and rollback tests with the clock and failure-injection fakes so no real LLM or network is needed.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/testing.py` | Provide deterministic embedder, generator, reranker, planner, grader, router, clock, and failure-injection fakes plus reusable per-protocol contract-test suites. |
| CREATE | `tests/contract/test_registry_contract.py` | Verify registry precedence, conflict detection, protocol mismatch, capability rejection, and lazy discovery against the fakes. |
| CREATE | `tests/plugins/sample_plugin/` | Ship a real installable sample distribution with its own `pyproject.toml` and a `beacon_kb.connectors` entry point registering a connector. |
| CREATE | `tests/plugins/test_sample_plugin_discovery.py` | Install the sample distribution and assert entry-point discovery, precedence against the built-in connector, and `PluginConflict` on a colliding name. |

### Acceptance Criteria

- [ ] Each fake satisfies its target protocol and is `runtime_checkable`-compatible.
- [ ] Every fake is deterministic under a fixed seed, producing identical embeddings, gradings, plans, routes, and answers across runs.
- [ ] The embedder fake exposes provider-owned batching so the contract suite proves core logic never hardcodes a batch limit.
- [ ] The clock and failure-injection fakes drive budget arithmetic and rollback paths with no real LLM or network.
- [ ] The per-protocol contract suites are importable and parametrizable so a plugin author runs them against a candidate implementation.
- [ ] The registry contract test asserts precedence order, `PluginConflict`, `PluginNotFound`, `ProtocolMismatch`, capability rejection, and lazy discovery.
- [ ] The sample distribution under `tests/plugins/sample_plugin/` installs into the test environment and its connector is discovered through its `beacon_kb.connectors` entry point, proving the out-of-tree third-party path against `importlib.metadata`.
- [ ] The sample-plugin test asserts the installed distribution's precedence relative to the built-in connector and raises `PluginConflict` naming both distributions when it registers a colliding built-in name.
- [ ] Importing `beacon_kb.testing` performs no side effect.

### Validation Steps

```bash
python -m pip install -e tests/plugins/sample_plugin
python -m pytest tests/contract/test_registry_contract.py tests/plugins/test_sample_plugin_discovery.py
python -m mypy src
python -m ruff check src/beacon_kb/testing.py tests/contract/test_registry_contract.py
```

---

## Task 02.1.1: Implement the transactional SQLite store with staged atomic promotion
Epic: 02 - Local Storage, Ingestion, and Incremental Indexing | Feature: 02.1 - Transactional Local Store
Size: L | 5 files | ~600 LOC | Track: B

### Current State

- No storage backend, vector math, schema migration, or index manifest exists yet.
- A design pitfall to avoid: committing sparse documents independently from vector and metadata persistence, so one transaction must control visibility.
- A design pitfall to avoid: mutating an in-memory vector store that rewrites a separate JSON file, so embeddings live in the transactional SQLite database.
- A design pitfall to avoid: truncating and rewriting a standalone JSON manifest, so fingerprints and build-run state persist durably inside the database.
- A design pitfall to avoid: swallowing index-write failures so stores drift apart, so failures raise typed errors and keep the prior active revision searchable.
- A design pitfall to avoid: defaulting missing distance metadata to zero, so vectors carry a declared dimension and similarity direction rather than an inferred one.

### Desired State

- One SQLite database holds corpora, revisions, chunks, FTS5 BM25 rows, embedding rows, build runs, fingerprints, and active-revision pointers.
- A staged revision is invisible to readers until validation completes and one promotion transaction flips the active pointers.
- `storage/vector_math.py` validates normalized vectors and performs local NumPy similarity search with a declared similarity direction.
- `indexing/manifest.py` builds and validates index fingerprints and revision metadata from the persisted database state.
- The store registers as the default `sqlite` store through the same registry path as any plugin.

### Gap Analysis

- Missing: atomic cross-index visibility, versioned schema, active-revision state, restart recovery, and store contract tests.
- Changes: model isolation through an explicit corpus namespace and revision records rather than filesystem-path reconciliation or ad hoc JSON persistence.
- Blockers: Task 01.2.1 must provide the registry and Task 01.2.2 must provide the store contract harness and fakes.

### Implementation Research

- SQLite FTS5 supplies weighted BM25 candidates through the built-in `bm25()` rank function and column weighting.
- Promotion itself is the visibility boundary, so a partially persisted revision is never query-visible.
- Store embeddings with a model and dimension fingerprint and never infer similarity or distance from an untyped metadata key.
- Restart recovery reconstructs readiness and active revisions from durable state, never from an in-memory counter.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/storage/__init__.py | Export store implementations and migration helpers and register the default store. |
| CREATE | src/beacon_kb/storage/sqlite.py | Implement connections, transactions, active-revision promotion, FTS5 BM25, and embedding persistence. |
| CREATE | src/beacon_kb/storage/vector_math.py | Validate normalized vectors and perform local NumPy similarity search with a declared direction. |
| CREATE | src/beacon_kb/storage/migrations/0001_initial.sql | Define the versioned local schema, indexes, and FTS5 virtual table. |
| CREATE | src/beacon_kb/indexing/manifest.py | Build and validate index fingerprints and revision metadata from persisted state. |

### Acceptance Criteria

- [ ] Sparse rows, vector rows, manifest rows, and active-revision pointers become visible atomically in one promotion transaction.
- [ ] A staged revision is invisible to readers until promotion and rollback leaves the previously active corpus fully searchable.
- [ ] FTS5 capability and vector dimension are checked at startup with typed backend errors rather than silent defaults.
- [ ] Two corpus namespaces with identical source paths never read or mutate each other's records.
- [ ] Closing and reopening the store preserves active data, build status, and query results without an in-memory counter.
- [ ] A swallowed index-write is impossible: a failed write raises a typed error and keeps the prior active revision searchable.
- [ ] The store passes the reusable store contract suite from `testing.py` and registers as the default `sqlite` store through the registry.

### Validation Steps

```bash
python -m pytest tests/contract/test_knowledge_store_contract.py tests/unit/storage/test_sqlite_store.py
python -m mypy src
```

---

## Task 02.2.1: Implement source identity and filesystem and memory connectors
Epic: 02 - Local Storage, Ingestion, and Incremental Indexing | Feature: 02.2 - Sources and Parsing
Size: M | 5 files | ~380 LOC | Track: B

### Current State

- No source connectors, canonical-identity policy, or media resolution exists yet.
- A design pitfall to avoid: converting paths straight into framework documents or hardcoding a fixed extension list, so the connector normalizes identity and provenance without owning credentials.
- A design pitfall to avoid: letting one connector own auth lookup, remote fetch, child traversal, and conversion at once, so discovery, loading, and parsing stay separate.

### Desired State

- `ingestion/identity.py` produces canonical source URIs and stable content-addressed source and revision IDs independent of the current working directory.
- `ingestion/media.py` resolves media types and emits parser-selection hints without parsing.
- `connectors/filesystem.py` discovers files and globs, maps external links, and loads bytes or text without parsing or indexing.
- `connectors/memory.py` supplies deterministic documents for tests and embedding applications.
- Both connectors register as first-party connector plugins through the registry path.

### Gap Analysis

- Missing: connector contracts, canonical-identity policy, reusable in-memory fixtures, external-link mapping, and connector conformance tests.
- Changes: separate link mapping and media resolution from parsing and avoid absolute-versus-relative path matching heuristics.
- Blockers: Task 01.2.1 must provide the registry and Task 01.2.2 must provide the connector contract harness.

### Implementation Research

- Source descriptors carry display names and external citation links as first-class concepts.
- Filesystem sources support glob patterns and configurable external-link mapping with deterministic ordering.
- Credentials and client construction stay caller-owned, so a connector receives injected clients rather than reading secrets.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/ingestion/identity.py | Canonicalize source URIs and generate stable content-addressed source and revision IDs. |
| CREATE | src/beacon_kb/ingestion/media.py | Resolve media types and parser-selection hints without parsing. |
| CREATE | src/beacon_kb/connectors/__init__.py | Export connector implementations and register first-party connectors through the registry. |
| CREATE | src/beacon_kb/connectors/filesystem.py | Discover and load configured files, directories, and globs with external-link mapping. |
| CREATE | src/beacon_kb/connectors/memory.py | Supply deterministic in-memory documents for tests and embedding applications. |

### Acceptance Criteria

- [ ] Repeated scans produce identical canonical source IDs regardless of the current working directory.
- [ ] Source content changes alter the revision ID but not the logical source ID.
- [ ] Filesystem discovery supports glob patterns with deterministic ordering and never hardcodes a fixed extension list.
- [ ] External citation links are derived without leaking local absolute paths when a mapping is configured.
- [ ] Connector errors identify the source and operation without including credentials.
- [ ] Discovery, loading, and parsing stay separate: the connector loads bytes or text and never parses or indexes.
- [ ] Both connectors pass the reusable connector contract suite and register as first-party connector plugins.

### Validation Steps

```bash
python -m pytest tests/contract/test_source_connector_contract.py tests/unit/ingestion/test_filesystem_source.py
python -m mypy src
```

---

## Task 02.2.2: Implement structure-aware Markdown, HTML, and PDF parsers
Epic: 02 - Local Storage, Ingestion, and Incremental Indexing | Feature: 02.2 - Sources and Parsing
Size: L | 6 files | ~620 LOC | Track: B

### Current State

- No parsers, shared section helpers, or safe fixtures exist yet.
- A design pitfall to avoid: lowercasing parser output or dropping content not attached to a subheading, so case, code, tables, headings, links, page numbers, anchors, and offsets are preserved.
- A design pitfall to avoid: entangling generic HTML extraction with site-specific cleanup, so cleanup stays behind hooks.
- A design pitfall to avoid: letting PDF heuristics silently misclassify headings, headers, and footers, so the PDF parser emits typed warnings.

### Desired State

- `parsing/base.py` provides section and provenance helpers shared by every parser.
- `parsing/markdown.py` preserves case, fenced code, tables, links, and heading paths as the default parser.
- `parsing/html.py` performs generic extraction with cleanup hooks behind the `html` extra.
- `parsing/pdf.py` records page-level provenance and emits typed warnings behind the `pdf` extra.
- Parsers emit typed sections with heading paths, anchors, and page or offset locators and register as parser plugins.

### Gap Analysis

- Missing: typed parser output, consistent structural locators, parse-warning contracts, cross-format fixtures, and extra isolation.
- Changes: extract rich structural metadata without forced lowercasing or silent original-document fallback.
- Blockers: Task 02.2.1 must provide connectors, identity, and media resolution.

### Implementation Research

- HTML fixtures cover headings, URLs, code, and table metadata and heading-path normalization needs dedicated tests.
- LLM table summarization belongs to optional enrichment, not deterministic parsing.
- Base-package import succeeds when the `html` and `pdf` extras are absent, so those parsers import their dependency lazily.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/parsing/__init__.py | Export parser types and register parsers through the registry without importing optional dependencies. |
| CREATE | src/beacon_kb/parsing/base.py | Provide shared section and provenance helpers for all parsers. |
| CREATE | src/beacon_kb/parsing/markdown.py | Parse headings, anchors, prose, tables, links, and fenced code with case preserved. |
| CREATE | src/beacon_kb/parsing/html.py | Parse semantic sections, links, tables, and code with cleanup hooks behind the html extra. |
| CREATE | src/beacon_kb/parsing/pdf.py | Parse page-aware PDF text with page provenance and typed warnings behind the pdf extra. |
| CREATE | tests/fixtures/documents/sample.md | Supply safe Markdown regression content covering headings, code, tables, and links. |

### Acceptance Criteria

- [ ] Parsers preserve original case, commands, code blocks, links, tables, and source text needed for citation.
- [ ] Every emitted section has a source URI and at least one stable structural locator: heading or anchor, page, or character span.
- [ ] Unsupported or malformed content returns typed warnings or errors and is never silently indexed as empty text.
- [ ] PDF heading, header, and footer misclassification surfaces as a typed warning rather than a silent guess.
- [ ] Generic HTML extraction is separate from site-specific cleanup, which stays behind hooks.
- [ ] Base-package import succeeds when the html and pdf extras are absent.
- [ ] Fixtures are newly authored safe content containing no proprietary documentation or secrets, and parsers register as plugins.

### Validation Steps

```bash
python -m pytest tests/unit/parsing
python -m mypy src
```

---

## Task 02.3.1: Implement parent/child chunking, optional enrichment, and batched embeddings
Epic: 02 - Local Storage, Ingestion, and Incremental Indexing | Feature: 02.3 - Chunking, Embedding, and Sync
Size: L | 5 files | ~560 LOC | Track: B

### Current State

- No chunking, enrichment, embedding, or progress pipeline exists yet.
- A design pitfall to avoid: deriving chunk identity from random IDs, so identity derives from corpus, canonical source, revision, pipeline fingerprint, parent locator, and child ordinal.
- A design pitfall to avoid: treating the documented overlap parameter as a minimum chunk length, so real token overlap is implemented.
- A design pitfall to avoid: making LLM enrichment mandatory per chunk, so enrichment is optional, cached, and failure-policy controlled.
- A design pitfall to avoid: hardcoding provider batch limits in core logic, so batching comes from the injected provider.

### Desired State

- `ingestion/chunking.py` produces heading-aware parent and child chunks with real token overlap and deterministic parent, child, and neighbor IDs.
- `ingestion/enrichment.py` orchestrates optional enrichment cached by content plus prompt and model version under a failure policy.
- `indexing/embedding.py` batches, validates, retries, and caches embeddings through the Embedder protocol with provider-owned batch sizes.
- `progress.py` emits structured stage and progress events plus a logging and TTY-neutral adapter.
- The default chunker registers through the registry path.

### Gap Analysis

- Missing: real overlap, parent and child identity, provider-neutral batching, an enrichment cache, structured progress, and deterministic failure tests.
- Changes: keep summaries, keywords, and FAQs as optional searchable metadata instead of embedding prerequisites.
- Blockers: Task 02.2.2 must provide parsers and Task 02.1.1 must provide the store and manifest.

### Implementation Research

- Neighbor links are generated only after stable ordering and IDs exist.
- Sparse retrieval may search both parent and child levels while dense retrieval defaults to children.
- Long stages emit start, end, elapsed time, and `current/total` progress through the structured progress adapter.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/ingestion/chunking.py | Implement heading-aware parent and child chunks, real overlap, stable IDs, and neighbor links. |
| CREATE | src/beacon_kb/ingestion/enrichment.py | Orchestrate optional cached enrichment with a failure policy. |
| CREATE | src/beacon_kb/indexing/embedding.py | Batch, validate, retry, and cache embeddings through the Embedder protocol. |
| CREATE | src/beacon_kb/progress.py | Define structured stage and progress events plus a logging and TTY-neutral adapter. |
| CREATE | tests/unit/ingestion/test_chunking.py | Verify boundaries, real overlap, deterministic IDs, parent and neighbor links, and code preservation. |

### Acceptance Criteria

- [ ] Consecutive child chunks share the configured token overlap and never split inside a fenced code block when avoidable.
- [ ] Identical input and fingerprint produce identical parent, child, previous, and next IDs across processes.
- [ ] Chunk identity derives from corpus, canonical source, revision, pipeline fingerprint, parent locator, and child ordinal, never a random ID.
- [ ] Ingestion succeeds with enrichment disabled and with an enrichment-provider failure under the configured best-effort policy.
- [ ] Batch sizes never exceed the injected provider contract and are never hardcoded in core logic.
- [ ] Every long stage emits start, end, elapsed time, and `current/total` progress through the structured adapter.
- [ ] The default chunker registers through the registry path.

### Validation Steps

```bash
python -m pytest tests/unit/ingestion/test_chunking.py tests/unit/ingestion/test_enrichment.py tests/contract/test_embedding_provider_contract.py
python -m mypy src
```

---

## Task 02.3.2: Implement staged full and incremental synchronization
Epic: 02 - Local Storage, Ingestion, and Incremental Indexing | Feature: 02.3 - Chunking, Embedding, and Sync
Size: L | 5 files | ~570 LOC | Track: B

### Current State

- No synchronization lifecycle, change planner, coordinator, or validation stage exists yet.
- A design pitfall to avoid: detecting change from raw content hashes alone, so the fingerprint includes parser, chunker, enrichment, embedding model, embedding dimension, and schema versions.
- A design pitfall to avoid: writing an index version that change analysis never compares, so fingerprints are compared on every sync.
- A design pitfall to avoid: clearing shared state per source on full rebuild, so a full rebuild creates a new corpus generation once.
- A design pitfall to avoid: swallowing index-write failures so stores drift apart, so a failed stage keeps the previous revision active.

### Desired State

- `ingestion/planning.py` classifies each source revision as unchanged, new, changed, deleted, or pipeline-incompatible.
- `ingestion/sync.py` scans, parses, chunks, enriches, embeds, stages, validates, and atomically promotes as one recoverable operation.
- `indexing/coordinator.py` coordinates sparse, vector, and metadata writes inside one revision transaction.
- `indexing/validation.py` validates counts, IDs, dimensions, links, and fingerprint consistency before promotion.
- Sync exposes EMPTY, BUILDING, READY, and FAILED health and returns a typed `SyncReport`, and it wires the facade sync path.

### Gap Analysis

- Missing: fingerprint invalidation, staged promotion, crash recovery, idempotency, source-level failure policy, and reliable readiness state.
- Changes: replace callback counters and mutable status with persisted build-run state and explicit health semantics.
- Blockers: Task 02.3.1 must provide chunking, embedding, enrichment, and progress.

### Implementation Research

- Every sync result reports new, modified, deleted, and unchanged sources.
- Promotion is the visibility boundary, so partially persisted revisions are never readable.
- A restart with unchanged sources never produces an empty index because readiness is reconstructed from durable state.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/ingestion/planning.py | Compute source and pipeline change sets from compared fingerprints. |
| CREATE | src/beacon_kb/ingestion/sync.py | Orchestrate scan, parse, chunk, enrich, embed, stage, validate, and promote as one operation. |
| CREATE | src/beacon_kb/indexing/coordinator.py | Coordinate sparse, vector, and metadata writes inside one revision transaction. |
| CREATE | src/beacon_kb/indexing/validation.py | Validate counts, IDs, dimensions, links, and fingerprint consistency before promotion. |
| CREATE | tests/integration/test_sync_lifecycle.py | Verify empty, full, unchanged, add, modify, delete, rollback, and restart behavior. |

### Acceptance Criteria

- [ ] An unchanged second sync performs zero parsing, enrichment, embedding, and index writes.
- [ ] New, changed, and deleted sources update both sparse and dense views after one promotion.
- [ ] Any parser, chunker, enrichment, embedding, or schema fingerprint change triggers the documented affected reindex scope, and fingerprints are compared on every sync.
- [ ] Simulated failure at every stage leaves the previous active corpus searchable and a recoverable failed build record.
- [ ] Restart reconstructs readiness and active revisions from durable state without relying on an in-memory counter.
- [ ] A full rebuild creates a new corpus generation once rather than clearing shared state per source.
- [ ] Sync returns a typed `SyncReport` with counts, timings, warnings, fingerprints, failed sources, and active build identity, and exposes EMPTY, BUILDING, READY, and FAILED health.

### Validation Steps

```bash
python -m pytest tests/integration/test_sync_lifecycle.py tests/integration/test_sync_rollback.py tests/integration/test_fingerprint_migration.py tests/integration/test_multi_source_concurrency.py
python -m mypy src
```

---

## Task 03.1.1: Implement sparse and dense candidate retrieval with typed scores
Epic: 03 - Hybrid Retrieval, Context, and Grounded Answers | Feature: 03.1 - Hybrid Retrieval and Context
Size: L | 6 files | ~520 LOC | Track: C

### Current State

- No retrieval package, query policy, sparse retriever, dense retriever, or filter layer exists yet.
- The transactional SQLite store, its weighted FTS5 BM25 tables, embedding rows, and active-revision pointers already exist from Task 02.1.1 and are the only backend these retrievers read.
- Pitfall to avoid: do not search every field with equal weight or default missing distance metadata to zero; use weighted fields and a typed score contract.
- Pitfall to avoid: do not reuse one rewritten query for both sparse and dense retrieval; keep the original question for lexical precision and record any rewrite separately.
- Pitfall to avoid: do not query a vector store without a typed score contract or infer similarity direction from an untyped metadata key.

### Desired State

- `retrieval/query.py` validates the incoming query, preserves the original question verbatim for lexical precision, and selects sparse and dense query variants as independent, separately recorded values.
- `retrieval/sparse.py` runs weighted FTS5 BM25 over the active revision with exact-token boosts for error codes, command names, identifiers, headings, and code fields, returning independent ranks with explicit score direction.
- `retrieval/dense.py` embeds the query through the injected Embedder, retrieves candidates from embedding rows with a declared similarity semantic, and returns independent ranks, degrading to no candidates when no embedder is configured.
- Sparse-only degraded mode is first class: with no embedder configured, retrieval returns BM25 candidates alone with zero downloads and zero credentials.
- `retrieval/filters.py` applies provider-neutral namespace, ACL, source, tag, media, and date filters consistently before candidates leave either retriever.
- Both retrievers resolve through the `beacon_kb.retrievers` group so a third-party retriever is a first-party-equivalent plugin.

### Gap Analysis

- Missing: query validation and variant selection, weighted BM25 execution, dense candidate retrieval with a typed score contract, provider-neutral filters, and per-stage diagnostics.
- Changes: none; this is greenfield within the existing store.
- Blockers: Task 02.1.1 must provide the active-revision FTS5 and embedding-row store and its declared similarity semantics.

### Implementation Research

- SQLite FTS5 supports per-column weighting through the `bm25()` rank arguments, so field weights are expressed at query time rather than by duplicating text.
- The declared similarity direction comes from the store's typed contract, never from an untyped metadata key, so missing distance metadata is a typed error rather than a silent zero.
- Query rewriting is an optional, separately observable step; the original question always drives sparse retrieval for exact-token precision.
- The sparse-only path is the offline default and must return typed readiness or backend errors rather than empty results on a missing or dimension-incompatible index.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/retrieval/__init__.py | Export retrieval package surface without importing providers. |
| CREATE | src/beacon_kb/retrieval/query.py | Validate queries, select sparse and dense variants, and preserve the original question. |
| CREATE | src/beacon_kb/retrieval/sparse.py | Execute weighted FTS5 BM25 with exact-token boosts and typed score direction. |
| CREATE | src/beacon_kb/retrieval/dense.py | Embed queries and retrieve declared-similarity candidates with sparse-only fallback. |
| CREATE | src/beacon_kb/retrieval/filters.py | Apply provider-neutral namespace, ACL, source, tag, media, and date filters. |
| CREATE | tests/unit/retrieval/test_sparse_dense.py | Verify weighting, exact boosts, variant separation, similarity, filters, and empty indexes. |

### Acceptance Criteria

- [ ] Sparse and dense candidates retain independent ranks and raw scores with an explicit score direction and no cross-normalization.
- [ ] The original question is preserved and any sparse or dense rewrite is recorded as a separate, independently tested value.
- [ ] Weighted BM25 boosts exact error codes and technical identifiers without altering the dense candidate ordering.
- [ ] With no embedder configured, retrieval returns sparse-only BM25 candidates offline with zero downloads and zero credentials.
- [ ] Namespace, ACL, source, tag, media, and date filters apply consistently and cannot be bypassed by either retriever.
- [ ] A missing, empty, or dimension-incompatible index returns a typed readiness or backend error rather than a silent zero score.
- [ ] Both retrievers are discovered and resolved through the `beacon_kb.retrievers` group by the same path as third-party plugins.

### Validation Steps

```bash
python -m pytest tests/unit/retrieval/test_sparse_dense.py
python -m mypy src
python -m ruff check .
```

---

## Task 03.1.2: Add RRF fusion, optional reranking, and diversity
Epic: 03 - Hybrid Retrieval, Context, and Grounded Answers | Feature: 03.1 - Hybrid Retrieval and Context
Size: M | 3 files | ~380 LOC | Track: C

### Current State

- No fusion, reranking, or diversity stage exists yet.
- Independent sparse and dense ranked candidate lists with typed scores already exist from Task 03.1.1 and are the only input to this stage.
- Pitfall to avoid: do not combine incomparable BM25 and cosine scores by fixed weighting; use rank-based RRF.
- Pitfall to avoid: do not deduplicate by exact document ID only; collapse content near-duplicates while preserving provenance.

### Desired State

- `retrieval/fusion.py` combines sparse and dense ranks with Reciprocal Rank Fusion and deterministic tie-breaking, treating raw scores as diagnostics only and never as calibrated inputs.
- `retrieval/rerank.py` optionally scores only the bounded fused candidate window through an injected Reranker, records its latency and score separately, and returns the fused order unchanged when the reranker is absent or fails under a best-effort policy.
- `retrieval/diversity.py` collapses content near-duplicates while preserving each source's provenance and optionally applies MMR-style diversity, never merging chunks from different sources merely because their text is similar.
- Every component rank and score is retained through the stage for diagnosis; fusion, rerank, and diversity affect only final ordering.
- Fusion and rerankers resolve through the `beacon_kb.fusion` and `beacon_kb.rerankers` groups.

### Gap Analysis

- Missing: rank-based fusion with deterministic ties, bounded optional reranking, content near-duplicate collapse, and retained component scores.
- Changes: none; this is greenfield over the Task 03.1.1 candidate lists.
- Blockers: Task 03.1.1 must provide independent sparse and dense ranks with typed scores.

### Implementation Research

- Reciprocal Rank Fusion combines multiple ranked lists using ranks alone, so incomparable BM25 and cosine scales never require a fixed weighting.
- Retrieve a larger candidate set before applying a bounded reranker to a smaller window, keeping rerank cost and latency optional.
- Near-duplicate collapse compares content while carrying every collapsed item's provenance, so a similar-text chunk from a different source is never silently discarded.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/retrieval/fusion.py | Implement rank-based RRF with deterministic tie-breaking and retained component scores. |
| CREATE | src/beacon_kb/retrieval/rerank.py | Invoke an optional reranker over a bounded window with separate score and latency. |
| CREATE | src/beacon_kb/retrieval/diversity.py | Collapse content near-duplicates preserving provenance and optionally diversify. |

### Acceptance Criteria

- [ ] Fused order depends only on ranks and is stable and deterministic for identical inputs and configuration.
- [ ] RRF parameters, candidate counts, and every component rank and score appear in diagnostics and are never discarded by fusion.
- [ ] Reranker absence or a best-effort reranker failure returns the fused order unchanged with the failure recorded.
- [ ] Reranking scores only the bounded fused window and records its latency and score separately from the fusion score.
- [ ] Near-duplicate collapse never merges chunks from different sources solely because their text is similar and preserves each collapsed item's provenance.
- [ ] Fusion and rerankers are discovered and resolved through the `beacon_kb.fusion` and `beacon_kb.rerankers` groups.

### Validation Steps

```bash
python -m pytest tests/unit/retrieval/test_fusion_rerank_diversity.py
python -m mypy src
python -m ruff check .
```

---

## Task 03.1.3: Assemble bounded context and expose RetrievalPipeline
Epic: 03 - Hybrid Retrieval, Context, and Grounded Answers | Feature: 03.1 - Hybrid Retrieval and Context
Size: M | 4 files | ~360 LOC | Track: C

### Current State

- No context assembly, snippet construction, or unified retrieval pipeline exists yet.
- Fused, reranked, and diversified candidates exist from Task 03.1.2, and deterministic parent, child, and neighbor IDs exist from Task 02.3.1.
- Pitfall to avoid: do not add previous and next chunks unconditionally or assign invented relevance to context; expand only after final ordering under a token budget and keep `context_of` relationships.
- Pitfall to avoid: do not skip a result-count or token recap before prompt construction; enforce the evidence budget.
- Pitfall to avoid: do not fall back to a document's first N characters; center snippets on the match.

### Desired State

- `retrieval/context.py` expands parents and neighbors only after final candidate ordering, allocates evidence under a token budget that reserves prompt overhead, and keeps `context_of` relationships without inventing relevance scores for context spans.
- `retrieval/snippets.py` builds match-centered, locator-preserving snippets that retain source URI, title, heading path or page, and character span rather than a document prefix.
- `retrieval/pipeline.py` exposes one deterministic `RetrievalPipeline.search(query, filters)` that runs query policy, sparse and dense retrieval, fusion, optional rerank, diversity, context, and snippets, returning `Evidence[]` with stable `[S1]`-style IDs that distinguish primary hits from context.
- The single `RetrievalPipeline.search` call is the one retrieval primitive later reused by `answer()` and `investigate()`, so citation logic is never duplicated.

### Gap Analysis

- Missing: token-budgeted parent and neighbor expansion, match-centered snippets, stable evidence IDs, and one reusable pipeline entry point.
- Changes: none; this is greenfield over the Task 03.1.2 output and Task 02.3.1 identity.
- Blockers: Task 03.1.2 must provide final ordering and Task 02.3.1 must provide deterministic parent, child, and neighbor IDs.

### Implementation Research

- Parent and neighbor links come from the chunking stage, so context reconstruction resolves stored relationships rather than re-splitting text.
- The token counter is a protocol because answer models tokenize the same evidence differently, so the budget is measured with the configured counter.
- A single `RetrievalPipeline.search` primitive keeps the degradation path from `search()` to `answer()` to `investigate()` exact and prevents a second retrieval stack.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/retrieval/context.py | Select parents and neighbors and pack evidence under a token budget keeping `context_of`. |
| CREATE | src/beacon_kb/retrieval/snippets.py | Build match-centered, locator-preserving snippets. |
| CREATE | src/beacon_kb/retrieval/pipeline.py | Expose one deterministic `RetrievalPipeline.search(query, filters)` returning cited evidence. |
| CREATE | tests/integration/test_retrieval_pipeline.py | Verify budgets, expansion, snippets, evidence IDs, and hit-versus-context distinction from an active index. |

### Acceptance Criteria

- [ ] Packed evidence never exceeds the configured budget under the selected token counter and always includes a result-count and token recap before prompt construction.
- [ ] Parent and neighbor expansion occurs only after final candidate ordering and cannot grow unbounded.
- [ ] Context spans keep `context_of` relationships and are never assigned invented relevance scores.
- [ ] Snippets center the match or selected span rather than returning a document prefix and preserve source URI, title, structural locator, and span.
- [ ] Every evidence item has a stable `[S1]`-style ID resolving to an active source revision, and context-only spans are distinguishable from primary retrieved hits.
- [ ] `RetrievalPipeline.search(query, filters)` is deterministic for identical inputs and is the single retrieval path reused by answer and investigate.

### Validation Steps

```bash
python -m pytest tests/integration/test_retrieval_pipeline.py
python -m mypy src
python -m ruff check .
```

---

## Task 03.2.1: Generate grounded answers with validated citations and abstention
Epic: 03 - Hybrid Retrieval, Context, and Grounded Answers | Feature: 03.2 - Grounded Single-Shot Answers
Size: L | 5 files | ~560 LOC | Track: C

### Current State

- No answer generation, citation validation, prompt versioning, or abstention policy exists yet.
- The single `RetrievalPipeline.search` primitive exists from Task 03.1.3 and the complete indexed corpus with readiness state exists from Task 02.3.2.
- Pitfall to avoid: do not rely on the model to preserve free-text citations; return evidence IDs and reject unknown IDs.
- Pitfall to avoid: do not mix rewrite, retrieval, formatting, and generation in one method; keep stages separated.
- Pitfall to avoid: do not return answer text while discarding structured evidence; preserve cited evidence in the response.
- Pitfall to avoid: do not silently enable web search inside generation; the generator protocol has no hidden web-search flag.

### Desired State

- `generation/answer.py` composes optional rewrite, `RetrievalPipeline.search`, abstention gate, grounded generation, and citation validation as separate stages, so `search()` performs zero LLM calls and `answer()` performs exactly one.
- `generation/citations.py` resolves and validates every evidence ID and citation locator against the evidence in the same response and rejects unknown IDs so free-text citations cannot escape validation.
- `generation/prompts.py` holds versioned grounded prompts that delimit retrieved content as untrusted context that cannot alter system instructions.
- `generation/abstention.py` applies a deterministic pre-generation abstention policy for no-evidence and below-policy cases without calling the generator, plus a post-generation policy that converts invalid output to a typed grounded failure or safe abstention.
- The facade `search` and `answer` paths are wired to this stage, preserving cited structured evidence in every response and recording prompt version, provider, query variants, timings, and token counts without secrets.

### Gap Analysis

- Missing: staged answer orchestration, evidence-ID citation schema and validation, untrusted-context delimiters, deterministic abstention, versioned prompts, and complete diagnostics.
- Changes: none; this is greenfield over the Task 03.1.3 pipeline and Task 02.3.2 readiness.
- Blockers: Task 03.1.3 must provide `RetrievalPipeline.search` and Task 02.3.2 must provide a complete indexed corpus with readiness state.

### Implementation Research

- Ground answers with an explicit context-only, no-prior-knowledge instruction and delimit retrieved content as untrusted data.
- The model returns claims with evidence IDs and code rejects unknown IDs, so citations are structural rather than free text.
- The base answer path performs no web search; web retrieval is an out-of-scope future retriever composition and the generator protocol carries no hidden web-search flag.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/generation/__init__.py | Export the generation package surface without importing providers. |
| CREATE | src/beacon_kb/generation/answer.py | Orchestrate optional rewrite, retrieval, abstention, one-LLM generation, and validation. |
| CREATE | src/beacon_kb/generation/citations.py | Resolve and validate evidence IDs and citation locators and reject unknown IDs. |
| CREATE | src/beacon_kb/generation/prompts.py | Version grounded prompts with untrusted-context delimiters. |
| CREATE | src/beacon_kb/generation/abstention.py | Apply deterministic pre- and post-generation abstention policy. |

### Acceptance Criteria

- [ ] `search()` performs zero LLM calls and returns cited evidence, while `answer()` performs exactly one LLM call over that evidence.
- [ ] Every returned citation references an evidence item in the same response, and unknown or malformed evidence IDs cannot escape validation.
- [ ] Empty or below-policy evidence deterministically returns the configured abstention response without calling the generator.
- [ ] Retrieved content is delimited as untrusted context and cannot alter system instructions, and prompts are versioned.
- [ ] The response preserves cited structured evidence alongside the answer text and never discards it.
- [ ] Diagnostics record prompt version, provider, query variants, timings, and token counts without recording secrets.
- [ ] The answer path performs no web search and the generator protocol exposes no hidden web-search flag.

### Validation Steps

```bash
python -m pytest tests/unit/generation tests/integration/test_grounded_answer.py
python -m mypy src
python -m ruff check .
```

---

## Task 04.1.1: Implement budgets, stop conditions, and the always-on trace
Epic: 04 - Agentic Layer | Feature: 04.1 - Budgeted Loop and Evidence Grading
Size: M | 5 files | ~380 LOC | Track: D

### Current State

- No agentic subpackage exists yet, so there is no budget object, no stop-condition strategy, and no trace record backing `investigate()`.
- The `AgenticTrace` record and the budget, agentic, and `StopCondition` protocol surfaces are declared in Epic 01, but no runtime enforcement or trace assembly exists.
- Pitfall to avoid: an agentic loop with no central ceiling lets a strategy run away on LLM calls, retrievals, tokens, or wall-clock time; budget checks must be central so no strategy can exceed them.
- Pitfall to avoid: budget exhaustion that raises through the facade surprises callers; exhaustion must produce a graceful partial answer carried through the trace, never a facade exception.
- Pitfall to avoid: a trace that is optional or non-deterministic cannot be inspected or replayed; the trace is always on, append-only, and deterministic under fakes.

### Desired State

- `agentic/budget.py` holds a frozen budget with ceilings on LLM calls, retrievals, total tokens, wall-clock time, and minimum marginal gain per iteration, plus a live counter that debits against those ceilings.
- The budget is consulted before every retrieval and every LLM call, and it exposes a typed remaining-capacity view and a reason when a ceiling is reached.
- `agentic/trace.py` holds the always-on append-only `AgenticTrace` builder that records every planner decision, retrieval, grading verdict, routing choice, and stop trigger in order.
- Default `StopCondition` strategies cover budget exhausted, no marginal evidence gain over the last iteration, all subqueries answered, and confidence threshold met.
- Budget arithmetic and trace shape are fully testable with no LLM and no network, using the injected clock fake.

### Gap Analysis

- Missing: the runtime budget counter, the wall-clock source seam, the append-only trace builder, and the default stop-condition strategies.
- Changes: none; this is a greenfield subpackage.
- Blockers: Task 03.1.3 provides `RetrievalPipeline.search` and Task 03.2.1 provides `generation.answer`, which the budget and stop conditions are shaped around; Task 01.1.2 defines the `AgenticTrace`, budget, and `StopCondition` contracts.

### Implementation Research

- The central budget mirrors the facade cost contract: `investigate()` caps LLM calls, retrievals, tokens, and wall-clock time, so the budget object is the single enforcement point behind that contract.
- The wall-clock ceiling reads from an injected clock so tests using the deterministic clock fake from `testing.py` stay reproducible.
- Marginal-gain is measured against the prior iteration's evidence set, so the budget carries the per-iteration gain floor while the loop supplies the observed gain.
- `StopCondition` is a protocol in `protocols.py`; v1 ships no `stop_conditions` entry-point group, so default strategies are wired as explicit instances only.
- The token counter comes from `tokens.py` so budget token accounting matches the counter used by context assembly.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/agentic/__init__.py` | Establish the agentic subpackage with no import-time side effects and no third-party imports. |
| CREATE | `src/beacon_kb/agentic/budget.py` | Define the frozen budget ceilings, the live counter debiting LLM calls, retrievals, tokens, and wall-clock, and the marginal-gain floor. |
| CREATE | `src/beacon_kb/agentic/trace.py` | Define the always-on append-only `AgenticTrace` builder and the default `StopCondition` strategies. |
| CREATE | `tests/unit/agentic/test_budget.py` | Verify ceiling debits, remaining-capacity views, exhaustion reasons, and clock-driven wall-clock limits. |
| CREATE | `tests/unit/agentic/test_trace.py` | Verify append order, determinism under the clock fake, and stop-condition firing. |

### Acceptance Criteria

- [ ] The budget debits LLM calls, retrievals, tokens, and wall-clock time independently and reports which ceiling is reached first with a typed reason.
- [ ] A budget check before a retrieval or an LLM call that would exceed a ceiling returns a stop signal rather than raising, so exhaustion is always graceful.
- [ ] The marginal-gain floor triggers a stop when the observed evidence gain over the prior iteration falls below the configured minimum.
- [ ] The `AgenticTrace` is append-only, records planner, retrieval, grading, routing, and stop entries in order, and never mutates a prior entry.
- [ ] Two runs with identical inputs and the deterministic clock fake produce byte-identical trace structures.
- [ ] Every default stop condition (budget exhausted, no marginal gain, all answered, confidence met) is individually testable with no LLM call.
- [ ] Importing `beacon_kb.agentic.budget` and `beacon_kb.agentic.trace` pulls in no third-party dependency.

### Validation Steps

```bash
python -m pytest tests/unit/agentic/test_budget.py tests/unit/agentic/test_trace.py
python -m mypy src
```

---

## Task 04.1.2: Implement evidence grading and the reflect loop
Epic: 04 - Agentic Layer | Feature: 04.1 - Budgeted Loop and Evidence Grading
Size: L | 4 files | ~560 LOC | Track: D

### Current State

- No evidence grader and no retrieve-reflect-refine controller exist yet.
- The budget, stop conditions, and trace from Task 04.1.1 exist, and `RetrievalPipeline.search` from Task 03.1.3 is the single retrieval primitive the loop must reuse.
- Pitfall to avoid: a default grader that calls an LLM adds hidden per-iteration cost; the default must be a deterministic heuristic over retrieval-stage signals with zero LLM calls, and any LLM grader is opt-in.
- Pitfall to avoid: a loop that can exceed the budget or skip the trace breaks the facade cost contract; the loop must never exceed the budget and must always emit a trace.
- Pitfall to avoid: a reflection loop that becomes a hidden default changes `investigate()` behavior silently; with reflection disabled the loop performs exactly one retrieval.

### Desired State

- `agentic/grading.py` implements `EvidenceGrader` returning keep, discard, or re-retrieve verdicts per evidence item.
- The default grader is a deterministic heuristic reading fusion rank, reranker score, and source diversity from retrieval-stage signals with zero LLM calls.
- An LLM grader is an opt-in adapter behind the same protocol, selected only by explicit config under `[agentic]`.
- `agentic/loop.py` runs one retrieve-reflect-refine iteration over `RetrievalPipeline.search`: retrieve for the active subquery, grade evidence, reflect to detect gaps or contradictions, and refine by emitting follow-up subqueries.
- The budget is consulted before every retrieval and every grading LLM call, the loop always emits an `AgenticTrace`, and with reflection disabled it performs exactly one retrieval and stops.

### Gap Analysis

- Missing: the heuristic grader, the LLM grader adapter, the iteration controller, and the reflect-and-refine follow-up derivation.
- Changes: none; greenfield.
- Blockers: Task 04.1.1 for the budget, stop conditions, and trace; Task 03.1.3 for `RetrievalPipeline.search`.

### Implementation Research

- The heuristic grader consumes the component scores retained through fusion and rerank rather than recomputing relevance, so grading stays a pure function of retrieval-stage signals.
- The loop reuses `RetrievalPipeline.search` as the one retrieval primitive so the degradation path back to `answer()` and `search()` stays exact and citation logic is never duplicated.
- Re-retrieve verdicts feed follow-up subqueries into the next iteration, and each retrieval and LLM call is gated by the budget from Task 04.1.1.
- With reflection off, the loop runs exactly one retrieval and grade pass so it stays behaviorally equal to the single-shot retrieval used by `answer()`.
- The trace records each grading verdict and each stop trigger so a run is fully replayable under the clock fake.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/agentic/grading.py` | Implement the deterministic heuristic grader over retrieval-stage signals and the opt-in LLM grader adapter behind `EvidenceGrader`. |
| CREATE | `src/beacon_kb/agentic/loop.py` | Implement the retrieve-reflect-refine controller over `RetrievalPipeline.search` under the budget with always-on trace emission. |
| CREATE | `tests/unit/agentic/test_grading.py` | Verify keep/discard/re-retrieve verdicts from the heuristic and the opt-in LLM adapter path. |
| CREATE | `tests/integration/test_reflect_loop.py` | Verify budget adherence, single-retrieval fallback with reflection off, and trace emission across iterations. |

### Acceptance Criteria

- [ ] The default grader produces keep, discard, and re-retrieve verdicts from fusion rank, reranker score, and source diversity with zero LLM calls.
- [ ] The LLM grader is reachable only through explicit `[agentic]` config and is never the default.
- [ ] The loop consults the budget before every retrieval and every grading LLM call and never exceeds any ceiling.
- [ ] The loop always emits an `AgenticTrace`, including on budget exhaustion, and the trace records every grading verdict and stop trigger.
- [ ] With reflection disabled the loop performs exactly one retrieval and one grade pass, then stops.
- [ ] Re-retrieve verdicts produce follow-up subqueries consumed by the next iteration under the remaining budget.
- [ ] A run with the deterministic fakes and clock fake is replayable and produces an identical trace structure across runs.

### Validation Steps

```bash
python -m pytest tests/unit/agentic/test_grading.py tests/integration/test_reflect_loop.py
python -m mypy src
```

---

## Task 04.2.1: Implement query planning and multi-corpus routing
Epic: 04 - Agentic Layer | Feature: 04.2 - Planning, Routing, Memory, Synthesis, and Facade
Size: L | 4 files | ~500 LOC | Track: D

### Current State

- No query planner and no multi-corpus router exist yet.
- The budget, trace, grader, and reflect loop from Feature 04.1 exist, and the registry from Task 01.2.1 exposes per-plugin capability metadata the router selects against.
- Pitfall to avoid: a planner or router that becomes a hidden default changes `investigate()` behavior silently; each must degrade to identity and be explicit config under `[agentic]`.
- Pitfall to avoid: routing that hardcodes corpus knowledge instead of reading registry-declared capabilities cannot extend to third-party corpora; routing must score corpora via registry-declared capabilities.

### Desired State

- `agentic/planner.py` implements `QueryPlanner`, decomposing a question into an ordered subquery plan, with the identity planner yielding one subquery equal to the question when disabled.
- `agentic/router.py` implements `CorpusRouter`, scoring and selecting corpora per subquery via registry-declared capabilities, with the default selecting all corpora when disabled.
- Both strategies are wired through explicit `[agentic]` config, and neither is ever a hidden default.
- Planner and router emit their decisions into the `AgenticTrace` so the plan and the routing choice are inspectable.

### Gap Analysis

- Missing: the decomposition planner, the identity planner default, the capability-scoring router, and the all-corpora router default.
- Changes: none; greenfield.
- Blockers: Task 04.1.2 for the loop the planner and router feed; Task 01.2.1 for the registry capability metadata the router scores against.

### Implementation Research

- The identity planner returns one subquery equal to the question, so with planning off the loop sees the same single query the single-shot path uses.
- The router reads declared capabilities such as supported media types and corpus scope from the registry rather than embedding corpus-specific logic, so a third-party corpus routes without code changes.
- The all-corpora default keeps routing transparent until configured, so an unconfigured `investigate()` searches every corpus exactly as the single-shot path does.
- Planner and router run per subquery and their choices are appended to the trace so the plan-then-route sequence is replayable.
- Both strategies are `QueryPlanner` and `CorpusRouter` protocols from `protocols.py` and are selected by config name, never auto-enabled.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/agentic/planner.py` | Implement the decomposing `QueryPlanner` and the identity planner default that yields one subquery equal to the question. |
| CREATE | `src/beacon_kb/agentic/router.py` | Implement the `CorpusRouter` scoring corpora via registry-declared capabilities and the all-corpora default. |
| CREATE | `tests/unit/agentic/test_planner.py` | Verify decomposition into an ordered plan and identity-default no-op behavior. |
| CREATE | `tests/unit/agentic/test_router.py` | Verify capability-based corpus selection and the all-corpora default. |

### Acceptance Criteria

- [ ] The decomposing planner returns an ordered subquery plan and the identity planner returns exactly one subquery equal to the question.
- [ ] The planner is off by default and enabled only through explicit `[agentic]` config.
- [ ] The router scores and selects corpora per subquery from registry-declared capabilities and never embeds corpus-specific hardcoding.
- [ ] The router default selects all corpora, so unconfigured routing is transparent.
- [ ] Planner and router decisions are appended to the `AgenticTrace` and are inspectable per subquery.
- [ ] With both planner and router disabled, the loop sees one subquery over all corpora, identical to the single-shot retrieval.
- [ ] A third-party corpus registered with capability metadata is routable without modifying the router.

### Validation Steps

```bash
python -m pytest tests/unit/agentic/test_planner.py tests/unit/agentic/test_router.py
python -m mypy src
```

---

## Task 04.2.2: Implement session memory, synthesis, engine, and facade investigate
Epic: 04 - Agentic Layer | Feature: 04.2 - Planning, Routing, Memory, Synthesis, and Facade
Size: M | 5 files | ~460 LOC | Track: D

### Current State

- No session memory, no cross-subquery synthesis, no `AgenticEngine`, and no wired `facade.investigate()` exist yet.
- The planner, router, loop, grader, budget, and trace from Feature 04.1 and Task 04.2.1 exist, and `generation.answer` from Task 03.2.1 is the single-shot answer path synthesis must reuse.
- Pitfall to avoid: synthesis that re-implements answer assembly diverges from the single-shot path; synthesis must reuse `generation.answer` so citation validation and abstention behave identically.
- Pitfall to avoid: importing `beacon_kb` pulling in the agentic subpackage adds import cost and side effects; `investigate()` must lazily import the subpackage on first call only.
- Pitfall to avoid: session memory as a hidden default makes stateless calls carry history; session is off unless a session is passed.

### Desired State

- `agentic/session.py` implements optional session memory with turn history, carried evidence, and follow-up rewriting, and a stateless call passes no session.
- `agentic/synthesis.py` merges cross-subquery evidence and assembles the final answer by reusing `generation.answer`, so citation validation and abstention are identical to the single-shot path.
- `agentic/orchestrator.py` provides `AgenticEngine`, wiring planner, router, loop, grader, session, and synthesis behind one call.
- `facade.investigate(question, session, budget)` lazily imports the agentic subpackage on first call, runs the engine, and always returns an inspectable `AgenticTrace`.
- With every agentic feature off, `investigate()` is behaviorally identical to `answer()`, enforced by a hard equivalence test.

### Gap Analysis

- Missing: the session store, the cross-subquery synthesis reusing generation, the engine wiring, the lazy-import facade method, and the equivalence test.
- Changes: `facade.py` gains the concrete `investigate` implementation over the Task 01.1.3 shell.
- Blockers: Task 04.2.1 for planner and router; Task 03.2.1 for `generation.answer`.

### Implementation Research

- Synthesis merges evidence across subqueries and then calls `generation.answer`, so validated evidence-ID citations and deterministic abstention behave exactly as in the single-shot path.
- `facade.investigate()` imports `beacon_kb.agentic` inside the method body so importing `beacon_kb` never imports the agentic subpackage and stays side-effect free.
- Budget exhaustion in the engine yields a graceful partial answer carried through the trace rather than a facade exception, matching the Task 04.1.1 contract.
- The equivalence test runs `investigate()` with planner identity, router all-corpora, heuristic grader, reflection off, and session off, and asserts the answer and citations match `answer()` for the same question and corpus.
- Session memory is engaged only when a session object is passed, so single-turn calls stay stateless.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/agentic/session.py` | Implement optional session memory: turn history, carried evidence, and follow-up rewriting. |
| CREATE | `src/beacon_kb/agentic/synthesis.py` | Merge cross-subquery evidence and assemble the final answer by reusing `generation.answer`. |
| CREATE | `src/beacon_kb/agentic/orchestrator.py` | Provide `AgenticEngine` wiring planner, router, loop, grader, session, and synthesis behind one call. |
| MODIFY | `src/beacon_kb/facade.py` | Implement `investigate(question, session, budget)` with the lazy agentic import and always-on trace return. |
| CREATE | `tests/integration/test_investigate_equivalence.py` | Assert `investigate()` with all features off is behaviorally identical to `answer()`. |

### Acceptance Criteria

- [ ] Synthesis reuses `generation.answer` so citation validation and abstention are identical to the single-shot path, and unknown evidence IDs are rejected identically.
- [ ] Importing `beacon_kb` does not import `beacon_kb.agentic`; the subpackage is imported only on the first `investigate()` call.
- [ ] `investigate()` always returns an inspectable `AgenticTrace`, including on budget exhaustion, and never raises on exhaustion.
- [ ] Budget exhaustion yields a graceful partial answer carried through the trace rather than a facade exception.
- [ ] Session memory engages only when a session is passed; a stateless call carries no history.
- [ ] The equivalence test proves `investigate()` with planner identity, router all-corpora, heuristic grader, reflection off, and session off returns the same answer and citations as `answer()`.
- [ ] The `AgenticEngine` wires planner, router, loop, grader, session, and synthesis behind one call with no duplicated retrieval or citation logic.

### Validation Steps

```bash
python -m pytest tests/integration/test_investigate_equivalence.py tests/unit/agentic
python -m pytest -k "import" tests/unit/test_import_side_effects.py
python -m mypy src
```

---

## Task 05.1.1: Implement the framework-neutral tool surface and the optional MCP server extra
Epic: 05 - Tool Surface and MCP | Feature: 05.1 - Tool Surface and MCP Server
Size: L | 5 files | ~620 LOC | Track: E

### Current State

- No tool surface, tool schemas, or MCP adapter exists yet.
- The `tools/` package is unpopulated, so there is no framework-neutral way to call the facade with a schema-validated contract and no way to expose the facade over MCP.
- Design pitfall to avoid: define the tool schemas once and reuse them across in-process callables and the MCP server, so there is never a second copy of a schema that drifts from the facade contract.
- Design pitfall to avoid: never return answer text while discarding structured evidence, so every tool result preserves cited evidence and stable IDs rather than flattening to a formatted string.
- Design pitfall to avoid: never silently enable web search inside generation, so the `answer` and `investigate` tools carry no hidden retrieval or web-search flag and expose only the declared facade cost contracts.
- Design pitfall to avoid: never leak secrets through tool payloads, so tool results redact credential-shaped fields and reference secrets by env-var name only.
- The `mcp` extra is declared in `pyproject.toml` but no module imports its dependency, so the MCP dependency must never be imported at base install time.
- The `serve-mcp` CLI command has no backing surface yet, so the command layer in Epic 06 has nothing to call.

### Desired State

- `tools/schema.py` defines the five tool schemas exactly once: `search`, `fetch_evidence`, `answer`, `investigate`, and `list_corpora`, each with a typed input model, a typed result model, a stable name, and a description.
- Each schema encodes the facade cost contract in its metadata: `search` declares zero LLM calls, `answer` declares exactly one LLM call, and `investigate` declares a budgeted loop that always returns a trace.
- `tools/surface.py` exposes in-process callables that validate input against the schema, dispatch to the matching `KnowledgeBase` facade method, and return structured results carrying evidence with stable `[S1]` IDs, never preformatted prose.
- The surface redacts credential-shaped values and never echoes secret contents, referencing secrets by env-var name only.
- `tools/mcp.py` is a thin MCP server that reuses the same schemas and callables from `schema.py` and `surface.py`, so there is one source of truth for every tool contract.
- The MCP server exposes each corpus as a listable resource and maps each schema to an MCP tool without redefining any input or result shape.
- Importing `beacon_kb` and importing `beacon_kb.tools` succeed with the `mcp` extra absent; only `tools/mcp.py` imports the MCP dependency, and it does so lazily so its absence raises a typed error that names the exact `pip install 'beacon-kb[mcp]'` fix.
- `investigate` over the tool surface returns the always-on `AgenticTrace` in its structured result so agentic decisions stay inspectable through the tool boundary.
- The surface wires the callable set consumed by the Epic 06 `serve-mcp` command without holding any CLI logic.

### Gap Analysis

- Missing: a single schema definition module, in-process callable surface, MCP adapter, corpus resource listing, and the redaction and graceful-absence policy.
- Changes: none; this is a greenfield package built against the approved contracts from Epic 01 and the facade methods completed in Epic 03 and Epic 04.
- Blockers: Task 03.2.1 must provide `answer()` over the retrieval pipeline, and Task 04.2.2 must provide `investigate()` and the `AgenticTrace`; both are prerequisites in the task graph.

### Implementation Research

- The five tool names and their split (`search`, `fetch_evidence`, `answer`, `investigate`, `list_corpora`) are fixed by the design spec and are the API surface reused by both in-process callers and the MCP server.
- The `mcp` extra is the only place an MCP dependency may enter; the spec forbids importing it at base install, so `tools/mcp.py` performs its import inside the server constructor and translates `ImportError` into a typed error with the install command.
- The `StopCondition` and tool protocols may exist in `protocols.py`, but v1 ships no `tools` entry-point group, so the tool surface is wired by explicit construction over the facade rather than resolved through the registry.
- Evidence identity from Epic 03 uses stable `[S1]`-style IDs; the tool result models carry those IDs so a caller can call `fetch_evidence` for a returned ID without a second search.
- The facade cost contracts are the binding invariants the schemas advertise: `search` zero LLM, `answer` one LLM, `investigate` budgeted with an always-returned trace.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/tools/__init__.py` | Export the tool-surface public names and the schema catalog without importing the MCP dependency. |
| CREATE | `src/beacon_kb/tools/schema.py` | Define the five tool schemas once with typed input and result models, stable names, descriptions, and declared cost contracts. |
| CREATE | `src/beacon_kb/tools/surface.py` | Implement in-process callables mapping schemas to facade methods with structured results, evidence-ID preservation, and secret redaction. |
| CREATE | `src/beacon_kb/tools/mcp.py` | Implement the thin MCP server reusing the schemas and callables behind the `mcp` extra with corpus resource listing and typed graceful absence. |
| CREATE | `tests/unit/tools/test_tool_surface.py` | Verify schema-validated dispatch, cost-contract metadata, evidence-ID preservation, redaction, and graceful MCP absence with the extra uninstalled. |

### Acceptance Criteria

- [ ] The five schemas `search`, `fetch_evidence`, `answer`, `investigate`, and `list_corpora` are defined exactly once in `tools/schema.py` and are imported unchanged by both `tools/surface.py` and `tools/mcp.py`.
- [ ] Each schema advertises its facade cost contract, and a test asserts `search` declares zero LLM calls, `answer` declares exactly one, and `investigate` declares a budgeted loop that always returns a trace.
- [ ] The `search` tool callable performs zero LLM calls and returns structured evidence with stable `[S1]` IDs, and the `answer` tool callable performs exactly one LLM call over that evidence.
- [ ] The `investigate` tool callable returns the always-on `AgenticTrace` in its structured result and never exceeds the configured budget.
- [ ] Every tool result preserves structured cited evidence and never returns preformatted prose that discards evidence identity.
- [ ] Tool results redact credential-shaped fields and reference secrets by env-var name only, verified by a test that passes a secret-shaped value and asserts it is absent from the result.
- [ ] Importing `beacon_kb.tools` and constructing the in-process surface succeed with the `mcp` extra uninstalled, and only `tools/mcp.py` imports the MCP dependency.
- [ ] Constructing the MCP server with the `mcp` extra absent raises a typed error whose message contains the exact `pip install 'beacon-kb[mcp]'` command.
- [ ] The MCP server lists each corpus as a resource and exposes each of the five schemas as an MCP tool without redefining any input or result shape.
- [ ] `fetch_evidence` resolves an evidence ID returned by `search` or `answer` to its full citation locator without performing a new retrieval.

### Validation Steps

```bash
python -m pytest tests/unit/tools/test_tool_surface.py
python -m mypy src
python -m ruff check .
```

---

## Task 06.1.1: Implement the CLI app, dispatch, renderers, config-driven init, and all commands including doctor and serve-mcp
Epic: 06 - CLI Tool Mode | Feature: 06.1 - CLI Application and Commands
Size: L | 6 files | ~720 LOC | Track: E

### Current State

- No `cli/` package, no `beacon-kb` or `bkb` console-script entry, and no CLI journey tests exist yet.
- The console scripts and the entry-point group declarations were declared in `pyproject.toml` at scaffold time, but no `main` dispatch, renderer, or command implementation backs them.
- Design pitfall to avoid: the CLI must hold no business logic and must call only the `KnowledgeBase` facade, so operations are never exposed as one-off framework tools that duplicate the pipeline.
- Design pitfall to avoid: render Markdown, human tables, and JSON only at the CLI boundary, never inside return values, so structured domain objects stay formatting-free.
- Design pitfall to avoid: do not silently enable web search or any hidden retrieval path from a command; every command routes through the same facade cost contracts.
- Design pitfall to avoid: `init` scaffolds a `beacon-kb.toml` whose secrets are referenced by env-var name only, never inline, so a shared or committed config never leaks credentials.

### Desired State

- `cli/app.py` dispatches every command behind one `main`, parses the global flags `--config`, `--corpus`, `--json`, `--quiet`, `--no-color`, and `--verbose`, and enforces the stable exit-code policy.
- `cli/render.py` provides human, plain, and JSON renderers plus a live-progress adapter, choosing mode from `--json`, `--quiet`, and `--no-color` and never emitting color to a non-TTY.
- `cli/commands.py` implements `init`, `index`, `search`, `ask`, `investigate`, `inspect`, `doctor`, `plugins`, `evaluate`, and `serve-mcp`, each calling only the facade and the config loader.
- The facade method stays `answer()` while the terminal verb is `ask`, and `investigate` prints the always-on `AgenticTrace`.
- `doctor` diagnoses config validity, installed extras, backend readiness, credential presence, and corpus readiness, and every error message names the failing thing and the exact fix.
- `serve-mcp` starts the MCP server when the `mcp` extra is installed and prints the exact `pip install 'beacon-kb[mcp]'` command when it is absent, exiting non-zero.
- CLI journey tests drive `init -> index -> search -> ask -> investigate` over a temporary offline workspace with deterministic fakes and assert exit codes and rendered output.
- `examples/getting_started.toml` and `examples/multi_source.toml` ship as reference configs that mirror the `init` offline defaults and reference every secret by env-var name only.

### Gap Analysis

- Missing: command dispatch, the exit-code policy, three renderers, a progress adapter, config-driven `init` scaffolding, ten command handlers, and CLI journey tests.
- Missing: the `doctor` diagnosis matrix mapping each failure class to an exact-fix message, and the `plugins` precedence report over the registry.
- Missing: the `examples/getting_started.toml` and `examples/multi_source.toml` reference configs the quickstart docs point at.
- Changes: wire the console-script `main` declared in `pyproject.toml` to `cli/app.py:main` for both `beacon-kb` and `bkb`.
- Blockers: Task 01.1.3 for config, the config loader, and the facade shell; Task 02.3.2 for the sync path behind `index`; Task 03.2.1 for `search` and `ask`; Task 04.2.2 for `investigate` and the trace; Task 05.1.1 for the tool surface behind `serve-mcp`.

### Implementation Research

- The CLI owns no business logic and every operation goes through the public `KnowledgeBase` facade and the typed config tree, so the boundary layer stays thin and testable.
- Exit codes are stable API: 0 success, 1 usage or config error, 2 readiness error when the corpus is not indexed, 3 backend error, and 4 abstention when and only when `--strict` is set.
- `--json` selects the machine renderer for every command so downstream tooling never parses human text, and `--no-color` plus non-TTY detection suppress color.
- `init` writes a commented `beacon-kb.toml` with offline sparse-only defaults (`embedder.provider = "none"`, `agentic.enabled = false`) so `init -> index -> search` works with zero credentials and zero model downloads.
- `serve-mcp` is available only when the `mcp` extra is installed; the command surface was wired in Task 05.1.1, and the CLI imports it lazily so a base install imports no MCP dependency.
- `doctor` and every command map a missing extra to the exact `pip install 'beacon-kb[extra]'` command, a missing credential to its env-var name, and an unindexed corpus to `beacon-kb index`.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon_kb/cli/app.py` | Implement `main`, command dispatch, global flags, color policy, and the stable exit-code policy over typed facade errors. |
| CREATE | `src/beacon_kb/cli/render.py` | Implement human, plain, and JSON renderers and the live-progress adapter selected by `--json`, `--quiet`, and `--no-color`. |
| CREATE | `src/beacon_kb/cli/commands.py` | Implement `init`, `index`, `search`, `ask`, `investigate`, `inspect`, `doctor`, `plugins`, `evaluate`, and `serve-mcp` calling only the facade and config loader. |
| CREATE | `tests/cli/test_cli_journey.py` | Drive the `init -> index -> search -> ask -> investigate` journey over a temp offline workspace and assert exit codes, renderer modes, and doctor fixes. |
| CREATE | `examples/getting_started.toml` | Ship a minimal single-source local config with no credentials that mirrors the `init` offline defaults. |
| CREATE | `examples/multi_source.toml` | Ship a filesystem-plus-web-plus-confluence-style multi-source config referencing every secret by env-var name only. |

### Acceptance Criteria

- [ ] `cli/commands.py` imports only the facade, the config loader, and renderers, and contains no retrieval, generation, or storage logic.
- [ ] The two console scripts `beacon-kb` and `bkb` both resolve to the same `main` and expose identical commands and flags.
- [ ] Global flags `--config`, `--corpus`, `--json`, `--quiet`, `--no-color`, and `--verbose` are parsed before dispatch and honored by every command.
- [ ] Exit codes are exactly 0 success, 1 usage or config error, 2 readiness error, 3 backend error, and 4 abstention only when `--strict` is set.
- [ ] `--json` produces a machine-readable object for every command with no color and no interleaved progress on stdout.
- [ ] `init` scaffolds a commented `beacon-kb.toml` with offline sparse-only defaults and references every secret by env-var name only, never inline.
- [ ] After `init`, the `index -> search` path runs fully offline with zero credentials and zero model downloads, and the CLI journey test proves it end to end.
- [ ] `ask` maps to the facade `answer()` and makes exactly one LLM call under a configured generator, while `search` makes zero LLM calls.
- [ ] `investigate` prints the always-on `AgenticTrace` and never exceeds the configured budget, returning a graceful partial answer on exhaustion rather than a nonzero backend exit.
- [ ] `inspect` reports corpus health, counts, fingerprints, the active revision, and the plugin capability report.
- [ ] `plugins` lists discovered plugins with distribution, group, and resolved precedence, reflecting the registry precedence order.
- [ ] `doctor` diagnoses config validity, installed extras, backend readiness, credential presence, and corpus readiness, and each failure names the exact fix: a missing extra prints the exact `pip install 'beacon-kb[extra]'`, a missing credential names its env var, and an unindexed corpus tells the user to run `beacon-kb index`.
- [ ] `serve-mcp` starts the MCP server when the `mcp` extra is installed and, when it is absent, prints `pip install 'beacon-kb[mcp]'` and exits with the usage error code without importing any MCP dependency.
- [ ] A readiness failure from running `search` or `ask` before `index` exits with code 2 and instructs the user to run `beacon-kb index`.
- [ ] `examples/getting_started.toml` and `examples/multi_source.toml` ship as valid reference configs that load without error and reference every secret by env-var name only.

### Validation Steps

```bash
python -m pytest tests/cli/test_cli_journey.py
python -m mypy src
python -m ruff check src/beacon_kb/cli tests/cli
beacon-kb --help
bkb --help
```

---

## Task 07.1.1: Build the offline evaluation and resilience suite with core gates
Epic: 07 - Evaluation, Provider Adapters, and Release | Feature: 07.1 - Evaluation and Quality Gates
Size: L | 7 files | ~560 LOC | Track: E

### Current State

- No evaluation metrics, corpus, runner, or resilience suite exists yet.
- Nothing measures retrieval quality, citation validity, or abstention accuracy against fixed thresholds.
- No test exercises restart, add/modify/delete, fingerprint migration, rollback, or concurrency as a single resilience matrix.
- Pitfall to avoid: do not detect change from raw content hashes alone; the resilience matrix must prove that parser, chunker, enrichment, embedding model, embedding dimension, and schema versions in the fingerprint drive reindex scope.
- Pitfall to avoid: do not commit sparse documents independently from vector and metadata persistence; concurrency and rollback tests must prove one promotion transaction controls visibility so active readers never observe staging revisions.
- Pitfall to avoid: do not swallow index-write failures so stores drift apart; the rollback matrix must prove the prior active revision stays searchable after failure at every stage.
- Pitfall to avoid: do not rely on the model to preserve free-text citations; citation validity must be measured as exact evidence-ID resolution, not string matching.

### Desired State

- `evaluation/metrics.py` computes Recall@K, MRR, nDCG, citation validity, citation recall, abstention accuracy, latency, and provider call counts from typed evidence and answer records.
- `evaluation/runner.py` runs a versioned `corpus.jsonl` end to end against the facade with deterministic fakes and emits both a JSON record and a Markdown summary carrying package version, corpus version, index fingerprint, providers, configuration, and seed.
- A safe versioned `tests/evaluation/corpus.jsonl` pairs each question with relevant source and chunk locators, an answerability flag, expected technical tokens, and forbidden citations, authored as new safe content with no proprietary text or secrets.
- Quality-gate tests enforce Recall@5 at least 0.85, MRR@10 at least 0.75, citation validity exactly 1.0, and abstention accuracy at least 0.90 on the committed corpus.
- A resilience matrix exercises restart consistency, add/modify/delete sync, fingerprint migration, rollback under injected stage failure, and concurrent read/sync, proving stores never drift apart.
- The facade and CLI `evaluate` path runs the runner over a corpus and reports pass or fail per gate.

### Gap Analysis

- Missing: metrics, runner, safe labeled corpus, quality-gate thresholds, resilience matrix, and a wired evaluate path.
- Changes: none; this is greenfield and depends only on the sync lifecycle from 02.3.2 and the grounded answer path from 03.2.1.
- Blockers: Task 02.3.2 provides staged sync, rollback, and health; Task 03.2.1 provides search, grounded answer, citations, and abstention.

### Implementation Research

- Use deterministic fakes for every provider so CI results cannot drift with network or model changes.
- Final ordering is rank-based RRF, so recall and MRR are computed over fused ranks rather than raw scores.
- Latency and live-provider quality are recorded comparison data, not correctness gates, so they never become flaky.
- The corpus format is line-delimited JSON so labels are diffable and each row is an independent test case.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/evaluation/__init__.py | Export evaluation metrics and runner surface. |
| CREATE | src/beacon_kb/evaluation/metrics.py | Compute Recall@K, MRR, nDCG, citation validity, citation recall, abstention accuracy, latency, and cost. |
| CREATE | src/beacon_kb/evaluation/runner.py | Run a versioned corpus over the facade and emit JSON plus Markdown summaries with full provenance. |
| CREATE | tests/evaluation/corpus.jsonl | Store safe labeled queries with relevant locators, answerability, expected tokens, and forbidden citations. |
| CREATE | tests/evaluation/test_quality_gates.py | Enforce committed offline thresholds for recall, MRR, citation validity, and abstention. |
| CREATE | tests/integration/test_resilience_matrix.py | Verify restart, add/modify/delete, fingerprint migration, rollback, and concurrency in one matrix. |
| CREATE | tests/performance/test_local_baseline.py | Record local indexing and search timing and memory without becoming a correctness gate. |

### Acceptance Criteria

- [ ] Offline quality gates meet or exceed Recall@5 >= 0.85, MRR@10 >= 0.75, citation validity == 1.0, and abstention accuracy >= 0.90 on the committed corpus.
- [ ] Citation validity is exactly 1.0 for every generated answer because every citation resolves to an evidence ID in the same response and unknown IDs are rejected.
- [ ] Every labeled unanswerable item abstains deterministically before generation and cites nothing from the forbidden list.
- [ ] The resilience matrix proves an unchanged second sync performs zero parsing, embedding, and index writes.
- [ ] The resilience matrix proves any parser, chunker, enrichment, embedding-model, embedding-dimension, or schema fingerprint change triggers the documented reindex scope.
- [ ] Injected failure at every sync stage leaves the previous active corpus fully searchable and records a recoverable failed build.
- [ ] Concurrent readers never observe a staging revision; only the atomic promotion makes new evidence visible.
- [ ] Evaluation output records package version, corpus version, index fingerprint, providers, configuration, and seed.
- [ ] The corpus fixture is newly authored safe content and contains no proprietary documentation or secrets.
- [ ] Performance tests report stage start, end, elapsed time, and current/total progress without gating correctness.

### Validation Steps

```bash
python -m pytest tests/evaluation/test_quality_gates.py tests/integration/test_resilience_matrix.py tests/performance/test_local_baseline.py
python -m mypy src
```

---

## Task 07.1.2: Add agentic evaluation and budget-regression gates
Epic: 07 - Evaluation, Provider Adapters, and Release | Feature: 07.1 - Evaluation and Quality Gates
Size: M | 3 files | ~340 LOC | Track: E

### Current State

- No agentic evaluation metrics or budget-regression gates exist yet.
- Nothing measures loop iterations, budget usage, grading precision, or the answer gain of `investigate()` over the single-shot `answer()` baseline.
- Pitfall to avoid: the central budget must cap LLM calls, retrievals, tokens, and wall-clock time, so a regression gate must prove no run ever exceeds any ceiling rather than assuming it.
- Pitfall to avoid: the default evidence grader is a deterministic heuristic with zero LLM calls, so grading-precision measurement must run under fakes and stay deterministic.
- Pitfall to avoid: `investigate()` with all agentic features off is behaviorally identical to `answer()`, so the baseline comparison must hold that equivalence as its floor.

### Desired State

- `evaluation/agentic_metrics.py` computes iterations, budget usage per dimension, grading precision, and answer gain over the single-shot baseline from the always-on `AgenticTrace`.
- A budget-regression gate asserts the loop never exceeds the configured LLM-call, retrieval, token, or wall-clock ceiling on any labeled query.
- A quality-regression gate asserts `investigate()` never scores below the single-shot `answer()` baseline on the labeled corpus.
- The agentic gates reuse the core corpus and runner from Task 07.1.1 so there is one labeled source of truth.

### Gap Analysis

- Missing: agentic metrics, a budget-never-exceeded gate, and an investigate-versus-baseline gate.
- Changes: none; the agentic runner path extends the core runner from 07.1.1.
- Blockers: Task 07.1.1 provides the corpus, metrics, and runner; Task 04.2.2 provides `facade.investigate` and the engine.

### Implementation Research

- Budget usage is read from the trace rather than instrumented separately so the gate measures the same numbers the facade enforces.
- Grading precision is computed against the labeled relevant locators already present in the corpus.
- Answer gain is measured on the same metrics as the core gates so investigate and answer are directly comparable.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/evaluation/agentic_metrics.py | Compute iterations, per-dimension budget usage, grading precision, and answer gain from the trace. |
| CREATE | tests/evaluation/test_agentic_gates.py | Assert the loop never exceeds budget and investigate never scores below the single-shot baseline. |
| MODIFY | src/beacon_kb/evaluation/runner.py | Add the agentic run path that records trace-derived metrics alongside core metrics. |

### Acceptance Criteria

- [ ] The budget-regression gate proves no run exceeds the LLM-call, retrieval, token, or wall-clock ceiling on any labeled query.
- [ ] The quality-regression gate proves `investigate()` never scores below the single-shot `answer()` baseline on the labeled corpus.
- [ ] Budget exhaustion produces a graceful partial answer carried through the trace and never a facade exception during evaluation.
- [ ] Grading precision, iterations, and answer gain are computed deterministically under fakes with zero real LLM calls.
- [ ] With all agentic features off, the agentic run reproduces the single-shot baseline metrics exactly.
- [ ] Agentic metrics are derived from the always-on `AgenticTrace` rather than a separate instrumentation path.

### Validation Steps

```bash
python -m pytest tests/evaluation/test_agentic_gates.py
python -m mypy src
```

---

## Task 07.2.1: Add remote and local providers, web, and confluence connectors as plugins
Epic: 07 - Evaluation, Provider Adapters, and Release | Feature: 07.2 - Adapters, Docs, and Release
Size: M | 4 files | ~420 LOC | Track: E

### Current State

- No remote provider, local runtime, web connector, or confluence connector exists yet.
- Nothing proves the entry-point extension path works for providers and connectors behind their extras.
- Pitfall to avoid: do not let one connector own auth lookup, remote fetch, child traversal, and conversion at once; the web and confluence connectors receive injected clients and reference credentials by env-var name only.
- Pitfall to avoid: do not hardcode provider batch limits in core logic; the remote and local providers own their own batching, retry, and score semantics behind the Embedder, Generator, and Reranker protocols.
- Pitfall to avoid: do not search every field with equal weight or default missing distance metadata to zero; provider adapters declare embedding dimension and similarity direction through a typed score contract.
- Pitfall to avoid: heavy runtimes must stay out of the base install, so each adapter imports its dependency only when resolved and each rides behind its own extra.

### Desired State

- `providers/remote.py` supplies remote embedding, generation, and rerank adapters behind the `remote` extra with injected clients and env-var-named credentials.
- `providers/local.py` supplies local ONNX embedding and rerank runtimes behind the `local` extra requiring no credentials.
- `connectors/web.py` supplies a web-page connector behind the `web` extra, and `connectors/confluence.py` supplies a confluence-style page-tree connector behind the `confluence` extra with an injected client.
- Each adapter registers through its documented entry-point group so the registry resolves it on the same path as built-ins.
- Offline contract tests verify batching, dimensions, similarity direction, retries, pagination, child traversal, and error redaction without live calls.

### Gap Analysis

- Missing: remote and local provider adapters, web and confluence connectors, their entry-point registrations, and offline contract coverage.
- Changes: none; each adapter satisfies an already-defined protocol and rides an already-declared extra.
- Blockers: Task 02.3.2 provides sync and embedding wiring; Task 03.2.1 provides the generation path the remote generator serves.

### Implementation Research

- Contract tests use the store's declared score semantics and injected fakes rather than synthetic score metadata so contract mismatches surface.
- Web and confluence connectors normalize canonical identity and provenance in the connector without owning credential lookup.
- Provider-specific batch, retry, region, model, and pagination semantics stay inside the adapter, never in core logic.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | src/beacon_kb/providers/remote.py | Implement remote embedding, generation, and rerank adapters behind the remote extra. |
| CREATE | src/beacon_kb/providers/local.py | Implement local ONNX embedding and rerank runtimes behind the local extra. |
| CREATE | src/beacon_kb/connectors/web.py | Implement the web-page source connector behind the web extra with an injected client. |
| CREATE | src/beacon_kb/connectors/confluence.py | Implement the confluence-style page-tree connector behind the confluence extra with an injected client. |

### Acceptance Criteria

- [ ] Base installation and base tests import no remote, local, web, or confluence dependency.
- [ ] Each adapter resolves through its documented entry-point group on the same registry path as built-ins.
- [ ] Provider adapters take batch size and retry policy from the injected client and never hardcode a provider batch limit in core logic.
- [ ] Embedding adapters declare dimension and similarity direction through the typed score contract and pass the generic provider contract suite.
- [ ] Web and confluence connectors receive clients explicitly, reference credentials by env-var name only, and never read or log secrets.
- [ ] Connector and provider errors identify the source and operation and redact request secrets.
- [ ] Confluence child traversal and pagination are covered by offline contract tests with an injected fake client.

### Validation Steps

```bash
python -m pytest tests/contract/test_remote_provider_contract.py tests/contract/test_local_provider_contract.py tests/contract/test_web_connector_contract.py tests/contract/test_confluence_connector_contract.py
python -m mypy src
```

---

## Task 07.2.2: Finalize extras hygiene, docs, benchmark, and release
Epic: 07 - Evaluation, Provider Adapters, and Release | Feature: 07.2 - Adapters, Docs, and Release
Size: M | 13 files | ~420 LOC | Track: E

### Current State

- No extras-isolation check, wheel/sdist hygiene check, benchmark report, or release checklist exists yet.
- The `docs/` pages for architecture, configuration, cli, extending, plugins, agentic, operations, benchmark-report, and quickstart are not complete.
- Pitfall to avoid: the shipped artifact must carry SQL migrations but never fixtures, caches, secrets, or local index directories, so wheel and sdist contents must be asserted rather than assumed.
- Pitfall to avoid: there is no `agentic` extra, so the extras-isolation check must prove the base install carries the agentic subpackage while html, pdf, web, confluence, remote, local, and mcp stay optional.
- Pitfall to avoid: importing `beacon_kb` must have no agentic import and no side effects, so the hygiene check must prove the base import loads no plugin and no heavy runtime.

### Desired State

- An extras-isolation test proves the base wheel installs without html, pdf, web, confluence, remote, local, or mcp dependencies and that each extra pulls exactly its declared dependency.
- A wheel and sdist hygiene test proves SQL migrations are included and no fixtures, caches, secrets, or local index directories are shipped.
- A benchmark run indexes the committed safe corpus, records core and agentic gate results, and documents accepted trade-offs.
- Every `docs/` page is complete: architecture, configuration, cli, extending, plugins, agentic, operations, benchmark-report, and quickstart.
- The `examples/` tree ships `library_quickstart.py`, the library-mode ingest, search, answer, and investigate walk-through over the facade, and `custom_plugin/`, the minimal third-party plugin skeleton the extending guide references.
- A release checklist records the accepted trade-offs, the classification of every capability as implemented, optional, deferred, or excluded, and the gate results.

### Gap Analysis

- Missing: extras-isolation and wheel-hygiene checks, the benchmark report, the completed docs set, the `examples/` walk-through and plugin skeleton, and the release checklist.
- Changes: none; this task only adds tests, docs, examples, and release records and touches no pipeline code.
- Blockers: Task 07.1.2 and Task 07.2.1 provide the gates and adapters to verify; Task 05.1.1 and Task 06.1.1 provide the tool surface and CLI the docs describe.

### Implementation Research

- Wheel and sdist contents are asserted with the standard build and zipfile listing tools so the check runs in CI.
- The benchmark uses safe synthetic documents and states that final ordering is rank-based RRF, not raw-score based.
- Docs repeat the non-goals so the excluded Jira, RCA, Slack, hosted-service, UI, crawler, and web-search scope stays legible.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | tests/test_extras_isolation.py | Prove the base install excludes optional extras and each extra pulls exactly its declared dependency. |
| CREATE | tests/test_wheel_hygiene.py | Prove the wheel and sdist include migrations and exclude fixtures, caches, secrets, and local indexes. |
| CREATE | docs/benchmark-report.md | Record corpus, commands, core and agentic gate results, and accepted trade-offs. |
| CREATE | docs/architecture.md | Record boundaries, data flow, identity, storage, scoring, citation, and security decisions. |
| CREATE | docs/operations.md | Document rebuild, fingerprint migration, backup, rollback, and budget tuning. |
| CREATE | docs/extending.md | Document protocol and plugin-registry authoring and shipping a plugin as a package. |
| CREATE | docs/plugins.md | Document the entry-point group catalog, precedence, capability metadata, and PLUGIN_API_VERSION. |
| CREATE | docs/agentic.md | Document budgets, stop conditions, the trace, the degradation matrix, tools, and MCP. |
| CREATE | docs/configuration.md | Document the full beacon-kb.toml schema reference with defaults and env overrides. |
| CREATE | docs/cli.md | Document every command, flag, exit code, output mode, and doctor. |
| CREATE | docs/quickstart.md | Document the init/index/search/ask journey with copy-paste blocks. |
| CREATE | examples/library_quickstart.py | Ship the library-mode ingest, search, answer, and investigate example over the facade. |
| CREATE | examples/custom_plugin/ | Ship the minimal third-party plugin package skeleton registering a connector for the extending guide. |

### Acceptance Criteria

- [ ] The base wheel installs without html, pdf, web, confluence, remote, local, or mcp dependencies and still carries the agentic subpackage.
- [ ] There is no `agentic` extra in the declared extras matrix.
- [ ] Importing `beacon_kb` performs no plugin load, no agentic import, and no network, filesystem, or credential side effect.
- [ ] The wheel and sdist include SQL migrations and contain no test fixtures, caches, secrets, or local index directories.
- [ ] Benchmark results meet the core and agentic quality gates and state that final ordering is rank-based RRF.
- [ ] Every planned capability is classified as implemented, optional, deferred, or intentionally excluded.
- [ ] Docs repeat the Jira, RCA, Slack, hosted-service, UI, crawler, and web-search exclusions.
- [ ] Every design-spec docs page ships: architecture, configuration, cli, extending, plugins, agentic, operations, benchmark-report, and quickstart.
- [ ] `examples/library_quickstart.py` runs the facade ingest, search, answer, and investigate journey, and `examples/custom_plugin/` provides a minimal third-party plugin skeleton the extending guide references.
- [ ] The release checklist records the accepted trade-offs and the core and agentic gate results.

### Validation Steps

```bash
python -m pytest tests/test_extras_isolation.py tests/test_wheel_hygiene.py
python -m ruff check .
python -m mypy src
python -m build
python -m zipfile -l dist/*.whl
```

---

## Execution Assumptions

- Python 3.11 is the minimum supported runtime, and the CI matrix may add newer versions after confirming deployment images.
- The `pipx install -> init -> index -> search` journey runs fully offline on first run with zero credentials and zero model downloads, and every deterministic path stays reproducible under the fakes in `testing.py`.
- The agentic subpackage ships in the base install with zero third-party dependencies, and importing `beacon_kb` never imports `beacon_kb.agentic`; `investigate()` lazily imports it on first call.
- Sparse-only BM25 retrieval is the default mode and requires no embedder, so dense retrieval activates only when an embedder is configured through the `local` or `remote` extra.
- SQLite FTS5 plus NumPy is the reference local backend for correctness and laptop-scale corpora, not a claim that it is the final production-scale vector engine.
- Security-sensitive source content, ACL handling, and redaction policy are supplied by the caller, and the library preserves and enforces metadata but never invents application authorization.

## Open Decisions That Do Not Change the Task Graph

- Confirm the specific lightweight ONNX embedding model shipped or fetched by the `local` extra and its wheel-size budget, since the offline dense-quality promise depends on it while the sparse-only mode is unaffected.
- Confirm the exact env-var names and provider SDK pins the `remote` extra targets before external release.
- Curate and approve the safe evaluation corpus labels before the recommended thresholds harden into release gates.
- Select the first production-scale external vector adapter only after measuring real corpus size, update rate, and latency, since the store protocol and adapter seam are already planned.
