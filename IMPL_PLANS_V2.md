# Implementation Plans V2: Beacon - Industry-Standard Knowledge-Base RAG Server

*Source spec: docs/superpowers/specs/2026-07-23-beacon-industry-standard-rag-design.md*
*Epics: EPICS_V2.md | Tracker: FOCUS_CHAIN_V2.md*

> Plans cover: fully detailed tasks for Epic 01 Core Service and Storage, Epic 02 Ingestion Wave 1, and Epic 03 Retrieval and Grounded Answers.
> Epics 04 through 08 appear at the task-list level and are detailed just-in-time before each epic starts.

## Goal

Rebuild Beacon as a self-hosted knowledge-base RAG server on industry-standard components, consumed day one by AI agents over MCP and by developers over REST.
The existing `beacon_kb` package and its 1,101 tests are the executable spec for the ported safety behaviors: never-break-the-live-corpus sync, structural citation validation, deterministic abstention, prompt-injection defenses, and enforced cost contracts.
`beacon_kb` stays untouched and green until the Epic 08 cleanup deletes it.

## Target Architecture

```text
            config (pydantic-settings, env-only secrets, local-first defaults)
                                      |
   +----------------------------------+----------------------------------+
   |                                                                     |
   v                     INGESTION (Epic 02)                             |
 Connector (folder | upload | web | ... waves 2-3)                       |
   -> raw doc store + content-hash dedupe (SQLite state DB)              |
   -> Docling parse -> structured sections (heading paths preserved)     |
   -> LlamaIndex hierarchical chunking (parent/child)                    |
   -> embeddings (LiteLLM auto-detect | local | sparse-only floor)       |
   -> staged upsert into SHADOW Qdrant collection                        |
   -> validation (counts, dimensions, fingerprint)                       |
   -> atomic Qdrant ALIAS FLIP -> live          <- sync job records      |
                                      |                                  |
                                      v                                  |
                     RETRIEVAL (Epic 03, search: 0 LLM calls)            |
 Qdrant Query API hybrid (dense + sparse prefetch, native RRF)           |
   -> payload filters enforced at the boundary (not bypassable)          |
   -> optional cross-encoder rerank                                      |
   -> parent/context expansion under token budgets                       |
   -> Evidence[] with stable gap-free [S1] labels + real provenance      |
                                      |                                  |
                                      v                                  |
                     ANSWER (Epic 03, answer: exactly 1 LLM call)        |
 deterministic pre-abstention (0 LLM) -> one LiteLLM call over           |
 delimiter-neutralized untrusted context -> post-abstention              |
   -> structural citation validation vs canonical evidence               |
                                      |                                  |
                                      v                                  |
        INVESTIGATE (Epic 05, budget-bounded LangGraph loop)  <----------+
 plan -> retrieve -> grade -> reflect -> synthesize, checkpointed,
 SSE-streamed step trace
                                      |
                                      v
              SERVING: FastAPI REST + FastMCP (/mcp + stdio) + Typer CLI
              /healthz /readyz problem-details errors, optional API key
```

## Tech Stack

| Concern | Component |
|---|---|
| Document parsing | Docling |
| Connectors | LlamaIndex readers behind one `Connector` interface |
| Chunking | LlamaIndex hierarchical node parser (parent/child) |
| Vector store and hybrid search | Qdrant (dense + sparse, native RRF fusion; embedded local mode for single-process) |
| Sync bookkeeping | SQLite state DB (fingerprints, revisions, job history) |
| LLMs and embeddings | LiteLLM with auto-detect; Ollama fallback; local embeddings default |
| Reranking | Optional sentence-transformers cross-encoder |
| Agentic loop | LangGraph with budgets and SQLite checkpointing |
| Serving | FastAPI (REST) + FastMCP in-process (`/mcp` streamable HTTP and stdio) |
| CLI | Typer: `beacon serve / sync / search / ask` |
| Evaluation | RAGAS golden-set gates in CI; OpenTelemetry/Langfuse-compatible tracing |

## Global Constraints

- `requires-python = ">=3.11"` stays as-is.
- All new code lives in the new top-level package `src/beacon/`; new tests live in `tests/beacon/unit` and `tests/beacon/integration`, separate from the old suite.
- `src/beacon_kb` and its existing tests stay untouched and passing until Epic 08 deletes them; every task must leave the old suite green.
- New dependencies are added as an additive dependency group so the `beacon_kb` base install keeps working unchanged.
- One independent green commit per task; each epic is one branch and one PR.
- No AI attribution anywhere in any git artifact: no generated-with footers or co-author trailers in commits, PR titles or bodies, or issue text.
- No em dashes in any document; plain "-" only.
- Quality gates on every task: `mypy --strict` clean over `src/beacon` and `ruff check` clean over the new code.
- Cost contracts are hard invariants: `search` performs zero LLM calls; `answer` performs exactly one LLM call; `investigate` is budget-bounded.
- Local-first defaults: API-key auth is off on localhost; `docker compose up` and `beacon serve` work with no credentials.
- Secrets are configured by environment variable only and are never logged.
- TDD is the standing rule for every task: write the failing test first, then the implementation; this document states it once and does not repeat per-step test instructions.
- Validation steps run with the project virtualenv python (activate the venv or prefix with the venv interpreter path).
- Integration tests run against embedded Qdrant (local mode), not mocks; LLM behavior is tested with deterministic counting fakes.

---

## Task 01.1: Scaffold the beacon package, dependency group, config, and error taxonomy

Epic: 01 - Core Service and Storage
Size: M | 9 files | ~450 LOC

### Current State

The repository holds only the `beacon_kb` library: `pyproject.toml` declares base dependency `numpy` plus small extras, and there is no `src/beacon/` package, no server-stack dependency, and no pydantic anywhere.
`beacon_kb` has its own config (`config.py`, TOML plus env overlay) and error hierarchy (`errors.py`), which serve as behavioral references but are not reused.
There is no RFC 9457 problem-details representation anywhere in the codebase.
Pitfall to avoid: do not modify the `beacon_kb` base dependency list or its extras; any change that alters what the old package installs or imports breaks the executable spec.

### Desired State

`pyproject.toml` gains an additive optional-dependency group `server` containing the new stack: fastapi, uvicorn, pydantic-settings, qdrant-client, llama-index-core plus the specific integrations later tasks need, docling, litellm, langgraph, typer, fastmcp, and httpx.
The dev group gains pytest-asyncio (or the equivalent async test support) and an HTTP test client dependency if one is needed beyond fastapi's own; ragas is explicitly deferred to Epic 06.
`src/beacon/` exists as a typed package (`py.typed`) with subpackage skeletons matching the spec layout: `config.py`, `models.py`, `errors.py`, `problems.py`, and empty `ingest/`, `retrieval/`, `answer/`, `investigate/`, `server/`, `evals/` packages.
`src/beacon/config.py` defines a pydantic-settings `BeaconSettings` tree (server, qdrant, state, models, retrieval, answer, investigate sections) with env-var configuration under a `BEACON_` prefix, local-first defaults, and secret fields that are env-only and excluded from any repr or dump used in logs.
`src/beacon/errors.py` defines the typed taxonomy from the spec with a machine-readable `kind` for each error family: readiness, backend, ingestion, citation, and budget.
`src/beacon/problems.py` defines transport-neutral RFC 9457 problem-details helpers: a frozen problem model (type, title, status, detail, instance, plus the `kind` extension) and a mapping from each error family to its HTTP status.
`mypy --strict` and ruff cover `src/beacon` and `tests/beacon` without weakening the existing tool config for `beacon_kb`.

### Gap Analysis

- Missing: the `server` dependency group and dev-group additions in `pyproject.toml`.
- Missing: the entire `src/beacon/` package skeleton and `py.typed` marker.
- Missing: pydantic-settings config, the typed error taxonomy, and the problem-details helpers.
- Changes: `pyproject.toml` (additive only) and the mypy/ruff/pytest tool sections extended to cover the new trees.
- Blockers: none; this is the first task of the rebuild.

### Implementation Research

Consult the current pydantic-settings docs for nested settings models, env prefixes, and secret-field handling; the required behavior is that every setting is overridable via `BEACON_`-prefixed env vars, nested sections use a documented delimiter, and secret values never appear in reprs, dumps, or error messages.
Keep the error taxonomy independent of FastAPI so Epic 04 can map the same errors to MCP tool errors; the problem-details helpers must be pure functions over the error types.
Model the error-to-status mapping as one explicit table so a new error kind cannot silently default to 500 without a conscious decision.
Register a dedicated pytest marker or testpath configuration so `tests/beacon` runs both standalone and alongside the old suite.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| MODIFY | `pyproject.toml` | Add the `server` optional-dependency group, dev-group additions, and tool-config coverage for `src/beacon` and `tests/beacon`. |
| CREATE | `src/beacon/__init__.py` | Establish the package with curated exports and no import-time side effects. |
| CREATE | `src/beacon/py.typed` | Mark the package as typed under PEP 561. |
| CREATE | `src/beacon/config.py` | Define the `BeaconSettings` pydantic-settings tree with local-first defaults and env-only secrets. |
| CREATE | `src/beacon/errors.py` | Define the typed error taxonomy with kinds: readiness, backend, ingestion, citation, budget. |
| CREATE | `src/beacon/problems.py` | Define RFC 9457 problem-details helpers and the error-to-status mapping table. |
| CREATE | `tests/beacon/conftest.py` | Provide shared fixtures (tmp settings, deterministic env) for the new suite. |
| CREATE | `tests/beacon/unit/test_config.py` | Verify defaults, env overrides, nesting, and secret redaction. |
| CREATE | `tests/beacon/unit/test_problems.py` | Verify the taxonomy-to-problem mapping and RFC 9457 field shape. |

### Acceptance Criteria

- [ ] Installing the repo without the `server` group leaves `beacon_kb` importable and its full test suite passing, proving the dependency change is additive.
- [ ] `pip install -e ".[server,dev]"` succeeds and `import beacon` performs no network, filesystem, or logging side effect.
- [ ] Every `BeaconSettings` field has a local-first default: no credentials required, embedded Qdrant path under a local data dir, auth disabled on localhost.
- [ ] Any setting is overridable via a `BEACON_`-prefixed environment variable, including nested sections.
- [ ] Secret-typed settings never appear in `repr`, model dumps used for logging, or error messages, verified by test.
- [ ] Each error class carries a stable machine-readable `kind` in {readiness, backend, ingestion, citation, budget} and maps to exactly one HTTP status in the problem-details table.
- [ ] Problem-details output contains `type`, `title`, `status`, `detail`, and the `kind` extension, and round-trips through JSON.
- [ ] `mypy --strict` over `src/beacon` and `ruff check` over the new files pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_config.py tests/beacon/unit/test_problems.py
python -m pytest tests/unit tests/contract -q
python -m mypy src/beacon
python -m ruff check src/beacon tests/beacon
python -c "import beacon; import beacon.config; import beacon.errors; import beacon.problems"
```

---

## Task 01.2: Qdrant store layer with shadow collections and alias-flip promotion

Epic: 01 - Core Service and Storage
Size: L | 6 files | ~600 LOC

### Current State

No Qdrant integration exists; `beacon_kb` implements staged atomic promotion over SQLite in `storage/sqlite.py` and `indexing/coordinator.py`, where staged writes are invisible until one promotion transaction and the prior active revision stays searchable on failure.
That revision-promotion behavior is the reference semantics this task reproduces on Qdrant using shadow collections and alias flips.
`tests/integration/test_sync_rollback.py` (for example `test_promote_revision_failure_leaves_prior_revision_searchable`) documents the exact guarantees.
Pitfall to avoid: do not let any caller write directly to a live collection name; every live read must resolve through an alias so promotion is a metadata operation, never a data copy.

### Desired State

`src/beacon/storage/qdrant.py` wraps `qdrant-client` behind a small typed `QdrantStore` interface: create/delete/list physical collections, upsert points, query, resolve aliases, and report collection info.
The wrapper is constructed from `BeaconSettings`: embedded local mode (path-based, in-process) when no server URL is configured, server mode (URL plus optional API key) otherwise, with the mode decision logged once at startup.
`src/beacon/storage/lifecycle.py` implements the staged promotion protocol: `begin_stage(collection)` creates a shadow physical collection named from the logical collection plus a new revision id; `promote(stage)` atomically flips the logical alias to the shadow and schedules the previous physical collection for cleanup; `abort(stage)` drops the shadow and leaves the alias untouched.
`src/beacon/storage/payload.py` defines the typed payload schema stored with every point: chunk text, source uri, title, heading path, tags, dates (created, modified, ingested), content hash, chunk hash, parent chunk id, and fingerprint; plus the payload-index declarations needed for filterable fields.
Live reads always resolve the logical name through the alias; a logical collection with no alias is reported as empty/absent rather than an error leaking Qdrant internals.
All Qdrant client errors are translated to the `backend` error kind from task 01.1.

### Gap Analysis

- Missing: the qdrant-client wrapper, embedded-vs-server mode selection, and typed error translation.
- Missing: the shadow-collection lifecycle with alias-flip promotion and abort cleanup.
- Missing: the payload schema and payload-index declarations.
- Changes: none outside `src/beacon/storage/`.
- Blockers: task 01.1 must provide config and the error taxonomy.

### Implementation Research

Consult the current qdrant-client docs for three capabilities: embedded local mode (path-based client), collection alias operations, and named dense plus sparse vector configuration; do not assume method signatures.
The required behavior for promotion is: the alias update that retargets the logical name is a single atomic alias-operations request, so there is no observable moment where the logical name resolves to nothing or to a half-written collection.
The required behavior for staging is: points written to a shadow collection are never visible through the logical alias until promotion, verified by querying the alias mid-stage.
Sparse vector support must be configured at collection creation so Epic 02 can write sparse representations and Epic 03 can hybrid-query them; record the chosen named-vector layout (dense name, sparse name) as constants in `payload.py`.
Cleanup of the previous physical collection after a flip must be failure-tolerant: a failed cleanup logs and leaves an orphan for a later sweep, never un-promotes.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/storage/__init__.py` | Expose the store, lifecycle, and payload schema surface. |
| CREATE | `src/beacon/storage/qdrant.py` | Wrap qdrant-client with embedded/server mode selection and backend-error translation. |
| CREATE | `src/beacon/storage/lifecycle.py` | Implement begin_stage, promote (alias flip), abort, and orphan cleanup. |
| CREATE | `src/beacon/storage/payload.py` | Define the typed point payload schema, named-vector constants, and payload-index declarations. |
| CREATE | `tests/beacon/integration/test_qdrant_lifecycle.py` | Verify staging invisibility, atomic alias flip, abort, and failure semantics against embedded Qdrant. |
| CREATE | `tests/beacon/unit/test_payload_schema.py` | Verify payload schema shape, required fields, and filterable-field declarations. |

### Acceptance Criteria

- [ ] With no server URL configured the store runs embedded from a local path; with a URL configured it targets the server; the selected mode is exposed for `/readyz` reporting.
- [ ] Points upserted into a stage are not visible through the logical alias before promotion, verified by querying the alias mid-stage.
- [ ] `promote` retargets the alias in one atomic alias-operations call; a query through the logical name succeeds at every point before, during, and after promotion.
- [ ] `abort` drops the shadow collection and leaves the previously live collection serving unchanged.
- [ ] A simulated failure between stage write and promote leaves the prior collection serving and the shadow removable, mirroring the `test_sync_rollback.py` guarantees.
- [ ] Every stored point carries the full payload schema and the collection declares payload indexes for source uri, tags, and date fields.
- [ ] Qdrant client exceptions surface as typed `backend` errors, never raw client exceptions.
- [ ] `mypy --strict` and ruff pass over the new files.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_qdrant_lifecycle.py tests/beacon/unit/test_payload_schema.py
python -m mypy src/beacon
python -m ruff check src/beacon/storage
```

---

## Task 01.3: SQLite state DB with migrations for sources, revisions, fingerprints, and jobs

Epic: 01 - Core Service and Storage
Size: M | 6 files | ~500 LOC

### Current State

No state DB exists in the new package; `beacon_kb` keeps sources, revisions, fingerprints, and build runs inside its single SQLite store (`storage/sqlite.py`, `storage/migrations/0001_initial.sql`) and derives corpus health from durable state (`ingestion/sync.py::derive_corpus_health`, exercised by `tests/integration/test_corpus_health.py` including `test_health_reconstructed_after_restart`).
In the new architecture chunk data lives in Qdrant, so the SQLite state DB holds only bookkeeping: fingerprints, sources, revisions, and sync jobs.
Pitfall to avoid: do not store any state needed for correctness only in process memory; async sync jobs must survive restart per the spec operations section.

### Desired State

`src/beacon/state/db.py` opens the SQLite database from config (WAL mode, foreign keys on), applies orderable SQL migrations from `src/beacon/state/migrations/`, and exposes a thin typed connection helper.
`src/beacon/state/repo.py` provides typed repositories over four table groups: collections (logical name, created, settings snapshot), sources (canonical uri, connector kind, content hash, status active/retired, timestamps), revisions (per collection: revision id, fingerprint, status staged/live/failed/retired, counts), and sync_jobs (job id, collection, state pending/running/succeeded/failed, change-plan summary, error detail, started/finished timestamps).
Corpus state per collection (empty, building, ready, failed) is derivable from revisions plus jobs by one pure function, reproducing the `beacon_kb` health semantics: a failed rebuild after a successful one still reports ready because the prior revision serves.
All timestamps are UTC ISO 8601; all writes are transactional; the module has no dependency on Qdrant or FastAPI.

### Gap Analysis

- Missing: the migration runner and initial schema.
- Missing: typed repositories for collections, sources, revisions, and sync jobs.
- Missing: the pure corpus-state derivation function.
- Changes: none outside `src/beacon/state/`.
- Blockers: task 01.1 must provide config and the error taxonomy.

### Implementation Research

Use stdlib `sqlite3` with explicit transactions; no ORM is needed for four table groups, and keeping this layer dependency-free simplifies the CLI and tests.
Migrations are numbered SQL files applied in order inside one transaction each, with the applied set recorded in a `schema_migrations` table; re-running is idempotent.
The corpus-state derivation must reproduce these `beacon_kb` behaviors, which become tests: empty when no live revision and no jobs, building while a job is running, ready when a live revision exists even if the latest job failed, failed when the first-ever sync failed and nothing serves.
Job records must carry enough to answer `GET /jobs/{id}` in task 02.5 without joining Qdrant: state, change-plan counts, error problem-details payload on failure.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/state/__init__.py` | Expose the state DB surface. |
| CREATE | `src/beacon/state/db.py` | Open SQLite with WAL and foreign keys, run migrations, provide the connection helper. |
| CREATE | `src/beacon/state/migrations/0001_initial.sql` | Create collections, sources, revisions, sync_jobs, and schema_migrations tables. |
| CREATE | `src/beacon/state/repo.py` | Typed repositories and the pure corpus-state derivation function. |
| CREATE | `tests/beacon/unit/test_state_migrations.py` | Verify migration application, idempotency, and schema shape. |
| CREATE | `tests/beacon/integration/test_state_repo.py` | Verify repositories, transactionality, restart durability, and corpus-state derivation. |

### Acceptance Criteria

- [ ] Opening a fresh database applies all migrations; reopening applies none and preserves data, proving idempotency and restart durability.
- [ ] Sources support the active/retired status transition and record content hash and connector kind.
- [ ] Revisions record fingerprint and status, and exactly one revision per collection can be live.
- [ ] Sync jobs persist state transitions pending -> running -> succeeded/failed with timestamps and an error payload on failure, and are readable after a simulated restart (new connection).
- [ ] The corpus-state function returns empty, building, ready, and failed per the four reference behaviors, including ready-despite-last-build-failed.
- [ ] The module imports neither Qdrant nor FastAPI.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_state_migrations.py tests/beacon/integration/test_state_repo.py
python -m mypy src/beacon
python -m ruff check src/beacon/state
```

---

## Task 01.4: FastAPI skeleton with health, readiness, problem details, auth, and telemetry

Epic: 01 - Core Service and Storage
Size: M | 7 files | ~550 LOC

### Current State

No server exists.
Tasks 01.1 through 01.3 provide config, the error taxonomy with problem-details helpers, the Qdrant store, and the state DB, but nothing serves HTTP.
Pitfall to avoid: do not construct global singletons at import; the app factory must build per-instance dependencies from settings so tests can run isolated instances.

### Desired State

`src/beacon/server/app.py` exposes `create_app(settings)` returning a FastAPI application with dependency-injected store, state DB, and settings; nothing is created at module import.
`src/beacon/server/routes/health.py` serves `GET /healthz` (process liveness, always 200 when serving) and `GET /readyz` (per-collection corpus state empty/building/ready/failed derived from the state DB, plus backend reachability), with `/readyz` returning 503 with a `readiness` problem when any configured backend is unreachable.
`src/beacon/server/error_handlers.py` registers exception handlers translating every taxonomy error into its RFC 9457 problem-details response with `application/problem+json`, and a catch-all handler that emits a generic problem without leaking internals.
`src/beacon/server/auth.py` implements optional bearer API-key middleware: when a key is configured, non-localhost requests must present it; requests from localhost are exempt by default per the local-first rule; when no key is configured, auth is off entirely.
`src/beacon/server/telemetry.py` provides OpenTelemetry hooks: tracer acquisition, span helpers for pipeline stages, and FastAPI instrumentation wiring that is a no-op when the OTel SDK or exporter is not configured.

### Gap Analysis

- Missing: the app factory and dependency wiring.
- Missing: health and readiness routes with per-collection state.
- Missing: problem-details exception handlers.
- Missing: API-key middleware with the localhost exemption.
- Missing: OTel hooks with graceful no-op behavior.
- Changes: none outside `src/beacon/server/`.
- Blockers: tasks 01.1 and 01.3; task 01.2 for backend reachability in `/readyz`.

### Implementation Research

Consult the current FastAPI docs for lifespan-based startup/shutdown and dependency overrides in tests; the required behavior is that two `create_app` instances with different settings never share state.
The localhost exemption must be decided from the actual client address of the connection, not from a spoofable header; document that operators fronting Beacon with a proxy must configure the key and disable the exemption.
Consult the current opentelemetry-python docs for optional instrumentation; the required behavior is that missing OTel configuration yields zero overhead and zero warnings, and configured OTel yields spans around request handling that later tasks can nest pipeline-stage spans under.
Handlers must set `Content-Type: application/problem+json` on every error response, including validation errors from FastAPI itself, which are remapped to the problem shape.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/server/__init__.py` | Expose `create_app`. |
| CREATE | `src/beacon/server/app.py` | App factory with lifespan wiring of settings, store, and state DB. |
| CREATE | `src/beacon/server/routes/health.py` | `/healthz` and `/readyz` with per-collection corpus state. |
| CREATE | `src/beacon/server/error_handlers.py` | Taxonomy-to-problem-details exception handlers and the catch-all. |
| CREATE | `src/beacon/server/auth.py` | Optional bearer API-key middleware with localhost exemption. |
| CREATE | `src/beacon/server/telemetry.py` | OTel tracer helpers and no-op fallback. |
| CREATE | `tests/beacon/integration/test_server_skeleton.py` | Verify factory isolation, health, readiness states, problem responses, and auth matrix. |

### Acceptance Criteria

- [ ] `create_app` builds an isolated instance; two instances with different settings do not share store or DB state.
- [ ] `/healthz` returns 200 while serving; `/readyz` reports each collection as empty, building, ready, or failed from durable state and returns 503 with a readiness problem when a backend is unreachable.
- [ ] Every taxonomy error raised in a route becomes an `application/problem+json` response with the mapped status and `kind`; unexpected exceptions become a generic problem with no stack trace or internal detail in the body.
- [ ] With no API key configured, all requests pass; with a key configured, localhost requests pass without a key and non-localhost requests without the correct bearer key receive a 401 problem response.
- [ ] With no OTel configuration the app starts with no warnings and no exporter activity; with a tracer configured, request spans are produced.
- [ ] FastAPI request-validation failures are returned in problem-details shape, not the framework default shape.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_server_skeleton.py
python -m mypy src/beacon
python -m ruff check src/beacon/server
```

---

## Task 01.5: Collections REST resource, compose file, Dockerfile, and smoke test

Epic: 01 - Core Service and Storage
Size: M | 6 files | ~450 LOC

### Current State

The app factory serves health and readiness but no domain resource, and there is no container or compose deployment surface.
The spec requires `POST /collections` in the API surface and `docker compose up` as the first deployment path.
Pitfall to avoid: do not create Qdrant physical collections eagerly at collection registration; a registered-but-never-synced collection is state-DB-only and reports empty, matching the corpus-state model from task 01.3.

### Desired State

`src/beacon/server/routes/collections.py` serves `POST /collections` (create a logical collection with a validated name, 409 problem on duplicate), `GET /collections` (list with per-collection corpus state), and `GET /collections/{name}` (detail with revision and last-job summary).
`Dockerfile` builds the server image from the repo with the `server` dependency group and runs uvicorn against `create_app`.
`docker-compose.yml` defines the `beacon` service and a `qdrant` service, wired by env vars, with a volume for the state DB and Qdrant storage, plus a commented optional `ollama` profile stub that Epic 04 completes.
`tests/beacon/integration/test_collections_api.py` covers the resource; a smoke test boots the app with embedded Qdrant in a temp dir and walks create -> list -> readyz reporting the new collection as empty.
The epic exit criterion holds: a fresh checkout can `docker compose up` and receive 200 from `/healthz` and a truthful `/readyz`.

### Gap Analysis

- Missing: the collections routes and their request/response models in `src/beacon/models.py`.
- Missing: Dockerfile and docker-compose.yml.
- Missing: the in-process smoke test.
- Changes: `src/beacon/server/app.py` registers the new router.
- Blockers: tasks 01.2 and 01.4.

### Implementation Research

Collection names become Qdrant alias names and URL path segments, so validate them against one conservative pattern (lowercase alphanumerics, dash, underscore, bounded length) and reject everything else with a 422 problem.
The compose file must work with zero env configuration: the beacon service points at the qdrant service by service name; secrets and API keys appear only as commented examples referencing env vars, never inline values.
Consult the current Docling and qdrant-client release notes when pinning the image's Python base version so wheels are available; the required behavior is a build with no compilation step.
Keep the Dockerfile layered so dependency installation caches separately from source copy.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/models.py` | Pydantic API schemas for collection create/list/detail responses. |
| CREATE | `src/beacon/server/routes/collections.py` | Collections REST resource over the state DB and store. |
| MODIFY | `src/beacon/server/app.py` | Register the collections router. |
| CREATE | `Dockerfile` | Build and run the server image. |
| CREATE | `docker-compose.yml` | Beacon plus Qdrant services, volumes, env wiring, ollama profile stub. |
| CREATE | `tests/beacon/integration/test_collections_api.py` | Resource tests plus the embedded end-to-end smoke walk. |

### Acceptance Criteria

- [ ] `POST /collections` creates a collection, rejects an invalid name with a 422 problem, and rejects a duplicate with a 409 problem.
- [ ] `GET /collections` and `GET /collections/{name}` report per-collection corpus state consistent with `/readyz`.
- [ ] Creating a collection performs no Qdrant write; the physical collection appears only at first sync staging.
- [ ] The smoke test boots the app with embedded Qdrant in a temp dir and walks create -> list -> readyz with the collection reported empty.
- [ ] `docker build` succeeds; `docker compose config` validates; the compose file contains no inline secret values.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_collections_api.py
python -m mypy src/beacon
python -m ruff check src/beacon
docker compose config -q
```

---

## Task 02.1: Connector interface, folder connector, sources resource, and upload endpoint

Epic: 02 - Ingestion Wave 1
Size: L | 8 files | ~600 LOC

### Current State

No ingestion code exists in the new package.
`beacon_kb` defines the reference connector semantics in `protocols.py` (SourceConnector), `ingestion/identity.py` (canonical URIs, content-addressed IDs), and `connectors/filesystem.py`, with contract tests in `tests/contract/test_source_connector_contract.py`.
The spec requires all seven source types behind one `Connector` interface, with folders and uploads in wave 1.
Pitfall to avoid: do not derive source identity from filesystem paths alone; canonical URI plus content hash is the identity pair, so a moved-but-identical file is recognized and an in-place edit is detected.

### Desired State

`src/beacon/ingest/connectors/base.py` defines the `Connector` interface: enumerate sources (canonical uri, title, connector kind, metadata) and fetch a source's raw content with its content hash, with typed distinction between a transient fetch failure and a confirmed deletion, because task 02.5 must never retire a source on a transient failure.
`src/beacon/ingest/connectors/folder.py` implements the folder connector: recursive discovery with include/exclude globs, canonical `file://` URIs, content hashing, and media-type detection for the parser.
`src/beacon/server/routes/documents.py` implements `POST /documents` (multipart upload into a raw-document store under the data dir, deduped by content hash, registered as an `upload://` source) and `POST /collections/{c}/sources` (attach a connector-backed source definition such as a folder root or, later, a URL to a collection).
The state DB `sources` table from task 01.3 records every discovered or uploaded source with its content hash and status.
LlamaIndex readers may back later connectors; the interface is ours, and this task documents that adapter seam in `base.py`.

### Gap Analysis

- Missing: the `Connector` interface with the transient-vs-deleted distinction.
- Missing: the folder connector and the raw-document upload store.
- Missing: the documents and sources routes with their API schemas.
- Changes: `src/beacon/server/app.py` registers routers; `src/beacon/models.py` gains request/response schemas.
- Blockers: Epic 01 complete (01.5).

### Implementation Research

The transient-vs-deleted distinction is the load-bearing design point: enumerate must be able to say "this source exists but fetch failed" separately from "this source is gone", mirroring `beacon_kb` `tests/integration/test_sync_transient_fetch_failure.py` (`test_transient_fetch_failure_does_not_retire_indexed_source` and `test_true_deletion_still_retires_source`).
Uploads store raw bytes under a content-addressed path (hash-prefixed) so re-uploading identical content is a no-op returning the existing source, verified by test.
Consult the current FastAPI docs for multipart upload streaming so large files do not buffer fully in memory; the required behavior is bounded memory per upload and a configurable size limit rejected with a 413 problem.
Media-type detection prefers declared content type, falls back to extension, and records the decision in source metadata for the parser.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/ingest/__init__.py` | Establish the ingest package. |
| CREATE | `src/beacon/ingest/connectors/__init__.py` | Expose the connector surface and registry-by-kind lookup. |
| CREATE | `src/beacon/ingest/connectors/base.py` | Define the `Connector` interface with transient-vs-deleted semantics and the LlamaIndex adapter seam note. |
| CREATE | `src/beacon/ingest/connectors/folder.py` | Folder connector with globs, canonical URIs, hashing, and media types. |
| CREATE | `src/beacon/server/routes/documents.py` | Upload endpoint with content-hash dedupe and the sources attach endpoint. |
| MODIFY | `src/beacon/server/app.py` | Register the documents and sources routers. |
| CREATE | `tests/beacon/unit/test_folder_connector.py` | Discovery, globs, identity, hashing, and transient-vs-deleted behavior. |
| CREATE | `tests/beacon/integration/test_documents_api.py` | Upload, dedupe by content hash, size limit, and source registration. |

### Acceptance Criteria

- [ ] The `Connector` interface distinguishes transient fetch failure from confirmed deletion as distinct typed outcomes.
- [ ] The folder connector discovers files by glob, produces stable canonical URIs and content hashes, and reports an unreadable existing file as transient, not deleted.
- [ ] `POST /documents` stores raw content once per content hash; re-uploading identical bytes returns the existing source without a new stored copy.
- [ ] Uploads over the configured size limit are rejected with a 413 problem without buffering the full body.
- [ ] `POST /collections/{c}/sources` records the source definition in the state DB and validates the connector kind.
- [ ] All new routes return taxonomy errors as problem details.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_folder_connector.py tests/beacon/integration/test_documents_api.py
python -m mypy src/beacon
python -m ruff check src/beacon/ingest src/beacon/server
```

---

## Task 02.2: Web and sitemap connector with depth limits and robots respect

Epic: 02 - Ingestion Wave 1
Size: M | 4 files | ~450 LOC

### Current State

Only folder and upload sources exist.
`beacon_kb` deferred its web connector to an extra and never shipped crawling depth or robots handling, so this connector is new behavior guided by the spec's wave-1 scope.
Pitfall to avoid: do not let the crawler follow links off the configured origin or past the depth limit; an unbounded crawl in a sync job would hold the job forever and hammer external sites.

### Desired State

`src/beacon/ingest/connectors/web.py` implements the web connector over httpx: seed URLs or a sitemap.xml, same-origin scope by default, a configurable max depth and max page count, robots.txt fetched and respected per origin, and a per-origin request delay.
Each fetched page becomes a source with canonical URL (normalized: scheme lowered, fragment stripped, tracking params removed per a documented list), content hash of the response body, and media type from headers.
Transient HTTP failures (timeouts, 5xx, connection errors) surface as the transient outcome; a 404/410 on a previously indexed URL surfaces as deleted, feeding the task 02.5 retirement rules.
Sitemap discovery expands `sitemap.xml` and nested sitemap indexes into the seed set, still subject to the same origin, depth, and count limits.
All tests run against a local httpx transport or local test server; no network access in the suite.

### Gap Analysis

- Missing: the entire web connector, URL normalization, robots handling, and crawl limits.
- Changes: `src/beacon/ingest/connectors/__init__.py` registers the `web` kind.
- Blockers: task 02.1 defines the interface.

### Implementation Research

Consult the current httpx docs for mock transports; the required behavior is that every crawl test injects a transport and asserts exact request sequences, including that a disallowed path is never requested.
Use stdlib `urllib.robotparser` semantics as the reference for robots interpretation: fetch failure of robots.txt itself is treated as allow-all with a logged warning, and a robots disallow excludes the URL from both fetch and discovery.
Depth is measured from the seed (seed = 0); the crawler is breadth-first so the page-count limit truncates deterministically.
Canonical URL normalization must be one pure function with table-driven tests, because it defines source identity for the whole web wave.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/ingest/connectors/web.py` | Web and sitemap connector with limits, robots, normalization, and transient-vs-deleted mapping. |
| MODIFY | `src/beacon/ingest/connectors/__init__.py` | Register the `web` connector kind. |
| CREATE | `tests/beacon/unit/test_url_normalization.py` | Table-driven canonical URL normalization tests. |
| CREATE | `tests/beacon/integration/test_web_connector.py` | Crawl limits, robots respect, sitemap expansion, and failure mapping over a mock transport. |

### Acceptance Criteria

- [ ] The crawler never requests a URL outside the configured origin, past max depth, past max pages, or disallowed by robots.txt, asserted by inspecting the mock transport's request log.
- [ ] Sitemap and sitemap-index expansion feeds the seed set and respects the same limits.
- [ ] Canonical URL normalization is deterministic and covered by table-driven tests.
- [ ] Timeouts and 5xx map to the transient outcome; 404/410 maps to deleted.
- [ ] Robots.txt fetch failure allows crawling with a logged warning, and a disallow rule excludes fetch and discovery.
- [ ] No test performs real network I/O.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_url_normalization.py tests/beacon/integration/test_web_connector.py
python -m mypy src/beacon
python -m ruff check src/beacon/ingest
```

---

## Task 02.3: Docling parsing to structured sections

Epic: 02 - Ingestion Wave 1
Size: M | 5 files | ~450 LOC

### Current State

Raw documents arrive from connectors but nothing parses them.
`beacon_kb` ships hand-written Markdown/HTML/PDF parsers (`parsing/markdown.py`, `parsing/html.py`, `parsing/pdf.py`) whose behavioral guarantees are the reference: preserve case, code, tables, and links; emit heading paths; emit typed warnings instead of silently misclassifying structure (see `tests/unit/parsing/`).
The spec replaces these with Docling.
Pitfall to avoid: do not flatten the document to plain text before chunking; the section structure with heading paths is what makes hierarchical chunking and heading-weighted retrieval possible.

### Desired State

`src/beacon/ingest/parsing.py` wraps Docling behind one `parse(raw_document) -> ParsedDocument` function returning a typed structure: document title, ordered sections with heading path (list of ancestor headings), body text with case preserved, section kind (text, code, table, list), and locators (page number for PDF, anchor/offset where available).
A `parser_version` constant captures the Docling major version plus our adapter version and feeds the task 02.5 fingerprint.
Unsupported or corrupt inputs raise typed `ingestion` errors carrying the source uri; recoverable oddities (for example an unrecognized element kind) degrade to text sections with a typed warning list on the result.
Fixtures cover Markdown, HTML, and PDF inputs; parsing is deterministic for identical input bytes.

### Gap Analysis

- Missing: the Docling adapter, the `ParsedDocument` types, and the warning model.
- Missing: parser fixtures for the three wave-1 formats.
- Changes: none outside `src/beacon/ingest/`.
- Blockers: task 02.1 provides raw documents and media types.

### Implementation Research

Consult the current Docling docs for the converter API and its structured document model; do not assume class names.
The required behavior is: given bytes plus a media type, produce the section structure above without writing temp files where avoidable, and map Docling's hierarchy into heading paths such that a section nested under "Install" under "Guide" carries the path ["Guide", "Install"].
Verify how Docling represents code blocks and tables and preserve them as their own section kinds with original text, because the `beacon_kb` parser tests treat case and code preservation as hard requirements to port.
Measure and record cold-start model/asset behavior of Docling in the task notes; if Docling downloads assets on first use, the adapter must surface a clear `ingestion` error pointing at the docs when running fully offline rather than hanging.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/ingest/parsing.py` | Docling adapter producing `ParsedDocument` with sections, heading paths, kinds, locators, warnings, and `parser_version`. |
| CREATE | `tests/beacon/fixtures/docs/sample.md` | Markdown fixture with nested headings, code, and a table. |
| CREATE | `tests/beacon/fixtures/docs/sample.html` | HTML fixture with nested headings and links. |
| CREATE | `tests/beacon/fixtures/docs/sample.pdf` | Small PDF fixture with headings and page structure. |
| CREATE | `tests/beacon/integration/test_parsing.py` | Parse all fixtures; assert structure, preservation, warnings, determinism, and error typing. |

### Acceptance Criteria

- [ ] Markdown, HTML, and PDF fixtures parse into ordered sections with correct heading paths.
- [ ] Case, code blocks, and tables are preserved; code and table sections carry their own kinds.
- [ ] PDF sections carry page locators.
- [ ] Corrupt input raises a typed `ingestion` error naming the source; recoverable oddities produce warnings, not silence.
- [ ] Parsing identical bytes twice yields identical `ParsedDocument` values.
- [ ] `parser_version` is exported and changes force fingerprint incompatibility in task 02.5 by construction.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_parsing.py
python -m mypy src/beacon
python -m ruff check src/beacon/ingest
```

---

## Task 02.4: Hierarchical chunking with parent/child links and heading paths

Epic: 02 - Ingestion Wave 1
Size: M | 5 files | ~450 LOC

### Current State

Parsed documents have structure but no chunks.
`beacon_kb` implements heading-aware parent/child chunking with deterministic IDs and real token overlap in `ingestion/chunking.py` (tests in `tests/unit/ingestion/test_chunking.py`); the spec replaces the implementation with the LlamaIndex hierarchical node parser while keeping the behavioral guarantees.
Pitfall to avoid: do not use random or run-scoped chunk IDs; chunk identity must be deterministic from content and configuration so an unchanged source produces byte-identical chunks and the sync planner can skip it.

### Desired State

`src/beacon/ingest/chunking.py` maps `ParsedDocument` sections into LlamaIndex nodes, runs the hierarchical node parser with configured parent and child sizes and overlap, and converts the result into our typed `Chunk` records carrying: deterministic chunk id (hash of collection, canonical uri, content hash, chunker config, parent locator, child ordinal), parent chunk id, heading path, section kind, text, and locators.
Chunker configuration (sizes, overlap, parser choice) is captured as a canonical `chunker_config` string feeding the task 02.5 fingerprint.
Chunks map one-to-one onto the task 01.2 payload schema; parent chunks are stored as payload context (retrievable for expansion) rather than as separately ranked points, and the chosen representation is documented in the module docstring.
Empty documents produce zero chunks without error; a section larger than the child size splits with real overlap, and overlap is never misread as a minimum chunk length.

### Gap Analysis

- Missing: the section-to-node mapping, the hierarchical parse, and the node-to-`Chunk` conversion with deterministic IDs.
- Missing: the `chunker_config` canonical string.
- Changes: `src/beacon/storage/payload.py` may gain the parent-context field if not already present.
- Blockers: task 02.3 provides `ParsedDocument`.

### Implementation Research

Consult the current llama-index-core docs for the hierarchical node parser and node relationship metadata; do not assume constructor signatures.
The required behavior is: parent nodes cover contiguous child ranges, every child links to exactly one parent, heading paths survive the round trip into node metadata and back, and the parser's own IDs are discarded in favor of our deterministic IDs.
Determinism must be proven by chunking the same fixture twice in separate processes' worth of state (fresh objects) and comparing full chunk lists.
Record how the LlamaIndex version affects output (node boundaries can shift between versions) and pin the dependency tightly enough that `chunker_config` plus the pin makes fingerprints honest.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/ingest/chunking.py` | Hierarchical chunking adapter with deterministic IDs, parent links, heading paths, and `chunker_config`. |
| MODIFY | `src/beacon/storage/payload.py` | Ensure parent-context and heading-path payload fields cover the chunk mapping. |
| MODIFY | `pyproject.toml` | Pin llama-index-core tightly enough for chunk determinism across installs. |
| CREATE | `tests/beacon/unit/test_chunking.py` | Determinism, parent/child integrity, overlap, heading paths, empty and oversized inputs. |
| CREATE | `tests/beacon/unit/test_chunk_payload_mapping.py` | Chunk-to-payload round trip completeness. |

### Acceptance Criteria

- [ ] Chunking the same `ParsedDocument` with the same config twice yields identical chunk IDs, texts, and parent links.
- [ ] Every child chunk references exactly one parent and carries the heading path of its section.
- [ ] Configured overlap produces real shared tokens between adjacent children and does not act as a minimum chunk length.
- [ ] Changing any chunker config value changes `chunker_config` and therefore chunk IDs.
- [ ] Chunks convert losslessly to the payload schema and back for the fields retrieval needs.
- [ ] Empty documents yield zero chunks without error.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_chunking.py tests/beacon/unit/test_chunk_payload_mapping.py
python -m mypy src/beacon
python -m ruff check src/beacon/ingest
```

---

## Task 02.5: Embeddings auto-detect and the incremental sync engine with staged promotion

Epic: 02 - Ingestion Wave 1
Size: L | 10 files | ~700 LOC

### Current State

Connectors, parsing, and chunking exist but nothing embeds chunks or orchestrates a sync, and the REST surface has no sync endpoints.
`beacon_kb` `ingestion/sync.py` (SyncEngine), `ingestion/planning.py`, `indexing/coordinator.py`, and `indexing/validation.py` define the reference semantics, with the regression suites this task must port: `tests/integration/test_sync_lifecycle.py` (`test_unchanged_second_sync_zero_writes`), `test_sync_rollback.py` (failure at every stage leaves the prior revision searchable), `test_sync_transient_fetch_failure.py` (fetch failure never retires an indexed source), and `test_fingerprint_migration.py` (fingerprint composition and incompatibility).
Pitfall to avoid: do not compare content hashes alone to decide work; the pipeline fingerprint (parser + chunker config + embedding model + dimension + schema version) must be compared on every sync so a config change triggers a full rebuild instead of serving mixed-generation chunks.

### Desired State

`src/beacon/ingest/embeddings.py` provides the embedding provider with auto-detect: cloud providers through LiteLLM when API keys are present in the environment, otherwise local sentence-transformers embeddings, otherwise a sparse-only floor where no dense vectors are produced and search runs sparse-only; the active mode is exposed for diagnostics and the model name and dimension feed the fingerprint.
Sparse representations for hybrid search are computed for every chunk in all modes.
`src/beacon/ingest/fingerprint.py` computes the pipeline fingerprint from parser version, chunker config, embedding model name, embedding dimension, and payload schema version.
`src/beacon/ingest/planner.py` classifies every known and discovered source into new, changed, deleted, unchanged, or incompatible (fingerprint drift forces all sources incompatible), using content hashes from connectors and the state DB, and applying the transient-failure rule: a transient fetch outcome keeps the source and its indexed chunks, records a warning, and never retires the source.
`src/beacon/ingest/sync.py` executes the plan as one staged operation: parse, chunk, and embed only new/changed/incompatible sources; write all surviving chunks to a shadow collection via the task 01.2 lifecycle; validate (point counts, vector dimensions, fingerprint match); flip the alias; then update sources, revisions, and the job record in the state DB.
`src/beacon/server/routes/sync.py` serves `POST /collections/{c}/sync` returning 202 with a job id (job runs as an async background task with durable state transitions) and `GET /jobs/{id}` serving job state and the change-plan summary from the state DB.
An unchanged sync performs zero parse, zero embed, and zero Qdrant write calls, asserted by counting fakes; any stage failure aborts the stage, leaves the prior collection serving, and records a failed recoverable job.

### Gap Analysis

- Missing: embedding auto-detect with the three modes and sparse computation.
- Missing: fingerprint computation, change planning, and the staged sync engine.
- Missing: sync and jobs REST endpoints with async job execution.
- Changes: `src/beacon/server/app.py` registers the sync router; `src/beacon/models.py` gains job and sync schemas.
- Blockers: tasks 02.2, 02.4, and Epic 01.

### Implementation Research

Consult the current LiteLLM docs for embedding calls and provider key detection; the required behavior is that detection reads only environment configuration, never performs a probe call at import, and the chosen provider is logged once per process.
Consult the current qdrant-client docs for the sparse representation Qdrant expects at upsert; the required behavior is that every point carries the named sparse vector from task 01.2 and, in dense modes, the named dense vector with the fingerprinted dimension.
Port the counting-fake technique from `beacon_kb` `testing.py`: fake parser, embedder, and store wrappers that count calls, so the zero-work assertion for unchanged syncs is by call counts, not timing.
Async jobs use FastAPI background execution over the durable job record; the job function must transition state in the DB at every boundary so `GET /jobs/{id}` is truthful mid-run and after a crash the job reads as failed-or-running, never silently lost.
Validation before flip mirrors `beacon_kb` `indexing/validation.py`: staged point count equals planned count, dimensions match the fingerprint, and the revision fingerprint equals the computed fingerprint.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/ingest/embeddings.py` | Embedding auto-detect (LiteLLM, local, sparse-only floor) plus sparse computation. |
| CREATE | `src/beacon/ingest/fingerprint.py` | Pipeline fingerprint from parser, chunker config, model, dimension, schema version. |
| CREATE | `src/beacon/ingest/planner.py` | Change classification with the transient-failure retention rule. |
| CREATE | `src/beacon/ingest/sync.py` | Staged sync engine: selective work, shadow write, validation, alias flip, state updates. |
| CREATE | `src/beacon/server/routes/sync.py` | `POST /collections/{c}/sync` (202 + job) and `GET /jobs/{id}`. |
| MODIFY | `src/beacon/server/app.py` | Register the sync router and job runner wiring. |
| CREATE | `tests/beacon/integration/test_sync_lifecycle.py` | Full sync, add/modify/delete, unchanged-sync zero work by call counts, restart durability. |
| CREATE | `tests/beacon/integration/test_sync_rollback.py` | Failure injected at parse, chunk, embed, stage-write, validate, and promote each leaves the prior collection serving and records a failed job. |
| CREATE | `tests/beacon/integration/test_sync_transient_fetch_failure.py` | Transient failure never retires an indexed source; true deletion retires it. |
| CREATE | `tests/beacon/integration/test_fingerprint_migration.py` | Fingerprint determinism, per-component sensitivity, and incompatible-triggers-rebuild. |

### Acceptance Criteria

- [ ] With provider keys in the env the embedder uses LiteLLM; without keys but with local embeddings available it uses them; with neither it enters the sparse-only floor; the active mode is queryable.
- [ ] The fingerprint is deterministic and changes when any of parser version, chunker config, embedding model, dimension, or schema version changes.
- [ ] An unchanged second sync performs zero parse, zero embed, and zero store-write calls, asserted by counting fakes.
- [ ] A transient fetch failure on an indexed source keeps the source active and its chunks serving, and the sync report carries a warning; a confirmed deletion retires the source and removes its chunks from the new revision.
- [ ] A failure injected at every stage (parse, chunk, embed, stage write, validation, promote) leaves the prior collection serving through the alias and records a recoverable failed job with problem detail.
- [ ] Fingerprint drift marks all sources incompatible and rebuilds everything into the shadow before one flip.
- [ ] `POST /collections/{c}/sync` returns 202 with a job id; `GET /jobs/{id}` reports pending/running/succeeded/failed truthfully, including after process restart.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_sync_lifecycle.py tests/beacon/integration/test_sync_rollback.py tests/beacon/integration/test_sync_transient_fetch_failure.py tests/beacon/integration/test_fingerprint_migration.py
python -m pytest tests/unit tests/contract tests/integration -q
python -m mypy src/beacon
python -m ruff check src/beacon
```

---

## Task 03.1: Hybrid retrieval with enforced payload filters and optional rerank

Epic: 03 - Retrieval and Grounded Answers
Size: L | 7 files | ~550 LOC

### Current State

Synced collections hold dense and sparse vectors with full payloads, but nothing queries them.
`beacon_kb` implements hybrid retrieval as separate sparse/dense retrievers with hand-rolled RRF (`retrieval/sparse.py`, `retrieval/dense.py`, `retrieval/fusion.py`) and boundary-enforced filters (`retrieval/filters.py`, proven by `tests/integration/test_retrieval_pipeline.py::test_injected_override_retriever_honours_filter_spec`); the spec replaces the mechanics with the Qdrant Query API's native RRF while keeping the guarantees.
Pitfall to avoid: do not apply filters after retrieval or leave them to retriever implementations; filters must be compiled into the Qdrant query itself at the pipeline boundary so no retriever implementation can bypass them.

### Desired State

`src/beacon/retrieval/filters.py` defines the typed `FilterSpec` (collection, source uris, tags, date ranges) and one compiler from `FilterSpec` to a Qdrant payload filter; the compiler is the only path from user filters to the store.
`src/beacon/retrieval/hybrid.py` executes one Qdrant Query API request per search: dense and sparse prefetch branches fused with native RRF, the compiled payload filter applied to every branch, and a sparse-only degraded mode when the collection or config has no dense vectors.
`src/beacon/retrieval/rerank.py` provides the optional cross-encoder reranker over the fused candidates, bounded to the candidate list (never fetching new candidates), lazily importing sentence-transformers only when enabled.
Results are typed `Hit` records carrying chunk id, payload, and per-stage scores with documented direction; the retriever performs zero LLM calls by construction.
The retrieval entry point takes `(query_text, filter_spec, top_k)` and is the single search path that tasks 03.2 through 03.4 and Epic 05 reuse.

### Gap Analysis

- Missing: `FilterSpec`, the filter compiler, the hybrid query, the degraded mode, and the optional reranker.
- Missing: retrieval-side typed `Hit` models in `src/beacon/models.py` or a retrieval-local module.
- Changes: none outside `src/beacon/retrieval/` and models.
- Blockers: task 02.5 produces queryable collections.

### Implementation Research

Consult the current qdrant-client docs for the Query API prefetch-and-fusion capability and for filter syntax over payload indexes; do not assume request shapes.
The required behavior is: one request per search containing both branches and the fusion directive, with the filter attached so filtering happens inside Qdrant on indexed payload fields, and query-time embedding of the query text using the same embedding mode the collection was synced with (mode mismatch is a typed `backend` error, not silent wrong results).
Filter enforcement is proven the same way `beacon_kb` proved it: a test injects an alternative retriever implementation through the seam and asserts the compiled filter still constrains results, because compilation happens before the implementation is invoked.
Consult the current sentence-transformers docs for cross-encoder usage; required behavior is batch scoring of (query, chunk text) pairs, stable sorting, and full determinism given the same model and inputs.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/retrieval/__init__.py` | Expose the retrieval surface. |
| CREATE | `src/beacon/retrieval/filters.py` | `FilterSpec` and the single boundary compiler to Qdrant payload filters. |
| CREATE | `src/beacon/retrieval/hybrid.py` | One-request hybrid query with native RRF, degraded sparse-only mode, and typed hits. |
| CREATE | `src/beacon/retrieval/rerank.py` | Optional bounded cross-encoder rerank with lazy import. |
| CREATE | `tests/beacon/integration/test_hybrid_retrieval.py` | Hybrid vs sparse-only, RRF fusion presence, scores, and determinism against embedded Qdrant. |
| CREATE | `tests/beacon/integration/test_filter_enforcement.py` | Filters constrain results and cannot be bypassed by an injected retriever implementation. |
| CREATE | `tests/beacon/unit/test_rerank.py` | Rerank ordering, boundedness, and lazy-import behavior with a fake scorer. |

### Acceptance Criteria

- [ ] A search issues exactly one Qdrant query containing dense and sparse prefetch branches fused with native RRF, verified via a recording wrapper.
- [ ] Collection, source, tag, and date filters restrict results correctly and are applied inside the Qdrant query, not post-hoc.
- [ ] An injected alternative retriever implementation still operates under the compiled filter, proving boundary enforcement.
- [ ] With no dense vectors configured, search degrades to sparse-only and still returns results.
- [ ] Rerank reorders only the fused candidates, never fetches new ones, and is skipped entirely when disabled with no sentence-transformers import.
- [ ] The search path performs zero LLM calls, asserted with a counting fake provider.
- [ ] Embedding-mode mismatch between query and collection raises a typed `backend` error.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_hybrid_retrieval.py tests/beacon/integration/test_filter_enforcement.py tests/beacon/unit/test_rerank.py
python -m mypy src/beacon
python -m ruff check src/beacon/retrieval
```

---

## Task 03.2: Evidence assembly with expansion, budgets, labels, and snippets

Epic: 03 - Retrieval and Grounded Answers
Size: M | 5 files | ~500 LOC

### Current State

Retrieval returns ranked hits but no evidence suitable for citation.
`beacon_kb` `retrieval/context.py`, `retrieval/snippets.py`, and `retrieval/pipeline.py` define the reference semantics, proven by `tests/integration/test_retrieval_pipeline.py`: expansion only after final ordering, token budgets enforced, gap-free `[S1]` labels after overflow (`test_gap_free_labels_after_overflow`, `test_oversized_rank2_skipped_and_labels_contiguous`), match-centered snippets (`test_snippet_centers_match_not_prefix`), and preserved provenance (`test_snippet_source_uri_is_canonical_not_hash`).
Pitfall to avoid: do not let a skipped oversized chunk leave a hole in the label sequence; labels are assigned to packed evidence in final order so the sequence S1, S2, S3 is always contiguous.

### Desired State

`src/beacon/retrieval/evidence.py` converts final-ordered hits into an `EvidenceBundle`: parent/context expansion (fetch parent text from payload for each packed hit) applied only after ordering and only within the token budget, primary hits packed before context, context spans marked with a `context_of` reference and no relevance score, and stable labels S1..Sn assigned gap-free to packed evidence.
`src/beacon/retrieval/snippets.py` produces match-centered snippets: locate the best query-term window in the chunk text, center the snippet there, and carry real source uri, title, heading path, and locator from the payload.
Token accounting uses one documented counter (heuristic acceptable, stated in the module) and the bundle records a budget recap: requested, packed, skipped, and token totals.
The bundle is the canonical evidence input for task 03.3 citation validation and the `POST /search` response shape in task 03.4.

### Gap Analysis

- Missing: evidence packing, expansion, budgets, labels, and snippets in the new package.
- Changes: `src/beacon/models.py` gains `Evidence` and `EvidenceBundle` schemas shared by search and answer responses.
- Blockers: task 03.1 provides ordered hits with payloads.

### Implementation Research

Port the packing algorithm shape from `beacon_kb` `retrieval/context.py`: pack primary hits in final order while budget remains, skip any hit whose text exceeds remaining budget and continue with the next (labels stay contiguous over packed items), then expand context for packed hits with remaining budget.
Snippet centering follows `retrieval/snippets.py`: find the query-term match span, expand to the window size around it, fall back to the chunk center when no term matches, and always record the span so tests can assert the match is inside the snippet.
Deduplicate by chunk id across primary and context so the same chunk never appears twice with two labels.
Keep every function pure over its inputs so the whole module is unit-testable without Qdrant.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/retrieval/evidence.py` | Budgeted packing, post-ordering expansion, gap-free labels, and the budget recap. |
| CREATE | `src/beacon/retrieval/snippets.py` | Match-centered snippets with real provenance fields. |
| MODIFY | `src/beacon/models.py` | Add `Evidence` and `EvidenceBundle` API schemas. |
| CREATE | `tests/beacon/unit/test_evidence_assembly.py` | Packing order, budget enforcement, oversized-skip label contiguity, context_of marking, dedupe. |
| CREATE | `tests/beacon/unit/test_snippets.py` | Match centering, fallback, span-in-snippet, and provenance preservation. |

### Acceptance Criteria

- [ ] Primary hits pack before context, expansion happens only after final ordering, and the token budget is never exceeded.
- [ ] Skipping an oversized hit leaves labels contiguous: packed evidence is always labeled S1..Sn with no gaps.
- [ ] Context spans carry `context_of` and no relevance score, and are distinguishable from primary hits.
- [ ] No chunk id appears twice in a bundle.
- [ ] Snippets center on the query match, fall back to center when no match exists, and carry the canonical source uri, title, heading path, and locator (never an internal hash as the uri).
- [ ] The bundle's budget recap reports requested, packed, skipped, and token totals accurately.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_evidence_assembly.py tests/beacon/unit/test_snippets.py
python -m mypy src/beacon
python -m ruff check src/beacon/retrieval
```

---

## Task 03.3: Grounded answer with abstention, injection defense, and citation validation

Epic: 03 - Retrieval and Grounded Answers
Size: L | 8 files | ~650 LOC

### Current State

Evidence bundles exist but nothing generates answers.
`beacon_kb` `generation/` is the reference implementation: `abstention.py` (deterministic `should_pre_abstain` with zero LLM calls, `is_post_abstain` sentinel detection), `prompts.py` (`neutralize_delimiters`, `build_context_block`, `build_user_message` with untrusted-context delimiters), `citations.py` (`extract_cited_labels`, `resolve_citations`, `validate_no_unknown_evidence_ids`), and `answer.py` (`run_answer` staging), with the adversarial suites in `tests/integration/test_grounded_answer.py` including `test_hostile_generator_fabricated_evidence_is_rejected`, `test_literal_close_delimiter_in_evidence_is_neutralized`, and the one-call cost-contract classes.
Pitfall to avoid: do not validate citations against evidence text echoed back by the model; validation must run against the retrieval-derived canonical evidence bundle held by the server, so a hostile model cannot smuggle fabricated evidence into the response.

### Desired State

`src/beacon/answer/abstention.py` ports pre-abstention (deterministic, zero LLM calls, triggered by empty evidence or all scores below the configured policy threshold) and post-abstention (sentinel text or provider abstained flag) with the same default-threshold semantics `beacon_kb` settled on for RRF-scale scores.
`src/beacon/answer/prompts.py` ports the untrusted-context discipline: versioned prompt constants, a context block that wraps every piece of evidence between open and close delimiters, delimiter-token neutralization applied to all evidence text so a literal close delimiter in a document cannot escape the block, and a user message instructing citation by `[S#]` label only.
`src/beacon/answer/citations.py` ports structural validation: extract cited labels from the answer text, resolve each against the canonical evidence bundle from task 03.2, reject unknown labels with a typed `citation` error, and emit typed citation records referencing evidence by id.
`src/beacon/answer/generate.py` orchestrates the stages: pre-abstain gate (zero calls), exactly one LiteLLM chat call over the built prompt, post-abstain gate, citation validation, and a typed `AnswerResult` preserving answer text, citations, the evidence bundle, and diagnostics (prompt version, model, token counts from the provider response, elapsed stages); the exactly-one-call contract is structural: the provider client is invoked at most once per answer and zero times on pre-abstention.
Both the answer and abstention paths run citation validation, matching the ported guarantee.

### Gap Analysis

- Missing: the four answer-path modules and the `AnswerResult` schema.
- Missing: the ported adversarial and cost-contract test suites.
- Changes: `src/beacon/models.py` gains `AnswerResult` and citation schemas.
- Blockers: task 03.2 provides the canonical evidence bundle.

### Implementation Research

Consult the current LiteLLM docs for the chat completion call, usage/token fields on responses, and error types; the required behavior is one call with configured model and timeout, provider errors translated to typed `backend` errors, and token usage captured into diagnostics when the provider reports it.
Port the delimiter scheme from `beacon_kb` `prompts.py` conceptually: unique open/close delimiter tokens, neutralization rewriting any literal occurrence of either token inside evidence text, and a context block that always emits both delimiters even for empty evidence.
Port the citation grammar: labels are `[S<n>]`, extraction tolerates adjacent punctuation and repeated citations, resolution is strict, and validation failure is an error, never a silent drop.
The abstention threshold must be expressed against the score scale actually produced by task 03.1 (RRF-scale), with the default chosen so that normal fused scores never spuriously pre-abstain; port the regression that documents this (`test_rrf_scale_score_does_not_pre_abstain_with_default_config`).

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/answer/__init__.py` | Expose the answer surface. |
| CREATE | `src/beacon/answer/abstention.py` | Deterministic pre-abstention and sentinel post-abstention. |
| CREATE | `src/beacon/answer/prompts.py` | Versioned prompts, untrusted-context delimiters, and neutralization. |
| CREATE | `src/beacon/answer/citations.py` | Label extraction, canonical resolution, unknown-label rejection, citation records. |
| CREATE | `src/beacon/answer/generate.py` | Stage orchestration with the exactly-one-call structure and diagnostics. |
| MODIFY | `src/beacon/models.py` | Add `AnswerResult` and citation API schemas. |
| CREATE | `tests/beacon/unit/test_abstention_prompts_citations.py` | Pre/post abstention, delimiter neutralization, extraction and resolution unit coverage. |
| CREATE | `tests/beacon/integration/test_grounded_answer.py` | One-call and zero-call cost contracts, hostile-generator fabricated-evidence rejection, injection-bearing evidence stays inside delimiters, evidence preserved in the result. |

### Acceptance Criteria

- [ ] Empty or below-threshold evidence abstains deterministically with zero provider calls, and the default threshold does not spuriously abstain on RRF-scale scores.
- [ ] A normal answer performs exactly one provider call; post-abstention still counts exactly one; pre-abstention counts zero; all asserted with a counting fake.
- [ ] Every evidence text is enclosed between the delimiters, and a literal close-delimiter token inside evidence is neutralized so it cannot terminate the untrusted block, ported from the adversarial suite.
- [ ] A cited label not present in the canonical evidence bundle raises a typed `citation` error; a hostile generator fabricating evidence or citing out-of-bundle labels is rejected.
- [ ] Citation validation runs on both the answer and abstention paths.
- [ ] `AnswerResult` preserves the evidence bundle and citations alongside the answer text, plus diagnostics with prompt version, model, and token counts.
- [ ] Provider failures surface as typed `backend` errors with no secret material in messages.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/unit/test_abstention_prompts_citations.py tests/beacon/integration/test_grounded_answer.py
python -m mypy src/beacon
python -m ruff check src/beacon/answer
```

---

## Task 03.4: REST search and answer with cost-contract, recall, and smoke tests

Epic: 03 - Retrieval and Grounded Answers
Size: M | 6 files | ~500 LOC

### Current State

The retrieval and answer pipelines work in-process but are not exposed over REST, and no test exercises the full default-config path from sync to cited answer.
The spec API surface requires `POST /search` and `POST /answer`, and the testing section requires counting-fake cost-contract tests at the API level, natural-language recall tests, and an end-to-end smoke.
Pitfall to avoid: do not let route handlers re-implement any pipeline logic; the routes are thin adapters over the task 03.1-03.3 functions so the cost contracts proven in-process hold identically over HTTP.

### Desired State

`src/beacon/server/routes/search.py` serves `POST /search`: request carries query text, collection, optional filters (sources, tags, date range), and top-k; response carries the evidence bundle with labels, snippets, provenance, scores, and the budget recap; a readiness problem is returned for a collection that is not ready.
`src/beacon/server/routes/answer.py` serves `POST /answer`: request mirrors search plus answer options; response is the `AnswerResult` including abstention outcome as a first-class field, never an error.
API-level cost-contract tests wire a counting fake LLM provider through the app factory and assert: `POST /search` performs zero provider calls; `POST /answer` performs exactly one (zero on pre-abstention).
Natural-language recall tests sync a fixture corpus of distinct topical documents and assert that plain-English questions retrieve the expected source in the top results and that answers cite it.
The default-config smoke test drives the full journey with default settings and no credentials (sparse-only floor plus fake LLM injected only for the answer step): create collection -> attach folder source -> sync -> readyz ready -> search -> answer with a valid citation.

### Gap Analysis

- Missing: the search and answer routes and their API schemas.
- Missing: API-level cost-contract, recall, and smoke suites with a fixture corpus.
- Changes: `src/beacon/server/app.py` registers the routers; `src/beacon/models.py` gains request schemas.
- Blockers: task 03.3.

### Implementation Research

The counting fake is injected through the app factory's dependency seam, not by monkeypatching, so the test proves the production wiring path has no hidden provider call.
Recall fixtures port the spirit of the `beacon_kb` natural-language tests: several short documents on clearly distinct topics, questions phrased conversationally rather than as keyword echoes, and assertions on which source uri appears in top-k, keeping thresholds loose enough to be model-independent in sparse-only mode.
Answer responses must serialize abstention as data (answered false, reason) with HTTP 200, because abstention is a correct outcome, not a failure; only taxonomy errors produce problem responses.
Route latency accounting hooks into the task 01.4 telemetry helpers so Epic 06 can extend spans without touching handlers.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `src/beacon/server/routes/search.py` | `POST /search` over the retrieval and evidence pipeline. |
| CREATE | `src/beacon/server/routes/answer.py` | `POST /answer` over the grounded answer pipeline with abstention as data. |
| MODIFY | `src/beacon/server/app.py` | Register routers and the provider dependency seam. |
| CREATE | `tests/beacon/fixtures/corpus/` | Small topical fixture corpus for recall and smoke tests. |
| CREATE | `tests/beacon/integration/test_search_answer_api.py` | Route behavior, filters, readiness problems, and API-level cost contracts with the counting fake. |
| CREATE | `tests/beacon/integration/test_e2e_smoke.py` | Default-config journey: create, attach, sync, ready, search, cited answer; plus natural-language recall assertions. |

### Acceptance Criteria

- [ ] `POST /search` returns labeled evidence with snippets, provenance, scores, and the budget recap, honoring filters and top-k.
- [ ] `POST /search` performs zero LLM provider calls and `POST /answer` exactly one (zero on pre-abstention), asserted at the API level via the dependency seam.
- [ ] Querying a not-ready collection returns a `readiness` problem; abstention returns 200 with answered=false and a reason.
- [ ] Natural-language recall tests pass over the fixture corpus in sparse-only mode.
- [ ] The default-config smoke completes the full journey with no credentials and asserts the final answer cites a label resolving to the expected source.
- [ ] Epic 03 exit criterion met: search and answer are fully exercised over REST with all Epic 03 in-process guarantees holding over HTTP.
- [ ] `mypy --strict` and ruff pass.

### Validation Steps

```bash
python -m pytest tests/beacon/integration/test_search_answer_api.py tests/beacon/integration/test_e2e_smoke.py
python -m pytest tests/beacon -q
python -m pytest tests/unit tests/contract tests/integration -q
python -m mypy src/beacon
python -m ruff check src/beacon
```

---

## Epic 04: MCP and CLI (task list; detailed just-in-time before this epic starts)

| Task | Title | Scope (one line) | Depends On | Files Sketch |
|------|-------|------------------|------------|--------------|
| 04.1 | FastMCP server in-process | Mount FastMCP on the app (`/mcp` streamable HTTP) exposing kb_search, kb_answer, kb_sync_status, kb_list_collections over the same handlers as REST. | 03.4 | `src/beacon/server/mcp.py`, `tests/beacon/integration/test_mcp_tools.py` |
| 04.2 | MCP stdio mode and error mapping | `beacon mcp-stdio` entry running the same tool set over stdio; taxonomy errors mapped to typed MCP tool errors. | 04.1 | `src/beacon/server/mcp_stdio.py`, `tests/beacon/integration/test_mcp_stdio.py` |
| 04.3 | Typer CLI | `beacon serve / sync / search / ask` over the in-process app or a remote base URL; problem details rendered humanely; stable exit codes. | 03.4 | `src/beacon/cli.py`, `tests/beacon/integration/test_cli.py` |
| 04.4 | Model auto-detect UX and doctor | Startup provider report (which LLM/embedding mode and why), `beacon doctor` diagnostics with exact-fix messages, completed Ollama compose profile. | 04.3 | `src/beacon/cli.py`, `docker-compose.yml`, `tests/beacon/unit/test_doctor.py` |

Cost contracts carry over unchanged: MCP `kb_search` performs zero LLM calls and `kb_answer` exactly one, asserted with the same counting-fake seam.

---

## Epic 05: Investigate (task list; detailed just-in-time before this epic starts)

| Task | Title | Scope (one line) | Depends On | Files Sketch |
|------|-------|------------------|------------|--------------|
| 05.1 | LangGraph skeleton, budgets, checkpointing | Graph state model, explicit budget (LLM calls, retrievals, tokens, wall clock), SQLite checkpointer wiring, budget exhaustion as a typed `budget` outcome. | 03.4 | `src/beacon/investigate/graph.py`, `src/beacon/investigate/budget.py`, `tests/beacon/unit/test_budget.py` |
| 05.2 | Plan, retrieve, grade, reflect, synthesize nodes | Nodes reusing the Epic 03 retrieval path and answer generator; deterministic heuristic grader default; synthesis through the task 03.3 citation-validated path. | 05.1 | `src/beacon/investigate/nodes.py`, `src/beacon/investigate/synthesis.py`, `tests/beacon/integration/test_investigate_loop.py` |
| 05.3 | POST /investigate with SSE trace and kb_investigate | SSE-streamed step trace over REST, final cited result event, and the MCP tool over the same handler. | 05.2, 04.1 | `src/beacon/server/routes/investigate.py`, `src/beacon/server/mcp.py`, `tests/beacon/integration/test_investigate_api.py` |
| 05.4 | Budget-regression and cost-contract gates | Counting-fake proof the loop never exceeds any budget dimension and always returns an inspectable trace, including on exhaustion. | 05.2 | `tests/beacon/integration/test_investigate_budgets.py` |

---

## Epic 06: Evaluation and Hardening (task list; detailed just-in-time before this epic starts)

| Task | Title | Scope (one line) | Depends On | Files Sketch |
|------|-------|------------------|------------|--------------|
| 06.1 | RAGAS golden-set gates in CI | Add ragas to the dev group (deferred from 01.1), curate the golden set, gate CI on faithfulness, answer relevancy, and context precision/recall thresholds. | 03.4 | `src/beacon/evals/ragas_gates.py`, `tests/beacon/evaluation/`, CI workflow file |
| 06.2 | Tracing and cost accounting | OTel spans around every pipeline stage, structured logs, per-request token and cost accounting via LiteLLM callbacks, secrets never logged (audited). | 03.4 | `src/beacon/server/telemetry.py`, `src/beacon/answer/generate.py`, `tests/beacon/integration/test_telemetry.py` |
| 06.3 | Load sanity, docs, quickstarts | Basic load sanity checks against compose, docs for REST/MCP/CLI, docker and pipx quickstarts verified end to end. | 06.1, 06.2 | `docs/`, `README.md`, `tests/beacon/performance/` |

---

## Epic 07: Connector Waves 2 and 3 (task list; detailed just-in-time before this epic starts)

Each connector implements the task 02.1 `Connector` interface with an injected client, env-var credentials only, incremental change detection feeding the task 02.5 planner, and offline contract tests over recorded fixtures; LlamaIndex readers are wrapped where they fit.

| Task | Title | Scope (one line) | Depends On | Files Sketch |
|------|-------|------------------|------------|--------------|
| 07.1 | Confluence connector (wave 2) | Spaces/pages enumeration, canonical page URIs, version-based change detection. | 02.5 | `src/beacon/ingest/connectors/confluence.py`, `tests/beacon/integration/test_confluence_connector.py` |
| 07.2 | Notion connector (wave 2) | Databases/pages enumeration, block-to-document flattening, last-edited change detection. | 02.5 | `src/beacon/ingest/connectors/notion.py`, `tests/beacon/integration/test_notion_connector.py` |
| 07.3 | Google Drive connector (wave 3) | Folder/file enumeration, export of Google-native formats, revision-based change detection. | 02.5 | `src/beacon/ingest/connectors/gdrive.py`, `tests/beacon/integration/test_gdrive_connector.py` |
| 07.4 | Slack connector (wave 3) | Channel history windows as documents, thread grouping, cursor-based incremental fetch. | 02.5 | `src/beacon/ingest/connectors/slack.py`, `tests/beacon/integration/test_slack_connector.py` |
| 07.5 | GitHub connector (wave 3) | Repo docs (markdown, wikis, optionally issues), ref-pinned enumeration, commit-based change detection. | 02.5 | `src/beacon/ingest/connectors/github.py`, `tests/beacon/integration/test_github_connector.py` |

---

## Epic 08: Cleanup and Release (task list; detailed just-in-time before this epic starts)

| Task | Title | Scope (one line) | Depends On | Files Sketch |
|------|-------|------------------|------------|--------------|
| 08.1 | Delete beacon_kb after parity check | Verify every ported guarantee has a green `tests/beacon` counterpart, then delete `src/beacon_kb`, the old test tree, and superseded planning docs, and simplify `pyproject.toml` to the new package. | Epics 04-07 complete | Deletions repo-wide, `pyproject.toml` |
| 08.2 | Final docs and v1.0.0 | Final documentation pass, release checklist, version 1.0.0, and the release PR. | 08.1 | `README.md`, `docs/`, version metadata |
