# Implementation Epics V2: Beacon Industry-Standard Knowledge-Base RAG Server

*Source of truth: docs/superpowers/specs/2026-07-23-beacon-industry-standard-rag-design.md*
*Detailed plans: IMPL_PLANS_V2.md | Tracker: FOCUS_CHAIN_V2.md*

## Product Summary

Beacon is a self-hosted knowledge-base RAG server that anyone can run and integrate.
It is built on industry-standard components (Docling, LlamaIndex, Qdrant, LiteLLM, LangGraph, FastAPI, FastMCP, Typer, RAGAS) rather than bespoke infrastructure.
Day-one consumers are AI agents via MCP and developers via a REST API.
Deployment is local-first: `docker compose up`, or `pipx install beacon-kb && beacon serve` for a zero-infrastructure single process.
The differentiators are operational and safety behaviors ported from the existing `beacon_kb` codebase: never-break-the-live-corpus sync, structural citation validation, deterministic abstention, prompt-injection defenses, and enforced cost contracts.
The existing `beacon_kb` package and its 1,101 tests serve as an executable spec during the port and are deleted in the final cleanup epic.
All new code lives in a new top-level package `src/beacon/` with its own test tree under `tests/beacon/`.

## Epic Overview

| # | Epic | Depends On | Tasks | Detail Level |
|---|------|------------|-------|--------------|
| 01 | Core Service and Storage | None | 5 | Fully planned in IMPL_PLANS_V2.md |
| 02 | Ingestion Wave 1 | Epic 01 | 5 | Fully planned in IMPL_PLANS_V2.md |
| 03 | Retrieval and Grounded Answers | Epic 02 | 4 | Fully planned in IMPL_PLANS_V2.md |
| 04 | MCP and CLI | Epic 03 | 4 | Task list; detailed just-in-time |
| 05 | Investigate | Epic 03 (05.3 also 04.1) | 4 | Task list; detailed just-in-time |
| 06 | Evaluation and Hardening | Epic 03 | 3 | Task list; detailed just-in-time |
| 07 | Connector Waves 2 and 3 | Epic 03 (interface from Epic 02) | 5 | Task list; detailed just-in-time |
| 08 | Cleanup and Release | All previous epics | 2 | Task list; detailed just-in-time |

## Parallelism Map

Epics 01, 02, and 03 are strictly sequential and form the spine of the product.
After Epic 03 completes, Epics 04, 05, and 06 run in parallel worktrees; the only cross-link is that task 05.3 (the `kb_investigate` MCP tool) additionally depends on task 04.1 (the FastMCP server).
Epic 07 connector waves also start independently after Epic 03; each connector task depends only on the Connector interface and sync engine from Epic 02, and wave 2 (Confluence, Notion) precedes wave 3 (Google Drive, Slack, GitHub) only as a scheduling preference, not a hard dependency.
Epic 08 closes last, after every other epic has merged.
Each epic is one branch and one PR; each task is one independent green commit; the per-task and whole-branch review process from the previous generation continues unchanged.

```text
Epic 01 -> Epic 02 -> Epic 03 -+-> Epic 04 (MCP + CLI) ------+
                               |         \                    |
                               |          +-> 05.3 needs 04.1 |
                               +-> Epic 05 (Investigate) -----+-> Epic 08 (Cleanup, release)
                               +-> Epic 06 (Eval + hardening)-+
                               +-> Epic 07 (Connector waves) -+
```

---

## Epic 01: Core Service and Storage

**Depends on:** None
**Goal:** Stand up the new `src/beacon/` package beside the untouched `beacon_kb`, with typed config, a typed error taxonomy mapped to RFC 9457 problem details, a Qdrant store layer with shadow-collection staging and atomic alias-flip promotion, a durable SQLite state DB, and a FastAPI skeleton with health, readiness, optional API-key auth, and OpenTelemetry hooks, closed out by a collections REST resource and a Docker deployment surface.

### Features

- Additive packaging: a new dependency group for the server stack that leaves the `beacon_kb` base install and its 1,101 tests fully working.
- Config via pydantic-settings with env-only secrets and local-first defaults.
- Typed error taxonomy (readiness, backend, ingestion, citation, budget) with RFC 9457 problem-details helpers.
- Qdrant client wrapper supporting embedded local mode and server mode from config.
- Shadow-collection staging with atomic alias-flip promotion as the only write path to a live collection.
- Payload schema carrying chunk text, source uri, title, heading path, tags, dates, and hashes.
- SQLite state DB for fingerprints, sources, revisions, and sync jobs, with migrations and restart durability.
- FastAPI app factory, `/healthz`, `/readyz` with per-collection empty/building/ready/failed states, problem-details error handlers, bearer API-key middleware that is off on localhost, and OpenTelemetry hooks.
- `POST /collections` REST resource, docker-compose.yml, Dockerfile, and a smoke test.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 01.1 | Scaffold the beacon package, dependency group, config, and error taxonomy | None |
| 01.2 | Qdrant store layer with shadow collections and alias-flip promotion | 01.1 |
| 01.3 | SQLite state DB with migrations for sources, revisions, fingerprints, and jobs | 01.1 |
| 01.4 | FastAPI skeleton with health, readiness, problem details, auth, and telemetry | 01.1, 01.3 |
| 01.5 | Collections REST resource, compose file, Dockerfile, and smoke test | 01.2, 01.4 |

---

## Epic 02: Ingestion Wave 1

**Depends on:** Epic 01
**Goal:** Ingest documents from folders, uploads, and the web through one Connector interface, parse with Docling into structured sections, chunk hierarchically with parent/child relationships and heading paths, embed with auto-detected models, and synchronize incrementally through the staged shadow-collection write and alias flip, porting the sync safety guarantees and their regression tests from `beacon_kb`.

### Features

- One `Connector` interface shared by all seven source types across all waves.
- Folder connector and upload endpoint with raw-document storage and content-hash dedupe.
- Web and sitemap connector with depth limits and robots.txt respect.
- Docling parsing to structured sections preserving heading paths, case, code, and tables.
- Hierarchical chunking via the LlamaIndex hierarchical node parser with parent/child links and heading paths.
- Embeddings via LiteLLM auto-detect with local fallback and a sparse-only floor requiring zero models.
- Fingerprint-driven incremental sync (parser + chunker config + embedding model + dimension + schema version) with change classes new/changed/deleted/unchanged/incompatible.
- Staged shadow-collection write, validation, atomic alias flip, and durable async job records.
- Ported regression tests: a transient fetch failure never retires an indexed source; an unchanged sync performs zero parse, embed, and write work asserted by call counts; a failure at every stage leaves the prior collection serving.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 02.1 | Connector interface, folder connector, sources resource, and upload endpoint | 01.5 |
| 02.2 | Web and sitemap connector with depth limits and robots respect | 02.1 |
| 02.3 | Docling parsing to structured sections | 02.1 |
| 02.4 | Hierarchical chunking with parent/child links and heading paths | 02.3 |
| 02.5 | Embeddings auto-detect and the incremental sync engine with staged promotion | 02.2, 02.4 |

---

## Epic 03: Retrieval and Grounded Answers

**Depends on:** Epic 02
**Goal:** Deliver the zero-LLM `/search` and one-LLM `/answer` core: Qdrant hybrid retrieval with native RRF and boundary-enforced payload filters, optional cross-encoder rerank, evidence assembly with stable gap-free `[S1]` labels and real provenance, and grounded generation with deterministic abstention, delimiter-neutralized untrusted context, and structural citation validation, exposed over REST with cost-contract and recall tests.

### Features

- Hybrid retrieval via the Qdrant Query API with dense plus sparse prefetch and native RRF fusion.
- Collection, source, tag, and date filters compiled to Qdrant payload filters at the pipeline boundary, not bypassable by any retriever implementation.
- Optional sentence-transformers cross-encoder rerank.
- Parent/context expansion after final ordering under token budgets.
- Stable gap-free `[S1]` evidence labels and match-centered snippets carrying real source uri and title.
- Deterministic pre-abstention with zero LLM calls; exactly one LiteLLM call per answer; post-abstention on sentinel output.
- Structural citation validation against retrieval-derived canonical evidence; fabricated or out-of-response evidence rejected with typed errors.
- REST `POST /search` and `POST /answer` with counting-fake cost-contract tests at the API level, natural-language recall tests, and a default-config end-to-end smoke.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 03.1 | Hybrid retrieval with enforced payload filters and optional rerank | 02.5 |
| 03.2 | Evidence assembly with expansion, budgets, labels, and snippets | 03.1 |
| 03.3 | Grounded answer with abstention, injection defense, and citation validation | 03.2 |
| 03.4 | REST search and answer with cost-contract, recall, and smoke tests | 03.3 |

---

## Epic 04: MCP and CLI

**Depends on:** Epic 03
**Goal:** Expose the same handlers to AI agents via FastMCP (`/mcp` streamable HTTP and stdio) and to humans via a Typer CLI (`beacon serve / sync / search / ask`), with a model auto-detect UX that reports which providers are active and why.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 04.1 | FastMCP server in-process with kb_search, kb_answer, kb_sync_status, kb_list_collections | 03.4 |
| 04.2 | MCP stdio mode and typed MCP error mapping | 04.1 |
| 04.3 | Typer CLI: beacon serve, sync, search, ask | 03.4 |
| 04.4 | Model auto-detect UX, doctor diagnostics, and Ollama compose profile | 04.3 |

---

## Epic 05: Investigate

**Depends on:** Epic 03 (05.3 also depends on 04.1)
**Goal:** Ship the agentic deep-research loop on LangGraph: plan, retrieve, grade, reflect, and synthesize under explicit budgets with SQLite checkpointing, an SSE-streamed step trace over `POST /investigate`, the `kb_investigate` MCP tool, and budget-regression gates.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 05.1 | LangGraph graph skeleton, budget model, and SQLite checkpointing | 03.4 |
| 05.2 | Plan, retrieve, grade, reflect, and synthesize nodes reusing retrieval and answer | 05.1 |
| 05.3 | POST /investigate with SSE trace and the kb_investigate MCP tool | 05.2, 04.1 |
| 05.4 | Investigate budget-regression and cost-contract tests | 05.2 |

---

## Epic 06: Evaluation and Hardening

**Depends on:** Epic 03
**Goal:** Gate CI on RAGAS golden-set metrics (faithfulness, answer relevancy, context precision and recall), wire OpenTelemetry spans around every pipeline stage with per-request token and cost accounting via LiteLLM callbacks, and finish load sanity, docs, and quickstarts.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 06.1 | RAGAS golden-set gates in CI | 03.4 |
| 06.2 | OpenTelemetry tracing, structured logs, and per-request cost accounting | 03.4 |
| 06.3 | Load sanity, docs, and quickstarts | 06.1, 06.2 |

---

## Epic 07: Connector Waves 2 and 3

**Depends on:** Epic 03 (uses the Connector interface and sync engine from Epic 02)
**Goal:** Add the remaining five source types behind the same Connector interface: Confluence and Notion in wave 2, then Google Drive, Slack, and GitHub in wave 3, each with injected clients, env-var credentials, and offline contract tests.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 07.1 | Confluence connector (wave 2) | 02.5 |
| 07.2 | Notion connector (wave 2) | 02.5 |
| 07.3 | Google Drive connector (wave 3) | 02.5 |
| 07.4 | Slack connector (wave 3) | 02.5 |
| 07.5 | GitHub connector (wave 3) | 02.5 |

---

## Epic 08: Cleanup and Release

**Depends on:** All previous epics
**Goal:** Retire the executable spec: delete `src/beacon_kb` and its test suite after a final behavior-parity check, remove the superseded planning docs, finish final documentation, and cut the v1.0 release.

### Tasks

| Task | Title | Depends On |
|------|-------|------------|
| 08.1 | Delete beacon_kb and legacy tests after parity check | Epics 04, 05, 06, 07 complete |
| 08.2 | Final docs, release checklist, and v1.0.0 | 08.1 |
