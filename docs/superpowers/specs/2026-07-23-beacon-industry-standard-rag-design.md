# Beacon: Industry-Standard Knowledge-Base RAG - Design

Date: 2026-07-23.
Status: approved by Rohit (brainstorming session, this date).
Supersedes: the standalone zero-dependency library direction (Tracks D and E of IMPL_PLANS.md are replaced by the epics below).

## 1. Product statement

Beacon is a self-hosted knowledge-base RAG server that anyone can run and integrate.
It is built on industry-standard components rather than bespoke infrastructure.
Day-one consumers are AI agents via MCP and developers via a REST API.
Deployment is local-first: `docker compose up`, or `pipx install beacon-kb && beacon serve` for a zero-infrastructure single process.
The differentiators are operational and safety behaviors ported from the existing codebase: never-break-the-live-corpus sync, structural citation validation, deterministic abstention, prompt-injection defenses, and enforced cost contracts.

## 2. Decisions of record

- Product-first rebuild in this repository.
- New top-level package `src/beacon/`; the existing `beacon_kb` package and its 1,101 tests serve as an executable spec during the port and are deleted in the final cleanup epic.
- Models auto-detect: cloud providers via LiteLLM when API keys are present, otherwise a local Ollama compose profile plus local sentence-transformers embeddings, with a BM25-only floor so search always works with zero models.
- All seven source types are in scope, phased in three connector waves behind one Connector interface.
- The agentic deep-research loop (`/investigate`) ships in v1, built on LangGraph.
- Stack approach A: LlamaIndex-centric ingestion and retrieval, LangGraph only for the investigate loop.

## 3. Stack

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

## 4. Package layout

`src/beacon/`: `config` (pydantic-settings), `models` (pydantic API schemas), `ingest/` (connectors, parsing, chunking, sync engine), `retrieval/`, `answer/` (grounded generation, citation validation, abstention), `investigate/` (LangGraph graph), `server/` (REST routes, MCP tools, optional API-key auth - off by default on localhost), `cli`, `evals/`.

## 5. API surface

REST: `POST /collections`, `POST /collections/{c}/sources`, `POST /collections/{c}/sync` (async job), `GET /jobs/{id}`, `POST /documents` (upload), `POST /search`, `POST /answer`, `POST /investigate` (SSE-streamed trace), `GET /healthz`, `GET /readyz`.
MCP tools over the same handlers: `kb_search`, `kb_answer`, `kb_investigate`, `kb_sync_status`, `kb_list_collections`.
Errors are RFC 9457 problem-details JSON with a typed `kind` (readiness, backend, ingestion, citation, budget), mapped to HTTP codes and MCP tool errors.

## 6. Data flow

Source -> Docling parse -> hierarchical chunks -> embeddings -> staged upsert into a shadow Qdrant collection -> validation (counts, dimensions, fingerprint) -> atomic Qdrant alias flip -> live.
Retrieval: Qdrant hybrid query with native RRF -> optional cross-encoder rerank -> parent/context expansion -> evidence with stable `[S1]` labels and snippets.
Answer: deterministic pre-abstention -> exactly one LLM call over delimited untrusted context -> post-abstention -> structural citation validation against retrieval-derived canonical evidence.
Investigate: LangGraph plan -> retrieve -> grade -> reflect -> synthesize under explicit budgets, checkpointed, with a streamed step trace.
Cost contracts: `search` performs zero LLM calls; `answer` exactly one; `investigate` is budget-bounded.

## 7. Ported behavior guarantees

Each guarantee arrives with its ported regression test from `beacon_kb`:

- Staged sync with alias-flip promotion; any failure leaves the previous collection serving and records a recoverable failed job; a transient connector failure never retires an indexed source.
- Fingerprint-driven incremental sync (parser + chunker config + embedding model + dimension + schema version); an unchanged source performs zero parse, embed, and write work, asserted by call counts.
- Structural citation validation on both the answer and abstention paths; fabricated or out-of-response evidence is rejected with typed errors.
- Deterministic abstention with zero LLM calls on empty or below-policy evidence.
- Untrusted-context delimiters with delimiter-token neutralization; adversarial tests ported.
- Retrieval filters (collection, source, tag, date) enforced at the pipeline boundary via Qdrant payload filters and not bypassable by any retriever implementation.

## 8. Operations

Async sync jobs with durable state that survive restart.
`/readyz` reports per-collection corpus state (empty, building, ready, failed) derived from durable state.
Structured logs and OpenTelemetry spans around every pipeline stage.
Per-request token and cost accounting via LiteLLM callbacks.
Secrets are configured by environment variable and never logged.

## 9. Testing and evaluation

Unit, contract, and integration tests run against embedded Qdrant, not mocks.
Counting-fake tests enforce the cost contracts at unit, integration, and API levels.
Adversarial suites cover citation fabrication and prompt injection.
An end-to-end smoke test runs compose up, syncs fixture documents, and asserts a cited MCP answer.
RAGAS metrics (faithfulness, answer relevancy, context precision and recall) over golden sets gate CI, plus a budget-regression gate for investigate.

## 10. Epics

1. Core service and storage: FastAPI skeleton, config, Qdrant with alias-flip staging, state DB, health.
2. Ingestion wave 1: Docling parsing, hierarchical chunking, folders/uploads/web connectors, incremental sync engine.
3. Retrieval and grounded answers: hybrid search, rerank, evidence and citations, answer with abstention; REST search and answer.
4. MCP and CLI: FastMCP tools, stdio mode, Typer CLI, model auto-detect UX.
5. Investigate: LangGraph loop with budgets, checkpoints, SSE trace.
6. Evaluation and hardening: RAGAS gates, tracing, load sanity, docs, quickstarts.
7. Connector wave 2 (Confluence, Notion), then wave 3 (Google Drive, Slack, GitHub).
8. Cleanup: delete `beacon_kb`, final docs, v1.0 release.

Epics 1 to 3 are sequential; epics 4, 5, and 6 can run in parallel worktrees after epic 3; waves in epic 7 follow independently; epic 8 closes.
Each epic is one branch and one PR; each task is one independent green commit; the per-task and whole-branch review process from the previous epics continues unchanged.
