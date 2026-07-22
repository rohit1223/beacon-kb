# Implementation Epics: Python Generic RAG Knowledge Library
*plans-task-gap-plan-create | 2026-07-22 | python-generic-rag-library*

## Planning Envelope

**Objective:** Build a reusable Python knowledge-base/RAG library with framework-neutral ingestion, indexing, retrieval, and grounded-answer APIs.

**Current state:** No implementation exists in this repository yet.

**Target package:** `python/knowledge-rag/`, distributed from `src/knowledge_rag`, supporting Python 3.11 or newer.

**Included:** Typed knowledge models, source and provider protocols, filesystem/Markdown/HTML/PDF ingestion, incremental indexing, local sparse+dense storage, hybrid retrieval,
optional reranking, context assembly, grounded answer generation, citations, evaluation, CLI, and optional remote-provider/Confluence adapters.

**Explicitly excluded:** Jira, Jira RCA, incident analysis, Slack crawling/triage/synthesis, Slack polling or response delivery, hybrid-search UI, application-framework wiring,
web search, and any other agent workflow.

**Input boundary:** Slack-distilled Markdown may be consumed as ordinary filesystem input, but any Slack knowledge-production pipeline is not part of this library.

**Implementation stance:** Design against known RAG implementation defects such as non-atomic cross-index updates, random chunk identity,
raw-score fusion, unverified citations, forced lowercasing, and provider-specific batch constants.

## Epic Overview

| # | Epic | Track | Repo/Module | Depends On | Total Size | Tasks |
|---|------|-------|-------------|------------|------------|-------|
| 01 | Library Contracts and Storage | A | `python/knowledge-rag` | — | L | 3 |
| 02 | Content Ingestion and Indexing | B | `python/knowledge-rag` | Epic 01 | XL | 4 |
| 03 | Retrieval and Grounded Answers | C | `python/knowledge-rag` | Epic 01; selected tasks in Epic 02 | XL | 4 |
| 04 | Quality, Adapters, and Handoff | D | `python/knowledge-rag`, repository docs | Epic 02, Epic 03 | L | 3 |

*Track A establishes the contract. Tracks B and C can then progress in parallel until context assembly needs canonical chunks and answer generation needs a complete indexed corpus.
Track D is the release gate.*

---

## Epic 01: Library Contracts and Storage

**Track:** A
**Repo/module:** `python/knowledge-rag`
**Depends on:** None
**Summary:** Establish a distributable Python package, typed public contracts, and a consistent local storage baseline that all later ingestion and retrieval work can use.
**Total size:** L | ~17 files | ~1,350 LOC

### Feature 01.1: Reusable Python Foundation

**Size:** L | ~17 files | ~1,350 LOC
**Depends on:** None

#### Task 01.1.1: Scaffold the Python distribution and quality toolchain

**Size:** M | 7 files | ~260 LOC
**Complexity rationale:** The repository has no Python packaging convention yet, so this task establishes an isolated PEP 621 `src`-layout package.
**Depends on:** None
**Description:** Create the package skeleton, dependency extras, test layout, lint/type/build configuration, and Python ignore rules.

#### Task 01.1.2: Define typed domain models, protocols, configuration, and facade

**Size:** L | 8 files | ~500 LOC
**Complexity rationale:** This establishes stable domain types and narrow provider protocols instead of string-keyed metadata and concrete provider dependencies.
**Depends on:** Task 01.1.1
**Description:** Define the public `KnowledgeBase` API and typed contracts for sources, parsing, chunking, embedding, indexes, reranking, generation, progress, citations, and diagnostics.

#### Task 01.1.3: Implement the transactional local knowledge store

**Size:** L | 9 files | ~590 LOC
**Complexity rationale:** Separate sparse, vector, and metadata stores can diverge; the baseline needs one versioned SQLite lifecycle with FTS5, stored embeddings,
schema migration, and atomic active-revision promotion.
**Depends on:** Task 01.1.2
**Description:** Implement corpus namespaces, source revisions, chunks, FTS5 BM25, embedding rows, build runs, index fingerprints, active revision pointers, and store contract tests.

---

## Epic 02: Content Ingestion and Indexing

**Track:** B
**Repo/module:** `python/knowledge-rag`
**Depends on:** Epic 01
**Summary:** Build a generic, deterministic ingestion pipeline that preserves source structure and incrementally promotes consistent index revisions.
**Total size:** XL | ~27 files | ~2,250 LOC

### Feature 02.1: Source Loading and Parsing

**Size:** L | ~16 files | ~1,120 LOC
**Depends on:** Task 01.1.2

#### Task 02.1.1: Implement source discovery and canonical identity

**Size:** M | 6 files | ~360 LOC
**Complexity rationale:** Readers that mix filesystem paths, link rewriting, authentication, and document creation are not reusable; a generic connector boundary must normalize identity and provenance
without owning credentials.
**Depends on:** Task 01.1.2
**Description:** Implement filesystem/glob and in-memory source connectors, MIME detection, canonical source URIs, external-link mapping, metadata filters, and connector contract tests.

#### Task 02.1.2: Implement structure-aware Markdown, HTML, and PDF parsers

**Size:** L | 12 files | ~760 LOC
**Complexity rationale:** Structure-aware parsing must extract heading, URL, table, and code metadata without introducing lowercasing or section-loss defects.
**Depends on:** Task 02.1.1
**Description:** Produce typed document sections with heading paths, anchors, page/offset locators, code, tables, links, source metadata, warnings, and optional parser dependencies.

### Feature 02.2: Deterministic Incremental Indexing

**Size:** L | ~15 files | ~1,130 LOC
**Depends on:** Feature 02.1, Task 01.1.3

#### Task 02.2.1: Add parent-child chunking, optional enrichment, and batched embeddings

**Size:** L | 8 files | ~560 LOC
**Complexity rationale:** Chunking needs model-aware token limits, real configurable overlap, stable IDs, caching,
and provider-owned batch constraints; LLM enrichment must stay optional.
**Depends on:** Task 02.1.2, Task 01.1.3
**Description:** Create heading-aware parent and child chunks, real configurable overlap, deterministic identity, neighbor links, optional cached enrichment, embedding batching, and progress events.

#### Task 02.2.2: Implement staged full and incremental synchronization

**Size:** L | 10 files | ~570 LOC
**Complexity rationale:** Add/modify/delete behavior must update sparse, dense, manifest, and active source revisions as one recoverable operation while invalidating on any pipeline fingerprint
change.
**Depends on:** Task 02.2.1
**Description:** Plan and execute unchanged/new/modified/deleted source revisions, stage writes, validate counts and dimensions, atomically promote, roll back failures, and expose typed sync reports.

---

## Epic 03: Retrieval and Grounded Answers

**Track:** C
**Repo/module:** `python/knowledge-rag`
**Depends on:** Epic 01; Task 03.1.3 depends on Task 02.2.1; Task 03.1.4 depends on Task 02.2.2
**Summary:** Provide calibrated hybrid retrieval, bounded evidence assembly, and answer generation whose citations and abstention behavior are programmatically verifiable.
**Total size:** XL | ~25 files | ~1,850 LOC

### Feature 03.1: Search, Evidence, and Answers

**Size:** XL | ~25 files | ~1,850 LOC
**Depends on:** Task 01.1.3

#### Task 03.1.1: Implement sparse and dense candidate retrieval

**Size:** L | 8 files | ~520 LOC
**Complexity rationale:** Retrievers must define score semantics and query policies explicitly instead of assuming provider-specific score metadata or reusing one rewritten query for both indexes.
**Depends on:** Task 01.1.3
**Description:** Retrieve independent BM25 and vector candidate lists using the appropriate original or rewritten query, namespace/ACL/source filters, exact technical-token boosts,
and structured diagnostics.

#### Task 03.1.2: Add rank fusion, optional reranking, and diversity controls

**Size:** M | 6 files | ~380 LOC
**Complexity rationale:** Fixed weighted averaging mixes incomparable BM25 and cosine scales; rank-based fusion and adapter-based reranking avoid that defect while keeping expensive models optional.
**Depends on:** Task 03.1.1
**Description:** Implement Reciprocal Rank Fusion, deterministic deduplication, optional cross-encoder reranking, near-duplicate collapse, and optional MMR-style diversity.

#### Task 03.1.3: Assemble bounded parent and neighbor context

**Size:** M | 5 files | ~300 LOC
**Complexity rationale:** Unbounded neighbor expansion can triple results without recapping them; assembly must preserve structure, generate match-centered snippets, and obey an explicit token budget.
**Depends on:** Task 03.1.2, Task 02.2.1
**Description:** Expand selected child hits to parents or neighbors after reranking, deduplicate spans, allocate the evidence budget, and emit stable evidence IDs with provenance.

#### Task 03.1.4: Generate grounded answers with validated citations and abstention

**Size:** L | 8 files | ~650 LOC
**Complexity rationale:** Free-form model citations cannot be verified; the response contract must retain and validate every cited evidence item.
**Depends on:** Task 03.1.3, Task 02.2.2
**Description:** Add optional query rewriting and answer generation, untrusted-context boundaries, structured evidence citations, citation validation, no-evidence abstention, and complete
retrieval/generation diagnostics.

---

## Epic 04: Quality, Adapters, and Handoff

**Track:** D
**Repo/module:** `python/knowledge-rag`, `README.md`, `.gitignore`
**Depends on:** Epic 02, Epic 03
**Summary:** Prove retrieval and lifecycle quality, provide thin optional provider and source adapters, and document a safe release.
**Total size:** L | ~22 files | ~1,350 LOC

### Feature 04.1: Release Readiness

**Size:** L | ~22 files | ~1,350 LOC
**Depends on:** Task 02.2.2, Task 03.1.4

#### Task 04.1.1: Build the offline evaluation and resilience suite

**Size:** L | 9 files | ~600 LOC
**Complexity rationale:** Rollout needs a versioned corpus and measurable gates covering end-to-end ingestion, restart consistency, retrieval quality, citation validity, and crash recovery.
**Depends on:** Task 02.2.2, Task 03.1.4
**Description:** Add golden fixtures and metrics for Recall@K, MRR/nDCG, citation validity, abstention, latency, idempotency, restart, add/modify/delete, fingerprint migration, concurrency,
and rollback.

#### Task 04.1.2: Add thin CLI and optional provider/source adapters

**Size:** M | 8 files | ~400 LOC
**Complexity rationale:** The core must remain framework-neutral while still proving that remote-provider embeddings/generation and Confluence pages can satisfy the generic contracts without owning
application auth or orchestration.
**Depends on:** Task 02.2.2, Task 03.1.4
**Description:** Add `index`, `search`, `answer`, `inspect`, and `evaluate` commands plus optional remote-provider and Confluence adapters with injected clients, contract tests, and integration markers.

#### Task 04.1.3: Document architecture, operations, and package release

**Size:** M | 5 files | ~350 LOC
**Complexity rationale:** This repository has no Python release path yet, so adoption needs explicit documentation,
quality gates, and rollback guidance.
**Depends on:** Task 04.1.1, Task 04.1.2
**Description:** Complete package and repository documentation, build wheel/sdist, run the benchmark corpus against the quality gates, and record accepted trade-offs.

---

## Recommended Enhancements Included in the Plan

- Use rank-based fusion instead of fixed raw-score weighting because BM25 and dense scores are not calibrated to the same scale.
- Use deterministic content-addressed source, section, and chunk IDs so reindexing, citation links, and deletions are stable.
- Include parser, chunker, enrichment, embedding model, embedding dimension, and schema versions in the index fingerprint.
- Stage revisions and atomically promote them so partial sparse, vector, or metadata writes are never query-visible.
- Keep LLM enrichment, query rewriting, reranking, and answer generation optional; retrieval remains usable without them.
- Preserve original case, commands, code, tables, headings, links, page numbers, anchors, and structural offsets.
- Retrieve child chunks, then expand only selected parents/neighbors inside a token budget.
- Return typed evidence and diagnostics; render Markdown only at the CLI or consuming-application boundary.
- Make web search a separate opt-in retriever outside this library, never a hidden flag on answer generation.
- Calibrate abstention and rollout thresholds from a versioned evaluation corpus rather than hard-coded relevance values.

## Deferred Beyond This Plan

- Jira, Jira Service Desk, incident RCA, ticket context, or operational diagnostic logic.
- A REST service, browser UI, Slack bot, scheduled crawler, or web-search implementation.
- Production-scale vector backends beyond the protocol and optional adapter seam; add one only after corpus-size and latency benchmarks justify it.
