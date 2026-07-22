# Implementation Plans: Python Generic RAG Knowledge Library
*plans-task-gap-plan-create | 2026-07-22 | python-generic-rag-library*

> Plans cover: Epic 01 Library Contracts and Storage, Epic 02 Content Ingestion and Indexing, Epic 03 Retrieval and Grounded Answers, and Epic 04 Quality, Adapters, and Handoff.
> Source: `EPICS.md`

## Scope Guardrails

- Build only the reusable knowledge/RAG package under `python/knowledge-rag` plus its package and repository documentation.
- Do not build Jira integration, incident workflows, Slack crawling/triage/synthesis, Slack polling/delivery, application UI, web routing, or web search.
- Treat generated Slack Markdown and existing runbooks as ordinary documents supplied through source adapters.

## Target Architecture

```text
SourceConnector -> RawDocument -> Parser -> DocumentSection -> Chunker -> Chunk
       -> optional Enricher -> Embedder -> transactional IndexCoordinator
       -> SparseRetriever + DenseRetriever -> RRF -> optional Reranker
       -> ContextAssembler -> optional AnswerGenerator -> QueryResponse + Evidence
```

The package exposes typed objects at every boundary. The default local implementation uses SQLite for manifests, active source revisions, chunks, FTS5 BM25, and embedding rows so one
transaction controls visibility. External vector or model systems remain adapters behind protocols.

---

## Task 01.1.1: Scaffold the Python distribution and quality toolchain
**Epic:** 01 — Library Contracts and Storage | **Feature:** 01.1 — Reusable Python Foundation
**Size:** M | 7 files | ~260 LOC | **Track:** A

### Current State

- No `pyproject.toml`, Python package, or Python test configuration exists.
- `.gitignore` does not yet cover Python virtual environments, caches, coverage files, wheels, or source distributions.

### Desired State

- `python/knowledge-rag` is an isolated PEP 621 distribution using a `src` layout and `requires-python = ">=3.11"`.
- The base install remains small; HTML, PDF, Confluence, remote-provider, local-model, and development dependencies are declared as optional extras.
- Ruff, strict mypy, pytest, coverage, wheel, and sdist checks have one documented command sequence.

### Gap Analysis

- Missing: Python build metadata, package layout, test markers, quality configuration, and ignore patterns.
- Changes: Add Python-only ignore rules at the root.
- Blockers: None.

### Implementation Research

- Packaging convention: the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) documents the `[build-system]`, `[project]`, and `[tool]`
  sections used here.
- Repository convention: long-running stages must log start, end, elapsed time, and `current/total` progress.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/pyproject.toml` | Declare package metadata, extras, entry point, pytest, Ruff, mypy, and coverage settings. |
| CREATE | `python/knowledge-rag/README.md` | Introduce the package, scope, install modes, API examples, and development commands. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/__init__.py` | Establish the import package and curated public exports. |
| CREATE | `python/knowledge-rag/tests/conftest.py` | Provide shared deterministic fixtures and test markers. |
| CREATE | `python/knowledge-rag/tests/unit/__init__.py` | Establish unit-test package layout. |
| CREATE | `python/knowledge-rag/tests/contract/__init__.py` | Establish adapter contract-test layout. |
| MODIFY | `.gitignore` | Ignore Python environments, caches, coverage, and build artifacts. |

### Acceptance Criteria

- [ ] `python -m build` creates both wheel and sdist from a clean checkout.
- [ ] Installing the base wheel does not install Confluence, HTML, PDF, LangChain, remote-provider, or local-model dependencies.
- [ ] `python -m ruff check .`, `python -m mypy src`, and an empty pytest discovery run succeed.
- [ ] Python artifact patterns are present in `.gitignore` without removing existing entries.

### Validation Steps

```bash
cd python/knowledge-rag
python -m build
python -m ruff check .
python -m mypy src
python -m pytest --collect-only
```

---

## Task 01.1.2: Define typed domain models, protocols, configuration, and facade
**Epic:** 01 — Library Contracts and Storage | **Feature:** 01.1 — Reusable Python Foundation
**Size:** L | 8 files | ~500 LOC | **Track:** A

### Current State

- The package has no domain models, protocols, configuration, or facade yet.
- Common RAG design defects to avoid: source metadata kept as repeated string constants, answer shapes that lack evidence identity and retrieval-stage diagnostics, and a single mutable result object that mixes content, metadata, and every score.

### Desired State

- Frozen typed models represent corpus, source, revision, raw document, section, chunk, index fingerprint, search query, search hit, evidence, citation, sync report, and answer response.
- Runtime-checkable `Protocol` contracts isolate connectors, parsers, chunkers, embedders, stores, retrievers, fusion, rerankers, token counters, generators, and progress observers.
- The `KnowledgeBase` facade exposes `sync`, `search`, `answer`, `inspect`, and health/status methods without importing an application framework.

### Gap Analysis

- Missing: Stable public Python contracts and explicit score, identity, provenance, filter, status, and error semantics.
- Changes: Use typed core fields plus a constrained extension metadata map instead of loose metadata maps.
- Blockers: Task 01.1.1 must establish packaging and typing configuration.

### Implementation Research

- Strong typing is required; formatting belongs at the presentation boundary rather than in return values.
- Keep a clear answer/evidence split while adding stable IDs and per-stage scores.
- Keep provider coupling out of the core: no provider imports or credential state in models, protocols, or the facade.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/models.py` | Define immutable domain records and enums. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/protocols.py` | Define narrow provider and pipeline protocols. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/config.py` | Define typed library configuration and validation. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/errors.py` | Define typed configuration, readiness, backend, ingestion, and citation errors. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/knowledge_base.py` | Implement the public orchestration facade without concrete providers. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/__init__.py` | Export only the supported public API. |
| CREATE | `python/knowledge-rag/tests/unit/test_models.py` | Verify validation, immutability, serialization, and deterministic equality. |
| CREATE | `python/knowledge-rag/tests/contract/test_protocol_contracts.py` | Verify deterministic fakes satisfy every public protocol. |

### Acceptance Criteria

- [ ] Public methods return structured domain objects and never preformatted Markdown.
- [ ] Sparse, dense, fusion, and rerank scores have separate optional fields with documented direction and range semantics.
- [ ] Every source, section, chunk, evidence item, build run, and corpus has a stable typed identifier.
- [ ] Namespace, ACL, source, tag, and date filters are represented without provider-specific syntax.
- [ ] Importing `knowledge_rag` performs no network, filesystem, logging-handler, or credential side effect.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/test_models.py tests/contract/test_protocol_contracts.py
python -m mypy src
```

---

## Task 01.1.3: Implement the transactional local knowledge store
**Epic:** 01 — Library Contracts and Storage | **Feature:** 01.1 — Reusable Python Foundation
**Size:** L | 9 files | ~590 LOC | **Track:** A

### Current State

- No storage backend exists yet.
- Design pitfalls to avoid: committing sparse documents independently from vector and metadata persistence, mutating an in-memory vector store that rewrites a separate JSON file, truncating and rewriting a standalone JSON manifest, and swallowing index-write failures so stores drift apart.

### Desired State

- The local backend stores corpora, source revisions, sections, chunks, FTS rows, embedding vectors, fingerprints, build runs, and active revision pointers in one SQLite database.
- A staged revision is invisible until validation completes and one transaction promotes its active pointers.
- SQLite FTS5 supplies weighted BM25 candidates; vectors have declared dimension and similarity semantics and are searched locally with NumPy for the reference corpus.

### Gap Analysis

- Missing: Atomic visibility, schema migrations, index integrity checks, active revision state, and backend contract tests.
- Changes: Use canonical source URIs and revision records rather than filesystem-path reconciliation or ad hoc JSON persistence.
- Blockers: Task 01.1.2 must define storage, sparse-index, and vector-index contracts.

### Implementation Research

- SQLite's official [FTS5 documentation](https://www.sqlite.org/fts5.html) supports full-text virtual tables, column weighting, and the built-in `bm25()` rank function.
- Isolation must come from an explicit corpus namespace, never from filesystem path conventions.
- Store embeddings with a model and dimension fingerprint; never infer whether a provider returned similarity or distance from an untyped metadata key.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/storage/__init__.py` | Export storage implementations and migration helpers. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/storage/sqlite.py` | Implement connections, transactions, active revision queries, FTS, and embedding persistence. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/storage/migrations/0001_initial.sql` | Define the versioned local schema and indexes. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/storage/vector_math.py` | Implement normalized vector validation and local similarity search. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/indexing/manifest.py` | Build and validate index fingerprints and revision metadata. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/testing.py` | Provide deterministic embedder, generator, clock, and failure-injection fakes. |
| CREATE | `python/knowledge-rag/tests/contract/test_knowledge_store_contract.py` | Verify generic store behavior. |
| CREATE | `python/knowledge-rag/tests/unit/storage/test_sqlite_store.py` | Verify schema, transactions, FTS, vectors, namespaces, and restart. |
| MODIFY | `python/knowledge-rag/pyproject.toml` | Add NumPy to the local backend and include SQL migration resources. |

### Acceptance Criteria

- [ ] Sparse rows, vector rows, manifest rows, and active revision pointers become visible atomically.
- [ ] Rollback leaves the previously active corpus fully searchable and removes or marks incomplete staging rows.
- [ ] FTS5 capability and vector dimension are checked at startup with typed failures.
- [ ] Two corpus namespaces with identical source paths never read or mutate each other's records.
- [ ] Closing and reopening the store preserves active data, build status, and query results.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/contract/test_knowledge_store_contract.py tests/unit/storage/test_sqlite_store.py
```

---

## Task 02.1.1: Implement source discovery and canonical identity
**Epic:** 02 — Content Ingestion and Indexing | **Feature:** 02.1 — Source Loading and Parsing
**Size:** M | 6 files | ~360 LOC | **Track:** B

### Current State

- No source connectors exist yet.
- Design pitfalls to avoid: readers that convert paths straight into framework documents, discovery hardcoded to a fixed extension list, and one connector class that owns authentication lookup, remote fetching, child traversal, and document conversion.

### Desired State

- `SourceConnector` implementations discover typed descriptors and load bytes/text without parsing or indexing them.
- Built-in filesystem and in-memory connectors normalize canonical URIs, media types, source hashes, timestamps, external links, and caller-supplied ACL/tags.
- Credentials and client construction remain caller-owned; remote connectors receive injected clients.

### Gap Analysis

- Missing: Connector contracts, canonical identity policy, reusable in-memory fixtures, and connector conformance tests.
- Changes: Separate link mapping and MIME discovery from parsing; avoid absolute-versus-relative path matching heuristics.
- Blockers: Task 01.1.2.

### Implementation Research

- Source descriptors need display names and external citation links as first-class concepts.
- Filesystem sources must support glob patterns and configurable external-link mapping.
- Treat channel-scoped Slack Markdown exports as normal filesystem input.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/identity.py` | Canonicalize source URIs and generate stable source/revision IDs. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/adapters/filesystem.py` | Discover and load configured files, directories, and globs. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/adapters/memory.py` | Supply deterministic documents for tests and embedding applications. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/media.py` | Resolve media types, extensions, and parser selection hints. |
| CREATE | `python/knowledge-rag/tests/contract/test_source_connector_contract.py` | Verify connector identity, pagination/iteration, errors, and metadata rules. |
| CREATE | `python/knowledge-rag/tests/unit/ingestion/test_filesystem_source.py` | Verify files, directories, globs, links, hashes, and path edge cases. |

### Acceptance Criteria

- [ ] Repeated scans produce identical canonical source IDs regardless of current working directory.
- [ ] Source content changes alter the revision hash but not the logical source ID.
- [ ] Filesystem discovery supports Markdown, HTML/HTM, and PDF patterns and deterministic ordering.
- [ ] External citation links are derived without leaking local absolute paths when a mapping is configured.
- [ ] Connector errors identify the source and operation without including credentials.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/contract/test_source_connector_contract.py tests/unit/ingestion/test_filesystem_source.py
```

---

## Task 02.1.2: Implement structure-aware Markdown, HTML, and PDF parsers
**Epic:** 02 — Content Ingestion and Indexing | **Feature:** 02.1 — Source Loading and Parsing
**Size:** L | 12 files | ~760 LOC | **Track:** B

### Current State

- No parsers exist yet.
- Design pitfalls to avoid: heading splitters that lowercase output and drop content not attached to a subheading, HTML extraction entangled with site-specific cleanup, and PDF heuristics that silently misclassify headings, headers, and footers.

### Desired State

- Parsers emit `DocumentSection` records with original text, heading path, anchor or page, character offsets, code blocks, tables, links, media type, and parse warnings.
- Markdown and HTML preserve case and fenced code; PDF support is an optional extra and records page-level provenance when available.
- Parser registration is extensible and deterministic, with generic behavior separated from site-specific HTML cleanup hooks.

### Gap Analysis

- Missing: Typed parser output, consistent structural locators, parse-warning contracts, and cross-format fixtures.
- Changes: Extract rich structural metadata without forced lowercasing or silent original-document fallback.
- Blockers: Task 02.1.1.

### Implementation Research

- HTML fixtures must cover headings, URLs, code, and table metadata.
- Heading-path and URL-normalization behavior needs dedicated parser-level tests.
- LLM table summarization belongs to optional enrichment, not deterministic parsing.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/parsing/base.py` | Register parsers and common section/provenance helpers. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/parsing/markdown.py` | Parse headings, anchors, prose, tables, links, and fenced code. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/parsing/html.py` | Parse semantic sections, links, tables, code, and optional cleanup hooks. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/parsing/pdf.py` | Parse page-aware PDF text and warnings behind the PDF extra. |
| CREATE | `python/knowledge-rag/tests/unit/parsing/test_markdown_parser.py` | Verify structural preservation and edge cases. |
| CREATE | `python/knowledge-rag/tests/unit/parsing/test_html_parser.py` | Verify heading, URL, table, code, tab, and accordion fixtures. |
| CREATE | `python/knowledge-rag/tests/unit/parsing/test_pdf_parser.py` | Verify page provenance, repeated header removal, and malformed input. |
| CREATE | `python/knowledge-rag/tests/fixtures/documents/sample.md` | Supply safe Markdown regression content. |
| CREATE | `python/knowledge-rag/tests/fixtures/documents/sample.html` | Supply safe HTML regression content. |
| CREATE | `python/knowledge-rag/tests/fixtures/documents/sample.pdf` | Supply a small generated PDF fixture. |
| MODIFY | `python/knowledge-rag/pyproject.toml` | Add parser extras and test dependencies. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/__init__.py` | Export parser registration types, not concrete optional dependencies. |

### Acceptance Criteria

- [ ] Parsers preserve original case, commands, code blocks, links, tables, and source text needed for citation.
- [ ] Every emitted section has a source URI and at least one stable structural locator: heading/anchor, page, or character span.
- [ ] Unsupported or malformed content returns typed warnings/errors; it is never silently indexed as empty text.
- [ ] Base-package import succeeds when HTML and PDF extras are absent.
- [ ] Parser fixtures are newly authored safe content and contain no proprietary documentation or secrets.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/parsing
```

---

## Task 02.2.1: Add parent-child chunking, optional enrichment, and batched embeddings
**Epic:** 02 — Content Ingestion and Indexing | **Feature:** 02.2 — Deterministic Incremental Indexing
**Size:** L | 8 files | ~560 LOC | **Track:** B

### Current State

- No chunking, enrichment, or embedding pipeline exists yet.
- Design pitfalls to avoid: splitters whose documented overlap parameter is actually a minimum chunk length, chunk identity from randomly generated IDs, LLM enrichment as a mandatory step for every chunk, and provider batch limits hardcoded in core logic.

### Desired State

- Heading-aware parent sections produce child chunks using a configurable provider-aware token counter, 350–600 target tokens by default, and a real 50–100 token overlap.
- IDs derive from corpus, canonical source, revision, parser/chunker fingerprint, parent locator, and child ordinal.
- Enrichment is optional, cached by content plus prompt/model version, failure-policy controlled, and never required for baseline retrieval.
- Embedding batch size and retry policy come from the injected provider; progress emits structured start/end/elapsed and `current/total` events.

### Gap Analysis

- Missing: Real overlap, parent-child identity, provider-neutral batching, enrichment cache, and deterministic failure tests.
- Changes: Keep summaries/keywords/FAQs as optional searchable metadata instead of embedding prerequisites.
- Blockers: Task 02.1.2 and Task 01.1.3.

### Implementation Research

- Generate neighbor links only after stable ordering and IDs exist.
- Use parent/child indexing: sparse may search both levels; dense defaults to children.
- Pipeline progress must follow the repository logging rules and expose structured progress callbacks.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/chunking.py` | Implement parent-child, overlap, token limits, stable IDs, and neighbor links. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/enrichment.py` | Orchestrate optional cached enrichment and failure policy. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/indexing/embedding.py` | Batch, validate, retry, and cache embeddings through the provider protocol. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/progress.py` | Define structured stages and progress events plus logging adapter. |
| CREATE | `python/knowledge-rag/tests/unit/ingestion/test_chunking.py` | Verify boundaries, real overlap, IDs, parent links, and code preservation. |
| CREATE | `python/knowledge-rag/tests/unit/ingestion/test_enrichment.py` | Verify disabled, cached, failed, and version-changed enrichment. |
| CREATE | `python/knowledge-rag/tests/contract/test_embedding_provider_contract.py` | Verify dimensions, score metadata, batch limits, retries, and partial failures. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/config.py` | Add chunking, enrichment, embedding, retry, and progress settings. |

### Acceptance Criteria

- [ ] Tests prove consecutive child chunks share the configured token overlap and never split inside a fenced code block when avoidable.
- [ ] Identical input and fingerprint produce identical parent, child, previous, and next IDs across processes.
- [ ] Changing parser, chunker, enrichment, or embedding configuration changes the relevant fingerprint.
- [ ] Ingestion succeeds with enrichment disabled and with an enrichment provider failure under the configured best-effort policy.
- [ ] Batch sizes never exceed the provider contract and every long stage emits start/end, elapsed time, and `current/total` progress.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/ingestion/test_chunking.py tests/unit/ingestion/test_enrichment.py tests/contract/test_embedding_provider_contract.py
```

---

## Task 02.2.2: Implement staged full and incremental synchronization
**Epic:** 02 — Content Ingestion and Indexing | **Feature:** 02.2 — Deterministic Incremental Indexing
**Size:** L | 10 files | ~570 LOC | **Track:** B

### Current State

- No synchronization lifecycle exists yet.
- Design pitfalls to avoid: change detection from raw content hashes alone, deleting and re-adding sparse, dense, and metadata entries independently while swallowing errors, writing an index version that change analysis never compares, and full-reindex tasks that clear shared state per source.

### Desired State

- A deterministic plan classifies source revisions as unchanged, new, changed, deleted, or pipeline-incompatible.
- Sync stages all affected sections/chunks/embeddings, validates them, and atomically promotes active revisions; failure keeps the previous revision active.
- Full rebuild creates a new corpus generation once, rather than clearing shared state per source.
- The returned `SyncReport` includes counts, timings, warnings, fingerprints, failed sources, and active build identity.

### Gap Analysis

- Missing: Fingerprint invalidation, staged promotion, crash recovery, idempotency, source-level failure policy, and reliable readiness state.
- Changes: Replace callback counters and mutable status with persisted build-run state and explicit `EMPTY`, `BUILDING`, `READY`, and `FAILED` health semantics.
- Blockers: Task 02.2.1.

### Implementation Research

- Report new, modified, deleted, and unchanged sources in every sync result.
- Make promotion itself the visibility boundary so partially persisted revisions are never readable.
- A restart with unchanged sources must never produce an empty index.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/planning.py` | Compute source and pipeline change sets. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/ingestion/sync.py` | Orchestrate scan, load, parse, chunk, enrich, embed, stage, validate, and promote. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/indexing/coordinator.py` | Coordinate sparse/vector writes through one revision transaction. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/indexing/validation.py` | Validate counts, IDs, dimensions, links, and fingerprint consistency. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/knowledge_base.py` | Wire full and incremental sync plus health inspection. |
| CREATE | `python/knowledge-rag/tests/integration/test_sync_lifecycle.py` | Verify empty, full, unchanged, add, modify, delete, and restart behavior. |
| CREATE | `python/knowledge-rag/tests/integration/test_sync_rollback.py` | Inject parser, embedding, sparse, vector, and promotion failures. |
| CREATE | `python/knowledge-rag/tests/integration/test_fingerprint_migration.py` | Verify schema/model/parser/chunker changes trigger safe replacement. |
| CREATE | `python/knowledge-rag/tests/integration/test_multi_source_concurrency.py` | Verify concurrent sources cannot clear or corrupt one another. |
| MODIFY | `python/knowledge-rag/tests/conftest.py` | Add temporary corpus/store and failure-injection fixtures. |

### Acceptance Criteria

- [ ] An unchanged second sync performs zero parsing, enrichment, embedding, and index writes.
- [ ] New, changed, and deleted sources update both sparse and dense views after one promotion.
- [ ] Any parser/chunker/enrichment/embedding/schema fingerprint change triggers the documented affected reindex scope.
- [ ] Simulated failure at every stage leaves the previous active corpus searchable and a recoverable failed build record.
- [ ] Restart reconstructs readiness and active revisions from durable state without relying on an in-memory counter.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/integration/test_sync_lifecycle.py tests/integration/test_sync_rollback.py tests/integration/test_fingerprint_migration.py tests/integration/test_multi_source_concurrency.py
```

---

## Task 03.1.1: Implement sparse and dense candidate retrieval
**Epic:** 03 — Retrieval and Grounded Answers | **Feature:** 03.1 — Search, Evidence, and Answers
**Size:** L | 8 files | ~520 LOC | **Track:** C

### Current State

- No retrievers exist yet.
- Design pitfalls to avoid: searching every field with equal weight, querying a vector store without a typed score contract, defaulting missing distance metadata to zero, and reusing one rewritten query for both sparse and dense retrieval.

### Desired State

- Sparse and dense retrievers return independent ranked candidate lists with explicit raw score, rank, score direction, provider, query variant, and timing.
- Sparse retrieval favors exact error codes, command names, identifiers, headings, and code fields while dense retrieval may use an optional rewrite or expansion.
- Namespace, ACL, source, tag, media type, and date filters apply consistently before results leave either retriever.

### Gap Analysis

- Missing: Typed score semantics, query-policy separation, weighted sparse fields, consistent filters, and stage diagnostics.
- Changes: Use the user's current question for lexical precision; let dense query rewriting be optional and separately observable.
- Blockers: Task 01.1.3. It can proceed in parallel with most of Epic 02 using deterministic stored fixtures.

### Implementation Research

- SQLite FTS5 supports weighted columns through `bm25()` arguments in the official [FTS5 ranking documentation](https://www.sqlite.org/fts5.html#the_bm25_function).
- Contract tests must use the store's declared score semantics; injecting synthetic score metadata masks contract mismatches.
- Keep query rewriting as the optional provider protocol already defined in Task 01.1.2.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/query.py` | Validate queries, select sparse/dense variants, and preserve the original question. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/sparse.py` | Execute weighted FTS5 BM25 and exact-token boosts. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/dense.py` | Embed queries and retrieve declared-similarity candidates. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/filters.py` | Apply provider-neutral filter semantics. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/hybrid.py` | Run candidate stages, collect diagnostics, and pass ranked lists to fusion. |
| CREATE | `python/knowledge-rag/tests/unit/retrieval/test_sparse.py` | Verify weighting, escaping, exact boosts, filters, and score direction. |
| CREATE | `python/knowledge-rag/tests/unit/retrieval/test_dense.py` | Verify query variants, dimensions, similarity, filters, and empty indexes. |
| CREATE | `python/knowledge-rag/tests/integration/test_candidate_retrieval.py` | Verify both retrievers against the same active corpus. |

### Acceptance Criteria

- [ ] Sparse and dense results retain independent ranks and raw scores without cross-normalization.
- [ ] Error codes and exact technical identifiers can be boosted without altering dense retrieval.
- [ ] Original and rewritten queries are recorded in diagnostics and tested independently.
- [ ] Corpus and ACL filters cannot be bypassed by either backend.
- [ ] Missing, empty, or dimension-incompatible indexes return typed readiness/backend errors.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/retrieval/test_sparse.py tests/unit/retrieval/test_dense.py tests/integration/test_candidate_retrieval.py
```

---

## Task 03.1.2: Add rank fusion, optional reranking, and diversity controls
**Epic:** 03 — Retrieval and Grounded Answers | **Feature:** 03.1 — Search, Evidence, and Answers
**Size:** M | 6 files | ~380 LOC | **Track:** C

### Current State

- No fusion or reranking exists yet.
- Design pitfalls to avoid: hand-written score normalization with rank boosts, fixed-weight combination of incomparable sparse and dense scores, and deduplication by exact document ID only.

### Desired State

- Reciprocal Rank Fusion combines sparse and dense ranks without pretending their raw scores are calibrated.
- A configurable optional reranker scores only the fused candidate window and reports its latency and score separately.
- Deterministic near-duplicate collapse and optional MMR-style diversity reduce repeated chunks without hiding provenance.

### Gap Analysis

- Missing: Rank-based fusion, candidate caps, reranker protocol use, content-near-duplicate handling, and diversity tests.
- Changes: Retain all component scores for diagnosis; use fusion/rerank score only for final ordering.
- Blockers: Task 03.1.1.

### Implementation Research

- The original [Reciprocal Rank Fusion paper](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) establishes rank fusion for
  combining retrieval systems.
- Sentence Transformers' official [retrieve-and-rerank guidance](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) describes retrieving a larger
  candidate set before applying a cross-encoder to the smaller set.
- Reranking remains optional because it adds model cost and latency and may be unavailable in some deployments.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/fusion.py` | Implement configurable RRF with deterministic tie-breaking. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/rerank.py` | Invoke optional rerankers on a bounded candidate window. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/diversity.py` | Collapse near duplicates and optionally select diverse evidence. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/retrieval/hybrid.py` | Apply fusion, rerank, diversity, and result caps. |
| CREATE | `python/knowledge-rag/tests/unit/retrieval/test_fusion.py` | Verify RRF, ties, missing lists, and stable ordering. |
| CREATE | `python/knowledge-rag/tests/unit/retrieval/test_rerank_diversity.py` | Verify optional rerank, caps, duplicate collapse, and failure policy. |

### Acceptance Criteria

- [ ] Fused order depends on ranks, not provider-specific raw score scales.
- [ ] RRF parameters, candidate counts, and every component rank/score appear in diagnostics.
- [ ] Reranker absence or configured best-effort failure returns the fused order unchanged.
- [ ] Duplicate collapse never merges chunks from different sources merely because their text is similar.
- [ ] Final retrieval ordering is deterministic for identical inputs and configuration.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/retrieval/test_fusion.py tests/unit/retrieval/test_rerank_diversity.py
```

---

## Task 03.1.3: Assemble bounded parent and neighbor context
**Epic:** 03 — Retrieval and Grounded Answers | **Feature:** 03.1 — Search, Evidence, and Answers
**Size:** M | 5 files | ~300 LOC | **Track:** C

### Current State

- No context assembly exists yet.
- Design pitfalls to avoid: adding previous and next chunks for every result unconditionally, assigning invented relevance scores to context chunks, skipping any result-count or token recap before prompt construction, and fallbacks that always select a document's first N characters.

### Desired State

- Context expansion happens after reranking and selects parents or neighbors only when they add bounded structural continuity.
- A token budget reserves prompt overhead and allocates evidence by rank, source diversity, and non-overlapping span coverage.
- Each evidence item has stable `[S1]`-style identity, source URI, title, structural locator, text span, and component diagnostics.

### Gap Analysis

- Missing: Match-centered snippets, token accounting, span deduplication, evidence IDs, and bounded structural expansion.
- Changes: Context neighbors do not receive invented relevance scores; they retain `context_of` relationships.
- Blockers: Task 03.1.2 and Task 02.2.1.

### Implementation Research

- Parent and neighbor links come from the chunking stage in Task 02.2.1.
- Preserve source and structural metadata extracted by Task 02.1.2 rather than flattening it into prompt strings.
- The token counter is a protocol because answer models may tokenize the same evidence differently.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/context.py` | Select parents/neighbors and pack evidence under a token budget. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/retrieval/snippets.py` | Build match-centered, locator-preserving snippets. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/retrieval/hybrid.py` | Return packed evidence and expansion diagnostics. |
| CREATE | `python/knowledge-rag/tests/unit/retrieval/test_context.py` | Verify budgets, expansion, ordering, IDs, and span deduplication. |
| CREATE | `python/knowledge-rag/tests/integration/test_context_from_index.py` | Verify evidence reconstruction from active parent/child revisions. |

### Acceptance Criteria

- [ ] Packed evidence never exceeds the configured budget under the selected token counter.
- [ ] Parent/neighbor expansion occurs only after final candidate ordering and cannot grow unbounded.
- [ ] Snippets center the match or selected semantic span rather than always returning the document prefix.
- [ ] Every evidence ID resolves to an active source revision and complete citation locator.
- [ ] Context-only spans are distinguishable from primary retrieved hits in the response.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/retrieval/test_context.py tests/integration/test_context_from_index.py
```

---

## Task 03.1.4: Generate grounded answers with validated citations and abstention
**Epic:** 03 — Retrieval and Grounded Answers | **Feature:** 03.1 — Search, Evidence, and Answers
**Size:** L | 8 files | ~650 LOC | **Track:** C

### Current State

- No answer generation exists yet.
- Design pitfalls to avoid: relying on the model to preserve free-text citations embedded in context, one service method that mixes rewrite, retrieval, formatting, and generation, returning answer text while discarding structured evidence, and silently enabling web search inside generation.

### Desired State

- `search()` returns evidence without requiring an LLM; `answer()` composes an optional query rewriter, retriever, context assembler, and answer generator.
- The model returns answer claims with evidence IDs; code rejects unknown IDs, preserves cited evidence, and emits structured citations separately from rendering.
- No-evidence and below-policy cases abstain deterministically before generation; post-generation validation can convert invalid output to a typed grounded failure or safe abstention.
- Retrieved content is clearly delimited as untrusted data and cannot alter system instructions.

### Gap Analysis

- Missing: Separation of retrieval and generation, citation schema and validation, explicit abstention policy, prompt versioning, and complete diagnostics.
- Changes: Scores remain retrieval diagnostics and are not presented as calibrated confidence unless evaluation explicitly calibrates them.
- Blockers: Task 03.1.3 and Task 02.2.2.

### Implementation Research

- Ground answers with an explicit context-only/no-prior-knowledge instruction.
- Use the original user question for answer generation and record any retrieval rewrite separately.
- Web retrieval is an out-of-scope future `Retriever` composition; the generator protocol has no hidden web-search flag.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/generation/answer.py` | Orchestrate optional rewrite, retrieval, abstention, generation, and validation. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/generation/citations.py` | Resolve and validate evidence IDs and citation locators. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/generation/prompts.py` | Version grounded prompts and untrusted-context delimiters. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/generation/abstention.py` | Apply configurable pre- and post-generation abstention policy. |
| MODIFY | `python/knowledge-rag/src/knowledge_rag/knowledge_base.py` | Expose independent `search` and `answer` APIs. |
| CREATE | `python/knowledge-rag/tests/unit/generation/test_citations.py` | Verify valid, unknown, duplicate, missing, and malformed citations. |
| CREATE | `python/knowledge-rag/tests/unit/generation/test_abstention.py` | Verify empty, weak, conflicting, and generator-failure policies. |
| CREATE | `python/knowledge-rag/tests/integration/test_grounded_answer.py` | Verify ingest-to-answer evidence retention with deterministic providers. |

### Acceptance Criteria

- [ ] Retrieval works with no answer model configured.
- [ ] Every returned citation references an evidence item in the same `QueryResponse`; unknown evidence IDs cannot escape validation.
- [ ] Empty evidence deterministically returns the configured abstention response without calling the generator.
- [ ] Prompt and answer diagnostics record versions, providers, query variants, timings, and token counts without recording secrets.
- [ ] The base answer path performs no web search and does not depend on Jira, Slack, or application-framework classes.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/unit/generation tests/integration/test_grounded_answer.py
```

---

## Task 04.1.1: Build the offline evaluation and resilience suite
**Epic:** 04 — Quality, Adapters, and Handoff | **Feature:** 04.1 — Release Readiness
**Size:** L | 9 files | ~600 LOC | **Track:** D

### Current State

- No evaluation corpus, metrics, or resilience suite exists yet.
- Nothing covers end-to-end ingestion, restart consistency, incremental rollback, retrieval metrics, citation validity, abstention, or model/pipeline migration.

### Desired State

- A safe versioned corpus pairs questions with relevant source/chunk locators, answerability, expected technical tokens, and forbidden citations.
- An offline runner reports Recall@K, MRR, nDCG, citation validity, citation recall, abstention accuracy, index counts, latency, and provider call counts.
- Resilience tests exercise restart, unchanged sync, add/modify/delete, pipeline fingerprint change, partial failure, rollback, and concurrency.

### Gap Analysis

- Missing: Quality corpus, metrics, baseline report, regression thresholds, resilience matrix, and benchmark command.
- Changes: Author all fixtures as new safe content; do not copy internal sensitive documents or Slack transcripts.
- Blockers: Task 02.2.2 and Task 03.1.4.

### Implementation Research

- Use deterministic fakes for CI so network and model drift cannot change correctness results.
- Recommended initial release gates are Recall@5 at least 0.85, MRR@10 at least 0.75, citation ID validity 1.0, and abstention accuracy at least 0.90 on the committed corpus.
- Treat latency and live-provider quality as recorded comparison data until representative deployment hardware and corpus size establish stable thresholds.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/evaluation/metrics.py` | Calculate retrieval, citation, abstention, latency, and cost metrics. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/evaluation/runner.py` | Run a versioned corpus and emit JSON plus Markdown summaries. |
| CREATE | `python/knowledge-rag/tests/evaluation/corpus.jsonl` | Store safe labeled queries and expected evidence. |
| CREATE | `python/knowledge-rag/tests/evaluation/test_quality_gates.py` | Enforce committed offline thresholds. |
| CREATE | `python/knowledge-rag/tests/integration/test_restart_and_idempotency.py` | Verify durable unchanged and restart behavior. |
| CREATE | `python/knowledge-rag/tests/integration/test_partial_failure_matrix.py` | Verify rollback and failure records across stages. |
| CREATE | `python/knowledge-rag/tests/integration/test_concurrent_queries_and_sync.py` | Verify active readers never observe staging revisions. |
| CREATE | `python/knowledge-rag/tests/performance/test_local_baseline.py` | Record local indexing/search timing and memory for a fixed corpus. |
| MODIFY | `python/knowledge-rag/pyproject.toml` | Register evaluation and performance markers and coverage rules. |

### Acceptance Criteria

- [ ] Offline quality gates meet or exceed the documented initial thresholds.
- [ ] Citation validation is 100% for all generated test answers and all unanswerable items abstain as labeled.
- [ ] Failure injection never exposes mixed source revisions or loses the prior active corpus.
- [ ] Evaluation output records package version, corpus version, index fingerprint, providers, configuration, and seed.
- [ ] Performance tests report stage start/end, elapsed time, and `current/total` progress without becoming flaky correctness gates.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/evaluation tests/integration
python -m knowledge_rag.cli evaluate --corpus tests/evaluation/corpus.jsonl --output build/evaluation
```

---

## Task 04.1.2: Add thin CLI and optional provider/source adapters
**Epic:** 04 — Quality, Adapters, and Handoff | **Feature:** 04.1 — Release Readiness
**Size:** M | 8 files | ~400 LOC | **Track:** D

### Current State

- No CLI or optional adapters exist yet.
- Design pitfalls to avoid: wiring vector storage directly into framework configuration, embedding provider batch constraints in core logic, coupling credential access and region-specific fetching to document reading, and exposing operations only as framework tools instead of a reusable CLI.

### Desired State

- A thin CLI exposes `index`, `search`, `answer`, `inspect`, and `evaluate` by calling only the public `KnowledgeBase` facade.
- Optional remote-provider embedding/generation and Confluence source adapters accept injected or standard SDK clients and own no application-global credential state.
- Provider contract tests run offline; live tests require explicit markers and environment configuration.

### Gap Analysis

- Missing: Executable reference integration, optional provider extras, safe credential seams, and adapter contract coverage.
- Changes: Keep provider-specific batch, retry, region, model, score, and pagination semantics inside adapters.
- Blockers: Task 02.2.2 and Task 03.1.4.

### Implementation Research

- The CLI owns no business logic; every operation goes through the public facade.
- The package must not contain hardcoded credentials or executable demo auth.
- Remote-provider and Confluence integration errors must map to typed provider/source errors and redact request secrets.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `python/knowledge-rag/src/knowledge_rag/cli.py` | Implement thin library commands and structured/Markdown output modes. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/adapters/remote_provider.py` | Implement optional remote embedding and answer provider adapters. |
| CREATE | `python/knowledge-rag/src/knowledge_rag/adapters/confluence.py` | Implement optional page-source adapter with injected client and child traversal. |
| CREATE | `python/knowledge-rag/examples/local_knowledge_base.py` | Show deterministic local ingestion, search, and answer use. |
| CREATE | `python/knowledge-rag/examples/source_config.toml` | Show a realistic multi-source configuration without credentials. |
| CREATE | `python/knowledge-rag/tests/contract/test_remote_provider_adapter_contract.py` | Verify batching, dimensions, retries, and response mapping offline. |
| CREATE | `python/knowledge-rag/tests/contract/test_confluence_adapter_contract.py` | Verify page identity, child traversal, pagination, and errors offline. |
| MODIFY | `python/knowledge-rag/pyproject.toml` | Add entry point and remote-provider/Confluence extras. |

### Acceptance Criteria

- [ ] CLI commands call the public facade and support structured JSON output without parsing formatted text.
- [ ] Base installation and local tests do not import remote-provider or Confluence dependencies.
- [ ] Adapters receive clients/configuration explicitly and never read or log secrets implicitly.
- [ ] Provider-owned batch and score semantics pass the generic contract suite.
- [ ] Live integration tests are excluded unless an explicit marker and required environment are supplied.

### Validation Steps

```bash
cd python/knowledge-rag
python -m pytest tests/contract/test_remote_provider_adapter_contract.py tests/contract/test_confluence_adapter_contract.py
python -m knowledge_rag.cli --help
```

---

## Task 04.1.3: Document architecture, operations, and package release
**Epic:** 04 — Quality, Adapters, and Handoff | **Feature:** 04.1 — Release Readiness
**Size:** M | 5 files | ~350 LOC | **Track:** D

### Current State

- The package has no completed architecture documentation, operations guide, or release checklist.
- The repository has no Python build or release job.
- Repository documentation does not yet point to the Python knowledge library.

### Desired State

- Package documentation covers architecture, configuration, APIs, operations, and extension points.
- A benchmark run indexes the committed safe corpus, records retrieval metrics against the quality gates, and documents accepted trade-offs.
- Wheel/sdist contents, optional extras, API stability, upgrade/fingerprint migration, backup, rollback, and future consumer adoption are documented.

### Gap Analysis

- Missing: Architecture decision record, benchmark report, operations guide, repository discoverability, and package release checklist.
- Changes: Update only repository/package documentation.
- Blockers: Task 04.1.1 and Task 04.1.2.

### Implementation Research

- Use safe synthetic documents as benchmark evidence; document that final ordering is rank-based (RRF), not raw-score based.
- Follow the root PR and commit wording requirements and keep all descriptions in present-tense imperative tone.

### Files to Create or Modify

| Action | Path | Purpose |
|--------|------|---------|
| MODIFY | `python/knowledge-rag/README.md` | Complete architecture, configuration, APIs, operations, and extension documentation. |
| CREATE | `python/knowledge-rag/docs/operations.md` | Document index rebuild, upgrade/fingerprint migration, backup, rollback, and consumer adoption. |
| CREATE | `python/knowledge-rag/docs/architecture.md` | Record boundaries, data flow, identity, storage, scoring, citation, and security decisions. |
| CREATE | `python/knowledge-rag/docs/benchmark-report.md` | Record corpus, commands, metrics, accepted trade-offs, and release recommendation. |
| MODIFY | `README.md` | Add a concise pointer to the standalone Python knowledge library and preserve unrelated repository instructions. |

### Acceptance Criteria

- [ ] Documentation clearly repeats the Jira/RCA/Slack-orchestration/UI/web-search exclusions.
- [ ] Every planned capability is classified as implemented, optional, deferred, or intentionally excluded.
- [ ] Benchmark results meet Task 04.1.1 quality gates and document the effect of stable IDs, real overlap, RRF, bounded context, and citation validation.
- [ ] `python -m build` produces installable artifacts containing SQL migrations and no test fixtures, caches, secrets, or local indexes.
- [ ] No non-knowledge application source or behavior changes as part of this plan.

### Validation Steps

```bash
cd python/knowledge-rag
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest -m "not live"
python -m build
python -m zipfile -l dist/*.whl
```

---

## Execution Assumptions

- Python 3.11 is the minimum supported runtime; implementation may add newer versions to CI after confirming deployment images.
- SQLite FTS5 plus stored vectors is the reference local backend for correctness and laptop-scale corpora, not a claim that it is the final production-scale vector engine.
- The base package uses no LangChain or comparable framework. Provider integration occurs through narrow protocols and optional extras.
- LLM enrichment, query rewriting, reranking, and generation default to optional. Sparse+dense retrieval remains independently usable.
- Security-sensitive source content, ACL handling, and redaction policies are supplied by the caller; the library preserves and enforces metadata but does not invent application authorization.
- Any future production consumer adoption begins with shadow reads and a rollback switch; it is not authorized by this planning task.

## Open Decisions That Do Not Change the Task Graph

- Confirm which Python 3.11+ versions the eventual deployment environment supports and add that test matrix.
- Select the first production-scale external vector adapter only after measuring real corpus size, update rate, and latency; the core protocol is already planned.
- Curate the initial safe evaluation corpus and approve its quality labels before using the recommended thresholds as a release gate.
