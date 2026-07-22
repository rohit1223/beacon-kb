# beacon-kb Roadmap

This file tracks capabilities that are deferred from the current epic.
Each entry records what is missing, why it was deferred, and which epic will address it.

## Deferred items

- [DONE - Task 03.1.3] Query.top_k vs config.retrieval.top_k reconciliation - per-query top_k overrides config when it differs from the default (10); documented in pipeline.py at _resolve_top_k().
- StopCondition trace parameter typing to AgenticTrace - the current `Any` annotation should be narrowed once the contract suite and fakes feed real AgenticTrace values instead of None/dict placeholders (Epic 04).
- Entry-point scan caching per group if hot - avoid repeated metadata scans in tight loops once Epic 02 introduces high-frequency resolution (Epic 02).
- Config-docs note that name-shaped inline secrets are indistinguishable from env-var names - document this footgun clearly and consider a distinct prefix for secret references (Epic 06).
- Consolidate pytest marker registration to one location and enable --strict-markers - markers are currently registered in both pyproject.toml and tests/conftest.py; enabling strict mode will catch typos in test files.
- pyproject-vs-ALL_GROUPS sync guard test - a test that parses pyproject.toml and asserts that the entry-point group headers match groups.ALL_GROUPS exactly (added in tests/contract/).
- Sample-plugin CI env flag to fail instead of skip when plugin missing - introduce a BEACON_REQUIRE_SAMPLE_PLUGIN env flag so CI can distinguish "not installed" from "skipped deliberately".
- Registry test reset should re-register builtins - clear_registry() in tests currently leaves the built-in HeuristicTokenCounter unregistered; reset helpers should restore builtins automatically.
- ParserContract suite - ChunkerContract and StoreContract now ship in beacon_kb.testing; ParserContract remains intentionally absent until a reusable parser contract is needed.
- [DONE - Task 03.1.3] Sparse.py per-column bm25() weighting adoption - BM25SparseRetriever now calls store.retrieve(weights=(1.0, 10.0, 5.0)) for text/heading/code column weights. The exact-token OR-boost is retained as a complementary mechanism for technical identifiers (see sparse.py module docstring).
- Registry factory / deferred-construction registration - the registry currently only accepts pre-built instances via ``register()`` and ``register_builtin()``.
  Components that require caller-supplied configuration (FilesystemConnector, HtmlParser, PdfParser) cannot register as builtins without a factory or deferred-construction mechanism.
  Until this is added, all configuration-requiring or optional-dependency components must be registered explicitly by the caller after construction.
  This affects the FilesystemConnector, HtmlParser, and PdfParser, all of which document explicit registration as the workaround in ``registry/builtins.py``.
- [PARTIALLY ADDRESSED - Task 03.1.3] Parent-chunk gap: ChunkKind.PARENT records are not materialized; the implicit parent is section_id plus parent_locator carried on every child. Task 03.1.3's context assembly (context.py) treats neighbor expansion via prev/next chunk chains - it does NOT fetch a non-existent PARENT record. Parent-level sparse retrieval (emitting real PARENT records to the store) remains a follow-on task.
- Embedding retry back-off: time.sleep(0.0) in indexing/embedding.py is a placeholder; replace with real exponential back-off once retry policy is finalized (Epic 03).
- dense_retrieve full-scan scaling: SQLiteStore.dense_retrieve loads ALL active embedding vectors into memory and scores them with NumPy on every query. This is acceptable for local/embedded corpora but does not scale; add an ANN index (e.g. sqlite-vss / hnswlib) or a pluggable vector backend (Epic 03+).
- FTS post-match filter perf: the FTS5 sparse query joins to chunks and filters on active=1 after the MATCH; on large corpora with many retired chunks this scans matched-but-inactive rows. Consider a partial/covering index or a content-referenced FTS table pruned on retire (Epic 03+).
- Staged-row garbage collection: rolled-back and superseded revisions leave inactive chunk/embedding/revision rows and audit-preserved revision records; add a bounded GC/vacuum pass so a long-lived database does not accumulate dead rows unbounded.
- Enriched-text persistence: SyncEngine calls enrichment for side effects but discards the returned text; wire the enriched output into an 'enriched_text' column on chunks, add it to FTS5, and extend the Store protocol contract so summaries/keywords/FAQs become searchable metadata.
- Pooled/locked store variant: SQLiteStore owns one connection and is single-threaded (check_same_thread stays True; cross-thread use fails fast). Add a pooled or lock-guarded multi-threaded store variant for concurrent callers.
- First provider Generator must route through prompts.build_context_block: the current FakeGenerator and any future provider implementation must call prompts.build_context_block() to format the retrieved evidence into the prompt context block, rather than formatting evidence ad-hoc. This ensures consistent prompt formatting across all generator implementations (Epic 04 / first-provider epic).
- Epic 04 query rewriting must switch retrievers to QueryVariants texts: the optional query-rewrite stage (Epic 04) must wire sparse_retriever to query_variants.sparse_text and dense_retriever to query_variants.dense_text instead of the original query.text. The current pipeline.py uses query.text for both legs; the ROADMAP note at pipeline.py line ~212 captures this exactly.
