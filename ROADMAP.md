# beacon-kb Roadmap

This file tracks capabilities that are deferred from the current epic.
Each entry records what is missing, why it was deferred, and which epic will address it.

## Deferred items

- Query.top_k vs config.retrieval.top_k reconciliation - resolve which value takes precedence at query time (Epic 03).
- StopCondition trace parameter typing to AgenticTrace - the current `Any` annotation should be narrowed once the contract suite and fakes feed real AgenticTrace values instead of None/dict placeholders (Epic 04).
- Entry-point scan caching per group if hot - avoid repeated metadata scans in tight loops once Epic 02 introduces high-frequency resolution (Epic 02).
- Config-docs note that name-shaped inline secrets are indistinguishable from env-var names - document this footgun clearly and consider a distinct prefix for secret references (Epic 06).
- Consolidate pytest marker registration to one location and enable --strict-markers - markers are currently registered in both pyproject.toml and tests/conftest.py; enabling strict mode will catch typos in test files.
- pyproject-vs-ALL_GROUPS sync guard test - a test that parses pyproject.toml and asserts that the entry-point group headers match groups.ALL_GROUPS exactly (added in tests/contract/).
- Sample-plugin CI env flag to fail instead of skip when plugin missing - introduce a BEACON_REQUIRE_SAMPLE_PLUGIN env flag so CI can distinguish "not installed" from "skipped deliberately".
- Registry test reset should re-register builtins - clear_registry() in tests currently leaves the built-in HeuristicTokenCounter unregistered; reset helpers should restore builtins automatically.
- ParserContract suite - ChunkerContract and StoreContract now ship in beacon_kb.testing; ParserContract remains intentionally absent until a reusable parser contract is needed.
- Registry factory / deferred-construction registration - the registry currently only accepts pre-built instances via ``register()`` and ``register_builtin()``.
  Components that require caller-supplied configuration (FilesystemConnector, HtmlParser, PdfParser) cannot register as builtins without a factory or deferred-construction mechanism.
  Until this is added, all configuration-requiring or optional-dependency components must be registered explicitly by the caller after construction.
  This affects the FilesystemConnector, HtmlParser, and PdfParser, all of which document explicit registration as the workaround in ``registry/builtins.py``.
- Parent-chunk gap: ChunkKind.PARENT records are not materialized; the implicit parent is section_id plus parent_locator carried on every child; Epic 03's context assembly (Task 03.1.3) must treat section_id as parent identity, and parent-level sparse retrieval requires a follow-on task emitting PARENT records to the store.
- Embedding retry back-off: time.sleep(0.0) in indexing/embedding.py is a placeholder; replace with real exponential back-off once retry policy is finalized (Epic 03).
- dense_retrieve full-scan scaling: SQLiteStore.dense_retrieve loads ALL active embedding vectors into memory and scores them with NumPy on every query. This is acceptable for local/embedded corpora but does not scale; add an ANN index (e.g. sqlite-vss / hnswlib) or a pluggable vector backend (Epic 03+).
- FTS post-match filter perf: the FTS5 sparse query joins to chunks and filters on active=1 after the MATCH; on large corpora with many retired chunks this scans matched-but-inactive rows. Consider a partial/covering index or a content-referenced FTS table pruned on retire (Epic 03+).
- Staged-row garbage collection: rolled-back and superseded revisions leave inactive chunk/embedding/revision rows and audit-preserved revision records; add a bounded GC/vacuum pass so a long-lived database does not accumulate dead rows unbounded.
- Enriched-text persistence: SyncEngine calls enrichment for side effects but discards the returned text; wire the enriched output into an 'enriched_text' column on chunks, add it to FTS5, and extend the Store protocol contract so summaries/keywords/FAQs become searchable metadata.
- Pooled/locked store variant: SQLiteStore owns one connection and is single-threaded (check_same_thread stays True; cross-thread use fails fast). Add a pooled or lock-guarded multi-threaded store variant for concurrent callers.
