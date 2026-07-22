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
- Parser, Chunker, and Store contract suites - ParserContract, ChunkerContract, and StoreContract are absent until their concrete implementations arrive (Epic 02).
- FTS5 multi-column schema extension and sparse.py per-column bm25() weighting - Epic 02 migration 0002 will extend chunks_fts to separate heading, body, code, and identifiers columns; once that schema lands, BM25SparseRetriever must adopt per-column bm25() weights (e.g. bm25(chunks_fts, 10.0, 1.0, 5.0, 8.0)) replacing the current exact-token OR-boost approach (Epic 02).
