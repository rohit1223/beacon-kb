## FOCUS CHAIN
*plans-implement - agentic-rag-library - started 2026-07-22*

> **Track structure:** Track A completes first. Tracks B and C then run in parallel after Track A. Track D starts after Track C. Track E (tool surface, CLI, evaluation, and release) closes out after Tracks C and D, with Epic 07 gating the release.
> Tasks marked [parallel-safe] can be worked concurrently by separate agents once their listed dependencies are complete.
> **Default commit strategy:** Commit at cohesive checkpoints around 700-900 LOC and at the end of every epic.
> **Scope guardrail:** Execute only the standalone `beacon-kb` library. Do not build Jira, incident RCA, Slack crawling or triage, a hosted service, a browser UI, a scheduled crawler, or a web-search retriever, and add no `agentic` extra.

---

### Track A: Standalone Package, Contracts, and Registry (Epic 01, starts immediately)

- [x] 1. EPIC-01 TASK-01.1.1 [S | 8f | ~250 LOC] Scaffold the standalone distribution and quality toolchain - depends: none
- [x] 2. EPIC-01 TASK-01.1.2 [L | 4f | ~700 LOC] Define frozen domain models, typed IDs, errors, and all pipeline and agentic-strategy protocols - depends: 01.1.1
- [x] 3. EPIC-01 TASK-01.1.3 [M | 5f | ~470 LOC] Define typed config, the loader, and the facade shell with PLUGIN_API_VERSION - depends: 01.1.2 [parallel-safe]
- [x] 4. EPIC-01 TASK-01.2.2 [M | 4f | ~340 LOC] Provide deterministic fakes and contract harnesses - depends: 01.1.2 [parallel-safe]
- [x] 5. EPIC-01 TASK-01.2.1 [M | 5f | ~460 LOC] Implement the entry-point plugin registry - depends: 01.1.3
  ↳ Commit checkpoint after TASK-01.2.1 (contracts, config, registry, and fakes; end of Epic 01)

### Track B: Local Storage, Ingestion, and Incremental Indexing (Epic 02, parallel with Track C - depends: Track A complete)

- [x] 6. EPIC-02 TASK-02.1.1 [L | 5f | ~600 LOC] Implement the transactional SQLite store with staged atomic promotion - depends: 01.2.1, 01.2.2 [parallel-safe]
- [x] 7. EPIC-02 TASK-02.2.1 [M | 5f | ~380 LOC] Implement source identity and filesystem and memory connectors - depends: 01.2.1, 01.2.2 [parallel-safe]
- [x] 8. EPIC-02 TASK-02.2.2 [L | 6f | ~620 LOC] Implement structure-aware Markdown, HTML, and PDF parsers - depends: 02.2.1
  ↳ Commit checkpoint after TASK-02.2.2 (store plus source-and-parser slice around 1600 LOC)
- [x] 9. EPIC-02 TASK-02.3.1 [L | 5f | ~560 LOC] Implement parent/child chunking, optional enrichment, and batched embeddings - depends: 02.2.2, 02.1.1
- [x] 10. EPIC-02 TASK-02.3.2 [L | 5f | ~570 LOC] Implement staged full and incremental synchronization - depends: 02.3.1
  ↳ Commit checkpoint after TASK-02.3.2 (deterministic indexing lifecycle; end of Epic 02)

### Track C: Hybrid Retrieval, Context, and Grounded Answers (Epic 03, parallel with Track B - depends: Track A complete; joins Track B at chunking and sync boundaries)

- [ ] 11. EPIC-03 TASK-03.1.1 [L | 6f | ~520 LOC] Implement sparse and dense candidate retrieval with typed scores - depends: 02.1.1
- [ ] 12. EPIC-03 TASK-03.1.2 [M | 3f | ~380 LOC] Add RRF fusion, optional reranking, and diversity - depends: 03.1.1
  ↳ Commit checkpoint after TASK-03.1.2 (hybrid-retrieval slice around 900 LOC)
- [ ] 13. EPIC-03 TASK-03.1.3 [M | 4f | ~360 LOC] Assemble bounded context and expose RetrievalPipeline - depends: 03.1.2, 02.3.1
- [ ] 14. EPIC-03 TASK-03.2.1 [L | 5f | ~560 LOC] Generate grounded answers with validated citations and abstention - depends: 03.1.3, 02.3.2
  ↳ Commit checkpoint after TASK-03.2.1 (grounded single-shot answer contract; end of Epic 03)

### Track D: Agentic Layer (Epic 04, sequential - depends: Track C complete; multi-corpus parts of Track B)

- [ ] 15. EPIC-04 TASK-04.1.1 [M | 5f | ~380 LOC] Implement budgets, stop conditions, and the always-on trace - depends: 03.1.3, 03.2.1
- [ ] 16. EPIC-04 TASK-04.1.2 [L | 4f | ~560 LOC] Implement evidence grading and the reflect loop - depends: 04.1.1, 03.1.3
  ↳ Commit checkpoint after TASK-04.1.2 (budgeted loop plus grading slice around 940 LOC)
- [ ] 17. EPIC-04 TASK-04.2.1 [L | 4f | ~500 LOC] Implement query planning and multi-corpus routing - depends: 04.1.2, 01.2.1
- [ ] 18. EPIC-04 TASK-04.2.2 [M | 5f | ~460 LOC] Implement session memory, synthesis, engine, and facade investigate - depends: 04.2.1, 03.2.1
  ↳ Commit checkpoint after TASK-04.2.2 (agentic engine and facade investigate; end of Epic 04)

### Track E: Tool Surface, CLI, Evaluation, and Release (Epics 05, 06, 07 - depends: Track C and Track D complete; Epic 07 gates release)

- [ ] 19. EPIC-05 TASK-05.1.1 [L | 5f | ~620 LOC] Implement the framework-neutral tool surface and the optional MCP server extra - depends: 03.2.1, 04.2.2 [parallel-safe]
  ↳ Commit checkpoint after TASK-05.1.1 (tool surface and MCP; end of Epic 05)
- [ ] 20. EPIC-07 TASK-07.1.1 [L | 7f | ~560 LOC] Build the offline evaluation and resilience suite with core gates - depends: 02.3.2, 03.2.1 [parallel-safe]
- [ ] 21. EPIC-07 TASK-07.2.1 [M | 4f | ~420 LOC] Add remote and local providers, web, and confluence connectors as plugins - depends: 02.3.2, 03.2.1 [parallel-safe]
- [ ] 22. EPIC-06 TASK-06.1.1 [L | 6f | ~720 LOC] Implement the CLI app, dispatch, renderers, config-driven init, and all commands including doctor and serve-mcp - depends: 01.1.3, 02.3.2, 03.2.1, 04.2.2, 05.1.1
  ↳ Commit checkpoint after TASK-06.1.1 (CLI tool mode; end of Epic 06)
- [ ] 23. EPIC-07 TASK-07.1.2 [M | 3f | ~340 LOC] Add agentic evaluation and budget-regression gates - depends: 07.1.1, 04.2.2
- [ ] 24. EPIC-07 TASK-07.2.2 [M | 13f | ~420 LOC] Finalize extras hygiene, docs, benchmark, and release - depends: 07.1.2, 07.2.1, 05.1.1, 06.1.1
  ↳ Commit checkpoint after TASK-07.2.2 (evaluation, adapters, docs, and release; end of Epic 07)

---

*To resume: find the first [ ] or [~] item and continue from there.*
*To work in parallel: after Track A, assign Track B and Track C separately; within Track A, assign TASK-01.1.3 and TASK-01.2.2 separately after TASK-01.1.2; within Track B, assign TASK-02.1.1 and TASK-02.2.1 separately after Track A; within Track E, assign TASK-05.1.1, TASK-07.1.1, and TASK-07.2.1 separately once their dependencies complete.*
*Reference: EPICS.md | IMPL_PLANS.md*
