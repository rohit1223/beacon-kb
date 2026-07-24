## FOCUS CHAIN V2
*beacon-industry-standard-rag - planned 2026-07-23*

> **Track structure:** Epics 01, 02, and 03 form the sequential spine and must complete in order.
> After Epic 03, Epics 04 (MCP + CLI), 05 (Investigate), and 06 (Evaluation + Hardening) run in parallel worktrees; the single cross-link is TASK-05.3, which also needs TASK-04.1.
> Epic 07 connector waves run independently after Epic 03; within Epic 07 every task is parallel-safe, with wave 2 (07.1, 07.2) scheduled before wave 3 (07.3, 07.4, 07.5) by preference only.
> Epic 08 closes last after all other epics merge.
> Each epic is one branch and one PR.
> **Commit strategy:** One independent green commit per task; every commit leaves `tests/beacon` and the untouched `beacon_kb` suite passing, with mypy strict and ruff clean.
> **Scope guardrail:** All new code under `src/beacon/` and `tests/beacon/`; do not modify `src/beacon_kb` or its tests before TASK-08.1; no AI attribution in any git artifact; no em dashes in docs.

---

### Track 1: Sequential Spine (Epics 01-03, starts immediately)

- [x] 1. EPIC-01 TASK-01.1 [M | 9f | ~450 LOC] Scaffold the beacon package, dependency group, config, and error taxonomy - depends: none
- [x] 2. EPIC-01 TASK-01.2 [L | 6f | ~600 LOC] Qdrant store layer with shadow collections and alias-flip promotion - depends: 01.1
- [x] 3. EPIC-01 TASK-01.3 [M | 6f | ~500 LOC] SQLite state DB with migrations for sources, revisions, fingerprints, and jobs - depends: 01.1 [parallel-safe with 01.2]
- [x] 4. EPIC-01 TASK-01.4 [M | 7f | ~550 LOC] FastAPI skeleton with health, readiness, problem details, auth, and telemetry - depends: 01.1, 01.3
- [x] 5. EPIC-01 TASK-01.5 [M | 6f | ~450 LOC] Collections REST resource, compose file, Dockerfile, and smoke test - depends: 01.2, 01.4
  ↳ End of Epic 01 (branch PR: core service and storage)
- [x] 6. EPIC-02 TASK-02.1 [L | 8f | ~600 LOC] Connector interface, folder connector, sources resource, and upload endpoint - depends: 01.5
- [x] 7. EPIC-02 TASK-02.2 [M | 4f | ~450 LOC] Web and sitemap connector with depth limits and robots respect - depends: 02.1 [parallel-safe with 02.3]
- [x] 8. EPIC-02 TASK-02.3 [M | 5f | ~450 LOC] Docling parsing to structured sections - depends: 02.1 [parallel-safe with 02.2]
- [x] 9. EPIC-02 TASK-02.4 [M | 5f | ~450 LOC] Hierarchical chunking with parent/child links and heading paths - depends: 02.3
- [x] 10. EPIC-02 TASK-02.5 [L | 10f | ~700 LOC] Embeddings auto-detect and the incremental sync engine with staged promotion - depends: 02.2, 02.4
  ↳ End of Epic 02 (branch PR: ingestion wave 1; ported sync regression suites green)
- [x] 11. EPIC-03 TASK-03.1 [L | 7f | ~550 LOC] Hybrid retrieval with enforced payload filters and optional rerank - depends: 02.5
- [x] 12. EPIC-03 TASK-03.2 [M | 5f | ~500 LOC] Evidence assembly with expansion, budgets, labels, and snippets - depends: 03.1
- [x] 13. EPIC-03 TASK-03.3 [L | 8f | ~650 LOC] Grounded answer with abstention, injection defense, and citation validation - depends: 03.2
- [x] 14. EPIC-03 TASK-03.4 [M | 6f | ~500 LOC] REST search and answer with cost-contract, recall, and smoke tests - depends: 03.3
  ↳ End of Epic 03 (branch PR: retrieval and grounded answers; parallel tracks unlock)

### Track 2: MCP and CLI (Epic 04 - depends: Epic 03 complete; parallel worktree)

- [ ] 15. EPIC-04 TASK-04.1 [M | 4f | ~450 LOC] FastMCP server in-process with kb_search, kb_answer, kb_sync_status, kb_list_collections - depends: 03.4 [parallel-safe]
- [ ] 16. EPIC-04 TASK-04.2 [S | 3f | ~250 LOC] MCP stdio mode and typed MCP error mapping - depends: 04.1
- [ ] 17. EPIC-04 TASK-04.3 [M | 4f | ~450 LOC] Typer CLI: beacon serve, sync, search, ask - depends: 03.4 [parallel-safe]
- [ ] 18. EPIC-04 TASK-04.4 [M | 4f | ~350 LOC] Model auto-detect UX, doctor diagnostics, and Ollama compose profile - depends: 04.3
  ↳ End of Epic 04 (branch PR: MCP and CLI)

### Track 3: Investigate (Epic 05 - depends: Epic 03 complete; parallel worktree; 05.3 joins Track 2 at 04.1)

- [ ] 19. EPIC-05 TASK-05.1 [M | 5f | ~450 LOC] LangGraph graph skeleton, budget model, and SQLite checkpointing - depends: 03.4 [parallel-safe]
- [ ] 20. EPIC-05 TASK-05.2 [L | 6f | ~600 LOC] Plan, retrieve, grade, reflect, and synthesize nodes reusing retrieval and answer - depends: 05.1
- [ ] 21. EPIC-05 TASK-05.3 [M | 5f | ~450 LOC] POST /investigate with SSE trace and the kb_investigate MCP tool - depends: 05.2, 04.1
- [ ] 22. EPIC-05 TASK-05.4 [M | 3f | ~300 LOC] Investigate budget-regression and cost-contract tests - depends: 05.2 [parallel-safe with 05.3]
  ↳ End of Epic 05 (branch PR: investigate loop)

### Track 4: Evaluation and Hardening (Epic 06 - depends: Epic 03 complete; parallel worktree)

- [ ] 23. EPIC-06 TASK-06.1 [M | 5f | ~450 LOC] RAGAS golden-set gates in CI - depends: 03.4 [parallel-safe]
- [ ] 24. EPIC-06 TASK-06.2 [M | 5f | ~450 LOC] OpenTelemetry tracing, structured logs, and per-request cost accounting - depends: 03.4 [parallel-safe]
- [ ] 25. EPIC-06 TASK-06.3 [M | 8f | ~400 LOC] Load sanity, docs, and quickstarts - depends: 06.1, 06.2
  ↳ End of Epic 06 (branch PR: evaluation and hardening)

### Track 5: Connector Waves 2 and 3 (Epic 07 - depends: Epic 03 complete; interface from Epic 02; all tasks parallel-safe)

- [ ] 26. EPIC-07 TASK-07.1 [M | 3f | ~400 LOC] Confluence connector (wave 2) - depends: 02.5 [parallel-safe]
- [ ] 27. EPIC-07 TASK-07.2 [M | 3f | ~400 LOC] Notion connector (wave 2) - depends: 02.5 [parallel-safe]
- [ ] 28. EPIC-07 TASK-07.3 [M | 3f | ~400 LOC] Google Drive connector (wave 3) - depends: 02.5 [parallel-safe]
- [ ] 29. EPIC-07 TASK-07.4 [M | 3f | ~400 LOC] Slack connector (wave 3) - depends: 02.5 [parallel-safe]
- [ ] 30. EPIC-07 TASK-07.5 [M | 3f | ~400 LOC] GitHub connector (wave 3) - depends: 02.5 [parallel-safe]
  ↳ End of Epic 07 (one branch PR per wave is acceptable; wave 2 lands before wave 3)

### Track 6: Cleanup and Release (Epic 08 - depends: Epics 04, 05, 06, 07 complete; runs last)

- [ ] 31. EPIC-08 TASK-08.1 [M | repo-wide deletions | ~200 net LOC] Delete beacon_kb and legacy tests after parity check - depends: 04.4, 05.4, 06.3, 07.5
- [ ] 32. EPIC-08 TASK-08.2 [S | 6f | ~300 LOC] Final docs, release checklist, and v1.0.0 - depends: 08.1
  ↳ End of Epic 08 (release PR: v1.0.0)

---

*To resume: find the first [ ] or [~] item whose dependencies are all [x] and continue from there; Epics 01-03 tasks are detailed in IMPL_PLANS_V2.md, and Epics 04-08 tasks must be detailed just-in-time (same per-task format) before their epic starts.*
*To work in parallel: within Epic 01, assign 01.2 and 01.3 separately after 01.1; within Epic 02, assign 02.2 and 02.3 separately after 02.1; after Epic 03, assign Tracks 2, 3, 4, and 5 to separate worktrees, holding TASK-05.3 until TASK-04.1 lands.*
*Reference: EPICS_V2.md | IMPL_PLANS_V2.md | spec: docs/superpowers/specs/2026-07-23-beacon-industry-standard-rag-design.md*
