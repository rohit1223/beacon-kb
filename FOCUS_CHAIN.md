## FOCUS CHAIN
*plans-implement — python-generic-rag-library — started 2026-07-22*

> **Parallel execution:** Track A completes first. Tracks B and C then run in parallel, with explicit cross-track gates before context assembly and answer generation. Track D starts after
> Tracks B and C complete.
> Tasks marked [✦ parallel-safe] can be worked concurrently by separate agents when their listed dependencies are complete.
> **Default commit strategy:** Commit at cohesive checkpoints around 700–900 LOC, when a self-contained feature block over ~350 LOC completes, and at the end of every epic.
> **Scope guardrail:** Execute only the generic Python knowledge/RAG library. Do not build Jira/RCA, Slack orchestration, UI, or web-search features.

---

### Track A: Library Contracts and Storage (repo: `python/knowledge-rag`, starts immediately)

- [ ] 1. EPIC-01 TASK-01.1.1 [M | 7f | ~260 LOC] Scaffold the Python distribution and quality toolchain
- [ ] 2. EPIC-01 TASK-01.1.2 [L | 8f | ~500 LOC] Define typed domain models, protocols, configuration, and facade — depends: TASK-01.1.1
- [ ] 3. EPIC-01 TASK-01.1.3 [L | 9f | ~590 LOC] Implement the transactional local knowledge store — depends: TASK-01.1.2
  ↳ Note: Commit checkpoint after TASK-01.1.3 (package contract plus local storage foundation; end of Epic 01)

### Track B: Content Ingestion and Indexing (repo: `python/knowledge-rag`, parallel with Track C — depends: Track A complete)

- [ ] 4. EPIC-02 TASK-02.1.1 [M | 6f | ~360 LOC] Implement source discovery and canonical identity — depends: TASK-01.1.2
- [ ] 5. EPIC-02 TASK-02.1.2 [L | 12f | ~760 LOC] Implement structure-aware Markdown, HTML, and PDF parsers — depends: TASK-02.1.1
  ↳ Note: Commit checkpoint after TASK-02.1.2 (complete source-and-parser feature block over ~350 LOC)
- [ ] 6. EPIC-02 TASK-02.2.1 [L | 8f | ~560 LOC] Add parent-child chunking, optional enrichment, and batched embeddings — depends: TASK-02.1.2, TASK-01.1.3
- [ ] 7. EPIC-02 TASK-02.2.2 [L | 10f | ~570 LOC] Implement staged full and incremental synchronization — depends: TASK-02.2.1
  ↳ Note: Commit checkpoint after TASK-02.2.2 (deterministic indexing lifecycle; end of Epic 02)

### Track C: Retrieval and Grounded Answers (repo: `python/knowledge-rag`, parallel with Track B — depends: Track A complete)

- [ ] 8. EPIC-03 TASK-03.1.1 [L | 8f | ~520 LOC] Implement sparse and dense candidate retrieval — depends: TASK-01.1.3
- [ ] 9. EPIC-03 TASK-03.1.2 [M | 6f | ~380 LOC] Add rank fusion, optional reranking, and diversity controls — depends: TASK-03.1.1
  ↳ Note: Commit checkpoint after TASK-03.1.2 (cohesive hybrid-retrieval slice around 900 LOC)
- [ ] 10. EPIC-03 TASK-03.1.3 [M | 5f | ~300 LOC] Assemble bounded parent and neighbor context — depends: TASK-03.1.2, TASK-02.2.1
- [ ] 11. EPIC-03 TASK-03.1.4 [L | 8f | ~650 LOC] Generate grounded answers with validated citations and abstention — depends: TASK-03.1.3, TASK-02.2.2
  ↳ Note: Commit checkpoint after TASK-03.1.4 (grounded answer contract; end of Epic 03)

### Track D: Quality, Adapters, and Handoff (repo: `python/knowledge-rag` and repository docs, sequential — depends: Track B + Track C complete)

- [ ] 12. EPIC-04 TASK-04.1.1 [L | 9f | ~600 LOC] Build the offline evaluation and resilience suite [✦ parallel-safe with TASK-04.1.2] — depends: TASK-02.2.2, TASK-03.1.4
- [ ] 13. EPIC-04 TASK-04.1.2 [M | 8f | ~400 LOC] Add thin CLI and optional provider/source adapters [✦ parallel-safe with TASK-04.1.1] — depends: TASK-02.2.2, TASK-03.1.4
- [ ] 14. EPIC-04 TASK-04.1.3 [M | 5f | ~350 LOC] Document architecture, operations, and package release — depends: TASK-04.1.1, TASK-04.1.2
  ↳ Note: Commit checkpoint after TASK-04.1.3 (release-ready Python knowledge library; end of Epic 04)

---

*To resume: find the first [ ] or [~] item and continue from there.*
*To work in parallel: after Track A, assign Track B and Track C separately; assign TASK-04.1.1 and TASK-04.1.2 separately after both tracks complete.*
*Reference: `EPICS.md` | `IMPL_PLANS.md`*
