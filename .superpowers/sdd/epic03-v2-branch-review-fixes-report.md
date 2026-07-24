# Epic 03 Branch Review Fixes - Implementation Report

**Branch:** feat/beacon-03-retrieval
**Base commit:** 7ca951c
**Plan:** docs/superpowers/plans/2026-07-24-epic03-branch-review-fixes.md

---

## Summary

All Tasks 1-7 landed in a single hardening commit on top of 7ca951c.

---

## Per-Task Changes

### Task 1: Evidence.text field

- `src/beacon/models.py`: Added `text: str = ""` field to `Evidence` with docstring noting it carries the full chunk_text (not the 400-char snippet excerpt), and a note on Evidence.score vs rerank ordering.
- `src/beacon/retrieval/evidence.py`: Populated `ev.text = str(payload.get("chunk_text", ""))` for both primary HIT and CONTEXT span items. Dropped unused `chunk_id` first parameter from `_resolve_neighbors` and updated both call sites.
- `tests/beacon/unit/test_evidence_text_field.py`: Created with TestEvidenceTextField, TestRecapVsPromptConsistency, and TestBuildMessagesUsesFullText classes.

### Task 2: _build_messages uses ev.text

- `src/beacon/answer/generate.py`: Changed `_build_messages` list comprehension from `ev.snippet.text if ev.snippet is not None else ""` to `ev.text`. Updated docstring.
- `tests/beacon/integration/test_grounded_answer.py`: Fixed `_hit` helper to pass `text=text` to `Evidence.text` so adversarial/reference text flows through the prompt path correctly.
- `tests/beacon/unit/test_evidence_text_field.py`: TestBuildMessagesUsesFullText class verifies the distinctive tail phrase beyond 400 chars appears in the built prompt.

### Task 3: Sparse-floor missing collection BackendError

- `src/beacon/retrieval/hybrid.py`: Hoisted collection-existence check out of `_check_dense_dimension` into `search()`, before the dense/sparse branch. Both modes now raise `BackendError` when the live revision points at a missing physical collection.
- `tests/beacon/unit/test_sparse_floor_missing_collection.py`: Created with TestSparseFloorMissingCollection verifying BackendError is raised (not silent empty list) in sparse-only mode.

### Task 4: Config knobs wired + dead ones deleted

- `src/beacon/config.py`:
  - `RetrievalSettings`: deleted `rerank` and `parent_expansion` fields; updated `top_k` docstring to note it is the server-wide default.
  - `AnswerSettings`: added `llm_timeout_s: float = 60.0`; expanded `abstain_when_uncertain` docstring with score-scale note.
  - `safe_dump()`: removed `rerank` and `parent_expansion` from retrieval section; added `llm_timeout_s` to answer section.
- `src/beacon/server/routes/answer.py`: wired `abstain_threshold` from config (`score_threshold` when `abstain_when_uncertain=True`, else 0.0); wired `LiteLlmClient(timeout=settings.answer.llm_timeout_s)`.
- `src/beacon/server/routes/search.py`: changed `top_k` field to `int | None = Field(default=None, ge=1)`; route resolves `body.top_k if body.top_k is not None else settings.retrieval.top_k`.
- `tests/beacon/unit/test_config_knob_wiring.py`: Created verifying deleted fields absent, `llm_timeout_s` present and defaulting to 60.0, safe_dump includes/excludes correct keys.

### Task 5: Transport-free pipeline extraction

- `src/beacon/retrieval/pipeline.py`: Created with `TOKEN_BUDGET = 8192`, `build_fetch_chunk(store, collection)`, and `run_search_pipeline(*, state_db, store, embedder, spec, query_text, top_k, token_budget, ...)`. Uses `Embedder` protocol (not `Any`). Zero FastAPI imports.
- `src/beacon/server/routes/search.py`: Rewritten as thin adapter. Removed `_make_fetch_chunk`, `build_evidence_bundle`, inline `HybridRetriever` usage. Added `response_model=EvidenceBundle`. Returns `EvidenceBundle` directly.
- `src/beacon/server/routes/answer.py`: Rewritten as thin adapter. Imports `run_search_pipeline` from pipeline (not `build_evidence_bundle` from search). Added `response_model=AnswerResult`. Returns `AnswerResult` directly.
- `tests/beacon/unit/test_pipeline_standalone.py`: Created verifying `run_search_pipeline` returns `EvidenceBundle` with no FastAPI objects in the call signature.

### Task 6: Minor diagnostics, comments, docs, ROADMAP

- `src/beacon/models.py`: Added `uncited_answer: bool = False` to `AnswerDiagnostics` with docstring linking to Epic 06 RAGAS eval.
- `src/beacon/answer/generate.py`: Wired `uncited_answer = len(citations) == 0` into Stage 5 assembly. Added post-abstain sentinel comment explaining exact-match semantics.
- `src/beacon/answer/abstention.py`: Added score-scale note to module docstring (hybrid RRF bounded by 2/k ~0.033 vs sparse-only unbounded TF scores).
- `ROADMAP.md`: Added DONE entry for pipeline extraction; added ROADMAP entries for deleted rerank and parent_expansion knobs; added entries for llm_timeout_s follow-on (Epic 06) and uncited_answer diagnostic feed-in.

### Rerank-knob decision

`RetrievalSettings.rerank` and `.parent_expansion` were deleted (honest option from the plan). No CrossEncoderReranker can be tested offline without model downloads. ROADMAP entries added to re-add them wired in Epic 04/05 when a FakeScorer-backed offline path is available.

---

## Gate Output

**mypy (cold - no cache):**
```
Success: no issues found in 146 source files
```

**ruff:**
```
All checks passed!
```

**pytest:**
```
1746 passed, 2 skipped in 8.80s
```

(Baseline was 1726 passed + 2 skipped; 20 new tests added.)
