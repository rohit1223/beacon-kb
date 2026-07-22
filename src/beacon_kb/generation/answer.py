"""Grounded answer orchestration: retrieval -> abstention -> generation -> validation.

This module composes the following stages (in order) for a single answer() call:

  1. Retrieval  - RetrievalPipeline.search() returns Evidence[].  Zero LLM calls.
  2. Pre-abstention - Deterministic policy: abstain immediately when evidence is
     empty or below the configured threshold.  Zero LLM calls.
  3. Generation - Generator.generate() is called exactly once with the hits.
  4. Post-abstention - Convert "ABSTAIN" sentinel or empty answer_text to safe
     abstention without raising.
  5. Citation validation - Resolve every inline [Sn] label to a concrete
     Evidence item; unknown labels raise CitationError.
  6. Diagnostics - Emit a structured event recording prompt version, provider
     type, query variants, timings, and token counts.  NO secrets.

search() performs ZERO LLM calls.
answer() performs EXACTLY ONE generator call (enforced by the orchestrator; the
  counting fake in tests verifies this).

The Facade's answer() method delegates to run_answer() from this module so the
facade remains a thin shell with no generation logic.  The facade times its
retrieval call and threads the measurement in via ``elapsed_retrieval_s``; it
likewise threads the prepared ``QueryVariants`` so diagnostics record which
query texts drove retrieval.

Query rewrite status: the optional rewrite stage is NOT implemented yet - it
arrives with the agentic layer (Epic 04).  Until then diagnostics record the
variants that exist today (original/sparse/dense from prepare_query), which
all equal the original text because no rewriter is wired in.

Importing this module performs no side effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from beacon_kb.config import AnswerConfig
from beacon_kb.errors import CitationError
from beacon_kb.generation.abstention import is_post_abstain, should_pre_abstain
from beacon_kb.generation.citations import resolve_citations
from beacon_kb.generation.prompts import PROMPT_VERSION
from beacon_kb.models import AnswerResponse, Evidence, Hit, Query, QueryId
from beacon_kb.protocols import Generator, ProgressObserver
from beacon_kb.retrieval.query import QueryVariants

# ---------------------------------------------------------------------------
# Diagnostics record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnswerDiagnostics:
    """Diagnostics captured during a single answer() call.

    Records prompt version, provider type, evidence count, timings, and token
    counts.  Never records secrets (API keys, raw content, personal data).
    """

    query_id: QueryId
    prompt_version: str
    """Stable version string from generation.prompts.PROMPT_VERSION."""

    provider_type: str
    """type(generator).__name__ - identifies the generator class, not credentials."""

    evidence_count: int
    """Number of evidence items passed to the generator."""

    abstained: bool

    query_variants: tuple[tuple[str, str], ...]
    """(kind, text) pairs recording the query variants that drove retrieval.

    Kinds are 'original', 'sparse', and 'dense' (from retrieval.query.QueryVariants).
    Empty when the caller did not supply variants.  The optional rewrite stage
    is not implemented yet (arrives with the agentic layer); until then all
    variants equal the original text.
    """

    elapsed_retrieval_s: float
    """Wall-clock seconds spent in retrieval, as measured by the caller.

    0.0 means "not measured": run_answer() receives pre-computed hits and
    cannot time retrieval itself, so the value is only meaningful when the
    caller (e.g. the facade) times its search() call and supplies it.
    """

    elapsed_generation_s: float
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_answer(
    query: Query,
    generator: Generator,
    hits: list[Hit],
    *,
    evidence: list[Evidence] | None = None,
    config: AnswerConfig | None = None,
    observer: ProgressObserver | None = None,
    query_variants: QueryVariants | None = None,
    elapsed_retrieval_s: float = 0.0,
) -> tuple[AnswerResponse, AnswerDiagnostics]:
    """Orchestrate pre-abstention, generation, post-abstention, and citation validation.

    This function is called by the Facade's answer() method AFTER retrieval.
    It performs exactly one generator call (or zero when pre-abstention fires).

    Args:
        query:     Original user query.
        generator: Injected Generator (must conform to protocols.Generator).
        hits:      Evidence hits from RetrievalPipeline.search() (zero LLM calls).
        evidence:  Canonical Evidence list from RetrievalPipeline.search().
                   When supplied, the generator's returned evidence is validated
                   against it: any chunk ID in the generator's output that is
                   not present in the canonical retrieved evidence raises
                   CitationError.  This closes the loophole where a hostile or
                   buggy generator could fabricate its own evidence tuple and
                   have citations validated only against that fabrication.
        config:    AnswerConfig driving abstain_threshold and token budgets.
                   Defaults to AnswerConfig() if not provided.
        observer:  Optional ProgressObserver for structured events.
        query_variants: Prepared query variants from retrieval (prepare_query).
                   Recorded in diagnostics as (kind, text) pairs; empty when
                   the caller does not supply them.
        elapsed_retrieval_s: Wall-clock seconds the caller spent in retrieval.
                   Defaults to 0.0, meaning "not measured when the caller does
                   not supply it" - run_answer() receives pre-computed hits and
                   cannot time retrieval itself.

    Returns:
        (AnswerResponse, AnswerDiagnostics) - both are always returned,
        even for abstained responses.

    Raises:
        CitationError: If citation validation fails (unknown label in answer_text,
                       or generator evidence not grounded in canonical retrieval).
        BackendError:  If the generator raises a backend failure.
    """
    if config is None:
        config = AnswerConfig()

    provider_type = type(generator).__name__
    variant_pairs: tuple[tuple[str, str], ...] = (
        (
            ("original", query_variants.original_text),
            ("sparse", query_variants.sparse_text),
            ("dense", query_variants.dense_text),
        )
        if query_variants is not None
        else ()
    )
    t_gen_start = time.monotonic()

    # Stage 2: Pre-generation abstention gate.
    if should_pre_abstain(hits, abstain_threshold=config.abstain_threshold):
        elapsed_gen = time.monotonic() - t_gen_start
        response = AnswerResponse(
            query_id=query.id,
            answer_text="",
            evidence=(),
            citations=(),
            abstained=True,
            input_tokens=0,
            output_tokens=0,
        )
        diag = AnswerDiagnostics(
            query_id=query.id,
            prompt_version=PROMPT_VERSION,
            provider_type=provider_type,
            evidence_count=0,
            abstained=True,
            query_variants=variant_pairs,
            elapsed_retrieval_s=elapsed_retrieval_s,
            elapsed_generation_s=elapsed_gen,
            input_tokens=0,
            output_tokens=0,
        )
        _emit(observer, {
            "stage": "answer",
            "status": "abstained",
            "reason": "pre_generation",
            "query_id": str(query.id),
            "prompt_version": PROMPT_VERSION,
            "query_variants": variant_pairs,
            "elapsed_retrieval_s": elapsed_retrieval_s,
        })
        return response, diag

    # Stage 3: Exactly ONE generator call.
    raw: AnswerResponse = generator.generate(
        query,
        hits,
        max_input_tokens=config.max_input_tokens,
        max_output_tokens=config.max_output_tokens,
    )
    elapsed_gen = time.monotonic() - t_gen_start

    # Stage 4: Post-generation abstention.
    # I2 fix: apply the same generator-chunk-ID subset check here as in Stage 5,
    # so a fabricated-evidence abstaining generator is also rejected.
    if is_post_abstain(raw.answer_text) or raw.abstained:
        if evidence is not None:
            canonical_chunk_ids = {str(ev.hit.chunk.id) for ev in evidence}
            for gen_ev in raw.evidence:
                gen_chunk_id = str(gen_ev.hit.chunk.id)
                if gen_chunk_id not in canonical_chunk_ids:
                    raise CitationError(
                        f"Generator returned abstention response with "
                        f"fabricated evidence chunk ID {gen_chunk_id!r} "
                        f"that is not in the canonical retrieved evidence. "
                        f"Rejecting to prevent ungrounded evidence from "
                        f"escaping validation on the abstention path."
                    )
        response = AnswerResponse(
            query_id=query.id,
            answer_text="",
            evidence=raw.evidence,
            citations=(),
            abstained=True,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
        )
        diag = AnswerDiagnostics(
            query_id=query.id,
            prompt_version=PROMPT_VERSION,
            provider_type=provider_type,
            evidence_count=len(raw.evidence),
            abstained=True,
            query_variants=variant_pairs,
            elapsed_retrieval_s=elapsed_retrieval_s,
            elapsed_generation_s=elapsed_gen,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
        )
        _emit(observer, {
            "stage": "answer",
            "status": "abstained",
            "reason": "post_generation",
            "query_id": str(query.id),
            "prompt_version": PROMPT_VERSION,
            "query_variants": variant_pairs,
            "elapsed_retrieval_s": elapsed_retrieval_s,
        })
        return response, diag

    # Stage 5: Citation validation - structural, cannot be bypassed.
    # When canonical retrieval evidence is supplied, first verify the
    # generator's evidence is grounded in it: every chunk ID the generator
    # returned must have been retrieved.  A fabricated chunk is rejected
    # before any label resolution, so a hostile generator cannot smuggle
    # invented evidence past validation by citing its own fabrication.
    if evidence is not None:
        canonical_chunk_ids = {str(ev.hit.chunk.id) for ev in evidence}
        for gen_ev in raw.evidence:
            gen_chunk_id = str(gen_ev.hit.chunk.id)
            if gen_chunk_id not in canonical_chunk_ids:
                raise CitationError(
                    f"Generator returned evidence with chunk ID {gen_chunk_id!r} "
                    f"that is not in the canonical retrieved evidence for this "
                    f"query.  Rejecting to prevent ungrounded citations from "
                    f"escaping validation."
                )
    citations = resolve_citations(raw.answer_text, raw.evidence)

    # Assemble the final response preserving cited evidence.
    response = AnswerResponse(
        query_id=query.id,
        answer_text=raw.answer_text,
        evidence=raw.evidence,
        citations=citations,
        abstained=False,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
    )

    # Stage 6: Diagnostics event (no secrets).
    diag = AnswerDiagnostics(
        query_id=query.id,
        prompt_version=PROMPT_VERSION,
        provider_type=provider_type,
        evidence_count=len(raw.evidence),
        abstained=False,
        query_variants=variant_pairs,
        elapsed_retrieval_s=elapsed_retrieval_s,
        elapsed_generation_s=elapsed_gen,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
    )
    _emit(observer, {
        "stage": "answer",
        "status": "done",
        "query_id": str(query.id),
        "prompt_version": PROMPT_VERSION,
        "provider_type": provider_type,
        "evidence_count": len(raw.evidence),
        "citation_count": len(citations),
        "input_tokens": raw.input_tokens,
        "output_tokens": raw.output_tokens,
        "query_variants": variant_pairs,
        "elapsed_retrieval_s": elapsed_retrieval_s,
        "elapsed_generation_s": elapsed_gen,
    })

    return response, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(observer: ProgressObserver | None, event: dict[str, Any]) -> None:
    """Fire an event to *observer* if one is registered; swallow errors."""
    if observer is not None:
        try:
            observer.on_event(event)
        except Exception:  # noqa: S110 - observers must never disrupt the pipeline
            pass
