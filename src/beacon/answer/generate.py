"""Grounded answer orchestration: exactly one provider call, staged (Task 03.3).

Ported from beacon_kb.generation.answer onto the beacon EvidenceBundle and a
LiteLLM chat completion seam.

Stages, in order, for one ``run_answer`` call:

  1. Pre-abstention gate - deterministic, ZERO provider calls: abstain when the
     bundle has no primary HIT evidence or every HIT score is below threshold.
  2. Generation - the LLM client is invoked EXACTLY ONCE over the built prompt
     (system + delimiter-wrapped untrusted context + question).
  3. Post-abstention gate - convert the ABSTAIN sentinel or an empty answer to a
     safe abstention with empty citations.
  4. Citation validation - resolve every ``[S#]`` label against the canonical
     bundle; unknown labels raise a typed CitationError.
  5. Result assembly - a typed AnswerResult preserving answer text, citations,
     the evidence bundle, and diagnostics (prompt version, model, token counts,
     elapsed generation time).  No secrets are recorded.

The exactly-one-call contract is structural: the client is invoked at most once
per answer and zero times when pre-abstention fires.  Provider failures are
translated to the typed ``backend`` error kind with no secret material in the
message.  Both the answer and abstention paths run citation validation.

Importing this module performs no side effects (LiteLLM is imported lazily
inside ``LiteLlmClient.complete`` so the module has no import-time provider
dependency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from beacon.answer.abstention import is_post_abstain, should_pre_abstain
from beacon.answer.citations import resolve_citations
from beacon.answer.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_context_block,
    build_user_message,
)
from beacon.errors import BackendError, BeaconError
from beacon.models import (
    AnswerDiagnostics,
    AnswerResult,
    EvidenceBundle,
    EvidenceRole,
)

__all__ = [
    "DEFAULT_ABSTAIN_THRESHOLD",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "LiteLlmClient",
    "LlmClient",
    "LlmResponse",
    "run_answer",
]

DEFAULT_ABSTAIN_THRESHOLD: float = 0.0
"""Default pre-abstention threshold.

0.0 disables the score gate.  Retrieval produces RRF-scale scores bounded by
2/k (approx 0.033 at k=60); any positive default silences every hybrid answer
before the model is called (the v1 bug), so the default gate is off.
"""

DEFAULT_TEMPERATURE: float = 0.0
"""Deterministic decoding by default."""

DEFAULT_MAX_TOKENS: int = 2048
"""Default completion token ceiling for the single provider call."""


# ---------------------------------------------------------------------------
# Provider seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """The provider's completion, normalized for the answer stage.

    Attributes:
        text:          The completion text (may be empty).
        input_tokens:  Prompt token count reported by the provider (0 if unreported).
        output_tokens: Completion token count reported by the provider (0 if unreported).
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LlmClient(Protocol):
    """The single-call chat completion seam consumed by ``run_answer``.

    Tests inject a deterministic counting fake; production wires ``LiteLlmClient``.
    Implementations must not perform retries that turn one logical answer into
    multiple provider calls - the exactly-one-call contract is measured here.
    """

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LlmResponse:
        """Run one chat completion and return the normalized response."""
        ...


class LiteLlmClient:
    """LiteLLM-backed ``LlmClient`` making exactly one chat completion per call.

    LiteLLM is imported lazily so importing this module has no provider
    dependency.  Any provider exception is translated to a typed ``BackendError``
    with a fixed, secret-free message; the original exception is chained via
    ``from`` for local debugging but never rendered into the error text.
    """

    def __init__(self, *, timeout: float | None = None) -> None:
        self._timeout = timeout

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LlmResponse:
        """Perform one LiteLLM completion and normalize the response.

        Raises:
            BackendError: On any provider failure.  The message never contains
                          provider internals or secret material.
        """
        import litellm

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._timeout,
            )
        except Exception as exc:
            # Every provider error becomes a typed, secret-free backend error.
            raise BackendError(
                "The answer provider call failed. See server logs for details."
            ) from exc

        return _normalize_litellm_response(response)


def _normalize_litellm_response(response: object) -> LlmResponse:
    """Extract text and token usage from a LiteLLM ModelResponse defensively.

    The provider response shape varies across models; missing usage is captured
    as 0 rather than raising, so token counts are best-effort diagnostics.
    """
    text = ""
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content

    input_tokens = 0
    output_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(prompt, int):
            input_tokens = prompt
        if isinstance(completion, int):
            output_tokens = completion

    return LlmResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_answer(
    bundle: EvidenceBundle,
    query_text: str,
    client: LlmClient,
    *,
    model: str,
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AnswerResult:
    """Generate a grounded answer from a canonical evidence bundle.

    Performs the pre-abstain gate (zero calls), exactly one provider call, the
    post-abstain gate, and citation validation against the canonical bundle,
    then returns a typed ``AnswerResult``.

    Args:
        bundle:            Canonical evidence bundle from retrieval assembly.  Its
                           labels are the ONLY citation targets; model-echoed
                           content is never trusted.
        query_text:        Plain user query text.
        client:            Single-call chat completion seam.  Invoked at most once,
                           zero times on pre-abstention.
        model:             LiteLLM model name recorded in diagnostics.
        abstain_threshold: RRF-scale pre-abstention threshold.  Default 0.0 (off).
        temperature:       Decoding temperature for the single call.
        max_tokens:        Completion token ceiling for the single call.

    Returns:
        AnswerResult preserving answer text, citations, the evidence bundle, and
        diagnostics.

    Raises:
        CitationError: If any cited label is absent from the canonical bundle.
        BackendError:  If the provider call fails.
    """
    evidence_count = sum(1 for ev in bundle.evidence if ev.role is EvidenceRole.HIT)
    t_start = time.monotonic()

    # Stage 1: Pre-generation abstention gate - ZERO provider calls.
    if should_pre_abstain(bundle, abstain_threshold=abstain_threshold):
        return _abstained_result(
            bundle,
            model=model,
            evidence_count=evidence_count,
            input_tokens=0,
            output_tokens=0,
            elapsed=time.monotonic() - t_start,
        )

    # Stage 2: Build the prompt and make EXACTLY ONE provider call.
    # Any client failure that is not already a typed Beacon error is translated
    # to a secret-free BackendError so provider internals never leak.
    messages = _build_messages(bundle, query_text)
    try:
        response = client.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except BeaconError:
        raise
    except Exception as exc:
        # Every non-typed client failure becomes a secret-free backend error.
        raise BackendError(
            "The answer provider call failed. See server logs for details."
        ) from exc
    elapsed = time.monotonic() - t_start

    # Stage 3: Post-generation abstention gate.
    if is_post_abstain(response.text):
        return _abstained_result(
            bundle,
            model=model,
            evidence_count=evidence_count,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed=elapsed,
        )

    # Stage 4: Citation validation against the canonical bundle (never model echo).
    # Note: validate_no_unknown_evidence_ids is intentionally not called here.  The
    # LlmClient type boundary (text-only responses, no structured model output) closes
    # the fabricated-evidence surface; the function remains exported for future callers
    # that accept structured model output.
    citations = resolve_citations(response.text, bundle)

    # Stage 5: Assemble the grounded result.
    diagnostics = AnswerDiagnostics(
        prompt_version=PROMPT_VERSION,
        model=model,
        evidence_count=evidence_count,
        abstained=False,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        elapsed_generation_s=elapsed,
    )
    return AnswerResult(
        answer_text=response.text,
        citations=citations,
        evidence=bundle,
        abstained=False,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages(bundle: EvidenceBundle, query_text: str) -> list[dict[str, str]]:
    """Build the chat messages: trusted system prompt, then delimiter-wrapped context.

    Only primary HIT evidence and its expanded CONTEXT spans are labelled for the
    model; every evidence text is neutralized and wrapped in untrusted-context
    delimiters by ``build_context_block``.
    """
    evidence_texts: list[tuple[str, str]] = [
        (ev.label, ev.snippet.text if ev.snippet is not None else "")
        for ev in bundle.evidence
    ]
    context_block = build_context_block(evidence_texts)
    user_message = build_user_message(query_text, context_block)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _abstained_result(
    bundle: EvidenceBundle,
    *,
    model: str,
    evidence_count: int,
    input_tokens: int,
    output_tokens: int,
    elapsed: float,
) -> AnswerResult:
    """Build a safe abstained AnswerResult with empty answer text and citations."""
    diagnostics = AnswerDiagnostics(
        prompt_version=PROMPT_VERSION,
        model=model,
        evidence_count=evidence_count,
        abstained=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_generation_s=elapsed,
    )
    return AnswerResult(
        answer_text="",
        citations=(),
        evidence=bundle,
        abstained=True,
        diagnostics=diagnostics,
    )
