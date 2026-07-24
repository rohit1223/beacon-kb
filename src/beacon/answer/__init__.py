"""Answer pipeline package (Task 03.3).

Grounded answer generation with deterministic pre-abstention (zero LLM calls),
exactly one LiteLLM chat call over delimiter-neutralized untrusted context,
post-abstention, and structural citation validation against the canonical
evidence bundle.

Importing this package performs no side effects.
"""

from __future__ import annotations

from beacon.answer.abstention import (
    ABSTAIN_SENTINEL,
    is_post_abstain,
    should_pre_abstain,
)
from beacon.answer.citations import (
    extract_cited_labels,
    resolve_citations,
    validate_no_unknown_evidence_ids,
)
from beacon.answer.generate import (
    DEFAULT_ABSTAIN_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    LiteLlmClient,
    LlmClient,
    LlmResponse,
    run_answer,
)
from beacon.answer.prompts import (
    NEUTRALIZED_CLOSE,
    NEUTRALIZED_OPEN,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNTRUSTED_CONTEXT_CLOSE,
    UNTRUSTED_CONTEXT_OPEN,
    build_context_block,
    build_user_message,
    neutralize_delimiters,
)

__all__ = [
    "ABSTAIN_SENTINEL",
    "DEFAULT_ABSTAIN_THRESHOLD",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "NEUTRALIZED_CLOSE",
    "NEUTRALIZED_OPEN",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "UNTRUSTED_CONTEXT_CLOSE",
    "UNTRUSTED_CONTEXT_OPEN",
    "LiteLlmClient",
    "LlmClient",
    "LlmResponse",
    "build_context_block",
    "build_user_message",
    "extract_cited_labels",
    "is_post_abstain",
    "neutralize_delimiters",
    "resolve_citations",
    "run_answer",
    "should_pre_abstain",
    "validate_no_unknown_evidence_ids",
]
