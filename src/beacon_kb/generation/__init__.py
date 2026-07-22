"""Generation package: grounded answer synthesis with validated citations.

Public surface (no provider imports at package level):
  - run_answer: Orchestrate retrieval hits -> pre-abstention -> generation -> validation.
  - AnswerDiagnostics: Diagnostics record for a single answer() call.
  - PROMPT_VERSION: Stable version string for the current grounded-answer prompt.
  - should_pre_abstain: Deterministic pre-generation abstention gate.
  - is_post_abstain: Post-generation abstention detector.
  - resolve_citations: Structural citation validator (label -> Evidence).
  - build_context_block: Build labelled untrusted-context block for prompts.
  - build_user_message: Compose user-turn message with context block.
  - neutralize_delimiters: Neutralize literal delimiter tokens in untrusted text.

Importing this package performs no side effects and does NOT import any LLM
provider libraries.  Verify with:

    python -c "import beacon_kb.generation; import sys; \\
        assert not any('openai' in m or 'anthropic' in m for m in sys.modules)"
"""

from __future__ import annotations

from beacon_kb.generation.abstention import is_post_abstain, should_pre_abstain
from beacon_kb.generation.answer import AnswerDiagnostics, run_answer
from beacon_kb.generation.citations import resolve_citations
from beacon_kb.generation.prompts import (
    PROMPT_VERSION,
    build_context_block,
    build_user_message,
    neutralize_delimiters,
)

__all__ = [
    "PROMPT_VERSION",
    "AnswerDiagnostics",
    "build_context_block",
    "build_user_message",
    "is_post_abstain",
    "neutralize_delimiters",
    "resolve_citations",
    "run_answer",
    "should_pre_abstain",
]
