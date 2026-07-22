"""Query validation and variant selection for hybrid retrieval.

Design rules enforced here:
- The original question is preserved verbatim for lexical precision.
- Any rewrite (sparse or dense) is recorded as a separately observable value.
- Sparse retrieval always uses the original question unless an explicit rewriter
  is provided; dense retrieval likewise.
- Empty or whitespace-only query text is rejected before reaching the backend.

Importing this module performs no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from beacon_kb.models import Query


@dataclass(frozen=True, slots=True)
class QueryVariants:
    """Immutable record of validated query text variants.

    original_text is always the verbatim user input.
    sparse_text is the text used for BM25 sparse retrieval (may equal original_text
    or be a separately recorded rewrite).
    dense_text is the text used for dense embedding retrieval (may equal original_text
    or be a separately recorded rewrite).
    Both rewrites are independently observable and testable.
    """

    original_text: str
    """Verbatim query text from the user. Always preserved; never overwritten."""

    sparse_text: str
    """Text for BM25 sparse retrieval.

    Equals original_text unless a sparse_rewriter was provided.
    """

    dense_text: str
    """Text for dense embedding retrieval.

    Equals original_text unless a dense_rewriter was provided.
    """


def prepare_query(
    query: Query,
    *,
    sparse_rewriter: Callable[[str], str] | None = None,
    dense_rewriter: Callable[[str], str] | None = None,
) -> QueryVariants:
    """Validate the query and produce independent sparse and dense text variants.

    The original question is always preserved verbatim in QueryVariants.original_text
    for lexical precision in sparse (BM25) retrieval.
    Any rewrite is recorded as a separately observable value in sparse_text or dense_text.

    Args:
        query:           Query record to validate and prepare.
        sparse_rewriter: Optional callable transforming the query text for sparse retrieval.
                         If None, sparse_text equals the original.
        dense_rewriter:  Optional callable transforming the query text for dense retrieval.
                         If None, dense_text equals the original.

    Returns:
        QueryVariants with original_text, sparse_text, and dense_text.

    Raises:
        ValueError: If query.text is empty or whitespace-only.
    """
    original = query.text
    if not original or not original.strip():
        raise ValueError(
            "Query text must not be empty or whitespace-only. "
            "Provide a non-empty question before calling retrieve()."
        )

    sparse_text = sparse_rewriter(original) if sparse_rewriter is not None else original
    dense_text = dense_rewriter(original) if dense_rewriter is not None else original

    return QueryVariants(
        original_text=original,
        sparse_text=sparse_text,
        dense_text=dense_text,
    )
