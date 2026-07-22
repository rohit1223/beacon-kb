"""Weighted FTS5 BM25 sparse retrieval with exact-token boosts.

Design rules enforced here:
- BM25 scores are explicit and direction-declared: higher sparse_score = more relevant.
- Only sparse_score is set on returned hits; dense_score, fusion_score, and
  rerank_score remain None.
- Exact-token boosts for error codes, command names, and identifiers are applied at
  query construction time as exact-phrase OR clauses appended to the base FTS5 query.
- Filters are applied consistently before candidates leave this retriever.
- Sparse-only mode is first-class: no embedder dependency anywhere in this module.
- BackendError from the underlying store propagates typed; never swallowed.

Current boost mechanism - exact-token OR-boosting:
  The chunks_fts FTS5 table has a SINGLE text column (schema from migration 0001).
  Per-column bm25() weighting - e.g. bm25(chunks_fts, 10.0, 1.0, 5.0) to boost
  heading matches over body matches - is NOT possible with a single-column schema.
  The current implementation approximates importance via exact-phrase OR clauses
  for technical identifiers (error codes, command names, CamelCase identifiers).
  These appear as additional OR alternatives in the FTS5 MATCH expression so that
  chunks containing the exact token surface higher in BM25 ranking, without altering
  dense candidate ordering.

Roadmap - per-column bm25() weighting (Epic 02 follow-up):
  Epic 02 migration 0002 extends chunks_fts to a multi-column schema (heading, body,
  code, identifiers). Once that schema lands, this module will adopt per-column
  bm25() weights - e.g. bm25(chunks_fts, 10.0, 1.0, 5.0, 8.0) - replacing the
  current OR-boost approach with proper column-weighted BM25. See ROADMAP.md.

Importing this module performs no side effects.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from beacon_kb.models import Hit, Query
from beacon_kb.retrieval.filters import FilterSpec, apply_filters

if TYPE_CHECKING:
    from beacon_kb.storage.sqlite import SQLiteStore


# ---------------------------------------------------------------------------
# Token boost patterns for exact technical identifiers
# These patterns identify tokens that should rank strongly when matched exactly:
# - Error codes: uppercase_SNAKE_CASE, ERR_*, ERROR_*, E_*
# - Command names: hyphenated-commands, slash/commands
# - Identifiers: CamelCase, ALL_CAPS
# ---------------------------------------------------------------------------

_ERROR_CODE_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_]{2,}(?:_[A-Z0-9]+)+|E[A-Z0-9]{2,}|ERR(?:OR)?[_A-Z0-9]*)\b"
)
_COMMAND_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b|/[a-z][a-z0-9/_-]+")
_IDENTIFIER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b")  # CamelCase


def _extract_boost_tokens(text: str) -> list[str]:
    """Extract exact-token candidates for boosting from the query text.

    Returns a deduplicated list of tokens that warrant exact-match boosting:
    error codes, command names, and identifiers. These are used to construct
    a boosted FTS5 query string.

    Args:
        text: Raw query text.

    Returns:
        List of distinct token strings to boost.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    for pattern in (_ERROR_CODE_RE, _COMMAND_RE, _IDENTIFIER_RE):
        for match in pattern.finditer(text):
            tok = match.group(0)
            if tok not in seen:
                tokens.append(tok)
                seen.add(tok)

    return tokens


def _build_fts5_query(text: str) -> str:
    """Build a FTS5 MATCH expression from query text with exact-token boosts.

    Preserves the original text as the primary match expression.
    Extracted boost tokens are appended as OR clauses with phrase quotes
    so FTS5 treats them as exact-phrase matches rather than tokenized terms.
    This does NOT alter dense candidate ordering.

    Args:
        text: Query text (original or sparse-variant).

    Returns:
        FTS5 MATCH expression string.
    """
    # FTS5 special characters that must be escaped in unquoted terms.
    # We use phrase quoting ("...") for the whole query to avoid injection.
    # However, double-quotes inside the query must be doubled for FTS5.
    escaped = text.replace('"', '""')
    base_expr = f'"{escaped}"'

    boost_tokens = _extract_boost_tokens(text)
    if not boost_tokens:
        return base_expr

    # Append exact token phrases as OR clauses.
    boost_clauses = " OR ".join(f'"{tok.replace(chr(34), chr(34)*2)}"' for tok in boost_tokens)
    return f"({base_expr} OR {boost_clauses})"


class BM25SparseRetriever:
    """Weighted FTS5 BM25 sparse retriever.

    Retrieves candidates via the store's FTS5 index using BM25 scoring.
    Exact-token boosts for error codes, command names, and identifiers are applied
    at query time as OR clauses (see module docstring for current vs. future mechanism).

    Score direction: sparse_score higher = more relevant (BM25, range >= 0).
    Only sparse_score is set on returned hits; other score fields remain None.

    Sparse-only degraded mode is first-class: this retriever has no Embedder
    dependency and requires no downloads or credentials.

    Args:
        store:       SQLiteStore instance (read-only access via retrieve()).
        filter_spec: Optional provider-neutral filter to apply before returning hits.
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        filter_spec: FilterSpec | None = None,
    ) -> None:
        self._store: SQLiteStore = store
        self._filter_spec: FilterSpec = filter_spec if filter_spec is not None else FilterSpec()

    def retrieve(self, query: Query) -> list[Hit]:
        """Return ranked hits using weighted FTS5 BM25 sparse retrieval.

        The original query text is used verbatim for lexical precision.
        Exact-token boosts for technical identifiers are applied without
        altering dense candidate ordering.
        Filters are applied before candidates are returned.

        Args:
            query: Query record with text and optional corpus_id / top_k.

        Returns:
            List of Hit records ordered by sparse_score descending (higher is better).
            Each Hit has sparse_score set; dense_score, fusion_score, and rerank_score
            are None.

        Raises:
            BackendError: On FTS5 index read failure.
        """
        # Build FTS5 query expression with exact-token boosts.
        fts_expr = _build_fts5_query(query.text)

        # Issue retrieval via the store with the boosted expression.
        boosted_query = Query(
            id=query.id,
            text=fts_expr,
            corpus_id=query.corpus_id,
            top_k=query.top_k,
        )
        hits: list[Hit] = self._store.retrieve(boosted_query)

        # Apply provider-neutral filters before returning.
        return apply_filters(hits, self._filter_spec)
