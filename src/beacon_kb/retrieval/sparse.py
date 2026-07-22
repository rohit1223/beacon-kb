"""Weighted FTS5 BM25 sparse retrieval with per-column weights and exact-token boosts.

Design rules enforced here:
- BM25 scores are explicit and direction-declared: higher sparse_score = more relevant.
- Only sparse_score is set on returned hits; dense_score, fusion_score, and
  rerank_score remain None.
- Per-column bm25() weights are applied via the store's weighted retrieve() API
  (Epic 02 migration 0002 added the multi-column chunks_fts schema with separate
  text, heading, and code columns).
- Exact-token OR-boost clauses are RETAINED as a complementary mechanism for
  technical identifiers (error codes, command names, CamelCase identifiers)
  because they surface exact-match precision on top of what column weighting
  alone provides.  Column weighting and OR-boosts are additive: column weights
  handle structural importance (heading > code > text), OR-boosts handle exact
  technical term anchoring.  Both are needed for good recall + precision.
- Filters are applied consistently before candidates leave this retriever.
- Sparse-only mode is first-class: no embedder dependency anywhere in this module.
- BackendError from the underlying store propagates typed; never swallowed.

Per-column bm25() weighting adoption (Epic 03 obligation, ROADMAP item done):
  Epic 02 migration 0002 extended chunks_fts to a multi-column schema (heading,
  code, text).  This module now calls store.retrieve(query, weights=...) with
  sensible column defaults:
    - heading weight: 10.0 (section heading path carries high structural signal)
    - code weight:    5.0  (identifier + API names in fenced blocks matter)
    - text weight:    1.0  (body text is the baseline)
  These weights are the BM25 column importances passed as (w_text, w_heading,
  w_code) to the store's retrieve() method, which maps them to the bm25() call
  as bm25(chunks_fts, 0.0, 0.0, w_text, w_heading, w_code) per migration 0002's
  column order (chunk_id, corpus_id, text, heading, code).

  The exact-token OR-boost approach from Epic 02 is KEPT because it still adds
  measurable recall value for technical identifiers that appear verbatim in the
  query: error codes like ERROR_CODE_404, hyphenated commands, and CamelCase
  class names benefit from exact-phrase matching on top of column-weighted BM25.
  Dropping OR-boosts entirely would reduce precision for those token types.
  The two mechanisms are orthogonal and work together.

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
# Per-column BM25 weight defaults
# Rationale: heading (parent_locator) is the highest-signal structural field
# because users often search for section names or heading keywords; code content
# (fenced blocks) carries identifier and API names that are strong exact signals;
# body text is the baseline.
# ---------------------------------------------------------------------------

_DEFAULT_TEXT_WEIGHT: float = 1.0
_DEFAULT_HEADING_WEIGHT: float = 10.0
_DEFAULT_CODE_WEIGHT: float = 5.0

# Default weights tuple passed to store.retrieve().
_DEFAULT_WEIGHTS: tuple[float, float, float] = (
    _DEFAULT_TEXT_WEIGHT,
    _DEFAULT_HEADING_WEIGHT,
    _DEFAULT_CODE_WEIGHT,
)


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

    This mechanism is retained on top of per-column bm25() weighting because
    it provides additive precision for technical identifiers that appear verbatim
    in the query.  See module docstring for the full design rationale.

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
    """Weighted FTS5 BM25 sparse retriever with per-column weights.

    Retrieves candidates via the store's FTS5 index using BM25 scoring with
    per-column weights (heading > code > text, see module docstring for rationale).
    Exact-token boosts for error codes, command names, and identifiers are also
    applied as OR clauses (complementary to column weighting, not replacing it).

    Score direction: sparse_score higher = more relevant (BM25, range >= 0).
    Only sparse_score is set on returned hits; other score fields remain None.

    Sparse-only degraded mode is first-class: this retriever has no Embedder
    dependency and requires no downloads or credentials.

    Args:
        store:         SQLiteStore instance (read-only access via retrieve()).
        filter_spec:   Optional provider-neutral filter to apply before returning hits.
        column_weights: Optional (w_text, w_heading, w_code) tuple overriding the
                        default column weights.  None uses the module defaults
                        (text=1.0, heading=10.0, code=5.0).
    """

    def __init__(
        self,
        *,
        store: SQLiteStore,
        filter_spec: FilterSpec | None = None,
        column_weights: tuple[float, float, float] | None = None,
    ) -> None:
        self._store: SQLiteStore = store
        self._filter_spec: FilterSpec = filter_spec if filter_spec is not None else FilterSpec()
        self._column_weights: tuple[float, float, float] = (
            column_weights if column_weights is not None else _DEFAULT_WEIGHTS
        )

    def retrieve(self, query: Query) -> list[Hit]:
        """Return ranked hits using weighted FTS5 BM25 sparse retrieval.

        Uses per-column bm25() weights via store.retrieve(weights=...) to boost
        heading matches over body text and code over prose.  Exact-token OR-boosts
        for technical identifiers are also applied for additive precision.
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
        # Build FTS5 query expression with exact-token boosts (retained, see module docstring).
        fts_expr = _build_fts5_query(query.text)

        # Issue retrieval via the store with per-column bm25() weights and the
        # boosted FTS5 MATCH expression.
        boosted_query = Query(
            id=query.id,
            text=fts_expr,
            corpus_id=query.corpus_id,
            top_k=query.top_k,
        )
        hits: list[Hit] = self._store.retrieve(boosted_query, weights=self._column_weights)

        # Apply provider-neutral filters before returning.
        return apply_filters(hits, self._filter_spec)
