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

Stopword handling (C2 fix):
  Common English function words (how, do, I, what, is, the, ...) are removed from
  the AND-required token set so queries like "how do I install the parser" and
  "What is the capital of France?" are not forced to match all stopwords verbatim.
  Stopwords are defined in _STOPWORDS (a module-level frozenset, ~30 words).
  If ALL tokens are stopwords, the full original token set is kept to avoid an
  empty query.

OR-fallback for zero-hit AND queries:
  If the AND-over-content-words query returns zero rows, the retriever
  automatically re-runs the same query with OR between all non-stopword tokens.
  BM25 still ranks the OR results.  This maximises recall for stopword-heavy
  natural-language questions while keeping the default AND precision for queries
  that contain meaningful content words.

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
# English stopword list (curated, ~30 common function words).
# These words carry very little retrieval signal on their own, so they are
# excluded from the AND-required token set to avoid zero-recall on
# natural-language questions.  All comparisons are case-insensitive (tokens
# are lowercased before lookup).
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "how", "do", "i", "what", "is", "the", "a", "an", "of", "to",
    "in", "for", "on", "at", "my", "me", "can", "does", "are", "was",
    "were", "be", "it", "this", "that", "and", "or", "with", "from", "by",
})


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


def _content_tokens(raw_tokens: list[str]) -> list[str]:
    """Return the subset of tokens that are not English stopwords.

    Comparison is case-insensitive: 'How', 'HOW', and 'how' all match the
    stopword.  If every token is a stopword, the full list is returned as-is
    to avoid producing an empty query.

    Args:
        raw_tokens: Whitespace-split tokens from the original query text.

    Returns:
        Non-stopword tokens, or the original list when all are stopwords.
    """
    filtered = [t for t in raw_tokens if t.lower() not in _STOPWORDS]
    return filtered if filtered else raw_tokens


def _build_fts5_query(text: str) -> str:
    """Build a FTS5 MATCH expression from query text with exact-token boosts.

    Base expression: each non-stopword whitespace-delimited token is
    individually quoted so FTS5 applies its own tokenizer to each term.
    The implicit conjunction (AND semantics) between terms matches documents
    that contain ALL non-stopword query tokens regardless of word order or
    exact phrasing.  Common English function words (how, do, I, what, is, the,
    ...) are excluded from the AND set so queries like "how do I install the
    parser" do not return 0 hits because the document text lacks those words.
    If all tokens are stopwords, they are all included to avoid an empty query.

    AND was chosen over OR for precision: querying N content tokens AND-style
    retrieves documents that are semantically more aligned to ALL aspects of
    the query.  For recall-sensitive cases, an OR-fallback is applied in
    BM25SparseRetriever.retrieve() when the AND query yields zero rows.

    Boost tokens (error codes, commands, identifiers) are appended as OR
    clauses using phrase quotes for exact-match boosting; these are additive
    to column weighting and do not change the base AND expression.

    Args:
        text: Query text (original or sparse-variant).

    Returns:
        FTS5 MATCH expression string (AND over content words).
    """
    tokens = text.split()
    if not tokens:
        # Degenerate empty query: return a safe always-false expression.
        return '""'

    # Drop stopwords from the AND-required set (fallback keeps all if all stopwords).
    content_toks = _content_tokens(tokens)

    # Build per-token phrase terms - implicit AND between space-separated terms.
    token_terms = " ".join(
        f'"{tok.replace(chr(34), chr(34) * 2)}"' for tok in content_toks
    )

    boost_tokens = _extract_boost_tokens(text)
    if not boost_tokens:
        return token_terms

    # Append exact technical identifier phrases as OR clauses (additive boost).
    boost_clauses = " OR ".join(
        f'"{tok.replace(chr(34), chr(34) * 2)}"' for tok in boost_tokens
    )
    return f"({token_terms} OR {boost_clauses})"


def _build_fts5_or_query(text: str) -> str:
    """Build a FTS5 MATCH expression using OR between all non-stopword tokens.

    Used as a fallback when the primary AND query returns zero rows.  OR
    semantics maximise recall; BM25 still ranks the results so the most
    relevant documents surface first.

    Args:
        text: Query text (original or sparse-variant).

    Returns:
        FTS5 MATCH expression string (OR over content words).
    """
    tokens = text.split()
    if not tokens:
        return '""'
    content_toks = _content_tokens(tokens)
    return " OR ".join(
        f'"{tok.replace(chr(34), chr(34) * 2)}"' for tok in content_toks
    )


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
        Stopwords are dropped from the AND-required set (see module docstring).
        If the AND query yields zero rows, an OR-fallback over content tokens
        is issued automatically (OR-fallback, BM25 still ranks).
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
        # Build FTS5 AND query expression (stopwords dropped, exact-token boosts added).
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

        # OR-fallback: if AND over content words returned nothing, re-run with OR.
        if not hits:
            or_expr = _build_fts5_or_query(query.text)
            or_query = Query(
                id=query.id,
                text=or_expr,
                corpus_id=query.corpus_id,
                top_k=query.top_k,
            )
            hits = self._store.retrieve(or_query, weights=self._column_weights)

        # Apply provider-neutral filters before returning.
        return apply_filters(hits, self._filter_spec)
