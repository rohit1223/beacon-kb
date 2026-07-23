"""Hierarchical chunker: ParsedDocument sections to Chunk records (Task 02.4).

Architecture
------------
This module converts structured ``ParsedDocument`` sections into a flat list
of typed ``Chunk`` records carrying deterministic content-addressed IDs,
parent/child relationships, heading paths, and neighbor links.

**LlamaIndex role (text splitting only).**
``HierarchicalNodeParser`` with two ``SentenceSplitter`` levels (parent size
and child size) is used exclusively to split section text into overlapping
chunks.  Its random node UUIDs are *discarded*; every chunk ID is recomputed
as a SHA-256 digest over stable inputs (see ``_make_chunk_id``).

**Beacon-native logic.**
- SHA-256 deterministic chunk identity (no random IDs, no hash() salt).
- Parent chunk records (``ChunkKind.PARENT``) and child chunk records
  (``ChunkKind.CHILD``): every child carries ``parent_chunk_id`` pointing
  to the parent whose text it sub-divides.
- Neighbor links (``prev_chunk_id``, ``next_chunk_id``) are threaded only
  after all child chunk IDs are stable, within each section only.
- ``chunker_config`` canonical string: encodes the LlamaIndex version plus
  all chunker parameters so that any parameter change invalidates previously
  computed fingerprints (Task 02.5 consumption point).
- Empty sections and blank documents produce zero chunks without error.
- CODE sections are split at line boundaries only; a section whose text fits
  the parent budget is emitted as a single parent+child pair.

**Parent vs child storage.**
Parent chunks are stored in Qdrant as context carriers: their payload has
``parent_chunk_id = None`` (they are roots) and ``kind = ChunkKind.PARENT``.
The retrieval layer expands from a child hit to the parent for richer context.
Child chunks are the units actually ranked by similarity search.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass
from enum import StrEnum

from llama_index.core.node_parser.relational.hierarchical import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode

from beacon.ingest.parsing import ParsedDocument, SectionKind
from beacon.storage.payload import ChunkPayload

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

#: Adapter version - bump whenever the chunking mapping logic changes.
_ADAPTER_VERSION = 1

_ver_parts = importlib.metadata.version("llama-index-core").split(".")
_LLAMA_INDEX_MAJOR_MINOR = f"{_ver_parts[0]}.{_ver_parts[1]}"

#: Published chunker version string, included in chunker_config so that a
#: llama-index-core version upgrade invalidates old fingerprints.
CHUNKER_VERSION = f"llama-index-{_LLAMA_INDEX_MAJOR_MINOR}.beacon-chunker-{_ADAPTER_VERSION}"

# ---------------------------------------------------------------------------
# Chunk data model
# ---------------------------------------------------------------------------


class ChunkKind(StrEnum):
    """Hierarchy role of a chunk."""

    PARENT = "parent"
    """Parent chunk: broader context carrier; not ranked directly."""

    CHILD = "child"
    """Child chunk: smallest searchable unit; links to exactly one parent."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One typed chunk record produced by the hierarchical chunker.

    Attributes:
        chunk_id:        Deterministic 64-char hex SHA-256 digest over stable
                         inputs (collection, canonical_uri, content_hash,
                         chunker_config, section_locator, kind, ordinal).
        parent_chunk_id: ``chunk_id`` of the parent for ``ChunkKind.CHILD``
                         chunks; ``None`` for ``ChunkKind.PARENT`` chunks.
        kind:            PARENT or CHILD.
        text:            Full chunk text (original case preserved).
        heading_path:    Section heading path tuple, e.g. ``("Guide", "Install")``.
        section_kind:    Section content kind (TEXT/CODE/TABLE/LIST) from parsing.
        section_locator: Unique slash-delimited locator from the source section.
        ordinal:         Zero-based position of this chunk among same-kind siblings
                         within the same section.
        prev_chunk_id:   ``chunk_id`` of the preceding CHILD chunk in document
                         order (same section), or ``None`` for the first child.
        next_chunk_id:   ``chunk_id`` of the following CHILD chunk in document
                         order (same section), or ``None`` for the last child.
        token_count:     Approximate token count (via the LlamaIndex tokenizer).
    """

    chunk_id: str
    parent_chunk_id: str | None
    kind: ChunkKind
    text: str
    heading_path: tuple[str, ...]
    section_kind: SectionKind
    section_locator: str
    ordinal: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    token_count: int = 0


# ---------------------------------------------------------------------------
# Chunker configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Chunker parameter bundle.

    Attributes:
        parent_chunk_size: Maximum tokens per parent chunk (default 512).
        child_chunk_size:  Maximum tokens per child chunk (default 128).
        chunk_overlap:     Tokens repeated at boundaries between adjacent
                           child chunks (default 20). Must be less than
                           ``child_chunk_size``.
    """

    parent_chunk_size: int = 512
    child_chunk_size: int = 128
    chunk_overlap: int = 20

    def __post_init__(self) -> None:
        if self.chunk_overlap <= 0:
            raise ValueError("ChunkerConfig: chunk_overlap must be positive")
        if self.child_chunk_size <= 0:
            raise ValueError("ChunkerConfig: child_chunk_size must be positive")
        if self.parent_chunk_size <= 0:
            raise ValueError("ChunkerConfig: parent_chunk_size must be positive")
        if self.chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                "ChunkerConfig: chunk_overlap must be less than child_chunk_size"
            )
        if self.child_chunk_size > self.parent_chunk_size:
            raise ValueError(
                "ChunkerConfig: child_chunk_size must be less than or equal to parent_chunk_size"
            )


def chunker_config_str(config: ChunkerConfig) -> str:
    """Return a canonical, stable string encoding the full chunker configuration.

    The string is stable across processes (no hash() salt) and changes
    whenever any parameter or the CHUNKER_VERSION changes.  Task 02.5 folds
    this string into the revision fingerprint so that re-chunking is triggered
    automatically on any configuration change.

    Args:
        config: The chunker configuration to encode.

    Returns:
        A plain ASCII string; safe to embed in log lines and fingerprints.
    """
    return (
        f"v={CHUNKER_VERSION}"
        f",parent={config.parent_chunk_size}"
        f",child={config.child_chunk_size}"
        f",overlap={config.chunk_overlap}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunk_id(
    *,
    collection: str,
    canonical_uri: str,
    content_hash: str,
    config_str: str,
    section_locator: str,
    kind: ChunkKind,
    ordinal: int,
) -> str:
    """Compute a deterministic SHA-256 chunk ID.

    All inputs are stable across processes; the result is a 64-character
    lowercase hex string.  Changing any input changes the ID.

    Args:
        collection:      Logical collection name.
        canonical_uri:   Connector's stable URI for the document.
        content_hash:    Hex SHA-256 of the raw document bytes.
        config_str:      ``chunker_config_str()`` output.
        section_locator: Unique locator of the source section.
        kind:            PARENT or CHILD.
        ordinal:         Position among same-kind peers in this section.

    Returns:
        64-char lowercase hex SHA-256 digest.
    """
    payload = "\x00".join(
        [
            collection,
            canonical_uri,
            content_hash,
            config_str,
            section_locator,
            kind.value,
            str(ordinal),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _count_tokens(text: str) -> int:
    """Count tokens using the LlamaIndex default tokenizer (tiktoken cl100k_base).

    Deterministic, no network access.

    Args:
        text: The text to tokenize.

    Returns:
        Integer token count.
    """
    from llama_index.core.utils import get_tokenizer

    return len(get_tokenizer()(text))


def _split_section_text(
    text: str, config: ChunkerConfig
) -> list[tuple[str, list[str]]]:
    """Split section text into parent and child chunk texts.

    Uses ``HierarchicalNodeParser`` with two levels: parent (larger) and
    child (smaller with overlap).  LlamaIndex node IDs are random and
    discarded; only the text content is used.

    Args:
        text:   Section body text.
        config: Chunker configuration.

    Returns:
        An ordered list of ``(parent_text, child_texts)`` pairs where each
        tuple contains the parent chunk text and its ordered list of child
        chunk texts.  Ordering within each list matches document order.
        Using positional pairs (not a dict) avoids key-collision bugs when
        two parents share identical stripped text.
    """
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[config.parent_chunk_size, config.child_chunk_size],
        chunk_overlap=config.chunk_overlap,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    doc = Document(text=text, doc_id="__section__")
    all_nodes = parser.get_nodes_from_documents([doc])
    leaf_nodes = get_leaf_nodes(all_nodes)

    # All nodes returned by HierarchicalNodeParser are TextNode instances; the
    # type annotation on get_nodes_from_documents uses the wider BaseNode so we
    # narrow explicitly for attribute access.
    text_nodes = [n for n in all_nodes if isinstance(n, TextNode)]
    leaf_text_nodes = [n for n in leaf_nodes if isinstance(n, TextNode)]

    # Identify parent nodes (those with CHILD relationship, no PARENT relationship).
    parent_nodes = [n for n in text_nodes if NodeRelationship.CHILD in n.relationships]
    if not parent_nodes:
        # Text fit entirely in one chunk: no hierarchical split occurred.
        # The leaf nodes are already the only level; treat them all as children
        # of a single synthetic parent that covers the whole text.
        child_texts_all = [n.text.strip() for n in leaf_text_nodes if n.text.strip()]
        return [(text.strip(), child_texts_all)]

    # Map each parent node ID to its children via the PARENT relationship on
    # leaf nodes.  Use an ordered list of (parent_id, children) pairs to
    # preserve document order and avoid key-collision when two parents have
    # identical stripped text.
    parent_id_to_children: dict[str, list[str]] = {n.id_: [] for n in parent_nodes}
    parent_id_order: list[str] = [n.id_ for n in parent_nodes]
    parent_id_to_text: dict[str, str] = {n.id_: n.text.strip() for n in parent_nodes}

    for leaf in leaf_text_nodes:
        parent_rel_raw = leaf.relationships.get(NodeRelationship.PARENT)
        # The relationship value is always RelatedNodeInfo (not a list) for PARENT.
        if not isinstance(parent_rel_raw, RelatedNodeInfo):
            continue
        if parent_rel_raw.node_id in parent_id_to_children:
            leaf_text = leaf.text.strip()
            if leaf_text:
                parent_id_to_children[parent_rel_raw.node_id].append(leaf_text)

    # Build ordered list of (parent_text, child_texts) pairs, skipping empty parents.
    result: list[tuple[str, list[str]]] = []
    for pid in parent_id_order:
        ptext = parent_id_to_text[pid]
        if ptext:
            result.append((ptext, parent_id_to_children[pid]))

    return result


def _split_code_section(
    text: str, config: ChunkerConfig
) -> list[tuple[str, list[str]]]:
    """Split a CODE section into parent+child pairs at line boundaries only.

    Never splits mid-line.  If the full section text fits within
    ``config.parent_chunk_size`` tokens, returns a single
    ``(text, [text])`` pair.  Otherwise accumulates lines until the token
    budget would be exceeded and starts a new parent.

    Each parent's child list contains exactly the parent text itself (one
    child per parent), preserving the invariant that every parent has at
    least one child.

    Args:
        text:   Section body text (already stripped, non-empty).
        config: Chunker configuration.

    Returns:
        An ordered list of ``(parent_text, [child_text])`` pairs.
    """
    from llama_index.core.utils import get_tokenizer

    tokenizer = get_tokenizer()
    lines = text.split("\n")

    result: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer(line)) + 1  # +1 for the newline
        if current_lines and current_tokens + line_tokens > config.parent_chunk_size:
            # Flush current accumulation as a parent+child pair.
            parent_text = "\n".join(current_lines)
            result.append((parent_text, [parent_text]))
            current_lines = [line]
            current_tokens = line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    if current_lines:
        parent_text = "\n".join(current_lines)
        result.append((parent_text, [parent_text]))

    return result


def _apply_neighbor_links(
    section_chunks: list[Chunk],
    section_children: list[Chunk],
) -> list[Chunk]:
    """Thread ``prev_chunk_id`` / ``next_chunk_id`` links within a section.

    Links are computed strictly within ``section_children`` so that the last
    child of one section never links to the first child of another section.

    Args:
        section_chunks:   All chunks (parents + children) for one section.
        section_children: Only the CHILD chunks for that section, in order.

    Returns:
        A new list of chunks with neighbor links applied to CHILD chunks.
        PARENT chunks are returned unchanged.
    """
    if not section_children:
        return list(section_chunks)

    child_id_to_prev: dict[str, str | None] = {}
    child_id_to_next: dict[str, str | None] = {}
    for i, child in enumerate(section_children):
        child_id_to_prev[child.chunk_id] = (
            section_children[i - 1].chunk_id if i > 0 else None
        )
        child_id_to_next[child.chunk_id] = (
            section_children[i + 1].chunk_id if i < len(section_children) - 1 else None
        )

    linked: list[Chunk] = []
    for chunk in section_chunks:
        if chunk.kind == ChunkKind.CHILD:
            linked.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    parent_chunk_id=chunk.parent_chunk_id,
                    kind=chunk.kind,
                    text=chunk.text,
                    heading_path=chunk.heading_path,
                    section_kind=chunk.section_kind,
                    section_locator=chunk.section_locator,
                    ordinal=chunk.ordinal,
                    prev_chunk_id=child_id_to_prev[chunk.chunk_id],
                    next_chunk_id=child_id_to_next[chunk.chunk_id],
                    token_count=chunk.token_count,
                )
            )
        else:
            linked.append(chunk)

    return linked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DocumentChunker:
    """Converts a ``ParsedDocument`` into hierarchical ``Chunk`` records.

    Each section of the document is split independently.  Chunks from
    different sections carry different ``section_locator`` values and
    therefore different IDs.

    The returned list contains both ``ChunkKind.PARENT`` and
    ``ChunkKind.CHILD`` records in document order (parent immediately
    followed by its children).  Neighbor links (``prev_chunk_id``,
    ``next_chunk_id``) connect CHILD chunks within each section only -
    the last child of section N has ``next_chunk_id = None`` and the
    first child of section N+1 has ``prev_chunk_id = None``.

    Args:
        collection:    Logical collection name (feeds chunk ID).
        canonical_uri: Connector's stable URI for the document (feeds chunk ID).
        content_hash:  Hex SHA-256 of the raw document bytes (feeds chunk ID).
        config:        Chunker parameter bundle.
    """

    def __init__(
        self,
        *,
        collection: str,
        canonical_uri: str,
        content_hash: str,
        config: ChunkerConfig,
    ) -> None:
        self._collection = collection
        self._canonical_uri = canonical_uri
        self._content_hash = content_hash
        self._config = config
        self._config_str = chunker_config_str(config)

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk all sections of a ``ParsedDocument``.

        Args:
            doc: The parsed document whose sections are to be chunked.

        Returns:
            Ordered list of ``Chunk`` records (parent+children per section).
            Empty sections and blank documents produce an empty list without
            error.
        """
        all_chunks: list[Chunk] = []

        for section in doc.sections:
            text = (section.text or "").strip()
            if not text:
                continue

            section_chunks = self._chunk_section(
                text=text,
                section_locator=section.locator,
                heading_path=section.heading_path,
                section_kind=section.kind,
            )
            if not section_chunks:
                continue

            # Thread neighbor links within this section only.
            section_children = [c for c in section_chunks if c.kind == ChunkKind.CHILD]
            linked_section = _apply_neighbor_links(section_chunks, section_children)
            all_chunks.extend(linked_section)

        return all_chunks

    def _chunk_section(
        self,
        *,
        text: str,
        section_locator: str,
        heading_path: tuple[str, ...],
        section_kind: SectionKind,
    ) -> list[Chunk]:
        """Chunk a single section's text into parent+child Chunk records.

        For CODE sections, splitting is done at line boundaries only (via
        ``_split_code_section``).  For all other section kinds,
        ``_split_section_text`` is used.

        Args:
            text:             Section body text (already stripped, non-empty).
            section_locator:  Unique locator from the source section.
            heading_path:     Heading path tuple.
            section_kind:     Content kind (text/code/table/list).

        Returns:
            Ordered list of Chunk records: one parent per split parent chunk,
            each followed by its child chunks.
        """
        if section_kind == SectionKind.CODE:
            parent_child_pairs = _split_code_section(text, self._config)
        else:
            parent_child_pairs = _split_section_text(text, self._config)

        result: list[Chunk] = []
        child_ordinal = 0  # Global child ordinal within this section

        for parent_ordinal, (parent_text, child_texts) in enumerate(parent_child_pairs):
            parent_chunk_id = _make_chunk_id(
                collection=self._collection,
                canonical_uri=self._canonical_uri,
                content_hash=self._content_hash,
                config_str=self._config_str,
                section_locator=section_locator,
                kind=ChunkKind.PARENT,
                ordinal=parent_ordinal,
            )
            parent_chunk = Chunk(
                chunk_id=parent_chunk_id,
                parent_chunk_id=None,
                kind=ChunkKind.PARENT,
                text=parent_text,
                heading_path=heading_path,
                section_kind=section_kind,
                section_locator=section_locator,
                ordinal=parent_ordinal,
                token_count=_count_tokens(parent_text),
            )
            result.append(parent_chunk)

            for child_text in child_texts:
                child_chunk_id = _make_chunk_id(
                    collection=self._collection,
                    canonical_uri=self._canonical_uri,
                    content_hash=self._content_hash,
                    config_str=self._config_str,
                    section_locator=section_locator,
                    kind=ChunkKind.CHILD,
                    ordinal=child_ordinal,
                )
                child_chunk = Chunk(
                    chunk_id=child_chunk_id,
                    parent_chunk_id=parent_chunk_id,
                    kind=ChunkKind.CHILD,
                    text=child_text,
                    heading_path=heading_path,
                    section_kind=section_kind,
                    section_locator=section_locator,
                    ordinal=child_ordinal,
                    token_count=_count_tokens(child_text),
                )
                result.append(child_chunk)
                child_ordinal += 1

        return result


# ---------------------------------------------------------------------------
# Payload conversion helper
# ---------------------------------------------------------------------------


def chunk_to_payload(
    chunk: Chunk,
    *,
    source_uri: str,
    title: str,
    tags: list[str],
    ingested_at: str,
    content_hash: str,
    fingerprint: str,
    created_at: str | None = None,
    modified_at: str | None = None,
) -> ChunkPayload:
    """Convert a ``Chunk`` to a ``ChunkPayload`` for Qdrant point storage.

    All retrieval-relevant fields are mapped losslessly.  ``chunk_hash`` is
    set to ``chunk.chunk_id`` (the same deterministic SHA-256 digest) so that
    the payload index is queryable by the stable chunk identity.

    Args:
        chunk:        The chunk record to convert.
        source_uri:   Canonical URI of the source document.
        title:        Document title.
        tags:         Filterable tag list.
        ingested_at:  UTC ISO 8601 ingestion timestamp.
        content_hash: Hex SHA-256 of the raw document bytes.
        fingerprint:  Pipeline revision fingerprint (Task 02.5 output).
        created_at:   Document creation date, or ``None``.
        modified_at:  Document modification date, or ``None``.

    Returns:
        ``ChunkPayload`` instance ready for ``PointStruct.payload``.
    """
    return ChunkPayload(
        chunk_text=chunk.text,
        source_uri=source_uri,
        title=title,
        heading_path=list(chunk.heading_path),
        tags=tags,
        ingested_at=ingested_at,
        content_hash=content_hash,
        chunk_hash=chunk.chunk_id,
        fingerprint=fingerprint,
        kind=chunk.kind.value,
        section_kind=chunk.section_kind.value,
        created_at=created_at,
        modified_at=modified_at,
        parent_chunk_id=chunk.parent_chunk_id,
        prev_chunk_id=chunk.prev_chunk_id,
        next_chunk_id=chunk.next_chunk_id,
    )
