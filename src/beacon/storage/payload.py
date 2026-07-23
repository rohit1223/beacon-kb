"""Typed payload schema stored with every Qdrant point.

Every chunk upserted into a Qdrant collection carries this payload so that
retrieval, filtering, and citation can work off stable named fields.

Named-vector constants
----------------------
``DENSE_VECTOR_NAME`` and ``SPARSE_VECTOR_NAME`` are the keys used when
creating collections with named vectors.  Epic 02 writes under these names;
Epic 03 queries under them.  They are declared here as the single source of
truth so all layers stay in sync without magic strings.

Payload index declarations
--------------------------
``PAYLOAD_INDEX_FIELDS`` is a list of ``(field_name, PayloadSchemaType)``
pairs.  The store layer creates these indexes at collection-creation time so
that ``source_uri``, ``tags``, and date fields are filterable without full
scans.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client.http.models import PayloadSchemaType

# ---------------------------------------------------------------------------
# Chunk-to-point ID conversion
# ---------------------------------------------------------------------------


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Convert a 64-char hex chunk_id to a Qdrant point ID (UUID string).

    The UUID is derived from the first 32 hex characters (128 bits) of the
    SHA-256 chunk identifier.  This conversion is the single source of truth
    shared by the sync engine (write path, Epic 02) and the retrieval layer
    (read path, Epic 03): both sides must derive identical point IDs for the
    same chunk so reads and carryover copies address the points the engine
    wrote.

    Args:
        chunk_id: 64-character hex SHA-256 chunk identifier.

    Returns:
        UUID string derived from the first 32 hex chars.
    """
    return str(uuid.UUID(chunk_id[:32]))

# ---------------------------------------------------------------------------
# Named-vector constants
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME: str = "dense"
"""Name of the dense (HNSW) vector in every collection."""

SPARSE_VECTOR_NAME: str = "sparse"
"""Name of the sparse (BM25/SPLADE) vector in every collection."""

# ---------------------------------------------------------------------------
# Payload index declarations
# ---------------------------------------------------------------------------

PAYLOAD_INDEX_FIELDS: list[tuple[str, PayloadSchemaType]] = [
    ("source_uri", PayloadSchemaType.KEYWORD),
    ("tags", PayloadSchemaType.KEYWORD),
    ("kind", PayloadSchemaType.KEYWORD),
    ("ingested_at", PayloadSchemaType.DATETIME),
    ("created_at", PayloadSchemaType.DATETIME),
    ("modified_at", PayloadSchemaType.DATETIME),
    ("content_hash", PayloadSchemaType.KEYWORD),
    ("chunk_hash", PayloadSchemaType.KEYWORD),
]
"""Filterable payload fields; declared at collection creation time."""

# ---------------------------------------------------------------------------
# Typed payload dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkPayload:
    """Typed payload stored alongside every Qdrant point.

    All fields map 1-to-1 with Qdrant payload keys so ``to_dict()`` produces
    the exact dict passed to ``PointStruct.payload``.

    Date fields (``created_at``, ``modified_at``) are ``None`` when the
    source document does not provide them.  ``ingested_at`` is always set by
    the ingestion pipeline.

    ``parent_chunk_id`` is ``None`` for top-level chunks and carries the
    ``str`` UUID of the parent for child chunks (LlamaIndex hierarchical
    nodes).

    ``kind`` and ``section_kind`` store the string values of ``ChunkKind``
    and ``SectionKind`` respectively (e.g. ``"parent"``/``"child"`` and
    ``"text"``/``"code"``/``"table"``/``"list"``).

    ``prev_chunk_id`` and ``next_chunk_id`` carry neighbor links within the
    same section; both are ``None`` for parents and for terminal children.
    """

    chunk_text: str
    source_uri: str
    title: str
    heading_path: list[str]
    tags: list[str]
    ingested_at: str
    content_hash: str
    chunk_hash: str
    fingerprint: str
    kind: str
    section_kind: str
    created_at: str | None = None
    modified_at: str | None = None
    parent_chunk_id: str | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for use as a Qdrant point payload."""
        return {
            "chunk_text": self.chunk_text,
            "source_uri": self.source_uri,
            "title": self.title,
            "heading_path": self.heading_path,
            "tags": self.tags,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "ingested_at": self.ingested_at,
            "content_hash": self.content_hash,
            "chunk_hash": self.chunk_hash,
            "parent_chunk_id": self.parent_chunk_id,
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "section_kind": self.section_kind,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }
