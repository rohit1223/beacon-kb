"""Beacon server configuration.

All settings are driven by environment variables under the ``BEACON_`` prefix.
Nested sections use ``__`` as the delimiter (e.g. ``BEACON_SERVER__PORT=9000``).
Secret fields use pydantic ``SecretStr`` so their values are never included in
``repr`` output or any model dump used for logging.

Use ``BeaconSettings.safe_dump()`` when you need a dict safe to pass to a
structured logger - it serializes secrets as ``"**REDACTED**"`` strings.

Local-first defaults mean that ``beacon serve`` works with no credentials and
no remote services configured: Qdrant runs embedded from a local path, auth is
disabled, and the default embedding model requires no API key.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Section models
#
# IMPORTANT: Section classes must be plain ``pydantic.BaseModel``, NOT
# ``BaseSettings``.  If sections inherit ``BaseSettings`` each one
# independently scans the environment for its bare field names, so an ambient
# ``PORT=80`` would override ``server.port`` and the shell ``$PATH`` would
# land in ``qdrant.path``.  Only the root ``BeaconSettings`` class is a
# ``BaseSettings`` instance; it owns the ``BEACON_`` prefix and ``__`` nested
# delimiter and populates the section models itself.
# ---------------------------------------------------------------------------


class ServerSettings(BaseModel):
    """HTTP server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    api_key: SecretStr | None = None
    """When set, non-localhost requests must present this key as a Bearer token."""

    def __repr__(self) -> str:
        return (
            f"ServerSettings(host={self.host!r}, port={self.port!r}, "
            f"workers={self.workers!r}, api_key={'**REDACTED**' if self.api_key else None!r})"
        )


class QdrantSettings(BaseModel):
    """Qdrant vector store configuration.

    When ``url`` is ``None`` the store runs in embedded local mode using
    ``path`` as the on-disk location.
    When ``url`` is set the store targets the given Qdrant server.

    ``path`` defaults to ``"data/qdrant"`` - a relative path resolved against
    the process working directory at construction time.
    Use an absolute path or ``BEACON_QDRANT__PATH`` to anchor it explicitly.
    """

    url: str | None = None
    path: str = Field(default_factory=lambda: "data/qdrant")
    api_key: SecretStr | None = None
    # collection_prefix removed: physical collection naming uses the revision ID
    # directly (e.g. __rev_<id> shadow names) and does not rely on a prefix field.
    timeout: float = 10.0

    def __repr__(self) -> str:
        return (
            f"QdrantSettings(url={self.url!r}, path={self.path!r}, "
            f"api_key={'**REDACTED**' if self.api_key else None!r}, "
            f"timeout={self.timeout!r})"
        )


class StateSettings(BaseModel):
    """SQLite state database configuration.

    ``db_path`` defaults to ``"data/beacon.db"`` - a relative path resolved
    against the process working directory at construction time.
    """

    db_path: str = Field(default_factory=lambda: "data/beacon.db")


class ModelsSettings(BaseModel):
    """LLM and embedding model configuration."""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    """Local sentence-transformers model used when no cloud API key is present."""

    llm_model: str = "gpt-4o-mini"
    """LiteLLM model name for the answer pipeline."""

    llm_api_key: SecretStr | None = None
    """API key for the LLM provider (OpenAI, Anthropic, etc.)."""

    embedding_dimension: int = 384
    """Dimensionality of the dense embedding vectors."""

    def __repr__(self) -> str:
        return (
            f"ModelsSettings(embedding_model={self.embedding_model!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_api_key={'**REDACTED**' if self.llm_api_key else None!r}, "
            f"embedding_dimension={self.embedding_dimension!r})"
        )


class RetrievalSettings(BaseModel):
    """Retrieval pipeline configuration."""

    top_k: int = 10
    """Number of results to return from the vector store."""

    rerank: bool = False
    """Whether to apply cross-encoder reranking."""

    parent_expansion: bool = True
    """Whether to expand retrieved child chunks to their parent context."""

    score_threshold: float = 0.0
    """Minimum relevance score; results below this are discarded."""


class AnswerSettings(BaseModel):
    """Answer pipeline configuration."""

    max_tokens: int = 2048
    """Maximum tokens for the LLM answer response."""

    temperature: float = 0.0
    """LLM temperature; 0 for deterministic outputs."""

    abstain_when_uncertain: bool = True
    """If True, the pipeline abstains rather than hallucinating low-confidence answers."""


class InvestigateSettings(BaseModel):
    """Investigate pipeline (LangGraph agentic loop) configuration."""

    max_iterations: int = 5
    """Maximum plan/retrieve/grade/reflect iterations before forced synthesis."""

    max_cost_usd: float = 0.05
    """Hard budget ceiling in USD; the loop stops and raises BudgetError if exceeded."""

    enable_checkpointing: bool = True
    """Whether to persist loop state to the SQLite state DB for resume on failure."""


class IngestSettings(BaseModel):
    """Ingestion pipeline configuration."""

    data_dir: str = Field(default_factory=lambda: "data")
    """Root directory for connector data and uploaded files.

    Uploaded documents are stored under ``<data_dir>/uploads/<hash_prefix>/<hash>/``.
    """

    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MiB
    """Maximum allowed upload size in bytes. Requests exceeding this are
    rejected with a 413 problem-details response without buffering the body.
    """


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class BeaconSettings(BaseSettings):
    """Root configuration tree for the Beacon server.

    Instantiate once at startup and pass the instance through dependency
    injection; do not construct global singletons at module import time.

    All fields are overridable via ``BEACON_``-prefixed environment variables.
    Nested sections use the ``__`` delimiter:
    ``BEACON_SERVER__PORT=9000``, ``BEACON_QDRANT__URL=http://qdrant:6333``.

    A ``.env`` file in the working directory is loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="BEACON_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    investigate: InvestigateSettings = Field(default_factory=InvestigateSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)

    def __repr__(self) -> str:
        return (
            f"BeaconSettings("
            f"server={self.server!r}, "
            f"qdrant={self.qdrant!r}, "
            f"state={self.state!r}, "
            f"models={self.models!r}, "
            f"retrieval={self.retrieval!r}, "
            f"answer={self.answer!r}, "
            f"investigate={self.investigate!r}, "
            f"ingest={self.ingest!r})"
        )

    def safe_dump(self) -> dict[str, Any]:
        """Return a log-safe dict with all secret values redacted.

        Secrets appear as the string ``"**REDACTED**"`` so structured loggers
        can capture the full settings tree without leaking credentials.
        """
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "api_key": "**REDACTED**" if self.server.api_key else None,
            },
            "qdrant": {
                "url": self.qdrant.url,
                "path": self.qdrant.path,
                "api_key": "**REDACTED**" if self.qdrant.api_key else None,
                "timeout": self.qdrant.timeout,
            },
            "state": {
                "db_path": self.state.db_path,
            },
            "models": {
                "embedding_model": self.models.embedding_model,
                "llm_model": self.models.llm_model,
                "llm_api_key": "**REDACTED**" if self.models.llm_api_key else None,
                "embedding_dimension": self.models.embedding_dimension,
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "rerank": self.retrieval.rerank,
                "parent_expansion": self.retrieval.parent_expansion,
                "score_threshold": self.retrieval.score_threshold,
            },
            "answer": {
                "max_tokens": self.answer.max_tokens,
                "temperature": self.answer.temperature,
                "abstain_when_uncertain": self.answer.abstain_when_uncertain,
            },
            "investigate": {
                "max_iterations": self.investigate.max_iterations,
                "max_cost_usd": self.investigate.max_cost_usd,
                "enable_checkpointing": self.investigate.enable_checkpointing,
            },
            "ingest": {
                "data_dir": self.ingest.data_dir,
                "max_upload_bytes": self.ingest.max_upload_bytes,
            },
        }
