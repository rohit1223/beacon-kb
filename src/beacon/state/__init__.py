"""State DB surface for the Beacon server.

This package provides the SQLite-backed bookkeeping layer for the new Beacon
server.  Chunk data lives in Qdrant; only bookkeeping rows (collections,
sources, revisions, sync jobs) live here.

Typical usage::

    from beacon.state.db import StateDB
    from beacon.state.repo import CollectionRepo, SourceRepo, RevisionRepo, SyncJobRepo
    from beacon.state.repo import derive_corpus_state, CorpusState

    db = StateDB(db_path="data/beacon.db")
    collections = CollectionRepo(db)
    jobs = SyncJobRepo(db)
"""

from beacon.state.db import StateDB
from beacon.state.repo import (
    CollectionRepo,
    CorpusState,
    RevisionRepo,
    RevisionStatus,
    SourceRepo,
    SourceStatus,
    SyncJobRepo,
    SyncJobState,
    derive_corpus_state,
)

__all__ = [
    "CollectionRepo",
    "CorpusState",
    "RevisionRepo",
    "RevisionStatus",
    "SourceRepo",
    "SourceStatus",
    "StateDB",
    "SyncJobRepo",
    "SyncJobState",
    "derive_corpus_state",
]
