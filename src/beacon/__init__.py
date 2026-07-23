"""Beacon - industry-standard knowledge-base RAG server.

Public surface exported from this module is intentionally minimal.
Import submodules directly for the full API.
"""

from __future__ import annotations

__all__ = [
    "AnswerError",
    "BackendError",
    "BeaconError",
    "BeaconErrorKind",
    "BeaconSettings",
    "BudgetError",
    "CitationError",
    "IngestionError",
    "ProblemDetail",
    "ReadinessError",
    "error_to_problem",
]

from beacon.config import BeaconSettings
from beacon.errors import (
    AnswerError,
    BackendError,
    BeaconError,
    BeaconErrorKind,
    BudgetError,
    CitationError,
    IngestionError,
    ReadinessError,
)
from beacon.problems import ProblemDetail, error_to_problem
