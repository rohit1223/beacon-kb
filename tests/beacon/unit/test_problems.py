"""Tests for the RFC 9457 problem-details helpers and error taxonomy mapping."""

from __future__ import annotations

import json

import pytest

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
from beacon.problems import (
    KIND_TO_STATUS,
    ProblemDetail,
    error_to_problem,
)


class TestErrorTaxonomy:
    """Each error family must carry a stable machine-readable kind."""

    def test_readiness_kind(self) -> None:
        err = ReadinessError("service not ready")
        assert err.kind == BeaconErrorKind.READINESS

    def test_backend_kind(self) -> None:
        err = BackendError("qdrant unreachable")
        assert err.kind == BeaconErrorKind.BACKEND

    def test_ingestion_kind(self) -> None:
        err = IngestionError("parse failed", source_uri="file://doc.pdf")
        assert err.kind == BeaconErrorKind.INGESTION

    def test_citation_kind(self) -> None:
        err = CitationError("invalid citation [S99]")
        assert err.kind == BeaconErrorKind.CITATION

    def test_budget_kind(self) -> None:
        err = BudgetError("token budget exceeded")
        assert err.kind == BeaconErrorKind.BUDGET

    def test_answer_error_kind(self) -> None:
        err = AnswerError("llm refused")
        assert err.kind == BeaconErrorKind.BACKEND

    def test_all_errors_are_beacon_errors(self) -> None:
        errors = [
            ReadinessError("x"),
            BackendError("x"),
            IngestionError("x"),
            CitationError("x"),
            BudgetError("x"),
        ]
        for err in errors:
            assert isinstance(err, BeaconError)

    def test_error_kind_values(self) -> None:
        assert BeaconErrorKind.READINESS.value == "readiness"
        assert BeaconErrorKind.BACKEND.value == "backend"
        assert BeaconErrorKind.INGESTION.value == "ingestion"
        assert BeaconErrorKind.CITATION.value == "citation"
        assert BeaconErrorKind.BUDGET.value == "budget"


class TestKindToStatusMapping:
    """Every kind must map to exactly one HTTP status."""

    def test_readiness_maps_to_503(self) -> None:
        assert KIND_TO_STATUS[BeaconErrorKind.READINESS] == 503

    def test_backend_maps_to_502(self) -> None:
        assert KIND_TO_STATUS[BeaconErrorKind.BACKEND] == 502

    def test_ingestion_maps_to_422(self) -> None:
        assert KIND_TO_STATUS[BeaconErrorKind.INGESTION] == 422

    def test_citation_maps_to_422(self) -> None:
        assert KIND_TO_STATUS[BeaconErrorKind.CITATION] == 422

    def test_budget_maps_to_402(self) -> None:
        assert KIND_TO_STATUS[BeaconErrorKind.BUDGET] == 402

    def test_all_kinds_covered(self) -> None:
        for kind in BeaconErrorKind:
            assert kind in KIND_TO_STATUS, f"Kind {kind!r} missing from KIND_TO_STATUS"


class TestProblemDetail:
    """RFC 9457 problem-details model must contain required fields."""

    def test_required_fields_present(self) -> None:
        problem = ProblemDetail(
            type="https://beacon.example/errors/readiness",
            title="Service Not Ready",
            status=503,
            detail="Qdrant is unreachable",
            kind=BeaconErrorKind.READINESS,
        )
        assert problem.type == "https://beacon.example/errors/readiness"
        assert problem.title == "Service Not Ready"
        assert problem.status == 503
        assert problem.detail == "Qdrant is unreachable"
        assert problem.kind == BeaconErrorKind.READINESS

    def test_instance_optional(self) -> None:
        problem = ProblemDetail(
            type="https://beacon.example/errors/budget",
            title="Budget Exceeded",
            status=402,
            detail="Token budget exceeded",
            kind=BeaconErrorKind.BUDGET,
        )
        assert problem.instance is None

    def test_instance_can_be_set(self) -> None:
        problem = ProblemDetail(
            type="https://beacon.example/errors/budget",
            title="Budget Exceeded",
            status=402,
            detail="Token budget exceeded",
            kind=BeaconErrorKind.BUDGET,
            instance="/requests/abc123",
        )
        assert problem.instance == "/requests/abc123"

    def test_json_round_trip(self) -> None:
        problem = ProblemDetail(
            type="https://beacon.example/errors/backend",
            title="Backend Error",
            status=502,
            detail="Qdrant unreachable",
            kind=BeaconErrorKind.BACKEND,
            instance="/collections/docs/sync",
        )
        serialized = problem.model_dump_json()
        data = json.loads(serialized)
        assert data["type"] == "https://beacon.example/errors/backend"
        assert data["title"] == "Backend Error"
        assert data["status"] == 502
        assert data["detail"] == "Qdrant unreachable"
        assert data["kind"] == "backend"
        assert data["instance"] == "/collections/docs/sync"

    def test_is_frozen(self) -> None:
        from pydantic import ValidationError

        problem = ProblemDetail(
            type="https://beacon.example/errors/backend",
            title="Backend Error",
            status=502,
            detail="Qdrant unreachable",
            kind=BeaconErrorKind.BACKEND,
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            problem.status = 500  # assignment to frozen model raises at runtime


class TestErrorToProblem:
    """error_to_problem must convert any BeaconError to a ProblemDetail."""

    def test_readiness_error_maps_correctly(self) -> None:
        err = ReadinessError("Qdrant is unreachable")
        problem = error_to_problem(err)
        assert problem.status == 503
        assert problem.kind == BeaconErrorKind.READINESS
        assert "Qdrant is unreachable" in problem.detail

    def test_backend_error_maps_correctly(self) -> None:
        err = BackendError("connection refused")
        problem = error_to_problem(err)
        assert problem.status == 502
        assert problem.kind == BeaconErrorKind.BACKEND

    def test_ingestion_error_maps_correctly(self) -> None:
        err = IngestionError("corrupt PDF", source_uri="file://bad.pdf")
        problem = error_to_problem(err)
        assert problem.status == 422
        assert problem.kind == BeaconErrorKind.INGESTION

    def test_citation_error_maps_correctly(self) -> None:
        err = CitationError("label [S99] references missing evidence")
        problem = error_to_problem(err)
        assert problem.status == 422
        assert problem.kind == BeaconErrorKind.CITATION

    def test_budget_error_maps_correctly(self) -> None:
        err = BudgetError("max cost 0.05 USD exceeded")
        problem = error_to_problem(err)
        assert problem.status == 402
        assert problem.kind == BeaconErrorKind.BUDGET

    def test_with_instance(self) -> None:
        err = ReadinessError("not ready")
        problem = error_to_problem(err, instance="/healthz")
        assert problem.instance == "/healthz"

    def test_problem_type_contains_kind(self) -> None:
        err = BackendError("fail")
        problem = error_to_problem(err)
        assert "backend" in problem.type

    def test_problem_fields_all_present(self) -> None:
        err = BudgetError("too expensive")
        problem = error_to_problem(err)
        assert problem.type
        assert problem.title
        assert problem.status
        assert problem.detail
        assert problem.kind is not None
