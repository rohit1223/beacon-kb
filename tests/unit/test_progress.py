"""Unit tests for progress.py."""
from __future__ import annotations

from typing import Any

import pytest

from beacon_kb.progress import (
    LoggingProgressAdapter,
    NullProgressObserver,
    StageEvent,
    make_stage_emitter,
)
from beacon_kb.testing import FakeClock, FakeProgressObserver


def test_stage_event_fields() -> None:
    evt = StageEvent(stage="chunk", status="start", current=0, total=10, elapsed_seconds=0.0)
    assert evt.stage == "chunk"
    assert evt.status == "start"
    assert evt.current == 0
    assert evt.total == 10
    assert evt.elapsed_seconds == 0.0
    assert evt.message == ""


def test_stage_event_as_dict_has_required_keys() -> None:
    evt = StageEvent(stage="embed", status="end", current=5, total=5, elapsed_seconds=1.5)
    d = evt.as_dict()
    assert d["stage"] == "embed"
    assert d["status"] == "end"
    assert d["current"] == 5
    assert d["total"] == 5
    assert d["elapsed_seconds"] == 1.5


def test_null_observer_does_not_raise() -> None:
    obs = NullProgressObserver()
    obs.on_event({"stage": "chunk", "status": "start"})
    obs.on_event({})


def test_logging_adapter_does_not_raise(caplog: Any) -> None:
    inner = FakeProgressObserver()
    clock = FakeClock(start=0.0)
    adapter = LoggingProgressAdapter(observer=inner, clock=clock)
    adapter.on_event({"stage": "embed", "status": "progress", "current": 3, "total": 10})
    assert len(inner.events) == 1


def test_logging_adapter_swallows_internal_errors() -> None:
    """on_event must never raise even if the inner observer raises."""

    class RaisingObserver:
        def on_event(self, event: dict[str, Any]) -> None:
            raise RuntimeError("boom")

    adapter = LoggingProgressAdapter(observer=RaisingObserver(), clock=FakeClock())
    adapter.on_event({"stage": "x", "status": "start"})  # Must not raise


def test_make_stage_emitter_emits_start_end(monkeypatch: Any) -> None:
    obs = FakeProgressObserver()
    clock = FakeClock(start=0.0)

    with make_stage_emitter("chunk", observer=obs, clock=clock, total=5) as emit:
        clock.tick(2.0)
        emit(current=3, message="halfway")

    statuses = [e["status"] for e in obs.events]
    assert "start" in statuses
    assert "end" in statuses
    progress_events = [e for e in obs.events if e["status"] == "progress"]
    assert len(progress_events) == 1
    assert progress_events[0]["current"] == 3


def test_make_stage_emitter_emits_end_on_exception() -> None:
    obs = FakeProgressObserver()
    clock = FakeClock()

    with pytest.raises(ValueError):
        with make_stage_emitter("chunk", observer=obs, clock=clock, total=5):
            raise ValueError("fail inside")

    statuses = [e["status"] for e in obs.events]
    assert "error" in statuses
    assert "end" not in statuses
