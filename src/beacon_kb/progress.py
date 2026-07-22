"""Structured pipeline progress events and observer adapters.

This module provides:
- ``StageEvent``: a frozen structured record for pipeline stage progress
  carrying stage, status, current/total, and elapsed time.
- ``NullProgressObserver``: a no-op observer used as the safe default.
- ``LoggingProgressAdapter``: forwards events to an inner observer and logs
  them in a TTY-neutral way (plain logging records, no carriage-return
  animation, no ANSI escapes).
- ``make_stage_emitter``: a context manager that brackets a long-running
  stage with start/end (or error) events and yields an ``emit`` callable
  for ``current/total`` progress updates.

The clock is injectable (any object with ``now() -> float``) so tests can
drive elapsed-time assertions deterministically with ``FakeClock``.

Importing this module performs no side effects.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from beacon_kb.protocols import ProgressObserver

logger = logging.getLogger(__name__)


@runtime_checkable
class Clock(Protocol):
    """Protocol for injectable clocks used in elapsed-time computation.

    Implementations expose ``now()`` returning float seconds from an
    arbitrary but monotonically consistent epoch.  ``FakeClock`` from
    ``beacon_kb.testing`` satisfies this protocol.
    """

    def now(self) -> float:
        """Return the current time in float seconds."""
        ...


class _WallClock:
    """Real monotonic wall clock used when no clock is injected."""

    def now(self) -> float:
        """Return monotonic time in float seconds."""
        return time.monotonic()


def _default_clock() -> Clock:
    """Return a clock reading real monotonic wall time."""
    return _WallClock()


class EmitFn(Protocol):
    """Callable yielded by :func:`make_stage_emitter` for progress updates."""

    def __call__(self, current: int = 0, message: str = "") -> None:
        """Emit a 'progress' event with *current* items done."""
        ...


@dataclass(frozen=True, slots=True)
class StageEvent:
    """Structured event emitted at pipeline stage boundaries.

    Attributes:
        stage:           Name of the pipeline stage ('chunk', 'enrich', 'embed').
        status:          One of 'start', 'progress', 'end', 'error'.
        current:         Items processed so far (0 for start).
        total:           Total items expected (0 if unknown).
        elapsed_seconds: Seconds since stage start (0.0 for start events).
        message:         Optional human-readable note.
    """

    stage: str
    status: str
    current: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for ProgressObserver.on_event()."""
        return {
            "stage": self.stage,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "elapsed_seconds": self.elapsed_seconds,
            "message": self.message,
        }


class NullProgressObserver:
    """No-op ProgressObserver that silently discards all events.

    Use as a default when the caller supplies no observer.
    """

    def on_event(self, event: dict[str, Any]) -> None:
        """Accept and discard the event silently."""
        return


class LoggingProgressAdapter:
    """ProgressObserver that forwards events to an inner observer and logs them.

    The logging output is TTY-neutral: it uses plain ``logging`` records with
    no carriage returns, ANSI escapes, or terminal-width assumptions, so it is
    equally safe when stdout is a pipe, a file, or an interactive terminal.

    Error contract: on_event() must never raise regardless of inner observer
    or logging failures (ProgressObserver protocol requirement).

    Args:
        observer: Inner ProgressObserver to forward events to.
        clock:    Injectable clock exposing ``now() -> float``.
                  Defaults to a real monotonic wall clock.
    """

    def __init__(
        self,
        *,
        observer: ProgressObserver,
        clock: Clock | None = None,
    ) -> None:
        self._observer = observer
        self._clock: Clock = clock if clock is not None else _default_clock()

    def on_event(self, event: dict[str, Any]) -> None:
        """Forward event to inner observer and log it at DEBUG level.

        Must not raise under any circumstances; internal errors are
        intentionally suppressed per the ProgressObserver error contract.
        """
        with contextlib.suppress(Exception):
            self._observer.on_event(event)
        with contextlib.suppress(Exception):
            logger.debug(
                "[%s] %s  %s/%s  %.2fs",
                event.get("stage", "?"),
                event.get("status", "?"),
                event.get("current", 0),
                event.get("total", 0),
                event.get("elapsed_seconds", 0.0),
            )


@contextlib.contextmanager
def make_stage_emitter(
    stage: str,
    *,
    observer: ProgressObserver,
    clock: Clock | None = None,
    total: int = 0,
) -> Generator[EmitFn, None, None]:
    """Context manager that emits start/progress/end events for a named stage.

    Usage::

        with make_stage_emitter("embed", observer=obs, clock=clock, total=n) as emit:
            for i, item in enumerate(items):
                process(item)
                emit(current=i + 1)

    The yielded ``emit`` callable accepts ``current`` and optional ``message``
    keyword arguments and fires a 'progress' event through the observer.

    A 'start' event is emitted on enter.  On normal exit an 'end' event is
    emitted with ``current=total``; if the body raises, an 'error' event is
    emitted (carrying the exception message) and the exception propagates.
    All events carry ``elapsed_seconds`` measured from the injected clock.

    Args:
        stage:    Name of the pipeline stage.
        observer: ProgressObserver instance receiving the events.
        clock:    Injectable clock (defaults to monotonic wall clock).
        total:    Total item count for current/total progress reporting.
    """
    active_clock: Clock = clock if clock is not None else _default_clock()
    t0: float = active_clock.now()

    def _emit_event(status: str, current: int = 0, message: str = "") -> None:
        evt = StageEvent(
            stage=stage,
            status=status,
            current=current,
            total=total,
            elapsed_seconds=active_clock.now() - t0,
            message=message,
        )
        with contextlib.suppress(Exception):
            observer.on_event(evt.as_dict())

    def emit(current: int = 0, message: str = "") -> None:
        _emit_event("progress", current=current, message=message)

    _emit_event("start")
    try:
        yield emit
        _emit_event("end", current=total)
    except BaseException as exc:
        _emit_event("error", message=str(exc))
        raise
