"""Shared fixtures for the beacon test suite."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Remove all BEACON_ env vars so each test starts from defaults."""
    for key in list(os.environ):
        if key.startswith("BEACON_"):
            monkeypatch.delenv(key, raising=False)
    yield
