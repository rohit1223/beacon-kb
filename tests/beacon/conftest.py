"""Shared fixtures for the beacon test suite."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

# ---------------------------------------------------------------------------
# Offline guards for model-backed libraries (Task 02.3)
# ---------------------------------------------------------------------------
# The beacon test suite must never download model artifacts. Docling's PDF
# pipeline (and any HuggingFace-backed code path) attempts network fetches on
# first use; with these flags set, an accidental fetch fails fast with a clear
# error instead of hanging on an unreachable registry. Offline parsing
# coverage is limited to Markdown, HTML, and DOCX, which use declarative
# backends with no model weights. The PDF path is exercised only where
# BEACON_PDF_MODELS_AVAILABLE=1 signals a pre-populated model cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Remove all BEACON_ env vars so each test starts from defaults."""
    for key in list(os.environ):
        if key.startswith("BEACON_"):
            monkeypatch.delenv(key, raising=False)
    yield
