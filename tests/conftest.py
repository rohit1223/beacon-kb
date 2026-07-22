"""Root conftest for the beacon-kb test suite.

Registers all custom pytest markers so collection does not emit
``PytestUnknownMarkWarning``.  Fixtures shared across multiple test
packages should also live here.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers used across the test suite."""
    config.addinivalue_line("markers", "unit: Fast, pure-Python unit tests with no I/O")
    config.addinivalue_line(
        "markers",
        "contract: Protocol-conformance and interface-contract tests",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests requiring external services or real I/O",
    )
    config.addinivalue_line(
        "markers",
        "plugins: Plugin discovery and entry-point loading tests",
    )
    config.addinivalue_line("markers", "cli: Command-line interface tests")
    config.addinivalue_line(
        "markers",
        "evaluation: RAG evaluation and quality-metric tests",
    )
    config.addinivalue_line(
        "markers",
        "performance: Latency and throughput benchmarks",
    )
