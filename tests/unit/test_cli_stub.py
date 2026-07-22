"""Unit tests for the beacon-kb CLI stub (Epic 01 placeholder).

Verifies:
- main() returns exit code 0.
- main() prints the package version to stdout.
"""

from __future__ import annotations

import pytest

from beacon_kb.cli import main
from beacon_kb.version import __version__


def test_main_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """main() must return 0."""
    code = main()
    assert code == 0


def test_main_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """main() must print the package version string to stdout."""
    main()
    captured = capsys.readouterr()
    assert __version__ in captured.out
