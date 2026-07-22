"""Minimal CLI stub for beacon-kb.

The full CLI is delivered in Epic 06.
This stub exposes the console-script entry points so that ``beacon-kb`` and
``bkb`` are importable and return exit code 0 from the moment the package is
installed.

Importing this module performs no side effects.
"""

from __future__ import annotations

import sys

from beacon_kb.version import __version__


def main() -> int:
    """Print the package version and a notice that the full CLI arrives in a later release.

    Returns:
        0 on success.
    """
    print(f"beacon-kb {__version__}")
    print("Full CLI arrives in Epic 06 - stay tuned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
