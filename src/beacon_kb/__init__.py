"""beacon-kb: Agentic retrieval-augmented generation library.

This package provides a modular, plugin-driven pipeline for building
knowledge bases and retrieval-augmented generation systems.

All imports in this module are lazy or type-only to ensure zero side effects
at import time. No network, filesystem, logging-handler, or credential
operations are performed when this module is imported.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Rohit Kumar"

__all__ = [
    "__author__",
    "__version__",
]
