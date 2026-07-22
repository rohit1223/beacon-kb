"""Single source of truth for version and plugin API version.

Importing this module performs no side effects.
This module is intentionally dependency-free so setuptools can import it
at build time to resolve the dynamic version attribute.
"""

from __future__ import annotations

__version__: str = "0.1.0"
"""Package version string following semantic versioning (MAJOR.MINOR.PATCH)."""

PLUGIN_API_VERSION: int = 1
"""Integer API version for the beacon-kb plugin contract.

Third-party plugins declare the API version they target.  When the plugin
contract changes in a backward-incompatible way this integer is incremented.
Plugins that declare a lower version are refused at registration time.
"""
