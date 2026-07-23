"""FastAPI server package (Epic 01.4+).

App factory, routes, middleware, error handlers, auth, and telemetry live here.
"""

from __future__ import annotations

from beacon.server.app import create_app

__all__ = ["create_app"]
