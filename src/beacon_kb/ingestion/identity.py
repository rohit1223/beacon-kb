"""Canonical source URI construction and stable source/revision ID helpers.

All functions in this module are pure and deterministic: identical inputs
always produce identical outputs across processes, independent of the
current working directory or platform path separators.

Importing this module performs no side effects.
"""

from __future__ import annotations

import pathlib
import urllib.parse


def canonicalize_path(path: str) -> str:
    """Return a stable ``file://`` URI for *path*.

    The path is resolved to an absolute POSIX path before encoding so that
    two callers referencing the same file from different working directories
    always receive identical URIs.

    Symlink behaviour:
        ``Path.resolve()`` follows symlinks on POSIX systems, so two symlinks
        that point to the same physical file will produce the same URI.
        On Windows, ``Path.resolve()`` also resolves symlinks but the
        behaviour is subject to OS-level symlink privilege restrictions and
        may differ from POSIX in edge cases (e.g. junction points, network
        paths).  Callers that need to treat each symlink as a distinct source
        should canonicalize the symlink path without resolving it.

    Args:
        path: A filesystem path, either absolute or relative.

    Returns:
        A ``file://`` URI with forward-slash separators and percent-encoded
        special characters.
        Example: ``file:///home/user/docs/guide.md``
    """
    resolved = pathlib.Path(path).resolve()
    posix = resolved.as_posix()
    # urllib.parse.quote handles spaces and special chars; safe=":/" keeps the
    # slash separators readable.
    quoted = urllib.parse.quote(posix, safe=":/")
    return f"file://{quoted}"


def make_file_source_uri(path: pathlib.Path) -> str:
    """Return a stable ``file://`` URI for a :class:`pathlib.Path`.

    Equivalent to :func:`canonicalize_path` for a ``Path`` object.
    Callers should prefer this function when they already have a ``Path``
    instance to avoid an extra string-to-Path round-trip.

    Args:
        path: Filesystem path (absolute or relative).

    Returns:
        A stable ``file://`` URI string.
    """
    return canonicalize_path(str(path))
