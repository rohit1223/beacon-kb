"""Contract test: pyproject.toml entry-point groups must match groups.ALL_GROUPS.

Parses pyproject.toml with tomllib and asserts that the set of group names
declared under [project.entry-points.*] matches beacon_kb.registry.groups.ALL_GROUPS
exactly.
This prevents the two sources of truth from drifting silently.
"""

from __future__ import annotations

import pathlib
import tomllib

from beacon_kb.registry.groups import ALL_GROUPS


def _repo_root() -> pathlib.Path:
    """Return the repository root (the directory containing pyproject.toml)."""
    # Walk up from this test file until pyproject.toml is found.
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found in any parent directory.")


def test_pyproject_entry_point_groups_match_all_groups() -> None:
    """The entry-point group headers in pyproject.toml must match ALL_GROUPS exactly.

    Ensures that adding a group constant to groups.ALL_GROUPS without adding the
    corresponding [project.entry-points.*] header (or vice versa) is caught by CI.
    """
    root = _repo_root()
    with (root / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)

    # TOML key: [project.entry-points] -> dict of group-name -> dict
    entry_points: dict[str, object] = data.get("project", {}).get("entry-points", {})
    toml_groups = set(entry_points.keys())
    expected_groups = set(ALL_GROUPS)

    missing_from_toml = expected_groups - toml_groups
    missing_from_code = toml_groups - expected_groups

    assert not missing_from_toml, (
        f"Groups in ALL_GROUPS but missing from pyproject.toml entry-points: "
        f"{sorted(missing_from_toml)}. "
        "Add a [project.entry-points.\"<group>\"] header for each."
    )
    assert not missing_from_code, (
        f"Groups in pyproject.toml entry-points but missing from ALL_GROUPS: "
        f"{sorted(missing_from_code)}. "
        f"Add each group name to ALL_GROUPS in src/beacon_kb/registry/groups.py."
    )
