"""Sample plugin discovery tests.

Asserts that the already-installed ``beacon-kb-sample-plugin`` distribution
is discovered correctly through ``importlib.metadata`` entry points.

Tests cover:
- Registry discovery of the ``sample-memory`` connector entry point.
- Precedence: a separately registered explicit connector beats the entry-point.
- ``PluginConflict`` when the same name is also registered explicitly (simulating
  a name collision between the installed entry point and another provider).

These tests depend on ``beacon-kb-sample-plugin`` being installed in the
active environment (editable install under ``tests/plugins/sample_plugin/``).
If the package is not installed the entry-point tests will fail with
``PluginNotFound``; run:

    uv pip install -e tests/plugins/sample_plugin --python .venv/bin/python
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

from beacon_kb.errors import PluginConflict, PluginNotFound
from beacon_kb.protocols import Connector
from beacon_kb.registry import discovery, groups, precedence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_NAME = "sample-memory"
_SAMPLE_DIST = "beacon-kb-sample-plugin"


def _reset() -> None:
    """Reset registry and discovery state between tests."""
    precedence.clear_registry()
    discovery.reset_scan_state()


def _sample_plugin_installed() -> bool:
    """Return True if the sample plugin is installed and entry point is visible."""
    eps = entry_points(group=groups.CONNECTORS)
    return any(ep.name == _SAMPLE_NAME for ep in eps)


# Mark tests that need the sample plugin installed.
_INSTALL_HINT = (
    "beacon-kb-sample-plugin not installed; "
    "run: uv pip install -e tests/plugins/sample_plugin"
)
needs_sample = pytest.mark.skipif(not _sample_plugin_installed(), reason=_INSTALL_HINT)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


class TestSamplePluginDiscovery:
    """Verify the sample plugin connector is discovered via entry points."""

    def setup_method(self) -> None:
        _reset()

    @needs_sample
    def test_entry_point_visible_in_metadata(self) -> None:
        """The sample-memory entry point appears in importlib.metadata."""
        eps = entry_points(group=groups.CONNECTORS)
        names = [ep.name for ep in eps]
        assert _SAMPLE_NAME in names, (
            f"Expected '{_SAMPLE_NAME}' in entry points for group '{groups.CONNECTORS}'. "
            f"Found: {names}. Install the sample plugin."
        )

    @needs_sample
    def test_resolve_sample_by_name(self) -> None:
        """Resolving 'sample-memory' by name loads the installed entry point."""
        instance = precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
        assert instance is not None

    @needs_sample
    def test_resolved_sample_satisfies_connector_protocol(self) -> None:
        """The resolved sample connector satisfies the Connector protocol."""
        instance = precedence.resolve(
            group=groups.CONNECTORS,
            name=_SAMPLE_NAME,
            protocol=Connector,
        )
        assert isinstance(instance, Connector)

    @needs_sample
    def test_discovery_scanned_after_resolve(self) -> None:
        """has_scanned() becomes True after resolving an entry-point plugin."""
        assert not discovery.has_scanned()
        precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
        assert discovery.has_scanned()

    @needs_sample
    def test_resolved_sample_is_callable_connector(self) -> None:
        """The resolved connector can list sources and is usable."""
        instance = precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
        sources = instance.list_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0

    @needs_sample
    def test_entry_point_distribution_is_sample_plugin(self) -> None:
        """The sample-memory entry point belongs to beacon-kb-sample-plugin."""
        eps = entry_points(group=groups.CONNECTORS)
        matching = [ep for ep in eps if ep.name == _SAMPLE_NAME]
        assert matching, f"No entry point named {_SAMPLE_NAME!r} found."
        ep = matching[0]
        dist = ep.dist
        assert dist is not None
        dist_name = dist.metadata.get("Name", "")
        assert dist_name == _SAMPLE_DIST, (
            f"Expected distribution '{_SAMPLE_DIST}', got '{dist_name}'."
        )


# ---------------------------------------------------------------------------
# Precedence: explicit wins over entry-point
# ---------------------------------------------------------------------------


class TestExplicitBeatsEntryPoint:
    """Explicit registration wins over the installed entry point."""

    def setup_method(self) -> None:
        _reset()

    @needs_sample
    def test_explicit_wins_over_entry_point(self) -> None:
        """An explicitly registered connector takes precedence over the entry point."""
        from beacon_kb.testing import FakeConnector

        explicit = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name=_SAMPLE_NAME, instance=explicit)
        resolved = precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
        # The explicit instance must win; the entry point must not be loaded.
        assert resolved is explicit

    @needs_sample
    def test_entry_point_used_when_no_explicit(self) -> None:
        """Without an explicit registration, the entry-point plugin is used."""
        resolved = precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
        # Must not be None and must satisfy the Connector protocol.
        assert isinstance(resolved, Connector)


# ---------------------------------------------------------------------------
# Precedence: entry point wins over a built-in connector
# ---------------------------------------------------------------------------


class TestEntryPointBeatsBuiltin:
    """The installed entry point outranks a registered built-in connector."""

    def setup_method(self) -> None:
        _reset()

    @needs_sample
    def test_sole_entry_point_beats_builtin_default(self) -> None:
        """With one installed entry point and a built-in, the entry point wins.

        Precedence level 3 (sole entry-point default) outranks level 4
        (built-in default) when no name is requested.
        """
        from beacon_kb.testing import FakeConnector

        builtin = FakeConnector()
        precedence.register_builtin(
            group=groups.CONNECTORS,
            name="builtin-memory",
            instance=builtin,
        )
        resolved = precedence.resolve(group=groups.CONNECTORS)
        # The sample plugin is the sole installed entry point in the group,
        # so it must be chosen over the built-in default.
        assert resolved is not builtin
        assert type(resolved).__name__ == "SampleMemoryConnector"

    @needs_sample
    def test_builtin_still_reachable_by_name(self) -> None:
        """The built-in remains resolvable by its own name alongside the plugin."""
        from beacon_kb.testing import FakeConnector

        builtin = FakeConnector()
        precedence.register_builtin(
            group=groups.CONNECTORS,
            name="builtin-memory",
            instance=builtin,
        )
        resolved = precedence.resolve(group=groups.CONNECTORS, name="builtin-memory")
        assert resolved is builtin


# ---------------------------------------------------------------------------
# PluginConflict: name collision arranged via scan_group mock
# ---------------------------------------------------------------------------


class TestPluginConflictOnCollision:
    """PluginConflict is raised and names both distributions on a name collision."""

    def setup_method(self) -> None:
        _reset()

    def test_conflict_names_both_distributions(self) -> None:
        """When two entry points share a name, PluginConflict names both."""
        from importlib.metadata import EntryPoint

        # Simulate an installed collision: the real sample plugin AND a hypothetical
        # second distribution both claim the same name.
        def _fake_scan(group: str) -> dict[str, list[EntryPoint]]:
            ep_real = EntryPoint(
                name=_SAMPLE_NAME,
                value="beacon_kb_sample_plugin.connector:SampleMemoryConnector",
                group=groups.CONNECTORS,
            )
            ep_fake = EntryPoint(
                name=_SAMPLE_NAME,
                value="another_pkg.connector:AnotherConnector",
                group=groups.CONNECTORS,
            )
            return {_SAMPLE_NAME: [ep_real, ep_fake]}

        original_scan = discovery.scan_group
        discovery.scan_group = _fake_scan  # type: ignore[assignment]
        try:
            with pytest.raises(PluginConflict) as exc_info:
                precedence.resolve(group=groups.CONNECTORS, name=_SAMPLE_NAME)
            conflict = exc_info.value
            assert conflict.name == _SAMPLE_NAME
            assert conflict.group == groups.CONNECTORS
            assert len(conflict.providers) == 2, (
                "PluginConflict must name BOTH providers, not just one."
            )
        finally:
            discovery.scan_group = original_scan  # type: ignore[assignment]

    def test_conflict_error_message_includes_both_providers(self) -> None:
        """PluginConflict error message references both conflicting providers."""
        from importlib.metadata import EntryPoint

        def _fake_scan(group: str) -> dict[str, list[EntryPoint]]:
            return {
                "colliding": [
                    EntryPoint(name="colliding", value="pkg_a:A", group=group),
                    EntryPoint(name="colliding", value="pkg_b:B", group=group),
                ]
            }

        original_scan = discovery.scan_group
        discovery.scan_group = _fake_scan  # type: ignore[assignment]
        try:
            with pytest.raises(PluginConflict) as exc_info:
                precedence.resolve(group=groups.CONNECTORS, name="colliding")
            msg = str(exc_info.value)
            # Both provider values must appear in the error message.
            assert len(exc_info.value.providers) == 2
            assert "pkg_a:A" in msg and "pkg_b:B" in msg
        finally:
            discovery.scan_group = original_scan  # type: ignore[assignment]

    @needs_sample
    def test_missing_name_raises_not_found(self) -> None:
        """PluginNotFound raised when a non-existent name is requested."""
        with pytest.raises(PluginNotFound):
            precedence.resolve(group=groups.CONNECTORS, name="nonexistent-connector-xyz")
