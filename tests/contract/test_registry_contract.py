"""Registry contract tests.

Exercises the plugin registry against fakes from ``beacon_kb.testing``
to prove:
- Precedence order (explicit > config-named > sole entry-point > built-in).
- ``PluginConflict`` on duplicate names (both distributions named).
- ``PluginNotFound`` for unknown plugin names.
- ``ProtocolMismatch`` when resolved object fails protocol check.
- Capability rejection (incompatible ``plugin_api_version``).
- Lazy discovery (importing the registry must not scan entry points).

These tests reset the registry and discovery state around each test so
they remain order-independent.
"""

from __future__ import annotations

import pytest

from beacon_kb.errors import PluginConflict, PluginError, PluginNotFound, ProtocolMismatch
from beacon_kb.protocols import Connector, TokenCounter
from beacon_kb.registry import discovery, groups, precedence
from beacon_kb.registry import resolve as registry_resolve
from beacon_kb.testing import FakeConnector, FakeTokenCounter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset() -> None:
    """Reset registry and discovery state between tests."""
    precedence.clear_registry()
    discovery.reset_scan_state()


# ---------------------------------------------------------------------------
# Lazy discovery
# ---------------------------------------------------------------------------


class TestLazyDiscovery:
    """Importing the registry module must not scan entry points."""

    def setup_method(self) -> None:
        _reset()

    def test_has_scanned_false_after_reset(self) -> None:
        """has_scanned() must remain False until resolve() is called."""
        # True test: python -c "import beacon_kb.registry; \
        # assert not beacon_kb.registry.discovery.has_scanned()"
        # We can't re-import fresh, but we can reset and confirm the predicate
        # stays False before any resolution.
        assert not discovery.has_scanned()

    def test_has_scanned_true_after_resolve_triggers_scan(self) -> None:
        """has_scanned() becomes True after a group scan is triggered."""
        # Register a built-in so resolve doesn't raise PluginNotFound.
        precedence.register_builtin(
            group=groups.TOKEN_COUNTERS,
            name="heuristic",
            instance=FakeTokenCounter(),
        )
        precedence.resolve(group=groups.TOKEN_COUNTERS)
        assert discovery.has_scanned()

    def test_register_does_not_trigger_scan(self) -> None:
        """register() must not scan entry points."""
        precedence.register(group=groups.CONNECTORS, name="my-conn", instance=FakeConnector())
        assert not discovery.has_scanned()

    def test_list_registered_does_not_trigger_scan(self) -> None:
        """list_registered() must not scan entry points."""
        precedence.list_registered(group=groups.CONNECTORS)
        assert not discovery.has_scanned()


# ---------------------------------------------------------------------------
# Precedence order
# ---------------------------------------------------------------------------


class TestPrecedenceOrder:
    """Verify the four-level precedence chain in documented order."""

    def setup_method(self) -> None:
        _reset()

    def test_explicit_instance_wins_over_builtin(self) -> None:
        """Explicit instance (level 1) beats the built-in default (level 4)."""
        builtin = FakeTokenCounter()
        explicit = FakeTokenCounter()

        # Register built-in first.
        precedence.register_builtin(group=groups.TOKEN_COUNTERS, name="heuristic", instance=builtin)
        # Then register explicit under the same group with a different name.
        precedence.register(group=groups.TOKEN_COUNTERS, name="my-counter", instance=explicit)

        # Resolving by name returns the explicit one.
        resolved = precedence.resolve(group=groups.TOKEN_COUNTERS, name="my-counter")
        assert resolved is explicit

    def test_builtin_default_returned_when_no_explicit(self) -> None:
        """Built-in default (level 4) is returned when nothing else matches."""
        builtin = FakeTokenCounter()
        precedence.register_builtin(group=groups.TOKEN_COUNTERS, name="heuristic", instance=builtin)
        resolved = precedence.resolve(group=groups.TOKEN_COUNTERS)
        assert resolved is builtin

    def test_explicit_name_wins_over_builtin_by_name(self) -> None:
        """Named explicit wins over named built-in when names match."""
        builtin = FakeTokenCounter()
        override = FakeTokenCounter()
        precedence.register_builtin(group=groups.TOKEN_COUNTERS, name="heuristic", instance=builtin)
        precedence.register(group=groups.TOKEN_COUNTERS, name="heuristic", instance=override)
        resolved = precedence.resolve(group=groups.TOKEN_COUNTERS, name="heuristic")
        assert resolved is override

    def test_resolve_sole_explicit_without_name(self) -> None:
        """When only one explicit is registered and name=None, it is returned."""
        only_one = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name="my-conn", instance=only_one)
        resolved = precedence.resolve(group=groups.CONNECTORS)
        assert resolved is only_one

    def test_plugin_not_found_when_group_empty(self) -> None:
        """PluginNotFound raised when group has nothing registered.

        Uses groups.GRADERS which has no installed entry points, no built-in,
        and no explicit registrations in the reset state.
        """
        with pytest.raises(PluginNotFound) as exc_info:
            precedence.resolve(group=groups.GRADERS)
        assert exc_info.value.group == groups.GRADERS

    def test_plugin_not_found_includes_group(self) -> None:
        """PluginNotFound message includes the group name."""
        with pytest.raises(PluginNotFound) as exc_info:
            precedence.resolve(group=groups.GRADERS, name="nonexistent")
        assert groups.GRADERS in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)


# ---------------------------------------------------------------------------
# PluginConflict
# ---------------------------------------------------------------------------


class TestPluginConflict:
    """Verify that duplicate names raise PluginConflict naming both providers."""

    def setup_method(self) -> None:
        _reset()

    def test_conflict_raised_with_both_providers(self) -> None:
        """PluginConflict must name both conflicting distributions."""
        from importlib.metadata import EntryPoint

        # Patch scan_group to return a fake conflict.
        def _fake_scan(group: str) -> dict[str, list[EntryPoint]]:
            # Simulate two entry points registering the same name.
            ep_a = EntryPoint(
                name="dupe-connector",
                value="pkg_a.connector:ConnA",
                group=groups.CONNECTORS,
            )
            ep_b = EntryPoint(
                name="dupe-connector",
                value="pkg_b.connector:ConnB",
                group=groups.CONNECTORS,
            )
            return {"dupe-connector": [ep_a, ep_b]}

        original_scan = discovery.scan_group
        discovery.scan_group = _fake_scan  # type: ignore[assignment]
        try:
            with pytest.raises(PluginConflict) as exc_info:
                precedence.resolve(group=groups.CONNECTORS, name="dupe-connector")
            conflict = exc_info.value
            assert conflict.group == groups.CONNECTORS
            assert conflict.name == "dupe-connector"
            assert len(conflict.providers) == 2
        finally:
            discovery.scan_group = original_scan  # type: ignore[assignment]

    def test_conflict_no_last_wins(self) -> None:
        """Conflict detection must not silently pick the last-installed plugin."""
        from importlib.metadata import EntryPoint

        def _fake_scan(group: str) -> dict[str, list[EntryPoint]]:
            ep_a = EntryPoint(
                name="clash",
                value="pkg_a:A",
                group=group,
            )
            ep_b = EntryPoint(
                name="clash",
                value="pkg_b:B",
                group=group,
            )
            return {"clash": [ep_a, ep_b]}

        original_scan = discovery.scan_group
        discovery.scan_group = _fake_scan  # type: ignore[assignment]
        try:
            with pytest.raises(PluginConflict):
                precedence.resolve(group=groups.CONNECTORS, name="clash")
        finally:
            discovery.scan_group = original_scan  # type: ignore[assignment]

    def test_conflict_with_name_none_raises_plugin_conflict(self) -> None:
        """Resolving with name=None in a group with duplicate names raises PluginConflict."""
        from importlib.metadata import EntryPoint

        def _fake_scan(group: str) -> dict[str, list[EntryPoint]]:
            ep_a = EntryPoint(
                name="dupe",
                value="pkg_a:A",
                group=group,
            )
            ep_b = EntryPoint(
                name="dupe",
                value="pkg_b:B",
                group=group,
            )
            return {"dupe": [ep_a, ep_b]}

        original_scan = discovery.scan_group
        discovery.scan_group = _fake_scan  # type: ignore[assignment]
        try:
            with pytest.raises(PluginConflict) as exc_info:
                precedence.resolve(group=groups.CONNECTORS, name=None)
            assert len(exc_info.value.providers) == 2
        finally:
            discovery.scan_group = original_scan  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# PluginNotFound
# ---------------------------------------------------------------------------


class TestPluginNotFound:
    """Verify PluginNotFound is raised with group and name context."""

    def setup_method(self) -> None:
        _reset()

    def test_not_found_for_unknown_name(self) -> None:
        """Requesting a non-existent name raises PluginNotFound."""
        # Register something so the group exists but with a different name.
        precedence.register(
            group=groups.CONNECTORS,
            name="existing-conn",
            instance=FakeConnector(),
        )
        with pytest.raises(PluginNotFound) as exc_info:
            precedence.resolve(group=groups.CONNECTORS, name="missing-name")
        err = exc_info.value
        assert err.group == groups.CONNECTORS
        assert err.name == "missing-name"

    def test_not_found_group_in_message(self) -> None:
        """PluginNotFound message references the group that was searched."""
        with pytest.raises(PluginNotFound) as exc_info:
            precedence.resolve(group=groups.PLANNERS, name="nonexistent")
        assert groups.PLANNERS in str(exc_info.value)

    def test_not_found_lists_installed_names(self) -> None:
        """PluginNotFound lists the names that ARE available in the group."""
        precedence.register(
            group=groups.CONNECTORS,
            name="existing-conn",
            instance=FakeConnector(),
        )
        precedence.register(
            group=groups.CONNECTORS,
            name="another-conn",
            instance=FakeConnector(),
        )
        with pytest.raises(PluginNotFound) as exc_info:
            precedence.resolve(group=groups.CONNECTORS, name="missing-name")
        err = exc_info.value
        assert "existing-conn" in err.installed
        assert "another-conn" in err.installed
        assert "existing-conn" in str(err)
        assert "another-conn" in str(err)


# ---------------------------------------------------------------------------
# ProtocolMismatch
# ---------------------------------------------------------------------------


class TestProtocolMismatch:
    """Verify ProtocolMismatch when an object fails a protocol check."""

    def setup_method(self) -> None:
        _reset()

    def test_protocol_mismatch_for_wrong_type(self) -> None:
        """An object missing protocol members raises ProtocolMismatch."""

        class NotAConnector:
            """Object that does not implement Connector."""
            pass

        precedence.register(
            group=groups.CONNECTORS,
            name="bad-conn",
            instance=NotAConnector(),
        )
        with pytest.raises(ProtocolMismatch) as exc_info:
            precedence.resolve(
                group=groups.CONNECTORS,
                name="bad-conn",
                protocol=Connector,
            )
        err = exc_info.value
        assert err.group == groups.CONNECTORS
        assert err.name == "bad-conn"
        assert len(err.missing_members) > 0

    def test_protocol_ok_for_correct_type(self) -> None:
        """A valid Connector passes protocol validation without error."""
        conn = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name="good-conn", instance=conn)
        resolved = precedence.resolve(
            group=groups.CONNECTORS,
            name="good-conn",
            protocol=Connector,
        )
        assert resolved is conn

    def test_protocol_ok_for_token_counter(self) -> None:
        """FakeTokenCounter satisfies the TokenCounter protocol."""
        tc = FakeTokenCounter()
        precedence.register(group=groups.TOKEN_COUNTERS, name="fake-tc", instance=tc)
        resolved = precedence.resolve(
            group=groups.TOKEN_COUNTERS,
            name="fake-tc",
            protocol=TokenCounter,
        )
        assert resolved is tc

    def test_protocol_mismatch_lists_missing_members(self) -> None:
        """ProtocolMismatch names the missing protocol members."""

        class EmptyObject:
            pass

        precedence.register(group=groups.CONNECTORS, name="empty", instance=EmptyObject())
        with pytest.raises(ProtocolMismatch) as exc_info:
            precedence.resolve(
                group=groups.CONNECTORS,
                name="empty",
                protocol=Connector,
            )
        assert exc_info.value.missing_members  # at least one missing member

    def test_auto_protocol_raises_mismatch_without_kwarg(self) -> None:
        """registry.resolve() raises ProtocolMismatch even when no protocol kwarg is passed.

        The group's canonical protocol is applied automatically via
        groups.get_protocol_for_group(), so callers get the same guarantee
        whether or not they supply the protocol argument explicitly.
        """

        class NotAConnector:
            """Object that does not implement Connector."""
            pass

        precedence.register(
            group=groups.CONNECTORS,
            name="non-conforming",
            instance=NotAConnector(),
        )
        # Note: no protocol= kwarg - the registry facade should infer Connector.
        with pytest.raises(ProtocolMismatch) as exc_info:
            registry_resolve(group=groups.CONNECTORS, name="non-conforming")
        err = exc_info.value
        assert err.group == groups.CONNECTORS
        assert err.name == "non-conforming"

    def test_auto_protocol_conforming_object_passes(self) -> None:
        """A conforming object resolves successfully without an explicit protocol kwarg."""
        conn = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name="good-conn-auto", instance=conn)
        resolved = registry_resolve(group=groups.CONNECTORS, name="good-conn-auto")
        assert resolved is conn

    def test_auto_protocol_builtin_token_counter_passes(self) -> None:
        """The built-in HeuristicTokenCounter satisfies the TokenCounter protocol automatically."""
        from beacon_kb.tokens import HeuristicTokenCounter
        tc = HeuristicTokenCounter()
        precedence.register_builtin(group=groups.TOKEN_COUNTERS, name="heuristic", instance=tc)
        resolved = registry_resolve(group=groups.TOKEN_COUNTERS)
        assert resolved is tc


# ---------------------------------------------------------------------------
# Capability rejection
# ---------------------------------------------------------------------------


class TestCapabilityRejection:
    """Verify plugins with incompatible API versions are rejected."""

    def setup_method(self) -> None:
        _reset()

    def test_plugin_with_old_api_version_rejected(self) -> None:
        """A plugin declaring plugin_api_version < current is rejected."""

        class OldPlugin:
            plugin_api_version: int = 0

        precedence.register(group=groups.CONNECTORS, name="old-plugin", instance=OldPlugin())
        with pytest.raises(PluginError) as exc_info:
            precedence.resolve(group=groups.CONNECTORS, name="old-plugin")
        assert "plugin_api_version" in str(exc_info.value)

    def test_plugin_with_current_api_version_accepted(self) -> None:
        """A plugin declaring the current API version is accepted."""
        from beacon_kb.version import PLUGIN_API_VERSION

        class CurrentPlugin:
            plugin_api_version: int = PLUGIN_API_VERSION

            def list_sources(self) -> list[str]:
                return []

            def fetch(self, uri: str) -> None:  # type: ignore[return]
                ...

        instance = CurrentPlugin()
        precedence.register(group=groups.CONNECTORS, name="current-plugin", instance=instance)
        resolved = precedence.resolve(group=groups.CONNECTORS, name="current-plugin")
        assert resolved is instance

    def test_plugin_with_no_version_attr_accepted(self) -> None:
        """A plugin without plugin_api_version attribute is accepted (opt-in)."""
        conn = FakeConnector()
        # FakeConnector has no plugin_api_version; it must be accepted.
        assert not hasattr(conn, "plugin_api_version")
        precedence.register(group=groups.CONNECTORS, name="versionless", instance=conn)
        resolved = precedence.resolve(group=groups.CONNECTORS, name="versionless")
        assert resolved is conn

    def test_config_conflicting_capability_rejected(self) -> None:
        """A plugin whose declared capability conflicts with config is rejected."""
        from beacon_kb.testing import FakeEmbedder

        embedder = FakeEmbedder(dim=16)
        precedence.register(group=groups.EMBEDDERS, name="fake-embedder", instance=embedder)
        with pytest.raises(PluginError) as exc_info:
            precedence.resolve(
                group=groups.EMBEDDERS,
                name="fake-embedder",
                capabilities={"dimension": 768},
            )
        msg = str(exc_info.value)
        assert "dimension" in msg
        assert "768" in msg
        assert "16" in msg

    def test_matching_capability_accepted(self) -> None:
        """A plugin whose declared capability matches config is accepted."""
        from beacon_kb.testing import FakeEmbedder

        embedder = FakeEmbedder(dim=768)
        precedence.register(group=groups.EMBEDDERS, name="fake-embedder", instance=embedder)
        resolved = precedence.resolve(
            group=groups.EMBEDDERS,
            name="fake-embedder",
            capabilities={"dimension": 768},
        )
        assert resolved is embedder

    def test_undeclared_capability_accepted(self) -> None:
        """A plugin that does not declare the requested capability is accepted."""
        conn = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name="my-conn", instance=conn)
        resolved = precedence.resolve(
            group=groups.CONNECTORS,
            name="my-conn",
            capabilities={"supports_streaming": True},
        )
        assert resolved is conn

    def test_plugin_with_non_int_api_version_rejected(self) -> None:
        """A plugin declaring a non-integer plugin_api_version is rejected."""

        class BadVersion:
            plugin_api_version: str = "1"  # type: ignore[assignment]

        precedence.register(group=groups.CONNECTORS, name="bad-ver", instance=BadVersion())
        with pytest.raises(PluginError):
            precedence.resolve(group=groups.CONNECTORS, name="bad-ver")

    def test_plugin_with_future_api_version_rejected(self) -> None:
        """A plugin declaring a future plugin_api_version (e.g. 2) is rejected.

        The check is now strict equality: any version != PLUGIN_API_VERSION is
        refused to avoid silent compatibility assumptions in both directions.
        """
        from beacon_kb.version import PLUGIN_API_VERSION

        class FuturePlugin:
            plugin_api_version: int = PLUGIN_API_VERSION + 1

        precedence.register(
            group=groups.CONNECTORS, name="future-plugin", instance=FuturePlugin()
        )
        with pytest.raises(PluginError) as exc_info:
            precedence.resolve(group=groups.CONNECTORS, name="future-plugin")
        assert "plugin_api_version" in str(exc_info.value)
        assert "incompatible" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


class TestBuiltinRegistration:
    """Verify built-ins are registered through the same path as third-party plugins."""

    def setup_method(self) -> None:
        _reset()

    def test_builtin_accessible_via_resolve(self) -> None:
        """A registered built-in is accessible through resolve()."""
        builtin = FakeTokenCounter()
        precedence.register_builtin(
            group=groups.TOKEN_COUNTERS,
            name="heuristic",
            instance=builtin,
        )
        resolved = precedence.resolve(group=groups.TOKEN_COUNTERS)
        assert resolved is builtin

    def test_no_privileged_path_for_builtins(self) -> None:
        """Built-ins go through the same register_builtin() path as third-party plugins."""
        # The built-in registered via register_builtin must be resolvable
        # with or without a name argument.
        builtin = FakeTokenCounter()
        precedence.register_builtin(
            group=groups.TOKEN_COUNTERS,
            name="heuristic",
            instance=builtin,
        )
        by_name = precedence.resolve(group=groups.TOKEN_COUNTERS, name="heuristic")
        by_default = precedence.resolve(group=groups.TOKEN_COUNTERS)
        assert by_name is builtin
        assert by_default is builtin

    def test_empty_group_raises_not_found(self) -> None:
        """PluginNotFound raised for a group with no built-in and no registered plugins."""
        with pytest.raises(PluginNotFound):
            precedence.resolve(group=groups.GRADERS)

    def test_register_and_list(self) -> None:
        """list_registered() returns names that were registered."""
        precedence.register(group=groups.CONNECTORS, name="a", instance=FakeConnector())
        precedence.register(group=groups.CONNECTORS, name="b", instance=FakeConnector())
        names = precedence.list_registered(group=groups.CONNECTORS)
        assert "a" in names
        assert "b" in names

    def test_describe_registered_plugin(self) -> None:
        """describe() returns a dict with group, name, instance keys."""
        conn = FakeConnector()
        precedence.register(group=groups.CONNECTORS, name="my-conn", instance=conn)
        info = precedence.describe(group=groups.CONNECTORS, name="my-conn")
        assert info["group"] == groups.CONNECTORS
        assert info["name"] == "my-conn"
        assert info["instance"] is conn

    def test_describe_unknown_raises_not_found(self) -> None:
        """describe() raises PluginNotFound for an unknown plugin name."""
        with pytest.raises(PluginNotFound):
            precedence.describe(group=groups.CONNECTORS, name="ghost")
