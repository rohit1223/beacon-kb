"""Typed error hierarchy for beacon-kb.

All exceptions inherit from BeaconError. The hierarchy covers config,
readiness, backend, ingestion, citation, plugin, budget, and agentic errors.

Plugin errors include PluginConflict, PluginNotFound, and ProtocolMismatch
as distinct typed classes so callers can handle each independently.

Importing this module performs no side effects.
"""

from __future__ import annotations


class BeaconError(Exception):
    """Base exception for all beacon-kb errors.

    Subclass and raise this to signal any beacon-kb-specific failure.
    """


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigError(BeaconError):
    """Raised when the beacon-kb configuration is invalid or missing required fields.

    Examples: missing required key, type mismatch, unsupported combination.
    """


# ---------------------------------------------------------------------------
# Readiness errors
# ---------------------------------------------------------------------------


class ReadinessError(BeaconError):
    """Raised when the knowledge base is not ready to serve requests.

    Examples: no active revision exists, schema migration pending,
    index corrupt and requires rebuild.
    """


# ---------------------------------------------------------------------------
# Backend errors
# ---------------------------------------------------------------------------


class BackendError(BeaconError):
    """Raised when an underlying storage or computation backend fails.

    Examples: SQLite write failure, NumPy dimension mismatch,
    vector store connection timeout.
    """


# ---------------------------------------------------------------------------
# Ingestion errors
# ---------------------------------------------------------------------------


class IngestionError(BeaconError):
    """Raised when document ingestion fails at any pipeline stage.

    Examples: connector fetch failure, parser decode error,
    chunker produces zero chunks, embedding batch rejected.
    """


# ---------------------------------------------------------------------------
# Citation errors
# ---------------------------------------------------------------------------


class CitationError(BeaconError):
    """Raised when citation validation or grounding fails.

    Examples: generated citation references a non-existent chunk,
    citation label does not match any evidence item,
    abstention required but generator did not abstain.
    """


# ---------------------------------------------------------------------------
# Plugin errors
# ---------------------------------------------------------------------------


class PluginError(BeaconError):
    """Base class for plugin registry errors.

    Subclasses distinguish conflict detection, lookup failures,
    and protocol conformance failures.
    """


class PluginConflict(PluginError):
    """Raised when two plugins register the same name within one entry-point group.

    Args:
        group: The entry-point group in which the conflict was detected.
        name: The conflicting plugin name.
        providers: List of provider identifiers (package names or module paths) that conflict.
    """

    def __init__(self, group: str, name: str, providers: list[str]) -> None:
        self.group = group
        self.name = name
        self.providers = providers
        super().__init__(
            f"Plugin conflict in group '{group}': name '{name}' registered by multiple "
            f"providers: {providers}. Remove or alias all but one."
        )


class PluginNotFound(PluginError):
    """Raised when a requested plugin name is not registered in the given group.

    Args:
        group: The entry-point group that was searched.
        name: The plugin name that was not found.
        installed: Optional list of plugin names that ARE available in the group,
            included in the message so the caller can pick a valid name.
    """

    def __init__(self, group: str, name: str, installed: list[str] | None = None) -> None:
        self.group = group
        self.name = name
        self.installed = installed if installed is not None else []
        if self.installed:
            available = f"Installed names: {self.installed}."
        else:
            available = "No plugins are installed in this group."
        super().__init__(
            f"Plugin '{name}' not found in group '{group}'. {available} "
            f"Ensure the package is installed and the entry-point is declared."
        )


class ProtocolMismatch(PluginError):
    """Raised when a plugin object does not conform to the expected Protocol.

    Args:
        group: The entry-point group for which conformance was checked.
        name: The plugin name that failed the check.
        missing_members: List of protocol member names absent on the plugin object.
    """

    def __init__(self, group: str, name: str, missing_members: list[str]) -> None:
        self.group = group
        self.name = name
        self.missing_members = missing_members
        super().__init__(
            f"Plugin '{name}' in group '{group}' does not conform to the required protocol. "
            f"Missing members: {missing_members}."
        )


# ---------------------------------------------------------------------------
# Budget errors
# ---------------------------------------------------------------------------


class BudgetError(BeaconError):
    """Raised when a token, cost, or iteration budget is exceeded.

    Examples: input_tokens + output_tokens exceeds context window budget,
    investigate() exceeds max_steps before stop condition is met,
    per-query cost ceiling reached.
    """


# ---------------------------------------------------------------------------
# Agentic errors
# ---------------------------------------------------------------------------


class AgenticError(BeaconError):
    """Raised when the agentic investigation loop encounters an unrecoverable state.

    Examples: no valid tool calls produced in a step, loop diverges,
    stop condition raises an exception, session state is corrupt.
    """
