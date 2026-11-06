from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable


PluginCallback = Callable[..., Any]


@runtime_checkable
class DecisionSimulationPlugin(Protocol):
    """Domain boundary for one pinned decision-simulation model.

    The registry deliberately describes capabilities as callbacks and imports no
    concrete candidate, engine, critic, or persistence types.  Domain modules can
    therefore depend on this boundary without making the simulation package's
    service/jobs import graph circular.
    """

    model_version: str
    family: str
    operation: str
    build_candidates: PluginCallback
    validate_candidate: PluginCallback
    create_engine: PluginCallback
    build_runtime_card: PluginCallback
    run_critic: PluginCallback
    project_result: PluginCallback
    refresh_snapshot: PluginCallback


@dataclass(frozen=True)
class DecisionSimulationPluginSpec:
    """Callable-backed implementation convenient for application wiring."""

    model_version: str
    family: str
    operation: str
    build_candidates: PluginCallback
    validate_candidate: PluginCallback
    create_engine: PluginCallback
    build_runtime_card: PluginCallback
    run_critic: PluginCallback
    project_result: PluginCallback
    refresh_snapshot: PluginCallback


class SimulationPluginRegistryError(Exception):
    """Base class for deterministic plugin-registry failures."""


class InvalidSimulationPluginError(SimulationPluginRegistryError, ValueError):
    """Raised when a plugin does not satisfy the domain boundary."""


class DuplicateSimulationPluginError(SimulationPluginRegistryError, ValueError):
    """Raised instead of silently replacing a pinned model or domain."""


class UnknownSimulationPluginError(SimulationPluginRegistryError, LookupError):
    """Raised instead of falling back to an unrelated simulation model."""


class DecisionSimulationPluginRegistry:
    """Process-local, fail-closed registry for simulation domain plugins."""

    _CALLBACK_FIELDS = (
        "build_candidates",
        "validate_candidate",
        "create_engine",
        "build_runtime_card",
        "run_critic",
        "project_result",
        "refresh_snapshot",
    )

    def __init__(self) -> None:
        self._by_model_version: dict[str, DecisionSimulationPlugin] = {}
        self._by_domain: dict[tuple[str, str], DecisionSimulationPlugin] = {}
        self._lock = RLock()

    def register(self, plugin: DecisionSimulationPlugin) -> DecisionSimulationPlugin:
        """Register one plugin atomically.

        Durable runs are pinned by ``model_version`` and new decisions are routed
        by ``(family, operation)``.  Neither key is replaceable implicitly.
        """

        model_version, domain = self._validated_identity(plugin)
        self._validate_callbacks(plugin)
        with self._lock:
            if model_version in self._by_model_version:
                raise DuplicateSimulationPluginError(
                    f"simulation model version is already registered: {model_version}"
                )
            if domain in self._by_domain:
                family, operation = domain
                raise DuplicateSimulationPluginError(
                    "simulation domain is already registered: "
                    f"family={family}, operation={operation}"
                )
            # Validate both uniqueness constraints before changing either index.
            self._by_model_version[model_version] = plugin
            self._by_domain[domain] = plugin
        return plugin

    def get_by_model_version(self, model_version: str) -> DecisionSimulationPlugin:
        normalized = _required_text(model_version, field="model_version")
        with self._lock:
            plugin = self._by_model_version.get(normalized)
        if plugin is None:
            raise UnknownSimulationPluginError(
                f"unknown simulation model version: {normalized}"
            )
        return plugin

    def get_by_domain(self, family: str, operation: str) -> DecisionSimulationPlugin:
        domain = _domain_key(family, operation)
        with self._lock:
            plugin = self._by_domain.get(domain)
        if plugin is None:
            normalized_family, normalized_operation = domain
            raise UnknownSimulationPluginError(
                "unknown simulation domain: "
                f"family={normalized_family}, operation={normalized_operation}"
            )
        return plugin

    def registered_model_versions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._by_model_version))

    def registered_domains(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(sorted(self._by_domain))

    @staticmethod
    def _validated_identity(
        plugin: DecisionSimulationPlugin,
    ) -> tuple[str, tuple[str, str]]:
        try:
            model_version = _required_text(plugin.model_version, field="model_version")
            domain = _domain_key(plugin.family, plugin.operation)
        except AttributeError as exc:
            raise InvalidSimulationPluginError(
                f"simulation plugin is missing identity field: {exc}"
            ) from exc
        return model_version, domain

    def _validate_callbacks(self, plugin: DecisionSimulationPlugin) -> None:
        for field in self._CALLBACK_FIELDS:
            callback = getattr(plugin, field, None)
            if not callable(callback):
                raise InvalidSimulationPluginError(
                    f"simulation plugin callback must be callable: {field}"
                )


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSimulationPluginError(
            f"simulation plugin {field} must be a non-empty string"
        )
    return value.strip()


def _domain_key(family: Any, operation: Any) -> tuple[str, str]:
    return (
        _required_text(family, field="family").casefold(),
        _required_text(operation, field="operation").casefold(),
    )


default_simulation_plugin_registry = DecisionSimulationPluginRegistry()


def register_decision_simulation_plugin(
    plugin: DecisionSimulationPlugin,
) -> DecisionSimulationPlugin:
    return default_simulation_plugin_registry.register(plugin)


def get_decision_simulation_plugin(model_version: str) -> DecisionSimulationPlugin:
    return default_simulation_plugin_registry.get_by_model_version(model_version)


def get_decision_simulation_plugin_for_domain(
    family: str,
    operation: str,
) -> DecisionSimulationPlugin:
    return default_simulation_plugin_registry.get_by_domain(family, operation)


__all__ = [
    "DecisionSimulationPlugin",
    "DecisionSimulationPluginRegistry",
    "DecisionSimulationPluginSpec",
    "DuplicateSimulationPluginError",
    "InvalidSimulationPluginError",
    "SimulationPluginRegistryError",
    "UnknownSimulationPluginError",
    "default_simulation_plugin_registry",
    "get_decision_simulation_plugin",
    "get_decision_simulation_plugin_for_domain",
    "register_decision_simulation_plugin",
]
