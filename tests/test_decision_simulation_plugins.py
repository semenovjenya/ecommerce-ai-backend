from __future__ import annotations

import pytest

from agi_ecom_app.decision_simulation.plugins import (
    DecisionSimulationPlugin,
    DecisionSimulationPluginRegistry,
    DecisionSimulationPluginSpec,
    DuplicateSimulationPluginError,
    InvalidSimulationPluginError,
    UnknownSimulationPluginError,
)


def _callback(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


def _plugin(
    *,
    model_version: str = "advertising-bootstrap-v1",
    family: str = "advertising",
    operation: str = "pause_campaign",
) -> DecisionSimulationPluginSpec:
    return DecisionSimulationPluginSpec(
        model_version=model_version,
        family=family,
        operation=operation,
        build_candidates=_callback,
        validate_candidate=_callback,
        create_engine=_callback,
        build_runtime_card=_callback,
        run_critic=_callback,
        project_result=_callback,
        refresh_snapshot=_callback,
    )


def test_registers_and_resolves_the_same_plugin_by_both_keys() -> None:
    registry = DecisionSimulationPluginRegistry()
    plugin = _plugin(family=" Advertising ", operation=" PAUSE_CAMPAIGN ")

    assert registry.register(plugin) is plugin
    assert registry.get_by_model_version(" advertising-bootstrap-v1 ") is plugin
    assert registry.get_by_domain("advertising", "pause_campaign") is plugin
    assert registry.get_by_domain("ADVERTISING", "Pause_Campaign") is plugin
    assert registry.registered_model_versions() == ("advertising-bootstrap-v1",)
    assert registry.registered_domains() == (("advertising", "pause_campaign"),)


def test_callable_spec_satisfies_the_runtime_protocol() -> None:
    assert isinstance(_plugin(), DecisionSimulationPlugin)


def test_unknown_model_version_and_domain_fail_closed() -> None:
    registry = DecisionSimulationPluginRegistry()

    with pytest.raises(
        UnknownSimulationPluginError,
        match="unknown simulation model version: supply-bootstrap-v1",
    ):
        registry.get_by_model_version("supply-bootstrap-v1")

    with pytest.raises(
        UnknownSimulationPluginError,
        match="family=supply, operation=reorder",
    ):
        registry.get_by_domain("supply", "reorder")


def test_duplicate_model_version_does_not_replace_original_plugin() -> None:
    registry = DecisionSimulationPluginRegistry()
    original = _plugin()
    duplicate = _plugin(family="supply", operation="reorder")
    registry.register(original)

    with pytest.raises(
        DuplicateSimulationPluginError,
        match="model version is already registered",
    ):
        registry.register(duplicate)

    assert registry.get_by_model_version(original.model_version) is original
    with pytest.raises(UnknownSimulationPluginError):
        registry.get_by_domain("supply", "reorder")


def test_duplicate_domain_is_rejected_atomically() -> None:
    registry = DecisionSimulationPluginRegistry()
    original = _plugin()
    conflicting = _plugin(model_version="advertising-bootstrap-v2")
    registry.register(original)

    with pytest.raises(
        DuplicateSimulationPluginError,
        match="simulation domain is already registered",
    ):
        registry.register(conflicting)

    assert registry.get_by_domain("advertising", "pause_campaign") is original
    with pytest.raises(UnknownSimulationPluginError):
        registry.get_by_model_version("advertising-bootstrap-v2")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_version", ""),
        ("family", "  "),
        ("operation", ""),
    ],
)
def test_blank_identity_fields_are_rejected(field: str, value: str) -> None:
    registry = DecisionSimulationPluginRegistry()
    values = {
        "model_version": "advertising-bootstrap-v1",
        "family": "advertising",
        "operation": "pause_campaign",
    }
    values[field] = value

    with pytest.raises(InvalidSimulationPluginError, match=field):
        registry.register(_plugin(**values))

    assert registry.registered_model_versions() == ()
    assert registry.registered_domains() == ()


def test_missing_capability_is_rejected_before_registry_mutation() -> None:
    registry = DecisionSimulationPluginRegistry()
    values = {
        field: getattr(_plugin(), field)
        for field in DecisionSimulationPluginSpec.__dataclass_fields__
    }
    values["run_critic"] = None
    invalid = DecisionSimulationPluginSpec(**values)  # type: ignore[arg-type]

    with pytest.raises(
        InvalidSimulationPluginError,
        match="callback must be callable: run_critic",
    ):
        registry.register(invalid)

    assert registry.registered_model_versions() == ()
    assert registry.registered_domains() == ()
