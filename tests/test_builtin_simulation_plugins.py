from __future__ import annotations

from types import SimpleNamespace

import pytest

from agi_ecom_app.decision_simulation.builtin_plugins import (
    ADVERTISING_PAUSE_PLUGIN,
    BUILTIN_SIMULATION_PLUGINS,
    PRICING_SET_PRICE_PLUGIN,
    ensure_builtin_simulation_plugins_registered,
)
from agi_ecom_app.decision_simulation.engine import (
    ADVERTISING_BOOTSTRAP_MODEL_VERSION,
    DecisionSimulationEngine,
    simulation_runtime_card,
)
from agi_ecom_app.decision_simulation.models import (
    AdvertisingPauseCandidateV1,
    AdvertisingPauseEvidenceV1,
)
from agi_ecom_app.decision_simulation.plugins import (
    DecisionSimulationPluginRegistry,
    DecisionSimulationPluginSpec,
    DuplicateSimulationPluginError,
)
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    PricingSimulationEngine,
)


def _callback(*args, **kwargs):
    return args, kwargs


def test_builtin_specs_pin_models_domains_engines_and_runtime() -> None:
    assert BUILTIN_SIMULATION_PLUGINS == (
        ADVERTISING_PAUSE_PLUGIN,
        PRICING_SET_PRICE_PLUGIN,
    )
    assert (
        ADVERTISING_PAUSE_PLUGIN.model_version
        == ADVERTISING_BOOTSTRAP_MODEL_VERSION
    )
    assert (
        ADVERTISING_PAUSE_PLUGIN.family,
        ADVERTISING_PAUSE_PLUGIN.operation,
    ) == ("advertising", "pause_campaign")
    assert isinstance(ADVERTISING_PAUSE_PLUGIN.create_engine(), DecisionSimulationEngine)
    assert ADVERTISING_PAUSE_PLUGIN.build_runtime_card() == simulation_runtime_card()

    assert (
        PRICING_SET_PRICE_PLUGIN.model_version
        == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION
    )
    assert (
        PRICING_SET_PRICE_PLUGIN.family,
        PRICING_SET_PRICE_PLUGIN.operation,
    ) == ("pricing", "set_price")
    assert isinstance(PRICING_SET_PRICE_PLUGIN.create_engine(), PricingSimulationEngine)
    assert PRICING_SET_PRICE_PLUGIN.build_runtime_card() == simulation_runtime_card()


def test_registration_is_idempotent_and_resolves_every_builtin() -> None:
    registry = DecisionSimulationPluginRegistry()

    first = ensure_builtin_simulation_plugins_registered(registry)
    second = ensure_builtin_simulation_plugins_registered(registry)

    assert first == second == BUILTIN_SIMULATION_PLUGINS
    assert registry.registered_model_versions() == (
        ADVERTISING_BOOTSTRAP_MODEL_VERSION,
        PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    )
    assert registry.get_by_domain("advertising", "pause_campaign") is (
        ADVERTISING_PAUSE_PLUGIN
    )
    assert registry.get_by_domain("pricing", "set_price") is PRICING_SET_PRICE_PLUGIN


def test_registration_does_not_mask_a_conflicting_model_plugin() -> None:
    registry = DecisionSimulationPluginRegistry()
    conflicting = DecisionSimulationPluginSpec(
        model_version=ADVERTISING_BOOTSTRAP_MODEL_VERSION,
        family="other",
        operation="other",
        build_candidates=_callback,
        validate_candidate=_callback,
        create_engine=_callback,
        build_runtime_card=_callback,
        run_critic=_callback,
        project_result=_callback,
        refresh_snapshot=_callback,
    )
    registry.register(conflicting)

    with pytest.raises(
        DuplicateSimulationPluginError,
        match="registered by a different plugin",
    ):
        ensure_builtin_simulation_plugins_registered(registry)

    assert registry.get_by_model_version(ADVERTISING_BOOTSTRAP_MODEL_VERSION) is (
        conflicting
    )


def test_advertising_validator_supports_service_and_worker_call_shapes() -> None:
    candidate = AdvertisingPauseCandidateV1(
        candidateId="ad-1",
        actionId="action-1",
        evidence=AdvertisingPauseEvidenceV1(),
    )

    service_violations = ADVERTISING_PAUSE_PLUGIN.validate_candidate(candidate)
    worker_violations = ADVERTISING_PAUSE_PLUGIN.validate_candidate(
        SimpleNamespace(),
        candidate,
    )

    assert [item.code for item in service_violations] == [
        item.code for item in worker_violations
    ]
    assert "missing_daily_spend_history" in {
        item.code for item in service_violations
    }


def test_pricing_candidate_builder_keeps_hold_and_requested_prices() -> None:
    candidates = PRICING_SET_PRICE_PLUGIN.build_candidates(
        nm_id=123,
        current_price_rub=1_000,
        candidate_prices=[900, 950, 1_050],
    )

    assert [item.operation for item in candidates] == [
        "hold_price",
        "set_price",
        "set_price",
        "set_price",
    ]
    assert [item.price_rub for item in candidates] == [1_000, 900, 950, 1_050]


def test_pricing_projection_fails_closed_outside_shadow_before_importing_jobs() -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        PRICING_SET_PRICE_PLUGIN.project_result(
            SimpleNamespace(mode="enforced"),
            [],
        )
