from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from agi_ecom_app.decision_simulation.engine import (
    ADVERTISING_BOOTSTRAP_MODEL_VERSION,
    DecisionSimulationEngine,
    simulation_runtime_card,
    validate_advertising_pause_candidate,
)
from agi_ecom_app.decision_simulation.plugins import (
    DecisionSimulationPlugin,
    DecisionSimulationPluginRegistry,
    DecisionSimulationPluginSpec,
    DuplicateSimulationPluginError,
    UnknownSimulationPluginError,
    default_simulation_plugin_registry,
)
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    PricingSimulationEngine,
    validate_pricing_candidate,
)


def _advertising_candidates(decision: Any) -> list[Any]:
    """Translate a deterministic ManagerDecision into simulation candidates.

    Imports the manager workflow adapter only when candidate construction is
    requested.  Importing this module therefore remains safe from both
    ``service.py`` and ``jobs.py``.
    """

    from agi_ecom_app.decision_simulation.service import (
        advertising_candidate_from_action,
    )

    actions = list(getattr(decision, "ranked_actions", ()))
    if not 3 <= len(actions) <= 7:
        return []
    if any(
        getattr(action, "family", None) != "advertising"
        or getattr(action, "operation", None) != "pause_campaign"
        for action in actions
    ):
        return []
    return [advertising_candidate_from_action(action) for action in actions]


def _pricing_candidates(
    *,
    nm_id: int,
    current_price_rub: float,
    candidate_prices: Sequence[float] | None = None,
) -> list[Any]:
    from agi_ecom_app.decision_simulation.pricing_source import (
        build_pricing_candidates,
    )

    return build_pricing_candidates(
        nm_id=nm_id,
        current_price_rub=current_price_rub,
        candidate_prices=candidate_prices,
    )


def _validate_advertising_candidate(request: Any, candidate: Any | None = None) -> list[Any]:
    """Worker-compatible validation callback.

    Advertising validation does not need request-level evidence, but accepting
    ``(request, candidate)`` gives every built-in plugin the same worker call
    shape.  The one-argument form is retained for service hard-gate use.
    """

    target = request if candidate is None else candidate
    return validate_advertising_pause_candidate(target)


def _validate_pricing_candidate(request: Any, candidate: Any) -> list[Any]:
    return validate_pricing_candidate(request, candidate)


def _advertising_engine() -> DecisionSimulationEngine:
    return DecisionSimulationEngine()


def _pricing_engine() -> PricingSimulationEngine:
    return PricingSimulationEngine()


def _run_critic(**kwargs: Any) -> Any:
    from agi_ecom_app.decision_simulation.critic import run_decision_critic

    return run_decision_critic(**kwargs)


def _project_advertising_result(
    run: Any,
    ranked: Sequence[Any],
) -> None:
    """Project an advertising terminal result using the existing durable path."""

    from agi_ecom_app.decision_simulation.jobs import (
        _publish_enforced_decision,
        _publish_shadow_decision,
    )

    if run.mode == "shadow":
        _publish_shadow_decision(run)
        return
    if run.mode == "enforced":
        _publish_enforced_decision(run, ranked)
        return
    raise ValueError(f"unsupported decision simulation mode: {run.mode}")


def _project_pricing_result(
    run: Any,
    ranked: Sequence[Any],
) -> None:
    """Pricing v1 is intentionally non-executable and projects shadow only."""

    del ranked
    if run.mode != "shadow":
        raise ValueError("pricing-elasticity-bootstrap-v1 is shadow-only")
    from agi_ecom_app.decision_simulation.jobs import _publish_shadow_decision

    _publish_shadow_decision(run)


def _refresh_advertising_snapshot(run: Any) -> dict[str, Any]:
    # This is a lazy bridge while the durable refresh implementation still
    # lives in the job module.  It avoids a module-import cycle and keeps the
    # plugin callback immediately usable by the orchestrator.
    from agi_ecom_app.decision_simulation.jobs import _refreshed_snapshot

    return _refreshed_snapshot(run)


def _refresh_pricing_snapshot(run: Any) -> dict[str, Any]:
    """Reload the same SKU and price grid into a new immutable snapshot."""

    from agi_ecom_app.decision_simulation.pricing_source import (
        load_pricing_simulation_source,
        pricing_request_snapshot_payload,
    )

    snapshot = dict(run.evidence_snapshot)
    source = snapshot.get("pricingSource")
    request_payload = snapshot.get("simulationRequest")
    if not isinstance(source, dict) or not isinstance(request_payload, dict):
        raise ValueError("pricing simulation snapshot is missing source/request payload")
    raw_nm_id = source.get("nmId", request_payload.get("skuId"))
    try:
        nm_id = int(raw_nm_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("pricing simulation snapshot has no valid nmId") from exc

    raw_candidates = request_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("pricing simulation snapshot has no candidates")
    candidate_prices: list[float] = []
    for payload in raw_candidates:
        if not isinstance(payload, dict) or payload.get("operation") != "set_price":
            continue
        raw_price = payload.get("priceRub", payload.get("price_rub"))
        try:
            candidate_prices.append(float(raw_price))
        except (TypeError, ValueError) as exc:
            raise ValueError("pricing candidate has an invalid price") from exc

    horizon_days = int(request_payload.get("horizonDays", 7))
    request, source_metadata = load_pricing_simulation_source(
        nm_id=nm_id,
        candidate_prices=candidate_prices,
        horizon_days=horizon_days,
    )
    snapshot["pricingSource"] = source_metadata
    snapshot["simulationRequest"] = pricing_request_snapshot_payload(request)
    snapshot["simulationCandidates"] = [
        candidate.model_dump(mode="json", by_alias=True)
        for candidate in request.candidates
    ]

    manager_payload = snapshot.get("managerDecision")
    if isinstance(manager_payload, dict):
        refreshed_manager = dict(manager_payload)
        refreshed_manager["createdAt"] = datetime.now(timezone.utc).isoformat()
        current_price = source_metadata.get("currentPriceSnapshot")
        if isinstance(current_price, dict) and current_price.get("last_synced_at"):
            refreshed_manager["evidenceAsOf"] = current_price["last_synced_at"]
        snapshot["managerDecision"] = refreshed_manager
    return snapshot


ADVERTISING_PAUSE_PLUGIN = DecisionSimulationPluginSpec(
    model_version=ADVERTISING_BOOTSTRAP_MODEL_VERSION,
    family="advertising",
    operation="pause_campaign",
    build_candidates=_advertising_candidates,
    validate_candidate=_validate_advertising_candidate,
    create_engine=_advertising_engine,
    build_runtime_card=simulation_runtime_card,
    run_critic=_run_critic,
    project_result=_project_advertising_result,
    refresh_snapshot=_refresh_advertising_snapshot,
)

PRICING_SET_PRICE_PLUGIN = DecisionSimulationPluginSpec(
    model_version=PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    family="pricing",
    operation="set_price",
    build_candidates=_pricing_candidates,
    validate_candidate=_validate_pricing_candidate,
    create_engine=_pricing_engine,
    build_runtime_card=simulation_runtime_card,
    run_critic=_run_critic,
    project_result=_project_pricing_result,
    refresh_snapshot=_refresh_pricing_snapshot,
)

BUILTIN_SIMULATION_PLUGINS: tuple[DecisionSimulationPlugin, ...] = (
    ADVERTISING_PAUSE_PLUGIN,
    PRICING_SET_PRICE_PLUGIN,
)


def ensure_builtin_simulation_plugins_registered(
    registry: DecisionSimulationPluginRegistry = default_simulation_plugin_registry,
) -> tuple[DecisionSimulationPlugin, ...]:
    """Idempotently install the exact built-ins without masking conflicts."""

    installed: list[DecisionSimulationPlugin] = []
    for plugin in BUILTIN_SIMULATION_PLUGINS:
        try:
            current = registry.get_by_model_version(plugin.model_version)
        except UnknownSimulationPluginError:
            current = None
        if current is not None:
            if current is not plugin:
                raise DuplicateSimulationPluginError(
                    "simulation model version is registered by a different plugin: "
                    f"{plugin.model_version}"
                )
            domain_plugin = registry.get_by_domain(plugin.family, plugin.operation)
            if domain_plugin is not plugin:
                raise DuplicateSimulationPluginError(
                    "simulation domain is registered by a different plugin: "
                    f"family={plugin.family}, operation={plugin.operation}"
                )
            installed.append(plugin)
            continue

        try:
            domain_plugin = registry.get_by_domain(plugin.family, plugin.operation)
        except UnknownSimulationPluginError:
            domain_plugin = None
        if domain_plugin is not None:
            raise DuplicateSimulationPluginError(
                "simulation domain is registered by a different plugin: "
                f"family={plugin.family}, operation={plugin.operation}"
            )

        try:
            registry.register(plugin)
        except DuplicateSimulationPluginError:
            # Another initializer can win between lookup and registration.
            # Treat it as idempotent only when both indexes now contain this
            # exact immutable built-in object.
            if (
                registry.get_by_model_version(plugin.model_version) is not plugin
                or registry.get_by_domain(plugin.family, plugin.operation) is not plugin
            ):
                raise
        installed.append(plugin)
    return tuple(installed)


__all__ = [
    "ADVERTISING_PAUSE_PLUGIN",
    "BUILTIN_SIMULATION_PLUGINS",
    "PRICING_SET_PRICE_PLUGIN",
    "ensure_builtin_simulation_plugins_registered",
]
