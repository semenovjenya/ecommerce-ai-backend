from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agi_ecom_app.manager_lifecycle import service as lifecycle_service
from agi_ecom_app.manager_lifecycle.auth import AuthorizationError, ManagerActor
from agi_ecom_app.manager_lifecycle.measurement import (
    calculate_measurement,
    compute_difference_in_differences,
)
from agi_ecom_app.manager_lifecycle.service import (
    ActuationResult,
    LiveGuardrailSnapshot,
    PreflightResult,
    approve_manager_action,
    complete_manual_manager_action,
    evaluate_live_guardrails,
    persist_manager_decision_bundle,
    record_manager_action_measurement,
    register_manager_action_actuator,
)
from agi_ecom_app.services.memory import (
    AgentActionExperimentRecord,
    ManagerDecisionRecord,
)


NOW = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)


def _decision() -> ManagerDecisionRecord:
    return ManagerDecisionRecord(
        id="decision-1",
        source_key="chat:1",
        chat_id="chat-1",
        run_id=1,
        decision_date=date(2026, 7, 13),
        objective="incremental_contribution_profit",
        status="ready_for_approval",
        request_coverage={"complete": True},
        evidence_as_of=NOW,
        ranked_actions=[],
        rejected_candidates=[],
        blockers=[],
        decision_json={},
        public_json={},
        created_at=NOW,
        updated_at=NOW,
    )


def _experiment(
    *,
    mode: str = "wb_ads",
    state: str = "proposed",
) -> AgentActionExperimentRecord:
    return AgentActionExperimentRecord(
        id="experiment-1",
        decision_id="decision-1",
        action_id="action-1",
        family="advertising",
        target={
            "operation": "set_bid",
            "payload": {"advert_id": 10, "nm_id": 20},
        },
        execution_mode=mode,
        state=state,
        from_value={"value": 100, "unit": "RUB"},
        to_value={"value": 80, "unit": "RUB"},
        expected_effect={
            "contribution_profit_low_per_day": 10,
            "contribution_profit_base_per_day": 20,
            "contribution_profit_high_per_day": 30,
        },
        horizon={},
        evidence=[{"evidence_ref": "fact-1"}],
        prerequisites=["campaign exists"],
        guardrails={"stock_days_min": 7},
        stop_condition={"instruction": "rollback when contribution falls"},
        rollback={"operation": "set_bid", "to": 100},
        measurement_spec={"evaluation_days": 7},
        expires_at=NOW + timedelta(days=1),
        owner_user_id=None,
        approved_by=None,
        approved_at=None,
        approval_idempotency_key=None,
        rejected_by=None,
        rejected_at=None,
        rejection_idempotency_key=None,
        rejection_reason=None,
        manager_label=None,
        preflight_json={},
        actuator_result={},
        external_operation_id=None,
        applied_at=None,
        measurement_due_at=None,
        close_outcome=None,
        last_error=None,
        version=1,
        created_at=NOW,
        updated_at=NOW,
    )


class ApplyingActuator:
    def __init__(self) -> None:
        self.apply_calls = 0

    def preflight(self, experiment):
        return PreflightResult(
            allowed=True,
            current_value=experiment.from_value,
            observed_at=NOW,
        )

    def apply(self, experiment, *, idempotency_key, actor):
        self.apply_calls += 1
        return ActuationResult(
            status="applying",
            external_operation_id="wb-op-1",
            verification_due_seconds=30,
        )

    def verify(self, experiment, *, idempotency_key):
        return ActuationResult(status="applied", external_operation_id="wb-op-1")


def _enabled_settings():
    return SimpleNamespace(manager_decision_enabled=True)


def test_approval_stays_applying_until_actuator_verifies() -> None:
    proposed = _experiment()
    approved = replace(
        proposed,
        state="approved",
        approved_by=2,
        approval_idempotency_key="approve-1",
    )
    applying = replace(approved, state="applying")
    applying_with_result = replace(
        applying,
        actuator_result={"status": "applying"},
        external_operation_id="wb-op-1",
    )
    actuator = ApplyingActuator()
    register_manager_action_actuator("wb_ads", actuator)
    transitions = iter([(applying, True), (applying_with_result, True)])
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=proposed,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_decision",
            return_value=_decision(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_run_trace_by_id",
            return_value=SimpleNamespace(status="completed"),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service._utc_now",
            return_value=NOW,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.claim_agent_action_experiment_approval",
            return_value=(approved, True, False),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.transition_agent_action_experiment",
            side_effect=lambda **_: next(transitions),
        ),
        patch("agi_ecom_app.manager_lifecycle.service.append_agent_action_measurement") as append,
    ):
        result = approve_manager_action(
            experiment_id=proposed.id,
            actor=ManagerActor(user_id=2, username="manager", role="manager"),
            idempotency_key="approve-1",
        )

    assert result.state == "applying"
    assert result.external_operation_id == "wb-op-1"
    assert actuator.apply_calls == 1
    append.assert_not_called()


def test_idempotent_approval_replay_does_not_call_actuator() -> None:
    proposed = _experiment()
    applying = replace(
        proposed,
        state="applying",
        approved_by=2,
        approval_idempotency_key="approve-1",
    )
    actuator = ApplyingActuator()
    register_manager_action_actuator("wb_ads", actuator)
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=proposed,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_decision",
            return_value=_decision(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_run_trace_by_id",
            return_value=SimpleNamespace(status="completed"),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service._utc_now",
            return_value=NOW,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.claim_agent_action_experiment_approval",
            return_value=(applying, False, True),
        ),
    ):
        result = approve_manager_action(
            experiment_id=proposed.id,
            actor=ManagerActor(user_id=2, username="manager", role="manager"),
            idempotency_key="approve-1",
        )
    assert result.state == "applying"
    assert actuator.apply_calls == 0


def test_approval_requires_completed_parent_agent_run() -> None:
    proposed = _experiment()
    claim = patch(
        "agi_ecom_app.manager_lifecycle.service.claim_agent_action_experiment_approval"
    )
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=proposed,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_decision",
            return_value=_decision(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_run_trace_by_id",
            return_value=SimpleNamespace(status="failed"),
        ),
        claim as claim_approval,
    ):
        with pytest.raises(
            lifecycle_service.ManagerLifecycleConflict,
            match="parent agent run is not completed",
        ):
            approve_manager_action(
                experiment_id=proposed.id,
                actor=ManagerActor(user_id=2, username="manager", role="manager"),
                idempotency_key="approve-parent-failed",
            )

    claim_approval.assert_not_called()


def test_viewer_cannot_approve() -> None:
    with patch(
        "agi_ecom_app.manager_lifecycle.service.get_settings",
        return_value=_enabled_settings(),
    ):
        with pytest.raises(AuthorizationError):
            approve_manager_action(
                experiment_id="experiment-1",
                actor=ManagerActor(user_id=1, username="viewer", role="viewer"),
                idempotency_key="approve-1",
            )


def test_manual_supply_measurement_starts_from_actual_delivery_time() -> None:
    delivered_at = NOW - timedelta(hours=1)
    approved = replace(
        _experiment(mode="human_task", state="approved"),
        family="supply",
        approved_by=2,
    )
    applied = replace(approved, state="applied", applied_at=delivered_at)
    measuring = replace(
        applied,
        state="measuring",
        measurement_due_at=delivered_at + timedelta(days=14),
    )
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_settings",
            return_value=_enabled_settings(),
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=approved,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.transition_agent_action_experiment",
            return_value=(applied, True),
        ) as transition,
        patch(
            "agi_ecom_app.manager_lifecycle.service._schedule_measurement",
            return_value=measuring,
        ) as schedule,
    ):
        result = complete_manual_manager_action(
            experiment_id=approved.id,
            actor=ManagerActor(user_id=2, username="manager", role="manager"),
            idempotency_key="delivery-1",
            actual_at=delivered_at,
        )

    assert result.state == "measuring"
    assert transition.call_args.kwargs["applied_at"] == delivered_at
    schedule.assert_called_once_with(applied, delivered_at)


def test_advertising_measurement_schedule_waits_for_full_utc_evaluation_days() -> None:
    applied_at = datetime(2026, 7, 23, 15, 45, tzinfo=timezone.utc)
    evaluation_start = datetime(2026, 7, 24, tzinfo=timezone.utc)
    evaluation_end = datetime(2026, 7, 31, tzinfo=timezone.utc)
    applied = replace(
        _experiment(state="applied"),
        applied_at=applied_at,
        measurement_spec={"baselineDays": 7, "evaluationDays": 7},
    )
    measuring = replace(
        applied,
        state="measuring",
        measurement_due_at=evaluation_end,
    )

    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.append_agent_action_measurement"
        ) as append_measurement,
        patch(
            "agi_ecom_app.manager_lifecycle.service.transition_agent_action_experiment",
            return_value=(measuring, True),
        ) as transition,
        patch(
            "agi_ecom_app.manager_lifecycle.service._schedule_measurement_safely"
        ) as schedule,
    ):
        result = lifecycle_service._schedule_measurement(applied, applied_at)

    assert result is measuring
    assert append_measurement.call_args.kwargs["period_start"] == evaluation_start
    assert append_measurement.call_args.kwargs["period_end"] == evaluation_end
    assert transition.call_args.kwargs["measurement_due_at"] == evaluation_end
    schedule.assert_called_once_with(measuring, due_at=evaluation_end)


def test_difference_in_differences_and_waiting_for_data() -> None:
    did = compute_difference_in_differences(
        baseline_treated=100,
        post_treated=130,
        baseline_control=80,
        post_control=90,
    )
    assert did.observed_delta == 30
    assert did.did_effect == 20
    assert did.causal_valid is True

    waiting = calculate_measurement(
        data_complete=False,
        evaluation_end=NOW - timedelta(days=1),
        data_as_of=NOW,
        baseline_treated=0,
        post_treated=0,
        now=NOW,
    )
    assert waiting.status == "waiting_for_data"
    assert waiting.observed_delta is None


def test_zero_is_valid_only_after_source_confirms_complete_window() -> None:
    result = calculate_measurement(
        data_complete=True,
        evaluation_end=NOW - timedelta(days=1),
        data_as_of=NOW,
        baseline_treated=0,
        post_treated=0,
        forecast_low=-1,
        forecast_base=1,
        forecast_high=2,
        now=NOW,
    )
    assert result.status == "computed"
    assert result.observed_delta == 0
    assert result.causal_valid is False
    assert result.signed_bias == -1


def test_measurement_reads_public_camel_case_forecast_range() -> None:
    experiment = replace(
        _experiment(state="measuring"),
        expected_effect={
            "contributionProfitLowPerDay": 10,
            "contributionProfitBasePerDay": 20,
            "contributionProfitHighPerDay": 30,
        },
    )
    sentinel = object()
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=experiment,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.append_agent_action_measurement",
            return_value=sentinel,
        ) as append,
        patch("agi_ecom_app.manager_lifecycle.service.transition_agent_action_experiment"),
    ):
        result = record_manager_action_measurement(
            experiment_id=experiment.id,
            evaluation_start=NOW - timedelta(days=8),
            evaluation_end=NOW - timedelta(days=1),
            data_as_of=NOW,
            data_complete=True,
            baseline_treated=100,
            post_treated=125,
            now=NOW,
        )

    assert result is sentinel
    assert append.call_args.kwargs["forecast_low"] == 10
    assert append.call_args.kwargs["forecast_base"] == 20
    assert append.call_args.kwargs["forecast_high"] == 30
    assert append.call_args.kwargs["signed_bias"] == 5


def test_measurement_preserves_provider_and_calculation_reasons_separately() -> None:
    experiment = replace(_experiment(state="measuring"))
    sentinel = object()
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_agent_action_experiment",
            return_value=experiment,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.append_agent_action_measurement",
            return_value=sentinel,
        ) as append,
    ):
        result = record_manager_action_measurement(
            experiment_id=experiment.id,
            evaluation_start=NOW - timedelta(days=8),
            evaluation_end=NOW - timedelta(days=1),
            data_as_of=None,
            data_complete=False,
            baseline_treated=None,
            post_treated=None,
            metrics={"reason": "missing_sku_days"},
            idempotency_key="measurement-readout-1",
            now=NOW,
        )

    assert result is sentinel
    persisted = append.call_args.kwargs
    assert persisted["idempotency_key"] == "measurement-readout-1"
    assert persisted["metrics"]["reason"] == "missing_sku_days"
    assert persisted["metrics"]["providerReason"] == "missing_sku_days"
    assert persisted["metrics"]["calculationReason"] == "source_lag"


def test_measurement_schema_has_append_only_idempotency_boundary() -> None:
    from agi_ecom_app.services.memory import MEMORY_SCHEMA_SQL

    normalized = " ".join(MEMORY_SCHEMA_SQL.split()).lower()
    assert "add column if not exists idempotency_key text" in normalized
    assert "agent_action_measurements_idempotency_idx" in normalized
    assert "on agent_action_measurements(experiment_id, idempotency_key)" in normalized


def test_live_guardrails_require_exact_fresh_metric_values() -> None:
    experiment = replace(
        _experiment(),
        guardrails={
            "rules": [
                {
                    "metric": "stock_on_hand",
                    "operator": "gte",
                    "threshold": {"value": 21, "unit": "units"},
                },
                {
                    "metric": "contribution_profit",
                    "operator": "gte",
                    "threshold": {"value": 0, "unit": "RUB"},
                },
            ]
        },
    )
    safe = LiveGuardrailSnapshot(
        observed_at=NOW,
        observations={
            "stock_on_hand": {"value": 30, "unit": "units"},
            "contribution_profit": {"value": 100, "unit": "RUB"},
        },
    )
    breached = replace(
        safe,
        observations={
            **safe.observations,
            "stock_on_hand": {"value": 20, "unit": "units"},
        },
    )

    assert evaluate_live_guardrails(experiment, safe) == ()
    assert evaluate_live_guardrails(experiment, breached) == (
        "live guardrail stock_on_hand breached",
    )


def test_default_wb_actuator_blocks_when_live_guardrail_provider_is_missing() -> None:
    experiment = replace(
        _experiment(),
        guardrails={"rules": [{"metric": "stock_on_hand"}]},
    )
    actuator = lifecycle_service._WBExecutorActuator.__new__(
        lifecycle_service._WBExecutorActuator
    )
    actuator._executor = SimpleNamespace(
        preflight=lambda _: {"current_value": 100, "status": "active"}
    )

    with patch.dict(lifecycle_service._LIVE_GUARDRAIL_PROVIDERS, {}, clear=True):
        result = actuator.preflight(experiment)

    assert result.allowed is False
    assert result.current_value == experiment.from_value
    assert result.blockers == ("live execution guardrail provider is not configured",)


def test_persistence_adapter_maps_manager_core_contract() -> None:
    payload = {
        "decision_id": "decision-1",
        "query_mode": "decision",
        "objective": "incremental_contribution_profit",
        "status": "ready_for_approval",
        "created_at": NOW,
        "evidence_as_of": NOW,
        "request_coverage": {"complete": True},
        "ranked_actions": [
            {
                "action_id": "action-1",
                "family": "advertising",
                "target": "campaign:10",
                "target_type": "campaign",
                "target_refs": ["10"],
                "operation": "set_bid",
                "execution_mode": "wb_api",
                "execution_payload": {"advert_id": 10, "nm_id": 20},
                "from": {"value": 100, "unit": "RUB"},
                "to": {"value": 80, "unit": "RUB"},
                "expected_effect": {
                    "contributionProfitLowPerDay": 10,
                    "contributionProfitBasePerDay": 20,
                    "contributionProfitHighPerDay": 30,
                    "horizonDays": 7,
                },
                "economics": {
                    "deltaRevenuePerDay": {"low": 10, "base": 20, "high": 30}
                },
                "evidence": [{"evidence_ref": "fact-1"}],
                "prerequisites": ["campaign exists"],
                "stop_condition": "rollback on guardrail breach",
                "rollback": {"operation": "set_bid"},
                "measurement_spec": {"evaluation_days": 7},
                "expires_at": NOW + timedelta(days=1),
            }
        ],
        "rejected_candidates": [],
        "blockers": [],
    }
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_decision_by_source_key",
            return_value=None,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.upsert_manager_decision",
            return_value=_decision(),
        ) as save_decision,
        patch(
            "agi_ecom_app.manager_lifecycle.service.upsert_agent_action_experiment",
            return_value=_experiment(),
        ) as save_action,
    ):
        persist_manager_decision_bundle(payload)

    assert save_decision.call_args.kwargs["decision_id"] == "decision-1"
    action_kwargs = save_action.call_args.kwargs
    assert action_kwargs["execution_mode"] == "wb_ads"
    assert action_kwargs["target"]["operation"] == "set_bid"
    assert action_kwargs["target"]["payload"]["advert_id"] == 10
    assert action_kwargs["stop_condition"]["instruction"].startswith("rollback")
    assert action_kwargs["horizon"] == {"days": 7}
    assert action_kwargs["expected_effect"]["contributionProfitLowPerDay"] == 10
    assert action_kwargs["expected_effect"]["economics"]["deltaRevenuePerDay"]["base"] == 20


def test_duplicate_source_key_returns_stored_bundle_without_rewriting_actions() -> None:
    existing = _decision()
    actions = [_experiment()]
    with (
        patch(
            "agi_ecom_app.manager_lifecycle.service.get_manager_decision_by_source_key",
            return_value=existing,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.list_agent_action_experiments_for_decision",
            return_value=actions,
        ),
        patch(
            "agi_ecom_app.manager_lifecycle.service.upsert_manager_decision"
        ) as write_decision,
        patch(
            "agi_ecom_app.manager_lifecycle.service.upsert_agent_action_experiment"
        ) as write_action,
    ):
        result = persist_manager_decision_bundle(
            {"decision_id": "different", "status": "blocked"},
            source_key="chat:1",
        )
    assert result == (existing, actions)
    write_decision.assert_not_called()
    write_action.assert_not_called()
