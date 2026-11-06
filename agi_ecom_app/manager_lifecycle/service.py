from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Protocol

from agi_ecom_app.config import get_settings
from agi_ecom_app.manager_lifecycle.auth import ManagerActor, require_role
from agi_ecom_app.manager_lifecycle.measurement import calculate_measurement
from agi_ecom_app.services.memory import (
    AgentActionExperimentRecord,
    AgentActionMeasurementRecord,
    ManagerDecisionRecord,
    append_agent_action_measurement,
    claim_agent_action_experiment_approval,
    get_agent_action_experiment,
    get_agent_run_trace_by_id,
    get_decision_simulation_run,
    get_manager_decision,
    get_manager_decision_by_source_key,
    list_agent_action_experiments_for_decision,
    list_manager_decisions_for_date,
    reject_agent_action_experiment,
    transition_agent_action_experiment,
    upsert_agent_action_experiment,
    upsert_manager_decision,
)


logger = logging.getLogger(__name__)


class ManagerLifecycleError(RuntimeError):
    pass


class ManagerLifecycleNotFound(ManagerLifecycleError):
    pass


class ManagerLifecycleConflict(ManagerLifecycleError):
    pass


class ManagerFeatureDisabled(ManagerLifecycleError):
    pass


@dataclass(frozen=True)
class PreflightResult:
    allowed: bool
    current_value: Any = None
    observed_at: datetime | None = None
    blockers: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActuationResult:
    status: Literal["applying", "applied", "blocked", "failed", "unknown"]
    external_operation_id: str | None = None
    applied_value: Any = None
    rollback_payload: dict[str, Any] = field(default_factory=dict)
    verification_due_seconds: int | None = None
    detail: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ManagerActionActuator(Protocol):
    def preflight(self, experiment: AgentActionExperimentRecord) -> PreflightResult: ...

    def apply(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        idempotency_key: str,
        actor: ManagerActor,
    ) -> ActuationResult: ...


@dataclass(frozen=True)
class LiveGuardrailSnapshot:
    observed_at: datetime
    observations: dict[str, Any]
    blockers: tuple[str, ...] = ()


class ManagerLiveGuardrailProvider(Protocol):
    def collect(self, experiment: AgentActionExperimentRecord) -> LiveGuardrailSnapshot: ...

    def verify(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        idempotency_key: str,
    ) -> ActuationResult: ...


_ACTUATORS: dict[str, ManagerActionActuator] = {}
_LIVE_GUARDRAIL_PROVIDERS: dict[str, ManagerLiveGuardrailProvider] = {}


def register_manager_action_actuator(
    execution_mode: str,
    actuator: ManagerActionActuator,
) -> None:
    if execution_mode not in {"wb_ads", "wb_price"}:
        raise ValueError("only WB execution modes accept an external actuator")
    _ACTUATORS[execution_mode] = actuator


def register_manager_live_guardrail_provider(
    execution_mode: str,
    provider: ManagerLiveGuardrailProvider,
) -> None:
    if execution_mode not in {"wb_ads", "wb_price"}:
        raise ValueError("live guardrails are supported only for WB execution modes")
    _LIVE_GUARDRAIL_PROVIDERS[execution_mode] = provider


def persist_manager_decision_bundle(
    decision: Any,
    *,
    chat_id: str | None = None,
    run_id: int | None = None,
    source_key: str | None = None,
    create_experiments: bool = True,
) -> tuple[ManagerDecisionRecord, list[AgentActionExperimentRecord]]:
    """Store a ManagerDecisionV1-like model without coupling to its package.

    Both camelCase public payloads and snake_case internal payloads are accepted,
    allowing the manager core to evolve independently while persistence remains
    backward compatible.
    """
    payload = _as_mapping(decision)
    if source_key:
        existing = get_manager_decision_by_source_key(source_key)
        if existing is not None:
            actions = list_agent_action_experiments_for_decision(existing.id)
            return existing, [
                _sync_program_action_link(
                    action,
                    expected_program_id=getattr(existing, "experiment_program_id", None),
                )
                for action in actions
            ]
    decision_id = str(_pick(payload, "id", "decisionId", "decision_id") or uuid.uuid4())
    objective = str(_pick(payload, "objective") or "incremental_contribution_profit")
    status = str(_pick(payload, "status") or "blocked")
    if status not in {"evaluating", "ready_for_approval", "needs_data", "blocked"}:
        raise ValueError(f"unsupported manager decision status: {status}")
    ranked_actions = _mapping_list(
        _pick(payload, "rankedActions", "ranked_actions", default=[])
    )
    record = upsert_manager_decision(
        decision_id=decision_id,
        source_key=source_key,
        chat_id=chat_id,
        run_id=run_id,
        decision_date=_parse_date(_pick(payload, "decisionDate", "decision_date")),
        objective=objective,
        status=status,
        request_coverage=dict(
            _pick(payload, "requestCoverage", "request_coverage", default={}) or {}
        ),
        evidence_as_of=_parse_datetime(
            _pick(payload, "evidenceAsOf", "evidence_as_of")
        ),
        ranked_actions=ranked_actions,
        rejected_candidates=_mapping_list(
            _pick(payload, "rejectedCandidates", "rejected_candidates", default=[])
        ),
        blockers=list(_pick(payload, "blockers", default=[]) or []),
        decision_json=payload,
        public_json=dict(_pick(payload, "public", "publicJson", default=payload) or payload),
    )
    if source_key and record.decision_json != payload:
        actions = list_agent_action_experiments_for_decision(record.id)
        return record, [
            _sync_program_action_link(
                action,
                expected_program_id=getattr(record, "experiment_program_id", None),
            )
            for action in actions
        ]

    if not create_experiments:
        return record, []

    experiments: list[AgentActionExperimentRecord] = []
    for index, action in enumerate(ranked_actions):
        action_id = str(_pick(action, "actionId", "action_id", "id") or f"action-{index + 1}")
        experiment_id = str(
            _pick(action, "experimentId", "experiment_id")
            or uuid.uuid5(uuid.NAMESPACE_URL, f"manager:{record.id}:{action_id}")
        )
        family = str(_pick(action, "family") or "unknown")
        execution_mode = _normalize_execution_mode(
            _pick(action, "executionMode", "execution_mode"), family
        )
        target_value = _pick(action, "target", default={})
        execution_payload = dict(
            _pick(action, "executionPayload", "execution_payload", default={}) or {}
        )
        if isinstance(target_value, Mapping):
            target_payload = dict(target_value)
        else:
            target_payload = {"label": str(target_value or action_id)}
        target_payload.update(
            {
                "target_type": _pick(action, "targetType", "target_type"),
                "target_refs": list(
                    _pick(action, "targetRefs", "target_refs", default=[]) or []
                ),
                "operation": _pick(action, "operation"),
                "payload": execution_payload,
            }
        )
        stop_condition = _pick(
            action, "stopCondition", "stop_condition", default={}
        )
        if not isinstance(stop_condition, Mapping):
            stop_condition = {"instruction": str(stop_condition or "")}
        expected_effect = dict(
            _pick(action, "expectedEffect", "expected_effect", default={}) or {}
        )
        economics = _pick(action, "economics", "actionEconomics", "action_economics")
        if isinstance(economics, Mapping):
            expected_effect["economics"] = dict(economics)
        horizon = dict(_pick(action, "horizon", default={}) or {})
        if not horizon:
            horizon_days = _pick(expected_effect, "horizonDays", "horizon_days")
            if horizon_days is not None:
                horizon = {"days": horizon_days}
        raw_guardrails = _pick(action, "guardrails", default={}) or {}
        if isinstance(raw_guardrails, Mapping):
            guardrails = dict(raw_guardrails)
        elif isinstance(raw_guardrails, list):
            guardrails = {"rules": list(raw_guardrails)}
        else:
            raise ValueError("manager action guardrails must be an object or list")
        stored_experiment = upsert_agent_action_experiment(
            experiment_id=experiment_id,
            decision_id=record.id,
            action_id=action_id,
            family=family,
            execution_mode=execution_mode,
            target=target_payload,
            from_value=_pick(
                action, "from", "fromValue", "from_value", "currentValue"
            ),
            to_value=_pick(action, "to", "toValue", "to_value", "proposedValue"),
            expected_effect=expected_effect,
            horizon=horizon,
            evidence=list(
                _pick(action, "evidence", "evidenceRefs", default=[]) or []
            ),
            prerequisites=list(_pick(action, "prerequisites", default=[]) or []),
            guardrails=guardrails,
            stop_condition=dict(stop_condition),
            rollback=dict(_pick(action, "rollback", default={}) or {}),
            measurement_spec=dict(
                _pick(action, "measurementSpec", "measurement_spec", default={}) or {}
            ),
            expires_at=_parse_datetime(_pick(action, "expiresAt", "expires_at")),
        )
        experiments.append(
            _sync_program_action_link(
                stored_experiment,
                expected_program_id=getattr(record, "experiment_program_id", None),
            )
        )
    return record, experiments


def get_decision_bundle(
    decision_id: str,
) -> tuple[ManagerDecisionRecord, list[AgentActionExperimentRecord]]:
    decision = get_manager_decision(decision_id)
    if decision is None:
        raise ManagerLifecycleNotFound("manager decision not found")
    return decision, list_agent_action_experiments_for_decision(decision_id)


def get_today_decision_bundles(
    *,
    today: date,
) -> list[tuple[ManagerDecisionRecord, list[AgentActionExperimentRecord]]]:
    return [
        (decision, list_agent_action_experiments_for_decision(decision.id))
        for decision in list_manager_decisions_for_date(today)
    ]


def get_manager_action(experiment_id: str) -> AgentActionExperimentRecord:
    experiment = get_agent_action_experiment(experiment_id)
    if experiment is None:
        raise ManagerLifecycleNotFound("manager action not found")
    return experiment


def _sync_program_action_link(
    experiment: AgentActionExperimentRecord,
    *,
    expected_program_id: str | None,
) -> AgentActionExperimentRecord:
    """Repair the optional program link without changing the legacy path."""

    if expected_program_id is None:
        return experiment
    from agi_ecom_app.experiment_program.lifecycle_adapter import (
        ExperimentProgramActionLinkConflict,
        ExperimentProgramActionLinkNotFound,
        sync_action_experiment_link,
    )

    try:
        sync_action_experiment_link(
            experiment.id,
            expected_program_id=expected_program_id,
        )
    except ExperimentProgramActionLinkNotFound:
        raise ManagerLifecycleNotFound("manager action not found") from None
    except ExperimentProgramActionLinkConflict as exc:
        raise ManagerLifecycleConflict(str(exc)) from None
    refreshed = get_agent_action_experiment(experiment.id)
    if refreshed is None:
        raise ManagerLifecycleNotFound("manager action not found")
    return refreshed


def approve_manager_action(
    *,
    experiment_id: str,
    actor: ManagerActor,
    idempotency_key: str,
) -> AgentActionExperimentRecord:
    _ensure_manager_feature_enabled()
    require_role(actor, "manager")
    key = _require_idempotency_key(idempotency_key)
    experiment = get_manager_action(experiment_id)
    decision = get_manager_decision(experiment.decision_id)
    if decision is None or decision.status != "ready_for_approval":
        raise ManagerLifecycleConflict("decision is not ready for approval")
    experiment = _sync_program_action_link(
        experiment,
        expected_program_id=getattr(decision, "experiment_program_id", None),
    )
    parent_run_id = getattr(decision, "run_id", None)
    if parent_run_id is not None:
        parent_run = get_agent_run_trace_by_id(parent_run_id)
        if parent_run is None or parent_run.status != "completed":
            raise ManagerLifecycleConflict(
                "parent agent run is not completed"
            )
    if decision.simulation_run_id:
        simulation = get_decision_simulation_run(decision.simulation_run_id)
        if simulation is not None and simulation.mode == "enforced":
            critic_result = simulation.critic_result.get("result")
            critic_verdict = (
                critic_result.get("verdict")
                if isinstance(critic_result, Mapping)
                else simulation.critic_result.get("verdict")
            )
            if (
                simulation.status != "completed"
                or not simulation.winner_candidate_id
                or critic_verdict != "approved"
            ):
                raise ManagerLifecycleConflict(
                    "enforced decision simulation is not completed and approved"
                )
    if experiment.expires_at and _aware_utc(experiment.expires_at) <= _utc_now():
        expired, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"proposed"},
            state="blocked",
            event="expired",
            actor_user_id=actor.user_id,
            idempotency_key=key,
            last_error="action expired before approval",
        )
        if expired is not None:
            _sync_program_action_link(
                expired,
                expected_program_id=getattr(expired, "experiment_program_id", None),
            )
        raise ManagerLifecycleConflict("manager action has expired")
    _validate_exact_action_contract(experiment)

    claimed, changed, replay = claim_agent_action_experiment_approval(
        experiment_id=experiment.id,
        actor_user_id=actor.user_id,
        idempotency_key=key,
    )
    if claimed is None:
        raise ManagerLifecycleNotFound("manager action not found")
    if replay and claimed.state != "approved":
        return _sync_program_action_link(
            claimed,
            expected_program_id=getattr(claimed, "experiment_program_id", None),
        )
    if not changed:
        if not replay:
            raise ManagerLifecycleConflict(
                f"manager action cannot be approved from state {claimed.state}"
            )
    if claimed.execution_mode == "human_task":
        return _sync_program_action_link(
            claimed,
            expected_program_id=getattr(claimed, "experiment_program_id", None),
        )

    actuator = _get_actuator(claimed.execution_mode)
    try:
        preflight = actuator.preflight(claimed)
    except Exception as exc:
        return _block_after_preflight(claimed, actor, key, str(exc))
    blocker = _preflight_blocker(claimed, preflight)
    if blocker:
        return _block_after_preflight(claimed, actor, key, blocker, preflight)

    preflight_json = _preflight_json(preflight)
    applying, transitioned = transition_agent_action_experiment(
        experiment_id=claimed.id,
        expected_states={"approved"},
        state="applying",
        event="preflight_passed",
        actor_user_id=actor.user_id,
        idempotency_key=key,
        preflight_json=preflight_json,
    )
    if applying is None:
        raise ManagerLifecycleNotFound("manager action not found")
    if not transitioned:
        return _sync_program_action_link(
            applying,
            expected_program_id=getattr(applying, "experiment_program_id", None),
        )
    try:
        outcome = actuator.apply(applying, idempotency_key=key, actor=actor)
    except Exception as exc:
        return _record_unknown_actuation(applying, actor, key, exc)
    return _apply_outcome(applying, actor, key, outcome)


def verify_manager_action(experiment_id: str) -> AgentActionExperimentRecord:
    experiment = get_manager_action(experiment_id)
    experiment = _sync_program_action_link(
        experiment,
        expected_program_id=getattr(experiment, "experiment_program_id", None),
    )
    if experiment.state != "applying":
        return experiment
    key = experiment.approval_idempotency_key
    if not key:
        raise ManagerLifecycleConflict("applying action has no idempotency key")
    actuator = _get_actuator(experiment.execution_mode)
    try:
        outcome = actuator.verify(experiment, idempotency_key=key)
    except Exception as exc:
        updated, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applying"},
            state="applying",
            event="verification_pending",
            idempotency_key=key,
            last_error=str(exc),
        )
        pending = updated or experiment
        return _sync_program_action_link(
            pending,
            expected_program_id=getattr(pending, "experiment_program_id", None),
        )
    actor = ManagerActor(
        user_id=experiment.approved_by or 0,
        username="system-verifier",
        role="admin",
    )
    return _apply_outcome(experiment, actor, key, outcome)


def reject_manager_action(
    *,
    experiment_id: str,
    actor: ManagerActor,
    idempotency_key: str,
    reason: str,
    manager_label: str | None,
) -> AgentActionExperimentRecord:
    _ensure_manager_feature_enabled()
    require_role(actor, "manager")
    key = _require_idempotency_key(idempotency_key)
    cleaned_reason = reason.strip()
    if not cleaned_reason:
        raise ValueError("rejection reason is required")
    existing = get_manager_action(experiment_id)
    _sync_program_action_link(
        existing,
        expected_program_id=getattr(existing, "experiment_program_id", None),
    )
    record, changed, replay = reject_agent_action_experiment(
        experiment_id=experiment_id,
        actor_user_id=actor.user_id,
        idempotency_key=key,
        reason=cleaned_reason,
        manager_label=manager_label,
    )
    if record is None:
        raise ManagerLifecycleNotFound("manager action not found")
    if replay:
        return _sync_program_action_link(
            record,
            expected_program_id=getattr(record, "experiment_program_id", None),
        )
    if not changed:
        raise ManagerLifecycleConflict(
            f"manager action cannot be rejected from state {record.state}"
        )
    return _sync_program_action_link(
        record,
        expected_program_id=getattr(record, "experiment_program_id", None),
    )


def complete_manual_manager_action(
    *,
    experiment_id: str,
    actor: ManagerActor,
    idempotency_key: str,
    actual_at: datetime,
) -> AgentActionExperimentRecord:
    """Record completion of a human task and start its measurement window.

    For supply tasks ``actual_at`` is the confirmed supplier-delivery time, so
    the evaluation window never starts from the recommendation or approval time.
    """

    _ensure_manager_feature_enabled()
    require_role(actor, "manager")
    key = _require_idempotency_key(idempotency_key)
    completed_at = _aware_utc(actual_at)
    if completed_at > _utc_now() + timedelta(seconds=5):
        raise ManagerLifecycleConflict("manual completion timestamp is in the future")
    experiment = get_manager_action(experiment_id)
    experiment = _sync_program_action_link(
        experiment,
        expected_program_id=getattr(experiment, "experiment_program_id", None),
    )
    if experiment.execution_mode != "human_task":
        raise ManagerLifecycleConflict("only a human task can be completed manually")
    if experiment.state in {"applied", "measuring", "measured", "closed"}:
        return _sync_program_action_link(
            experiment,
            expected_program_id=getattr(experiment, "experiment_program_id", None),
        )
    if experiment.state != "approved":
        raise ManagerLifecycleConflict(
            f"manual task cannot be completed from state {experiment.state}"
        )
    applied, changed = transition_agent_action_experiment(
        experiment_id=experiment.id,
        expected_states={"approved"},
        state="applied",
        event="manual_task_completed",
        actor_user_id=actor.user_id,
        idempotency_key=key,
        applied_at=completed_at,
    )
    if applied is None:
        raise ManagerLifecycleNotFound("manager action not found")
    if not changed:
        return _sync_program_action_link(
            applied,
            expected_program_id=getattr(applied, "experiment_program_id", None),
        )
    return _schedule_measurement(applied, completed_at)


def record_manager_action_measurement(
    *,
    experiment_id: str,
    evaluation_start: datetime,
    evaluation_end: datetime,
    data_as_of: datetime | None,
    data_complete: bool,
    baseline_treated: float | None,
    post_treated: float | None,
    baseline_control: float | None = None,
    post_control: float | None = None,
    metrics: dict[str, Any] | None = None,
    created_by: int | None = None,
    now: datetime | None = None,
    idempotency_key: str | None = None,
) -> AgentActionMeasurementRecord:
    experiment = get_manager_action(experiment_id)
    experiment = _sync_program_action_link(
        experiment,
        expected_program_id=getattr(experiment, "experiment_program_id", None),
    )
    if experiment.state not in {"applied", "measuring"}:
        raise ManagerLifecycleConflict("action is not ready for measurement")
    forecast = experiment.expected_effect
    low = _number(
        _pick(
            forecast,
            "low",
            "forecastLow",
            "forecast_low",
            "contribution_profit_low_per_day",
            "contributionProfitLowPerDay",
        )
    )
    base = _number(
        _pick(
            forecast,
            "base",
            "forecastBase",
            "forecast_base",
            "contribution_profit_base_per_day",
            "contributionProfitBasePerDay",
        )
    )
    high = _number(
        _pick(
            forecast,
            "high",
            "forecastHigh",
            "forecast_high",
            "contribution_profit_high_per_day",
            "contributionProfitHighPerDay",
        )
    )
    calculation = calculate_measurement(
        data_complete=data_complete,
        evaluation_end=evaluation_end,
        data_as_of=data_as_of,
        baseline_treated=baseline_treated,
        post_treated=post_treated,
        baseline_control=baseline_control,
        post_control=post_control,
        forecast_low=low,
        forecast_base=base,
        forecast_high=high,
        now=now,
    )
    provider_metrics = dict(metrics or {})
    provider_reason = provider_metrics.get("reason")
    measurement = append_agent_action_measurement(
        experiment_id=experiment.id,
        phase="readout",
        status=calculation.status,
        period_start=evaluation_start,
        period_end=evaluation_end,
        data_as_of=data_as_of,
        idempotency_key=idempotency_key,
        metrics={
            **provider_metrics,
            "data_complete": data_complete,
            "baseline_treated": baseline_treated,
            "post_treated": post_treated,
            "baseline_control": baseline_control,
            "post_control": post_control,
            "providerReason": provider_reason,
            "calculationReason": calculation.reason,
            "reason": provider_reason or calculation.reason,
        },
        causal_valid=calculation.causal_valid,
        observed_delta=calculation.observed_delta,
        did_effect=calculation.did_effect,
        forecast_low=low,
        forecast_base=base,
        forecast_high=high,
        signed_bias=calculation.signed_bias,
        mae=calculation.mae,
        inside_forecast_range=calculation.inside_forecast_range,
        created_by=created_by,
    )
    if calculation.status in {"computed", "inconclusive"}:
        transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applied", "measuring"},
            state="measured",
            event="measurement_completed",
            actor_user_id=created_by,
        )
    elif experiment.state == "applied":
        transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applied"},
            state="measuring",
            event="measurement_waiting",
            actor_user_id=created_by,
            measurement_due_at=evaluation_end,
        )
    current = get_agent_action_experiment(experiment.id)
    if current is not None:
        _sync_program_action_link(
            current,
            expected_program_id=getattr(current, "experiment_program_id", None),
        )
    return measurement


def _apply_outcome(
    experiment: AgentActionExperimentRecord,
    actor: ManagerActor,
    key: str,
    outcome: ActuationResult,
) -> AgentActionExperimentRecord:
    payload = _actuation_json(outcome)
    if outcome.status in {"applying", "unknown"}:
        updated, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applying"},
            state="applying",
            event="actuation_pending_verification",
            actor_user_id=actor.user_id or None,
            idempotency_key=key,
            actuator_result=payload,
            external_operation_id=outcome.external_operation_id,
            last_error=outcome.detail,
        )
        pending = updated or experiment
        _schedule_verification_safely(
            pending,
            delay_seconds=outcome.verification_due_seconds or 30,
        )
        return _sync_program_action_link(
            pending,
            expected_program_id=getattr(pending, "experiment_program_id", None),
        )
    if outcome.status == "blocked":
        updated, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applying"},
            state="blocked",
            event="actuation_blocked",
            actor_user_id=actor.user_id or None,
            idempotency_key=key,
            actuator_result=payload,
            external_operation_id=outcome.external_operation_id,
            last_error=outcome.detail or "actuator blocked the action",
        )
        blocked = updated or experiment
        return _sync_program_action_link(
            blocked,
            expected_program_id=getattr(blocked, "experiment_program_id", None),
        )
    if outcome.status == "failed":
        updated, _ = transition_agent_action_experiment(
            experiment_id=experiment.id,
            expected_states={"applying"},
            state="failed",
            event="actuation_failed",
            actor_user_id=actor.user_id or None,
            idempotency_key=key,
            actuator_result=payload,
            external_operation_id=outcome.external_operation_id,
            last_error=outcome.detail or "actuator failed",
        )
        failed = updated or experiment
        return _sync_program_action_link(
            failed,
            expected_program_id=getattr(failed, "experiment_program_id", None),
        )
    if outcome.status != "applied":
        raise ManagerLifecycleConflict(f"unsupported actuator status: {outcome.status}")

    applied_at = _utc_now()
    applied, changed = transition_agent_action_experiment(
        experiment_id=experiment.id,
        expected_states={"applying"},
        state="applied",
        event="actuation_verified",
        actor_user_id=actor.user_id or None,
        idempotency_key=key,
        actuator_result=payload,
        external_operation_id=outcome.external_operation_id,
        applied_at=applied_at,
    )
    if applied is None or not changed:
        current = applied or experiment
        return _sync_program_action_link(
            current,
            expected_program_id=getattr(current, "experiment_program_id", None),
        )
    return _schedule_measurement(applied, applied_at)


def _schedule_measurement(
    experiment: AgentActionExperimentRecord,
    applied_at: datetime,
) -> AgentActionExperimentRecord:
    default_days = 7 if experiment.execution_mode == "wb_ads" else 14
    evaluation_days = int(
        _pick(
            experiment.measurement_spec,
            "evaluationDays",
            "evaluation_days",
            default=default_days,
        )
        or default_days
    )
    evaluation_days = max(1, min(evaluation_days, 90))
    baseline_days = int(
        _pick(
            experiment.measurement_spec,
            "baselineDays",
            "baseline_days",
            default=7 if experiment.family == "advertising" else 14,
        )
        or (7 if experiment.family == "advertising" else 14)
    )
    baseline_days = max(1, min(baseline_days, 90))
    from agi_ecom_app.manager_lifecycle.jobs import manager_measurement_windows

    _, _, evaluation_start, due_at = manager_measurement_windows(
        family=experiment.family,
        applied_at=applied_at,
        baseline_days=baseline_days,
        evaluation_days=evaluation_days,
    )
    append_agent_action_measurement(
        experiment_id=experiment.id,
        phase="readout",
        status="waiting_for_data",
        period_start=evaluation_start,
        period_end=due_at,
        metrics={"reason": "evaluation_window_open"},
        causal_valid=False,
    )
    measuring, _ = transition_agent_action_experiment(
        experiment_id=experiment.id,
        expected_states={"applied"},
        state="measuring",
        event="measurement_scheduled",
        measurement_due_at=due_at,
    )
    scheduled = measuring or experiment
    _schedule_measurement_safely(scheduled, due_at=due_at)
    return _sync_program_action_link(
        scheduled,
        expected_program_id=getattr(scheduled, "experiment_program_id", None),
    )


def _schedule_verification_safely(
    experiment: AgentActionExperimentRecord,
    *,
    delay_seconds: int,
) -> None:
    try:
        from agi_ecom_app.manager_lifecycle.jobs import (
            schedule_manager_action_verification,
        )

        schedule_manager_action_verification(
            experiment.id,
            version=experiment.version,
            delay_seconds=delay_seconds,
        )
    except Exception:
        # The lifecycle row remains `applying`, so a reconciliation worker can
        # safely recover without ever replaying the external write.
        logger.exception(
            "manager_action_verification_schedule_failed experiment_id=%s",
            experiment.id,
        )


def _schedule_measurement_safely(
    experiment: AgentActionExperimentRecord,
    *,
    due_at: datetime,
) -> None:
    try:
        from agi_ecom_app.manager_lifecycle.jobs import schedule_manager_action_measurement

        schedule_manager_action_measurement(
            experiment.id,
            version=experiment.version,
            due_at=due_at,
        )
    except Exception:
        # The durable `measurement_due_at` plus waiting readout is the source of
        # truth; a periodic reconciliation can requeue it without fabricating 0.
        logger.exception(
            "manager_action_measurement_schedule_failed experiment_id=%s",
            experiment.id,
        )


def _record_unknown_actuation(
    experiment: AgentActionExperimentRecord,
    actor: ManagerActor,
    key: str,
    exc: Exception,
) -> AgentActionExperimentRecord:
    outcome = ActuationResult(
        status="unknown",
        detail=f"{type(exc).__name__}: {exc}",
        raw={"exception_type": type(exc).__name__},
    )
    return _apply_outcome(experiment, actor, key, outcome)


def _block_after_preflight(
    experiment: AgentActionExperimentRecord,
    actor: ManagerActor,
    key: str,
    blocker: str,
    preflight: PreflightResult | None = None,
) -> AgentActionExperimentRecord:
    blocked, _ = transition_agent_action_experiment(
        experiment_id=experiment.id,
        expected_states={"approved"},
        state="blocked",
        event="preflight_blocked",
        actor_user_id=actor.user_id,
        idempotency_key=key,
        preflight_json=_preflight_json(preflight) if preflight else {},
        last_error=blocker,
    )
    current = blocked or experiment
    return _sync_program_action_link(
        current,
        expected_program_id=getattr(current, "experiment_program_id", None),
    )


def _preflight_blocker(
    experiment: AgentActionExperimentRecord,
    preflight: PreflightResult,
) -> str | None:
    if not preflight.allowed:
        return "; ".join(preflight.blockers) or "preflight guardrails failed"
    if preflight.observed_at is None:
        return "preflight did not provide an observation timestamp"
    observed_at = _aware_utc(preflight.observed_at)
    now = _utc_now()
    if observed_at > now + timedelta(seconds=5):
        return "preflight timestamp is in the future"
    if now - observed_at > timedelta(minutes=5):
        return "preflight state is older than five minutes"
    if not _values_equal(experiment.from_value, preflight.current_value):
        return "external state changed after the recommendation was produced"
    return None


def _validate_exact_action_contract(experiment: AgentActionExperimentRecord) -> None:
    if experiment.from_value is None or experiment.from_value == {}:
        raise ManagerLifecycleConflict("exact from-value is required")
    if experiment.to_value is None or experiment.to_value == {}:
        raise ManagerLifecycleConflict("exact to-value is required")
    if experiment.execution_mode in {"wb_ads", "wb_price"} and not experiment.rollback:
        raise ManagerLifecycleConflict("a reversible WB action requires rollback data")


def _get_actuator(execution_mode: str) -> ManagerActionActuator:
    actuator = _ACTUATORS.get(execution_mode)
    if actuator is not None:
        return actuator
    if execution_mode in {"wb_ads", "wb_price"}:
        actuator = _WBExecutorActuator()
        _ACTUATORS[execution_mode] = actuator
        return actuator
    raise ManagerLifecycleConflict(f"no actuator for execution mode {execution_mode}")


class _WBExecutorActuator:
    def __init__(self) -> None:
        from agi_ecom_app.services.wb_manager_execution import WBManagerActionExecutor

        self._executor = WBManagerActionExecutor()

    def preflight(self, experiment: AgentActionExperimentRecord) -> PreflightResult:
        raw = self._executor.preflight(_wb_action_spec(experiment))
        current_value = _preflight_comparable_current(
            experiment, raw.get("current_value")
        )
        provider = _LIVE_GUARDRAIL_PROVIDERS.get(experiment.execution_mode)
        if provider is None:
            return PreflightResult(
                allowed=False,
                current_value=current_value,
                observed_at=_utc_now(),
                blockers=("live execution guardrail provider is not configured",),
                raw={"wb": raw},
            )
        snapshot = provider.collect(experiment)
        guardrail_blockers = evaluate_live_guardrails(experiment, snapshot)
        return PreflightResult(
            allowed=not guardrail_blockers,
            current_value=current_value,
            observed_at=snapshot.observed_at,
            blockers=guardrail_blockers,
            raw={
                "wb": raw,
                "guardrail_observations": snapshot.observations,
            },
        )

    def apply(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        idempotency_key: str,
        actor: ManagerActor,
    ) -> ActuationResult:
        del actor
        from agi_ecom_app.services.wb_manager_execution import (
            WBActionBlockedError,
            WBActionExecutionError,
            WBActionUnknownResultError,
        )

        try:
            outcome = self._executor.execute(
                _wb_action_spec(experiment),
                idempotency_key=idempotency_key,
            )
        except WBActionBlockedError as exc:
            return ActuationResult(status="blocked", detail=str(exc))
        except WBActionUnknownResultError as exc:
            return ActuationResult(status="unknown", detail=str(exc))
        except WBActionExecutionError as exc:
            return ActuationResult(status="failed", detail=str(exc))
        return _from_wb_outcome(outcome)

    def verify(
        self,
        experiment: AgentActionExperimentRecord,
        *,
        idempotency_key: str,
    ) -> ActuationResult:
        from agi_ecom_app.services.wb_manager_execution import (
            WBActionBlockedError,
            WBActionExecutionError,
            WBActionUnknownResultError,
        )

        try:
            outcome = self._executor.verify(
                _wb_action_spec(experiment),
                external_reference=experiment.external_operation_id,
                idempotency_key=idempotency_key,
            )
        except WBActionBlockedError as exc:
            return ActuationResult(status="blocked", detail=str(exc))
        except WBActionUnknownResultError as exc:
            return ActuationResult(status="unknown", detail=str(exc))
        except WBActionExecutionError as exc:
            return ActuationResult(status="failed", detail=str(exc))
        return _from_wb_outcome(outcome)


def _wb_action_spec(experiment: AgentActionExperimentRecord) -> dict[str, Any]:
    target = dict(experiment.target)
    payload = target.pop("payload", None)
    if isinstance(payload, Mapping):
        target.update(payload)
    operation = str(target.get("operation") or "")
    target["operation"] = {
        "set_bid": "change_bid",
        "resume_campaign": "start_campaign",
        "set_price": "set_price_discount",
        "set_discount": "set_price_discount",
    }.get(operation, operation)
    from_value = _exact_scalar(experiment.from_value)
    to_value = _exact_scalar(experiment.to_value)
    if operation == "set_price":
        from_value = {"price": from_value}
        to_value = {"price": to_value}
    elif operation == "set_discount":
        from_value = {"discount": from_value}
        to_value = {"discount": to_value}
    target["from"] = from_value
    target["to"] = to_value
    target["guardrails"] = experiment.guardrails
    target["rollback"] = experiment.rollback
    return target


def _from_wb_outcome(outcome: Any) -> ActuationResult:
    raw = outcome.model_dump() if hasattr(outcome, "model_dump") else asdict(outcome)
    return ActuationResult(
        status=str(outcome.status),
        external_operation_id=getattr(outcome, "external_reference", None),
        applied_value=(getattr(outcome, "applied_spec", {}) or {}).get("to"),
        verification_due_seconds=getattr(outcome, "verification_due_seconds", None),
        detail=getattr(outcome, "detail", None),
        raw=raw,
    )


def _preflight_json(preflight: PreflightResult | None) -> dict[str, Any]:
    if preflight is None:
        return {}
    return {
        "allowed": preflight.allowed,
        "current_value": preflight.current_value,
        "observed_at": preflight.observed_at.isoformat() if preflight.observed_at else None,
        "blockers": list(preflight.blockers),
        "raw": preflight.raw,
    }


def evaluate_live_guardrails(
    experiment: AgentActionExperimentRecord,
    snapshot: LiveGuardrailSnapshot,
) -> tuple[str, ...]:
    """Evaluate exact guardrail rules against a freshly collected snapshot."""

    blockers = list(snapshot.blockers)
    rules = experiment.guardrails.get("rules") if isinstance(experiment.guardrails, Mapping) else None
    if not isinstance(rules, list) or not rules:
        blockers.append("structured execution guardrails are required")
        return tuple(dict.fromkeys(blockers))
    for index, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, Mapping):
            blockers.append(f"guardrail[{index}] is not an object")
            continue
        metric = str(raw_rule.get("metric") or "").strip()
        operator = str(raw_rule.get("operator") or "").strip().lower()
        threshold = raw_rule.get("threshold")
        if not metric or not isinstance(threshold, Mapping) or "value" not in threshold:
            blockers.append(f"guardrail[{index}] is incomplete")
            continue
        if metric not in snapshot.observations:
            blockers.append(f"live guardrail metric {metric} is unavailable")
            continue
        observed = snapshot.observations[metric]
        if isinstance(observed, Mapping):
            observed_value = observed.get("value")
            observed_unit = observed.get("unit")
        else:
            observed_value = observed
            observed_unit = None
        threshold_value = threshold.get("value")
        threshold_unit = threshold.get("unit")
        if observed_unit is None or observed_unit != threshold_unit:
            blockers.append(f"live guardrail metric {metric} has an incompatible unit")
            continue
        if not _guardrail_comparison(observed_value, operator, threshold_value):
            blockers.append(f"live guardrail {metric} breached")
    return tuple(dict.fromkeys(blockers))


def _guardrail_comparison(left: Any, operator: str, right: Any) -> bool:
    if operator == "eq":
        return left == right
    if operator == "neq":
        return left != right
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    if operator == "gt":
        return left_number > right_number
    if operator == "gte":
        return left_number >= right_number
    if operator == "lt":
        return left_number < right_number
    if operator == "lte":
        return left_number <= right_number
    return False


def _actuation_json(outcome: ActuationResult) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "external_operation_id": outcome.external_operation_id,
        "applied_value": outcome.applied_value,
        "rollback_payload": outcome.rollback_payload,
        "verification_due_seconds": outcome.verification_due_seconds,
        "detail": outcome.detail,
        "raw": outcome.raw,
    }


def _normalize_execution_mode(value: Any, family: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "human": "human_task",
        "manual": "human_task",
        "manual_task": "human_task",
        "task": "human_task",
        "ads": "wb_ads",
        "marketing": "wb_ads",
        "price": "wb_price",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"human_task", "wb_ads", "wb_price"}:
        return normalized
    family_normalized = family.strip().lower()
    if normalized in {"api", "auto", "automatic", "wb_api"}:
        if family_normalized in {"ads", "advertising", "marketing", "реклама"}:
            return "wb_ads"
        if family_normalized in {"price", "pricing", "цена", "цены"}:
            return "wb_price"
    return "human_task"


def _ensure_manager_feature_enabled() -> None:
    if not bool(get_settings().manager_decision_enabled):
        raise ManagerFeatureDisabled("MANAGER_DECISION_ENABLED is disabled")


def _require_idempotency_key(value: str) -> str:
    key = value.strip()
    if not key or len(key) > 200:
        raise ValueError("Idempotency-Key must contain between 1 and 200 characters")
    return key


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_safe_mapping(dict(value))
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json", by_alias=True))
    if hasattr(value, "dict"):
        return dict(value.dict(by_alias=True))
    raise TypeError("manager decision must be a mapping or Pydantic model")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    def default(item: Any) -> Any:
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json", by_alias=True)
        raise TypeError(f"not JSON serializable: {type(item).__name__}")

    return json.loads(json.dumps(dict(value), default=default, ensure_ascii=False))


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    return [_as_mapping(item) for item in (value or [])]


def _pick(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"invalid datetime value: {value!r}")


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"invalid date value: {value!r}")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _exact_scalar(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _preflight_comparable_current(
    experiment: AgentActionExperimentRecord,
    current: Any,
) -> Any:
    expected = experiment.from_value
    if not isinstance(expected, Mapping) or "value" not in expected:
        return current
    operation = str(experiment.target.get("operation") or "")
    if isinstance(current, Mapping):
        if operation == "set_price":
            current = current.get("price")
        elif operation == "set_discount":
            current = current.get("discount")
    return {"value": current, "unit": expected.get("unit")}


def _values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return all(actual.get(key) == value for key, value in expected.items())
    return json.dumps(expected, sort_keys=True, ensure_ascii=False) == json.dumps(
        actual, sort_keys=True, ensure_ascii=False
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "ActuationResult",
    "ManagerActionActuator",
    "ManagerLiveGuardrailProvider",
    "LiveGuardrailSnapshot",
    "ManagerLifecycleConflict",
    "ManagerLifecycleError",
    "ManagerFeatureDisabled",
    "ManagerLifecycleNotFound",
    "PreflightResult",
    "approve_manager_action",
    "complete_manual_manager_action",
    "get_decision_bundle",
    "get_manager_action",
    "get_today_decision_bundles",
    "persist_manager_decision_bundle",
    "record_manager_action_measurement",
    "register_manager_action_actuator",
    "register_manager_live_guardrail_provider",
    "evaluate_live_guardrails",
    "reject_manager_action",
    "verify_manager_action",
]
