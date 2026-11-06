from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

from agi_ecom_app.config import get_settings
from agi_ecom_app.decision_simulation.engine import (
    ADVERTISING_BOOTSTRAP_MODEL_VERSION,
    derive_master_seed,
    simulation_runtime_card,
    validate_advertising_pause_candidate,
)
from agi_ecom_app.decision_simulation.models import (
    AdvertisingPauseCandidateV1,
    AdvertisingPauseEvidenceV1,
    DailySpendEvidenceV1,
    ScalarEvidenceV1,
)
from agi_ecom_app.decision_simulation.metrics import ELIGIBILITY_OUTCOMES
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    derive_pricing_master_seed,
)
from agi_ecom_app.decision_simulation.pricing_models import PricingSimulationRequestV1
from agi_ecom_app.manager_decision.models import (
    DecisionBlockerV1,
    ManagerActionV1,
    ManagerDecisionV1,
    ManagerDecisionWorkflowResult,
    SimulationEligibilityV1,
)
from agi_ecom_app.manager_decision.workflow import render_manager_decision
from agi_ecom_app.manager_lifecycle.auth import ManagerActor, require_role
from agi_ecom_app.manager_lifecycle.service import (
    ManagerLifecycleConflict,
    persist_manager_decision_bundle,
)
from agi_ecom_app.services.memory import (
    DecisionSimulationRunRecord,
    append_decision_simulation_event,
    claim_decision_simulation_initialization,
    create_decision_simulation_run,
    get_decision_simulation_run,
    get_manager_decision,
    get_manager_decision_by_source_key,
    list_decision_simulation_runs,
    mark_decision_simulation_run_superseded,
    set_manager_decision_simulation_run,
    update_decision_simulation_run,
    upsert_decision_simulation_candidate,
)


logger = logging.getLogger(__name__)

SIMULATION_SNAPSHOT_SCHEMA_VERSION = "decision-simulation-snapshot-v1"
_TERMINAL_RUN_STATUSES = frozenset({"completed", "blocked", "failed", "superseded"})
_MINIMUM_CANDIDATE_COUNT = 3
_MAXIMUM_CANDIDATE_COUNT = 7


class SimulationExecutor(Protocol):
    """Replaceable delivery boundary; Postgres remains the source of truth."""

    def enqueue(self, simulation_run_id: str, *, replace_stale: bool = False) -> str: ...


class RQSimulationExecutor:
    def __init__(self, queue: Any | None = None) -> None:
        self._queue = queue

    def enqueue(self, simulation_run_id: str, *, replace_stale: bool = False) -> str:
        return enqueue_decision_simulation(
            simulation_run_id,
            queue=self._queue,
            replace_stale=replace_stale,
        )

    def delivery_status(self, simulation_run_id: str) -> str | None:
        queue = self._queue if self._queue is not None else _build_rq_queue()
        if not hasattr(queue, "fetch_job"):
            return None
        job = queue.fetch_job(decision_simulation_job_id(simulation_run_id))
        if job is None:
            return None
        raw_status = job.get_status(refresh=True)
        return str(getattr(raw_status, "value", raw_status)).lower()


def decision_simulation_job_id(simulation_run_id: str) -> str:
    # RQ rejects ':' in job IDs, so the specified logical
    # `decision-simulation:{run_id}` key is encoded with its supported '-'
    # separator while remaining deterministic and one-to-one.
    return f"decision-simulation-{simulation_run_id}"


def canonical_snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(snapshot),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fact_by_metric(action: ManagerActionV1, *metrics: str) -> Any | None:
    wanted = {metric.casefold() for metric in metrics}
    return next(
        (fact for fact in action.evidence if fact.metric.casefold() in wanted),
        None,
    )


def _scalar_evidence(action: ManagerActionV1, *metrics: str) -> ScalarEvidenceV1 | None:
    fact = _fact_by_metric(action, *metrics)
    if fact is None:
        return None
    value = fact.value
    if isinstance(value, bool):
        numeric: float | None = float(value)
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        try:
            numeric = float(value) if value is not None else None
        except (TypeError, ValueError):
            numeric = None
    return ScalarEvidenceV1(value=numeric, evidenceRef=fact.evidence_ref)


def _daily_spend_evidence(action: ManagerActionV1) -> DailySpendEvidenceV1 | None:
    fact = _fact_by_metric(action, "daily_ad_spend_distribution", "daily_ad_spend")
    if fact is None:
        return None
    raw = fact.value
    if isinstance(raw, Mapping):
        raw = raw.get("valuesRub", raw.get("values_rub", raw.get("values")))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        values = None
    else:
        try:
            values = [float(value) for value in raw]
        except (TypeError, ValueError):
            values = None
    return DailySpendEvidenceV1(valuesRub=values, evidenceRef=fact.evidence_ref)


def advertising_candidate_from_action(action: ManagerActionV1) -> AdvertisingPauseCandidateV1:
    if action.family != "advertising" or action.operation != "pause_campaign":
        raise ValueError("only deterministic advertising pause_campaign actions are supported")
    confidence_risk = {"high": 0.0, "medium": 1.0, "low": 2.0}[action.confidence]
    execution_risk = confidence_risk + (0.0 if action.execution_mode == "wb_api" else 1.0)
    return AdvertisingPauseCandidateV1(
        candidateId=action.action_id,
        actionId=action.action_id,
        horizonDays=action.expected_effect.horizon_days,
        executionRisk=execution_risk,
        evidence=AdvertisingPauseEvidenceV1(
            dailySpend=_daily_spend_evidence(action),
            stockOnHand=_scalar_evidence(action, "stock_on_hand"),
            supplierInbound=_scalar_evidence(action, "supplier_inbound"),
            adOrders=_scalar_evidence(action, "ad_orders"),
            attributedRevenue=_scalar_evidence(
                action,
                "attributed_revenue",
                "revenue",
            ),
            contributionProfit=_scalar_evidence(action, "contribution_profit"),
        ),
    )


def _simulation_candidates(decision: ManagerDecisionV1) -> list[AdvertisingPauseCandidateV1]:
    actions = list(decision.ranked_actions)
    if not _MINIMUM_CANDIDATE_COUNT <= len(actions) <= _MAXIMUM_CANDIDATE_COUNT:
        return []
    if any(
        action.family != "advertising" or action.operation != "pause_campaign"
        for action in actions
    ):
        return []
    return [advertising_candidate_from_action(action) for action in actions]


def _simulation_eligibility(
    *,
    outcome: str,
    mode: str,
    submitted_candidate_count: int,
    eligible_candidate_count: int,
    reason_code: str,
    record_metric: bool = True,
) -> SimulationEligibilityV1:
    eligibility = SimulationEligibilityV1(
        outcome=outcome,
        mode=mode,
        submittedCandidateCount=submitted_candidate_count,
        eligibleCandidateCount=eligible_candidate_count,
        rejectedCandidateCount=(
            submitted_candidate_count - eligible_candidate_count
        ),
        reasonCode=reason_code,
    )
    if record_metric:
        ELIGIBILITY_OUTCOMES.labels(mode=mode, outcome=outcome).inc()
    return eligibility


def _record_simulation_eligibility(
    eligibility: SimulationEligibilityV1,
) -> None:
    ELIGIBILITY_OUTCOMES.labels(
        mode=eligibility.mode,
        outcome=eligibility.outcome,
    ).inc()


def _with_simulation_eligibility(
    result: ManagerDecisionWorkflowResult,
    eligibility: SimulationEligibilityV1,
) -> ManagerDecisionWorkflowResult:
    decision = result.decision.model_copy(
        update={"simulation_eligibility": eligibility}
    )
    return ManagerDecisionWorkflowResult(
        decision=decision,
        renderedMarkdown=render_manager_decision(decision),
        factsOnly=result.facts_only,
    )


def _with_delivery_failed_eligibility(
    result: ManagerDecisionWorkflowResult,
) -> ManagerDecisionWorkflowResult:
    eligibility = result.decision.simulation_eligibility
    if eligibility is None or eligibility.outcome != "eligible":
        return result
    return _with_simulation_eligibility(
        result,
        eligibility.model_copy(
            update={"reason_code": "eligible_candidates_delivery_failed"}
        ),
    )


def build_evidence_snapshot(
    decision: ManagerDecisionV1,
    candidates: Sequence[AdvertisingPauseCandidateV1],
    *,
    hard_gate_rejected: Sequence[tuple[AdvertisingPauseCandidateV1, Sequence[Any]]] = (),
) -> dict[str, Any]:
    return {
        "schemaVersion": SIMULATION_SNAPSHOT_SCHEMA_VERSION,
        "objective": decision.objective,
        "modelRuntime": simulation_runtime_card(),
        "managerDecision": decision.model_dump(mode="json", by_alias=True),
        "simulationCandidates": [
            candidate.model_dump(mode="json", by_alias=True) for candidate in candidates
        ],
        "hardGateRejectedCandidates": [
            {
                "candidate": candidate.model_dump(mode="json", by_alias=True),
                "violations": [
                    violation.model_dump(mode="json", by_alias=True)
                    for violation in violations
                ],
            }
            for candidate, violations in hard_gate_rejected
        ],
    }


def _candidate_evidence_refs(candidate: AdvertisingPauseCandidateV1) -> list[str]:
    refs: list[str] = []
    for evidence in (
        candidate.evidence.daily_spend,
        candidate.evidence.stock_on_hand,
        candidate.evidence.supplier_inbound,
        candidate.evidence.ad_orders,
        candidate.evidence.attributed_revenue,
        candidate.evidence.contribution_profit,
    ):
        ref = getattr(evidence, "evidence_ref", None)
        if isinstance(ref, str) and ref and ref not in refs:
            refs.append(ref)
    return refs


def _evaluating_workflow(
    result: ManagerDecisionWorkflowResult,
) -> ManagerDecisionWorkflowResult:
    decision = ManagerDecisionV1.model_validate(
        {
            **result.decision.model_dump(mode="json", by_alias=True),
            "status": "evaluating",
            "rankedActions": [],
            "blockers": [],
        }
    )
    return ManagerDecisionWorkflowResult(
        decision=decision,
        renderedMarkdown=render_manager_decision(decision),
        factsOnly=True,
    )


def _ineligible_workflow(
    result: ManagerDecisionWorkflowResult,
    *,
    eligible_count: int,
) -> ManagerDecisionWorkflowResult:
    blockers = [
        *result.decision.blockers,
        DecisionBlockerV1(
            code="decision_simulation_insufficient_candidates",
            message=(
                "Decision Simulation требует минимум три рекламных pause_campaign "
                f"кандидата с полным evidence; прошло hard gate: {eligible_count}."
            ),
            severity="needs_data",
        ),
    ]
    decision = ManagerDecisionV1.model_validate(
        {
            **result.decision.model_dump(mode="json", by_alias=True),
            "status": "needs_data",
            "rankedActions": [],
            "blockers": [item.model_dump(mode="json", by_alias=True) for item in blockers],
        }
    )
    return ManagerDecisionWorkflowResult(
        decision=decision,
        renderedMarkdown=render_manager_decision(decision),
        factsOnly=True,
    )


def _blocked_workflow(
    result: ManagerDecisionWorkflowResult,
    *,
    code: str,
    message: str,
) -> ManagerDecisionWorkflowResult:
    blocker = DecisionBlockerV1(code=code, message=message, severity="blocked")
    decision = ManagerDecisionV1.model_validate(
        {
            **result.decision.model_dump(mode="json", by_alias=True),
            "status": "blocked",
            "rankedActions": [],
            "blockers": [blocker.model_dump(mode="json", by_alias=True)],
        }
    )
    return ManagerDecisionWorkflowResult(
        decision=decision,
        renderedMarkdown=render_manager_decision(decision),
        factsOnly=True,
    )


def _stored_manager_workflow(
    decision_id: str,
    fallback: ManagerDecisionWorkflowResult,
) -> ManagerDecisionWorkflowResult:
    record = get_manager_decision(decision_id)
    if record is None or not record.decision_json:
        return fallback
    try:
        decision = ManagerDecisionV1.model_validate(record.decision_json)
    except Exception:
        logger.exception(
            "decision_simulation_stored_decision_invalid decision_id=%s",
            decision_id,
        )
        return fallback
    return ManagerDecisionWorkflowResult(
        decision=decision,
        renderedMarkdown=render_manager_decision(decision),
        factsOnly=decision.status != "ready_for_approval",
    )


def manager_decision_experiments_deferred(decision_id: str) -> bool:
    """Return true while a linked simulation still owns experiment publication."""

    if get_settings().decision_simulation_mode == "off":
        return False
    decision = get_manager_decision(decision_id)
    if decision is None or not decision.simulation_run_id:
        return False
    run = get_decision_simulation_run(decision.simulation_run_id)
    return run is not None and run.status not in {"completed", "blocked", "failed"}


def maybe_schedule_decision_simulation(
    result: ManagerDecisionWorkflowResult,
    *,
    chat_id: str | None,
    run_id: int | None,
    source_key: str | None,
    executor: SimulationExecutor | None = None,
) -> ManagerDecisionWorkflowResult:
    """Persist and enqueue a simulation, or preserve the exact legacy path in off mode."""

    settings = get_settings()
    mode = settings.decision_simulation_mode
    if mode == "off":
        return result
    if source_key:
        existing_decision = get_manager_decision_by_source_key(source_key)
        if existing_decision is not None:
            stored = _stored_manager_workflow(existing_decision.id, result)
            if existing_decision.simulation_run_id is not None:
                return stored
            # A prior process may have died after claiming source_key but
            # before creating/linking the run. Resume from that exact durable
            # decision even if volatile timestamps changed the newly computed
            # decision ID.
            result = stored
    if result.decision.status != "ready_for_approval":
        return _with_simulation_eligibility(
            result,
            _simulation_eligibility(
                outcome="not_ready",
                mode=mode,
                submitted_candidate_count=0,
                eligible_candidate_count=0,
                reason_code="decision_not_ready_for_approval",
            ),
        )

    submitted_candidates = _simulation_candidates(result.decision)
    gate_results = [
        (candidate, validate_advertising_pause_candidate(candidate))
        for candidate in submitted_candidates
    ]
    candidates = [candidate for candidate, violations in gate_results if not violations]
    rejected_candidates = [item for item in gate_results if item[1]]
    eligible_count = len(candidates)
    submitted_count = len(result.decision.ranked_actions)
    if not submitted_candidates or eligible_count < _MINIMUM_CANDIDATE_COUNT:
        result = _with_simulation_eligibility(
            result,
            _simulation_eligibility(
                outcome="insufficient",
                mode=mode,
                submitted_candidate_count=submitted_count,
                eligible_candidate_count=eligible_count,
                reason_code=(
                    "no_supported_simulation_candidates"
                    if not submitted_candidates
                    else "insufficient_eligible_candidates"
                ),
            ),
        )
        return (
            _ineligible_workflow(result, eligible_count=eligible_count)
            if mode == "enforced"
            else result
        )

    result = _with_simulation_eligibility(
        result,
        _simulation_eligibility(
            outcome="eligible",
            mode=mode,
            submitted_candidate_count=submitted_count,
            eligible_candidate_count=eligible_count,
            reason_code="eligible_candidates_validated",
        ),
    )
    queued_result = _with_simulation_eligibility(
        result,
        _simulation_eligibility(
            outcome="queued",
            mode=mode,
            submitted_candidate_count=submitted_count,
            eligible_candidate_count=eligible_count,
            reason_code="eligible_candidates_queued",
            record_metric=False,
        ),
    )
    queued_eligibility = queued_result.decision.simulation_eligibility
    assert queued_eligibility is not None

    snapshot = build_evidence_snapshot(
        queued_result.decision,
        candidates,
        hard_gate_rejected=rejected_candidates,
    )
    if mode == "shadow":
        snapshot["shadowPublicDecision"] = queued_result.decision.model_dump(
            mode="json", by_alias=True
        )
    snapshot_hash = canonical_snapshot_hash(snapshot)
    simulation_run_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            ":".join(
                (
                    "decision-simulation",
                    result.decision.decision_id,
                    ADVERTISING_BOOTSTRAP_MODEL_VERSION,
                    snapshot_hash,
                    mode,
                )
            ),
        )
    )
    public_result = _evaluating_workflow(result) if mode == "enforced" else result
    queued_public_result = (
        _evaluating_workflow(queued_result) if mode == "enforced" else queued_result
    )
    existing_run = get_decision_simulation_run(simulation_run_id)
    if existing_run is not None and existing_run.status in _TERMINAL_RUN_STATUSES:
        # A repeated synthesis must never regress a completed/blocked decision
        # to evaluating. The durable manager row is authoritative for every
        # already-created run, including an in-flight or superseded one.
        try:
            from agi_ecom_app.decision_simulation.jobs import (
                repair_terminal_decision_simulation_projection,
            )

            repair_terminal_decision_simulation_projection(existing_run)
        except Exception:
            logger.exception(
                "decision_simulation_terminal_projection_repair_failed "
                "simulation_run_id=%s",
                existing_run.id,
            )
        return _stored_manager_workflow(result.decision.decision_id, public_result)

    run = existing_run
    try:
        if run is None:
            # Stage the source-key owner without experiments. If the process
            # dies before run creation, the stored ready decision is harmless
            # and a duplicate delivery can resume initialization.
            persisted_bundle = persist_manager_decision_bundle(
                result.decision,
                chat_id=chat_id,
                run_id=run_id,
                source_key=source_key,
                create_experiments=False,
            )
            if isinstance(persisted_bundle, tuple) and persisted_bundle:
                persisted_decision = persisted_bundle[0]
                if getattr(persisted_decision, "id", result.decision.decision_id) != (
                    result.decision.decision_id
                ):
                    # A concurrent delivery won the unique source_key race.
                    return _stored_manager_workflow(persisted_decision.id, public_result)
            run = create_decision_simulation_run(
                simulation_run_id=simulation_run_id,
                decision_id=result.decision.decision_id,
                chat_id=chat_id,
                run_id=run_id,
                mode=mode,
                model_version=ADVERTISING_BOOTSTRAP_MODEL_VERSION,
                evidence_snapshot=snapshot,
                evidence_hash=snapshot_hash,
                master_seed=derive_master_seed(snapshot_hash),
                sample_count=settings.decision_simulation_sample_count,
                batch_size=settings.decision_simulation_batch_size,
                candidate_count=len(candidates),
                objective=result.decision.objective,
                status="initializing",
            )
        if run.status != "initializing":
            # Another producer already initialized this deterministic run.
            # Do not overwrite a worker's terminal/active projection.
            delivery_confirmed = run.status == "running"
            if run.status == "queued":
                try:
                    (executor or RQSimulationExecutor()).enqueue(run.id)
                    delivery_confirmed = True
                except Exception:
                    logger.exception(
                        "decision_simulation_duplicate_delivery_refresh_failed "
                        "simulation_run_id=%s",
                        run.id,
                    )
            if delivery_confirmed:
                persist_manager_decision_bundle(
                    queued_public_result.decision,
                    chat_id=chat_id,
                    run_id=run_id,
                    source_key=None,
                    create_experiments=not pricing_retry,
                )
                _record_simulation_eligibility(queued_eligibility)
                return queued_public_result
            return _stored_manager_workflow(result.decision.decision_id, public_result)
        claimed = claim_decision_simulation_initialization(run.id)
        if claimed is None:
            # A concurrent delivery owns initialization. Re-read so a very
            # fast worker cannot be regressed from completed to evaluating.
            current = get_decision_simulation_run(run.id)
            if current is not None and current.status in _TERMINAL_RUN_STATUSES:
                try:
                    from agi_ecom_app.decision_simulation.jobs import (
                        repair_terminal_decision_simulation_projection,
                    )

                    repair_terminal_decision_simulation_projection(current)
                except Exception:
                    logger.exception(
                        "decision_simulation_terminal_projection_repair_failed "
                        "simulation_run_id=%s",
                        run.id,
                    )
                return _stored_manager_workflow(
                    result.decision.decision_id,
                    public_result,
                )
            # The winning initializer has not projected the durable row yet.
            # Enforced callers must still observe evaluating, never the
            # temporary ready staging row used only for crash recovery.
            return public_result
        run = claimed
        for candidate in candidates:
            upsert_decision_simulation_candidate(
                simulation_run_id=run.id,
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                candidate_payload=candidate.model_dump(mode="json", by_alias=True),
            )
        for candidate, violations in rejected_candidates:
            rejected_status = (
                "needs_data"
                if any(violation.severity == "needs_data" for violation in violations)
                else "rejected"
            )
            upsert_decision_simulation_candidate(
                simulation_run_id=run.id,
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                status=rejected_status,
                candidate_payload=candidate.model_dump(mode="json", by_alias=True),
                evidence_refs=_candidate_evidence_refs(candidate),
                constraint_violations=[
                    violation.model_dump(mode="json", by_alias=True)
                    for violation in violations
                ],
            )
            append_decision_simulation_event(
                simulation_run_id=run.id,
                event_type="candidate_rejected",
                event_id=f"{run.id}:candidate:{candidate.candidate_id}:rejected:hard-gate",
                candidate_id=candidate.candidate_id,
                progress=0,
                payload={
                    "status": rejected_status,
                    "constraintViolations": [
                        violation.model_dump(mode="json", by_alias=True)
                        for violation in violations
                    ],
                    "stage": "pre_simulation_hard_gate",
                },
            )
        # Link first, then project evaluating. A crash at either boundary leaves
        # a recoverable run reference rather than an orphan evaluating decision.
        set_manager_decision_simulation_run(
            result.decision.decision_id,
            simulation_run_id=run.id,
            status="evaluating" if mode == "enforced" else None,
        )
        # source_key=None updates by decision ID and preserves ownership.
        persist_manager_decision_bundle(
            public_result.decision,
            chat_id=chat_id,
            run_id=run_id,
            source_key=None,
            create_experiments=False,
        )
        run = update_decision_simulation_run(
            run.id,
            status="queued",
            heartbeat_at=datetime.now(timezone.utc),
            completed_at=None,
            error={},
        )
        append_decision_simulation_event(
            simulation_run_id=run.id,
            event_type="queued",
            event_id=f"{run.id}:queued",
            progress=0,
            payload={
                "completedSamples": 0,
                "totalSamples": run.total_samples,
                "candidateCount": len(candidates),
                "submittedCandidateCount": len(submitted_candidates),
                "rejectedCandidateCount": len(rejected_candidates),
                "mode": mode,
            },
        )
    except Exception as exc:
        logger.exception(
            "decision_simulation_initialization_failed decision_id=%s simulation_run_id=%s",
            result.decision.decision_id,
            simulation_run_id,
        )
        if run is not None:
            try:
                update_decision_simulation_run(
                    run.id,
                    status="failed",
                    error={"code": "initialization_failed", "message": str(exc)},
                )
                append_decision_simulation_event(
                    simulation_run_id=run.id,
                    event_type="failed",
                    event_id=f"{run.id}:failed:initialization",
                    payload={"code": "initialization_failed", "message": str(exc)},
                )
            except Exception:
                logger.exception(
                    "decision_simulation_initialization_failure_not_persisted simulation_run_id=%s",
                    run.id,
                )
        delivery_failed_result = _with_delivery_failed_eligibility(result)
        if mode == "shadow":
            try:
                persist_manager_decision_bundle(
                    delivery_failed_result.decision,
                    chat_id=chat_id,
                    run_id=run_id,
                    source_key=None,
                )
            except Exception:
                logger.exception(
                    "decision_simulation_shadow_projection_failed decision_id=%s",
                    result.decision.decision_id,
                )
            return delivery_failed_result
        blocked = _blocked_workflow(
            delivery_failed_result,
            code="decision_simulation_initialization_failed",
            message="Не удалось надёжно инициализировать Decision Simulation.",
        )
        try:
            persist_manager_decision_bundle(
                blocked.decision,
                chat_id=chat_id,
                run_id=run_id,
                source_key=None,
            )
        except Exception:
            logger.exception(
                "decision_simulation_initialization_failed_to_block decision_id=%s",
                result.decision.decision_id,
            )
        return blocked
    try:
        (executor or RQSimulationExecutor()).enqueue(run.id)
    except Exception as exc:
        logger.exception(
            "decision_simulation_enqueue_failed decision_id=%s simulation_run_id=%s",
            result.decision.decision_id,
            run.id,
        )
        run = update_decision_simulation_run(
            run.id,
            status="failed",
            error={"code": "enqueue_failed", "message": str(exc)},
        )
        append_decision_simulation_event(
            simulation_run_id=run.id,
            event_type="failed",
            event_id=f"{run.id}:failed:enqueue",
            payload={"code": "enqueue_failed", "message": str(exc)},
        )
        delivery_failed_result = _with_delivery_failed_eligibility(result)
        if mode == "enforced":
            return block_manager_decision_for_simulation(
                run,
                code="decision_simulation_enqueue_failed",
                message="Не удалось поставить Decision Simulation в очередь.",
            )
        # Shadow mode must preserve the existing approval path even when the
        # observational delivery fails.
        persist_manager_decision_bundle(
            delivery_failed_result.decision,
            chat_id=chat_id,
            run_id=run_id,
            source_key=None,
        )
        return delivery_failed_result
    persist_manager_decision_bundle(
        queued_public_result.decision,
        chat_id=chat_id,
        run_id=run_id,
        source_key=None,
        create_experiments=False,
    )
    _record_simulation_eligibility(queued_eligibility)
    return queued_public_result


def _build_rq_queue(queue_name: str | None = None) -> Any:
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise RuntimeError("Redis/RQ dependencies are not installed") from exc
    settings = get_settings()
    return Queue(
        queue_name or settings.decision_simulation_queue_name,
        connection=Redis.from_url(settings.redis_url),
    )


def enqueue_decision_simulation(
    simulation_run_id: str,
    *,
    queue: Any | None = None,
    replace_stale: bool = False,
) -> str:
    run = get_decision_simulation_run(simulation_run_id)
    if run is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    if run.status in _TERMINAL_RUN_STATUSES:
        return decision_simulation_job_id(simulation_run_id)

    resolved_queue = queue if queue is not None else _build_rq_queue()
    job_id = decision_simulation_job_id(simulation_run_id)
    existing = resolved_queue.fetch_job(job_id) if hasattr(resolved_queue, "fetch_job") else None
    if existing is not None:
        raw_status = existing.get_status(refresh=True)
        status = str(getattr(raw_status, "value", raw_status)).lower()
        if not replace_stale and status not in {"finished", "failed", "stopped", "canceled"}:
            return job_id
        try:
            existing.cancel()
        except Exception:
            pass
        try:
            existing.delete()
        except Exception:
            if not replace_stale:
                return job_id

    kwargs = {
        "func": "agi_ecom_app.decision_simulation.jobs.run_decision_simulation_job",
        "args": (simulation_run_id,),
        "job_id": job_id,
        "result_ttl": 86_400,
        "failure_ttl": 604_800,
    }
    timeout = get_settings().decision_simulation_job_timeout_seconds
    try:
        try:
            resolved_queue.enqueue_call(timeout=timeout, **kwargs)
        except TypeError:
            resolved_queue.enqueue_call(job_timeout=timeout, **kwargs)
    except Exception:
        # fetch_job -> enqueue is not atomic. If another producer enqueued the
        # same deterministic ID in that window, the durable delivery already
        # exists and must not be turned into a failed simulation.
        concurrent = (
            resolved_queue.fetch_job(job_id)
            if hasattr(resolved_queue, "fetch_job")
            else None
        )
        if concurrent is None:
            raise
        logger.info(
            "decision_simulation_duplicate_enqueue simulation_run_id=%s job_id=%s",
            simulation_run_id,
            job_id,
        )
    try:
        from agi_ecom_app.decision_simulation.metrics import QUEUED_JOBS

        QUEUED_JOBS.set(len(resolved_queue))
    except Exception:
        pass
    return job_id


def _decision_from_snapshot(run: DecisionSimulationRunRecord) -> ManagerDecisionV1:
    payload = run.evidence_snapshot.get("managerDecision")
    if not isinstance(payload, Mapping):
        raise ValueError("simulation snapshot has no managerDecision")
    return ManagerDecisionV1.model_validate(payload)


def block_manager_decision_for_simulation(
    run: DecisionSimulationRunRecord,
    *,
    code: str,
    message: str,
) -> ManagerDecisionWorkflowResult:
    original = _decision_from_snapshot(run)
    eligibility = original.simulation_eligibility
    delivery_failed = (
        str(run.error.get("code") or "")
        in {"initialization_failed", "enqueue_failed"}
        or code
        in {
            "decision_simulation_initialization_failed",
            "decision_simulation_enqueue_failed",
        }
    )
    if delivery_failed and eligibility is not None and eligibility.outcome == "queued":
        original = original.model_copy(
            update={
                "simulation_eligibility": eligibility.model_copy(
                    update={
                        "outcome": "eligible",
                        "reason_code": "eligible_candidates_delivery_failed",
                    }
                )
            }
        )
    blocker = DecisionBlockerV1(code=code, message=message, severity="blocked")
    blocked = ManagerDecisionV1.model_validate(
        {
            **original.model_dump(mode="json", by_alias=True),
            "status": "blocked",
            "rankedActions": [],
            "blockers": [blocker.model_dump(mode="json", by_alias=True)],
        }
    )
    persist_manager_decision_bundle(
        blocked,
        chat_id=run.chat_id,
        run_id=run.run_id,
        source_key=None,
    )
    set_manager_decision_simulation_run(
        run.decision_id,
        simulation_run_id=run.id,
        status="blocked",
    )
    return ManagerDecisionWorkflowResult(
        decision=blocked,
        renderedMarkdown=render_manager_decision(blocked),
        factsOnly=True,
    )


def _create_retry_run(
    previous: DecisionSimulationRunRecord,
    *,
    idempotency_key: str,
    evidence_snapshot: dict[str, Any] | None = None,
    executor: SimulationExecutor | None = None,
) -> DecisionSimulationRunRecord:
    snapshot = dict(evidence_snapshot or previous.evidence_snapshot)
    if previous.mode == "shadow":
        public_decision = previous.evidence_snapshot.get("shadowPublicDecision")
        if not isinstance(public_decision, Mapping):
            public_decision = previous.evidence_snapshot.get("managerDecision")
        if isinstance(public_decision, Mapping):
            snapshot["shadowPublicDecision"] = dict(public_decision)
    snapshot_hash = canonical_snapshot_hash(snapshot)
    new_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"decision-simulation-retry:{previous.id}:{idempotency_key}:{snapshot_hash}",
        )
    )
    payloads = snapshot.get("simulationCandidates")
    if not isinstance(payloads, list):
        raise ValueError("simulation snapshot has no simulationCandidates")
    pricing_retry = (
        previous.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION
    )
    if pricing_retry:
        raw_request = snapshot.get("simulationRequest")
        if not isinstance(raw_request, Mapping):
            raise ValueError("pricing retry snapshot has no simulationRequest")
        pricing_request = PricingSimulationRequestV1.model_validate(
            {
                **dict(raw_request),
                "snapshotHash": snapshot_hash,
                "modelVersion": previous.model_version,
                "sampleCount": previous.sample_count,
                "batchSize": previous.batch_size,
                "candidates": payloads,
            }
        )
        candidates: list[Any] = list(pricing_request.candidates)
        master_seed = derive_pricing_master_seed(
            snapshot_hash,
            pricing_request.decision_group_id,
        )
    else:
        candidates = [
            AdvertisingPauseCandidateV1.model_validate(item) for item in payloads
        ]
        master_seed = derive_master_seed(snapshot_hash)
    if not 3 <= len(candidates) <= 7:
        raise ValueError("retry snapshot must contain 3-7 candidates")

    original = _decision_from_snapshot(previous)
    if previous.mode == "shadow":
        shadow_payload = previous.evidence_snapshot.get("shadowPublicDecision")
        if isinstance(shadow_payload, Mapping):
            original = ManagerDecisionV1.model_validate(shadow_payload)
    placeholder = ManagerDecisionV1.model_validate(
        {
            **original.model_dump(mode="json", by_alias=True),
            "status": (
                "evaluating"
                if previous.mode == "enforced" or pricing_retry
                else original.status
            ),
            "rankedActions": [] if previous.mode == "enforced" or pricing_retry else [
                item.model_dump(mode="json", by_alias=True) for item in original.ranked_actions
            ],
            "blockers": [],
        }
    )

    def shadow_failure_projection(*, code: str, message: str) -> ManagerDecisionV1:
        if not pricing_retry:
            return original
        return ManagerDecisionV1.model_validate(
            {
                **original.model_dump(mode="json", by_alias=True),
                "status": "needs_data",
                "rankedActions": [],
                "blockers": [
                    DecisionBlockerV1(
                        code=code,
                        message=message,
                        severity="needs_data",
                    ).model_dump(mode="json", by_alias=True)
                ],
            }
        )
    new_run = create_decision_simulation_run(
        simulation_run_id=new_id,
        decision_id=previous.decision_id,
        chat_id=previous.chat_id,
        run_id=previous.run_id,
        mode=previous.mode,
        model_version=previous.model_version,
        evidence_snapshot=snapshot,
        evidence_hash=snapshot_hash,
        master_seed=master_seed,
        sample_count=previous.sample_count,
        batch_size=previous.batch_size,
        candidate_count=len(candidates),
        objective=previous.objective,
        idempotency_key=idempotency_key,
        retry_of_run_id=previous.id,
        retry_attempt=previous.retry_attempt + 1,
        status="initializing",
    )
    claimed = claim_decision_simulation_initialization(new_run.id)
    if claimed is None:
        # The unique child/idempotency constraints selected another concurrent
        # creator. Its initialization and delivery are authoritative.
        current = get_decision_simulation_run(new_run.id)
        return current or new_run
    new_run = claimed
    try:
        for candidate in candidates:
            upsert_decision_simulation_candidate(
                simulation_run_id=new_run.id,
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                candidate_payload=candidate.model_dump(mode="json", by_alias=True),
            )
        mark_decision_simulation_run_superseded(
            previous.id,
            superseded_by_run_id=new_run.id,
        )
        set_manager_decision_simulation_run(
            previous.decision_id,
            simulation_run_id=new_run.id,
            status=(
                "evaluating"
                if previous.mode == "enforced" or pricing_retry
                else None
            ),
        )
        persist_manager_decision_bundle(
            placeholder,
            chat_id=previous.chat_id,
            run_id=previous.run_id,
            source_key=None,
            create_experiments=False,
        )
        new_run = update_decision_simulation_run(
            new_run.id,
            status="queued",
            heartbeat_at=datetime.now(timezone.utc),
            completed_at=None,
            error={},
        )
        append_decision_simulation_event(
            simulation_run_id=new_run.id,
            event_type="queued",
            event_id=f"{new_run.id}:queued",
            progress=0,
            payload={
                "completedSamples": 0,
                "totalSamples": new_run.total_samples,
                "retryOfRunId": previous.id,
                "retryAttempt": new_run.retry_attempt,
            },
        )
    except Exception as exc:
        logger.exception(
            "decision_simulation_retry_initialization_failed "
            "simulation_run_id=%s retry_of_run_id=%s",
            new_run.id,
            previous.id,
        )
        error = {"code": "retry_initialization_failed", "message": str(exc)}
        try:
            new_run = update_decision_simulation_run(
                new_run.id,
                status="failed",
                winner_candidate_id=None,
                error=error,
            )
            append_decision_simulation_event(
                simulation_run_id=new_run.id,
                event_type="failed",
                event_id=f"{new_run.id}:failed:initialization",
                payload=error,
            )
            if previous.mode == "enforced":
                block_manager_decision_for_simulation(
                    new_run,
                    code="decision_simulation_retry_initialization_failed",
                    message="Не удалось надёжно инициализировать пересчёт Decision Simulation.",
                )
            else:
                restored = shadow_failure_projection(
                    code="pricing_simulation_retry_initialization_failed",
                    message="Не удалось надёжно инициализировать пересчёт цены.",
                )
                persist_manager_decision_bundle(
                    restored,
                    chat_id=previous.chat_id,
                    run_id=previous.run_id,
                    source_key=None,
                    create_experiments=False,
                )
                set_manager_decision_simulation_run(
                    previous.decision_id,
                    simulation_run_id=new_run.id,
                    status="needs_data" if pricing_retry else None,
                )
        except Exception:
            logger.exception(
                "decision_simulation_retry_initialization_failure_not_persisted "
                "simulation_run_id=%s",
                new_run.id,
            )
        raise
    try:
        (executor or RQSimulationExecutor()).enqueue(new_run.id)
    except Exception as exc:
        logger.exception(
            "decision_simulation_retry_enqueue_failed simulation_run_id=%s retry_of_run_id=%s",
            new_run.id,
            previous.id,
        )
        error = {"code": "retry_enqueue_failed", "message": str(exc)}
        new_run = update_decision_simulation_run(
            new_run.id,
            status="failed",
            error=error,
        )
        append_decision_simulation_event(
            simulation_run_id=new_run.id,
            event_type="failed",
            event_id=f"{new_run.id}:failed:enqueue",
            payload=error,
        )
        if previous.mode == "enforced":
            block_manager_decision_for_simulation(
                new_run,
                code="decision_simulation_retry_enqueue_failed",
                message="Не удалось поставить пересчёт Decision Simulation в очередь.",
            )
        else:
            restored = shadow_failure_projection(
                code="pricing_simulation_retry_enqueue_failed",
                message="Не удалось поставить пересчёт цены в очередь.",
            )
            persist_manager_decision_bundle(
                restored,
                chat_id=previous.chat_id,
                run_id=previous.run_id,
                source_key=None,
                create_experiments=not pricing_retry,
            )
            if pricing_retry:
                set_manager_decision_simulation_run(
                    previous.decision_id,
                    simulation_run_id=new_run.id,
                    status="needs_data",
                )
    return new_run


def retry_decision_simulation(
    *,
    simulation_run_id: str,
    idempotency_key: str,
    actor: ManagerActor,
) -> DecisionSimulationRunRecord:
    require_role(actor, "admin")
    key = str(idempotency_key or "").strip()
    if not key:
        raise ValueError("Idempotency-Key must not be blank")
    previous = get_decision_simulation_run(simulation_run_id)
    if previous is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    existing_retry = next(
        (
            item
            for item in list_decision_simulation_runs(
                decision_id=previous.decision_id,
                limit=100,
            )
            if item.retry_of_run_id == previous.id and item.idempotency_key == key
        ),
        None,
    )
    if existing_retry is not None:
        return existing_retry
    if previous.status not in {"failed", "blocked"}:
        raise ManagerLifecycleConflict("simulation retry is allowed only for failed/blocked runs")
    return _create_retry_run(previous, idempotency_key=key)


def create_critic_recalculation(
    previous: DecisionSimulationRunRecord,
    *,
    evidence_snapshot: dict[str, Any] | None = None,
) -> DecisionSimulationRunRecord:
    if previous.retry_attempt >= 1:
        raise ManagerLifecycleConflict("automatic critic recalculation is already exhausted")
    return _create_retry_run(
        previous,
        idempotency_key=f"critic-recalculate:{previous.id}",
        evidence_snapshot=evidence_snapshot,
    )


__all__ = [
    "RQSimulationExecutor",
    "SIMULATION_SNAPSHOT_SCHEMA_VERSION",
    "SimulationExecutor",
    "advertising_candidate_from_action",
    "block_manager_decision_for_simulation",
    "build_evidence_snapshot",
    "canonical_snapshot_hash",
    "create_critic_recalculation",
    "decision_simulation_job_id",
    "enqueue_decision_simulation",
    "maybe_schedule_decision_simulation",
    "retry_decision_simulation",
]
