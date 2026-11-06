from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Sequence

from agi_ecom_app.config import get_settings
from agi_ecom_app.observability import trace_simulation_job
from agi_ecom_app.decision_simulation.builtin_plugins import (
    ensure_builtin_simulation_plugins_registered,
)
from agi_ecom_app.decision_simulation.critic import (
    DecisionCriticExecution,
    DecisionCriticInput,
    DecisionCriticIssue,
    DecisionCriticResult,
    run_decision_critic,
)
from agi_ecom_app.decision_simulation.engine import (
    DecisionSimulationEngine,
    simulation_runtime_card,
)
from agi_ecom_app.decision_simulation.metrics import (
    ACTIVE_JOBS,
    BATCH_DURATION,
    CRITIC_VERDICTS,
    FAILURES,
    FORECAST_COVERAGE,
    JOB_DURATION,
    REJECTED_CANDIDATES,
    SAMPLES_COMPLETED,
    SAMPLES_PER_SECOND,
    STALE_JOBS,
)
from agi_ecom_app.decision_simulation.models import (
    AdvertisingPauseCandidateV1,
    CandidateSimulationResultV1,
    ConstraintViolationV1,
    DecisionSimulationRequestV1,
    DistributionPairV1,
    RankingScoreBreakdownV1,
    SimulationBatchResultV1,
)
from agi_ecom_app.decision_simulation.pricing_engine import (
    MAX_DEMAND_OBSERVATION_AGE_DAYS,
    MAX_STOCK_AGE_DAYS,
    MAX_UNIT_ECONOMICS_AGE_DAYS,
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    PricingSimulationEngine,
)
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingCandidateSimulationResultV1,
    PricingCandidateV1,
    PricingRankingScoreV1,
    PricingSimulationBatchResultV1,
    PricingSimulationRequestV1,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    PRICING_SOURCE_SCHEMA_VERSION,
    load_pricing_simulation_source,
    pricing_request_snapshot_payload,
)
from agi_ecom_app.decision_simulation.plugins import (
    get_decision_simulation_plugin,
)
from agi_ecom_app.decision_simulation.service import (
    RQSimulationExecutor,
    advertising_candidate_from_action,
    block_manager_decision_for_simulation,
    build_evidence_snapshot,
    canonical_snapshot_hash,
    create_critic_recalculation,
)
from agi_ecom_app.manager_decision.candidate_builder import build_manager_action_candidates
from agi_ecom_app.manager_decision.candidate_source import load_manager_candidate_snapshot
from agi_ecom_app.manager_decision.models import (
    DecisionBlockerV1,
    EconomicRangeV1,
    ManagerActionV1,
    ManagerDecisionV1,
)
from agi_ecom_app.manager_lifecycle.service import persist_manager_decision_bundle
from agi_ecom_app.services.memory import (
    DecisionSimulationRunRecord,
    append_decision_simulation_event,
    backfill_decision_simulation_superseded_events,
    claim_stale_decision_simulation_run,
    clear_decision_simulation_winner_flags,
    complete_decision_simulation_run,
    get_decision_simulation_run,
    get_manager_decision,
    get_memory_connection,
    list_decision_simulation_batches,
    list_decision_simulation_candidates,
    list_decision_simulation_events,
    list_decision_simulation_runs,
    list_agent_action_experiments_for_decision,
    list_stale_decision_simulation_runs,
    list_unprojected_terminal_decision_simulation_runs,
    set_manager_decision_simulation_run,
    touch_decision_simulation_heartbeat,
    update_decision_simulation_progress,
    update_decision_simulation_run,
    upsert_decision_simulation_batch,
    upsert_decision_simulation_candidate,
)


logger = logging.getLogger(__name__)
_RUN_TERMINAL = frozenset({"completed", "blocked", "failed", "superseded"})


@contextmanager
def _periodic_simulation_heartbeat(
    simulation_run_id: str,
    *,
    interval_seconds: float,
) -> Iterator[None]:
    """Keep a synchronous external critic call from looking abandoned."""

    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(max(1.0, interval_seconds)):
            try:
                touch_decision_simulation_heartbeat(simulation_run_id)
            except Exception:
                logger.exception(
                    "decision_simulation_critic_heartbeat_failed "
                    "simulation_run_id=%s",
                    simulation_run_id,
                )

    thread = threading.Thread(
        target=beat,
        name=f"decision-simulation-heartbeat-{simulation_run_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


@contextmanager
def _simulation_run_lock(simulation_run_id: str) -> Iterator[bool]:
    """A connection-scoped lock makes duplicate RQ delivery harmless."""

    with get_memory_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select pg_try_advisory_lock(hashtext(%s)::bigint) as acquired",
                [f"decision-simulation:{simulation_run_id}"],
            )
            row = cur.fetchone()
            acquired = bool(row["acquired"] if isinstance(row, dict) else row[0])
        try:
            yield acquired
        finally:
            if acquired:
                with conn.cursor() as cur:
                    cur.execute(
                        "select pg_advisory_unlock(hashtext(%s)::bigint)",
                        [f"decision-simulation:{simulation_run_id}"],
                    )


def _event(
    run: DecisionSimulationRunRecord,
    event_type: str,
    *,
    suffix: str,
    candidate_id: str | None = None,
    batch_index: int | None = None,
    progress: float | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    append_decision_simulation_event(
        simulation_run_id=run.id,
        event_type=event_type,
        event_id=f"{run.id}:{suffix}",
        candidate_id=candidate_id,
        batch_index=batch_index,
        progress=progress,
        payload=payload or {},
    )


def _ranking_key(
    item: Any,
) -> tuple[float, float, float, float, str]:
    score = item.score_breakdown
    if score is None:
        raise ValueError("completed candidate has no score breakdown")
    risk_or_move = getattr(score, "execution_risk", None)
    if risk_or_move is None:
        risk_or_move = getattr(score, "absolute_price_move_rub", None)
    if risk_or_move is None:
        raise ValueError("completed candidate score has no deterministic risk tie-break")
    return (
        -score.cvar_10_per_day,
        -score.p10_per_day,
        -score.mean_per_day,
        float(risk_or_move),
        item.action_id,
    )


def _ranked(
    results: Sequence[Any],
) -> list[Any]:
    ordered = sorted(
        (item for item in results if item.status == "completed"),
        key=_ranking_key,
    )
    return [
        item.model_copy(
            # Until critic completion this is only a provisional numerical
            # order. The durable winner flag is assigned at finalization.
            update={"rank": index, "is_winner": False}
        )
        for index, item in enumerate(ordered, start=1)
    ]


def _request_from_run(run: DecisionSimulationRunRecord) -> Any:
    actual_hash = canonical_snapshot_hash(run.evidence_snapshot)
    if actual_hash != run.evidence_hash:
        raise ValueError("evidence snapshot SHA-256 does not match the durable hash")
    ensure_builtin_simulation_plugins_registered()
    plugin = get_decision_simulation_plugin(run.model_version)
    expected_runtime = run.evidence_snapshot.get("modelRuntime")
    if expected_runtime != plugin.build_runtime_card():
        raise ValueError(
            "simulation runtime does not match the pinned snapshot model card"
        )
    if run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION:
        raw_request = run.evidence_snapshot.get("simulationRequest")
        if not isinstance(raw_request, dict):
            raise ValueError("pricing simulation snapshot has no simulationRequest")
        return PricingSimulationRequestV1.model_validate(
            {
                **raw_request,
                "snapshotHash": run.evidence_hash,
                "modelVersion": run.model_version,
                "sampleCount": run.sample_count,
                "batchSize": run.batch_size,
            }
        )
    raw_candidates = run.evidence_snapshot.get("simulationCandidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("simulation snapshot has no candidates")
    return DecisionSimulationRequestV1(
        snapshotHash=run.evidence_hash,
        candidates=[AdvertisingPauseCandidateV1.model_validate(item) for item in raw_candidates],
        modelVersion=run.model_version,
        sampleCount=run.sample_count,
        batchSize=run.batch_size,
    )


def _candidate_payload(
    request: Any,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = next(item for item in request.candidates if item.candidate_id == candidate_id)
    return candidate.model_dump(mode="json", by_alias=True)


def _persist_candidate_result(
    run: DecisionSimulationRunRecord,
    request: Any,
    result: Any,
) -> None:
    distribution = result.distribution
    score = result.score_breakdown
    status = "rejected" if result.status == "blocked" else result.status
    upsert_decision_simulation_candidate(
        simulation_run_id=run.id,
        candidate_id=result.candidate_id,
        action_id=result.action_id,
        status=status,
        candidate_payload=_candidate_payload(request, result.candidate_id),
        evidence_refs=result.evidence_refs,
        constraint_violations=[
            item.model_dump(mode="json", by_alias=True)
            for item in result.constraint_violations
        ],
        distribution_summary=(
            distribution.model_dump(mode="json", by_alias=True) if distribution else {}
        ),
        score_breakdown=_durable_score_payload(result),
        expected_effect_mean=(distribution.per_day.mean if distribution else None),
        profit_p10=(distribution.per_day.p10 if distribution else None),
        cvar_10_per_day=(score.cvar_10_per_day if score else None),
        execution_risk=(
            getattr(score, "execution_risk", None)
            if score is not None
            else None
        ),
        utility_score=(score.cvar_10_per_day if score else None),
        rank=result.rank,
        is_winner=result.is_winner,
        completed_samples=(distribution.total.count if distribution else 0),
    )


def _durable_score_payload(result: Any) -> dict[str, Any]:
    score = result.score_breakdown
    if score is None:
        return {}
    payload = score.model_dump(mode="json", by_alias=True)
    if isinstance(result, PricingCandidateSimulationResultV1):
        payload.update(
            {
                "unitsSoldDistribution": (
                    result.units_sold_distribution.model_dump(
                        mode="json", by_alias=True
                    )
                    if result.units_sold_distribution is not None
                    else None
                ),
                "stockoutProbability": result.stockout_probability,
                "unitContributionRub": result.unit_contribution_rub,
            }
        )
    return payload


def _restore_completed_candidate(
    candidate: Any,
    stored: Any,
    batches: Sequence[Any],
    *,
    expected_batch_count: int,
) -> Any | None:
    """Resume without re-running a candidate whose durable batches are complete."""

    if stored is None or stored.status != "completed":
        return None
    ordered = sorted(batches, key=lambda item: item.batch_index)
    if [item.batch_index for item in ordered] != list(range(expected_batch_count)):
        return None
    try:
        if isinstance(candidate, PricingCandidateV1):
            return PricingCandidateSimulationResultV1(
                candidateId=candidate.candidate_id,
                actionId=candidate.action_id,
                operation=candidate.operation,
                priceRub=candidate.price_rub,
                status="completed",
                evidenceRefs=[str(item) for item in stored.evidence_refs],
                batchResults=[
                    _pricing_batch_from_record(candidate, batch)
                    for batch in ordered
                ],
                distribution=DistributionPairV1.model_validate(
                    stored.distribution_summary
                ),
                unitsSoldDistribution=DistributionPairV1.model_validate(
                    (stored.score_breakdown or {}).get("unitsSoldDistribution")
                ),
                stockoutProbability=(stored.score_breakdown or {}).get(
                    "stockoutProbability"
                ),
                unitContributionRub=(stored.score_breakdown or {}).get(
                    "unitContributionRub"
                ),
                scoreBreakdown=PricingRankingScoreV1.model_validate(
                    {
                        key: value
                        for key, value in (stored.score_breakdown or {}).items()
                        if key
                        in {
                            "cvar10PerDay",
                            "p10PerDay",
                            "meanPerDay",
                            "absolutePriceMoveRub",
                        }
                    }
                ),
                rank=stored.rank,
                isWinner=stored.is_winner,
            )
        return CandidateSimulationResultV1(
            candidateId=candidate.candidate_id,
            actionId=candidate.action_id,
            status="completed",
            evidenceRefs=[str(item) for item in stored.evidence_refs],
            batchResults=[
                SimulationBatchResultV1(
                    candidateId=candidate.candidate_id,
                    batchIndex=batch.batch_index,
                    seed=batch.seed,
                    sampleCount=batch.sample_count,
                    distribution=DistributionPairV1.model_validate(batch.aggregates),
                )
                for batch in ordered
            ],
            distribution=DistributionPairV1.model_validate(
                stored.distribution_summary
            ),
            scoreBreakdown=RankingScoreBreakdownV1.model_validate(
                stored.score_breakdown
            ),
            rank=stored.rank,
            isWinner=stored.is_winner,
        )
    except Exception:
        logger.exception(
            "decision_simulation_resume_payload_invalid simulation_run_id=%s candidate_id=%s",
            stored.simulation_run_id,
            candidate.candidate_id,
        )
        return None


def _pricing_batch_from_record(
    candidate: PricingCandidateV1,
    batch: Any,
) -> PricingSimulationBatchResultV1:
    aggregates = batch.aggregates if isinstance(batch.aggregates, dict) else {}
    return PricingSimulationBatchResultV1(
        candidateId=candidate.candidate_id,
        batchIndex=batch.batch_index,
        seed=batch.seed,
        sampleCount=batch.sample_count,
        distribution=DistributionPairV1.model_validate(
            aggregates.get("distribution")
        ),
        unitsSoldDistribution=DistributionPairV1.model_validate(
            aggregates.get("unitsSoldDistribution")
        ),
        stockoutProbability=aggregates.get("stockoutProbability"),
    )


def _batch_from_record(request: Any, candidate: Any, batch: Any) -> Any:
    if isinstance(request, PricingSimulationRequestV1):
        return _pricing_batch_from_record(candidate, batch)
    return SimulationBatchResultV1(
        candidateId=candidate.candidate_id,
        batchIndex=batch.batch_index,
        seed=batch.seed,
        sampleCount=batch.sample_count,
        distribution=DistributionPairV1.model_validate(batch.aggregates),
    )


def _batch_aggregates(batch: Any) -> dict[str, Any]:
    if isinstance(batch, PricingSimulationBatchResultV1):
        return {
            "distribution": batch.distribution.model_dump(
                mode="json", by_alias=True
            ),
            "unitsSoldDistribution": batch.units_sold_distribution.model_dump(
                mode="json", by_alias=True
            ),
            "stockoutProbability": batch.stockout_probability,
        }
    return batch.distribution.model_dump(mode="json", by_alias=True)


def _engine_for_request(request: Any) -> Any:
    ensure_builtin_simulation_plugins_registered()
    plugin = get_decision_simulation_plugin(request.model_version)
    # Keep the two built-in names as injectable test seams while every unknown
    # or future model is routed exclusively through its registered plugin.
    if isinstance(request, PricingSimulationRequestV1):
        return PricingSimulationEngine()
    if isinstance(request, DecisionSimulationRequestV1):
        return DecisionSimulationEngine()
    return plugin.create_engine()


def _candidate_violations(engine: Any, request: Any, candidate: Any) -> list[Any]:
    del engine
    ensure_builtin_simulation_plugins_registered()
    plugin = get_decision_simulation_plugin(request.model_version)
    return plugin.validate_candidate(request, candidate)


def _critic_input(
    run: DecisionSimulationRunRecord,
    ranked: Sequence[Any],
    rejected: Sequence[Any],
) -> DecisionCriticInput:
    top_three = [item.candidate_id for item in ranked[:3]]
    leader = ranked[0].score_breakdown
    runner_up = ranked[1].score_breakdown if len(ranked) > 1 else None
    if leader is None:
        raise ValueError("ranking leader has no score")
    manager_payload = run.evidence_snapshot.get("managerDecision") or {}
    manager_decision = ManagerDecisionV1.model_validate(manager_payload)
    actions_by_id = {
        action.action_id: action for action in manager_decision.ranked_actions
    }
    evidence_as_of = manager_payload.get("evidenceAsOf")
    evidence_refs = list(
        dict.fromkeys(
            ref
            for item in [*ranked, *rejected]
            for ref in item.evidence_refs
        )
    )
    raw_hard_gate_rejected = run.evidence_snapshot.get("hardGateRejectedCandidates")
    pre_rejected = (
        [item for item in raw_hard_gate_rejected if isinstance(item, dict)]
        if isinstance(raw_hard_gate_rejected, list)
        else []
    )

    def collect_refs(value: Any, target: list[str]) -> None:
        if isinstance(value, dict):
            ref = value.get("evidenceRef")
            if isinstance(ref, str) and ref and ref not in target:
                target.append(ref)
            for nested in value.values():
                collect_refs(nested, target)
        elif isinstance(value, list):
            for nested in value:
                collect_refs(nested, target)

    def action_contract(action: ManagerActionV1) -> dict[str, Any]:
        return {
            "operation": action.operation,
            "executionMode": action.execution_mode,
            "executionPayload": action.execution_payload,
            "requiredMetrics": action.required_metrics,
            "prerequisites": action.prerequisites,
            "stopCondition": action.stop_condition,
            "rollback": action.rollback.model_dump(mode="json", by_alias=True),
            "measurementSpec": action.measurement_spec.model_dump(
                mode="json", by_alias=True
            ),
            "guardrails": [
                guardrail.model_dump(mode="json", by_alias=True)
                for guardrail in action.guardrails
            ],
            "expiresAt": action.expires_at.isoformat(),
        }

    aggregates: list[dict[str, Any]] = []
    for item in [*ranked, *rejected]:
        action = actions_by_id.get(item.action_id)
        aggregate: dict[str, Any] = {
            "candidateId": item.candidate_id,
            "actionId": item.action_id,
            "status": item.status,
            "rank": item.rank,
            "evidenceRefs": item.evidence_refs,
            "constraintViolations": [
                violation.model_dump(mode="json", by_alias=True)
                for violation in item.constraint_violations
            ],
        }
        if item.distribution is not None:
            aggregate["distribution"] = item.distribution.model_dump(
                mode="json", by_alias=True
            )
        if item.score_breakdown is not None:
            aggregate["scoreBreakdown"] = item.score_breakdown.model_dump(
                mode="json", by_alias=True
            )
        if isinstance(item, PricingCandidateSimulationResultV1):
            aggregate["pricing"] = {
                "operation": item.operation,
                "priceRub": item.price_rub,
                "unitContributionRub": item.unit_contribution_rub,
                "stockoutProbability": item.stockout_probability,
                "unitsSoldDistribution": (
                    item.units_sold_distribution.model_dump(
                        mode="json", by_alias=True
                    )
                    if item.units_sold_distribution is not None
                    else None
                ),
            }
        if action is not None:
            aggregate["action"] = action_contract(action)
        aggregates.append(aggregate)

    pre_hard_constraints: list[dict[str, Any]] = []
    for entry in pre_rejected:
        candidate_payload = entry.get("candidate")
        violations = entry.get("violations")
        if not isinstance(candidate_payload, dict) or not isinstance(violations, list):
            continue
        candidate_id = str(candidate_payload.get("candidateId") or "").strip()
        action_id = str(candidate_payload.get("actionId") or candidate_id).strip()
        if not candidate_id or any(
            aggregate.get("candidateId") == candidate_id for aggregate in aggregates
        ):
            continue
        candidate_refs: list[str] = []
        collect_refs(candidate_payload, candidate_refs)
        collect_refs(violations, candidate_refs)
        for ref in candidate_refs:
            if ref not in evidence_refs:
                evidence_refs.append(ref)
        status = (
            "needs_data"
            if any(
                isinstance(violation, dict)
                and violation.get("severity") == "needs_data"
                for violation in violations
            )
            else "blocked"
        )
        aggregate = {
            "candidateId": candidate_id,
            "actionId": action_id,
            "status": status,
            "rank": None,
            "evidenceRefs": candidate_refs,
            "constraintViolations": violations,
        }
        action = actions_by_id.get(action_id)
        if action is not None:
            aggregate["action"] = action_contract(action)
        aggregates.append(aggregate)
        pre_hard_constraints.append(
            {"candidateId": candidate_id, "violations": violations}
        )

    aggregate_ids = {str(item.get("candidateId")) for item in aggregates}
    fact_freshness = [
        {
            "candidateId": action.action_id,
            "metric": fact.metric,
            "evidenceRef": fact.evidence_ref,
            "snapshotAt": fact.snapshot_at.isoformat() if fact.snapshot_at else None,
            "retrievedAt": fact.retrieved_at.isoformat() if fact.retrieved_at else None,
            "freshnessStatus": fact.freshness_status,
        }
        for action in manager_decision.ranked_actions
        for fact in action.evidence
        if action.action_id in aggregate_ids
    ]
    if run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION:
        pricing_request = _request_from_run(run)
        pricing_evidence = pricing_request.evidence
        pricing_facts = (
            (
                "price_demand_history",
                pricing_evidence.daily_observations.evidence_ref,
                pricing_evidence.daily_observations.latest_observation_age_days,
                MAX_DEMAND_OBSERVATION_AGE_DAYS,
            ),
            (
                "unit_economics",
                pricing_evidence.unit_economics.evidence_ref,
                pricing_evidence.unit_economics.latest_observation_age_days,
                MAX_UNIT_ECONOMICS_AGE_DAYS,
            ),
            (
                "available_stock",
                pricing_evidence.available_stock.evidence_ref,
                pricing_evidence.available_stock.age_days_at_snapshot,
                MAX_STOCK_AGE_DAYS,
            ),
        )
        fact_freshness.extend(
            {
                "candidateId": pricing_request.decision_group_id,
                "metric": metric,
                "evidenceRef": evidence_ref,
                "snapshotAt": None,
                "retrievedAt": None,
                "ageDaysAtSnapshot": age_days,
                "freshnessStatus": (
                    "unknown"
                    if age_days is None
                    else "fresh" if age_days <= maximum_age else "stale"
                ),
            }
            for metric, evidence_ref, age_days, maximum_age in pricing_facts
        )
        model_card = {
            "name": run.model_version,
            "domain": "pricing",
            "objective": run.objective,
            "sampleCountPerCandidate": run.sample_count,
            "batchSize": run.batch_size,
            "uncertaintySource": (
                "completed in-stock price-volume days plus empirical "
                "elasticity residual bootstrap"
            ),
            "causalScope": (
                "observational shadow estimate; it cannot authorize a price write"
            ),
            "selectionMode": "winner_only",
        }
    else:
        model_card = {
            "name": run.model_version,
            "domain": "advertising",
            "objective": run.objective,
            "sampleCountPerCandidate": run.sample_count,
            "batchSize": run.batch_size,
            "uncertaintySource": "seven completed empirical daily ad spends",
            "causalScope": "avoided advertising spend for a proven zero-stock pause",
        }
    return DecisionCriticInput(
        objective=run.objective,
        modelCard=model_card,
        snapshotFreshness={
            "snapshotHash": run.evidence_hash,
            "evidenceAsOf": evidence_as_of,
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "evidenceRefs": evidence_refs,
            "facts": fact_freshness,
        },
        candidateAggregates=aggregates,
        top3=top_three,
        leaderDelta={
            "leaderCandidateId": top_three[0],
            "runnerUpCandidateId": top_three[1] if len(top_three) > 1 else None,
            "cvar10PerDay": (
                leader.cvar_10_per_day - runner_up.cvar_10_per_day
                if runner_up is not None
                else leader.cvar_10_per_day
            ),
            "p10PerDay": (
                leader.p10_per_day - runner_up.p10_per_day
                if runner_up is not None
                else leader.p10_per_day
            ),
            "meanPerDay": (
                leader.mean_per_day - runner_up.mean_per_day
                if runner_up is not None
                else leader.mean_per_day
            ),
        },
        hardConstraints=[
            {
                "candidateId": item.candidate_id,
                "violations": [
                    violation.model_dump(mode="json", by_alias=True)
                    for violation in item.constraint_violations
                ],
            }
            for item in rejected
        ] + pre_hard_constraints,
        evidenceRefs=evidence_refs,
        recalculationAttempt=min(run.retry_attempt, 1),
    )


def _disabled_critic_execution(
    critic_input: DecisionCriticInput,
    *,
    mode: str,
) -> DecisionCriticExecution:
    result = DecisionCriticResult(
        verdict="blocked",
        issues=[
            DecisionCriticIssue(
                code="llm_critic_unavailable",
                message="LLM Critic disabled by configuration.",
            )
        ],
        summary="Numerical ranking is unchanged; critic is unavailable.",
    )
    effective_action = "record_only" if mode == "shadow" else "block"
    return DecisionCriticExecution(
        mode=mode,
        result=result,
        effectiveAction=effective_action,
        llmAvailable=False,
        blocksDecision=effective_action == "block",
        rankingFingerprint=critic_input.ranking_fingerprint,
        debug={"reason": "disabled_by_configuration"},
    )


def _indeterminate_critic_execution(
    critic_input: DecisionCriticInput,
    *,
    mode: str,
) -> DecisionCriticExecution:
    """Fail closed after a crash window instead of repeating an LLM request."""

    effective_action = "record_only" if mode == "shadow" else "block"
    return DecisionCriticExecution(
        mode=mode,
        result=DecisionCriticResult(
            verdict="blocked",
            issues=[
                DecisionCriticIssue(
                    code="llm_critic_unavailable",
                    message=(
                        "A prior critic request was durably claimed but its response "
                        "was not committed; automatic resubmission is disabled."
                    ),
                )
            ],
            summary=(
                "Numerical ranking is unchanged; critic delivery is indeterminate."
            ),
        ),
        effectiveAction=effective_action,
        llmAvailable=False,
        blocksDecision=effective_action == "block",
        rankingFingerprint=critic_input.ranking_fingerprint,
        debug={"reason": "durable_in_flight_claim_without_result"},
    )


def _execute_or_reuse_decision_critic(
    run: DecisionSimulationRunRecord,
    critic_input: DecisionCriticInput,
    *,
    settings: Any,
) -> tuple[DecisionCriticExecution, bool]:
    """Execute the critic at most once across worker crash recovery.

    The durable ``in_flight`` marker is deliberately treated as an
    indeterminate delivery, not as permission to resubmit the external LLM
    request. This favors a fail-closed result over asking the same question
    twice after a crash between delivery and response persistence.
    """

    critic_result = run.critic_result if isinstance(run.critic_result, dict) else {}
    submission_status = critic_result.get("submissionStatus")
    if critic_result and submission_status != "in_flight":
        return DecisionCriticExecution.model_validate(critic_result), True
    if submission_status == "in_flight":
        return _indeterminate_critic_execution(critic_input, mode=run.mode), False
    if not settings.decision_simulation_llm_critic_enabled:
        return _disabled_critic_execution(critic_input, mode=run.mode), False

    update_decision_simulation_run(
        run.id,
        critic_result={
            "submissionStatus": "in_flight",
            "rankingFingerprint": critic_input.ranking_fingerprint,
            "claimedAt": datetime.now(timezone.utc).isoformat(),
        },
    )
    heartbeat_interval = min(
        30.0,
        max(
            1.0,
            float(
                getattr(
                    settings,
                    "decision_simulation_stale_after_seconds",
                    1_200,
                )
            )
            / 3,
        ),
    )
    with _periodic_simulation_heartbeat(
        run.id,
        interval_seconds=heartbeat_interval,
    ):
        return (
            run_decision_critic(
                critic_input=critic_input,
                mode=run.mode,
            ),
            False,
        )


def _original_decision(run: DecisionSimulationRunRecord) -> ManagerDecisionV1:
    payload = run.evidence_snapshot.get("managerDecision")
    if not isinstance(payload, dict):
        raise ValueError("simulation snapshot has no manager decision")
    return ManagerDecisionV1.model_validate(payload)


def _shadow_public_decision(run: DecisionSimulationRunRecord) -> ManagerDecisionV1:
    payload = run.evidence_snapshot.get("shadowPublicDecision")
    if not isinstance(payload, dict):
        decision = _original_decision(run)
    else:
        decision = ManagerDecisionV1.model_validate(payload)
    eligibility = decision.simulation_eligibility
    if (
        str(run.error.get("code") or "")
        in {"initialization_failed", "enqueue_failed"}
        and eligibility is not None
        and eligibility.outcome == "queued"
    ):
        decision = decision.model_copy(
            update={
                "simulation_eligibility": eligibility.model_copy(
                    update={
                        "outcome": "eligible",
                        "reason_code": "eligible_candidates_delivery_failed",
                    }
                )
            }
        )
    return decision


def _publish_shadow_decision(run: DecisionSimulationRunRecord) -> None:
    """Materialize the unchanged public ranking after the shadow run is terminal."""

    if run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION:
        original = _original_decision(run)
        final = ManagerDecisionV1.model_validate(
            {
                **original.model_dump(mode="json", by_alias=True),
                "status": "needs_data",
                "rankedActions": [],
                "blockers": [
                    DecisionBlockerV1(
                        code="pricing_shadow_only",
                        message=(
                            "Ценовая симуляция завершена в shadow-режиме; "
                            "winner доступен в simulationSummary, но не может "
                            "быть отправлен в WB до появления complete watermarks "
                            "и измеренной калибровки."
                        ),
                        severity="needs_data",
                    ).model_dump(mode="json", by_alias=True)
                ],
            }
        )
        persist_manager_decision_bundle(
            final,
            chat_id=run.chat_id,
            run_id=run.run_id,
            source_key=None,
            create_experiments=False,
        )
        set_manager_decision_simulation_run(
            run.decision_id,
            simulation_run_id=run.id,
            status="needs_data",
        )
        return

    persist_manager_decision_bundle(
        _shadow_public_decision(run),
        chat_id=run.chat_id,
        run_id=run.run_id,
        source_key=None,
    )
    set_manager_decision_simulation_run(
        run.decision_id,
        simulation_run_id=run.id,
        status=None,
    )


def _final_action(
    action: ManagerActionV1,
    result: CandidateSimulationResultV1,
    run: DecisionSimulationRunRecord,
) -> ManagerActionV1:
    if result.distribution is None:
        raise ValueError("ranked result has no distribution")
    per_day = result.distribution.per_day
    simulation_ref = f"decision-simulation-run:{run.id}"
    refs = list(dict.fromkeys([*result.evidence_refs, simulation_ref]))
    method = f"{run.model_version};simulation_run_id={run.id}"
    expected_effect = action.expected_effect.model_copy(
        update={
            "contribution_profit_low_per_day": per_day.p10,
            "contribution_profit_base_per_day": per_day.p50,
            "contribution_profit_high_per_day": per_day.p90,
            "calculation_method": method,
            "evidence_refs": refs,
            "causal_valid": False,
        }
    )
    economics = action.economics.model_copy(
        update={
            "delta_advertising_per_day": EconomicRangeV1(
                low=-per_day.p90,
                base=-per_day.p50,
                high=-per_day.p10,
            ),
            "calculation_method": method,
            "evidence_refs": refs,
            "causal_valid": False,
        }
    )
    return ManagerActionV1.model_validate(
        {
            **action.model_dump(mode="json", by_alias=True),
            "expectedEffect": expected_effect.model_dump(mode="json", by_alias=True),
            "economics": economics.model_dump(mode="json", by_alias=True),
        }
    )


def _publish_enforced_decision(
    run: DecisionSimulationRunRecord,
    ranked: Sequence[CandidateSimulationResultV1],
) -> None:
    if run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION:
        raise ValueError(
            "pricing simulation v1 is shadow-only and cannot publish an executable decision"
        )
    original = _original_decision(run)
    actions = {action.action_id: action for action in original.ranked_actions}
    ranked_actions = [_final_action(actions[item.action_id], item, run) for item in ranked]
    final = ManagerDecisionV1.model_validate(
        {
            **original.model_dump(mode="json", by_alias=True),
            "status": "ready_for_approval",
            "rankedActions": [
                action.model_dump(mode="json", by_alias=True) for action in ranked_actions
            ],
            "blockers": [],
        }
    )
    persist_manager_decision_bundle(
        final,
        chat_id=run.chat_id,
        run_id=run.run_id,
        source_key=None,
    )
    set_manager_decision_simulation_run(
        run.decision_id,
        simulation_run_id=run.id,
        status="ready_for_approval",
    )


def _durable_ranked_results(
    run: DecisionSimulationRunRecord,
) -> list[CandidateSimulationResultV1]:
    request = _request_from_run(run)
    stored_candidates = {
        item.candidate_id: item
        for item in list_decision_simulation_candidates(run.id)
    }
    batches_by_candidate: dict[str, list[Any]] = {}
    for batch in list_decision_simulation_batches(run.id, statuses=("completed",)):
        batches_by_candidate.setdefault(batch.candidate_id, []).append(batch)
    restored = [
        result
        for candidate in request.candidates
        if (
            result := _restore_completed_candidate(
                candidate,
                stored_candidates.get(candidate.candidate_id),
                batches_by_candidate.get(candidate.candidate_id, []),
                expected_batch_count=request.sample_count // request.batch_size,
            )
        )
        is not None
    ]
    ranked = _ranked(restored)
    if len(ranked) < 3:
        raise ValueError("terminal projection has fewer than three durable candidates")
    return ranked


def _terminal_projection_is_complete(run: DecisionSimulationRunRecord) -> bool:
    decision = get_manager_decision(run.decision_id)
    if decision is None or decision.simulation_run_id != run.id:
        return False
    event_types = {
        event.event_type
        for event in list_decision_simulation_events(run.id, after_id=0, limit=5000)
    }
    experiment_count = len(list_agent_action_experiments_for_decision(run.decision_id))
    if (
        run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION
        and run.mode == "shadow"
        and run.status in {"completed", "blocked", "failed"}
    ):
        return (
            decision.status == "needs_data"
            and experiment_count == 0
            and run.status in event_types
        )
    if run.status == "completed":
        expected = (
            len(_shadow_public_decision(run).ranked_actions)
            if run.mode == "shadow"
            else run.completed_candidates
        )
        return (
            decision.status == "ready_for_approval"
            and experiment_count >= expected
            and "completed" in event_types
        )
    if run.status in {"blocked", "failed"}:
        if run.mode == "shadow":
            return (
                experiment_count >= len(_shadow_public_decision(run).ranked_actions)
                and run.status in event_types
            )
        return decision.status == "blocked" and run.status in event_types
    return True


def repair_terminal_decision_simulation_projection(
    run: DecisionSimulationRunRecord,
) -> bool:
    """Idempotently repair manager state/events after a terminal crash window."""

    if run.status not in _RUN_TERMINAL or run.status == "superseded":
        return False
    if run.superseded_by_run_id is not None:
        return False
    # A retry child is authoritative even if the process died before it could
    # link the manager row or mark its parent superseded. Never project the old
    # terminal parent back over the newer attempt.
    related_runs = list_decision_simulation_runs(
        decision_id=run.decision_id,
        limit=1000,
    )
    if any(item.retry_of_run_id == run.id for item in related_runs):
        return False
    if not isinstance(run.evidence_snapshot.get("managerDecision"), dict):
        return False
    if _terminal_projection_is_complete(run):
        return False
    if run.status == "completed":
        ranked = _durable_ranked_results(run)
        if run.mode == "enforced":
            _publish_enforced_decision(run, ranked)
        else:
            _publish_shadow_decision(run)
        verdict = None
        if isinstance(run.critic_result, dict):
            result = run.critic_result.get("result")
            if isinstance(result, dict):
                verdict = result.get("verdict")
        _event(
            run,
            "completed",
            suffix="completed",
            progress=1,
            payload={
                "winnerCandidateId": run.winner_candidate_id,
                "completedSamples": run.completed_samples,
                "totalSamples": run.total_samples,
                "criticVerdict": verdict,
                "projectionRepaired": True,
            },
        )
        return True
    if run.mode == "shadow":
        _publish_shadow_decision(run)
    else:
        error_code = str(run.error.get("code") or f"decision_simulation_{run.status}")
        message = str(
            run.error.get("message")
            or "Decision Simulation завершилась без разрешения на выполнение."
        )
        block_manager_decision_for_simulation(run, code=error_code, message=message)
    _event(
        run,
        run.status,
        suffix=f"{run.status}:projection-repair",
        progress=float(run.progress),
        payload={
            "code": run.error.get("code"),
            "projectionRepaired": True,
        },
    )
    return True


def _refreshed_snapshot(run: DecisionSimulationRunRecord) -> dict[str, Any]:
    """Reload source facts for the critic's single stale/missing-data retry."""

    if run.model_version == PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION:
        source_metadata = run.evidence_snapshot.get("pricingSource")
        if not isinstance(source_metadata, dict):
            raise ValueError("pricing simulation snapshot has no pricingSource")
        nm_id = source_metadata.get("nmId")
        if not isinstance(nm_id, int) or nm_id <= 0:
            raise ValueError("pricing simulation snapshot has no valid nmId")
        previous_request = _request_from_run(run)
        request, refreshed_source = load_pricing_simulation_source(
            nm_id=nm_id,
            candidate_prices=[
                candidate.price_rub
                for candidate in previous_request.candidates
                if candidate.operation == "set_price"
            ],
            horizon_days=previous_request.horizon_days,
        )
        original = _original_decision(run)
        current_snapshot = refreshed_source.get("currentPriceSnapshot") or {}
        raw_as_of = current_snapshot.get("last_synced_at")
        refreshed_decision_payload = original.model_dump(mode="json", by_alias=True)
        refreshed_decision_payload.update(
            {
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "evidenceAsOf": raw_as_of,
            }
        )
        refreshed_decision = ManagerDecisionV1.model_validate(
            refreshed_decision_payload
        )
        return {
            "schemaVersion": "pricing-decision-simulation-snapshot-v1",
            "sourceSchemaVersion": PRICING_SOURCE_SCHEMA_VERSION,
            "domain": "pricing",
            "operation": "set_price",
            "selectionMode": "winner_only",
            "objective": run.objective,
            "modelRuntime": simulation_runtime_card(),
            "managerDecision": refreshed_decision.model_dump(
                mode="json", by_alias=True
            ),
            "simulationRequest": pricing_request_snapshot_payload(request),
            "simulationCandidates": [
                candidate.model_dump(mode="json", by_alias=True)
                for candidate in request.candidates
            ],
            "pricingSource": refreshed_source,
            "hardGateRejectedCandidates": [],
        }

    source = load_manager_candidate_snapshot(limit=100)
    built = build_manager_action_candidates(source, max_candidates=7)
    actions: list[ManagerActionV1] = []
    for payload in built.candidates:
        if payload.get("candidateBlockers"):
            continue
        try:
            action = ManagerActionV1.model_validate(payload)
        except Exception:
            continue
        if action.family == "advertising" and action.operation == "pause_campaign":
            actions.append(action)
    actions = actions[:7]
    candidate_pairs = [
        (action, advertising_candidate_from_action(action)) for action in actions
    ]
    eligible_pairs = [
        (action, candidate)
        for action, candidate in candidate_pairs
        if not DecisionSimulationEngine().validate_candidate(candidate)
    ]
    actions = [action for action, _ in eligible_pairs]
    if len(actions) < 3:
        raise ValueError("fresh snapshot has fewer than three eligible advertising candidates")
    original = _original_decision(run)
    refreshed_decision = ManagerDecisionV1.model_validate(
        {
            **original.model_dump(mode="json", by_alias=True),
            "status": "ready_for_approval",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "evidenceAsOf": source.get("retrievedAt"),
            "rankedActions": [
                action.model_dump(mode="json", by_alias=True) for action in actions
            ],
            "blockers": [],
            "evidenceSnapshot": [
                fact.model_dump(mode="json", by_alias=True)
                for action in actions
                for fact in action.evidence
            ],
        }
    )
    candidates = [candidate for _, candidate in eligible_pairs]
    return build_evidence_snapshot(refreshed_decision, candidates)


@trace_simulation_job
def run_decision_simulation_job(simulation_run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    initial = get_decision_simulation_run(simulation_run_id)
    if initial is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    if initial.status in _RUN_TERMINAL:
        try:
            repair_terminal_decision_simulation_projection(initial)
        except Exception:
            logger.exception(
                "decision_simulation_terminal_projection_repair_failed "
                "simulation_run_id=%s",
                initial.id,
            )
        return {"simulation_run_id": initial.id, "status": initial.status, "duplicate": True}

    with _simulation_run_lock(simulation_run_id) as acquired:
        if not acquired:
            return {"simulation_run_id": simulation_run_id, "status": "already_running"}
        run = get_decision_simulation_run(simulation_run_id)
        if run is None:
            raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
        if run.status in _RUN_TERMINAL:
            try:
                repair_terminal_decision_simulation_projection(run)
            except Exception:
                logger.exception(
                    "decision_simulation_terminal_projection_repair_failed "
                    "simulation_run_id=%s",
                    run.id,
                )
            return {"simulation_run_id": run.id, "status": run.status, "duplicate": True}

        ACTIVE_JOBS.inc()
        outcome = "failed"
        try:
            run = update_decision_simulation_run(run.id, status="running", error={})
            _event(
                run,
                "started",
                suffix="started",
                progress=float(run.progress),
                payload={
                    "snapshotHash": run.evidence_hash,
                    "modelVersion": run.model_version,
                },
            )
            request = _request_from_run(run)
            try:
                from agi_ecom_app.experiment_program.simulation_adapter import (
                    maybe_attach_experiment_program,
                )

                experiment_program_id = maybe_attach_experiment_program(run)
                if experiment_program_id:
                    logger.info(
                        "decision_simulation_experiment_program_attached "
                        "simulation_run_id=%s experiment_program_id=%s",
                        run.id,
                        experiment_program_id,
                    )
            except Exception:
                # ExperimentProgram attachment is an observability/portfolio
                # projection. It must never corrupt or re-run the pinned NumPy
                # calculation if its own projection is temporarily unavailable.
                logger.exception(
                    "decision_simulation_experiment_program_attach_failed "
                    "simulation_run_id=%s",
                    run.id,
                )
            hard_gate_rejected = run.evidence_snapshot.get(
                "hardGateRejectedCandidates"
            )
            hard_gate_rejected = (
                hard_gate_rejected if isinstance(hard_gate_rejected, list) else []
            )
            for entry in hard_gate_rejected:
                if not isinstance(entry, dict):
                    continue
                violations = entry.get("violations")
                first = violations[0] if isinstance(violations, list) and violations else {}
                reason = (
                    str(first.get("code") or "hard_gate")
                    if isinstance(first, dict)
                    else "hard_gate"
                )
                REJECTED_CANDIDATES.labels(reason=reason).inc()
            _event(
                run,
                "snapshot_loaded",
                suffix="snapshot-loaded",
                progress=float(run.progress),
                payload={
                    "snapshotHash": run.evidence_hash,
                    "candidateCount": len(request.candidates),
                    "submittedCandidateCount": (
                        len(request.candidates) + len(hard_gate_rejected)
                    ),
                    "rejectedCandidateCount": len(hard_gate_rejected),
                },
            )
            engine = _engine_for_request(request)
            eligible_count = sum(
                not _candidate_violations(engine, request, candidate)
                for candidate in request.candidates
            )
            effective_total = eligible_count * request.sample_count
            existing_batches = {
                (item.candidate_id, item.batch_index): item
                for item in list_decision_simulation_batches(
                    run.id,
                    statuses=("completed",),
                )
            }
            stored_candidates = {
                item.candidate_id: item
                for item in list_decision_simulation_candidates(run.id)
            }
            completed_samples = sum(item.sample_count for item in existing_batches.values())
            update_decision_simulation_progress(
                run.id,
                completed_samples=completed_samples,
                total_samples=effective_total,
            )
            repaired_completed = 0
            candidate_order = {
                candidate.candidate_id: index
                for index, candidate in enumerate(request.candidates)
            }
            for stored_batch in sorted(
                existing_batches.values(),
                key=lambda batch: (
                    candidate_order.get(batch.candidate_id, len(candidate_order)),
                    batch.batch_index,
                ),
            ):
                repaired_completed += stored_batch.sample_count
                _event(
                    run,
                    "batch_completed",
                    suffix=(
                        f"candidate:{stored_batch.candidate_id}:"
                        f"batch:{stored_batch.batch_index}:completed"
                    ),
                    candidate_id=stored_batch.candidate_id,
                    batch_index=stored_batch.batch_index,
                    progress=(
                        repaired_completed / effective_total if effective_total else 1.0
                    ),
                    payload={
                        "seed": stored_batch.seed,
                        "sampleCount": stored_batch.sample_count,
                        "completedSamples": repaired_completed,
                        "totalSamples": effective_total,
                        "durationMs": stored_batch.duration_ms,
                        "resumedFromDurableBatch": True,
                    },
                )

            results: list[CandidateSimulationResultV1] = []
            for candidate in request.candidates:
                restored = _restore_completed_candidate(
                    candidate,
                    stored_candidates.get(candidate.candidate_id),
                    [
                        batch
                        for (candidate_id, _), batch in existing_batches.items()
                        if candidate_id == candidate.candidate_id
                    ],
                    expected_batch_count=request.sample_count // request.batch_size,
                )
                if restored is not None:
                    results.append(restored)
                    provisional = _ranked(results)[0].candidate_id
                    _event(
                        run,
                        "candidate_completed",
                        suffix=f"candidate:{candidate.candidate_id}:completed",
                        candidate_id=candidate.candidate_id,
                        progress=(
                            completed_samples / effective_total
                            if effective_total
                            else 1.0
                        ),
                        payload={
                            "completedSamples": restored.distribution.total.count,
                            "provisionalLeader": provisional,
                            "resumedFromDurableAggregate": True,
                        },
                    )
                    continue
                _event(
                    run,
                    "candidate_started",
                    suffix=f"candidate:{candidate.candidate_id}:started",
                    candidate_id=candidate.candidate_id,
                    progress=(completed_samples / effective_total if effective_total else 0),
                    payload={"actionId": candidate.action_id},
                )
                upsert_decision_simulation_candidate(
                    simulation_run_id=run.id,
                    candidate_id=candidate.candidate_id,
                    action_id=candidate.action_id,
                    status="running",
                    candidate_payload=candidate.model_dump(mode="json", by_alias=True),
                )
                violations = _candidate_violations(engine, request, candidate)
                if violations:
                    result = engine.simulate_candidate(request, candidate)
                    results.append(result)
                    _persist_candidate_result(run, request, result)
                    reason = result.constraint_violations[0].code
                    REJECTED_CANDIDATES.labels(reason=reason).inc()
                    _event(
                        run,
                        "candidate_rejected",
                        suffix=f"candidate:{candidate.candidate_id}:rejected",
                        candidate_id=candidate.candidate_id,
                        progress=(completed_samples / effective_total if effective_total else 0),
                        payload={
                            "status": result.status,
                            "constraintViolations": [
                                item.model_dump(mode="json", by_alias=True)
                                for item in result.constraint_violations
                            ],
                        },
                    )
                    continue

                incremental = callable(getattr(engine, "simulate_batch", None)) and callable(
                    getattr(engine, "aggregate_candidate", None)
                )
                if incremental:
                    batch_results: list[Any] = []
                    for batch_index in range(
                        request.sample_count // request.batch_size
                    ):
                        key = (candidate.candidate_id, batch_index)
                        stored = existing_batches.get(key)
                        if stored is not None:
                            batch_results.append(
                                _batch_from_record(request, candidate, stored)
                            )
                            continue
                        batch_started = time.perf_counter()
                        batch = engine.simulate_batch(
                            request,
                            candidate,
                            batch_index,
                        )
                        batch_duration = time.perf_counter() - batch_started
                        aggregates = _batch_aggregates(batch)
                        stored_batch = upsert_decision_simulation_batch(
                            simulation_run_id=run.id,
                            candidate_id=candidate.candidate_id,
                            batch_index=batch.batch_index,
                            seed=batch.seed,
                            sample_count=batch.sample_count,
                            status="completed",
                            aggregates=aggregates,
                            duration_ms=batch_duration * 1000,
                        )
                        existing_batches[key] = stored_batch
                        batch_results.append(batch)
                        completed_samples += batch.sample_count
                        progress = (
                            completed_samples / effective_total
                            if effective_total
                            else 1.0
                        )
                        update_decision_simulation_progress(
                            run.id,
                            completed_samples=completed_samples,
                            total_samples=effective_total,
                        )
                        SAMPLES_COMPLETED.labels(
                            model_version=run.model_version
                        ).inc(batch.sample_count)
                        BATCH_DURATION.labels(model_version=run.model_version).observe(
                            batch_duration
                        )
                        SAMPLES_PER_SECOND.labels(
                            model_version=run.model_version
                        ).set(batch.sample_count / max(batch_duration, 1e-9))
                        _event(
                            run,
                            "batch_completed",
                            suffix=(
                                f"candidate:{candidate.candidate_id}:"
                                f"batch:{batch.batch_index}:completed"
                            ),
                            candidate_id=candidate.candidate_id,
                            batch_index=batch.batch_index,
                            progress=progress,
                            payload={
                                "seed": batch.seed,
                                "sampleCount": batch.sample_count,
                                "completedSamples": completed_samples,
                                "totalSamples": effective_total,
                                "durationMs": batch_duration * 1000,
                            },
                        )
                        logger.info(
                            "decision_simulation_batch_completed decision_id=%s "
                            "simulation_run_id=%s candidate_id=%s batch_index=%s "
                            "snapshot_hash=%s model_version=%s samples=%s",
                            run.decision_id,
                            run.id,
                            candidate.candidate_id,
                            batch.batch_index,
                            run.evidence_hash,
                            run.model_version,
                            batch.sample_count,
                        )
                    result = engine.aggregate_candidate(
                        request,
                        candidate,
                        batch_results,
                    )
                    results.append(result)
                else:
                    # Compatibility for injected test/custom engines. The
                    # production engine always exposes durable per-batch APIs.
                    candidate_started = time.perf_counter()
                    result = engine.simulate_candidate(request, candidate)
                    results.append(result)
                    if result.status != "completed":
                        _persist_candidate_result(run, request, result)
                        continue
                    per_batch_duration = (
                        (time.perf_counter() - candidate_started)
                        / max(1, len(result.batch_results))
                    )
                    for batch in result.batch_results:
                        key = (candidate.candidate_id, batch.batch_index)
                        if key in existing_batches:
                            continue
                        aggregates = _batch_aggregates(batch)
                        stored_batch = upsert_decision_simulation_batch(
                            simulation_run_id=run.id,
                            candidate_id=candidate.candidate_id,
                            batch_index=batch.batch_index,
                            seed=batch.seed,
                            sample_count=batch.sample_count,
                            status="completed",
                            aggregates=aggregates,
                            duration_ms=per_batch_duration * 1000,
                        )
                        existing_batches[key] = stored_batch
                        completed_samples += batch.sample_count
                        update_decision_simulation_progress(
                            run.id,
                            completed_samples=completed_samples,
                            total_samples=effective_total,
                        )
                _persist_candidate_result(run, request, result)
                provisional = _ranked(results)[0].candidate_id
                completed_count = sum(item.status == "completed" for item in results)
                update_decision_simulation_progress(
                    run.id,
                    completed_samples=completed_samples,
                    total_samples=effective_total,
                    completed_candidates=completed_count,
                    provisional_winner_candidate_id=provisional,
                )
                _event(
                    run,
                    "candidate_completed",
                    suffix=f"candidate:{candidate.candidate_id}:completed",
                    candidate_id=candidate.candidate_id,
                    progress=(completed_samples / effective_total if effective_total else 1.0),
                    payload={
                        "completedSamples": result.distribution.total.count,
                        "provisionalLeader": provisional,
                    },
                )

            ranked = _ranked(results)
            rejected = [item for item in results if item.status != "completed"]
            submitted_count = len(request.candidates) + len(hard_gate_rejected)
            FORECAST_COVERAGE.set(len(ranked) / submitted_count)
            for item in ranked:
                _persist_candidate_result(run, request, item)
            if len(ranked) < 3:
                outcome = "blocked"
                error = {
                    "code": "insufficient_completed_candidates",
                    "message": "Fewer than three candidates produced a forecast.",
                }
                clear_decision_simulation_winner_flags(run.id)
                update_decision_simulation_run(
                    run.id,
                    status="blocked",
                    completed_samples=completed_samples,
                    total_samples=effective_total,
                    completed_candidates=len(ranked),
                    error=error,
                )
                _event(run, "blocked", suffix="blocked:coverage", payload=error)
                if run.mode == "enforced":
                    block_manager_decision_for_simulation(
                        run,
                        code="decision_simulation_insufficient_candidates",
                        message="Меньше трёх кандидатов получили численный forecast.",
                    )
                else:
                    _publish_shadow_decision(run)
                return {"simulation_run_id": run.id, "status": "blocked"}

            winner = ranked[0].candidate_id
            update_decision_simulation_progress(
                run.id,
                completed_samples=completed_samples,
                total_samples=effective_total,
                completed_candidates=len(ranked),
                provisional_winner_candidate_id=winner,
            )
            _event(
                run,
                "ranking_completed",
                suffix="ranking-completed",
                progress=1,
                payload={
                    "provisionalLeader": winner,
                    "ranking": [item.candidate_id for item in ranked],
                    "score": [
                        item.score_breakdown.model_dump(mode="json", by_alias=True)
                        for item in ranked
                    ],
                },
            )

            critic_input = _critic_input(run, ranked, rejected)
            update_decision_simulation_run(run.id, status="critic_running")
            _event(
                run,
                "critic_started",
                suffix="critic-started",
                progress=1,
                payload={"rankingFingerprint": critic_input.ranking_fingerprint},
            )
            settings = get_settings()
            critic, reused_critic = _execute_or_reuse_decision_critic(
                run,
                critic_input,
                settings=settings,
            )
            critic_payload = critic.model_dump(mode="json", by_alias=True)
            if not reused_critic:
                CRITIC_VERDICTS.labels(
                    mode=run.mode,
                    verdict=critic.result.verdict,
                    llm_available=str(critic.llm_available).lower(),
                ).inc()
            update_decision_simulation_run(run.id, critic_result=critic_payload)
            _event(
                run,
                "critic_completed",
                suffix="critic-completed",
                progress=1,
                payload=critic_payload,
            )

            should_recalculate = (
                critic.llm_available and critic.result.verdict == "recalculate"
            )
            should_block = critic.result.verdict == "blocked" and (
                run.mode == "enforced" or critic.llm_available
            )

            if should_recalculate:
                try:
                    replacement = create_critic_recalculation(
                        run,
                        evidence_snapshot=_refreshed_snapshot(run),
                    )
                except Exception as exc:
                    outcome = "blocked"
                    error = {
                        "code": "critic_recalculation_failed",
                        "message": str(exc),
                    }
                    clear_decision_simulation_winner_flags(run.id)
                    update_decision_simulation_run(
                        run.id,
                        status="blocked",
                        critic_result=critic_payload,
                        error=error,
                    )
                    _event(run, "blocked", suffix="blocked:recalculation", payload=error)
                    if run.mode == "enforced":
                        block_manager_decision_for_simulation(
                            run,
                            code="critic_recalculation_failed",
                            message="Не удалось получить новый evidence snapshot для пересчёта.",
                        )
                    else:
                        _publish_shadow_decision(run)
                    return {"simulation_run_id": run.id, "status": "blocked"}
                outcome = "superseded"
                return {
                    "simulation_run_id": run.id,
                    "status": "superseded",
                    "replacement_run_id": replacement.id,
                }

            if should_block:
                outcome = "blocked"
                clear_decision_simulation_winner_flags(run.id)
                update_decision_simulation_run(
                    run.id,
                    status="blocked",
                    critic_result=critic_payload,
                    winner_candidate_id=None,
                    completed_samples=completed_samples,
                    total_samples=effective_total,
                    completed_candidates=len(ranked),
                )
                _event(
                    run,
                    "blocked",
                    suffix="blocked:critic",
                    progress=1,
                    payload={
                        "code": critic.result.issues[0].code,
                        "criticVerdict": critic.result.verdict,
                    },
                )
                if run.mode == "enforced":
                    block_manager_decision_for_simulation(
                        run,
                        code=str(critic.result.issues[0].code),
                        message=critic.result.summary,
                    )
                else:
                    _publish_shadow_decision(run)
                return {"simulation_run_id": run.id, "status": "blocked"}

            complete_decision_simulation_run(
                run.id,
                winner_candidate_id=winner,
                completed_samples=completed_samples,
                total_samples=effective_total,
                completed_candidates=len(ranked),
                critic_result=critic_payload,
            )
            # Experiments are materialized only after the durable run is
            # completed. Approval still validates the linked run, closing the
            # publication race if this idempotent projection is retried.
            if run.mode == "enforced":
                _publish_enforced_decision(run, ranked)
            else:
                _publish_shadow_decision(run)
            _event(
                run,
                "completed",
                suffix="completed",
                progress=1,
                payload={
                    "winnerCandidateId": winner,
                    "completedSamples": completed_samples,
                    "totalSamples": effective_total,
                    "criticVerdict": critic.result.verdict,
                },
            )
            outcome = "completed"
            return {
                "simulation_run_id": run.id,
                "status": "completed",
                "winner_candidate_id": winner,
            }
        except Exception as exc:
            FAILURES.labels(stage="job").inc()
            logger.exception(
                "decision_simulation_failed decision_id=%s simulation_run_id=%s "
                "snapshot_hash=%s model_version=%s",
                run.decision_id,
                run.id,
                run.evidence_hash,
                run.model_version,
            )
            error = {"code": "simulation_failed", "type": type(exc).__name__, "message": str(exc)}
            clear_decision_simulation_winner_flags(run.id)
            update_decision_simulation_run(
                run.id,
                status="failed",
                winner_candidate_id=None,
                error=error,
            )
            _event(run, "failed", suffix="failed:job", payload=error)
            if run.mode == "enforced":
                try:
                    block_manager_decision_for_simulation(
                        run,
                        code="decision_simulation_failed",
                        message="Decision Simulation завершилась технической ошибкой.",
                    )
                except Exception:
                    logger.exception(
                        "decision_simulation_failed_to_block_decision decision_id=%s simulation_run_id=%s",
                        run.decision_id,
                        run.id,
                    )
            else:
                try:
                    _publish_shadow_decision(run)
                except Exception:
                    logger.exception(
                        "decision_simulation_failed_to_restore_shadow_decision "
                        "decision_id=%s simulation_run_id=%s",
                        run.decision_id,
                        run.id,
                    )
            return {"simulation_run_id": run.id, "status": "failed", "error": error}
        finally:
            try:
                from agi_ecom_app.experiment_program.simulation_adapter import (
                    sync_experiment_program_from_simulation_run,
                )

                sync_experiment_program_from_simulation_run(run.id)
            except Exception:
                logger.exception(
                    "decision_simulation_experiment_program_sync_failed "
                    "simulation_run_id=%s",
                    run.id,
                )
            ACTIVE_JOBS.dec()
            JOB_DURATION.labels(mode=run.mode, status=outcome).observe(
                time.perf_counter() - started
            )


def reconcile_decision_simulations(
    *,
    now: datetime | None = None,
    executor: RQSimulationExecutor | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(seconds=settings.decision_simulation_stale_after_seconds)
    stale = list_stale_decision_simulation_runs(stale_before=cutoff, limit=100)
    STALE_JOBS.set(len(stale))
    recovered: list[str] = []
    initialization_failed: list[str] = []
    delivery = executor or RQSimulationExecutor()
    slot = int(current.timestamp()) // max(30, settings.decision_simulation_stale_after_seconds // 2)
    for run in stale:
        if run.status == "initializing":
            error = {
                "code": "retry_initialization_timeout",
                "message": "Retry initialization heartbeat expired before queueing.",
            }
            failed_run = claim_stale_decision_simulation_run(
                run.id,
                expected_status="initializing",
                stale_before=cutoff,
                claimed_status="failed",
                heartbeat_at=current,
                error=error,
            )
            if failed_run is None:
                continue
            clear_decision_simulation_winner_flags(run.id)
            _event(
                failed_run,
                "failed",
                suffix=f"failed:initialization-timeout:{slot}",
                progress=float(run.progress),
                payload=error,
            )
            try:
                repair_terminal_decision_simulation_projection(failed_run)
            except Exception:
                logger.exception(
                    "decision_simulation_initialization_timeout_projection_failed "
                    "simulation_run_id=%s",
                    run.id,
                )
            initialization_failed.append(run.id)
            continue
        delivery_status = (
            delivery.delivery_status(run.id)
            if hasattr(delivery, "delivery_status")
            else None
        )
        if run.status == "queued" and delivery_status in {
            "queued",
            "deferred",
            "scheduled",
            "started",
        }:
            # An old queue timestamp does not mean delivery is lost. Moving a
            # valid FIFO job to the tail repeatedly can starve it forever.
            continue
        claimed = claim_stale_decision_simulation_run(
            run.id,
            expected_status=run.status,
            stale_before=cutoff,
            claimed_status="queued",
            heartbeat_at=current,
        )
        if claimed is None:
            continue
        _event(
            claimed,
            "queued",
            suffix=f"recovered:{slot}",
            progress=float(run.progress),
            payload={"recovered": True, "staleBefore": cutoff.isoformat()},
        )
        delivery.enqueue(run.id, replace_stale=True)
        recovered.append(run.id)
    projection_repaired: list[str] = []
    for run in list_unprojected_terminal_decision_simulation_runs(limit=100):
        try:
            if repair_terminal_decision_simulation_projection(run):
                projection_repaired.append(run.id)
        except Exception:
            logger.exception(
                "decision_simulation_terminal_projection_reconcile_failed "
                "simulation_run_id=%s",
                run.id,
            )
    result: dict[str, Any] = {
        "stale": len(stale),
        "recovered": recovered,
    }
    if projection_repaired:
        result["projection_repaired"] = projection_repaired
    if initialization_failed:
        result["initialization_failed"] = initialization_failed
    superseded_events_backfilled = backfill_decision_simulation_superseded_events(
        limit=100
    )
    if superseded_events_backfilled:
        result["superseded_events_backfilled"] = superseded_events_backfilled
    return result


def schedule_decision_simulation_reconciliation(
    *,
    queue: Any | None = None,
    now: datetime | None = None,
) -> str:
    settings = get_settings()
    resolved_queue = queue
    if resolved_queue is None:
        from agi_ecom_app.decision_simulation.service import _build_rq_queue

        resolved_queue = _build_rq_queue(
            settings.decision_simulation_maintenance_queue_name
        )
    interval = max(30, settings.decision_simulation_stale_after_seconds // 2)
    current = now or datetime.now(timezone.utc)
    run_at = current + timedelta(seconds=interval)
    slot = int(run_at.timestamp()) // interval
    job_id = f"decision-simulation-reconcile-{slot}"
    try:
        resolved_queue.enqueue_at(
            run_at,
            run_decision_simulation_reconciler_job,
            job_id=job_id,
            result_ttl=interval * 2,
            failure_ttl=86_400,
        )
    except Exception as exc:
        if "already exists" not in str(exc).casefold():
            raise
    return job_id


def run_decision_simulation_reconciler_job() -> dict[str, Any]:
    try:
        return reconcile_decision_simulations()
    finally:
        schedule_decision_simulation_reconciliation()


__all__ = [
    "reconcile_decision_simulations",
    "repair_terminal_decision_simulation_projection",
    "run_decision_simulation_job",
    "run_decision_simulation_reconciler_job",
    "schedule_decision_simulation_reconciliation",
]
