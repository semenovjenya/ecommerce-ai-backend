from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Sequence
import uuid

from agi_ecom_app.config import normalize_decision_simulation_mode
from agi_ecom_app.decision_simulation.engine import simulation_runtime_card
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    derive_pricing_master_seed,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    PRICING_SOURCE_SCHEMA_VERSION,
    PricingSimulationSourceError,
    load_pricing_simulation_source,
    pricing_request_snapshot_payload,
)
from agi_ecom_app.decision_simulation.service import (
    RQSimulationExecutor,
    SimulationExecutor,
    canonical_snapshot_hash,
)
from agi_ecom_app.manager_decision.models import (
    DecisionBlockerV1,
    ManagerDecisionV1,
    RequestCoverageV1,
)
from agi_ecom_app.manager_lifecycle.auth import ManagerActor, require_role
from agi_ecom_app.manager_lifecycle.service import persist_manager_decision_bundle
from agi_ecom_app.services.memory import (
    DecisionSimulationRunRecord,
    append_decision_simulation_event,
    claim_decision_simulation_initialization,
    create_decision_simulation_run,
    get_decision_simulation_run,
    get_manager_decision_by_source_key,
    set_manager_decision_simulation_run,
    update_decision_simulation_run,
    upsert_decision_simulation_candidate,
)


PRICING_SIMULATION_SNAPSHOT_SCHEMA_VERSION = "pricing-decision-simulation-snapshot-v1"
logger = logging.getLogger(__name__)


class PricingSimulationIdempotencyConflict(RuntimeError):
    pass


class PricingSimulationDeliveryError(RuntimeError):
    def __init__(self, simulation_run_id: str, message: str) -> None:
        super().__init__(message)
        self.simulation_run_id = simulation_run_id


def pricing_simulation_mode() -> str:
    return normalize_decision_simulation_mode(
        os.getenv("DECISION_SIMULATION_PRICING_MODE", "off")
    )


def start_pricing_simulation(
    *,
    nm_id: int,
    actor: ManagerActor,
    idempotency_key: str,
    candidate_prices: Sequence[float] | None = None,
    horizon_days: int = 7,
    executor: SimulationExecutor | None = None,
) -> DecisionSimulationRunRecord:
    """Create one observable, winner-only pricing simulation in shadow mode."""

    require_role(actor, "manager")
    mode = pricing_simulation_mode()
    if mode == "off":
        raise RuntimeError("pricing decision simulation is disabled")
    if mode != "shadow":
        raise RuntimeError(
            "pricing simulation v1 is shadow-only until complete source watermarks exist"
        )
    key = str(idempotency_key or "").strip()
    if not key:
        raise ValueError("Idempotency-Key is required")
    if len(key) > 200:
        raise ValueError("Idempotency-Key is too long")

    source_key = f"pricing-simulation:{int(nm_id)}:{key}"
    start_contract = _start_contract(
        nm_id=int(nm_id),
        candidate_prices=candidate_prices,
        horizon_days=horizon_days,
    )
    start_contract_hash = canonical_snapshot_hash(start_contract)
    existing_decision = get_manager_decision_by_source_key(source_key)
    if existing_decision is not None and existing_decision.simulation_run_id:
        existing_run = get_decision_simulation_run(
            existing_decision.simulation_run_id
        )
        if existing_run is not None:
            _assert_start_contract(existing_run, start_contract_hash)
            return existing_run

    request, source_metadata = load_pricing_simulation_source(
        nm_id=int(nm_id),
        candidate_prices=candidate_prices,
        horizon_days=horizon_days,
    )
    now = datetime.now(timezone.utc)
    decision_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"manager:{source_key}")
    )
    decision = ManagerDecisionV1(
        decisionId=decision_id,
        queryMode="decision",
        status="evaluating",
        createdAt=now,
        evidenceAsOf=_source_evidence_as_of(source_metadata),
        requestCoverage=RequestCoverageV1(
            explicitTargets=[str(nm_id)],
            coveredTargets=[str(nm_id)],
            requestedOutputs=["price_simulation"],
            coveredOutputs=["price_simulation"],
            complete=True,
        ),
        rankedActions=[],
        rejectedCandidates=[],
        blockers=[],
        evidenceSnapshot=[],
    )
    persisted, _ = persist_manager_decision_bundle(
        decision,
        source_key=source_key,
        create_experiments=False,
    )
    if persisted.simulation_run_id:
        existing_run = get_decision_simulation_run(persisted.simulation_run_id)
        if existing_run is not None:
            _assert_start_contract(existing_run, start_contract_hash)
            return existing_run

    snapshot = {
        "schemaVersion": PRICING_SIMULATION_SNAPSHOT_SCHEMA_VERSION,
        "sourceSchemaVersion": PRICING_SOURCE_SCHEMA_VERSION,
        "domain": "pricing",
        "operation": "set_price",
        "selectionMode": "winner_only",
        "objective": decision.objective,
        "modelRuntime": simulation_runtime_card(),
        "managerDecision": decision.model_dump(mode="json", by_alias=True),
        "simulationRequest": pricing_request_snapshot_payload(request),
        "simulationCandidates": [
            candidate.model_dump(mode="json", by_alias=True)
            for candidate in request.candidates
        ],
        "pricingSource": source_metadata,
        "hardGateRejectedCandidates": [],
        "startRequest": start_contract,
        "startRequestHash": start_contract_hash,
    }
    snapshot_hash = canonical_snapshot_hash(snapshot)
    run_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            ":".join(
                (
                    "pricing-decision-simulation",
                    persisted.id,
                    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
                    snapshot_hash,
                    mode,
                )
            ),
        )
    )
    run = create_decision_simulation_run(
        simulation_run_id=run_id,
        decision_id=persisted.id,
        mode=mode,
        model_version=PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
        evidence_snapshot=snapshot,
        evidence_hash=snapshot_hash,
        master_seed=derive_pricing_master_seed(
            snapshot_hash,
            request.decision_group_id,
        ),
        candidate_count=len(request.candidates),
        sample_count=request.sample_count,
        batch_size=request.batch_size,
        objective=decision.objective,
        status="initializing",
        idempotency_key=key,
    )
    _assert_start_contract(run, start_contract_hash)
    if run.status != "initializing":
        return run

    claimed = claim_decision_simulation_initialization(run.id)
    if claimed is None:
        # A concurrent request with the same deterministic source/idempotency
        # key owns candidate creation and queue delivery. Never initialize the
        # same run twice.
        return get_decision_simulation_run(run.id) or run
    run = claimed
    request = _pricing_request_from_durable_run(run)

    try:
        for candidate in request.candidates:
            upsert_decision_simulation_candidate(
                simulation_run_id=run.id,
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                status="pending",
                candidate_payload=candidate.model_dump(mode="json", by_alias=True),
                evidence_refs=_pricing_evidence_refs(request),
            )
        set_manager_decision_simulation_run(
            persisted.id,
            simulation_run_id=run.id,
            status="evaluating",
        )
        run = update_decision_simulation_run(
            run.id,
            status="queued",
            completed_samples=0,
            total_samples=len(request.candidates) * request.sample_count,
            completed_candidates=0,
            progress=0,
            error={},
        )
        append_decision_simulation_event(
            simulation_run_id=run.id,
            event_type="queued",
            event_id=f"{run.id}:queued",
            progress=0,
            payload={
                "domain": "pricing",
                "selectionMode": "winner_only",
                "modelVersion": run.model_version,
                "candidateCount": len(request.candidates),
                "sampleCount": request.sample_count,
                "completedSamples": 0,
                "totalSamples": len(request.candidates) * request.sample_count,
                "snapshotHash": run.evidence_hash,
            },
        )
        (executor or RQSimulationExecutor()).enqueue(run.id)
        return run
    except Exception as exc:
        failed_run = update_decision_simulation_run(
            run.id,
            status="failed",
            error={
                "code": "pricing_simulation_initialization_failed",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        append_decision_simulation_event(
            simulation_run_id=run.id,
            event_type="failed",
            event_id=f"{run.id}:failed:initialization",
            payload={
                "code": "pricing_simulation_initialization_failed",
                "message": str(exc),
            },
        )
        try:
            _project_pricing_needs_data(
                failed_run,
                code="pricing_simulation_delivery_failed",
                message=(
                    "Ценовая симуляция не была надёжно доставлена worker; "
                    "решение остаётся только наблюдаемым и требует повторного запуска."
                ),
            )
        except Exception:
            logger.exception(
                "pricing_simulation_failure_projection_failed simulation_run_id=%s",
                run.id,
            )
        raise PricingSimulationDeliveryError(run.id, str(exc)) from exc


def _start_contract(
    *,
    nm_id: int,
    candidate_prices: Sequence[float] | None,
    horizon_days: int,
) -> dict[str, Any]:
    return {
        "nmId": int(nm_id),
        "candidatePrices": (
            [float(value) for value in candidate_prices]
            if candidate_prices is not None
            else None
        ),
        "horizonDays": int(horizon_days),
    }


def _assert_start_contract(
    run: DecisionSimulationRunRecord,
    expected_hash: str,
) -> None:
    stored_hash = run.evidence_snapshot.get("startRequestHash")
    if stored_hash is not None and stored_hash != expected_hash:
        raise PricingSimulationIdempotencyConflict(
            "Idempotency-Key was already used with a different pricing request"
        )


def _pricing_request_from_durable_run(
    run: DecisionSimulationRunRecord,
) -> Any:
    if canonical_snapshot_hash(run.evidence_snapshot) != run.evidence_hash:
        raise ValueError("durable pricing snapshot hash does not match")
    raw_request = run.evidence_snapshot.get("simulationRequest")
    raw_candidates = run.evidence_snapshot.get("simulationCandidates")
    if not isinstance(raw_request, dict) or not isinstance(raw_candidates, list):
        raise ValueError("durable pricing snapshot is incomplete")
    from agi_ecom_app.decision_simulation.pricing_models import (
        PricingSimulationRequestV1,
    )

    return PricingSimulationRequestV1.model_validate(
        {
            **raw_request,
            "snapshotHash": run.evidence_hash,
            "modelVersion": run.model_version,
            "sampleCount": run.sample_count,
            "batchSize": run.batch_size,
            "candidates": raw_candidates,
        }
    )


def _project_pricing_needs_data(
    run: DecisionSimulationRunRecord,
    *,
    code: str,
    message: str,
) -> None:
    payload = run.evidence_snapshot.get("managerDecision")
    if not isinstance(payload, dict):
        raise ValueError("pricing snapshot has no manager decision")
    original = ManagerDecisionV1.model_validate(payload)
    final = ManagerDecisionV1.model_validate(
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


def _pricing_evidence_refs(request: Any) -> list[str]:
    evidence = request.evidence
    raw = [
        request.current_price_evidence_ref,
        evidence.daily_observations.evidence_ref,
        evidence.elasticity.evidence_ref,
        evidence.realization.evidence_ref,
        evidence.unit_economics.evidence_ref,
        evidence.available_stock.evidence_ref,
    ]
    raw.extend(
        item.evidence_ref for item in evidence.daily_observations.values or []
    )
    return list(dict.fromkeys(str(item) for item in raw if item))


def _source_evidence_as_of(metadata: dict[str, Any]) -> datetime | None:
    raw = (metadata.get("currentPriceSnapshot") or {}).get("last_synced_at")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


__all__ = [
    "PRICING_SIMULATION_SNAPSHOT_SCHEMA_VERSION",
    "PricingSimulationDeliveryError",
    "PricingSimulationIdempotencyConflict",
    "PricingSimulationSourceError",
    "pricing_simulation_mode",
    "start_pricing_simulation",
]
