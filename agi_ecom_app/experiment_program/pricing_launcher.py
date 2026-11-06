from __future__ import annotations

import hashlib
import logging
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any

from redis import Redis
from rq import Queue, Retry

from agi_ecom_app.config import get_settings
from agi_ecom_app.decision_simulation.engine import simulation_runtime_card
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    derive_pricing_master_seed,
)
from agi_ecom_app.decision_simulation.pricing_models import PricingSimulationRequestV1
from agi_ecom_app.decision_simulation.pricing_service import (
    PRICING_SIMULATION_SNAPSHOT_SCHEMA_VERSION,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    pricing_request_snapshot_payload,
)
from agi_ecom_app.decision_simulation.service import (
    RQSimulationExecutor,
    canonical_snapshot_hash,
)
from agi_ecom_app.experiment_program.executor_contract import (
    ProgramSimulationLaunch,
    ProgramSimulationUnsupported,
)
from agi_ecom_app.experiment_program.repository import (
    ExperimentProgramStorageConflict,
    bind_program_simulation_run,
    fail_program_simulation_dispatch,
    get_program_bundle,
    sync_linked_simulation_run,
)
from agi_ecom_app.manager_decision.models import (
    ManagerDecisionV1,
    RequestCoverageV1,
)
from agi_ecom_app.manager_lifecycle.service import persist_manager_decision_bundle
from agi_ecom_app.services.memory import (
    append_decision_simulation_event,
    create_decision_simulation_run,
    get_decision_simulation_run,
    list_decision_simulation_candidates,
    set_manager_decision_simulation_run,
    update_decision_simulation_run,
    upsert_decision_simulation_candidate,
)


logger = logging.getLogger(__name__)
_PRICING_INPUT_METRIC = "pricing_simulation_request"
_TERMINAL_RUN_STATUSES = frozenset({"completed", "blocked", "failed", "superseded"})


def enqueue_supported_program_simulation(
    *,
    program_id: str,
    source_event_id: str,
    expected_epoch: int,
    expected_snapshot_hash: str,
    queue: Queue | None = None,
) -> ProgramSimulationLaunch:
    """Validate a finite execution group and enqueue its deterministic launcher."""

    bundle = get_program_bundle(program_id)
    if bundle is None:
        raise LookupError(program_id)
    _validate_pricing_program_surface(
        bundle,
        expected_epoch=expected_epoch,
        expected_snapshot_hash=expected_snapshot_hash,
    )
    target_queue = queue or _build_simulation_queue()
    job_id = _pricing_launch_job_id(
        program_id=program_id,
        epoch=expected_epoch,
        snapshot_hash=expected_snapshot_hash,
    )
    existing = target_queue.fetch_job(job_id)
    should_enqueue = existing is None
    if existing is not None:
        status = _job_status(existing)
        if status == "failed" and hasattr(existing, "requeue"):
            existing.requeue()
            return ProgramSimulationLaunch(
                simulation_run_ids=(),
                executor_name="pricing-program-v1",
                details={"launchJobId": job_id, "queue": target_queue.name},
            )
        if status in {"queued", "started", "running", "scheduled", "deferred"}:
            return ProgramSimulationLaunch(
                simulation_run_ids=(),
                executor_name="pricing-program-v1",
                details={"launchJobId": job_id, "queue": target_queue.name},
            )
        if hasattr(existing, "delete"):
            # A finished/cancelled launcher without a bound run represents a
            # crash window or stale no-op. Recreate the deterministic command.
            existing.delete()
            should_enqueue = True
        else:
            raise RuntimeError(
                f"cannot recover experiment program launch job in status {status}"
            )
    if should_enqueue:
        target_queue.enqueue_call(
            launch_pricing_program_simulation_job,
            kwargs={
                "program_id": program_id,
                "source_event_id": source_event_id,
                "expected_epoch": expected_epoch,
                "expected_snapshot_hash": expected_snapshot_hash,
            },
            job_id=job_id,
            retry=Retry(max=3, interval=[5, 30, 120]),
            result_ttl=86_400,
            failure_ttl=604_800,
        )
    return ProgramSimulationLaunch(
        simulation_run_ids=(),
        executor_name="pricing-program-v1",
        details={"launchJobId": job_id, "queue": target_queue.name},
    )


def launch_pricing_program_simulation_job(
    *,
    program_id: str,
    source_event_id: str,
    expected_epoch: int,
    expected_snapshot_hash: str,
    executor: Any | None = None,
) -> dict[str, Any]:
    """Build and bind a pricing run; numerical batches remain a second RQ job."""

    bundle = get_program_bundle(program_id)
    if bundle is None:
        raise LookupError(program_id)
    program = bundle.program
    if (
        int(program["current_epoch"]) != int(expected_epoch)
        or program["snapshot_hash"] != expected_snapshot_hash
        or program["status"] != "simulating"
    ):
        return {
            "status": "stale_command",
            "programId": program_id,
            "programStatus": program["status"],
            "epoch": program["current_epoch"],
            "snapshotHash": program["snapshot_hash"],
        }

    try:
        request, candidate_links, packet = _pricing_request_and_links(bundle)
    except ProgramSimulationUnsupported as exc:
        fail_program_simulation_dispatch(
            program_id=program_id,
            source_event_id=source_event_id,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            expected_epoch=expected_epoch,
            expected_snapshot_hash=expected_snapshot_hash,
        )
        return {
            "status": "failed_closed",
            "programId": program_id,
            "code": exc.code,
        }

    decision_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"experiment-program-decision:{program_id}:{expected_epoch}:pricing",
        )
    )
    decision_created_at = (
        program.get("created_at")
        or bundle.snapshot.get("created_at")
        or bundle.snapshot.get("data_as_of")
    )
    if decision_created_at is None:
        raise RuntimeError("durable experiment program has no creation timestamp")
    decision = ManagerDecisionV1(
        decisionId=decision_id,
        queryMode="decision",
        status="evaluating",
        createdAt=decision_created_at,
        evidenceAsOf=bundle.snapshot.get("data_as_of"),
        requestCoverage=RequestCoverageV1(
            explicitTargets=[request.sku_id],
            coveredTargets=[request.sku_id],
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
        chat_id=program.get("source_chat_id"),
        run_id=program.get("source_run_id"),
        source_key=(
            f"experiment-program:{program_id}:{expected_epoch}:pricing"
        ),
        create_experiments=False,
    )
    snapshot = {
        "schemaVersion": PRICING_SIMULATION_SNAPSHOT_SCHEMA_VERSION,
        "sourceSchemaVersion": "experiment-program-pricing-input-v1",
        "domain": "pricing",
        "operation": "set_price",
        "selectionMode": "winner_only",
        "objective": program["objective"],
        "modelRuntime": simulation_runtime_card(),
        "managerDecision": dict(persisted.decision_json or {}),
        "simulationRequest": pricing_request_snapshot_payload(request),
        "simulationCandidates": [
            candidate.model_dump(mode="json", by_alias=True)
            for candidate in request.candidates
        ],
        "hardGateRejectedCandidates": [],
        "experimentProgram": {
            "programId": program_id,
            "epoch": expected_epoch,
            "snapshotHash": expected_snapshot_hash,
            "evidenceRef": packet.get("evidenceRef")
            or packet.get("evidence_ref"),
        },
    }
    run_snapshot_hash = canonical_snapshot_hash(snapshot)
    simulation_run_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            ":".join(
                (
                    "experiment-program-pricing-simulation",
                    program_id,
                    str(expected_epoch),
                    run_snapshot_hash,
                )
            ),
        )
    )
    run = create_decision_simulation_run(
        simulation_run_id=simulation_run_id,
        decision_id=persisted.id,
        chat_id=program.get("source_chat_id"),
        run_id=program.get("source_run_id"),
        mode="shadow",
        model_version=PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
        evidence_snapshot=snapshot,
        evidence_hash=run_snapshot_hash,
        master_seed=derive_pricing_master_seed(
            run_snapshot_hash,
            request.decision_group_id,
        ),
        candidate_count=len(request.candidates),
        sample_count=request.sample_count,
        batch_size=request.batch_size,
        objective=program["objective"],
        status="initializing",
        idempotency_key=(
            f"experiment-program:{program_id}:{expected_epoch}:"
            f"{expected_snapshot_hash}:pricing:set_price"
        ),
    )

    if run.status == "initializing":
        evidence_refs = _pricing_evidence_refs(request)
        for candidate in request.candidates:
            upsert_decision_simulation_candidate(
                simulation_run_id=run.id,
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                status="pending",
                candidate_payload=candidate.model_dump(mode="json", by_alias=True),
                evidence_refs=evidence_refs,
            )

    durable_candidates = list_decision_simulation_candidates(run.id)
    durable_ids = {candidate.candidate_id for candidate in durable_candidates}
    if durable_ids != set(candidate_links):
        raise ExperimentProgramStorageConflict(
            "durable pricing candidates do not match the pinned program frontier"
        )
    bind_program_simulation_run(
        program_id=program_id,
        decision_id=persisted.id,
        simulation_run_id=run.id,
        candidate_links=candidate_links,
        expected_epoch=expected_epoch,
        expected_snapshot_hash=expected_snapshot_hash,
    )
    set_manager_decision_simulation_run(
        persisted.id,
        simulation_run_id=run.id,
        status="evaluating",
    )

    current = get_decision_simulation_run(run.id) or run
    if current.status == "initializing":
        current = update_decision_simulation_run(
            current.id,
            status="queued",
            completed_samples=0,
            total_samples=len(request.candidates) * request.sample_count,
            completed_candidates=0,
            progress=0,
            error={},
        )
        append_decision_simulation_event(
            simulation_run_id=current.id,
            event_type="queued",
            event_id=f"{current.id}:queued",
            progress=0,
            payload={
                "domain": "pricing",
                "selectionMode": "winner_only",
                "modelVersion": current.model_version,
                "candidateCount": len(request.candidates),
                "sampleCount": request.sample_count,
                "completedSamples": 0,
                "totalSamples": len(request.candidates) * request.sample_count,
                "snapshotHash": current.evidence_hash,
                "experimentProgramId": program_id,
            },
        )
    if current.status == "queued":
        (executor or RQSimulationExecutor()).enqueue(current.id)
    elif current.status in _TERMINAL_RUN_STATUSES:
        sync_linked_simulation_run(current.id)
    return {
        "status": "queued" if current.status == "queued" else current.status,
        "programId": program_id,
        "decisionId": persisted.id,
        "simulationRunId": current.id,
        "candidateCount": len(candidate_links),
    }


def _validate_pricing_program_surface(
    bundle: Any,
    *,
    expected_epoch: int,
    expected_snapshot_hash: str,
) -> None:
    program = bundle.program
    if int(program["current_epoch"]) != int(expected_epoch):
        raise ProgramSimulationUnsupported(
            code="stale_program_epoch",
            message="Program epoch changed before simulation launch.",
        )
    if program["snapshot_hash"] != expected_snapshot_hash:
        raise ProgramSimulationUnsupported(
            code="stale_program_snapshot",
            message="Program snapshot changed before simulation launch.",
        )
    if program["mode"] != "shadow":
        raise ProgramSimulationUnsupported(
            code="pricing_program_shadow_only",
            message="Pricing program v1 supports shadow mode only.",
        )
    eligible = [
        row
        for row in bundle.candidates
        if row["epoch"] == expected_epoch and row["status"] == "eligible"
    ]
    groups = sorted(
        {
            (str(row["domain"]), str(row["family"]), str(row["operation"]))
            for row in eligible
        }
    )
    if not eligible or any(
        domain != "pricing"
        or family != "pricing"
        or operation not in {"set_price", "hold_price"}
        for domain, family, operation in groups
    ):
        raise ProgramSimulationUnsupported(
            code="unsupported_or_mixed_simulation_group",
            message=(
                "Program simulation v1 accepts one pricing execution group; "
                "mixed or unsupported eligible groups require a later portfolio executor."
            ),
            details={"eligibleGroups": groups},
        )
    if not 3 <= len(eligible) <= 7:
        raise ProgramSimulationUnsupported(
            code="pricing_candidate_count_out_of_bounds",
            message="Pricing simulation requires between three and seven eligible candidates.",
            details={"candidateCount": len(eligible)},
        )
    _pricing_packet(bundle.snapshot)


def _pricing_request_and_links(
    bundle: Any,
) -> tuple[PricingSimulationRequestV1, dict[str, str], dict[str, Any]]:
    program = bundle.program
    _validate_pricing_program_surface(
        bundle,
        expected_epoch=int(program["current_epoch"]),
        expected_snapshot_hash=str(program["snapshot_hash"]),
    )
    packet = _pricing_packet(bundle.snapshot)
    raw_value = packet.get("value")
    if not isinstance(raw_value, dict):
        raise ProgramSimulationUnsupported(
            code="invalid_pricing_simulation_contract",
            message="pricing_simulation_request evidence must contain an object value.",
        )
    raw_request = raw_value.get("simulationRequest", raw_value)
    if not isinstance(raw_request, dict):
        raise ProgramSimulationUnsupported(
            code="invalid_pricing_simulation_contract",
            message="Pinned pricing simulationRequest must be an object.",
        )
    try:
        request = PricingSimulationRequestV1.model_validate(
            {
                **raw_request,
                "snapshotHash": program["snapshot_hash"],
                "modelVersion": PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
                "sampleCount": 500,
                "batchSize": 50,
            }
        )
    except Exception as exc:
        raise ProgramSimulationUnsupported(
            code="invalid_pricing_simulation_contract",
            message="Pinned pricing request failed strict validation.",
            details={"validationError": str(exc)[:1000]},
        ) from exc

    eligible = [
        row
        for row in bundle.candidates
        if row["epoch"] == program["current_epoch"] and row["status"] == "eligible"
    ]
    program_by_price: dict[Decimal, dict[str, Any]] = {}
    sku_values: set[str] = set()
    for candidate in eligible:
        price = _candidate_price(candidate)
        if price in program_by_price:
            raise ProgramSimulationUnsupported(
                code="duplicate_program_candidate_price",
                message="Program frontier contains duplicate pricing candidates.",
            )
        program_by_price[price] = candidate
        sku_values.add(_candidate_sku(candidate))
    if sku_values != {request.sku_id}:
        raise ProgramSimulationUnsupported(
            code="pricing_candidate_target_mismatch",
            message="Every pricing candidate must target the request skuId.",
            details={"programSkuIds": sorted(sku_values), "requestSkuId": request.sku_id},
        )
    request_by_price = {_price_key(item.price_rub): item for item in request.candidates}
    if set(program_by_price) != set(request_by_price):
        raise ProgramSimulationUnsupported(
            code="pricing_candidate_mapping_mismatch",
            message="Pinned request prices must exactly match the finite program frontier.",
            details={
                "programPrices": sorted(str(value) for value in program_by_price),
                "requestPrices": sorted(str(value) for value in request_by_price),
            },
        )
    links: dict[str, str] = {}
    for price, request_candidate in request_by_price.items():
        program_candidate = program_by_price[price]
        if program_candidate["operation"] != request_candidate.operation:
            raise ProgramSimulationUnsupported(
                code="pricing_candidate_operation_mismatch",
                message="Program and request operations disagree for the same price.",
                details={"priceRub": str(price)},
            )
        links[request_candidate.candidate_id] = program_candidate["id"]
    return request, links, packet


def _pricing_packet(snapshot: dict[str, Any]) -> dict[str, Any]:
    packets = snapshot.get("evidence_packets") or []
    matches = [
        packet
        for packet in packets
        if isinstance(packet, dict)
        and packet.get("metric") == _PRICING_INPUT_METRIC
        and packet.get("status") == "available"
    ]
    if len(matches) != 1:
        raise ProgramSimulationUnsupported(
            code="missing_pricing_simulation_contract",
            message=(
                "Exactly one available pricing_simulation_request evidence packet "
                "is required for forward launch."
            ),
            details={"matchingPacketCount": len(matches)},
        )
    return matches[0]


def _candidate_sku(candidate: dict[str, Any]) -> str:
    target = candidate.get("target") or {}
    for key in ("nmId", "nm_id", "skuId", "sku_id", "sku"):
        value = target.get(key)
        if value not in (None, ""):
            return str(value)
    raise ProgramSimulationUnsupported(
        code="pricing_candidate_target_missing",
        message="Every pricing candidate requires nmId or skuId in target.",
    )


def _candidate_price(candidate: dict[str, Any]) -> Decimal:
    parameters = candidate.get("parameters") or {}
    for key in ("priceRub", "price_rub", "price", "toValue", "to_value"):
        value = parameters.get(key)
        if value not in (None, ""):
            return _price_key(value)
    raise ProgramSimulationUnsupported(
        code="pricing_candidate_price_missing",
        message="Every pricing candidate requires priceRub in parameters.",
    )


def _price_key(value: Any) -> Decimal:
    try:
        price = Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise ProgramSimulationUnsupported(
            code="invalid_pricing_candidate_price",
            message="Pricing candidate price must be a finite decimal number.",
        ) from exc
    if not price.is_finite() or price <= 0:
        raise ProgramSimulationUnsupported(
            code="invalid_pricing_candidate_price",
            message="Pricing candidate price must be finite and positive.",
        )
    return price


def _pricing_evidence_refs(request: PricingSimulationRequestV1) -> list[str]:
    evidence = request.evidence
    raw = [
        request.current_price_evidence_ref,
        evidence.daily_observations.evidence_ref,
        evidence.elasticity.evidence_ref,
        evidence.realization.evidence_ref,
        evidence.unit_economics.evidence_ref,
        evidence.available_stock.evidence_ref,
    ]
    raw.extend(item.evidence_ref for item in evidence.daily_observations.values or [])
    return list(dict.fromkeys(str(item) for item in raw if item))


def _pricing_launch_job_id(*, program_id: str, epoch: int, snapshot_hash: str) -> str:
    digest = hashlib.sha256(
        f"{program_id}:{epoch}:{snapshot_hash}:pricing:set_price".encode("utf-8")
    ).hexdigest()
    return f"experiment-program-launch-{digest}"


def _build_simulation_queue() -> Queue:
    settings = get_settings()
    return Queue(
        name=settings.decision_simulation_queue_name,
        connection=Redis.from_url(settings.redis_url),
        default_timeout=settings.decision_simulation_job_timeout_seconds,
    )


def _job_status(job: Any) -> str:
    if not hasattr(job, "get_status"):
        return "unknown"
    raw = job.get_status(refresh=True)
    return str(getattr(raw, "value", raw)).lower()


__all__ = [
    "enqueue_supported_program_simulation",
    "launch_pricing_program_simulation_job",
]
