from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from agi_ecom_app.decision_simulation import pricing_service
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingAvailableStockEvidenceV1,
    PricingCandidateV1,
    PricingCompletedDayObservationV1,
    PricingDailyObservationsEvidenceV1,
    PricingElasticityEvidenceV1,
    PricingRealizationEvidenceV1,
    PricingSimulationEvidenceV1,
    PricingSimulationRequestV1,
    PricingUnitEconomicsEvidenceV1,
)
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.services.memory import DecisionSimulationRunRecord


NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
MANAGER = ManagerActor(user_id=2, username="manager", role="manager")


class RecordingExecutor:
    def __init__(self) -> None:
        self.run_ids: list[str] = []

    def enqueue(self, simulation_run_id: str, *, replace_stale: bool = False) -> str:
        assert replace_stale is False
        self.run_ids.append(simulation_run_id)
        return f"decision-simulation-{simulation_run_id}"


def _request() -> PricingSimulationRequestV1:
    return PricingSimulationRequestV1(
        snapshotHash="0" * 64,
        decisionGroupId="pricing:sku:42",
        skuId="42",
        currentPriceRub=100,
        currentPriceEvidenceRef="pricing.wb_supplier_prices:42:2026-07-16",
        candidates=[
            PricingCandidateV1(
                candidateId="pricing:42:100",
                actionId="pricing:42:hold",
                operation="hold_price",
                priceRub=100,
            ),
            PricingCandidateV1(
                candidateId="pricing:42:90",
                actionId="pricing:42:set:90",
                operation="set_price",
                priceRub=90,
            ),
            PricingCandidateV1(
                candidateId="pricing:42:110",
                actionId="pricing:42:set:110",
                operation="set_price",
                priceRub=110,
            ),
        ],
        evidence=PricingSimulationEvidenceV1(
            dailyObservations=PricingDailyObservationsEvidenceV1(
                values=[
                    PricingCompletedDayObservationV1(
                        observedDate=date(2026, 7, 15),
                        priceRub=100,
                        unitsSold=10,
                        evidenceRef="price-day:42:2026-07-15",
                    )
                ],
                latestObservationAgeDays=1,
                evidenceRef="price-history:42",
            ),
            elasticity=PricingElasticityEvidenceV1(
                bootstrapValues=[-1.0, -1.1, -0.9],
                source="historical_price_volume_residual_bootstrap_v1",
                evidenceRef="price-elasticity:42",
            ),
            realization=PricingRealizationEvidenceV1(
                bootstrapValues=[0.8] * 100,
                source="cohort_order_to_realized_sale_day_bootstrap_v1",
                evidenceRef="sales-realization:42",
            ),
            unitEconomics=PricingUnitEconomicsEvidenceV1(
                netRevenueRate=0.75,
                variableCostPerUnitRub=25,
                observationCount=28,
                latestObservationAgeDays=1,
                basis="wb_net_payout_before_cogs",
                evidenceRef="price-economics:42",
            ),
            availableStock=PricingAvailableStockEvidenceV1(
                units=500,
                snapshotDate=date(2026, 7, 16),
                ageDaysAtSnapshot=1,
                evidenceRef="stock:42:2026-07-16",
            ),
        ),
    )


def _run_from_create(kwargs: dict[str, object]) -> DecisionSimulationRunRecord:
    return DecisionSimulationRunRecord(
        id=str(kwargs["simulation_run_id"]),
        decision_id=str(kwargs["decision_id"]),
        chat_id=None,
        run_id=None,
        mode=str(kwargs["mode"]),
        status=str(kwargs["status"]),
        objective=str(kwargs["objective"]),
        model_version=str(kwargs["model_version"]),
        evidence_snapshot=dict(kwargs["evidence_snapshot"]),  # type: ignore[arg-type]
        evidence_hash=str(kwargs["evidence_hash"]),
        master_seed=int(kwargs["master_seed"]),
        sample_count=int(kwargs["sample_count"]),
        batch_size=int(kwargs["batch_size"]),
        candidate_count=int(kwargs["candidate_count"]),
        completed_samples=0,
        total_samples=int(kwargs["candidate_count"]) * int(kwargs["sample_count"]),
        completed_candidates=0,
        progress=0.0,
        provisional_winner_candidate_id=None,
        winner_candidate_id=None,
        critic_result={},
        error={},
        idempotency_key=str(kwargs["idempotency_key"]),
        retry_of_run_id=None,
        superseded_by_run_id=None,
        retry_attempt=0,
        heartbeat_at=None,
        started_at=None,
        completed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _durable_run(
    *,
    snapshot: dict[str, object],
    status: str = "initializing",
    run_id: str = "pricing-durable-run",
) -> DecisionSimulationRunRecord:
    snapshot_hash = pricing_service.canonical_snapshot_hash(snapshot)
    return _run_from_create(
        {
            "simulation_run_id": run_id,
            "decision_id": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "manager:pricing-simulation:42:price-shadow-42",
                )
            ),
            "mode": "shadow",
            "status": status,
            "objective": "incremental_contribution_profit",
            "model_version": "pricing-elasticity-bootstrap-v1",
            "evidence_snapshot": snapshot,
            "evidence_hash": snapshot_hash,
            "master_seed": 42,
            "candidate_count": 3,
            "sample_count": 500,
            "batch_size": 50,
            "idempotency_key": "price-shadow-42",
        }
    )


def test_start_pricing_simulation_persists_snapshot_candidates_and_queues_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    request = _request()
    source_metadata = {
        "schemaVersion": "pricing-source-snapshot-v1",
        "currentPriceSnapshot": {"last_synced_at": NOW.isoformat()},
        "shadowOnly": True,
        "causalValid": False,
    }
    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        lambda **_: (request, source_metadata),
    )
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: None,
    )

    persist_calls: list[tuple[object, dict[str, object]]] = []

    def persist(decision: object, **kwargs: object):
        persist_calls.append((decision, kwargs))
        return (
            SimpleNamespace(
                id=decision.decision_id,  # type: ignore[attr-defined]
                simulation_run_id=None,
            ),
            [],
        )

    monkeypatch.setattr(pricing_service, "persist_manager_decision_bundle", persist)

    runs: dict[str, DecisionSimulationRunRecord] = {}

    def create_run(**kwargs: object) -> DecisionSimulationRunRecord:
        run = _run_from_create(kwargs)
        runs[run.id] = run
        return run

    def claim(run_id: str) -> DecisionSimulationRunRecord | None:
        run = runs[run_id]
        if run.status != "initializing" or run.heartbeat_at is not None:
            return None
        claimed = replace(run, heartbeat_at=NOW)
        runs[run_id] = claimed
        return claimed

    def update(run_id: str, **changes: object) -> DecisionSimulationRunRecord:
        updated = replace(runs[run_id], **changes)
        runs[run_id] = updated
        return updated

    monkeypatch.setattr(pricing_service, "create_decision_simulation_run", create_run)
    monkeypatch.setattr(pricing_service, "get_decision_simulation_run", runs.get)
    monkeypatch.setattr(
        pricing_service,
        "claim_decision_simulation_initialization",
        claim,
    )
    monkeypatch.setattr(pricing_service, "update_decision_simulation_run", update)

    candidates: list[dict[str, object]] = []
    links: list[tuple[str, dict[str, object]]] = []
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        pricing_service,
        "upsert_decision_simulation_candidate",
        lambda **kwargs: candidates.append(kwargs),
    )
    monkeypatch.setattr(
        pricing_service,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append((decision_id, kwargs)),
    )
    monkeypatch.setattr(
        pricing_service,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(kwargs),
    )
    executor = RecordingExecutor()

    run = pricing_service.start_pricing_simulation(
        nm_id=42,
        candidate_prices=[90, 110],
        horizon_days=7,
        actor=MANAGER,
        idempotency_key="price-shadow-42",
        executor=executor,
    )

    assert run.status == "queued"
    assert run.mode == "shadow"
    assert run.candidate_count == 3
    assert run.total_samples == 1500
    assert executor.run_ids == [run.id]
    assert len(candidates) == 3
    assert {item["status"] for item in candidates} == {"pending"}
    assert links == [
        (
            run.decision_id,
            {"simulation_run_id": run.id, "status": "evaluating"},
        )
    ]
    assert events[0]["event_type"] == "queued"
    assert events[0]["event_id"] == f"{run.id}:queued"
    assert events[0]["payload"] == {
        "domain": "pricing",
        "selectionMode": "winner_only",
        "modelVersion": "pricing-elasticity-bootstrap-v1",
        "candidateCount": 3,
        "sampleCount": 500,
        "completedSamples": 0,
        "totalSamples": 1500,
        "snapshotHash": run.evidence_hash,
    }
    snapshot = run.evidence_snapshot
    assert snapshot["domain"] == "pricing"
    assert snapshot["operation"] == "set_price"
    assert snapshot["selectionMode"] == "winner_only"
    assert "snapshotHash" not in snapshot["simulationRequest"]
    assert len(snapshot["simulationCandidates"]) == 3
    assert persist_calls[0][1] == {
        "source_key": "pricing-simulation:42:price-shadow-42",
        "create_experiments": False,
    }


def test_pricing_mode_off_fails_before_loading_analytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "off")

    def forbidden(**_: object):
        raise AssertionError("analytics source must not be loaded while mode is off")

    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        forbidden,
    )

    with pytest.raises(RuntimeError, match="disabled"):
        pricing_service.start_pricing_simulation(
            nm_id=42,
            actor=MANAGER,
            idempotency_key="price-shadow-42",
        )


def test_concurrent_initializer_does_not_duplicate_candidates_or_queue_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        lambda **_: (
            _request(),
            {"currentPriceSnapshot": {"last_synced_at": NOW.isoformat()}},
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: None,
    )
    monkeypatch.setattr(
        pricing_service,
        "persist_manager_decision_bundle",
        lambda decision, **_: (
            SimpleNamespace(id=decision.decision_id, simulation_run_id=None),
            [],
        ),
    )
    stored: dict[str, DecisionSimulationRunRecord] = {}

    def create(**kwargs: object) -> DecisionSimulationRunRecord:
        run = _run_from_create(kwargs)
        stored[run.id] = run
        return run

    monkeypatch.setattr(pricing_service, "create_decision_simulation_run", create)
    monkeypatch.setattr(pricing_service, "get_decision_simulation_run", stored.get)
    monkeypatch.setattr(
        pricing_service,
        "claim_decision_simulation_initialization",
        lambda _: None,
    )

    def forbidden(*_: object, **__: object):
        raise AssertionError("the losing initializer must not mutate durable state")

    monkeypatch.setattr(
        pricing_service,
        "upsert_decision_simulation_candidate",
        forbidden,
    )
    monkeypatch.setattr(
        pricing_service,
        "set_manager_decision_simulation_run",
        forbidden,
    )
    monkeypatch.setattr(
        pricing_service,
        "append_decision_simulation_event",
        forbidden,
    )
    executor = RecordingExecutor()

    run = pricing_service.start_pricing_simulation(
        nm_id=42,
        actor=MANAGER,
        idempotency_key="price-shadow-42",
        executor=executor,
    )

    assert run.status == "initializing"
    assert executor.run_ids == []


def test_existing_idempotency_key_rejects_a_different_start_contract_before_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    original_contract = pricing_service._start_contract(
        nm_id=42,
        candidate_prices=[90, 110],
        horizon_days=7,
    )
    existing = _durable_run(
        snapshot={
            "startRequest": original_contract,
            "startRequestHash": pricing_service.canonical_snapshot_hash(
                original_contract
            ),
        },
        status="queued",
    )
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: SimpleNamespace(simulation_run_id=existing.id),
    )
    monkeypatch.setattr(
        pricing_service,
        "get_decision_simulation_run",
        lambda _: existing,
    )

    def forbidden(**_: object):
        raise AssertionError("a conflicting replay must not reload analytics")

    monkeypatch.setattr(pricing_service, "load_pricing_simulation_source", forbidden)

    with pytest.raises(
        pricing_service.PricingSimulationIdempotencyConflict,
        match="different pricing request",
    ):
        pricing_service.start_pricing_simulation(
            nm_id=42,
            candidate_prices=[95, 105],
            horizon_days=7,
            actor=MANAGER,
            idempotency_key="price-shadow-42",
        )


def test_persist_race_rejects_a_different_idempotent_start_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    original_contract = pricing_service._start_contract(
        nm_id=42,
        candidate_prices=[90, 110],
        horizon_days=7,
    )
    existing = _durable_run(
        snapshot={
            "startRequest": original_contract,
            "startRequestHash": pricing_service.canonical_snapshot_hash(
                original_contract
            ),
        },
        status="queued",
    )
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: None,
    )
    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        lambda **_: (
            _request(),
            {"currentPriceSnapshot": {"last_synced_at": NOW.isoformat()}},
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "persist_manager_decision_bundle",
        lambda decision, **_: (
            SimpleNamespace(
                id=decision.decision_id,
                simulation_run_id=existing.id,
            ),
            [],
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "get_decision_simulation_run",
        lambda _: existing,
    )

    with pytest.raises(
        pricing_service.PricingSimulationIdempotencyConflict,
        match="different pricing request",
    ):
        pricing_service.start_pricing_simulation(
            nm_id=42,
            candidate_prices=[95, 105],
            horizon_days=7,
            actor=MANAGER,
            idempotency_key="price-shadow-42",
        )


def test_initializer_uses_the_claimed_durable_snapshot_not_fresh_source_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    fresh_request = _request()
    durable_request = fresh_request.model_copy(
        update={
            "candidates": [
                candidate.model_copy(
                    update={
                        "candidate_id": f"durable:{candidate.candidate_id}",
                        "action_id": f"durable:{candidate.action_id}",
                    }
                )
                for candidate in fresh_request.candidates
            ]
        }
    )
    start_contract = pricing_service._start_contract(
        nm_id=42,
        candidate_prices=[90, 110],
        horizon_days=7,
    )
    snapshot: dict[str, object] = {
        "simulationRequest": pricing_service.pricing_request_snapshot_payload(
            durable_request
        ),
        "simulationCandidates": [
            candidate.model_dump(mode="json", by_alias=True)
            for candidate in durable_request.candidates
        ],
        "startRequest": start_contract,
        "startRequestHash": pricing_service.canonical_snapshot_hash(start_contract),
    }
    stored = _durable_run(snapshot=snapshot)
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: None,
    )
    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        lambda **_: (
            fresh_request,
            {"currentPriceSnapshot": {"last_synced_at": NOW.isoformat()}},
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "persist_manager_decision_bundle",
        lambda decision, **_: (
            SimpleNamespace(id=decision.decision_id, simulation_run_id=None),
            [],
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "create_decision_simulation_run",
        lambda **_: stored,
    )
    monkeypatch.setattr(
        pricing_service,
        "claim_decision_simulation_initialization",
        lambda _: replace(stored, heartbeat_at=NOW),
    )
    monkeypatch.setattr(
        pricing_service,
        "update_decision_simulation_run",
        lambda _run_id, **changes: replace(stored, **changes),
    )
    upserts: list[dict[str, object]] = []
    monkeypatch.setattr(
        pricing_service,
        "upsert_decision_simulation_candidate",
        lambda **kwargs: upserts.append(kwargs),
    )
    monkeypatch.setattr(
        pricing_service,
        "set_manager_decision_simulation_run",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pricing_service,
        "append_decision_simulation_event",
        lambda **_: None,
    )
    executor = RecordingExecutor()

    run = pricing_service.start_pricing_simulation(
        nm_id=42,
        candidate_prices=[90, 110],
        horizon_days=7,
        actor=MANAGER,
        idempotency_key="price-shadow-42",
        executor=executor,
    )

    assert run.status == "queued"
    assert executor.run_ids == [stored.id]
    assert [item["candidate_id"] for item in upserts] == [
        candidate.candidate_id for candidate in durable_request.candidates
    ]
    assert not any(
        item["candidate_id"] in {
            candidate.candidate_id for candidate in fresh_request.candidates
        }
        for item in upserts
    )


@pytest.mark.parametrize(
    ("failure_stage", "failure_message", "expected_events"),
    [
        ("candidate_upsert", "candidate persistence failed", ["failed"]),
        ("enqueue", "redis is unavailable", ["queued", "failed"]),
    ],
)
def test_initialization_or_enqueue_failure_projects_decision_to_needs_data(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    failure_message: str,
    expected_events: list[str],
) -> None:
    monkeypatch.setenv("DECISION_SIMULATION_PRICING_MODE", "shadow")
    request = _request()
    monkeypatch.setattr(
        pricing_service,
        "load_pricing_simulation_source",
        lambda **_: (
            request,
            {"currentPriceSnapshot": {"last_synced_at": NOW.isoformat()}},
        ),
    )
    monkeypatch.setattr(
        pricing_service,
        "get_manager_decision_by_source_key",
        lambda _: None,
    )
    persisted_decisions: list[object] = []

    def persist(decision: object, **_: object):
        persisted_decisions.append(decision)
        return (
            SimpleNamespace(
                id=decision.decision_id,  # type: ignore[attr-defined]
                simulation_run_id=None,
            ),
            [],
        )

    monkeypatch.setattr(pricing_service, "persist_manager_decision_bundle", persist)
    runs: dict[str, DecisionSimulationRunRecord] = {}

    def create(**kwargs: object) -> DecisionSimulationRunRecord:
        run = _run_from_create(kwargs)
        runs[run.id] = run
        return run

    def update(run_id: str, **changes: object) -> DecisionSimulationRunRecord:
        runs[run_id] = replace(runs[run_id], **changes)
        return runs[run_id]

    monkeypatch.setattr(pricing_service, "create_decision_simulation_run", create)
    monkeypatch.setattr(pricing_service, "get_decision_simulation_run", runs.get)
    monkeypatch.setattr(
        pricing_service,
        "claim_decision_simulation_initialization",
        lambda run_id: replace(runs[run_id], heartbeat_at=NOW),
    )
    monkeypatch.setattr(pricing_service, "update_decision_simulation_run", update)
    def upsert(**_: object) -> None:
        if failure_stage == "candidate_upsert":
            raise RuntimeError(failure_message)

    monkeypatch.setattr(pricing_service, "upsert_decision_simulation_candidate", upsert)
    links: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        pricing_service,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append((decision_id, kwargs)),
    )
    events: list[str] = []
    monkeypatch.setattr(
        pricing_service,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(str(kwargs["event_type"])),
    )

    class FailingExecutor:
        def enqueue(
            self,
            simulation_run_id: str,
            *,
            replace_stale: bool = False,
        ) -> str:
            if failure_stage == "enqueue":
                raise RuntimeError(failure_message)
            return f"decision-simulation-{simulation_run_id}"

    with pytest.raises(pricing_service.PricingSimulationDeliveryError) as raised:
        pricing_service.start_pricing_simulation(
            nm_id=42,
            candidate_prices=[90, 110],
            horizon_days=7,
            actor=MANAGER,
            idempotency_key="price-shadow-42",
            executor=FailingExecutor(),
        )

    failed = runs[raised.value.simulation_run_id]
    assert failed.status == "failed"
    assert failed.error == {
        "code": "pricing_simulation_initialization_failed",
        "type": "RuntimeError",
        "message": failure_message,
    }
    projected = persisted_decisions[-1]
    assert projected.status == "needs_data"  # type: ignore[attr-defined]
    assert projected.ranked_actions == []  # type: ignore[attr-defined]
    assert [item.code for item in projected.blockers] == [  # type: ignore[attr-defined]
        "pricing_simulation_delivery_failed"
    ]
    assert links[-1] == (
        failed.decision_id,
        {"simulation_run_id": failed.id, "status": "needs_data"},
    )
    assert events == expected_events
