from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("numpy")

from agi_ecom_app.experiment_program import pricing_launcher
from agi_ecom_app.experiment_program.executor_contract import (
    ProgramSimulationUnsupported,
)


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
SNAPSHOT_HASH = "a" * 64


def _request_payload() -> dict:
    observations = [
        {
            "observedDate": f"2026-07-{day:02d}",
            "priceRub": 100,
            "unitsSold": 10 + day,
            "completed": True,
            "notObservedOutOfStock": True,
            "evidenceRef": f"daily:{day}",
        }
        for day in range(1, 8)
    ]
    return {
        "decisionGroupId": "pricing:sku-1",
        "skuId": "sku-1",
        "currentPriceRub": 100,
        "currentPriceEvidenceRef": "price:sku-1",
        "horizonDays": 7,
        "candidates": [
            {
                "candidateId": "price-90",
                "actionId": "price-90",
                "operation": "set_price",
                "priceRub": 90,
            },
            {
                "candidateId": "price-100",
                "actionId": "price-100",
                "operation": "hold_price",
                "priceRub": 100,
            },
            {
                "candidateId": "price-110",
                "actionId": "price-110",
                "operation": "set_price",
                "priceRub": 110,
            },
        ],
        "evidence": {
            "dailyObservations": {
                "values": observations,
                "evidenceRef": "daily:sku-1",
                "latestObservationAgeDays": 1,
            },
            "elasticity": {
                "bootstrapValues": [-1.5, -1.2, -1.0],
                "source": "measured",
                "evidenceRef": "elasticity:sku-1",
            },
            "realization": {
                "bootstrapValues": [0.8, 0.9, 1.0],
                "source": "orders-to-sales",
                "evidenceRef": "realization:sku-1",
            },
            "unitEconomics": {
                "netRevenueRate": 0.8,
                "variableCostPerUnitRub": 50,
                "minimumUnitContributionRub": 0,
                "observationCount": 30,
                "latestObservationAgeDays": 1,
                "basis": "actual",
                "evidenceRef": "economics:sku-1",
            },
            "availableStock": {
                "units": 1000,
                "snapshotDate": "2026-07-19",
                "ageDaysAtSnapshot": 1,
                "evidenceRef": "stock:sku-1",
            },
        },
    }


def _bundle(*, prices: tuple[int, ...] = (90, 100, 110)) -> SimpleNamespace:
    candidates = []
    for index, price in enumerate(prices, start=1):
        candidates.append(
            {
                "id": f"program-candidate-{index}",
                "program_id": "program-1",
                "epoch": 1,
                "candidate_key": f"candidate-{price}",
                "domain": "pricing",
                "family": "pricing",
                "operation": "hold_price" if price == 100 else "set_price",
                "target": {"nmId": "sku-1"},
                "parameters": {"priceRub": price},
                "status": "eligible",
            }
        )
    return SimpleNamespace(
        program={
            "id": "program-1",
            "objective": "incremental_contribution_profit",
            "mode": "shadow",
            "status": "simulating",
            "current_epoch": 1,
            "snapshot_hash": SNAPSHOT_HASH,
            "source_chat_id": None,
            "source_run_id": None,
            "created_at": NOW,
        },
        snapshot={
            "data_as_of": NOW,
            "created_at": NOW,
            "evidence_packets": [
                {
                    "metric": "pricing_simulation_request",
                    "status": "available",
                    "value": _request_payload(),
                    "evidenceRef": "program:pricing-input",
                }
            ],
        },
        candidates=candidates,
    )


def test_pricing_contract_maps_by_sku_and_price_not_candidate_name() -> None:
    request, links, _packet = pricing_launcher._pricing_request_and_links(_bundle())

    assert request.sku_id == "sku-1"
    assert links == {
        "price-90": "program-candidate-1",
        "price-100": "program-candidate-2",
        "price-110": "program-candidate-3",
    }


def test_pricing_contract_rejects_partial_frontier_mapping() -> None:
    with pytest.raises(ProgramSimulationUnsupported) as caught:
        pricing_launcher._pricing_request_and_links(_bundle(prices=(90, 100, 120)))

    assert caught.value.code == "pricing_candidate_mapping_mismatch"


def test_enqueue_uses_one_deterministic_launch_job(monkeypatch) -> None:
    queue = Mock()
    queue.name = "decision-simulations"
    queue.fetch_job.return_value = None
    monkeypatch.setattr(pricing_launcher, "get_program_bundle", lambda _id: _bundle())

    first = pricing_launcher.enqueue_supported_program_simulation(
        program_id="program-1",
        source_event_id="event-1",
        expected_epoch=1,
        expected_snapshot_hash=SNAPSHOT_HASH,
        queue=queue,
    )
    second_job_id = pricing_launcher._pricing_launch_job_id(
        program_id="program-1",
        epoch=1,
        snapshot_hash=SNAPSHOT_HASH,
    )

    assert first.details["launchJobId"] == second_job_id
    assert queue.enqueue_call.call_args.kwargs["job_id"] == second_job_id


def test_launch_job_binds_complete_mapping_before_numerical_enqueue(monkeypatch) -> None:
    bundle = _bundle()
    monkeypatch.setattr(pricing_launcher, "get_program_bundle", lambda _id: bundle)
    persisted = SimpleNamespace(
        id="decision-1",
        decision_json={
            "decisionId": "decision-1",
            "queryMode": "decision",
            "status": "evaluating",
            "createdAt": NOW.isoformat(),
            "requestCoverage": {
                "explicitTargets": ["sku-1"],
                "coveredTargets": ["sku-1"],
                "requestedOutputs": ["price_simulation"],
                "coveredOutputs": ["price_simulation"],
                "complete": True,
            },
            "rankedActions": [],
            "rejectedCandidates": [],
            "blockers": [],
            "evidenceSnapshot": [],
        },
    )
    monkeypatch.setattr(
        pricing_launcher,
        "persist_manager_decision_bundle",
        lambda *_args, **_kwargs: (persisted, []),
    )
    initial_run = SimpleNamespace(id="run-1", status="initializing")
    queued_run = SimpleNamespace(
        id="run-1",
        status="queued",
        model_version="pricing-elasticity-bootstrap-v1",
        evidence_hash="b" * 64,
    )
    monkeypatch.setattr(
        pricing_launcher,
        "create_decision_simulation_run",
        lambda **_kwargs: initial_run,
    )
    candidate_ids: list[str] = []

    def upsert_candidate(**kwargs) -> None:
        candidate_ids.append(kwargs["candidate_id"])

    monkeypatch.setattr(
        pricing_launcher, "upsert_decision_simulation_candidate", upsert_candidate
    )
    monkeypatch.setattr(
        pricing_launcher,
        "list_decision_simulation_candidates",
        lambda _run_id: [
            SimpleNamespace(candidate_id=candidate_id) for candidate_id in candidate_ids
        ],
    )
    bind = Mock(return_value=3)
    monkeypatch.setattr(pricing_launcher, "bind_program_simulation_run", bind)
    monkeypatch.setattr(
        pricing_launcher, "set_manager_decision_simulation_run", Mock()
    )
    monkeypatch.setattr(
        pricing_launcher, "get_decision_simulation_run", lambda _run_id: initial_run
    )
    monkeypatch.setattr(
        pricing_launcher, "update_decision_simulation_run", lambda *_args, **_kwargs: queued_run
    )
    monkeypatch.setattr(pricing_launcher, "append_decision_simulation_event", Mock())
    numerical_enqueue = Mock()

    result = pricing_launcher.launch_pricing_program_simulation_job(
        program_id="program-1",
        source_event_id="event-1",
        expected_epoch=1,
        expected_snapshot_hash=SNAPSHOT_HASH,
        executor=SimpleNamespace(enqueue=numerical_enqueue),
    )

    assert sorted(candidate_ids) == ["price-100", "price-110", "price-90"]
    assert bind.call_args.kwargs["candidate_links"] == {
        "price-90": "program-candidate-1",
        "price-100": "program-candidate-2",
        "price-110": "program-candidate-3",
    }
    numerical_enqueue.assert_called_once_with("run-1")
    assert result["status"] == "queued"
