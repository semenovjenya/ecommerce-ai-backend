from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agi_ecom_app.decision_simulation import jobs, service
from agi_ecom_app.decision_simulation.engine import simulation_runtime_card
from agi_ecom_app.decision_simulation.pricing_engine import (
    PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
    derive_pricing_master_seed,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    pricing_request_snapshot_payload,
)
from agi_ecom_app.decision_simulation.service import canonical_snapshot_hash
from agi_ecom_app.manager_decision.models import ManagerDecisionV1, RequestCoverageV1
from agi_ecom_app.services.memory import (
    DecisionSimulationBatchRecord,
    DecisionSimulationRunRecord,
)
from tests.test_pricing_simulation_engine import _default_request


NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)


def _pricing_run() -> DecisionSimulationRunRecord:
    request = _default_request()
    decision = ManagerDecisionV1(
        decisionId="pricing-decision-42",
        queryMode="decision",
        status="evaluating",
        createdAt=NOW,
        evidenceAsOf=NOW,
        requestCoverage=RequestCoverageV1(
            explicitTargets=["42"],
            coveredTargets=["42"],
            requestedOutputs=["price_simulation"],
            coveredOutputs=["price_simulation"],
            complete=True,
        ),
        rankedActions=[],
        rejectedCandidates=[],
        blockers=[],
        evidenceSnapshot=[],
    )
    snapshot = {
        "schemaVersion": "pricing-decision-simulation-snapshot-v1",
        "sourceSchemaVersion": "pricing-source-snapshot-v1",
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
        "pricingSource": {"nmId": 42, "shadowOnly": True},
        "hardGateRejectedCandidates": [],
    }
    snapshot_hash = canonical_snapshot_hash(snapshot)
    return DecisionSimulationRunRecord(
        id="pricing-run-42",
        decision_id=decision.decision_id,
        chat_id=None,
        run_id=None,
        mode="shadow",
        status="queued",
        objective=decision.objective,
        model_version=PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION,
        evidence_snapshot=snapshot,
        evidence_hash=snapshot_hash,
        master_seed=derive_pricing_master_seed(
            snapshot_hash,
            request.decision_group_id,
        ),
        sample_count=500,
        batch_size=50,
        candidate_count=len(request.candidates),
        completed_samples=0,
        total_samples=len(request.candidates) * 500,
        completed_candidates=0,
        progress=0,
        provisional_winner_candidate_id=None,
        winner_candidate_id=None,
        critic_result={},
        error={},
        idempotency_key="pricing-test",
        retry_of_run_id=None,
        superseded_by_run_id=None,
        retry_attempt=0,
        heartbeat_at=None,
        started_at=None,
        completed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


@contextmanager
def _acquired_lock(_: str):
    yield True


def test_pricing_job_persists_10x50_batches_and_shadow_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _pricing_run()
    candidates: dict[str, SimpleNamespace] = {}
    batches: dict[tuple[str, int], DecisionSimulationBatchRecord] = {}
    events: list[dict] = []
    projections: list[ManagerDecisionV1] = []
    links: list[dict] = []

    def get_run(run_id: str):
        assert run_id == state.id
        return state

    def update_run(run_id: str, **changes):
        nonlocal state
        assert run_id == state.id
        state = replace(state, **changes, updated_at=NOW)
        return state

    def update_progress(run_id: str, **changes):
        completed = int(changes.get("completed_samples", state.completed_samples))
        total = int(changes.get("total_samples", state.total_samples))
        return update_run(
            run_id,
            **changes,
            progress=(completed / total if total else 0),
        )

    def complete_run(run_id: str, **changes):
        return update_run(
            run_id,
            **changes,
            status="completed",
            progress=1.0,
            completed_at=NOW,
        )

    def upsert_candidate(**payload):
        current = candidates.get(payload["candidate_id"])
        values = {
            "simulation_run_id": payload["simulation_run_id"],
            "candidate_id": payload["candidate_id"],
            "action_id": payload["action_id"],
            "status": payload.get("status", getattr(current, "status", "pending")),
            "candidate_payload": payload.get("candidate_payload", {}),
            "evidence_refs": payload.get("evidence_refs", []),
            "constraint_violations": payload.get("constraint_violations", []),
            "distribution_summary": payload.get("distribution_summary", {}),
            "score_breakdown": payload.get("score_breakdown", {}),
            "rank": payload.get("rank"),
            "is_winner": payload.get("is_winner", False),
        }
        record = SimpleNamespace(**values)
        candidates[record.candidate_id] = record
        return record

    def upsert_batch(**payload):
        key = (payload["candidate_id"], payload["batch_index"])
        record = DecisionSimulationBatchRecord(
            id=len(batches) + 1,
            simulation_run_id=payload["simulation_run_id"],
            candidate_id=payload["candidate_id"],
            batch_index=payload["batch_index"],
            seed=payload["seed"],
            sample_count=payload["sample_count"],
            status=payload["status"],
            aggregates=payload["aggregates"],
            duration_ms=payload["duration_ms"],
            error={},
            completed_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        )
        batches[key] = record
        return record

    monkeypatch.setattr(jobs, "get_decision_simulation_run", get_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_progress", update_progress)
    monkeypatch.setattr(jobs, "complete_decision_simulation_run", complete_run)
    monkeypatch.setattr(jobs, "_simulation_run_lock", _acquired_lock)
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_batches",
        lambda *args, **kwargs: list(batches.values()),
    )
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_candidates",
        lambda *args, **kwargs: list(candidates.values()),
    )
    monkeypatch.setattr(jobs, "upsert_decision_simulation_batch", upsert_batch)
    monkeypatch.setattr(jobs, "upsert_decision_simulation_candidate", upsert_candidate)
    monkeypatch.setattr(
        jobs,
        "append_decision_simulation_event",
        lambda **payload: events.append(payload),
    )
    monkeypatch.setattr(jobs, "clear_decision_simulation_winner_flags", lambda _: None)
    monkeypatch.setattr(jobs, "list_decision_simulation_runs", lambda **_: [])
    monkeypatch.setattr(
        jobs,
        "backfill_decision_simulation_superseded_events",
        lambda **_: [],
    )
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(
            decision_simulation_llm_critic_enabled=False,
            decision_simulation_stale_after_seconds=1_200,
        ),
    )
    monkeypatch.setattr(
        jobs,
        "persist_manager_decision_bundle",
        lambda decision, **kwargs: projections.append(decision),
    )
    monkeypatch.setattr(
        jobs,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append(
            {"decision_id": decision_id, **kwargs}
        ),
    )

    result = jobs.run_decision_simulation_job(state.id)

    assert result["status"] == "completed"
    assert result["winner_candidate_id"] is not None
    assert state.status == "completed"
    assert state.completed_samples == 1_500
    assert len(batches) == 30
    assert all(batch.sample_count == 50 for batch in batches.values())
    assert {batch.batch_index for batch in batches.values()} == set(range(10))
    assert len({batch.seed for batch in batches.values()}) == 10
    for index in range(10):
        assert len(
            {
                batch.seed
                for (candidate_id, batch_index), batch in batches.items()
                if batch_index == index
            }
        ) == 1
    assert projections[-1].status == "needs_data"
    assert projections[-1].ranked_actions == []
    assert projections[-1].blockers[0].code == "pricing_shadow_only"
    assert links[-1] == {
        "decision_id": state.decision_id,
        "simulation_run_id": state.id,
        "status": "needs_data",
    }
    assert any(event["event_type"] == "ranking_completed" for event in events)
    assert any(event["event_type"] == "critic_completed" for event in events)
    assert any(event["event_type"] == "completed" for event in events)


def test_pricing_shadow_terminal_projection_requires_no_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = replace(_pricing_run(), status="completed", completed_at=NOW)
    monkeypatch.setattr(
        jobs,
        "get_manager_decision",
        lambda _: SimpleNamespace(simulation_run_id=run.id, status="needs_data"),
    )
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_events",
        lambda *args, **kwargs: [SimpleNamespace(event_type="completed")],
    )
    monkeypatch.setattr(
        jobs,
        "list_agent_action_experiments_for_decision",
        lambda _: [],
    )

    assert jobs._terminal_projection_is_complete(run) is True


def test_pricing_critic_in_flight_claim_is_not_submitted_to_llm_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = replace(
        _pricing_run(),
        critic_result={
            "submissionStatus": "in_flight",
            "rankingFingerprint": "f" * 64,
            "claimedAt": NOW.isoformat(),
        },
    )
    request = _default_request()
    numeric = jobs.PricingSimulationEngine().run(request)
    critic_input = jobs._critic_input(run, jobs._ranked(numeric.candidates), [])
    external_critic = Mock(
        side_effect=AssertionError("durably claimed LLM request was submitted twice")
    )
    durable_update = Mock()
    monkeypatch.setattr(jobs, "run_decision_critic", external_critic)
    monkeypatch.setattr(jobs, "update_decision_simulation_run", durable_update)

    execution, reused = jobs._execute_or_reuse_decision_critic(
        run,
        critic_input,
        settings=SimpleNamespace(
            decision_simulation_llm_critic_enabled=True,
            decision_simulation_stale_after_seconds=1_200,
        ),
    )

    external_critic.assert_not_called()
    durable_update.assert_not_called()
    assert reused is False
    assert execution.llm_available is False
    assert execution.effective_action == "record_only"
    assert execution.result.verdict == "blocked"
    assert execution.result.issues[0].code == "llm_critic_unavailable"
    assert execution.debug == {"reason": "durable_in_flight_claim_without_result"}


def test_pricing_critic_retry_keeps_pricing_candidates_and_group_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = replace(_pricing_run(), status="blocked", retry_attempt=0)
    created: list[dict] = []
    runs: dict[str, DecisionSimulationRunRecord] = {}
    persisted_candidates: list[dict] = []
    enqueued: list[str] = []

    def create_run(**payload):
        created.append(payload)
        run = replace(
            previous,
            id=payload["simulation_run_id"],
            status=payload["status"],
            evidence_snapshot=payload["evidence_snapshot"],
            evidence_hash=payload["evidence_hash"],
            master_seed=payload["master_seed"],
            idempotency_key=payload["idempotency_key"],
            retry_of_run_id=payload["retry_of_run_id"],
            retry_attempt=payload["retry_attempt"],
            superseded_by_run_id=None,
            heartbeat_at=None,
            completed_at=None,
        )
        runs[run.id] = run
        return run

    def update_run(run_id: str, **changes):
        runs[run_id] = replace(runs[run_id], **changes)
        return runs[run_id]

    monkeypatch.setattr(service, "create_decision_simulation_run", create_run)
    monkeypatch.setattr(
        service,
        "claim_decision_simulation_initialization",
        lambda run_id: runs[run_id],
    )
    monkeypatch.setattr(service, "get_decision_simulation_run", runs.get)
    monkeypatch.setattr(service, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(
        service,
        "upsert_decision_simulation_candidate",
        lambda **payload: persisted_candidates.append(payload),
    )
    monkeypatch.setattr(
        service,
        "mark_decision_simulation_run_superseded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service,
        "set_manager_decision_simulation_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service,
        "persist_manager_decision_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service,
        "append_decision_simulation_event",
        lambda **payload: payload,
    )
    monkeypatch.setattr(
        service,
        "RQSimulationExecutor",
        lambda: SimpleNamespace(enqueue=lambda run_id: enqueued.append(run_id)),
    )

    replacement = service.create_critic_recalculation(previous)

    assert replacement.status == "queued"
    assert replacement.retry_of_run_id == previous.id
    assert replacement.retry_attempt == 1
    assert enqueued == [replacement.id]
    assert len(persisted_candidates) == 3
    assert all(
        payload["candidate_payload"]["priceRub"] > 0
        for payload in persisted_candidates
    )
    snapshot_hash = canonical_snapshot_hash(created[0]["evidence_snapshot"])
    assert created[0]["master_seed"] == derive_pricing_master_seed(
        snapshot_hash,
        "price:sku-42",
    )
