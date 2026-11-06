from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from agi_ecom_app.decision_simulation import jobs, service
from agi_ecom_app.manager_decision.candidate_builder import (
    build_manager_action_candidates,
)
from agi_ecom_app.manager_decision.models import ManagerDecisionWorkflowResult
from agi_ecom_app.manager_decision.workflow import run_manager_decision_workflow
from agi_ecom_app.manager_lifecycle import service as lifecycle_service
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.manager_lifecycle.service import ManagerLifecycleConflict
from agi_ecom_app.services.memory import (
    DecisionSimulationBatchRecord,
    DecisionSimulationRunRecord,
)


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_simulation_recovery_queries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jobs, "list_decision_simulation_runs", lambda **_: [])
    monkeypatch.setattr(
        jobs,
        "backfill_decision_simulation_superseded_events",
        lambda **_: [],
    )

    def claim_stale(run_id, *, claimed_status, heartbeat_at, error=None, **_):
        changes = {"status": claimed_status, "heartbeat_at": heartbeat_at}
        if error is not None:
            changes["error"] = error
        return jobs.update_decision_simulation_run(run_id, **changes)

    monkeypatch.setattr(jobs, "claim_stale_decision_simulation_run", claim_stale)


def _advertising_row(index: int) -> dict[str, object]:
    advert_id = 1_000 + index
    nm_id = 70_000_000 + index
    daily_spend = [
        80 + index * 10,
        90 + index * 10,
        95 + index * 10,
        100 + index * 10,
        105 + index * 10,
        110 + index * 10,
        120 + index * 10,
    ]
    return {
        "advert_id": advert_id,
        "campaign_name": f"Campaign {advert_id}",
        "campaign_status": 9,
        "campaign_snapshot_date": "2026-07-13",
        "campaign_retrieved_at": "2026-07-14T08:00:00Z",
        "nm_id": nm_id,
        "active_sku_count": 1,
        "bid_search_kopecks": 420,
        "bid_recommendations_kopecks": 0,
        "bid_snapshot_date": "2026-07-13",
        "bid_retrieved_at": "2026-07-14T08:00:00Z",
        "budget_remaining_rub": 900,
        "budget_snapshot_date": "2026-07-13",
        "budget_retrieved_at": "2026-07-14T08:00:00Z",
        "stats_period_start": "2026-07-07",
        "stats_period_end": "2026-07-13",
        "stats_day_count": 7,
        "stats_window_complete": True,
        "campaign_sku_attribution_reconciled": True,
        "ad_spend_7d_rub": sum(daily_spend),
        "clicks_7d": 100,
        "ad_orders_7d": 0,
        "ad_units_7d": 0,
        "daily_spend_min_rub": min(daily_spend),
        "daily_spend_median_rub": daily_spend[3],
        "daily_spend_max_rub": max(daily_spend),
        "daily_spend_values_rub": daily_spend,
        "stock_snapshot_date": "2026-07-13",
        "stock_on_hand": 0,
        "stock_full_for_audit": 25,
        "stock_snapshot_complete": True,
        "supplier_snapshot_date": "2026-07-13",
        "supplier_inbound_qty": 0,
        "supplier_eta_date": None,
        "supplier_goods_rows": 1,
        "supplier_snapshot_complete": True,
        "unit_cogs_rub": 250,
        "cogs_snapshot_date": "2026-07-13",
        "cogs_retrieved_at": "2026-07-14T08:00:00Z",
        "attributed_revenue_7d_rub": 0,
        "attributed_revenue_source_table": "test.campaign_attribution_daily",
        "campaign_contribution_profit_7d_rub": 0,
        "campaign_contribution_source_table": "test.campaign_contribution_daily",
    }


def _ready_workflow(count: int = 3) -> ManagerDecisionWorkflowResult:
    source = {
        "schemaVersion": "1.0",
        "retrievedAt": "2026-07-14T08:00:00Z",
        "advertisingRows": [_advertising_row(index) for index in range(1, count + 1)],
        "supplyRows": [],
    }
    previous_mode = os.environ.get("DECISION_SIMULATION_MODE")
    os.environ["DECISION_SIMULATION_MODE"] = "enforced"
    try:
        built = build_manager_action_candidates(source, now=NOW)
    finally:
        if previous_mode is None:
            os.environ.pop("DECISION_SIMULATION_MODE", None)
        else:
            os.environ["DECISION_SIMULATION_MODE"] = previous_mode
    result = run_manager_decision_workflow(
        {"actionCandidates": built.candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert result.decision.status == "ready_for_approval"
    return result


def _settings(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        decision_simulation_mode=mode,
        decision_simulation_sample_count=500,
        decision_simulation_batch_size=50,
        decision_simulation_job_timeout_seconds=900,
    )


def _run_record(
    *,
    run_id: str = "simulation-1",
    decision_id: str = "decision-1",
    status: str = "queued",
    mode: str = "enforced",
    snapshot: dict | None = None,
    evidence_hash: str = "a" * 64,
    idempotency_key: str | None = None,
    retry_of_run_id: str | None = None,
    retry_attempt: int = 0,
) -> DecisionSimulationRunRecord:
    return DecisionSimulationRunRecord(
        id=run_id,
        decision_id=decision_id,
        chat_id="chat-1",
        run_id=17,
        mode=mode,
        status=status,
        objective="incremental_contribution_profit",
        model_version="advertising-bootstrap-v1",
        evidence_snapshot=snapshot or {},
        evidence_hash=evidence_hash,
        master_seed=123,
        sample_count=500,
        batch_size=50,
        candidate_count=3,
        completed_samples=0,
        total_samples=1_500,
        completed_candidates=0,
        progress=0.0,
        provisional_winner_candidate_id=None,
        winner_candidate_id=None,
        critic_result={},
        error={},
        idempotency_key=idempotency_key,
        retry_of_run_id=retry_of_run_id,
        superseded_by_run_id=None,
        retry_attempt=retry_attempt,
        heartbeat_at=None,
        started_at=None,
        completed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _patch_schedule_storage(monkeypatch: pytest.MonkeyPatch):
    persisted: list = []
    candidates: list[dict] = []
    events: list[dict] = []
    links: list[dict] = []
    runs: dict[str, DecisionSimulationRunRecord] = {}
    monkeypatch.setattr(service, "get_manager_decision_by_source_key", lambda _: None)

    monkeypatch.setattr(
        service,
        "persist_manager_decision_bundle",
        lambda decision, **kwargs: persisted.append((decision, kwargs)),
    )

    def create_run(**kwargs):
        run = _run_record(
            run_id=kwargs["simulation_run_id"],
            decision_id=kwargs["decision_id"],
            status=kwargs.get("status", "queued"),
            mode=kwargs["mode"],
            snapshot=kwargs["evidence_snapshot"],
            evidence_hash=kwargs["evidence_hash"],
        )
        runs[run.id] = run
        return run

    monkeypatch.setattr(service, "create_decision_simulation_run", create_run)
    monkeypatch.setattr(service, "get_decision_simulation_run", runs.get)

    def claim_initialization(run_id):
        run = runs[run_id]
        if run.status != "initializing" or run.heartbeat_at is not None:
            return None
        claimed = replace(run, heartbeat_at=NOW)
        runs[run_id] = claimed
        return claimed

    def update_run(run_id, **changes):
        run = replace(runs[run_id], **changes)
        runs[run_id] = run
        return run

    monkeypatch.setattr(
        service,
        "claim_decision_simulation_initialization",
        claim_initialization,
    )
    monkeypatch.setattr(service, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(
        service,
        "upsert_decision_simulation_candidate",
        lambda **kwargs: candidates.append(kwargs),
    )
    monkeypatch.setattr(
        service,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        service,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: links.append(
            {"decision_id": decision_id, **kwargs}
        ),
    )
    return persisted, candidates, events, links


def test_mode_off_is_an_exact_no_side_effect_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ready_workflow()
    monkeypatch.setattr(service, "get_settings", lambda: _settings("off"))
    forbidden = Mock(side_effect=AssertionError("off mode touched durable state"))
    monkeypatch.setattr(service, "persist_manager_decision_bundle", forbidden)
    monkeypatch.setattr(service, "create_decision_simulation_run", forbidden)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned is result
    forbidden.assert_not_called()
    executor.enqueue.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "public_status", "persisted_status", "link_status"),
    [
        ("shadow", "ready_for_approval", "ready_for_approval", None),
        ("enforced", "evaluating", "evaluating", "evaluating"),
    ],
)
def test_shadow_and_enforced_create_the_same_durable_simulation_but_different_public_state(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    public_status: str,
    persisted_status: str,
    link_status: str | None,
) -> None:
    result = _ready_workflow()
    monkeypatch.setattr(service, "get_settings", lambda: _settings(mode))
    persisted, candidates, events, links = _patch_schedule_storage(monkeypatch)
    metric = Mock()
    monkeypatch.setattr(service, "ELIGIBILITY_OUTCOMES", metric)
    executor = Mock()
    executor.enqueue.return_value = "job-id"

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned.decision.status == public_status
    assert returned.decision.simulation_eligibility is not None
    assert returned.decision.simulation_eligibility.outcome == "queued"
    assert returned.decision.simulation_eligibility.submitted_candidate_count == 3
    assert returned.decision.simulation_eligibility.eligible_candidate_count == 3
    assert returned.decision.simulation_eligibility.rejected_candidate_count == 0
    # Initialization first claims the durable source key as a harmless ready
    # staging row, then projects evaluating only after the run/journal exist.
    assert persisted[-1][0].status == persisted_status
    assert persisted[-1][0].simulation_eligibility is not None
    assert persisted[-1][0].simulation_eligibility.outcome == "queued"
    assert persisted[0][1]["create_experiments"] is False
    assert len(candidates) == 3
    assert {item["candidate_payload"]["operation"] for item in candidates} == {
        "pause_campaign"
    }
    assert events[0]["event_type"] == "queued"
    assert events[0]["payload"]["totalSamples"] == 1_500
    assert links == [
        {
            "decision_id": result.decision.decision_id,
            "simulation_run_id": candidates[0]["simulation_run_id"],
            "status": link_status,
        }
    ]
    executor.enqueue.assert_called_once_with(candidates[0]["simulation_run_id"])
    if mode == "shadow":
        assert returned.decision.ranked_actions == result.decision.ranked_actions
    else:
        assert returned.decision.ranked_actions == []
    assert metric.labels.call_args_list == [
        call(mode=mode, outcome="eligible"),
        call(mode=mode, outcome="queued"),
    ]
    assert metric.labels.return_value.inc.call_count == 2


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    [("shadow", "ready_for_approval"), ("enforced", "blocked")],
)
def test_enqueue_failure_keeps_durable_eligibility_at_eligible_not_queued(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_status: str,
) -> None:
    result = _ready_workflow()
    monkeypatch.setattr(service, "get_settings", lambda: _settings(mode))
    persisted, _, events, _ = _patch_schedule_storage(monkeypatch)
    metric = Mock()
    monkeypatch.setattr(service, "ELIGIBILITY_OUTCOMES", metric)
    executor = Mock()
    executor.enqueue.side_effect = RuntimeError("redis unavailable")

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned.decision.status == expected_status
    assert returned.decision.simulation_eligibility is not None
    assert returned.decision.simulation_eligibility.outcome == "eligible"
    assert returned.decision.simulation_eligibility.reason_code == (
        "eligible_candidates_delivery_failed"
    )
    assert persisted[-1][0].simulation_eligibility is not None
    assert persisted[-1][0].simulation_eligibility.outcome == "eligible"
    assert [item["event_type"] for item in events][-1] == "failed"
    metric.labels.assert_called_once_with(mode=mode, outcome="eligible")
    metric.labels.return_value.inc.assert_called_once_with()


def test_shadow_failure_projection_cannot_restore_queued_eligibility() -> None:
    result = _ready_workflow()
    queued = service._with_simulation_eligibility(
        result,
        service._simulation_eligibility(
            outcome="queued",
            mode="shadow",
            submitted_candidate_count=3,
            eligible_candidate_count=3,
            reason_code="eligible_candidates_queued",
            record_metric=False,
        ),
    ).decision
    failed = replace(
        _run_record(
            mode="shadow",
            status="failed",
            snapshot={
                "managerDecision": queued.model_dump(mode="json", by_alias=True),
                "shadowPublicDecision": queued.model_dump(mode="json", by_alias=True),
            },
        ),
        error={"code": "enqueue_failed", "message": "redis unavailable"},
    )

    projected = jobs._shadow_public_decision(failed)

    assert projected.simulation_eligibility is not None
    assert projected.simulation_eligibility.outcome == "eligible"
    assert projected.simulation_eligibility.reason_code == (
        "eligible_candidates_delivery_failed"
    )


def test_initialization_failure_never_persists_or_counts_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ready_workflow()
    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))
    persisted, _, _, _ = _patch_schedule_storage(monkeypatch)
    monkeypatch.setattr(
        service,
        "create_decision_simulation_run",
        Mock(side_effect=RuntimeError("postgres unavailable")),
    )
    metric = Mock()
    monkeypatch.setattr(service, "ELIGIBILITY_OUTCOMES", metric)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned.decision.status == "blocked"
    assert returned.decision.simulation_eligibility is not None
    assert returned.decision.simulation_eligibility.outcome == "eligible"
    assert returned.decision.simulation_eligibility.reason_code == (
        "eligible_candidates_delivery_failed"
    )
    assert all(
        item[0].simulation_eligibility is not None
        and item[0].simulation_eligibility.outcome != "queued"
        for item in persisted
    )
    metric.labels.assert_called_once_with(mode="enforced", outcome="eligible")
    executor.enqueue.assert_not_called()


def test_enforced_does_not_schedule_when_fewer_than_three_candidates_pass_evidence_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ready_workflow()
    actions = list(original.decision.ranked_actions)
    incomplete_evidence = [
        fact
        for fact in actions[0].evidence
        if fact.metric != "daily_ad_spend_distribution"
    ]
    actions[0] = actions[0].model_copy(update={"evidence": incomplete_evidence})
    decision = original.decision.model_copy(update={"ranked_actions": actions})
    result = original.model_copy(update={"decision": decision})
    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))
    monkeypatch.setattr(service, "get_manager_decision_by_source_key", lambda _: None)
    forbidden = Mock(side_effect=AssertionError("ineligible run was persisted"))
    monkeypatch.setattr(service, "persist_manager_decision_bundle", forbidden)
    monkeypatch.setattr(service, "create_decision_simulation_run", forbidden)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned.decision.status == "needs_data"
    assert returned.decision.ranked_actions == []
    assert returned.decision.simulation_eligibility is not None
    assert returned.decision.simulation_eligibility.outcome == "insufficient"
    assert returned.decision.simulation_eligibility.submitted_candidate_count == 3
    assert returned.decision.simulation_eligibility.eligible_candidate_count == 2
    assert returned.decision.simulation_eligibility.rejected_candidate_count == 1
    assert returned.decision.blockers[0].code == "decision_simulation_insufficient_candidates"
    assert "hard gate: 2" in returned.decision.blockers[0].message
    forbidden.assert_not_called()
    executor.enqueue.assert_not_called()


def test_not_ready_decision_exposes_eligibility_without_creating_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = _ready_workflow()
    decision = ready.decision.model_copy(
        update={"status": "blocked", "ranked_actions": []}
    )
    result = ready.model_copy(update={"decision": decision})
    monkeypatch.setattr(service, "get_settings", lambda: _settings("shadow"))
    monkeypatch.setattr(service, "get_manager_decision_by_source_key", lambda _: None)
    forbidden = Mock(side_effect=AssertionError("not-ready decision created a run"))
    monkeypatch.setattr(service, "create_decision_simulation_run", forbidden)
    metric = Mock()
    monkeypatch.setattr(service, "ELIGIBILITY_OUTCOMES", metric)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    eligibility = returned.decision.simulation_eligibility
    assert eligibility is not None
    assert eligibility.outcome == "not_ready"
    assert eligibility.mode == "shadow"
    assert eligibility.reason_code == "decision_not_ready_for_approval"
    assert eligibility.submitted_candidate_count == 0
    forbidden.assert_not_called()
    executor.enqueue.assert_not_called()
    metric.labels.assert_called_once_with(mode="shadow", outcome="not_ready")
    metric.labels.return_value.inc.assert_called_once_with()


@pytest.mark.parametrize(
    ("mode", "expected_outcome"),
    [("shadow", "insufficient"), ("enforced", "insufficient")],
)
def test_insufficient_eligibility_outcome_is_counted_without_a_run(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_outcome: str,
) -> None:
    original = _ready_workflow()
    actions = list(original.decision.ranked_actions)
    actions[0] = actions[0].model_copy(
        update={
            "evidence": [
                fact
                for fact in actions[0].evidence
                if fact.metric != "daily_ad_spend_distribution"
            ]
        }
    )
    result = original.model_copy(
        update={
            "decision": original.decision.model_copy(
                update={"ranked_actions": actions}
            )
        }
    )
    monkeypatch.setattr(service, "get_settings", lambda: _settings(mode))
    monkeypatch.setattr(service, "get_manager_decision_by_source_key", lambda _: None)
    metric = Mock()
    monkeypatch.setattr(service, "ELIGIBILITY_OUTCOMES", metric)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        result,
        chat_id="chat-1",
        run_id=17,
        source_key="source-1",
        executor=executor,
    )

    assert returned.decision.simulation_eligibility is not None
    assert returned.decision.simulation_eligibility.outcome == expected_outcome
    metric.labels.assert_called_once_with(mode=mode, outcome=expected_outcome)
    metric.labels.return_value.inc.assert_called_once_with()
    executor.enqueue.assert_not_called()


def test_repeated_enforced_synthesis_with_completed_run_returns_stored_final_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = _ready_workflow()
    stored = incoming.decision.model_copy(
        update={"ranked_actions": list(reversed(incoming.decision.ranked_actions))}
    )
    completed = _run_record(
        run_id="completed-run",
        decision_id=incoming.decision.decision_id,
        status="completed",
        mode="enforced",
    )
    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))
    monkeypatch.setattr(service, "get_manager_decision_by_source_key", lambda _: None)
    monkeypatch.setattr(service, "get_decision_simulation_run", lambda _: completed)
    monkeypatch.setattr(
        service,
        "get_manager_decision",
        lambda _: SimpleNamespace(
            decision_json=stored.model_dump(mode="json", by_alias=True)
        ),
    )
    forbidden = Mock(side_effect=AssertionError("terminal run was scheduled again"))
    monkeypatch.setattr(service, "persist_manager_decision_bundle", forbidden)
    monkeypatch.setattr(service, "create_decision_simulation_run", forbidden)
    monkeypatch.setattr(service, "set_manager_decision_simulation_run", forbidden)
    executor = Mock()

    returned = service.maybe_schedule_decision_simulation(
        incoming,
        chat_id="chat-1",
        run_id=17,
        source_key="same-source",
        executor=executor,
    )

    assert returned.decision.status == "ready_for_approval"
    assert [item.action_id for item in returned.decision.ranked_actions] == [
        item.action_id for item in stored.ranked_actions
    ]
    assert returned.decision.ranked_actions != incoming.decision.ranked_actions
    forbidden.assert_not_called()
    executor.enqueue.assert_not_called()


class _FakeRQJob:
    def __init__(self, status: str = "queued") -> None:
        self.status = status
        self.cancelled = False
        self.deleted = False

    def get_status(self, *, refresh: bool) -> str:
        assert refresh is True
        return self.status

    def cancel(self) -> None:
        self.cancelled = True

    def delete(self) -> None:
        self.deleted = True


class _FakeRQQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, _FakeRQJob] = {}
        self.enqueue_calls: list[dict] = []

    def fetch_job(self, job_id: str):
        return self.jobs.get(job_id)

    def enqueue_call(self, **kwargs):
        self.enqueue_calls.append(kwargs)
        self.jobs[kwargs["job_id"]] = _FakeRQJob()

    def __len__(self) -> int:
        return len(self.jobs)


def test_rq_delivery_uses_deterministic_job_id_and_does_not_enqueue_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rq.job import validate_job_id

    run = _run_record(run_id="simulation-stable")
    queue = _FakeRQQueue()
    monkeypatch.setattr(service, "get_decision_simulation_run", lambda _: run)
    monkeypatch.setattr(service, "get_settings", lambda: _settings("enforced"))

    first = service.enqueue_decision_simulation(run.id, queue=queue)
    second = service.enqueue_decision_simulation(run.id, queue=queue)

    assert first == second == "decision-simulation-simulation-stable"
    validate_job_id(first)
    assert len(queue.enqueue_calls) == 1
    assert queue.enqueue_calls[0]["job_id"] == first
    assert queue.enqueue_calls[0]["args"] == (run.id,)
    assert queue.enqueue_calls[0]["func"].endswith(".run_decision_simulation_job")


def test_admin_retry_returns_the_existing_idempotent_run_without_new_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _run_record(status="blocked", run_id="failed-run")
    retry = _run_record(
        run_id="retry-run",
        status="queued",
        decision_id=previous.decision_id,
        idempotency_key="retry-key",
        retry_of_run_id=previous.id,
        retry_attempt=1,
    )
    monkeypatch.setattr(
        service,
        "get_decision_simulation_run",
        lambda simulation_run_id: previous if simulation_run_id == previous.id else None,
    )
    monkeypatch.setattr(
        service,
        "list_decision_simulation_runs",
        lambda **kwargs: [retry],
    )
    create_retry = Mock(side_effect=AssertionError("idempotent retry was recreated"))
    monkeypatch.setattr(service, "_create_retry_run", create_retry)

    returned = service.retry_decision_simulation(
        simulation_run_id=previous.id,
        idempotency_key="retry-key",
        actor=ManagerActor(user_id=1, username="admin", role="admin"),
    )

    assert returned is retry
    create_retry.assert_not_called()


def test_approval_race_is_closed_while_linked_enforced_critic_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = replace(
        _run_record(run_id="critic-running", status="critic_running"),
        critic_result={},
        winner_candidate_id=None,
    )
    experiment = SimpleNamespace(id="experiment-1", decision_id=simulation.decision_id)
    decision = SimpleNamespace(
        id=simulation.decision_id,
        status="ready_for_approval",
        simulation_run_id=simulation.id,
    )
    monkeypatch.setattr(
        lifecycle_service,
        "get_settings",
        lambda: SimpleNamespace(manager_decision_enabled=True),
    )
    monkeypatch.setattr(
        lifecycle_service,
        "get_agent_action_experiment",
        lambda _: experiment,
    )
    monkeypatch.setattr(lifecycle_service, "get_manager_decision", lambda _: decision)
    monkeypatch.setattr(
        lifecycle_service,
        "get_decision_simulation_run",
        lambda _: simulation,
    )
    claim = Mock(side_effect=AssertionError("approval was claimed during critic race"))
    monkeypatch.setattr(
        lifecycle_service,
        "claim_agent_action_experiment_approval",
        claim,
    )

    with pytest.raises(
        ManagerLifecycleConflict,
        match="enforced decision simulation is not completed and approved",
    ):
        lifecycle_service.approve_manager_action(
            experiment_id=experiment.id,
            actor=ManagerActor(user_id=2, username="manager", role="manager"),
            idempotency_key="approval-race",
        )

    claim.assert_not_called()


def _job_run(result: ManagerDecisionWorkflowResult) -> DecisionSimulationRunRecord:
    simulation_candidates = [
        service.advertising_candidate_from_action(action)
        for action in result.decision.ranked_actions
    ]
    snapshot = service.build_evidence_snapshot(result.decision, simulation_candidates)
    evidence_hash = service.canonical_snapshot_hash(snapshot)
    return replace(
        _run_record(
            run_id="worker-run",
            decision_id=result.decision.decision_id,
            status="queued",
            mode="enforced",
            snapshot=snapshot,
            evidence_hash=evidence_hash,
        ),
        master_seed=service.derive_master_seed(evidence_hash),
    )


class _CriticWithIgnoredWinnerOverride:
    def __init__(self, *, ranking_fingerprint: str, attempted_winner: str) -> None:
        self.effective_action = "continue"
        self.llm_available = True
        self.blocks_decision = False
        self.result = SimpleNamespace(
            verdict="approved",
            issues=[],
            summary="Grounded numerical ranking is safe to publish.",
        )
        self.ranking_fingerprint = ranking_fingerprint
        self.attempted_winner = attempted_winner

    def model_dump(self, **kwargs) -> dict[str, object]:
        del kwargs
        return {
            "mode": "enforced",
            "result": {
                "verdict": "approved",
                "issues": [],
                "summary": self.result.summary,
                # Deliberately simulate a non-contract field from an untrusted LLM
                # boundary. The worker must never use it for publication.
                "attemptedWinnerCandidateId": self.attempted_winner,
            },
            "effectiveAction": "continue",
            "llmAvailable": True,
            "blocksDecision": False,
            "rankingFingerprint": self.ranking_fingerprint,
        }


def test_worker_resumes_batches_publishes_numeric_winner_and_simulation_economics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _ready_workflow()
    initial = _job_run(workflow)
    state = initial
    events: list[dict] = []
    batch_writes: list[dict] = []
    candidate_writes: list[dict] = []
    published = []
    decision_links: list[dict] = []
    critic_inputs = []
    simulated_batches: list[tuple[str, int]] = []

    existing_candidate_id = initial.evidence_snapshot["simulationCandidates"][0][
        "candidateId"
    ]
    existing_request = jobs._request_from_run(initial)
    existing_batch_result = jobs.DecisionSimulationEngine().simulate_batch(
        existing_request,
        existing_request.candidates[0],
        0,
    )
    existing_batch = DecisionSimulationBatchRecord(
        id=1,
        simulation_run_id=initial.id,
        candidate_id=existing_candidate_id,
        batch_index=0,
        seed=existing_batch_result.seed,
        sample_count=50,
        status="completed",
        aggregates=existing_batch_result.distribution.model_dump(
            mode="json", by_alias=True
        ),
        duration_ms=1.0,
        error={},
        completed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )

    def get_run(simulation_run_id: str):
        assert simulation_run_id == initial.id
        return state

    def update_run(simulation_run_id: str, **changes):
        nonlocal state
        assert simulation_run_id == initial.id
        state = replace(state, **changes)
        return state

    def update_progress(simulation_run_id: str, **changes):
        nonlocal state
        assert simulation_run_id == initial.id
        values = dict(changes)
        completed = int(values.get("completed_samples", state.completed_samples))
        total = int(values.get("total_samples", state.total_samples))
        values["progress"] = completed / total if total else 0.0
        state = replace(state, **values)
        return state

    @contextmanager
    def acquired_lock(simulation_run_id: str):
        assert simulation_run_id == initial.id
        yield True

    def critic(*, critic_input, mode):
        assert mode == "enforced"
        critic_inputs.append(critic_input)
        attempted_winner = critic_input.top_three[-1]
        assert attempted_winner != critic_input.top_three[0]
        return _CriticWithIgnoredWinnerOverride(
            ranking_fingerprint=critic_input.ranking_fingerprint,
            attempted_winner=attempted_winner,
        )

    monkeypatch.setattr(jobs, "get_decision_simulation_run", get_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_run", update_run)
    monkeypatch.setattr(jobs, "update_decision_simulation_progress", update_progress)
    def complete_run(simulation_run_id: str, **changes):
        nonlocal state
        assert simulation_run_id == initial.id
        winner = changes["winner_candidate_id"]
        state = replace(
            state,
            status="completed",
            progress=1,
            provisional_winner_candidate_id=winner,
            winner_candidate_id=winner,
            completed_at=NOW,
            **{
                key: value
                for key, value in changes.items()
                if key != "winner_candidate_id"
            },
        )
        return state

    monkeypatch.setattr(jobs, "complete_decision_simulation_run", complete_run)
    monkeypatch.setattr(jobs, "clear_decision_simulation_winner_flags", lambda _: None)
    monkeypatch.setattr(jobs, "_simulation_run_lock", acquired_lock)
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_batches",
        lambda *args, **kwargs: [existing_batch],
    )
    monkeypatch.setattr(
        jobs,
        "list_decision_simulation_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        jobs,
        "upsert_decision_simulation_batch",
        lambda **kwargs: batch_writes.append(kwargs),
    )
    monkeypatch.setattr(
        jobs,
        "upsert_decision_simulation_candidate",
        lambda **kwargs: candidate_writes.append(kwargs),
    )
    monkeypatch.setattr(
        jobs,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        jobs,
        "persist_manager_decision_bundle",
        lambda decision, **kwargs: published.append((decision, kwargs)),
    )
    monkeypatch.setattr(
        jobs,
        "set_manager_decision_simulation_run",
        lambda decision_id, **kwargs: decision_links.append(
            {"decision_id": decision_id, **kwargs}
        ),
    )
    monkeypatch.setattr(jobs, "run_decision_critic", critic)
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(
            decision_simulation_llm_critic_enabled=True,
            decision_simulation_stale_after_seconds=1_200,
        ),
    )
    block = Mock(side_effect=AssertionError("approved simulation was blocked"))
    monkeypatch.setattr(jobs, "block_manager_decision_for_simulation", block)

    base_engine = jobs.DecisionSimulationEngine

    class TrackingEngine(base_engine):
        def simulate_batch(self, request, candidate, batch_index):  # type: ignore[no-untyped-def]
            simulated_batches.append((candidate.candidate_id, batch_index))
            return super().simulate_batch(request, candidate, batch_index)

    monkeypatch.setattr(jobs, "DecisionSimulationEngine", TrackingEngine)

    outcome = jobs.run_decision_simulation_job(initial.id)

    assert outcome == {
        "simulation_run_id": initial.id,
        "status": "completed",
        "winner_candidate_id": state.winner_candidate_id,
    }
    assert state.status == "completed"
    assert state.completed_samples == 1_500
    assert state.progress == 1
    assert len(batch_writes) == 29
    assert not any(
        item["candidate_id"] == existing_batch.candidate_id
        and item["batch_index"] == existing_batch.batch_index
        for item in batch_writes
    )
    assert (existing_batch.candidate_id, existing_batch.batch_index) not in simulated_batches
    assert len(simulated_batches) == 29
    assert {item["event_type"] for item in events} >= {
        "started",
        "snapshot_loaded",
        "batch_completed",
        "ranking_completed",
        "critic_started",
        "critic_completed",
        "completed",
    }

    assert len(critic_inputs) == 1
    numeric_winner = critic_inputs[0].top_three[0]
    attempted_llm_winner = critic_inputs[0].top_three[-1]
    assert state.winner_candidate_id == numeric_winner
    assert state.winner_candidate_id != attempted_llm_winner

    assert len(published) == 1
    final_decision = published[0][0]
    assert final_decision.status == "ready_for_approval"
    assert final_decision.ranked_actions[0].action_id == numeric_winner
    final_candidate_rows = {
        item["action_id"]: item
        for item in candidate_writes
        if item.get("rank") is not None
    }
    assert len(final_candidate_rows) == 3
    for action in final_decision.ranked_actions:
        per_day = final_candidate_rows[action.action_id]["distribution_summary"][
            "perDay"
        ]
        assert action.expected_effect.contribution_profit_low_per_day == pytest.approx(
            per_day["p10"]
        )
        assert action.expected_effect.contribution_profit_base_per_day == pytest.approx(
            per_day["p50"]
        )
        assert action.expected_effect.contribution_profit_high_per_day == pytest.approx(
            per_day["p90"]
        )
        assert f"decision-simulation-run:{initial.id}" in (
            action.expected_effect.evidence_refs
        )
        assert f"simulation_run_id={initial.id}" in (
            action.expected_effect.calculation_method
        )
    assert decision_links[-1]["status"] == "ready_for_approval"
    block.assert_not_called()


@pytest.mark.parametrize("terminal_status", ["completed", "blocked", "failed", "superseded"])
def test_duplicate_terminal_worker_delivery_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    run = _run_record(run_id="terminal-run", status=terminal_status)
    monkeypatch.setattr(jobs, "get_decision_simulation_run", lambda _: run)
    lock = Mock(side_effect=AssertionError("terminal job acquired a database lock"))
    monkeypatch.setattr(jobs, "_simulation_run_lock", lock)

    outcome = jobs.run_decision_simulation_job(run.id)

    assert outcome == {
        "simulation_run_id": run.id,
        "status": terminal_status,
        "duplicate": True,
    }
    lock.assert_not_called()


def test_reconciler_requeues_stale_runs_with_replace_stale_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    stale = replace(
        _run_record(run_id="stale-run", status="running"),
        heartbeat_at=current - timedelta(minutes=30),
        progress=0.4,
        completed_samples=600,
    )
    updates: list[dict] = []
    claims: list[dict] = []
    events: list[dict] = []
    executor = Mock()
    executor.enqueue.return_value = "decision-simulation-stale-run"
    monkeypatch.setattr(
        jobs,
        "get_settings",
        lambda: SimpleNamespace(decision_simulation_stale_after_seconds=1_200),
    )

    def list_stale(*, stale_before, limit):
        assert stale_before == current - timedelta(seconds=1_200)
        assert limit == 100
        return [stale]

    monkeypatch.setattr(jobs, "list_stale_decision_simulation_runs", list_stale)
    monkeypatch.setattr(
        jobs,
        "list_unprojected_terminal_decision_simulation_runs",
        lambda **_: [],
    )
    monkeypatch.setattr(
        jobs,
        "update_decision_simulation_run",
        lambda simulation_run_id, **kwargs: updates.append(
            {"simulation_run_id": simulation_run_id, **kwargs}
        ),
    )
    def claim_stale(simulation_run_id, **kwargs):
        claims.append({"simulation_run_id": simulation_run_id, **kwargs})
        return replace(
            stale,
            status=kwargs["claimed_status"],
            heartbeat_at=kwargs["heartbeat_at"],
        )

    monkeypatch.setattr(jobs, "claim_stale_decision_simulation_run", claim_stale)
    monkeypatch.setattr(
        jobs,
        "append_decision_simulation_event",
        lambda **kwargs: events.append(kwargs),
    )

    result = jobs.reconcile_decision_simulations(now=current, executor=executor)

    assert result == {"stale": 1, "recovered": [stale.id]}
    assert updates == []
    assert claims == [
        {
            "simulation_run_id": stale.id,
            "expected_status": "running",
            "stale_before": current - timedelta(seconds=1_200),
            "claimed_status": "queued",
            "heartbeat_at": current,
        }
    ]
    assert events[0]["event_type"] == "queued"
    assert events[0]["payload"]["recovered"] is True
    executor.enqueue.assert_called_once_with(stale.id, replace_stale=True)
