from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from agi_ecom_app.experiment_program import simulation_adapter as adapter
from agi_ecom_app.services.memory import (
    DecisionSimulationCandidateRecord,
    DecisionSimulationRunRecord,
)


NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _run() -> DecisionSimulationRunRecord:
    return DecisionSimulationRunRecord(
        id="run-1",
        decision_id="decision-1",
        chat_id=None,
        run_id=None,
        mode="shadow",
        status="completed",
        objective="incremental_contribution_profit",
        model_version="pricing-elasticity-bootstrap-v1",
        evidence_snapshot={},
        evidence_hash="a" * 64,
        master_seed=1,
        sample_count=500,
        batch_size=50,
        candidate_count=2,
        completed_samples=1000,
        total_samples=1000,
        completed_candidates=2,
        progress=1,
        provisional_winner_candidate_id="price-1",
        winner_candidate_id="price-1",
        critic_result={},
        error={},
        idempotency_key=None,
        retry_of_run_id=None,
        superseded_by_run_id=None,
        retry_attempt=0,
        heartbeat_at=NOW,
        started_at=NOW,
        completed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _candidate(candidate_id: str, price: float) -> DecisionSimulationCandidateRecord:
    return DecisionSimulationCandidateRecord(
        simulation_run_id="run-1",
        candidate_id=candidate_id,
        action_id=candidate_id,
        status="completed",
        candidate_payload={
            "candidateId": candidate_id,
            "operation": "set_price",
            "skuId": 123,
            "priceRub": price,
        },
        evidence_refs=[],
        constraint_violations=[],
        distribution_summary={},
        score_breakdown={},
        expected_effect_mean=1,
        profit_p10=1,
        cvar_10_per_day=1,
        execution_risk=0,
        utility_score=1,
        rank=1,
        is_winner=candidate_id == "price-1",
        completed_samples=500,
        error={},
        completed_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def test_adapter_builds_five_domain_program_and_links_every_candidate(monkeypatch) -> None:
    run = _run()
    candidates = [_candidate("price-1", 5310), _candidate("price-2", 5450)]
    monkeypatch.setattr(adapter, "get_decision_simulation_run", lambda _: run)
    monkeypatch.setattr(adapter, "list_decision_simulation_candidates", lambda _: candidates)
    monkeypatch.setattr(adapter, "ensure_builtin_simulation_plugins_registered", lambda: ())
    monkeypatch.setattr(
        adapter,
        "get_decision_simulation_plugin",
        lambda _: SimpleNamespace(family="pricing", operation="set_price"),
    )
    captured: dict = {}

    def create(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return (
            SimpleNamespace(
                program=SimpleNamespace(id="program-1"),
                candidates=[
                    SimpleNamespace(candidate_key="price-1", id="program-candidate-1"),
                    SimpleNamespace(candidate_key="price-2", id="program-candidate-2"),
                ],
            ),
            True,
        )

    monkeypatch.setattr(adapter, "create_experiment_program", create)
    link = Mock(return_value=2)
    monkeypatch.setattr(adapter, "link_simulation_run", link)

    program_id = adapter.attach_experiment_program_to_simulation_run("run-1")

    assert program_id == "program-1"
    assert captured["request"].domains == [
        "pricing",
        "advertising",
        "supply",
        "sales",
        "finance",
    ]
    assert captured["request"].target == {"nmId": 123}
    assert {item.candidate_key for item in captured["request"].candidates} == {
        "price-1",
        "price-2",
    }
    link.assert_called_once_with(
        program_id="program-1",
        simulation_run_id="run-1",
        candidate_links={
            "price-1": "program-candidate-1",
            "price-2": "program-candidate-2",
        },
    )
