from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agi_ecom_app.experiment_program import lifecycle_adapter as adapter
from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.manager_lifecycle import service as lifecycle_service


class _ScriptedCursor:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.executed: list[tuple[str, list[object]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((" ".join(sql.split()).lower(), list(params or [])))

    def fetchone(self):
        return self._responses.pop(0)

    def fetchall(self):
        return self._responses.pop(0)


class _ScriptedConnection:
    def __init__(self, cursor: _ScriptedCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def cursor(self) -> _ScriptedCursor:
        return self._cursor


def _install_storage(monkeypatch, responses: list[object]) -> _ScriptedCursor:
    cursor = _ScriptedCursor(responses)
    connection = _ScriptedConnection(cursor)
    monkeypatch.setattr(adapter, "ensure_memory_schema", lambda: None)
    monkeypatch.setattr(adapter, "get_memory_connection", lambda: connection)
    monkeypatch.setattr(adapter, "_project_action_state", Mock())
    return cursor


def _action_row(**overrides):
    values = {
        "id": "action-experiment-1",
        "decision_id": "decision-1",
        "action_id": "price-action-1",
        "state": "proposed",
        "action_program_id": None,
        "action_candidate_id": None,
        "decision_program_id": "program-1",
        "decision_simulation_run_id": "simulation-1",
    }
    values.update(overrides)
    return values


def test_sync_links_exact_durable_simulation_candidate_and_journals_once(
    monkeypatch,
) -> None:
    cursor = _install_storage(
        monkeypatch,
        [
            _action_row(),
            {"current_epoch": 1},
            [{"candidate_id": "candidate-1"}],
            {"id": "action-experiment-1"},
        ],
    )

    result = adapter.sync_action_experiment_link(
        "action-experiment-1",
        expected_program_id="program-1",
    )

    assert result.status == "linked"
    assert result.changed is True
    assert result.candidate_id == "candidate-1"
    sql = [statement for statement, _ in cursor.executed]
    assert any(statement.startswith("update agent_action_experiments") for statement in sql)
    assert any("insert into experiment_program_events" in statement for statement in sql)
    assert any("insert into experiment_program_outbox" in statement for statement in sql)
    event_params = next(
        params
        for statement, params in cursor.executed
        if "insert into experiment_program_events" in statement
    )
    assert event_params[:3] == [
        "program-1:action-experiment:action-experiment-1:linked",
        "program-1",
        1,
    ]


def test_sync_replay_does_not_rewrite_an_exact_existing_link(monkeypatch) -> None:
    cursor = _install_storage(
        monkeypatch,
        [
            _action_row(
                action_program_id="program-1",
                action_candidate_id="candidate-1",
            ),
            {"current_epoch": 1},
            {"id": "candidate-1"},
        ],
    )

    result = adapter.sync_action_experiment_link(
        "action-experiment-1",
        expected_program_id="program-1",
    )

    assert result.status == "already_linked"
    assert result.changed is False
    assert not any(
        statement.startswith("update agent_action_experiments")
        for statement, _ in cursor.executed
    )
    assert sum(
        "insert into experiment_program_events" in statement
        for statement, _ in cursor.executed
    ) == 1


def test_sync_fails_closed_on_ambiguous_candidate_mapping(monkeypatch) -> None:
    cursor = _install_storage(
        monkeypatch,
        [
            _action_row(),
            {"current_epoch": 1},
            [{"candidate_id": "candidate-1"}, {"candidate_id": "candidate-2"}],
        ],
    )

    with pytest.raises(
        adapter.ExperimentProgramActionLinkConflict,
        match="multiple experiment-program candidates",
    ):
        adapter.sync_action_experiment_link("action-experiment-1")

    assert not any(
        statement.startswith("update agent_action_experiments")
        for statement, _ in cursor.executed
    )
    assert not any(
        "insert into experiment_program_events" in statement
        for statement, _ in cursor.executed
    )


def test_sync_uses_only_exact_candidate_key_for_non_simulation_program(
    monkeypatch,
) -> None:
    cursor = _install_storage(
        monkeypatch,
        [
            _action_row(decision_simulation_run_id=None),
            {"current_epoch": 3},
            [],
            [{"candidate_id": "candidate-manual"}],
            {"id": "action-experiment-1"},
        ],
    )

    result = adapter.sync_action_experiment_link("action-experiment-1")

    assert result.candidate_id == "candidate-manual"
    fallback_params = next(
        params
        for statement, params in cursor.executed
        if "candidate_key = %s" in statement
    )
    assert fallback_params == ["program-1", 3, "price-action-1"]


def test_sync_repairs_partial_candidate_link_without_remapping(monkeypatch) -> None:
    cursor = _install_storage(
        monkeypatch,
        [
            _action_row(action_candidate_id="candidate-1"),
            {"current_epoch": 1},
            {"id": "candidate-1"},
            {"id": "action-experiment-1"},
        ],
    )

    result = adapter.sync_action_experiment_link("action-experiment-1")

    assert result.status == "linked"
    assert result.changed is True
    assert result.candidate_id == "candidate-1"
    assert not any(
        "from decision_simulation_candidates" in statement
        for statement, _ in cursor.executed
    )


def test_sync_is_a_noop_for_legacy_decision_without_program(monkeypatch) -> None:
    _install_storage(
        monkeypatch,
        [_action_row(decision_program_id=None, decision_simulation_run_id=None)],
    )

    result = adapter.sync_action_experiment_link("action-experiment-1")

    assert result.status == "legacy"
    assert result.program_id is None
    assert result.candidate_id is None
    assert result.changed is False


@pytest.mark.parametrize(
    (
        "current_program_status",
        "action_state",
        "close_outcome",
        "candidate_status",
        "program_status",
    ),
    [
        ("awaiting_approval", "approved", None, "selected", "executing"),
        ("executing", "measuring", None, "running", "measuring"),
        ("measuring", "measured", None, "measured", "learning"),
        ("learning", "closed", "inconclusive", "inconclusive", "learning"),
        ("awaiting_approval", "rejected", None, "rejected", "replanning"),
    ],
)
def test_action_states_have_explicit_program_projection(
    current_program_status,
    action_state,
    close_outcome,
    candidate_status,
    program_status,
) -> None:
    assert adapter._candidate_status_for_action(
        action_state=action_state,
        close_outcome=close_outcome,
    ) == candidate_status
    assert adapter._program_status_for_action(
        current_status=current_program_status,
        action_state=action_state,
    ) == program_status


def test_action_state_projection_updates_candidate_program_and_journal() -> None:
    cursor = _ScriptedCursor(
        [
            None,
            {"id": "candidate-1"},
            {"status": "learning", "version": 5},
        ]
    )

    adapter._project_action_state(
        cursor,
        experiment_id="action-experiment-1",
        program_id="program-1",
        candidate_id="candidate-1",
        epoch=1,
        action_state="measured",
        action_version=4,
        close_outcome=None,
        current_program_status="measuring",
    )

    sql = [statement for statement, _params in cursor.executed]
    assert any(statement.startswith("update experiment_candidates") for statement in sql)
    assert any(statement.startswith("update experiment_programs") for statement in sql)
    assert any("action_experiment_state_changed" in statement for statement in sql)
    assert any("insert into experiment_program_outbox" in statement for statement in sql)


def _decision_payload() -> dict:
    return {
        "decision_id": "decision-1",
        "objective": "incremental_contribution_profit",
        "status": "ready_for_approval",
        "ranked_actions": [
            {
                "action_id": "price-action-1",
                "family": "pricing",
                "execution_mode": "human_task",
                "target": {"nmId": 123},
                "guardrails": {},
            }
        ],
    }


def test_persisted_program_action_is_linked_and_returned_refreshed(monkeypatch) -> None:
    decision = SimpleNamespace(
        id="decision-1",
        experiment_program_id="program-1",
        decision_json={},
    )
    stored = SimpleNamespace(
        id="action-experiment-1",
        experiment_program_id=None,
        experiment_candidate_id=None,
    )
    linked = SimpleNamespace(
        id="action-experiment-1",
        experiment_program_id="program-1",
        experiment_candidate_id="candidate-1",
    )
    monkeypatch.setattr(lifecycle_service, "upsert_manager_decision", lambda **_: decision)
    monkeypatch.setattr(
        lifecycle_service,
        "upsert_agent_action_experiment",
        lambda **_: stored,
    )
    monkeypatch.setattr(
        lifecycle_service,
        "get_agent_action_experiment",
        lambda _: linked,
    )
    sync = Mock()
    monkeypatch.setattr(adapter, "sync_action_experiment_link", sync)

    _, experiments = lifecycle_service.persist_manager_decision_bundle(
        _decision_payload()
    )

    assert experiments == [linked]
    sync.assert_called_once_with(
        "action-experiment-1",
        expected_program_id="program-1",
    )


def test_legacy_persistence_does_not_invoke_program_adapter(monkeypatch) -> None:
    decision = SimpleNamespace(
        id="decision-1",
        experiment_program_id=None,
        decision_json={},
    )
    stored = SimpleNamespace(
        id="action-experiment-1",
        experiment_program_id=None,
        experiment_candidate_id=None,
    )
    monkeypatch.setattr(lifecycle_service, "upsert_manager_decision", lambda **_: decision)
    monkeypatch.setattr(
        lifecycle_service,
        "upsert_agent_action_experiment",
        lambda **_: stored,
    )
    sync = Mock()
    monkeypatch.setattr(adapter, "sync_action_experiment_link", sync)

    _, experiments = lifecycle_service.persist_manager_decision_bundle(
        _decision_payload()
    )

    assert experiments == [stored]
    sync.assert_not_called()


def test_approval_repairs_program_link_before_lifecycle_claim(monkeypatch) -> None:
    unlinked = SimpleNamespace(
        id="action-experiment-1",
        decision_id="decision-1",
        experiment_program_id=None,
        experiment_candidate_id=None,
        state="proposed",
        expires_at=None,
        execution_mode="human_task",
    )
    linked = SimpleNamespace(
        **{
            **vars(unlinked),
            "experiment_program_id": "program-1",
            "experiment_candidate_id": "candidate-1",
        }
    )
    approved = SimpleNamespace(**{**vars(linked), "state": "approved"})
    decision = SimpleNamespace(
        id="decision-1",
        status="ready_for_approval",
        experiment_program_id="program-1",
        run_id=None,
        simulation_run_id=None,
    )
    reads = iter([unlinked, linked, approved])
    monkeypatch.setattr(
        lifecycle_service,
        "get_agent_action_experiment",
        lambda _: next(reads),
    )
    monkeypatch.setattr(lifecycle_service, "get_manager_decision", lambda _: decision)
    monkeypatch.setattr(
        lifecycle_service,
        "get_settings",
        lambda: SimpleNamespace(manager_decision_enabled=True),
    )
    monkeypatch.setattr(lifecycle_service, "_validate_exact_action_contract", lambda _: None)
    monkeypatch.setattr(
        lifecycle_service,
        "claim_agent_action_experiment_approval",
        lambda **_: (approved, True, False),
    )
    sync = Mock()
    monkeypatch.setattr(adapter, "sync_action_experiment_link", sync)

    result = lifecycle_service.approve_manager_action(
        experiment_id="action-experiment-1",
        actor=ManagerActor(user_id=7, username="manager", role="manager"),
        idempotency_key="approve-1",
    )

    assert result is approved
    assert sync.call_count == 2
    assert all(
        call.args == ("action-experiment-1",)
        and call.kwargs == {"expected_program_id": "program-1"}
        for call in sync.call_args_list
    )
