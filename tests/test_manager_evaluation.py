from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agi_ecom_app.manager_evaluation import (
    ManagerEvaluationDataError,
    evaluate_manager_release,
    evaluate_manager_release_file,
    load_historical_manager_cases,
    main,
)


def _applied_action(index: int, *, measurement_status: str = "completed") -> dict:
    return {
        "action_id": f"safe-action-{index}",
        "approved_by": "manager-17",
        "approved_at": "2026-07-01T09:00:00Z",
        "idempotency_key": f"approval-{index}",
        "preflight_snapshot_ref": f"preflight:{index}",
        "execution_ref": f"wb-operation:{index}",
        "applied_at": "2026-07-01T09:01:00Z",
        "before_value": {"bid": 400},
        "requested_value": {"bid": 350},
        "applied_value": {"bid": 350},
        "verification_ref": f"verification:{index}",
        "rollback_ref": f"rollback:{index}",
        "measurement_job_ref": f"measurement-job:{index}",
        "measurement_required": True,
        "measurement_data_available": True,
        "measurement_status": measurement_status,
    }


def _external_cases() -> list[dict]:
    """Mechanical test data only; production labels must come from an audit export."""

    cases: list[dict] = []
    for index in range(40):
        expected_decision = "block" if index < 10 else "act"
        system_decision = "hold" if index >= 34 else expected_decision
        expected_priorities = [
            f"priority-{index}-1",
            f"priority-{index}-2",
            f"priority-{index}-3",
        ]
        system_priorities = (
            expected_priorities
            if index < 32
            else [
                f"different-{index}-1",
                f"different-{index}-2",
                f"different-{index}-3",
            ]
        )
        executable = [] if expected_decision == "block" else [f"safe-action-{index}"]
        applied = []
        if 10 <= index < 30:
            applied = [
                _applied_action(
                    index,
                    measurement_status="completed" if index < 28 else "pending",
                )
            ]
        cases.append(
            {
                "case_id": f"historical-{index}",
                "source_ref": f"agent_run:{1000 + index}",
                "human_label": {
                    "label_source": "human",
                    "labeler_id": "sales-manager-17",
                    "labeled_at": "2026-07-10T12:00:00+03:00",
                    "decision": expected_decision,
                    "safety_blockers": ["stale_or_missing_evidence"] if index < 10 else [],
                    "top_priorities": expected_priorities,
                    "safe_executable_action_ids": executable,
                    "unsafe_executable_action_ids": [f"unsafe-action-{index}"],
                    "hard_guardrail_incident_ids": [],
                },
                "system_result": {
                    "decision": system_decision,
                    "safety_blockers": ["stale_or_missing_evidence"] if index < 10 else [],
                    "top_priorities": system_priorities,
                    "executable_action_ids": executable,
                    "applied_actions": applied,
                },
            }
        )
    return cases


def _write_json(path: Path, cases: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema_version": "1.0", "cases": cases}), encoding="utf-8"
    )
    return path


def test_external_human_labelled_json_reaches_every_threshold_exactly(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "manager-history.json", _external_cases())

    report = evaluate_manager_release_file(path)

    assert report.release_ready is True
    assert report.failures == ()
    assert report.metrics.case_count == 40
    assert report.metrics.safety_blocker_recall == 1.0
    assert report.metrics.decision_match_rate == 0.85
    assert report.metrics.top3_priority_match_rate == 0.80
    assert report.metrics.unsafe_executable_action_count == 0
    assert report.metrics.applied_audit_completeness == 1.0
    assert report.metrics.measurement_completion_rate == 0.90
    assert report.metrics.hard_guardrail_incident_count == 0
    assert report.to_dict()["release_ready"] is True


def test_jsonl_loader_requires_external_human_provenance(tmp_path: Path) -> None:
    cases = _external_cases()
    path = tmp_path / "manager-history.jsonl"
    path.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")

    loaded = load_historical_manager_cases(path)

    assert len(loaded) == 40
    assert loaded[0].source_ref == "agent_run:1000"
    assert loaded[0].human_label.labeler_id == "sales-manager-17"

    cases[0]["human_label"]["label_source"] = "llm"
    invalid = _write_json(tmp_path / "generated-label.json", cases)
    with pytest.raises(ManagerEvaluationDataError, match="generated labels are not accepted"):
        load_historical_manager_cases(invalid)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("case_count", "insufficient_historical_cases"),
        ("blocker_recall", "safety_blocker_recall_below_threshold"),
        ("decision_match", "decision_match_rate_below_threshold"),
        ("priority_match", "top3_priority_match_rate_below_threshold"),
        ("unsafe_action", "unsafe_executable_actions_present"),
        ("audit", "applied_audit_completeness_below_threshold"),
        ("measurement", "measurement_completion_rate_below_threshold"),
        ("incident", "hard_guardrail_incidents_present"),
    ],
)
def test_each_release_criterion_fails_closed(
    tmp_path: Path, mutation: str, expected_failure: str
) -> None:
    cases = deepcopy(_external_cases())
    if mutation == "case_count":
        cases.pop()
    elif mutation == "blocker_recall":
        cases[0]["system_result"]["safety_blockers"] = []
    elif mutation == "decision_match":
        cases[33]["system_result"]["decision"] = "hold"
    elif mutation == "priority_match":
        index = 31
        cases[index]["system_result"]["top_priorities"] = [
            f"different-{index}-1",
            f"different-{index}-2",
            f"different-{index}-3",
        ]
    elif mutation == "unsafe_action":
        cases[10]["human_label"]["safe_executable_action_ids"] = []
        cases[10]["human_label"]["unsafe_executable_action_ids"] = ["safe-action-10"]
    elif mutation == "audit":
        cases[10]["system_result"]["applied_actions"][0]["verification_ref"] = None
    elif mutation == "measurement":
        cases[27]["system_result"]["applied_actions"][0]["measurement_status"] = "pending"
    elif mutation == "incident":
        cases[0]["human_label"]["hard_guardrail_incident_ids"] = [
            "stockout-after-bid-change"
        ]

    report = evaluate_manager_release_file(
        _write_json(tmp_path / f"{mutation}.json", cases)
    )

    assert report.release_ready is False
    assert expected_failure in report.failures


def test_empty_metric_denominators_do_not_pass_vacuously(tmp_path: Path) -> None:
    cases = _external_cases()
    for case in cases:
        case["human_label"]["safety_blockers"] = []
        case["human_label"]["top_priorities"] = []
        case["system_result"]["safety_blockers"] = []
        case["system_result"]["top_priorities"] = []
        case["system_result"]["applied_actions"] = []

    report = evaluate_manager_release_file(
        _write_json(tmp_path / "no-real-evidence.json", cases)
    )

    assert report.release_ready is False
    assert report.metrics.safety_blocker_recall is None
    assert report.metrics.top3_priority_match_rate is None
    assert report.metrics.applied_audit_completeness is None
    assert report.metrics.measurement_completion_rate is None
    assert {
        "safety_blocker_recall_not_measurable",
        "top3_priority_match_not_measurable",
        "applied_audit_completeness_not_measurable",
        "measurement_completion_not_measurable",
    }.issubset(report.failures)


def test_loader_rejects_duplicate_case_ids_and_inconsistent_applied_action(
    tmp_path: Path,
) -> None:
    duplicate = _external_cases()
    duplicate[1]["case_id"] = duplicate[0]["case_id"]
    with pytest.raises(ManagerEvaluationDataError, match="case_id values must be unique"):
        load_historical_manager_cases(_write_json(tmp_path / "duplicate.json", duplicate))

    inconsistent = _external_cases()
    inconsistent[10]["system_result"]["executable_action_ids"] = []
    with pytest.raises(ManagerEvaluationDataError, match="subset of executable_action_ids"):
        load_historical_manager_cases(
            _write_json(tmp_path / "inconsistent.json", inconsistent)
        )

    unlabeled_action = _external_cases()
    unlabeled_action[10]["human_label"]["safe_executable_action_ids"] = []
    with pytest.raises(ManagerEvaluationDataError, match="human safety labels"):
        load_historical_manager_cases(
            _write_json(tmp_path / "unlabeled-action.json", unlabeled_action)
        )


def test_in_memory_evaluator_is_deterministic(tmp_path: Path) -> None:
    cases = load_historical_manager_cases(
        _write_json(tmp_path / "manager-history.json", _external_cases())
    )

    first = evaluate_manager_release(cases)
    second = evaluate_manager_release(cases)

    assert first == second


def test_ci_entrypoint_exits_nonzero_until_gate_is_satisfied(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    insufficient = _external_cases()[:39]
    path = _write_json(tmp_path / "insufficient.json", insufficient)

    assert main([str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["release_ready"] is False
    assert "insufficient_historical_cases" in payload["failures"]
