from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agi_ecom_app.agent_queue import (
    _attach_deterministic_manager_candidates,
    _execute_synthesis_question,
)
from agi_ecom_app.manager_decision import run_manager_decision_workflow
from agi_ecom_app.schemas import (
    EcomRunContext,
    QueryIntent,
    QueryMode,
    RoundType,
    WorkflowVersion,
)


def test_manager_attachment_discards_untrusted_synthesis_candidates() -> None:
    generated = {"actionId": "deterministic", "family": "advertising"}
    result = SimpleNamespace(candidates=[generated], evidence_facts=[], diagnostics=[])
    snapshot = {"schemaVersion": "1.0", "retrievedAt": "2026-07-14T08:00:00Z"}

    with patch(
        "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
        return_value=snapshot,
    ), patch(
        "agi_ecom_app.agent_queue.build_manager_action_candidates",
        return_value=result,
    ):
        package = _attach_deterministic_manager_candidates(
            {
                "manager_action_candidates": [
                    {"actionId": "llm-invented", "economics": {"fake": True}}
                ]
            },
            run_id=77,
        )

    assert package["manager_action_candidates"] == [generated]
    assert package["manager_candidate_diagnostics"] == [
        {"code": "untrusted_manager_candidates_ignored", "count": 1}
    ]


def test_manager_attachment_fails_closed_when_source_is_unavailable() -> None:
    with patch(
        "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
        side_effect=RuntimeError("database unavailable"),
    ):
        package = _attach_deterministic_manager_candidates(
            {"blocking_gaps": []},
            run_id=78,
        )

    assert package["blocking_gaps"] == [
        {
            "gap_id": "manager-candidate-source-unavailable",
            "status": "open",
            "metric": "manager_candidate_snapshot",
            "reason": "Не удалось прочитать обязательный снимок данных для расчёта действий.",
            "severity": "blocked",
        }
    ]
    assert package["manager_candidate_diagnostics"] == [
        {"code": "candidate_source_error", "error_type": "RuntimeError"}
    ]


def test_manager_attachment_adds_shadow_advisory_for_explicit_scenario_request() -> None:
    result = SimpleNamespace(candidates=[], evidence_facts=[], diagnostics=[])
    snapshot = {"schemaVersion": "1.0", "retrievedAt": "2026-07-20T08:00:00Z"}
    advisory = {
        "schemaVersion": "manager-advisory-v1",
        "mode": "shadow",
        "requestedTargets": ["46303"],
        "forecasts": [{"target": "46303"}],
        "priceSimulations": [{"target": "46303"}],
        "errors": [],
    }
    query_plan = {
        "query_mode": "decision",
        "scope": "single_sku",
        "decision_kind": "single_action",
        "targets": [{"target_type": "vendor_code", "value": "46303"}],
        "requested_analyses": ["price_scenarios", "forecast"],
    }

    with (
        patch(
            "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
            return_value=snapshot,
        ),
        patch(
            "agi_ecom_app.agent_queue.build_manager_action_candidates",
            return_value=result,
        ) as build_candidates,
        patch(
            "agi_ecom_app.agent_queue.should_build_manager_advisory",
            return_value=True,
        ),
        patch(
            "agi_ecom_app.agent_queue.build_manager_advisory",
            return_value=advisory,
        ) as build_advisory,
    ):
            package = _attach_deterministic_manager_candidates(
                {},
                run_id=79,
                user_query="Для SKU 46303 сравни варианты цены",
                query_plan=query_plan,
            )

    assert package["manager_advisory"] == advisory
    assert build_candidates.call_args.kwargs["requested_targets"] == ["46303"]
    build_advisory.assert_called_once_with(
        "Для SKU 46303 сравни варианты цены",
        query_plan=query_plan,
        candidate_snapshot=snapshot,
    )


def test_portfolio_focus_targets_do_not_filter_manager_candidate_universe() -> None:
    result = SimpleNamespace(candidates=[], evidence_facts=[], diagnostics=[])
    snapshot = {"schemaVersion": "1.0", "retrievedAt": "2026-07-21T08:00:00Z"}
    query_plan = {
        "query_mode": "decision",
        "scope": "portfolio",
        "decision_kind": "action_portfolio",
        "targets": [
            {"target_type": "vendor_code", "value": "46227"},
            {"target_type": "vendor_code", "value": "46303"},
        ],
        "requested_analyses": ["advertising"],
    }

    with (
        patch(
            "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
            return_value=snapshot,
        ),
        patch(
            "agi_ecom_app.agent_queue.build_manager_action_candidates",
            return_value=result,
        ) as build_candidates,
    ):
        package = _attach_deterministic_manager_candidates(
            {},
            run_id=80,
            query_plan=query_plan,
        )

    build_candidates.assert_called_once_with(snapshot, requested_targets=[])
    assert package["query_plan"] == query_plan


@pytest.mark.parametrize(
    ("user_query", "uses_manager_core"),
    [
        ("Составь план действий", True),
        (
            "Нужно ли дозаказать SKU 46227? Проверь остаток, скорость продаж, "
            "покрытие и спрос до рекомендации.",
            False,
        ),
    ],
)
def test_v2_decision_uses_manager_core_only_for_portfolio_requests(
    user_query: str,
    uses_manager_core: bool,
) -> None:
    query_plan = {
        "query_mode": "decision",
        "scope": "portfolio" if uses_manager_core else "single_sku",
        "decision_kind": "action_portfolio" if uses_manager_core else "single_action",
        "targets": (
            []
            if uses_manager_core
            else [{"target_type": "vendor_code", "value": "46227"}]
        ),
        "requested_analyses": ["inventory_matrix"],
    }
    trace = SimpleNamespace(
        id=77,
        chat_id="manager-chat",
        client_message_id="client-77",
        primary_agent="sales_agent",
        summary={
            "workflow_version": "v2_optimized",
            "prompt_version": "v2_optimized",
            "query_mode": "decision",
            "intent_router": {"query_plan": query_plan},
        },
    )
    question = SimpleNamespace(id=8, round_id=9, verification_goal={})
    context = EcomRunContext(
        user_query=user_query,
        intent=QueryIntent.SALES,
        today=date(2026, 7, 13),
        db_target="test",
        chat_id="manager-chat",
        workflow_version=WorkflowVersion.V2_OPTIMIZED,
        round_type=RoundType.SYNTHESIS,
        query_mode=QueryMode.DECISION,
        query_plan=query_plan,
    )
    package = {
        "confirmed_facts": [],
        "blocking_gaps": [],
        "accepted_limitations": [],
        "non_blocking_gaps": [],
        "resolved_gaps": [],
        "sku_evidence_matrix": [],
        "domain_coverage": {},
        "evidence_quality": {},
        "runtime_failures": [],
    }
    persisted = {
        "primary_output": {"summary": "facts"},
        "round_1_evidence": [],
        "followup_evidence": [],
    }
    compact_report = SimpleNamespace(model_dump=lambda **_: {"total_chars": 10})
    turn = SimpleNamespace(assistant_message_seq=91)
    generated_candidate = {
        "candidateId": "deterministic-pause-campaign-42",
        "family": "advertising",
        "target": "42",
        "targetType": "campaign",
        "targetRefs": ["sku:100"],
        "operation": "pause_campaign",
        "executionMode": "wb_api",
        "executionPayload": {"advert_id": 42},
        "from": {"value": 9, "unit": "wb_status_code"},
        "to": {"value": 11, "unit": "wb_status_code"},
        "candidateBlockers": [
            {
                "code": "missing_campaign_contribution",
                "message": "Нет подтвержденной маржинальной прибыли кампании.",
                "severity": "needs_data",
            }
        ],
    }
    candidate_snapshot = {
        "schemaVersion": "manager-candidate-source-v1",
        "advertisingRows": [{"campaign_id": 42, "nm_id": 100}],
        "supplyRows": [],
    }
    candidate_build_result = SimpleNamespace(
        candidates=[generated_candidate],
        evidence_facts=[],
        diagnostics=[{"code": "missing_campaign_contribution"}],
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace)
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._load_persisted_evidence_safely", return_value=persisted)
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._open_blocking_gap_payloads_safely", return_value=[])
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._build_synthesis_package", return_value=package)
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.compact_synthesis_package",
                return_value=(package, compact_report),
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.get_settings",
                return_value=SimpleNamespace(manager_decision_enabled=True),
            )
        )
        load_candidate_snapshot = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
                return_value=candidate_snapshot,
                create=True,
            )
        )
        build_candidates = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.build_manager_action_candidates",
                return_value=candidate_build_result,
                create=True,
            )
        )
        manager_workflow = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.run_manager_decision_workflow",
                wraps=run_manager_decision_workflow,
            )
        )
        execute_action = stack.enter_context(
            patch("agi_ecom_app.agent_queue.execute_action_handler")
        )
        run_llm = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._run_agent",
                return_value="# Решение\n\nСрочный дозаказ не нужен.",
            )
        )
        stack.enter_context(
            patch("agi_ecom_app.runner.validate_final_diagnostic_markdown", return_value=[])
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._sku_matrix_quality_violations", return_value=[])
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._build_sku_evidence_matrix",
                return_value={"called_domains": []},
            )
        )
        for target in (
            "_persist_agent_run_response_safely",
            "update_agent_run_question_status",
            "update_agent_run_round_status",
            "complete_agent_run_trace",
        ):
            stack.enter_context(patch(f"agi_ecom_app.agent_queue.{target}"))
        stack.enter_context(patch("agi_ecom_app.agent_queue._trace_metadata", return_value={}))
        stack.enter_context(patch("agi_ecom_app.agent_queue._agent_names_for_round", return_value=[]))
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._agent_names_for_followups", return_value=[])
        )
        persist_turn = stack.enter_context(
            patch("agi_ecom_app.agent_queue._persist_assistant_turn", return_value=turn)
        )
        persist_decision = stack.enter_context(
            patch("agi_ecom_app.agent_queue.persist_manager_decision_bundle")
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._agent_run_trace_summary",
                return_value={"primary_agent": "sales_agent"},
            )
        )
        _execute_synthesis_question(
            trace_id=77,
            question=question,
            context=context,
            intent=QueryIntent.SALES,
            user_query=user_query,
        )

    metadata = persist_turn.call_args.kwargs["metadata"]
    if uses_manager_core:
        run_llm.assert_not_called()
        load_candidate_snapshot.assert_called_once_with()
        build_candidates.assert_called_once()
        manager_workflow.assert_called_once()
        manager_package = manager_workflow.call_args.args[0]
        assert manager_package["manager_action_candidates"] == [generated_candidate]
        decision = persist_decision.call_args.args[0]
        assert decision.status == "needs_data"
        assert decision.ranked_actions == []
        assert [item.candidate_id for item in decision.rejected_candidates] == [
            "deterministic-pause-campaign-42"
        ]
        assert metadata["manager_decision"]["status"] == "needs_data"
    else:
        assert run_llm.called
        load_candidate_snapshot.assert_not_called()
        build_candidates.assert_not_called()
        manager_workflow.assert_not_called()
        persist_decision.assert_not_called()
        assert "manager_decision" not in metadata
    execute_action.assert_not_called()
    assert "action_tasks" not in metadata
