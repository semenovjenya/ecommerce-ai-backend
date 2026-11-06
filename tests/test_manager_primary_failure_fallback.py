from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from agents import OutputGuardrailTripwireTriggered

from agi_ecom_app.agent_queue import (
    _execute_manager_primary_failure_fallback,
    _execute_question,
    _persist_question_failure,
)
from agi_ecom_app.schemas import (
    EcomRunContext,
    MemoryMode,
    QueryIntent,
    QueryMode,
    RoundType,
    WorkflowVersion,
)


class PermissionDeniedError(Exception):
    pass


def test_manager_primary_guardrail_uses_deterministic_fallback() -> None:
    query = "Сформируй единый план действий и сравни варианты для SKU 46303."
    trace = _trace(query=query)
    context = _context(trace, query_mode=QueryMode.DECISION)
    question = _question()
    exc = OutputGuardrailTripwireTriggered(
        SimpleNamespace(
            guardrail=SimpleNamespace(),
            output=SimpleNamespace(output_info={"violations": ["finding has no evidence"]}),
            agent_output={"summary": "Частичный ответ", "findings": []},
        )
    )

    with (
        patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace),
        patch("agi_ecom_app.agent_queue._build_context_from_trace", return_value=context),
        patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=SimpleNamespace()),
        patch(
            "agi_ecom_app.agent_queue._round_payload_for_question",
            return_value={"round_type": "primary"},
        ),
        patch(
            "agi_ecom_app.agent_queue.get_settings",
            return_value=SimpleNamespace(manager_decision_enabled=True),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_tool_policy",
            return_value=SimpleNamespace(required_tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_agent_builder_for_intent",
            return_value=lambda: SimpleNamespace(tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue.build_round_prompt",
            return_value=SimpleNamespace(
                text="primary prompt",
                report=SimpleNamespace(model_dump=lambda **_: {}),
            ),
        ),
        patch("agi_ecom_app.agent_queue._run_agent", side_effect=exc),
        patch("agi_ecom_app.agent_queue._persist_question_failure") as persist_failure,
        patch(
            "agi_ecom_app.agent_queue._execute_manager_primary_failure_fallback"
        ) as fallback,
    ):
        _execute_question(question)

    persist_failure.assert_called_once()
    assert persist_failure.call_args.kwargs["blocking"] is False
    assert persist_failure.call_args.kwargs["transition_question"] is False
    fallback.assert_called_once()
    assert fallback.call_args.kwargs["exc"] is exc


def _trace(*, query: str, query_mode: str = "decision") -> SimpleNamespace:
    query_plan = {
        "query_mode": query_mode,
        "scope": "portfolio",
        "decision_kind": "action_portfolio",
        "targets": [],
        "requested_analyses": ["advertising"],
    }
    return SimpleNamespace(
        id=71,
        chat_id="manager-primary-fallback",
        client_message_id="client-71",
        intent="marketing",
        primary_agent="marketing_agent",
        status="running",
        summary={
            "user_query": query,
            "workflow_version": "v2_optimized",
            "prompt_version": "v2_optimized",
            "query_mode": query_mode,
            "intent_router": {"query_plan": query_plan},
        },
    )


def _context(trace: SimpleNamespace, *, query_mode: QueryMode) -> EcomRunContext:
    return EcomRunContext(
        user_query=trace.summary["user_query"],
        intent=QueryIntent.MARKETING,
        today=date(2026, 7, 16),
        db_target="test",
        chat_id=trace.chat_id,
        workflow_version=WorkflowVersion.V2_OPTIMIZED,
        round_type=RoundType.PRIMARY,
        query_mode=query_mode,
        memory_mode=MemoryMode.SUMMARY_RECENT,
        query_plan=dict(trace.summary["intent_router"]["query_plan"]),
    )


def _question() -> SimpleNamespace:
    return SimpleNamespace(
        id=81,
        run_id=71,
        round_id=91,
        source_agent="orchestrator",
        target_agent="marketing",
        question="primary",
        reason="primary",
        verification_goal={},
    )


def test_manager_primary_permission_failure_uses_deterministic_snapshot_and_completes() -> None:
    query = "Сформируй 3 действия и ранжируй действия по прибыли."
    trace = _trace(query=query)
    context = _context(trace, query_mode=QueryMode.DECISION)
    question = _question()
    snapshot = {
        "schemaVersion": "1.0",
        "retrievedAt": "2026-07-16T09:00:00+00:00",
        "sourceTables": ["marketing.wb_marketing_campaign_stats_daily"],
    }
    deterministic_candidate = {
        "candidateId": "pause-101",
        "family": "advertising",
        "target": "101",
    }
    build_result = SimpleNamespace(
        candidates=[deterministic_candidate],
        evidence_facts=[
            {
                "metric": "ad_spend",
                "value": 700,
                "unit": "RUB",
                "evidenceRef": "manager:advertising:campaign:101:ad_spend",
            }
        ],
        diagnostics=[],
    )
    decision_payload = {
        "schemaVersion": "1.0",
        "decisionId": "md-fallback-71",
        "status": "needs_data",
        "rankedActions": [],
    }
    decision = SimpleNamespace(
        decision_id="md-fallback-71",
        status="needs_data",
        schema_version="1.0",
        ranked_actions=[],
        model_dump=Mock(return_value=decision_payload),
    )
    workflow_result = SimpleNamespace(
        decision=decision,
        rendered_markdown="# Недостаточно данных для решения",
        facts_only=True,
    )
    assistant_turn = SimpleNamespace(assistant_message_seq=111)

    with ExitStack() as stack:
        stack.enter_context(
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace)
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._build_context_from_trace", return_value=context)
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=SimpleNamespace())
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._round_payload_for_question",
                return_value={"round_type": "primary"},
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.get_settings",
                return_value=SimpleNamespace(manager_decision_enabled=True),
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._domain_tool_policy",
                return_value=SimpleNamespace(required_tools=[]),
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._domain_agent_builder_for_intent",
                return_value=lambda: SimpleNamespace(tools=[]),
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.build_round_prompt",
                return_value=SimpleNamespace(
                    text="primary prompt",
                    report=SimpleNamespace(model_dump=lambda **_: {"input_chars": 14}),
                ),
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._run_agent",
                side_effect=PermissionDeniedError("project is not permitted"),
            )
        )
        persist_failure = stack.enter_context(
            patch("agi_ecom_app.agent_queue._persist_question_failure")
        )
        load_snapshot = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
                return_value=snapshot,
            )
        )
        build_candidates = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.build_manager_action_candidates",
                return_value=build_result,
            )
        )
        manager_workflow = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue.run_manager_decision_workflow",
                return_value=workflow_result,
            )
        )
        maybe_schedule = stack.enter_context(
            patch(
                "agi_ecom_app.decision_simulation.service.maybe_schedule_decision_simulation",
                return_value=workflow_result,
            )
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.decision_simulation.service.manager_decision_experiments_deferred",
                return_value=True,
            )
        )
        persist_fallback_response = stack.enter_context(
            patch("agi_ecom_app.agent_queue._persist_agent_run_response_safely")
        )
        stack.enter_context(
            patch("agi_ecom_app.agent_queue._trace_metadata", return_value={})
        )
        persist_turn = stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._persist_assistant_turn",
                return_value=assistant_turn,
            )
        )
        persist_decision = stack.enter_context(
            patch("agi_ecom_app.agent_queue.persist_manager_decision_bundle")
        )
        stack.enter_context(
            patch(
                "agi_ecom_app.agent_queue._agent_run_trace_summary",
                return_value={"primary_agent": "marketing_agent"},
            )
        )
        complete_trace = stack.enter_context(
            patch("agi_ecom_app.agent_queue.complete_agent_run_trace")
        )
        execute_action = stack.enter_context(
            patch("agi_ecom_app.agent_queue.execute_action_handler")
        )
        persist_action_tasks = stack.enter_context(
            patch("agi_ecom_app.agent_queue._persist_agent_action_tasks_safely")
        )

        _execute_question(question)

    persist_failure.assert_called_once()
    assert persist_failure.call_args.kwargs["blocking"] is False
    assert persist_failure.call_args.kwargs["transition_question"] is False
    load_snapshot.assert_called_once_with()
    build_candidates.assert_called_once_with(snapshot, requested_targets=[])
    manager_package = manager_workflow.call_args.args[0]
    assert manager_package["manager_action_candidates"] == [deterministic_candidate]
    assert manager_package["manager_candidate_evidence"] == build_result.evidence_facts
    assert manager_package["confirmed_facts"] == []
    assert manager_package["runtime_failures"][0]["error_type"] == "PermissionDeniedError"
    maybe_schedule.assert_called_once_with(
        workflow_result,
        chat_id=trace.chat_id,
        run_id=trace.id,
        source_key="agent-run:71",
    )
    fallback_payload = persist_fallback_response.call_args.kwargs["payload"]
    assert fallback_payload["manager_decision"] == decision_payload
    assert fallback_payload["_manager_primary_fallback"] is True
    metadata = persist_turn.call_args.kwargs["metadata"]
    assert metadata["manager_primary_fallback"] == {
        "used": True,
        "source": "deterministic_manager_candidate_snapshot",
        "primary_llm_error_type": "PermissionDeniedError",
        "actions_executed": False,
    }
    assert "action_tasks" not in metadata
    assert persist_decision.call_args.kwargs["create_experiments"] is False
    assert complete_trace.call_args.kwargs["status"] == "completed"
    assert complete_trace.call_args.kwargs["summary"]["manager_decision_status"] == "needs_data"
    assert complete_trace.call_args.kwargs["question_id"] == question.id
    assert complete_trace.call_args.kwargs["round_id"] == question.round_id
    execute_action.assert_not_called()
    persist_action_tasks.assert_not_called()


def test_manager_primary_fallback_with_no_candidates_is_needs_data_and_facts_only() -> None:
    query = "Сформируй 3 действия и ранжируй действия по прибыли."
    trace = _trace(query=query)
    context = _context(trace, query_mode=QueryMode.DECISION)
    question = _question()
    assistant_turn = SimpleNamespace(assistant_message_seq=112)
    empty_build = SimpleNamespace(candidates=[], evidence_facts=[], diagnostics=[])

    with (
        patch(
            "agi_ecom_app.agent_queue.load_manager_candidate_snapshot",
            return_value={
                "schemaVersion": "1.0",
                "retrievedAt": "2026-07-16T09:00:00+00:00",
                "sourceTables": ["marketing.wb_marketing_campaign_stats_daily"],
            },
        ),
        patch(
            "agi_ecom_app.agent_queue.build_manager_action_candidates",
            return_value=empty_build,
        ),
        patch(
            "agi_ecom_app.decision_simulation.service.maybe_schedule_decision_simulation",
            side_effect=lambda result, **_: result,
        ),
        patch(
            "agi_ecom_app.decision_simulation.service.manager_decision_experiments_deferred",
            return_value=False,
        ),
        patch("agi_ecom_app.agent_queue._persist_agent_run_response_safely"),
        patch("agi_ecom_app.agent_queue._trace_metadata", return_value={}),
        patch(
            "agi_ecom_app.agent_queue._persist_assistant_turn",
            return_value=assistant_turn,
        ) as persist_turn,
        patch("agi_ecom_app.agent_queue.persist_manager_decision_bundle") as persist_decision,
        patch(
            "agi_ecom_app.agent_queue._agent_run_trace_summary",
            return_value={"primary_agent": "marketing_agent"},
        ),
        patch("agi_ecom_app.agent_queue.complete_agent_run_trace") as complete_trace,
        patch("agi_ecom_app.agent_queue.execute_action_handler") as execute_action,
    ):
        _execute_manager_primary_failure_fallback(
            trace=trace,
            question=question,
            context=context,
            intent=QueryIntent.MARKETING,
            user_query=query,
            exc=PermissionDeniedError("project is not permitted"),
        )

    decision = persist_decision.call_args.args[0]
    assert decision.status == "needs_data"
    assert decision.ranked_actions == []
    assert any(blocker.code == "insufficient_action_portfolio" for blocker in decision.blockers)
    assert persist_turn.call_args.kwargs["output_text"].startswith(
        "# Недостаточно данных для решения"
    )
    assert complete_trace.call_args.kwargs["status"] == "completed"
    assert complete_trace.call_args.kwargs["question_id"] == question.id
    assert complete_trace.call_args.kwargs["round_id"] == question.round_id
    execute_action.assert_not_called()


def test_primary_failure_diagnostic_can_leave_question_running_for_fallback() -> None:
    question = _question()
    response = SimpleNamespace(id=901)

    with (
        patch("agi_ecom_app.agent_queue.mark_agent_run_question_failed") as mark_failed,
        patch(
            "agi_ecom_app.agent_queue.save_agent_run_response",
            return_value=response,
        ) as save_response,
        patch("agi_ecom_app.agent_queue.save_agent_run_gaps") as save_gaps,
    ):
        _persist_question_failure(
            question=question,
            exc=PermissionDeniedError("project is not permitted"),
            blocking=False,
            transition_question=False,
        )

    mark_failed.assert_not_called()
    assert save_response.call_args.kwargs["status"] == "failed"
    gap = save_gaps.call_args.kwargs["gaps"][0]
    assert gap["blocking"] is False


def test_manager_fallback_failure_propagates_to_job_level_handler() -> None:
    query = "Сформируй 3 действия и ранжируй действия по прибыли."
    trace = _trace(query=query)
    context = _context(trace, query_mode=QueryMode.DECISION)
    question = _question()

    with (
        patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace),
        patch("agi_ecom_app.agent_queue._build_context_from_trace", return_value=context),
        patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=SimpleNamespace()),
        patch(
            "agi_ecom_app.agent_queue._round_payload_for_question",
            return_value={"round_type": "primary"},
        ),
        patch(
            "agi_ecom_app.agent_queue.get_settings",
            return_value=SimpleNamespace(manager_decision_enabled=True),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_tool_policy",
            return_value=SimpleNamespace(required_tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_agent_builder_for_intent",
            return_value=lambda: SimpleNamespace(tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue.build_round_prompt",
            return_value=SimpleNamespace(
                text="primary prompt",
                report=SimpleNamespace(model_dump=lambda **_: {}),
            ),
        ),
        patch(
            "agi_ecom_app.agent_queue._run_agent",
            side_effect=PermissionDeniedError("project is not permitted"),
        ),
        patch("agi_ecom_app.agent_queue._persist_question_failure") as persist_failure,
        patch(
            "agi_ecom_app.agent_queue._execute_manager_primary_failure_fallback",
            side_effect=RuntimeError("fallback persistence failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="fallback persistence failed"):
            _execute_question(question)

    persist_failure.assert_called_once()
    assert persist_failure.call_args.kwargs["transition_question"] is False


def test_non_manager_primary_permission_failure_keeps_existing_failure_behavior() -> None:
    query = "Почему снизились продажи?"
    trace = _trace(query=query, query_mode="diagnostic")
    context = _context(trace, query_mode=QueryMode.DIAGNOSTIC)
    question = _question()

    with (
        patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace),
        patch("agi_ecom_app.agent_queue._build_context_from_trace", return_value=context),
        patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=SimpleNamespace()),
        patch(
            "agi_ecom_app.agent_queue._round_payload_for_question",
            return_value={"round_type": "primary"},
        ),
        patch(
            "agi_ecom_app.agent_queue.get_settings",
            return_value=SimpleNamespace(manager_decision_enabled=True),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_tool_policy",
            return_value=SimpleNamespace(required_tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue._domain_agent_builder_for_intent",
            return_value=lambda: SimpleNamespace(tools=[]),
        ),
        patch(
            "agi_ecom_app.agent_queue.build_round_prompt",
            return_value=SimpleNamespace(
                text="primary prompt",
                report=SimpleNamespace(model_dump=lambda **_: {}),
            ),
        ),
        patch(
            "agi_ecom_app.agent_queue._run_agent",
            side_effect=PermissionDeniedError("project is not permitted"),
        ),
        patch("agi_ecom_app.agent_queue._persist_question_failure") as persist_failure,
        patch("agi_ecom_app.agent_queue.load_manager_candidate_snapshot") as load_snapshot,
        patch("agi_ecom_app.agent_queue._persist_assistant_turn") as persist_turn,
    ):
        _execute_question(question)

    persist_failure.assert_called_once()
    assert "blocking" not in persist_failure.call_args.kwargs
    load_snapshot.assert_not_called()
    persist_turn.assert_not_called()
