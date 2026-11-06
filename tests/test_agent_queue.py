import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents import OutputGuardrailTripwireTriggered

from agi_ecom_app.action_execution import ActionHandlerOutcome
from agi_ecom_app.agent_queue import (
    _compose_manager_answer_markdown,
    _deduplicate_actionable_gaps,
    _execute_question,
    _persist_question_failure,
    _persist_simple_final_answer,
    _portfolio_priority_quality_violations,
    _round_failure_reason,
    _resolve_terminal_action_output,
    _should_retry_question_failure,
    _synthesis_runtime_fallback_allowed,
    enqueue_action_execution_run,
    enqueue_agent_question,
    enqueue_initial_run,
)
from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.llm_routing import (
    DomainRouteDecision,
    QueryPlan,
    _repair_domain_route_decision,
)
from agi_ecom_app.runner import _gap_entries_from_payload
from agi_ecom_app.schemas import (
    ActionExecutionCheck,
    ActionExecutionForecast,
    ActionExecutionResult,
    ChatContextWindow,
    EcomRunContext,
    MemoryMode,
    QueryIntent,
    QueryMode,
    RoundType,
    WorkflowVersion,
)


class AgentQueueTests(unittest.TestCase):
    def test_price_router_defaults_action_scope_to_price_not_supporting_domains(self):
        decision = _repair_domain_route_decision(
            DomainRouteDecision(
                route_to_domain=True,
                intent=QueryIntent.PRICE,
                confidence="high",
                reason="ценовое решение",
                supporting_agents=[
                    QueryIntent.FINANCE,
                    QueryIntent.MARKETING,
                    QueryIntent.SUPPLY,
                ],
                query_plan=QueryPlan(
                    query_mode=QueryMode.DECISION,
                    decision_kind="action_portfolio",
                    required_domains=[
                        QueryIntent.PRICE,
                        QueryIntent.FINANCE,
                        QueryIntent.MARKETING,
                        QueryIntent.SUPPLY,
                    ],
                ),
            )
        )

        self.assertEqual(decision.query_plan.allowed_action_families, ["price"])

    def test_explicit_action_scope_uses_single_canonical_advisory_renderer(self):
        output = _compose_manager_answer_markdown(
            decision_markdown="# Решение заблокировано\n\nПроверить рекламу и поставку.",
            advisory_markdown="## Итог для бизнеса\n\nТолько ценовые варианты.",
            advisory={
                "priceSimulations": [{"target": "A"}],
                "portfolioOptimization": {"complete": False},
            },
            query_plan={"allowed_action_families": ["price"]},
        )

        self.assertEqual(
            output,
            "## Итог для бизнеса\n\nТолько ценовые варианты.",
        )

    def test_explicit_action_scope_falls_back_when_advisory_has_no_artifacts(self):
        output = _compose_manager_answer_markdown(
            decision_markdown="# Решение заблокировано\n\nНе хватает исходных данных.",
            advisory_markdown="## Итог для бизнеса\n\nРассчитано 0 из 5.",
            advisory={
                "errors": [{"stage": "manager_advisory", "error_type": "TimeoutError"}],
                "priceSimulations": [],
            },
            query_plan={
                "decision_kind": "action_portfolio",
                "allowed_action_families": ["price"],
            },
        )

        self.assertEqual(
            output,
            "# Решение заблокировано\n\nНе хватает исходных данных.",
        )

    def test_blocked_executable_shortage_is_not_a_final_checklist_violation(self):
        violations = _portfolio_priority_quality_violations(
            {
                "violations": ["expected 5 optimized recommendations, got 0"],
                "priorityViolations": [],
                "priorityRecommendations": [
                    {"eligible": False, "executionStatus": "blocked"}
                    for _ in range(5)
                ],
            },
            {"decision_kind": "action_portfolio"},
        )

        self.assertEqual(violations, [])

    def test_priority_checklist_violation_remains_a_final_quality_violation(self):
        violations = _portfolio_priority_quality_violations(
            {
                "violations": [],
                "priorityViolations": [
                    "expected 5 priority recommendations, got 4"
                ],
            },
            {"decision_kind": "action_portfolio"},
        )

        self.assertEqual(
            violations,
            ["expected 5 priority recommendations, got 4"],
        )

    def test_typed_cross_check_no_data_blocks_same_unconfirmed_gap(self):
        entries = _gap_entries_from_payload(
            payload={
                "_round_result": {
                    "round_type": "cross_check",
                    "verdict": "no_data",
                    "facts": [],
                },
                "data_gaps": [
                    {
                        "metric": "Рекламные расходы за последние 7 дней",
                        "reason": "Источник не вернул строки за последние 7 дней.",
                        "impact": "Причина изменения продаж не подтверждена.",
                    }
                ],
            },
            source_agent="marketing",
            blocking=True,
        )

        self.assertEqual(entries[0]["evidence_status"], "no_data")
        terminal_gap = SimpleNamespace(sku=None, **entries[0])
        stale_unconfirmed_gap = SimpleNamespace(
            target_agent="marketing",
            source_agent="marketing",
            sku=None,
            metric="Рекламные расходы за последние 7 дней",
            reason="Проверить рекламные расходы за последние 7 дней.",
            evidence_status="unconfirmed",
        )

        self.assertEqual(
            _deduplicate_actionable_gaps([stale_unconfirmed_gap, terminal_gap]),
            [],
        )
        self.assertEqual(
            _deduplicate_actionable_gaps([terminal_gap, stale_unconfirmed_gap]),
            [],
        )

    def test_primary_prompt_failure_retry_policy_keeps_transport_errors_retriable(self):
        api_connection_error = type("APIConnectionError", (Exception,), {})
        api_timeout_error = type("APITimeoutError", (Exception,), {})
        model_behavior_error = type("ModelBehaviorError", (Exception,), {})

        self.assertTrue(_should_retry_question_failure(api_connection_error()))
        self.assertTrue(_should_retry_question_failure(api_timeout_error()))
        self.assertFalse(_should_retry_question_failure(model_behavior_error()))

    def test_v2_synthesis_can_fall_back_on_exhausted_quota(self):
        quota_error = type("RateLimitError", (Exception,), {})

        self.assertTrue(
            _synthesis_runtime_fallback_allowed(
                quota_error("429 insufficient_quota"),
                WorkflowVersion.V2_OPTIMIZED,
            )
        )
        self.assertFalse(
            _synthesis_runtime_fallback_allowed(
                quota_error("429 insufficient_quota"),
                WorkflowVersion.V1,
            )
        )

    def test_v2_synthesis_does_not_hide_model_validation_errors(self):
        validation_error = type("ModelBehaviorError", (Exception,), {})

        self.assertFalse(
            _synthesis_runtime_fallback_allowed(
                validation_error("invalid structured output"),
                WorkflowVersion.V2_OPTIMIZED,
            )
        )

    def test_primary_validation_failure_persists_prompt_report(self):
        question = SimpleNamespace(
            id=17,
            run_id=9,
            round_id=3,
            source_agent="orchestrator",
            target_agent="sales",
        )
        trace = SimpleNamespace(
            id=9,
            chat_id="chat-validation-failure",
            intent="sales",
            summary={
                "user_query": "Почему изменились продажи?",
                "workflow_version": "v2_optimized",
                "query_mode": "diagnostic",
            },
        )
        context = EcomRunContext(
            user_query="Почему изменились продажи?",
            intent=QueryIntent.SALES,
            today=date.today(),
            db_target="memory",
            chat_id=trace.chat_id,
            workflow_version=WorkflowVersion.V2_OPTIMIZED,
            round_type=RoundType.PRIMARY,
            query_mode=QueryMode.DIAGNOSTIC,
            memory_mode=MemoryMode.SUMMARY_RECENT,
        )
        failed_question = SimpleNamespace(**question.__dict__)
        response = SimpleNamespace(id=23)

        class ModelBehaviorError(Exception):
            pass

        def fail_with_prompt_report(*_args, **kwargs):
            run_context = kwargs["context"]
            run_context.last_prompt_report = {
                **(kwargs.get("prompt_report") or {}),
                "system_chars": 321,
                "input_chars": 654,
            }
            raise ModelBehaviorError("facts.11: unit is required")

        with (
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id", return_value=trace),
            patch("agi_ecom_app.agent_queue._build_context_from_trace", return_value=context),
            patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=SimpleNamespace()),
            patch(
                "agi_ecom_app.agent_queue._round_payload_for_question",
                return_value={"round_type": "primary"},
            ),
            patch(
                "agi_ecom_app.agent_queue._domain_tool_policy",
                return_value=SimpleNamespace(required_tools=[]),
            ),
            patch(
                "agi_ecom_app.agent_queue._domain_agent_builder_for_intent",
                return_value=lambda: SimpleNamespace(tools=[]),
            ),
            patch("agi_ecom_app.agent_queue._run_agent", side_effect=fail_with_prompt_report),
            patch(
                "agi_ecom_app.agent_queue.mark_agent_run_question_failed",
                return_value=failed_question,
            ),
            patch("agi_ecom_app.agent_queue.save_agent_run_response", return_value=response) as save_response,
            patch("agi_ecom_app.agent_queue.save_agent_run_gaps"),
        ):
            _execute_question(question)

        raw_output = save_response.call_args.kwargs["raw_output"]
        self.assertEqual(raw_output["_prompt_report"]["system_chars"], 321)
        self.assertEqual(raw_output["_prompt_report"]["input_chars"], 654)
        self.assertEqual(
            save_response.call_args.kwargs["runtime_failure"]["error_type"],
            "ModelBehaviorError",
        )

    @patch("agi_ecom_app.agent_queue.save_agent_run_gaps")
    @patch("agi_ecom_app.agent_queue.save_agent_run_response")
    @patch("agi_ecom_app.agent_queue.mark_agent_run_question_failed")
    def test_persist_question_failure_keeps_guardrail_output_and_violations(
        self,
        mock_mark_failed,
        mock_save_response,
        mock_save_gaps,
    ):
        question = SimpleNamespace(id=17)
        failed_question = SimpleNamespace(
            id=17,
            run_id=9,
            source_agent="orchestrator",
            target_agent="sales",
        )
        mock_mark_failed.return_value = failed_question
        mock_save_response.return_value = SimpleNamespace(id=23)
        exc = OutputGuardrailTripwireTriggered(
            SimpleNamespace(
                guardrail=SimpleNamespace(),
                output=SimpleNamespace(
                    output_info={"violations": ["sales result has invalid decomposition"]}
                ),
                agent_output={
                    "summary": "Собран частичный ответ.",
                    "tables": [],
                    "findings": [],
                    "recommendations": [],
                    "data_gaps": [],
                    "cross_agent_requests": [],
                },
            )
        )

        _persist_question_failure(question=question, exc=exc)

        response_kwargs = mock_save_response.call_args.kwargs
        self.assertEqual(response_kwargs["summary"], "Собран частичный ответ.")
        self.assertEqual(
            response_kwargs["raw_output"]["_guardrail_violations"],
            ["sales result has invalid decomposition"],
        )
        self.assertEqual(response_kwargs["runtime_failure"]["kind"], "guardrail")
        self.assertEqual(
            response_kwargs["runtime_failure"]["violations"],
            ["sales result has invalid decomposition"],
        )
        gap = mock_save_gaps.call_args.kwargs["gaps"][0]
        self.assertEqual(gap["metric"], "sales output guardrail")
        self.assertEqual(gap["reason"], "sales result has invalid decomposition")

    @patch("agi_ecom_app.agent_queue.enqueue_agent_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_round")
    @patch("agi_ecom_app.agent_queue.create_agent_run_trace")
    @patch("agi_ecom_app.agent_queue.append_user_message_idempotent")
    @patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id", return_value=None)
    @patch("agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id", return_value=None)
    @patch(
        "agi_ecom_app.agent_queue.get_context_window",
        return_value=ChatContextWindow(chat_id="chat-queue-1", summary_text=None, recent_messages=[]),
    )
    @patch(
        "agi_ecom_app.agent_queue.route_query_intent_with_llm",
        return_value=DomainRouteDecision(
            route_to_domain=True,
            intent=QueryIntent.SALES,
            confidence="high",
            reason="test route",
            supporting_agents=[],
        ),
    )
    @patch("agi_ecom_app.agent_queue.ensure_chat_session")
    def test_enqueue_initial_run_creates_primary_question_and_returns_queued(
        self,
        _mock_ensure_chat_session,
        _mock_route_intent,
        _mock_get_context_window,
        _mock_get_reply,
        _mock_get_run,
        mock_append_user_message,
        mock_create_trace,
        mock_create_round,
        mock_create_question,
        mock_enqueue_question,
    ):
        mock_append_user_message.return_value = (SimpleNamespace(seq=3), True)
        mock_create_trace.return_value = SimpleNamespace(id=11)
        mock_create_round.return_value = SimpleNamespace(id=21)
        mock_create_question.return_value = SimpleNamespace(id=31)

        result = enqueue_initial_run(
            chat_id="chat-queue-1",
            user_query="Проверь продажи SKU 46300",
            client_message_id="client-1",
        )

        self.assertEqual(result.status, "queued")
        self.assertEqual(result.run_id, 11)
        mock_create_trace.assert_called_once()
        self.assertEqual(mock_create_trace.call_args.kwargs["intent"], "sales")
        mock_create_round.assert_called_once_with(
            run_id=11,
            round_number=0,
            round_type="primary",
            status="pending",
        )
        mock_enqueue_question.assert_called_once_with(31)

    def test_enqueue_initial_run_uses_llm_route_over_keyword_fallback(self):
        query = "Какую цену на товар с низкой оборачиваемостью мы можем поставить?"
        with (
            patch("agi_ecom_app.agent_queue.ensure_chat_session"),
            patch("agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id", return_value=None),
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id", return_value=None),
            patch("agi_ecom_app.agent_queue.determine_query_intent", return_value=QueryIntent.SUPPLY),
            patch(
                "agi_ecom_app.agent_queue.get_context_window",
                return_value=ChatContextWindow(chat_id="chat-price-route", summary_text=None, recent_messages=[]),
            ),
            patch(
                "agi_ecom_app.agent_queue.route_query_intent_with_llm",
                return_value=DomainRouteDecision(
                    route_to_domain=True,
                    intent=QueryIntent.PRICE,
                    confidence="high",
                    reason="price decision by meaning",
                    supporting_agents=[QueryIntent.SUPPLY, QueryIntent.FINANCE],
                    query_plan=QueryPlan(
                        query_mode=QueryMode.DECISION,
                        scope="general",
                        decision_kind="single_action",
                        requested_analyses=["pricing"],
                        required_domains=[
                            QueryIntent.PRICE,
                            QueryIntent.SUPPLY,
                            QueryIntent.FINANCE,
                        ],
                    ),
                ),
            ),
            patch("agi_ecom_app.agent_queue.append_user_message_idempotent") as mock_append_user_message,
            patch("agi_ecom_app.agent_queue.create_agent_run_trace") as mock_create_trace,
            patch("agi_ecom_app.agent_queue.create_agent_run_round") as mock_create_round,
            patch("agi_ecom_app.agent_queue.create_agent_run_question") as mock_create_question,
            patch("agi_ecom_app.agent_queue.enqueue_agent_question") as mock_enqueue_question,
        ):
            mock_append_user_message.return_value = (SimpleNamespace(seq=3), True)
            mock_create_trace.return_value = SimpleNamespace(id=11)
            mock_create_round.return_value = SimpleNamespace(id=21)
            mock_create_question.return_value = SimpleNamespace(id=31)

            result = enqueue_initial_run(
                chat_id="chat-price-route",
                user_query=query,
                client_message_id="client-price-route",
            )

        self.assertEqual(result.intent, QueryIntent.PRICE)
        self.assertEqual(mock_create_trace.call_args.kwargs["intent"], "price")
        self.assertEqual(mock_create_trace.call_args.kwargs["primary_agent"], "price_agent")
        self.assertEqual(mock_create_question.call_args.kwargs["target_agent"], "price")
        append_metadata = mock_append_user_message.call_args.kwargs["metadata"]
        self.assertEqual(append_metadata["intent"], "price")
        self.assertEqual(append_metadata["intent_router"]["intent"], "price")
        mock_enqueue_question.assert_called_once_with(31)

    def test_manager_decision_cannot_be_diverted_to_legacy_domain_router(self):
        query = (
            "Сформируй единый план действий по продажам, финансам, рекламе, "
            "ценам и остаткам и ранжируй действия по прибыли."
        )
        settings = SimpleNamespace(
            agent_workflow_version="v2_optimized",
            manager_decision_enabled=True,
            memory_context_message_limit=12,
            db_host="db",
            db_port=5432,
            db_name="analytics",
        )
        with (
            patch("agi_ecom_app.agent_queue.get_settings", return_value=settings),
            patch("agi_ecom_app.agent_queue._active_model_label", return_value="test:model"),
            patch("agi_ecom_app.agent_queue._active_model_metadata_safe", return_value={}),
            patch("agi_ecom_app.agent_queue.ensure_chat_session"),
            patch("agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id", return_value=None),
            patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id", return_value=None),
            patch(
                "agi_ecom_app.agent_queue.get_context_window",
                return_value=ChatContextWindow(
                    chat_id="manager-route",
                    summary_text=None,
                    recent_messages=[],
                ),
            ),
            patch(
                "agi_ecom_app.agent_queue.route_query_intent_with_llm",
                return_value=DomainRouteDecision(
                    route_to_domain=False,
                    intent=QueryIntent.FINANCE,
                    confidence="high",
                    reason="generic answer would normally use domain_router",
                    supporting_agents=[],
                    query_plan=QueryPlan(
                        query_mode=QueryMode.DECISION,
                        scope="portfolio",
                        decision_kind="action_portfolio",
                        requested_analyses=[
                            "contribution_economics",
                            "advertising",
                            "pricing",
                            "inventory_matrix",
                        ],
                        required_domains=list(QueryIntent),
                    ),
                ),
            ),
            patch("agi_ecom_app.agent_queue.append_user_message_idempotent") as append_message,
            patch("agi_ecom_app.agent_queue.create_agent_run_trace") as create_trace,
            patch("agi_ecom_app.agent_queue.create_agent_run_round") as create_round,
            patch("agi_ecom_app.agent_queue.create_agent_run_question") as create_question,
            patch("agi_ecom_app.agent_queue.enqueue_agent_question") as enqueue_question,
        ):
            append_message.return_value = (SimpleNamespace(seq=1), True)
            create_trace.return_value = SimpleNamespace(id=12)
            create_round.return_value = SimpleNamespace(id=22)
            create_question.return_value = SimpleNamespace(id=32)

            result = enqueue_initial_run(
                chat_id="manager-route",
                user_query=query,
                client_message_id="manager-route-1",
            )

        self.assertEqual(result.status, "queued")
        self.assertEqual(create_trace.call_args.kwargs["primary_agent"], "finance_agent")
        self.assertTrue(
            create_trace.call_args.kwargs["summary"]["manager_decision_forced_domain_route"]
        )
        self.assertEqual(create_question.call_args.kwargs["target_agent"], "finance")
        self.assertEqual(
            create_trace.call_args.kwargs["summary"]["intent_router"]["query_plan"]["scope"],
            "portfolio",
        )
        self.assertEqual(
            create_question.call_args.kwargs["verification_goal"]["intent_router"]["query_plan"]["decision_kind"],
            "action_portfolio",
        )
        enqueue_question.assert_called_once_with(32)

    @patch(
        "agi_ecom_app.agent_queue.get_agent_run_question",
        return_value=SimpleNamespace(status="pending", job_id=None),
    )
    @patch("agi_ecom_app.agent_queue.mark_agent_run_question_queued")
    @patch("agi_ecom_app.agent_queue._build_rq_queue")
    def test_enqueue_agent_question_records_rq_job_id(
        self,
        mock_build_queue,
        mock_mark_queued,
        _mock_get_question,
    ):
        queue = SimpleNamespace()
        queue.enqueue_call = Mock(return_value=SimpleNamespace(id="agent-question-42"))
        mock_build_queue.return_value = queue

        job_id = enqueue_agent_question(42)

        self.assertEqual(job_id, "agent-question-42")
        queue.enqueue_call.assert_called_once()
        self.assertEqual(queue.enqueue_call.call_args.kwargs["job_id"], "agent-question-42")
        mock_mark_queued.assert_called_once()
        self.assertEqual(mock_mark_queued.call_args.kwargs["question_id"], 42)
        self.assertEqual(mock_mark_queued.call_args.kwargs["job_id"], "agent-question-42")

    @patch("agi_ecom_app.agent_queue.enqueue_agent_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_round")
    @patch("agi_ecom_app.agent_queue.create_agent_run_trace")
    @patch("agi_ecom_app.agent_queue.append_user_message_idempotent")
    @patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id", return_value=None)
    @patch("agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id", return_value=None)
    @patch("agi_ecom_app.agent_queue.ensure_chat_session")
    def test_enqueue_initial_run_honors_intent_override(
        self,
        _mock_ensure_chat_session,
        _mock_get_reply,
        _mock_get_run,
        mock_append_user_message,
        mock_create_trace,
        mock_create_round,
        mock_create_question,
        _mock_enqueue_question,
    ):
        mock_append_user_message.return_value = (SimpleNamespace(seq=3), True)
        mock_create_trace.return_value = SimpleNamespace(id=11)
        mock_create_round.return_value = SimpleNamespace(id=21)
        mock_create_question.return_value = SimpleNamespace(id=31)

        result = enqueue_initial_run(
            chat_id="chat-queue-override",
            user_query="Выполни подготовленную задачу",
            client_message_id="client-override",
            intent_override=QueryIntent.MARKETING,
        )

        self.assertEqual(result.intent, QueryIntent.MARKETING)
        self.assertEqual(mock_create_trace.call_args.kwargs["intent"], "marketing")
        self.assertEqual(mock_create_trace.call_args.kwargs["primary_agent"], "marketing_agent")
        self.assertEqual(mock_create_question.call_args.kwargs["target_agent"], "marketing")

    @patch("agi_ecom_app.agent_queue.enqueue_agent_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_question")
    @patch("agi_ecom_app.agent_queue.create_agent_run_round")
    @patch("agi_ecom_app.agent_queue.create_agent_run_trace")
    @patch("agi_ecom_app.agent_queue.append_user_message_idempotent")
    @patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id", return_value=None)
    @patch("agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id", return_value=None)
    @patch("agi_ecom_app.agent_queue.ensure_chat_session")
    def test_enqueue_action_execution_run_creates_terminal_action_question(
        self,
        _mock_ensure_chat_session,
        _mock_get_reply,
        _mock_get_run,
        mock_append_user_message,
        mock_create_trace,
        mock_create_round,
        mock_create_question,
        mock_enqueue_question,
    ):
        mock_append_user_message.return_value = (SimpleNamespace(seq=9), True)
        mock_create_trace.return_value = SimpleNamespace(id=15)
        mock_create_round.return_value = SimpleNamespace(id=25)
        mock_create_question.return_value = SimpleNamespace(id=35)

        result = enqueue_action_execution_run(
            chat_id="chat-exec-1",
            source_chat_id="chat-main-1",
            source_execution_id=77,
            prompt="Выполни terminal action по SKU 46227.",
            client_message_id="client-terminal-1",
            intent_override=QueryIntent.SUPPLY,
        )

        self.assertEqual(result.status, "queued")
        self.assertEqual(result.run_id, 15)
        self.assertEqual(mock_create_round.call_args.kwargs["round_type"], "terminal_action")
        self.assertEqual(mock_create_question.call_args.kwargs["target_agent"], "action_executor")
        self.assertTrue(mock_create_question.call_args.kwargs["verification_goal"]["suppress_action_tasks"])
        append_metadata = mock_append_user_message.call_args.kwargs["metadata"]
        self.assertTrue(append_metadata["suppress_action_tasks"])
        self.assertEqual(append_metadata["execution_mode"], "terminal_action")
        mock_enqueue_question.assert_called_once_with(35)

    def test_round_failure_reason_surfaces_openai_quota_error(self):
        reason = _round_failure_reason(
            {
                "questions": [
                    {
                        "question": SimpleNamespace(
                            status="failed",
                            last_error=(
                                "RateLimitError: Error code: 429 - "
                                "{'error': {'code': 'insufficient_quota'}}"
                            ),
                        ),
                        "responses": [],
                    }
                ],
            },
            "primary question failed",
        )

        self.assertIn("Недостаточно квоты OpenAI API", reason)
        self.assertIn("insufficient_quota", reason)

    @patch("agi_ecom_app.agent_queue.complete_agent_run_trace")
    @patch("agi_ecom_app.agent_queue._persist_assistant_turn")
    @patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_id")
    @patch("agi_ecom_app.agent_queue.extract_action_tasks_from_agent_output")
    @patch("agi_ecom_app.agent_queue._trace_metadata")
    @patch("agi_ecom_app.agent_queue.update_agent_run_round_status")
    @patch("agi_ecom_app.agent_queue.update_agent_run_question_status")
    @patch("agi_ecom_app.agent_queue._persist_agent_run_response_safely")
    def test_persist_simple_final_answer_loads_trace_before_action_task_suppression(
        self,
        mock_persist_response,
        mock_update_question,
        mock_update_round,
        mock_trace_metadata,
        mock_extract_tasks,
        mock_get_trace,
        mock_persist_turn,
        mock_complete_trace,
    ):
        mock_trace_metadata.return_value = {}
        mock_extract_tasks.return_value = []
        mock_get_trace.return_value = SimpleNamespace(
            id=55,
            chat_id="chat-trace-1",
            summary={},
            primary_agent="domain_router_agent",
        )
        mock_persist_turn.return_value = SimpleNamespace(assistant_message_seq=77)

        result = _persist_simple_final_answer(
            trace_id=55,
            context=SimpleNamespace(
                decision_trace=[],
                agent_trace=[],
                tool_trace=[],
                cross_agent_trace=[],
            ),
            final_output="Готовый ответ",
            intent=QueryIntent.SALES,
            client_message_id="client-55",
            question=SimpleNamespace(
                id=9,
                round_id=11,
                verification_goal={},
            ),
        )

        self.assertEqual(result.assistant_message_seq, 77)
        mock_get_trace.assert_called_once_with(55)
        mock_extract_tasks.assert_called_once_with("Готовый ответ", "Готовый ответ")
        mock_persist_response.assert_called_once()
        mock_update_question.assert_called_once_with(question_id=9, status="completed")
        mock_update_round.assert_called_once_with(round_id=11, status="completed")
        mock_complete_trace.assert_called_once()

    @patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=object())
    @patch("agi_ecom_app.agent_queue._run_agent")
    @patch("agi_ecom_app.agent_queue.build_action_result_formatter_input", return_value="format me")
    @patch("agi_ecom_app.agent_queue.execute_action_handler")
    @patch("agi_ecom_app.agent_queue.run_forecast_workflow", return_value=None)
    @patch("agi_ecom_app.agent_queue.get_agent_action_execution")
    def test_supported_terminal_action_uses_handler_and_formatter(
        self,
        mock_get_execution,
        _mock_forecast,
        mock_execute_handler,
        _mock_formatter_input,
        mock_run_agent,
        _mock_build_hooks,
    ):
        draft = ActionExecutionResult(
            businessMeaning="Черновой смысл",
            checks=[
                ActionExecutionCheck(
                    name="остаток",
                    status="warning",
                    summary="Черновая проверка",
                    evidence=["coverage_days=2"],
                )
            ],
            forecast=ActionExecutionForecast(
                salesVelocity=5.0,
                activeDays=7,
                availableStock=10.0,
                inboundStock=3.0,
                coverageDays=2.0,
                safetyStockUnits=35.0,
                reorderQty=92.0,
                stockoutRisk="high",
            ),
            decision="Черновое решение",
            decisionStatus="completed",
            confidence="high",
            blockedBy=[],
            finalRecommendation="Черновая рекомендация",
            summary="Черновое summary",
        )
        outcome = ActionHandlerOutcome(
            handler="preserve_availability_and_sales",
            data_version="v1",
            draft_result=draft,
            evidence_package={"metrics": {"reorder_qty": 92}},
        )
        formatted = ActionExecutionResult(
            businessMeaning="Отформатированный смысл",
            checks=[
                ActionExecutionCheck(
                    name="остаток",
                    status="ok",
                    summary="Отформатированная проверка",
                    evidence=[],
                )
            ],
            forecast=ActionExecutionForecast(stockoutRisk="low"),
            decision="Отформатированное решение",
            decisionStatus="blocked",
            confidence="low",
            blockedBy=["ignore-this"],
            finalRecommendation="Отформатированная рекомендация",
            summary="Отформатированное summary",
        )

        mock_get_execution.return_value = SimpleNamespace(
            task_snapshot={"actionKind": "preserve_availability_and_sales", "taskId": "task-1"}
        )
        mock_execute_handler.return_value = outcome
        mock_run_agent.return_value = formatted

        final_output, returned_outcome, agent_name = _resolve_terminal_action_output(
            trace=SimpleNamespace(summary={"source_chat_id": "chat-1", "source_execution_id": 7}),
            question=SimpleNamespace(question="Выполни terminal action"),
            context=SimpleNamespace(today=date(2026, 4, 23)),
        )

        self.assertEqual(agent_name, "action_result_formatter")
        self.assertEqual(returned_outcome, outcome)
        self.assertEqual(final_output.businessMeaning, "Отформатированный смысл")
        self.assertEqual(final_output.decisionStatus, "completed")
        self.assertEqual(final_output.summary, "Отформатированное summary")
        self.assertEqual(final_output.forecast.reorderQty, 92.0)
        self.assertEqual(final_output.checks[0].status, "warning")
        self.assertEqual(final_output.checks[0].summary, "Отформатированная проверка")
        self.assertEqual(mock_run_agent.call_args.args[0].__name__, "build_action_result_formatter_agent")

    @patch("agi_ecom_app.agent_queue.build_run_hooks", return_value=object())
    @patch("agi_ecom_app.agent_queue._run_agent", return_value="generic-output")
    @patch("agi_ecom_app.agent_queue.execute_action_handler", return_value=None)
    @patch("agi_ecom_app.agent_queue.run_forecast_workflow", return_value=None)
    @patch("agi_ecom_app.agent_queue.get_agent_action_execution")
    def test_unsupported_terminal_action_keeps_generic_executor_fallback(
        self,
        mock_get_execution,
        _mock_forecast,
        _mock_execute_handler,
        mock_run_agent,
        _mock_build_hooks,
    ):
        mock_get_execution.return_value = SimpleNamespace(
            task_snapshot={"actionKind": "validate_unit_economics", "taskId": "task-2"}
        )

        final_output, returned_outcome, agent_name = _resolve_terminal_action_output(
            trace=SimpleNamespace(summary={"source_chat_id": "chat-1", "source_execution_id": 8}),
            question=SimpleNamespace(question="Выполни terminal action"),
            context=SimpleNamespace(today=date(2026, 4, 23)),
        )

        self.assertEqual(final_output, "generic-output")
        self.assertIsNone(returned_outcome)
        self.assertEqual(agent_name, "action_executor")
        self.assertEqual(mock_run_agent.call_args.args[0].__name__, "build_action_executor_agent")

    @patch("agi_ecom_app.agent_queue.execute_action_handler")
    @patch("agi_ecom_app.agent_queue.run_forecast_workflow")
    @patch("agi_ecom_app.agent_queue.get_agent_action_execution")
    def test_terminal_action_prefers_forecast_when_applicable(
        self,
        mock_get_execution,
        mock_forecast,
        mock_execute_handler,
    ):
        forecast = ForecastDecisionResult(
            hypothesis="stop_or_reduce_campaigns: 309891193",
            domain="marketing",
            baseline={"name": "baseline", "metrics": {"current_spend_rub": 2013.75}},
            actionScenario={"name": "action", "metrics": {"campaigns_to_reduce": "309891193"}},
            delta={"metrics": {"ad_spend_saved_rub": 2013.75}},
            decision="Гипотеза подтверждена.",
            confidence="high",
            finalRecommendation="Остановить подтвержденную слабую кампанию.",
            summary="Экономия 2013.75 руб.",
        )
        mock_get_execution.return_value = SimpleNamespace(
            task_snapshot={
                "actionKind": "optimize_marketing_visibility",
                "taskId": "task-forecast",
                "title": "Сократить или остановить слабые кампании: 309891193",
                "targets": ["309891193"],
            }
        )
        mock_forecast.return_value = forecast

        final_output, returned_outcome, agent_name = _resolve_terminal_action_output(
            trace=SimpleNamespace(summary={"source_chat_id": "chat-1", "source_execution_id": 9}),
            question=SimpleNamespace(question="Выполни terminal action"),
            context=SimpleNamespace(today=date(2026, 4, 24)),
        )

        self.assertEqual(final_output, forecast)
        self.assertIsNone(returned_outcome)
        self.assertEqual(agent_name, "forecast_orchestrator")
        mock_execute_handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
