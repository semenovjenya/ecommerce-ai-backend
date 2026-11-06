from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping

from agents import OutputGuardrailTripwireTriggered

from agi_ecom_app.action_execution import (
    build_action_result_formatter_input,
    execute_action_handler,
    merge_formatted_action_result,
)
from agi_ecom_app.config import DEFAULT_AGENT_WORKFLOW_VERSION, get_settings
from agi_ecom_app.main_agent import (
    build_action_executor_agent,
    build_action_result_formatter_agent,
    build_diagnostic_orchestrator_agent,
    build_domain_router_agent,
    build_run_hooks,
    ensure_supply_aggregated_stock_facts,
)
from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.forecast.workflow import run_forecast_workflow
from agi_ecom_app.llm_runtime import get_active_llm_runtime_spec, get_active_model_metadata
from agi_ecom_app.llm_routing import DomainRouteDecision, QueryPlan, route_query_intent_with_llm
from agi_ecom_app.observability import (
    agent_trace_id,
    flush_langfuse,
    init_langfuse,
    observation_scope,
    observability_summary,
    trace_context_for_agent_run,
    update_observation,
)
from agi_ecom_app.manager_decision import (
    manager_actions_to_action_board,
    run_manager_decision_workflow,
)
from agi_ecom_app.manager_decision.candidate_builder import (
    build_manager_action_candidates,
)
from agi_ecom_app.manager_decision.candidate_source import (
    load_manager_candidate_snapshot,
)
from agi_ecom_app.manager_decision.advisory import (
    build_manager_advisory,
    render_manager_advisory,
    should_build_manager_advisory,
    validate_scenario_completeness,
)
from agi_ecom_app.manager_lifecycle.service import persist_manager_decision_bundle
from agi_ecom_app.prompt_pipeline import (
    build_round_prompt,
    compact_synthesis_package,
    determine_query_mode as determine_prompt_query_mode,
    memory_mode_for_round,
)
from agi_ecom_app.prompt_quality import (
    build_prompt_v2_repair_prompt,
    validate_prompt_v2_answer,
)
from agi_ecom_app.routing import determine_query_intent, has_domain_routing_signal
from agi_ecom_app.runner import (
    ACTION_TASKS_METADATA_KEY,
    ACTION_TASKS_SCHEMA_VERSION,
    ACTION_TASKS_SCHEMA_VERSION_KEY,
    CLAIM_VERIFICATION_LIMIT,
    CLAIM_VERIFICATION_ROUND_NUMBER,
    DIAGNOSTIC_SECONDARY_MAX_TURNS,
    DIAGNOSTIC_SYNTHESIS_MAX_TURNS,
    MAX_DIAGNOSTIC_ROUNDS,
    MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
    ROUND_1_SECONDARY_LIMIT,
    ROUND_2_FOLLOWUP_LIMIT,
    ROUND_3_GAP_RESOLUTION_LIMIT,
    ROUND_RESULT_TYPES,
    SYNTHESIS_ROUND_NUMBER,
    AgentRuntimeError,
    ChatTurnResult,
    DuplicateClientMessageError,
    ToolPolicyViolation,
    _agent_run_trace_summary,
    _agent_tool_name,
    _asked_item,
    _attach_prompt_report,
    _attach_tool_report,
    _build_action_safety_plan,
    _build_secondary_task_prompt,
    _build_sku_evidence_matrix,
    _build_synthesis_package,
    build_query_plan_inventory_matrix,
    classify_gap_resolution_response,
    _coerce_intent,
    _coerce_output,
    _diagnostic_cross_agent_requests,
    _diagnostic_data_gaps,
    _diagnostic_findings,
    _diagnostic_payload,
    _diagnostic_summary,
    _domain_tool_policy,
    _domain_agent_builder_for_intent,
    _enforce_gap_resolution_fact_policy,
    _extract_guardrail_violations,
    _find_agent_run_gap_safely,
    _finalize_action_safety_checks,
    _gap_entries_from_payload,
    _gap_entries_from_sku_matrix,
    _inventory_turnover_coverage_plan,
    _claim_verification_plan,
    _intent_from_agent_name,
    _is_non_actionable_sku_source_gap,
    _is_inventory_turnover_causal_query,
    _load_persisted_evidence_safely,
    _merge_dependency_plans,
    _open_blocking_gap_payloads_safely,
    _payload_has_business_evidence,
    _persist_agent_action_tasks_safely,
    _persist_agent_run_response_safely,
    _persist_assistant_turn,
    _render_prompt_v2_facts_only_answer,
    _persist_gap_entries_safely,
    _plan_items,
    _prepend_required_report_snapshots,
    _run_agent,
    _primary_output_type,
    _secondary_allowed_tools,
    _v2_required_tools_for_plan,
    _v2_required_tools_for_query,
    _sku_matrix_quality_violations,
    _sku_only_display_was_sanitized,
    remove_empty_markdown_tables,
    _sanitize_inventory_turnover_cause_claims,
    _trace_metadata,
    _update_agent_run_gap_status_safely,
    _verification_goal_from_item,
    should_persist_gap_from_gap_resolution,
)
from agi_ecom_app.schemas import (
    ActionExecutionResult,
    ChatRole,
    EcomRunContext,
    QueryIntent,
    QueryMode,
    RoundTask,
    RoundType,
    WorkflowVersion,
)
from agi_ecom_app.services.memory import (
    AgentRunQuestionRecord,
    append_user_message_idempotent,
    claim_agent_run_question_running,
    claim_stale_agent_questions_for_delivery,
    claim_timed_out_agent_run_questions,
    complete_agent_run_trace,
    create_agent_run_question,
    create_agent_run_round,
    create_agent_run_trace,
    ensure_chat_session,
    get_agent_action_execution,
    get_agent_run_question,
    get_agent_run_trace_by_client_message_id,
    get_agent_run_trace_by_id,
    get_assistant_reply_for_client_message_id,
    get_context_window,
    get_memory_connection,
    list_agent_run_gaps,
    list_agent_run_responses,
    list_agent_run_rounds,
    mark_agent_run_question_failed,
    mark_agent_run_question_queued,
    save_agent_run_gaps,
    save_agent_run_response,
    update_agent_run_question_status,
    update_agent_run_round_status,
    update_agent_run_trace_status,
)
from agi_ecom_app.tools.strategic_tools import build_cross_domain_dependency_plan
from agi_ecom_app.action_board.service import (
    EXECUTION_MODE_METADATA_KEY,
    SUPPRESS_ACTION_TASKS_METADATA_KEY,
    TERMINAL_ACTION_EXECUTION_MODE,
    extract_action_tasks_from_agent_output,
)

logger = logging.getLogger(__name__)

TERMINAL_QUESTION_STATUSES = {"completed", "failed", "skipped"}
RUN_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
MAX_FAILURE_REASON_LENGTH = 500
V2_OPTIMIZED_WORKFLOW_VERSION = "v2_optimized"
QUERY_MODES = frozenset({"lookup", "report", "diagnostic", "decision"})
AGENT_QUESTION_RECONCILIATION_INTERVAL_SECONDS = 60
AGENT_QUESTION_RECONCILIATION_HORIZON_SLOTS = 5
# The router and one possible synthesis repair are model executions even though
# they are not persisted as question rows.  Eleven persisted slots let a manager
# decision inspect the primary domain plus all four remaining business domains,
# run safety and gap checks, and still reserve synthesis capacity.
MAX_OPTIMIZED_MODEL_EXECUTIONS = 13
OPTIMIZED_ROUTER_EXECUTIONS = 1
OPTIMIZED_REPAIR_EXECUTION_RESERVE = 1
MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS = (
    MAX_OPTIMIZED_MODEL_EXECUTIONS
    - OPTIMIZED_ROUTER_EXECUTIONS
    - OPTIMIZED_REPAIR_EXECUTION_RESERVE
)
MAX_OPTIMIZED_FAST_PATH_QUESTION_ROWS = 2
MAX_OPTIMIZED_CROSS_CHECKS = 3
MAX_OPTIMIZED_DECISION_CROSS_CHECKS = 4
MAX_OPTIMIZED_SAFETY_CHECKS = 2
MAX_OPTIMIZED_GAP_RESOLUTIONS = 2
MAX_OPTIMIZED_CLAIM_VERIFICATIONS = 1
OPTIMIZED_SYNTHESIS_QUESTION_RESERVE = 1

# Internal compatibility aliases. The preferred names above distinguish persisted
# question rows from the unpersisted router and the optional synthesis repair call.
MAX_OPTIMIZED_MODEL_QUESTIONS = MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS
OPTIMIZED_FINAL_CALL_RESERVE = OPTIMIZED_SYNTHESIS_QUESTION_RESERVE

MANAGER_DOMAIN_REQUIRED_CHECKS: dict[QueryIntent, list[str]] = {
    QueryIntent.SALES: [
        "sales_units_and_revenue_labeled_separately",
        "conversion_funnel",
        "period_and_source_snapshot",
    ],
    QueryIntent.FINANCE: [
        "revenue",
        "cogs",
        "commission_logistics_storage",
        "contribution_profit",
    ],
    QueryIntent.MARKETING: [
        "campaign_status_and_current_bid",
        "spend_clicks_orders_same_period",
        "attributed_revenue_and_budget",
    ],
    QueryIntent.PRICE: [
        "current_price_and_discount",
        "price_volume_history",
        "comparable_competitors",
    ],
    QueryIntent.SUPPLY: [
        "available_on_hand_stock",
        "supplier_inbound_not_customer_transit",
        "sales_velocity_and_lead_time",
    ],
}

MANAGER_DOMAIN_REQUIRED_TOOLS: dict[QueryIntent, list[str]] = {
    QueryIntent.SALES: ["get_sales_report_snapshot", "get_sales_overview"],
    QueryIntent.FINANCE: [
        "get_database_time",
        "get_finance_overview",
        "get_finance_sku_report_text",
    ],
    QueryIntent.MARKETING: [
        "get_marketing_report_snapshot",
        "get_marketing_campaign_efficiency_snapshot",
    ],
    QueryIntent.PRICE: ["get_price_context", "get_price_decision_overview"],
    QueryIntent.SUPPLY: ["get_supply_overview"],
}

MANAGER_DOMAIN_QUESTIONS: dict[QueryIntent, str] = {
    QueryIntent.SALES: "Проверь продажи в штуках, выручку и конверсию на одном периоде; верни даты и источники.",
    QueryIntent.FINANCE: "Проверь полную contribution-экономику после COGS, комиссий, логистики, хранения и рекламы.",
    QueryIntent.MARKETING: "Проверь текущие кампании, ставки, бюджет, расход, клики, заказы и атрибутированную выручку на одном периоде.",
    QueryIntent.PRICE: "Проверь текущую цену и скидку, price-volume историю и сопоставимость конкурентов.",
    QueryIntent.SUPPLY: "Проверь доступный on-hand, скорость продаж, реальную поставку поставщика и lead time; клиентский транзит не считай inbound.",
}

def _route_decision_metadata(decision: DomainRouteDecision) -> dict[str, Any]:
    return {
        "route_to_domain": decision.route_to_domain,
        "intent": decision.intent.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "supporting_agents": [agent.value for agent in decision.supporting_agents],
        "query_plan": decision.query_plan.model_dump(mode="json"),
    }


def _determine_query_mode(user_query: str) -> str:
    return determine_prompt_query_mode(user_query).value


def _is_manager_portfolio_request(
    user_query: str | None = None,
    *,
    query_plan: dict[str, Any] | None = None,
) -> bool:
    """Use the LLM router contract instead of parsing the user's language again."""

    del user_query
    return str((query_plan or {}).get("decision_kind") or "") == "action_portfolio"


def _trace_query_plan(trace: Any) -> dict[str, Any]:
    summary = dict(getattr(trace, "summary", {}) or {})
    router = dict(summary.get("intent_router") or {})
    plan = router.get("query_plan")
    return dict(plan) if isinstance(plan, dict) else {}


def _query_plan_targets(query_plan: dict[str, Any] | None) -> list[str]:
    values: list[str] = []
    for target in list((query_plan or {}).get("targets") or []):
        if not isinstance(target, dict):
            continue
        value = str(target.get("value") or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _portfolio_priority_quality_violations(
    portfolio_optimization: dict[str, Any] | None,
    query_plan: dict[str, Any] | None,
) -> list[str]:
    """Return manager-facing checklist violations, not executable-gate diagnostics."""

    if str((query_plan or {}).get("decision_kind") or "") != "action_portfolio":
        return []
    optimization = dict(portfolio_optimization or {})
    return [
        str(item)
        for item in list(
            optimization.get("priorityViolations")
            or optimization.get("priority_violations")
            or []
        )
        if str(item)
    ]


def _workflow_trace_metadata(*, workflow_version: str, query_mode: str) -> dict[str, str]:
    return {
        "workflow_version": workflow_version,
        "prompt_version": workflow_version,
        "query_mode": query_mode,
    }


def _trace_workflow_version(trace: Any) -> str:
    """Read the version pinned when the run was created, never live settings."""

    summary = dict(getattr(trace, "summary", {}) or {})
    version = str(summary.get("workflow_version") or DEFAULT_AGENT_WORKFLOW_VERSION)
    if version not in {"v1", "v2_prompt", V2_OPTIMIZED_WORKFLOW_VERSION}:
        return DEFAULT_AGENT_WORKFLOW_VERSION
    return version


def _trace_query_mode(trace: Any) -> str:
    summary = dict(getattr(trace, "summary", {}) or {})
    mode = str(summary.get("query_mode") or "")
    if mode in QUERY_MODES:
        return mode
    return _determine_query_mode(str(summary.get("user_query") or ""))


def _synthesis_repair_allowed(
    workflow_version: WorkflowVersion,
    query_mode: QueryMode,
) -> bool:
    return not (
        workflow_version == WorkflowVersion.V2_OPTIMIZED
        and query_mode in {QueryMode.LOOKUP, QueryMode.REPORT}
    )


@dataclass(frozen=True)
class AgentRunQueueResult:
    chat_id: str
    intent: QueryIntent
    run_id: int | None
    status: str
    answer: str | None = None
    server_message_id: int | None = None
    manager_decision: dict[str, Any] | None = None


def _active_model_metadata_safe() -> dict[str, Any]:
    settings = get_settings()
    try:
        return dict(get_active_model_metadata(require_configured=False, settings=settings))
    except Exception:
        return {
            "model_profile": getattr(settings, "llm_model_profile", "openai_default"),
            "model_provider": "openai",
            "model_name": settings.openai_model,
            "model_label": f"openai:{settings.openai_model}",
            "model_experimental": False,
            "model_missing_env_vars": [],
        }


def _active_model_label() -> str:
    return str(_active_model_metadata_safe().get("model_label") or "openai:unknown")


def _should_suppress_action_tasks(question: AgentRunQuestionRecord | None, trace: Any) -> bool:
    verification_goal = getattr(question, "verification_goal", {}) or {}
    trace_summary = getattr(trace, "summary", {}) or {}
    return bool(
        verification_goal.get("suppress_action_tasks")
        or trace_summary.get("suppress_action_tasks")
        or verification_goal.get("execution_mode") == TERMINAL_ACTION_EXECUTION_MODE
        or trace_summary.get("execution_mode") == TERMINAL_ACTION_EXECUTION_MODE
    )


def enqueue_initial_run(
    *,
    chat_id: str,
    user_query: str,
    client_message_id: str | None = None,
    intent_override: QueryIntent | str | None = None,
) -> AgentRunQueueResult:
    settings = get_settings()
    workflow_version = str(
        getattr(settings, "agent_workflow_version", DEFAULT_AGENT_WORKFLOW_VERSION)
    )
    query_mode = _determine_query_mode(user_query)
    manager_decision_requested = bool(
        getattr(settings, "manager_decision_enabled", False)
        and workflow_version == V2_OPTIMIZED_WORKFLOW_VERSION
        and query_mode == QueryMode.DECISION.value
    )
    workflow_metadata = _workflow_trace_metadata(
        workflow_version=workflow_version,
        query_mode=query_mode,
    )
    forced_intent = _coerce_intent_override(intent_override)
    fallback_intent = determine_query_intent(user_query)
    ensure_chat_session(chat_id=chat_id, model_name=_active_model_label())

    if client_message_id:
        existing_reply = get_assistant_reply_for_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_reply:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_reply.metadata.get("intent"), fallback_intent),
                run_id=None,
                status="completed",
                answer=existing_reply.content,
                server_message_id=existing_reply.seq,
                manager_decision=(existing_reply.metadata or {}).get("manager_decision"),
            )
        existing_run = get_agent_run_trace_by_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_run:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_run.intent, fallback_intent),
                run_id=existing_run.id,
                status=existing_run.status,
                server_message_id=existing_run.assistant_message_seq,
            )

    intake_trace_id = agent_trace_id(
        chat_id=chat_id,
        client_message_id=client_message_id,
        settings=settings,
    )
    with observation_scope(
        name="agent-run-intake",
        as_type="chain",
        trace_id=intake_trace_id,
        input={"chat_id": chat_id, "user_query": user_query},
        metadata={
            "intent": fallback_intent.value,
            "query_mode": query_mode,
            "workflow_version": workflow_version,
            **_active_model_metadata_safe(),
        },
        session_id=chat_id,
        tags=["agent-run", "intake", query_mode],
        version=workflow_version,
        root=True,
        settings=settings,
    ) as intake_observation:
        observability = observability_summary(intake_observation, settings=settings)
        if forced_intent is not None:
            intent = forced_intent
            route_decision = DomainRouteDecision(
                route_to_domain=True,
                intent=forced_intent,
                confidence="high",
                reason="intent_override",
                supporting_agents=[],
                query_plan=QueryPlan(
                    query_mode=QueryMode(query_mode),
                    decision_kind=(
                        "single_action"
                        if query_mode == QueryMode.DECISION.value
                        else "none"
                    ),
                    required_domains=[forced_intent],
                ),
            )
        else:
            routing_window = get_context_window(
                chat_id=chat_id,
                limit=settings.memory_context_message_limit,
            )
            routing_context = EcomRunContext(
                user_query=user_query,
                intent=fallback_intent,
                today=date.today(),
                db_target=f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
                chat_id=chat_id,
                workflow_version=WorkflowVersion(workflow_version),
                round_type=RoundType.ROUTER,
                query_mode=QueryMode(query_mode),
                memory_mode=memory_mode_for_round(RoundType.ROUTER),
                chat_summary=routing_window.summary_text,
                recent_messages=routing_window.recent_messages,
                evidence_budget=0,
            )
            try:
                route_decision = route_query_intent_with_llm(user_query, context=routing_context)
            except Exception as exc:
                update_observation(intake_observation, error=exc)
                if manager_decision_requested:
                    route_decision = DomainRouteDecision(
                        route_to_domain=True,
                        intent=fallback_intent,
                        confidence="low",
                        reason=f"manager_decision_router_fallback:{type(exc).__name__}",
                        supporting_agents=[
                            candidate for candidate in QueryIntent if candidate != fallback_intent
                        ],
                        query_plan=QueryPlan(
                            query_mode=QueryMode(query_mode),
                            decision_kind="single_action",
                            required_domains=list(QueryIntent),
                        ),
                    )
                else:
                    user_message, _ = append_user_message_idempotent(
                        chat_id=chat_id,
                        content=user_query,
                        metadata={
                            "intent": fallback_intent.value,
                            "intent_router": {
                                "route_to_domain": False,
                                "intent": fallback_intent.value,
                                "confidence": "low",
                                "reason": f"llm_router_failed:{type(exc).__name__}",
                                "supporting_agents": [],
                            },
                            **workflow_metadata,
                            **_active_model_metadata_safe(),
                        },
                        client_message_id=client_message_id,
                    )
                    trace = create_agent_run_trace(
                        chat_id=chat_id,
                        intent=fallback_intent.value,
                        client_message_id=client_message_id,
                        primary_agent="intent_router",
                        status="failed",
                        summary={
                            "status": "failed",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "user_query": user_query,
                            "user_message_seq": user_message.seq,
                            **({"observability": observability} if observability else {}),
                            **workflow_metadata,
                            **_active_model_metadata_safe(),
                        },
                    )
                    _emit_agent_run_result_observation(trace)
                    return AgentRunQueueResult(
                        chat_id=chat_id,
                        intent=fallback_intent,
                        run_id=trace.id,
                        status="failed",
                    )
            intent = route_decision.intent
        query_mode = route_decision.query_plan.query_mode.value
        manager_decision_requested = bool(
            getattr(settings, "manager_decision_enabled", False)
            and workflow_version == V2_OPTIMIZED_WORKFLOW_VERSION
            and query_mode == QueryMode.DECISION.value
        )
        workflow_metadata = _workflow_trace_metadata(
            workflow_version=workflow_version,
            query_mode=query_mode,
        )
        update_observation(
            intake_observation,
            output={"intent_router": _route_decision_metadata(route_decision)},
            metadata={"intent": intent.value, "query_mode": query_mode},
        )

    user_message, created_new = append_user_message_idempotent(
        chat_id=chat_id,
        content=user_query,
        metadata={
            "intent": intent.value,
            "intent_router": _route_decision_metadata(route_decision),
            **workflow_metadata,
            **_active_model_metadata_safe(),
        },
        client_message_id=client_message_id,
    )
    if client_message_id and not created_new:
        existing_run = get_agent_run_trace_by_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_run:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_run.intent, intent),
                run_id=existing_run.id,
                status=existing_run.status,
                server_message_id=existing_run.assistant_message_seq,
            )
        raise DuplicateClientMessageError(client_message_id=client_message_id)

    domain_routing_signal = bool(route_decision.route_to_domain or manager_decision_requested)
    primary_agent = _agent_tool_name(intent) if domain_routing_signal else "domain_router"
    trace = create_agent_run_trace(
        chat_id=chat_id,
        intent=intent.value,
        client_message_id=client_message_id,
        primary_agent=primary_agent,
        status="queued",
        summary={
            "status": "queued",
            "user_query": user_query,
            "user_message_seq": user_message.seq,
            "domain_routing_signal": domain_routing_signal,
            "manager_decision_forced_domain_route": bool(
                manager_decision_requested and not route_decision.route_to_domain
            ),
            "intent_router": _route_decision_metadata(route_decision),
            **({"observability": observability} if observability else {}),
            **workflow_metadata,
            **_active_model_metadata_safe(),
        },
    )
    primary_round = create_agent_run_round(
        run_id=trace.id,
        round_number=0,
        round_type="primary",
        status="pending",
    )
    primary_question = create_agent_run_question(
        run_id=trace.id,
        round_id=primary_round.id,
        source_agent="orchestrator",
        target_agent=intent.value if domain_routing_signal else "domain_router",
        question=user_query,
        reason="Primary agent selected by query router.",
        verification_goal={
            "original_user_query": user_query,
            "primary_agent": intent.value if domain_routing_signal else "domain_router",
            "domain_routing_signal": domain_routing_signal,
            "intent_router": _route_decision_metadata(route_decision),
            "round": "primary",
            **workflow_metadata,
        },
        priority=100,
        status="pending",
    )
    enqueue_agent_question(primary_question.id)
    return AgentRunQueueResult(
        chat_id=chat_id,
        intent=intent,
        run_id=trace.id,
        status="queued",
    )


def enqueue_action_execution_run(
    *,
    chat_id: str,
    source_chat_id: str,
    source_execution_id: int,
    prompt: str,
    client_message_id: str | None = None,
    intent_override: QueryIntent | str | None = None,
) -> AgentRunQueueResult:
    settings = get_settings()
    forced_intent = _coerce_intent_override(intent_override)
    intent = forced_intent or determine_query_intent(prompt)
    ensure_chat_session(chat_id=chat_id, model_name=_active_model_label())

    if client_message_id:
        existing_reply = get_assistant_reply_for_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_reply:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_reply.metadata.get("intent"), intent),
                run_id=None,
                status="completed",
                answer=existing_reply.content,
                server_message_id=existing_reply.seq,
            )
        existing_run = get_agent_run_trace_by_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_run:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_run.intent, intent),
                run_id=existing_run.id,
                status=existing_run.status,
                server_message_id=existing_run.assistant_message_seq,
            )

    user_message, created_new = append_user_message_idempotent(
        chat_id=chat_id,
        content=prompt,
        metadata={
            "intent": intent.value,
            EXECUTION_MODE_METADATA_KEY: TERMINAL_ACTION_EXECUTION_MODE,
            SUPPRESS_ACTION_TASKS_METADATA_KEY: True,
            "source_chat_id": source_chat_id,
            "source_execution_id": source_execution_id,
            **_active_model_metadata_safe(),
        },
        client_message_id=client_message_id,
    )
    if client_message_id and not created_new:
        existing_run = get_agent_run_trace_by_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_run:
            return AgentRunQueueResult(
                chat_id=chat_id,
                intent=_coerce_intent(existing_run.intent, intent),
                run_id=existing_run.id,
                status=existing_run.status,
                server_message_id=existing_run.assistant_message_seq,
            )
        raise DuplicateClientMessageError(client_message_id=client_message_id)

    execution_trace_id = agent_trace_id(
        chat_id=chat_id,
        client_message_id=client_message_id,
        settings=settings,
    )
    with observation_scope(
        name="agent-run-intake",
        as_type="chain",
        trace_id=execution_trace_id,
        input={"chat_id": chat_id, "user_query": prompt},
        metadata={
            "intent": intent.value,
            "execution_mode": TERMINAL_ACTION_EXECUTION_MODE,
            **_active_model_metadata_safe(),
        },
        session_id=chat_id,
        tags=["agent-run", "intake", "terminal-action"],
        root=True,
        settings=settings,
    ) as intake_observation:
        observability = observability_summary(intake_observation, settings=settings)
        trace = create_agent_run_trace(
            chat_id=chat_id,
            intent=intent.value,
            client_message_id=client_message_id,
            primary_agent="action_executor",
            status="queued",
            summary={
                "status": "queued",
                "user_query": prompt,
                "user_message_seq": user_message.seq,
                "execution_mode": TERMINAL_ACTION_EXECUTION_MODE,
                "source_chat_id": source_chat_id,
                "source_execution_id": source_execution_id,
                "suppress_action_tasks": True,
                **({"observability": observability} if observability else {}),
                **_active_model_metadata_safe(),
            },
        )
        update_observation(
            intake_observation,
            output={"run_id": trace.id, "status": getattr(trace, "status", "queued")},
        )
    primary_round = create_agent_run_round(
        run_id=trace.id,
        round_number=0,
        round_type="terminal_action",
        status="pending",
    )
    primary_question = create_agent_run_question(
        run_id=trace.id,
        round_id=primary_round.id,
        source_agent="orchestrator",
        target_agent="action_executor",
        question=prompt,
        reason="Terminal action execution for Action Board task.",
        verification_goal={
            "execution_mode": TERMINAL_ACTION_EXECUTION_MODE,
            "source_chat_id": source_chat_id,
            "source_execution_id": source_execution_id,
            "suppress_action_tasks": True,
            "primary_agent": "action_executor",
        },
        priority=100,
        status="pending",
    )
    enqueue_agent_question(primary_question.id)
    return AgentRunQueueResult(
        chat_id=chat_id,
        intent=intent,
        run_id=trace.id,
        status="queued",
    )


def _coerce_intent_override(value: QueryIntent | str | None) -> QueryIntent | None:
    if value is None:
        return None
    if isinstance(value, QueryIntent):
        return value
    try:
        return QueryIntent(str(value))
    except ValueError:
        return None


def enqueue_agent_question(question_id: int) -> str:
    durable_question = get_agent_run_question(question_id)
    if durable_question is None:
        raise LookupError(question_id)

    deterministic_job_id = f"agent-question-{int(question_id)}"
    if durable_question.status in TERMINAL_QUESTION_STATUSES:
        return str(durable_question.job_id or deterministic_job_id)
    if durable_question.status in {"queued", "running"}:
        return str(durable_question.job_id or deterministic_job_id)

    settings = get_settings()
    queue = _build_rq_queue(settings.redis_url, settings.agent_queue_name)
    retry = _build_rq_retry()
    existing = (
        queue.fetch_job(deterministic_job_id)
        if hasattr(queue, "fetch_job")
        else None
    )
    if existing is not None:
        raw_status = existing.get_status(refresh=True)
        delivery_status = str(getattr(raw_status, "value", raw_status)).lower()
        if delivery_status not in {"finished", "failed", "stopped", "canceled"}:
            mark_agent_run_question_queued(
                question_id=question_id,
                job_id=existing.id,
                timeout_seconds=settings.agent_job_timeout_seconds,
            )
            return str(existing.id)
        try:
            existing.delete()
        except Exception:
            logger.warning(
                "agent_question_terminal_delivery_delete_failed "
                "question_id=%s job_id=%s",
                question_id,
                deterministic_job_id,
                exc_info=True,
            )
            raise
    try:
        try:
            job = queue.enqueue_call(
                func=run_agent_question_job,
                args=(question_id,),
                job_id=deterministic_job_id,
                timeout=settings.agent_job_timeout_seconds,
                retry=retry,
            )
        except TypeError:
            job = queue.enqueue_call(
                func=run_agent_question_job,
                args=(question_id,),
                job_id=deterministic_job_id,
                job_timeout=settings.agent_job_timeout_seconds,
                retry=retry,
            )
    except Exception:
        # Concurrent producers may both observe ``pending``. The deterministic
        # RQ ID makes one delivery authoritative; recover it instead of turning
        # a harmless duplicate enqueue into a workflow failure.
        job = (
            queue.fetch_job(deterministic_job_id)
            if hasattr(queue, "fetch_job")
            else None
        )
        if job is None:
            raise
    mark_agent_run_question_queued(
        question_id=question_id,
        job_id=job.id,
        timeout_seconds=settings.agent_job_timeout_seconds,
    )
    return str(job.id)


def run_agent_question_job(question_id: int) -> None:
    question = claim_agent_run_question_running(question_id)
    if question is None:
        current = get_agent_run_question(question_id)
        if current is None:
            raise LookupError(question_id)
        logger.info(
            "agent_question_duplicate_delivery_skipped question_id=%s run_id=%s status=%s job_id=%s",
            question_id,
            current.run_id,
            current.status,
            current.job_id,
        )
        return
    trace = get_agent_run_trace_by_id(question.run_id)
    if trace is None:
        raise LookupError(question.run_id)
    if trace.status == "queued":
        update_agent_run_trace_status(trace_id=trace.id, status="running")

    init_langfuse(service_name="agent-worker", worker_role="question-job")
    langfuse_trace_id, root_observation_id = trace_context_for_agent_run(
        summary=getattr(trace, "summary", None),
        run_id=trace.id,
    )
    trace_summary = dict(getattr(trace, "summary", {}) or {})
    try:
        with observation_scope(
            name="agent-question",
            as_type="agent",
            trace_id=langfuse_trace_id,
            parent_observation_id=root_observation_id,
            input={
                "question": getattr(question, "question", ""),
                "verification_goal": getattr(question, "verification_goal", {}),
            },
            metadata={
                "run_id": trace.id,
                "question_id": question.id,
                "round_id": getattr(question, "round_id", None),
                "round_type": trace_summary.get("round_type"),
                "intent": getattr(trace, "intent", None),
                "target_agent": getattr(question, "target_agent", None),
                "query_mode": trace_summary.get("query_mode"),
                "workflow_version": trace_summary.get("workflow_version"),
                "prompt_version": trace_summary.get("prompt_version"),
                "model_profile": trace_summary.get("model_profile"),
                "model_provider": trace_summary.get("model_provider"),
                "model_name": trace_summary.get("model_name"),
            },
            session_id=getattr(trace, "chat_id", None),
            tags=["agent-run", "question", str(getattr(question, "target_agent", "unknown"))],
            version=str(trace_summary.get("workflow_version") or "v1"),
        ) as question_observation:
            try:
                _execute_question(question)
            except Exception as exc:
                latest = get_agent_run_question(question_id) or question
                if latest.attempt_count <= 1 and _should_retry_question_failure(exc):
                    update_agent_run_question_status(question_id=question_id, status="queued")
                    update_observation(question_observation, error=exc)
                    raise
                _persist_question_failure(question=latest, exc=exc)
                update_observation(question_observation, error=exc)
            finally:
                try:
                    advance_agent_run(question.run_id)
                except Exception as advance_exc:
                    logger.exception(
                        "agent_run_advance_failed run_id=%s question_id=%s error=%s",
                        question.run_id,
                        question_id,
                        advance_exc,
                    )
            if question_observation is not None:
                latest_question = get_agent_run_question(question_id) or question
                latest_trace = get_agent_run_trace_by_id(question.run_id) or trace
                update_observation(
                    question_observation,
                    output={
                        "question_status": latest_question.status,
                        "run_status": latest_trace.status,
                    },
                    metadata={"status": latest_question.status},
                )
                if latest_trace.status in RUN_TERMINAL_STATUSES:
                    _emit_agent_run_result_observation(latest_trace)
    finally:
        flush_langfuse()


def reconcile_stale_agent_questions(
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail killed work-horse questions and idempotently advance their runs."""

    settings = get_settings()
    current = now or datetime.now(timezone.utc)
    delivery_recovered = claim_stale_agent_questions_for_delivery(
        now=current,
        default_timeout_seconds=settings.agent_job_timeout_seconds,
        pending_lease_seconds=60,
        grace_seconds=60,
        limit=100,
    )
    requeued_ids: list[int] = []
    for question in delivery_recovered:
        try:
            enqueue_agent_question(question.id)
            requeued_ids.append(question.id)
        except Exception:
            # The pending lease expires after one minute, so transient Redis
            # failures are retried without duplicating the LLM execution.
            logger.exception(
                "agent_question_delivery_recovery_failed question_id=%s run_id=%s",
                question.id,
                question.run_id,
            )
    timed_out = claim_timed_out_agent_run_questions(
        now=current,
        default_timeout_seconds=settings.agent_job_timeout_seconds,
        grace_seconds=60,
        limit=100,
    )
    advanced_runs: list[int] = []
    for question in timed_out:
        try:
            advance_agent_run(question.run_id)
            advanced_runs.append(question.run_id)
        except Exception:
            # The timeout marker stays durable and is selected again until the
            # trace reaches a terminal state, so a transient DB failure cannot
            # leave the round stuck forever.
            logger.exception(
                "agent_question_timeout_advance_failed question_id=%s run_id=%s",
                question.id,
                question.run_id,
            )
    return {
        "requeued_question_ids": requeued_ids,
        "timed_out_question_ids": [question.id for question in timed_out],
        "advanced_run_ids": list(dict.fromkeys(advanced_runs)),
    }


def schedule_agent_question_reconciliation(
    *,
    queue: Any | None = None,
    now: datetime | None = None,
) -> str:
    """Maintain several deterministic future slots across worker crashes."""

    settings = get_settings()
    resolved_queue = queue or _build_rq_queue(
        settings.redis_url,
        settings.agent_queue_name,
    )
    current = now or datetime.now(timezone.utc)
    first_job_id = ""
    for offset in range(1, AGENT_QUESTION_RECONCILIATION_HORIZON_SLOTS + 1):
        run_at = current + timedelta(
            seconds=AGENT_QUESTION_RECONCILIATION_INTERVAL_SECONDS * offset
        )
        slot = int(run_at.timestamp()) // AGENT_QUESTION_RECONCILIATION_INTERVAL_SECONDS
        job_id = f"agent-question-reconcile-{slot}"
        if not first_job_id:
            first_job_id = job_id
        try:
            resolved_queue.enqueue_at(
                run_at,
                run_agent_question_reconciler_job,
                job_id=job_id,
                result_ttl=AGENT_QUESTION_RECONCILIATION_INTERVAL_SECONDS * 2,
                failure_ttl=86_400,
            )
        except Exception as exc:
            if "already exists" not in str(exc).casefold():
                raise
    return first_job_id


def run_agent_question_reconciler_job() -> dict[str, Any]:
    # Top up the future horizon before touching Postgres. One killed maintenance
    # job therefore cannot break the recurring chain.
    schedule_agent_question_reconciliation()
    return reconcile_stale_agent_questions()


def _should_retry_question_failure(exc: Exception) -> bool:
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _synthesis_runtime_fallback_allowed(
    exc: Exception,
    workflow_version: WorkflowVersion,
) -> bool:
    """Keep persisted v2 evidence usable when only final LLM synthesis is unavailable."""

    if workflow_version == WorkflowVersion.V1:
        return False
    error_text = str(exc).lower()
    return type(exc).__name__ == "RateLimitError" or "insufficient_quota" in error_text


def advance_agent_run(run_id: int) -> None:
    with get_memory_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("select pg_try_advisory_xact_lock(%s)", [run_id])
            row = cur.fetchone()
            if not row or not row["pg_try_advisory_xact_lock"]:
                return
            for _ in range(6):
                progressed = _advance_agent_run_locked(run_id)
                if not progressed:
                    break


def _execute_question(question: AgentRunQuestionRecord) -> None:
    trace = get_agent_run_trace_by_id(question.run_id)
    if trace is None:
        raise LookupError(question.run_id)
    context = _build_context_from_trace(trace)
    hooks = build_run_hooks()
    round_payload = _round_payload_for_question(run_id=question.run_id, round_id=question.round_id)
    round_type = (round_payload or {}).get("round_type")
    context.workflow_version = WorkflowVersion(_trace_workflow_version(trace))
    context.round_type = RoundType(str(round_type or RoundType.PRIMARY.value))
    context.query_mode = QueryMode(_trace_query_mode(trace))
    context.memory_mode = memory_mode_for_round(context.round_type)
    user_query = _trace_user_query(trace)
    intent = _coerce_intent(trace.intent, determine_query_intent(user_query))

    if question.target_agent == "domain_router":
        final_output = _run_agent(
            build_domain_router_agent,
            user_input=user_query,
            context=context,
            hooks=hooks,
            max_turns=24,
        )
        _persist_simple_final_answer(
            trace_id=trace.id,
            context=context,
            final_output=final_output,
            intent=intent,
            client_message_id=trace.client_message_id,
            question=question,
        )
        return

    if round_type == "terminal_action" or question.target_agent == "action_executor":
        _execute_terminal_action_question(
            trace_id=trace.id,
            question=question,
            context=context,
            intent=intent,
        )
        return

    if round_type == "primary":
        runtime_failure: dict[str, Any] | None = None
        workflow_version = context.workflow_version
        primary_required_tools = (
            MANAGER_DOMAIN_REQUIRED_TOOLS[intent]
            if (
                workflow_version == WorkflowVersion.V2_OPTIMIZED
                and context.query_mode == QueryMode.DECISION
                and bool(getattr(get_settings(), "manager_decision_enabled", False))
                and _is_manager_portfolio_request(query_plan=context.query_plan)
            )
            else []
        )
        tool_policy = _domain_tool_policy(
            target=intent,
            user_query=user_query,
            round_name="primary",
            question_id=question.id,
            explicit_required_tools=[
                *primary_required_tools,
                *(
                    (
                        _v2_required_tools_for_plan(intent, context.query_plan)
                        if context.query_plan
                        else _v2_required_tools_for_query(intent, user_query)
                    )
                    if workflow_version != WorkflowVersion.V1
                    else ()
                ),
            ],
        )
        primary_input = user_query
        primary_prompt_report: dict[str, Any] | None = None
        primary_builder = _domain_agent_builder_for_intent(
            intent,
            workflow_version=workflow_version,
            round_type=RoundType.PRIMARY,
        )
        if workflow_version != WorkflowVersion.V1:
            primary_agent_preview = primary_builder()
            primary_allowed_tools = [
                str(getattr(tool, "name", "") or "")
                for tool in getattr(primary_agent_preview, "tools", [])
                if str(getattr(tool, "name", "") or "")
            ]
            primary_task = RoundTask(
                round_type=RoundType.PRIMARY,
                target_agent=intent.value,
                question=user_query,
                original_user_query=user_query,
                verification_goal={
                    "original_user_query": user_query,
                    "primary_agent": intent.value,
                },
                required_checks=[],
                source="router",
                gap_id=None,
                claim_policy=None,
                period=None,
                sku=None,
                vendor_codes=[],
                nm_ids=[],
                allowed_tools=primary_allowed_tools,
                required_tools=list(tool_policy.required_tools),
                query_plan=dict(context.query_plan),
            )
            primary_build = build_round_prompt(primary_task, None, [], workflow_version)
            primary_input = primary_build.text
            primary_prompt_report = primary_build.report.model_dump(mode="json")
        try:
            output = _run_agent(
                primary_builder,
                user_input=primary_input,
                context=context,
                hooks=hooks,
                max_turns=12,
                tool_policy=tool_policy,
                expected_output_type=(
                    _primary_output_type(intent)
                    if workflow_version != WorkflowVersion.V1
                    else None
                ),
                prompt_report=primary_prompt_report,
            )
        except ToolPolicyViolation as exc:
            output = exc.output
            context.last_tool_report = exc.tool_report
            runtime_failure = {"target_agent": intent.value, **exc.runtime_failure()}
        except OutputGuardrailTripwireTriggered as exc:
            if _manager_primary_failure_fallback_allowed(
                trace=trace,
                context=context,
                user_query=user_query,
            ):
                _persist_question_failure(
                    question=question,
                    exc=exc,
                    prompt_report=context.last_prompt_report,
                    blocking=False,
                    transition_question=False,
                )
                try:
                    _execute_manager_primary_failure_fallback(
                        trace=trace,
                        question=question,
                        context=context,
                        intent=intent,
                        user_query=user_query,
                        exc=exc,
                    )
                except Exception as fallback_exc:
                    logger.exception(
                        "manager_primary_guardrail_fallback_failed "
                        "run_id=%s question_id=%s error=%s",
                        trace.id,
                        question.id,
                        type(fallback_exc).__name__,
                    )
                    raise
                return
            output = exc.guardrail_result.agent_output
            runtime_failure = {
                "target_agent": intent.value,
                "kind": "guardrail",
                "error_type": type(exc).__name__,
                "message": "Primary output failed internal quality validation.",
                "violations": _extract_guardrail_violations(exc),
            }
        except Exception as exc:
            # Connection/time-out failures must reach the job-level retry policy.
            # Non-retriable parsing/validation errors are persisted here while the
            # context still contains the exact prompt report used by the model.
            if _should_retry_question_failure(exc):
                raise
            if _manager_primary_failure_fallback_allowed(
                trace=trace,
                context=context,
                user_query=user_query,
            ):
                _persist_question_failure(
                    question=question,
                    exc=exc,
                    prompt_report=context.last_prompt_report,
                    blocking=False,
                    transition_question=False,
                )
                try:
                    _execute_manager_primary_failure_fallback(
                        trace=trace,
                        question=question,
                        context=context,
                        intent=intent,
                        user_query=user_query,
                        exc=exc,
                    )
                except Exception as fallback_exc:
                    # Keep the question non-terminal until the deterministic
                    # result is fully durable.  The job-level handler must see a
                    # fallback persistence error so it can fail the question and
                    # advance the run closed instead of starting cross-checks.
                    logger.exception(
                        "manager_primary_failure_fallback_failed "
                        "run_id=%s question_id=%s error=%s",
                        trace.id,
                        question.id,
                        type(fallback_exc).__name__,
                    )
                    raise
                return
            _persist_question_failure(
                question=question,
                exc=exc,
                prompt_report=context.last_prompt_report,
            )
            return
        context.cross_agent_trace.append(f"primary:{intent.value}")
        payload = _attach_prompt_report(
            _attach_tool_report(_diagnostic_payload(output), context),
            context,
        )
        _persist_agent_run_response_safely(
            run_id=question.run_id,
            question_record=question,
            agent_name=intent.value,
            payload=payload,
            runtime_failure=runtime_failure,
            blocking_gaps=True,
        )
        update_agent_run_question_status(
            question_id=question.id,
            status="failed" if runtime_failure else "completed",
        )
        update_agent_run_round_status(
            round_id=question.round_id,
            status="failed" if runtime_failure else "completed",
        )
        return

    if round_type == "synthesis":
        _execute_synthesis_question(
            trace_id=trace.id,
            question=question,
            context=context,
            intent=intent,
            user_query=user_query,
        )
        return

    persisted = _load_persisted_evidence_safely(question.run_id)
    if not persisted:
        raise AgentRuntimeError("Primary evidence is not available for secondary question.")
    item = _question_to_plan_item(question)
    prior_evidence = [
        *persisted.get("round_1_evidence", []),
        *persisted.get("followup_evidence", []),
    ]
    if round_type == "cross_check":
        prior_evidence = []
    evidence = _run_domain_question(
        question=question,
        item=item,
        user_query=user_query,
        primary_output=persisted["primary_output"],
        prior_evidence=prior_evidence,
        context=context,
        hooks=hooks,
        round_name=_round_name(round_type),
    )


def _run_domain_question(
    *,
    question: AgentRunQuestionRecord,
    item: dict[str, Any],
    user_query: str,
    primary_output: Any,
    prior_evidence: list[dict[str, Any]],
    context: EcomRunContext,
    hooks: Any,
    round_name: str,
) -> dict[str, Any]:
    target = _intent_from_agent_name(item.get("target_agent"))
    workflow_version = context.workflow_version
    round_type = context.round_type
    tool_policy = _domain_tool_policy(
        target=target,
        user_query=user_query,
        round_name=round_name,
        question_id=question.id,
        explicit_required_tools=[
            *(item.get("required_tools") or ()),
            *(
                (
                    _v2_required_tools_for_plan(target, context.query_plan)
                    if context.query_plan
                    else _v2_required_tools_for_query(target, user_query)
                )
                if workflow_version != WorkflowVersion.V1
                else ()
            ),
        ],
    )
    allowed_tools = (
        _secondary_allowed_tools(
            intent=target,
            question=" ".join(
                [
                    question.question,
                    json.dumps(
                        _verification_goal_from_item(item, round_name=round_name),
                        ensure_ascii=False,
                        default=str,
                    ),
                    " ".join(str(value) for value in (item.get("required_checks") or [])),
                ]
            ),
            required_tools=tool_policy.required_tools,
        )
        if workflow_version != WorkflowVersion.V1
        else []
    )
    started_at = time.monotonic()
    context.add_decision_trace(
        "secondary_agent_requested",
        round=round_name,
        target_agent=target.value,
        question=question.question,
        status="started",
        details={"source": item.get("source"), "priority": item.get("priority")},
    )
    prompt_build = _build_secondary_task_prompt(
        user_input=user_query,
        item=item,
        primary_output=primary_output,
        prior_evidence=prior_evidence,
        workflow_version=workflow_version,
        round_type=round_type,
        allowed_tools=allowed_tools,
        required_tools=tool_policy.required_tools,
        query_plan=context.query_plan,
        return_build_result=workflow_version != WorkflowVersion.V1,
    )
    if workflow_version == WorkflowVersion.V1:
        task_prompt = str(prompt_build)
        prompt_report = None
    else:
        task_prompt = prompt_build.text
        prompt_report = prompt_build.report.model_dump(mode="json")
    runtime_failure: dict[str, Any] | None = None
    try:
        output = _run_agent(
            _domain_agent_builder_for_intent(
                target,
                workflow_version=workflow_version,
                round_type=round_type,
                allowed_tools=allowed_tools,
            ),
            user_input=task_prompt,
            context=context,
            hooks=hooks,
            max_turns=DIAGNOSTIC_SECONDARY_MAX_TURNS,
            tool_policy=tool_policy,
            expected_output_type=(
                ROUND_RESULT_TYPES.get(round_type)
                if workflow_version != WorkflowVersion.V1
                else None
            ),
            prompt_report=prompt_report,
        )
        if target == QueryIntent.SUPPLY and workflow_version != WorkflowVersion.V1:
            output = ensure_supply_aggregated_stock_facts(
                output,
                replace_summary=True,
            )
        payload = _attach_tool_report(_diagnostic_payload(output), context)
    except ToolPolicyViolation as exc:
        context.last_tool_report = exc.tool_report
        payload = _attach_tool_report(_diagnostic_payload(exc.output), context)
        runtime_failure = {"target_agent": target.value, **exc.runtime_failure()}
    except OutputGuardrailTripwireTriggered as exc:
        violations = _extract_guardrail_violations(exc)
        agent_output = exc.guardrail_result.agent_output
        payload = _diagnostic_payload(agent_output)
        if not payload.get("summary"):
            payload["summary"] = _coerce_output(agent_output)
        payload = _attach_tool_report(payload, context)
        payload.setdefault("tables", [])
        payload.setdefault("findings", [])
        payload.setdefault("recommendations", [])
        payload.setdefault("data_gaps", [])
        payload.setdefault("cross_agent_requests", [])
        runtime_failure = {"kind": "guardrail", "violations": violations}
    except Exception as exc:
        payload = {
            "summary": "",
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [
                {
                    "metric": f"{target.value} agent runtime",
                    "reason": f"Агент {target.value} не завершил проверку: {type(exc).__name__}.",
                    "impact": "Проверка останется ограничением финального ответа.",
                }
            ],
            "cross_agent_requests": [],
        }
        runtime_failure = {"kind": "runtime", "error_type": type(exc).__name__, "message": str(exc)}

    payload = _attach_prompt_report(payload, context)
    original_gap = (
        _find_agent_run_gap_safely(question.run_id, (question.verification_goal or {}).get("gap_id"))
        if round_name == "round_3"
        else None
    )
    if round_name == "round_3":
        payload = _enforce_gap_resolution_fact_policy(payload, original_gap)
    response = save_agent_run_response(
        question_id=question.id,
        agent_name=target.value,
        status="failed" if runtime_failure else "completed",
        summary=str(payload.get("summary") or ""),
        raw_output=payload,
        has_business_evidence=False if runtime_failure else _payload_has_business_evidence(payload),
        runtime_failure=runtime_failure,
    )
    gaps = _gap_entries_from_payload(
        payload=payload,
        source_agent=target.value,
        blocking=True,
    )
    if round_name == "round_3":
        gaps = [
            gap
            for gap in gaps
            if should_persist_gap_from_gap_resolution(gap, original_gap)
        ]
    if gaps:
        save_agent_run_gaps(run_id=question.run_id, response_id=response.id, gaps=gaps)
    gap_resolution_status: str | None = None
    if round_name == "round_3":
        gap_resolution_status = classify_gap_resolution_response(
            payload,
            original_gap=original_gap,
            runtime_failure=runtime_failure,
        )
        if gap_resolution_status in {"resolved", "accepted_limitation"}:
            _update_agent_run_gap_status_safely(
                (question.verification_goal or {}).get("gap_id"),
                gap_resolution_status,
            )
    update_agent_run_question_status(
        question_id=question.id,
        status="failed" if runtime_failure else "completed",
    )
    context.add_decision_trace(
        "secondary_agent_completed",
        round=round_name,
        target_agent=target.value,
        question=question.question,
        status="failed" if runtime_failure else "completed",
        duration_ms=int((time.monotonic() - started_at) * 1000),
        details={"has_business_evidence": _payload_has_business_evidence(payload)},
    )
    return {
        "round": round_name,
        "target_agent": target.value,
        "question": question.question,
        "reason": question.reason,
        "output": payload,
        **({"gap_resolution_status": gap_resolution_status} if gap_resolution_status else {}),
        **({"runtime_failure": runtime_failure} if runtime_failure else {}),
    }


def _attach_deterministic_manager_candidates(
    synthesis_package: dict[str, Any],
    *,
    run_id: int,
    user_query: str | None = None,
    query_plan: dict[str, Any] | None = None,
    include_advisory: bool = True,
) -> dict[str, Any]:
    """Add source-derived candidates without turning an unavailable source into advice."""

    package = dict(synthesis_package)
    if query_plan:
        package["query_plan"] = dict(query_plan)
        allowed_action_families = list(
            dict.fromkeys(
                str(item).strip()
                for item in list(query_plan.get("allowed_action_families") or [])
                if str(item).strip()
            )
        )
        if allowed_action_families:
            package["requested_action_families"] = allowed_action_families
    effective_query = str(
        user_query or synthesis_package.get("original_user_query") or ""
    )
    try:
        source_snapshot = load_manager_candidate_snapshot()
        explicit_targets = _query_plan_targets(query_plan)
        # In a portfolio plan, explicit targets are a focus overlay for a
        # deeper check, not an exclusive filter for the action universe.
        # sku_list/single_sku keep the historical exclusive semantics.
        candidate_filter_targets = (
            []
            if str((query_plan or {}).get("scope") or "") == "portfolio"
            else explicit_targets
        )
        build_result = build_manager_action_candidates(
            source_snapshot,
            requested_targets=candidate_filter_targets,
        )
    except Exception as exc:
        logger.exception(
            "manager_candidate_generation_failed run_id=%s error=%s",
            run_id,
            type(exc).__name__,
        )
        source_gap = {
            "gap_id": "manager-candidate-source-unavailable",
            "status": "open",
            "metric": "manager_candidate_snapshot",
            "reason": "Не удалось прочитать обязательный снимок данных для расчёта действий.",
            "severity": "blocked",
        }
        package["blocking_gaps"] = [
            *list(package.get("blocking_gaps") or []),
            source_gap,
        ]
        package["manager_candidate_diagnostics"] = [
            {
                "code": "candidate_source_error",
                "error_type": type(exc).__name__,
            }
        ]
        return package

    generated = list(getattr(build_result, "candidates", None) or [])
    ignored_untrusted_count = len(list(package.get("manager_action_candidates") or []))
    unique_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Synthesis/LLM payloads may formulate hypotheses, but only candidates made
    # by deterministic source code are allowed into the executable contract.
    for candidate in generated:
        payload = dict(candidate) if isinstance(candidate, dict) else candidate
        fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_candidates.append(payload)

    package["manager_action_candidates"] = unique_candidates
    package["manager_candidate_evidence"] = list(
        getattr(build_result, "evidence_facts", None) or []
    )
    package["manager_candidate_diagnostics"] = [
        *list(getattr(build_result, "diagnostics", None) or []),
        *(
            [
                {
                    "code": "untrusted_manager_candidates_ignored",
                    "count": ignored_untrusted_count,
                }
            ]
            if ignored_untrusted_count
            else []
        ),
    ]
    package["manager_candidate_source"] = {
        "schemaVersion": source_snapshot.get("schemaVersion"),
        "retrievedAt": source_snapshot.get("retrievedAt"),
        "sourceTables": list(source_snapshot.get("sourceTables") or []),
    }
    if include_advisory and should_build_manager_advisory(
        effective_query,
        query_plan=query_plan,
    ):
        try:
            with observation_scope(
                name="manager-advisory",
                as_type="chain",
                input={
                    "run_id": run_id,
                    "focus_targets": explicit_targets,
                    "candidate_filter_targets": candidate_filter_targets,
                },
                metadata={"run_id": run_id, "query_mode": "decision"},
            ) as observation:
                advisory = build_manager_advisory(
                    effective_query,
                    query_plan=query_plan,
                    candidate_snapshot=source_snapshot,
                )
                package["manager_advisory"] = advisory
                update_observation(observation, output=advisory)
        except Exception as exc:
            logger.exception(
                "manager_advisory_failed run_id=%s error=%s",
                run_id,
                type(exc).__name__,
            )
            package["manager_advisory"] = {
                "schemaVersion": "manager-advisory-v1",
                "mode": "shadow",
                "requestedTargets": explicit_targets,
                "forecasts": [],
                "dailyComparisons": [],
                "priceSimulations": [],
                "errors": [
                    {
                        "stage": "manager_advisory",
                        "error_type": type(exc).__name__,
                    }
                ],
            }
    return package


def _manager_primary_failure_fallback_allowed(
    *,
    trace: Any,
    context: EcomRunContext,
    user_query: str,
) -> bool:
    """Allow a DB-backed manager decision when only the primary LLM is unavailable."""

    settings = get_settings()
    return bool(
        getattr(settings, "manager_decision_enabled", False)
        and _trace_workflow_version(trace) == V2_OPTIMIZED_WORKFLOW_VERSION
        and context.workflow_version == WorkflowVersion.V2_OPTIMIZED
        and _trace_query_mode(trace) == QueryMode.DECISION.value
        and context.query_mode == QueryMode.DECISION
        and _is_manager_portfolio_request(query_plan=_trace_query_plan(trace))
    )


def _compose_manager_answer_markdown(
    *,
    decision_markdown: str,
    advisory_markdown: str,
    advisory: Mapping[str, Any] | None,
    query_plan: Mapping[str, Any] | None,
) -> str:
    """Use one canonical renderer for plans with an explicit action scope."""

    if not advisory_markdown:
        return decision_markdown
    if list((query_plan or {}).get("allowed_action_families") or []):
        source = dict(advisory or {})
        optimization = dict(source.get("portfolioOptimization") or {})
        has_business_artifact = bool(
            list(source.get("forecasts") or [])
            or list(source.get("dailyComparisons") or [])
            or list(source.get("priceSimulations") or [])
            or list(optimization.get("scenarios") or [])
            or list(optimization.get("priorityRecommendations") or [])
        )
        if not has_business_artifact or validate_scenario_completeness(
            advisory_markdown,
            query_plan,
        ):
            return decision_markdown
        return advisory_markdown
    optimization_complete = bool(
        (dict(advisory or {}).get("portfolioOptimization") or {}).get("complete")
    )
    return (
        f"{advisory_markdown}\n\n{decision_markdown}"
        if optimization_complete
        else f"{decision_markdown}\n\n{advisory_markdown}"
    )


def _execute_manager_primary_failure_fallback(
    *,
    trace: Any,
    question: AgentRunQuestionRecord,
    context: EcomRunContext,
    intent: QueryIntent,
    user_query: str,
    exc: Exception,
) -> None:
    """Complete a manager run from deterministic source data, without LLM advice.

    The empty package is intentional: the only facts and candidates admitted to
    the decision are produced by ``load_manager_candidate_snapshot`` and
    ``build_manager_action_candidates``. Missing/invalid source data therefore
    becomes ``needs_data``/``blocked`` in the normal ManagerDecision hard gate.
    """

    runtime_failure = {
        "target_agent": intent.value,
        "kind": (
            "guardrail"
            if isinstance(exc, OutputGuardrailTripwireTriggered)
            else "runtime"
        ),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "fallback": "deterministic_manager_candidate_snapshot",
        **(
            {"violations": _extract_guardrail_violations(exc)}
            if isinstance(exc, OutputGuardrailTripwireTriggered)
            else {}
        ),
    }
    manager_package = _attach_deterministic_manager_candidates(
        {
            "run_id": trace.id,
            "workflow_version": WorkflowVersion.V2_OPTIMIZED.value,
            "query_mode": QueryMode.DECISION.value,
            "original_user_query": user_query,
            "confirmed_facts": [],
            "blocking_gaps": [],
            "accepted_limitations": [],
            "non_blocking_gaps": [],
            "resolved_gaps": [],
            "sku_evidence_matrix": [],
            "domain_coverage": {},
            "evidence_quality": {
                "decision_source": "deterministic_manager_candidate_snapshot",
                "llm_primary_used": False,
            },
            "action_safety_checks": [],
            "runtime_failures": [runtime_failure],
        },
        run_id=trace.id,
        user_query=user_query,
        query_plan=_trace_query_plan(trace),
        include_advisory=True,
    )
    manager_decision_result = run_manager_decision_workflow(
        manager_package,
        user_query=user_query,
        query_mode=QueryMode.DECISION,
    )

    from agi_ecom_app.decision_simulation.service import (
        maybe_schedule_decision_simulation,
    )

    manager_decision_result = maybe_schedule_decision_simulation(
        manager_decision_result,
        chat_id=trace.chat_id,
        run_id=trace.id,
        source_key=f"agent-run:{trace.id}",
    )
    decision_payload = manager_decision_result.decision.model_dump(
        mode="json",
        by_alias=True,
    )
    output_text = manager_decision_result.rendered_markdown
    advisory_markdown = render_manager_advisory(
        manager_package.get("manager_advisory"),
        manager_decision=decision_payload,
        query_plan=_trace_query_plan(trace),
    )
    output_text = _compose_manager_answer_markdown(
        decision_markdown=output_text,
        advisory_markdown=advisory_markdown,
        advisory=manager_package.get("manager_advisory"),
        query_plan=_trace_query_plan(trace),
    )
    fallback_payload = _attach_prompt_report(
        {
            "summary": output_text,
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [],
            "cross_agent_requests": [],
            "manager_decision": decision_payload,
            "manager_advisory": manager_package.get("manager_advisory") or {},
            "_primary_runtime_failure": runtime_failure,
            "_manager_primary_fallback": True,
        },
        context,
    )
    _persist_agent_run_response_safely(
        run_id=trace.id,
        question_record=question,
        agent_name="manager_decision_fallback",
        payload=fallback_payload,
        blocking_gaps=False,
    )
    metadata = _trace_metadata(context)
    metadata.update(
        {
            "primary_agent": _agent_tool_name(intent),
            "workflow_version": _trace_workflow_version(trace),
            "prompt_version": str(
                (trace.summary or {}).get("prompt_version")
                or _trace_workflow_version(trace)
            ),
            "query_mode": _trace_query_mode(trace),
            "manager_decision": decision_payload,
            "manager_decision_status": manager_decision_result.decision.status,
            "manager_decision_schema_version": (
                manager_decision_result.decision.schema_version
            ),
            "manager_primary_fallback": {
                "used": True,
                "source": "deterministic_manager_candidate_snapshot",
                "primary_llm_error_type": type(exc).__name__,
                "actions_executed": False,
            },
            "runtime_failures": [runtime_failure],
            "blocking_gaps": list(manager_package.get("blocking_gaps") or []),
            "evidence_quality": dict(manager_package.get("evidence_quality") or {}),
            "manager_candidate_source": dict(
                manager_package.get("manager_candidate_source") or {}
            ),
            "manager_candidate_diagnostics": list(
                manager_package.get("manager_candidate_diagnostics") or []
            ),
            "manager_advisory": manager_package.get("manager_advisory") or {},
        }
    )
    result = _persist_assistant_turn(
        chat_id=trace.chat_id,
        output_text=output_text,
        intent=intent,
        client_message_id=trace.client_message_id,
        metadata=metadata,
    )
    persist_manager_decision_bundle(
        manager_decision_result.decision,
        chat_id=trace.chat_id,
        run_id=trace.id,
        source_key=f"agent-run:{trace.id}",
        # A fallback is a fail-closed recovery path.  Publish the durable
        # decision, but never expose approvable experiments before the parent
        # question/round/trace transaction commits.  Enforced simulations can
        # project experiments later through their terminal reconciler.
        create_experiments=False,
    )
    context.add_decision_trace(
        "manager_primary_llm_fallback",
        target_agent=intent.value,
        status="completed",
        details={
            "error_type": type(exc).__name__,
            "decision_id": manager_decision_result.decision.decision_id,
            "decision_status": manager_decision_result.decision.status,
            "actions_executed": False,
        },
    )
    context.add_decision_trace(
        "run_completed",
        target_agent=intent.value,
        status="completed",
        details={"assistant_message_seq": result.assistant_message_seq},
    )
    trace_summary = _agent_run_trace_summary(
        context=context,
        metadata=metadata,
        status="completed",
    )
    complete_agent_run_trace(
        trace_id=trace.id,
        assistant_message_seq=result.assistant_message_seq,
        status="completed",
        events=list(context.decision_trace),
        summary={
            **(trace.summary or {}),
            **trace_summary,
            "status": "completed",
            "manager_primary_fallback": metadata["manager_primary_fallback"],
            "manager_decision": decision_payload,
            "manager_decision_status": manager_decision_result.decision.status,
            "manager_advisory": manager_package.get("manager_advisory") or {},
            "runtime_failures": [runtime_failure],
        },
        primary_agent=trace.primary_agent,
        question_id=question.id,
        round_id=question.round_id,
    )
    logger.warning(
        "manager_primary_llm_fallback_completed "
        "run_id=%s question_id=%s decision_id=%s status=%s error=%s",
        trace.id,
        question.id,
        manager_decision_result.decision.decision_id,
        manager_decision_result.decision.status,
        type(exc).__name__,
    )


def _execute_synthesis_question(
    *,
    trace_id: int,
    question: AgentRunQuestionRecord,
    context: EcomRunContext,
    intent: QueryIntent,
    user_query: str,
) -> None:
    trace = get_agent_run_trace_by_id(trace_id)
    if trace is None:
        raise LookupError(trace_id)
    persisted = _load_persisted_evidence_safely(trace_id)
    if not persisted:
        raise AgentRuntimeError("Persisted evidence package is empty.")
    summary = dict(trace.summary or {})
    action_safety_checks = list(summary.get("action_safety_checks") or [])
    action_safety_checks = _finalize_action_safety_checks(
        action_safety_checks,
        persisted.get("followup_evidence", []),
    )
    open_gaps = _open_blocking_gap_payloads_safely(
        trace_id,
        statuses=("open", "accepted_limitation"),
    )
    synthesis_package = _build_synthesis_package(
        user_input=user_query,
        primary_agent=intent,
        primary_output=persisted["primary_output"],
        round_1_plan=summary.get("round_1_plan") or {},
        round_1_evidence=persisted.get("round_1_evidence", []),
        round_2_plan=summary.get("round_2_plan") or {},
        followup_evidence=persisted.get("followup_evidence", []),
        action_safety_checks=action_safety_checks,
        persisted_blocking_gaps=open_gaps,
        query_plan=context.query_plan,
        deterministic_sku_metric_matrix=build_query_plan_inventory_matrix(
            context.query_plan
        ),
    )
    workflow_version = context.workflow_version
    synthesis_input_package = synthesis_package
    synthesis_prompt_report: dict[str, Any] | None = None
    if workflow_version != WorkflowVersion.V1:
        synthesis_input_package, compact_report = compact_synthesis_package(
            {
                **synthesis_package,
                "workflow_version": workflow_version.value,
                "query_mode": context.query_mode.value,
                "original_user_query": user_query,
            }
        )
        synthesis_prompt_report = compact_report.model_dump(mode="json")
    manager_enabled = bool(get_settings().manager_decision_enabled)
    manager_decision_result = None
    is_manager_decision = (
        manager_enabled
        and workflow_version == WorkflowVersion.V2_OPTIMIZED
        and context.query_mode == QueryMode.DECISION
        and _is_manager_portfolio_request(query_plan=context.query_plan)
    )
    prompt_v2_quality_violations: list[str] = []
    prompt_v2_repair_attempted = False
    facts_only_after_failed_repair = False
    synthesis_runtime_failure: dict[str, Any] | None = None
    if is_manager_decision:
        synthesis_package = _attach_deterministic_manager_candidates(
            synthesis_package,
            run_id=trace_id,
            user_query=user_query,
            query_plan=context.query_plan,
        )
        manager_package = {
            **synthesis_package,
            "run_id": trace_id,
            "workflow_version": workflow_version.value,
            "query_mode": context.query_mode.value,
            "original_user_query": user_query,
            "query_plan": dict(context.query_plan),
        }
        manager_decision_result = run_manager_decision_workflow(
            manager_package,
            user_query=user_query,
            query_mode=context.query_mode,
        )
        from agi_ecom_app.decision_simulation.service import (
            maybe_schedule_decision_simulation,
        )

        manager_decision_result = maybe_schedule_decision_simulation(
            manager_decision_result,
            chat_id=trace.chat_id,
            run_id=trace_id,
            source_key=f"agent-run:{trace_id}",
        )
        final_output = manager_decision_result.rendered_markdown
        decision_payload = manager_decision_result.decision.model_dump(
            mode="json",
            by_alias=True,
        )
        advisory_markdown = render_manager_advisory(
            synthesis_package.get("manager_advisory"),
            manager_decision=decision_payload,
            query_plan=context.query_plan,
        )
        final_output = _compose_manager_answer_markdown(
            decision_markdown=final_output,
            advisory_markdown=advisory_markdown,
            advisory=synthesis_package.get("manager_advisory"),
            query_plan=context.query_plan,
        )
    else:
        try:
            final_output = _run_agent(
                build_diagnostic_orchestrator_agent,
                user_input=json.dumps(synthesis_input_package, ensure_ascii=False, default=str),
                context=context,
                hooks=build_run_hooks(),
                max_turns=DIAGNOSTIC_SYNTHESIS_MAX_TURNS,
                prompt_report=synthesis_prompt_report,
            )
        except Exception as exc:
            if not _synthesis_runtime_fallback_allowed(exc, workflow_version):
                raise
            synthesis_runtime_failure = {
                "target_agent": "diagnostic_orchestrator",
                "kind": "runtime",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "fallback": "facts_only",
            }
            logger.warning(
                "synthesis_runtime_facts_only_fallback run_id=%s error=%s",
                trace_id,
                type(exc).__name__,
            )
            facts_only_after_failed_repair = True
            final_output = _render_prompt_v2_facts_only_answer(
                user_query,
                synthesis_input_package,
            )
            if context.last_prompt_report is None and synthesis_prompt_report is not None:
                context.last_prompt_report = dict(synthesis_prompt_report)
    if (
        workflow_version != WorkflowVersion.V1
        and not is_manager_decision
        and synthesis_runtime_failure is None
    ):
        prompt_v2_quality_violations = validate_prompt_v2_answer(
            _coerce_output(final_output),
            synthesis_input_package,
        )
        if prompt_v2_quality_violations and _synthesis_repair_allowed(
            workflow_version,
            context.query_mode,
        ):
            prompt_v2_repair_attempted = True
            repair_prompt = build_prompt_v2_repair_prompt(
                _coerce_output(final_output),
                prompt_v2_quality_violations,
                synthesis_input_package,
            )
            try:
                final_output = _run_agent(
                    build_diagnostic_orchestrator_agent,
                    user_input=repair_prompt,
                    context=context,
                    hooks=build_run_hooks(),
                    max_turns=DIAGNOSTIC_SYNTHESIS_MAX_TURNS,
                    prompt_report=synthesis_prompt_report,
                )
                prompt_v2_quality_violations = validate_prompt_v2_answer(
                    _coerce_output(final_output),
                    synthesis_input_package,
                )
            except Exception:
                logger.exception("prompt_v2_synthesis_repair_failed run_id=%s", trace_id)
        if prompt_v2_quality_violations:
            facts_only_after_failed_repair = True
            final_output = _render_prompt_v2_facts_only_answer(
                user_query,
                synthesis_input_package,
            )
    output_text = _coerce_output(final_output)
    from agi_ecom_app.runner import sanitize_user_facing_markdown, validate_final_diagnostic_markdown

    if not is_manager_decision:
        output_text = _prepend_required_report_snapshots(output_text, synthesis_package)
    pre_sanitize_output_text = output_text
    output_text = sanitize_user_facing_markdown(output_text)
    sku_only_display_sanitized = _sku_only_display_was_sanitized(
        pre_sanitize_output_text,
        output_text,
    )
    from agi_ecom_app.runner import apply_query_plan_output_contract

    output_text = apply_query_plan_output_contract(
        output_text,
        synthesis_package,
        context.query_plan,
    )
    output_text, _empty_table_count = remove_empty_markdown_tables(output_text)
    cause_claim_sanitized_output_text = _sanitize_inventory_turnover_cause_claims(
        output_text,
        user_query=user_query,
    )
    inventory_turnover_cause_claim_sanitized = cause_claim_sanitized_output_text != output_text
    output_text = cause_claim_sanitized_output_text
    final_quality_violations = validate_final_diagnostic_markdown(
        output_text,
        user_query=user_query,
    )
    final_quality_violations.extend(
        validate_scenario_completeness(output_text, context.query_plan)
    )
    portfolio_optimization = dict(
        (synthesis_package.get("manager_advisory") or {}).get(
            "portfolioOptimization"
        )
        or {}
    )
    final_quality_violations.extend(
        _portfolio_priority_quality_violations(
            portfolio_optimization,
            context.query_plan,
        )
    )
    if sku_only_display_sanitized:
        final_quality_violations.append("final markdown was sanitized for SKU-only display")
    if inventory_turnover_cause_claim_sanitized:
        final_quality_violations.append("inventory turnover cause overclaim softened")
    final_quality_violations.extend(
        _sku_matrix_quality_violations(
            {
                "rows": synthesis_package.get("sku_evidence_matrix", []),
                "domain_coverage": synthesis_package.get("domain_coverage", {}),
                "called_domains": _build_sku_evidence_matrix(
                    user_input=user_query,
                    primary_agent=intent,
                    primary_payload=_diagnostic_payload(persisted["primary_output"]),
                    round_1_evidence=persisted.get("round_1_evidence", []),
                    followup_evidence=persisted.get("followup_evidence", []),
                ).get("called_domains", []),
            }
        )
    )
    payload = {
        "summary": output_text,
        "tables": [],
        "findings": [],
        "recommendations": [],
        "data_gaps": [],
        "cross_agent_requests": [],
        **(
            {
                "manager_decision": manager_decision_result.decision.model_dump(
                    mode="json", by_alias=True
                ),
                "manager_advisory": synthesis_package.get("manager_advisory") or {},
            }
            if manager_decision_result is not None
            else {}
        ),
        **(
            {
                "_prompt_quality": {
                    "violations": prompt_v2_quality_violations,
                    "repair_attempted": prompt_v2_repair_attempted,
                }
            }
            if workflow_version != WorkflowVersion.V1
            else {}
        ),
        **(
            {"_synthesis_runtime_failure": synthesis_runtime_failure}
            if synthesis_runtime_failure is not None
            else {}
        ),
    }
    payload = _attach_prompt_report(payload, context)
    _persist_agent_run_response_safely(
        run_id=trace_id,
        question_record=question,
        agent_name="diagnostic_orchestrator",
        payload=payload,
        blocking_gaps=False,
    )
    update_agent_run_question_status(question_id=question.id, status="completed")
    update_agent_run_round_status(round_id=question.round_id, status="completed")

    metadata = _trace_metadata(context)
    metadata.update(
        {
            "primary_agent": _agent_tool_name(intent),
            "workflow_version": _trace_workflow_version(trace),
            "prompt_version": str(
                summary.get("prompt_version") or _trace_workflow_version(trace)
            ),
            "query_mode": _trace_query_mode(trace),
            "secondary_agents": _agent_names_for_round(trace_id, "cross_check"),
            "followup_agents": _agent_names_for_followups(trace_id),
            "dependency_plans": [
                plan
                for plan in [
                    summary.get("round_1_plan"),
                    summary.get("round_2_plan"),
                    summary.get("claim_verification_plan"),
                ]
                if plan
            ],
            "max_rounds": MAX_DIAGNOSTIC_ROUNDS,
            "max_total_followup_domain_calls": MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
            "blocking_gaps": synthesis_package.get("blocking_gaps", []),
            "accepted_limitations": synthesis_package.get("accepted_limitations", []),
            "non_blocking_gaps": synthesis_package.get("non_blocking_gaps", []),
            "resolved_gaps": synthesis_package.get("resolved_gaps", []),
            "claim_verification": synthesis_package.get("claim_verification", {}),
            "claim_verification_plan": summary.get("claim_verification_plan") or {},
            "sku_evidence_matrix": synthesis_package.get("sku_evidence_matrix", []),
            "domain_coverage": synthesis_package.get("domain_coverage", {}),
            "evidence_quality": synthesis_package.get("evidence_quality", {}),
            "manager_advisory": synthesis_package.get("manager_advisory") or {},
            "action_safety_checks": action_safety_checks,
            "runtime_failures": [
                *synthesis_package.get("runtime_failures", []),
                *(
                    [synthesis_runtime_failure]
                    if synthesis_runtime_failure is not None
                    else []
                ),
            ],
            "final_quality_violations": final_quality_violations,
            "prompt_v2_quality_violations": prompt_v2_quality_violations,
            "prompt_v2_repair_attempted": prompt_v2_repair_attempted,
            "facts_only_after_failed_repair": facts_only_after_failed_repair,
            "facts_only_after_synthesis_runtime_failure": (
                synthesis_runtime_failure is not None
            ),
            "synthesis_prompt_chars": len(
                json.dumps(synthesis_input_package, ensure_ascii=False, default=str)
            ),
            **(
                {
                    "manager_decision": manager_decision_result.decision.model_dump(
                        mode="json", by_alias=True
                    ),
                    "manager_decision_status": manager_decision_result.decision.status,
                    "manager_decision_schema_version": (
                        manager_decision_result.decision.schema_version
                    ),
                }
                if manager_decision_result is not None
                else {}
            ),
        }
    )
    action_tasks: list[dict[str, Any]] = []
    if manager_decision_result is not None:
        action_tasks = manager_actions_to_action_board(manager_decision_result.decision)
    elif (
        not manager_enabled
        and not facts_only_after_failed_repair
        and not _should_suppress_action_tasks(question, trace)
    ):
        action_tasks = extract_action_tasks_from_agent_output(output_text, output_text)
    if action_tasks:
        metadata[ACTION_TASKS_METADATA_KEY] = action_tasks
        metadata[ACTION_TASKS_SCHEMA_VERSION_KEY] = ACTION_TASKS_SCHEMA_VERSION
    result = _persist_assistant_turn(
        chat_id=trace.chat_id,
        output_text=output_text,
        intent=intent,
        client_message_id=trace.client_message_id,
        metadata=metadata,
    )
    if manager_decision_result is not None:
        from agi_ecom_app.decision_simulation.service import (
            manager_decision_experiments_deferred,
        )

        persist_manager_decision_bundle(
            manager_decision_result.decision,
            chat_id=trace.chat_id,
            run_id=trace_id,
            source_key=f"agent-run:{trace_id}",
            create_experiments=not manager_decision_experiments_deferred(
                manager_decision_result.decision.decision_id
            ),
        )
    if action_tasks:
        _persist_agent_action_tasks_safely(
            chat_id=trace.chat_id,
            assistant_message_seq=result.assistant_message_seq,
            raw_actions=action_tasks,
            intent=intent,
        )
    context.add_decision_trace(
        "run_completed",
        target_agent=intent.value,
        status="completed",
        details={"assistant_message_seq": result.assistant_message_seq},
    )
    trace_summary = _agent_run_trace_summary(
        context=context,
        metadata=metadata,
        final_quality_violations=final_quality_violations,
        status="completed",
    )
    complete_agent_run_trace(
        trace_id=trace_id,
        assistant_message_seq=result.assistant_message_seq,
        status="completed",
        events=list(context.decision_trace),
        summary={**summary, **trace_summary, "status": "completed"},
        primary_agent=trace_summary.get("primary_agent") or _agent_tool_name(intent),
    )


def _resolve_terminal_action_output(
    *,
    trace: Any,
    question: AgentRunQuestionRecord,
    context: EcomRunContext,
) -> tuple[Any, Any | None, str]:
    task_snapshot = _load_terminal_action_task_snapshot(trace)
    try:
        forecast_output = run_forecast_workflow(
            task_snapshot=task_snapshot,
            today=getattr(context, "today", None),
        )
    except Exception:
        logger.exception("forecast_workflow_failed; falling back to terminal action executor")
        forecast_output = None
    if forecast_output is not None:
        return forecast_output, None, "forecast_orchestrator"

    handler_outcome = execute_action_handler(task_snapshot, today=context.today)
    if handler_outcome is None:
        return (
            _run_agent(
                build_action_executor_agent,
                user_input=question.question,
                context=context,
                hooks=build_run_hooks(),
                max_turns=12,
            ),
            None,
            "action_executor",
        )

    formatter_input = build_action_result_formatter_input(task_snapshot, handler_outcome)
    formatted_output = _run_agent(
        build_action_result_formatter_agent,
        user_input=formatter_input,
        context=context,
        hooks=build_run_hooks(),
        max_turns=6,
    )
    if isinstance(formatted_output, ActionExecutionResult):
        final_output = merge_formatted_action_result(
            handler_outcome.draft_result,
            formatted_output,
        )
    else:
        final_output = handler_outcome.draft_result
    return final_output, handler_outcome, "action_result_formatter"


def _load_terminal_action_task_snapshot(trace: Any) -> dict[str, Any] | None:
    summary = dict(getattr(trace, "summary", {}) or {})
    source_execution_id = summary.get("source_execution_id")
    if source_execution_id is None:
        return None
    execution = get_agent_action_execution(int(source_execution_id))
    if execution is None or not isinstance(execution.task_snapshot, dict):
        return None
    return dict(execution.task_snapshot)


def _execute_terminal_action_question(
    *,
    trace_id: int,
    question: AgentRunQuestionRecord,
    context: EcomRunContext,
    intent: QueryIntent,
) -> None:
    trace = get_agent_run_trace_by_id(trace_id)
    if trace is None:
        raise LookupError(trace_id)
    final_output, handler_outcome, agent_name = _resolve_terminal_action_output(
        trace=trace,
        question=question,
        context=context,
    )
    payload = _diagnostic_payload(final_output)
    if isinstance(payload, dict) and handler_outcome is not None:
        payload["handler"] = handler_outcome.handler
        payload["data_version"] = handler_outcome.data_version
        payload["evidence_package"] = handler_outcome.evidence_package
        payload.setdefault("summary", handler_outcome.draft_result.summary)
        payload.setdefault("decisionStatus", handler_outcome.draft_result.decisionStatus)
    _persist_agent_run_response_safely(
        run_id=trace_id,
        question_record=question,
        agent_name=agent_name,
        payload=payload,
        blocking_gaps=False,
        gap_filter=lambda _gap: False,
    )
    update_agent_run_question_status(question_id=question.id, status="completed")
    update_agent_run_round_status(round_id=question.round_id, status="completed")

    output_text = _coerce_output(final_output)
    metadata = _trace_metadata(context)
    metadata.update(
        {
            "primary_agent": agent_name,
            EXECUTION_MODE_METADATA_KEY: TERMINAL_ACTION_EXECUTION_MODE,
            SUPPRESS_ACTION_TASKS_METADATA_KEY: True,
            "source_chat_id": (trace.summary or {}).get("source_chat_id"),
            "source_execution_id": (trace.summary or {}).get("source_execution_id"),
        }
    )
    result = _persist_assistant_turn(
        chat_id=trace.chat_id,
        output_text=output_text,
        intent=intent,
        client_message_id=trace.client_message_id,
        metadata=metadata,
    )
    trace_summary = _agent_run_trace_summary(
        context=context,
        metadata=metadata,
        status="completed",
    )
    if isinstance(payload, dict):
        if isinstance(final_output, ForecastDecisionResult):
            trace_summary["forecast_decision_result"] = payload
            if isinstance(payload.get("plan"), dict):
                trace_summary["forecast_plan"] = dict(payload["plan"])
            resolved_targets = payload.get("resolvedTargets")
            if isinstance(resolved_targets, list):
                trace_summary["resolved_targets"] = list(resolved_targets)
            evidence_payload = payload.get("evidence")
            if isinstance(evidence_payload, dict):
                trace_summary["evidence_bundle"] = dict(evidence_payload)
            debug_data = payload.get("debugData")
            if isinstance(debug_data, dict):
                scenario_outputs = debug_data.get("scenario_outputs")
                if isinstance(scenario_outputs, dict):
                    trace_summary["scenario_outputs"] = dict(scenario_outputs)
            ui_card = payload.get("uiCard")
            if isinstance(ui_card, dict):
                trace_summary["forecast_ui_card"] = dict(ui_card)
        else:
            trace_summary["action_execution_result"] = payload
        trace_summary["decision_status"] = payload.get("decisionStatus")
        trace_summary["review_summary"] = payload.get("summary") or payload.get("finalRecommendation")
        if handler_outcome is not None:
            trace_summary["handler"] = handler_outcome.handler
            trace_summary["data_version"] = handler_outcome.data_version
            trace_summary["evidence_package"] = handler_outcome.evidence_package
    complete_agent_run_trace(
        trace_id=trace_id,
        assistant_message_seq=result.assistant_message_seq,
        status="completed",
        events=list(context.decision_trace),
        summary={**(trace.summary or {}), **trace_summary, "status": "completed"},
        primary_agent=agent_name,
    )


def _advance_agent_run_locked(run_id: int) -> bool:
    trace = get_agent_run_trace_by_id(run_id)
    if trace is None or trace.status in RUN_TERMINAL_STATUSES:
        return False
    rounds = list_agent_run_rounds(run_id)
    if _has_active_questions(rounds):
        return False
    if _mark_finished_round_statuses(rounds):
        return True
    if _primary_failed(rounds):
        _fail_run(
            run_id,
            _round_failure_reason(
                _round_by_type(rounds, "primary"),
                "primary question failed",
            ),
        )
        return False

    terminal_round = _round_by_type(rounds, "terminal_action")
    if terminal_round is not None:
        if _round_failed(terminal_round):
            _fail_run(
                run_id,
                _round_failure_reason(terminal_round, "terminal action question failed"),
            )
            return False
        if _round_done(terminal_round):
            return False

    primary_round = _round_by_type(rounds, "primary")
    if not _round_done(primary_round):
        return False
    if _trace_workflow_version(trace) == V2_OPTIMIZED_WORKFLOW_VERSION:
        return _advance_v2_optimized_run(run_id, trace, rounds)
    if _round_by_type(rounds, "cross_check") is None:
        return _create_round_1(run_id, trace)

    round_1 = _round_by_type(rounds, "cross_check")
    if not _round_done(round_1):
        return False
    if _round_by_type(rounds, "followup") is None:
        return _create_round_2(run_id, trace)

    round_2 = _round_by_type(rounds, "followup")
    if not _round_done(round_2):
        return False
    if _round_by_type(rounds, "gap_resolution") is None:
        return _create_gap_resolution_round(run_id, trace)

    round_3 = _round_by_type(rounds, "gap_resolution")
    if not _round_done(round_3):
        return False
    user_query = _trace_user_query(trace)
    if _is_inventory_turnover_causal_query(user_query) and _round_by_type(rounds, "claim_verification") is None:
        return _create_claim_verification_round(run_id, trace)

    claim_round = _round_by_type(rounds, "claim_verification")
    if claim_round is not None and not _round_done(claim_round):
        return False
    if _round_by_type(rounds, "synthesis") is None:
        return _create_synthesis_round(run_id)

    synthesis = _round_by_type(rounds, "synthesis")
    if _round_failed(synthesis):
        _fail_run(
            run_id,
            _round_failure_reason(synthesis, "synthesis question failed"),
        )
    return False


def _advance_v2_optimized_run(
    run_id: int,
    trace: Any,
    rounds: list[dict[str, Any]],
) -> bool:
    query_mode = _trace_query_mode(trace)

    if query_mode in {"lookup", "report"}:
        return _advance_to_synthesis(run_id, rounds)

    cross_limit = (
        MAX_OPTIMIZED_DECISION_CROSS_CHECKS
        if query_mode == "decision"
        else MAX_OPTIMIZED_CROSS_CHECKS
    )
    if _round_by_type(rounds, "cross_check") is None:
        available = _remaining_optimized_question_slots(
            rounds,
            reserve=OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
        )
        return _create_round_1(run_id, trace, limit=min(cross_limit, available))

    cross_round = _round_by_type(rounds, "cross_check")
    if not _round_done(cross_round):
        return False

    if query_mode == "decision":
        if _round_by_type(rounds, "followup") is None:
            available = _remaining_optimized_question_slots(
                rounds,
                reserve=OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
            )
            return _create_round_2(
                run_id,
                trace,
                limit=min(MAX_OPTIMIZED_SAFETY_CHECKS, available),
                action_safety_only=True,
            )
        followup_round = _round_by_type(rounds, "followup")
        if not _round_done(followup_round):
            return False

    if _round_by_type(rounds, "gap_resolution") is None:
        available = _remaining_optimized_question_slots(
            rounds,
            reserve=OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
        )
        return _create_gap_resolution_round(
            run_id,
            trace,
            limit=min(MAX_OPTIMIZED_GAP_RESOLUTIONS, available),
            deduplicate=True,
        )

    gap_round = _round_by_type(rounds, "gap_resolution")
    if not _round_done(gap_round):
        return False

    user_query = _trace_user_query(trace)
    if (
        query_mode == "diagnostic"
        and _is_inventory_turnover_causal_query(user_query)
        and _round_by_type(rounds, "claim_verification") is None
    ):
        available = _remaining_optimized_question_slots(
            rounds,
            reserve=OPTIMIZED_SYNTHESIS_QUESTION_RESERVE,
        )
        return _create_claim_verification_round(
            run_id,
            trace,
            limit=min(MAX_OPTIMIZED_CLAIM_VERIFICATIONS, available),
        )

    claim_round = _round_by_type(rounds, "claim_verification")
    if claim_round is not None and not _round_done(claim_round):
        return False
    return _advance_to_synthesis(run_id, rounds)


def _advance_to_synthesis(run_id: int, rounds: list[dict[str, Any]]) -> bool:
    if _round_by_type(rounds, "synthesis") is None:
        return _create_synthesis_round(run_id)
    synthesis = _round_by_type(rounds, "synthesis")
    if _round_failed(synthesis):
        _fail_run(
            run_id,
            _round_failure_reason(synthesis, "synthesis question failed"),
        )
    return False


def _remaining_optimized_question_slots(
    rounds: list[dict[str, Any]],
    *,
    reserve: int,
) -> int:
    question_count = sum(
        len(round_payload.get("questions", []))
        for round_payload in rounds
    )
    return max(
        0,
        MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS - reserve - question_count,
    )


def _manager_domain_coverage_plan(
    *,
    primary_agent: QueryIntent,
    max_secondary_agents: int,
) -> dict[str, Any] | None:
    settings = get_settings()
    if not getattr(settings, "manager_decision_enabled", False):
        return None
    items = []
    for index, target in enumerate(QueryIntent):
        if target == primary_agent:
            continue
        items.append(
            {
                "target_agent": target.value,
                "reason": (
                    "ManagerDecision требует единого evidence snapshot по всем пяти "
                    "бизнес-доменам до расчёта экономики и приоритета."
                ),
                "question": MANAGER_DOMAIN_QUESTIONS[target],
                "priority": 120 - index,
                "source": "manager_decision_domain_contract",
                "period": None,
                "vendor_codes": [],
                "nm_ids": [],
                "required_checks": MANAGER_DOMAIN_REQUIRED_CHECKS[target],
                "required_tools": MANAGER_DOMAIN_REQUIRED_TOOLS[target],
            }
        )
    limit = max(0, max_secondary_agents)
    return {
        "primary_agent": primary_agent.value,
        "recommended_secondary_agents": items[:limit],
        "skipped_agents": items[limit:],
        "max_secondary_agents": limit,
    }


def _create_round_1(run_id: int, trace: Any, *, limit: int | None = None) -> bool:
    persisted = _load_persisted_evidence_safely(run_id)
    if not persisted:
        return False
    user_query = _trace_user_query(trace)
    intent = _coerce_intent(trace.intent, determine_query_intent(user_query))
    primary_output = persisted["primary_output"]
    round_limit = ROUND_1_SECONDARY_LIMIT if limit is None else max(0, int(limit))
    agent_requested_plan = build_cross_domain_dependency_plan(
        user_query=user_query,
        primary_agent=intent.value,
        primary_summary=_diagnostic_summary(primary_output),
        findings=_diagnostic_findings(primary_output),
        data_gaps=_diagnostic_data_gaps(primary_output),
        cross_agent_requests=_diagnostic_cross_agent_requests(primary_output),
        max_secondary_agents=round_limit,
    )
    coverage_policy_plan = _inventory_turnover_coverage_plan(
        user_input=user_query,
        primary_agent=intent,
        existing_items=_plan_items(agent_requested_plan),
        max_secondary_agents=round_limit,
        query_plan=_trace_query_plan(trace),
    )
    manager_coverage_plan = (
        _manager_domain_coverage_plan(
            primary_agent=intent,
            max_secondary_agents=round_limit,
        )
        if (
            _trace_query_mode(trace) == "decision"
            and _is_manager_portfolio_request(query_plan=_trace_query_plan(trace))
        )
        else None
    )
    plan = (
        _merge_dependency_plans(
            primary_agent=intent,
            plans=[manager_coverage_plan, coverage_policy_plan, agent_requested_plan],
            max_secondary_agents=round_limit,
        )
        or {
            "primary_agent": intent.value,
            "recommended_secondary_agents": [],
            "skipped_agents": [],
            "max_secondary_agents": round_limit,
        }
    )
    items = _plan_items(plan)[:round_limit]
    _merge_trace_summary(trace.id, {"round_1_plan": plan})
    round_record = create_agent_run_round(
        run_id=run_id,
        round_number=1,
        round_type="cross_check",
        status="pending" if items else "skipped",
    )
    for item in items:
        question = _create_question_from_item(
            run_id=run_id,
            round_id=round_record.id,
            item=item,
            source_agent=intent.value,
            round_name="round_1",
        )
        enqueue_agent_question(question.id)
    if not items:
        update_agent_run_round_status(round_id=round_record.id, status="skipped")
    return True


def _create_round_2(
    run_id: int,
    trace: Any,
    *,
    limit: int | None = None,
    action_safety_only: bool = False,
) -> bool:
    persisted = _load_persisted_evidence_safely(run_id)
    if not persisted:
        return False
    user_query = _trace_user_query(trace)
    intent = _coerce_intent(trace.intent, determine_query_intent(user_query))
    round_1_evidence = persisted.get("round_1_evidence", [])
    already_asked = [
        _asked_item(evidence["target_agent"], evidence["question"])
        for evidence in round_1_evidence
    ]
    round_1_requests: list[dict[str, Any]] = []
    for evidence in round_1_evidence:
        round_1_requests.extend((evidence.get("output") or {}).get("cross_agent_requests", []))
    requested_limit = ROUND_2_FOLLOWUP_LIMIT if limit is None else max(0, int(limit))
    action_safety_plan, action_safety_checks = _build_action_safety_plan(
        user_input=user_query,
        primary_agent=intent,
        primary_output=persisted["primary_output"],
        round_1_evidence=round_1_evidence,
        max_secondary_agents=requested_limit,
    )
    if action_safety_only:
        action_safety_plan, action_safety_checks = _limit_action_safety_checks(
            action_safety_plan,
            action_safety_checks,
            max_checks=requested_limit,
        )
    remaining_calls = MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS - len(round_1_evidence)
    round_2_limit = max(0, min(requested_limit, remaining_calls))
    cross_request_plan = None
    if round_1_requests and not action_safety_only:
        cross_request_plan = build_cross_domain_dependency_plan(
            user_query="",
            primary_agent=intent.value,
            primary_summary="",
            findings=[],
            data_gaps=[],
            cross_agent_requests=round_1_requests,
            already_asked=already_asked,
            max_secondary_agents=round_2_limit,
        )
    plan = _merge_dependency_plans(
        primary_agent=intent,
        plans=[cross_request_plan, action_safety_plan],
        max_secondary_agents=round_2_limit,
    )
    items = _plan_items(plan or {})[:round_2_limit]
    _merge_trace_summary(trace.id, {"round_2_plan": plan or {}, "action_safety_checks": action_safety_checks})
    round_record = create_agent_run_round(
        run_id=run_id,
        round_number=2,
        round_type="followup",
        status="pending" if items else "skipped",
    )
    for item in items:
        question = _create_question_from_item(
            run_id=run_id,
            round_id=round_record.id,
            item=item,
            source_agent=intent.value,
            round_name="round_2",
        )
        enqueue_agent_question(question.id)
    if not items:
        update_agent_run_round_status(round_id=round_record.id, status="skipped")
    return True


def _limit_action_safety_checks(
    plan: dict[str, Any] | None,
    checks: list[dict[str, Any]],
    *,
    max_checks: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    limit = max(0, int(max_checks))
    selected_checks = list(checks[:limit])
    if not plan:
        return plan, selected_checks

    checks_by_id = {
        str(check.get("check_id")): check
        for check in selected_checks
        if check.get("check_id")
    }
    selected_items: list[dict[str, Any]] = []
    for original in _plan_items(plan):
        item = dict(original)
        selected_ids = [
            str(check_id)
            for check_id in item.get("safety_check_ids", [])
            if str(check_id) in checks_by_id
        ]
        if not selected_ids:
            continue
        item["safety_check_ids"] = selected_ids
        item["required_checks"] = [
            str(checks_by_id[check_id].get("required_check") or "")
            for check_id in selected_ids
            if checks_by_id[check_id].get("required_check")
        ]
        item["question"] = " ".join(
            [
                "Проверь action-safety перед финальными задачами:",
                " ".join(item["required_checks"]),
            ]
        ).strip()
        selected_items.append(item)

    limited_plan = {
        **plan,
        "recommended_secondary_agents": selected_items,
        "max_secondary_agents": min(limit, len(selected_items)),
    }
    return limited_plan, selected_checks


def _create_gap_resolution_round(
    run_id: int,
    trace: Any,
    *,
    limit: int | None = None,
    deduplicate: bool = False,
) -> bool:
    persisted = _load_persisted_evidence_safely(run_id)
    if persisted and not (trace.summary or {}).get("sku_gap_matrix_persisted"):
        user_query = _trace_user_query(trace)
        intent = _coerce_intent(trace.intent, determine_query_intent(user_query))
        matrix = _build_sku_evidence_matrix(
            user_input=user_query,
            primary_agent=intent,
            primary_payload=_diagnostic_payload(persisted["primary_output"]),
            round_1_evidence=persisted.get("round_1_evidence", []),
            followup_evidence=persisted.get("followup_evidence", []),
        )
        _persist_gap_entries_safely(
            run_id=run_id,
            response_id=None,
            gaps=_gap_entries_from_sku_matrix(matrix),
        )
        _merge_trace_summary(trace.id, {"sku_gap_matrix_persisted": True})
    open_gaps = list_agent_run_gaps(
        run_id=run_id,
        status="open",
        blocking=True,
        evidence_statuses=("no_data", "unconfirmed"),
    )
    actionable_open_gaps = [
        gap for gap in open_gaps if not _is_non_actionable_sku_source_gap(gap)
    ]
    if deduplicate:
        actionable_open_gaps = _deduplicate_actionable_gaps(actionable_open_gaps)
    gap_limit = (
        ROUND_3_GAP_RESOLUTION_LIMIT
        if limit is None
        else max(0, int(limit))
    )
    items = [
        _gap_to_plan_item(gap, index)
        for index, gap in enumerate(actionable_open_gaps[:gap_limit], start=1)
    ]
    round_record = create_agent_run_round(
        run_id=run_id,
        round_number=3,
        round_type="gap_resolution",
        status="pending" if items else "skipped",
    )
    for item in items:
        question = _create_question_from_item(
            run_id=run_id,
            round_id=round_record.id,
            item=item,
            source_agent="orchestrator",
            round_name="round_3",
        )
        enqueue_agent_question(question.id)
    if not items:
        update_agent_run_round_status(round_id=round_record.id, status="skipped")
    return True


def _deduplicate_actionable_gaps(gaps: list[Any]) -> list[Any]:
    deduplicated: list[Any] = []
    blocked_by_no_data = {
        _actionable_gap_key(gap)
        for gap in gaps
        if str(getattr(gap, "evidence_status", "")).strip().lower() == "no_data"
    }
    seen: set[tuple[str, str, str, str]] = set()
    for gap in gaps:
        key = _actionable_gap_key(gap)
        if key in blocked_by_no_data or key in seen:
            continue
        seen.add(key)
        deduplicated.append(gap)
    return deduplicated


def _actionable_gap_key(gap: Any) -> tuple[str, str, str, str]:
    target_agent = _intent_from_agent_name(
        getattr(gap, "target_agent", None) or getattr(gap, "source_agent", None)
    ).value
    return (
        target_agent,
        _normalize_gap_component(getattr(gap, "sku", None)),
        _normalize_gap_component(getattr(gap, "metric", None)),
        _gap_period_key(gap),
    )


def _normalize_gap_component(value: Any) -> str:
    return " ".join(
        part
        for part in re.split(r"[^\w]+", str(value or "").casefold())
        if part
    )


def _gap_period_key(gap: Any) -> str:
    explicit_period = getattr(gap, "period", None)
    if explicit_period:
        return _normalize_gap_component(explicit_period)

    corpus = " ".join(
        str(value or "")
        for value in (
            getattr(gap, "metric", None),
            getattr(gap, "reason", None),
        )
    ).casefold()
    dates = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4})\b", corpus)
    if dates:
        return "|".join(dates[:2])
    relative_period = re.search(
        r"\b(?:последн\w*|предыдущ\w*|текущ\w*|прошл\w*)\s+"
        r"(?:\d+\s+)?(?:дн\w*|недел\w*|месяц\w*|квартал\w*)\b",
        corpus,
    )
    return _normalize_gap_component(relative_period.group(0) if relative_period else "")


def _create_claim_verification_round(
    run_id: int,
    trace: Any,
    *,
    limit: int | None = None,
) -> bool:
    persisted = _load_persisted_evidence_safely(run_id)
    if not persisted:
        return False
    user_query = _trace_user_query(trace)
    intent = _coerce_intent(trace.intent, determine_query_intent(user_query))
    if not _is_inventory_turnover_causal_query(user_query):
        return False

    claim_limit = CLAIM_VERIFICATION_LIMIT if limit is None else max(0, int(limit))
    plan = _claim_verification_plan(
        user_input=user_query,
        primary_agent=intent,
        primary_output=persisted["primary_output"],
        round_1_evidence=persisted.get("round_1_evidence", []),
        followup_evidence=persisted.get("followup_evidence", []),
        max_secondary_agents=claim_limit,
    )
    items = _plan_items(plan or {})[:claim_limit]
    _merge_trace_summary(trace.id, {"claim_verification_plan": plan or {}})
    round_record = create_agent_run_round(
        run_id=run_id,
        round_number=CLAIM_VERIFICATION_ROUND_NUMBER,
        round_type="claim_verification",
        status="pending" if items else "skipped",
    )
    for item in items:
        question = _create_question_from_item(
            run_id=run_id,
            round_id=round_record.id,
            item=item,
            source_agent="orchestrator",
            round_name="claim_verification",
        )
        enqueue_agent_question(question.id)
    if not items:
        update_agent_run_round_status(round_id=round_record.id, status="skipped")
    return True


def _create_synthesis_round(run_id: int) -> bool:
    round_record = create_agent_run_round(
        run_id=run_id,
        round_number=SYNTHESIS_ROUND_NUMBER,
        round_type="synthesis",
        status="pending",
    )
    question = create_agent_run_question(
        run_id=run_id,
        round_id=round_record.id,
        source_agent="orchestrator",
        target_agent="diagnostic_orchestrator",
        question="Собери финальный ответ из persisted evidence package.",
        reason="Final synthesis after persisted agent rounds.",
        verification_goal={"round": "synthesis"},
        priority=100,
        status="pending",
    )
    enqueue_agent_question(question.id)
    return True


def _persist_simple_final_answer(
    *,
    trace_id: int,
    context: EcomRunContext,
    final_output: Any,
    intent: QueryIntent,
    client_message_id: str | None,
    question: AgentRunQuestionRecord,
) -> ChatTurnResult:
    output_text = _coerce_output(final_output)
    payload = _diagnostic_payload(final_output)
    trace = get_agent_run_trace_by_id(trace_id)
    if trace is None:
        raise LookupError(trace_id)
    _persist_agent_run_response_safely(
        run_id=trace_id,
        question_record=question,
        agent_name="domain_router",
        payload=payload,
        blocking_gaps=False,
    )
    update_agent_run_question_status(question_id=question.id, status="completed")
    update_agent_run_round_status(round_id=question.round_id, status="completed")
    metadata = _trace_metadata(context)
    action_tasks: list[dict[str, Any]] = []
    if not _should_suppress_action_tasks(question, trace):
        action_tasks = extract_action_tasks_from_agent_output(final_output, output_text)
    if action_tasks:
        metadata[ACTION_TASKS_METADATA_KEY] = action_tasks
        metadata[ACTION_TASKS_SCHEMA_VERSION_KEY] = ACTION_TASKS_SCHEMA_VERSION
    result = _persist_assistant_turn(
        chat_id=trace.chat_id,
        output_text=output_text,
        intent=intent,
        client_message_id=client_message_id,
        metadata=metadata,
    )
    if action_tasks:
        _persist_agent_action_tasks_safely(
            chat_id=trace.chat_id,
            assistant_message_seq=result.assistant_message_seq,
            raw_actions=action_tasks,
            intent=intent,
        )
    trace_summary = _agent_run_trace_summary(
        context=context,
        metadata=metadata,
        status="completed",
    )
    complete_agent_run_trace(
        trace_id=trace_id,
        assistant_message_seq=result.assistant_message_seq,
        status="completed",
        events=list(context.decision_trace),
        summary={**(trace.summary or {}), **trace_summary, "status": "completed"},
        primary_agent=trace.primary_agent,
    )
    return result


def _persist_question_failure(
    *,
    question: AgentRunQuestionRecord,
    exc: Exception,
    prompt_report: dict[str, Any] | None = None,
    blocking: bool = True,
    transition_question: bool = True,
) -> None:
    error = f"{type(exc).__name__}: {exc}"
    failed_question = (
        mark_agent_run_question_failed(question_id=question.id, last_error=error)
        if transition_question
        else question
    )
    raw_output: dict[str, Any]
    runtime_failure: dict[str, Any]
    gap_metric = f"{failed_question.target_agent} runtime"
    gap_reason = error
    if isinstance(exc, OutputGuardrailTripwireTriggered):
        violations = _extract_guardrail_violations(exc)
        agent_output = exc.guardrail_result.agent_output
        raw_output = _diagnostic_payload(agent_output)
        raw_output.setdefault("summary", "")
        raw_output.setdefault("tables", [])
        raw_output.setdefault("findings", [])
        raw_output.setdefault("recommendations", [])
        raw_output.setdefault("data_gaps", [])
        raw_output.setdefault("cross_agent_requests", [])
        raw_output["_guardrail_violations"] = violations
        runtime_failure = {
            "kind": "guardrail",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "violations": violations,
        }
        gap_metric = f"{failed_question.target_agent} output guardrail"
        if violations:
            gap_reason = "; ".join(violations)
    else:
        raw_output = {
            "summary": "",
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [
                {
                    "metric": gap_metric,
                    "reason": error,
                    "impact": "Проверка не завершилась и останется ограничением финального ответа.",
                }
            ],
            "cross_agent_requests": [],
        }
        runtime_failure = {
            "kind": "runtime",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    if prompt_report:
        raw_output["_prompt_report"] = dict(prompt_report)
    response = save_agent_run_response(
        question_id=failed_question.id,
        agent_name=failed_question.target_agent,
        status="failed",
        summary=str(raw_output.get("summary") or ""),
        raw_output=raw_output,
        has_business_evidence=False,
        runtime_failure=runtime_failure,
    )
    save_agent_run_gaps(
        run_id=failed_question.run_id,
        response_id=response.id,
        gaps=[
            {
                "source_agent": failed_question.source_agent,
                "target_agent": failed_question.target_agent,
                "metric": gap_metric,
                "reason": gap_reason,
                "impact": "Проверка не завершилась после retry.",
                "blocking": blocking,
                "status": "open",
                "evidence_status": "no_data",
            }
        ],
    )


def _build_context_from_trace(trace: Any) -> EcomRunContext:
    settings = get_settings()
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    if runtime_spec.missing_env_vars:
        raise AgentRuntimeError(
            "Не настроен ключ активного LLM-профиля "
            f"`{runtime_spec.profile}` ({', '.join(runtime_spec.missing_env_vars)}). "
            "Установи переменную окружения и повтори запрос."
        )
    context_window = get_context_window(
        chat_id=trace.chat_id,
        limit=settings.memory_context_message_limit,
    )
    user_query = _trace_user_query(trace)
    return EcomRunContext(
        user_query=user_query,
        intent=_coerce_intent(trace.intent, determine_query_intent(user_query)),
        today=date.today(),
        db_target=f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
        chat_id=trace.chat_id,
        run_id=trace.id,
        workflow_version=WorkflowVersion(_trace_workflow_version(trace)),
        round_type=RoundType.PRIMARY,
        query_mode=QueryMode(_trace_query_mode(trace)),
        memory_mode=memory_mode_for_round(RoundType.PRIMARY),
        chat_summary=context_window.summary_text,
        recent_messages=context_window.recent_messages,
        query_plan=_trace_query_plan(trace),
    )


def _build_rq_queue(redis_url: str, queue_name: str):
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        raise AgentRuntimeError(
            "Redis/RQ зависимости не установлены. Выполни `pip install -r requirements.txt`."
        ) from exc
    return Queue(queue_name, connection=Redis.from_url(redis_url))


def _build_rq_retry():
    try:
        from rq import Retry
    except ImportError:
        return None
    return Retry(max=1, interval=[5])


def _round_payload_for_question(*, run_id: int, round_id: int) -> dict[str, Any] | None:
    for payload in list_agent_run_rounds(run_id):
        round_record = payload["round"]
        if round_record.id == round_id:
            return {
                "round": round_record,
                "round_type": round_record.round_type,
                "questions": payload.get("questions", []),
            }
    return None


def _round_by_type(rounds: list[dict[str, Any]], round_type: str) -> dict[str, Any] | None:
    for payload in rounds:
        if payload["round"].round_type == round_type:
            return payload
    return None


def _round_done(round_payload: dict[str, Any] | None) -> bool:
    if not round_payload:
        return False
    if round_payload["round"].status in {"completed", "skipped"}:
        return True
    questions = [payload["question"] for payload in round_payload.get("questions", [])]
    return bool(questions) and all(question.status in TERMINAL_QUESTION_STATUSES for question in questions)


def _round_failed(round_payload: dict[str, Any] | None) -> bool:
    if not round_payload:
        return False
    return any(
        payload["question"].status == "failed"
        for payload in round_payload.get("questions", [])
    )


def _round_failure_reason(round_payload: dict[str, Any] | None, fallback: str) -> str:
    if not round_payload:
        return fallback
    for payload in round_payload.get("questions", []):
        question = payload["question"]
        if question.status != "failed":
            continue
        detail = _question_failure_detail(payload)
        return _compact_failure_reason(detail or fallback)
    return fallback


def _question_failure_detail(question_payload: dict[str, Any]) -> str | None:
    question = question_payload["question"]
    if question.last_error:
        return str(question.last_error)
    for response in question_payload.get("responses", []):
        runtime_failure = getattr(response, "runtime_failure", None) or {}
        message = runtime_failure.get("message")
        error_type = runtime_failure.get("error_type")
        if message and error_type:
            return f"{error_type}: {message}"
        if message:
            return str(message)
        if error_type:
            return str(error_type)
    return None


def _compact_failure_reason(reason: str) -> str:
    normalized = " ".join(str(reason or "").split())
    lower = normalized.lower()
    if "insufficient_quota" in lower:
        return (
            "Недостаточно квоты OpenAI API (429 insufficient_quota). "
            "Проверьте billing/лимиты и OPENAI_API_KEY."
        )
    if "rate limit" in lower or "ratelimiterror" in lower or "error code: 429" in lower:
        return "LLM API вернул 429 rate limit. Повторите позже или проверьте лимиты проекта."
    if len(normalized) > MAX_FAILURE_REASON_LENGTH:
        return f"{normalized[:MAX_FAILURE_REASON_LENGTH].rstrip()}..."
    return normalized


def _mark_finished_round_statuses(rounds: list[dict[str, Any]]) -> bool:
    changed = False
    for round_payload in rounds:
        round_record = round_payload["round"]
        if round_record.status not in {"pending", "queued", "running"}:
            continue
        questions = [payload["question"] for payload in round_payload.get("questions", [])]
        if not questions:
            continue
        if all(question.status in TERMINAL_QUESTION_STATUSES for question in questions):
            update_agent_run_round_status(round_id=round_record.id, status="completed")
            changed = True
    return changed


def _has_active_questions(rounds: list[dict[str, Any]]) -> bool:
    for round_payload in rounds:
        for payload in round_payload.get("questions", []):
            if payload["question"].status in {"pending", "queued", "running"}:
                return True
    return False


def _primary_failed(rounds: list[dict[str, Any]]) -> bool:
    return _round_failed(_round_by_type(rounds, "primary"))


def _create_question_from_item(
    *,
    run_id: int,
    round_id: int,
    item: dict[str, Any],
    source_agent: str,
    round_name: str,
) -> AgentRunQuestionRecord:
    return create_agent_run_question(
        run_id=run_id,
        round_id=round_id,
        source_agent=source_agent,
        target_agent=_intent_from_agent_name(item.get("target_agent")).value,
        question=str(item.get("question") or ""),
        reason=str(item.get("reason") or ""),
        verification_goal=_verification_goal_from_item(item, round_name=round_name),
        priority=int(item.get("priority") or 0),
        status="pending",
        parent_question_id=item.get("parent_question_id"),
    )


def _question_to_plan_item(question: AgentRunQuestionRecord) -> dict[str, Any]:
    goal = dict(question.verification_goal or {})
    return {
        "target_agent": question.target_agent,
        "reason": question.reason,
        "question": question.question,
        "priority": question.priority,
        "source": goal.get("source"),
        "round": goal.get("round"),
        "gap_id": goal.get("gap_id"),
        "sku": goal.get("sku"),
        "period": goal.get("period"),
        "vendor_codes": goal.get("vendor_codes", []),
        "nm_ids": goal.get("nm_ids", []),
        "safety_check_ids": goal.get("safety_check_ids", []),
        "required_checks": goal.get("required_checks", []),
        "required_tools": goal.get("required_tools", []),
        "claim_policy": goal.get("claim_policy"),
    }


def _gap_to_plan_item(gap: Any, index: int) -> dict[str, Any]:
    target_agent = _intent_from_agent_name(gap.target_agent or gap.source_agent).value
    sku_text = f" по SKU {gap.sku}" if gap.sku else ""
    return {
        "target_agent": target_agent,
        "reason": gap.reason,
        "question": (
            f"Закрой blocking gap{sku_text}: {gap.metric}. "
            f"Причина пробела: {gap.reason}. Подтверди факт или явно оставь no_data."
        ),
        "priority": max(0, 90 - index),
        "source": "gap_resolution",
        "gap_id": gap.id,
        "sku": gap.sku,
        "period": _gap_period_key(gap) or None,
        "required_checks": [gap.metric],
    }


def _round_name(round_type: str | None) -> str:
    return {
        "cross_check": "round_1",
        "followup": "round_2",
        "gap_resolution": "round_3",
        "claim_verification": "claim_verification",
    }.get(str(round_type), str(round_type or "round"))


def _trace_user_query(trace: Any) -> str:
    summary = dict(trace.summary or {})
    return str(summary.get("user_query") or "")


def _emit_agent_run_result_observation(trace: Any) -> None:
    summary = dict(getattr(trace, "summary", {}) or {})
    raw_observability = summary.get("observability")
    observability = (
        dict(raw_observability) if isinstance(raw_observability, dict) else {}
    )
    if observability.get("result_observation_id"):
        return
    trace_id, parent_observation_id = trace_context_for_agent_run(
        summary=summary,
        run_id=int(trace.id),
    )
    status = str(getattr(trace, "status", summary.get("status") or "unknown"))
    with observation_scope(
        name="agent-run-result",
        as_type="span",
        trace_id=trace_id,
        parent_observation_id=parent_observation_id,
        input={"run_id": trace.id},
        metadata={
            "run_id": trace.id,
            "intent": getattr(trace, "intent", None),
            "status": status,
            "primary_agent": getattr(trace, "primary_agent", None),
            "manager_decision_status": summary.get("manager_decision_status"),
            "workflow_version": summary.get("workflow_version"),
            "prompt_version": summary.get("prompt_version"),
        },
        session_id=getattr(trace, "chat_id", None),
        tags=["agent-run", "result", status],
        version=str(summary.get("workflow_version") or "v1"),
    ) as result_observation:
        run_error = (
            RuntimeError(str(summary.get("error") or "agent run failed"))
            if status == "failed"
            else None
        )
        update_observation(
            result_observation,
            output={"status": status, "summary": summary},
            error=run_error,
        )
    if result_observation is not None:
        observability["result_observation_id"] = str(result_observation.id)
        _merge_trace_summary(trace.id, {"observability": observability})


def _merge_trace_summary(trace_id: int, values: dict[str, Any]) -> None:
    trace = get_agent_run_trace_by_id(trace_id)
    if trace is None:
        raise LookupError(trace_id)
    update_agent_run_trace_status(
        trace_id=trace_id,
        status=trace.status,
        summary={**(trace.summary or {}), **values},
    )


def _agent_names_for_round(run_id: int, round_type: str) -> list[str]:
    names: list[str] = []
    for row in list_agent_run_responses(run_id):
        if row.get("round_type") == round_type:
            names.append(f"{_intent_from_agent_name(row.get('target_agent')).value}_agent")
    return names


def _agent_names_for_followups(run_id: int) -> list[str]:
    names: list[str] = []
    for row in list_agent_run_responses(run_id):
        if row.get("round_type") in {"followup", "gap_resolution", "claim_verification"}:
            names.append(f"{_intent_from_agent_name(row.get('target_agent')).value}_agent")
    return names


def _fail_run(run_id: int, reason: str) -> None:
    trace = get_agent_run_trace_by_id(run_id)
    if trace is None:
        raise LookupError(run_id)
    update_agent_run_trace_status(
        trace_id=run_id,
        status="failed",
        summary={**(trace.summary or {}), "status": "failed", "error": reason},
    )
