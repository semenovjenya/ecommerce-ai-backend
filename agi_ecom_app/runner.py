from __future__ import annotations

import logging
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from types import SimpleNamespace
from typing import Any, Callable

from agents import OutputGuardrailTripwireTriggered, Runner
from pydantic import BaseModel

from agi_ecom_app.config import get_settings
from agi_ecom_app.main_agent import (
    build_diagnostic_orchestrator_agent,
    build_domain_agent_for_round,
    build_domain_router_agent,
    build_finance_agent,
    build_marketing_agent,
    build_price_agent,
    build_run_hooks,
    build_sales_agent,
    build_supply_agent,
    ensure_domain_report_snapshots,
    ensure_supply_aggregated_stock_facts,
)
from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.forecast.rendering import render_forecast_decision_result
from agi_ecom_app.llm_runtime import get_active_llm_runtime_spec, get_active_model_metadata
from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text
from agi_ecom_app.inventory_policy import (
    calculate_coverage_days,
    classify_availability,
    classify_coverage_days,
)
from agi_ecom_app.prompt_pipeline import (
    adapt_round_result_to_legacy,
    build_round_prompt,
    compact_synthesis_package,
    determine_query_mode,
    memory_mode_for_round,
    resolve_workflow_version,
)
from agi_ecom_app.prompt_quality import (
    build_prompt_v2_repair_prompt,
    validate_prompt_v2_answer,
)
from agi_ecom_app.llm_routing import DomainRouteDecision, route_query_intent_with_llm
from agi_ecom_app.rendering import (
    render_action_execution_result,
    render_agent_diagnostic,
    render_report_table,
    render_sales_diagnostic,
)
from agi_ecom_app.routing import determine_query_intent, has_domain_routing_signal
from agi_ecom_app.schemas import (
    ActionExecutionResult,
    AgentDiagnosticResult,
    ClaimVerificationResult,
    ChatRole,
    CrossCheckResult,
    DataGap,
    EcomRunContext,
    FollowupResult,
    GapResolutionResult,
    MarketingDiagnosticResult,
    QueryMode,
    QueryIntent,
    ReportTable,
    RoundTask,
    RoundType,
    SalesDiagnosticResult,
    WorkflowVersion,
)
from agi_ecom_app.services import strategic_db
from agi_ecom_app.services.memory import (
    AgentRunGapRecord,
    AgentRunQuestionRecord,
    AgentRunResponseRecord,
    AgentRunRoundRecord,
    append_message,
    append_user_message_idempotent,
    complete_agent_run_trace,
    create_agent_run_question,
    create_agent_run_round,
    create_agent_run_trace,
    ensure_chat_session,
    get_assistant_reply_for_client_message_id,
    get_context_window,
    list_agent_run_gaps,
    list_agent_run_responses,
    refresh_chat_summary,
    save_agent_run_gaps,
    save_agent_run_response,
    update_agent_run_gap_status,
    update_agent_run_question_status,
    update_agent_run_round_status,
)
from agi_ecom_app.tools.strategic_tools import build_cross_domain_dependency_plan
from agi_ecom_app.action_board.service import (
    EXECUTION_MODE_METADATA_KEY,
    SUPPRESS_ACTION_TASKS_METADATA_KEY,
    TERMINAL_ACTION_EXECUTION_MODE,
    ACTION_TASKS_METADATA_KEY,
    ACTION_TASKS_SCHEMA_VERSION,
    ACTION_TASKS_SCHEMA_VERSION_KEY,
    extract_action_tasks_from_agent_output,
    persist_agent_action_tasks_for_message,
)

logger = logging.getLogger(__name__)

ROUND_1_SECONDARY_LIMIT = 5
ROUND_2_FOLLOWUP_LIMIT = 6
ROUND_3_GAP_RESOLUTION_LIMIT = 8
CLAIM_VERIFICATION_LIMIT = 5
MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS = 12
OPTIMIZED_SECONDARY_QUESTION_BUDGET = 4
DIAGNOSTIC_SECONDARY_MAX_TURNS = 14
DIAGNOSTIC_SYNTHESIS_MAX_TURNS = 12
TRANSIENT_MODEL_ERROR_TYPES = {"APIConnectionError", "APITimeoutError"}
TRANSIENT_MODEL_RETRY_ATTEMPTS = 2
TRANSIENT_MODEL_RETRY_DELAY_SECONDS = 1.0
DEFAULT_SYNC_CROSS_CHECK_CONCURRENCY = 4
CLAIM_VERIFICATION_ROUND_NUMBER = 4
SYNTHESIS_ROUND_NUMBER = 5
# User-facing diagnostic rounds are 1..5; primary is persisted as bootstrap round 0.
MAX_DIAGNOSTIC_ROUNDS = SYNTHESIS_ROUND_NUMBER
ACTION_SAFETY_SOURCE = "action_safety"
COVERAGE_POLICY_SOURCE = "coverage_policy"
CLAIM_VERIFICATION_SOURCE = "claim_verification"
DOMAIN_TOOL_NAMES = {
    "sales_agent",
    "marketing_agent",
    "price_agent",
    "supply_agent",
    "finance_agent",
}
SKU_EVIDENCE_DOMAINS = ("sales", "marketing", "price", "supply", "finance")
INVENTORY_TURNOVER_CAUSAL_DOMAINS = (
    QueryIntent.SALES,
    QueryIntent.PRICE,
    QueryIntent.MARKETING,
    QueryIntent.FINANCE,
)
DOMAIN_FORMULA_TOOLS: dict[QueryIntent, str] = {
    QueryIntent.SALES: "get_sales_formula_catalog",
    QueryIntent.MARKETING: "get_marketing_formula_catalog",
    QueryIntent.PRICE: "get_price_formula_catalog",
    QueryIntent.SUPPLY: "get_supply_formula_catalog",
    QueryIntent.FINANCE: "get_finance_formula_catalog",
}
SALES_REPORT_SNAPSHOT_TOOL = "get_sales_report_snapshot"
MARKETING_REPORT_SNAPSHOT_TOOL = "get_marketing_report_snapshot"
PRICE_DECISION_TOOL = "get_price_decision_overview"
ROUND_RESULT_TYPES: dict[RoundType, type[BaseModel]] = {
    RoundType.CROSS_CHECK: CrossCheckResult,
    RoundType.FOLLOWUP: FollowupResult,
    RoundType.GAP_RESOLUTION: GapResolutionResult,
    RoundType.CLAIM_VERIFICATION: ClaimVerificationResult,
}
ROUND_NAME_TYPES: dict[str, RoundType] = {
    "primary": RoundType.PRIMARY,
    "round_1": RoundType.CROSS_CHECK,
    "cross_check": RoundType.CROSS_CHECK,
    "round_2": RoundType.FOLLOWUP,
    "followup": RoundType.FOLLOWUP,
    "round_3": RoundType.GAP_RESOLUTION,
    "gap_resolution": RoundType.GAP_RESOLUTION,
    "claim_verification": RoundType.CLAIM_VERIFICATION,
    "synthesis": RoundType.SYNTHESIS,
}
V2_SECONDARY_TOOL_KEYWORDS: dict[QueryIntent, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = {
    QueryIntent.SALES: (
        (("отчет", "свод", "report"), ("get_sales_report_snapshot",)),
        (("выруч", "продаж", "заказ", "sku"), ("get_sales_sku_revenue_change_report", "get_sales_overview")),
    ),
    QueryIntent.MARKETING: (
        (("дрр", "drr"), ("get_marketing_drr_report",)),
        (("кампан",), ("get_marketing_campaign_period_report", "get_marketing_campaign_efficiency_snapshot")),
        (("запрос", "поиск"), ("get_marketing_query_performance_snapshot", "get_marketing_search_demand_snapshot")),
        (("действ", "останов", "масштаб"), ("get_marketing_action_matrix",)),
    ),
    QueryIntent.PRICE: (
        (("какую цену", "сниз", "повыс", "ценов", "решен"), ("get_price_context", "get_price_decision_overview")),
        (("конкур", "рын", "медиан"), ("get_price_context", "get_price_competitor_overview", "get_price_competitor_details")),
    ),
    QueryIntent.SUPPLY: (
        (("оборач", "дни запас", "средн", "turnover"), ("get_inventory_turnover",)),
        (
            ("деньги в остат", "стоимост остат", "inventory value", "stock value"),
            ("get_current_inventory_value",),
        ),
        (("остат", "постав", "дозаказ", "запас"), ("get_supply_overview",)),
    ),
    QueryIntent.FINANCE: (
        (("sku", "артикул", "марж", "прибыл", "себесто"), ("get_finance_sku_report_text", "get_finance_sku_profit_change_report_text")),
        (("отчет", "маркетплейс", "комисс", "логист", "выплат"), ("get_finance_marketplace_report_text", "get_finance_overview")),
    ),
}
SKU_STATUS_PRIORITY = {
    "not_checked": 0,
    "no_data": 0,
    "unconfirmed": 1,
    "refuted": 1,
    "partial": 2,
    "confirmed": 3,
}
SKU_CONTEXT_PATTERN = re.compile(
    r"\b(?:sku|артикул\w*|товар\w*)\s*(?:[:=#№-]\s*)?(\d{4,12})\b",
    re.IGNORECASE,
)
NUMERIC_SKU_PATTERN = re.compile(r"\b\d{4,12}\b")

NM_ID_PATTERN = re.compile(r"\bnm[_\s-]?id\b", re.IGNORECASE)
NM_ID_SLASH_WITH_VALUE_PATTERN = re.compile(
    r"\s*/\s*\bnm[_\s-]?id\b\s*(?:[:=#№-]\s*)?\d+",
    re.IGNORECASE,
)
NM_ID_WITH_VALUE_PATTERN = re.compile(
    r"\bnm[_\s-]?id\b\s*(?:[:=#№-]\s*)?\d+",
    re.IGNORECASE,
)
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
SALES_SNAPSHOT_HTML_TABLE_PATTERN = re.compile(
    r"\n*<table\b[^>]*>.*?(?:вчера\s+к\s+позавчера|последний\s+день\s+в\s+данных).*?"
    r"(?:неделя\s+к\s+неделе|последние\s+7\s+дней).*?"
    r"(?:месяц\s+к\s+месяцу|последние\s+30\s+дней).*?</table>\n*",
    re.IGNORECASE | re.DOTALL,
)
SALES_SNAPSHOT_MARKER_GROUPS = (
    ("вчера к позавчера", "последний день в данных"),
    ("неделя к неделе", "последние 7 дней"),
    ("месяц к месяцу", "последние 30 дней"),
)
MARKETING_SNAPSHOT_HTML_TABLE_PATTERN = re.compile(
    r"\n*<table\b[^>]*>.*?(?:вчера\s+к\s+позавчера|последний\s+день\s+в\s+данных).*?"
    r"(?:неделя\s+к\s+неделе|последние\s+7\s+дней).*?"
    r"(?:месяц\s+к\s+месяцу|последние\s+30\s+дней).*?дрр.*?</table>\n*",
    re.IGNORECASE | re.DOTALL,
)
MARKETING_SNAPSHOT_MARKER_GROUPS = (
    ("вчера к позавчера", "последний день в данных"),
    ("неделя к неделе", "последние 7 дней"),
    ("месяц к месяцу", "последние 30 дней"),
    ("дрр",),
)
# Free prose cannot be localized safely one token at a time: replacing
# ``price`` inside ``price-слоя`` produced hybrids such as ``цену-слоя``.
# User-facing wording belongs to the renderer/prompt; this sanitizer only
# performs structural redaction below (for example, internal nm_id values).
USER_FACING_MARKDOWN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = ()
BROKEN_DOMAIN_LOCALIZATION_PATTERN = re.compile(
    r"\b(?:цену-сло\w*|остатки-tools?\b|рекламу\s+spend\b|экономику\s+data\b|"
    r"ценовые\s+данные\s+(?:decision|mart|AI)\b)",
    re.IGNORECASE,
)
RISKY_ACTION_PATTERN = re.compile(
    r"\b(?:сократить|остановить|снизить|масштабировать|распродать|распродавать)\b",
    re.IGNORECASE,
)
ACTION_CONDITION_PATTERN = re.compile(
    r"\b(?:если|при условии|после подтверждения|после проверки|при подтверждении|"
    r"только после|с контролем|проверить|сверить|рассчитать)\b",
    re.IGNORECASE,
)
ACTION_TABLE_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:иначе\s+зафиксировать|при\s+повторной\s+сверке|если\s+[^|\n.]{0,120}\s+иначе)\b",
    re.IGNORECASE,
)
PERSISTABLE_SKU_GAP_REASONS = {
    "agent_failed_without_evidence",
    "metric_absent",
}
BARE_ACTION_TABLE_TASK_PATTERN = re.compile(
    r"^\s*(?:проверить|рассчитать|остановить|удержать|снизить|поднять|дозаказать|"
    r"обновить|сформировать|создать|запустить|исправить)\s*\.?\s*$",
    re.IGNORECASE,
)

ACTION_SAFETY_RULES: dict[str, dict[str, Any]] = {
    "price_change": {
        "label": "снижение цены или скидка",
        "patterns": (
            "снижать цену",
            "снизить цену",
            "снижение цены",
            "скидк",
            "переоцен",
        ),
        "checks": (
            {
                "target_agent": QueryIntent.FINANCE,
                "required_check": "Проверить маржу и экономику перед снижением цены.",
            },
            {
                "target_agent": QueryIntent.PRICE,
                "required_check": "Проверить текущую цену, скидку и ценовой разрыв с конкурентами.",
            },
        ),
    },
    "marketing_scale": {
        "label": "масштабирование рекламы",
        "patterns": (
            "масштабировать рекламу",
            "масштабировать камп",
            "усилить рекламу",
            "увеличить рекламу",
            "увеличить бюджет",
            "промо",
        ),
        "checks": (
            {
                "target_agent": QueryIntent.SUPPLY,
                "required_check": "Проверить запас и доступность перед масштабированием рекламы.",
            },
            {
                "target_agent": QueryIntent.MARKETING,
                "required_check": "Проверить кампании, видимость и рекламный спрос перед масштабированием.",
            },
            {
                "target_agent": QueryIntent.FINANCE,
                "required_check": "Проверить экономику и прибыльность перед ростом рекламной нагрузки.",
            },
        ),
    },
    "reorder": {
        "label": "дозаказ",
        "patterns": (
            "дозаказ",
            "дозаказывать",
            "дозаказать",
            "заказать",
            "пополнить",
        ),
        "checks": (
            {
                "target_agent": QueryIntent.SALES,
                "required_check": "Проверить фактический спрос и темп продаж перед дозаказом.",
            },
            {
                "target_agent": QueryIntent.MARKETING,
                "required_check": "Проверить поисковый и рекламный спрос перед дозаказом.",
            },
        ),
    },
    "sell_off": {
        "label": "распродажа или вывод из закупки",
        "patterns": (
            "распродавать",
            "распродать",
            "распродаж",
            "не закупать",
            "сократить закуп",
        ),
        "checks": (
            {
                "target_agent": QueryIntent.PRICE,
                "required_check": "Проверить цену и конкурентное позиционирование перед распродажей.",
            },
            {
                "target_agent": QueryIntent.FINANCE,
                "required_check": "Проверить маржу и финансовый эффект перед распродажей.",
            },
        ),
    },
}


class AgentRuntimeError(RuntimeError):
    """Raised when the agent runtime cannot be started cleanly."""


@dataclass(frozen=True)
class ToolPolicy:
    agent_name: str
    target_agent: QueryIntent
    round_name: str | None
    question_id: int | None
    required_tools: tuple[str, ...]
    formula_tool: str
    allowed_prelude: tuple[str, ...] = ("get_database_time",)


class ToolPolicyViolation(AgentRuntimeError):
    """Raised when a domain agent does not satisfy required tool policy after retry."""

    def __init__(
        self,
        *,
        output: Any,
        tool_report: dict[str, Any],
        violations: list[str],
    ):
        super().__init__("Domain agent did not satisfy required tool policy.")
        self.output = output
        self.tool_report = tool_report
        self.violations = violations

    def runtime_failure(self) -> dict[str, Any]:
        return {
            "kind": "tool_policy",
            "message": "Агент не выполнил обязательную tool-policy перед аналитическим выводом.",
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class ChatTurnResult:
    answer: str
    intent: QueryIntent
    assistant_message_seq: int | None = None
    reused_existing: bool = False


@dataclass(frozen=True)
class OrchestratedDiagnosticRun:
    final_output: Any
    primary_agent: QueryIntent
    secondary_agents: list[str]
    followup_agents: list[str]
    dependency_plans: list[dict[str, Any]]
    synthesis_debug: dict[str, Any]


class DuplicateClientMessageError(AgentRuntimeError):
    """Raised when a websocket client reuses a client message id."""

    def __init__(
        self,
        client_message_id: str,
        answer: str | None = None,
        intent: QueryIntent | None = None,
        assistant_message_seq: int | None = None,
    ):
        self.client_message_id = client_message_id
        self.answer = answer
        self.intent = intent
        self.assistant_message_seq = assistant_message_seq

        detail = (
            f"Сообщение с client_message_id={client_message_id} уже обработано."
            if answer is not None
            else f"Сообщение с client_message_id={client_message_id} уже выполняется."
        )
        super().__init__(detail)


def _coerce_output(value: Any) -> str:
    if isinstance(value, ForecastDecisionResult):
        return render_forecast_decision_result(value)
    if isinstance(value, ActionExecutionResult):
        return render_action_execution_result(value)
    if isinstance(value, AgentDiagnosticResult):
        return render_agent_diagnostic(value)
    if isinstance(value, SalesDiagnosticResult):
        return render_sales_diagnostic(value)
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, str):
        return value
    return str(value)


def _extract_guardrail_violations(exc: OutputGuardrailTripwireTriggered) -> list[str]:
    output_info = getattr(exc.guardrail_result.output, "output_info", None)
    if isinstance(output_info, dict):
        violations = output_info.get("violations")
        if isinstance(violations, list):
            return [str(violation) for violation in violations]
    return []


def _persist_assistant_turn(
    *,
    chat_id: str,
    output_text: str,
    intent: QueryIntent,
    client_message_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> ChatTurnResult:
    assistant_message = append_message(
        chat_id=chat_id,
        role=ChatRole.ASSISTANT,
        content=output_text,
        metadata={
            "intent": intent.value,
            **(
                {"reply_to_client_message_id": client_message_id}
                if client_message_id
                else {}
            ),
            **(metadata or {}),
        },
    )
    try:
        refresh_chat_summary(chat_id=chat_id)
    except Exception as exc:
        logger.warning("chat_summary_refresh_failed chat_id=%s error=%s", chat_id, exc)
    return ChatTurnResult(
        answer=output_text,
        intent=intent,
        assistant_message_seq=assistant_message.seq,
    )


def _trace_metadata(context: EcomRunContext) -> dict[str, Any]:
    metadata: dict[str, Any] = _active_model_metadata_safe()
    metadata.update(
        {
            "workflow_version": _workflow_version_for_context(context).value,
            "prompt_version": _workflow_version_for_context(context).value,
            "query_mode": getattr(
                getattr(context, "query_mode", QueryMode.REPORT),
                "value",
                str(getattr(context, "query_mode", QueryMode.REPORT.value)),
            ),
        }
    )
    if context.last_prompt_report:
        metadata["_prompt_report"] = dict(context.last_prompt_report)
    if context.agent_trace:
        metadata["agent_trace"] = list(context.agent_trace)
    if context.tool_trace:
        metadata["tool_trace"] = list(context.tool_trace)
    if context.cross_agent_trace:
        metadata["cross_agent_trace"] = list(context.cross_agent_trace)
    return metadata


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


def _route_decision_metadata(decision: DomainRouteDecision) -> dict[str, Any]:
    return {
        "route_to_domain": decision.route_to_domain,
        "intent": decision.intent.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "supporting_agents": [agent.value for agent in decision.supporting_agents],
        "query_plan": decision.query_plan.model_dump(mode="json"),
    }


def _should_extract_action_tasks(metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or {}
    if metadata.get(SUPPRESS_ACTION_TASKS_METADATA_KEY):
        return False
    return metadata.get(EXECUTION_MODE_METADATA_KEY) != TERMINAL_ACTION_EXECUTION_MODE


def _agent_run_trace_summary(
    *,
    context: EcomRunContext,
    metadata: dict[str, Any] | None = None,
    diagnostic_run: OrchestratedDiagnosticRun | None = None,
    final_quality_violations: list[str] | None = None,
    status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    runtime_failures = list(metadata.get("runtime_failures") or [])
    if diagnostic_run:
        primary_agent = _agent_tool_name(diagnostic_run.primary_agent)
        secondary_agents = [f"{agent}_agent" for agent in diagnostic_run.secondary_agents]
        followup_agents = [f"{agent}_agent" for agent in diagnostic_run.followup_agents]
        dependency_plan_count = len(diagnostic_run.dependency_plans)
    else:
        primary_agent = metadata.get("primary_agent")
        secondary_agents = list(metadata.get("secondary_agents") or [])
        followup_agents = list(metadata.get("followup_agents") or [])
        dependency_plan_count = len(metadata.get("dependency_plans") or [])

    summary = {
        **_active_model_metadata_safe(),
        "workflow_version": _workflow_version_for_context(context).value,
        "prompt_version": _workflow_version_for_context(context).value,
        "query_mode": getattr(
            getattr(context, "query_mode", QueryMode.REPORT),
            "value",
            str(getattr(context, "query_mode", QueryMode.REPORT.value)),
        ),
        "agent_trace": list(context.agent_trace),
        "tool_trace": list(context.tool_trace),
        "cross_agent_trace": list(context.cross_agent_trace),
        "primary_agent": primary_agent,
        "secondary_agents": secondary_agents,
        "followup_agents": followup_agents,
        "dependency_plan_count": dependency_plan_count,
        "runtime_failure_count": len(runtime_failures),
        "final_quality_violations": list(final_quality_violations or metadata.get("final_quality_violations") or []),
        "status": status,
    }
    if error:
        summary["error"] = error
    return summary


def _complete_agent_run_trace_safely(
    *,
    trace_id: int | None,
    context: EcomRunContext,
    assistant_message_seq: int | None,
    status: str,
    summary: dict[str, Any],
    primary_agent: str | None = None,
) -> None:
    if trace_id is None:
        return
    try:
        complete_agent_run_trace(
            trace_id=trace_id,
            assistant_message_seq=assistant_message_seq,
            status=status,
            events=list(context.decision_trace),
            summary=summary,
            primary_agent=primary_agent,
        )
    except Exception as exc:
        logger.warning("agent_run_trace_complete_failed trace_id=%s error=%s", trace_id, exc)


def _persist_agent_action_tasks_safely(
    *,
    chat_id: str,
    assistant_message_seq: int | None,
    raw_actions: list[dict[str, Any]],
    intent: QueryIntent,
) -> None:
    if assistant_message_seq is None:
        return
    try:
        persist_agent_action_tasks_for_message(
            chat_id=chat_id,
            server_message_id=assistant_message_seq,
            raw_actions=raw_actions,
            intent=intent.value,
        )
    except Exception as exc:
        logger.warning(
            "agent_action_tasks_persist_failed chat_id=%s assistant_message_seq=%s error=%s",
            chat_id,
            assistant_message_seq,
            exc,
        )


def _create_agent_run_round_safely(
    *,
    run_id: int | None,
    round_number: int,
    round_type: str,
    status: str = "running",
) -> AgentRunRoundRecord | None:
    if run_id is None:
        return None
    try:
        return create_agent_run_round(
            run_id=run_id,
            round_number=round_number,
            round_type=round_type,
            status=status,
        )
    except Exception as exc:
        logger.warning(
            "agent_round_persist_failed run_id=%s round_type=%s error=%s",
            run_id,
            round_type,
            exc,
        )
        return None


def _update_agent_run_round_status_safely(
    round_record: AgentRunRoundRecord | None,
    status: str,
) -> None:
    if round_record is None:
        return
    try:
        update_agent_run_round_status(round_id=round_record.id, status=status)
    except Exception as exc:
        logger.warning(
            "agent_round_status_update_failed round_id=%s status=%s error=%s",
            round_record.id,
            status,
            exc,
        )


def _create_agent_run_question_safely(
    *,
    run_id: int | None,
    round_record: AgentRunRoundRecord | None,
    source_agent: str | None,
    target_agent: str,
    question: str,
    reason: str | None = None,
    verification_goal: dict[str, Any] | None = None,
    priority: int | None = None,
    status: str = "pending",
    parent_question_id: int | None = None,
) -> AgentRunQuestionRecord | None:
    if run_id is None or round_record is None:
        return None
    try:
        return create_agent_run_question(
            run_id=run_id,
            round_id=round_record.id,
            source_agent=source_agent,
            target_agent=target_agent,
            question=question,
            reason=reason,
            verification_goal=verification_goal or {},
            priority=priority,
            status=status,
            parent_question_id=parent_question_id,
        )
    except Exception as exc:
        logger.warning(
            "agent_question_persist_failed run_id=%s target_agent=%s error=%s",
            run_id,
            target_agent,
            exc,
        )
        return None


def _update_agent_run_question_status_safely(
    question_record: AgentRunQuestionRecord | None,
    status: str,
) -> None:
    if question_record is None:
        return
    try:
        update_agent_run_question_status(question_id=question_record.id, status=status)
    except Exception as exc:
        logger.warning(
            "agent_question_status_update_failed question_id=%s status=%s error=%s",
            question_record.id,
            status,
            exc,
        )


def _gap_entries_from_payload(
    *,
    payload: dict[str, Any],
    source_agent: str,
    blocking: bool,
) -> list[dict[str, Any]]:
    round_result = payload.get("_round_result")
    typed_verdict = (
        str(round_result.get("verdict") or "").strip().lower()
        if isinstance(round_result, dict)
        else ""
    )
    # A typed secondary `no_data` result is terminal evidence for the exact
    # gap emitted by that check.  Persisting it as `unconfirmed` caused the
    # optimized orchestrator to ask the same domain/metric again during
    # gap-resolution.  The existing key-based deduplication already treats a
    # `no_data` row as a blocker, including when an older duplicate remains
    # `unconfirmed`.
    evidence_status = "no_data" if typed_verdict == "no_data" else "unconfirmed"
    entries: list[dict[str, Any]] = []
    for gap in payload.get("data_gaps", []):
        if not isinstance(gap, dict):
            continue
        entries.append(
            {
                "source_agent": source_agent,
                "target_agent": _gap_target_agent(source_agent, gap),
                "metric": str(gap.get("metric") or f"{source_agent} data gap"),
                "reason": str(gap.get("reason") or "Данных недостаточно для подтверждения."),
                "impact": gap.get("impact"),
                "blocking": blocking,
                "status": "open",
                "evidence_status": evidence_status,
            }
        )
    return entries


def _persist_agent_run_response_safely(
    *,
    run_id: int | None,
    question_record: AgentRunQuestionRecord | None,
    agent_name: str,
    payload: dict[str, Any],
    runtime_failure: dict[str, Any] | None = None,
    blocking_gaps: bool = True,
    gap_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> AgentRunResponseRecord | None:
    if run_id is None or question_record is None:
        return None
    try:
        response = save_agent_run_response(
            question_id=question_record.id,
            agent_name=agent_name,
            status="failed" if runtime_failure else "completed",
            summary=str(payload.get("summary") or ""),
            raw_output=payload,
            has_business_evidence=False if runtime_failure else _payload_has_business_evidence(payload),
            runtime_failure=runtime_failure,
        )
        gaps = _gap_entries_from_payload(
            payload=payload,
            source_agent=_intent_from_agent_name(agent_name).value,
            blocking=blocking_gaps,
        )
        if gap_filter is not None:
            gaps = [gap for gap in gaps if gap_filter(gap)]
        if gaps:
            save_agent_run_gaps(
                run_id=run_id,
                response_id=response.id,
                gaps=gaps,
            )
        return response
    except Exception as exc:
        logger.warning(
            "agent_response_persist_failed run_id=%s agent_name=%s error=%s",
            run_id,
            agent_name,
            exc,
        )
        return None


def _persist_gap_entries_safely(
    *,
    run_id: int | None,
    response_id: int | None,
    gaps: list[dict[str, Any]],
) -> list[AgentRunGapRecord]:
    if run_id is None or not gaps:
        return []
    try:
        return save_agent_run_gaps(run_id=run_id, response_id=response_id, gaps=gaps)
    except Exception as exc:
        logger.warning("agent_gaps_persist_failed run_id=%s error=%s", run_id, exc)
        return []


def _update_agent_run_gap_status_safely(gap_id: int | None, status: str) -> None:
    if gap_id is None:
        return
    try:
        update_agent_run_gap_status(gap_id=gap_id, status=status)
    except Exception as exc:
        logger.warning(
            "agent_gap_status_update_failed gap_id=%s status=%s error=%s",
            gap_id,
            status,
            exc,
        )


def _prepare_chat_context(
    user_input: str,
    chat_id: str,
    client_message_id: str | None = None,
) -> tuple[EcomRunContext, QueryIntent, bool, DomainRouteDecision]:
    settings = get_settings()
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    workflow_version = resolve_workflow_version(
        getattr(settings, "agent_workflow_version", WorkflowVersion.V1.value)
    )
    query_mode = determine_query_mode(user_input)

    if runtime_spec.missing_env_vars:
        raise AgentRuntimeError(
            "Не настроен ключ активного LLM-профиля "
            f"`{runtime_spec.profile}` ({', '.join(runtime_spec.missing_env_vars)}). "
            "Установи переменную окружения и повтори запрос."
        )

    fallback_intent = determine_query_intent(user_input)

    try:
        ensure_chat_session(chat_id=chat_id, model_name=runtime_spec.display_name)
        routing_window = get_context_window(
            chat_id=chat_id,
            limit=settings.memory_context_message_limit,
        )
        routing_context = EcomRunContext(
            user_query=user_input,
            intent=fallback_intent,
            today=date.today(),
            db_target=f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
            chat_id=chat_id,
            chat_summary=routing_window.summary_text,
            recent_messages=routing_window.recent_messages,
            evidence_budget=0,
            workflow_version=workflow_version,
            round_type=RoundType.ROUTER,
            query_mode=query_mode,
            memory_mode=memory_mode_for_round(RoundType.ROUTER),
        )
        route_decision = route_query_intent_with_llm(user_input, context=routing_context)
        intent = route_decision.intent
        query_mode = route_decision.query_plan.query_mode
        _, created_new = append_user_message_idempotent(
            chat_id=chat_id,
            content=user_input,
            metadata={
                "intent": intent.value,
                "intent_router": _route_decision_metadata(route_decision),
                **_active_model_metadata_safe(),
            },
            client_message_id=client_message_id,
        )
        context_window = get_context_window(
            chat_id=chat_id,
            limit=settings.memory_context_message_limit,
        )
    except Exception as exc:
        raise AgentRuntimeError(
            f"Не удалось прочитать память чата из postgresmemory: {exc}"
        ) from exc

    context = EcomRunContext(
        user_query=user_input,
        intent=intent,
        today=date.today(),
        db_target=f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
        chat_id=chat_id,
        chat_summary=context_window.summary_text,
        recent_messages=context_window.recent_messages,
        workflow_version=workflow_version,
        round_type=RoundType.PRIMARY,
        query_mode=query_mode,
        memory_mode=memory_mode_for_round(RoundType.PRIMARY),
        query_plan=route_decision.query_plan.model_dump(mode="json"),
    )
    context.decision_trace.extend(routing_context.decision_trace)
    return context, intent, created_new, route_decision


def _is_sales_report_query(user_query: str) -> bool:
    text = _normalize_text(user_query)
    return (
        ("отчет" in text or "report" in text or "сводк" in text)
        and ("продаж" in text or "sales" in text or "выруч" in text or "заказ" in text)
    ) or _text_contains_any(
        text,
        (
            "вчера к позавчера",
            "последний день в данных",
            "неделя к неделе",
            "последние 7 дней",
            "месяц к месяцу",
            "последние 30 дней",
            "week to week",
            "month to month",
        ),
    )


def _is_marketing_report_query(user_query: str) -> bool:
    text = _normalize_text(user_query)
    has_report_word = "отчет" in text or "report" in text or "сводк" in text
    has_marketing_word = _text_contains_any(
        text,
        (
            "маркетинг",
            "реклам",
            "кампани",
            "дрр",
            "drr",
            "cpc",
            "клик",
            "показ",
            "расход",
            "spend",
        ),
    )
    has_comparison = _text_contains_any(
        text,
        (
            "вчера к позавчера",
            "последний день в данных",
            "неделя к неделе",
            "последние 7 дней",
            "месяц к месяцу",
            "последние 30 дней",
            "week to week",
            "month to month",
        ),
    )
    return (has_report_word and (has_marketing_word or has_comparison)) or (
        has_marketing_word and has_comparison
    )


def _is_price_decision_query(user_query: str) -> bool:
    text = _normalize_text(user_query)
    if not text:
        return False
    if _text_contains_any(text, ("цена клика", "cpc", "ставк")):
        return False
    return _text_contains_any(
        text,
        (
            "какую цену",
            "цену поставить",
            "поставить цену",
            "какая цена",
            "рекомендованная цена",
            "рекомендовать цену",
            "нужно ли снизить цену",
            "снизить цену",
            "снижение цены",
            "поднять цену",
            "повысить цену",
            "скидк",
            "дисконт",
            "переоцен",
            "репрайс",
            "ценовой тест",
            "тест цены",
            "стартовая цена",
            "минимально безопасная цена",
            "price decision",
            "repricing",
        ),
    )


def _domain_tool_policy(
    *,
    target: QueryIntent,
    user_query: str,
    round_name: str | None,
    question_id: int | None,
    explicit_required_tools: tuple[str, ...] | list[str] = (),
) -> ToolPolicy:
    formula_tool = DOMAIN_FORMULA_TOOLS[target]
    # Domain agents choose the business method themselves; this policy only records tool usage.
    allowed_prelude = ["get_database_time"]
    if target == QueryIntent.SALES and _is_sales_report_query(user_query):
        allowed_prelude.append(SALES_REPORT_SNAPSHOT_TOOL)
    if target == QueryIntent.MARKETING and _is_marketing_report_query(user_query):
        allowed_prelude.append(MARKETING_REPORT_SNAPSHOT_TOOL)
    required_tools = list(dict.fromkeys(str(name) for name in explicit_required_tools if name))
    if target == QueryIntent.PRICE and round_name == "primary" and _is_price_decision_query(user_query):
        required_tools = list(dict.fromkeys([*required_tools, PRICE_DECISION_TOOL]))
    return ToolPolicy(
        agent_name=f"{target.value}_agent",
        target_agent=target,
        round_name=round_name,
        question_id=question_id,
        required_tools=tuple(required_tools),
        formula_tool=formula_tool,
        allowed_prelude=tuple(allowed_prelude),
    )


def _agent_has_callable_tools(agent: Any) -> bool:
    tools = getattr(agent, "tools", None)
    return isinstance(tools, (list, tuple)) and bool(tools)


def _tool_report_from_trace_delta(
    *,
    policy: ToolPolicy,
    tool_names: list[str],
    decision_events: list[dict[str, Any]],
    policy_violations: list[str],
) -> dict[str, Any]:
    if not tool_names:
        tool_names = [
            str(event.get("tool"))
            for event in decision_events
            if event.get("event") == "tool_start" and event.get("tool")
        ]
    end_events = [
        event
        for event in decision_events
        if event.get("event") == "tool_end" and event.get("tool")
    ]
    used_end_indexes: set[int] = set()
    tools: list[dict[str, Any]] = []
    for name in tool_names:
        matching_end: dict[str, Any] | None = None
        for index, event in enumerate(end_events):
            if index in used_end_indexes or event.get("tool") != name:
                continue
            matching_end = event
            used_end_indexes.add(index)
            break
        status = str(matching_end.get("status") or "completed") if matching_end else (
            "started" if decision_events else "completed"
        )
        duration_ms = matching_end.get("duration_ms") if matching_end else None
        tools.append(
            {
                "name": name,
                "status": status,
                "duration_ms": duration_ms if isinstance(duration_ms, int) else None,
            }
        )
    return {
        "agent_name": policy.agent_name,
        "target_agent": policy.target_agent.value,
        "round": policy.round_name,
        "question_id": policy.question_id,
        "tool_names": list(tool_names),
        "tools": tools,
        "required_tools": list(policy.required_tools),
        "policy_violations": list(policy_violations),
        "formula_catalog_used": policy.formula_tool in tool_names,
    }


def _tool_policy_violations(policy: ToolPolicy, tool_names: list[str]) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    hard_violations: list[str] = []
    for required_tool in policy.required_tools:
        if required_tool not in tool_names:
            violation = f"missing_required_tool:{required_tool}"
            violations.append(violation)
            hard_violations.append(violation)
    return violations, hard_violations


def _tool_policy_retry_prompt(user_input: str, policy: ToolPolicy, hard_violations: list[str]) -> str:
    required_tools = ", ".join(f"`{name}`" for name in policy.required_tools)
    return (
        "Служебная повторная попытка для явно заданной ToolPolicy.\n"
        f"Недостающие проверки: {required_tools}.\n"
        "Выполни эти проверки, затем формируй вывод по данным.\n"
        f"Нарушения первой попытки: {', '.join(hard_violations)}.\n\n"
        f"Исходный запрос:\n{user_input}"
    )


def _attach_tool_report(payload: dict[str, Any], context: EcomRunContext) -> dict[str, Any]:
    if context.last_tool_report:
        payload = dict(payload)
        payload["_tool_report"] = context.last_tool_report
    return payload


def _attach_prompt_report(payload: dict[str, Any], context: EcomRunContext) -> dict[str, Any]:
    prompt_report = getattr(context, "last_prompt_report", None)
    if prompt_report:
        payload = dict(payload)
        payload["_prompt_report"] = dict(prompt_report)
    return payload


def _is_transient_model_error(exc: Exception) -> bool:
    return type(exc).__name__ in TRANSIENT_MODEL_ERROR_TYPES


def _run_agent_once(
    builder: Callable[[], Any],
    user_input: str,
    context: EcomRunContext,
    hooks: Any,
    max_turns: int,
    expected_output_type: type[BaseModel] | None = None,
) -> tuple[Any, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, TRANSIENT_MODEL_RETRY_ATTEMPTS + 1):
        agent = builder()
        prompt_report = dict(getattr(context, "last_prompt_report", None) or {})
        if prompt_report:
            prompt_report["input_chars"] = len(user_input)
            prompt_report["allowed_tools"] = [
                str(getattr(tool, "name", "") or "") for tool in getattr(agent, "tools", [])
            ]
            try:
                instructions = agent.instructions
                rendered = (
                    instructions(SimpleNamespace(context=context), agent)
                    if callable(instructions)
                    else instructions
                )
                if isinstance(rendered, str):
                    prompt_report["system_chars"] = len(rendered)
            except Exception:
                logger.debug("prompt_report_system_measure_failed", exc_info=True)
            prompt_report["total_chars"] = int(prompt_report.get("system_chars") or 0) + len(user_input)
            context.last_prompt_report = prompt_report
        try:
            output = Runner.run_sync(
                agent,
                user_input,
                context=context,
                hooks=hooks,
                max_turns=max_turns,
            ).final_output
            if expected_output_type is not None and not isinstance(output, expected_output_type):
                output = coerce_model_from_json_text(output, expected_output_type)
            return agent, ensure_domain_report_snapshots(
                output,
                user_query=context.user_query,
                query_plan=context.query_plan,
            )
        except Exception as exc:
            if not _is_transient_model_error(exc) or attempt >= TRANSIENT_MODEL_RETRY_ATTEMPTS:
                raise
            last_exc = exc
            logger.warning(
                "agent_model_transient_error_retry attempt=%s/%s error_type=%s",
                attempt,
                TRANSIENT_MODEL_RETRY_ATTEMPTS,
                type(exc).__name__,
            )
            time.sleep(TRANSIENT_MODEL_RETRY_DELAY_SECONDS)
    raise last_exc or RuntimeError("agent model call failed without an exception")



def _run_agent(
    builder: Callable[[], Any],
    user_input: str,
    context: EcomRunContext,
    hooks: Any,
    max_turns: int,
    tool_policy: ToolPolicy | None = None,
    expected_output_type: type[BaseModel] | None = None,
    prompt_report: dict[str, Any] | None = None,
) -> Any:
    context.last_tool_report = None
    context.last_prompt_report = dict(prompt_report or {}) or None
    if tool_policy is None:
        _, output = _run_agent_once(
            builder,
            user_input,
            context,
            hooks,
            max_turns,
            expected_output_type=expected_output_type,
        )
        return output

    attempts = [user_input]
    last_output: Any = None
    last_report: dict[str, Any] | None = None
    last_hard_violations: list[str] = []
    for attempt_index, attempt_input in enumerate(attempts):
        tool_start_index = len(context.tool_trace)
        decision_start_index = len(context.decision_trace)
        agent, output = _run_agent_once(
            builder,
            attempt_input,
            context,
            hooks,
            max_turns,
            expected_output_type=expected_output_type,
        )
        last_output = output
        if not _agent_has_callable_tools(agent):
            return output
        tool_names = list(context.tool_trace[tool_start_index:])
        decision_events = list(context.decision_trace[decision_start_index:])
        violations, hard_violations = _tool_policy_violations(tool_policy, tool_names)
        report = _tool_report_from_trace_delta(
            policy=tool_policy,
            tool_names=tool_names,
            decision_events=decision_events,
            policy_violations=violations,
        )
        last_report = report
        context.last_tool_report = report
        if not hard_violations:
            return output
        last_hard_violations = hard_violations
        if attempt_index == 0:
            attempts.append(_tool_policy_retry_prompt(user_input, tool_policy, hard_violations))

    if last_report is None:
        last_report = _tool_report_from_trace_delta(
            policy=tool_policy,
            tool_names=[],
            decision_events=[],
            policy_violations=last_hard_violations,
        )
        context.last_tool_report = last_report
    raise ToolPolicyViolation(
        output=last_output,
        tool_report=last_report,
        violations=last_hard_violations,
    )


def _workflow_version_for_context(context: EcomRunContext) -> WorkflowVersion:
    return resolve_workflow_version(getattr(context, "workflow_version", WorkflowVersion.V1))


def _round_type_from_name(round_name: str | RoundType | None) -> RoundType:
    if isinstance(round_name, RoundType):
        return round_name
    return ROUND_NAME_TYPES.get(str(round_name or "").strip().lower(), RoundType.PRIMARY)


def _primary_output_type(intent: QueryIntent) -> type[BaseModel]:
    if intent == QueryIntent.SALES:
        return SalesDiagnosticResult
    if intent == QueryIntent.MARKETING:
        return MarketingDiagnosticResult
    return AgentDiagnosticResult


def _secondary_allowed_tools(
    *,
    intent: QueryIntent,
    question: str,
    required_tools: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return a deterministic, narrow tool allow-list for a v2 secondary task."""

    normalized = _normalize_text(question)
    selected = {
        "get_database_time",
        f"{intent.value}_list_tables",
        f"{intent.value}_describe_table",
        f"{intent.value}_search_columns",
        f"{intent.value}_sample_rows",
        f"{intent.value}_run_sql",
        *required_tools,
    }
    defaults = {
        QueryIntent.SALES: ("get_sales_overview", "get_sales_sku_revenue_change_report"),
        QueryIntent.MARKETING: ("get_marketing_overview",),
        QueryIntent.PRICE: ("get_price_context", "get_price_decision_overview"),
        QueryIntent.SUPPLY: ("get_supply_overview",),
        QueryIntent.FINANCE: ("get_finance_overview",),
    }
    selected.update(defaults[intent])
    for markers, tool_names in V2_SECONDARY_TOOL_KEYWORDS.get(intent, ()):
        if any(marker in normalized for marker in markers):
            selected.update(tool_names)
    return sorted(selected)


def _v2_required_tools_for_query(
    intent: QueryIntent,
    user_query: str,
) -> tuple[str, ...]:
    """Require deterministic calculators for explicit derived supply metrics."""

    if intent != QueryIntent.SUPPLY:
        return ()
    normalized = _normalize_text(user_query)
    required: list[str] = []
    if any(marker in normalized for marker in ("оборач", "turnover", "дни запас")):
        required.append("get_inventory_turnover")
    if any(
        marker in normalized
        for marker in (
            "деньги в текущих остат",
            "деньги в остат",
            "стоимост текущих остат",
            "стоимост остат",
            "inventory value",
            "stock value",
        )
    ):
        required.append("get_current_inventory_value")
    if any(
        marker in normalized
        for marker in (
            "остат",
            "stock",
            "inventory",
            "дозаказ",
            "reorder",
            "покрыт",
            "дефиц",
            "out-of-stock",
        )
    ):
        required.append("get_supply_overview")
    return tuple(required)


def _v2_required_tools_for_plan(
    intent: QueryIntent,
    query_plan: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Map canonical router capabilities to tools without re-parsing user text."""

    analyses = {
        str(value)
        for value in list((query_plan or {}).get("requested_analyses") or [])
    }
    required: list[str] = []
    if intent == QueryIntent.SUPPLY:
        if analyses.intersection({"inventory_matrix", "inventory_turnover"}):
            required.append("get_inventory_turnover")
        if analyses.intersection({"current_stock", "inventory_matrix", "replenishment"}):
            required.append("get_supply_overview")
        if analyses.intersection({"inventory_value", "inventory_matrix"}):
            required.append("get_current_inventory_value")
    return tuple(dict.fromkeys(required))


def _domain_agent_builder_for_intent(
    intent: QueryIntent,
    *,
    workflow_version: WorkflowVersion | str = WorkflowVersion.V1,
    round_type: RoundType | str = RoundType.PRIMARY,
    allowed_tools: list[str] | tuple[str, ...] = (),
) -> Callable[[], Any]:
    version = resolve_workflow_version(workflow_version)
    resolved_round = _round_type_from_name(round_type)
    if version != WorkflowVersion.V1:
        return lambda: build_domain_agent_for_round(
            intent,
            workflow_version=version,
            round_type=resolved_round,
            allowed_tools=allowed_tools,
        )
    return {
        QueryIntent.SALES: build_sales_agent,
        QueryIntent.MARKETING: build_marketing_agent,
        QueryIntent.PRICE: build_price_agent,
        QueryIntent.SUPPLY: build_supply_agent,
        QueryIntent.FINANCE: build_finance_agent,
    }[intent]


def _agent_tool_name(intent: QueryIntent) -> str:
    return f"{intent.value}_agent"


def _intent_from_agent_name(value: Any) -> QueryIntent:
    normalized = str(value or "").lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.removesuffix("_agent")
    try:
        return QueryIntent(normalized)
    except ValueError:
        return QueryIntent.SALES


def _compact_table(table: ReportTable, max_rows: int = 8) -> dict[str, Any]:
    columns_text = _normalize_text(" ".join(str(column) for column in table.columns))
    title_text = _normalize_text(table.title)
    row_limit = (
        len(table.rows)
        if (
            "sku" in columns_text
            or "артикул" in columns_text
            or "sku-матриц" in title_text
        )
        else max_rows
    )
    return {
        "title": table.title,
        "columns": table.columns,
        "rows": table.rows[:row_limit],
    }


def _diagnostic_payload(output: Any) -> dict[str, Any]:
    if isinstance(
        output,
        (CrossCheckResult, FollowupResult, GapResolutionResult, ClaimVerificationResult),
    ):
        return adapt_round_result_to_legacy(output)
    if isinstance(output, dict):
        payload = dict(output)
        payload.setdefault("summary", "")
        payload.setdefault("tables", [])
        payload.setdefault("findings", [])
        payload.setdefault("recommendations", [])
        payload.setdefault("data_gaps", [])
        payload.setdefault("cross_agent_requests", [])
        return payload
    if isinstance(output, SalesDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts],
            "sales_report_snapshot": (
                _compact_table(output.sales_report_snapshot)
                if output.sales_report_snapshot
                else None
            ),
            "tables": [_compact_table(table) for table in output.as_tables()],
            "findings": [finding.model_dump(mode="json") for finding in output.findings],
            "recommendations": output.recommendations,
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, MarketingDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts],
            "marketing_report_snapshot": (
                _compact_table(output.marketing_report_snapshot)
                if output.marketing_report_snapshot
                else None
            ),
            "tables": [_compact_table(table) for table in output.tables[:3]],
            "findings": [finding.model_dump(mode="json") for finding in output.findings],
            "recommendations": output.recommendations,
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, AgentDiagnosticResult):
        return {
            "agent_name": output.agent_name,
            "summary": output.summary,
            "facts": [fact.model_dump(mode="json") for fact in output.facts],
            "tables": [_compact_table(table) for table in output.tables[:3]],
            "findings": [finding.model_dump(mode="json") for finding in output.findings],
            "recommendations": output.recommendations,
            "data_gaps": [gap.model_dump(mode="json") for gap in output.data_gaps],
            "cross_agent_requests": [
                request.model_dump(mode="json") for request in output.cross_agent_requests
            ],
        }
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    return {"summary": str(output), "findings": [], "data_gaps": [], "cross_agent_requests": []}


def _diagnostic_summary(output: Any) -> str:
    return str(getattr(output, "summary", "") or "")


def _diagnostic_findings(output: Any) -> list[dict[str, Any]]:
    return list(_diagnostic_payload(output).get("findings", []))


def _diagnostic_data_gaps(output: Any) -> list[dict[str, Any]]:
    return list(_diagnostic_payload(output).get("data_gaps", []))


def _diagnostic_cross_agent_requests(output: Any) -> list[dict[str, Any]]:
    return list(_diagnostic_payload(output).get("cross_agent_requests", []))


def _normalize_text(value: Any) -> str:
    return str(value or "").lower().replace("ё", "е")


def _gap_period_from_record(gap: Any) -> str:
    corpus = _normalize_text(
        " ".join(
            str(value or "")
            for value in (getattr(gap, "metric", None), getattr(gap, "reason", None))
        )
    )
    dates = re.findall(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4})\b",
        corpus,
    )
    if dates:
        return "|".join(dates[:2])
    relative = re.search(
        r"\b(?:последн\w*|предыдущ\w*|текущ\w*|прошл\w*)\s+"
        r"(?:\d+\s+)?(?:дн\w*|недел\w*|месяц\w*|квартал\w*)\b",
        corpus,
    )
    return " ".join((relative.group(0) if relative else "").split())


def _deduplicate_optimized_gap_records(gaps: list[Any]) -> list[Any]:
    deduplicated: list[Any] = []
    blocked_by_no_data = {
        (
            _intent_from_agent_name(gap.target_agent or gap.source_agent).value,
            _normalize_text(gap.sku),
            _normalize_text(gap.metric),
            _gap_period_from_record(gap),
        )
        for gap in gaps
        if str(gap.evidence_status).lower() == "no_data"
    }
    seen_gap_keys: set[tuple[str, str, str, str]] = set()
    for gap in gaps:
        key = (
            _intent_from_agent_name(gap.target_agent or gap.source_agent).value,
            _normalize_text(gap.sku),
            _normalize_text(gap.metric),
            _gap_period_from_record(gap),
        )
        if key in blocked_by_no_data or key in seen_gap_keys:
            continue
        seen_gap_keys.add(key)
        deduplicated.append(gap)
    return deduplicated


def _text_contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _safety_corpus(
    *,
    user_input: str,
    primary_output: Any,
    round_1_evidence: list[dict[str, Any]],
) -> str:
    parts = [user_input, _diagnostic_summary(primary_output)]
    for payload in [_diagnostic_payload(primary_output)] + [
        evidence.get("output") or {} for evidence in round_1_evidence
    ]:
        parts.append(str(payload.get("summary") or ""))
        parts.extend(str(item) for item in payload.get("recommendations", [])[:5])
        for finding in payload.get("findings", [])[:5]:
            parts.extend(
                str(finding.get(key) or "")
                for key in ("factor", "observation", "impact", "cause", "action")
            )
    return _normalize_text(" ".join(parts))


def _detected_action_safety_families(corpus: str) -> list[str]:
    families: list[str] = []
    for family, config in ACTION_SAFETY_RULES.items():
        if _text_contains_any(corpus, config["patterns"]):
            families.append(family)
    return families


def _requires_action_safety_round(user_input: str, families: list[str]) -> bool:
    if not families:
        return False
    text = _normalize_text(user_input)
    explicit = _text_contains_any(
        text,
        (
            "второй раунд",
            "для каждого предлагаемого действия",
            "для каждого действия",
            "если снижать цену",
            "если масштабировать рекламу",
            "если дозаказывать",
            "если распродавать",
        ),
    )
    broad_diagnostic = (
        len(families) >= 2
        and _text_contains_any(text, ("продаж", "выруч"))
        and _text_contains_any(text, ("прибыл", "марж", "экономик"))
        and _text_contains_any(text, ("гипотез", "причин", "подтвержд", "разлож"))
    )
    return explicit or broad_diagnostic


def _safety_evidence_snippets(payload: dict[str, Any], limit: int = 3) -> list[str]:
    snippets: list[str] = []
    for finding in payload.get("findings", [])[:limit]:
        factor = str(finding.get("factor") or "").strip()
        observation = str(finding.get("observation") or "").strip()
        if factor or observation:
            snippets.append(": ".join(item for item in (factor, observation) if item))
    if snippets:
        return snippets[:limit]
    for table in payload.get("tables", [])[:limit]:
        title = str(table.get("title") or "").strip()
        rows = table.get("rows") or []
        if title and rows:
            snippets.append(f"{title}: есть строки данных")
    return snippets[:limit]


def _json_for_prompt(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, indent=2)


def _payload_without_cross_requests(payload: dict[str, Any]) -> dict[str, Any]:
    clean = dict(payload)
    clean.pop("cross_agent_requests", None)
    return clean


def _plan_items(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return list(plan.get("recommended_secondary_agents") or [])


def _is_inventory_turnover_causal_query(user_query: str) -> bool:
    return _is_inventory_turnover_query(user_query) and _is_explicit_cause_query(user_query)


def _inventory_turnover_required_check(target: QueryIntent) -> tuple[str, str, list[str]]:
    if target == QueryIntent.SALES:
        return (
            "Проверить спрос и продажи для объяснения медленной оборачиваемости.",
            "Проверь продажи, заказы, выкупы, отмены и SKU-драйверы за тот же период: объясняет ли спрос медленную оборачиваемость.",
            ["продажи и заказы", "темп спроса", "SKU-драйверы", "недельная динамика"],
        )
    if target == QueryIntent.PRICE:
        return (
            "Проверить ценовой барьер для SKU с медленной оборачиваемостью.",
            "Проверь цену, скидку и конкурентное позиционирование SKU с медленной оборачиваемостью: есть ли ценовой барьер.",
            ["текущая цена и скидка", "конкурентная медиана", "доля конкурентов дешевле", "ценовой барьер"],
        )
    if target == QueryIntent.MARKETING:
        return (
            "Проверить маркетинговый спрос и видимость для SKU с медленной оборачиваемостью.",
            "Проверь рекламу, видимость, показы, клики, CTR/CR и поисковый спрос SKU с медленной оборачиваемостью.",
            ["показы и клики", "CTR/CR", "видимость", "поисковый спрос"],
        )
    if target == QueryIntent.FINANCE:
        return (
            "Проверить финансовую экономику SKU с медленной оборачиваемостью.",
            "Проверь маржу, комиссии, логистику, хранение, выплаты и прибыльность SKU с медленной оборачиваемостью.",
            ["маржа", "комиссии и логистика", "хранение", "прибыльность SKU"],
        )
    return (
        "Проверить доменный фактор медленной оборачиваемости.",
        "Проверь доменный фактор медленной оборачиваемости.",
        [],
    )


def _plan_has_target(items: list[dict[str, Any]], target: QueryIntent) -> bool:
    return any(_intent_from_agent_name(item.get("target_agent")) == target for item in items)


def _inventory_turnover_coverage_plan(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    existing_items: list[dict[str, Any]] | None = None,
    max_secondary_agents: int,
    query_plan: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    analyses = {
        str(value) for value in list((query_plan or {}).get("requested_analyses") or [])
    }
    raw_required_domains = list((query_plan or {}).get("required_domains") or [])
    has_semantic_plan = bool(
        analyses
        or raw_required_domains
        or str((query_plan or {}).get("objective") or "") not in {"", "answer_user_request"}
    )
    if query_plan and has_semantic_plan:
        mode = str(query_plan.get("query_mode") or "")
        if "inventory_turnover" not in analyses or mode not in {"diagnostic", "decision"}:
            return None
        requested_domains: list[QueryIntent] = []
        for value in raw_required_domains:
            try:
                domain = QueryIntent(str(value))
            except ValueError:
                continue
            if domain not in requested_domains:
                requested_domains.append(domain)
    else:
        if not _is_inventory_turnover_causal_query(user_input):
            return None
        requested_domains = list(INVENTORY_TURNOVER_CAUSAL_DOMAINS)

    existing_items = existing_items or []
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, target in enumerate(requested_domains, start=1):
        reason, question, required_checks = _inventory_turnover_required_check(target)
        item = {
            "target_agent": target.value,
            "reason": reason,
            "question": question,
            "priority": 130 - index,
            "source": COVERAGE_POLICY_SOURCE,
            "required_checks": required_checks,
            "claim_policy": "inventory_turnover_causal",
        }
        if target == primary_agent or _plan_has_target(existing_items, target):
            continue
        items.append(item)

    limit = max(0, max_secondary_agents)
    return {
        "primary_agent": primary_agent.value,
        "recommended_secondary_agents": items[:limit],
        "skipped_agents": [*items[limit:], *skipped],
        "max_secondary_agents": limit,
        "policy": "inventory_turnover_causal",
    }


def _verification_goal_from_item(item: dict[str, Any], *, round_name: str | None = None) -> dict[str, Any]:
    goal: dict[str, Any] = {
        "reason": item.get("reason"),
        "question": item.get("question"),
        "source": item.get("source"),
        "priority": item.get("priority"),
        "required_checks": item.get("required_checks", []),
        "required_tools": item.get("required_tools", []),
    }
    if round_name:
        goal["round"] = round_name
    for key in ("period", "vendor_codes", "nm_ids", "safety_check_ids", "gap_id", "sku", "claim_policy"):
        value = item.get(key)
        if value not in (None, "", []):
            goal[key] = value
    return goal


def _build_action_safety_plan(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    primary_output: Any,
    round_1_evidence: list[dict[str, Any]],
    max_secondary_agents: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    corpus = _safety_corpus(
        user_input=user_input,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
    )
    families = _detected_action_safety_families(corpus)
    if not _requires_action_safety_round(user_input, families):
        return None, []

    primary_payload = _diagnostic_payload(primary_output)
    primary_has_evidence = _payload_has_business_evidence(primary_payload)
    checks: list[dict[str, Any]] = []
    items_by_agent: dict[str, dict[str, Any]] = {}

    for family in families:
        config = ACTION_SAFETY_RULES[family]
        for index, requirement in enumerate(config["checks"], start=1):
            target = requirement["target_agent"]
            target_value = target.value
            check_id = f"{family}:{target_value}:{index}"
            check = {
                "check_id": check_id,
                "action_family": config["label"],
                "required_check": requirement["required_check"],
                "status": "missing",
                "source_agent": target_value,
                "evidence": [],
            }
            if target == primary_agent:
                check["status"] = "covered_by_primary" if primary_has_evidence else "unconfirmed"
                check["evidence"] = _safety_evidence_snippets(primary_payload)
                checks.append(check)
                continue

            checks.append(check)
            item = items_by_agent.get(target_value)
            if item is None:
                item = {
                    "target_agent": target_value,
                    "reason": "Нужно закрыть второй раунд action-safety перед финальными задачами.",
                    "question": "Проверь action-safety перед финальными задачами.",
                    "priority": 110,
                    "source": ACTION_SAFETY_SOURCE,
                    "safety_check_ids": [],
                    "required_checks": [],
                }
                items_by_agent[target_value] = item
            item["safety_check_ids"].append(check_id)
            item["required_checks"].append(requirement["required_check"])

    for item in items_by_agent.values():
        item["question"] = " ".join(
            [
                "Проверь action-safety перед финальными задачами:",
                " ".join(item["required_checks"]),
            ]
        )

    ranked_items = list(items_by_agent.values())
    selected_items = ranked_items[:max(0, max_secondary_agents)]
    skipped_items = ranked_items[max(0, max_secondary_agents):]
    plan = {
        "primary_agent": primary_agent.value,
        "recommended_secondary_agents": selected_items,
        "skipped_agents": skipped_items,
        "max_secondary_agents": max_secondary_agents,
    }
    return plan, checks


def _merge_dependency_plans(
    *,
    primary_agent: QueryIntent,
    plans: list[dict[str, Any] | None],
    max_secondary_agents: int,
) -> dict[str, Any] | None:
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for plan in plans:
        if not plan:
            continue
        items.extend(_plan_items(plan))
        skipped.extend(list(plan.get("skipped_agents") or []))

    if not items and not skipped:
        return None

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(items, key=lambda value: -int(value.get("priority") or 0)):
        key = (
            _intent_from_agent_name(item.get("target_agent")).value,
            _normalize_text(item.get("question")),
        )
        if key in seen:
            skipped.append(item)
            continue
        seen.add(key)
        deduped.append(item)

    limit = max(0, max_secondary_agents)
    selected = deduped[:limit]
    skipped = [*deduped[limit:], *skipped]
    return {
        "primary_agent": primary_agent.value,
        "recommended_secondary_agents": selected,
        "skipped_agents": skipped,
        "max_secondary_agents": limit,
    }


def _claim_verification_status(
    *,
    user_input: str,
    evidence_by_agent: dict[str, bool],
) -> dict[str, Any]:
    if not _is_inventory_turnover_causal_query(user_input):
        return {}

    required_domains = [domain.value for domain in INVENTORY_TURNOVER_CAUSAL_DOMAINS]
    covered_domains = [
        domain
        for domain in required_domains
        if evidence_by_agent.get(domain)
    ]
    missing_domains = [
        domain
        for domain in required_domains
        if not evidence_by_agent.get(domain)
    ]
    return {
        "policy": "inventory_turnover_causal",
        "required_domains": required_domains,
        "covered_domains": covered_domains,
        "missing_domains": missing_domains,
        "status": "complete" if not missing_domains else "incomplete",
        "rule": (
            "Причину медленной оборачиваемости можно называть подтвержденной только "
            "по доменам с covered evidence; для missing_domains формулируй как "
            "`не подтверждено по доступным данным` или `не проверено`."
        ),
    }


def _claim_verification_plan(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    primary_output: Any,
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
    max_secondary_agents: int,
) -> dict[str, Any] | None:
    evidence_by_agent = _business_evidence_by_agent(
        primary_agent=primary_agent,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
    )
    status = _claim_verification_status(
        user_input=user_input,
        evidence_by_agent=evidence_by_agent,
    )
    missing_domains = status.get("missing_domains") or []
    if not missing_domains:
        return None

    items: list[dict[str, Any]] = []
    for index, domain in enumerate(missing_domains, start=1):
        target = _intent_from_agent_name(domain)
        reason, base_question, required_checks = _inventory_turnover_required_check(target)
        items.append(
            {
                "target_agent": target.value,
                "reason": f"Перед финальным выводом нужно проверить сильный causal-claim: {reason}",
                "question": (
                    f"{base_question} Если данных не хватает, явно верни data_gap и не подтверждай причину."
                ),
                "priority": 120 - index,
                "source": CLAIM_VERIFICATION_SOURCE,
                "required_checks": required_checks,
                "claim_policy": status.get("policy"),
            }
        )

    limit = max(0, max_secondary_agents)
    return {
        "primary_agent": primary_agent.value,
        "recommended_secondary_agents": items[:limit],
        "skipped_agents": items[limit:],
        "max_secondary_agents": limit,
        "policy": status.get("policy"),
    }


def _finalize_action_safety_checks(
    checks: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not checks:
        return []

    evidence_by_check_id: dict[str, dict[str, Any]] = {}
    for evidence in followup_evidence:
        for check_id in evidence.get("safety_check_ids") or []:
            evidence_by_check_id[str(check_id)] = evidence

    finalized: list[dict[str, Any]] = []
    for check in checks:
        if check.get("status") in {"covered_by_primary", "unconfirmed"}:
            finalized.append(check)
            continue

        evidence = evidence_by_check_id.get(str(check.get("check_id")))
        if evidence is None:
            finalized.append(check)
            continue

        payload = evidence.get("output") or {}
        typed_status = str(
            (payload.get("_round_result") or {}).get("verdict") or ""
        ).lower()
        status = {
            "safe": "confirmed",
            "unsafe": "unsafe",
            "needs_review": "needs_review",
            "no_data": "missing",
        }.get(
            typed_status,
            "confirmed" if _payload_has_business_evidence(payload) else "unconfirmed",
        )
        if evidence.get("runtime_failure"):
            status = "missing"
        finalized.append(
            {
                **check,
                "status": status,
                "source_agent": str(evidence.get("target_agent") or check.get("source_agent")),
                "evidence": _safety_evidence_snippets(payload),
            }
        )
    return finalized


def _requires_exact_five_actions(user_input: str) -> bool:
    text = _normalize_text(user_input)
    return "ровно 5" in text or "5 задач" in text or "пять задач" in text


def _requires_sku_cause_action_matrix(user_input: str) -> bool:
    text = _normalize_text(user_input)
    has_sales_profit_scope = _text_contains_any(text, ("продаж", "выруч")) and _text_contains_any(
        text,
        ("прибыл", "марж", "экономик"),
    )
    has_sku_scope = _text_contains_any(text, ("sku", "артикул", "ключев"))
    has_diagnostic_scope = _text_contains_any(
        text,
        ("почему", "гипотез", "причин", "подтвержд", "разлож", "вклад"),
    )
    return has_sales_profit_scope and has_sku_scope and has_diagnostic_scope


def _should_include_sku_evidence_matrix(user_input: str) -> bool:
    text = _normalize_text(user_input)
    if _requires_sku_cause_action_matrix(user_input):
        return True
    has_diagnostic_scope = _text_contains_any(
        text,
        ("почему", "гипотез", "причин", "причина", "подтвержд", "разлож", "вклад"),
    )
    has_business_object_scope = _text_contains_any(
        text,
        ("sku", "артикул", "товар", "продаж", "выруч", "прибыл", "марж", "экономик"),
    )
    return has_diagnostic_scope and has_business_object_scope


def _requires_sku_metric_matrix(user_input: str) -> bool:
    text = _normalize_text(user_input)
    if _is_inventory_turnover_query(user_input):
        return True
    metric_markers = (
        "остат",
        "запас",
        "оборачиваем",
        "продаж",
        "заказ",
        "выруч",
        "марж",
        "прибыл",
        "цена",
        "скидк",
        "дрр",
        "расход",
        "клик",
        "кампан",
    )
    sku_scope_markers = ("sku", "скю", "артикул", "товар", "позици", "кажд")
    return _text_contains_any(text, metric_markers) and _text_contains_any(text, sku_scope_markers)


def _table_has_sku_column(table: Any) -> bool:
    if not isinstance(table, dict):
        return False
    columns = table.get("columns") or []
    columns_text = _normalize_text(" ".join(str(column) for column in columns))
    title_text = _normalize_text(table.get("title"))
    return _text_contains_any(
        columns_text,
        ("sku", "скю", "артикул", "товар", "vendor_code", "supplier_article"),
    ) or _text_contains_any(title_text, ("sku", "скю", "артикул"))


def _extract_sku_metric_matrix(payload: dict[str, Any]) -> list[dict[str, Any]]:
    matrices: list[dict[str, Any]] = []
    raw_matrix = payload.get("sku_metric_matrix")
    if isinstance(raw_matrix, list) and raw_matrix:
        matrices.append(
            {
                "title": "SKU-матрица",
                "source": "sku_metric_matrix",
                "rows": raw_matrix,
            }
        )
    for table in payload.get("tables") or []:
        if not _table_has_sku_column(table):
            continue
        matrices.append(
            {
                "title": table.get("title") or "SKU-матрица",
                "source": "primary_evidence_table",
                "columns": table.get("columns") or [],
                "rows": table.get("rows") or [],
            }
        )
    return matrices


def build_query_plan_inventory_matrix(
    query_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Resolve the complete inventory matrix directly from source data.

    The LLM decides whether the matrix is required. Once requested, row
    coverage and derived values are assembled deterministically so synthesis
    cannot shorten the portfolio by choosing only a few illustrative SKUs.
    """

    plan = dict(query_plan or {})
    analyses = {
        str(value) for value in list(plan.get("requested_analyses") or [])
    }
    if not analyses.intersection(
        {"inventory_matrix", "inventory_turnover", "current_stock", "inventory_value"}
    ):
        return []

    scope = str(plan.get("scope") or "")
    exclusive_scope = scope in {"single_sku", "sku_list"}
    vendor_code_filter: list[str] = []
    if exclusive_scope:
        raw_targets = [
            dict(target)
            for target in list(plan.get("targets") or [])
            if isinstance(target, dict) and str(target.get("value") or "").strip()
        ]
        if not raw_targets:
            return []
        target_values = [str(target["value"]).strip() for target in raw_targets]
        try:
            identities = strategic_db.get_sku_identity_map(target_values)
        except Exception:
            logger.warning("query_plan_inventory_target_resolution_failed", exc_info=True)
            return []
        for target in raw_targets:
            value = str(target["value"]).strip()
            target_type = str(target.get("target_type") or "")
            identity = dict(identities.get(value) or {})
            vendor_code = str(identity.get("vendor_code") or "").strip()
            if not vendor_code and target_type == "vendor_code":
                vendor_code = value
            if vendor_code and vendor_code not in vendor_code_filter:
                vendor_code_filter.append(vendor_code)
        if not vendor_code_filter:
            return []

    try:
        supply = strategic_db.get_supply_overview(limit=100)
        report_date_value = (supply.get("totals") or {}).get("report_date")
        if report_date_value is None:
            for row in list(supply.get("rows") or []):
                if row.get("report_date") is not None:
                    report_date_value = row.get("report_date")
                    break
        if report_date_value is None:
            return []
        report_date = date.fromisoformat(str(report_date_value)[:10])
        date_from = report_date.fromordinal(report_date.toordinal() - 29)
        turnover_kwargs: dict[str, Any] = {
            "date_from": date_from.isoformat(),
            "date_to": report_date.isoformat(),
            "limit": 1000,
        }
        if exclusive_scope:
            turnover_kwargs["vendor_codes"] = ",".join(vendor_code_filter)
        turnover = strategic_db.get_inventory_turnover(**turnover_kwargs)
        valuation = strategic_db.get_current_inventory_value(limit=100)
    except Exception:
        logger.warning("query_plan_inventory_matrix_resolution_failed", exc_info=True)
        return []

    values_by_sku = {
        str(row.get("vendor_code")): row
        for row in list(valuation.get("rows") or [])
        if row.get("vendor_code") is not None
    }
    rows: list[dict[str, Any]] = []
    for turnover_row in list(turnover.get("sku_metric_matrix") or []):
        sku = str(turnover_row.get("sku") or "").strip()
        if not sku:
            continue
        value_row = values_by_sku.get(sku, {})
        rows.append(
            {
                "sku": sku,
                "current_stock_qty": turnover_row.get("current_stock_qty"),
                "sales_7d_units": turnover_row.get("sales_7d_units"),
                "sales_30d_units": turnover_row.get("sales_units"),
                "avg_stock_qty": turnover_row.get("avg_stock_qty"),
                "days_on_hand": turnover_row.get("days_on_hand"),
                "turnover_band": turnover_row.get("turnover_band"),
                "availability_status": turnover_row.get("availability_status"),
                "current_stock_value_rub": value_row.get("current_stock_value_rub"),
                "valuation_status": value_row.get("valuation_status"),
            }
        )

    if not rows:
        return []
    return [
        {
            "title": "Полная SKU-матрица из QueryPlan evidence resolver",
            "source": "query_plan_inventory_resolver",
            "period": {
                "date_from": date_from.isoformat(),
                "date_to": report_date.isoformat(),
            },
            "rows": rows,
        }
    ]


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _matrix_rows_as_mappings(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = matrix.get("rows") or []
    columns = matrix.get("columns") or []
    if columns:
        return [
            {
                str(column): row[index] if index < len(row) else None
                for index, column in enumerate(columns)
            }
            for row in rows
            if isinstance(row, list)
        ]
    return [dict(row) for row in rows if isinstance(row, dict)]


def _normalized_matrix_key(value: Any) -> str:
    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^0-9a-zа-яё]+", "_", _normalize_text(value)),
    ).strip("_")


def _normalized_inventory_matrix(
    synthesis_package: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not synthesis_package:
        return None
    matrices = list(synthesis_package.get("sku_metric_matrix") or [])
    if not matrices:
        return None

    merged_rows: dict[str, dict[str, Any]] = {}
    for matrix in matrices:
        if not isinstance(matrix, dict):
            continue
        for raw_row in _matrix_rows_as_mappings(matrix):
            normalized_keys = {
                _normalized_matrix_key(key): value
                for key, value in raw_row.items()
            }
            sku = _first_present(
                normalized_keys,
                "sku",
                "vendor_code",
                "артикул",
                "supplier_article",
            )
            if sku is None:
                continue
            sku_text = str(sku).strip()
            if not sku_text:
                continue

            merged = merged_rows.setdefault(sku_text, {})
            for key, value in normalized_keys.items():
                if value is not None and merged.get(key) is None:
                    merged[key] = value

    normalized_rows: list[list[Any]] = []
    for sku_text, normalized_keys in merged_rows.items():
        available = _first_present(
            normalized_keys,
            "avail_stock",
            "available_stock_qty",
            "current_stock_qty",
            "stock_qty",
            "доступный_остаток",
            "доступный_остаток_шт",
            "текущий_доступный_остаток",
        )
        sales_7d = _first_present(
            normalized_keys,
            "sales_7d_units",
            "sales_7d",
            "продажи_7д",
            "продажи_7д_шт",
        )
        sales_30d = _first_present(
            normalized_keys,
            "sales_30d_units",
            "sales_30d",
            "sales_units",
            "продажи_30д",
            "продажи_30д_шт",
            "продажи_шт",
        )
        average_stock = _first_present(
            normalized_keys,
            "avg_stock_qty",
            "average_stock_qty",
            "средний_остаток",
            "средний_остаток_шт",
        )
        raw_current_coverage = _first_present(
            normalized_keys,
            "days_of_stock",
            "coverage_days",
            "покрытие_7д",
            "покрытие_по_скорости_7д_дней",
        )
        calculated_current_coverage = calculate_coverage_days(
            stock_qty=available,
            sales_units=sales_7d,
            period_days=7,
        )
        current_coverage = (
            calculated_current_coverage
            if calculated_current_coverage is not None
            else raw_current_coverage
        )
        raw_turnover_coverage = _first_present(
            normalized_keys,
            "turnover_30d_days_on_hand",
            "days_on_hand",
            "оборачиваемость_30д",
            "оборачиваемость_по_среднему_остатку_30д_дней",
        )
        calculated_turnover_coverage = calculate_coverage_days(
            stock_qty=average_stock,
            sales_units=sales_30d,
            period_days=30,
        )
        turnover_coverage = (
            calculated_turnover_coverage
            if calculated_turnover_coverage is not None
            else raw_turnover_coverage
        )
        classification_basis = (
            turnover_coverage
            if turnover_coverage is not None
            else current_coverage
        )
        band = classify_coverage_days(classification_basis)
        availability = classify_availability(available)
        inventory_value = _first_present(
            normalized_keys,
            "current_stock_value_rub",
            "stock_value_rub",
            "стоимость_текущего_остатка",
            "стоимость_остатка",
        )
        valuation_status = _first_present(
            normalized_keys,
            "valuation_status",
            "статус_оценки",
        )
        conclusion = (
            f"{availability}; {band}"
            if availability != "доступный остаток есть"
            else band
        )
        normalized_rows.append(
            [
                sku_text,
                available,
                sales_7d,
                sales_30d,
                current_coverage,
                turnover_coverage,
                band,
                availability,
                inventory_value,
                valuation_status,
                conclusion,
            ]
        )

    if not normalized_rows:
        return None
    return {
        "title": "SKU-матрица: остатки, продажи и покрытие",
        "columns": [
            "SKU",
            "Доступный остаток, шт.",
            "Продажи 7д, шт.",
            "Продажи 30д, шт.",
            "Покрытие по скорости 7д, дней",
            "Оборачиваемость по среднему остатку 30д, дней",
            "Сегмент покрытия",
            "Текущее наличие",
            "Стоимость текущего остатка, ₽",
            "Статус оценки",
            "Вывод",
        ],
        "rows": normalized_rows,
    }


def _strip_markdown_section(markdown: str, heading_prefix: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    skipping = False
    target_level = 0
    normalized_prefix = _normalize_text(heading_prefix)
    for line in lines:
        heading = re.match(r"^(#+)\s+(.+?)\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = _normalize_text(heading.group(2))
            if not skipping and title.startswith(normalized_prefix):
                skipping = True
                target_level = level
                continue
            if skipping and level <= target_level:
                skipping = False
        if not skipping:
            output.append(line)
    return "\n".join(output).strip()


def apply_query_plan_output_contract(
    markdown: str,
    synthesis_package: dict[str, Any] | None,
    query_plan: dict[str, Any] | None,
) -> str:
    """Deterministically preserve mandatory QueryPlan artifacts in the answer."""

    plan = dict(query_plan or {})
    analyses = {str(value) for value in plan.get("requested_analyses") or []}
    sections: list[str] = []
    objective = str(plan.get("objective") or "").strip()
    if objective == "answer_user_request":
        objective = ""
    constraints = dict(plan.get("constraints") or {})
    if objective:
        objective_lines = ["## Цель решения", "", objective]
        target_days = constraints.get("target_cover_days")
        horizon_days = constraints.get("horizon_days")
        if target_days is not None or horizon_days is not None:
            details = []
            if target_days is not None:
                details.append(
                    f"целевое покрытие — {_format_plan_number(target_days)} дней"
                )
            if horizon_days is not None:
                details.append(f"горизонт — {_format_plan_number(horizon_days)} дней")
            objective_lines.extend(["", "Параметры: " + "; ".join(details) + "."])
        sections.append("\n".join(objective_lines))

    requires_inventory_matrix = bool(
        analyses.intersection(
            {"inventory_matrix", "inventory_turnover", "current_stock", "inventory_value"}
        )
    )
    if requires_inventory_matrix:
        matrix = _normalized_inventory_matrix(synthesis_package)
        if matrix is not None:
            rendered = render_report_table(ReportTable.model_validate(matrix)).strip()
            sections.append(
                "\n".join(
                    [
                        "## SKU-матрица",
                        "",
                        rendered,
                        "",
                        "Формулы: покрытие 7д = текущий доступный остаток / среднедневные продажи за 7 дней; "
                        "оборачиваемость 30д = средний доступный остаток / среднедневные продажи за период. "
                        "Сегменты рассчитываются кодом: <14, 14–30, >30–60 и >60 дней.",
                    ]
                )
            )
        else:
            sections.append(
                "## SKU-матрица\n\nМатрица запрошена в QueryPlan, но в подтверждённом evidence нет строк SKU."
            )

    body = _strip_markdown_section(markdown, "цель решения")
    body = _strip_markdown_section(body, "sku-матрица")
    if not sections:
        return body
    return "\n\n".join([*sections, body]).strip()


def _format_plan_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _is_nm_id_label(value: Any) -> bool:
    return bool(NM_ID_PATTERN.search(str(value or "").strip()))


def _remove_nm_id_columns_from_table(table: dict[str, Any]) -> dict[str, Any]:
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return table

    keep_indexes = [
        index
        for index, column in enumerate(columns)
        if not _is_nm_id_label(column)
    ]
    if len(keep_indexes) == len(columns):
        return table

    clean = dict(table)
    clean["columns"] = [columns[index] for index in keep_indexes]
    clean["rows"] = [
        [row[index] for index in keep_indexes if isinstance(row, list) and index < len(row)]
        if isinstance(row, list)
        else row
        for row in rows
    ]
    return clean


def _sanitize_user_facing_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("columns"), list) and isinstance(value.get("rows"), list):
            value = _remove_nm_id_columns_from_table(value)
        clean: dict[str, Any] = {}
        for key, item in value.items():
            if _is_nm_id_label(key):
                continue
            clean[key] = _sanitize_user_facing_payload(item)
        return clean
    if isinstance(value, list):
        return [_sanitize_user_facing_payload(item) for item in value]
    if isinstance(value, str):
        return NM_ID_PATTERN.sub("SKU", value)
    return value


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _format_markdown_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _strip_nm_id_from_markdown_table(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    header_cells = _split_markdown_table_row(lines[0])
    remove_indexes = {
        index
        for index, cell in enumerate(header_cells)
        if _is_nm_id_label(cell)
    }
    clean_lines: list[str] = []
    for line_index, line in enumerate(lines):
        cells = _split_markdown_table_row(line)
        kept = [cell for index, cell in enumerate(cells) if index not in remove_indexes]
        if line_index == 1 and MARKDOWN_TABLE_SEPARATOR_PATTERN.match(line):
            kept = ["---" for _ in kept]
        else:
            kept = [_sanitize_user_facing_line(cell) for cell in kept]
        clean_lines.append(_format_markdown_table_row(kept))
    return clean_lines


def _sanitize_user_facing_line(text: str) -> str:
    clean_line = text
    for pattern, replacement in USER_FACING_MARKDOWN_REPLACEMENTS:
        clean_line = pattern.sub(replacement, clean_line)
    clean_line = NM_ID_SLASH_WITH_VALUE_PATTERN.sub("", clean_line)
    clean_line = NM_ID_WITH_VALUE_PATTERN.sub("SKU не сопоставлен", clean_line)
    clean_line = NM_ID_PATTERN.sub("SKU", clean_line)
    return clean_line


def sanitize_user_facing_markdown(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    table_buffer: list[str] = []

    def flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            output.extend(_strip_nm_id_from_markdown_table(table_buffer))
            table_buffer = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            table_buffer.append(line)
            continue
        flush_table()
        output.append(_sanitize_user_facing_line(line))
    flush_table()
    return "\n".join(output)


def _sku_only_display_was_sanitized(original: str, sanitized: str) -> bool:
    """Report only removal of an internal nm_id, not ordinary copy localization."""

    return original != sanitized and NM_ID_PATTERN.search(original) is not None


def remove_empty_markdown_tables(markdown: str) -> tuple[str, int]:
    """Replace header-only Markdown tables with an explicit no-data statement."""

    lines = markdown.splitlines()
    output: list[str] = []
    table: list[str] = []
    removed = 0

    def flush() -> None:
        nonlocal table, removed
        if not table:
            return
        is_table = len(table) >= 2 and bool(
            MARKDOWN_TABLE_SEPARATOR_PATTERN.match(table[1])
        )
        if is_table and len(table) == 2:
            output.append(
                "Данные для таблицы не подтверждены; пустая таблица не публикуется."
            )
            removed += 1
        else:
            output.extend(table)
        table = []

    for line in lines:
        if line.strip().startswith("|") and "|" in line:
            table.append(line)
            continue
        flush()
        output.append(line)
    flush()
    return "\n".join(output), removed


def _is_sku_cause_matrix_heading(line: str) -> bool:
    normalized = _normalize_text(line.lstrip("#").strip(" :"))
    return normalized.startswith("почему это произошло")


def _strip_unrequested_sku_cause_matrix(
    markdown: str,
    *,
    user_query: str,
    sku_evidence_matrix: list[dict[str, Any]] | None = None,
) -> str:
    if _should_include_sku_evidence_matrix(user_query) and sku_evidence_matrix:
        return markdown

    lines = markdown.splitlines()
    output: list[str] = []
    skipping = False
    removed = False
    for line in lines:
        if _is_sku_cause_matrix_heading(line):
            skipping = True
            removed = True
            continue
        if skipping and line.lstrip().startswith("#"):
            skipping = False
        if not skipping:
            output.append(line)

    if not removed:
        return markdown
    return "\n".join(output).strip()


def _is_inventory_turnover_query(user_query: str) -> bool:
    text = _normalize_text(user_query)
    return "оборачиваем" in text and any(marker in text for marker in ("склад", "запас"))


def _is_explicit_cause_query(user_query: str) -> bool:
    text = _normalize_text(user_query)
    return any(
        marker in text
        for marker in (
            "почему",
            "причин",
            "гипотез",
            "из-за чего",
            "из за чего",
            "что влияет",
        )
    )


TURNOVER_CAUSE_OVERCLAIM_REPLACEMENTS = (
    (
        re.compile(
            r"(?i)\bпричин[аы]\s+замедлен\w*\s+подтвержден[аоы]?\b"
        ),
        "подтвержден механический вклад в медленную оборачиваемость",
    ),
    (
        re.compile(
            r"(?i)\bпричин[аы]\s+медленн\w*\s+оборачиваемост\w*\s+подтвержден[аоы]?\b"
        ),
        "подтвержден механический вклад в медленную оборачиваемость",
    ),
)


def _sanitize_inventory_turnover_cause_claims(markdown: str, *, user_query: str) -> str:
    if not _is_inventory_turnover_query(user_query) or _is_explicit_cause_query(user_query):
        return markdown

    cleaned = markdown
    for pattern, replacement in TURNOVER_CAUSE_OVERCLAIM_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def _sales_report_snapshot_from_package(synthesis_package: dict[str, Any]) -> ReportTable | None:
    primary_evidence = synthesis_package.get("primary_evidence")
    if not isinstance(primary_evidence, dict):
        return None
    snapshot = primary_evidence.get("sales_report_snapshot")
    if not snapshot:
        return None
    try:
        return ReportTable.model_validate(snapshot)
    except Exception:
        logger.exception("Failed to parse sales_report_snapshot from synthesis package.")
        return None


def _marketing_report_snapshot_from_package(synthesis_package: dict[str, Any]) -> ReportTable | None:
    primary_evidence = synthesis_package.get("primary_evidence")
    if not isinstance(primary_evidence, dict):
        return None
    snapshot = primary_evidence.get("marketing_report_snapshot")
    if not snapshot:
        return None
    try:
        return ReportTable.model_validate(snapshot)
    except Exception:
        logger.exception("Failed to parse marketing_report_snapshot from synthesis package.")
        return None


def _final_answer_has_sales_report_snapshot(markdown: str) -> bool:
    head = markdown.lstrip()[:2000].lower()
    has_periods = all(
        any(marker in head for marker in marker_group)
        for marker_group in SALES_SNAPSHOT_MARKER_GROUPS
    )
    return (
        head.startswith("**sales-сводка**")
        or head.startswith("| сравнение")
        or (head.startswith("<table") and has_periods)
        or (head.startswith("сводка продаж") and has_periods)
    )


def _final_answer_has_marketing_report_snapshot(markdown: str) -> bool:
    head = markdown.lstrip()[:2000].lower()
    has_periods = all(
        any(marker in head for marker in marker_group)
        for marker_group in MARKETING_SNAPSHOT_MARKER_GROUPS
    )
    return (
        head.startswith("**marketing-сводка**")
        or (head.startswith("| сравнение") and "дрр" in head)
        or (head.startswith("<table") and has_periods)
        or (head.startswith("сводка маркетинга") and has_periods)
    )


def _is_sales_snapshot_markdown_table(lines: list[str]) -> bool:
    if not lines:
        return False
    text = "\n".join(lines).lower()
    return all(
        any(marker in text for marker in marker_group)
        for marker_group in SALES_SNAPSHOT_MARKER_GROUPS
    )


def _is_marketing_snapshot_markdown_table(lines: list[str]) -> bool:
    if not lines:
        return False
    text = "\n".join(lines).lower()
    return all(
        any(marker in text for marker in marker_group)
        for marker_group in MARKETING_SNAPSHOT_MARKER_GROUPS
    )


def _strip_existing_sales_snapshot_tables(markdown: str) -> str:
    without_html = SALES_SNAPSHOT_HTML_TABLE_PATTERN.sub("\n\n", markdown, count=1)
    lines = without_html.splitlines()
    output: list[str] = []
    index = 0
    removed = False

    while index < len(lines):
        line = lines[index]
        if "|" not in line or not line.strip().startswith("|"):
            output.append(line)
            index += 1
            continue

        start = index
        while index < len(lines) and "|" in lines[index] and lines[index].strip().startswith("|"):
            index += 1
        table_lines = lines[start:index]

        if not removed and _is_sales_snapshot_markdown_table(table_lines):
            removed = True
            while output and not output[-1].strip():
                output.pop()
            if output and output[-1].strip().lower() in {"**sales-сводка**", "сводка продаж"}:
                output.pop()
            continue

        output.extend(table_lines)

    return "\n".join(output)


def _strip_existing_marketing_snapshot_tables(markdown: str) -> str:
    without_html = MARKETING_SNAPSHOT_HTML_TABLE_PATTERN.sub("\n\n", markdown, count=1)
    lines = without_html.splitlines()
    output: list[str] = []
    index = 0
    removed = False

    while index < len(lines):
        line = lines[index]
        if "|" not in line or not line.strip().startswith("|"):
            output.append(line)
            index += 1
            continue

        start = index
        while index < len(lines) and "|" in lines[index] and lines[index].strip().startswith("|"):
            index += 1
        table_lines = lines[start:index]

        if not removed and _is_marketing_snapshot_markdown_table(table_lines):
            removed = True
            while output and not output[-1].strip():
                output.pop()
            if output and output[-1].strip().lower() in {"**marketing-сводка**", "сводка маркетинга"}:
                output.pop()
            continue

        output.extend(table_lines)

    return "\n".join(output)


def _prepend_sales_report_snapshot(
    markdown: str,
    synthesis_package: dict[str, Any] | None,
) -> str:
    if not synthesis_package or _final_answer_has_sales_report_snapshot(markdown):
        return markdown
    table = _sales_report_snapshot_from_package(synthesis_package)
    if table is None:
        return markdown
    rendered = render_report_table(table).strip()
    if not rendered:
        return markdown
    markdown_without_duplicate = _strip_existing_sales_snapshot_tables(markdown).lstrip()
    return f"{rendered}\n\n{markdown_without_duplicate}"


def _prepend_marketing_report_snapshot(
    markdown: str,
    synthesis_package: dict[str, Any] | None,
) -> str:
    if not synthesis_package or _final_answer_has_marketing_report_snapshot(markdown):
        return markdown
    table = _marketing_report_snapshot_from_package(synthesis_package)
    if table is None:
        return markdown
    rendered = render_report_table(table).strip()
    if not rendered:
        return markdown
    markdown_without_duplicate = _strip_existing_marketing_snapshot_tables(markdown).lstrip()
    return f"{rendered}\n\n{markdown_without_duplicate}"


def _prepend_required_report_snapshots(
    markdown: str,
    synthesis_package: dict[str, Any] | None,
) -> str:
    markdown = _prepend_sales_report_snapshot(markdown, synthesis_package)
    return _prepend_marketing_report_snapshot(markdown, synthesis_package)


def _final_action_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    in_action_section = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:#+\s*)?что делать\s*:?\s*$", line, flags=re.IGNORECASE):
            in_action_section = True
            continue
        if in_action_section and re.match(r"^(?:#+\s+|[A-Za-zА-Яа-я0-9 _-]+:\s*$)", line):
            break
        if in_action_section and re.match(r"^\d+[.)]\s+", line):
            lines.append(line)
    return lines


def _final_action_table_rows(markdown: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_action_section = False
    table_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:#+\s*)?что делать\s*:?\s*$", line, flags=re.IGNORECASE):
            in_action_section = True
            table_lines = []
            continue
        if in_action_section and re.match(r"^(?:#+\s+|[A-Za-zА-Яа-я0-9 _-]+:\s*$)", line):
            break
        if not in_action_section:
            continue
        if "|" not in line or not line.startswith("|"):
            if table_lines:
                break
            continue
        table_lines.append(line)

    if len(table_lines) < 2:
        return rows

    header = _split_markdown_table_row(table_lines[0])
    normalized = [cell.strip().lower().replace("ё", "е") for cell in header]
    try:
        task_index = normalized.index("задача")
        object_index = normalized.index("объект")
        why_index = normalized.index("почему")
    except ValueError:
        return rows

    data_lines = table_lines[1:]
    if data_lines and MARKDOWN_TABLE_SEPARATOR_PATTERN.match(data_lines[0]):
        data_lines = data_lines[1:]

    for line in data_lines:
        cells = _split_markdown_table_row(line)
        if max(task_index, object_index, why_index) >= len(cells):
            continue
        rows.append(
            {
                "task": cells[task_index].strip(),
                "object": cells[object_index].strip(),
                "why": cells[why_index].strip(),
                "raw": line,
            }
        )
    return rows


def validate_final_diagnostic_markdown(markdown: str, user_query: str = "") -> list[str]:
    text = _normalize_text(markdown)
    query_text = _normalize_text(user_query)
    violations: list[str] = []

    if NM_ID_PATTERN.search(markdown):
        violations.append("final markdown exposes nm_id")

    if BROKEN_DOMAIN_LOCALIZATION_PATTERN.search(markdown):
        violations.append("final markdown contains broken domain localization")

    volume_as_primary = re.search(r"главн\w+\s+фактор[^.\n]*(?:снижен|паден)[^.\n]*(?:объем|обьем)", text)
    if volume_as_primary and not any(
        marker in text
        for marker in ("механическ", "не первоприч", "первичн", "почему")
    ):
        violations.append("volume decline is stated as root cause without mechanical-effect caveat")

    for line in _final_action_lines(markdown):
        if RISKY_ACTION_PATTERN.search(line) and not ACTION_CONDITION_PATTERN.search(line):
            violations.append(f"risky action lacks condition: {line}")

    action_table_rows = _final_action_table_rows(markdown)
    for row in action_table_rows:
        row_text = " ".join(str(row.get(key) or "") for key in ("task", "object", "why"))
        if ACTION_TABLE_FORBIDDEN_PATTERN.search(row_text):
            violations.append(f"action table row contains fallback phrasing: {row['raw']}")
        if BARE_ACTION_TABLE_TASK_PATTERN.search(str(row.get("task") or "")):
            violations.append(f"action table row is a bare verb: {row['raw']}")

    if _requires_exact_five_actions(user_query):
        action_count = len(action_table_rows) if action_table_rows else len(_final_action_lines(markdown))
        if action_count and action_count != 5:
            violations.append(f"expected 5 final action tasks, got {action_count}")

    if (
        ("подтвержден" in query_text or "подтвержд" in query_text)
        and ("неподтвержден" in query_text or "неподтвержд" in query_text)
        and "подтверждено" not in text
        and "матриц" not in text
    ):
        violations.append("answer misses confirmed causes section")
    if (
        ("неподтвержден" in query_text or "неподтвержд" in query_text)
        and "неподтверждено" not in text
        and "неподтверждено" not in text
        and "матриц" not in text
    ):
        violations.append("answer misses unconfirmed causes section or hypothesis matrix")

    return violations


def _asked_item(target_agent: str, question: str) -> dict[str, str]:
    return {
        "target_agent": _intent_from_agent_name(target_agent).value,
        "question": question,
    }


def _evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    output = evidence.get("output") or {}
    return {
        "round": evidence.get("round"),
        "target_agent": evidence.get("target_agent"),
        "reason": evidence.get("reason"),
        "question": evidence.get("question"),
        "summary": output.get("summary"),
        "findings": output.get("findings", [])[:4],
        "tables": output.get("tables", [])[:2],
        "data_gaps": output.get("data_gaps", [])[:3],
    }


def _sales_secondary_contract_text() -> str:
    return (
        "\nДля target_agent=sales действует жёсткий контракт: всегда заполняй "
        "`sales_decomposition` строками из `get_sales_sku_revenue_change_report`. "
        "Даже если verification_goal узкий про заказы, выкупы или воронку, таблица "
        "`sales_decomposition` обязана содержать колонки текущей выручки, предыдущей "
        "выручки, Δ выручки, текущих продаж, предыдущих продаж и Δ продаж. "
        "Дополнительные узкие факты добавляй в `sales_overview`, `sales_dynamics` "
        "или findings, но не заменяй обязательную decomposition-таблицу произвольной таблицей.\n"
    )


def _build_secondary_task_prompt(
    *,
    user_input: str,
    item: dict[str, Any],
    primary_output: Any,
    prior_evidence: list[dict[str, Any]] | None = None,
    workflow_version: WorkflowVersion | str = WorkflowVersion.V1,
    round_type: RoundType | str | None = None,
    allowed_tools: list[str] | None = None,
    required_tools: list[str] | tuple[str, ...] = (),
    query_plan: dict[str, Any] | None = None,
    return_build_result: bool = False,
) -> str | Any:
    target = _intent_from_agent_name(item.get("target_agent"))
    version = resolve_workflow_version(workflow_version)
    resolved_round = _round_type_from_name(round_type or item.get("round") or "round_1")
    if version != WorkflowVersion.V1:
        verification_goal = _verification_goal_from_item(
            item,
            round_name=getattr(resolved_round, "value", str(resolved_round)),
        )
        claim_policy_value = item.get("claim_policy")
        if claim_policy_value is None and isinstance(verification_goal, dict):
            claim_policy_value = verification_goal.get("claim_policy")
        if claim_policy_value is not None and not isinstance(claim_policy_value, str):
            claim_policy_value = _json_for_prompt(claim_policy_value)
        task_period = str(item.get("period")) if item.get("period") is not None else None
        query_has_explicit_period = bool(
            re.search(
                r"(?:\d{4}-\d{2}-\d{2}|\d+\s*(?:дн(?:я|ей)?|недел\w*|месяц\w*|days?|weeks?|months?))",
                _normalize_text(user_input),
            )
        )
        if "get_inventory_turnover" in required_tools and not query_has_explicit_period:
            normalized_period = _normalize_text(task_period)
            if task_period is None:
                task_period = "last complete 30-day period ending at the latest available database date"
            elif _text_contains_any(
                normalized_period,
                ("current snapshot", "latest available", "текущ", "последн"),
            ):
                task_period = (
                    "inventory snapshot: latest available; turnover: last complete 30-day period "
                    "ending at the latest available database date"
                )
        task = RoundTask(
            round_type=resolved_round,
            target_agent=target.value,
            question=str(item.get("question") or user_input),
            original_user_query=user_input,
            verification_goal=verification_goal,
            required_checks=[str(value) for value in (item.get("required_checks") or [])],
            source=str(item.get("source") or "orchestrator"),
            gap_id=item.get("gap_id"),
            claim_policy=claim_policy_value,
            period=task_period,
            sku=(str(item.get("sku")) if item.get("sku") is not None else None),
            vendor_codes=[str(value) for value in (item.get("vendor_codes") or [])],
            nm_ids=list(item.get("nm_ids") or []),
            allowed_tools=list(allowed_tools or []),
            required_tools=[str(value) for value in required_tools],
            query_plan=dict(query_plan or {}),
        )
        result = build_round_prompt(
            task,
            _diagnostic_payload(primary_output),
            list(prior_evidence or []),
            version,
        )
        return result if return_build_result else result.text

    task_payload = {
        "original_user_query": user_input,
        "target_agent": target.value,
        "verification_goal": _verification_goal_from_item(item),
        "primary_evidence": _payload_without_cross_requests(_diagnostic_payload(primary_output)),
        "known_cross_evidence": [
            _evidence_for_prompt(evidence) for evidence in (prior_evidence or [])
        ],
    }
    return (
        "Служебная смежная проверка для скрытого оркестратора.\n"
        "Ответь только в своей доменной зоне ответственности и верни обычный структурированный результат агента.\n"
        "Не пересказывай весь primary-ответ. Подтверди или опровергни именно verification_goal.\n"
        "Если вывод требует другого домена, заполни cross_agent_requests; не пиши пользователю `спросить агента`.\n"
        "Data gaps добавляй только если они реально блокируют проверку verification_goal; технические детали не выноси в пользовательские поля.\n\n"
        f"{_sales_secondary_contract_text() if target == QueryIntent.SALES else ''}"
        f"{_json_for_prompt(task_payload)}"
    )


def _run_domain_evidence(
    *,
    item: dict[str, Any],
    user_input: str,
    primary_output: Any,
    prior_evidence: list[dict[str, Any]] | None = None,
    context: EcomRunContext,
    hooks: Any,
    round_name: str,
    allow_failure: bool,
    run_id: int | None = None,
    question_record: AgentRunQuestionRecord | None = None,
) -> dict[str, Any]:
    target = _intent_from_agent_name(item.get("target_agent"))
    question = str(item.get("question") or user_input)
    workflow_version = _workflow_version_for_context(context)
    round_type = _round_type_from_name(round_name)
    context.round_type = round_type
    context.memory_mode = memory_mode_for_round(round_type)
    tool_policy = _domain_tool_policy(
        target=target,
        user_query=user_input,
        round_name=round_name,
        question_id=question_record.id if question_record else None,
        explicit_required_tools=[
            *(item.get("required_tools") or ()),
            *(
                (
                    _v2_required_tools_for_plan(target, context.query_plan)
                    if context.query_plan
                    else _v2_required_tools_for_query(target, user_input)
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
                    question,
                    _json_for_prompt(_verification_goal_from_item(item, round_name=round_name)),
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
        question=question,
        status="started",
        details={
            "source": item.get("source"),
            "priority": item.get("priority"),
        },
    )
    prompt_build = _build_secondary_task_prompt(
        user_input=user_input,
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
    _update_agent_run_question_status_safely(question_record, "running")
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
        payload = _attach_prompt_report(
            _attach_tool_report(_diagnostic_payload(output), context),
            context,
        )
    except ToolPolicyViolation as exc:
        payload = _attach_prompt_report(
            _attach_tool_report(_diagnostic_payload(exc.output), context),
            context,
        )
        payload["_tool_report"] = exc.tool_report
        runtime_failure = {
            "target_agent": target.value,
            **exc.runtime_failure(),
        }
    except OutputGuardrailTripwireTriggered as exc:
        if not allow_failure:
            context.add_decision_trace(
                "secondary_agent_completed",
                round=round_name,
                target_agent=target.value,
                question=question,
                status="guardrail_failed",
                duration_ms=int((time.monotonic() - started_at) * 1000),
                details={"violations": _extract_guardrail_violations(exc)},
            )
            raise
        violations = _extract_guardrail_violations(exc)
        agent_output = exc.guardrail_result.agent_output
        payload = _diagnostic_payload(agent_output)
        if not payload.get("summary"):
            payload["summary"] = _coerce_output(agent_output)
        payload = _attach_prompt_report(_attach_tool_report(payload, context), context)
        payload.setdefault("tables", [])
        payload.setdefault("findings", [])
        payload.setdefault("recommendations", [])
        payload.setdefault("data_gaps", [])
        payload.setdefault("cross_agent_requests", [])
        runtime_failure = {
            "target_agent": target.value,
            "kind": "guardrail",
            "message": "Смежная проверка не прошла внутреннюю проверку качества.",
            "violations": violations,
        }
    except Exception as exc:
        if not allow_failure:
            context.add_decision_trace(
                "secondary_agent_completed",
                round=round_name,
                target_agent=target.value,
                question=question,
                status="failed",
                duration_ms=int((time.monotonic() - started_at) * 1000),
                details={"error_type": type(exc).__name__},
            )
            raise
        payload = {
            "summary": "Смежную проверку не удалось выполнить.",
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [],
            "cross_agent_requests": [],
        }
        payload = _attach_prompt_report(payload, context)
        runtime_failure = {
            "target_agent": target.value,
            "kind": "exception",
            "message": str(exc),
        }
    original_gap = (
        _find_agent_run_gap_safely(run_id, item.get("gap_id"))
        if round_name == "round_3"
        else None
    )
    if round_name == "round_3":
        payload = _enforce_gap_resolution_fact_policy(payload, original_gap)
    persisted_response = _persist_agent_run_response_safely(
        run_id=run_id,
        question_record=question_record,
        agent_name=target.value,
        payload=payload,
        runtime_failure=runtime_failure,
        blocking_gaps=bool(runtime_failure) or not _payload_has_business_evidence(payload),
        gap_filter=(
            (lambda gap: should_persist_gap_from_gap_resolution(gap, original_gap))
            if round_name == "round_3"
            else None
        ),
    )
    gap_resolution_status: str | None = None
    if round_name == "round_3":
        gap_resolution_status = classify_gap_resolution_response(
            payload,
            original_gap=original_gap,
            runtime_failure=runtime_failure,
        )
        if gap_resolution_status in {"resolved", "accepted_limitation"}:
            _update_agent_run_gap_status_safely(item.get("gap_id"), gap_resolution_status)
    _update_agent_run_question_status_safely(
        question_record,
        "failed" if runtime_failure else "completed",
    )
    context.cross_agent_trace.append(f"{round_name}:{target.value}:{question}")
    context.add_decision_trace(
        "secondary_agent_completed",
        round=round_name,
        target_agent=target.value,
        question=question,
        status="failed" if runtime_failure else "completed",
        duration_ms=int((time.monotonic() - started_at) * 1000),
        details={
            "runtime_failure": (
                {
                    "kind": runtime_failure.get("kind"),
                    "violations": runtime_failure.get("violations", []),
                }
                if runtime_failure
                else None
            ),
            "has_business_evidence": _payload_has_business_evidence(payload),
        },
    )
    return {
        "round": round_name,
        "target_agent": target.value,
        "reason": item.get("reason"),
        "question": question,
        "source": item.get("source"),
        "priority": item.get("priority"),
        "safety_check_ids": item.get("safety_check_ids", []),
        "output": payload,
        **({"gap_resolution_status": gap_resolution_status} if gap_resolution_status else {}),
        **({"response_id": persisted_response.id} if persisted_response else {}),
        **({"runtime_failure": runtime_failure} if runtime_failure else {}),
    }


def _sync_cross_check_concurrency() -> int:
    raw_value = os.getenv(
        "AGENT_SYNC_CROSS_CHECK_CONCURRENCY",
        str(DEFAULT_SYNC_CROSS_CHECK_CONCURRENCY),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_SYNC_CROSS_CHECK_CONCURRENCY


def _isolated_round_context(context: EcomRunContext) -> EcomRunContext:
    return replace(
        context,
        data_gaps=[gap.model_copy(deep=True) for gap in context.data_gaps],
        agent_trace=[],
        tool_trace=[],
        cross_agent_trace=[],
        decision_trace=[],
        last_tool_report=None,
        last_prompt_report=None,
    )


def _merge_round_contexts(
    context: EcomRunContext,
    child_contexts: list[EcomRunContext],
) -> None:
    for child in child_contexts:
        context.agent_trace.extend(child.agent_trace)
        context.tool_trace.extend(child.tool_trace)
        context.cross_agent_trace.extend(child.cross_agent_trace)
    child_events = [
        event
        for child in child_contexts
        for event in child.decision_trace
    ]
    context.decision_trace.extend(
        sorted(child_events, key=lambda event: str(event.get("ts") or ""))
    )


def _run_round_1_evidence_batch(
    *,
    items: list[dict[str, Any]],
    questions: list[AgentRunQuestionRecord | None],
    user_input: str,
    primary_output: Any,
    context: EcomRunContext,
    run_id: int | None,
) -> list[dict[str, Any]]:
    """Run independent first-round domain checks concurrently.

    Each Runner invocation receives its own mutable context and hooks. The
    resulting traces are merged only after all calls complete, avoiding shared
    list and tool-policy state between threads.
    """

    if not items:
        return []
    max_workers = min(len(items), _sync_cross_check_concurrency())
    child_contexts = [_isolated_round_context(context) for _ in items]
    started_at = time.monotonic()
    context.add_decision_trace(
        "parallel_cross_check_started",
        round="round_1",
        status="started",
        details={"question_count": len(items), "max_workers": max_workers},
    )

    def run_one(
        item: dict[str, Any],
        question: AgentRunQuestionRecord | None,
        child_context: EcomRunContext,
    ) -> dict[str, Any]:
        return _run_domain_evidence(
            item=item,
            user_input=user_input,
            primary_output=primary_output,
            prior_evidence=[],
            context=child_context,
            hooks=build_run_hooks(),
            round_name="round_1",
            allow_failure=True,
            run_id=run_id,
            question_record=question,
        )

    if max_workers == 1:
        evidence = [
            run_one(item, question, child_context)
            for item, question, child_context in zip(items, questions, child_contexts)
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agent-cross-check",
        ) as executor:
            futures = [
                executor.submit(run_one, item, question, child_context)
                for item, question, child_context in zip(items, questions, child_contexts)
            ]
            evidence = [future.result() for future in futures]

    _merge_round_contexts(context, child_contexts)
    context.add_decision_trace(
        "parallel_cross_check_completed",
        round="round_1",
        status="completed",
        duration_ms=int((time.monotonic() - started_at) * 1000),
        details={"question_count": len(items), "max_workers": max_workers},
    )
    return evidence


def _table_has_rows(table: dict[str, Any]) -> bool:
    return bool(table.get("rows"))


def _payload_has_business_evidence(payload: dict[str, Any]) -> bool:
    return bool(payload.get("findings")) or any(
        _table_has_rows(table) for table in payload.get("tables", [])
    )


NO_DATA_GAP_MARKERS = (
    "no_data",
    "нет данных",
    "нет строк",
    "не найдено строк",
    "не нашлось строк",
    "отсутствуют строки",
    "отсутствуют данные",
    "пустой результат",
    "вернул пустой",
    "не вернули корректные строки",
    "не вернул корректные строки",
    "не закрыта данными",
    "не закрыт данными",
    "нельзя подтвердить",
    "причинную роль",
)
GAP_RESOLUTION_MARKERS = (
    "подтвержден",
    "подтверждён",
    "подтверждается",
    "частично закрыт",
    "частично закрыта",
    "закрыт не как no_data",
    "закрыта не как no_data",
    "не как no_data",
    "опроверг",
    "не является причиной",
    "фактор снят",
)


def _payload_text_for_gap_resolution(payload: dict[str, Any]) -> str:
    parts: list[str] = [str(payload.get("summary") or "")]
    for collection_name in ("findings", "data_gaps", "recommendations", "tables"):
        collection = payload.get(collection_name) or []
        if not isinstance(collection, list):
            continue
        for item in collection:
            if isinstance(item, dict):
                parts.extend(str(value or "") for value in item.values())
            else:
                parts.append(str(item or ""))
    return " ".join(parts).lower().replace("ё", "е")


def _gap_resolution_has_no_data_signal(payload: dict[str, Any]) -> bool:
    text = _payload_text_for_gap_resolution(payload)
    return any(marker in text for marker in NO_DATA_GAP_MARKERS)


def _gap_resolution_has_resolution_signal(payload: dict[str, Any]) -> bool:
    text = _payload_text_for_gap_resolution(payload)
    return any(marker in text for marker in GAP_RESOLUTION_MARKERS)


def _enforce_gap_resolution_fact_policy(
    payload: dict[str, Any],
    original_gap: AgentRunGapRecord | None,
) -> dict[str, Any]:
    """Downgrade a scenario price that is mislabeled as a verified price floor."""

    if original_gap is None:
        return payload
    metric = _normalize_text(original_gap.metric)
    if not (
        ("minimum" in metric and "profitable price" in metric)
        or "минимальн" in metric and "цен" in metric and "прибыл" in metric
    ):
        return payload
    round_result = payload.get("_round_result") or {}
    if str(round_result.get("verdict") or "").strip().lower() != "resolved":
        return payload
    facts = [fact for fact in round_result.get("facts") or [] if isinstance(fact, dict)]
    verified_refs = (
        "minimum_profitable_price",
        "break_even_price",
        "safe_price_floor",
        "price_floor",
    )
    if any(
        any(marker in _normalize_text(fact.get("evidence_ref")) for marker in verified_refs)
        for fact in facts
    ):
        return payload

    downgraded = dict(payload)
    downgraded_result = dict(round_result)
    downgraded_result["verdict"] = "no_data"
    downgraded_result["facts"] = []
    downgraded["_round_result"] = downgraded_result
    downgraded["gap_resolution_status"] = "no_data"
    downgraded["summary"] = (
        "No_data: положительная маржа при одной проверенной цене не подтверждает "
        "минимально прибыльную цену или безопасный ценовой порог."
    )
    downgraded["findings"] = []
    downgraded["tables"] = []
    downgraded["recommendations"] = []
    downgraded["data_gaps"] = []
    downgraded["cross_agent_requests"] = []
    return downgraded


def classify_gap_resolution_response(
    payload: dict[str, Any],
    original_gap: AgentRunGapRecord | None = None,
    runtime_failure: dict[str, Any] | None = None,
) -> str:
    """Classify a round-3 response without turning source-empty answers into new work."""
    if runtime_failure:
        return "open"
    typed_outcome = str(
        (payload.get("_round_result") or {}).get("verdict")
        or payload.get("gap_resolution_status")
        or ""
    ).strip().lower()
    if typed_outcome == "resolved":
        return "resolved"
    if typed_outcome in {"no_data", "accepted_limitation"}:
        # `no_data` is the typed model outcome; the persisted gap lifecycle uses
        # `accepted_limitation` so the same unavailable source is not retried.
        return "accepted_limitation"
    if _gap_resolution_has_no_data_signal(
        payload
    ) and not _gap_resolution_has_resolution_signal(payload):
        return "accepted_limitation"
    if _payload_has_business_evidence(payload):
        return "resolved"
    if _gap_resolution_has_no_data_signal(payload):
        return "accepted_limitation"
    return "open"


def should_persist_gap_from_gap_resolution(
    new_gap: dict[str, Any],
    original_gap: AgentRunGapRecord | None,
) -> bool:
    if original_gap is None:
        return True
    if _gap_resolution_has_no_data_signal(
        {
            "summary": " ".join(
                str(new_gap.get(key) or "")
                for key in ("metric", "reason", "impact", "evidence_status")
            ),
            "data_gaps": [new_gap],
        }
    ):
        return False

    new_sku = str(new_gap.get("sku") or "")
    original_sku = str(original_gap.sku or "")
    new_source = str(new_gap.get("source_agent") or "")
    original_source = str(original_gap.source_agent or "")
    new_target = str(new_gap.get("target_agent") or "")
    original_target = str(original_gap.target_agent or "")
    new_metric = _normalized_key(new_gap.get("metric"))
    original_metric = _normalized_key(original_gap.metric)

    if original_sku and new_sku == original_sku:
        return False
    if new_metric and original_metric and (new_metric in original_metric or original_metric in new_metric):
        return False
    same_domain = (
        bool(new_source and new_source in {original_source, original_target})
        or bool(new_target and new_target in {original_source, original_target})
    )
    if same_domain and (not new_sku or not original_sku):
        return False
    return True


def _find_agent_run_gap_safely(run_id: int | None, gap_id: Any) -> AgentRunGapRecord | None:
    if run_id is None or gap_id is None:
        return None
    try:
        gap_id_int = int(gap_id)
    except (TypeError, ValueError):
        return None
    try:
        for gap in list_agent_run_gaps(run_id=run_id):
            if gap.id == gap_id_int:
                return gap
    except Exception as exc:
        logger.warning("agent_gap_load_failed run_id=%s gap_id=%s error=%s", run_id, gap_id, exc)
    return None


def _normalized_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def _is_sku_column(column: Any) -> bool:
    text = _normalized_key(column)
    if _is_nm_id_label(text):
        return False
    return (
        text == "sku"
        or "артикул" in text
        or "vendor_code" in text
        or "supplier_article" in text
    )


def _looks_like_sku(value: Any) -> bool:
    text = str(value or "").strip()
    if not NUMERIC_SKU_PATTERN.fullmatch(text):
        return False
    if len(text) == 4 and text.startswith(("19", "20")):
        return False
    return True


def _extract_context_skus_from_text(value: Any) -> list[str]:
    text = str(value or "")
    skus: list[str] = []
    for match in SKU_CONTEXT_PATTERN.finditer(text):
        sku = match.group(1)
        if _looks_like_sku(sku) and sku not in skus:
            skus.append(sku)
    return skus


def _extract_explicit_skus_from_user_query(value: str) -> list[str]:
    skus = _extract_context_skus_from_text(value)
    if skus:
        return skus
    explicit: list[str] = []
    for match in NUMERIC_SKU_PATTERN.finditer(value):
        sku = match.group(0)
        if _looks_like_sku(sku) and sku not in explicit:
            explicit.append(sku)
    return explicit


def _row_cells(row: Any, columns: list[Any]) -> list[Any]:
    if isinstance(row, dict):
        return [row.get(column) for column in columns]
    if isinstance(row, list):
        return row
    return []


def _row_value(row: Any, columns: list[Any], index: int) -> Any:
    if isinstance(row, dict):
        if index < len(columns):
            return row.get(columns[index])
        return None
    if isinstance(row, list) and index < len(row):
        return row[index]
    return None


def _table_sku_indexes(table: dict[str, Any]) -> list[int]:
    columns = table.get("columns")
    if not isinstance(columns, list):
        return []
    return [index for index, column in enumerate(columns) if _is_sku_column(column)]


def _extract_skus_from_table(table: dict[str, Any]) -> list[str]:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    sku_indexes = _table_sku_indexes(table)
    skus: list[str] = []

    for row in rows:
        for index in sku_indexes:
            sku = str(_row_value(row, columns, index) or "").strip()
            if _looks_like_sku(sku) and sku not in skus:
                skus.append(sku)

    if skus:
        return skus

    # Some agent-created tables omit a clear SKU header but title the table as SKU-level.
    table_hint = _normalized_key(" ".join(str(item) for item in [table.get("title"), *columns]))
    if "sku" not in table_hint and "артикул" not in table_hint:
        return []

    for row in rows:
        for cell in _row_cells(row, columns)[:3]:
            sku = str(cell or "").strip()
            if _looks_like_sku(sku) and sku not in skus:
                skus.append(sku)
    return skus


def _extract_skus_from_payload(payload: dict[str, Any]) -> list[str]:
    skus: list[str] = []
    for table in payload.get("tables", []):
        if not isinstance(table, dict):
            continue
        for sku in _extract_skus_from_table(table):
            if sku not in skus:
                skus.append(sku)
    for finding in payload.get("findings", []):
        for sku in _extract_context_skus_from_text(finding):
            if sku not in skus:
                skus.append(sku)
    return skus


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace("₽", "").replace(" ", "").replace(",", ".")
    if not text or text in {"—", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sku_weight_from_row(row: Any, columns: list[Any]) -> float:
    weight = 0.0
    for index, column in enumerate(columns):
        column_text = _normalized_key(column)
        value = _as_float(_row_value(row, columns, index))
        if value is None:
            continue
        if "δ" in column_text or "delta" in column_text or "дельт" in column_text or "измен" in column_text:
            weight = max(weight, abs(value))
        elif "марж" in column_text:
            weight = max(weight, abs(value) * 0.5)
    return weight


def _collect_sku_universe(
    *,
    user_input: str,
    primary_payload: dict[str, Any],
    evidence_payloads: list[dict[str, Any]],
    max_skus: int = 12,
) -> list[str]:
    scores: dict[str, float] = {}
    order: dict[str, int] = {}

    def add_sku(sku: str, score: float = 0.0) -> None:
        if not _looks_like_sku(sku):
            return
        order.setdefault(sku, len(order))
        scores[sku] = max(scores.get(sku, 0.0), score)

    for sku in _extract_explicit_skus_from_user_query(user_input):
        add_sku(sku, score=1_000_000_000.0)

    for payload in [primary_payload, *evidence_payloads]:
        for table in payload.get("tables", []):
            if not isinstance(table, dict):
                continue
            columns = table.get("columns") if isinstance(table.get("columns"), list) else []
            rows = table.get("rows") if isinstance(table.get("rows"), list) else []
            sku_indexes = _table_sku_indexes(table)
            if not sku_indexes and not _extract_skus_from_table(table):
                continue
            for row in rows:
                row_score = _sku_weight_from_row(row, columns)
                for index in sku_indexes:
                    sku = str(_row_value(row, columns, index) or "").strip()
                    add_sku(sku, score=row_score)
            for sku in _extract_skus_from_table(table):
                add_sku(sku)
        for sku in _extract_skus_from_payload(payload):
            add_sku(sku)

    ranked = sorted(scores, key=lambda sku: (-scores[sku], order[sku]))
    return ranked[:max_skus]


def _format_metric_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value).strip()


def _preferred_fact_columns(domain: str) -> tuple[str, ...]:
    return {
        "sales": ("заказ", "продаж", "выруч", "открыт", "корз", "delta", "дельт", "δ"),
        "marketing": ("кампан", "расход", "заказ", "клик", "показ", "cpc", "ctr", "видим", "позици"),
        "price": ("статус", "цен", "медиан", "разрыв", "дороже", "дешевле", "скид"),
        "supply": ("статус", "остат", "доступ", "покрыт", "critical", "warning", "дозаказ"),
        "finance": ("марж", "рентаб", "выруч", "расход", "комис", "δ", "дельт"),
    }.get(domain, ())


def _fact_from_table_row(domain: str, table: dict[str, Any], row: Any) -> str:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    preferred = _preferred_fact_columns(domain)
    parts: list[str] = []
    for index, column in enumerate(columns):
        if _is_sku_column(column) or _is_nm_id_label(column):
            continue
        column_text = _normalized_key(column)
        if preferred and not any(token in column_text for token in preferred):
            continue
        value = _format_metric_value(_row_value(row, columns, index))
        if not value or value in {"—", "-"}:
            continue
        label = str(column).strip()
        parts.append(f"{label}: {value}")
        if len(parts) >= 3:
            break
    if parts:
        return "; ".join(parts)
    title = str(table.get("title") or "").strip()
    return f"{title}: есть строка данных" if title else "есть строка данных"


def _status_from_row(domain: str, table: dict[str, Any], row: Any) -> str:
    row_text = _normalize_text({"columns": table.get("columns", []), "row": row})
    if any(
        marker in row_text
        for marker in (
            "not_a_driver",
            "not a driver",
            "опроверг",
            "не найден",
            "не выявлен",
            "не обнаруж",
            "дефицит не найден",
        )
    ):
        return "refuted"
    if any(marker in row_text for marker in ("не подтверждено", "не подтвержд")):
        return "unconfirmed"
    if any(marker in row_text for marker in ("частично", "warning", "усилитель", "amplifier")):
        return "partial"
    if domain in {"finance", "sales"}:
        return "confirmed"
    if domain == "supply" and any(marker in row_text for marker in ("critical", "остат", "доступ", "покрыт")):
        return "confirmed"
    if domain == "price" and any(marker in row_text for marker in ("выше", "ниже", "медиан", "разрыв", "рын")):
        return "partial"
    if domain == "marketing" and any(marker in row_text for marker in ("заказ", "расход", "кампан", "видим", "клик")):
        return "partial"
    return "partial"


def _entry_from_finding(domain: str, sku: str, finding: dict[str, Any]) -> dict[str, Any] | None:
    finding_text = _normalize_text(finding)
    if sku not in finding_text and sku not in _extract_context_skus_from_text(finding):
        return None
    driver_class = str(finding.get("driver_class") or "")
    if driver_class == "not_a_driver":
        status = "refuted"
    elif driver_class == "amplifier":
        status = "partial"
    else:
        status = "confirmed"
    fact = str(finding.get("observation") or finding.get("cause") or finding.get("factor") or "").strip()
    return {
        "status": status,
        "fact": fact or "фактор найден в выводах агента",
        "reason": None,
        "source": "finding",
        "source_agent": domain,
    }


def _domain_entry_from_payload(domain: str, sku: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for table in payload.get("tables", []):
        if not isinstance(table, dict):
            continue
        columns = table.get("columns") if isinstance(table.get("columns"), list) else []
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        sku_indexes = _table_sku_indexes(table)
        if not sku_indexes:
            continue
        for row in rows:
            row_skus = {
                str(_row_value(row, columns, index) or "").strip()
                for index in sku_indexes
            }
            if sku not in row_skus:
                continue
            candidate = {
                "status": _status_from_row(domain, table, row),
                "fact": _fact_from_table_row(domain, table, row),
                "reason": None,
                "source": "table",
                "source_agent": domain,
                "table_title": table.get("title"),
            }
            if best is None or SKU_STATUS_PRIORITY[candidate["status"]] > SKU_STATUS_PRIORITY[best["status"]]:
                best = candidate

    for finding in payload.get("findings", []):
        if not isinstance(finding, dict):
            continue
        candidate = _entry_from_finding(domain, sku, finding)
        if candidate is None:
            continue
        if best is None or SKU_STATUS_PRIORITY[candidate["status"]] > SKU_STATUS_PRIORITY[best["status"]]:
            best = candidate
    return best


def _default_domain_entry(domain: str, called: bool, has_evidence: bool, sku_in_payload: bool) -> dict[str, Any]:
    if not called:
        return {
            "status": "not_checked",
            "fact": "не проверялось",
            "reason": "domain_not_called",
            "source_agent": domain,
        }
    if not has_evidence:
        return {
            "status": "no_data",
            "fact": "нет данных",
            "reason": "agent_failed_without_evidence",
            "source_agent": domain,
        }
    return {
        "status": "unconfirmed" if sku_in_payload else "no_data",
        "fact": (
            "фактор проверен, подтверждения нет"
            if sku_in_payload
            else "нет строк по SKU в источнике"
        ),
        "reason": "metric_absent" if sku_in_payload else "sku_not_in_source",
        "source_agent": domain,
    }


def _build_sku_evidence_matrix(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    primary_payload: dict[str, Any],
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    payloads_by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in SKU_EVIDENCE_DOMAINS}
    payloads_by_domain[primary_agent.value].append(primary_payload)
    for evidence in [*round_1_evidence, *followup_evidence]:
        domain = str(evidence.get("target_agent") or "")
        if domain in payloads_by_domain:
            payloads_by_domain[domain].append(evidence.get("output") or {})
    called_domains = [
        domain
        for domain, payloads in payloads_by_domain.items()
        if payloads
    ]

    all_payloads = [primary_payload, *[evidence.get("output") or {} for evidence in [*round_1_evidence, *followup_evidence]]]
    sku_universe = _collect_sku_universe(
        user_input=user_input,
        primary_payload=primary_payload,
        evidence_payloads=all_payloads[1:],
    )
    coverage = {
        domain: {
            "confirmed": 0,
            "partial": 0,
            "refuted": 0,
            "unconfirmed": 0,
            "no_data": 0,
            "not_checked": 0,
        }
        for domain in SKU_EVIDENCE_DOMAINS
    }
    missing_reasons: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for sku in sku_universe:
        domain_entries: dict[str, dict[str, Any]] = {}
        for domain in SKU_EVIDENCE_DOMAINS:
            payloads = payloads_by_domain[domain]
            called = bool(payloads)
            has_evidence = any(_payload_has_business_evidence(payload) for payload in payloads)
            sku_in_payload = any(sku in _extract_skus_from_payload(payload) for payload in payloads)
            best: dict[str, Any] | None = None
            for payload in payloads:
                candidate = _domain_entry_from_payload(domain, sku, payload)
                if candidate is None:
                    continue
                if best is None or SKU_STATUS_PRIORITY[candidate["status"]] > SKU_STATUS_PRIORITY[best["status"]]:
                    best = candidate
            entry = best or _default_domain_entry(domain, called, has_evidence, sku_in_payload)
            coverage[domain][entry["status"]] += 1
            if entry.get("status") in {"no_data", "not_checked"}:
                missing_reasons.append(
                    {
                        "sku": sku,
                        "domain": domain,
                        "reason": str(entry.get("reason") or "unknown"),
                    }
                )
            domain_entries[domain] = entry

        conclusion = _sku_matrix_conclusion(domain_entries)
        rows.append(
            {
                "sku": sku,
                "domains": domain_entries,
                "summary": conclusion["fact"],
                "conclusion": conclusion,
            }
        )

    return {
        "rows": rows,
        "domain_coverage": coverage,
        "missing_domain_reasons": missing_reasons,
        "called_domains": called_domains,
    }


def _sku_matrix_summary(domain_entries: dict[str, dict[str, Any]]) -> str:
    return _sku_matrix_conclusion(domain_entries)["fact"]


def _sales_fact_is_primary_factor(fact: Any) -> bool:
    text = _normalize_text(fact)
    if any(marker in text for marker in ("конверс", "корзин", "открыт", "выкуп", "отмен", "0 заказ")):
        return True
    mechanic_markers = (
        "продаж",
        "выруч",
        "заказ",
        "сниз",
        "упал",
        "просел",
        "паден",
        "объем",
        "обьем",
    )
    return not any(marker in text for marker in mechanic_markers)


def _sku_matrix_conclusion(domain_entries: dict[str, dict[str, Any]]) -> dict[str, str]:
    for domain in ("marketing", "price", "supply", "finance"):
        entry = domain_entries.get(domain, {})
        if entry.get("status") in {"confirmed", "partial"} and entry.get("fact"):
            return {
                "status": str(entry.get("status")),
                "domain": domain,
                "fact": str(entry.get("fact")),
            }

    sales = domain_entries.get("sales", {})
    if (
        sales.get("status") in {"confirmed", "partial"}
        and sales.get("fact")
        and _sales_fact_is_primary_factor(sales.get("fact"))
    ):
        return {
            "status": str(sales.get("status")),
            "domain": "sales",
            "fact": str(sales.get("fact")),
        }
    if sales.get("status") in {"confirmed", "partial"}:
        return {
            "status": "unconfirmed",
            "domain": "cross_domain",
            "fact": "изменение подтверждено, первопричина не закрыта",
        }

    if any(entry.get("status") == "refuted" for entry in domain_entries.values()):
        return {
            "status": "refuted",
            "domain": "cross_domain",
            "fact": "проверенные гипотезы не подтвердились",
        }
    if all(entry.get("status") == "not_checked" for entry in domain_entries.values()):
        return {
            "status": "not_checked",
            "domain": "cross_domain",
            "fact": "причина не проверялась",
        }
    if any(entry.get("status") == "no_data" for entry in domain_entries.values()):
        return {
            "status": "no_data",
            "domain": "cross_domain",
            "fact": "причина не закрыта из-за отсутствия данных",
        }
    return {
        "status": "unconfirmed",
        "domain": "cross_domain",
        "fact": "первичная причина не подтверждена после проверки",
    }


def _sku_matrix_quality_violations(matrix: dict[str, Any]) -> list[str]:
    rows = matrix.get("rows") or []
    if not rows:
        return []
    violations: list[str] = []
    total = len(rows)
    coverage = matrix.get("domain_coverage") or {}
    called_domains = set(matrix.get("called_domains") or [])
    for domain, counts in coverage.items():
        if domain not in called_domains:
            continue
        if not isinstance(counts, dict):
            continue
        no_data_count = sum(
            1
            for row in rows
            if (
                ((row.get("domains") or {}).get(domain) or {}).get("status") == "no_data"
                and ((row.get("domains") or {}).get(domain) or {}).get("reason")
                != "sku_not_in_source"
            )
        )
        if no_data_count / total > 0.5:
            violations.append(
                f"sku evidence matrix has high no_data ratio for {domain}: {no_data_count}/{total}"
            )
    return violations


def _gap_entries_from_sku_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str, str]] = set()
    for row in matrix.get("rows") or []:
        sku = str(row.get("sku") or "").strip() or None
        domains = row.get("domains") or {}
        if not isinstance(domains, dict):
            continue
        for domain, entry in domains.items():
            if not isinstance(entry, dict):
                continue
            evidence_status = str(entry.get("status") or "")
            if evidence_status not in {"no_data", "unconfirmed", "refuted"}:
                continue
            explicit_reason = str(entry.get("reason") or "")
            if explicit_reason in {"domain_not_called", "sku_not_in_source"} or evidence_status == "refuted":
                continue
            if evidence_status == "unconfirmed" and explicit_reason not in PERSISTABLE_SKU_GAP_REASONS:
                continue
            reason = explicit_reason or str(entry.get("fact") or "Фактор не подтвержден.")
            key = (sku, str(domain), reason)
            if key in seen:
                continue
            seen.add(key)
            gaps.append(
                {
                    "sku": sku,
                    "source_agent": str(domain),
                    "target_agent": str(domain),
                    "metric": f"{domain} factor",
                    "reason": reason,
                    "impact": "Фактор нельзя использовать как подтвержденную причину в финальном выводе.",
                    "blocking": True,
                    "status": "open",
                    "evidence_status": evidence_status,
                }
            )
    return gaps


def _gap_record_payload(gap: AgentRunGapRecord) -> dict[str, Any]:
    return {
        "source_agent": gap.source_agent,
        "target_agent": gap.target_agent,
        "sku": gap.sku,
        "metric": gap.metric,
        "reason": gap.reason,
        "impact": gap.impact,
        "blocking": gap.blocking,
        "status": gap.status,
        "evidence_status": gap.evidence_status,
    }


def _open_blocking_gap_payloads_safely(
    run_id: int | None,
    statuses: tuple[str, ...] = ("open",),
) -> list[dict[str, Any]]:
    if run_id is None:
        return []
    try:
        gaps: list[dict[str, Any]] = []
        for status in statuses:
            gaps.extend(
                _gap_record_payload(gap)
                for gap in list_agent_run_gaps(
                    run_id=run_id,
                    status=status,
                    blocking=True,
                )
            )
        return gaps
    except Exception as exc:
        logger.warning("agent_open_gaps_load_failed run_id=%s error=%s", run_id, exc)
        return []


def _load_persisted_evidence_safely(run_id: int | None) -> dict[str, Any] | None:
    if run_id is None:
        return None
    try:
        rows = list_agent_run_responses(run_id)
    except Exception as exc:
        logger.warning("agent_persisted_evidence_load_failed run_id=%s error=%s", run_id, exc)
        return None
    if not rows:
        return None

    primary_output: dict[str, Any] | None = None
    round_1_evidence: list[dict[str, Any]] = []
    followup_evidence: list[dict[str, Any]] = []
    for row in rows:
        response: AgentRunResponseRecord = row["response"]
        payload = response.raw_output
        verification_goal = dict(row.get("verification_goal") or {})
        evidence = {
            "round": row.get("round_type") or f"round_{row.get('round_number')}",
            "target_agent": _intent_from_agent_name(row.get("target_agent")).value,
            "reason": row.get("reason"),
            "question": row.get("question"),
            "source": verification_goal.get("source"),
            "priority": verification_goal.get("priority"),
            "gap_id": verification_goal.get("gap_id"),
            "claim_policy": verification_goal.get("claim_policy"),
            "safety_check_ids": list(verification_goal.get("safety_check_ids") or []),
            "output": payload,
            **({"runtime_failure": response.runtime_failure} if response.runtime_failure else {}),
        }
        if row.get("round_type") == "primary":
            primary_output = payload
        elif row.get("round_type") == "cross_check":
            round_1_evidence.append(evidence)
        elif row.get("round_type") in {
            "followup",
            "gap_resolution",
            "claim_verification",
        }:
            followup_evidence.append(evidence)

    if primary_output is None:
        return None
    return {
        "primary_output": primary_output,
        "round_1_evidence": round_1_evidence,
        "followup_evidence": followup_evidence,
    }


def _evidence_quality_summary(
    *,
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    evidence_items = [*round_1_evidence, *followup_evidence]
    guardrail_failures = [
        evidence["runtime_failure"]
        for evidence in evidence_items
        if (evidence.get("runtime_failure") or {}).get("kind") == "guardrail"
    ]
    return {
        "guardrail_failures": guardrail_failures,
        "trimmed_evidence_count": 0,
        "missing_domain_reasons": matrix.get("missing_domain_reasons", []),
    }


def _finding_fact(source_agent: str, finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_agent": source_agent,
        "factor": finding.get("factor"),
        "observation": finding.get("observation"),
        "impact": finding.get("impact"),
        "cause": finding.get("cause"),
        "confidence": finding.get("confidence"),
        "driver_class": finding.get("driver_class"),
        "evidence_status": finding.get("evidence_status"),
        "verification_verdict": finding.get("verification_verdict"),
        "evidence": finding.get("evidence", []),
    }


def _typed_fact_for_synthesis(
    source_agent: str,
    fact: dict[str, Any],
    *,
    verification_verdict: str | None = None,
) -> dict[str, Any]:
    # `source_table`, `evidence_ref`, and the fact-level `source_agent` are all
    # generated by the model.  Matching domain-looking prefixes in those
    # strings is therefore not provenance: a secondary agent can manufacture
    # all three fields.  The caller's source is persisted orchestration
    # metadata (`target_agent`/primary route), so keep that authoritative.  A
    # future cross-domain tool can relabel a fact only after its *runtime* tool
    # report is threaded here and checked against an exact tool allow-list.
    fact_source_agent = _normalize_text(source_agent) or str(source_agent)
    return {
        "source_agent": fact_source_agent,
        "metric": fact.get("metric"),
        "value": fact.get("value"),
        "unit": fact.get("unit"),
        "period": fact.get("period"),
        "period_start": fact.get("period_start"),
        "period_end": fact.get("period_end"),
        "sku": fact.get("sku"),
        "target_type": fact.get("target_type"),
        "target_id": fact.get("target_id"),
        "source_table": fact.get("source_table"),
        "source_snapshot_at": fact.get("source_snapshot_at"),
        "retrieved_at": fact.get("retrieved_at"),
        "metric_definition": fact.get("metric_definition"),
        "freshness_status": fact.get("freshness_status"),
        "freshness_lag_days": fact.get("freshness_lag_days"),
        "statement": fact.get("statement"),
        "evidence_ref": fact.get("evidence_ref"),
        "status": fact.get("status"),
        "confidence": fact.get("confidence"),
        "verification_verdict": verification_verdict,
    }


def _gap_entry(source_agent: str, gap: dict[str, Any], blocking: bool) -> dict[str, Any]:
    return {
        "source_agent": source_agent,
        "metric": gap.get("metric"),
        "reason": gap.get("reason"),
        "impact": gap.get("impact"),
        "blocking": blocking,
    }


def _runtime_gap(evidence: dict[str, Any]) -> dict[str, Any]:
    target = str(evidence.get("target_agent") or "secondary")
    return {
        "source_agent": target,
        "metric": f"{target} factor",
        "reason": "Смежная проверка не дала подтверждаемых фактов по доступным данным.",
        "impact": "Фактор нельзя использовать как подтверждённую причину в итоговом выводе.",
        "blocking": True,
    }


def _runtime_quality_gap(evidence: dict[str, Any]) -> dict[str, Any]:
    target = str(evidence.get("target_agent") or "secondary")
    failure = evidence.get("runtime_failure") or {}
    violations = failure.get("violations") or []
    reason = "Смежная проверка вернула факты, но часть формата не прошла внутреннюю проверку качества."
    if violations:
        reason += " " + "; ".join(str(item) for item in violations[:3])
    return {
        "source_agent": target,
        "metric": f"{target} answer quality",
        "reason": reason,
        "impact": "Факты можно использовать осторожно, но не как самостоятельное пользовательское ограничение.",
        "blocking": False,
    }


GAP_RESOLUTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sales": (
        "sales",
        "продаж",
        "выруч",
        "заказ",
        "выкуп",
        "отмен",
        "фактическ",
        "реализац",
    ),
    "marketing": (
        "marketing",
        "маркет",
        "реклам",
        "кампан",
        "показ",
        "клик",
        "ctr",
        "cpc",
        "дрр",
        "видим",
        "поисков",
        "трафик",
        "промо",
    ),
    "price": (
        "price",
        "цен",
        "скид",
        "дешев",
        "дороже",
        "конкурент",
        "переоцен",
        "сегмент",
        "discount",
    ),
    "supply": (
        "supply",
        "остат",
        "склад",
        "налич",
        "доступн",
        "дефиц",
        "out-of-stock",
        "постав",
        "запас",
        "оборач",
        "покрыт",
        "days of stock",
        "stock coverage",
    ),
    "finance": (
        "finance",
        "финанс",
        "марж",
        "прибыл",
        "комис",
        "логист",
        "хранен",
        "штраф",
        "выплат",
        "себестоим",
        "cogs",
        "romi",
        "mer",
        "окупаем",
    ),
}
GAP_RESOLUTION_DOMAIN_ORDER = ("marketing", "price", "supply", "finance", "sales")

GAP_RESOLUTION_DOMAIN_OVERRIDES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "finance",
        (
            "net profit",
            "чистая прибыль",
            "чистой прибыли",
            "полного p&l",
            "финансовый результат",
            "финрезульт",
        ),
    ),
    (
        "sales",
        (
            "фактическая выруч",
            "фактическую выруч",
            "фактические продажи",
            "фактических продаж",
            "sales-выруч",
            "данных sales",
        ),
    ),
    (
        "marketing",
        (
            "traffic source attribution",
            "источник траф",
            "атрибуц",
            "кампан",
            "поисков",
            "инкрементальн",
        ),
    ),
    (
        "finance",
        (
            "profitability",
            "after costs",
            "contribution after costs",
            "прибыльност",
            "марж",
            "комисс",
            "логист",
            "хранен",
            "штраф",
            "себестоим",
            "cogs",
        ),
    ),
)


def _gap_text(gap: dict[str, Any]) -> str:
    return " ".join(
        str(gap.get(key) or "")
        for key in ("metric", "reason", "impact")
    ).lower().replace("ё", "е")


def _gap_keyword_matches(text: str, keyword: str) -> bool:
    if keyword == "показ":
        # Match impressions (показ/показы/показов), but not the generic word
        # "показатель", which previously sent sales/supply gaps to marketing.
        return bool(
            re.search(
                r"(?<![a-zа-я0-9])показ(?:а|ов|ы|ам|ами|ах|у|ом)?(?![a-zа-я0-9])",
                text,
            )
        )
    if keyword in {"цен", "заказ"}:
        # These roots must start a word: "оценить" is not about price and
        # "дозаказ" is a supply action rather than a marketing order metric.
        return bool(
            re.search(
                rf"(?<![a-zа-я0-9]){re.escape(keyword)}[а-я]*(?![a-zа-я0-9])",
                text,
            )
        )
    if keyword == "цен":
        # Price stems must not match verbs such as "оценить"/"оценка".
        return bool(
            re.search(
                r"(?<![a-zа-я0-9])цен(?:а|ы|е|у|ой|ою|ам|ами|ах|ник\w*|ов\w*)?(?![a-zа-я0-9])",
                text,
            )
        )
    return keyword in text


def _gap_resolution_domain(gap: dict[str, Any]) -> str | None:
    text = _gap_text(gap)
    for domain, keywords in GAP_RESOLUTION_DOMAIN_OVERRIDES:
        if any(_gap_keyword_matches(text, keyword) for keyword in keywords):
            return domain
    for domain in GAP_RESOLUTION_DOMAIN_ORDER:
        keywords = GAP_RESOLUTION_KEYWORDS[domain]
        if any(_gap_keyword_matches(text, keyword) for keyword in keywords):
            return domain
    return None


def _gap_target_agent(source_agent: str, gap: dict[str, Any]) -> str:
    resolution_domain = _gap_resolution_domain(gap)
    return resolution_domain or source_agent


def _business_evidence_by_agent(
    *,
    primary_agent: QueryIntent,
    primary_output: Any,
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
) -> dict[str, bool]:
    evidence_by_agent = {
        primary_agent.value: _payload_has_business_evidence(_diagnostic_payload(primary_output))
    }
    for evidence in [*round_1_evidence, *followup_evidence]:
        source_agent = str(evidence.get("target_agent") or "secondary")
        payload = evidence.get("output") or {}
        evidence_by_agent[source_agent] = (
            evidence_by_agent.get(source_agent, False)
            or _payload_has_business_evidence(payload)
        )
    return evidence_by_agent


def _gap_resolved_by_evidence(
    gap: dict[str, Any],
    *,
    evidence_by_agent: dict[str, bool],
) -> str | None:
    if str(gap.get("metric") or "").endswith(" factor"):
        return None
    source_agent = str(gap.get("source_agent") or "")
    resolution_domain = _gap_resolution_domain(gap)
    if (
        resolution_domain
        and resolution_domain != source_agent
        and evidence_by_agent.get(resolution_domain)
    ):
        return resolution_domain
    return None


def _should_keep_persisted_blocking_gap(gap: dict[str, Any]) -> bool:
    metric = str(gap.get("metric") or "")
    reason = str(gap.get("reason") or "")
    evidence_status = str(gap.get("evidence_status") or "")
    if reason == "sku_not_in_source":
        return False
    if (
        metric.endswith(" factor")
        and evidence_status == "unconfirmed"
        and reason not in PERSISTABLE_SKU_GAP_REASONS
    ):
        return False
    return True


def _is_non_actionable_sku_source_gap(gap: Any) -> bool:
    reason = getattr(gap, "reason", None)
    if reason is None and isinstance(gap, dict):
        reason = gap.get("reason")
    return str(reason or "") == "sku_not_in_source"


def _resolve_cross_domain_gaps(
    gaps: list[dict[str, Any]],
    *,
    evidence_by_agent: dict[str, bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blocking_gaps: list[dict[str, Any]] = []
    non_blocking_gaps: list[dict[str, Any]] = []
    resolved_gaps: list[dict[str, Any]] = []

    for gap in gaps:
        resolved_by = _gap_resolved_by_evidence(gap, evidence_by_agent=evidence_by_agent)
        if resolved_by:
            resolved_gaps.append(
                {
                    **gap,
                    "blocking": False,
                    "resolved_by": resolved_by,
                }
            )
            continue

        if gap.get("blocking"):
            blocking_gaps.append(gap)
        else:
            non_blocking_gaps.append(gap)

    return blocking_gaps, non_blocking_gaps, resolved_gaps


def _trim_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _trim_payload_for_synthesis(payload: dict[str, Any]) -> dict[str, Any]:
    clean = _payload_without_cross_requests(payload)
    clean["tables"] = _trim_list(clean.get("tables"), 3)
    clean["findings"] = _trim_list(clean.get("findings"), 5)
    clean["recommendations"] = _trim_list(clean.get("recommendations"), 5)
    clean["data_gaps"] = _trim_list(clean.get("data_gaps"), 3)
    return clean


def _collect_manager_action_candidates(
    primary_payload: dict[str, Any],
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Carry only explicit structured candidates produced by deterministic tools.

    Free-text recommendations are deliberately ignored. This keeps the legacy
    domain schemas compatible while allowing a tool or versioned calculator to
    attach a complete ManagerAction candidate for the hard gate.
    """

    payloads = [
        primary_payload,
        *[
            evidence.get("output") or {}
            for evidence in [*round_1_evidence, *followup_evidence]
            if isinstance(evidence, dict)
        ],
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        values: list[Any] = []
        for key in (
            "manager_action_candidates",
            "managerActionCandidates",
            "action_candidates",
            "actionCandidates",
        ):
            raw = payload.get(key)
            if isinstance(raw, list):
                values.extend(raw)
        for value in values:
            if not isinstance(value, dict):
                continue
            fingerprint = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(dict(value))
    return candidates


def _reduce_evidence(
    *,
    primary_agent: QueryIntent,
    primary_output: Any,
    round_1_evidence: list[dict[str, Any]],
    followup_evidence: list[dict[str, Any]],
    user_input: str = "",
) -> dict[str, Any]:
    confirmed_facts: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    raw_gaps: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    evidence_by_agent: dict[str, bool] = {}

    primary_payload = _diagnostic_payload(primary_output)
    evidence_by_agent[primary_agent.value] = _payload_has_business_evidence(primary_payload)
    for typed_fact in primary_payload.get("facts", []):
        if not isinstance(typed_fact, dict):
            continue
        fact = _typed_fact_for_synthesis(primary_agent.value, typed_fact)
        fact_status = str(typed_fact.get("status") or "confirmed").lower()
        freshness_status = str(typed_fact.get("freshness_status") or "").lower()
        if fact_status == "refuted":
            contradictions.append(fact)
        elif (
            fact_status == "confirmed"
            and typed_fact.get("value") is not None
            and freshness_status != "future"
        ):
            confirmed_facts.append(fact)
        else:
            raw_gaps.append(
                {
                    "source_agent": primary_agent.value,
                    "metric": typed_fact.get("metric") or f"{primary_agent.value} fact",
                    "reason": "Primary evidence remained unconfirmed or had no usable value.",
                    "impact": typed_fact.get("statement"),
                    "blocking": False,
                }
            )
    for finding in primary_payload.get("findings", []):
        fact = _finding_fact(primary_agent.value, finding)
        if finding.get("driver_class") == "not_a_driver":
            contradictions.append(fact)
        else:
            confirmed_facts.append(fact)
    for gap in primary_payload.get("data_gaps", []):
        raw_gaps.append(_gap_entry(primary_agent.value, gap, blocking=True))

    supporting_evidence: list[dict[str, Any]] = []
    for evidence in [*round_1_evidence, *followup_evidence]:
        source_agent = str(evidence.get("target_agent") or "secondary")
        payload = evidence.get("output") or {}
        runtime_failure = evidence.get("runtime_failure") or {}
        policy_failed = runtime_failure.get("kind") == "tool_policy"
        typed_round_result = payload.get("_round_result") or {}
        verification_verdict = str(typed_round_result.get("verdict") or "").lower()
        typed_non_evidence = verification_verdict in {
            "no_data",
            "unverified",
            "accepted_limitation",
        }
        has_business_evidence = (
            False
            if policy_failed or typed_non_evidence
            else _payload_has_business_evidence(payload)
        )
        evidence_by_agent[source_agent] = evidence_by_agent.get(source_agent, False) or has_business_evidence

        if runtime_failure:
            runtime_failures.append(runtime_failure)
            raw_gaps.append(
                _runtime_quality_gap(evidence)
                if has_business_evidence
                else _runtime_gap(evidence)
            )
            if policy_failed:
                supporting_evidence.append(
                    {
                        "round": evidence.get("round"),
                        "target_agent": source_agent,
                        "question": evidence.get("question"),
                        "reason": evidence.get("reason"),
                        "summary": str(payload.get("summary") or ""),
                        "has_business_evidence": False,
                        "has_runtime_failure": True,
                    }
                )
                continue

        typed_facts = list(typed_round_result.get("facts") or [])
        if typed_facts:
            for typed_fact in typed_facts:
                evidence_status = str(typed_fact.get("status") or "confirmed").lower()
                fact = _typed_fact_for_synthesis(
                    source_agent,
                    typed_fact,
                    verification_verdict=verification_verdict,
                )
                if evidence_status == "refuted" or verification_verdict == "refuted":
                    contradictions.append(fact)
                elif (
                    evidence_status != "confirmed"
                    or typed_fact.get("value") is None
                    or str(typed_fact.get("freshness_status") or "").lower() == "future"
                ):
                    raw_gaps.append(
                        {
                            "source_agent": source_agent,
                            "metric": typed_fact.get("metric") or f"{source_agent} fact",
                            "reason": "Secondary evidence remained unconfirmed.",
                            "impact": typed_fact.get("statement"),
                            "blocking": False,
                        }
                    )
                else:
                    confirmed_facts.append(fact)
        else:
            for finding in payload.get("findings", []):
                fact = _finding_fact(source_agent, finding)
                evidence_status = str(finding.get("evidence_status") or "").lower()
                verification_verdict = str(
                    finding.get("verification_verdict") or ""
                ).lower()
                if (
                    evidence_status == "refuted"
                    or verification_verdict == "refuted"
                    or (
                        not evidence_status
                        and finding.get("driver_class") == "not_a_driver"
                    )
                ):
                    contradictions.append(fact)
                elif evidence_status == "unconfirmed":
                    raw_gaps.append(
                        {
                            "source_agent": source_agent,
                            "metric": finding.get("factor") or f"{source_agent} fact",
                            "reason": "Secondary evidence remained unconfirmed.",
                            "impact": finding.get("impact"),
                            "blocking": False,
                        }
                    )
                else:
                    confirmed_facts.append(fact)

        for gap in payload.get("data_gaps", []):
            if has_business_evidence:
                raw_gaps.append(_gap_entry(source_agent, gap, blocking=False))
            else:
                raw_gaps.append(_gap_entry(source_agent, gap, blocking=True))

        supporting_evidence.append(
            {
                "round": evidence.get("round"),
                "target_agent": source_agent,
                "reason": evidence.get("reason"),
                "question": evidence.get("question"),
                "summary": payload.get("summary"),
                "tables": [] if typed_non_evidence else _trim_list(payload.get("tables"), 2),
                "findings": [] if typed_non_evidence else _trim_list(payload.get("findings"), 4),
                "has_runtime_failure": bool(evidence.get("runtime_failure")),
            }
        )

    blocking_gaps, non_blocking_gaps, resolved_gaps = _resolve_cross_domain_gaps(
        raw_gaps,
        evidence_by_agent=evidence_by_agent,
    )

    confirmed_facts = _select_confirmed_facts_for_query(
        confirmed_facts,
        user_input=user_input,
        primary_agent=primary_agent,
        limit=12,
    )
    return {
        "primary_evidence": _trim_payload_for_synthesis(primary_payload),
        "supporting_evidence": supporting_evidence,
        "confirmed_facts": confirmed_facts,
        "contradictions": contradictions[:5],
        "blocking_gaps": blocking_gaps,
        "non_blocking_gaps": non_blocking_gaps,
        "resolved_gaps": resolved_gaps,
        "runtime_failures": runtime_failures,
    }


def _select_confirmed_facts_for_query(
    facts: list[dict[str, Any]],
    *,
    user_input: str,
    primary_agent: QueryIntent,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep query-critical secondary evidence ahead of generic primary facts."""

    if not facts or limit <= 0:
        return []
    query = _normalize_text(user_input)
    concept_groups = (
        (
            ("оборач", "turnover", "дни запас"),
            ("turnover", "days_on_hand", "avg_stock", "оборач", "дни запас"),
            "supply",
        ),
        (
            (
                "деньги в текущих остат",
                "деньги в остат",
                "стоимост остат",
                "inventory value",
                "stock value",
            ),
            (
                "current_inventory_value",
                "current_stock_value",
                "available_stock",
                "unit_cost",
                "стоимост текущ",
                "деньги в остат",
            ),
            "supply",
        ),
        (
            ("прибыл", "марж", "profit", "margin"),
            ("contribution_margin", "net_profit", "profit", "марж", "прибыл"),
            "finance",
        ),
        (
            ("продаж", "выруч", "sales", "revenue"),
            (
                "sales",
                "revenue",
                "units",
                "orders",
                "order_",
                "продаж",
                "выруч",
                "заказ",
                "спрос",
                "demand",
            ),
            "sales",
        ),
        (
            ("реклам", "дрр", "marketing", "advert"),
            ("marketing", "advert", "дрр", "spend", "click", "order"),
            "marketing",
        ),
        (
            ("цен", "price", "конкурент"),
            ("price", "цен", "competitor", "конкурент"),
            "price",
        ),
        (
            ("остат", "склад", "stock", "inventory"),
            ("stock", "inventory", "остат", "склад"),
            "supply",
        ),
    )
    query_tokens = {
        token
        for token in re.findall(r"[a-zа-я0-9_]+", query)
        if len(token) >= 5
    }

    def fact_text(fact: dict[str, Any]) -> str:
        return _normalize_text(
            " ".join(
                str(fact.get(key) or "")
                for key in (
                    "metric",
                    "statement",
                    "metric_definition",
                    "source_agent",
                    "source_table",
                )
            )
        )

    def semantic_fact_text(fact: dict[str, Any]) -> str:
        """Fact meaning without model-authored provenance labels."""

        return _normalize_text(
            " ".join(
                str(fact.get(key) or "")
                for key in ("metric", "statement", "metric_definition")
            )
        )

    requested_groups = [
        group
        for group in concept_groups
        if any(marker in query for marker in group[0])
    ]

    def is_genuine_requested_fact(fact: dict[str, Any]) -> bool:
        if not requested_groups:
            return True
        source_agent = _normalize_text(fact.get("source_agent"))
        semantic_text = semantic_fact_text(fact)
        matching_groups = [
            group
            for group in requested_groups
            if any(marker in semantic_text for marker in group[1])
        ]
        if matching_groups:
            # A supply-authored `sales_units` (or analogous mismatch) cannot
            # occupy the sales slot merely because its metric name says sales.
            return any(group[2] == source_agent for group in matching_groups)
        # Genuine evidence from another domain may be needed by downstream
        # consistency checks (for example positive sales proving that an empty
        # finance aggregate is no_data).  It cannot fill any requested concept
        # slot because it did not semantically match one.
        return True

    def score(index_and_fact: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, fact = index_and_fact
        text = fact_text(fact)
        relevance = 0
        source_agent = _normalize_text(fact.get("source_agent"))
        for query_markers, fact_markers, preferred_source in concept_groups:
            if any(marker in query for marker in query_markers) and any(
                marker in text for marker in fact_markers
            ) and source_agent == preferred_source:
                relevance += 100
        relevance += min(20, 2 * sum(token in text for token in query_tokens))
        if relevance > 0 and str(fact.get("source_agent") or "") != primary_agent.value:
            relevance += 15
        if str(fact.get("freshness_status") or "").lower() == "stale":
            relevance -= 5
        return (-relevance, index)

    ranked = sorted(
        (
            (index, fact)
            for index, fact in enumerate(facts)
            if is_genuine_requested_fact(fact)
        ),
        key=score,
    )
    selected_indexes: list[int] = []
    selected_set: set[int] = set()

    # Reserve one slot for every concept explicitly requested by the user.
    # This prevents a large SKU matrix from crowding price or marketing out of
    # the compact ledger before synthesis/fallback gets a chance to see it.
    for query_markers, fact_markers, preferred_source in concept_groups:
        if not any(marker in query for marker in query_markers):
            continue
        for index, fact in ranked:
            if index in selected_set:
                continue
            if _normalize_text(fact.get("source_agent")) != preferred_source:
                continue
            text = fact_text(fact)
            if not any(marker in text for marker in fact_markers):
                continue
            selected_indexes.append(index)
            selected_set.add(index)
            break
        if len(selected_indexes) >= limit:
            break

    # Keep the existing relevance order among the reserved representatives so
    # diversity does not reshuffle established high-priority metrics.
    selected_indexes = [
        index for index, _ in ranked if index in selected_set
    ]
    for index, _ in ranked:
        if len(selected_indexes) >= limit:
            break
        if index in selected_set:
            continue
        selected_indexes.append(index)
        selected_set.add(index)
    return [facts[index] for index in selected_indexes]


def _metric_key(value: Any) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "_", _normalize_text(value)).strip("_")


def _supply_metric_dimension(value: Any) -> str | None:
    """Map only an explicit metric label, never surrounding prose, to a dimension."""

    payload = value if isinstance(value, dict) else {}
    label = (
        payload.get("metric")
        or payload.get("factor")
        or payload.get("column")
        or (value if not isinstance(value, dict) else "")
    )
    key = _metric_key(label)
    if not key:
        return None
    if key in {
        "current_stock_value_rub",
        "current_stock_value_confirmed",
        "current_inventory_value_total",
        "current_inventory_value_rub",
        "confirmed_stock_value_rub",
        "confirmed_current_inventory_value",
        "current_stock_cost_rub",
        "inventory_value_rub",
        "stock_value_rub",
        "стоимость_текущих_остатков",
        "стоимость_текущего_остатка",
        "стоимость_остатка",
        "деньги_в_текущих_остатках",
        "подтвержденная_стоимость_текущих_остатков",
    }:
        return "inventory_value"
    if not any(marker in key for marker in ("margin", "марж", "profit", "прибыл")) and (
        any(marker in key for marker in ("stock", "inventory", "остат"))
        and any(marker in key for marker in ("value", "cost", "стоимост", "деньг"))
    ):
        return "inventory_value"
    if key in {
        "current_stock_qty",
        "current_stock_units",
        "current_stock_units_total",
        "current_stock",
        "available_stock",
        "sellable_stock",
        "on_hand_qty",
        "stock_qty",
        "текущий_остаток",
        "текущие_остатки",
        "доступный_остаток",
    }:
        return "stock_quantity"
    if key in {
        "valued_sku_count",
        "current_stock_sku_count",
        "sku_count_in_model",
        "sku_count",
        "skus_with_cost",
        "sku_в_срезе",
        "sku_в_расчете",
    }:
        return "valued_sku_count"
    if key in {
        "inventory_turnover_ratio",
        "turnover_ratio",
        "inventory_turnover",
        "коэффициент_оборачиваемости",
        "оборачиваемость_склада",
    }:
        return "turnover_ratio"
    if key in {
        "days_on_hand",
        "avg_turnover_days",
        "turnover_days",
        "stock_coverage_days",
        "days_of_stock",
        "средняя_оборачиваемость",
        "дни_запаса",
        "покрытие_запасом",
    }:
        return "days_on_hand"
    return None


def _evidence_scope(value: Any) -> str:
    payload = value if isinstance(value, dict) else {}
    sku = str(payload.get("sku") or payload.get("vendor_code") or "").strip()
    if sku:
        return sku
    target_type = _normalize_text(payload.get("target_type"))
    target_id = str(payload.get("target_id") or "").strip()
    if target_type == "sku" and target_id:
        return target_id.split(":", 1)[-1].strip() or "*"
    return "*"


def _supply_authority_scopes(facts: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Collect SKU scopes proven by the deterministic supply evidence layer."""

    scopes: dict[str, set[str]] = {}
    for fact in facts:
        if _normalize_text(str(fact.get("source_agent") or "")) != "supply":
            continue
        if str(fact.get("status") or "").strip().lower() != "confirmed":
            continue
        if fact.get("value") is None:
            continue
        if str(fact.get("freshness_status") or "").strip().lower() == "future":
            continue
        dimension = _supply_metric_dimension(fact)
        evidence_ref = _normalize_text(str(fact.get("evidence_ref") or ""))
        if (
            dimension in {"inventory_value", "stock_quantity", "valued_sku_count"}
            and "get_current_inventory_value" in evidence_ref
        ):
            governed_dimensions = {
                "inventory_value",
                "stock_quantity",
                "valued_sku_count",
            }
        elif (
            dimension in {"turnover_ratio", "days_on_hand"}
            and "get_inventory_turnover" in evidence_ref
        ):
            governed_dimensions = {"turnover_ratio", "days_on_hand"}
        else:
            continue
        scope = _evidence_scope(fact)
        for governed_dimension in governed_dimensions:
            scopes.setdefault(governed_dimension, set()).add(scope)
    return scopes


def _is_shadowed_by_supply_authority(
    value: Any,
    *,
    source_agent: str,
    scopes: dict[str, set[str]],
) -> bool:
    if _normalize_text(source_agent) == "supply":
        return False
    dimension = _supply_metric_dimension(value)
    if not dimension or dimension not in scopes:
        return False
    return _evidence_scope(value) in scopes[dimension]


def _sanitize_non_supply_table(
    table: Any,
    *,
    scopes: dict[str, set[str]],
) -> dict[str, Any] | None:
    if not isinstance(table, dict):
        return table
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(rows, list):
        return table

    clean = dict(table)
    if not isinstance(columns, list):
        clean_rows: list[Any] = []
        for row in rows:
            if not isinstance(row, dict):
                clean_rows.append(row)
                continue
            row_scope = _evidence_scope(row)
            clean_rows.append(
                {
                    key: cell
                    for key, cell in row.items()
                    if not (
                        (dimension := _supply_metric_dimension(key))
                        and dimension in scopes
                        and ("*" in scopes[dimension] or row_scope in scopes[dimension])
                    )
                }
            )
        clean["rows"] = clean_rows
        return clean if clean_rows else None

    sku_column_indexes = [
        index
        for index, column in enumerate(columns)
        if _metric_key(column) in {"sku", "vendor_code", "артикул", "скю"}
    ]

    def row_scope(row: Any) -> str:
        if isinstance(row, dict):
            return _evidence_scope(row)
        if isinstance(row, list) and sku_column_indexes:
            sku_index = sku_column_indexes[0]
            if sku_index < len(row) and str(row[sku_index] or "").strip():
                return str(row[sku_index]).strip()
        return "*"

    def authority_applies(dimension: str | None, row: Any) -> bool:
        if not dimension or dimension not in scopes:
            return False
        scope = row_scope(row)
        return "*" in scopes[dimension] or scope in scopes[dimension]

    metric_column_indexes = [
        index
        for index, column in enumerate(columns)
        if _metric_key(column) in {"metric", "метрика", "показатель", "factor", "фактор"}
    ]
    if metric_column_indexes:
        metric_index = metric_column_indexes[0]
        clean["rows"] = [
            row
            for row in rows
            if not (
                isinstance(row, list)
                and metric_index < len(row)
                and authority_applies(_supply_metric_dimension(row[metric_index]), row)
            )
        ]
        return clean if clean["rows"] else None

    globally_authoritative_indexes = {
        index
        for index, column in enumerate(columns)
        if (
            (dimension := _supply_metric_dimension(column))
            and dimension in scopes
            and "*" in scopes[dimension]
        )
    }
    scoped_indexes = {
        index: dimension
        for index, column in enumerate(columns)
        if (
            (dimension := _supply_metric_dimension(column))
            and dimension in scopes
            and "*" not in scopes[dimension]
        )
    }
    if not globally_authoritative_indexes and not scoped_indexes:
        return clean
    clean["columns"] = [
        column
        for index, column in enumerate(columns)
        if index not in globally_authoritative_indexes
    ]
    clean_rows = []
    for row in rows:
        if not isinstance(row, list):
            clean_rows.append(row)
            continue
        scope = row_scope(row)
        clean_rows.append(
            [
                (
                    None
                    if index in scoped_indexes and scope in scopes[scoped_indexes[index]]
                    else cell
                )
                for index, cell in enumerate(row)
                if index not in globally_authoritative_indexes
            ]
        )
    clean["rows"] = clean_rows
    return clean if clean["columns"] and clean["rows"] else None


def _sanitize_non_supply_tables(
    tables: Any,
    *,
    scopes: dict[str, set[str]],
) -> list[Any]:
    cleaned: list[Any] = []
    for table in tables or []:
        value = _sanitize_non_supply_table(
            table,
            scopes=scopes,
        )
        if value is not None:
            cleaned.append(value)
    return cleaned


def _apply_supply_domain_authority(package: dict[str, Any]) -> None:
    """Remove conflicting non-supply stock evidence before final synthesis.

    Financial report rows describe sales and payouts, not an authoritative
    sellable-stock snapshot.  Once the dedicated supply calculators produced a
    confirmed fact, their stock value/quantity and turnover evidence wins over
    similarly named facts inferred by another domain.
    """

    confirmed_facts = [
        fact
        for fact in package.get("confirmed_facts") or []
        if isinstance(fact, dict)
        and not (
            fact.get("metric")
            and (
                fact.get("value") is None
                or str(fact.get("status") or "confirmed").lower() != "confirmed"
                or str(fact.get("freshness_status") or "").lower() == "future"
            )
        )
    ]
    package["confirmed_facts"] = confirmed_facts
    scopes = _supply_authority_scopes(confirmed_facts)
    if not scopes:
        return

    package["confirmed_facts"] = [
        fact
        for fact in confirmed_facts
        if not _is_shadowed_by_supply_authority(
            fact,
            source_agent=str(fact.get("source_agent") or ""),
            scopes=scopes,
        )
    ]

    primary_agent = str(package.get("primary_agent") or "")
    primary_evidence = package.get("primary_evidence")
    if isinstance(primary_evidence, dict) and _normalize_text(primary_agent) != "supply":
        for key in ("facts", "findings"):
            primary_evidence[key] = [
                item
                for item in primary_evidence.get(key) or []
                if not _is_shadowed_by_supply_authority(
                    item,
                    source_agent=primary_agent,
                    scopes=scopes,
                )
            ]
        primary_evidence["tables"] = _sanitize_non_supply_tables(
            primary_evidence.get("tables"),
            scopes=scopes,
        )
        for snapshot_key in ("sales_report_snapshot", "marketing_report_snapshot"):
            if isinstance(primary_evidence.get(snapshot_key), dict):
                primary_evidence[snapshot_key] = _sanitize_non_supply_table(
                    primary_evidence[snapshot_key],
                    scopes=scopes,
                )

    for evidence in package.get("supporting_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        source_agent = str(evidence.get("target_agent") or "")
        if _normalize_text(source_agent) == "supply":
            continue
        evidence["findings"] = [
            item
            for item in evidence.get("findings") or []
            if not _is_shadowed_by_supply_authority(
                item,
                source_agent=source_agent,
                scopes=scopes,
            )
        ]
        evidence["tables"] = _sanitize_non_supply_tables(
            evidence.get("tables"),
            scopes=scopes,
        )

    package["contradictions"] = [
        item
        for item in package.get("contradictions") or []
        if not _is_shadowed_by_supply_authority(
            item,
            source_agent=str(item.get("source_agent") or "")
            if isinstance(item, dict)
            else "",
            scopes=scopes,
        )
    ]
    package["sku_metric_matrix"] = _sanitize_non_supply_tables(
        package.get("sku_metric_matrix"),
        scopes=scopes,
    )


def _is_positive_sales_evidence(fact: Any) -> bool:
    if not isinstance(fact, dict):
        return False
    if _normalize_text(fact.get("source_agent")) not in {"sales", "supply"}:
        return False
    value = _as_float(fact.get("value"))
    if value is None or value <= 0:
        return False
    key = _metric_key(fact.get("metric"))
    unit = _metric_key(fact.get("unit"))
    return (
        "sales" in key
        and unit in {"unit", "units", "шт", "штук", "pcs"}
    )


def _is_empty_finance_marker(fact: Any) -> bool:
    if not isinstance(fact, dict):
        return False
    return bool(
        _normalize_text(fact.get("source_agent")) == "finance"
        and _as_float(fact.get("value")) == 0
        and "wb_report_detail_daily" in _normalize_text(fact.get("source_table"))
        and _metric_key(fact.get("metric"))
        in {"skus_with_sales_90d", "sku_with_sales", "sold_sku_count"}
    )


def _is_zero_finance_aggregate(fact: Any) -> bool:
    if not isinstance(fact, dict):
        return False
    if _normalize_text(fact.get("source_agent")) != "finance":
        return False
    if _as_float(fact.get("value")) != 0:
        return False
    if "wb_report_detail_daily" not in _normalize_text(fact.get("source_table")):
        return False
    key = _metric_key(fact.get("metric"))
    return any(
        marker in key
        for marker in (
            "revenue",
            "commission",
            "logistics",
            "storage",
            "contribution_margin",
            "sales",
            "payout",
            "acquiring",
        )
    )


def _is_empty_finance_metric_label(value: Any) -> bool:
    key = _metric_key(value)
    return any(
        marker in key
        for marker in (
            "revenue",
            "выруч",
            "commission",
            "комисс",
            "logistics",
            "логист",
            "storage",
            "хранен",
            "contribution_margin",
            "sku_with_sales",
            "skus_with_sales",
        )
    )


def _empty_finance_row(row: Any, columns: list[Any]) -> bool:
    if not isinstance(row, list):
        return False
    metric_indexes = [
        index
        for index, column in enumerate(columns)
        if _metric_key(column) in {"metric", "метрика", "показатель", "factor", "фактор"}
    ]
    value_indexes = [
        index
        for index, column in enumerate(columns)
        if _metric_key(column) in {"value", "значение"}
    ]
    if not metric_indexes or not value_indexes:
        return False
    metric_index = metric_indexes[0]
    value_index = value_indexes[0]
    if metric_index >= len(row) or value_index >= len(row):
        return False
    return bool(
        _as_float(row[value_index]) == 0
        and _is_empty_finance_metric_label(row[metric_index])
    )


def _sanitize_empty_finance_table(table: Any) -> dict[str, Any] | None:
    if not isinstance(table, dict):
        return table
    rows = table.get("rows")
    if not isinstance(rows, list):
        return table
    columns = table.get("columns")
    clean = dict(table)
    if not isinstance(columns, list):
        clean["rows"] = [
            {
                key: value
                for key, value in row.items()
                if not (_is_empty_finance_metric_label(key) and _as_float(value) == 0)
            }
            if isinstance(row, dict)
            else row
            for row in rows
        ]
        return clean if clean["rows"] else None

    metric_columns = {
        _metric_key(column) for column in columns
    } & {"metric", "метрика", "показатель", "factor", "фактор"}
    if metric_columns:
        clean["rows"] = [row for row in rows if not _empty_finance_row(row, columns)]
        return clean if clean["rows"] else None

    remove_indexes = {
        index
        for index, column in enumerate(columns)
        if _is_empty_finance_metric_label(column)
    }
    if not remove_indexes:
        return clean
    clean["columns"] = [
        column for index, column in enumerate(columns) if index not in remove_indexes
    ]
    clean["rows"] = [
        [value for index, value in enumerate(row) if index not in remove_indexes]
        if isinstance(row, list)
        else row
        for row in rows
    ]
    return clean if clean["columns"] and clean["rows"] else None


def _remove_empty_finance_aggregate(package: dict[str, Any]) -> None:
    """Treat a zero COALESCE aggregate as no_data when sales are proven elsewhere."""

    primary_evidence = package.get("primary_evidence") or {}
    primary_facts = primary_evidence.get("facts") or [] if isinstance(primary_evidence, dict) else []
    primary_agent = str(package.get("primary_agent") or "")

    def tagged_primary_fact(fact: Any) -> Any:
        if not isinstance(fact, dict):
            return fact
        return {**fact, "source_agent": fact.get("source_agent") or primary_agent}

    if not any(_is_empty_finance_marker(tagged_primary_fact(fact)) for fact in primary_facts):
        return
    if not any(
        _is_positive_sales_evidence(fact)
        for fact in package.get("confirmed_facts") or []
    ):
        return

    package["confirmed_facts"] = [
        fact
        for fact in package.get("confirmed_facts") or []
        if not _is_zero_finance_aggregate(fact)
    ]
    if isinstance(primary_evidence, dict):
        primary_evidence["facts"] = [
            fact
            for fact in primary_facts
            if not _is_zero_finance_aggregate(tagged_primary_fact(fact))
        ]
        cleaned_tables: list[Any] = []
        for table in primary_evidence.get("tables") or []:
            clean = _sanitize_empty_finance_table(table)
            if clean is not None:
                cleaned_tables.append(clean)
        primary_evidence["tables"] = cleaned_tables
        package["sku_metric_matrix"] = [
            clean
            for table in package.get("sku_metric_matrix") or []
            if (clean := _sanitize_empty_finance_table(table)) is not None
        ]

    if not any(
        "finance aggregate coverage" in _normalize_text(gap.get("metric"))
        for gap in package.get("blocking_gaps") or []
        if isinstance(gap, dict)
    ):
        package.setdefault("blocking_gaps", []).append(
            {
                "source_agent": "finance",
                "metric": "finance aggregate coverage",
                "reason": (
                    "Finance-агрегат вернул пустое покрытие при подтверждённых продажах; "
                    "его нули трактуются как no_data, а не как нулевая выручка или маржа."
                ),
                "impact": "Contribution margin и net profit по этому срезу не подтверждены.",
                "blocking": True,
                "status": "accepted_limitation",
            }
        )


def _render_prompt_v2_facts_only_answer(
    user_query: str,
    synthesis_package: dict[str, Any],
) -> str:
    """Render a safe, evidence-complete fallback when synthesis repair fails."""

    def parsed_numeric_fact_value(value: Any) -> int | float | None:
        """Accept numbers, not arbitrary model text that happens to avoid a blacklist."""

        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if not isinstance(value, str):
            return None
        candidate = (
            value.strip()
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
        )
        if not re.fullmatch(
            r"[+-]?(?:\d{1,3}(?:[ _]\d{3})+|\d+)(?:[.,]\d+)?",
            candidate,
        ):
            return None
        normalized = candidate.replace(" ", "").replace("_", "").replace(",", ".")
        if "." not in normalized:
            try:
                return int(normalized)
            except ValueError:
                return None
        try:
            parsed = float(normalized)
        except ValueError:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    facts: list[dict[str, Any]] = []
    for fact in synthesis_package.get("confirmed_facts") or []:
        if not isinstance(fact, dict) or not fact.get("metric"):
            continue
        numeric_value = parsed_numeric_fact_value(fact.get("value"))
        if numeric_value is None:
            continue
        if str(fact.get("status") or "confirmed").lower() != "confirmed":
            continue
        if str(fact.get("freshness_status") or "").lower() == "future":
            continue
        renderable_fact = dict(fact)
        renderable_fact["_numeric_value"] = numeric_value
        facts.append(renderable_fact)
    query_text = _normalize_text(user_query)

    def relevant_to_query(fact: dict[str, Any]) -> bool:
        fact_text = _normalize_text(
            " ".join(
                str(fact.get(key) or "")
                for key in ("metric", "metric_definition", "statement")
            )
        )
        groups = (
            (
                ("оборач", "turnover", "дни запас"),
                ("turnover", "days_on_hand", "оборач", "дни запас"),
            ),
            (
                ("остат", "stock", "inventory"),
                ("stock", "inventory", "остат", "valued_sku_count"),
            ),
            (
                ("прибыл", "марж", "profit", "margin"),
                ("profit", "margin", "марж", "прибыл"),
            ),
            (
                ("продаж", "выруч", "sales", "revenue"),
                ("sales", "revenue", "продаж", "выруч"),
            ),
            (
                ("реклам", "дрр", "marketing", "advert"),
                (
                    "marketing",
                    "advert",
                    "реклам",
                    "дрр",
                    "spend",
                    "click",
                    "ctr",
                    "cpc",
                ),
            ),
            (
                ("цен", "price", "конкурент"),
                ("price", "цен", "competitor", "конкурент", "discount", "promo"),
            ),
        )
        return any(
            any(marker in query_text for marker in query_markers)
            and any(marker in fact_text for marker in fact_markers)
            for query_markers, fact_markers in groups
        )

    relevant_facts = [fact for fact in facts if relevant_to_query(fact)]
    if relevant_facts:
        facts = relevant_facts
    facts = _select_confirmed_facts_for_query(
        facts,
        user_input=user_query,
        primary_agent=QueryIntent.SALES,
        limit=10,
    )
    labels = {
        "current_inventory_value_total": "Стоимость текущих sellable-остатков",
        "current_stock_value_rub": "Стоимость текущих sellable-остатков",
        "confirmed_stock_value_rub": "Стоимость текущих sellable-остатков",
        "confirmed_current_inventory_value": "Стоимость текущих sellable-остатков",
        "turnover_ratio": "Оборачиваемость склада",
        "inventory_turnover_ratio": "Оборачиваемость склада",
        "days_on_hand": "Покрытие запасом",
        "days_of_stock": "Покрытие запасом",
        "current_stock_qty": "Текущий доступный остаток",
        "available_stock_qty": "Текущий доступный остаток",
        "valued_sku_count": "SKU с подтверждённой оценкой остатка",
        "contribution_margin_total": "Contribution margin (не чистая прибыль)",
        "sales_units": "Продажи",
        "sales_units_delta": "Изменение продаж",
        "latest price decision snapshot date": "Дата доступного ценового снимка",
        "price decision snapshot count for requested nm_ids": (
            "SKU с доступным ценовым снимком"
        ),
    }
    lines = [
        "# Подтверждённые данные",
        "",
        "Ниже только факты из подтверждённых источников с их периодами.",
    ]
    causal_query = any(
        marker in query_text
        for marker in ("почему", "причин", "влияни", "драйвер", "из-за")
    )
    if causal_query:
        lines.extend(
            [
                "",
                (
                    "Сами по себе эти факты не доказывают причинную связь; "
                    "неподтверждённые объяснения вынесены отдельно."
                ),
            ]
        )
    if not facts:
        lines.extend(["", "Подтверждённых числовых фактов для ответа недостаточно."])
    else:
        lines.extend(["", "## Факты"])
        for fact in facts:
            metric = str(fact.get("metric") or "metric")
            label = labels.get(metric, metric.replace("_", " "))
            value = fact.get("_numeric_value")
            if isinstance(value, float):
                rendered_value = f"{value:,.4f}".replace(",", " ").rstrip("0").rstrip(".")
            else:
                rendered_value = str(value)
            unit = str(fact.get("unit") or "").strip()
            if unit.lower() == "date":
                unit = ""
            elif unit.lower() == "rows":
                unit = "SKU"
            period = str(fact.get("period") or "").strip()
            sku = str(fact.get("sku") or "").strip()
            scope = f"SKU {sku}: " if sku else ""
            suffix = f" {unit}" if unit else ""
            if period:
                suffix += f" за {period}"
            lines.append(f"- {scope}{label}: {rendered_value}{suffix}.")

    confirmed_text = _normalize_text(
        " ".join(
            str(fact.get("metric") or "") + " " + str(fact.get("statement") or "")
            for fact in facts
        )
    )
    gaps: list[dict[str, Any]] = []
    for gap in synthesis_package.get("blocking_gaps") or []:
        if not isinstance(gap, dict):
            continue
        gap_text = _normalize_text(
            " ".join(str(gap.get(key) or "") for key in ("metric", "reason", "impact"))
        )
        stock_history_gap = any(
            marker in gap_text
            for marker in ("истори", "динамик", "дневн", "весь период", "по дат", "ряд ")
        )
        stock_covered = (
            not stock_history_gap
            and any(marker in gap_text for marker in ("остат", "stock", "supply"))
            and any(
                marker in confirmed_text
                for marker in (
                    "current_inventory_value",
                    "current_stock",
                    "confirmed_stock_value",
                    "confirmed_current_inventory",
                    "available_stock",
                    "sellable_stock",
                )
            )
        )
        turnover_covered = "оборач" in gap_text and any(
            marker in confirmed_text for marker in ("turnover", "days_on_hand")
        )
        if stock_covered or turnover_covered:
            continue
        query_relevant_gap = any(
            any(marker in query_text for marker in query_markers)
            and any(value in gap_text for value in gap_markers)
            for query_markers, gap_markers in (
                (("прибыл", "марж", "profit"), ("profit", "прибыл", "марж", "cogs", "p&l", "pl ")),
                (("оборач", "turnover"), ("оборач", "turnover", "stock", "остат", "запас")),
                (("остат", "stock", "inventory"), ("stock", "остат", "inventory", "запас", "supply")),
                (
                    ("продаж", "спрос", "sales", "demand"),
                    (
                        "sales",
                        "продаж",
                        "спрос",
                        "demand",
                        "attribution",
                        "incrementality",
                        "причин",
                    ),
                ),
                (
                    ("реклам", "дрр", "marketing", "advert"),
                    (
                        "marketing",
                        "advert",
                        "реклам",
                        "дрр",
                        "spend",
                        "click",
                        "ctr",
                        "cpc",
                        "attribution",
                        "incrementality",
                    ),
                ),
                (
                    ("цен", "price", "конкурент"),
                    ("price", "цен", "discount", "promo", "скид", "конкурент"),
                ),
            )
        )
        if not query_relevant_gap:
            continue
        gaps.append(gap)
    if gaps:
        lines.extend(
            [
                "",
                (
                    "## Неподтверждённые гипотезы и ограничения"
                    if causal_query
                    else "## Ограничения"
                ),
            ]
        )
        seen: set[str] = set()
        for gap in gaps[:8]:
            reason = str(gap.get("reason") or gap.get("metric") or "").strip()
            reason_key = _normalize_text(reason)
            if reason and reason_key not in seen:
                seen.add(reason_key)
                lines.append(f"- {reason}")
    return "\n".join(lines).strip()


def _table_has_price_scenario_columns(table: Any) -> bool:
    if not isinstance(table, dict):
        return False
    columns_text = _normalize_text(" ".join(str(column) for column in table.get("columns") or []))
    if not columns_text:
        return False
    has_price_decision = _text_contains_any(
        columns_text,
        (
            "recommended_price",
            "рекомендован",
            "рекомендац",
            "стартов",
            "тестов",
            "сценар",
            "новая цена",
            "целевая цена",
        ),
    )
    has_guardrail = _text_contains_any(
        columns_text,
        (
            "expected_orders",
            "expected_net_profit",
            "expected_margin",
            "ожидаем",
            "марж",
            "confidence",
            "уверенность",
            "decision_status",
            "статус",
        ),
    )
    return has_price_decision and has_guardrail


def _payload_has_price_scenario_decision(payload: dict[str, Any]) -> bool:
    tool_report = payload.get("_tool_report") or {}
    tool_names = [str(name) for name in tool_report.get("tool_names") or []]
    if PRICE_DECISION_TOOL in tool_names:
        return True
    for table in payload.get("tables") or []:
        if _table_has_price_scenario_columns(table):
            return True
    text = _normalize_text(
        " ".join(
            [
                str(payload.get("summary") or ""),
                " ".join(str(item) for item in payload.get("recommendations") or []),
            ]
        )
    )
    return _text_contains_any(text, ("recommended_price", "рекомендованная цена", "стартовая цена"))


def _is_exact_price_model_gap(gap: dict[str, Any]) -> bool:
    text = _normalize_text(
        " ".join(
            str(gap.get(key) or "")
            for key in ("metric", "reason", "impact")
        )
    )
    return _text_contains_any(
        text,
        (
            "эластич",
            "elastic",
            "uplift",
            "точный прогноз",
            "прогноз продаж",
            "ценовым сценар",
            "минимально безопас",
            "нижняя цена",
            "safe floor",
            "sku-level floor",
            "целевой марж",
            "price-volume",
            "цена продажи",
            "цена–продажи",
            "цена-продажи",
        ),
    )


def _price_decision_mode(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    primary_payload: dict[str, Any],
) -> dict[str, Any]:
    if primary_agent != QueryIntent.PRICE or not _is_price_decision_query(user_input):
        return {
            "mode": "evidence_report",
            "scenario_decision_required": False,
            "exact_model_required": False,
        }
    has_scenario = _payload_has_price_scenario_decision(primary_payload)
    return {
        "mode": "scenario_decision" if has_scenario else "decision_needed",
        "scenario_decision_required": True,
        "scenario_decision_available": has_scenario,
        "exact_model_required": False,
        "principle": (
            "Отсутствие точной эластичности или окончательного safe floor не блокирует "
            "стартовое сценарное price decision; это снижает confidence и добавляет guardrails."
        ),
        "required_output": [
            "SKU",
            "текущая цена",
            "стартовая/рекомендованная цена",
            "дельта цены",
            "ожидаемые заказы/маржа, если есть",
            "confidence/status",
            "стоп-условие",
        ],
    }


def _demote_exact_price_model_gaps_for_scenario(
    package: dict[str, Any],
) -> None:
    decision_mode = package.get("decision_mode") or {}
    if decision_mode.get("mode") != "scenario_decision":
        return
    kept_blocking: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for gap in package.get("blocking_gaps") or []:
        if _is_exact_price_model_gap(gap):
            demoted.append(
                {
                    **gap,
                    "blocking": False,
                    "status": "scenario_limitation",
                    "resolution": "does_not_block_starting_price_decision",
                }
            )
        else:
            kept_blocking.append(gap)
    package["blocking_gaps"] = kept_blocking
    if demoted:
        package["non_blocking_gaps"] = [*(package.get("non_blocking_gaps") or []), *demoted]
        package["scenario_limitations"] = demoted


def _build_synthesis_package(
    *,
    user_input: str,
    primary_agent: QueryIntent,
    primary_output: Any,
    round_1_plan: dict[str, Any],
    round_1_evidence: list[dict[str, Any]],
    round_2_plan: dict[str, Any] | None,
    followup_evidence: list[dict[str, Any]],
    action_safety_checks: list[dict[str, Any]] | None = None,
    persisted_blocking_gaps: list[dict[str, Any]] | None = None,
    query_plan: dict[str, Any] | None = None,
    deterministic_sku_metric_matrix: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    primary_payload = _diagnostic_payload(primary_output)
    reduced = _reduce_evidence(
        primary_agent=primary_agent,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
        user_input=user_input,
    )
    decision_mode = _price_decision_mode(
        user_input=user_input,
        primary_agent=primary_agent,
        primary_payload=primary_payload,
    )
    sku_matrix = _build_sku_evidence_matrix(
        user_input=user_input,
        primary_agent=primary_agent,
        primary_payload=primary_payload,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
    )
    requested_analyses = {
        str(value) for value in list((query_plan or {}).get("requested_analyses") or [])
    }
    include_sku_matrix = _should_include_sku_evidence_matrix(user_input)
    include_sku_metric_matrix = bool(
        _requires_sku_metric_matrix(user_input)
        or requested_analyses.intersection(
            {"inventory_matrix", "inventory_turnover", "current_stock", "inventory_value"}
        )
    )
    sku_metric_matrix = (
        [
            *list(deterministic_sku_metric_matrix or []),
            *_extract_sku_metric_matrix(primary_payload),
        ]
        if include_sku_metric_matrix
        else []
    )
    evidence_quality = _evidence_quality_summary(
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
        matrix=sku_matrix,
    )
    evidence_by_agent = _business_evidence_by_agent(
        primary_agent=primary_agent,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
    )
    package = {
        "user_query": user_input,
        "query_plan": dict(query_plan or {}),
        "limits": {
            "round_1_secondary_limit": ROUND_1_SECONDARY_LIMIT,
            "round_2_followup_limit": ROUND_2_FOLLOWUP_LIMIT,
            "max_total_followup_domain_calls": MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
            "max_rounds": MAX_DIAGNOSTIC_ROUNDS,
        },
        "primary_agent": primary_agent.value,
        "decision_mode": decision_mode,
        "primary_evidence": reduced["primary_evidence"],
        "supporting_evidence": reduced["supporting_evidence"],
        "confirmed_facts": reduced["confirmed_facts"],
        "contradictions": reduced["contradictions"],
        "blocking_gaps": reduced["blocking_gaps"],
        "non_blocking_gaps": reduced["non_blocking_gaps"],
        "resolved_gaps": reduced["resolved_gaps"],
        "claim_verification": _claim_verification_status(
            user_input=user_input,
            evidence_by_agent=evidence_by_agent,
        ),
        "sku_metric_matrix": sku_metric_matrix,
        "sku_metric_matrix_required": include_sku_metric_matrix,
        "sku_evidence_matrix": sku_matrix.get("rows", []) if include_sku_matrix else [],
        "domain_coverage": sku_matrix.get("domain_coverage", {}) if include_sku_matrix else {},
        "evidence_quality": evidence_quality if include_sku_matrix else {},
        "action_safety_checks": action_safety_checks or [],
        "manager_action_candidates": _collect_manager_action_candidates(
            primary_payload,
            round_1_evidence,
            followup_evidence,
        ),
        "round_1_plan": round_1_plan,
        "round_2_plan": round_2_plan or {},
        "display_policy": {
            "user_facing_identifier": "sku_only",
            "forbidden_user_facing_terms": ["nm_id"],
            "if_sku_missing": "Do not make nm_id the visible object; state that SKU is not mapped in limitations.",
        },
        "format_policy": {
            "max_confirmed_sentences": 5,
            "max_visible_limitations": 2,
            "avoid_bullet_lists_in_confirmed_section": True,
            "use_numbered_list_only_for_actions": True,
            "action_task_count": 5 if _requires_exact_five_actions(user_input) else None,
            "sku_metric_matrix": include_sku_metric_matrix,
            "sku_cause_action_matrix": include_sku_matrix,
            "show_action_safety_summary": bool(action_safety_checks),
            "treat_volume_as_mechanical_effect": True,
        },
        "final_answer_rules": [
            "Do not expose cross_agent_requests or technical JSON.",
            "Do not use service headings like Marketing Agent or Supply Agent.",
            "Use only blocking_gaps for the user-visible limitations section.",
            "Treat blocking_gaps with status accepted_limitation as source limitations, not unfinished work.",
            "For scenario_decision mode, exact elasticity/no_data/floor gaps reduce confidence but must not replace the price recommendation table.",
            "Do not expose resolved_gaps; they were already closed by supporting evidence.",
            "Use only SKU/artikul as user-facing product identifiers; do not mention nm_id in the final markdown.",
            "If sku_evidence_matrix is present, render the SKU hypothesis matrix from it and preserve its statuses.",
            "Never turn an unconfirmed sku_evidence_matrix cell into no_data; no_data is allowed only when the cell reason says domain_not_called, agent_failed_without_evidence, sku_not_in_source, or metric_absent.",
            "If sku_metric_matrix_required is true, render a SKU metric matrix and include a short conclusion for every SKU row; do not collapse SKU rows into only portfolio totals.",
            "Use action_safety_checks before recommending price, advertising, reorder, or sell-off actions.",
            "Treat volume decline as a mechanical contribution, not a root cause; name the underlying confirmed or unconfirmed hypothesis per SKU.",
            "For SKU sales-and-profit diagnostics, prefer a compact SKU matrix: SKU, sales or revenue delta, margin delta, confirmed cause, unconfirmed hypotheses, safety status, next action.",
            "Do not render confirmed_facts as a long markdown bullet list.",
            "Do not mention runtime failures or tool errors; convert them into short business limitations.",
            "If claim_verification.status is incomplete, do not state root-cause claims for claim_verification.missing_domains.",
            "If decision_mode.mode is scenario_decision, answer the business action first: provide a scenario decision table and guardrails even when exact_model_required is false.",
        ],
    }
    if persisted_blocking_gaps:
        existing_keys = {
            (
                str(gap.get("source_agent") or ""),
                str(gap.get("sku") or ""),
                str(gap.get("metric") or ""),
                str(gap.get("reason") or ""),
            )
            for gap in package["blocking_gaps"]
        }
        resolved_keys = {
            (
                str(gap.get("source_agent") or ""),
                str(gap.get("sku") or ""),
                str(gap.get("metric") or ""),
                str(gap.get("reason") or ""),
            )
            for gap in package["resolved_gaps"]
        }
        for gap in persisted_blocking_gaps:
            if not _should_keep_persisted_blocking_gap(gap):
                continue
            resolved_by = _gap_resolved_by_evidence(gap, evidence_by_agent=evidence_by_agent)
            key = (
                str(gap.get("source_agent") or ""),
                str(gap.get("sku") or ""),
                str(gap.get("metric") or ""),
                str(gap.get("reason") or ""),
            )
            if resolved_by:
                if key not in resolved_keys:
                    package["resolved_gaps"].append(
                        {
                            **gap,
                            "blocking": False,
                            "resolved_by": resolved_by,
                        }
                    )
                    resolved_keys.add(key)
                continue
            if key in existing_keys:
                continue
            existing_keys.add(key)
            package["blocking_gaps"].append(gap)
    _demote_exact_price_model_gaps_for_scenario(package)
    _apply_supply_domain_authority(package)
    _remove_empty_finance_aggregate(package)
    package["accepted_limitations"] = [
        gap
        for gap in package["blocking_gaps"]
        if str(gap.get("status") or "") == "accepted_limitation"
    ]
    for key in (
        "primary_evidence",
        "decision_mode",
        "sku_metric_matrix",
        "sku_metric_matrix_required",
        "supporting_evidence",
        "confirmed_facts",
        "contradictions",
        "blocking_gaps",
        "accepted_limitations",
        "non_blocking_gaps",
        "scenario_limitations",
        "resolved_gaps",
        "sku_evidence_matrix",
        "domain_coverage",
        "evidence_quality",
    ):
        package[key] = _sanitize_user_facing_payload(package.get(key))
    return package


def run_orchestrated_diagnostic_workflow(
    *,
    user_input: str,
    intent: QueryIntent,
    context: EcomRunContext,
    hooks: Any,
    run_id: int | None = None,
) -> OrchestratedDiagnosticRun:
    workflow_version = _workflow_version_for_context(context)
    context.round_type = RoundType.PRIMARY
    context.memory_mode = memory_mode_for_round(RoundType.PRIMARY)
    try:
        context.query_mode = QueryMode(str(context.query_plan.get("query_mode")))
    except (TypeError, ValueError):
        context.query_mode = determine_query_mode(user_input)
    context.add_decision_trace(
        "primary_selected",
        target_agent=intent.value,
        status="selected",
        details={"primary_agent": _agent_tool_name(intent)},
    )
    primary_round = _create_agent_run_round_safely(
        run_id=run_id,
        round_number=0,
        round_type="primary",
        status="running",
    )
    primary_question = _create_agent_run_question_safely(
        run_id=run_id,
        round_record=primary_round,
        source_agent="orchestrator",
        target_agent=intent.value,
        question=user_input,
        reason="Primary agent selected by query router.",
        verification_goal={"original_user_query": user_input, "primary_agent": intent.value},
        priority=100,
        status="running",
    )
    primary_runtime_failure: dict[str, Any] | None = None
    primary_tool_policy = _domain_tool_policy(
        target=intent,
        user_query=user_input,
        round_name="primary",
        question_id=primary_question.id if primary_question else None,
        explicit_required_tools=(
            (
                _v2_required_tools_for_plan(intent, context.query_plan)
                if context.query_plan
                else _v2_required_tools_for_query(intent, user_input)
            )
            if workflow_version != WorkflowVersion.V1
            else ()
        ),
    )
    primary_prompt = user_input
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
            question=user_input,
            original_user_query=user_input,
            verification_goal={"original_user_query": user_input, "primary_agent": intent.value},
            required_checks=[],
            source="router",
            gap_id=None,
            claim_policy=None,
            period=None,
            sku=None,
            vendor_codes=[],
            nm_ids=[],
            allowed_tools=primary_allowed_tools,
            required_tools=list(primary_tool_policy.required_tools),
            query_plan=dict(context.query_plan),
        )
        primary_build = build_round_prompt(primary_task, None, [], workflow_version)
        primary_prompt = primary_build.text
        primary_prompt_report = primary_build.report.model_dump(mode="json")
    try:
        primary_output = _run_agent(
            primary_builder,
            user_input=primary_prompt,
            context=context,
            hooks=hooks,
            max_turns=12,
            tool_policy=primary_tool_policy,
            expected_output_type=(
                _primary_output_type(intent)
                if workflow_version != WorkflowVersion.V1
                else None
            ),
            prompt_report=primary_prompt_report,
        )
    except ToolPolicyViolation as exc:
        primary_output = exc.output
        primary_runtime_failure = {
            "target_agent": intent.value,
            **exc.runtime_failure(),
        }
        context.last_tool_report = exc.tool_report
    except OutputGuardrailTripwireTriggered:
        # The outer chat boundary persists the rendered guardrail output and
        # its violations for backward compatibility. Do not reclassify this
        # intentional SDK signal as a malformed primary response.
        raise
    except Exception as exc:
        # Transport failures remain eligible for the caller's retry policy.
        # Parsing/schema failures are terminal evidence failures: persist the
        # exact prompt report, but never treat malformed model text as evidence.
        if _is_transient_model_error(exc):
            raise
        primary_output = {
            "summary": "",
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [
                {
                    "metric": f"{intent.value} output validation",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "impact": "Primary-проверка не завершилась и не может использоваться как evidence.",
                }
            ],
            "cross_agent_requests": [],
        }
        primary_runtime_failure = {
            "target_agent": intent.value,
            "kind": "runtime",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    context.cross_agent_trace.append(f"primary:{intent.value}")
    primary_payload = _attach_prompt_report(
        _attach_tool_report(_diagnostic_payload(primary_output), context),
        context,
    )
    _persist_agent_run_response_safely(
        run_id=run_id,
        question_record=primary_question,
        agent_name=intent.value,
        payload=primary_payload,
        runtime_failure=primary_runtime_failure,
        blocking_gaps=True,
    )
    _update_agent_run_question_status_safely(
        primary_question,
        "failed" if primary_runtime_failure else "completed",
    )
    _update_agent_run_round_status_safely(
        primary_round,
        "failed" if primary_runtime_failure else "completed",
    )
    if primary_runtime_failure:
        raise AgentRuntimeError("Primary domain agent failed validation or required tool policy.")

    optimized = workflow_version == WorkflowVersion.V2_OPTIMIZED
    query_mode = context.query_mode
    # Eight total model questions include router, primary, synthesis and one
    # possible repair, leaving at most four secondary questions.
    optimized_secondary_budget = OPTIMIZED_SECONDARY_QUESTION_BUDGET
    if not optimized:
        round_1_limit = ROUND_1_SECONDARY_LIMIT
    elif query_mode in {QueryMode.REPORT, QueryMode.LOOKUP}:
        round_1_limit = 0
    elif query_mode == QueryMode.DIAGNOSTIC:
        round_1_limit = 3
    else:
        round_1_limit = 2

    agent_requested_round_1_plan = build_cross_domain_dependency_plan(
        user_query=user_input,
        primary_agent=intent.value,
        primary_summary=_diagnostic_summary(primary_output),
        findings=_diagnostic_findings(primary_output),
        data_gaps=_diagnostic_data_gaps(primary_output),
        cross_agent_requests=_diagnostic_cross_agent_requests(primary_output),
        max_secondary_agents=round_1_limit,
    )
    coverage_policy_plan = _inventory_turnover_coverage_plan(
        user_input=user_input,
        primary_agent=intent,
        existing_items=_plan_items(agent_requested_round_1_plan),
        max_secondary_agents=round_1_limit,
        query_plan=context.query_plan,
    )
    round_1_plan = (
        _merge_dependency_plans(
            primary_agent=intent,
            plans=[coverage_policy_plan, agent_requested_round_1_plan],
            max_secondary_agents=round_1_limit,
        )
        or {
            "primary_agent": intent.value,
            "recommended_secondary_agents": [],
            "skipped_agents": [],
            "max_secondary_agents": ROUND_1_SECONDARY_LIMIT,
        }
    )
    context.add_decision_trace(
        "dependency_plan_built",
        round="round_1",
        target_agent=intent.value,
        status="completed",
        details={
            "recommended_count": len(_plan_items(round_1_plan)),
            "skipped_count": len(round_1_plan.get("skipped_agents") or []),
            "max_secondary_agents": round_1_plan.get("max_secondary_agents"),
        },
    )

    already_asked: list[dict[str, str]] = []
    round_1_evidence: list[dict[str, Any]] = []
    round_1_items = _plan_items(round_1_plan)[:round_1_limit]
    round_1_round = _create_agent_run_round_safely(
        run_id=run_id,
        round_number=1,
        round_type="cross_check",
        status="running" if round_1_items else "skipped",
    )
    round_1_questions = [
        _create_agent_run_question_safely(
            run_id=run_id,
            round_record=round_1_round,
            source_agent=intent.value,
            target_agent=_intent_from_agent_name(item.get("target_agent")).value,
            question=str(item.get("question") or ""),
            reason=str(item.get("reason") or ""),
            verification_goal=_verification_goal_from_item(item, round_name="round_1"),
            priority=int(item.get("priority") or 0),
            status="pending",
        )
        for item in round_1_items
    ]
    round_1_evidence = _run_round_1_evidence_batch(
        items=round_1_items,
        questions=round_1_questions,
        user_input=user_input,
        primary_output=primary_output,
        context=context,
        run_id=run_id,
    )
    for evidence in round_1_evidence:
        already_asked.append(_asked_item(evidence["target_agent"], evidence["question"]))
    _update_agent_run_round_status_safely(
        round_1_round,
        "completed" if round_1_items else "skipped",
    )

    followup_evidence: list[dict[str, Any]] = []
    round_2_plan: dict[str, Any] | None = None
    round_1_followup_requests: list[dict[str, Any]] = []
    for evidence in round_1_evidence:
        output = evidence["output"]
        round_1_followup_requests.extend(output.get("cross_agent_requests", []))

    action_safety_plan, action_safety_checks = _build_action_safety_plan(
        user_input=user_input,
        primary_agent=intent,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
        max_secondary_agents=ROUND_2_FOLLOWUP_LIMIT,
    )
    allow_round_2 = not optimized or query_mode == QueryMode.DECISION
    if allow_round_2 and (round_1_followup_requests or _plan_items(action_safety_plan or {})):
        remaining_calls = MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS - len(round_1_evidence)
        if optimized:
            remaining_calls = min(
                remaining_calls,
                max(0, optimized_secondary_budget - len(round_1_evidence)),
            )
        round_2_cap = 2 if optimized else ROUND_2_FOLLOWUP_LIMIT
        round_2_limit = max(0, min(round_2_cap, remaining_calls))
        cross_request_plan = None
        if round_1_followup_requests:
            cross_request_plan = build_cross_domain_dependency_plan(
                user_query="",
                primary_agent=intent.value,
                primary_summary="",
                findings=[],
                data_gaps=[],
                cross_agent_requests=round_1_followup_requests,
                already_asked=already_asked,
                max_secondary_agents=round_2_limit,
            )
        round_2_plan = _merge_dependency_plans(
            primary_agent=intent,
            plans=[cross_request_plan, action_safety_plan],
            max_secondary_agents=round_2_limit,
        )
        context.add_decision_trace(
            "round_2_plan_built",
            round="round_2",
            target_agent=intent.value,
            status="completed",
            details={
                "recommended_count": len(_plan_items(round_2_plan or {})),
                "max_secondary_agents": round_2_limit,
                "has_cross_request_plan": bool(cross_request_plan),
                "has_action_safety_plan": bool(action_safety_plan),
            },
        )
        round_2_items = _plan_items(round_2_plan or {})[:round_2_limit]
        round_2_round = _create_agent_run_round_safely(
            run_id=run_id,
            round_number=2,
            round_type="followup",
            status="running" if round_2_items else "skipped",
        )
        round_2_questions = [
            _create_agent_run_question_safely(
                run_id=run_id,
                round_record=round_2_round,
                source_agent=intent.value,
                target_agent=_intent_from_agent_name(item.get("target_agent")).value,
                question=str(item.get("question") or ""),
                reason=str(item.get("reason") or ""),
                verification_goal=_verification_goal_from_item(item, round_name="round_2"),
                priority=int(item.get("priority") or 0),
                status="pending",
            )
            for item in round_2_items
        ]
        for item, question_record in zip(round_2_items, round_2_questions):
            evidence = _run_domain_evidence(
                item=item,
                user_input=user_input,
                primary_output=primary_output,
                prior_evidence=round_1_evidence,
                context=context,
                hooks=hooks,
                round_name="round_2",
                allow_failure=True,
                run_id=run_id,
                question_record=question_record,
            )
            already_asked.append(_asked_item(evidence["target_agent"], evidence["question"]))
            followup_evidence.append(evidence)
        _update_agent_run_round_status_safely(
            round_2_round,
            "completed" if round_2_items else "skipped",
        )

    action_safety_checks = _finalize_action_safety_checks(action_safety_checks, followup_evidence)
    pre_gap_matrix = _build_sku_evidence_matrix(
        user_input=user_input,
        primary_agent=intent,
        primary_payload=primary_payload,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
    )
    _persist_gap_entries_safely(
        run_id=run_id,
        response_id=None,
        gaps=_gap_entries_from_sku_matrix(pre_gap_matrix),
    )
    round_3_evidence: list[dict[str, Any]] = []
    open_gap_records: list[AgentRunGapRecord] = []
    if run_id is not None:
        try:
            open_gap_records = list_agent_run_gaps(
                run_id=run_id,
                status="open",
                blocking=True,
                evidence_statuses=("no_data", "unconfirmed"),
            )
        except Exception as exc:
            logger.warning("agent_round3_gap_load_failed run_id=%s error=%s", run_id, exc)
            open_gap_records = []
    gap_resolution_items: list[dict[str, Any]] = []
    actionable_open_gaps = [
        gap for gap in open_gap_records if not _is_non_actionable_sku_source_gap(gap)
    ]
    if optimized and query_mode in {QueryMode.REPORT, QueryMode.LOOKUP}:
        actionable_open_gaps = []
    if optimized:
        actionable_open_gaps = _deduplicate_optimized_gap_records(actionable_open_gaps)
        remaining_secondary_budget = max(
            0,
            optimized_secondary_budget - len(round_1_evidence) - len(followup_evidence),
        )
        round_3_limit = min(2, remaining_secondary_budget)
    else:
        round_3_limit = ROUND_3_GAP_RESOLUTION_LIMIT
    for index, gap in enumerate(actionable_open_gaps[:round_3_limit], start=1):
        target_agent = _intent_from_agent_name(gap.target_agent or gap.source_agent).value
        sku_text = f" по SKU {gap.sku}" if gap.sku else ""
        question = (
            f"Закрой blocking gap{sku_text}: {gap.metric}. "
            f"Причина пробела: {gap.reason}. Подтверди факт или явно оставь no_data."
        )
        gap_resolution_items.append(
            {
                "target_agent": target_agent,
                "reason": gap.reason,
                "question": question,
                "priority": max(0, 90 - index),
                "source": "gap_resolution",
                "gap_id": gap.id,
                "sku": gap.sku,
                "required_checks": [gap.metric],
            }
        )
    if gap_resolution_items:
        round_3_round = _create_agent_run_round_safely(
            run_id=run_id,
            round_number=3,
            round_type="gap_resolution",
            status="running",
        )
        round_3_questions = [
            _create_agent_run_question_safely(
                run_id=run_id,
                round_record=round_3_round,
                source_agent="orchestrator",
                target_agent=_intent_from_agent_name(item.get("target_agent")).value,
                question=str(item.get("question") or ""),
                reason=str(item.get("reason") or ""),
                verification_goal=_verification_goal_from_item(item, round_name="round_3"),
                priority=int(item.get("priority") or 0),
                status="pending",
            )
            for item in gap_resolution_items
        ]
        for item, question_record in zip(gap_resolution_items, round_3_questions):
            evidence = _run_domain_evidence(
                item=item,
                user_input=user_input,
                primary_output=primary_output,
                prior_evidence=[*round_1_evidence, *followup_evidence],
                context=context,
                hooks=hooks,
                round_name="round_3",
                allow_failure=True,
                run_id=run_id,
                question_record=question_record,
            )
            round_3_evidence.append(evidence)
            followup_evidence.append(evidence)
        _update_agent_run_round_status_safely(round_3_round, "completed")
    claim_verification_plan = _claim_verification_plan(
        user_input=user_input,
        primary_agent=intent,
        primary_output=primary_output,
        round_1_evidence=round_1_evidence,
        followup_evidence=followup_evidence,
        max_secondary_agents=CLAIM_VERIFICATION_LIMIT,
    )
    claim_verification_required = _is_inventory_turnover_causal_query(user_input)
    if optimized:
        claim_verification_required = (
            claim_verification_required and query_mode == QueryMode.DIAGNOSTIC
        )
        remaining_secondary_budget = max(
            0,
            optimized_secondary_budget
            - len(round_1_evidence)
            - len(followup_evidence),
        )
        claim_verification_items = _plan_items(claim_verification_plan or {})[
            : min(1, remaining_secondary_budget)
        ]
    else:
        claim_verification_items = _plan_items(claim_verification_plan or {})
    claim_verification_evidence: list[dict[str, Any]] = []
    if claim_verification_required:
        claim_round = _create_agent_run_round_safely(
            run_id=run_id,
            round_number=CLAIM_VERIFICATION_ROUND_NUMBER,
            round_type="claim_verification",
            status="running" if claim_verification_items else "skipped",
        )
        claim_questions = [
            _create_agent_run_question_safely(
                run_id=run_id,
                round_record=claim_round,
                source_agent="orchestrator",
                target_agent=_intent_from_agent_name(item.get("target_agent")).value,
                question=str(item.get("question") or ""),
                reason=str(item.get("reason") or ""),
                verification_goal=_verification_goal_from_item(item, round_name="claim_verification"),
                priority=int(item.get("priority") or 0),
                status="pending",
            )
            for item in claim_verification_items
        ]
        for item, question_record in zip(claim_verification_items, claim_questions):
            evidence = _run_domain_evidence(
                item=item,
                user_input=user_input,
                primary_output=primary_output,
                prior_evidence=[*round_1_evidence, *followup_evidence],
                context=context,
                hooks=hooks,
                round_name="claim_verification",
                allow_failure=True,
                run_id=run_id,
                question_record=question_record,
            )
            claim_verification_evidence.append(evidence)
            followup_evidence.append(evidence)
        _update_agent_run_round_status_safely(
            claim_round,
            "completed" if claim_verification_items else "skipped",
        )
    persisted_evidence = _load_persisted_evidence_safely(run_id)
    synthesis_primary_output = primary_output
    synthesis_round_1_evidence = round_1_evidence
    synthesis_followup_evidence = followup_evidence
    if persisted_evidence:
        synthesis_primary_output = persisted_evidence["primary_output"]
        synthesis_round_1_evidence = persisted_evidence["round_1_evidence"]
        synthesis_followup_evidence = persisted_evidence["followup_evidence"]
    persisted_open_gaps = _open_blocking_gap_payloads_safely(
        run_id,
        statuses=("open", "accepted_limitation"),
    )
    synthesis_package = _build_synthesis_package(
        user_input=user_input,
        primary_agent=intent,
        primary_output=synthesis_primary_output,
        round_1_plan=round_1_plan,
        round_1_evidence=synthesis_round_1_evidence,
        round_2_plan=round_2_plan,
        followup_evidence=synthesis_followup_evidence,
        action_safety_checks=action_safety_checks,
        persisted_blocking_gaps=persisted_open_gaps,
        query_plan=context.query_plan,
        deterministic_sku_metric_matrix=build_query_plan_inventory_matrix(
            context.query_plan
        ),
    )
    synthesis_debug = {
        "sales_report_snapshot": (
            synthesis_package.get("primary_evidence", {}).get("sales_report_snapshot")
            if isinstance(synthesis_package.get("primary_evidence"), dict)
            else None
        ),
        "marketing_report_snapshot": (
            synthesis_package.get("primary_evidence", {}).get("marketing_report_snapshot")
            if isinstance(synthesis_package.get("primary_evidence"), dict)
            else None
        ),
        "blocking_gaps": synthesis_package.get("blocking_gaps", []),
        "accepted_limitations": synthesis_package.get("accepted_limitations", []),
        "non_blocking_gaps": synthesis_package.get("non_blocking_gaps", []),
        "resolved_gaps": synthesis_package.get("resolved_gaps", []),
        "claim_verification": synthesis_package.get("claim_verification", {}),
        "claim_verification_plan": claim_verification_plan or {},
        "sku_evidence_matrix": synthesis_package.get("sku_evidence_matrix", []),
        "sku_metric_matrix": synthesis_package.get("sku_metric_matrix", []),
        "domain_coverage": synthesis_package.get("domain_coverage", {}),
        "evidence_quality": synthesis_package.get("evidence_quality", {}),
        "sku_matrix_quality_violations": _sku_matrix_quality_violations(
            {
                "rows": synthesis_package.get("sku_evidence_matrix", []),
                "domain_coverage": synthesis_package.get("domain_coverage", {}),
                "called_domains": _build_sku_evidence_matrix(
                    user_input=user_input,
                    primary_agent=intent,
                    primary_payload=_diagnostic_payload(primary_output),
                    round_1_evidence=round_1_evidence,
                    followup_evidence=followup_evidence,
                ).get("called_domains", []),
            }
        ),
        "action_safety_checks": action_safety_checks,
        "runtime_failures": _reduce_evidence(
            primary_agent=intent,
            primary_output=primary_output,
            round_1_evidence=round_1_evidence,
            followup_evidence=followup_evidence,
        ).get("runtime_failures", []),
    }
    logger.info(
        "sku_evidence_matrix_built primary=%s secondary=%s followup=%s coverage=%s no_data=%s",
        intent.value,
        [evidence["target_agent"] for evidence in round_1_evidence],
        [evidence["target_agent"] for evidence in followup_evidence],
        synthesis_debug.get("domain_coverage", {}),
        len((synthesis_debug.get("evidence_quality") or {}).get("missing_domain_reasons") or []),
    )
    synthesis_started_at = time.monotonic()
    context.add_decision_trace(
        "synthesis_started",
        agent="Diagnostic Orchestrator",
        status="started",
        details={
            "supporting_evidence_count": len(round_1_evidence) + len(followup_evidence),
            "dependency_plan_count": len([plan for plan in [round_1_plan, round_2_plan] if plan]),
        },
    )
    synthesis_round = _create_agent_run_round_safely(
        run_id=run_id,
        round_number=SYNTHESIS_ROUND_NUMBER,
        round_type="synthesis",
        status="running",
    )
    synthesis_question = _create_agent_run_question_safely(
        run_id=run_id,
        round_record=synthesis_round,
        source_agent="orchestrator",
        target_agent="diagnostic_orchestrator",
        question="Собери финальный ответ из persisted evidence package.",
        reason="Final synthesis after persisted agent rounds.",
        verification_goal={
            "round": "synthesis",
            "supporting_evidence_count": len(round_1_evidence) + len(followup_evidence),
        },
        priority=100,
        status="running",
    )
    context.round_type = RoundType.SYNTHESIS
    context.memory_mode = memory_mode_for_round(RoundType.SYNTHESIS)
    synthesis_input_package: dict[str, Any] = synthesis_package
    synthesis_prompt_report: dict[str, Any] | None = None
    if workflow_version != WorkflowVersion.V1:
        synthesis_input_package, compact_report = compact_synthesis_package(
            {
                **synthesis_package,
                "workflow_version": workflow_version.value,
                "query_mode": context.query_mode.value,
                "original_user_query": user_input,
            }
        )
        synthesis_prompt_report = compact_report.model_dump(mode="json")
    final_output = _run_agent(
        build_diagnostic_orchestrator_agent,
        user_input=json.dumps(synthesis_input_package, ensure_ascii=False, default=str),
        context=context,
        hooks=hooks,
        max_turns=DIAGNOSTIC_SYNTHESIS_MAX_TURNS,
        prompt_report=synthesis_prompt_report,
    )
    prompt_v2_quality_violations: list[str] = []
    prompt_v2_repair_attempted = False
    if workflow_version != WorkflowVersion.V1:
        prompt_v2_quality_violations = validate_prompt_v2_answer(
            _coerce_output(final_output),
            synthesis_input_package,
        )
        repair_allowed = not (
            workflow_version == WorkflowVersion.V2_OPTIMIZED
            and context.query_mode in {QueryMode.REPORT, QueryMode.LOOKUP}
        )
        if prompt_v2_quality_violations and repair_allowed:
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
                    hooks=hooks,
                    max_turns=DIAGNOSTIC_SYNTHESIS_MAX_TURNS,
                    prompt_report=synthesis_prompt_report,
                )
                prompt_v2_quality_violations = validate_prompt_v2_answer(
                    _coerce_output(final_output),
                    synthesis_input_package,
                )
            except Exception:
                logger.exception("prompt_v2_synthesis_repair_failed run_id=%s", run_id)
        if prompt_v2_quality_violations:
            final_output = _render_prompt_v2_facts_only_answer(
                user_input,
                synthesis_input_package,
            )
            synthesis_debug["facts_only_after_failed_repair"] = True
        synthesis_debug["prompt_v2_quality_violations"] = prompt_v2_quality_violations
        synthesis_debug["prompt_v2_repair_attempted"] = prompt_v2_repair_attempted
        synthesis_debug["synthesis_prompt_chars"] = len(
            json.dumps(synthesis_input_package, ensure_ascii=False, default=str)
        )
    _persist_agent_run_response_safely(
        run_id=run_id,
        question_record=synthesis_question,
        agent_name="diagnostic_orchestrator",
        payload={
            "summary": _coerce_output(final_output),
            "tables": [],
            "findings": [],
            "recommendations": [],
            "data_gaps": [],
            "cross_agent_requests": [],
            **(
                {"_prompt_report": dict(context.last_prompt_report)}
                if context.last_prompt_report
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
        },
        blocking_gaps=False,
    )
    _update_agent_run_question_status_safely(synthesis_question, "completed")
    _update_agent_run_round_status_safely(synthesis_round, "completed")
    context.add_decision_trace(
        "synthesis_completed",
        agent="Diagnostic Orchestrator",
        status="completed",
        duration_ms=int((time.monotonic() - synthesis_started_at) * 1000),
        details={
            "output_type": type(final_output).__name__,
            "prompt_v2_repair_attempted": prompt_v2_repair_attempted,
            "prompt_v2_quality_violations": prompt_v2_quality_violations,
        },
    )
    return OrchestratedDiagnosticRun(
        final_output=final_output,
        primary_agent=intent,
        secondary_agents=[evidence["target_agent"] for evidence in round_1_evidence],
        followup_agents=[evidence["target_agent"] for evidence in followup_evidence],
        dependency_plans=[plan for plan in [round_1_plan, round_2_plan, claim_verification_plan] if plan],
        synthesis_debug=synthesis_debug,
    )


def _coerce_intent(value: str | None, fallback: QueryIntent) -> QueryIntent:
    if not value:
        return fallback
    legacy_map = {
        "factual_qa": QueryIntent.FINANCE,
    }
    if value in legacy_map:
        return legacy_map[value]
    try:
        return QueryIntent(value)
    except ValueError:
        return fallback


def run_chat_turn(
    user_input: str,
    chat_id: str,
    client_message_id: str | None = None,
) -> ChatTurnResult:
    fallback_intent = determine_query_intent(user_input)

    if client_message_id:
        existing_reply = get_assistant_reply_for_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_reply:
            return ChatTurnResult(
                answer=existing_reply.content,
                intent=_coerce_intent(existing_reply.metadata.get("intent"), fallback_intent),
                assistant_message_seq=existing_reply.seq,
                reused_existing=True,
            )

    context, intent, created_new, route_decision = _prepare_chat_context(
        user_input=user_input,
        chat_id=chat_id,
        client_message_id=client_message_id,
    )

    if client_message_id and not created_new:
        existing_reply = get_assistant_reply_for_client_message_id(
            chat_id=chat_id,
            client_message_id=client_message_id,
        )
        if existing_reply:
            return ChatTurnResult(
                answer=existing_reply.content,
                intent=_coerce_intent(existing_reply.metadata.get("intent"), intent),
                assistant_message_seq=existing_reply.seq,
                reused_existing=True,
            )
        raise DuplicateClientMessageError(client_message_id=client_message_id)

    trace_id: int | None = None
    try:
        domain_routing_signal = route_decision.route_to_domain
        initial_primary_agent = _agent_tool_name(intent) if domain_routing_signal else None
        trace_record = create_agent_run_trace(
            chat_id=chat_id,
            intent=intent.value,
            client_message_id=client_message_id,
            primary_agent=initial_primary_agent,
            summary={
                "status": "running",
                "intent_router": _route_decision_metadata(route_decision),
                "workflow_version": _workflow_version_for_context(context).value,
                "prompt_version": _workflow_version_for_context(context).value,
                "query_mode": context.query_mode.value,
                **_active_model_metadata_safe(),
            },
        )
        trace_id = trace_record.id
        context.run_id = trace_id
        context.add_decision_trace(
            "run_started",
            target_agent=intent.value,
            status="started",
            details={
                "domain_routing_signal": domain_routing_signal,
                "client_message_id": client_message_id,
                "intent_router": _route_decision_metadata(route_decision),
                **_active_model_metadata_safe(),
            },
        )
        hooks = build_run_hooks()
        diagnostic_run: OrchestratedDiagnosticRun | None = None
        if domain_routing_signal:
            diagnostic_run = run_orchestrated_diagnostic_workflow(
                user_input=user_input,
                intent=intent,
                context=context,
                hooks=hooks,
                run_id=trace_id,
            )
            final_output = diagnostic_run.final_output
        else:
            context.round_type = RoundType.ROUTER
            context.memory_mode = memory_mode_for_round(RoundType.ROUTER)
            final_output = _run_agent(
                build_domain_router_agent,
                user_input=user_input,
                context=context,
                hooks=hooks,
                max_turns=24,
            )

        output_text = _coerce_output(final_output)
        final_quality_violations: list[str] = []
        if diagnostic_run:
            output_text = _prepend_required_report_snapshots(
                output_text,
                {
                    "primary_evidence": {
                        "sales_report_snapshot": diagnostic_run.synthesis_debug.get(
                            "sales_report_snapshot"
                        ),
                        "marketing_report_snapshot": diagnostic_run.synthesis_debug.get(
                            "marketing_report_snapshot"
                        ),
                    }
                },
            )
            pre_sanitize_output_text = output_text
            output_text = sanitize_user_facing_markdown(output_text)
            sku_only_display_sanitized = _sku_only_display_was_sanitized(
                pre_sanitize_output_text,
                output_text,
            )
            stripped_output_text = _strip_unrequested_sku_cause_matrix(
                output_text,
                user_query=user_input,
                sku_evidence_matrix=diagnostic_run.synthesis_debug.get(
                    "sku_evidence_matrix",
                    [],
                ),
            )
            sku_matrix_section_removed = stripped_output_text != output_text
            output_text = stripped_output_text
            output_text = apply_query_plan_output_contract(
                output_text,
                {
                    "sku_metric_matrix": diagnostic_run.synthesis_debug.get(
                        "sku_metric_matrix", []
                    )
                },
                context.query_plan,
            )
            output_text, _empty_table_count = remove_empty_markdown_tables(output_text)
            cause_claim_sanitized_output_text = _sanitize_inventory_turnover_cause_claims(
                output_text,
                user_query=user_input,
            )
            inventory_turnover_cause_claim_sanitized = (
                cause_claim_sanitized_output_text != output_text
            )
            output_text = cause_claim_sanitized_output_text
            final_quality_violations = validate_final_diagnostic_markdown(
                output_text,
                user_query=user_input,
            )
            final_quality_violations.extend(
                diagnostic_run.synthesis_debug.get("sku_matrix_quality_violations", [])
            )
            if sku_only_display_sanitized:
                final_quality_violations.append("final markdown was sanitized for SKU-only display")
            if sku_matrix_section_removed:
                final_quality_violations.append("unrequested sku cause matrix removed")
            if inventory_turnover_cause_claim_sanitized:
                final_quality_violations.append("inventory turnover cause overclaim softened")
        metadata = _trace_metadata(context)
        metadata["intent_router"] = _route_decision_metadata(route_decision)
        if diagnostic_run:
            metadata.update(
                {
                    "primary_agent": _agent_tool_name(diagnostic_run.primary_agent),
                    "secondary_agents": [
                        f"{agent}_agent" for agent in diagnostic_run.secondary_agents
                    ],
                    "followup_agents": [
                        f"{agent}_agent" for agent in diagnostic_run.followup_agents
                    ],
                    "dependency_plans": diagnostic_run.dependency_plans,
                    "max_rounds": MAX_DIAGNOSTIC_ROUNDS,
                    "max_total_followup_domain_calls": MAX_TOTAL_FOLLOWUP_DOMAIN_CALLS,
                    "blocking_gaps": diagnostic_run.synthesis_debug.get("blocking_gaps", []),
                    "accepted_limitations": diagnostic_run.synthesis_debug.get(
                        "accepted_limitations", []
                    ),
                    "non_blocking_gaps": diagnostic_run.synthesis_debug.get(
                        "non_blocking_gaps", []
                    ),
                    "resolved_gaps": diagnostic_run.synthesis_debug.get("resolved_gaps", []),
                    "claim_verification": diagnostic_run.synthesis_debug.get(
                        "claim_verification", {}
                    ),
                    "claim_verification_plan": diagnostic_run.synthesis_debug.get(
                        "claim_verification_plan", {}
                    ),
                    "sku_evidence_matrix": diagnostic_run.synthesis_debug.get(
                        "sku_evidence_matrix", []
                    ),
                    "domain_coverage": diagnostic_run.synthesis_debug.get(
                        "domain_coverage", {}
                    ),
                    "evidence_quality": diagnostic_run.synthesis_debug.get(
                        "evidence_quality", {}
                    ),
                    "action_safety_checks": diagnostic_run.synthesis_debug.get(
                        "action_safety_checks", []
                    ),
                    "runtime_failures": diagnostic_run.synthesis_debug.get(
                        "runtime_failures", []
                    ),
                    "final_quality_violations": final_quality_violations,
                }
            )
        else:
            domain_tools = [
                tool_name
                for tool_name in context.tool_trace
                if tool_name in DOMAIN_TOOL_NAMES
            ]
            if domain_tools:
                metadata["primary_agent"] = domain_tools[0]
                metadata["secondary_agents"] = domain_tools[1:]
        action_source = output_text if diagnostic_run else final_output
        action_tasks: list[dict[str, Any]] = []
        if _should_extract_action_tasks(metadata):
            action_tasks = extract_action_tasks_from_agent_output(action_source, output_text)
        if action_tasks:
            metadata[ACTION_TASKS_METADATA_KEY] = action_tasks
            metadata[ACTION_TASKS_SCHEMA_VERSION_KEY] = ACTION_TASKS_SCHEMA_VERSION
        result = _persist_assistant_turn(
            chat_id=chat_id,
            output_text=output_text,
            intent=intent,
            client_message_id=client_message_id,
            metadata=metadata,
        )
        if action_tasks:
            _persist_agent_action_tasks_safely(
                chat_id=chat_id,
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
            diagnostic_run=diagnostic_run,
            final_quality_violations=final_quality_violations,
            status="completed",
        )
        _complete_agent_run_trace_safely(
            trace_id=trace_id,
            context=context,
            assistant_message_seq=result.assistant_message_seq,
            status="completed",
            summary=trace_summary,
            primary_agent=trace_summary.get("primary_agent"),
        )
        return result
    except OutputGuardrailTripwireTriggered as exc:
        violations = _extract_guardrail_violations(exc)
        logger.warning(
            "agent_output_guardrail_triggered chat_id=%s intent=%s violations=%s",
            chat_id,
            intent.value,
            violations,
        )
        context.add_decision_trace(
            "guardrail_failed",
            target_agent=intent.value,
            status="guardrail_failed",
            details={"violations": violations},
        )
        try:
            output_text = _coerce_output(exc.guardrail_result.agent_output)
        except Exception as render_exc:
            trace_summary = _agent_run_trace_summary(
                context=context,
                status="failed",
                error="guardrail_output_render_failed",
            )
            _complete_agent_run_trace_safely(
                trace_id=trace_id,
                context=context,
                assistant_message_seq=None,
                status="failed",
                summary=trace_summary,
                primary_agent=trace_summary.get("primary_agent"),
            )
            raise AgentRuntimeError(
                "Агент сформировал ответ, но он не прошёл внутреннюю проверку и не смог быть отрендерен. "
                "Попробуйте уточнить период или маркетплейс."
            ) from render_exc

        if not output_text.strip():
            trace_summary = _agent_run_trace_summary(
                context=context,
                status="failed",
                error="guardrail_output_empty",
            )
            _complete_agent_run_trace_safely(
                trace_id=trace_id,
                context=context,
                assistant_message_seq=None,
                status="failed",
                summary=trace_summary,
                primary_agent=trace_summary.get("primary_agent"),
            )
            raise AgentRuntimeError(
                "Агент сформировал пустой ответ после внутренней проверки. "
                "Попробуйте уточнить период или маркетплейс."
            ) from exc

        metadata = {
            "guardrail_violations": violations,
            **_trace_metadata(context),
        }
        action_tasks: list[dict[str, Any]] = []
        if _should_extract_action_tasks(metadata):
            action_tasks = extract_action_tasks_from_agent_output(
                exc.guardrail_result.agent_output,
                output_text,
            )
        if action_tasks:
            metadata[ACTION_TASKS_METADATA_KEY] = action_tasks
            metadata[ACTION_TASKS_SCHEMA_VERSION_KEY] = ACTION_TASKS_SCHEMA_VERSION
        result = _persist_assistant_turn(
            chat_id=chat_id,
            output_text=output_text,
            intent=intent,
            client_message_id=client_message_id,
            metadata=metadata,
        )
        if action_tasks:
            _persist_agent_action_tasks_safely(
                chat_id=chat_id,
                assistant_message_seq=result.assistant_message_seq,
                raw_actions=action_tasks,
                intent=intent,
            )
        context.add_decision_trace(
            "run_completed",
            target_agent=intent.value,
            status="guardrail_failed",
            details={"assistant_message_seq": result.assistant_message_seq},
        )
        trace_summary = _agent_run_trace_summary(
            context=context,
            metadata=metadata,
            status="guardrail_failed",
        )
        _complete_agent_run_trace_safely(
            trace_id=trace_id,
            context=context,
            assistant_message_seq=result.assistant_message_seq,
            status="guardrail_failed",
            summary=trace_summary,
            primary_agent=trace_summary.get("primary_agent"),
        )
        return result
    except AgentRuntimeError as exc:
        context.add_decision_trace(
            "runtime_failed",
            target_agent=intent.value,
            status="failed",
            details={"error_type": type(exc).__name__},
        )
        trace_summary = _agent_run_trace_summary(
            context=context,
            status="failed",
            error=type(exc).__name__,
        )
        _complete_agent_run_trace_safely(
            trace_id=trace_id,
            context=context,
            assistant_message_seq=None,
            status="failed",
            summary=trace_summary,
            primary_agent=trace_summary.get("primary_agent"),
        )
        raise
    except Exception as exc:
        context.add_decision_trace(
            "runtime_failed",
            target_agent=intent.value,
            status="failed",
            details={"error_type": type(exc).__name__},
        )
        trace_summary = _agent_run_trace_summary(
            context=context,
            status="failed",
            error=type(exc).__name__,
        )
        _complete_agent_run_trace_safely(
            trace_id=trace_id,
            context=context,
            assistant_message_seq=None,
            status="failed",
            summary=trace_summary,
            primary_agent=trace_summary.get("primary_agent"),
        )
        raise AgentRuntimeError(f"Ошибка во время выполнения агента: {exc}") from exc


def run_main_agent(user_input: str, chat_id: str) -> str:
    return run_chat_turn(user_input=user_input, chat_id=chat_id).answer
