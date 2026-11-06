from __future__ import annotations

from datetime import date
from functools import lru_cache
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents import Agent, GuardrailFunctionOutput, ModelSettings, OutputGuardrailTripwireTriggered, Runner, output_guardrail

from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    LLMRuntimeSpec,
    build_agent_model,
    get_active_llm_runtime_spec,
)
from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text
from agi_ecom_app.main_agent import build_run_hooks
from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import (
    ForecastPlan,
    ForecastRevisionInput,
    ForecastReasoningInput,
    ForecastReasoningResult,
    GrowthOpportunity,
    ResolvedTarget,
)
from agi_ecom_app.schemas import EcomRunContext, QueryIntent
from agi_ecom_app.tools.db_tools import get_database_time
from agi_ecom_app.tools.role_db_tools import (
    FINANCE_SCHEMAS,
    MARKETING_SCHEMAS,
    PRICE_SCHEMAS,
    SALES_SCHEMAS,
    SUPPLY_SCHEMAS,
    build_role_db_tools,
)
from agi_ecom_app.tools.strategic_tools import (
    get_finance_overview,
    get_marketing_action_matrix,
    get_marketing_campaign_daily_mart,
    get_marketing_campaign_demand_forecast_mart,
    get_marketing_campaign_efficiency_snapshot,
    get_marketing_forecast_ready_mart,
    get_marketing_overview,
    get_marketing_query_bands_mart,
    get_marketing_query_demand_daily_mart,
    get_marketing_query_forecast_ready_mart,
    get_marketing_query_performance_snapshot,
    get_marketing_query_share_daily_mart,
    get_marketing_search_demand_snapshot,
    get_marketing_visibility_snapshot,
    get_price_competitor_overview,
    get_price_context,
    get_price_decision_overview,
    get_sales_overview,
    get_sales_sku_revenue_change_report,
    get_supply_overview,
)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "forecast_reasoning_agent.md"
SOFT_WARNING_TOKENS = ("stale_", "low_", "unpopulated", "test_only", "needs_review")


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _context_header(context: EcomRunContext) -> str:
    return (
        "Runtime context:\n"
        f"- today: {context.today.isoformat()}\n"
        f"- intent: {context.intent.value}\n"
        f"- db_target: {context.db_target}\n"
        f"- chat_id: {context.chat_id}\n"
        f"- evidence_budget: {context.evidence_budget}\n"
        f"- user_query: {context.user_query}"
    )


def _dynamic_instructions(context_wrapper, agent) -> str:
    base_prompt = _load_prompt()
    context = getattr(context_wrapper, "context", None)
    if isinstance(context, EcomRunContext):
        return f"{base_prompt}\n\n{_context_header(context)}"
    return base_prompt


@output_guardrail
def forecast_reasoning_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    violations: list[str] = []
    if not isinstance(output, ForecastReasoningResult):
        violations.append("Final output is not ForecastReasoningResult.")
    else:
        if not str(output.final_recommendation or "").strip():
            violations.append("final_recommendation is empty.")
        if not str(output.summary or "").strip():
            violations.append("summary is empty.")
        if not str(output.primary_action or "").strip():
            violations.append("primary_action is empty.")
        if not str(output.decision or "").strip():
            violations.append("decision is empty.")
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


def build_forecast_reasoning_agent(*, allow_tools: bool = True) -> Agent[EcomRunContext]:
    return _build_forecast_reasoning_agent(
        get_active_llm_runtime_spec(),
        allow_tools=allow_tools,
    )


@lru_cache(maxsize=16)
def _build_forecast_reasoning_agent(
    runtime_spec: LLMRuntimeSpec,
    *,
    allow_tools: bool,
) -> Agent[EcomRunContext]:
    return Agent(
        name="Forecast Reasoning Agent",
        instructions=_dynamic_instructions,
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="auto" if allow_tools else "none"),
        tools=_forecast_reasoning_tools() if allow_tools else [],
        # Forecast reasoning already has a deterministic fallback. Plain JSON
        # avoids marking the whole Agents SDK turn as ERROR when a provider
        # emits an empty structured payload; Pydantic validation below remains
        # authoritative.
        output_type=None,
        output_guardrails=[],
    )


build_forecast_reasoning_agent.cache_clear = _build_forecast_reasoning_agent.cache_clear  # type: ignore[attr-defined]


def _forecast_reasoning_tools() -> list[Any]:
    return [
        get_database_time,
        get_sales_overview,
        get_sales_sku_revenue_change_report,
        get_marketing_overview,
        get_marketing_action_matrix,
        get_marketing_campaign_daily_mart,
        get_marketing_query_bands_mart,
        get_marketing_forecast_ready_mart,
        get_marketing_query_demand_daily_mart,
        get_marketing_query_share_daily_mart,
        get_marketing_query_forecast_ready_mart,
        get_marketing_campaign_demand_forecast_mart,
        get_marketing_campaign_efficiency_snapshot,
        get_marketing_query_performance_snapshot,
        get_marketing_visibility_snapshot,
        get_marketing_search_demand_snapshot,
        get_price_context,
        get_price_decision_overview,
        get_price_competitor_overview,
        get_supply_overview,
        get_finance_overview,
        *build_role_db_tools("sales", SALES_SCHEMAS),
        *build_role_db_tools("marketing", MARKETING_SCHEMAS),
        *build_role_db_tools("price", PRICE_SCHEMAS),
        *build_role_db_tools("supply", SUPPLY_SCHEMAS),
        *build_role_db_tools("finance", FINANCE_SCHEMAS),
    ]


def build_forecast_reasoning_input(
    *,
    hypothesis: str,
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    scenario_output: dict[str, Any],
    opportunity: GrowthOpportunity | None = None,
) -> ForecastReasoningInput:
    hard_blockers, soft_warnings = _split_blockers(
        scenario_blockers=scenario_output.get("blocked_by"),
        bundle_blockers=evidence_bundle.blocking_gaps,
    )
    return ForecastReasoningInput(
        hypothesis=hypothesis,
        plan=plan.model_dump(mode="json"),
        resolved_targets=[target.model_dump(mode="json") for target in resolved_targets],
        per_target_evidence=_merge_per_target_evidence(
            resolved_targets=resolved_targets,
            evidence_bundle=evidence_bundle,
            scenario_output=scenario_output,
        ),
        baseline_metrics=_clean_dict(scenario_output.get("baseline_metrics")),
        action_metrics=_clean_dict(scenario_output.get("action_metrics")),
        delta_metrics=_clean_dict(scenario_output.get("delta_metrics")),
        hard_blockers=hard_blockers,
        soft_warnings=soft_warnings,
        data_freshness=_build_data_freshness(hard_blockers, soft_warnings, evidence_bundle, scenario_output),
        source_status=_build_source_status(evidence_bundle),
        margin_bridge=_clean_dict(scenario_output.get("margin_bridge")),
        opportunity=opportunity.model_dump(mode="json") if opportunity is not None else _dict_or_none(evidence_bundle.diagnostic.get("opportunity")),
        evidence_summary={
            "diagnostic": _clean_dict(evidence_bundle.diagnostic),
            "confirmed_facts": list(evidence_bundle.confirmed_facts),
            "worker_confidence": {
                worker_name: worker_output.get("confidence")
                for worker_name, worker_output in evidence_bundle.workers.items()
                if isinstance(worker_output, dict)
            },
        },
        calculator_output=_clean_dict(scenario_output),
    )


def run_forecast_reasoning(
    *,
    reasoning_input: ForecastReasoningInput | ForecastRevisionInput,
    plan: ForecastPlan,
    today: date | None = None,
    max_turns: int = 10,
) -> tuple[ForecastReasoningResult, dict[str, Any]]:
    settings = get_settings()
    run_day = today or date.today()
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    if runtime_spec.missing_env_vars:
        if runtime_spec.provider != "openai":
            missing = ", ".join(runtime_spec.missing_env_vars)
            raise LLMRuntimeConfigurationError(
                f"Model profile `{runtime_spec.profile}` requires missing environment variable(s): {missing}."
            )
        fallback = _fallback_reasoning(reasoning_input)
        return fallback, {
            "mode": "fallback",
            "reason": "missing_llm_api_key",
            "provider": runtime_spec.provider,
            "profile": runtime_spec.profile,
            "missing_env_vars": list(runtime_spec.missing_env_vars),
        }

    context = EcomRunContext(
        user_query=_reasoning_query(reasoning_input),
        intent=_intent_for_plan(plan),
        today=run_day,
        db_target="analytics",
        chat_id=f"forecast-reasoning-{uuid4()}",
        evidence_budget=8,
    )
    hooks = build_run_hooks()
    try:
        raw_output = Runner.run_sync(
            build_forecast_reasoning_agent(
                # The deterministic evidence bundle already contains every
                # allowed fact for this forecast. Re-exposing the full DB tool
                # catalogue adds thousands of schema tokens per SKU and can
                # introduce a second, inconsistent snapshot.
                allow_tools=False
            ),
            json.dumps(
                build_forecast_reasoning_payload(reasoning_input),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            context=context,
            hooks=hooks,
            max_turns=max_turns,
        ).final_output
        result = coerce_model_from_json_text(raw_output, ForecastReasoningResult)
        empty_fields = [
            field
            for field, value in (
                ("decision", result.decision),
                ("final_recommendation", result.final_recommendation),
                ("summary", result.summary),
                ("primary_action", result.primary_action),
            )
            if not str(value or "").strip()
        ]
        if empty_fields:
            raise ValueError(
                "Forecast reasoning returned empty required fields: "
                + ", ".join(empty_fields)
            )
        debug = {
            "mode": "agent",
            "used_tools": list(dict.fromkeys(context.tool_trace)),
            "tool_trace": list(context.tool_trace),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }
        return result, debug
    except (OutputGuardrailTripwireTriggered, Exception) as exc:
        if runtime_spec.provider != "openai":
            raise
        fallback = _fallback_reasoning(reasoning_input)
        return fallback, {
            "mode": "fallback",
            "reason": type(exc).__name__,
            "error": str(exc),
            "used_tools": list(dict.fromkeys(context.tool_trace)),
            "tool_trace": list(context.tool_trace),
            "agent_trace": list(context.agent_trace),
            "decision_trace": list(context.decision_trace),
        }


def _reasoning_query(reasoning_input: ForecastReasoningInput) -> str:
    if isinstance(reasoning_input, ForecastRevisionInput):
        return (
            f"Пересобери forecast-решение после замечаний критика для action family "
            f"`{reasoning_input.plan.get('action_family')}`. Попытка: {reasoning_input.retry_attempt}."
        )
    targets = ", ".join(
        str(target.get("id"))
        for target in reasoning_input.resolved_targets
        if isinstance(target, dict) and target.get("id")
    )
    return (
        f"Прими финальное growth/forecast решение для action family "
        f"`{reasoning_input.plan.get('action_family')}` по целям: {targets or 'n/a'}."
    )


def _intent_for_plan(plan: ForecastPlan) -> QueryIntent:
    mapping = {
        "marketing": QueryIntent.MARKETING,
        "price": QueryIntent.PRICE,
        "supply": QueryIntent.SUPPLY,
        "finance": QueryIntent.FINANCE,
        "sales": QueryIntent.SALES,
        "catalog_demand": QueryIntent.SALES,
    }
    return mapping.get(plan.primary_domain, QueryIntent.SALES)


def _fallback_reasoning(
    reasoning_input: ForecastReasoningInput | ForecastRevisionInput,
) -> ForecastReasoningResult:
    recommendation = (
        _string(reasoning_input.calculator_output.get("final_recommendation"))
        or _string(reasoning_input.calculator_output.get("decision"))
        or "Не удалось получить AI reasoning; нужен повторный запуск."
    )
    summary = (
        _string(reasoning_input.calculator_output.get("summary"))
        or _string(reasoning_input.calculator_output.get("effect_narrative"))
        or "Сценарий посчитан детерминированно, но reasoning-слой недоступен."
    )
    decision_why = (
        _string(reasoning_input.calculator_output.get("decision_why"))
        or "AI reasoning недоступен, поэтому показан fallback на deterministic calculator."
    )
    primary_action = _string(reasoning_input.plan.get("action_family")) or "needs_review"
    warnings = list(reasoning_input.soft_warnings)
    if "reasoning_agent_unavailable" not in warnings:
        warnings.append("reasoning_agent_unavailable")
    return ForecastReasoningResult(
        decision_status=_normalize_status(reasoning_input.calculator_output.get("decision_status")),
        confidence=_normalize_confidence(reasoning_input.calculator_output.get("confidence")),
        decision=_string(reasoning_input.calculator_output.get("decision")) or recommendation,
        final_recommendation=recommendation,
        summary=summary,
        decision_why=decision_why,
        primary_action=primary_action,
        rejected_actions=[],
        warnings=warnings,
        evidence_used=["deterministic_math_layer"],
        ui_card_fields=None,
        resolution_plan_patch=[],
        forecast_details_patch={
            "reasoningPrimaryAction": primary_action,
            "reasoningRejectedActions": [],
            "reasoningEvidenceUsed": ["deterministic_math_layer"],
            "reasoningWarnings": warnings,
            "reasoningMode": "fallback",
        },
    )


def build_forecast_revision_input(
    *,
    reasoning_input: ForecastReasoningInput,
    previous_decision,
    critic_result,
    retry_attempt: int,
) -> ForecastRevisionInput:
    payload = reasoning_input.model_dump(mode="json")
    for key in ("previous_decision", "critic_issues", "revision_brief", "retry_attempt"):
        payload.pop(key, None)
    return ForecastRevisionInput(
        **payload,
        previous_decision=_compact_previous_decision(previous_decision),
        critic_issues=[issue.model_dump(mode="json") for issue in critic_result.issues],
        revision_brief=list(critic_result.revisionBrief),
        retry_attempt=retry_attempt,
    )


def build_forecast_reasoning_payload(
    reasoning_input: ForecastReasoningInput | ForecastRevisionInput,
) -> dict[str, Any]:
    """Build the unique business context sent to the reasoning model.

    Internal forecast objects intentionally keep rich evidence/debug payloads for
    persistence. The LLM does not need repeated copies of opportunity facts,
    per-target calculator rows, or the complete prior debug tree.
    """

    payload = reasoning_input.model_dump(mode="json", exclude_none=True)
    payload["per_target_evidence"] = _strip_repeated_evidence(
        payload.get("per_target_evidence") or []
    )

    calculator_output = dict(payload.get("calculator_output") or {})
    for key in ("per_target_evidence", "opportunity", "source_facts"):
        calculator_output.pop(key, None)
    payload["calculator_output"] = _strip_repeated_evidence(calculator_output)

    evidence_summary = dict(payload.get("evidence_summary") or {})
    diagnostic = dict(evidence_summary.get("diagnostic") or {})
    diagnostic.pop("opportunity", None)
    evidence_summary["diagnostic"] = _strip_repeated_evidence(diagnostic)
    payload["evidence_summary"] = evidence_summary

    if isinstance(reasoning_input, ForecastRevisionInput):
        payload["previous_decision"] = _compact_previous_decision_payload(
            payload.get("previous_decision") or {}
        )
    return payload


def _compact_previous_decision(previous_decision: Any) -> dict[str, Any]:
    if hasattr(previous_decision, "model_dump"):
        payload = previous_decision.model_dump(mode="json", exclude_none=True)
    elif isinstance(previous_decision, dict):
        payload = dict(previous_decision)
    else:
        return {}
    return _compact_previous_decision_payload(payload)


def _compact_previous_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "hypothesis",
        "domain",
        "baseline",
        "actionScenario",
        "delta",
        "decision",
        "decisionStatus",
        "confidence",
        "confidencePercent",
        "profitForecastRub",
        "economicEffectRub",
        "economicEffectPctOfCurrentSpend",
        "blockedBy",
        "finalRecommendation",
        "summary",
        "resolutionActions",
        "resolutionPlan",
        "plan",
        "resolvedTargets",
        "baselineNarrative",
        "actionNarrative",
        "effectNarrative",
        "decisionWhy",
        "reviewBlocked",
        "reviewBlockReason",
        "uiCard",
    )
    return {key: payload[key] for key in fields if key in payload}


def _strip_repeated_evidence(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_repeated_evidence(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _strip_repeated_evidence(item)
        for key, item in value.items()
        if key
        not in {
            "entity_matches",
            "opportunity",
            "opportunity_source_facts",
            "query_rows",
            "source_facts",
        }
    }


def _merge_per_target_evidence(
    *,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    scenario_output: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario_rows = scenario_output.get("per_target_evidence")
    indexed_rows: dict[str, dict[str, Any]] = {}
    if isinstance(scenario_rows, list):
        for row in scenario_rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("target") or row.get("target_id") or "").strip()
            if key:
                indexed_rows[key] = row

    result: list[dict[str, Any]] = []
    for target in resolved_targets:
        facts = evidence_bundle.per_target.get(target.id) or {}
        merged = {
            "target": target.id,
            "entity_type": target.entity_type,
            "facts": facts,
        }
        if target.id in indexed_rows:
            merged["scenario"] = indexed_rows[target.id]
        result.append(merged)
    return result


def _split_blockers(
    *,
    scenario_blockers: Any,
    bundle_blockers: list[str],
) -> tuple[list[str], list[str]]:
    hard_blockers: list[str] = []
    soft_warnings: list[str] = []
    for item in [*list(scenario_blockers or []), *list(bundle_blockers or [])]:
        text = _string(item)
        if not text:
            continue
        target_list = soft_warnings if _is_soft_warning(text) else hard_blockers
        if text not in target_list:
            target_list.append(text)
    return hard_blockers, soft_warnings


def _is_soft_warning(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in SOFT_WARNING_TOKENS)


def _build_data_freshness(
    hard_blockers: list[str],
    soft_warnings: list[str],
    evidence_bundle: ForecastEvidenceBundle,
    scenario_output: dict[str, Any],
) -> dict[str, Any]:
    freshness_signals = [item for item in [*hard_blockers, *soft_warnings] if "stale" in item or "fresh" in item]
    unpopulated_sources = [item for item in [*hard_blockers, *soft_warnings] if "unpopulated" in item]
    return {
        "freshness_signals": freshness_signals,
        "unpopulated_sources": unpopulated_sources,
        "blocking_gaps": list(evidence_bundle.blocking_gaps),
        "calculator_confidence": scenario_output.get("confidence"),
    }


def _build_source_status(evidence_bundle: ForecastEvidenceBundle) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for worker_name, worker_output in evidence_bundle.workers.items():
        if not isinstance(worker_output, dict):
            continue
        result[worker_name] = {
            "confidence": worker_output.get("confidence"),
            "matched_targets": _dict_or_none(worker_output.get("facts")) or {},
            "data_gaps": list(worker_output.get("data_gaps") or []),
            "recommended_interpretation": worker_output.get("recommended_interpretation"),
        }
    return result


def _clean_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if item is not None}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _normalize_status(value: Any) -> str:
    text = _string(value) or "needs_review"
    if text in {"completed", "needs_review", "blocked"}:
        return text
    return "needs_review"


def _normalize_confidence(value: Any) -> str:
    text = _string(value) or "medium"
    if text in {"low", "medium", "high"}:
        return text
    return "medium"


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if value is None:
        return None
    return str(value)
