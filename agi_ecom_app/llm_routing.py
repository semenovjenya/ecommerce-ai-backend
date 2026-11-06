from __future__ import annotations

import json
import time
from datetime import date
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from agents import (
    Agent,
    AgentOutputSchema,
    GuardrailFunctionOutput,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    Runner,
    output_guardrail,
)
from pydantic import BaseModel, Field, ValidationError

from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    LLMRuntimeSpec,
    build_agent_model,
    get_active_llm_runtime_spec,
)
from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text, extract_model_payload
from agi_ecom_app.prompt_pipeline import determine_query_mode as determine_prompt_query_mode
from agi_ecom_app.routing import determine_query_intent, has_domain_routing_signal
from agi_ecom_app.schemas import EcomRunContext, QueryIntent, QueryMode
from agi_ecom_app.services.memory import build_memory_context_block


class QueryTarget(BaseModel):
    target_type: Literal[
        "vendor_code",
        "nm_id",
        "campaign_id",
        "category",
        "brand",
        "warehouse",
        "unknown",
    ] = Field(..., description="Business entity type inferred from the request.")
    value: str = Field(..., min_length=1, description="Target value exactly as understood by the router.")


class QueryConstraints(BaseModel):
    target_cover_days: float | None = Field(default=None, ge=0)
    horizon_days: int | None = Field(default=None, ge=1)
    avoid_stockout: bool | None = None
    preserve_contribution_profit: bool | None = None
    notes: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Semantic contract produced once by the LLM router and reused downstream."""

    query_mode: QueryMode = QueryMode.REPORT
    scope: Literal[
        "general",
        "portfolio",
        "category",
        "single_sku",
        "sku_list",
        "campaign",
        "warehouse",
    ] = "general"
    decision_kind: Literal["none", "single_action", "action_portfolio"] = "none"
    objective: str = "answer_user_request"
    targets: list[QueryTarget] = Field(default_factory=list)
    requested_analyses: list[
        Literal[
            "inventory_matrix",
            "inventory_turnover",
            "current_stock",
            "inventory_value",
            "replenishment",
            "sales_velocity",
            "contribution_economics",
            "advertising",
            "pricing",
            "price_scenarios",
            "forecast",
            "conversions",
            "data_freshness",
        ]
    ] = Field(default_factory=list)
    scenario_requests: list[str] = Field(default_factory=list)
    required_domains: list[QueryIntent] = Field(default_factory=list)
    allowed_action_families: list[
        Literal["price", "marketing", "supply"]
    ] = Field(
        default_factory=list,
        description=(
            "Action families explicitly allowed in the final recommendation. "
            "Supporting evidence domains must not expand this list."
        ),
    )
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)


def _fallback_query_plan(user_input: str) -> QueryPlan:
    """Compatibility plan used only when semantic LLM routing is unavailable."""

    query_mode = determine_prompt_query_mode(user_input)
    return QueryPlan(
        query_mode=query_mode,
        decision_kind=("single_action" if query_mode == QueryMode.DECISION else "none"),
    )


class DomainRouteDecision(BaseModel):
    route_to_domain: bool = Field(
        ...,
        description="Whether the message should enter the domain analytics workflow.",
    )
    intent: QueryIntent = Field(
        ...,
        description="Primary domain agent for the user's actual business goal.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence in the primary domain decision.",
    )
    reason: str = Field(..., description="Short routing reason in Russian.")
    supporting_agents: list[QueryIntent] = Field(
        default_factory=list,
        description="Useful supporting domains, not the primary route.",
    )
    query_plan: QueryPlan = Field(
        default_factory=QueryPlan,
        description="Structured semantic interpretation shared by every downstream agent.",
    )


def _fallback_route_decision(user_input: str, reason: str = "fallback_keyword_router") -> DomainRouteDecision:
    intent = determine_query_intent(user_input)
    return DomainRouteDecision(
        route_to_domain=has_domain_routing_signal(user_input),
        intent=intent,
        confidence="low",
        reason=reason,
        supporting_agents=[],
        query_plan=_fallback_query_plan(user_input),
    )


def _coerce_domain_route_decision(raw_output: Any, fallback_decision: DomainRouteDecision) -> DomainRouteDecision:
    try:
        decision = coerce_model_from_json_text(raw_output, DomainRouteDecision)
    except ValidationError:
        payload = extract_model_payload(raw_output, DomainRouteDecision)
        decision = DomainRouteDecision.model_validate(
            _normalize_domain_route_payload(payload, fallback_decision)
        )
    return _repair_domain_route_decision(decision)


def _repair_domain_route_decision(
    decision: DomainRouteDecision,
) -> DomainRouteDecision:
    """Repair relational inconsistencies without re-parsing the user's words.

    Structured output already validates individual fields. Scope/target combinations
    are semantic hints and should not discard an otherwise useful LLM plan.
    """

    supporting_agents = list(
        dict.fromkeys(
            agent
            for agent in decision.supporting_agents
            if agent != decision.intent
        )
    )
    plan = decision.query_plan
    scope = plan.scope
    target_count = len(plan.targets)
    if scope == "single_sku" and target_count != 1:
        scope = "sku_list" if target_count > 1 else "general"
    elif scope == "sku_list" and target_count == 1:
        scope = "single_sku"

    route_to_domain = bool(
        decision.route_to_domain
        or plan.decision_kind != "none"
        or plan.requested_analyses
        or plan.scenario_requests
    )
    required_domains = list(dict.fromkeys(plan.required_domains))
    if route_to_domain and not required_domains:
        required_domains = [decision.intent]
    allowed_action_families = list(dict.fromkeys(plan.allowed_action_families))
    if plan.decision_kind in {"single_action", "action_portfolio"} and not allowed_action_families:
        primary_family = {
            QueryIntent.PRICE: "price",
            QueryIntent.MARKETING: "marketing",
            QueryIntent.SUPPLY: "supply",
        }.get(decision.intent)
        if primary_family:
            allowed_action_families = [primary_family]
    query_mode = plan.query_mode
    if plan.decision_kind in {"single_action", "action_portfolio"}:
        query_mode = QueryMode.DECISION
    repaired_plan = plan.model_copy(
        update={
            "scope": scope,
            "query_mode": query_mode,
            "required_domains": required_domains,
            "allowed_action_families": allowed_action_families,
        }
    )
    return decision.model_copy(
        update={
            "route_to_domain": route_to_domain,
            "supporting_agents": supporting_agents,
            "query_plan": repaired_plan,
        }
    )


def _normalize_domain_route_payload(
    payload: dict[str, Any],
    fallback_decision: DomainRouteDecision,
) -> dict[str, Any]:
    normalized = dict(payload)
    intent = _normalize_domain_intent(normalized.get("intent"))
    route_value = normalized.get("route_to_domain")
    route_domain = _normalize_domain_intent(route_value)
    primary_domain = _normalize_domain_intent(normalized.get("primary_domain"))

    normalized["intent"] = (
        intent
        or route_domain
        or primary_domain
        or fallback_decision.intent.value
    )
    normalized["route_to_domain"] = _normalize_route_flag(
        route_value,
        fallback=fallback_decision.route_to_domain,
        inferred_intent=normalized["intent"],
    )
    normalized["confidence"] = _normalize_confidence(
        normalized.get("confidence"),
        fallback=fallback_decision.confidence,
    )
    reason = str(normalized.get("reason") or normalized.get("rationale") or "").strip()
    normalized["reason"] = reason or fallback_decision.reason
    normalized["supporting_agents"] = [
        item
        for item in (_normalize_domain_intent(agent) for agent in normalized.get("supporting_agents") or [])
        if item and item != normalized["intent"]
    ]
    raw_plan = normalized.get("query_plan")
    if not isinstance(raw_plan, dict):
        normalized["query_plan"] = fallback_decision.query_plan.model_dump(mode="json")
    else:
        raw_plan = dict(raw_plan)
        raw_plan.setdefault("query_mode", fallback_decision.query_plan.query_mode.value)
        raw_plan.setdefault("scope", "general")
        raw_plan.setdefault("decision_kind", "none")
        raw_plan.setdefault("objective", "answer_user_request")
        raw_plan.setdefault("targets", [])
        raw_plan.setdefault("requested_analyses", [])
        raw_plan.setdefault("scenario_requests", [])
        raw_plan.setdefault("required_domains", [normalized["intent"]])
        raw_plan.setdefault("allowed_action_families", [])
        raw_plan.setdefault("constraints", {})
        normalized["query_plan"] = raw_plan
    return normalized


def _normalize_domain_intent(value: Any) -> str | None:
    if isinstance(value, QueryIntent):
        return value.value
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {intent.value for intent in QueryIntent}:
        return normalized
    for intent in QueryIntent:
        if normalized == f"{intent.value}_agent" or normalized.startswith(f"{intent.value}_"):
            return intent.value
    if "sale" in normalized or "продаж" in normalized or "заказ" in normalized:
        return QueryIntent.SALES.value
    if "market" in normalized or "advert" in normalized or "реклам" in normalized:
        return QueryIntent.MARKETING.value
    if "price" in normalized or "цен" in normalized:
        return QueryIntent.PRICE.value
    if "supply" in normalized or "stock" in normalized or "остат" in normalized:
        return QueryIntent.SUPPLY.value
    if "finance" in normalized or "profit" in normalized or "марж" in normalized:
        return QueryIntent.FINANCE.value
    return None


def _normalize_route_flag(value: Any, *, fallback: bool, inferred_intent: str) -> bool:
    if isinstance(value, bool):
        return value
    if _normalize_domain_intent(value):
        return True
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1", "да"}:
        return True
    if normalized in {"false", "no", "0", "нет"}:
        return False
    return bool(fallback or inferred_intent)


def _normalize_confidence(value: Any, *, fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        if normalized in {"высокая", "высокий"}:
            return "high"
        if normalized in {"средняя", "средний"}:
            return "medium"
        if normalized in {"низкая", "низкий"}:
            return "low"
    if isinstance(value, (int, float)):
        if value >= 0.75:
            return "high"
        if value >= 0.4:
            return "medium"
        return "low"
    return fallback


def _intent_router_instructions(context_wrapper, agent) -> str:
    del agent
    context = getattr(context_wrapper, "context", None)
    memory_block = ""
    if isinstance(context, EcomRunContext):
        memory_block = build_memory_context_block(context.as_chat_context_window())
    return (
        "Ты смысловой LLM-router для e-commerce аналитики. Твоя задача — выбрать primary domain "
        "по намерению пользователя, а не по отдельным ключевым словам.\n\n"
        "Верни только DomainRouteDecision JSON без markdown и без пояснений. Поля: "
        "route_to_domain, intent, confidence, reason, supporting_agents, query_plan.\n\n"
        "query_plan — единственный семантический контракт для downstream-кода. Заполни:\n"
        "- query_mode: decision, diagnostic, report или lookup;\n"
        "- scope: general, portfolio, category, single_sku, sku_list, campaign или warehouse;\n"
        "- decision_kind: none, single_action или action_portfolio;\n"
        "- objective: краткая нормализованная бизнес-цель;\n"
        "- targets: только реально названные сущности с target_type и value. Не считай слова "
        "`матрица`, `сравни`, `товары`, `SKU-матрица` артикулами;\n"
        "- requested_analyses: выбери из inventory_matrix, inventory_turnover, current_stock, "
        "inventory_value, replenishment, sales_velocity, contribution_economics, advertising, "
        "pricing, price_scenarios, forecast, conversions, data_freshness; scenario_requests — "
        "какие именно варианты просит пользователь;\n"
        "- required_domains: все домены, без которых нельзя ответить;\n"
        "- allowed_action_families: только типы действий, которые пользователь разрешил "
        "показывать в финальных рекомендациях: price, marketing и/или supply. Домены, "
        "вызванные только для проверки данных, этот список не расширяют. Например, если "
        "пользователь просит только ценовые решения, но разрешает учитывать рекламу, остатки "
        "и поставки как факторы, верни [price];\n"
        "- constraints: target_cover_days, horizon_days, avoid_stockout, "
        "preserve_contribution_profit и дополнительные notes.\n\n"
        "Если запрос относится ко всей товарной матрице, scope=portfolio. targets оставь пустым, "
        "если отдельных фокусных SKU нет; если пользователь явно назвал SKU для углублённой "
        "проверки внутри всей матрицы, сохрани их в targets — это не ограничивает portfolio scope. "
        "Если про один явно названный артикул, scope=single_sku. Если пользователь просит "
        "3–7 или несколько приоритетных действий, decision_kind=action_portfolio.\n\n"
        "Домены:\n"
        "- sales: продажи, заказы, выручка продаж, спрос, динамика SKU, воронка карточки.\n"
        "- marketing: реклама, кампании, CPC/цена клика, ставки, CTR, показы, клики, поисковые запросы.\n"
        "- price: товарная цена, какую цену поставить, скидка, дисконт, переоценка, цены конкурентов, дороже/дешевле рынка.\n"
        "- supply: остатки, оборачиваемость, запас, наличие, поставки, дозаказ, дефицит, out-of-stock.\n"
        "- finance: прибыль, маржа, комиссии, логистика, выплаты, unit economics, финансовые ограничения.\n\n"
        "Правила выбора primary domain:\n"
        "- Сначала определи, какое бизнес-действие или решение просит пользователь.\n"
        "- Если пользователь спрашивает, какую товарную цену поставить, снизить ли цену, какую скидку/дисконт "
        "дать или как переоценить товар, primary = price, даже если причина/фильтр звучит как низкая "
        "оборачиваемость, запас или остатки. Supply/finance могут быть supporting_agents.\n"
        "- Если пользователь просит только посчитать или объяснить оборачиваемость, запас, остатки или дозаказ, primary = supply.\n"
        "- Если пользователь спрашивает про цену клика, CPC или рекламные ставки, primary = marketing, а не price.\n"
        "- Если пользователь спрашивает про прибыль, маржу, финансовый отчет или минимально допустимую цену "
        "по марже без рыночного price-decision, primary = finance.\n"
        "- Если пользователь просит широкую диагностику причин изменения продаж/прибыли по нескольким доменам, "
        "выбери домен главного бизнес-результата; чаще finance для прибыли и sales для продаж.\n"
        "- На приветствия, короткие разговорные реплики, уточнения без аналитической задачи и служебные ответы "
        "ставь route_to_domain=false. Intent в этом случае выбери как лучший fallback, обычно finance.\n"
        "- Не выбирай primary по первому найденному слову. Например, `Какую цену на товар с низкой "
        "оборачиваемостью мы можем поставить?` = price, потому что решение про цену.\n\n"
        f"{memory_block}"
    )


@output_guardrail
def intent_router_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    del context, agent
    violations: list[str] = []
    if not isinstance(output, DomainRouteDecision):
        violations.append("Final output is not DomainRouteDecision.")
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


def build_intent_router_agent() -> Agent[EcomRunContext]:
    return _build_intent_router_agent(get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_intent_router_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Intent Router Agent",
        instructions=_intent_router_instructions,
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(),
        tools=[],
        output_type=(
            AgentOutputSchema(DomainRouteDecision, strict_json_schema=False)
            if runtime_spec.supports_agent_structured_outputs
            else None
        ),
        output_guardrails=(
            [intent_router_output_guardrail]
            if runtime_spec.supports_agent_structured_outputs
            else []
        ),
    )


def route_query_intent_with_llm(
    user_input: str,
    *,
    context: EcomRunContext | None = None,
) -> DomainRouteDecision:
    settings = get_settings()
    fallback_decision = _fallback_route_decision(user_input)
    runtime_spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    if runtime_spec.missing_env_vars and runtime_spec.provider == "openai":
        _record_routing_decision(
            context,
            fallback_decision,
            mode="fallback",
            status="missing_llm_api_key",
            duration_ms=0,
        )
        return fallback_decision
    if runtime_spec.missing_env_vars:
        missing = ", ".join(runtime_spec.missing_env_vars)
        error = LLMRuntimeConfigurationError(
            f"Model profile `{runtime_spec.profile}` requires missing environment variable(s): {missing}."
        )
        _record_routing_decision(
            context,
            fallback_decision,
            mode="failed",
            status="missing_llm_api_key",
            duration_ms=0,
            error=str(error),
        )
        raise error
    if runtime_spec.provider == "openrouter" and not fallback_decision.route_to_domain:
        _record_routing_decision(
            context,
            fallback_decision,
            mode="fallback",
            status="openrouter_simple_chat_bypass",
            duration_ms=0,
        )
        return fallback_decision

    os_context = context or EcomRunContext(
        user_query=user_input,
        intent=fallback_decision.intent,
        today=date.today(),
        db_target=f"{settings.db_host}:{settings.db_port}/{settings.db_name}",
        chat_id=f"intent-router-{uuid4()}",
        evidence_budget=0,
    )
    started_at = time.monotonic()
    payload = {
        "user_query": user_input,
        "fallback_keyword_intent": fallback_decision.intent.value,
        "fallback_keyword_route_to_domain": fallback_decision.route_to_domain,
    }
    try:
        raw_output = Runner.run_sync(
            build_intent_router_agent(),
            json.dumps(payload, ensure_ascii=False, indent=2),
            context=os_context,
            max_turns=4,
        ).final_output
        result = _coerce_domain_route_decision(raw_output, fallback_decision)
        _record_routing_decision(
            context,
            result,
            mode="llm",
            status="completed",
            duration_ms=int((time.monotonic() - started_at) * 1000),
            fallback_decision=fallback_decision,
        )
        return result
    except Exception as exc:
        if isinstance(exc, OutputGuardrailTripwireTriggered):
            try:
                recovered = _coerce_domain_route_decision(
                    exc.guardrail_result.agent_output,
                    fallback_decision,
                )
            except Exception:
                recovered = None
            if recovered is not None:
                _record_routing_decision(
                    context,
                    recovered,
                    mode="llm",
                    status="guardrail_recovered",
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    fallback_decision=fallback_decision,
                    error=str(exc),
                )
                return recovered
        if runtime_spec.provider != "openai":
            _record_routing_decision(
                context,
                fallback_decision,
                mode="failed",
                status="failed",
                duration_ms=int((time.monotonic() - started_at) * 1000),
                fallback_decision=fallback_decision,
                error=str(exc),
            )
            raise
        fallback_with_reason = _fallback_route_decision(
            user_input,
            reason=f"llm_router_fallback:{type(exc).__name__}",
        )
        _record_routing_decision(
            context,
            fallback_with_reason,
            mode="fallback",
            status="failed",
            duration_ms=int((time.monotonic() - started_at) * 1000),
            fallback_decision=fallback_decision,
            error=str(exc),
        )
        return fallback_with_reason


build_intent_router_agent.cache_clear = _build_intent_router_agent.cache_clear  # type: ignore[attr-defined]


def _record_routing_decision(
    context: EcomRunContext | None,
    decision: DomainRouteDecision,
    *,
    mode: str,
    status: str,
    duration_ms: int,
    fallback_decision: DomainRouteDecision | None = None,
    error: str | None = None,
) -> None:
    if context is None:
        return
    details: dict[str, Any] = {
        "mode": mode,
        "route_to_domain": decision.route_to_domain,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "supporting_agents": [agent.value for agent in decision.supporting_agents],
    }
    if fallback_decision is not None:
        details["fallback_keyword_intent"] = fallback_decision.intent.value
        details["fallback_keyword_route_to_domain"] = fallback_decision.route_to_domain
    if error:
        details["error"] = error
    context.add_decision_trace(
        "intent_router_completed",
        agent="Intent Router Agent" if mode == "llm" else "keyword_fallback_router",
        target_agent=decision.intent.value,
        status=status,
        duration_ms=duration_ms,
        details=details,
    )
