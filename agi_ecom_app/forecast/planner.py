from __future__ import annotations

import re
from typing import Any

from agi_ecom_app.forecast.llm_router import route_task_with_llm
from agi_ecom_app.forecast.models import ForecastHypothesis, ForecastPlan, ForecastPlannerHint, ResolvedTarget

_NUMERIC_TARGET_PATTERN = re.compile(r"\b\d{4,}\b")


def build_forecast_plan(
    task_snapshot: dict[str, Any] | None,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> ForecastPlan | None:
    if not isinstance(task_snapshot, dict):
        return None

    task_snapshot = _normalized_snapshot(task_snapshot)
    context = _task_context(task_snapshot)
    routing_context = _routing_context(task_snapshot)
    planner_hints = list(planner_hints or [])
    forced_action_family = _forced_action_family(planner_hints)
    forced_target_mode = _forced_target_mode(planner_hints)
    excluded_families = _excluded_action_families(planner_hints)
    action_kind = _string(task_snapshot.get("actionKind"))
    decision_type = _string(task_snapshot.get("decisionType"))
    effect_metrics = _effect_metrics(task_snapshot)
    explicit_action_family = _string(_forecast_spec(task_snapshot).get("action_family"))
    if explicit_action_family:
        target_mode = _forecast_spec(task_snapshot).get("target_mode")
        return _build_plan_from_family(
            task_snapshot=task_snapshot,
            action_family=explicit_action_family,
            primary_domain=_explicit_primary_domain(task_snapshot, explicit_action_family),
            target_mode=(
                target_mode
                if target_mode in {"sku", "campaign", "query", "mixed", "unknown"}
                else _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints)
            ),
            source_action_kind=action_kind,
            effect_metrics=effect_metrics,
            planner_hints=planner_hints,
        )

    if forced_action_family and forced_action_family not in excluded_families:
        return _build_plan_from_family(
            task_snapshot=task_snapshot,
            action_family=forced_action_family,
            primary_domain=_explicit_primary_domain(task_snapshot, forced_action_family),
            target_mode=forced_target_mode or _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints),
            source_action_kind=action_kind,
            effect_metrics=effect_metrics,
            planner_hints=planner_hints,
        )

    if (
        "stop_or_reduce_campaigns" not in excluded_families
        and _is_marketing_stop_reduce(routing_context, action_kind, decision_type)
    ):
        return ForecastPlan(
            primary_domain="marketing",
            action_family="stop_or_reduce_campaigns",
            target_mode=forced_target_mode or "campaign",
            secondary_domains=["sales"],
            horizon_days=_horizon_days(task_snapshot),
            period_days=_period_days(task_snapshot),
            effect_metrics=effect_metrics
            or ["ad_spend_saved_rub", "orders_at_risk", "estimated_drr"],
            goal=_string(task_snapshot.get("businessGoal"))
            or "Снизить неэффективный рекламный расход без потери полезных заказов.",
            source_action_kind=action_kind,
        )

    if (
        "availability_before_scale_guardrail" not in excluded_families
        and _is_availability_before_scale(routing_context, action_kind, decision_type)
    ):
        return ForecastPlan(
            primary_domain="supply",
            action_family="availability_before_scale_guardrail",
            target_mode=forced_target_mode or "sku",
            secondary_domains=["marketing"],
            horizon_days=_horizon_days(task_snapshot),
            period_days=_period_days(task_snapshot),
            effect_metrics=effect_metrics
            or ["coverage_days", "oos_risk", "reorder_qty", "lost_sales_risk"],
            goal=_string(task_snapshot.get("businessGoal"))
            or "Понять, выдержат ли остатки расширение рекламы без риска out-of-stock.",
            source_action_kind=action_kind,
        )

    if (
        "weak_sku_demand_recovery" not in excluded_families
        and _is_weak_sku_recovery(routing_context, action_kind, decision_type)
    ):
        return ForecastPlan(
            primary_domain="sales",
            action_family="weak_sku_demand_recovery",
            target_mode=forced_target_mode or "sku",
            secondary_domains=["catalog_demand", "marketing", "supply", "price"],
            horizon_days=_horizon_days(task_snapshot),
            period_days=_period_days(task_snapshot),
            effect_metrics=effect_metrics
            or ["delta_sales", "delta_revenue", "trend_direction", "risk_level"],
            goal=_string(task_snapshot.get("businessGoal"))
            or "Понять причину просадки по слабым SKU и выбрать следующий лучший шаг.",
            source_action_kind=action_kind,
        )

    if (
        "normquery_bid_optimization" not in excluded_families
        and _is_single_sku_marketing_review(task_snapshot, routing_context, action_kind, decision_type)
    ):
        return _build_plan_from_family(
            task_snapshot=task_snapshot,
            action_family="normquery_bid_optimization",
            primary_domain="marketing",
            target_mode=forced_target_mode or "sku",
            source_action_kind=action_kind,
            effect_metrics=effect_metrics,
            planner_hints=planner_hints,
        )

    if action_kind == "diagnose_sales_performance" or decision_type == "sales_performance_diagnostic":
        return ForecastPlan(
            primary_domain="sales",
            action_family="sales_trend_guardrail",
            target_mode=forced_target_mode or _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints),
            secondary_domains=["price", "supply"],
            horizon_days=_horizon_days(task_snapshot),
            period_days=_period_days(task_snapshot),
            effect_metrics=effect_metrics
            or ["current_sales", "current_revenue", "delta_sales", "delta_revenue"],
            goal=_string(task_snapshot.get("businessGoal")) or "Понять тренд продаж и риск дальнейшей просадки.",
            source_action_kind=action_kind,
        )

    if action_kind == "check_supply_readiness" or decision_type == "supply_readiness_check":
        return ForecastPlan(
            primary_domain="supply",
            action_family="availability_before_scale_guardrail",
            target_mode=forced_target_mode or _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints),
            secondary_domains=["sales"],
            horizon_days=_horizon_days(task_snapshot),
            period_days=_period_days(task_snapshot),
            effect_metrics=effect_metrics
            or ["coverage_days", "oos_risk", "reorder_qty", "lost_sales_risk"],
            goal=_string(task_snapshot.get("businessGoal")) or "Понять риск OOS и необходимость дозаказа.",
            source_action_kind=action_kind,
        )

    if action_kind == "validate_price_positioning" or decision_type == "price_margin_competitor_guardrail":
        return _build_plan_from_family(
            task_snapshot=task_snapshot,
            action_family="price_positioning_guardrail",
            primary_domain="price",
            target_mode=forced_target_mode or _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints),
            source_action_kind=action_kind,
            effect_metrics=effect_metrics,
            planner_hints=planner_hints,
        )

    llm_plan = _llm_route_plan(
        task_snapshot=task_snapshot,
        action_kind=action_kind,
        decision_type=decision_type,
        effect_metrics=effect_metrics,
        context=routing_context,
        planner_hints=planner_hints,
    )
    if llm_plan is not None:
        return llm_plan

    return None


def resolve_forecast_targets(
    task_snapshot: dict[str, Any] | None,
    plan: ForecastPlan,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> list[ResolvedTarget]:
    if not isinstance(task_snapshot, dict):
        return []

    planner_hints = list(planner_hints or [])
    ids = _resolve_targets(task_snapshot, plan)
    confidence = "high" if ids else "low"
    targets = [
        ResolvedTarget(
            id=target_id,
            entity_type=_entity_type_for_target(plan, task_snapshot, target_id, planner_hints=planner_hints),
            source_type="task_snapshot",
            confidence=confidence,
        )
        for target_id in ids
    ]
    return _deduplicate_targets(targets)


def build_forecast_hypothesis(task_snapshot: dict[str, Any] | None) -> ForecastHypothesis | None:
    plan = build_forecast_plan(task_snapshot)
    if plan is None:
        return None
    resolved_targets = resolve_forecast_targets(task_snapshot, plan)
    return ForecastHypothesis(
        domain=plan.primary_domain,
        action=plan.action_family,
        targets=[target.id for target in resolved_targets],
        horizon_days=plan.horizon_days,
        period_days=plan.period_days,
        goal=plan.goal,
        effect_metrics=list(plan.effect_metrics),
        source_action_kind=plan.source_action_kind,
    )


def _is_marketing_stop_reduce(context: str, action_kind: str | None, decision_type: str | None) -> bool:
    is_marketing_visibility = (
        action_kind == "optimize_marketing_visibility"
        or decision_type == "marketing_visibility_guardrail"
    )
    has_stop_reduce = any(marker in context for marker in ("останов", "сократ", "stop", "reduce"))
    has_campaign_context = any(marker in context for marker in ("кампан", "advert", "реклам"))
    weak_sku_context = any(marker in context for marker in ("sku", "карточк", "просад", "nm_id", "артикул"))
    return is_marketing_visibility and has_stop_reduce and has_campaign_context and not weak_sku_context


def _is_availability_before_scale(context: str, action_kind: str | None, decision_type: str | None) -> bool:
    if action_kind == "preserve_availability_and_sales" or decision_type == "availability_replenishment_marketing_guardrail":
        return True
    has_supply = any(marker in context for marker in ("доступност", "остат", "налич", "запас", "постав", "coverage"))
    has_scale = any(marker in context for marker in ("расшир", "масштаб", "реклам", "кампан", "scale"))
    return has_supply and has_scale


def _is_weak_sku_recovery(context: str, action_kind: str | None, decision_type: str | None) -> bool:
    if action_kind == "diagnose_card_funnel" or decision_type == "card_funnel_diagnostic":
        return True
    has_sku = any(marker in context for marker in ("sku", "артикул", "nm_id", "карточк"))
    has_weakness = any(
        marker in context
        for marker in ("слаб", "просад", "падает", "падени", "спрос", "выручк", "продаж")
    )
    has_campaign_stop = (
        not has_sku
        and any(marker in context for marker in ("останов", "сократ", "stop", "reduce"))
        and any(marker in context for marker in ("кампан", "advert", "реклам"))
    )
    return has_sku and has_weakness and not has_campaign_stop


def _is_single_sku_marketing_review(
    task_snapshot: dict[str, Any],
    context: str,
    action_kind: str | None,
    decision_type: str | None,
) -> bool:
    is_marketing_visibility = (
        action_kind == "optimize_marketing_visibility"
        or decision_type == "marketing_visibility_guardrail"
    )
    if not is_marketing_visibility:
        return False
    object_targets = _numeric_targets_from_objects(task_snapshot.get("objects"))
    explicit_targets = [
        _extract_numeric_identifier(str(item).strip())
        for item in (task_snapshot.get("targets") or [])
        if _extract_numeric_identifier(str(item).strip())
    ]
    sku_targets = [target for target in object_targets or explicit_targets if target and len(target) <= 6]
    has_sku_object = _has_sku_object(task_snapshot.get("objects"))
    has_marketing_review_markers = any(marker in context for marker in ("расход", "кампан", "ставк", "видимост", "запрос"))
    return has_sku_object and len(set(sku_targets)) == 1 and has_marketing_review_markers


def _entity_type_for_plan(plan: ForecastPlan) -> str:
    if plan.target_mode in {"campaign", "query", "sku"}:
        return plan.target_mode
    return "unknown"


def _entity_type_for_target(
    plan: ForecastPlan,
    task_snapshot: dict[str, Any],
    target_id: str,
    *,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> str:
    forced_entity_type = _forced_entity_type(planner_hints or [])
    if forced_entity_type:
        return forced_entity_type
    if plan.target_mode in {"campaign", "query", "sku"}:
        return plan.target_mode

    context = _routing_context(task_snapshot)
    numeric = _extract_numeric_identifier(target_id)
    if numeric:
        if len(numeric) <= 6:
            return "sku"
        if "запрос" in context or "query" in context:
            return "query"
        if "кампан" in context or "advert" in context:
            return "campaign"
    if "запрос" in context or "query" in context:
        return "query"
    if "кампан" in context or "advert" in context:
        return "campaign"
    return "unknown"


def _infer_target_mode(
    task_snapshot: dict[str, Any],
    context: str,
    *,
    default: str,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> str:
    forced_target_mode = _forced_target_mode(planner_hints or [])
    if forced_target_mode:
        return forced_target_mode
    has_campaign = any(marker in context for marker in ("кампан", "advert"))
    has_query = any(marker in context for marker in ("запрос", "query", "search"))
    has_sku = any(marker in context for marker in ("sku", "артикул", "nm_id", "карточк", "остат", "продаж", "цен"))
    explicit = task_snapshot.get("targets") or []
    numeric_explicit = [str(item).strip() for item in explicit if _extract_numeric_identifier(str(item).strip())]
    short_numeric = [item for item in numeric_explicit if len(_extract_numeric_identifier(item) or "") <= 6]
    long_numeric = [item for item in numeric_explicit if len(_extract_numeric_identifier(item) or "") >= 7]
    if has_campaign and has_query:
        return "mixed" if short_numeric else "query"
    if has_query and has_sku:
        return "mixed"
    if has_campaign and has_sku:
        return "mixed"
    if has_campaign and short_numeric and long_numeric:
        return "mixed"
    if has_query and short_numeric and long_numeric:
        return "mixed"
    if has_campaign:
        return "campaign"
    if has_query:
        return "query"
    if has_sku or explicit:
        return "sku"
    return default


def _resolve_targets(task_snapshot: dict[str, Any], plan: ForecastPlan) -> list[str]:
    object_targets = _numeric_targets_from_objects(task_snapshot.get("objects"))
    if object_targets:
        return object_targets

    explicit_targets: list[str] = []
    for raw_target in task_snapshot.get("targets") or []:
        candidate = _string(raw_target)
        if not candidate or _is_target_noise(candidate):
            continue
        if candidate not in explicit_targets:
            explicit_targets.append(candidate)

    numeric_targets = [candidate for candidate in explicit_targets if _extract_numeric_identifier(candidate)]
    if plan.target_mode in {"sku", "campaign"}:
        if numeric_targets:
            return [_extract_numeric_identifier(candidate) for candidate in numeric_targets if _extract_numeric_identifier(candidate)]
        return []
    if plan.target_mode == "query":
        if numeric_targets:
            return [_extract_numeric_identifier(candidate) for candidate in numeric_targets if _extract_numeric_identifier(candidate)]
        return explicit_targets
    if plan.target_mode == "mixed":
        if numeric_targets:
            return [_extract_numeric_identifier(candidate) for candidate in numeric_targets if _extract_numeric_identifier(candidate)]
        return explicit_targets
    if explicit_targets:
        return explicit_targets

    targets: list[str] = []
    for match in _NUMERIC_TARGET_PATTERN.findall(_task_context(task_snapshot)):
        if match not in targets:
            targets.append(match)
    return targets


def _numeric_targets_from_objects(raw_objects: Any) -> list[str]:
    if not isinstance(raw_objects, list):
        return []
    targets: list[str] = []
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            continue
        payload = raw_object.get("payload") if isinstance(raw_object.get("payload"), dict) else {}
        for raw_candidate in (
            raw_object.get("externalId"),
            payload.get("nm_id"),
            raw_object.get("objectId"),
            raw_object.get("label"),
        ):
            candidate = _string(raw_candidate)
            if not candidate:
                continue
            normalized = _extract_numeric_identifier(candidate)
            if normalized and normalized not in targets:
                targets.append(normalized)
                break
    return targets


def _extract_numeric_identifier(value: str) -> str | None:
    stripped = value.strip()
    if stripped.isdigit():
        return stripped
    match = _NUMERIC_TARGET_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return None


def _is_target_noise(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "кампания",
        "кампании",
        "кампаний",
        "кампанию",
        "запрос",
        "запросы",
        "запросов",
        "видимость",
        "ставки",
        "ценовое",
        "цену",
        "цена",
        "лидеров",
        "лидеры",
        "план",
    }


def _explicit_primary_domain(task_snapshot: dict[str, Any], action_family: str) -> str:
    forecast_domain = _string(_forecast_spec(task_snapshot).get("forecast_domain"))
    if forecast_domain in {"marketing", "supply", "sales", "price", "finance", "catalog_demand"}:
        return forecast_domain
    mapping = {
        "stop_or_reduce_campaigns": "marketing",
        "availability_before_scale_guardrail": "supply",
        "weak_sku_demand_recovery": "sales",
        "sales_trend_guardrail": "sales",
        "price_positioning_guardrail": "price",
        "scale_profitable_sku_ads": "marketing",
        "reprice_with_margin_guardrail": "price",
        "reorder_for_growth": "supply",
        "overstock_liquidation": "supply",
        "normquery_bid_optimization": "marketing",
        "recover_conversion_drop": "sales",
    }
    return mapping.get(action_family, "sales")


def _secondary_domains_for_action_family(action_family: str) -> list[str]:
    mapping = {
        "stop_or_reduce_campaigns": ["sales"],
        "availability_before_scale_guardrail": ["marketing"],
        "weak_sku_demand_recovery": ["catalog_demand", "marketing", "supply", "price"],
        "sales_trend_guardrail": ["price", "supply"],
        "price_positioning_guardrail": ["sales", "finance"],
        "scale_profitable_sku_ads": ["sales", "supply", "price", "finance"],
        "reprice_with_margin_guardrail": ["sales", "marketing", "supply", "finance"],
        "reorder_for_growth": ["sales", "marketing", "finance"],
        "overstock_liquidation": ["sales", "price", "finance"],
        "normquery_bid_optimization": ["sales", "supply", "finance"],
        "recover_conversion_drop": ["catalog_demand", "marketing", "supply", "price", "finance"],
    }
    return list(mapping.get(action_family, []))


def _default_effect_metrics(action_family: str) -> list[str]:
    mapping = {
        "stop_or_reduce_campaigns": ["ad_spend_saved_rub", "orders_at_risk", "estimated_drr"],
        "availability_before_scale_guardrail": ["coverage_days", "oos_risk", "reorder_qty", "lost_sales_risk"],
        "weak_sku_demand_recovery": ["delta_sales", "delta_revenue", "trend_direction", "risk_level"],
        "sales_trend_guardrail": ["current_sales", "current_revenue", "delta_sales", "delta_revenue"],
        "price_positioning_guardrail": ["price_gap_rub", "price_gap_pct", "market_position_risk"],
        "scale_profitable_sku_ads": ["revenue_uplift", "ad_spend", "drr", "contribution_margin"],
        "reprice_with_margin_guardrail": ["price_gap_pct", "candidate_price", "margin_after_price"],
        "reorder_for_growth": ["reorder_qty", "coverage_days", "lost_sales_risk"],
        "overstock_liquidation": ["safe_discount", "stock_turnover", "margin_after_discount"],
        "normquery_bid_optimization": ["cpc", "ad_orders", "drr"],
        "recover_conversion_drop": ["conversion_recovery", "delta_revenue", "blockers"],
    }
    return list(mapping.get(action_family, []))


def _default_goal(action_family: str) -> str:
    mapping = {
        "stop_or_reduce_campaigns": "Снизить неэффективный рекламный расход без потери полезных заказов.",
        "availability_before_scale_guardrail": "Понять, выдержат ли остатки расширение рекламы без риска out-of-stock.",
        "weak_sku_demand_recovery": "Понять причину просадки по слабым SKU и выбрать следующий лучший шаг.",
        "sales_trend_guardrail": "Понять тренд продаж и риск дальнейшей просадки.",
        "price_positioning_guardrail": "Понять безопасное ценовое решение относительно рынка.",
        "scale_profitable_sku_ads": "Оценить growth-возможность и безопасность действия.",
        "reprice_with_margin_guardrail": "Оценить безопасное изменение цены с контролем маржи и спроса.",
        "reorder_for_growth": "Понять риск OOS и необходимость дозаказа.",
        "overstock_liquidation": "Снизить избыточный остаток без разрушения маржи.",
        "normquery_bid_optimization": "Понять, какие изменения по запросам и ставкам безопасно применить.",
        "recover_conversion_drop": "Найти и исправить барьер конверсии без ложного масштабирования.",
    }
    return mapping.get(action_family, "Оценить growth-возможность и безопасность действия.")


def _build_plan_from_family(
    *,
    task_snapshot: dict[str, Any],
    action_family: str,
    primary_domain: str,
    target_mode: str,
    source_action_kind: str | None,
    effect_metrics: list[str] | None = None,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> ForecastPlan:
    context = _task_context(task_snapshot)
    return ForecastPlan(
        primary_domain=primary_domain,
        action_family=action_family,
        target_mode=(
            target_mode
            if target_mode in {"sku", "campaign", "query", "mixed", "unknown"}
            else _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints)
        ),
        secondary_domains=_secondary_domains_for_action_family(action_family),
        horizon_days=_horizon_days(task_snapshot),
        period_days=_period_days(task_snapshot),
        effect_metrics=effect_metrics or _default_effect_metrics(action_family),
        goal=_string(task_snapshot.get("businessGoal")) or _default_goal(action_family),
        source_action_kind=source_action_kind,
    )


def _llm_route_plan(
    *,
    task_snapshot: dict[str, Any],
    action_kind: str | None,
    decision_type: str | None,
    effect_metrics: list[str],
    context: str,
    planner_hints: list[ForecastPlannerHint] | None = None,
) -> ForecastPlan | None:
    if not _should_try_llm_route(action_kind, decision_type):
        return None
    planner_hints = list(planner_hints or [])
    excluded_families = _excluded_action_families(planner_hints)
    candidates = [candidate for candidate in _router_candidates(task_snapshot, context) if candidate not in excluded_families]
    if len(candidates) == 1:
        return _build_plan_from_family(
            task_snapshot=task_snapshot,
            action_family=candidates[0],
            primary_domain=_explicit_primary_domain(task_snapshot, candidates[0]),
            target_mode=_forced_target_mode(planner_hints) or _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints),
            source_action_kind=action_kind,
            effect_metrics=effect_metrics,
            planner_hints=planner_hints,
        )
    route_result, _ = route_task_with_llm(task_snapshot=task_snapshot, candidate_families=candidates)
    if route_result is None or not route_result.route or not route_result.action_family:
        return None
    if candidates and route_result.action_family not in candidates:
        return None
    target_mode = (
        route_result.target_mode
        if route_result.target_mode in {"sku", "campaign", "query", "mixed", "unknown"}
        else _infer_target_mode(task_snapshot, context, default="sku", planner_hints=planner_hints)
    )
    primary_domain = route_result.primary_domain or _explicit_primary_domain(task_snapshot, route_result.action_family)
    return _build_plan_from_family(
        task_snapshot=task_snapshot,
        action_family=route_result.action_family,
        primary_domain=primary_domain,
        target_mode=_forced_target_mode(planner_hints) or target_mode,
        source_action_kind=action_kind,
        effect_metrics=effect_metrics,
        planner_hints=planner_hints,
    )


def _should_try_llm_route(action_kind: str | None, decision_type: str | None) -> bool:
    return (
        action_kind in {"optimize_marketing_visibility", "validate_unit_economics"}
        or decision_type in {"marketing_visibility_guardrail", "unit_economics_guardrail"}
    )


def _router_candidates(task_snapshot: dict[str, Any], context: str) -> list[str]:
    title = _string(task_snapshot.get("title")) or ""
    action = _string(task_snapshot.get("action")) or ""
    combined = f"{title} {action} {context}".lower()

    if any(marker in combined for marker in ("марж", "рентабель", "unit_economics")):
        return ["reprice_with_margin_guardrail", "price_positioning_guardrail"]
    if any(marker in combined for marker in ("ценов", "цена", "снижени")):
        return ["price_positioning_guardrail", "reprice_with_margin_guardrail"]
    if "перенести бюджет" in combined:
        return ["stop_or_reduce_campaigns", "normquery_bid_optimization"]
    if any(marker in combined for marker in ("запрос", "ставк", "видимост")):
        return ["normquery_bid_optimization"]
    if any(marker in combined for marker in ("останов", "сократ")) and any(
        marker in combined for marker in ("кампан", "advert", "реклам")
    ):
        return ["stop_or_reduce_campaigns"]
    if (
        "рост расход" in combined
        or ("расход" in combined and "кампан" in combined)
        or ("слаб" in combined and "кампан" in combined)
    ):
        return ["stop_or_reduce_campaigns", "normquery_bid_optimization"]
    if "готовность кампан" in combined:
        return ["normquery_bid_optimization", "stop_or_reduce_campaigns"]
    if "кампан" in combined and "запрос" in combined:
        return ["normquery_bid_optimization", "stop_or_reduce_campaigns"]
    return []


def _normalized_snapshot(task_snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task_snapshot)
    aliases = {
        "actionKind": "action_kind",
        "decisionType": "decision_type",
        "businessGoal": "business_goal",
        "requiredMetrics": "required_metrics",
        "forecastSpec": "forecast_spec",
        "executionPlan": "execution_plan",
        "agentPrompt": "agent_prompt",
    }
    for camel_key, snake_key in aliases.items():
        if normalized.get(camel_key) is None and normalized.get(snake_key) is not None:
            normalized[camel_key] = normalized.get(snake_key)
    return normalized


def _routing_context(task_snapshot: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in (
        "action",
        "title",
        "businessGoal",
        "decisionType",
        "actionKind",
        "factor",
        "observation",
        "impact",
        "cause",
    ):
        value = _string(task_snapshot.get(key))
        if value:
            fragments.append(value)
    return " ".join(fragments).lower()


def _deduplicate_targets(targets: list[ResolvedTarget]) -> list[ResolvedTarget]:
    seen: set[tuple[str, str]] = set()
    result: list[ResolvedTarget] = []
    for target in targets:
        key = (target.id, target.entity_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(target)
    return result


def _has_sku_object(raw_objects: Any) -> bool:
    if not isinstance(raw_objects, list):
        return False
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            continue
        object_type = _string(raw_object.get("type"))
        object_id = _string(raw_object.get("objectId"))
        payload = raw_object.get("payload") if isinstance(raw_object.get("payload"), dict) else {}
        if object_type == "sku":
            return True
        if object_id and object_id.startswith("sku-"):
            return True
        if payload.get("nm_id") is not None or payload.get("vendor_code") is not None:
            return True
    return False


def _forecast_spec(task_snapshot: dict[str, Any]) -> dict[str, Any]:
    forecast_spec = task_snapshot.get("forecastSpec")
    if not isinstance(forecast_spec, dict):
        forecast_spec = task_snapshot.get("forecast_spec")
    return forecast_spec if isinstance(forecast_spec, dict) else {}


def _horizon_days(task_snapshot: dict[str, Any]) -> int:
    return _positive_int(_forecast_spec(task_snapshot).get("horizon_days"), default=14)


def _period_days(task_snapshot: dict[str, Any]) -> int:
    return _positive_int(_forecast_spec(task_snapshot).get("period_days"), default=7)


def _effect_metrics(task_snapshot: dict[str, Any]) -> list[str]:
    value = _forecast_spec(task_snapshot).get("effect_metrics")
    if isinstance(value, list):
        metrics = [str(item).strip() for item in value if str(item).strip()]
        if metrics:
            return metrics
    return []


def _task_context(task_snapshot: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in (
        "action",
        "title",
        "businessGoal",
        "decisionType",
        "actionKind",
        "executionPlan",
        "agentPrompt",
        "factor",
        "observation",
        "impact",
        "cause",
    ):
        value = _string(task_snapshot.get(key))
        if value:
            fragments.append(value)
    for item in task_snapshot.get("evidence") or []:
        value = _string(item)
        if value:
            fragments.append(value)
    return " ".join(fragments).lower()


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, int):
        return str(value)
    return None


def _forced_action_family(planner_hints: list[ForecastPlannerHint]) -> str | None:
    for hint in planner_hints:
        if hint.forcedActionFamily:
            return hint.forcedActionFamily
    return None


def _forced_target_mode(planner_hints: list[ForecastPlannerHint]) -> str | None:
    for hint in planner_hints:
        if hint.forcedTargetMode:
            return hint.forcedTargetMode
    return None


def _forced_entity_type(planner_hints: list[ForecastPlannerHint]) -> str | None:
    for hint in planner_hints:
        if hint.forcedEntityType:
            return hint.forcedEntityType
    return None


def _excluded_action_families(planner_hints: list[ForecastPlannerHint]) -> set[str]:
    excluded: set[str] = set()
    for hint in planner_hints:
        excluded.update(hint.excludedActionFamilies)
    return excluded
