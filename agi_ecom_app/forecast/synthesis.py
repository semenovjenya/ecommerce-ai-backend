from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import (
    ForecastDecisionResult,
    ForecastDelta,
    ForecastHypothesis,
    ForecastPlan,
    ForecastReasoningResult,
    ForecastResolutionAction,
    ForecastResolutionPlanStep,
    ForecastScenario,
    ForecastUiCard,
    ForecastUiMetric,
    ResolvedTarget,
)
from agi_ecom_app.forecast.orchestration_policy import is_canonical_deterministic_plan


def build_forecast_decision(
    *,
    hypothesis: ForecastHypothesis,
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    scenario_output: dict[str, Any],
    reasoning_result: ForecastReasoningResult | None = None,
    reasoning_debug: dict[str, Any] | None = None,
) -> ForecastDecisionResult:
    deterministic_plan_items = _scenario_resolution_plan_items(scenario_output)
    canonical_deterministic_plan = is_canonical_deterministic_plan(plan, deterministic_plan_items)
    guarded_action_family = plan.action_family in {
        "overstock_liquidation",
        "reorder_for_growth",
        "reprice_with_margin_guardrail",
    }
    lock_calculator_output = canonical_deterministic_plan or guarded_action_family
    baseline = ForecastScenario(
        name="baseline",
        metrics=_clean_metrics(scenario_output.get("baseline_metrics")),
        assumptions=_assumptions(evidence_bundle, scenario_output),
    )
    action = ForecastScenario(
        name="action",
        metrics=_clean_metrics(scenario_output.get("action_metrics")),
        assumptions=_assumptions(evidence_bundle, scenario_output),
    )
    delta = ForecastDelta(metrics=_clean_metrics(scenario_output.get("delta_metrics")))
    blocked_by = _blocked_by(scenario_output, evidence_bundle)
    scenario_status = _decision_status(scenario_output.get("decision_status"))
    reasoning_status = _decision_status(reasoning_result.decision_status) if reasoning_result is not None else scenario_status
    decision_status = _more_restrictive_status(scenario_status, reasoning_status)
    if blocked_by and decision_status == "completed":
        decision_status = "needs_review"
    confidence = _confidence(
        reasoning_result.confidence if reasoning_result is not None else scenario_output.get("confidence")
    )
    display_reasoning_result = None if lock_calculator_output else reasoning_result
    ui_card = _build_ui_card(
        plan,
        blocked_by,
        scenario_output,
        reasoning_result=display_reasoning_result,
    )
    details = _build_forecast_details(
        plan=plan,
        scenario_output=scenario_output,
        resolved_targets=resolved_targets,
        baseline=baseline,
        action=action,
        delta=delta,
        ui_card=ui_card,
        reasoning_result=reasoning_result,
    )
    debug_data = {
        "plan": plan.model_dump(mode="json"),
        "resolved_targets": [target.model_dump(mode="json") for target in resolved_targets],
        "evidence_bundle": evidence_bundle.model_dump(mode="json"),
        "scenario_outputs": scenario_output,
        "canonical_plan_source": "deterministic" if canonical_deterministic_plan else "reasoning_or_fallback",
        "reasoning": reasoning_debug or {},
    }
    return ForecastDecisionResult(
        hypothesis=_hypothesis_text(hypothesis, plan),
        domain=hypothesis.domain,
        baseline=baseline,
        actionScenario=action,
        delta=delta,
        decision=_decision_text(
            scenario_output,
            ui_card,
            reasoning_result,
            allow_reasoning=not lock_calculator_output,
        ),
        decisionStatus=decision_status,
        confidence=confidence,
        confidencePercent=_confidence_percent(confidence, decision_status, blocked_by),
        profitForecastRub=_profit_forecast_rub(scenario_output),
        economicEffectRub=_economic_effect_rub(scenario_output),
        economicEffectPctOfCurrentSpend=_economic_effect_pct_of_current_spend(scenario_output),
        blockedBy=blocked_by,
        finalRecommendation=_final_recommendation_text(
            scenario_output,
            ui_card,
            reasoning_result,
            allow_reasoning=not lock_calculator_output,
        ),
        summary=_summary_text(
            scenario_output,
            ui_card,
            reasoning_result,
            allow_reasoning=not lock_calculator_output,
        ),
        resolutionActions=[
            ForecastResolutionAction.model_validate(item)
            for item in scenario_output.get("resolution_actions") or []
            if isinstance(item, dict) and item.get("area") and item.get("action")
        ],
        resolutionPlan=_resolution_plan_items(
            scenario_output,
            reasoning_result=reasoning_result,
            prefer_reasoning=not lock_calculator_output,
        ),
        plan=plan,
        resolvedTargets=resolved_targets,
        baselineNarrative=_string(scenario_output.get("baseline_narrative")) or ui_card.whatFound,
        actionNarrative=_string(scenario_output.get("action_narrative")) or ui_card.ifAction,
        effectNarrative=_string(scenario_output.get("effect_narrative")) or ui_card.expectedEffect,
        decisionWhy=_decision_why_text(scenario_output, ui_card, display_reasoning_result),
        uiCard=ui_card,
        forecastDetails=details,
        debugData=debug_data,
        evidence=evidence_bundle.model_dump(mode="json"),
    )


def _build_ui_card(
    plan: ForecastPlan,
    blocked_by: list[str],
    scenario_output: dict[str, Any],
    *,
    reasoning_result: ForecastReasoningResult | None = None,
) -> ForecastUiCard:
    card = ForecastUiCard(
        title="Прогноз эффекта",
        whatChecking=_human_checking_text(plan),
        whatFound=_string(scenario_output.get("baseline_narrative")) or "Факты по задаче собраны.",
        ifNoAction=_string(scenario_output.get("summary"))
        or _string(scenario_output.get("baseline_narrative"))
        or "Сценарий без действия не рассчитан.",
        ifAction=_string(scenario_output.get("action_narrative")) or "Сценарий действия не рассчитан.",
        expectedEffect=_string(scenario_output.get("card_expected_effect"))
        or _string(scenario_output.get("effect_narrative"))
        or "Ожидаемый эффект не рассчитан.",
        recommendation=_string(scenario_output.get("card_recommendation"))
        or _string(scenario_output.get("final_recommendation"))
        or _string(scenario_output.get("decision"))
        or "Рекомендация не сформирована.",
        confidenceReason=_confidence_reason(scenario_output, blocked_by),
        blockers=blocked_by,
        highlightMetrics=[
            ForecastUiMetric.model_validate(metric)
            for metric in scenario_output.get("highlight_metrics") or []
            if isinstance(metric, dict) and metric.get("label") and metric.get("value")
        ],
        nextChecks=_next_checks(blocked_by),
    )
    if reasoning_result is None or reasoning_result.ui_card_fields is None:
        return card
    fields = reasoning_result.ui_card_fields
    if fields.title:
        card.title = fields.title
    if fields.whatChecking:
        card.whatChecking = fields.whatChecking
    if fields.whatFound:
        card.whatFound = fields.whatFound
    if fields.ifNoAction:
        card.ifNoAction = fields.ifNoAction
    if fields.ifAction:
        card.ifAction = fields.ifAction
    if fields.expectedEffect:
        card.expectedEffect = fields.expectedEffect
    if fields.recommendation:
        card.recommendation = fields.recommendation
    if fields.confidenceReason:
        card.confidenceReason = fields.confidenceReason
    if fields.blockers:
        card.blockers = list(fields.blockers)
    if fields.nextChecks:
        card.nextChecks = list(fields.nextChecks)
    locked_effect = _string(scenario_output.get("card_expected_effect"))
    if locked_effect:
        card.expectedEffect = locked_effect
    locked_recommendation = _string(scenario_output.get("card_recommendation"))
    if locked_recommendation:
        card.recommendation = locked_recommendation
    return card


def _build_forecast_details(
    *,
    plan: ForecastPlan,
    scenario_output: dict[str, Any],
    resolved_targets: list[ResolvedTarget],
    baseline: ForecastScenario,
    action: ForecastScenario,
    delta: ForecastDelta,
    ui_card: ForecastUiCard,
    reasoning_result: ForecastReasoningResult | None = None,
) -> dict[str, Any]:
    details = {
        "headline": _hypothesis_label(plan),
        "hypothesis": _string(reasoning_result.decision if reasoning_result is not None else scenario_output.get("decision"))
        or ui_card.recommendation,
        "baseline": _metric_details(baseline.metrics),
        "action": _metric_details(action.metrics),
        "delta": _metric_details(delta.metrics),
        "perTargetEvidence": scenario_output.get("per_target_evidence") or [],
        "marginBridge": scenario_output.get("margin_bridge") or {},
        "opportunityScore": scenario_output.get("opportunity_score"),
        "growthReason": _string(scenario_output.get("growth_reason")),
        "safetyBlockers": scenario_output.get("safety_blockers") or [],
        "recommendedActionFamily": scenario_output.get("recommended_action_family") or plan.action_family,
        "marketingDecisionHint": _string(scenario_output.get("marketing_decision_hint")),
        "marketingConfidence": _string(scenario_output.get("marketing_confidence")),
        "safeGrowthPct": scenario_output.get("safe_growth_pct"),
        "winnerQueryShare": scenario_output.get("winner_query_share"),
        "wasteQueryShare": scenario_output.get("waste_query_share"),
        "queryBand": _string(scenario_output.get("query_band")),
        "marketingSourceStatus": _string(scenario_output.get("marketing_source_status")),
        "confidenceWhy": ui_card.confidenceReason,
        "whatElseToCheck": list(ui_card.nextChecks),
        "resolvedTargets": [target.model_dump(mode="json") for target in resolved_targets],
    }
    if reasoning_result is not None:
        details.setdefault("reasoningPrimaryAction", reasoning_result.primary_action)
        details.setdefault("reasoningRejectedActions", list(reasoning_result.rejected_actions))
        details.setdefault("reasoningEvidenceUsed", list(reasoning_result.evidence_used))
        details.setdefault("reasoningWarnings", list(reasoning_result.warnings))
        details.update(reasoning_result.forecast_details_patch)
    details["canonicalPlanSource"] = (
        "deterministic"
        if is_canonical_deterministic_plan(plan, _scenario_resolution_plan_items(scenario_output))
        else "reasoning_or_fallback"
    )
    return details


def _decision_text(
    scenario_output: dict[str, Any],
    ui_card: ForecastUiCard,
    reasoning_result: ForecastReasoningResult | None,
    *,
    allow_reasoning: bool,
) -> str:
    if allow_reasoning and reasoning_result is not None and reasoning_result.decision:
        return reasoning_result.decision
    return str(scenario_output.get("decision") or ui_card.recommendation)


def _final_recommendation_text(
    scenario_output: dict[str, Any],
    ui_card: ForecastUiCard,
    reasoning_result: ForecastReasoningResult | None,
    *,
    allow_reasoning: bool,
) -> str:
    if allow_reasoning and reasoning_result is not None and reasoning_result.final_recommendation:
        return reasoning_result.final_recommendation
    locked_recommendation = _string(scenario_output.get("card_recommendation"))
    if locked_recommendation:
        return locked_recommendation
    return str(scenario_output.get("final_recommendation") or ui_card.recommendation)


def _summary_text(
    scenario_output: dict[str, Any],
    ui_card: ForecastUiCard,
    reasoning_result: ForecastReasoningResult | None,
    *,
    allow_reasoning: bool,
) -> str:
    if allow_reasoning and reasoning_result is not None and reasoning_result.summary:
        return reasoning_result.summary
    locked_summary = _string(scenario_output.get("card_summary"))
    if locked_summary:
        return locked_summary
    return str(
        scenario_output.get("summary")
        or scenario_output.get("final_recommendation")
        or scenario_output.get("decision")
        or ui_card.recommendation
        or ui_card.expectedEffect
    )


def _decision_why_text(
    scenario_output: dict[str, Any],
    ui_card: ForecastUiCard,
    reasoning_result: ForecastReasoningResult | None,
) -> str | None:
    if reasoning_result is not None and reasoning_result.decision_why:
        return reasoning_result.decision_why
    return _string(scenario_output.get("decision_why")) or ui_card.confidenceReason


def _human_checking_text(plan: ForecastPlan) -> str:
    mapping = {
        "stop_or_reduce_campaigns": "Проверяем, какие рекламные кампании реально можно сократить без лишней потери заказов.",
        "availability_before_scale_guardrail": "Проверяем, выдержат ли остатки и поставки план по спросу или расширению рекламы.",
        "weak_sku_demand_recovery": "Проверяем, из-за чего именно проседают слабые SKU и какой следующий шаг даст смысл для бизнеса.",
        "sales_trend_guardrail": "Проверяем, продолжается ли просадка продаж и нужен ли cross-domain разбор.",
        "price_positioning_guardrail": "Проверяем, безопасна ли текущая ценовая позиция относительно рынка.",
        "scale_profitable_sku_ads": "Проверяем, можно ли масштабировать прибыльные SKU без риска для маржи, цены и остатков.",
        "reprice_with_margin_guardrail": "Проверяем, безопасна ли переоценка через конкурентный коридор и маржу.",
        "reorder_for_growth": "Проверяем, где дозаказ защищает рост от out-of-stock.",
        "overstock_liquidation": "Проверяем, где зависший запас можно распродавать без сжигания маржи.",
        "normquery_bid_optimization": "Проверяем ставки и normquery как управляемый источник роста.",
        "recover_conversion_drop": "Проверяем просадку конверсии и лучший следующий шаг восстановления спроса.",
    }
    return mapping.get(plan.action_family, "Проверяем ожидаемый эффект по задаче.")


def _hypothesis_label(plan: ForecastPlan) -> str:
    mapping = {
        "stop_or_reduce_campaigns": "Сократить или остановить слабые кампании",
        "availability_before_scale_guardrail": "Сверить доступность перед ростом спроса",
        "weak_sku_demand_recovery": "Найти причину просадки и следующий шаг по слабым SKU",
        "sales_trend_guardrail": "Оценить тренд продаж и guardrail",
        "price_positioning_guardrail": "Оценить безопасное решение по цене",
        "scale_profitable_sku_ads": "Масштабировать прибыльный SKU",
        "reprice_with_margin_guardrail": "Переоценка с маржинальным guardrail",
        "reorder_for_growth": "Дозаказ для защиты роста",
        "overstock_liquidation": "Безопасная распродажа overstock",
        "normquery_bid_optimization": "Оптимизация ставок и normquery",
        "recover_conversion_drop": "Восстановить просевшую конверсию",
    }
    return mapping.get(plan.action_family, plan.action_family)


def _hypothesis_text(hypothesis: ForecastHypothesis, plan: ForecastPlan) -> str:
    targets = ", ".join(hypothesis.targets) if hypothesis.targets else "объекты не определены"
    return f"{_hypothesis_label(plan)}: {targets}"


def _metric_details(metrics: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"label": _metric_label(key), "value": _normalize_value(value)}
        for key, value in metrics.items()
        if _is_meaningful(value)
    ]


def _clean_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {key: value for key, value in metrics.items() if _is_meaningful(value)}


def _profit_forecast_rub(scenario_output: dict[str, Any]) -> float | None:
    delta_metrics = scenario_output.get("delta_metrics") or {}
    if not isinstance(delta_metrics, dict):
        return None
    for key in ("estimated_profit_recovery_rub", "expected_net_profit_delta", "net_profit_delta"):
        value = _as_float(delta_metrics.get(key))
        if value is not None:
            return round(value, 2)
    return None


def _economic_effect_rub(scenario_output: dict[str, Any]) -> float | None:
    profit_value = _profit_forecast_rub(scenario_output)
    if profit_value is not None:
        return profit_value
    delta_metrics = scenario_output.get("delta_metrics") or {}
    if not isinstance(delta_metrics, dict):
        return None
    for key in ("ad_spend_saved_rub", "estimated_revenue_recovery", "expected_net_profit_7d"):
        value = _as_float(delta_metrics.get(key))
        if value is not None:
            return round(value, 2)
    return None


def _economic_effect_pct_of_current_spend(scenario_output: dict[str, Any]) -> float | None:
    effect_value = _economic_effect_rub(scenario_output)
    if effect_value is None:
        return None
    current_spend = _current_spend_rub(scenario_output)
    if current_spend is None or current_spend <= 0:
        return None
    return round(effect_value * 100 / current_spend, 2)


def _current_spend_rub(scenario_output: dict[str, Any]) -> float | None:
    baseline_metrics = scenario_output.get("baseline_metrics") or {}
    if isinstance(baseline_metrics, dict):
        value = _as_float(baseline_metrics.get("current_spend_rub"))
        if value is not None:
            return value
    action_metrics = scenario_output.get("action_metrics") or {}
    if isinstance(action_metrics, dict):
        value = _as_float(action_metrics.get("current_spend_rub"))
        if value is not None:
            return value
    delta_metrics = scenario_output.get("delta_metrics") or {}
    if isinstance(delta_metrics, dict):
        value = _as_float(delta_metrics.get("current_spend_rub"))
        if value is not None:
            return value
    return None


def _assumptions(evidence_bundle: ForecastEvidenceBundle, scenario_output: dict[str, Any]) -> list[str]:
    assumptions = [item for item in scenario_output.get("assumptions") or [] if isinstance(item, str) and item.strip()]
    assumptions.extend(
        _string(item)
        for item in evidence_bundle.diagnostic.get("blocking_gaps") or []
        if _string(item)
    )
    result: list[str] = []
    for item in assumptions:
        if item and item not in result:
            result.append(item)
    return result


def _resolution_plan_items(
    scenario_output: dict[str, Any],
    *,
    reasoning_result: ForecastReasoningResult | None = None,
    prefer_reasoning: bool = True,
) -> list[ForecastResolutionPlanStep]:
    if prefer_reasoning and reasoning_result is not None and reasoning_result.resolution_plan_patch:
        return [ForecastResolutionPlanStep.model_validate(item) for item in reasoning_result.resolution_plan_patch]
    explicit_items = _scenario_resolution_plan_items(scenario_output)
    if explicit_items:
        return explicit_items

    fallback_items: list[ForecastResolutionPlanStep] = []
    for index, item in enumerate(scenario_output.get("resolution_actions") or [], start=1):
        if not isinstance(item, dict):
            continue
        area = _string(item.get("area")) or f"шаг_{index}"
        action = _string(item.get("action"))
        if not action:
            continue
        fallback_items.append(
            ForecastResolutionPlanStep(
                stepId=f"шаг-{index}",
                area=area,
                operation=_fallback_operation(area),
                instruction=action,
                targetMetric=_string(item.get("expectedMetric")),
            )
        )
    if fallback_items:
        return fallback_items

    recommendation = _string(scenario_output.get("final_recommendation")) or _string(scenario_output.get("summary"))
    if recommendation:
        return [
            ForecastResolutionPlanStep(
                stepId="шаг-1",
                area="decision",
                operation="выполнить_рекомендацию",
                instruction=recommendation,
            )
        ]
    return []


def _scenario_resolution_plan_items(
    scenario_output: dict[str, Any],
) -> list[ForecastResolutionPlanStep]:
    return [
        ForecastResolutionPlanStep.model_validate(item)
        for item in scenario_output.get("resolution_plan") or []
        if isinstance(item, dict) and item.get("stepId") and item.get("operation") and item.get("instruction")
    ]


def _blocked_by(scenario_output: dict[str, Any], evidence_bundle: ForecastEvidenceBundle) -> list[str]:
    result: list[str] = []
    source_blockers = (
        list(scenario_output.get("blocked_by") or [])
        if "blocked_by" in scenario_output
        else list(evidence_bundle.blocking_gaps)
    )
    for item in source_blockers:
        text = _human_blocker(item)
        if text and text not in result:
            result.append(text)
    return result


def _confidence_reason(scenario_output: dict[str, Any], blocked_by: list[str]) -> str:
    confidence = _confidence(scenario_output.get("confidence"))
    if confidence == "high":
        return "Уверенность высокая: по ключевым объектам нашлись факты и сценарий опирается на реальные данные по текущему срезу."
    if blocked_by:
        return f"Уверенность не максимальная: часть решения держится на guardrail-логике, а не на полном наборе фактов ({blocked_by[0]})."
    return "Уверенность средняя: данных достаточно для сценария, но не для точного causal прогноза."


def _next_checks(blocked_by: list[str]) -> list[str]:
    checks: list[str] = []
    for blocker in blocked_by:
        lowered = blocker.lower()
        if "цен" in lowered or "price" in lowered:
            checks.append("Досмотреть ценовой коридор и маржу по целевым SKU.")
        elif "остат" in lowered or "supply" in lowered or "oos" in lowered:
            checks.append("Пересчитать остатки, товар в пути и покрытие в днях.")
        elif "marketing demand mart" in lowered or "marketing_low_confidence" in lowered or "test-only" in lowered:
            checks.append("Проверить marketing marts, confidence и не расширять трафик без demand-подтверждения.")
        elif "campaign" in lowered or "кампан" in lowered:
            checks.append("Подтянуть campaign-level spend/orders и action matrix по нужным кампаниям.")
        else:
            checks.append("Добрать недостающие данные по целевым объектам и повторить прогноз.")
    return checks[:3]


def _metric_label(value: str) -> str:
    labels = {
        "campaigns_checked": "Кампаний в проверке",
        "current_spend_rub": "Текущий расход",
        "current_ad_orders": "Рекламные заказы",
        "estimated_drr": "ДРР",
        "campaigns_to_reduce": "К сокращению",
        "campaigns_to_hold": "Под контролем",
        "orders_at_risk": "Заказы под риском",
        "ad_spend_saved_rub": "Экономия расхода",
        "available_stock": "Доступный остаток",
        "inbound_stock": "В пути",
        "sales_velocity": "Скорость продаж",
        "coverage_days": "Покрытие, дней",
        "oos_risk": "Риск OOS",
        "reorder_qty": "Дозаказ",
        "days_until_oos": "Дней до OOS",
        "lost_sales_risk": "Риск потери продаж",
        "current_sales": "Текущие продажи",
        "previous_sales": "Продажи в прошлом периоде",
        "current_revenue": "Текущая выручка",
        "previous_revenue": "Выручка в прошлом периоде",
        "delta_sales": "Изменение продаж",
        "delta_revenue": "Изменение выручки",
        "trend_direction": "Тренд",
        "risk_level": "Уровень риска",
        "recommended_action": "Следующий шаг",
        "current_price": "Текущая цена",
        "competitor_median_price": "Медиана конкурентов",
        "price_gap_rub": "Разрыв к рынку",
        "price_gap_pct": "Разрыв, %",
        "competitor_count": "Конкурентов",
        "price_status": "Статус цены",
        "market_position_risk": "Рыночный риск",
        "price_guardrail": "Guardrail по цене",
        "test_ad_budget_rub": "Тестовый бюджет",
        "growth_targets": "Цели роста",
        "estimated_revenue_uplift": "Оценка роста выручки",
        "margin_guardrail": "Guardrail по марже",
        "candidate_price": "Кандидат цены",
        "safe_discount_pct": "Безопасная скидка",
        "stock_turnover_action": "Действие по запасу",
        "bid_optimization": "Оптимизация ставок",
        "estimated_revenue_recovery": "Оценка восстановления выручки",
        "contribution_margin": "Contribution margin",
    }
    return labels.get(value, value.replace("_", " "))


def _human_blocker(value: Any) -> str | None:
    raw = _string(value)
    if not raw:
        return None
    for prefix, text in {
        "marketing_demand_mart_unpopulated": "Demand marketing marts не populated, поэтому growth-сигнал остается только review/test.",
        "marketing_low_confidence": "Marketing mart помечает решение как low confidence.",
        "marketing_test_only": "Marketing signal разрешает только тест, а не auto-scale.",
        "marketing_mart_unpopulated": "Marketing mart не готов или не populated.",
    }.items():
        if raw == prefix or raw.startswith(f"{prefix}:"):
            return text
    mapping = {
        "campaign_target_not_found": "Нет campaign-level фактов по части объектов.",
        "no_confirmed_weak_campaigns": "Слабые кампании не подтверждены по текущему срезу.",
        "missing_supply_snapshot": "Нет полного supply-среза по целевым SKU.",
        "missing_sales_snapshot": "Нет полного sales-среза по целевым SKU.",
        "missing_price_snapshot": "Нет конкурентного ценового среза по целевым SKU.",
        "thin_competitor_data": "Данных по конкурентам недостаточно для уверенного price guardrail.",
        "missing_weak_sku_evidence": "Не хватает фактов, чтобы выбрать главный драйвер просадки по SKU.",
        "missing_finance_evidence": "Нет finance evidence для безопасного growth-действия.",
        "missing_cogs": "Себестоимость покрыта не полностью.",
        "supply_risk": "Есть риск доступного остатка или покрытия.",
        "thin_price_data": "Конкурентный price-срез тонкий или неполный.",
        "low_price_decision_confidence": "Price decision mart сам помечает решение как low confidence.",
        "stale_price_decision": "Price decision опирается на устаревший price AI слой.",
        "future_price_decision": "Price decision датирован будущим числом и не может использоваться.",
        "invalid_price_freshness": "Не удалось подтвердить дату ценового расчёта.",
        "reorder_not_supported_by_current_supply": "Текущий остаток и покрытие не подтверждают необходимость дозаказа.",
        "negative_margin": "Маржа не подтверждает growth-действие.",
    }
    return mapping.get(raw, raw.replace("_", " "))


def _normalize_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "0", "0.0", "none", "unknown", "нет", "—"}
    return True


def _decision_status(value: Any) -> str:
    if value in {"completed", "needs_review", "blocked"}:
        return str(value)
    return "needs_review"


def _more_restrictive_status(left: str, right: str) -> str:
    priority = {"completed": 0, "needs_review": 1, "blocked": 2}
    return left if priority[left] >= priority[right] else right


def _confidence(value: Any) -> str:
    if value in {"low", "medium", "high"}:
        return str(value)
    return "medium"


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", ""))
    except (TypeError, ValueError):
        return None


def _fallback_operation(area: str) -> str:
    mapping = {
        "supply": "действие_по_остатку",
        "price": "действие_по_цене",
        "guardrail": "ограничить_масштабирование",
        "marketing_guardrail": "ограничить_масштабирование",
        "waste_traffic": "отключить_нецелевой_трафик",
        "traffic_only": "удерживать_запрос",
        "ctr": "обновить_заголовок",
        "atc": "обновить_оффер",
        "positions": "зафиксировать_рабочее_ядро",
    }
    return mapping.get(area, "выполнить_действие")


def _confidence_percent(
    confidence: str,
    decision_status: str,
    blocked_by: list[str],
) -> int:
    base = {
        "high": 82,
        "medium": 64,
        "low": 38,
    }.get(confidence, 64)
    if decision_status == "blocked":
        base -= 15
    elif decision_status == "needs_review":
        base -= 8
    base -= min(len(blocked_by) * 6, 18)
    return max(10, min(95, base))
