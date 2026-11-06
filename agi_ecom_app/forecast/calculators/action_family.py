from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import ForecastPlan, ResolvedTarget
from agi_ecom_app.services import strategic_db

_WEAK_STATUSES = {"inefficient", "negative_outlier"}
_ACTION_REDUCE_STATUSES = {"остановить", "сократить"}
_SUPPLIER_INBOUND_KEYS = (
    "supplier_inbound_qty",
    "supplier_inbound_stock",
    "confirmed_supplier_inbound_qty",
)


def calculate_forecast_scenario(
    *,
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    today: date | None = None,
) -> dict[str, Any]:
    if plan.action_family == "stop_or_reduce_campaigns":
        return _stop_or_reduce_campaigns(plan, resolved_targets, evidence_bundle, today=today)
    if plan.action_family == "availability_before_scale_guardrail":
        return _availability_before_scale_guardrail(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "weak_sku_demand_recovery":
        return _weak_sku_demand_recovery(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "sales_trend_guardrail":
        return _sales_trend_guardrail(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "price_positioning_guardrail":
        return _price_positioning_guardrail(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "scale_profitable_sku_ads":
        return _scale_profitable_sku_ads(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "reprice_with_margin_guardrail":
        return _reprice_with_margin_guardrail(plan, resolved_targets, evidence_bundle, today=today)
    if plan.action_family == "reorder_for_growth":
        return _reorder_for_growth(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "overstock_liquidation":
        return _overstock_liquidation(plan, resolved_targets, evidence_bundle, today=today)
    if plan.action_family == "normquery_bid_optimization":
        return _normquery_bid_optimization(plan, resolved_targets, evidence_bundle)
    if plan.action_family == "recover_conversion_drop":
        return _recover_conversion_drop(plan, resolved_targets, evidence_bundle)
    return {
        "decision_status": "blocked",
        "confidence": "low",
        "blocked_by": ["unsupported_forecast_action_family"],
        "baseline_metrics": {},
        "action_metrics": {},
        "delta_metrics": {},
        "baseline_narrative": "Подходящий прогнозный сценарий не найден.",
        "action_narrative": "Действие не рассчитано.",
        "effect_narrative": "Бизнес-эффект пока не оценен.",
        "decision": "Прогнозный сценарий не рассчитан.",
        "decision_why": "Для этого типа задачи нет action-family calculator.",
        "final_recommendation": "Вернуться к legacy action execution или добавить calculator для новой action family.",
        "summary": "Forecast vNext не применился.",
        "per_target_evidence": [],
        "highlight_metrics": [],
    }


def _stop_or_reduce_campaigns(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    *,
    today: date | None,
) -> dict[str, Any]:
    target_rows: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    for target in resolved_targets:
        marketing = _target_match(evidence_bundle, "forecast_marketing_agent", target.id)
        campaign_row = marketing.get("campaign_row") or {}
        action_row = marketing.get("campaign_action") or {}
        if not campaign_row and not action_row:
            missing_targets.append(target.id)
            continue
        spend = _number(campaign_row.get("sum_spend") or campaign_row.get("spend_rub"))
        orders = _number(campaign_row.get("orders") or campaign_row.get("ad_orders"))
        status = _string(campaign_row.get("status"))
        matrix_status = _string(action_row.get("status"))
        weak_by_matrix = matrix_status in _ACTION_REDUCE_STATUSES
        weak_by_efficiency = status in _WEAK_STATUSES
        decision = "stop_or_reduce" if weak_by_matrix or weak_by_efficiency else "hold"
        target_rows.append(
            {
                "target": target.id,
                "spend": spend or 0.0,
                "orders": orders or 0.0,
                "status": status or matrix_status or "unknown",
                "decision": decision,
                "why": _campaign_reason(weak_by_matrix, weak_by_efficiency, spend, orders),
                "marketing_decision_hint": _string(marketing.get("decision_hint") or action_row.get("decision_hint")),
                "marketing_confidence": _string(marketing.get("marketing_confidence") or action_row.get("confidence")),
                "safe_growth_pct": _number(marketing.get("safe_growth_pct") or action_row.get("safe_growth_pct")),
                "winner_query_share": _number(marketing.get("winner_query_share") or action_row.get("winner_query_share")),
                "waste_query_share": _number(marketing.get("waste_query_share") or action_row.get("waste_query_share")),
                "marketing_source_status": _string(marketing.get("marketing_source_status") or action_row.get("marketing_source_status")),
            }
        )

    total_spend = round(sum(item["spend"] for item in target_rows), 2)
    spend_saved = round(sum(item["spend"] for item in target_rows if item["decision"] == "stop_or_reduce"), 2)
    orders_at_risk = round(sum(item["orders"] for item in target_rows if item["decision"] == "stop_or_reduce"), 2)
    reduce_targets = [item["target"] for item in target_rows if item["decision"] == "stop_or_reduce"]
    hold_targets = [item["target"] for item in target_rows if item["decision"] == "hold"]
    drr_signal = _load_period_drr(today=today, period_days=plan.period_days)
    blocked_by = list(evidence_bundle.blocking_gaps)
    if not resolved_targets:
        blocked_by.append("missing_campaign_targets")
    if missing_targets:
        blocked_by.append("campaign_target_not_found")
    if not reduce_targets:
        blocked_by.append("no_confirmed_weak_campaigns")
    confidence = "high" if reduce_targets and not missing_targets else "medium" if reduce_targets else "medium"
    decision_status = "completed" if reduce_targets else "needs_review"
    first_marketing_row = next((item for item in target_rows if isinstance(item, dict)), {})
    what_found = _marketing_findings_text(reduce_targets, hold_targets, missing_targets)
    effect_text = (
        f"Если сократить подтвержденные кампании, можно высвободить около {spend_saved:.0f} ₽ за {plan.period_days} дней; "
        f"под риском остаётся около {orders_at_risk:.0f} рекламных заказов."
        if reduce_targets
        else "По доступным фактам слабые кампании не подтверждены; резкое сокращение расходов сейчас создаст риск ложного решения."
    )
    recommendation = (
        f"В кампаниях {', '.join(reduce_targets)} сократить расход или остановить показ; {', '.join(hold_targets) or 'остальные'} не расширять без нового сигнала по заказам."
        if reduce_targets
        else "Не останавливать кампании автоматически; сначала подтвердить слабость по spend, orders и campaign-level статусу."
    )
    resolution_actions = _campaign_resolution_actions(target_rows)
    resolution_plan = _campaign_resolution_plan(target_rows)
    return {
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "campaigns_checked": len(resolved_targets),
            "current_spend_rub": total_spend,
            "current_ad_orders": round(sum(item["orders"] for item in target_rows), 2),
            "estimated_drr": f"{drr_signal:.2f}%" if drr_signal is not None else None,
        },
        "action_metrics": {
            "campaigns_to_reduce": ", ".join(reduce_targets) or None,
            "campaigns_to_hold": ", ".join(hold_targets) or None,
            "orders_at_risk": orders_at_risk if orders_at_risk > 0 else None,
        },
        "delta_metrics": {
            "ad_spend_saved_rub": spend_saved if spend_saved > 0 else None,
            "orders_at_risk": orders_at_risk if orders_at_risk > 0 else None,
            "drr_signal": "расход снизится быстрее заказов" if reduce_targets else None,
        },
        "baseline_narrative": f"Сейчас под контролем {len(target_rows)} кампаний с расходом {total_spend:.0f} ₽ за {plan.period_days} дней.",
        "action_narrative": what_found,
        "effect_narrative": effect_text,
        "decision": recommendation,
        "decision_why": "Маркетинговый сценарий считает только campaign-level stop/reduce и не подменяет SKU-задачи рекламной логикой.",
        "final_recommendation": recommendation,
        "summary": recommendation,
        "card_summary": recommendation,
        "card_recommendation": recommendation,
        "card_expected_effect": effect_text,
        "resolution_actions": resolution_actions,
        "resolution_plan": resolution_plan,
        "per_target_evidence": target_rows,
        "marketing_decision_hint": _string(first_marketing_row.get("marketing_decision_hint")),
        "marketing_confidence": _string(first_marketing_row.get("marketing_confidence")),
        "safe_growth_pct": _number(first_marketing_row.get("safe_growth_pct")),
        "winner_query_share": _number(first_marketing_row.get("winner_query_share")),
        "waste_query_share": _number(first_marketing_row.get("waste_query_share")),
        "marketing_source_status": _string(first_marketing_row.get("marketing_source_status")),
        "highlight_metrics": _metric_cards(
            [
                ("Экономия расхода", _money(spend_saved) if spend_saved > 0 else None),
                ("Заказы под риском", _plain_number(orders_at_risk) if orders_at_risk > 0 else None),
                ("Под контролем", _plain_number(len(hold_targets)) if hold_targets else None),
            ]
        ),
    }


def _availability_before_scale_guardrail(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in resolved_targets:
        supply = _target_match(evidence_bundle, "forecast_supply_agent", target.id).get("supply_row") or {}
        marketing = _target_match(evidence_bundle, "forecast_marketing_agent", target.id)
        if not supply:
            continue
        stock = _number(supply.get("stock_qty"))
        if stock is None:
            stock = _number(supply.get("quantity"))
        inbound = _supplier_inbound(supply)
        sales_7d = _number(supply.get("sales_7d_units"))
        velocity = round((sales_7d or 0) / 7, 2) if sales_7d else 0
        coverage = _number(supply.get("days_of_stock"))
        reorder_qty = _number(supply.get("reorder_qty_recommended")) or 0
        reorder_flag = _string(supply.get("reorder_flag")) or "unknown"
        rows.append(
            {
                "target": target.id,
                "available_stock": stock,
                "inbound_stock": inbound,
                "supplier_inbound_confirmed": inbound is not None,
                "customer_transit_to_client": _number(supply.get("in_way_to_client")),
                "customer_transit_from_client": _number(supply.get("in_way_from_client")),
                "sales_velocity": velocity,
                "coverage_days": coverage,
                "reorder_qty": reorder_qty,
                "oos_risk": _oos_risk(reorder_flag, coverage, velocity),
                "query_rows": marketing.get("query_rows") or [],
                "query_action": marketing.get("query_action") or {},
                "query_row": marketing.get("query_row") or {},
                "visibility_row": marketing.get("visibility_row") or {},
            }
        )
    blocked_by = list(evidence_bundle.blocking_gaps)
    if not rows:
        blocked_by.append("missing_supply_snapshot")
    total_reorder = round(sum(item["reorder_qty"] for item in rows), 2)
    reorder_targets = [item for item in rows if (_number(item.get("reorder_qty")) or 0) > 0]
    supplier_inbound_confirmed = bool(reorder_targets) and all(
        item.get("supplier_inbound_confirmed") for item in reorder_targets
    )
    if reorder_targets and not supplier_inbound_confirmed:
        blocked_by.append("missing_supplier_inbound_snapshot")
    min_coverage = _min_number(item["coverage_days"] for item in rows)
    aggregate_risk = _aggregate_risk([str(item["oos_risk"]) for item in rows])
    priority = [item["target"] for item in rows if item["oos_risk"] in {"high", "medium"}]
    lead_row = rows[0] if rows else {}
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    decision_status = (
        "blocked"
        if not rows
        else "needs_review"
        if reorder_targets and not supplier_inbound_confirmed
        else "completed"
    )
    confidence = (
        "high"
        if rows and min_coverage is not None and decision_status == "completed"
        else "medium"
        if rows
        else "low"
    )
    what_found = (
        f"Часть SKU уже подходит к риску OOS: {', '.join(priority)}."
        if priority
        else "Критичных SKU по покрытию не видно, но guardrail по остаткам всё равно нужен перед масштабированием рекламы."
    )
    if_no_action = (
        f"Если расширять рекламу без сверки остатков, покрытие может упасть до {min_coverage:.0f} дней и часть SKU уйдёт в OOS."
        if min_coverage is not None
        else "Если масштабировать спрос без supply-фактов, риск out-of-stock остаётся неопределённым."
    )
    if_action = (
        (
            f"Консервативный черновик дозаказа — около {total_reorder:.0f} шт. для SKU "
            f"{', '.join(priority) or 'под контролем'}; сначала нужно вычесть подтверждённые входящие поставки поставщика."
            if not supplier_inbound_confirmed
            else f"Если сначала выровнять supply, приоритет на дозаказ около {total_reorder:.0f} шт. для SKU {', '.join(priority) or 'под контролем'}."
        )
        if total_reorder > 0
        else "Если оставить рекламу под контролем и не масштабировать её сразу, можно избежать OOS до новой supply-сверки."
    )
    recommendation = (
        (
            f"Не расширять рекламу; сначала сверить входящие поставки поставщика, затем пересчитать черновик дозаказа {total_reorder:.0f} шт."
            if not supplier_inbound_confirmed
            else f"Не расширять рекламу, пока не закрыт supply-риск; дозаказать около {total_reorder:.0f} шт. и пересчитать coverage."
        )
        if priority or total_reorder > 0
        else _availability_campaign_freeze_action_text(
            lead_target=str(lead_row.get("target") or "SKU"),
            advert_id=advert_id,
            query_rows=traffic_queries or waste_queries,
            min_coverage=min_coverage,
        )
    )
    resolution_actions = _availability_resolution_actions(
        rows,
        priority,
        total_reorder,
        min_coverage,
        reorder_executable=supplier_inbound_confirmed,
    )
    resolution_plan = _availability_resolution_plan(
        rows,
        priority,
        total_reorder,
        min_coverage,
        reorder_executable=supplier_inbound_confirmed,
    )
    return {
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "available_stock": round(sum(_number(item["available_stock"]) or 0 for item in rows), 2) if rows else None,
            "inbound_stock": (
                round(sum(_number(item["inbound_stock"]) or 0 for item in rows), 2)
                if rows and all(item.get("supplier_inbound_confirmed") for item in rows)
                else None
            ),
            "sales_velocity": round(sum(item["sales_velocity"] for item in rows), 2) if rows else None,
            "coverage_days": min_coverage,
            "oos_risk": aggregate_risk if rows else None,
        },
        "action_metrics": {
            "recommended_action": recommendation,
            "priority_targets": ", ".join(priority) or None,
            "reorder_qty": total_reorder if total_reorder > 0 else None,
        },
        "delta_metrics": {
            "reorder_qty": total_reorder if total_reorder > 0 else None,
            "days_until_oos": min_coverage,
            "lost_sales_risk": aggregate_risk if rows else None,
        },
        "baseline_narrative": f"Supply-срез показывает риск {aggregate_risk or 'unknown'} на горизонте {plan.horizon_days} дней.",
        "action_narrative": if_action,
        "effect_narrative": "Главный эффект здесь не uplift, а защита продаж от OOS и срыва рекламы на пустом остатке.",
        "decision": recommendation,
        "decision_why": "Supply guardrail решает, можно ли вообще усиливать спрос, а не обещает рост продаж сам по себе.",
        "final_recommendation": recommendation,
        "summary": if_no_action,
        "card_summary": (
            (
                f"Сначала сверить входящие поставки и уточнить черновик дозаказа {total_reorder:.0f} шт.; рекламу не расширять."
                if not supplier_inbound_confirmed
                else f"Сначала дозаказать около {total_reorder:.0f} шт. и не расширять рекламу, пока покрытие не выйдет из риска."
            )
            if priority or total_reorder > 0
            else (
                f"Сначала отключить нецелевые запросы в кампании {advert_id} и заморозить её расширение до появления заказного ядра."
                if advert_id
                else "Сначала отключить нецелевые запросы и заморозить расширение рекламы до появления заказного ядра."
            )
        ),
        "card_recommendation": (
            (
                f"Сначала подтвердить SKU-level входящие поставки и уточнить черновик {total_reorder:.0f} шт.; затем пересчитать покрытие."
                if not supplier_inbound_confirmed
                else f"Сначала дозаказать около {total_reorder:.0f} шт., затем пересчитать покрытие и только после этого возвращаться к росту спроса."
            )
            if priority or total_reorder > 0
            else recommendation
        ),
        "resolution_actions": resolution_actions,
        "resolution_plan": resolution_plan,
        "per_target_evidence": rows,
        "highlight_metrics": _metric_cards(
            [
                ("Покрытие", _plain_number(min_coverage, suffix=" дн.") if min_coverage is not None else None),
                ("Риск OOS", aggregate_risk if rows else None),
                ("Дозаказ", _plain_number(total_reorder) if total_reorder > 0 else None),
            ]
        ),
    }


def _weak_sku_demand_recovery(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in resolved_targets:
        sales = _target_match(evidence_bundle, "forecast_sales_agent", target.id)
        catalog = _target_match(evidence_bundle, "forecast_catalog_agent", target.id)
        marketing = _target_match(evidence_bundle, "forecast_marketing_agent", target.id)
        supply = _target_match(evidence_bundle, "forecast_supply_agent", target.id)
        price = _target_match(evidence_bundle, "forecast_price_agent", target.id)
        finance = _target_match(evidence_bundle, "forecast_finance_agent", target.id)

        trend_row = sales.get("sales_trend_row") or {}
        conversion_row = catalog.get("conversion_row") or sales.get("sales_watch_row") or {}
        visibility_row = catalog.get("visibility_row") or marketing.get("visibility_row") or {}
        query_row = marketing.get("query_row") or {}
        query_action = marketing.get("query_action") or {}
        demand_row = marketing.get("demand_row") or {}
        supply_row = supply.get("supply_row") or {}
        price_row = price.get("price_row") or {}
        finance_row = finance.get("finance_row") or {}

        current_sales = _number(trend_row.get("current_sales"))
        previous_sales = _number(trend_row.get("previous_sales"))
        current_revenue = _number(trend_row.get("current_revenue"))
        previous_revenue = _number(trend_row.get("previous_revenue"))
        delta_sales = _safe_delta(current_sales, previous_sales)
        delta_revenue = _safe_delta(current_revenue, previous_revenue)
        trend = _trend_direction(delta_sales, delta_revenue)
        driver = _detect_weak_sku_driver(
            conversion_row=conversion_row,
            visibility_row=visibility_row,
            demand_row=demand_row,
            supply_row=supply_row,
            price_row=price_row,
        )
        next_action = _weak_sku_next_action(driver)
        lost_revenue = max((previous_revenue or 0) - (current_revenue or 0), 0)
        lost_sales = max((previous_sales or 0) - (current_sales or 0), 0)
        rows.append(
            {
                "target": target.id,
                "trend_direction": trend,
                "current_sales": current_sales,
                "previous_sales": previous_sales,
                "current_revenue": current_revenue,
                "previous_revenue": previous_revenue,
                "delta_sales": delta_sales,
                "delta_revenue": delta_revenue,
                "driver": driver,
                "next_action": next_action,
                "oos_risk": _supply_risk_from_row(supply_row),
                "product_name": _string(visibility_row.get("name")) if visibility_row else None,
                "visibility_row": visibility_row,
                "query_row": query_row,
                "query_action": query_action,
                "query_rows": marketing.get("query_rows") or [],
                "demand_row": demand_row,
                "supply_row": supply_row,
                "price_row": price_row,
                "sales_trend_row": trend_row,
                "price_status": _string(price_row.get("price_status")) if price_row else None,
                "recommended_price": _number(price_row.get("recommended_price")) if price_row else None,
                "price_decision_confidence": _string(price_row.get("decision_confidence")) if price_row else None,
                "finance_row": finance_row,
                "lost_revenue": lost_revenue,
                "lost_sales": lost_sales,
                # Observed losses are evidence. A recovery share is not; keep
                # causal effect empty until a calibrated experiment/model exists.
                "estimated_profit_recovery": None,
                "estimated_sales_recovery": None,
            }
        )

    blocked_by = list(evidence_bundle.blocking_gaps)
    if not rows:
        blocked_by.append("missing_weak_sku_evidence")
    declines = [item for item in rows if item["trend_direction"] == "decline"]
    supply_risks = [item["target"] for item in rows if item["oos_risk"] in {"high", "medium"}]
    price_risks = [item["target"] for item in rows if item["price_status"] == "above_market"]
    card_risks = [item["target"] for item in rows if item["driver"] in {"card_conversion", "visibility_drop"}]
    primary_driver = _primary_driver(rows)
    decision_status = "completed" if rows else "blocked"
    confidence = "high" if rows and declines else "medium" if rows else "low"
    current_sales_sum = round(sum(_number(item["current_sales"]) or 0 for item in rows), 2)
    previous_sales_sum = round(sum(_number(item["previous_sales"]) or 0 for item in rows), 2)
    current_revenue_sum = round(sum(_number(item["current_revenue"]) or 0 for item in rows), 2)
    previous_revenue_sum = round(sum(_number(item["previous_revenue"]) or 0 for item in rows), 2)
    delta_sales_sum = round(sum(item["delta_sales"] for item in rows), 2)
    delta_revenue_sum = round(sum(item["delta_revenue"] for item in rows), 2)
    lost_revenue_sum = round(sum(item["lost_revenue"] for item in rows), 2)
    estimated_profit_recovery = None
    estimated_sales_recovery = None
    what_found = _weak_sku_findings(rows, primary_driver, supply_risks, price_risks)
    if_no_action = (
        f"Если ничего не делать, недельная просадка по SKU сохранится: продажи уже ниже прошлого периода на {abs(delta_sales_sum):.0f} шт., "
        f"а выручка на {abs(delta_revenue_sum):.0f} ₽."
        if delta_sales_sum < 0 or delta_revenue_sum < 0
        else "Если ничего не делать, слабые SKU останутся без подтверждённой коррекции и будут продолжать тянуть вниз фокус команды."
    )
    recommendation = _weak_sku_recommendation(primary_driver, rows)
    if_action = _weak_sku_if_action_text(primary_driver, rows)
    expected_effect = _weak_sku_effect_text(
        primary_driver,
        rows,
        estimated_profit_recovery=estimated_profit_recovery,
        estimated_sales_recovery=estimated_sales_recovery,
        lost_revenue_sum=lost_revenue_sum,
        supply_risks=supply_risks,
        price_risks=price_risks,
        card_risks=card_risks,
    )
    resolution_plan = _weak_sku_resolution_plan(primary_driver, rows)
    return {
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "current_sales": current_sales_sum if current_sales_sum > 0 else None,
            "previous_sales": previous_sales_sum if previous_sales_sum > 0 else None,
            "current_revenue": current_revenue_sum if current_revenue_sum > 0 else None,
            "previous_revenue": previous_revenue_sum if previous_revenue_sum > 0 else None,
        },
        "action_metrics": {
            "trend_direction": "decline" if declines else "mixed_or_stable",
            "risk_level": "high" if declines else "medium",
            "recommended_action": recommendation,
            "estimated_profit_recovery_rub": None,
        },
        "delta_metrics": {
            "delta_sales": delta_sales_sum if delta_sales_sum != 0 else None,
            "delta_revenue": delta_revenue_sum if delta_revenue_sum != 0 else None,
            "risk_level": "high" if declines else "medium",
            "estimated_profit_recovery_rub": None,
            "estimated_sales_recovery": None,
        },
        "baseline_narrative": what_found,
        "action_narrative": if_action,
        "effect_narrative": expected_effect,
        "decision": recommendation,
        "decision_why": "Для слабых SKU прогноз сначала ищет главный драйвер просадки, а не подменяет задачу рекламной остановкой.",
        "final_recommendation": recommendation,
        "summary": recommendation,
        "card_summary": recommendation,
        "card_recommendation": recommendation,
        "card_expected_effect": expected_effect,
        "resolution_actions": _weak_sku_resolution_actions(primary_driver, rows),
        "resolution_plan": resolution_plan,
        "per_target_evidence": rows,
        "highlight_metrics": _metric_cards(
            [
                ("Просадка выручки", _money(lost_revenue_sum) if lost_revenue_sum > 0 else None),
                ("Главное действие", _driver_label(primary_driver)),
            ]
        ),
    }


def _sales_trend_guardrail(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in resolved_targets:
        sales = _target_match(evidence_bundle, "forecast_sales_agent", target.id).get("sales_trend_row") or {}
        if not sales:
            continue
        current_sales = _number(sales.get("current_sales"))
        previous_sales = _number(sales.get("previous_sales"))
        current_revenue = _number(sales.get("current_revenue"))
        previous_revenue = _number(sales.get("previous_revenue"))
        rows.append(
            {
                "target": target.id,
                "current_sales": current_sales,
                "previous_sales": previous_sales,
                "current_revenue": current_revenue,
                "previous_revenue": previous_revenue,
                "delta_sales": _safe_delta(current_sales, previous_sales),
                "delta_revenue": _safe_delta(current_revenue, previous_revenue),
            }
        )
    blocked_by = list(evidence_bundle.blocking_gaps)
    if not rows:
        blocked_by.append("missing_sales_snapshot")
    delta_sales = round(sum(item["delta_sales"] for item in rows), 2)
    delta_revenue = round(sum(item["delta_revenue"] for item in rows), 2)
    trend = _trend_direction(delta_sales, delta_revenue)
    risk = "high" if trend == "decline" and any((_number(item["current_sales"]) or 0) <= 0 for item in rows) else "medium" if trend == "decline" else "low"
    recommendation = (
        f"По SKU {', '.join(item['target'] for item in rows)} запустить исправление воронки: карточка, цена, реклама и наличие без автоматического масштабирования."
        if trend == "decline"
        else "Сохранить текущие условия и не обещать uplift без отдельной гипотезы."
    )
    resolution_actions = _sales_trend_resolution_actions(rows, trend)
    resolution_plan = _sales_trend_resolution_plan(rows, trend, risk)
    return {
        "decision_status": "completed" if rows else "blocked",
        "confidence": "high" if rows else "low",
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "current_sales": round(sum((_number(item["current_sales"]) or 0) for item in rows), 2) if rows else None,
            "previous_sales": round(sum((_number(item["previous_sales"]) or 0) for item in rows), 2) if rows else None,
            "current_revenue": round(sum((_number(item["current_revenue"]) or 0) for item in rows), 2) if rows else None,
            "previous_revenue": round(sum((_number(item["previous_revenue"]) or 0) for item in rows), 2) if rows else None,
        },
        "action_metrics": {
            "trend_direction": trend,
            "risk_level": risk,
            "recommended_action": recommendation,
        },
        "delta_metrics": {
            "delta_sales": delta_sales if delta_sales != 0 else None,
            "delta_revenue": delta_revenue if delta_revenue != 0 else None,
            "trend_direction": trend,
            "risk_level": risk,
        },
        "baseline_narrative": f"Сравнение периодов показывает тренд {trend} по продажам.",
        "action_narrative": recommendation,
        "effect_narrative": "Этот сценарий даёт guardrail по тренду и не обещает искусственный uplift.",
        "decision": recommendation,
        "decision_why": "Sales guardrail отделяет сам факт просадки от причин, которые нужно проверять дальше.",
        "final_recommendation": recommendation,
        "summary": recommendation,
        "card_summary": recommendation,
        "card_recommendation": recommendation,
        "card_expected_effect": "Прогноз эффекта: остановить дальнейшую просадку и вернуть спрос после исправления главного барьера воронки.",
        "resolution_actions": resolution_actions,
        "resolution_plan": resolution_plan,
        "per_target_evidence": rows,
        "highlight_metrics": _metric_cards(
            [
                ("Изменение продаж", _plain_number(delta_sales) if delta_sales != 0 else None),
                ("Изменение выручки", _money(delta_revenue) if delta_revenue != 0 else None),
                ("Риск", risk),
            ]
        ),
    }


def _price_positioning_guardrail(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in resolved_targets:
        price = _target_match(evidence_bundle, "forecast_price_agent", target.id).get("price_row") or {}
        if not price:
            continue
        rows.append(
            {
                "target": target.id,
                "current_price": _number(price.get("source_price_promo")),
                "competitor_median_price": _number(price.get("competitor_price_median")),
                "price_gap_rub": _number(price.get("source_vs_median_gap_rub")),
                "price_gap_pct": _number(price.get("source_vs_median_gap_pct")),
                "competitor_count": _number(price.get("competitor_count")),
                "price_status": _string(price.get("price_status")) or "unknown",
                "recommended_price": _number(price.get("recommended_price")),
                "recommended_price_delta_pct": _number(price.get("recommended_price_delta_pct")),
                "expected_orders_7d": _number(price.get("expected_orders_7d")),
                "expected_net_profit_7d": _number(price.get("expected_net_profit_7d")),
                "decision_status": _string(price.get("decision_status")),
                "decision_confidence": _string(price.get("decision_confidence")),
            }
        )
    blocked_by = list(evidence_bundle.blocking_gaps)
    if not rows:
        blocked_by.append("missing_price_snapshot")
    primary = rows[0] if rows else {}
    competitor_count = _number(primary.get("competitor_count")) or 0
    price_status = _string(primary.get("price_status")) or "unknown"
    risk = _market_position_risk(price_status, competitor_count)
    recommended_price = _number(primary.get("recommended_price"))
    recommendation = (
        f"По SKU {primary.get('target') or 'SKU'} изменить цену до {recommended_price:.0f} ₽ и не расширять рекламу до повторного замера заказов и прибыли."
        if recommended_price is not None
        else _price_recommendation(price_status)
    )
    resolution_actions = _price_guardrail_resolution_actions(primary, recommendation)
    resolution_plan = _price_guardrail_resolution_plan(primary, recommendation)
    return {
        "decision_status": "completed" if rows else "blocked",
        "confidence": "high" if rows and competitor_count >= 5 and price_status != "thin_data" else "medium" if rows else "low",
        "blocked_by": blocked_by + (["thin_competitor_data"] if rows and (competitor_count < 5 or price_status == "thin_data") else []),
        "baseline_metrics": primary,
        "action_metrics": {
            "price_guardrail": recommendation,
            "market_position_risk": risk,
            "recommended_price": recommended_price,
            "decision_confidence": primary.get("decision_confidence"),
        },
        "delta_metrics": {
            "price_gap_rub": primary.get("price_gap_rub"),
            "price_gap_pct": primary.get("price_gap_pct"),
            "market_position_risk": risk,
        },
        "baseline_narrative": f"Цена сейчас в статусе {price_status}; конкурентов в срезе: {competitor_count:.0f}.",
        "action_narrative": _price_action_text(price_status),
        "effect_narrative": "Это guardrail по безопасности ценового решения; при наличии mart price decision можно опираться на готовый repricing-кандидат и expected economics.",
        "decision": recommendation,
        "decision_why": "Price forecast блокирует рискованные ценовые движения и, если price mart доступен, использует его готовый decision contour вместо сырого движения к медиане.",
        "final_recommendation": recommendation,
        "summary": recommendation,
        "card_summary": recommendation,
        "card_recommendation": recommendation,
        "card_expected_effect": "Прогноз эффекта: снять ценовое трение и вернуть заказы без провала по марже.",
        "resolution_actions": resolution_actions,
        "resolution_plan": resolution_plan,
        "per_target_evidence": rows,
        "highlight_metrics": _metric_cards(
            [
                ("Разрыв к рынку", _money(primary.get("price_gap_rub")) if primary.get("price_gap_rub") is not None else None),
                ("Разрыв, %", _plain_number(primary.get("price_gap_pct"), suffix="%") if primary.get("price_gap_pct") is not None else None),
                ("Риск", risk if rows else None),
            ]
        ),
    }


def _scale_profitable_sku_ads(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows = [_growth_row(evidence_bundle, target.id) for target in resolved_targets]
    rows = [row for row in rows if row]
    for row in rows:
        proposed_budget, current_budget, growth_pct = _evidence_based_ad_budget(row)
        row["proposed_ad_budget_rub"] = proposed_budget
        row["current_ad_budget_rub"] = current_budget
        row["evidence_safe_growth_pct"] = growth_pct
        if proposed_budget is None:
            row["safety_blockers"].append("missing_evidence_based_ad_budget")
    blockers = _growth_blockers(evidence_bundle, rows)
    actionable = [
        row
        for row in rows
        if not row["safety_blockers"] and row.get("proposed_ad_budget_rub") is not None
    ]
    current_revenue = round(sum(_number(row.get("current_revenue")) or 0 for row in rows), 2)
    current_margin = round(sum(_number(row.get("contribution_margin")) or 0 for row in rows), 2)
    suggested_spend = (
        round(sum(_number(row.get("proposed_ad_budget_rub")) or 0 for row in actionable), 2)
        if actionable
        else None
    )
    recommendation = (
        f"По SKU {', '.join(row['target'] for row in actionable)} поднять лимит рекламы до {suggested_spend:.0f} ₽ на 7 дней и не выходить за guardrail по марже и покрытию."
        if actionable
        else (
            f"По SKU {', '.join(row['target'] for row in rows) or 'не определён'} не масштабировать рекламу: "
            "нет подтверждённого текущего бюджета и evidence-based лимита изменения."
        )
    )
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=blockers,
        decision_status="completed" if actionable and not blockers else "needs_review" if rows else "blocked",
        confidence="high" if actionable and not blockers else "medium" if rows else "low",
        baseline_metrics={
            "current_revenue": current_revenue,
            "contribution_margin": current_margin,
            "coverage_days": _min_number(row.get("coverage_days") for row in rows),
        },
        action_metrics={
            "recommended_action": recommendation,
            "test_ad_budget_rub": suggested_spend,
            "growth_targets": ", ".join(row["target"] for row in actionable) or None,
        },
        delta_metrics={
            "estimated_revenue_uplift": None,
            "margin_guardrail": "positive" if current_margin > 0 else None,
        },
        summary=recommendation,
        recommendation=recommendation,
        why="Scale-сценарий разрешает рост только при положительной экономике, достаточном stock coverage и отсутствии price/supply blockers.",
    )


def _reprice_with_margin_guardrail(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    *,
    today: date | None,
) -> dict[str, Any]:
    rows = [_growth_row(evidence_bundle, target.id) for target in resolved_targets]
    rows = [row for row in rows if row]
    for row in rows:
        _append_price_safety_blockers(row, today=today)
        if (
            row.get("contribution_margin") is None
            and row.get("margin_before_cogs_pct") is None
            and row.get("expected_margin_pct") is None
            and row.get("cost_price_rub") is None
        ):
            row["safety_blockers"].append("missing_finance_evidence")
    blockers = _price_action_blockers(evidence_bundle, rows)
    actionable = [row for row in rows if not row["safety_blockers"] and row.get("price_status") in {"above_market", "below_market"}]
    primary = actionable[0] if actionable else rows[0] if rows else {}
    candidate_price = _candidate_price(primary)
    current_price = _number(primary.get("current_price"))
    target_id = primary.get("target")
    review_reasons: list[str] = []
    if "low_price_decision_confidence" in blockers:
        review_reasons.append("текущий ценовой расчёт пока ненадёжен")
    if "stale_price_decision" in blockers:
        review_reasons.append("ценовой расчёт устарел")
    if "supply_risk" in blockers:
        review_reasons.append("остаток/покрытие под риском")
    review_reason_text = "; ".join(review_reasons)
    if candidate_price and actionable:
        recommendation = (
            f"Снизить цену по SKU {target_id} до {candidate_price:.0f} ₽"
            + (f" с текущих {current_price:.0f} ₽." if current_price is not None else ".")
        )
    elif candidate_price:
        recommendation = (
            f"Подготовить снижение цены по SKU {target_id} до {candidate_price:.0f} ₽"
            + (f" с текущих {current_price:.0f} ₽," if current_price is not None else ",")
            + " но не запускать сразу"
            + (f": {review_reason_text}." if review_reason_text else ".")
        )
    else:
        recommendation = "Не менять цену: пока не хватает надёжных данных по цене и экономике."
    margin_bridge = _build_price_margin_bridge(primary, candidate_price)
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=blockers,
        decision_status="completed" if actionable else "needs_review" if rows else "blocked",
        confidence="high" if actionable else "medium" if rows else "low",
        baseline_metrics={
            "current_price": primary.get("current_price"),
            "competitor_median_price": primary.get("competitor_median_price"),
            "price_gap_pct": primary.get("price_gap_pct"),
            "current_net_profit_7d": primary.get("current_net_profit_7d"),
            "margin_pct_current": primary.get("margin_pct_current"),
            "elasticity_to_apply": primary.get("elasticity_to_apply"),
        },
        action_metrics={
            "candidate_price": candidate_price,
            "expected_orders_7d": primary.get("expected_orders_7d"),
            "expected_net_profit_7d": primary.get("expected_net_profit_7d"),
            "expected_margin_pct": primary.get("expected_margin_pct"),
            "decision_status": primary.get("price_decision_status"),
            "decision_confidence": primary.get("price_decision_confidence"),
            "recommended_action": recommendation,
        },
        delta_metrics={
            "recommended_price_delta_pct": primary.get("recommended_price_delta_pct"),
            "expected_orders_delta": _delta_number(primary.get("expected_orders_7d"), primary.get("baseline_orders_7d")),
            "expected_net_profit_delta": _delta_number(primary.get("expected_net_profit_7d"), primary.get("current_net_profit_7d")),
        },
        summary=recommendation,
        recommendation=recommendation,
        why="Ценовой сценарий опирается на pricing.mart_price_decision_latest и использует mart recommendation вместо грубого движения к медиане рынка.",
        margin_bridge=margin_bridge,
    )


def _reorder_for_growth(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows = [_growth_row(evidence_bundle, target.id) for target in resolved_targets]
    rows = [row for row in rows if row]
    blockers = _growth_blockers(evidence_bundle, rows)
    reorder_rows = [row for row in rows if (_number(row.get("reorder_qty")) or 0) > 0]
    supplier_inbound_confirmed = bool(reorder_rows) and all(
        row.get("supplier_inbound_confirmed") for row in reorder_rows
    )
    if reorder_rows and not supplier_inbound_confirmed:
        blockers.append("missing_supplier_inbound_snapshot")
    if rows and not reorder_rows:
        blockers.append("reorder_not_supported_by_current_supply")
    total_reorder = round(sum(_number(row.get("reorder_qty")) or 0 for row in reorder_rows), 2)
    recommendation = (
        (
            f"Запланировать дозаказ около {total_reorder:.0f} шт. по SKU {', '.join(row['target'] for row in reorder_rows)}."
            if supplier_inbound_confirmed
            else (
                f"Сначала сверить входящие поставки поставщика по SKU {', '.join(row['target'] for row in reorder_rows)}; "
                f"{total_reorder:.0f} шт. — консервативный черновик без supplier inbound."
            )
        )
        if total_reorder > 0
        else "Дозаказ не подтвержден: нет сочетания скорости продаж и критичного покрытия."
    )
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=[item for item in blockers if item != "supply_risk"],
        decision_status=(
            "completed"
            if reorder_rows and supplier_inbound_confirmed
            else "needs_review"
            if rows
            else "blocked"
        ),
        confidence="high" if reorder_rows and supplier_inbound_confirmed else "medium" if rows else "low",
        baseline_metrics={
            "coverage_days": _min_number(row.get("coverage_days") for row in rows),
            "sales_velocity": round(sum(_number(row.get("sales_velocity")) or 0 for row in rows), 2),
            "available_stock": round(sum(_number(row.get("available_stock")) or 0 for row in rows), 2),
        },
        action_metrics={"reorder_qty": total_reorder or None, "recommended_action": recommendation},
        delta_metrics={"reorder_qty": total_reorder or None, "lost_sales_risk": "high" if reorder_rows else None},
        summary=recommendation,
        recommendation=recommendation,
        why="Reorder-сценарий требует скорости продаж и считает покрытие по продаваемому остатку quantity.",
    )


def _overstock_liquidation(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
    *,
    today: date | None,
) -> dict[str, Any]:
    rows = [_growth_row(evidence_bundle, target.id) for target in resolved_targets]
    rows = [row for row in rows if row]
    overstock_rows = [
        row for row in rows
        if (_number(row.get("available_stock")) or 0) >= 60 and ((_number(row.get("sales_velocity")) or 0) <= 0 or (_number(row.get("coverage_days")) or 0) >= 90)
    ]
    for row in rows:
        _append_price_safety_blockers(row, today=today)
    blockers = _price_action_blockers(evidence_bundle, rows)
    discount = _evidence_discount_pct(overstock_rows[0]) if overstock_rows else None
    if overstock_rows and discount is None:
        blockers.append("missing_evidence_based_discount")
    unsafe_price_blockers = {
        "thin_price_data",
        "low_price_decision_confidence",
        "stale_price_decision",
        "future_price_decision",
        "invalid_price_freshness",
        "missing_finance_evidence",
        "missing_cogs",
    }
    actionable = discount is not None and not (unsafe_price_blockers & set(blockers))
    recommendation = (
        f"Подготовить скидку {discount:g}% по SKU {', '.join(row['target'] for row in overstock_rows)} из подтверждённого price decision, с контролем маржи."
        if actionable
        else "Не запускать распродажу: нужен свежий ценовой расчёт, достаточный конкурентный срез и подтверждённая экономика скидки."
    )
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=blockers,
        decision_status="completed" if actionable else "needs_review" if rows else "blocked",
        confidence="medium" if actionable or rows else "low",
        baseline_metrics={
            "available_stock": round(sum(_number(row.get("available_stock")) or 0 for row in rows), 2),
            "coverage_days": _min_number(row.get("coverage_days") for row in rows),
        },
        action_metrics={"safe_discount_pct": discount if actionable else None, "recommended_action": recommendation},
        delta_metrics={"stock_turnover_action": "liquidation_review" if actionable else None},
        summary=recommendation,
        recommendation=recommendation,
        why="Overstock-сценарий не разрешает скидку без price/finance guardrails.",
    )


def _normquery_bid_optimization(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows = []
    for target in resolved_targets:
        match = _target_match(evidence_bundle, "forecast_marketing_agent", target.id)
        opp = _opportunity_facts(evidence_bundle, target.id)
        action = opp.get("marketing_action") or match.get("campaign_action") or {}
        safety_blockers = list(action.get("blockers") or [])
        if _string(action.get("demand_source_status")) == "unpopulated":
            safety_blockers.append("marketing_demand_mart_unpopulated")
        if _string(action.get("confidence")) == "low":
            safety_blockers.append("marketing_low_confidence")
        if _string(action.get("action_mode")) == "test":
            safety_blockers.append("marketing_test_only")
        rows.append(
            {
                "target": target.id,
                "entity_type": target.entity_type,
                "marketing_action": action,
                "marketing_decision_hint": _string(action.get("decision_hint")),
                "marketing_confidence": _string(action.get("confidence")),
                "safe_growth_pct": _number(action.get("safe_growth_pct")),
                "winner_query_share": _number(action.get("winner_query_share")),
                "waste_query_share": _number(action.get("waste_query_share")),
                "query_band": _string(action.get("query_band")),
                "marketing_source_status": _string(action.get("marketing_source_status")),
                "safety_blockers": list(dict.fromkeys(str(item) for item in safety_blockers if item)),
                "query_rows": match.get("query_rows") or [],
                "query_row": match.get("query_row") or {},
                "visibility_row": match.get("visibility_row") or {},
                "demand_row": match.get("demand_row") or {},
                "campaign_row": match.get("campaign_row") or {},
                "forecast_row": match.get("forecast_row") or {},
                "product_name": _string((match.get("visibility_row") or {}).get("name")),
                "source_facts": match,
            }
        )
    blockers = _growth_blockers(evidence_bundle, rows)
    review_blockers = {
        "marketing_demand_mart_unpopulated",
        "marketing_low_confidence",
        "marketing_test_only",
        "marketing_mart_unpopulated",
    }
    review_needed = any(any(str(blocker).startswith(prefix) for prefix in review_blockers) for blocker in blockers)
    recommendation = _normquery_recommendation(rows)
    effect_text = _normquery_effect_text(rows)
    summary = _merge_action_and_effect(recommendation, effect_text)
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=blockers,
        decision_status="needs_review" if rows and review_needed else "completed" if rows else "blocked",
        confidence="medium" if rows and review_needed else "high" if rows else "low",
        baseline_metrics={
            "targets_checked": len(rows),
            "current_spend_rub": _normquery_current_spend(rows),
        },
        action_metrics={"recommended_action": recommendation},
        delta_metrics={
            "bid_optimization": "review" if review_needed else "confirmed_optimization",
            "ad_spend_saved_rub": _normquery_saved_spend(rows),
        },
        summary=summary,
        recommendation=recommendation,
        why="Normquery-сценарий требует рекламного evidence и смежной проверки продаж, остатков и маржи.",
        effect_narrative=effect_text,
    )


def _recover_conversion_drop(
    plan: ForecastPlan,
    resolved_targets: list[ResolvedTarget],
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    rows = [_growth_row(evidence_bundle, target.id) for target in resolved_targets]
    rows = [row for row in rows if row]
    blockers = _growth_blockers(evidence_bundle, rows)
    current_revenue = round(sum(_number(row.get("current_revenue")) or 0 for row in rows), 2)
    recommendation = _conversion_recovery_recommendation(rows)
    return _growth_output(
        plan=plan,
        rows=rows,
        blockers=blockers,
        decision_status="needs_review" if rows else "blocked",
        confidence="medium" if rows else "low",
        baseline_metrics={"current_revenue": current_revenue, "conversion_risk": "confirmed" if rows else None},
        action_metrics={"recommended_action": recommendation},
        delta_metrics={"estimated_revenue_recovery": None},
        summary=recommendation,
        recommendation=recommendation,
        why="Conversion recovery не обещает точный uplift: численный эффект появится только после контролируемого замера.",
    )


def _growth_row(evidence_bundle: ForecastEvidenceBundle, target_id: str) -> dict[str, Any]:
    opportunity_facts = _opportunity_facts(evidence_bundle, target_id)
    sales = _target_match(evidence_bundle, "forecast_sales_agent", target_id).get("sales_trend_row") or opportunity_facts.get("sales") or {}
    supply = _target_match(evidence_bundle, "forecast_supply_agent", target_id).get("supply_row") or opportunity_facts.get("supply") or {}
    price = _target_match(evidence_bundle, "forecast_price_agent", target_id).get("price_row") or opportunity_facts.get("price") or {}
    finance = _target_match(evidence_bundle, "forecast_finance_agent", target_id).get("finance_row") or opportunity_facts.get("finance") or {}
    marketing_match = _target_match(evidence_bundle, "forecast_marketing_agent", target_id)
    catalog_match = _target_match(evidence_bundle, "forecast_catalog_agent", target_id)
    conversion = (
        catalog_match.get("conversion_row")
        or _target_match(evidence_bundle, "forecast_sales_agent", target_id).get("sales_watch_row")
        or opportunity_facts.get("conversion")
        or {}
    )
    opportunity = _target_match(evidence_bundle, "forecast_sales_agent", target_id).get("opportunity") or evidence_bundle.per_target.get(target_id, {}).get("opportunity") or {}

    available_stock = _number(supply.get("stock_qty"))
    if available_stock is None:
        available_stock = _number(supply.get("quantity"))
    supplier_inbound = _supplier_inbound(supply)
    sales_7d = _number(supply.get("sales_7d_units")) or _number(sales.get("current_sales"))
    sales_velocity = round((sales_7d or 0) / 7, 2) if sales_7d is not None else None
    contribution_margin = _number(finance.get("contribution_margin")) or _number(price.get("current_net_profit_7d"))
    margin_before_cogs_pct = _number(finance.get("payout_margin_before_cogs_pct")) or _number(price.get("margin_pct_current"))
    has_price_profit_signal = (
        _number(price.get("cost_price_rub")) is not None
        and (
            _number(price.get("expected_margin_pct")) is not None
            or _number(price.get("margin_pct_current")) is not None
            or _number(price.get("current_net_profit_7d")) is not None
        )
    )
    blockers = list(opportunity.get("blockers") or [])
    if not finance and not has_price_profit_signal:
        blockers.append("missing_finance_evidence")
    if finance and (_number(finance.get("missing_cost_rows")) or 0) > 0:
        blockers.append("missing_cogs")
    if supply and ((available_stock is not None and available_stock <= 0) or ((_number(supply.get("days_of_stock")) or 999) < 21)):
        blockers.append("supply_risk")
    if price and (_string(price.get("price_status")) == "thin_data" or (_number(price.get("competitor_count")) or 0) < 5):
        blockers.append("thin_price_data")
    if contribution_margin is not None and contribution_margin <= 0:
        blockers.append("negative_margin")

    return {
        "target": target_id,
        "current_sales": _number(sales.get("current_sales")),
        "previous_sales": _number(sales.get("previous_sales")),
        "current_revenue": _number(sales.get("current_revenue")),
        "previous_revenue": _number(sales.get("previous_revenue")),
        "delta_revenue": _number(sales.get("delta_revenue")),
        "available_stock": available_stock,
        "sales_velocity": sales_velocity,
        "coverage_days": _number(supply.get("days_of_stock")),
        "reorder_qty": _number(supply.get("reorder_qty_recommended")),
        "supplier_inbound_qty": supplier_inbound,
        "supplier_inbound_confirmed": supplier_inbound is not None,
        "customer_transit_to_client": _number(supply.get("in_way_to_client")),
        "customer_transit_from_client": _number(supply.get("in_way_from_client")),
        "current_price": _number(price.get("source_price_promo")),
        "product_name": _string((catalog_match.get("visibility_row") or marketing_match.get("visibility_row") or {}).get("name")),
        "baseline_revenue_gross_7d": _number(price.get("baseline_revenue_gross_7d")),
        "baseline_revenue_net_7d": _number(price.get("baseline_revenue_net_7d")),
        "baseline_adv_spend_7d": _number(price.get("baseline_adv_spend_7d")),
        "competitor_median_price": _number(price.get("competitor_price_median")),
        "price_gap_pct": _number(price.get("source_vs_median_gap_pct")),
        "competitor_count": _number(price.get("competitor_count")),
        "price_status": _string(price.get("price_status")),
        "candidate_price_min": _number(price.get("candidate_price_min") or price.get("source_price_segment_min")),
        "candidate_price_max": _number(price.get("candidate_price_max") or price.get("source_price_segment_max")),
        "recommended_price": _number(price.get("recommended_price")),
        "recommended_price_delta_pct": _number(price.get("recommended_price_delta_pct")),
        "expected_orders_7d": _number(price.get("expected_orders_7d")),
        "expected_net_profit_7d": _number(price.get("expected_net_profit_7d")),
        "expected_margin_pct": _number(price.get("expected_margin_pct")),
        "price_decision_status": _string(price.get("decision_status")),
        "price_decision_confidence": _string(price.get("decision_confidence")),
        "price_decision_reason_code": _string(price.get("decision_reason_code")),
        "price_snapshot_date": price.get("snapshot_date"),
        "price_ai_run_date": price.get("price_ai_run_date"),
        "elasticity_to_apply": _number(price.get("elasticity_to_apply")),
        "elasticity_source": _string(price.get("elasticity_source")),
        "cost_price_rub": _number(price.get("cost_price_rub")),
        "ad_percent": _number(price.get("ad_percent")),
        "tax_percent": _number(price.get("tax_percent")),
        "expected_outbound_per_buyout_rub": _number(price.get("expected_outbound_per_buyout_rub")),
        "expected_return_logistics_rub": _number(price.get("expected_return_logistics_rub")),
        "storage_total_rub": _number(price.get("storage_total_rub")),
        "other_expenses_rub": _number(price.get("other_expenses_rub")),
        "baseline_orders_7d": _number(price.get("baseline_orders_7d")),
        "current_net_profit_7d": _number(price.get("current_net_profit_7d")),
        "margin_pct_current": _number(price.get("margin_pct_current")),
        "contribution_margin": contribution_margin,
        "margin_before_cogs_pct": margin_before_cogs_pct,
        "cart_to_order_conversion": _number(conversion.get("cart_to_order_conversion")),
        "visibility_row": catalog_match.get("visibility_row") or marketing_match.get("visibility_row") or {},
        "marketing_action": opportunity_facts.get("marketing_action") or marketing_match.get("query_action") or {},
        "query_rows": marketing_match.get("query_rows") or [],
        "query_row": marketing_match.get("query_row") or {},
        "safety_blockers": list(dict.fromkeys(blockers)),
        "opportunity": opportunity,
        "source_facts": opportunity_facts,
    }


def _opportunity_facts(evidence_bundle: ForecastEvidenceBundle, target_id: str) -> dict[str, Any]:
    target_payload = evidence_bundle.per_target.get(target_id) or {}
    source_facts = target_payload.get("opportunity_source_facts")
    return source_facts if isinstance(source_facts, dict) else {}


def _growth_blockers(evidence_bundle: ForecastEvidenceBundle, rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = list(evidence_bundle.blocking_gaps)
    for row in rows:
        blockers.extend(str(item) for item in row.get("safety_blockers") or [] if item)
    result: list[str] = []
    for blocker in blockers:
        if blocker not in result:
            result.append(blocker)
    return result


def _price_action_blockers(evidence_bundle: ForecastEvidenceBundle, rows: list[dict[str, Any]]) -> list[str]:
    blockers = _growth_blockers(evidence_bundle, rows)
    blockers = [
        blocker
        for blocker in blockers
        if not blocker.startswith(("marketing_visibility_missing", "marketing_demand_missing"))
    ]
    has_embedded_finance = bool(rows) and all(
        row.get("cost_price_rub") is not None
        and (row.get("expected_margin_pct") is not None or row.get("margin_pct_current") is not None)
        for row in rows
    )
    if has_embedded_finance:
        blockers = [blocker for blocker in blockers if not blocker.startswith("finance_not_found")]
    return list(dict.fromkeys(blockers))


def _append_price_safety_blockers(row: dict[str, Any], *, today: date | None) -> None:
    blockers = list(row.get("safety_blockers") or [])
    if row.get("price_status") == "thin_data" or (_number(row.get("competitor_count")) or 0) < 5:
        blockers.append("thin_price_data")
    if _string(row.get("price_decision_confidence")) == "low":
        blockers.append("low_price_decision_confidence")
    if "stale" in (_string(row.get("price_decision_reason_code")) or "").lower():
        blockers.append("stale_price_decision")

    raw_date = row.get("price_ai_run_date") or row.get("price_snapshot_date")
    price_date = _parse_date(raw_date)
    run_date = today or date.today()
    if raw_date and price_date is None:
        blockers.append("invalid_price_freshness")
    elif price_date is not None and price_date > run_date:
        blockers.append("future_price_decision")
    elif price_date is not None and (run_date - price_date).days > 14:
        blockers.append("stale_price_decision")
    row["safety_blockers"] = list(dict.fromkeys(blockers))


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()[:10]
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _candidate_price(row: dict[str, Any]) -> float | None:
    recommended = _number(row.get("recommended_price"))
    if recommended is not None:
        return round(recommended, 2)
    # A market median is evidence about positioning, not an executable price.
    # Only the versioned price-decision mart may provide a numeric candidate.
    return None


def _evidence_discount_pct(row: dict[str, Any]) -> float | None:
    if _string(row.get("price_decision_status")) not in {"lower", "discount", "liquidate"}:
        return None
    delta_pct = _number(row.get("recommended_price_delta_pct"))
    if delta_pct is not None and delta_pct < 0:
        return round(abs(delta_pct), 2)
    current_price = _number(row.get("current_price"))
    candidate_price = _number(row.get("recommended_price"))
    if current_price is None or current_price <= 0 or candidate_price is None:
        return None
    if not 0 < candidate_price < current_price:
        return None
    return round((current_price - candidate_price) * 100 / current_price, 2)


def _evidence_based_ad_budget(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    action = row.get("marketing_action")
    if not isinstance(action, dict):
        return None, None, None
    current_budget = None
    for key in ("current_budget_rub", "budget_rub", "current_spend_rub", "spend_7d"):
        current_budget = _number(action.get(key))
        if current_budget is not None:
            break
    growth_pct = _number(action.get("safe_growth_pct"))
    if current_budget is None or current_budget < 0 or growth_pct is None or growth_pct <= 0:
        return None, current_budget, growth_pct
    return round(current_budget * (1 + growth_pct / 100), 2), current_budget, growth_pct


def _delta_number(current: Any, baseline: Any) -> float | None:
    current_value = _number(current)
    baseline_value = _number(baseline)
    if current_value is None or baseline_value is None:
        return None
    return round(current_value - baseline_value, 2)


def _build_price_margin_bridge(row: dict[str, Any], candidate_price: float | None) -> dict[str, Any]:
    if not row or candidate_price is None:
        return {}
    baseline_orders = _number(row.get("baseline_orders_7d"))
    action_orders = _number(row.get("expected_orders_7d"))
    current_price = _number(row.get("current_price"))
    cost_price = _number(row.get("cost_price_rub"))
    if baseline_orders is None or action_orders is None or current_price is None or cost_price is None:
        return {}

    ad_percent = ((_number(row.get("ad_percent")) or 0) / 100)
    tax_percent = ((_number(row.get("tax_percent")) or 0) / 100)
    outbound_per_order = _number(row.get("expected_outbound_per_buyout_rub")) or 0
    return_logistics_per_order = _number(row.get("expected_return_logistics_rub")) or 0
    storage_total = _number(row.get("storage_total_rub")) or 0
    other_expenses = _number(row.get("other_expenses_rub")) or 0

    baseline_revenue = _number(row.get("baseline_revenue_net_7d")) or round(baseline_orders * current_price, 2)
    action_revenue = round(action_orders * candidate_price, 2)
    baseline_ad_spend = _number(row.get("baseline_adv_spend_7d"))
    if baseline_ad_spend is None:
        baseline_ad_spend = round(baseline_revenue * ad_percent, 2)
    action_ad_spend = round(action_revenue * ad_percent, 2)
    baseline_tax = round(baseline_revenue * tax_percent, 2)
    action_tax = round(action_revenue * tax_percent, 2)
    baseline_cogs = round(baseline_orders * cost_price, 2)
    action_cogs = round(action_orders * cost_price, 2)
    baseline_outbound = round(baseline_orders * outbound_per_order, 2)
    action_outbound = round(action_orders * outbound_per_order, 2)
    baseline_return_logistics = round(baseline_orders * return_logistics_per_order, 2)
    action_return_logistics = round(action_orders * return_logistics_per_order, 2)

    baseline_profit_known = _number(row.get("current_net_profit_7d"))
    action_profit_known = _number(row.get("expected_net_profit_7d"))
    baseline_profit_estimated = round(
        baseline_revenue
        - baseline_cogs
        - baseline_ad_spend
        - baseline_tax
        - baseline_outbound
        - baseline_return_logistics
        - storage_total
        - other_expenses,
        2,
    )
    action_profit_estimated = round(
        action_revenue
        - action_cogs
        - action_ad_spend
        - action_tax
        - action_outbound
        - action_return_logistics
        - storage_total
        - other_expenses,
        2,
    )
    baseline_reconciliation = (
        round((baseline_profit_known or baseline_profit_estimated) - baseline_profit_estimated, 2)
        if baseline_profit_known is not None
        else 0.0
    )
    action_reconciliation = (
        round((action_profit_known or action_profit_estimated) - action_profit_estimated, 2)
        if action_profit_known is not None
        else 0.0
    )

    baseline_unit_gross = current_price - cost_price
    action_unit_gross = candidate_price - cost_price
    price_cut_effect = round((action_unit_gross - baseline_unit_gross) * baseline_orders, 2)
    volume_compensation_effect = round(max(action_orders - baseline_orders, 0) * action_unit_gross, 2)
    net_profit_delta = round((action_profit_known or action_profit_estimated) - (baseline_profit_known or baseline_profit_estimated), 2)

    def _component(baseline_value: float, action_value: float) -> dict[str, float]:
        return {
            "baseline": round(baseline_value, 2),
            "action": round(action_value, 2),
            "delta": round(action_value - baseline_value, 2),
        }

    components = {
        "revenue": _component(baseline_revenue, action_revenue),
        "cogs": _component(baseline_cogs, action_cogs),
        "ad_spend": _component(baseline_ad_spend, action_ad_spend),
        "tax": _component(baseline_tax, action_tax),
        "outbound_logistics": _component(baseline_outbound, action_outbound),
        "return_logistics": _component(baseline_return_logistics, action_return_logistics),
        "storage": _component(storage_total, storage_total),
        "other_expenses": _component(other_expenses, other_expenses),
        "reconciliation_adjustment": _component(baseline_reconciliation, action_reconciliation),
        "net_profit": _component(
            baseline_profit_known or baseline_profit_estimated,
            action_profit_known or action_profit_estimated,
        ),
    }
    components["cogs"]["label"] = "затраты на себестоимость"
    components["cogs"]["delta_label"] = "прирост затрат на себестоимость"
    components["cogs"]["delta_reason"] = "рост объема заказов"

    primary_risk = "price"
    blockers = list(row.get("safety_blockers") or [])
    if "supply_risk" in blockers:
        primary_risk = "остаток"
    elif "stale_price_decision" in blockers or "low_price_decision_confidence" in blockers:
        primary_risk = "сигнал цены"
    elif abs(components["outbound_logistics"]["delta"]) > abs(net_profit_delta):
        primary_risk = "логистика"
    elif abs(components["ad_spend"]["delta"]) > abs(net_profit_delta):
        primary_risk = "реклама"

    return {
        "type": "price_margin_bridge",
        "baseline_orders_7d": baseline_orders,
        "action_orders_7d": action_orders,
        "current_price": round(current_price, 2),
        "candidate_price": round(candidate_price, 2),
        "unit_cost_rub": round(cost_price, 2),
        "baseline_unit_gross_margin": round(baseline_unit_gross, 2),
        "action_unit_gross_margin": round(action_unit_gross, 2),
        "price_cut_effect_on_gross_margin": price_cut_effect,
        "volume_compensation_effect_on_gross_margin": volume_compensation_effect,
        "incremental_cogs_due_to_volume": round(components["cogs"]["delta"], 2),
        "net_profit_delta": net_profit_delta,
        "primary_risk_component": primary_risk,
        "components": components,
    }


def _growth_resolution_actions(
    *,
    plan: ForecastPlan,
    rows: list[dict[str, Any]],
    recommendation: str,
    baseline_metrics: dict[str, Any],
    action_metrics: dict[str, Any],
    delta_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    plan_items = _growth_resolution_plan(
        plan=plan,
        rows=rows,
        recommendation=recommendation,
        baseline_metrics=baseline_metrics,
        action_metrics=action_metrics,
        delta_metrics=delta_metrics,
    )
    actions: list[dict[str, str]] = []
    for item in plan_items:
        actions.append(
            {
                "area": str(item.get("area") or "action"),
                "action": str(item.get("instruction") or recommendation),
                "reason": str(item.get("currentMetric") or ""),
                "expectedMetric": str(item.get("targetMetric") or ""),
            }
        )
    return actions


def _growth_resolution_plan(
    *,
    plan: ForecastPlan,
    rows: list[dict[str, Any]],
    recommendation: str,
    baseline_metrics: dict[str, Any],
    action_metrics: dict[str, Any],
    delta_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if plan.action_family == "scale_profitable_sku_ads":
        return _scale_growth_resolution_plan(rows, action_metrics, delta_metrics)
    if plan.action_family == "reprice_with_margin_guardrail":
        return _reprice_growth_resolution_plan(rows)
    if plan.action_family == "reorder_for_growth":
        return _reorder_growth_resolution_plan(rows, action_metrics)
    if plan.action_family == "overstock_liquidation":
        return _overstock_growth_resolution_plan(rows, action_metrics)
    if plan.action_family == "normquery_bid_optimization":
        return _normquery_resolution_plan(rows, recommendation)
    if plan.action_family == "recover_conversion_drop":
        return _conversion_recovery_resolution_plan(rows)
    return []


def _scale_growth_resolution_plan(
    rows: list[dict[str, Any]],
    action_metrics: dict[str, Any],
    delta_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = rows[0]
    target = str(row.get("target") or "SKU")
    budget = _number(action_metrics.get("test_ad_budget_rub"))
    current_revenue = _number(row.get("current_revenue"))
    coverage = _number(row.get("coverage_days"))
    if budget is None:
        return [
            {
                "stepId": "подтвердить-рекламный-лимит",
                "area": "marketing",
                "operation": "подтвердить_рекламный_лимит",
                "instruction": (
                    f"По SKU {target} получить текущий бюджет кампании и evidence-based safe_growth_pct; "
                    "до этого лимит не повышать."
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "currentMetric": (
                    f"выручка за 7 дней={current_revenue:.0f} ₽; покрытие={coverage:.0f} дней"
                    if current_revenue is not None and coverage is not None
                    else None
                ),
                "targetMetric": "current_budget_rub и safe_growth_pct подтверждены свежим рекламным источником",
                "stopCondition": "не создавать write-action без текущего бюджета и доказанного лимита изменения",
                "payload": {"status": "needs_data"},
            }
        ]
    return [
        {
            "stepId": "поднять-лимит-рекламы",
            "area": "marketing",
            "operation": "поднять_лимит_рекламы",
            "instruction": f"По SKU {target} поднять лимит рекламы до {budget:.0f} ₽ на 7 дней и не добавлять новые запросы сверх текущего ядра.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"выручка за 7 дней={current_revenue:.0f} ₽; покрытие={coverage:.0f} дней" if current_revenue is not None and coverage is not None else None,
            "targetMetric": "покрытие >= 21 дня и contribution margin не ниже baseline",
            "stopCondition": "остановить увеличение лимита, если покрытие падает ниже 21 дня или маржа становится неположительной",
            "payload": {
                "budget7d": budget,
                "currentBudgetRub": _number(row.get("current_ad_budget_rub")),
                "safeGrowthPct": _number(row.get("evidence_safe_growth_pct")),
                "coverageDays": coverage,
            },
        },
        {
            "stepId": "удержать-цену-при-росте",
            "area": "guardrail",
            "operation": "удержать_цену",
            "instruction": f"По SKU {target} цену не менять, пока рост рекламы не подтвердит выручку и маржу на следующем замере.",
            "ownerAgent": "агент_цены",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "маржа > 0; покрытие >= 21 дня",
            "stopCondition": "к цене возвращаться только после завершения цикла замера рекламы",
            "payload": {"status": "hold"},
        },
    ]


def _reprice_growth_resolution_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = rows[0]
    target = str(row.get("target") or "SKU")
    candidate_price = _candidate_price(row)
    current_price = _number(row.get("current_price"))
    expected_profit = _number(row.get("expected_net_profit_7d"))
    blockers = {str(item) for item in row.get("safety_blockers") or [] if item}
    blocked_price_test = any(
        blocker in blockers
        for blocker in {"supply_risk", "stale_price_decision", "low_price_decision_confidence", "thin_price_data"}
    )
    return [
        {
            "stepId": "подготовить-ценовой-тест" if blocked_price_test else "изменить-цену",
            "area": "price",
            "operation": "подготовить_тест_цены" if blocked_price_test else "изменить_цену",
            "instruction": (
                (
                    f"По SKU {target} подготовить ценовой тест: после снятия блокеров изменить цену с {current_price:.0f} ₽ до {candidate_price:.0f} ₽."
                    if candidate_price is not None and current_price is not None
                    else f"По SKU {target} подготовить ценовой тест после снятия блокеров по цене и остатку."
                )
                if blocked_price_test
                else (
                    f"По SKU {target} изменить цену с {current_price:.0f} ₽ до {candidate_price:.0f} ₽."
                    if candidate_price is not None and current_price is not None
                    else f"По SKU {target} изменить цену до расчётного безопасного коридора."
                )
            ),
            "ownerAgent": "агент_цены",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"текущая цена={current_price:.0f} ₽" if current_price is not None else None,
            "targetMetric": (
                "снять блокеры по остатку и обновить ценовой сигнал до запуска теста"
                if blocked_price_test
                else f"ожидаемая прибыль за 7 дней >= {expected_profit:.0f} ₽" if expected_profit is not None else "заказы и прибыль выше текущей базы"
            ),
            "stopCondition": (
                "не запускать ценовой тест, пока не снят блокер по остатку или confidence ценового сигнала"
                if blocked_price_test
                else "откатить изменение, если прибыль или заказы не растут на следующем замере"
            ),
            "payload": {
                "currentPrice": current_price,
                "candidatePrice": candidate_price,
                "expectedNetProfit7d": expected_profit,
                "blockedPriceTest": blocked_price_test,
                "blockers": sorted(blockers),
            },
        },
        {
            "stepId": "заморозить-рост-рекламы",
            "area": "guardrail",
            "operation": "заморозить_расширение_кампании",
            "instruction": f"По SKU {target} не расширять рекламу, пока новая цена не покажет рост заказов и прибыли.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "заказы за 7 дней > база; прибыль за 7 дней > база",
            "stopCondition": "снимать заморозку только после подтверждения эффекта новой цены",
            "payload": {"status": "freeze"},
        },
    ]


def _reorder_growth_resolution_plan(rows: list[dict[str, Any]], action_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = rows[0]
    target = str(row.get("target") or "SKU")
    reorder_qty = _number(action_metrics.get("reorder_qty")) or _number(row.get("reorder_qty")) or 0
    coverage = _number(row.get("coverage_days"))
    if reorder_qty <= 0:
        return [
            {
                "stepId": "оставить-остаток-без-дозаказа",
                "area": "supply",
                "operation": "удержать_остаток_без_дозаказа",
                "instruction": f"По SKU {target} не оформлять дозаказ; текущий остаток пока достаточен.",
                "ownerAgent": "агент_поставки",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "currentMetric": f"покрытие={coverage:.0f} дней" if coverage is not None else None,
                "targetMetric": "покрытие >= 21 дня без нового дефицита",
                "stopCondition": "возвращаться к дозаказу только при новом сигнале дефицита или падении покрытия ниже 21 дня",
                "payload": {"reorderQty": 0, "coverageDays": coverage},
            },
            {
                "stepId": "заморозить-рост-до-нового-дефицита",
                "area": "guardrail",
                "operation": "заморозить_расширение_кампании",
                "instruction": f"По SKU {target} не расширять рекламу, пока не появится новый дефицит или новый сигнал по спросу.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "targetMetric": "покрытие >= 21 дня; новый дефицит не подтверждён",
                "stopCondition": "снимать заморозку только после нового сигнала дефицита или отдельного ростового сценария",
                "payload": {"status": "freeze"},
            },
        ]
    if not row.get("supplier_inbound_confirmed"):
        return [
            {
                "stepId": "проверить-входящие-поставки",
                "area": "supply",
                "operation": "проверить_входящие_поставки",
                "instruction": (
                    f"По SKU {target} подтвердить SKU-level объём входящей поставки поставщика; "
                    "in_way_to_client/in_way_from_client в расчёт не включать."
                ),
                "ownerAgent": "агент_поставки",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "currentMetric": f"покрытие={coverage:.0f} дней" if coverage is not None else None,
                "targetMetric": "supplier_inbound_qty подтверждён по SKU",
                "stopCondition": "не утверждать закупку до сверки входящей поставки",
                "payload": {
                    "conservativeReorderPreview": reorder_qty,
                    "customerTransitExcluded": True,
                },
            },
            {
                "stepId": "подготовить-дозаказ",
                "area": "supply",
                "operation": "подготовить_дозаказ",
                "instruction": (
                    f"Подготовить по SKU {target} консервативный черновик {reorder_qty:.0f} шт.; "
                    "финальный объём определить после вычета подтверждённого supplier inbound."
                ),
                "ownerAgent": "агент_поставки",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "targetMetric": "покрытие >= 21 дня после учёта входящей поставки",
                "stopCondition": "не переводить черновик в заказ без подтверждённого supplier inbound",
                "payload": {"reorderQty": reorder_qty, "previewOnly": True, "coverageDays": coverage},
            },
        ]
    return [
        {
            "stepId": "дозаказать-ростовой-остаток",
            "area": "supply",
            "operation": "дозаказать_остаток",
            "instruction": f"По SKU {target} оформить дозаказ на {reorder_qty:.0f} шт.",
            "ownerAgent": "агент_поставки",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"покрытие={coverage:.0f} дней" if coverage is not None else None,
            "targetMetric": "покрытие >= 21 дня",
            "stopCondition": "не расширять спрос, пока поставка не отражена в sellable stock",
            "payload": {"reorderQty": reorder_qty, "coverageDays": coverage},
        },
        {
            "stepId": "удержать-рекламу-до-поставки",
            "area": "guardrail",
            "operation": "заморозить_расширение_кампании",
            "instruction": f"По SKU {target} не расширять рекламу до прихода дозаказа в продаваемый остаток.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "sellable stock > 0; покрытие >= 21 дня",
            "stopCondition": "снимать заморозку после появления дозаказа в остатке и повторного расчёта покрытия",
            "payload": {"status": "freeze"},
        },
    ]


def _overstock_growth_resolution_plan(rows: list[dict[str, Any]], action_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = rows[0]
    target = str(row.get("target") or "SKU")
    discount = _number(action_metrics.get("safe_discount_pct"))
    available_stock = _number(row.get("available_stock"))
    coverage = _number(row.get("coverage_days"))
    if discount is None:
        return [
            {
                "stepId": "получить-ценовой-сценарий",
                "area": "price",
                "operation": "получить_ценовой_сценарий",
                "instruction": (
                    f"По SKU {target} получить свежий price decision с конкретной ценой, скидкой и маржой; "
                    "до этого распродажу не запускать."
                ),
                "ownerAgent": "агент_цены",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "currentMetric": (
                    f"остаток={available_stock:.0f}; покрытие={coverage:.0f} дней"
                    if available_stock is not None and coverage is not None
                    else None
                ),
                "targetMetric": "recommended_price, discount_pct и margin_after_discount подтверждены",
                "stopCondition": "не создавать price write-action без подтверждённой экономики скидки",
                "payload": {"status": "needs_data"},
            }
        ]
    return [
        {
            "stepId": "запустить-скидку",
            "area": "price",
            "operation": "запустить_скидку",
            "instruction": f"По SKU {target} запустить скидку {discount:.0f}% на 7 дней.",
            "ownerAgent": "агент_цены",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"остаток={available_stock:.0f}; покрытие={coverage:.0f} дней" if available_stock is not None and coverage is not None else None,
            "targetMetric": "сократить покрытие и ускорить оборачиваемость без ухода в отрицательную маржу",
            "stopCondition": "остановить скидку, если маржа становится неположительной",
            "payload": {"discountPct": discount, "availableStock": available_stock, "coverageDays": coverage},
        },
        {
            "stepId": "не-расширять-рекламу-на-плохой-скидке",
            "area": "guardrail",
            "operation": "удержать_рекламу",
            "instruction": f"По SKU {target} не расширять рекламу поверх скидки, пока не подтверждена положительная маржа после дисконта.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "маржа после скидки > 0",
            "stopCondition": "расширять рекламу только после подтверждения маржи и оборота",
            "payload": {"status": "hold"},
        },
    ]


def _normquery_resolution_plan(rows: list[dict[str, Any]], recommendation: str) -> list[dict[str, Any]]:
    row = rows[0] if rows else {}
    marketing_action = row.get("marketing_action") if isinstance(row.get("marketing_action"), dict) else {}
    source = row.get("source_facts") if isinstance(row.get("source_facts"), dict) else {}
    action = marketing_action or (source.get("marketing_action") if isinstance(source.get("marketing_action"), dict) else {}) or {}
    if not action:
        return _normquery_fact_resolution_plan(rows, recommendation)
    entity_id = _string(action.get("norm_query") or action.get("entity_id") or action.get("advert_id")) or "объект"
    advert_id = _string(str(action.get("advert_id"))) if action.get("advert_id") is not None else None
    query_band = _string(action.get("query_band"))
    action_mode = _string(action.get("action_mode")) or "needs_review"
    status = _string(action.get("status")) or "держать"
    reason = _string(action.get("reason")) or recommendation
    if action_mode == "reduce":
        operation = "отключить_запрос" if advert_id else "сократить_кампанию"
        instruction = (
            f"В кампании {advert_id} отключить запрос `{entity_id}`."
            if advert_id
            else f"Сократить кампанию `{entity_id}` по рекламному расходу до повторного замера."
        )
        target_metric = "расход по объекту -> 0" if advert_id else "расход кампании ниже текущей базы"
        stop_condition = "не возвращать объект в рост, пока заказы за 7 дней не подтверждены"
    else:
        operation = "заморозить_расширение_запроса" if advert_id else "заморозить_расширение_кампании"
        instruction = (
            f"В кампании {advert_id} не расширять запрос `{entity_id}` и не добавлять похожие ключи."
            if advert_id
            else f"По кампании `{entity_id}` не поднимать бюджет и не добавлять новые ключи."
        )
        target_metric = "появление заказов за 7 дней и готовность demand mart"
        stop_condition = "снимать заморозку только после появления заказов и подтверждённого спроса"
    return [
        {
            "stepId": "выполнить-рекламное-действие",
            "area": "marketing",
            "operation": operation,
            "instruction": instruction,
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "query" if advert_id else "campaign",
            "targetEntityId": entity_id,
            "advertId": advert_id,
            "normQuery": entity_id if advert_id else None,
            "currentMetric": f"статус={status}; band={query_band or 'unknown'}; причина={reason}",
            "targetMetric": target_metric,
            "stopCondition": stop_condition,
            "payload": {
                "actionMode": action_mode,
                "queryBand": query_band,
                "reason": reason,
                "status": status,
            },
        }
    ]


def _conversion_recovery_resolution_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = rows[0]
    target = str(row.get("target") or "SKU")
    product_name = _string(row.get("product_name")) or f"SKU {target}"
    visibility_row = row.get("visibility_row") if isinstance(row.get("visibility_row"), dict) else {}
    traffic_queries = _query_band_rows(row, "traffic_only")
    waste_queries = _query_band_rows(row, "waste")
    main_query = _primary_query_text(traffic_queries, waste_queries)
    title_draft = _title_rewrite_text(product_name, main_query)
    hero_block = _hero_content_block(product_name, main_query)
    return [
        {
            "stepId": "переписать-заголовок-карточки",
            "area": "card",
            "operation": "переписать_заголовок",
            "instruction": f"По SKU {target} переписать заголовок карточки с `{product_name}` на `{title_draft}`.",
            "ownerAgent": "агент_карточки",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"текущий заголовок={product_name}",
            "targetMetric": _visibility_metric_target("open_card_current", visibility_row),
            "stopCondition": "держать новый заголовок до следующего замера открытий карточки",
            "payload": {"currentTitle": product_name, "draftTitle": title_draft, "mainQuery": main_query},
        },
        {
            "stepId": "заменить-первый-экран-конверсии",
            "area": "card",
            "operation": "заменить_первый_экран",
            "instruction": (
                f"По SKU {target} заменить первый экран карточки: заголовок `{hero_block['headline']}`, "
                f"подзаголовок `{hero_block['subheadline']}`, плашки `{hero_block['badge_one']}` / `{hero_block['badge_two']}`."
            ),
            "ownerAgent": "агент_карточки",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": _funnel_snapshot(visibility_row),
            "targetMetric": _visibility_metric_target("add_to_cart_current", visibility_row),
            "stopCondition": "держать новый первый экран до роста корзины на следующем замере",
            "payload": hero_block,
        },
        {
            "stepId": "заморозить-рост-рекламы-до-конверсии",
            "area": "guardrail",
            "operation": "заморозить_расширение_кампании",
            "instruction": f"По SKU {target} не расширять рекламу, пока карточка не покажет рост открытий и добавлений в корзину.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "открытия карточки и корзина выше текущей базы",
            "stopCondition": "снимать заморозку только после роста конверсии карточки на следующем замере",
            "payload": {"status": "freeze"},
        },
    ]


def _growth_output(
    *,
    plan: ForecastPlan,
    rows: list[dict[str, Any]],
    blockers: list[str],
    decision_status: str,
    confidence: str,
    baseline_metrics: dict[str, Any],
    action_metrics: dict[str, Any],
    delta_metrics: dict[str, Any],
    summary: str,
    recommendation: str,
    why: str,
    margin_bridge: dict[str, Any] | None = None,
    effect_narrative: str | None = None,
) -> dict[str, Any]:
    first_opportunity = next((row.get("opportunity") for row in rows if isinstance(row.get("opportunity"), dict)), {})
    first_row = next((row for row in rows if isinstance(row, dict)), {})
    score = _number(first_opportunity.get("score")) if isinstance(first_opportunity, dict) else None
    growth_reason = _string(first_opportunity.get("growth_reason") or first_opportunity.get("evidence_summary")) if isinstance(first_opportunity, dict) else None
    resolution_plan = _growth_resolution_plan(
        plan=plan,
        rows=rows,
        recommendation=recommendation,
        baseline_metrics=baseline_metrics,
        action_metrics=action_metrics,
        delta_metrics=delta_metrics,
    )
    resolution_actions = _growth_resolution_actions(
        plan=plan,
        rows=rows,
        recommendation=recommendation,
        baseline_metrics=baseline_metrics,
        action_metrics=action_metrics,
        delta_metrics=delta_metrics,
    )
    return {
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blockers,
        "baseline_metrics": baseline_metrics,
        "action_metrics": action_metrics,
        "delta_metrics": delta_metrics,
        "baseline_narrative": summary,
        "action_narrative": recommendation,
        "effect_narrative": effect_narrative
        or "Эффект считается сценарно: forecast показывает expected delta и blockers, но не выдаёт causal uplift без доказательств.",
        "decision": recommendation,
        "decision_why": why,
        "final_recommendation": recommendation,
        "summary": summary,
        "card_summary": summary,
        "card_recommendation": recommendation,
        "card_expected_effect": effect_narrative
        or "Прогноз эффекта считается по guardrail-метрикам сценария и не опирается на абстрактные рассуждения.",
        "resolution_actions": resolution_actions,
        "resolution_plan": resolution_plan,
        "per_target_evidence": rows,
        "margin_bridge": margin_bridge or {},
        "opportunity_score": score,
        "growth_reason": growth_reason or summary,
        "safety_blockers": blockers,
        "recommended_action_family": plan.action_family,
        "marketing_decision_hint": _string(first_row.get("marketing_decision_hint")),
        "marketing_confidence": _string(first_row.get("marketing_confidence")),
        "safe_growth_pct": _number(first_row.get("safe_growth_pct")),
        "winner_query_share": _number(first_row.get("winner_query_share")),
        "waste_query_share": _number(first_row.get("waste_query_share")),
        "query_band": _string(first_row.get("query_band")),
        "marketing_source_status": _string(first_row.get("marketing_source_status")),
        "highlight_metrics": _metric_cards(
            [
                ("Сила сигнала", _plain_number(score) if score is not None else None),
                ("Блокеры", _plain_number(len(blockers)) if blockers else None),
                ("Сценарий", _action_family_label(plan.action_family)),
            ]
        ),
    }


def _target_match(bundle: ForecastEvidenceBundle, worker_name: str, target_id: str) -> dict[str, Any]:
    worker = bundle.workers.get(worker_name) or {}
    for match in worker.get("entity_matches") or []:
        if isinstance(match, dict) and str(match.get("target_id") or "") == target_id:
            return match.get("facts") or {}
    return bundle.per_target.get(target_id) or {}


def _campaign_resolution_actions(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for row in rows[:5]:
        target = str(row.get("target") or "кампания")
        decision = str(row.get("decision") or "hold")
        if decision == "stop_or_reduce":
            actions.append(
                {
                    "area": "marketing",
                    "action": f"По кампании {target} сократить расход или остановить показ до повторного замера заказов.",
                    "reason": str(row.get("why") or ""),
                    "expectedMetric": "расход ниже текущей базы без роста потери заказов",
                }
            )
        else:
            actions.append(
                {
                    "area": "guardrail",
                    "action": f"По кампании {target} не поднимать бюджет и не расширять ключи до нового сигнала по заказам.",
                    "reason": str(row.get("why") or ""),
                    "expectedMetric": "заказы за 7 дней подтверждены до расширения",
                }
            )
    return actions


def _campaign_resolution_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:5], start=1):
        target = str(row.get("target") or "кампания")
        decision = str(row.get("decision") or "hold")
        spend = _number(row.get("spend"))
        orders = _number(row.get("orders"))
        if decision == "stop_or_reduce":
            plan.append(
                {
                    "stepId": f"сократить-кампанию-{index}",
                    "area": "marketing",
                    "operation": "сократить_кампанию",
                    "instruction": f"По кампании {target} сократить расход или остановить показ до повторного замера заказов.",
                    "ownerAgent": "агент_маркетинга",
                    "targetEntityType": "campaign",
                    "targetEntityId": target,
                    "advertId": target,
                    "currentMetric": f"расход за 7 дней={spend:.0f} ₽; рекламные заказы за 7 дней={orders:.0f}" if spend is not None and orders is not None else str(row.get("why") or ""),
                    "targetMetric": "расход ниже текущей базы без потери подтверждённых заказов",
                    "stopCondition": "возвращать кампанию к росту только после подтверждения заказов и эффективности",
                    "payload": {"spend7d": spend, "adOrders7d": orders},
                }
            )
        else:
            plan.append(
                {
                    "stepId": f"заморозить-кампанию-{index}",
                    "area": "guardrail",
                    "operation": "заморозить_расширение_кампании",
                    "instruction": f"По кампании {target} не поднимать бюджет и не добавлять новые ключи до нового сигнала по заказам.",
                    "ownerAgent": "агент_маркетинга",
                    "targetEntityType": "campaign",
                    "targetEntityId": target,
                    "advertId": target,
                    "currentMetric": f"расход за 7 дней={spend:.0f} ₽; рекламные заказы за 7 дней={orders:.0f}" if spend is not None and orders is not None else str(row.get("why") or ""),
                    "targetMetric": "заказы за 7 дней подтверждены до снятия заморозки",
                    "stopCondition": "снимать заморозку только после нового положительного сигнала по эффективности",
                    "payload": {"spend7d": spend, "adOrders7d": orders, "status": "freeze"},
                }
            )
    return plan


def _sales_trend_resolution_actions(rows: list[dict[str, Any]], trend: str) -> list[dict[str, str]]:
    if trend != "decline":
        return [
            {
                "area": "guardrail",
                "action": "Не менять цену и рекламу без нового сигнала по продажам.",
                "reason": "Просадка не подтверждена.",
                "expectedMetric": "новый отрицательный тренд до изменения сценария",
            }
        ]
    targets = ", ".join(str(item.get("target") or "SKU") for item in rows[:3])
    return [
        {
            "area": "card",
            "action": f"По SKU {targets} перепаковать карточку и первый экран под возврат открытий и корзины.",
            "reason": "При падении продаж первым делом нужно убрать барьер в воронке карточки.",
            "expectedMetric": "рост открытий карточки и корзины на следующем замере",
        },
        {
            "area": "guardrail",
            "action": f"По SKU {targets} не расширять рекламу и не менять цену, пока карточка не вернёт конверсию.",
            "reason": "Нельзя усиливать спрос на сломанную воронку.",
            "expectedMetric": "рост конверсии карточки до снятия заморозки",
        },
    ]


def _sales_trend_resolution_plan(rows: list[dict[str, Any]], trend: str, risk: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = str(rows[0].get("target") or "SKU")
    current_sales = _number(rows[0].get("current_sales"))
    previous_sales = _number(rows[0].get("previous_sales"))
    current_revenue = _number(rows[0].get("current_revenue"))
    previous_revenue = _number(rows[0].get("previous_revenue"))
    if trend != "decline":
        return [
            {
                "stepId": "удержать-текущий-сценарий",
                "area": "guardrail",
                "operation": "удержать_текущий_сценарий",
                "instruction": f"По SKU {target} не менять цену и рекламу без нового сигнала по продажам.",
                "ownerAgent": "агент_оркестратор",
                "targetEntityType": "sku",
                "targetEntityId": target,
                "targetMetric": "отрицательный тренд подтверждён до запуска нового сценария",
                "stopCondition": "запускать новый сценарий только после нового отрицательного тренда",
                "payload": {"status": "hold"},
            }
        ]
    return [
        {
            "stepId": "перепаковать-карточку",
            "area": "card",
            "operation": "перепаковать_карточку",
            "instruction": f"По SKU {target} перепаковать карточку и первый экран под возврат открытий и корзины.",
            "ownerAgent": "агент_карточки",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"продажи: {current_sales:.0f} vs {previous_sales:.0f}; выручка: {current_revenue:.0f} ₽ vs {previous_revenue:.0f} ₽" if None not in {current_sales, previous_sales, current_revenue, previous_revenue} else f"риск={risk}",
            "targetMetric": "рост открытий карточки и корзины на следующем замере",
            "stopCondition": "держать обновлённую карточку до следующего замера воронки",
            "payload": {"risk": risk},
        },
        {
            "stepId": "заморозить-рост-рекламы",
            "area": "guardrail",
            "operation": "заморозить_расширение_кампании",
            "instruction": f"По SKU {target} не расширять рекламу и не менять цену, пока карточка не вернёт конверсию.",
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "targetMetric": "рост конверсии карточки до снятия заморозки",
            "stopCondition": "снимать заморозку только после возврата конверсии и повторного роста продаж",
            "payload": {"status": "freeze"},
        },
    ]


def _price_guardrail_resolution_actions(primary: dict[str, Any], recommendation: str) -> list[dict[str, str]]:
    return [
        {
            "area": "price",
            "action": recommendation,
            "reason": f"ценовой статус={primary.get('price_status') or 'unknown'}",
            "expectedMetric": "заказы и прибыль выше текущей базы",
        }
    ]


def _price_guardrail_resolution_plan(primary: dict[str, Any], recommendation: str) -> list[dict[str, Any]]:
    if not primary:
        return []
    target = str(primary.get("target") or "SKU")
    current_price = _number(primary.get("current_price"))
    recommended_price = _number(primary.get("recommended_price"))
    return [
        {
            "stepId": "выполнить-ценовой-шаг",
            "area": "price",
            "operation": "изменить_цену" if recommended_price is not None else "удержать_цену",
            "instruction": recommendation,
            "ownerAgent": "агент_цены",
            "targetEntityType": "sku",
            "targetEntityId": target,
            "currentMetric": f"текущая цена={current_price:.0f} ₽" if current_price is not None else f"ценовой статус={primary.get('price_status') or 'unknown'}",
            "targetMetric": "заказы и прибыль выше текущей базы",
            "stopCondition": "откатить шаг, если новая цена не возвращает заказы или ухудшает прибыль",
            "payload": {"currentPrice": current_price, "recommendedPrice": recommended_price, "priceStatus": primary.get("price_status")},
        }
    ]


def _campaign_reason(
    weak_by_matrix: bool,
    weak_by_efficiency: bool,
    spend: float | None,
    orders: float | None,
) -> str:
    if weak_by_matrix:
        return "матрица действий подтверждает stop/reduce"
    if weak_by_efficiency:
        return "кампания выглядит слабой по efficiency snapshot"
    if (orders or 0) > 0:
        return "кампания приносит заказы и требует аккуратной сверки"
    if (spend or 0) > 0:
        return "расход есть, но слабость не подтверждена"
    return "нужна дополнительная проверка"


def _marketing_findings_text(reduce_targets: list[str], hold_targets: list[str], missing_targets: list[str]) -> str:
    parts: list[str] = []
    if reduce_targets:
        parts.append(f"Подтверждённые слабые кампании: {', '.join(reduce_targets)}.")
    if hold_targets:
        parts.append(f"Оставить под контролем: {', '.join(hold_targets)}.")
    if missing_targets:
        parts.append(f"Не нашлись campaign-level факты по: {', '.join(missing_targets)}.")
    return " ".join(parts) or "Кампании проверены, но без убедительного stop/reduce сигнала."


def _detect_weak_sku_driver(
    *,
    conversion_row: dict[str, Any],
    visibility_row: dict[str, Any],
    demand_row: dict[str, Any],
    supply_row: dict[str, Any],
    price_row: dict[str, Any],
) -> str:
    if supply_row and _supply_risk_from_row(supply_row) in {"high", "medium"}:
        return "supply_constraint"
    if price_row and _string(price_row.get("price_status")) == "above_market":
        return "price_gap"
    # The inputs are already selected by deterministic watchlists. Avoid adding
    # another fixed percentage threshold in the recommendation layer.
    if conversion_row and _number(conversion_row.get("cart_to_order_conversion")) is not None:
        return "card_conversion"
    if visibility_row and (_number(visibility_row.get("avg_position_dynamics")) or 0) < 0:
        return "visibility_drop"
    if demand_row:
        return "weak_search_conversion"
    return "sales_decline"


def _weak_sku_next_action(driver: str) -> str:
    mapping = {
        "supply_constraint": "сначала закрыть риск OOS и только потом усиливать спрос",
        "price_gap": "снизить цену в безопасный коридор и удержать маржу",
        "card_conversion": "усилить карточку и post-click конверсию",
        "visibility_drop": "вернуть позиции, видимость и качественный трафик",
        "weak_search_conversion": "пересобрать поисковую гипотезу и карточку под спрос",
        "sales_decline": "перезапустить спрос через карточку и чистку трафика",
    }
    return mapping.get(driver, "выбрать конкретный growth-шаг по SKU")


def _weak_sku_findings(
    rows: list[dict[str, Any]],
    primary_driver: str,
    supply_risks: list[str],
    price_risks: list[str],
) -> str:
    base = f"Главный барьер роста по этим SKU: {_driver_label(primary_driver)}."
    if supply_risks:
        base += f" Есть supply-риск по {', '.join(supply_risks)}."
    if price_risks:
        base += f" Цена выше рынка по {', '.join(price_risks)}."
    return base


def _weak_sku_effect_text(
    primary_driver: str,
    rows: list[dict[str, Any]],
    *,
    estimated_profit_recovery: float | None,
    estimated_sales_recovery: float | None,
    lost_revenue_sum: float,
    supply_risks: list[str],
    price_risks: list[str],
    card_risks: list[str],
) -> str:
    driver_label = _driver_label(primary_driver)
    top_target = str(rows[0]["target"]) if rows else "SKU"
    if estimated_profit_recovery is not None and estimated_profit_recovery > 0:
        sales_tail = (
            f" и вернуть до {estimated_sales_recovery:.0f} продаж за неделю"
            if estimated_sales_recovery is not None and estimated_sales_recovery > 0
            else ""
        )
        return (
            f"Прогноз эффекта: вернуть около {estimated_profit_recovery:,.0f} ₽ недельной прибыли{sales_tail} "
            f"за счет устранения барьера `{driver_label}` по SKU {top_target}."
        ).replace(",", " ")
    if primary_driver == "supply_constraint":
        return "Ожидаемое направление эффекта: защита текущих продаж от OOS; численный uplift не рассчитан без измеримой модели."
    if primary_driver == "price_gap":
        return "Ожидаемое направление эффекта: снижение ценового трения; численный эффект берётся только из подтверждённой price-модели."
    if primary_driver in {"card_conversion", "visibility_drop", "weak_search_conversion"}:
        return (
            f"Наблюдаемая недельная просадка выручки — около {lost_revenue_sum:,.0f} ₽; "
            f"какая её часть вернётся после устранения барьера `{driver_label}`, без эксперимента не доказано."
        ).replace(",", " ")
    if supply_risks or price_risks or card_risks:
        return "Ожидаемое направление эффекта: снять главный барьер спроса; численный uplift не заявляется без доказательств."
    return "Численный эффект не рассчитан: сначала нужен контролируемый замер выбранного действия."


def _weak_sku_recommendation(primary_driver: str, rows: list[dict[str, Any]]) -> str:
    lead_target = str(rows[0]["target"]) if rows else "SKU"
    lead_row = rows[0] if rows else {}
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    product_name = _string(lead_row.get("product_name")) or f"SKU {lead_target}"
    if primary_driver == "supply_constraint":
        return (
            f"Дозаказать критичный объём по SKU {lead_target} и удерживать маркетинг без расширения, "
            "пока остаток не выйдет из OOS-риска."
        )
    if primary_driver == "price_gap":
        candidate_price = _number(rows[0].get("recommended_price")) if rows else None
        if candidate_price is not None:
            return (
                f"Снизить цену по SKU {lead_target} к ориентиру {candidate_price:.0f} ₽, "
                "удерживая маржу под контролем; рекламу не масштабировать до восстановления конверсии."
            )
        return (
            f"Снизить цену по SKU {lead_target} в конкурентный коридор и удержать маржу; "
            "рекламу не масштабировать, пока цена не перестанет душить спрос."
        )
    if primary_driver in {"card_conversion", "visibility_drop", "weak_search_conversion"}:
        advert_id = _lead_advert_id(waste_queries, traffic_queries)
        waste_text = _query_names_text(waste_queries[:2]) or "нецелевые запросы"
        traffic_text = _query_names_text(traffic_queries[:2]) or "рабочее ядро спроса"
        budget_rule = _budget_guardrail_text(traffic_queries[:1])
        main_query = _primary_query_text(traffic_queries, waste_queries)
        title_draft = _title_rewrite_text(product_name, main_query)
        hero_block = _hero_content_block(product_name, main_query)
        campaign_hint = f"В кампании {advert_id} " if advert_id else ""
        return (
            f"{campaign_hint}отключить нецелевые запросы {waste_text}; "
            f"по запросам {traffic_text} оставить только режим удержания {budget_rule}; "
            f"заголовок карточки переписать в `{title_draft}` и заменить первый экран: "
            f"заголовок `{hero_block['headline']}`, подзаголовок `{hero_block['subheadline']}`; цену пока не менять."
        )
    return (
        f"Перезапустить спрос по SKU {lead_target} через чистку трафика, усиление карточки и возврат позиций; "
        "цену и рекламу менять только после возврата стабильной конверсии."
    )


def _normquery_recommendation(rows: list[dict[str, Any]]) -> str:
    row = rows[0] if rows else {}
    marketing_action = row.get("marketing_action") if isinstance(row.get("marketing_action"), dict) else {}
    source = row.get("source_facts") if isinstance(row.get("source_facts"), dict) else {}
    action = marketing_action or (source.get("marketing_action") if isinstance(source.get("marketing_action"), dict) else {}) or {}
    if not action:
        return _normquery_fact_recommendation(rows)
    entity_id = _string(action.get("norm_query") or action.get("entity_id") or action.get("advert_id")) or "объект"
    advert_id = _string(str(action.get("advert_id"))) if action.get("advert_id") is not None else None
    action_mode = _string(action.get("action_mode")) or "needs_review"
    if action_mode == "reduce":
        return (
            f"В кампании {advert_id} отключить запрос `{entity_id}`."
            if advert_id
            else f"Сократить кампанию `{entity_id}` по расходу до повторного замера."
        )
    return (
        f"В кампании {advert_id} не расширять запрос `{entity_id}` и не добавлять похожие ключи."
        if advert_id
        else f"По кампании `{entity_id}` не поднимать бюджет и не добавлять новые ключи."
    )


def _normquery_fact_recommendation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Нет рекламных фактов для исполнимого шага по ставкам и normquery."
    lead_row = rows[0]
    lead_target = str(lead_row.get("target") or "объект")
    entity_type = _string(lead_row.get("entity_type")) or "unknown"
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    campaign_row = lead_row.get("campaign_row") if isinstance(lead_row.get("campaign_row"), dict) else {}
    visibility_row = lead_row.get("visibility_row") if isinstance(lead_row.get("visibility_row"), dict) else {}

    if waste_queries or traffic_queries:
        parts: list[str] = []
        if waste_queries:
            parts.append(_waste_traffic_action_text(lead_target, waste_queries))
        if traffic_queries:
            parts.append(_traffic_only_action_text(traffic_queries))
        if visibility_row:
            parts.append(
                f"Не расширять рекламное ядро по SKU {lead_target}, пока { _position_metric_target(visibility_row).lower() }."
            )
        return " ".join(part.rstrip(".") + "." for part in parts if part).strip()

    if campaign_row:
        campaign_id = _string(campaign_row.get("advert_id")) or lead_target
        spend = _number(campaign_row.get("sum_spend") or campaign_row.get("spend_rub"))
        orders = _number(campaign_row.get("orders") or campaign_row.get("ad_orders"))
        if (orders or 0) <= 0 and (spend or 0) > 0:
            return f"По кампании {campaign_id} сократить расход и не добавлять новые ключи, пока заказы за 7 дней не появятся."
        return f"По кампании {campaign_id} не поднимать бюджет и не добавлять новые ключи до повторного замера заказов."

    if visibility_row and entity_type == "sku":
        return (
            f"По SKU {lead_target} не поднимать ставки и не расширять ключи, пока "
            f"{_position_metric_target(visibility_row).lower()}."
        )

    if entity_type == "query":
        return f"По запросу {lead_target} не расширять трафик, пока не появятся заказы и не подтвердится рабочая ставка."

    return "Нет рекламных фактов для исполнимого шага по ставкам и normquery."


def _normquery_saved_spend(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    lead_row = rows[0]
    waste_queries = _query_band_rows(lead_row, "waste")
    if not waste_queries:
        return None
    spend_saved = round(sum(_number(item.get("spend_7d")) or 0 for item in waste_queries[:3]), 2)
    return spend_saved if spend_saved > 0 else None


def _normquery_current_spend(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    lead_row = rows[0]
    query_rows = [
        item
        for item in (
            _query_band_rows(lead_row, "waste")
            + _query_band_rows(lead_row, "traffic_only")
            + _query_band_rows(lead_row, "winner")
            + _query_band_rows(lead_row, "hold")
        )
        if isinstance(item, dict)
    ]
    if query_rows:
        total_spend = round(sum(_number(item.get("spend_7d")) or 0 for item in query_rows), 2)
        if total_spend > 0:
            return total_spend
    campaign_row = lead_row.get("campaign_row") if isinstance(lead_row.get("campaign_row"), dict) else {}
    campaign_spend = _number(campaign_row.get("sum_spend") or campaign_row.get("spend_rub"))
    if campaign_spend is not None and campaign_spend > 0:
        return round(campaign_spend, 2)
    return None


def _normquery_effect_text(rows: list[dict[str, Any]]) -> str:
    spend_saved = _normquery_saved_spend(rows)
    if spend_saved and spend_saved > 0:
        return (
            f"Экономический эффект: сократить около {spend_saved:.0f} ₽ пустого рекламного расхода за 7 дней "
            "без потери подтверждённых заказов."
        )
    if rows:
        lead_target = str(rows[0].get("target") or "SKU")
        return (
            f"Экономический эффект: не наращивать лишний рекламный расход по SKU {lead_target}, "
            "пока видимость и позиция не потребуют нового расширения."
        )
    return "Экономический эффект пока не оценен."


def _merge_action_and_effect(recommendation: str, effect_text: str) -> str:
    recommendation = recommendation.strip()
    effect_text = effect_text.strip()
    if not recommendation:
        return effect_text
    if not effect_text:
        return recommendation
    return f"{recommendation} {effect_text}"


def _normquery_fact_resolution_plan(rows: list[dict[str, Any]], recommendation: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    lead_row = rows[0]
    lead_target = str(lead_row.get("target") or "объект")
    entity_type = _string(lead_row.get("entity_type")) or "unknown"
    visibility_row = lead_row.get("visibility_row") if isinstance(lead_row.get("visibility_row"), dict) else {}
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    campaign_name = _lead_campaign_name(waste_queries, traffic_queries)
    main_query = _primary_query_text(traffic_queries, waste_queries)
    plan: list[dict[str, Any]] = []

    for index, row in enumerate(waste_queries[:3], start=1):
        query = _string(row.get("norm_query")) or f"waste-query-{index}"
        plan.append(
            {
                "stepId": f"отключить-waste-запрос-{index}",
                "area": "traffic",
                "operation": "отключить_запрос",
                "instruction": f"В кампании {advert_id} отключить запрос `{query}`." if advert_id else f"Отключить запрос `{query}`.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": "расход за 7 дней -> 0",
                "stopCondition": "не возвращать запрос, пока заказы за 7 дней = 0",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "spend7d": _number(row.get("spend_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    for index, row in enumerate(traffic_queries[:2], start=1):
        query = _string(row.get("norm_query")) or f"traffic-query-{index}"
        plan.append(
            {
                "stepId": f"удерживать-query-{index}",
                "area": "traffic",
                "operation": "удерживать_запрос",
                "instruction": (
                    f"В кампании {advert_id} оставить запрос `{query}` только в режиме удержания: "
                    f"{_budget_guardrail_text([row])}, не добавлять похожие ключи и не переводить в масштабирование."
                    if advert_id
                    else f"Оставить запрос `{query}` только в режиме удержания: {_budget_guardrail_text([row])}."
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": _budget_target_metric(row),
                "stopCondition": "масштабировать запрос только после первых заказов",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "effectiveCpc7d": _number(row.get("effective_cpc_7d")),
                    "effectiveCpm7d": _number(row.get("effective_cpm_7d")),
                    "clicks7d": _number(row.get("clicks_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    if visibility_row and entity_type == "sku":
        target_metric = _position_metric_target(visibility_row)
        plan.append(
            {
                "stepId": "заморозить-рост-до-видимости",
                "area": "guardrail",
                "operation": "зафиксировать_ставки_и_ключи",
                "instruction": (
                    f"В кампании {advert_id} на 7 дней оставить текущие ставки без изменений и не добавлять новые ключи "
                    f"по SKU {lead_target}; повторно открывать расширение только если {target_metric.lower()}."
                    if advert_id
                    else (
                        f"На 7 дней оставить текущие ставки без изменений и не добавлять новые ключи по SKU {lead_target}; "
                        f"повторно открывать расширение только если {target_metric.lower()}."
                    )
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "campaign" if advert_id else "sku",
                "targetEntityId": advert_id or lead_target,
                "advertId": advert_id,
                "normQuery": main_query,
                "currentMetric": _visibility_snapshot(visibility_row),
                "targetMetric": target_metric,
                "stopCondition": (
                    "через 7 дней пересмотреть кампанию; расширение разрешать только если порог по видимости и позиции выполнен"
                ),
                "payload": {
                    "campaignName": campaign_name,
                    "mainQuery": main_query,
                    "skuId": lead_target,
                    "freezeDays": 7,
                },
            }
        )

    campaign_row = lead_row.get("campaign_row") if isinstance(lead_row.get("campaign_row"), dict) else {}
    if not plan and campaign_row:
        campaign_id = _string(campaign_row.get("advert_id")) or lead_target
        spend = _number(campaign_row.get("sum_spend") or campaign_row.get("spend_rub"))
        orders = _number(campaign_row.get("orders") or campaign_row.get("ad_orders"))
        operation = "сократить_кампанию" if (orders or 0) <= 0 and (spend or 0) > 0 else "заморозить_расширение_кампании"
        instruction = (
            f"По кампании {campaign_id} сократить расход и не добавлять новые ключи, пока заказы за 7 дней не появятся."
            if operation == "сократить_кампанию"
            else f"По кампании {campaign_id} не поднимать бюджет и не добавлять новые ключи до повторного замера заказов."
        )
        plan.append(
            {
                "stepId": "удержать-кампанию",
                "area": "marketing",
                "operation": operation,
                "instruction": instruction,
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "campaign",
                "targetEntityId": campaign_id,
                "advertId": campaign_id,
                "currentMetric": _query_metric_snapshot(campaign_row),
                "targetMetric": "заказы за 7 дней >= 1 до расширения",
                "stopCondition": "возвращаться к росту только после появления заказов",
                "payload": dict(campaign_row),
            }
        )

    if not plan and entity_type == "query":
        plan.append(
            {
                "stepId": "заморозить-query",
                "area": "guardrail",
                "operation": "заморозить_расширение_запроса",
                "instruction": f"По запросу `{lead_target}` не расширять трафик, пока не появятся заказы и не подтвердится рабочая ставка.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": lead_target,
                "normQuery": lead_target,
                "targetMetric": "заказы за 7 дней >= 1",
                "stopCondition": "снимать заморозку только после появления заказов и рабочего CPC/CPM",
                "payload": {"reason": "missing_query_action"},
            }
        )

    if not plan and recommendation:
        plan.append(
            {
                "stepId": "удержать-рекламное-ядро",
                "area": "decision",
                "operation": "удержать_текущее_ядро",
                "instruction": recommendation,
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": entity_type if entity_type in {"sku", "campaign", "query"} else "unknown",
                "targetEntityId": lead_target,
                "stopCondition": "повторить решение после появления query-level фактов",
                "payload": {"reason": "missing_marketing_action"},
            }
        )
    return plan


def _conversion_recovery_recommendation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Нет данных по SKU для восстановления конверсии."
    row = rows[0]
    target = str(row.get("target") or "SKU")
    product_name = _string(row.get("product_name")) or f"SKU {target}"
    visibility_row = row.get("visibility_row") if isinstance(row.get("visibility_row"), dict) else {}
    traffic_queries = _query_band_rows(row, "traffic_only")
    waste_queries = _query_band_rows(row, "waste")
    main_query = _primary_query_text(traffic_queries, waste_queries)
    title_draft = _title_rewrite_text(product_name, main_query)
    hero_block = _hero_content_block(product_name, main_query)
    return (
        f"По SKU {target} переписать заголовок карточки в `{title_draft}` и заменить первый экран: "
        f"заголовок `{hero_block['headline']}`, подзаголовок `{hero_block['subheadline']}`; "
        "рекламу не расширять до роста открытий карточки и корзины."
    )


def _weak_sku_if_action_text(primary_driver: str, rows: list[dict[str, Any]]) -> str:
    top_targets = ", ".join(item["target"] for item in rows[:3])
    if primary_driver == "supply_constraint":
        return f"После закрытия OOS-риска по SKU {top_targets} продажи можно восстанавливать без потери доступности."
    if primary_driver == "price_gap":
        return f"После приведения цены по SKU {top_targets} в рабочий коридор конверсия и net profit должны начать восстанавливаться."
    return f"После исправления карточки и чистки трафика по SKU {top_targets} воронка должна вернуть продажи без лишнего расхода на маркетинг."


def _weak_sku_resolution_actions(primary_driver: str, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    lead_target = str(rows[0]["target"]) if rows else "SKU"
    lead_row = rows[0] if rows else {}
    product_name = _string(lead_row.get("product_name")) or f"SKU {lead_target}"
    visibility_row = lead_row.get("visibility_row") if isinstance(lead_row.get("visibility_row"), dict) else {}
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    if primary_driver == "supply_constraint":
        return [
            {
                "area": "supply",
                "action": f"Дозаказать критичный объём по SKU {lead_target} до выхода покрытия выше 21 дня.",
                "reason": "Без sellable stock рост спроса только ускорит OOS.",
                "expectedMetric": "coverage_days / oos_risk",
            },
            {
                "area": "marketing_guardrail",
                "action": "Не расширять рекламу, пока остаток не стабилизирован.",
                "reason": "Маркетинг не должен усиливать дефицит.",
                "expectedMetric": "oos_risk / ad_spend_efficiency",
            },
        ]
    if primary_driver == "price_gap":
        candidate_price = _number(rows[0].get("recommended_price")) if rows else None
        price_text = f"{candidate_price:.0f} ₽" if candidate_price is not None else "безопасному конкурентному коридору"
        return [
            {
                "area": "price",
                "action": f"Снизить цену по SKU {lead_target} к {price_text}.",
                "reason": "Текущая цена тормозит конверсию быстрее, чем приносит дополнительную маржу.",
                "expectedMetric": "orders / net_profit",
            },
            {
                "area": "guardrail",
                "action": "Не масштабировать рекламу до возврата стабильной конверсии после ценового шага.",
                "reason": "Нельзя усиливать платный трафик на SKU с ценовым трением.",
                "expectedMetric": "cart_to_order / drr",
            },
        ]

    default_actions = [
        {
            "area": "waste_traffic",
            "action": _waste_traffic_action_text(lead_target, waste_queries),
            "reason": "Некачественный трафик сжигает бюджет без возврата продаж.",
            "expectedMetric": "расход на нецелевые запросы / доля качественного трафика",
        },
        {
            "area": "traffic_only",
            "action": _traffic_only_action_text(traffic_queries),
            "reason": "Трафик без заказов нельзя масштабировать, пока карточка не возвращает конверсию.",
            "expectedMetric": "клики в заказы / эффективность расхода",
        },
        {
            "area": "ctr",
            "action": _ctr_action_text(product_name, visibility_row, waste_queries, traffic_queries),
            "reason": "Нужно поднять кликабельность и открытия карточки из выдачи по целевому поисковому интенту.",
            "expectedMetric": _visibility_metric_target("open_card_current", visibility_row),
        },
        {
            "area": "atc",
            "action": _atc_action_text(product_name, visibility_row),
            "reason": "Нужно поднять добавления в корзину и не терять трафик после открытия карточки.",
            "expectedMetric": _visibility_metric_target("add_to_cart_current", visibility_row),
        },
        {
            "area": "positions",
            "action": _positions_action_text(lead_target, visibility_row, waste_queries, traffic_queries),
            "reason": "Потеря позиций режет органический спрос и снижает объём входящего трафика.",
            "expectedMetric": _position_metric_target(visibility_row),
        },
        {
            "area": "guardrail",
            "action": "Цену не менять и не расширять рекламное ядро, пока запросы без заказов не дадут первые продажи.",
            "reason": "Сначала нужно починить воронку, а потом усиливать спрос.",
            "expectedMetric": "конверсия из корзины в заказ / восстановление выручки",
        },
    ]
    if primary_driver in {"card_conversion", "visibility_drop", "weak_search_conversion", "sales_decline"}:
        return default_actions
    return default_actions


def _query_band_rows(row: dict[str, Any], band: str) -> list[dict[str, Any]]:
    query_rows = row.get("query_rows")
    if not isinstance(query_rows, list):
        return []
    return [
        item for item in query_rows
        if isinstance(item, dict) and _string(item.get("query_band")) == band
    ]


def _query_names_text(rows: list[dict[str, Any]]) -> str:
    names = [f"`{name}`" for name in (_string(row.get("norm_query")) for row in rows) if name]
    if not names:
        return ""
    return ", ".join(names)


def _waste_traffic_action_text(lead_target: str, waste_queries: list[dict[str, Any]]) -> str:
    if not waste_queries:
        return f"Отключить нецелевые запросы и объявления без заказов по SKU {lead_target}."
    top = waste_queries[:3]
    queries = _query_names_text(top)
    spend = sum(_number(item.get("spend_7d")) or 0 for item in top)
    advert_id = _lead_advert_id(waste_queries)
    advert_hint = f" в кампании {advert_id}" if advert_id else ""
    return (
        f"Отключить{advert_hint} нецелевые запросы {queries}; "
        f"сейчас они уже сожгли {spend:.0f} ₽ без заказов за 7 дней."
    )


def _traffic_only_action_text(traffic_queries: list[dict[str, Any]]) -> str:
    if not traffic_queries:
        return "Оставить запросы без заказов без расширения бюджета и не добавлять похожие ключи, пока не появятся первые заказы."
    top = traffic_queries[:2]
    queries = _query_names_text(top)
    spend = sum(_number(item.get("spend_7d")) or 0 for item in top)
    clicks = sum(_number(item.get("clicks_7d")) or 0 for item in top)
    advert_id = _lead_advert_id(traffic_queries)
    advert_hint = f"В кампании {advert_id} " if advert_id else ""
    budget_rule = _budget_guardrail_text(top)
    return (
        f"{advert_hint}по запросам {queries} оставить только режим удержания {budget_rule}; "
        f"они дали {clicks:.0f} кликов и {spend:.0f} ₽ расхода без заказов."
    )


def _ctr_action_text(
    product_name: str,
    visibility_row: dict[str, Any],
    waste_queries: list[dict[str, Any]],
    traffic_queries: list[dict[str, Any]],
) -> str:
    main_query = _primary_query_text(traffic_queries, waste_queries)
    open_cards = _number(visibility_row.get("open_card_current"))
    visibility = _number(visibility_row.get("visibility_current"))
    query_hint = f" под запрос `{main_query}`" if main_query else ""
    metric_hint = ""
    if visibility is not None and open_cards is not None:
        metric_hint = f" Сейчас видимость {visibility:.0f}, открытия карточки {open_cards:.0f}."
    title_draft = _title_rewrite_text(product_name, main_query)
    return (
        f"Переписать заголовок карточки `{product_name}`{query_hint} в `{title_draft}` и вынести ключевой интент в первые 70 символов."
        f"{metric_hint}"
    )


def _atc_action_text(product_name: str, visibility_row: dict[str, Any]) -> str:
    add_to_cart = _number(visibility_row.get("add_to_cart_current"))
    cart_to_order = _number(visibility_row.get("cart_to_order_current"))
    metric_hint = ""
    if add_to_cart is not None:
        metric_hint = f" Сейчас добавления в корзину {add_to_cart:.0f}"
        if cart_to_order is not None:
            metric_hint += f", конверсия корзина->заказ {cart_to_order:.0f}"
        metric_hint += "."
    hero_block = _hero_content_block(product_name, None)
    return (
        f"Заменить первый экран карточки `{product_name}`: заголовок `{hero_block['headline']}`, подзаголовок `{hero_block['subheadline']}` "
        f"и добавить ниже 3 секции: `{hero_block['usage_line']}`, `{hero_block['result_line']}`, `{hero_block['trust_line']}`."
        f"{metric_hint}"
    )


def _positions_action_text(
    lead_target: str,
    visibility_row: dict[str, Any],
    waste_queries: list[dict[str, Any]],
    traffic_queries: list[dict[str, Any]],
) -> str:
    avg_position = _number(visibility_row.get("avg_position_current"))
    visibility = _number(visibility_row.get("visibility_current"))
    best_query = _primary_query_text(traffic_queries, waste_queries)
    core_queries = _query_names_text((traffic_queries or waste_queries)[:2])
    advert_id = _lead_advert_id(traffic_queries, waste_queries)
    target_hint = f" в кампании {advert_id}" if advert_id else ""
    metric_hint = ""
    if avg_position is not None and visibility is not None:
        metric_hint = f" Сейчас средняя позиция {avg_position:.0f}, видимость {visibility:.0f}."
    return (
        f"После чистки waste оставить{target_hint} рабочее ядро {core_queries or f'по SKU {lead_target}'} "
        f"и не расширять остальные запросы без заказов; контроль по ядру `{best_query}`."
        f"{metric_hint}"
    )


def _weak_sku_resolution_plan(primary_driver: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lead_target = str(rows[0]["target"]) if rows else "SKU"
    lead_row = rows[0] if rows else {}
    product_name = _string(lead_row.get("product_name")) or f"SKU {lead_target}"
    visibility_row = lead_row.get("visibility_row") if isinstance(lead_row.get("visibility_row"), dict) else {}
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    plan: list[dict[str, Any]] = []

    if primary_driver == "supply_constraint":
        return [
            {
                "stepId": "дозаказать-остаток",
                "area": "supply",
                "operation": "дозаказать_остаток",
                "instruction": f"Дозаказать по SKU {lead_target} объём до выхода покрытия выше 21 дня.",
                "ownerAgent": "агент_поставки",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": _coverage_snapshot(lead_row.get("supply_row") or {}),
                "targetMetric": "покрытие > 21 дня",
                "stopCondition": "после выхода из риска отсутствия товара можно возвращать спрос",
                "payload": dict(lead_row.get('supply_row') or {}),
            }
        ]
    if primary_driver == "price_gap":
        candidate_price = _number(rows[0].get("recommended_price")) if rows else None
        return [
            {
                "stepId": "снизить-цену",
                "area": "price",
                "operation": "снизить_цену",
                "instruction": f"Снизить цену по SKU {lead_target} до {candidate_price:.0f} ₽." if candidate_price is not None else f"Снизить цену по SKU {lead_target} в конкурентный коридор.",
                "ownerAgent": "агент_цены",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "targetMetric": "заказы выше базы без провала по марже",
                "stopCondition": "рекламу не масштабировать до возврата конверсии",
                "payload": dict(lead_row.get('price_row') or {}),
            }
        ]

    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    campaign_name = _lead_campaign_name(waste_queries, traffic_queries)
    main_query = _primary_query_text(traffic_queries, waste_queries)
    title_draft = _title_rewrite_text(product_name, main_query)
    hero_block = _hero_content_block(product_name, main_query)
    core_queries = (traffic_queries or waste_queries)[:2]

    for index, row in enumerate(waste_queries[:3], start=1):
        query = _string(row.get("norm_query")) or f"waste-query-{index}"
        plan.append(
            {
                "stepId": f"отключить-нецелевой-запрос-{index}",
                "area": "traffic",
                "operation": "отключить_запрос",
                "instruction": f"В кампании {advert_id} отключить запрос `{query}`." if advert_id else f"Отключить запрос `{query}`.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": "расход за 7 дней -> 0",
                "stopCondition": "не возвращать в кампанию, пока запрос остаётся нецелевым и заказов за 7 дней = 0",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "spend7d": _number(row.get("spend_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    for index, row in enumerate(traffic_queries[:2], start=1):
        query = _string(row.get("norm_query")) or f"traffic-query-{index}"
        plan.append(
            {
                "stepId": f"удерживать-запрос-{index}",
                "area": "traffic",
                "operation": "удерживать_запрос",
                "instruction": (
                    f"В кампании {advert_id} оставить запрос `{query}` только в режиме удержания: "
                    f"{_budget_guardrail_text([row])}, не добавлять похожие ключи и не переводить в масштабирование."
                    if advert_id
                    else f"Оставить запрос `{query}` только в режиме удержания: {_budget_guardrail_text([row])}."
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": _budget_target_metric(row),
                "stopCondition": "масштабировать только после первых заказов по запросу",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "effectiveCpc7d": _number(row.get("effective_cpc_7d")),
                    "effectiveCpm7d": _number(row.get("effective_cpm_7d")),
                    "clicks7d": _number(row.get("clicks_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    plan.extend(
        [
            {
                "stepId": "переписать-заголовок",
                "area": "card",
                "operation": "переписать_заголовок",
                "instruction": f"Переписать заголовок карточки SKU {lead_target}: было `{product_name}`, сделать `{title_draft}`.",
                "ownerAgent": "агент_карточки",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": f"текущий заголовок: {product_name}",
                "targetMetric": f"новый заголовок: {title_draft}",
                "stopCondition": "оставить новый заголовок минимум на один цикл замера открытий карточки",
                "payload": {
                    "currentTitle": product_name,
                    "draftTitle": title_draft,
                    "mainQuery": main_query,
                },
            },
            {
                "stepId": "заменить-первый-экран",
                "area": "card",
                "operation": "заменить_первый_экран",
                "instruction": (
                    f"Заменить первый экран карточки: заголовок `{hero_block['headline']}`, "
                    f"подзаголовок `{hero_block['subheadline']}`, плашки `{hero_block['badge_one']}` / `{hero_block['badge_two']}`."
                ),
                "ownerAgent": "агент_карточки",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": _visibility_snapshot(visibility_row),
                "targetMetric": _visibility_metric_target("open_card_current", visibility_row),
                "stopCondition": "держать креатив до роста открытий карточки на следующем замере",
                "payload": hero_block,
            },
            {
                "stepId": "обновить-оффер",
                "area": "card",
                "operation": "обновить_оффер",
                "instruction": (
                    f"Во 2-3 экране карточки добавить блоки `{hero_block['usage_line']}`, "
                    f"`{hero_block['result_line']}` и `{hero_block['trust_line']}`."
                ),
                "ownerAgent": "агент_продаж",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": _funnel_snapshot(visibility_row),
                "targetMetric": _visibility_metric_target("add_to_cart_current", visibility_row),
                "stopCondition": "держать изменения до роста добавлений в корзину на следующем замере",
                "payload": {
                    "usageLine": hero_block["usage_line"],
                    "resultLine": hero_block["result_line"],
                    "trustLine": hero_block["trust_line"],
                },
            },
            {
                "stepId": "зафиксировать-рабочее-ядро",
                "area": "positions",
                "operation": "зафиксировать_рабочее_ядро",
                "instruction": (
                    f"После отключения нецелевых запросов оставить в кампании {advert_id} рабочее ядро {_query_names_text(core_queries) or f'по SKU {lead_target}'} "
                    "и не расширять остальные запросы без заказов."
                    if advert_id
                    else f"После отключения нецелевых запросов оставить рабочее ядро {_query_names_text(core_queries) or f'по SKU {lead_target}'} и не расширять остальные запросы без заказов."
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "advertId": advert_id,
                "normQuery": main_query,
                "currentMetric": _visibility_snapshot(visibility_row),
                "targetMetric": _position_metric_target(visibility_row),
                "stopCondition": "не добавлять новые ключи, пока средняя позиция > 10 или видимость < 85",
                "payload": {
                    "campaignName": campaign_name,
                    "coreQueries": [_string(row.get("norm_query")) for row in core_queries if _string(row.get("norm_query"))],
                    "mainQuery": main_query,
                },
            },
            {
                "stepId": "удержать-цену",
                "area": "guardrail",
                "operation": "удержать_цену",
                "instruction": "Цену не менять и не расширять рекламное ядро, пока запросы без заказов не дадут первые продажи.",
                "ownerAgent": "агент_цены",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "stopCondition": "возвращаться к цене только после первых заказов и роста открытий карточки и корзины",
                "payload": {"status": "hold"},
            },
        ]
    )
    return plan


def _availability_resolution_actions(
    rows: list[dict[str, Any]],
    priority: list[str],
    total_reorder: float,
    min_coverage: float | None,
    *,
    reorder_executable: bool,
) -> list[dict[str, str]]:
    if not rows:
        return []
    lead_row = rows[0]
    lead_target = str(lead_row.get("target") or "SKU")
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    actions: list[dict[str, str]] = []
    if priority or total_reorder > 0:
        supply_action = (
            f"Дозаказать по SKU {lead_target} около {total_reorder:.0f} шт. и пересчитать покрытие перед любым расширением рекламы."
            if reorder_executable
            else (
                f"Сверить SKU-level входящие поставки поставщика по SKU {lead_target}; "
                f"после сверки уточнить консервативный черновик {total_reorder:.0f} шт."
            )
        )
        actions.append(
            {
                "area": "supply",
                "action": supply_action,
                "reason": (
                    "Сначала нужно убрать риск out-of-stock, иначе спрос упрётся в отсутствие товара."
                    if reorder_executable
                    else "Customer transit не является поставкой поставщика; без supplier inbound нельзя утверждать объём закупки."
                ),
                "expectedMetric": "покрытие > 21 дня / риск OOS = low",
            }
        )
        actions.append(
            {
                "area": "guardrail",
                "action": "Не расширять рекламу, пока покрытие не выйдет из зоны риска.",
                "reason": "Рост спроса на дефицитном SKU только ускоряет потерю продаж.",
                "expectedMetric": "риск OOS = low",
            }
        )
        return actions
    if waste_queries:
        actions.append(
            {
                "area": "waste_traffic",
                "action": _availability_waste_action_text(waste_queries),
                "reason": "При стабильном остатке главный следующий шаг — не разгонять waste-трафик.",
                "expectedMetric": "расход на нецелевые запросы -> 0",
            }
        )
    if traffic_queries:
        actions.append(
            {
                "area": "traffic",
                "action": _traffic_only_action_text(traffic_queries),
                "reason": "Пока заказных запросов нет, рабочее ядро можно только удерживать в текущем коридоре ставки.",
                "expectedMetric": _budget_target_metric(traffic_queries[0]),
            }
        )
    else:
        actions.append(
            {
                "area": "guardrail",
                "action": _availability_campaign_freeze_action_text(
                    lead_target=lead_target,
                    advert_id=advert_id,
                    query_rows=waste_queries,
                    min_coverage=min_coverage,
                ),
                "reason": "Пока подтверждённого заказного ядра нет, расширение кампании нужно заморозить, а не описывать общими словами.",
                "expectedMetric": _availability_campaign_freeze_target_metric(waste_queries, min_coverage),
            }
        )
    return actions


def _availability_resolution_plan(
    rows: list[dict[str, Any]],
    priority: list[str],
    total_reorder: float,
    min_coverage: float | None,
    *,
    reorder_executable: bool,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    lead_row = rows[0]
    lead_target = str(lead_row.get("target") or "SKU")
    waste_queries = _query_band_rows(lead_row, "waste")
    traffic_queries = _query_band_rows(lead_row, "traffic_only")
    advert_id = _lead_advert_id(waste_queries, traffic_queries)
    campaign_name = _lead_campaign_name(waste_queries, traffic_queries)
    plan: list[dict[str, Any]] = []
    if priority or total_reorder > 0:
        if not reorder_executable:
            plan.append(
                {
                    "stepId": "проверить-входящие-поставки",
                    "area": "supply",
                    "operation": "проверить_входящие_поставки",
                    "instruction": (
                        f"По SKU {lead_target} сверить незавершённые поставки поставщика; "
                        "не использовать in_way_to_client/in_way_from_client как входящий товар."
                    ),
                    "ownerAgent": "агент_поставки",
                    "targetEntityType": "sku",
                    "targetEntityId": lead_target,
                    "currentMetric": _availability_supply_snapshot(lead_row),
                    "targetMetric": "подтверждённый supplier_inbound_qty по SKU",
                    "stopCondition": "не утверждать дозаказ до SKU-level сверки входящей поставки",
                    "payload": {
                        "customerTransitExcluded": True,
                        "conservativeReorderPreview": total_reorder,
                    },
                }
            )
        plan.append(
            {
                "stepId": "дозаказать-остаток" if reorder_executable else "подготовить-дозаказ",
                "area": "supply",
                "operation": "дозаказать_остаток" if reorder_executable else "подготовить_дозаказ",
                "instruction": (
                    f"Дозаказать по SKU {lead_target} около {total_reorder:.0f} шт. и пересчитать покрытие."
                    if reorder_executable
                    else (
                        f"Подготовить консервативный черновик дозаказа {total_reorder:.0f} шт. по SKU {lead_target}; "
                        "финальный объём определить после вычета подтверждённой входящей поставки поставщика."
                    )
                ),
                "ownerAgent": "агент_поставки",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": _availability_supply_snapshot(lead_row),
                "targetMetric": "покрытие > 21 дня",
                "stopCondition": "к расширению спроса возвращаться только после выхода риска OOS в low",
                "payload": {
                    "reorderQty": total_reorder,
                    "previewOnly": not reorder_executable,
                    "coverageDays": _number(lead_row.get("coverage_days")),
                },
            }
        )
        plan.append(
            {
                "stepId": "удержать-масштабирование",
                "area": "guardrail",
                "operation": "удержать_масштабирование",
                "instruction": "Не расширять рекламу, пока покрытие не выйдет из зоны риска.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "sku",
                "targetEntityId": lead_target,
                "currentMetric": _availability_supply_snapshot(lead_row),
                "targetMetric": "риск OOS = low",
                "stopCondition": "масштабирование открывать только после повторной сверки остатков",
                "payload": {"status": "hold"},
            }
        )
        return plan

    for index, row in enumerate(waste_queries[:3], start=1):
        query = _string(row.get("norm_query")) or f"запрос-{index}"
        plan.append(
            {
                "stepId": f"отключить-нецелевой-запрос-{index}",
                "area": "traffic",
                "operation": "отключить_запрос",
                "instruction": f"В кампании {advert_id} отключить запрос `{query}`." if advert_id else f"Отключить запрос `{query}`.",
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": "расход за 7 дней -> 0",
                "stopCondition": "не возвращать запрос в кампанию, пока заказов за 7 дней = 0",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "spend7d": _number(row.get("spend_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    for index, row in enumerate(traffic_queries[:2], start=1):
        query = _string(row.get("norm_query")) or f"traffic-query-{index}"
        plan.append(
            {
                "stepId": f"удерживать-запрос-{index}",
                "area": "traffic",
                "operation": "удерживать_запрос",
                "instruction": (
                    f"В кампании {advert_id} оставить запрос `{query}` только в режиме удержания: "
                    f"{_budget_guardrail_text([row])}, не добавлять похожие ключи и не переводить в расширение."
                    if advert_id
                    else f"Оставить запрос `{query}` только в режиме удержания: {_budget_guardrail_text([row])}."
                ),
                "ownerAgent": "агент_маркетинга",
                "targetEntityType": "query",
                "targetEntityId": query,
                "advertId": _string(row.get("advert_id")) or advert_id,
                "normQuery": query,
                "currentMetric": _query_metric_snapshot(row),
                "targetMetric": _budget_target_metric(row),
                "stopCondition": "расширять запрос только после первого заказа за 7 дней",
                "payload": {
                    "campaignName": _string(row.get("campaign_name")) or campaign_name,
                    "paymentType": _string(row.get("payment_type")),
                    "effectiveCpc7d": _number(row.get("effective_cpc_7d")),
                    "effectiveCpm7d": _number(row.get("effective_cpm_7d")),
                    "clicks7d": _number(row.get("clicks_7d")),
                    "orders7d": _number(row.get("orders_7d")),
                },
            }
        )

    plan.append(
        {
            "stepId": "заморозить-расширение-кампании",
            "area": "guardrail",
            "operation": "заморозить_расширение_кампании",
            "instruction": _availability_campaign_freeze_action_text(
                lead_target=lead_target,
                advert_id=advert_id,
                query_rows=traffic_queries or waste_queries,
                min_coverage=min_coverage,
            ),
            "ownerAgent": "агент_маркетинга",
            "targetEntityType": "campaign" if advert_id else "sku",
            "targetEntityId": advert_id or lead_target,
            "advertId": advert_id,
            "currentMetric": _availability_supply_snapshot(lead_row),
            "targetMetric": _availability_campaign_freeze_target_metric(traffic_queries or waste_queries, min_coverage),
            "stopCondition": _availability_campaign_freeze_stop_condition(min_coverage),
            "payload": _availability_campaign_freeze_payload(
                campaign_name=campaign_name,
                query_rows=traffic_queries or waste_queries,
                min_coverage=min_coverage,
            ),
        }
    )
    return plan


def _availability_waste_action_text(waste_queries: list[dict[str, Any]]) -> str:
    advert_id = _lead_advert_id(waste_queries)
    queries = _query_names_text(waste_queries[:3])
    spend = sum(_number(item.get("spend_7d")) or 0 for item in waste_queries[:3])
    prefix = f"В кампании {advert_id} " if advert_id else ""
    return (
        f"{prefix}отключить нецелевые запросы {queries}; "
        f"сейчас они сожгли {spend:.0f} ₽ без заказов за 7 дней."
    )


def _availability_supply_snapshot(row: dict[str, Any]) -> str:
    parts: list[str] = []
    available = _number(row.get("available_stock"))
    inbound = _number(row.get("inbound_stock"))
    velocity = _number(row.get("sales_velocity"))
    coverage = _number(row.get("coverage_days"))
    risk = _string(row.get("oos_risk"))
    if available is not None:
        parts.append(f"остаток={available:.0f}")
    if inbound is not None:
        parts.append(f"в пути={inbound:.0f}")
    if velocity is not None:
        parts.append(f"скорость продаж={velocity:.1f}")
    if coverage is not None:
        parts.append(f"покрытие={coverage:.1f} дня")
    if risk:
        parts.append(f"риск OOS={risk}")
    return "; ".join(parts) or "снимок supply недоступен"


def _availability_campaign_freeze_action_text(
    *,
    lead_target: str,
    advert_id: str | None,
    query_rows: list[dict[str, Any]],
    min_coverage: float | None,
) -> str:
    limit_text, _ = _availability_campaign_limit(query_rows)
    campaign_ref = f"В кампании {advert_id}" if advert_id else f"По SKU {lead_target}"
    coverage_part = (
        f" до выхода покрытия не ниже {min_coverage:.1f} дня"
        if min_coverage is not None
        else ""
    )
    return (
        f"{campaign_ref} не добавлять новые запросы и не поднимать бюджет{limit_text}; "
        f"расширение кампании оставить замороженным{coverage_part}, пока не появится хотя бы один запрос с заказами за 7 дней."
    )


def _availability_campaign_freeze_target_metric(
    query_rows: list[dict[str, Any]],
    min_coverage: float | None,
) -> str:
    _, metric_text = _availability_campaign_limit(query_rows)
    coverage_metric = f"покрытие >= {min_coverage:.1f} дня" if min_coverage is not None else "покрытие подтверждено"
    return f"{metric_text}; {coverage_metric}; запросов с заказами за 7 дней >= 1"


def _availability_campaign_freeze_stop_condition(min_coverage: float | None) -> str:
    if min_coverage is not None:
        return (
            f"снимать заморозку только после появления хотя бы одного запроса с заказами за 7 дней "
            f"и подтверждённого покрытия не ниже {min_coverage:.1f} дня"
        )
    return "снимать заморозку только после появления хотя бы одного запроса с заказами за 7 дней и повторной сверки покрытия"


def _availability_campaign_freeze_payload(
    *,
    campaign_name: str | None,
    query_rows: list[dict[str, Any]],
    min_coverage: float | None,
) -> dict[str, Any]:
    limit_value, limit_metric = _availability_campaign_limit_value(query_rows)
    payload: dict[str, Any] = {
        "campaignName": campaign_name,
        "limitMetric": limit_metric,
        "limitValue": limit_value,
        "coverageGateDays": min_coverage,
        "status": "freeze",
    }
    return payload


def _availability_campaign_limit(query_rows: list[dict[str, Any]]) -> tuple[str, str]:
    limit_value, limit_metric = _availability_campaign_limit_value(query_rows)
    if limit_value is None or limit_metric is None:
        return (" и не расширять текущий бюджет", "бюджет кампании без расширения")
    if limit_metric == "cpm":
        return (
            f" выше {limit_value:.0f} ₽ за 1000 показов",
            f"ставка за 1000 показов <= {limit_value:.0f} ₽",
        )
    return (
        f" выше {limit_value:.0f} ₽ за клик",
        f"ставка за клик <= {limit_value:.0f} ₽",
    )


def _availability_campaign_limit_value(query_rows: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    cpm_values = sorted(
        {
            value
            for value in (_number(row.get("effective_cpm_7d")) for row in query_rows)
            if value is not None and value > 0
        }
    )
    if cpm_values:
        return cpm_values[0], "cpm"
    cpc_values = sorted(
        {
            value
            for value in (_number(row.get("effective_cpc_7d")) for row in query_rows)
            if value is not None and value > 0
        }
    )
    if cpc_values:
        return cpc_values[0], "cpc"
    return None, None


def _lead_advert_id(*query_groups: list[dict[str, Any]]) -> str | None:
    for rows in query_groups:
        for row in rows:
            advert_id = _string(str(row.get("advert_id"))) if row.get("advert_id") is not None else None
            if advert_id:
                return advert_id
    return None


def _lead_campaign_name(*query_groups: list[dict[str, Any]]) -> str | None:
    for rows in query_groups:
        for row in rows:
            campaign_name = _string(row.get("campaign_name"))
            if campaign_name:
                return campaign_name
    return None


def _primary_query_text(traffic_queries: list[dict[str, Any]], waste_queries: list[dict[str, Any]]) -> str | None:
    return _string((traffic_queries or waste_queries or [{}])[0].get("norm_query"))


def _budget_guardrail_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "без расширения бюджета"
    row = rows[0]
    payment_type = (_string(row.get("payment_type")) or "").lower()
    effective_cpm = _number(row.get("effective_cpm_7d"))
    effective_cpc = _number(row.get("effective_cpc_7d"))
    if payment_type == "cpm" and effective_cpm is not None:
        return f"со ставкой за 1000 показов не выше {effective_cpm:.0f} ₽ и без новых ключей"
    if effective_cpc is not None:
        return f"со ставкой за клик не выше {effective_cpc:.0f} ₽ и без новых ключей"
    return "без расширения бюджета и без новых ключей"


def _budget_target_metric(row: dict[str, Any]) -> str:
    payment_type = (_string(row.get("payment_type")) or "").lower()
    effective_cpm = _number(row.get("effective_cpm_7d"))
    effective_cpc = _number(row.get("effective_cpc_7d"))
    if payment_type == "cpm" and effective_cpm is not None:
        return f"ставка за 1000 показов <= {effective_cpm:.0f} ₽; заказов за 7 дней >= 1 до масштабирования"
    if effective_cpc is not None:
        return f"ставка за клик <= {effective_cpc:.0f} ₽; заказов за 7 дней >= 1 до масштабирования"
    return "заказов за 7 дней >= 1 до масштабирования"


def _title_rewrite_text(product_name: str, focus_query: str | None) -> str:
    base_title = _string(product_name) or "Товар для лица"
    if not focus_query:
        return base_title
    lower_title = base_title.lower()
    focus = focus_query.lower()
    vitamins_suffix = " с витаминами" if "витамин" in lower_title else ""
    if "спрей-сыворот" in lower_title:
        if "увлаж" in focus:
            return f"Спрей-сыворотка для увлажнения лица{vitamins_suffix}"
        return f"Спрей-сыворотка для лица{vitamins_suffix}"
    if "сыворот" in lower_title:
        if "увлаж" in focus:
            return f"Сыворотка для увлажнения лица{vitamins_suffix}"
        return f"Сыворотка для лица{vitamins_suffix}"
    if "спрей" in lower_title:
        if "увлаж" in focus:
            return f"Увлажняющий спрей для лица{vitamins_suffix}"
        return f"Спрей для лица{vitamins_suffix}"
    return _sentence_case(base_title)


def _hero_content_block(product_name: str, focus_query: str | None) -> dict[str, str]:
    base_title = _string(product_name) or "Товар для лица"
    headline = _sentence_case(focus_query or base_title)
    lower_title = base_title.lower()
    product_form = "Средство для лица"
    if "спрей-сыворот" in lower_title:
        product_form = "Спрей-сыворотка"
    elif "сыворот" in lower_title:
        product_form = "Сыворотка"
    suffix = "Для лица"
    marker = lower_title.find(" с ")
    if marker >= 0:
        suffix = _sentence_case(base_title[marker:].strip())
    badge_two = suffix
    return {
        "headline": headline,
        "subheadline": f"{product_form} {suffix.lower()}".strip(),
        "badge_one": "Увлажнение",
        "badge_two": badge_two,
        "usage_line": "Как использовать: распылить на чистую кожу лица и распределить",
        "result_line": "Что обещаем на первом экране: увлажнение и быстрый комфорт без тяжести",
        "trust_line": "Что добавляем рядом: состав/витамины и короткое социальное доказательство",
    }


def _query_metric_snapshot(row: dict[str, Any]) -> str:
    parts: list[str] = []
    spend = _number(row.get("spend_7d"))
    views = _number(row.get("views_7d"))
    clicks = _number(row.get("clicks_7d"))
    orders = _number(row.get("orders_7d"))
    avg_pos = _number(row.get("avg_pos_7d"))
    if spend is not None:
        parts.append(f"расход за 7 дней={spend:.0f} ₽")
    if views is not None:
        parts.append(f"показы за 7 дней={views:.0f}")
    if clicks is not None:
        parts.append(f"клики за 7 дней={clicks:.0f}")
    if orders is not None:
        parts.append(f"заказы за 7 дней={orders:.0f}")
    if avg_pos is not None:
        parts.append(f"средняя позиция за 7 дней={avg_pos:.0f}")
    return "; ".join(parts)


def _visibility_snapshot(visibility_row: dict[str, Any]) -> str:
    if not visibility_row:
        return "снимок видимости недоступен"
    parts: list[str] = []
    mapping = {
        "visibility_current": "видимость",
        "avg_position_current": "средняя позиция",
        "open_card_current": "открытия карточки",
    }
    for key, label in mapping.items():
        value = _number(visibility_row.get(key))
        if value is not None:
            parts.append(f"{label}={value:.0f}")
    return "; ".join(parts) or "снимок видимости недоступен"


def _funnel_snapshot(visibility_row: dict[str, Any]) -> str:
    if not visibility_row:
        return "снимок воронки недоступен"
    parts: list[str] = []
    mapping = {
        "open_card_current": "открытия карточки",
        "add_to_cart_current": "добавления в корзину",
        "cart_to_order_current": "конверсия корзина->заказ",
    }
    for key, label in mapping.items():
        value = _number(visibility_row.get(key))
        if value is not None:
            parts.append(f"{label}={value:.0f}")
    return "; ".join(parts) or "снимок воронки недоступен"


def _coverage_snapshot(supply_row: dict[str, Any]) -> str:
    if not supply_row:
        return "снимок остатков недоступен"
    parts: list[str] = []
    coverage = _number(supply_row.get("days_of_stock"))
    reorder_qty = _number(supply_row.get("reorder_qty_recommended"))
    if coverage is not None:
        parts.append(f"покрытие={coverage:.0f} дней")
    if reorder_qty is not None:
        parts.append(f"дозаказ={reorder_qty:.0f}")
    return "; ".join(parts) or "снимок остатков недоступен"


def _sentence_case(text: str | None) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]


def _visibility_metric_target(metric_key: str, visibility_row: dict[str, Any]) -> str:
    current = _number(visibility_row.get(metric_key))
    if current is None:
        return metric_key
    label = "открытия карточки" if metric_key == "open_card_current" else "добавления в корзину"
    return f"{label}: baseline {current:.0f}; target — только из измеримого эксперимента"


def _position_metric_target(visibility_row: dict[str, Any]) -> str:
    avg_position = _number(visibility_row.get("avg_position_current"))
    visibility = _number(visibility_row.get("visibility_current"))
    if avg_position is None and visibility is None:
        return "средняя позиция / видимость"
    parts: list[str] = []
    if avg_position is not None:
        parts.append(f"средняя позиция: baseline {avg_position:.0f}; target не рассчитан")
    if visibility is not None:
        parts.append(f"видимость: baseline {visibility:.0f}; target не рассчитан")
    return "; ".join(parts)


def _primary_driver(rows: list[dict[str, Any]]) -> str:
    priority = [
        "supply_constraint",
        "price_gap",
        "card_conversion",
        "visibility_drop",
        "weak_search_conversion",
        "sales_decline",
    ]
    seen = {str(item.get("driver") or "sales_decline") for item in rows}
    for driver in priority:
        if driver in seen:
            return driver
    return "sales_decline"


def _driver_label(driver: str) -> str:
    labels = {
        "supply_constraint": "ограничение по остатку",
        "price_gap": "ценовой разрыв с рынком",
        "card_conversion": "слабая конверсия карточки",
        "visibility_drop": "потеря видимости",
        "weak_search_conversion": "поисковый спрос не конвертируется",
        "sales_decline": "подтверждённая просадка продаж",
    }
    return labels.get(driver, driver)


def _action_family_label(family: str) -> str:
    labels = {
        "stop_or_reduce_campaigns": "Сокращение слабых кампаний",
        "availability_before_scale_guardrail": "Остаток перед ростом спроса",
        "weak_sku_demand_recovery": "Восстановление слабого SKU",
        "sales_trend_guardrail": "Снижение продаж",
        "price_positioning_guardrail": "Ценовой guardrail",
        "scale_profitable_sku_ads": "Рост прибыльного SKU",
        "reprice_with_margin_guardrail": "Переоценка с маржой",
        "reorder_for_growth": "Дозаказ для роста",
        "overstock_liquidation": "Распродажа overstock",
        "normquery_bid_optimization": "Оптимизация ставок и запросов",
        "recover_conversion_drop": "Восстановление конверсии",
    }
    return labels.get(family, family)


def _oos_risk(reorder_flag: str, coverage_days: float | None, sales_velocity: float) -> str:
    if sales_velocity <= 0:
        return "unknown"
    if reorder_flag == "critical" or (coverage_days is not None and coverage_days < 7):
        return "high"
    if reorder_flag == "warning" or (coverage_days is not None and coverage_days < 21):
        return "medium"
    return "low"


def _supply_risk_from_row(supply_row: dict[str, Any]) -> str | None:
    if not supply_row:
        return None
    coverage = _number(supply_row.get("days_of_stock"))
    velocity = ((_number(supply_row.get("sales_7d_units")) or 0) / 7) if supply_row.get("sales_7d_units") is not None else 0
    reorder_flag = _string(supply_row.get("reorder_flag")) or "unknown"
    return _oos_risk(reorder_flag, coverage, velocity)


def _aggregate_risk(risks: list[str]) -> str:
    normalized = set(risks)
    if "high" in normalized:
        return "high"
    if "medium" in normalized:
        return "medium"
    if "low" in normalized:
        return "low"
    return "unknown"


def _market_position_risk(price_status: str, competitor_count: float) -> str:
    if competitor_count < 5 or price_status == "thin_data":
        return "unknown"
    if price_status == "above_market":
        return "high"
    if price_status == "below_market":
        return "medium"
    return "low"


def _price_action_text(price_status: str) -> str:
    if price_status == "above_market":
        return "Проверить безопасное снижение цены после сверки маржи."
    if price_status == "below_market":
        return "Проверить возможность поднять цену или удержать текущую маржу."
    if price_status == "market_aligned":
        return "Оставить цену без изменения."
    return "Не менять цену, пока конкурентных данных недостаточно."


def _price_recommendation(price_status: str) -> str:
    if price_status == "above_market":
        return "Проверить снижение цены только после подтверждения маржи и минимально допустимой цены."
    if price_status == "below_market":
        return "Не снижать цену автоматически; проверить, можно ли поднять цену или удержать маржу."
    if price_status == "market_aligned":
        return "Оставить цену без изменения по текущему конкурентному срезу."
    return "Заблокировать изменение цены, пока competitor data остаётся тонким."


def _safe_delta(current: float | None, previous: float | None) -> float:
    return round((current or 0) - (previous or 0), 2)


def _trend_direction(delta_sales: float, delta_revenue: float) -> str:
    if delta_sales < 0 or delta_revenue < 0:
        return "decline"
    if delta_sales > 0 or delta_revenue > 0:
        return "growth"
    return "stable"


def _load_period_drr(*, today: date | None, period_days: int) -> float | None:
    if today is None:
        return None
    date_to = today.isoformat()
    date_from = (today - timedelta(days=period_days - 1)).isoformat()
    try:
        report = strategic_db.get_marketing_drr_report(date_from=date_from, date_to=date_to, limit=50)
    except Exception:
        return None
    totals = report.get("totals") if isinstance(report, dict) else None
    return _number(totals.get("estimated_drr_promoted_sku_pct")) if isinstance(totals, dict) else None


def _min_number(values: Any) -> float | None:
    numbers = [_number(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return min(clean) if clean else None


def _metric_cards(items: list[tuple[str, str | None]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for label, value in items:
        if not value:
            continue
        cards.append({"label": label, "value": value})
    return cards


def _money(value: Any) -> str | None:
    number = _number(value)
    if number is None or number == 0:
        return None
    sign = "-" if number < 0 else ""
    return f"{sign}{abs(number):,.0f} ₽".replace(",", " ")


def _plain_number(value: Any, *, suffix: str = "") -> str | None:
    number = _number(value)
    if number is None:
        return None
    if number.is_integer():
        return f"{int(number)}{suffix}"
    return f"{number:.2f}{suffix}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _supplier_inbound(row: dict[str, Any]) -> float | None:
    for key in _SUPPLIER_INBOUND_KEYS:
        if key not in row:
            continue
        value = _number(row.get(key))
        if value is not None and value >= 0:
            return value
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
