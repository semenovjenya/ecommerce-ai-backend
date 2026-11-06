from __future__ import annotations

from datetime import date, timedelta
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from agi_ecom_app.decision_simulation.pricing_engine import run_pricing_simulation
from agi_ecom_app.decision_simulation.pricing_hypotheses import (
    build_pricing_hypotheses,
)
from agi_ecom_app.decision_simulation.pricing_source import (
    load_pricing_simulation_source,
)
from agi_ecom_app.forecast.opportunity_finder import find_growth_opportunities
from agi_ecom_app.forecast.workflow import run_forecast_workflow
from agi_ecom_app.manager_decision.portfolio_optimizer import (
    allowed_action_families,
    optimize_portfolio,
)
from agi_ecom_app.services.strategic_db import (
    PORTFOLIO_ANALYSIS_LIMIT,
    get_inventory_turnover,
    get_price_decision_overview,
    get_sku_daily_price_sales_comparison,
    get_sku_identity_map,
)


def _plan_targets(query_plan: Mapping[str, Any] | None) -> list[str]:
    targets: list[str] = []
    for target in list((query_plan or {}).get("targets") or []):
        if not isinstance(target, Mapping):
            continue
        value = str(target.get("value") or "").strip()
        if value and value not in targets:
            targets.append(value)
    return targets


def _artifact_target(item: Mapping[str, Any]) -> str:
    return str(item.get("vendorCode") or item.get("target") or "").strip()


def _has_numeric_price_move(simulation: Mapping[str, Any]) -> bool:
    raw_hypotheses = simulation.get("hypothesisCandidates")
    hypothesis_candidates = (
        list(raw_hypotheses.get("candidates") or [])
        if isinstance(raw_hypotheses, Mapping)
        else list(raw_hypotheses or [])
    )
    candidates = [
        *list(simulation.get("candidates") or []),
        *hypothesis_candidates,
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("operation") == "hold_price":
            continue
        mean = candidate.get("meanEffectRub", candidate.get("mean"))
        p10 = candidate.get("p10EffectRub", candidate.get("p10"))
        stockout = candidate.get(
            "stockoutProbability",
            candidate.get("stockout"),
        )
        if all(
            _number_or_none(value) is not None
            for value in (candidate.get("priceRub"), mean, p10, stockout)
        ):
            return True
    return False


def _advisory_coverage(
    *,
    portfolio_targets: list[str],
    requested_targets: list[str],
    forecasts: list[dict[str, Any]],
    daily_comparisons: list[dict[str, Any]],
    price_simulations: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    total_targets = list(
        dict.fromkeys(portfolio_targets or requested_targets)
    )
    total_set = set(total_targets)
    artifact_targets = {
        target
        for collection in (forecasts, daily_comparisons, price_simulations)
        for item in collection
        if (target := _artifact_target(item)) in total_set
    }
    error_targets = {
        str(item.get("target") or "").strip()
        for item in errors
        if str(item.get("target") or "").strip() in total_set
    }
    covered_targets = artifact_targets | error_targets
    numeric_price_targets = {
        target
        for item in price_simulations
        if (target := _artifact_target(item)) in total_set
        and _has_numeric_price_move(item)
    }
    return {
        "targetSkuCount": len(total_targets),
        "coveredSkuCount": len(covered_targets),
        "coveredTargets": sorted(covered_targets),
        "artifactTargets": sorted(artifact_targets),
        "errorTargets": sorted(error_targets),
        "missingArtifactTargets": sorted(total_set - covered_targets),
        "numericPriceScenarioSkuCount": len(numeric_price_targets),
        "numericPriceScenarioTargets": sorted(numeric_price_targets),
        "missingNumericPriceScenarioTargets": sorted(
            total_set - numeric_price_targets
        ),
    }


def _portfolio_targets(
    query_plan: Mapping[str, Any],
    *,
    max_targets: int,
    turnover: Mapping[str, Any] | None = None,
) -> list[str]:
    if turnover is None:
        turnover = _portfolio_turnover_snapshot()
    constraints = dict(query_plan.get("constraints") or {})
    target_cover_days = float(constraints.get("target_cover_days") or 30)
    period_days = max(1, int((turnover.get("totals") or {}).get("period_days") or 30))
    ranked: dict[str, tuple[int, float, float, float, str]] = {}
    for row in list(turnover.get("rows") or []):
        target = str(row.get("vendor_code") or "").strip()
        sales = float(row.get("sales_units") or 0)
        stock = float(row.get("current_stock_qty") or 0)
        if not target:
            continue
        if sales <= 0:
            excess = max(0.0, stock)
            current_cover = float("inf") if stock > 0 else 0.0
            priority = 3 if stock > 0 else 0
        else:
            velocity = sales / period_days
            target_stock = velocity * target_cover_days
            excess = stock - target_stock
            current_cover = stock / velocity
            priority = 2 if excess > 0 else 1
        rank = (priority, excess, current_cover, stock, target)
        previous = ranked.get(target)
        if previous is None or rank > previous:
            ranked[target] = rank
    ordered = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
    return [target for target, _rank in ordered[:max_targets]]


def _portfolio_turnover_snapshot() -> dict[str, Any]:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=29)
    return get_inventory_turnover(
        date_from=start.isoformat(),
        date_to=end.isoformat(),
        vendor_codes=None,
        limit=100,
    )


def should_build_manager_advisory(
    user_query: str,
    *,
    query_plan: Mapping[str, Any] | None = None,
) -> bool:
    del user_query
    plan = dict(query_plan or {})
    analyses = {str(item) for item in list(plan.get("requested_analyses") or [])}
    asks_for_scenarios = bool(
        analyses.intersection({"forecast", "price_scenarios"})
        or plan.get("scenario_requests")
    )
    has_scope = bool(_plan_targets(plan) or plan.get("scope") == "portfolio")
    return asks_for_scenarios and has_scope


def build_manager_advisory(
    user_query: str,
    *,
    query_plan: Mapping[str, Any] | None = None,
    max_targets: int = PORTFOLIO_ANALYSIS_LIMIT,
    candidate_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del user_query
    plan = dict(query_plan or {})
    focus_targets = _plan_targets(plan)
    requested_targets = focus_targets[:max_targets]
    portfolio_targets: list[str] = []
    turnover_snapshot: dict[str, Any] | None = None
    if plan.get("scope") == "portfolio":
        turnover_snapshot = _portfolio_turnover_snapshot()
        portfolio_targets = _portfolio_targets(
            plan,
            max_targets=max_targets,
            turnover=turnover_snapshot,
        )
        # Explicit SKU targets in portfolio scope are guaranteed deep dives.
        # Fill the remaining budget with every portfolio row, ranked by the
        # urgency of zero-sales stock and excess cover. The focus list is not a
        # portfolio filter.
        requested_targets = list(
            dict.fromkeys([*focus_targets, *portfolio_targets])
        )[:max_targets]
    analyses = {str(item) for item in list(plan.get("requested_analyses") or [])}
    horizon_days = max(
        1,
        min(90, int((plan.get("constraints") or {}).get("horizon_days") or 7)),
    )
    pricing_targets = (
        set(requested_targets)
        if analyses.intersection({"pricing", "price_scenarios"})
        else set()
    )
    errors: list[dict[str, str]] = []
    price_priors: dict[str, dict[str, Any]] = {}
    if pricing_targets:
        try:
            overview = get_price_decision_overview(None, limit=max_targets)
            for row in list(overview.get("rows") or []):
                if not isinstance(row, Mapping):
                    continue
                compact_row = dict(row)
                for alias in (row.get("vendor_code"), row.get("nm_id")):
                    normalized = str(alias or "").strip()
                    if normalized:
                        price_priors[normalized] = compact_row
        except Exception as exc:
            errors.append(
                {
                    "target": "portfolio",
                    "stage": "price_decision_overview",
                    "error_type": type(exc).__name__,
                }
            )
    identities = get_sku_identity_map(requested_targets)
    opportunities = find_growth_opportunities(
        period_days=14,
        horizon_days=14,
        limit=100,
        focus="margin_revenue",
    )
    forecasts: list[dict[str, Any]] = []
    daily_comparisons: list[dict[str, Any]] = []
    price_simulations: list[dict[str, Any]] = []

    for requested_target in requested_targets:
        daily_comparison: dict[str, Any] | None = None
        identity = dict(identities.get(requested_target) or {})
        aliases = {
            requested_target,
            str(identity.get("nm_id") or "").strip(),
            str(identity.get("vendor_code") or "").strip(),
        }
        aliases.discard("")
        matching = [
            opportunity
            for opportunity in opportunities
            if aliases.intersection(str(item) for item in opportunity.affected_skus)
        ]
        if matching:
            opportunity = max(matching, key=lambda item: item.score)
            try:
                forecast = run_forecast_workflow(
                    task_snapshot=None,
                    opportunity=opportunity,
                )
                if forecast is not None:
                    forecasts.append(
                        _compact_forecast(
                            requested_target=requested_target,
                            identity=identity,
                            opportunity=opportunity.model_dump(mode="json"),
                            forecast=forecast.model_dump(mode="json", by_alias=True),
                        )
                    )
            except Exception as exc:
                errors.append(
                    {
                        "target": requested_target,
                        "stage": "forecast",
                        "error_type": type(exc).__name__,
                    }
                )

        nm_id = identity.get("nm_id")
        if requested_target not in pricing_targets:
            continue
        if nm_id is None:
            errors.append(
                {
                    "target": requested_target,
                    "stage": "sku_identity",
                    "error_type": "MissingSkuIdentity",
                }
            )
            continue
        try:
            daily_comparison = _compact_daily_comparison(
                requested_target=requested_target,
                identity=identity,
                comparison=get_sku_daily_price_sales_comparison(
                    requested_target,
                    days=14,
                ),
            )
            daily_comparisons.append(daily_comparison)
        except Exception as exc:
            errors.append(
                {
                    "target": requested_target,
                    "stage": "daily_price_sales_comparison",
                    "error_type": type(exc).__name__,
                }
            )
        try:
            request, source = load_pricing_simulation_source(
                nm_id=int(nm_id),
                horizon_days=horizon_days,
            )
            snapshot_hash = sha256(
                json.dumps(
                    source,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            simulation_request = request.model_copy(
                update={"snapshot_hash": snapshot_hash}
            )
            result = run_pricing_simulation(simulation_request)
            result_payload = result.model_dump(mode="json", by_alias=True)
            compact_simulation = _compact_pricing_simulation(
                requested_target=requested_target,
                identity=identity,
                source=source,
                result=result_payload,
            )
            strict_incomplete_price_move = any(
                candidate.get("operation") != "hold_price"
                and candidate.get("status") != "completed"
                for candidate in result_payload.get("candidates") or []
            )
            if strict_incomplete_price_move:
                price_prior = next(
                    (
                        price_priors.get(alias)
                        for alias in (
                            requested_target,
                            str(identity.get("vendor_code") or "").strip(),
                            str(identity.get("nm_id") or "").strip(),
                        )
                        if alias and price_priors.get(alias) is not None
                    ),
                    None,
                )
                hypothesis = build_pricing_hypotheses(
                    simulation_request,
                    result,
                    price_prior=price_prior,
                )
                compact_simulation.update(
                    {
                        "hypothesisStatus": hypothesis.get("status"),
                        "hypothesisCalculationMode": hypothesis.get("calculationMode"),
                        "hypothesisConfidence": hypothesis.get("confidence"),
                        "hypothesisConfidenceScore": hypothesis.get("confidenceScore"),
                        "hypothesisExecutionEligible": bool(
                            hypothesis.get("executionEligible", False)
                        ),
                        "hypothesisAssumptions": list(
                            hypothesis.get("assumptions") or []
                        ),
                        "hypothesisEvidenceGaps": list(
                            hypothesis.get("evidenceGaps") or []
                        ),
                        "hypothesisCandidates": list(
                            hypothesis.get("candidates") or []
                        ),
                    }
                )
            compact_simulation["dailyPriceRows"] = list(
                (daily_comparison or {}).get("rows") or []
            )
            price_simulations.append(compact_simulation)
        except Exception as exc:
            errors.append(
                {
                    "target": requested_target,
                    "stage": "pricing_simulation",
                    "error_type": type(exc).__name__,
                }
            )

    inventory_rows = list((turnover_snapshot or {}).get("rows") or [])
    if not inventory_rows:
        inventory_rows = [
            {
                "vendor_code": target,
                "nm_id": (identities.get(target) or {}).get("nm_id"),
            }
            for target in requested_targets
        ]
    portfolio_optimization = optimize_portfolio(
        query_plan=plan,
        price_simulations=price_simulations,
        inventory_rows=inventory_rows,
        candidate_snapshot=candidate_snapshot,
    )
    coverage = _advisory_coverage(
        portfolio_targets=portfolio_targets,
        requested_targets=requested_targets,
        forecasts=forecasts,
        daily_comparisons=daily_comparisons,
        price_simulations=price_simulations,
        errors=errors,
    )
    return {
        "schemaVersion": "manager-advisory-v1",
        "mode": "shadow",
        "requestedTargets": requested_targets,
        "focusTargets": focus_targets,
        "portfolioTargets": portfolio_targets,
        "forecasts": forecasts,
        "dailyComparisons": daily_comparisons,
        "priceSimulations": price_simulations,
        "portfolioOptimization": portfolio_optimization,
        "coverage": coverage,
        "errors": errors,
    }


def _targets_near_markers(
    user_query: str,
    requested_targets: list[str],
    *,
    markers: tuple[str, ...],
) -> set[str]:
    result: set[str] = set()
    for segment in re.split(r"[.!?;\n]+", str(user_query or "")):
        normalized = segment.casefold().replace("ё", "е")
        if not any(marker in normalized for marker in markers):
            continue
        result.update(target for target in requested_targets if target in segment)
    return result


def _compact_forecast(
    *,
    requested_target: str,
    identity: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    forecast: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target": requested_target,
        "nmId": identity.get("nm_id"),
        "vendorCode": identity.get("vendor_code"),
        "opportunityId": opportunity.get("opportunity_id"),
        "actionFamily": opportunity.get("action_family"),
        "score": opportunity.get("score"),
        "summary": forecast.get("summary"),
        "finalRecommendation": forecast.get("finalRecommendation"),
        "decisionWhy": forecast.get("decisionWhy"),
        "economicEffectRub": forecast.get("economicEffectRub"),
        "profitForecastRub": forecast.get("profitForecastRub"),
        "blockedBy": list(forecast.get("blockedBy") or []),
        "resolutionPlan": list(forecast.get("resolutionPlan") or []),
        "iterationCount": forecast.get("iterationCount"),
    }


def _compact_pricing_simulation(
    *,
    requested_target: str,
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = []
    current_price = (source.get("currentPriceSnapshot") or {}).get(
        "seller_effective_price_rub"
    )
    for candidate in result.get("candidates") or []:
        distribution = candidate.get("distribution") or {}
        total = distribution.get("total") or {}
        units = (candidate.get("unitsSoldDistribution") or {}).get("total") or {}
        candidates.append(
            {
                "priceRub": candidate.get("priceRub"),
                "operation": candidate.get("operation"),
                "status": candidate.get("status"),
                "rank": candidate.get("rank"),
                "isWinner": candidate.get("isWinner", False),
                "meanEffectRub": total.get("mean"),
                "p10EffectRub": total.get("p10"),
                "cvar10EffectRub": total.get("cvar10"),
                "probabilityPositive": total.get("probabilityPositive"),
                "expectedUnits": units.get("mean"),
                "stockoutProbability": candidate.get("stockoutProbability"),
                "unitContributionRub": candidate.get("unitContributionRub"),
                "priceMovePct": (
                    round((float(candidate.get("priceRub")) / float(current_price) - 1) * 100, 2)
                    if candidate.get("priceRub") is not None and current_price
                    else None
                ),
                "violations": list(candidate.get("constraintViolations") or []),
            }
        )
    candidates.sort(
        key=lambda item: (
            item.get("rank") is None,
            item.get("rank") or 999,
            item.get("priceRub") or 0,
        )
    )
    return {
        "target": requested_target,
        "nmId": identity.get("nm_id"),
        "vendorCode": identity.get("vendor_code"),
        "modelVersion": result.get("modelVersion"),
        "sampleCount": result.get("sampleCount"),
        "candidateCount": result.get("candidateCount"),
        "winnerCandidateId": result.get("winnerCandidateId"),
        "horizonDays": source.get("horizonDays") or 7,
        "currentPriceRub": current_price,
        "availableStockUnits": (source.get("stockSnapshot") or {}).get("available_stock"),
        "currentPriceSnapshot": source.get("currentPriceSnapshot"),
        "currentMarketContext": source.get("currentMarketContext"),
        "marketContextHistory": list(source.get("marketContextHistory") or []),
        "historyPeriodStart": source.get("historyPeriodStart"),
        "historyPeriodEnd": source.get("historyPeriodEnd"),
        "historyRowCount": source.get("historyRowCount"),
        "causalValid": bool(source.get("causalValid", False)),
        "contextSchemaVersion": source.get("contextSchemaVersion"),
        "activeContextFeatures": list(source.get("activeContextFeatures") or []),
        "featureCoverage": dict(source.get("featureCoverage") or {}),
        "contextDataGaps": list(source.get("contextDataGaps") or []),
        "futureInboundEvents": list(source.get("futureInboundEvents") or []),
        "futurePromotionEvents": list(source.get("futurePromotionEvents") or []),
        "candidates": candidates,
    }


def _price_candidate_key(candidate: Mapping[str, Any]) -> tuple[str, str]:
    operation = str(candidate.get("operation") or "").strip()
    price = _number_or_none(candidate.get("priceRub"))
    if price is not None:
        return operation, f"{price:.9g}"
    return operation, str(candidate.get("candidateId") or "").strip()


def _merge_price_candidates(
    strict_completed: list[Mapping[str, Any]],
    hypotheses: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in [*strict_completed, *hypotheses]:
        key = _price_candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(candidate))
    return merged


def _compact_daily_comparison(
    *,
    requested_target: str,
    identity: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in comparison.get("rows") or []:
        rows.append(
            {
                "date": row.get("report_date"),
                "basePriceRub": row.get("base_price_rub"),
                "sellerDiscountPct": row.get("seller_discount_pct"),
                "sellerEffectivePriceRub": row.get("seller_effective_price_rub"),
                "observedSellerPriceRub": row.get(
                    "observed_transaction_seller_price_avg_rub"
                ),
                "observedBuyerPriceRub": row.get("observed_buyer_price_avg_rub"),
                "sppPct": row.get("observed_spp_pct_avg"),
                "priceBasis": row.get("price_basis"),
                "opens": row.get("open_count"),
                "carts": row.get("cart_count"),
                "funnelOrders": row.get("funnel_order_count"),
                "addToCartConversionPct": row.get("add_to_cart_conversion"),
                "cartToOrderConversionPct": row.get("cart_to_order_conversion"),
                "orders": row.get("non_cancelled_order_units"),
                "realizedSales": row.get("realized_sales_units"),
                "realizedForPayRub": row.get("realized_for_pay_rub"),
                "complete": bool(row.get("business_day_complete")),
            }
        )
    return {
        "target": requested_target,
        "nmId": identity.get("nm_id"),
        "vendorCode": identity.get("vendor_code"),
        "causalValid": bool(comparison.get("causal_valid", False)),
        "rows": rows,
    }


_SCENARIO_FAMILY_LABELS = {
    "price": "ценовые",
    "marketing": "рекламные",
    "supply": "поставочные",
}

_CONTEXT_FEATURE_LABELS = {
    "ad_spend_rub": "рекламный расход",
    "ad_views": "рекламные показы",
    "ad_clicks": "рекламные клики",
    "lagged_organic_position": "органическая позиция предыдущего дня",
    "lagged_open_to_cart_conversion": "конверсия карточка→корзина предыдущего дня",
    "lagged_cart_to_order_conversion": "конверсия корзина→заказ предыдущего дня",
    "lagged_competitor_median_price_rub": "медианная цена конкурентов предыдущего дня",
    "promotion_active": "участие в акции",
    "promotion_discount_pct": "скидка акции",
    "annual_season_sin": "годовая сезонность",
    "annual_season_cos": "годовая сезонность",
}

_CONTEXT_GAP_LABELS = {
    "context_ad_spend_rub_insufficient_history": (
        "недостаточно дней истории рекламных расходов"
    ),
    "context_ad_views_insufficient_history": (
        "недостаточно дней истории рекламных показов"
    ),
    "context_ad_clicks_insufficient_history": (
        "недостаточно дней истории рекламных кликов"
    ),
    "context_ad_views_collinear_with_ad_spend_rub": (
        "рекламные показы не включены отдельно, потому что дублируют динамику расходов"
    ),
    "context_lagged_competitor_median_price_rub_missing": (
        "нет сопоставимой исторической медианы цен конкурентов"
    ),
    "context_promotion_active_insufficient_history": (
        "недостаточно дней истории участия в акциях"
    ),
    "context_promotion_discount_pct_missing": "нет истории размера промо-скидки",
    "context_annual_season_sin_insufficient_history": (
        "для годовой сезонности нужно не менее 365 дней"
    ),
    "context_annual_season_cos_insufficient_history": (
        "для годовой сезонности нужно не менее 365 дней"
    ),
    "future_supplier_inbound_not_confirmed": (
        "нет подтверждённой будущей поставки с ETA"
    ),
}

_ACTION_LABELS = {
    "pause_campaign": "приостановить кампанию",
    "resume_campaign": "возобновить кампанию",
    "change_bid": "изменить ставку",
    "set_price": "изменить цену",
    "create_replenishment_task": "создать задачу на поставку",
}

_DOMAIN_LABELS = {
    "price": "цена",
    "pricing": "цена",
    "marketing": "реклама",
    "advertising": "реклама",
    "supply": "поставки",
    "forecast": "прогноз",
    "finance": "экономика",
    "sales": "продажи",
}

_COVERAGE_METRIC_LABELS = {
    "current_cover_days_7d": "текущее покрытие по продажам 7д",
    "turnover_days_30d_average_stock": "оборачиваемость по среднему остатку 30д",
    "simulated_forward_cover_days": "прогноз покрытия по 500 симуляциям",
}

_PUBLIC_TEXT_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bprice[-_ ]?срез\w*\b", re.IGNORECASE), "ценовой срез"),
    (re.compile(r"\bthin_price_data\b", re.IGNORECASE), "недостаточно ценовых данных"),
    (
        re.compile(r"\bstale_demand_observations\b", re.IGNORECASE),
        "история спроса недостаточно свежая",
    ),
    (re.compile(r"\bprice_status\b", re.IGNORECASE), "статус ценовых данных"),
    (re.compile(r"\bprice_ai_run_date\b", re.IGNORECASE), "дата расчёта цены"),
    (re.compile(r"\bmarketing_confidence\b", re.IGNORECASE), "уверенность рекламных данных"),
    (
        re.compile(r"\bmarketing_waste_query_share_7d\b", re.IGNORECASE),
        "доля неэффективных рекламных запросов за 7 дней",
    ),
    (re.compile(r"\bcart_to_order_conversion\b", re.IGNORECASE), "конверсия корзина→заказ"),
    (re.compile(r"\bcoverage_days\b", re.IGNORECASE), "покрытие в днях"),
    (re.compile(r"\breorder_qty(?:_recommended)?\b", re.IGNORECASE), "рекомендованный дозаказ"),
    (re.compile(r"\bavailability_status\b", re.IGNORECASE), "статус наличия"),
    (re.compile(r"\bcompetitor_count\b", re.IGNORECASE), "число конкурентов"),
    (
        re.compile(r"\bcritic_rejected_action_plan\b", re.IGNORECASE),
        "план отклонён проверяющей моделью",
    ),
    (re.compile(r"\bcausal[-_ ]?valid\b", re.IGNORECASE), "причинный"),
    (re.compile(r"\bshadow\b", re.IGNORECASE), "теневой режим"),
)


def _public_text(value: Any) -> str:
    text = str(value or "").strip()
    for pattern, replacement in _PUBLIC_TEXT_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _confidence_text(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "high": "высокая",
        "medium": "средняя",
        "low": "низкая",
    }.get(normalized, normalized or "не указана")


def _required_scenario_families(query_plan: Mapping[str, Any] | None) -> list[str]:
    plan = dict(query_plan or {})
    if not list(plan.get("scenario_requests") or []):
        return []
    if "allowed_action_families" in plan:
        return allowed_action_families(plan)
    analyses = {str(item) for item in list(plan.get("requested_analyses") or [])}
    domains = {
        str(getattr(item, "value", item))
        for item in list(plan.get("required_domains") or [])
    }
    families: list[str] = []
    if "price" in domains or analyses.intersection({"pricing", "price_scenarios"}):
        families.append("price")
    if "marketing" in domains or "advertising" in analyses:
        families.append("marketing")
    if "supply" in domains or analyses.intersection(
        {"replenishment", "inventory_turnover", "current_stock"}
    ):
        families.append("supply")
    return families


def _decision_items(
    manager_decision: Mapping[str, Any] | None,
) -> list[tuple[dict[str, Any], str]]:
    decision = dict(manager_decision or {})
    result: list[tuple[dict[str, Any], str]] = []
    for item in list(decision.get("rankedActions") or decision.get("ranked_actions") or []):
        if isinstance(item, Mapping):
            result.append((dict(item), "ready"))
    for item in list(
        decision.get("rejectedCandidates") or decision.get("rejected_candidates") or []
    ):
        if isinstance(item, Mapping):
            result.append((dict(item), "blocked"))
    return result


def _family_matches(item: Mapping[str, Any], family: str) -> bool:
    value = str(item.get("family") or "").strip().lower()
    if family == "marketing":
        return value in {"advertising", "marketing"}
    if family == "price":
        return value in {"pricing", "price"}
    return value == family


def _exact_value_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "—"
    raw = value.get("value")
    unit = {
        "wb_status_code": "статус WB",
        "units_ordered": "шт. к заказу",
    }.get(str(value.get("unit") or "").strip(), str(value.get("unit") or "").strip())
    return f"{raw} {unit}".strip() if raw is not None else "—"


def _candidate_limit(item: Mapping[str, Any]) -> str:
    messages: list[str] = []
    for reason in list(item.get("reasons") or []):
        message = (
            str(reason.get("message") or "").strip()
            if isinstance(reason, Mapping)
            else str(reason).strip()
        )
        if message and message not in messages:
            messages.append(message)
    stop = str(item.get("stopCondition") or item.get("stop_condition") or "").strip()
    if stop and stop not in messages:
        messages.append(stop)
    return "; ".join(messages[:2]) or "Только после проверки ограничений и экономики."


def _candidate_change(item: Mapping[str, Any]) -> str:
    before = item.get("from") or item.get("from_value")
    after = item.get("to") or item.get("to_value")
    if before is None or after is None:
        return "Изменение не рассчитано"
    return f"{_exact_value_text(before)} → {_exact_value_text(after)}"


def _candidate_instruction(item: Mapping[str, Any]) -> str:
    operation = str(item.get("operation") or "действие").strip()
    label = _ACTION_LABELS.get(operation, operation.replace("_", " "))
    target = str(item.get("target") or item.get("candidateId") or "портфель").strip()
    return f"{label.capitalize()}: {target} ({_candidate_change(item)})"


def _append_family_scenarios(
    lines: list[str],
    *,
    family: str,
    manager_decision: Mapping[str, Any] | None,
    portfolio_optimization: Mapping[str, Any] | None = None,
) -> None:
    title = {
        "marketing": "## Рекламные сценарии",
        "supply": "## Поставочные сценарии",
    }[family]
    optimized = [
        dict(item)
        for item in list((portfolio_optimization or {}).get("scenarios") or [])
        if isinstance(item, Mapping) and item.get("family") == family
    ]
    lines.extend(["", title, ""])
    if not optimized:
        lines.append(
            "Числовые варианты не рассчитаны: подтверждённых входных данных для этого семейства недостаточно."
        )
        return
    lines.extend(
        [
            "| Решение | Δ покрытия по метрике сценария, дней | Эффект CP | p10 CP | Риск дефицита | Статус | Ограничение |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in optimized:
        reasons = "; ".join(str(value) for value in item.get("constraintReasons") or []) or "—"
        cells = [
            str(item.get("action") or "—"),
            _number_text(item.get("inventoryDaysReduction")),
            _rub(item.get("expectedContributionProfitRub")),
            _rub(item.get("p10ContributionProfitRub")),
            _pct(item.get("stockoutProbability")),
            str(item.get("calculationStatus") or "—"),
            reasons,
        ]
        lines.append("| " + " | ".join(_public_text(cell).replace("|", "/") for cell in cells) + " |")


def _optimized_recommendations(
    advisory: Mapping[str, Any],
    manager_decision: Mapping[str, Any] | None,
    query_plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    optimization = dict(advisory.get("portfolioOptimization") or {})
    allowed = set(allowed_action_families(query_plan))
    restrict_families = bool(
        query_plan is not None and "allowed_action_families" in query_plan
    )
    rows: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    for collection in (
        optimization.get("priorityRecommendations") or [],
        optimization.get("topRecommendations") or [],
        optimization.get("topHypotheses") or [],
    ):
        for item in list(collection):
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            family = str(row.get("family") or "").strip()
            if restrict_families and family not in allowed:
                continue
            scenario_id = str(row.get("scenarioId") or "").strip()
            target_key = (family, str(row.get("target") or "").strip())
            if (scenario_id and scenario_id in seen_scenarios) or target_key in seen_targets:
                continue
            if scenario_id:
                seen_scenarios.add(scenario_id)
            seen_targets.add(target_key)
            rows.append(row)
            if len(rows) == 5:
                break
        if len(rows) == 5:
            break
    decision_status = str((manager_decision or {}).get("status") or "blocked")
    simulations_by_target = {
        _artifact_target(item): dict(item)
        for item in list(advisory.get("priceSimulations") or [])
        if isinstance(item, Mapping) and _artifact_target(item)
    }
    for row in rows:
        if str(row.get("family") or "") == "price":
            simulation = simulations_by_target.get(str(row.get("target") or "")) or {}
            current_price = _number_or_none(row.get("currentPriceRub"))
            if current_price is None:
                current_price = _number_or_none(simulation.get("currentPriceRub"))
                if current_price is not None:
                    row["currentPriceRub"] = current_price
            candidate_price = _number_or_none(row.get("candidatePriceRub"))
            if candidate_price is None:
                scenario_parts = str(row.get("scenarioId") or "").rsplit(":", 1)
                candidate_price = (
                    _number_or_none(scenario_parts[-1])
                    if len(scenario_parts) == 2
                    else None
                )
                if candidate_price is not None:
                    row["candidatePriceRub"] = candidate_price
            if row.get("priceMovePct") is None and current_price and candidate_price is not None:
                row["priceMovePct"] = (candidate_price / current_price - 1) * 100
            if row.get("availableStockUnits") is None:
                row["availableStockUnits"] = simulation.get("availableStockUnits")
            raw_hypotheses = simulation.get("hypothesisCandidates")
            hypothesis_candidates = (
                list(raw_hypotheses.get("candidates") or [])
                if isinstance(raw_hypotheses, Mapping)
                else list(raw_hypotheses or [])
            )
            matching_candidate = next(
                (
                    item
                    for item in [
                        *list(simulation.get("candidates") or []),
                        *hypothesis_candidates,
                    ]
                    if isinstance(item, Mapping)
                    and candidate_price is not None
                    and _number_or_none(item.get("priceRub")) is not None
                    and abs(float(item["priceRub"]) - candidate_price)
                    <= max(1.0, abs(candidate_price) * 0.001)
                ),
                None,
            )
            if row.get("probabilityPositive") is None and isinstance(
                matching_candidate, Mapping
            ):
                row["probabilityPositive"] = matching_candidate.get(
                    "probabilityPositive",
                    matching_candidate.get("probPositive"),
                )
        if row.get("calculationStatus") == "hypothesis":
            row["executionStatus"] = "hypothesis"
        else:
            row["executionStatus"] = (
                "ready"
                if decision_status == "ready_for_approval" and row.get("eligible") is True
                else "blocked"
            )
    return rows


def _coverage_metric_label(item: Mapping[str, Any]) -> str:
    metric = str(item.get("inventoryMetric") or "").strip()
    label = _COVERAGE_METRIC_LABELS.get(metric, metric or "метрика не указана")
    horizon = item.get("horizonDays")
    if metric == "simulated_forward_cover_days" and horizon is not None:
        return f"{label}, {horizon}д"
    return label


def _hypothesis_goal_view(
    optimization: Mapping[str, Any],
) -> dict[str, Any]:
    stored = optimization.get("hypothesisGoalAttainment")
    if isinstance(stored, Mapping):
        return dict(stored)
    strict_goal = dict(optimization.get("goalAttainment") or {})
    hypotheses = [
        dict(item)
        for item in list(optimization.get("topHypotheses") or [])
        if isinstance(item, Mapping)
    ]
    if not strict_goal or not hypotheses:
        return {}
    above_target = [
        dict(item)
        for item in list(
            strict_goal.get("aboveTargetSkus")
            or [
                *list(strict_goal.get("remainingAboveTargetSkus") or []),
                *list(strict_goal.get("resolvedByTopRecommendations") or []),
            ]
        )
        if isinstance(item, Mapping)
    ]
    target_cover = _number_or_none(strict_goal.get("targetCoverDays"))
    remaining: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for row in above_target:
        target = str(row.get("target") or "")
        matching = [
            item
            for item in hypotheses
            if str(item.get("target") or "") == target
            and _number_or_none(item.get("inventoryDaysAfter")) is not None
        ]
        best = min(
            matching,
            key=lambda item: _number_or_none(item.get("inventoryDaysAfter"))
            or float("inf"),
            default=None,
        )
        enriched = {
            **row,
            "bestScenarioCoverDays": (
                _number_or_none(best.get("inventoryDaysAfter")) if best else None
            ),
            "bestScenarioMetric": best.get("inventoryMetric") if best else None,
            "bestScenarioId": best.get("scenarioId") if best else None,
        }
        if (
            best is not None
            and target_cover is not None
            and float(best["inventoryDaysAfter"]) <= target_cover
        ):
            resolved.append(enriched)
        else:
            remaining.append(enriched)
    unmeasured = list(strict_goal.get("unmeasuredSkus") or [])
    stockout_skus = [
        dict(item)
        for item in list(strict_goal.get("stockoutSkus") or [])
        if isinstance(item, Mapping)
    ]
    status = (
        "not_achieved"
        if remaining or stockout_skus
        else ("not_proven" if unmeasured else "achieved")
    )
    return {
        "status": status,
        "targetCoverDays": strict_goal.get("targetCoverDays"),
        "resolvedByTopRecommendationsCount": len(resolved),
        "remainingAboveTargetSkuCount": len(remaining),
        "resolvedByTopRecommendations": resolved,
        "remainingAboveTargetSkus": remaining,
        "stockoutSkuCount": len(stockout_skus),
        "stockoutSkus": stockout_skus,
        "unmeasuredSkus": unmeasured,
    }


def _append_goal_attainment(
    lines: list[str],
    *,
    advisory: Mapping[str, Any],
    query_plan: Mapping[str, Any] | None,
) -> None:
    optimization = dict(advisory.get("portfolioOptimization") or {})
    goal = dict(optimization.get("goalAttainment") or {})
    hypothesis_goal = _hypothesis_goal_view(optimization)
    constraints = dict((query_plan or {}).get("constraints") or {})
    target = goal.get("targetCoverDays", constraints.get("target_cover_days", 30))
    target_text = _number_text(target)
    status = str(goal.get("status") or "not_proven")
    recommendation_count = len(
        list(optimization.get("topRecommendations") or [])
    )
    strict_remaining = [
        dict(item)
        for item in list(goal.get("remainingAboveTargetSkus") or [])
        if isinstance(item, Mapping)
    ]
    strict_stockouts = [
        dict(item)
        for item in list(goal.get("stockoutSkus") or [])
        if isinstance(item, Mapping)
    ]
    remaining = [
        dict(item)
        for item in list(
            (hypothesis_goal or goal).get("remainingAboveTargetSkus") or []
        )
        if isinstance(item, Mapping)
    ]
    unmeasured = [
        str(value)
        for value in list((hypothesis_goal or goal).get("unmeasuredSkus") or [])
    ]

    if status == "achieved":
        verdict = f"Итог: цель достигнута — для всех проверенных SKU подтверждено покрытие не выше {target_text} дней."
    elif status == "not_achieved":
        verdict = (
            f"Итог: цель не достигнута — после учёта {recommendation_count} допустимых приоритетных сценариев "
            f"остаются SKU выше {target_text} дней: {len(strict_remaining)}; "
            f"SKU без остатка: {len(strict_stockouts)}."
        )
    else:
        verdict = (
            f"Итог: достижение цели не доказано — для {len(unmeasured)} SKU "
            "недостаточно данных о текущей скорости продаж."
        )

    lines.extend(
        [
            "",
            f"## Достижение цели {target_text} дней",
            "",
            verdict,
            "",
            "Текущее покрытие считается только как доступный остаток / среднедневные продажи за последние 7 дней. "
            "Покрытие ценового сценария — отдельный прогноз по 500 симуляциям; оборачиваемость 30д — отдельная метрика по среднему остатку.",
        ]
    )
    if hypothesis_goal:
        hypothesis_status = str(hypothesis_goal.get("status") or "not_proven")
        hypothesis_remaining = len(
            list(hypothesis_goal.get("remainingAboveTargetSkus") or [])
        )
        hypothesis_resolved = int(
            hypothesis_goal.get("resolvedByTopRecommendationsCount") or 0
        )
        hypothesis_stockouts = int(hypothesis_goal.get("stockoutSkuCount") or 0)
        if hypothesis_status == "achieved":
            hypothesis_verdict = (
                "По top-5 ценовым гипотезам цель достигалась бы для всех измеримых SKU."
            )
        elif hypothesis_status == "not_achieved":
            hypothesis_verdict = (
                "По top-5 ценовым гипотезам до цели доходят "
                f"{hypothesis_resolved} SKU; выше цели остаются {hypothesis_remaining}; "
                f"без остатка {hypothesis_stockouts}."
            )
        else:
            hypothesis_verdict = (
                "Достижение цели даже по ценовым гипотезам не доказано из-за пробелов данных."
            )
        lines.extend(["", hypothesis_verdict])
    if remaining:
        lines.extend(
            [
                "",
                f"### SKU, остающиеся выше цели {target_text} дней",
                "",
                "| SKU | Текущее покрытие 7д, дней | Лучшее покрытие сценария, дней | Метрика сценария | Вывод |",
                "|---|---:|---:|---|---|",
            ]
        )
        for row in remaining:
            scenario_metric = _COVERAGE_METRIC_LABELS.get(
                str(row.get("bestScenarioMetric") or ""),
                str(row.get("bestScenarioMetric") or "сценарий в top-5 отсутствует"),
            )
            conclusion = (
                "сценарий top-5 не доводит до цели"
                if row.get("bestScenarioId")
                else "для SKU нет допустимого сценария в top-5"
            )
            cells = [
                str(row.get("target") or "—"),
                _number_text(row.get("currentCoverDays7d")),
                _number_text(row.get("bestScenarioCoverDays")),
                scenario_metric,
                conclusion,
            ]
            lines.append("| " + " | ".join(_public_text(cell).replace("|", "/") for cell in cells) + " |")
    if unmeasured:
        lines.extend(
            [
                "",
                "Не удалось оценить по формуле 7д: " + ", ".join(unmeasured) + ".",
            ]
        )


def _append_business_overview(
    lines: list[str],
    *,
    advisory: Mapping[str, Any],
    manager_decision: Mapping[str, Any] | None,
    query_plan: Mapping[str, Any] | None,
) -> None:
    optimization = dict(advisory.get("portfolioOptimization") or {})
    coverage = dict(advisory.get("coverage") or {})
    recommendations = _optimized_recommendations(
        advisory,
        manager_decision,
        query_plan,
    )
    simulations = [
        dict(item)
        for item in list(advisory.get("priceSimulations") or [])
        if isinstance(item, Mapping)
    ]
    status = str((manager_decision or {}).get("status") or "blocked")
    status_text = {
        "ready_for_approval": "готово к подтверждению менеджером",
        "blocked": "только гипотезы; автоматическое исполнение заблокировано",
        "needs_data": "недостаточно данных для исполнения",
        "evaluating": "расчёт продолжается",
    }.get(status, status)
    if recommendations and not any(
        str(item.get("executionStatus") or "") == "ready"
        for item in recommendations
    ):
        status_text = "только гипотезы; автоматическое исполнение заблокировано"
    target_count = int(coverage.get("targetSkuCount") or len(simulations))
    artifact_targets = [
        str(item)
        for item in list(coverage.get("artifactTargets") or [])
        if str(item).strip()
    ]
    artifact_count = len(artifact_targets) if artifact_targets else len(simulations)
    error_count = len(list(coverage.get("errorTargets") or []))
    if not error_count:
        error_count = len(list(advisory.get("errors") or []))
    missing_count = len(list(coverage.get("missingArtifactTargets") or []))
    numeric_count = int(
        coverage.get("numericPriceScenarioSkuCount")
        or len([item for item in simulations if _has_numeric_price_move(item)])
    )
    scenario_count = int(optimization.get("scenarioCount") or 0)
    horizon_days = int(
        optimization.get("horizonDays")
        or ((query_plan or {}).get("constraints") or {}).get("horizon_days")
        or 30
    )
    goal = dict(optimization.get("goalAttainment") or {})
    hypothesis_goal = _hypothesis_goal_view(optimization)
    target_cover = goal.get(
        "targetCoverDays",
        ((query_plan or {}).get("constraints") or {}).get("target_cover_days"),
    )
    goal_status = str(goal.get("status") or "not_proven")
    remaining_count = len(
        [
            item
            for item in list(goal.get("remainingAboveTargetSkus") or [])
            if isinstance(item, Mapping)
        ]
    )
    stockout_count = len(
        [
            item
            for item in list(goal.get("stockoutSkus") or [])
            if isinstance(item, Mapping)
        ]
    )
    goal_text = {
        "achieved": f"достигнута: покрытие не выше {_number_text(target_cover)} дней",
        "not_achieved": (
            f"не достигнута: {remaining_count} SKU остаются выше "
            f"{_number_text(target_cover)} дней; без остатка {stockout_count}"
        ),
        "not_proven": "не доказана из-за пробелов данных",
    }.get(goal_status, goal_status)
    hypothesis_goal_text = ""
    if hypothesis_goal:
        hypothesis_status = str(hypothesis_goal.get("status") or "not_proven")
        hypothesis_remaining = len(
            list(hypothesis_goal.get("remainingAboveTargetSkus") or [])
        )
        hypothesis_resolved = int(
            hypothesis_goal.get("resolvedByTopRecommendationsCount") or 0
        )
        hypothesis_stockouts = int(hypothesis_goal.get("stockoutSkuCount") or 0)
        hypothesis_goal_text = {
            "achieved": "достигалась бы для всех измеримых SKU",
            "not_achieved": (
                f"до цели доходят {hypothesis_resolved} SKU; "
                f"выше цели остаются {hypothesis_remaining}; "
                f"без остатка {hypothesis_stockouts}"
            ),
            "not_proven": "не доказана из-за пробелов данных",
        }.get(hypothesis_status, hypothesis_status)
    mean_effects = [
        _number_or_none(item.get("expectedContributionProfitRub"))
        for item in recommendations
    ]
    mean_effects = [value for value in mean_effects if value is not None]
    total_mean = sum(mean_effects) if mean_effects else None
    stockout_values = [
        _number_or_none(item.get("stockoutProbability"))
        for item in recommendations
    ]
    stockout_values = [value for value in stockout_values if value is not None]
    confidence_counts: dict[str, int] = {}
    for item in recommendations:
        confidence = _confidence_text(item.get("confidence"))
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
    confidence_text = ", ".join(
        f"{key}: {value}" for key, value in confidence_counts.items()
    ) or "не рассчитана"
    history_starts = [
        str(item.get("historyPeriodStart"))
        for item in simulations
        if item.get("historyPeriodStart")
    ]
    history_ends = [
        str(item.get("historyPeriodEnd"))
        for item in simulations
        if item.get("historyPeriodEnd")
    ]
    snapshot_dates = [
        str(snapshot.get("snapshot_date") or snapshot.get("report_date"))
        for item in simulations
        if isinstance((snapshot := item.get("currentPriceSnapshot")), Mapping)
        and (snapshot.get("snapshot_date") or snapshot.get("report_date"))
    ]
    price_ready = len(
        [item for item in simulations if _number_or_none(item.get("currentPriceRub")) is not None]
    )
    stock_ready = len(
        [item for item in simulations if _number_or_none(item.get("availableStockUnits")) is not None]
    )
    causal_ready = len([item for item in simulations if item.get("causalValid") is True])
    if history_starts and history_ends:
        common_start = max(history_starts)
        common_end = min(history_ends)
        period_text = (
            f"{common_start}–{common_end}"
            if common_start <= common_end
            else "нет общего сопоставимого периода"
        )
    else:
        period_text = "период не указан"
    snapshot_text = (
        min(snapshot_dates)
        if snapshot_dates and min(snapshot_dates) == max(snapshot_dates)
        else (
            f"{min(snapshot_dates)}–{max(snapshot_dates)}"
            if snapshot_dates
            else "дата среза не указана"
        )
    )
    leader = recommendations[0].get("action") if recommendations else None

    rows = [
        ("Статус решения", status_text),
        (
            "Охват",
            f"рассчитано {artifact_count}/{target_count} SKU; ошибки {error_count}; "
            f"без результата {missing_count}; численные цены {numeric_count}/{target_count}; "
            f"сценариев {scenario_count}",
        ),
        ("Горизонт", f"{horizon_days} дней"),
        ("Складская цель", goal_text),
        (
            "Сумма средних ΔCP выбранных вариантов",
            _rub(total_mean) if total_mean is not None else "не рассчитана",
        ),
        (
            "Максимальный риск дефицита в top-5",
            _pct(max(stockout_values)) if stockout_values else "не рассчитан",
        ),
        ("Уверенность top-5", confidence_text),
        (
            "Готовность данных",
            f"цена {price_ready}/{len(simulations)}, остаток {stock_ready}/{len(simulations)}, "
            f"причинно подтверждено {causal_ready}/{len(simulations)}; "
            f"срезы {snapshot_text}; общий период {period_text}",
        ),
    ]
    if hypothesis_goal_text:
        rows.insert(4, ("Складская цель по top-5 гипотезам", hypothesis_goal_text))
    if leader:
        rows.append(("Первый вариант для проверки", str(leader)))

    lines.extend(
        [
            "",
            "## Итог для бизнеса",
            "",
            "| Показатель | Значение |",
            "|---|---|",
        ]
    )
    for label, value in rows:
        lines.append(
            "| "
            + " | ".join(
                _public_text(cell).replace("|", "/") for cell in (label, value)
            )
            + " |"
        )
    if total_mean is not None:
        lines.extend(
            [
                "",
                "Сумма ΔCP — арифметическая сумма средних эффектов отдельных SKU, а не совместный портфельный прогноз. "
                "p10 оценивается отдельно для каждого ценового теста и не суммируется как портфельный p10.",
            ]
        )


def _append_price_recommendations(
    lines: list[str],
    recommendations: list[dict[str, Any]],
) -> None:
    lines.extend(
        [
            "",
            "## 5 приоритетных рекомендаций",
            "",
            "В таблице только ценовые варианты. Проверяющие агенты влияют на основания и ограничения, но не создают действия других типов.",
            "",
            "| # | Статус | SKU | Цена сейчас → тест | Δ цены | Остаток, шт. | Продажи за горизонт, база → тест | Покрытие, до → после | ΔCP средний | p10 ΔCP | Вероятность плюса | OOS | Уверенность | Почему этот вариант |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    if len(recommendations) != 5:
        lines.extend(
            [
                f"Рассчитано ценовых вариантов: {len(recommendations)} из 5. Заполнители других типов действий не добавляются.",
                "",
            ]
        )
    status_labels = {
        "ready": "готово к подтверждению",
        "hypothesis": "теневой тест",
        "blocked": "заблокировано",
        "shadow_test": "теневой тест",
    }
    for index, item in enumerate(recommendations, start=1):
        current_price = item.get("currentPriceRub")
        candidate_price = item.get("candidatePriceRub")
        price_change = (
            f"{_rub(current_price)} → {_rub(candidate_price)}"
            if current_price is not None or candidate_price is not None
            else str(item.get("action") or "—")
        )
        baseline_units = item.get("baselineExpectedUnits")
        expected_units = item.get("expectedUnits")
        units_change = (
            f"{_number_text(baseline_units)} → {_number_text(expected_units)}"
            if baseline_units is not None or expected_units is not None
            else "—"
        )
        cover_change = (
            f"{_number_text(item.get('inventoryDaysBefore'))} → "
            f"{_number_text(item.get('inventoryDaysAfter'))}"
        )
        drivers: list[str] = []
        p10 = _number_or_none(item.get("p10ContributionProfitRub"))
        reduction = _number_or_none(item.get("inventoryDaysReduction"))
        units_before = _number_or_none(baseline_units)
        units_after = _number_or_none(expected_units)
        if p10 is not None and p10 > 0:
            drivers.append("p10 ΔCP положительный")
        if reduction is not None and reduction > 0:
            drivers.append(f"покрытие −{_number_text(reduction)} дн.")
        if units_before is not None and units_after is not None and units_after > units_before:
            drivers.append(f"продажи +{_number_text(units_after - units_before)} шт.")
        limitation = next(
            (
                _public_text(value)
                for value in [
                    *list(item.get("constraintReasons") or []),
                    *list(item.get("evidenceGaps") or []),
                    *list(item.get("limitations") or []),
                ]
                if str(value).strip()
            ),
            "",
        )
        if limitation:
            drivers.append("ограничение: " + limitation)
        cells = [
            str(index),
            status_labels.get(
                str(item.get("executionStatus") or "blocked"),
                str(item.get("executionStatus") or "blocked"),
            ),
            str(item.get("target") or "—"),
            price_change,
            _pct_points(item.get("priceMovePct")),
            _number_text(item.get("availableStockUnits")),
            units_change,
            cover_change,
            _signed_rub(item.get("expectedContributionProfitRub")),
            _signed_rub(item.get("p10ContributionProfitRub")),
            _pct(item.get("probabilityPositive")),
            _pct(item.get("stockoutProbability")),
            _confidence_text(item.get("confidence")),
            "; ".join(drivers) or "числовой вариант требует проверки",
        ]
        lines.append(
            "| "
            + " | ".join(_public_text(cell).replace("|", "/") for cell in cells)
            + " |"
        )


def _append_compact_price_matrix(
    lines: list[str],
    *,
    simulations: list[Mapping[str, Any]],
    portfolio_optimization: Mapping[str, Any],
) -> None:
    if not simulations:
        return
    optimized_prices = {
        str(item.get("scenarioId") or ""): dict(item)
        for item in list(portfolio_optimization.get("scenarios") or [])
        if isinstance(item, Mapping) and item.get("family") == "price"
    }
    lines.extend(
        [
            "",
            "## Полная матрица ценовых вариантов",
            "",
            "Одна строка — один вариант. Подробные дневные данные остаются только у фокусных SKU.",
            "",
            "| SKU | Вариант | Цена | Продажи за горизонт | Покрытие после, дней | ΔCP средний | p10 ΔCP | OOS | Статус |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for raw_simulation in simulations:
        simulation = dict(raw_simulation)
        target = str(simulation.get("vendorCode") or simulation.get("target") or "—")
        current_price = _number_or_none(simulation.get("currentPriceRub"))
        strict = [
            dict(item)
            for item in list(simulation.get("candidates") or [])
            if isinstance(item, Mapping) and item.get("status") == "completed"
        ]
        hypotheses = [
            dict(item)
            for item in list(simulation.get("hypothesisCandidates") or [])
            if isinstance(item, Mapping)
        ]
        for candidate in _merge_price_candidates(strict, hypotheses):
            price = _number_or_none(candidate.get("priceRub"))
            move_pct = _number_or_none(candidate.get("priceMovePct"))
            if move_pct is None and price is not None and current_price:
                move_pct = (price / current_price - 1) * 100
            scenario_key = (
                f"price:{target}:{float(price):g}" if price is not None else ""
            )
            optimized = optimized_prices.get(scenario_key) or {}
            is_hold = candidate.get("operation") == "hold_price" or (
                move_pct is not None and abs(move_pct) < 0.05
            )
            cover_after = (
                optimized.get("inventoryDaysAfter")
                if not is_hold
                else next(
                    (
                        item.get("inventoryDaysBefore")
                        for item in optimized_prices.values()
                        if str(item.get("target") or "") == target
                        and item.get("inventoryDaysBefore") is not None
                    ),
                    None,
                )
            )
            status = (
                "база"
                if is_hold
                else (
                    "гипотеза"
                    if candidate.get("status") == "hypothesis"
                    or optimized.get("calculationStatus") == "hypothesis"
                    else "строгий расчёт"
                )
            )
            cells = [
                target,
                _pct_points(move_pct),
                _rub(price),
                _number_text(candidate.get("expectedUnits")),
                _number_text(cover_after),
                _signed_rub(candidate.get("meanEffectRub", candidate.get("mean"))),
                _signed_rub(candidate.get("p10EffectRub", candidate.get("p10"))),
                _pct(
                    candidate.get(
                        "stockoutProbability",
                        candidate.get("stockout"),
                    )
                ),
                status,
            ]
            lines.append(
                "| "
                + " | ".join(
                    _public_text(cell).replace("|", "/") for cell in cells
                )
                + " |"
            )


def _append_optimized_recommendations(
    lines: list[str],
    *,
    advisory: Mapping[str, Any],
    manager_decision: Mapping[str, Any] | None,
    query_plan: Mapping[str, Any] | None = None,
) -> None:
    recommendations = _optimized_recommendations(
        advisory,
        manager_decision,
        query_plan,
    )
    price_only = allowed_action_families(query_plan) == ["price"]
    if price_only:
        _append_price_recommendations(lines, recommendations)
        return
    lines.extend(
        [
            "",
            "## 5 приоритетных рекомендаций",
            "",
            "Статус `blocked` запрещает исполнение, но не скрывает рассчитанный план проверки.",
            "",
            "| # | Статус | Домен | Объект | Решение | Метрика покрытия | До → после, дней | Эффект CP | p10 CP | Риск дефицита | Основание |",
            "|---:|---|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    if len(recommendations) != 5:
        lines.extend(
            [
                f"Рассчитано допустимых решений: {len(recommendations)} из 5. Заполнители не добавляются.",
                "",
            ]
        )
    for index, item in enumerate(recommendations, start=1):
        before = item.get("inventoryDaysBefore")
        after = item.get("inventoryDaysAfter")
        reduction = item.get("inventoryDaysReduction")
        if before is not None and after is not None:
            reduction_value = _number_or_none(reduction)
            delta = (
                f"−{_number_text(reduction_value)}"
                if reduction_value is not None and reduction_value > 0
                else (
                    f"+{_number_text(abs(reduction_value))}"
                    if reduction_value is not None and reduction_value < 0
                    else _number_text(reduction_value)
                )
            )
            inventory_change = f"{_number_text(before)} → {_number_text(after)} ({delta})"
        else:
            inventory_change = f"изменение {_number_text(reduction)}"
        reasons = "; ".join(
            dict.fromkeys(
                str(value)
                for value in [
                    *list(item.get("constraintReasons") or []),
                    *list(item.get("limitations") or []),
                ]
                if str(value).strip()
            )
        )
        score = item.get("businessScore")
        cells = [
            str(index),
            str(item.get("executionStatus") or "blocked"),
            _DOMAIN_LABELS.get(str(item.get("family") or ""), str(item.get("family") or "")),
            str(item.get("target") or "портфель"),
            str(item.get("action") or "—"),
            _coverage_metric_label(item),
            inventory_change,
            _rub(item.get("expectedContributionProfitRub")),
            _rub(item.get("p10ContributionProfitRub")),
            _pct(item.get("stockoutProbability")),
            f"score {score}; {reasons}".strip("; "),
        ]
        lines.append(
            "| "
            + " | ".join(_public_text(cell).replace("|", "/") for cell in cells)
            + " |"
        )


def validate_scenario_completeness(
    markdown: str,
    query_plan: Mapping[str, Any] | None,
) -> list[str]:
    families = _required_scenario_families(query_plan)
    violations: list[str] = []
    headings = {
        "price": "## Проверка вариантов цены",
        "marketing": "## Рекламные сценарии",
        "supply": "## Поставочные сценарии",
    }
    for family in families:
        if headings[family] not in markdown:
            violations.append(f"missing {family} scenario section")

    plan = dict(query_plan or {})
    if plan.get("decision_kind") == "action_portfolio":
        if (plan.get("constraints") or {}).get("target_cover_days") is not None:
            goal_marker = "## Достижение цели"
            if goal_marker not in markdown:
                violations.append("missing goal attainment section")
            elif not re.search(
                r"Итог: (?:цель достигнута|цель не достигнута|достижение цели не доказано)",
                markdown,
            ):
                violations.append("missing explicit goal attainment verdict")
        marker = "## 5 приоритетных рекомендаций"
        if marker not in markdown:
            violations.append("missing five priority recommendations section")
        else:
            section = markdown.split(marker, 1)[1]
            next_heading = re.search(r"\n##\s", section)
            if next_heading:
                section = section[: next_heading.start()]
            rows = [
                line
                for line in section.splitlines()
                if re.match(r"^\|\s*\d+\s*\|", line.strip())
            ]
            if len(rows) != 5:
                violations.append(f"expected 5 priority recommendations, got {len(rows)}")
    return violations


def render_manager_advisory(
    advisory: Mapping[str, Any] | None,
    *,
    manager_decision: Mapping[str, Any] | None = None,
    query_plan: Mapping[str, Any] | None = None,
) -> str:
    source = dict(advisory or {})
    forecasts = list(source.get("forecasts") or [])
    daily_comparisons = list(source.get("dailyComparisons") or [])
    simulations = list(source.get("priceSimulations") or [])
    all_price_simulations = list(simulations)
    portfolio_optimization = dict(source.get("portfolioOptimization") or {})
    required_families = _required_scenario_families(query_plan)
    needs_five = dict(query_plan or {}).get("decision_kind") == "action_portfolio"
    needs_goal = bool(
        needs_five
        and (dict(query_plan or {}).get("constraints") or {}).get("target_cover_days") is not None
    )
    if not forecasts and not daily_comparisons and not simulations and not required_families and not needs_five and not needs_goal:
        return ""

    lines = [
        "## Расчётные рекомендации",
    ]
    _append_business_overview(
        lines,
        advisory=source,
        manager_decision=manager_decision,
        query_plan=query_plan,
    )
    lines.extend(
        [
            "",
            "Автоматическое исполнение может оставаться заблокированным; статус каждого варианта указан ниже.",
        ]
    )
    if dict(query_plan or {}).get("scope") == "portfolio":
        portfolio_targets = [
            str(item).strip()
            for item in source.get("portfolioTargets") or []
            if str(item).strip()
        ]
        requested_targets = [
            str(item).strip()
            for item in source.get("requestedTargets") or []
            if str(item).strip()
        ]
        coverage = dict(source.get("coverage") or {})
        if not coverage:
            coverage = _advisory_coverage(
                portfolio_targets=portfolio_targets,
                requested_targets=requested_targets,
                forecasts=[
                    dict(item) for item in forecasts if isinstance(item, Mapping)
                ],
                daily_comparisons=[
                    dict(item)
                    for item in daily_comparisons
                    if isinstance(item, Mapping)
                ],
                price_simulations=[
                    dict(item)
                    for item in simulations
                    if isinstance(item, Mapping)
                ],
                errors=[
                    dict(item)
                    for item in source.get("errors") or []
                    if isinstance(item, Mapping)
                ],
            )
        total_count = int(coverage.get("targetSkuCount") or 0)
        if total_count:
            lines.extend(
                [
                    "",
                    "Охвачено SKU: "
                    f"{int(coverage.get('coveredSkuCount') or 0)}/{total_count} "
                    "(получен артефакт либо явно зафиксирована ошибка/пробел).",
                    f"Всего SKU в целевой матрице: {total_count}.",
                ]
            )
            analyses = {
                str(item)
                for item in list((query_plan or {}).get("requested_analyses") or [])
            }
            if analyses.intersection({"pricing", "price_scenarios"}):
                lines.append(
                    "Численные ценовые сценарии: "
                    f"{int(coverage.get('numericPriceScenarioSkuCount') or 0)}/"
                    f"{total_count}."
                )
                missing_numeric = list(
                    coverage.get("missingNumericPriceScenarioTargets") or []
                )
                if missing_numeric:
                    lines.append(
                        "Без численного ценового сценария: "
                        + ", ".join(str(item) for item in missing_numeric)
                        + "."
                    )
            error_targets = list(coverage.get("errorTargets") or [])
            if error_targets:
                lines.append(
                    "SKU с явно зафиксированной ошибкой/пробелом: "
                    + ", ".join(str(item) for item in error_targets)
                    + "."
                )
            missing_artifacts = list(coverage.get("missingArtifactTargets") or [])
            if missing_artifacts:
                lines.append(
                    "Без артефакта и без записи об ошибке: "
                    + ", ".join(str(item) for item in missing_artifacts)
                    + "."
                )
    if needs_goal:
        _append_goal_attainment(
            lines,
            advisory=source,
            query_plan=query_plan,
        )
    if needs_five:
        _append_optimized_recommendations(
            lines,
            advisory=source,
            manager_decision=manager_decision,
            query_plan=query_plan,
        )
    price_only_portfolio = bool(
        dict(query_plan or {}).get("scope") == "portfolio"
        and "allowed_action_families" in dict(query_plan or {})
        and allowed_action_families(query_plan) == ["price"]
    )
    if price_only_portfolio:
        detail_targets = {
            str(item).strip()
            for item in list(source.get("focusTargets") or [])
            if str(item).strip()
        }
        if not detail_targets:
            detail_targets = {
                str(item.get("target") or "").strip()
                for item in _optimized_recommendations(
                    source,
                    manager_decision,
                    query_plan,
                )[:2]
                if str(item.get("target") or "").strip()
            }
        # Forecast-agent prose and daily rows remain trace evidence. The final
        # price answer keeps only focused detail plus one compact scenario grid.
        forecasts = []
        daily_comparisons = [
            item
            for item in daily_comparisons
            if _artifact_target(item) in detail_targets
        ]
        simulations = [
            item for item in simulations if _artifact_target(item) in detail_targets
        ]
    if forecasts:
        lines.extend(
            [
                "",
                "Примечание к прогнозным блокам: их показатели покрытия относятся к отдельному прогнозному процессу и не заменяют текущее покрытие по продажам 7д из SKU-матрицы.",
            ]
        )
    for forecast in forecasts:
        target = forecast.get("vendorCode") or forecast.get("target")
        lines.extend(
            [
                "",
                f"### SKU {target}",
                "",
                _public_text(
                    forecast.get("finalRecommendation")
                    or forecast.get("summary")
                    or "Расчёт выполнен."
                ),
            ]
        )
        why = _public_text(forecast.get("decisionWhy"))
        if why:
            lines.append(f"Причина: {why}")
        effect = forecast.get("economicEffectRub")
        if effect is not None:
            lines.append(f"Ожидаемый сценарный эффект: {_rub(effect)} за горизонт расчёта.")
        blockers = [_public_text(item) for item in forecast.get("blockedBy") or [] if str(item)]
        if blockers:
            lines.append("Ограничения: " + "; ".join(blockers) + ".")
        for step in (forecast.get("resolutionPlan") or [])[:3]:
            instruction = _public_text(step.get("instruction"))
            stop = _public_text(step.get("stopCondition"))
            if instruction:
                lines.append(f"- Действие: {instruction}")
            if stop:
                lines.append(f"  Стоп-условие: {stop}")

    daily_comparisons = [item for item in daily_comparisons if item.get("rows")]
    if daily_comparisons:
        lines.extend(
            [
                "",
                "## Цена, скидка, конверсия и продажи по дням",
                "",
                "Дневное сопоставление показывает совместное движение метрик; само по себе оно не доказывает причинность.",
            ]
        )
        for comparison in daily_comparisons:
            target = comparison.get("vendorCode") or comparison.get("target")
            lines.extend(
                [
                    "",
                    f"### SKU {target}",
                    "",
                    "| Дата | Цена продавца | Скидка | CR карточка→корзина | CR корзина→заказ | Заказы | Продажи |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in comparison.get("rows") or []:
                price = row.get("sellerEffectivePriceRub")
                if price is None:
                    price = row.get("observedSellerPriceRub")
                lines.append(
                    "| {date} | {price} | {discount} | {cart} | {order} | {orders} | {sales} |".format(
                        date=row.get("date") or "—",
                        price=_rub(price),
                        discount=_pct_points(row.get("sellerDiscountPct")),
                        cart=_pct_points(row.get("addToCartConversionPct")),
                        order=_pct_points(row.get("cartToOrderConversionPct")),
                        orders=_number_text(row.get("orders")),
                        sales=_number_text(row.get("realizedSales")),
                    )
                )

    if simulations or "price" in required_families:
        lines.extend(
            [
                "",
                "## Проверка вариантов цены",
                "",
                "Это наблюдательная shadow-симуляция, а не разрешение на автоматическую смену цены.",
            ]
        )
        for simulation in simulations:
            target = simulation.get("vendorCode") or simulation.get("target")
            active_context = [
                _CONTEXT_FEATURE_LABELS.get(str(item), str(item))
                for item in simulation.get("activeContextFeatures") or []
            ]
            context_gaps = list(simulation.get("contextDataGaps") or [])
            future_promotions = [
                item
                for item in simulation.get("futurePromotionEvents") or []
                if isinstance(item, Mapping) and item.get("in_action") is True
            ]
            completed_candidates = [
                candidate
                for candidate in simulation.get("candidates") or []
                if candidate.get("status") == "completed"
            ]
            completed_price_moves = [
                candidate
                for candidate in completed_candidates
                if candidate.get("operation") != "hold_price"
            ]
            hypothesis_candidates = [
                dict(candidate)
                for candidate in simulation.get("hypothesisCandidates") or []
                if isinstance(candidate, Mapping)
            ]
            if not completed_price_moves:
                if hypothesis_candidates:
                    confidence = str(
                        simulation.get("hypothesisConfidence") or "не определена"
                    )
                    confidence_score = simulation.get("hypothesisConfidenceScore")
                    assumptions = [
                        _public_text(item)
                        for item in simulation.get("hypothesisAssumptions") or []
                        if str(item).strip()
                    ]
                    evidence_gaps = [
                        _public_text(item)
                        for item in simulation.get("hypothesisEvidenceGaps") or []
                        if str(item).strip()
                    ]
                    lines.extend(
                        [
                            "",
                            f"### SKU {target}: гипотезы чувствительности",
                            "",
                            (
                                "Строгая модель не допустила ценовые варианты к исполнению. "
                                "Ниже показаны расчётные гипотезы для сравнения и проверки; "
                                "автоматическое исполнение остаётся отдельным и заблокировано."
                            ),
                            (
                                f"Уверенность гипотезы: {confidence} "
                                f"({_pct(confidence_score)})."
                                if confidence_score is not None
                                else f"Уверенность гипотезы: {confidence}."
                            ),
                            (
                                "Допущения: " + "; ".join(assumptions) + "."
                                if assumptions
                                else "Допущения явно не зафиксированы."
                            ),
                            (
                                "Что нужно подтвердить: " + "; ".join(evidence_gaps) + "."
                                if evidence_gaps
                                else "Дополнительные пробелы данных не указаны."
                            ),
                            "",
                            "| Тип | Ранг | Цена | Изменение к текущей | Ожидаемые продажи | Средний эффект | p10 | Вероятность плюса | Риск OOS |",
                            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                        ]
                    )
                    current_price = simulation.get("currentPriceRub")
                    for candidate in hypothesis_candidates:
                        price = candidate.get("priceRub")
                        price_move_pct = None
                        if price is not None and current_price:
                            price_move_pct = float(price) / float(current_price) - 1
                        lines.append(
                            "| {kind} | {rank} | {price} | {move} | {units} | {mean} | {p10} | {positive} | {stockout} |".format(
                                kind="гипотеза / sensitivity",
                                rank=candidate.get("rank") or "—",
                                price=_rub(price),
                                move=_pct(price_move_pct),
                                units=_number_text(candidate.get("expectedUnits")),
                                mean=_rub(candidate.get("mean")),
                                p10=_rub(candidate.get("p10")),
                                positive=_pct(candidate.get("probPositive")),
                                stockout=_pct(candidate.get("stockout")),
                            )
                        )
                    continue
                lines.extend(
                    [
                        "",
                        f"### SKU {target}",
                        "",
                        "Статус: ценовые варианты заблокированы guardrails; пустая таблица не публикуется.",
                        (
                            "В модель спроса загружены: " + ", ".join(active_context) + "."
                            if active_context
                            else "Контекстные факторы не прошли минимальный порог истории."
                        ),
                        (
                            "Ограничения данных: "
                            + "; ".join(
                                dict.fromkeys(
                                    _CONTEXT_GAP_LABELS.get(str(item), str(item))
                                    for item in context_gaps
                                )
                            )
                            + "."
                            if context_gaps
                            else "Дополнительных контекстных ограничений нет."
                        ),
                        (
                            "Подтверждённые SKU-акции в горизонте: "
                            + str(len(future_promotions))
                            + "."
                            if future_promotions
                            else "Подтверждённых SKU-акций в горизонте нет."
                        ),
                        (
                            "Подтверждённые будущие поставки с ETA: "
                            + str(
                                len(
                                    [
                                        item
                                        for item in simulation.get("futureInboundEvents") or []
                                        if isinstance(item, Mapping)
                                        and item.get("includeInCapacity") is True
                                    ]
                                )
                            )
                            + "."
                        ),
                        "Для следующего расчёта нужно закрыть указанные ограничения и повторить shadow-симуляцию.",
                    ]
                )
                continue
            display_candidates = _merge_price_candidates(
                completed_candidates,
                hypothesis_candidates,
            )
            optimized_prices = {
                str(item.get("scenarioId") or ""): item
                for item in list(portfolio_optimization.get("scenarios") or [])
                if isinstance(item, Mapping) and item.get("family") == "price"
            }
            lines.extend(
                [
                    "",
                    f"### SKU {target}: {simulation.get('sampleCount') or 0} расчётов на вариант",
                    "",
                    (
                        "Строгие completed-варианты имеют приоритет; варианты hypothesis "
                        "добавлены только вместо заблокированных цен и доступны лишь для теста."
                        if hypothesis_candidates
                        else "Все показанные варианты прошли строгий расчётный gate."
                    ),
                    (
                        "В прогноз спроса включены: " + ", ".join(active_context) + "."
                        if active_context
                        else "Контекстные факторы не прошли порог истории; расчёт использует базовую модель цены и спроса."
                    ),
                    (
                        f"Контекстные ограничения: {len(context_gaps)}; они сохранены в feature coverage и не подменяются нулями."
                        if context_gaps
                        else "Контекстные ограничения отсутствуют."
                    ),
                    (
                        "Подтверждённые акции в горизонте: "
                        + "; ".join(
                            f"{item.get('start_date') or '—'}–{item.get('end_date') or '—'}, "
                            f"скидка {_pct_points(item.get('discount_pct'))}"
                            for item in future_promotions
                        )
                        + "."
                        if future_promotions
                        else "Подтверждённых SKU-акций в горизонте нет."
                    ),
                    "",
                    "| Ранг | Цена | Снижение прогнозного покрытия (500 симуляций), дней | Средний эффект | p10 | Вероятность плюса | Риск OOS | Допуск |",
                    "|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for candidate in display_candidates:
                candidate_price = candidate.get("priceRub")
                scenario_key = f"price:{target}:{float(candidate_price):g}"
                optimized = optimized_prices.get(scenario_key) or {}
                if candidate.get("status") == "hypothesis":
                    reasons = list(optimized.get("constraintReasons") or [])
                    admission = "гипотеза / только тест"
                    if reasons:
                        admission += ": " + "; ".join(str(item) for item in reasons)
                elif candidate.get("operation") == "hold_price":
                    admission = "базовый вариант"
                elif optimized.get("eligible") is True:
                    admission = "допущен"
                else:
                    reasons = list(optimized.get("constraintReasons") or [])
                    admission = (
                        "отклонён: " + "; ".join(str(item) for item in reasons)
                        if reasons
                        else "не проверен оптимизатором"
                    )
                lines.append(
                    "| {rank} | {price} | {days} | {mean} | {p10} | {positive} | {stockout} | {admission} |".format(
                        rank=candidate.get("rank") or "—",
                        price=_rub(candidate_price),
                        days=_number_text(optimized.get("inventoryDaysReduction")),
                        mean=_rub(
                            candidate.get("meanEffectRub", candidate.get("mean"))
                        ),
                        p10=_rub(
                            candidate.get("p10EffectRub", candidate.get("p10"))
                        ),
                        positive=_pct(
                            candidate.get(
                                "probabilityPositive",
                                candidate.get("probPositive"),
                            )
                        ),
                        stockout=_pct(
                            candidate.get(
                                "stockoutProbability",
                                candidate.get("stockout"),
                            )
                        ),
                        admission=admission,
                    )
                )
            guardrail = next(
                (
                    item.get("competitorGuardrail")
                    for item in optimized_prices.values()
                    if isinstance(item.get("competitorGuardrail"), Mapping)
                ),
                None,
            )
            if isinstance(guardrail, Mapping):
                if guardrail.get("status") == "applied":
                    lines.append(
                        "Конкурентный guardrail применён: медиана сопоставимых цен "
                        f"{_rub(guardrail.get('competitorMedianBuyerPriceRub'))}."
                    )
                elif guardrail.get("status") == "applied_competitor_temporal_fallback":
                    lines.append(
                        "Конкурентный коридор рассчитан по данным БД: медиана вчера "
                        f"{_rub(guardrail.get('competitorYesterdayMedianRub'))}, сегодня "
                        f"{_rub(guardrail.get('competitorTodayMedianRub'))}, изменение "
                        f"{_number_text(guardrail.get('competitorYesterdayToTodayChangePct'))}%."
                    )
                elif guardrail.get("status") == "applied_competitor_single_day_fallback":
                    lines.append(
                        "В БД есть только один конкурентный срез; применён консервативный "
                        "коридор ±2,5% вокруг текущей цены."
                    )
                elif guardrail.get("status") == "excluded_low_quality":
                    lines.append(
                        "Цены конкурентов не влияли на выбор: срез исключён из-за "
                        "несопоставимой ценовой базы или недостаточного покрытия."
                    )
                elif guardrail.get("status") == "missing":
                    lines.append("Цены сопоставимых конкурентов отсутствуют.")
                elif guardrail.get("status") == "applied_temporal_fallback":
                    lines.append(
                        "Конкурентный срез заменён временным ценовым коридором: "
                        f"вчера {_rub(guardrail.get('yesterdayPriceRub'))}, сегодня "
                        f"{_rub(guardrail.get('todayPriceRub'))}, изменение "
                        f"{_number_text(guardrail.get('yesterdayToTodayChangePct'))}%."
                    )
                elif guardrail.get("status") == "applied_current_anchor_fallback":
                    lines.append(
                        "История и конкурентный срез отсутствуют; применён консервативный "
                        "коридор вокруг текущей цены."
                    )
            winner = next(
                (
                    candidate
                    for candidate in completed_candidates
                    if (
                        optimized_prices.get(
                            f"price:{target}:{float(candidate.get('priceRub')):g}",
                            {},
                        ).get("eligible")
                        is True
                    )
                ),
                None,
            )
            if winner is not None:
                lines.append(
                    f"Лучший допустимый расчётный вариант: {_rub(winner.get('priceRub'))}; "
                    f"средний эффект {_rub(winner.get('meanEffectRub'))}, "
                    f"p10 {_rub(winner.get('p10EffectRub'))}."
                )
            else:
                lines.append(
                    "Допустимого изменения цены нет: все рассчитанные варианты "
                    "отклонены ограничениями или не прошли проверку данных."
                )
        if not simulations:
            lines.extend(
                [
                    "",
                    "Статус: ценовые варианты не рассчитаны; пустая таблица не публикуется. Нужны актуальные цена, скидка, COGS и история продаж.",
                ]
            )

    if price_only_portfolio:
        _append_compact_price_matrix(
            lines,
            simulations=[
                dict(item)
                for item in all_price_simulations
                if isinstance(item, Mapping)
            ],
            portfolio_optimization=portfolio_optimization,
        )

    for family in ("marketing", "supply"):
        if family in required_families:
            _append_family_scenarios(
                lines,
                family=family,
                manager_decision=manager_decision,
                portfolio_optimization=portfolio_optimization,
            )

    return "\n".join(lines).strip()


def _rub(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ₽".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _signed_rub(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "—"
    prefix = "+" if amount > 0 else ""
    return (prefix + f"{amount:,.0f} ₽").replace(",", " ")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _pct_points(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _number_text(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "—"


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "build_manager_advisory",
    "render_manager_advisory",
    "should_build_manager_advisory",
    "validate_scenario_completeness",
]
