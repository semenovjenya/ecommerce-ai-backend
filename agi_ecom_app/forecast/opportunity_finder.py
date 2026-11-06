from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.models import ForecastActionFamily, GrowthOpportunity
from agi_ecom_app.services import strategic_db


def find_growth_opportunities(
    *,
    period_days: int = 7,
    horizon_days: int = 14,
    limit: int = 20,
    focus: str = "margin_revenue",
) -> list[GrowthOpportunity]:
    """Find growth candidates; final decisions still belong to forecast calculators."""

    period_days = _bounded_int(period_days, default=7, low=1, high=31)
    horizon_days = _bounded_int(horizon_days, default=14, low=1, high=90)
    limit = _bounded_int(limit, default=20, low=1, high=100)
    fetch_limit = max(limit, 20)

    sales_change = _safe_call(
        strategic_db.get_sales_sku_revenue_change,
        period_days=period_days,
        limit=fetch_limit,
    )
    sales_overview = _safe_call(strategic_db.get_sales_overview, limit=min(fetch_limit, 10))
    supply = _safe_call(strategic_db.get_supply_overview, limit=min(fetch_limit, 10))
    price = _safe_call(strategic_db.get_price_decision_overview, limit=fetch_limit)
    if not _rows(price, "rows"):
        price = _safe_call(strategic_db.get_price_competitor_overview, limit=fetch_limit)
    finance = _safe_call(strategic_db.get_finance_overview, limit=min(fetch_limit, 10))
    marketing = _safe_call(strategic_db.get_marketing_action_matrix, days=period_days, limit=fetch_limit)

    sales_rows = _rows(sales_change, "rows")
    conversion_rows = _rows(sales_overview, "conversion_watchlist")
    supply_rows = _rows(supply, "rows")
    price_rows = _rows(price, "rows")
    finance_rows = [
        *_rows(finance, "unit_economics_watchlist"),
        *_rows(finance, "margin_watchlist"),
        *_rows(finance, "fee_pressure_rows"),
    ]

    sales_by_key = _index_by_entity(sales_rows)
    supply_by_key = _index_by_entity(supply_rows)
    price_by_key = _index_by_entity(price_rows)
    finance_by_key = _index_by_entity(finance_rows)
    conversion_by_key = _index_by_entity(conversion_rows)

    opportunities: list[GrowthOpportunity] = []
    for row in sales_rows:
        key = _entity_key(row)
        if not key:
            continue
        sales = row
        stock = supply_by_key.get(key, {})
        price_row = price_by_key.get(key, {})
        finance_row = finance_by_key.get(key, {})
        conversion_row = conversion_by_key.get(key, {})
        opportunities.extend(
            _sku_opportunities(
                key=key,
                sales=sales,
                stock=stock,
                price=price_row,
                finance=finance_row,
                conversion=conversion_row,
                period_days=period_days,
                horizon_days=horizon_days,
                focus=focus,
            )
        )

    for row in supply_rows:
        key = _entity_key(row)
        if not key or key in sales_by_key:
            continue
        opportunities.extend(
            _sku_opportunities(
                key=key,
                sales={},
                stock=row,
                price=price_by_key.get(key, {}),
                finance=finance_by_key.get(key, {}),
                conversion=conversion_by_key.get(key, {}),
                period_days=period_days,
                horizon_days=horizon_days,
                focus=focus,
            )
        )

    opportunities.extend(_marketing_opportunities(marketing, period_days=period_days, horizon_days=horizon_days))

    deduped = _dedupe_opportunities(opportunities)
    deduped.sort(key=lambda item: item.score, reverse=True)
    return deduped[:limit]


def _sku_opportunities(
    *,
    key: str,
    sales: dict[str, Any],
    stock: dict[str, Any],
    price: dict[str, Any],
    finance: dict[str, Any],
    conversion: dict[str, Any],
    period_days: int,
    horizon_days: int,
    focus: str,
) -> list[GrowthOpportunity]:
    current_revenue = _number(sales.get("current_revenue"))
    previous_revenue = _number(sales.get("previous_revenue"))
    delta_revenue = _number(sales.get("delta_revenue"))
    current_sales = _number(sales.get("current_sales"))
    previous_sales = _number(sales.get("previous_sales"))
    days_of_stock = _number(stock.get("days_of_stock"))
    stock_qty = _number(stock.get("stock_qty"))
    sales_7d = _number(stock.get("sales_7d_units")) or current_sales
    reorder_qty = _number(stock.get("reorder_qty_recommended"))
    contribution_margin = _number(finance.get("contribution_margin")) or _number(price.get("expected_net_profit_7d"))
    margin_before_cogs_pct = _number(finance.get("payout_margin_before_cogs_pct")) or _number(price.get("expected_margin_pct"))
    price_status = _string(price.get("price_status"))
    competitor_count = _number(price.get("competitor_count"))
    conversion_rate = _number(conversion.get("cart_to_order_conversion"))
    blockers = _sku_blockers(stock=stock, price=price, finance=finance)
    sku = _target_id(sales, stock, price, finance, default=key)
    facts = {
        "sales": sales,
        "supply": stock,
        "price": price,
        "finance": finance,
        "conversion": conversion,
        "focus": focus,
    }

    result: list[GrowthOpportunity] = []
    if (
        (delta_revenue or 0) > 0
        and (current_revenue or 0) > 0
        and _margin_ok(contribution_margin, margin_before_cogs_pct)
        and _coverage_ok(days_of_stock)
        and price_status not in {"above_market", "thin_data"}
    ):
        score = 80 + min((delta_revenue or 0) / max(current_revenue or 1, 1) * 20, 20)
        result.append(
            _opportunity(
                sku=sku,
                family="scale_profitable_sku_ads",
                score=score,
                reason="SKU растет, экономика не блокирует масштабирование, запас выдерживает горизонт.",
                focus=["revenue_uplift", "contribution_margin", "drr", "stock_coverage"],
                blockers=blockers,
                source_facts=facts,
                period_days=period_days,
                horizon_days=horizon_days,
            )
        )

    if (
        price_status == "above_market"
        and (competitor_count or 0) >= 5
        and ((delta_revenue or 0) < 0 or (current_sales or 0) < (previous_sales or 0))
    ):
        result.append(
            _opportunity(
                sku=sku,
                family="reprice_with_margin_guardrail",
                score=78 + min(abs(delta_revenue or 0) / max(previous_revenue or 1, 1) * 20, 20),
                reason="SKU проседает, а цена выше конкурентного коридора; нужно проверить безопасную цену через маржу.",
                focus=["price_gap_pct", "safe_price_floor", "sales_recovery", "margin_after_price"],
                blockers=blockers,
                source_facts=facts,
                period_days=period_days,
                horizon_days=horizon_days,
            )
        )

    confirmed_stockout = stock_qty is not None and stock_qty <= 0 and (sales_7d or 0) > 0
    confirmed_low_coverage_reorder = (
        (reorder_qty or 0) > 0
        and days_of_stock is not None
        and days_of_stock < 21
    )
    if confirmed_stockout or confirmed_low_coverage_reorder:
        result.append(
            _opportunity(
                sku=sku,
                family="reorder_for_growth",
                score=82 if (stock_qty or 0) <= 0 else 72,
                reason="Есть скорость продаж и риск дефицита; рост нужно защищать дозаказом.",
                focus=["reorder_qty", "coverage_days", "lost_sales_risk", "contribution_margin"],
                blockers=[item for item in blockers if item not in {"supply_risk"}],
                source_facts=facts,
                period_days=period_days,
                horizon_days=horizon_days,
            )
        )

    if _overstock_candidate(stock=stock, sales=sales):
        result.append(
            _opportunity(
                sku=sku,
                family="overstock_liquidation",
                score=64,
                reason="Запас высокий относительно продаж; скидка или перераспределение требуют price/finance guardrail.",
                focus=["stock_turnover", "safe_discount", "storage_cost", "margin_after_discount"],
                blockers=blockers,
                source_facts=facts,
                period_days=period_days,
                horizon_days=horizon_days,
            )
        )

    if conversion_rate is not None and conversion_rate < 0.08:
        result.append(
            _opportunity(
                sku=sku,
                family="recover_conversion_drop",
                score=68,
                reason="Воронка карточки просела; нужен разбор конверсии через sales, marketing, price и supply.",
                focus=["cart_to_order_delta", "visibility", "price_gap", "stock_coverage"],
                blockers=blockers,
                source_facts=facts,
                period_days=period_days,
                horizon_days=horizon_days,
            )
        )

    return result


def _marketing_opportunities(
    marketing: dict[str, Any],
    *,
    period_days: int,
    horizon_days: int,
) -> list[GrowthOpportunity]:
    result: list[GrowthOpportunity] = []
    for row in [*_rows(marketing, "query_actions"), *_rows(marketing, "campaign_actions")]:
        action_mode = _string(row.get("action_mode")) or ""
        status = _string(row.get("status")) or ""
        entity_id = str(row.get("entity_id") or row.get("query_text") or row.get("norm_query") or row.get("advert_id") or "").strip()
        if not entity_id or action_mode not in {"reduce", "scale_review", "needs_review", "test"}:
            continue
        is_query = bool(row.get("query_text") or row.get("norm_query"))
        blockers = list(row.get("blockers") or [])
        if _string(row.get("demand_source_status")) == "unpopulated" and action_mode in {"scale_review", "needs_review", "test"}:
            blockers.append("marketing_demand_mart_unpopulated")
        if _string(row.get("confidence")) == "low":
            blockers.append("marketing_low_confidence")
        if action_mode == "test":
            blockers.append("marketing_test_only")
        unique_blockers = list(dict.fromkeys(str(item) for item in blockers if item))
        query_band = _string(row.get("query_band"))
        score = {
            "reduce": 72,
            "scale_review": 68,
            "needs_review": 62,
            "test": 60,
        }.get(action_mode, 60)
        reason = {
            "reduce": "Найден рекламный объект с waste/cut сигналом.",
            "scale_review": "Найден winner/scale-review сигнал, но рост нужно подтверждать консервативно.",
            "needs_review": "Найден рекламный сигнал, но confidence или demand-контур не дают auto-scale.",
            "test": "Найден test-only маркетинговый сигнал без права на auto-scale.",
        }.get(action_mode, "Найден маркетинговый action-кандидат.")
        result.append(
            GrowthOpportunity(
                opportunity_id=f"normquery_bid_optimization:{entity_id}",
                action_family="normquery_bid_optimization",
                score=score,
                evidence_summary=f"{reason} Объект: {entity_id}.",
                blockers=unique_blockers,
                affected_queries=[entity_id] if is_query else [],
                affected_campaigns=[] if is_query else [entity_id],
                expected_metric_focus=["cpc", "ad_orders", "drr", "query_efficiency"],
                target_mode="query" if is_query else "campaign",
                primary_domain="marketing",
                growth_reason=reason if not query_band else f"{reason} Query band: {query_band}.",
                source_facts={"marketing_action": row, "period_days": period_days, "horizon_days": horizon_days},
            )
        )
    return result


def _opportunity(
    *,
    sku: str,
    family: ForecastActionFamily,
    score: float,
    reason: str,
    focus: list[str],
    blockers: list[str],
    source_facts: dict[str, Any],
    period_days: int,
    horizon_days: int,
) -> GrowthOpportunity:
    return GrowthOpportunity(
        opportunity_id=f"{family}:{sku}",
        action_family=family,
        score=round(max(min(score, 100), 0), 2),
        evidence_summary=reason,
        blockers=list(dict.fromkeys(blockers)),
        affected_skus=[sku],
        expected_metric_focus=focus,
        target_mode="sku",
        primary_domain=_primary_domain_for_family(family),
        growth_reason=reason,
        source_facts={**source_facts, "period_days": period_days, "horizon_days": horizon_days},
    )


def _primary_domain_for_family(family: str) -> str:
    mapping = {
        "scale_profitable_sku_ads": "marketing",
        "reprice_with_margin_guardrail": "price",
        "reorder_for_growth": "supply",
        "overstock_liquidation": "supply",
        "normquery_bid_optimization": "marketing",
        "recover_conversion_drop": "sales",
    }
    return mapping.get(family, "sales")


def _sku_blockers(*, stock: dict[str, Any], price: dict[str, Any], finance: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    stock_qty = _number(stock.get("stock_qty"))
    days = _number(stock.get("days_of_stock"))
    has_price_profit_signal = (
        _number(price.get("cost_price_rub")) is not None
        and (
            _number(price.get("expected_margin_pct")) is not None
            or _number(price.get("net_profit_rub_current")) is not None
        )
    )
    if stock and ((stock_qty is not None and stock_qty <= 0) or (days is not None and days < 21)):
        blockers.append("supply_risk")
    if price and (_string(price.get("price_status")) == "thin_data" or (_number(price.get("competitor_count")) or 0) < 5):
        blockers.append("thin_price_data")
    if _string(price.get("decision_confidence")) == "low":
        blockers.append("low_price_decision_confidence")
    if "stale" in (_string(price.get("decision_reason_code")) or "").lower():
        blockers.append("stale_price_decision")
    if finance and _number(finance.get("missing_cost_rows")) and (_number(finance.get("missing_cost_rows")) or 0) > 0:
        blockers.append("missing_cogs")
    if not finance and not has_price_profit_signal:
        blockers.append("missing_finance_evidence")
    return blockers


def _margin_ok(contribution_margin: float | None, margin_before_cogs_pct: float | None) -> bool:
    if contribution_margin is not None:
        return contribution_margin > 0
    if margin_before_cogs_pct is not None:
        return margin_before_cogs_pct > 0
    return False


def _coverage_ok(days_of_stock: float | None) -> bool:
    return days_of_stock is None or days_of_stock >= 21


def _overstock_candidate(*, stock: dict[str, Any], sales: dict[str, Any]) -> bool:
    stock_qty = _number(stock.get("stock_qty"))
    days = _number(stock.get("days_of_stock"))
    sales_7d = _number(stock.get("sales_7d_units")) or _number(sales.get("current_sales"))
    return bool(stock_qty is not None and stock_qty >= 60 and ((sales_7d or 0) <= 0 or (days or 0) >= 90))


def _dedupe_opportunities(opportunities: list[GrowthOpportunity]) -> list[GrowthOpportunity]:
    result: dict[str, GrowthOpportunity] = {}
    for item in opportunities:
        current = result.get(item.opportunity_id)
        if current is None or item.score > current.score:
            result[item.opportunity_id] = item
    return list(result.values())


def _index_by_entity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _entity_key(row)
        if key and key not in result:
            result[key] = row
    return result


def _entity_key(row: dict[str, Any]) -> str | None:
    for key in ("vendor_code", "nm_id", "source_nm_id", "entity_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _target_id(*rows: dict[str, Any], default: str) -> str:
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("vendor_code", "nm_id", "source_nm_id"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


def _safe_call(func: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        value = func(**kwargs)
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}
    return value if isinstance(value, dict) else {}


def _rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key) if isinstance(payload, dict) else None
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _bounded_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min(parsed, high), low)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
