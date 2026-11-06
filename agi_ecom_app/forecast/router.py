from __future__ import annotations

from agi_ecom_app.forecast.models import ForecastPlan


def select_workers(plan: ForecastPlan) -> list[str]:
    mapping = {
        "stop_or_reduce_campaigns": ["forecast_marketing_agent"],
        "availability_before_scale_guardrail": ["forecast_supply_agent", "forecast_marketing_agent"],
        "weak_sku_demand_recovery": [
            "forecast_sales_agent",
            "forecast_catalog_agent",
            "forecast_marketing_agent",
            "forecast_supply_agent",
            "forecast_price_agent",
            "forecast_finance_agent",
        ],
        "sales_trend_guardrail": ["forecast_sales_agent", "forecast_supply_agent", "forecast_price_agent"],
        "price_positioning_guardrail": ["forecast_price_agent", "forecast_sales_agent"],
        "scale_profitable_sku_ads": [
            "forecast_sales_agent",
            "forecast_marketing_agent",
            "forecast_supply_agent",
            "forecast_price_agent",
            "forecast_finance_agent",
        ],
        "reprice_with_margin_guardrail": [
            "forecast_price_agent",
            "forecast_finance_agent",
            "forecast_sales_agent",
            "forecast_marketing_agent",
            "forecast_supply_agent",
        ],
        "reorder_for_growth": ["forecast_supply_agent", "forecast_sales_agent", "forecast_finance_agent"],
        "overstock_liquidation": ["forecast_supply_agent", "forecast_sales_agent", "forecast_price_agent", "forecast_finance_agent"],
        "normquery_bid_optimization": ["forecast_marketing_agent", "forecast_sales_agent", "forecast_supply_agent", "forecast_finance_agent"],
        "recover_conversion_drop": [
            "forecast_sales_agent",
            "forecast_catalog_agent",
            "forecast_marketing_agent",
            "forecast_supply_agent",
            "forecast_price_agent",
            "forecast_finance_agent",
        ],
    }
    return list(mapping.get(plan.action_family, []))


def select_calculator(plan: ForecastPlan) -> str | None:
    supported = {
        "stop_or_reduce_campaigns",
        "availability_before_scale_guardrail",
        "weak_sku_demand_recovery",
        "sales_trend_guardrail",
        "price_positioning_guardrail",
        "scale_profitable_sku_ads",
        "reprice_with_margin_guardrail",
        "reorder_for_growth",
        "overstock_liquidation",
        "normquery_bid_optimization",
        "recover_conversion_drop",
    }
    if plan.action_family in supported:
        return plan.action_family
    return None
