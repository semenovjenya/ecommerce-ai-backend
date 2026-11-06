from __future__ import annotations

from typing import Any, Mapping


DEFICIT_COVER_DAYS = 14.0
TARGET_COVER_DAYS = 30.0
MODERATE_EXCESS_COVER_DAYS = 60.0


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_coverage_days(
    *,
    stock_qty: Any,
    sales_units: Any,
    period_days: Any,
    precision: int = 1,
) -> float | None:
    """Return stock coverage using one explicit, deterministic formula.

    ``stock_qty / (sales_units / period_days)`` is intentionally kept outside
    of LLM prompts. A zero-sales period has no finite coverage estimate and is
    returned as ``None`` instead of an invented large number.
    """

    stock = _number(stock_qty)
    sales = _number(sales_units)
    period = _number(period_days)
    if stock is None or sales is None or period is None or period <= 0:
        return None
    if sales <= 0:
        return None
    return round(max(stock, 0.0) / (sales / period), precision)


def classify_coverage_days(days: Any) -> str:
    """Classify coverage using the product contract requested by QueryPlan."""

    value = _number(days)
    if value is None:
        return "покрытие не определено"
    if value < DEFICIT_COVER_DAYS:
        return "дефицит <14 дней"
    if value <= TARGET_COVER_DAYS:
        return "норма 14–30 дней"
    if value <= MODERATE_EXCESS_COVER_DAYS:
        return "умеренный избыток 31–60 дней"
    return "критический избыток >60 дней"


def classify_availability(stock_qty: Any) -> str:
    stock = _number(stock_qty)
    if stock is None:
        return "остаток не подтверждён"
    if stock <= 0:
        return "нет доступного остатка"
    return "доступный остаток есть"


def normalize_turnover_row(
    row: Mapping[str, Any],
    *,
    period_days: int,
) -> dict[str, Any]:
    """Recalculate turnover coverage and attach non-overlapping status axes."""

    normalized = dict(row)
    coverage = calculate_coverage_days(
        stock_qty=normalized.get("avg_stock_qty"),
        sales_units=normalized.get("sales_units"),
        period_days=period_days,
    )
    normalized["days_on_hand"] = coverage
    normalized["turnover_band"] = classify_coverage_days(coverage)
    normalized["availability_status"] = classify_availability(
        normalized.get("current_stock_qty")
    )
    return normalized


def normalize_current_coverage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recalculate the 7-day current-stock coverage returned by Supply."""

    normalized = dict(row)
    coverage = calculate_coverage_days(
        stock_qty=normalized.get("stock_qty"),
        sales_units=normalized.get("sales_7d_units"),
        period_days=7,
    )
    normalized["days_of_stock"] = coverage
    normalized["coverage_band"] = classify_coverage_days(coverage)
    normalized["availability_status"] = classify_availability(
        normalized.get("stock_qty")
    )
    return normalized


__all__ = [
    "DEFICIT_COVER_DAYS",
    "MODERATE_EXCESS_COVER_DAYS",
    "TARGET_COVER_DAYS",
    "calculate_coverage_days",
    "classify_availability",
    "classify_coverage_days",
    "normalize_current_coverage_row",
    "normalize_turnover_row",
]
