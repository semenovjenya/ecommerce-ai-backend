from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import ForecastHypothesis
from agi_ecom_app.services import strategic_db


def calculate_sales_scenario(
    hypothesis: ForecastHypothesis,
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    target_results = [
        _load_sales_target(target, period_days=hypothesis.period_days or 7)
        for target in hypothesis.targets
    ]
    found_results = [item for item in target_results if item["found"]]
    missing_targets = [item["target"] for item in target_results if not item["found"]]

    current_sales = round(sum(_number(item.get("current_sales")) or 0 for item in found_results), 2)
    previous_sales = round(sum(_number(item.get("previous_sales")) or 0 for item in found_results), 2)
    current_revenue = round(sum(_number(item.get("current_revenue")) or 0 for item in found_results), 2)
    previous_revenue = round(sum(_number(item.get("previous_revenue")) or 0 for item in found_results), 2)
    delta_sales = round(current_sales - previous_sales, 2)
    delta_revenue = round(current_revenue - previous_revenue, 2)
    trend_direction = _trend_direction(delta_sales=delta_sales, delta_revenue=delta_revenue)
    risk_level = _risk_level(trend_direction=trend_direction, delta_sales=delta_sales, current_sales=current_sales)

    blocked_by: list[str] = []
    if not hypothesis.targets:
        blocked_by.append("missing_sales_targets")
    if missing_targets:
        blocked_by.append("sales_target_not_found")
    if not found_results:
        blocked_by.append("missing_sales_snapshot")

    confidence = "high"
    decision_status = "completed"
    if not found_results:
        confidence = "low"
        decision_status = "blocked"
    elif missing_targets or previous_sales == 0 or current_sales == 0:
        confidence = "medium"
        decision_status = "needs_review"

    action_text = _action_text(trend_direction=trend_direction, risk_level=risk_level)
    return {
        "calculator": "sales",
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "period_days": hypothesis.period_days,
            "targets_checked": len(hypothesis.targets),
            "targets_found": len(found_results),
            "current_sales": current_sales,
            "previous_sales": previous_sales,
            "current_revenue": current_revenue,
            "previous_revenue": previous_revenue,
            "delta_sales": delta_sales,
            "delta_revenue": delta_revenue,
        },
        "action_metrics": {
            "recommended_action": action_text,
            "trend_direction": trend_direction,
            "risk_level": risk_level,
        },
        "delta_metrics": {
            "delta_sales": delta_sales,
            "delta_revenue": delta_revenue,
            "trend_direction": trend_direction,
            "risk_level": risk_level,
        },
        "target_results": target_results,
        "assumptions": [
            "sales trend compares current period to previous equal period",
            "action scenario is a guardrail and does not assume uplift",
            *evidence_bundle.blocking_gaps[:3],
        ],
        "decision": _decision_text(trend_direction=trend_direction, risk_level=risk_level),
        "final_recommendation": _recommendation_text(trend_direction=trend_direction),
        "summary": f"Тренд продаж: {trend_direction}; риск: {risk_level}.",
    }


def _load_sales_target(target: str, *, period_days: int) -> dict[str, Any]:
    report = strategic_db.get_sales_sku_revenue_change(
        product_keyword=target,
        period_days=period_days,
        limit=10,
    )
    row = _matching_row(report.get("rows"), target)
    if row is None:
        return {"target": target, "found": False, "decision": "missing"}
    return {
        "target": target,
        "found": True,
        "current_sales": _number(row.get("current_sales")),
        "previous_sales": _number(row.get("previous_sales")),
        "current_revenue": _number(row.get("current_revenue")),
        "previous_revenue": _number(row.get("previous_revenue")),
        "delta_sales": _number(row.get("delta_sales")),
        "delta_revenue": _number(row.get("delta_revenue")),
        "driver_role": row.get("driver_role"),
    }


def _matching_row(rows: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and target in {str(row.get("nm_id") or ""), str(row.get("vendor_code") or "")}:
            return row
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _trend_direction(*, delta_sales: float, delta_revenue: float) -> str:
    if delta_sales < 0 or delta_revenue < 0:
        return "decline"
    if delta_sales > 0 or delta_revenue > 0:
        return "growth"
    return "stable"


def _risk_level(*, trend_direction: str, delta_sales: float, current_sales: float) -> str:
    if trend_direction == "decline" and current_sales <= 0:
        return "high"
    if trend_direction == "decline" or delta_sales < 0:
        return "medium"
    return "low"


def _action_text(*, trend_direction: str, risk_level: str) -> str:
    if trend_direction == "decline":
        return f"проверить причины просадки, риск {risk_level}"
    if trend_direction == "growth":
        return "удержать рабочие факторы без обещания дополнительного uplift"
    return "оставить под контролем, резких действий не подтверждено"


def _decision_text(*, trend_direction: str, risk_level: str) -> str:
    if trend_direction == "decline":
        return f"Подтвержден sales-риск {risk_level}: продажи/выручка ниже предыдущего периода."
    if trend_direction == "growth":
        return "Sales-тренд положительный; прогноз v2 не считает дополнительный uplift без отдельной гипотезы."
    return "Sales-тренд стабильный; вмешательство не подтверждено по текущему срезу."


def _recommendation_text(*, trend_direction: str) -> str:
    if trend_direction == "decline":
        return "Запустить проверку причин просадки через рекламу, цену, остатки и карточку."
    if trend_direction == "growth":
        return "Удержать текущие условия и не менять рабочие факторы без отдельного guardrail."
    return "Оставить продажи под наблюдением и не создавать резких действий без нового сигнала."


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None
