from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import ForecastHypothesis
from agi_ecom_app.services import strategic_db

_SUPPLIER_INBOUND_KEYS = (
    "supplier_inbound_qty",
    "supplier_inbound_stock",
    "confirmed_supplier_inbound_qty",
)


def calculate_supply_scenario(
    hypothesis: ForecastHypothesis,
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    target_results = [_load_supply_target(target) for target in hypothesis.targets]
    found_results = [item for item in target_results if item["found"]]
    missing_targets = [item["target"] for item in target_results if not item["found"]]

    total_stock = round(sum(_number(item.get("available_stock")) or 0 for item in found_results), 2)
    supplier_inbound_confirmed = bool(found_results) and all(
        item.get("supplier_inbound_confirmed") for item in found_results
    )
    total_inbound = (
        round(sum(_number(item.get("inbound_stock")) or 0 for item in found_results), 2)
        if supplier_inbound_confirmed
        else None
    )
    total_sales_7d = round(sum(_number(item.get("sales_7d_units")) or 0 for item in found_results), 2)
    total_velocity = round(total_sales_7d / 7, 2) if total_sales_7d > 0 else 0
    min_coverage = _min_number(item.get("coverage_days") for item in found_results)
    total_reorder = round(sum(_number(item.get("reorder_qty")) or 0 for item in found_results), 2)
    oos_risk = _aggregate_oos_risk(found_results)

    blocked_by: list[str] = []
    if not hypothesis.targets:
        blocked_by.append("missing_supply_targets")
    if missing_targets:
        blocked_by.append("supply_target_not_found")
    if not found_results:
        blocked_by.append("missing_supply_snapshot")
    if found_results and total_velocity <= 0:
        blocked_by.append("missing_sales_velocity")
    unconfirmed_inbound_targets = [
        item["target"]
        for item in found_results
        if (_number(item.get("reorder_qty")) or 0) > 0
        and not item.get("supplier_inbound_confirmed")
    ]
    if unconfirmed_inbound_targets:
        blocked_by.append("missing_supplier_inbound_snapshot")

    confidence = "high"
    decision_status = "completed"
    if not found_results:
        confidence = "low"
        decision_status = "blocked"
    elif total_velocity <= 0 or missing_targets or unconfirmed_inbound_targets:
        confidence = "medium"
        decision_status = "needs_review"

    priority_targets = ", ".join(item["target"] for item in found_results if item["oos_risk"] in {"high", "medium"}) or "нет"
    action_text = (
        f"дозаказать примерно {total_reorder:g} шт."
        if total_reorder > 0
        else "дозаказ не требуется, оставить контроль покрытия"
    )

    return {
        "calculator": "supply",
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "targets_checked": len(hypothesis.targets),
            "targets_found": len(found_results),
            "available_stock": total_stock,
            "inbound_stock": total_inbound,
            "sales_7d_units": total_sales_7d,
            "sales_velocity": total_velocity,
            "coverage_days": min_coverage,
            "oos_risk": oos_risk,
        },
        "action_metrics": {
            "recommended_action": action_text,
            "priority_targets": priority_targets,
            "reorder_qty": total_reorder,
        },
        "delta_metrics": {
            "reorder_qty": total_reorder,
            "days_until_oos": min_coverage,
            "lost_sales_risk": _lost_sales_risk(oos_risk),
        },
        "target_results": target_results,
        "assumptions": [
            "sales_velocity = sales_7d_units / 7",
            "days_until_oos uses current coverage_days from supply overview",
            "stock_qty is sellable on-hand; stock_qty_full is not used for availability",
            "customer transit is excluded from supplier inbound",
            *(
                ["reorder_qty is a conservative preview with supplier inbound assumed to be zero"]
                if unconfirmed_inbound_targets
                else []
            ),
            *evidence_bundle.blocking_gaps[:3],
        ],
        "decision": _decision_text(
            total_reorder=total_reorder,
            oos_risk=oos_risk,
            priority_targets=priority_targets,
            inbound_confirmed=not unconfirmed_inbound_targets,
        ),
        "final_recommendation": _recommendation_text(
            total_reorder=total_reorder,
            priority_targets=priority_targets,
            inbound_confirmed=not unconfirmed_inbound_targets,
        ),
        "summary": f"Риск OOS: {oos_risk}; дозаказ: {total_reorder:g} шт.",
    }


def _load_supply_target(target: str) -> dict[str, Any]:
    report = strategic_db.get_supply_overview(product_keyword=target, limit=5)
    row = _matching_row(report.get("rows"), target)
    if row is None:
        return {"target": target, "found": False, "decision": "missing"}

    stock = _number(row.get("stock_qty"))
    inbound = _supplier_inbound(row)
    sales_7d = _number(row.get("sales_7d_units"))
    velocity = round(sales_7d / 7, 2) if sales_7d and sales_7d > 0 else 0
    coverage = _number(row.get("days_of_stock"))
    reorder_flag = str(row.get("reorder_flag") or "unknown")
    oos_risk = _oos_risk(reorder_flag=reorder_flag, coverage_days=coverage, sales_velocity=velocity)

    return {
        "target": target,
        "found": True,
        "available_stock": stock,
        "inbound_stock": round(inbound, 2) if inbound is not None else None,
        "supplier_inbound_confirmed": inbound is not None,
        "customer_transit_to_client": _number(row.get("in_way_to_client")),
        "customer_transit_from_client": _number(row.get("in_way_from_client")),
        "sales_7d_units": sales_7d,
        "sales_velocity": velocity,
        "coverage_days": coverage,
        "reorder_flag": reorder_flag,
        "reorder_qty": _number(row.get("reorder_qty_recommended")) or 0,
        "oos_risk": oos_risk,
    }


def _matching_row(rows: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and target in {str(row.get("nm_id") or ""), str(row.get("vendor_code") or "")}:
            return row
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _oos_risk(*, reorder_flag: str, coverage_days: float | None, sales_velocity: float) -> str:
    if sales_velocity <= 0:
        return "unknown"
    if reorder_flag == "critical" or (coverage_days is not None and coverage_days < 7):
        return "high"
    if reorder_flag == "warning" or (coverage_days is not None and coverage_days < 21):
        return "medium"
    return "low"


def _aggregate_oos_risk(items: list[dict[str, Any]]) -> str:
    risks = {str(item.get("oos_risk") or "unknown") for item in items}
    if "high" in risks:
        return "high"
    if "medium" in risks:
        return "medium"
    if "low" in risks:
        return "low"
    return "unknown"


def _lost_sales_risk(oos_risk: str) -> str:
    return {"high": "high", "medium": "medium", "low": "low"}.get(oos_risk, "unknown")


def _decision_text(
    *,
    total_reorder: float,
    oos_risk: str,
    priority_targets: str,
    inbound_confirmed: bool,
) -> str:
    if total_reorder > 0:
        if not inbound_confirmed:
            return (
                f"Есть supply-риск {oos_risk}; по SKU {priority_targets} рассчитан только "
                f"консервативный черновик дозаказа {total_reorder:g} шт. без учёта входящих поставок поставщика."
            )
        return f"Есть supply-риск {oos_risk}; приоритетные SKU для дозаказа: {priority_targets}."
    return f"Supply-риск {oos_risk}; срочный дозаказ по текущему срезу не подтвержден."


def _recommendation_text(
    *,
    total_reorder: float,
    priority_targets: str,
    inbound_confirmed: bool,
) -> str:
    if total_reorder > 0:
        if not inbound_confirmed:
            return (
                f"Сначала сверить входящие поставки поставщика по SKU {priority_targets}; "
                f"до сверки {total_reorder:g} шт. — только консервативный черновик."
            )
        return f"Запланировать дозаказ примерно {total_reorder:g} шт. по SKU: {priority_targets}."
    return "Оставить SKU под контролем покрытия, без срочного дозаказа по текущему срезу."


def _supplier_inbound(row: dict[str, Any]) -> float | None:
    for key in _SUPPLIER_INBOUND_KEYS:
        if key not in row:
            continue
        value = _number(row.get(key))
        if value is not None and value >= 0:
            return value
    return None


def _min_number(values: Any) -> float | None:
    numbers = [_number(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return min(clean) if clean else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None
