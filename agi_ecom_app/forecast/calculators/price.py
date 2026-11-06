from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import ForecastHypothesis
from agi_ecom_app.services import strategic_db


def calculate_price_scenario(
    hypothesis: ForecastHypothesis,
    evidence_bundle: ForecastEvidenceBundle,
) -> dict[str, Any]:
    target_results = [_load_price_target(target) for target in hypothesis.targets]
    found_results = [item for item in target_results if item["found"]]
    missing_targets = [item["target"] for item in target_results if not item["found"]]

    primary = found_results[0] if found_results else {}
    price_status = str(primary.get("price_status") or "unknown")
    competitor_count = _number(primary.get("competitor_count")) or 0
    gap_pct = _number(primary.get("price_gap_pct"))
    gap_rub = _number(primary.get("price_gap_rub"))
    risk = _market_position_risk(price_status=price_status, competitor_count=competitor_count)

    blocked_by: list[str] = []
    if not hypothesis.targets:
        blocked_by.append("missing_price_targets")
    if missing_targets:
        blocked_by.append("price_target_not_found")
    if not found_results:
        blocked_by.append("missing_price_snapshot")
    if price_status == "thin_data" or competitor_count < 5:
        blocked_by.append("thin_competitor_data")

    confidence = "high"
    decision_status = "completed"
    if not found_results:
        confidence = "low"
        decision_status = "blocked"
    elif missing_targets or price_status == "thin_data" or competitor_count < 5:
        confidence = "medium"
        decision_status = "needs_review"

    guardrail = _guardrail(price_status=price_status)
    return {
        "calculator": "price",
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "targets_checked": len(hypothesis.targets),
            "targets_found": len(found_results),
            "current_price": primary.get("current_price"),
            "competitor_median_price": primary.get("competitor_median_price"),
            "price_gap_rub": gap_rub,
            "price_gap_pct": gap_pct,
            "competitor_count": competitor_count,
            "cheaper_competitor_share_pct": primary.get("cheaper_competitor_share_pct"),
            "pricier_competitor_share_pct": primary.get("pricier_competitor_share_pct"),
            "price_status": price_status,
        },
        "action_metrics": {
            "price_guardrail": guardrail,
            "recommended_action": _action_text(price_status=price_status),
            "market_position_risk": risk,
        },
        "delta_metrics": {
            "price_gap_rub": gap_rub,
            "price_gap_pct": gap_pct,
            "market_position_risk": risk,
        },
        "target_results": target_results,
        "assumptions": [
            "price forecast v2 is guardrail-only and does not estimate elasticity",
            "market position uses competitor median and competitor count",
            *evidence_bundle.blocking_gaps[:3],
        ],
        "decision": _decision_text(price_status=price_status, risk=risk),
        "final_recommendation": _recommendation_text(price_status=price_status),
        "summary": f"Ценовая позиция: {price_status}; риск: {risk}.",
    }


def _load_price_target(target: str) -> dict[str, Any]:
    report = strategic_db.get_price_competitor_overview(product_keyword=target, limit=10)
    row = _matching_row(report.get("rows"), target)
    if row is None:
        return {"target": target, "found": False, "decision": "missing"}
    return {
        "target": target,
        "found": True,
        "current_price": _number(row.get("source_price_promo")),
        "competitor_median_price": _number(row.get("competitor_price_median")),
        "price_gap_rub": _number(row.get("source_vs_median_gap_rub")),
        "price_gap_pct": _number(row.get("source_vs_median_gap_pct")),
        "competitor_count": _number(row.get("competitor_count")),
        "cheaper_competitor_share_pct": _number(row.get("cheaper_competitor_share_pct")),
        "pricier_competitor_share_pct": _number(row.get("pricier_competitor_share_pct")),
        "price_status": row.get("price_status") or "unknown",
    }


def _matching_row(rows: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and target in {str(row.get("nm_id") or ""), str(row.get("vendor_code") or "")}:
            return row
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _market_position_risk(*, price_status: str, competitor_count: float) -> str:
    if competitor_count < 5 or price_status == "thin_data":
        return "unknown"
    if price_status == "above_market":
        return "high"
    if price_status == "below_market":
        return "medium"
    return "low"


def _guardrail(*, price_status: str) -> str:
    if price_status == "above_market":
        return "review_reprice_down"
    if price_status == "below_market":
        return "review_reprice_up_or_hold_margin"
    if price_status == "market_aligned":
        return "hold"
    return "block_price_change"


def _action_text(*, price_status: str) -> str:
    if price_status == "above_market":
        return "проверить безопасное снижение цены через маржу"
    if price_status == "below_market":
        return "проверить повышение цены или удержать маржу"
    if price_status == "market_aligned":
        return "удержать цену"
    return "заблокировать изменение цены до расширения competitor data"


def _decision_text(*, price_status: str, risk: str) -> str:
    return f"Price guardrail: статус {price_status}, рыночный риск {risk}; elasticity не рассчитывается в v2."


def _recommendation_text(*, price_status: str) -> str:
    if price_status == "above_market":
        return "Проверить снижение цены только после подтверждения маржи и минимально допустимой цены."
    if price_status == "below_market":
        return "Не снижать цену; проверить возможность повышения или удержания цены с маржинальным guardrail."
    if price_status == "market_aligned":
        return "Оставить цену без изменения по текущему конкурентному срезу."
    return "Не менять цену, пока конкурентный набор тонкий или target не найден."


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None
