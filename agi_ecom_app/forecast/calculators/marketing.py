from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.models import ForecastHypothesis
from agi_ecom_app.services import strategic_db

_WEAK_STATUSES = {"inefficient", "negative_outlier"}
_ACTION_REDUCE_STATUSES = {"остановить", "сократить"}


def calculate_marketing_scenario(
    hypothesis: ForecastHypothesis,
    evidence_bundle: ForecastEvidenceBundle,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    period_days = hypothesis.period_days or 7
    target_results = [_load_campaign_target(target, period_days=period_days) for target in hypothesis.targets]

    found_results = [item for item in target_results if item["found"]]
    reduce_results = [item for item in found_results if item["decision"] == "reduce_or_stop"]
    hold_results = [item for item in found_results if item["decision"] == "hold_control"]
    missing_targets = [item["target"] for item in target_results if not item["found"]]

    total_spend = round(sum(_number(item.get("spend")) or 0 for item in found_results), 2)
    spend_saved = round(sum(_number(item.get("spend")) or 0 for item in reduce_results), 2)
    total_orders = round(sum(_number(item.get("orders")) or 0 for item in found_results), 2)
    orders_at_risk = round(sum(_number(item.get("orders")) or 0 for item in reduce_results), 2)
    drr_pct = _load_period_drr(today=today, period_days=period_days)

    blocked_by: list[str] = []
    if not hypothesis.targets:
        blocked_by.append("missing_campaign_targets")
    if missing_targets:
        blocked_by.append("campaign_target_not_found")
    if not found_results:
        blocked_by.append("missing_campaign_spend_or_orders")
    if not reduce_results:
        blocked_by.append("no_confirmed_weak_campaigns")
    if any(item.get("target_type") == "sku_or_non_campaign" for item in target_results):
        blocked_by.append("mixed_or_non_campaign_targets")
    if drr_pct is None:
        blocked_by.append("missing_drr")
    if any(item.get("action_mode") == "test" for item in found_results):
        blocked_by.append("marketing_test_only")
    if any(item.get("action_mode") == "needs_review" for item in found_results):
        blocked_by.append("marketing_low_confidence")

    decision_status = "completed"
    confidence = "high"
    if not reduce_results:
        decision_status = "blocked"
        confidence = "low"
    elif missing_targets or drr_pct is None or any(item.get("action_mode") in {"test", "needs_review"} for item in found_results):
        decision_status = "needs_review"
        confidence = "medium"

    reduce_targets = ", ".join(item["target"] for item in reduce_results) or "нет"
    hold_targets = ", ".join(item["target"] for item in hold_results) or "нет"
    missing_text = ", ".join(missing_targets) or "нет"
    drr_signal = f"{drr_pct:g}%" if drr_pct is not None else "нет точного ДРР на campaign-level"
    primary_marketing_row = next((item for item in reduce_results + hold_results if isinstance(item, dict)), {})

    return {
        "calculator": "marketing",
        "decision_status": decision_status,
        "confidence": confidence,
        "blocked_by": blocked_by,
        "baseline_metrics": {
            "period_days": period_days,
            "campaigns_checked": len(hypothesis.targets),
            "campaigns_found": len(found_results),
            "current_spend_rub": total_spend,
            "current_ad_orders": total_orders,
            "estimated_drr": drr_signal,
        },
        "action_metrics": {
            "campaigns_to_reduce": reduce_targets,
            "campaigns_to_hold": hold_targets,
            "missing_campaigns": missing_text,
            "spend_after_action_rub": round(max(total_spend - spend_saved, 0), 2),
            "orders_at_risk": orders_at_risk,
            "estimated_drr": "должен снизиться при сохранении полезных заказов"
            if spend_saved > 0
            else "нет подтвержденного эффекта",
        },
        "delta_metrics": {
            "ad_spend_saved_rub": spend_saved,
            "orders_at_risk": orders_at_risk,
            "confirmed_reduce_count": len(reduce_results),
            "hold_control_count": len(hold_results),
            "missing_target_count": len(missing_targets),
        },
        "target_results": target_results,
        "marketing_decision_hint": _string(primary_marketing_row.get("decision_hint")),
        "marketing_confidence": _string(primary_marketing_row.get("marketing_confidence")),
        "safe_growth_pct": _number(primary_marketing_row.get("safe_growth_pct")),
        "winner_query_share": _number(primary_marketing_row.get("winner_query_share")),
        "waste_query_share": _number(primary_marketing_row.get("waste_query_share")),
        "marketing_source_status": _string(primary_marketing_row.get("marketing_source_status")),
        "assumptions": [
            f"marketing baseline uses last {period_days} days",
            "stop/reduce effect is estimated from current campaign spend and ad orders",
            "orders_at_risk equals recent ad orders in campaigns confirmed for stop/reduce",
            *evidence_bundle.blocking_gaps[:3],
        ],
    }


def _load_campaign_target(target: str, *, period_days: int) -> dict[str, Any]:
    efficiency = strategic_db.get_marketing_campaign_efficiency_snapshot(
        product_keyword=target,
        days=period_days,
        limit=10,
    )
    action_matrix = strategic_db.get_marketing_action_matrix(
        product_keyword=target,
        days=period_days,
        limit=10,
    )
    campaign_row = _matching_campaign_row(efficiency.get("campaign_rows"), target)
    action_row = _matching_action_row(action_matrix.get("campaign_actions"), target)

    if campaign_row is None and action_row is None:
        target_type = (
            "sku_or_non_campaign"
            if _target_seen_outside_campaign_actions(action_matrix, target)
            else "unknown"
        )
        return {
            "target": target,
            "found": False,
            "target_type": target_type,
            "decision": "missing",
            "reason": "campaign target was not found in marketing campaign facts",
        }

    status = _string(campaign_row.get("status")) if campaign_row else None
    action_status = _string(action_row.get("status")) if action_row else None
    spend = _number(_value(campaign_row, "sum_spend", "spend_rub"))
    orders = _number(_value(campaign_row, "orders", "ad_orders"))
    weak_by_matrix = action_status in _ACTION_REDUCE_STATUSES
    weak_by_efficiency = status in _WEAK_STATUSES
    decision = "reduce_or_stop" if weak_by_matrix or weak_by_efficiency else "hold_control"

    return {
        "target": target,
        "found": True,
        "target_type": "campaign",
        "decision": decision,
        "action_mode": _string(action_row.get("action_mode")) if action_row else None,
        "status": status or action_status,
        "weak_by_matrix": weak_by_matrix,
        "weak_by_efficiency": weak_by_efficiency,
        "spend": spend,
        "orders": orders,
        "cpc": _number(_value(campaign_row, "cpc")),
        "ctr": _number(_value(campaign_row, "ctr")),
        "reason": _string(action_row.get("reason")) if action_row else None,
        "decision_hint": _string(action_row.get("decision_hint")) if action_row else None,
        "marketing_confidence": _string(action_row.get("confidence")) if action_row else None,
        "safe_growth_pct": _number(_value(action_row, "safe_growth_pct")),
        "winner_query_share": _number(_value(action_row, "winner_query_share")),
        "waste_query_share": _number(_value(action_row, "waste_query_share")),
        "marketing_source_status": _string(action_row.get("marketing_source_status")) if action_row else None,
    }


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
    if not isinstance(totals, dict):
        return None
    return _number(totals.get("estimated_drr_promoted_sku_pct"))


def _matching_campaign_row(rows: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("advert_id") or "") == target:
            return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _matching_action_row(rows: Any, target: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and str(row.get("entity_id") or "") == target:
            return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _target_seen_outside_campaign_actions(action_matrix: dict[str, Any], target: str) -> bool:
    for key in ("query_actions", "sku_actions"):
        rows = action_matrix.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("entity_id") or "") == target:
                return True
    return False


def _value(row: dict[str, Any] | None, *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    for key in keys:
        if key in row:
            return row.get(key)
    return None


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
