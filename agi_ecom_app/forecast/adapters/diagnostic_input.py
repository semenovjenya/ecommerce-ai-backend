from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agi_ecom_app.forecast.models import ForecastPlan, ResolvedTarget
from agi_ecom_app.forecast.router import select_workers
from agi_ecom_app.services import strategic_db


class ForecastEvidenceBundle(BaseModel):
    diagnostic: dict[str, Any] = Field(default_factory=dict)
    workers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    per_target: dict[str, dict[str, Any]] = Field(default_factory=dict)
    blocking_gaps: list[str] = Field(default_factory=list)
    confirmed_facts: list[dict[str, Any]] = Field(default_factory=list)


def build_forecast_evidence_bundle(
    diagnostic_package: dict[str, Any] | None,
    *,
    plan: ForecastPlan | None = None,
    resolved_targets: list[ResolvedTarget] | None = None,
) -> ForecastEvidenceBundle:
    diagnostic = _normalize_diagnostic_package(diagnostic_package)
    bundle = ForecastEvidenceBundle(
        diagnostic=diagnostic,
        blocking_gaps=list(diagnostic.get("blocking_gaps") or []),
        confirmed_facts=list(diagnostic.get("confirmed_facts") or []),
    )
    if plan is None or not resolved_targets:
        return bundle

    worker_names = _select_workers(plan)
    workers: dict[str, dict[str, Any]] = {}
    per_target: dict[str, dict[str, Any]] = _opportunity_per_target(diagnostic, resolved_targets)
    for worker_name in worker_names:
        worker_output = _run_worker(worker_name, resolved_targets, plan)
        workers[worker_name] = worker_output
        for match in worker_output.get("entity_matches") or []:
            if not isinstance(match, dict):
                continue
            target_id = str(match.get("target_id") or "").strip()
            if not target_id:
                continue
            per_target.setdefault(target_id, {}).update(match.get("facts") or {})
            per_target[target_id].setdefault("entity_matches", []).append(match)
        for gap in worker_output.get("data_gaps") or []:
            if gap not in bundle.blocking_gaps:
                bundle.blocking_gaps.append(gap)
    bundle.workers = workers
    bundle.per_target = per_target
    return bundle


def _select_workers(plan: ForecastPlan) -> list[str]:
    return select_workers(plan)


def _run_worker(
    worker_name: str,
    resolved_targets: list[ResolvedTarget],
    plan: ForecastPlan,
) -> dict[str, Any]:
    if worker_name == "forecast_marketing_agent":
        return _marketing_worker(resolved_targets, plan)
    if worker_name == "forecast_supply_agent":
        return _supply_worker(resolved_targets)
    if worker_name == "forecast_sales_agent":
        return _sales_worker(resolved_targets, plan.period_days)
    if worker_name == "forecast_price_agent":
        return _price_worker(resolved_targets)
    if worker_name == "forecast_finance_agent":
        return _finance_worker(resolved_targets)
    if worker_name == "forecast_catalog_agent":
        return _catalog_worker(resolved_targets)
    return {
        "worker": worker_name,
        "facts": {},
        "data_gaps": ["unknown_worker"],
        "confidence": "low",
        "recommended_interpretation": "Источник фактов для этого worker не реализован.",
        "entity_matches": [],
    }


def _marketing_worker(resolved_targets: list[ResolvedTarget], plan: ForecastPlan) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    matrix = strategic_db.get_marketing_action_matrix(
        product_keyword=None,
        days=plan.period_days,
        limit=max(10, len(resolved_targets) * 3),
    )
    campaign_mart = strategic_db.get_marketing_campaign_daily_mart(
        product_keyword=None,
        days=plan.period_days,
        limit=max(10, len(resolved_targets) * 3),
    )
    forecast_ready = strategic_db.get_marketing_forecast_ready_mart(
        product_keyword=None,
        limit=max(10, len(resolved_targets) * 3),
    )
    query_bands = strategic_db.get_marketing_query_bands_mart(
        product_keyword=None,
        limit=max(10, len(resolved_targets) * 5),
    )
    visibility = strategic_db.get_marketing_visibility_snapshot(product_keyword=None, limit=max(10, len(resolved_targets) * 3))
    demand = strategic_db.get_marketing_search_demand_snapshot(
        product_keyword=None,
        days=plan.period_days,
        limit=max(10, len(resolved_targets) * 5),
    )

    for target in resolved_targets:
        if target.entity_type == "campaign":
            campaign_rows = campaign_mart.get("rows") if isinstance(campaign_mart, dict) else []
            campaign_row = _find_row(campaign_rows, target.id, "advert_id")
            forecast_rows = forecast_ready.get("rows") if isinstance(forecast_ready, dict) else []
            forecast_row = _find_row(forecast_rows, target.id, "advert_id")
            matrix_rows = matrix.get("campaign_actions") if isinstance(matrix, dict) else []
            action_row = _find_row(matrix_rows, target.id, "entity_id")
            marketing_source_status = (
                _string((action_row or {}).get("marketing_source_status"))
                or _string((matrix or {}).get("marketing_source_status"))
                or _string((campaign_mart or {}).get("source_status"))
            )
            demand_source_status = (
                _string((action_row or {}).get("demand_source_status"))
                or _string((matrix or {}).get("demand_source_status"))
            )
            facts = {
                "campaign_row": campaign_row,
                "forecast_row": forecast_row,
                "campaign_action": action_row,
                "marketing_source_status": marketing_source_status,
                "demand_source_status": demand_source_status,
                "decision_hint": (action_row or {}).get("decision_hint") or (forecast_row or {}).get("decision_hint"),
                "safe_growth_pct": (action_row or {}).get("safe_growth_pct") or (forecast_row or {}).get("safe_growth_pct"),
                "marginal_quality_coef": (action_row or {}).get("marginal_quality_coef") or (forecast_row or {}).get("marginal_quality_coef"),
                "winner_query_share": (action_row or {}).get("winner_query_share") or (forecast_row or {}).get("winner_query_share_7d"),
                "waste_query_share": (action_row or {}).get("waste_query_share") or (forecast_row or {}).get("waste_query_share_7d"),
                "marketing_confidence": (action_row or {}).get("confidence") or (forecast_row or {}).get("confidence"),
            }
            if not campaign_row and not action_row:
                data_gaps.append(f"marketing_campaign_not_found:{target.id}")
            if marketing_source_status in {"unpopulated", "missing"}:
                data_gaps.append(f"marketing_mart_unpopulated:{target.id}")
            action_mode = _string((action_row or {}).get("action_mode"))
            if demand_source_status == "unpopulated" and action_mode in {"scale_review", "needs_review", "test"}:
                data_gaps.append(f"marketing_demand_mart_unpopulated:{target.id}")
            if _string((action_row or {}).get("confidence")) == "low" and action_mode in {"scale_review", "needs_review", "test"}:
                data_gaps.append(f"marketing_low_confidence:{target.id}")
            if action_mode == "test":
                data_gaps.append(f"marketing_test_only:{target.id}")
            entity_matches.append(
                {
                    "target_id": target.id,
                    "entity_type": target.entity_type,
                    "facts": facts,
                }
            )
            continue

        target_query_bands = strategic_db.get_marketing_query_bands_mart(product_keyword=target.id, limit=20)
        target_visibility = strategic_db.get_marketing_visibility_snapshot(product_keyword=target.id, limit=10)
        target_demand = strategic_db.get_marketing_search_demand_snapshot(
            product_keyword=target.id,
            days=plan.period_days,
            limit=10,
        )
        query_rows = query_bands.get("rows") if isinstance(query_bands, dict) else []
        query_row = _find_row(query_rows, target.id, "nm_id", "vendor_code")
        query_action_rows = matrix.get("query_actions") if isinstance(matrix, dict) else []
        query_action = _find_row(query_action_rows, target.id, "nm_id", "vendor_code", "entity_id")
        visibility_row = _find_row(visibility.get("visibility_rows"), target.id, "nm_id", "vendor_code")
        if not query_row:
            query_row = _find_row(target_query_bands.get("rows"), target.id, "nm_id", "vendor_code")
        if not visibility_row:
            visibility_row = (
                _find_row(target_visibility.get("visibility_rows"), target.id, "nm_id", "vendor_code")
                or _find_row(target_visibility.get("position_loss_rows"), target.id, "nm_id", "vendor_code")
                or _find_row(target_visibility.get("open_card_gap_rows"), target.id, "nm_id", "vendor_code")
            )
        top_query = (
            _find_row(target_demand.get("order_driving_query_rows"), target.id, "nm_id", "vendor_code")
            or _find_row(target_demand.get("high_frequency_low_order_rows"), target.id, "nm_id", "vendor_code")
            or _first_row(target_demand.get("order_driving_query_rows") or target_demand.get("high_frequency_low_order_rows"))
            or _first_row(demand.get("order_driving_query_rows") or demand.get("high_frequency_low_order_rows"))
        )
        marketing_source_status = (
            _string((query_action or {}).get("marketing_source_status"))
            or _string((matrix or {}).get("marketing_source_status"))
            or _string((query_bands or {}).get("source_status"))
        )
        demand_source_status = (
            _string((query_action or {}).get("demand_source_status"))
            or _string((matrix or {}).get("demand_source_status"))
        )
        facts = {
            "query_row": query_row,
            "query_action": query_action,
            "query_rows": _list_rows(target_query_bands.get("rows"))[:10],
            "visibility_row": visibility_row,
            "demand_row": top_query,
            "position_loss_count": len(target_visibility.get("position_loss_rows") or visibility.get("position_loss_rows") or []),
            "open_card_gap_count": len(target_visibility.get("open_card_gap_rows") or visibility.get("open_card_gap_rows") or []),
            "high_frequency_low_order_count": len(target_demand.get("high_frequency_low_order_rows") or demand.get("high_frequency_low_order_rows") or []),
            "marketing_source_status": marketing_source_status,
            "demand_source_status": demand_source_status,
            "query_band": (query_action or {}).get("query_band") or (query_row or {}).get("query_band"),
            "decision_hint": (query_action or {}).get("decision_hint"),
            "safe_growth_pct": (query_action or {}).get("safe_growth_pct"),
            "marginal_quality_coef": (query_action or {}).get("marginal_quality_coef"),
            "winner_query_share": (query_action or {}).get("winner_query_share"),
            "waste_query_share": (query_action or {}).get("waste_query_share"),
            "marketing_confidence": (query_action or {}).get("confidence"),
        }
        if not visibility_row and not top_query and not query_row and not query_action:
            data_gaps.append(f"marketing_visibility_missing:{target.id}")
        if marketing_source_status in {"unpopulated", "missing"}:
            data_gaps.append(f"marketing_mart_unpopulated:{target.id}")
        query_action_mode = _string((query_action or {}).get("action_mode"))
        if demand_source_status == "unpopulated" and query_action_mode in {"scale_review", "needs_review", "test"}:
            data_gaps.append(f"marketing_demand_mart_unpopulated:{target.id}")
        if _string((query_action or {}).get("confidence")) == "low" and query_action_mode in {"scale_review", "needs_review", "test"}:
            data_gaps.append(f"marketing_low_confidence:{target.id}")
        if query_action_mode == "test":
            data_gaps.append(f"marketing_test_only:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": facts,
            }
        )

    return {
        "worker": "forecast_marketing_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Маркетинговый worker собрал факты из marketing marts и безопасно добрал fallback demand/visibility при необходимости.",
        "entity_matches": entity_matches,
    }


def _supply_worker(resolved_targets: list[ResolvedTarget]) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    for target in resolved_targets:
        report = strategic_db.get_supply_overview(product_keyword=target.id, limit=5)
        row = _find_row(report.get("rows"), target.id, "nm_id", "vendor_code")
        if not row:
            data_gaps.append(f"supply_not_found:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": {"supply_row": row},
            }
        )
    return {
        "worker": "forecast_supply_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Supply worker собрал факты по остаткам, поставкам и покрытию.",
        "entity_matches": entity_matches,
    }


def _sales_worker(resolved_targets: list[ResolvedTarget], period_days: int) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    for target in resolved_targets:
        trend = strategic_db.get_sales_sku_revenue_change(
            product_keyword=target.id,
            period_days=period_days,
            limit=10,
        )
        overview = strategic_db.get_sales_overview(limit=10)
        trend_row = _find_row(trend.get("rows"), target.id, "nm_id", "vendor_code")
        watch_row = _find_row(overview.get("conversion_watchlist"), target.id, "nm_id", "vendor_code")
        revenue_row = _find_row(overview.get("top_revenue_rows"), target.id, "nm_id", "vendor_code")
        if not trend_row and not watch_row and not revenue_row:
            data_gaps.append(f"sales_not_found:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": {
                    "sales_trend_row": trend_row,
                    "sales_watch_row": watch_row,
                    "sales_revenue_row": revenue_row,
                },
            }
        )
    return {
        "worker": "forecast_sales_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Sales worker собрал факты по тренду продаж, выручке и конверсии.",
        "entity_matches": entity_matches,
    }


def _price_worker(resolved_targets: list[ResolvedTarget]) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    for target in resolved_targets:
        decision_report = strategic_db.get_price_decision_overview(product_keyword=target.id, limit=10)
        row = _find_row(decision_report.get("rows"), target.id, "nm_id", "vendor_code")
        if not row:
            fallback_report = strategic_db.get_price_competitor_overview(product_keyword=target.id, limit=10)
            row = _find_row(fallback_report.get("rows"), target.id, "nm_id", "vendor_code")
        if not row:
            data_gaps.append(f"price_not_found:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": {"price_row": row},
            }
        )
    return {
        "worker": "forecast_price_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Price worker собрал факты по конкурентному коридору и позиции цены.",
        "entity_matches": entity_matches,
    }


def _finance_worker(resolved_targets: list[ResolvedTarget]) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    for target in resolved_targets:
        overview = strategic_db.get_finance_overview(product_keyword=target.id, limit=10)
        rows = [
            *_list_rows(overview.get("unit_economics_watchlist")),
            *_list_rows(overview.get("margin_watchlist")),
            *_list_rows(overview.get("fee_pressure_rows")),
        ]
        row = _find_row(rows, target.id, "nm_id", "vendor_code")
        if not row:
            data_gaps.append(f"finance_not_found:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": {"finance_row": row},
            }
        )
    return {
        "worker": "forecast_finance_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Finance worker собрал факты по марже, выплатам и расходам маркетплейса.",
        "entity_matches": entity_matches,
    }


def _catalog_worker(resolved_targets: list[ResolvedTarget]) -> dict[str, Any]:
    entity_matches: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    for target in resolved_targets:
        overview = strategic_db.get_sales_overview(limit=10)
        visibility = strategic_db.get_marketing_visibility_snapshot(product_keyword=target.id, limit=10)
        watch_row = _find_row(overview.get("conversion_watchlist"), target.id, "nm_id", "vendor_code")
        visibility_row = _find_row(visibility.get("visibility_rows"), target.id, "nm_id", "vendor_code")
        if not watch_row and not visibility_row:
            data_gaps.append(f"catalog_not_found:{target.id}")
        entity_matches.append(
            {
                "target_id": target.id,
                "entity_type": target.entity_type,
                "facts": {
                    "conversion_row": watch_row,
                    "visibility_row": visibility_row,
                },
            }
        )
    return {
        "worker": "forecast_catalog_agent",
        "facts": {"matched_targets": len(entity_matches)},
        "data_gaps": data_gaps,
        "confidence": _confidence_from_gaps(entity_matches, data_gaps),
        "recommended_interpretation": "Catalog worker собрал факты по карточке, видимости и воронке.",
        "entity_matches": entity_matches,
    }


def _normalize_diagnostic_package(diagnostic_package: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnostic_package, dict):
        return {
            "primary_summary": None,
            "blocking_gaps": [],
            "confirmed_facts": [],
        }
    primary = diagnostic_package.get("primary_evidence")
    if not isinstance(primary, dict):
        primary = diagnostic_package
    return {
        "primary_agent": _string(primary.get("agent_name") or diagnostic_package.get("primary_agent")),
        "primary_summary": _string(primary.get("summary") or diagnostic_package.get("summary")),
        "blocking_gaps": [str(item) for item in diagnostic_package.get("blocking_gaps") or [] if item],
        "confirmed_facts": [
            item for item in diagnostic_package.get("confirmed_facts") or [] if isinstance(item, dict)
        ],
        "supporting_evidence": [
            item for item in diagnostic_package.get("supporting_evidence") or [] if isinstance(item, dict)
        ],
        "opportunity": diagnostic_package.get("opportunity") if isinstance(diagnostic_package.get("opportunity"), dict) else None,
    }


def _opportunity_per_target(
    diagnostic: dict[str, Any],
    resolved_targets: list[ResolvedTarget],
) -> dict[str, dict[str, Any]]:
    opportunity = diagnostic.get("opportunity")
    if not isinstance(opportunity, dict):
        return {}
    source_facts = opportunity.get("source_facts") if isinstance(opportunity.get("source_facts"), dict) else {}
    facts = {
        "opportunity": opportunity,
        "opportunity_source_facts": source_facts,
    }
    per_target: dict[str, dict[str, Any]] = {}
    for target in resolved_targets:
        per_target[target.id] = dict(facts)
    return per_target


def _find_row(rows: Any, target: str, *keys: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_values = {str(row.get(key) or "") for key in keys}
        if target in row_values:
            return row
    return None


def _first_row(rows: Any) -> dict[str, Any] | None:
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None


def _list_rows(rows: Any) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _confidence_from_gaps(entity_matches: list[dict[str, Any]], data_gaps: list[str]) -> str:
    if not entity_matches:
        return "low"
    if data_gaps:
        return "medium"
    return "high"


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
