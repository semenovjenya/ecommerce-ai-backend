from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Support direct execution: `python scripts/apply_agent_marts.py ...`.
    sys.path.insert(0, str(ROOT))

from agi_ecom_app.manager_decision.candidate_source import _CAMPAIGN_SNAPSHOT_SQL
from agi_ecom_app.services.strategic_db import get_connection


MART_ROOT = ROOT / "sql" / "agent_marts"
SQL_FILES = (
    "01_add_ad_attribution_columns.sql",
    "00_create_agent_mart_schema.sql",
    "sales_mart.sql",
    "supply_mart.sql",
    "marketing_mart.sql",
    "finance_mart.sql",
    "ad_campaign_sku_daily_mart.sql",
    "price_sku_daily_mart.sql",
    "price_context_latest.sql",
)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_BASE_COLUMNS: dict[str, frozenset[str]] = {
    "finance.wb_product_coast": frozenset(
        {"nm_id", "vendor_code", "cost_price_rub", "upload_date", "updated_at"}
    ),
    "finance.wb_report_detail_daily": frozenset(
        {
            "rrd_id",
            "rr_dt",
            "nm_id",
            "sa_name",
            "brand_name",
            "subject_name",
            "doc_type_name",
            "quantity",
            "retail_price_withdisc_rub",
            "ppvz_sales_commission",
            "acquiring_fee",
            "delivery_rub",
            "storage_fee",
            "penalty",
            "deduction",
            "acceptance",
            "rebill_logistic_cost",
            "additional_payment",
            "ppvz_for_pay",
            "supplier_oper_name",
            "last_synced_at",
        }
    ),
    "marketing.wb_marketing_budget": frozenset(
        {"id", "advert_id", "snapshot_date", "total", "last_synced_at"}
    ),
    "marketing.wb_marketing_campaign_nms": frozenset(
        {
            "id",
            "advert_id",
            "nm_id",
            "snapshot_date",
            "subject_name",
            "bid_search_kopecks",
            "bid_recommendations_kopecks",
            "last_synced_at",
        }
    ),
    "marketing.wb_marketing_campaign_stats_daily": frozenset(
        {
            "id",
            "stat_date",
            "advert_id",
            "views",
            "clicks",
            "orders",
            "shks",
            "sum_spend",
            "last_synced_at",
        }
    ),
    "marketing.wb_marketing_campaign_stats_daily_nms": frozenset(
        {
            "id",
            "stat_date",
            "advert_id",
            "nm_id",
            "nm_name",
            "views",
            "clicks",
            "orders",
            "shks",
            "sum_spend",
            "last_synced_at",
        }
    ),
    "marketing.wb_marketing_campaigns": frozenset(
        {
            "id",
            "external_id",
            "snapshot_date",
            "name",
            "status",
            "payment_type",
            "brand",
            "last_synced_at",
        }
    ),
    "marketing.wb_search_query_details": frozenset(
        {
            "id",
            "current_period_end",
            "nm_id",
            "vendor_code",
            "brand_name",
            "subject_name",
            "visibility_current",
            "visibility_dynamics",
            "avg_position_current",
            "avg_position_dynamics",
            "open_card_current",
            "add_to_cart_current",
            "orders_current",
            "snapshot_hour",
            "last_synced_at",
        }
    ),
    "marketing.wb_search_query_orders_text_daily": frozenset(
        {"dt", "nm_id", "text", "frequency"}
    ),
    "sales.wb_sales_funnel_products_daily": frozenset(
        {
            "report_date",
            "nm_id",
            "vendor_code",
            "brand_name",
            "subject_name",
            "open_count",
            "cart_count",
            "order_count",
            "order_sum",
            "buyout_count",
            "avg_price",
        }
    ),
    "sales.wb_supplier_orders": frozenset(
        {
            "id",
            "date_ord",
            "last_change_date",
            "nm_id",
            "supplier_article",
            "brand",
            "subject",
            "is_cancel",
            "price_with_disc",
            "finished_price",
            "spp",
            "srid",
            "g_number",
            "sticker",
            "last_synced_at",
        }
    ),
    "sales.wb_supplier_orders_progress": frozenset(
        {"last_change_date_from", "updated_at"}
    ),
    "sales.wb_supplier_sales": frozenset(
        {
            "id",
            "date_ord",
            "last_change_date",
            "nm_id",
            "supplier_article",
            "brand",
            "subject",
            "is_realization",
            "for_pay",
            "price_with_disc",
            "finished_price",
            "spp",
            "srid",
            "g_number",
            "sticker",
            "last_synced_at",
        }
    ),
    "sales.wb_supplier_sales_progress": frozenset(
        {"last_change_date_from", "updated_at"}
    ),
    "pricing.wb_supplier_prices": frozenset(
        {
            "id",
            "snapshot_date",
            "nm_id",
            "vendor_code",
            "currency_iso_code",
            "discount",
            "club_discount",
            "editable_size_price",
            "is_bad_turnover",
            "price",
            "discounted_price",
            "club_discounted_price",
            "last_synced_at",
        }
    ),
    "analytics_ai.wb_price_ai_analysis": frozenset(
        {
            "id",
            "run_date",
            "source_snapshot_date",
            "source_nm_id",
            "source_vendor_code",
            "source_brand",
            "source_subject_name",
            "source_title",
            "source_product_url",
            "source_price_site_display",
            "source_price_site_wallet",
            "source_price_site_effective",
            "source_platform_discount_percent",
            "source_wallet_discount_percent",
            "source_report_to_site_discount_percent",
            "competitor_selection_mode",
            "competitor_count",
            "competitor_price_min",
            "competitor_price_median",
            "competitor_price_max",
            "strict_competitor_count",
            "strict_competitor_price_min",
            "strict_competitor_price_median",
            "strict_competitor_price_max",
            "market_candidate_count",
            "market_price_min",
            "market_price_median",
            "market_price_max",
            "llm_checked_count",
            "llm_kept_count",
            "vision_checked_count",
            "vision_absolute_count",
            "vision_close_count",
            "vision_similar_count",
            "vision_not_analog_count",
            "used_market_fallback",
            "is_price_outlier",
            "analysis_status",
            "last_synced_at",
        }
    ),
    "analytics_ai.mart_price_context_daily": frozenset(
        {
            "snapshot_date",
            "source_nm_id",
            "source_price_site_display",
            "source_price_site_wallet",
            "source_price_site_effective",
            "source_platform_discount_percent",
            "source_wallet_discount_percent",
            "source_report_to_site_discount_percent",
            "query_count",
            "source_found_query_count",
            "source_found_query_share",
            "weighted_organic_position",
            "market_price_min",
            "market_price_median",
            "market_price_max",
            "organic_competitors_before_source",
            "price_before_median",
            "source_to_before_median_price_ratio",
        }
    ),
    "supply.wb_supplier_stocks": frozenset(
        {
            "id",
            "snapshot_date",
            "nm_id",
            "supplier_article",
            "brand",
            "subject",
            "barcode",
            "warehouse_name",
            "quantity",
            "quantity_full",
            "in_way_to_client",
            "in_way_from_client",
            "last_change_date",
            "last_synced_at",
        }
    ),
    "supply.wb_supplies": frozenset(
        {"snapshot_date", "supply_id", "supply_date", "last_synced_at"}
    ),
    "supply.wb_supplies_goods": frozenset(
        {
            "snapshot_date",
            "supply_id",
            "nm_id",
            "quantity",
            "ready_for_sale_quantity",
            "unloading_quantity",
            "last_synced_at",
        }
    ),
}


def _sql(filename: str, *, schema: str) -> str:
    return (MART_ROOT / filename).read_text(encoding="utf-8").replace(
        "{{target_schema}}",
        schema,
    )


def _preflight_base_contract(cursor: Any) -> None:
    required_tables = sorted(REQUIRED_BASE_COLUMNS)
    cursor.execute(
        """
        select
            required_name,
            to_regclass(required_name)::text as resolved_name
        from unnest(%s::text[]) as required(required_name)
        """,
        [required_tables],
    )
    table_rows = cursor.fetchall()
    missing_tables = sorted(
        row["required_name"]
        for row in table_rows
        if row["resolved_name"] is None
    )

    required_schemas = sorted({name.split(".", 1)[0] for name in required_tables})
    cursor.execute(
        """
        select table_schema, table_name, column_name
        from information_schema.columns
        where table_schema = any(%s::text[])
        """,
        [required_schemas],
    )
    available_columns = {
        (row["table_schema"], row["table_name"], row["column_name"])
        for row in cursor.fetchall()
    }
    missing_columns = sorted(
        f"{qualified_name}.{column_name}"
        for qualified_name, columns in REQUIRED_BASE_COLUMNS.items()
        for column_name in columns
        if (*qualified_name.split(".", 1), column_name) not in available_columns
    )

    if missing_tables or missing_columns:
        details: list[str] = []
        if missing_tables:
            details.append("tables=" + ",".join(missing_tables))
        if missing_columns:
            details.append("columns=" + ",".join(missing_columns))
        raise RuntimeError(
            "analytics base schema preflight failed: " + "; ".join(details)
        )


def apply_agent_marts(*, schema: str, commit: bool) -> dict[str, object]:
    if not IDENTIFIER.fullmatch(schema):
        raise ValueError("schema must be a simple PostgreSQL identifier")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            # Multiple worker replicas may start together. Serialize the
            # transactional DDL without introducing a separate lock table.
            cursor.execute(
                "select pg_advisory_xact_lock(hashtext(%s)::bigint)",
                [f"agent-marts-migration:{schema}"],
            )
            _preflight_base_contract(cursor)
            for filename in SQL_FILES:
                cursor.execute(_sql(filename, schema=schema))

            # Compile the exact query used by ManagerDecision after the schema
            # migration. This catches drift between the mart contract and the
            # runtime source before a deployment is committed.
            cursor.execute("explain " + _CAMPAIGN_SNAPSHOT_SQL, [1])

            cursor.execute(
                f"""
                select
                    count(*)::bigint as row_count,
                    count(distinct (report_date, advert_id, nm_id))::bigint
                        as distinct_key_count,
                    count(*) filter (where not stats_row_present)::bigint
                        as date_spine_row_count,
                    count(*) filter (where decision_input_complete)::bigint
                        as decision_input_complete_count
                from {schema}.v_agent_ad_campaign_sku_daily
                """
            )
            validation = dict(cursor.fetchone())
            if validation["row_count"] != validation["distinct_key_count"]:
                raise RuntimeError(
                    "v_agent_ad_campaign_sku_daily violates its "
                    "(report_date, advert_id, nm_id) grain"
                )

            cursor.execute(
                f"""
                select
                    count(*)::bigint as price_row_count,
                    count(distinct (report_date, nm_id))::bigint
                        as price_distinct_key_count,
                    count(*) filter (
                        where shadow_demand_observation_usable
                    )::bigint as price_shadow_demand_count,
                    count(*) filter (
                        where shadow_profit_observation_usable
                    )::bigint as price_shadow_profit_count,
                    count(*) filter (
                        where enforced_input_complete
                    )::bigint as price_enforced_input_count
                from {schema}.v_agent_price_sku_daily
                """
            )
            price_validation = dict(cursor.fetchone())
            if (
                price_validation["price_row_count"]
                != price_validation["price_distinct_key_count"]
            ):
                raise RuntimeError(
                    "v_agent_price_sku_daily violates its "
                    "(report_date, nm_id) grain"
                )

        if commit:
            connection.commit()
        else:
            connection.rollback()

    return {
        "schema": schema,
        "mode": "apply" if commit else "check_rollback",
        **validation,
        **price_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or deploy the Agent PostgreSQL marts atomically."
    )
    parser.add_argument("--schema", default="agent_marts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Execute every migration/view and runtime EXPLAIN, then roll back.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Execute every migration/view and commit atomically.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            apply_agent_marts(schema=args.schema, commit=bool(args.apply)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
