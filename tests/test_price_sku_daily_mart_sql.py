from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "agent_marts" / "price_sku_daily_mart.sql"


def _sql() -> str:
    return SQL_PATH.read_text(encoding="utf-8").lower()


def test_price_mart_has_one_completed_business_day_sku_grain() -> None:
    sql = _sql()

    assert "v_agent_price_sku_daily" in sql
    assert "select distinct k.report_date, k.nm_id" in sql
    assert "now() at time zone 'europe/moscow'" in sql
    assert "k.report_date < clock.current_business_date" in sql
    assert "'europe/moscow'::text as business_timezone" in sql


def test_price_rows_are_exact_day_and_deduplicated_by_business_key() -> None:
    sql = _sql()

    assert "partition by p.snapshot_date, p.nm_id" in sql
    assert "p.last_synced_at desc nulls last, p.id desc" in sql
    assert "p.report_date = k.report_date" in sql
    assert "as price_snapshot_exact_day" in sql
    assert "price_revisions_dropped" in sql


def test_price_contract_keeps_seller_and_buyer_prices_separate() -> None:
    sql = _sql()

    assert "p.price as base_price_rub" in sql
    assert "p.discount as seller_discount_pct" in sql
    assert "p.discounted_price as seller_effective_price_rub" in sql
    assert "p.club_discounted_price as club_effective_price_rub" in sql
    assert "avg(o.finished_price) as observed_order_buyer_price_avg_rub" in sql
    assert "avg(o.spp) as observed_order_spp_pct_avg" in sql
    assert "as observed_buyer_price_source" in sql


def test_orders_and_sales_use_real_business_keys_and_progress_watermarks() -> None:
    sql = _sql()

    for key in ("srid", "g_number", "sticker"):
        assert f"nullif(o.{key}, '')" in sql
        assert f"nullif(s.{key}, '')" in sql
    assert "sales.wb_supplier_orders_progress" in sql
    assert "sales.wb_supplier_sales_progress" in sql
    assert "as orders_day_complete" in sql
    assert "as sales_day_complete" in sql
    assert "then 'complete_day_absence'" in sql
    assert "orders_missing_business_key_count" in sql
    assert "sales_missing_business_key_count" in sql


def test_missing_event_rows_become_zero_only_after_a_completion_watermark() -> None:
    sql = _sql()

    assert "when k.report_date < (" in sql
    assert "orders_w.orders_change_watermark_at" in sql
    assert "sales_w.sales_change_watermark_at" in sql
    assert "when coalesce(w.campaign_sku_stats_complete, false)" in sql
    assert "then coalesce(a.ad_spend_observed_rub, 0::numeric)" in sql
    assert "as ad_spend_complete" in sql


def test_stock_is_deduplicated_but_presence_is_not_called_complete() -> None:
    sql = _sql()

    assert "s.snapshot_date," in sql
    assert "nullif(s.warehouse_name, '')," in sql
    assert "nullif(s.barcode, '')" in sql
    assert "as stock_snapshot_present" in sql
    assert "false::boolean as stock_snapshot_complete" in sql
    assert "stock_revisions_dropped" in sql


def test_finance_is_source_backed_deduplicated_and_never_uses_future_cogs() -> None:
    sql = _sql()

    assert "partition by r.rrd_id" in sql
    assert "and c.upload_date <= r.rr_dt" in sql
    assert "when f.missing_cost_row_count = 0" in sql
    assert "as contribution_before_ads_rub" in sql
    assert "as contribution_before_ads_per_realized_unit_rub" in sql
    assert "as contribution_after_ads_rub" in sql
    assert "false::boolean as finance_snapshot_complete" in sql
    assert "finance_missing_business_key_count" in sql


def test_pricing_contribution_uses_payout_without_signed_fee_recomposition() -> None:
    sql = _sql()

    assert "j.payout_total_rub as pricing_payout_before_cogs_rub" in sql
    assert (
        "f.payout_total_rub - greatest(f.cogs_total_rub, 0::numeric)" in sql
    )
    assert (
        "'finance.wb_report_detail_daily.ppvz_for_pay'::text" in sql
    )
    assert "as pricing_contribution_basis" in sql
    assert "payout_missing_row_count" in sql
    assert (
        "f.net_revenue_rub - f.marketplace_costs_rub - f.cogs_total_rub"
        not in sql
    )


def test_signed_fees_remain_diagnostics_and_partial_ads_are_not_double_counted() -> None:
    sql = _sql()

    for diagnostic in (
        "commission_signed_raw_rub",
        "acquiring_signed_raw_rub",
        "logistics_signed_raw_rub",
        "storage_signed_raw_rub",
        "penalties_signed_raw_rub",
        "other_charges_signed_raw_rub",
        "marketplace_costs_signed_raw_rub",
    ):
        assert diagnostic in sql
    assert "greatest(f.cogs_total_rub, 0::numeric)" in sql
    assert "j.contribution_before_ads_rub - j.ad_spend_rub" not in sql
    assert "null::numeric as contribution_after_ads_rub" in sql
    assert "null::numeric as contribution_after_ads_per_realized_unit_rub" in sql
    assert "false::boolean as contribution_after_ads_complete" in sql
    assert "false::boolean as partial_ad_spend_subtracted_from_pricing_contribution" in sql


def test_completeness_is_split_between_shadow_and_enforced_usage() -> None:
    sql = _sql()

    assert "as shadow_demand_observation_usable" in sql
    assert "as shadow_profit_observation_usable" in sql
    assert "and j.price_snapshot_complete" in sql
    assert "and j.stock_snapshot_complete" in sql
    assert "and j.finance_snapshot_complete" in sql
    assert "as enforced_input_complete" in sql
    assert "false::boolean as causal_valid" in sql


def test_provenance_is_exposed_and_no_elasticity_is_synthesized() -> None:
    sql = _sql()

    for source in (
        "pricing.wb_supplier_prices",
        "sales.wb_supplier_orders",
        "sales.wb_supplier_sales",
        "supply.wb_supplier_stocks",
        "marketing.wb_marketing_campaign_stats_daily_nms",
        "finance.wb_report_detail_daily",
        "finance.wb_product_coast",
    ):
        assert f"'{source}'::text" in sql
    assert "marketing_source_run_id" in sql
    assert "orders_change_watermark_at" in sql
    assert "sales_change_watermark_at" in sql
    assert "elasticity_to_apply" not in sql
    assert "as elasticity" not in sql
    assert "fallback_default" not in sql
