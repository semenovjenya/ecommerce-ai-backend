from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from scripts.apply_agent_marts import (
    REQUIRED_BASE_COLUMNS,
    _preflight_base_contract,
)


ROOT = Path(__file__).resolve().parents[1]
MART_ROOT = ROOT / "sql" / "agent_marts"


class PriceContextLatestMartTests(unittest.TestCase):
    def setUp(self):
        self.sql = (MART_ROOT / "price_context_latest.sql").read_text(
            encoding="utf-8"
        ).lower()

    def test_exposes_one_latest_row_per_sku_without_rewriting_sources(self):
        self.assertIn(
            "create or replace view analytics_ai.v_agent_price_context_latest",
            self.sql,
        )
        self.assertIn("select distinct on (p.nm_id)", self.sql)
        self.assertIn("select distinct on (a.source_nm_id)", self.sql)
        self.assertIn("seller_controlled_price", self.sql)
        self.assertIn("customer_price_effective", self.sql)

    def test_fails_closed_for_weak_competitor_sets(self):
        self.assertIn("strict_competitor_count, 0) < 5", self.sql)
        self.assertIn("competitor_quality <> 'trusted'", self.sql)
        self.assertIn("insufficient_competitor_quality", self.sql)

    def test_keeps_broad_market_and_strict_analogs_separate(self):
        self.assertIn("broad_market_price_median", self.sql)
        self.assertIn("strict_competitor_price_median", self.sql)


class AdAttributionMigrationTests(unittest.TestCase):
    def setUp(self):
        self.sql = (MART_ROOT / "01_add_ad_attribution_columns.sql").read_text(
            encoding="utf-8"
        ).lower()

    def test_adds_nullable_raw_sum_price_to_both_supported_grains(self):
        self.assertIn(
            "alter table if exists marketing.wb_marketing_campaign_stats_daily\n"
            "    add column if not exists sum_price numeric",
            self.sql,
        )
        self.assertIn(
            "alter table if exists marketing.wb_marketing_campaign_stats_daily_nms\n"
            "    add column if not exists sum_price numeric",
            self.sql,
        )
        self.assertNotIn("sum_price numeric not null", self.sql)
        self.assertNotIn("default 0", self.sql)

    def test_migration_is_idempotent(self):
        self.assertEqual(self.sql.count("alter table if exists"), 2)
        self.assertEqual(self.sql.count("add column if not exists sum_price numeric"), 2)
        self.assertIn(
            "create table if not exists "
            "marketing.wb_marketing_campaign_stats_daily_watermarks",
            self.sql,
        )

    def test_watermark_is_fail_closed_by_default(self):
        self.assertIn(
            "campaign_stats_complete boolean not null default false",
            self.sql,
        )
        self.assertIn(
            "campaign_sku_stats_complete boolean not null default false",
            self.sql,
        )

    def test_complete_watermark_requires_provenance(self):
        self.assertIn(
            "wb_marketing_campaign_stats_daily_watermarks_proven_complete",
            self.sql,
        )
        self.assertIn(
            "not (campaign_stats_complete or campaign_sku_stats_complete)",
            self.sql,
        )
        self.assertIn("source_run_id is not null", self.sql)
        self.assertIn("source_completed_at is not null", self.sql)


class AdCampaignSkuDailyMartTests(unittest.TestCase):
    def setUp(self):
        self.sql = (MART_ROOT / "ad_campaign_sku_daily_mart.sql").read_text(
            encoding="utf-8"
        ).lower()

    def test_uses_campaign_sku_day_as_the_output_grain(self):
        self.assertIn("date_spine as (", self.sql)
        self.assertIn("from normalized_ad_stats s", self.sql)
        self.assertIn("s.report_date", self.sql)
        self.assertIn("s.advert_id", self.sql)
        self.assertIn("s.nm_id", self.sql)

    def test_deduplicates_raw_stats_before_joining_the_public_grain(self):
        self.assertIn(
            "select distinct on (raw.stat_date, raw.advert_id, raw.nm_id)",
            self.sql,
        )
        self.assertIn(
            "select distinct on (raw.stat_date, raw.advert_id)",
            self.sql,
        )
        self.assertIn("raw.last_synced_at desc nulls last", self.sql)
        self.assertIn("raw.id desc", self.sql)
        self.assertIn("left join latest_raw_sku_stats raw", self.sql)
        self.assertIn("left join latest_raw_campaign_stats raw", self.sql)

    def test_date_spine_keeps_post_pause_days_without_assuming_zero(self):
        self.assertIn("select f.report_date, f.nm_id", self.sql)
        self.assertIn("select s.snapshot_date, s.nm_id", self.sql)
        self.assertIn("membership.last_observed_date + 35", self.sql)
        self.assertIn(
            "marketing.wb_marketing_campaign_stats_daily_watermarks watermark",
            self.sql,
        )
        self.assertIn("when watermark.campaign_sku_stats_complete then 0::numeric", self.sql)
        self.assertIn("as stats_day_watermark_complete", self.sql)

    def test_preserves_missing_attributed_revenue(self):
        self.assertIn("s.sum_price as attributed_order_revenue_rub", self.sql)
        self.assertIn(
            "when raw.id is not null then raw.sum_price is not null",
            self.sql,
        )
        self.assertNotIn(
            "coalesce(raw.sum_price, 0)",
            self.sql,
        )

    def test_exposes_reconciliation_and_provenance(self):
        self.assertIn("campaign_sku_attribution_reconciled", self.sql)
        self.assertIn("campaign_sku_attributed_revenue_delta_rub", self.sql)
        self.assertIn("when s.stats_row_present then 'wb_adv_fullstats.sum_price'", self.sql)
        self.assertIn("'derived:complete_day_absence(wb_adv_fullstats)'", self.sql)
        self.assertIn("as stats_evidence_origin", self.sql)
        self.assertIn("s.stats_source_run_id", self.sql)
        self.assertIn("false as causal_valid", self.sql)

    def test_exposes_unambiguous_sku_contribution_after_all_ad_spend(self):
        self.assertIn("count(distinct s.advert_id)::integer as sku_campaign_count", self.sql)
        self.assertIn("sum(s.sum_spend) as sku_ad_spend_rub", self.sql)
        self.assertIn(
            "finance.sku_contribution_before_ads_rub - sku_ads.sku_ad_spend_rub",
            self.sql,
        )
        self.assertIn("as sku_contribution_after_ads_complete", self.sql)

    def test_requires_exact_date_matches_for_stock_supply_and_finance(self):
        self.assertIn("stock.snapshot_date = s.report_date", self.sql)
        self.assertIn("supply_day.snapshot_date = s.report_date", self.sql)
        self.assertIn("finance.report_date = s.report_date", self.sql)
        self.assertNotIn("current_date - interval", self.sql)

    def test_decision_completeness_is_fail_closed(self):
        self.assertIn("s.attributed_revenue_complete", self.sql)
        self.assertIn("s.ad_metrics_complete", self.sql)
        self.assertIn(
            "attribution.campaign_sku_attribution_reconciled is true",
            self.sql,
        )
        self.assertIn("and source_state.stock_snapshot_complete", self.sql)
        self.assertIn("and source_state.supplier_snapshot_complete", self.sql)
        self.assertIn("as decision_input_complete", self.sql)
        self.assertIn("false::boolean as finance_snapshot_complete", self.sql)

    def test_inventory_presence_is_not_claimed_as_snapshot_completeness(self):
        self.assertIn("as stock_snapshot_present", self.sql)
        self.assertIn("false::boolean as stock_snapshot_complete", self.sql)
        self.assertIn("as supplier_snapshot_present", self.sql)
        self.assertIn("false::boolean as supplier_snapshot_complete", self.sql)

    def test_never_uses_future_cogs(self):
        self.assertIn("and c.upload_date <= s.report_date", self.sql)


class FinanceMartSqlTests(unittest.TestCase):
    def test_never_falls_forward_to_future_cogs(self):
        sql = (MART_ROOT / "finance_mart.sql").read_text(encoding="utf-8").lower()

        self.assertIn("and c.upload_date <= r.report_date", sql)
        self.assertNotIn("(c.upload_date <= r.report_date) desc", sql)
        self.assertNotIn(
            "case when c.upload_date <= r.report_date then c.upload_date end",
            sql,
        )


class _PreflightCursor:
    def __init__(self, *, missing_table: str | None = None, missing_column: str | None = None):
        self.missing_table = missing_table
        self.missing_column = missing_column
        self._rows = []

    def execute(self, sql, _params):
        if "to_regclass" in sql:
            self._rows = [
                {
                    "required_name": table_name,
                    "resolved_name": (
                        None if table_name == self.missing_table else table_name
                    ),
                }
                for table_name in sorted(REQUIRED_BASE_COLUMNS)
            ]
            return

        rows = []
        for table_name, columns in REQUIRED_BASE_COLUMNS.items():
            schema_name, unqualified_name = table_name.split(".", 1)
            for column_name in columns:
                qualified_column = f"{table_name}.{column_name}"
                if qualified_column == self.missing_column:
                    continue
                rows.append(
                    {
                        "table_schema": schema_name,
                        "table_name": unqualified_name,
                        "column_name": column_name,
                    }
                )
        self._rows = rows

    def fetchall(self):
        return self._rows


class ApplyAgentMartsTests(unittest.TestCase):
    def test_apply_script_serializes_concurrent_worker_migrations(self):
        source = (ROOT / "scripts" / "apply_agent_marts.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("pg_advisory_xact_lock", source)
        self.assertIn("agent-marts-migration:{schema}", source)

    def test_script_supports_direct_invocation(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "apply_agent_marts.py"),
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--check", completed.stdout)
        self.assertIn("--apply", completed.stdout)

    def test_preflight_reports_missing_base_table(self):
        table_name = "marketing.wb_marketing_campaign_stats_daily_nms"

        with self.assertRaisesRegex(RuntimeError, table_name):
            _preflight_base_contract(
                _PreflightCursor(missing_table=table_name)
            )

    def test_preflight_reports_missing_required_column(self):
        column_name = "marketing.wb_marketing_campaign_stats_daily_nms.sum_spend"

        with self.assertRaisesRegex(RuntimeError, column_name):
            _preflight_base_contract(
                _PreflightCursor(missing_column=column_name)
            )

    def test_preflight_accepts_complete_contract(self):
        _preflight_base_contract(_PreflightCursor())


if __name__ == "__main__":
    unittest.main()
