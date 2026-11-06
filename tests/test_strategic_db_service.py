import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.services import strategic_db as strategic_db_module
from agi_ecom_app.services.strategic_db import (
    _build_supply_stock_lookup,
    _source_freshness,
    get_marketing_deep_diagnostic,
    get_marketing_action_matrix,
    get_marketing_overview,
    get_marketing_campaign_period_report,
    get_marketing_drr_report,
    get_marketing_sku_period_comparison_report,
    get_marketing_query_performance_snapshot,
    get_marketing_report_snapshot,
    get_marketing_search_demand_snapshot,
    get_price_competitor_details,
    get_price_competitor_overview,
    get_price_context,
    get_price_decision_overview,
    get_sales_overview,
    get_sales_report_snapshot,
    get_sales_sku_revenue_change,
    get_sku_daily_price_sales_comparison,
    get_inventory_turnover,
    get_current_inventory_value,
    get_supply_overview,
)


class FakeCursor:
    def __init__(self, row=None, rows=None):
        self.row = row
        self.rows = rows or []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, row=None, rows=None):
        self.cursor_obj = FakeCursor(row=row, rows=rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


LEGACY_RUNTIME_TABLES = (
    "supply.v_wb_supplier_stocks_latest_daily",
    "supply.v_wb_supplies_latest_daily",
    "supply.v_wb_supplies_goods_latest_daily",
    "marketing.v_agent_marketing_campaign_daily",
    "marketing.v_agent_marketing_search_text_daily",
    "marketing.v_agent_marketing_search_sku_daily",
)


def _executed_sql(fake_connection):
    return "\n".join(query for query, _params in fake_connection.cursor_obj.executed)


class RuntimeDataDependencyTests(unittest.TestCase):
    def test_future_dated_source_is_invalid_not_fresh(self):
        result = _source_freshness(
            report_date=date(2026, 7, 11),
            reference_date=date(2026, 6, 23),
            max_lag_days=2,
        )

        self.assertEqual(result["lag_days"], -18)
        self.assertEqual(result["status"], "future")

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_sales_overview_uses_source_specific_keyword_columns(
        self,
        mock_get_connection,
    ):
        fake_connection = FakeConnection(row={}, rows=[])
        mock_get_connection.return_value = fake_connection

        get_sales_overview(product_keyword="46303", limit=5)

        totals_sql, totals_params = fake_connection.cursor_obj.executed[0]
        grouped_sql, grouped_params = fake_connection.cursor_obj.executed[1]
        funnel_sql, funnel_params = fake_connection.cursor_obj.executed[2]
        raw_sql, raw_params = fake_connection.cursor_obj.executed[3]

        self.assertIn("coalesce(s.supplier_article, '') ilike %s", totals_sql)
        self.assertIn("coalesce(s.brand, '') ilike %s", totals_sql)
        self.assertNotIn("coalesce(vendor_code, '') ilike %s", totals_sql)
        self.assertIn("coalesce(vendor_code, '') ilike %s", grouped_sql)
        self.assertIn("coalesce(vendor_code, '') ilike %s", funnel_sql)
        self.assertIn("coalesce(s.supplier_article, '') ilike %s", raw_sql)
        self.assertIn("s.supplier_article as vendor_code", raw_sql)
        expected_params = ["%46303%"] * 4
        self.assertEqual(totals_params, expected_params)
        self.assertEqual(grouped_params, [*expected_params, 5])
        self.assertEqual(funnel_params, [*expected_params, 5])
        self.assertEqual(raw_params, [*expected_params, 5])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_daily_price_sales_comparison_joins_funnel_by_day_and_sku(
        self,
        mock_get_connection,
    ):
        fake_connection = FakeConnection(
            rows=[
                {
                    "report_date": date(2026, 7, 19),
                    "nm_id": 171742855,
                    "vendor_code": "46303",
                    "seller_effective_price_rub": Decimal("5615.85"),
                    "seller_discount_pct": Decimal("13.0"),
                    "cart_to_order_conversion": Decimal("25.0"),
                    "realized_sales_units": 5,
                }
            ]
        )
        mock_get_connection.return_value = fake_connection

        result = get_sku_daily_price_sales_comparison("46303", days=14)

        sql, params = fake_connection.cursor_obj.executed[0]
        self.assertIn("agent_marts.v_agent_price_sku_daily", sql)
        self.assertIn("sales.wb_sales_funnel_products_daily", sql)
        self.assertIn("f.report_date = p.report_date", sql)
        self.assertIn("f.nm_id = p.nm_id", sql)
        self.assertEqual(params, ["46303", "46303", "46303", "46303", 14])
        self.assertEqual(result["row_count"], 1)
        self.assertFalse(result["causal_valid"])

    def test_future_dated_marketing_rows_cannot_create_current_actions(self):
        future_date = date(2026, 7, 11)
        reference_date = date(2026, 6, 23)
        empty_fresh = {"source_status": "ready", "report_date": reference_date, "rows": []}

        with (
            patch.object(strategic_db_module, "_marketing_latest_stats_date", return_value=reference_date),
            patch.object(
                strategic_db_module,
                "get_marketing_campaign_daily_mart",
                return_value={
                    "source_status": "ready",
                    "report_date": future_date,
                    "rows": [
                        {
                            "advert_id": 123,
                            "orders": 0,
                            "spend": 500,
                            "is_cut_candidate": True,
                        }
                    ],
                },
            ),
            patch.object(strategic_db_module, "get_marketing_query_bands_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_forecast_ready_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_query_demand_daily_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_query_share_daily_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_query_forecast_ready_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_campaign_demand_forecast_mart", return_value=empty_fresh),
            patch.object(strategic_db_module, "get_marketing_visibility_snapshot", return_value={"position_loss_rows": []}),
            patch.object(
                strategic_db_module,
                "get_marketing_campaign_efficiency_snapshot",
                return_value={
                    "report_date": future_date,
                    "campaign_rows": [
                        {"advert_id": 123, "status": "inefficient", "sum_spend": 500, "orders": 0}
                    ],
                },
            ),
            patch.object(
                strategic_db_module,
                "get_marketing_query_performance_snapshot",
                return_value={"report_date": future_date, "query_waste_rows": []},
            ),
        ):
            result = get_marketing_action_matrix(days=7, limit=5)

        self.assertEqual(
            result["source_freshness"]["sources"]["campaign_mart"]["status"],
            "future",
        )
        self.assertIn("campaign_mart", result["source_freshness"]["invalid_sources"])
        self.assertEqual(result["campaign_actions"], [])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_current_inventory_value_uses_sellable_on_hand_times_unit_cost(self, mock_get_connection):
        fake_connection = FakeConnection(
            rows=[
                {
                    "report_date": date(2026, 7, 10),
                    "nm_id": 1,
                    "vendor_code": "46227",
                    "available_stock_qty": 5,
                    "unit_cost_rub": Decimal("150.00"),
                    "current_stock_value_rub": Decimal("750.00"),
                    "valuation_status": "confirmed",
                },
                {
                    "report_date": date(2026, 7, 10),
                    "nm_id": 2,
                    "vendor_code": "46230",
                    "available_stock_qty": 20,
                    "unit_cost_rub": None,
                    "current_stock_value_rub": None,
                    "valuation_status": "missing_cogs",
                },
            ]
        )
        mock_get_connection.return_value = fake_connection

        result = get_current_inventory_value(limit=20)

        sql = _executed_sql(fake_connection)
        self.assertIn("sum(coalesce(quantity, 0)) as available_stock_qty", sql)
        self.assertIn("st.available_stock_qty * cost.cost_price_rub", sql)
        self.assertNotIn("sales_7d_units * cost", sql)
        self.assertNotIn("quantity_full * cost", sql)
        self.assertEqual(result["totals"]["confirmed_stock_value_rub"], 750.0)
        self.assertEqual(result["totals"]["missing_cogs_sku_count"], 1)

    def test_app_runtime_no_longer_references_legacy_helper_views(self):
        app_root = Path(__file__).resolve().parents[1] / "agi_ecom_app"
        offenders = []
        for path in app_root.rglob("*"):
            if path.suffix not in {".py", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for legacy_name in LEGACY_RUNTIME_TABLES:
                if legacy_name in text:
                    offenders.append(f"{path.relative_to(app_root)}: {legacy_name}")

        self.assertEqual(offenders, [])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_supply_overview_uses_raw_supply_tables(self, mock_get_connection):
        fake_connection = FakeConnection(row={}, rows=[])
        mock_get_connection.return_value = fake_connection

        result = get_supply_overview(limit=5)

        self.assertEqual(
            set(result),
            {
                "focus_keyword",
                "data_mode",
                "simulation",
                "sku_stock_lookup",
                "totals",
                "rows",
                "critical_reorder_rows",
                "overstock_rows",
                "pipeline_rows",
                "stock_snapshot_rows",
                "calculation_methods",
            },
        )
        executed_sql = _executed_sql(fake_connection)
        self.assertIn("supply.wb_supplier_stocks", executed_sql)
        self.assertIn("supply.wb_supplies", executed_sql)
        self.assertIn("supply.wb_supplies_goods", executed_sql)
        self.assertIn("coalesce(inbound.supplier_inbound_qty, 0::numeric)", executed_sql)
        self.assertIn("supplier_audit_snapshot_date", executed_sql)
        self.assertIn("st.stock_qty::numeric / (s7.sales_7d_units::numeric / 7)", executed_sql)
        self.assertIn("total_stock_days_for_audit", executed_sql)
        self.assertIn("unavailable_stock_qty", executed_sql)
        self.assertIn("quantity / (sales_7d_units / 7)", result["calculation_methods"]["days_of_stock"])
        self.assertIn("audited absent SKU is explicit zero", result["calculation_methods"]["supplier_inbound_qty"])
        for legacy_name in LEGACY_RUNTIME_TABLES:
            self.assertNotIn(legacy_name, executed_sql)

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_supply_overview_accepts_full_portfolio_limit(self, mock_get_connection):
        fake_connection = FakeConnection(row={}, rows=[])
        mock_get_connection.return_value = fake_connection

        get_supply_overview(limit=17)

        for _query, params in fake_connection.cursor_obj.executed:
            if params:
                self.assertEqual(params[-1], 17)

    @patch(
        "agi_ecom_app.services.strategic_db.get_settings",
        return_value=SimpleNamespace(supply_latest_known_as_current=True),
    )
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_supply_overview_simulates_latest_snapshot_and_filters_multiple_skus(
        self,
        mock_get_connection,
        _mock_get_settings,
    ):
        rows = [
            {
                "report_date": date(2026, 7, 8),
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 70195171,
                "vendor_code": "46229",
                "stock_qty": 0,
                "stock_qty_full": 3,
                "supplier_audit_snapshot_date": date(2026, 7, 8),
                "supplier_inbound_qty": 0,
            },
            {
                "report_date": date(2026, 7, 8),
                "snapshot_date": date(2026, 7, 8),
                "nm_id": 768260062,
                "vendor_code": "46313",
                "stock_qty": 1,
                "stock_qty_full": 4,
            },
        ]
        fake_connection = FakeConnection(row={"report_date": date(2026, 7, 8)}, rows=rows)
        mock_get_connection.return_value = fake_connection

        result = get_supply_overview(
            product_keyword="70195171,768260062",
            limit=5,
        )

        stock_sql, stock_params = fake_connection.cursor_obj.executed[-1]
        self.assertNotIn("ilike", stock_sql.lower())
        self.assertEqual(
            stock_params,
            ["70195171", "70195171", "768260062", "768260062", 5],
        )
        self.assertEqual(result["data_mode"], "simulation_latest_known")
        self.assertEqual(
            result["simulation"],
            {
                "enabled": True,
                "policy": "latest_known_as_current",
                "source_snapshot_date": "2026-07-08",
                "effective_status": "simulated_current",
            },
        )
        self.assertEqual(result["sku_stock_lookup"]["status"], "found_multiple")
        self.assertEqual(result["sku_stock_lookup"]["missing_queries"], [])
        self.assertEqual(result["sku_stock_lookup"]["match_count"], 2)
        self.assertEqual(
            result["sku_stock_lookup"]["matches"][0]["supplier_inbound_qty"],
            0,
        )

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_supply_overview_aggregates_and_filters_stock_snapshot_rows(
        self,
        mock_get_connection,
    ):
        fake_connection = FakeConnection(row={}, rows=[])
        mock_get_connection.return_value = fake_connection

        result = get_supply_overview(product_keyword="46227", limit=5)

        stock_sql, stock_params = fake_connection.cursor_obj.executed[-1]
        self.assertIn("sum(coalesce(quantity, 0)) as quantity", stock_sql)
        self.assertIn(
            "group by snapshot_date, nm_id, supplier_article, brand, subject",
            stock_sql,
        )
        self.assertNotIn("max(coalesce(quantity, 0))", stock_sql)
        self.assertEqual(stock_params, ["%46227%"] * 4 + [5])
        self.assertEqual(result["sku_stock_lookup"]["status"], "no_data")
        self.assertIsNone(result["sku_stock_lookup"]["available_stock_qty"])

    def test_supply_stock_lookup_exposes_aggregated_value_not_warehouse_max(self):
        lookup = _build_supply_stock_lookup(
            "46227",
            [
                {
                    "report_date": date(2026, 7, 8),
                    "nm_id": 70194686,
                    "vendor_code": "46227",
                    "stock_qty": 327,
                    "stock_qty_full": 343,
                    "unavailable_stock_qty": 16,
                    "in_way_to_client": 13,
                    "in_way_from_client": 3,
                }
            ],
        )

        self.assertEqual(lookup["status"], "found")
        self.assertEqual(lookup["available_stock_qty"], 327)
        self.assertEqual(lookup["match_count"], 1)
        self.assertIn("sum(quantity)", lookup["aggregation_method"])

    def test_supply_stock_lookup_preserves_missing_row_as_no_data(self):
        lookup = _build_supply_stock_lookup("missing-sku", [])

        self.assertEqual(lookup["status"], "no_data")
        self.assertEqual(lookup["match_count"], 0)
        self.assertIsNone(lookup["available_stock_qty"])

    def test_supply_stock_lookup_reports_partial_multi_sku_match(self):
        lookup = _build_supply_stock_lookup(
            "70195171,768260062",
            [
                {
                    "report_date": date(2026, 7, 8),
                    "nm_id": 70195171,
                    "vendor_code": "46229",
                    "stock_qty": 0,
                    "stock_qty_full": 3,
                }
            ],
        )

        self.assertEqual(lookup["status"], "partial")
        self.assertEqual(lookup["matched_queries"], ["70195171"])
        self.assertEqual(lookup["missing_queries"], ["768260062"])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_inventory_turnover_uses_sales_and_supply_tables(self, mock_get_connection):
        fake_connection = FakeConnection(
            row={
                "date_from": date(2026, 4, 1),
                "date_to": date(2026, 4, 7),
                "period_days": 7,
                "stock_snapshot_days": 7,
                "avg_stock_qty": Decimal("10.00"),
                "avg_stock_qty_full": Decimal("10.00"),
                "sales_units": 5,
                "sales_revenue": Decimal("1000.00"),
                "turnover_ratio": Decimal("0.5000"),
                "days_on_hand": Decimal("14.0"),
            },
            rows=[
                {
                    "nm_id": 1,
                    "vendor_code": "46312",
                    "sales_units": 5,
                    "sales_7d_units": 5,
                    "avg_stock_qty": Decimal("10.00"),
                    "avg_stock_qty_full": Decimal("10.00"),
                    "current_stock_qty": 0,
                    "turnover_ratio": Decimal("0.5000"),
                    "days_on_hand": Decimal("14.0"),
                }
            ],
        )
        mock_get_connection.return_value = fake_connection

        result = get_inventory_turnover(
            date_from="2026-04-01",
            date_to="2026-04-07",
            vendor_codes="46312",
            limit=5,
        )

        executed_sql = _executed_sql(fake_connection)
        self.assertIn("supply.wb_supplier_stocks", executed_sql)
        self.assertIn("sales.wb_supplier_sales", executed_sql)
        self.assertIn("supplier_article = any(%s::text[])", executed_sql)
        self.assertEqual(
            result["method"],
            "primary: sales_units / average_sellable_stock_qty from daily stock snapshots over selected period",
        )
        self.assertEqual(result["calculation_status"], "exact")
        self.assertTrue(result["exact_available"])
        self.assertEqual(result["vendor_codes"], ["46312"])
        self.assertEqual(result["source_coverage"], {"stock_rows": 1, "sales_rows": 1})
        self.assertEqual(result["rows"][0]["sku_conclusion"], "продажи были, текущий доступный остаток отсутствует")
        self.assertEqual(result["rows"][0]["turnover_band"], "норма 14–30 дней")
        self.assertEqual(result["rows"][0]["availability_status"], "нет доступного остатка")
        self.assertEqual(result["sku_metric_matrix"][0]["sales_7d_units"], 5)
        self.assertEqual(result["sku_metric_matrix"][0]["sku"], "46312")
        self.assertIn("conclusion", result["sku_metric_matrix"][0])
        self.assertIn("per-SKU conclusion", result["sku_metric_matrix_contract"])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_inventory_turnover_gap_requests_alternative_search(self, mock_get_connection):
        fake_connection = FakeConnection(
            row={
                "date_from": date(2026, 4, 1),
                "date_to": date(2026, 4, 7),
                "period_days": 7,
                "stock_snapshot_days": 0,
                "avg_stock_qty": None,
                "avg_stock_qty_full": None,
                "sales_units": 5,
            },
            rows=[],
        )
        mock_get_connection.return_value = fake_connection

        result = get_inventory_turnover(date_from="2026-04-01", date_to="2026-04-07")

        self.assertFalse(result["exact_available"])
        self.assertEqual(result["calculation_status"], "needs_alternative_search")
        self.assertIn("альтернативные методы", result["limitation"])
        self.assertEqual(
            [method["name"] for method in result["alternative_methods"]],
            [
                "boundary_snapshot_average",
                "movement_reconstruction",
                "latest_stock_velocity_proxy",
            ],
        )

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_marketing_diagnostics_use_raw_marketing_tables(self, mock_get_connection):
        fake_connection = FakeConnection(
            row={
                "stat_date": date(2026, 4, 6),
                "report_date": date(2026, 4, 6),
                "snapshot_date": date(2026, 4, 6),
            },
            rows=[],
        )
        mock_get_connection.return_value = fake_connection

        overview = get_marketing_overview(limit=5)
        diagnostic = get_marketing_deep_diagnostic(days=7, limit=5)

        self.assertEqual(
            set(overview),
            {
                "focus_keyword",
                "totals",
                "budget",
                "top_spend_campaigns",
                "inefficient_campaigns",
                "campaign_nm_rows",
            },
        )
        self.assertEqual(diagnostic["period_days"], 7)
        executed_sql = _executed_sql(fake_connection)
        self.assertIn("marketing.wb_marketing_campaign_stats_daily", executed_sql)
        self.assertIn("marketing.wb_marketing_campaigns", executed_sql)
        self.assertIn("marketing.wb_marketing_campaign_nms", executed_sql)
        self.assertIn("marketing.wb_marketing_budget", executed_sql)
        self.assertIn("marketing.wb_search_query_details", executed_sql)
        self.assertIn("marketing.wb_search_query_orders_text_daily", executed_sql)
        for legacy_name in LEGACY_RUNTIME_TABLES:
            self.assertNotIn(legacy_name, executed_sql)


class MarketingCampaignPeriodReportTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_returns_expected_period_report(self, mock_get_connection):
        rows = [
            {
                "ID кампании": 12345,
                "Кампания, Тип": "Тестовая кампания, Ручная ставка",
                "Дата начала": date(2026, 3, 20),
                "Статус": "Активна",
                "Артикулы в кампании": "1001, 1002",
                "Цена продажи без СПП, ₽": "1001: 999.00; 1002: 1199.00",
                "Показы": 1000,
                "Клики": 50,
                "Кликабельность, %": Decimal("5.00"),
                "Цена клика, ₽": Decimal("12.35"),
                "Конверсия клика в заказ, %": Decimal("4.00"),
                "Все добавления в корзину": 10,
                "Все заказанные товары, шт": 2,
                "Затраты, ₽": Decimal("617.50"),
                "Доля затрат (%)": Decimal("100.00"),
            }
        ]
        mock_get_connection.return_value = FakeConnection(rows=rows)

        result = get_marketing_campaign_period_report("2026-03-23", "2026-03-29", limit=50)

        self.assertEqual(result["date_from"], "2026-03-23")
        self.assertEqual(result["date_to"], "2026-03-29")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["rows"][0]["ID кампании"], 12345)
        self.assertEqual(result["rows"][0]["Дата начала"], "2026-03-20")
        self.assertEqual(result["rows"][0]["Кликабельность, %"], 5.0)
        self.assertEqual(result["rows"][0]["Затраты, ₽"], 617.5)
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[0][1],
            [date(2026, 3, 23), date(2026, 3, 29), 50],
        )
        executed_query = mock_get_connection.return_value.cursor_obj.executed[0][0]
        self.assertIn('"Доля затрат (%%)"', executed_query)

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_accepts_compact_period_string(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(rows=[])

        result = get_marketing_campaign_period_report("23.03-29.03")

        self.assertEqual(result["date_from"], "2026-03-23")
        self.assertEqual(result["date_to"], "2026-03-29")
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[0][1],
            [date(2026, 3, 23), date(2026, 3, 29), 200],
        )

    def test_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            get_marketing_campaign_period_report("2026-03-23", "2026-03-29", limit=0)


class MarketingDrrReportTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_returns_deterministic_drr_report(self, mock_get_connection):
        totals = {
            "campaign_count": 2,
            "promoted_nm_count": 2,
            "marketing_spend": Decimal("100.00"),
            "views": 1000,
            "clicks": 100,
            "ad_orders": 10,
            "store_revenue": Decimal("1000.00"),
            "store_sales": 20,
            "promoted_sku_revenue": Decimal("500.00"),
            "promoted_sku_sales": 12,
            "promoted_revenue_share_pct": Decimal("50.00"),
            "promoted_sales_share_pct": Decimal("60.00"),
            "estimated_drr_promoted_sku_pct": Decimal("20.00"),
            "estimated_drr_store_pct": Decimal("10.00"),
        }
        campaign_rows = [
            {
                "advert_id": 1,
                "campaign_name": "Кампания",
                "spend_rub": Decimal("100.00"),
                "nm_ids": "123",
            }
        ]
        sku_rows = [
            {
                "nm_id": 123,
                "vendor_code": "A1",
                "subject_name": "Кремы",
                "revenue": Decimal("500.00"),
                "sales": 12,
            }
        ]
        fake_connection = FakeConnection(row=totals)
        fake_connection.cursor_obj.rows = campaign_rows
        mock_get_connection.return_value = fake_connection

        def fetchall_side_effect():
            calls = len(fake_connection.cursor_obj.executed)
            return campaign_rows if calls == 2 else sku_rows

        fake_connection.cursor_obj.fetchall = fetchall_side_effect

        result = get_marketing_drr_report("2026-03-08", "2026-04-06", limit=10)

        self.assertEqual(result["date_from"], "2026-03-08")
        self.assertEqual(result["date_to"], "2026-04-06")
        self.assertEqual(result["spend_source"], "marketing.wb_marketing_campaign_stats_daily.sum_spend")
        self.assertEqual(result["totals"]["estimated_drr_promoted_sku_pct"], 20.0)
        self.assertEqual(result["campaign_rows"][0]["advert_id"], 1)
        self.assertEqual(result["promoted_sku_rows"][0]["nm_id"], 123)
        self.assertEqual(
            fake_connection.cursor_obj.executed[0][1],
            [date(2026, 3, 8), date(2026, 4, 6)],
        )


class MarketingSkuPeriodComparisonReportTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_returns_sku_period_comparison_with_requested_skus(self, mock_get_connection):
        rows = [
            {
                "current_date_from": date(2026, 4, 14),
                "current_date_to": date(2026, 4, 20),
                "previous_date_from": date(2026, 4, 7),
                "previous_date_to": date(2026, 4, 13),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "subject_name": "Косметика",
                "current_impressions": Decimal("1000"),
                "previous_impressions": Decimal("1500"),
                "current_clicks": Decimal("50"),
                "previous_clicks": Decimal("90"),
                "current_ctr_pct": Decimal("5.00"),
                "previous_ctr_pct": Decimal("6.00"),
                "current_cpc_rub": Decimal("20.00"),
                "previous_cpc_rub": Decimal("15.00"),
                "current_ad_orders": Decimal("1"),
                "previous_ad_orders": Decimal("5"),
                "current_spend_rub": Decimal("1000.00"),
                "previous_spend_rub": Decimal("1350.00"),
                "current_estimated_drr_pct": Decimal("12.50"),
                "previous_estimated_drr_pct": Decimal("8.10"),
                "attribution_quality": "estimated_allocated_from_campaign",
            }
        ]
        mock_get_connection.return_value = FakeConnection(rows=rows)

        result = get_marketing_sku_period_comparison_report(
            "2026-04-14",
            "2026-04-20",
            sku_list="46227, 46303",
            limit=20,
        )

        self.assertEqual(result["date_from"], "2026-04-14")
        self.assertEqual(result["date_to"], "2026-04-20")
        self.assertEqual(result["previous_date_from"], "2026-04-07")
        self.assertEqual(result["previous_date_to"], "2026-04-13")
        self.assertEqual(result["focus_sku_list"], ["46227", "46303"])
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["rows"][0]["vendor_code"], "46227")
        self.assertEqual(result["rows"][0]["current_estimated_drr_pct"], 12.5)
        self.assertIn("estimated", result["allocation_note"])

        executed_query, params = mock_get_connection.return_value.cursor_obj.executed[0]
        self.assertIn("marketing.wb_marketing_campaign_stats_daily", executed_query)
        self.assertIn("marketing.wb_marketing_campaign_nms", executed_query)
        self.assertIn("cardinality(%s::text[])", executed_query)
        self.assertEqual(params[:6], [
            date(2026, 4, 14),
            date(2026, 4, 20),
            7,
            ["46227", "46303"],
            ["46227", "46303"],
            ["46227", "46303"],
        ])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_defaults_to_latest_period_when_dates_are_not_provided(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(rows=[])

        result = get_marketing_sku_period_comparison_report(period_days=7)

        self.assertEqual(result["period_days"], 7)
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[0][1][:3],
            [None, None, 7],
        )


class MarketingSearchSnapshotTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_query_performance_keyword_filter_uses_outer_alias(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(
            rows=[
                {
                    "report_date": date(2026, 4, 20),
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "norm_query": "мист для тела",
                    "orders": 1,
                    "spend": Decimal("0"),
                    "avg_pos": Decimal("4"),
                }
            ]
        )

        result = get_marketing_query_performance_snapshot(product_keyword="46227", limit=5)

        self.assertEqual(len(result["top_query_rows"]), 1)
        executed_query = mock_get_connection.return_value.cursor_obj.executed[0][0].lower()
        self.assertIn("coalesce(norm_query, '') ilike %s", executed_query)
        self.assertIn("coalesce(vendor_code, '') ilike %s", executed_query)
        self.assertNotIn("and (query_text ilike %s", executed_query)

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_search_demand_keyword_filter_uses_qualified_alias(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(
            rows=[
                {
                    "report_date": date(2026, 4, 20),
                    "query_text": "мист для тела",
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "orders_count": 1,
                    "orders_delta": 1,
                }
            ]
        )

        result = get_marketing_search_demand_snapshot(product_keyword="46227", limit=5)

        self.assertEqual(len(result["search_text_rows"]), 1)
        executed_query = mock_get_connection.return_value.cursor_obj.executed[0][0].lower()
        self.assertIn("coalesce(c.query_text, '') ilike %s", executed_query)
        self.assertIn("coalesce(c.vendor_code, '') ilike %s", executed_query)
        self.assertNotIn("and (query_text ilike %s", executed_query)


class SalesSkuRevenueChangeTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_report_snapshot_returns_required_period_comparisons(self, mock_get_connection):
        rows = [
            {
                "label": "Последний день в данных к предыдущему",
                "sort_order": 1,
                "side": "current",
                "date_from": date(2026, 4, 6),
                "date_to": date(2026, 4, 6),
                "sales_units": 10,
                "revenue_rub": Decimal("1000.00"),
            },
            {
                "label": "Последний день в данных к предыдущему",
                "sort_order": 1,
                "side": "previous",
                "date_from": date(2026, 4, 5),
                "date_to": date(2026, 4, 5),
                "sales_units": 8,
                "revenue_rub": Decimal("800.00"),
            },
            {
                "label": "Последние 7 дней к предыдущим 7",
                "sort_order": 2,
                "side": "current",
                "date_from": date(2026, 3, 31),
                "date_to": date(2026, 4, 6),
                "sales_units": 70,
                "revenue_rub": Decimal("7000.00"),
            },
            {
                "label": "Последние 7 дней к предыдущим 7",
                "sort_order": 2,
                "side": "previous",
                "date_from": date(2026, 3, 24),
                "date_to": date(2026, 3, 30),
                "sales_units": 100,
                "revenue_rub": Decimal("10000.00"),
            },
            {
                "label": "Последние 30 дней к предыдущим 30",
                "sort_order": 3,
                "side": "current",
                "date_from": date(2026, 3, 8),
                "date_to": date(2026, 4, 6),
                "sales_units": 300,
                "revenue_rub": Decimal("30000.00"),
            },
            {
                "label": "Последние 30 дней к предыдущим 30",
                "sort_order": 3,
                "side": "previous",
                "date_from": date(2026, 2, 6),
                "date_to": date(2026, 3, 7),
                "sales_units": 200,
                "revenue_rub": Decimal("25000.00"),
            },
        ]
        mock_get_connection.return_value = FakeConnection(
            row={"latest_sales_date": date(2026, 4, 6)},
            rows=rows,
        )

        result = get_sales_report_snapshot()

        self.assertEqual(result["report_date"], "2026-04-06")
        self.assertEqual(result["source_table"], "sales.wb_supplier_sales")
        self.assertEqual([row["period"] for row in result["rows"]], [
            "Последний день в данных к предыдущему",
            "Последние 7 дней к предыдущим 7",
            "Последние 30 дней к предыдущим 30",
        ])
        self.assertEqual(result["rows"][0]["sales_units_delta"], 2)
        self.assertEqual(result["rows"][0]["sales_units_delta_pct"], 25.0)
        self.assertEqual(result["rows"][1]["revenue_delta_pct"], -30.0)
        self.assertEqual(result["rows"][2]["revenue_delta_rub"], 5000.0)
        self.assertIn("| Сравнение |", result["markdown_table"])
        self.assertEqual(result["report_table"]["title"], "Sales-сводка")
        self.assertEqual(
            result["report_table"]["columns"],
            [
                "Сравнение",
                "Периоды",
                "Продажи тек./пред.",
                "Δ продаж",
                "Выручка тек./пред.",
                "Δ выручки",
            ],
        )
        self.assertEqual(result["report_table"]["rows"][0][0], "Последний день в данных к предыдущему")
        self.assertEqual(result["report_table"]["rows"][0][1], "06.04.2026 к 05.04.2026")
        self.assertEqual(result["report_table"]["rows"][0][2], "10 / 8")
        self.assertIn("rolling 30 days", result["calculations"]["month_to_month_default"])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_marketing_report_snapshot_returns_period_rows_with_drr(self, mock_get_connection):
        rows = [
            {
                "label": "Последний день в данных к предыдущему",
                "sort_order": 1,
                "side": "current",
                "date_from": date(2026, 4, 6),
                "date_to": date(2026, 4, 6),
                "campaign_count": 2,
                "promoted_nm_count": 3,
                "spend_rub": Decimal("1000.00"),
                "views": 10000,
                "clicks": 200,
                "cpc_rub": Decimal("5.00"),
                "ad_orders": 20,
                "promoted_sku_revenue_rub": Decimal("10000.00"),
                "drr_pct": Decimal("10.00"),
            },
            {
                "label": "Последний день в данных к предыдущему",
                "sort_order": 1,
                "side": "previous",
                "date_from": date(2026, 4, 5),
                "date_to": date(2026, 4, 5),
                "campaign_count": 1,
                "promoted_nm_count": 2,
                "spend_rub": Decimal("500.00"),
                "views": 8000,
                "clicks": 100,
                "cpc_rub": Decimal("5.00"),
                "ad_orders": 10,
                "promoted_sku_revenue_rub": Decimal("10000.00"),
                "drr_pct": Decimal("5.00"),
            },
            {
                "label": "Последние 7 дней к предыдущим 7",
                "sort_order": 2,
                "side": "current",
                "date_from": date(2026, 3, 31),
                "date_to": date(2026, 4, 6),
                "campaign_count": 4,
                "promoted_nm_count": 5,
                "spend_rub": Decimal("7000.00"),
                "views": 50000,
                "clicks": 700,
                "cpc_rub": Decimal("10.00"),
                "ad_orders": 70,
                "promoted_sku_revenue_rub": Decimal("35000.00"),
                "drr_pct": Decimal("20.00"),
            },
            {
                "label": "Последние 7 дней к предыдущим 7",
                "sort_order": 2,
                "side": "previous",
                "date_from": date(2026, 3, 24),
                "date_to": date(2026, 3, 30),
                "campaign_count": 4,
                "promoted_nm_count": 5,
                "spend_rub": Decimal("5000.00"),
                "views": 45000,
                "clicks": 500,
                "cpc_rub": Decimal("10.00"),
                "ad_orders": 60,
                "promoted_sku_revenue_rub": Decimal("50000.00"),
                "drr_pct": Decimal("10.00"),
            },
            {
                "label": "Последние 30 дней к предыдущим 30",
                "sort_order": 3,
                "side": "current",
                "date_from": date(2026, 3, 8),
                "date_to": date(2026, 4, 6),
                "campaign_count": 5,
                "promoted_nm_count": 6,
                "spend_rub": Decimal("30000.00"),
                "views": 100000,
                "clicks": 3000,
                "cpc_rub": Decimal("10.00"),
                "ad_orders": 300,
                "promoted_sku_revenue_rub": Decimal("150000.00"),
                "drr_pct": Decimal("20.00"),
            },
            {
                "label": "Последние 30 дней к предыдущим 30",
                "sort_order": 3,
                "side": "previous",
                "date_from": date(2026, 2, 6),
                "date_to": date(2026, 3, 7),
                "campaign_count": 5,
                "promoted_nm_count": 6,
                "spend_rub": Decimal("20000.00"),
                "views": 90000,
                "clicks": 2000,
                "cpc_rub": Decimal("10.00"),
                "ad_orders": 250,
                "promoted_sku_revenue_rub": Decimal("200000.00"),
                "drr_pct": Decimal("10.00"),
            },
        ]
        mock_get_connection.return_value = FakeConnection(
            row={"report_date": date(2026, 4, 6)},
            rows=rows,
        )

        result = get_marketing_report_snapshot()

        self.assertEqual(result["report_date"], "2026-04-06")
        self.assertEqual([row["period"] for row in result["rows"]], [
            "Последний день в данных к предыдущему",
            "Последние 7 дней к предыдущим 7",
            "Последние 30 дней к предыдущим 30",
        ])
        self.assertEqual(result["rows"][0]["spend_delta_rub"], 500.0)
        self.assertEqual(result["rows"][0]["spend_delta_pct"], 100.0)
        self.assertEqual(result["rows"][0]["current_cpc_rub"], 5.0)
        self.assertEqual(result["rows"][0]["current_promoted_sku_revenue_rub"], 10000.0)
        self.assertEqual(result["rows"][0]["current_drr_pct"], 10.0)
        self.assertEqual(result["rows"][0]["drr_delta_pp"], 5.0)
        self.assertEqual(result["rows"][1]["drr_delta_pp"], 10.0)
        self.assertIn("ДРР тек./пред.", result["report_table"]["columns"])
        self.assertEqual(result["report_table"]["title"], "Marketing-сводка")
        self.assertEqual(result["report_table"]["rows"][0][0], "Последний день в данных к предыдущему")
        self.assertEqual(result["report_table"]["rows"][0][1], "06.04.2026 к 05.04.2026")
        self.assertIn("| Сравнение |", result["markdown_table"])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_marketing_report_snapshot_returns_empty_table_without_source_data(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(row={"report_date": None}, rows=[])

        result = get_marketing_report_snapshot()

        self.assertIsNone(result["report_date"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["report_table"]["title"], "Marketing-сводка")
        self.assertIn("ДРР тек./пред.", result["report_table"]["columns"])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_marketing_report_snapshot_filters_brand_identity(self, mock_get_connection):
        fake_connection = FakeConnection(row={"report_date": date(2026, 4, 6)}, rows=[])
        mock_get_connection.return_value = fake_connection

        get_marketing_report_snapshot(product_keyword="LIFECODE")

        executed_sql = _executed_sql(fake_connection)
        self.assertIn("si.brand_name", executed_sql)
        self.assertIn("max(nullif(s.brand, '')) as brand_name", executed_sql)
        self.assertIn("max(nullif(f.brand_name, '')) as brand_name", executed_sql)

    @patch("agi_ecom_app.services.strategic_db.get_marketing_query_performance_snapshot")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_campaign_efficiency_snapshot")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_visibility_snapshot")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_campaign_demand_forecast_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_query_forecast_ready_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_query_share_daily_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_query_demand_daily_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_forecast_ready_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_query_bands_mart")
    @patch("agi_ecom_app.services.strategic_db.get_marketing_campaign_daily_mart")
    @patch("agi_ecom_app.services.strategic_db._marketing_latest_stats_date")
    def test_marketing_action_matrix_skips_stale_query_mart_actions(
        self,
        mock_latest_stats_date,
        mock_campaign_mart,
        mock_query_bands,
        mock_forecast_ready,
        mock_query_demand,
        mock_query_share,
        mock_query_forecast,
        mock_campaign_demand,
        mock_visibility,
        mock_legacy_campaign,
        mock_legacy_query,
    ):
        mock_latest_stats_date.return_value = date(2026, 6, 23)
        mock_campaign_mart.return_value = {
            "source_status": "ready",
            "report_date": None,
            "rows": [],
        }
        mock_query_bands.return_value = {
            "source_status": "ready",
            "report_date": date(2026, 5, 1),
            "rows": [
                {
                    "query_band": "waste",
                    "norm_query": "stale query",
                    "orders_7d": 0,
                    "spend_7d": 100,
                    "avg_pos_7d": 1,
                }
            ],
        }
        mock_forecast_ready.return_value = {"source_status": "ready", "report_date": None, "rows": []}
        mock_query_demand.return_value = {"source_status": "ready", "report_date": date(2026, 6, 23), "rows": []}
        mock_query_share.return_value = {"source_status": "ready", "report_date": date(2026, 5, 1), "rows": []}
        mock_query_forecast.return_value = {"source_status": "ready", "report_date": date(2026, 5, 1), "rows": []}
        mock_campaign_demand.return_value = {"source_status": "ready", "report_date": date(2026, 4, 30), "rows": []}
        mock_visibility.return_value = {"position_loss_rows": []}
        mock_legacy_campaign.return_value = {
            "report_date": date(2026, 6, 23),
            "campaign_rows": [
                {
                    "advert_id": 25370757,
                    "campaign_name": "156051054 46300 МПГ",
                    "status": "inefficient",
                    "orders": 0,
                    "sum_spend": 7258.46,
                    "cpc": 25.23,
                    "cr": None,
                }
            ],
        }
        mock_legacy_query.return_value = {
            "report_date": date(2026, 6, 23),
            "query_waste_rows": [],
        }

        result = get_marketing_action_matrix(product_keyword="LIFECODE", days=7, limit=5)

        self.assertEqual(result["report_date"], date(2026, 6, 23))
        self.assertIn("query_bands_mart", result["source_freshness"]["stale_sources"])
        self.assertEqual(result["query_actions"], [])
        self.assertEqual(result["campaign_actions"][0]["entity_id"], 25370757)
        self.assertEqual(result["campaign_actions"][0]["action_mode"], "reduce")

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_default_period_returns_revenue_and_sales_deltas(self, mock_get_connection):
        rows = [
            {
                "current_period_start": date(2026, 3, 31),
                "current_period_end": date(2026, 4, 6),
                "previous_period_start": date(2026, 3, 24),
                "previous_period_end": date(2026, 3, 30),
                "nm_id": 768260062,
                "vendor_code": "46313",
                "brand_name": "LIFECODE",
                "subject_name": "Кремы",
                "current_revenue": Decimal("31054.43"),
                "previous_revenue": Decimal("52769.82"),
                "delta_revenue": Decimal("-21715.39"),
                "delta_revenue_pct": Decimal("-41.15"),
                "current_sales": 12,
                "previous_sales": 20,
                "delta_sales": -8,
                "driver_role": "decline_driver",
            },
            {
                "current_period_start": date(2026, 3, 31),
                "current_period_end": date(2026, 4, 6),
                "previous_period_start": date(2026, 3, 24),
                "previous_period_end": date(2026, 3, 30),
                "nm_id": 70194686,
                "vendor_code": "46227",
                "brand_name": "LIFECODE",
                "subject_name": "Сыворотки",
                "current_revenue": Decimal("262893.69"),
                "previous_revenue": Decimal("229900.00"),
                "delta_revenue": Decimal("32993.69"),
                "delta_revenue_pct": Decimal("14.35"),
                "current_sales": 77,
                "previous_sales": 70,
                "delta_sales": 7,
                "driver_role": "growth_driver",
            },
            {
                "current_period_start": date(2026, 3, 31),
                "current_period_end": date(2026, 4, 6),
                "previous_period_start": date(2026, 3, 24),
                "previous_period_end": date(2026, 3, 30),
                "nm_id": 1,
                "vendor_code": "new",
                "brand_name": "Brand",
                "subject_name": "Subject",
                "current_revenue": Decimal("100.00"),
                "previous_revenue": Decimal("0.00"),
                "delta_revenue": Decimal("100.00"),
                "delta_revenue_pct": None,
                "current_sales": 1,
                "previous_sales": 0,
                "delta_sales": 1,
                "driver_role": "new_in_current",
            },
            {
                "current_period_start": date(2026, 3, 31),
                "current_period_end": date(2026, 4, 6),
                "previous_period_start": date(2026, 3, 24),
                "previous_period_end": date(2026, 3, 30),
                "nm_id": 2,
                "vendor_code": "lost",
                "brand_name": "Brand",
                "subject_name": "Subject",
                "current_revenue": Decimal("0.00"),
                "previous_revenue": Decimal("50.00"),
                "delta_revenue": Decimal("-50.00"),
                "delta_revenue_pct": Decimal("-100.00"),
                "current_sales": 0,
                "previous_sales": 1,
                "delta_sales": -1,
                "driver_role": "lost_in_current",
            },
        ]
        mock_get_connection.return_value = FakeConnection(
            row={"latest_sales_date": date(2026, 4, 6)},
            rows=rows,
        )

        result = get_sales_sku_revenue_change(limit=20)

        self.assertEqual(result["current_period_start"], "2026-03-31")
        self.assertEqual(result["current_period_end"], "2026-04-06")
        self.assertEqual(result["previous_period_start"], "2026-03-24")
        self.assertEqual(result["previous_period_end"], "2026-03-30")
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["source_table"], "sales.wb_supplier_sales")
        self.assertEqual(result["rows"][0]["vendor_code"], "46313")
        self.assertEqual(result["rows"][0]["previous_revenue"], 52769.82)
        self.assertEqual(result["rows"][0]["delta_revenue"], -21715.39)
        self.assertEqual(result["rows"][2]["driver_role"], "new_in_current")
        self.assertIsNone(result["rows"][2]["delta_revenue_pct"])
        self.assertEqual(result["rows"][3]["driver_role"], "lost_in_current")
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[1][1],
            [
                date(2026, 3, 31),
                date(2026, 4, 6),
                date(2026, 3, 24),
                date(2026, 3, 30),
                date(2026, 3, 31),
                date(2026, 4, 6),
                date(2026, 3, 24),
                date(2026, 3, 30),
                20,
            ],
        )

    def test_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            get_sales_sku_revenue_change(limit=0)


class PriceCompetitorReportTests(unittest.TestCase):
    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_price_context_returns_unambiguous_price_layers(
        self,
        mock_get_connection,
    ):
        rows = [
            {
                "nm_id": 70194686,
                "vendor_code": "46227",
                "seller_price_regular": Decimal("8440.00"),
                "seller_controlled_price": Decimal("5739.20"),
                "customer_price_display": Decimal("3449.00"),
                "customer_price_wallet": Decimal("3380.00"),
                "customer_price_effective": Decimal("3380.00"),
                "strict_competitor_count": 1,
                "strict_competitor_price_median": Decimal("2849.00"),
                "competitor_quality": "insufficient",
                "freshness_status": "fresh",
                "price_status": "insufficient_competitor_quality",
                "seller_snapshot_date": date(2026, 7, 22),
                "analysis_run_date": date(2026, 7, 22),
                "context_snapshot_date": date(2026, 7, 22),
                "analysis_id": 42,
            }
        ]
        mock_get_connection.return_value = FakeConnection(rows=rows)

        result = get_price_context(nm_id="70194686", limit=5)

        self.assertEqual(result["contract_version"], "price-context-v1")
        self.assertEqual(
            result["source_view"],
            "analytics_ai.v_agent_price_context_latest",
        )
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["rows"][0]["seller_controlled_price"], 5739.2)
        self.assertEqual(result["rows"][0]["customer_price_effective"], 3380.0)
        self.assertEqual(
            result["rows"][0]["price_status"],
            "insufficient_competitor_quality",
        )
        self.assertEqual(result["evidence"], [])
        executed_query, params = (
            mock_get_connection.return_value.cursor_obj.executed[0]
        )
        self.assertIn("analytics_ai.v_agent_price_context_latest", executed_query)
        self.assertIn("p.nm_id = %s", executed_query)
        self.assertEqual(params, [70194686, 5])

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_price_decision_overview_exposes_context_freshness(
        self,
        mock_get_connection,
    ):
        mock_get_connection.return_value = FakeConnection(
            rows=[
                {
                    "nm_id": 70194686,
                    "snapshot_date": date(2026, 4, 24),
                    "price_context_as_of_date": date(2026, 7, 22),
                    "decision_lag_days": 89,
                    "decision_freshness": "stale",
                }
            ]
        )

        result = get_price_decision_overview("70194686", limit=1)

        self.assertEqual(result["rows"][0]["decision_freshness"], "stale")
        self.assertIn("historical only", result["freshness_rule"])
        executed_query = mock_get_connection.return_value.cursor_obj.executed[0][0]
        self.assertIn("analytics_ai.v_agent_price_context_latest", executed_query)
        self.assertIn("decision_lag_days", executed_query)

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_overview_returns_deduped_market_metrics(self, mock_get_connection):
        rows = [
            {
                "run_date": date(2026, 4, 15),
                "nm_id": 46258724,
                "vendor_code": "46203",
                "brand_name": "LIFECODE",
                "subject_name": "Спреи",
                "title": "Мист для тела",
                "source_price_promo": Decimal("500.00"),
                "source_price_regular": Decimal("500.00"),
                "source_price_segment_min": Decimal("325.00"),
                "source_price_segment_max": Decimal("675.00"),
                "competitor_count": 3,
                "cheaper_competitor_count": 1,
                "pricier_competitor_count": 2,
                "within_price_segment_count": 2,
                "same_subject_count": 2,
                "photo_ai_competitor_count": 1,
                "photo_ai_unchecked_count": 2,
                "competitor_price_min": Decimal("206.00"),
                "competitor_price_median": Decimal("566.00"),
                "competitor_price_max": Decimal("577.00"),
                "competitor_price_avg": Decimal("449.67"),
                "keyword_count": 5,
                "raw_rows": 200,
                "cheaper_competitor_share_pct": Decimal("33.33"),
                "pricier_competitor_share_pct": Decimal("66.67"),
                "source_vs_median_gap_pct": Decimal("-11.66"),
                "source_vs_median_gap_rub": Decimal("-66.00"),
                "price_status": "market_aligned",
            }
        ]
        mock_get_connection.return_value = FakeConnection(
            row={"run_date": date(2026, 4, 15)},
            rows=rows,
        )

        result = get_price_competitor_overview(product_keyword="46203", limit=10)

        self.assertEqual(result["run_date"], "2026-04-15")
        self.assertEqual(result["dedupe_key"], "run_date + source_nm_id + competitor_nm_id")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["rows"][0]["vendor_code"], "46203")
        self.assertEqual(result["rows"][0]["competitor_price_median"], 566.0)
        self.assertEqual(result["rows"][0]["price_status"], "market_aligned")
        executed_query = mock_get_connection.return_value.cursor_obj.executed[1][0]
        self.assertIn("distinct on (source_nm_id, competitor_nm_id)", executed_query)
        self.assertIn("latest_sku", executed_query)
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[1][1],
            ["%46203%", "%46203%", "%46203%", "%46203%", "%46203%", 10],
        )

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_overview_returns_empty_payload_without_raw_run(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(row={"run_date": None}, rows=[])

        result = get_price_competitor_overview()

        self.assertIsNone(result["run_date"])
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["rows"], [])
        self.assertEqual(len(mock_get_connection.return_value.cursor_obj.executed), 1)

    @patch("agi_ecom_app.services.strategic_db.get_connection")
    def test_details_filters_by_vendor_code_and_dedupes(self, mock_get_connection):
        rows = [
            {
                "run_date": date(2026, 4, 15),
                "keyword": "увлажняющий спрей",
                "source_nm_id": 46258724,
                "vendor_code": "46203",
                "source_price_promo": Decimal("500.00"),
                "competitor_nm_id": 17737159,
                "competitor_brand": "EDEN",
                "competitor_title": "Спрей для лица",
                "competitor_price_promo": Decimal("206.00"),
                "price_delta_abs": Decimal("-294.00"),
                "price_delta_percent": Decimal("-58.80"),
                "within_price_segment": False,
                "same_subject": True,
                "same_color": None,
                "photo_ai_is_competitor": None,
                "photo_ai_confidence": None,
                "competitor_rating": Decimal("4.90"),
                "competitor_feedbacks": 2530,
                "catalog_position_absolute": 1,
            }
        ]
        mock_get_connection.return_value = FakeConnection(
            row={"run_date": date(2026, 4, 15)},
            rows=rows,
        )

        result = get_price_competitor_details(vendor_code="46203", product_keyword="спрей", limit=5)

        self.assertEqual(result["run_date"], "2026-04-15")
        self.assertEqual(result["vendor_code"], "46203")
        self.assertEqual(result["rows"][0]["competitor_price_promo"], 206.0)
        executed_query = mock_get_connection.return_value.cursor_obj.executed[1][0]
        self.assertIn("row_number() over", executed_query)
        self.assertIn("partition by r.source_nm_id, r.competitor_nm_id", executed_query)
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[1][1],
            ["46203", "%спрей%", "%спрей%", "%спрей%", "%спрей%", "%спрей%", "%спрей%", "%спрей%", "%спрей%", "%спрей%", 5],
        )


if __name__ == "__main__":
    unittest.main()
