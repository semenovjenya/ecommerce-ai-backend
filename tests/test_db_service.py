import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.services.db import (
    CliDashboardKpis,
    get_cli_dashboard_kpis,
    get_finance_marketplace_report,
    get_finance_marketplace_report_text,
    get_finance_sku_profit_change_report,
    get_finance_sku_profit_change_report_text,
    get_finance_sku_report,
    get_finance_sku_report_text,
    validate_readonly_query,
)
from agi_ecom_app.sku_previews.service import get_sku_previews_payload


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


class FakeSkuPreviewCursor:
    def __init__(self, fetchall_results):
        self.fetchall_results = fetchall_results
        self.fetchall_index = 0
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        result = self.fetchall_results[self.fetchall_index]
        self.fetchall_index += 1
        return result


class FakeSkuPreviewConnection:
    def __init__(self, fetchall_results):
        self.cursor_obj = FakeSkuPreviewCursor(fetchall_results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


class ValidateReadonlyQueryTests(unittest.TestCase):
    def test_allows_select(self):
        query = validate_readonly_query("SELECT * FROM reports.wb_supplier_sales")
        self.assertEqual(query, "SELECT * FROM reports.wb_supplier_sales")

    def test_allows_cte(self):
        query = validate_readonly_query("WITH x AS (SELECT 1) SELECT * FROM x;")
        self.assertEqual(query, "WITH x AS (SELECT 1) SELECT * FROM x")

    def test_blocks_insert(self):
        with self.assertRaises(ValueError):
            validate_readonly_query("INSERT INTO foo VALUES (1)")


class CliDashboardKpisTests(unittest.TestCase):
    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_cli_dashboard_kpis_returns_expected_metrics(self, mock_get_connection):
        row = {
            "sales_date": date(2026, 3, 18),
            "prev_sales_date": date(2026, 3, 17),
            "sales_units": 179,
            "sales_rub": Decimal("76521.15"),
            "prev_sales_units": 80,
            "prev_sales_rub": Decimal("35956.21"),
            "orders_date": date(2026, 3, 19),
            "orders_today": 7,
            "stocks_snapshot_hour": datetime(2026, 3, 18, 21, 0, tzinfo=timezone.utc),
            "in_way_to_client_units": 66,
        }
        mock_get_connection.return_value = FakeConnection(row)

        result = get_cli_dashboard_kpis()

        self.assertEqual(result.sales_units, 179)
        self.assertEqual(result.sales_rub, 76521.15)
        self.assertEqual(result.orders_today, 7)
        self.assertEqual(result.in_way_to_client_units, 66)
        self.assertEqual(result.sales_units_delta_pct, 123.75)
        self.assertEqual(result.sales_rub_delta_pct, 112.82)
        self.assertEqual(result.sales_date, date(2026, 3, 18))
        self.assertEqual(result.prev_sales_date, date(2026, 3, 17))
        self.assertEqual(result.orders_date, date(2026, 3, 19))
        self.assertFalse(result.is_partial)
        self.assertIsNone(result.warning)

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_cli_dashboard_kpis_marks_missing_sources_as_partial(self, mock_get_connection):
        row = {
            "sales_date": None,
            "prev_sales_date": None,
            "sales_units": 0,
            "sales_rub": None,
            "prev_sales_units": 0,
            "prev_sales_rub": None,
            "orders_date": date(2026, 3, 19),
            "orders_today": 7,
            "stocks_snapshot_hour": None,
            "in_way_to_client_units": None,
        }
        mock_get_connection.return_value = FakeConnection(row)

        result = get_cli_dashboard_kpis()

        self.assertIsNone(result.sales_units)
        self.assertIsNone(result.sales_rub)
        self.assertEqual(result.orders_today, 7)
        self.assertIsNone(result.in_way_to_client_units)
        self.assertIsNone(result.sales_units_delta_pct)
        self.assertIsNone(result.sales_rub_delta_pct)
        self.assertTrue(result.is_partial)
        self.assertEqual(result.warning, "Часть KPI недоступна: продажи, остатки")

    def test_unavailable_factory_returns_safe_fallback(self):
        result = CliDashboardKpis.unavailable()

        self.assertTrue(result.is_partial)
        self.assertEqual(result.warning, "Данные KPI временно недоступны")
        self.assertIsNone(result.sales_units)
        self.assertIsNone(result.orders_today)


class FinanceMarketplaceReportTests(unittest.TestCase):
    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_marketplace_report_returns_expected_breakdown(self, mock_get_connection):
        row = {
            "revenue": Decimal("1000.00"),
            "returns": Decimal("100.00"),
            "sales_cogs": Decimal("400.00"),
            "return_cogs": Decimal("50.00"),
            "net_cogs": Decimal("350.00"),
            "total_marketplace_expenses": Decimal("300.00"),
            "commission_and_acquiring": Decimal("200.00"),
            "marketing": Decimal("30.00"),
            "logistics": Decimal("40.00"),
            "acceptance": Decimal("5.00"),
            "storage_fee": Decimal("10.00"),
            "cross_docking": Decimal("5.00"),
            "penalties_and_withholdings": Decimal("10.00"),
            "misc_accruals": Decimal("20.00"),
            "total_to_pay": Decimal("620.00"),
            "margin_rub": Decimal("270.00"),
            "margin_percent": Decimal("30.00"),
            "missing_cost_row_count": 2,
            "missing_cost_sales_quantity": Decimal("3"),
            "missing_cost_return_quantity": Decimal("1"),
        }
        mock_get_connection.return_value = FakeConnection(row=row)

        result = get_finance_marketplace_report("2026-02-23", "2026-03-01")

        self.assertEqual(result["date_from"], "2026-02-23")
        self.assertEqual(result["date_to"], "2026-03-01")
        self.assertEqual(result["source_table"], "finance.wb_report_detail_daily")
        self.assertEqual(result["metrics"][0], {"label": "ИТОГО К ОПЛАТЕ", "value": 620.0})
        self.assertEqual(result["metrics"][1], {"label": "Выручка", "value": 1000.0})
        self.assertEqual(result["metrics"][2], {"label": "Возвраты", "value": 100.0})
        self.assertEqual(result["metrics"][3], {"label": "Себестоимость продаж", "value": 400.0})
        self.assertEqual(result["metrics"][4], {"label": "Себестоимость возвратов", "value": 50.0})
        self.assertEqual(result["metrics"][5], {"label": "Себестоимость нетто", "value": 350.0})
        self.assertEqual(result["metrics"][-1], {"label": "МАРЖА", "value": 270.0, "percent": 30.0})
        self.assertEqual(result["totals"]["missing_cost_row_count"], 2)
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[0][1],
            [date(2026, 2, 23), date(2026, 3, 1), date(2026, 3, 1), date(2026, 3, 1)],
        )

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_marketplace_report_accepts_compact_period_string(self, mock_get_connection):
        row = {
            "revenue": Decimal("1"),
            "returns": Decimal("2"),
            "sales_cogs": Decimal("3"),
            "return_cogs": Decimal("4"),
            "net_cogs": Decimal("-1"),
            "total_marketplace_expenses": Decimal("5"),
            "commission_and_acquiring": Decimal("6"),
            "marketing": Decimal("7"),
            "logistics": Decimal("8"),
            "acceptance": Decimal("9"),
            "storage_fee": Decimal("10"),
            "cross_docking": Decimal("11"),
            "penalties_and_withholdings": Decimal("12"),
            "misc_accruals": Decimal("13"),
            "total_to_pay": Decimal("14"),
            "margin_rub": Decimal("15"),
            "margin_percent": None,
            "missing_cost_row_count": 0,
            "missing_cost_sales_quantity": Decimal("0"),
            "missing_cost_return_quantity": Decimal("0"),
        }
        mock_get_connection.return_value = FakeConnection(row=row)

        result = get_finance_marketplace_report("23.02-01.03")

        self.assertEqual(result["date_from"], "2026-02-23")
        self.assertEqual(result["date_to"], "2026-03-01")
        self.assertEqual(
            mock_get_connection.return_value.cursor_obj.executed[0][1],
            [date(2026, 2, 23), date(2026, 3, 1), date(2026, 3, 1), date(2026, 3, 1)],
        )
        self.assertEqual(result["metrics"][-1], {"label": "МАРЖА", "value": 15.0})

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_marketplace_report_sql_joins_cost_with_version_fallback(self, mock_get_connection):
        row = {
            "revenue": Decimal("0"),
            "returns": Decimal("0"),
            "sales_cogs": Decimal("0"),
            "return_cogs": Decimal("0"),
            "net_cogs": Decimal("0"),
            "total_marketplace_expenses": Decimal("0"),
            "commission_and_acquiring": Decimal("0"),
            "marketing": Decimal("0"),
            "logistics": Decimal("0"),
            "acceptance": Decimal("0"),
            "storage_fee": Decimal("0"),
            "cross_docking": Decimal("0"),
            "penalties_and_withholdings": Decimal("0"),
            "misc_accruals": Decimal("0"),
            "total_to_pay": Decimal("0"),
            "margin_rub": Decimal("0"),
            "margin_percent": None,
            "missing_cost_row_count": 0,
            "missing_cost_sales_quantity": Decimal("0"),
            "missing_cost_return_quantity": Decimal("0"),
        }
        mock_get_connection.return_value = FakeConnection(row=row)

        get_finance_marketplace_report("2026-02-23", "2026-03-01")

        query = mock_get_connection.return_value.cursor_obj.executed[0][0]
        self.assertIn("from finance.wb_product_coast c", query)
        self.assertIn("r.nm_id is not null and c.nm_id = r.nm_id", query)
        self.assertIn("c.vendor_code = r.vendor_code", query)
        self.assertIn("case when r.nm_id is not null and c.nm_id = r.nm_id then 0 else 1 end", query)
        self.assertIn("(c.upload_date <= %s) desc", query)
        self.assertIn("retail_price_withdisc_rub <> 0", query)

    @patch("agi_ecom_app.services.db.get_finance_marketplace_report")
    def test_get_finance_marketplace_report_text_formats_money_lines(self, mock_get_report):
        mock_get_report.return_value = {
            "date_from": "2026-02-23",
            "date_to": "2026-03-01",
            "source_table": "finance.wb_report_detail_daily",
            "metrics": [
                {"label": "ИТОГО К ОПЛАТЕ", "value": 620.0},
                {"label": "Выручка", "value": 1000.0},
                {"label": "Возвраты", "value": 100.0},
                {"label": "Себестоимость продаж", "value": 400.0},
                {"label": "Себестоимость возвратов", "value": 50.0},
                {"label": "Себестоимость нетто", "value": 350.0},
                {"label": "ИТОГО РАСХОДЫ МП", "value": 300.0},
                {"label": "КОМИССИЯ и ЭКВАЙРИНГ", "value": 200.0},
                {"label": "МАРЖА", "value": 270.0, "percent": 30.0},
            ],
            "totals": {"missing_cost_row_count": 2},
        }

        text = get_finance_marketplace_report_text("23.02.2026", "01.03.2026")

        self.assertIn("Период: 23.02–01.03 (2026-02-23 - 2026-03-01)", text)
        self.assertIn("Источник: finance.wb_report_detail_daily", text)
        self.assertIn("ИТОГО К ОПЛАТЕ: 620.00 ₽", text)
        self.assertIn("Возвраты: 100.00 ₽", text)
        self.assertIn("Себестоимость нетто: 350.00 ₽", text)
        self.assertIn("МАРЖА: 270.00 ₽ (30.00%)", text)
        self.assertIn("Пробел в данных: себестоимость не найдена для 2 строк", text)

        metric_lines = [
            line
            for line in text.splitlines()
            if ": " in line and not line.startswith(("Период", "Источник", "Пробел"))
        ]
        self.assertEqual(
            metric_lines,
            [
                "ИТОГО К ОПЛАТЕ: 620.00 ₽",
                "Выручка: 1 000.00 ₽",
                "Возвраты: 100.00 ₽",
                "Себестоимость продаж: 400.00 ₽",
                "Себестоимость возвратов: 50.00 ₽",
                "Себестоимость нетто: 350.00 ₽",
                "ИТОГО РАСХОДЫ МП: 300.00 ₽",
                "КОМИССИЯ и ЭКВАЙРИНГ: 200.00 ₽",
                "МАРЖА: 270.00 ₽ (30.00%)",
            ],
        )

    @patch("agi_ecom_app.services.db.get_finance_marketplace_report")
    def test_get_finance_marketplace_report_text_keeps_full_metric_labels(self, mock_get_report):
        mock_get_report.return_value = {
            "date_from": "2026-02-23",
            "date_to": "2026-03-01",
            "source_table": "finance.wb_report_detail_daily",
            "metrics": [
                {"label": "Приемка", "value": 0},
                {"label": "Кросс-докинг", "value": 3322.43},
                {"label": "ИТОГО РАСХОДЫ МП", "value": 355445.20},
            ],
            "totals": {},
        }

        text = get_finance_marketplace_report_text("с 23.02 по 01.03")

        self.assertIn("Приемка: 0.00 ₽", text)
        self.assertIn("Кросс-докинг: 3 322.43 ₽", text)
        self.assertIn("ИТОГО РАСХОДЫ МП: 355 445.20 ₽", text)


class FinanceSkuReportTests(unittest.TestCase):
    def _sku_row(
        self,
        *,
        nm_id=70195339,
        vendor_code="46230",
        unit_cost_rub=Decimal("100.00"),
        delivered_qty=Decimal("10"),
        net_revenue=Decimal("900.00"),
        return_qty=Decimal("1"),
        returns_rub=Decimal("100.00"),
        total_marketplace_expenses=Decimal("220.00"),
        commission_and_acquiring=Decimal("150.00"),
        marketing=Decimal("20.00"),
        logistics=Decimal("30.00"),
        return_cogs=Decimal("100.00"),
        cost_share_before_returns_pct=Decimal("100.00"),
        abc_revenue_class="A",
        margin_rub=Decimal("-220.00"),
        profitability_pct=Decimal("-24.44"),
        missing_cost=False,
        total_sku_count=3,
        missing_cost_sku_count=1,
    ):
        return {
            "nm_id": nm_id,
            "vendor_code": vendor_code,
            "unit_cost_rub": unit_cost_rub,
            "delivered_qty": delivered_qty,
            "net_revenue": net_revenue,
            "return_qty": return_qty,
            "returns_rub": returns_rub,
            "total_marketplace_expenses": total_marketplace_expenses,
            "commission_and_acquiring": commission_and_acquiring,
            "marketing": marketing,
            "logistics": logistics,
            "return_cogs": return_cogs,
            "cost_share_before_returns_pct": cost_share_before_returns_pct,
            "abc_revenue_class": abc_revenue_class,
            "margin_rub": margin_rub,
            "profitability_pct": profitability_pct,
            "missing_cost": missing_cost,
            "total_sku_count": total_sku_count,
            "missing_cost_sku_count": missing_cost_sku_count,
        }

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_sku_report_returns_expected_columns_and_rows(self, mock_get_connection):
        rows = [
            self._sku_row(),
            self._sku_row(
                nm_id=70195340,
                vendor_code="A-100",
                unit_cost_rub=None,
                net_revenue=Decimal("200.00"),
                abc_revenue_class="B",
                missing_cost=True,
                margin_rub=Decimal("120.00"),
                profitability_pct=Decimal("60.00"),
            ),
            self._sku_row(
                nm_id=70195341,
                vendor_code="C-100",
                net_revenue=Decimal("50.00"),
                abc_revenue_class="C",
                margin_rub=Decimal("10.00"),
                profitability_pct=Decimal("20.00"),
            ),
        ]
        mock_get_connection.return_value = FakeConnection(rows=rows)

        result = get_finance_sku_report(
            "2026-02-23",
            "2026-03-01",
            product_keyword="46230",
            limit=10,
        )

        self.assertEqual(result["date_from"], "2026-02-23")
        self.assertEqual(result["date_to"], "2026-03-01")
        self.assertEqual(result["product_keyword"], "46230")
        self.assertEqual(result["sort_by"], "net_revenue")
        self.assertEqual(
            result["source_tables"],
            ["finance.wb_report_detail_daily", "finance.wb_product_coast"],
        )
        self.assertEqual(
            result["columns"],
            [
                "nm_id",
                "Артикул",
                "Себестоимость",
                "Доставлено, шт",
                "Выручка, руб (после возвратов)",
                "Возвраты, шт",
                "Возвраты, руб",
                "ИТОГО Расходы МП на артикул",
                "КОМИССИИ и ЭКВАЙРИНГ",
                "МАРКЕТИНГ",
                "ЛОГИСТИКА",
                "Себестоимость возвратов",
                "Доля себестоимости в выручке до возвратов",
                "ABC по выручке",
                "Маржа, руб",
                "Рентабельность, %",
            ],
        )
        self.assertEqual(result["rows"][0]["nm_id"], 70195339)
        self.assertEqual(result["rows"][0]["Артикул"], "46230")
        self.assertEqual(result["rows"][0]["Себестоимость"], 100.0)
        self.assertEqual(result["rows"][0]["Доставлено, шт"], 10.0)
        self.assertEqual(result["rows"][0]["Возвраты, шт"], 1.0)
        self.assertEqual(result["rows"][0]["ABC по выручке"], "A")
        self.assertEqual(result["rows"][1]["Себестоимость"], None)
        self.assertEqual(result["totals"], {
            "total_sku_count": 3,
            "returned_sku_count": 3,
            "missing_cost_sku_count": 1,
        })

        executed_query, params = mock_get_connection.return_value.cursor_obj.executed[0]
        self.assertIn("from finance.wb_report_detail_daily", executed_query)
        self.assertIn("from finance.wb_product_coast c", executed_query)
        self.assertIn("case when p.nm_id is not null and c.nm_id = p.nm_id then 0 else 1 end", executed_query)
        self.assertIn("ABC", "".join(result["columns"]))
        self.assertIn("count(*) over () as total_sku_count", executed_query)
        self.assertIn("rows between unbounded preceding and current row", executed_query)
        self.assertIn("order by net_revenue desc", executed_query)
        self.assertEqual(
            params,
            [
                date(2026, 2, 23),
                date(2026, 3, 1),
                "%46230%",
                "%46230%",
                "%46230%",
                "%46230%",
                date(2026, 3, 1),
                date(2026, 3, 1),
                10,
            ],
        )

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_sku_report_sql_contains_expected_finance_formulas(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(rows=[])

        get_finance_sku_report("2026-02-23", "2026-03-01")

        query = mock_get_connection.return_value.cursor_obj.executed[0][0]
        self.assertIn("where doc_type_name = 'Продажа'", query)
        self.assertIn("where doc_type_name = 'Возврат'", query)
        self.assertIn("abs(coalesce(sum(quantity) filter", query)
        self.assertIn("(revenue_before_returns - returns_rub) as net_revenue", query)
        self.assertIn("cost_price_rub * return_qty", query)
        self.assertIn("100 * sales_cogs / revenue_before_returns", query)
        self.assertIn("100 * margin_rub / net_revenue", query)
        self.assertIn("(c.upload_date <= %s) desc", query)

    @patch("agi_ecom_app.services.db.get_connection")
    def test_get_finance_sku_report_can_sort_by_margin(self, mock_get_connection):
        mock_get_connection.return_value = FakeConnection(rows=[])

        result = get_finance_sku_report(
            "2026-02-23",
            "2026-03-01",
            sort_by="margin_rub",
            limit=1,
        )

        query = mock_get_connection.return_value.cursor_obj.executed[0][0]
        self.assertEqual(result["sort_by"], "margin_rub")
        self.assertIn("order by margin_rub desc", query)

    @patch("agi_ecom_app.services.db.get_finance_sku_report")
    def test_get_finance_sku_profit_change_report_compares_periods(self, mock_get_report):
        current_report = {
            "rows": [
                {"nm_id": 1, "Артикул": "A", "Маржа, руб": 150.0, "Выручка, руб (после возвратов)": 300.0, "Рентабельность, %": 50.0},
                {"nm_id": 2, "Артикул": "B", "Маржа, руб": 20.0, "Выручка, руб (после возвратов)": 100.0, "Рентабельность, %": 20.0},
            ],
            "totals": {"total_sku_count": 2, "missing_cost_sku_count": 0},
        }
        previous_report = {
            "rows": [
                {"nm_id": 1, "Артикул": "A", "Маржа, руб": 100.0, "Выручка, руб (после возвратов)": 250.0, "Рентабельность, %": 40.0},
                {"nm_id": 3, "Артикул": "C", "Маржа, руб": 80.0, "Выручка, руб (после возвратов)": 160.0, "Рентабельность, %": 50.0},
            ],
            "totals": {"total_sku_count": 2, "missing_cost_sku_count": 1},
        }
        mock_get_report.side_effect = [current_report, previous_report]

        result = get_finance_sku_profit_change_report(
            current_date_from="2026-04-08",
            current_date_to="2026-04-14",
            limit=10,
        )

        self.assertEqual(result["current_period_start"], "2026-04-08")
        self.assertEqual(result["previous_period_start"], "2026-04-01")
        self.assertEqual(result["previous_period_end"], "2026-04-07")
        self.assertEqual(result["rows"][0]["Артикул"], "C")
        self.assertEqual(result["rows"][0]["Δ маржи, руб"], -80.0)
        self.assertEqual(result["rows"][0]["Роль"], "lost_in_current")
        self.assertEqual(result["rows"][1]["Артикул"], "A")
        self.assertEqual(result["rows"][1]["Δ маржи, руб"], 50.0)
        self.assertEqual(result["totals"]["missing_cost_sku_count"], 1)

    @patch("agi_ecom_app.services.db.get_finance_sku_profit_change_report")
    def test_get_finance_sku_profit_change_report_text_formats_table(self, mock_get_report):
        mock_get_report.return_value = {
            "current_period_start": "2026-04-08",
            "current_period_end": "2026-04-14",
            "previous_period_start": "2026-04-01",
            "previous_period_end": "2026-04-07",
            "product_keyword": None,
            "source_tables": ["finance.wb_report_detail_daily", "finance.wb_product_coast"],
            "columns": ["nm_id", "Артикул", "Δ маржи, руб"],
            "rows": [{"nm_id": 1, "Артикул": "A", "Δ маржи, руб": 50.0}],
            "totals": {
                "current_total_sku_count": 2,
                "previous_total_sku_count": 2,
                "returned_sku_count": 1,
                "missing_cost_sku_count": 0,
            },
        }

        text = get_finance_sku_profit_change_report_text("2026-04-08", "2026-04-14")

        self.assertIn("2026-04-08 - 2026-04-14 vs 2026-04-01 - 2026-04-07", text)
        self.assertIn("| nm_id | Артикул | Δ маржи, руб |", text)
        self.assertIn("| 1 | A | 50.00 |", text)

    @patch("agi_ecom_app.services.db.get_finance_sku_report")
    def test_get_finance_sku_report_text_formats_markdown_table(self, mock_get_report):
        mock_get_report.return_value = {
            "date_from": "2026-02-23",
            "date_to": "2026-03-01",
            "product_keyword": "46230",
            "source_tables": ["finance.wb_report_detail_daily", "finance.wb_product_coast"],
            "columns": [
                "nm_id",
                "Артикул",
                "Себестоимость",
                "Доставлено, шт",
                "Выручка, руб (после возвратов)",
                "Возвраты, шт",
                "Возвраты, руб",
                "ИТОГО Расходы МП на артикул",
                "КОМИССИИ и ЭКВАЙРИНГ",
                "МАРКЕТИНГ",
                "ЛОГИСТИКА",
                "Себестоимость возвратов",
                "Доля себестоимости в выручке до возвратов",
                "ABC по выручке",
                "Маржа, руб",
                "Рентабельность, %",
            ],
            "rows": [
                {
                    "nm_id": 70195339,
                    "Артикул": "46230",
                    "Себестоимость": 100.0,
                    "Доставлено, шт": 10.0,
                    "Выручка, руб (после возвратов)": 900.0,
                    "Возвраты, шт": 1.0,
                    "Возвраты, руб": 100.0,
                    "ИТОГО Расходы МП на артикул": 220.0,
                    "КОМИССИИ и ЭКВАЙРИНГ": 150.0,
                    "МАРКЕТИНГ": 20.0,
                    "ЛОГИСТИКА": 30.0,
                    "Себестоимость возвратов": 100.0,
                    "Доля себестоимости в выручке до возвратов": 100.0,
                    "ABC по выручке": "A",
                    "Маржа, руб": -220.0,
                    "Рентабельность, %": -24.44,
                }
            ],
            "totals": {
                "total_sku_count": 3,
                "returned_sku_count": 1,
                "missing_cost_sku_count": 1,
            },
        }

        text = get_finance_sku_report_text("23.02.2026", "01.03.2026", "46230")

        self.assertIn("Период: 23.02–01.03 (2026-02-23 - 2026-03-01)", text)
        self.assertIn("Фильтр: 46230", text)
        self.assertIn("SKU в выборке: 1 из 3", text)
        self.assertIn("Пробел в данных: себестоимость не найдена для 1 SKU.", text)
        self.assertIn("| nm_id | Артикул | Себестоимость |", text)
        self.assertIn("| 70195339 | 46230 | 100.00 |", text)
        self.assertIn("-24.44%", text)


class SkuPreviewsTests(unittest.TestCase):
    @patch("agi_ecom_app.sku_previews.service.get_settings")
    @patch("agi_ecom_app.sku_previews.service.get_connection")
    def test_get_sku_previews_payload_reads_configured_table(self, mock_get_connection, mock_get_settings):
        mock_get_settings.return_value = SimpleNamespace(
            sku_preview=SimpleNamespace(
                schema="analitics_Ai",
                table="wb sku ai analis",
                sku_column="vendor_code",
                wb_nm_id_column="nm_id",
                title_column="title",
                image_url_column="photo_url",
                product_url_column="product_url",
            )
        )
        connection = FakeSkuPreviewConnection(
            fetchall_results=[
                [
                    {"column_name": "vendor_code"},
                    {"column_name": "nm_id"},
                    {"column_name": "name"},
                    {"column_name": "main_photo"},
                ],
                [
                    {
                        "sku": "46303",
                        "wb_nm_id": "70195339",
                        "title": "Название товара",
                        "image_url": "https://example.test/photo.jpg",
                        "product_url": None,
                    }
                ],
            ]
        )
        mock_get_connection.return_value = connection

        result = get_sku_previews_payload("46303, 46303,81558446")

        self.assertEqual(result.items["46303"].title, "Название товара")
        self.assertEqual(result.items["46303"].imageUrl, "https://example.test/photo.jpg")
        self.assertEqual(
            result.items["46303"].productUrl,
            "https://www.wildberries.ru/catalog/70195339/detail.aspx",
        )
        self.assertEqual(connection.cursor_obj.executed[0][1], ["analitics_Ai", "wb sku ai analis"])
        self.assertEqual(connection.cursor_obj.executed[1][1], [["46303", "81558446"]])

    def test_get_sku_previews_payload_rejects_invalid_ids(self):
        with self.assertRaises(ValueError):
            get_sku_previews_payload("70195339, bad id")


if __name__ == "__main__":
    unittest.main()
