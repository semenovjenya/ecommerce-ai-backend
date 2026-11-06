import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.unit_economics.models import (
    FinanceReportOption,
    FinanceReportOptionsResponse,
    UnitEconomicsColumn,
    UnitEconomicsPeriod,
    UnitEconomicsResponse,
    UnitEconomicsRow,
    UnitEconomicsTotals,
)
from agi_ecom_app.unit_economics.service import (
    get_finance_report_options_payload,
    get_unit_economics_payload,
)


class FakeUnitEconomicsCursor:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows


class FakeUnitEconomicsConnection:
    def __init__(self, rows=None, error=None):
        self.cursor_obj = FakeUnitEconomicsCursor(rows=rows, error=error)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


def _profit_calculator_rows():
    return [
        {
            "scenario_id": 10,
            "scenario_name": "46227",
            "nm_id": 70194686,
            "vendor_code": "46227",
            "brand_name": "LIFECODE",
            "subject_name": "Сыворотки",
            "price_after_discount_rub": 4944.50,
            "cost_price_rub": 344.0,
            "kvv_pct": 32.5,
            "kvv_rub": 1606.96,
            "expected_outbound_per_buyout_rub": 101.20,
            "expected_return_logistics_rub": 0.0,
            "storage_total_rub": 4.03,
            "ad_rub": 0.0,
            "other_expenses_rub": 0.0,
            "tax_rub": 0.0,
            "net_profit_rub": 3232.31,
            "margin_pct": 65.37,
            "calculation_status": "ok",
            "total_row_count": 38,
        }
    ]


class Feature4UnitEconomicsServiceTests(unittest.TestCase):
    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_maps_profit_calculator_results_for_frontend_table(
        self,
        mock_get_connection,
    ):
        connection = FakeUnitEconomicsConnection(rows=_profit_calculator_rows())
        mock_get_connection.return_value = connection

        response = get_unit_economics_payload(
            date_from="2026-02-23",
            date_to="2026-03-01",
            limit=100,
        )

        self.assertEqual(response.report.reportId, "unit_economics")
        self.assertEqual(response.report.label, "Юнит-экономика")
        self.assertFalse(response.period.isDefaultPeriod)
        self.assertEqual(
            response.period.label,
            "Актуальный расчет из finance.wb_profit_calculator_results "
            "+ реклама за 7 дней из finance.wb_report_detail_daily",
        )
        self.assertEqual(response.columns[0].key, "nmId")
        self.assertEqual(response.columns[0].type, "text")
        self.assertEqual(response.columns[3].key, "subjectName")
        self.assertEqual(response.columns[3].align, "left")
        self.assertEqual(response.rows[0].rowId, "unit_10_70194686")
        self.assertEqual(response.rows[0].nmId, "70194686")
        self.assertEqual(response.rows[0].vendorCode, "46227")
        self.assertEqual(response.rows[0].brandName, "LIFECODE")
        self.assertEqual(response.rows[0].subjectName, "Сыворотки")
        self.assertEqual(response.rows[0].salePriceRub, 4944.5)
        self.assertEqual(response.rows[0].unitCostRub, 344)
        self.assertEqual(response.rows[0].commissionPct, 32.5)
        self.assertEqual(response.rows[0].commissionRub, 1606.96)
        self.assertEqual(response.rows[0].logisticsRub, 101.2)
        self.assertEqual(response.rows[0].returnLogisticsRub, 0)
        self.assertEqual(response.rows[0].storageRub, 4.03)
        self.assertEqual(response.rows[0].adRub, 0)
        self.assertEqual(response.rows[0].otherExpensesRub, 0)
        self.assertEqual(response.rows[0].taxRub, 0)
        self.assertEqual(response.rows[0].profitRub, 3232.31)
        self.assertEqual(response.rows[0].marginPct, 65.37)
        self.assertEqual(response.totals.totalRowCount, 38)
        self.assertEqual(response.totals.returnedRowCount, 1)
        self.assertEqual(response.totals.sourceTable, "finance.wb_profit_calculator_results")

        executed_query, params = connection.cursor_obj.executed[0]
        self.assertIn("from finance.wb_profit_calculator_results", executed_query)
        self.assertIn("from finance.wb_product_coast c", executed_query)
        self.assertIn("from finance.wb_report_detail_daily r", executed_query)
        self.assertIn("sum(r.deduction) filter", executed_query)
        self.assertIn("and r.supplier_oper_name = 'Удержание'", executed_query)
        self.assertIn("substring(r.bonus_type_name from 'товар\\s+([0-9]+)')", executed_query)
        self.assertIn("r.bonus_type_name ilike '%%WB Продвижение%%'", executed_query)
        self.assertIn("r.bonus_type_name ilike '%%Списание за отзыв%%'", executed_query)
        self.assertIn("totals.unlinked_marketing_sum", executed_query)
        self.assertIn("* fs.revenue_sum / totals.total_revenue_sum", executed_query)
        self.assertIn("then round((ads.marketing_sum / ads.delivered_qty)::numeric, 2)", executed_query)
        self.assertIn("else p.ad_rub", executed_query)
        self.assertIn("coalesce(nullif(p.cost_price_rub, 0), cost.cost_price_rub)", executed_query)
        self.assertEqual(params, ["2026-02-23", "2026-03-01", 100])

    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_maps_ad_rub_from_finance_report_per_unit(
        self,
        mock_get_connection,
    ):
        rows = _profit_calculator_rows()
        rows[0]["ad_rub"] = 123.45
        mock_get_connection.return_value = FakeUnitEconomicsConnection(rows=rows)

        response = get_unit_economics_payload()

        self.assertEqual(response.rows[0].adRub, 123.45)

    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_keeps_profit_calculator_ad_rub_as_sql_fallback(
        self,
        mock_get_connection,
    ):
        rows = _profit_calculator_rows()
        rows[0]["ad_rub"] = 77
        connection = FakeUnitEconomicsConnection(rows=rows)
        mock_get_connection.return_value = connection

        response = get_unit_economics_payload()

        executed_query, params = connection.cursor_obj.executed[0]
        self.assertIn("else p.ad_rub", executed_query)
        self.assertEqual(response.rows[0].adRub, 77)
        self.assertEqual(params, [None, None, 100])

    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_uses_db_result_period_label_by_default(
        self,
        mock_get_connection,
    ):
        mock_get_connection.return_value = FakeUnitEconomicsConnection(rows=_profit_calculator_rows())

        response = get_unit_economics_payload()

        self.assertTrue(response.period.isDefaultPeriod)
        self.assertIsNone(response.period.dateFrom)
        self.assertIsNone(response.period.dateTo)
        self.assertEqual(
            response.period.label,
            "Актуальный расчет из finance.wb_profit_calculator_results "
            "+ реклама за 7 дней из finance.wb_report_detail_daily",
        )

    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_filters_by_product_keyword(self, mock_get_connection):
        connection = FakeUnitEconomicsConnection(rows=_profit_calculator_rows())
        mock_get_connection.return_value = connection

        get_unit_economics_payload(product_keyword="LIFECODE", limit=50)

        executed_query, params = connection.cursor_obj.executed[0]
        self.assertIn("coalesce(p.brand_name, '') ilike", executed_query)
        self.assertEqual(params, [None, None] + ["%LIFECODE%"] * 5 + [50])

    def test_unit_economics_payload_requires_supported_limit(self):
        with self.assertRaisesRegex(ValueError, "limit must be between 1 and 500"):
            get_unit_economics_payload(limit=0)

    def test_unit_economics_payload_requires_date_from_when_date_to_is_provided(self):
        with self.assertRaisesRegex(ValueError, "date_from is required"):
            get_unit_economics_payload(date_to="2026-03-01")

    @patch("agi_ecom_app.unit_economics.service.db.get_connection")
    def test_unit_economics_payload_wraps_profit_calculator_db_errors(
        self,
        mock_get_connection,
    ):
        mock_get_connection.return_value = FakeUnitEconomicsConnection(error=RuntimeError("missing table"))

        with self.assertRaisesRegex(RuntimeError, "failed to load finance.wb_profit_calculator_results"):
            get_unit_economics_payload()

    def test_finance_report_options_opens_unit_economics_from_unit_tab(self):
        response = get_finance_report_options_payload()

        self.assertEqual(response.items[0].reportId, "unit_economics")
        self.assertEqual(response.items[0].label, "Юнит-экономика")
        self.assertEqual(response.items[0].sortOrder, 1)


class Feature4UnitEconomicsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.unit_economics.router.get_unit_economics_payload")
    def test_unit_economics_endpoint_returns_json_payload(self, mock_get_payload):
        mock_get_payload.return_value = UnitEconomicsResponse(
            report=FinanceReportOption(
                reportId="unit_economics",
                label="Юнит-экономика",
                sortOrder=1,
            ),
            period=UnitEconomicsPeriod(
                dateFrom="2026-02-23",
                dateTo="2026-03-01",
                label=(
                    "Актуальный расчет из finance.wb_profit_calculator_results "
                    "+ реклама за 7 дней из finance.wb_report_detail_daily"
                ),
                isDefaultPeriod=False,
            ),
            columns=[
                UnitEconomicsColumn(key="nmId", label="NM_ID", type="text"),
                UnitEconomicsColumn(key="subjectName", label="ПРЕДМЕТ", type="text", align="left"),
            ],
            rows=[
                UnitEconomicsRow(
                    rowId="unit_10_70194686",
                    nmId="70194686",
                    vendorCode="46227",
                    brandName="LIFECODE",
                    subjectName="Сыворотки",
                    salePriceRub=4944.5,
                    unitCostRub=344,
                    commissionPct=32.5,
                    commissionRub=1606.96,
                    logisticsRub=101.2,
                    returnLogisticsRub=0,
                    storageRub=4.03,
                    adRub=0,
                    otherExpensesRub=0,
                    taxRub=0,
                    profitRub=3232.31,
                    marginPct=65.37,
                )
            ],
            totals=UnitEconomicsTotals(
                totalRowCount=38,
                returnedRowCount=1,
                sourceTable="finance.wb_profit_calculator_results",
            ),
        )

        response = self.client.get(
            "/v1/finance/unit-economics",
            params={"date_from": "2026-02-23", "date_to": "2026-03-01", "limit": 100},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"]["reportId"], "unit_economics")
        self.assertEqual(response.json()["report"]["label"], "Юнит-экономика")
        self.assertEqual(response.json()["columns"][0]["key"], "nmId")
        self.assertEqual(response.json()["columns"][0]["type"], "text")
        self.assertEqual(response.json()["columns"][1]["align"], "left")
        self.assertEqual(response.json()["rows"][0]["rowId"], "unit_10_70194686")
        self.assertEqual(response.json()["rows"][0]["nmId"], "70194686")
        self.assertNotIn("scenarioId", response.json()["rows"][0])
        self.assertNotIn("scenarioName", response.json()["rows"][0])
        self.assertNotIn("calculationStatus", response.json()["rows"][0])
        self.assertEqual(response.json()["rows"][0]["commissionPct"], 32.5)
        self.assertEqual(response.json()["rows"][0]["commissionRub"], 1606.96)
        self.assertEqual(response.json()["rows"][0]["profitRub"], 3232.31)
        self.assertEqual(response.json()["rows"][0]["marginPct"], 65.37)
        mock_get_payload.assert_called_once_with(
            date_from="2026-02-23",
            date_to="2026-03-01",
            product_keyword=None,
            limit=100,
        )

    @patch("agi_ecom_app.unit_economics.router.get_unit_economics_payload")
    def test_unit_economics_endpoint_returns_503_for_profit_calculator_db_errors(
        self,
        mock_get_payload,
    ):
        mock_get_payload.side_effect = RuntimeError("failed to load finance.wb_profit_calculator_results")

        response = self.client.get("/v1/finance/unit-economics")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "failed to load finance.wb_profit_calculator_results")

    @patch("agi_ecom_app.unit_economics.router.get_finance_report_options_payload")
    def test_report_options_endpoint_returns_unit_economics_for_unit_tab(
        self,
        mock_get_payload,
    ):
        mock_get_payload.return_value = FinanceReportOptionsResponse(
            items=[
                FinanceReportOption(
                    reportId="unit_economics",
                    label="Юнит-экономика",
                    sortOrder=1,
                )
            ]
        )

        response = self.client.get("/v1/finance/report-options")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["reportId"], "unit_economics")
        self.assertEqual(response.json()["items"][0]["label"], "Юнит-экономика")
        self.assertEqual(response.json()["items"][0]["sortOrder"], 1)


if __name__ == "__main__":
    unittest.main()
