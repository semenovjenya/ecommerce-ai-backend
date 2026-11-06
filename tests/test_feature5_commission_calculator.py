import unittest
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.commission_calculator.models import (
    CommissionCalculatorColumn,
    CommissionCalculatorPeriod,
    CommissionCalculatorReport,
    CommissionCalculatorResponse,
    CommissionCalculatorRow,
    CommissionCalculatorTotals,
)
from agi_ecom_app.commission_calculator.service import get_commission_calculator_payload


def _sku_context():
    return {
        "product_name": "Сыворотка для лица",
        "vendor_code": "46 227",
        "brand_name": "LIFECODE",
        "subject_name": "Сыворотки",
        "price_before_discount": 1990.0,
        "price_after_discount": 1490.0,
        "seller_discount_pct": 25.13,
        "discount_pct": 25.13,
        "club_discount_pct": 0.0,
        "price_with_wb_club": 1490.0,
        "spp_pct": 12.0,
        "price_with_spp": 1311.2,
        "buyout_percent": 88.4,
        "cost_price_rub": 344.0,
        "calculator_tax_pct": 0.0,
        "calculator_tax_rub": 0.0,
    }


def _sku_finance():
    return {
        "vendor_code": "46 227",
        "brand_name": "LIFECODE",
        "subject_name": "Сыворотки",
        "delivered_qty": 74.0,
        "revenue_sum": 366106.44,
        "commission_sum": 113631.96,
        "acquiring_sum": 8037.52,
        "logistics_sum": 7813.64,
        "storage_sum": 592.0,
        "acceptance_sum": 148.0,
        "advertising_sum": 0.0,
        "penalty_sum": 74.0,
        "base_price_rub": 1990.0,
        "sale_price_without_spp_rub": 1490.0,
        "price_with_spp_rub": 1311.2,
        "discount_pct": 0.0,
        "wb_site_discount_pct": 25.13,
        "spp_pct": 12.0,
    }


class Feature5CommissionCalculatorServiceTests(unittest.TestCase):
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_finance_metrics")
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_context")
    def test_commission_calculator_returns_single_sku_row(
        self,
        mock_context,
        mock_finance,
    ):
        mock_context.return_value = _sku_context()
        mock_finance.return_value = _sku_finance()

        response = get_commission_calculator_payload(
            nm_id=70194686,
            date_from="2026-03-30",
            date_to="2026-04-05",
        )

        self.assertEqual(response.report.reportId, "commission_calculator")
        self.assertEqual(response.rows[0].rowId, "commission_70194686")
        self.assertEqual(response.rows[0].brand, "LIFECODE")
        self.assertEqual(response.rows[0].subject, "Сыворотки")
        self.assertEqual(response.rows[0].productName, "Сыворотка для лица")
        self.assertEqual(response.rows[0].vendorArticle, "46 227")
        self.assertEqual(response.rows[0].article, 70194686)
        self.assertEqual(response.rows[0].basePriceRub, 1990)
        self.assertEqual(response.rows[0].discountPct, 25.13)
        self.assertEqual(response.rows[0].wbSiteDiscountPct, 25.13)
        self.assertEqual(response.rows[0].salePriceWithoutSppRub, 1490)
        self.assertEqual(response.rows[0].wbClubDiscountPct, 0)
        self.assertEqual(response.rows[0].priceWithWbClubRub, 1490)
        self.assertEqual(response.rows[0].sppPct, 12)
        self.assertEqual(response.rows[0].priceWithSppRub, 1311.2)
        self.assertEqual(response.rows[0].commissionPct, 31.04)
        self.assertEqual(response.rows[0].acquiringPct, 2.2)
        self.assertEqual(response.rows[0].totalCommissionPct, 33.23)
        self.assertEqual(response.rows[0].commissionRub, 495.13)
        self.assertEqual(response.rows[0].logisticsRub, 105.59)
        self.assertEqual(response.rows[0].buyoutPct, 88.4)
        self.assertEqual(response.rows[0].unitCostRub, 344)
        self.assertEqual(response.rows[0].defectRub, 1)
        self.assertEqual(response.rows[0].totalCostRub, 345)
        self.assertEqual(response.rows[0].wbPayoutRub, 879.28)
        self.assertEqual(response.rows[0].profitRub, 534.28)
        self.assertEqual(response.totals.requestedNmId, 70194686)
        self.assertEqual(response.totals.returnedRowCount, 1)
        self.assertTrue(response.totals.hasPriceData)
        self.assertTrue(response.totals.hasFinanceData)
        self.assertTrue(response.totals.hasCostData)
        column_keys = [column.key for column in response.columns]
        self.assertIn("wbClubDiscountPct", column_keys)
        self.assertIn("priceWithWbClubRub", column_keys)
        self.assertNotIn("priceWithWalletRub", column_keys)
        self.assertIn("taxRub", column_keys)
        self.assertIn("taxSalePct", column_keys)
        self.assertNotIn("vatRub", column_keys)
        mock_context.assert_called_once_with(nm_id=70194686, report_date="2026-04-05")
        mock_finance.assert_called_once_with(
            nm_id=70194686,
            date_from="2026-03-30",
            date_to="2026-04-05",
        )

    @patch("agi_ecom_app.commission_calculator.service._get_latest_finance_report_date")
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_finance_metrics")
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_context")
    def test_commission_calculator_uses_last_seven_days_by_default(
        self,
        mock_context,
        mock_finance,
        mock_latest_date,
    ):
        mock_latest_date.return_value = date(2026, 4, 5)
        mock_context.return_value = _sku_context()
        mock_finance.return_value = _sku_finance()

        response = get_commission_calculator_payload(nm_id=70194686)

        self.assertTrue(response.period.isDefaultPeriod)
        self.assertEqual(response.period.dateFrom, "2026-03-30")
        self.assertEqual(response.period.dateTo, "2026-04-05")

    def test_commission_calculator_requires_positive_nm_id(self):
        with self.assertRaisesRegex(ValueError, "nm_id must be a positive integer"):
            get_commission_calculator_payload(nm_id=0, date_from="2026-03-30")

    def test_commission_calculator_requires_date_from_when_date_to_is_provided(self):
        with self.assertRaisesRegex(ValueError, "date_from is required"):
            get_commission_calculator_payload(nm_id=70194686, date_to="2026-04-05")

    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_finance_metrics")
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_context")
    def test_commission_calculator_returns_not_found_when_sku_absent(
        self,
        mock_context,
        mock_finance,
    ):
        mock_context.return_value = None
        mock_finance.return_value = None

        with self.assertRaisesRegex(LookupError, "nm_id=70194686 was not found"):
            get_commission_calculator_payload(
                nm_id=70194686,
                date_from="2026-03-30",
                date_to="2026-04-05",
            )

    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_finance_metrics")
    @patch("agi_ecom_app.commission_calculator.service._fetch_sku_context")
    def test_commission_calculator_falls_back_to_finance_prices(
        self,
        mock_context,
        mock_finance,
    ):
        context = _sku_context()
        context["price_before_discount"] = None
        context["price_after_discount"] = None
        context["seller_discount_pct"] = None
        context["discount_pct"] = None
        context["club_discount_pct"] = None
        context["price_with_wb_club"] = None
        context["spp_pct"] = None
        context["price_with_spp"] = None
        mock_context.return_value = context
        mock_finance.return_value = _sku_finance()

        response = get_commission_calculator_payload(
            nm_id=70194686,
            date_from="2026-03-30",
            date_to="2026-04-05",
        )

        row = response.rows[0]
        self.assertEqual(row.basePriceRub, 1990)
        self.assertEqual(row.discountPct, 0)
        self.assertEqual(row.wbSiteDiscountPct, 25.13)
        self.assertEqual(row.salePriceWithoutSppRub, 1490)
        self.assertEqual(row.sppPct, 12)
        self.assertEqual(row.priceWithSppRub, 1311.2)
        self.assertEqual(row.checkStatus, "Цены заполнены")
        self.assertTrue(response.totals.hasPriceData)


class Feature5CommissionCalculatorApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.commission_calculator.router.get_commission_calculator_payload")
    def test_commission_calculator_endpoint_returns_json_payload(self, mock_get_payload):
        mock_get_payload.return_value = CommissionCalculatorResponse(
            report=CommissionCalculatorReport(
                reportId="commission_calculator",
                label="Калькулятор комиссий",
                sortOrder=1,
            ),
            period=CommissionCalculatorPeriod(
                dateFrom="2026-03-30",
                dateTo="2026-04-05",
                isDefaultPeriod=False,
            ),
            columns=[CommissionCalculatorColumn(key="brand", label="Бренд", type="text")],
            rows=[
                CommissionCalculatorRow(
                    rowId="commission_70194686",
                    brand="LIFECODE",
                    subject="Сыворотки",
                    productName="Сыворотка для лица",
                    vendorArticle="46 227",
                    article=70194686,
                    salePriceWithoutSppRub=1490,
                    commissionPct=31.04,
                    totalCommissionPct=33.23,
                )
            ],
            totals=CommissionCalculatorTotals(
                requestedNmId=70194686,
                returnedRowCount=1,
                hasPriceData=True,
                hasFinanceData=True,
                hasCostData=True,
            ),
        )

        response = self.client.get(
            "/v1/finance/commission-calculator",
            params={
                "nm_id": 70194686,
                "date_from": "2026-03-30",
                "date_to": "2026-04-05",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rows"][0]["rowId"], "commission_70194686")
        self.assertEqual(response.json()["rows"][0]["brand"], "LIFECODE")
        self.assertEqual(response.json()["rows"][0]["totalCommissionPct"], 33.23)
        mock_get_payload.assert_called_once_with(
            nm_id=70194686,
            date_from="2026-03-30",
            date_to="2026-04-05",
        )

    @patch("agi_ecom_app.commission_calculator.router.get_commission_calculator_payload")
    def test_commission_calculator_endpoint_can_return_catalog_payload(self, mock_get_payload):
        mock_get_payload.return_value = CommissionCalculatorResponse(
            report=CommissionCalculatorReport(
                reportId="commission_calculator",
                label="Калькулятор комиссий",
                sortOrder=1,
            ),
            period=CommissionCalculatorPeriod(
                dateFrom="2026-03-30",
                dateTo="2026-04-05",
                isDefaultPeriod=True,
            ),
            columns=[CommissionCalculatorColumn(key="brand", label="Бренд", type="text")],
            rows=[
                CommissionCalculatorRow(
                    rowId="commission_70194686",
                    brand="LIFECODE",
                    article=70194686,
                ),
                CommissionCalculatorRow(
                    rowId="commission_171742855",
                    brand="LIFECODE",
                    article=171742855,
                ),
            ],
            totals=CommissionCalculatorTotals(
                returnedRowCount=2,
                hasPriceData=False,
                hasFinanceData=True,
                hasCostData=True,
            ),
        )

        response = self.client.get("/v1/finance/commission-calculator")

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["totals"]["requestedNmId"])
        self.assertEqual(response.json()["totals"]["returnedRowCount"], 2)
        mock_get_payload.assert_called_once_with(
            nm_id=None,
            date_from=None,
            date_to=None,
        )

    @patch("agi_ecom_app.commission_calculator.router.get_commission_calculator_payload")
    def test_commission_calculator_endpoint_returns_404_for_unknown_sku(
        self,
        mock_get_payload,
    ):
        mock_get_payload.side_effect = LookupError("nm_id=70194686 was not found in calculator sources")

        response = self.client.get(
            "/v1/finance/commission-calculator",
            params={"nm_id": 70194686, "date_from": "2026-03-30"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("nm_id=70194686", response.json()["detail"])

    @patch("agi_ecom_app.commission_calculator.router.get_commission_calculator_payload")
    def test_commission_calculator_endpoint_returns_422_for_validation_error(
        self,
        mock_get_payload,
    ):
        mock_get_payload.side_effect = ValueError("nm_id must be a positive integer")

        response = self.client.get(
            "/v1/finance/commission-calculator",
            params={"nm_id": 0, "date_from": "2026-03-30"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "nm_id must be a positive integer")

    @patch("agi_ecom_app.commission_calculator.router.get_commission_calculator_payload")
    def test_commission_calculator_endpoint_returns_503_for_runtime_error(
        self,
        mock_get_payload,
    ):
        mock_get_payload.side_effect = RuntimeError("failed to load SKU finance metrics")

        response = self.client.get(
            "/v1/finance/commission-calculator",
            params={"nm_id": 70194686, "date_from": "2026-03-30"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "failed to load SKU finance metrics")


if __name__ == "__main__":
    unittest.main()
