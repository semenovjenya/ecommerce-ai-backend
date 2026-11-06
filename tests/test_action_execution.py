import unittest
from datetime import date
from unittest.mock import patch

from agi_ecom_app.action_execution import (
    classify_marketing_guardrail,
    classify_stockout_risk,
    execute_action_handler,
    resolve_numeric_targets,
)


def _task_snapshot(**updates):
    payload = {
        "taskId": "action-1",
        "actionKind": "preserve_availability_and_sales",
        "action": "Удержать наличие и продажи: 46227",
        "title": "Удержать наличие и продажи: 46227",
        "targets": ["46227"],
        "businessGoal": "Сохранить наличие товара и не потерять продажи из-за дефицита.",
        "decisionType": "availability_replenishment_marketing_guardrail",
    }
    payload.update(updates)
    return payload


class ActionExecutionTests(unittest.TestCase):
    def test_resolve_numeric_targets_prefers_explicit_targets_over_text_scan(self):
        targets = resolve_numeric_targets(
            {
                "taskId": "action-1:sku-46227",
                "action": "Удержать наличие и продажи: 46227, 46303",
                "title": "Удержать наличие и продажи: 46227, 46303",
                "targets": ["46227"],
                "objects": [
                    {
                        "objectId": "sku-46227",
                        "externalId": "46227",
                        "label": "SKU 46227",
                        "payload": {"nm_id": 46227},
                    }
                ],
            }
        )

        self.assertEqual(targets, ["46227"])

    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview")
    def test_handler_returns_needs_review_when_target_missing(self, mock_supply):
        outcome = execute_action_handler(
            {
                "taskId": "action-1",
                "actionKind": "preserve_availability_and_sales",
                "action": "Удержать наличие и продажи",
            },
            today=date(2026, 4, 23),
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "needs_review")
        self.assertIn("target_resolution_failed", outcome.draft_result.blockedBy)
        mock_supply.assert_not_called()

    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview")
    def test_handler_returns_needs_review_for_multiple_targets(self, mock_supply):
        outcome = execute_action_handler(
            _task_snapshot(
                action="Удержать наличие и продажи: 46227, 46303",
                title="Удержать наличие и продажи: 46227, 46303",
                targets=["46227", "46303"],
            ),
            today=date(2026, 4, 23),
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "needs_review")
        self.assertIn("multiple_targets_not_supported", outcome.draft_result.blockedBy)
        mock_supply.assert_not_called()

    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_sku_period_comparison_report")
    @patch("agi_ecom_app.action_execution.get_finance_sku_report")
    @patch("agi_ecom_app.action_execution.strategic_db.get_sales_sku_revenue_change")
    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview")
    def test_handler_computes_reorder_and_blocks_ad_scale_on_low_coverage(
        self,
        mock_supply,
        mock_sales,
        mock_finance,
        mock_marketing_report,
        mock_marketing_actions,
    ):
        mock_supply.return_value = {
            "report_date": "2026-04-22",
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "stock_qty": 10,
                    "stock_qty_full": 10,
                    "supplier_inbound_qty": 3,
                    "in_way_to_client": 2,
                    "in_way_from_client": 1,
                    "sales_7d_units": 35,
                    "reorder_flag": "critical",
                }
            ]
        }
        mock_sales.return_value = {
            "current_period_end": "2026-04-22",
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "current_sales": 35,
                    "previous_sales": 28,
                    "delta_sales": 7,
                    "current_revenue": 70000,
                    "previous_revenue": 56000,
                    "delta_revenue": 14000,
                }
            ]
        }
        mock_finance.return_value = {
            "date_to": "2026-04-23",
            "rows": [
                {
                    "nm_id": 46227,
                    "Артикул": "46227",
                    "Себестоимость": 150,
                    "Маржа, руб": 20000,
                    "Рентабельность, %": 18,
                }
            ]
        }
        mock_marketing_report.return_value = {
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "current_spend_rub": 5000,
                    "current_ad_orders": 20,
                    "current_estimated_drr_pct": 7,
                }
            ]
        }
        mock_marketing_actions.return_value = {
            "source_freshness": {
                "invalid_sources": [],
                "sources": {"campaign_mart": {"status": "fresh"}},
            },
            "campaign_actions": [{"status": "масштабировать"}],
            "query_actions": [],
            "sku_actions": [],
        }

        outcome = execute_action_handler(_task_snapshot(), today=date(2026, 4, 23))

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "completed")
        self.assertEqual(outcome.draft_result.forecast.salesVelocity, 5.0)
        self.assertEqual(outcome.draft_result.forecast.coverageDays, 2.0)
        self.assertEqual(outcome.draft_result.forecast.safetyStockUnits, 35.0)
        self.assertEqual(outcome.draft_result.forecast.reorderQty, 92.0)
        self.assertEqual(outcome.draft_result.forecast.stockoutRisk, "high")
        self.assertEqual(
            outcome.evidence_package["metrics"]["marketing_guardrail"],
            "block_scale_low_coverage",
        )
        self.assertIn("92", outcome.draft_result.summary or "")

    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_sku_period_comparison_report")
    @patch("agi_ecom_app.action_execution.get_finance_sku_report")
    @patch("agi_ecom_app.action_execution.strategic_db.get_sales_sku_revenue_change")
    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview")
    def test_handler_returns_needs_review_when_margin_not_confirmed(
        self,
        mock_supply,
        mock_sales,
        mock_finance,
        mock_marketing_report,
        mock_marketing_actions,
    ):
        mock_supply.return_value = {
            "report_date": "2026-04-22",
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "stock_qty": 60,
                    "stock_qty_full": 60,
                    "supplier_inbound_qty": 15,
                    "in_way_to_client": 10,
                    "in_way_from_client": 5,
                    "sales_7d_units": 21,
                    "reorder_flag": "warning",
                }
            ]
        }
        mock_sales.return_value = {
            "current_period_end": "2026-04-22",
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "current_sales": 21,
                    "previous_sales": 20,
                    "delta_sales": 1,
                }
            ]
        }
        mock_finance.return_value = {
            "date_to": "2026-04-23",
            "rows": [
                {
                    "nm_id": 46227,
                    "Артикул": "46227",
                    "Себестоимость": None,
                    "Маржа, руб": 12000,
                    "Рентабельность, %": None,
                }
            ]
        }
        mock_marketing_report.return_value = {"rows": []}
        mock_marketing_actions.return_value = {
            "source_freshness": {
                "invalid_sources": [],
                "sources": {"campaign_mart": {"status": "fresh"}},
            },
            "campaign_actions": [{"status": "масштабировать"}],
            "query_actions": [],
            "sku_actions": [],
        }

        outcome = execute_action_handler(_task_snapshot(), today=date(2026, 4, 23))

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "needs_review")
        self.assertIn("missing_confirmed_margin", outcome.draft_result.blockedBy)
        self.assertEqual(
            outcome.evidence_package["metrics"]["marketing_guardrail"],
            "block_scale_missing_or_negative_margin",
        )

    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_action_matrix")
    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_sku_period_comparison_report")
    @patch("agi_ecom_app.action_execution.get_finance_sku_report")
    @patch("agi_ecom_app.action_execution.strategic_db.get_sales_sku_revenue_change")
    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview")
    def test_handler_uses_sellable_stock_and_never_treats_customer_transit_as_supplier_inbound(
        self,
        mock_supply,
        mock_sales,
        mock_finance,
        mock_marketing_report,
        mock_marketing_actions,
    ):
        mock_supply.return_value = {
            "report_date": "2026-04-22",
            "rows": [
                {
                    "nm_id": 46227,
                    "vendor_code": "46227",
                    "stock_qty": 5,
                    "stock_qty_full": 100,
                    "in_way_to_client": 80,
                    "in_way_from_client": 20,
                    "sales_7d_units": 35,
                    "reorder_flag": "critical",
                }
            ]
        }
        mock_sales.return_value = {
            "current_period_end": "2026-04-22",
            "rows": [{"vendor_code": "46227", "current_sales": 35}],
        }
        mock_finance.return_value = {
            "date_to": "2026-04-23",
            "rows": [
                {
                    "Артикул": "46227",
                    "Себестоимость": 150,
                    "Маржа, руб": 20_000,
                    "Рентабельность, %": 18,
                }
            ]
        }
        mock_marketing_report.return_value = {"rows": []}
        mock_marketing_actions.return_value = {}

        outcome = execute_action_handler(_task_snapshot(), today=date(2026, 4, 23))

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.forecast.availableStock, 5.0)
        self.assertIsNone(outcome.draft_result.forecast.inboundStock)
        self.assertEqual(outcome.draft_result.forecast.reorderQty, 100.0)
        self.assertEqual(outcome.draft_result.decisionStatus, "needs_review")
        self.assertIn("missing_supplier_inbound_snapshot", outcome.draft_result.blockedBy)
        self.assertTrue(
            outcome.evidence_package["metrics"]["reorder_qty_is_conservative_preview"]
        )

    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_action_matrix", return_value={})
    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_sku_period_comparison_report", return_value={"rows": []})
    @patch(
        "agi_ecom_app.action_execution.get_finance_sku_report",
        return_value={
            "date_to": "2026-04-23",
            "rows": [
                {
                    "Артикул": "46227",
                    "Себестоимость": 150,
                    "Маржа, руб": 20_000,
                    "Рентабельность, %": 18,
                }
            ],
        },
    )
    @patch(
        "agi_ecom_app.action_execution.strategic_db.get_sales_sku_revenue_change",
        return_value={
            "current_period_end": "2026-04-22",
            "rows": [{"vendor_code": "46227", "current_sales": 35}],
        },
    )
    @patch(
        "agi_ecom_app.action_execution.strategic_db.get_supply_overview",
        return_value={
            "report_date": "2026-04-24",
            "rows": [
                {
                    "vendor_code": "46227",
                    "stock_qty": 10,
                    "supplier_inbound_qty": 0,
                    "sales_7d_units": 35,
                    "reorder_flag": "critical",
                }
            ],
        },
    )
    def test_future_dated_supply_snapshot_hard_blocks_action(
        self,
        _mock_supply,
        _mock_sales,
        _mock_finance,
        _mock_marketing_report,
        _mock_marketing_actions,
    ):
        outcome = execute_action_handler(_task_snapshot(), today=date(2026, 4, 23))

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "blocked")
        self.assertIn("future_supply_snapshot", outcome.draft_result.blockedBy)
        self.assertEqual(
            outcome.evidence_package["source_freshness"]["supply"]["status"],
            "future",
        )

    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_action_matrix", return_value={})
    @patch("agi_ecom_app.action_execution.strategic_db.get_marketing_sku_period_comparison_report", return_value={})
    @patch("agi_ecom_app.action_execution.get_finance_sku_report", return_value={"rows": []})
    @patch("agi_ecom_app.action_execution.strategic_db.get_sales_sku_revenue_change", return_value={"rows": []})
    @patch("agi_ecom_app.action_execution.strategic_db.get_supply_overview", return_value={"rows": []})
    def test_handler_returns_blocked_when_supply_and_sales_base_missing(
        self,
        _mock_supply,
        _mock_sales,
        _mock_finance,
        _mock_marketing_report,
        _mock_marketing_actions,
    ):
        outcome = execute_action_handler(_task_snapshot(), today=date(2026, 4, 23))

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.draft_result.decisionStatus, "blocked")
        self.assertIn("missing_supply_snapshot", outcome.draft_result.blockedBy)
        self.assertIn("missing_sales_velocity", outcome.draft_result.blockedBy)

    def test_stockout_risk_thresholds(self):
        self.assertEqual(
            classify_stockout_risk(coverage_days=5, reorder_flag=None, sales_velocity=1.0),
            "high",
        )
        self.assertEqual(
            classify_stockout_risk(coverage_days=10, reorder_flag="warning", sales_velocity=1.0),
            "medium",
        )
        self.assertEqual(
            classify_stockout_risk(coverage_days=25, reorder_flag="ok", sales_velocity=1.0),
            "low",
        )
        self.assertEqual(
            classify_stockout_risk(coverage_days=None, reorder_flag=None, sales_velocity=None),
            "unknown",
        )

    def test_marketing_guardrail_matrix(self):
        self.assertEqual(
            classify_marketing_guardrail(
                profitability_pct=12,
                coverage_days=30,
                stockout_risk="low",
                has_reduce_signal=False,
                has_scale_signal=True,
            ),
            "allow_scale",
        )
        self.assertEqual(
            classify_marketing_guardrail(
                profitability_pct=None,
                coverage_days=30,
                stockout_risk="low",
                has_reduce_signal=False,
                has_scale_signal=True,
            ),
            "block_scale_missing_or_negative_margin",
        )
        self.assertEqual(
            classify_marketing_guardrail(
                profitability_pct=12,
                coverage_days=10,
                stockout_risk="high",
                has_reduce_signal=False,
                has_scale_signal=True,
            ),
            "block_scale_low_coverage",
        )


if __name__ == "__main__":
    unittest.main()
