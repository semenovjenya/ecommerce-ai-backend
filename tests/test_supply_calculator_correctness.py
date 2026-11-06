import unittest
from unittest.mock import patch

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.calculators.supply import calculate_supply_scenario
from agi_ecom_app.forecast.models import ForecastHypothesis


class SupplyCalculatorCorrectnessTests(unittest.TestCase):
    def _hypothesis(self) -> ForecastHypothesis:
        return ForecastHypothesis(
            domain="supply",
            action="reorder_for_growth",
            targets=["46227"],
        )

    @patch("agi_ecom_app.forecast.calculators.supply.strategic_db.get_supply_overview")
    def test_sellable_stock_is_used_and_customer_transit_does_not_reduce_reorder(
        self,
        mock_supply,
    ):
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46227",
                    "stock_qty": 5,
                    "stock_qty_full": 100,
                    "in_way_to_client": 80,
                    "in_way_from_client": 20,
                    "sales_7d_units": 35,
                    "days_of_stock": 1,
                    "reorder_flag": "critical",
                    "reorder_qty_recommended": 100,
                }
            ]
        }

        result = calculate_supply_scenario(
            self._hypothesis(),
            ForecastEvidenceBundle(),
        )

        row = result["target_results"][0]
        self.assertEqual(row["available_stock"], 5.0)
        self.assertIsNone(row["inbound_stock"])
        self.assertEqual(row["customer_transit_to_client"], 80.0)
        self.assertEqual(row["customer_transit_from_client"], 20.0)
        self.assertEqual(result["decision_status"], "needs_review")
        self.assertIn("missing_supplier_inbound_snapshot", result["blocked_by"])
        self.assertIn("консерватив", result["final_recommendation"].lower())

    @patch("agi_ecom_app.forecast.calculators.supply.strategic_db.get_supply_overview")
    def test_explicit_zero_supplier_inbound_allows_reorder_contract(self, mock_supply):
        mock_supply.return_value = {
            "rows": [
                {
                    "vendor_code": "46227",
                    "stock_qty": 5,
                    "stock_qty_full": 100,
                    "supplier_inbound_qty": 0,
                    "in_way_to_client": 80,
                    "in_way_from_client": 20,
                    "sales_7d_units": 35,
                    "days_of_stock": 1,
                    "reorder_flag": "critical",
                    "reorder_qty_recommended": 100,
                }
            ]
        }

        result = calculate_supply_scenario(
            self._hypothesis(),
            ForecastEvidenceBundle(),
        )

        self.assertEqual(result["decision_status"], "completed")
        self.assertEqual(result["baseline_metrics"]["inbound_stock"], 0.0)
        self.assertNotIn("missing_supplier_inbound_snapshot", result["blocked_by"])


if __name__ == "__main__":
    unittest.main()
