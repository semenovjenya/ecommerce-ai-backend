import unittest

from agi_ecom_app.forecast.adapters.diagnostic_input import ForecastEvidenceBundle
from agi_ecom_app.forecast.calculators.action_family import (
    _position_metric_target,
    _visibility_metric_target,
    calculate_forecast_scenario,
)
from agi_ecom_app.forecast.models import ForecastPlan, ResolvedTarget


def _bundle(*, sales=None, supply=None, price=None, finance=None, marketing_action=None):
    return ForecastEvidenceBundle(
        per_target={
            "46227": {
                "opportunity_source_facts": {
                    "sales": sales or {},
                    "supply": supply or {},
                    "price": price or {},
                    "finance": finance or {},
                    "marketing_action": marketing_action or {},
                },
                "opportunity": {},
            }
        }
    )


class ActionFamilyCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.targets = [ResolvedTarget(id="46227", entity_type="sku")]

    def test_scale_does_not_invent_budget_or_revenue_uplift(self):
        result = calculate_forecast_scenario(
            plan=ForecastPlan(
                primary_domain="marketing",
                action_family="scale_profitable_sku_ads",
                target_mode="sku",
            ),
            resolved_targets=self.targets,
            evidence_bundle=_bundle(
                sales={"current_revenue": 100_000, "current_sales": 50},
                supply={"stock_qty": 100, "sales_7d_units": 10, "days_of_stock": 70},
                finance={"contribution_margin": 25_000},
            ),
        )

        self.assertEqual(result["decision_status"], "needs_review")
        self.assertIsNone(result["action_metrics"]["test_ad_budget_rub"])
        self.assertIsNone(result["delta_metrics"]["estimated_revenue_uplift"])
        self.assertIn("missing_evidence_based_ad_budget", result["blocked_by"])
        self.assertFalse(
            any(step["operation"] == "поднять_лимит_рекламы" for step in result["resolution_plan"])
        )

    def test_overstock_does_not_invent_ten_percent_discount(self):
        result = calculate_forecast_scenario(
            plan=ForecastPlan(
                primary_domain="price",
                action_family="overstock_liquidation",
                target_mode="sku",
            ),
            resolved_targets=self.targets,
            evidence_bundle=_bundle(
                sales={"current_revenue": 0, "current_sales": 0},
                supply={"stock_qty": 120, "sales_7d_units": 0, "days_of_stock": 120},
                finance={"contribution_margin": 10_000},
                price={"source_price_promo": 1500, "competitor_price_median": 1000},
            ),
        )

        self.assertEqual(result["decision_status"], "needs_review")
        self.assertIsNone(result["action_metrics"]["safe_discount_pct"])
        self.assertIn("missing_evidence_based_discount", result["blocked_by"])
        self.assertEqual(result["resolution_plan"][0]["operation"], "получить_ценовой_сценарий")

    def test_conversion_recovery_has_no_fixed_eight_percent_effect(self):
        result = calculate_forecast_scenario(
            plan=ForecastPlan(
                primary_domain="sales",
                action_family="recover_conversion_drop",
                target_mode="sku",
            ),
            resolved_targets=self.targets,
            evidence_bundle=_bundle(
                sales={"current_revenue": 50_000, "current_sales": 20},
                supply={"stock_qty": 100, "sales_7d_units": 20, "days_of_stock": 35},
                finance={"contribution_margin": 15_000},
            ),
        )

        self.assertIsNone(result["delta_metrics"]["estimated_revenue_recovery"])

    def test_visibility_plan_does_not_invent_twenty_percent_or_fixed_position_target(self):
        visibility = {
            "open_card_current": 100,
            "avg_position_current": 12,
            "visibility_current": 79,
        }

        conversion_target = _visibility_metric_target("open_card_current", visibility)
        position_target = _position_metric_target(visibility)

        self.assertIn("baseline 100", conversion_target)
        self.assertIn("target", conversion_target)
        self.assertNotIn("120", conversion_target)
        self.assertNotIn("<=10", position_target)
        self.assertNotIn(">=85", position_target)


if __name__ == "__main__":
    unittest.main()
