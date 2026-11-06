from __future__ import annotations

import unittest

from agi_ecom_app.prompt_quality import (
    CONTRADICTORY_SKU_ACTIONS,
    CONTRIBUTION_MARGIN_AS_NET_PROFIT,
    INCOMPLETE_TARGET_COVERAGE,
    MAX_REPAIR_PROMPT_CHARS,
    MISSING_REQUIRED_METRICS,
    NO_DATA_METRIC_STATED_AS_ZERO,
    PERIOD_CONFLICT,
    SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP,
    build_prompt_v2_repair_prompt,
    validate_prompt_v2_answer,
    validate_evidence_contract,
)


class PromptQualityTests(unittest.TestCase):
    def test_run_191_revenue_delta_cannot_be_labeled_as_sales_units(self):
        package = {
            "canonical_facts": [
                {"metric": "sales_units_change_pct", "value": -7.0, "status": "confirmed"},
                {"metric": "revenue_change_pct", "value": -3.7, "status": "confirmed"},
            ]
        }

        violations = validate_prompt_v2_answer(
            "За последние 7 дней продажи снизились на 3,7%. Выручка снизилась на 3,7%.",
            package,
        )

        self.assertEqual(violations, [SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP])
        self.assertEqual(
            validate_prompt_v2_answer(
                "Продажи в штуках снизились на 7,0%, выручка снизилась на 3,7%.",
                package,
            ),
            [],
        )

    def test_run_196_contribution_margin_cannot_be_called_net_profit(self):
        package = {
            "canonical_facts": [
                {
                    "sku": "46227",
                    "metric": "contribution_margin_rub",
                    "value": 576_686.67,
                    "status": "confirmed",
                },
                {"sku": "46227", "metric": "net_profit_rub", "status": "no_data"},
            ]
        }

        self.assertEqual(
            validate_prompt_v2_answer(
                "Чистая прибыль SKU 46227 составила 125 000,00 ₽.",
                package,
            ),
            [CONTRIBUTION_MARGIN_AS_NET_PROFIT],
        )
        self.assertEqual(
            validate_prompt_v2_answer(
                "Маржинальная прибыль SKU 46227 составила 125 000,00 ₽; чистая прибыль не рассчитана.",
                package,
            ),
            [],
        )

    def test_net_profit_disclaimers_are_not_contribution_substitution(self):
        package = {
            "canonical_facts": [
                {
                    "sku": "70194686",
                    "metric": "contribution_margin_rub",
                    "value": 7_911_909.83,
                    "status": "confirmed",
                }
            ]
        }

        disclaimers = (
            "Contribution margin не является чистой прибылью.",
            "Contribution margin нельзя называть чистой прибылью.",
            "Не называй contribution margin чистой прибылью.",
            (
                "Есть contribution margin и отдельные расходы, но нет полного "
                "подтвержденного P&L слоя, который бы свёл всё в net profit."
            ),
        )

        for markdown in disclaimers:
            with self.subTest(markdown=markdown):
                self.assertEqual(validate_prompt_v2_answer(markdown, package), [])

    def test_real_contribution_to_net_profit_substitution_is_still_rejected(self):
        package = {
            "canonical_facts": [
                {
                    "sku": "70194686",
                    "metric": "contribution_margin_rub",
                    "value": 7_911_909.83,
                    "status": "confirmed",
                }
            ]
        }

        substitutions = (
            "Contribution margin считается чистой прибылью.",
            (
                "Contribution margin не является метрикой затрат, "
                "а считается чистой прибылью."
            ),
            "Чистая прибыль SKU 70194686 составила 7 911 909,83 ₽.",
        )

        for markdown in substitutions:
            with self.subTest(markdown=markdown):
                self.assertEqual(
                    validate_prompt_v2_answer(markdown, package),
                    [CONTRIBUTION_MARGIN_AS_NET_PROFIT],
                )

    def test_runs_191_and_197_no_data_zero_and_conflicting_reorder_actions(self):
        package = {
            "sku_evidence_matrix": [
                {
                    "sku": "46224",
                    "domains": {
                        "supply": {"status": "no_data", "reason": "sku_not_in_source"}
                    },
                }
            ]
        }
        markdown = """
| SKU | Остаток | Действие |
|---|---:|---|
| 46224 | 0 шт. | Приостановить дозаказ |

- SKU 46224: поддерживать регулярное пополнение.
"""

        self.assertEqual(
            validate_prompt_v2_answer(markdown, package),
            [NO_DATA_METRIC_STATED_AS_ZERO, CONTRADICTORY_SKU_ACTIONS],
        )

    def test_repair_prompt_is_deterministic_and_hard_bounded(self):
        package = {
            "canonical_facts": [
                {"metric": "sales_units_change_pct", "value": -7.0},
                {"metric": "revenue_change_pct", "value": -3.7},
            ],
            "supporting_evidence": ["not part of canonical repair facts"],
        }
        draft = "За последние 7 дней продажи снизились на 3,7%. " + ("Длинный текст. " * 2_000)
        violations = [SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP]

        first = build_prompt_v2_repair_prompt(
            draft,
            violations,
            package,
            max_chars=1_200,
        )
        second = build_prompt_v2_repair_prompt(
            draft,
            violations,
            package,
            max_chars=1_200,
        )

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 1_200)
        self.assertIn("VIOLATIONS:", first)
        self.assertIn(SALES_UNITS_PCT_REVENUE_PCT_LABEL_SWAP, first)
        self.assertIn("CANONICAL_FACTS:", first)
        self.assertIn("sales_units_change_pct", first)
        self.assertIn("DRAFT:", first)
        self.assertIn("За последние 7 дней", first)

        default_prompt = build_prompt_v2_repair_prompt(draft, violations, package)
        self.assertLessEqual(len(default_prompt), MAX_REPAIR_PROMPT_CHARS)

    def test_explicit_evidence_contract_blocks_partial_targets_missing_metrics_and_period_conflicts(self):
        package = {
            "request_coverage": {
                "requested_targets": ["46227", "46230", "46303"],
                "covered_targets": ["46227"],
                "missing_targets": ["46230", "46303"],
            },
            "required_metrics": ["unit_cost_rub", "contribution_margin_rub"],
            "canonical_facts": [
                {
                    "sku": "46227",
                    "metric": "contribution_margin_rub",
                    "value": 1000,
                    "status": "confirmed",
                }
            ],
            "period_conflicts": [
                {"left": "2026-06-01/2026-06-07", "right": "2026-06-08/2026-06-14"}
            ],
        }

        self.assertEqual(
            validate_evidence_contract(package),
            [INCOMPLETE_TARGET_COVERAGE, MISSING_REQUIRED_METRICS, PERIOD_CONFLICT],
        )
        self.assertEqual(
            validate_prompt_v2_answer("Факты собраны.", package),
            [INCOMPLETE_TARGET_COVERAGE, MISSING_REQUIRED_METRICS, PERIOD_CONFLICT],
        )

    def test_legacy_evidence_without_contract_fields_is_not_rejected(self):
        self.assertEqual(validate_evidence_contract({"supporting_evidence": ["legacy"]}), [])


if __name__ == "__main__":
    unittest.main()
