from __future__ import annotations

from datetime import date
import threading
import time
from unittest.mock import patch

from agi_ecom_app.runner import (
    _compact_table,
    _extract_sku_metric_matrix,
    _run_round_1_evidence_batch,
    apply_query_plan_output_contract,
    build_query_plan_inventory_matrix,
)
from agi_ecom_app.schemas import EcomRunContext, QueryIntent, ReportTable


def test_compact_table_preserves_complete_sku_matrix() -> None:
    table = ReportTable(
        title="SKU-матрица",
        columns=["SKU", "Продажи"],
        rows=[[str(index), index] for index in range(125)],
    )

    assert len(_compact_table(table)["rows"]) == 125


def test_sku_matrix_is_detected_by_vendor_code_contract() -> None:
    matrices = _extract_sku_metric_matrix(
        {
            "tables": [
                {
                    "title": "SKU-матрица с покрытием",
                    "columns": ["nm_id", "vendor_code", "days_on_hand"],
                    "rows": [[70195171, "46229", 76.7]],
                }
            ]
        }
    )

    assert len(matrices) == 1
    assert matrices[0]["rows"] == [[70195171, "46229", 76.7]]


def test_query_plan_inventory_resolver_builds_full_source_matrix() -> None:
    sku_rows = [
        {
            "sku": str(46220 + index),
            "current_stock_qty": 10 + index,
            "sales_7d_units": 7,
            "sales_units": 30,
            "avg_stock_qty": 18,
            "days_on_hand": 18,
        }
        for index in range(17)
    ]
    with (
        patch(
            "agi_ecom_app.runner.strategic_db.get_supply_overview",
            return_value={"totals": {"report_date": "2026-07-19"}, "rows": []},
        ) as supply,
        patch(
            "agi_ecom_app.runner.strategic_db.get_inventory_turnover",
            return_value={"sku_metric_matrix": sku_rows},
        ) as turnover,
        patch(
            "agi_ecom_app.runner.strategic_db.get_current_inventory_value",
            return_value={
                "rows": [
                    {
                        "vendor_code": "46220",
                        "current_stock_value_rub": 25000,
                        "valuation_status": "confirmed",
                    }
                ]
            },
        ),
    ):
        matrices = build_query_plan_inventory_matrix(
            {"requested_analyses": ["inventory_matrix", "inventory_value"]}
        )

    assert len(matrices[0]["rows"]) == 17
    assert matrices[0]["rows"][0]["current_stock_value_rub"] == 25000
    supply.assert_called_once_with(limit=100)
    turnover.assert_called_once_with(
        date_from="2026-06-20",
        date_to="2026-07-19",
        limit=1000,
    )


def test_portfolio_focus_targets_keep_full_inventory_matrix() -> None:
    sku_rows = [
        {
            "sku": target,
            "current_stock_qty": 10,
            "sales_7d_units": 7,
            "sales_units": 30,
            "avg_stock_qty": 18,
            "days_on_hand": 18,
        }
        for target in ("46227", "46230", "46303")
    ]
    with (
        patch(
            "agi_ecom_app.runner.strategic_db.get_sku_identity_map"
        ) as identity_map,
        patch(
            "agi_ecom_app.runner.strategic_db.get_supply_overview",
            return_value={"totals": {"report_date": "2026-07-19"}, "rows": []},
        ),
        patch(
            "agi_ecom_app.runner.strategic_db.get_inventory_turnover",
            return_value={"sku_metric_matrix": sku_rows},
        ) as turnover,
        patch(
            "agi_ecom_app.runner.strategic_db.get_current_inventory_value",
            return_value={"rows": []},
        ),
    ):
        matrices = build_query_plan_inventory_matrix(
            {
                "scope": "portfolio",
                "targets": [
                    {"target_type": "vendor_code", "value": "46227"},
                    {"target_type": "vendor_code", "value": "46303"},
                ],
                "requested_analyses": ["inventory_matrix"],
            }
        )

    assert [row["sku"] for row in matrices[0]["rows"]] == [
        "46227",
        "46230",
        "46303",
    ]
    identity_map.assert_not_called()
    turnover.assert_called_once_with(
        date_from="2026-06-20",
        date_to="2026-07-19",
        limit=1000,
    )


def test_sku_list_inventory_matrix_remains_an_exclusive_filter() -> None:
    sku_rows = [
        {
            "sku": target,
            "current_stock_qty": 10,
            "sales_7d_units": 7,
            "sales_units": 30,
            "avg_stock_qty": 18,
            "days_on_hand": 18,
        }
        for target in ("46227", "46303")
    ]
    with (
        patch(
            "agi_ecom_app.runner.strategic_db.get_sku_identity_map",
            return_value={
                "70194686": {"nm_id": 70194686, "vendor_code": "46227"},
                "46303": {"nm_id": 171742855, "vendor_code": "46303"},
            },
        ) as identity_map,
        patch(
            "agi_ecom_app.runner.strategic_db.get_supply_overview",
            return_value={"totals": {"report_date": "2026-07-19"}, "rows": []},
        ),
        patch(
            "agi_ecom_app.runner.strategic_db.get_inventory_turnover",
            return_value={"sku_metric_matrix": sku_rows},
        ) as turnover,
        patch(
            "agi_ecom_app.runner.strategic_db.get_current_inventory_value",
            return_value={"rows": []},
        ),
    ):
        matrices = build_query_plan_inventory_matrix(
            {
                "scope": "sku_list",
                "targets": [
                    {"target_type": "nm_id", "value": "70194686"},
                    {"target_type": "vendor_code", "value": "46303"},
                ],
                "requested_analyses": ["inventory_matrix"],
            }
        )

    assert [row["sku"] for row in matrices[0]["rows"]] == ["46227", "46303"]
    identity_map.assert_called_once_with(["70194686", "46303"])
    turnover.assert_called_once_with(
        date_from="2026-06-20",
        date_to="2026-07-19",
        limit=1000,
        vendor_codes="46227,46303",
    )


def test_query_plan_contract_restores_full_deterministic_inventory_matrix() -> None:
    package = {
        "sku_metric_matrix": [
            {
                "columns": ["SKU", "Доступный остаток, шт.", "Продажи 7д, шт."],
                "rows": [["46229", 10, 1], ["46313", 0, 5]],
            },
            {
                "rows": [
                    {
                        "sku": "46229",
                        "sales_units": 12,
                        "days_on_hand": 76.7,
                    },
                    {
                        "sku": "46313",
                        "sales_units": 15,
                        "avg_stock_qty": 9.2,
                        "days_on_hand": 999,
                    },
                ]
            },
        ]
    }
    plan = {
        "objective": "Снизить оборачиваемость всей матрицы до 30 дней",
        "requested_analyses": ["inventory_matrix", "inventory_turnover"],
        "constraints": {"target_cover_days": "30", "horizon_days": 60},
    }

    result = apply_query_plan_output_contract(
        "## SKU-матрица\n\nСтарая сокращённая таблица.\n\n## Рекомендации\n\nТекст LLM.",
        package,
        plan,
    )

    assert "Снизить оборачиваемость всей матрицы до 30 дней" in result
    assert "целевое покрытие — 30 дней" in result
    assert "горизонт — 60 дней" in result
    assert result.count("46229") == 1
    assert result.count("46313") == 1
    assert "критический избыток &gt;60 дней" in result or "критический избыток >60 дней" in result
    assert "норма 14–30 дней" in result
    assert "18.4" in result
    assert "999" not in result
    assert "нет доступного остатка" in result
    assert "Старая сокращённая таблица" not in result
    assert "Текст LLM" in result


def test_round_one_cross_checks_use_parallel_isolated_contexts(monkeypatch) -> None:
    context = EcomRunContext(
        user_query="portfolio",
        intent=QueryIntent.SUPPLY,
        today=date(2026, 7, 20),
        db_target="test",
        chat_id="chat",
        run_id=10,
    )
    items = [
        {"target_agent": agent, "question": agent}
        for agent in ("finance", "marketing", "price", "sales")
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_run_domain_evidence(*, item, context, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        context.tool_trace.append(f"tool:{item['target_agent']}")
        time.sleep(0.05)
        with lock:
            active -= 1
        return {
            "target_agent": item["target_agent"],
            "question": item["question"],
            "output": {},
        }

    monkeypatch.setenv("AGENT_SYNC_CROSS_CHECK_CONCURRENCY", "4")
    with patch(
        "agi_ecom_app.runner._run_domain_evidence",
        side_effect=fake_run_domain_evidence,
    ):
        evidence = _run_round_1_evidence_batch(
            items=items,
            questions=[None] * len(items),
            user_input="portfolio",
            primary_output={},
            context=context,
            run_id=10,
        )

    assert [item["target_agent"] for item in evidence] == [
        "finance",
        "marketing",
        "price",
        "sales",
    ]
    assert max_active == 4
    assert len(context.tool_trace) == 4
