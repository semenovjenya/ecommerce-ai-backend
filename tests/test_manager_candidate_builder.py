from __future__ import annotations

from datetime import datetime, timezone

from agi_ecom_app.manager_decision.candidate_builder import (
    build_manager_action_candidates,
)
from agi_ecom_app.manager_decision.workflow import run_manager_decision_workflow


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def _complete_ad_row(advert_id: int, nm_id: int, *, spend: int = 700):
    return {
        "advert_id": advert_id,
        "campaign_name": f"Campaign {advert_id}",
        "campaign_status": 9,
        "campaign_snapshot_date": "2026-07-13",
        "campaign_retrieved_at": "2026-07-14T08:00:00Z",
        "nm_id": nm_id,
        "active_sku_count": 1,
        "bid_search_kopecks": 420,
        # WB encodes an inactive/unset placement as zero. The sole positive
        # search bid must be selected without an ambiguity blocker.
        "bid_recommendations_kopecks": 0,
        "bid_snapshot_date": "2026-07-13",
        "bid_retrieved_at": "2026-07-14T08:00:00Z",
        "budget_remaining_rub": 900,
        "budget_snapshot_date": "2026-07-13",
        "budget_retrieved_at": "2026-07-14T08:00:00Z",
        "stats_period_start": "2026-07-07",
        "stats_period_end": "2026-07-13",
        "stats_day_count": 7,
        "stats_window_complete": True,
        "campaign_sku_attribution_reconciled": True,
        "ad_spend_7d_rub": spend,
        "clicks_7d": 100,
        "ad_orders_7d": 0,
        "ad_units_7d": 0,
        "daily_spend_min_rub": 80,
        "daily_spend_median_rub": 100,
        "daily_spend_max_rub": 120,
        "daily_spend_values_rub": [80, 90, 95, 100, 105, 110, 120],
        "stock_snapshot_date": "2026-07-13",
        "stock_on_hand": 0,
        "stock_full_for_audit": 25,
        "stock_snapshot_complete": True,
        "supplier_snapshot_date": "2026-07-13",
        "supplier_inbound_qty": 0,
        "supplier_eta_date": None,
        "supplier_goods_rows": 1,
        "supplier_snapshot_complete": True,
        "unit_cogs_rub": 250,
        "cogs_snapshot_date": "2026-07-13",
        "cogs_retrieved_at": "2026-07-14T08:00:00Z",
        "attributed_revenue_7d_rub": 0,
        "attributed_revenue_source_table": "test.campaign_attribution_daily",
        "campaign_contribution_profit_7d_rub": 0,
        "campaign_contribution_source_table": "test.campaign_contribution_daily",
    }


def _snapshot(*, advertising_rows=None, supply_rows=None):
    return {
        "schemaVersion": "1.0",
        "retrievedAt": "2026-07-14T08:00:00Z",
        "advertisingRows": list(advertising_rows or []),
        "supplyRows": list(supply_rows or []),
    }


def _blocked_supply_row(nm_id: int, vendor_code: str, *, quantity: int = 24):
    return {
        "report_date": "2026-07-13",
        "nm_id": nm_id,
        "vendor_code": vendor_code,
        "stock_qty": 0,
        "stock_qty_full": 3,
        "sales_7d_units": 3,
        "avg_daily_sales_7d": 0.43,
        "days_of_stock": 0,
        "reorder_flag": "critical",
        "reorder_qty_recommended": quantity,
        "unit_cogs_rub": 379,
        "cogs_snapshot_date": "2026-07-13",
        "cogs_retrieved_at": "2026-07-14T08:00:00Z",
        "supplier_snapshot_date": "2026-07-13",
        "supplier_inbound_qty": 0,
        "lead_time_days": None,
        "target_cover_days": None,
        "minimum_order_qty": None,
        "replenishment_policy_version": None,
    }


def test_current_like_ad_signal_is_exact_but_rejected_without_fake_economics(
    monkeypatch,
):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "off")
    row = {
        **_complete_ad_row(37424786, 180286536, spend=5321),
        "campaign_snapshot_date": "2026-07-09",
        "bid_snapshot_date": "2026-07-09",
        "budget_snapshot_date": "2026-07-09",
        "stats_period_start": "2026-06-17",
        "stats_period_end": "2026-06-23",
        "ad_orders_7d": 2,
        "daily_spend_min_rub": 30.79,
        "daily_spend_median_rub": 608.74,
        "daily_spend_max_rub": 2112.42,
        "daily_spend_values_rub": [30.79, 608.74, 400, 600, 700, 869.05, 2112.42],
        "stock_snapshot_date": "2026-07-08",
        "supplier_snapshot_date": None,
        "supplier_inbound_qty": None,
        "attributed_revenue_7d_rub": None,
        "campaign_contribution_profit_7d_rub": None,
        "bid_recommendations_kopecks": 0,
    }

    result = build_manager_action_candidates(_snapshot(advertising_rows=[row]), now=NOW)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate["actionId"] == "det-ad-pause-37424786"
    assert candidate["target"] == "37424786"
    assert candidate["targetRefs"] == ["180286536"]
    assert candidate["operation"] == "pause_campaign"
    assert candidate["executionPayload"] == {"advert_id": 37424786}
    assert candidate["from"] == {"value": 9, "unit": "wb_status_code"}
    assert candidate["to"] == {"value": 11, "unit": "wb_status_code"}
    assert "economics" not in candidate
    assert "expectedEffect" not in candidate
    codes = {item["code"] for item in candidate["candidateBlockers"]}
    assert {
        "missing_confirmed_supplier_inbound",
        "missing_attributed_revenue",
        "missing_campaign_contribution",
        "unsafe_pause_with_attributed_orders",
        "stale_source_data",
        "missing_deterministic_economics",
    } <= codes
    facts = {item["metric"]: item for item in candidate["evidence"]}
    assert facts["stock_on_hand"]["value"] == 0
    assert facts["bid"]["value"] == 420
    assert "ambiguous_current_bid_placement" not in codes
    assert facts["stock_on_hand"]["target"] == "180286536"
    assert "supplier_inbound" not in facts
    assert facts["daily_ad_spend_distribution"]["value"] == {
        "min": 30.79,
        "median": 608.74,
        "max": 2112.42,
    }


def test_pause_candidate_requires_and_carries_seven_daily_spend_values(
    monkeypatch,
):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    row = _complete_ad_row(1001, 70195171)

    complete = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    ).candidates[0]
    fact = next(
        item
        for item in complete["evidence"]
        if item["metric"] == "daily_ad_spend_distribution"
    )

    assert fact["value"] == {
        "valuesRub": [80, 90, 95, 100, 105, 110, 120],
        "min": 80,
        "median": 100,
        "max": 120,
    }

    incomplete_row = {**row, "daily_spend_values_rub": [80, 90, 95, 100, 105, 110]}
    incomplete = build_manager_action_candidates(
        _snapshot(advertising_rows=[incomplete_row]),
        now=NOW,
    ).candidates[0]
    codes = {item["code"] for item in incomplete["candidateBlockers"]}

    assert "incomplete_daily_spend_history" in codes
    assert not any(
        item["metric"] == "daily_ad_spend_distribution"
        for item in incomplete["evidence"]
    )


def test_complete_zero_stock_pause_contract_passes_manager_hard_gate(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    rows = [
        _complete_ad_row(1001, 70195171, spend=700),
        _complete_ad_row(1002, 155863996, spend=800),
        _complete_ad_row(1003, 309891193, spend=900),
    ]
    built = build_manager_action_candidates(_snapshot(advertising_rows=rows), now=NOW)

    assert len(built.candidates) == 3
    assert all("candidateBlockers" not in candidate for candidate in built.candidates)
    first = built.candidates[0]
    assert first["economics"]["deltaAdvertisingPerDay"] == {
        "low": -120.0,
        "base": -100.0,
        "high": -80.0,
    }
    assert first["expectedEffect"]["contributionProfitLowPerDay"] == 80.0
    assert first["economics"]["calculationMethod"] == "zero_stock_daily_spend_distribution_v1"
    assert first["economics"]["causalValid"] is False
    guardrail_metrics = {rule["metric"] for rule in first["guardrails"]}
    assert {
        "stock_on_hand",
        "supplier_inbound",
        "ad_orders",
        "attributed_revenue",
        "contribution_profit",
        "budget_remaining",
    } <= guardrail_metrics
    assert {"ad_orders", "ad_units", "revenue"} <= set(first["requiredMetrics"])

    decision = run_manager_decision_workflow(
        {"actionCandidates": built.candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert decision.decision.status == "ready_for_approval"
    assert len(decision.decision.ranked_actions) == 3
    assert all(action.operation == "pause_campaign" for action in decision.decision.ranked_actions)


def test_zero_orders_units_and_revenue_do_not_require_unused_cogs(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    rows = []
    for index in range(3):
        row = _complete_ad_row(1101 + index, 71000001 + index)
        row.update(
            {
                "unit_cogs_rub": None,
                "cogs_snapshot_date": None,
                "cogs_retrieved_at": None,
            }
        )
        rows.append(row)

    built = build_manager_action_candidates(
        _snapshot(advertising_rows=rows),
        now=NOW,
    )

    assert all("candidateBlockers" not in candidate for candidate in built.candidates)
    assert all(
        not any(fact["metric"] == "cogs" for fact in candidate["evidence"])
        for candidate in built.candidates
    )
    assert all(
        not any(ref.endswith(":cogs") for ref in candidate["economics"]["evidenceRefs"])
        for candidate in built.candidates
    )

    decision = run_manager_decision_workflow(
        {"actionCandidates": built.candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert decision.decision.status == "ready_for_approval"
    assert len(decision.decision.ranked_actions) == 3


def test_zero_metrics_without_complete_campaign_sku_window_are_not_proof(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    row = _complete_ad_row(1201, 72000001)
    row.update(
        {
            "stats_window_complete": False,
            "unit_cogs_rub": None,
            "cogs_snapshot_date": None,
            "cogs_retrieved_at": None,
        }
    )

    candidate = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    ).candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}

    assert {"incomplete_campaign_sku_stats", "missing_cogs"} <= codes
    assert "economics" not in candidate


def test_unreconciled_campaign_sku_revenue_is_not_simulation_evidence(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    row = _complete_ad_row(1202, 72000002)
    row["campaign_sku_attribution_reconciled"] = False

    candidate = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    ).candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}

    assert "unreconciled_campaign_sku_attribution" in codes
    assert "economics" not in candidate


def test_source_shaped_incomplete_window_remains_visible_as_needs_data(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    row = _complete_ad_row(1204, 72000004)
    row.update(
        {
            "observed_ad_spend_7d_rub": row["ad_spend_7d_rub"],
            "ad_spend_7d_rub": None,
            "stats_window_complete": False,
            "campaign_sku_attribution_reconciled": False,
            "clicks_7d": None,
            "ad_orders_7d": None,
            "ad_units_7d": None,
            "attributed_revenue_7d_rub": None,
            "campaign_contribution_profit_7d_rub": None,
            "daily_spend_min_rub": None,
            "daily_spend_median_rub": None,
            "daily_spend_max_rub": None,
            "daily_spend_values_rub": None,
        }
    )

    built = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    )

    assert len(built.candidates) == 1
    candidate = built.candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}
    assert {
        "incomplete_campaign_sku_stats",
        "unreconciled_campaign_sku_attribution",
        "missing_deterministic_economics",
    } <= codes
    assert "economics" not in candidate


def test_raw_diagnostic_spend_without_any_complete_window_remains_visible(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "shadow")
    row = _complete_ad_row(1205, 72000005)
    row.update(
        {
            "observed_ad_spend_7d_rub": 700,
            "ad_spend_7d_rub": None,
            "stats_period_start": None,
            "stats_period_end": None,
            "stats_day_count": None,
            "stats_window_complete": None,
            "campaign_sku_attribution_reconciled": None,
            "clicks_7d": None,
            "ad_orders_7d": None,
            "ad_units_7d": None,
            "attributed_revenue_7d_rub": None,
            "campaign_contribution_profit_7d_rub": None,
            "daily_spend_min_rub": None,
            "daily_spend_median_rub": None,
            "daily_spend_max_rub": None,
            "daily_spend_values_rub": None,
            "attributed_revenue_source_table": None,
            "campaign_contribution_source_table": None,
        }
    )

    built = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    )

    assert len(built.candidates) == 1
    candidate = built.candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}
    assert {
        "incomplete_campaign_stats_window",
        "incomplete_campaign_sku_stats",
        "unreconciled_campaign_sku_attribution",
        "missing_deterministic_economics",
    } <= codes
    assert "economics" not in candidate


def test_raw_zero_snapshot_presence_does_not_prove_complete_inputs(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    row = _complete_ad_row(1203, 72000003)
    row.update(
        {
            "stock_snapshot_complete": False,
            "supplier_snapshot_complete": False,
            # Raw rows and zero values are present, but there is no ETL proof
            # that either source snapshot finished.
            "stock_on_hand": 0,
            "supplier_inbound_qty": 0,
            "supplier_audit_snapshot_date": "2026-07-13",
            "supplier_inbound_audited_qty": 0,
        }
    )

    candidate = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]),
        now=NOW,
    ).candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}

    assert {
        "incomplete_stock_snapshot",
        "incomplete_supplier_snapshot",
        "missing_deterministic_economics",
    } <= codes
    assert "economics" not in candidate


def test_ready_candidates_are_selected_before_rejected_diagnostics(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    rejected = []
    for index in range(7):
        row = _complete_ad_row(2001 + index, 73000001 + index, spend=2000 - index)
        row["ad_orders_7d"] = 1
        rejected.append(row)
    ready = [
        _complete_ad_row(3001 + index, 74000001 + index, spend=100 - index)
        for index in range(3)
    ]

    built = build_manager_action_candidates(
        _snapshot(advertising_rows=[*rejected, *ready]),
        now=NOW,
    )

    assert len(built.candidates) == 7
    assert [candidate["target"] for candidate in built.candidates[:3]] == [
        "3001",
        "3002",
        "3003",
    ]
    assert all(
        "candidateBlockers" not in candidate for candidate in built.candidates[:3]
    )
    assert all(
        candidate.get("candidateBlockers") for candidate in built.candidates[3:]
    )


def test_rejected_candidate_blocker_does_not_block_three_independent_actions(
    monkeypatch,
):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    ready = [
        _complete_ad_row(4001 + index, 75000001 + index, spend=700 + index)
        for index in range(3)
    ]
    rejected = _complete_ad_row(4999, 75999999, spend=2000)
    rejected["ad_orders_7d"] = 1
    built = build_manager_action_candidates(
        _snapshot(advertising_rows=[rejected, *ready]),
        now=NOW,
    )

    decision = run_manager_decision_workflow(
        {"actionCandidates": built.candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    ).decision

    assert decision.status == "ready_for_approval"
    assert {action.target for action in decision.ranked_actions} == {
        "4001",
        "4002",
        "4003",
    }
    assert [candidate.candidate_id for candidate in decision.rejected_candidates] == [
        "det-ad-pause-4999"
    ]
    assert decision.blockers == []


def test_simulation_mode_accepts_zero_spend_days_but_legacy_mode_keeps_old_gate(
    monkeypatch,
):
    row = _complete_ad_row(1001, 70195171)
    row.update(
        {
            "daily_spend_values_rub": [0, 40, 80, 100, 105, 110, 120],
            "daily_spend_min_rub": 0,
        }
    )

    monkeypatch.setenv("DECISION_SIMULATION_MODE", "off")
    legacy = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]), now=NOW
    ).candidates[0]
    assert "non_positive_daily_spend_floor" in {
        blocker["code"] for blocker in legacy["candidateBlockers"]
    }

    monkeypatch.setenv("DECISION_SIMULATION_MODE", "enforced")
    simulated = build_manager_action_candidates(
        _snapshot(advertising_rows=[row]), now=NOW
    ).candidates[0]
    assert "candidateBlockers" not in simulated


def test_feature_off_keeps_legacy_advertising_contract(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "off")
    legacy_row = _complete_ad_row(1001, 70195171)
    legacy_row.pop("stats_window_complete")
    legacy_row.pop("campaign_sku_attribution_reconciled")
    legacy_row.pop("ad_units_7d")
    legacy_row.pop("daily_spend_values_rub")
    legacy_row.pop("stock_snapshot_complete")
    legacy_row.pop("supplier_snapshot_complete")
    legacy_row.update(
        {
            "attributed_revenue_source_table": (
                "derived:zero_attributed_orders("
                "marketing.wb_marketing_campaign_stats_daily)"
            ),
            "campaign_contribution_source_table": (
                "derived:zero_order_contribution_after_ad_spend_v1("
                "marketing.wb_marketing_campaign_stats_daily)"
            ),
        }
    )
    candidate = build_manager_action_candidates(
        _snapshot(advertising_rows=[legacy_row]),
        now=NOW,
    ).candidates[0]

    assert "candidateBlockers" not in candidate
    assert candidate["requiredMetrics"] == ["supplier_inbound"]
    assert "attributed_revenue" not in {
        rule["metric"] for rule in candidate["guardrails"]
    }
    daily_fact = next(
        fact
        for fact in candidate["evidence"]
        if fact["metric"] == "daily_ad_spend_distribution"
    )
    assert "valuesRub" not in daily_fact["value"]
    facts = {fact["metric"]: fact for fact in candidate["evidence"]}
    assert "ad_units" not in facts
    assert (
        facts["ad_spend"]["sourceTable"]
        == "marketing.wb_marketing_campaign_stats_daily"
    )


def test_treating_latest_dates_as_current_does_not_invent_missing_economics() -> None:
    row = {
        **_complete_ad_row(37424786, 180286536, spend=5321),
        "supplier_snapshot_date": None,
        "supplier_inbound_qty": None,
        "attributed_revenue_7d_rub": None,
        "campaign_contribution_profit_7d_rub": None,
        "ad_orders_7d": 2,
    }

    built = build_manager_action_candidates(_snapshot(advertising_rows=[row]), now=NOW)
    candidate = built.candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}

    assert "stale_source_data" not in codes
    assert "incoherent_source_dates" not in codes
    assert {
        "missing_confirmed_supplier_inbound",
        "missing_attributed_revenue",
        "missing_campaign_contribution",
        "unsafe_pause_with_attributed_orders",
        "missing_deterministic_economics",
    } <= codes
    assert "economics" not in candidate


def test_campaign_financial_values_without_source_tables_are_not_publishable() -> None:
    row = {
        **_complete_ad_row(37424786, 180286536),
        "attributed_revenue_source_table": None,
        "campaign_contribution_source_table": None,
    }

    built = build_manager_action_candidates(_snapshot(advertising_rows=[row]), now=NOW)
    candidate = built.candidates[0]
    codes = {item["code"] for item in candidate["candidateBlockers"]}

    assert {
        "missing_attributed_revenue_source",
        "missing_campaign_contribution_source",
        "missing_deterministic_economics",
    } <= codes
    assert "economics" not in candidate


def test_supply_preview_uses_on_hand_and_never_treats_customer_transit_as_inbound():
    row = {
        "report_date": "2026-07-13",
        "nm_id": 70195171,
        "vendor_code": "46227",
        "stock_qty": 0,
        "stock_qty_full": 100,
        "in_way_to_client": 40,
        "in_way_from_client": 60,
        "sales_7d_units": 35,
        "avg_daily_sales_7d": 5,
        "days_of_stock": 0,
        "reorder_flag": "critical",
        "reorder_qty_recommended": 270,
        "unit_cogs_rub": 250,
        "cogs_snapshot_date": "2026-07-13",
        "cogs_retrieved_at": "2026-07-14T08:00:00Z",
        "supplier_snapshot_date": None,
        "supplier_inbound_qty": None,
        "lead_time_days": None,
        "target_cover_days": None,
        "minimum_order_qty": None,
        "replenishment_policy_version": None,
    }

    result = build_manager_action_candidates(_snapshot(supply_rows=[row]), now=NOW)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate["actionId"] == "det-supply-task-70195171"
    assert candidate["executionMode"] == "manual_task"
    assert candidate["executionPayload"]["quantity"] == 270
    assert candidate["executionPayload"]["quantity_source"] == "legacy_supply_overview_preview"
    assert candidate["from"] == {"value": 0, "unit": "units_ordered"}
    assert candidate["to"] == {"value": 270, "unit": "units_ordered"}
    assert "economics" not in candidate
    codes = {item["code"] for item in candidate["candidateBlockers"]}
    assert {
        "legacy_reorder_formula_not_action_safe",
        "missing_confirmed_supplier_inbound",
        "missing_supplier_lead_time",
        "missing_target_cover_policy",
        "missing_minimum_order_quantity",
        "missing_replenishment_policy",
        "missing_deterministic_economics",
    } <= codes
    facts = {item["metric"]: item for item in candidate["evidence"]}
    assert facts["stock_on_hand"]["value"] == 0
    assert "supplier_inbound" not in facts
    assert not any("in_way" in item["metric"] for item in candidate["evidence"])


def test_builder_is_deterministic_and_caps_output_at_seven():
    rows = [
        _complete_ad_row(2000 + index, 70000000 + index, spend=1000 - index)
        for index in range(10)
    ]
    snapshot = _snapshot(advertising_rows=rows)

    first = build_manager_action_candidates(snapshot, now=NOW)
    second = build_manager_action_candidates(snapshot, now=NOW)

    assert first.as_dict() == second.as_dict()
    assert len(first.candidates) == 7
    assert [item["target"] for item in first.candidates] == [
        str(2000 + index) for index in range(7)
    ]
    assert len(first.evidence_facts) == sum(len(item["evidence"]) for item in first.candidates)


def test_blocked_candidate_slots_are_balanced_between_domains(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "shadow")
    advertising = []
    for index in range(10):
        row = _complete_ad_row(4000 + index, 80000000 + index)
        row["ad_orders_7d"] = 1
        advertising.append(row)
    supply = [
        _blocked_supply_row(70195171, "46229", quantity=24),
        _blocked_supply_row(768260062, "46313", quantity=39),
        _blocked_supply_row(171742855, "46303", quantity=12),
    ]

    result = build_manager_action_candidates(
        _snapshot(advertising_rows=advertising, supply_rows=supply),
        now=NOW,
    )

    families = [candidate["family"] for candidate in result.candidates]
    assert len(families) == 7
    assert families.count("advertising") == 4
    assert families.count("supply") == 3


def test_requested_vendor_codes_filter_candidates_through_nm_aliases(monkeypatch):
    monkeypatch.setenv("DECISION_SIMULATION_MODE", "shadow")
    advertising = [
        {**_complete_ad_row(5001, 70195171), "ad_orders_7d": 1},
        {**_complete_ad_row(5002, 768260062), "ad_orders_7d": 1},
    ]
    supply = [
        _blocked_supply_row(70195171, "46229", quantity=24),
        _blocked_supply_row(768260062, "46313", quantity=39),
    ]

    result = build_manager_action_candidates(
        _snapshot(advertising_rows=advertising, supply_rows=supply),
        now=NOW,
        requested_targets=["46229"],
    )

    assert [(item["family"], item["target"]) for item in result.candidates] == [
        ("advertising", "5001"),
        ("supply", "70195171"),
    ]
    assert result.diagnostics[-1]["requestedTargets"] == ["46229"]


def test_equal_positive_bids_across_placements_are_one_exact_bid():
    row = {
        **_complete_ad_row(37424786, 180286536),
        "bid_search_kopecks": 12000,
        "bid_recommendations_kopecks": 12000,
    }

    result = build_manager_action_candidates(_snapshot(advertising_rows=[row]), now=NOW)

    candidate = result.candidates[0]
    codes = {item["code"] for item in candidate.get("candidateBlockers", [])}
    bid = next(item for item in candidate["evidence"] if item["metric"] == "bid")
    assert bid["value"] == 12000
    assert "ambiguous_current_bid_placement" not in codes
