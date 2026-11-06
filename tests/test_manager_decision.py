from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from agi_ecom_app.manager_decision import (
    extract_manager_request_targets,
    manager_action_candidates_from_forecast,
    manager_actions_to_action_board,
    run_manager_decision_workflow,
)


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _query_plan(*targets: str, analyses: list[str] | None = None) -> dict:
    return {
        "query_mode": "decision",
        "scope": "sku_list" if targets else "portfolio",
        "decision_kind": "action_portfolio",
        "targets": [
            {"target_type": "vendor_code", "value": target}
            for target in targets
        ],
        "requested_analyses": analyses or ["advertising"],
    }


def test_generic_sku_matrix_language_is_not_an_explicit_target():
    query = (
        "Построй SKU-матрицу по всей товарной матрице и по проблемным SKU "
        "сравни варианты цены."
    )

    assert extract_manager_request_targets(query, {"query_plan": _query_plan()}) == []


def test_alphanumeric_vendor_code_remains_an_explicit_target():
    assert extract_manager_request_targets(
        "Сравни SKU LC-46303-A",
        {"query_plan": _query_plan("LC-46303-A")},
    ) == ["LC-46303-A"]


AD_METRICS = {
    "campaign_status": ("active", "state"),
    "bid": (420, "kopecks/1000_impressions"),
    "ad_spend": (2200, "RUB"),
    "clicks": (140, "count"),
    "ad_orders": (8, "count"),
    "revenue": (16000, "RUB"),
    "contribution_profit": (3100, "RUB"),
    "cogs": (670, "RUB/unit"),
    "budget_remaining": (9000, "RUB"),
    "stock_on_hand": (84, "units"),
}


def _fact(target: str, metric: str, value, unit: str, *, period_end: str = "2026-07-12", snapshot: str = "2026-07-12T23:00:00Z"):
    return {
        "metric": metric,
        "value": value,
        "unit": unit,
        "target": target,
        "period": f"2026-07-06/{period_end}",
        "sourceTable": f"mart_{metric}",
        "snapshotAt": snapshot,
        "retrievedAt": "2026-07-13T11:55:00Z",
        "definition": f"Canonical definition of {metric}",
        "confidence": "high",
        "evidenceRef": f"{target}:{metric}",
        "sourceAgent": "marketing",
    }


def _ad_candidate(index: int, *, target: str | None = None, to_bid: int | None = None):
    campaign = target or f"campaign-{index}"
    sku = str(46000 + index)
    evidence = [_fact(campaign, metric, value, unit) for metric, (value, unit) in AD_METRICS.items()]
    to_value = to_bid if to_bid is not None else 300 + index
    return {
        "actionId": f"ad-{index}",
        "family": "advertising",
        "target": campaign,
        "targetType": "campaign",
        "targetRefs": [sku],
        "operation": "set_bid",
        "executionMode": "wb_api",
        "executionPayload": {
            "advert_id": 1000 + index,
            "nm_id": int(sku),
            "bid": to_value,
            "placement": "search",
        },
        "from": {"value": 420, "unit": "kopecks/1000_impressions"},
        "to": {"value": to_value, "unit": "kopecks/1000_impressions"},
        "economics": {
            "deltaRevenuePerDay": {
                "low": 120 * index,
                "base": 170 * index,
                "high": 240 * index,
            },
            "deltaCogsPerDay": {"low": 10 * index, "base": 10 * index, "high": 10 * index},
            "deltaCommissionPerDay": {"low": 10 * index, "base": 10 * index, "high": 10 * index},
            "deltaLogisticsPerDay": {"low": 0, "base": 0, "high": 0},
            "deltaStoragePerDay": {"low": 0, "base": 0, "high": 0},
            "deltaAdvertisingPerDay": {"low": 0, "base": 0, "high": 0},
            "actionCostPerDay": {"low": 0, "base": 0, "high": 0},
            "currency": "RUB",
            "horizonDays": 7,
            "calculationMethod": "campaign_matched_control_v1",
            "evidenceRefs": [
                f"{campaign}:contribution_profit",
                f"{campaign}:ad_spend",
                f"{campaign}:cogs",
            ],
            "causalValid": True,
        },
        "evidence": evidence,
        "prerequisites": ["WB preflight confirms current bid and stock"],
        "stopCondition": "Stop if contribution profit is below zero for two completed days",
        "rollback": {
            "operation": "set_bid",
            "from": {"value": to_value, "unit": "kopecks/1000_impressions"},
            "to": {"value": 420, "unit": "kopecks/1000_impressions"},
            "instruction": "Restore the confirmed previous bid",
        },
        "measurementSpec": {
            "primaryMetric": "contribution_profit_rub",
            "baselineDays": 7,
            "evaluationDays": 7,
            "attributionMethod": "observed_change",
            "guardrailMetrics": ["stock_on_hand", "ad_orders"],
        },
        "guardrails": [
            {
                "metric": "stock_on_hand",
                "operator": "gte",
                "threshold": {"value": 21, "unit": "units"},
                "onBreach": "block",
                "evidenceRef": f"{campaign}:stock_on_hand",
            },
            {
                "metric": "contribution_profit",
                "operator": "gte",
                "threshold": {"value": 0, "unit": "RUB"},
                "onBreach": "rollback",
                "evidenceRef": f"{campaign}:contribution_profit",
            },
            {
                "metric": "budget_remaining",
                "operator": "gte",
                "threshold": {"value": 0, "unit": "RUB"},
                "onBreach": "block",
                "evidenceRef": f"{campaign}:budget_remaining",
            },
        ],
        "expiresAt": "2026-07-15T09:00:00Z",
        "urgency": "high",
        "confidence": "high",
    }


def test_ready_decision_has_three_ranked_actions_and_action_board_adapter():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "ready_for_approval"
    assert result.facts_only is False
    assert [action.action_id for action in result.decision.ranked_actions] == ["ad-3", "ad-2", "ad-1"]
    assert "Изменение:" in result.rendered_markdown
    tasks = manager_actions_to_action_board(result.decision)
    assert len(tasks) == 3
    assert tasks[0]["executionPayload"]["advert_id"] == 1003
    assert tasks[0]["managerDecisionId"] == result.decision.decision_id


def test_non_decision_mode_is_facts_only_even_with_complete_candidates():
    result = run_manager_decision_workflow(
        {"actionCandidates": [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]},
        "Отчет по рекламе",
        "report",
        now=NOW,
    )

    assert result.decision.status == "blocked"
    assert result.decision.ranked_actions == []
    assert result.facts_only is True
    assert "non_decision_mode" in {blocker.code for blocker in result.decision.blockers}


def test_missing_economics_is_rejected_without_inventing_a_number():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0].pop("economics")

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "needs_data"
    assert result.decision.ranked_actions == []
    assert any(
        reason.code == "missing_or_invalid_economics"
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    )
    assert "Изменение:" not in result.rendered_markdown


def test_missing_required_metric_fails_closed():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["evidence"] = [
        fact for fact in candidates[0]["evidence"] if fact["metric"] != "ad_spend"
    ]

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "needs_data"
    assert result.decision.ranked_actions == []
    assert any(
        reason.code == "missing_required_metrics"
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    )


def test_campaign_action_accepts_sku_evidence_only_through_target_refs():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    for candidate in candidates:
        sku = candidate["targetRefs"][0]
        for fact in candidate["evidence"]:
            if fact["metric"] in {"stock_on_hand", "cogs"}:
                fact["target"] = sku

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "ready_for_approval"


def test_numeric_target_ids_do_not_match_by_substring():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["targetRefs"] = ["12345"]

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates, "query_plan": _query_plan("1234")},
        "Что делать с рекламой SKU 1234?",
        "decision",
        now=NOW,
    )

    assert result.decision.request_coverage.missing_targets == ["1234"]


def test_efficiency_status_is_not_accepted_as_live_campaign_status():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    status_fact = next(
        fact for fact in candidates[0]["evidence"] if fact["metric"] == "campaign_status"
    )
    status_fact["metric"] = "status"
    status_fact["value"] = "negative_outlier"

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert any(
        reason.code == "missing_required_metrics" and "campaign_status" in reason.message
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    )


def test_non_ready_portfolio_accounts_for_every_valid_candidate():
    result = run_manager_decision_workflow(
        {"actionCandidates": [_ad_candidate(1), _ad_candidate(2)]},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "needs_data"
    assert result.decision.ranked_actions == []
    assert {item.candidate_id for item in result.decision.rejected_candidates} == {
        "ad-1",
        "ad-2",
    }
    assert all(
        any(reason.code == "insufficient_action_portfolio" for reason in item.reasons)
        for item in result.decision.rejected_candidates
    )


def test_rejected_candidate_preview_is_rendered_as_non_executable():
    candidate = _ad_candidate(1)
    candidate["candidateBlockers"] = [
        {
            "code": "missing_campaign_contribution",
            "message": "Нет campaign-level contribution profit.",
            "severity": "needs_data",
        }
    ]

    result = run_manager_decision_workflow(
        {"actionCandidates": [candidate]},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "needs_data"
    assert result.decision.rejected_candidates[0].operation == "set_bid"
    assert result.decision.rejected_candidates[0].from_value is not None
    assert "Возможные действия — не исполнять" in result.rendered_markdown
    assert "campaign-1" in result.rendered_markdown
    assert "Нет campaign-level contribution profit" in result.rendered_markdown


def test_explicit_sku_coverage_fails_closed():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(4)]

    result = run_manager_decision_workflow(
        {
            "actionCandidates": candidates,
            "query_plan": _query_plan("46001", "46002", "46003"),
        },
        "Что делать с рекламой SKU 46001, 46002 и 46003?",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "needs_data"
    assert result.decision.request_coverage.missing_targets == ["46003"]
    assert result.decision.ranked_actions == []


def test_vendor_code_target_coverage_uses_cooccurring_nm_id_alias():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["target"] = "70194686"
    candidates[0]["targetType"] = "sku"
    candidates[0]["targetRefs"] = []
    for fact in candidates[0]["evidence"]:
        fact["targetId"] = "70194686"
        fact["vendorCode"] = "46227"
        fact.pop("target")

    result = run_manager_decision_workflow(
        {
            "actionCandidates": candidates,
            "requestedOutputs": ["actions"],
            "query_plan": _query_plan("46227"),
        },
        "Составь план действий по рекламе для SKU 46227",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "ready_for_approval"
    assert result.decision.request_coverage.covered_targets == ["46227"]
    assert result.decision.request_coverage.missing_targets == []


def test_stale_evidence_rejects_action_and_future_evidence_blocks_decision():
    stale = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    for candidate in stale:
        for fact in candidate["evidence"]:
            fact["snapshotAt"] = "2026-07-01T00:00:00Z"
    stale_result = run_manager_decision_workflow(
        {"actionCandidates": stale},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert stale_result.decision.status == "needs_data"
    assert any(
        reason.code == "incomplete_or_stale_evidence"
        for candidate in stale_result.decision.rejected_candidates
        for reason in candidate.reasons
    )

    future = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    future[0]["evidence"][0]["snapshotAt"] = "2026-07-14T00:00:00Z"
    future_result = run_manager_decision_workflow(
        {"actionCandidates": future},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert future_result.decision.status == "blocked"
    assert "future_evidence" in {blocker.code for blocker in future_result.decision.blockers}


def test_evidence_fact_public_source_aliases_and_future_period_are_enforced():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    for candidate in candidates:
        for fact in candidate["evidence"]:
            fact["targetId"] = fact.pop("target")
            fact["sourceSnapshotAt"] = fact.pop("snapshotAt")
            fact["metricDefinition"] = fact.pop("definition")
    ready = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert ready.decision.status == "ready_for_approval"

    candidates[0]["evidence"][0]["periodEnd"] = "2026-07-14T00:00:00Z"
    candidates[0]["evidence"][0]["period"] = "2026-07-06/2026-07-14"
    future = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert future.decision.status == "blocked"
    assert "future_evidence" in {blocker.code for blocker in future.decision.blockers}


def test_period_mismatch_and_same_target_conflict_are_detected():
    incoherent = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    incoherent[0]["evidence"][0]["period"] = "2026-06-01/2026-06-30"
    mismatch_result = run_manager_decision_workflow(
        {"actionCandidates": incoherent},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert any(
        reason.code == "incoherent_evidence_periods"
        for candidate in mismatch_result.decision.rejected_candidates
        for reason in candidate.reasons
    )

    conflicting = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    conflicting[1] = _ad_candidate(2, target="campaign-1", to_bid=600)
    conflict_result = run_manager_decision_workflow(
        {"actionCandidates": conflicting},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert conflict_result.decision.status == "blocked"
    assert "conflicting_target_actions" in {blocker.code for blocker in conflict_result.decision.blockers}


def test_flow_metrics_with_different_window_lengths_are_rejected() -> None:
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    ad_spend = next(
        fact for fact in candidates[0]["evidence"] if fact["metric"] == "ad_spend"
    )
    ad_spend["period"] = "2026-06-01/2026-07-12"

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert any(
        reason.code == "incoherent_flow_periods"
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    )


def test_family_mismatch_and_missing_wb_payload_are_blocking():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["family"] = "pricing"
    candidates[1]["executionPayload"] = {}

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates, "query_plan": _query_plan()},
        "Какие действия выполнить по рекламе?",
        "decision",
        now=NOW,
    )

    reason_codes = {
        reason.code
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    }
    assert "action_family_mismatch" in reason_codes
    assert "missing_execution_payload" in reason_codes
    assert result.decision.status == "blocked"


def test_forecast_adapter_does_not_promote_point_estimate_to_effect_range():
    forecast = {
        "domain": "marketing",
        "confidence": "high",
        "economicEffectRub": 7000,
        "plan": {"action_family": "stop_or_reduce_campaigns"},
        "resolutionPlan": [
            {
                "stepId": "forecast-1",
                "operation": "set_bid",
                "targetEntityType": "campaign",
                "targetEntityId": "campaign-1",
                "currentMetric": "420 RUB",
                "targetMetric": "300 RUB",
                "stopCondition": "Stop on negative contribution",
                "payload": {"executionPayload": {"advert_id": 1001, "bid": 300}},
            }
        ],
    }

    adapted = manager_action_candidates_from_forecast(forecast)
    assert len(adapted) == 1
    assert adapted[0]["expectedEffect"] is None
    result = run_manager_decision_workflow(
        {"forecastDecision": forecast},
        "Что делать по рекламе?",
        "decision",
        now=NOW,
    )
    assert result.decision.status == "needs_data"
    assert result.decision.ranked_actions == []


def test_candidate_effect_numbers_cannot_override_deterministic_economics():
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["expectedEffect"] = {
        "contributionProfitLowPerDay": 999_999,
        "contributionProfitBasePerDay": 999_999,
        "contributionProfitHighPerDay": 999_999,
        "currency": "RUB",
        "horizonDays": 7,
        "calculationMethod": "llm_assumption",
        "evidenceRefs": ["invented"],
    }

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "ready_for_approval"
    action = next(item for item in result.decision.ranked_actions if item.action_id == "ad-1")
    assert action.expected_effect.contribution_profit_low_per_day == 100
    assert action.expected_effect.contribution_profit_base_per_day == 150
    assert action.expected_effect.contribution_profit_high_per_day == 220


def test_profit_action_without_cogs_and_wb_bid_with_wrong_unit_are_blocked():
    without_cogs = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    without_cogs[0]["evidence"] = [
        fact for fact in without_cogs[0]["evidence"] if fact["metric"] != "cogs"
    ]
    no_cogs_result = run_manager_decision_workflow(
        {"actionCandidates": without_cogs},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert any(
        reason.code == "missing_required_metrics" and "cogs" in reason.message
        for candidate in no_cogs_result.decision.rejected_candidates
        for reason in candidate.reasons
    )

    wrong_unit = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    wrong_unit[0]["from"]["unit"] = "RUB/1000_impressions"
    wrong_unit[0]["to"]["unit"] = "RUB/1000_impressions"
    wrong_unit[0]["rollback"]["from"]["unit"] = "RUB/1000_impressions"
    wrong_unit[0]["rollback"]["to"]["unit"] = "RUB/1000_impressions"
    wrong_unit_result = run_manager_decision_workflow(
        {"actionCandidates": wrong_unit},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )
    assert any(
        reason.code == "invalid_execution_unit"
        for candidate in wrong_unit_result.decision.rejected_candidates
        for reason in candidate.reasons
    )


def test_wb_action_without_complete_live_guardrails_is_blocked() -> None:
    candidates = [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]
    candidates[0]["guardrails"] = candidates[0]["guardrails"][:1]

    result = run_manager_decision_workflow(
        {"actionCandidates": candidates},
        "Составь план действий по рекламе",
        "decision",
        now=NOW,
    )

    assert result.decision.status == "blocked"
    assert any(
        reason.code == "missing_live_guardrails"
        for candidate in result.decision.rejected_candidates
        for reason in candidate.reasons
    )


def test_decision_id_is_stable_for_duplicate_request():
    package = {"run_id": 900, "actionCandidates": [_ad_candidate(1), _ad_candidate(2), _ad_candidate(3)]}
    first = run_manager_decision_workflow(package, "Составь план действий по рекламе", "decision", now=NOW)
    second = run_manager_decision_workflow(deepcopy(package), "Составь план действий по рекламе", "decision", now=NOW)

    assert first.decision.decision_id == second.decision.decision_id
