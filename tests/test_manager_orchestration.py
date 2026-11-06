from types import SimpleNamespace
from unittest.mock import patch

from agi_ecom_app.agent_queue import (
    MAX_OPTIMIZED_DECISION_CROSS_CHECKS,
    MAX_OPTIMIZED_MODEL_EXECUTIONS,
    MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS,
    MANAGER_DOMAIN_REQUIRED_TOOLS,
    OPTIMIZED_REPAIR_EXECUTION_RESERVE,
    OPTIMIZED_ROUTER_EXECUTIONS,
    _advance_v2_optimized_run,
    _is_manager_portfolio_request,
    _manager_domain_coverage_plan,
    _question_to_plan_item,
)
from agi_ecom_app.runner import _domain_tool_policy, _verification_goal_from_item
from agi_ecom_app.schemas import QueryIntent


def test_manager_decision_calls_every_non_primary_domain():
    settings = SimpleNamespace(manager_decision_enabled=True)
    with patch("agi_ecom_app.agent_queue.get_settings", return_value=settings):
        plan = _manager_domain_coverage_plan(
            primary_agent=QueryIntent.SALES,
            max_secondary_agents=4,
        )
    assert plan is not None
    targets = {
        item["target_agent"] for item in plan["recommended_secondary_agents"]
    }
    assert targets == {"finance", "marketing", "price", "supply"}
    assert all(item["required_checks"] for item in plan["recommended_secondary_agents"])
    assert all(item["required_tools"] for item in plan["recommended_secondary_agents"])
    assert "get_finance_sku_report_text" in MANAGER_DOMAIN_REQUIRED_TOOLS[QueryIntent.FINANCE]
    assert "get_marketing_campaign_efficiency_snapshot" in MANAGER_DOMAIN_REQUIRED_TOOLS[
        QueryIntent.MARKETING
    ]
    assert {
        item["target_agent"]: item["required_tools"]
        for item in plan["recommended_secondary_agents"]
    } == {
        intent.value: MANAGER_DOMAIN_REQUIRED_TOOLS[intent]
        for intent in QueryIntent
        if intent != QueryIntent.SALES
    }


def test_manager_domain_coverage_is_disabled_by_default_flag():
    settings = SimpleNamespace(manager_decision_enabled=False)
    with patch("agi_ecom_app.agent_queue.get_settings", return_value=settings):
        assert (
            _manager_domain_coverage_plan(
                primary_agent=QueryIntent.FINANCE,
                max_secondary_agents=4,
            )
            is None
        )


def test_manager_portfolio_scope_excludes_single_sku_yes_no_decision():
    assert _is_manager_portfolio_request(
        query_plan={"decision_kind": "action_portfolio"}
    )
    assert not _is_manager_portfolio_request(
        query_plan={"decision_kind": "single_action"}
    )


def test_manager_required_tools_survive_persisted_question_roundtrip():
    settings = SimpleNamespace(manager_decision_enabled=True)
    with patch("agi_ecom_app.agent_queue.get_settings", return_value=settings):
        plan = _manager_domain_coverage_plan(
            primary_agent=QueryIntent.SALES,
            max_secondary_agents=4,
        )
    item = next(
        candidate
        for candidate in plan["recommended_secondary_agents"]
        if candidate["target_agent"] == QueryIntent.MARKETING.value
    )
    goal = _verification_goal_from_item(item, round_name="round_1")
    restored = _question_to_plan_item(
        SimpleNamespace(
            target_agent=item["target_agent"],
            reason=item["reason"],
            question=item["question"],
            priority=item["priority"],
            verification_goal=goal,
        )
    )
    policy = _domain_tool_policy(
        target=QueryIntent.MARKETING,
        user_query="Составь план действий",
        round_name="round_1",
        question_id=7,
        explicit_required_tools=restored["required_tools"],
    )

    assert policy.required_tools == tuple(MANAGER_DOMAIN_REQUIRED_TOOLS[QueryIntent.MARKETING])
    assert _domain_tool_policy(
        target=QueryIntent.MARKETING,
        user_query="Проверь гипотезу",
        round_name="round_1",
        question_id=8,
    ).required_tools == ()


def test_real_decision_path_uses_the_strict_cross_check_limit():
    trace = SimpleNamespace(summary={"query_mode": "decision"})
    rounds = [
        {
            "round": SimpleNamespace(round_type="primary", status="completed"),
            "questions": [],
        }
    ]
    with patch(
        "agi_ecom_app.agent_queue._create_round_1",
        return_value=True,
    ) as create_round:
        assert _advance_v2_optimized_run(17, trace, rounds) is True

    create_round.assert_called_once_with(
        17,
        trace,
        limit=MAX_OPTIMIZED_DECISION_CROSS_CHECKS,
    )


def test_optimized_budget_accounts_for_router_and_repair_explicitly():
    assert MAX_OPTIMIZED_DECISION_CROSS_CHECKS == 4
    assert MAX_OPTIMIZED_MODEL_EXECUTIONS == 13
    assert MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS == 11
    # Primary + four domain checks + two safety checks + two gap repairs +
    # synthesis use ten rows; one row remains as fail-closed scheduling slack.
    assert MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS >= 1 + 4 + 2 + 2 + 1
    assert (
        MAX_OPTIMIZED_PERSISTED_QUESTION_ROWS
        + OPTIMIZED_ROUTER_EXECUTIONS
        + OPTIMIZED_REPAIR_EXECUTION_RESERVE
        == MAX_OPTIMIZED_MODEL_EXECUTIONS
    )
