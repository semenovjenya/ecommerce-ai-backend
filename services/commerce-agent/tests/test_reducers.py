from __future__ import annotations

import pytest

from commerce_agent.models import AgentStatus, DataGap, DomainAnalysis, DomainName, Incident
from commerce_agent.reducers import (
    StateConflictError,
    merge_agent_statuses,
    merge_domain_analysis,
    merge_gaps,
    merge_immutable_items,
    merge_sales_analysis,
    schema_version_reducer,
    set_once_equal,
)


def test_schema_version_accepts_runtime_zero_then_only_version_one() -> None:
    assert schema_version_reducer(0, 1) == 1
    assert schema_version_reducer(1, 1) == 1
    with pytest.raises(StateConflictError):
        schema_version_reducer(1, 2)


def test_identity_is_set_once_and_must_remain_equal() -> None:
    assert set_once_equal("", "tenant-1") == "tenant-1"
    assert set_once_equal("tenant-1", "tenant-1") == "tenant-1"
    with pytest.raises(StateConflictError, match="cannot change"):
        set_once_equal("tenant-1", "tenant-2")


def test_parallel_agent_statuses_merge_without_lost_updates() -> None:
    sales = AgentStatus(domain=DomainName.SALES, status="running", attempt=1)
    finance = AgentStatus(domain=DomainName.FINANCE, status="running", attempt=1)

    left_then_right = merge_agent_statuses(
        merge_agent_statuses({}, {"sales": sales}), {"finance": finance}
    )
    right_then_left = merge_agent_statuses(
        merge_agent_statuses({}, {"finance": finance}), {"sales": sales}
    )
    assert left_then_right == right_then_left
    assert set(left_then_right) == {"sales", "finance"}


def test_status_revisions_are_monotonic_and_conflicts_are_rejected() -> None:
    current = {
        "sales": AgentStatus(domain=DomainName.SALES, status="running", attempt=1)
    }
    completed = AgentStatus(domain=DomainName.SALES, status="completed", attempt=2)
    assert merge_agent_statuses(current, {"sales": completed})["sales"] == completed

    conflicting = AgentStatus(
        domain=DomainName.SALES,
        status="failed",
        attempt=1,
    )
    with pytest.raises(StateConflictError, match="conflicting status"):
        merge_agent_statuses(current, {"sales": conflicting})


def test_domain_analysis_requires_same_owner_and_higher_revision() -> None:
    first = DomainAnalysis(
        domain=DomainName.SALES,
        revision=1,
        status="partial",
        summary="first",
    )
    second = first.model_copy(update={"revision": 2, "summary": "second"})
    assert merge_domain_analysis(first, second) == second

    wrong_owner = DomainAnalysis(
        domain=DomainName.FINANCE,
        revision=2,
        status="partial",
        summary="wrong owner",
    )
    with pytest.raises(StateConflictError, match="owner"):
        merge_domain_analysis(first, wrong_owner)


def test_field_specific_domain_reducer_rejects_wrong_initial_owner() -> None:
    wrong_owner = DomainAnalysis(
        domain=DomainName.FINANCE,
        revision=1,
        status="partial",
        summary="cannot enter sales field",
    )
    with pytest.raises(StateConflictError, match="sales_analysis"):
        merge_sales_analysis(None, wrong_owner)


def test_immutable_collections_deduplicate_by_stable_id() -> None:
    incident = Incident(incident_id="incident-1", severity="warning", summary="x")
    assert merge_immutable_items([incident], [incident]) == [incident]

    changed = incident.model_copy(update={"summary": "changed"})
    with pytest.raises(StateConflictError, match="changed"):
        merge_immutable_items([incident], [changed])


def test_gap_revisions_are_monotonic_and_order_is_stable() -> None:
    gap_a = DataGap(gap_id="a", reason="missing A")
    gap_b = DataGap(gap_id="b", reason="missing B")
    resolved = gap_a.model_copy(update={"revision": 2, "status": "resolved"})

    merged = merge_gaps([gap_b, gap_a], [resolved])
    assert [gap.gap_id for gap in merged] == ["a", "b"]
    assert merged[0] == resolved
