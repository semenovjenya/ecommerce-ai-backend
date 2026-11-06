from __future__ import annotations

from commerce_agent.models import DataGap
from commerce_agent.routing import route_after_evidence_gate, route_after_gap_resolution


def test_blocking_open_gap_enters_single_m1_gap_resolution_pass() -> None:
    assert (
        route_after_evidence_gate(
            {"data_gaps": [DataGap(gap_id="gap-1", reason="missing data")]}
        )
        == "gap_resolution"
    )


def test_resolved_or_absent_gaps_continue_to_synthesis() -> None:
    resolved = DataGap(
        gap_id="gap-1",
        reason="resolved",
        status="resolved",
        revision=2,
    )
    assert route_after_evidence_gate({"data_gaps": [resolved]}) == "synthesis"
    assert route_after_evidence_gate({}) == "synthesis"


def test_deep_research_is_unreachable_in_m1() -> None:
    assert route_after_gap_resolution({}) == "synthesis"
