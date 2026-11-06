from __future__ import annotations

from typing import Literal

from commerce_agent.state import CommerceState


PostEvidenceRoute = Literal["gap_resolution", "synthesis"]
PostGapRoute = Literal["deep_research", "synthesis"]


def route_after_evidence_gate(state: CommerceState) -> PostEvidenceRoute:
    """Expose the future gap branch without implementing the M4 gap loop."""

    has_blocking_gap = any(
        gap.blocking and gap.status == "open" for gap in state.get("data_gaps", [])
    )
    return "gap_resolution" if has_blocking_gap else "synthesis"


def route_after_gap_resolution(_: CommerceState) -> PostGapRoute:
    """Deep research remains disabled until the bounded M5 capability exists."""

    return "synthesis"


__all__ = ["route_after_evidence_gate", "route_after_gap_resolution"]
