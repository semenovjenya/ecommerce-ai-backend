from __future__ import annotations

import pytest

from commerce_agent.models import AgentStatus, DomainAnalysis, DomainName
from scripts.checkpoint_db import _state_summary


def test_checkpoint_summary_reports_actual_skeleton_statuses_without_raw_state() -> None:
    values = {
        "tenant_id": "private-tenant",
        "messages": ["private-user-message"],
        "agent_statuses": {
            domain.value: AgentStatus(
                domain=domain,
                status="planned",
                detail="private-node-detail",
            )
            for domain in DomainName
        },
        "sales_analysis": DomainAnalysis(
            domain=DomainName.SALES,
            status="not_implemented",
            summary="private-analysis-summary",
        ),
        "data_gaps": ["private-gap"],
        "evidence_matrix": [],
        "action_tasks": [],
        "execution_receipts": [],
    }

    summary = _state_summary(values)

    assert summary["domain_statuses"] == {
        domain.value: "planned" for domain in DomainName
    }
    assert summary["analysis_statuses"]["sales"] == "not_implemented"
    assert summary["analysis_statuses"]["finance"] is None
    assert summary["data_gap_count"] == 1
    assert summary["pending_interrupt"] is False
    assert "private" not in str(summary)
    assert "completed_domains" not in summary


def test_checkpoint_summary_fails_closed_on_unexpected_status() -> None:
    with pytest.raises(RuntimeError, match="invalid sales status"):
        _state_summary({"agent_statuses": {"sales": {"status": "secret payload"}}})
