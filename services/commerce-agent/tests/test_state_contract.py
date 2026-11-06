from __future__ import annotations

import pytest
from pydantic import ValidationError

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.state import CommerceStateSnapshot, STATE_FIELD_NAMES


EXPECTED_STATE_FIELDS = {
    "schema_version",
    "messages",
    "tenant_id",
    "user_id",
    "request_id",
    "query_plan",
    "agent_statuses",
    "sales_analysis",
    "marketing_analysis",
    "supply_analysis",
    "price_analysis",
    "finance_analysis",
    "research_result",
    "detected_incidents",
    "evidence_matrix",
    "data_gaps",
    "action_tasks",
    "confidence",
    "manager_decision",
    "pending_interrupt",
    "execution_receipts",
    "errors",
}


def test_durable_state_field_names_are_frozen_for_schema_v1() -> None:
    assert STATE_FIELD_NAMES == EXPECTED_STATE_FIELDS
    assert set(CommerceStateSnapshot.model_fields) == EXPECTED_STATE_FIELDS


def test_snapshot_rejects_unknown_fields_and_versions() -> None:
    base = {
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "request_id": "request-1",
    }
    with pytest.raises(ValidationError):
        CommerceStateSnapshot.model_validate({**base, "unknown": True})
    with pytest.raises(ValidationError):
        CommerceStateSnapshot.model_validate({**base, "schema_version": 2})


def test_runtime_context_requires_verified_identity_and_immutable_roles() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        CommerceRuntimeContext(
            tenant_id="",
            user_id="user-1",
            roles=("manager",),
            request_id="request-1",
        )

    context = CommerceRuntimeContext(
        tenant_id="tenant-1",
        user_id="user-1",
        roles=("manager",),
        request_id="request-1",
    )
    assert context.roles == ("manager",)
    with pytest.raises(ValidationError):
        CommerceRuntimeContext(
            tenant_id=123,
            user_id="user-1",
            roles=("manager",),
            request_id="request-1",
        )
    with pytest.raises(ValidationError):
        CommerceRuntimeContext(
            tenant_id="tenant-1",
            user_id="user-1",
            roles=("manager",),
            request_id="request-1",
            unexpected="forbidden",
        )
    with pytest.raises(ValidationError):
        context.tenant_id = "tenant-2"
