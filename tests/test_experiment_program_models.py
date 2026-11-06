from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agi_ecom_app.experiment_program.models import (
    ExperimentCandidateSpec,
    ExperimentEvidencePacket,
    ExperimentNegotiationMessageInput,
    ExperimentProgramCreate,
)
from agi_ecom_app.experiment_program.service import (
    ExperimentProgramContractError,
    canonical_contract_hash,
)


NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _candidate(index: int, *, domain: str = "pricing") -> ExperimentCandidateSpec:
    return ExperimentCandidateSpec(
        candidateKey=f"{domain}-{index}",
        domain=domain,
        family=domain,
        operation="set_price" if domain == "pricing" else "pause_campaign",
        target={"nmId": 123},
        parameters={"value": index},
    )


def test_program_rejects_more_than_seven_candidates_per_domain() -> None:
    with pytest.raises(ValidationError, match="at most seven candidates"):
        ExperimentProgramCreate(
            domains=["pricing"],
            candidates=[_candidate(index) for index in range(8)],
        )


def test_program_rejects_candidate_from_an_undeclared_domain() -> None:
    with pytest.raises(ValidationError, match="must be declared"):
        ExperimentProgramCreate(
            domains=["pricing"],
            candidates=[_candidate(1, domain="advertising")],
        )


def test_available_evidence_requires_reproducible_provenance() -> None:
    with pytest.raises(ValidationError, match="available evidence requires"):
        ExperimentEvidencePacket(metric="orders", value=0, status="available")

    packet = ExperimentEvidencePacket(
        metric="orders",
        value=0,
        status="available",
        evidenceRef="agent_marts.orders:123:2026-07-20",
        sourceTable="agent_marts.orders",
        grain="nm_id/day",
        dataAsOf=NOW,
    )
    assert packet.value == 0


def test_only_data_authority_can_publish_evidence_response() -> None:
    with pytest.raises(ValidationError, match="only data_authority"):
        ExperimentNegotiationMessageInput(
            messageType="evidence_response",
            sourceRole="pricing",
            payload={"metric": "orders"},
            evidenceRefs=["ref-1"],
        )


def test_evidence_request_and_domain_done_are_typed() -> None:
    with pytest.raises(ValidationError, match="must target data_authority"):
        ExperimentNegotiationMessageInput(
            messageType="evidence_request",
            sourceRole="pricing",
            targetRole="finance",
            payload={"metric": "unit_margin"},
        )
    with pytest.raises(ValidationError, match="coverageStatus"):
        ExperimentNegotiationMessageInput(
            messageType="domain_done",
            sourceRole="pricing",
            payload={"domain": "pricing", "coverageStatus": "maybe"},
        )


def test_contract_hash_is_order_independent_and_rejects_non_finite_values() -> None:
    assert canonical_contract_hash({"b": 2, "a": 1}) == canonical_contract_hash(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ExperimentProgramContractError, match="NaN"):
        canonical_contract_hash({"value": float("nan")})
