from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ExperimentDomain = Literal["pricing", "advertising", "supply", "sales", "finance"]
ExperimentProgramMode = Literal["shadow", "enforced"]
ExperimentProgramStatus = Literal[
    "created",
    "data_audit",
    "proposing",
    "negotiating",
    "simulating",
    "selecting",
    "awaiting_approval",
    "executing",
    "measuring",
    "learning",
    "replanning",
    "exhausted",
    "blocked_data",
    "budget_exhausted",
    "paused",
    "failed",
]
ExperimentEvidenceStatus = Literal[
    "available",
    "missing",
    "stale",
    "conflicting",
    "causally_insufficient",
]
ExperimentCandidateStatus = Literal[
    "unseen",
    "eligible",
    "blocked_data",
    "blocked_budget",
    "ineligible",
    "unsafe",
    "dominated",
    "simulated",
    "selected",
    "running",
    "measured",
    "inconclusive",
    "rejected",
    "superseded",
    "not_controllable",
]
ExperimentMessageType = Literal[
    "proposal",
    "challenge",
    "evidence_request",
    "evidence_response",
    "constraint",
    "proposal_revision",
    "domain_done",
]

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

TERMINAL_CANDIDATE_STATUSES: frozenset[str] = frozenset(
    {
        "blocked_data",
        "blocked_budget",
        "ineligible",
        "unsafe",
        "dominated",
        "measured",
        "inconclusive",
        "rejected",
        "superseded",
        "not_controllable",
    }
)
TERMINAL_PROGRAM_STATUSES: frozenset[str] = frozenset(
    {"exhausted", "blocked_data", "budget_exhausted", "failed"}
)


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


class _ProgramModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ExperimentEvidencePacket(_ProgramModel):
    metric: NonEmptyText
    value: Any = None
    unit: str | None = None
    source_table: str | None = None
    grain: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    data_as_of: datetime | None = None
    max_age_seconds: int | None = Field(default=None, ge=0)
    status: ExperimentEvidenceStatus
    evidence_ref: str | None = None
    lineage: list[str] = Field(default_factory=list)
    definition: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_available_packet(self) -> "ExperimentEvidencePacket":
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("evidence periodEnd cannot be before periodStart")
        if self.status == "available":
            required = {
                "evidenceRef": self.evidence_ref,
                "sourceTable": self.source_table,
                "grain": self.grain,
                "dataAsOf": self.data_as_of,
            }
            missing = [name for name, value in required.items() if value in (None, "")]
            if missing:
                raise ValueError(
                    "available evidence requires " + ", ".join(sorted(missing))
                )
        return self


class ExperimentCandidateSpec(_ProgramModel):
    candidate_key: str | None = None
    domain: ExperimentDomain
    family: NonEmptyText
    operation: NonEmptyText
    target: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str | None = None
    primary_metric: str | None = None
    guardrail_metrics: list[str] = Field(default_factory=list)
    required_metrics: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    causal_design: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    reversibility: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    interference_keys: list[str] = Field(default_factory=list)
    stop_conditions: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_contract_sets(self) -> "ExperimentCandidateSpec":
        for name in (
            "guardrail_metrics",
            "required_metrics",
            "evidence_refs",
            "dependencies",
            "conflicts",
            "interference_keys",
        ):
            values = getattr(self, name)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{_to_camel(name)} cannot contain empty values")
            if len(set(values)) != len(values):
                raise ValueError(f"{_to_camel(name)} cannot contain duplicates")
        return self


class ExperimentProgramCreate(_ProgramModel):
    objective: NonEmptyText = "incremental_contribution_profit"
    mode: ExperimentProgramMode = "shadow"
    target: dict[str, Any] = Field(default_factory=dict)
    domains: list[ExperimentDomain] = Field(min_length=1, max_length=5)
    generator_version: NonEmptyText = "experiment-frontier-v1"
    evidence: list[ExperimentEvidencePacket] = Field(default_factory=list)
    candidates: list[ExperimentCandidateSpec] = Field(min_length=1, max_length=35)
    experiment_budget: dict[str, Any] = Field(default_factory=dict)
    source_chat_id: str | None = None
    source_run_id: int | None = Field(default=None, ge=1)
    source_decision_id: str | None = None

    @model_validator(mode="after")
    def validate_finite_frontier(self) -> "ExperimentProgramCreate":
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("domains cannot contain duplicates")
        unknown_candidate_domains = {
            candidate.domain for candidate in self.candidates
        } - set(self.domains)
        if unknown_candidate_domains:
            raise ValueError(
                "candidate domains must be declared by the program: "
                + ", ".join(sorted(unknown_candidate_domains))
            )
        counts = Counter(candidate.domain for candidate in self.candidates)
        over_limit = {domain: count for domain, count in counts.items() if count > 7}
        if over_limit:
            rendered = ", ".join(
                f"{domain}={count}" for domain, count in sorted(over_limit.items())
            )
            raise ValueError(f"at most seven candidates are allowed per domain: {rendered}")
        return self


class ExperimentNegotiationMessageInput(_ProgramModel):
    message_type: ExperimentMessageType
    source_role: NonEmptyText
    target_role: str | None = None
    correlation_id: str | None = None
    round_id: int | None = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_typed_message(self) -> "ExperimentNegotiationMessageInput":
        if self.message_type == "evidence_response":
            if self.source_role != "data_authority":
                raise ValueError("only data_authority can publish evidence_response")
            if not self.evidence_refs:
                raise ValueError("evidence_response requires evidenceRefs")
        if self.message_type == "evidence_request":
            if self.target_role != "data_authority":
                raise ValueError("evidence_request must target data_authority")
            if not isinstance(self.payload.get("metric"), str) or not self.payload[
                "metric"
            ].strip():
                raise ValueError("evidence_request requires payload.metric")
        if self.message_type == "domain_done":
            allowed = {
                "ready",
                "blocked_data",
                "constraint_only",
                "no_controllable_lever",
                "exhausted",
            }
            coverage_status = self.payload.get("coverageStatus")
            if coverage_status not in allowed:
                raise ValueError(
                    "domain_done requires a supported payload.coverageStatus"
                )
            domain = self.payload.get("domain")
            if domain not in {"pricing", "advertising", "supply", "sales", "finance"}:
                raise ValueError("domain_done requires a supported payload.domain")
        return self


class ExperimentProgramView(_ProgramModel):
    id: str
    objective: str
    mode: ExperimentProgramMode
    status: ExperimentProgramStatus
    target: dict[str, Any]
    domains: list[ExperimentDomain]
    generator_version: str
    snapshot_hash: str
    current_epoch: int
    experiment_budget: dict[str, Any]
    domain_coverage: dict[str, Any]
    source_chat_id: str | None = None
    source_run_id: int | None = None
    source_decision_id: str | None = None
    heartbeat_at: datetime | None = None
    simulation_dispatch_attempts: int = 0
    error: dict[str, Any] = Field(default_factory=dict)
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentSnapshotView(_ProgramModel):
    id: str
    program_id: str
    epoch: int
    snapshot_hash: str
    status: Literal["ready", "blocked_data", "conflicting"]
    data_as_of: datetime | None = None
    evidence_packets: list[dict[str, Any]] = Field(default_factory=list)
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ExperimentDomainFrontierView(_ProgramModel):
    program_id: str
    domain: ExperimentDomain
    epoch: int
    generator_version: str
    snapshot_hash: str
    status: str
    candidate_count: int
    eligible_count: int
    terminal_count: int
    coverage: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentCandidateView(_ProgramModel):
    id: str
    program_id: str
    epoch: int
    candidate_key: str
    domain: ExperimentDomain
    family: str
    operation: str
    target: dict[str, Any]
    parameters: dict[str, Any]
    hypothesis: str | None = None
    primary_metric: str | None = None
    guardrail_metrics: list[str] = Field(default_factory=list)
    required_metrics: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_status: ExperimentEvidenceStatus
    causal_design: dict[str, Any]
    estimated_cost: dict[str, Any]
    risk: dict[str, Any]
    reversibility: dict[str, Any]
    stop_conditions: list[dict[str, Any]] = Field(default_factory=list)
    status: ExperimentCandidateStatus
    score: dict[str, Any]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentNegotiationMessageView(_ProgramModel):
    id: int
    message_id: str
    program_id: str
    round_id: int | None = None
    epoch: int
    message_type: ExperimentMessageType
    source_role: str
    target_role: str | None = None
    correlation_id: str | None = None
    idempotency_key: str
    fingerprint: str
    payload: dict[str, Any]
    evidence_refs: list[str]
    created_at: datetime | None = None


class ExperimentProgramEventView(_ProgramModel):
    id: int
    event_id: str
    program_id: str
    epoch: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime | None = None


class ExperimentProgramBundleView(_ProgramModel):
    program: ExperimentProgramView
    snapshot: ExperimentSnapshotView
    frontiers: list[ExperimentDomainFrontierView]
    candidates: list[ExperimentCandidateView]


class ExperimentNegotiationMessagesResponse(_ProgramModel):
    program_id: str
    items: list[ExperimentNegotiationMessageView]


class ExperimentProgramEventsResponse(_ProgramModel):
    program_id: str
    items: list[ExperimentProgramEventView]


__all__ = [
    "ExperimentCandidateSpec",
    "ExperimentCandidateStatus",
    "ExperimentCandidateView",
    "ExperimentDomain",
    "ExperimentDomainFrontierView",
    "ExperimentEvidencePacket",
    "ExperimentEvidenceStatus",
    "ExperimentMessageType",
    "ExperimentNegotiationMessageInput",
    "ExperimentNegotiationMessageView",
    "ExperimentNegotiationMessagesResponse",
    "ExperimentProgramBundleView",
    "ExperimentProgramCreate",
    "ExperimentProgramEventView",
    "ExperimentProgramEventsResponse",
    "ExperimentProgramMode",
    "ExperimentProgramStatus",
    "ExperimentProgramView",
    "ExperimentSnapshotView",
    "TERMINAL_CANDIDATE_STATUSES",
    "TERMINAL_PROGRAM_STATUSES",
]
