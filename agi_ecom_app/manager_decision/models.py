from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


DecisionStatus = Literal["evaluating", "ready_for_approval", "needs_data", "blocked"]
SimulationEligibilityOutcome = Literal[
    "not_ready",
    "insufficient",
    "eligible",
    "queued",
]
ExecutionMode = Literal["wb_api", "manual_task"]
ActionFamily = Literal["advertising", "pricing", "supply", "finance", "sales"]
FreshnessStatus = Literal["fresh", "stale", "future", "unknown"]
Confidence = Literal["low", "medium", "high"]
ScalarValue = Union[StrictBool, StrictInt, StrictFloat, StrictStr]


class _PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExactValueV1(_PublicModel):
    value: ScalarValue
    unit: str = Field(..., min_length=1)

    @field_validator("value")
    @classmethod
    def reject_blank_string(cls, value: ScalarValue) -> ScalarValue:
        if isinstance(value, str) and not value.strip():
            raise ValueError("exact value cannot be blank")
        return value


class ManagerEvidenceFactV1(_PublicModel):
    """A source-addressable fact suitable for deterministic action checks."""

    metric: str = Field(..., min_length=1)
    value: Any = None
    unit: str | None = None
    target: str | None = None
    period: str | None = None
    period_start: datetime | None = Field(default=None, alias="periodStart")
    period_end: datetime | None = Field(default=None, alias="periodEnd")
    source_table: str | None = Field(default=None, alias="sourceTable")
    snapshot_at: datetime | None = Field(default=None, alias="snapshotAt")
    retrieved_at: datetime | None = Field(default=None, alias="retrievedAt")
    definition: str | None = None
    freshness_status: FreshnessStatus = Field(default="unknown", alias="freshnessStatus")
    confidence: Confidence = "medium"
    evidence_ref: str = Field(..., min_length=1, alias="evidenceRef")
    source_agent: str | None = Field(default=None, alias="sourceAgent")


class ExpectedEffectV1(_PublicModel):
    """Forecast supplied by deterministic business logic, never generated here."""

    contribution_profit_low_per_day: float = Field(..., alias="contributionProfitLowPerDay")
    contribution_profit_base_per_day: float = Field(..., alias="contributionProfitBasePerDay")
    contribution_profit_high_per_day: float = Field(..., alias="contributionProfitHighPerDay")
    currency: Literal["RUB"] = "RUB"
    horizon_days: int = Field(..., ge=1, le=365, alias="horizonDays")
    calculation_method: str = Field(..., min_length=1, alias="calculationMethod")
    evidence_refs: list[str] = Field(..., min_length=1, alias="evidenceRefs")
    causal_valid: bool = Field(default=False, alias="causalValid")

    @model_validator(mode="after")
    def validate_interval(self) -> "ExpectedEffectV1":
        if not (
            self.contribution_profit_low_per_day
            <= self.contribution_profit_base_per_day
            <= self.contribution_profit_high_per_day
        ):
            raise ValueError("expected effect must satisfy low <= base <= high")
        return self


class EconomicRangeV1(_PublicModel):
    low: float
    base: float
    high: float

    @model_validator(mode="after")
    def validate_interval(self) -> "EconomicRangeV1":
        if not all(math.isfinite(value) for value in (self.low, self.base, self.high)):
            raise ValueError("economic component values must be finite")
        if not self.low <= self.base <= self.high:
            raise ValueError("economic component must satisfy low <= base <= high")
        return self


class ActionEconomicsV1(_PublicModel):
    delta_revenue_per_day: EconomicRangeV1 = Field(..., alias="deltaRevenuePerDay")
    delta_cogs_per_day: EconomicRangeV1 = Field(..., alias="deltaCogsPerDay")
    delta_commission_per_day: EconomicRangeV1 = Field(..., alias="deltaCommissionPerDay")
    delta_logistics_per_day: EconomicRangeV1 = Field(..., alias="deltaLogisticsPerDay")
    delta_storage_per_day: EconomicRangeV1 = Field(..., alias="deltaStoragePerDay")
    delta_advertising_per_day: EconomicRangeV1 = Field(..., alias="deltaAdvertisingPerDay")
    action_cost_per_day: EconomicRangeV1 = Field(..., alias="actionCostPerDay")
    currency: Literal["RUB"] = "RUB"
    horizon_days: int = Field(..., ge=1, le=365, alias="horizonDays")
    calculation_method: str = Field(..., min_length=1, alias="calculationMethod")
    evidence_refs: list[str] = Field(..., min_length=1, alias="evidenceRefs")
    causal_valid: bool = Field(default=False, alias="causalValid")


class MeasurementSpecV1(_PublicModel):
    primary_metric: Literal["contribution_profit_rub"] = Field(
        default="contribution_profit_rub", alias="primaryMetric"
    )
    baseline_days: int = Field(..., ge=1, le=90, alias="baselineDays")
    evaluation_days: int = Field(..., ge=1, le=90, alias="evaluationDays")
    attribution_method: Literal["difference_in_differences", "observed_change"] = Field(
        ..., alias="attributionMethod"
    )
    guardrail_metrics: list[str] = Field(..., min_length=1, alias="guardrailMetrics")
    control_target: str | None = Field(default=None, alias="controlTarget")

    @model_validator(mode="after")
    def validate_control(self) -> "MeasurementSpecV1":
        if self.attribution_method == "difference_in_differences" and not str(
            self.control_target or ""
        ).strip():
            raise ValueError("control_target is required for difference_in_differences")
        return self


class GuardrailRuleV1(_PublicModel):
    metric: str = Field(..., min_length=1)
    operator: Literal["gt", "gte", "lt", "lte", "eq", "neq"]
    threshold: ExactValueV1
    on_breach: Literal["block", "rollback", "pause", "hold"] = Field(..., alias="onBreach")
    evidence_ref: str = Field(..., min_length=1, alias="evidenceRef")


class RollbackSpecV1(_PublicModel):
    operation: str = Field(..., min_length=1)
    from_value: ExactValueV1 = Field(..., alias="from")
    to_value: ExactValueV1 = Field(..., alias="to")
    instruction: str = Field(..., min_length=1)


class ManagerActionV1(_PublicModel):
    action_id: str = Field(..., min_length=1, alias="actionId")
    family: ActionFamily
    target: str = Field(..., min_length=1)
    target_type: Literal["sku", "campaign", "portfolio", "task"] = Field(..., alias="targetType")
    target_refs: list[str] = Field(default_factory=list, alias="targetRefs")
    operation: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = Field(..., alias="executionMode")
    execution_payload: dict[str, Any] = Field(..., alias="executionPayload")
    from_value: ExactValueV1 = Field(..., alias="from")
    to_value: ExactValueV1 = Field(..., alias="to")
    economics: ActionEconomicsV1
    expected_effect: ExpectedEffectV1 = Field(..., alias="expectedEffect")
    evidence: list[ManagerEvidenceFactV1] = Field(..., min_length=1)
    required_metrics: list[str] = Field(default_factory=list, alias="requiredMetrics")
    prerequisites: list[str] = Field(..., min_length=1)
    stop_condition: str = Field(..., min_length=1, alias="stopCondition")
    rollback: RollbackSpecV1
    measurement_spec: MeasurementSpecV1 = Field(..., alias="measurementSpec")
    guardrails: list[GuardrailRuleV1] = Field(..., min_length=1)
    expires_at: datetime = Field(..., alias="expiresAt")
    urgency: Literal["low", "medium", "high", "critical"] = "medium"
    confidence: Confidence = "medium"

    @model_validator(mode="after")
    def validate_exact_change(self) -> "ManagerActionV1":
        if self.from_value.unit != self.to_value.unit:
            raise ValueError("from/to units must match")
        if self.from_value.value == self.to_value.value:
            raise ValueError("from/to values must differ")
        return self


class DecisionBlockerV1(_PublicModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    action_id: str | None = Field(default=None, alias="actionId")
    target: str | None = None
    severity: Literal["needs_data", "blocked"] = "needs_data"


class RejectedCandidateV1(_PublicModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    family: str | None = None
    target: str | None = None
    target_refs: list[str] = Field(default_factory=list, alias="targetRefs")
    operation: str | None = None
    execution_mode: ExecutionMode | None = Field(default=None, alias="executionMode")
    from_value: ExactValueV1 | None = Field(default=None, alias="from")
    to_value: ExactValueV1 | None = Field(default=None, alias="to")
    evidence_refs: list[str] = Field(default_factory=list, alias="evidenceRefs")
    reasons: list[DecisionBlockerV1] = Field(..., min_length=1)


class RequestCoverageV1(_PublicModel):
    explicit_targets: list[str] = Field(default_factory=list, alias="explicitTargets")
    covered_targets: list[str] = Field(default_factory=list, alias="coveredTargets")
    missing_targets: list[str] = Field(default_factory=list, alias="missingTargets")
    requested_outputs: list[str] = Field(default_factory=list, alias="requestedOutputs")
    covered_outputs: list[str] = Field(default_factory=list, alias="coveredOutputs")
    missing_outputs: list[str] = Field(default_factory=list, alias="missingOutputs")
    complete: bool = False


class SimulationEligibilityV1(_PublicModel):
    """Observable result of the simulation scheduling eligibility check.

    This is deliberately part of the durable ManagerDecision payload rather
    than the simulation-run row: an ineligible decision has no run, but API
    clients still need to distinguish "not ready" from "too few candidates".
    """

    outcome: SimulationEligibilityOutcome
    mode: Literal["shadow", "enforced"]
    submitted_candidate_count: int = Field(
        ..., ge=0, le=7, alias="submittedCandidateCount"
    )
    eligible_candidate_count: int = Field(
        ..., ge=0, le=7, alias="eligibleCandidateCount"
    )
    rejected_candidate_count: int = Field(
        ..., ge=0, le=7, alias="rejectedCandidateCount"
    )
    minimum_candidate_count: Literal[3] = Field(
        default=3, alias="minimumCandidateCount"
    )
    maximum_candidate_count: Literal[7] = Field(
        default=7, alias="maximumCandidateCount"
    )
    reason_code: str = Field(..., min_length=1, alias="reasonCode")

    @model_validator(mode="after")
    def validate_candidate_counts(self) -> "SimulationEligibilityV1":
        if self.eligible_candidate_count > self.submitted_candidate_count:
            raise ValueError("eligible candidates cannot exceed submitted candidates")
        if self.rejected_candidate_count != (
            self.submitted_candidate_count - self.eligible_candidate_count
        ):
            raise ValueError("rejected candidates must equal submitted minus eligible")
        if self.outcome in {"eligible", "queued"} and self.eligible_candidate_count < 3:
            raise ValueError(
                f"{self.outcome} eligibility requires at least three candidates"
            )
        if self.outcome == "insufficient" and self.eligible_candidate_count >= 3:
            raise ValueError("insufficient eligibility requires fewer than three candidates")
        return self


class ManagerDecisionV1(_PublicModel):
    schema_version: Literal["1.0"] = Field(default="1.0", alias="schemaVersion")
    decision_id: str = Field(..., min_length=1, alias="decisionId")
    query_mode: Literal["decision", "diagnostic", "report", "lookup"] = Field(..., alias="queryMode")
    objective: Literal["incremental_contribution_profit"] = "incremental_contribution_profit"
    objective_formula: str = Field(
        default=(
            "delta_revenue-delta_cogs-delta_commission-delta_logistics-"
            "delta_storage-delta_advertising-action_cost"
        ),
        alias="objectiveFormula",
    )
    status: DecisionStatus
    created_at: datetime = Field(..., alias="createdAt")
    evidence_as_of: datetime | None = Field(default=None, alias="evidenceAsOf")
    request_coverage: RequestCoverageV1 = Field(..., alias="requestCoverage")
    ranked_actions: list[ManagerActionV1] = Field(default_factory=list, max_length=7, alias="rankedActions")
    rejected_candidates: list[RejectedCandidateV1] = Field(default_factory=list, alias="rejectedCandidates")
    blockers: list[DecisionBlockerV1] = Field(default_factory=list)
    evidence_snapshot: list[ManagerEvidenceFactV1] = Field(default_factory=list, alias="evidenceSnapshot")
    simulation_eligibility: SimulationEligibilityV1 | None = Field(
        default=None,
        alias="simulationEligibility",
    )

    @model_validator(mode="after")
    def validate_ready_portfolio(self) -> "ManagerDecisionV1":
        if self.status == "ready_for_approval":
            if not 3 <= len(self.ranked_actions) <= 7:
                raise ValueError("ready decision must contain 3-7 ranked actions")
            if not self.request_coverage.complete:
                raise ValueError("ready decision requires complete request coverage")
            if self.blockers:
                raise ValueError("ready decision cannot contain blockers")
        elif self.ranked_actions:
            raise ValueError("non-ready decision must not expose actionable recommendations")
        return self


class ManagerDecisionWorkflowResult(_PublicModel):
    decision: ManagerDecisionV1
    rendered_markdown: str = Field(..., alias="renderedMarkdown")
    facts_only: bool = Field(..., alias="factsOnly")


__all__ = [
    "ActionEconomicsV1",
    "ActionFamily",
    "Confidence",
    "DecisionBlockerV1",
    "DecisionStatus",
    "ExactValueV1",
    "ExecutionMode",
    "EconomicRangeV1",
    "ExpectedEffectV1",
    "FreshnessStatus",
    "GuardrailRuleV1",
    "ManagerActionV1",
    "ManagerDecisionV1",
    "ManagerDecisionWorkflowResult",
    "ManagerEvidenceFactV1",
    "MeasurementSpecV1",
    "RejectedCandidateV1",
    "RequestCoverageV1",
    "RollbackSpecV1",
    "SimulationEligibilityOutcome",
    "SimulationEligibilityV1",
]
