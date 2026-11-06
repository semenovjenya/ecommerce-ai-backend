from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SimulationCandidateStatus = Literal["completed", "needs_data", "blocked"]
ConstraintSeverity = Literal["needs_data", "blocked"]


class _SimulationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ScalarEvidenceV1(_SimulationModel):
    """One source-addressable scalar used by the pause-campaign hard gate."""

    value: float | None = None
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class DailySpendEvidenceV1(_SimulationModel):
    """Empirical campaign spend for the seven completed source days."""

    values_rub: list[float] | None = Field(default=None, alias="valuesRub")
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AdvertisingPauseEvidenceV1(_SimulationModel):
    daily_spend: DailySpendEvidenceV1 | None = Field(default=None, alias="dailySpend")
    stock_on_hand: ScalarEvidenceV1 | None = Field(default=None, alias="stockOnHand")
    supplier_inbound: ScalarEvidenceV1 | None = Field(default=None, alias="supplierInbound")
    ad_orders: ScalarEvidenceV1 | None = Field(default=None, alias="adOrders")
    attributed_revenue: ScalarEvidenceV1 | None = Field(
        default=None, alias="attributedRevenue"
    )
    contribution_profit: ScalarEvidenceV1 | None = Field(
        default=None, alias="contributionProfit"
    )


class AdvertisingPauseCandidateV1(_SimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    action_id: str = Field(..., min_length=1, alias="actionId")
    operation: Literal["pause_campaign"] = "pause_campaign"
    horizon_days: int = Field(default=7, ge=1, le=365, alias="horizonDays")
    execution_risk: float = Field(default=0.0, ge=0, alias="executionRisk")
    evidence: AdvertisingPauseEvidenceV1

    @field_validator("execution_risk")
    @classmethod
    def execution_risk_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("execution_risk must be finite")
        return value


class DecisionSimulationRequestV1(_SimulationModel):
    snapshot_hash: str = Field(..., min_length=64, max_length=64, alias="snapshotHash")
    candidates: list[AdvertisingPauseCandidateV1] = Field(..., min_length=3, max_length=7)
    model_version: Literal["advertising-bootstrap-v1"] = Field(
        default="advertising-bootstrap-v1", alias="modelVersion"
    )
    sample_count: Literal[500] = Field(default=500, alias="sampleCount")
    batch_size: Literal[50] = Field(default=50, alias="batchSize")

    @field_validator("snapshot_hash")
    @classmethod
    def normalize_snapshot_hash(cls, value: str) -> str:
        normalized = value.lower()
        if any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("snapshot_hash must be a 64-character hexadecimal SHA-256")
        return normalized

    @model_validator(mode="after")
    def candidate_identifiers_must_be_unique(self) -> "DecisionSimulationRequestV1":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        action_ids = [item.action_id for item in self.candidates]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be unique")
        return self


class ConstraintViolationV1(_SimulationModel):
    code: str = Field(..., min_length=1)
    severity: ConstraintSeverity
    message: str = Field(..., min_length=1)
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")


class HistogramSummaryV1(_SimulationModel):
    bin_edges: list[float] = Field(..., min_length=2, alias="binEdges")
    counts: list[int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_histogram_shape(self) -> "HistogramSummaryV1":
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError("histogram must contain exactly one more edge than count")
        if any(count < 0 for count in self.counts):
            raise ValueError("histogram counts cannot be negative")
        return self


class DistributionSummaryV1(_SimulationModel):
    count: int = Field(..., ge=1)
    mean: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    cvar_10: float = Field(..., alias="cvar10")
    probability_positive: float = Field(..., ge=0, le=1, alias="probabilityPositive")
    histogram: HistogramSummaryV1

    @model_validator(mode="after")
    def validate_distribution(self) -> "DistributionSummaryV1":
        numeric_values = (
            self.mean,
            self.p05,
            self.p10,
            self.p25,
            self.p50,
            self.p75,
            self.p90,
            self.p95,
            self.cvar_10,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("distribution statistics must be finite")
        if not self.p05 <= self.p10 <= self.p25 <= self.p50 <= self.p75 <= self.p90 <= self.p95:
            raise ValueError("distribution quantiles must be monotonic")
        if sum(self.histogram.counts) != self.count:
            raise ValueError("histogram counts must equal distribution count")
        return self


class DistributionPairV1(_SimulationModel):
    total: DistributionSummaryV1
    per_day: DistributionSummaryV1 = Field(..., alias="perDay")


class SimulationBatchResultV1(_SimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    batch_index: int = Field(..., ge=0, alias="batchIndex")
    seed: int = Field(..., ge=0)
    sample_count: int = Field(..., ge=1, alias="sampleCount")
    distribution: DistributionPairV1


class RankingScoreBreakdownV1(_SimulationModel):
    cvar_10_per_day: float = Field(..., alias="cvar10PerDay")
    p10_per_day: float = Field(..., alias="p10PerDay")
    mean_per_day: float = Field(..., alias="meanPerDay")
    execution_risk: float = Field(..., ge=0, alias="executionRisk")


class CandidateSimulationResultV1(_SimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    action_id: str = Field(..., min_length=1, alias="actionId")
    status: SimulationCandidateStatus
    constraint_violations: list[ConstraintViolationV1] = Field(
        default_factory=list, alias="constraintViolations"
    )
    evidence_refs: list[str] = Field(default_factory=list, alias="evidenceRefs")
    batch_results: list[SimulationBatchResultV1] = Field(
        default_factory=list, alias="batchResults"
    )
    distribution: DistributionPairV1 | None = None
    score_breakdown: RankingScoreBreakdownV1 | None = Field(
        default=None, alias="scoreBreakdown"
    )
    rank: int | None = Field(default=None, ge=1)
    is_winner: bool = Field(default=False, alias="isWinner")

    @model_validator(mode="after")
    def validate_status_payload(self) -> "CandidateSimulationResultV1":
        if self.status == "completed":
            if self.constraint_violations:
                raise ValueError("completed candidate cannot have constraint violations")
            if self.distribution is None or self.score_breakdown is None or not self.batch_results:
                raise ValueError("completed candidate requires batches, distribution and score")
        elif self.batch_results or self.distribution is not None or self.score_breakdown is not None:
            raise ValueError("non-completed candidate cannot expose numerical results")
        return self


class DecisionSimulationResultV1(_SimulationModel):
    snapshot_hash: str = Field(..., alias="snapshotHash")
    model_version: Literal["advertising-bootstrap-v1"] = Field(..., alias="modelVersion")
    master_seed: int = Field(..., ge=0, alias="masterSeed")
    sample_count: int = Field(..., ge=1, alias="sampleCount")
    batch_size: int = Field(..., ge=1, alias="batchSize")
    candidate_count: int = Field(..., ge=3, le=7, alias="candidateCount")
    completed_candidate_count: int = Field(..., ge=0, alias="completedCandidateCount")
    candidates: list[CandidateSimulationResultV1]
    ranked_candidate_ids: list[str] = Field(default_factory=list, alias="rankedCandidateIds")
    winner_candidate_id: str | None = Field(default=None, alias="winnerCandidateId")


__all__ = [
    "AdvertisingPauseCandidateV1",
    "AdvertisingPauseEvidenceV1",
    "CandidateSimulationResultV1",
    "ConstraintSeverity",
    "ConstraintViolationV1",
    "DailySpendEvidenceV1",
    "DecisionSimulationRequestV1",
    "DecisionSimulationResultV1",
    "DistributionPairV1",
    "DistributionSummaryV1",
    "HistogramSummaryV1",
    "RankingScoreBreakdownV1",
    "ScalarEvidenceV1",
    "SimulationBatchResultV1",
    "SimulationCandidateStatus",
]
