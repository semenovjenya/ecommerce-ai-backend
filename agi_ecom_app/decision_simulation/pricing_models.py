from __future__ import annotations

from datetime import date
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agi_ecom_app.decision_simulation.models import (
    ConstraintViolationV1,
    DistributionPairV1,
    SimulationCandidateStatus,
)


PricingOperation = Literal["hold_price", "set_price"]


class _PricingSimulationModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class PricingCompletedDayObservationV1(_PricingSimulationModel):
    """One completed SKU day without an observed daily OOS signal."""

    observed_date: date = Field(..., alias="observedDate")
    price_rub: float = Field(..., gt=0, alias="priceRub")
    units_sold: float = Field(..., ge=0, alias="unitsSold")
    completed: bool = True
    not_observed_out_of_stock: bool = Field(
        default=True, alias="notObservedOutOfStock"
    )
    context_values: dict[str, float] | None = Field(
        default=None,
        alias="contextValues",
        description=(
            "Optional point-in-time demand controls available for the observation day. "
            "Missing controls stay missing and are never coerced to observed zero."
        ),
    )
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PricingDailyObservationsEvidenceV1(_PricingSimulationModel):
    values: list[PricingCompletedDayObservationV1] | None = None
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")
    latest_observation_age_days: int | None = Field(
        default=None, ge=0, alias="latestObservationAgeDays"
    )

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def observation_dates_must_be_unique(self) -> "PricingDailyObservationsEvidenceV1":
        if self.values is None:
            return self
        dates = [item.observed_date for item in self.values]
        if len(dates) != len(set(dates)):
            raise ValueError("daily observation dates must be unique")
        return self


class PricingElasticityEvidenceV1(_PricingSimulationModel):
    """Empirical elasticity draws produced by an experiment or fitted model."""

    bootstrap_values: list[float] | None = Field(default=None, alias="bootstrapValues")
    source: str | None = None
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", "source", mode="before")
    @classmethod
    def blank_string_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PricingRealizationEvidenceV1(_PricingSimulationModel):
    """Empirical order-to-realized-sale conversion shared across prices."""

    bootstrap_values: list[float] | None = Field(default=None, alias="bootstrapValues")
    source: str | None = None
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", "source", mode="before")
    @classmethod
    def blank_string_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PricingUnitEconomicsEvidenceV1(_PricingSimulationModel):
    """Price-to-contribution bridge shared by every candidate in the SKU group."""

    net_revenue_rate: float | None = Field(default=None, alias="netRevenueRate")
    variable_cost_per_unit_rub: float | None = Field(
        default=None, alias="variableCostPerUnitRub"
    )
    minimum_unit_contribution_rub: float = Field(
        default=0.0, alias="minimumUnitContributionRub"
    )
    observation_count: int | None = Field(default=None, ge=0, alias="observationCount")
    latest_observation_age_days: int | None = Field(
        default=None, ge=0, alias="latestObservationAgeDays"
    )
    basis: str | None = None
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PricingAvailableStockEvidenceV1(_PricingSimulationModel):
    """Maximum sellable units over the simulation horizon."""

    units: float | None = None
    snapshot_date: date | None = Field(default=None, alias="snapshotDate")
    age_days_at_snapshot: int | None = Field(
        default=None, ge=0, alias="ageDaysAtSnapshot"
    )
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")
    future_inbound: list["PricingFutureInboundEventV1"] = Field(
        default_factory=list,
        alias="futureInbound",
    )

    @field_validator("evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class PricingFutureInboundEventV1(_PricingSimulationModel):
    """One supplier inbound event; customer return/transit is not accepted here."""

    eta_date: date = Field(..., alias="etaDate")
    units: float = Field(..., gt=0)
    include_in_capacity: bool = Field(default=True, alias="includeInCapacity")
    status: Literal["scheduled", "overdue", "unconfirmed", "cancelled"] = "scheduled"
    supply_id: str | None = Field(default=None, alias="supplyId")
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")

    @field_validator("evidence_ref", "supply_id", mode="before")
    @classmethod
    def blank_inbound_text_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


PricingContextTransform = Literal["identity", "log1p"]
PricingContextFeatureStatus = Literal["modeled", "observed_only", "missing"]


class PricingContextFeatureEvidenceV1(_PricingSimulationModel):
    """A versioned contextual demand control fitted beside price elasticity."""

    name: str = Field(..., min_length=1)
    transform: PricingContextTransform = "identity"
    status: PricingContextFeatureStatus
    observed_days: int = Field(default=0, ge=0, alias="observedDays")
    coverage_ratio: float = Field(default=0.0, ge=0, le=1, alias="coverageRatio")
    center: float | None = None
    scale: float | None = Field(default=None, gt=0)
    bootstrap_coefficients: list[float] | None = Field(
        default=None,
        alias="bootstrapCoefficients",
    )
    forward_values: list[float | None] = Field(
        default_factory=list,
        alias="forwardValues",
    )
    latest_value: float | None = Field(default=None, alias="latestValue")
    evidence_ref: str | None = Field(default=None, alias="evidenceRef")
    data_gap: str | None = Field(default=None, alias="dataGap")

    @field_validator("evidence_ref", "data_gap", mode="before")
    @classmethod
    def blank_context_text_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def modeled_feature_requires_coefficients(self) -> "PricingContextFeatureEvidenceV1":
        if self.status == "modeled":
            if self.center is None or self.scale is None or not self.bootstrap_coefficients:
                raise ValueError(
                    "modeled context feature requires center, scale and bootstrap coefficients"
                )
        elif self.bootstrap_coefficients is not None:
            raise ValueError(
                "non-modeled context feature cannot expose bootstrap coefficients"
            )
        return self


class PricingForecastContextEvidenceV1(_PricingSimulationModel):
    """Pinned future context and feature-coverage report for a shadow forecast."""

    schema_version: Literal["pricing-context-controls-v1"] = Field(
        default="pricing-context-controls-v1",
        alias="schemaVersion",
    )
    snapshot_date: date = Field(..., alias="snapshotDate")
    features: list[PricingContextFeatureEvidenceV1] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list, alias="dataGaps")
    shadow_only: bool = Field(default=True, alias="shadowOnly")

    @model_validator(mode="after")
    def feature_names_are_unique(self) -> "PricingForecastContextEvidenceV1":
        names = [item.name for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("context feature names must be unique")
        return self


class PricingSimulationEvidenceV1(_PricingSimulationModel):
    daily_observations: PricingDailyObservationsEvidenceV1 = Field(
        ..., alias="dailyObservations"
    )
    elasticity: PricingElasticityEvidenceV1
    realization: PricingRealizationEvidenceV1
    unit_economics: PricingUnitEconomicsEvidenceV1 = Field(..., alias="unitEconomics")
    available_stock: PricingAvailableStockEvidenceV1 = Field(..., alias="availableStock")
    forecast_context: PricingForecastContextEvidenceV1 | None = Field(
        default=None,
        alias="forecastContext",
    )


class PricingCandidateV1(_PricingSimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    action_id: str = Field(..., min_length=1, alias="actionId")
    operation: PricingOperation
    price_rub: float = Field(..., gt=0, alias="priceRub")


class PricingSimulationRequestV1(_PricingSimulationModel):
    snapshot_hash: str = Field(..., min_length=64, max_length=64, alias="snapshotHash")
    decision_group_id: str = Field(..., min_length=1, alias="decisionGroupId")
    sku_id: str = Field(..., min_length=1, alias="skuId")
    current_price_rub: float = Field(..., gt=0, alias="currentPriceRub")
    current_price_evidence_ref: str | None = Field(
        default=None, alias="currentPriceEvidenceRef"
    )
    horizon_days: int = Field(default=7, ge=1, le=90, alias="horizonDays")
    candidates: list[PricingCandidateV1] = Field(..., min_length=3, max_length=7)
    evidence: PricingSimulationEvidenceV1
    model_version: Literal["pricing-elasticity-bootstrap-v1"] = Field(
        default="pricing-elasticity-bootstrap-v1", alias="modelVersion"
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

    @field_validator("current_price_evidence_ref", mode="before")
    @classmethod
    def blank_reference_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_candidate_group(self) -> "PricingSimulationRequestV1":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        action_ids = [item.action_id for item in self.candidates]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be unique")
        prices = [item.price_rub for item in self.candidates]
        if len(prices) != len(set(prices)):
            raise ValueError("candidate prices must be unique within a decision group")

        hold_candidates = [
            item for item in self.candidates if item.operation == "hold_price"
        ]
        if len(hold_candidates) != 1:
            raise ValueError("a pricing decision group requires exactly one hold candidate")
        if not math.isclose(
            hold_candidates[0].price_rub,
            self.current_price_rub,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("hold candidate price must equal current_price_rub")
        if any(
            item.operation == "set_price"
            and math.isclose(
                item.price_rub,
                self.current_price_rub,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for item in self.candidates
        ):
            raise ValueError("set_price candidate cannot duplicate the current price")

        context = self.evidence.forecast_context
        if context is not None:
            elasticity_draws = self.evidence.elasticity.bootstrap_values or []
            for feature in context.features:
                if (
                    feature.status == "modeled"
                    and len(feature.forward_values) != self.horizon_days
                ):
                    raise ValueError(
                        "modeled context forward values must match the pricing horizon"
                    )
                if (
                    feature.bootstrap_coefficients is not None
                    and len(feature.bootstrap_coefficients) != len(elasticity_draws)
                ):
                    raise ValueError(
                        "context coefficient draws must align with elasticity draws"
                    )
        return self


class PricingSimulationBatchResultV1(_PricingSimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    batch_index: int = Field(..., ge=0, alias="batchIndex")
    seed: int = Field(..., ge=0)
    sample_count: int = Field(..., ge=1, alias="sampleCount")
    distribution: DistributionPairV1
    units_sold_distribution: DistributionPairV1 = Field(
        ..., alias="unitsSoldDistribution"
    )
    stockout_probability: float = Field(..., ge=0, le=1, alias="stockoutProbability")


class PricingRankingScoreV1(_PricingSimulationModel):
    cvar_10_per_day: float = Field(..., alias="cvar10PerDay")
    p10_per_day: float = Field(..., alias="p10PerDay")
    mean_per_day: float = Field(..., alias="meanPerDay")
    absolute_price_move_rub: float = Field(..., ge=0, alias="absolutePriceMoveRub")


class PricingCandidateSimulationResultV1(_PricingSimulationModel):
    candidate_id: str = Field(..., min_length=1, alias="candidateId")
    action_id: str = Field(..., min_length=1, alias="actionId")
    operation: PricingOperation
    price_rub: float = Field(..., gt=0, alias="priceRub")
    status: SimulationCandidateStatus
    constraint_violations: list[ConstraintViolationV1] = Field(
        default_factory=list, alias="constraintViolations"
    )
    evidence_refs: list[str] = Field(default_factory=list, alias="evidenceRefs")
    batch_results: list[PricingSimulationBatchResultV1] = Field(
        default_factory=list, alias="batchResults"
    )
    distribution: DistributionPairV1 | None = None
    units_sold_distribution: DistributionPairV1 | None = Field(
        default=None, alias="unitsSoldDistribution"
    )
    stockout_probability: float | None = Field(
        default=None, ge=0, le=1, alias="stockoutProbability"
    )
    unit_contribution_rub: float | None = Field(
        default=None, alias="unitContributionRub"
    )
    score_breakdown: PricingRankingScoreV1 | None = Field(
        default=None, alias="scoreBreakdown"
    )
    rank: int | None = Field(default=None, ge=1)
    is_winner: bool = Field(default=False, alias="isWinner")

    @model_validator(mode="after")
    def validate_status_payload(self) -> "PricingCandidateSimulationResultV1":
        numerical_payload = (
            bool(self.batch_results)
            or self.distribution is not None
            or self.units_sold_distribution is not None
            or self.stockout_probability is not None
            or self.unit_contribution_rub is not None
            or self.score_breakdown is not None
        )
        if self.status == "completed":
            if self.constraint_violations:
                raise ValueError("completed candidate cannot have constraint violations")
            if (
                not self.batch_results
                or self.distribution is None
                or self.units_sold_distribution is None
                or self.stockout_probability is None
                or self.unit_contribution_rub is None
                or self.score_breakdown is None
            ):
                raise ValueError("completed candidate requires batches and numerical summaries")
        elif numerical_payload:
            raise ValueError("non-completed candidate cannot expose numerical results")
        return self


class PricingSimulationResultV1(_PricingSimulationModel):
    snapshot_hash: str = Field(..., alias="snapshotHash")
    decision_group_id: str = Field(..., alias="decisionGroupId")
    sku_id: str = Field(..., alias="skuId")
    model_version: Literal["pricing-elasticity-bootstrap-v1"] = Field(
        ..., alias="modelVersion"
    )
    master_seed: int = Field(..., ge=0, alias="masterSeed")
    sample_count: int = Field(..., ge=1, alias="sampleCount")
    batch_size: int = Field(..., ge=1, alias="batchSize")
    candidate_count: int = Field(..., ge=3, le=7, alias="candidateCount")
    completed_candidate_count: int = Field(..., ge=0, alias="completedCandidateCount")
    candidates: list[PricingCandidateSimulationResultV1]
    ranked_candidate_ids: list[str] = Field(default_factory=list, alias="rankedCandidateIds")
    winner_candidate_id: str | None = Field(default=None, alias="winnerCandidateId")


__all__ = [
    "PricingAvailableStockEvidenceV1",
    "PricingCandidateSimulationResultV1",
    "PricingCandidateV1",
    "PricingCompletedDayObservationV1",
    "PricingContextFeatureEvidenceV1",
    "PricingContextFeatureStatus",
    "PricingContextTransform",
    "PricingDailyObservationsEvidenceV1",
    "PricingElasticityEvidenceV1",
    "PricingOperation",
    "PricingRealizationEvidenceV1",
    "PricingForecastContextEvidenceV1",
    "PricingFutureInboundEventV1",
    "PricingRankingScoreV1",
    "PricingSimulationBatchResultV1",
    "PricingSimulationEvidenceV1",
    "PricingSimulationRequestV1",
    "PricingSimulationResultV1",
    "PricingUnitEconomicsEvidenceV1",
]
