from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from agi_ecom_app.decision_simulation.engine import _distribution_pair
from agi_ecom_app.decision_simulation.models import ConstraintViolationV1
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingCandidateSimulationResultV1,
    PricingCandidateV1,
    PricingCompletedDayObservationV1,
    PricingRankingScoreV1,
    PricingSimulationBatchResultV1,
    PricingSimulationRequestV1,
    PricingSimulationResultV1,
)


PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION = "pricing-elasticity-bootstrap-v1"
MIN_COMPLETED_OBSERVATION_DAYS = 28
MIN_ELASTICITY_BOOTSTRAP_VALUES = 100
MIN_REALIZATION_BOOTSTRAP_VALUES = 100
MIN_UNIT_ECONOMICS_OBSERVATION_DAYS = 14
MAX_DEMAND_OBSERVATION_AGE_DAYS = 14
MAX_UNIT_ECONOMICS_AGE_DAYS = 30
MAX_STOCK_AGE_DAYS = 14
MAX_PRICE_MOVE_FRACTION = 0.10
MAX_CONTEXT_DEMAND_MULTIPLIER = 4.0
MIN_PLAUSIBLE_PRICE_ELASTICITY = -8.0
MIN_MODELED_CONTEXT_OBSERVATION_DAYS = 14
MIN_MODELED_CONTEXT_COVERAGE_RATIO = 0.25
_UNSUPPORTED_ELASTICITY_SOURCE_MARKERS = (
    "fallback",
    "default",
    "synthetic",
)


@dataclass(frozen=True)
class PricingBatchValues:
    seed: int
    incremental_contribution_profit: NDArray[np.float64]
    units_sold: NDArray[np.float64]
    stockout: NDArray[np.bool_]


def _normalized_snapshot_hash(snapshot_hash: str) -> str:
    normalized = snapshot_hash.strip().lower()
    if len(normalized) != 64 or any(value not in "0123456789abcdef" for value in normalized):
        raise ValueError("snapshot_hash must be a 64-character hexadecimal SHA-256")
    return normalized


def derive_pricing_master_seed(snapshot_hash: str, decision_group_id: str) -> int:
    normalized = _normalized_snapshot_hash(snapshot_hash)
    group = decision_group_id.strip()
    if not group:
        raise ValueError("decision_group_id cannot be blank")
    material = f"{normalized}\x00{group}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], byteorder="big", signed=False)


def derive_pricing_batch_seed(
    snapshot_hash: str,
    decision_group_id: str,
    batch_index: int,
) -> int:
    """Derive one group seed shared by every price candidate in the batch."""

    normalized = _normalized_snapshot_hash(snapshot_hash)
    group = decision_group_id.strip()
    if not group:
        raise ValueError("decision_group_id cannot be blank")
    if batch_index < 0:
        raise ValueError("batch_index cannot be negative")
    material = f"{normalized}\x00{group}\x00{batch_index}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], byteorder="big", signed=False)


def _violation(
    code: str,
    severity: str,
    message: str,
    *,
    evidence_ref: str | None = None,
) -> ConstraintViolationV1:
    return ConstraintViolationV1(
        code=code,
        severity=severity,
        message=message,
        evidence_ref=evidence_ref,
    )


def _eligible_observations(
    request: PricingSimulationRequestV1,
) -> list[PricingCompletedDayObservationV1]:
    observations = request.evidence.daily_observations.values or []
    return [
        item
        for item in observations
        if item.completed and item.not_observed_out_of_stock
    ]


def validate_pricing_request_evidence(
    request: PricingSimulationRequestV1,
) -> list[ConstraintViolationV1]:
    """Fail closed before sampling; no inferred or fallback elasticity is allowed."""

    evidence = request.evidence
    violations: list[ConstraintViolationV1] = []

    if request.current_price_evidence_ref is None:
        violations.append(
            _violation(
                "missing_current_price_evidence_ref",
                "needs_data",
                "Current seller-controlled price has no source reference.",
            )
        )

    observations = evidence.daily_observations
    if observations.evidence_ref is None:
        violations.append(
            _violation(
                "missing_daily_observations_evidence_ref",
                "needs_data",
                "Completed-day price and sales observations have no source reference.",
            )
        )
    if observations.values is None:
        violations.append(
            _violation(
                "missing_completed_day_observations",
                "needs_data",
                "Completed SKU days without an observed OOS signal are required.",
            )
        )
    else:
        eligible = _eligible_observations(request)
        if len(eligible) < MIN_COMPLETED_OBSERVATION_DAYS:
            violations.append(
                _violation(
                    "insufficient_completed_day_observations",
                    "needs_data",
                    f"At least {MIN_COMPLETED_OBSERVATION_DAYS} completed non-OOS-observed SKU days are required.",
                    evidence_ref=observations.evidence_ref,
                )
            )
        elif not any(item.units_sold > 0 for item in eligible):
            violations.append(
                _violation(
                    "non_positive_demand_history",
                    "blocked",
                    "Completed-day history contains no positive observed demand.",
                    evidence_ref=observations.evidence_ref,
                )
            )
    if observations.latest_observation_age_days is None:
        violations.append(
            _violation(
                "missing_demand_observation_freshness",
                "needs_data",
                "Demand observation freshness is required.",
                evidence_ref=observations.evidence_ref,
            )
        )
    elif observations.latest_observation_age_days > MAX_DEMAND_OBSERVATION_AGE_DAYS:
        violations.append(
            _violation(
                "stale_demand_observations",
                "needs_data",
                "Latest price-demand observation is too old for pricing simulation.",
                evidence_ref=observations.evidence_ref,
            )
        )

    elasticity = evidence.elasticity
    if elasticity.evidence_ref is None:
        violations.append(
            _violation(
                "missing_elasticity_evidence_ref",
                "needs_data",
                "Price elasticity has no source reference.",
            )
        )
    if elasticity.source is None:
        violations.append(
            _violation(
                "missing_elasticity_source",
                "needs_data",
                "Price elasticity source is required.",
            )
        )
    else:
        normalized_source = elasticity.source.casefold().replace("-", "_")
        if any(
            marker in normalized_source
            for marker in _UNSUPPORTED_ELASTICITY_SOURCE_MARKERS
        ):
            violations.append(
                _violation(
                    "fallback_elasticity_not_allowed",
                    "blocked",
                    "Fallback, default, or synthetic elasticity cannot authorize a price simulation.",
                    evidence_ref=elasticity.evidence_ref,
                )
            )

    elasticity_values = elasticity.bootstrap_values
    if elasticity_values is None:
        violations.append(
            _violation(
                "missing_elasticity_bootstrap_values",
                "needs_data",
                "Empirical elasticity bootstrap values are required.",
                evidence_ref=elasticity.evidence_ref,
            )
        )
    elif len(elasticity_values) < MIN_ELASTICITY_BOOTSTRAP_VALUES:
        violations.append(
            _violation(
                "insufficient_elasticity_bootstrap_values",
                "needs_data",
                f"At least {MIN_ELASTICITY_BOOTSTRAP_VALUES} elasticity values are required.",
                evidence_ref=elasticity.evidence_ref,
            )
        )
    elif any(not math.isfinite(value) or value >= 0 for value in elasticity_values):
        violations.append(
            _violation(
                "invalid_price_elasticity",
                "blocked",
                "Every price elasticity bootstrap value must be finite and negative.",
                evidence_ref=elasticity.evidence_ref,
            )
        )
    if elasticity_values is not None and any(
        math.isfinite(value) and value < MIN_PLAUSIBLE_PRICE_ELASTICITY
        for value in elasticity_values
    ):
        violations.append(
            _violation(
                "implausible_price_elasticity_magnitude",
                "blocked",
                "Observed price elasticity is too extreme for a pricing recommendation without a controlled experiment.",
                evidence_ref=elasticity.evidence_ref,
            )
        )

    realization = evidence.realization
    if realization.evidence_ref is None:
        violations.append(
            _violation(
                "missing_realization_evidence_ref",
                "needs_data",
                "Order-to-realized-sale evidence has no source reference.",
            )
        )
    if realization.source is None:
        violations.append(
            _violation(
                "missing_realization_source",
                "needs_data",
                "Order-to-realized-sale source is required.",
            )
        )
    realization_values = realization.bootstrap_values
    if realization_values is None:
        violations.append(
            _violation(
                "missing_realization_bootstrap_values",
                "needs_data",
                "Empirical order-to-realized-sale bootstrap values are required.",
                evidence_ref=realization.evidence_ref,
            )
        )
    elif len(realization_values) < MIN_REALIZATION_BOOTSTRAP_VALUES:
        violations.append(
            _violation(
                "insufficient_realization_bootstrap_values",
                "needs_data",
                f"At least {MIN_REALIZATION_BOOTSTRAP_VALUES} realization values are required.",
                evidence_ref=realization.evidence_ref,
            )
        )
    elif any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in realization_values
    ) or not any(value > 0 for value in realization_values):
        violations.append(
            _violation(
                "invalid_realization_distribution",
                "blocked",
                "Every realization value must be finite and in [0, 1], with positive support.",
                evidence_ref=realization.evidence_ref,
            )
        )

    economics = evidence.unit_economics
    if economics.evidence_ref is None:
        violations.append(
            _violation(
                "missing_unit_economics_evidence_ref",
                "needs_data",
                "Unit economics have no source reference.",
            )
        )
    if economics.basis != "wb_net_payout_before_cogs":
        violations.append(
            _violation(
                "unsupported_unit_economics_basis",
                "needs_data",
                "Pricing unit economics must use reconciled WB net payout before COGS.",
                evidence_ref=economics.evidence_ref,
            )
        )
    if (
        economics.observation_count is None
        or economics.observation_count < MIN_UNIT_ECONOMICS_OBSERVATION_DAYS
    ):
        violations.append(
            _violation(
                "insufficient_unit_economics_observations",
                "needs_data",
                f"At least {MIN_UNIT_ECONOMICS_OBSERVATION_DAYS} unit-economics days are required.",
                evidence_ref=economics.evidence_ref,
            )
        )
    if economics.latest_observation_age_days is None:
        violations.append(
            _violation(
                "missing_unit_economics_freshness",
                "needs_data",
                "Unit-economics freshness is required.",
                evidence_ref=economics.evidence_ref,
            )
        )
    elif economics.latest_observation_age_days > MAX_UNIT_ECONOMICS_AGE_DAYS:
        violations.append(
            _violation(
                "stale_unit_economics",
                "needs_data",
                "Unit-economics observations are too old for pricing simulation.",
                evidence_ref=economics.evidence_ref,
            )
        )
    if economics.net_revenue_rate is None:
        violations.append(
            _violation(
                "missing_net_revenue_rate",
                "needs_data",
                "Net revenue rate is required.",
                evidence_ref=economics.evidence_ref,
            )
        )
    elif not math.isfinite(economics.net_revenue_rate) or not (
        0 < economics.net_revenue_rate <= 1
    ):
        violations.append(
            _violation(
                "invalid_net_revenue_rate",
                "blocked",
                "Net revenue rate must be finite and in the interval (0, 1].",
                evidence_ref=economics.evidence_ref,
            )
        )
    if economics.variable_cost_per_unit_rub is None:
        violations.append(
            _violation(
                "missing_variable_cost_per_unit",
                "needs_data",
                "Variable cost per sold unit is required.",
                evidence_ref=economics.evidence_ref,
            )
        )
    elif (
        not math.isfinite(economics.variable_cost_per_unit_rub)
        or economics.variable_cost_per_unit_rub < 0
    ):
        violations.append(
            _violation(
                "invalid_variable_cost_per_unit",
                "blocked",
                "Variable cost per sold unit must be finite and non-negative.",
                evidence_ref=economics.evidence_ref,
            )
        )

    stock = evidence.available_stock
    if stock.evidence_ref is None:
        violations.append(
            _violation(
                "missing_available_stock_evidence_ref",
                "needs_data",
                "Available stock over the action horizon has no source reference.",
            )
        )
    if stock.snapshot_date is None or stock.age_days_at_snapshot is None:
        violations.append(
            _violation(
                "missing_stock_freshness",
                "needs_data",
                "Stock snapshot date and pinned age are required.",
                evidence_ref=stock.evidence_ref,
            )
        )
    elif stock.age_days_at_snapshot > MAX_STOCK_AGE_DAYS:
        violations.append(
            _violation(
                "stale_available_stock",
                "needs_data",
                "Available-stock snapshot is too old for the pricing horizon.",
                evidence_ref=stock.evidence_ref,
            )
        )
    if stock.units is None:
        violations.append(
            _violation(
                "missing_available_stock",
                "needs_data",
                "Available stock over the action horizon is required.",
                evidence_ref=stock.evidence_ref,
            )
        )
    elif not math.isfinite(stock.units) or stock.units < 0:
        violations.append(
            _violation(
                "invalid_available_stock",
                "blocked",
                "Available stock must be finite and non-negative.",
                evidence_ref=stock.evidence_ref,
            )
        )

    context = evidence.forecast_context
    if context is not None:
        elasticity_draw_count = len(elasticity_values or [])
        for feature in context.features:
            if feature.status != "modeled":
                continue
            if feature.evidence_ref is None:
                violations.append(
                    _violation(
                        "missing_context_feature_evidence_ref",
                        "needs_data",
                        f"Modeled context feature {feature.name!r} has no source reference.",
                    )
                )
            if (
                feature.observed_days < MIN_MODELED_CONTEXT_OBSERVATION_DAYS
                or feature.coverage_ratio < MIN_MODELED_CONTEXT_COVERAGE_RATIO
            ):
                violations.append(
                    _violation(
                        "insufficient_context_feature_history",
                        "needs_data",
                        f"Modeled context feature {feature.name!r} has insufficient point-in-time history.",
                        evidence_ref=feature.evidence_ref,
                    )
                )
            coefficients = feature.bootstrap_coefficients or []
            if (
                len(coefficients) != elasticity_draw_count
                or any(not math.isfinite(value) for value in coefficients)
            ):
                violations.append(
                    _violation(
                        "invalid_context_bootstrap_coefficients",
                        "blocked",
                        f"Modeled context feature {feature.name!r} has invalid bootstrap coefficients.",
                        evidence_ref=feature.evidence_ref,
                    )
                )
            if (
                len(feature.forward_values) != request.horizon_days
                or any(value is None for value in feature.forward_values)
            ):
                violations.append(
                    _violation(
                        "missing_context_forecast_path",
                        "needs_data",
                        f"Modeled context feature {feature.name!r} requires a complete pinned forecast path.",
                        evidence_ref=feature.evidence_ref,
                    )
                )
            else:
                invalid_forward = any(
                    not math.isfinite(float(value))
                    or (feature.transform == "log1p" and float(value) < 0)
                    for value in feature.forward_values
                )
                if invalid_forward:
                    violations.append(
                        _violation(
                            "invalid_context_forecast_value",
                            "blocked",
                            f"Modeled context feature {feature.name!r} has an invalid forecast value.",
                            evidence_ref=feature.evidence_ref,
                        )
                    )
            invalid_history = any(
                raw is not None
                and (
                    not math.isfinite(float(raw))
                    or (feature.transform == "log1p" and float(raw) < 0)
                )
                for item in _eligible_observations(request)
                for raw in [
                    (item.context_values or {}).get(feature.name)
                ]
            )
            if invalid_history:
                violations.append(
                    _violation(
                        "invalid_context_observation_value",
                        "blocked",
                        f"Modeled context feature {feature.name!r} has an invalid historical value.",
                        evidence_ref=feature.evidence_ref,
                    )
                )

    inbound_snapshot = (
        context.snapshot_date if context is not None else stock.snapshot_date
    )
    seen_inbound: set[tuple[str | None, object]] = set()
    for inbound in stock.future_inbound:
        identity = (inbound.supply_id, inbound.eta_date)
        if identity in seen_inbound:
            violations.append(
                _violation(
                    "duplicate_future_inbound_event",
                    "blocked",
                    "Future supplier inbound events must be unique by supply and ETA.",
                    evidence_ref=inbound.evidence_ref,
                )
            )
        seen_inbound.add(identity)
        if not inbound.include_in_capacity:
            continue
        if inbound.status != "scheduled":
            violations.append(
                _violation(
                    "unconfirmed_future_inbound_in_capacity",
                    "blocked",
                    "Only confirmed scheduled supplier inbound may increase forecast capacity.",
                    evidence_ref=inbound.evidence_ref,
                )
            )
        if inbound.evidence_ref is None:
            violations.append(
                _violation(
                    "missing_future_inbound_evidence_ref",
                    "needs_data",
                    "Included supplier inbound has no source reference.",
                )
            )
        if inbound_snapshot is None:
            violations.append(
                _violation(
                    "missing_future_inbound_forecast_snapshot",
                    "needs_data",
                    "Included supplier inbound requires a pinned forecast snapshot date.",
                    evidence_ref=inbound.evidence_ref,
                )
            )
        else:
            if inbound.eta_date <= inbound_snapshot:
                violations.append(
                    _violation(
                        "past_future_inbound_eta",
                        "blocked",
                        "Included supplier inbound ETA is before the forecast snapshot.",
                        evidence_ref=inbound.evidence_ref,
                    )
                )
            if inbound.eta_date > inbound_snapshot + timedelta(
                days=request.horizon_days
            ):
                violations.append(
                    _violation(
                        "future_inbound_outside_horizon",
                        "blocked",
                        "Included supplier inbound ETA is outside the pricing horizon.",
                        evidence_ref=inbound.evidence_ref,
                    )
                )
        if not math.isfinite(inbound.units) or inbound.units <= 0:
            violations.append(
                _violation(
                    "invalid_future_inbound_units",
                    "blocked",
                    "Included supplier inbound units must be positive and finite.",
                    evidence_ref=inbound.evidence_ref,
                )
            )
    return violations


def _unit_contribution(
    request: PricingSimulationRequestV1,
    price_rub: float,
) -> float:
    economics = request.evidence.unit_economics
    if (
        economics.net_revenue_rate is None
        or economics.variable_cost_per_unit_rub is None
    ):
        raise ValueError("validated request has incomplete unit economics")
    return (
        price_rub * economics.net_revenue_rate
        - economics.variable_cost_per_unit_rub
    )


def validate_pricing_candidate(
    request: PricingSimulationRequestV1,
    candidate: PricingCandidateV1,
) -> list[ConstraintViolationV1]:
    violations = validate_pricing_request_evidence(request)
    relative_move = abs(candidate.price_rub / request.current_price_rub - 1)
    if relative_move > MAX_PRICE_MOVE_FRACTION + 1e-12:
        violations.append(
            _violation(
                "price_move_out_of_v1_bounds",
                "blocked",
                f"Pricing v1 candidates are limited to +/-{MAX_PRICE_MOVE_FRACTION:.0%} of current price.",
                evidence_ref=request.current_price_evidence_ref,
            )
        )
    economics = request.evidence.unit_economics
    if (
        candidate.operation == "set_price"
        and economics.net_revenue_rate is not None
        and economics.variable_cost_per_unit_rub is not None
        and math.isfinite(economics.net_revenue_rate)
        and math.isfinite(economics.variable_cost_per_unit_rub)
    ):
        contribution = _unit_contribution(request, candidate.price_rub)
        if contribution < economics.minimum_unit_contribution_rub:
            violations.append(
                _violation(
                    "minimum_unit_contribution_violated",
                    "blocked",
                    "Candidate price is below the configured unit-contribution floor.",
                    evidence_ref=economics.evidence_ref,
                )
            )
    return violations


def _evidence_refs(request: PricingSimulationRequestV1) -> list[str]:
    evidence = request.evidence
    references: list[str | None] = [
        request.current_price_evidence_ref,
        evidence.daily_observations.evidence_ref,
        evidence.elasticity.evidence_ref,
        evidence.realization.evidence_ref,
        evidence.unit_economics.evidence_ref,
        evidence.available_stock.evidence_ref,
    ]
    for observation in evidence.daily_observations.values or []:
        references.append(observation.evidence_ref)
    context = evidence.forecast_context
    if context is not None:
        for feature in context.features:
            references.append(feature.evidence_ref)
    for inbound in evidence.available_stock.future_inbound:
        references.append(inbound.evidence_ref)
    deduplicated: list[str] = []
    for reference in references:
        if reference is not None and reference not in deduplicated:
            deduplicated.append(reference)
    return deduplicated


def simulate_pricing_batch_values(
    request: PricingSimulationRequestV1,
    candidate: PricingCandidateV1,
    batch_index: int,
) -> PricingBatchValues:
    """Replay one batch using random draws shared by all price alternatives."""

    observations = _eligible_observations(request)
    elasticity_values = request.evidence.elasticity.bootstrap_values
    realization_values = request.evidence.realization.bootstrap_values
    stock_units = request.evidence.available_stock.units
    if (
        not observations
        or elasticity_values is None
        or realization_values is None
        or stock_units is None
    ):
        raise ValueError("validated pricing evidence is incomplete")

    seed = derive_pricing_batch_seed(
        request.snapshot_hash,
        request.decision_group_id,
        batch_index,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    observation_indices = generator.integers(
        0,
        len(observations),
        size=(request.batch_size, request.horizon_days),
    )
    elasticity_indices = generator.integers(
        0,
        len(elasticity_values),
        size=request.batch_size,
    )
    realization_indices = generator.integers(
        0,
        len(realization_values),
        size=request.batch_size,
    )

    observed_prices = np.asarray(
        [item.price_rub for item in observations], dtype=np.float64
    )[observation_indices]
    observed_units = np.asarray(
        [item.units_sold for item in observations], dtype=np.float64
    )[observation_indices]
    sampled_elasticity = np.asarray(elasticity_values, dtype=np.float64)[
        elasticity_indices
    ][:, np.newaxis]
    sampled_realization = np.asarray(realization_values, dtype=np.float64)[
        realization_indices
    ]
    context_log_multiplier = _context_log_multiplier(
        request,
        observations=observations,
        observation_indices=observation_indices,
        bootstrap_indices=elasticity_indices,
    )
    inbound_by_day = _future_inbound_by_day(request)
    extended_forecast = bool(
        request.evidence.forecast_context
        and any(
            feature.status == "modeled"
            for feature in request.evidence.forecast_context.features
        )
    ) or bool(np.any(inbound_by_day > 0))

    def sold_units_at(price_rub: float) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        price_ratio = price_rub / observed_prices
        projected_daily_demand = observed_units * np.power(
            price_ratio,
            sampled_elasticity,
        )
        if not extended_forecast:
            # Preserve byte-for-byte v1 replay semantics for durable snapshots
            # created before contextual controls and dated inbound were added.
            total_orders = np.sum(
                projected_daily_demand,
                axis=1,
                dtype=np.float64,
            )
            realized_demand = total_orders * sampled_realization
            units_sold = np.minimum(realized_demand, float(stock_units))
            return units_sold.astype(np.float64), (
                realized_demand > float(stock_units)
            )
        projected_daily_demand *= np.exp(context_log_multiplier)
        realized_daily_demand = projected_daily_demand * sampled_realization[:, np.newaxis]
        available = np.full(request.batch_size, float(stock_units), dtype=np.float64)
        units_sold = np.zeros(request.batch_size, dtype=np.float64)
        stockout = np.zeros(request.batch_size, dtype=np.bool_)
        for day_index in range(request.horizon_days):
            available += inbound_by_day[day_index]
            day_demand = realized_daily_demand[:, day_index]
            stockout |= day_demand > available
            sold = np.minimum(day_demand, available)
            units_sold += sold
            available -= sold
        return units_sold.astype(np.float64), stockout

    hold_units, _ = sold_units_at(request.current_price_rub)
    candidate_units, candidate_stockout = sold_units_at(candidate.price_rub)
    hold_profit = hold_units * _unit_contribution(
        request,
        request.current_price_rub,
    )
    candidate_profit = candidate_units * _unit_contribution(
        request,
        candidate.price_rub,
    )
    incremental = candidate_profit - hold_profit
    if not bool(np.all(np.isfinite(incremental))) or not bool(
        np.all(np.isfinite(candidate_units))
    ):
        raise ValueError("pricing simulation produced non-finite values")
    return PricingBatchValues(
        seed=seed,
        incremental_contribution_profit=incremental.astype(np.float64),
        units_sold=candidate_units,
        stockout=candidate_stockout,
    )


def _context_log_multiplier(
    request: PricingSimulationRequestV1,
    *,
    observations: list[PricingCompletedDayObservationV1],
    observation_indices: NDArray[np.int64],
    bootstrap_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    context = request.evidence.forecast_context
    result = np.zeros(
        (request.batch_size, request.horizon_days),
        dtype=np.float64,
    )
    if context is None:
        return result
    for feature in context.features:
        if (
            feature.status != "modeled"
            or feature.bootstrap_coefficients is None
            or feature.center is None
            or feature.scale is None
            or not feature.forward_values
        ):
            continue
        observed_raw = np.asarray(
            [
                _feature_value(item.context_values, feature.name)
                for item in observations
            ],
            dtype=object,
        )[observation_indices]
        observed_transformed = np.full(
            observed_raw.shape,
            float(feature.center),
            dtype=np.float64,
        )
        for index in np.ndindex(observed_raw.shape):
            raw = observed_raw[index]
            if raw is not None:
                observed_transformed[index] = _transform_feature(
                    float(raw),
                    feature.transform,
                )
        forward_transformed = np.asarray(
            [
                float(feature.center)
                if raw is None
                else _transform_feature(float(raw), feature.transform)
                for raw in feature.forward_values
            ],
            dtype=np.float64,
        )
        observed_normalized = (
            observed_transformed - float(feature.center)
        ) / float(feature.scale)
        forward_normalized = (
            forward_transformed - float(feature.center)
        ) / float(feature.scale)
        coefficients = np.asarray(
            feature.bootstrap_coefficients,
            dtype=np.float64,
        )[bootstrap_indices][:, np.newaxis]
        result += coefficients * (
            forward_normalized[np.newaxis, :] - observed_normalized
        )
    limit = math.log(MAX_CONTEXT_DEMAND_MULTIPLIER)
    return np.clip(result, -limit, limit)


def _future_inbound_by_day(
    request: PricingSimulationRequestV1,
) -> NDArray[np.float64]:
    result = np.zeros(request.horizon_days, dtype=np.float64)
    context = request.evidence.forecast_context
    as_of = (
        context.snapshot_date
        if context is not None
        else request.evidence.available_stock.snapshot_date
    )
    if as_of is None:
        return result
    seen: set[tuple[str | None, object]] = set()
    for event in request.evidence.available_stock.future_inbound:
        if not event.include_in_capacity or event.status != "scheduled":
            continue
        identity = (event.supply_id, event.eta_date)
        if identity in seen or event.eta_date <= as_of:
            continue
        seen.add(identity)
        day_index = (event.eta_date - as_of).days - 1
        day_index = max(0, day_index)
        if day_index < request.horizon_days:
            result[day_index] += event.units
    return result


def _feature_value(values: dict[str, float] | None, name: str) -> float | None:
    if not values or name not in values:
        return None
    value = float(values[name])
    return value if math.isfinite(value) else None


def _transform_feature(value: float, transform: str) -> float:
    if transform == "identity":
        return value
    if transform == "log1p":
        if value < 0:
            raise ValueError("log1p context feature cannot be negative")
        return math.log1p(value)
    raise ValueError(f"unsupported context transform: {transform}")


class PricingSimulationEngine:
    """Pure deterministic one-SKU price-alternative simulation engine."""

    def validate_candidate(
        self,
        request: PricingSimulationRequestV1,
        candidate: PricingCandidateV1,
    ) -> list[ConstraintViolationV1]:
        return validate_pricing_candidate(request, candidate)

    def simulate_batch(
        self,
        request: PricingSimulationRequestV1,
        candidate: PricingCandidateV1,
        batch_index: int,
    ) -> PricingSimulationBatchResultV1:
        violations = self.validate_candidate(request, candidate)
        if violations:
            raise ValueError("cannot simulate a pricing candidate that failed the hard gate")
        batch_count = request.sample_count // request.batch_size
        if batch_index < 0 or batch_index >= batch_count:
            raise ValueError("batch_index is outside the pricing simulation request")
        values = simulate_pricing_batch_values(request, candidate, batch_index)
        return PricingSimulationBatchResultV1(
            candidate_id=candidate.candidate_id,
            batch_index=batch_index,
            seed=values.seed,
            sample_count=request.batch_size,
            distribution=_distribution_pair(
                values.incremental_contribution_profit,
                horizon_days=request.horizon_days,
            ),
            units_sold_distribution=_distribution_pair(
                values.units_sold,
                horizon_days=request.horizon_days,
            ),
            stockout_probability=float(np.mean(values.stockout)),
        )

    def simulate_candidate(
        self,
        request: PricingSimulationRequestV1,
        candidate: PricingCandidateV1,
    ) -> PricingCandidateSimulationResultV1:
        violations = self.validate_candidate(request, candidate)
        if violations:
            status = (
                "blocked"
                if any(item.severity == "blocked" for item in violations)
                else "needs_data"
            )
            return PricingCandidateSimulationResultV1(
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                operation=candidate.operation,
                price_rub=candidate.price_rub,
                status=status,
                constraint_violations=violations,
                evidence_refs=_evidence_refs(request),
            )
        batches = [
            self.simulate_batch(request, candidate, batch_index)
            for batch_index in range(request.sample_count // request.batch_size)
        ]
        return self.aggregate_candidate(request, candidate, batches)

    def aggregate_candidate(
        self,
        request: PricingSimulationRequestV1,
        candidate: PricingCandidateV1,
        batch_results: Iterable[PricingSimulationBatchResultV1],
    ) -> PricingCandidateSimulationResultV1:
        if self.validate_candidate(request, candidate):
            raise ValueError("cannot aggregate a pricing candidate that failed the hard gate")
        ordered = sorted(batch_results, key=lambda item: item.batch_index)
        batch_count = request.sample_count // request.batch_size
        if [item.batch_index for item in ordered] != list(range(batch_count)):
            raise ValueError("candidate aggregation requires every batch exactly once")

        incremental_values: list[NDArray[np.float64]] = []
        units_values: list[NDArray[np.float64]] = []
        stockout_values: list[NDArray[np.bool_]] = []
        for batch in ordered:
            replay = simulate_pricing_batch_values(
                request,
                candidate,
                batch.batch_index,
            )
            if batch.seed != replay.seed or batch.sample_count != request.batch_size:
                raise ValueError("durable pricing batch does not match its group seed")
            incremental_values.append(replay.incremental_contribution_profit)
            units_values.append(replay.units_sold)
            stockout_values.append(replay.stockout)

        all_incremental = np.concatenate(incremental_values)
        all_units = np.concatenate(units_values)
        all_stockouts = np.concatenate(stockout_values)
        distribution = _distribution_pair(
            all_incremental,
            horizon_days=request.horizon_days,
        )
        units_distribution = _distribution_pair(
            all_units,
            horizon_days=request.horizon_days,
        )
        score = PricingRankingScoreV1(
            cvar_10_per_day=distribution.per_day.cvar_10,
            p10_per_day=distribution.per_day.p10,
            mean_per_day=distribution.per_day.mean,
            absolute_price_move_rub=abs(
                candidate.price_rub - request.current_price_rub
            ),
        )
        return PricingCandidateSimulationResultV1(
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            operation=candidate.operation,
            price_rub=candidate.price_rub,
            status="completed",
            evidence_refs=_evidence_refs(request),
            batch_results=ordered,
            distribution=distribution,
            units_sold_distribution=units_distribution,
            stockout_probability=float(np.mean(all_stockouts)),
            unit_contribution_rub=_unit_contribution(request, candidate.price_rub),
            score_breakdown=score,
        )

    def run(self, request: PricingSimulationRequestV1) -> PricingSimulationResultV1:
        candidate_results = [
            self.simulate_candidate(request, candidate)
            for candidate in request.candidates
        ]
        completed = [item for item in candidate_results if item.status == "completed"]

        def ranking_key(
            item: PricingCandidateSimulationResultV1,
        ) -> tuple[float, float, float, float, str]:
            score = item.score_breakdown
            if score is None:
                raise ValueError("completed pricing candidate is missing score_breakdown")
            return (
                -score.cvar_10_per_day,
                -score.p10_per_day,
                -score.mean_per_day,
                score.absolute_price_move_rub,
                item.action_id,
            )

        ranked = sorted(completed, key=ranking_key)
        ranks = {item.candidate_id: index for index, item in enumerate(ranked, start=1)}
        winner_candidate_id = ranked[0].candidate_id if ranked else None
        finalized = [
            item.model_copy(
                update={
                    "rank": ranks.get(item.candidate_id),
                    "is_winner": item.candidate_id == winner_candidate_id,
                }
            )
            for item in candidate_results
        ]
        return PricingSimulationResultV1(
            snapshot_hash=request.snapshot_hash,
            decision_group_id=request.decision_group_id,
            sku_id=request.sku_id,
            model_version=request.model_version,
            master_seed=derive_pricing_master_seed(
                request.snapshot_hash,
                request.decision_group_id,
            ),
            sample_count=request.sample_count,
            batch_size=request.batch_size,
            candidate_count=len(request.candidates),
            completed_candidate_count=len(completed),
            candidates=finalized,
            ranked_candidate_ids=[item.candidate_id for item in ranked],
            winner_candidate_id=winner_candidate_id,
        )


def run_pricing_simulation(
    request: PricingSimulationRequestV1,
) -> PricingSimulationResultV1:
    return PricingSimulationEngine().run(request)


__all__ = [
    "MIN_COMPLETED_OBSERVATION_DAYS",
    "MIN_ELASTICITY_BOOTSTRAP_VALUES",
    "MIN_REALIZATION_BOOTSTRAP_VALUES",
    "PRICING_ELASTICITY_BOOTSTRAP_MODEL_VERSION",
    "PricingBatchValues",
    "PricingSimulationEngine",
    "derive_pricing_batch_seed",
    "derive_pricing_master_seed",
    "run_pricing_simulation",
    "simulate_pricing_batch_values",
    "validate_pricing_candidate",
    "validate_pricing_request_evidence",
]
