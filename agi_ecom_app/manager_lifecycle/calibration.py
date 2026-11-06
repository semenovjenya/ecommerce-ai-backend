"""Deterministic calibration proposals for measured manager actions.

The module deliberately stops at a proposal.  It does not mutate forecast
models, widen guardrails, or apply an adjustment automatically.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


MIN_CAUSAL_VALID_ACTIONS = 20
CALIBRATION_SCHEMA_VERSION = "effect_range_calibration.v1"
CALIBRATION_METHOD = "additive_signed_bias_shift_v1"

CalibrationStatus = Literal["insufficient_data", "proposed"]


@dataclass(frozen=True)
class EffectRangeMeasurement:
    """One measured action expressed in the common effect metric.

    ``actual_effect`` must be the causal estimate (for example a
    difference-in-differences result) whenever ``causal_valid`` is true.
    Non-causal observations are accepted for an honest sample count but are
    never used for calibration.
    """

    action_id: str
    action_family: str
    causal_valid: bool
    actual_effect: float | None
    forecast_low: float | None
    forecast_base: float | None
    forecast_high: float | None


@dataclass(frozen=True)
class ProposedEffectRangeAdjustment:
    """A reviewable additive shift that preserves the forecast-range width."""

    schema_version: str
    action_family: str
    base_version: int
    proposed_version: int
    method: str
    low_delta: float
    base_delta: float
    high_delta: float
    interval_width_delta: float
    auto_apply: bool
    requires_manual_approval: bool


@dataclass(frozen=True)
class EffectRangeCalibrationResult:
    """Calibration evidence and, when eligible, a versioned proposal."""

    status: CalibrationStatus
    schema_version: str
    action_family: str
    minimum_sample_size: int
    target_family_action_count: int
    causal_valid_action_count: int
    excluded_non_causal_count: int
    ignored_other_family_count: int
    signed_bias: float | None = None
    mae: float | None = None
    coverage: float | None = None
    proposed_adjustment: ProposedEffectRangeAdjustment | None = None
    reason: str | None = None


def propose_effect_range_calibration(
    measurements: Iterable[EffectRangeMeasurement],
    *,
    action_family: str,
    current_version: int = 0,
) -> EffectRangeCalibrationResult:
    """Build a deterministic calibration proposal for one action family.

    Eligibility requires at least :data:`MIN_CAUSAL_VALID_ACTIONS` distinct,
    causally valid actions from the requested family.  The signed bias is
    ``mean(actual_effect - forecast_base)``; positive values mean that the
    forecast under-estimated the measured effect.  MAE uses the same residuals,
    and coverage is the share of measured effects inside ``[low, high]``.

    The proposal shifts low/base/high by the signed bias.  Shifting every bound
    equally keeps the interval width unchanged.  The returned object is always
    marked as non-auto-applying and requiring manual approval.
    """

    normalized_family = _normalize_non_empty(action_family, "action_family")
    if isinstance(current_version, bool) or not isinstance(current_version, int):
        raise ValueError("current_version must be a non-negative integer")
    if current_version < 0:
        raise ValueError("current_version must be a non-negative integer")

    target_measurements: list[EffectRangeMeasurement] = []
    eligible: list[tuple[float, float, float, float]] = []
    seen_action_ids: set[str] = set()
    ignored_other_family_count = 0
    excluded_non_causal_count = 0

    for measurement in measurements:
        if not isinstance(measurement, EffectRangeMeasurement):
            raise TypeError("measurements must contain EffectRangeMeasurement values")

        measurement_family = _normalize_non_empty(
            measurement.action_family,
            "measurement.action_family",
        )
        if measurement_family != normalized_family:
            ignored_other_family_count += 1
            continue

        action_id = _normalize_non_empty(measurement.action_id, "measurement.action_id")
        if action_id in seen_action_ids:
            raise ValueError(
                f"duplicate action_id in action family {normalized_family!r}: {action_id!r}"
            )
        seen_action_ids.add(action_id)
        target_measurements.append(measurement)

        if not isinstance(measurement.causal_valid, bool):
            raise ValueError("measurement.causal_valid must be a boolean")
        if not measurement.causal_valid:
            excluded_non_causal_count += 1
            continue

        actual = _finite_number(measurement.actual_effect, "measurement.actual_effect")
        low = _finite_number(measurement.forecast_low, "measurement.forecast_low")
        base = _finite_number(measurement.forecast_base, "measurement.forecast_base")
        high = _finite_number(measurement.forecast_high, "measurement.forecast_high")
        if not low <= base <= high:
            raise ValueError("forecast bounds must satisfy low <= base <= high")
        eligible.append((actual, low, base, high))

    causal_valid_action_count = len(eligible)
    common = dict(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        action_family=normalized_family,
        minimum_sample_size=MIN_CAUSAL_VALID_ACTIONS,
        target_family_action_count=len(target_measurements),
        causal_valid_action_count=causal_valid_action_count,
        excluded_non_causal_count=excluded_non_causal_count,
        ignored_other_family_count=ignored_other_family_count,
    )
    if causal_valid_action_count < MIN_CAUSAL_VALID_ACTIONS:
        return EffectRangeCalibrationResult(
            status="insufficient_data",
            reason=(
                f"requires at least {MIN_CAUSAL_VALID_ACTIONS} distinct causal_valid "
                f"measured actions in action family {normalized_family!r}"
            ),
            **common,
        )

    residuals = [actual - base for actual, _low, base, _high in eligible]
    signed_bias = math.fsum(residuals) / causal_valid_action_count
    mae = math.fsum(abs(residual) for residual in residuals) / causal_valid_action_count
    coverage = (
        math.fsum(1.0 for actual, low, _base, high in eligible if low <= actual <= high)
        / causal_valid_action_count
    )
    proposed_adjustment = ProposedEffectRangeAdjustment(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        action_family=normalized_family,
        base_version=current_version,
        proposed_version=current_version + 1,
        method=CALIBRATION_METHOD,
        low_delta=signed_bias,
        base_delta=signed_bias,
        high_delta=signed_bias,
        interval_width_delta=0.0,
        auto_apply=False,
        requires_manual_approval=True,
    )
    return EffectRangeCalibrationResult(
        status="proposed",
        signed_bias=signed_bias,
        mae=mae,
        coverage=coverage,
        proposed_adjustment=proposed_adjustment,
        **common,
    )


def _normalize_non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip().lower()


def _finite_number(value: object, field_name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number for causal_valid actions")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a finite number for causal_valid actions"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number for causal_valid actions")
    return number


__all__ = [
    "CALIBRATION_METHOD",
    "CALIBRATION_SCHEMA_VERSION",
    "MIN_CAUSAL_VALID_ACTIONS",
    "EffectRangeCalibrationResult",
    "EffectRangeMeasurement",
    "ProposedEffectRangeAdjustment",
    "propose_effect_range_calibration",
]
