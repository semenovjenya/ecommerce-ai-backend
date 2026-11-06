from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class DifferenceInDifferencesResult:
    observed_delta: float
    did_effect: float | None
    causal_valid: bool


@dataclass(frozen=True)
class MeasurementCalculation:
    status: str
    observed_delta: float | None = None
    did_effect: float | None = None
    causal_valid: bool = False
    signed_bias: float | None = None
    mae: float | None = None
    inside_forecast_range: bool | None = None
    reason: str | None = None


def compute_difference_in_differences(
    *,
    baseline_treated: float,
    post_treated: float,
    baseline_control: float | None = None,
    post_control: float | None = None,
) -> DifferenceInDifferencesResult:
    values = [baseline_treated, post_treated]
    if baseline_control is not None:
        values.append(baseline_control)
    if post_control is not None:
        values.append(post_control)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("measurement values must be finite")

    observed_delta = float(post_treated) - float(baseline_treated)
    if baseline_control is None or post_control is None:
        return DifferenceInDifferencesResult(
            observed_delta=observed_delta,
            did_effect=None,
            causal_valid=False,
        )
    control_delta = float(post_control) - float(baseline_control)
    return DifferenceInDifferencesResult(
        observed_delta=observed_delta,
        did_effect=observed_delta - control_delta,
        causal_valid=True,
    )


def calculate_measurement(
    *,
    data_complete: bool,
    evaluation_end: datetime,
    data_as_of: datetime | None,
    baseline_treated: float | None,
    post_treated: float | None,
    baseline_control: float | None = None,
    post_control: float | None = None,
    forecast_low: float | None = None,
    forecast_base: float | None = None,
    forecast_high: float | None = None,
    now: datetime | None = None,
) -> MeasurementCalculation:
    """Calculate an outcome only when the source confirms the complete window.

    Zero values are valid only when ``data_complete`` is true. A missing/lagging
    source produces a waiting record instead of silently turning missing data into
    zero sales or zero contribution.
    """
    now = now or datetime.now(timezone.utc)
    evaluation_end = _aware_utc(evaluation_end)
    normalized_as_of = _aware_utc(data_as_of) if data_as_of else None
    if evaluation_end > _aware_utc(now):
        return MeasurementCalculation(status="waiting_for_data", reason="window_open")
    if not data_complete or normalized_as_of is None or normalized_as_of < evaluation_end:
        return MeasurementCalculation(status="waiting_for_data", reason="source_lag")
    if baseline_treated is None or post_treated is None:
        return MeasurementCalculation(status="inconclusive", reason="treated_series_missing")

    effect = compute_difference_in_differences(
        baseline_treated=baseline_treated,
        post_treated=post_treated,
        baseline_control=baseline_control,
        post_control=post_control,
    )
    actual = effect.did_effect if effect.causal_valid else effect.observed_delta
    signed_bias = (
        actual - float(forecast_base)
        if actual is not None and forecast_base is not None
        else None
    )
    inside_range = None
    if actual is not None and forecast_low is not None and forecast_high is not None:
        lower, upper = sorted((float(forecast_low), float(forecast_high)))
        inside_range = lower <= actual <= upper
    return MeasurementCalculation(
        status="computed",
        observed_delta=effect.observed_delta,
        did_effect=effect.did_effect,
        causal_valid=effect.causal_valid,
        signed_bias=signed_bias,
        mae=abs(signed_bias) if signed_bias is not None else None,
        inside_forecast_range=inside_range,
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "DifferenceInDifferencesResult",
    "MeasurementCalculation",
    "calculate_measurement",
    "compute_difference_in_differences",
]
