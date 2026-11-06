from __future__ import annotations

import pytest

from agi_ecom_app.manager_lifecycle import (
    CALIBRATION_METHOD,
    CALIBRATION_SCHEMA_VERSION,
    MIN_CAUSAL_VALID_ACTIONS,
    EffectRangeCalibrationResult,
    EffectRangeMeasurement,
    ProposedEffectRangeAdjustment,
    propose_effect_range_calibration,
)


def _measurement(
    action_id: str,
    *,
    family: str = "advertising",
    causal_valid: bool = True,
    actual: float | None = 105.0,
    low: float | None = 90.0,
    base: float | None = 100.0,
    high: float | None = 110.0,
) -> EffectRangeMeasurement:
    return EffectRangeMeasurement(
        action_id=action_id,
        action_family=family,
        causal_valid=causal_valid,
        actual_effect=actual,
        forecast_low=low,
        forecast_base=base,
        forecast_high=high,
    )


def test_calibration_requires_20_causal_valid_actions_in_same_family() -> None:
    measurements = [
        _measurement(f"advertising-{index}")
        for index in range(MIN_CAUSAL_VALID_ACTIONS - 1)
    ]
    measurements.append(
        _measurement(
            "advertising-non-causal",
            causal_valid=False,
            actual=None,
            low=None,
            base=None,
            high=None,
        )
    )
    measurements.extend(
        _measurement(f"pricing-{index}", family="pricing") for index in range(30)
    )

    result = propose_effect_range_calibration(
        measurements,
        action_family="advertising",
        current_version=7,
    )

    assert result.status == "insufficient_data"
    assert result.target_family_action_count == 20
    assert result.causal_valid_action_count == 19
    assert result.excluded_non_causal_count == 1
    assert result.ignored_other_family_count == 30
    assert result.signed_bias is None
    assert result.mae is None
    assert result.coverage is None
    assert result.proposed_adjustment is None


def test_calibration_computes_bias_mae_coverage_and_versioned_shift() -> None:
    measurements = [
        _measurement(f"inside-{index}", actual=100.0) for index in range(10)
    ]
    measurements.extend(
        _measurement(f"outside-{index}", actual=120.0) for index in range(10)
    )

    result = propose_effect_range_calibration(
        measurements,
        action_family="advertising",
        current_version=7,
    )

    assert result.status == "proposed"
    assert result.schema_version == CALIBRATION_SCHEMA_VERSION
    assert result.causal_valid_action_count == MIN_CAUSAL_VALID_ACTIONS
    assert result.signed_bias == pytest.approx(10.0)
    assert result.mae == pytest.approx(10.0)
    assert result.coverage == pytest.approx(0.5)

    adjustment = result.proposed_adjustment
    assert adjustment is not None
    assert adjustment.method == CALIBRATION_METHOD
    assert adjustment.base_version == 7
    assert adjustment.proposed_version == 8
    assert adjustment.low_delta == pytest.approx(10.0)
    assert adjustment.base_delta == pytest.approx(10.0)
    assert adjustment.high_delta == pytest.approx(10.0)
    assert adjustment.interval_width_delta == 0.0
    assert adjustment.auto_apply is False
    assert adjustment.requires_manual_approval is True


def test_negative_bias_shifts_all_bounds_without_widening_range() -> None:
    result = propose_effect_range_calibration(
        [
            _measurement(
                f"action-{index}",
                actual=85.0,
                low=80.0,
                base=100.0,
                high=120.0,
            )
            for index in range(MIN_CAUSAL_VALID_ACTIONS)
        ],
        action_family="advertising",
    )

    assert result.signed_bias == pytest.approx(-15.0)
    assert result.mae == pytest.approx(15.0)
    assert result.coverage == pytest.approx(1.0)
    assert result.proposed_adjustment is not None
    assert result.proposed_adjustment.low_delta == pytest.approx(-15.0)
    assert result.proposed_adjustment.base_delta == pytest.approx(-15.0)
    assert result.proposed_adjustment.high_delta == pytest.approx(-15.0)
    assert result.proposed_adjustment.interval_width_delta == 0.0


def test_duplicate_action_cannot_inflate_calibration_sample() -> None:
    measurements = [
        _measurement(f"action-{index}")
        for index in range(MIN_CAUSAL_VALID_ACTIONS - 1)
    ]
    measurements.append(_measurement("action-0"))

    with pytest.raises(ValueError, match="duplicate action_id"):
        propose_effect_range_calibration(
            measurements,
            action_family="advertising",
        )


def test_causal_valid_measurement_requires_ordered_finite_range() -> None:
    measurements = [
        _measurement(f"action-{index}")
        for index in range(MIN_CAUSAL_VALID_ACTIONS - 1)
    ]
    measurements.append(_measurement("bad-range", low=110.0, base=100.0, high=90.0))

    with pytest.raises(ValueError, match="low <= base <= high"):
        propose_effect_range_calibration(
            measurements,
            action_family="advertising",
        )


def test_calibration_contract_is_exported_from_manager_lifecycle() -> None:
    assert EffectRangeCalibrationResult.__module__.endswith("manager_lifecycle.calibration")
    assert ProposedEffectRangeAdjustment.__module__.endswith("manager_lifecycle.calibration")
