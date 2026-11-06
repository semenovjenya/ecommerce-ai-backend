"""Persistence, authorization, execution and measurement for manager actions."""

from agi_ecom_app.manager_lifecycle.auth import ManagerActor
from agi_ecom_app.manager_lifecycle.advertising_measurement import (
    AdvertisingMeasurementProvider,
    register_advertising_measurement_provider,
)
from agi_ecom_app.manager_lifecycle.calibration import (
    CALIBRATION_METHOD,
    CALIBRATION_SCHEMA_VERSION,
    MIN_CAUSAL_VALID_ACTIONS,
    EffectRangeCalibrationResult,
    EffectRangeMeasurement,
    ProposedEffectRangeAdjustment,
    propose_effect_range_calibration,
)
from agi_ecom_app.manager_lifecycle.jobs import (
    ManagerMeasurementReadout,
    register_manager_measurement_provider,
)
from agi_ecom_app.manager_lifecycle.pricing_measurement import (
    PricingMeasurementProvider,
    register_pricing_measurement_provider,
)
from agi_ecom_app.manager_lifecycle.service import (
    ActuationResult,
    LiveGuardrailSnapshot,
    ManagerActionActuator,
    ManagerLiveGuardrailProvider,
    PreflightResult,
    approve_manager_action,
    complete_manual_manager_action,
    persist_manager_decision_bundle,
    record_manager_action_measurement,
    evaluate_live_guardrails,
    register_manager_action_actuator,
    register_manager_live_guardrail_provider,
    reject_manager_action,
    verify_manager_action,
)

__all__ = [
    "ActuationResult",
    "AdvertisingMeasurementProvider",
    "CALIBRATION_METHOD",
    "CALIBRATION_SCHEMA_VERSION",
    "EffectRangeCalibrationResult",
    "EffectRangeMeasurement",
    "ManagerActionActuator",
    "ManagerLiveGuardrailProvider",
    "LiveGuardrailSnapshot",
    "ManagerActor",
    "ManagerMeasurementReadout",
    "MIN_CAUSAL_VALID_ACTIONS",
    "PreflightResult",
    "PricingMeasurementProvider",
    "ProposedEffectRangeAdjustment",
    "approve_manager_action",
    "complete_manual_manager_action",
    "persist_manager_decision_bundle",
    "propose_effect_range_calibration",
    "record_manager_action_measurement",
    "evaluate_live_guardrails",
    "register_manager_action_actuator",
    "register_advertising_measurement_provider",
    "register_manager_live_guardrail_provider",
    "register_manager_measurement_provider",
    "register_pricing_measurement_provider",
    "reject_manager_action",
    "verify_manager_action",
]
