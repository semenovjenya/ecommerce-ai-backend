from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import math
from typing import Any

import numpy as np

from agi_ecom_app.decision_simulation.pricing_engine import (
    MIN_PLAUSIBLE_PRICE_ELASTICITY,
    simulate_pricing_batch_values,
)
from agi_ecom_app.decision_simulation.pricing_models import (
    PricingCandidateV1,
    PricingSimulationRequestV1,
)


_ELASTICITY_GAP_CODES = frozenset(
    {
        "missing_elasticity_evidence_ref",
        "missing_elasticity_source",
        "fallback_elasticity_not_allowed",
        "missing_elasticity_bootstrap_values",
        "insufficient_elasticity_bootstrap_values",
        "invalid_price_elasticity",
        "implausible_price_elasticity_magnitude",
    }
)
_MIN_OWN_PLAUSIBLE_DRAWS = 3
_MIN_OWN_PLAUSIBLE_SHARE = 0.25
_DEFAULT_DRAW_COUNT = 100
_MIN_HYPOTHESIS_ELASTICITY = MIN_PLAUSIBLE_PRICE_ELASTICITY
_MAX_HYPOTHESIS_ELASTICITY = -0.05
_DEFAULT_SENSITIVITY_GRID = (
    -0.50,
    -0.75,
    -1.00,
    -1.25,
    -1.50,
    -2.00,
    -2.50,
)
_PRIOR_MULTIPLIERS = (0.70, 0.82, 0.92, 1.00, 1.08, 1.18, 1.30)


def build_pricing_hypotheses(
    request: PricingSimulationRequestV1,
    strict_result: Any,
    price_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build non-executable price sensitivities beside a fail-closed result.

    The strict engine and its result remain authoritative.  This helper only
    replaces the elasticity draws in a deep copy of the request, then reuses
    the deterministic pricing sampler to quantify the already-pinned price
    candidates as hypotheses.
    """

    strict_gaps = _strict_evidence_gaps(strict_result)
    basic_gaps = _basic_evidence_gaps(request)
    evidence_gaps = _deduplicate([*strict_gaps, *basic_gaps])

    if basic_gaps:
        return _qualitative_result(
            request,
            evidence_gaps=evidence_gaps,
            assumptions=[
                "Price variants are hypotheses only and cannot be executed.",
                "Numerical effects are withheld until the listed non-elastic evidence exists.",
                "The strict pricing result remains authoritative and unchanged.",
            ],
        )

    (
        elasticity_values,
        calculation_mode,
        confidence,
        confidence_score,
        assumptions,
        mode_gaps,
    ) = _hypothesis_elasticity(request, price_prior)
    evidence_gaps = _deduplicate([*evidence_gaps, *mode_gaps])
    hypothesis_request = _request_with_elasticity(request, elasticity_values)

    try:
        rows = [
            _simulate_candidate(hypothesis_request, candidate)
            for candidate in hypothesis_request.candidates
        ]
    except (ArithmeticError, IndexError, TypeError, ValueError):
        return _qualitative_result(
            request,
            evidence_gaps=_deduplicate(
                [*evidence_gaps, "hypothesis_simulation_failed"]
            ),
            assumptions=[
                "Price variants are hypotheses only and cannot be executed.",
                "The available evidence could not produce a stable numerical sensitivity replay.",
                "The strict pricing result remains authoritative and unchanged.",
            ],
        )

    _rank_candidates(rows, current_price_rub=request.current_price_rub)
    winner = next((row["candidateId"] for row in rows if row["isWinner"]), None)
    if not any(gap in _ELASTICITY_GAP_CODES for gap in evidence_gaps):
        evidence_gaps.append("strict_result_has_no_unreliable_elasticity_gap")

    return {
        "skuId": request.sku_id,
        "currentPriceRub": request.current_price_rub,
        "horizonDays": request.horizon_days,
        "status": "hypothesis",
        "executionEligible": False,
        "calculationMode": calculation_mode,
        "confidence": confidence,
        "confidenceScore": confidence_score,
        "assumptions": [
            "Price variants are hypotheses only and cannot be executed.",
            *assumptions,
            "The strict pricing result remains authoritative and unchanged.",
        ],
        "evidenceGaps": evidence_gaps,
        "winnerCandidateId": winner,
        "candidates": rows,
    }


def _strict_evidence_gaps(strict_result: Any) -> list[str]:
    candidates = _field(strict_result, "candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ["missing_strict_pricing_result"]
    result: list[str] = []
    for candidate in candidates:
        violations = _field(candidate, "constraint_violations", "constraintViolations")
        if not isinstance(violations, Sequence) or isinstance(
            violations, (str, bytes)
        ):
            continue
        for violation in violations:
            code = _field(violation, "code")
            if isinstance(code, str) and code:
                result.append(code)
    return _deduplicate(result)


def _basic_evidence_gaps(request: PricingSimulationRequestV1) -> list[str]:
    evidence = request.evidence
    observations = [
        item
        for item in (evidence.daily_observations.values or [])
        if item.completed and item.not_observed_out_of_stock
    ]
    gaps: list[str] = []
    if not observations:
        gaps.append("missing_completed_day_observations")
    elif not any(
        math.isfinite(item.price_rub)
        and item.price_rub > 0
        and math.isfinite(item.units_sold)
        and item.units_sold > 0
        for item in observations
    ):
        gaps.append("non_positive_demand_history")

    realization = evidence.realization.bootstrap_values or []
    if not realization:
        gaps.append("missing_realization_bootstrap_values")
    elif any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in realization
    ) or not any(value > 0 for value in realization):
        gaps.append("invalid_realization_distribution")

    stock = evidence.available_stock.units
    if stock is None:
        gaps.append("missing_available_stock")
    elif not math.isfinite(stock) or stock < 0:
        gaps.append("invalid_available_stock")

    economics = evidence.unit_economics
    rate = economics.net_revenue_rate
    if rate is None:
        gaps.append("missing_net_revenue_rate")
    elif not math.isfinite(rate) or not 0 < rate <= 1:
        gaps.append("invalid_net_revenue_rate")
    cost = economics.variable_cost_per_unit_rub
    if cost is None:
        gaps.append("missing_variable_cost_per_unit")
    elif not math.isfinite(cost) or cost < 0:
        gaps.append("invalid_variable_cost_per_unit")
    return gaps


def _hypothesis_elasticity(
    request: PricingSimulationRequestV1,
    price_prior: Mapping[str, Any] | None,
) -> tuple[list[float], str, str, float, list[str], list[str]]:
    original = list(request.evidence.elasticity.bootstrap_values or [])
    plausible = [value for value in original if _is_plausible_elasticity(value)]
    plausible_share = len(plausible) / len(original) if original else 0.0
    target_count = len(original) or _DEFAULT_DRAW_COUNT

    if (
        len(plausible) >= _MIN_OWN_PLAUSIBLE_DRAWS
        and plausible_share >= _MIN_OWN_PLAUSIBLE_SHARE
    ):
        replacement_index = 0
        transformed: list[float] = []
        for value in original:
            if _is_plausible_elasticity(value):
                transformed.append(float(value))
            else:
                transformed.append(float(plausible[replacement_index % len(plausible)]))
                replacement_index += 1
        score = round(min(0.60, 0.30 + plausible_share * 0.30), 3)
        return (
            transformed,
            "observed_plausible_elasticity_sensitivity",
            "medium" if score >= 0.5 else "low",
            score,
            [
                "Plausible own-SKU elasticity draws are retained.",
                (
                    "Invalid or extreme own-SKU draws are replaced deterministically "
                    "without changing draw count or context-coefficient alignment."
                ),
            ],
            [],
        )

    prior = _finite_number(
        price_prior.get("elasticity_to_apply") if price_prior is not None else None
    )
    if prior is not None:
        center = _bounded_negative(prior)
        prior_score, prior_assumptions, prior_gaps = _price_prior_quality(
            request,
            price_prior or {},
        )
        values = _repeat_to_length(
            [_bounded_negative(center * multiplier) for multiplier in _PRIOR_MULTIPLIERS],
            target_count,
        )
        return (
            values,
            "price_prior_elasticity_sensitivity",
            "low",
            prior_score,
            [
                (
                    "The price mart elasticity is used as a bounded sensitivity prior, "
                    "not as causal proof."
                ),
                (
                    "Sensitivity draws vary deterministically around the prior and "
                    "preserve context-coefficient alignment."
                ),
                *prior_assumptions,
            ],
            prior_gaps,
        )

    return (
        _repeat_to_length(_DEFAULT_SENSITIVITY_GRID, target_count),
        "bounded_elasticity_sensitivity_grid",
        "low",
        0.15,
        [
            "No reliable own-SKU elasticity or price-mart prior is available.",
            (
                "A deterministic bounded negative-elasticity grid is used only to "
                "compare scenario sensitivity."
            ),
        ],
        ["missing_reliable_elasticity_prior"],
    )


def _price_prior_quality(
    request: PricingSimulationRequestV1,
    price_prior: Mapping[str, Any],
) -> tuple[float, list[str], list[str]]:
    score = 0.30
    assumptions: list[str] = []
    gaps: list[str] = []

    confidence = str(price_prior.get("decision_confidence") or "").casefold()
    if confidence in {"low", "insufficient", "unknown"}:
        score = min(score, 0.20)
        gaps.append("low_confidence_price_prior")
        assumptions.append("The price-mart decision confidence is low.")

    status = str(price_prior.get("decision_status") or "").casefold()
    if status in {"blocked", "needs_data", "needs_review", "failed"}:
        score = min(score, 0.15)
        gaps.append("non_actionable_price_prior")
        assumptions.append("The price-mart decision itself is not actionable.")

    as_of = (
        request.evidence.forecast_context.snapshot_date
        if request.evidence.forecast_context is not None
        else request.evidence.available_stock.snapshot_date
    )
    prior_date = _date_value(
        price_prior.get("price_ai_run_date") or price_prior.get("snapshot_date")
    )
    if prior_date is None:
        score = min(score, 0.20)
        gaps.append("missing_price_prior_freshness")
    elif as_of is not None and (as_of - prior_date).days > 30:
        score = min(score, 0.10)
        gaps.append("stale_price_prior")
        assumptions.append(
            "The price-mart prior is stale and is used only as a broad sensitivity center."
        )

    return round(score, 3), assumptions, gaps


def _request_with_elasticity(
    request: PricingSimulationRequestV1,
    elasticity_values: list[float],
) -> PricingSimulationRequestV1:
    copied = request.model_copy(deep=True)
    copied_elasticity = copied.evidence.elasticity.model_copy(
        update={"bootstrap_values": list(elasticity_values)}
    )
    copied_evidence = copied.evidence.model_copy(
        update={"elasticity": copied_elasticity},
        deep=True,
    )
    return copied.model_copy(update={"evidence": copied_evidence}, deep=True)


def _simulate_candidate(
    request: PricingSimulationRequestV1,
    candidate: PricingCandidateV1,
) -> dict[str, Any]:
    incremental: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    units: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    stockouts: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
    for batch_index in range(request.sample_count // request.batch_size):
        values = simulate_pricing_batch_values(request, candidate, batch_index)
        incremental.append(values.incremental_contribution_profit)
        units.append(values.units_sold)
        stockouts.append(values.stockout)
    all_incremental = np.concatenate(incremental)
    all_units = np.concatenate(units)
    all_stockouts = np.concatenate(stockouts)
    economics = request.evidence.unit_economics
    if (
        economics.net_revenue_rate is None
        or economics.variable_cost_per_unit_rub is None
    ):
        raise ValueError("hypothesis simulation requires unit economics")
    return {
        "candidateId": candidate.candidate_id,
        "actionId": candidate.action_id,
        "operation": candidate.operation,
        "priceRub": candidate.price_rub,
        "status": "hypothesis",
        "executionEligible": False,
        "mean": _rounded(float(np.mean(all_incremental))),
        "p10": _rounded(
            float(np.quantile(all_incremental, 0.1, method="linear"))
        ),
        "cvar10": _rounded(_cvar_10(all_incremental)),
        "probPositive": _rounded(float(np.mean(all_incremental > 0))),
        "expectedUnits": _rounded(float(np.mean(all_units))),
        "stockout": _rounded(float(np.mean(all_stockouts))),
        "unitContribution": _rounded(
            candidate.price_rub * economics.net_revenue_rate
            - economics.variable_cost_per_unit_rub
        ),
        "rank": None,
        "isWinner": False,
    }


def _rank_candidates(
    rows: list[dict[str, Any]],
    *,
    current_price_rub: float,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["cvar10"]),
            -float(row["p10"]),
            -float(row["mean"]),
            abs(float(row["priceRub"]) - current_price_rub),
            str(row["actionId"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["isWinner"] = rank == 1


def _qualitative_result(
    request: PricingSimulationRequestV1,
    *,
    evidence_gaps: list[str],
    assumptions: list[str],
) -> dict[str, Any]:
    return {
        "skuId": request.sku_id,
        "currentPriceRub": request.current_price_rub,
        "horizonDays": request.horizon_days,
        "status": "qualitative",
        "executionEligible": False,
        "calculationMode": "qualitative_only",
        "confidence": "insufficient",
        "confidenceScore": 0.0,
        "assumptions": assumptions,
        "evidenceGaps": evidence_gaps,
        "winnerCandidateId": None,
        "candidates": [
            {
                "candidateId": candidate.candidate_id,
                "actionId": candidate.action_id,
                "operation": candidate.operation,
                "priceRub": candidate.price_rub,
                "status": "hypothesis",
                "executionEligible": False,
                "mean": None,
                "p10": None,
                "cvar10": None,
                "probPositive": None,
                "expectedUnits": None,
                "stockout": None,
                "unitContribution": None,
                "rank": None,
                "isWinner": False,
            }
            for candidate in request.candidates
        ],
    }


def _is_plausible_elasticity(value: float) -> bool:
    return (
        math.isfinite(value)
        and _MIN_HYPOTHESIS_ELASTICITY <= value <= _MAX_HYPOTHESIS_ELASTICITY
    )


def _bounded_negative(value: float) -> float:
    normalized = -abs(float(value))
    return min(
        _MAX_HYPOTHESIS_ELASTICITY,
        max(_MIN_HYPOTHESIS_ELASTICITY, normalized),
    )


def _repeat_to_length(values: Sequence[float], length: int) -> list[float]:
    if not values or length < 1:
        raise ValueError("elasticity sensitivity requires values and a positive length")
    return [float(values[index % len(values)]) for index in range(length)]


def _finite_number(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) and normalized != 0 else None


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _cvar_10(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    ordered = np.sort(values)
    count = max(1, math.ceil(len(ordered) * 0.1))
    return float(np.mean(ordered[:count]))


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("hypothesis metric is not finite")
    return round(value, 6)


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = ["build_pricing_hypotheses"]
