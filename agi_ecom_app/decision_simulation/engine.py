from __future__ import annotations

from hashlib import sha256
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from agi_ecom_app.decision_simulation.models import (
    AdvertisingPauseCandidateV1,
    CandidateSimulationResultV1,
    ConstraintViolationV1,
    DecisionSimulationRequestV1,
    DecisionSimulationResultV1,
    DistributionPairV1,
    DistributionSummaryV1,
    HistogramSummaryV1,
    RankingScoreBreakdownV1,
    ScalarEvidenceV1,
    SimulationBatchResultV1,
)


ADVERTISING_BOOTSTRAP_MODEL_VERSION = "advertising-bootstrap-v1"
_EMPIRICAL_DAY_COUNT = 7
_HISTOGRAM_BIN_COUNT = 10
NUMPY_RUNTIME_VERSION = np.__version__
BOOTSTRAP_BIT_GENERATOR = "PCG64"
QUANTILE_METHOD = "linear"


def simulation_runtime_card() -> dict[str, str]:
    return {
        "numpyVersion": NUMPY_RUNTIME_VERSION,
        "bitGenerator": BOOTSTRAP_BIT_GENERATOR,
        "quantileMethod": QUANTILE_METHOD,
    }


def derive_master_seed(snapshot_hash: str) -> int:
    """Return the stable, recorded seed of a snapshot without platform dependence."""

    normalized = _normalized_snapshot_hash(snapshot_hash)
    return int.from_bytes(bytes.fromhex(normalized)[:8], byteorder="big", signed=False)


def derive_batch_seed(snapshot_hash: str, candidate_id: str, batch_index: int) -> int:
    """Derive an independent uint64 seed for one candidate batch."""

    normalized = _normalized_snapshot_hash(snapshot_hash)
    if not candidate_id.strip():
        raise ValueError("candidate_id cannot be blank")
    if batch_index < 0:
        raise ValueError("batch_index cannot be negative")
    material = f"{normalized}\x00{candidate_id.strip()}\x00{batch_index}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], byteorder="big", signed=False)


def _normalized_snapshot_hash(snapshot_hash: str) -> str:
    normalized = snapshot_hash.strip().lower()
    if len(normalized) != 64 or any(value not in "0123456789abcdef" for value in normalized):
        raise ValueError("snapshot_hash must be a 64-character hexadecimal SHA-256")
    return normalized


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


def _validate_zero_evidence(
    evidence: ScalarEvidenceV1 | None,
    *,
    metric: str,
    nonzero_code: str,
    nonzero_message: str,
) -> list[ConstraintViolationV1]:
    if evidence is None or evidence.value is None:
        return [
            _violation(
                f"missing_{metric}",
                "needs_data",
                f"Required evidence is missing: {metric}.",
            )
        ]
    if not math.isfinite(evidence.value):
        return [
            _violation(
                f"invalid_{metric}",
                "blocked",
                f"Required evidence is not finite: {metric}.",
                evidence_ref=evidence.evidence_ref,
            )
        ]
    violations: list[ConstraintViolationV1] = []
    if evidence.evidence_ref is None:
        violations.append(
            _violation(
                f"missing_{metric}_evidence_ref",
                "needs_data",
                f"Required evidence has no source reference: {metric}.",
            )
        )
    if evidence.value != 0:
        violations.append(
            _violation(
                nonzero_code,
                "blocked",
                nonzero_message,
                evidence_ref=evidence.evidence_ref,
            )
        )
    return violations


def _validate_contribution_profit(
    evidence: ScalarEvidenceV1 | None,
) -> list[ConstraintViolationV1]:
    metric = "contribution_profit"
    if evidence is None or evidence.value is None:
        return [
            _violation(
                "missing_contribution_profit",
                "needs_data",
                "Campaign contribution profit is required.",
            )
        ]
    if not math.isfinite(evidence.value):
        return [
            _violation(
                "invalid_contribution_profit",
                "blocked",
                "Campaign contribution profit must be finite.",
                evidence_ref=evidence.evidence_ref,
            )
        ]
    violations: list[ConstraintViolationV1] = []
    if evidence.evidence_ref is None:
        violations.append(
            _violation(
                f"missing_{metric}_evidence_ref",
                "needs_data",
                "Campaign contribution profit has no source reference.",
            )
        )
    if evidence.value > 0:
        violations.append(
            _violation(
                "positive_campaign_contribution",
                "blocked",
                "A profitable campaign cannot be valued as a zero-cost pause.",
                evidence_ref=evidence.evidence_ref,
            )
        )
    return violations


def validate_advertising_pause_candidate(
    candidate: AdvertisingPauseCandidateV1,
) -> list[ConstraintViolationV1]:
    """Apply the complete deterministic hard gate before any random sampling."""

    evidence = candidate.evidence
    violations: list[ConstraintViolationV1] = []
    daily_spend = evidence.daily_spend
    if daily_spend is None or daily_spend.values_rub is None:
        violations.append(
            _violation(
                "missing_daily_spend_history",
                "needs_data",
                "Seven empirical completed-day campaign spends are required.",
            )
        )
    else:
        values = daily_spend.values_rub
        if daily_spend.evidence_ref is None:
            violations.append(
                _violation(
                    "missing_daily_spend_evidence_ref",
                    "needs_data",
                    "Daily campaign spend history has no source reference.",
                )
            )
        if len(values) != _EMPIRICAL_DAY_COUNT:
            violations.append(
                _violation(
                    "incomplete_daily_spend_history",
                    "needs_data",
                    "Exactly seven empirical completed-day campaign spends are required.",
                    evidence_ref=daily_spend.evidence_ref,
                )
            )
        elif not all(math.isfinite(value) for value in values):
            violations.append(
                _violation(
                    "invalid_daily_spend_history",
                    "blocked",
                    "Daily campaign spend history must contain finite values.",
                    evidence_ref=daily_spend.evidence_ref,
                )
            )
        elif any(value < 0 for value in values):
            violations.append(
                _violation(
                    "negative_daily_spend",
                    "blocked",
                    "Daily campaign spend cannot be negative.",
                    evidence_ref=daily_spend.evidence_ref,
                )
            )
        elif not any(value > 0 for value in values):
            violations.append(
                _violation(
                    "non_positive_daily_spend_history",
                    "blocked",
                    "Observed campaign spend does not prove a positive avoided cost.",
                    evidence_ref=daily_spend.evidence_ref,
                )
            )

    violations.extend(
        _validate_zero_evidence(
            evidence.stock_on_hand,
            metric="stock_on_hand",
            nonzero_code="stock_available",
            nonzero_message="A campaign cannot be paused under the zero-stock model while stock is available.",
        )
    )
    violations.extend(
        _validate_zero_evidence(
            evidence.supplier_inbound,
            metric="supplier_inbound",
            nonzero_code="confirmed_supplier_inbound_exists",
            nonzero_message="Confirmed supplier inbound requires recalculation to its ETA.",
        )
    )
    violations.extend(
        _validate_zero_evidence(
            evidence.ad_orders,
            metric="ad_orders",
            nonzero_code="advertising_orders_exist",
            nonzero_message="Attributed advertising orders invalidate a zero-opportunity-cost pause.",
        )
    )
    violations.extend(
        _validate_zero_evidence(
            evidence.attributed_revenue,
            metric="attributed_revenue",
            nonzero_code="attributed_revenue_exists",
            nonzero_message="Attributed revenue invalidates a zero-opportunity-cost pause.",
        )
    )
    violations.extend(_validate_contribution_profit(evidence.contribution_profit))
    return violations


def _histogram(values: NDArray[np.float64]) -> HistogramSummaryV1:
    counts, bin_edges = np.histogram(values, bins=_HISTOGRAM_BIN_COUNT)
    return HistogramSummaryV1(
        bin_edges=[float(item) for item in bin_edges],
        counts=[int(item) for item in counts],
    )


def _distribution(values: NDArray[np.float64]) -> DistributionSummaryV1:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("distribution requires a non-empty one-dimensional array")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("distribution values must be finite")
    quantiles = np.quantile(
        values,
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        method=QUANTILE_METHOD,
    )
    tail_count = max(1, int(math.ceil(values.size * 0.10)))
    cvar_10 = float(np.mean(np.sort(values)[:tail_count]))
    return DistributionSummaryV1(
        count=int(values.size),
        mean=float(np.mean(values)),
        p05=float(quantiles[0]),
        p10=float(quantiles[1]),
        p25=float(quantiles[2]),
        p50=float(quantiles[3]),
        p75=float(quantiles[4]),
        p90=float(quantiles[5]),
        p95=float(quantiles[6]),
        cvar_10=cvar_10,
        probability_positive=float(np.mean(values > 0)),
        histogram=_histogram(values),
    )


def _distribution_pair(
    total_values: NDArray[np.float64],
    *,
    horizon_days: int,
) -> DistributionPairV1:
    per_day_values = total_values / float(horizon_days)
    return DistributionPairV1(
        total=_distribution(total_values),
        per_day=_distribution(per_day_values),
    )


def _evidence_refs(candidate: AdvertisingPauseCandidateV1) -> list[str]:
    evidence_items: Iterable[object | None] = (
        candidate.evidence.daily_spend,
        candidate.evidence.stock_on_hand,
        candidate.evidence.supplier_inbound,
        candidate.evidence.ad_orders,
        candidate.evidence.attributed_revenue,
        candidate.evidence.contribution_profit,
    )
    references: list[str] = []
    for item in evidence_items:
        reference = getattr(item, "evidence_ref", None)
        if isinstance(reference, str) and reference not in references:
            references.append(reference)
    return references


def _simulate_batch_values(
    *,
    snapshot_hash: str,
    candidate: AdvertisingPauseCandidateV1,
    batch_index: int,
    batch_size: int,
) -> tuple[int, NDArray[np.float64]]:
    daily_spend = candidate.evidence.daily_spend
    if daily_spend is None or daily_spend.values_rub is None:
        raise ValueError("validated candidate has no daily spend history")
    observed = np.asarray(daily_spend.values_rub, dtype=np.float64)
    seed = derive_batch_seed(snapshot_hash, candidate.candidate_id, batch_index)
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled_indices = generator.integers(
        low=0,
        high=observed.size,
        size=(batch_size, candidate.horizon_days),
    )
    total_values = np.sum(observed[sampled_indices], axis=1, dtype=np.float64)
    return seed, total_values


class DecisionSimulationEngine:
    """Pure, deterministic advertising bootstrap simulation engine."""

    def validate_candidate(
        self,
        candidate: AdvertisingPauseCandidateV1,
    ) -> list[ConstraintViolationV1]:
        return validate_advertising_pause_candidate(candidate)

    def simulate_candidate(
        self,
        request: DecisionSimulationRequestV1,
        candidate: AdvertisingPauseCandidateV1,
    ) -> CandidateSimulationResultV1:
        violations = self.validate_candidate(candidate)
        if violations:
            status = (
                "blocked"
                if any(item.severity == "blocked" for item in violations)
                else "needs_data"
            )
            return CandidateSimulationResultV1(
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                status=status,
                constraint_violations=violations,
                evidence_refs=_evidence_refs(candidate),
            )

        batch_results = [
            self.simulate_batch(request, candidate, batch_index)
            for batch_index in range(request.sample_count // request.batch_size)
        ]
        return self.aggregate_candidate(request, candidate, batch_results)

    def simulate_batch(
        self,
        request: DecisionSimulationRequestV1,
        candidate: AdvertisingPauseCandidateV1,
        batch_index: int,
    ) -> SimulationBatchResultV1:
        """Compute one independently reproducible batch for immediate persistence."""

        violations = self.validate_candidate(candidate)
        if violations:
            raise ValueError("cannot simulate a candidate that failed the hard gate")
        batch_count = request.sample_count // request.batch_size
        if batch_index < 0 or batch_index >= batch_count:
            raise ValueError("batch_index is outside the simulation request")
        seed, total_values = _simulate_batch_values(
            snapshot_hash=request.snapshot_hash,
            candidate=candidate,
            batch_index=batch_index,
            batch_size=request.batch_size,
        )
        return SimulationBatchResultV1(
            candidate_id=candidate.candidate_id,
            batch_index=batch_index,
            seed=seed,
            sample_count=request.batch_size,
            distribution=_distribution_pair(
                total_values,
                horizon_days=candidate.horizon_days,
            ),
        )

    def aggregate_candidate(
        self,
        request: DecisionSimulationRequestV1,
        candidate: AdvertisingPauseCandidateV1,
        batch_results: Iterable[SimulationBatchResultV1],
    ) -> CandidateSimulationResultV1:
        """Reproduce exact final quantiles from the durable batch seed journal."""

        violations = self.validate_candidate(candidate)
        if violations:
            raise ValueError("cannot aggregate a candidate that failed the hard gate")
        ordered = sorted(batch_results, key=lambda item: item.batch_index)
        batch_count = request.sample_count // request.batch_size
        if [item.batch_index for item in ordered] != list(range(batch_count)):
            raise ValueError("candidate aggregation requires every batch exactly once")
        candidate_values: list[NDArray[np.float64]] = []
        for batch in ordered:
            expected_seed, total_values = _simulate_batch_values(
                snapshot_hash=request.snapshot_hash,
                candidate=candidate,
                batch_index=batch.batch_index,
                batch_size=request.batch_size,
            )
            if batch.seed != expected_seed or batch.sample_count != request.batch_size:
                raise ValueError("durable batch seed/sample count does not match the snapshot")
            candidate_values.append(total_values)

        all_values = np.concatenate(candidate_values)
        distribution = _distribution_pair(all_values, horizon_days=candidate.horizon_days)
        score_breakdown = RankingScoreBreakdownV1(
            cvar_10_per_day=distribution.per_day.cvar_10,
            p10_per_day=distribution.per_day.p10,
            mean_per_day=distribution.per_day.mean,
            execution_risk=candidate.execution_risk,
        )
        return CandidateSimulationResultV1(
            candidate_id=candidate.candidate_id,
            action_id=candidate.action_id,
            status="completed",
            evidence_refs=_evidence_refs(candidate),
            batch_results=ordered,
            distribution=distribution,
            score_breakdown=score_breakdown,
        )

    def run(self, request: DecisionSimulationRequestV1) -> DecisionSimulationResultV1:
        candidate_results = [
            self.simulate_candidate(request, candidate) for candidate in request.candidates
        ]
        completed = [item for item in candidate_results if item.status == "completed"]

        def ranking_key(item: CandidateSimulationResultV1) -> tuple[float, float, float, float, str]:
            score = item.score_breakdown
            if score is None:  # Defensive: completed result validation already prevents this.
                raise ValueError("completed candidate is missing score_breakdown")
            return (
                -score.cvar_10_per_day,
                -score.p10_per_day,
                -score.mean_per_day,
                score.execution_risk,
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
        return DecisionSimulationResultV1(
            snapshot_hash=request.snapshot_hash,
            model_version=request.model_version,
            master_seed=derive_master_seed(request.snapshot_hash),
            sample_count=request.sample_count,
            batch_size=request.batch_size,
            candidate_count=len(request.candidates),
            completed_candidate_count=len(completed),
            candidates=finalized,
            ranked_candidate_ids=[item.candidate_id for item in ranked],
            winner_candidate_id=winner_candidate_id,
        )


def run_decision_simulation(
    request: DecisionSimulationRequestV1,
) -> DecisionSimulationResultV1:
    return DecisionSimulationEngine().run(request)


__all__ = [
    "ADVERTISING_BOOTSTRAP_MODEL_VERSION",
    "DecisionSimulationEngine",
    "derive_batch_seed",
    "derive_master_seed",
    "run_decision_simulation",
    "validate_advertising_pause_candidate",
]
