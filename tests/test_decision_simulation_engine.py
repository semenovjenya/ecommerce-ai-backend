from __future__ import annotations

import math
import time

import numpy as np
import pytest

from agi_ecom_app.decision_simulation import (
    AdvertisingPauseCandidateV1,
    AdvertisingPauseEvidenceV1,
    CandidateSimulationResultV1,
    DailySpendEvidenceV1,
    DecisionSimulationEngine,
    DecisionSimulationRequestV1,
    DistributionPairV1,
    DistributionSummaryV1,
    HistogramSummaryV1,
    RankingScoreBreakdownV1,
    ScalarEvidenceV1,
    SimulationBatchResultV1,
    derive_batch_seed,
    run_decision_simulation,
)


SNAPSHOT_HASH = "a" * 64


def _scalar(value: float | None, reference: str | None) -> ScalarEvidenceV1:
    return ScalarEvidenceV1(value=value, evidence_ref=reference)


def _candidate(
    suffix: str,
    daily_spend: list[float] | None,
    *,
    execution_risk: float = 0,
    stock_on_hand: float | None = 0,
    supplier_inbound: float | None = 0,
    ad_orders: float | None = 0,
    attributed_revenue: float | None = 0,
    contribution_profit: float | None = 0,
) -> AdvertisingPauseCandidateV1:
    return AdvertisingPauseCandidateV1(
        candidate_id=f"candidate-{suffix}",
        action_id=f"action-{suffix}",
        execution_risk=execution_risk,
        evidence=AdvertisingPauseEvidenceV1(
            daily_spend=DailySpendEvidenceV1(
                values_rub=daily_spend,
                evidence_ref=f"evidence:spend:{suffix}",
            ),
            stock_on_hand=_scalar(stock_on_hand, f"evidence:stock:{suffix}"),
            supplier_inbound=_scalar(
                supplier_inbound, f"evidence:inbound:{suffix}"
            ),
            ad_orders=_scalar(ad_orders, f"evidence:orders:{suffix}"),
            attributed_revenue=_scalar(
                attributed_revenue, f"evidence:revenue:{suffix}"
            ),
            contribution_profit=_scalar(
                contribution_profit, f"evidence:profit:{suffix}"
            ),
        ),
    )


def _request(*candidates: AdvertisingPauseCandidateV1) -> DecisionSimulationRequestV1:
    return DecisionSimulationRequestV1(
        snapshot_hash=SNAPSHOT_HASH,
        candidates=list(candidates),
    )


def test_default_run_is_500_samples_in_ten_observable_batches() -> None:
    request = _request(
        _candidate("one", [80, 90, 100, 110, 120, 130, 140]),
        _candidate("two", [90, 100, 110, 120, 130, 140, 150]),
        _candidate("three", [100, 110, 120, 130, 140, 150, 160]),
    )

    result = run_decision_simulation(request)

    assert result.sample_count == 500
    assert result.batch_size == 50
    assert result.candidate_count == 3
    assert result.completed_candidate_count == 3
    assert result.winner_candidate_id == "candidate-three"
    for candidate in result.candidates:
        assert candidate.status == "completed"
        assert candidate.distribution is not None
        assert candidate.distribution.total.count == 500
        assert candidate.distribution.per_day.count == 500
        assert len(candidate.batch_results) == 10
        assert [batch.batch_index for batch in candidate.batch_results] == list(range(10))
        assert all(batch.sample_count == 50 for batch in candidate.batch_results)


def test_seven_candidates_times_500_samples_finish_under_five_seconds() -> None:
    request = _request(
        *[
            _candidate(
                str(index),
                [80 + index + day * 10 for day in range(7)],
            )
            for index in range(7)
        ]
    )

    started = time.perf_counter()
    result = run_decision_simulation(request)
    duration = time.perf_counter() - started

    assert result.completed_candidate_count == 7
    assert sum(
        candidate.distribution.total.count
        for candidate in result.candidates
        if candidate.distribution is not None
    ) == 3_500
    assert duration < 5.0


def test_same_snapshot_candidates_and_seeds_are_exactly_reproducible() -> None:
    request = _request(
        _candidate("one", [10, 20, 30, 40, 50, 60, 70]),
        _candidate("two", [11, 21, 31, 41, 51, 61, 71]),
        _candidate("three", [12, 22, 32, 42, 52, 62, 72]),
    )

    first = run_decision_simulation(request)
    second = run_decision_simulation(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    first_seeds = [batch.seed for batch in first.candidates[0].batch_results]
    assert len(set(first_seeds)) == 10
    assert first_seeds[0] == derive_batch_seed(
        SNAPSHOT_HASH, "candidate-one", 0
    )
    assert first_seeds[0] != derive_batch_seed(
        SNAPSHOT_HASH, "candidate-two", 0
    )


def test_candidate_aggregate_is_computed_from_all_ten_batches() -> None:
    observed = [20, 40, 60, 80, 100, 120, 140]
    candidate = _candidate("one", observed)
    request = _request(
        candidate,
        _candidate("two", [1, 2, 3, 4, 5, 6, 7]),
        _candidate("three", [2, 3, 4, 5, 6, 7, 8]),
    )

    result = run_decision_simulation(request)
    aggregate = result.candidates[0].distribution
    assert aggregate is not None

    observed_array = np.asarray(observed, dtype=np.float64)
    independently_replayed: list[np.ndarray] = []
    for batch_index in range(10):
        generator = np.random.Generator(
            np.random.PCG64(
                derive_batch_seed(SNAPSHOT_HASH, candidate.candidate_id, batch_index)
            )
        )
        indices = generator.integers(0, 7, size=(50, 7))
        independently_replayed.append(observed_array[indices].sum(axis=1))
    totals = np.concatenate(independently_replayed)
    per_day = totals / 7
    sorted_per_day = np.sort(per_day)

    assert aggregate.total.mean == pytest.approx(float(np.mean(totals)))
    assert aggregate.per_day.mean == pytest.approx(float(np.mean(per_day)))
    assert aggregate.per_day.p10 == pytest.approx(
        float(np.quantile(per_day, 0.10, method="linear"))
    )
    assert aggregate.per_day.p50 == pytest.approx(
        float(np.quantile(per_day, 0.50, method="linear"))
    )
    assert aggregate.per_day.p90 == pytest.approx(
        float(np.quantile(per_day, 0.90, method="linear"))
    )
    assert aggregate.per_day.cvar_10 == pytest.approx(
        float(np.mean(sorted_per_day[: math.ceil(500 * 0.10)]))
    )
    assert sum(aggregate.per_day.histogram.counts) == 500


def test_missing_empirical_or_required_evidence_is_needs_data_without_sampling() -> None:
    incomplete = _candidate("incomplete", [80, 100, 120])
    incomplete.evidence.supplier_inbound = None
    request = _request(
        incomplete,
        _candidate("two", [1, 2, 3, 4, 5, 6, 7]),
        _candidate("three", [2, 3, 4, 5, 6, 7, 8]),
    )

    result = run_decision_simulation(request)
    rejected = result.candidates[0]

    assert rejected.status == "needs_data"
    assert rejected.batch_results == []
    assert rejected.distribution is None
    assert rejected.score_breakdown is None
    assert {item.code for item in rejected.constraint_violations} == {
        "incomplete_daily_spend_history",
        "missing_supplier_inbound",
    }


@pytest.mark.parametrize(
    ("override", "expected_code"),
    [
        ({"stock_on_hand": 2}, "stock_available"),
        ({"supplier_inbound": 5}, "confirmed_supplier_inbound_exists"),
        ({"ad_orders": 1}, "advertising_orders_exist"),
        ({"attributed_revenue": 500}, "attributed_revenue_exists"),
        ({"contribution_profit": 1}, "positive_campaign_contribution"),
    ],
)
def test_incompatible_pause_evidence_blocks_candidate(
    override: dict[str, float], expected_code: str
) -> None:
    unsafe = _candidate("unsafe", [10, 20, 30, 40, 50, 60, 70], **override)
    request = _request(
        unsafe,
        _candidate("two", [1, 2, 3, 4, 5, 6, 7]),
        _candidate("three", [2, 3, 4, 5, 6, 7, 8]),
    )

    result = run_decision_simulation(request)

    assert result.candidates[0].status == "blocked"
    assert expected_code in {
        item.code for item in result.candidates[0].constraint_violations
    }
    assert result.candidates[0].batch_results == []


def _fixed_distribution() -> DistributionPairV1:
    histogram = HistogramSummaryV1(bin_edges=[0, 1], counts=[50])
    summary = DistributionSummaryV1(
        count=50,
        mean=1,
        p05=1,
        p10=1,
        p25=1,
        p50=1,
        p75=1,
        p90=1,
        p95=1,
        cvar_10=1,
        probability_positive=1,
        histogram=histogram,
    )
    return DistributionPairV1(total=summary, per_day=summary)


def test_ranking_uses_cvar_then_p10_mean_risk_and_action_id() -> None:
    candidates = [
        _candidate("a", [1] * 7),
        _candidate("b", [1] * 7),
        _candidate("c", [1] * 7),
        _candidate("d", [1] * 7),
        _candidate("e", [1] * 7),
    ]
    candidates[0].action_id = "action-z-risky"
    candidates[1].action_id = "action-z"
    candidates[2].action_id = "action-a"
    score_by_candidate = {
        "candidate-a": (10, 20, 30, 2),
        "candidate-b": (10, 20, 30, 1),
        "candidate-c": (10, 20, 30, 1),
        "candidate-d": (10, 21, 0, 99),
        "candidate-e": (11, 0, 0, 99),
    }

    class FixedScoreEngine(DecisionSimulationEngine):
        def simulate_candidate(self, request, candidate):  # type: ignore[no-untyped-def]
            cvar, p10, mean, risk = score_by_candidate[candidate.candidate_id]
            distribution = _fixed_distribution()
            return CandidateSimulationResultV1(
                candidate_id=candidate.candidate_id,
                action_id=candidate.action_id,
                status="completed",
                evidence_refs=["evidence:test"],
                batch_results=[
                    SimulationBatchResultV1(
                        candidate_id=candidate.candidate_id,
                        batch_index=0,
                        seed=1,
                        sample_count=50,
                        distribution=distribution,
                    )
                ],
                distribution=distribution,
                score_breakdown=RankingScoreBreakdownV1(
                    cvar_10_per_day=cvar,
                    p10_per_day=p10,
                    mean_per_day=mean,
                    execution_risk=risk,
                ),
            )

    result = FixedScoreEngine().run(_request(*candidates))

    assert result.ranked_candidate_ids == [
        "candidate-e",
        "candidate-d",
        "candidate-c",
        "candidate-b",
        "candidate-a",
    ]
    assert result.winner_candidate_id == "candidate-e"
    assert [item.rank for item in result.candidates] == [5, 4, 3, 2, 1]
    assert [item.is_winner for item in result.candidates] == [False, False, False, False, True]
