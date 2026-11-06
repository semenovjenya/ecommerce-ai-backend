from __future__ import annotations

"""Prometheus instruments shared by the simulation producer and worker."""

from prometheus_client import Counter, Gauge, Histogram


QUEUED_JOBS = Gauge(
    "decision_simulation_queue_depth",
    "Number of jobs waiting in the decision-simulations RQ queue.",
    multiprocess_mode="mostrecent",
)
ACTIVE_JOBS = Gauge(
    "decision_simulation_active_jobs",
    "Number of simulation jobs currently executing in this worker process.",
    multiprocess_mode="sum",
)
STALE_JOBS = Gauge(
    "decision_simulation_stale_jobs",
    "Number of stale simulation runs observed by the reconciler.",
    multiprocess_mode="mostrecent",
)
SAMPLES_COMPLETED = Counter(
    "decision_simulation_samples_completed_total",
    "Durably completed decision-simulation samples.",
    ("model_version",),
)
SAMPLES_PER_SECOND = Gauge(
    "decision_simulation_samples_per_second",
    "Most recently observed numerical throughput in samples per second.",
    ("model_version",),
    multiprocess_mode="mostrecent",
)
BATCH_DURATION = Histogram(
    "decision_simulation_batch_duration_seconds",
    "Wall-clock duration of a numerical batch.",
    ("model_version",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
JOB_DURATION = Histogram(
    "decision_simulation_job_duration_seconds",
    "Wall-clock duration of a complete simulation job.",
    ("mode", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300, 900),
)
FAILURES = Counter(
    "decision_simulation_failures_total",
    "Simulation failures by failure stage.",
    ("stage",),
)
REJECTED_CANDIDATES = Counter(
    "decision_simulation_rejected_candidates_total",
    "Candidates rejected by the deterministic hard gate.",
    ("reason",),
)
CRITIC_VERDICTS = Counter(
    "decision_simulation_critic_verdicts_total",
    "LLM critic verdicts.",
    ("mode", "verdict", "llm_available"),
)
FORECAST_COVERAGE = Gauge(
    "decision_simulation_forecast_coverage_ratio",
    "Share of submitted candidates that produced a numerical forecast.",
    multiprocess_mode="mostrecent",
)
ELIGIBILITY_OUTCOMES = Counter(
    "decision_simulation_eligibility_outcomes_total",
    "Manager-decision eligibility transitions, including successful queue delivery.",
    ("mode", "outcome"),
)


__all__ = [
    "ACTIVE_JOBS",
    "BATCH_DURATION",
    "CRITIC_VERDICTS",
    "ELIGIBILITY_OUTCOMES",
    "FAILURES",
    "FORECAST_COVERAGE",
    "JOB_DURATION",
    "QUEUED_JOBS",
    "REJECTED_CANDIDATES",
    "SAMPLES_COMPLETED",
    "SAMPLES_PER_SECOND",
    "STALE_JOBS",
]
