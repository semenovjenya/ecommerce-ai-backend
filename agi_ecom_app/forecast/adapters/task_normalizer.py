from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.models import ForecastHypothesis, ForecastPlan, ResolvedTarget
from agi_ecom_app.forecast.planner import (
    build_forecast_hypothesis,
    build_forecast_plan,
    resolve_forecast_targets,
)

__all__ = [
    "ForecastHypothesis",
    "ForecastPlan",
    "ResolvedTarget",
    "build_forecast_hypothesis",
    "build_forecast_plan",
    "resolve_forecast_targets",
]


def normalize_task_for_forecast(
    task_snapshot: dict[str, Any] | None,
) -> tuple[ForecastPlan | None, list[ResolvedTarget], ForecastHypothesis | None]:
    plan = build_forecast_plan(task_snapshot)
    if plan is None:
        return None, [], None
    resolved_targets = resolve_forecast_targets(task_snapshot, plan)
    hypothesis = build_forecast_hypothesis(task_snapshot)
    return plan, resolved_targets, hypothesis
