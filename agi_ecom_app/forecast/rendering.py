from __future__ import annotations

from typing import Any

from agi_ecom_app.forecast.models import ForecastDecisionResult


def render_forecast_decision_result(result: ForecastDecisionResult) -> str:
    ui_card = result.uiCard
    if ui_card is not None:
        lines = [
            "# Прогноз эффекта",
            "",
            "Что проверяем:",
            ui_card.whatChecking,
            "",
            "Что нашли:",
            ui_card.whatFound,
            "",
            "Если ничего не делать:",
            ui_card.ifNoAction,
            "",
            "Если сделать действие:",
            ui_card.ifAction,
            "",
            "Ожидаемый бизнес-эффект:",
            ui_card.expectedEffect,
            "",
            "Что рекомендую:",
            ui_card.recommendation,
        ]
        if ui_card.highlightMetrics:
            lines.extend(
                [
                    "",
                    "Ключевые сигналы:",
                    *_metric_list(
                        {item.label: item.value for item in ui_card.highlightMetrics if item.label and item.value}
                    ),
                ]
            )
        if result.resolutionActions:
            lines.extend(
                [
                    "",
                    "Конкретные действия:",
                    *[
                        f"- [{item.area}] {item.action}"
                        + (f" Причина: {item.reason}." if item.reason else "")
                        + (f" Метрика: {item.expectedMetric}." if item.expectedMetric else "")
                        for item in result.resolutionActions
                    ],
                ]
            )
        if result.resolutionPlan:
            lines.extend(
                [
                    "",
                    "План выполнения:",
                    *[
                        f"- [{step.operation}] {step.instruction}"
                        + (f" Метрика: {step.targetMetric}." if step.targetMetric else "")
                        + (f" Стоп-условие: {step.stopCondition}." if step.stopCondition else "")
                        for step in result.resolutionPlan
                    ],
                ]
            )
        if ui_card.confidenceReason:
            lines.extend(["", "Почему такая уверенность:", ui_card.confidenceReason])
        if ui_card.nextChecks:
            lines.extend(["", "Что ещё проверить:", *[f"- {item}" for item in ui_card.nextChecks]])
        if ui_card.blockers:
            lines.extend(["", "Ограничения:", *[f"- {item}" for item in ui_card.blockers]])
        return "\n".join(lines).strip()

    lines = [
        "# Прогноз эффекта",
        "",
        "Гипотеза:",
        result.hypothesis,
        "",
        "Если ничего не делать:",
        *_metric_list(result.baseline.metrics),
        "",
        "Если выполнить действие:",
        *_metric_list(result.actionScenario.metrics),
        "",
        "Ожидаемый эффект:",
        *_metric_list(result.delta.metrics),
    ]
    if result.blockedBy:
        lines.extend(["", "Ограничения:", *[f"- {item}" for item in result.blockedBy]])
    lines.extend(["", "Решение:", result.finalRecommendation or result.decision])
    return "\n".join(lines).strip()


def _metric_list(metrics: dict[str, Any]) -> list[str]:
    clean = [f"- {_normalize_label(key)}: {_normalize_value(value)}" for key, value in metrics.items() if _meaningful(value)]
    return clean or ["Нет рассчитанных сигналов."]


def _normalize_label(value: str) -> str:
    return value.replace("_", " ")


def _normalize_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "0.0", "none", "нет", "—", "unknown"}
    return True
