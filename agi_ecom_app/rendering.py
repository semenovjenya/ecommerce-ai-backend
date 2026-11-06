from __future__ import annotations

from agi_ecom_app.schemas import (
    ActionExecutionResult,
    AgentDiagnosticResult,
    MarketingDiagnosticResult,
    ReportTable,
    SalesDiagnosticResult,
)


def _is_delta_column(column: str) -> bool:
    normalized = column.lower()
    return (
        "изменен" in normalized
        or "изменён" in normalized
        or "дельт" in normalized
        or "delta" in normalized
        or "Δ" in column
    )


def _format_signed_value(value: str | int | float | None) -> str | int | float | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return f"+{value}" if value > 0 else value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped.startswith(("+", "-")):
        return value

    compact = stripped.replace(" ", "").replace(",", ".")
    try:
        numeric = float(compact)
    except ValueError:
        return value
    if numeric > 0:
        return f"+{stripped}"
    return value


def _normalize_cell(value: str | int | float | None, max_len: int = 120, signed: bool = False) -> str:
    if value is None or value == "":
        return "—"

    if signed:
        value = _format_signed_value(value)

    text = " ".join(str(value).split()).replace("|", "/")
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _render_table(table: ReportTable) -> list[str]:
    columns = [_normalize_cell(column, max_len=40) for column in table.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(
            _normalize_cell(
                cell,
                signed=index < len(columns) and _is_delta_column(columns[index]),
            )
            for index, cell in enumerate(row)
        )
        + " |"
        for row in (table.rows or [["Нет данных"] + [None] * (len(columns) - 1)])
    ]
    return [f"**{table.title}**", "", header, separator, *rows]


def render_report_table(table: ReportTable) -> str:
    return "\n".join(_render_table(table))


def render_action_execution_result(result: ActionExecutionResult) -> str:
    lines = [
        "# Результат выполнения",
        "",
        "Смысл для бизнеса:",
        result.businessMeaning,
        "",
        "Проверки:",
    ]
    if result.checks:
        lines.extend(
            [
                "| Проверка | Статус | Вывод |",
                "| --- | --- | --- |",
                *[
                    "| "
                    + " | ".join(
                        [
                            _normalize_cell(check.name, max_len=70),
                            _normalize_cell(check.status, max_len=20),
                            _normalize_cell(check.summary, max_len=180),
                        ]
                    )
                    + " |"
                    for check in result.checks
                ],
            ]
        )
    else:
        lines.append("Нет подтвержденных проверок.")

    if result.forecast:
        forecast = result.forecast
        lines.extend(
            [
                "",
                "Прогноз:",
                "| Метрика | Значение |",
                "| --- | --- |",
                f"| Скорость продаж | {_normalize_cell(forecast.salesVelocity)} |",
                f"| Активных дней | {_normalize_cell(forecast.activeDays)} |",
                f"| Доступный остаток | {_normalize_cell(forecast.availableStock)} |",
                f"| Товары в пути | {_normalize_cell(forecast.inboundStock)} |",
                f"| Покрытие, дней | {_normalize_cell(forecast.coverageDays)} |",
                f"| Риск out-of-stock | {_normalize_cell(forecast.stockoutRisk)} |",
                f"| Рекомендованный дозаказ | {_normalize_cell(forecast.reorderQty)} |",
            ]
        )
        if forecast.assumptions:
            lines.extend(["", "Допущения:", *[f"- {item}" for item in forecast.assumptions]])

    lines.extend(
        [
            "",
            "Решение:",
            result.decision,
            "",
            f"Статус решения: {result.decisionStatus}",
            f"Уверенность: {result.confidence}",
        ]
    )
    if result.blockedBy:
        lines.extend(["", "Блокеры:", *[f"- {item}" for item in result.blockedBy]])
    lines.extend(["", "Финальная рекомендация:", result.finalRecommendation])
    return "\n".join(lines).strip()


def render_agent_diagnostic(result: AgentDiagnosticResult) -> str:
    lines: list[str] = []

    if isinstance(result, MarketingDiagnosticResult) and result.marketing_report_snapshot:
        lines.extend(_render_table(result.marketing_report_snapshot))
        lines.append("")

    lines.extend(
        [
            f"# {result.agent_name}",
            "",
            result.summary,
            "",
        ]
    )

    for index, table in enumerate(result.tables):
        if index:
            lines.append("")
        lines.extend(_render_table(table))
        lines.append("")

    if result.findings:
        lines.append("Выводы:")
        for finding in result.findings:
            lines.append(
                "- "
                f"{_normalize_cell(finding.factor, max_len=80)}: "
                f"{_normalize_cell(finding.observation, max_len=180)} "
                f"Влияние: {_normalize_cell(finding.impact, max_len=140)}"
            )
        lines.append("")

    if result.recommendations:
        lines.append("Рекомендации:")
        for recommendation in result.recommendations:
            lines.append(f"- {_normalize_cell(recommendation, max_len=180)}")
        lines.append("")

    if result.data_gaps:
        lines.append("Ограничения данных:")
        for gap in result.data_gaps:
            lines.append(f"- {gap.metric}: {_normalize_cell(gap.impact, max_len=180)}")

    return "\n".join(line for line in lines if line is not None).strip()


def render_sales_diagnostic(result: SalesDiagnosticResult) -> str:
    lines: list[str] = []

    if result.sales_report_snapshot:
        lines.extend(_render_table(result.sales_report_snapshot))
        lines.append("")

    lines.extend(
        [
            f"# {result.agent_name}",
            "",
            result.summary,
            "",
        ]
    )

    for index, table in enumerate(result.as_tables()):
        if index:
            lines.append("")
        lines.extend(_render_table(table))
        lines.append("")

    if result.findings:
        lines.append("Выводы:")
        for finding in result.findings:
            lines.append(
                "- "
                f"{_normalize_cell(finding.factor, max_len=80)}: "
                f"{_normalize_cell(finding.observation, max_len=180)} "
                f"Влияние: {_normalize_cell(finding.impact, max_len=140)}"
            )
        lines.append("")

    if result.recommendations:
        lines.append("Рекомендации:")
        for recommendation in result.recommendations:
            lines.append(f"- {_normalize_cell(recommendation, max_len=180)}")
        lines.append("")

    if result.data_gaps:
        lines.append("Ограничения данных:")
        for gap in result.data_gaps:
            lines.append(f"- {gap.metric}: {_normalize_cell(gap.impact, max_len=180)}")

    return "\n".join(line for line in lines if line is not None).strip()
