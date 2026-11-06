from __future__ import annotations

import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from agi_ecom_app.config import get_settings
from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.llm_runtime import get_active_llm_runtime_spec
from agi_ecom_app.schemas import (
    ActionExecutionResult,
    AgentDiagnosticResult,
    ChatRole,
    DiagnosticFinding,
    QueryIntent,
    SalesDiagnosticResult,
)
from agi_ecom_app.services.memory import (
    AgentActionExecutionRecord,
    StoredAgentActionTask,
    StoredChatMessage,
    create_agent_action_execution,
    create_chat_session,
    get_agent_action_execution,
    get_agent_action_execution_for_object,
    get_agent_run_trace_by_id,
    list_agent_action_executions,
    list_all_stored_messages,
    list_selected_action_task_ids,
    list_stored_agent_action_tasks,
    replace_agent_action_tasks,
    select_action_task,
    unselect_action_task,
    update_agent_action_execution,
)
from agi_ecom_app.services import strategic_db
from agi_ecom_app.action_board.models import (
    AgentActionExecutionBatchResponse,
    AgentActionExecutionItem,
    AgentActionExecutionRequest,
    AgentActionExecutionForecastView,
    AgentActionExecutionResultView,
    AgentActionExecutionViewItem,
    AgentActionExecutionViewPlanStep,
    AgentActionExecutionsResponse,
    AgentActionTaskObject,
    AgentActionRecommendationItem,
    AgentActionTaskItem,
    AgentActionTaskSnapshot,
    AgentActionTasksResponse,
)

ACTION_TASKS_METADATA_KEY = "feature3_actions"
ACTION_TASKS_SCHEMA_VERSION_KEY = "feature3_schema"
ACTION_TASKS_SCHEMA_VERSION = "action_tasks.v4"
EXECUTION_MODE_METADATA_KEY = "execution_mode"
SUPPRESS_ACTION_TASKS_METADATA_KEY = "suppress_action_tasks"
TERMINAL_ACTION_EXECUTION_MODE = "terminal_action"
_FORECAST_RESULT_KEYS = {"hypothesis", "baseline", "actionScenario", "delta"}

_MARKDOWN_FINDING_PATTERN = re.compile(
    r"^- (?P<factor>[^:\n]+): (?P<observation>.*?)\s+"
    r"Влияние: (?P<impact>.*?)\s+"
    r"Действие: (?P<action>.*)$"
)
_FINAL_ACTION_SECTION_PATTERN = re.compile(r"^\s*(?:#+\s*)?что делать\s*:?\s*$", re.IGNORECASE)
_NUMBERED_ACTION_PATTERN = re.compile(r"^\s*\d+[.)]\s+(?P<action>.+?)\s*$")
_MARKDOWN_SECTION_HEADING_PATTERN = re.compile(r"^\s*(?:#{1,6}\s+|[A-Za-zА-Яа-я0-9 _-]+:\s*$)")
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_MORE_OBJECTS_PATTERN = re.compile(r"\+\s*ещ[её]\s+\d+\b", re.IGNORECASE)
_AGGREGATE_TARGET_PATTERN = re.compile(
    r"\b(?:все|вся|весь|групп\w*|категори\w*|ассортимент\w*|подборк\w*)\b",
    re.IGNORECASE,
)
_NOISY_TARGET_VALUES = {
    "кампания",
    "кампании",
    "кампаний",
    "кампанию",
    "запрос",
    "запросы",
    "запросов",
    "видимость",
    "ставки",
    "ценовое",
    "цену",
    "цена",
    "лидеров",
    "лидеры",
    "план",
    "продаж",
    "продажи",
    "группы",
    "группа",
    "sku",
    "все продвигаемые sku",
}

_RECOMMENDATION_PATTERN = re.compile(
    r"^(?:если хотите|для точного отч[её]та|для диагностики|для воронки|"
    r"предлагаю|лучше|можно|стоит|рекомендуется)\b",
    re.IGNORECASE,
)
_EMOTIONAL_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:срочно|важно|критично|в\s+приоритете|приоритетно|"
    r"обратить\s+внимание(?:\s+на)?)[\s:—-]*)+",
    re.IGNORECASE,
)
_CLASSIFIER_PREFIX_PATTERN = re.compile(
    r"^\s*(?:удержать|исправить|усилить|сократить|остановить|масштабировать|"
    r"оставить(?:\s+с\s+контролем)?|проверить\s+поставку|"
    r"проверить\s+снижение\s+цены|проверить\s+повышение\s+цены|"
    r"удержать\s+цену|проверить)\s*:\s*",
    re.IGNORECASE,
)
_LEADING_TASK_VERB_PATTERN = re.compile(
    r"^(?:проверить|сверить|разобрать|передать|вынести|сопоставить|"
    r"исправить|дозаказать|пополнить|создать|запустить|собрать|"
    r"обновить|снизить|повысить|остановить|рассчитать|мониторить|"
    r"контролировать|продолжать|удержать|усилить|сократить|"
    r"масштабировать|оставить|зафиксировать|сформировать|перенести|"
    r"разделить|держать)\b",
    re.IGNORECASE,
)
_TARGET_NUMBER_PATTERN = re.compile(r"\b\d{3,}\b")
_DATE_TARGET_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)\b"
)
_BUSINESS_OBJECT_PATTERN = re.compile(
    r"\b(?:карточк\w*|артикул\w*|nm_id|поставк\w*|кампани\w*|"
    r"запрос\w*|остатк\w*|наличи\w*|цен\w*|видимост\w*|"
    r"реклам\w*|марж\w*|прибыл\w*|экономик\w*|метрик\w*|"
    r"ассортимент\w*|воронк\w*|отмен\w*|заказ\w*|продаж\w*|"
    r"лидер\w*|дозаказ\w*|план\w*)\b",
    re.IGNORECASE,
)
_TARGET_PHRASE_PATTERN = re.compile(
    r"\b(?:по|для|на|у|в)\s+("
    r"(?:проблемн\w+\s+|готов\w+\s+|входящ\w+\s+)?"
    r"(?:карточк\w*|артикул\w*|поставк\w*|кампани\w*|запрос\w*|"
    r"остатк\w*|наличи\w*|цен\w*|видимост\w*|ассортимент\w*|"
    r"воронк\w*|отмен\w*|заказ\w*|продаж\w*|лидер\w*)"
    r"(?:\s+[0-9A-Za-zА-Яа-я_-]+){0,3})",
    re.IGNORECASE,
)
_DIRECT_TARGET_PHRASE_PATTERN = re.compile(
    r"\b((?:проблемн\w+\s+|готов\w+\s+|входящ\w+\s+)?"
    r"(?:карточк\w*|артикул\w*|поставк\w*|кампани\w*|запрос\w*|"
    r"остатк\w*|наличи\w*|цен\w*|видимост\w*|ассортимент\w*|"
    r"воронк\w*|отмен\w*|заказ\w*|продаж\w*|лидер\w*))\b",
    re.IGNORECASE,
)
_CRITICAL_PRIORITY_PATTERN = re.compile(
    r"\b(?:срочно|критично|critical|out[- ]?of[- ]?stock|сорван\w*\s+план\w*|"
    r"дефицит|нулев\w+\s+остат\w*)\b",
    re.IGNORECASE,
)
_HIGH_PRIORITY_PATTERN = re.compile(
    r"\b(?:warning|риск|проверить|проверь|в\s+приоритете|приоритетно|важно|"
    r"недобор|просадк\w*|"
    r"падени\w*|слаб\w*)\b",
    re.IGNORECASE,
)
_ACTION_TYPE_PATTERNS = (
    ("calculate", re.compile(r"^(?:рассчитать)\b", re.IGNORECASE)),
    (
        "create",
        re.compile(
            r"^(?:дозаказать|пополнить|создать|запустить|собрать|сформировать)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "update",
        re.compile(
            r"^(?:обновить|исправить|снизить|повысить|остановить|перенести|"
            r"разделить|усилить|сократить|масштабировать)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "monitor",
        re.compile(
            r"^(?:мониторить|контролировать|продолжать|удержать|оставить|держать)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "check",
        re.compile(
            r"^(?:проверить|сверить|разобрать|передать|вынести|сопоставить|"
            r"зафиксировать)\b",
            re.IGNORECASE,
        ),
    ),
)

_DOMAIN_CATEGORY_LABELS = {
    "marketing": "Маркетинг",
    "finance": "Финансы",
    "price": "Цены",
    "sales": "Продажи",
    "supply": "Поставка/остатки",
    "catalog_demand": "Карточка",
}


def extract_action_tasks_from_agent_output(
    final_output: Any,
    rendered_text: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(final_output, SalesDiagnosticResult):
        return [_finding_to_action_task(finding) for finding in final_output.findings]
    if isinstance(final_output, AgentDiagnosticResult):
        return [_finding_to_action_task(finding) for finding in final_output.findings]
    if isinstance(final_output, BaseModel):
        findings = getattr(final_output, "findings", None)
        if isinstance(findings, list):
            return [
                _finding_to_action_task(finding)
                for finding in findings
                if isinstance(finding, DiagnosticFinding)
            ]
    if rendered_text:
        return _extract_action_tasks_from_markdown(rendered_text)
    if isinstance(final_output, str):
        return _extract_action_tasks_from_markdown(final_output)
    return []


def get_agent_action_tasks_payload(
    chat_id: str,
    server_message_id: int | None = None,
) -> AgentActionTasksResponse:
    selected_task_ids = list_selected_action_task_ids(chat_id)
    try:
        stored_tasks = list_stored_agent_action_tasks(
            chat_id=chat_id,
            assistant_message_seq=server_message_id,
        )
    except Exception:
        stored_tasks = []
    if stored_tasks:
        return AgentActionTasksResponse(
            chatId=chat_id,
            sourceMessageId=server_message_id,
            items=[
                _stored_task_to_action_item(task, selected=task.task_id in selected_task_ids)
                for task in stored_tasks
            ],
            recommendations=[],
        )

    messages = [
        message
        for message in list_all_stored_messages(chat_id)
        if message.role == ChatRole.ASSISTANT
    ]
    if server_message_id is not None:
        messages = [message for message in messages if message.seq == server_message_id]

    items: list[AgentActionTaskItem] = []
    recommendations: list[AgentActionRecommendationItem] = []
    for message in messages:
        message_items, message_recommendations = _message_to_action_items(
            chat_id=chat_id,
            message=message,
            selected_task_ids=selected_task_ids,
        )
        items.extend(message_items)
        recommendations.extend(message_recommendations)

    return AgentActionTasksResponse(
        chatId=chat_id,
        sourceMessageId=server_message_id,
        items=items,
        recommendations=recommendations,
    )


def persist_agent_action_tasks_for_message(
    *,
    chat_id: str,
    server_message_id: int,
    raw_actions: list[dict[str, Any]],
    intent: str | None,
) -> list[AgentActionTaskItem]:
    items, _ = _raw_actions_to_action_items(
        chat_id=chat_id,
        message_seq=server_message_id,
        raw_actions=raw_actions,
        intent=intent,
        created_at=None,
        selected_task_ids=set(),
    )
    replace_agent_action_tasks(
        chat_id=chat_id,
        assistant_message_seq=server_message_id,
        tasks=[_action_item_to_storage_row(item, raw_actions) for item in items],
    )
    return items


def select_agent_action_task(chat_id: str, task_id: str) -> AgentActionTaskItem:
    item = _find_action_task(chat_id=chat_id, task_id=task_id)
    if item is None:
        raise ValueError("task_id does not match an agent action for this chat")

    select_action_task(chat_id=chat_id, task_id=task_id)
    return item.model_copy(update={"selected": True})


def unselect_agent_action_task(chat_id: str, task_id: str) -> AgentActionTaskItem:
    item = _find_action_task(chat_id=chat_id, task_id=task_id)
    if item is None:
        raise ValueError("task_id does not match an agent action for this chat")

    unselect_action_task(chat_id=chat_id, task_id=task_id)
    return item.model_copy(update={"selected": False})


def start_agent_action_execution(
    payload: AgentActionExecutionRequest,
) -> AgentActionExecutionBatchResponse:
    task = _resolve_execution_task(payload)
    settings = get_settings()
    items: list[AgentActionExecutionItem] = []
    execution_objects = _task_objects_for_execution(task)
    readiness = _optional_string(task.executionReadiness)
    if readiness is None:
        readiness, _ = _task_execution_readiness(
            targets=list(task.targets),
            raw_object_text=task.rawObjectText,
            objects=execution_objects,
        )
    if readiness in {"needs_resolution", "blocked"}:
        return AgentActionExecutionBatchResponse(
            chatId=payload.chat_id,
            sourceTaskId=payload.task_id,
            items=[],
        )

    for index, task_object in enumerate(execution_objects, start=1):
        child_task = _snapshot_for_task_object(
            source_task=task,
            source_task_id=payload.task_id,
            task_object=task_object,
        )
        existing = get_agent_action_execution_for_object(
            source_chat_id=payload.chat_id,
            source_task_id=payload.task_id,
            object_id=task_object.objectId,
        )
        if existing is not None:
            items.append(_execution_record_to_item(_sync_agent_action_execution(existing)))
            continue

        prompt = _build_execution_prompt(child_task)
        try:
            execution_model_name = get_active_llm_runtime_spec(require_configured=False).display_name
        except Exception:
            execution_model_name = f"openai:{settings.openai_model}"
        execution_chat_id = create_chat_session(model_name=execution_model_name)
        client_message_id = f"action-execution-{uuid.uuid4()}"
        record = create_agent_action_execution(
            source_chat_id=payload.chat_id,
            source_task_id=payload.task_id,
            task_id=child_task.taskId,
            object_id=task_object.objectId,
            object_label=task_object.label,
            object_type=task_object.type,
            object_external_id=task_object.externalId,
            object_payload=task_object.payload,
            execution_chat_id=execution_chat_id,
            client_message_id=client_message_id,
            prompt=prompt,
            task_snapshot=child_task.model_dump(mode="json"),
            execution_mode=TERMINAL_ACTION_EXECUTION_MODE,
        )

        try:
            from agi_ecom_app.agent_queue import enqueue_action_execution_run

            queue_result = enqueue_action_execution_run(
                chat_id=execution_chat_id,
                source_chat_id=payload.chat_id,
                source_execution_id=record.id,
                prompt=prompt,
                client_message_id=client_message_id,
                intent_override=_execution_intent(child_task),
            )
            record = update_agent_action_execution(
                execution_id=record.id,
                status=queue_result.status,
                run_id=queue_result.run_id,
                answer=queue_result.answer,
                server_message_id=queue_result.server_message_id,
                review_summary=queue_result.answer,
            )
        except Exception as exc:
            update_agent_action_execution(
                execution_id=record.id,
                status="failed",
                error=str(exc),
            )
            raise

        items.append(_execution_record_to_item(_sync_agent_action_execution(record)))

    return AgentActionExecutionBatchResponse(
        chatId=payload.chat_id,
        sourceTaskId=payload.task_id,
        items=items,
    )


def list_agent_action_executions_payload(chat_id: str) -> AgentActionExecutionsResponse:
    return AgentActionExecutionsResponse(
        chatId=chat_id,
        items=[
            _execution_record_to_item(_sync_agent_action_execution(record))
            for record in list_agent_action_executions(chat_id)
        ],
    )


def get_agent_action_execution_payload(execution_id: int) -> AgentActionExecutionItem:
    record = get_agent_action_execution(execution_id)
    if record is None:
        raise LookupError(execution_id)
    return _execution_record_to_item(_sync_agent_action_execution(record))


def get_agent_action_execution_view_payload(execution_id: int) -> AgentActionExecutionViewItem:
    return _execution_item_to_view(get_agent_action_execution_payload(execution_id))


def complete_agent_action_execution(execution_id: int) -> AgentActionExecutionItem:
    record = get_agent_action_execution(execution_id)
    if record is None:
        raise LookupError(execution_id)
    synced = _sync_agent_action_execution(record)
    completed = update_agent_action_execution(
        execution_id=synced.id,
        status="done",
    )
    return _execution_record_to_item(completed)


def _finding_to_action_task(finding: DiagnosticFinding) -> dict[str, Any]:
    return {
        "source": "agent_finding",
        "factor": finding.factor,
        "observation": finding.observation,
        "impact": finding.impact,
        "cause": finding.cause,
        "action": finding.action,
        "evidence": list(finding.evidence),
        "confidence": finding.confidence,
        "driverClass": finding.driver_class.value,
    }


def _message_to_action_items(
    *,
    chat_id: str,
    message: StoredChatMessage,
    selected_task_ids: set[str],
) -> tuple[list[AgentActionTaskItem], list[AgentActionRecommendationItem]]:
    raw_actions = message.metadata.get(ACTION_TASKS_METADATA_KEY)
    if not isinstance(raw_actions, list):
        raw_actions = _extract_action_tasks_from_markdown(message.content)

    intent = _metadata_string(message.metadata, "intent")
    return _raw_actions_to_action_items(
        chat_id=chat_id,
        message_seq=message.seq,
        raw_actions=raw_actions,
        intent=intent,
        created_at=message.created_at,
        selected_task_ids=selected_task_ids,
    )


def _raw_actions_to_action_items(
    *,
    chat_id: str,
    message_seq: int,
    raw_actions: list[Any],
    intent: str | None,
    created_at,
    selected_task_ids: set[str],
) -> tuple[list[AgentActionTaskItem], list[AgentActionRecommendationItem]]:
    items: list[AgentActionTaskItem] = []
    recommendations: list[AgentActionRecommendationItem] = []
    for index, raw_action in enumerate(raw_actions, start=1):
        if not isinstance(raw_action, dict):
            continue

        action_text = _optional_string(raw_action.get("action"))
        factor = _optional_string(raw_action.get("factor"))
        if not action_text or not factor:
            continue

        normalized_task = _normalize_action_task(raw_action, action_text, intent)
        if normalized_task is None:
            recommendations.append(
                AgentActionRecommendationItem(
                    recommendationId=_recommendation_id(
                        chat_id=chat_id,
                        message_seq=message_seq,
                        index=index,
                    ),
                    chatId=chat_id,
                    serverMessageId=message_seq,
                    source=_optional_string(raw_action.get("source")) or "markdown_fallback",
                    factor=factor,
                    observation=_optional_string(raw_action.get("observation")),
                    impact=_optional_string(raw_action.get("impact")),
                    recommendation=action_text,
                    intent=intent,
                    createdAt=created_at,
                )
            )
            continue

        action_id = _action_id(chat_id=chat_id, message_seq=message_seq, index=index)
        items.append(
            AgentActionTaskItem(
                taskId=action_id,
                actionId=action_id,
                chatId=chat_id,
                serverMessageId=message_seq,
                source=_optional_string(raw_action.get("source")) or "agent_finding",
                factor=factor,
                observation=_optional_string(raw_action.get("observation")),
                impact=_optional_string(raw_action.get("impact")),
                cause=_optional_string(raw_action.get("cause")),
                action=normalized_task["title"],
                title=normalized_task["title"],
                rawObjectText=normalized_task["rawObjectText"],
                executionReadiness=normalized_task["executionReadiness"],
                executionReason=normalized_task["executionReason"],
                actionType=normalized_task["actionType"],
                priority=normalized_task["priority"],
                targets=normalized_task["targets"],
                successCriteria=normalized_task["successCriteria"],
                agentPrompt=normalized_task["agentPrompt"],
                actionKind=normalized_task["actionKind"],
                businessGoal=normalized_task["businessGoal"],
                decisionType=normalized_task["decisionType"],
                requiredMetrics=normalized_task["requiredMetrics"],
                forecastSpec=normalized_task["forecastSpec"],
                executionPlan=normalized_task["executionPlan"],
                objects=_task_objects_from_targets(normalized_task["targets"]),
                evidence=_string_list(raw_action.get("evidence")),
                confidence=_optional_string(raw_action.get("confidence")),
                driverClass=_optional_string(raw_action.get("driverClass")),
                intent=intent,
                createdAt=created_at,
                selected=action_id in selected_task_ids,
            )
        )
    return items, recommendations


def _find_action_task(chat_id: str, task_id: str) -> AgentActionTaskItem | None:
    response = get_agent_action_tasks_payload(chat_id)
    return next((item for item in response.items if item.taskId == task_id), None)


def _stored_task_to_action_item(
    task: StoredAgentActionTask,
    *,
    selected: bool,
) -> AgentActionTaskItem:
    title, raw_object_text = _stored_task_title_and_object_text(task)
    targets = _normalized_task_targets(
        _string_list(task.raw_action.get("targets")) or list(task.targets),
        raw_object_text=raw_object_text,
    )
    objects = _task_objects_from_targets(targets)
    readiness, readiness_reason = _task_execution_readiness(
        targets=targets,
        raw_object_text=raw_object_text,
        objects=objects,
    )
    return AgentActionTaskItem(
        taskId=task.task_id,
        actionId=task.action_id,
        chatId=task.chat_id,
        serverMessageId=task.assistant_message_seq,
        source=task.source,
        factor=task.factor,
        observation=task.observation,
        impact=task.impact,
        cause=task.cause,
        action=title or task.action,
        title=title,
        rawObjectText=raw_object_text,
        executionReadiness=readiness,
        executionReason=readiness_reason,
        actionType=task.action_type,
        priority=task.priority,
        targets=targets,
        successCriteria=task.success_criteria,
        agentPrompt=task.agent_prompt,
        actionKind=task.action_kind,
        businessGoal=task.business_goal,
        decisionType=task.decision_type,
        requiredMetrics=task.required_metrics,
        forecastSpec=task.forecast_spec,
        executionPlan=task.execution_plan,
        objects=objects,
        evidence=task.evidence,
        confidence=task.confidence,
        driverClass=task.driver_class,
        intent=task.intent,
        createdAt=task.created_at,
        selected=selected,
    )


def _stored_task_title_and_object_text(task: StoredAgentActionTask) -> tuple[str | None, str | None]:
    raw_object_text = _optional_string(task.raw_action.get("rawObjectText"))
    title = _optional_string(task.raw_action.get("title")) or _optional_string(task.title)
    if task.source != "final_table_action":
        return title or _optional_string(task.action), raw_object_text

    action_text = _optional_string(task.action) or _optional_string(task.raw_action.get("action"))
    if raw_object_text is None and action_text and ":" in action_text:
        prefix, suffix = action_text.split(":", 1)
        prefix = _compact_text(prefix)
        suffix = _compact_text(suffix)
        if prefix and suffix:
            return prefix, suffix
    return title or action_text, raw_object_text


def _action_item_to_storage_row(
    item: AgentActionTaskItem,
    raw_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_action = _raw_action_for_task(item, raw_actions)
    return {
        "task_id": item.taskId,
        "action_id": item.actionId,
        "source": item.source,
        "factor": item.factor,
        "observation": item.observation,
        "impact": item.impact,
        "cause": item.cause,
        "action": item.action,
        "title": item.title,
        "action_type": item.actionType,
        "priority": item.priority,
        "targets": item.targets,
        "success_criteria": item.successCriteria,
        "agent_prompt": item.agentPrompt,
        "action_kind": item.actionKind,
        "business_goal": item.businessGoal,
        "decision_type": item.decisionType,
        "required_metrics": item.requiredMetrics,
        "forecast_spec": item.forecastSpec,
        "execution_plan": item.executionPlan,
        "evidence": item.evidence,
        "confidence": item.confidence,
        "driver_class": item.driverClass,
        "intent": item.intent,
        "raw_action": raw_action,
    }


def _resolve_execution_task(payload: AgentActionExecutionRequest) -> AgentActionTaskSnapshot:
    item = _find_action_task(chat_id=payload.chat_id, task_id=payload.task_id)
    if payload.task is None and item is None:
        raise ValueError("task_id does not match an agent action for this chat")

    if payload.task is not None and payload.task.chatId and payload.task.chatId != payload.chat_id:
        raise ValueError("task.chatId does not match chat_id")

    base = _snapshot_from_action_item(item) if item is not None else None
    if payload.task is None:
        if base is None:
            raise ValueError("task_id does not match an agent action for this chat")
        return base

    payload_dump = payload.task.model_dump(
        mode="json",
        exclude_none=True,
        exclude_unset=True,
    )
    merged = base.model_dump(mode="json") if base is not None else {}
    merged.update(payload_dump)
    merged.update(
        {
            "taskId": payload.task_id,
            "chatId": payload.chat_id,
        }
    )
    if "selected" not in merged:
        merged["selected"] = True
    return AgentActionTaskSnapshot.model_validate(merged)


def _snapshot_from_action_item(item: AgentActionTaskItem) -> AgentActionTaskSnapshot:
    return AgentActionTaskSnapshot(
        taskId=item.taskId,
        actionId=item.actionId,
        chatId=item.chatId,
        serverMessageId=item.serverMessageId,
        source=item.source,
        factor=item.factor,
        observation=item.observation,
        impact=item.impact,
        cause=item.cause,
        action=item.action,
        title=item.title,
        rawObjectText=item.rawObjectText,
        executionReadiness=item.executionReadiness,
        executionReason=item.executionReason,
        actionType=item.actionType,
        priority=item.priority,
        targets=item.targets,
        successCriteria=item.successCriteria,
        agentPrompt=item.agentPrompt,
        actionKind=item.actionKind,
        businessGoal=item.businessGoal,
        decisionType=item.decisionType,
        requiredMetrics=item.requiredMetrics,
        forecastSpec=item.forecastSpec,
        executionPlan=item.executionPlan,
        objects=item.objects,
        evidence=item.evidence,
        confidence=item.confidence,
        driverClass=item.driverClass,
        intent=item.intent,
        createdAt=item.createdAt,
        selected=item.selected,
    )


def _task_objects_for_execution(task: AgentActionTaskSnapshot) -> list[AgentActionTaskObject]:
    objects = [item for item in task.objects if _optional_string(item.objectId)]
    if objects:
        deduped: list[AgentActionTaskObject] = []
        seen: set[str] = set()
        for item in objects:
            object_key = _executable_target_key(item.externalId or item.objectId or item.label)
            if object_key is None or object_key in seen:
                continue
            seen.add(object_key)
            deduped.append(item)
        return _enrich_task_objects_for_execution(deduped, task)
    if _optional_string(task.rawObjectText):
        return []
    return _enrich_task_objects_for_execution([
        AgentActionTaskObject(
            objectId=task.taskId,
            label=_optional_string(task.title) or _optional_string(task.action),
            type=None,
            externalId=_first_nonempty(*task.targets),
            payload={},
        )
    ], task)


def _task_objects_from_targets(targets: list[str]) -> list[AgentActionTaskObject]:
    objects: list[AgentActionTaskObject] = []
    seen: set[str] = set()
    for target in targets:
        candidate = _optional_string(target)
        object_key = _executable_target_key(candidate)
        if object_key is None or object_key in seen:
            continue
        seen.add(object_key)
        objects.append(
            AgentActionTaskObject(
                objectId=_task_object_id(object_key),
                label=_task_object_label(object_key),
                type=_task_object_type(object_key),
                externalId=object_key if _looks_numeric_identifier(object_key) else None,
                payload=_task_object_payload(object_key),
            )
        )
    return objects


def _executable_target_key(value: str | None) -> str | None:
    candidate = _optional_string(value)
    if not candidate:
        return None
    numeric_candidate = _extract_numeric_target(candidate)
    if numeric_candidate:
        return numeric_candidate
    if _is_target_noise(candidate):
        return None
    return candidate


def _extract_numeric_target(value: str) -> str | None:
    stripped = value.strip()
    if stripped.isdigit():
        return stripped
    match = _TARGET_NUMBER_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return None


def _is_target_noise(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in _NOISY_TARGET_VALUES:
        return True
    if _AGGREGATE_TARGET_PATTERN.search(normalized):
        return True
    return False


def _normalized_task_targets(
    targets: list[str],
    *,
    raw_object_text: str | None,
) -> list[str]:
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for target in targets:
        object_key = _executable_target_key(target)
        if object_key is None or object_key in seen:
            continue
        seen.add(object_key)
        normalized_targets.append(object_key)
    if normalized_targets or raw_object_text:
        return normalized_targets
    return targets


def _task_execution_readiness(
    *,
    targets: list[str],
    raw_object_text: str | None,
    objects: list[AgentActionTaskObject],
) -> tuple[str, str]:
    executable_count = len(objects)
    if executable_count > 1:
        return (
            "fanout_ready",
            f"Extracted {executable_count} executable targets for per-object execution.",
        )
    if executable_count == 1:
        return ("single_target_ready", "Extracted 1 executable target.")
    if _optional_string(raw_object_text):
        return (
            "needs_resolution",
            "Raw object text is present, but no executable targets were extracted.",
        )
    if targets:
        return ("blocked", "Targets are present, but none can be executed automatically.")
    return ("blocked", "Task has no executable targets.")


def _snapshot_for_task_object(
    *,
    source_task: AgentActionTaskSnapshot,
    source_task_id: str,
    task_object: AgentActionTaskObject,
) -> AgentActionTaskSnapshot:
    object_target = _task_object_target(source_task, task_object)
    data = source_task.model_dump(mode="json")
    data.update(
        {
            "taskId": _child_task_id(source_task_id, task_object.objectId),
            "targets": [object_target] if object_target else [],
            "objects": [task_object.model_dump(mode="json")],
            "rawObjectText": object_target,
            "executionReadiness": "single_target_ready",
            "executionReason": "Derived child execution target from fanout task.",
        }
    )
    return AgentActionTaskSnapshot.model_validate(data)


def _child_task_id(source_task_id: str, object_id: str) -> str:
    if object_id == source_task_id:
        return source_task_id
    return f"{source_task_id}:{object_id}"


def _task_object_target(
    source_task: AgentActionTaskSnapshot,
    task_object: AgentActionTaskObject,
) -> str | None:
    candidates = [
        task_object.externalId,
        task_object.objectId,
        *source_task.targets,
        task_object.label,
    ]
    numeric_candidate = next(
        (candidate for candidate in candidates if _looks_numeric_identifier(candidate)),
        None,
    )
    return _optional_string(numeric_candidate) or _first_nonempty(*candidates)


def _looks_numeric_identifier(value: str | None) -> bool:
    candidate = _optional_string(value)
    return bool(candidate and candidate.isdigit())


def _task_object_id(target: str) -> str:
    if _looks_numeric_identifier(target):
        return f"sku-{target}"
    return target


def _task_object_label(target: str) -> str:
    if _looks_numeric_identifier(target):
        return f"SKU {target}"
    return target


def _task_object_type(target: str) -> str | None:
    if _looks_numeric_identifier(target):
        return "sku"
    return None


def _task_object_payload(target: str) -> dict[str, Any]:
    if _looks_numeric_identifier(target):
        return {"nm_id": int(target)}
    return {}


def _enrich_task_objects_for_execution(
    objects: list[AgentActionTaskObject],
    task: AgentActionTaskSnapshot,
) -> list[AgentActionTaskObject]:
    numeric_ids: list[str] = []
    for item in objects:
        nm_id = _task_object_nm_id(item)
        if nm_id and nm_id not in numeric_ids:
            numeric_ids.append(nm_id)

    identity_by_nm_id: dict[str, dict[str, Any]] = {}
    if numeric_ids:
        try:
            identity_by_nm_id = strategic_db.get_sku_identity_map(numeric_ids)
        except Exception:
            identity_by_nm_id = {}

    domain_category = _task_domain_category(task)
    enriched: list[AgentActionTaskObject] = []
    for item in objects:
        payload = dict(item.payload or {})
        nm_id = _task_object_nm_id(item)
        identity = identity_by_nm_id.get(nm_id or "")
        if nm_id:
            payload["nm_id"] = int(nm_id)
        if identity:
            if identity.get("vendor_code"):
                payload["vendor_code"] = identity["vendor_code"]
            if identity.get("subject_name"):
                payload["subject_name"] = identity["subject_name"]
        if "subject_name" not in payload and _optional_string(item.label):
            payload["subject_name"] = _optional_string(item.label)
        if domain_category:
            payload["domain_category"] = domain_category
        enriched.append(item.model_copy(update={"payload": payload}))
    return enriched


def _task_object_nm_id(task_object: AgentActionTaskObject) -> str | None:
    payload_nm_id = task_object.payload.get("nm_id") if isinstance(task_object.payload, dict) else None
    candidate = _optional_string(str(payload_nm_id)) if payload_nm_id is not None else None
    if candidate and candidate.isdigit():
        return candidate
    for raw_candidate in (task_object.externalId, task_object.objectId, task_object.label):
        value = _optional_string(raw_candidate)
        if value and value.isdigit():
            return value
    return None


def _task_domain_category(task: AgentActionTaskSnapshot) -> str | None:
    forecast_domain = task.forecastSpec.get("forecast_domain") if isinstance(task.forecastSpec, dict) else None
    domain = _optional_string(forecast_domain) or _optional_string(task.intent)
    if domain and domain in _DOMAIN_CATEGORY_LABELS:
        return _DOMAIN_CATEGORY_LABELS[domain]
    return None


def _first_nonempty(*values: str | None) -> str | None:
    for value in values:
        candidate = _optional_string(value)
        if candidate:
            return candidate
    return None


def _build_execution_prompt(task: AgentActionTaskSnapshot) -> str:
    if task.agentPrompt and task.agentPrompt.strip():
        return task.agentPrompt.strip()

    title = _optional_string(task.title) or _optional_string(task.action) or "Выполнить задачу"
    parts = [f"Выполни задачу из Action Board: {title}."]
    if task.actionType:
        parts.append(f"Тип задачи: {task.actionType}.")
    if task.priority:
        parts.append(f"Приоритет: {task.priority}.")
    if task.intent:
        parts.append(f"Домен: {task.intent}.")
    if task.targets:
        parts.append(f"Объекты: {_format_targets(task.targets)}.")
    for label, value in (
        ("Фактор", task.factor),
        ("Наблюдение", task.observation),
        ("Влияние", task.impact),
        ("Причина", task.cause),
    ):
        normalized_value = _optional_string(value)
        if normalized_value:
            parts.append(f"{label}: {normalized_value}.")
    if task.evidence:
        parts.append(f"Факты: {'; '.join(task.evidence[:4])}.")
    if task.successCriteria:
        parts.append(f"Критерий готовности: {task.successCriteria}")
    parts.append("Верни краткий результат таблицей: объект, статус, причина, следующее действие.")
    return " ".join(parts)


def _execution_intent(task: AgentActionTaskSnapshot) -> QueryIntent | None:
    if not task.intent:
        return None
    try:
        return QueryIntent(task.intent)
    except ValueError:
        return None


def _sync_agent_action_execution(
    record: AgentActionExecutionRecord,
) -> AgentActionExecutionRecord:
    if record.status == "done" or record.run_id is None:
        return record

    trace = get_agent_run_trace_by_id(record.run_id)
    if trace is None:
        return record

    answer = record.answer
    server_message_id = record.server_message_id
    error = record.error
    if trace.assistant_message_seq is not None:
        server_message_id = trace.assistant_message_seq
        answer = _assistant_answer_for_trace(trace.chat_id, trace.assistant_message_seq) or answer
    if trace.status == "failed":
        error = str((trace.summary or {}).get("error") or error or "agent run failed")
    if trace.status == "cancelled":
        error = error or "agent run cancelled"

    if (
        trace.status == record.status
        and answer == record.answer
        and server_message_id == record.server_message_id
        and error == record.error
    ):
        return record

    return update_agent_action_execution(
        execution_id=record.id,
        status=trace.status,
        answer=answer,
        server_message_id=server_message_id,
        error=error,
        result_json=_execution_result_from_trace(trace),
        decision_status=_decision_status_from_trace(trace),
        review_summary=_review_summary_from_trace(trace, answer),
    )


def _assistant_answer_for_trace(chat_id: str, assistant_message_seq: int) -> str | None:
    for message in list_all_stored_messages(chat_id):
        if message.role == ChatRole.ASSISTANT and message.seq == assistant_message_seq:
            return message.content
    return None


def _execution_record_to_item(record: AgentActionExecutionRecord) -> AgentActionExecutionItem:
    task_data = dict(record.task_snapshot or {})
    task_data.setdefault("taskId", record.task_id)
    task_data.setdefault("chatId", record.source_chat_id)
    if not task_data.get("objects") and record.object_id:
        task_data["objects"] = [
            AgentActionTaskObject(
                objectId=record.object_id,
                label=record.object_label,
                type=record.object_type,
                externalId=record.object_external_id,
                payload=dict(record.object_payload or {}),
            ).model_dump(mode="json")
        ]
    result_json = dict(record.result_json or {})
    result_kind = _execution_result_kind(result_json)
    return AgentActionExecutionItem(
        id=record.id,
        sourceChatId=record.source_chat_id,
        taskId=record.task_id,
        sourceTaskId=record.source_task_id,
        executionChatId=record.execution_chat_id,
        runId=record.run_id,
        clientMessageId=record.client_message_id,
        status=_normalize_execution_status(record.status),
        executionMode=record.execution_mode,
        prompt=record.prompt,
        task=AgentActionTaskSnapshot.model_validate(task_data),
        objectId=record.object_id,
        objectLabel=record.object_label,
        objectType=record.object_type,
        objectExternalId=record.object_external_id,
        objectPayload=dict(record.object_payload or {}),
        resultKind=result_kind,
        result=(
            ActionExecutionResult.model_validate(result_json)
            if result_kind == "action_execution"
            else None
        ),
        forecastResult=(
            ForecastDecisionResult.model_validate(result_json)
            if result_kind == "forecast_decision"
            else None
        ),
        decisionStatus=record.decision_status,
        reviewSummary=record.review_summary,
        answer=record.answer,
        serverMessageId=record.server_message_id,
        error=record.error,
        createdAt=record.created_at,
        updatedAt=record.updated_at,
        completedAt=record.completed_at,
    )


def _execution_item_to_view(item: AgentActionExecutionItem) -> AgentActionExecutionViewItem:
    return AgentActionExecutionViewItem(
        id=item.id,
        status=item.status,
        resultKind=item.resultKind,
        decisionStatus=item.decisionStatus,
        reviewSummary=item.reviewSummary,
        objectExternalId=item.objectExternalId,
        objectPayload=dict(item.objectPayload or {}),
        result=_compact_action_execution_result(item.result),
        forecastResult=_compact_forecast_result(item.forecastResult),
        error=item.error,
    )


def _compact_action_execution_result(
    result: ActionExecutionResult | None,
) -> AgentActionExecutionResultView | None:
    if result is None:
        return None
    return AgentActionExecutionResultView(
        decisionStatus=result.decisionStatus,
        confidence=result.confidence,
        blockedBy=list(result.blockedBy or []),
        summary=result.summary,
        finalRecommendation=result.finalRecommendation,
    )


def _compact_forecast_result(
    forecast_result: ForecastDecisionResult | None,
) -> AgentActionExecutionForecastView | None:
    if forecast_result is None:
        return None
    plan_family = None
    if forecast_result.plan is not None:
        plan_family = forecast_result.plan.action_family
    if plan_family is None:
        plan_family = _optional_string(
            (forecast_result.forecastDetails or {}).get("recommendedActionFamily")
        )
    return AgentActionExecutionForecastView(
        planFamily=plan_family,
        decisionStatus=forecast_result.decisionStatus,
        confidencePercent=forecast_result.confidencePercent,
        profitForecastRub=forecast_result.profitForecastRub,
        economicEffectRub=forecast_result.economicEffectRub,
        economicEffectPctOfCurrentSpend=forecast_result.economicEffectPctOfCurrentSpend,
        blockedBy=list(forecast_result.blockedBy or []),
        reviewBlocked=forecast_result.reviewBlocked,
        publishedWithTextFallback=bool(
            (forecast_result.forecastDetails or {}).get("publishedWithTextFallback")
        ),
        summary=forecast_result.summary,
        finalRecommendation=forecast_result.finalRecommendation,
        resolutionPlan=[
            AgentActionExecutionViewPlanStep(
                operation=step.operation,
                instruction=step.instruction,
            )
            for step in forecast_result.resolutionPlan
        ],
    )


def _normalize_execution_status(value: str) -> str:
    if value in {"queued", "running", "completed", "failed", "cancelled", "done"}:
        return value
    return "running"


def _execution_result_kind(
    result_json: dict[str, Any] | None,
) -> Literal["action_execution", "forecast_decision"] | None:
    if not result_json:
        return None
    if _FORECAST_RESULT_KEYS.issubset(result_json):
        return "forecast_decision"
    return "action_execution"


def _execution_result_from_trace(trace) -> dict[str, Any] | None:
    summary = dict(trace.summary or {})
    for key in ("forecast_decision_result", "action_execution_result"):
        result = summary.get(key)
        if isinstance(result, dict):
            return dict(result)
    return None


def _decision_status_from_trace(trace) -> str | None:
    summary = dict(trace.summary or {})
    value = summary.get("decision_status")
    return str(value) if isinstance(value, str) else None


def _review_summary_from_trace(trace, answer: str | None) -> str | None:
    summary = dict(trace.summary or {})
    review_summary = summary.get("review_summary")
    if isinstance(review_summary, str) and review_summary.strip():
        return review_summary.strip()
    if answer:
        compact = " ".join(answer.split())
        return compact[:197] + "..." if len(compact) > 200 else compact
    return None


def _raw_action_for_task(
    item: AgentActionTaskItem,
    raw_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        index = int(item.taskId.rsplit("_", 1)[-1]) - 1
    except ValueError:
        return {}
    if 0 <= index < len(raw_actions) and isinstance(raw_actions[index], dict):
        return raw_actions[index]
    return {}


def _extract_action_tasks_from_markdown(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    actions.extend(_extract_final_action_table(text))
    actions.extend(_extract_final_numbered_actions(text))
    for line in text.splitlines():
        match = _MARKDOWN_FINDING_PATTERN.match(line.strip())
        if not match:
            continue
        actions.append(
            {
                "source": "markdown_fallback",
                "factor": match.group("factor").strip(),
                "observation": match.group("observation").strip(),
                "impact": match.group("impact").strip(),
                "action": match.group("action").strip(),
                "evidence": [],
            }
        )
    return actions


def _extract_final_action_table(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    in_action_section = False
    table_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _FINAL_ACTION_SECTION_PATTERN.match(stripped):
            in_action_section = True
            table_lines = []
            continue
        if in_action_section and _MARKDOWN_SECTION_HEADING_PATTERN.match(stripped):
            break
        if not in_action_section:
            continue
        if "|" not in stripped or not stripped.startswith("|"):
            if table_lines:
                break
            continue
        table_lines.append(stripped)

    if len(table_lines) < 2:
        return actions

    header = _split_markdown_table_row(table_lines[0])
    indexes = _action_table_indexes(header)
    if indexes is None:
        return actions

    data_lines = table_lines[1:]
    if data_lines and _MARKDOWN_TABLE_SEPARATOR_PATTERN.match(data_lines[0]):
        data_lines = data_lines[1:]

    for line in data_lines:
        cells = _split_markdown_table_row(line)
        if len(cells) < len(header):
            continue
        task = cells[indexes["task"]].strip()
        object_cell = cells[indexes["object"]].strip()
        why = cells[indexes["why"]].strip()
        if not task:
            continue
        actions.append(
            {
                "source": "final_table_action",
                "factor": "Что делать",
                "observation": why or "Итоговая задача из финального диагностического ответа.",
                "impact": "Задача подготовлена после cross-domain диагностики.",
                "cause": why,
                "action": task,
                "title": task,
                "rawObjectText": object_cell or None,
                "targets": _visible_action_table_targets(object_cell),
                "evidence": [why] if why else [],
            }
        )
    return actions


def _action_table_indexes(header: list[str]) -> dict[str, int] | None:
    normalized = [_normalize_action_table_header(cell) for cell in header]

    def find(*candidates: str) -> int | None:
        for candidate in candidates:
            if candidate in normalized:
                return normalized.index(candidate)
        return None

    task_index = find("задача", "что сделать")
    object_index = find("объект", "где")
    why_index = find("почему", "причина")
    if task_index is None or object_index is None or why_index is None:
        return None
    return {"task": task_index, "object": object_index, "why": why_index}


def _normalize_action_table_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("ё", "е"))


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _visible_action_table_targets(value: str) -> list[str]:
    visible_value = _MORE_OBJECTS_PATTERN.sub("", value)
    targets: list[str] = []
    seen: set[str] = set()
    for match in _TARGET_NUMBER_PATTERN.findall(visible_value):
        if match in seen:
            continue
        seen.add(match)
        targets.append(match)
    return targets[:12]


def _extract_final_numbered_actions(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    in_action_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _FINAL_ACTION_SECTION_PATTERN.match(stripped):
            in_action_section = True
            continue
        if in_action_section and _MARKDOWN_SECTION_HEADING_PATTERN.match(stripped):
            break
        if not in_action_section:
            continue
        match = _NUMBERED_ACTION_PATTERN.match(stripped)
        if not match:
            continue
        action = match.group("action").strip()
        if not action:
            continue
        actions.append(
            {
                "source": "final_numbered_action",
                "factor": "Что делать",
                "observation": "Итоговая задача из финального диагностического ответа.",
                "impact": "Задача подготовлена после cross-domain диагностики.",
                "action": action,
                "evidence": [],
            }
        )
    return actions


def _action_id(*, chat_id: str, message_seq: int, index: int) -> str:
    safe_chat_id = re.sub(r"[^0-9A-Za-z_-]+", "-", chat_id).strip("-") or "chat"
    return f"action_{safe_chat_id}_{message_seq}_{index}"


def _recommendation_id(*, chat_id: str, message_seq: int, index: int) -> str:
    safe_chat_id = re.sub(r"[^0-9A-Za-z_-]+", "-", chat_id).strip("-") or "chat"
    return f"recommendation_{safe_chat_id}_{message_seq}_{index}"


def _normalize_action_task(
    raw_action: dict[str, Any],
    action_text: str,
    intent: str | None,
) -> dict[str, Any] | None:
    if _RECOMMENDATION_PATTERN.search(action_text.strip()):
        return None

    original_text = _compact_text(action_text)
    stripped_text = _strip_action_noise(original_text)
    raw_object_text = _optional_string(raw_action.get("rawObjectText"))
    targets = _normalized_task_targets(
        _extract_targets(raw_action, original_text, stripped_text),
        raw_object_text=raw_object_text,
    )
    objects = _task_objects_from_targets(targets)
    readiness, readiness_reason = _task_execution_readiness(
        targets=targets,
        raw_object_text=raw_object_text,
        objects=objects,
    )
    title = _optional_string(raw_action.get("title")) or _build_action_title(
        stripped_text=stripped_text,
        targets=targets,
        raw_action=raw_action,
    )
    if title is None:
        return None

    title = _compact_text(title.rstrip("."))
    if not _has_actionable_shape(title, targets, raw_object_text=raw_object_text):
        return None

    action_type = _valid_action_type(raw_action.get("actionType")) or _infer_action_type(title)
    priority = _valid_priority(raw_action.get("priority")) or _infer_priority(
        raw_action=raw_action,
        original_text=original_text,
    )
    success_criteria = _optional_string(raw_action.get("successCriteria")) or _success_criteria(
        title=title,
        action_type=action_type,
        targets=targets,
    )
    execution_fields = _compose_action_execution_fields(
        title=title,
        action_type=action_type,
        targets=targets,
        raw_action=raw_action,
        intent=intent,
    )
    agent_prompt = _optional_string(raw_action.get("agentPrompt")) or _agent_prompt(
        title=title,
        action_type=action_type,
        priority=priority,
        targets=targets,
        success_criteria=success_criteria,
        execution_fields=execution_fields,
        raw_action=raw_action,
        intent=intent,
    )

    return {
        "title": title,
        "rawObjectText": raw_object_text,
        "executionReadiness": readiness,
        "executionReason": readiness_reason,
        "actionType": action_type,
        "priority": priority,
        "targets": targets,
        "successCriteria": success_criteria,
        "agentPrompt": agent_prompt,
        **execution_fields,
    }


def _strip_action_noise(text: str) -> str:
    text = _CLASSIFIER_PREFIX_PATTERN.sub("", text).strip()
    text = _EMOTIONAL_PREFIX_PATTERN.sub("", text).strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    return _compact_text(text)


def _build_action_title(
    *,
    stripped_text: str,
    targets: list[str],
    raw_action: dict[str, Any],
) -> str | None:
    if stripped_text and _LEADING_TASK_VERB_PATTERN.search(stripped_text):
        return _capitalize_first(stripped_text)

    if targets and any(_TARGET_NUMBER_PATTERN.search(target) for target in targets):
        object_label = _target_object_label(raw_action)
        return f"Проверить готовность {object_label} {_format_targets(targets)} к выполнению плана"

    return None


def _has_actionable_shape(
    title: str,
    targets: list[str],
    *,
    raw_object_text: str | None = None,
) -> bool:
    if not _LEADING_TASK_VERB_PATTERN.search(title):
        return False
    return bool(targets) or bool(_BUSINESS_OBJECT_PATTERN.search(title)) or bool(_optional_string(raw_object_text))


def _extract_targets(
    raw_action: dict[str, Any],
    original_text: str,
    stripped_text: str,
) -> list[str]:
    targets: list[str] = []
    _extend_unique(targets, _string_list(raw_action.get("targets")))
    texts = [
        original_text,
        stripped_text,
        _optional_string(raw_action.get("factor")),
        _optional_string(raw_action.get("observation")),
        _optional_string(raw_action.get("impact")),
        _optional_string(raw_action.get("cause")),
        *_string_list(raw_action.get("evidence")),
    ]
    for index, text in enumerate(item for item in texts if item):
        date_matches = _DATE_TARGET_PATTERN.findall(text)
        _extend_unique(targets, date_matches)
        text_without_dates = _DATE_TARGET_PATTERN.sub(" ", text)
        _extend_unique(targets, _TARGET_NUMBER_PATTERN.findall(text_without_dates))
        if index > 1:
            continue
        phrase_matches = [
            _clean_target_phrase(match)
            for match in _TARGET_PHRASE_PATTERN.findall(text_without_dates)
        ]
        phrase_matches.extend(
            _clean_target_phrase(match)
            for match in _DIRECT_TARGET_PHRASE_PATTERN.findall(text_without_dates)
        )
        _extend_unique(targets, [match for match in phrase_matches if match])
    return targets[:12]


def _extend_unique(values: list[str], additions: list[str]) -> None:
    seen = {value.lower() for value in values}
    for addition in additions:
        value = _compact_text(str(addition).strip(" .,:;()[]"))
        if not value:
            continue
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)


def _clean_target_phrase(value: str) -> str:
    value = _compact_text(value.strip(" .,:;()[]"))
    value = re.sub(
        r"\s+(?:и|или|а|но|если|иначе|пока|чтобы)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value


def _target_object_label(raw_action: dict[str, Any]) -> str:
    context = " ".join(
        item
        for item in (
            _optional_string(raw_action.get("factor")),
            _optional_string(raw_action.get("observation")),
            _optional_string(raw_action.get("impact")),
            _optional_string(raw_action.get("action")),
        )
        if item
    ).lower()
    if "постав" in context:
        return "поставок"
    if "кампан" in context:
        return "кампаний"
    if "запрос" in context:
        return "поисковых запросов"
    return "карточек"


def _format_targets(targets: list[str]) -> str:
    if not targets:
        return ""
    if len(targets) == 1:
        return targets[0]
    return f"{', '.join(targets[:-1])} и {targets[-1]}"


def _infer_action_type(title: str) -> str:
    for action_type, pattern in _ACTION_TYPE_PATTERNS:
        if pattern.search(title):
            return action_type
    return "request_info"


def _valid_action_type(value: Any) -> str | None:
    if value in {"check", "create", "update", "calculate", "monitor", "request_info"}:
        return str(value)
    return None


def _infer_priority(raw_action: dict[str, Any], original_text: str) -> str:
    context = " ".join(
        item
        for item in [
            original_text,
            _optional_string(raw_action.get("factor")),
            _optional_string(raw_action.get("observation")),
            _optional_string(raw_action.get("impact")),
            _optional_string(raw_action.get("cause")),
            *_string_list(raw_action.get("evidence")),
        ]
        if item
    )
    if _CRITICAL_PRIORITY_PATTERN.search(context):
        return "critical"
    if _HIGH_PRIORITY_PATTERN.search(context):
        return "high"
    return "medium"


def _valid_priority(value: Any) -> str | None:
    if value in {"low", "medium", "high", "critical"}:
        return str(value)
    return None


def _success_criteria(title: str, action_type: str, targets: list[str]) -> str:
    target_part = "по каждому объекту" if len(targets) > 1 else "по объекту"
    if not targets:
        target_part = "по задаче"
    if action_type == "check":
        return f"Получен статус {target_part}: готово, не готово или требуется действие, с причиной."
    if action_type == "create":
        return f"Создано действие {target_part} или зафиксирована причина, почему оно не требуется."
    if action_type == "update":
        return f"Изменение {target_part} выполнено или зафиксирован блокер с причиной."
    if action_type == "calculate":
        return f"Расчет {target_part} выполнен, ключевые числа и вывод зафиксированы."
    if action_type == "monitor":
        return f"Статус контроля {target_part} зафиксирован, указаны отклонения и следующее действие."
    return f"Собрана недостающая информация {target_part}, результат зафиксирован."


def _normalize_action_execution_metadata(
    *,
    title: str,
    action_type: str,
    targets: list[str],
    raw_action: dict[str, Any],
    intent: str | None,
) -> dict[str, Any]:
    return _compose_action_execution_fields(
        title=title,
        action_type=action_type,
        targets=targets,
        raw_action=raw_action,
        intent=intent,
    )


def _compose_action_execution_fields(
    *,
    title: str,
    action_type: str,
    targets: list[str],
    raw_action: dict[str, Any],
    intent: str | None,
) -> dict[str, Any]:
    context = " ".join(
        item
        for item in [
            title,
            _optional_string(raw_action.get("factor")),
            _optional_string(raw_action.get("observation")),
            _optional_string(raw_action.get("impact")),
            _optional_string(raw_action.get("cause")),
            *_string_list(raw_action.get("evidence")),
            intent,
        ]
        if item
    ).lower()
    multi_target_note = _multi_target_note(targets)
    if (
        ("налич" in context or "остат" in context or "запас" in context)
        and (
            "продаж" in context
            or "выруч" in context
            or "спрос" in context
            or "реклам" in context
            or "масштаб" in context
            or "расшир" in context
        )
    ):
        return {
            "actionKind": "preserve_availability_and_sales",
            "businessGoal": "Сохранить наличие товара и не потерять продажи из-за дефицита.",
            "decisionType": "availability_replenishment_marketing_guardrail",
            "requiredMetrics": [
                "available_stock",
                "inbound_stock",
                "sales_units_recent",
                "active_days",
                "coverage_days",
                "unit_margin",
                "marketing_demand_or_spend",
            ],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "supply",
                "effect_metrics": [
                    "sales_velocity",
                    "coverage_days",
                    "oos_risk",
                    "reorder_qty",
                    "lost_sales_risk",
                ],
                "sales_velocity_formula": "sales_units_recent / active_days",
                "coverage_days_formula": "available_stock / sales_velocity",
                "reorder_qty_formula": "14d demand + safety_stock - available_stock - inbound_stock",
                "marketing_guardrail": "do not scale ads when stock coverage or margin is not confirmed",
                **multi_target_note,
            },
            "executionPlan": (
                "Проверить остаток и товары в пути, рассчитать скорость продаж и покрытие, "
                "оценить риск out-of-stock, подтвердить маржу и только после этого решить, "
                "нужен ли дозаказ или маркетинговая поддержка."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if "марж" in context or "экономик" in context or intent == "finance":
        return {
            "actionKind": "validate_unit_economics",
            "businessGoal": "Понять, можно ли выполнять действие без ухудшения экономики.",
            "decisionType": "unit_economics_guardrail",
            "requiredMetrics": ["revenue", "marketplace_costs", "cogs", "gross_margin", "contribution_margin"],
            "forecastSpec": {
                "horizon_days": 14,
                "guardrail": "do not approve scale or discount without margin",
                **multi_target_note,
            },
            "executionPlan": (
                "Проверить маржинальность, вклад расходов и финансовый блокер перед действием."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if _has_marketing_context(context, intent):
        return {
            "actionKind": "optimize_marketing_visibility",
            "businessGoal": "Понять, какие рекламные или поисковые действия можно менять без потери эффективности.",
            "decisionType": "marketing_visibility_guardrail",
            "requiredMetrics": [
                "visibility",
                "average_position",
                "demand_frequency",
                "ctr",
                "orders",
                "marketing_demand_or_spend",
            ],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "marketing",
                "effect_metrics": [
                    "ad_spend_saved",
                    "orders_at_risk",
                    "drr_signal",
                    "campaign_decision",
                ],
                "guardrail": "do not scale spend when demand, stock, or margin is not confirmed",
                **multi_target_note,
            },
            "executionPlan": (
                "Проверить видимость, позиции, спрос, рекламные расходы и заказы; затем выбрать: "
                "масштабировать, сократить, остановить или оставить с контролем."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if _has_price_context(context, intent):
        return {
            "actionKind": "validate_price_positioning",
            "businessGoal": "Понять, можно ли менять цену без ухудшения маржи, спроса и позиции относительно конкурентов.",
            "decisionType": "price_margin_competitor_guardrail",
            "requiredMetrics": [
                "current_price",
                "median_competitor_price",
                "price_delta",
                "orders",
                "gross_margin",
            ],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "price",
                "effect_metrics": [
                    "price_gap_rub",
                    "price_gap_pct",
                    "market_position_risk",
                    "price_guardrail",
                ],
                "guardrail": "do not approve price change without margin and competitor context",
                **multi_target_note,
            },
            "executionPlan": (
                "Сверить текущую цену, конкурентный коридор, продажи и маржинальность; затем решить, "
                "снижать, повышать или удерживать цену."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if _has_card_funnel_context(context):
        return {
            "actionKind": "diagnose_card_funnel",
            "businessGoal": "Найти, где карточка теряет спрос: открытие, корзина, заказ или выкуп.",
            "decisionType": "card_funnel_diagnostic",
            "requiredMetrics": [
                "open_card",
                "add_to_cart",
                "orders",
                "add_to_cart_conversion",
                "cart_to_order_conversion",
            ],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "sales",
                "effect_metrics": [
                    "delta_sales",
                    "delta_revenue",
                    "trend_direction",
                    "risk_level",
                ],
                "guardrail": "treat card quality as a hypothesis until funnel and price signals confirm it",
                **multi_target_note,
            },
            "executionPlan": (
                "Проверить открытия, корзины, заказы и конверсии карточки; отделить проблему трафика "
                "от проблемы оффера, цены или наличия."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if _has_supply_context(context, intent):
        return {
            "actionKind": "check_supply_readiness",
            "businessGoal": "Понять, хватает ли остатков и поставок для выполнения ближайшего плана.",
            "decisionType": "supply_readiness_check",
            "requiredMetrics": [
                "available_stock",
                "inbound_stock",
                "sales_units_recent",
                "coverage_days",
                "active_days",
            ],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "supply",
                "effect_metrics": [
                    "sales_velocity",
                    "coverage_days",
                    "oos_risk",
                    "reorder_qty",
                    "lost_sales_risk",
                ],
                "coverage_days_formula": "available_stock / sales_velocity",
                **multi_target_note,
            },
            "executionPlan": (
                "Проверить текущий остаток, товар в пути, скорость продаж и покрытие в днях; затем "
                "зафиксировать необходимость поставки или блокер."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    if _has_sales_context(context, intent):
        return {
            "actionKind": "diagnose_sales_performance",
            "businessGoal": "Понять, почему меняются продажи и какие бизнес-действия подтверждены данными.",
            "decisionType": "sales_performance_diagnostic",
            "requiredMetrics": ["orders", "revenue", "sales_units_recent", "open_card", "cart_to_order_conversion"],
            "forecastSpec": {
                "horizon_days": 14,
                "period_days": 7,
                "forecast_capable": True,
                "forecast_domain": "sales",
                "effect_metrics": [
                    "current_sales",
                    "current_revenue",
                    "delta_sales",
                    "delta_revenue",
                    "trend_direction",
                    "risk_level",
                ],
                **multi_target_note,
            },
            "executionPlan": (
                "Сверить продажи, заказы, выручку и воронку; затем отделить подтвержденную причину "
                "от гипотезы."
                f"{_multi_target_execution_note(targets)}"
            ),
        }

    return {
        "actionKind": f"{action_type}_business_action",
        "businessGoal": "Закрыть задачу Action Board конкретным бизнес-решением.",
        "decisionType": "operational_check",
        "requiredMetrics": ["object_status", "decision_blocker", "next_action"],
        "forecastSpec": {"horizon_days": 14, **multi_target_note},
        "executionPlan": (
            "Собрать факты по объекту, зафиксировать статус, решение и блокеры."
            f"{_multi_target_execution_note(targets)}"
        ),
    }


def _has_marketing_context(context: str, intent: str | None) -> bool:
    return intent == "marketing" or any(
        marker in context
        for marker in (
            "реклам",
            "кампан",
            "видимост",
            "ставк",
            "поисков",
            "запрос",
            "позици",
            "ctr",
            "cpc",
            "ддр",
            "drr",
        )
    )


def _has_price_context(context: str, intent: str | None) -> bool:
    return intent == "price" or any(
        marker in context
        for marker in ("цен", "скидк", "конкурент", "промо", "promo", "price")
    )


def _has_card_funnel_context(context: str) -> bool:
    return any(
        marker in context
        for marker in (
            "карточк",
            "воронк",
            "открыт",
            "корзин",
            "конверси",
            "выкуп",
        )
    )


def _has_supply_context(context: str, intent: str | None) -> bool:
    return intent == "supply" or any(
        marker in context
        for marker in ("остат", "налич", "запас", "постав", "дозаказ", "дефицит")
    )


def _has_sales_context(context: str, intent: str | None) -> bool:
    return intent == "sales" or any(
        marker in context
        for marker in ("продаж", "заказ", "выруч", "отмен", "спрос")
    )


def _multi_target_note(targets: list[str]) -> dict[str, str]:
    if len(targets) <= 1:
        return {}
    return {"target_support": "multiple_targets_review_only"}


def _multi_target_execution_note(targets: list[str]) -> str:
    if len(targets) <= 1:
        return ""
    return " Если объектов несколько, в v1 не считать общий deterministic forecast; вернуть раздельный статус и блокер multiple_targets_not_supported."


def _agent_prompt(
    *,
    title: str,
    action_type: str,
    priority: str,
    targets: list[str],
    success_criteria: str,
    execution_fields: dict[str, Any],
    raw_action: dict[str, Any],
    intent: str | None,
) -> str:
    parts = [
        f"Выполни terminal action из Action Board: {title}.",
        f"Тип задачи: {action_type}. Приоритет: {priority}.",
    ]
    parts.append(f"Action kind: {execution_fields['actionKind']}.")
    parts.append(f"Цель бизнеса: {execution_fields['businessGoal']}")
    parts.append(f"Тип решения: {execution_fields['decisionType']}.")
    if intent:
        parts.append(f"Домен: {intent}.")
    if targets:
        parts.append(f"Объекты: {_format_targets(targets)}.")
    if execution_fields["requiredMetrics"]:
        parts.append(f"Обязательные метрики: {', '.join(execution_fields['requiredMetrics'])}.")
    if execution_fields["forecastSpec"]:
        parts.append(f"Прогноз: {execution_fields['forecastSpec']}.")
    parts.append(f"План выполнения: {execution_fields['executionPlan']}")
    for label, key in (
        ("Фактор", "factor"),
        ("Наблюдение", "observation"),
        ("Влияние", "impact"),
        ("Причина", "cause"),
    ):
        value = _optional_string(raw_action.get(key))
        if value:
            parts.append(f"{label}: {value}.")
    evidence = _string_list(raw_action.get("evidence"))
    if evidence:
        parts.append(f"Факты: {'; '.join(evidence[:4])}.")
    parts.append(f"Критерий готовности: {success_criteria}")
    parts.append(
        "Верни финальный ActionExecutionResult: businessMeaning, checks, forecast, decision, "
        "decisionStatus, confidence, blockedBy, finalRecommendation, summary. "
        "Не создавай новые задачи и не выводи раздел `Что делать`."
    )
    return " ".join(parts)


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    return _optional_string(metadata.get(key))


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (_optional_string(item) for item in value) if item]
