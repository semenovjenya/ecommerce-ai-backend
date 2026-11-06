from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, Field

from agi_ecom_app.api_models import AgentRunDetailResponse
from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import get_active_llm_runtime
from agi_ecom_app.observability import generation_scope, response_payload, update_observation
from agi_ecom_app.services import strategic_db
from agi_ecom_app.services.db import get_finance_sku_report, serialize_value
from agi_ecom_app.services.memory import (
    AgentActionExecutionRecord,
    AgentRunGapRecord,
    AgentRunTraceRecord,
    StoredAgentActionTask,
    get_agent_run_trace_by_id,
    list_agent_action_executions,
    list_agent_run_gaps,
    list_agent_run_responses,
    list_agent_run_rounds,
    list_stored_agent_action_tasks,
)
from agi_ecom_app.action_board.models import (
    AgentActionExecutionItem,
    AgentActionExecutionViewItem,
    AgentActionTaskItem,
    AgentActionTasksResponse,
)
from agi_ecom_app.action_board.service import get_agent_action_tasks_payload

VerificationStatus = Literal[
    "confirmed",
    "partially_confirmed",
    "blocked",
    "no_execution",
    "not_verifiable",
]
HttpRequestFunc = Callable[..., Any]
VERIFICATION_STATUS_VALUES = [
    "confirmed",
    "partially_confirmed",
    "blocked",
    "no_execution",
    "not_verifiable",
]

SCENARIO_1_FILENAME = "scenario_1_db_agents_model_review.md"
SCENARIO_2_FILENAME = "scenario_2_backend_http_report.md"
SYNTHETIC_OBJECT_LABEL = "service token / synthetic object"
SCENARIO_MODE = "historical-only"


class Scenario1ObjectReview(BaseModel):
    object_id: str
    object_label: str | None = None
    object_type: str | None = None
    verification_status: VerificationStatus
    supports_stored_result: bool
    key_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    next_step: str


class Scenario1VerificationVerdict(BaseModel):
    verification_status: VerificationStatus
    supports_stored_result: bool
    summary: str
    key_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    next_step: str
    object_reviews: list[Scenario1ObjectReview] = Field(default_factory=list)


@dataclass(frozen=True)
class ScenarioReportRequest:
    chat_id: str
    server_message_id: int
    source_run_id: int
    source_task_ids: list[str]
    output_dir: Path

    def normalized(self) -> "ScenarioReportRequest":
        source_task_ids = [task_id.strip() for task_id in self.source_task_ids if task_id.strip()]
        if not self.chat_id.strip():
            raise ValueError("chat_id must not be empty")
        if self.server_message_id <= 0:
            raise ValueError("server_message_id must be positive")
        if self.source_run_id <= 0:
            raise ValueError("source_run_id must be positive")
        if not source_task_ids:
            raise ValueError("at least one source_task_id is required")
        if len(source_task_ids) != len(set(source_task_ids)):
            raise ValueError("source_task_ids must be unique")
        return ScenarioReportRequest(
            chat_id=self.chat_id.strip(),
            server_message_id=self.server_message_id,
            source_run_id=self.source_run_id,
            source_task_ids=source_task_ids,
            output_dir=self.output_dir,
        )


@dataclass(frozen=True)
class ParentRunBundle:
    trace: AgentRunTraceRecord
    rounds: list[dict[str, Any]]
    responses: list[dict[str, Any]]
    gaps: list[AgentRunGapRecord]


@dataclass(frozen=True)
class ScenarioObject:
    object_id: str
    label: str | None = None
    object_type: str | None = None
    external_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return is_synthetic_object(
            object_id=self.object_id,
            object_label=self.label,
            object_type=self.object_type,
            object_external_id=self.external_id,
            object_payload=self.payload,
        )

    @property
    def display_name(self) -> str:
        return self.label or self.object_id


@dataclass(frozen=True)
class Scenario1TaskReport:
    task: StoredAgentActionTask
    normalized_task: AgentActionTaskItem | None
    objects: list[ScenarioObject]
    latest_executions: list[AgentActionExecutionRecord]
    direct_facts: dict[str, Any]
    verification: Scenario1VerificationVerdict


@dataclass(frozen=True)
class Scenario2ExecutionReport:
    execution: AgentActionExecutionItem
    view: AgentActionExecutionViewItem | None
    run_detail: AgentRunDetailResponse | None


@dataclass(frozen=True)
class Scenario2TaskReport:
    task: AgentActionTaskItem
    objects: list[ScenarioObject]
    executions: list[Scenario2ExecutionReport]


def export_action_scenario_reports(
    request: ScenarioReportRequest,
    *,
    base_url: str | None = None,
    request_func: HttpRequestFunc = httpx.request,
) -> dict[str, Path]:
    normalized_request = request.normalized()
    scenario_1_markdown = build_db_agents_model_review_markdown(normalized_request)
    scenario_2_markdown = build_backend_http_markdown(
        normalized_request,
        base_url=base_url,
        request_func=request_func,
    )
    normalized_request.output_dir.mkdir(parents=True, exist_ok=True)

    scenario_1_path = normalized_request.output_dir / SCENARIO_1_FILENAME
    scenario_2_path = normalized_request.output_dir / SCENARIO_2_FILENAME
    scenario_1_path.write_text(scenario_1_markdown, encoding="utf-8")
    scenario_2_path.write_text(scenario_2_markdown, encoding="utf-8")
    return {
        "scenario_1": scenario_1_path,
        "scenario_2": scenario_2_path,
    }


def build_db_agents_model_review_markdown(
    request: ScenarioReportRequest,
) -> str:
    parent_run = load_parent_run_bundle(request)
    tasks = load_db_source_tasks(
        chat_id=request.chat_id,
        server_message_id=request.server_message_id,
        source_task_ids=request.source_task_ids,
    )
    normalized_tasks = load_normalized_source_tasks(
        chat_id=request.chat_id,
        server_message_id=request.server_message_id,
        source_task_ids=request.source_task_ids,
    )
    executions_by_task = load_latest_db_executions_by_task(
        chat_id=request.chat_id,
        source_task_ids=request.source_task_ids,
    )
    reports: list[Scenario1TaskReport] = []
    run_trace_cache: dict[int, AgentRunTraceRecord | None] = {}
    facts_cache: dict[str, dict[str, Any]] = {}

    for task in tasks:
        normalized_task = normalized_tasks.get(task.task_id)
        latest_executions = executions_by_task.get(task.task_id, [])
        objects = build_scenario_objects_from_db_task(
            task,
            latest_executions,
            normalized_task=normalized_task,
        )
        direct_facts = collect_task_direct_facts(objects=objects, facts_cache=facts_cache)
        verification = verify_task_with_model(
            task=task,
            normalized_task=normalized_task,
            objects=objects,
            latest_executions=latest_executions,
            parent_run=parent_run,
            direct_facts=direct_facts,
            run_trace_cache=run_trace_cache,
        )
        reports.append(
            Scenario1TaskReport(
                task=task,
                normalized_task=normalized_task,
                objects=objects,
                latest_executions=latest_executions,
                direct_facts=direct_facts,
                verification=verification,
            )
        )

    return render_db_agents_model_review_markdown(
        request=request,
        parent_run=parent_run,
        reports=reports,
    )


def build_backend_http_markdown(
    request: ScenarioReportRequest,
    *,
    base_url: str | None = None,
    request_func: HttpRequestFunc = httpx.request,
) -> str:
    backend_reports = fetch_backend_task_reports(
        request=request,
        base_url=base_url,
        request_func=request_func,
    )
    return render_backend_http_markdown(
        request=request,
        reports=backend_reports,
    )


def load_db_source_tasks(
    *,
    chat_id: str,
    server_message_id: int,
    source_task_ids: list[str],
) -> list[StoredAgentActionTask]:
    stored_tasks = list_stored_agent_action_tasks(
        chat_id=chat_id,
        assistant_message_seq=server_message_id,
    )
    tasks_by_id = {task.task_id: task for task in stored_tasks}
    missing = [task_id for task_id in source_task_ids if task_id not in tasks_by_id]
    if missing:
        raise LookupError(f"stored action tasks not found: {', '.join(missing)}")
    return [tasks_by_id[task_id] for task_id in source_task_ids]


def load_normalized_source_tasks(
    *,
    chat_id: str,
    server_message_id: int,
    source_task_ids: list[str],
) -> dict[str, AgentActionTaskItem]:
    response = get_agent_action_tasks_payload(
        chat_id=chat_id,
        server_message_id=server_message_id,
    )
    return {
        item.taskId: item
        for item in response.items
        if item.taskId in source_task_ids
    }


def load_parent_run_bundle(request: ScenarioReportRequest) -> ParentRunBundle:
    trace = get_agent_run_trace_by_id(request.source_run_id)
    if trace is None:
        raise LookupError(f"agent run not found: {request.source_run_id}")
    if trace.chat_id != request.chat_id:
        raise ValueError(
            f"agent run {request.source_run_id} belongs to chat {trace.chat_id}, expected {request.chat_id}"
        )
    if trace.assistant_message_seq != request.server_message_id:
        raise ValueError(
            "source_run_id does not match the provided server_message_id"
        )
    return ParentRunBundle(
        trace=trace,
        rounds=list_agent_run_rounds(request.source_run_id),
        responses=list_agent_run_responses(request.source_run_id),
        gaps=list_agent_run_gaps(run_id=request.source_run_id),
    )


def load_latest_db_executions_by_task(
    *,
    chat_id: str,
    source_task_ids: list[str],
) -> dict[str, list[AgentActionExecutionRecord]]:
    executions = [
        record
        for record in list_agent_action_executions(chat_id)
        if record.source_task_id in source_task_ids
    ]
    grouped: dict[str, dict[str, AgentActionExecutionRecord]] = {}
    for record in executions:
        task_group = grouped.setdefault(record.source_task_id, {})
        current = task_group.get(record.object_id)
        if current is None or execution_sort_key(record) > execution_sort_key(current):
            task_group[record.object_id] = record

    return {
        task_id: sorted(task_group.values(), key=execution_sort_key, reverse=True)
        for task_id, task_group in grouped.items()
    }


def build_scenario_objects_from_db_task(
    task: StoredAgentActionTask,
    latest_executions: list[AgentActionExecutionRecord],
    *,
    normalized_task: AgentActionTaskItem | None = None,
) -> list[ScenarioObject]:
    execution_objects = {
        record.object_id: scenario_object_from_execution_record(record)
        for record in latest_executions
    }
    ordered: list[ScenarioObject] = []
    seen: set[str] = set()
    task_objects = scenario_objects_from_task_sources(task=task, normalized_task=normalized_task)
    for target_object in task_objects:
        candidate = execution_objects.get(target_object.object_id, target_object)
        if candidate.object_id not in seen:
            ordered.append(candidate)
            seen.add(candidate.object_id)
    for record in latest_executions:
        candidate = scenario_object_from_execution_record(record)
        if candidate.object_id not in seen:
            ordered.append(candidate)
            seen.add(candidate.object_id)
    if ordered:
        return ordered
    return [scenario_object_from_target(task.task_id)]


def scenario_objects_from_task_sources(
    *,
    task: StoredAgentActionTask,
    normalized_task: AgentActionTaskItem | None,
) -> list[ScenarioObject]:
    if normalized_task is not None:
        if normalized_task.objects:
            return [
                ScenarioObject(
                    object_id=item.objectId,
                    label=item.label,
                    object_type=item.type,
                    external_id=item.externalId,
                    payload=dict(item.payload or {}),
                )
                for item in normalized_task.objects
            ]
        if normalized_task.targets:
            return [scenario_object_from_target(target) for target in normalized_task.targets]
        raw_object_text = task_raw_object_text(task, normalized_task=normalized_task)
        if raw_object_text:
            return [scenario_object_from_target(raw_object_text)]
    return [scenario_object_from_target(target) for target in task.targets]


def collect_task_direct_facts(
    *,
    objects: list[ScenarioObject],
    facts_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "facts_generated_at": date.today().isoformat(),
        "objects": {},
    }
    for item in objects:
        numeric_target = numeric_target_from_object(item)
        if not numeric_target:
            facts["objects"][item.object_id] = {
                "object_kind": SYNTHETIC_OBJECT_LABEL if item.is_synthetic else "non-numeric object",
                "facts_available": False,
            }
            continue
        if numeric_target not in facts_cache:
            facts_cache[numeric_target] = load_numeric_target_facts(numeric_target)
        facts["objects"][item.object_id] = facts_cache[numeric_target]
    return facts


def load_numeric_target_facts(numeric_target: str) -> dict[str, Any]:
    today = date.today()
    # Keep finance period stable and explicit.
    finance_from = date.fromordinal(today.toordinal() - 6).isoformat()
    facts = {
        "target": numeric_target,
        "facts_available": True,
        "identity": {},
        "supply_overview": {},
        "sales_revenue_change": {},
        "finance_sku": {},
        "marketing_period_comparison": {},
        "marketing_actions": {},
        "price_competitor_overview": {},
    }
    identity_map = strategic_db.get_sku_identity_map([numeric_target])
    facts["identity"] = identity_map.get(numeric_target, {})
    facts["supply_overview"] = safe_db_snapshot(
        strategic_db.get_supply_overview,
        product_keyword=numeric_target,
        limit=3,
    )
    facts["sales_revenue_change"] = safe_db_snapshot(
        strategic_db.get_sales_sku_revenue_change,
        product_keyword=numeric_target,
        period_days=7,
        limit=10,
    )
    facts["finance_sku"] = safe_db_snapshot(
        get_finance_sku_report,
        date_from=finance_from,
        date_to=today.isoformat(),
        product_keyword=numeric_target,
        limit=5,
    )
    facts["marketing_period_comparison"] = safe_db_snapshot(
        strategic_db.get_marketing_sku_period_comparison_report,
        product_keyword=numeric_target,
        period_days=7,
        limit=20,
    )
    facts["marketing_actions"] = safe_db_snapshot(
        strategic_db.get_marketing_action_matrix,
        product_keyword=numeric_target,
        days=7,
        limit=5,
    )
    facts["price_competitor_overview"] = safe_db_snapshot(
        strategic_db.get_price_competitor_overview,
        product_keyword=numeric_target,
        limit=10,
    )
    facts["finance_window"] = {
        "date_from": finance_from,
        "date_to": today.isoformat(),
        "anchor_date": today.isoformat(),
    }
    return facts


def safe_db_snapshot(func: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    try:
        payload = func(**kwargs)
    except Exception as exc:
        return {
            "error": str(exc),
            "row_count": 0,
            "rows": [],
        }
    if not isinstance(payload, dict):
        return {
            "row_count": 0,
            "rows": [],
        }
    rows = payload.get("rows")
    if isinstance(rows, list):
        payload = dict(payload)
        payload["rows"] = rows[:5]
    for key in ("campaign_actions", "query_actions", "sku_actions"):
        if isinstance(payload.get(key), list):
            payload[key] = payload[key][:5]
    return serialize_value(payload)


def verify_task_with_model(
    *,
    task: StoredAgentActionTask,
    normalized_task: AgentActionTaskItem | None,
    objects: list[ScenarioObject],
    latest_executions: list[AgentActionExecutionRecord],
    parent_run: ParentRunBundle,
    direct_facts: dict[str, Any],
    run_trace_cache: dict[int, AgentRunTraceRecord | None],
) -> Scenario1VerificationVerdict:
    if not latest_executions:
        return build_no_execution_verdict(
            task=task,
            normalized_task=normalized_task,
            objects=objects,
        )

    settings = get_settings()
    try:
        runtime = get_active_llm_runtime(settings=settings)
    except Exception:
        return build_not_verifiable_verdict(
            objects=objects,
            summary="Verification skipped: active LLM profile is not configured.",
            contradiction="Verification model is unavailable in this environment.",
        )

    prompt_payload = {
        "mode": SCENARIO_MODE,
        "task": compact_task_for_prompt(task, objects, normalized_task=normalized_task),
        "parent_run": compact_parent_run_for_prompt(parent_run),
        "latest_execution_artifacts": [
            compact_execution_for_prompt(record, run_trace_cache=run_trace_cache)
            for record in latest_executions
        ],
        "direct_business_facts": compact_direct_facts_for_prompt(direct_facts),
        "execution_present": True,
    }
    prompt = build_verification_prompt(prompt_payload)
    try:
        with generation_scope(
            name="action-scenario-verification",
            model=runtime.model_name,
            input=prompt,
            metadata={
                "model_profile": getattr(runtime, "profile", None),
                "model_provider": getattr(runtime, "provider", None),
                "model_name": runtime.model_name,
                "scenario_mode": SCENARIO_MODE,
                "execution_count": len(latest_executions),
            },
            settings=settings,
        ) as observation:
            try:
                response = runtime.sync_client.responses.create(
                    model=runtime.model_name,
                    input=prompt,
                )
                response_text = (response.output_text or "").strip()
                if not response_text:
                    raise RuntimeError("verification model returned empty response")
                verdict = Scenario1VerificationVerdict.model_validate_json(
                    extract_json_object(response_text)
                )
            except Exception as exc:
                update_observation(observation, error=exc)
                raise
            update_observation(
                observation,
                output={
                    "response": response_payload(response),
                    "parsed_verdict": verdict.model_dump(mode="json"),
                },
                response=response,
            )
    except Exception as exc:
        return build_not_verifiable_verdict(
            objects=objects,
            summary="Verification did not finish reliably.",
            contradiction=f"Model verification error: {exc}",
        )
    return finalize_verdict_objects(verdict=verdict, objects=objects)


def build_no_execution_verdict(
    *,
    task: StoredAgentActionTask,
    normalized_task: AgentActionTaskItem | None,
    objects: list[ScenarioObject],
) -> Scenario1VerificationVerdict:
    readiness = task_execution_readiness(normalized_task)
    readiness_reason = task_execution_reason(normalized_task)
    if readiness in {"needs_resolution", "blocked"}:
        object_reviews = [
            Scenario1ObjectReview(
                object_id=item.object_id,
                object_label=item.label,
                object_type=item.object_type,
                verification_status="blocked",
                supports_stored_result=False,
                key_evidence=[
                    "Historical execution artifact was not found for this task object.",
                    f"Current backend readiness is `{readiness}`.",
                ],
                contradictions=[],
                next_step=readiness_reason
                or "Resolve the task target list before inferring any completed execution.",
            )
            for item in objects
        ]
        return Scenario1VerificationVerdict(
            verification_status="blocked",
            supports_stored_result=False,
            summary=(
                f"Task `{task.task_id}` has no historical execution artifacts and is currently "
                f"`{readiness}` in backend task normalization."
            ),
            key_evidence=[
                "No matching rows were found in agent_action_executions for this source_task_id.",
                f"Current backend readiness is `{readiness}`.",
            ],
            contradictions=[],
            next_step=readiness_reason
            or "Resolve the task target list before treating this as executable.",
            object_reviews=object_reviews,
        )
    object_reviews = [
        Scenario1ObjectReview(
            object_id=item.object_id,
            object_label=item.label,
            object_type=item.object_type,
            verification_status="no_execution",
            supports_stored_result=False,
            key_evidence=["Historical execution artifact was not found for this task object."],
            contradictions=[],
            next_step="Keep the task as historical-only input; do not infer a completed result.",
        )
        for item in objects
    ]
    return Scenario1VerificationVerdict(
        verification_status="no_execution",
        supports_stored_result=False,
        summary=(
            f"Task `{task.task_id}` exists in the action board, but no historical execution artifacts were found."
        ),
        key_evidence=["No matching rows were found in agent_action_executions for this source_task_id."],
        contradictions=[],
        next_step="Treat this task as planned but not executed in the historical window.",
        object_reviews=object_reviews,
    )


def build_not_verifiable_verdict(
    *,
    objects: list[ScenarioObject],
    summary: str,
    contradiction: str,
) -> Scenario1VerificationVerdict:
    object_reviews = [
        Scenario1ObjectReview(
            object_id=item.object_id,
            object_label=item.label,
            object_type=item.object_type,
            verification_status="not_verifiable",
            supports_stored_result=False,
            key_evidence=[],
            contradictions=[contradiction],
            next_step="Review the raw execution artifact and direct DB facts manually.",
        )
        for item in objects
    ]
    return Scenario1VerificationVerdict(
        verification_status="not_verifiable",
        supports_stored_result=False,
        summary=summary,
        key_evidence=[],
        contradictions=[contradiction],
        next_step="Fallback to manual review of the stored artifact and DB facts.",
        object_reviews=object_reviews,
    )


def finalize_verdict_objects(
    *,
    verdict: Scenario1VerificationVerdict,
    objects: list[ScenarioObject],
) -> Scenario1VerificationVerdict:
    reviews_by_id = {review.object_id: review for review in verdict.object_reviews}
    finalized_reviews: list[Scenario1ObjectReview] = []
    for item in objects:
        review = reviews_by_id.get(item.object_id)
        if review is None:
            review = Scenario1ObjectReview(
                object_id=item.object_id,
                object_label=item.label,
                object_type=item.object_type,
                verification_status="not_verifiable",
                supports_stored_result=False,
                key_evidence=[],
                contradictions=["The model did not return an object-level review for this object."],
                next_step="Review this object manually.",
            )
        review = review.model_copy(
            update={
                "object_label": item.label,
                "object_type": item.object_type,
            }
        )
        finalized_reviews.append(review)
    return verdict.model_copy(update={"object_reviews": finalized_reviews})


def build_verification_prompt(prompt_payload: dict[str, Any]) -> str:
    schema = {
        "verification_status": VERIFICATION_STATUS_VALUES,
        "supports_stored_result": "boolean",
        "summary": "string",
        "key_evidence": ["string"],
        "contradictions": ["string"],
        "next_step": "string",
        "object_reviews": [
            {
                "object_id": "string",
                "object_label": "string|null",
                "object_type": "string|null",
                "verification_status": VERIFICATION_STATUS_VALUES,
                "supports_stored_result": "boolean",
                "key_evidence": ["string"],
                "contradictions": ["string"],
                "next_step": "string",
            }
        ],
    }
    return (
        "You validate historical action execution evidence for an e-commerce analytics workflow.\n"
        "Return only one JSON object. No markdown, no prose outside JSON.\n"
        "Be conservative. Never invent missing metrics or claim confirmation without explicit evidence.\n"
        "Allowed verification_status values: confirmed, partially_confirmed, blocked, no_execution, not_verifiable.\n"
        "supports_stored_result may be true only when the stored execution result is materially supported by the evidence.\n"
        "If evidence is mixed, prefer partially_confirmed or blocked.\n"
        "If synthetic objects like campaign/service tokens cannot be validated as business objects, say so explicitly.\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Payload:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2, default=json_default)}"
    )


def compact_task_for_prompt(
    task: StoredAgentActionTask,
    objects: list[ScenarioObject],
    *,
    normalized_task: AgentActionTaskItem | None,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "action": task.action,
        "factor": task.factor,
        "observation": task.observation,
        "impact": task.impact,
        "cause": task.cause,
        "targets": list(task.targets),
        "required_metrics": list(task.required_metrics),
        "action_kind": task.action_kind,
        "business_goal": task.business_goal,
        "decision_type": task.decision_type,
        "forecast_spec": dict(task.forecast_spec or {}),
        "execution_plan": task.execution_plan,
        "evidence": list(task.evidence),
        "normalized_targets": list(normalized_task.targets) if normalized_task is not None else [],
        "raw_object_text": task_raw_object_text(task, normalized_task=normalized_task),
        "execution_readiness": task_execution_readiness(normalized_task),
        "execution_reason": task_execution_reason(normalized_task),
        "objects": [
            {
                "object_id": item.object_id,
                "label": item.label,
                "object_type": item.object_type,
                "external_id": item.external_id,
                "payload": item.payload,
                "is_synthetic": item.is_synthetic,
            }
            for item in objects
        ],
    }


def compact_parent_run_for_prompt(parent_run: ParentRunBundle) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    for round_payload in parent_run.rounds:
        round_record = round_payload["round"]
        questions: list[dict[str, Any]] = []
        for question_payload in round_payload["questions"][:5]:
            question = question_payload["question"]
            questions.append(
                {
                    "question_id": question.id,
                    "target_agent": question.target_agent,
                    "status": question.status,
                    "reason": question.reason,
                    "question": excerpt(question.question, limit=220),
                    "responses": [
                        {
                            "response_id": response.id,
                            "agent_name": response.agent_name,
                            "status": response.status,
                            "summary": excerpt(response.summary, limit=220),
                            "has_business_evidence": response.has_business_evidence,
                        }
                        for response in question_payload["responses"][:3]
                    ],
                }
            )
        rounds.append(
            {
                "round_number": round_record.round_number,
                "round_type": round_record.round_type,
                "status": round_record.status,
                "questions": questions,
            }
        )
    return {
        "run_id": parent_run.trace.id,
        "status": parent_run.trace.status,
        "primary_agent": parent_run.trace.primary_agent,
        "assistant_message_seq": parent_run.trace.assistant_message_seq,
        "started_at": isoformat_or_none(parent_run.trace.started_at),
        "completed_at": isoformat_or_none(parent_run.trace.completed_at),
        "summary": {
            "status": parent_run.trace.summary.get("status"),
            "secondary_agents": parent_run.trace.summary.get("secondary_agents"),
            "followup_agents": parent_run.trace.summary.get("followup_agents"),
            "final_quality_violations": parent_run.trace.summary.get("final_quality_violations"),
            "round_1_plan": parent_run.trace.summary.get("round_1_plan"),
            "round_2_plan": parent_run.trace.summary.get("round_2_plan"),
        },
        "rounds": rounds,
        "gaps": [
            {
                "gap_id": gap.id,
                "metric": gap.metric,
                "source_agent": gap.source_agent,
                "target_agent": gap.target_agent,
                "blocking": gap.blocking,
                "status": gap.status,
                "evidence_status": gap.evidence_status,
                "reason": excerpt(gap.reason, limit=220),
            }
            for gap in parent_run.gaps[:10]
        ],
    }


def compact_execution_for_prompt(
    record: AgentActionExecutionRecord,
    *,
    run_trace_cache: dict[int, AgentRunTraceRecord | None],
) -> dict[str, Any]:
    run_trace = None
    if record.run_id is not None:
        if record.run_id not in run_trace_cache:
            run_trace_cache[record.run_id] = get_agent_run_trace_by_id(record.run_id)
        run_trace = run_trace_cache[record.run_id]
    return {
        "execution_id": record.id,
        "source_task_id": record.source_task_id,
        "object_id": record.object_id,
        "object_label": record.object_label,
        "object_type": record.object_type,
        "object_external_id": record.object_external_id,
        "object_payload": dict(record.object_payload or {}),
        "status": record.status,
        "decision_status": record.decision_status,
        "review_summary": excerpt(record.review_summary, limit=320),
        "answer_excerpt": excerpt(record.answer, limit=400),
        "result_kind": infer_result_kind(record.result_json),
        "result_json": compact_result_json(record.result_json),
        "run_trace": (
            {
                "run_id": run_trace.id,
                "primary_agent": run_trace.primary_agent,
                "status": run_trace.status,
                "assistant_message_seq": run_trace.assistant_message_seq,
                "summary": {
                    "decision_status": run_trace.summary.get("decision_status"),
                    "forecast_plan": run_trace.summary.get("forecast_plan"),
                    "resolved_targets": run_trace.summary.get("resolved_targets"),
                },
            }
            if run_trace is not None
            else None
        ),
    }


def compact_direct_facts_for_prompt(direct_facts: dict[str, Any]) -> dict[str, Any]:
    compact_objects: dict[str, Any] = {}
    for object_id, object_facts in (direct_facts.get("objects") or {}).items():
        if not isinstance(object_facts, dict):
            compact_objects[object_id] = object_facts
            continue
        if not object_facts.get("facts_available", True):
            compact_objects[object_id] = object_facts
            continue
        compact_objects[object_id] = {
            "target": object_facts.get("target"),
            "identity": object_facts.get("identity"),
            "supply_overview": compact_report_row(object_facts.get("supply_overview")),
            "sales_revenue_change": compact_report_row(object_facts.get("sales_revenue_change")),
            "finance_sku": compact_report_row(object_facts.get("finance_sku")),
            "marketing_period_comparison": compact_report_row(
                object_facts.get("marketing_period_comparison")
            ),
            "price_competitor_overview": compact_report_row(
                object_facts.get("price_competitor_overview")
            ),
            "marketing_actions": {
                key: object_facts.get("marketing_actions", {}).get(key, [])
                for key in ("campaign_actions", "query_actions", "sku_actions")
            }
            if isinstance(object_facts.get("marketing_actions"), dict)
            else object_facts.get("marketing_actions"),
        }
    return {
        "facts_generated_at": direct_facts.get("facts_generated_at"),
        "objects": compact_objects,
    }


def compact_report_row(report_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report_payload, dict):
        return {}
    rows = report_payload.get("rows")
    compact = {key: value for key, value in report_payload.items() if key != "rows"}
    if isinstance(rows, list):
        compact["rows"] = rows[:2]
    return compact


def render_db_agents_model_review_markdown(
    *,
    request: ScenarioReportRequest,
    parent_run: ParentRunBundle,
    reports: list[Scenario1TaskReport],
) -> str:
    lines = render_report_header(
        title="Scenario 1: DB + existing agent artifacts + model review",
        request=request,
    )
    lines.extend(
        [
            "",
            "## Parent Run",
            "",
            f"- run_id: `{parent_run.trace.id}`",
            f"- primary_agent: `{parent_run.trace.primary_agent}`",
            f"- status: `{parent_run.trace.status}`",
            f"- started_at: `{isoformat_or_none(parent_run.trace.started_at)}`",
            f"- completed_at: `{isoformat_or_none(parent_run.trace.completed_at)}`",
            f"- round_count: `{len(parent_run.rounds)}`",
            f"- gap_count: `{len(parent_run.gaps)}`",
        ]
    )
    lines.extend(render_summary_table_scenario_1(reports))
    appendix_rows: list[list[str]] = []
    for index, report in enumerate(reports, start=1):
        lines.extend(render_scenario_1_task_section(index=index, report=report))
        appendix_rows.extend(render_scenario_1_appendix_rows(report, request))
    lines.extend(render_appendix_table(appendix_rows))
    return "\n".join(lines).rstrip() + "\n"


def render_backend_http_markdown(
    *,
    request: ScenarioReportRequest,
    reports: list[Scenario2TaskReport],
) -> str:
    lines = render_report_header(
        title="Scenario 2: Backend HTTP GET only",
        request=request,
    )
    lines.extend(render_summary_table_scenario_2(reports))
    appendix_rows: list[list[str]] = []
    for index, report in enumerate(reports, start=1):
        lines.extend(render_scenario_2_task_section(index=index, report=report))
        appendix_rows.extend(render_scenario_2_appendix_rows(report, request))
    lines.extend(render_appendix_table(appendix_rows))
    return "\n".join(lines).rstrip() + "\n"


def render_report_header(title: str, request: ScenarioReportRequest) -> list[str]:
    return [
        f"# {title}",
        "",
        f"- mode: `{SCENARIO_MODE}`",
        f"- chat_id: `{request.chat_id}`",
        f"- server_message_id: `{request.server_message_id}`",
        f"- source_run_id: `{request.source_run_id}`",
        f"- source_task_ids: `{', '.join(request.source_task_ids)}`",
    ]


def render_summary_table_scenario_1(reports: list[Scenario1TaskReport]) -> list[str]:
    lines = [
        "",
        "## Summary",
        "",
        "| # | Task | Objects | Latest executions | Model status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, report in enumerate(reports, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    md_cell(report.task.title or report.task.action),
                    md_cell(str(len(report.objects))),
                    md_cell(str(len(report.latest_executions))),
                    md_cell(report.verification.verification_status),
                ]
            )
            + " |"
        )
    return lines


def render_summary_table_scenario_2(reports: list[Scenario2TaskReport]) -> list[str]:
    lines = [
        "",
        "## Summary",
        "",
        "| # | Task | Objects | Backend executions | Backend status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, report in enumerate(reports, start=1):
        status = no_execution_status_label(report.task)
        if report.executions:
            status = ", ".join(
                sorted({execution.execution.status for execution in report.executions})
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    md_cell(report.task.title or report.task.action),
                    md_cell(str(len(report.objects))),
                    md_cell(str(len(report.executions))),
                    md_cell(status),
                ]
            )
            + " |"
        )
    return lines


def render_scenario_1_task_section(
    *,
    index: int,
    report: Scenario1TaskReport,
) -> list[str]:
    raw_object_text = task_raw_object_text(report.task, normalized_task=report.normalized_task)
    lines = [
        "",
        f"## Task {index}: {report.task.title or report.task.action}",
        "",
        f"- task_id: `{report.task.task_id}`",
        f"- action_kind: `{report.task.action_kind}`",
        f"- decision_type: `{report.task.decision_type}`",
        f"- targets: `{task_targets_text(report.task.targets, normalized_task=report.normalized_task, raw_object_text=raw_object_text)}`",
        f"- raw_object_text: `{raw_object_text or 'n/a'}`",
        f"- execution_readiness: `{task_execution_readiness(report.normalized_task) or 'n/a'}`",
        f"- execution_reason: {task_execution_reason(report.normalized_task) or 'n/a'}",
        f"- success_criteria: {report.task.success_criteria or 'n/a'}",
        f"- latest_execution_count: `{len(report.latest_executions)}`",
        "",
        "### Model Verification",
        "",
        f"- verification_status: `{report.verification.verification_status}`",
        f"- supports_stored_result: `{str(report.verification.supports_stored_result).lower()}`",
        f"- summary: {report.verification.summary}",
        f"- next_step: {report.verification.next_step}",
    ]
    lines.extend(render_text_list("key_evidence", report.verification.key_evidence))
    lines.extend(render_text_list("contradictions", report.verification.contradictions))
    lines.extend(
        [
            "",
            "### Object Results",
            "",
            "| object_id | class | execution_status | decision_status | result_kind | review_status | note |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    reviews_by_id = {item.object_id: item for item in report.verification.object_reviews}
    latest_by_object = {item.object_id: item for item in report.latest_executions}
    for item in report.objects:
        execution = latest_by_object.get(item.object_id)
        review = reviews_by_id.get(item.object_id)
        note = SYNTHETIC_OBJECT_LABEL if item.is_synthetic else "business object"
        execution_note = "n/a"
        if execution is not None and execution.review_summary:
            execution_note = execution.review_summary
        elif review is not None:
            execution_note = review.next_step
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(item.object_id),
                    md_cell(note),
                    md_cell(execution.status if execution else "no_execution"),
                    md_cell(execution.decision_status or "n/a") if execution else "no_execution",
                    md_cell(infer_result_kind(execution.result_json) if execution else "n/a"),
                    md_cell(review.verification_status if review else "not_verifiable"),
                    md_cell(execution_note),
                ]
            )
            + " |"
        )
        if execution:
            lines.extend(
                [
                    "",
                    f"Raw object payload for `{item.object_id}`:",
                    "```json",
                    json.dumps(execution.object_payload or {}, ensure_ascii=False, indent=2, default=json_default),
                    "```",
                ]
            )
    lines.extend(
        [
            "",
            "### Direct DB Facts",
            "",
            "```json",
            json.dumps(report.direct_facts, ensure_ascii=False, indent=2, default=json_default),
            "```",
        ]
    )
    return lines


def render_scenario_2_task_section(
    *,
    index: int,
    report: Scenario2TaskReport,
) -> list[str]:
    raw_object_text = task_raw_object_text(report.task)
    lines = [
        "",
        f"## Task {index}: {report.task.title or report.task.action}",
        "",
        f"- task_id: `{report.task.taskId}`",
        f"- action_kind: `{report.task.actionKind}`",
        f"- decision_type: `{report.task.decisionType}`",
        f"- targets: `{task_targets_text(report.task.targets, raw_object_text=raw_object_text)}`",
        f"- raw_object_text: `{raw_object_text or 'n/a'}`",
        f"- execution_readiness: `{task_execution_readiness(report.task) or 'n/a'}`",
        f"- execution_reason: {task_execution_reason(report.task) or 'n/a'}",
        f"- backend_execution_count: `{len(report.executions)}`",
    ]
    if not report.executions:
        no_execution_status = no_execution_status_label(report.task)
        no_execution_note = (
            "Task exists in backend `/v1/agent-actions`, but no historical execution was returned by "
            "`/v1/agent-actions/executions`."
        )
        if no_execution_status in {"needs_resolution", "blocked"}:
            no_execution_note = (
                "Task exists in backend `/v1/agent-actions`, but current backend normalization marks "
                f"it as `{no_execution_status}`, so historical auto-execution is not expected until "
                "targets are resolved."
            )
        lines.extend(
            [
                "",
                "### Execution State",
                "",
                f"- status: `{no_execution_status}`",
                f"- note: {no_execution_note}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Object Results",
                "",
                "| object_id | class | status | decision_status | result_kind | run_id | review_summary |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for execution_report in report.executions:
            execution = execution_report.execution
            scenario_object = scenario_object_from_backend_execution(execution)
            note = SYNTHETIC_OBJECT_LABEL if scenario_object.is_synthetic else "business object"
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(execution.objectId),
                        md_cell(note),
                        md_cell(execution.status),
                        md_cell(execution.decisionStatus or "n/a"),
                        md_cell(execution.resultKind or "n/a"),
                        md_cell(str(execution.runId or "n/a")),
                        md_cell(execution.reviewSummary or "n/a"),
                    ]
                )
                + " |"
            )
            lines.extend(
                [
                    "",
                    f"Raw object payload for `{execution.objectId}`:",
                    "```json",
                    json.dumps(execution.objectPayload, ensure_ascii=False, indent=2, default=json_default),
                    "```",
                    "",
                    f"Assistant answer excerpt for `{execution.objectId}`:",
                    "",
                    f"> {md_quote(excerpt(execution.answer, limit=420) or 'n/a')}",
                ]
            )
            if execution_report.run_detail is not None:
                lines.extend(
                    [
                        "",
                        f"Run metadata for `{execution.objectId}`:",
                        "```json",
                        json.dumps(
                            compact_backend_run_detail(execution_report.run_detail),
                            ensure_ascii=False,
                            indent=2,
                            default=json_default,
                        ),
                        "```",
                    ]
                )
            if execution_report.view is not None:
                lines.extend(
                    [
                        "",
                        f"Backend view payload for `{execution.objectId}`:",
                        "```json",
                        json.dumps(
                            execution_report.view.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                            default=json_default,
                        ),
                        "```",
                    ]
                )
    return lines


def render_appendix_table(rows: list[list[str]]) -> list[str]:
    lines = [
        "",
        "## Appendix",
        "",
        "| task_id | execution_id | object_id | run_id | chat_id | server_message_id |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in rows:
        lines.append("| " + " | ".join(md_cell(value) for value in row) + " |")
    return lines


def render_scenario_1_appendix_rows(
    report: Scenario1TaskReport,
    request: ScenarioReportRequest,
) -> list[list[str]]:
    if not report.latest_executions:
        return [[report.task.task_id, "n/a", "n/a", "n/a", request.chat_id, str(request.server_message_id)]]
    return [
        [
            report.task.task_id,
            str(item.id),
            item.object_id,
            str(item.run_id or "n/a"),
            request.chat_id,
            str(request.server_message_id),
        ]
        for item in report.latest_executions
    ]


def render_scenario_2_appendix_rows(
    report: Scenario2TaskReport,
    request: ScenarioReportRequest,
) -> list[list[str]]:
    if not report.executions:
        return [[report.task.taskId, "n/a", "n/a", "n/a", request.chat_id, str(request.server_message_id)]]
    return [
        [
            report.task.taskId,
            str(item.execution.id),
            item.execution.objectId,
            str(item.execution.runId or "n/a"),
            request.chat_id,
            str(request.server_message_id),
        ]
        for item in report.executions
    ]


def render_text_list(label: str, items: list[str]) -> list[str]:
    if not items:
        return [f"- {label}: none"]
    lines = [f"- {label}:"]
    lines.extend([f"  - {item}" for item in items])
    return lines


def task_raw_object_text(
    task: StoredAgentActionTask | AgentActionTaskItem,
    *,
    normalized_task: AgentActionTaskItem | None = None,
) -> str | None:
    if normalized_task is not None:
        return optional_str(normalized_task.rawObjectText)
    if isinstance(task, AgentActionTaskItem):
        return optional_str(task.rawObjectText)
    raw_action = task.raw_action if isinstance(task.raw_action, dict) else {}
    raw_object_text = optional_str(raw_action.get("rawObjectText"))
    if raw_object_text:
        return raw_object_text
    action_text = optional_str(raw_action.get("action")) or optional_str(task.action)
    if action_text and ":" in action_text:
        _, suffix = action_text.split(":", 1)
        return optional_str(suffix)
    return None


def task_execution_readiness(task: AgentActionTaskItem | None) -> str | None:
    if task is None:
        return None
    return optional_str(task.executionReadiness)


def task_execution_reason(task: AgentActionTaskItem | None) -> str | None:
    if task is None:
        return None
    return optional_str(task.executionReason)


def task_targets_text(
    targets: list[str],
    *,
    normalized_task: AgentActionTaskItem | None = None,
    raw_object_text: str | None = None,
) -> str:
    normalized_targets = list(normalized_task.targets) if normalized_task is not None else list(targets)
    if normalized_targets:
        return ", ".join(normalized_targets)
    return raw_object_text or "n/a"


def no_execution_status_label(task: AgentActionTaskItem) -> str:
    readiness = task_execution_readiness(task)
    if readiness in {"needs_resolution", "blocked"}:
        return readiness
    return "no_execution"


def optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def fetch_backend_task_reports(
    *,
    request: ScenarioReportRequest,
    base_url: str | None = None,
    request_func: HttpRequestFunc = httpx.request,
) -> list[Scenario2TaskReport]:
    payload = backend_get_json(
        base_url=base_url,
        path="/v1/agent-actions",
        params={
            "chat_id": request.chat_id,
            "server_message_id": request.server_message_id,
        },
        request_func=request_func,
    )
    tasks_response = AgentActionTasksResponse.model_validate(payload)
    tasks_by_id = {item.taskId: item for item in tasks_response.items}
    missing = [task_id for task_id in request.source_task_ids if task_id not in tasks_by_id]
    if missing:
        raise LookupError(f"backend action tasks not found: {', '.join(missing)}")

    executions_payload = backend_get_json(
        base_url=base_url,
        path="/v1/agent-actions/executions",
        params={"chat_id": request.chat_id},
        request_func=request_func,
    )
    execution_items = [
        AgentActionExecutionItem.model_validate(item)
        for item in executions_payload.get("items", [])
        if item.get("sourceTaskId") in request.source_task_ids
    ]
    executions_by_task: dict[str, dict[str, AgentActionExecutionItem]] = {}
    for item in execution_items:
        task_group = executions_by_task.setdefault(item.sourceTaskId, {})
        current = task_group.get(item.objectId)
        if current is None or backend_execution_sort_key(item) > backend_execution_sort_key(current):
            task_group[item.objectId] = item

    reports: list[Scenario2TaskReport] = []
    for task_id in request.source_task_ids:
        task = tasks_by_id[task_id]
        objects = build_scenario_objects_from_backend_task(task)
        latest_executions = sorted(
            executions_by_task.get(task_id, {}).values(),
            key=backend_execution_sort_key,
            reverse=True,
        )
        execution_reports: list[Scenario2ExecutionReport] = []
        for execution in latest_executions:
            view_payload = backend_get_json(
                base_url=base_url,
                path=f"/v1/agent-actions/executions/{execution.id}/view",
                request_func=request_func,
            )
            run_detail = None
            if execution.runId is not None:
                run_payload = backend_get_json(
                    base_url=base_url,
                    path=f"/v1/chat/agent-runs/{execution.runId}",
                    request_func=request_func,
                )
                run_detail = AgentRunDetailResponse.model_validate(run_payload)
            execution_reports.append(
                Scenario2ExecutionReport(
                    execution=execution,
                    view=AgentActionExecutionViewItem.model_validate(view_payload),
                    run_detail=run_detail,
                )
            )
        reports.append(
            Scenario2TaskReport(
                task=task,
                objects=objects,
                executions=execution_reports,
            )
        )
    return reports


def build_scenario_objects_from_backend_task(task: AgentActionTaskItem) -> list[ScenarioObject]:
    if task.objects:
        return [
            ScenarioObject(
                object_id=item.objectId,
                label=item.label,
                object_type=item.type,
                external_id=item.externalId,
                payload=dict(item.payload or {}),
            )
            for item in task.objects
        ]
    if task.targets:
        return [scenario_object_from_target(target) for target in task.targets]
    raw_object_text = task_raw_object_text(task)
    if raw_object_text:
        return [scenario_object_from_target(raw_object_text)]
    return [scenario_object_from_target(task.taskId)]


def scenario_object_from_execution_record(record: AgentActionExecutionRecord) -> ScenarioObject:
    return ScenarioObject(
        object_id=record.object_id,
        label=record.object_label,
        object_type=record.object_type,
        external_id=record.object_external_id,
        payload=dict(record.object_payload or {}),
    )


def scenario_object_from_backend_execution(execution: AgentActionExecutionItem) -> ScenarioObject:
    return ScenarioObject(
        object_id=execution.objectId,
        label=execution.objectLabel,
        object_type=execution.objectType,
        external_id=execution.objectExternalId,
        payload=dict(execution.objectPayload or {}),
    )


def scenario_object_from_target(target: str) -> ScenarioObject:
    cleaned = (target or "").strip()
    if cleaned.isdigit():
        return ScenarioObject(
            object_id=f"sku-{cleaned}",
            label=f"SKU {cleaned}",
            object_type="sku",
            external_id=cleaned,
            payload={"nm_id": int(cleaned)},
        )
    return ScenarioObject(
        object_id=cleaned or "unknown-object",
        label=cleaned or "unknown-object",
        object_type=None,
        external_id=None,
        payload={},
    )


def is_synthetic_object(
    *,
    object_id: str,
    object_label: str | None,
    object_type: str | None,
    object_external_id: str | None,
    object_payload: dict[str, Any] | None,
) -> bool:
    if object_type == "sku":
        return False
    if object_external_id and object_external_id.isdigit():
        return False
    payload = object_payload or {}
    nm_id = payload.get("nm_id")
    if nm_id is not None and str(nm_id).isdigit():
        return False
    normalized = " ".join(
        item.strip().lower()
        for item in (object_id, object_label or "", str(payload.get("vendor_code") or ""))
        if item and item.strip()
    )
    if "все продвигаемые sku" in normalized:
        return True
    return not any(token.isdigit() for token in normalized.split())


def numeric_target_from_object(item: ScenarioObject) -> str | None:
    if item.external_id and item.external_id.isdigit():
        return item.external_id
    nm_id = item.payload.get("nm_id") if isinstance(item.payload, dict) else None
    if nm_id is not None and str(nm_id).isdigit():
        return str(nm_id)
    if item.object_id.startswith("sku-") and item.object_id[4:].isdigit():
        return item.object_id[4:]
    return None


def compact_result_json(result_json: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result_json, dict) or not result_json:
        return {}
    if infer_result_kind(result_json) == "forecast_decision":
        plan = result_json.get("plan") if isinstance(result_json.get("plan"), dict) else {}
        return {
            "decisionStatus": result_json.get("decisionStatus"),
            "confidence": result_json.get("confidence"),
            "confidencePercent": result_json.get("confidencePercent"),
            "summary": excerpt(result_json.get("summary"), limit=240),
            "finalRecommendation": excerpt(result_json.get("finalRecommendation"), limit=240),
            "blockedBy": result_json.get("blockedBy", []),
            "action_family": plan.get("action_family"),
            "resolvedTargets": result_json.get("resolvedTargets", []),
        }
    forecast = result_json.get("forecast") if isinstance(result_json.get("forecast"), dict) else {}
    return {
        "decisionStatus": result_json.get("decisionStatus"),
        "confidence": result_json.get("confidence"),
        "summary": excerpt(result_json.get("summary"), limit=240),
        "finalRecommendation": excerpt(result_json.get("finalRecommendation"), limit=240),
        "blockedBy": result_json.get("blockedBy", []),
        "forecast": {
            "coverageDays": forecast.get("coverageDays"),
            "reorderQty": forecast.get("reorderQty"),
            "stockoutRisk": forecast.get("stockoutRisk"),
        },
    }


def infer_result_kind(result_json: dict[str, Any] | None) -> str:
    if not isinstance(result_json, dict) or not result_json:
        return "empty"
    if "confidencePercent" in result_json:
        return "forecast_decision"
    return "action_execution"


def backend_get_json(
    *,
    base_url: str | None,
    path: str,
    params: dict[str, Any] | None = None,
    request_func: HttpRequestFunc = httpx.request,
    timeout: float = 60.0,
) -> dict[str, Any]:
    settings = get_settings()
    normalized_base_url = (base_url or settings.backend_base_url).rstrip("/")
    response = request_func(
        "GET",
        f"{normalized_base_url}{path}",
        params=params,
        timeout=timeout,
    )
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        raise RuntimeError("backend request did not return an HTTP-like response")
    if status_code >= 400:
        detail = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("message") or payload
        except Exception:
            detail = getattr(response, "text", "")
        raise RuntimeError(f"Backend HTTP {status_code}: {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object payload from backend path {path}")
    return payload


def compact_backend_run_detail(run_detail: AgentRunDetailResponse) -> dict[str, Any]:
    return {
        "run_id": run_detail.run.id,
        "chat_id": run_detail.run.chat_id,
        "server_message_id": run_detail.run.server_message_id,
        "primary_agent": run_detail.run.primary_agent,
        "status": run_detail.run.status,
        "started_at": isoformat_or_none(run_detail.run.started_at),
        "completed_at": isoformat_or_none(run_detail.run.completed_at),
        "summary": {
            "status": run_detail.run.summary.get("status"),
            "decision_status": run_detail.run.summary.get("decision_status"),
            "primary_agent": run_detail.run.summary.get("primary_agent"),
            "forecast_plan": run_detail.run.summary.get("forecast_plan"),
            "resolved_targets": run_detail.run.summary.get("resolved_targets"),
        },
        "rounds": [
            {
                "round_number": round_item.round_number,
                "round_type": round_item.round_type,
                "status": round_item.status,
                "question_count": len(round_item.questions),
            }
            for round_item in run_detail.rounds
        ],
    }


def execution_sort_key(record: AgentActionExecutionRecord) -> tuple[datetime, int]:
    return (record.created_at or datetime.min, record.id)


def backend_execution_sort_key(record: AgentActionExecutionItem) -> tuple[datetime, int]:
    return (record.createdAt or datetime.min, record.id)


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("response does not contain a JSON object")
    return stripped[start : end + 1]


def excerpt(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def isoformat_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def md_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return text or "n/a"


def md_quote(value: str) -> str:
    return value.replace("\n", "\n> ")


def json_default(value: Any) -> Any:
    return serialize_value(value)
