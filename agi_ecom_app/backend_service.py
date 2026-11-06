from __future__ import annotations

from dataclasses import asdict
from typing import Any

from agi_ecom_app.api_models import (
    ActivePromosResponse,
    AgentRunDetailResponse,
    AgentRunGap,
    AgentRunGapsResponse,
    AgentRunQuestion,
    AgentRunResponseSummary,
    AgentRunRound,
    AgentRunRoundsResponse,
    AgentRunStatusResponse,
    AgentRunTrace,
    AgentRunTraceListResponse,
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatSessionListItem,
    ChatSessionListResponse,
    ChatSessionResponse,
    DashboardResponse,
    DependencyStatus,
    FlashSaleSkuDetail,
    FlashSaleWinterResponse,
    HealthResponse,
    ModelProfileResponse,
    ModelProfileUpdateRequest,
    PromoItem,
)
from agi_ecom_app.agent_queue import enqueue_initial_run
from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import (
    LLMRuntimeConfigurationError,
    get_active_llm_runtime_spec,
    get_available_model_profile_descriptions,
    validate_model_profile_configured,
)
from agi_ecom_app.model_profiles import describe_model_profile, get_model_profile
from agi_ecom_app.manager_decision.models import ManagerDecisionV1
from agi_ecom_app.services.db import CliDashboardKpis, get_cli_dashboard_kpis, ping_database
from agi_ecom_app.services.memory import (
    AgentRunGapRecord,
    AgentRunQuestionRecord,
    AgentRunResponseRecord,
    AgentRunTraceRecord,
    chat_session_exists,
    create_chat_session,
    get_agent_run_trace,
    get_agent_run_trace_by_id,
    list_all_stored_messages,
    list_agent_run_gaps,
    list_agent_run_rounds,
    list_agent_run_traces,
    list_chat_session_summaries,
    ping_memory_db,
)
from agi_ecom_app.services.runtime_settings import set_active_model_profile_name


def create_backend_chat_session() -> ChatSessionResponse:
    return ChatSessionResponse(chat_id=create_chat_session(model_name=_active_model_label()))


def get_chat_sessions_payload(limit: int = 50) -> ChatSessionListResponse:
    return ChatSessionListResponse(
        sessions=[
            ChatSessionListItem(
                chat_id=session.chat_id,
                status=session.status,
                model_name=session.model_name,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=session.message_count,
                run_count=session.run_count,
                latest_run_id=session.latest_run_id,
                latest_run_status=session.latest_run_status,
                latest_run_intent=session.latest_run_intent,
                latest_run_created_at=session.latest_run_created_at,
                latest_server_message_id=session.latest_server_message_id,
                last_user_message=session.last_user_message,
                last_user_message_at=session.last_user_message_at,
                last_assistant_message=session.last_assistant_message,
                last_assistant_message_at=session.last_assistant_message_at,
            )
            for session in list_chat_session_summaries(limit=limit)
        ],
    )


def process_chat_message(
    chat_id: str,
    message: str,
    client_message_id: str | None = None,
) -> ChatMessageResponse:
    result = enqueue_initial_run(
        chat_id=chat_id,
        user_query=message,
        client_message_id=client_message_id,
    )
    return ChatMessageResponse(
        chat_id=chat_id,
        answer=result.answer,
        intent=result.intent.value,
        run_id=result.run_id,
        server_message_id=result.server_message_id,
        status=result.status,
        managerDecision=_manager_decision_from_value(result.manager_decision),
    )


def get_model_profile_payload() -> ModelProfileResponse:
    return _model_profile_response()


def update_model_profile_payload(payload: ModelProfileUpdateRequest) -> ModelProfileResponse:
    settings = get_settings()
    profile_name = payload.profile.strip()
    profile = get_model_profile(profile_name, settings)
    try:
        validate_model_profile_configured(profile.profile, settings=settings)
    except LLMRuntimeConfigurationError as exc:
        raise ValueError(str(exc)) from exc
    set_active_model_profile_name(profile.profile)
    return _model_profile_response()


def get_chat_history_payload(chat_id: str) -> ChatHistoryResponse:
    if not chat_session_exists(chat_id):
        raise LookupError(chat_id)

    stored_messages = list_all_stored_messages(chat_id)
    return ChatHistoryResponse(
        chat_id=chat_id,
        messages=[
            ChatHistoryMessage(
                seq=message.seq,
                role=message.role.value,
                content=message.content,
                created_at=message.created_at,
                client_message_id=_read_metadata_string(message.metadata, "client_message_id"),
                reply_to_client_message_id=_read_metadata_string(
                    message.metadata,
                    "reply_to_client_message_id",
                ),
                server_message_id=message.seq if message.role.value == "assistant" else None,
                intent=_read_metadata_string(message.metadata, "intent"),
                managerDecision=_manager_decision_from_value(
                    (message.metadata or {}).get("manager_decision")
                ),
            )
            for message in stored_messages
        ],
    )


def get_agent_run_traces_payload(chat_id: str, limit: int = 50) -> AgentRunTraceListResponse:
    if not chat_session_exists(chat_id):
        raise LookupError(chat_id)

    return AgentRunTraceListResponse(
        chat_id=chat_id,
        traces=[
            _agent_run_trace_response(trace)
            for trace in list_agent_run_traces(chat_id=chat_id, limit=limit)
        ],
    )


def get_agent_run_trace_payload(chat_id: str, server_message_id: int) -> AgentRunTrace:
    if not chat_session_exists(chat_id):
        raise LookupError(chat_id)

    trace = get_agent_run_trace(
        chat_id=chat_id,
        assistant_message_seq=server_message_id,
    )
    if not trace:
        raise LookupError(f"{chat_id}:{server_message_id}")
    return _agent_run_trace_response(trace)


def get_agent_run_detail_payload(run_id: int, chat_id: str | None = None) -> AgentRunDetailResponse:
    trace = _get_agent_run_trace_by_id_or_raise(run_id=run_id, chat_id=chat_id)
    return AgentRunDetailResponse(
        run=_agent_run_trace_response(trace),
        rounds=_agent_run_rounds_response(run_id),
    )


def get_agent_run_rounds_payload(run_id: int, chat_id: str | None = None) -> AgentRunRoundsResponse:
    _get_agent_run_trace_by_id_or_raise(run_id=run_id, chat_id=chat_id)
    return AgentRunRoundsResponse(
        run_id=run_id,
        rounds=_agent_run_rounds_response(run_id),
    )


def get_agent_run_gaps_payload(
    *,
    run_id: int,
    chat_id: str | None = None,
    status: str | None = None,
    blocking: bool | None = None,
    sku: str | None = None,
) -> AgentRunGapsResponse:
    _get_agent_run_trace_by_id_or_raise(run_id=run_id, chat_id=chat_id)
    _validate_gap_status(status)
    return AgentRunGapsResponse(
        run_id=run_id,
        gaps=[
            _agent_run_gap_response(gap)
            for gap in list_agent_run_gaps(
                run_id=run_id,
                status=status,
                blocking=blocking,
                sku=sku,
            )
        ],
    )


def get_agent_run_status_payload(run_id: int, chat_id: str | None = None) -> AgentRunStatusResponse:
    trace = _get_agent_run_trace_by_id_or_raise(run_id=run_id, chat_id=chat_id)
    answer = None
    manager_decision = None
    if trace.assistant_message_seq is not None:
        for message in list_all_stored_messages(trace.chat_id):
            if message.role.value == "assistant" and message.seq == trace.assistant_message_seq:
                answer = message.content
                manager_decision = _manager_decision_from_value(
                    (message.metadata or {}).get("manager_decision")
                )
                break
    error = None
    if trace.status == "failed":
        error = str((trace.summary or {}).get("error") or "agent run failed")
    return AgentRunStatusResponse(
        run_id=trace.id,
        chat_id=trace.chat_id,
        status=trace.status,
        intent=trace.intent,
        server_message_id=trace.assistant_message_seq,
        answer=answer if trace.status == "completed" else None,
        error=error,
        managerDecision=manager_decision,
    )


def get_daily_dashboard_payload() -> DashboardResponse:
    try:
        kpis = get_cli_dashboard_kpis()
    except Exception:
        kpis = CliDashboardKpis.unavailable()
    return DashboardResponse.model_validate(asdict(kpis))


def get_active_promos_payload() -> ActivePromosResponse:
    return ActivePromosResponse(
        items=[
            PromoItem(
                name="РАСПРОДАЖА -15%",
                eff="88%",
                sku=12,
                alerts="Дефицит (2 SKU)",
                status="pos",
            ),
            PromoItem(
                name="WEEKEND BOOST",
                eff="94%",
                sku=45,
                alerts="Норма",
                status="pos",
            ),
            PromoItem(
                name="NEW COLLECTION",
                eff="72%",
                sku=120,
                alerts="Высокая цена",
                status="warn",
            ),
            PromoItem(
                name="ЛИКВИДАЦИЯ",
                eff="65%",
                sku=500,
                alerts="Низкая конв.",
                status="neg",
            ),
            PromoItem(
                name="VIP EXCLUSIVE",
                eff="98%",
                sku=8,
                alerts="Норма",
                status="pos",
            ),
        ]
    )


def get_flash_sale_winter_payload() -> FlashSaleWinterResponse:
    return FlashSaleWinterResponse(
        analyzed_items_text="Анализирую 12 товаров в этой кампании...",
        impact_on_plan="+5.2% к цели месяца",
        impact_on_plan_status="pos",
        revenue_text="1.84 млн ₽",
        revenue_status="neutral",
        profit_text="412 000 ₽",
        profit_status="pos",
        plan_status_text="102% (опережаем)",
        plan_status_kind="pos",
        margin_text="22.4% (была 27%, компенсирована объёмом)",
        margin_status="warn",
        drr_text="12% (норма)",
        drr_status="neutral",
        turnover_text="+14 дней (высвободили 860 000 ₽)",
        turnover_status="pos",
        storage_savings_text="8 000 ₽/мес",
        storage_savings_status="pos",
        sku_details=[
            FlashSaleSkuDetail(
                sku_label='SKU-123 "Кроссовки беговые"',
                badge="✅",
                badge_color="green",
                action_text="Продлить до 14.02",
                action_color="green",
                sales="+34% (12 → 16/день)",
                sales_status="pos",
                position="45 → 28",
                position_status="pos",
                margin="19%",
                margin_status="neutral",
                stock="на 8 дней, поставка 12.02 в пути",
                stock_status="neutral",
            ),
            FlashSaleSkuDetail(
                sku_label='SKU-456 "Легинсы Classic"',
                badge="⚠️",
                badge_color="yellow",
                action_text="Исключить СЕГОДНЯ",
                action_color="red",
                sales="+76% (аномальный спрос)",
                sales_status="pos",
                position="12 → 4 (уже ТОП)",
                position_status="pos",
                margin="11% (минимум)",
                margin_status="neg",
                stock="на 3 дня — риск Stockout",
                stock_status="neg",
                note="Поднять цену, притормозить до поставки",
                note_color="magenta",
            ),
        ],
        recommendation_text="Продлить 10 SKU, исключить 2 SKU.",
    )


def get_health_payload() -> HealthResponse:
    settings = get_settings()

    memory_status = _dependency_status(ping_memory_db)
    analytics_status = _dependency_status(ping_database)
    openai_status = (
        DependencyStatus(status="healthy")
        if settings.openai_api_key
        else DependencyStatus(status="degraded", detail="OPENAI_API_KEY is not configured")
    )
    llm_status = _active_llm_dependency_status()

    statuses = [memory_status.status, analytics_status.status, llm_status.status]
    overall_status = "healthy" if all(status == "healthy" for status in statuses) else "degraded"

    return HealthResponse(
        status=overall_status,
        memory_db=memory_status,
        analytics_db=analytics_status,
        openai_config=openai_status,
        llm_config=llm_status,
    )


def _dependency_status(check) -> DependencyStatus:
    try:
        check()
        return DependencyStatus(status="healthy")
    except Exception as exc:
        return DependencyStatus(status="degraded", detail=str(exc))


def _active_llm_dependency_status() -> DependencyStatus:
    try:
        spec = get_active_llm_runtime_spec(require_configured=False)
    except Exception as exc:
        return DependencyStatus(status="degraded", detail=str(exc))
    if spec.missing_env_vars:
        missing = ", ".join(spec.missing_env_vars)
        return DependencyStatus(
            status="degraded",
            detail=f"{spec.display_name} requires missing env var(s): {missing}",
        )
    return DependencyStatus(status="healthy", detail=spec.display_name)


def _model_profile_response() -> ModelProfileResponse:
    settings = get_settings()
    spec = get_active_llm_runtime_spec(require_configured=False, settings=settings)
    status = "degraded" if spec.missing_env_vars else "healthy"
    detail = None
    if spec.missing_env_vars:
        detail = f"Missing env var(s): {', '.join(spec.missing_env_vars)}"
    return ModelProfileResponse(
        profile=spec.profile,
        provider=spec.provider,
        model=spec.model_name,
        model_label=spec.display_name,
        api_style=spec.api_style,
        status=status,
        detail=detail,
        memory_provider="openai",
        memory_model=settings.openai_memory_model,
        available_profiles=[
            describe_model_profile(get_model_profile(str(item["profile"]), settings), settings)
            for item in get_available_model_profile_descriptions(settings)
        ],
    )


def _active_model_label() -> str:
    try:
        return str(get_active_llm_runtime_spec(require_configured=False).display_name)
    except Exception:
        settings = get_settings()
        return f"openai:{settings.openai_model}"


def _read_metadata_string(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _manager_decision_from_value(value: Any) -> ManagerDecisionV1 | None:
    if isinstance(value, ManagerDecisionV1):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return ManagerDecisionV1.model_validate(value)
    except Exception:
        # Old chat messages and partially written runs must remain readable.
        return None


def _agent_run_trace_response(trace: AgentRunTraceRecord) -> AgentRunTrace:
    return AgentRunTrace(
        id=trace.id,
        chat_id=trace.chat_id,
        server_message_id=trace.assistant_message_seq,
        client_message_id=trace.client_message_id,
        intent=trace.intent,
        primary_agent=trace.primary_agent,
        status=trace.status,
        started_at=trace.started_at,
        completed_at=trace.completed_at,
        events=trace.events,
        summary=trace.summary,
        created_at=trace.created_at,
    )


def _get_agent_run_trace_by_id_or_raise(
    *,
    run_id: int,
    chat_id: str | None = None,
) -> AgentRunTraceRecord:
    trace = get_agent_run_trace_by_id(run_id)
    if not trace or (chat_id is not None and trace.chat_id != chat_id):
        raise LookupError(run_id)
    return trace


def _agent_run_rounds_response(run_id: int) -> list[AgentRunRound]:
    return [
        AgentRunRound(
            id=round_payload["round"].id,
            run_id=round_payload["round"].run_id,
            round_number=round_payload["round"].round_number,
            round_type=round_payload["round"].round_type,
            status=round_payload["round"].status,
            created_at=round_payload["round"].created_at,
            updated_at=round_payload["round"].updated_at,
            completed_at=round_payload["round"].completed_at,
            questions=[
                _agent_run_question_response(question_payload)
                for question_payload in round_payload["questions"]
            ],
        )
        for round_payload in list_agent_run_rounds(run_id)
    ]


def _agent_run_question_response(payload: dict) -> AgentRunQuestion:
    question: AgentRunQuestionRecord = payload["question"]
    return AgentRunQuestion(
        id=question.id,
        run_id=question.run_id,
        round_id=question.round_id,
        source_agent=question.source_agent,
        target_agent=question.target_agent,
        question=question.question,
        reason=question.reason,
        verification_goal=question.verification_goal,
        priority=question.priority,
        status=question.status,
        job_id=question.job_id,
        queued_at=question.queued_at,
        attempt_count=question.attempt_count,
        last_error=question.last_error,
        timeout_seconds=question.timeout_seconds,
        parent_question_id=question.parent_question_id,
        created_at=question.created_at,
        updated_at=question.updated_at,
        started_at=question.started_at,
        completed_at=question.completed_at,
        responses=[
            _agent_run_response_summary(response)
            for response in payload.get("responses", [])
        ],
        gaps=[
            _agent_run_gap_response(gap)
            for gap in payload.get("gaps", [])
        ],
    )


def _agent_run_response_summary(response: AgentRunResponseRecord) -> AgentRunResponseSummary:
    return AgentRunResponseSummary(
        id=response.id,
        question_id=response.question_id,
        agent_name=response.agent_name,
        status=response.status,
        summary=response.summary,
        has_business_evidence=response.has_business_evidence,
        runtime_failure=response.runtime_failure,
        tool_report=_safe_tool_report(response.raw_output),
        created_at=response.created_at,
        completed_at=response.completed_at,
    )


def _safe_tool_report(raw_output: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw_output, dict):
        return None
    report = raw_output.get("_tool_report")
    if not isinstance(report, dict):
        return None
    safe_tools: list[dict[str, Any]] = []
    for item in report.get("tools") or []:
        if not isinstance(item, dict):
            continue
        safe_tools.append(
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "duration_ms": item.get("duration_ms"),
            }
        )
    return {
        "agent_name": report.get("agent_name"),
        "target_agent": report.get("target_agent"),
        "round": report.get("round"),
        "question_id": report.get("question_id"),
        "tool_names": list(report.get("tool_names") or []),
        "tools": safe_tools,
        "required_tools": list(report.get("required_tools") or []),
        "policy_violations": list(report.get("policy_violations") or []),
        "formula_catalog_used": bool(report.get("formula_catalog_used")),
    }


def _agent_run_gap_response(gap: AgentRunGapRecord) -> AgentRunGap:
    return AgentRunGap(
        id=gap.id,
        run_id=gap.run_id,
        response_id=gap.response_id,
        sku=gap.sku,
        source_agent=gap.source_agent,
        target_agent=gap.target_agent,
        metric=gap.metric,
        reason=gap.reason,
        impact=gap.impact,
        blocking=gap.blocking,
        status=gap.status,
        evidence_status=gap.evidence_status,
        created_at=gap.created_at,
        updated_at=gap.updated_at,
    )


def _validate_gap_status(status: str | None) -> None:
    if status is None:
        return
    if status not in {"open", "resolved", "accepted_limitation"}:
        raise ValueError("invalid gap status")
