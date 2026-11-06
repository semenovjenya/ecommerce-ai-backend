from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.api_models import AgentRunStatusResponse, ChatMessageResponse
from agi_ecom_app.backend_service import (
    get_agent_run_status_payload,
    get_chat_history_payload,
    process_chat_message,
)
from agi_ecom_app.manager_decision.models import ManagerDecisionV1
from agi_ecom_app.schemas import ChatRole


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def _manager_decision() -> ManagerDecisionV1:
    return ManagerDecisionV1.model_validate(
        {
            "schemaVersion": "1.0",
            "decisionId": "manager-decision-transport-1",
            "queryMode": "decision",
            "status": "needs_data",
            "createdAt": NOW.isoformat(),
            "requestCoverage": {
                "explicitTargets": ["46300"],
                "coveredTargets": [],
                "missingTargets": ["46300"],
                "requestedOutputs": ["manager_action"],
                "coveredOutputs": [],
                "missingOutputs": ["manager_action"],
                "complete": False,
            },
            "rankedActions": [],
            "rejectedCandidates": [],
            "blockers": [],
            "evidenceSnapshot": [],
        }
    )


def _public_decision_payload() -> dict:
    return _manager_decision().model_dump(mode="json", by_alias=True)


def test_duplicate_request_reuses_same_structured_manager_decision() -> None:
    decision_payload = _public_decision_payload()
    cached_reply = SimpleNamespace(
        seq=19,
        content="Нужны данные по SKU 46300.",
        metadata={
            "intent": "sales",
            "manager_decision": decision_payload,
        },
    )

    with (
        patch(
            "agi_ecom_app.agent_queue.get_settings",
            return_value=SimpleNamespace(agent_workflow_version="v2_optimized"),
        ),
        patch("agi_ecom_app.agent_queue._active_model_label", return_value="test:model"),
        patch("agi_ecom_app.agent_queue.ensure_chat_session"),
        patch(
            "agi_ecom_app.agent_queue.get_assistant_reply_for_client_message_id",
            return_value=cached_reply,
        ) as get_cached_reply,
        patch("agi_ecom_app.agent_queue.get_agent_run_trace_by_client_message_id") as get_run,
        patch("agi_ecom_app.agent_queue.create_agent_run_trace") as create_run,
    ):
        first = process_chat_message(
            "manager-transport-chat",
            "Составь план действий",
            "manager-client-message-1",
        )
        duplicate = process_chat_message(
            "manager-transport-chat",
            "Составь план действий",
            "manager-client-message-1",
        )

    assert first.status == duplicate.status == "completed"
    assert first.server_message_id == duplicate.server_message_id == 19
    assert first.managerDecision is not None
    assert duplicate.managerDecision is not None
    assert first.managerDecision.model_dump(mode="json", by_alias=True) == decision_payload
    assert duplicate.managerDecision.model_dump(mode="json", by_alias=True) == decision_payload
    assert get_cached_reply.call_count == 2
    get_run.assert_not_called()
    create_run.assert_not_called()


def test_status_and_history_restore_identical_structured_manager_decision() -> None:
    decision_payload = _public_decision_payload()
    assistant_message = SimpleNamespace(
        seq=19,
        role=ChatRole.ASSISTANT,
        content="Нужны данные по SKU 46300.",
        created_at=NOW,
        metadata={
            "intent": "sales",
            "reply_to_client_message_id": "manager-client-message-1",
            "manager_decision": decision_payload,
        },
    )
    trace = SimpleNamespace(
        id=77,
        chat_id="manager-transport-chat",
        status="completed",
        intent="sales",
        assistant_message_seq=19,
        summary={},
    )

    with (
        patch("agi_ecom_app.backend_service.chat_session_exists", return_value=True),
        patch(
            "agi_ecom_app.backend_service.list_all_stored_messages",
            return_value=[assistant_message],
        ),
        patch("agi_ecom_app.backend_service.get_agent_run_trace_by_id", return_value=trace),
    ):
        history = get_chat_history_payload("manager-transport-chat")
        status = get_agent_run_status_payload(77, "manager-transport-chat")

    history_decision = history.messages[0].managerDecision
    assert history_decision is not None
    assert status.managerDecision is not None
    assert history_decision.model_dump(mode="json", by_alias=True) == decision_payload
    assert status.managerDecision.model_dump(mode="json", by_alias=True) == decision_payload


def test_websocket_duplicate_cached_response_keeps_public_manager_decision() -> None:
    decision = _manager_decision()
    expected_payload = _public_decision_payload()
    response = ChatMessageResponse(
        chat_id="manager-transport-chat",
        answer="Нужны данные по SKU 46300.",
        intent="sales",
        server_message_id=19,
        status="completed",
        managerDecision=decision,
    )

    with (
        patch("agi_ecom_app.websocket_chat.ensure_chat_session"),
        patch("agi_ecom_app.websocket_chat.list_agent_run_traces", return_value=[]),
        patch(
            "agi_ecom_app.websocket_chat.process_chat_message",
            return_value=response,
        ) as process_message,
    ):
        with TestClient(app).websocket_connect(
            "/v1/chat/ws?chat_id=manager-transport-chat"
        ) as websocket:
            assert websocket.receive_json()["type"] == "connection.ready"
            completed_payloads = []
            for _ in range(2):
                websocket.send_json(
                    {
                        "type": "message.send",
                        "client_message_id": "manager-client-message-1",
                        "chat_id": "manager-transport-chat",
                        "text": "Составь план действий",
                    }
                )
                assert websocket.receive_json()["type"] == "message.accepted"
                completed_payloads.append(websocket.receive_json())

    assert process_message.call_count == 2
    assert completed_payloads[0]["managerDecision"] == expected_payload
    assert completed_payloads[1]["managerDecision"] == expected_payload
    assert "decisionId" in completed_payloads[1]["managerDecision"]
    assert "decision_id" not in completed_payloads[1]["managerDecision"]


def test_websocket_reconnect_replays_same_structured_manager_decision() -> None:
    decision = _manager_decision()
    expected_payload = _public_decision_payload()
    trace = SimpleNamespace(
        id=77,
        chat_id="manager-reconnect-chat",
        client_message_id="manager-client-message-1",
    )
    status_payload = AgentRunStatusResponse(
        run_id=77,
        chat_id="manager-reconnect-chat",
        status="completed",
        intent="sales",
        server_message_id=19,
        answer="Нужны данные по SKU 46300.",
        managerDecision=decision,
    )

    with (
        patch("agi_ecom_app.websocket_chat.ensure_chat_session"),
        patch(
            "agi_ecom_app.websocket_chat.list_agent_run_traces",
            return_value=[trace],
        ),
        patch(
            "agi_ecom_app.websocket_chat.get_agent_run_status_payload",
            return_value=status_payload,
        ),
    ):
        with TestClient(app).websocket_connect(
            "/v1/chat/ws?chat_id=manager-reconnect-chat"
        ) as websocket:
            assert websocket.receive_json()["type"] == "connection.ready"
            completed = websocket.receive_json()

    assert completed["type"] == "message.completed"
    assert completed["client_message_id"] == "manager-client-message-1"
    assert completed["managerDecision"] == expected_payload
    assert "decisionId" in completed["managerDecision"]
    assert "decision_id" not in completed["managerDecision"]
