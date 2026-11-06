import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.backend_service import _agent_run_response_summary
from agi_ecom_app.runner import AgentRuntimeError


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_response_summary_exposes_safe_tool_report_only(self):
        summary = _agent_run_response_summary(
            SimpleNamespace(
                id=1,
                question_id=7,
                agent_name="sales",
                status="completed",
                summary="ok",
                has_business_evidence=True,
                runtime_failure=None,
                created_at=None,
                completed_at=None,
                raw_output={
                    "_tool_report": {
                        "agent_name": "sales_agent",
                        "target_agent": "sales",
                        "round": "primary",
                        "question_id": 7,
                        "tool_names": ["get_sales_formula_catalog"],
                        "tools": [
                            {
                                "name": "get_sales_formula_catalog",
                                "status": "completed",
                                "duration_ms": 11,
                                "result": {"rows": ["secret"]},
                            }
                        ],
                        "required_tools": [],
                        "policy_violations": [],
                        "formula_catalog_used": True,
                        "prompt": "secret prompt",
                    }
                },
            )
        )

        self.assertEqual(summary.tool_report["tool_names"], ["get_sales_formula_catalog"])
        self.assertNotIn("result", summary.tool_report["tools"][0])
        self.assertNotIn("prompt", summary.tool_report)

    @patch("agi_ecom_app.api.get_health_payload")
    def test_health_endpoint_returns_payload(self, mock_get_health_payload):
        mock_get_health_payload.return_value = {
            "status": "degraded",
            "memory_db": {"status": "healthy", "detail": None},
            "analytics_db": {"status": "degraded", "detail": "db down"},
            "openai_config": {"status": "healthy", "detail": None},
        }

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertEqual(response.json()["analytics_db"]["detail"], "db down")

    @patch("agi_ecom_app.api.create_backend_chat_session")
    def test_chat_session_endpoint_returns_new_chat_id(self, mock_create_backend_chat_session):
        mock_create_backend_chat_session.return_value = {
            "chat_id": "chat-api-1",
            "status": "created",
        }

        response = self.client.post("/v1/chat/sessions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chat_id"], "chat-api-1")

    @patch("agi_ecom_app.api.get_chat_sessions_payload")
    def test_chat_sessions_endpoint_returns_saved_dialog_list(self, mock_get_chat_sessions_payload):
        mock_get_chat_sessions_payload.return_value = {
            "sessions": [
                {
                    "chat_id": "chat-1",
                    "status": "active",
                    "model_name": "gpt-test",
                    "created_at": "2026-04-03T12:00:00Z",
                    "updated_at": "2026-04-03T12:01:00Z",
                    "message_count": 2,
                    "run_count": 1,
                    "latest_run_id": 12,
                    "latest_run_status": "failed",
                    "latest_run_intent": "supply",
                    "latest_run_created_at": "2026-04-03T12:00:30Z",
                    "latest_server_message_id": 2,
                    "last_user_message": "Посчитай оборачиваемость склада",
                    "last_user_message_at": "2026-04-03T12:00:00Z",
                    "last_assistant_message": "Недостаточно квоты OpenAI API.",
                    "last_assistant_message_at": "2026-04-03T12:01:00Z",
                }
            ],
        }

        response = self.client.get("/v1/chat/sessions", params={"limit": 20})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sessions"][0]["chat_id"], "chat-1")
        self.assertEqual(response.json()["sessions"][0]["latest_run_id"], 12)
        self.assertEqual(
            response.json()["sessions"][0]["last_user_message"],
            "Посчитай оборачиваемость склада",
        )
        mock_get_chat_sessions_payload.assert_called_once_with(limit=20)

    @patch("agi_ecom_app.api.process_chat_message")
    def test_chat_messages_endpoint_returns_queued_run(self, mock_process_chat_message):
        mock_process_chat_message.return_value = {
            "chat_id": "chat-1",
            "answer": None,
            "intent": "finance",
            "run_id": 12,
            "server_message_id": None,
            "status": "queued",
        }

        response = self.client.post(
            "/v1/chat/messages",
            json={"chat_id": "chat-1", "message": "Какие продажи?"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["answer"], None)
        self.assertEqual(response.json()["run_id"], 12)
        self.assertEqual(response.json()["status"], "queued")

    @patch("agi_ecom_app.api.get_chat_history_payload")
    def test_chat_history_endpoint_returns_saved_messages(self, mock_get_chat_history_payload):
        mock_get_chat_history_payload.return_value = {
            "chat_id": "chat-1",
            "messages": [
                {
                    "seq": 1,
                    "role": "user",
                    "content": "Привет",
                    "created_at": "2026-04-03T12:00:00Z",
                    "client_message_id": "client-1",
                    "reply_to_client_message_id": None,
                    "server_message_id": None,
                    "intent": "finance",
                }
            ],
        }

        response = self.client.get("/v1/chat/messages", params={"chat_id": "chat-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chat_id"], "chat-1")
        self.assertEqual(response.json()["messages"][0]["client_message_id"], "client-1")

    @patch("agi_ecom_app.api.get_chat_history_payload")
    def test_chat_history_endpoint_returns_404_when_chat_is_missing(self, mock_get_chat_history_payload):
        mock_get_chat_history_payload.side_effect = LookupError("missing-chat")

        response = self.client.get("/v1/chat/messages", params={"chat_id": "missing-chat"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "chat session not found")

    @patch("agi_ecom_app.api.get_agent_run_traces_payload")
    def test_agent_traces_endpoint_returns_internal_trace_list(self, mock_get_traces_payload):
        mock_get_traces_payload.return_value = {
            "chat_id": "chat-1",
            "traces": [
                {
                    "id": 1,
                    "chat_id": "chat-1",
                    "server_message_id": 2,
                    "client_message_id": "client-1",
                    "intent": "sales",
                    "primary_agent": "sales_agent",
                    "status": "completed",
                    "started_at": "2026-04-03T12:00:00Z",
                    "completed_at": "2026-04-03T12:00:05Z",
                    "events": [
                        {
                            "ts": "2026-04-03T12:00:00Z",
                            "event": "run_started",
                            "status": "started",
                        }
                    ],
                    "summary": {"runtime_failure_count": 0},
                    "created_at": "2026-04-03T12:00:00Z",
                }
            ],
        }

        response = self.client.get(
            "/v1/chat/agent-traces",
            params={"chat_id": "chat-1", "limit": 10},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["traces"][0]["primary_agent"], "sales_agent")
        self.assertNotIn("answer", response.json()["traces"][0])
        mock_get_traces_payload.assert_called_once_with(chat_id="chat-1", limit=10)

    @patch("agi_ecom_app.api.get_agent_run_trace_payload")
    def test_agent_trace_detail_endpoint_returns_events(self, mock_get_trace_payload):
        mock_get_trace_payload.return_value = {
            "id": 1,
            "chat_id": "chat-1",
            "server_message_id": 2,
            "client_message_id": None,
            "intent": "sales",
            "primary_agent": "sales_agent",
            "status": "completed",
            "started_at": "2026-04-03T12:00:00Z",
            "completed_at": "2026-04-03T12:00:05Z",
            "events": [
                {
                    "ts": "2026-04-03T12:00:01Z",
                    "event": "primary_selected",
                    "target_agent": "sales",
                    "status": "selected",
                    "details": {"primary_agent": "sales_agent"},
                }
            ],
            "summary": {"primary_agent": "sales_agent"},
            "created_at": "2026-04-03T12:00:00Z",
        }

        response = self.client.get(
            "/v1/chat/agent-traces/2",
            params={"chat_id": "chat-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["events"][0]["event"], "primary_selected")
        self.assertNotIn("answer", response.json())
        mock_get_trace_payload.assert_called_once_with(chat_id="chat-1", server_message_id=2)

    @patch("agi_ecom_app.api.get_agent_run_trace_payload")
    def test_agent_trace_detail_endpoint_returns_404_when_missing(self, mock_get_trace_payload):
        mock_get_trace_payload.side_effect = LookupError("missing-trace")

        response = self.client.get(
            "/v1/chat/agent-traces/99",
            params={"chat_id": "chat-1"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "agent trace not found")

    @patch("agi_ecom_app.api.get_agent_run_detail_payload")
    def test_agent_run_detail_endpoint_returns_round_summary(self, mock_get_run_payload):
        mock_get_run_payload.return_value = {
            "run": {
                "id": 10,
                "chat_id": "chat-1",
                "server_message_id": 2,
                "client_message_id": None,
                "intent": "finance",
                "primary_agent": "finance_agent",
                "status": "completed",
                "started_at": "2026-04-03T12:00:00Z",
                "completed_at": "2026-04-03T12:00:05Z",
                "events": [],
                "summary": {},
                "created_at": "2026-04-03T12:00:00Z",
            },
            "rounds": [
                {
                    "id": 1,
                    "run_id": 10,
                    "round_number": 1,
                    "round_type": "cross_check",
                    "status": "completed",
                    "questions": [],
                }
            ],
        }

        response = self.client.get("/v1/chat/agent-runs/10", params={"chat_id": "chat-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run"]["id"], 10)
        self.assertEqual(response.json()["rounds"][0]["round_type"], "cross_check")
        mock_get_run_payload.assert_called_once_with(run_id=10, chat_id="chat-1")

    @patch("agi_ecom_app.api.get_agent_run_rounds_payload")
    def test_agent_run_rounds_endpoint_returns_questions_without_answer_body(self, mock_get_rounds_payload):
        mock_get_rounds_payload.return_value = {
            "run_id": 10,
            "rounds": [
                {
                    "id": 1,
                    "run_id": 10,
                    "round_number": 1,
                    "round_type": "cross_check",
                    "status": "completed",
                    "questions": [
                        {
                            "id": 7,
                            "run_id": 10,
                            "round_id": 1,
                            "source_agent": "finance",
                            "target_agent": "supply",
                            "question": "Проверь остатки.",
                            "verification_goal": {},
                            "status": "completed",
                            "responses": [
                                {
                                    "id": 3,
                                    "question_id": 7,
                                    "agent_name": "supply",
                                    "status": "completed",
                                    "summary": "Остаток низкий.",
                                    "has_business_evidence": True,
                                    "tool_report": {
                                        "agent_name": "supply_agent",
                                        "target_agent": "supply",
                                        "round": "round_1",
                                        "question_id": 7,
                                        "tool_names": ["get_supply_formula_catalog"],
                                        "tools": [
                                            {
                                                "name": "get_supply_formula_catalog",
                                                "status": "completed",
                                                "duration_ms": 5,
                                            }
                                        ],
                                        "required_tools": [],
                                        "policy_violations": [],
                                        "formula_catalog_used": True,
                                    },
                                }
                            ],
                            "gaps": [],
                        }
                    ],
                }
            ],
        }

        response = self.client.get("/v1/chat/agent-runs/10/rounds")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["rounds"][0]["questions"][0]["target_agent"], "supply")
        response_summary = response.json()["rounds"][0]["questions"][0]["responses"][0]
        self.assertNotIn("raw_output", response_summary)
        self.assertEqual(
            response_summary["tool_report"]["tool_names"],
            ["get_supply_formula_catalog"],
        )

    @patch("agi_ecom_app.api.get_agent_run_gaps_payload")
    def test_agent_run_gaps_endpoint_passes_filters(self, mock_get_gaps_payload):
        mock_get_gaps_payload.return_value = {
            "run_id": 10,
            "gaps": [
                {
                    "id": 1,
                    "run_id": 10,
                    "response_id": None,
                    "sku": "46300",
                    "source_agent": "supply",
                    "target_agent": "supply",
                    "metric": "stock",
                    "reason": "Нет данных.",
                    "blocking": True,
                    "status": "open",
                    "evidence_status": "no_data",
                }
            ],
        }

        response = self.client.get(
            "/v1/chat/agent-runs/10/gaps",
            params={"status": "open", "blocking": True, "sku": "46300"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["gaps"][0]["sku"], "46300")
        mock_get_gaps_payload.assert_called_once_with(
            run_id=10,
            chat_id=None,
            status="open",
            blocking=True,
            sku="46300",
        )

    @patch("agi_ecom_app.api.get_agent_run_rounds_payload")
    def test_agent_run_rounds_endpoint_returns_404_when_missing(self, mock_get_rounds_payload):
        mock_get_rounds_payload.side_effect = LookupError("missing-run")

        response = self.client.get("/v1/chat/agent-runs/999/rounds")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "agent run not found")

    @patch("agi_ecom_app.api.get_agent_run_status_payload")
    def test_agent_run_status_endpoint_returns_completed_answer(self, mock_get_status_payload):
        mock_get_status_payload.return_value = {
            "run_id": 10,
            "chat_id": "chat-1",
            "status": "completed",
            "intent": "finance",
            "server_message_id": 4,
            "answer": "final answer",
            "error": None,
        }

        response = self.client.get("/v1/chat/agent-runs/10/status", params={"chat_id": "chat-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "final answer")
        mock_get_status_payload.assert_called_once_with(run_id=10, chat_id="chat-1")

    @patch("agi_ecom_app.api.get_agent_run_gaps_payload")
    def test_agent_run_gaps_endpoint_returns_422_for_invalid_filter(self, mock_get_gaps_payload):
        mock_get_gaps_payload.side_effect = ValueError("invalid gap status")

        response = self.client.get(
            "/v1/chat/agent-runs/10/gaps",
            params={"status": "bad"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "invalid gap status")

    @patch("agi_ecom_app.api.process_chat_message")
    def test_chat_messages_endpoint_returns_service_error(self, mock_process_chat_message):
        mock_process_chat_message.side_effect = AgentRuntimeError("memory unavailable")

        response = self.client.post(
            "/v1/chat/messages",
            json={"chat_id": "chat-1", "message": "Какие продажи?"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "error")
        self.assertIn("memory unavailable", response.json()["detail"])

    @patch("agi_ecom_app.sku_previews.router.get_sku_previews_payload")
    def test_sku_previews_endpoint_returns_items_by_sku(self, mock_get_sku_previews_payload):
        mock_get_sku_previews_payload.return_value = {
            "items": {
                "70195339": {
                    "sku": "70195339",
                    "title": "Название товара",
                    "imageUrl": "https://example.test/photo.jpg",
                    "productUrl": "https://example.test/card",
                }
            }
        }

        response = self.client.get(
            "/v1/skus/previews",
            params={"ids": "70195339,81558446"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["items"]["70195339"]["imageUrl"],
            "https://example.test/photo.jpg",
        )
        mock_get_sku_previews_payload.assert_called_once_with("70195339,81558446")

    @patch("agi_ecom_app.sku_previews.router.get_sku_previews_payload")
    def test_sku_previews_endpoint_returns_422_for_invalid_ids(self, mock_get_sku_previews_payload):
        mock_get_sku_previews_payload.side_effect = ValueError("invalid SKU id: bad id")

        response = self.client.get("/v1/skus/previews", params={"ids": "bad id"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "invalid SKU id: bad id")


if __name__ == "__main__":
    unittest.main()
