from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agi_ecom_app.reporting.action_scenarios import (
    ParentRunBundle,
    Scenario1TaskReport,
    Scenario1VerificationVerdict,
    Scenario2ExecutionReport,
    Scenario2TaskReport,
    ScenarioObject,
    ScenarioReportRequest,
    build_no_execution_verdict,
    build_scenario_objects_from_db_task,
    fetch_backend_task_reports,
    is_synthetic_object,
    load_db_source_tasks,
    render_backend_http_markdown,
    render_db_agents_model_review_markdown,
)
from agi_ecom_app.services.memory import AgentRunTraceRecord, StoredAgentActionTask
from agi_ecom_app.api_models import AgentRunDetailResponse, AgentRunTrace
from agi_ecom_app.action_board.models import (
    AgentActionExecutionItem,
    AgentActionExecutionViewItem,
    AgentActionTaskItem,
    AgentActionTaskSnapshot,
)


def _stored_task(task_id: str, *, targets: list[str] | None = None) -> StoredAgentActionTask:
    return StoredAgentActionTask(
        chat_id="chat-1",
        task_id=task_id,
        action_id=task_id,
        assistant_message_seq=4,
        source="final_table_action",
        factor="Что делать",
        observation="obs",
        impact="impact",
        cause="cause",
        action=f"Action {task_id}",
        title=f"Title {task_id}",
        action_type="check",
        priority="high",
        targets=targets or ["46227"],
        success_criteria="criteria",
        agent_prompt="prompt",
        evidence=["evidence"],
        confidence=None,
        driver_class=None,
        intent="marketing",
        raw_action={"action": f"Action {task_id}"},
        action_kind="optimize_marketing_visibility",
        business_goal="goal",
        decision_type="decision",
        required_metrics=["metric"],
        forecast_spec={"forecast_domain": "marketing"},
        execution_plan="plan",
    )


def _parent_run_bundle() -> ParentRunBundle:
    return ParentRunBundle(
        trace=AgentRunTraceRecord(
            id=87,
            chat_id="chat-1",
            assistant_message_seq=4,
            client_message_id="client-1",
            intent="marketing",
            primary_agent="marketing_agent",
            status="completed",
            started_at=datetime(2026, 4, 26, 13, 21, 17),
            completed_at=datetime(2026, 4, 26, 13, 27, 10),
            events=[],
            summary={"status": "completed"},
        ),
        rounds=[],
        responses=[],
        gaps=[],
    )


class DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class ActionScenarioReportTests(unittest.TestCase):
    @patch("agi_ecom_app.reporting.action_scenarios.list_stored_agent_action_tasks")
    def test_load_db_source_tasks_returns_only_requested_items(self, mock_list):
        mock_list.return_value = [
            _stored_task("task-1"),
            _stored_task("task-2"),
            _stored_task("task-3"),
        ]

        result = load_db_source_tasks(
            chat_id="chat-1",
            server_message_id=4,
            source_task_ids=["task-3", "task-1"],
        )

        self.assertEqual([item.task_id for item in result], ["task-3", "task-1"])

    def test_renderer_marks_synthetic_objects(self):
        report = Scenario1TaskReport(
            task=_stored_task("task-1", targets=["46227", "кампании"]),
            normalized_task=None,
            objects=[
                ScenarioObject("sku-46227", "SKU 46227", "sku", "46227", {"nm_id": 46227}),
                ScenarioObject("кампании", "кампании", None, None, {}),
            ],
            latest_executions=[],
            direct_facts={"objects": {}},
            verification=build_no_execution_verdict(
                task=_stored_task("task-1", targets=["46227", "кампании"]),
                normalized_task=None,
                objects=[
                    ScenarioObject("sku-46227", "SKU 46227", "sku", "46227", {"nm_id": 46227}),
                    ScenarioObject("кампании", "кампании", None, None, {}),
                ],
            ),
        )

        markdown = render_db_agents_model_review_markdown(
            request=ScenarioReportRequest(
                chat_id="chat-1",
                server_message_id=4,
                source_run_id=87,
                source_task_ids=["task-1"],
                output_dir=Path("docs/action-scenario-reports"),
            ),
            parent_run=_parent_run_bundle(),
            reports=[report],
        )

        self.assertIn("service token / synthetic object", markdown)

    def test_no_execution_verdict_is_conservative(self):
        verdict = build_no_execution_verdict(
            task=_stored_task("task-5", targets=["все продвигаемые SKU"]),
            normalized_task=None,
            objects=[ScenarioObject("все продвигаемые SKU", "все продвигаемые SKU", None, None, {})],
        )

        self.assertEqual(verdict.verification_status, "no_execution")
        self.assertFalse(verdict.supports_stored_result)
        self.assertIn("No matching rows", verdict.key_evidence[0])

    def test_no_execution_verdict_uses_blocked_for_needs_resolution_tasks(self):
        verdict = build_no_execution_verdict(
            task=_stored_task("task-5", targets=["все продвигаемые SKU"]),
            normalized_task=AgentActionTaskItem(
                taskId="task-5",
                actionId="task-5",
                chatId="chat-1",
                serverMessageId=4,
                factor="Что делать",
                action="Сверить группы с ростом CPC",
                title="Сверить группы с ростом CPC",
                rawObjectText="все продвигаемые SKU",
                executionReadiness="needs_resolution",
                executionReason="Raw object text is present, but no executable targets were extracted.",
                targets=[],
                objects=[],
            ),
            objects=[ScenarioObject("все продвигаемые SKU", "все продвигаемые SKU", None, None, {})],
        )

        self.assertEqual(verdict.verification_status, "blocked")
        self.assertIn("needs_resolution", verdict.summary)

    def test_is_synthetic_object_detects_service_tokens(self):
        self.assertTrue(
            is_synthetic_object(
                object_id="кампании",
                object_label="кампании",
                object_type=None,
                object_external_id=None,
                object_payload={},
            )
        )
        self.assertFalse(
            is_synthetic_object(
                object_id="sku-46227",
                object_label="SKU 46227",
                object_type="sku",
                object_external_id="46227",
                object_payload={"nm_id": 46227},
            )
        )

    def test_build_scenario_objects_from_db_task_reconstructs_targets(self):
        task = _stored_task("task-1", targets=["46227", "цену"])

        objects = build_scenario_objects_from_db_task(task, [])

        self.assertEqual([item.object_id for item in objects], ["sku-46227", "цену"])

    def test_build_scenario_objects_from_db_task_prefers_normalized_raw_object_text(self):
        task = _stored_task("task-5", targets=["все продвигаемые SKU"])

        objects = build_scenario_objects_from_db_task(
            task,
            [],
            normalized_task=AgentActionTaskItem(
                taskId="task-5",
                actionId="task-5",
                chatId="chat-1",
                serverMessageId=4,
                factor="Что делать",
                action="Сверить группы с ростом CPC",
                title="Сверить группы с ростом CPC",
                rawObjectText="все продвигаемые SKU",
                executionReadiness="needs_resolution",
                executionReason="Raw object text is present, but no executable targets were extracted.",
                targets=[],
                objects=[],
            ),
        )

        self.assertEqual([item.object_id for item in objects], ["все продвигаемые SKU"])

    def test_fetch_backend_task_reports_filters_to_requested_source_task_ids(self):
        task_1 = AgentActionTaskItem(
            taskId="task-1",
            actionId="task-1",
            chatId="chat-1",
            serverMessageId=4,
            factor="Что делать",
            action="Task 1",
            title="Task 1",
            targets=["46227"],
            objects=[],
        )
        task_2 = AgentActionTaskItem(
            taskId="task-2",
            actionId="task-2",
            chatId="chat-1",
            serverMessageId=4,
            factor="Что делать",
            action="Task 2",
            title="Task 2",
            targets=["46230"],
            objects=[],
        )
        execution = AgentActionExecutionItem(
            id=10,
            sourceChatId="chat-1",
            taskId="task-1:sku-46227",
            sourceTaskId="task-1",
            executionChatId="exec-chat-1",
            runId=87,
            clientMessageId="client-1",
            status="completed",
            prompt="prompt",
            task=AgentActionTaskSnapshot.model_validate(task_1.model_dump(mode="json")),
            objectId="sku-46227",
            objectLabel="SKU 46227",
            objectType="sku",
            objectExternalId="46227",
            objectPayload={"nm_id": 46227},
            resultKind="action_execution",
            decisionStatus="completed",
            reviewSummary="ok",
            answer="done",
        )
        run_detail = AgentRunDetailResponse(
            run=AgentRunTrace(
                id=87,
                chat_id="chat-1",
                server_message_id=2,
                intent="marketing",
                primary_agent="action_executor",
                status="completed",
                summary={"status": "completed"},
            ),
            rounds=[],
        )

        payloads = {
            "/v1/agent-actions": DummyResponse(
                200,
                {
                    "status": "completed",
                    "chatId": "chat-1",
                    "sourceMessageId": 4,
                    "items": [
                        task_1.model_dump(mode="json"),
                        task_2.model_dump(mode="json"),
                    ],
                    "recommendations": [],
                },
            ),
            "/v1/agent-actions/executions": DummyResponse(
                200,
                {
                    "chatId": "chat-1",
                    "items": [
                        execution.model_dump(mode="json"),
                        execution.model_copy(
                            update={"id": 11, "sourceTaskId": "other-task", "taskId": "other-task:sku-1"}
                        ).model_dump(mode="json"),
                    ],
                },
            ),
            "/v1/agent-actions/executions/10/view": DummyResponse(
                200,
                AgentActionExecutionViewItem(
                    id=10,
                    status="completed",
                    resultKind="action_execution",
                    decisionStatus="completed",
                    reviewSummary="ok",
                    objectExternalId="46227",
                    objectPayload={"nm_id": 46227},
                ).model_dump(mode="json"),
            ),
            "/v1/chat/agent-runs/87": DummyResponse(
                200,
                run_detail.model_dump(mode="json"),
            ),
        }

        def fake_request(method, url, **kwargs):
            self.assertEqual(method, "GET")
            path = url.replace("http://127.0.0.1:8000", "")
            return payloads[path]

        reports = fetch_backend_task_reports(
            request=ScenarioReportRequest(
                chat_id="chat-1",
                server_message_id=4,
                source_run_id=87,
                source_task_ids=["task-1", "task-2"],
                output_dir=Path("docs/action-scenario-reports"),
            ),
            base_url="http://127.0.0.1:8000",
            request_func=fake_request,
        )

        self.assertEqual([item.task.taskId for item in reports], ["task-1", "task-2"])
        self.assertEqual(len(reports[0].executions), 1)
        self.assertEqual(reports[0].executions[0].execution.sourceTaskId, "task-1")
        self.assertEqual(len(reports[1].executions), 0)

    def test_backend_renderer_marks_needs_resolution_without_execution(self):
        backend_report = Scenario2TaskReport(
            task=AgentActionTaskItem(
                taskId="task-5",
                actionId="task-5",
                chatId="chat-1",
                serverMessageId=4,
                factor="Что делать",
                action="Сверить группы с ростом CPC",
                title="Сверить группы с ростом CPC",
                rawObjectText="все продвигаемые SKU",
                executionReadiness="needs_resolution",
                executionReason="Raw object text is present, but no executable targets were extracted.",
                targets=[],
                objects=[],
            ),
            objects=[ScenarioObject("все продвигаемые SKU", "все продвигаемые SKU", None, None, {})],
            executions=[],
        )

        markdown = render_backend_http_markdown(
            request=ScenarioReportRequest(
                chat_id="chat-1",
                server_message_id=4,
                source_run_id=87,
                source_task_ids=["task-5"],
                output_dir=Path("docs/action-scenario-reports"),
            ),
            reports=[backend_report],
        )

        self.assertIn("`needs_resolution`", markdown)
        self.assertIn("current backend normalization marks it as `needs_resolution`", markdown)

    def test_export_writes_two_files(self):
        report = Scenario1TaskReport(
            task=_stored_task("task-1"),
            normalized_task=None,
            objects=[ScenarioObject("sku-46227", "SKU 46227", "sku", "46227", {"nm_id": 46227})],
            latest_executions=[],
            direct_facts={"objects": {}},
            verification=Scenario1VerificationVerdict(
                verification_status="no_execution",
                supports_stored_result=False,
                summary="none",
                key_evidence=[],
                contradictions=[],
                next_step="wait",
                object_reviews=[],
            ),
        )
        backend_report = Scenario2TaskReport(
            task=AgentActionTaskItem(
                taskId="task-1",
                actionId="task-1",
                chatId="chat-1",
                serverMessageId=4,
                factor="Что делать",
                action="Task 1",
                title="Task 1",
                targets=["46227"],
                objects=[],
            ),
            objects=[ScenarioObject("sku-46227", "SKU 46227", "sku", "46227", {"nm_id": 46227})],
            executions=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            request = ScenarioReportRequest(
                chat_id="chat-1",
                server_message_id=4,
                source_run_id=87,
                source_task_ids=["task-1"],
                output_dir=output_dir,
            )
            with patch(
                "agi_ecom_app.reporting.action_scenarios.build_db_agents_model_review_markdown",
                return_value=render_db_agents_model_review_markdown(
                    request=request,
                    parent_run=_parent_run_bundle(),
                    reports=[report],
                ),
            ), patch(
                "agi_ecom_app.reporting.action_scenarios.build_backend_http_markdown",
                return_value="# backend\n",
            ):
                from agi_ecom_app.reporting.action_scenarios import export_action_scenario_reports

                result = export_action_scenario_reports(request)

            self.assertTrue((output_dir / "scenario_1_db_agents_model_review.md").exists())
            self.assertTrue((output_dir / "scenario_2_backend_http_report.md").exists())
            self.assertEqual(set(result.keys()), {"scenario_1", "scenario_2"})
