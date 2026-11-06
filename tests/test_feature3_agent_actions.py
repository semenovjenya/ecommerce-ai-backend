import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.agent_actions_view_api import (
    AgentActionExecutionForecastView,
    AgentActionExecutionViewItem,
    AgentActionExecutionViewPlanStep,
)
from agi_ecom_app.forecast.models import ForecastDecisionResult
from agi_ecom_app.schemas import (
    ActionExecutionResult,
    ChatRole,
    DiagnosticFinding,
    DriverClass,
    ReportTable,
    SalesDiagnosticResult,
)
from agi_ecom_app.services.memory import (
    AgentActionExecutionRecord,
    StoredAgentActionTask,
    StoredChatMessage,
)
from agi_ecom_app.action_board.models import (
    AgentActionExecutionBatchResponse,
    AgentActionExecutionItem,
    AgentActionExecutionRequest,
    AgentActionExecutionsResponse,
    AgentActionTaskObject,
    AgentActionTaskItem,
    AgentActionTaskSnapshot,
    AgentActionTasksResponse,
)
from agi_ecom_app.action_board.service import (
    ACTION_TASKS_METADATA_KEY,
    _build_execution_prompt,
    _execution_record_to_item,
    _normalize_action_execution_metadata,
    _task_objects_for_execution,
    _sync_agent_action_execution,
    extract_action_tasks_from_agent_output,
    get_agent_action_tasks_payload,
    select_agent_action_task,
    start_agent_action_execution,
    unselect_agent_action_task,
)


def _sample_forecast_result_payload() -> dict:
    return {
        "hypothesis": "Остановить слабую кампанию 309891193.",
        "domain": "marketing",
        "baseline": {
            "name": "baseline",
            "metrics": {"ad_spend": 2013.75, "ad_orders": 0},
            "assumptions": ["Период анализа 7 дней."],
        },
        "actionScenario": {
            "name": "action",
            "metrics": {"expected_spend_saved": 2013.75, "lost_orders_risk": "low"},
            "assumptions": ["Остановить подтверждённо слабую кампанию."],
        },
        "delta": {
            "metrics": {
                "spend_saved": 2013.75,
                "orders_at_risk": 0,
                "drr_signal": "improves",
            }
        },
        "decision": "Остановить кампанию 309891193.",
        "decisionStatus": "completed",
        "confidence": "high",
        "confidencePercent": 82,
        "blockedBy": [],
        "finalRecommendation": "Остановить 309891193 и оставить контроль метрик.",
        "summary": "Экономия 2013.75 ₽ без подтверждённого риска потери заказов.",
        "evidence": {"targets": ["309891193"]},
    }


class Feature3ActionExtractionTests(unittest.TestCase):
    def test_extracts_actions_from_structured_sales_output(self):
        result = SalesDiagnosticResult(
            agent_name="Sales Agent",
            summary="Продажи удержались.",
            sales_overview=ReportTable(title="Обзор", columns=["Метрика"], rows=[]),
            sales_dynamics=ReportTable(title="Динамика", columns=["Метрика"], rows=[]),
            sales_decomposition=ReportTable(title="Декомпозиция", columns=["Метрика"], rows=[]),
            findings=[
                DiagnosticFinding(
                    factor="Лидеры ассортимента",
                    observation="46303 дал +52.3 тыс. ₽.",
                    impact="Перекрыл слабость остальных товаров.",
                    cause="Рост выручки по SKU.",
                    action="удержать: проверить наличие и цену по 46303.",
                    evidence=["46303: +52.3 тыс. ₽"],
                    confidence="high",
                    driver_class=DriverClass.PRIMARY_DRIVER,
                )
            ],
        )

        actions = extract_action_tasks_from_agent_output(result)

        self.assertEqual(actions[0]["factor"], "Лидеры ассортимента")
        self.assertEqual(actions[0]["action"], "удержать: проверить наличие и цену по 46303.")
        self.assertEqual(actions[0]["driverClass"], "primary_driver")

    def test_extracts_final_numbered_action_list(self):
        text = (
            "# Диагностика\n\n"
            "Что делать:\n"
            "1. Проверить SKU 46313 по запасу и удержанию роста.\n"
            "2. Разобрать SKU 46303 по цене, рекламе и остаткам.\n"
            "3. Рассчитать маржу SKU 46230 перед снижением цены.\n"
            "4. Сверить кампанию 777 с запасом и экономикой.\n"
            "5. Сформировать план дозаказа по SKU 46313."
        )

        actions = extract_action_tasks_from_agent_output(text)

        self.assertEqual(len(actions), 5)
        self.assertTrue(all(action["source"] == "final_numbered_action" for action in actions))
        self.assertEqual(actions[0]["action"], "Проверить SKU 46313 по запасу и удержанию роста.")

    def test_extracts_final_action_table(self):
        text = (
            "# Диагностика\n\n"
            "Что делать:\n"
            "| # | Задача | Объект | Почему |\n"
            "| --- | --- | --- | --- |\n"
            "| 1 | Проверить доступный остаток | 46230, 46306, 46225 + ещё 12 | OOS или критично низкий запас |\n"
            "| 2 | Остановить слабую рекламу | 46312 | расход есть, заказов 0 |\n"
            "| 3 | Рассчитать ценовой тест | 46230, 46303 | маржа есть, цена выше рынка |\n"
            "| 4 | Проверить карточку и корзину | 46303, 46230 | заказы и конверсия просели |\n"
            "| 5 | Удержать рекламу без расширения | 46227 | реклама рабочая |\n"
        )

        actions = extract_action_tasks_from_agent_output(text)

        self.assertEqual(len(actions), 5)
        self.assertTrue(all(action["source"] == "final_table_action" for action in actions))
        self.assertEqual(actions[0]["action"], "Проверить доступный остаток")
        self.assertEqual(actions[0]["title"], "Проверить доступный остаток")
        self.assertEqual(actions[0]["rawObjectText"], "46230, 46306, 46225 + ещё 12")
        self.assertEqual(actions[0]["observation"], "OOS или критично низкий запас")
        self.assertEqual(actions[0]["targets"], ["46230", "46306", "46225"])


class Feature3ActionPayloadTests(unittest.TestCase):
    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_returns_action_items_with_stable_ids(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=2,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                created_at=created_at,
                metadata={
                    "intent": "sales",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Верх воронки карточки",
                            "observation": "Открытия снизились.",
                            "impact": "Риск будущего падения продаж.",
                            "cause": "Недобор спроса.",
                            "action": "исправить: разобрать проблемные карточки.",
                            "evidence": ["открытия 9695 vs 12677"],
                            "confidence": "medium",
                            "driverClass": "primary_driver",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1")

        self.assertEqual(response.items[0].taskId, "action_chat-1_2_1")
        self.assertEqual(response.items[0].actionId, "action_chat-1_2_1")
        self.assertEqual(response.items[0].serverMessageId, 2)
        self.assertEqual(response.items[0].intent, "sales")
        self.assertEqual(response.items[0].action, "Разобрать проблемные карточки")
        self.assertEqual(response.items[0].title, "Разобрать проблемные карточки")
        self.assertEqual(response.items[0].actionType, "check")
        self.assertEqual(response.items[0].priority, "high")
        self.assertIn("проблемные карточки", response.items[0].targets)
        self.assertIsNotNone(response.items[0].successCriteria)
        self.assertIsNotNone(response.items[0].agentPrompt)
        self.assertEqual(response.items[0].evidence, ["открытия 9695 vs 12677"])
        self.assertFalse(response.items[0].selected)

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value={"action_chat-1_2_1"})
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    @patch("agi_ecom_app.action_board.service.list_stored_agent_action_tasks")
    def test_payload_prefers_persisted_action_tasks_table(
        self,
        mock_list_tasks,
        mock_list_messages,
        _mock_selected_ids,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        mock_list_tasks.return_value = [
            StoredAgentActionTask(
                chat_id="chat-1",
                task_id="action_chat-1_2_1",
                action_id="action_chat-1_2_1",
                assistant_message_seq=2,
                source="final_table_action",
                factor="Что делать",
                observation="полка пустая",
                impact="Задача подготовлена после диагностики.",
                cause="полка пустая",
                action="Проверить доступный остаток",
                title="Проверить доступный остаток",
                action_type="check",
                priority="high",
                targets=["46230"],
                success_criteria="Статус получен.",
                agent_prompt="Проверить доступный остаток.",
                evidence=["полка пустая"],
                confidence="high",
                driver_class="primary_driver",
                intent="sales",
                raw_action={"action": "Проверить доступный остаток"},
                created_at=created_at,
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=2)

        mock_list_messages.assert_not_called()
        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].taskId, "action_chat-1_2_1")
        self.assertEqual(response.items[0].serverMessageId, 2)
        self.assertEqual(response.items[0].targets, ["46230"])
        self.assertIsNone(response.items[0].rawObjectText)
        self.assertEqual(response.items[0].executionReadiness, "single_target_ready")
        self.assertTrue(response.items[0].selected)

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    @patch("agi_ecom_app.action_board.service.list_stored_agent_action_tasks")
    def test_payload_normalizes_legacy_persisted_final_table_task(
        self,
        mock_list_tasks,
        mock_list_messages,
        _mock_selected_ids,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        mock_list_tasks.return_value = [
            StoredAgentActionTask(
                chat_id="chat-1",
                task_id="action_chat-1_2_1",
                action_id="action_chat-1_2_1",
                assistant_message_seq=2,
                source="final_table_action",
                factor="Что делать",
                observation="цена кратно выше рынка",
                impact="Задача подготовлена после диагностики.",
                cause="цена кратно выше рынка",
                action="Проверить цену перед снижением: 46227, 46230, 46303",
                title="Проверить цену перед снижением: 46227, 46230, 46303",
                action_type="check",
                priority="high",
                targets=["46227", "46230", "46303", "цену"],
                success_criteria="Статус получен.",
                agent_prompt="Проверить цену перед снижением.",
                evidence=["цена кратно выше рынка"],
                confidence="high",
                driver_class="primary_driver",
                intent="price",
                raw_action={
                    "source": "final_table_action",
                    "factor": "Что делать",
                    "action": "Проверить цену перед снижением: 46227, 46230, 46303",
                    "targets": ["46227", "46230", "46303"],
                },
                created_at=created_at,
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=2)

        mock_list_messages.assert_not_called()
        self.assertEqual(response.items[0].title, "Проверить цену перед снижением")
        self.assertEqual(response.items[0].action, "Проверить цену перед снижением")
        self.assertEqual(response.items[0].rawObjectText, "46227, 46230, 46303")
        self.assertEqual(response.items[0].targets, ["46227", "46230", "46303"])
        self.assertEqual(response.items[0].executionReadiness, "fanout_ready")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_falls_back_to_markdown_actions(self, mock_list_messages, _mock_selected_ids):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=4,
                role=ChatRole.ASSISTANT,
                content=(
                    "Выводы:\n"
                    "- Отмены: Отмен стало меньше: 8 против 25 неделей ранее. "
                    "Влияние: Это не главный негативный драйвер периода. "
                    "Действие: удержать: продолжать мониторинг отмен."
                ),
                metadata={"intent": "sales"},
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=4)

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].taskId, "action_chat-1_4_1")
        self.assertEqual(response.items[0].source, "markdown_fallback")
        self.assertEqual(response.items[0].factor, "Отмены")
        self.assertEqual(response.items[0].action, "Продолжать мониторинг отмен")
        self.assertEqual(response.items[0].actionType, "monitor")
        self.assertIn("отмен", " ".join(response.items[0].targets))
        self.assertEqual(response.recommendations, [])

    @patch("agi_ecom_app.action_board.service.strategic_db.get_sku_identity_map")
    def test_task_objects_for_execution_enriches_numeric_sku_payload(self, mock_identity_map):
        mock_identity_map.return_value = {
            "46230": {
                "nm_id": 46230,
                "vendor_code": "LIFECODE-46230",
                "subject_name": "Патчи",
                "brand_name": "LIFECODE",
                "title": "Патчи для глаз",
            }
        }
        task = AgentActionTaskSnapshot(
            taskId="action_chat-1_2_1",
            title="Проверить готовность карточки 46230",
            targets=["46230"],
            forecastSpec={"forecast_domain": "marketing"},
            objects=[
                AgentActionTaskObject(
                    objectId="sku-46230",
                    label="SKU 46230",
                    type="sku",
                    externalId="46230",
                    payload={"nm_id": 46230},
                )
            ],
        )

        objects = _task_objects_for_execution(task)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].payload["nm_id"], 46230)
        self.assertEqual(objects[0].payload["vendor_code"], "LIFECODE-46230")
        self.assertEqual(objects[0].payload["subject_name"], "Патчи")
        self.assertEqual(objects[0].payload["domain_category"], "Маркетинг")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_normalizes_final_numbered_actions(self, mock_list_messages, _mock_selected_ids):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=5,
                role=ChatRole.ASSISTANT,
                content=(
                    "Что делать:\n"
                    "1. Проверить SKU 46313 по запасу и удержанию роста.\n"
                    "2. Разобрать SKU 46303 по цене, рекламе и остаткам.\n"
                    "3. Рассчитать маржу SKU 46230 перед снижением цены.\n"
                    "4. Сверить кампанию 777 с запасом и экономикой.\n"
                    "5. Сформировать план дозаказа по SKU 46313."
                ),
                created_at=created_at,
                metadata={"intent": "finance"},
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=5)

        self.assertEqual(len(response.items), 5)
        self.assertEqual(response.items[0].taskId, "action_chat-1_5_1")
        self.assertEqual(response.items[0].source, "final_numbered_action")
        self.assertEqual(response.items[0].action, "Проверить SKU 46313 по запасу и удержанию роста")
        self.assertEqual(response.items[0].intent, "finance")

    @patch(
        "agi_ecom_app.action_board.service.list_selected_action_task_ids",
        return_value={"action_chat-1_2_1"},
    )
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_marks_selected_items(self, mock_list_messages, _mock_selected_ids):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=2,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                created_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
                metadata={
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Наличие",
                            "action": "проверить остатки по лидерам.",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1")

        self.assertTrue(response.items[0].selected)

    @patch("agi_ecom_app.action_board.service.select_action_task")
    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_select_agent_action_task_is_idempotent_at_service_boundary(
        self,
        mock_list_messages,
        _mock_selected_ids,
        mock_select_action_task,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=2,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                created_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
                metadata={
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Наличие",
                            "action": "проверить остатки по лидерам.",
                        }
                    ],
                },
            )
        ]

        first = select_agent_action_task("chat-1", "action_chat-1_2_1")
        second = select_agent_action_task("chat-1", "action_chat-1_2_1")

        self.assertTrue(first.selected)
        self.assertTrue(second.selected)
        self.assertEqual(mock_select_action_task.call_count, 2)

    @patch("agi_ecom_app.action_board.service.unselect_action_task")
    @patch(
        "agi_ecom_app.action_board.service.list_selected_action_task_ids",
        return_value={"action_chat-1_2_1"},
    )
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_unselect_agent_action_task_returns_unselected_item(
        self,
        mock_list_messages,
        _mock_selected_ids,
        mock_unselect_action_task,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=2,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                created_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
                metadata={
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Наличие",
                            "action": "проверить остатки по лидерам.",
                        }
                    ],
                },
            )
        ]

        item = unselect_agent_action_task("chat-1", "action_chat-1_2_1")

        self.assertFalse(item.selected)
        mock_unselect_action_task.assert_called_once_with(
            chat_id="chat-1",
            task_id="action_chat-1_2_1",
        )

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages", return_value=[])
    def test_select_agent_action_task_rejects_unknown_task(
        self,
        _mock_list_messages,
        _mock_selected_ids,
    ):
        with self.assertRaises(ValueError):
            select_agent_action_task("chat-1", "missing")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_separates_markdown_recommendations_from_action_items(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=6,
                role=ChatRole.ASSISTANT,
                content=(
                    "Выводы:\n"
                    "- Наличие продажных таблиц: В sales-схеме доступны таблицы продаж. "
                    "Влияние: Можно строить отчеты. "
                    "Действие: Если хотите, следующим сообщением соберу конкретный отчёт.\n"
                    "- Последний корректный период: Последний доступный день — 2026-04-05. "
                    "Влияние: Анализ лучше строить без неполного дня. "
                    "Действие: Для точного отчёта предлагаю брать последние 7 дней.\n"
                    "- Лидеры продаж: 46303 и 46227 удерживают общий результат. "
                    "Влияние: Они могут маскировать просадку по остальному ассортименту. "
                    "Действие: Удержать наличие и видимость 46303 и 46227, отдельно проверить остальной ассортимент."
                ),
                metadata={"intent": "sales"},
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=6)

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].factor, "Лидеры продаж")
        self.assertEqual(
            response.items[0].action,
            "Удержать наличие и видимость 46303 и 46227, отдельно проверить остальной ассортимент",
        )
        self.assertEqual(response.items[0].taskId, "action_chat-1_6_3")
        self.assertEqual(len(response.recommendations), 2)
        self.assertEqual(response.recommendations[0].factor, "Наличие продажных таблиц")
        self.assertEqual(
            response.recommendations[0].recommendation,
            "Если хотите, следующим сообщением соберу конкретный отчёт.",
        )

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_normalizes_urgent_replenishment_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=7,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "supply",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Дефицит",
                            "observation": "4625 не закрывает месячный план.",
                            "impact": "Есть риск out-of-stock.",
                            "action": "Срочно дозаказать 4625: без пополнения он недоберёт спрос.",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=7)

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].action, "Дозаказать 4625: без пополнения он недоберёт спрос")
        self.assertEqual(response.items[0].priority, "critical")
        self.assertEqual(response.items[0].actionType, "create")
        self.assertEqual(response.items[0].targets, ["4625"])
        self.assertIn("Критерий готовности", response.items[0].agentPrompt or "")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_composes_terminal_execution_fields_for_availability_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=11,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "supply",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Что делать",
                            "observation": "46227 - главный источник прибыли месяца.",
                            "impact": "Потеря остатка ударит по продажам.",
                            "cause": "Риск дефицита при сохранении спроса.",
                            "action": "Удержать наличие и продажи: 46227",
                            "evidence": ["остаток ограничен", "спрос стабилен"],
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=11)

        self.assertEqual(response.items[0].actionKind, "preserve_availability_and_sales")
        self.assertEqual(
            response.items[0].decisionType,
            "availability_replenishment_marketing_guardrail",
        )
        self.assertIn("available_stock", response.items[0].requiredMetrics)
        self.assertIn("Сохранить наличие товара", response.items[0].businessGoal or "")
        self.assertIn("Не создавай новые задачи", response.items[0].agentPrompt or "")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_composes_terminal_execution_fields_for_marketing_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=12,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "marketing",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Видимость",
                            "observation": "Позиция и CTR просели по запросу.",
                            "impact": "Рекламные заказы снижаются.",
                            "action": "Проверить рекламную кампанию 46227 и поисковый запрос",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=12)

        self.assertEqual(response.items[0].actionKind, "optimize_marketing_visibility")
        self.assertEqual(response.items[0].decisionType, "marketing_visibility_guardrail")
        self.assertIn("visibility", response.items[0].requiredMetrics)
        self.assertIn("average_position", response.items[0].requiredMetrics)

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_composes_terminal_execution_fields_for_card_funnel_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=13,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "sales",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Воронка карточки",
                            "observation": "Открытия есть, корзины и заказы просели.",
                            "impact": "Карточка теряет конверсию.",
                            "action": "Разобрать воронку карточки 46227",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=13)

        self.assertEqual(response.items[0].actionKind, "diagnose_card_funnel")
        self.assertEqual(response.items[0].decisionType, "card_funnel_diagnostic")
        self.assertIn("open_card", response.items[0].requiredMetrics)
        self.assertIn("cart_to_order_conversion", response.items[0].requiredMetrics)

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_composes_terminal_execution_fields_for_price_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=14,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "price",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Цена",
                            "observation": "Цена выше медианы конкурентов.",
                            "impact": "Заказы могут проседать.",
                            "action": "Проверить снижение цены 46227",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=14)

        self.assertEqual(response.items[0].actionKind, "validate_price_positioning")
        self.assertEqual(response.items[0].decisionType, "price_margin_competitor_guardrail")
        self.assertIn("current_price", response.items[0].requiredMetrics)
        self.assertIn("median_competitor_price", response.items[0].requiredMetrics)

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_marks_multi_target_forecast_as_review_only(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=15,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "supply",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Остатки",
                            "observation": "46227 и 46303 могут сорвать план.",
                            "impact": "Есть риск потери продаж.",
                            "action": "Проверить наличие 46227 и 46303",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=15)

        self.assertEqual(response.items[0].forecastSpec["target_support"], "multiple_targets_review_only")
        self.assertIn("multiple_targets_not_supported", response.items[0].executionPlan or "")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_turns_priority_shout_into_check_task(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=8,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "supply",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Готовность плана",
                            "observation": "4627 и 4303 могут сорвать план.",
                            "impact": "Риск недовыполнения.",
                            "action": "В приоритете 4627 и 4303",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=8)

        self.assertEqual(len(response.items), 1)
        self.assertEqual(
            response.items[0].action,
            "Проверить готовность карточек 4627 и 4303 к выполнению плана",
        )
        self.assertEqual(response.items[0].actionType, "check")
        self.assertEqual(response.items[0].priority, "high")
        self.assertEqual(response.items[0].targets, ["4627", "4303"])

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_keeps_vague_attention_call_as_recommendation(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=9,
                role=ChatRole.ASSISTANT,
                content="markdown answer",
                metadata={
                    "intent": "supply",
                    ACTION_TASKS_METADATA_KEY: [
                        {
                            "source": "agent_finding",
                            "factor": "Поставки",
                            "observation": "Есть входящие поставки.",
                            "impact": "Может повлиять на остатки.",
                            "action": "Обратить внимание на поставки",
                        }
                    ],
                },
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=9)

        self.assertEqual(response.items, [])
        self.assertEqual(len(response.recommendations), 1)
        self.assertEqual(response.recommendations[0].recommendation, "Обратить внимание на поставки")

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_markdown_fallback_normalizes_urgent_action(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=10,
                role=ChatRole.ASSISTANT,
                content=(
                    "Выводы:\n"
                    "- Остатки: 4625 не закрывает план. "
                    "Влияние: Есть риск out-of-stock. "
                    "Действие: Срочно дозаказать 4625 без пополнения спрос будет недобран."
                ),
                metadata={"intent": "supply"},
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=10)

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].actionType, "create")
        self.assertEqual(response.items[0].priority, "critical")
        self.assertEqual(response.items[0].targets, ["4625"])

    def test_execution_prompt_uses_agent_prompt_when_available(self):
        prompt = _build_execution_prompt(
            AgentActionTaskSnapshot(
                taskId="action-1",
                action="Проверить остаток",
                agentPrompt="Проверить остаток SKU 46227 и вернуть статус.",
            )
        )

        self.assertEqual(prompt, "Проверить остаток SKU 46227 и вернуть статус.")

    def test_execution_prompt_falls_back_to_task_context(self):
        prompt = _build_execution_prompt(
            AgentActionTaskSnapshot(
                taskId="local-1",
                action="Проверить возможность масштабирования экономики",
                factor="Что делать",
                observation="высокая маржа",
                intent="finance",
                targets=["46227"],
                successCriteria="Получен статус по объекту.",
                evidence=["маржа высокая"],
            )
        )

        self.assertIn("Проверить возможность масштабирования экономики", prompt)
        self.assertIn("Домен: finance", prompt)
        self.assertIn("Объекты: 46227", prompt)
        self.assertIn("маржа высокая", prompt)

    @patch("agi_ecom_app.agent_queue.enqueue_action_execution_run")
    @patch("agi_ecom_app.action_board.service._sync_agent_action_execution", side_effect=lambda record: record)
    @patch("agi_ecom_app.action_board.service.update_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_chat_session", return_value="chat-exec-1")
    @patch("agi_ecom_app.action_board.service.get_settings", return_value=SimpleNamespace(openai_model="gpt-test"))
    @patch("agi_ecom_app.action_board.service._find_action_task", return_value=None)
    @patch("agi_ecom_app.action_board.service.get_agent_action_execution_for_object", return_value=None)
    def test_start_execution_accepts_local_task_snapshot_and_overrides_intent(
        self,
        _mock_existing,
        _mock_find_task,
        _mock_settings,
        _mock_create_chat,
        mock_create_execution,
        mock_update_execution,
        _mock_sync,
        mock_enqueue,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        captured = {}

        def fake_create_execution(**kwargs):
            captured.update(kwargs)
            return AgentActionExecutionRecord(
                id=7,
                source_chat_id=kwargs["source_chat_id"],
                source_task_id=kwargs["source_task_id"],
                task_id=kwargs["task_id"],
                execution_chat_id=kwargs["execution_chat_id"],
                run_id=None,
                client_message_id=kwargs["client_message_id"],
                status="queued",
                prompt=kwargs["prompt"],
                task_snapshot=kwargs["task_snapshot"],
                answer=None,
                server_message_id=None,
                error=None,
                object_id=kwargs["object_id"],
                object_label=kwargs["object_label"],
                object_type=kwargs["object_type"],
                object_external_id=kwargs["object_external_id"],
                object_payload=kwargs["object_payload"],
                execution_mode=kwargs["execution_mode"],
                result_json={},
                decision_status=None,
                review_summary=None,
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        def fake_update_execution(**kwargs):
            return AgentActionExecutionRecord(
                id=7,
                source_chat_id="chat-1",
                source_task_id="local:1",
                task_id="local:1",
                execution_chat_id="chat-exec-1",
                run_id=42,
                client_message_id=captured["client_message_id"],
                status=kwargs["status"],
                prompt=captured["prompt"],
                task_snapshot=captured["task_snapshot"],
                answer=None,
                server_message_id=None,
                error=None,
                object_id="sku-46227",
                object_label="SKU 46227",
                object_type=None,
                object_external_id="46227",
                object_payload={"nm_id": 46227},
                execution_mode="terminal_action",
                result_json={},
                decision_status=None,
                review_summary=kwargs.get("review_summary"),
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        mock_create_execution.side_effect = fake_create_execution
        mock_update_execution.side_effect = fake_update_execution
        mock_enqueue.return_value = SimpleNamespace(
            status="queued",
            run_id=42,
            answer=None,
            server_message_id=None,
        )

        result = start_agent_action_execution(
            AgentActionExecutionRequest(
                chat_id="chat-1",
                task_id="local:1",
                task=AgentActionTaskSnapshot(
                    taskId="local:1",
                    action="Проверить возможность масштабирования экономики",
                    intent="finance",
                    targets=["46227"],
                    objects=[
                        AgentActionTaskObject(
                            objectId="sku-46227",
                            label="SKU 46227",
                            type="sku",
                            externalId="46227",
                            payload={"nm_id": 46227},
                        )
                    ],
                ),
            )
        )

        self.assertEqual(result.chatId, "chat-1")
        self.assertEqual(result.sourceTaskId, "local:1")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].executionChatId, "chat-exec-1")
        self.assertEqual(result.items[0].sourceTaskId, "local:1")
        self.assertEqual(result.items[0].objectId, "sku-46227")
        self.assertIn("Проверить возможность масштабирования экономики", captured["prompt"])
        self.assertEqual(captured["execution_mode"], "terminal_action")
        self.assertEqual(mock_enqueue.call_args.kwargs["intent_override"].value, "finance")
        self.assertEqual(mock_enqueue.call_args.kwargs["source_chat_id"], "chat-1")
        self.assertEqual(mock_enqueue.call_args.kwargs["source_execution_id"], 7)

    @patch("agi_ecom_app.agent_queue.enqueue_action_execution_run")
    @patch("agi_ecom_app.action_board.service._sync_agent_action_execution", side_effect=lambda record: record)
    @patch("agi_ecom_app.action_board.service.update_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_chat_session", return_value="chat-exec-1")
    @patch("agi_ecom_app.action_board.service.get_settings", return_value=SimpleNamespace(openai_model="gpt-test"))
    @patch("agi_ecom_app.action_board.service._find_action_task", return_value=None)
    @patch("agi_ecom_app.action_board.service.get_agent_action_execution_for_object", return_value=None)
    def test_start_execution_accepts_local_task_with_targets_only(
        self,
        _mock_existing,
        _mock_find_task,
        _mock_settings,
        _mock_create_chat,
        mock_create_execution,
        mock_update_execution,
        _mock_sync,
        mock_enqueue,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)

        def fake_create_execution(**kwargs):
            return AgentActionExecutionRecord(
                id=8,
                source_chat_id=kwargs["source_chat_id"],
                source_task_id=kwargs["source_task_id"],
                task_id=kwargs["task_id"],
                execution_chat_id=kwargs["execution_chat_id"],
                run_id=None,
                client_message_id=kwargs["client_message_id"],
                status="queued",
                prompt=kwargs["prompt"],
                task_snapshot=kwargs["task_snapshot"],
                answer=None,
                server_message_id=None,
                error=None,
                object_id=kwargs["object_id"],
                object_label=kwargs["object_label"],
                object_type=kwargs["object_type"],
                object_external_id=kwargs["object_external_id"],
                object_payload=kwargs["object_payload"],
                execution_mode=kwargs["execution_mode"],
                result_json={},
                decision_status=None,
                review_summary=None,
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        def fake_update_execution(**kwargs):
            return AgentActionExecutionRecord(
                id=8,
                source_chat_id="chat-1",
                source_task_id="local:targets-only",
                task_id="local:targets-only",
                execution_chat_id="chat-exec-1",
                run_id=42,
                client_message_id="client-8",
                status=kwargs["status"],
                prompt="prompt",
                task_snapshot={"taskId": "local:targets-only", "targets": ["46227"]},
                answer=None,
                server_message_id=None,
                error=None,
                object_id="local:targets-only",
                object_label="Проверить цену перед снижением",
                object_type=None,
                object_external_id="46227",
                object_payload={},
                execution_mode="terminal_action",
                result_json={},
                decision_status=None,
                review_summary=kwargs.get("review_summary"),
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        mock_create_execution.side_effect = fake_create_execution
        mock_update_execution.side_effect = fake_update_execution
        mock_enqueue.return_value = SimpleNamespace(
            status="queued",
            run_id=42,
            answer=None,
            server_message_id=None,
        )

        result = start_agent_action_execution(
            AgentActionExecutionRequest(
                chat_id="chat-1",
                task_id="local:targets-only",
                task=AgentActionTaskSnapshot(
                    taskId="local:targets-only",
                    title="Проверить цену перед снижением",
                    targets=["46227"],
                ),
            )
        )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].objectExternalId, "46227")
        self.assertEqual(mock_create_execution.call_count, 1)
        self.assertEqual(mock_enqueue.call_count, 1)

    @patch("agi_ecom_app.agent_queue.enqueue_action_execution_run")
    @patch("agi_ecom_app.action_board.service._sync_agent_action_execution", side_effect=lambda record: record)
    @patch("agi_ecom_app.action_board.service.update_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_chat_session", return_value="chat-exec-1")
    @patch("agi_ecom_app.action_board.service.get_settings", return_value=SimpleNamespace(openai_model="gpt-test"))
    @patch("agi_ecom_app.action_board.service._find_action_task", return_value=None)
    @patch("agi_ecom_app.action_board.service.get_agent_action_execution_for_object", return_value=None)
    def test_start_execution_splits_task_objects_into_batch_items(
        self,
        _mock_existing,
        _mock_find_task,
        _mock_settings,
        _mock_create_chat,
        mock_create_execution,
        mock_update_execution,
        _mock_sync,
        mock_enqueue,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        captured_task_ids: list[str] = []
        captured_object_ids: list[str] = []

        def fake_create_execution(**kwargs):
            captured_task_ids.append(kwargs["task_id"])
            captured_object_ids.append(kwargs["object_id"])
            execution_id = 100 + len(captured_task_ids)
            return AgentActionExecutionRecord(
                id=execution_id,
                source_chat_id=kwargs["source_chat_id"],
                source_task_id=kwargs["source_task_id"],
                task_id=kwargs["task_id"],
                execution_chat_id=kwargs["execution_chat_id"],
                run_id=None,
                client_message_id=kwargs["client_message_id"],
                status="queued",
                prompt=kwargs["prompt"],
                task_snapshot=kwargs["task_snapshot"],
                answer=None,
                server_message_id=None,
                error=None,
                object_id=kwargs["object_id"],
                object_label=kwargs["object_label"],
                object_type=kwargs["object_type"],
                object_external_id=kwargs["object_external_id"],
                object_payload=kwargs["object_payload"],
                execution_mode=kwargs["execution_mode"],
                result_json={},
                decision_status=None,
                review_summary=None,
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        def fake_update_execution(**kwargs):
            execution_id = kwargs["execution_id"]
            index = execution_id - 101
            return AgentActionExecutionRecord(
                id=execution_id,
                source_chat_id="chat-1",
                source_task_id="local:multi",
                task_id=captured_task_ids[index],
                execution_chat_id="chat-exec-1",
                run_id=1000 + execution_id,
                client_message_id=f"client-{execution_id}",
                status=kwargs["status"],
                prompt="prompt",
                task_snapshot={
                    "taskId": captured_task_ids[index],
                    "targets": [str(46227 + index)],
                    "objects": [
                        {
                            "objectId": captured_object_ids[index],
                            "externalId": str(46227 + index),
                            "payload": {"nm_id": 46227 + index},
                        }
                    ],
                },
                answer=None,
                server_message_id=None,
                error=None,
                object_id=captured_object_ids[index],
                object_label=f"SKU {46227 + index}",
                object_type="sku",
                object_external_id=str(46227 + index),
                object_payload={"nm_id": 46227 + index},
                execution_mode="terminal_action",
                result_json={},
                decision_status=None,
                review_summary=kwargs.get("review_summary"),
                created_at=created_at,
                updated_at=created_at,
                completed_at=None,
            )

        mock_create_execution.side_effect = fake_create_execution
        mock_update_execution.side_effect = fake_update_execution
        mock_enqueue.return_value = SimpleNamespace(
            status="queued",
            run_id=42,
            answer=None,
            server_message_id=None,
        )

        result = start_agent_action_execution(
            AgentActionExecutionRequest(
                chat_id="chat-1",
                task_id="local:multi",
                task=AgentActionTaskSnapshot(
                    taskId="local:multi",
                    action="Проверить наличие по SKU",
                    intent="supply",
                    objects=[
                        AgentActionTaskObject(
                            objectId="sku-46227",
                            label="SKU 46227",
                            type="sku",
                            externalId="46227",
                            payload={"nm_id": 46227},
                        ),
                        AgentActionTaskObject(
                            objectId="sku-46228",
                            label="SKU 46228",
                            type="sku",
                            externalId="46228",
                            payload={"nm_id": 46228},
                        ),
                    ],
                ),
            )
        )

        self.assertEqual(result.chatId, "chat-1")
        self.assertEqual(result.sourceTaskId, "local:multi")
        self.assertEqual(len(result.items), 2)
        self.assertEqual(captured_object_ids, ["sku-46227", "sku-46228"])
        self.assertEqual(captured_task_ids, ["local:multi:sku-46227", "local:multi:sku-46228"])
        self.assertEqual(result.items[0].objectExternalId, "46227")
        self.assertEqual(result.items[1].objectExternalId, "46228")
        self.assertEqual(mock_create_execution.call_count, 2)
        self.assertEqual(mock_enqueue.call_count, 2)

    @patch("agi_ecom_app.agent_queue.enqueue_action_execution_run")
    @patch("agi_ecom_app.action_board.service.create_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.create_chat_session")
    @patch("agi_ecom_app.action_board.service.get_settings", return_value=SimpleNamespace(openai_model="gpt-test"))
    @patch("agi_ecom_app.action_board.service._find_action_task", return_value=None)
    def test_start_execution_skips_placeholder_for_raw_object_text_without_targets(
        self,
        _mock_find_task,
        _mock_settings,
        mock_create_chat,
        mock_create_execution,
        mock_enqueue,
    ):
        result = start_agent_action_execution(
            AgentActionExecutionRequest(
                chat_id="chat-1",
                task_id="local:aggregate",
                task=AgentActionTaskSnapshot(
                    taskId="local:aggregate",
                    title="Сверить группы с ростом CPC",
                    rawObjectText="все продвигаемые SKU",
                    targets=[],
                    objects=[],
                ),
            )
        )

        self.assertEqual(result.chatId, "chat-1")
        self.assertEqual(result.sourceTaskId, "local:aggregate")
        self.assertEqual(result.items, [])
        mock_create_chat.assert_not_called()
        mock_create_execution.assert_not_called()
        mock_enqueue.assert_not_called()

    @patch("agi_ecom_app.action_board.service.list_selected_action_task_ids", return_value=set())
    @patch("agi_ecom_app.action_board.service.list_all_stored_messages")
    def test_payload_marks_aggregate_object_text_as_needs_resolution(
        self,
        mock_list_messages,
        _mock_selected_ids,
    ):
        mock_list_messages.return_value = [
            StoredChatMessage(
                seq=17,
                role=ChatRole.ASSISTANT,
                content=(
                    "Что делать:\n"
                    "| # | Задача | Объект | Почему |\n"
                    "| --- | --- | --- | --- |\n"
                    "| 1 | Сверить группы с ростом CPC | все продвигаемые SKU | CPC вырос во всех сравнениях |\n"
                ),
                metadata={"intent": "marketing"},
            )
        ]

        response = get_agent_action_tasks_payload("chat-1", server_message_id=17)

        self.assertEqual(response.items[0].title, "Сверить группы с ростом CPC")
        self.assertEqual(response.items[0].rawObjectText, "все продвигаемые SKU")
        self.assertEqual(response.items[0].targets, [])
        self.assertEqual(response.items[0].objects, [])
        self.assertEqual(response.items[0].executionReadiness, "needs_resolution")

    def test_supply_guardrail_wins_for_marketing_expansion_with_stock_risk(self):
        result = _normalize_action_execution_metadata(
            title="Сверить доступность перед расширением рекламы: 46310, 46313, 46227",
            action_type="check",
            targets=["46310", "46313", "46227"],
            raw_action={
                "factor": "Наличие перед ростом трафика",
                "observation": "Есть риск ограничений по остатку у части SKU.",
                "impact": "Расширение рекламы без проверки может привести к дефициту.",
                "evidence": [],
            },
            intent="marketing",
        )

        self.assertEqual(result["actionKind"], "preserve_availability_and_sales")
        self.assertEqual(result["forecastSpec"]["forecast_domain"], "supply")

    @patch("agi_ecom_app.action_board.service.update_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.get_agent_run_trace_by_id")
    def test_sync_execution_keeps_handler_debug_payload_from_trace(
        self,
        mock_get_trace,
        mock_update_execution,
    ):
        record = AgentActionExecutionRecord(
            id=7,
            source_chat_id="chat-1",
            task_id="action_chat-1_2_1",
            execution_chat_id="chat-exec-1",
            run_id=42,
            client_message_id="client-1",
            status="queued",
            execution_mode="terminal_action",
            prompt="Проверить остаток.",
            task_snapshot={"taskId": "action_chat-1_2_1"},
            result_json={},
            decision_status=None,
            review_summary=None,
            answer=None,
            server_message_id=None,
            error=None,
            created_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc),
            completed_at=None,
        )
        trace_result = {
            "businessMeaning": "SKU критичен для прибыли.",
            "checks": [],
            "forecast": {"stockoutRisk": "high", "reorderQty": 84},
            "decision": "Нужен дозаказ.",
            "decisionStatus": "completed",
            "confidence": "high",
            "blockedBy": [],
            "finalRecommendation": "Подтвердить дозаказ.",
            "summary": "Дозаказ 84 шт.; рекламу не расширять.",
            "handler": "preserve_availability_and_sales",
            "data_version": "v1",
            "evidence_package": {"metrics": {"reorder_qty": 84}},
        }
        mock_get_trace.return_value = SimpleNamespace(
            status="completed",
            assistant_message_seq=None,
            summary={
                "action_execution_result": trace_result,
                "decision_status": "completed",
                "review_summary": "Дозаказ 84 шт.; рекламу не расширять.",
            },
            chat_id="chat-exec-1",
        )

        def fake_update_execution(**kwargs):
            return AgentActionExecutionRecord(
                id=record.id,
                source_chat_id=record.source_chat_id,
                task_id=record.task_id,
                execution_chat_id=record.execution_chat_id,
                run_id=record.run_id,
                client_message_id=record.client_message_id,
                status=kwargs["status"],
                execution_mode=record.execution_mode,
                prompt=record.prompt,
                task_snapshot=record.task_snapshot,
                result_json=kwargs["result_json"],
                decision_status=kwargs["decision_status"],
                review_summary=kwargs["review_summary"],
                answer=record.answer,
                server_message_id=record.server_message_id,
                error=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
                completed_at=record.completed_at,
            )

        mock_update_execution.side_effect = fake_update_execution

        updated = _sync_agent_action_execution(record)

        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.result_json["handler"], "preserve_availability_and_sales")
        self.assertEqual(updated.result_json["data_version"], "v1")
        self.assertEqual(updated.result_json["evidence_package"]["metrics"]["reorder_qty"], 84)
        self.assertEqual(updated.decision_status, "completed")
        self.assertEqual(updated.review_summary, "Дозаказ 84 шт.; рекламу не расширять.")

    @patch("agi_ecom_app.action_board.service.update_agent_action_execution")
    @patch("agi_ecom_app.action_board.service.get_agent_run_trace_by_id")
    def test_sync_execution_prefers_forecast_result_from_trace(
        self,
        mock_get_trace,
        mock_update_execution,
    ):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        record = AgentActionExecutionRecord(
            id=7,
            source_chat_id="chat-1",
            task_id="action_chat-1_2_1",
            execution_chat_id="chat-exec-1",
            run_id=42,
            client_message_id="client-1",
            status="queued",
            execution_mode="terminal_action",
            prompt="Остановить слабые кампании.",
            task_snapshot={"taskId": "action_chat-1_2_1"},
            result_json={},
            decision_status=None,
            review_summary=None,
            answer=None,
            server_message_id=None,
            error=None,
            created_at=created_at,
            updated_at=created_at,
            completed_at=None,
        )
        forecast_payload = _sample_forecast_result_payload()
        mock_get_trace.return_value = SimpleNamespace(
            status="completed",
            assistant_message_seq=None,
            summary={
                "forecast_decision_result": forecast_payload,
                "action_execution_result": {"summary": "legacy fallback"},
                "decision_status": "completed",
                "review_summary": forecast_payload["summary"],
            },
            chat_id="chat-exec-1",
        )

        def fake_update_execution(**kwargs):
            return AgentActionExecutionRecord(
                id=record.id,
                source_chat_id=record.source_chat_id,
                task_id=record.task_id,
                execution_chat_id=record.execution_chat_id,
                run_id=record.run_id,
                client_message_id=record.client_message_id,
                status=kwargs["status"],
                execution_mode=record.execution_mode,
                prompt=record.prompt,
                task_snapshot=record.task_snapshot,
                result_json=kwargs["result_json"],
                decision_status=kwargs["decision_status"],
                review_summary=kwargs["review_summary"],
                answer=record.answer,
                server_message_id=record.server_message_id,
                error=record.error,
                created_at=record.created_at,
                updated_at=record.updated_at,
                completed_at=record.completed_at,
            )

        mock_update_execution.side_effect = fake_update_execution

        updated = _sync_agent_action_execution(record)

        self.assertEqual(updated.result_json["hypothesis"], forecast_payload["hypothesis"])
        self.assertEqual(updated.result_json["domain"], "marketing")
        self.assertNotIn("businessMeaning", updated.result_json)
        self.assertEqual(updated.decision_status, "completed")
        self.assertEqual(updated.review_summary, forecast_payload["summary"])

    def test_execution_record_to_item_exposes_forecast_result(self):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        record = AgentActionExecutionRecord(
            id=7,
            source_chat_id="chat-1",
            task_id="action_chat-1_2_1",
            execution_chat_id="chat-exec-1",
            run_id=42,
            client_message_id="client-1",
            status="completed",
            execution_mode="terminal_action",
            prompt="Остановить слабые кампании.",
            task_snapshot={"taskId": "action_chat-1_2_1"},
            result_json=_sample_forecast_result_payload(),
            decision_status="completed",
            review_summary="Экономия 2013.75 ₽.",
            answer="# Прогноз эффекта",
            server_message_id=12,
            error=None,
            created_at=created_at,
            updated_at=created_at,
            completed_at=created_at,
        )

        item = _execution_record_to_item(record)

        self.assertEqual(item.resultKind, "forecast_decision")
        self.assertIsNone(item.result)
        self.assertIsNotNone(item.forecastResult)
        self.assertEqual(item.forecastResult.domain, "marketing")
        self.assertEqual(item.forecastResult.confidencePercent, 82)
        self.assertEqual(item.forecastResult.delta.metrics["spend_saved"], 2013.75)

    def test_execution_record_to_item_keeps_legacy_action_result(self):
        created_at = datetime(2026, 4, 7, 10, 0, tzinfo=timezone.utc)
        record = AgentActionExecutionRecord(
            id=7,
            source_chat_id="chat-1",
            task_id="action_chat-1_2_1",
            execution_chat_id="chat-exec-1",
            run_id=42,
            client_message_id="client-1",
            status="completed",
            execution_mode="terminal_action",
            prompt="Проверить остаток.",
            task_snapshot={"taskId": "action_chat-1_2_1"},
            result_json={
                "businessMeaning": "SKU критичен для прибыли.",
                "checks": [],
                "decision": "Нужен дозаказ.",
                "decisionStatus": "completed",
                "confidence": "high",
                "blockedBy": [],
                "finalRecommendation": "Подтвердить дозаказ.",
                "summary": "Нужен дозаказ.",
            },
            decision_status="completed",
            review_summary="Нужен дозаказ.",
            answer="Готово",
            server_message_id=12,
            error=None,
            created_at=created_at,
            updated_at=created_at,
            completed_at=created_at,
        )

        item = _execution_record_to_item(record)

        self.assertEqual(item.resultKind, "action_execution")
        self.assertIsNotNone(item.result)
        self.assertIsNone(item.forecastResult)
        self.assertEqual(item.result.summary, "Нужен дозаказ.")


class Feature3ActionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.action_board.router.get_agent_action_tasks_payload")
    def test_agent_actions_endpoint_returns_json_payload(self, mock_get_payload):
        mock_get_payload.return_value = AgentActionTasksResponse(
            chatId="chat-1",
            sourceMessageId=2,
            items=[
                AgentActionTaskItem(
                    taskId="action_chat-1_2_1",
                    actionId="action_chat-1_2_1",
                    chatId="chat-1",
                    serverMessageId=2,
                    factor="Лидеры ассортимента",
                    action="удержать: проверить наличие.",
                )
            ],
        )

        response = self.client.get(
            "/v1/agent-actions",
            params={"chat_id": "chat-1", "server_message_id": 2},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chatId"], "chat-1")
        self.assertEqual(response.json()["sourceMessageId"], 2)
        self.assertEqual(response.json()["items"][0]["taskId"], "action_chat-1_2_1")
        self.assertEqual(response.json()["items"][0]["action"], "удержать: проверить наличие.")
        mock_get_payload.assert_called_once_with(chat_id="chat-1", server_message_id=2)

    @patch("agi_ecom_app.action_board.router.select_agent_action_task")
    def test_select_agent_action_endpoint_returns_selected_item(self, mock_select):
        mock_select.return_value = AgentActionTaskItem(
            taskId="action_chat-1_2_1",
            actionId="action_chat-1_2_1",
            chatId="chat-1",
            serverMessageId=2,
            factor="Лидеры ассортимента",
            action="удержать: проверить наличие.",
            selected=True,
        )

        response = self.client.post(
            "/v1/agent-actions/selected",
            json={"chat_id": "chat-1", "task_id": "action_chat-1_2_1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["selected"])
        self.assertEqual(response.json()["taskId"], "action_chat-1_2_1")
        mock_select.assert_called_once_with(chat_id="chat-1", task_id="action_chat-1_2_1")

    @patch("agi_ecom_app.action_board.router.unselect_agent_action_task")
    def test_unselect_agent_action_endpoint_returns_unselected_item(self, mock_unselect):
        mock_unselect.return_value = AgentActionTaskItem(
            taskId="action_chat-1_2_1",
            actionId="action_chat-1_2_1",
            chatId="chat-1",
            serverMessageId=2,
            factor="Лидеры ассортимента",
            action="удержать: проверить наличие.",
            selected=False,
        )

        response = self.client.delete(
            "/v1/agent-actions/selected",
            params={"chat_id": "chat-1", "task_id": "action_chat-1_2_1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["selected"])
        self.assertEqual(response.json()["taskId"], "action_chat-1_2_1")
        mock_unselect.assert_called_once_with(chat_id="chat-1", task_id="action_chat-1_2_1")

    @patch(
        "agi_ecom_app.action_board.router.select_agent_action_task",
        side_effect=ValueError("task_id does not match an agent action for this chat"),
    )
    def test_select_agent_action_endpoint_rejects_unknown_task(self, _mock_select):
        response = self.client.post(
            "/v1/agent-actions/selected",
            json={"chat_id": "chat-1", "task_id": "missing"},
        )

        self.assertEqual(response.status_code, 422)

    @patch("agi_ecom_app.action_board.router.start_agent_action_execution")
    def test_start_agent_action_execution_endpoint_returns_queued_execution(self, mock_start):
        mock_start.return_value = AgentActionExecutionBatchResponse(
            chatId="chat-1",
            sourceTaskId="action_chat-1_2_1",
            items=[
                AgentActionExecutionItem(
                    id=7,
                    sourceChatId="chat-1",
                    taskId="action_chat-1_2_1:46230",
                    sourceTaskId="action_chat-1_2_1",
                    executionChatId="chat-exec-1",
                    runId=12,
                    clientMessageId="client-exec-1",
                    status="queued",
                    prompt="Проверить остаток.",
                    task=AgentActionTaskSnapshot(
                        taskId="action_chat-1_2_1:46230",
                        chatId="chat-1",
                        action="Проверить остаток.",
                    ),
                    objectId="46230",
                    objectLabel="SKU 46230",
                    objectType="sku",
                    objectExternalId="46230",
                    objectPayload={"nm_id": 46230},
                )
            ],
        )

        response = self.client.post(
            "/v1/agent-actions/executions",
            json={"chat_id": "chat-1", "task_id": "action_chat-1_2_1"},
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["chatId"], "chat-1")
        self.assertEqual(response.json()["sourceTaskId"], "action_chat-1_2_1")
        self.assertEqual(response.json()["items"][0]["id"], 7)
        self.assertEqual(response.json()["items"][0]["executionChatId"], "chat-exec-1")

    @patch("agi_ecom_app.action_board.router.list_agent_action_executions_payload")
    def test_agent_action_executions_endpoint_returns_items(self, mock_list):
        mock_list.return_value = AgentActionExecutionsResponse(
            chatId="chat-1",
            items=[
                AgentActionExecutionItem(
                    id=7,
                    sourceChatId="chat-1",
                    taskId="action_chat-1_2_1:46230",
                    sourceTaskId="action_chat-1_2_1",
                    executionChatId="chat-exec-1",
                    runId=12,
                    clientMessageId="client-exec-1",
                    status="running",
                    prompt="Проверить остаток.",
                    task=AgentActionTaskSnapshot(taskId="action_chat-1_2_1:46230"),
                    objectId="46230",
                    objectLabel="SKU 46230",
                    objectType="sku",
                    objectExternalId="46230",
                    objectPayload={"nm_id": 46230},
                )
            ],
        )

        response = self.client.get("/v1/agent-actions/executions", params={"chat_id": "chat-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["status"], "running")
        mock_list.assert_called_once_with("chat-1")

    @patch("agi_ecom_app.action_board.router.get_agent_action_execution_payload")
    def test_agent_action_execution_status_endpoint_returns_synced_item(self, mock_get):
        mock_get.return_value = AgentActionExecutionItem(
            id=7,
            sourceChatId="chat-1",
            taskId="action_chat-1_2_1:46230",
            sourceTaskId="action_chat-1_2_1",
            executionChatId="chat-exec-1",
            runId=12,
            clientMessageId="client-exec-1",
            status="completed",
            prompt="Проверить остаток.",
            task=AgentActionTaskSnapshot(taskId="action_chat-1_2_1:46230"),
            objectId="46230",
            objectLabel="SKU 46230",
            objectType="sku",
            objectExternalId="46230",
            objectPayload={"nm_id": 46230},
            result=ActionExecutionResult(
                businessMeaning="SKU критичен для прибыли.",
                checks=[],
                decision="Нужен дозаказ.",
                decisionStatus="completed",
                confidence="high",
                blockedBy=[],
                finalRecommendation="Подтвердить дозаказ сегодня.",
                summary="Нужен дозаказ.",
            ),
            decisionStatus="completed",
            reviewSummary="Нужен дозаказ.",
            answer="Готово",
        )

        response = self.client.get("/v1/agent-actions/executions/7")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Готово")
        self.assertEqual(response.json()["decisionStatus"], "completed")
        self.assertEqual(response.json()["reviewSummary"], "Нужен дозаказ.")
        mock_get.assert_called_once_with(7)

    @patch("agi_ecom_app.action_board.router.get_agent_action_execution_payload")
    def test_agent_action_execution_status_endpoint_returns_forecast_result(self, mock_get):
        forecast_payload = _sample_forecast_result_payload()
        mock_get.return_value = AgentActionExecutionItem(
            id=7,
            sourceChatId="chat-1",
            taskId="action_chat-1_2_1:309891193",
            sourceTaskId="action_chat-1_2_1",
            executionChatId="chat-exec-1",
            runId=12,
            clientMessageId="client-exec-1",
            status="completed",
            prompt="Остановить слабые кампании.",
            task=AgentActionTaskSnapshot(taskId="action_chat-1_2_1:309891193"),
            objectId="309891193",
            objectLabel="Campaign 309891193",
            objectType="campaign",
            objectExternalId="309891193",
            objectPayload={"campaign_id": 309891193},
            resultKind="forecast_decision",
            forecastResult=ForecastDecisionResult.model_validate(forecast_payload),
            decisionStatus="completed",
            reviewSummary=forecast_payload["summary"],
            answer="# Прогноз эффекта",
        )

        response = self.client.get("/v1/agent-actions/executions/7")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resultKind"], "forecast_decision")
        self.assertIsNone(payload["result"])
        self.assertEqual(payload["forecastResult"]["domain"], "marketing")
        self.assertEqual(payload["forecastResult"]["delta"]["metrics"]["spend_saved"], 2013.75)
        self.assertEqual(payload["decisionStatus"], "completed")
        self.assertEqual(payload["answer"], "# Прогноз эффекта")
        mock_get.assert_called_once_with(7)

    @patch("agi_ecom_app.agent_actions_view_api.get_agent_action_execution_payload")
    def test_agent_action_execution_view_endpoint_returns_compact_forecast_shape(self, mock_get):
        mock_get.return_value = AgentActionExecutionItem(
            id=69,
            sourceChatId="chat-1",
            taskId="task-69",
            sourceTaskId="source-task-69",
            executionChatId="exec-chat-69",
            runId=117,
            clientMessageId="client-69",
            status="completed",
            prompt="Проверить кампании с ростом расхода.",
            task=AgentActionTaskSnapshot(taskId="task-69"),
            objectId="sku-46227",
            objectLabel="SKU 46227",
            objectType="sku",
            objectExternalId="46227",
            objectPayload={
                "nm_id": 46227,
                "vendor_code": "46227",
                "subject_name": "Сыворотки",
                "domain_category": "Маркетинг",
            },
            resultKind="forecast_decision",
            forecastResult=ForecastDecisionResult.model_validate(
                {
                    **_sample_forecast_result_payload(),
                    "profitForecastRub": None,
                    "economicEffectRub": 52.49,
                    "economicEffectPctOfCurrentSpend": 39.45,
                    "reviewBlocked": False,
                    "plan": {
                        "primary_domain": "marketing",
                        "action_family": "normquery_bid_optimization",
                        "target_mode": "sku",
                        "secondary_domains": [],
                        "horizon_days": 14,
                        "period_days": 7,
                        "effect_metrics": [],
                        "goal": None,
                        "source_action_kind": "optimize_marketing_visibility",
                    },
                    "resolutionPlan": [
                        {
                            "stepId": "s1",
                            "area": "traffic",
                            "operation": "отключить_запрос",
                            "instruction": "В кампании 16255557 отключить запрос `сыворотка с лифтинг эффектом`.",
                        },
                        {
                            "stepId": "s2",
                            "area": "guardrail",
                            "operation": "зафиксировать_ставки_и_ключи",
                            "instruction": "В кампании 16255557 на 7 дней оставить ставки без изменений.",
                        },
                    ],
                    "forecastDetails": {"publishedWithTextFallback": True},
                }
            ),
            decisionStatus="completed",
            reviewSummary="Отключить нецелевые запросы и не расширять кампанию.",
        )

        response = self.client.get("/v1/agent-actions/executions/69/view")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], 69)
        self.assertEqual(payload["resultKind"], "forecast_decision")
        self.assertEqual(payload["objectExternalId"], "46227")
        self.assertNotIn("answer", payload)
        self.assertNotIn("task", payload)
        self.assertEqual(payload["forecastResult"]["planFamily"], "normquery_bid_optimization")
        self.assertEqual(payload["forecastResult"]["economicEffectRub"], 52.49)
        self.assertEqual(payload["forecastResult"]["economicEffectPctOfCurrentSpend"], 39.45)
        self.assertEqual(
            payload["forecastResult"]["resolutionPlan"][0]["operation"],
            "отключить_запрос",
        )
        mock_get.assert_called_once_with(69)

    @patch("agi_ecom_app.action_board.router.complete_agent_action_execution")
    def test_agent_action_execution_done_endpoint_marks_done(self, mock_done):
        mock_done.return_value = AgentActionExecutionItem(
            id=7,
            sourceChatId="chat-1",
            taskId="action_chat-1_2_1:46230",
            sourceTaskId="action_chat-1_2_1",
            executionChatId="chat-exec-1",
            runId=12,
            clientMessageId="client-exec-1",
            status="done",
            prompt="Проверить остаток.",
            task=AgentActionTaskSnapshot(taskId="action_chat-1_2_1:46230"),
            objectId="46230",
            objectLabel="SKU 46230",
            objectType="sku",
            objectExternalId="46230",
            objectPayload={"nm_id": 46230},
        )

        response = self.client.post("/v1/agent-actions/executions/7/done")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "done")
        mock_done.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
