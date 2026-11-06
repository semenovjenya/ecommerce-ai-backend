import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.runner import DuplicateClientMessageError
from agi_ecom_app.schemas import QueryIntent


class WebSocketChatTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.websocket_chat.ensure_chat_session")
    def test_websocket_connect_returns_ready_event(self, mock_ensure_chat_session):
        with self.client.websocket_connect("/v1/chat/ws?chat_id=chat-ws-1") as websocket:
            payload = websocket.receive_json()

        self.assertEqual(
            payload,
            {
                "type": "connection.ready",
                "chat_id": "chat-ws-1",
            },
        )
        mock_ensure_chat_session.assert_called_once()

    @patch("agi_ecom_app.websocket_chat.process_chat_message")
    @patch("agi_ecom_app.websocket_chat.ensure_chat_session")
    def test_websocket_message_flow_returns_accepted_then_completed(
        self,
        _mock_ensure_chat_session,
        mock_process_chat_message,
    ):
        mock_process_chat_message.return_value = type(
            "Response",
            (),
            {
                "server_message_id": 9,
                "answer": "Готово",
                "intent": "finance",
            },
        )()

        with self.client.websocket_connect("/v1/chat/ws?chat_id=chat-ws-2") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "message.send",
                    "client_message_id": "msg-1",
                    "chat_id": "chat-ws-2",
                    "text": "Какие продажи?",
                }
            )

            accepted = websocket.receive_json()
            completed = websocket.receive_json()

        self.assertEqual(
            accepted,
            {
                "type": "message.accepted",
                "client_message_id": "msg-1",
            },
        )
        self.assertEqual(
            completed,
            {
                "type": "message.completed",
                "client_message_id": "msg-1",
                "server_message_id": 9,
                "answer": "Готово",
                "intent": "finance",
            },
        )
        mock_process_chat_message.assert_called_once_with("chat-ws-2", "Какие продажи?", "msg-1")

    @patch("agi_ecom_app.websocket_chat.ensure_chat_session")
    def test_websocket_empty_message_returns_validation_error(self, _mock_ensure_chat_session):
        with self.client.websocket_connect("/v1/chat/ws?chat_id=chat-ws-3") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "message.send",
                    "client_message_id": "msg-2",
                    "chat_id": "chat-ws-3",
                    "text": "   ",
                }
            )

            error_payload = websocket.receive_json()

        self.assertEqual(error_payload["type"], "message.error")
        self.assertEqual(error_payload["code"], "validation_error")

    @patch("agi_ecom_app.websocket_chat.process_chat_message")
    @patch("agi_ecom_app.websocket_chat.ensure_chat_session")
    def test_websocket_duplicate_message_can_return_completed_from_cached_result(
        self,
        _mock_ensure_chat_session,
        mock_process_chat_message,
    ):
        mock_process_chat_message.side_effect = DuplicateClientMessageError(
            client_message_id="msg-3",
            answer="Уже готово",
            intent=QueryIntent.FINANCE,
            assistant_message_seq=11,
        )

        with self.client.websocket_connect("/v1/chat/ws?chat_id=chat-ws-4") as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "message.send",
                    "client_message_id": "msg-3",
                    "chat_id": "chat-ws-4",
                    "text": "Повтори ответ",
                }
            )

            websocket.receive_json()
            completed = websocket.receive_json()

        self.assertEqual(completed["type"], "message.completed")
        self.assertEqual(completed["answer"], "Уже готово")
        self.assertEqual(completed["server_message_id"], 11)


if __name__ == "__main__":
    unittest.main()
