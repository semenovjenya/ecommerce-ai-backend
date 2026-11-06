import unittest
from unittest.mock import patch

from agi_ecom_app.backend_client import BackendClient, BackendClientError


class FakeResponse:
    def __init__(self, status_code=200, json_payload=None, text=""):
        self.status_code = status_code
        self._json_payload = json_payload if json_payload is not None else {}
        self.text = text

    def json(self):
        return self._json_payload


class BackendClientTests(unittest.TestCase):
    def setUp(self):
        self.client = BackendClient(base_url="http://127.0.0.1:8000")

    @patch("agi_ecom_app.backend_client.httpx.request")
    def test_check_health_calls_expected_endpoint(self, mock_request):
        mock_request.return_value = FakeResponse(
            json_payload={
                "status": "healthy",
                "memory_db": {"status": "healthy", "detail": None},
                "analytics_db": {"status": "healthy", "detail": None},
                "openai_config": {"status": "healthy", "detail": None},
            }
        )

        response = self.client.check_backend_health()

        self.assertEqual(response.status, "healthy")
        mock_request.assert_called_once_with(
            "GET",
            "http://127.0.0.1:8000/health",
            timeout=60.0,
        )

    @patch("agi_ecom_app.backend_client.httpx.request")
    def test_send_chat_message_posts_to_chat_endpoint(self, mock_request):
        mock_request.return_value = FakeResponse(
            json_payload={
                "chat_id": "chat-1",
                "answer": "ok",
                "intent": "finance",
                "status": "completed",
            }
        )

        response = self.client.send_chat_message("chat-1", "Какие продажи?")

        self.assertEqual(response.answer, "ok")
        mock_request.assert_called_once_with(
            "POST",
            "http://127.0.0.1:8000/v1/chat/messages",
            timeout=60.0,
            json={"chat_id": "chat-1", "message": "Какие продажи?"},
        )

    @patch("agi_ecom_app.backend_client.httpx.request")
    def test_fetch_promos_calls_expected_endpoint(self, mock_request):
        mock_request.return_value = FakeResponse(json_payload={"items": []})

        response = self.client.fetch_active_promos()

        self.assertEqual(response.items, [])
        mock_request.assert_called_once_with(
            "GET",
            "http://127.0.0.1:8000/v1/promos/active",
            timeout=60.0,
        )

    @patch("agi_ecom_app.backend_client.httpx.request")
    def test_backend_error_raises_client_error(self, mock_request):
        mock_request.return_value = FakeResponse(
            status_code=503,
            json_payload={"detail": "memory unavailable"},
        )

        with self.assertRaises(BackendClientError) as ctx:
            self.client.fetch_daily_dashboard()

        self.assertIn("memory unavailable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
