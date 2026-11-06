import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import agi_ecom


class CliChatSessionTests(unittest.TestCase):
    @patch("agi_ecom.get_backend_client")
    def test_resolve_chat_id_requests_session_from_backend(self, mock_get_backend_client):
        mock_get_backend_client.return_value.create_chat_session.return_value = SimpleNamespace(chat_id="chat-api")

        chat_id = agi_ecom.resolve_chat_id(None)

        self.assertEqual(chat_id, "chat-api")
        mock_get_backend_client.return_value.create_chat_session.assert_called_once_with()

    @patch("agi_ecom.type_text")
    @patch("agi_ecom.render_agent_response")
    @patch("agi_ecom.clear_screen")
    @patch("agi_ecom.render_chat_session_status")
    @patch("agi_ecom.render_daily_report")
    @patch("agi_ecom.render_backend_health_warning")
    @patch("agi_ecom.read_command_prompt")
    @patch("agi_ecom.resolve_chat_id")
    @patch("argparse.ArgumentParser.parse_args")
    def test_clear_creates_new_chat_id(
        self,
        mock_parse_args,
        mock_resolve_chat_id,
        mock_read_command_prompt,
        mock_render_backend_health_warning,
        mock_render_daily_report,
        mock_render_chat_session_status,
        mock_clear_screen,
        mock_render_agent_response,
        mock_type_text,
    ):
        mock_parse_args.return_value = SimpleNamespace(skip_intro=True, chat_id="chat-1")
        mock_resolve_chat_id.side_effect = ["chat-1", "chat-2"]
        mock_read_command_prompt.side_effect = ["/clear", "/exit"]

        agi_ecom.main()

        self.assertEqual(mock_resolve_chat_id.call_args_list, [call("chat-1"), call(None)])
        mock_render_backend_health_warning.assert_called_once_with()
        self.assertEqual(
            mock_render_chat_session_status.call_args_list,
            [call("chat-1"), call("chat-2")],
        )
        self.assertEqual(mock_render_daily_report.call_count, 2)
        mock_clear_screen.assert_called_once_with()
        mock_render_agent_response.assert_not_called()
        mock_type_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
