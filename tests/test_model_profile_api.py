from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.model_profiles import OPENAI_DEFAULT_PROFILE, OPENROUTER_GEMMA_FREE_PROFILE


def _settings(**overrides):
    base = {
        "openai_api_key": "sk-openai-test",
        "openai_model": "gpt-test",
        "openai_memory_model": "gpt-memory",
        "openrouter_api_key": None,
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_model": "google/gemma-4-31b-it:free",
        "openrouter_http_referer": None,
        "openrouter_title": "AI-ECOM Test",
        "llm_model_profile": OPENAI_DEFAULT_PROFILE,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class ModelProfileApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db")
    @patch("agi_ecom_app.backend_service.get_settings")
    def test_get_model_profile_returns_active_profile_and_available_profiles(
        self,
        mock_get_settings,
        mock_load_profile,
    ):
        mock_get_settings.return_value = _settings()
        mock_load_profile.return_value = OPENAI_DEFAULT_PROFILE

        response = self.client.get("/v1/model-profile")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["profile"], OPENAI_DEFAULT_PROFILE)
        self.assertEqual(payload["model_label"], "openai:gpt-test")
        self.assertEqual(payload["memory_model"], "gpt-memory")
        self.assertIn(OPENROUTER_GEMMA_FREE_PROFILE, [item["profile"] for item in payload["available_profiles"]])
        self.assertNotIn("sk-openai-test", str(payload))

    @patch("agi_ecom_app.backend_service.set_active_model_profile_name")
    @patch("agi_ecom_app.services.runtime_settings._load_runtime_profile_from_db")
    @patch("agi_ecom_app.backend_service.get_settings")
    def test_put_model_profile_saves_openrouter_when_key_is_configured(
        self,
        mock_get_settings,
        mock_load_profile,
        mock_set_profile,
    ):
        mock_get_settings.return_value = _settings(openrouter_api_key="sk-or-test")
        mock_load_profile.return_value = OPENROUTER_GEMMA_FREE_PROFILE
        mock_set_profile.return_value = OPENROUTER_GEMMA_FREE_PROFILE

        response = self.client.put("/v1/model-profile", json={"profile": OPENROUTER_GEMMA_FREE_PROFILE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["profile"], OPENROUTER_GEMMA_FREE_PROFILE)
        mock_set_profile.assert_called_once_with(OPENROUTER_GEMMA_FREE_PROFILE)

    @patch("agi_ecom_app.backend_service.set_active_model_profile_name")
    @patch("agi_ecom_app.backend_service.get_settings")
    def test_put_model_profile_rejects_openrouter_without_key(
        self,
        mock_get_settings,
        mock_set_profile,
    ):
        mock_get_settings.return_value = _settings(openrouter_api_key=None)

        response = self.client.put("/v1/model-profile", json={"profile": OPENROUTER_GEMMA_FREE_PROFILE})

        self.assertEqual(response.status_code, 422)
        self.assertIn("OPENROUTER_API_KEY", response.json()["detail"])
        mock_set_profile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
