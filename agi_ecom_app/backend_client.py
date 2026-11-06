from __future__ import annotations

from typing import Any

import httpx

from agi_ecom_app.api_models import (
    ActivePromosResponse,
    AgentRunStatusResponse,
    ChatMessageResponse,
    ChatSessionResponse,
    DashboardResponse,
    FlashSaleWinterResponse,
    HealthResponse,
)
from agi_ecom_app.config import get_settings


class BackendClientError(RuntimeError):
    """Raised when the backend API is unavailable or returns an error."""


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        settings = get_settings()
        self.base_url = (base_url or settings.backend_base_url).rstrip("/")
        self.timeout = timeout

    def check_backend_health(self) -> HealthResponse:
        payload = self._request("GET", "/health")
        return HealthResponse.model_validate(payload)

    def create_chat_session(self) -> ChatSessionResponse:
        payload = self._request("POST", "/v1/chat/sessions")
        return ChatSessionResponse.model_validate(payload)

    def send_chat_message(self, chat_id: str, message: str) -> ChatMessageResponse:
        payload = self._request(
            "POST",
            "/v1/chat/messages",
            json={"chat_id": chat_id, "message": message},
        )
        return ChatMessageResponse.model_validate(payload)

    def fetch_agent_run_status(self, run_id: int, chat_id: str | None = None) -> AgentRunStatusResponse:
        params = {"chat_id": chat_id} if chat_id else None
        payload = self._request("GET", f"/v1/chat/agent-runs/{run_id}/status", params=params)
        return AgentRunStatusResponse.model_validate(payload)

    def fetch_daily_dashboard(self) -> DashboardResponse:
        payload = self._request("GET", "/v1/dashboard/daily")
        return DashboardResponse.model_validate(payload)

    def fetch_active_promos(self) -> ActivePromosResponse:
        payload = self._request("GET", "/v1/promos/active")
        return ActivePromosResponse.model_validate(payload)

    def fetch_flash_sale_winter(self) -> FlashSaleWinterResponse:
        payload = self._request("GET", "/v1/promos/flash-sale-winter")
        return FlashSaleWinterResponse.model_validate(payload)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(method, url, timeout=self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendClientError(f"Backend API недоступен: {exc}") from exc

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise BackendClientError(f"Backend API ошибка {response.status_code}: {detail}")

        return response.json()


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("message") or payload)
    return str(payload)
