from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import httpx

from agi_ecom_app.config import Settings, get_settings


ExecutionState = Literal["applying", "applied", "blocked", "failed"]


class WBActionExecutionError(RuntimeError):
    """A WB action could not be completed safely."""


class WBActionBlockedError(WBActionExecutionError):
    """The current external state no longer matches the approved action."""


class WBActionUnknownResultError(WBActionExecutionError):
    """A write may have reached WB, so blind retry is unsafe."""


@dataclass(frozen=True)
class WBExecutionOutcome:
    status: ExecutionState
    operation: str
    idempotency_key: str
    applied_spec: dict[str, Any] = field(default_factory=dict)
    external_reference: str | None = None
    verification_due_seconds: int | None = None
    detail: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class WBManagerActionExecutor:
    """Fail-closed executor for the reversible WB actions supported by manager v1.

    Idempotency is enforced by the business lifecycle in Postgres. The WB APIs do
    not expose a common idempotency header, therefore transport/5xx failures after
    a write are treated as an unknown result and are never retried blindly.
    """

    PROMOTION_PROD = "https://advert-api.wildberries.ru"
    PROMOTION_SANDBOX = "https://advert-api-sandbox.wildberries.ru"
    PRICES_PROD = "https://discounts-prices-api.wildberries.ru"
    PRICES_SANDBOX = "https://discounts-prices-api-sandbox.wildberries.ru"

    def __init__(
        self,
        *,
        settings: Settings | Any | None = None,
        promotion_client: httpx.Client | None = None,
        prices_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        sandbox = bool(getattr(self.settings, "wb_api_sandbox", True))
        promotion_url = self.PROMOTION_SANDBOX if sandbox else self.PROMOTION_PROD
        prices_url = self.PRICES_SANDBOX if sandbox else self.PRICES_PROD
        promotion_headers = self._auth_headers(
            getattr(self.settings, "wb_promotion_token", None)
        )
        prices_headers = self._auth_headers(getattr(self.settings, "wb_prices_token", None))
        self._promotion_client = promotion_client or httpx.Client(
            base_url=promotion_url,
            timeout=10.0,
            headers=promotion_headers,
        )
        self._prices_client = prices_client or httpx.Client(
            base_url=prices_url,
            timeout=10.0,
            headers=prices_headers,
        )
        if promotion_client is not None:
            self._promotion_client.headers.update(promotion_headers)
        if prices_client is not None:
            self._prices_client.headers.update(prices_headers)

    @staticmethod
    def _auth_headers(token: str | None) -> dict[str, str]:
        return {"Authorization": token, "Content-Type": "application/json"} if token else {
            "Content-Type": "application/json"
        }

    def execute(
        self,
        action_spec: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> WBExecutionOutcome:
        operation = self._operation(action_spec)
        if not idempotency_key.strip():
            raise WBActionBlockedError("idempotency_key is required")
        self._ensure_enabled(operation)
        current = self.preflight(action_spec)
        self._assert_expected_from(action_spec, current)

        if operation == "change_bid":
            payload = self._bid_payload(action_spec)
            self._write(self._promotion_client, "PATCH", "/api/advert/v1/bids", json=payload)
            return WBExecutionOutcome(
                status="applying",
                operation=operation,
                idempotency_key=idempotency_key,
                applied_spec={**dict(action_spec), "preflight": current},
                external_reference=self._campaign_reference(action_spec),
                verification_due_seconds=30,
            )

        if operation in {"pause_campaign", "start_campaign"}:
            advert_id = self._required_int(action_spec, "advert_id", "target_id")
            path = "/adv/v0/pause" if operation == "pause_campaign" else "/adv/v0/start"
            self._write(self._promotion_client, "GET", path, params={"id": advert_id})
            return WBExecutionOutcome(
                status="applying",
                operation=operation,
                idempotency_key=idempotency_key,
                applied_spec={**dict(action_spec), "preflight": current},
                external_reference=str(advert_id),
                verification_due_seconds=60,
            )

        if operation == "set_price_discount":
            payload = self._price_payload(action_spec)
            response = self._write(
                self._prices_client,
                "POST",
                "/api/v2/upload/task",
                json={"data": [payload]},
            )
            body = self._json(response)
            upload_id = self._nested(body, "data", "id") or self._nested(body, "data", "uploadID")
            if upload_id is None:
                raise WBActionUnknownResultError("WB accepted a price write without an upload id")
            return WBExecutionOutcome(
                status="applying",
                operation=operation,
                idempotency_key=idempotency_key,
                applied_spec={**dict(action_spec), "preflight": current},
                external_reference=str(upload_id),
                verification_due_seconds=15,
            )

        raise WBActionBlockedError(f"unsupported WB operation: {operation}")

    def preflight(self, action_spec: Mapping[str, Any]) -> dict[str, Any]:
        operation = self._operation(action_spec)
        self._ensure_enabled(operation)
        if operation in {"change_bid", "pause_campaign", "start_campaign"}:
            advert_id = self._required_int(action_spec, "advert_id", "target_id")
            response = self._read(
                self._promotion_client,
                "GET",
                "/api/advert/v2/adverts",
                params={"ids": str(advert_id)},
            )
            adverts = self._json(response).get("adverts") or []
            campaign = next(
                (item for item in adverts if int(item.get("id", -1)) == advert_id),
                None,
            )
            if not isinstance(campaign, dict):
                raise WBActionBlockedError(f"campaign {advert_id} not found during preflight")
            if operation == "change_bid":
                nm_id = self._required_int(action_spec, "nm_id")
                placement = str(action_spec.get("placement") or "combined")
                nm_settings = campaign.get("nm_settings") or []
                nm_row = next(
                    (item for item in nm_settings if int(item.get("nm_id", -1)) == nm_id),
                    None,
                )
                if not isinstance(nm_row, dict):
                    raise WBActionBlockedError(
                        f"nm_id {nm_id} is not attached to campaign {advert_id}"
                    )
                bids = nm_row.get("bids_kopecks") or {}
                if placement == "combined":
                    values = [bids.get("search"), bids.get("recommendations")]
                    normalized = [value for value in values if value is not None]
                    current_bid = normalized[0] if normalized and len(set(normalized)) == 1 else None
                else:
                    current_bid = bids.get(placement)
                return {
                    "advert_id": advert_id,
                    "nm_id": nm_id,
                    "placement": placement,
                    "current_value": current_bid,
                    "status": campaign.get("status"),
                }
            return {
                "advert_id": advert_id,
                "current_value": campaign.get("status"),
                "status": campaign.get("status"),
            }

        if operation == "set_price_discount":
            nm_id = self._required_int(action_spec, "nm_id", "target_id")
            response = self._read(
                self._prices_client,
                "POST",
                "/api/v2/list/goods/filter",
                json={"nmList": [nm_id]},
            )
            goods = self._nested(self._json(response), "data", "listGoods") or []
            item = next((row for row in goods if int(row.get("nmID", -1)) == nm_id), None)
            if not isinstance(item, dict):
                raise WBActionBlockedError(f"nm_id {nm_id} price was not found during preflight")
            sizes = item.get("sizes") or []
            price_values = {row.get("price") for row in sizes if row.get("price") is not None}
            price = next(iter(price_values)) if len(price_values) == 1 else None
            return {
                "nm_id": nm_id,
                "price": price,
                "discount": item.get("discount"),
                "current_value": {
                    "price": price,
                    "discount": item.get("discount"),
                },
            }

        raise WBActionBlockedError(f"unsupported WB operation: {operation}")

    def verify(
        self,
        action_spec: Mapping[str, Any],
        *,
        external_reference: str | None,
        idempotency_key: str,
    ) -> WBExecutionOutcome:
        operation = self._operation(action_spec)
        if operation == "set_price_discount":
            if not external_reference:
                current = self.preflight(action_spec)
                if self._values_equal(
                    self._expected_to(action_spec), current.get("current_value")
                ):
                    return WBExecutionOutcome(
                        status="applied",
                        operation=operation,
                        idempotency_key=idempotency_key,
                        applied_spec={**dict(action_spec), "verified_state": current},
                    )
                return WBExecutionOutcome(
                    status="applying",
                    operation=operation,
                    idempotency_key=idempotency_key,
                    verification_due_seconds=30,
                    detail=(
                        "Price write result is unknown and the target state is not visible yet; "
                        "the write will not be replayed"
                    ),
                )
            response = self._read(
                self._prices_client,
                "GET",
                "/api/v2/history/tasks",
                params={"uploadID": external_reference},
            )
            body = self._json(response)
            status = self._nested(body, "data", "status")
            try:
                status = int(status)
            except (TypeError, ValueError):
                status = None
            if status == 3:
                current = self.preflight(action_spec)
                if self._values_equal(
                    self._expected_to(action_spec), current.get("current_value")
                ):
                    return WBExecutionOutcome(
                        status="applied",
                        operation=operation,
                        idempotency_key=idempotency_key,
                        applied_spec={**dict(action_spec), "verified_state": current},
                        external_reference=external_reference,
                    )
                return WBExecutionOutcome(
                    status="applying",
                    operation=operation,
                    idempotency_key=idempotency_key,
                    external_reference=external_reference,
                    verification_due_seconds=30,
                    detail="WB marked the upload successful but read-after-write has not converged",
                )
            if status in {4, 5, 6}:
                return WBExecutionOutcome(
                    status="failed",
                    operation=operation,
                    idempotency_key=idempotency_key,
                    external_reference=external_reference,
                    detail=f"WB price upload finished with status {status}",
                )
            return WBExecutionOutcome(
                status="applying",
                operation=operation,
                idempotency_key=idempotency_key,
                external_reference=external_reference,
                verification_due_seconds=30,
            )

        current = self.preflight(action_spec)
        expected = self._expected_to(action_spec)
        if self._values_equal(expected, current.get("current_value")):
            return WBExecutionOutcome(
                status="applied",
                operation=operation,
                idempotency_key=idempotency_key,
                applied_spec={**dict(action_spec), "verified_state": current},
                external_reference=external_reference,
            )
        return WBExecutionOutcome(
            status="applying",
            operation=operation,
            idempotency_key=idempotency_key,
            external_reference=external_reference,
            verification_due_seconds=30,
            detail="WB state has not converged to the approved target yet",
        )

    def rollback(
        self,
        action_spec: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> WBExecutionOutcome:
        operation = self._operation(action_spec)
        inverse = dict(action_spec)
        inverse["from"] = self._expected_to(action_spec)
        inverse["to"] = self._expected_from(action_spec)
        if operation == "pause_campaign":
            inverse["operation"] = "start_campaign"
        elif operation == "start_campaign":
            inverse["operation"] = "pause_campaign"
        return self.execute(inverse, idempotency_key=idempotency_key)

    def _ensure_enabled(self, operation: str) -> None:
        if operation in {"change_bid", "pause_campaign", "start_campaign"}:
            if not bool(getattr(self.settings, "wb_ads_write_enabled", False)):
                raise WBActionBlockedError("WB advertising writes are disabled")
            if not getattr(self.settings, "wb_promotion_token", None):
                raise WBActionBlockedError("WB promotion token is not configured")
            return
        if operation == "set_price_discount":
            if not bool(getattr(self.settings, "wb_price_write_enabled", False)):
                raise WBActionBlockedError("WB price writes are disabled")
            if not getattr(self.settings, "wb_prices_token", None):
                raise WBActionBlockedError("WB prices token is not configured")
            return
        raise WBActionBlockedError(f"unsupported WB operation: {operation}")

    @staticmethod
    def _operation(action_spec: Mapping[str, Any]) -> str:
        value = str(action_spec.get("operation") or "").strip()
        if not value:
            raise WBActionBlockedError("action operation is required")
        return value

    def _assert_expected_from(
        self,
        action_spec: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> None:
        expected = self._expected_from(action_spec)
        if expected is None:
            raise WBActionBlockedError("exact approved from-value is required")
        actual = current.get("current_value")
        if not self._values_equal(expected, actual):
            raise WBActionBlockedError(
                f"external state changed after approval: expected {expected!r}, got {actual!r}"
            )

    @staticmethod
    def _values_equal(expected: Any, actual: Any) -> bool:
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            return all(actual.get(key) == value for key, value in expected.items())
        return expected == actual

    @staticmethod
    def _expected_from(action_spec: Mapping[str, Any]) -> Any:
        return action_spec.get("from", action_spec.get("from_value"))

    @staticmethod
    def _expected_to(action_spec: Mapping[str, Any]) -> Any:
        return action_spec.get("to", action_spec.get("to_value"))

    def _bid_payload(self, action_spec: Mapping[str, Any]) -> dict[str, Any]:
        advert_id = self._required_int(action_spec, "advert_id", "target_id")
        nm_id = self._required_int(action_spec, "nm_id")
        to_value = self._expected_to(action_spec)
        if not isinstance(to_value, (int, float)) or to_value < 0:
            raise WBActionBlockedError("target bid in kopecks must be a non-negative number")
        placement = str(action_spec.get("placement") or "combined")
        if placement not in {"combined", "search", "recommendations"}:
            raise WBActionBlockedError("unsupported campaign placement")
        return {
            "bids": [
                {
                    "advert_id": advert_id,
                    "nm_bids": [
                        {
                            "nm_id": nm_id,
                            "bid_kopecks": int(to_value),
                            "placement": placement,
                        }
                    ],
                }
            ]
        }

    def _price_payload(self, action_spec: Mapping[str, Any]) -> dict[str, Any]:
        nm_id = self._required_int(action_spec, "nm_id", "target_id")
        to_value = self._expected_to(action_spec)
        if not isinstance(to_value, Mapping):
            raise WBActionBlockedError("price action to-value must contain price and/or discount")
        payload: dict[str, Any] = {"nmID": nm_id}
        if to_value.get("price") is not None:
            price = to_value["price"]
            if not isinstance(price, (int, float)) or price <= 0:
                raise WBActionBlockedError("target price must be positive")
            payload["price"] = int(price)
        if to_value.get("discount") is not None:
            discount = to_value["discount"]
            if not isinstance(discount, (int, float)) or not 0 <= discount <= 99:
                raise WBActionBlockedError("target discount must be between 0 and 99")
            payload["discount"] = int(discount)
        if len(payload) == 1:
            raise WBActionBlockedError("price action must change price or discount")
        return payload

    @staticmethod
    def _required_int(action_spec: Mapping[str, Any], *keys: str) -> int:
        for key in keys:
            value = action_spec.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                break
        raise WBActionBlockedError(f"integer {'/'.join(keys)} is required")

    @staticmethod
    def _campaign_reference(action_spec: Mapping[str, Any]) -> str:
        return str(action_spec.get("advert_id") or action_spec.get("target_id") or "")

    def _read(self, client: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_response: httpx.Response | None = None
        for _ in range(3):
            try:
                response = client.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                raise WBActionExecutionError(f"WB preflight failed: {type(exc).__name__}") from exc
            last_response = response
            if response.status_code != 429:
                break
        assert last_response is not None
        if last_response.status_code >= 400:
            raise WBActionExecutionError(
                f"WB preflight failed with HTTP {last_response.status_code}"
            )
        return last_response

    def _write(self, client: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise WBActionUnknownResultError(
                f"WB write result is unknown after {type(exc).__name__}; verification is required"
            ) from exc
        if response.status_code == 429:
            raise WBActionExecutionError("WB rate limit rejected the write; safe retry must be queued")
        if response.status_code >= 500:
            raise WBActionUnknownResultError(
                f"WB write returned HTTP {response.status_code}; verify before retry"
            )
        if response.status_code >= 400:
            raise WBActionExecutionError(f"WB rejected the write with HTTP {response.status_code}")
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise WBActionExecutionError("WB returned invalid JSON") from exc
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _nested(value: Mapping[str, Any], *path: str) -> Any:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current


__all__ = [
    "WBActionBlockedError",
    "WBActionExecutionError",
    "WBActionUnknownResultError",
    "WBExecutionOutcome",
    "WBManagerActionExecutor",
]
