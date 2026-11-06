import json
from types import SimpleNamespace

import httpx
import pytest

from agi_ecom_app.services.wb_manager_execution import (
    WBActionBlockedError,
    WBActionExecutionError,
    WBActionUnknownResultError,
    WBManagerActionExecutor,
)


def _settings(**overrides):
    values = {
        "wb_api_sandbox": True,
        "wb_ads_write_enabled": True,
        "wb_price_write_enabled": True,
        "wb_promotion_token": "promotion-token",
        "wb_prices_token": "prices-token",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://wb.test")


def test_change_bid_requires_exact_external_from_value():
    def promotion(request):
        assert request.headers["Authorization"] == "promotion-token"
        return httpx.Response(
            200,
            json={
                "adverts": [
                    {
                        "id": 10,
                        "status": 9,
                        "nm_settings": [
                            {"nm_id": 20, "bids_kopecks": {"search": 300}}
                        ],
                    }
                ]
            },
        )

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    with pytest.raises(WBActionBlockedError, match="external state changed"):
        executor.execute(
            {
                "operation": "change_bid",
                "advert_id": 10,
                "nm_id": 20,
                "placement": "search",
                "from": 250,
                "to": 200,
            },
            idempotency_key="approve-1",
        )


def test_change_bid_writes_once_and_returns_applying():
    calls = []

    def promotion(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/advert/v2/adverts":
            return httpx.Response(
                200,
                json={
                    "adverts": [
                        {
                            "id": 10,
                            "status": 9,
                            "nm_settings": [
                                {"nm_id": 20, "bids_kopecks": {"search": 300}}
                            ],
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"bids": []})

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    result = executor.execute(
        {
            "operation": "change_bid",
            "advert_id": 10,
            "nm_id": 20,
            "placement": "search",
            "from": 300,
            "to": 200,
        },
        idempotency_key="approve-2",
    )
    assert result.status == "applying"
    assert result.verification_due_seconds == 30
    assert calls == [
        ("GET", "/api/advert/v2/adverts"),
        ("PATCH", "/api/advert/v1/bids"),
    ]


def test_price_write_is_async_and_verified_by_upload_status():
    calls = []
    uploaded = False

    def prices(request):
        nonlocal uploaded
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v2/list/goods/filter":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "listGoods": [
                            {
                                "nmID": 55,
                                "discount": 10,
                                    "sizes": [{"price": 950 if uploaded else 1000}],
                            }
                        ]
                    }
                },
            )
        if request.url.path == "/api/v2/upload/task":
            uploaded = True
            return httpx.Response(200, json={"data": {"id": 987}})
        return httpx.Response(200, json={"data": {"uploadID": 987, "status": 3}})

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(lambda _: httpx.Response(500)),
        prices_client=_client(prices),
    )
    spec = {
        "operation": "set_price_discount",
        "nm_id": 55,
        "from": {"price": 1000, "discount": 10},
        "to": {"price": 950, "discount": 10},
    }
    accepted = executor.execute(spec, idempotency_key="approve-3")
    assert accepted.status == "applying"
    assert accepted.external_reference == "987"
    verified = executor.verify(
        spec,
        external_reference=accepted.external_reference,
        idempotency_key="approve-3",
    )
    assert verified.status == "applied"


@pytest.mark.parametrize("status_code", [401, 429])
def test_rejected_ad_write_is_not_retried(status_code):
    write_calls = 0

    def promotion(request):
        nonlocal write_calls
        if request.url.path == "/api/advert/v2/adverts":
            return httpx.Response(
                200,
                json={"adverts": [{"id": 10, "status": 9, "nm_settings": []}]},
            )
        write_calls += 1
        return httpx.Response(status_code)

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    with pytest.raises(WBActionExecutionError):
        executor.execute(
            {"operation": "pause_campaign", "advert_id": 10, "from": 9, "to": 11},
            idempotency_key=f"reject-{status_code}",
        )
    assert write_calls == 1


def test_server_error_after_write_is_unknown_and_not_retried():
    write_calls = 0

    def promotion(request):
        nonlocal write_calls
        if request.url.path == "/api/advert/v2/adverts":
            return httpx.Response(
                200,
                json={"adverts": [{"id": 10, "status": 9, "nm_settings": []}]},
            )
        write_calls += 1
        return httpx.Response(503)

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    with pytest.raises(WBActionUnknownResultError):
        executor.execute(
            {"operation": "pause_campaign", "advert_id": 10, "from": 9, "to": 11},
            idempotency_key="unknown-503",
        )
    assert write_calls == 1


def test_price_quarantine_status_fails_without_marking_applied():
    def prices(request):
        if request.url.path == "/api/v2/history/tasks":
            return httpx.Response(200, json={"data": {"uploadID": 987, "status": 5}})
        return httpx.Response(500)

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(lambda _: httpx.Response(500)),
        prices_client=_client(prices),
    )
    result = executor.verify(
        {
            "operation": "set_price_discount",
            "nm_id": 55,
            "from": {"price": 1000},
            "to": {"price": 950},
        },
        external_reference="987",
        idempotency_key="quarantine-1",
    )
    assert result.status == "failed"
    assert "status 5" in (result.detail or "")


def test_rollback_uses_exact_inverse_change():
    writes = []

    def promotion(request):
        if request.url.path == "/api/advert/v2/adverts":
            return httpx.Response(
                200,
                json={
                    "adverts": [
                        {
                            "id": 10,
                            "status": 9,
                            "nm_settings": [
                                {"nm_id": 20, "bids_kopecks": {"search": 200}}
                            ],
                        }
                    ]
                },
            )
        writes.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200)

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    result = executor.rollback(
        {
            "operation": "change_bid",
            "advert_id": 10,
            "nm_id": 20,
            "placement": "search",
            "from": 300,
            "to": 200,
        },
        idempotency_key="rollback-1",
    )
    assert result.status == "applying"
    assert writes[0]["bids"][0]["nm_bids"][0]["bid_kopecks"] == 300


def test_write_transport_failure_is_unknown_and_not_retried():
    calls = 0

    def promotion(request):
        nonlocal calls
        calls += 1
        if request.url.path == "/api/advert/v2/adverts":
            return httpx.Response(
                200,
                json={"adverts": [{"id": 10, "status": 9, "nm_settings": []}]},
            )
        raise httpx.ReadTimeout("unknown", request=request)

    executor = WBManagerActionExecutor(
        settings=_settings(),
        promotion_client=_client(promotion),
        prices_client=_client(lambda _: httpx.Response(500)),
    )
    with pytest.raises(WBActionUnknownResultError):
        executor.execute(
            {
                "operation": "pause_campaign",
                "advert_id": 10,
                "from": 9,
                "to": 11,
            },
            idempotency_key="approve-4",
        )
    assert calls == 2


def test_executor_is_fail_closed_when_write_flag_is_off():
    executor = WBManagerActionExecutor(
        settings=_settings(wb_ads_write_enabled=False),
        promotion_client=_client(lambda _: httpx.Response(200)),
        prices_client=_client(lambda _: httpx.Response(200)),
    )
    with pytest.raises(WBActionBlockedError, match="disabled"):
        executor.execute(
            {"operation": "pause_campaign", "advert_id": 10, "from": 9, "to": 11},
            idempotency_key="approve-5",
        )
