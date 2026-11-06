from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agi_ecom_app.experiment_program import service
from agi_ecom_app.experiment_program.models import ExperimentProgramCreate
from agi_ecom_app.experiment_program.repository import ExperimentProgramBundleRows


NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


def _evidence(metric: str, *, value: object = 1, status: str = "available") -> dict:
    payload = {
        "metric": metric,
        "value": value,
        "status": status,
    }
    if status == "available":
        payload.update(
            {
                "evidenceRef": f"agent_marts.{metric}:123:2026-07-20",
                "sourceTable": f"agent_marts.{metric}",
                "grain": "nm_id/day",
                "dataAsOf": NOW.isoformat(),
                "maxAgeSeconds": 86_400,
            }
        )
    return payload


def _program_payload(*, missing_orders: bool = False) -> ExperimentProgramCreate:
    evidence = [
        _evidence("orders", status="missing" if missing_orders else "available"),
        _evidence("unit_margin", value=500),
        _evidence("ad_spend", value=1000),
    ]
    return ExperimentProgramCreate.model_validate(
        {
            "objective": "incremental_contribution_profit",
            "mode": "shadow",
            "target": {"nmId": 123},
            "domains": ["pricing", "advertising", "finance"],
            "generatorVersion": "joint-price-ads-v1",
            "evidence": evidence,
            "candidates": [
                {
                    "candidateKey": "price-5310",
                    "domain": "pricing",
                    "family": "pricing",
                    "operation": "set_price",
                    "target": {"nmId": 123},
                    "parameters": {"priceRub": 5310},
                    "requiredMetrics": ["orders", "unit_margin"],
                },
                {
                    "candidateKey": "ads-pause",
                    "domain": "advertising",
                    "family": "advertising",
                    "operation": "pause_campaign",
                    "target": {"nmId": 123, "advertId": 77},
                    "parameters": {"paused": True},
                    "requiredMetrics": ["ad_spend"],
                },
            ],
        }
    )


@pytest.fixture
def captured_create(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        program = {
            **kwargs["program"],
            "created_at": NOW,
            "updated_at": NOW,
            "completed_at": None,
            "heartbeat_at": None,
        }
        snapshot = {**kwargs["snapshot"], "created_at": NOW}
        frontiers = [
            {
                **row,
                "created_at": NOW,
                "updated_at": NOW,
                "completed_at": None,
            }
            for row in kwargs["frontiers"]
        ]
        candidates = [
            {
                **row,
                "created_at": NOW,
                "updated_at": NOW,
                "completed_at": None,
            }
            for row in kwargs["candidates"]
        ]
        return (
            ExperimentProgramBundleRows(
                program=program,
                snapshot=snapshot,
                frontiers=frontiers,
                candidates=candidates,
            ),
            True,
        )

    monkeypatch.setattr(service, "create_program_bundle_atomic", fake_create)
    return calls


def test_evidence_authority_blocks_missing_metric_without_imputation(
    captured_create: list[dict],
) -> None:
    bundle, created = service.create_experiment_program(
        _program_payload(missing_orders=True),
        idempotency_key="program-1",
        now=NOW,
    )

    assert created is True
    candidates = {item.candidate_key: item for item in bundle.candidates}
    assert candidates["price-5310"].status == "blocked_data"
    assert candidates["price-5310"].evidence_status == "missing"
    assert candidates["price-5310"].evidence_refs == [
        "agent_marts.unit_margin:123:2026-07-20"
    ]
    assert candidates["ads-pause"].status == "eligible"
    assert bundle.program.status == "proposing"
    requirements = captured_create[0]["requirements"]
    assert next(row for row in requirements if row["metric"] == "orders")["status"] == "missing"


def test_stale_evidence_is_evaluated_at_snapshot_time(
    captured_create: list[dict],
) -> None:
    payload = _program_payload()
    stale_packet = payload.evidence[0].model_copy(
        update={"data_as_of": NOW - timedelta(days=3), "max_age_seconds": 60}
    )
    payload = payload.model_copy(update={"evidence": [stale_packet, *payload.evidence[1:]]})

    bundle, _ = service.create_experiment_program(
        payload,
        idempotency_key="program-stale",
        now=NOW,
    )

    price = next(item for item in bundle.candidates if item.domain == "pricing")
    assert price.evidence_status == "stale"
    assert price.status == "blocked_data"


def test_price_and_advertising_on_same_sku_get_automatic_causal_conflict(
    captured_create: list[dict],
) -> None:
    bundle, _ = service.create_experiment_program(
        _program_payload(),
        idempotency_key="program-conflict",
        now=NOW,
    )

    assert all(item.status == "eligible" for item in bundle.candidates)
    conflicts = captured_create[0]["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["interference_key"] == "sku:123"
    assert conflicts[0]["details"]["source"] == "automatic_price_ads_gate"


def test_domain_without_action_lever_is_visible_not_silently_omitted(
    captured_create: list[dict],
) -> None:
    bundle, _ = service.create_experiment_program(
        _program_payload(),
        idempotency_key="program-domain-coverage",
        now=NOW,
    )

    finance = next(item for item in bundle.frontiers if item.domain == "finance")
    assert finance.status == "no_controllable_lever"
    assert finance.candidate_count == 0
    assert finance.coverage["candidateCatalogComplete"] is True
    assert finance.coverage["negotiationDone"] is False


def test_same_contract_and_key_produce_identical_ids_and_snapshot(
    captured_create: list[dict],
) -> None:
    first, _ = service.create_experiment_program(
        _program_payload(), idempotency_key="same-key", now=NOW
    )
    second, _ = service.create_experiment_program(
        _program_payload(), idempotency_key="same-key", now=NOW
    )

    assert first.program.id == second.program.id
    assert first.program.snapshot_hash == second.program.snapshot_hash
    assert [item.id for item in first.candidates] == [item.id for item in second.candidates]
