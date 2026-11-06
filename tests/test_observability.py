from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from agi_ecom_app import observability
from agi_ecom_app.config import get_settings


def test_secret_masking_preserves_usage_and_business_ids() -> None:
    payload = json.dumps(
        {
            "authorization": "Bearer secret-token",
            "nested": {"api_key": "sk-proj-secretvalue123456", "nm_id": 12345},
            "input_tokens": 120,
            "business_token_count": 7,
        }
    )

    masked = observability._mask_attribute(
        "gen_ai.request.input", payload, max_bytes=262_144
    )
    parsed = json.loads(masked)

    assert parsed["authorization"] == "[REDACTED]"
    assert parsed["nested"]["api_key"] == "[REDACTED]"
    assert parsed["nested"]["nm_id"] == 12345
    assert parsed["input_tokens"] == 120
    assert parsed["business_token_count"] == 7


def test_masking_redacts_bearer_dsn_and_known_key_patterns() -> None:
    value = (
        "Authorization: Bearer abc.def.ghi "
        "postgresql://user:password@db:5432/app "
        "api_key=sk-proj-abcdefghijklmnop"
    )

    masked = observability._mask_attribute("gen_ai.prompt", value, max_bytes=262_144)

    assert "abc.def.ghi" not in masked
    assert ":password@" not in masked
    assert "sk-proj-abcdefghijklmnop" not in masked
    assert masked.count("[REDACTED]") >= 3


def test_large_attribute_is_bounded_and_has_integrity_metadata() -> None:
    original = "данные" * 10_000

    masked = observability._mask_attribute("gen_ai.output", original, max_bytes=1_024)
    payload = json.loads(masked)

    assert len(masked.encode("utf-8")) <= 1_024
    assert payload["truncated"] is True
    assert payload["original_bytes"] == len(original.encode("utf-8"))
    assert len(payload["sha256"]) == 64
    assert payload["prefix"]


def test_legacy_agent_trace_context_is_stable() -> None:
    first = observability.trace_context_for_agent_run(summary={}, run_id=42)
    second = observability.trace_context_for_agent_run(summary={}, run_id=42)

    assert first == second
    assert len(first[0]) == 32
    assert len(first[1]) == 16


def test_observation_scope_does_not_swallow_business_error_when_disabled() -> None:
    with pytest.raises(RuntimeError, match="business failure"):
        with observability.observation_scope(name="disabled-test"):
            raise RuntimeError("business failure")


def test_root_observation_uses_direct_ingestion_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = SimpleNamespace()
    fake_observation = SimpleNamespace(
        trace_id="1" * 32,
        id="2" * 16,
        update=lambda **_kwargs: None,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        observability,
        "ensure_langfuse_initialized",
        lambda **_kwargs: fake_client,
    )

    def fake_start(**kwargs):
        captured.update(kwargs)
        return fake_observation

    @contextmanager
    def fake_scope(**kwargs):
        captured["scope"] = kwargs
        yield kwargs["observation"]

    monkeypatch.setattr(observability, "_start_direct_root_observation", fake_start)
    monkeypatch.setattr(observability, "_direct_root_observation_scope", fake_scope)

    with observability.observation_scope(
        name="agent-run-intake",
        as_type="chain",
        trace_id="1" * 32,
        root=True,
        input={"query": "test"},
        session_id="chat-1",
        settings=SimpleNamespace(
            langfuse_max_field_bytes=262_144,
            langfuse_tracing_environment="test",
        ),
    ) as observation:
        assert observation is fake_observation

    assert captured["trace_id"] == "1" * 32
    assert captured["name"] == "agent-run-intake"
    assert captured["as_type"] == "chain"


def test_direct_root_payload_masking_matches_otel_boundary() -> None:
    payload = observability._safe_direct_payload(
        {
            "authorization": "Bearer abc.def.ghi",
            "nested": {"password": "secret", "sku": "46227"},
        },
        max_bytes=262_144,
    )

    assert payload == {
        "authorization": "[REDACTED]",
        "nested": {"password": "[REDACTED]", "sku": "46227"},
    }


def test_direct_root_exports_readable_span_without_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    exported = []
    processor = SimpleNamespace(on_end=exported.append)
    provider = SimpleNamespace(
        resource=Resource.create({"service.name": "test"}),
        _active_span_processor=processor,
    )
    client = SimpleNamespace(
        _otel_tracer=SimpleNamespace(
            _instrumentation_scope=InstrumentationScope(
                "langfuse-sdk",
                attributes={"public_key": "pk-test"},
            )
        )
    )
    monkeypatch.setattr(observability._state, "tracer_provider", provider)
    root = observability._DirectRootObservation(
        client=client,
        trace_id="1" * 32,
        id="2" * 16,
        name="agent-run-intake",
        environment="test",
        release="release-1",
        version="v2",
        input={"query": "test"},
        metadata={"run_id": 1},
        session_id="chat-1",
        tags=["agent-run"],
        start_time_ns=1,
        pending={"output": {"status": "queued"}},
    )

    root.finish(max_bytes=262_144)

    assert len(exported) == 1
    span = exported[0]
    assert span.parent is None
    assert f"{span.context.trace_id:032x}" == "1" * 32
    assert f"{span.context.span_id:016x}" == "2" * 16
    assert span.name == "agent-run-intake"
    assert span.attributes["langfuse.trace.name"] == "agent-run"
    assert span.attributes["session.id"] == "chat-1"


def test_missing_langfuse_credentials_is_fail_open() -> None:
    settings = SimpleNamespace(
        langfuse_enabled=True,
        langfuse_public_key=None,
        langfuse_secret_key=None,
        langfuse_tracing_environment="test",
    )

    assert observability.init_langfuse(settings=settings) is None


def test_initializer_is_idempotent_within_process(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = object()
    monkeypatch.setattr(observability._state, "pid", 101)
    monkeypatch.setattr(observability._state, "client", fake_client)
    monkeypatch.setattr(observability._state, "enabled", True)
    monkeypatch.setattr(observability.os, "getpid", lambda: 101)

    result = observability.init_langfuse(
        settings=SimpleNamespace(langfuse_enabled=True)
    )

    assert result is fake_client


def test_initializer_discards_inherited_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(observability._state, "pid", 101)
    monkeypatch.setattr(observability._state, "client", object())
    monkeypatch.setattr(observability._state, "enabled", True)
    monkeypatch.setattr(observability.os, "getpid", lambda: 202)
    monkeypatch.setattr(observability, "_reset_inherited_sdk_state", lambda: None)

    result = observability.init_langfuse(
        settings=SimpleNamespace(
            langfuse_enabled=True,
            langfuse_public_key=None,
            langfuse_secret_key=None,
            langfuse_tracing_environment="test",
        )
    )

    assert result is None
    assert observability._state.pid == 202
    assert observability._state.client is None
    assert observability._state.enabled is False


def test_settings_hide_secrets_and_validate_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-visible-only-to-client")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-never-in-repr")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0.5")
    settings = get_settings()

    rendered = repr(settings)
    assert settings.langfuse_sample_rate == 0.5
    assert "pk-lf-visible-only-to-client" not in rendered
    assert "sk-lf-never-in-repr" not in rendered

    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.1")
    with pytest.raises(ValueError, match="LANGFUSE_SAMPLE_RATE"):
        get_settings()
