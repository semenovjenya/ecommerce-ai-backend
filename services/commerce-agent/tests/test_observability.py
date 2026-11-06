from __future__ import annotations

import logging
import os
from concurrent.futures import Future
from functools import partial
from threading import Event
import time
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch
from uuid import uuid4

import pytest

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith import run_trees as langsmith_run_trees
from langsmith.schemas import LangSmithInfo

from commerce_agent.graph import build_graph
from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.ingress import GuardedGraph
from commerce_agent.observability import (
    EU_LANGSMITH_API_ENDPOINT,
    ObservabilityConfig,
    ObservabilityConfigurationError,
    configure_graph_observability,
    mask_langfuse_otel_spans,
    project_langsmith_run_ops,
    should_export_safe_langfuse_span,
)
from commerce_agent import observability


class _TestCallback(BaseCallbackHandler):
    pass


def _dual_environment() -> dict[str, str]:
    return {
        "COMMERCE_AGENT_LANGSMITH_TRACING": "true",
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_API_KEY": "langsmith-secret-canary",
        "LANGSMITH_ENDPOINT": EU_LANGSMITH_API_ENDPOINT,
        "LANGSMITH_HIDE_INPUTS": "true",
        "LANGSMITH_HIDE_OUTPUTS": "true",
        "LANGSMITH_HIDE_METADATA": "true",
        "LANGSMITH_PROJECT": "ai-ecom-test",
        "LANGFUSE_ENABLED": "true",
        "LANGFUSE_PUBLIC_KEY": "langfuse-public-canary",
        "LANGFUSE_SECRET_KEY": "langfuse-secret-canary",
        "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
        "LANGFUSE_TRACING_ENVIRONMENT": "test",
    }


def test_dual_tracing_attaches_safe_langsmith_and_langfuse_callbacks() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )

    assert graph.config["metadata"] == {
        "service": "commerce-agent",
        "graph_id": "commerce_agent",
        "observability_mode": "dual",
        "langsmith_project": "ai-ecom-test",
        "environment": "test",
    }
    assert len(graph.config["callbacks"]) == 2
    tracer, langfuse = graph.config["callbacks"]
    assert isinstance(tracer, LangChainTracer)
    assert tracer.client._hide_inputs is True
    assert tracer.client._hide_outputs is True
    assert tracer.client._hide_metadata is True
    assert tracer.client._omit_traced_runtime_info is True
    projector = tracer.client._process_buffered_run_ops
    assert isinstance(projector, partial)
    assert projector.func is project_langsmith_run_ops
    assert projector.keywords == {"project_name": "ai-ecom-test"}
    assert isinstance(langfuse, _TestCallback)
    assert "langsmith-secret-canary" not in repr(graph.config)
    assert "langfuse-secret-canary" not in repr(graph.config)
    assert "langfuse-public-canary" not in repr(graph.config)


def test_missing_credentials_disable_exporters_without_breaking_graph(
    caplog,
) -> None:
    caplog.set_level(logging.WARNING)
    graph = configure_graph_observability(
        build_graph(),
        environment={
            "COMMERCE_AGENT_LANGSMITH_TRACING": "true",
            "LANGFUSE_ENABLED": "true",
        },
    )

    assert graph.config["metadata"]["observability_mode"] == "disabled"
    assert "callbacks" not in graph.config
    assert "LANGSMITH_API_KEY is missing" in caplog.text
    assert "LANGFUSE_PUBLIC_KEY" in caplog.text


def test_wrong_langsmith_region_is_rejected_before_graph_config() -> None:
    unsafe = _dual_environment() | {
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com"
    }
    with pytest.raises(ObservabilityConfigurationError, match="EU API endpoint"):
        configure_graph_observability(build_graph(), environment=unsafe)


def test_langsmith_raw_export_is_rejected_before_graph_config() -> None:
    unsafe = _dual_environment() | {"LANGSMITH_HIDE_OUTPUTS": "false"}
    with pytest.raises(ObservabilityConfigurationError, match="hidden inputs"):
        configure_graph_observability(build_graph(), environment=unsafe)


def test_observability_config_never_retains_credentials() -> None:
    config = ObservabilityConfig.from_environment(_dual_environment())

    assert config.langsmith_has_api_key is True
    assert config.langsmith_hide_inputs is True
    assert config.langsmith_hide_outputs is True
    assert config.langsmith_hide_metadata is True
    assert config.langfuse_has_public_key is True
    assert config.langfuse_has_secret_key is True
    rendered = repr(config)
    assert "langsmith-secret-canary" not in rendered
    assert "langfuse-secret-canary" not in rendered
    assert "langfuse-public-canary" not in rendered


def test_langfuse_initialization_failure_is_fail_open(caplog) -> None:
    caplog.set_level(logging.ERROR)

    def fail() -> _TestCallback:
        raise RuntimeError("provider unavailable")

    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=fail,
    )

    assert graph.config["metadata"]["observability_mode"] == "langsmith"
    assert len(graph.config["callbacks"]) == 1
    assert isinstance(graph.config["callbacks"][0], LangChainTracer)
    assert "commerce_agent_langfuse_callback_initialization_failed" in caplog.text


def test_langsmith_callback_failure_cannot_fall_back_to_implicit_tracer() -> None:
    def fail(_config, _environment):
        raise RuntimeError("synthetic initialization failure")

    with pytest.raises(
        ObservabilityConfigurationError, match="safe LangSmith callback"
    ):
        configure_graph_observability(
            build_graph(),
            environment=_dual_environment(),
            langsmith_callback_factory=fail,
        )


@pytest.mark.parametrize(
    "flag",
    (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ),
)
def test_implicit_langsmith_tracer_is_rejected_before_graph_config(flag) -> None:
    with pytest.raises(
        ObservabilityConfigurationError, match="implicit LangSmith tracing disabled"
    ):
        configure_graph_observability(
            build_graph(),
            environment=_dual_environment() | {flag: "true"},
        )


def test_explicit_langsmith_tracer_is_the_only_tracer() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )
    with patch.dict("os.environ", _dual_environment()):
        manager = CallbackManager.configure(
            inheritable_callbacks=graph.config["callbacks"]
        )
    assert sum(isinstance(handler, LangChainTracer) for handler in manager.handlers) == 1


def test_guarded_graph_flushes_explicit_langsmith_client() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )
    tracer = graph.config["callbacks"][0]
    with patch.object(Client, "flush", autospec=True) as flush, patch(
        "langchain_core.tracers.langchain.wait_for_all_tracers",
        side_effect=AssertionError("global client must not be flushed"),
    ):
        GuardedGraph(graph).flush_traces(timeout=3)
    assert flush.call_count == 2
    assert all(call.args[0] is tracer.client for call in flush.call_args_list)


def test_trace_flush_waits_for_pending_projection_worker() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )
    tracer = graph.config["callbacks"][0]
    pending: Future[None] = Future()
    with tracer.client._projection_futures_lock:
        tracer.client._projection_futures.add(pending)

    with patch.object(Client, "flush", autospec=True):
        with pytest.raises(TimeoutError, match="projection drain"):
            GuardedGraph(graph).flush_traces(timeout=0.01)
        pending.set_result(None)
        GuardedGraph(graph).flush_traces(timeout=1)


def test_pinned_client_tracks_background_projection_before_flush() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )
    client = graph.config["callbacks"][0].client
    started = Event()
    release = Event()

    def blocked_worker(_client, _batch):
        started.set()
        assert release.wait(3)

    with patch(
        "langsmith._internal._background_thread._process_buffered_run_ops_batch",
        side_effect=blocked_worker,
    ), patch.object(Client, "flush", autospec=True):
        try:
            with client._run_ops_buffer_lock:
                client._run_ops_buffer.append(("post", {"id": uuid4()}, {}))
                client._flush_run_ops_buffer()
            assert started.wait(2)
            with pytest.raises(TimeoutError, match="projection drain"):
                GuardedGraph(graph).flush_traces(timeout=0.01)
        finally:
            release.set()
        GuardedGraph(graph).flush_traces(timeout=2)


def test_langsmith_projection_keeps_graph_structure_but_drops_payloads() -> None:
    run_id = uuid4()
    trace_id = uuid4()
    canary = "lsv2_" + "a" * 32
    operation = {
        "id": run_id,
        "trace_id": trace_id,
        "parent_run_id": trace_id,
        "dotted_order": f"20261009T000000000000Z{trace_id}",
        "start_time": "2026-10-09T00:00:00+00:00",
        "name": "sales",
        "session_name": "ai-ecom-test",
        "run_type": "chain",
        "inputs": {"private": canary},
        "outputs": {"private": canary},
        "error": canary,
        "events": [{"name": "failure", "data": canary}],
        "tags": [canary],
        "extra": {"metadata": {"private": canary}},
        "attachments": {"private": canary},
        "future_sdk_field": canary,
    }

    projected = project_langsmith_run_ops([operation])[0]

    assert projected["id"] == run_id
    assert projected["trace_id"] == trace_id
    assert projected["parent_run_id"] == trace_id
    assert projected["name"] == "sales"
    assert projected["run_type"] == "chain"
    assert projected["inputs"] == {}
    assert projected["outputs"] == {}
    assert projected["error"] == "redacted_error"
    assert projected["events"] == []
    assert projected["tags"] == []
    assert projected["extra"] == {}
    assert canary not in repr(projected)
    assert "future_sdk_field" not in projected
    assert "attachments" not in projected


def test_langsmith_projection_redacts_unsafe_name_and_rejects_bad_structure() -> None:
    run_id = uuid4()
    canary = "lsv2_" + "a" * 32
    assert project_langsmith_run_ops([{"id": run_id, "name": canary}])[0]["name"] == "redacted_run"
    projected = project_langsmith_run_ops(
        [{"id": run_id, "name": "customer_private_order_12345", "session_name": "forged"}],
        project_name="ai-ecom-test",
    )[0]
    assert projected["name"] == "redacted_run"
    assert projected["session_name"] == "ai-ecom-test"
    with pytest.raises(ObservabilityConfigurationError, match="run identifier"):
        project_langsmith_run_ops([{"id": canary}])


def test_pinned_langsmith_client_applies_projection_before_create_export() -> None:
    graph = configure_graph_observability(
        build_graph(),
        environment=_dual_environment(),
        langfuse_callback_factory=_TestCallback,
    )
    client = graph.config["callbacks"][0].client
    exported = []
    sent = Event()
    run_id = uuid4()
    canary = "lsv2_" + "a" * 32

    def capture(_client, run, **_kwargs):
        exported.append(run)
        sent.set()

    with patch.object(type(client), "_create_run", autospec=True, side_effect=capture):
        client.create_run(
            "sales",
            {"private": canary},
            "chain",
            id=run_id,
            trace_id=run_id,
            dotted_order=f"20261009T000000000000Z{run_id}",
            project_name="ai-ecom-test",
            extra={"metadata": {"private": canary}},
            error=canary,
            events=[{"name": "failure", "data": canary}],
        )
        assert sent.wait(5), "LangSmith buffered create was not processed"

    assert len(exported) == 1
    assert exported[0]["id"] == run_id
    assert exported[0]["name"] == "sales"
    assert exported[0]["inputs"] == {}
    assert exported[0]["error"] == "redacted_error"
    assert exported[0]["events"] == []
    assert exported[0]["extra"] == {}
    assert canary not in repr(exported[0])


def test_full_skeleton_graph_exports_projected_nodes_without_network() -> None:
    environment = _dual_environment() | {
        "LANGFUSE_ENABLED": "false",
    }
    exported: list[dict] = []
    send_clients: list[Client] = []
    network_calls: list[str] = []

    def capture(_client, run, **_kwargs):
        exported.append(run)

    def capture_send(client, *_args, **_kwargs):
        send_clients.append(client)

    def block_network(_client, _method, url, **_kwargs):
        network_calls.append(url)
        raise RuntimeError("synthetic network blocked")

    with patch.dict(os.environ, environment), patch.object(
        Client, "_create_run", autospec=True, side_effect=capture
    ), patch.object(Client, "_update_run", autospec=True, side_effect=capture), patch.object(
        Client, "_send_multipart_req", autospec=True, side_effect=capture_send
    ), patch.object(
        Client, "info", new_callable=PropertyMock, return_value=LangSmithInfo()
    ), patch.object(
        Client, "request_with_retries", autospec=True, side_effect=block_network
    ):
        compiled = build_graph()
        tracer = compiled.config["callbacks"][0]
        graph = GuardedGraph(compiled)
        result = graph.invoke(
            {"schema_version": 1, "messages": [HumanMessage(content="synthetic safe request")]},
            context=CommerceRuntimeContext(
                tenant_id="synthetic-tenant",
                user_id="synthetic-user",
                roles=("manager",),
                request_id="synthetic-request",
            ),
        )
        graph.flush_traces(timeout=5)
        graph.flush_traces(timeout=5)
        assert (
            langsmith_run_trees._CLIENT is None
            or langsmith_run_trees._CLIENT is tracer.client
        )
        for _ in range(50):
            names = {run.get("name") for run in exported}
            if {
                "commerce_agent", "intake", "sales", "marketing", "supply",
                "price", "finance",
            } <= names:
                break
            time.sleep(0.05)
        time.sleep(0.5)

    assert result["schema_version"] == 1
    assert not network_calls, network_calls
    assert all(client is tracer.client for client in send_clients)
    assert {
        "commerce_agent", "intake", "sales", "marketing", "supply",
        "price", "finance",
    } <= {
        run.get("name") for run in exported
    }
    assert all(
        run.get("inputs") in (None, {})
        and run.get("outputs") in (None, {})
        and run.get("events") in (None, [])
        and run.get("extra") in (None, {})
        for run in exported
    )
    assert "synthetic safe request" not in repr(exported)
    assert "synthetic-tenant" not in repr(exported)


def test_langfuse_export_mask_drops_text_and_preserves_numeric_usage(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_MODEL_ALLOWLIST", "gpt-4.1-mini")
    identifier = ("trace-id", "span-id")
    canary = "lsv2_" + "a" * 32
    params = SimpleNamespace(
        spans={
            identifier: SimpleNamespace(
                attributes={
                    "gen_ai.prompt.0.content": canary,
                    "langfuse.observation.input": "ordinary user text",
                    "gen_ai.usage.input_tokens": 12,
                    "gen_ai.usage.output_tokens": 5,
                    "langfuse.observation.usage_details": '{"input": 12, "output": 5}',
                    "langfuse.observation.cost_details": '{"total": 0.002}',
                    "langfuse.observation.model.name": "gpt-4.1-mini",
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.level": "DEFAULT",
                    "langfuse.environment": "development",
                    "langfuse.observation.metadata": '{"customer": "private"}',
                    "arbitrary_numeric_customer_id": 12345,
                    "gen_ai.usage.customer_id": 12345,
                    "secret": 123,
                }
            )
        }
    )

    result = mask_langfuse_otel_spans(params=params)

    assert result is not None
    deleted = set(result.span_patches[identifier].delete_attributes)
    assert deleted == {
        "gen_ai.prompt.0.content",
        "langfuse.observation.input",
        "langfuse.observation.metadata",
        "arbitrary_numeric_customer_id",
        "gen_ai.usage.customer_id",
        "secret",
    }
    assert "gen_ai.usage.input_tokens" not in deleted
    assert "gen_ai.usage.output_tokens" not in deleted
    assert "langfuse.observation.usage_details" not in deleted
    assert "langfuse.observation.cost_details" not in deleted
    assert "langfuse.observation.model.name" not in deleted
    assert "langfuse.observation.type" not in deleted
    assert "langfuse.observation.level" not in deleted
    assert "langfuse.environment" not in deleted


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("langfuse.observation.usage_details", '{"input": "lsv2_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'),
        ("langfuse.observation.usage_details", '{"input": -1}'),
        ("langfuse.observation.cost_details", '{"customer_id": 123}'),
        ("langfuse.observation.model.name", "gpt-4.1-mini-user-secret"),
        ("langfuse.observation.type", "lsv2_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    ],
)
def test_langfuse_export_mask_rejects_unapproved_metric_or_model(
    monkeypatch, key, value
) -> None:
    monkeypatch.setenv("LANGFUSE_MODEL_ALLOWLIST", "gpt-4.1-mini")
    identifier = ("trace-id", "span-id")
    params = SimpleNamespace(
        spans={identifier: SimpleNamespace(attributes={key: value})}
    )

    result = mask_langfuse_otel_spans(params=params)

    assert result is not None
    assert key in result.span_patches[identifier].delete_attributes


def test_default_langfuse_callback_configures_export_mask_first(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test-canary")
    with patch("langfuse.Langfuse") as client, patch(
        "langfuse.langchain.CallbackHandler"
    ) as handler:
        client.return_value._resources.mask_otel_spans = mask_langfuse_otel_spans
        client.return_value._resources.should_export_span = should_export_safe_langfuse_span
        callback = observability._create_langfuse_callback()

    client.assert_called_once_with(
        mask_otel_spans=mask_langfuse_otel_spans,
        should_export_span=should_export_safe_langfuse_span,
    )
    handler.assert_called_once_with(public_key="public-test-canary")
    assert callback is handler.return_value


def test_existing_unmasked_langfuse_client_is_not_attached() -> None:
    with patch("langfuse.Langfuse") as client, patch(
        "langfuse.langchain.CallbackHandler"
    ) as handler:
        client.return_value._resources.mask_otel_spans = None
        client.return_value._resources.should_export_span = should_export_safe_langfuse_span
        with pytest.raises(ObservabilityConfigurationError, match="safety hooks"):
            observability._create_langfuse_callback()

    handler.assert_not_called()


def test_existing_unfiltered_langfuse_client_is_not_attached() -> None:
    with patch("langfuse.Langfuse") as client, patch(
        "langfuse.langchain.CallbackHandler"
    ) as handler:
        client.return_value._resources.mask_otel_spans = mask_langfuse_otel_spans
        client.return_value._resources.should_export_span = None
        with pytest.raises(ObservabilityConfigurationError, match="safety hooks"):
            observability._create_langfuse_callback()

    handler.assert_not_called()


def _span_for_export(**overrides):
    resource = {
        "langfuse.environment": "development",
        "service.name": "unknown_service",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.0.0",
    }
    return SimpleNamespace(
        name=overrides.get("name", "commerce_agent"),
        instrumentation_scope=overrides.get(
            "instrumentation_scope", SimpleNamespace(name="langfuse-sdk", version=None)
        ),
        resource=SimpleNamespace(attributes=overrides.get("resource", resource)),
        events=overrides.get("events", ()),
        links=overrides.get("links", ()),
        status=SimpleNamespace(description=overrides.get("status_description", None)),
    )


def test_langfuse_span_filter_keeps_safe_sdk_span() -> None:
    with patch("langfuse.span_filter.is_default_export_span", return_value=True):
        assert should_export_safe_langfuse_span(_span_for_export())


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "lsv2_" + "a" * 32},
        {"name": "user supplied title"},
        {"events": (SimpleNamespace(name="user prompt"),)},
        {"links": (SimpleNamespace(),)},
        {"status_description": "provider returned user payload"},
        {"resource": {"service.name": "safe", "process.command_args": "private"}},
        {"resource": {"service.name": "lsv2_" + "a" * 32}},
        {"instrumentation_scope": SimpleNamespace(name="openai", version=None)},
    ],
)
def test_langfuse_span_filter_drops_unpatchable_payloads(overrides) -> None:
    with patch("langfuse.span_filter.is_default_export_span", return_value=True):
        assert not should_export_safe_langfuse_span(_span_for_export(**overrides))
