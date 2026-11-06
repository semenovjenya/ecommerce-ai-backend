from __future__ import annotations

import asyncio
import uuid

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph, IngressConfigurationError
from commerce_agent.safety import SensitiveStateError
from commerce_agent.serialization import checkpoint_serializer


class _StartSpy(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chain_start(self, _serialized, _inputs, **_kwargs) -> None:
        self.starts += 1


def _context() -> CommerceRuntimeContext:
    return CommerceRuntimeContext(
        tenant_id="tenant-1",
        user_id="user-1",
        roles=("manager",),
        request_id="request-1",
    )


def test_guarded_invocation_rejects_canary_before_callbacks_and_checkpoint() -> None:
    spy = _StartSpy()
    graph = GuardedGraph(
        build_graph(checkpointer=InMemorySaver(serde=checkpoint_serializer())).with_config(
            {"callbacks": [spy]}
        )
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    canary = "lsv2_" + "a" * 32

    with pytest.raises(SensitiveStateError):
        graph.invoke(
            {
                "schema_version": 1,
                "messages": [HumanMessage(content=canary, id="canary-message")],
            },
            config=config,
            context=_context(),
        )

    assert spy.starts == 0
    assert canary not in repr(
        [snapshot.values for snapshot in graph.get_state_history(config)]
    )

    safe_result = graph.invoke(
        {
            "schema_version": 1,
            "messages": [HumanMessage(content="safe message", id="safe-message")],
        },
        config=config,
        context=_context(),
    )
    assert safe_result["schema_version"] == 1
    assert spy.starts > 0


@pytest.mark.parametrize(
    "override",
    (
        {"callbacks": [_StartSpy()]},
        {"run_name": "customer_private_order_12345"},
        {"metadata": {"customer": "private"}},
        {"tags": ["private"]},
        {"configurable": {"thread_id": "safe", "callbacks": [_StartSpy()]}},
    ),
)
def test_guarded_invocation_rejects_untrusted_trace_config(override) -> None:
    compiled = _CallSpy()
    guarded = GuardedGraph(compiled)
    with pytest.raises(IngressConfigurationError):
        guarded.invoke(
            {"schema_version": 1, "messages": []},
            config=override,
            context=_context(),
        )
    assert not compiled.called


def test_guarded_async_stream_rejects_before_subscription() -> None:
    graph = GuardedGraph(build_graph())
    canary = "lsv2_" + "a" * 32

    with pytest.raises(SensitiveStateError):
        graph.astream(
            {"schema_version": 1, "messages": [HumanMessage(content=canary)]},
            config={"configurable": {"thread_id": "guarded-stream"}},
            context=_context(),
        )


def test_guarded_v3_protocol_stream_checks_input_before_callbacks() -> None:
    spy = _StartSpy()
    graph = GuardedGraph(
        build_graph(checkpointer=InMemorySaver(serde=checkpoint_serializer())).with_config(
            {"callbacks": [spy]}
        )
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    canary = "lsv2_" + "a" * 32

    async def exercise() -> None:
        with pytest.raises(SensitiveStateError):
            await graph.astream_protocol_events(
                {"schema_version": 1, "messages": [HumanMessage(content=canary)]},
                config=config,
                context=_context(),
            )
        with pytest.raises(SensitiveStateError):
            await graph.astream_protocol_events(
                Command(resume={"approval": canary}),
                config=config,
                context=_context(),
            )
        with pytest.raises(IngressConfigurationError):
            await graph.astream_protocol_events(
                {"schema_version": 1, "messages": []},
                config=config,
                context=_context(),
                callbacks=[spy],
            )
        assert spy.starts == 0
        stream = await graph.astream_protocol_events(
            {"schema_version": 1, "messages": [HumanMessage(content="safe test")]},
            config=config,
            context=_context(),
        )
        events = [event async for event in stream]
        assert events
        assert any(event.get("method") == "values" for event in events)
        assert spy.starts > 0

    asyncio.run(exercise())


class _CallSpy:
    def __init__(self) -> None:
        self.called = False

    def invoke(self, *_args, **_kwargs):
        self.called = True

    async def ainvoke(self, *_args, **_kwargs):
        self.called = True

    def stream(self, *_args, **_kwargs):
        self.called = True

    def astream(self, *_args, **_kwargs):
        self.called = True

    def astream_events(self, *_args, **_kwargs):
        self.called = True

    def get_state(self, *_args, **_kwargs):
        self.called = True

    async def aget_state(self, *_args, **_kwargs):
        self.called = True

    def get_state_history(self, *_args, **_kwargs):
        self.called = True

    def aget_state_history(self, *_args, **_kwargs):
        self.called = True


def test_guarded_graph_rejects_secret_in_extra_options_before_dispatch() -> None:
    compiled = _CallSpy()
    guarded = GuardedGraph(compiled)
    canary = "lsv2_" + "a" * 32

    with pytest.raises(SensitiveStateError):
        guarded.invoke(None, context=_context(), print_mode=canary)
    with pytest.raises(SensitiveStateError):
        guarded.stream(None, context=_context(), print_mode=canary)
    with pytest.raises(SensitiveStateError):
        guarded.astream(None, context=_context(), print_mode=canary)

    async def check_async() -> None:
        with pytest.raises(SensitiveStateError):
            await guarded.ainvoke(None, context=_context(), print_mode=canary)

    asyncio.run(check_async())
    assert not compiled.called


def test_guarded_state_reads_reject_secret_config_before_dispatch() -> None:
    compiled = _CallSpy()
    guarded = GuardedGraph(compiled)
    config = {"configurable": {"thread_id": "lsv2_" + "a" * 32}}

    with pytest.raises(SensitiveStateError):
        guarded.get_state(config)
    with pytest.raises(SensitiveStateError):
        guarded.get_state_history(config)
    with pytest.raises(SensitiveStateError):
        guarded.aget_state_history(config)

    async def check_async() -> None:
        with pytest.raises(SensitiveStateError):
            await guarded.aget_state(config)

    asyncio.run(check_async())
    assert not compiled.called
