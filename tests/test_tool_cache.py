from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agents import FunctionTool

from agi_ecom_app import tool_cache


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, **kwargs):
        if kwargs.get("nx") and key in self.values:
            return False
        self.values[key] = value
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class BrokenRedis(FakeRedis):
    def get(self, key: str):
        raise ConnectionError("redis unavailable")


class TraceContext:
    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self.events: list[tuple[str, dict]] = []

    def add_decision_trace(self, event: str, **kwargs) -> None:
        self.events.append((event, kwargs))


def _function_tool(callback) -> FunctionTool:
    return FunctionTool(
        name="read_metric",
        description="read-only test tool",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=callback,
    )


def test_identical_calls_are_cached_with_canonical_arguments(monkeypatch) -> None:
    redis = FakeRedis()
    calls = 0

    async def original(_context, _arguments):
        nonlocal calls
        calls += 1
        return {"value": 42}

    monkeypatch.setattr(tool_cache, "_redis_client", lambda _url: redis)
    monkeypatch.setattr(
        tool_cache,
        "get_settings",
        lambda: SimpleNamespace(redis_url="redis://test"),
    )
    wrapped = tool_cache.trace_cached_tool(_function_tool(original))
    trace_context = TraceContext(run_id=77)
    context = SimpleNamespace(context=trace_context)

    async def invoke_twice():
        first = await wrapped.on_invoke_tool(context, '{"a":1,"b":2}')
        second = await wrapped.on_invoke_tool(context, '{"b":2,"a":1}')
        return first, second

    first, second = asyncio.run(invoke_twice())

    assert first == second == {"value": 42}
    assert calls == 1
    assert [event for event, _ in trace_context.events] == [
        "tool_cache_miss",
        "tool_cache_hit",
    ]


def test_cache_is_scoped_to_run_id(monkeypatch) -> None:
    redis = FakeRedis()
    calls = 0

    async def original(_context, _arguments):
        nonlocal calls
        calls += 1
        return calls

    monkeypatch.setattr(tool_cache, "_redis_client", lambda _url: redis)
    monkeypatch.setattr(
        tool_cache,
        "get_settings",
        lambda: SimpleNamespace(redis_url="redis://test"),
    )
    wrapped = tool_cache.trace_cached_tool(_function_tool(original))

    async def invoke_for_runs():
        first = await wrapped.on_invoke_tool(
            SimpleNamespace(context=TraceContext(1)), "{}"
        )
        second = await wrapped.on_invoke_tool(
            SimpleNamespace(context=TraceContext(2)), "{}"
        )
        return first, second

    assert asyncio.run(invoke_for_runs()) == (1, 2)


def test_cache_failure_is_fail_open_without_double_tool_execution(monkeypatch) -> None:
    calls = 0

    async def original(_context, _arguments):
        nonlocal calls
        calls += 1
        return "ok"

    monkeypatch.setattr(tool_cache, "_redis_client", lambda _url: BrokenRedis())
    monkeypatch.setattr(
        tool_cache,
        "get_settings",
        lambda: SimpleNamespace(redis_url="redis://test"),
    )
    wrapped = tool_cache.trace_cached_tool(_function_tool(original))

    result = asyncio.run(
        wrapped.on_invoke_tool(SimpleNamespace(context=TraceContext(9)), "{}")
    )

    assert result == "ok"
    assert calls == 1
