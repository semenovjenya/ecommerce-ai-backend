from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
from functools import lru_cache
import json
import logging
import os
from typing import Any
from uuid import uuid4

from agents import FunctionTool

from agi_ecom_app.config import get_settings


logger = logging.getLogger(__name__)

CACHE_PREFIX = "agi-ecom:trace-tool-cache:v1"
DEFAULT_CACHE_TTL_SECONDS = 3_600
LOCK_TTL_SECONDS = 30
LOCK_WAIT_SECONDS = 2.0


@lru_cache(maxsize=4)
def _redis_client(redis_url: str):
    from redis import Redis

    return Redis.from_url(redis_url, decode_responses=True)


def _canonical_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return str(arguments or "")
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cache_key(*, run_id: int, tool_name: str, arguments: str) -> str:
    digest = hashlib.sha256(
        f"{tool_name}\0{_canonical_arguments(arguments)}".encode("utf-8")
    ).hexdigest()
    return f"{CACHE_PREFIX}:{run_id}:{digest}"


def _cache_ttl_seconds() -> int:
    try:
        return max(60, int(os.getenv("AGENT_TOOL_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS))))
    except ValueError:
        return DEFAULT_CACHE_TTL_SECONDS


def _encode(value: Any) -> str | None:
    try:
        return json.dumps({"value": value}, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _decode(value: str | bytes | None) -> tuple[bool, Any]:
    if value is None:
        return False, None
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return False, None
    if not isinstance(payload, dict) or "value" not in payload:
        return False, None
    return True, payload["value"]


def _cacheable_result(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized.startswith("an error occurred while running the tool"):
            return False
    return isinstance(value, (dict, list, str, int, float, bool)) or value is None


def _record_cache_event(context: Any, *, tool_name: str, hit: bool) -> None:
    run_context = getattr(context, "context", None)
    add_event = getattr(run_context, "add_decision_trace", None)
    if callable(add_event):
        add_event(
            "tool_cache_hit" if hit else "tool_cache_miss",
            tool=tool_name,
            status="completed",
        )


def trace_cached_tool(tool: Any) -> Any:
    """Wrap a read-only FunctionTool with a Redis cache scoped to one run."""

    if not isinstance(tool, FunctionTool) or getattr(tool, "_trace_cache_wrapped", False):
        return tool

    original = tool.on_invoke_tool

    async def invoke(context: Any, arguments: str) -> Any:
        run_context = getattr(context, "context", None)
        run_id = getattr(run_context, "run_id", None)
        if run_id is None:
            return await original(context, arguments)

        try:
            client = _redis_client(get_settings().redis_url)
            key = _cache_key(
                run_id=int(run_id),
                tool_name=tool.name,
                arguments=arguments,
            )
            found, cached = _decode(client.get(key))
            if found:
                _record_cache_event(context, tool_name=tool.name, hit=True)
                return cached
        except Exception:
            logger.warning(
                "trace_tool_cache_read_failed run_id=%s tool=%s",
                run_id,
                tool.name,
                exc_info=True,
            )
            return await original(context, arguments)

        lock_key = f"{key}:lock"
        lock_token = uuid4().hex
        acquired = False
        try:
            acquired = bool(client.set(lock_key, lock_token, nx=True, ex=LOCK_TTL_SECONDS))
            if not acquired:
                deadline = asyncio.get_running_loop().time() + LOCK_WAIT_SECONDS
                while asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.025)
                    found, cached = _decode(client.get(key))
                    if found:
                        _record_cache_event(context, tool_name=tool.name, hit=True)
                        return cached

        except Exception:
            logger.warning(
                "trace_tool_cache_lock_failed run_id=%s tool=%s",
                run_id,
                tool.name,
                exc_info=True,
            )
            acquired = False

        _record_cache_event(context, tool_name=tool.name, hit=False)
        try:
            result = await original(context, arguments)
            if _cacheable_result(result):
                encoded = _encode(result)
                if encoded is not None:
                    try:
                        client.setex(key, _cache_ttl_seconds(), encoded)
                    except Exception:
                        logger.warning(
                            "trace_tool_cache_write_failed run_id=%s tool=%s",
                            run_id,
                            tool.name,
                            exc_info=True,
                        )
            return result
        finally:
            if acquired:
                try:
                    if client.get(lock_key) == lock_token:
                        client.delete(lock_key)
                except Exception:
                    logger.debug(
                        "trace_tool_cache_unlock_failed run_id=%s tool=%s",
                        run_id,
                        tool.name,
                        exc_info=True,
                    )

    wrapped = replace(tool, on_invoke_tool=invoke)
    setattr(wrapped, "_trace_cache_wrapped", True)
    return wrapped


def trace_cached_tools(tools: list[Any]) -> list[Any]:
    return [trace_cached_tool(tool) for tool in tools]


__all__ = ["trace_cached_tool", "trace_cached_tools"]
