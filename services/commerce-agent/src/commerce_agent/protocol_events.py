"""Normalize trusted LangGraph v3 events for the durable frontend protocol.

LangGraph's event sequence is local to a stream. PostgreSQL assigns the
thread-wide replay sequence, so the source sequence must never be forwarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from commerce_agent.protocol_store import ProtocolRecordError
from commerce_agent.safety import assert_secret_free


def protocol_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 64:
        raise ProtocolRecordError("graph event exceeds nesting limit")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return protocol_json_value(value.model_dump(mode="json"), depth=depth + 1)
    if isinstance(value, Enum):
        return protocol_json_value(value.value, depth=depth + 1)
    if isinstance(value, (UUID, date, datetime, time, Decimal)):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return protocol_json_value(
            {item.name: getattr(value, item.name) for item in fields(value)},
            depth=depth + 1,
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProtocolRecordError("graph event object keys must be strings")
        return {
            key: protocol_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [protocol_json_value(item, depth=depth + 1) for item in value]
    raise ProtocolRecordError(f"unsupported graph event value: {type(value).__name__}")


def normalize_graph_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON event, dropping stream-local IDs before durable append.

    Call only for events produced by the guarded local graph, never for browser
    payloads. The store performs a second safety/envelope check before insert.
    """

    if not isinstance(raw, Mapping):
        raise ProtocolRecordError("graph event must be an object")
    assert_secret_free(raw, path="graph.event")
    if raw.get("type") != "event" or not isinstance(raw.get("params"), Mapping):
        raise ProtocolRecordError("invalid graph event envelope")
    params = raw["params"]
    namespace = params.get("namespace")
    if not isinstance(namespace, (list, tuple)):
        raise ProtocolRecordError("graph event namespace must be a list")
    timestamp = params.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise ProtocolRecordError("graph event timestamp must be epoch milliseconds")
    data = params.get("data")
    # v3 messages/tools carry (payload, metadata). The frontend consumes the
    # payload; source metadata is transport-internal and not a state field.
    if raw.get("method") in {"messages", "tools"} and isinstance(data, tuple):
        if len(data) != 2:
            raise ProtocolRecordError("invalid message/tool event data")
        data = data[0]
    event = {
        "type": "event",
        "method": raw.get("method"),
        "params": {
            "namespace": protocol_json_value(namespace),
            "timestamp": timestamp,
            "data": protocol_json_value(data),
        },
    }
    if "interrupts" in params:
        interrupts = params["interrupts"]
        if not isinstance(interrupts, (list, tuple)):
            raise ProtocolRecordError("graph event interrupts must be a list")
        event["params"]["interrupts"] = protocol_json_value(interrupts)
    assert_secret_free(event, path="protocol.event")
    return event


__all__ = ["normalize_graph_event", "protocol_json_value"]
