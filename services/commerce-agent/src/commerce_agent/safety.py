from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel


class SensitiveStateError(ValueError):
    """Raised before a value containing likely credentials can be checkpointed."""


_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credentials",
        "dsn",
        "password",
        "refresh_token",
        "secret",
    }
)
_SENSITIVE_VALUES = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\blsv2_[A-Za-z0-9_]{16,}\b", re.IGNORECASE),
    re.compile(r"\b(?:pk|sk)-lf-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bgh[a-z]_[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b(?:postgres|postgresql|redis)://[^\s]+", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|password|secret)\s*[:=]\s*\S+", re.IGNORECASE),
)


def assert_secret_free(value: Any, *, path: str = "state") -> None:
    if isinstance(value, BaseModel):
        assert_secret_free(value.model_dump(mode="python"), path=path)
        return
    if is_dataclass(value) and not isinstance(value, type):
        assert_secret_free(
            {item.name: getattr(value, item.name) for item in fields(value)},
            path=path,
        )
        return
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            child_path = f"{path}.{key}"
            if key in _SENSITIVE_KEYS or key.endswith("_password") or key.endswith("_secret"):
                raise SensitiveStateError(f"sensitive field is forbidden in durable state: {child_path}")
            assert_secret_free(item, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_secret_free(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise SensitiveStateError(f"binary payload is forbidden in durable state: {path}")
    if isinstance(value, str) and any(pattern.search(value) for pattern in _SENSITIVE_VALUES):
        raise SensitiveStateError(f"credential-like value is forbidden in durable state: {path}")


__all__ = ["SensitiveStateError", "assert_secret_free"]
