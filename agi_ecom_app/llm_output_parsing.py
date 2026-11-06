from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def coerce_model_from_json_text(value: Any, model_type: type[T]) -> T:
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(extract_model_payload(value, model_type))


def extract_model_payload(value: Any, model_type: type[BaseModel]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return _unwrap_model_payload(value, model_type)
    text = str(value or "").strip()
    payload = extract_json_object(text)
    return _unwrap_model_payload(payload, model_type)


def extract_json_object(text: str) -> dict[str, Any]:
    candidates = _json_candidates(text)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError("JSON payload is not an object.")
    raise ValueError(f"Could not extract JSON object from model output: {last_error}")


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _FENCED_JSON_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    object_slice = _first_json_object_slice(stripped)
    if object_slice and object_slice not in candidates:
        candidates.append(object_slice)
    return candidates


def _first_json_object_slice(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _unwrap_model_payload(payload: dict[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    expected_fields = set(model_type.model_fields)
    if expected_fields.intersection(payload):
        return payload
    for key in _model_wrapper_keys(model_type):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    if len(payload) == 1:
        nested = next(iter(payload.values()))
        if isinstance(nested, dict):
            return nested
    return payload


def _model_wrapper_keys(model_type: type[BaseModel]) -> tuple[str, ...]:
    name = model_type.__name__
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return (name, name[:1].lower() + name[1:], snake)
