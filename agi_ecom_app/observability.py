from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping, TypeVar


logger = logging.getLogger(__name__)

_SECRET_KEY_RE = re.compile(
    r"(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"(?:^|[._-])token$|password|secret|cookie|dsn)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_VALUE_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\b\s*[:=]\s*)"
    r"([^\s,;\}\]]+)",
)
_DSN_PASSWORD_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/\s]+:)([^@/\s]+)(@)")
_KNOWN_SECRET_RE = re.compile(r"\b(?:sk|pk)-(?:proj-|lf-)?[A-Za-z0-9_-]{12,}\b")
_T = TypeVar("_T")


@dataclass
class _ObservabilityState:
    pid: int | None = None
    client: Any | None = None
    tracer_provider: Any | None = None
    instrumentor: Any | None = None
    service_name: str = ""
    worker_role: str = ""
    enabled: bool = False


_state = _ObservabilityState()
_state_lock = threading.RLock()


@dataclass
class _DirectRootObservation:
    """Small adapter for a real root observation with a caller-provided trace id."""

    client: Any
    trace_id: str
    id: str
    name: str
    environment: str
    release: str | None
    version: str | None
    input: Any
    metadata: Mapping[str, Any]
    session_id: str | None
    tags: list[str]
    start_time_ns: int
    pending: dict[str, Any]

    def update(self, **kwargs: Any) -> None:
        for key in ("output", "metadata", "level", "status_message"):
            if key in kwargs:
                self.pending[key] = kwargs[key]

    def finish(self, *, error: BaseException | None = None, max_bytes: int) -> None:
        try:
            from langfuse._client.attributes import (
                LangfuseOtelSpanAttributes,
                create_span_attributes,
                create_trace_attributes,
            )
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.trace import Status, StatusCode

            if error is not None:
                self.pending.update(
                    level="ERROR",
                    status_message=f"{type(error).__name__}: {error}",
                )
            output = _safe_direct_payload(
                self.pending.get("output"),
                max_bytes=max_bytes,
            )
            metadata = {
                **dict(self.metadata),
                **dict(self.pending.get("metadata") or {}),
            }
            level = self.pending.get("level")
            status_message = self.pending.get("status_message")
            attributes = {
                **create_span_attributes(
                    metadata=_safe_direct_payload(metadata, max_bytes=max_bytes),
                    input=_safe_direct_payload(self.input, max_bytes=max_bytes),
                    output=output,
                    level=level,
                    status_message=status_message,
                    version=self.version,
                    observation_type="span",
                ),
                **create_trace_attributes(
                    input=_safe_direct_payload(self.input, max_bytes=max_bytes),
                    output=output,
                    public=False,
                ),
                LangfuseOtelSpanAttributes.TRACE_NAME: (
                    "agent-run" if self.name == "agent-run-intake" else self.name
                ),
                LangfuseOtelSpanAttributes.ENVIRONMENT: self.environment,
            }
            if self.session_id:
                attributes[LangfuseOtelSpanAttributes.TRACE_SESSION_ID] = self.session_id
            if self.tags:
                attributes[LangfuseOtelSpanAttributes.TRACE_TAGS] = self.tags
            if self.release:
                attributes[LangfuseOtelSpanAttributes.RELEASE] = self.release
            context = otel_trace.SpanContext(
                trace_id=int(self.trace_id, 16),
                span_id=int(self.id, 16),
                trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
                is_remote=False,
            )
            status = (
                Status(StatusCode.ERROR, str(status_message or type(error).__name__))
                if error is not None
                else Status(StatusCode.UNSET)
            )
            span = ReadableSpan(
                name=self.name,
                context=context,
                parent=None,
                resource=_state.tracer_provider.resource,
                attributes=attributes,
                status=status,
                start_time=self.start_time_ns,
                end_time=time.time_ns(),
                instrumentation_scope=self.client._otel_tracer._instrumentation_scope,
            )
            _state.tracer_provider._active_span_processor.on_end(span)
        except Exception:
            logger.exception(
                "langfuse_root_observation_export_failed trace_id=%s pid=%s",
                self.trace_id,
                os.getpid(),
            )


def init_langfuse(
    *,
    service_name: str | None = None,
    worker_role: str | None = None,
    settings: Any | None = None,
) -> Any | None:
    """Initialize one Langfuse/OTel client in the process that performs LLM work.

    The function is deliberately lazy. RQ parent processes can import application
    modules without creating exporter threads that would be invalid after fork.
    """

    from agi_ecom_app.config import get_settings

    resolved_settings = settings or get_settings()
    if not bool(getattr(resolved_settings, "langfuse_enabled", False)):
        return None

    pid = os.getpid()
    public_key = getattr(resolved_settings, "langfuse_public_key", None)
    secret_key = getattr(resolved_settings, "langfuse_secret_key", None)
    explicit_credentials = settings is not None and (
        hasattr(settings, "langfuse_public_key")
        or hasattr(settings, "langfuse_secret_key")
    )
    with _state_lock:
        if _state.pid == pid:
            if explicit_credentials and (not public_key or not secret_key):
                logger.warning(
                    "langfuse_disabled_missing_credentials service=%s environment=%s",
                    service_name or os.getenv("LANGFUSE_SERVICE_NAME", "application"),
                    getattr(
                        resolved_settings,
                        "langfuse_tracing_environment",
                        "development",
                    ),
                )
                return None
            return _state.client if _state.enabled else None

        inherited_process = _state.pid is not None and _state.pid != pid
        if inherited_process:
            _reset_inherited_sdk_state()
        _reset_local_state(pid)

        if not public_key or not secret_key:
            logger.warning(
                "langfuse_disabled_missing_credentials service=%s environment=%s",
                service_name or os.getenv("LANGFUSE_SERVICE_NAME", "application"),
                getattr(resolved_settings, "langfuse_tracing_environment", "development"),
            )
            return None

        client: Any | None = None
        try:
            from langfuse import Langfuse
            from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

            sample_rate = float(getattr(resolved_settings, "langfuse_sample_rate", 1.0))
            environment = str(
                getattr(resolved_settings, "langfuse_tracing_environment", "development")
            )
            release = getattr(resolved_settings, "langfuse_release", None)
            resource = Resource.create(
                {
                    "service.name": service_name
                    or os.getenv("LANGFUSE_SERVICE_NAME", "ai-ecom"),
                    "langfuse.environment": environment,
                    **({"langfuse.release": release} if release else {}),
                }
            )
            tracer_provider = TracerProvider(
                resource=resource,
                sampler=TraceIdRatioBased(sample_rate),
            )
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=str(
                    getattr(
                        resolved_settings,
                        "langfuse_base_url",
                        "https://cloud.langfuse.com",
                    )
                ),
                timeout=5,
                tracing_enabled=True,
                flush_at=64,
                flush_interval=1.0,
                environment=environment,
                release=release,
                sample_rate=sample_rate,
                mask_otel_spans=mask_otel_spans,
                tracer_provider=tracer_provider,
            )
            instrumentor = OpenAIAgentsInstrumentor()
            if inherited_process:
                try:
                    instrumentor.uninstrument()
                except Exception:
                    logger.debug("langfuse_inherited_instrumentor_reset_failed", exc_info=True)
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                exclusive_processor=not bool(
                    getattr(
                        resolved_settings,
                        "langfuse_openai_agents_dual_export",
                        True,
                    )
                ),
            )
        except Exception:
            logger.exception(
                "langfuse_initialization_failed service=%s pid=%s",
                service_name or os.getenv("LANGFUSE_SERVICE_NAME", "application"),
                pid,
            )
            if client is not None:
                try:
                    client.shutdown()
                except Exception:
                    logger.debug("langfuse_failed_client_cleanup_failed", exc_info=True)
            return None

        _state.client = client
        _state.tracer_provider = tracer_provider
        _state.instrumentor = instrumentor
        _state.service_name = service_name or os.getenv("LANGFUSE_SERVICE_NAME", "ai-ecom")
        _state.worker_role = worker_role or os.getenv("LANGFUSE_WORKER_ROLE", "process")
        _state.enabled = True
        logger.info(
            "langfuse_initialized service=%s worker_role=%s environment=%s export_mode=%s pid=%s",
            _state.service_name,
            _state.worker_role,
            environment,
            "dual" if getattr(resolved_settings, "langfuse_openai_agents_dual_export", True) else "langfuse",
            pid,
        )
        return client


def ensure_langfuse_initialized(*, settings: Any | None = None) -> Any | None:
    return init_langfuse(
        service_name=os.getenv("LANGFUSE_SERVICE_NAME", "ai-ecom"),
        worker_role=os.getenv("LANGFUSE_WORKER_ROLE", "process"),
        settings=settings,
    )


def flush_langfuse() -> None:
    client = _current_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        logger.exception("langfuse_flush_failed pid=%s", os.getpid())


def shutdown_langfuse() -> None:
    with _state_lock:
        client = _current_client()
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:
            logger.exception("langfuse_shutdown_failed pid=%s", os.getpid())
        finally:
            _reset_local_state(None)


def create_trace_id(*, seed: str | None = None, settings: Any | None = None) -> str | None:
    if ensure_langfuse_initialized(settings=settings) is None:
        return None
    if seed:
        return hashlib.sha256(seed.encode("utf-8")).digest()[:16].hex()
    return secrets.token_hex(16)


def create_observation_id(*, seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).digest()[:8].hex()


def agent_trace_id(
    *,
    chat_id: str,
    client_message_id: str | None,
    settings: Any | None = None,
) -> str | None:
    seed = (
        f"agent-message:{chat_id}:{client_message_id}"
        if client_message_id
        else None
    )
    return create_trace_id(seed=seed, settings=settings)


def observability_summary(observation: Any | None, *, settings: Any | None = None) -> dict[str, Any]:
    if observation is None:
        return {}
    from agi_ecom_app.config import get_settings

    resolved_settings = settings or get_settings()
    return {
        "provider": "langfuse",
        "trace_id": str(observation.trace_id),
        "root_observation_id": str(observation.id),
        "capture_mode": str(getattr(resolved_settings, "langfuse_capture_mode", "full")),
        "export_mode": (
            "dual"
            if bool(getattr(resolved_settings, "langfuse_openai_agents_dual_export", True))
            else "langfuse"
        ),
    }


def trace_context_for_agent_run(
    *, summary: Mapping[str, Any] | None, run_id: int
) -> tuple[str, str]:
    raw = (summary or {}).get("observability")
    data = raw if isinstance(raw, Mapping) else {}
    trace_id = str(data.get("trace_id") or "")
    parent_id = str(data.get("root_observation_id") or "")
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        trace_id = hashlib.sha256(f"agent-run:{run_id}".encode("utf-8")).digest()[:16].hex()
    if not re.fullmatch(r"[0-9a-f]{16}", parent_id):
        parent_id = create_observation_id(seed=f"agent-run-root:{run_id}")
    return trace_id, parent_id


def _safe_direct_payload(value: Any, *, max_bytes: int) -> Any:
    masked = _mask_json(value)
    if isinstance(masked, Mapping):
        return {
            str(key): _safe_direct_payload(item, max_bytes=max_bytes)
            for key, item in masked.items()
        }
    if isinstance(masked, list):
        return [_safe_direct_payload(item, max_bytes=max_bytes) for item in masked]
    if isinstance(masked, tuple):
        return tuple(_safe_direct_payload(item, max_bytes=max_bytes) for item in masked)
    if isinstance(masked, str):
        return _truncate_string(masked, max_bytes=max_bytes)
    return masked


def _start_direct_root_observation(
    *,
    client: Any,
    trace_id: str,
    name: str,
    as_type: str,
    input: Any,
    metadata: Mapping[str, Any],
    session_id: str | None,
    tags: list[str],
    version: str | None,
    settings: Any,
) -> _DirectRootObservation | None:
    """Prepare a real root span, avoiding the SDK's synthetic parent span."""

    try:
        environment = str(
            getattr(settings, "langfuse_tracing_environment", "development")
        )
        release = getattr(settings, "langfuse_release", None)
        max_bytes = int(getattr(settings, "langfuse_max_field_bytes", 262_144))
        observation_id = create_observation_id(seed=f"langfuse-root:{trace_id}:{name}")
        safe_input = _safe_direct_payload(input, max_bytes=max_bytes)
        safe_metadata = _safe_direct_payload(
            {**dict(metadata), "root_requested_type": str(as_type or "span")},
            max_bytes=max_bytes,
        )
        return _DirectRootObservation(
            client=client,
            trace_id=trace_id,
            id=observation_id,
            name=name,
            environment=environment,
            release=str(release) if release else None,
            version=version,
            input=safe_input,
            metadata=safe_metadata,
            session_id=session_id,
            tags=tags,
            start_time_ns=time.time_ns(),
            pending={},
        )
    except Exception:
        logger.exception(
            "langfuse_root_observation_create_failed name=%s pid=%s",
            name,
            os.getpid(),
        )
        return None


@contextmanager
def _direct_root_observation_scope(
    *,
    observation: _DirectRootObservation,
    metadata: Mapping[str, Any],
    session_id: str | None,
    tags: list[str],
    version: str | None,
    max_bytes: int,
) -> Iterator[_DirectRootObservation | None]:
    propagation: Any | None = None
    parent_context: Any | None = None
    body_error: BaseException | None = None
    try:
        from langfuse import propagate_attributes
        from opentelemetry import trace as otel_trace

        span_context = otel_trace.SpanContext(
            trace_id=int(observation.trace_id, 16),
            span_id=int(observation.id, 16),
            trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
            is_remote=True,
        )
        parent_context = otel_trace.use_span(
            otel_trace.NonRecordingSpan(span_context),
            end_on_exit=False,
        )
        parent_context.__enter__()
        propagation = propagate_attributes(
            session_id=session_id,
            tags=tags,
            metadata=metadata,
            version=version,
        )
        propagation.__enter__()
    except Exception as exc:
        logger.exception(
            "langfuse_root_context_failed trace_id=%s pid=%s",
            observation.trace_id,
            os.getpid(),
        )
        observation.finish(error=exc, max_bytes=max_bytes)
        if parent_context is not None:
            try:
                parent_context.__exit__(None, None, None)
            except Exception:
                logger.debug("langfuse_root_parent_cleanup_failed", exc_info=True)
        yield None
        return
    try:
        yield observation
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        observation.finish(error=body_error, max_bytes=max_bytes)
        if propagation is not None:
            try:
                propagation.__exit__(None, None, None)
            except Exception:
                logger.debug("langfuse_root_propagation_cleanup_failed", exc_info=True)
        if parent_context is not None:
            try:
                parent_context.__exit__(None, None, None)
            except Exception:
                logger.debug("langfuse_root_parent_cleanup_failed", exc_info=True)


@contextmanager
def observation_scope(
    *,
    name: str,
    as_type: str = "span",
    trace_id: str | None = None,
    parent_observation_id: str | None = None,
    input: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    version: str | None = None,
    model: str | None = None,
    root: bool = False,
    settings: Any | None = None,
) -> Iterator[Any | None]:
    client = ensure_langfuse_initialized(settings=settings)
    if client is None:
        yield None
        return

    base_metadata = _base_metadata(metadata)
    resolved_tags = list(
        dict.fromkeys(tags or [_state.service_name, _state.worker_role])
    )
    if root and trace_id and not parent_observation_id:
        from agi_ecom_app.config import get_settings

        resolved_settings = settings or get_settings()
        direct_observation = _start_direct_root_observation(
            client=client,
            trace_id=trace_id,
            name=name,
            as_type=as_type,
            input=input,
            metadata=base_metadata,
            session_id=session_id,
            tags=resolved_tags,
            version=version,
            settings=resolved_settings,
        )
        if direct_observation is None:
            yield None
            return
        with _direct_root_observation_scope(
            observation=direct_observation,
            metadata=base_metadata,
            session_id=session_id,
            tags=resolved_tags,
            version=version,
            max_bytes=int(
                getattr(resolved_settings, "langfuse_max_field_bytes", 262_144)
            ),
        ) as observation:
            yield observation
        return

    propagation: Any | None = None
    observation_context: Any | None = None
    try:
        from langfuse import propagate_attributes

        trace_context = None
        if trace_id:
            trace_context = {"trace_id": trace_id}
            if parent_observation_id:
                trace_context["parent_span_id"] = parent_observation_id
        propagation = propagate_attributes(
            session_id=session_id,
            tags=resolved_tags,
            metadata=base_metadata,
            version=version,
        )
        propagation.__enter__()
        observation_context = client.start_as_current_observation(
            trace_context=trace_context,
            name=name,
            as_type=as_type,
            input=input,
            metadata=base_metadata,
            version=version,
            model=model,
        )
        observation = observation_context.__enter__()
    except Exception:
        logger.exception("langfuse_observation_failed name=%s pid=%s", name, os.getpid())
        if propagation is not None:
            try:
                propagation.__exit__(None, None, None)
            except Exception:
                logger.debug("langfuse_propagation_cleanup_failed", exc_info=True)
        yield None
        return

    body_exception: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield observation
    except BaseException:
        body_exception = sys.exc_info()
        raise
    finally:
        try:
            observation_context.__exit__(*body_exception)
        except Exception:
            logger.exception("langfuse_observation_close_failed name=%s pid=%s", name, os.getpid())
        try:
            propagation.__exit__(*body_exception)
        except Exception:
            logger.exception("langfuse_propagation_close_failed name=%s pid=%s", name, os.getpid())


@contextmanager
def generation_scope(
    *,
    name: str,
    model: str,
    input: Any,
    metadata: Mapping[str, Any] | None = None,
    settings: Any | None = None,
) -> Iterator[Any | None]:
    with observation_scope(
        name=name,
        as_type="generation",
        input=input,
        metadata=metadata,
        model=model,
        settings=settings,
    ) as observation:
        yield observation


def update_observation(
    observation: Any | None,
    *,
    output: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
    response: Any | None = None,
) -> None:
    if observation is None:
        return
    try:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = _base_metadata(metadata)
        usage = _response_usage(response)
        if usage:
            kwargs["usage_details"] = usage
        if error is not None:
            kwargs.update(
                level="ERROR",
                status_message=f"{type(error).__name__}: {error}",
            )
        observation.update(**kwargs)
    except Exception:
        logger.exception("langfuse_observation_update_failed pid=%s", os.getpid())


def response_payload(response: Any | None) -> Any:
    if response is None:
        return None
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
    except Exception:
        logger.debug("langfuse_response_serialization_failed", exc_info=True)
    return {"output_text": getattr(response, "output_text", None)}


def trace_simulation_job(func: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(func)
    def wrapped(simulation_run_id: str, *args: Any, **kwargs: Any) -> _T:
        trace_id = create_trace_id(seed=f"decision-simulation:{simulation_run_id}")
        try:
            with observation_scope(
                name="decision-simulation",
                as_type="chain",
                trace_id=trace_id,
                input={"simulation_run_id": simulation_run_id},
                metadata={"simulation_run_id": simulation_run_id},
                tags=["decision-simulation", _state.service_name, _state.worker_role],
                root=True,
            ) as observation:
                try:
                    result = func(simulation_run_id, *args, **kwargs)
                except Exception as exc:
                    update_observation(observation, error=exc)
                    raise
                update_observation(observation, output=result)
                return result
        finally:
            flush_langfuse()

    return wrapped


def mask_otel_spans(*, params: Any) -> Any:
    """Redact secrets and cap large attributes at the final export boundary."""

    from agi_ecom_app.config import get_settings
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    max_bytes = int(getattr(get_settings(), "langfuse_max_field_bytes", 262_144))
    patches: dict[Any, Any] = {}
    for identifier, span in params.spans.items():
        changed: dict[str, Any] = {}
        for key, value in span.attributes.items():
            try:
                masked = _mask_attribute(key, value, max_bytes=max_bytes)
            except Exception:
                logger.exception("langfuse_mask_attribute_failed attribute=%s", key)
                masked = "[REDACTION_ERROR]"
            if masked != value:
                changed[key] = masked
        if changed:
            patches[identifier] = OtelSpanPatch(
                set_attributes={**changed, "app.masking.applied": True}
            )
    return MaskOtelSpansResult(span_patches=patches)


def _mask_attribute(key: str, value: Any, *, max_bytes: int) -> Any:
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        masked = _mask_string(value)
        return _truncate_string(masked, max_bytes=max_bytes)
    if isinstance(value, tuple):
        return tuple(
            _truncate_string(_mask_string(item), max_bytes=max_bytes)
            if isinstance(item, str)
            else item
            for item in value
        )
    if isinstance(value, list):
        return tuple(
            _truncate_string(_mask_string(item), max_bytes=max_bytes)
            if isinstance(item, str)
            else item
            for item in value
        )
    return value


def _mask_string(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        return json.dumps(_mask_json(parsed), ensure_ascii=False, separators=(",", ":"))
    masked = _BEARER_RE.sub("Bearer [REDACTED]", value)
    masked = _KEY_VALUE_RE.sub(r"\1[REDACTED]", masked)
    masked = _DSN_PASSWORD_RE.sub(r"\1[REDACTED]\3", masked)
    return _KNOWN_SECRET_RE.sub("[REDACTED]", masked)


def _mask_json(value: Any, *, key: str | None = None) -> Any:
    if key and _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _mask_json(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_mask_json(item) for item in value]
    if isinstance(value, str):
        return _mask_string(value)
    return value


def _truncate_string(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    prefix_bytes = max_bytes
    while prefix_bytes >= 0:
        prefix = encoded[:prefix_bytes].decode("utf-8", errors="ignore")
        result = json.dumps(
            {
                "truncated": True,
                "original_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "prefix": prefix,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(result.encode("utf-8")) <= max_bytes:
            return result
        prefix_bytes = prefix_bytes // 2 if prefix_bytes else -1
    return "[TRUNCATED]"


def _base_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "service": _state.service_name or os.getenv("LANGFUSE_SERVICE_NAME", "ai-ecom"),
        "worker_role": _state.worker_role or os.getenv("LANGFUSE_WORKER_ROLE", "process"),
    }
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value[:200] if isinstance(value, str) else value
    return result


def _response_usage(response: Any | None) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, Mapping):
        data = dict(usage)
    else:
        data = {
            key: getattr(usage, key, None)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
    result: dict[str, int] = {}
    for source, target in (
        ("input_tokens", "input"),
        ("prompt_tokens", "input"),
        ("output_tokens", "output"),
        ("completion_tokens", "output"),
        ("total_tokens", "total"),
    ):
        raw = data.get(source)
        if raw is not None and target not in result:
            result[target] = int(raw)
    return result


def _current_client() -> Any | None:
    if _state.pid != os.getpid() or not _state.enabled:
        return None
    return _state.client


def _reset_local_state(pid: int | None) -> None:
    _state.pid = pid
    _state.client = None
    _state.tracer_provider = None
    _state.instrumentor = None
    _state.service_name = ""
    _state.worker_role = ""
    _state.enabled = False


def _reset_inherited_sdk_state() -> None:
    try:
        from langfuse._client.resource_manager import LangfuseResourceManager

        LangfuseResourceManager._instances.clear()
    except Exception:
        logger.debug("langfuse_inherited_resource_reset_failed", exc_info=True)


__all__ = [
    "agent_trace_id",
    "create_observation_id",
    "create_trace_id",
    "ensure_langfuse_initialized",
    "flush_langfuse",
    "generation_scope",
    "init_langfuse",
    "mask_otel_spans",
    "observation_scope",
    "observability_summary",
    "response_payload",
    "shutdown_langfuse",
    "trace_context_for_agent_run",
    "trace_simulation_job",
    "update_observation",
]
