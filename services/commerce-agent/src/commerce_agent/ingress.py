from __future__ import annotations

from collections.abc import Mapping
import inspect
from typing import Any

from commerce_agent.safety import assert_secret_free


class IngressConfigurationError(ValueError):
    """An invocation attempted to override the graph's trusted runtime config."""


_ALLOWED_CONFIG_KEYS = frozenset({"configurable", "run_id"})
_ALLOWED_CHECKPOINT_KEYS = frozenset(
    {"thread_id", "checkpoint_id", "checkpoint_ns"}
)


def _validate_runtime_config(config: Any) -> None:
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise IngressConfigurationError("graph config must be a mapping")
    if set(config) - _ALLOWED_CONFIG_KEYS:
        raise IngressConfigurationError("invocation may not override graph tracing config")
    configurable = config.get("configurable")
    if configurable is not None:
        if not isinstance(configurable, Mapping):
            raise IngressConfigurationError("checkpoint config must be a mapping")
        if set(configurable) - _ALLOWED_CHECKPOINT_KEYS:
            raise IngressConfigurationError("unsupported checkpoint config field")


class GuardedGraph:
    """Validate a request before LangGraph invokes callbacks or writes checkpoints.

    The self-managed service must use this facade, not a raw compiled graph.
    It deliberately exposes only the operations needed by the M1 runtime.
    """

    def __init__(self, compiled_graph: Any) -> None:
        self._compiled_graph = compiled_graph

    def _validate(
        self,
        input: Any,
        *,
        config: Any = None,
        context: Any = None,
        options: Any = None,
    ) -> None:
        assert_secret_free(input, path="request.input")
        assert_secret_free(config, path="request.config")
        assert_secret_free(context, path="request.context")
        assert_secret_free(options, path="request.options")
        _validate_runtime_config(config)

    def invoke(self, input: Any, *, config: Any = None, context: Any = None, **kwargs: Any) -> Any:
        self._validate(input, config=config, context=context, options=kwargs)
        return self._compiled_graph.invoke(input, config=config, context=context, **kwargs)

    async def ainvoke(
        self, input: Any, *, config: Any = None, context: Any = None, **kwargs: Any
    ) -> Any:
        self._validate(input, config=config, context=context, options=kwargs)
        return await self._compiled_graph.ainvoke(
            input, config=config, context=context, **kwargs
        )

    def stream(self, input: Any, *, config: Any = None, context: Any = None, **kwargs: Any) -> Any:
        self._validate(input, config=config, context=context, options=kwargs)
        return self._compiled_graph.stream(input, config=config, context=context, **kwargs)

    def astream(self, input: Any, *, config: Any = None, context: Any = None, **kwargs: Any) -> Any:
        self._validate(input, config=config, context=context, options=kwargs)
        return self._compiled_graph.astream(input, config=config, context=context, **kwargs)

    async def astream_protocol_events(
        self, input: Any, *, config: Any = None, context: Any = None, **kwargs: Any
    ) -> Any:
        """Start a v3 protocol stream only after the pre-trace ingress guard."""

        self._validate(input, config=config, context=context, options=kwargs)
        if kwargs:
            raise IngressConfigurationError("protocol stream options are service-owned")
        result = self._compiled_graph.astream_events(
            input, config=config, context=context, version="v3"
        )
        return await result if inspect.isawaitable(result) else result

    def get_state(self, config: Any, **kwargs: Any) -> Any:
        assert_secret_free({"config": config, "options": kwargs}, path="request")
        _validate_runtime_config(config)
        return self._compiled_graph.get_state(config, **kwargs)

    async def aget_state(self, config: Any, **kwargs: Any) -> Any:
        assert_secret_free({"config": config, "options": kwargs}, path="request")
        _validate_runtime_config(config)
        return await self._compiled_graph.aget_state(config, **kwargs)

    def get_state_history(self, config: Any, **kwargs: Any) -> Any:
        assert_secret_free({"config": config, "options": kwargs}, path="request")
        _validate_runtime_config(config)
        return self._compiled_graph.get_state_history(config, **kwargs)

    def aget_state_history(self, config: Any, **kwargs: Any) -> Any:
        assert_secret_free({"config": config, "options": kwargs}, path="request")
        _validate_runtime_config(config)
        return self._compiled_graph.aget_state_history(config, **kwargs)

    def flush_traces(self, *, timeout: float = 30.0) -> None:
        from commerce_agent.observability import flush_langsmith_traces

        flush_langsmith_traces(self._compiled_graph, timeout=timeout)


__all__ = ["GuardedGraph", "IngressConfigurationError"]
