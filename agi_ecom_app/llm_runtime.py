from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import OpenAIResponsesModel
from openai import AsyncOpenAI, OpenAI

from agi_ecom_app.config import get_settings
from agi_ecom_app.model_profiles import (
    ModelProfile,
    build_model_profiles,
    describe_model_profile,
    get_model_profile,
    required_env_values_missing,
)
from agi_ecom_app.services.runtime_settings import get_active_model_profile_name


class LLMRuntimeConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMRuntimeSpec:
    profile: str
    provider: str
    model_name: str
    api_style: str
    base_url: str | None
    default_headers: tuple[tuple[str, str], ...]
    experimental: bool
    missing_env_vars: tuple[str, ...]

    @property
    def display_name(self) -> str:
        return f"{self.provider}:{self.model_name}"

    @property
    def cache_key(self) -> str:
        headers_key = "|".join(f"{key}={value}" for key, value in self.default_headers)
        return (
            f"{self.profile}:{self.provider}:{self.api_style}:{self.model_name}:"
            f"{self.base_url or ''}:{headers_key}"
        )

    @property
    def supports_agent_structured_outputs(self) -> bool:
        return self.provider == "openai"


@dataclass(frozen=True)
class LLMRuntime:
    spec: LLMRuntimeSpec
    agent_model: Any
    sync_client: OpenAI

    @property
    def profile(self) -> str:
        return self.spec.profile

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    @property
    def cache_key(self) -> str:
        return self.spec.cache_key


def get_available_model_profile_descriptions(settings=None) -> list[dict[str, object]]:
    settings = settings or get_settings()
    return [
        describe_model_profile(profile, settings)
        for profile in build_model_profiles(settings).values()
    ]


def get_active_llm_runtime_spec(*, require_configured: bool = False, settings=None) -> LLMRuntimeSpec:
    settings = settings or get_settings()
    # Lazy initialization keeps RQ parent processes free of exporter threads;
    # the work horse that is about to build/run a model owns the client.
    from agi_ecom_app.observability import ensure_langfuse_initialized

    ensure_langfuse_initialized(settings=settings)
    profile_name = get_active_model_profile_name(settings)
    profile = get_model_profile(profile_name, settings)
    spec = _spec_from_profile(profile, settings)
    if require_configured and spec.missing_env_vars:
        missing = ", ".join(spec.missing_env_vars)
        raise LLMRuntimeConfigurationError(
            f"Model profile `{profile.profile}` requires missing environment variable(s): {missing}."
        )
    return spec


def get_active_llm_runtime(*, settings=None) -> LLMRuntime:
    settings = settings or get_settings()
    spec = get_active_llm_runtime_spec(require_configured=True, settings=settings)
    return LLMRuntime(
        spec=spec,
        agent_model=build_agent_model(spec, settings=settings),
        sync_client=build_sync_client(spec, settings=settings),
    )


def get_active_model_metadata(*, require_configured: bool = False, settings=None) -> dict[str, object]:
    spec = get_active_llm_runtime_spec(require_configured=require_configured, settings=settings)
    return {
        "model_profile": spec.profile,
        "model_provider": spec.provider,
        "model_name": spec.model_name,
        "model_label": spec.display_name,
        "model_experimental": spec.experimental,
        "model_missing_env_vars": list(spec.missing_env_vars),
    }


def build_agent_model(spec: LLMRuntimeSpec, *, settings=None):
    settings = settings or get_settings()
    if spec.provider == "openai":
        return spec.model_name
    return OpenAIResponsesModel(
        model=spec.model_name,
        openai_client=build_async_client(spec, settings=settings),
    )


def build_sync_client(spec: LLMRuntimeSpec, *, settings=None) -> OpenAI:
    settings = settings or get_settings()
    api_key = _api_key_for_provider(spec.provider, settings)
    if not api_key:
        raise LLMRuntimeConfigurationError(
            f"Model profile `{spec.profile}` requires {', '.join(spec.missing_env_vars)}."
        )
    return OpenAI(
        api_key=api_key,
        base_url=spec.base_url,
        default_headers=dict(spec.default_headers) or None,
    )


def build_direct_openai_client(*, api_key: str, settings=None) -> OpenAI:
    settings = settings or get_settings()
    from agi_ecom_app.observability import ensure_langfuse_initialized

    ensure_langfuse_initialized(settings=settings)
    return OpenAI(api_key=api_key)


def build_async_client(spec: LLMRuntimeSpec, *, settings=None) -> AsyncOpenAI:
    settings = settings or get_settings()
    api_key = _api_key_for_provider(spec.provider, settings)
    if not api_key:
        raise LLMRuntimeConfigurationError(
            f"Model profile `{spec.profile}` requires {', '.join(spec.missing_env_vars)}."
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=spec.base_url,
        default_headers=dict(spec.default_headers) or None,
    )


def validate_model_profile_configured(profile_name: str, *, settings=None) -> ModelProfile:
    settings = settings or get_settings()
    profile = get_model_profile(profile_name, settings)
    missing = required_env_values_missing(profile, settings)
    if missing:
        raise LLMRuntimeConfigurationError(
            f"Model profile `{profile.profile}` requires missing environment variable(s): "
            f"{', '.join(missing)}."
        )
    return profile


def _spec_from_profile(profile: ModelProfile, settings) -> LLMRuntimeSpec:
    return LLMRuntimeSpec(
        profile=profile.profile,
        provider=profile.provider,
        model_name=profile.model,
        api_style=profile.api_style,
        base_url=profile.base_url,
        default_headers=_default_headers_for_profile(profile, settings),
        experimental=profile.experimental,
        missing_env_vars=required_env_values_missing(profile, settings),
    )


def _default_headers_for_profile(profile: ModelProfile, settings) -> tuple[tuple[str, str], ...]:
    if profile.provider != "openrouter":
        return ()
    headers: list[tuple[str, str]] = []
    referer = getattr(settings, "openrouter_http_referer", None)
    title = getattr(settings, "openrouter_title", None)
    if referer:
        headers.append(("HTTP-Referer", referer))
    if title:
        headers.append(("X-Title", title))
    return tuple(headers)


def _api_key_for_provider(provider: str, settings) -> str | None:
    if provider == "openai":
        return getattr(settings, "openai_api_key", None)
    if provider == "openrouter":
        return getattr(settings, "openrouter_api_key", None)
    return None
