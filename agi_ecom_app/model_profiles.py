from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelProvider = Literal["openai", "openrouter"]
ModelApiStyle = Literal["responses"]

OPENAI_DEFAULT_PROFILE = "openai_default"
OPENROUTER_GEMMA_FREE_PROFILE = "openrouter_gemma_free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"


@dataclass(frozen=True)
class ModelProfile:
    profile: str
    provider: ModelProvider
    model: str
    api_style: ModelApiStyle
    base_url: str | None
    required_env_vars: tuple[str, ...]
    experimental: bool = False
    description: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.provider}:{self.model}"


def build_model_profiles(settings) -> dict[str, ModelProfile]:
    openai_model = getattr(settings, "openai_model", "gpt-4.1-mini")
    openrouter_model = getattr(settings, "openrouter_model", DEFAULT_OPENROUTER_MODEL)
    openrouter_base_url = getattr(settings, "openrouter_base_url", DEFAULT_OPENROUTER_BASE_URL)
    return {
        OPENAI_DEFAULT_PROFILE: ModelProfile(
            profile=OPENAI_DEFAULT_PROFILE,
            provider="openai",
            model=openai_model,
            api_style="responses",
            base_url=None,
            required_env_vars=("OPENAI_API_KEY",),
            experimental=False,
            description="Stable OpenAI Responses API profile.",
        ),
        OPENROUTER_GEMMA_FREE_PROFILE: ModelProfile(
            profile=OPENROUTER_GEMMA_FREE_PROFILE,
            provider="openrouter",
            model=openrouter_model,
            api_style="responses",
            base_url=openrouter_base_url,
            required_env_vars=("OPENROUTER_API_KEY",),
            experimental=True,
            description="Experimental OpenRouter free Gemma profile.",
        ),
    }


def get_model_profile(profile: str, settings) -> ModelProfile:
    profiles = build_model_profiles(settings)
    if profile not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown model profile `{profile}`. Known profiles: {known}.")
    return profiles[profile]


def required_env_values_missing(profile: ModelProfile, settings) -> tuple[str, ...]:
    missing: list[str] = []
    for env_var in profile.required_env_vars:
        value = _env_value_from_settings(env_var, settings)
        if not value:
            missing.append(env_var)
    return tuple(missing)


def describe_model_profile(profile: ModelProfile, settings) -> dict[str, object]:
    missing_env_vars = required_env_values_missing(profile, settings)
    return {
        "profile": profile.profile,
        "provider": profile.provider,
        "model": profile.model,
        "api_style": profile.api_style,
        "base_url": profile.base_url,
        "required_env_vars": list(profile.required_env_vars),
        "missing_env_vars": list(missing_env_vars),
        "configured": not missing_env_vars,
        "experimental": profile.experimental,
        "description": profile.description,
    }


def _env_value_from_settings(env_var: str, settings) -> str | None:
    if env_var == "OPENAI_API_KEY":
        return getattr(settings, "openai_api_key", None)
    if env_var == "OPENROUTER_API_KEY":
        return getattr(settings, "openrouter_api_key", None)
    return None
