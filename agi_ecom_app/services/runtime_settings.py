from __future__ import annotations

from typing import Any

from psycopg.types.json import Json

from agi_ecom_app.config import get_settings
from agi_ecom_app.model_profiles import OPENAI_DEFAULT_PROFILE, get_model_profile

ACTIVE_MODEL_PROFILE_KEY = "active_model_profile"


def get_active_model_profile_name(settings=None) -> str:
    settings = settings or get_settings()
    stored_profile = _load_runtime_profile_from_db()
    if stored_profile:
        return _known_or_default(stored_profile, settings)
    return _known_or_default(getattr(settings, "llm_model_profile", None), settings)


def set_active_model_profile_name(profile: str) -> str:
    settings = get_settings()
    normalized_profile = _known_or_default(profile, settings, fallback_unknown=False)
    from agi_ecom_app.services.memory import ensure_memory_schema, get_memory_connection

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into app_runtime_settings (key, value)
            values (%s, %s)
            on conflict (key) do update
            set value = excluded.value,
                updated_at = now()
            """,
            [ACTIVE_MODEL_PROFILE_KEY, Json({"profile": normalized_profile})],
        )
    return normalized_profile


def _load_runtime_profile_from_db() -> str | None:
    try:
        from agi_ecom_app.services.memory import ensure_memory_schema, get_memory_connection

        ensure_memory_schema()
        with get_memory_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                select value
                from app_runtime_settings
                where key = %s
                """,
                [ACTIVE_MODEL_PROFILE_KEY],
            )
            row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    value: Any = row.get("value")
    if isinstance(value, dict):
        profile = value.get("profile")
        return profile if isinstance(profile, str) else None
    return None


def _known_or_default(profile: str | None, settings, *, fallback_unknown: bool = True) -> str:
    candidate = (profile or OPENAI_DEFAULT_PROFILE).strip() or OPENAI_DEFAULT_PROFILE
    try:
        get_model_profile(candidate, settings)
    except ValueError:
        if fallback_unknown:
            return OPENAI_DEFAULT_PROFILE
        raise
    return candidate
