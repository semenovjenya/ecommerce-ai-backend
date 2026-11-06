from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_langsmith_environment_cache():
    """LangSmith caches env lookups across tests that patch tracing settings."""

    from langsmith.utils import get_env_var

    get_env_var.cache_clear()
    try:
        yield
    finally:
        get_env_var.cache_clear()
