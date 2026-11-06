from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from commerce_agent.persistence import (
    PersistenceConfigurationError,
    PersistenceSettings,
    initialize_checkpoint_schema,
    open_postgres_checkpointer,
)


class _FakeSaver:
    def __init__(self) -> None:
        self.setup_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1


def _settings() -> PersistenceSettings:
    return PersistenceSettings.from_environment(
        {
            "COMMERCE_AGENT_ENVIRONMENT": "test",
            "COMMERCE_AGENT_DATABASE_URI": (
                "postgresql://user:secret-canary@localhost:5432/commerce_agent"
                "?sslmode=disable"
            ),
        }
    )


def test_settings_require_dedicated_postgres_uri_without_leaking_secret() -> None:
    settings = _settings()

    assert settings.environment == "test"
    assert "secret-canary" not in repr(settings)

    with pytest.raises(PersistenceConfigurationError, match="must be set"):
        PersistenceSettings.from_environment({})
    with pytest.raises(PersistenceConfigurationError, match="PostgreSQL"):
        PersistenceSettings.from_environment(
            {"COMMERCE_AGENT_DATABASE_URI": "sqlite:///tmp/checkpoints.db"}
        )


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_non_development_database_requires_tls(environment: str) -> None:
    with pytest.raises(PersistenceConfigurationError, match="enforce TLS"):
        PersistenceSettings.from_environment(
            {
                "COMMERCE_AGENT_ENVIRONMENT": environment,
                "COMMERCE_AGENT_DATABASE_URI": (
                    "postgresql://user:secret-canary@db/commerce_agent"
                    "?sslmode=disable"
                ),
            }
        )


def test_runtime_open_does_not_run_schema_migrations() -> None:
    saver = _FakeSaver()
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def factory(settings: PersistenceSettings, serde):
        captured["database_uri"] = settings.database_uri
        captured["serde"] = serde
        yield saver

    async def exercise() -> None:
        async with open_postgres_checkpointer(
            _settings(),
            saver_factory=factory,
        ) as opened:
            assert opened is saver

    asyncio.run(exercise())

    assert saver.setup_calls == 0
    assert isinstance(captured["serde"], JsonPlusSerializer)


def test_schema_setup_is_an_explicit_idempotent_deployment_step() -> None:
    saver = _FakeSaver()

    @asynccontextmanager
    async def factory(_settings: PersistenceSettings, _serde):
        yield saver

    asyncio.run(
        initialize_checkpoint_schema(
            _settings(),
            saver_factory=factory,
        )
    )

    assert saver.setup_calls == 1
