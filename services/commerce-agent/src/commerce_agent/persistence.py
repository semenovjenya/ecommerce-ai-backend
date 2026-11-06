from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph
from commerce_agent.serialization import checkpoint_serializer


DATABASE_URI_ENV = "COMMERCE_AGENT_DATABASE_URI"
RUNTIME_ENV_ENV = "COMMERCE_AGENT_ENVIRONMENT"
POOL_MIN_SIZE_ENV = "COMMERCE_AGENT_DB_POOL_MIN_SIZE"
POOL_MAX_SIZE_ENV = "COMMERCE_AGENT_DB_POOL_MAX_SIZE"


class PersistenceConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PersistenceSettings:
    environment: str
    pool_min_size: int
    pool_max_size: int
    database_uri: str = field(repr=False)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "PersistenceSettings":
        env = os.environ if environment is None else environment
        database_uri = env.get(DATABASE_URI_ENV, "").strip()
        runtime_environment = env.get(RUNTIME_ENV_ENV, "development").strip().lower()
        try:
            pool_min_size = int(env.get(POOL_MIN_SIZE_ENV, "1"))
            pool_max_size = int(env.get(POOL_MAX_SIZE_ENV, "10"))
        except ValueError as exc:
            raise PersistenceConfigurationError(
                "PostgreSQL pool sizes must be integers"
            ) from exc
        if pool_min_size < 1 or pool_max_size < pool_min_size:
            raise PersistenceConfigurationError(
                "PostgreSQL pool sizes must satisfy 1 <= min <= max"
            )
        if not database_uri:
            raise PersistenceConfigurationError(
                f"{DATABASE_URI_ENV} must be set in the runtime secret environment"
            )
        if runtime_environment not in {"development", "staging", "production", "test"}:
            raise PersistenceConfigurationError(
                f"{RUNTIME_ENV_ENV} must be development, staging, production, or test"
            )
        parsed = urlsplit(database_uri)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise PersistenceConfigurationError(
                f"{DATABASE_URI_ENV} must be a PostgreSQL connection URI"
            )
        sslmode = parse_qs(parsed.query).get("sslmode", [None])[-1]
        if runtime_environment in {"staging", "production"} and sslmode in {
            None,
            "allow",
            "prefer",
            "disable",
        }:
            raise PersistenceConfigurationError(
                f"{DATABASE_URI_ENV} must enforce TLS in {runtime_environment}"
            )
        return cls(
            environment=runtime_environment,
            pool_min_size=pool_min_size,
            pool_max_size=pool_max_size,
            database_uri=database_uri,
        )


class _Saver(Protocol):
    async def setup(self) -> None: ...


SaverFactory = Callable[[PersistenceSettings, Any], Any]


@asynccontextmanager
async def _postgres_saver_factory(
    settings: PersistenceSettings,
    serde: Any,
) -> AsyncIterator[AsyncPostgresSaver]:
    pool = AsyncConnectionPool(
        conninfo=settings.database_uri,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
        name="commerce-agent-checkpoints",
    )
    async with pool:
        yield AsyncPostgresSaver(pool, serde=serde)


@asynccontextmanager
async def open_postgres_checkpointer(
    settings: PersistenceSettings,
    *,
    saver_factory: SaverFactory = _postgres_saver_factory,
) -> AsyncIterator[_Saver]:
    """Open a checkpointer without running schema changes at service startup."""

    async with saver_factory(settings, checkpoint_serializer()) as checkpointer:
        yield checkpointer


async def initialize_checkpoint_schema(
    settings: PersistenceSettings,
    *,
    saver_factory: SaverFactory = _postgres_saver_factory,
) -> None:
    """Run idempotent checkpointer migrations as an explicit deployment step."""

    async with open_postgres_checkpointer(
        settings,
        saver_factory=saver_factory,
    ) as checkpointer:
        await checkpointer.setup()


@asynccontextmanager
async def open_postgres_graph(
    settings: PersistenceSettings,
    *,
    interrupt_before: tuple[str, ...] = (),
    interrupt_after: tuple[str, ...] = (),
) -> AsyncIterator[Any]:
    """Keep the saver connection alive for the complete graph lifecycle."""

    async with open_postgres_checkpointer(settings) as checkpointer:
        yield GuardedGraph(
            build_graph(
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
            )
        )


__all__ = [
    "DATABASE_URI_ENV",
    "PersistenceConfigurationError",
    "PersistenceSettings",
    "initialize_checkpoint_schema",
    "open_postgres_checkpointer",
    "open_postgres_graph",
]
