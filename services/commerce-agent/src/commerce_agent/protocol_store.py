"""Durable ownership, run and replay records for the self-managed protocol.

The LangGraph checkpointer owns graph state. These tables own transport/run
lifecycle and are deliberately separate from business data and legacy chats.
Schema setup is an explicit deployment step, never a service-startup side effect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from commerce_agent.persistence import PersistenceSettings
from commerce_agent.safety import assert_secret_free


_COMMAND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_EVENT_METHOD = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")


class ThreadNotOwned(LookupError):
    """The thread does not exist or does not belong to the verified actor."""


class IdempotencyConflict(ValueError):
    """A command ID was reused with different input or runtime context."""


class ProtocolRecordError(ValueError):
    """An untrusted protocol record failed validation."""


@dataclass(frozen=True)
class ThreadOwner:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id:
            raise ProtocolRecordError("verified tenant_id and user_id are required")
        if len(self.tenant_id) > 128 or len(self.user_id) > 128:
            raise ProtocolRecordError("verified identity exceeds its size limit")
        assert_secret_free(
            {"tenant_id": self.tenant_id, "user_id": self.user_id},
            path="verified_identity",
        )


def _thread_uuid(thread_id: str) -> UUID:
    try:
        return UUID(thread_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ProtocolRecordError("thread_id must be a UUID") from exc


def _json_payload(value: Any, *, path: str) -> tuple[str, Any]:
    assert_secret_free(value, path=path)
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProtocolRecordError("protocol payload must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > 262_144:
        raise ProtocolRecordError("protocol payload exceeds 256 KiB")
    return encoded, json.loads(encoded)


class ProtocolStore:
    """Every data read/write is scoped by verified tenant and user identity."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """Run the idempotent M1.5 schema migration explicitly."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_threads (
                        thread_id uuid PRIMARY KEY,
                        tenant_id text NOT NULL,
                        user_id text NOT NULL,
                        last_seq bigint NOT NULL DEFAULT -1,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_runs (
                        run_id uuid PRIMARY KEY,
                        thread_id uuid NOT NULL REFERENCES commerce_agent_threads(thread_id),
                        command_id text NOT NULL,
                        request_sha256 char(64) NOT NULL,
                        input_json jsonb NOT NULL,
                        context_json jsonb NOT NULL,
                        status text NOT NULL DEFAULT 'queued'
                            CHECK (status IN (
                                'queued', 'running', 'interrupted',
                                'completed', 'failed', 'cancelled'
                            )),
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        UNIQUE (thread_id, command_id),
                        UNIQUE (thread_id, run_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS commerce_agent_events (
                        thread_id uuid NOT NULL,
                        seq bigint NOT NULL,
                        run_id uuid NOT NULL,
                        method text NOT NULL,
                        namespace text[] NOT NULL,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        PRIMARY KEY (thread_id, seq),
                        FOREIGN KEY (thread_id, run_id)
                            REFERENCES commerce_agent_runs(thread_id, run_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS commerce_agent_runs_thread_status_idx
                    ON commerce_agent_runs(thread_id, status, created_at)
                    """
                )

    async def create_thread(self, owner: ThreadOwner, thread_id: str) -> None:
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO commerce_agent_threads(thread_id, tenant_id, user_id)
                SELECT %s, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints WHERE thread_id = %s
                )
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (ident, owner.tenant_id, owner.user_id, str(ident)),
            )
        await self.require_owned(owner, thread_id)

    async def require_owned(self, owner: ThreadOwner, thread_id: str) -> None:
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT 1 FROM commerce_agent_threads
                WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                """,
                (ident, owner.tenant_id, owner.user_id),
            )
            if await cursor.fetchone() is None:
                raise ThreadNotOwned("thread not found")

    async def create_run(
        self,
        owner: ThreadOwner,
        thread_id: str,
        *,
        command_id: str,
        input_payload: Mapping[str, Any],
        runtime_context: Mapping[str, Any],
    ) -> str:
        if not isinstance(command_id, str) or not _COMMAND_ID.fullmatch(command_id):
            raise ProtocolRecordError("invalid command ID")
        if not isinstance(input_payload, Mapping) or not isinstance(runtime_context, Mapping):
            raise ProtocolRecordError("run input and context must be objects")
        _, input_data = _json_payload(dict(input_payload), path="run.input")
        _, context_data = _json_payload(dict(runtime_context), path="run.context")
        if (
            context_data.get("tenant_id") != owner.tenant_id
            or context_data.get("user_id") != owner.user_id
        ):
            raise ProtocolRecordError("runtime identity disagrees with verified owner")
        # A transport retry may have a new correlation request_id. Compare
        # every other policy/context field so a command ID cannot silently
        # alias a run started under different roles or model policy.
        stable_context = {
            key: value for key, value in context_data.items() if key != "request_id"
        }
        digest = sha256(
            json.dumps(
                {"input": input_data, "context": stable_context},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        await self.create_thread(owner, thread_id)
        ident = _thread_uuid(thread_id)
        new_run_id = uuid4()
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO commerce_agent_runs (
                    run_id, thread_id, command_id, request_sha256,
                    input_json, context_json
                )
                SELECT %s, t.thread_id, %s, %s, %s, %s
                FROM commerce_agent_threads t
                WHERE t.thread_id = %s AND t.tenant_id = %s AND t.user_id = %s
                ON CONFLICT (thread_id, command_id) DO NOTHING
                RETURNING run_id, request_sha256
                """,
                (
                    new_run_id, command_id, digest,
                    Jsonb(input_data), Jsonb(context_data),
                    ident, owner.tenant_id, owner.user_id,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                cursor = await conn.execute(
                    """
                    SELECT r.run_id, r.request_sha256 FROM commerce_agent_runs r
                    JOIN commerce_agent_threads t ON t.thread_id = r.thread_id
                    WHERE r.thread_id = %s AND r.command_id = %s
                        AND t.tenant_id = %s AND t.user_id = %s
                    """,
                    (ident, command_id, owner.tenant_id, owner.user_id),
                )
                row = await cursor.fetchone()
            if row is None:
                raise ThreadNotOwned("thread not found")
            if row["request_sha256"] != digest:
                raise IdempotencyConflict("command ID conflicts with an existing run")
            return str(row["run_id"])

    async def append_event(
        self,
        owner: ThreadOwner,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise ProtocolRecordError("event must be an object")
        _, safe_event = _json_payload(dict(event), path="protocol.event")
        method = safe_event.get("method")
        params = safe_event.get("params")
        namespace = params.get("namespace") if isinstance(params, dict) else None
        if (
            safe_event.get("type") != "event"
            or not isinstance(method, str)
            or not _EVENT_METHOD.fullmatch(method)
            or not isinstance(namespace, list)
            or not all(isinstance(part, str) and len(part) <= 128 for part in namespace)
        ):
            raise ProtocolRecordError("invalid Agent Protocol event envelope")
        if "seq" in safe_event or "event_id" in safe_event:
            raise ProtocolRecordError("event sequence is assigned by PostgreSQL")
        ident = _thread_uuid(thread_id)
        try:
            run_ident = UUID(run_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolRecordError("run_id must be a UUID") from exc
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    """
                    UPDATE commerce_agent_threads
                    SET last_seq = last_seq + 1
                    WHERE thread_id = %s AND tenant_id = %s AND user_id = %s
                    AND EXISTS (
                        SELECT 1 FROM commerce_agent_runs
                        WHERE thread_id = %s AND run_id = %s
                    )
                    RETURNING last_seq
                    """,
                    (
                        ident, owner.tenant_id, owner.user_id,
                        ident, run_ident,
                    ),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise ThreadNotOwned("thread or run not found")
                seq = row["last_seq"]
                safe_event["seq"] = seq
                safe_event["event_id"] = f"{ident}:{seq}"
                await conn.execute(
                    """
                    INSERT INTO commerce_agent_events (
                        thread_id, seq, run_id, method, namespace, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ident, seq, run_ident, method, namespace,
                        Jsonb(safe_event),
                    ),
                )
                return safe_event

    async def read_events(
        self,
        owner: ThreadOwner,
        thread_id: str,
        *,
        since: int = -1,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if not isinstance(since, int) or since < -1:
            raise ProtocolRecordError("invalid replay cursor")
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ProtocolRecordError("invalid replay page size")
        await self.require_owned(owner, thread_id)
        ident = _thread_uuid(thread_id)
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT e.payload FROM commerce_agent_events e
                JOIN commerce_agent_threads t ON t.thread_id = e.thread_id
                WHERE e.thread_id = %s AND e.seq > %s
                    AND t.tenant_id = %s AND t.user_id = %s
                ORDER BY e.seq ASC LIMIT %s
                """,
                (ident, since, owner.tenant_id, owner.user_id, limit),
            )
            return [row["payload"] for row in await cursor.fetchall()]


@asynccontextmanager
async def open_protocol_store(settings: PersistenceSettings) -> AsyncIterator[ProtocolStore]:
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
        name="commerce-agent-protocol",
    )
    async with pool:
        yield ProtocolStore(pool)


async def initialize_protocol_schema(settings: PersistenceSettings) -> None:
    async with open_protocol_store(settings) as store:
        await store.setup()


__all__ = [
    "IdempotencyConflict",
    "ProtocolRecordError",
    "ProtocolStore",
    "ThreadNotOwned",
    "ThreadOwner",
    "initialize_protocol_schema",
    "open_protocol_store",
]
