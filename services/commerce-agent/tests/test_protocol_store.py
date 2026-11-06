from __future__ import annotations

import asyncio
import os
from typing import TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from psycopg.types.json import Jsonb

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph
from commerce_agent.persistence import PersistenceSettings, initialize_checkpoint_schema
from commerce_agent.protocol_events import normalize_graph_event
from commerce_agent.protocol_store import (
    IdempotencyConflict,
    ProtocolRecordError,
    ThreadNotOwned,
    ThreadOwner,
    open_protocol_store,
)
from commerce_agent.safety import SensitiveStateError
from commerce_agent.serialization import checkpoint_serializer


def test_protocol_owner_and_envelope_validation() -> None:
    with pytest.raises(ProtocolRecordError, match="required"):
        ThreadOwner(tenant_id="", user_id="user")
    with pytest.raises(SensitiveStateError):
        ThreadOwner(tenant_id="tenant", user_id="lsv2_credential_like_canary_12345")


def test_graph_event_normalizer_drops_source_sequence_and_blocks_secrets() -> None:
    event = normalize_graph_event({
        "type": "event",
        "method": "values",
        "seq": True,
        "event_id": "source-local-id",
        "params": {
            "namespace": (),
            "timestamp": 1_760_000_000_000,
            "data": {"messages": [HumanMessage(content="safe synthetic message")]},
            "interrupts": ({"value": {"kind": "approval"}, "id": "synthetic-interrupt"},),
        },
    })
    assert set(event) == {"type", "method", "params"}
    assert event["params"]["data"]["messages"][0]["content"] == "safe synthetic message"
    assert event["params"]["timestamp"] == 1_760_000_000_000
    assert event["params"]["interrupts"][0]["id"] == "synthetic-interrupt"
    message_event = normalize_graph_event({
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": 1_760_000_000_001,
            "data": (HumanMessage(content="safe token"), {"node": "synthetic"}),
        },
    })
    assert message_event["params"]["data"]["content"] == "safe token"
    with pytest.raises(SensitiveStateError):
        normalize_graph_event({
            "type": "event",
            "method": "values",
            "params": {
                "namespace": [],
                "timestamp": 1_760_000_000_000,
                "data": "lsv2_credential_like_canary_12345",
            },
        })


def test_real_graph_v3_interrupt_payload_survives_normalization() -> None:
    class ApprovalState(TypedDict, total=False):
        decision: str

    def approval(_: ApprovalState) -> ApprovalState:
        decision = interrupt({"kind": "approval", "action_id": "synthetic-action"})
        return {"decision": str(decision)}

    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    graph = GuardedGraph(builder.compile(checkpointer=InMemorySaver()))

    async def exercise() -> None:
        stream = await graph.astream_protocol_events(
            {}, config={"configurable": {"thread_id": str(uuid4())}}
        )
        raw_events = [event async for event in stream]
        assert raw_events
        interrupt_events = [
            event for event in raw_events if event["params"].get("interrupts")
        ]
        assert interrupt_events
        normalized = [normalize_graph_event(event) for event in interrupt_events]
        assert any(event["params"].get("interrupts") for event in normalized)

    asyncio.run(exercise())


def test_real_graph_v3_events_are_durable_and_replayable() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    thread_id = str(uuid4())
    owner = ThreadOwner(tenant_id="protocol-graph-tenant", user_id="user-a")
    runtime_context = CommerceRuntimeContext(
        tenant_id=owner.tenant_id,
        user_id=owner.user_id,
        roles=("manager",),
        request_id="protocol-graph-request",
    )

    async def cleanup(store) -> None:
        async with store._pool.connection() as conn:
            await conn.execute("DELETE FROM commerce_agent_events WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_runs WHERE thread_id = %s", (thread_id,))
            await conn.execute("DELETE FROM commerce_agent_threads WHERE thread_id = %s", (thread_id,))

    async def exercise() -> None:
        graph = GuardedGraph(build_graph(checkpointer=InMemorySaver(serde=checkpoint_serializer())))
        graph_input = {
            "schema_version": 1,
            "messages": [HumanMessage(content="safe synthetic query").model_dump(mode="json")],
        }
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            try:
                run_id = await store.create_run(
                    owner,
                    thread_id,
                    command_id="graph-command-1",
                    input_payload=graph_input,
                    runtime_context=runtime_context.model_dump(mode="json"),
                )
                stream = await graph.astream_protocol_events(
                    graph_input,
                    config={"configurable": {"thread_id": thread_id}},
                    context=runtime_context,
                )
                count = 0
                async for raw in stream:
                    saved = await store.append_event(
                        owner, thread_id, run_id, normalize_graph_event(raw)
                    )
                    assert saved["seq"] == count
                    count += 1
                assert count > 0
            except Exception:
                await cleanup(store)
                raise
        async with open_protocol_store(settings) as second_store:
            try:
                events = await second_store.read_events(owner, thread_id)
                assert len(events) == count
                assert [event["seq"] for event in events] == list(range(count))
                assert any(
                    event["method"] == "values"
                    and isinstance(event["params"]["data"], dict)
                    and "agent_statuses" in event["params"]["data"]
                    for event in events
                )
                assert all(event["event_id"] == f"{thread_id}:{event['seq']}" for event in events)
            finally:
                await cleanup(second_store)

    asyncio.run(exercise())


def test_protocol_store_is_durable_and_owner_scoped_in_postgres() -> None:
    if not os.getenv("COMMERCE_AGENT_DATABASE_URI"):
        pytest.skip("dedicated development PostgreSQL is not configured")
    settings = PersistenceSettings.from_environment()
    thread_id = str(uuid4())
    owner = ThreadOwner(tenant_id="protocol-test-tenant", user_id="user-a")
    other_user = ThreadOwner(tenant_id="protocol-test-tenant", user_id="user-b")
    other_tenant = ThreadOwner(tenant_id="other-tenant", user_id="user-a")
    context = {
        "tenant_id": owner.tenant_id,
        "user_id": owner.user_id,
        "roles": ["manager"],
        "request_id": "first-request",
    }
    rejected_thread_id = str(uuid4())
    unowned_checkpoint_thread_id = str(uuid4())
    unowned_checkpoint_id = str(uuid4())

    async def cleanup(store) -> None:
        # Delete only the UUID minted by this test, including after a failure.
        async with store._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM commerce_agent_events WHERE thread_id = %s",
                (thread_id,),
            )
            await conn.execute(
                "DELETE FROM commerce_agent_runs WHERE thread_id = %s",
                (thread_id,),
            )
            await conn.execute(
                "DELETE FROM commerce_agent_threads WHERE thread_id = %s",
                (thread_id,),
            )
            await conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_id = %s AND checkpoint_id = %s
                """,
                (unowned_checkpoint_thread_id, unowned_checkpoint_id),
            )

    async def exercise() -> None:
        await initialize_checkpoint_schema(settings)
        async with open_protocol_store(settings) as store:
            await store.setup()
            try:
                async with store._pool.connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO checkpoints (
                            thread_id, checkpoint_ns, checkpoint_id,
                            checkpoint, metadata
                        ) VALUES (%s, '', %s, %s, %s)
                        """,
                        (
                            unowned_checkpoint_thread_id,
                            unowned_checkpoint_id,
                            Jsonb({}),
                            Jsonb({}),
                        ),
                    )
                with pytest.raises(ThreadNotOwned):
                    await store.create_thread(owner, unowned_checkpoint_thread_id)
                with pytest.raises(SensitiveStateError):
                    await store.create_run(
                        owner,
                        rejected_thread_id,
                        command_id="credential-canary",
                        input_payload={
                            "schema_version": 1,
                            "messages": [{"content": "lsv2_credential_like_canary_12345"}],
                        },
                        runtime_context=context,
                    )
                with pytest.raises(ThreadNotOwned):
                    await store.require_owned(owner, rejected_thread_id)
                with pytest.raises(ProtocolRecordError, match="verified owner"):
                    await store.create_run(
                        owner,
                        rejected_thread_id,
                        command_id="forged-context",
                        input_payload={"schema_version": 1, "messages": []},
                        runtime_context={**context, "tenant_id": "other-tenant"},
                    )
                run_id = await store.create_run(
                    owner,
                    thread_id,
                    command_id="command-1",
                    input_payload={"schema_version": 1, "messages": []},
                    runtime_context=context,
                )
                # Transport retries must return the same run, not enqueue another.
                assert await store.create_run(
                    owner,
                    thread_id,
                    command_id="command-1",
                    input_payload={"schema_version": 1, "messages": []},
                    runtime_context={**context, "request_id": "retry-request"},
                ) == run_id
                with pytest.raises(IdempotencyConflict):
                    await store.create_run(
                        owner,
                        thread_id,
                        command_id="command-1",
                        input_payload={"schema_version": 1, "messages": [{"content": "changed"}]},
                        runtime_context=context,
                    )
                with pytest.raises(IdempotencyConflict):
                    await store.create_run(
                        owner,
                        thread_id,
                        command_id="command-1",
                        input_payload={"schema_version": 1, "messages": []},
                        runtime_context={**context, "roles": ["viewer"]},
                    )

                async def publish(number: int) -> dict[str, object]:
                    return await store.append_event(
                        owner,
                        thread_id,
                        run_id,
                        {
                            "type": "event",
                            "method": "lifecycle",
                            "params": {
                                "namespace": [],
                                "data": {"event": "running", "index": number},
                            },
                        },
                    )

                published = await asyncio.gather(*(publish(i) for i in range(12)))
                assert sorted(event["seq"] for event in published) == list(range(12))
                with pytest.raises(SensitiveStateError):
                    await store.append_event(
                        owner,
                        thread_id,
                        run_id,
                        {
                            "type": "event",
                            "method": "values",
                            "params": {
                                "namespace": [],
                                "data": "lsv2_credential_like_canary_12345",
                            },
                        },
                    )
                with pytest.raises(ProtocolRecordError, match="assigned by PostgreSQL"):
                    await store.append_event(
                        owner,
                        thread_id,
                        run_id,
                        {
                            "type": "event",
                            "method": "lifecycle",
                            "seq": 12,
                            "params": {"namespace": [], "data": {"event": "running"}},
                        },
                    )
            except Exception:
                await cleanup(store)
                raise

        # A fresh pool stands in for a new worker after process restart.
        async with open_protocol_store(settings) as second_store:
            try:
                events = await second_store.read_events(owner, thread_id)
                assert [event["seq"] for event in events] == list(range(12))
                assert [event["seq"] for event in await second_store.read_events(
                    owner, thread_id, since=8
                )] == [9, 10, 11]
                assert all(event["event_id"] == f"{thread_id}:{event['seq']}" for event in events)
                with pytest.raises(ThreadNotOwned):
                    await second_store.read_events(other_user, thread_id)
                with pytest.raises(ThreadNotOwned):
                    await second_store.read_events(other_tenant, thread_id)
                with pytest.raises(ThreadNotOwned):
                    await second_store.create_thread(other_tenant, thread_id)
                with pytest.raises(ProtocolRecordError, match="replay cursor"):
                    await second_store.read_events(owner, thread_id, since=-2)
            finally:
                await cleanup(second_store)

    asyncio.run(exercise())
