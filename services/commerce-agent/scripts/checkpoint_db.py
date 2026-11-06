from __future__ import annotations

import argparse
import asyncio
import json

from langchain_core.messages import HumanMessage

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.persistence import (
    PersistenceSettings,
    initialize_checkpoint_schema,
    open_postgres_graph,
)


_DOMAIN_NAMES = ("sales", "marketing", "supply", "price", "finance")


def _state_summary(values: dict[str, object]) -> dict[str, object]:
    """Expose typed progress without echoing messages, identity or free text."""

    statuses = values.get("agent_statuses") or {}
    if not isinstance(statuses, dict):
        raise RuntimeError("checkpoint agent_statuses has an unexpected shape")
    domain_statuses: dict[str, str] = {}
    analysis_statuses: dict[str, str | None] = {}
    for name in _DOMAIN_NAMES:
        raw_status = statuses.get(name)
        if raw_status is None:
            domain_statuses[name] = "missing"
        else:
            status = (
                raw_status.get("status")
                if isinstance(raw_status, dict)
                else getattr(raw_status, "status", None)
            )
            if status not in {
                "pending", "running", "planned", "completed", "failed",
                "skipped", "interrupted", "cancelled",
            }:
                raise RuntimeError(f"checkpoint has an invalid {name} status")
            domain_statuses[name] = status

        raw_analysis = values.get(f"{name}_analysis")
        if raw_analysis is None:
            analysis_statuses[name] = None
        else:
            analysis_status = (
                raw_analysis.get("status")
                if isinstance(raw_analysis, dict)
                else getattr(raw_analysis, "status", None)
            )
            if analysis_status not in {
                "not_implemented", "completed", "partial", "failed",
            }:
                raise RuntimeError(f"checkpoint has an invalid {name} analysis status")
            analysis_statuses[name] = analysis_status

    return {
        "domain_statuses": domain_statuses,
        "analysis_statuses": analysis_statuses,
        "pending_interrupt": values.get("pending_interrupt") is not None,
        "data_gap_count": len(values.get("data_gaps") or []),
        "evidence_count": len(values.get("evidence_matrix") or []),
        "action_task_count": len(values.get("action_tasks") or []),
        "execution_receipt_count": len(values.get("execution_receipts") or []),
    }


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _context(thread_id: str) -> CommerceRuntimeContext:
    return CommerceRuntimeContext(
        tenant_id="persistence-smoke-tenant",
        user_id="persistence-smoke-user",
        roles=("manager",),
        request_id=f"persistence-smoke-{thread_id}",
    )


async def _setup(settings: PersistenceSettings) -> dict[str, object]:
    await initialize_checkpoint_schema(settings)
    return {"status": "checkpoint_schema_ready"}


async def _pause(
    settings: PersistenceSettings,
    thread_id: str,
) -> dict[str, object]:
    config = _config(thread_id)
    async with open_postgres_graph(
        settings,
        interrupt_after=("query_plan",),
    ) as graph:
        await graph.ainvoke(
            {
                "schema_version": 1,
                "messages": [
                    HumanMessage(
                        content="PostgreSQL restart recovery smoke",
                        id=f"message-{thread_id}",
                    )
                ],
            },
            config=config,
            context=_context(thread_id),
        )
        snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise RuntimeError("graph did not pause at a durable checkpoint")
    return {
        "status": "paused",
        "thread_id": thread_id,
        "schema_version": snapshot.values.get("schema_version"),
        "next": sorted(snapshot.next),
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
    }


async def _resume(
    settings: PersistenceSettings,
    thread_id: str,
) -> dict[str, object]:
    config = _config(thread_id)
    async with open_postgres_graph(settings) as graph:
        before = await graph.aget_state(config)
        if not before.next:
            raise RuntimeError("no interrupted checkpoint found for thread")
        result = await graph.ainvoke(
            None,
            config=config,
            context=_context(thread_id),
        )
        after = await graph.aget_state(config)
        history_count = 0
        async for _ in graph.aget_state_history(config):
            history_count += 1
    if after.next:
        raise RuntimeError("graph did not finish after checkpoint resume")
    if result.get("schema_version") != 1:
        raise RuntimeError("restored state has an unexpected schema version")
    if result.get("tenant_id") != "persistence-smoke-tenant":
        raise RuntimeError("verified runtime identity was not restored")
    return {
        "status": "resumed_and_completed",
        "thread_id": thread_id,
        "schema_version": result["schema_version"],
        **_state_summary(result),
        "history_count": history_count,
        "checkpoint_id": after.config["configurable"].get("checkpoint_id"),
    }


async def _inspect(
    settings: PersistenceSettings,
    thread_id: str,
) -> dict[str, object]:
    config = _config(thread_id)
    async with open_postgres_graph(settings) as graph:
        snapshot = await graph.aget_state(config)
        if not snapshot.values:
            raise RuntimeError("thread checkpoint was not found")
        history_count = 0
        async for _ in graph.aget_state_history(config):
            history_count += 1
    return {
        "status": "checkpoint_found",
        "thread_id": thread_id,
        "schema_version": snapshot.values.get("schema_version"),
        **_state_summary(snapshot.values),
        "next": sorted(snapshot.next),
        "history_count": history_count,
    }


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup", "pause", "resume", "inspect"))
    parser.add_argument("--thread-id", default="m1-postgres-restart-smoke")
    args = parser.parse_args()
    settings = PersistenceSettings.from_environment()
    if args.command == "setup":
        output = await _setup(settings)
    elif args.command == "pause":
        output = await _pause(settings, args.thread_id)
    elif args.command == "resume":
        output = await _resume(settings, args.thread_id)
    else:
        output = await _inspect(settings, args.thread_id)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
