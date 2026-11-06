"""Verify EU LangSmith graph traces hide synthetic inputs and metadata.

This diagnostic never sends a credential-like canary to LangGraph. It first
proves that GuardedGraph rejects one locally, then traces a harmless marker and
checks the EU API readback. No LangSmith request is written before preflight.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langsmith import Client

from commerce_agent.observability import EU_LANGSMITH_API_ENDPOINT


def _pin_eu_environment(api_key: str, project: str, *, tracing: bool) -> None:
    """Keep SDK's cached env view aligned with this isolated EU process."""

    os.environ["LANGSMITH_ENDPOINT"] = EU_LANGSMITH_API_ENDPOINT
    os.environ["LANGCHAIN_ENDPOINT"] = EU_LANGSMITH_API_ENDPOINT
    os.environ["COMMERCE_AGENT_LANGSMITH_TRACING"] = "true" if tracing else "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_METADATA"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGFUSE_ENABLED"] = "false"

    from langsmith.utils import get_env_var

    get_env_var.cache_clear()


def _readback_is_redacted(runs: list[Any], forbidden_values: Sequence[str]) -> bool:
    if not runs:
        return False
    for run in runs:
        if run.inputs not in (None, {}) or run.outputs not in (None, {}):
            return False
        if getattr(run, "error", None) not in (None, "redacted_error"):
            return False
        if getattr(run, "events", None) not in (None, []):
            return False
        if getattr(run, "tags", None) not in (None, []):
            return False
        if getattr(run, "attachments", None) not in (None, {}):
            return False
        extra = getattr(run, "extra", None) or {}
        if not isinstance(extra, Mapping) or set(extra) - {"metadata"}:
            return False
        metadata = extra.get("metadata")
        # LangSmith adds this structural depth after ingest; the exporter sends
        # no application metadata.
        if metadata not in (None, {}) and not (
            isinstance(metadata, Mapping)
            and set(metadata) == {"ls_run_depth"}
            and type(metadata["ls_run_depth"]) is int
            and metadata["ls_run_depth"] >= 0
        ):
            return False
        serialized = str(run.model_dump(mode="json"))
        if any(value in serialized for value in forbidden_values):
            return False
    return True


async def _run_synthetic_graph(marker: str) -> tuple[uuid.UUID, uuid.UUID, str]:
    # Import only after main() has pinned the EU endpoint and masking flags.
    from langchain_core.messages import HumanMessage
    from commerce_agent.context import CommerceRuntimeContext
    from commerce_agent.persistence import PersistenceSettings, open_postgres_graph
    from commerce_agent.safety import SensitiveStateError

    settings = PersistenceSettings.from_environment()
    context = CommerceRuntimeContext(
        tenant_id="synthetic-tenant",
        user_id="synthetic-user",
        roles=("manager",),
        request_id=f"synthetic-{uuid.uuid4().hex}",
    )
    rejected_id = uuid.uuid4()
    traced_id = uuid.uuid4()
    async with open_postgres_graph(settings) as graph:
        try:
            await graph.ainvoke(
                {
                    "schema_version": 1,
                    "messages": [
                        HumanMessage(content="lsv2_" + "a" * 32, id="synthetic-rejected")
                    ],
                },
                config={
                    "run_id": rejected_id,
                    "configurable": {"thread_id": f"rejected-{rejected_id}"},
                },
                context=context,
            )
        except SensitiveStateError:
            pass
        else:
            raise RuntimeError("credential-like input reached LangGraph")

        result = await graph.ainvoke(
            {
                "schema_version": 1,
                "messages": [HumanMessage(content=marker, id="synthetic-traced")],
            },
            config={
                "run_id": traced_id,
                "configurable": {"thread_id": f"m1-eu-trace-{traced_id}"},
            },
            context=context,
        )
        if set(result["agent_statuses"]) != {
            "sales", "marketing", "supply", "price", "finance"
        }:
            raise RuntimeError("graph did not finish the five fixed branches")
        graph.flush_traces()
    return rejected_id, traced_id, context.request_id


def main() -> int:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY is not configured")
        return 2

    project = os.environ.get(
        "LANGSMITH_PROJECT", "ai-ecom-commerce-agent-development"
    )
    workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID") or None
    # Disable implicit tracing and pin EU before the first Client is created.
    _pin_eu_environment(api_key, project, tracing=False)
    try:
        client = Client(
            api_url=EU_LANGSMITH_API_ENDPOINT,
            api_key=api_key,
            workspace_id=workspace_id,
            hide_inputs=True,
            hide_outputs=True,
            hide_metadata=True,
        )
        next(client.list_projects(limit=1), None)
    except Exception as exc:
        print(f"LangSmith EU preflight failed: {type(exc).__name__}")
        return 1

    if not os.environ.get("COMMERCE_AGENT_DATABASE_URI"):
        print("COMMERCE_AGENT_DATABASE_URI is not configured")
        return 2

    _pin_eu_environment(api_key, project, tracing=True)

    marker = f"synthetic-private-{uuid.uuid4().hex}"
    try:
        rejected_id, traced_id, request_id = asyncio.run(_run_synthetic_graph(marker))
    except Exception as exc:
        print(f"Synthetic graph run failed: {type(exc).__name__}")
        return 1

    for _ in range(15):
        try:
            runs = list(
                client.list_runs(
                    project_name=project,
                    trace_id=traced_id,
                )
            )
        except Exception as exc:
            print(f"LangSmith EU readback failed: {type(exc).__name__}")
            return 1
        if runs:
            if not _readback_is_redacted(
                runs,
                (marker, "synthetic-tenant", "synthetic-user", request_id),
            ):
                print(f"LangSmith EU trace contains unmasked content: {traced_id}")
                return 1
            names = {run.name for run in runs}
            if {"commerce_agent", "intake", "sales", "finance"} <= names:
                try:
                    rejected_runs = list(
                        client.list_runs(
                            project_name=project,
                            run_ids=[rejected_id],
                            limit=1,
                        )
                    )
                except Exception as exc:
                    print(f"LangSmith rejected-run check failed: {type(exc).__name__}")
                    return 1
                if rejected_runs:
                    print(f"Rejected canary unexpectedly created trace: {rejected_id}")
                    return 1
                print(f"LangSmith EU masked graph trace verified: {traced_id}")
                return 0
        time.sleep(2)

    print(f"LangSmith EU graph nodes not visible after tracing: {traced_id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
