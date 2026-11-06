"""Verify a synthetic checkpointed graph run produces masked Langfuse nodes."""

from __future__ import annotations

import asyncio
import os
import time
import uuid

# This diagnostic is Langfuse-only. Never let an inherited US LangSmith endpoint
# receive the synthetic graph input while testing the independent exporter.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["COMMERCE_AGENT_LANGSMITH_TRACING"] = "false"
os.environ["LANGFUSE_ENABLED"] = "true"

from langchain_core.messages import HumanMessage  # noqa: E402
from langfuse import Langfuse  # noqa: E402

from commerce_agent.context import CommerceRuntimeContext  # noqa: E402
from commerce_agent.observability import (  # noqa: E402
    mask_langfuse_otel_spans,
    should_export_safe_langfuse_span,
)
from commerce_agent.persistence import PersistenceSettings, open_postgres_graph  # noqa: E402


async def _run_graph(client: Langfuse) -> str:
    settings = PersistenceSettings.from_environment()
    thread_id = f"m1-observability-{uuid.uuid4().hex}"
    context = CommerceRuntimeContext(
        tenant_id="synthetic-tenant",
        user_id="synthetic-user",
        roles=("manager",),
        request_id=f"synthetic-{uuid.uuid4().hex}",
    )
    async with open_postgres_graph(settings) as guarded_graph:
        with client.start_as_current_observation(name="commerce_agent_smoke") as root:
            trace_id = root.trace_id
            result = await guarded_graph.ainvoke(
                {
                    "schema_version": 1,
                    "messages": [
                        HumanMessage(
                            content="Synthetic M1 graph observability smoke",
                            id=f"synthetic-{uuid.uuid4().hex}",
                        )
                    ],
                },
                config={"configurable": {"thread_id": thread_id}},
                context=context,
            )
        if set(result["agent_statuses"]) != {
            "sales", "marketing", "supply", "price", "finance"
        }:
            raise RuntimeError("graph did not finish all fixed domain branches")
        saved = await guarded_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if saved.next or saved.values.get("schema_version") != 1:
            raise RuntimeError("graph checkpoint was not saved as completed")
    return trace_id


def main() -> int:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("Langfuse credentials are not configured")
        return 2
    client = Langfuse(
        mask_otel_spans=mask_langfuse_otel_spans,
        should_export_span=should_export_safe_langfuse_span,
    )
    resources = getattr(client, "_resources", None)
    if (
        getattr(resources, "mask_otel_spans", None) is not mask_langfuse_otel_spans
        or getattr(resources, "should_export_span", None)
        is not should_export_safe_langfuse_span
    ):
        print("Langfuse safety hooks are not configured")
        return 2

    try:
        trace_id = asyncio.run(_run_graph(client))
    except Exception as exc:
        print(f"Synthetic graph run failed: {type(exc).__name__}")
        return 1
    client.flush()

    for _ in range(10):
        try:
            response = client.api.observations.get_many(
                trace_id=trace_id,
                fields="core,basic,io,metadata",
                limit=100,
            )
        except Exception as exc:
            print(f"Langfuse readback failed: {type(exc).__name__}")
            return 1
        observations = response.data
        names = {item.name for item in observations}
        expected = {
            "commerce_agent_smoke", "commerce_agent", "intake", "router",
            "query_plan", "sales", "marketing", "supply", "price", "finance",
            "evidence_gate", "synthesis",
        }
        if expected <= names:
            serialized = [item.model_dump(mode="json") for item in observations]
            if (
                "Synthetic M1 graph observability smoke" in str(serialized)
                or any(item.input is not None or item.output is not None for item in observations)
            ):
                print(f"Langfuse graph payload was not masked: {trace_id}")
                return 1
            print(f"Langfuse graph trace verified: {trace_id}; nodes={len(names)}")
            return 0
        time.sleep(2)

    print(f"Langfuse graph nodes were not visible after flush: {trace_id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
