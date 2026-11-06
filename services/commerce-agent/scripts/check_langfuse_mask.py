"""Verify Langfuse masks payloads and filters unsafe span names on readback."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from langfuse import Langfuse

from commerce_agent.observability import (
    mask_langfuse_otel_spans,
    should_export_safe_langfuse_span,
)


def _readback_status(
    observations: list[Any], *, root_name: str, sentinel_name: str,
    canary: str, model_name: str,
) -> str:
    by_name = {item.name: item for item in observations}
    if canary in str([item.model_dump(mode="json") for item in observations]):
        return "unsafe"
    if root_name not in by_name or sentinel_name not in by_name:
        return "pending"
    root = by_name[root_name]
    if (
        root.type != "GENERATION"
        or not (root.provided_model_name == model_name or root.internal_model_id)
        or not root.usage_details
        or root.usage_details.get("input") != 12
        or root.usage_details.get("output") != 5
        or not root.cost_details
        or abs(root.cost_details.get("total", 0) - 0.002) > 1e-9
    ):
        return "missing-metrics"
    return "verified"


def main() -> int:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        print("Langfuse credentials are not configured")
        return 2

    client = Langfuse(
        mask_otel_spans=mask_langfuse_otel_spans,
        should_export_span=should_export_safe_langfuse_span,
    )
    if (
        getattr(getattr(client, "_resources", None), "mask_otel_spans", None)
        is not mask_langfuse_otel_spans
        or getattr(getattr(client, "_resources", None), "should_export_span", None)
        is not should_export_safe_langfuse_span
    ):
        print("Langfuse client does not have the required export safety hooks")
        return 2

    allowed_models = [
        name.strip()
        for name in os.environ.get("LANGFUSE_MODEL_ALLOWLIST", "").split(",")
        if name.strip()
    ]
    if not allowed_models:
        print("LANGFUSE_MODEL_ALLOWLIST is not configured")
        return 2
    model_name = allowed_models[0]

    canary = "lsv2_" + "a" * 32
    name = f"m1-mask-canary-{uuid.uuid4().hex[:12]}"
    sentinel_name = f"m1-safe-sentinel-{uuid.uuid4().hex[:12]}"
    with client.start_as_current_observation(
        name=name,
        as_type="generation",
        input={"content": canary},
        output={"content": canary},
        metadata={"canary": canary},
        model=model_name,
        usage_details={"input": 12, "output": 5},
        cost_details={"total": 0.002},
    ) as observation:
        trace_id = observation.trace_id
        # Span names cannot be patched by mask_otel_spans. The export filter
        # must drop this child entirely while preserving its safe sibling.
        with client.start_as_current_observation(
            name=canary, input={"content": canary}
        ) as unsafe_child:
            if unsafe_child.trace_id != trace_id:
                print("Langfuse unsafe child did not join the synthetic trace")
                return 1
        with client.start_as_current_observation(name=sentinel_name) as sentinel:
            if sentinel.trace_id != trace_id:
                print("Langfuse safe sentinel did not join the synthetic trace")
                return 1
    client.flush()

    consecutive_safe_reads = 0
    for _ in range(10):
        try:
            response = client.api.observations.get_many(
                trace_id=trace_id,
                fields="core,basic,io,metadata,model,usage",
                limit=100,
            )
        except Exception as exc:
            print(f"Langfuse readback failed: {type(exc).__name__}")
            return 1
        status = _readback_status(
            response.data,
            root_name=name,
            sentinel_name=sentinel_name,
            canary=canary,
            model_name=model_name,
        )
        if status == "unsafe":
            print(f"Langfuse export filter failed for trace {trace_id}")
            return 1
        if status == "missing-metrics":
            print(f"Langfuse model/usage/cost missing for trace {trace_id}")
            return 1
        if status == "verified":
            consecutive_safe_reads += 1
            if consecutive_safe_reads == 2:
                print(f"Langfuse masked and filtered trace verified: {trace_id}")
                return 0
        else:
            consecutive_safe_reads = 0
        time.sleep(2)

    print(f"Langfuse trace or safe sentinel was not visible after flush: {trace_id}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
