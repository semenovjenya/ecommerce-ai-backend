from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from scripts import check_langsmith_export as probe


def _run(
    *, input_payload=None, output_payload=None, marker="", error=None,
    events=None, tags=None, metadata=None, extra_fields=None,
):
    extra = {"metadata": metadata if metadata is not None else (marker or {})}
    extra.update(extra_fields or {})
    return SimpleNamespace(
        inputs=input_payload,
        outputs=output_payload,
        error=error,
        events=events,
        tags=tags,
        extra=extra,
        model_dump=lambda **_kwargs: {
            "inputs": input_payload,
            "outputs": output_payload,
            "extra": extra,
        },
    )


def test_langsmith_readback_requires_empty_io_and_no_marker() -> None:
    assert probe._readback_is_redacted([_run(input_payload={}, output_payload={})], ("private",))
    assert not probe._readback_is_redacted([], ("private",))
    assert not probe._readback_is_redacted(
        [_run(input_payload={"message": "private"}, output_payload={})], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(input_payload={}, output_payload={}, marker="private")], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(input_payload={}, output_payload={}, error="private")], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(input_payload={}, output_payload={}, events=[{"data": "private"}])], ("private",)
    )
    assert probe._readback_is_redacted(
        [_run(metadata={"ls_run_depth": 2})], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(metadata={"ls_run_depth": "2"})], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(metadata={"ls_run_depth": 2, "tenant_id": "private"})], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(tags=["private"])], ("private",)
    )
    assert not probe._readback_is_redacted(
        [_run(extra_fields={"unapproved": "private"})], ("private",)
    )


def test_langsmith_probe_stops_before_export_when_eu_access_fails() -> None:
    with patch.dict(
        os.environ,
        {
            "LANGSMITH_API_KEY": "synthetic-test-key",
            "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
            "COMMERCE_AGENT_DATABASE_URI": "synthetic-db-uri",
        },
    ), patch.object(probe, "Client") as client, patch.object(
        probe, "_run_synthetic_graph"
    ) as graph_run:
        client.return_value.list_projects.side_effect = PermissionError("forbidden")
        assert probe.main() == 1
        graph_run.assert_not_called()
        assert os.environ["LANGSMITH_ENDPOINT"] == probe.EU_LANGSMITH_API_ENDPOINT
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["COMMERCE_AGENT_LANGSMITH_TRACING"] == "false"
    client.assert_called_once_with(
        api_url=probe.EU_LANGSMITH_API_ENDPOINT,
        api_key="synthetic-test-key",
        workspace_id=None,
        hide_inputs=True,
        hide_outputs=True,
        hide_metadata=True,
    )


def test_langsmith_probe_pins_eu_settings_and_checks_rejected_run() -> None:
    rejected_id = uuid.uuid4()
    traced_id = uuid.uuid4()

    async def synthetic_graph(_marker):
        return rejected_id, traced_id, "synthetic-request-id"

    safe_run = SimpleNamespace(
        name="commerce_agent",
        inputs={},
        outputs={},
        extra={},
        model_dump=lambda **_kwargs: {"inputs": {}, "outputs": {}},
    )
    node_runs = [
        SimpleNamespace(
            name=name,
            inputs={},
            outputs={},
            extra={},
            model_dump=lambda **_kwargs: {"inputs": {}, "outputs": {}},
        )
        for name in ("intake", "sales", "finance")
    ]

    with patch.dict(
        os.environ,
        {
            "LANGSMITH_API_KEY": "synthetic-test-key",
            "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
            "COMMERCE_AGENT_DATABASE_URI": "synthetic-db-uri",
        },
    ), patch.object(probe, "Client") as client, patch.object(
        probe, "_run_synthetic_graph", side_effect=synthetic_graph
    ):
        client.return_value.list_projects.return_value = iter([])
        client.return_value.list_runs.side_effect = [
            iter([safe_run, *node_runs]),
            iter([]),
        ]
        assert probe.main() == 0
        assert os.environ["LANGSMITH_ENDPOINT"] == probe.EU_LANGSMITH_API_ENDPOINT
        assert os.environ["LANGCHAIN_ENDPOINT"] == probe.EU_LANGSMITH_API_ENDPOINT
        assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
        assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
        assert os.environ["LANGSMITH_HIDE_METADATA"] == "true"
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["COMMERCE_AGENT_LANGSMITH_TRACING"] == "true"
    assert client.return_value.list_runs.call_args_list[-1].kwargs["run_ids"] == [
        rejected_id
    ]
