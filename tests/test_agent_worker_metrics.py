from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from agi_ecom_app import worker


def test_agent_worker_concurrency_defaults_to_four_and_is_bounded(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_WORKER_CONCURRENCY", raising=False)
    assert worker._agent_worker_concurrency() == 4
    monkeypatch.setenv("AGENT_WORKER_CONCURRENCY", "0")
    assert worker._agent_worker_concurrency() == 1
    monkeypatch.setenv("AGENT_WORKER_CONCURRENCY", "invalid")
    assert worker._agent_worker_concurrency() == 4


def test_agent_worker_metrics_uses_its_own_multiprocess_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "agent-worker-metrics"
    metrics_dir.mkdir()
    stale = metrics_dir / "counter_123.db"
    stale.write_bytes(b"stale")
    monkeypatch.setenv("AGENT_WORKER_PROMETHEUS_MULTIPROC_DIR", str(metrics_dir))
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    registry = worker._prepare_agent_worker_metrics()

    assert registry is not None
    assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == str(metrics_dir)
    assert not stale.exists()


def test_agent_worker_metrics_starts_on_dedicated_port(monkeypatch) -> None:
    calls: list[tuple[int, object]] = []
    from prometheus_client import CollectorRegistry, generate_latest

    registry = CollectorRegistry()
    monkeypatch.setenv("AGENT_WORKER_METRICS_PORT", "19103")
    monkeypatch.setattr(
        "prometheus_client.start_http_server",
        lambda port, *, registry: calls.append((port, registry)),
    )

    port = worker._start_agent_worker_metrics(registry)

    assert port == 19103
    assert calls == [(19103, registry)]
    assert b"agent_worker_up 1.0" in generate_latest(registry)


def test_agent_worker_applies_marts_only_for_simulation_modes() -> None:
    calls: list[dict[str, object]] = []

    def apply_fn(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"mode": "apply"}

    assert worker._ensure_decision_simulation_marts(
        "off", apply_fn=apply_fn
    ) is None
    assert calls == []

    result = worker._ensure_decision_simulation_marts(
        "shadow", apply_fn=apply_fn
    )

    assert result == {"mode": "apply"}
    assert calls == [{"schema": "agent_marts", "commit": True}]


def test_agent_worker_applies_marts_when_only_pricing_simulation_is_enabled() -> None:
    calls: list[dict[str, object]] = []

    def apply_fn(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"mode": "apply"}

    result = worker._ensure_decision_simulation_marts(
        "off",
        pricing_mode="shadow",
        apply_fn=apply_fn,
    )

    assert result == {"mode": "apply"}
    assert calls == [{"schema": "agent_marts", "commit": True}]


def test_forked_job_eligibility_counter_is_exported_by_parent_registry(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["AGENT_WORKER_PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path / "metrics")
    env["PYTHONPATH"] = str(root)
    script = """
import os
from agi_ecom_app.worker import _prepare_agent_worker_metrics

registry = _prepare_agent_worker_metrics()
pid = os.fork()
if pid == 0:
    from agi_ecom_app.decision_simulation.metrics import ELIGIBILITY_OUTCOMES
    ELIGIBILITY_OUTCOMES.labels(mode="shadow", outcome="eligible").inc()
    os._exit(0)
os.waitpid(pid, 0)
from prometheus_client import generate_latest
payload = generate_latest(registry).decode("utf-8")
assert 'decision_simulation_eligibility_outcomes_total' in payload
assert 'mode="shadow",outcome="eligible"' in payload
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_compose_exposes_agent_worker_metrics_without_sharing_simulation_dir() -> None:
    compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    agent_worker = compose.split("  worker:\n", 1)[1].split(
        "  simulation-worker:\n", 1
    )[0]
    simulation_worker = compose.split("  simulation-worker:\n", 1)[1]

    assert (
        'AGENT_WORKER_METRICS_PORT: "${AGENT_WORKER_METRICS_PORT:-9103}"'
        in agent_worker
    )
    assert (
        "AGENT_WORKER_PROMETHEUS_MULTIPROC_DIR: "
        '"${AGENT_WORKER_PROMETHEUS_MULTIPROC_DIR:-/tmp/agent-worker-prometheus}"'
        in agent_worker
    )
    assert 'expose:\n      - "9103"' in agent_worker
    assert "/tmp/agent-worker-prometheus" in agent_worker
    assert 'PROMETHEUS_MULTIPROC_DIR: "/tmp/decision-simulation-prometheus"' in simulation_worker
    assert "/tmp/decision-simulation-prometheus" not in agent_worker
    assert 'AGENT_WORKER_CONCURRENCY: "${AGENT_WORKER_CONCURRENCY:-4}"' in agent_worker
    assert 'AGENT_TOOL_CACHE_TTL_SECONDS: "${AGENT_TOOL_CACHE_TTL_SECONDS:-3600}"' in agent_worker
