#!/usr/bin/env python3
"""Capture a redacted, aggregate-only legacy baseline for migration slice M0.

The collector deliberately never exports chat messages, tool payloads, credentials,
or trace inputs/outputs.  It exits non-zero until every required M0 gate is backed by
real evidence; ``--allow-partial`` is intended only for recording current blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "tests/corpora/legacy_m0/v1/manifest.json"
DEFAULT_HUMAN_CORPUS = ROOT / "tests/corpora/legacy_m0/v1/human_manager_cases.jsonl"
DEFAULT_DETERMINISTIC_OUTPUTS = ROOT / "tests/corpora/legacy_m0/v1/deterministic_outputs.json"
MINIMUM_HUMAN_CASES = 40
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ALLOWED_FIXTURE_ROOTS = (
    ROOT / "tests/fixtures/prompt_v2",
    ROOT / "tests/fixtures/manager_decision",
)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported manifest schema_version")
    records = payload.get("fixtures")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest.fixtures must be a non-empty list")

    errors: list[str] = []
    verified: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for item in records:
        case_id = str(item.get("case_id") or "")
        category = str(item.get("category") or "")
        relative = str(item.get("path") or "")
        expected = str(item.get("sha256") or "")
        source_run_id = item.get("source_run_id")
        fixture_path = (ROOT / relative).resolve()
        if ROOT not in fixture_path.parents:
            errors.append(f"fixture escapes repository root: {relative}")
            continue
        if not any(root == fixture_path.parent for root in ALLOWED_FIXTURE_ROOTS):
            errors.append(f"fixture path is not allowlisted: {relative}")
        if fixture_path.suffix != ".json":
            errors.append(f"fixture must be JSON: {relative}")
        if not case_id or case_id in case_ids:
            errors.append(f"empty or duplicate case_id: {case_id or '<empty>'}")
        case_ids.add(case_id)
        if not category:
            errors.append(f"fixture has no category: {case_id}")
        if not isinstance(source_run_id, int) or source_run_id <= 0:
            errors.append(f"fixture has invalid source_run_id: {case_id}")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            errors.append(f"fixture has invalid sha256: {relative}")
        if not fixture_path.is_file():
            errors.append(f"fixture is missing: {relative}")
            continue
        actual = _sha256(fixture_path)
        if actual != expected:
            errors.append(f"fixture hash mismatch: {relative}")
        verified.append(
            {
                "case_id": item.get("case_id"),
                "category": item.get("category"),
                "source_run_id": source_run_id,
                "path": relative,
                "sha256": actual,
            }
        )
    return verified, errors


def load_human_cases(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        return 0, [f"human-labelled corpus is missing: {display_path}"]

    from agi_ecom_app.manager_evaluation import (
        ManagerEvaluationDataError,
        load_historical_manager_cases,
    )

    try:
        cases = load_historical_manager_cases(path)
    except (ManagerEvaluationDataError, json.JSONDecodeError, ValueError) as exc:
        return 0, [f"invalid human corpus: {type(exc).__name__}"]
    return len(cases), []


def human_source_run_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    from agi_ecom_app.manager_evaluation import (
        ManagerEvaluationDataError,
        load_historical_manager_cases,
    )

    run_ids: list[int] = []
    try:
        cases = load_historical_manager_cases(path)
    except (ManagerEvaluationDataError, json.JSONDecodeError, ValueError):
        return []
    for case in cases:
        prefix, separator, raw_id = case.source_ref.partition(":")
        if prefix == "agent_run" and separator and raw_id.isdigit():
            run_ids.append(int(raw_id))
    return run_ids


def collect_manager_evaluation(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    from agi_ecom_app.manager_evaluation import evaluate_manager_release_file

    return evaluate_manager_release_file(path).to_dict()


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_runs(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    statuses = Counter(str(row.get("status") or "unknown") for row in materialized)
    durations = [
        float(row["duration_seconds"])
        for row in materialized
        if row.get("duration_seconds") is not None
        and float(row["duration_seconds"]) >= 0
    ]
    terminal = sum(statuses.get(status, 0) for status in TERMINAL_STATUSES)
    completed = statuses.get("completed", 0)
    return {
        "run_count": len(materialized),
        "status_counts": dict(sorted(statuses.items())),
        "terminal_count": terminal,
        "terminal_success_rate": completed / terminal if terminal else None,
        "latency_sample_count": len(durations),
        "latency_p50_seconds": percentile(durations, 0.50),
        "latency_p95_seconds": percentile(durations, 0.95),
    }


def collect_run_metrics(run_ids: Sequence[int]) -> dict[str, Any]:
    from agi_ecom_app.services.memory import get_memory_connection

    with get_memory_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            select status,
                   case
                     when completed_at is not null and started_at is not null
                     then extract(epoch from (completed_at - started_at))
                   end as duration_seconds
              from agent_run_traces
             where id = any(%s)
             order by id
            """,
            [list(run_ids)],
        )
        rows = list(cursor.fetchall())
        cursor.execute(
            """
            select min(id) as first_run_id,
                   max(id) as last_run_id,
                   min(created_at) as first_created_at,
                   max(created_at) as last_created_at
              from agent_run_traces
             where id = any(%s)
            """,
            [list(run_ids)],
        )
        window = cursor.fetchone() or {}

    result = summarize_runs(rows)
    result["window"] = {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in window.items()
    }
    return result


def collect_langfuse_metrics(run_ids: Sequence[int], environment: str) -> dict[str, Any]:
    from langfuse import get_client

    client = get_client()
    if not client.auth_check():
        raise RuntimeError("Langfuse authentication failed")

    traces: list[Any] = []
    page = 1
    while True:
        response = client.api.trace.list(
            page=page,
            limit=100,
            name="agent-run-result",
            environment=environment,
            order_by="timestamp.asc",
        )
        allowed = set(run_ids)
        traces.extend(
            trace
            for trace in response.data
            if str((trace.metadata or {}).get("run_id", "")).isdigit()
            and int((trace.metadata or {})["run_id"]) in allowed
        )
        if page >= response.meta.total_pages:
            break
        page += 1

    costs = [float(trace.total_cost) for trace in traces if trace.total_cost is not None]
    latencies = [float(trace.latency) for trace in traces if trace.latency is not None]
    return {
        "trace_name": "agent-run-result",
        "environment": environment,
        "covered_run_ids": sorted(
            {
                int((trace.metadata or {})["run_id"])
                for trace in traces
                if str((trace.metadata or {}).get("run_id", "")).isdigit()
            }
        ),
        "trace_count": len(traces),
        "cost_sample_count": len(costs),
        "mean_cost_usd": mean(costs) if costs else None,
        "cost_p50_usd": percentile(costs, 0.50),
        "cost_p95_usd": percentile(costs, 0.95),
        "latency_sample_count": len(latencies),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "first_timestamp": min((trace.timestamp for trace in traces), default=None),
        "last_timestamp": max((trace.timestamp for trace in traces), default=None),
    }


def _safe_collect(collector: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return collector(), None
    except Exception as exc:  # fail closed without leaking connection details
        return None, type(exc).__name__


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def repository_commit(root: Path = ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        head = (root / ".git/HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            reference = root / ".git" / head.removeprefix("ref: ")
            if reference.is_file():
                return reference.read_text(encoding="utf-8").strip()
        if len(head) == 40:
            return head
        raise RuntimeError("cannot determine repository commit")


def evidence_definition_hash(manifest: Path) -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), manifest.resolve()):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_tree_hash(root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    candidates = [
        root / "requirements.txt",
        *sorted((root / "agi_ecom_app").rglob("*.py")),
        *sorted((root / "agi_ecom_app/prompts").glob("*.md")),
        *sorted((root / "tests").glob("test_*.py")),
    ]
    for path in candidates:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_report(
    manifest: Path,
    human_corpus: Path,
    deterministic_outputs: Path,
    environment: str,
) -> dict[str, Any]:
    fixtures, fixture_errors = verify_manifest(manifest)
    human_count, human_errors = load_human_cases(human_corpus)
    fixture_run_ids = [int(item["source_run_id"]) for item in fixtures]
    run_ids = sorted(set(fixture_run_ids + human_source_run_ids(human_corpus)))
    run_metrics, run_error = _safe_collect(lambda: collect_run_metrics(run_ids))
    cost_metrics, cost_error = _safe_collect(
        lambda: collect_langfuse_metrics(run_ids, environment)
    )
    manager_evaluation, manager_error = _safe_collect(
        lambda: collect_manager_evaluation(human_corpus) or {}
    )
    deterministic_present = deterministic_outputs.is_file()
    cost_coverage = (
        cost_metrics is not None
        and run_metrics is not None
        and cost_metrics.get("cost_sample_count") == run_metrics.get("run_count")
        and run_metrics.get("run_count") == len(run_ids)
    )

    gates = [
        Gate("fixture_integrity", not fixture_errors, "; ".join(fixture_errors) or f"{len(fixtures)} fixtures verified"),
        Gate(
            "human_labelled_manager_cases",
            human_count >= MINIMUM_HUMAN_CASES and not human_errors,
            "; ".join(human_errors)
            or f"{human_count}/{MINIMUM_HUMAN_CASES} human-labelled cases",
        ),
        Gate(
            "fixed_corpus_environment",
            environment in {"staging", "production"},
            f"captured from {environment}; staging or production required",
        ),
        Gate(
            "legacy_run_metrics",
            run_metrics is not None and run_metrics.get("run_count") == len(run_ids),
            run_error or f"{(run_metrics or {}).get('run_count', 0)}/{len(run_ids)} corpus runs captured",
        ),
        Gate(
            "llm_cost_metrics",
            bool(cost_coverage),
            cost_error
            or f"{(cost_metrics or {}).get('cost_sample_count', 0)}/{len(run_ids)} corpus runs have correlated cost",
        ),
        Gate(
            "deterministic_output_hashes",
            deterministic_present,
            "captured" if deterministic_present else "deterministic_outputs.json is missing",
        ),
        Gate(
            "evidence_guardrail_metrics",
            human_count >= MINIMUM_HUMAN_CASES and manager_evaluation is not None,
            manager_error or f"manager evaluation cases: {human_count}/{MINIMUM_HUMAN_CASES}",
        ),
    ]
    commit = repository_commit()
    return _json_safe(
        {
            "schema_version": "1.0",
            "slice": "M0",
            "captured_at": datetime.now(timezone.utc),
            "source": {
                "repository_commit": commit,
                "source_tree_sha256": source_tree_hash(),
                "evidence_definition_sha256": evidence_definition_hash(manifest),
                "manifest": str(manifest.relative_to(ROOT)),
                "human_corpus": str(human_corpus.relative_to(ROOT)),
                "deterministic_outputs": str(deterministic_outputs.relative_to(ROOT)),
                "environment": environment,
                "payload_policy": "aggregate metrics and hashes only",
            },
            "fixture_inventory": fixtures,
            "human_labelled_case_count": human_count,
            "fixed_corpus_run_ids": run_ids,
            "legacy_runs": run_metrics,
            "langfuse": cost_metrics,
            "manager_evaluation": manager_evaluation,
            "gates": [gate.to_dict() for gate in gates],
            "complete": all(gate.passed for gate in gates),
            "verification_commands": [
                "uv run --isolated --python 3.11 --with-requirements requirements.txt --with pytest python -m pytest -q",
                "docker compose run --rm --no-deps -v \"$PWD:/app\" -w /app backend python scripts/capture_m0_baseline.py --allow-partial --environment development --output docs/ai-framework-migration/evidence/M0-legacy-baseline.json",
                "python3 .agents/skills/migrate-ai-ecom-langgraph/scripts/check_migration_contracts.py",
            ],
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human-corpus", type=Path, default=DEFAULT_HUMAN_CORPUS)
    parser.add_argument(
        "--deterministic-outputs", type=Path, default=DEFAULT_DETERMINISTIC_OUTPUTS
    )
    parser.add_argument("--environment", default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        args.manifest.resolve(),
        args.human_corpus.resolve(),
        args.deterministic_outputs.resolve(),
        args.environment,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0 if report["complete"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
