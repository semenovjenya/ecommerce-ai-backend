from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from agi_ecom_app.services.memory import ensure_memory_schema, get_memory_connection


class ExperimentProgramStorageConflict(RuntimeError):
    """A durable idempotency key was reused with a different contract."""


class ExperimentProgramStorageNotFound(LookupError):
    pass


@dataclass(frozen=True)
class ExperimentProgramBundleRows:
    program: dict[str, Any]
    snapshot: dict[str, Any]
    frontiers: list[dict[str, Any]]
    candidates: list[dict[str, Any]]


@dataclass(frozen=True)
class StoredNegotiationMessage:
    row: dict[str, Any]
    created: bool
    program_status: str
    program_version: int


@dataclass(frozen=True)
class ExperimentOutboxRecord:
    id: int
    event_id: str
    program_id: str
    topic: str
    message_key: str
    payload: dict[str, Any]
    status: str
    attempt_count: int
    max_attempts: int
    available_at: datetime
    claimed_at: datetime | None
    lease_owner: str | None
    lease_expires_at: datetime | None
    delivered_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


def create_program_bundle_atomic(
    *,
    program: dict[str, Any],
    snapshot: dict[str, Any],
    frontiers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[ExperimentProgramBundleRows, bool]:
    """Create one complete frontier and its event/outbox journal atomically."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into experiment_programs (
                id, idempotency_key, request_hash, objective, mode, status,
                target, domains, generator_version, snapshot_hash, current_epoch,
                experiment_budget, domain_coverage, source_chat_id, source_run_id,
                source_decision_id, heartbeat_at, error, version, completed_at
            )
            values (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            on conflict (idempotency_key) do nothing
            returning *
            """,
            [
                program["id"],
                program["idempotency_key"],
                program["request_hash"],
                program["objective"],
                program["mode"],
                program["status"],
                Json(program["target"]),
                Json(program["domains"]),
                program["generator_version"],
                program["snapshot_hash"],
                program["current_epoch"],
                Json(program["experiment_budget"]),
                Json(program["domain_coverage"]),
                program.get("source_chat_id"),
                program.get("source_run_id"),
                program.get("source_decision_id"),
                program.get("heartbeat_at"),
                Json(program.get("error") or {}),
                program.get("version", 1),
                program.get("completed_at"),
            ],
        )
        inserted_program = cur.fetchone()
        if inserted_program is None:
            cur.execute(
                "select * from experiment_programs where idempotency_key = %s",
                [program["idempotency_key"]],
            )
            existing = cur.fetchone()
            if existing is None:
                raise RuntimeError("experiment program idempotency winner disappeared")
            if existing["request_hash"] != program["request_hash"]:
                raise ExperimentProgramStorageConflict(
                    "Idempotency-Key is already bound to another experiment program contract"
                )
            return _get_program_bundle_with_cursor(cur, existing["id"]), False

        cur.execute(
            """
            insert into experiment_snapshots (
                id, program_id, epoch, snapshot_hash, status, data_as_of,
                evidence_packets, quality_summary
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                snapshot["id"],
                snapshot["program_id"],
                snapshot["epoch"],
                snapshot["snapshot_hash"],
                snapshot["status"],
                snapshot.get("data_as_of"),
                Json(snapshot["evidence_packets"]),
                Json(snapshot["quality_summary"]),
            ],
        )

        for frontier in frontiers:
            cur.execute(
                """
                insert into experiment_domain_frontiers (
                    program_id, domain, epoch, generator_version, snapshot_hash,
                    status, candidate_count, eligible_count, terminal_count, coverage,
                    completed_at
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    frontier["program_id"],
                    frontier["domain"],
                    frontier["epoch"],
                    frontier["generator_version"],
                    frontier["snapshot_hash"],
                    frontier["status"],
                    frontier["candidate_count"],
                    frontier["eligible_count"],
                    frontier["terminal_count"],
                    Json(frontier["coverage"]),
                    frontier.get("completed_at"),
                ],
            )

        for candidate in candidates:
            cur.execute(
                """
                insert into experiment_candidates (
                    id, program_id, epoch, candidate_key, domain, family, operation,
                    target, parameters, hypothesis, primary_metric, guardrail_metrics,
                    required_metrics, evidence_refs, evidence_status, causal_design,
                    estimated_cost, risk, reversibility, stop_conditions, status, score,
                    completed_at
                )
                values (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s
                )
                """,
                [
                    candidate["id"],
                    candidate["program_id"],
                    candidate["epoch"],
                    candidate["candidate_key"],
                    candidate["domain"],
                    candidate["family"],
                    candidate["operation"],
                    Json(candidate["target"]),
                    Json(candidate["parameters"]),
                    candidate.get("hypothesis"),
                    candidate.get("primary_metric"),
                    Json(candidate["guardrail_metrics"]),
                    Json(candidate["required_metrics"]),
                    Json(candidate["evidence_refs"]),
                    candidate["evidence_status"],
                    Json(candidate["causal_design"]),
                    Json(candidate["estimated_cost"]),
                    Json(candidate["risk"]),
                    Json(candidate["reversibility"]),
                    Json(candidate["stop_conditions"]),
                    candidate["status"],
                    Json(candidate.get("score") or {}),
                    candidate.get("completed_at"),
                ],
            )

        for requirement in requirements:
            cur.execute(
                """
                insert into experiment_data_requirements (
                    program_id, candidate_id, domain, metric, status, evidence_ref,
                    source_table, grain, data_as_of, max_age_seconds, details
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    requirement["program_id"],
                    requirement["candidate_id"],
                    requirement["domain"],
                    requirement["metric"],
                    requirement["status"],
                    requirement.get("evidence_ref"),
                    requirement.get("source_table"),
                    requirement.get("grain"),
                    requirement.get("data_as_of"),
                    requirement.get("max_age_seconds"),
                    Json(requirement.get("details") or {}),
                ],
            )

        for dependency in dependencies:
            cur.execute(
                """
                insert into experiment_candidate_dependencies (
                    program_id, candidate_id, depends_on_candidate_id,
                    dependency_type, details
                )
                values (%s, %s, %s, %s, %s)
                on conflict do nothing
                """,
                [
                    dependency["program_id"],
                    dependency["candidate_id"],
                    dependency["depends_on_candidate_id"],
                    dependency.get("dependency_type", "requires"),
                    Json(dependency.get("details") or {}),
                ],
            )

        for conflict in conflicts:
            cur.execute(
                """
                insert into experiment_candidate_conflicts (
                    program_id, left_candidate_id, right_candidate_id,
                    interference_key, conflict_type, details
                )
                values (%s, %s, %s, %s, %s, %s)
                on conflict do nothing
                """,
                [
                    conflict["program_id"],
                    conflict["left_candidate_id"],
                    conflict["right_candidate_id"],
                    conflict["interference_key"],
                    conflict.get("conflict_type", "causal_interference"),
                    Json(conflict.get("details") or {}),
                ],
            )

        for event in events:
            _append_event_and_outbox(cur, event)

        if program.get("source_decision_id"):
            _link_decision_with_cursor(
                cur,
                decision_id=program["source_decision_id"],
                program_id=program["id"],
            )

        return _get_program_bundle_with_cursor(cur, program["id"]), True


def get_program_bundle(program_id: str) -> ExperimentProgramBundleRows | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute("select 1 from experiment_programs where id = %s", [program_id])
        if cur.fetchone() is None:
            return None
        return _get_program_bundle_with_cursor(cur, program_id)


def _get_program_bundle_with_cursor(
    cur: Any,
    program_id: str,
) -> ExperimentProgramBundleRows:
    cur.execute("select * from experiment_programs where id = %s", [program_id])
    program = cur.fetchone()
    if program is None:
        raise ExperimentProgramStorageNotFound(program_id)
    cur.execute(
        """
        select * from experiment_snapshots
        where program_id = %s and epoch = %s
        """,
        [program_id, program["current_epoch"]],
    )
    snapshot = cur.fetchone()
    if snapshot is None:
        raise RuntimeError(f"experiment program {program_id} has no pinned snapshot")
    cur.execute(
        """
        select * from experiment_domain_frontiers
        where program_id = %s and epoch = %s
        order by domain
        """,
        [program_id, program["current_epoch"]],
    )
    frontiers = list(cur.fetchall())
    cur.execute(
        """
        select * from experiment_candidates
        where program_id = %s and epoch = %s
        order by domain, candidate_key, id
        """,
        [program_id, program["current_epoch"]],
    )
    candidates = list(cur.fetchall())
    return ExperimentProgramBundleRows(
        program=program,
        snapshot=snapshot,
        frontiers=frontiers,
        candidates=candidates,
    )


def append_negotiation_message_atomic(
    *,
    program_id: str,
    message: dict[str, Any],
) -> StoredNegotiationMessage:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from experiment_programs where id = %s for update",
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramStorageNotFound(program_id)
        if program["status"] in {
            "exhausted",
            "budget_exhausted",
            "paused",
            "failed",
        }:
            raise ExperimentProgramStorageConflict(
                f"experiment program does not accept negotiation messages in {program['status']}"
            )

        cur.execute(
            """
            insert into experiment_negotiation_messages (
                message_id, program_id, round_id, epoch, message_type, source_role,
                target_role, correlation_id, idempotency_key, fingerprint, payload,
                evidence_refs
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict do nothing
            returning *
            """,
            [
                message["message_id"],
                program_id,
                message.get("round_id"),
                program["current_epoch"],
                message["message_type"],
                message["source_role"],
                message.get("target_role"),
                message.get("correlation_id"),
                message["idempotency_key"],
                message["fingerprint"],
                Json(message["payload"]),
                Json(message["evidence_refs"]),
            ],
        )
        stored = cur.fetchone()
        if stored is None:
            cur.execute(
                """
                select * from experiment_negotiation_messages
                where program_id = %s
                  and (idempotency_key = %s or fingerprint = %s)
                order by id
                limit 1
                """,
                [program_id, message["idempotency_key"], message["fingerprint"]],
            )
            existing = cur.fetchone()
            if existing is None:
                raise RuntimeError("negotiation message conflict winner disappeared")
            if existing["fingerprint"] != message["fingerprint"]:
                raise ExperimentProgramStorageConflict(
                    "Idempotency-Key is already bound to another negotiation message"
                )
            return StoredNegotiationMessage(
                row=existing,
                created=False,
                program_status=program["status"],
                program_version=program["version"],
            )

        previous_status = program["status"]
        next_status = (
            "negotiating"
            if previous_status in {"created", "data_audit", "proposing", "blocked_data"}
            else previous_status
        )
        domain_coverage = dict(program["domain_coverage"] or {})

        if message["message_type"] == "domain_done":
            domain = message["payload"]["domain"]
            coverage_status = message["payload"]["coverageStatus"]
            cur.execute(
                """
                select * from experiment_domain_frontiers
                where program_id = %s and epoch = %s and domain = %s
                for update
                """,
                [program_id, program["current_epoch"], domain],
            )
            frontier = cur.fetchone()
            if frontier is None:
                raise ExperimentProgramStorageConflict(
                    f"domain is not part of the pinned frontier: {domain}"
                )
            if coverage_status == "ready" and int(frontier["eligible_count"]) < 1:
                raise ExperimentProgramStorageConflict(
                    f"domain {domain} cannot finish ready without eligible candidates"
                )
            frontier_status = {
                "ready": "ready",
                "blocked_data": "blocked_data",
                "constraint_only": "constraint_only",
                "no_controllable_lever": "no_controllable_lever",
                "exhausted": "exhausted",
            }[coverage_status]
            frontier_coverage = dict(frontier["coverage"] or {})
            frontier_coverage.update(
                {
                    "negotiationDone": True,
                    "coverageStatus": coverage_status,
                    "messageId": stored["message_id"],
                }
            )
            cur.execute(
                """
                update experiment_domain_frontiers
                set status = %s,
                    coverage = %s,
                    updated_at = now(),
                    completed_at = case
                        when %s in ('exhausted', 'constraint_only', 'no_controllable_lever')
                        then coalesce(completed_at, now())
                        else completed_at
                    end
                where program_id = %s and epoch = %s and domain = %s
                """,
                [
                    frontier_status,
                    Json(frontier_coverage),
                    frontier_status,
                    program_id,
                    program["current_epoch"],
                    domain,
                ],
            )
            domain_coverage[domain] = frontier_coverage
            cur.execute(
                """
                select status, eligible_count, coverage
                from experiment_domain_frontiers
                where program_id = %s and epoch = %s
                order by domain
                """,
                [program_id, program["current_epoch"]],
            )
            all_frontiers = list(cur.fetchall())
            all_done = bool(all_frontiers) and all(
                bool((row["coverage"] or {}).get("negotiationDone"))
                for row in all_frontiers
            )
            if all_done:
                total_eligible = sum(int(row["eligible_count"]) for row in all_frontiers)
                has_blocking_data = any(
                    row["status"] == "blocked_data" for row in all_frontiers
                )
                if total_eligible > 0 and not has_blocking_data:
                    next_status = "simulating"
                elif has_blocking_data:
                    next_status = "blocked_data"
                else:
                    next_status = "exhausted"

        cur.execute(
            """
            update experiment_programs
            set status = %s,
                domain_coverage = %s,
                version = version + 1,
                updated_at = now(),
                completed_at = case
                    when %s in ('exhausted', 'budget_exhausted', 'failed')
                    then coalesce(completed_at, now())
                    else completed_at
                end
            where id = %s and version = %s
            returning status, version
            """,
            [
                next_status,
                Json(domain_coverage),
                next_status,
                program_id,
                program["version"],
            ],
        )
        updated = cur.fetchone()
        if updated is None:
            raise ExperimentProgramStorageConflict(
                "experiment program changed concurrently; retry the message"
            )

        base_event = {
            "event_id": f"{program_id}:message:{stored['message_id']}",
            "program_id": program_id,
            "epoch": program["current_epoch"],
            "event_type": "negotiation_message_recorded",
            "payload": {
                "messageId": stored["message_id"],
                "messageType": stored["message_type"],
                "sourceRole": stored["source_role"],
                "targetRole": stored["target_role"],
            },
        }
        _append_event_and_outbox(cur, base_event)
        if next_status != previous_status:
            transition_event = {
                "event_id": (
                    f"{program_id}:status:{updated['version']}:{next_status}"
                ),
                "program_id": program_id,
                "epoch": program["current_epoch"],
                "event_type": "program_status_changed",
                "payload": {
                    "from": previous_status,
                    "to": next_status,
                    "version": updated["version"],
                },
            }
            _append_event_and_outbox(cur, transition_event)
            if next_status == "simulating":
                _append_event_and_outbox(
                    cur,
                    {
                        "event_id": f"{program_id}:simulation-requested:{program['current_epoch']}",
                        "program_id": program_id,
                        "epoch": program["current_epoch"],
                        "event_type": "simulation_requested",
                        "payload": {
                            "programId": program_id,
                            "epoch": program["current_epoch"],
                            "snapshotHash": program["snapshot_hash"],
                        },
                    },
                )

        return StoredNegotiationMessage(
            row=stored,
            created=True,
            program_status=updated["status"],
            program_version=updated["version"],
        )


def list_negotiation_messages(
    program_id: str,
    *,
    after_id: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from experiment_negotiation_messages
            where program_id = %s and id > %s
            order by id
            limit %s
            """,
            [program_id, max(0, after_id), max(1, min(limit, 5000))],
        )
        return list(cur.fetchall())


def list_program_events(
    program_id: str,
    *,
    after_id: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from experiment_program_events
            where program_id = %s and id > %s
            order by id
            limit %s
            """,
            [program_id, max(0, after_id), max(1, min(limit, 5000))],
        )
        return list(cur.fetchall())


def claim_program_outbox(
    *,
    lease_owner: str,
    limit: int = 100,
    lease_seconds: int = 60,
) -> list[ExperimentOutboxRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            with claimed as (
                select id
                from experiment_program_outbox
                where (
                    status = 'pending' and available_at <= now()
                ) or (
                    status = 'processing' and lease_expires_at < now()
                )
                order by available_at, id
                for update skip locked
                limit %s
            )
            update experiment_program_outbox as outbox
            set status = 'processing',
                attempt_count = outbox.attempt_count + 1,
                claimed_at = now(),
                lease_owner = %s,
                lease_expires_at = now() + make_interval(secs => %s),
                last_error = null,
                updated_at = now()
            from claimed
            where outbox.id = claimed.id
            returning outbox.*
            """,
            [max(1, min(limit, 1000)), lease_owner, max(1, lease_seconds)],
        )
        return [_outbox_record(row) for row in cur.fetchall()]


def mark_program_outbox_delivered(*, outbox_id: int, lease_owner: str) -> bool:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update experiment_program_outbox
            set status = 'delivered',
                delivered_at = now(),
                lease_expires_at = null,
                updated_at = now()
            where id = %s and status = 'processing' and lease_owner = %s
            returning id
            """,
            [outbox_id, lease_owner],
        )
        return cur.fetchone() is not None


def reschedule_program_outbox(
    *,
    outbox_id: int,
    lease_owner: str,
    error: str,
    delay_seconds: int,
) -> str | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update experiment_program_outbox
            set status = case when attempt_count >= max_attempts then 'dead' else 'pending' end,
                available_at = case
                    when attempt_count >= max_attempts then available_at
                    else now() + make_interval(secs => %s)
                end,
                lease_owner = null,
                lease_expires_at = null,
                last_error = %s,
                updated_at = now()
            where id = %s and status = 'processing' and lease_owner = %s
            returning status
            """,
            [max(0, delay_seconds), error[:4000], outbox_id, lease_owner],
        )
        row = cur.fetchone()
        return row["status"] if row else None


def link_simulation_run(
    *,
    program_id: str,
    simulation_run_id: str,
    candidate_links: dict[str, str],
) -> int:
    """Link one multi-candidate run and each durable simulation candidate."""

    if not candidate_links:
        raise ValueError("candidate_links cannot be empty")
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_runs as run
            set experiment_program_id = %s,
                updated_at = now()
            from manager_decisions as decision
            where run.id = %s
              and decision.id = run.decision_id
              and decision.experiment_program_id = %s
              and (run.experiment_program_id is null or run.experiment_program_id = %s)
            returning run.id
            """,
            [
                program_id,
                simulation_run_id,
                program_id,
                program_id,
            ],
        )
        if cur.fetchone() is None:
            raise ExperimentProgramStorageConflict(
                "simulation run does not exist, has another program, or its decision is not linked"
            )
        linked = 0
        for simulation_candidate_id, experiment_candidate_id in candidate_links.items():
            cur.execute(
                """
                update decision_simulation_candidates as simulation_candidate
                set experiment_program_id = %s,
                    experiment_candidate_id = %s,
                    updated_at = now()
                from experiment_candidates as candidate
                where simulation_candidate.simulation_run_id = %s
                  and simulation_candidate.candidate_id = %s
                  and candidate.id = %s
                  and candidate.program_id = %s
                  and (
                      simulation_candidate.experiment_program_id is null
                      or simulation_candidate.experiment_program_id = %s
                  )
                  and (
                      simulation_candidate.experiment_candidate_id is null
                      or simulation_candidate.experiment_candidate_id = %s
                  )
                returning simulation_candidate.candidate_id
                """,
                [
                    program_id,
                    experiment_candidate_id,
                    simulation_run_id,
                    simulation_candidate_id,
                    experiment_candidate_id,
                    program_id,
                    program_id,
                    experiment_candidate_id,
                ],
            )
            if cur.fetchone() is None:
                raise ExperimentProgramStorageConflict(
                    "simulation candidate mapping is missing or belongs to another program: "
                    f"{simulation_candidate_id}"
                )
            linked += 1
        return linked


def link_simulation_run_owner(*, program_id: str, simulation_run_id: str) -> bool:
    """Link the run before candidate initialization so crashes stay discoverable."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_runs as run
            set experiment_program_id = %s,
                updated_at = now()
            from manager_decisions as decision
            where run.id = %s
              and decision.id = run.decision_id
              and decision.experiment_program_id = %s
              and (run.experiment_program_id is null or run.experiment_program_id = %s)
            returning run.id
            """,
            [program_id, simulation_run_id, program_id, program_id],
        )
        return cur.fetchone() is not None


def link_program_decision(*, program_id: str, decision_id: str) -> bool:
    """Attach a generated ManagerDecision to a program without reassigning owners."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select id from experiment_programs where id = %s for update",
            [program_id],
        )
        if cur.fetchone() is None:
            raise ExperimentProgramStorageNotFound(program_id)
        _link_decision_with_cursor(
            cur,
            decision_id=decision_id,
            program_id=program_id,
        )
        cur.execute(
            """
            update experiment_programs
            set source_decision_id = coalesce(source_decision_id, %s),
                heartbeat_at = now(),
                updated_at = now()
            where id = %s
            returning id
            """,
            [decision_id, program_id],
        )
        return cur.fetchone() is not None


def bind_program_simulation_run(
    *,
    program_id: str,
    decision_id: str,
    simulation_run_id: str,
    candidate_links: dict[str, str],
    expected_epoch: int,
    expected_snapshot_hash: str,
) -> int:
    """Atomically bind decision, run and the complete candidate mapping."""

    if not candidate_links:
        raise ValueError("candidate_links cannot be empty")
    event_id = f"{program_id}:simulation-bound:{simulation_run_id}"
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from experiment_programs where id = %s for update",
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramStorageNotFound(program_id)
        if (
            int(program["current_epoch"]) != int(expected_epoch)
            or program["snapshot_hash"] != expected_snapshot_hash
        ):
            raise ExperimentProgramStorageConflict(
                "program epoch or snapshot changed before simulation binding"
            )
        _link_decision_with_cursor(
            cur,
            decision_id=decision_id,
            program_id=program_id,
        )
        cur.execute(
            """
            update experiment_programs
            set source_decision_id = coalesce(source_decision_id, %s),
                heartbeat_at = now(),
                simulation_dispatch_attempts = 0,
                updated_at = now()
            where id = %s
            """,
            [decision_id, program_id],
        )
        cur.execute(
            """
            update decision_simulation_runs as run
            set experiment_program_id = %s,
                updated_at = now()
            from manager_decisions as decision
            where run.id = %s
              and run.decision_id = %s
              and decision.id = run.decision_id
              and decision.experiment_program_id = %s
              and (run.experiment_program_id is null or run.experiment_program_id = %s)
            returning run.id
            """,
            [
                program_id,
                simulation_run_id,
                decision_id,
                program_id,
                program_id,
            ],
        )
        if cur.fetchone() is None:
            raise ExperimentProgramStorageConflict(
                "simulation run does not exist or belongs to another program"
            )
        linked = 0
        for simulation_candidate_id, experiment_candidate_id in candidate_links.items():
            cur.execute(
                """
                update decision_simulation_candidates as simulation_candidate
                set experiment_program_id = %s,
                    experiment_candidate_id = %s,
                    updated_at = now()
                from experiment_candidates as candidate
                where simulation_candidate.simulation_run_id = %s
                  and simulation_candidate.candidate_id = %s
                  and candidate.id = %s
                  and candidate.program_id = %s
                  and candidate.epoch = %s
                  and (
                      simulation_candidate.experiment_program_id is null
                      or simulation_candidate.experiment_program_id = %s
                  )
                  and (
                      simulation_candidate.experiment_candidate_id is null
                      or simulation_candidate.experiment_candidate_id = %s
                  )
                returning simulation_candidate.candidate_id
                """,
                [
                    program_id,
                    experiment_candidate_id,
                    simulation_run_id,
                    simulation_candidate_id,
                    experiment_candidate_id,
                    program_id,
                    expected_epoch,
                    program_id,
                    experiment_candidate_id,
                ],
            )
            if cur.fetchone() is None:
                raise ExperimentProgramStorageConflict(
                    "incomplete or cross-program simulation candidate mapping: "
                    f"{simulation_candidate_id}"
                )
            linked += 1
        _append_event_and_outbox(
            cur,
            {
                "event_id": event_id,
                "program_id": program_id,
                "epoch": expected_epoch,
                "event_type": "simulation_bound",
                "payload": {
                    "decisionId": decision_id,
                    "simulationRunId": simulation_run_id,
                    "linkedCandidateCount": linked,
                    "snapshotHash": expected_snapshot_hash,
                },
            },
        )
        return linked


def list_linked_simulation_runs(program_id: str) -> list[dict[str, Any]]:
    """Return the durable runs owned by one program, newest first."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_runs
            where experiment_program_id = %s
            order by created_at desc, id desc
            """,
            [program_id],
        )
        return list(cur.fetchall())


def record_program_simulation_dispatch(
    *,
    program_id: str,
    source_event_id: str,
    simulation_run_ids: list[str],
    expected_epoch: int | None = None,
    expected_snapshot_hash: str | None = None,
    details: dict[str, Any] | None = None,
) -> bool:
    """Persist a once-only acknowledgement that numerical work is reachable."""

    event_id = f"{source_event_id}:dispatched"
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from experiment_programs where id = %s for update",
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramStorageNotFound(program_id)
        if (
            (expected_epoch is not None and program["current_epoch"] != expected_epoch)
            or (
                expected_snapshot_hash is not None
                and program["snapshot_hash"] != expected_snapshot_hash
            )
            or program["status"] != "simulating"
        ):
            return False
        cur.execute(
            "select 1 from experiment_program_events where event_id = %s",
            [event_id],
        )
        if cur.fetchone() is not None:
            return False
        cur.execute(
            """
            update experiment_programs
            set heartbeat_at = now(),
                simulation_dispatch_attempts = 0,
                updated_at = now()
            where id = %s
            """,
            [program_id],
        )
        _append_event_and_outbox(
            cur,
            {
                "event_id": event_id,
                "program_id": program_id,
                "epoch": program["current_epoch"],
                "event_type": "simulation_dispatched",
                "payload": {
                    "sourceEventId": source_event_id,
                    "simulationRunIds": list(simulation_run_ids),
                    "details": dict(details or {}),
                },
            },
        )
        return True


def fail_program_simulation_dispatch(
    *,
    program_id: str,
    source_event_id: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    expected_epoch: int | None = None,
    expected_snapshot_hash: str | None = None,
) -> bool:
    """Fail closed when no executor can honor the pinned simulation contract."""

    event_id = f"{source_event_id}:dispatch-failed"
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from experiment_programs where id = %s for update",
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramStorageNotFound(program_id)
        cur.execute(
            "select 1 from experiment_program_events where event_id = %s",
            [event_id],
        )
        if cur.fetchone() is not None:
            return False
        if (
            program["status"] != "simulating"
            or (
                expected_epoch is not None
                and program["current_epoch"] != expected_epoch
            )
            or (
                expected_snapshot_hash is not None
                and program["snapshot_hash"] != expected_snapshot_hash
            )
        ):
            return False
        cur.execute(
            """
            select 1 from decision_simulation_runs
            where experiment_program_id = %s
            limit 1
            """,
            [program_id],
        )
        if cur.fetchone() is not None:
            # A concurrent launcher bound a valid run after the caller's
            # list-empty observation. Never overwrite that work with failure.
            return False
        error = {
            "code": code,
            "message": message,
            "sourceEventId": source_event_id,
            "details": dict(details or {}),
        }
        cur.execute(
            """
            update experiment_programs
            set status = 'failed',
                error = %s,
                heartbeat_at = now(),
                completed_at = coalesce(completed_at, now()),
                version = version + 1,
                updated_at = now()
            where id = %s and status = 'simulating'
            returning version
            """,
            [Json(error), program_id],
        )
        updated = cur.fetchone()
        if updated is None:
            return False
        cur.execute(
            """
            update experiment_domain_frontiers
            set status = 'failed', updated_at = now()
            where program_id = %s
              and epoch = %s
              and status = 'simulating'
            """,
            [program_id, program["current_epoch"]],
        )
        _append_event_and_outbox(
            cur,
            {
                "event_id": event_id,
                "program_id": program_id,
                "epoch": program["current_epoch"],
                "event_type": "simulation_dispatch_failed",
                "payload": {
                    **error,
                    "programStatus": "failed",
                    "version": updated["version"],
                },
            },
        )
        return True


def requeue_stale_program_simulations(
    *,
    stale_before: datetime,
    limit: int = 100,
    max_reconciles: int = 3,
) -> list[str]:
    """Create durable recovery commands for stale programs without tight polling."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from experiment_programs
            where status = 'simulating'
              and coalesce(heartbeat_at, updated_at, created_at) < %s
            order by coalesce(heartbeat_at, updated_at, created_at), id
            for update skip locked
            limit %s
            """,
            [stale_before, max(1, min(int(limit), 1000))],
        )
        programs = list(cur.fetchall())
        recovered: list[str] = []
        for program in programs:
            cur.execute(
                """
                select id, status
                from decision_simulation_runs
                where experiment_program_id = %s
                order by created_at desc, id desc
                """,
                [program["id"]],
            )
            linked_runs = list(cur.fetchall())
            if any(
                row["status"]
                in {"initializing", "queued", "running", "critic_running"}
                for row in linked_runs
            ):
                cur.execute(
                    """
                    update experiment_programs
                    set heartbeat_at = now(), updated_at = now()
                    where id = %s and status = 'simulating'
                    """,
                    [program["id"]],
                )
                continue
            next_version = int(program["version"]) + 1
            next_attempt = int(program.get("simulation_dispatch_attempts") or 0)
            if not linked_runs:
                next_attempt += 1
            if not linked_runs and next_attempt >= max(1, int(max_reconciles)):
                error = {
                    "code": "simulation_dispatch_retries_exhausted",
                    "message": "No simulation run was bound before the recovery budget expired.",
                    "attempts": next_attempt,
                }
                cur.execute(
                    """
                    update experiment_programs
                    set status = 'failed',
                        error = %s,
                        heartbeat_at = now(),
                        simulation_dispatch_attempts = %s,
                        version = %s,
                        completed_at = coalesce(completed_at, now()),
                        updated_at = now()
                    where id = %s and version = %s and status = 'simulating'
                    returning id
                    """,
                    [
                        Json(error),
                        next_attempt,
                        next_version,
                        program["id"],
                        program["version"],
                    ],
                )
                if cur.fetchone() is None:
                    continue
                _append_event_and_outbox(
                    cur,
                    {
                        "event_id": (
                            f"{program['id']}:simulation-reconcile-exhausted:"
                            f"{program['current_epoch']}"
                        ),
                        "program_id": program["id"],
                        "epoch": program["current_epoch"],
                        "event_type": "simulation_dispatch_failed",
                        "payload": {
                            **error,
                            "programStatus": "failed",
                            "version": next_version,
                        },
                    },
                )
                continue
            cur.execute(
                """
                update experiment_programs
                set heartbeat_at = now(),
                    simulation_dispatch_attempts = %s,
                    version = %s,
                    updated_at = now()
                where id = %s and version = %s and status = 'simulating'
                returning id
                """,
                [
                    next_attempt,
                    next_version,
                    program["id"],
                    program["version"],
                ],
            )
            if cur.fetchone() is None:
                continue
            _append_event_and_outbox(
                cur,
                {
                    "event_id": (
                        f"{program['id']}:simulation-reconcile:"
                        f"{program['current_epoch']}:{next_version}"
                    ),
                    "program_id": program["id"],
                    "epoch": program["current_epoch"],
                    "event_type": "simulation_reconcile_requested",
                    "payload": {
                        "programId": program["id"],
                        "epoch": program["current_epoch"],
                        "snapshotHash": program["snapshot_hash"],
                        "reason": "stale_program_heartbeat",
                    },
                },
            )
            recovered.append(program["id"])
        return recovered


def link_action_experiment(
    *,
    program_id: str,
    candidate_id: str,
    experiment_id: str,
) -> bool:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_action_experiments as action
            set experiment_program_id = %s,
                experiment_candidate_id = %s,
                updated_at = now()
            from experiment_candidates as candidate,
                 manager_decisions as decision
            where action.id = %s
              and candidate.id = %s
              and candidate.program_id = %s
              and decision.id = action.decision_id
              and decision.experiment_program_id = %s
              and (action.experiment_program_id is null or action.experiment_program_id = %s)
              and (action.experiment_candidate_id is null or action.experiment_candidate_id = %s)
            returning action.id
            """,
            [
                program_id,
                candidate_id,
                experiment_id,
                candidate_id,
                program_id,
                program_id,
                program_id,
                candidate_id,
            ],
        )
        return cur.fetchone() is not None


def sync_linked_simulation_run(simulation_run_id: str) -> str | None:
    """Project one durable multi-candidate run into its program exactly once per status."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from decision_simulation_runs where id = %s for update",
            [simulation_run_id],
        )
        run = cur.fetchone()
        if run is None:
            raise ExperimentProgramStorageNotFound(simulation_run_id)
        program_id = run.get("experiment_program_id")
        if not program_id:
            return None
        cur.execute(
            "select * from experiment_programs where id = %s for update",
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramStorageNotFound(program_id)
        event_id = f"{program_id}:simulation:{simulation_run_id}:{run['status']}"
        cur.execute(
            "select 1 from experiment_program_events where event_id = %s",
            [event_id],
        )
        if cur.fetchone() is not None:
            return program_id

        cur.execute(
            """
            select simulation_candidate.*, candidate.domain
            from decision_simulation_candidates as simulation_candidate
            join experiment_candidates as candidate
              on candidate.id = simulation_candidate.experiment_candidate_id
             and candidate.program_id = simulation_candidate.experiment_program_id
            where simulation_candidate.simulation_run_id = %s
              and simulation_candidate.experiment_program_id = %s
            order by simulation_candidate.candidate_id
            """,
            [simulation_run_id, program_id],
        )
        simulation_candidates = list(cur.fetchall())
        if not simulation_candidates:
            raise ExperimentProgramStorageConflict(
                "linked simulation run has no linked candidates"
            )

        affected_domains: set[str] = set()
        for candidate in simulation_candidates:
            affected_domains.add(candidate["domain"])
            projected_status = _project_simulation_candidate_status(
                run_status=run["status"],
                candidate_status=candidate["status"],
                is_winner=bool(candidate.get("is_winner")),
            )
            score = {
                "simulationRunId": simulation_run_id,
                "simulationCandidateId": candidate["candidate_id"],
                "rank": candidate.get("rank"),
                "winner": bool(candidate.get("is_winner")),
                "distributionSummary": dict(
                    candidate.get("distribution_summary") or {}
                ),
                "scoreBreakdown": dict(candidate.get("score_breakdown") or {}),
            }
            cur.execute(
                """
                update experiment_candidates
                set status = %s,
                    score = %s,
                    updated_at = now(),
                    completed_at = case
                        when %s in (
                            'blocked_data', 'blocked_budget', 'ineligible', 'unsafe',
                            'dominated', 'measured', 'inconclusive', 'rejected',
                            'superseded', 'not_controllable'
                        ) then coalesce(completed_at, now())
                        else completed_at
                    end
                where id = %s and program_id = %s
                """,
                [
                    projected_status,
                    Json(score),
                    projected_status,
                    candidate["experiment_candidate_id"],
                    program_id,
                ],
            )

        next_status = _project_simulation_program_status(
            run_status=run["status"],
            mode=run["mode"],
            simulation_candidates=simulation_candidates,
        )
        domain_coverage = dict(program.get("domain_coverage") or {})
        for domain in affected_domains:
            coverage = dict(domain_coverage.get(domain) or {})
            coverage.update(
                {
                    "simulationRunId": simulation_run_id,
                    "simulationStatus": run["status"],
                    "simulationCompleted": run["status"]
                    in {"completed", "blocked", "failed", "superseded"},
                    "winnerCandidateId": run.get("winner_candidate_id"),
                }
            )
            domain_coverage[domain] = coverage
            frontier_status = (
                "simulating"
                if run["status"] in {"initializing", "queued", "running", "critic_running"}
                else "ready"
                if run["status"] == "completed"
                else "failed"
            )
            cur.execute(
                """
                update experiment_domain_frontiers
                set status = %s,
                    coverage = %s,
                    terminal_count = (
                        select count(*)
                        from experiment_candidates candidate
                        where candidate.program_id = %s
                          and candidate.epoch = experiment_domain_frontiers.epoch
                          and candidate.domain = experiment_domain_frontiers.domain
                          and candidate.status in (
                              'blocked_data', 'blocked_budget', 'ineligible', 'unsafe',
                              'dominated', 'measured', 'inconclusive', 'rejected',
                              'superseded', 'not_controllable'
                          )
                    ),
                    updated_at = now()
                where program_id = %s and epoch = %s and domain = %s
                """,
                [
                    frontier_status,
                    Json(coverage),
                    program_id,
                    program_id,
                    program["current_epoch"],
                    domain,
                ],
            )

        error = dict(program.get("error") or {})
        if run["status"] in {"blocked", "failed"}:
            error = {
                "code": f"decision_simulation_{run['status']}",
                "simulationRunId": simulation_run_id,
                "simulationError": dict(run.get("error") or {}),
                "criticResult": dict(run.get("critic_result") or {}),
            }
        cur.execute(
            """
            update experiment_programs
            set status = %s,
                domain_coverage = %s,
                error = %s,
                version = version + 1,
                updated_at = now(),
                heartbeat_at = now(),
                completed_at = case
                    when %s in ('exhausted', 'budget_exhausted', 'failed')
                    then coalesce(completed_at, now())
                    else completed_at
                end
            where id = %s
            """,
            [
                next_status,
                Json(domain_coverage),
                Json(error),
                next_status,
                program_id,
            ],
        )
        _append_event_and_outbox(
            cur,
            {
                "event_id": event_id,
                "program_id": program_id,
                "epoch": program["current_epoch"],
                "event_type": f"simulation_{run['status']}",
                "payload": {
                    "simulationRunId": simulation_run_id,
                    "simulationStatus": run["status"],
                    "winnerCandidateId": run.get("winner_candidate_id"),
                    "programStatus": next_status,
                    "linkedCandidateCount": len(simulation_candidates),
                },
            },
        )
        return program_id


def _project_simulation_candidate_status(
    *,
    run_status: str,
    candidate_status: str,
    is_winner: bool,
) -> str:
    if candidate_status == "needs_data":
        return "blocked_data"
    if candidate_status in {"rejected", "failed"}:
        return "rejected"
    if run_status == "completed" and candidate_status == "completed":
        return "selected" if is_winner else "simulated"
    if run_status == "superseded":
        return "superseded"
    if run_status in {"blocked", "failed"}:
        return "simulated" if candidate_status == "completed" else "rejected"
    return "running"


def _project_simulation_program_status(
    *,
    run_status: str,
    mode: str,
    simulation_candidates: list[dict[str, Any]],
) -> str:
    if run_status == "completed":
        return "awaiting_approval" if mode == "enforced" else "selecting"
    if run_status == "superseded":
        return "replanning"
    if run_status == "blocked" and any(
        candidate["status"] == "needs_data" for candidate in simulation_candidates
    ):
        return "blocked_data"
    if run_status in {"blocked", "failed"}:
        return "failed"
    return "simulating"


def _link_decision_with_cursor(cur: Any, *, decision_id: str, program_id: str) -> None:
    cur.execute(
        """
        update manager_decisions
        set experiment_program_id = %s,
            updated_at = now()
        where id = %s
          and (experiment_program_id is null or experiment_program_id = %s)
        returning id
        """,
        [program_id, decision_id, program_id],
    )
    if cur.fetchone() is None:
        raise ExperimentProgramStorageConflict(
            "source decision does not exist or belongs to another experiment program"
        )


def _append_event_and_outbox(cur: Any, event: dict[str, Any]) -> None:
    cur.execute(
        """
        insert into experiment_program_events (
            event_id, program_id, epoch, event_type, payload
        )
        values (%s, %s, %s, %s, %s)
        on conflict (event_id) do nothing
        returning id
        """,
        [
            event["event_id"],
            event["program_id"],
            event["epoch"],
            event["event_type"],
            Json(event.get("payload") or {}),
        ],
    )
    inserted = cur.fetchone()
    if inserted is None:
        return
    outbox_payload = {
        "eventId": event["event_id"],
        "programId": event["program_id"],
        "epoch": event["epoch"],
        "eventType": event["event_type"],
        "payload": event.get("payload") or {},
    }
    cur.execute(
        """
        insert into experiment_program_outbox (
            event_id, program_id, topic, message_key, payload
        )
        values (%s, %s, 'experiment.program.events', %s, %s)
        on conflict (event_id) do nothing
        """,
        [
            event["event_id"],
            event["program_id"],
            event["event_id"],
            Json(outbox_payload),
        ],
    )


def _outbox_record(row: dict[str, Any]) -> ExperimentOutboxRecord:
    return ExperimentOutboxRecord(
        id=int(row["id"]),
        event_id=row["event_id"],
        program_id=row["program_id"],
        topic=row["topic"],
        message_key=row["message_key"],
        payload=dict(row["payload"] or {}),
        status=row["status"],
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        available_at=row["available_at"],
        claimed_at=row["claimed_at"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        delivered_at=row["delivered_at"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = [
    "ExperimentOutboxRecord",
    "ExperimentProgramBundleRows",
    "ExperimentProgramStorageConflict",
    "ExperimentProgramStorageNotFound",
    "StoredNegotiationMessage",
    "append_negotiation_message_atomic",
    "bind_program_simulation_run",
    "claim_program_outbox",
    "create_program_bundle_atomic",
    "fail_program_simulation_dispatch",
    "get_program_bundle",
    "link_action_experiment",
    "link_program_decision",
    "link_simulation_run",
    "link_simulation_run_owner",
    "list_linked_simulation_runs",
    "list_negotiation_messages",
    "list_program_events",
    "mark_program_outbox_delivered",
    "record_program_simulation_dispatch",
    "requeue_stale_program_simulations",
    "reschedule_program_outbox",
    "sync_linked_simulation_run",
]
