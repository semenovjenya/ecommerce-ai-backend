from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from psycopg.types.json import Json

from agi_ecom_app.services.memory import ensure_memory_schema, get_memory_connection


class ExperimentProgramActionLinkError(RuntimeError):
    """Base error for a manager action/program linkage failure."""


class ExperimentProgramActionLinkNotFound(ExperimentProgramActionLinkError):
    pass


class ExperimentProgramActionLinkConflict(ExperimentProgramActionLinkError):
    pass


@dataclass(frozen=True)
class ExperimentProgramActionLinkResult:
    experiment_id: str
    decision_id: str
    program_id: str | None
    candidate_id: str | None
    status: Literal["legacy", "linked", "already_linked"]
    changed: bool


def sync_action_experiment_link(
    experiment_id: str,
    *,
    expected_program_id: str | None = None,
) -> ExperimentProgramActionLinkResult:
    """Idempotently attach a lifecycle row to its finite program candidate.

    Candidate identity is taken from the already durable simulation mapping.
    The exact ``candidate_key == action_id`` fallback is reserved for programs
    created without a decision simulation. The function never guesses from a
    target or operation and never overwrites a different established link.

    A decision without an experiment program is the legacy path and remains a
    no-op. Link metadata and the durable program event/outbox record are written
    in one transaction.
    """

    normalized_id = experiment_id.strip()
    if not normalized_id:
        raise ValueError("experiment_id cannot be empty")
    normalized_expected_program_id = (
        expected_program_id.strip() if expected_program_id is not None else None
    )
    if expected_program_id is not None and not normalized_expected_program_id:
        raise ValueError("expected_program_id cannot be empty")

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                action.id,
                action.decision_id,
                action.action_id,
                action.state,
                action.version,
                action.close_outcome,
                action.experiment_program_id as action_program_id,
                action.experiment_candidate_id as action_candidate_id,
                decision.experiment_program_id as decision_program_id,
                decision.simulation_run_id as decision_simulation_run_id
            from agent_action_experiments as action
            join manager_decisions as decision on decision.id = action.decision_id
            where action.id = %s
            for update of action
            """,
            [normalized_id],
        )
        action = cur.fetchone()
        if action is None:
            raise ExperimentProgramActionLinkNotFound(normalized_id)

        program_id = action.get("decision_program_id")
        action_program_id = action.get("action_program_id")
        action_candidate_id = action.get("action_candidate_id")
        if program_id is None:
            if action_program_id is not None or action_candidate_id is not None:
                raise ExperimentProgramActionLinkConflict(
                    "legacy decision has stale experiment-program link metadata"
                )
            if normalized_expected_program_id is not None:
                raise ExperimentProgramActionLinkConflict(
                    "manager decision is not linked to the expected experiment program"
                )
            return ExperimentProgramActionLinkResult(
                experiment_id=normalized_id,
                decision_id=action["decision_id"],
                program_id=None,
                candidate_id=None,
                status="legacy",
                changed=False,
            )

        if (
            normalized_expected_program_id is not None
            and program_id != normalized_expected_program_id
        ):
            raise ExperimentProgramActionLinkConflict(
                "manager decision belongs to another experiment program"
            )
        if action_program_id not in {None, program_id}:
            raise ExperimentProgramActionLinkConflict(
                "manager action already belongs to another experiment program"
            )

        cur.execute(
            """
            select current_epoch, status
            from experiment_programs
            where id = %s
            for update
            """,
            [program_id],
        )
        program = cur.fetchone()
        if program is None:
            raise ExperimentProgramActionLinkConflict(
                "manager decision references a missing experiment program"
            )

        changed = False
        if action_candidate_id is not None:
            cur.execute(
                """
                select id
                from experiment_candidates
                where id = %s and program_id = %s
                """,
                [action_candidate_id, program_id],
            )
            if cur.fetchone() is None:
                raise ExperimentProgramActionLinkConflict(
                    "manager action candidate belongs to another experiment program"
                )
            candidate_id = action_candidate_id
            if action_program_id is None:
                _write_action_link(
                    cur,
                    experiment_id=normalized_id,
                    program_id=program_id,
                    candidate_id=candidate_id,
                )
                changed = True
                status: Literal["linked", "already_linked"] = "linked"
            else:
                status = "already_linked"
        else:
            candidate_ids = _simulation_candidate_ids(
                cur,
                decision_id=action["decision_id"],
                simulation_run_id=action.get("decision_simulation_run_id"),
                program_id=program_id,
                action_id=action["action_id"],
            )
            if not candidate_ids:
                candidate_ids = _exact_candidate_key_ids(
                    cur,
                    program_id=program_id,
                    epoch=int(program["current_epoch"]),
                    action_id=action["action_id"],
                )
            if not candidate_ids:
                raise ExperimentProgramActionLinkConflict(
                    "no experiment-program candidate is durably mapped to the manager action"
                )
            if len(candidate_ids) != 1:
                raise ExperimentProgramActionLinkConflict(
                    "multiple experiment-program candidates map to the manager action"
                )
            candidate_id = candidate_ids[0]
            _write_action_link(
                cur,
                experiment_id=normalized_id,
                program_id=program_id,
                candidate_id=candidate_id,
            )
            changed = True
            status = "linked"

        _append_link_event_and_outbox(
            cur,
            experiment_id=normalized_id,
            decision_id=action["decision_id"],
            action_id=action["action_id"],
            action_state=action["state"],
            program_id=program_id,
            candidate_id=candidate_id,
            epoch=int(program["current_epoch"]),
        )
        _project_action_state(
            cur,
            experiment_id=normalized_id,
            program_id=program_id,
            candidate_id=candidate_id,
            epoch=int(program["current_epoch"]),
            action_state=str(action["state"]),
            action_version=int(action.get("version") or 1),
            close_outcome=action.get("close_outcome"),
            current_program_status=str(program.get("status") or "selecting"),
        )
        return ExperimentProgramActionLinkResult(
            experiment_id=normalized_id,
            decision_id=action["decision_id"],
            program_id=program_id,
            candidate_id=candidate_id,
            status=status,
            changed=changed,
        )


def _write_action_link(
    cur: Any,
    *,
    experiment_id: str,
    program_id: str,
    candidate_id: str,
) -> None:
    cur.execute(
        """
        update agent_action_experiments
        set experiment_program_id = %s,
            experiment_candidate_id = %s,
            updated_at = now()
        where id = %s
          and (experiment_program_id is null or experiment_program_id = %s)
          and (experiment_candidate_id is null or experiment_candidate_id = %s)
        returning id
        """,
        [
            program_id,
            candidate_id,
            experiment_id,
            program_id,
            candidate_id,
        ],
    )
    if cur.fetchone() is None:
        raise ExperimentProgramActionLinkConflict(
            "manager action link changed concurrently"
        )


def _simulation_candidate_ids(
    cur: Any,
    *,
    decision_id: str,
    simulation_run_id: str | None,
    program_id: str,
    action_id: str,
) -> list[str]:
    cur.execute(
        """
        select distinct simulation_candidate.experiment_candidate_id as candidate_id
        from decision_simulation_candidates as simulation_candidate
        join decision_simulation_runs as run
          on run.id = simulation_candidate.simulation_run_id
         and run.experiment_program_id = simulation_candidate.experiment_program_id
        join experiment_candidates as candidate
          on candidate.id = simulation_candidate.experiment_candidate_id
         and candidate.program_id = simulation_candidate.experiment_program_id
        where run.decision_id = %s
          and run.experiment_program_id = %s
          and simulation_candidate.experiment_program_id = %s
          and simulation_candidate.action_id = %s
          and simulation_candidate.experiment_candidate_id is not null
          and (%s is null or run.id = %s)
        limit 2
        """,
        [
            decision_id,
            program_id,
            program_id,
            action_id,
            simulation_run_id,
            simulation_run_id,
        ],
    )
    return [str(row["candidate_id"]) for row in cur.fetchall()]


def _exact_candidate_key_ids(
    cur: Any,
    *,
    program_id: str,
    epoch: int,
    action_id: str,
) -> list[str]:
    cur.execute(
        """
        select id as candidate_id
        from experiment_candidates
        where program_id = %s
          and epoch = %s
          and candidate_key = %s
        order by id
        limit 2
        """,
        [program_id, epoch, action_id],
    )
    return [str(row["candidate_id"]) for row in cur.fetchall()]


def _append_link_event_and_outbox(
    cur: Any,
    *,
    experiment_id: str,
    decision_id: str,
    action_id: str,
    action_state: str,
    program_id: str,
    candidate_id: str,
    epoch: int,
) -> None:
    event_id = f"{program_id}:action-experiment:{experiment_id}:linked"
    payload = {
        "experimentId": experiment_id,
        "decisionId": decision_id,
        "actionId": action_id,
        "actionState": action_state,
        "candidateId": candidate_id,
    }
    cur.execute(
        """
        insert into experiment_program_events (
            event_id, program_id, epoch, event_type, payload
        )
        values (%s, %s, %s, 'action_experiment_linked', %s)
        on conflict (event_id) do nothing
        """,
        [event_id, program_id, epoch, Json(payload)],
    )
    outbox_payload = {
        "eventId": event_id,
        "programId": program_id,
        "epoch": epoch,
        "eventType": "action_experiment_linked",
        "payload": payload,
    }
    cur.execute(
        """
        insert into experiment_program_outbox (
            event_id, program_id, topic, message_key, payload
        )
        values (%s, %s, 'experiment.program.events', %s, %s)
        on conflict (event_id) do nothing
        """,
        [event_id, program_id, event_id, Json(outbox_payload)],
    )


def _project_action_state(
    cur: Any,
    *,
    experiment_id: str,
    program_id: str,
    candidate_id: str,
    epoch: int,
    action_state: str,
    action_version: int,
    close_outcome: str | None,
    current_program_status: str,
) -> None:
    event_id = (
        f"{program_id}:action-experiment:{experiment_id}:"
        f"state:{action_version}:{action_state}"
    )
    cur.execute(
        "select 1 from experiment_program_events where event_id = %s",
        [event_id],
    )
    if cur.fetchone() is not None:
        return

    candidate_status = _candidate_status_for_action(
        action_state=action_state,
        close_outcome=close_outcome,
    )
    next_program_status = _program_status_for_action(
        current_status=current_program_status,
        action_state=action_state,
    )
    cur.execute(
        """
        update experiment_candidates
        set status = %s,
            updated_at = now(),
            completed_at = case
                when %s in ('measured', 'inconclusive', 'rejected')
                then coalesce(completed_at, now())
                else completed_at
            end
        where id = %s and program_id = %s and epoch = %s
        returning id
        """,
        [
            candidate_status,
            candidate_status,
            candidate_id,
            program_id,
            epoch,
        ],
    )
    if cur.fetchone() is None:
        raise ExperimentProgramActionLinkConflict(
            "linked experiment candidate disappeared during lifecycle projection"
        )
    cur.execute(
        """
        update experiment_domain_frontiers
        set coverage = coverage || %s,
            updated_at = now()
        where program_id = %s and epoch = %s
          and domain = (
              select domain from experiment_candidates where id = %s
          )
        """,
        [
            Json(
                {
                    "actionExperimentId": experiment_id,
                    "actionState": action_state,
                    "candidateId": candidate_id,
                }
            ),
            program_id,
            epoch,
            candidate_id,
        ],
    )
    cur.execute(
        """
        update experiment_programs
        set status = %s,
            heartbeat_at = now(),
            version = version + case when status = %s then 0 else 1 end,
            updated_at = now()
        where id = %s
        returning status, version
        """,
        [next_program_status, next_program_status, program_id],
    )
    projected_program = cur.fetchone()
    if projected_program is None:
        raise ExperimentProgramActionLinkConflict(
            "experiment program disappeared during lifecycle projection"
        )
    payload = {
        "experimentId": experiment_id,
        "candidateId": candidate_id,
        "actionState": action_state,
        "actionVersion": action_version,
        "candidateStatus": candidate_status,
        "programStatus": projected_program["status"],
        "programVersion": projected_program["version"],
        "closeOutcome": close_outcome,
    }
    cur.execute(
        """
        insert into experiment_program_events (
            event_id, program_id, epoch, event_type, payload
        )
        values (%s, %s, %s, 'action_experiment_state_changed', %s)
        on conflict (event_id) do nothing
        """,
        [event_id, program_id, epoch, Json(payload)],
    )
    outbox_payload = {
        "eventId": event_id,
        "programId": program_id,
        "epoch": epoch,
        "eventType": "action_experiment_state_changed",
        "payload": payload,
    }
    cur.execute(
        """
        insert into experiment_program_outbox (
            event_id, program_id, topic, message_key, payload
        )
        values (%s, %s, 'experiment.program.events', %s, %s)
        on conflict (event_id) do nothing
        """,
        [event_id, program_id, event_id, Json(outbox_payload)],
    )


def _candidate_status_for_action(
    *,
    action_state: str,
    close_outcome: str | None,
) -> str:
    if action_state in {"proposed", "approved"}:
        return "selected"
    if action_state in {"applying", "applied", "measuring"}:
        return "running"
    if action_state in {"measured", "closed"}:
        return "inconclusive" if close_outcome == "inconclusive" else "measured"
    if action_state in {"rejected", "blocked", "failed"}:
        return "rejected"
    raise ExperimentProgramActionLinkConflict(
        f"unsupported manager action state for program projection: {action_state}"
    )


def _program_status_for_action(*, current_status: str, action_state: str) -> str:
    if current_status in {
        "exhausted",
        "blocked_data",
        "budget_exhausted",
        "paused",
        "failed",
    }:
        return current_status
    if action_state == "proposed":
        return current_status
    if action_state in {"approved", "applying", "applied"}:
        return (
            "executing"
            if current_status in {"selecting", "awaiting_approval", "executing"}
            else current_status
        )
    if action_state == "measuring":
        return (
            "measuring"
            if current_status in {"executing", "measuring"}
            else current_status
        )
    if action_state in {"measured", "closed"}:
        return (
            "learning"
            if current_status in {"executing", "measuring", "learning"}
            else current_status
        )
    if action_state in {"rejected", "blocked", "failed"}:
        return "replanning"
    raise ExperimentProgramActionLinkConflict(
        f"unsupported manager action state for program projection: {action_state}"
    )


__all__ = [
    "ExperimentProgramActionLinkConflict",
    "ExperimentProgramActionLinkError",
    "ExperimentProgramActionLinkNotFound",
    "ExperimentProgramActionLinkResult",
    "sync_action_experiment_link",
]
