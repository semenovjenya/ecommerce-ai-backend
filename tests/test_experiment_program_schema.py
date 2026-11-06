from __future__ import annotations

from pathlib import Path

from agi_ecom_app.services.memory import MEMORY_SCHEMA_SQL


def test_runtime_schema_contains_program_frontier_journal_and_outbox() -> None:
    normalized = " ".join(MEMORY_SCHEMA_SQL.split()).lower()
    for table in (
        "experiment_programs",
        "experiment_snapshots",
        "experiment_domain_frontiers",
        "experiment_candidates",
        "experiment_data_requirements",
        "experiment_negotiation_rounds",
        "experiment_negotiation_messages",
        "experiment_candidate_dependencies",
        "experiment_candidate_conflicts",
        "experiment_program_events",
        "experiment_program_outbox",
    ):
        assert f"create table if not exists {table}" in normalized
    assert "for update skip locked" not in normalized  # claim SQL belongs to repository
    assert "experiment_negotiation_messages_append_only" in normalized
    assert "experiment_program_events_append_only" in normalized
    assert "decision_simulation_runs_program_candidate_fk" in normalized
    assert "agent_action_experiments_program_candidate_fk" in normalized


def test_fresh_postgres_bootstrap_has_the_same_program_artifacts() -> None:
    bootstrap = Path("docker/postgresmemory/02_experiment_program.sql").read_text().lower()
    runtime = MEMORY_SCHEMA_SQL.lower()
    for marker in (
        "create table if not exists experiment_programs",
        "create table if not exists experiment_candidates",
        "create table if not exists experiment_program_outbox",
        "notify_experiment_program_event",
        "experiment_snapshots_append_only",
        "decision_simulation_runs_program_decision_fk",
    ):
        assert marker in bootstrap
        assert marker in runtime
