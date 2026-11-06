from __future__ import annotations

import atexit
import os
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Json

from agi_ecom_app.config import get_settings
from agi_ecom_app.llm_runtime import build_direct_openai_client
from agi_ecom_app.observability import generation_scope, response_payload, update_observation
from agi_ecom_app.schemas import ChatContextWindow, ChatMemoryMessage, ChatRole, ChatSummaryRecord

MEMORY_SCHEMA_SQL = """
create table if not exists chat_sessions (
    chat_id text primary key,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    model_name text not null,
    status text not null default 'active'
);

create table if not exists app_runtime_settings (
    key text primary key,
    value jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create table if not exists chat_messages (
    id bigserial primary key,
    chat_id text not null references chat_sessions(chat_id) on delete cascade,
    seq integer not null,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create unique index if not exists chat_messages_chat_id_seq_idx
    on chat_messages(chat_id, seq);

create unique index if not exists chat_messages_user_client_message_id_idx
    on chat_messages(chat_id, (metadata->>'client_message_id'))
    where role = 'user' and metadata ? 'client_message_id';

create index if not exists chat_messages_chat_id_created_at_idx
    on chat_messages(chat_id, created_at desc);

create table if not exists chat_summaries (
    chat_id text primary key references chat_sessions(chat_id) on delete cascade,
    summary_text text not null,
    summarized_through_seq integer not null,
    updated_at timestamptz not null default now()
);

create table if not exists chat_selected_action_tasks (
    chat_id text not null references chat_sessions(chat_id) on delete cascade,
    task_id text not null,
    selected_at timestamptz not null default now(),
    primary key (chat_id, task_id)
);

create table if not exists agent_action_tasks (
    chat_id text not null references chat_sessions(chat_id) on delete cascade,
    task_id text not null,
    action_id text not null,
    assistant_message_seq integer not null,
    source text not null,
    factor text not null,
    observation text,
    impact text,
    cause text,
    action text not null,
    title text,
    action_type text,
    priority text,
    targets jsonb not null default '[]'::jsonb,
    success_criteria text,
    agent_prompt text,
    action_kind text,
    business_goal text,
    decision_type text,
    required_metrics jsonb not null default '[]'::jsonb,
    forecast_spec jsonb not null default '{}'::jsonb,
    execution_plan text,
    evidence jsonb not null default '[]'::jsonb,
    confidence text,
    driver_class text,
    intent text,
    raw_action jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (chat_id, task_id)
);

create index if not exists agent_action_tasks_chat_message_idx
    on agent_action_tasks(chat_id, assistant_message_seq);

create index if not exists agent_action_tasks_created_at_idx
    on agent_action_tasks(created_at desc);

alter table agent_action_tasks
    add column if not exists action_kind text,
    add column if not exists business_goal text,
    add column if not exists decision_type text,
    add column if not exists required_metrics jsonb not null default '[]'::jsonb,
    add column if not exists forecast_spec jsonb not null default '{}'::jsonb,
    add column if not exists execution_plan text;

create table if not exists agent_run_traces (
    id bigserial primary key,
    chat_id text not null references chat_sessions(chat_id) on delete cascade,
    assistant_message_seq integer,
    client_message_id text,
    intent text not null,
    primary_agent text,
    status text not null default 'running',
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    events jsonb not null default '[]'::jsonb,
    summary jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists agent_run_traces_chat_message_idx
    on agent_run_traces(chat_id, assistant_message_seq);

create index if not exists agent_run_traces_created_at_idx
    on agent_run_traces(created_at desc);

create index if not exists agent_run_traces_client_message_id_idx
    on agent_run_traces(chat_id, client_message_id)
    where client_message_id is not null;

create or replace function notify_agent_run_trace_change()
returns trigger
language plpgsql
as $$
begin
    if tg_op = 'INSERT'
       or old.status is distinct from new.status
       or old.assistant_message_seq is distinct from new.assistant_message_seq then
        perform pg_notify(
            'agent_run_trace_events',
            json_build_object(
                'run_id', new.id,
                'chat_id', new.chat_id,
                'status', new.status,
                'server_message_id', new.assistant_message_seq
            )::text
        );
    end if;
    return new;
end;
$$;

drop trigger if exists agent_run_trace_notify_trigger on agent_run_traces;
create trigger agent_run_trace_notify_trigger
after insert or update of status, assistant_message_seq on agent_run_traces
for each row execute function notify_agent_run_trace_change();

create table if not exists agent_action_executions (
    id bigserial primary key,
    source_chat_id text not null references chat_sessions(chat_id) on delete cascade,
    source_task_id text not null,
    task_id text not null,
    object_id text not null,
    object_label text,
    object_type text,
    object_external_id text,
    object_payload jsonb not null default '{}'::jsonb,
    execution_chat_id text not null references chat_sessions(chat_id) on delete cascade,
    run_id bigint references agent_run_traces(id) on delete set null,
    client_message_id text not null,
    status text not null default 'queued',
    execution_mode text not null default 'terminal_action',
    prompt text not null,
    task_snapshot jsonb not null default '{}'::jsonb,
    result_json jsonb not null default '{}'::jsonb,
    decision_status text,
    review_summary text,
    answer text,
    server_message_id integer,
    error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz
);

drop index if exists agent_action_executions_task_idx;

alter table agent_action_executions
    add column if not exists source_task_id text,
    add column if not exists object_id text,
    add column if not exists object_label text,
    add column if not exists object_type text,
    add column if not exists object_external_id text,
    add column if not exists object_payload jsonb not null default '{}'::jsonb,
    add column if not exists execution_mode text not null default 'terminal_action',
    add column if not exists result_json jsonb not null default '{}'::jsonb,
    add column if not exists decision_status text,
    add column if not exists review_summary text;

update agent_action_executions
set source_task_id = coalesce(source_task_id, task_id),
    object_id = coalesce(object_id, task_id)
where source_task_id is null or object_id is null;

alter table agent_action_executions
    alter column source_task_id set not null,
    alter column object_id set not null;

create unique index if not exists agent_action_executions_source_task_object_idx
    on agent_action_executions(source_chat_id, source_task_id, object_id);

create index if not exists agent_action_executions_source_status_idx
    on agent_action_executions(source_chat_id, status, created_at desc);

create table if not exists agent_run_rounds (
    id bigserial primary key,
    run_id bigint not null references agent_run_traces(id) on delete cascade,
    round_number integer not null,
    round_type text not null,
    status text not null default 'pending',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz
);

create index if not exists agent_run_rounds_run_id_idx
    on agent_run_rounds(run_id, round_number, id);

create table if not exists agent_run_questions (
    id bigserial primary key,
    run_id bigint not null references agent_run_traces(id) on delete cascade,
    round_id bigint not null references agent_run_rounds(id) on delete cascade,
    source_agent text,
    target_agent text not null,
    question text not null,
    reason text,
    verification_goal jsonb not null default '{}'::jsonb,
    priority integer,
    status text not null default 'pending',
    job_id text,
    queued_at timestamptz,
    attempt_count integer not null default 0,
    last_error text,
    timeout_seconds integer,
    parent_question_id bigint references agent_run_questions(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz
);

create index if not exists agent_run_questions_run_id_idx
    on agent_run_questions(run_id, round_id, id);

create index if not exists agent_run_questions_status_idx
    on agent_run_questions(status);

alter table agent_run_questions
    add column if not exists job_id text,
    add column if not exists queued_at timestamptz,
    add column if not exists attempt_count integer not null default 0,
    add column if not exists last_error text,
    add column if not exists timeout_seconds integer;

create index if not exists agent_run_questions_job_id_idx
    on agent_run_questions(job_id)
    where job_id is not null;

create table if not exists agent_run_responses (
    id bigserial primary key,
    question_id bigint not null references agent_run_questions(id) on delete cascade,
    agent_name text not null,
    status text not null default 'completed',
    summary text,
    raw_output jsonb not null default '{}'::jsonb,
    has_business_evidence boolean not null default false,
    runtime_failure jsonb,
    created_at timestamptz not null default now(),
    completed_at timestamptz
);

create index if not exists agent_run_responses_question_id_idx
    on agent_run_responses(question_id);

create table if not exists agent_run_gaps (
    id bigserial primary key,
    run_id bigint not null references agent_run_traces(id) on delete cascade,
    response_id bigint references agent_run_responses(id) on delete cascade,
    sku text,
    source_agent text,
    target_agent text,
    metric text not null,
    reason text not null,
    impact text,
    blocking boolean not null default true,
    status text not null default 'open',
    evidence_status text not null default 'unconfirmed',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists agent_run_gaps_run_id_idx
    on agent_run_gaps(run_id, status, blocking, evidence_status);

create index if not exists agent_run_gaps_sku_idx
    on agent_run_gaps(sku);

create table if not exists app_users (
    id bigserial primary key,
    username text not null,
    password_hash text not null,
    role text not null check (role in ('viewer', 'manager', 'admin')),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists app_users_username_lower_idx
    on app_users(lower(username));

create table if not exists auth_audit_log (
    id bigserial primary key,
    user_id bigint references app_users(id) on delete restrict,
    username text not null,
    event text not null,
    success boolean not null,
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists auth_audit_log_created_at_idx
    on auth_audit_log(created_at desc);

create table if not exists manager_decisions (
    id text primary key,
    source_key text unique,
    chat_id text references chat_sessions(chat_id) on delete set null,
    run_id bigint references agent_run_traces(id) on delete set null,
    decision_date date not null default ((now() at time zone 'Europe/Moscow')::date),
    objective text not null,
    status text not null constraint manager_decisions_status_check
        check (status in ('evaluating', 'ready_for_approval', 'needs_data', 'blocked')),
    request_coverage jsonb not null default '{}'::jsonb,
    evidence_as_of timestamptz,
    ranked_actions jsonb not null default '[]'::jsonb,
    rejected_candidates jsonb not null default '[]'::jsonb,
    blockers jsonb not null default '[]'::jsonb,
    decision_json jsonb not null default '{}'::jsonb,
    public_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists manager_decisions_date_idx
    on manager_decisions(decision_date desc, created_at desc);

alter table manager_decisions
    drop constraint if exists manager_decisions_status_check;

alter table manager_decisions
    add constraint manager_decisions_status_check
    check (status in ('evaluating', 'ready_for_approval', 'needs_data', 'blocked'));

create table if not exists decision_simulation_runs (
    id text primary key,
    decision_id text not null references manager_decisions(id) on delete cascade,
    chat_id text references chat_sessions(chat_id) on delete set null,
    run_id bigint references agent_run_traces(id) on delete set null,
    mode text not null check (mode in ('shadow', 'enforced')),
    status text not null default 'queued' check (
        status in ('initializing', 'queued', 'running', 'critic_running', 'completed', 'blocked', 'failed', 'superseded')
    ),
    objective text not null default 'incremental_contribution_profit',
    model_version text not null,
    evidence_snapshot jsonb not null default '{}'::jsonb,
    evidence_hash text not null,
    master_seed numeric(20, 0) not null,
    sample_count integer not null default 500 check (sample_count > 0),
    batch_size integer not null default 50 check (batch_size > 0),
    candidate_count integer not null default 0 check (candidate_count >= 0),
    completed_samples integer not null default 0 check (completed_samples >= 0),
    total_samples integer not null default 0 check (total_samples >= 0),
    completed_candidates integer not null default 0 check (completed_candidates >= 0),
    progress numeric(7, 6) not null default 0 check (progress >= 0 and progress <= 1),
    provisional_winner_candidate_id text,
    winner_candidate_id text,
    critic_result jsonb not null default '{}'::jsonb,
    error jsonb not null default '{}'::jsonb,
    idempotency_key text,
    retry_of_run_id text references decision_simulation_runs(id) on delete set null,
    superseded_by_run_id text references decision_simulation_runs(id) on delete set null,
    retry_attempt integer not null default 0 check (retry_attempt >= 0),
    heartbeat_at timestamptz,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table decision_simulation_runs
    drop constraint if exists decision_simulation_runs_status_check;

alter table decision_simulation_runs
    add constraint decision_simulation_runs_status_check
    check (
        status in ('initializing', 'queued', 'running', 'critic_running', 'completed', 'blocked', 'failed', 'superseded')
    );

create index if not exists decision_simulation_runs_decision_idx
    on decision_simulation_runs(decision_id, created_at desc, id desc);

create index if not exists decision_simulation_runs_status_idx
    on decision_simulation_runs(status, heartbeat_at, created_at);

drop index if exists decision_simulation_runs_idempotency_idx;

create unique index if not exists decision_simulation_runs_initial_idempotency_idx
    on decision_simulation_runs(decision_id, idempotency_key)
    where retry_of_run_id is null and idempotency_key is not null;

create unique index if not exists decision_simulation_runs_retry_idempotency_idx
    on decision_simulation_runs(retry_of_run_id, idempotency_key)
    where retry_of_run_id is not null and idempotency_key is not null;

create unique index if not exists decision_simulation_runs_retry_parent_idx
    on decision_simulation_runs(retry_of_run_id)
    where retry_of_run_id is not null;

create table if not exists decision_simulation_candidates (
    simulation_run_id text not null references decision_simulation_runs(id) on delete cascade,
    candidate_id text not null,
    action_id text not null,
    status text not null default 'pending' check (
        status in ('pending', 'running', 'completed', 'rejected', 'needs_data', 'failed')
    ),
    candidate_payload jsonb not null default '{}'::jsonb,
    evidence_refs jsonb not null default '[]'::jsonb,
    constraint_violations jsonb not null default '[]'::jsonb,
    distribution_summary jsonb not null default '{}'::jsonb,
    score_breakdown jsonb not null default '{}'::jsonb,
    expected_effect_mean double precision,
    profit_p10 double precision,
    cvar_10_per_day double precision,
    execution_risk double precision,
    utility_score double precision,
    rank integer check (rank is null or rank > 0),
    is_winner boolean not null default false,
    completed_samples integer not null default 0 check (completed_samples >= 0),
    error jsonb not null default '{}'::jsonb,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (simulation_run_id, candidate_id)
);

create index if not exists decision_simulation_candidates_ranking_idx
    on decision_simulation_candidates(simulation_run_id, rank, candidate_id);

create table if not exists decision_simulation_batches (
    id bigserial primary key,
    simulation_run_id text not null,
    candidate_id text not null,
    batch_index integer not null check (batch_index >= 0),
    seed numeric(20, 0) not null,
    sample_count integer not null check (sample_count > 0),
    status text not null default 'pending' check (status in ('pending', 'running', 'completed', 'failed')),
    aggregates jsonb not null default '{}'::jsonb,
    duration_ms double precision,
    error jsonb not null default '{}'::jsonb,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (simulation_run_id, candidate_id, batch_index),
    foreign key (simulation_run_id, candidate_id)
        references decision_simulation_candidates(simulation_run_id, candidate_id)
        on delete cascade
);

create index if not exists decision_simulation_batches_run_status_idx
    on decision_simulation_batches(simulation_run_id, status, candidate_id, batch_index);

create table if not exists decision_simulation_events (
    id bigserial primary key,
    event_id text not null unique,
    simulation_run_id text not null references decision_simulation_runs(id) on delete cascade,
    event_type text not null,
    candidate_id text,
    batch_index integer,
    progress numeric(7, 6) check (progress is null or (progress >= 0 and progress <= 1)),
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists decision_simulation_events_run_idx
    on decision_simulation_events(simulation_run_id, id);

alter table manager_decisions
    add column if not exists simulation_run_id text;

do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'manager_decisions_simulation_run_id_fkey'
    ) then
        alter table manager_decisions
            add constraint manager_decisions_simulation_run_id_fkey
            foreign key (simulation_run_id) references decision_simulation_runs(id) on delete set null;
    end if;
end;
$$;

create index if not exists manager_decisions_simulation_run_idx
    on manager_decisions(simulation_run_id);

create table if not exists agent_action_experiments (
    id text primary key,
    decision_id text not null references manager_decisions(id) on delete cascade,
    action_id text not null,
    family text not null,
    target jsonb not null default '{}'::jsonb,
    execution_mode text not null check (execution_mode in ('human_task', 'wb_ads', 'wb_price')),
    state text not null default 'proposed' check (
        state in (
            'proposed', 'approved', 'applying', 'applied', 'measuring', 'measured',
            'closed', 'blocked', 'rejected', 'failed'
        )
    ),
    from_value jsonb not null default '{}'::jsonb,
    to_value jsonb not null default '{}'::jsonb,
    expected_effect jsonb not null default '{}'::jsonb,
    horizon jsonb not null default '{}'::jsonb,
    evidence jsonb not null default '[]'::jsonb,
    prerequisites jsonb not null default '[]'::jsonb,
    guardrails jsonb not null default '{}'::jsonb,
    stop_condition jsonb not null default '{}'::jsonb,
    rollback jsonb not null default '{}'::jsonb,
    measurement_spec jsonb not null default '{}'::jsonb,
    expires_at timestamptz,
    owner_user_id bigint references app_users(id) on delete set null,
    approved_by bigint references app_users(id) on delete set null,
    approved_at timestamptz,
    approval_idempotency_key text,
    rejected_by bigint references app_users(id) on delete set null,
    rejected_at timestamptz,
    rejection_idempotency_key text,
    rejection_reason text,
    manager_label text check (
        manager_label is null or manager_label in (
            'wrong_data', 'wrong_priority', 'unsafe', 'not_executable', 'missing_context'
        )
    ),
    preflight_json jsonb not null default '{}'::jsonb,
    actuator_result jsonb not null default '{}'::jsonb,
    external_operation_id text,
    applied_at timestamptz,
    measurement_due_at timestamptz,
    close_outcome text check (
        close_outcome is null or close_outcome in ('scale', 'hold', 'rollback', 'inconclusive')
    ),
    last_error text,
    version integer not null default 1,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (decision_id, action_id)
);

create index if not exists agent_action_experiments_state_idx
    on agent_action_experiments(state, measurement_due_at, created_at desc);

create table if not exists agent_action_measurements (
    id bigserial primary key,
    experiment_id text not null references agent_action_experiments(id) on delete cascade,
    phase text not null check (phase in ('baseline', 'post', 'control', 'readout')),
    status text not null check (status in ('waiting_for_data', 'observed', 'computed', 'inconclusive')),
    period_start timestamptz,
    period_end timestamptz,
    data_as_of timestamptz,
    idempotency_key text,
    metrics jsonb not null default '{}'::jsonb,
    causal_valid boolean not null default false,
    observed_delta numeric,
    did_effect numeric,
    forecast_low numeric,
    forecast_base numeric,
    forecast_high numeric,
    signed_bias numeric,
    mae numeric,
    inside_forecast_range boolean,
    created_by bigint references app_users(id) on delete restrict,
    created_at timestamptz not null default now()
);

alter table agent_action_measurements
    add column if not exists idempotency_key text;

create unique index if not exists agent_action_measurements_idempotency_idx
    on agent_action_measurements(experiment_id, idempotency_key)
    where idempotency_key is not null;

create index if not exists agent_action_measurements_experiment_idx
    on agent_action_measurements(experiment_id, created_at, id);

create table if not exists agent_action_audit_log (
    id bigserial primary key,
    experiment_id text not null references agent_action_experiments(id) on delete cascade,
    actor_user_id bigint references app_users(id) on delete restrict,
    event text not null,
    from_state text,
    to_state text,
    idempotency_key text,
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists agent_action_audit_log_experiment_idx
    on agent_action_audit_log(experiment_id, created_at, id);

create table if not exists experiment_programs (
    id text primary key,
    idempotency_key text not null unique,
    request_hash text not null,
    objective text not null,
    mode text not null check (mode in ('shadow', 'enforced')),
    status text not null default 'created' check (
        status in (
            'created', 'data_audit', 'proposing', 'negotiating', 'simulating',
            'selecting', 'awaiting_approval', 'executing', 'measuring',
            'learning', 'replanning', 'exhausted', 'blocked_data',
            'budget_exhausted', 'paused', 'failed'
        )
    ),
    target jsonb not null default '{}'::jsonb,
    domains jsonb not null default '[]'::jsonb,
    generator_version text not null,
    snapshot_hash text not null,
    current_epoch integer not null default 1 check (current_epoch > 0),
    experiment_budget jsonb not null default '{}'::jsonb,
    domain_coverage jsonb not null default '{}'::jsonb,
    source_chat_id text references chat_sessions(chat_id) on delete set null,
    source_run_id bigint references agent_run_traces(id) on delete set null,
    source_decision_id text references manager_decisions(id) on delete set null,
    heartbeat_at timestamptz,
    simulation_dispatch_attempts integer not null default 0 check (
        simulation_dispatch_attempts >= 0
    ),
    error jsonb not null default '{}'::jsonb,
    version integer not null default 1 check (version > 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz
);

create index if not exists experiment_programs_status_idx
    on experiment_programs(status, heartbeat_at, updated_at);

alter table experiment_programs
    add column if not exists simulation_dispatch_attempts integer not null default 0
    check (simulation_dispatch_attempts >= 0);

create index if not exists experiment_programs_target_idx
    on experiment_programs using gin(target);

create table if not exists experiment_snapshots (
    id text primary key,
    program_id text not null references experiment_programs(id) on delete cascade,
    epoch integer not null check (epoch > 0),
    snapshot_hash text not null,
    status text not null check (status in ('ready', 'blocked_data', 'conflicting')),
    data_as_of timestamptz,
    evidence_packets jsonb not null default '[]'::jsonb,
    quality_summary jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (program_id, epoch),
    unique (program_id, snapshot_hash)
);

create table if not exists experiment_domain_frontiers (
    program_id text not null references experiment_programs(id) on delete cascade,
    domain text not null check (
        domain in ('pricing', 'advertising', 'supply', 'sales', 'finance')
    ),
    epoch integer not null check (epoch > 0),
    generator_version text not null,
    snapshot_hash text not null,
    status text not null default 'pending' check (
        status in (
            'pending', 'blocked_data', 'ready', 'negotiating', 'simulating',
            'exhausted', 'constraint_only', 'no_controllable_lever', 'failed'
        )
    ),
    candidate_count integer not null default 0 check (candidate_count >= 0),
    eligible_count integer not null default 0 check (eligible_count >= 0),
    terminal_count integer not null default 0 check (terminal_count >= 0),
    coverage jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz,
    primary key (program_id, domain, epoch)
);

create table if not exists experiment_candidates (
    id text primary key,
    program_id text not null references experiment_programs(id) on delete cascade,
    epoch integer not null check (epoch > 0),
    candidate_key text not null,
    domain text not null check (
        domain in ('pricing', 'advertising', 'supply', 'sales', 'finance')
    ),
    family text not null,
    operation text not null,
    target jsonb not null default '{}'::jsonb,
    parameters jsonb not null default '{}'::jsonb,
    hypothesis text,
    primary_metric text,
    guardrail_metrics jsonb not null default '[]'::jsonb,
    required_metrics jsonb not null default '[]'::jsonb,
    evidence_refs jsonb not null default '[]'::jsonb,
    evidence_status text not null default 'missing' check (
        evidence_status in (
            'available', 'missing', 'stale', 'conflicting', 'causally_insufficient'
        )
    ),
    causal_design jsonb not null default '{}'::jsonb,
    estimated_cost jsonb not null default '{}'::jsonb,
    risk jsonb not null default '{}'::jsonb,
    reversibility jsonb not null default '{}'::jsonb,
    stop_conditions jsonb not null default '[]'::jsonb,
    status text not null default 'unseen' check (
        status in (
            'unseen', 'eligible', 'blocked_data', 'blocked_budget', 'ineligible',
            'unsafe', 'dominated', 'simulated', 'selected', 'running', 'measured',
            'inconclusive', 'rejected', 'superseded', 'not_controllable'
        )
    ),
    score jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz,
    unique (program_id, epoch, candidate_key)
);

create index if not exists experiment_candidates_frontier_idx
    on experiment_candidates(program_id, epoch, domain, status, candidate_key);

create unique index if not exists experiment_candidates_program_identity_idx
    on experiment_candidates(id, program_id);

create table if not exists experiment_data_requirements (
    id bigserial primary key,
    program_id text not null references experiment_programs(id) on delete cascade,
    candidate_id text references experiment_candidates(id) on delete cascade,
    domain text not null,
    metric text not null,
    status text not null check (
        status in ('available', 'missing', 'stale', 'conflicting', 'causally_insufficient')
    ),
    evidence_ref text,
    source_table text,
    grain text,
    data_as_of timestamptz,
    max_age_seconds integer check (max_age_seconds is null or max_age_seconds >= 0),
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (program_id, candidate_id, metric)
);

create index if not exists experiment_data_requirements_status_idx
    on experiment_data_requirements(program_id, status, domain, metric);

create table if not exists experiment_negotiation_rounds (
    id bigserial primary key,
    program_id text not null references experiment_programs(id) on delete cascade,
    epoch integer not null check (epoch > 0),
    round_number integer not null check (round_number > 0),
    round_type text not null check (
        round_type in ('propose', 'challenge', 'evidence_resolution', 'revise_and_close')
    ),
    status text not null default 'open' check (status in ('open', 'completed', 'failed')),
    input_fingerprint text not null,
    output_fingerprint text,
    created_at timestamptz not null default now(),
    completed_at timestamptz,
    unique (program_id, epoch, round_number),
    unique (program_id, epoch, input_fingerprint)
);

create table if not exists experiment_negotiation_messages (
    id bigserial primary key,
    message_id text not null unique,
    program_id text not null references experiment_programs(id) on delete cascade,
    round_id bigint references experiment_negotiation_rounds(id) on delete set null,
    epoch integer not null check (epoch > 0),
    message_type text not null check (
        message_type in (
            'proposal', 'challenge', 'evidence_request', 'evidence_response',
            'constraint', 'proposal_revision', 'domain_done'
        )
    ),
    source_role text not null,
    target_role text,
    correlation_id text,
    idempotency_key text not null,
    fingerprint text not null,
    payload jsonb not null default '{}'::jsonb,
    evidence_refs jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    unique (program_id, idempotency_key),
    unique (program_id, epoch, fingerprint)
);

create index if not exists experiment_negotiation_messages_program_idx
    on experiment_negotiation_messages(program_id, id);

create table if not exists experiment_candidate_dependencies (
    program_id text not null references experiment_programs(id) on delete cascade,
    candidate_id text not null references experiment_candidates(id) on delete cascade,
    depends_on_candidate_id text not null references experiment_candidates(id) on delete cascade,
    dependency_type text not null default 'requires' check (
        dependency_type in ('requires', 'precedes', 'shares_evidence')
    ),
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (candidate_id, depends_on_candidate_id, dependency_type),
    check (candidate_id <> depends_on_candidate_id)
);

create table if not exists experiment_candidate_conflicts (
    program_id text not null references experiment_programs(id) on delete cascade,
    left_candidate_id text not null references experiment_candidates(id) on delete cascade,
    right_candidate_id text not null references experiment_candidates(id) on delete cascade,
    interference_key text not null,
    conflict_type text not null default 'mutually_exclusive' check (
        conflict_type in ('mutually_exclusive', 'causal_interference', 'budget', 'capacity')
    ),
    details jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (left_candidate_id, right_candidate_id, interference_key),
    check (left_candidate_id < right_candidate_id)
);

create table if not exists experiment_program_events (
    id bigserial primary key,
    event_id text not null unique,
    program_id text not null references experiment_programs(id) on delete cascade,
    epoch integer not null check (epoch > 0),
    event_type text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists experiment_program_events_program_idx
    on experiment_program_events(program_id, id);

create table if not exists experiment_program_outbox (
    id bigserial primary key,
    event_id text not null unique,
    program_id text not null references experiment_programs(id) on delete cascade,
    topic text not null,
    message_key text not null,
    payload jsonb not null default '{}'::jsonb,
    status text not null default 'pending' check (
        status in ('pending', 'processing', 'delivered', 'dead')
    ),
    attempt_count integer not null default 0 check (attempt_count >= 0),
    max_attempts integer not null default 10 check (max_attempts > 0),
    available_at timestamptz not null default now(),
    claimed_at timestamptz,
    lease_owner text,
    lease_expires_at timestamptz,
    delivered_at timestamptz,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists experiment_program_outbox_delivery_idx
    on experiment_program_outbox(status, available_at, id);

create index if not exists experiment_program_outbox_lease_idx
    on experiment_program_outbox(lease_expires_at, id)
    where status = 'processing';

alter table decision_simulation_runs
    add column if not exists experiment_program_id text references experiment_programs(id) on delete set null,
    add column if not exists experiment_candidate_id text references experiment_candidates(id) on delete set null;

create unique index if not exists decision_simulation_runs_program_identity_idx
    on decision_simulation_runs(id, experiment_program_id);

alter table decision_simulation_candidates
    add column if not exists experiment_program_id text references experiment_programs(id) on delete set null,
    add column if not exists experiment_candidate_id text references experiment_candidates(id) on delete set null;

create index if not exists decision_simulation_candidates_program_idx
    on decision_simulation_candidates(experiment_program_id, experiment_candidate_id, rank);

create unique index if not exists decision_simulation_candidates_program_identity_idx
    on decision_simulation_candidates(simulation_run_id, experiment_candidate_id)
    where experiment_candidate_id is not null;

alter table manager_decisions
    add column if not exists experiment_program_id text references experiment_programs(id) on delete set null;

create unique index if not exists manager_decisions_program_identity_idx
    on manager_decisions(id, experiment_program_id);

create index if not exists manager_decisions_program_idx
    on manager_decisions(experiment_program_id, created_at desc);

create index if not exists decision_simulation_runs_program_idx
    on decision_simulation_runs(experiment_program_id, created_at desc);

alter table agent_action_experiments
    add column if not exists experiment_program_id text references experiment_programs(id) on delete set null,
    add column if not exists experiment_candidate_id text references experiment_candidates(id) on delete set null;

create index if not exists agent_action_experiments_program_idx
    on agent_action_experiments(experiment_program_id, state, created_at desc);

do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'decision_simulation_runs_program_decision_fk'
    ) then
        alter table decision_simulation_runs
            add constraint decision_simulation_runs_program_decision_fk
            foreign key (decision_id, experiment_program_id)
            references manager_decisions(id, experiment_program_id);
    end if;
    if not exists (
        select 1 from pg_constraint
        where conname = 'decision_simulation_runs_program_candidate_fk'
    ) then
        alter table decision_simulation_runs
            add constraint decision_simulation_runs_program_candidate_fk
            foreign key (experiment_candidate_id, experiment_program_id)
            references experiment_candidates(id, program_id);
    end if;
    if not exists (
        select 1 from pg_constraint
        where conname = 'decision_simulation_candidates_program_run_fk'
    ) then
        alter table decision_simulation_candidates
            add constraint decision_simulation_candidates_program_run_fk
            foreign key (simulation_run_id, experiment_program_id)
            references decision_simulation_runs(id, experiment_program_id);
    end if;
    if not exists (
        select 1 from pg_constraint
        where conname = 'decision_simulation_candidates_program_candidate_fk'
    ) then
        alter table decision_simulation_candidates
            add constraint decision_simulation_candidates_program_candidate_fk
            foreign key (experiment_candidate_id, experiment_program_id)
            references experiment_candidates(id, program_id);
    end if;
    if not exists (
        select 1 from pg_constraint
        where conname = 'agent_action_experiments_program_decision_fk'
    ) then
        alter table agent_action_experiments
            add constraint agent_action_experiments_program_decision_fk
            foreign key (decision_id, experiment_program_id)
            references manager_decisions(id, experiment_program_id);
    end if;
    if not exists (
        select 1 from pg_constraint
        where conname = 'agent_action_experiments_program_candidate_fk'
    ) then
        alter table agent_action_experiments
            add constraint agent_action_experiments_program_candidate_fk
            foreign key (experiment_candidate_id, experiment_program_id)
            references experiment_candidates(id, program_id);
    end if;
end;
$$;

create or replace function notify_experiment_program_event()
returns trigger language plpgsql as $$
begin
    perform pg_notify(
        'experiment_program_events',
        json_build_object(
            'program_id', new.program_id,
            'event_id', new.event_id,
            'event_type', new.event_type,
            'row_id', new.id
        )::text
    );
    return new;
end;
$$;

drop trigger if exists experiment_program_event_notify_trigger on experiment_program_events;
create trigger experiment_program_event_notify_trigger
after insert on experiment_program_events
for each row execute function notify_experiment_program_event();

create or replace function reject_append_only_mutation()
returns trigger language plpgsql as $$
begin
    raise exception '% is append-only', tg_table_name;
end;
$$;

drop trigger if exists agent_action_measurements_append_only on agent_action_measurements;
create trigger agent_action_measurements_append_only
before update or delete on agent_action_measurements
for each row execute function reject_append_only_mutation();

drop trigger if exists auth_audit_log_append_only on auth_audit_log;
create trigger auth_audit_log_append_only
before update or delete on auth_audit_log
for each row execute function reject_append_only_mutation();

drop trigger if exists agent_action_audit_log_append_only on agent_action_audit_log;
create trigger agent_action_audit_log_append_only
before update or delete on agent_action_audit_log
for each row execute function reject_append_only_mutation();

drop trigger if exists decision_simulation_events_append_only on decision_simulation_events;
create trigger decision_simulation_events_append_only
before update or delete on decision_simulation_events
for each row execute function reject_append_only_mutation();

drop trigger if exists experiment_snapshots_append_only on experiment_snapshots;
create trigger experiment_snapshots_append_only
before update or delete on experiment_snapshots
for each row execute function reject_append_only_mutation();

drop trigger if exists experiment_negotiation_messages_append_only on experiment_negotiation_messages;
create trigger experiment_negotiation_messages_append_only
before update or delete on experiment_negotiation_messages
for each row execute function reject_append_only_mutation();

drop trigger if exists experiment_program_events_append_only on experiment_program_events;
create trigger experiment_program_events_append_only
before update or delete on experiment_program_events
for each row execute function reject_append_only_mutation();
"""

MEMORY_SCHEMA_ADVISORY_LOCK_ID = 810_202_604
_MEMORY_SCHEMA_READY = os.getenv("MEMORY_SCHEMA_PREINITIALIZED", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_MEMORY_SCHEMA_LOCK = Lock()
_UNSET = object()


@dataclass(frozen=True)
class StoredChatMessage:
    seq: int
    role: ChatRole
    content: str
    metadata: dict[str, Any]
    created_at: datetime | None = None


@dataclass(frozen=True)
class ChatSessionSummaryRecord:
    chat_id: str
    status: str
    model_name: str | None
    created_at: datetime | None
    updated_at: datetime | None
    experiment_program_id: str | None
    experiment_candidate_id: str | None
    message_count: int
    run_count: int
    latest_run_id: int | None
    latest_run_status: str | None
    latest_run_intent: str | None
    latest_run_created_at: datetime | None
    latest_server_message_id: int | None
    last_user_message: str | None
    last_user_message_at: datetime | None
    last_assistant_message: str | None
    last_assistant_message_at: datetime | None


@dataclass(frozen=True)
class AgentRunTraceRecord:
    id: int
    chat_id: str
    assistant_message_seq: int | None
    client_message_id: str | None
    intent: str
    primary_agent: str | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    created_at: datetime | None = None


@dataclass(frozen=True)
class AgentRunRoundRecord:
    id: int
    run_id: int
    round_number: int
    round_type: str
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class AgentRunQuestionRecord:
    id: int
    run_id: int
    round_id: int
    source_agent: str | None
    target_agent: str
    question: str
    reason: str | None
    verification_goal: dict[str, Any]
    priority: int | None
    status: str
    job_id: str | None = None
    queued_at: datetime | None = None
    attempt_count: int = 0
    last_error: str | None = None
    timeout_seconds: int | None = None
    parent_question_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class AgentRunResponseRecord:
    id: int
    question_id: int
    agent_name: str
    status: str
    summary: str | None
    raw_output: dict[str, Any]
    has_business_evidence: bool
    runtime_failure: dict[str, Any] | None
    created_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class AgentRunGapRecord:
    id: int
    run_id: int
    response_id: int | None
    sku: str | None
    source_agent: str | None
    target_agent: str | None
    metric: str
    reason: str
    impact: str | None
    blocking: bool
    status: str
    evidence_status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class StoredAgentActionTask:
    chat_id: str
    task_id: str
    action_id: str
    assistant_message_seq: int
    source: str
    factor: str
    observation: str | None
    impact: str | None
    cause: str | None
    action: str
    title: str | None
    action_type: str | None
    priority: str | None
    targets: list[str]
    success_criteria: str | None
    agent_prompt: str | None
    evidence: list[str]
    confidence: str | None
    driver_class: str | None
    intent: str | None
    raw_action: dict[str, Any]
    action_kind: str | None = None
    business_goal: str | None = None
    decision_type: str | None = None
    required_metrics: list[str] = field(default_factory=list)
    forecast_spec: dict[str, Any] = field(default_factory=dict)
    execution_plan: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class AgentActionExecutionRecord:
    id: int
    source_chat_id: str
    task_id: str
    execution_chat_id: str
    run_id: int | None
    client_message_id: str
    status: str
    prompt: str
    task_snapshot: dict[str, Any]
    answer: str | None
    server_message_id: int | None
    error: str | None
    source_task_id: str = ""
    object_id: str = ""
    object_label: str | None = None
    object_type: str | None = None
    object_external_id: str | None = None
    object_payload: dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "terminal_action"
    result_json: dict[str, Any] = field(default_factory=dict)
    decision_status: str | None = None
    review_summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class AppUserRecord:
    id: int
    username: str
    password_hash: str
    role: str
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ManagerDecisionRecord:
    id: str
    source_key: str | None
    chat_id: str | None
    run_id: int | None
    decision_date: Any
    objective: str
    status: str
    request_coverage: dict[str, Any]
    evidence_as_of: datetime | None
    ranked_actions: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]
    blockers: list[Any]
    decision_json: dict[str, Any]
    public_json: dict[str, Any]
    simulation_run_id: str | None = None
    experiment_program_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class DecisionSimulationRunRecord:
    id: str
    decision_id: str
    chat_id: str | None
    run_id: int | None
    mode: str
    status: str
    objective: str
    model_version: str
    evidence_snapshot: dict[str, Any]
    evidence_hash: str
    master_seed: int
    sample_count: int
    batch_size: int
    candidate_count: int
    completed_samples: int
    total_samples: int
    completed_candidates: int
    progress: float
    provisional_winner_candidate_id: str | None
    winner_candidate_id: str | None
    critic_result: dict[str, Any]
    error: dict[str, Any]
    idempotency_key: str | None
    retry_of_run_id: str | None
    superseded_by_run_id: str | None
    retry_attempt: int
    heartbeat_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    experiment_program_id: str | None = None
    experiment_candidate_id: str | None = None


@dataclass(frozen=True)
class DecisionSimulationCandidateRecord:
    simulation_run_id: str
    candidate_id: str
    action_id: str
    status: str
    candidate_payload: dict[str, Any]
    evidence_refs: list[Any]
    constraint_violations: list[Any]
    distribution_summary: dict[str, Any]
    score_breakdown: dict[str, Any]
    expected_effect_mean: float | None
    profit_p10: float | None
    cvar_10_per_day: float | None
    execution_risk: float | None
    utility_score: float | None
    rank: int | None
    is_winner: bool
    completed_samples: int
    error: dict[str, Any]
    completed_at: datetime | None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    experiment_program_id: str | None = None
    experiment_candidate_id: str | None = None


@dataclass(frozen=True)
class DecisionSimulationBatchRecord:
    id: int
    simulation_run_id: str
    candidate_id: str
    batch_index: int
    seed: int
    sample_count: int
    status: str
    aggregates: dict[str, Any]
    duration_ms: float | None
    error: dict[str, Any]
    completed_at: datetime | None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class DecisionSimulationEventRecord:
    id: int
    event_id: str
    simulation_run_id: str
    event_type: str
    candidate_id: str | None
    batch_index: int | None
    progress: float | None
    payload: dict[str, Any]
    created_at: datetime | None = None


@dataclass(frozen=True)
class AgentActionExperimentRecord:
    id: str
    decision_id: str
    action_id: str
    family: str
    target: dict[str, Any]
    execution_mode: str
    state: str
    from_value: Any
    to_value: Any
    expected_effect: dict[str, Any]
    horizon: dict[str, Any]
    evidence: list[Any]
    prerequisites: list[Any]
    guardrails: dict[str, Any]
    stop_condition: dict[str, Any]
    rollback: dict[str, Any]
    measurement_spec: dict[str, Any]
    expires_at: datetime | None
    owner_user_id: int | None
    approved_by: int | None
    approved_at: datetime | None
    approval_idempotency_key: str | None
    rejected_by: int | None
    rejected_at: datetime | None
    rejection_idempotency_key: str | None
    rejection_reason: str | None
    manager_label: str | None
    preflight_json: dict[str, Any]
    actuator_result: dict[str, Any]
    external_operation_id: str | None
    applied_at: datetime | None
    measurement_due_at: datetime | None
    close_outcome: str | None
    last_error: str | None
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    experiment_program_id: str | None = None
    experiment_candidate_id: str | None = None


@dataclass(frozen=True)
class AgentActionMeasurementRecord:
    id: int
    experiment_id: str
    phase: str
    status: str
    period_start: datetime | None
    period_end: datetime | None
    data_as_of: datetime | None
    idempotency_key: str | None
    metrics: dict[str, Any]
    causal_valid: bool
    observed_delta: float | None
    did_effect: float | None
    forecast_low: float | None
    forecast_base: float | None
    forecast_high: float | None
    signed_bias: float | None
    mae: float | None
    inside_forecast_range: bool | None
    created_by: int | None
    created_at: datetime | None = None


_MEMORY_POOL: ConnectionPool | None = None
_MEMORY_POOL_DSN: str | None = None
_MEMORY_POOL_LOCK = Lock()


def _get_memory_pool() -> ConnectionPool:
    global _MEMORY_POOL, _MEMORY_POOL_DSN
    settings = get_settings()
    dsn = settings.memory_postgres_dsn
    if _MEMORY_POOL is not None and _MEMORY_POOL_DSN == dsn:
        return _MEMORY_POOL
    with _MEMORY_POOL_LOCK:
        if _MEMORY_POOL is not None and _MEMORY_POOL_DSN == dsn:
            return _MEMORY_POOL
        if _MEMORY_POOL is not None:
            _MEMORY_POOL.close()
        min_size = int(getattr(settings, "memory_postgres_pool_min_size", 1))
        max_size = max(
            min_size,
            int(getattr(settings, "memory_postgres_pool_max_size", 10)),
        )
        _MEMORY_POOL = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=float(
                getattr(settings, "memory_postgres_pool_timeout_seconds", 10.0)
            ),
            kwargs={"row_factory": dict_row},
            open=True,
            name="agi-ecom-memory",
        )
        _MEMORY_POOL_DSN = dsn
        return _MEMORY_POOL


def close_memory_connection_pool() -> None:
    global _MEMORY_POOL, _MEMORY_POOL_DSN
    with _MEMORY_POOL_LOCK:
        if _MEMORY_POOL is not None:
            _MEMORY_POOL.close()
        _MEMORY_POOL = None
        _MEMORY_POOL_DSN = None


def get_memory_connection() -> AbstractContextManager[psycopg.Connection]:
    """Borrow a transaction-scoped connection from the process-local pool."""

    return _get_memory_pool().connection()


atexit.register(close_memory_connection_pool)


def ping_memory_db() -> None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute("select 1 as ok")
        cur.fetchone()


def ensure_memory_schema() -> None:
    global _MEMORY_SCHEMA_READY
    if _MEMORY_SCHEMA_READY:
        return

    with _MEMORY_SCHEMA_LOCK:
        if _MEMORY_SCHEMA_READY:
            return
        with get_memory_connection() as conn, conn.cursor() as cur:
            # A transaction-scoped lock is released by COMMIT *or* ROLLBACK.
            # A session lock is unsafe with a pool: failed DDL aborts the
            # transaction, prevents the explicit unlock query, and returns a
            # still-locked session to the pool.
            cur.execute(
                "select pg_advisory_xact_lock(%s)",
                [MEMORY_SCHEMA_ADVISORY_LOCK_ID],
            )
            cur.execute(MEMORY_SCHEMA_SQL)
        _MEMORY_SCHEMA_READY = True


def ensure_chat_session(chat_id: str, model_name: str, status: str = "active") -> None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into chat_sessions (chat_id, model_name, status)
            values (%s, %s, %s)
            on conflict (chat_id) do update
            set updated_at = now(),
                model_name = excluded.model_name,
                status = excluded.status
            """,
            [chat_id, model_name, status],
        )


def create_chat_session(model_name: str) -> str:
    import uuid

    chat_id = str(uuid.uuid4())
    ensure_chat_session(chat_id=chat_id, model_name=model_name)
    return chat_id


def chat_session_exists(chat_id: str) -> bool:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select 1 as ok
            from chat_sessions
            where chat_id = %s
            limit 1
            """,
            [chat_id],
        )
        return cur.fetchone() is not None


def touch_chat_session(chat_id: str) -> None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )


def append_message(
    chat_id: str,
    role: ChatRole | str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMemoryMessage:
    normalized_role = ChatRole(role)
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select coalesce(max(seq), 0) + 1 as next_seq from chat_messages where chat_id = %s",
            [chat_id],
        )
        row = cur.fetchone() or {"next_seq": 1}
        next_seq = int(row["next_seq"])
        cur.execute(
            """
            insert into chat_messages (chat_id, seq, role, content, metadata)
            values (%s, %s, %s, %s, %s)
            returning seq, role, content, created_at
            """,
            [chat_id, next_seq, normalized_role.value, content, Json(metadata or {})],
        )
        inserted = cur.fetchone()
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )

    return _row_to_message(inserted)


def append_user_message_idempotent(
    chat_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    client_message_id: str | None = None,
) -> tuple[ChatMemoryMessage, bool]:
    normalized_metadata = dict(metadata or {})
    if client_message_id:
        normalized_metadata["client_message_id"] = client_message_id
        existing = get_user_message_by_client_message_id(chat_id=chat_id, client_message_id=client_message_id)
        if existing:
            return _stored_to_message(existing), False

    try:
        return (
            append_message(
                chat_id=chat_id,
                role=ChatRole.USER,
                content=content,
                metadata=normalized_metadata,
            ),
            True,
        )
    except psycopg.Error as exc:
        if client_message_id and _is_duplicate_client_message_constraint(exc):
            existing = get_user_message_by_client_message_id(
                chat_id=chat_id,
                client_message_id=client_message_id,
            )
            if existing:
                return _stored_to_message(existing), False
        raise


def get_recent_messages(chat_id: str, limit: int = 12) -> list[ChatMemoryMessage]:
    if limit <= 0:
        return []

    query = """
        select seq, role, content, created_at
        from (
            select seq, role, content, created_at
            from chat_messages
            where chat_id = %s
            order by seq desc
            limit %s
        ) recent
        order by seq asc
    """
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(query, [chat_id, limit])
        return [_row_to_message(row) for row in cur.fetchall()]


def list_messages_after_seq(chat_id: str, after_seq: int = 0) -> list[ChatMemoryMessage]:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select seq, role, content, created_at
            from chat_messages
            where chat_id = %s and seq > %s
            order by seq asc
            """,
            [chat_id, after_seq],
        )
        return [_row_to_message(row) for row in cur.fetchall()]


def list_all_messages(chat_id: str) -> list[ChatMemoryMessage]:
    return list_messages_after_seq(chat_id=chat_id, after_seq=0)


def list_all_stored_messages(chat_id: str) -> list[StoredChatMessage]:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select seq, role, content, created_at, metadata
            from chat_messages
            where chat_id = %s
            order by seq asc
            """,
            [chat_id],
        )
        return [_row_to_stored_message(row) for row in cur.fetchall()]


def list_chat_session_summaries(limit: int = 50) -> list[ChatSessionSummaryRecord]:
    if limit <= 0:
        return []

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                s.chat_id,
                s.created_at,
                s.updated_at,
                s.model_name,
                s.status,
                coalesce(message_stats.message_count, 0)::int as message_count,
                coalesce(run_stats.run_count, 0)::int as run_count,
                latest_run.id as latest_run_id,
                latest_run.status as latest_run_status,
                latest_run.intent as latest_run_intent,
                latest_run.created_at as latest_run_created_at,
                latest_run.assistant_message_seq as latest_server_message_id,
                last_user.content as last_user_message,
                last_user.created_at as last_user_message_at,
                last_assistant.content as last_assistant_message,
                last_assistant.created_at as last_assistant_message_at
            from chat_sessions s
            left join lateral (
                select count(*) as message_count
                from chat_messages m
                where m.chat_id = s.chat_id
            ) message_stats on true
            left join lateral (
                select count(*) as run_count
                from agent_run_traces r
                where r.chat_id = s.chat_id
            ) run_stats on true
            left join lateral (
                select id, status, intent, created_at, assistant_message_seq
                from agent_run_traces r
                where r.chat_id = s.chat_id
                order by created_at desc, id desc
                limit 1
            ) latest_run on true
            left join lateral (
                select content, created_at
                from chat_messages m
                where m.chat_id = s.chat_id
                  and m.role = 'user'
                order by seq desc
                limit 1
            ) last_user on true
            left join lateral (
                select content, created_at
                from chat_messages m
                where m.chat_id = s.chat_id
                  and m.role = 'assistant'
                order by seq desc
                limit 1
            ) last_assistant on true
            order by s.updated_at desc, s.created_at desc
            limit %s
            """,
            [limit],
        )
        rows = cur.fetchall()
    return [_row_to_chat_session_summary(row) for row in rows]


def list_selected_action_task_ids(chat_id: str) -> set[str]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select task_id
            from chat_selected_action_tasks
            where chat_id = %s
            """,
            [chat_id],
        )
        return {str(row["task_id"]) for row in cur.fetchall()}


def replace_agent_action_tasks(
    *,
    chat_id: str,
    assistant_message_seq: int,
    tasks: list[dict[str, Any]],
) -> list[StoredAgentActionTask]:
    ensure_memory_schema()
    rows: list[dict[str, Any]] = []
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            delete from agent_action_tasks
            where chat_id = %s and assistant_message_seq = %s
            """,
            [chat_id, assistant_message_seq],
        )
        for task in tasks:
            cur.execute(
                """
                insert into agent_action_tasks (
                    chat_id,
                    task_id,
                    action_id,
                    assistant_message_seq,
                    source,
                    factor,
                    observation,
                    impact,
                    cause,
                    action,
                    title,
                    action_type,
                    priority,
                    targets,
                    success_criteria,
                    agent_prompt,
                    action_kind,
                    business_goal,
                    decision_type,
                    required_metrics,
                    forecast_spec,
                    execution_plan,
                    evidence,
                    confidence,
                    driver_class,
                    intent,
                    raw_action
                )
                values (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                on conflict (chat_id, task_id) do update
                set action_id = excluded.action_id,
                    assistant_message_seq = excluded.assistant_message_seq,
                    source = excluded.source,
                    factor = excluded.factor,
                    observation = excluded.observation,
                    impact = excluded.impact,
                    cause = excluded.cause,
                    action = excluded.action,
                    title = excluded.title,
                    action_type = excluded.action_type,
                    priority = excluded.priority,
                    targets = excluded.targets,
                    success_criteria = excluded.success_criteria,
                    agent_prompt = excluded.agent_prompt,
                    action_kind = excluded.action_kind,
                    business_goal = excluded.business_goal,
                    decision_type = excluded.decision_type,
                    required_metrics = excluded.required_metrics,
                    forecast_spec = excluded.forecast_spec,
                    execution_plan = excluded.execution_plan,
                    evidence = excluded.evidence,
                    confidence = excluded.confidence,
                    driver_class = excluded.driver_class,
                    intent = excluded.intent,
                    raw_action = excluded.raw_action
                returning
                    chat_id,
                    task_id,
                    action_id,
                    assistant_message_seq,
                    source,
                    factor,
                    observation,
                    impact,
                    cause,
                    action,
                    title,
                    action_type,
                    priority,
                    targets,
                    success_criteria,
                    agent_prompt,
                    action_kind,
                    business_goal,
                    decision_type,
                    required_metrics,
                    forecast_spec,
                    execution_plan,
                    evidence,
                    confidence,
                    driver_class,
                    intent,
                    raw_action,
                    created_at
                """,
                [
                    chat_id,
                    task["task_id"],
                    task.get("action_id") or task["task_id"],
                    assistant_message_seq,
                    task.get("source") or "agent_finding",
                    task["factor"],
                    task.get("observation"),
                    task.get("impact"),
                    task.get("cause"),
                    task["action"],
                    task.get("title"),
                    task.get("action_type"),
                    task.get("priority"),
                    Json(task.get("targets") or []),
                    task.get("success_criteria"),
                    task.get("agent_prompt"),
                    task.get("action_kind"),
                    task.get("business_goal"),
                    task.get("decision_type"),
                    Json(task.get("required_metrics") or []),
                    Json(task.get("forecast_spec") or {}),
                    task.get("execution_plan"),
                    Json(task.get("evidence") or []),
                    task.get("confidence"),
                    task.get("driver_class"),
                    task.get("intent"),
                    Json(task.get("raw_action") or {}),
                ],
            )
            rows.append(cur.fetchone())
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )
    return [_row_to_stored_agent_action_task(row) for row in rows]


def list_stored_agent_action_tasks(
    chat_id: str,
    assistant_message_seq: int | None = None,
) -> list[StoredAgentActionTask]:
    params: list[Any] = [chat_id]
    message_filter = ""
    if assistant_message_seq is not None:
        message_filter = "and assistant_message_seq = %s"
        params.append(assistant_message_seq)

    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select
                chat_id,
                task_id,
                action_id,
                assistant_message_seq,
                source,
                factor,
                observation,
                impact,
                cause,
                action,
                title,
                action_type,
                priority,
                targets,
                success_criteria,
                agent_prompt,
                action_kind,
                business_goal,
                decision_type,
                required_metrics,
                forecast_spec,
                execution_plan,
                evidence,
                confidence,
                driver_class,
                intent,
                raw_action,
                created_at
            from agent_action_tasks
            where chat_id = %s
            {message_filter}
            order by assistant_message_seq asc, created_at asc, task_id asc
            """,
            params,
        )
        rows = cur.fetchall()
    return [_row_to_stored_agent_action_task(row) for row in rows]


def create_agent_action_execution(
    *,
    source_chat_id: str,
    source_task_id: str,
    task_id: str,
    object_id: str,
    object_label: str | None = None,
    object_type: str | None = None,
    object_external_id: str | None = None,
    object_payload: dict[str, Any] | None = None,
    execution_chat_id: str,
    client_message_id: str,
    prompt: str,
    task_snapshot: dict[str, Any],
    execution_mode: str = "terminal_action",
    result_json: dict[str, Any] | None = None,
    decision_status: str | None = None,
    review_summary: str | None = None,
    status: str = "queued",
    run_id: int | None = None,
) -> AgentActionExecutionRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_action_executions (
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            """,
            [
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                Json(object_payload or {}),
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                Json(task_snapshot),
                Json(result_json or {}),
                decision_status,
                review_summary,
            ],
        )
        row = cur.fetchone()
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [source_chat_id],
        )
    return _row_to_agent_action_execution(row)


def get_agent_action_execution(execution_id: int) -> AgentActionExecutionRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            from agent_action_executions
            where id = %s
            """,
            [execution_id],
        )
        row = cur.fetchone()
    return _row_to_agent_action_execution(row) if row else None


def get_agent_action_execution_for_task(
    *,
    source_chat_id: str,
    task_id: str,
) -> AgentActionExecutionRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            from agent_action_executions
            where source_chat_id = %s and task_id = %s
            order by created_at desc
            limit 1
            """,
            [source_chat_id, task_id],
        )
        row = cur.fetchone()
    return _row_to_agent_action_execution(row) if row else None


def get_agent_action_execution_for_object(
    *,
    source_chat_id: str,
    source_task_id: str,
    object_id: str,
) -> AgentActionExecutionRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            from agent_action_executions
            where source_chat_id = %s and source_task_id = %s and object_id = %s
            order by created_at desc
            limit 1
            """,
            [source_chat_id, source_task_id, object_id],
        )
        row = cur.fetchone()
    return _row_to_agent_action_execution(row) if row else None


def list_agent_action_executions(source_chat_id: str) -> list[AgentActionExecutionRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            from agent_action_executions
            where source_chat_id = %s
            order by created_at desc, id desc
            """,
            [source_chat_id],
        )
        rows = cur.fetchall()
    return [_row_to_agent_action_execution(row) for row in rows]


def update_agent_action_execution(
    *,
    execution_id: int,
    status: str | None = None,
    run_id: int | None = None,
    answer: str | None = None,
    server_message_id: int | None = None,
    error: str | None = None,
    result_json: dict[str, Any] | None = None,
    decision_status: str | None = None,
    review_summary: str | None = None,
) -> AgentActionExecutionRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_action_executions
            set status = coalesce(%s, status),
                run_id = coalesce(%s, run_id),
                answer = coalesce(%s, answer),
                server_message_id = coalesce(%s, server_message_id),
                error = coalesce(%s, error),
                result_json = coalesce(%s::jsonb, result_json),
                decision_status = coalesce(%s, decision_status),
                review_summary = coalesce(%s, review_summary),
                updated_at = now(),
                completed_at = case
                    when coalesce(%s, status) in ('completed', 'failed', 'cancelled', 'done')
                    then coalesce(completed_at, now())
                    else completed_at
                end
            where id = %s
            returning
                id,
                source_chat_id,
                source_task_id,
                task_id,
                object_id,
                object_label,
                object_type,
                object_external_id,
                object_payload,
                execution_chat_id,
                run_id,
                client_message_id,
                status,
                execution_mode,
                prompt,
                task_snapshot,
                result_json,
                decision_status,
                review_summary,
                answer,
                server_message_id,
                error,
                created_at,
                updated_at,
                completed_at
            """,
            [
                status,
                run_id,
                answer,
                server_message_id,
                error,
                Json(result_json) if result_json is not None else None,
                decision_status,
                review_summary,
                status,
                execution_id,
            ],
        )
        row = cur.fetchone()
        if not row:
            raise LookupError(execution_id)
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [row["source_chat_id"]],
        )
    return _row_to_agent_action_execution(row)


def create_agent_run_trace(
    *,
    chat_id: str,
    intent: str,
    client_message_id: str | None = None,
    primary_agent: str | None = None,
    events: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    status: str = "running",
) -> AgentRunTraceRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_run_traces (
                chat_id,
                client_message_id,
                intent,
                primary_agent,
                status,
                events,
                summary
            )
            values (%s, %s, %s, %s, %s, %s, %s)
            returning
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            """,
            [
                chat_id,
                client_message_id,
                intent,
                primary_agent,
                status,
                Json(events or []),
                Json(summary or {}),
            ],
        )
        row = cur.fetchone()
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )
    return _row_to_agent_run_trace(row)


def complete_agent_run_trace(
    *,
    trace_id: int,
    assistant_message_seq: int | None,
    status: str,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    primary_agent: str | None = None,
    question_id: int | None = None,
    round_id: int | None = None,
) -> AgentRunTraceRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        if question_id is not None:
            cur.execute(
                """
                update agent_run_questions
                set status = 'completed',
                    last_error = null,
                    completed_at = now(),
                    updated_at = now()
                where id = %s
                """,
                [question_id],
            )
            if cur.rowcount != 1:
                raise LookupError(question_id)
        if round_id is not None:
            cur.execute(
                """
                update agent_run_rounds
                set status = 'completed',
                    completed_at = now(),
                    updated_at = now()
                where id = %s
                """,
                [round_id],
            )
            if cur.rowcount != 1:
                raise LookupError(round_id)
        cur.execute(
            """
            update agent_run_traces
            set assistant_message_seq = %s,
                status = %s,
                primary_agent = coalesce(%s, primary_agent),
                completed_at = now(),
                events = %s,
                summary = %s
            where id = %s
            returning
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            """,
            [
                assistant_message_seq,
                status,
                primary_agent,
                Json(events),
                Json(summary),
                trace_id,
            ],
        )
        row = cur.fetchone()
        if not row:
            raise LookupError(trace_id)
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [row["chat_id"]],
        )
    return _row_to_agent_run_trace(row)


def update_agent_run_trace_status(
    *,
    trace_id: int,
    status: str,
    assistant_message_seq: int | None = None,
    events: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
    primary_agent: str | None = None,
) -> AgentRunTraceRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_traces
            set assistant_message_seq = coalesce(%s, assistant_message_seq),
                status = %s,
                primary_agent = coalesce(%s, primary_agent),
                completed_at = case when %s in ('completed', 'failed', 'cancelled') then now() else completed_at end,
                events = coalesce(%s::jsonb, events),
                summary = coalesce(%s::jsonb, summary)
            where id = %s
            returning
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            """,
            [
                assistant_message_seq,
                status,
                primary_agent,
                status,
                Json(events) if events is not None else None,
                Json(summary) if summary is not None else None,
                trace_id,
            ],
        )
        row = cur.fetchone()
        if not row:
            raise LookupError(trace_id)
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [row["chat_id"]],
        )
    return _row_to_agent_run_trace(row)


def get_agent_run_trace(chat_id: str, assistant_message_seq: int) -> AgentRunTraceRecord | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            from agent_run_traces
            where chat_id = %s
              and assistant_message_seq = %s
            order by created_at desc
            limit 1
            """,
            [chat_id, assistant_message_seq],
        )
        row = cur.fetchone()
    return _row_to_agent_run_trace(row) if row else None


def list_agent_run_traces(chat_id: str, limit: int = 50) -> list[AgentRunTraceRecord]:
    if limit <= 0:
        return []
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            from agent_run_traces
            where chat_id = %s
            order by created_at desc
            limit %s
            """,
            [chat_id, limit],
        )
        rows = cur.fetchall()
    return [_row_to_agent_run_trace(row) for row in rows]


def get_agent_run_trace_by_id(run_id: int) -> AgentRunTraceRecord | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            from agent_run_traces
            where id = %s
            """,
            [run_id],
        )
        row = cur.fetchone()
    return _row_to_agent_run_trace(row) if row else None


def get_agent_run_trace_by_client_message_id(
    *,
    chat_id: str,
    client_message_id: str,
) -> AgentRunTraceRecord | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                chat_id,
                assistant_message_seq,
                client_message_id,
                intent,
                primary_agent,
                status,
                started_at,
                completed_at,
                events,
                summary,
                created_at
            from agent_run_traces
            where chat_id = %s
              and client_message_id = %s
            order by created_at desc
            limit 1
            """,
            [chat_id, client_message_id],
        )
        row = cur.fetchone()
    return _row_to_agent_run_trace(row) if row else None


def create_agent_run_round(
    *,
    run_id: int,
    round_number: int,
    round_type: str,
    status: str = "running",
) -> AgentRunRoundRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_run_rounds (
                run_id,
                round_number,
                round_type,
                status
            )
            values (%s, %s, %s, %s)
            returning
                id,
                run_id,
                round_number,
                round_type,
                status,
                created_at,
                updated_at,
                completed_at
            """,
            [run_id, round_number, round_type, status],
        )
        row = cur.fetchone()
    return _row_to_agent_run_round(row)


def update_agent_run_round_status(
    *,
    round_id: int,
    status: str,
) -> AgentRunRoundRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_rounds
            set status = %s,
                updated_at = now(),
                completed_at = case when %s in ('completed', 'failed', 'skipped') then now() else completed_at end
            where id = %s
            returning
                id,
                run_id,
                round_number,
                round_type,
                status,
                created_at,
                updated_at,
                completed_at
            """,
            [status, status, round_id],
        )
        row = cur.fetchone()
    if not row:
        raise LookupError(round_id)
    return _row_to_agent_run_round(row)


def create_agent_run_question(
    *,
    run_id: int,
    round_id: int,
    source_agent: str | None,
    target_agent: str,
    question: str,
    reason: str | None = None,
    verification_goal: dict[str, Any] | None = None,
    priority: int | None = None,
    status: str = "pending",
    parent_question_id: int | None = None,
) -> AgentRunQuestionRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_run_questions (
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                parent_question_id
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            """,
            [
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                Json(verification_goal or {}),
                priority,
                status,
                parent_question_id,
            ],
        )
        row = cur.fetchone()
    return _row_to_agent_run_question(row)


def update_agent_run_question_status(
    *,
    question_id: int,
    status: str,
) -> AgentRunQuestionRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_questions
            set status = %s,
                updated_at = now(),
                started_at = case when %s = 'running' and started_at is null then now() else started_at end,
                completed_at = case when %s in ('completed', 'failed', 'skipped') then now() else completed_at end
            where id = %s
            returning
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            """,
            [status, status, status, question_id],
        )
        row = cur.fetchone()
    if not row:
        raise LookupError(question_id)
    return _row_to_agent_run_question(row)


def get_agent_run_question(question_id: int) -> AgentRunQuestionRecord | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            from agent_run_questions
            where id = %s
            """,
            [question_id],
        )
        row = cur.fetchone()
    return _row_to_agent_run_question(row) if row else None


def mark_agent_run_question_queued(
    *,
    question_id: int,
    job_id: str,
    timeout_seconds: int,
) -> AgentRunQuestionRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_questions
            set status = 'queued',
                job_id = %s,
                timeout_seconds = %s,
                queued_at = now(),
                updated_at = now(),
                last_error = null
            where id = %s
              and status in ('pending', 'queued')
            returning
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            """,
            [job_id, timeout_seconds, question_id],
        )
        row = cur.fetchone()
    if not row:
        # The RQ worker can claim and even finish a very small job before the
        # enqueueing process persists the job id.  Never move that newer state
        # back to ``queued``; return the current durable record instead.
        current = get_agent_run_question(question_id)
        if current is None:
            raise LookupError(question_id)
        return current
    return _row_to_agent_run_question(row)


def claim_agent_run_question_running(question_id: int) -> AgentRunQuestionRecord | None:
    """Atomically claim a pending delivery for execution.

    RQ delivery is at-least-once.  Restricting the state transition in the
    ``UPDATE`` makes Postgres the execution fence: only one worker can move a
    question from ``pending``/``queued`` to ``running`` and increment its
    attempt counter.  Duplicate or late deliveries receive ``None`` and must
    not call the LLM.
    """

    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_questions
            set status = 'running',
                attempt_count = attempt_count + 1,
                started_at = coalesce(started_at, now()),
                updated_at = now()
            where id = %s
              and status in ('pending', 'queued')
            returning
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            """,
            [question_id],
        )
        row = cur.fetchone()
    return _row_to_agent_run_question(row) if row else None


def mark_agent_run_question_running(question_id: int) -> AgentRunQuestionRecord:
    """Claim a question without regressing an already-running/terminal row.

    This compatibility wrapper retains the historical return type.  New job
    executors should use :func:`claim_agent_run_question_running` so they can
    distinguish a successful claim from duplicate delivery.
    """

    claimed = claim_agent_run_question_running(question_id)
    if claimed is not None:
        return claimed
    current = get_agent_run_question(question_id)
    if current is None:
        raise LookupError(question_id)
    return current


def mark_agent_run_question_failed(
    *,
    question_id: int,
    last_error: str,
) -> AgentRunQuestionRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_questions
            set status = 'failed',
                last_error = %s,
                completed_at = now(),
                updated_at = now()
            where id = %s
            returning
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            """,
            [last_error, question_id],
        )
        row = cur.fetchone()
    if not row:
        raise LookupError(question_id)
    return _row_to_agent_run_question(row)


def claim_timed_out_agent_run_questions(
    *,
    now: datetime,
    default_timeout_seconds: int,
    grace_seconds: int = 60,
    limit: int = 100,
) -> list[AgentRunQuestionRecord]:
    """Atomically fail abandoned running questions so their run can advance.

    RQ cannot update Postgres after a work horse is killed. ``FOR UPDATE SKIP
    LOCKED`` lets several reconcilers run safely while the status predicate
    prevents a late reconciler from overwriting a concurrently completed row.
    """

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            with timed_out as (
                select questions.id
                from agent_run_questions questions
                join agent_run_traces traces on traces.id = questions.run_id
                where traces.status not in ('completed', 'failed', 'cancelled')
                  and (
                    (
                      questions.status = 'running'
                      and coalesce(
                            questions.started_at,
                            questions.updated_at,
                            questions.created_at
                          )
                          + make_interval(
                                secs => greatest(
                                    coalesce(questions.timeout_seconds, %s),
                                    1
                                ) + %s
                            ) <= %s
                    )
                    or (
                      questions.status = 'failed'
                      and questions.last_error = 'agent_question_timeout_reconciled'
                    )
                  )
                order by coalesce(
                    questions.started_at,
                    questions.updated_at,
                    questions.created_at
                ), questions.id
                for update skip locked
                limit %s
            )
            update agent_run_questions questions
            set status = 'failed',
                last_error = 'agent_question_timeout_reconciled',
                completed_at = %s,
                updated_at = %s
            from timed_out
            where questions.id = timed_out.id
              and (
                questions.status = 'running'
                or (
                  questions.status = 'failed'
                  and questions.last_error = 'agent_question_timeout_reconciled'
                )
              )
            returning questions.*
            """,
            [
                max(1, int(default_timeout_seconds)),
                max(0, int(grace_seconds)),
                now,
                max(1, min(int(limit), 1000)),
                now,
                now,
            ],
        )
        rows = cur.fetchall()
    return [_row_to_agent_run_question(row) for row in rows]


def claim_stale_agent_questions_for_delivery(
    *,
    now: datetime,
    default_timeout_seconds: int,
    pending_lease_seconds: int = 60,
    grace_seconds: int = 60,
    limit: int = 100,
) -> list[AgentRunQuestionRecord]:
    """Lease queued/pending questions whose RQ delivery may have disappeared."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            with stale as (
                select questions.id
                from agent_run_questions questions
                join agent_run_traces traces on traces.id = questions.run_id
                where traces.status not in ('completed', 'failed', 'cancelled')
                  and (
                    (
                      questions.status = 'queued'
                      and coalesce(
                            questions.queued_at,
                            questions.updated_at,
                            questions.created_at
                          )
                          + make_interval(
                                secs => greatest(
                                    coalesce(questions.timeout_seconds, %s),
                                    1
                                ) + %s
                            ) <= %s
                    )
                    or (
                      questions.status = 'pending'
                      and coalesce(questions.updated_at, questions.created_at)
                          + make_interval(secs => %s) <= %s
                    )
                  )
                order by coalesce(
                    questions.queued_at,
                    questions.updated_at,
                    questions.created_at
                ), questions.id
                for update skip locked
                limit %s
            )
            update agent_run_questions questions
            set status = 'pending',
                updated_at = %s
            from stale
            where questions.id = stale.id
              and questions.status in ('pending', 'queued')
            returning questions.*
            """,
            [
                max(1, int(default_timeout_seconds)),
                max(0, int(grace_seconds)),
                now,
                max(1, int(pending_lease_seconds)),
                now,
                max(1, min(int(limit), 1000)),
                now,
            ],
        )
        rows = cur.fetchall()
    return [_row_to_agent_run_question(row) for row in rows]


def save_agent_run_response(
    *,
    question_id: int,
    agent_name: str,
    status: str,
    summary: str | None,
    raw_output: dict[str, Any],
    has_business_evidence: bool,
    runtime_failure: dict[str, Any] | None = None,
) -> AgentRunResponseRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_run_responses (
                question_id,
                agent_name,
                status,
                summary,
                raw_output,
                has_business_evidence,
                runtime_failure,
                completed_at
            )
            values (%s, %s, %s, %s, %s, %s, %s, now())
            returning
                id,
                question_id,
                agent_name,
                status,
                summary,
                raw_output,
                has_business_evidence,
                runtime_failure,
                created_at,
                completed_at
            """,
            [
                question_id,
                agent_name,
                status,
                summary,
                Json(raw_output or {}),
                has_business_evidence,
                Json(runtime_failure) if runtime_failure else None,
            ],
        )
        row = cur.fetchone()
    return _row_to_agent_run_response(row)


def save_agent_run_gaps(
    *,
    run_id: int,
    response_id: int | None,
    gaps: list[dict[str, Any]],
) -> list[AgentRunGapRecord]:
    ensure_memory_schema()
    rows: list[dict[str, Any]] = []
    with get_memory_connection() as conn, conn.cursor() as cur:
        for gap in gaps:
            cur.execute(
                """
                insert into agent_run_gaps (
                    run_id,
                    response_id,
                    sku,
                    source_agent,
                    target_agent,
                    metric,
                    reason,
                    impact,
                    blocking,
                    status,
                    evidence_status
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning
                    id,
                    run_id,
                    response_id,
                    sku,
                    source_agent,
                    target_agent,
                    metric,
                    reason,
                    impact,
                    blocking,
                    status,
                    evidence_status,
                    created_at,
                    updated_at
                """,
                [
                    run_id,
                    response_id,
                    gap.get("sku"),
                    gap.get("source_agent"),
                    gap.get("target_agent"),
                    gap.get("metric") or "unknown metric",
                    gap.get("reason") or "Нет данных для подтверждения.",
                    gap.get("impact"),
                    bool(gap.get("blocking", True)),
                    gap.get("status") or "open",
                    gap.get("evidence_status") or "unconfirmed",
                ],
            )
            rows.append(cur.fetchone())
    return [_row_to_agent_run_gap(row) for row in rows]


def update_agent_run_gap_status(
    *,
    gap_id: int,
    status: str,
) -> AgentRunGapRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update agent_run_gaps
            set status = %s,
                updated_at = now()
            where id = %s
            returning
                id,
                run_id,
                response_id,
                sku,
                source_agent,
                target_agent,
                metric,
                reason,
                impact,
                blocking,
                status,
                evidence_status,
                created_at,
                updated_at
            """,
            [status, gap_id],
        )
        row = cur.fetchone()
    if not row:
        raise LookupError(gap_id)
    return _row_to_agent_run_gap(row)


def list_agent_run_gaps(
    *,
    run_id: int,
    status: str | None = None,
    blocking: bool | None = None,
    sku: str | None = None,
    evidence_statuses: Iterable[str] | None = None,
) -> list[AgentRunGapRecord]:
    clauses = ["run_id = %s"]
    params: list[Any] = [run_id]
    if status is not None:
        clauses.append("status = %s")
        params.append(status)
    if blocking is not None:
        clauses.append("blocking = %s")
        params.append(blocking)
    if sku is not None:
        clauses.append("sku = %s")
        params.append(sku)
    evidence_status_values = list(evidence_statuses or [])
    if evidence_status_values:
        clauses.append("evidence_status = any(%s)")
        params.append(evidence_status_values)

    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select
                id,
                run_id,
                response_id,
                sku,
                source_agent,
                target_agent,
                metric,
                reason,
                impact,
                blocking,
                status,
                evidence_status,
                created_at,
                updated_at
            from agent_run_gaps
            where {" and ".join(clauses)}
            order by blocking desc, created_at asc, id asc
            """,
            params,
        )
        rows = cur.fetchall()
    return [_row_to_agent_run_gap(row) for row in rows]


def list_agent_run_rounds(run_id: int) -> list[dict[str, Any]]:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                id,
                run_id,
                round_number,
                round_type,
                status,
                created_at,
                updated_at,
                completed_at
            from agent_run_rounds
            where run_id = %s
            order by round_number asc, id asc
            """,
            [run_id],
        )
        round_rows = cur.fetchall()

        cur.execute(
            """
            select
                id,
                run_id,
                round_id,
                source_agent,
                target_agent,
                question,
                reason,
                verification_goal,
                priority,
                status,
                job_id,
                queued_at,
                attempt_count,
                last_error,
                timeout_seconds,
                parent_question_id,
                created_at,
                updated_at,
                started_at,
                completed_at
            from agent_run_questions
            where run_id = %s
            order by round_id asc, priority desc nulls last, id asc
            """,
            [run_id],
        )
        question_rows = cur.fetchall()

        question_ids = [row["id"] for row in question_rows]
        response_rows: list[dict[str, Any]] = []
        if question_ids:
            cur.execute(
                """
                select
                    id,
                    question_id,
                    agent_name,
                    status,
                    summary,
                    raw_output,
                    has_business_evidence,
                    runtime_failure,
                    created_at,
                    completed_at
                from agent_run_responses
                where question_id = any(%s)
                order by id asc
                """,
                [question_ids],
            )
            response_rows = cur.fetchall()

        cur.execute(
            """
            select
                id,
                run_id,
                response_id,
                sku,
                source_agent,
                target_agent,
                metric,
                reason,
                impact,
                blocking,
                status,
                evidence_status,
                created_at,
                updated_at
            from agent_run_gaps
            where run_id = %s
            order by id asc
            """,
            [run_id],
        )
        gap_rows = cur.fetchall()

    responses_by_question: dict[int, list[AgentRunResponseRecord]] = {}
    for row in response_rows:
        response = _row_to_agent_run_response(row)
        responses_by_question.setdefault(response.question_id, []).append(response)

    gaps_by_response: dict[int | None, list[AgentRunGapRecord]] = {}
    for row in gap_rows:
        gap = _row_to_agent_run_gap(row)
        gaps_by_response.setdefault(gap.response_id, []).append(gap)

    questions_by_round: dict[int, list[dict[str, Any]]] = {}
    for row in question_rows:
        question = _row_to_agent_run_question(row)
        question_responses = responses_by_question.get(question.id, [])
        questions_by_round.setdefault(question.round_id, []).append(
            {
                "question": question,
                "responses": question_responses,
                "gaps": [
                    gap
                    for response in question_responses
                    for gap in gaps_by_response.get(response.id, [])
                ],
            }
        )

    return [
        {
            "round": _row_to_agent_run_round(row),
            "questions": questions_by_round.get(row["id"], []),
        }
        for row in round_rows
    ]


def list_agent_run_responses(run_id: int) -> list[dict[str, Any]]:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
                q.round_id,
                r.round_number,
                r.round_type,
                q.target_agent,
                q.question,
                q.reason,
                q.verification_goal,
                resp.id,
                resp.question_id,
                resp.agent_name,
                resp.status,
                resp.summary,
                resp.raw_output,
                resp.has_business_evidence,
                resp.runtime_failure,
                resp.created_at,
                resp.completed_at
            from agent_run_responses resp
            join agent_run_questions q on q.id = resp.question_id
            join agent_run_rounds r on r.id = q.round_id
            where q.run_id = %s
            order by r.round_number asc, q.priority desc nulls last, q.id asc, resp.id asc
            """,
            [run_id],
        )
        rows = cur.fetchall()
    return [
        {
            "round_id": row["round_id"],
            "round_number": row["round_number"],
            "round_type": row["round_type"],
            "target_agent": row["target_agent"],
            "question": row["question"],
            "reason": row["reason"],
            "verification_goal": row["verification_goal"] or {},
            "response": _row_to_agent_run_response(row),
        }
        for row in rows
    ]


def select_action_task(chat_id: str, task_id: str) -> None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into chat_selected_action_tasks (chat_id, task_id)
            values (%s, %s)
            on conflict (chat_id, task_id) do nothing
            """,
            [chat_id, task_id],
        )
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )


def unselect_action_task(chat_id: str, task_id: str) -> None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            delete from chat_selected_action_tasks
            where chat_id = %s and task_id = %s
            """,
            [chat_id, task_id],
        )
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )


def get_user_message_by_client_message_id(
    chat_id: str,
    client_message_id: str,
) -> StoredChatMessage | None:
    return _load_message_by_metadata_key(
        chat_id=chat_id,
        metadata_key="client_message_id",
        metadata_value=client_message_id,
        role=ChatRole.USER,
    )


def get_assistant_reply_for_client_message_id(
    chat_id: str,
    client_message_id: str,
) -> StoredChatMessage | None:
    return _load_message_by_metadata_key(
        chat_id=chat_id,
        metadata_key="reply_to_client_message_id",
        metadata_value=client_message_id,
        role=ChatRole.ASSISTANT,
    )


def load_summary(chat_id: str) -> ChatSummaryRecord | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select summary_text, summarized_through_seq, updated_at
            from chat_summaries
            where chat_id = %s
            """,
            [chat_id],
        )
        row = cur.fetchone()

    if not row:
        return None

    return ChatSummaryRecord(
        summary_text=row["summary_text"],
        summarized_through_seq=int(row["summarized_through_seq"]),
        updated_at=row["updated_at"],
    )


def save_summary(
    chat_id: str,
    summary_text: str,
    summarized_through_seq: int,
) -> ChatSummaryRecord:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into chat_summaries (chat_id, summary_text, summarized_through_seq, updated_at)
            values (%s, %s, %s, now())
            on conflict (chat_id) do update
            set summary_text = excluded.summary_text,
                summarized_through_seq = excluded.summarized_through_seq,
                updated_at = now()
            returning summary_text, summarized_through_seq, updated_at
            """,
            [chat_id, summary_text, summarized_through_seq],
        )
        row = cur.fetchone()
        cur.execute(
            "update chat_sessions set updated_at = now() where chat_id = %s",
            [chat_id],
        )

    return ChatSummaryRecord(
        summary_text=row["summary_text"],
        summarized_through_seq=int(row["summarized_through_seq"]),
        updated_at=row["updated_at"],
    )


def get_context_window(chat_id: str, limit: int = 12) -> ChatContextWindow:
    summary = load_summary(chat_id)
    return ChatContextWindow(
        chat_id=chat_id,
        summary_text=summary.summary_text if summary else None,
        recent_messages=get_recent_messages(chat_id, limit=limit),
    )


def refresh_chat_summary(chat_id: str) -> ChatSummaryRecord | None:
    settings = get_settings()
    summary_record = load_summary(chat_id)
    after_seq = summary_record.summarized_through_seq if summary_record else 0
    pending_messages = list_messages_after_seq(chat_id, after_seq=after_seq)
    if not pending_messages:
        return summary_record

    summary_text = generate_summary_text(
        existing_summary=summary_record.summary_text if summary_record else None,
        pending_messages=pending_messages,
        api_key=settings.openai_api_key,
        model_name=settings.openai_memory_model,
    )
    return save_summary(
        chat_id=chat_id,
        summary_text=summary_text,
        summarized_through_seq=pending_messages[-1].seq,
    )


def build_memory_context_block(window: ChatContextWindow) -> str:
    summary = window.summary_text.strip() if window.summary_text else "No prior summary."
    if window.recent_messages:
        message_lines = [
            f"- #{message.seq} {message.role.value}: {message.content}"
            for message in window.recent_messages
        ]
        recent_messages = "\n".join(message_lines)
    else:
        recent_messages = "- none"
    return (
        "Chat memory:\n"
        f"- chat_id: {window.chat_id}\n"
        f"- summary: {summary}\n"
        "- recent_messages:\n"
        f"{recent_messages}"
    )


def generate_summary_text(
    existing_summary: str | None,
    pending_messages: Iterable[ChatMemoryMessage],
    api_key: str | None,
    model_name: str,
) -> str:
    messages = list(pending_messages)
    if not messages:
        return (existing_summary or "").strip()
    if not api_key:
        raise RuntimeError(
            "Не найден OPENAI_API_KEY. Невозможно обновить summary памяти чата без доступа к модели."
        )

    settings = get_settings()
    client = build_direct_openai_client(api_key=api_key, settings=settings)
    prompt = _build_summary_prompt(existing_summary=existing_summary, pending_messages=messages)
    with generation_scope(
        name="memory-summary",
        model=model_name,
        input=prompt,
        metadata={
            "model_provider": "openai",
            "model_name": model_name,
            "message_count": len(messages),
        },
        settings=settings,
    ) as observation:
        try:
            response = client.responses.create(model=model_name, input=prompt)
        except Exception as exc:
            update_observation(observation, error=exc)
            raise
        update_observation(
            observation,
            output=response_payload(response),
            response=response,
        )
    summary_text = (response.output_text or "").strip()
    if not summary_text:
        raise RuntimeError("Модель не вернула summary для памяти чата.")
    return summary_text


def _build_summary_prompt(
    existing_summary: str | None,
    pending_messages: list[ChatMemoryMessage],
) -> str:
    existing_block = existing_summary.strip() if existing_summary else "No summary yet."
    new_messages_block = _format_messages_for_summary(pending_messages)
    return (
        "You update rolling chat memory for a CLI assistant.\n"
        "Return a concise factual summary in Russian.\n"
        "Keep only durable user preferences, goals, prior asks, unresolved follow-ups, and important context.\n"
        "Do not include markdown or bullet nesting.\n\n"
        f"Existing summary:\n{existing_block}\n\n"
        f"New messages:\n{new_messages_block}\n\n"
        "Write the refreshed summary:"
    )


def _format_messages_for_summary(messages: list[ChatMemoryMessage]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message.seq}. {message.role.value}: {message.content}")
    return "\n".join(lines)


def upsert_app_user(
    *,
    username: str,
    password_hash: str,
    role: str,
    is_active: bool = True,
) -> AppUserRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into app_users (username, password_hash, role, is_active)
            values (%s, %s, %s, %s)
            on conflict (lower(username)) do update
            set password_hash = excluded.password_hash,
                role = excluded.role,
                is_active = excluded.is_active,
                updated_at = now()
            returning *
            """,
            [username.strip(), password_hash, role, is_active],
        )
        row = cur.fetchone()
    return _row_to_app_user(row)


def get_app_user_by_username(username: str) -> AppUserRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from app_users where lower(username) = lower(%s) limit 1",
            [username.strip()],
        )
        row = cur.fetchone()
    return _row_to_app_user(row) if row else None


def get_app_user(user_id: int) -> AppUserRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute("select * from app_users where id = %s", [user_id])
        row = cur.fetchone()
    return _row_to_app_user(row) if row else None


def record_auth_audit(
    *,
    username: str,
    event: str,
    success: bool,
    user_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into auth_audit_log (user_id, username, event, success, details)
            values (%s, %s, %s, %s, %s)
            """,
            [user_id, username, event, success, Json(details or {})],
        )


def upsert_manager_decision(
    *,
    decision_id: str,
    objective: str,
    status: str,
    request_coverage: dict[str, Any] | None = None,
    evidence_as_of: datetime | None = None,
    ranked_actions: list[dict[str, Any]] | None = None,
    rejected_candidates: list[dict[str, Any]] | None = None,
    blockers: list[Any] | None = None,
    decision_json: dict[str, Any] | None = None,
    public_json: dict[str, Any] | None = None,
    chat_id: str | None = None,
    run_id: int | None = None,
    source_key: str | None = None,
    decision_date: Any | None = None,
) -> ManagerDecisionRecord:
    """Persist a deterministic decision without changing an existing duplicate request."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        if source_key:
            cur.execute("select * from manager_decisions where source_key = %s", [source_key])
            existing = cur.fetchone()
            if existing:
                return _row_to_manager_decision(existing)
        cur.execute(
            """
            insert into manager_decisions (
                id, source_key, chat_id, run_id, decision_date, objective, status,
                request_coverage, evidence_as_of, ranked_actions, rejected_candidates,
                blockers, decision_json, public_json
            )
            values (
                %s, %s, %s, %s, coalesce(%s, (now() at time zone 'Europe/Moscow')::date),
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            on conflict (id) do update
            set objective = excluded.objective,
                status = excluded.status,
                request_coverage = excluded.request_coverage,
                evidence_as_of = excluded.evidence_as_of,
                ranked_actions = excluded.ranked_actions,
                rejected_candidates = excluded.rejected_candidates,
                blockers = excluded.blockers,
                decision_json = excluded.decision_json,
                public_json = excluded.public_json,
                updated_at = now()
            returning *
            """,
            [
                decision_id,
                source_key,
                chat_id,
                run_id,
                decision_date,
                objective,
                status,
                Json(request_coverage or {}),
                evidence_as_of,
                Json(ranked_actions or []),
                Json(rejected_candidates or []),
                Json(blockers or []),
                Json(decision_json or {}),
                Json(public_json or {}),
            ],
        )
        row = cur.fetchone()
    return _row_to_manager_decision(row)


def get_manager_decision(decision_id: str) -> ManagerDecisionRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute("select * from manager_decisions where id = %s", [decision_id])
        row = cur.fetchone()
    return _row_to_manager_decision(row) if row else None


def get_manager_decision_by_source_key(source_key: str) -> ManagerDecisionRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from manager_decisions where source_key = %s",
            [source_key],
        )
        row = cur.fetchone()
    return _row_to_manager_decision(row) if row else None


def list_manager_decisions_for_date(decision_date: Any) -> list[ManagerDecisionRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from manager_decisions
            where decision_date = %s
            order by created_at desc, id desc
            """,
            [decision_date],
        )
        rows = cur.fetchall()
    return [_row_to_manager_decision(row) for row in rows]


def create_decision_simulation_run(
    *,
    simulation_run_id: str,
    decision_id: str,
    mode: str,
    model_version: str,
    evidence_snapshot: dict[str, Any],
    evidence_hash: str,
    master_seed: int,
    candidate_count: int,
    sample_count: int = 500,
    batch_size: int = 50,
    objective: str = "incremental_contribution_profit",
    chat_id: str | None = None,
    run_id: int | None = None,
    status: str = "queued",
    idempotency_key: str | None = None,
    retry_of_run_id: str | None = None,
    retry_attempt: int = 0,
) -> DecisionSimulationRunRecord:
    """Create a durable simulation run, returning the prior idempotent run if present."""
    ensure_memory_schema()
    total_samples = int(candidate_count) * int(sample_count)
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into decision_simulation_runs (
                id, decision_id, chat_id, run_id, mode, status, objective,
                model_version, evidence_snapshot, evidence_hash, master_seed,
                sample_count, batch_size, candidate_count, total_samples,
                idempotency_key, retry_of_run_id, retry_attempt
            )
            values (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            on conflict do nothing
            returning *
            """,
            [
                simulation_run_id,
                decision_id,
                chat_id,
                run_id,
                mode,
                status,
                objective,
                model_version,
                Json(evidence_snapshot),
                evidence_hash,
                master_seed,
                sample_count,
                batch_size,
                candidate_count,
                total_samples,
                idempotency_key,
                retry_of_run_id,
                retry_attempt,
            ],
        )
        row = cur.fetchone()
        if row is None and idempotency_key:
            if retry_of_run_id:
                cur.execute(
                    """
                    select * from decision_simulation_runs
                    where retry_of_run_id = %s and idempotency_key = %s
                    """,
                    [retry_of_run_id, idempotency_key],
                )
            else:
                cur.execute(
                    """
                    select * from decision_simulation_runs
                    where decision_id = %s
                      and retry_of_run_id is null
                      and idempotency_key = %s
                    """,
                    [decision_id, idempotency_key],
                )
            row = cur.fetchone()
        if row is None and retry_of_run_id:
            # A concurrent retry with a different key may have won the
            # one-child-per-parent constraint after our status check.
            cur.execute(
                """
                select * from decision_simulation_runs
                where retry_of_run_id = %s
                order by created_at, id
                limit 1
                """,
                [retry_of_run_id],
            )
            row = cur.fetchone()
        if row is None:
            cur.execute(
                "select * from decision_simulation_runs where id = %s",
                [simulation_run_id],
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to create or recover decision simulation run")
    return _row_to_decision_simulation_run(row)


def get_decision_simulation_run(
    simulation_run_id: str,
) -> DecisionSimulationRunRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from decision_simulation_runs where id = %s",
            [simulation_run_id],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_run(row) if row else None


def get_latest_decision_simulation_run_for_decision(
    decision_id: str,
) -> DecisionSimulationRunRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_runs
            where decision_id = %s
            order by created_at desc, id desc
            limit 1
            """,
            [decision_id],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_run(row) if row else None


def list_decision_simulation_runs(
    *,
    decision_id: str | None = None,
    statuses: Iterable[str] | None = None,
    limit: int = 100,
) -> list[DecisionSimulationRunRecord]:
    ensure_memory_schema()
    conditions: list[str] = []
    params: list[Any] = []
    if decision_id is not None:
        conditions.append("decision_id = %s")
        params.append(decision_id)
    status_values = list(statuses or [])
    if status_values:
        conditions.append("status = any(%s)")
        params.append(status_values)
    where_sql = f"where {' and '.join(conditions)}" if conditions else ""
    params.append(max(1, min(int(limit), 1000)))
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select * from decision_simulation_runs
            {where_sql}
            order by created_at desc, id desc
            limit %s
            """,
            params,
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_run(row) for row in rows]


def list_stale_decision_simulation_runs(
    *,
    stale_before: datetime,
    statuses: Iterable[str] = (
        "initializing",
        "queued",
        "running",
        "critic_running",
    ),
    limit: int = 100,
) -> list[DecisionSimulationRunRecord]:
    """Return recoverable runs whose latest worker activity predates a cutoff."""
    ensure_memory_schema()
    status_values = list(statuses)
    if not status_values:
        return []
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_runs
            where status = any(%s)
              and coalesce(heartbeat_at, started_at, created_at) < %s
            order by coalesce(heartbeat_at, started_at, created_at), id
            limit %s
            """,
            [status_values, stale_before, max(1, min(int(limit), 1000))],
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_run(row) for row in rows]


def claim_stale_decision_simulation_run(
    simulation_run_id: str,
    *,
    expected_status: str,
    stale_before: datetime,
    claimed_status: str,
    heartbeat_at: datetime,
    error: dict[str, Any] | None = None,
) -> DecisionSimulationRunRecord | None:
    """Atomically claim a stale run without regressing a concurrently finished job."""

    ensure_memory_schema()
    terminal = claimed_status in {"completed", "blocked", "failed", "superseded"}
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_runs
            set status = %s,
                heartbeat_at = %s,
                error = case when %s then %s else error end,
                winner_candidate_id = case when %s then null else winner_candidate_id end,
                completed_at = case
                    when %s then coalesce(completed_at, now())
                    else completed_at
                end,
                updated_at = now()
            where id = %s
              and status = %s
              and coalesce(heartbeat_at, started_at, created_at) < %s
            returning *
            """,
            [
                claimed_status,
                heartbeat_at,
                error is not None,
                Json(error or {}),
                terminal,
                terminal,
                simulation_run_id,
                expected_status,
                stale_before,
            ],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_run(row) if row else None


def claim_decision_simulation_initialization(
    simulation_run_id: str,
) -> DecisionSimulationRunRecord | None:
    """Let exactly one concurrent producer initialize a newly-created run."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_runs
            set heartbeat_at = now(),
                updated_at = now()
            where id = %s
              and status = 'initializing'
              and heartbeat_at is null
            returning *
            """,
            [simulation_run_id],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_run(row) if row else None


def list_unprojected_terminal_decision_simulation_runs(
    *,
    limit: int = 100,
) -> list[DecisionSimulationRunRecord]:
    """Find terminal runs whose manager projection or journal is incomplete."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select runs.*
            from decision_simulation_runs runs
            join manager_decisions decisions on decisions.id = runs.decision_id
            where runs.superseded_by_run_id is null
              and not exists (
                  select 1
                  from decision_simulation_runs child
                  where child.retry_of_run_id = runs.id
              )
              and (
                (
                    runs.status = 'completed'
                    and (
                        decisions.simulation_run_id is distinct from runs.id
                        or not exists (
                            select 1 from decision_simulation_candidates candidates
                            where candidates.simulation_run_id = runs.id
                              and candidates.candidate_id = runs.winner_candidate_id
                              and candidates.is_winner
                        )
                        or not exists (
                            select 1 from decision_simulation_events events
                            where events.simulation_run_id = runs.id
                              and events.event_type = 'completed'
                        )
                        or (
                            runs.mode = 'shadow'
                            and runs.model_version = 'pricing-elasticity-bootstrap-v1'
                            and (
                                decisions.status <> 'needs_data'
                                or (
                                    select count(*)
                                    from agent_action_experiments experiments
                                    where experiments.decision_id = runs.decision_id
                                ) <> 0
                            )
                        )
                        or (
                            not (
                                runs.mode = 'shadow'
                                and runs.model_version = 'pricing-elasticity-bootstrap-v1'
                            )
                            and (
                                decisions.status <> 'ready_for_approval'
                                or (
                                    select count(*)
                                    from agent_action_experiments experiments
                                    where experiments.decision_id = runs.decision_id
                                ) < case
                                    when runs.mode = 'shadow' then case
                                        when jsonb_typeof(
                                            coalesce(
                                                runs.evidence_snapshot -> 'shadowPublicDecision',
                                                runs.evidence_snapshot -> 'managerDecision'
                                            ) -> 'rankedActions'
                                        ) = 'array'
                                        then jsonb_array_length(
                                            coalesce(
                                                runs.evidence_snapshot -> 'shadowPublicDecision',
                                                runs.evidence_snapshot -> 'managerDecision'
                                            ) -> 'rankedActions'
                                        )
                                        else runs.candidate_count
                                    end
                                    else runs.completed_candidates
                                end
                            )
                        )
                    )
                )
                or (
                    runs.status in ('blocked', 'failed')
                    and runs.mode = 'enforced'
                    and (
                        decisions.simulation_run_id is distinct from runs.id
                        or decisions.status <> 'blocked'
                        or not exists (
                            select 1 from decision_simulation_events events
                            where events.simulation_run_id = runs.id
                              and events.event_type = runs.status
                        )
                    )
                )
                or (
                    runs.status in ('blocked', 'failed')
                    and runs.mode = 'shadow'
                    and (
                        decisions.simulation_run_id is distinct from runs.id
                        or not exists (
                            select 1 from decision_simulation_events events
                            where events.simulation_run_id = runs.id
                              and events.event_type = runs.status
                        )
                        or (
                            runs.model_version = 'pricing-elasticity-bootstrap-v1'
                            and (
                                decisions.status <> 'needs_data'
                                or (
                                    select count(*)
                                    from agent_action_experiments experiments
                                    where experiments.decision_id = runs.decision_id
                                ) <> 0
                            )
                        )
                        or (
                            runs.model_version <> 'pricing-elasticity-bootstrap-v1'
                            and (
                                select count(*)
                                from agent_action_experiments experiments
                                where experiments.decision_id = runs.decision_id
                            ) < case
                                when jsonb_typeof(
                                    coalesce(
                                        runs.evidence_snapshot -> 'shadowPublicDecision',
                                        runs.evidence_snapshot -> 'managerDecision'
                                    ) -> 'rankedActions'
                                ) = 'array'
                                then jsonb_array_length(
                                    coalesce(
                                        runs.evidence_snapshot -> 'shadowPublicDecision',
                                        runs.evidence_snapshot -> 'managerDecision'
                                    ) -> 'rankedActions'
                                )
                                else runs.candidate_count
                            end
                        )
                    )
                )
              )
            order by runs.updated_at, runs.id
            limit %s
            """,
            [max(1, min(int(limit), 1000))],
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_run(row) for row in rows]


def update_decision_simulation_run(
    simulation_run_id: str,
    *,
    status: str | object = _UNSET,
    completed_samples: int | object = _UNSET,
    total_samples: int | object = _UNSET,
    completed_candidates: int | object = _UNSET,
    progress: float | object = _UNSET,
    provisional_winner_candidate_id: str | None | object = _UNSET,
    winner_candidate_id: str | None | object = _UNSET,
    critic_result: dict[str, Any] | object = _UNSET,
    error: dict[str, Any] | object = _UNSET,
    heartbeat_at: datetime | None | object = _UNSET,
    started_at: datetime | None | object = _UNSET,
    completed_at: datetime | None | object = _UNSET,
) -> DecisionSimulationRunRecord:
    """Update run state; nullable fields can be explicitly cleared with ``None``."""
    ensure_memory_schema()
    assignments: list[str] = []
    params: list[Any] = []

    def add_value(column: str, value: Any, *, json_value: bool = False) -> None:
        if value is _UNSET:
            return
        assignments.append(f"{column} = %s")
        params.append(Json(value) if json_value else value)

    add_value("status", status)
    add_value("completed_samples", completed_samples)
    add_value("total_samples", total_samples)
    add_value("completed_candidates", completed_candidates)
    add_value("progress", progress)
    add_value("provisional_winner_candidate_id", provisional_winner_candidate_id)
    add_value("winner_candidate_id", winner_candidate_id)
    add_value("critic_result", critic_result, json_value=True)
    add_value("error", error, json_value=True)
    add_value("heartbeat_at", heartbeat_at)
    add_value("started_at", started_at)
    add_value("completed_at", completed_at)

    if status in {"running", "critic_running"} and started_at is _UNSET:
        assignments.append("started_at = coalesce(started_at, now())")
        if heartbeat_at is _UNSET:
            assignments.append("heartbeat_at = now()")
    if status in {"completed", "blocked", "failed", "superseded"} and completed_at is _UNSET:
        assignments.append("completed_at = coalesce(completed_at, now())")
    if not assignments:
        record = get_decision_simulation_run(simulation_run_id)
        if record is None:
            raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
        return record

    assignments.append("updated_at = now()")
    params.append(simulation_run_id)
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            update decision_simulation_runs
            set {', '.join(assignments)}
            where id = %s
            returning *
            """,
            params,
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    return _row_to_decision_simulation_run(row)


def update_decision_simulation_progress(
    simulation_run_id: str,
    *,
    completed_samples: int | None = None,
    total_samples: int | None = None,
    completed_candidates: int | None = None,
    provisional_winner_candidate_id: str | None | object = _UNSET,
) -> DecisionSimulationRunRecord:
    """Atomically persist progress and refresh the worker heartbeat."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_runs
            set completed_samples = coalesce(%s, completed_samples),
                total_samples = coalesce(%s, total_samples),
                completed_candidates = coalesce(%s, completed_candidates),
                provisional_winner_candidate_id = case
                    when %s then %s
                    else provisional_winner_candidate_id
                end,
                progress = case
                    when coalesce(%s, total_samples) > 0 then least(
                        1::numeric,
                        coalesce(%s, completed_samples)::numeric
                            / coalesce(%s, total_samples)::numeric
                    )
                    else 0
                end,
                heartbeat_at = now(),
                updated_at = now()
            where id = %s
            returning *
            """,
            [
                completed_samples,
                total_samples,
                completed_candidates,
                provisional_winner_candidate_id is not _UNSET,
                None
                if provisional_winner_candidate_id is _UNSET
                else provisional_winner_candidate_id,
                total_samples,
                completed_samples,
                total_samples,
                simulation_run_id,
            ],
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    return _row_to_decision_simulation_run(row)


def complete_decision_simulation_run(
    simulation_run_id: str,
    *,
    winner_candidate_id: str,
    completed_samples: int,
    total_samples: int,
    completed_candidates: int,
    critic_result: dict[str, Any],
) -> DecisionSimulationRunRecord:
    """Commit the final winner flag and completed run state in one transaction."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select 1
            from decision_simulation_candidates
            where simulation_run_id = %s
              and candidate_id = %s
              and status = 'completed'
              and rank = 1
            for update
            """,
            [simulation_run_id, winner_candidate_id],
        )
        if cur.fetchone() is None:
            raise ValueError("final winner must be the completed rank-1 candidate")
        cur.execute(
            """
            update decision_simulation_candidates
            set is_winner = (candidate_id = %s),
                updated_at = now()
            where simulation_run_id = %s
            """,
            [winner_candidate_id, simulation_run_id],
        )
        cur.execute(
            """
            update decision_simulation_runs
            set status = 'completed',
                completed_samples = %s,
                total_samples = %s,
                completed_candidates = %s,
                progress = 1,
                provisional_winner_candidate_id = %s,
                winner_candidate_id = %s,
                critic_result = %s,
                error = '{}'::jsonb,
                heartbeat_at = now(),
                completed_at = coalesce(completed_at, now()),
                updated_at = now()
            where id = %s
            returning *
            """,
            [
                completed_samples,
                total_samples,
                completed_candidates,
                winner_candidate_id,
                winner_candidate_id,
                Json(critic_result),
                simulation_run_id,
            ],
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    return _row_to_decision_simulation_run(row)


def clear_decision_simulation_winner_flags(simulation_run_id: str) -> None:
    """Ensure a non-completed run cannot expose a final candidate winner."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_candidates
            set is_winner = false,
                updated_at = now()
            where simulation_run_id = %s and is_winner
            """,
            [simulation_run_id],
        )


def touch_decision_simulation_heartbeat(
    simulation_run_id: str,
) -> DecisionSimulationRunRecord:
    return update_decision_simulation_run(
        simulation_run_id,
        heartbeat_at=datetime.now(timezone.utc),
    )


def mark_decision_simulation_run_superseded(
    simulation_run_id: str,
    *,
    superseded_by_run_id: str,
) -> DecisionSimulationRunRecord:
    """Link a retry in both directions and make the old run terminal."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update decision_simulation_candidates
            set is_winner = false,
                updated_at = now()
            where simulation_run_id = %s and is_winner
            """,
            [simulation_run_id],
        )
        cur.execute(
            """
            update decision_simulation_runs
            set retry_of_run_id = coalesce(retry_of_run_id, %s),
                updated_at = now()
            where id = %s
            """,
            [simulation_run_id, superseded_by_run_id],
        )
        if cur.rowcount == 0:
            raise KeyError(f"Unknown replacement simulation run: {superseded_by_run_id}")
        cur.execute(
            """
            update decision_simulation_runs
            set status = 'superseded',
                superseded_by_run_id = %s,
                completed_at = coalesce(completed_at, now()),
                updated_at = now()
            where id = %s
            returning *
            """,
            [superseded_by_run_id, simulation_run_id],
        )
        row = cur.fetchone()
        if row is not None:
            event_id = f"{simulation_run_id}:superseded:{superseded_by_run_id}"
            cur.execute(
                """
                insert into decision_simulation_events (
                    event_id, simulation_run_id, event_type, progress, payload
                )
                values (
                    %s, %s, 'superseded', %s, jsonb_build_object(
                        'supersededByRunId', %s::text
                    )
                )
                on conflict (event_id) do nothing
                returning id
                """,
                [event_id, simulation_run_id, row["progress"], superseded_by_run_id],
            )
            event_row = cur.fetchone()
            if event_row is not None:
                cur.execute(
                    """
                    select pg_notify(
                        'decision_simulation_events',
                        json_build_object(
                            'event_id', %s::text,
                            'id', %s::bigint,
                            'simulation_run_id', %s::text,
                            'event_type', 'superseded'
                        )::text
                    )
                    """,
                    [event_id, event_row["id"], simulation_run_id],
                )
    if row is None:
        raise KeyError(f"Unknown decision simulation run: {simulation_run_id}")
    return _row_to_decision_simulation_run(row)


def backfill_decision_simulation_superseded_events(*, limit: int = 100) -> list[str]:
    """Repair legacy/crash-window superseded runs that lack their journal event."""

    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            with missing as (
                select runs.id, runs.superseded_by_run_id, runs.progress
                from decision_simulation_runs runs
                where runs.status = 'superseded'
                  and runs.superseded_by_run_id is not null
                  and not exists (
                      select 1 from decision_simulation_events events
                      where events.simulation_run_id = runs.id
                        and events.event_type = 'superseded'
                  )
                order by runs.updated_at, runs.id
                limit %s
            )
            insert into decision_simulation_events (
                event_id, simulation_run_id, event_type, progress, payload
            )
            select
                missing.id || ':superseded:' || missing.superseded_by_run_id,
                missing.id,
                'superseded',
                missing.progress,
                jsonb_build_object('supersededByRunId', missing.superseded_by_run_id)
            from missing
            on conflict (event_id) do nothing
            returning id, event_id, simulation_run_id
            """,
            [max(1, min(int(limit), 1000))],
        )
        rows = cur.fetchall()
        for event_row in rows:
            cur.execute(
                """
                select pg_notify(
                    'decision_simulation_events',
                    json_build_object(
                        'event_id', %s::text,
                        'id', %s::bigint,
                        'simulation_run_id', %s::text,
                        'event_type', 'superseded'
                    )::text
                )
                """,
                [
                    event_row["event_id"],
                    event_row["id"],
                    event_row["simulation_run_id"],
                ],
            )
    return [str(row["simulation_run_id"]) for row in rows]


def set_manager_decision_simulation_run(
    decision_id: str,
    *,
    simulation_run_id: str | None,
    status: str | None = None,
) -> ManagerDecisionRecord:
    """Point a decision at its current simulation and optionally transition its status."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            update manager_decisions
            set simulation_run_id = %s,
                status = coalesce(%s, status),
                updated_at = now()
            where id = %s
            returning *
            """,
            [simulation_run_id, status, decision_id],
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"Unknown manager decision: {decision_id}")
    return _row_to_manager_decision(row)


def upsert_decision_simulation_candidate(
    *,
    simulation_run_id: str,
    candidate_id: str,
    action_id: str,
    status: str = "pending",
    candidate_payload: dict[str, Any] | None = None,
    evidence_refs: list[Any] | None = None,
    constraint_violations: list[Any] | None = None,
    distribution_summary: dict[str, Any] | None = None,
    score_breakdown: dict[str, Any] | None = None,
    expected_effect_mean: float | None = None,
    profit_p10: float | None = None,
    cvar_10_per_day: float | None = None,
    execution_risk: float | None = None,
    utility_score: float | None = None,
    rank: int | None = None,
    is_winner: bool = False,
    completed_samples: int = 0,
    error: dict[str, Any] | None = None,
) -> DecisionSimulationCandidateRecord:
    """Idempotently store candidate state without regressing a terminal candidate."""
    ensure_memory_schema()
    terminal = status in {"completed", "rejected", "needs_data", "failed"}
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into decision_simulation_candidates (
                simulation_run_id, candidate_id, action_id, status,
                candidate_payload, evidence_refs, constraint_violations,
                distribution_summary, score_breakdown, expected_effect_mean,
                profit_p10, cvar_10_per_day, execution_risk, utility_score,
                rank, is_winner, completed_samples, error, completed_at
            )
            values (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s,
                case when %s then now() else null end
            )
            on conflict (simulation_run_id, candidate_id) do update
            set action_id = excluded.action_id,
                status = excluded.status,
                candidate_payload = excluded.candidate_payload,
                evidence_refs = excluded.evidence_refs,
                constraint_violations = excluded.constraint_violations,
                distribution_summary = excluded.distribution_summary,
                score_breakdown = excluded.score_breakdown,
                expected_effect_mean = excluded.expected_effect_mean,
                profit_p10 = excluded.profit_p10,
                cvar_10_per_day = excluded.cvar_10_per_day,
                execution_risk = excluded.execution_risk,
                utility_score = excluded.utility_score,
                rank = excluded.rank,
                is_winner = excluded.is_winner,
                completed_samples = excluded.completed_samples,
                error = excluded.error,
                completed_at = case
                    when excluded.status in ('completed', 'rejected', 'needs_data', 'failed')
                        then coalesce(decision_simulation_candidates.completed_at, now())
                    else null
                end,
                updated_at = now()
            where not (
                decision_simulation_candidates.status in ('completed', 'rejected', 'needs_data', 'failed')
                and excluded.status in ('pending', 'running')
            )
            returning *
            """,
            [
                simulation_run_id,
                candidate_id,
                action_id,
                status,
                Json(candidate_payload or {}),
                Json(evidence_refs or []),
                Json(constraint_violations or []),
                Json(distribution_summary or {}),
                Json(score_breakdown or {}),
                expected_effect_mean,
                profit_p10,
                cvar_10_per_day,
                execution_risk,
                utility_score,
                rank,
                is_winner,
                completed_samples,
                Json(error or {}),
                terminal,
            ],
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                select * from decision_simulation_candidates
                where simulation_run_id = %s and candidate_id = %s
                """,
                [simulation_run_id, candidate_id],
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to upsert decision simulation candidate")
    return _row_to_decision_simulation_candidate(row)


def get_decision_simulation_candidate(
    simulation_run_id: str,
    candidate_id: str,
) -> DecisionSimulationCandidateRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_candidates
            where simulation_run_id = %s and candidate_id = %s
            """,
            [simulation_run_id, candidate_id],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_candidate(row) if row else None


def list_decision_simulation_candidates(
    simulation_run_id: str,
) -> list[DecisionSimulationCandidateRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_candidates
            where simulation_run_id = %s
            order by rank nulls last, candidate_id
            """,
            [simulation_run_id],
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_candidate(row) for row in rows]


def upsert_decision_simulation_batch(
    *,
    simulation_run_id: str,
    candidate_id: str,
    batch_index: int,
    seed: int,
    sample_count: int,
    status: str,
    aggregates: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    error: dict[str, Any] | None = None,
) -> DecisionSimulationBatchRecord:
    """Idempotently store a batch; a completed batch cannot regress to in-flight."""
    ensure_memory_schema()
    terminal = status in {"completed", "failed"}
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into decision_simulation_batches (
                simulation_run_id, candidate_id, batch_index, seed,
                sample_count, status, aggregates, duration_ms, error, completed_at
            )
            values (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                case when %s then now() else null end
            )
            on conflict (simulation_run_id, candidate_id, batch_index) do update
            set seed = excluded.seed,
                sample_count = excluded.sample_count,
                status = excluded.status,
                aggregates = excluded.aggregates,
                duration_ms = excluded.duration_ms,
                error = excluded.error,
                completed_at = case
                    when excluded.status in ('completed', 'failed')
                        then coalesce(decision_simulation_batches.completed_at, now())
                    else null
                end,
                updated_at = now()
            where not (
                decision_simulation_batches.status = 'completed'
                and excluded.status in ('pending', 'running')
            )
            returning *
            """,
            [
                simulation_run_id,
                candidate_id,
                batch_index,
                seed,
                sample_count,
                status,
                Json(aggregates or {}),
                duration_ms,
                Json(error or {}),
                terminal,
            ],
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                select * from decision_simulation_batches
                where simulation_run_id = %s and candidate_id = %s and batch_index = %s
                """,
                [simulation_run_id, candidate_id, batch_index],
            )
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to upsert decision simulation batch")
    return _row_to_decision_simulation_batch(row)


def get_decision_simulation_batch(
    simulation_run_id: str,
    candidate_id: str,
    batch_index: int,
) -> DecisionSimulationBatchRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_batches
            where simulation_run_id = %s and candidate_id = %s and batch_index = %s
            """,
            [simulation_run_id, candidate_id, batch_index],
        )
        row = cur.fetchone()
    return _row_to_decision_simulation_batch(row) if row else None


def list_decision_simulation_batches(
    simulation_run_id: str,
    *,
    candidate_id: str | None = None,
    statuses: Iterable[str] | None = None,
) -> list[DecisionSimulationBatchRecord]:
    ensure_memory_schema()
    conditions = ["simulation_run_id = %s"]
    params: list[Any] = [simulation_run_id]
    if candidate_id is not None:
        conditions.append("candidate_id = %s")
        params.append(candidate_id)
    status_values = list(statuses or [])
    if status_values:
        conditions.append("status = any(%s)")
        params.append(status_values)
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select * from decision_simulation_batches
            where {' and '.join(conditions)}
            order by candidate_id, batch_index
            """,
            params,
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_batch(row) for row in rows]


def append_decision_simulation_event(
    *,
    simulation_run_id: str,
    event_type: str,
    event_id: str | None = None,
    candidate_id: str | None = None,
    batch_index: int | None = None,
    progress: float | None = None,
    payload: dict[str, Any] | None = None,
) -> DecisionSimulationEventRecord:
    """Append a replayable event and notify listeners in the same transaction."""
    import uuid

    ensure_memory_schema()
    resolved_event_id = event_id or str(uuid.uuid4())
    inserted = False
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into decision_simulation_events (
                event_id, simulation_run_id, event_type, candidate_id,
                batch_index, progress, payload
            )
            values (%s, %s, %s, %s, %s, %s, %s)
            on conflict (event_id) do nothing
            returning *
            """,
            [
                resolved_event_id,
                simulation_run_id,
                event_type,
                candidate_id,
                batch_index,
                progress,
                Json(payload or {}),
            ],
        )
        row = cur.fetchone()
        inserted = row is not None
        if row is None:
            cur.execute(
                "select * from decision_simulation_events where event_id = %s",
                [resolved_event_id],
            )
            row = cur.fetchone()
        if inserted and row is not None:
            cur.execute(
                """
                select pg_notify(
                    'decision_simulation_events',
                    json_build_object(
                        'event_id', %s::text,
                        'id', %s::bigint,
                        'simulation_run_id', %s::text,
                        'event_type', %s::text
                    )::text
                )
                """,
                [row["event_id"], row["id"], row["simulation_run_id"], row["event_type"]],
            )
    if row is None:
        raise RuntimeError("Failed to append or recover decision simulation event")
    return _row_to_decision_simulation_event(row)


def list_decision_simulation_events(
    simulation_run_id: str,
    *,
    after_id: int = 0,
    limit: int = 500,
) -> list[DecisionSimulationEventRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from decision_simulation_events
            where simulation_run_id = %s and id > %s
            order by id
            limit %s
            """,
            [simulation_run_id, after_id, max(1, min(int(limit), 5000))],
        )
        rows = cur.fetchall()
    return [_row_to_decision_simulation_event(row) for row in rows]


def upsert_agent_action_experiment(
    *,
    experiment_id: str,
    decision_id: str,
    action_id: str,
    family: str,
    execution_mode: str,
    target: dict[str, Any] | None = None,
    from_value: Any = None,
    to_value: Any = None,
    expected_effect: dict[str, Any] | None = None,
    horizon: dict[str, Any] | None = None,
    evidence: list[Any] | None = None,
    prerequisites: list[Any] | None = None,
    guardrails: dict[str, Any] | None = None,
    stop_condition: dict[str, Any] | None = None,
    rollback: dict[str, Any] | None = None,
    measurement_spec: dict[str, Any] | None = None,
    expires_at: datetime | None = None,
) -> AgentActionExperimentRecord:
    """Update proposal data only while the action has not entered its lifecycle."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_action_experiments (
                id, decision_id, action_id, family, execution_mode, target,
                from_value, to_value, expected_effect, horizon, evidence,
                prerequisites, guardrails, stop_condition, rollback,
                measurement_spec, expires_at
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (id) do update
            set family = excluded.family,
                execution_mode = excluded.execution_mode,
                target = excluded.target,
                from_value = excluded.from_value,
                to_value = excluded.to_value,
                expected_effect = excluded.expected_effect,
                horizon = excluded.horizon,
                evidence = excluded.evidence,
                prerequisites = excluded.prerequisites,
                guardrails = excluded.guardrails,
                stop_condition = excluded.stop_condition,
                rollback = excluded.rollback,
                measurement_spec = excluded.measurement_spec,
                expires_at = excluded.expires_at,
                version = agent_action_experiments.version + 1,
                updated_at = now()
            where agent_action_experiments.state = 'proposed'
            returning *
            """,
            [
                experiment_id,
                decision_id,
                action_id,
                family,
                execution_mode,
                Json(target or {}),
                Json(from_value if from_value is not None else {}),
                Json(to_value if to_value is not None else {}),
                Json(expected_effect or {}),
                Json(horizon or {}),
                Json(evidence or []),
                Json(prerequisites or []),
                Json(guardrails or {}),
                Json(stop_condition or {}),
                Json(rollback or {}),
                Json(measurement_spec or {}),
                expires_at,
            ],
        )
        row = cur.fetchone()
        if row is None:
            cur.execute("select * from agent_action_experiments where id = %s", [experiment_id])
            row = cur.fetchone()
    return _row_to_agent_action_experiment(row)


def get_agent_action_experiment(experiment_id: str) -> AgentActionExperimentRecord | None:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute("select * from agent_action_experiments where id = %s", [experiment_id])
        row = cur.fetchone()
    return _row_to_agent_action_experiment(row) if row else None


def list_agent_action_experiments_for_decision(
    decision_id: str,
) -> list[AgentActionExperimentRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from agent_action_experiments
            where decision_id = %s
            order by created_at, id
            """,
            [decision_id],
        )
        rows = cur.fetchall()
    return [_row_to_agent_action_experiment(row) for row in rows]


def list_pending_agent_action_experiments(
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> list[AgentActionExperimentRecord]:
    """Return durable lifecycle rows whose asynchronous work needs reconciliation."""

    ensure_memory_schema()
    bounded_limit = max(1, min(int(limit), 500))
    observed_at = now or datetime.now(timezone.utc)
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from agent_action_experiments
            where state = 'applying'
               or (
                    state in ('applied', 'measuring')
                    and measurement_due_at is not null
                    and measurement_due_at <= %s
               )
            order by
                case when state = 'applying' then 0 else 1 end,
                coalesce(measurement_due_at, updated_at),
                id
            limit %s
            """,
            [observed_at, bounded_limit],
        )
        rows = cur.fetchall()
    return [_row_to_agent_action_experiment(row) for row in rows]


def claim_agent_action_experiment_approval(
    *,
    experiment_id: str,
    actor_user_id: int,
    idempotency_key: str,
) -> tuple[AgentActionExperimentRecord | None, bool, bool]:
    """Return (record, claimed, replay). The row lock makes concurrent approval safe."""
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from agent_action_experiments where id = %s for update",
            [experiment_id],
        )
        row = cur.fetchone()
        if row is None:
            return None, False, False
        if row.get("approval_idempotency_key") == idempotency_key:
            return _row_to_agent_action_experiment(row), False, True
        if row.get("approval_idempotency_key") or row.get("state") != "proposed":
            return _row_to_agent_action_experiment(row), False, False
        cur.execute(
            """
            update agent_action_experiments
            set state = 'approved',
                approved_by = %s,
                owner_user_id = coalesce(owner_user_id, %s),
                approved_at = now(),
                approval_idempotency_key = %s,
                version = version + 1,
                updated_at = now()
            where id = %s
            returning *
            """,
            [actor_user_id, actor_user_id, idempotency_key, experiment_id],
        )
        updated = cur.fetchone()
        _insert_action_audit(
            cur,
            experiment_id=experiment_id,
            actor_user_id=actor_user_id,
            event="approved",
            from_state="proposed",
            to_state="approved",
            idempotency_key=idempotency_key,
        )
    return _row_to_agent_action_experiment(updated), True, False


def reject_agent_action_experiment(
    *,
    experiment_id: str,
    actor_user_id: int,
    idempotency_key: str,
    reason: str,
    manager_label: str | None,
) -> tuple[AgentActionExperimentRecord | None, bool, bool]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from agent_action_experiments where id = %s for update",
            [experiment_id],
        )
        row = cur.fetchone()
        if row is None:
            return None, False, False
        if row.get("rejection_idempotency_key") == idempotency_key:
            return _row_to_agent_action_experiment(row), False, True
        if row.get("rejection_idempotency_key") or row.get("state") != "proposed":
            return _row_to_agent_action_experiment(row), False, False
        cur.execute(
            """
            update agent_action_experiments
            set state = 'rejected',
                rejected_by = %s,
                rejected_at = now(),
                rejection_idempotency_key = %s,
                rejection_reason = %s,
                manager_label = %s,
                version = version + 1,
                updated_at = now()
            where id = %s
            returning *
            """,
            [actor_user_id, idempotency_key, reason, manager_label, experiment_id],
        )
        updated = cur.fetchone()
        _insert_action_audit(
            cur,
            experiment_id=experiment_id,
            actor_user_id=actor_user_id,
            event="rejected",
            from_state="proposed",
            to_state="rejected",
            idempotency_key=idempotency_key,
            details={"reason": reason, "manager_label": manager_label},
        )
    return _row_to_agent_action_experiment(updated), True, False


def transition_agent_action_experiment(
    *,
    experiment_id: str,
    expected_states: set[str],
    state: str,
    event: str,
    actor_user_id: int | None = None,
    idempotency_key: str | None = None,
    preflight_json: dict[str, Any] | None = None,
    actuator_result: dict[str, Any] | None = None,
    external_operation_id: str | None = None,
    applied_at: datetime | None = None,
    measurement_due_at: datetime | None = None,
    close_outcome: str | None = None,
    last_error: str | None = None,
    expected_version: int | None = None,
) -> tuple[AgentActionExperimentRecord | None, bool]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "select * from agent_action_experiments where id = %s for update",
            [experiment_id],
        )
        row = cur.fetchone()
        if row is None:
            return None, False
        from_state = str(row["state"])
        if (
            from_state not in expected_states
            or (
                expected_version is not None
                and int(row.get("version") or 1) != int(expected_version)
            )
        ):
            return _row_to_agent_action_experiment(row), False
        cur.execute(
            """
            update agent_action_experiments
            set state = %s,
                preflight_json = case when %s is null then preflight_json else %s end,
                actuator_result = case when %s is null then actuator_result else %s end,
                external_operation_id = coalesce(%s, external_operation_id),
                applied_at = coalesce(%s, applied_at),
                measurement_due_at = coalesce(%s, measurement_due_at),
                close_outcome = coalesce(%s, close_outcome),
                last_error = %s,
                version = version + 1,
                updated_at = now()
            where id = %s
            returning *
            """,
            [
                state,
                Json(preflight_json) if preflight_json is not None else None,
                Json(preflight_json or {}),
                Json(actuator_result) if actuator_result is not None else None,
                Json(actuator_result or {}),
                external_operation_id,
                applied_at,
                measurement_due_at,
                close_outcome,
                last_error,
                experiment_id,
            ],
        )
        updated = cur.fetchone()
        _insert_action_audit(
            cur,
            experiment_id=experiment_id,
            actor_user_id=actor_user_id,
            event=event,
            from_state=from_state,
            to_state=state,
            idempotency_key=idempotency_key,
            details={"error": last_error} if last_error else {},
        )
    return _row_to_agent_action_experiment(updated), True


def append_agent_action_measurement(
    *,
    experiment_id: str,
    phase: str,
    status: str,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    data_as_of: datetime | None = None,
    idempotency_key: str | None = None,
    metrics: dict[str, Any] | None = None,
    causal_valid: bool = False,
    observed_delta: float | None = None,
    did_effect: float | None = None,
    forecast_low: float | None = None,
    forecast_base: float | None = None,
    forecast_high: float | None = None,
    signed_bias: float | None = None,
    mae: float | None = None,
    inside_forecast_range: bool | None = None,
    created_by: int | None = None,
) -> AgentActionMeasurementRecord:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            insert into agent_action_measurements (
                experiment_id, phase, status, period_start, period_end, data_as_of,
                idempotency_key, metrics, causal_valid, observed_delta, did_effect, forecast_low,
                forecast_base, forecast_high, signed_bias, mae,
                inside_forecast_range, created_by
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (experiment_id, idempotency_key)
                where idempotency_key is not null
                do nothing
            returning *
            """,
            [
                experiment_id,
                phase,
                status,
                period_start,
                period_end,
                data_as_of,
                idempotency_key,
                Json(metrics or {}),
                causal_valid,
                observed_delta,
                did_effect,
                forecast_low,
                forecast_base,
                forecast_high,
                signed_bias,
                mae,
                inside_forecast_range,
                created_by,
            ],
        )
        row = cur.fetchone()
        if row is None and idempotency_key is not None:
            cur.execute(
                """
                select * from agent_action_measurements
                where experiment_id = %s and idempotency_key = %s
                """,
                [experiment_id, idempotency_key],
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("measurement insert returned no durable row")
    return _row_to_agent_action_measurement(row)


def list_agent_action_measurements(
    experiment_id: str,
) -> list[AgentActionMeasurementRecord]:
    ensure_memory_schema()
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select * from agent_action_measurements
            where experiment_id = %s
            order by created_at, id
            """,
            [experiment_id],
        )
        rows = cur.fetchall()
    return [_row_to_agent_action_measurement(row) for row in rows]


def _insert_action_audit(
    cur: Any,
    *,
    experiment_id: str,
    actor_user_id: int | None,
    event: str,
    from_state: str | None,
    to_state: str | None,
    idempotency_key: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    cur.execute(
        """
        insert into agent_action_audit_log (
            experiment_id, actor_user_id, event, from_state, to_state,
            idempotency_key, details
        )
        values (%s, %s, %s, %s, %s, %s, %s)
        """,
        [
            experiment_id,
            actor_user_id,
            event,
            from_state,
            to_state,
            idempotency_key,
            Json(details or {}),
        ],
    )


def _row_to_message(row: dict[str, Any]) -> ChatMemoryMessage:
    return ChatMemoryMessage(
        seq=int(row["seq"]),
        role=ChatRole(row["role"]),
        content=row["content"],
        created_at=row.get("created_at"),
    )


def _row_to_stored_message(row: dict[str, Any]) -> StoredChatMessage:
    return StoredChatMessage(
        seq=int(row["seq"]),
        role=ChatRole(row["role"]),
        content=row["content"],
        metadata=dict(row.get("metadata") or {}),
        created_at=row.get("created_at"),
    )


def _row_to_chat_session_summary(row: dict[str, Any]) -> ChatSessionSummaryRecord:
    return ChatSessionSummaryRecord(
        chat_id=str(row["chat_id"]),
        status=str(row["status"]),
        model_name=row.get("model_name"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        experiment_program_id=row.get("experiment_program_id"),
        experiment_candidate_id=row.get("experiment_candidate_id"),
        message_count=int(row.get("message_count") or 0),
        run_count=int(row.get("run_count") or 0),
        latest_run_id=(
            int(row["latest_run_id"])
            if row.get("latest_run_id") is not None
            else None
        ),
        latest_run_status=row.get("latest_run_status"),
        latest_run_intent=row.get("latest_run_intent"),
        latest_run_created_at=row.get("latest_run_created_at"),
        latest_server_message_id=(
            int(row["latest_server_message_id"])
            if row.get("latest_server_message_id") is not None
            else None
        ),
        last_user_message=row.get("last_user_message"),
        last_user_message_at=row.get("last_user_message_at"),
        last_assistant_message=row.get("last_assistant_message"),
        last_assistant_message_at=row.get("last_assistant_message_at"),
    )


def _row_to_agent_run_trace(row: dict[str, Any]) -> AgentRunTraceRecord:
    return AgentRunTraceRecord(
        id=int(row["id"]),
        chat_id=str(row["chat_id"]),
        assistant_message_seq=(
            int(row["assistant_message_seq"])
            if row.get("assistant_message_seq") is not None
            else None
        ),
        client_message_id=row.get("client_message_id"),
        intent=str(row["intent"]),
        primary_agent=row.get("primary_agent"),
        status=str(row["status"]),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        events=list(row.get("events") or []),
        summary=dict(row.get("summary") or {}),
        created_at=row.get("created_at"),
    )


def _row_to_agent_run_round(row: dict[str, Any]) -> AgentRunRoundRecord:
    return AgentRunRoundRecord(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        round_number=int(row["round_number"]),
        round_type=str(row["round_type"]),
        status=str(row["status"]),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        completed_at=row.get("completed_at"),
    )


def _row_to_agent_run_question(row: dict[str, Any]) -> AgentRunQuestionRecord:
    return AgentRunQuestionRecord(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        round_id=int(row["round_id"]),
        source_agent=row.get("source_agent"),
        target_agent=str(row["target_agent"]),
        question=str(row["question"]),
        reason=row.get("reason"),
        verification_goal=dict(row.get("verification_goal") or {}),
        priority=int(row["priority"]) if row.get("priority") is not None else None,
        status=str(row["status"]),
        job_id=row.get("job_id"),
        queued_at=row.get("queued_at"),
        attempt_count=int(row.get("attempt_count") or 0),
        last_error=row.get("last_error"),
        timeout_seconds=(
            int(row["timeout_seconds"])
            if row.get("timeout_seconds") is not None
            else None
        ),
        parent_question_id=(
            int(row["parent_question_id"])
            if row.get("parent_question_id") is not None
            else None
        ),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
    )


def _row_to_agent_run_response(row: dict[str, Any]) -> AgentRunResponseRecord:
    return AgentRunResponseRecord(
        id=int(row["id"]),
        question_id=int(row["question_id"]),
        agent_name=str(row["agent_name"]),
        status=str(row["status"]),
        summary=row.get("summary"),
        raw_output=dict(row.get("raw_output") or {}),
        has_business_evidence=bool(row.get("has_business_evidence")),
        runtime_failure=(
            dict(row["runtime_failure"])
            if row.get("runtime_failure") is not None
            else None
        ),
        created_at=row.get("created_at"),
        completed_at=row.get("completed_at"),
    )


def _row_to_agent_run_gap(row: dict[str, Any]) -> AgentRunGapRecord:
    return AgentRunGapRecord(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        response_id=int(row["response_id"]) if row.get("response_id") is not None else None,
        sku=row.get("sku"),
        source_agent=row.get("source_agent"),
        target_agent=row.get("target_agent"),
        metric=str(row["metric"]),
        reason=str(row["reason"]),
        impact=row.get("impact"),
        blocking=bool(row.get("blocking")),
        status=str(row["status"]),
        evidence_status=str(row.get("evidence_status") or "unconfirmed"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_stored_agent_action_task(row: dict[str, Any]) -> StoredAgentActionTask:
    return StoredAgentActionTask(
        chat_id=str(row["chat_id"]),
        task_id=str(row["task_id"]),
        action_id=str(row["action_id"]),
        assistant_message_seq=int(row["assistant_message_seq"]),
        source=str(row["source"]),
        factor=str(row["factor"]),
        observation=row.get("observation"),
        impact=row.get("impact"),
        cause=row.get("cause"),
        action=str(row["action"]),
        title=row.get("title"),
        action_type=row.get("action_type"),
        priority=row.get("priority"),
        targets=[str(item) for item in (row.get("targets") or [])],
        success_criteria=row.get("success_criteria"),
        agent_prompt=row.get("agent_prompt"),
        evidence=[str(item) for item in (row.get("evidence") or [])],
        confidence=row.get("confidence"),
        driver_class=row.get("driver_class"),
        intent=row.get("intent"),
        raw_action=dict(row.get("raw_action") or {}),
        action_kind=row.get("action_kind"),
        business_goal=row.get("business_goal"),
        decision_type=row.get("decision_type"),
        required_metrics=[str(item) for item in (row.get("required_metrics") or [])],
        forecast_spec=dict(row.get("forecast_spec") or {}),
        execution_plan=row.get("execution_plan"),
        created_at=row.get("created_at"),
    )


def _row_to_agent_action_execution(row: dict[str, Any]) -> AgentActionExecutionRecord:
    return AgentActionExecutionRecord(
        id=int(row["id"]),
        source_chat_id=str(row["source_chat_id"]),
        source_task_id=str(row.get("source_task_id") or row["task_id"]),
        task_id=str(row["task_id"]),
        object_id=str(row.get("object_id") or row["task_id"]),
        object_label=row.get("object_label"),
        object_type=row.get("object_type"),
        object_external_id=row.get("object_external_id"),
        object_payload=dict(row.get("object_payload") or {}),
        execution_chat_id=str(row["execution_chat_id"]),
        run_id=int(row["run_id"]) if row.get("run_id") is not None else None,
        client_message_id=str(row["client_message_id"]),
        status=str(row["status"]),
        execution_mode=str(row.get("execution_mode") or "terminal_action"),
        prompt=str(row["prompt"]),
        task_snapshot=dict(row.get("task_snapshot") or {}),
        result_json=dict(row.get("result_json") or {}),
        decision_status=row.get("decision_status"),
        review_summary=row.get("review_summary"),
        answer=row.get("answer"),
        server_message_id=(
            int(row["server_message_id"])
            if row.get("server_message_id") is not None
            else None
        ),
        error=row.get("error"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        completed_at=row.get("completed_at"),
    )


def _row_to_app_user(row: dict[str, Any]) -> AppUserRecord:
    return AppUserRecord(
        id=int(row["id"]),
        username=str(row["username"]),
        password_hash=str(row["password_hash"]),
        role=str(row["role"]),
        is_active=bool(row.get("is_active")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_manager_decision(row: dict[str, Any]) -> ManagerDecisionRecord:
    return ManagerDecisionRecord(
        id=str(row["id"]),
        source_key=row.get("source_key"),
        chat_id=row.get("chat_id"),
        run_id=int(row["run_id"]) if row.get("run_id") is not None else None,
        decision_date=row.get("decision_date"),
        objective=str(row["objective"]),
        status=str(row["status"]),
        request_coverage=dict(row.get("request_coverage") or {}),
        evidence_as_of=row.get("evidence_as_of"),
        ranked_actions=[dict(item) for item in (row.get("ranked_actions") or [])],
        rejected_candidates=[
            dict(item) for item in (row.get("rejected_candidates") or [])
        ],
        blockers=list(row.get("blockers") or []),
        decision_json=dict(row.get("decision_json") or {}),
        public_json=dict(row.get("public_json") or {}),
        simulation_run_id=row.get("simulation_run_id"),
        experiment_program_id=row.get("experiment_program_id"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_decision_simulation_run(
    row: dict[str, Any],
) -> DecisionSimulationRunRecord:
    return DecisionSimulationRunRecord(
        id=str(row["id"]),
        decision_id=str(row["decision_id"]),
        chat_id=row.get("chat_id"),
        run_id=int(row["run_id"]) if row.get("run_id") is not None else None,
        mode=str(row["mode"]),
        status=str(row["status"]),
        objective=str(row["objective"]),
        model_version=str(row["model_version"]),
        evidence_snapshot=dict(row.get("evidence_snapshot") or {}),
        evidence_hash=str(row["evidence_hash"]),
        master_seed=int(row["master_seed"]),
        sample_count=int(row["sample_count"]),
        batch_size=int(row["batch_size"]),
        candidate_count=int(row["candidate_count"]),
        completed_samples=int(row["completed_samples"]),
        total_samples=int(row["total_samples"]),
        completed_candidates=int(row["completed_candidates"]),
        progress=float(row["progress"]),
        provisional_winner_candidate_id=row.get("provisional_winner_candidate_id"),
        winner_candidate_id=row.get("winner_candidate_id"),
        critic_result=dict(row.get("critic_result") or {}),
        error=dict(row.get("error") or {}),
        idempotency_key=row.get("idempotency_key"),
        retry_of_run_id=row.get("retry_of_run_id"),
        superseded_by_run_id=row.get("superseded_by_run_id"),
        retry_attempt=int(row.get("retry_attempt") or 0),
        heartbeat_at=row.get("heartbeat_at"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        experiment_program_id=row.get("experiment_program_id"),
        experiment_candidate_id=row.get("experiment_candidate_id"),
    )


def _row_to_decision_simulation_candidate(
    row: dict[str, Any],
) -> DecisionSimulationCandidateRecord:
    return DecisionSimulationCandidateRecord(
        simulation_run_id=str(row["simulation_run_id"]),
        candidate_id=str(row["candidate_id"]),
        action_id=str(row["action_id"]),
        status=str(row["status"]),
        candidate_payload=dict(row.get("candidate_payload") or {}),
        evidence_refs=list(row.get("evidence_refs") or []),
        constraint_violations=list(row.get("constraint_violations") or []),
        distribution_summary=dict(row.get("distribution_summary") or {}),
        score_breakdown=dict(row.get("score_breakdown") or {}),
        expected_effect_mean=(
            float(row["expected_effect_mean"])
            if row.get("expected_effect_mean") is not None
            else None
        ),
        profit_p10=(
            float(row["profit_p10"]) if row.get("profit_p10") is not None else None
        ),
        cvar_10_per_day=(
            float(row["cvar_10_per_day"])
            if row.get("cvar_10_per_day") is not None
            else None
        ),
        execution_risk=(
            float(row["execution_risk"])
            if row.get("execution_risk") is not None
            else None
        ),
        utility_score=(
            float(row["utility_score"])
            if row.get("utility_score") is not None
            else None
        ),
        rank=int(row["rank"]) if row.get("rank") is not None else None,
        is_winner=bool(row.get("is_winner")),
        completed_samples=int(row.get("completed_samples") or 0),
        error=dict(row.get("error") or {}),
        completed_at=row.get("completed_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        experiment_program_id=row.get("experiment_program_id"),
        experiment_candidate_id=row.get("experiment_candidate_id"),
    )


def _row_to_decision_simulation_batch(
    row: dict[str, Any],
) -> DecisionSimulationBatchRecord:
    return DecisionSimulationBatchRecord(
        id=int(row["id"]),
        simulation_run_id=str(row["simulation_run_id"]),
        candidate_id=str(row["candidate_id"]),
        batch_index=int(row["batch_index"]),
        seed=int(row["seed"]),
        sample_count=int(row["sample_count"]),
        status=str(row["status"]),
        aggregates=dict(row.get("aggregates") or {}),
        duration_ms=(
            float(row["duration_ms"]) if row.get("duration_ms") is not None else None
        ),
        error=dict(row.get("error") or {}),
        completed_at=row.get("completed_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_decision_simulation_event(
    row: dict[str, Any],
) -> DecisionSimulationEventRecord:
    return DecisionSimulationEventRecord(
        id=int(row["id"]),
        event_id=str(row["event_id"]),
        simulation_run_id=str(row["simulation_run_id"]),
        event_type=str(row["event_type"]),
        candidate_id=row.get("candidate_id"),
        batch_index=(
            int(row["batch_index"]) if row.get("batch_index") is not None else None
        ),
        progress=float(row["progress"]) if row.get("progress") is not None else None,
        payload=dict(row.get("payload") or {}),
        created_at=row.get("created_at"),
    )


def _row_to_agent_action_experiment(
    row: dict[str, Any],
) -> AgentActionExperimentRecord:
    return AgentActionExperimentRecord(
        id=str(row["id"]),
        decision_id=str(row["decision_id"]),
        action_id=str(row["action_id"]),
        family=str(row["family"]),
        target=dict(row.get("target") or {}),
        execution_mode=str(row["execution_mode"]),
        state=str(row["state"]),
        from_value=row.get("from_value"),
        to_value=row.get("to_value"),
        expected_effect=dict(row.get("expected_effect") or {}),
        horizon=dict(row.get("horizon") or {}),
        evidence=list(row.get("evidence") or []),
        prerequisites=list(row.get("prerequisites") or []),
        guardrails=dict(row.get("guardrails") or {}),
        stop_condition=dict(row.get("stop_condition") or {}),
        rollback=dict(row.get("rollback") or {}),
        measurement_spec=dict(row.get("measurement_spec") or {}),
        expires_at=row.get("expires_at"),
        owner_user_id=(
            int(row["owner_user_id"])
            if row.get("owner_user_id") is not None
            else None
        ),
        approved_by=(
            int(row["approved_by"]) if row.get("approved_by") is not None else None
        ),
        approved_at=row.get("approved_at"),
        approval_idempotency_key=row.get("approval_idempotency_key"),
        rejected_by=(
            int(row["rejected_by"]) if row.get("rejected_by") is not None else None
        ),
        rejected_at=row.get("rejected_at"),
        rejection_idempotency_key=row.get("rejection_idempotency_key"),
        rejection_reason=row.get("rejection_reason"),
        manager_label=row.get("manager_label"),
        preflight_json=dict(row.get("preflight_json") or {}),
        actuator_result=dict(row.get("actuator_result") or {}),
        external_operation_id=row.get("external_operation_id"),
        applied_at=row.get("applied_at"),
        measurement_due_at=row.get("measurement_due_at"),
        close_outcome=row.get("close_outcome"),
        last_error=row.get("last_error"),
        version=int(row.get("version") or 1),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        experiment_program_id=row.get("experiment_program_id"),
        experiment_candidate_id=row.get("experiment_candidate_id"),
    )


def _row_to_agent_action_measurement(
    row: dict[str, Any],
) -> AgentActionMeasurementRecord:
    return AgentActionMeasurementRecord(
        id=int(row["id"]),
        experiment_id=str(row["experiment_id"]),
        phase=str(row["phase"]),
        status=str(row["status"]),
        period_start=row.get("period_start"),
        period_end=row.get("period_end"),
        data_as_of=row.get("data_as_of"),
        idempotency_key=row.get("idempotency_key"),
        metrics=dict(row.get("metrics") or {}),
        causal_valid=bool(row.get("causal_valid")),
        observed_delta=_optional_float(row.get("observed_delta")),
        did_effect=_optional_float(row.get("did_effect")),
        forecast_low=_optional_float(row.get("forecast_low")),
        forecast_base=_optional_float(row.get("forecast_base")),
        forecast_high=_optional_float(row.get("forecast_high")),
        signed_bias=_optional_float(row.get("signed_bias")),
        mae=_optional_float(row.get("mae")),
        inside_forecast_range=(
            bool(row["inside_forecast_range"])
            if row.get("inside_forecast_range") is not None
            else None
        ),
        created_by=(
            int(row["created_by"]) if row.get("created_by") is not None else None
        ),
        created_at=row.get("created_at"),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _stored_to_message(message: StoredChatMessage) -> ChatMemoryMessage:
    return ChatMemoryMessage(
        seq=message.seq,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _load_message_by_metadata_key(
    chat_id: str,
    metadata_key: str,
    metadata_value: str,
    role: ChatRole,
) -> StoredChatMessage | None:
    with get_memory_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select seq, role, content, created_at, metadata
            from chat_messages
            where chat_id = %s
              and role = %s
              and metadata ->> %s = %s
            order by seq desc
            limit 1
            """,
            [chat_id, role.value, metadata_key, metadata_value],
        )
        row = cur.fetchone()
    return _row_to_stored_message(row) if row else None


def _is_duplicate_client_message_constraint(exc: psycopg.Error) -> bool:
    constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
    return constraint_name == "chat_messages_user_client_message_id_idx"
