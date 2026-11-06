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
