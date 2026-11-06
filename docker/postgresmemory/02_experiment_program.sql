-- Fresh-clone bootstrap for the durable ExperimentProgram subsystem.
-- Runtime upgrades use the same DDL embedded in services/memory.py.
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
