# Manager Decision v1

`ManagerDecision` is the only source of executable recommendations for an
explicit multi-action portfolio request when `MANAGER_DECISION_ENABLED=true`
and `AGENT_WORKFLOW_VERSION=v2_optimized`. Ordinary one-action yes/no decisions
use evidence synthesis and never get padded to the manager contract's 3–7
actions. Lookup, report and diagnostic queries remain facts-only.

For portfolio decisions, synthesis loads a read-only action snapshot and runs
the deterministic candidate builder before the hard gate. Agent/LLM output may
describe hypotheses, but any `manager_action_candidates` supplied by synthesis
is discarded; only source-derived candidates enter the executable contract.
The current builder covers zero-stock advertising pauses and replenishment-task
previews. A candidate with missing economics remains visible in
`rejectedCandidates` with exact target/change and can never reach Action Board.

The legacy executable pricing path remains blocked. A separate pricing
simulation source now reads `agent_marts.v_agent_price_sku_daily` and may run
only with `DECISION_SIMULATION_PRICING_MODE=shadow`. It pins the current seller
price, completed price/order observations, empirical elasticity and
order-to-realization bootstrap distributions, payout-based unit economics,
stock and freshness in one immutable snapshot. Missing/stale evidence, a price
outside the ±10% safety bound, non-negative or unstable elasticity, or
insufficient history fails closed. The numerical winner is observable but
never creates an experiment or a WB write because the observational model is
not causal-valid and complete source watermarks are not yet available.

Implementation baseline: Git `c1832be4473a4e856c7924973b801e8eb1c21040` on
`main`. The pre-existing uncommitted `v2_optimized` worktree was kept in place;
the manager implementation does not reset or roll back those changes.

## Safe rollout

All manager and WB write flags default to `false`. Use the following order:

1. Collect at least 40 external, human-labelled historical cases.
2. Run the deterministic release gate:

   ```bash
   python3 -m agi_ecom_app.manager_evaluation /path/to/manager-history.json
   ```

   Exit code `0` means every release criterion passed. Any missing denominator,
   missing label provenance or threshold failure returns a non-zero exit code.
3. Enable `MANAGER_DECISION_ENABLED` for read-only decisions.
4. Configure a live guardrail provider and measurement provider for each action
   family. Without them, execution and causal readout deliberately stay blocked
   or `waiting_for_data`.
5. Validate writes in WB sandbox, then enable `WB_ADS_WRITE_ENABLED` and/or
   `WB_PRICE_WRITE_ENABLED` separately.
6. Enable `MANAGER_DAILY_ENABLED` only after chat decisions are stable.

For the advertising simulation pilot, first use
`DECISION_SIMULATION_MODE=shadow`. Move to `enforced` only after replayed SSE
events, deterministic seeds/winners and critic verdicts have been checked. In
enforced mode a decision is persisted as `evaluating`, contains no approvable
experiments, and transitions to `ready_for_approval` only after the numerical
run and critic both allow it.

Pricing has an independent flag: set
`DECISION_SIMULATION_PRICING_MODE=shadow` to evaluate 3–7 explicit price
candidates over a fixed seven-day horizon. Pricing v1 rejects `enforced`; a
completed run projects the ManagerDecision to `needs_data` and keeps zero
approvable experiments.

The simulation worker and reconciler use separate RQ queues. The reconciler
repairs missing delivery, expired heartbeats, durable batch-event gaps and a
terminal run whose ManagerDecision projection was interrupted. API SSE and
chat-run WebSockets use shared PostgreSQL `LISTEN/NOTIFY` listeners with durable
replay; neither path polls PostgreSQL every two seconds. The general agent
worker also keeps a rolling horizon of reconciliation jobs so a question left
`running` by a killed work horse is atomically timed out and its run advances.

Never enable production writes by supplying only WB tokens. Approval, exact
external-state preflight, a fresh live guardrail snapshot and read-after-write
verification are independent requirements.

## API

- `POST /v1/auth/login`
- `GET /v1/manager/decisions/today`
- `GET /v1/manager/decisions/{id}`
- `POST /v1/manager/pricing/simulations` (manager/admin, `Idempotency-Key`)
- `GET /v1/manager/simulations/{run_id}`
- `GET /v1/manager/simulations/{run_id}/candidates`
- `GET /v1/manager/simulations/{run_id}/events`
- `GET /v1/manager/simulations/{run_id}/events/stream` (SSE replay via
  `Last-Event-ID` plus PostgreSQL `LISTEN/NOTIFY`)
- `POST /v1/manager/simulations/{run_id}/retry` (admin only)
- `GET /v1/manager/actions/{id}`
- `POST /v1/manager/actions/{id}/approve`
- `POST /v1/manager/actions/{id}/reject`
- `POST /v1/manager/actions/{id}/complete` for a confirmed human task; supply
  measurement starts from `actualAt`, the actual delivery time.

Mutation requests require a short-lived bearer token for a `manager` or `admin`
and an `Idempotency-Key` header. WB tokens and write switches are environment
secrets and are never persisted in the manager tables.

## Extension contracts

- Register execution-preflight facts with
  `register_manager_live_guardrail_provider(execution_mode, provider)`. The
  provider supplies current values and an observation timestamp; the lifecycle
  evaluates typed operators and thresholds itself.
- Register measurement facts with
  `register_manager_measurement_provider(action_family, provider)`. Missing or
  lagging data is saved as `waiting_for_data`; it is never converted to zero.
- Effect-range calibration is only proposed after 20 distinct causal-valid
  actions in one family. Proposals are versioned, never auto-applied and require
  manual approval.

The eight bundled run fixtures are regression cases, not substitutes for the 40
real manager-labelled release cases.
