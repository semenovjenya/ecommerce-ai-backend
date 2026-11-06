from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agi_ecom_app.decision_simulation.builtin_plugins import (
    ensure_builtin_simulation_plugins_registered,
)
from agi_ecom_app.decision_simulation.plugins import get_decision_simulation_plugin
from agi_ecom_app.experiment_program.models import ExperimentProgramCreate
from agi_ecom_app.experiment_program.repository import (
    link_simulation_run,
    sync_linked_simulation_run,
)
from agi_ecom_app.experiment_program.service import create_experiment_program
from agi_ecom_app.services.memory import (
    DecisionSimulationCandidateRecord,
    DecisionSimulationRunRecord,
    get_decision_simulation_run,
    list_decision_simulation_candidates,
)


_PROGRAM_DOMAINS = ["pricing", "advertising", "supply", "sales", "finance"]


def attach_experiment_program_to_simulation_run(
    simulation_run_id: str,
) -> str:
    """Backfill/attach a finite frontier around an existing durable run."""

    run = get_decision_simulation_run(simulation_run_id)
    if run is None:
        raise LookupError(simulation_run_id)
    candidates = list_decision_simulation_candidates(run.id)
    if not candidates:
        raise ValueError("simulation run has no durable candidates to attach")
    if run.experiment_program_id:
        return run.experiment_program_id

    ensure_builtin_simulation_plugins_registered()
    plugin = get_decision_simulation_plugin(run.model_version)
    domain = plugin.family
    if domain not in {"pricing", "advertising"}:
        raise ValueError(f"unsupported experiment-program simulation family: {domain}")

    data_as_of = run.created_at or datetime.now(timezone.utc)
    evidence_ref = (
        f"decision_simulation_runs:{run.id}:snapshot:{run.evidence_hash}"
    )
    candidate_payloads = [
        _candidate_contract(candidate, domain=domain, operation=plugin.operation)
        for candidate in candidates
        if candidate.status not in {"rejected", "needs_data", "failed"}
    ]
    if not candidate_payloads:
        raise ValueError("simulation run has no evidence-eligible candidates to attach")
    target = _common_target(candidate_payloads)
    request = ExperimentProgramCreate.model_validate(
        {
            "objective": run.objective,
            "mode": run.mode,
            "target": target,
            "domains": _PROGRAM_DOMAINS,
            "generatorVersion": f"simulation-adapter-v1:{run.model_version}",
            "evidence": [
                {
                    "metric": "simulation_input_contract",
                    "value": {
                        "simulationRunId": run.id,
                        "snapshotHash": run.evidence_hash,
                        "modelVersion": run.model_version,
                    },
                    "status": "available",
                    "evidenceRef": evidence_ref,
                    "sourceTable": "decision_simulation_runs",
                    "grain": "simulation_run",
                    "dataAsOf": data_as_of,
                    "lineage": [
                        "decision_simulation_runs.evidence_snapshot",
                        "decision_simulation_candidates.candidate_payload",
                    ],
                    "details": {
                        "immutableSnapshot": True,
                        "evidenceHash": run.evidence_hash,
                    },
                }
            ],
            "candidates": candidate_payloads,
            "experimentBudget": {
                "sampleCount": run.sample_count,
                "candidateCount": len(candidate_payloads),
                "totalSamples": run.total_samples,
            },
            "sourceChatId": run.chat_id,
            "sourceRunId": run.run_id,
            "sourceDecisionId": run.decision_id,
        }
    )
    bundle, _ = create_experiment_program(
        request,
        idempotency_key=f"simulation-program:{run.id}",
        now=data_as_of,
    )
    program_candidate_by_key = {
        candidate.candidate_key: candidate.id for candidate in bundle.candidates
    }
    links = {
        candidate.candidate_id: program_candidate_by_key[candidate.candidate_id]
        for candidate in candidates
        if candidate.candidate_id in program_candidate_by_key
    }
    linked_count = link_simulation_run(
        program_id=bundle.program.id,
        simulation_run_id=run.id,
        candidate_links=links,
    )
    if linked_count != len(candidate_payloads):
        raise RuntimeError("not every program candidate was linked to the simulation run")
    return bundle.program.id


def maybe_attach_experiment_program(run: DecisionSimulationRunRecord) -> str | None:
    from agi_ecom_app.config import get_settings

    if not get_settings().experiment_program_auto_attach:
        return None
    return attach_experiment_program_to_simulation_run(run.id)


def sync_experiment_program_from_simulation_run(
    simulation_run_id: str,
) -> str | None:
    return sync_linked_simulation_run(simulation_run_id)


def _candidate_contract(
    candidate: DecisionSimulationCandidateRecord,
    *,
    domain: str,
    operation: str,
) -> dict[str, Any]:
    payload = dict(candidate.candidate_payload or {})
    target = _target_from_payload(payload)
    causal_design = (
        {
            "minimumDesign": "switchback_or_matched_sku",
            "simultaneousAdvertisingChange": False,
        }
        if domain == "pricing"
        else {
            "minimumDesign": "randomized_holdout_or_matched_control",
            "simultaneousPriceChange": False,
        }
    )
    return {
        "candidateKey": candidate.candidate_id,
        "domain": domain,
        "family": domain,
        "operation": operation,
        "target": target,
        "parameters": payload,
        "hypothesis": f"Evaluate {domain}.{operation} under the pinned simulation snapshot",
        "primaryMetric": "incremental_contribution_profit",
        "requiredMetrics": ["simulation_input_contract"],
        "causalDesign": causal_design,
        "estimatedCost": {"simulationSamples": candidate.completed_samples},
        "risk": {"source": "decision_simulation_candidates.score_breakdown"},
        "reversibility": {"required": True},
        "stopConditions": [
            {"code": "guardrail_breach"},
            {"code": "causal_power_insufficient"},
        ],
    }


def _target_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_target = payload.get("target")
    target = dict(raw_target) if isinstance(raw_target, dict) else {}
    aliases = {
        "nmId": ("nmId", "nm_id", "skuId", "sku_id"),
        "advertId": ("advertId", "advert_id", "campaignId", "campaign_id"),
    }
    for public_key, keys in aliases.items():
        if public_key in target:
            continue
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                target[public_key] = value
                break
    return target


def _common_target(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [candidate.get("target") or {} for candidate in candidates]
    if not targets:
        return {}
    common: dict[str, Any] = {}
    for key, value in targets[0].items():
        if all(target.get(key) == value for target in targets[1:]):
            common[key] = value
    return common


__all__ = [
    "attach_experiment_program_to_simulation_run",
    "maybe_attach_experiment_program",
    "sync_experiment_program_from_simulation_run",
]
