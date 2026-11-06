from __future__ import annotations

import argparse
import json

from agi_ecom_app.experiment_program.models import (
    ExperimentNegotiationMessageInput,
    ExperimentProgramCreate,
)
from agi_ecom_app.experiment_program.service import (
    create_experiment_program,
    post_experiment_negotiation_message,
)
from agi_ecom_app.services.memory import get_decision_simulation_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a standalone pricing ExperimentProgram from an existing "
            "durable pricing simulation input contract."
        )
    )
    parser.add_argument("--simulation-run-id", required=True)
    parser.add_argument("--idempotency-key")
    args = parser.parse_args()

    source_run = get_decision_simulation_run(args.simulation_run_id)
    if source_run is None:
        raise SystemExit(f"simulation run not found: {args.simulation_run_id}")
    if source_run.model_version != "pricing-elasticity-bootstrap-v1":
        raise SystemExit("source run is not a pricing simulation")
    raw_request = source_run.evidence_snapshot.get("simulationRequest")
    raw_candidates = source_run.evidence_snapshot.get("simulationCandidates")
    if not isinstance(raw_request, dict) or not isinstance(raw_candidates, list):
        raise SystemExit("source run has no durable pricing request")

    sku_id = str(raw_request["skuId"])
    candidates = [
        {
            "candidateKey": str(candidate["candidateId"]),
            "domain": "pricing",
            "family": "pricing",
            "operation": str(candidate["operation"]),
            "target": {"nmId": sku_id},
            "parameters": {"priceRub": candidate["priceRub"]},
            "hypothesis": "Evaluate price under the pinned empirical contract",
            "primaryMetric": "incremental_contribution_profit",
            "requiredMetrics": ["pricing_simulation_request"],
            "causalDesign": {
                "minimumDesign": "switchback_or_matched_sku",
                "simultaneousAdvertisingChange": False,
            },
            "reversibility": {"required": True},
            "stopConditions": [{"code": "guardrail_breach"}],
        }
        for candidate in raw_candidates
    ]
    evidence_ref = (
        f"decision_simulation_runs:{source_run.id}:pricing-input:"
        f"{source_run.evidence_hash}"
    )
    request = ExperimentProgramCreate.model_validate(
        {
            "objective": source_run.objective,
            "mode": "shadow",
            "target": {"nmId": sku_id},
            "domains": ["pricing"],
            "generatorVersion": "pricing-forward-launch-v1",
            "evidence": [
                {
                    "metric": "pricing_simulation_request",
                    "value": {"simulationRequest": raw_request},
                    "status": "available",
                    "evidenceRef": evidence_ref,
                    "sourceTable": "decision_simulation_runs",
                    "grain": "simulation_run",
                    "dataAsOf": source_run.created_at,
                    "lineage": [
                        "decision_simulation_runs.evidence_snapshot.simulationRequest"
                    ],
                }
            ],
            "candidates": candidates,
            "experimentBudget": {
                "sampleCount": 500,
                "batchSize": 50,
                "candidateCount": len(candidates),
            },
        }
    )
    program, created = create_experiment_program(
        request,
        idempotency_key=(
            args.idempotency_key
            or f"pricing-forward-launch:{source_run.id}"
        ),
    )
    _message, message_created = post_experiment_negotiation_message(
        program.program.id,
        ExperimentNegotiationMessageInput(
            messageType="domain_done",
            sourceRole="pricing_agent",
            payload={"domain": "pricing", "coverageStatus": "ready"},
        ),
        idempotency_key="pricing-domain-done",
    )
    print(
        json.dumps(
            {
                "programId": program.program.id,
                "created": created,
                "domainDoneCreated": message_created,
                "statusBeforeDomainDone": program.program.status,
                "candidateCount": len(program.candidates),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
