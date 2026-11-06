from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from agi_ecom_app.experiment_program.models import (
    ExperimentCandidateSpec,
    ExperimentCandidateView,
    ExperimentDomainFrontierView,
    ExperimentEvidencePacket,
    ExperimentNegotiationMessageInput,
    ExperimentNegotiationMessageView,
    ExperimentProgramBundleView,
    ExperimentProgramCreate,
    ExperimentProgramEventView,
    ExperimentProgramView,
    ExperimentSnapshotView,
    TERMINAL_CANDIDATE_STATUSES,
)
from agi_ecom_app.experiment_program.repository import (
    ExperimentProgramBundleRows,
    ExperimentProgramStorageConflict,
    ExperimentProgramStorageNotFound,
    append_negotiation_message_atomic,
    create_program_bundle_atomic,
    get_program_bundle,
    list_negotiation_messages,
    list_program_events,
)


class ExperimentProgramError(RuntimeError):
    pass


class ExperimentProgramConflict(ExperimentProgramError):
    pass


class ExperimentProgramNotFound(ExperimentProgramError, LookupError):
    pass


class ExperimentProgramContractError(ExperimentProgramError, ValueError):
    pass


def create_experiment_program(
    request: ExperimentProgramCreate,
    *,
    idempotency_key: str,
    now: datetime | None = None,
) -> tuple[ExperimentProgramBundleView, bool]:
    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise ExperimentProgramContractError("Idempotency-Key cannot be empty")
    current_time = _aware_utc(now or datetime.now(timezone.utc))
    request_payload = request.model_dump(mode="json", by_alias=True)
    request_hash = canonical_contract_hash(request_payload)
    program_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"experiment-program:{normalized_key}",
        )
    )

    effective_packets = [
        _effective_evidence_packet(packet, now=current_time) for packet in request.evidence
    ]
    snapshot_payload = {
        "schemaVersion": "experiment-evidence-v1",
        "programId": program_id,
        "epoch": 1,
        "evaluatedAt": current_time.isoformat(),
        "generatorVersion": request.generator_version,
        "target": request.target,
        "evidence": [
            packet.model_dump(mode="json", by_alias=True) for packet in effective_packets
        ],
    }
    snapshot_hash = canonical_contract_hash(snapshot_payload)
    snapshot_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"experiment-snapshot:{program_id}:{snapshot_hash}",
        )
    )

    packets_by_metric: dict[str, list[ExperimentEvidencePacket]] = defaultdict(list)
    for packet in effective_packets:
        packets_by_metric[packet.metric].append(packet)

    prepared_candidates: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    candidate_id_by_key: dict[str, str] = {}
    spec_by_key: dict[str, ExperimentCandidateSpec] = {}
    for spec in request.candidates:
        candidate_key = _candidate_key(spec, generator_version=request.generator_version)
        if candidate_key in spec_by_key:
            raise ExperimentProgramContractError(
                f"duplicate candidateKey in finite frontier: {candidate_key}"
            )
        candidate_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"experiment-candidate:{program_id}:1:{candidate_key}",
            )
        )
        candidate_id_by_key[candidate_key] = candidate_id
        spec_by_key[candidate_key] = spec
        evidence_status, evidence_refs, candidate_requirements = _candidate_evidence(
            program_id=program_id,
            candidate_id=candidate_id,
            spec=spec,
            packets_by_metric=packets_by_metric,
        )
        candidate_status = "eligible" if evidence_status == "available" else "blocked_data"
        prepared_candidates.append(
            {
                "id": candidate_id,
                "program_id": program_id,
                "epoch": 1,
                "candidate_key": candidate_key,
                "domain": spec.domain,
                "family": spec.family,
                "operation": spec.operation,
                "target": spec.target,
                "parameters": spec.parameters,
                "hypothesis": spec.hypothesis,
                "primary_metric": spec.primary_metric,
                "guardrail_metrics": spec.guardrail_metrics,
                "required_metrics": spec.required_metrics,
                "evidence_refs": evidence_refs,
                "evidence_status": evidence_status,
                "causal_design": spec.causal_design,
                "estimated_cost": spec.estimated_cost,
                "risk": spec.risk,
                "reversibility": spec.reversibility,
                "stop_conditions": spec.stop_conditions,
                "status": candidate_status,
                "score": {},
            }
        )
        requirements.extend(candidate_requirements)

    dependencies = _prepare_dependencies(
        program_id=program_id,
        candidate_id_by_key=candidate_id_by_key,
        spec_by_key=spec_by_key,
    )
    conflicts = _prepare_conflicts(
        program_id=program_id,
        candidate_id_by_key=candidate_id_by_key,
        spec_by_key=spec_by_key,
    )

    candidates_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in prepared_candidates:
        candidates_by_domain[candidate["domain"]].append(candidate)

    frontiers: list[dict[str, Any]] = []
    domain_coverage: dict[str, Any] = {}
    for domain in request.domains:
        items = candidates_by_domain.get(domain, [])
        eligible_count = sum(item["status"] == "eligible" for item in items)
        terminal_count = sum(
            item["status"] in TERMINAL_CANDIDATE_STATUSES for item in items
        )
        if not items:
            frontier_status = "no_controllable_lever"
            reason_code = "candidate_catalog_empty"
        elif eligible_count:
            frontier_status = "ready"
            reason_code = "evidence_gate_passed"
        else:
            frontier_status = "blocked_data"
            reason_code = "required_evidence_unavailable"
        coverage = {
            "candidateCatalogComplete": True,
            "negotiationDone": False,
            "reasonCode": reason_code,
            "candidateKeys": [item["candidate_key"] for item in items],
            "eligibleCandidateKeys": [
                item["candidate_key"] for item in items if item["status"] == "eligible"
            ],
        }
        domain_coverage[domain] = coverage
        frontiers.append(
            {
                "program_id": program_id,
                "domain": domain,
                "epoch": 1,
                "generator_version": request.generator_version,
                "snapshot_hash": snapshot_hash,
                "status": frontier_status,
                "candidate_count": len(items),
                "eligible_count": eligible_count,
                "terminal_count": terminal_count,
                "coverage": coverage,
            }
        )

    packet_status_counts = Counter(packet.status for packet in effective_packets)
    eligible_total = sum(item["status"] == "eligible" for item in prepared_candidates)
    conflicting = packet_status_counts.get("conflicting", 0) > 0
    initial_status = "proposing" if eligible_total > 0 and not conflicting else "blocked_data"
    snapshot_status = (
        "conflicting"
        if conflicting
        else "ready"
        if all(packet.status == "available" for packet in effective_packets)
        else "blocked_data"
    )
    quality_summary = {
        "packetCount": len(effective_packets),
        "statusCounts": dict(sorted(packet_status_counts.items())),
        "candidateCount": len(prepared_candidates),
        "eligibleCandidateCount": eligible_total,
        "blockedCandidateCount": len(prepared_candidates) - eligible_total,
        "authority": "data_authority",
    }
    data_as_of_values = [
        packet.data_as_of for packet in effective_packets if packet.data_as_of is not None
    ]
    data_as_of = min(data_as_of_values) if data_as_of_values else None

    program = {
        "id": program_id,
        "idempotency_key": normalized_key,
        "request_hash": request_hash,
        "objective": request.objective,
        "mode": request.mode,
        "status": initial_status,
        "target": request.target,
        "domains": request.domains,
        "generator_version": request.generator_version,
        "snapshot_hash": snapshot_hash,
        "current_epoch": 1,
        "experiment_budget": request.experiment_budget,
        "domain_coverage": domain_coverage,
        "source_chat_id": request.source_chat_id,
        "source_run_id": request.source_run_id,
        "source_decision_id": request.source_decision_id,
        "error": {},
        "version": 1,
    }
    snapshot = {
        "id": snapshot_id,
        "program_id": program_id,
        "epoch": 1,
        "snapshot_hash": snapshot_hash,
        "status": snapshot_status,
        "data_as_of": data_as_of,
        "evidence_packets": snapshot_payload["evidence"],
        "quality_summary": quality_summary,
    }
    events = [
        {
            "event_id": f"{program_id}:created",
            "program_id": program_id,
            "epoch": 1,
            "event_type": "created",
            "payload": {
                "objective": request.objective,
                "mode": request.mode,
                "domains": request.domains,
                "generatorVersion": request.generator_version,
            },
        },
        {
            "event_id": f"{program_id}:snapshot:{snapshot_hash}",
            "program_id": program_id,
            "epoch": 1,
            "event_type": "snapshot_pinned",
            "payload": {
                "snapshotId": snapshot_id,
                "snapshotHash": snapshot_hash,
                "status": snapshot_status,
                "dataAsOf": data_as_of.isoformat() if data_as_of else None,
            },
        },
        {
            "event_id": f"{program_id}:evidence-audited:1",
            "program_id": program_id,
            "epoch": 1,
            "event_type": "evidence_audited",
            "payload": {
                **quality_summary,
                "programStatus": initial_status,
            },
        },
    ]

    try:
        rows, created = create_program_bundle_atomic(
            program=program,
            snapshot=snapshot,
            frontiers=frontiers,
            candidates=prepared_candidates,
            requirements=requirements,
            dependencies=dependencies,
            conflicts=conflicts,
            events=events,
        )
    except ExperimentProgramStorageConflict as exc:
        raise ExperimentProgramConflict(str(exc)) from None
    return _bundle_view(rows), created


def get_experiment_program_bundle(program_id: str) -> ExperimentProgramBundleView:
    rows = get_program_bundle(program_id)
    if rows is None:
        raise ExperimentProgramNotFound(program_id)
    return _bundle_view(rows)


def post_experiment_negotiation_message(
    program_id: str,
    request: ExperimentNegotiationMessageInput,
    *,
    idempotency_key: str,
) -> tuple[ExperimentNegotiationMessageView, bool]:
    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise ExperimentProgramContractError("Idempotency-Key cannot be empty")
    fingerprint_payload = request.model_dump(mode="json", by_alias=True)
    fingerprint = canonical_contract_hash(fingerprint_payload)
    message_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"experiment-message:{program_id}:{normalized_key}",
        )
    )
    try:
        stored = append_negotiation_message_atomic(
            program_id=program_id,
            message={
                "message_id": message_id,
                "round_id": request.round_id,
                "message_type": request.message_type,
                "source_role": request.source_role,
                "target_role": request.target_role,
                "correlation_id": request.correlation_id,
                "idempotency_key": normalized_key,
                "fingerprint": fingerprint,
                "payload": request.payload,
                "evidence_refs": request.evidence_refs,
            },
        )
    except ExperimentProgramStorageNotFound:
        raise ExperimentProgramNotFound(program_id) from None
    except ExperimentProgramStorageConflict as exc:
        raise ExperimentProgramConflict(str(exc)) from None
    return ExperimentNegotiationMessageView.model_validate(stored.row), stored.created


def get_experiment_negotiation_messages(
    program_id: str,
    *,
    after_id: int = 0,
    limit: int = 1000,
) -> list[ExperimentNegotiationMessageView]:
    _require_program(program_id)
    return [
        ExperimentNegotiationMessageView.model_validate(row)
        for row in list_negotiation_messages(
            program_id,
            after_id=after_id,
            limit=limit,
        )
    ]


def get_experiment_program_events(
    program_id: str,
    *,
    after_id: int = 0,
    limit: int = 1000,
) -> list[ExperimentProgramEventView]:
    _require_program(program_id)
    return [
        ExperimentProgramEventView.model_validate(row)
        for row in list_program_events(program_id, after_id=after_id, limit=limit)
    ]


def canonical_contract_hash(payload: Any) -> str:
    _reject_non_finite(payload)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_key(spec: ExperimentCandidateSpec, *, generator_version: str) -> str:
    if spec.candidate_key is not None:
        normalized = spec.candidate_key.strip()
        if not normalized:
            raise ExperimentProgramContractError("candidateKey cannot be empty")
        return normalized
    fingerprint = canonical_contract_hash(
        {
            "domain": spec.domain,
            "family": spec.family,
            "operation": spec.operation,
            "target": spec.target,
            "parameters": spec.parameters,
            "generatorVersion": generator_version,
        }
    )
    return f"{spec.domain}:{spec.operation}:{fingerprint[:24]}"


def _effective_evidence_packet(
    packet: ExperimentEvidencePacket,
    *,
    now: datetime,
) -> ExperimentEvidencePacket:
    if (
        packet.status == "available"
        and packet.max_age_seconds is not None
        and packet.data_as_of is not None
        and (now - _aware_utc(packet.data_as_of)).total_seconds() > packet.max_age_seconds
    ):
        return packet.model_copy(update={"status": "stale"})
    return packet


def _candidate_evidence(
    *,
    program_id: str,
    candidate_id: str,
    spec: ExperimentCandidateSpec,
    packets_by_metric: dict[str, list[ExperimentEvidencePacket]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    if not spec.required_metrics:
        return "available", [], []
    statuses: list[str] = []
    refs: list[str] = []
    requirements: list[dict[str, Any]] = []
    for metric in spec.required_metrics:
        packets = packets_by_metric.get(metric, [])
        selected, status = _select_evidence_packet(packets)
        statuses.append(status)
        evidence_ref = selected.evidence_ref if selected else None
        if evidence_ref and evidence_ref not in refs:
            refs.append(evidence_ref)
        requirements.append(
            {
                "program_id": program_id,
                "candidate_id": candidate_id,
                "domain": spec.domain,
                "metric": metric,
                "status": status,
                "evidence_ref": evidence_ref,
                "source_table": selected.source_table if selected else None,
                "grain": selected.grain if selected else None,
                "data_as_of": selected.data_as_of if selected else None,
                "max_age_seconds": selected.max_age_seconds if selected else None,
                "details": selected.details if selected else {"reason": "metric_missing"},
            }
        )
    priority = (
        "conflicting",
        "causally_insufficient",
        "stale",
        "missing",
        "available",
    )
    evidence_status = next(status for status in priority if status in statuses)
    return evidence_status, refs, requirements


def _select_evidence_packet(
    packets: Iterable[ExperimentEvidencePacket],
) -> tuple[ExperimentEvidencePacket | None, str]:
    items = list(packets)
    if not items:
        return None, "missing"
    priority = {
        "conflicting": 0,
        "causally_insufficient": 1,
        "available": 2,
        "stale": 3,
        "missing": 4,
    }
    selected = min(items, key=lambda item: priority[item.status])
    # A definition conflict is a veto even if another source claims availability.
    if any(item.status == "conflicting" for item in items):
        selected = next(item for item in items if item.status == "conflicting")
        return selected, "conflicting"
    available = [item for item in items if item.status == "available"]
    if available:
        selected = max(
            available,
            key=lambda item: item.data_as_of or datetime.min.replace(tzinfo=timezone.utc),
        )
        return selected, "available"
    return selected, selected.status


def _prepare_dependencies(
    *,
    program_id: str,
    candidate_id_by_key: dict[str, str],
    spec_by_key: dict[str, ExperimentCandidateSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_key, spec in spec_by_key.items():
        for dependency_key in spec.dependencies:
            dependency_id = candidate_id_by_key.get(dependency_key)
            if dependency_id is None:
                raise ExperimentProgramContractError(
                    f"candidate {candidate_key} depends on unknown {dependency_key}"
                )
            rows.append(
                {
                    "program_id": program_id,
                    "candidate_id": candidate_id_by_key[candidate_key],
                    "depends_on_candidate_id": dependency_id,
                    "dependency_type": "requires",
                    "details": {"candidateKey": dependency_key},
                }
            )
    return rows


def _prepare_conflicts(
    *,
    program_id: str,
    candidate_id_by_key: dict[str, str],
    spec_by_key: dict[str, ExperimentCandidateSpec],
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(left_key: str, right_key: str, interference_key: str, source: str) -> None:
        if left_key == right_key:
            return
        if right_key not in candidate_id_by_key:
            raise ExperimentProgramContractError(
                f"candidate {left_key} conflicts with unknown {right_key}"
            )
        left_id, right_id = sorted(
            (candidate_id_by_key[left_key], candidate_id_by_key[right_key])
        )
        key = (left_id, right_id, interference_key)
        rows_by_key[key] = {
            "program_id": program_id,
            "left_candidate_id": left_id,
            "right_candidate_id": right_id,
            "interference_key": interference_key,
            "conflict_type": "causal_interference",
            "details": {"source": source},
        }

    for candidate_key, spec in spec_by_key.items():
        interference_keys = spec.interference_keys or _derived_interference_keys(spec)
        for conflict_key in spec.conflicts:
            for interference_key in interference_keys or ["explicit"]:
                add(candidate_key, conflict_key, interference_key, "explicit")

    keys = sorted(spec_by_key)
    for index, left_key in enumerate(keys):
        left = spec_by_key[left_key]
        if left.domain not in {"pricing", "advertising"}:
            continue
        left_interference = set(
            left.interference_keys or _derived_interference_keys(left)
        )
        for right_key in keys[index + 1 :]:
            right = spec_by_key[right_key]
            if {left.domain, right.domain} != {"pricing", "advertising"}:
                continue
            shared = left_interference.intersection(
                right.interference_keys or _derived_interference_keys(right)
            )
            for interference_key in sorted(shared):
                add(left_key, right_key, interference_key, "automatic_price_ads_gate")
    return list(rows_by_key.values())


def _derived_interference_keys(spec: ExperimentCandidateSpec) -> list[str]:
    for key in ("nmId", "nm_id", "skuId", "sku_id", "sku"):
        value = spec.target.get(key)
        if value not in (None, ""):
            return [f"sku:{value}"]
    for key in ("advertId", "advert_id", "campaignId", "campaign_id"):
        value = spec.target.get(key)
        if value not in (None, ""):
            return [f"campaign:{value}"]
    return []


def _bundle_view(rows: ExperimentProgramBundleRows) -> ExperimentProgramBundleView:
    program_fields = set(ExperimentProgramView.model_fields)
    return ExperimentProgramBundleView(
        program=ExperimentProgramView.model_validate(
            {key: value for key, value in rows.program.items() if key in program_fields}
        ),
        snapshot=ExperimentSnapshotView.model_validate(rows.snapshot),
        frontiers=[
            ExperimentDomainFrontierView.model_validate(row) for row in rows.frontiers
        ],
        candidates=[ExperimentCandidateView.model_validate(row) for row in rows.candidates],
    )


def _require_program(program_id: str) -> None:
    if get_program_bundle(program_id) is None:
        raise ExperimentProgramNotFound(program_id)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ExperimentProgramContractError("timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentProgramContractError("contracts cannot contain NaN or Infinity")
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


__all__ = [
    "ExperimentProgramConflict",
    "ExperimentProgramContractError",
    "ExperimentProgramError",
    "ExperimentProgramNotFound",
    "canonical_contract_hash",
    "create_experiment_program",
    "get_experiment_negotiation_messages",
    "get_experiment_program_bundle",
    "get_experiment_program_events",
    "post_experiment_negotiation_message",
]
