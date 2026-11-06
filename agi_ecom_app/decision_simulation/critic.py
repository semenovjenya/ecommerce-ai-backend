from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from agents import (
    Agent,
    AgentOutputSchema,
    GuardrailFunctionOutput,
    ModelSettings,
    Runner,
    output_guardrail,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agi_ecom_app.llm_output_parsing import coerce_model_from_json_text
from agi_ecom_app.llm_runtime import (
    LLMRuntimeSpec,
    build_agent_model,
    get_active_llm_runtime_spec,
)
from agi_ecom_app.main_agent import build_run_hooks
from agi_ecom_app.schemas import EcomRunContext, QueryIntent


DecisionCriticMode = Literal["shadow", "enforced"]
DecisionCriticVerdict = Literal["approved", "blocked", "recalculate"]
DecisionCriticEffectiveAction = Literal["record_only", "continue", "block", "recalculate"]
DecisionCriticIssueCode = Literal[
    "stale_evidence",
    "missing_required_metric",
    "candidate_conflict",
    "unsupported_causal_claim",
    "unsafe_execution",
    "missing_monitoring_specification",
    "llm_critic_unavailable",
]

_LLM_ALLOWED_ISSUE_CODES = frozenset(
    {
        "stale_evidence",
        "missing_required_metric",
        "candidate_conflict",
        "unsupported_causal_claim",
        "unsafe_execution",
        "missing_monitoring_specification",
    }
)
_RECALCULABLE_ISSUE_CODES = frozenset({"stale_evidence", "missing_required_metric"})
_INTERNAL_UNAVAILABLE_CODE = "llm_critic_unavailable"
_CANDIDATE_ID_KEYS = ("candidate_id", "candidateId")
_EVIDENCE_KEY_PATTERN = re.compile(r"[^a-z0-9]+")

_DECISION_CRITIC_INSTRUCTIONS = """
Ты — независимый критик численного Decision Simulation Engine.

Тебе передаются неизменяемые результаты симуляции: objective, modelCard,
snapshotFreshness, candidateAggregates, top3, leaderDelta, hardConstraints и
evidenceRefs. Проверь только качество данных, причинные допущения, безопасность
исполнения и измеримость результата.

Жёсткие ограничения:
- не пересчитывай и не изменяй числа, score, rank, top3 или winner;
- не предлагай другого победителя и не добавляй поля winner/ranking/score;
- используй только candidate IDs и evidence refs, присутствующие во входе;
- verdict=approved требует пустой issues;
- verdict=blocked требует хотя бы один подтверждённый issue;
- verdict=recalculate допустим только для stale_evidence или
  missing_required_metric;
- каждый issue обязан содержать непустые evidenceRefs и affectedCandidates.

Допустимые issue code:
stale_evidence, missing_required_metric, candidate_conflict,
unsupported_causal_claim, unsafe_execution, missing_monitoring_specification.

Верни только structured output по заданной схеме. Поле
llm_critic_unavailable зарезервировано системой и не может возвращаться моделью.
""".strip()


class _CriticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DecisionCriticIssue(_CriticModel):
    code: DecisionCriticIssueCode
    message: str = Field(..., min_length=1, max_length=2_000)
    evidence_refs: list[str] = Field(default_factory=list, alias="evidenceRefs", max_length=20)
    affected_candidates: list[str] = Field(
        default_factory=list,
        alias="affectedCandidates",
        max_length=7,
    )

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("critic issue message cannot be blank")
        return normalized

    @field_validator("evidence_refs", "affected_candidates")
    @classmethod
    def normalize_identifiers(cls, values: list[str]) -> list[str]:
        normalized = [str(value).strip() for value in values if str(value).strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_grounding_for_llm_issue(self) -> "DecisionCriticIssue":
        if self.code == _INTERNAL_UNAVAILABLE_CODE:
            return self
        if not self.evidence_refs:
            raise ValueError("critic issue requires at least one evidence ref")
        if not self.affected_candidates:
            raise ValueError("critic issue requires at least one affected candidate")
        return self


class DecisionCriticResult(_CriticModel):
    """LLM-owned assessment only; ranking and winner fields are intentionally absent."""

    verdict: DecisionCriticVerdict
    issues: list[DecisionCriticIssue] = Field(default_factory=list, max_length=20)
    summary: str = Field(..., min_length=1, max_length=4_000)

    @field_validator("summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("critic summary cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_verdict_contract(self) -> "DecisionCriticResult":
        if self.verdict == "approved" and self.issues:
            raise ValueError("approved critic result cannot contain issues")
        if self.verdict != "approved" and not self.issues:
            raise ValueError(f"{self.verdict} critic result requires at least one issue")
        if self.verdict == "recalculate":
            invalid = [
                issue.code
                for issue in self.issues
                if issue.code not in _RECALCULABLE_ISSUE_CODES
            ]
            if invalid:
                raise ValueError(
                    "recalculate is allowed only for stale/missing evidence issues: "
                    + ", ".join(invalid)
                )
        return self


class DecisionCriticInput(_CriticModel):
    objective: Literal["incremental_contribution_profit"] = (
        "incremental_contribution_profit"
    )
    model_card: dict[str, Any] = Field(..., alias="modelCard", min_length=1)
    snapshot_freshness: dict[str, Any] = Field(
        ...,
        alias="snapshotFreshness",
        min_length=1,
    )
    candidate_aggregates: list[dict[str, Any]] = Field(
        ...,
        alias="candidateAggregates",
        min_length=3,
        max_length=7,
    )
    top_three: list[str] = Field(..., alias="top3", min_length=1, max_length=3)
    leader_delta: dict[str, Any] = Field(..., alias="leaderDelta", min_length=1)
    hard_constraints: list[dict[str, Any] | str] = Field(
        default_factory=list,
        alias="hardConstraints",
    )
    evidence_refs: list[str] = Field(default_factory=list, alias="evidenceRefs")
    recalculation_attempt: int = Field(default=0, ge=0, le=1, alias="recalculationAttempt")

    @field_validator("model_card", "snapshot_freshness", "leader_delta", mode="before")
    @classmethod
    def normalize_mapping(cls, value: Any) -> dict[str, Any]:
        return _as_plain_mapping(value)

    @field_validator("candidate_aggregates", mode="before")
    @classmethod
    def normalize_candidates(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("candidateAggregates must be a sequence")
        return [_as_plain_mapping(item) for item in value]

    @field_validator("hard_constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> list[dict[str, Any] | str]:
        if value is None:
            return []
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("hardConstraints must be a sequence")
        normalized: list[dict[str, Any] | str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            else:
                normalized.append(_as_plain_mapping(item))
        return normalized

    @field_validator("top_three", mode="before")
    @classmethod
    def normalize_top_three(cls, value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("top3 must be a sequence")
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                candidate_id = item
            else:
                candidate_id = _candidate_id(_as_plain_mapping(item))
            candidate_id = str(candidate_id).strip()
            if not candidate_id:
                raise ValueError("top3 candidate id cannot be blank")
            normalized.append(candidate_id)
        return normalized

    @field_validator("evidence_refs")
    @classmethod
    def normalize_evidence_refs(cls, values: list[str]) -> list[str]:
        normalized = [str(value).strip() for value in values if str(value).strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_ranking_references(self) -> "DecisionCriticInput":
        candidate_ids = self.candidate_ids
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidateAggregates contains duplicate candidate IDs")
        if len(self.top_three) != len(set(self.top_three)):
            raise ValueError("top3 contains duplicate candidate IDs")
        unknown = set(self.top_three) - set(candidate_ids)
        if unknown:
            raise ValueError(f"top3 references unknown candidates: {sorted(unknown)}")

        leader_id = _optional_id(
            self.leader_delta,
            "leader_candidate_id",
            "leaderCandidateId",
            "winner_candidate_id",
            "winnerCandidateId",
        )
        if leader_id and leader_id != self.top_three[0]:
            raise ValueError("leaderDelta leader must match top3[0]")
        runner_up_id = _optional_id(
            self.leader_delta,
            "runner_up_candidate_id",
            "runnerUpCandidateId",
        )
        if runner_up_id and (len(self.top_three) < 2 or runner_up_id != self.top_three[1]):
            raise ValueError("leaderDelta runner-up must match top3[1]")
        return self

    @property
    def candidate_ids(self) -> list[str]:
        return [_candidate_id(candidate) for candidate in self.candidate_aggregates]

    @property
    def known_evidence_refs(self) -> frozenset[str]:
        refs = set(self.evidence_refs)
        _collect_evidence_refs(self.model_card, refs)
        _collect_evidence_refs(self.snapshot_freshness, refs)
        _collect_evidence_refs(self.candidate_aggregates, refs)
        _collect_evidence_refs(self.leader_delta, refs)
        _collect_evidence_refs(self.hard_constraints, refs)
        return frozenset(refs)

    @property
    def ranking_fingerprint(self) -> str:
        payload = {
            "candidateAggregates": self.candidate_aggregates,
            "top3": self.top_three,
            "leaderDelta": self.leader_delta,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DecisionCriticExecution(_CriticModel):
    mode: DecisionCriticMode
    result: DecisionCriticResult
    effective_action: DecisionCriticEffectiveAction = Field(..., alias="effectiveAction")
    llm_available: bool = Field(..., alias="llmAvailable")
    blocks_decision: bool = Field(..., alias="blocksDecision")
    ranking_fingerprint: str = Field(..., alias="rankingFingerprint", min_length=64, max_length=64)
    debug: dict[str, Any] = Field(default_factory=dict)


@output_guardrail
def decision_critic_output_guardrail(context, agent, output: Any) -> GuardrailFunctionOutput:
    del context, agent
    violations = [] if isinstance(output, DecisionCriticResult) else [
        "Final output is not DecisionCriticResult."
    ]
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=bool(violations),
    )


def _dynamic_instructions(context_wrapper, agent) -> str:
    del context_wrapper, agent
    return _DECISION_CRITIC_INSTRUCTIONS


def build_decision_critic_agent(
    runtime_spec: LLMRuntimeSpec | None = None,
) -> Agent[EcomRunContext]:
    return _build_decision_critic_agent(runtime_spec or get_active_llm_runtime_spec())


@lru_cache(maxsize=8)
def _build_decision_critic_agent(runtime_spec: LLMRuntimeSpec) -> Agent[EcomRunContext]:
    return Agent(
        name="Decision Simulation Critic",
        instructions=_dynamic_instructions,
        model=build_agent_model(runtime_spec),
        model_settings=ModelSettings(tool_choice="none"),
        tools=[],
        output_type=(
            AgentOutputSchema(DecisionCriticResult, strict_json_schema=False)
            if runtime_spec.supports_agent_structured_outputs
            else None
        ),
        output_guardrails=(
            [decision_critic_output_guardrail]
            if runtime_spec.supports_agent_structured_outputs
            else []
        ),
    )


build_decision_critic_agent.cache_clear = _build_decision_critic_agent.cache_clear  # type: ignore[attr-defined]


def run_decision_critic(
    *,
    critic_input: DecisionCriticInput,
    mode: DecisionCriticMode,
    today: date | None = None,
    max_turns: int = 3,
) -> DecisionCriticExecution:
    """Run the read-only LLM critic and convert failures into an explicit policy result."""

    try:
        runtime_spec = get_active_llm_runtime_spec(require_configured=False)
    except Exception as exc:
        return _unavailable_execution(
            critic_input=critic_input,
            mode=mode,
            reason=type(exc).__name__,
            error=str(exc),
        )

    if runtime_spec.missing_env_vars:
        return _unavailable_execution(
            critic_input=critic_input,
            mode=mode,
            reason="missing_llm_configuration",
            error=", ".join(runtime_spec.missing_env_vars),
            runtime_spec=runtime_spec,
        )

    user_query, intent = _critic_domain_route(critic_input.model_card)
    context = EcomRunContext(
        user_query=user_query,
        intent=intent,
        today=today or date.today(),
        db_target="analytics",
        chat_id=f"decision-simulation-critic-{uuid4()}",
        evidence_budget=0,
    )
    payload = {
        "criticInput": critic_input.model_dump(mode="json", by_alias=True),
        "rankingIntegrity": {
            "fingerprint": critic_input.ranking_fingerprint,
            "immutable": True,
        },
    }
    try:
        raw_output = Runner.run_sync(
            build_decision_critic_agent(runtime_spec),
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            context=context,
            hooks=build_run_hooks(),
            max_turns=max_turns,
        ).final_output
        result = coerce_model_from_json_text(raw_output, DecisionCriticResult)
        _validate_result_against_input(result, critic_input)
    except Exception as exc:
        return _unavailable_execution(
            critic_input=critic_input,
            mode=mode,
            reason=type(exc).__name__,
            error=str(exc),
            runtime_spec=runtime_spec,
            context=context,
        )

    result, recalculation_exhausted = _apply_recalculation_policy(result, critic_input)
    debug = {
        "mode": "agent",
        "provider": runtime_spec.provider,
        "profile": runtime_spec.profile,
        "model": runtime_spec.model_name,
        "recalculation_exhausted": recalculation_exhausted,
        "tool_trace": list(context.tool_trace),
        "agent_trace": list(context.agent_trace),
        "decision_trace": list(context.decision_trace),
    }
    return _execution(
        critic_input=critic_input,
        mode=mode,
        result=result,
        llm_available=True,
        debug=debug,
    )


def _critic_domain_route(model_card: Mapping[str, Any]) -> tuple[str, QueryIntent]:
    domain = str(model_card.get("domain") or "").strip().lower()
    if domain == "pricing":
        return (
            "Проверь численное ценовое решение, не изменяя ranking.",
            QueryIntent.PRICE,
        )
    return (
        "Проверь численное рекламное решение, не изменяя ranking.",
        QueryIntent.MARKETING,
    )


def _validate_result_against_input(
    result: DecisionCriticResult,
    critic_input: DecisionCriticInput,
) -> None:
    candidate_ids = set(critic_input.candidate_ids)
    evidence_refs = set(critic_input.known_evidence_refs)
    for issue in result.issues:
        if issue.code not in _LLM_ALLOWED_ISSUE_CODES:
            raise ValueError(f"LLM cannot emit internal issue code: {issue.code}")
        unknown_candidates = set(issue.affected_candidates) - candidate_ids
        if unknown_candidates:
            raise ValueError(
                f"critic references unknown candidates: {sorted(unknown_candidates)}"
            )
        unknown_evidence = set(issue.evidence_refs) - evidence_refs
        if unknown_evidence:
            raise ValueError(
                f"critic references unknown evidence: {sorted(unknown_evidence)}"
            )


def _apply_recalculation_policy(
    result: DecisionCriticResult,
    critic_input: DecisionCriticInput,
) -> tuple[DecisionCriticResult, bool]:
    if result.verdict != "recalculate" or critic_input.recalculation_attempt == 0:
        return result, False
    return (
        result.model_copy(
            update={
                "verdict": "blocked",
                "summary": (
                    f"{result.summary} Автоматический пересчёт уже выполнялся; "
                    "повторный запрос преобразован в блокировку."
                ),
            }
        ),
        True,
    )


def _execution(
    *,
    critic_input: DecisionCriticInput,
    mode: DecisionCriticMode,
    result: DecisionCriticResult,
    llm_available: bool,
    debug: dict[str, Any],
) -> DecisionCriticExecution:
    if mode == "shadow":
        effective_action: DecisionCriticEffectiveAction = "record_only"
    elif result.verdict == "approved":
        effective_action = "continue"
    elif result.verdict == "recalculate":
        effective_action = "recalculate"
    else:
        effective_action = "block"
    return DecisionCriticExecution(
        mode=mode,
        result=result,
        effective_action=effective_action,
        llm_available=llm_available,
        blocks_decision=effective_action in {"block", "recalculate"},
        ranking_fingerprint=critic_input.ranking_fingerprint,
        debug=debug,
    )


def _unavailable_execution(
    *,
    critic_input: DecisionCriticInput,
    mode: DecisionCriticMode,
    reason: str,
    error: str,
    runtime_spec: LLMRuntimeSpec | None = None,
    context: EcomRunContext | None = None,
) -> DecisionCriticExecution:
    result = DecisionCriticResult(
        verdict="blocked",
        issues=[
            DecisionCriticIssue(
                code=_INTERNAL_UNAVAILABLE_CODE,
                message="LLM Critic недоступен или вернул недопустимый structured output.",
            )
        ],
        summary="Численный ranking сохранён без изменений; результат критика недоступен.",
    )
    debug: dict[str, Any] = {
        "mode": "unavailable",
        "reason": reason,
        "error": error,
    }
    if runtime_spec is not None:
        debug.update(
            {
                "provider": runtime_spec.provider,
                "profile": runtime_spec.profile,
                "model": runtime_spec.model_name,
            }
        )
    if context is not None:
        debug.update(
            {
                "tool_trace": list(context.tool_trace),
                "agent_trace": list(context.agent_trace),
                "decision_trace": list(context.decision_trace),
            }
        )
    return _execution(
        critic_input=critic_input,
        mode=mode,
        result=result,
        llm_available=False,
        debug=debug,
    )


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"expected mapping, got {type(value).__name__}")


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    for key in _CANDIDATE_ID_KEYS:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("candidate aggregate requires candidate_id")


def _optional_id(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _collect_evidence_refs(value: Any, refs: set[str]) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = _EVIDENCE_KEY_PATTERN.sub("", str(key).lower())
            if normalized_key in {"evidenceref", "evidencerefs"}:
                if isinstance(nested, str) and nested.strip():
                    refs.add(nested.strip())
                elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                    refs.update(str(item).strip() for item in nested if str(item).strip())
            _collect_evidence_refs(nested, refs)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _collect_evidence_refs(nested, refs)


__all__ = [
    "DecisionCriticEffectiveAction",
    "DecisionCriticExecution",
    "DecisionCriticInput",
    "DecisionCriticIssue",
    "DecisionCriticIssueCode",
    "DecisionCriticMode",
    "DecisionCriticResult",
    "DecisionCriticVerdict",
    "build_decision_critic_agent",
    "run_decision_critic",
]
