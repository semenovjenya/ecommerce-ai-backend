from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel
from langgraph.graph.message import add_messages

from commerce_agent.models import (
    AgentStatus,
    DataGap,
    DomainAnalysis,
    DomainName,
    SchemaVersion,
)
from commerce_agent.safety import assert_secret_free


class StateConflictError(ValueError):
    """Raised when parallel branches produce incompatible authoritative updates."""


T = TypeVar("T")


def safe_add_messages(current: Any, update: Any) -> Any:
    """Reject credentials before LangGraph writes input to a checkpoint."""

    assert_secret_free(update, path="input.messages")
    return add_messages(current, update)


def set_once_equal(current: str | None, update: str | None) -> str:
    if update is None:
        if current is None:
            raise StateConflictError("identity value is required")
        return current
    if current is None or current == "":
        return update
    if current != update:
        raise StateConflictError("immutable run identity cannot change")
    return current


def schema_version_reducer(
    current: int | None,
    update: int | None,
) -> SchemaVersion:
    # LangGraph initializes integer reducer channels with ``0`` before the
    # first real update. Treat that runtime identity as unset, never as a
    # supported durable schema version.
    candidate = update if current in (None, 0) else current
    if candidate != 1 or (update is not None and update != candidate):
        raise StateConflictError("only CommerceState schema_version=1 is supported")
    return SchemaVersion(candidate)


def merge_agent_statuses(
    current: Mapping[str, AgentStatus | Mapping[str, Any]] | None,
    update: Mapping[str, AgentStatus | Mapping[str, Any]] | None,
) -> dict[str, AgentStatus]:
    merged = {
        str(key): value if isinstance(value, AgentStatus) else AgentStatus.model_validate(value)
        for key, value in (current or {}).items()
    }
    for raw_key, raw_value in (update or {}).items():
        key = str(raw_key)
        if key not in {domain.value for domain in DomainName}:
            raise StateConflictError(f"unknown domain status key: {key}")
        value = raw_value if isinstance(raw_value, AgentStatus) else AgentStatus.model_validate(raw_value)
        if value.domain.value != key:
            raise StateConflictError(f"status owner mismatch: {key} != {value.domain.value}")
        previous = merged.get(key)
        if previous is None or value.attempt > previous.attempt:
            merged[key] = value
        elif value.attempt == previous.attempt and value != previous:
            raise StateConflictError(f"conflicting status update for {key} attempt {value.attempt}")
    return dict(sorted(merged.items()))


def merge_domain_analysis(
    current: DomainAnalysis | Mapping[str, Any] | None,
    update: DomainAnalysis | Mapping[str, Any] | None,
) -> DomainAnalysis | None:
    if update is None:
        return current if isinstance(current, DomainAnalysis) or current is None else DomainAnalysis.model_validate(current)
    candidate = update if isinstance(update, DomainAnalysis) else DomainAnalysis.model_validate(update)
    if current is None:
        return candidate
    previous = current if isinstance(current, DomainAnalysis) else DomainAnalysis.model_validate(current)
    if candidate.domain != previous.domain:
        raise StateConflictError("domain analysis owner cannot change")
    if candidate.revision > previous.revision:
        return candidate
    if candidate.revision == previous.revision and candidate == previous:
        return previous
    raise StateConflictError(
        f"non-monotonic or conflicting {candidate.domain.value} analysis revision"
    )


def make_domain_analysis_reducer(
    expected_domain: DomainName,
) -> Callable[
    [DomainAnalysis | Mapping[str, Any] | None, DomainAnalysis | Mapping[str, Any] | None],
    DomainAnalysis | None,
]:
    def reducer(
        current: DomainAnalysis | Mapping[str, Any] | None,
        update: DomainAnalysis | Mapping[str, Any] | None,
    ) -> DomainAnalysis | None:
        for raw_value in (current, update):
            if raw_value is None:
                continue
            value = (
                raw_value
                if isinstance(raw_value, DomainAnalysis)
                else DomainAnalysis.model_validate(raw_value)
            )
            if value.domain != expected_domain:
                raise StateConflictError(
                    f"{expected_domain.value}_analysis cannot contain "
                    f"{value.domain.value} result"
                )
        return merge_domain_analysis(current, update)

    reducer.__name__ = f"merge_{expected_domain.value}_analysis"
    return reducer


merge_sales_analysis = make_domain_analysis_reducer(DomainName.SALES)
merge_marketing_analysis = make_domain_analysis_reducer(DomainName.MARKETING)
merge_supply_analysis = make_domain_analysis_reducer(DomainName.SUPPLY)
merge_price_analysis = make_domain_analysis_reducer(DomainName.PRICE)
merge_finance_analysis = make_domain_analysis_reducer(DomainName.FINANCE)


def _stable_id(item: Any) -> str:
    for field_name in (
        "evidence_ref",
        "gap_id",
        "incident_id",
        "action_id",
        "receipt_id",
        "error_id",
    ):
        if isinstance(item, Mapping) and item.get(field_name):
            return str(item[field_name])
        value = getattr(item, field_name, None)
        if value:
            return str(value)
    raise StateConflictError("collection item has no stable identifier")


def _canonical(item: Any) -> Any:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    return item


def merge_immutable_items(
    current: Sequence[T] | None,
    update: Sequence[T] | None,
) -> list[T]:
    merged: dict[str, T] = {_stable_id(item): item for item in (current or ())}
    for item in update or ():
        stable_id = _stable_id(item)
        previous = merged.get(stable_id)
        if previous is not None and _canonical(previous) != _canonical(item):
            raise StateConflictError(f"immutable collection item changed: {stable_id}")
        merged[stable_id] = previous if previous is not None else item
    return [merged[key] for key in sorted(merged)]


def merge_gaps(
    current: Sequence[DataGap | Mapping[str, Any]] | None,
    update: Sequence[DataGap | Mapping[str, Any]] | None,
) -> list[DataGap]:
    merged = {
        _stable_id(item): item if isinstance(item, DataGap) else DataGap.model_validate(item)
        for item in (current or ())
    }
    for raw_item in update or ():
        item = raw_item if isinstance(raw_item, DataGap) else DataGap.model_validate(raw_item)
        previous = merged.get(item.gap_id)
        if previous is None or item.revision > previous.revision:
            merged[item.gap_id] = item
        elif item.revision == previous.revision and item != previous:
            raise StateConflictError(f"conflicting gap revision: {item.gap_id}")
    return [merged[key] for key in sorted(merged)]


def make_optional_replace_validator(
    validator: Callable[[Any], T],
) -> Callable[[T | Any | None, T | Any | None], T | None]:
    def reducer(current: T | Any | None, update: T | Any | None) -> T | None:
        if update is None:
            return current if current is None else validator(current)
        candidate = validator(update)
        if current is None:
            return candidate
        previous = validator(current)
        if previous != candidate:
            raise StateConflictError("single-owner state field received conflicting updates")
        return previous

    return reducer


__all__ = [
    "StateConflictError",
    "make_optional_replace_validator",
    "make_domain_analysis_reducer",
    "merge_agent_statuses",
    "merge_domain_analysis",
    "merge_finance_analysis",
    "merge_gaps",
    "merge_immutable_items",
    "merge_marketing_analysis",
    "merge_price_analysis",
    "merge_sales_analysis",
    "merge_supply_analysis",
    "schema_version_reducer",
    "safe_add_messages",
    "set_once_equal",
]
