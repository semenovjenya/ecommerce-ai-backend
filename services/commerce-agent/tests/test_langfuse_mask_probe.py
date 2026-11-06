from __future__ import annotations

from dataclasses import dataclass

from scripts.check_langfuse_mask import _readback_status


CANARY = "lsv2_" + "a" * 32


@dataclass
class _Observation:
    name: str
    type: str = "SPAN"
    provided_model_name: str | None = None
    internal_model_id: str | None = None
    usage_details: dict[str, int] | None = None
    cost_details: dict[str, float] | None = None

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.__dict__


def _safe_observations() -> list[_Observation]:
    return [
        _Observation(
            name="root",
            type="GENERATION",
            provided_model_name="gpt-4.1-mini",
            usage_details={"input": 12, "output": 5},
            cost_details={"total": 0.002},
        ),
        _Observation(name="sentinel"),
    ]


def _status(observations: list[_Observation]) -> str:
    return _readback_status(
        observations,
        root_name="root",
        sentinel_name="sentinel",
        canary=CANARY,
        model_name="gpt-4.1-mini",
    )


def test_readback_requires_root_and_safe_sentinel() -> None:
    assert _status(_safe_observations()[:1]) == "pending"
    assert _status(_safe_observations()) == "verified"


def test_readback_rejects_unsafe_child_name_even_with_safe_sentinel() -> None:
    observations = _safe_observations() + [_Observation(name=CANARY)]
    assert _status(observations) == "unsafe"


def test_readback_rejects_missing_usage_or_cost() -> None:
    observations = _safe_observations()
    observations[0].cost_details = None
    assert _status(observations) == "missing-metrics"
